from flask import Flask, render_template, request, jsonify, make_response
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from html import escape
import csv, io, json, sqlite3, re
from database import init_db, save_analysis, get_history, clear_history, get_trend
from model import load_model, predict_sentiment, model_status

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
init_db()
load_model()

def clean_text(text):
    text = str(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\sёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

TOPICS = {
    "Качество контента": ["видео", "ролик", "звук", "съёмк", "съемк", "контент", "монтаж", "quality"],
    "Автор и подача": ["автор", "блогер", "голос", "подача", "ведущий"],
    "Цена и покупка": ["цена", "дорого", "купил", "заказ", "товар", "доставка", "баға", "қымбат"],
    "Вопросы и просьбы": ["где", "когда", "можно", "подскаж", "снимешь", "ссылка", "қайда", "қалай"],
}
STOPWORDS = set("это как что где когда почему чтобы очень такой такая мне тебя твой ваш для или но если уже еще всё все они она оно мы вы он их его ее был была быть есть нет да не же бы то а и в на по с к у о от до за из под при".split())

def analyze_one(text):
    raw = str(text or "").strip()
    prediction = predict_sentiment(raw)
    normalized = clean_text(raw).lower()
    topic_scores = {name: sum(1 for key in keys if key in normalized) for name, keys in TOPICS.items()}
    topic = max(topic_scores, key=topic_scores.get) if topic_scores else "Общее мнение"
    if not topic_scores or not topic_scores[topic]:
        topic = "Общее мнение"
    return {
        "text": raw, "cleaned": clean_text(raw),
        "sentiment": prediction["sentiment"],
        "confidence": round(float(prediction["confidence"]), 4),
        "topic": topic,
        "model": prediction["model"]
    }

def summarize(items):
    counts = Counter(x["sentiment"] for x in items)
    topics = Counter(x["topic"] for x in items)
    words = Counter(w.lower() for item in items for w in clean_text(item["text"]).lower().split()
                    if len(w) > 2 and w.lower() not in STOPWORDS)
    return {
        "total": len(items), "positive": counts["positive"],
        "negative": counts["negative"], "neutral": counts["neutral"],
        "topics": topics.most_common(8), "words": words.most_common(10)
    }

DEMO = [
    {"text":"Очень классное видео! Спасибо за полезный контент ❤️"},
    {"text":"Звук ужасный, ничего не слышно. Исправьте качество! 😡"},
    {"text":"Снимешь продолжение на следующей неделе? 😊"},
    {"text":"Обожаю твои ролики, ты лучший! Всегда поднимаешь настроение 🔥"},
    {"text":"Очень дорого и совсем не стоит своих денег. Разочарован. 👎"},
    {"text":"Подскажите, где найти ссылку на полный обзор?"},
    {"text":"Керемет видео, рахмет! 🔥"},
    {"text":"Бұл өте жаман, сапасыз жұмыс 😡"}
]

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/tools")
def tools_page():
    return render_template("tools.html")

@app.get("/api/status")
def api_status():
    return jsonify(model_status())

@app.post("/api/analyze")
def api_analyze():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("comments", [])
    if not isinstance(texts, list):
        return jsonify({"error": "comments должен быть списком"}), 400
    if not model_status()["ready"]:
        return jsonify({"error": "Модель ещё не обучена. Выполните: python train_model.py"}), 503
    results = [analyze_one(x.get("text", "") if isinstance(x, dict) else x)
               for x in texts[:1000] if str(x.get("text", "") if isinstance(x, dict) else x).strip()]
    summary = summarize(results)
    record_id = save_analysis(results, summary)
    return jsonify({"results": results, "summary": summary,
                    "history": get_history(20), "analysis_id": record_id,
                    "engine": "TF-IDF + Logistic Regression / SQLite"})

@app.get("/api/history")
def api_history():
    return jsonify(get_history(50))

@app.delete("/api/history")
def api_clear_history():
    clear_history()
    return jsonify({"ok": True})

@app.get("/api/trend")
def api_trend():
    days = request.args.get("days", "all")
    return jsonify(get_trend(days))

@app.post("/api/export.csv")
def export_csv():
    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["Комментарий", "Очищенный текст", "Тональность", "Уверенность", "Тема", "Модель"])
    for item in items:
        writer.writerow([item.get("text",""), item.get("cleaned",""), item.get("sentiment",""),
                         item.get("confidence",""), item.get("topic",""), item.get("model","")])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=commentpulse_analysis.csv"
    return response

@app.post("/api/export.pdf")
def export_pdf():
    payload = request.get_json(silent=True) or {}
    items = payload.get("items", [])
    summary = summarize(items)
    rows = "".join(
        f"<tr><td>{escape(str(x.get('text','')))}</td><td>{escape(str(x.get('sentiment','')))}</td>"
        f"<td>{float(x.get('confidence',0)):.1%}</td><td>{escape(str(x.get('topic','')))}</td></tr>"
        for x in items
    )
    return f"""<!doctype html><html lang='ru'><meta charset='utf-8'><title>CommentPulse report</title>
    <style>body{{font:14px Arial;margin:30px;color:#182033}}h1{{color:#5b4acb}}table{{width:100%;border-collapse:collapse}}td,th{{border:1px solid #ccd2df;padding:8px;text-align:left;vertical-align:top}}@media print{{button{{display:none}}}}</style>
    <h1>CommentPulse — отчёт анализа</h1>
    <p>Всего: {summary['total']} · Положительные: {summary['positive']} · Негативные: {summary['negative']} · Нейтральные: {summary['neutral']}</p>
    <p>Метод: TF-IDF + Logistic Regression</p><button onclick='window.print()'>Печать / Сохранить как PDF</button>
    <table><tr><th>Комментарий</th><th>Тональность</th><th>Уверенность</th><th>Тема</th></tr>{rows}</table></html>"""

@app.get("/api/demo")
def demo():
    if not model_status()["ready"]:
        return jsonify({"error": "Сначала обучите модель: python train_model.py"}), 503
    items = [analyze_one(x["text"]) for x in DEMO]
    return jsonify({"results": items, "summary": summarize(items), "engine": "trained ML model"})

if __name__ == "__main__":
    app.run(debug=True)
