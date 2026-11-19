from pathlib import Path
import joblib
import re

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "sentiment_model.joblib"
_PIPELINE = None
LABELS = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model():
    global _PIPELINE
    if MODEL_PATH.exists():
        try:
            _PIPELINE = joblib.load(MODEL_PATH)
        except Exception:
            _PIPELINE = None
    return _PIPELINE

def model_status():
    return {"ready": _PIPELINE is not None, "model": "TF-IDF + Logistic Regression",
            "model_file": str(MODEL_PATH), "database": "SQLite"}

def predict_sentiment(text):
    if _PIPELINE is None:
        load_model()
    if _PIPELINE is None:
        raise RuntimeError("Модель не обучена. Выполните python train_model.py")
    cleaned = normalize_text(text)
    probs = _PIPELINE.predict_proba([cleaned])[0]
    classes = list(_PIPELINE.classes_)
    idx = int(probs.argmax())
    label = str(classes[idx]).lower()
    return {"sentiment": LABELS.get(label, label), "confidence": float(probs[idx]),
            "model": "TF-IDF + Logistic Regression"}
