"""Train and evaluate CommentPulse sentiment ML model."""
from pathlib import Path
import csv, json, collections
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from model import normalize_text, MODEL_PATH
from prepare_dataset import main as prepare

BASE = Path(__file__).resolve().parent
CLEAN = BASE / "dataset" / "rusentitweet_cleaned.csv"

def main():
    if not CLEAN.exists():
        prepare()
    with CLEAN.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [normalize_text(x["text"]) for x in rows if x.get("text","").strip()]
    labels = [x["label"] for x in rows if x.get("text","").strip()]
    if len(set(labels)) < 3 or len(texts) < 100:
        raise SystemExit("Недостаточно корректных данных для обучения. Проверьте dataset/rusentitweet_cleaned.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000,
                                  sublinear_tf=True, strip_accents=None)),
        ("classifier", LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels_order = ["negative","neutral","positive"]
    report = classification_report(y_test, pred, labels=labels_order,
                                   target_names=labels_order, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "RuSentiTweet",
        "total_rows": len(texts), "train_rows": len(X_train), "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        "classification_report": report,
        "confusion_matrix_labels": labels_order,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels_order).tolist(),
        "class_distribution": dict(collections.Counter(labels)),
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "random_state": 42
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (BASE/"reports"/"model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Обучение завершено.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Модель: {MODEL_PATH}")
    print(f"Отчёт: {BASE/'reports'/'model_metrics.json'}")

if __name__ == "__main__":
    main()
