"""Download and clean RuSentiTweet for CommentPulse."""
from pathlib import Path
import csv, re, urllib.request, json, collections

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset"
DATA.mkdir(exist_ok=True)
SOURCE_URL = "https://raw.githubusercontent.com/sismetanin/rusentitweet/main/rusentitweet_full.csv"
RAW = DATA / "rusentitweet_full.csv"
CLEAN = DATA / "rusentitweet_cleaned.csv"
REPORT = BASE / "reports" / "dataset_analysis.json"

def pick(row, names):
    for key in names:
        if key in row and row[key] is not None:
            return row[key]
    return None

def normalize(text):
    text = str(text or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.I)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text

def main():
    if not RAW.exists():
        print("Downloading RuSentiTweet...")
        try:
            urllib.request.urlretrieve(SOURCE_URL, RAW)
        except Exception as exc:
            raise SystemExit(
                "Не удалось скачать датасет автоматически. Скачайте rusentitweet_full.csv "
                "из https://github.com/sismetanin/rusentitweet и положите в папку dataset/. "
                f"Причина: {exc}"
            )
    with RAW.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)
        source_columns = reader.fieldnames or []
        rows = list(reader)

    kept, removed_empty, removed_dupes, removed_labels = [], 0, 0, 0
    seen = set()
    counts_before = collections.Counter()
    # The source dataset labels are positive, neutral, negative, speech, skip.
    mapping = {"positive":"positive", "negative":"negative", "neutral":"neutral",
               "положительный":"positive", "отрицательный":"negative", "нейтральный":"neutral"}
    for row in rows:
        text = pick(row, ["text", "tweet", "content", "message", "sentence", "Текст"])
        label = pick(row, ["label", "sentiment", "class", "polarity", "тональность"])
        if text is None or label is None:
            # Try a positional fallback for CSV variants.
            vals = list(row.values())
            text = text if text is not None else (vals[0] if vals else "")
            label = label if label is not None else (vals[1] if len(vals)>1 else "")
        label = str(label).strip().lower().replace(" ", "_")
        counts_before[label] += 1
        text = normalize(text)
        if not text:
            removed_empty += 1; continue
        if label not in mapping:
            removed_labels += 1; continue
        key = (text.casefold(), mapping[label])
        if key in seen:
            removed_dupes += 1; continue
        seen.add(key)
        kept.append({"text": text, "label": mapping[label]})

    with CLEAN.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text","label"])
        writer.writeheader(); writer.writerows(kept)

    counts_after = collections.Counter(x["label"] for x in kept)
    report = {
        "dataset": "RuSentiTweet",
        "source": SOURCE_URL,
        "source_columns": source_columns,
        "raw_rows": len(rows),
        "rows_after_cleaning": len(kept),
        "removed_empty": removed_empty,
        "removed_duplicates": removed_dupes,
        "removed_non_sentiment_classes": removed_labels,
        "original_class_distribution": dict(counts_before),
        "training_class_distribution": dict(counts_after),
        "cleaning": ["remove empty text", "normalize whitespace", "replace URLs and usernames with tokens",
                     "remove duplicate text-label pairs", "keep positive/negative/neutral; exclude speech and skip"],
        "note": "The dataset is Russian social-media posts, not specifically Instagram/TikTok comments."
    }
    (BASE/"reports").mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Cleaned dataset saved: {CLEAN}")

if __name__ == "__main__":
    main()
