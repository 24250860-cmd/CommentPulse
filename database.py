from pathlib import Path
import sqlite3, json
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "instance" / "commentpulse.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            total INTEGER NOT NULL,
            positive INTEGER NOT NULL,
            negative INTEGER NOT NULL,
            neutral INTEGER NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            original_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            topic TEXT NOT NULL,
            model_name TEXT NOT NULL,
            FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        )""")
        con.execute("PRAGMA foreign_keys=ON")

def save_analysis(items, summary):
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    with connect() as con:
        cur = con.execute("""INSERT INTO analyses(created_at,total,positive,negative,neutral)
            VALUES(?,?,?,?,?)""", (created, summary["total"], summary["positive"],
            summary["negative"], summary["neutral"]))
        analysis_id = cur.lastrowid
        con.executemany("""INSERT INTO comments
            (analysis_id,original_text,cleaned_text,sentiment,confidence,topic,model_name)
            VALUES(?,?,?,?,?,?,?)""", [
            (analysis_id, x["text"], x["cleaned"], x["sentiment"], x["confidence"],
             x["topic"], x["model"]) for x in items
        ])
        return analysis_id

def get_history(limit=50):
    with connect() as con:
        rows = con.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

def get_trend(days="all"):
    with connect() as con:
        if str(days).isdigit():
            rows = con.execute("""SELECT date(created_at) day, SUM(total) total,
                SUM(positive) positive, SUM(negative) negative, SUM(neutral) neutral
                FROM analyses WHERE created_at >= datetime('now', ?) GROUP BY date(created_at)
                ORDER BY day""", (f"-{int(days)} days",)).fetchall()
        else:
            rows = con.execute("""SELECT date(created_at) day, SUM(total) total,
                SUM(positive) positive, SUM(negative) negative, SUM(neutral) neutral
                FROM analyses GROUP BY date(created_at) ORDER BY day""").fetchall()
        return [dict(r) for r in rows]

def clear_history():
    with connect() as con:
        con.execute("DELETE FROM comments")
        con.execute("DELETE FROM analyses")
