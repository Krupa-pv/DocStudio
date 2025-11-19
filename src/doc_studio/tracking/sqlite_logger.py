#  sqlite logger for runs

from __future__ import annotations
import sqlite3, json, time, os
from dataclasses import dataclass
from typing import Dict, Any

DEFAULT_DB = os.getenv("DOCSTUDIO_DB", "data/docstudio.db")

# schema for table creation
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  query TEXT,
  params_json TEXT,
  reward_overall REAL,
  faithfulness REAL,
  coverage REAL,
  clarity REAL,
  tokens_prompt INTEGER,
  tokens_completion INTEGER,
  latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts_utc);
"""

@dataclass
class SqliteRun:
    # container for run info
    query: str
    params: Dict[str, Any]
    scores: Dict[str, float]
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: int = 0

class SqliteLogger:
    # small wrapper over sqlite
    def __init__(self, db_path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        # make table if not exists
        with sqlite3.connect(self.db_path) as cx:
            for stmt in SCHEMA.strip().split(";"):
                s = stmt.strip()
                if s: cx.execute(s)

    def log(self, run: SqliteRun) -> int:
        # add new record
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with sqlite3.connect(self.db_path) as cx:
            cur = cx.cursor()
            cur.execute("""
              INSERT INTO runs
              (ts_utc, query, params_json, reward_overall, faithfulness, coverage, clarity,
               tokens_prompt, tokens_completion, latency_ms)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts,
                run.query,
                json.dumps(run.params, ensure_ascii=False),
                float(run.scores.get("overall", 0.0)),
                float(run.scores.get("faithfulness", 0.0)),
                float(run.scores.get("coverage", 0.0)),
                float(run.scores.get("clarity", 0.0)),
                int(run.tokens_prompt or 0),
                int(run.tokens_completion or 0),
                int(run.latency_ms or 0),
            ))
            cx.commit()
            return int(cur.lastrowid)
