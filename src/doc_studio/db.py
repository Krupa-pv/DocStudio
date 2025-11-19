from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any


DB_PATH = Path(os.getenv("DOCSTUDIO_DB", "docstudio.db"))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_documents_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            title TEXT,
            path TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _ensure_chunks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            chunk_id TEXT,
            text TEXT,
            position INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _ensure_queries_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            answer TEXT,
            overall REAL,
            faithfulness REAL,
            coverage REAL,
            clarity REAL,
            reward REAL,
            arm_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def upsert_document(
    doc_id: str,
    title: Optional[str] = None,
    path: Optional[str] = None,
    tags: Optional[str] = None,
) -> None:
    conn = get_connection()
    try:
        _ensure_documents_table(conn)
        conn.execute(
            """
            INSERT INTO documents (doc_id, title, path, tags)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title = excluded.title,
                path = excluded.path,
                tags = excluded.tags
            """,
            (doc_id, title, path, tags),
        )
        conn.commit()
    finally:
        conn.close()


def insert_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    position: int,
) -> None:
    conn = get_connection()
    try:
        _ensure_chunks_table(conn)
        conn.execute(
            """
            INSERT INTO chunks (doc_id, chunk_id, text, position)
            VALUES (?, ?, ?, ?)
            """,
            (doc_id, chunk_id, text, position),
        )
        conn.commit()
    finally:
        conn.close()


def insert_query_log(
    query: str,
    answer: str,
    scores: Dict[str, Any],
    arm_name: Optional[str] = None,
) -> None:
    """
    Insert a single query + answer + scores row into the `queries` table.

    scores should contain:
      overall, faithfulness, coverage, clarity, reward
    (missing keys are treated as 0.0)
    """
    conn = get_connection()
    try:
        _ensure_queries_table(conn)
        conn.execute(
            """
            INSERT INTO queries (
                query,
                answer,
                overall,
                faithfulness,
                coverage,
                clarity,
                reward,
                arm_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                answer,
                float(scores.get("overall", 0.0)),
                float(scores.get("faithfulness", 0.0)),
                float(scores.get("coverage", 0.0)),
                float(scores.get("clarity", 0.0)),
                float(scores.get("reward", 0.0)),
                arm_name,
            ),
        )
        conn.commit()
    finally:
        conn.close()
