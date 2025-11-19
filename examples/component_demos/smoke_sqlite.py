# check for sqlite logger

from doc_studio.tracking.sqlite_logger import SqliteLogger, SqliteRun

def main():
    log = SqliteLogger()
    run = SqliteRun(
        query="summarize article about AI safety",
        params={"k": 5, "mmr_lambda": 0.3},
        scores={"overall": 0.95, "faithfulness": 0.9, "coverage": 0.92, "clarity": 0.88},
        tokens_prompt=210,
        tokens_completion=95,
        latency_ms=1340,
    )
    row_id = log.log(run)
    print("inserted row id:", row_id)

    # verify data stored
    import sqlite3
    cx = sqlite3.connect("data/docstudio.db")
    cur = cx.cursor()
    cur.execute("SELECT id, ts_utc, query, reward_overall FROM runs ORDER BY id DESC LIMIT 1")
    print("latest:", cur.fetchone())
    cx.close()

if __name__ == "__main__":
    main()
