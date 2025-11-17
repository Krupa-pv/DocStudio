# doc_studio/core/run_logger.py
# central run logging; env toggles

from __future__ import annotations
import os, json, time
from typing import Dict, Any, Optional

# dotenv load (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# local loggers
from doc_studio.tracking.sqlite_logger import SqliteLogger, SqliteRun
try:
    from doc_studio.tracking.mlflow_logger import MLflowLogger
except Exception:
    MLflowLogger = None  # type: ignore


def _bool_env(name: str, default: str = "0") -> bool:
    # env -> bool
    return os.getenv(name, default) == "1"


def log_run(
    *,
    query: str,
    params: Dict[str, Any],
    scores: Dict[str, float],
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    latency_ms: int = 0,
) -> None:
    # read toggles
    use_sqlite = _bool_env("DOCSTUDIO_SQLITE", "1")
    use_mlflow = _bool_env("DOCSTUDIO_MLFLOW", "0")

    # sqlite path
    if use_sqlite:
        try:
            SqliteLogger(os.getenv("DOCSTUDIO_DB", "data/docstudio.db")).log(
                SqliteRun(
                    query=query,
                    params=params,
                    scores=scores,
                    tokens_prompt=int(tokens_prompt or 0),
                    tokens_completion=int(tokens_completion or 0),
                    latency_ms=int(latency_ms or 0),
                )
            )
        except Exception as e:
            # non-fatal
            print("sqlite_log_err:", e)

    # mlflow path
    if use_mlflow and MLflowLogger is not None:
        try:
            with MLflowLogger() as mlf:
                # flatten-ish params
                mlf.log_params({**params, "query_len": len(query)})
                mlf.log_metrics(
                    {
                        "overall": float(scores.get("overall", 0.0)),
                        "faithfulness": float(scores.get("faithfulness", 0.0)),
                        "coverage": float(scores.get("coverage", 0.0)),
                        "clarity": float(scores.get("clarity", 0.0)),
                        "latency_ms": float(latency_ms or 0),
                        "tokens_prompt": float(tokens_prompt or 0),
                        "tokens_completion": float(tokens_completion or 0),
                    }
                )
        except Exception as e:
            # non-fatal
            print("mlflow_log_err:", e)
