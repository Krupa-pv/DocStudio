
import os
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
os.environ.setdefault("MLFLOW_EXPERIMENT", "docstudio")

from doc_studio.tracking.mlflow_logger import MLflowLogger  # after env

import tempfile

def main():
    with MLflowLogger() as mlf:
        mlf.log_params({"llm": "gpt-4o", "k": 5, "mmr_lambda": 0.3})
        mlf.log_metrics({"overall": 0.97, "faithfulness": 0.95, "latency_ms": 1234})
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.write(b"artifact ok"); tmp.close()
        mlf.log_artifact_if_exists(tmp.name)
        print("ok -> mlruns / docstudio")

if __name__ == "__main__":
    main()
