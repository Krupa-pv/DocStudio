
# optional mlflow logger; pretty ui, screenshots

from __future__ import annotations
import os, json
from typing import Dict, Any, Optional

# lazy import; avoid cold start cost
def _lazy_mlflow():
    import mlflow  # type: ignore
    return mlflow

class MLflowLogger:
    # context manager; handles start/end
    def __init__(self, tracking_uri: Optional[str] = None, experiment: Optional[str] = None):
        self._mlf = _lazy_mlflow()
        self._uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self._exp = experiment or os.getenv("MLFLOW_EXPERIMENT", "docstudio")
        self._run = None
        # setup
        self._mlf.set_tracking_uri(self._uri)
        self._mlf.set_experiment(self._exp)

    def __enter__(self):
        self._run = self._mlf.start_run()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mlf.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        # flatten nested
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in params.items()}
        self._mlf.log_params(flat)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        # cast to float
        payload = {k: float(v) for k, v in metrics.items()}
        self._mlf.log_metrics(payload)

    def log_artifact_if_exists(self, path: str) -> None:
        # file check
        if os.path.exists(path):
            self._mlf.log_artifact(path)
