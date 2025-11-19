from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json, time, os, uuid

@dataclass
class TraceEvent:
    ts_ms: float
    phase: str                 # e.g., "retrieve", "summarize", "critique", "retry"
    data: Dict[str, Any]

@dataclass
class TraceRun:
    run_id: str
    query: str
    started_ms: float
    events: List[TraceEvent]
    finished_ms: Optional[float] = None
    reward: Optional[float] = None

    def record(self, phase: str, **data: Any) -> None:
        self.events.append(TraceEvent(ts_ms=time.time()*1000, phase=phase, data=data))

    def finish(self, reward: Optional[float] = None) -> None:
        self.finished_ms = time.time()*1000
        self.reward = reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms,
            "reward": self.reward,
            "events": [asdict(e) for e in self.events],
        }

class TraceLogger:
    def __init__(self, out_path: str = "data/traces.jsonl") -> None:
        self.out_path = out_path
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def start(self, query: str) -> TraceRun:
        return TraceRun(run_id=str(uuid.uuid4()), query=query, started_ms=time.time()*1000, events=[])

    def save(self, run: TraceRun) -> None:
        with open(self.out_path, "a") as f:
            f.write(json.dumps(run.to_dict()) + "\n")
