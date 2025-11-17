from __future__ import annotations
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass(frozen=True)
class Arm:
    name: str
    params: Dict[str, Any]  # e.g., {"k": 5, "temperature": 0.0, "use_rewrite_on_retry": True}


class EpsilonGreedyPolicy:
    """
    Minimal ε-greedy over discrete 'arms'. Uses the reward log to compute average reward per arm name.
    Assumes the orchestrator writes rows that include 'reward' and 'arm' name.
    """

    def __init__(self, arms: List[Arm], epsilon: float = 0.2, reward_log_path: str = "data/rewards.csv") -> None:
        assert 0.0 <= epsilon <= 1.0
        self.arms = arms
        self.epsilon = epsilon
        self.reward_log_path = reward_log_path

    def _load_stats(self) -> Dict[str, Tuple[float, int]]:
        """
        Returns {arm_name: (avg_reward, count)}
        If file missing or no rows for an arm, that arm has (0.0, 0).
        """
        stats: Dict[str, Tuple[float, int]] = {a.name: (0.0, 0) for a in self.arms}
        if not os.path.exists(self.reward_log_path):
            return stats

        try:
            with open(self.reward_log_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    arm = row.get("arm")
                    reward_str = row.get("reward")
                    if arm and reward_str is not None and arm in stats:
                        r = float(reward_str)
                        avg, n = stats[arm]
                        # incremental average
                        new_avg = (avg * n + r) / (n + 1)
                        stats[arm] = (new_avg, n + 1)
        except Exception:
            # best effort; if parsing fails, treat as cold start
            pass
        return stats

    def choose(self) -> Arm:
        stats = self._load_stats()

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        # Exploitation: pick arm with highest avg reward; break ties by counts, then random
        ranked = sorted(
            self.arms,
            key=lambda a: (stats[a.name][0], stats[a.name][1]),
            reverse=True,
        )
        return ranked[0]
