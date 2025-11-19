from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import csv
import math
import os
import random
import time


@dataclass
class Arm:
    """
    Represents a single configuration (arm) in the bandit problem
    Each arm corresponds to a specific set of retrieval hyperparameters
    """
    name: str
    params: Dict[str, Any]


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy multi-armed bandit for hyperparameter selection

    Balances exploration of untested configs vs exploitation of best known config
    Maintains running average of rewards per arm using incremental mean updates
    Supports optimistic initialization and decaying epsilon for better convergence
    """

    def __init__(
        self,
        arms: Optional[List[Arm]] = None,
        epsilon: float = 0.1,
        reward_log_path: Optional[str] = None,
        optimistic_init: float = 0.0,
        decay_epsilon: bool = False,
    ) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self._arms: List[Arm] = arms or []
        self._epsilon: float = float(epsilon)
        self._initial_epsilon: float = float(epsilon)
        self._decay_epsilon = decay_epsilon
        self._total_pulls = 0

        # Track per-arm statistics with optimistic initialization
        self._counts: Dict[str, int] = {a.name: 0 for a in self._arms}
        self._values: Dict[str, float] = {a.name: optimistic_init for a in self._arms}
        self._reward_log_path = reward_log_path

        if self._reward_log_path:
            self._ensure_log_header()

    def add_arm(self, arm: Arm) -> None:
        """Dynamically add a new arm to the policy at runtime"""
        if any(a.name == arm.name for a in self._arms):
            raise ValueError(f"Arm '{arm.name}' already exists")
        self._arms.append(arm)
        self._counts[arm.name] = 0
        self._values[arm.name] = 0.0

    @property
    def arms(self) -> List[Arm]:
        return list(self._arms)

    @property
    def epsilon(self) -> float:
        """Current epsilon value, potentially decayed"""
        if self._decay_epsilon and self._total_pulls > 0:
            # Decay epsilon as 1/sqrt(t) for theoretical regret bounds
            return self._initial_epsilon / math.sqrt(1 + self._total_pulls)
        return self._epsilon

    def choose(self) -> Arm:
        """
        Select an arm using epsilon-greedy strategy

        Prioritizes untried arms (ensures each arm sampled at least once)
        Then epsilon-greedy: explore random arm with probability epsilon,
        otherwise exploit best arm based on empirical mean reward
        """
        if not self._arms:
            raise RuntimeError("EpsilonGreedyPolicy.choose() called but no arms configured")

        # Always try untried arms first to avoid cold start issues
        untried = [a for a in self._arms if self._counts.get(a.name, 0) == 0]
        if untried:
            return random.choice(untried)

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: choose random arm
            return random.choice(self._arms)
        else:
            # Exploit: choose arm with highest empirical mean
            best = max(self._arms, key=lambda a: self._values.get(a.name, 0.0))
            return best

    def update(self, arm_name: str, reward: float) -> None:
        """
        Update arm statistics after observing reward

        Uses incremental mean formula for numerical stability:
        new_mean = old_mean + (reward - old_mean) / n
        """
        if arm_name not in self._counts:
            raise KeyError(f"Unknown arm '{arm_name}'")

        n = self._counts[arm_name] + 1
        self._counts[arm_name] = n
        self._total_pulls += 1

        # Incremental mean update for numerical stability
        old_mean = self._values[arm_name]
        new_mean = old_mean + (reward - old_mean) / float(n)
        self._values[arm_name] = new_mean

        if self._reward_log_path:
            self._log_reward(arm_name, reward, n, new_mean)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Return current statistics for all arms
        Useful for debugging and analysis
        """
        stats = {}
        for arm in self._arms:
            stats[arm.name] = {
                "mean_reward": self._values[arm.name],
                "count": float(self._counts[arm.name]),
                "empirical_std": 0.0,  # Could track variance for UCB
            }
        return stats

    def _ensure_log_header(self) -> None:
        """Initialize CSV log file with header if not exists"""
        path = self._reward_log_path
        if not path:
            return

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts_utc", "arm", "reward", "count", "mean_reward", "epsilon"])

    def _log_reward(self, arm_name: str, reward: float, count: int, mean_reward: float) -> None:
        """Append reward observation to CSV log"""
        path = self._reward_log_path
        if not path:
            return

        ts_utc = time.time()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ts_utc, arm_name, reward, count, mean_reward, self.epsilon])


class UCBPolicy:
    """
    Upper Confidence Bound (UCB1) policy for multi-armed bandits

    Provides theoretical guarantees on regret: O(sqrt(K * T * log(T)))
    Automatically balances exploration and exploitation using confidence bounds
    More principled than epsilon-greedy for some applications
    """

    def __init__(
        self,
        arms: Optional[List[Arm]] = None,
        c: float = 2.0,
        reward_log_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            c: Exploration constant (higher = more exploration)
               Typical values: 1.0 to 2.0 (sqrt(2) is theoretically optimal)
        """
        self._arms: List[Arm] = arms or []
        self._c = c
        self._total_pulls = 0
        self._counts: Dict[str, int] = {a.name: 0 for a in self._arms}
        self._values: Dict[str, float] = {a.name: 0.0 for a in self._arms}
        self._reward_log_path = reward_log_path

        if self._reward_log_path:
            self._ensure_log_header()

    def add_arm(self, arm: Arm) -> None:
        if any(a.name == arm.name for a in self._arms):
            raise ValueError(f"Arm '{arm.name}' already exists")
        self._arms.append(arm)
        self._counts[arm.name] = 0
        self._values[arm.name] = 0.0

    @property
    def arms(self) -> List[Arm]:
        return list(self._arms)

    def _ucb_score(self, arm_name: str) -> float:
        """
        Compute UCB1 score for an arm

        UCB = empirical_mean + c * sqrt(log(T) / n)
        where T is total pulls, n is arm pulls
        """
        if self._counts[arm_name] == 0:
            return float('inf')  # Untried arms get infinite score

        mean = self._values[arm_name]
        n = self._counts[arm_name]

        # Confidence bound grows with uncertainty (fewer samples)
        # and shrinks as we pull the arm more
        exploration_bonus = self._c * math.sqrt(math.log(self._total_pulls + 1) / n)

        return mean + exploration_bonus

    def choose(self) -> Arm:
        """Select arm with highest UCB score"""
        if not self._arms:
            raise RuntimeError("UCBPolicy.choose() called but no arms configured")

        # UCB automatically handles exploration via confidence bounds
        best = max(self._arms, key=lambda a: self._ucb_score(a.name))
        return best

    def update(self, arm_name: str, reward: float) -> None:
        if arm_name not in self._counts:
            raise KeyError(f"Unknown arm '{arm_name}'")

        n = self._counts[arm_name] + 1
        self._counts[arm_name] = n
        self._total_pulls += 1

        old_mean = self._values[arm_name]
        new_mean = old_mean + (reward - old_mean) / float(n)
        self._values[arm_name] = new_mean

        if self._reward_log_path:
            self._log_reward(arm_name, reward, n, new_mean)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for arm in self._arms:
            stats[arm.name] = {
                "mean_reward": self._values[arm.name],
                "count": float(self._counts[arm.name]),
                "ucb_score": self._ucb_score(arm.name),
            }
        return stats

    def _ensure_log_header(self) -> None:
        path = self._reward_log_path
        if not path:
            return

        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts_utc", "arm", "reward", "count", "mean_reward", "ucb_score"])

    def _log_reward(self, arm_name: str, reward: float, count: int, mean_reward: float) -> None:
        path = self._reward_log_path
        if not path:
            return

        ts_utc = time.time()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ts_utc, arm_name, reward, count, mean_reward, self._ucb_score(arm_name)])
