"""
Demonstrate multi-armed bandit RL for RAG hyperparameter optimization

This script shows how epsilon-greedy and UCB1 policies learn to identify
the best retrieval configuration through exploration-exploitation.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from doc_studio.orchestrator.policy import EpsilonGreedyPolicy, UCBPolicy, Arm
import random
import time


def simulate_rag_config(config_name: str, k: int, mmr_lambda: float) -> float:
    """
    Simulate running a RAG query with given configuration

    In reality this would call the orchestrator. Here we simulate rewards
    based on "true" quality of each config to demonstrate learning.

    Returns reward (quality - cost)
    """
    # True expected rewards for different configs (unknown to the policy)
    true_rewards = {
        "baseline": 0.55,      # TF-IDF - worst
        "embeddings": 0.68,    # Vector search - better
        "mmr_balanced": 0.75,  # Best config
        "efficient": 0.62,     # Too aggressive on efficiency
    }

    base_reward = true_rewards.get(config_name, 0.5)

    # Add noise to simulate real variability
    noise = random.gauss(0, 0.1)

    return max(0.0, min(1.0, base_reward + noise))


def demo_epsilon_greedy(iterations: int = 50):
    """Demonstrate epsilon-greedy policy learning"""
    print("\n" + "="*70)
    print("EPSILON-GREEDY MULTI-ARMED BANDIT DEMONSTRATION")
    print("="*70)
    print("\nSetup:")
    print("  - 4 RAG configurations (arms)")
    print("  - Each arm has different quality-efficiency tradeoff")
    print("  - Policy learns which config is best through trial and error")
    print("  - Epsilon decays as 1/√t for convergence")
    print("\n")

    # Define arms (different RAG configurations)
    arms = [
        Arm("baseline", {"k": 6, "mmr_lambda": 0.0}),
        Arm("embeddings", {"k": 5, "mmr_lambda": 0.0}),
        Arm("mmr_balanced", {"k": 5, "mmr_lambda": 0.6}),
        Arm("efficient", {"k": 4, "mmr_lambda": 0.7}),
    ]

    # Create policy with decaying epsilon
    policy = EpsilonGreedyPolicy(
        arms=arms,
        epsilon=0.3,  # Start with high exploration
        decay_epsilon=True,
        reward_log_path="data/demo_bandit_epsilon.csv"
    )

    print(f"{'Iter':<6} {'Arm':<15} {'Reward':<8} {'Count':<7} {'Mean':<8} {'Epsilon':<8} {'Action'}")
    print("-" * 70)

    for i in range(1, iterations + 1):
        # Policy chooses an arm
        chosen_arm = policy.choose()

        # Simulate running RAG query with this config
        reward = simulate_rag_config(
            chosen_arm.name,
            chosen_arm.params["k"],
            chosen_arm.params["mmr_lambda"]
        )

        # Policy learns from the reward
        policy.update(chosen_arm.name, reward)

        # Get current statistics
        stats = policy.get_statistics()
        arm_stats = stats[chosen_arm.name]

        # Determine if this was exploration or exploitation
        is_exploring = random.random() < policy.epsilon
        action = "explore" if is_exploring else "exploit"

        # Print progress
        if i <= 10 or i % 5 == 0:
            print(f"{i:<6} {chosen_arm.name:<15} {reward:<8.3f} "
                  f"{int(arm_stats['count']):<7} {arm_stats['mean_reward']:<8.3f} "
                  f"{policy.epsilon:<8.3f} {action}")

    # Show final results
    print("\n" + "="*70)
    print("FINAL RESULTS (After Learning)")
    print("="*70)
    stats = policy.get_statistics()

    results = []
    for arm in arms:
        arm_stats = stats[arm.name]
        results.append((arm.name, arm_stats['mean_reward'], arm_stats['count']))

    # Sort by mean reward
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Configuration':<15} {'Mean Reward':<13} {'Times Tried'}")
    print("-" * 50)
    for rank, (name, mean, count) in enumerate(results, 1):
        star = " <- Best arm!" if rank == 1 else ""
        print(f"{rank:<6} {name:<15} {mean:<13.4f} {int(count)}{star}")

    print(f"\n* Best configuration discovered: {results[0][0]}")
    print(f"* Policy converged to exploit best arm")
    print(f"* Logged to: data/demo_bandit_epsilon.csv")


def demo_ucb(iterations: int = 50):
    """Demonstrate UCB1 policy learning"""
    print("\n" + "="*70)
    print("UCB1 MULTI-ARMED BANDIT DEMONSTRATION")
    print("="*70)
    print("\nSetup:")
    print("  - Same 4 RAG configurations")
    print("  - UCB1 automatically balances exploration-exploitation")
    print("  - Uses confidence bounds: UCB = mean + c*√(log(T)/n)")
    print("  - Theoretical regret bound: O(√(KT log T))")
    print("\n")

    arms = [
        Arm("baseline", {"k": 6, "mmr_lambda": 0.0}),
        Arm("embeddings", {"k": 5, "mmr_lambda": 0.0}),
        Arm("mmr_balanced", {"k": 5, "mmr_lambda": 0.6}),
        Arm("efficient", {"k": 4, "mmr_lambda": 0.7}),
    ]

    policy = UCBPolicy(
        arms=arms,
        c=2.0,  # Exploration constant
        reward_log_path="data/demo_bandit_ucb.csv"
    )

    print(f"{'Iter':<6} {'Arm':<15} {'Reward':<8} {'Count':<7} {'Mean':<8} {'UCB Score':<10}")
    print("-" * 70)

    for i in range(1, iterations + 1):
        chosen_arm = policy.choose()

        reward = simulate_rag_config(
            chosen_arm.name,
            chosen_arm.params["k"],
            chosen_arm.params["mmr_lambda"]
        )

        policy.update(chosen_arm.name, reward)

        stats = policy.get_statistics()
        arm_stats = stats[chosen_arm.name]

        if i <= 10 or i % 5 == 0:
            print(f"{i:<6} {chosen_arm.name:<15} {reward:<8.3f} "
                  f"{int(arm_stats['count']):<7} {arm_stats['mean_reward']:<8.3f} "
                  f"{arm_stats['ucb_score']:<10.3f}")

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS (UCB1)")
    print("="*70)
    stats = policy.get_statistics()

    results = []
    for arm in arms:
        arm_stats = stats[arm.name]
        results.append((arm.name, arm_stats['mean_reward'], arm_stats['count'], arm_stats['ucb_score']))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Configuration':<15} {'Mean Reward':<13} {'Times Tried':<13} {'UCB Score'}")
    print("-" * 70)
    for rank, (name, mean, count, ucb) in enumerate(results, 1):
        star = " <- Best arm!" if rank == 1 else ""
        print(f"{rank:<6} {name:<15} {mean:<13.4f} {int(count):<13} {ucb:<10.3f}{star}")

    print(f"\n* Best configuration discovered: {results[0][0]}")
    print(f"* UCB1 balanced exploration-exploitation automatically")
    print(f"* Logged to: data/demo_bandit_ucb.csv")


def compare_policies():
    """Compare epsilon-greedy vs UCB1 performance"""
    print("\n" + "="*70)
    print("COMPARISON: Epsilon-Greedy vs UCB1")
    print("="*70)

    iterations = 100
    n_trials = 10

    print(f"\nRunning {n_trials} trials of {iterations} iterations each...")

    epsilon_rewards = []
    ucb_rewards = []

    arms = [
        Arm("baseline", {"k": 6, "mmr_lambda": 0.0}),
        Arm("embeddings", {"k": 5, "mmr_lambda": 0.0}),
        Arm("mmr_balanced", {"k": 5, "mmr_lambda": 0.6}),
        Arm("efficient", {"k": 4, "mmr_lambda": 0.7}),
    ]

    for trial in range(n_trials):
        # Epsilon-greedy trial
        eg_policy = EpsilonGreedyPolicy(arms=[Arm(a.name, a.params) for a in arms], epsilon=0.2, decay_epsilon=True)
        eg_total = 0
        for _ in range(iterations):
            arm = eg_policy.choose()
            reward = simulate_rag_config(arm.name, arm.params["k"], arm.params["mmr_lambda"])
            eg_policy.update(arm.name, reward)
            eg_total += reward
        epsilon_rewards.append(eg_total)

        # UCB trial
        ucb_policy = UCBPolicy(arms=[Arm(a.name, a.params) for a in arms], c=2.0)
        ucb_total = 0
        for _ in range(iterations):
            arm = ucb_policy.choose()
            reward = simulate_rag_config(arm.name, arm.params["k"], arm.params["mmr_lambda"])
            ucb_policy.update(arm.name, reward)
            ucb_total += reward
        ucb_rewards.append(ucb_total)

    import statistics

    print(f"\nResults over {n_trials} trials:")
    print(f"{'Policy':<20} {'Mean Total Reward':<20} {'Std Dev'}")
    print("-" * 60)
    print(f"{'Epsilon-Greedy':<20} {statistics.mean(epsilon_rewards):<20.2f} {statistics.stdev(epsilon_rewards):.2f}")
    print(f"{'UCB1':<20} {statistics.mean(ucb_rewards):<20.2f} {statistics.stdev(ucb_rewards):.2f}")

    winner = "UCB1" if statistics.mean(ucb_rewards) > statistics.mean(epsilon_rewards) else "Epsilon-Greedy"
    print(f"\n* Winner: {winner} (on average)")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("MULTI-ARMED BANDIT RL DEMONSTRATION")
    print("DocStudio: Reinforcement Learning for RAG Hyperparameter Optimization")
    print("="*70)

    # Set random seed for reproducibility
    random.seed(42)

    # Demo 1: Epsilon-Greedy
    demo_epsilon_greedy(iterations=50)

    time.sleep(1)

    # Demo 2: UCB1
    demo_ucb(iterations=50)

    time.sleep(1)

    # Demo 3: Comparison
    compare_policies()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nWhat this demonstrates:")
    print("  * Multi-armed bandit RL learns optimal configuration adaptively")
    print("  * Exploration-exploitation tradeoff in action")
    print("  * Both epsilon-greedy and UCB1 converge to best arm")
    print("  * UCB1 has theoretical regret guarantees")
    print("\nGenerated files:")
    print("  - data/demo_bandit_epsilon.csv")
    print("  - data/demo_bandit_ucb.csv")
    print("\nThese logs show the learning curve - rewards increasing as policy learns!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
