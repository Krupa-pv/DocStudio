"""
Visualize RL learning curves showing reward improvements over time

Shows how different bandit arms learn optimal configurations through exploration-exploitation
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bandit_learning_curve(
    csv_path: Path,
    output_path: Optional[Path] = None,
    window_size: int = 5,
) -> None:
    """
    Plot learning curve showing mean reward increasing over time for each arm

    Args:
        csv_path: Path to rl_bandit_rewards.csv
        output_path: Where to save plot (if None, displays interactively)
        window_size: Moving average window for smoothing
    """
    df = pd.read_csv(csv_path)

    # Convert timestamp to trial number per arm
    df = df.sort_values('ts_utc')
    df['trial'] = df.groupby('arm').cumcount() + 1

    # Setup plot
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Individual rewards over time (scatter + moving average)
    for arm in df['arm'].unique():
        arm_df = df[df['arm'] == arm]

        # Scatter plot of individual rewards
        ax1.scatter(
            arm_df['trial'],
            arm_df['reward'],
            alpha=0.3,
            label=f'Arm {arm} (raw)',
            s=20
        )

        # Moving average for trend
        if len(arm_df) >= window_size:
            ma = arm_df['reward'].rolling(window=window_size, min_periods=1).mean()
            ax1.plot(
                arm_df['trial'],
                ma,
                linewidth=2.5,
                label=f'Arm {arm} (MA-{window_size})',
                marker='o',
                markersize=4
            )

    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('RL Learning Curve: Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Running mean reward (what the policy actually uses)
    for arm in df['arm'].unique():
        arm_df = df[df['arm'] == arm]
        ax2.plot(
            arm_df['trial'],
            arm_df['mean_reward'],
            linewidth=2.5,
            marker='o',
            markersize=5,
            label=f'Arm {arm}'
        )

    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Running Mean Reward', fontsize=12)
    ax2.set_title('Policy Learning: Running Average Rewards', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curve to {output_path}")
    else:
        plt.show()


def plot_general_rewards(
    csv_path: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot reward trends from general rewards.csv (per-query logging)

    Shows quality, tokens, latency, and composite reward over time
    """
    df = pd.read_csv(csv_path)

    # Add trial number
    df['trial'] = range(1, len(df) + 1)

    # Setup plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot by arm if available
    if 'arm' in df.columns:
        arms = df['arm'].unique()

        # Plot 1: Overall quality by arm
        for arm in arms:
            arm_df = df[df['arm'] == arm]
            axes[0, 0].plot(
                arm_df['trial'],
                arm_df['overall'],
                marker='o',
                label=arm,
                alpha=0.7
            )
        axes[0, 0].set_ylabel('Overall Quality Score', fontsize=11)
        axes[0, 0].set_title('Quality Scores Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Tokens by arm
        for arm in arms:
            arm_df = df[df['arm'] == arm]
            total_tokens = arm_df['tokens_prompt'] + arm_df['tokens_completion']
            axes[0, 1].plot(
                arm_df['trial'],
                total_tokens,
                marker='o',
                label=arm,
                alpha=0.7
            )
        axes[0, 1].set_ylabel('Total Tokens', fontsize=11)
        axes[0, 1].set_title('Token Usage Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Latency by arm
        for arm in arms:
            arm_df = df[df['arm'] == arm]
            axes[1, 0].plot(
                arm_df['trial'],
                arm_df['latency_ms'] / 1000.0,  # Convert to seconds
                marker='o',
                label=arm,
                alpha=0.7
            )
        axes[1, 0].set_ylabel('Latency (seconds)', fontsize=11)
        axes[1, 0].set_xlabel('Trial Number', fontsize=11)
        axes[1, 0].set_title('Latency Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Composite reward by arm
        for arm in arms:
            arm_df = df[df['arm'] == arm]
            axes[1, 1].plot(
                arm_df['trial'],
                arm_df['reward'],
                marker='o',
                label=arm,
                alpha=0.7,
                linewidth=2
            )
        axes[1, 1].set_ylabel('Composite Reward', fontsize=11)
        axes[1, 1].set_xlabel('Trial Number', fontsize=11)
        axes[1, 1].set_title('Reward (Quality - Cost) Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Simple time series if no arm info
        axes[0, 0].plot(df['trial'], df['overall'], marker='o')
        axes[0, 0].set_ylabel('Overall Quality')
        axes[0, 0].set_title('Quality Over Time')

        total_tokens = df['tokens_prompt'] + df['tokens_completion']
        axes[0, 1].plot(df['trial'], total_tokens, marker='o', color='orange')
        axes[0, 1].set_ylabel('Total Tokens')
        axes[0, 1].set_title('Token Usage Over Time')

        axes[1, 0].plot(df['trial'], df['latency_ms'] / 1000.0, marker='o', color='green')
        axes[1, 0].set_ylabel('Latency (seconds)')
        axes[1, 0].set_xlabel('Trial Number')
        axes[1, 0].set_title('Latency Over Time')

        axes[1, 1].plot(df['trial'], df['reward'], marker='o', color='red', linewidth=2)
        axes[1, 1].set_ylabel('Composite Reward')
        axes[1, 1].set_xlabel('Trial Number')
        axes[1, 1].set_title('Reward Over Time')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reward trends to {output_path}")
    else:
        plt.show()


def print_summary_stats(csv_path: Path) -> None:
    """Print summary statistics for each arm"""
    df = pd.read_csv(csv_path)

    print("\n" + "="*60)
    print("BANDIT LEARNING SUMMARY STATISTICS")
    print("="*60)

    for arm in sorted(df['arm'].unique()):
        arm_df = df[df['arm'] == arm]
        print(f"\nArm {arm}:")
        print(f"  Total trials: {len(arm_df)}")
        print(f"  Mean reward: {arm_df['reward'].mean():.4f} (±{arm_df['reward'].std():.4f})")
        print(f"  Final running mean: {arm_df['mean_reward'].iloc[-1]:.4f}")
        print(f"  Reward range: [{arm_df['reward'].min():.4f}, {arm_df['reward'].max():.4f}]")

        # Show improvement
        if len(arm_df) >= 10:
            first_5 = arm_df.head(5)['reward'].mean()
            last_5 = arm_df.tail(5)['reward'].mean()
            improvement = ((last_5 - first_5) / first_5) * 100
            print(f"  Improvement (first 5 vs last 5): {improvement:+.1f}%")

    print("\n" + "="*60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize RL learning curves and reward trends"
    )
    parser.add_argument(
        "--bandit-csv",
        type=Path,
        default=Path("data/rl_bandit_rewards.csv"),
        help="Path to rl_bandit_rewards.csv"
    )
    parser.add_argument(
        "--rewards-csv",
        type=Path,
        default=Path("data/rewards.csv"),
        help="Path to rewards.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/plots"),
        help="Directory to save plots"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of saving"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window size"
    )

    args = parser.parse_args()

    # Create output directory
    if not args.show:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot bandit learning curve
    if args.bandit_csv.exists():
        print(f"\nPlotting bandit learning curve from {args.bandit_csv}")
        output_path = None if args.show else args.output_dir / "learning_curve.png"
        plot_bandit_learning_curve(args.bandit_csv, output_path, args.window)
        print_summary_stats(args.bandit_csv)
    else:
        print(f"Warning: {args.bandit_csv} not found, skipping bandit plot")

    # Plot general reward trends
    if args.rewards_csv.exists():
        print(f"\nPlotting reward trends from {args.rewards_csv}")
        output_path = None if args.show else args.output_dir / "reward_trends.png"
        plot_general_rewards(args.rewards_csv, output_path)
    else:
        print(f"Warning: {args.rewards_csv} not found, skipping reward trends plot")


if __name__ == "__main__":
    main()
