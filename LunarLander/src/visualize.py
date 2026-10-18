
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Directories
METRICS_PATH = Path("results/metrics/training_metrics.csv")
PLOTS_DIR = Path("results/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_rewards():
    """Plot episode rewards and 100-episode moving average."""

    df = pd.read_csv(METRICS_PATH)

    # Calculate moving average
    df["moving_average"] = (
        df["reward"]
        .rolling(window=100, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(12, 6))

    # Individual episode rewards
    plt.plot(
        df["episode"],
        df["reward"],
        label="Episode Reward",
        color="lightblue",
        alpha=0.5,
        linewidth=1
    )

    # Moving average
    plt.plot(
        df["episode"],
        df["moving_average"],
        label="100-Episode Moving Average",
        color="blue",
        linewidth=2
    )

    # Reference line
    plt.axhline(
        y=200,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Reward = 200"
    )

    plt.title("DQN LunarLander - Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / "training_rewards.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_training_loss():
    """Plot mean Huber loss per episode."""

    df = pd.read_csv(METRICS_PATH)

    # Smooth loss curve
    df["smoothed_loss"] = (
        df["mean_loss"]
        .rolling(window=50, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["episode"],
        df["mean_loss"],
        label="Mean Episode Loss",
        color="orange",
        alpha=0.4,
        linewidth=1
    )

    plt.plot(
        df["episode"],
        df["smoothed_loss"],
        label="50-Episode Moving Average",
        color="red",
        linewidth=2
    )

    plt.title("DQN LunarLander - Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Mean Huber Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / "training_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_checkpoint_comparison():
    """Compare checkpoint performance using fixed-seed evaluation results."""

    models = ["Episode 799", "Episode 899", "Final Model"]

    mean_rewards = np.array([
        221.62399317663795,
        157.84301368865982,
        71.51664548908047
    ])

    std_rewards = np.array([
        57.23996391117929,
        170.61260637747816,
        133.50002268263043
    ])

    success_rates = [85.0, 65.0, 15.0]

    x = np.arange(len(models))

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        x,
        mean_rewards,
        yerr=std_rewards,
        capsize=7,
        color=["seagreen", "steelblue", "gray"],
        alpha=0.85,
        error_kw={"elinewidth": 1.5}
    )

    plt.axhline(
        y=200,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Reward = 200"
    )

    # Add mean reward and success rate labels
    for bar, mean, success_rate in zip(
        bars, mean_rewards, success_rates
    ):
        plt.annotate(
            f"Mean: {mean:.1f}\n>=200: {success_rate:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xticks(x, models)
    plt.ylabel("Mean Evaluation Reward")
    plt.title("DQN LunarLander - Checkpoint Comparison (20 Episodes)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = PLOTS_DIR / "checkpoint_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    plot_training_rewards()
    plot_training_loss()
    plot_checkpoint_comparison()

    print("\nAll visualization plots generated successfully!")


if __name__ == "__main__":
    main()
