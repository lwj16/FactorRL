from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training Loss and Reward curves from history CSV.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "ashare_qfr" / "history.csv",
        help="Path to history CSV produced by run_experiment (default: artifacts/ashare_qfr/history.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "ashare_qfr",
        help="Directory to write plot images (default: same folder as history)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hist_path: Path = args.history
    out_dir: Path = args.output_dir
    if not hist_path.exists():
        print(f"History file not found: {hist_path}")
        sys.exit(1)

    try:
        import pandas as pd
    except Exception as e:
        print("pandas is required to read history CSV. Install with: pip install pandas")
        raise

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to plot. Install with: pip install matplotlib")
        sys.exit(1)

    df = pd.read_csv(hist_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot loss and mean_reward over episodes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    if "loss" in df.columns:
        axes[0].plot(df.index, df["loss"], label="loss")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No 'loss' column in history.csv", ha="center")

    if "mean_reward" in df.columns:
        axes[1].plot(df.index, df["mean_reward"], color="tab:orange", label="mean_reward")
        axes[1].set_ylabel("Mean Reward")
        axes[1].set_xlabel("Episode")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No 'mean_reward' column in history.csv", ha="center")

    fig.suptitle("Training History")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_file = out_dir / "history_plots.png"
    fig.savefig(out_file)
    print(f"Saved combined plot to: {out_file}")

    # Also save separate plots if available
    if "loss" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["loss"], label="loss")
        plt.ylabel("Loss")
        plt.xlabel("Episode")
        plt.title("Loss over Episodes")
        plt.legend()
        loss_file = out_dir / "loss.png"
        plt.savefig(loss_file)
        print(f"Saved loss plot to: {loss_file}")

    if "mean_reward" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df["mean_reward"], color="tab:orange", label="mean_reward")
        plt.ylabel("Mean Reward")
        plt.xlabel("Episode")
        plt.title("Mean Reward over Episodes")
        plt.legend()
        reward_file = out_dir / "mean_reward.png"
        plt.savefig(reward_file)
        print(f"Saved reward plot to: {reward_file}")


if __name__ == "__main__":
    main()
