from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfr.config import load_config
from qfr.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a QFR experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "ashare_reproduction.yaml",
        help="Path to the experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "ashare_qfr",
        help="Directory used for run outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'cpu', 'cuda', or 'auto' (default: use config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Resolve device: CLI overrides config; 'auto' selects CUDA if available
    if args.device is not None:
        device_choice = args.device
    else:
        device_choice = config.device
    if device_choice == "auto":
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device_choice
    summary = run_experiment(config=config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
