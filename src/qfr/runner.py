from __future__ import annotations

from pathlib import Path

import pandas as pd

from qfr.config import ExperimentConfig
from qfr.data import load_market_data
from qfr.env import FormulaEnv
from qfr.pool import FactorPool
from qfr.policy import PolicyNetwork
from qfr.tokens import build_vocabulary
from qfr.trainer import QFRTrainer
from qfr.utils import ensure_dir, set_seed, write_json


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> dict[str, object]:
    set_seed(config.seed)
    out_dir = ensure_dir(Path(output_dir))
    vocab = build_vocabulary(config.tokens)
    data = load_market_data(config.data, seed=config.seed, feature_names=config.tokens.features)
    env = FormulaEnv(vocab=vocab, max_length=config.tokens.max_length)
    pool = FactorPool(data=data, pool_cfg=config.pool, reward_cfg=config.reward)
    policy = PolicyNetwork(
        vocab_size=len(vocab),
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
    )
    trainer = QFRTrainer(
        config=config,
        vocab=vocab,
        env=env,
        pool=pool,
        policy=policy,
    )
    history, pool_summary = trainer.train()

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out_dir / "history.csv", index=False)
    write_json(out_dir / "pool.json", {"entries": pool_summary})

    summary = {
        "experiment_name": config.experiment_name,
        "history_path": str((out_dir / "history.csv").resolve()),
        "pool_path": str((out_dir / "pool.json").resolve()),
        "episodes": config.training.episodes,
        "final_mean_reward": float(history_frame["mean_reward"].iloc[-1]) if not history_frame.empty else None,
        "final_pool_size": int(history_frame["pool_size"].iloc[-1]) if not history_frame.empty else 0,
        "best_expression": history_frame["best_expression"].iloc[-1] if not history_frame.empty else None,
        "pool": pool_summary,
    }
    write_json(out_dir / "summary.json", summary)
    return summary