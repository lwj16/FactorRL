from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    kind: str
    n_dates: int = 0
    n_assets: int = 0
    lookback: int = 20
    forecast_horizon: int = 5
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    dataset_path: str | None = None
    date_column: str = "date"
    asset_column: str = "sample_code"
    start_date: str | None = None
    end_date: str | None = None


@dataclass(slots=True)
class TokenConfig:
    features: list[str]
    constants: list[float]
    windows: list[int]
    max_length: int


@dataclass(slots=True)
class ModelConfig:
    embedding_dim: int
    hidden_dim: int
    num_heads: int = 4


@dataclass(slots=True)
class TrainingConfig:
    episodes: int
    batch_size: int
    learning_rate: float
    entropy_weight: float
    grad_clip: float
    log_every: int


@dataclass(slots=True)
class RewardConfig:
    lambda_penalty: float
    alpha_delay: int
    eta: float
    delta: float


@dataclass(slots=True)
class PoolConfig:
    max_factors: int
    ridge: float


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    data: DataConfig
    tokens: TokenConfig
    model: ModelConfig
    training: TrainingConfig
    reward: RewardConfig
    pool: PoolConfig


def _build_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    return cls(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],
        data=_build_dataclass(DataConfig, raw["data"]),
        tokens=_build_dataclass(TokenConfig, raw["tokens"]),
        model=_build_dataclass(ModelConfig, raw["model"]),
        training=_build_dataclass(TrainingConfig, raw["training"]),
        reward=_build_dataclass(RewardConfig, raw["reward"]),
        pool=_build_dataclass(PoolConfig, raw["pool"]),
    )
