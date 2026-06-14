from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qfr.config import RewardConfig


@dataclass(slots=True)
class RewardStats:
    reward: float
    mean_ic: float
    ir: float
    threshold: float
    rank_ic: float


def normalize_cross_section(values: pd.DataFrame) -> pd.DataFrame:
    centered = values.sub(values.mean(axis=1), axis=0)
    scale = centered.abs().max(axis=1).replace(0.0, 1.0)
    return centered.div(scale, axis=0)


def _daily_pearson(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 2:
        return 0.0
    x = left[valid].to_numpy(dtype=float)
    y = right[valid].to_numpy(dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def daily_ic_series(values: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    aligned_values, aligned_target = values.align(target, join="inner", axis=0)
    scores = [
        _daily_pearson(aligned_values.iloc[idx], aligned_target.iloc[idx])
        for idx in range(len(aligned_values.index))
    ]
    return pd.Series(scores, index=aligned_values.index, dtype=float)


def daily_rank_ic_series(values: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    ranked_values = values.rank(axis=1, method="average")
    ranked_target = target.rank(axis=1, method="average")
    return daily_ic_series(ranked_values, ranked_target)


def compute_reward_stats(
    values: pd.DataFrame,
    target: pd.DataFrame,
    reward_cfg: RewardConfig,
    step: int,
) -> RewardStats:
    ic_series = daily_ic_series(values, target)
    rank_series = daily_rank_ic_series(values, target)
    mean_ic = float(ic_series.mean()) if not ic_series.empty else -1.0
    rank_ic = float(rank_series.mean()) if not rank_series.empty else -1.0
    std_ic = float(ic_series.std(ddof=0)) if not ic_series.empty else 0.0
    ir = float(mean_ic / max(std_ic, 1e-6))
    threshold = float(np.clip((step - reward_cfg.alpha_delay) * reward_cfg.eta, 0.0, reward_cfg.delta))
    reward = mean_ic - reward_cfg.lambda_penalty * float(ir <= threshold)
    return RewardStats(
        reward=reward,
        mean_ic=mean_ic,
        ir=ir,
        threshold=threshold,
        rank_ic=rank_ic,
    )
