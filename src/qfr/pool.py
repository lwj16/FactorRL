from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qfr.config import PoolConfig, RewardConfig
from qfr.data import MarketData
from qfr.reward import RewardStats, compute_reward_stats, normalize_cross_section
from qfr.rpn import Expression


@dataclass(slots=True)
class PoolEntry:
    expression: Expression
    weight: float
    stats: RewardStats


@dataclass(slots=True)
class CandidateResult:
    expression: Expression | None
    rendered: str
    stats: RewardStats
    weights: np.ndarray
    valid: bool


class FactorPool:
    def __init__(self, data: MarketData, pool_cfg: PoolConfig, reward_cfg: RewardConfig) -> None:
        self.data = data
        self.pool_cfg = pool_cfg
        self.reward_cfg = reward_cfg
        self.entries: list[PoolEntry] = []

    def _factor_matrix(self, expression: Expression) -> pd.DataFrame:
        raw = expression.evaluate(self.data)
        return normalize_cross_section(raw).replace([np.inf, -np.inf], np.nan)

    def _fit_weights(self, matrices: list[pd.DataFrame]) -> tuple[np.ndarray, pd.DataFrame]:
        target = self.data.target.iloc[self.data.splits.train]
        aligned = [matrix.loc[target.index] for matrix in matrices]
        flat_x = [matrix.to_numpy().reshape(-1) for matrix in aligned]
        x = np.column_stack(flat_x)
        y = target.to_numpy().reshape(-1)
        mask = np.isfinite(y)
        for idx in range(x.shape[1]):
            mask &= np.isfinite(x[:, idx])
        x = x[mask]
        y = y[mask]
        if len(y) == 0:
            raise ValueError("No valid samples were available for factor fitting.")
        ridge = self.pool_cfg.ridge * np.eye(x.shape[1])
        weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        combined = sum(weight * matrix for weight, matrix in zip(weights, matrices))
        return weights, combined

    def evaluate_candidate(self, expression: Expression, step: int) -> CandidateResult:
        try:
            candidate_matrix = self._factor_matrix(expression)
            matrices = [self._factor_matrix(entry.expression) for entry in self.entries] + [candidate_matrix]
            weights, combined = self._fit_weights(matrices)
            stats = compute_reward_stats(
                values=combined.loc[self.data.target.index[self.data.splits.train]],
                target=self.data.target.iloc[self.data.splits.train],
                reward_cfg=self.reward_cfg,
                step=step,
            )
            return CandidateResult(
                expression=expression,
                rendered=expression.render(),
                stats=stats,
                weights=weights,
                valid=True,
            )
        except Exception:
            return self.invalid_result()

    @staticmethod
    def invalid_result() -> CandidateResult:
        bad = RewardStats(reward=-1.0, mean_ic=-1.0, ir=-1.0, threshold=0.0, rank_ic=-1.0)
        return CandidateResult(
            expression=None,
            rendered="<invalid>",
            stats=bad,
            weights=np.array([], dtype=float),
            valid=False,
        )

    def maybe_add(self, result: CandidateResult) -> None:
        if not result.valid or result.expression is None:
            return
        weights = result.weights
        expressions = [entry.expression for entry in self.entries] + [result.expression]
        entries = [
            PoolEntry(expression=expr, weight=float(weight), stats=result.stats)
            for expr, weight in zip(expressions, weights)
        ]
        deduped: dict[str, PoolEntry] = {}
        for entry in entries:
            rendered = entry.expression.render()
            current = deduped.get(rendered)
            if current is None or abs(entry.weight) > abs(current.weight):
                deduped[rendered] = entry
        ranked = sorted(deduped.values(), key=lambda entry: abs(entry.weight), reverse=True)
        self.entries = ranked[: self.pool_cfg.max_factors]

    def summary(self) -> list[dict[str, float | str]]:
        return [
            {
                "expression": entry.expression.render(),
                "weight": entry.weight,
                "reward": entry.stats.reward,
                "mean_ic": entry.stats.mean_ic,
                "ir": entry.stats.ir,
            }
            for entry in self.entries
        ]
