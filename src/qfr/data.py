from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from qfr.config import DataConfig


@dataclass(slots=True)
class SplitSlices:
    train: slice
    valid: slice
    test: slice


@dataclass(slots=True)
class MarketData:
    features: dict[str, pd.DataFrame]
    target: pd.DataFrame
    splits: SplitSlices

    @property
    def index(self) -> pd.Index:
        return self.target.index

    @property
    def columns(self) -> pd.Index:
        return self.target.columns


DERIVED_FEATURE_BUILDERS = {
    "vwap": lambda frame: (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0,
}

REQUIRED_MARKET_COLUMNS = {"close", "volume"}


def _make_price_panel(cfg: DataConfig, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2018-01-01", periods=cfg.n_dates)
    assets = [f"asset_{idx:03d}" for idx in range(cfg.n_assets)]
    index = pd.Index(dates, name="date")
    columns = pd.Index(assets, name="asset")

    market_noise = rng.normal(0.0008, 0.01, size=(cfg.n_dates, 1))
    style_noise = rng.normal(0.0, 0.008, size=(cfg.n_dates, cfg.n_assets))
    drift = rng.normal(0.0002, 0.0004, size=(1, cfg.n_assets))

    close_returns = drift + market_noise + style_noise
    close = 100.0 * np.exp(np.cumsum(close_returns, axis=0))
    overnight = rng.normal(0.0, 0.004, size=close.shape)
    open_ = close * (1.0 + overnight)
    intraday = np.abs(rng.normal(0.008, 0.004, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    liquidity = np.abs(rng.normal(0.0, 0.12, size=close.shape))
    volume = 1.5e6 * np.exp(rng.normal(0.0, 0.35, size=close.shape)) * (1.0 + liquidity)
    vwap = ((open_ + high + low + close) / 4.0) * (1.0 + rng.normal(0.0, 0.002, size=close.shape))

    frames = {
        "open": pd.DataFrame(open_, index=index, columns=columns),
        "high": pd.DataFrame(high, index=index, columns=columns),
        "low": pd.DataFrame(low, index=index, columns=columns),
        "close": pd.DataFrame(close, index=index, columns=columns),
        "volume": pd.DataFrame(volume, index=index, columns=columns),
        "vwap": pd.DataFrame(vwap, index=index, columns=columns),
    }
    return frames


def _make_target(features: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    close = features["close"]
    volume = features["volume"]

    mean_reversion = -(close / close.rolling(20, min_periods=20).mean() - 1.0)
    price_momentum = close.pct_change(5)
    volume_shock = volume.pct_change(5)

    latent = 0.45 * mean_reversion + 0.35 * price_momentum + 0.20 * volume_shock
    latent = latent.replace([np.inf, -np.inf], np.nan)
    future_return = close.shift(-horizon) / close - 1.0
    blended = 0.55 * future_return + 0.45 * latent
    return blended.iloc[:-horizon].copy()


def _make_splits(length: int, train_ratio: float, valid_ratio: float) -> SplitSlices:
    train_end = int(length * train_ratio)
    valid_end = train_end + int(length * valid_ratio)
    return SplitSlices(
        train=slice(0, train_end),
        valid=slice(train_end, valid_end),
        test=slice(valid_end, length),
    )


def generate_toy_market_data(cfg: DataConfig, seed: int) -> MarketData:
    rng = np.random.default_rng(seed)
    features = _make_price_panel(cfg, rng)
    target = _make_target(features, cfg.forecast_horizon)
    aligned_features = {name: frame.loc[target.index].copy() for name, frame in features.items()}
    splits = _make_splits(len(target.index), cfg.train_ratio, cfg.valid_ratio)
    return MarketData(features=aligned_features, target=target, splits=splits)


def _resolve_dataset_path(cfg: DataConfig) -> Path:
    if not cfg.dataset_path:
        raise ValueError("data.dataset_path must be set when data.kind is not 'toy'.")
    path = Path(cfg.dataset_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return path


def _read_market_frame(path: Path) -> pd.DataFrame:
    return _read_market_frame_cached(str(path.resolve())).copy()


@lru_cache(maxsize=8)
def _read_market_frame_cached(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if path.suffix.lower() == ".pkl":
        frame = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"sample_code": str})
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset loader expects a pandas DataFrame.")
    return frame.copy()


def _normalize_market_frame(frame: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[cfg.date_column] = pd.to_datetime(normalized[cfg.date_column], errors="coerce")
    normalized[cfg.asset_column] = normalized[cfg.asset_column].astype(str).str.strip()
    if cfg.asset_column == "sample_code":
        normalized[cfg.asset_column] = normalized[cfg.asset_column].str.zfill(6)
    numeric_columns = [column for column in normalized.columns if column not in {cfg.date_column, cfg.asset_column, "exchange"}]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.loc[normalized[cfg.date_column].notna() & normalized[cfg.asset_column].ne("")].copy()


def _filter_date_range(frame: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    date_values = pd.to_datetime(frame[cfg.date_column])
    filtered = frame.assign(**{cfg.date_column: date_values})
    if cfg.start_date is not None:
        filtered = filtered.loc[filtered[cfg.date_column] >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date is not None:
        filtered = filtered.loc[filtered[cfg.date_column] <= pd.Timestamp(cfg.end_date)]
    return filtered


def _select_assets(frame: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    if cfg.n_assets <= 0:
        return frame
    coverage = (
        frame.groupby(cfg.asset_column)
        .agg(
            trading_days=(cfg.date_column, "nunique"),
            avg_volume=("volume", "mean"),
        )
        .sort_values(["trading_days", "avg_volume"], ascending=False)
    )
    selected_assets = coverage.head(cfg.n_assets).index
    return frame.loc[frame[cfg.asset_column].isin(selected_assets)].copy()


def _build_feature_panel(frame: pd.DataFrame, cfg: DataConfig, feature_name: str) -> pd.DataFrame:
    source = frame
    if feature_name not in source.columns:
        builder = DERIVED_FEATURE_BUILDERS.get(feature_name)
        if builder is None:
            available = ", ".join(sorted(source.columns))
            raise KeyError(f"Feature '{feature_name}' is missing from dataset. Available columns: {available}")
        source = source.assign(**{feature_name: builder(source)})
    panel = source.pivot_table(
        index=cfg.date_column,
        columns=cfg.asset_column,
        values=feature_name,
        aggfunc="last",
    )
    return panel.apply(pd.to_numeric, errors="coerce").sort_index().sort_index(axis=1)


def clear_market_data_cache() -> None:
    _read_market_frame_cached.cache_clear()


def load_ashare_market_data(cfg: DataConfig, feature_names: list[str]) -> MarketData:
    dataset_path = _resolve_dataset_path(cfg)
    raw = _normalize_market_frame(_read_market_frame(dataset_path), cfg)
    required_columns = {cfg.date_column, cfg.asset_column, *REQUIRED_MARKET_COLUMNS}
    missing_required = sorted(required_columns - set(raw.columns))
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise KeyError(f"Dataset is missing required columns: {missing_text}")

    frame = _filter_date_range(raw, cfg)
    frame = _select_assets(frame, cfg)
    frame = frame.sort_values([cfg.date_column, cfg.asset_column]).drop_duplicates(
        subset=[cfg.date_column, cfg.asset_column],
        keep="last",
    )

    if frame.empty:
        raise ValueError("No market rows remain after date and asset filtering.")

    features = {name: _build_feature_panel(frame, cfg, name) for name in feature_names}
    close = features["close"] if "close" in features else _build_feature_panel(frame, cfg, "close")
    target = close.shift(-cfg.forecast_horizon) / close - 1.0
    if cfg.forecast_horizon > 0:
        target = target.iloc[:-cfg.forecast_horizon]
    target = target.replace([np.inf, -np.inf], np.nan)

    aligned_features = {name: panel.loc[target.index].copy() for name, panel in features.items()}
    splits = _make_splits(len(target.index), cfg.train_ratio, cfg.valid_ratio)
    return MarketData(features=aligned_features, target=target, splits=splits)


def load_market_data(cfg: DataConfig, seed: int, feature_names: list[str]) -> MarketData:
    if cfg.kind == "toy":
        return generate_toy_market_data(cfg, seed)
    if cfg.kind == "ashare":
        return load_ashare_market_data(cfg, feature_names)
    raise ValueError(f"Unsupported data kind: {cfg.kind}")
