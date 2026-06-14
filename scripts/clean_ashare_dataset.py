from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINANCIAL_INDICATOR_MAP = {
    "roe": "净资产收益率(ROE)",
    "roa": "总资产报酬率(ROA)",
    "gross_margin": "毛利率",
    "net_margin": "销售净利率",
    "debt_to_assets": "资产负债率",
    "ocf_per_share": "每股经营现金流",
    "revenue": "营业总收入",
    "net_profit": "归母净利润",
}

PRICE_COLUMNS = ["date", "sample_code", "exchange", "open", "high", "low", "close", "volume", "amount"]
OUTPUT_COLUMNS = PRICE_COLUMNS + list(FINANCIAL_INDICATOR_MAP) + ["revenue_yoy", "net_profit_yoy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned A-share dataset CSV from raw SH/SZ files.")
    parser.add_argument("--dataset-root", default="dataset", help="Root directory containing SH, SZ, and financial_abstracts.")
    parser.add_argument("--output", default="dataset/ashare_merged.csv", help="Output CSV path.")
    return parser.parse_args()


def _load_price_frame(file_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    renamed = frame.rename(columns={"timetag": "date", "volumn": "volume"})
    required_columns = {"date", "open", "high", "low", "close", "volume"}
    missing_required = sorted(required_columns - set(renamed.columns))
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise KeyError(f"{file_path} is missing required columns: {missing_text}")
    selected_columns = [column for column in ["date", "open", "high", "low", "close", "volume", "amount"] if column in renamed.columns]
    selected = renamed[selected_columns].copy()
    selected["date"] = pd.to_datetime(selected["date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in [column for column in ["open", "high", "low", "close", "volume", "amount"] if column in selected.columns]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["sample_code"] = file_path.stem.removeprefix("price_")
    selected["exchange"] = file_path.parent.name
    return selected.loc[selected["date"].notna()].copy()


def load_price_data(dataset_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    file_patterns = {
        "SH": "price_60*.csv",
        "SZ": "price_00*.csv",
    }
    for exchange_name, pattern in file_patterns.items():
        exchange_dir = dataset_root / exchange_name
        if not exchange_dir.exists():
            continue
        for file_path in sorted(exchange_dir.glob(pattern)):
            frames.append(_load_price_frame(file_path))
    if not frames:
        raise FileNotFoundError(f"No eligible price files were found under {dataset_root}.")
    combined = pd.concat(frames, ignore_index=True)
    return combined[PRICE_COLUMNS].sort_values(["date", "sample_code"]).reset_index(drop=True)


def _append_yoy_features(financial_frame: pd.DataFrame) -> pd.DataFrame:
    frame = financial_frame.copy()
    frame["report_mmdd"] = frame.index.strftime("%m%d")
    for raw_column, target_column in (("revenue", "revenue_yoy"), ("net_profit", "net_profit_yoy")):
        if raw_column not in frame.columns:
            frame[target_column] = np.nan
            continue
        previous = frame[[raw_column, "report_mmdd"]].reset_index()
        previous["date"] = previous["date"] + pd.DateOffset(years=1)
        previous = previous.rename(columns={raw_column: f"{raw_column}_prev_year"})
        merged = frame.reset_index().merge(previous, on=["date", "report_mmdd"], how="left")
        denominator = merged[f"{raw_column}_prev_year"].abs().replace(0.0, np.nan)
        merged[target_column] = (merged[raw_column] - merged[f"{raw_column}_prev_year"]) / denominator
        frame = merged.drop(columns=[f"{raw_column}_prev_year"]).set_index("date")
    return frame.drop(columns=["report_mmdd"], errors="ignore")


def _load_financial_frame(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=[*FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"])
    frame = pd.read_csv(file_path, encoding="utf-8-sig")
    if frame.shape[1] < 3:
        return pd.DataFrame(columns=[*FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"])
    indicator_column = frame.columns[1]
    date_columns = frame.columns[2:]
    subset = frame.loc[frame[indicator_column].isin(FINANCIAL_INDICATOR_MAP.values()), [indicator_column, *date_columns]].copy()
    if subset.empty:
        return pd.DataFrame(columns=[*FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"])
    reverse_map = {value: key for key, value in FINANCIAL_INDICATOR_MAP.items()}
    melted = subset.melt(id_vars=[indicator_column], var_name="report_date", value_name="value")
    melted["feature_name"] = melted[indicator_column].map(reverse_map)
    melted["date"] = pd.to_datetime(melted["report_date"].astype(str), format="%Y%m%d", errors="coerce")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["date", "feature_name", "value"])
    if melted.empty:
        return pd.DataFrame(columns=[*FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"])
    panel = melted.pivot_table(index="date", columns="feature_name", values="value", aggfunc="last").sort_index()
    for column in FINANCIAL_INDICATOR_MAP:
        if column not in panel.columns:
            panel[column] = np.nan
    panel = panel[[*FINANCIAL_INDICATOR_MAP]]
    panel = _append_yoy_features(panel)
    for column in ["revenue_yoy", "net_profit_yoy"]:
        if column not in panel.columns:
            panel[column] = np.nan
    return panel[[*FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"]]


def load_financial_data(dataset_root: Path, sample_codes: pd.Series | list[str]) -> pd.DataFrame:
    financial_dir = dataset_root / "financial_abstracts"
    frames: list[pd.DataFrame] = []
    for sample_code in sorted({str(code) for code in sample_codes}):
        financial_frame = _load_financial_frame(financial_dir / f"{sample_code}.csv")
        if financial_frame.empty:
            continue
        enriched = financial_frame.reset_index()
        enriched["sample_code"] = sample_code
        frames.append(enriched)
    if not frames:
        return pd.DataFrame(columns=["date", "sample_code", *FINANCIAL_INDICATOR_MAP, "revenue_yoy", "net_profit_yoy"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["date", "sample_code"]).reset_index(drop=True)


def merge_financial_data(price_frame: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    price_view = price_frame[PRICE_COLUMNS].sort_values(["date", "sample_code"]).reset_index(drop=True)
    financial_view = load_financial_data(dataset_root, price_view["sample_code"].unique())
    if financial_view.empty:
        merged = price_view.copy()
    else:
        merged = pd.merge_asof(
            price_view,
            financial_view,
            on="date",
            by="sample_code",
            direction="backward",
        )
    for column in OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[OUTPUT_COLUMNS].sort_values(["date", "sample_code"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    price_frame = load_price_data(dataset_root)
    merged = merge_financial_data(price_frame, dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(merged):,} rows to {output_path}")


if __name__ == "__main__":
    main()