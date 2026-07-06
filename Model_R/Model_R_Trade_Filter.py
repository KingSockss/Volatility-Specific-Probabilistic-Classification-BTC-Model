from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


JOIN_KEYS = ["event_contract_id", "forecast_datetime_utc"]
OUTPUT_FOLDER_NAME = "Model_R_Trade_Filter_Outputs"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Filter Model R scored forecast rows to buy-YES opportunities where "
            "Model R probability exceeds Model K price/probability."
        )
    )
    parser.add_argument(
        "--model-r-raw-values",
        type=Path,
        default=root / "Model_R" / "model_R_Evals_Outputs" / "raw_values.csv",
        help="Model R eval raw_values.csv produced by Model_R_Eval.py.",
    )
    parser.add_argument(
        "--model-k-raw-values",
        type=Path,
        default=root / "Model_K" / "Model_K_outputs" / "raw_values.csv",
        help="Model K eval raw_values.csv used as the trade price/probability source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Directory where trade signal rows and summary files will be written.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum p_model_r - p_model_k edge required for a buy-YES signal.",
    )
    return parser.parse_args()


def load_raw_values(path: Path, *, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} raw_values.csv not found: {path}")

    frame = pd.read_csv(path)
    required = set(JOIN_KEYS) | {
        "event_ticker",
        "forecast_datetime_utc",
        "minute_number",
        "source_file",
        "strike",
        "market_ticker",
        "contract_label",
        "p_kalshi",
        "p_reality",
        "official_result",
        "minutes_to_settlement",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["forecast_datetime_utc"] = pd.to_datetime(frame["forecast_datetime_utc"], utc=True)
    for column in ["p_kalshi", "p_reality", "minute_number", "strike", "minutes_to_settlement"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def build_overlap(model_r: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    model_r_cols = [
        "event_contract_id",
        "forecast_datetime_utc",
        "event_ticker",
        "minute_number",
        "source_file",
        "strike",
        "market_ticker",
        "contract_label",
        "p_kalshi",
        "p_reality",
        "official_result",
        "event_datetime_utc",
        "minutes_to_settlement",
    ]
    optional_cols = ["binance_audit_price", "binance_reference_price", "join_key_used"]
    model_r_cols.extend([col for col in optional_cols if col in model_r.columns])

    model_k_cols = ["event_contract_id", "forecast_datetime_utc", "p_kalshi", "source_file"]
    overlap = model_r[model_r_cols].merge(
        model_k[model_k_cols],
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model_r", "_model_k"),
        validate="one_to_one",
    )

    overlap = overlap.rename(
        columns={
            "p_kalshi_model_r": "p_model_r",
            "p_kalshi_model_k": "p_model_k",
            "source_file_model_r": "model_r_source_file",
            "source_file_model_k": "model_k_source_file",
        }
    )
    overlap["edge"] = overlap["p_model_r"] - overlap["p_model_k"]
    overlap["trade_side"] = "BUY_YES"
    overlap["expected_value_per_contract"] = overlap["edge"]
    overlap["realized_pnl_per_contract"] = overlap["p_reality"] - overlap["p_model_k"]
    overlap["gross_payout_per_contract"] = overlap["p_reality"]
    overlap["cost_per_contract"] = overlap["p_model_k"]
    overlap["is_winning_trade"] = overlap["p_reality"] == 1.0
    return overlap.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True)


def summarize_segment(frame: pd.DataFrame, *, segment: str, overlap_rows: int) -> Dict[str, Any]:
    signal_rows = int(len(frame))
    total_cost = float(frame["cost_per_contract"].sum()) if signal_rows else 0.0
    total_pnl = float(frame["realized_pnl_per_contract"].sum()) if signal_rows else 0.0
    total_ev = float(frame["expected_value_per_contract"].sum()) if signal_rows else 0.0

    return {
        "segment": segment,
        "overlap_rows": int(overlap_rows),
        "signal_rows": signal_rows,
        "frequency": signal_rows / overlap_rows if overlap_rows else np.nan,
        "winrate": float(frame["is_winning_trade"].mean()) if signal_rows else np.nan,
        "average_model_r_probability": float(frame["p_model_r"].mean()) if signal_rows else np.nan,
        "average_model_k_price": float(frame["p_model_k"].mean()) if signal_rows else np.nan,
        "average_edge": float(frame["edge"].mean()) if signal_rows else np.nan,
        "average_ev": float(frame["expected_value_per_contract"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "average_realized_gain_loss": float(frame["realized_pnl_per_contract"].mean()) if signal_rows else np.nan,
        "total_realized_gain_loss": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
    }


def build_summary(overlap: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        summarize_segment(signals, segment="overall", overlap_rows=len(overlap))
    ]

    for label, part in signals.groupby("contract_label", dropna=False):
        overlap_count = int((overlap["contract_label"] == label).sum())
        rows.append(summarize_segment(part, segment=f"contract_label={label}", overlap_rows=overlap_count))

    bucketed = signals.copy()
    bucketed["minute_bucket"] = pd.cut(
        bucketed["minutes_to_settlement"],
        bins=[0, 10, 20, 30, 40, 50, 60],
        labels=["1-10", "11-20", "21-30", "31-40", "41-50", "51-60"],
        include_lowest=True,
        right=True,
    )
    overlap_bucketed = overlap.copy()
    overlap_bucketed["minute_bucket"] = pd.cut(
        overlap_bucketed["minutes_to_settlement"],
        bins=[0, 10, 20, 30, 40, 50, 60],
        labels=["1-10", "11-20", "21-30", "31-40", "41-50", "51-60"],
        include_lowest=True,
        right=True,
    )
    for bucket, part in bucketed.groupby("minute_bucket", observed=True, dropna=False):
        if pd.isna(bucket):
            continue
        overlap_count = int((overlap_bucketed["minute_bucket"] == bucket).sum())
        rows.append(summarize_segment(part, segment=f"minutes_to_settlement={bucket}", overlap_rows=overlap_count))

    return pd.DataFrame(rows)


def write_readme(output_dir: Path, *, edge_threshold: float) -> None:
    text = f"""Model R Trade Filter

Signal rule:
Keep overlapping scored rows where p_model_r - p_model_k > {edge_threshold}.

Trade assumption:
Each signal buys one YES contract with a $1 payout. Model K probability is treated as the entry cost.

Metrics:
- expected_value_per_contract = p_model_r - p_model_k
- realized_pnl_per_contract = official_outcome - p_model_k
- winrate = mean official_outcome across signal rows
- total_realized_gain_loss = sum realized_pnl_per_contract
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_r = load_raw_values(args.model_r_raw_values.resolve(), model_name="Model R")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = build_overlap(model_r, model_k)
    signals = overlap[overlap["edge"] > args.edge_threshold].copy()

    signal_cols = [
        "event_ticker",
        "event_contract_id",
        "forecast_datetime_utc",
        "event_datetime_utc",
        "minute_number",
        "minutes_to_settlement",
        "market_ticker",
        "contract_label",
        "strike",
        "trade_side",
        "p_model_r",
        "p_model_k",
        "edge",
        "expected_value_per_contract",
        "cost_per_contract",
        "gross_payout_per_contract",
        "realized_pnl_per_contract",
        "is_winning_trade",
        "official_result",
        "p_reality",
        "model_r_source_file",
        "model_k_source_file",
    ]
    signal_cols.extend([col for col in ["binance_audit_price", "binance_reference_price", "join_key_used"] if col in signals.columns])

    signals[signal_cols].to_csv(output_dir / "trade_signal_minutes.csv", index=False)
    build_summary(overlap, signals).to_csv(output_dir / "trade_summary.csv", index=False)
    write_readme(output_dir, edge_threshold=args.edge_threshold)

    print(f"Overlapping rows: {len(overlap):,}")
    print(f"Signal rows: {len(signals):,}")
    print(f"Trade signal minutes: {output_dir / 'trade_signal_minutes.csv'}")
    print(f"Trade summary: {output_dir / 'trade_summary.csv'}")


if __name__ == "__main__":
    main()
