from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    go = None
    make_subplots = None
    HAS_PLOTLY = False


JOIN_KEYS = ["event_contract_id", "forecast_datetime_utc"]
OUTPUT_FOLDER_NAME = "outputs"
DEFAULT_TAKER_FEE_RATE = 0.07

STRATEGIES: Tuple[Tuple[str, str, str], ...] = (
    ("model_a_normal", "Model A Normal", "p_normal"),
    ("model_n_exp_lambda_2", "Model N Exp Hybrid lambda=2", "p_exponential_hybrid"),
)
STRATEGY_COLORS = {
    "Model A Normal": "#35c7b7",
    "Model N Exp Hybrid lambda=2": "#fbbf24",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Taker-fee experiment for lower-fee Kalshi price ends. Compares Model A normal and "
            "Model N exponential hybrid lambda=2 under p_model_k low/high/outside filters."
        )
    )
    parser.add_argument(
        "--model-n-raw-values",
        type=Path,
        default=root / "Model_N" / "model_N_Evals_Outputs" / "raw_values.csv",
        help="Model N raw_values.csv containing p_normal, p_exponential_hybrid, and outcomes.",
    )
    parser.add_argument(
        "--model-k-raw-values",
        type=Path,
        default=root / "Model_K" / "Model_K_outputs" / "raw_values.csv",
        help="Model K raw_values.csv used as the market/entry probability source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Self-contained output folder for this experiment.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum favorable-direction gross edge required before entering a trade.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_TAKER_FEE_RATE,
        help="Kalshi taker fee rate applied to selected_price * (1 - selected_price).",
    )
    parser.add_argument(
        "--no-fee-round-up",
        action="store_true",
        help="Disable cent round-up for the fee model.",
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
    for column in [
        "p_kalshi",
        "p_reality",
        "minute_number",
        "strike",
        "minutes_to_settlement",
        "q_shock",
        "shock_weight_exponential",
        "p_normal",
        "p_exponential_hybrid",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def build_overlap(model_n: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    required_model_n = {"p_normal", "p_exponential_hybrid"}
    missing = required_model_n - set(model_n.columns)
    if missing:
        raise ValueError(f"Model N raw values are missing required strategy columns: {sorted(missing)}")

    model_n_cols = [
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
        "q_shock",
        "shock_weight_exponential",
        "p_normal",
        "p_exponential_hybrid",
    ]
    optional_cols = ["binance_audit_price", "binance_reference_price", "join_key_used"]
    model_n_cols.extend([col for col in optional_cols if col in model_n.columns])

    model_k_cols = ["event_contract_id", "forecast_datetime_utc", "p_kalshi", "source_file"]
    overlap = model_n[model_n_cols].merge(
        model_k[model_k_cols],
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model_n", "_model_k"),
        validate="one_to_one",
    )
    overlap = overlap.rename(
        columns={
            "p_kalshi_model_n": "p_model_n",
            "p_kalshi_model_k": "p_model_k",
            "source_file_model_n": "model_n_source_file",
            "source_file_model_k": "model_k_source_file",
        }
    )
    overlap["p_model_k"] = pd.to_numeric(overlap["p_model_k"], errors="coerce").clip(0.0, 1.0)
    return overlap.dropna(subset=["p_model_k", "p_reality"]).sort_values(
        ["forecast_datetime_utc", "event_ticker", "contract_label"]
    ).reset_index(drop=True)


def scenario_specs() -> List[Dict[str, Any]]:
    return [
        {
            "scenario_id": "baseline_all",
            "scenario_label": "Baseline: All Model K Overlap",
            "filter_family": "baseline",
            "filter_type": "all",
            "low_threshold": np.nan,
            "high_threshold": np.nan,
        },
        {
            "scenario_id": "p_kalshi_lt_0_25",
            "scenario_label": "p_model_k < 0.25",
            "filter_family": "0.25/0.75",
            "filter_type": "low",
            "low_threshold": 0.25,
            "high_threshold": 0.75,
        },
        {
            "scenario_id": "p_kalshi_gt_0_75",
            "scenario_label": "p_model_k > 0.75",
            "filter_family": "0.25/0.75",
            "filter_type": "high",
            "low_threshold": 0.25,
            "high_threshold": 0.75,
        },
        {
            "scenario_id": "p_kalshi_outside_0_26_0_75",
            "scenario_label": "p_model_k < 0.26 OR > 0.75",
            "filter_family": "0.26/0.75",
            "filter_type": "outside",
            "low_threshold": 0.26,
            "high_threshold": 0.75,
        },
        {
            "scenario_id": "p_kalshi_lt_0_10",
            "scenario_label": "p_model_k < 0.10",
            "filter_family": "0.10/0.90",
            "filter_type": "low",
            "low_threshold": 0.10,
            "high_threshold": 0.90,
        },
        {
            "scenario_id": "p_kalshi_gt_0_90",
            "scenario_label": "p_model_k > 0.90",
            "filter_family": "0.10/0.90",
            "filter_type": "high",
            "low_threshold": 0.10,
            "high_threshold": 0.90,
        },
        {
            "scenario_id": "p_kalshi_outside_0_10_0_90",
            "scenario_label": "p_model_k < 0.10 OR > 0.90",
            "filter_family": "0.10/0.90",
            "filter_type": "outside",
            "low_threshold": 0.10,
            "high_threshold": 0.90,
        },
    ]


def apply_scenario_filter(overlap: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    filter_type = spec["filter_type"]
    if filter_type == "all":
        return overlap.copy()
    if filter_type == "low":
        return overlap[overlap["p_model_k"] < float(spec["low_threshold"])].copy()
    if filter_type == "high":
        return overlap[overlap["p_model_k"] > float(spec["high_threshold"])].copy()
    if filter_type == "outside":
        return overlap[
            (overlap["p_model_k"] < float(spec["low_threshold"]))
            | (overlap["p_model_k"] > float(spec["high_threshold"]))
        ].copy()
    raise ValueError(f"Unknown filter type: {filter_type}")


def taker_fee_per_contract(price: pd.Series, *, fee_rate: float, round_up_to_cent: bool) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").clip(0.0, 1.0)
    fee = fee_rate * price * (1.0 - price)
    if round_up_to_cent:
        fee = np.ceil((fee * 100.0) - 1e-12) / 100.0
    return pd.Series(fee, index=price.index).clip(lower=0.0)


def build_strategy_trades(
    eligible: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    scenario: Dict[str, Any],
    edge_threshold: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    if probability_col not in eligible.columns:
        raise ValueError(f"Cannot build {strategy_label}; missing {probability_col}.")

    work = eligible.dropna(subset=[probability_col, "p_model_k", "p_reality"]).copy()
    work["p_strategy"] = pd.to_numeric(work[probability_col], errors="coerce").clip(0.0, 1.0)
    work["yes_edge"] = work["p_strategy"] - work["p_model_k"]
    work["no_edge"] = work["p_model_k"] - work["p_strategy"]

    buy_yes = work[work["yes_edge"] > edge_threshold].copy()
    buy_yes["trade_side"] = "BUY_YES"
    buy_yes["favorable_edge_gross"] = buy_yes["yes_edge"]
    buy_yes["model_probability_selected_side"] = buy_yes["p_strategy"]
    buy_yes["model_k_price_selected_side"] = buy_yes["p_model_k"]
    buy_yes["fee_per_contract"] = taker_fee_per_contract(
        buy_yes["model_k_price_selected_side"], fee_rate=fee_rate, round_up_to_cent=round_up_to_cent
    )
    buy_yes["cost_per_contract_before_fees"] = buy_yes["p_model_k"]
    buy_yes["gross_payout_per_contract"] = buy_yes["p_reality"]
    buy_yes["win"] = (buy_yes["p_reality"] == 1.0).astype(int)

    buy_no = work[work["no_edge"] > edge_threshold].copy()
    buy_no["trade_side"] = "BUY_NO"
    buy_no["favorable_edge_gross"] = buy_no["no_edge"]
    buy_no["model_probability_selected_side"] = 1.0 - buy_no["p_strategy"]
    buy_no["model_k_price_selected_side"] = 1.0 - buy_no["p_model_k"]
    buy_no["fee_per_contract"] = taker_fee_per_contract(
        buy_no["model_k_price_selected_side"], fee_rate=fee_rate, round_up_to_cent=round_up_to_cent
    )
    buy_no["cost_per_contract_before_fees"] = 1.0 - buy_no["p_model_k"]
    buy_no["gross_payout_per_contract"] = 1.0 - buy_no["p_reality"]
    buy_no["win"] = (buy_no["p_reality"] == 0.0).astype(int)

    trades = pd.concat([buy_yes, buy_no], ignore_index=True)
    if trades.empty:
        return trades

    trades["scenario_id"] = scenario["scenario_id"]
    trades["scenario_label"] = scenario["scenario_label"]
    trades["filter_family"] = scenario["filter_family"]
    trades["filter_type"] = scenario["filter_type"]
    trades["low_threshold"] = scenario["low_threshold"]
    trades["high_threshold"] = scenario["high_threshold"]
    trades["edge_threshold"] = float(edge_threshold)
    trades["fee_rate"] = float(fee_rate)
    trades["fee_round_up_to_cent"] = bool(round_up_to_cent)
    trades["strategy_id"] = strategy_id
    trades["strategy_label"] = strategy_label
    trades["strategy_probability_column"] = probability_col
    trades["favorable_edge_after_fees"] = trades["favorable_edge_gross"] - trades["fee_per_contract"]
    trades["after_fee_edge_positive"] = trades["favorable_edge_after_fees"] > 0.0
    trades["expected_value_before_fees"] = trades["favorable_edge_gross"]
    trades["expected_value_after_fees"] = trades["favorable_edge_after_fees"]
    trades["cost_per_contract_after_fees"] = trades["cost_per_contract_before_fees"] + trades["fee_per_contract"]
    trades["realized_pnl_before_fees"] = trades["gross_payout_per_contract"] - trades["cost_per_contract_before_fees"]
    trades["realized_pnl_after_fees"] = trades["gross_payout_per_contract"] - trades["cost_per_contract_after_fees"]
    trades["result_label"] = np.where(trades["win"] == 1, "win", "loss")
    return trades.sort_values(["strategy_id", "forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"]).reset_index(
        drop=True
    )


def build_scenario_trades(
    eligible: pd.DataFrame,
    *,
    scenario: Dict[str, Any],
    edge_threshold: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    frames = [
        build_strategy_trades(
            eligible,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            scenario=scenario,
            edge_threshold=edge_threshold,
            fee_rate=fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        for strategy_id, strategy_label, probability_col in STRATEGIES
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    grouped = (
        trades.groupby(["scenario_id", "scenario_label", "strategy_id", "strategy_label", "forecast_datetime_utc"], as_index=False)
        .agg(
            period_pnl_after_fees=("realized_pnl_after_fees", "sum"),
            period_pnl_before_fees=("realized_pnl_before_fees", "sum"),
            period_expected_value_after_fees=("expected_value_after_fees", "sum"),
            period_expected_value_before_fees=("expected_value_before_fees", "sum"),
            period_cost_after_fees=("cost_per_contract_after_fees", "sum"),
            period_cost_before_fees=("cost_per_contract_before_fees", "sum"),
            period_fees=("fee_per_contract", "sum"),
            period_trades=("event_contract_id", "size"),
            period_wins=("win", "sum"),
            period_after_fee_edge_positive=("after_fee_edge_positive", "sum"),
            mean_p_model_k=("p_model_k", "mean"),
        )
        .sort_values(["scenario_id", "strategy_id", "forecast_datetime_utc"])
        .reset_index(drop=True)
    )
    keys = ["scenario_id", "strategy_id"]
    grouped["cumulative_pnl_after_fees"] = grouped.groupby(keys)["period_pnl_after_fees"].cumsum()
    grouped["cumulative_pnl_before_fees"] = grouped.groupby(keys)["period_pnl_before_fees"].cumsum()
    grouped["cumulative_expected_value_after_fees"] = grouped.groupby(keys)["period_expected_value_after_fees"].cumsum()
    grouped["cumulative_expected_value_before_fees"] = grouped.groupby(keys)["period_expected_value_before_fees"].cumsum()
    grouped["cumulative_cost_after_fees"] = grouped.groupby(keys)["period_cost_after_fees"].cumsum()
    grouped["cumulative_cost_before_fees"] = grouped.groupby(keys)["period_cost_before_fees"].cumsum()
    grouped["cumulative_fees"] = grouped.groupby(keys)["period_fees"].cumsum()
    grouped["cumulative_trades"] = grouped.groupby(keys)["period_trades"].cumsum()
    grouped["cumulative_wins"] = grouped.groupby(keys)["period_wins"].cumsum()
    grouped["cumulative_after_fee_edge_positive"] = grouped.groupby(keys)["period_after_fee_edge_positive"].cumsum()
    grouped["cumulative_winrate"] = grouped["cumulative_wins"] / grouped["cumulative_trades"]
    grouped["cumulative_after_fee_edge_positive_rate"] = (
        grouped["cumulative_after_fee_edge_positive"] / grouped["cumulative_trades"]
    )
    grouped["cumulative_roi_after_fees"] = grouped["cumulative_pnl_after_fees"] / grouped[
        "cumulative_cost_after_fees"
    ].replace(0, np.nan)
    running_high = grouped.groupby(keys)["cumulative_pnl_after_fees"].cummax()
    grouped["running_pnl_high_water_mark"] = np.maximum(running_high, 0.0)
    grouped["pnl_drawdown_after_fees"] = grouped["running_pnl_high_water_mark"] - grouped["cumulative_pnl_after_fees"]
    return grouped


def summarize_strategy(
    *,
    scenario: Dict[str, Any],
    eligible: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_id: str,
    strategy_label: str,
    timeseries: pd.DataFrame,
    overlap_rows: int,
    overlap_timestamps: int,
) -> Dict[str, Any]:
    part = trades[(trades["scenario_id"] == scenario["scenario_id"]) & (trades["strategy_id"] == strategy_id)]
    ts = timeseries[(timeseries["scenario_id"] == scenario["scenario_id"]) & (timeseries["strategy_id"] == strategy_id)]
    signal_rows = int(len(part))
    signal_timestamps = int(part["forecast_datetime_utc"].nunique()) if signal_rows else 0
    eligible_timestamps = int(eligible["forecast_datetime_utc"].nunique()) if len(eligible) else 0
    total_cost_after_fees = float(part["cost_per_contract_after_fees"].sum()) if signal_rows else 0.0
    total_pnl_after_fees = float(part["realized_pnl_after_fees"].sum()) if signal_rows else 0.0
    total_pnl_before_fees = float(part["realized_pnl_before_fees"].sum()) if signal_rows else 0.0
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_label": scenario["scenario_label"],
        "filter_family": scenario["filter_family"],
        "filter_type": scenario["filter_type"],
        "low_threshold": scenario["low_threshold"],
        "high_threshold": scenario["high_threshold"],
        "strategy_id": strategy_id,
        "strategy_label": strategy_label,
        "overlap_rows": int(overlap_rows),
        "overlap_timestamps": int(overlap_timestamps),
        "eligible_rows": int(len(eligible)),
        "eligible_timestamps": eligible_timestamps,
        "eligible_row_share_of_baseline": len(eligible) / overlap_rows if overlap_rows else np.nan,
        "eligible_timestamp_share_of_baseline": eligible_timestamps / overlap_timestamps if overlap_timestamps else np.nan,
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "signal_row_share_of_baseline": signal_rows / overlap_rows if overlap_rows else np.nan,
        "winrate": float(part["win"].mean()) if signal_rows else np.nan,
        "average_p_model_k_yes_probability": float(part["p_model_k"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(part["model_probability_selected_side"].mean()) if signal_rows else np.nan,
        "average_model_k_selected_side_price": float(part["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_fee": float(part["fee_per_contract"].mean()) if signal_rows else np.nan,
        "average_gross_edge": float(part["favorable_edge_gross"].mean()) if signal_rows else np.nan,
        "average_after_fee_edge": float(part["favorable_edge_after_fees"].mean()) if signal_rows else np.nan,
        "after_fee_edge_positive_rate": float(part["after_fee_edge_positive"].mean()) if signal_rows else np.nan,
        "total_expected_value_before_fees": float(part["expected_value_before_fees"].sum()) if signal_rows else 0.0,
        "total_expected_value_after_fees": float(part["expected_value_after_fees"].sum()) if signal_rows else 0.0,
        "total_fees": float(part["fee_per_contract"].sum()) if signal_rows else 0.0,
        "total_cost_before_fees": float(part["cost_per_contract_before_fees"].sum()) if signal_rows else 0.0,
        "total_cost_after_fees": total_cost_after_fees,
        "total_pnl_before_fees": total_pnl_before_fees,
        "total_pnl_after_fees": total_pnl_after_fees,
        "fees_delta": total_pnl_after_fees - total_pnl_before_fees,
        "profitable_after_fees": bool(total_pnl_after_fees > 0.0),
        "average_pnl_after_fees": float(part["realized_pnl_after_fees"].mean()) if signal_rows else np.nan,
        "roi_after_fees": total_pnl_after_fees / total_cost_after_fees if total_cost_after_fees else np.nan,
        "best_period_pnl_after_fees": float(ts["period_pnl_after_fees"].max()) if not ts.empty else np.nan,
        "worst_period_pnl_after_fees": float(ts["period_pnl_after_fees"].min()) if not ts.empty else np.nan,
        "max_drawdown_after_fees": float(ts["pnl_drawdown_after_fees"].max()) if not ts.empty else np.nan,
        "final_cumulative_pnl_after_fees": float(ts["cumulative_pnl_after_fees"].iloc[-1]) if not ts.empty else np.nan,
    }


def build_summary(
    overlap: pd.DataFrame,
    scenario_eligible: Dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    timeseries: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for scenario in scenario_specs():
        eligible = scenario_eligible[scenario["scenario_id"]]
        for strategy_id, strategy_label, _probability_col in STRATEGIES:
            rows.append(
                summarize_strategy(
                    scenario=scenario,
                    eligible=eligible,
                    trades=trades,
                    strategy_id=strategy_id,
                    strategy_label=strategy_label,
                    timeseries=timeseries,
                    overlap_rows=overlap_rows,
                    overlap_timestamps=overlap_timestamps,
                )
            )
    summary = pd.DataFrame(rows)
    baseline = summary.loc[
        summary["scenario_id"] == "baseline_all",
        [
            "strategy_id",
            "signal_rows",
            "total_fees",
            "total_cost_after_fees",
            "total_pnl_after_fees",
            "roi_after_fees",
            "max_drawdown_after_fees",
        ],
    ].rename(
        columns={
            "signal_rows": "baseline_signal_rows",
            "total_fees": "baseline_total_fees",
            "total_cost_after_fees": "baseline_total_cost_after_fees",
            "total_pnl_after_fees": "baseline_total_pnl_after_fees",
            "roi_after_fees": "baseline_roi_after_fees",
            "max_drawdown_after_fees": "baseline_max_drawdown_after_fees",
        }
    )
    summary = summary.merge(baseline, on="strategy_id", how="left", validate="many_to_one")
    summary["delta_pnl_after_fees_vs_baseline"] = (
        summary["total_pnl_after_fees"] - summary["baseline_total_pnl_after_fees"]
    )
    summary["pnl_capture_vs_baseline"] = summary["total_pnl_after_fees"] / summary[
        "baseline_total_pnl_after_fees"
    ].replace(0, np.nan)
    summary["signal_row_reduction_vs_baseline"] = summary["baseline_signal_rows"] - summary["signal_rows"]
    summary["fee_reduction_vs_baseline"] = summary["baseline_total_fees"] - summary["total_fees"]
    summary["drawdown_delta_vs_baseline"] = (
        summary["max_drawdown_after_fees"] - summary["baseline_max_drawdown_after_fees"]
    )
    return summary


def build_side_summary(trades: pd.DataFrame, timeseries: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    scenarios = {spec["scenario_id"]: spec for spec in scenario_specs()}
    for (scenario_id, strategy_id, side), part in trades.groupby(["scenario_id", "strategy_id", "trade_side"], sort=False):
        strategy_label = str(part["strategy_label"].iloc[0])
        ts = build_pnl_timeseries(part)
        rows.append(
            summarize_strategy(
                scenario=scenarios[scenario_id],
                eligible=part,
                trades=part,
                strategy_id=strategy_id,
                strategy_label=strategy_label,
                timeseries=ts,
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
            )
            | {"trade_side": side}
        )
    return pd.DataFrame(rows)


def trade_output_columns(trades: pd.DataFrame) -> List[str]:
    cols = [
        "scenario_id",
        "scenario_label",
        "filter_family",
        "filter_type",
        "low_threshold",
        "high_threshold",
        "edge_threshold",
        "fee_rate",
        "fee_round_up_to_cent",
        "strategy_id",
        "strategy_label",
        "strategy_probability_column",
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
        "p_strategy",
        "p_exponential_hybrid",
        "p_normal",
        "p_model_k",
        "yes_edge",
        "no_edge",
        "favorable_edge_gross",
        "favorable_edge_after_fees",
        "after_fee_edge_positive",
        "model_probability_selected_side",
        "model_k_price_selected_side",
        "fee_per_contract",
        "expected_value_before_fees",
        "expected_value_after_fees",
        "cost_per_contract_before_fees",
        "cost_per_contract_after_fees",
        "gross_payout_per_contract",
        "realized_pnl_before_fees",
        "realized_pnl_after_fees",
        "result_label",
        "win",
        "official_result",
        "p_reality",
        "q_shock",
        "shock_weight_exponential",
        "model_n_source_file",
        "model_k_source_file",
    ]
    cols.extend([col for col in ["binance_audit_price", "binance_reference_price", "join_key_used"] if col in trades.columns])
    return [col for col in cols if col in trades.columns]


def format_number(value: Any, *, digits: int = 4, signed: bool = False, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if percent:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        prefix = "+" if signed and numeric_value >= 0 else ""
        return f"{prefix}{numeric_value * 100:,.2f}%"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        prefix = "+" if signed and float(value) >= 0 else ""
        return f"{prefix}{float(value):,.{digits}f}"
    return str(value)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for column in out.columns:
        lower = column.lower()
        if any(token in lower for token in ["rows", "timestamps", "trades", "wins"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=0))
        elif any(token in lower for token in ["share", "rate", "roi", "winrate", "capture"]):
            out[column] = out[column].map(lambda v: format_number(v, percent=True))
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability", "fee"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower or "delta" in lower))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def dashboard_style() -> str:
    return """
<style>
:root {
  color-scheme: dark;
  --bg: #0d1b26;
  --panel: #14283a;
  --panel-2: #193247;
  --text: #e8f1fa;
  --muted: #9fb4c7;
  --line: #35516c;
  --accent: #35c7b7;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1480px, calc(100% - 48px)); margin: 0 auto; padding: 30px 0 56px; }
.hero {
  padding: 28px 0 18px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 24px;
}
.eyebrow {
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0;
  font-size: 12px;
  margin: 0 0 8px;
}
h1 { font-size: clamp(30px, 4vw, 56px); line-height: 1; margin: 0 0 14px; letter-spacing: 0; }
h2 { font-size: 22px; margin: 34px 0 14px; letter-spacing: 0; }
.lead { max-width: 980px; color: var(--muted); line-height: 1.55; margin: 0; }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin-top: 22px;
}
.metric-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}
.metric-card span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.metric-card strong { display: block; font-size: 20px; line-height: 1.25; }
.panel, .chart-wrap {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 18px;
}
.chart-wrap { background: #0d1b26; padding: 0; border-color: transparent; }
.table-wrap { width: 100%; overflow: auto; }
table.data-table { border-collapse: collapse; width: 100%; min-width: 980px; font-size: 13px; }
table.data-table th {
  background: var(--panel-2);
  color: var(--text);
  text-align: left;
  font-weight: 700;
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
  position: sticky;
  top: 0;
}
table.data-table td {
  border-bottom: 1px solid rgba(53, 81, 108, 0.55);
  padding: 8px 10px;
  color: #dbe8f3;
  vertical-align: top;
}
code { color: #bceee8; }
@media (max-width: 760px) {
  main { width: min(100% - 24px, 1480px); padding-top: 20px; }
  .metric-card strong { font-size: 18px; }
}
</style>
"""


def plotly_theme(fig: Any, *, height: int, top_margin: int = 88) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1b26",
        plot_bgcolor="#0d1b26",
        font=dict(color="#e8f1fa"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.15,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=56, r=28, t=top_margin, b=52),
        height=height,
    )
    fig.update_xaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")
    fig.update_yaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")


def figures_to_html(figures: Sequence[Any]) -> str:
    if not HAS_PLOTLY:
        return "<section class='panel'><p>Plotly is not available in this environment.</p></section>"
    return "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, fig in enumerate(figures)
    )


def overview_figures(summary: pd.DataFrame) -> List[Any]:
    if not HAS_PLOTLY:
        return []

    figures: List[Any] = []
    scenario_order = [spec["scenario_label"] for spec in scenario_specs()]

    fig_pnl = go.Figure()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = summary[summary["strategy_label"] == strategy_label].set_index("scenario_label").reindex(scenario_order).reset_index()
        fig_pnl.add_trace(
            go.Bar(
                x=part["scenario_label"],
                y=part["total_pnl_after_fees"],
                name=strategy_label,
                marker_color=STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig_pnl.update_layout(title="Total Realized PnL After Taker Fees", barmode="group")
    fig_pnl.update_xaxes(title_text="Scenario")
    fig_pnl.update_yaxes(title_text="Total PnL after fees")
    plotly_theme(fig_pnl, height=620, top_margin=92)
    figures.append(fig_pnl)

    fig_delta = go.Figure()
    non_baseline = summary[summary["scenario_id"] != "baseline_all"].copy()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = non_baseline[non_baseline["strategy_label"] == strategy_label].copy()
        fig_delta.add_trace(
            go.Bar(
                x=part["scenario_label"],
                y=part["delta_pnl_after_fees_vs_baseline"],
                name=strategy_label,
                marker_color=STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig_delta.update_layout(title="After-Fee PnL Delta Vs Baseline")
    fig_delta.update_xaxes(title_text="Scenario")
    fig_delta.update_yaxes(title_text="Delta PnL after fees")
    plotly_theme(fig_delta, height=620, top_margin=92)
    figures.append(fig_delta)

    fig_quality = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total Fees Paid", "Trade Rows"),
        horizontal_spacing=0.14,
    )
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = summary[summary["strategy_label"] == strategy_label].set_index("scenario_label").reindex(scenario_order).reset_index()
        color = STRATEGY_COLORS.get(strategy_label)
        fig_quality.add_trace(go.Bar(x=part["scenario_label"], y=part["total_fees"], name=strategy_label, marker_color=color), row=1, col=1)
        fig_quality.add_trace(
            go.Bar(x=part["scenario_label"], y=part["signal_rows"], name=strategy_label, marker_color=color, showlegend=False),
            row=1,
            col=2,
        )
    fig_quality.update_layout(title="Fees And Trade Count", barmode="group")
    fig_quality.update_yaxes(title_text="Fees", row=1, col=1)
    fig_quality.update_yaxes(title_text="Rows", row=1, col=2)
    plotly_theme(fig_quality, height=620, top_margin=110)
    figures.append(fig_quality)

    return figures


def scenario_overlay_figure(timeseries: pd.DataFrame, *, scenario_label: str) -> Any:
    fig = go.Figure()
    scenario_ts = timeseries[timeseries["scenario_label"] == scenario_label].copy()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = scenario_ts[scenario_ts["strategy_label"] == strategy_label].sort_values("forecast_datetime_utc")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["forecast_datetime_utc"],
                y=part["cumulative_pnl_after_fees"],
                mode="lines",
                name=strategy_label,
                line=dict(color=STRATEGY_COLORS.get(strategy_label), width=3),
            )
        )
    fig.update_layout(title=f"{scenario_label}: Cumulative PnL After Taker Fees")
    fig.update_xaxes(title_text="Forecast timestamp")
    fig.update_yaxes(title_text="Cumulative PnL after fees")
    plotly_theme(fig, height=560, top_margin=92)
    return fig


def scenario_bar_figure(summary: pd.DataFrame, *, scenario_label: str) -> Any:
    scenario = summary[summary["scenario_label"] == scenario_label].copy()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Net PnL", "Total Fees", "Trade Rows", "Max Drawdown"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    labels = scenario["strategy_label"]
    colors = [STRATEGY_COLORS.get(label, "#6cb6ff") for label in labels]
    fig.add_trace(go.Bar(x=labels, y=scenario["total_pnl_after_fees"], marker_color=colors), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=scenario["total_fees"], marker_color=colors), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=scenario["signal_rows"], marker_color=colors), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=scenario["max_drawdown_after_fees"], marker_color=colors), row=2, col=2)
    fig.update_yaxes(title_text="PnL", row=1, col=1)
    fig.update_yaxes(title_text="Fees", row=1, col=2)
    fig.update_yaxes(title_text="Rows", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=2)
    fig.update_layout(showlegend=False)
    plotly_theme(fig, height=760, top_margin=104)
    return fig


def dashboard_cards(summary: pd.DataFrame) -> str:
    best = summary.loc[summary["total_pnl_after_fees"].idxmax()]
    cards = [
        ("Best Scenario", str(best["scenario_label"])),
        ("Best Strategy", str(best["strategy_label"])),
        ("Best Net PnL", format_number(best["total_pnl_after_fees"], digits=3, signed=True)),
        ("Profitable Rows", format_number(summary["profitable_after_fees"].sum(), digits=0)),
        ("Taker Fee Rate", format_number(best["fee_rate"] if "fee_rate" in best else DEFAULT_TAKER_FEE_RATE, digits=4)),
        ("Strategies", format_number(summary["strategy_id"].nunique(), digits=0)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def scenario_cards(summary: pd.DataFrame, scenario_label: str) -> str:
    scenario = summary[summary["scenario_label"] == scenario_label].copy()
    best = scenario.loc[scenario["total_pnl_after_fees"].idxmax()]
    cards = [
        ("Best Strategy", str(best["strategy_label"])),
        ("Best Net PnL", format_number(best["total_pnl_after_fees"], digits=3, signed=True)),
        ("Best ROI", format_number(best["roi_after_fees"], percent=True)),
        ("Total Fees", format_number(scenario["total_fees"].sum(), digits=3)),
        ("Trade Rows", format_number(scenario["signal_rows"].sum(), digits=0)),
        ("Profitable?", "Yes" if bool(scenario["profitable_after_fees"].any()) else "No"),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def build_overview_dashboard(
    *,
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    fee_rate: float,
    round_up_to_cent: bool,
) -> str:
    figures = overview_figures(summary)
    for spec in scenario_specs():
        figures.append(scenario_overlay_figure(timeseries, scenario_label=spec["scenario_label"]))
    slim_cols = [
        "scenario_label",
        "strategy_label",
        "signal_rows",
        "winrate",
        "average_model_k_selected_side_price",
        "average_fee",
        "average_after_fee_edge",
        "total_fees",
        "total_pnl_after_fees",
        "roi_after_fees",
        "max_drawdown_after_fees",
        "profitable_after_fees",
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model N Taker-Fee End Filter Experiment</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model N Experiment</p>
    <h1>Taker-Fee Lower-End Filters</h1>
    <p class="lead">
      Compares Model A Normal and Model N exponential hybrid lambda=2 when trades are limited to
      low or high Model K/Kalshi YES probabilities. A trade is entered when the strategy has a
      favorable gross directional edge; realized PnL subtracts taker entry fees using
      <code>{fee_rate} * selected_price * (1 - selected_price)</code>
      {"with cent round-up" if round_up_to_cent else "without cent round-up"}.
    </p>
    <div class="cards">{dashboard_cards(summary)}</div>
  </section>

  <h2>Overview Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Core Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(summary[slim_cols]))}</div></section>

  <h2>Full Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(summary))}</div></section>
</main>
</body>
</html>
"""


def build_scenario_dashboard(
    *,
    scenario_label: str,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    fee_rate: float,
    round_up_to_cent: bool,
) -> str:
    scenario_summary = summary[summary["scenario_label"] == scenario_label].copy()
    scenario_side = side_summary[side_summary["scenario_label"] == scenario_label].copy() if not side_summary.empty else pd.DataFrame()
    figures = [scenario_overlay_figure(timeseries, scenario_label=scenario_label), scenario_bar_figure(summary, scenario_label=scenario_label)]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model N Taker-Fee End Filter - {html.escape(scenario_label)}</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Taker-Fee End Filter</p>
    <h1>{html.escape(scenario_label)}</h1>
    <p class="lead">
      This scenario uses favorable-direction gross-edge entries and subtracts taker fees from
      realized PnL. Fee rate is <code>{fee_rate}</code>; cent round-up is
      <code>{round_up_to_cent}</code>.
    </p>
    <div class="cards">{scenario_cards(summary, scenario_label)}</div>
  </section>

  <h2>PnL Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(scenario_summary))}</div></section>

  <h2>Side Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(scenario_side))}</div></section>
</main>
</body>
</html>
"""


def write_readme(output_dir: Path, *, fee_rate: float, round_up_to_cent: bool, edge_threshold: float) -> None:
    text = f"""Model N Taker-Fee End Filter Experiment

Purpose:
- Test whether trading only the lower-fee ends remains profitable after Kalshi taker fees.
- Strategies are limited to Model A Normal and Model N exponential hybrid lambda=2.

Signal rule:
- First apply the scenario's Model K/Kalshi YES probability filter.
- BUY_YES when p_strategy - p_model_k > {edge_threshold}.
- BUY_NO when p_model_k - p_strategy > {edge_threshold}.
- This experiment does not require after-fee edge to be positive for entry; it measures the after-fee PnL of the filtered trade set.

Taker fee model:
- Entry fee per one $1 payout contract = fee_rate * selected_side_price * (1 - selected_side_price)
- fee_rate = {fee_rate}
- Cent round-up enabled = {round_up_to_cent}
- BUY_YES selected_side_price = p_model_k.
- BUY_NO selected_side_price = 1 - p_model_k.
- No exit or settlement fees are modeled.

Scenarios:
- baseline_all
- p_kalshi_lt_0_25
- p_kalshi_gt_0_75
- p_kalshi_outside_0_26_0_75
- p_kalshi_lt_0_10
- p_kalshi_gt_0_90
- p_kalshi_outside_0_10_0_90

Outputs:
- taker_fee_end_filter_strategy_summary.csv
- taker_fee_end_filter_strategy_side_summary.csv
- taker_fee_end_filter_pnl_timeseries.csv
- taker_fee_end_filter_trade_minutes.csv
- taker_fee_end_filter_dashboard.html
- one folder per scenario with per-scenario CSVs and dashboard.html
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fee_rate < 0.0:
        raise ValueError("--fee-rate must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round_up_to_cent = not args.no_fee_round_up

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = build_overlap(model_n, model_k)

    scenario_eligible: Dict[str, pd.DataFrame] = {}
    scenario_frames: List[pd.DataFrame] = []
    for scenario in scenario_specs():
        eligible = apply_scenario_filter(overlap, scenario)
        scenario_eligible[scenario["scenario_id"]] = eligible
        trades = build_scenario_trades(
            eligible,
            scenario=scenario,
            edge_threshold=args.edge_threshold,
            fee_rate=args.fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        scenario_frames.append(trades)

    all_trades = pd.concat([frame for frame in scenario_frames if not frame.empty], ignore_index=True)
    timeseries = build_pnl_timeseries(all_trades)
    summary = build_summary(overlap, scenario_eligible, all_trades, timeseries)
    summary["fee_rate"] = float(args.fee_rate)
    summary["fee_round_up_to_cent"] = bool(round_up_to_cent)
    side_summary = build_side_summary(all_trades, timeseries, overlap)

    summary.to_csv(output_dir / "taker_fee_end_filter_strategy_summary.csv", index=False)
    side_summary.to_csv(output_dir / "taker_fee_end_filter_strategy_side_summary.csv", index=False)
    timeseries.to_csv(output_dir / "taker_fee_end_filter_pnl_timeseries.csv", index=False)
    all_trades[trade_output_columns(all_trades)].to_csv(output_dir / "taker_fee_end_filter_trade_minutes.csv", index=False)

    for scenario in scenario_specs():
        scenario_dir = output_dir / scenario["scenario_id"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_trades = all_trades[all_trades["scenario_id"] == scenario["scenario_id"]].copy()
        scenario_ts = timeseries[timeseries["scenario_id"] == scenario["scenario_id"]].copy()
        scenario_summary = summary[summary["scenario_id"] == scenario["scenario_id"]].copy()
        scenario_side_summary = side_summary[side_summary["scenario_id"] == scenario["scenario_id"]].copy()

        scenario_trades[trade_output_columns(scenario_trades)].to_csv(scenario_dir / "trade_minutes.csv", index=False)
        scenario_ts.to_csv(scenario_dir / "pnl_timeseries.csv", index=False)
        scenario_summary.to_csv(scenario_dir / "strategy_summary.csv", index=False)
        scenario_side_summary.to_csv(scenario_dir / "strategy_side_summary.csv", index=False)
        scenario_html = build_scenario_dashboard(
            scenario_label=scenario["scenario_label"],
            summary=summary,
            side_summary=side_summary,
            timeseries=timeseries,
            fee_rate=args.fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        (scenario_dir / "dashboard.html").write_text(scenario_html, encoding="utf-8")

    overview_html = build_overview_dashboard(
        summary=summary,
        timeseries=timeseries,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
    )
    (output_dir / "taker_fee_end_filter_dashboard.html").write_text(overview_html, encoding="utf-8")
    write_readme(output_dir, fee_rate=args.fee_rate, round_up_to_cent=round_up_to_cent, edge_threshold=args.edge_threshold)

    display_cols = [
        "scenario_label",
        "strategy_label",
        "signal_rows",
        "total_fees",
        "total_pnl_before_fees",
        "total_pnl_after_fees",
        "roi_after_fees",
        "profitable_after_fees",
    ]
    print(f"Taker-fee end-filter output directory: {output_dir}")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
