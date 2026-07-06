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
STRATEGIES: Tuple[Tuple[str, str, str], ...] = (
    ("model_n_linear", "Model N Linear", "p_model_n"),
    ("model_n_exp_lambda_2", "Model N Exp lambda=2", "p_exponential_hybrid"),
    ("model_a_normal", "Model A Normal", "p_normal"),
    ("model_b_shock", "Model B Shock", "p_shock"),
)
STRATEGY_COLORS = {
    "Model N Linear": "#6cb6ff",
    "Model N Exp lambda=2": "#fbbf24",
    "Model A Normal": "#35c7b7",
    "Model B Shock": "#ff8f70",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Model N market-price filter experiment. Compares baseline favorable-direction "
            "PnL to p_model_k low/high/outside filters for all Model N strategy columns."
        )
    )
    parser.add_argument(
        "--model-n-raw-values",
        type=Path,
        default=root / "Model_N" / "model_N_Evals_Outputs" / "raw_values.csv",
        help="Model N raw_values.csv containing p_normal, p_shock, p_exponential_hybrid, and outcomes.",
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
        help="Minimum favorable-direction gross edge required for a trade.",
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
        "p_shock",
        "p_exponential_hybrid",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def build_overlap(model_n: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    required_model_n = {"p_normal", "p_shock", "p_exponential_hybrid"}
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
        "p_shock",
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
            "low_threshold": np.nan,
            "high_threshold": np.nan,
            "filter_type": "all",
        },
        {
            "scenario_id": "p_kalshi_lt_0_25",
            "scenario_label": "p_model_k < 0.25",
            "filter_family": "0.25/0.75",
            "low_threshold": 0.25,
            "high_threshold": 0.75,
            "filter_type": "low",
        },
        {
            "scenario_id": "p_kalshi_gt_0_75",
            "scenario_label": "p_model_k > 0.75",
            "filter_family": "0.25/0.75",
            "low_threshold": 0.25,
            "high_threshold": 0.75,
            "filter_type": "high",
        },
        {
            "scenario_id": "p_kalshi_outside_0_25_0_75",
            "scenario_label": "p_model_k < 0.25 OR > 0.75",
            "filter_family": "0.25/0.75",
            "low_threshold": 0.25,
            "high_threshold": 0.75,
            "filter_type": "outside",
        },
        {
            "scenario_id": "p_kalshi_lt_0_15",
            "scenario_label": "p_model_k < 0.15",
            "filter_family": "0.15/0.85",
            "low_threshold": 0.15,
            "high_threshold": 0.85,
            "filter_type": "low",
        },
        {
            "scenario_id": "p_kalshi_gt_0_85",
            "scenario_label": "p_model_k > 0.85",
            "filter_family": "0.15/0.85",
            "low_threshold": 0.15,
            "high_threshold": 0.85,
            "filter_type": "high",
        },
        {
            "scenario_id": "p_kalshi_outside_0_15_0_85",
            "scenario_label": "p_model_k < 0.15 OR > 0.85",
            "filter_family": "0.15/0.85",
            "low_threshold": 0.15,
            "high_threshold": 0.85,
            "filter_type": "outside",
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


def build_strategy_trades(
    eligible: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    scenario: Dict[str, Any],
    edge_threshold: float,
) -> pd.DataFrame:
    if probability_col not in eligible.columns:
        raise ValueError(f"Cannot build {strategy_label}; missing {probability_col}.")

    work = eligible.dropna(subset=[probability_col, "p_model_k", "p_reality"]).copy()
    work["p_strategy"] = pd.to_numeric(work[probability_col], errors="coerce").clip(0.0, 1.0)
    work["yes_edge"] = work["p_strategy"] - work["p_model_k"]
    work["no_edge"] = (1.0 - work["p_strategy"]) - (1.0 - work["p_model_k"])

    buy_yes = work[work["yes_edge"] > edge_threshold].copy()
    buy_yes["trade_side"] = "BUY_YES"
    buy_yes["favorable_edge"] = buy_yes["yes_edge"]
    buy_yes["model_probability_selected_side"] = buy_yes["p_strategy"]
    buy_yes["model_k_price_selected_side"] = buy_yes["p_model_k"]
    buy_yes["cost_per_contract"] = buy_yes["p_model_k"]
    buy_yes["gross_payout_per_contract"] = buy_yes["p_reality"]
    buy_yes["win"] = (buy_yes["p_reality"] == 1.0).astype(int)

    buy_no = work[work["no_edge"] > edge_threshold].copy()
    buy_no["trade_side"] = "BUY_NO"
    buy_no["favorable_edge"] = buy_no["no_edge"]
    buy_no["model_probability_selected_side"] = 1.0 - buy_no["p_strategy"]
    buy_no["model_k_price_selected_side"] = 1.0 - buy_no["p_model_k"]
    buy_no["cost_per_contract"] = 1.0 - buy_no["p_model_k"]
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
    trades["strategy_id"] = strategy_id
    trades["strategy_label"] = strategy_label
    trades["strategy_probability_column"] = probability_col
    trades["expected_value_per_contract"] = trades["favorable_edge"]
    trades["realized_pnl_per_contract"] = trades["gross_payout_per_contract"] - trades["cost_per_contract"]
    trades["result_label"] = np.where(trades["win"] == 1, "win", "loss")
    return trades.sort_values(["strategy_id", "forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"])


def build_scenario_trades(eligible: pd.DataFrame, *, scenario: Dict[str, Any], edge_threshold: float) -> pd.DataFrame:
    frames = [
        build_strategy_trades(
            eligible,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            scenario=scenario,
            edge_threshold=edge_threshold,
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
            period_realized_pnl=("realized_pnl_per_contract", "sum"),
            period_expected_value=("expected_value_per_contract", "sum"),
            period_cost=("cost_per_contract", "sum"),
            period_trades=("event_contract_id", "size"),
            period_wins=("win", "sum"),
            mean_p_model_k=("p_model_k", "mean"),
        )
        .sort_values(["scenario_id", "strategy_id", "forecast_datetime_utc"])
        .reset_index(drop=True)
    )
    keys = ["scenario_id", "strategy_id"]
    grouped["cumulative_realized_pnl"] = grouped.groupby(keys)["period_realized_pnl"].cumsum()
    grouped["cumulative_expected_value"] = grouped.groupby(keys)["period_expected_value"].cumsum()
    grouped["cumulative_cost"] = grouped.groupby(keys)["period_cost"].cumsum()
    grouped["cumulative_trades"] = grouped.groupby(keys)["period_trades"].cumsum()
    grouped["cumulative_wins"] = grouped.groupby(keys)["period_wins"].cumsum()
    grouped["cumulative_winrate"] = grouped["cumulative_wins"] / grouped["cumulative_trades"]
    grouped["cumulative_roi_on_cost"] = grouped["cumulative_realized_pnl"] / grouped["cumulative_cost"].replace(0, np.nan)
    grouped["running_pnl_high_water_mark"] = grouped.groupby(keys)["cumulative_realized_pnl"].cummax()
    grouped["pnl_drawdown"] = grouped["running_pnl_high_water_mark"] - grouped["cumulative_realized_pnl"]
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
    total_cost = float(part["cost_per_contract"].sum()) if signal_rows else 0.0
    total_pnl = float(part["realized_pnl_per_contract"].sum()) if signal_rows else 0.0
    total_ev = float(part["expected_value_per_contract"].sum()) if signal_rows else 0.0
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
        "eligible_timestamps": int(eligible["forecast_datetime_utc"].nunique()) if len(eligible) else 0,
        "eligible_row_share_of_baseline": len(eligible) / overlap_rows if overlap_rows else np.nan,
        "eligible_timestamp_share_of_baseline": eligible["forecast_datetime_utc"].nunique() / overlap_timestamps
        if overlap_timestamps and len(eligible)
        else np.nan,
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "signal_row_share_of_baseline": signal_rows / overlap_rows if overlap_rows else np.nan,
        "winrate": float(part["win"].mean()) if signal_rows else np.nan,
        "average_p_model_k": float(part["p_model_k"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(part["model_probability_selected_side"].mean()) if signal_rows else np.nan,
        "average_model_k_selected_side_price": float(part["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_edge": float(part["favorable_edge"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "total_realized_pnl": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
        "best_period_pnl": float(ts["period_realized_pnl"].max()) if not ts.empty else np.nan,
        "worst_period_pnl": float(ts["period_realized_pnl"].min()) if not ts.empty else np.nan,
        "max_drawdown": float(ts["pnl_drawdown"].max()) if not ts.empty else np.nan,
        "final_cumulative_pnl": float(ts["cumulative_realized_pnl"].iloc[-1]) if not ts.empty else np.nan,
    }


def build_summary(overlap: pd.DataFrame, scenario_eligible: Dict[str, pd.DataFrame], trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    scenarios = scenario_specs()
    for scenario in scenarios:
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
        ["strategy_id", "total_realized_pnl", "total_cost", "signal_rows", "max_drawdown"],
    ].rename(
        columns={
            "total_realized_pnl": "baseline_total_realized_pnl",
            "total_cost": "baseline_total_cost",
            "signal_rows": "baseline_signal_rows",
            "max_drawdown": "baseline_max_drawdown",
        }
    )
    summary = summary.merge(baseline, on="strategy_id", how="left", validate="many_to_one")
    summary["delta_pnl_vs_baseline"] = summary["total_realized_pnl"] - summary["baseline_total_realized_pnl"]
    summary["pnl_capture_vs_baseline"] = summary["total_realized_pnl"] / summary["baseline_total_realized_pnl"].replace(0, np.nan)
    summary["signal_row_reduction_vs_baseline"] = summary["baseline_signal_rows"] - summary["signal_rows"]
    summary["drawdown_delta_vs_baseline"] = summary["max_drawdown"] - summary["baseline_max_drawdown"]
    return summary


def build_side_summary(trades: pd.DataFrame, timeseries: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for (scenario_id, strategy_id, side), part in trades.groupby(["scenario_id", "strategy_id", "trade_side"], sort=False):
        scenario = next(spec for spec in scenario_specs() if spec["scenario_id"] == scenario_id)
        strategy_label = str(part["strategy_label"].iloc[0])
        ts = build_pnl_timeseries(part)
        rows.append(
            summarize_strategy(
                scenario=scenario,
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
        "p_model_n",
        "p_exponential_hybrid",
        "p_normal",
        "p_shock",
        "p_model_k",
        "yes_edge",
        "no_edge",
        "favorable_edge",
        "model_probability_selected_side",
        "model_k_price_selected_side",
        "expected_value_per_contract",
        "cost_per_contract",
        "gross_payout_per_contract",
        "realized_pnl_per_contract",
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
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower or "delta" in lower))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def plotly_theme(fig: Any, *, height: int, top_margin: int = 86) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1b26",
        plot_bgcolor="#0d1b26",
        font=dict(color="#e8f1fa"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.16,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=56, r=28, t=top_margin, b=48),
        height=height,
    )
    fig.update_xaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")
    fig.update_yaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")


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
                y=part["total_realized_pnl"],
                name=strategy_label,
                marker_color=STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig_pnl.update_layout(title="Total Realized PnL By Market-Price Filter", barmode="group")
    fig_pnl.update_xaxes(title_text="Scenario")
    fig_pnl.update_yaxes(title_text="Total realized PnL")
    plotly_theme(fig_pnl, height=620, top_margin=92)
    figures.append(fig_pnl)

    non_baseline = summary[summary["scenario_id"] != "baseline_all"].copy()
    fig_delta = go.Figure()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = non_baseline[non_baseline["strategy_label"] == strategy_label].copy()
        fig_delta.add_trace(
            go.Bar(
                x=part["scenario_label"],
                y=part["delta_pnl_vs_baseline"],
                name=strategy_label,
                marker_color=STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig_delta.update_layout(title="PnL Delta Vs Baseline")
    fig_delta.update_xaxes(title_text="Filter scenario")
    fig_delta.update_yaxes(title_text="Delta PnL")
    plotly_theme(fig_delta, height=620, top_margin=92)
    figures.append(fig_delta)

    fig_rows = go.Figure()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = summary[summary["strategy_label"] == strategy_label].set_index("scenario_label").reindex(scenario_order).reset_index()
        fig_rows.add_trace(
            go.Bar(
                x=part["scenario_label"],
                y=part["signal_rows"],
                name=strategy_label,
                marker_color=STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig_rows.update_layout(title="Trade Rows By Scenario", barmode="group")
    fig_rows.update_xaxes(title_text="Scenario")
    fig_rows.update_yaxes(title_text="Trade rows")
    plotly_theme(fig_rows, height=540, top_margin=92)
    figures.append(fig_rows)
    return figures


def scenario_figures(timeseries: pd.DataFrame, summary: pd.DataFrame, *, scenario_id: str) -> List[Any]:
    if not HAS_PLOTLY:
        return []
    scenario_summary = summary[summary["scenario_id"] == scenario_id].copy()
    scenario_label = str(scenario_summary["scenario_label"].iloc[0])
    scenario_ts = timeseries[timeseries["scenario_id"] == scenario_id].copy()

    fig_overlay = go.Figure()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = scenario_ts[scenario_ts["strategy_label"] == strategy_label].sort_values("forecast_datetime_utc")
        if part.empty:
            continue
        fig_overlay.add_trace(
            go.Scatter(
                x=part["forecast_datetime_utc"],
                y=part["cumulative_realized_pnl"],
                mode="lines",
                name=strategy_label,
                line=dict(color=STRATEGY_COLORS.get(strategy_label), width=3),
            )
        )
    fig_overlay.update_layout(title=f"{scenario_label}: Cumulative Realized PnL")
    fig_overlay.update_xaxes(title_text="Forecast timestamp")
    fig_overlay.update_yaxes(title_text="Cumulative PnL")
    plotly_theme(fig_overlay, height=540, top_margin=92)

    fig_bars = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Total PnL", "Delta Vs Baseline", "Trade Rows", "Max Drawdown"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    labels = scenario_summary["strategy_label"]
    fig_bars.add_trace(go.Bar(x=labels, y=scenario_summary["total_realized_pnl"], marker_color="#6cb6ff"), row=1, col=1)
    fig_bars.add_trace(go.Bar(x=labels, y=scenario_summary["delta_pnl_vs_baseline"], marker_color="#ff8f70"), row=1, col=2)
    fig_bars.add_trace(go.Bar(x=labels, y=scenario_summary["signal_rows"], marker_color="#35c7b7"), row=2, col=1)
    fig_bars.add_trace(go.Bar(x=labels, y=scenario_summary["max_drawdown"], marker_color="#fbbf24"), row=2, col=2)
    fig_bars.update_yaxes(title_text="PnL", row=1, col=1)
    fig_bars.update_yaxes(title_text="Delta", row=1, col=2)
    fig_bars.update_yaxes(title_text="Rows", row=2, col=1)
    fig_bars.update_yaxes(title_text="Drawdown", row=2, col=2)
    plotly_theme(fig_bars, height=740, top_margin=96)
    return [fig_overlay, fig_bars]


def figures_to_html(figures: Sequence[Any]) -> str:
    if not HAS_PLOTLY:
        return "<section class='panel'><p>Plotly is not available in this environment.</p></section>"
    return "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, fig in enumerate(figures)
    )


def dashboard_style() -> str:
    return """<style>
    :root {
      color-scheme: dark;
      --bg: #07131d;
      --panel: #10202d;
      --panel-2: #142736;
      --line: #22394b;
      --ink: #e8f1fa;
      --muted: #96aabd;
    }
    body {
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #0b1823 0%, var(--bg) 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1260px;
      margin: 0 auto;
      padding: 34px 24px 54px;
    }
    h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: 0;
    }
    h2 {
      margin: 36px 0 12px;
      font-size: 18px;
    }
    p, li {
      color: var(--muted);
      line-height: 1.5;
    }
    .hero, .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 20px;
    }
    .eyebrow {
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #a9c7e2;
    }
    .lead {
      max-width: 980px;
      margin: 14px 0 0;
      font-size: 15px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px 16px;
      background: var(--panel-2);
    }
    .metric-card span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .metric-card strong {
      display: block;
      font-size: 28px;
      line-height: 1.05;
    }
    .chart-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0d1b26;
      overflow: hidden;
      margin-bottom: 18px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.01);
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }
    .data-table th,
    .data-table td {
      padding: 10px 11px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }
    .data-table th {
      background: var(--panel-2);
      color: var(--ink);
      font-weight: 650;
    }
    code {
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 5px;
      color: var(--ink);
    }
  </style>"""


def summary_cards(summary: pd.DataFrame) -> str:
    non_baseline = summary[summary["scenario_id"] != "baseline_all"].copy()
    best = non_baseline.loc[non_baseline["total_realized_pnl"].idxmax()]
    best_delta = non_baseline.loc[non_baseline["delta_pnl_vs_baseline"].idxmax()]
    baseline_best = summary[summary["scenario_id"] == "baseline_all"].loc[
        summary[summary["scenario_id"] == "baseline_all"]["total_realized_pnl"].idxmax()
    ]
    cards = [
        ("Best Filter PnL", f"{best['strategy_label']} | {best['scenario_label']}"),
        ("Best Filter PnL Value", format_number(best["total_realized_pnl"], digits=3, signed=True)),
        ("Best Delta", f"{best_delta['strategy_label']} | {best_delta['scenario_label']}"),
        ("Best Delta Value", format_number(best_delta["delta_pnl_vs_baseline"], digits=3, signed=True)),
        ("Best Baseline", str(baseline_best["strategy_label"])),
        ("Best Baseline PnL", format_number(baseline_best["total_realized_pnl"], digits=3, signed=True)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def build_overview_dashboard(summary: pd.DataFrame, timeseries: pd.DataFrame, *, edge_threshold: float) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model N Market Price Filter Experiment</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Standalone Market-Price Filter Experiment</p>
    <h1>Model N Strategy PnL By Model K Price Filter</h1>
    <p class="lead">
      This experiment uses <code>p_model_k</code> as the Kalshi/market YES probability.
      Baseline trades the full Model K overlap universe. Variants only allow rows where
      <code>p_model_k</code> is below, above, or outside the requested threshold bands.
      Trade direction still uses the favorable selected-side model edge with threshold
      <code>{edge_threshold}</code>. Fees are not applied in this experiment.
    </p>
    <div class="cards">{summary_cards(summary)}</div>
  </section>

  <h2>Overview Charts</h2>
  <section class="chart-wrap">{figures_to_html(overview_figures(summary))}</section>

  <h2>Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(summary))}</div></section>
</main>
</body>
</html>
"""


def build_scenario_dashboard(summary: pd.DataFrame, side_summary: pd.DataFrame, timeseries: pd.DataFrame, *, scenario_id: str) -> str:
    scenario_summary = summary[summary["scenario_id"] == scenario_id].copy()
    scenario_side = side_summary[side_summary["scenario_id"] == scenario_id].copy()
    scenario_label = str(scenario_summary["scenario_label"].iloc[0])
    best = scenario_summary.loc[scenario_summary["total_realized_pnl"].idxmax()]
    cards = [
        ("Best Strategy", str(best["strategy_label"])),
        ("Best PnL", format_number(best["total_realized_pnl"], digits=3, signed=True)),
        ("Best Delta", format_number(best["delta_pnl_vs_baseline"], digits=3, signed=True)),
        ("Eligible Rows", format_number(best["eligible_rows"], digits=0)),
        ("Trade Rows", format_number(scenario_summary["signal_rows"].sum(), digits=0)),
        ("Max DD", format_number(best["max_drawdown"], digits=3)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(scenario_label)} - Market Price Filter</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Market-Price Filter Scenario</p>
    <h1>{html.escape(scenario_label)}</h1>
    <p class="lead">
      Uses the same favorable-direction non-fee PnL rule. The filter applies to the Model K YES probability
      <code>p_model_k</code> before trade direction is selected.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>PnL Charts</h2>
  <section class="chart-wrap">{figures_to_html(scenario_figures(timeseries, summary, scenario_id=scenario_id))}</section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(scenario_summary))}</div></section>

  <h2>Side Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(scenario_side))}</div></section>
</main>
</body>
</html>
"""


def write_readme(output_dir: Path, *, edge_threshold: float) -> None:
    text = f"""Model N Market Price Filter Experiment

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Market probability:
- p(kalshi) is modeled as Model K's YES probability column, p_model_k.

Strategies:
- Model N Linear: p_model_n
- Model N Exp lambda=2: p_exponential_hybrid
- Model A Normal: p_normal
- Model B Shock: p_shock

Trade rule:
- BUY_YES when p_strategy - p_model_k > {edge_threshold}
- BUY_NO when (1 - p_strategy) - (1 - p_model_k) > {edge_threshold}
- Fees are not applied in this experiment.

Scenarios:
- Baseline: all Model K overlap rows
- p_model_k < 0.25
- p_model_k > 0.75
- p_model_k < 0.25 OR > 0.75
- p_model_k < 0.15
- p_model_k > 0.85
- p_model_k < 0.15 OR > 0.85

Outputs:
- market_price_filter_summary.csv
- market_price_filter_side_summary.csv
- market_price_filter_pnl_timeseries.csv
- market_price_filter_trade_minutes.csv
- market_price_filter_dashboard.html
- scenario folders with per-scenario summaries, PnL time series, trade minutes, and dashboards.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.edge_threshold < 0.0:
        raise ValueError("--edge-threshold must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = build_overlap(model_n, model_k)

    scenario_eligible: Dict[str, pd.DataFrame] = {}
    scenario_trades: List[pd.DataFrame] = []
    for scenario in scenario_specs():
        eligible = apply_scenario_filter(overlap, scenario).reset_index(drop=True)
        scenario_eligible[scenario["scenario_id"]] = eligible
        trades = build_scenario_trades(eligible, scenario=scenario, edge_threshold=args.edge_threshold)
        scenario_trades.append(trades)

    all_trades = pd.concat(scenario_trades, ignore_index=True)
    timeseries = build_pnl_timeseries(all_trades)
    summary = build_summary(overlap, scenario_eligible, all_trades, timeseries)
    side_summary = build_side_summary(all_trades, timeseries, overlap)

    summary.to_csv(output_dir / "market_price_filter_summary.csv", index=False)
    side_summary.to_csv(output_dir / "market_price_filter_side_summary.csv", index=False)
    timeseries.to_csv(output_dir / "market_price_filter_pnl_timeseries.csv", index=False)
    all_trades[trade_output_columns(all_trades)].to_csv(output_dir / "market_price_filter_trade_minutes.csv", index=False)

    for scenario in scenario_specs():
        scenario_id = scenario["scenario_id"]
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_summary = summary[summary["scenario_id"] == scenario_id].copy()
        scenario_side = side_summary[side_summary["scenario_id"] == scenario_id].copy()
        scenario_ts = timeseries[timeseries["scenario_id"] == scenario_id].copy()
        scenario_trade_frame = all_trades[all_trades["scenario_id"] == scenario_id].copy()

        scenario_summary.to_csv(scenario_dir / "strategy_summary.csv", index=False)
        scenario_side.to_csv(scenario_dir / "strategy_side_summary.csv", index=False)
        scenario_ts.to_csv(scenario_dir / "pnl_timeseries.csv", index=False)
        scenario_trade_frame[trade_output_columns(scenario_trade_frame)].to_csv(scenario_dir / "trade_minutes.csv", index=False)
        (scenario_dir / "dashboard.html").write_text(
            build_scenario_dashboard(summary, side_summary, timeseries, scenario_id=scenario_id),
            encoding="utf-8",
        )

    (output_dir / "market_price_filter_dashboard.html").write_text(
        build_overview_dashboard(summary, timeseries, edge_threshold=args.edge_threshold),
        encoding="utf-8",
    )
    write_readme(output_dir, edge_threshold=args.edge_threshold)

    print(f"Market-price filter output directory: {output_dir}")
    print(
        summary[
            [
                "scenario_label",
                "strategy_label",
                "eligible_rows",
                "signal_rows",
                "total_realized_pnl",
                "delta_pnl_vs_baseline",
                "roi_on_cost",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
