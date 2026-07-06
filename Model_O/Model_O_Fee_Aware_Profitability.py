from __future__ import annotations

import argparse
import html
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_profitability_module() -> Any:
    module_path = Path(__file__).resolve().parent / "Model_O_Profitability.py"
    spec = importlib.util.spec_from_file_location("model_o_profitability_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load profitability module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_profitability_module()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    go = None
    make_subplots = None
    HAS_PLOTLY = False


OUTPUT_FOLDER_NAME = "Model_O_Fee_Aware_Profitability_Outputs"
DEFAULT_FEE_RATE = 0.07
SCENARIOS = (
    ("gross_edge_fee_pnl", "Gross Edge > 0 | Fee PnL", False, 0.00),
    ("after_fee_edge_0", "After-Fee Edge > 0", True, 0.00),
    ("after_fee_edge_001", "After-Fee Edge > 0.01", True, 0.01),
    ("after_fee_edge_002", "After-Fee Edge > 0.02", True, 0.02),
)


def parse_args() -> argparse.Namespace:
    root = base.repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build fee-aware Model O profitability outputs across linear Model O, "
            "exponential Model O lambda=2, Model A, and Model B."
        )
    )
    parser.add_argument(
        "--model-o-raw-values",
        type=Path,
        default=root / "Model_O" / "model_O_Evals_Outputs" / "raw_values.csv",
        help="Model O eval raw_values.csv produced by Model_O_Eval.py.",
    )
    parser.add_argument(
        "--model-k-raw-values",
        type=Path,
        default=root / "Model_K" / "Model_K_outputs" / "raw_values.csv",
        help="Model K eval raw_values.csv used as the entry price/probability source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Directory where fee-aware profitability outputs will be written.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_FEE_RATE,
        help="Entry fee rate applied to price * (1 - price), before cent rounding.",
    )
    parser.add_argument(
        "--fee-model-label",
        default="Fee-Aware",
        help="Short label shown in dashboards/readmes, e.g. Taker Fee or Maker Fee.",
    )
    parser.add_argument(
        "--no-fee-round-up",
        action="store_true",
        help="Disable cent round-up for the fee model.",
    )
    parser.add_argument(
        "--write-trade-minutes",
        action="store_true",
        help="Write one combined all-fee-aware-trades CSV. Per-scenario trade CSVs are always written.",
    )
    return parser.parse_args()


def fee_per_contract(price: pd.Series, *, fee_rate: float, round_up_to_cent: bool) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").clip(0.0, 1.0)
    fee = fee_rate * price * (1.0 - price)
    if round_up_to_cent:
        fee = np.ceil((fee * 100.0) - 1e-12) / 100.0
    return pd.Series(fee, index=price.index).clip(lower=0.0)


def signal_mask(work: pd.DataFrame, *, use_fee_for_signal: bool, required_edge: float, side: str) -> pd.Series:
    if side == "YES":
        gross_edge = work["yes_edge"]
        fee = work["yes_fee_per_contract"]
    elif side == "NO":
        gross_edge = work["no_edge"]
        fee = work["no_fee_per_contract"]
    else:
        raise ValueError(f"Unknown side: {side}")

    hurdle = required_edge + (fee if use_fee_for_signal else 0.0)
    return gross_edge > hurdle


def build_fee_aware_strategy_trades(
    overlap: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    scenario_id: str,
    scenario_label: str,
    use_fee_for_signal: bool,
    required_edge: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    if probability_col not in overlap.columns:
        raise ValueError(f"Cannot build {strategy_label}; missing {probability_col}.")

    work = overlap.dropna(subset=[probability_col, "p_model_k", "p_reality"]).copy()
    work["p_strategy"] = pd.to_numeric(work[probability_col], errors="coerce").clip(0.0, 1.0)
    work["yes_edge"] = work["p_strategy"] - work["p_model_k"]
    work["no_edge"] = (1.0 - work["p_strategy"]) - (1.0 - work["p_model_k"])
    work["yes_fee_per_contract"] = fee_per_contract(
        work["p_model_k"], fee_rate=fee_rate, round_up_to_cent=round_up_to_cent
    )
    work["no_fee_per_contract"] = fee_per_contract(
        1.0 - work["p_model_k"], fee_rate=fee_rate, round_up_to_cent=round_up_to_cent
    )

    buy_yes = work[signal_mask(work, use_fee_for_signal=use_fee_for_signal, required_edge=required_edge, side="YES")].copy()
    buy_yes["trade_side"] = "BUY_YES"
    buy_yes["favorable_edge_gross"] = buy_yes["yes_edge"]
    buy_yes["fee_per_contract"] = buy_yes["yes_fee_per_contract"]
    buy_yes["model_probability_selected_side"] = buy_yes["p_strategy"]
    buy_yes["model_k_price_selected_side"] = buy_yes["p_model_k"]
    buy_yes["cost_per_contract_before_fees"] = buy_yes["p_model_k"]
    buy_yes["gross_payout_per_contract"] = buy_yes["p_reality"]
    buy_yes["win"] = (buy_yes["p_reality"] == 1.0).astype(int)

    buy_no = work[signal_mask(work, use_fee_for_signal=use_fee_for_signal, required_edge=required_edge, side="NO")].copy()
    buy_no["trade_side"] = "BUY_NO"
    buy_no["favorable_edge_gross"] = buy_no["no_edge"]
    buy_no["fee_per_contract"] = buy_no["no_fee_per_contract"]
    buy_no["model_probability_selected_side"] = 1.0 - buy_no["p_strategy"]
    buy_no["model_k_price_selected_side"] = 1.0 - buy_no["p_model_k"]
    buy_no["cost_per_contract_before_fees"] = 1.0 - buy_no["p_model_k"]
    buy_no["gross_payout_per_contract"] = 1.0 - buy_no["p_reality"]
    buy_no["win"] = (buy_no["p_reality"] == 0.0).astype(int)

    trades = pd.concat([buy_yes, buy_no], ignore_index=True)
    if trades.empty:
        return trades

    trades["scenario_id"] = scenario_id
    trades["scenario_label"] = scenario_label
    trades["scenario_uses_fee_for_signal"] = bool(use_fee_for_signal)
    trades["required_edge_after_fee"] = float(required_edge)
    trades["fee_rate"] = float(fee_rate)
    trades["fee_round_up_to_cent"] = bool(round_up_to_cent)
    trades["strategy_id"] = strategy_id
    trades["strategy_label"] = strategy_label
    trades["strategy_probability_column"] = probability_col
    trades["favorable_edge_after_fees"] = trades["favorable_edge_gross"] - trades["fee_per_contract"]
    trades["signal_hurdle"] = required_edge + np.where(use_fee_for_signal, trades["fee_per_contract"], 0.0)
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
    overlap: pd.DataFrame,
    *,
    scenario_id: str,
    scenario_label: str,
    use_fee_for_signal: bool,
    required_edge: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    frames = [
        build_fee_aware_strategy_trades(
            overlap,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            scenario_id=scenario_id,
            scenario_label=scenario_label,
            use_fee_for_signal=use_fee_for_signal,
            required_edge=required_edge,
            fee_rate=fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        for strategy_id, strategy_label, probability_col in base.STRATEGIES
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_fee_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
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
    grouped["cumulative_winrate"] = grouped["cumulative_wins"] / grouped["cumulative_trades"]
    grouped["cumulative_roi_after_fees"] = grouped["cumulative_pnl_after_fees"] / grouped["cumulative_cost_after_fees"].replace(0, np.nan)
    grouped["running_pnl_high_water_mark"] = grouped.groupby(keys)["cumulative_pnl_after_fees"].cummax()
    grouped["pnl_drawdown_after_fees"] = grouped["running_pnl_high_water_mark"] - grouped["cumulative_pnl_after_fees"]
    return grouped


def summarize_fee_strategy(
    frame: pd.DataFrame,
    *,
    overlap_rows: int,
    overlap_timestamps: int,
    timeseries: pd.DataFrame,
) -> Dict[str, Any]:
    signal_rows = int(len(frame))
    signal_timestamps = int(frame["forecast_datetime_utc"].nunique()) if signal_rows else 0
    ts = timeseries.sort_values("forecast_datetime_utc").copy()
    total_cost_after_fees = float(frame["cost_per_contract_after_fees"].sum()) if signal_rows else 0.0
    total_pnl_after_fees = float(frame["realized_pnl_after_fees"].sum()) if signal_rows else 0.0
    return {
        "scenario_id": str(frame["scenario_id"].iloc[0]) if signal_rows else "",
        "scenario_label": str(frame["scenario_label"].iloc[0]) if signal_rows else "",
        "strategy_id": str(frame["strategy_id"].iloc[0]) if signal_rows else "",
        "strategy_label": str(frame["strategy_label"].iloc[0]) if signal_rows else "",
        "overlap_rows": int(overlap_rows),
        "overlap_timestamps": int(overlap_timestamps),
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "row_signal_rate": signal_rows / overlap_rows if overlap_rows else np.nan,
        "timestamp_signal_rate": signal_timestamps / overlap_timestamps if overlap_timestamps else np.nan,
        "winrate": float(frame["win"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(frame["model_probability_selected_side"].mean()) if signal_rows else np.nan,
        "average_model_k_selected_side_price": float(frame["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_gross_edge": float(frame["favorable_edge_gross"].mean()) if signal_rows else np.nan,
        "average_after_fee_edge": float(frame["favorable_edge_after_fees"].mean()) if signal_rows else np.nan,
        "total_expected_value_before_fees": float(frame["expected_value_before_fees"].sum()) if signal_rows else 0.0,
        "total_expected_value_after_fees": float(frame["expected_value_after_fees"].sum()) if signal_rows else 0.0,
        "total_fees": float(frame["fee_per_contract"].sum()) if signal_rows else 0.0,
        "average_fee": float(frame["fee_per_contract"].mean()) if signal_rows else np.nan,
        "total_cost_before_fees": float(frame["cost_per_contract_before_fees"].sum()) if signal_rows else 0.0,
        "total_cost_after_fees": total_cost_after_fees,
        "total_pnl_before_fees": float(frame["realized_pnl_before_fees"].sum()) if signal_rows else 0.0,
        "total_pnl_after_fees": total_pnl_after_fees,
        "average_pnl_after_fees": float(frame["realized_pnl_after_fees"].mean()) if signal_rows else np.nan,
        "roi_after_fees": total_pnl_after_fees / total_cost_after_fees if total_cost_after_fees else np.nan,
        "best_period_pnl_after_fees": float(ts["period_pnl_after_fees"].max()) if not ts.empty else np.nan,
        "worst_period_pnl_after_fees": float(ts["period_pnl_after_fees"].min()) if not ts.empty else np.nan,
        "max_drawdown_after_fees": float(ts["pnl_drawdown_after_fees"].max()) if not ts.empty else np.nan,
        "final_cumulative_pnl_after_fees": float(ts["cumulative_pnl_after_fees"].iloc[-1]) if not ts.empty else np.nan,
    }


def build_fee_strategy_summary(overlap: pd.DataFrame, trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for (scenario_id, strategy_id), part in trades.groupby(["scenario_id", "strategy_id"], sort=False):
        ts = timeseries[(timeseries["scenario_id"] == scenario_id) & (timeseries["strategy_id"] == strategy_id)]
        rows.append(
            summarize_fee_strategy(
                part,
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
                timeseries=ts,
            )
        )
    return pd.DataFrame(rows)


def build_fee_side_summary(overlap: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for (_scenario_id, _strategy_id, _side), part in trades.groupby(["scenario_id", "strategy_id", "trade_side"], sort=False):
        ts = build_fee_pnl_timeseries(part)
        rows.append(
            summarize_fee_strategy(
                part,
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
                timeseries=ts,
            )
            | {"trade_side": str(part["trade_side"].iloc[0])}
        )
    return pd.DataFrame(rows)


def trade_output_columns(trades: pd.DataFrame) -> List[str]:
    cols = [
        "scenario_id",
        "scenario_label",
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
        "p_model_o",
        "p_exponential_hybrid",
        "p_normal",
        "p_shock",
        "p_model_k",
        "yes_edge",
        "no_edge",
        "favorable_edge_gross",
        "favorable_edge_after_fees",
        "signal_hurdle",
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
        "model_o_source_file",
        "model_k_source_file",
    ]
    cols.extend([col for col in ["binance_audit_price", "binance_reference_price", "join_key_used"] if col in trades.columns])
    return [col for col in cols if col in trades.columns]


def format_fee_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    for column in out.columns:
        lower = column.lower()
        if any(token in lower for token in ["rows", "timestamps", "trades", "wins"]):
            out[column] = out[column].map(lambda v: base.format_number(v, digits=0))
        elif any(token in lower for token in ["rate", "roi", "winrate", "frequency"]):
            out[column] = out[column].map(lambda v: base.format_number(v, digits=4))
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability", "fee"]):
            out[column] = out[column].map(lambda v: base.format_number(v, digits=4, signed="pnl" in lower))
        else:
            out[column] = out[column].map(
                lambda v: base.format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def fee_dashboard_cards(summary: pd.DataFrame, scenario_label: str) -> str:
    scenario = summary[summary["scenario_label"] == scenario_label].copy()
    best = scenario.loc[scenario["total_pnl_after_fees"].idxmax()]
    cards = [
        ("Best Strategy", str(best["strategy_label"])),
        ("Best Net PnL", base.format_number(best["total_pnl_after_fees"], digits=3, signed=True)),
        ("Best ROI", base.format_number(best["roi_after_fees"], digits=4)),
        ("Best Max DD", base.format_number(best["max_drawdown_after_fees"], digits=3)),
        ("Signal Rows", base.format_number(scenario["signal_rows"].sum(), digits=0)),
        ("Total Fees", base.format_number(scenario["total_fees"].sum(), digits=3)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def scenario_overlay_figure(timeseries: pd.DataFrame, *, scenario_label: str) -> Any:
    fig = go.Figure()
    scenario_ts = timeseries[timeseries["scenario_label"] == scenario_label].copy()
    for _strategy_id, strategy_label, _probability_col in base.STRATEGIES:
        part = scenario_ts[scenario_ts["strategy_label"] == strategy_label].sort_values("forecast_datetime_utc")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["forecast_datetime_utc"],
                y=part["cumulative_pnl_after_fees"],
                mode="lines",
                name=strategy_label,
                line=dict(color=base.STRATEGY_COLORS.get(strategy_label), width=3),
            )
        )
    fig.update_layout(title=f"{scenario_label}: Cumulative Net PnL")
    fig.update_xaxes(title_text="Forecast timestamp")
    fig.update_yaxes(title_text="Cumulative PnL after fees")
    base.plotly_theme(fig, height=520, top_margin=92)
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
    fig.add_trace(go.Bar(x=labels, y=scenario["total_pnl_after_fees"], marker_color="#6cb6ff"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=scenario["total_fees"], marker_color="#ff8f70"), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=scenario["signal_rows"], marker_color="#35c7b7"), row=2, col=1)
    fig.add_trace(go.Bar(x=labels, y=scenario["max_drawdown_after_fees"], marker_color="#fbbf24"), row=2, col=2)
    fig.update_yaxes(title_text="PnL", row=1, col=1)
    fig.update_yaxes(title_text="Fees", row=1, col=2)
    fig.update_yaxes(title_text="Rows", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=2)
    base.plotly_theme(fig, height=760, top_margin=96)
    return fig


def overview_figure(summary: pd.DataFrame) -> Any:
    fig = go.Figure()
    for _strategy_id, strategy_label, _probability_col in base.STRATEGIES:
        part = summary[summary["strategy_label"] == strategy_label].copy()
        fig.add_trace(
            go.Bar(
                x=part["scenario_label"],
                y=part["total_pnl_after_fees"],
                name=strategy_label,
                marker_color=base.STRATEGY_COLORS.get(strategy_label),
            )
        )
    fig.update_layout(title="Fee-Aware Net PnL By Scenario", barmode="group")
    fig.update_xaxes(title_text="Scenario")
    fig.update_yaxes(title_text="Total PnL after fees")
    base.plotly_theme(fig, height=560, top_margin=92)
    return fig


def figures_to_html(figures: Sequence[Any]) -> str:
    if not HAS_PLOTLY:
        return "<section class='panel'><p>Plotly is not available in this environment.</p></section>"
    return "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, fig in enumerate(figures)
    )


def build_scenario_dashboard(
    *,
    scenario_label: str,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    fee_rate: float,
    round_up_to_cent: bool,
    fee_model_label: str,
) -> str:
    scenario_summary = summary[summary["scenario_label"] == scenario_label].copy()
    scenario_side = side_summary[side_summary["scenario_label"] == scenario_label].copy()
    figures = [scenario_overlay_figure(timeseries, scenario_label=scenario_label), scenario_bar_figure(summary, scenario_label=scenario_label)]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model O {html.escape(fee_model_label)} PnL - {html.escape(scenario_label)}</title>
  {base.dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">{html.escape(fee_model_label)} Model O Profitability</p>
    <h1>{html.escape(scenario_label)}</h1>
    <p class="lead">
      {html.escape(fee_model_label)} uses <code>fee_rate * selected_price * (1 - selected_price)</code>
      {"with cent round-up" if round_up_to_cent else "without cent round-up"}, fee rate
      <code>{fee_rate}</code>. PnL subtracts entry fees from each one-contract trade.
    </p>
    <div class="cards">{fee_dashboard_cards(summary, scenario_label)}</div>
  </section>

  <h2>PnL Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{base.dataframe_to_html_table(format_fee_table(scenario_summary))}</div></section>

  <h2>Side Summary</h2>
  <section class="panel"><div class="table-wrap">{base.dataframe_to_html_table(format_fee_table(scenario_side))}</div></section>
</main>
</body>
</html>
"""


def build_overview_dashboard(
    *,
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    fee_rate: float,
    round_up_to_cent: bool,
    fee_model_label: str,
) -> str:
    best = summary.loc[summary["total_pnl_after_fees"].idxmax()]
    cards = [
        ("Best Scenario", str(best["scenario_label"])),
        ("Best Strategy", str(best["strategy_label"])),
        ("Best Net PnL", base.format_number(best["total_pnl_after_fees"], digits=3, signed=True)),
        ("Total Scenarios", base.format_number(summary["scenario_id"].nunique(), digits=0)),
        ("Fee Rate", base.format_number(fee_rate, digits=4)),
        ("Round Up", str(round_up_to_cent)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    figures = [overview_figure(summary)]
    for _scenario_id, scenario_label, _use_fee, _edge in SCENARIOS:
        figures.append(scenario_overlay_figure(timeseries, scenario_label=scenario_label))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model O {html.escape(fee_model_label)} Profitability Overview</title>
  {base.dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">{html.escape(fee_model_label)} Model O Profitability</p>
    <h1>Scenario Overview</h1>
    <p class="lead">
      Compares gross-edge trading, after-fee break-even trading, and after-fee trading with
      0.01 / 0.02 required edge across linear Model O, exponential Model O lambda=2, Model A, and Model B.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Overview Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Scenario Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{base.dataframe_to_html_table(format_fee_table(summary))}</div></section>
</main>
</body>
</html>
"""


def write_readme(output_dir: Path, *, fee_rate: float, round_up_to_cent: bool, fee_model_label: str) -> None:
    text = f"""Model O {fee_model_label} Profitability

Fee model:
- Label = {fee_model_label}
- Entry fee per one $1 payout contract = fee_rate * selected_price * (1 - selected_price)
- fee_rate = {fee_rate}
- Cent round-up enabled = {round_up_to_cent}
- No exit or settlement fee is modeled.

Scenarios:
- gross_edge_fee_pnl: old signal rule, model probability greater than market selected-side price; PnL subtracts fees.
- after_fee_edge_0: model probability greater than market selected-side price plus fee.
- after_fee_edge_001: after-fee edge must exceed 0.01.
- after_fee_edge_002: after-fee edge must exceed 0.02.

Strategies:
- Linear Model O
- Exponential Model O lambda=2
- Model A Normal
- Model B Shock

Outputs:
- fee_aware_strategy_summary.csv
- fee_aware_strategy_side_summary.csv
- fee_aware_pnl_timeseries.csv
- fee_aware_dashboard.html
- scenario folders with per-scenario trade minutes, summaries, time series, and dashboards.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.fee_rate < 0.0:
        raise ValueError("--fee-rate must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round_up_to_cent = not args.no_fee_round_up

    model_o = base.load_raw_values(args.model_o_raw_values.resolve(), model_name="Model O")
    model_k = base.load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = base.build_overlap(model_o, model_k)

    scenario_frames: List[pd.DataFrame] = []
    for scenario_id, scenario_label, use_fee_for_signal, required_edge in SCENARIOS:
        trades = build_scenario_trades(
            overlap,
            scenario_id=scenario_id,
            scenario_label=scenario_label,
            use_fee_for_signal=use_fee_for_signal,
            required_edge=required_edge,
            fee_rate=args.fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        scenario_frames.append(trades)

    all_trades = pd.concat(scenario_frames, ignore_index=True)
    timeseries = build_fee_pnl_timeseries(all_trades)
    summary = build_fee_strategy_summary(overlap, all_trades, timeseries)
    side_summary = build_fee_side_summary(overlap, all_trades)

    summary.to_csv(output_dir / "fee_aware_strategy_summary.csv", index=False)
    side_summary.to_csv(output_dir / "fee_aware_strategy_side_summary.csv", index=False)
    timeseries.to_csv(output_dir / "fee_aware_pnl_timeseries.csv", index=False)
    if args.write_trade_minutes:
        all_trades[trade_output_columns(all_trades)].to_csv(output_dir / "fee_aware_trade_minutes.csv", index=False)

    for scenario_id, scenario_label, _use_fee_for_signal, _required_edge in SCENARIOS:
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_trades = all_trades[all_trades["scenario_id"] == scenario_id].copy()
        scenario_ts = timeseries[timeseries["scenario_id"] == scenario_id].copy()
        scenario_summary = summary[summary["scenario_id"] == scenario_id].copy()
        scenario_side_summary = side_summary[side_summary["scenario_id"] == scenario_id].copy()

        scenario_trades[trade_output_columns(scenario_trades)].to_csv(scenario_dir / "trade_minutes.csv", index=False)
        scenario_ts.to_csv(scenario_dir / "pnl_timeseries.csv", index=False)
        scenario_summary.to_csv(scenario_dir / "strategy_summary.csv", index=False)
        scenario_side_summary.to_csv(scenario_dir / "strategy_side_summary.csv", index=False)
        html_text = build_scenario_dashboard(
            scenario_label=scenario_label,
            summary=summary,
            side_summary=side_summary,
            timeseries=timeseries,
            fee_rate=args.fee_rate,
            round_up_to_cent=round_up_to_cent,
            fee_model_label=args.fee_model_label,
        )
        (scenario_dir / "dashboard.html").write_text(html_text, encoding="utf-8")

    overview_html = build_overview_dashboard(
        summary=summary,
        timeseries=timeseries,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
        fee_model_label=args.fee_model_label,
    )
    (output_dir / "fee_aware_dashboard.html").write_text(overview_html, encoding="utf-8")
    write_readme(
        output_dir,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
        fee_model_label=args.fee_model_label,
    )

    print(f"{args.fee_model_label} output directory: {output_dir}")
    print(
        summary[
            [
                "scenario_label",
                "strategy_label",
                "signal_rows",
                "total_fees",
                "total_pnl_after_fees",
                "roi_after_fees",
                "max_drawdown_after_fees",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
