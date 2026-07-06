from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover - only used if Plotly is unavailable.
    go = None
    make_subplots = None
    HAS_PLOTLY = False


JOIN_KEYS = ["event_contract_id", "forecast_datetime_utc"]
OUTPUT_FOLDER_NAME = "Model_O_Profitability_Outputs"
STRATEGIES = (
    ("model_o", "Model O Hybrid", "p_model_o"),
    ("model_o_exp_lambda_2", "Model O Exp Hybrid lambda=2", "p_exponential_hybrid"),
    ("model_a", "Model A Normal", "p_normal"),
    ("model_b", "Model B Shock", "p_shock"),
)
STRATEGY_COLORS = {
    "Model O Hybrid": "#6cb6ff",
    "Model O Exp Hybrid lambda=2": "#fbbf24",
    "Model A Normal": "#35c7b7",
    "Model B Shock": "#ff8f70",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Model O profitability versus Model K by taking favorable YES or NO trades. "
            "BUY_YES is allowed when p_model_o > p_model_k; BUY_NO is allowed when "
            "1 - p_model_o > 1 - p_model_k."
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
        help="Directory where profitability outputs will be written.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum favorable-direction edge required for a trade.",
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
    for column in ["q_shock", "shock_weight_exponential", "p_normal", "p_shock", "p_exponential_hybrid"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def build_overlap(model_o: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    model_o_cols = [
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
    optional_cols = [
        "binance_audit_price",
        "binance_reference_price",
        "join_key_used",
        "q_shock",
        "shock_weight_exponential",
        "p_normal",
        "p_shock",
        "p_exponential_hybrid",
        "p_raw_pre_isotonic",
        "p_calibrated_isotonic",
    ]
    model_o_cols.extend([col for col in optional_cols if col in model_o.columns])

    model_k_cols = ["event_contract_id", "forecast_datetime_utc", "p_kalshi", "source_file"]
    overlap = model_o[model_o_cols].merge(
        model_k[model_k_cols],
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model_o", "_model_k"),
        validate="one_to_one",
    )
    overlap = overlap.rename(
        columns={
            "p_kalshi_model_o": "p_model_o",
            "p_kalshi_model_k": "p_model_k",
            "source_file_model_o": "model_o_source_file",
            "source_file_model_k": "model_k_source_file",
        }
    )
    overlap["yes_edge"] = overlap["p_model_o"] - overlap["p_model_k"]
    overlap["no_edge"] = (1.0 - overlap["p_model_o"]) - (1.0 - overlap["p_model_k"])
    return overlap.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True)


def build_strategy_trades(
    overlap: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    edge_threshold: float,
) -> pd.DataFrame:
    if probability_col not in overlap.columns:
        raise ValueError(f"Cannot build {strategy_label} trades; missing {probability_col}.")

    work = overlap.dropna(subset=[probability_col, "p_model_k", "p_reality"]).copy()
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

    trades["strategy_id"] = strategy_id
    trades["strategy_label"] = strategy_label
    trades["strategy_probability_column"] = probability_col
    trades["expected_value_per_contract"] = trades["favorable_edge"]
    trades["realized_pnl_per_contract"] = trades["gross_payout_per_contract"] - trades["cost_per_contract"]
    trades["result_label"] = np.where(trades["win"] == 1, "win", "loss")
    return trades.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"]).reset_index(
        drop=True
    )


def build_trades(overlap: pd.DataFrame, *, edge_threshold: float) -> pd.DataFrame:
    return build_strategy_trades(
        overlap,
        strategy_id="model_o",
        strategy_label="Model O Hybrid",
        probability_col="p_model_o",
        edge_threshold=edge_threshold,
    )


def build_all_strategy_trades(overlap: pd.DataFrame, *, edge_threshold: float) -> pd.DataFrame:
    frames = [
        build_strategy_trades(
            overlap,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            edge_threshold=edge_threshold,
        )
        for strategy_id, strategy_label, probability_col in STRATEGIES
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["strategy_id", "forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"]
    ).reset_index(drop=True)


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
        "winrate": float(frame["win"].mean()) if signal_rows else np.nan,
        "average_model_o_selected_side_probability": float(frame["model_probability_selected_side"].mean())
        if signal_rows
        else np.nan,
        "average_model_k_selected_side_price": float(frame["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_edge": float(frame["favorable_edge"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "average_realized_pnl": float(frame["realized_pnl_per_contract"].mean()) if signal_rows else np.nan,
        "total_realized_pnl": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
    }


def build_summary(overlap: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [summarize_segment(trades, segment="overall", overlap_rows=len(overlap))]

    for side, part in trades.groupby("trade_side", dropna=False):
        rows.append(summarize_segment(part, segment=f"trade_side={side}", overlap_rows=len(overlap)))

    for label, part in trades.groupby("contract_label", dropna=False):
        overlap_count = int((overlap["contract_label"] == label).sum())
        rows.append(summarize_segment(part, segment=f"contract_label={label}", overlap_rows=overlap_count))

    bucketed = trades.copy()
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


def build_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "strategy_label",
                "forecast_datetime_utc",
                "period_realized_pnl",
                "period_expected_value",
                "period_cost",
                "period_trades",
                "period_wins",
                "cumulative_realized_pnl",
                "cumulative_expected_value",
                "cumulative_cost",
                "cumulative_trades",
                "cumulative_wins",
                "cumulative_winrate",
                "cumulative_roi_on_cost",
                "pnl_drawdown",
            ]
        )

    grouped = (
        trades.groupby(["strategy_id", "strategy_label", "forecast_datetime_utc"], as_index=False)
        .agg(
            period_realized_pnl=("realized_pnl_per_contract", "sum"),
            period_expected_value=("expected_value_per_contract", "sum"),
            period_cost=("cost_per_contract", "sum"),
            period_trades=("event_contract_id", "size"),
            period_wins=("win", "sum"),
        )
        .sort_values(["strategy_id", "forecast_datetime_utc"])
        .reset_index(drop=True)
    )
    grouped["cumulative_realized_pnl"] = grouped.groupby("strategy_id")["period_realized_pnl"].cumsum()
    grouped["cumulative_expected_value"] = grouped.groupby("strategy_id")["period_expected_value"].cumsum()
    grouped["cumulative_cost"] = grouped.groupby("strategy_id")["period_cost"].cumsum()
    grouped["cumulative_trades"] = grouped.groupby("strategy_id")["period_trades"].cumsum()
    grouped["cumulative_wins"] = grouped.groupby("strategy_id")["period_wins"].cumsum()
    grouped["cumulative_winrate"] = grouped["cumulative_wins"] / grouped["cumulative_trades"]
    grouped["cumulative_roi_on_cost"] = grouped["cumulative_realized_pnl"] / grouped["cumulative_cost"].replace(0, np.nan)
    grouped["running_pnl_high_water_mark"] = grouped.groupby("strategy_id")["cumulative_realized_pnl"].cummax()
    grouped["pnl_drawdown"] = grouped["running_pnl_high_water_mark"] - grouped["cumulative_realized_pnl"]
    return grouped


def summarize_strategy_frame(
    frame: pd.DataFrame,
    *,
    segment: str,
    overlap_rows: int,
    overlap_timestamps: int,
    timeseries: pd.DataFrame,
) -> Dict[str, Any]:
    signal_rows = int(len(frame))
    signal_timestamps = int(frame["forecast_datetime_utc"].nunique()) if signal_rows else 0
    total_cost = float(frame["cost_per_contract"].sum()) if signal_rows else 0.0
    total_pnl = float(frame["realized_pnl_per_contract"].sum()) if signal_rows else 0.0
    total_ev = float(frame["expected_value_per_contract"].sum()) if signal_rows else 0.0
    ts = timeseries.sort_values("forecast_datetime_utc").copy()
    return {
        "segment": segment,
        "overlap_rows": int(overlap_rows),
        "overlap_timestamps": int(overlap_timestamps),
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "row_frequency": signal_rows / overlap_rows if overlap_rows else np.nan,
        "timestamp_frequency": signal_timestamps / overlap_timestamps if overlap_timestamps else np.nan,
        "winrate": float(frame["win"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(frame["model_probability_selected_side"].mean())
        if signal_rows
        else np.nan,
        "average_model_k_selected_side_price": float(frame["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_edge": float(frame["favorable_edge"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "average_realized_pnl": float(frame["realized_pnl_per_contract"].mean()) if signal_rows else np.nan,
        "total_realized_pnl": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
        "best_period_pnl": float(ts["period_realized_pnl"].max()) if not ts.empty else np.nan,
        "worst_period_pnl": float(ts["period_realized_pnl"].min()) if not ts.empty else np.nan,
        "max_drawdown": float(ts["pnl_drawdown"].max()) if not ts.empty else np.nan,
        "final_cumulative_pnl": float(ts["cumulative_realized_pnl"].iloc[-1]) if not ts.empty else np.nan,
    }


def build_strategy_summary(overlap: pd.DataFrame, trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for _strategy_id, strategy_label, _probability_col in STRATEGIES:
        part = trades[trades["strategy_label"] == strategy_label].copy()
        ts = timeseries[timeseries["strategy_label"] == strategy_label].copy()
        rows.append(
            summarize_strategy_frame(
                part,
                segment=strategy_label,
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
                timeseries=ts,
            )
        )
    return pd.DataFrame(rows)


def build_strategy_side_summary(overlap: pd.DataFrame, trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    for (strategy_label, trade_side), part in trades.groupby(["strategy_label", "trade_side"], dropna=False):
        ts = build_pnl_timeseries(part)
        rows.append(
            summarize_strategy_frame(
                part,
                segment=f"{strategy_label} | {trade_side}",
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
                timeseries=ts,
            )
        )
    return pd.DataFrame(rows)


def build_trading_coverage(overlap: pd.DataFrame, all_trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    overlap_keys = set(zip(overlap["event_contract_id"].astype(str), overlap["forecast_datetime_utc"].astype(str)))
    overlap_timestamps = set(overlap["forecast_datetime_utc"].astype(str))
    for _strategy_id, strategy_label, _probability_col in STRATEGIES:
        part = all_trades[all_trades["strategy_label"] == strategy_label].copy()
        signal_keys = set(zip(part["event_contract_id"].astype(str), part["forecast_datetime_utc"].astype(str)))
        signal_timestamps = set(part["forecast_datetime_utc"].astype(str))
        rows.append(
            {
                "strategy": strategy_label,
                "overlap_rows": len(overlap_keys),
                "signal_rows": len(signal_keys),
                "row_signal_rate": len(signal_keys) / len(overlap_keys) if overlap_keys else np.nan,
                "overlap_timestamps": len(overlap_timestamps),
                "signal_timestamps": len(signal_timestamps),
                "timestamp_signal_rate": len(signal_timestamps) / len(overlap_timestamps) if overlap_timestamps else np.nan,
                "all_overlap_rows_traded": signal_keys == overlap_keys,
                "all_overlap_timestamps_traded": signal_timestamps == overlap_timestamps,
            }
        )
    return pd.DataFrame(rows)


def output_columns(trades: pd.DataFrame) -> List[str]:
    cols = [
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
        "model_o_source_file",
        "model_k_source_file",
    ]
    cols.extend(
        [
            col
            for col in [
                "q_shock",
                "shock_weight_exponential",
                "p_normal",
                "p_shock",
                "p_exponential_hybrid",
                "binance_audit_price",
                "binance_reference_price",
                "join_key_used",
            ]
            if col in trades.columns
        ]
    )
    return cols


def safe_output_columns(trades: pd.DataFrame) -> List[str]:
    return [col for col in output_columns(trades) if col in trades.columns]


def write_readme(output_dir: Path, *, edge_threshold: float) -> None:
    text = f"""Model O Profitability Analysis

Signal rules:
BUY_YES when p_model_o - p_model_k > {edge_threshold}.
BUY_NO when (1 - p_model_o) - (1 - p_model_k) > {edge_threshold}.

Trade assumption:
Each signal buys one $1 payout contract in the favorable direction. Model K probability is treated as the entry price.

Metrics:
- expected_value_per_contract = selected-side Model O probability - selected-side Model K price
- realized_pnl_per_contract = selected-side official payout - selected-side Model K price
- win = 1 when the selected side pays out, else 0
- total_realized_pnl = sum realized_pnl_per_contract

Additional strategy outputs:
- all_strategy_trade_minutes.csv includes linear Model O, exponential Model O lambda=2, model_a (p_normal), and model_b (p_shock).
- strategy_pnl_timeseries.csv aggregates realized PnL by forecast timestamp and strategy.
- pnl_dashboard.html overlays cumulative realized PnL for the strategy variants.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def format_number(value: Any, *, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
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
        elif "all_overlap" in lower:
            out[column] = out[column].map(str)
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower))
        elif any(token in lower for token in ["rate", "frequency", "roi", "winrate"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def plotly_theme(fig: Any, *, height: int, top_margin: int = 80) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1b26",
        plot_bgcolor="#0d1b26",
        font=dict(color="#e8f1fa"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.14,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=56, r=28, t=top_margin, b=44),
        height=height,
    )
    fig.update_xaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")
    fig.update_yaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")


def cumulative_overlay_figure(timeseries: pd.DataFrame) -> Any:
    fig = go.Figure()
    for _strategy_id, strategy_label, _probability_col in STRATEGIES:
        part = timeseries[timeseries["strategy_label"] == strategy_label].sort_values("forecast_datetime_utc")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["forecast_datetime_utc"],
                y=part["cumulative_realized_pnl"],
                mode="lines",
                name=strategy_label,
                line=dict(color=STRATEGY_COLORS.get(strategy_label), width=3),
            )
        )
    fig.update_layout(title="Cumulative Realized PnL Overlay")
    fig.update_xaxes(title_text="Forecast timestamp")
    fig.update_yaxes(title_text="Cumulative PnL")
    plotly_theme(fig, height=500, top_margin=92)
    return fig


def individual_pnl_figure(timeseries: pd.DataFrame) -> Any:
    subplot_titles = [strategy_label for _strategy_id, strategy_label, _probability_col in STRATEGIES]
    n_rows = len(STRATEGIES)
    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=subplot_titles, shared_xaxes=True, vertical_spacing=0.06)
    for row_idx, (_strategy_id, strategy_label, _probability_col) in enumerate(STRATEGIES, start=1):
        part = timeseries[timeseries["strategy_label"] == strategy_label].sort_values("forecast_datetime_utc")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["forecast_datetime_utc"],
                y=part["cumulative_realized_pnl"],
                mode="lines",
                name=strategy_label,
                line=dict(color=STRATEGY_COLORS.get(strategy_label), width=2.5),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=part["forecast_datetime_utc"],
                y=part["period_realized_pnl"],
                name=f"{strategy_label} period PnL",
                marker_color=STRATEGY_COLORS.get(strategy_label),
                opacity=0.28,
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )
    fig.update_layout(title="Individual Strategy PnL Curves")
    fig.update_xaxes(title_text="Forecast timestamp", row=n_rows, col=1)
    fig.update_yaxes(title_text="PnL")
    plotly_theme(fig, height=260 * n_rows + 120, top_margin=96)
    return fig


def side_mix_figure(all_trades: pd.DataFrame) -> Any:
    side_counts = (
        all_trades.groupby(["strategy_label", "trade_side"], as_index=False)
        .size()
        .rename(columns={"size": "trades"})
    )
    fig = go.Figure()
    for side, part in side_counts.groupby("trade_side"):
        fig.add_trace(go.Bar(x=part["strategy_label"], y=part["trades"], name=side))
    fig.update_layout(title="Trade Direction Counts", barmode="stack")
    fig.update_xaxes(title_text="Strategy")
    fig.update_yaxes(title_text="Trades")
    plotly_theme(fig, height=420, top_margin=92)
    return fig


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
      --blue: #6cb6ff;
      --good: #35c7b7;
      --warn: #fbbf24;
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
      border-radius: 14px;
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
      max-width: 940px;
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
      border-radius: 10px;
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
      border-radius: 14px;
      background: #0d1b26;
      overflow: hidden;
      margin-bottom: 18px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
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


def build_pnl_dashboard_html(
    *,
    strategy_summary: pd.DataFrame,
    strategy_side_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    timeseries: pd.DataFrame,
    all_trades: pd.DataFrame,
    edge_threshold: float,
) -> str:
    cards = []
    for _, row in strategy_summary.iterrows():
        cards.append(
            (
                str(row["segment"]),
                format_number(row["total_realized_pnl"], digits=3, signed=True),
                f"ROI {format_number(row['roi_on_cost'], digits=4)} | DD {format_number(row['max_drawdown'], digits=3)}",
            )
        )
    cards.append(
        (
            "Overlap Rows",
            format_number(int(coverage["overlap_rows"].iloc[0]), digits=0),
            f"{format_number(int(coverage['overlap_timestamps'].iloc[0]), digits=0)} timestamps",
        )
    )
    card_html = "".join(
        (
            f"<section class='metric-card'><span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong><span>{html.escape(note)}</span></section>"
        )
        for label, value, note in cards
    )
    figures_html = figures_to_html(
        [
            cumulative_overlay_figure(timeseries),
            individual_pnl_figure(timeseries),
            side_mix_figure(all_trades),
        ]
    )
    strategy_table = dataframe_to_html_table(format_table(strategy_summary))
    side_table = dataframe_to_html_table(format_table(strategy_side_summary))
    coverage_table = dataframe_to_html_table(format_table(coverage))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model O Strategy PnL Dashboard</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Strategy PnL Evaluation</p>
    <h1>Model O, Model A, and Model B PnL</h1>
    <p class="lead">
      PnL assumes one $1-payout contract per favorable row, priced at Model K. BUY_YES is taken when
      the strategy probability exceeds Model K by more than <code>{edge_threshold}</code>; BUY_NO is
      taken when the strategy NO probability exceeds the Model K NO price by more than
      <code>{edge_threshold}</code>.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Cumulative PnL</h2>
  <section class="chart-wrap">{figures_html}</section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{strategy_table}</div></section>

  <h2>Direction Summary</h2>
  <section class="panel"><div class="table-wrap">{side_table}</div></section>

  <h2>Trading Coverage</h2>
  <section class="panel">
    <p>
      Coverage is measured against the Model O / Model K overlapping scored-row universe. A row is one
      contract forecast at one timestamp, not just a timestamp.
    </p>
    <div class="table-wrap">{coverage_table}</div>
  </section>

  <h2>Output Files</h2>
  <section class="panel">
    <p>
      CSV outputs are saved next to this dashboard:
      <code>strategy_pnl_timeseries.csv</code>, <code>all_strategy_trade_minutes.csv</code>,
      <code>strategy_profitability_summary.csv</code>, and <code>strategy_trading_coverage.csv</code>.
    </p>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.edge_threshold < 0:
        raise ValueError("--edge-threshold must be non-negative.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_o = load_raw_values(args.model_o_raw_values.resolve(), model_name="Model O")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = build_overlap(model_o, model_k)
    trades = build_trades(overlap, edge_threshold=args.edge_threshold)
    all_strategy_trades = build_all_strategy_trades(overlap, edge_threshold=args.edge_threshold)
    pnl_timeseries = build_pnl_timeseries(all_strategy_trades)
    strategy_summary = build_strategy_summary(overlap, all_strategy_trades, pnl_timeseries)
    strategy_side_summary = build_strategy_side_summary(overlap, all_strategy_trades, pnl_timeseries)
    trading_coverage = build_trading_coverage(overlap, all_strategy_trades)

    overlap.to_csv(output_dir / "model_o_model_k_overlap.csv", index=False)
    trades[safe_output_columns(trades)].to_csv(output_dir / "profitable_minutes.csv", index=False)
    trades[safe_output_columns(trades)].to_csv(output_dir / "trade_signal_minutes.csv", index=False)
    all_strategy_trades[safe_output_columns(all_strategy_trades)].to_csv(
        output_dir / "all_strategy_trade_minutes.csv",
        index=False,
    )
    pnl_timeseries.to_csv(output_dir / "strategy_pnl_timeseries.csv", index=False)
    build_summary(overlap, trades).to_csv(output_dir / "profitability_summary.csv", index=False)
    strategy_summary.to_csv(output_dir / "strategy_profitability_summary.csv", index=False)
    strategy_side_summary.to_csv(output_dir / "strategy_side_profitability_summary.csv", index=False)
    trading_coverage.to_csv(output_dir / "strategy_trading_coverage.csv", index=False)
    dashboard_html = build_pnl_dashboard_html(
        strategy_summary=strategy_summary,
        strategy_side_summary=strategy_side_summary,
        coverage=trading_coverage,
        timeseries=pnl_timeseries,
        all_trades=all_strategy_trades,
        edge_threshold=args.edge_threshold,
    )
    (output_dir / "pnl_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    if HAS_PLOTLY:
        overlay_html = cumulative_overlay_figure(pnl_timeseries).to_html(full_html=True, include_plotlyjs="cdn")
        (output_dir / "pnl_overlay.html").write_text(overlay_html, encoding="utf-8")
    write_readme(output_dir, edge_threshold=args.edge_threshold)

    total_pnl = float(trades["realized_pnl_per_contract"].sum()) if len(trades) else 0.0
    print(f"Overlapping rows: {len(overlap):,}")
    print(f"Profitable-condition rows: {len(trades):,}")
    print(f"Total realized PnL: {total_pnl:,.6f}")
    print(f"Profitable minutes CSV: {output_dir / 'profitable_minutes.csv'}")
    print(f"Profitability summary: {output_dir / 'profitability_summary.csv'}")
    print(f"Strategy PnL time series: {output_dir / 'strategy_pnl_timeseries.csv'}")
    print(f"PnL dashboard: {output_dir / 'pnl_dashboard.html'}")


if __name__ == "__main__":
    main()
