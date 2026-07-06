from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List

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


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Standalone PnL experiment: trade Model B shock probabilities only on rows "
            "where the breakout/shock probability q_shock is above a threshold."
        )
    )
    parser.add_argument(
        "--model-n-raw-values",
        type=Path,
        default=root / "Model_N" / "model_N_Evals_Outputs" / "raw_values.csv",
        help="Model N raw_values.csv containing q_shock and p_shock columns.",
    )
    parser.add_argument(
        "--model-k-raw-values",
        type=Path,
        default=root / "Model_K" / "Model_K_outputs" / "raw_values.csv",
        help="Model K raw_values.csv used as the entry price/probability source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Self-contained output folder for this experiment.",
    )
    parser.add_argument(
        "--shock-threshold",
        type=float,
        default=None,
        help="Optional single-threshold compatibility mode. If set, overrides --shock-thresholds.",
    )
    parser.add_argument(
        "--shock-thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help="Threshold grid for q_shock eligibility.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum selected-side edge versus Model K required to trade.",
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
    numeric_columns = [
        "p_kalshi",
        "p_reality",
        "minute_number",
        "strike",
        "minutes_to_settlement",
        "q_shock",
        "p_shock",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def build_overlap(model_n: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    required = {"q_shock", "p_shock"}
    missing = required - set(model_n.columns)
    if missing:
        raise ValueError(f"Model N raw values are missing required shock columns: {sorted(missing)}")

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
        "p_shock",
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
    overlap["p_strategy"] = pd.to_numeric(overlap["p_shock"], errors="coerce").clip(0.0, 1.0)
    overlap["q_shock"] = pd.to_numeric(overlap["q_shock"], errors="coerce").clip(0.0, 1.0)
    overlap["p_model_k"] = pd.to_numeric(overlap["p_model_k"], errors="coerce").clip(0.0, 1.0)
    return overlap.dropna(subset=["p_strategy", "q_shock", "p_model_k"]).sort_values(
        ["forecast_datetime_utc", "event_ticker", "contract_label"]
    ).reset_index(drop=True)


def filter_breakout_rows(overlap: pd.DataFrame, *, shock_threshold: float) -> pd.DataFrame:
    return overlap[overlap["q_shock"] > shock_threshold].copy().reset_index(drop=True)


def threshold_slug(threshold: float) -> str:
    return f"q_gt_{threshold:.2f}".replace(".", "_")


def build_trades(filtered: pd.DataFrame, *, edge_threshold: float, shock_threshold: float) -> pd.DataFrame:
    work = filtered.copy()
    work["yes_edge"] = work["p_strategy"] - work["p_model_k"]
    work["no_edge"] = (1.0 - work["p_strategy"]) - (1.0 - work["p_model_k"])

    buy_yes = work[work["yes_edge"] > edge_threshold].copy()
    buy_yes["trade_side"] = "BUY_YES"
    buy_yes["favorable_edge"] = buy_yes["yes_edge"]
    buy_yes["strategy_probability_selected_side"] = buy_yes["p_strategy"]
    buy_yes["model_k_price_selected_side"] = buy_yes["p_model_k"]
    buy_yes["cost_per_contract"] = buy_yes["p_model_k"]
    buy_yes["gross_payout_per_contract"] = buy_yes["p_reality"]
    buy_yes["win"] = (buy_yes["p_reality"] == 1.0).astype(int)

    buy_no = work[work["no_edge"] > edge_threshold].copy()
    buy_no["trade_side"] = "BUY_NO"
    buy_no["favorable_edge"] = buy_no["no_edge"]
    buy_no["strategy_probability_selected_side"] = 1.0 - buy_no["p_strategy"]
    buy_no["model_k_price_selected_side"] = 1.0 - buy_no["p_model_k"]
    buy_no["cost_per_contract"] = 1.0 - buy_no["p_model_k"]
    buy_no["gross_payout_per_contract"] = 1.0 - buy_no["p_reality"]
    buy_no["win"] = (buy_no["p_reality"] == 0.0).astype(int)

    trades = pd.concat([buy_yes, buy_no], ignore_index=True)
    if trades.empty:
        return trades

    trades["shock_threshold"] = shock_threshold
    trades["threshold_slug"] = threshold_slug(shock_threshold)
    trades["strategy"] = f"Model B Shock | q_shock > {shock_threshold:.2f}"
    trades["expected_value_per_contract"] = trades["favorable_edge"]
    trades["realized_pnl_per_contract"] = trades["gross_payout_per_contract"] - trades["cost_per_contract"]
    trades["result_label"] = np.where(trades["win"] == 1, "win", "loss")
    return trades.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"]).reset_index(
        drop=True
    )


def build_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = (
        trades.groupby("forecast_datetime_utc", as_index=False)
        .agg(
            period_realized_pnl=("realized_pnl_per_contract", "sum"),
            period_expected_value=("expected_value_per_contract", "sum"),
            period_cost=("cost_per_contract", "sum"),
            period_trades=("event_contract_id", "size"),
            period_wins=("win", "sum"),
            mean_q_shock=("q_shock", "mean"),
        )
        .sort_values("forecast_datetime_utc")
        .reset_index(drop=True)
    )
    out["cumulative_realized_pnl"] = out["period_realized_pnl"].cumsum()
    out["cumulative_expected_value"] = out["period_expected_value"].cumsum()
    out["cumulative_cost"] = out["period_cost"].cumsum()
    out["cumulative_trades"] = out["period_trades"].cumsum()
    out["cumulative_wins"] = out["period_wins"].cumsum()
    out["cumulative_winrate"] = out["cumulative_wins"] / out["cumulative_trades"]
    out["cumulative_roi_on_cost"] = out["cumulative_realized_pnl"] / out["cumulative_cost"].replace(0, np.nan)
    out["running_pnl_high_water_mark"] = out["cumulative_realized_pnl"].cummax()
    out["pnl_drawdown"] = out["running_pnl_high_water_mark"] - out["cumulative_realized_pnl"]
    if "shock_threshold" in trades.columns:
        out.insert(0, "shock_threshold", float(trades["shock_threshold"].iloc[0]))
        out.insert(1, "threshold_slug", str(trades["threshold_slug"].iloc[0]))
    return out


def summarize_segment(
    frame: pd.DataFrame,
    *,
    segment: str,
    overlap_rows: int,
    overlap_timestamps: int,
    filtered_rows: int,
    filtered_timestamps: int,
    timeseries: pd.DataFrame,
) -> Dict[str, Any]:
    signal_rows = int(len(frame))
    signal_timestamps = int(frame["forecast_datetime_utc"].nunique()) if signal_rows else 0
    total_cost = float(frame["cost_per_contract"].sum()) if signal_rows else 0.0
    total_pnl = float(frame["realized_pnl_per_contract"].sum()) if signal_rows else 0.0
    total_ev = float(frame["expected_value_per_contract"].sum()) if signal_rows else 0.0
    return {
        "segment": segment,
        "overlap_rows": int(overlap_rows),
        "overlap_timestamps": int(overlap_timestamps),
        "eligible_rows_q_gt_threshold": int(filtered_rows),
        "eligible_timestamps_q_gt_threshold": int(filtered_timestamps),
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "eligible_row_trade_rate": signal_rows / filtered_rows if filtered_rows else np.nan,
        "eligible_timestamp_trade_rate": signal_timestamps / filtered_timestamps if filtered_timestamps else np.nan,
        "full_overlap_row_trade_rate": signal_rows / overlap_rows if overlap_rows else np.nan,
        "full_overlap_timestamp_trade_rate": signal_timestamps / overlap_timestamps if overlap_timestamps else np.nan,
        "winrate": float(frame["win"].mean()) if signal_rows else np.nan,
        "average_q_shock": float(frame["q_shock"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(frame["strategy_probability_selected_side"].mean())
        if signal_rows
        else np.nan,
        "average_model_k_selected_side_price": float(frame["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_edge": float(frame["favorable_edge"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "average_realized_pnl": float(frame["realized_pnl_per_contract"].mean()) if signal_rows else np.nan,
        "total_realized_pnl": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
        "best_period_pnl": float(timeseries["period_realized_pnl"].max()) if not timeseries.empty else np.nan,
        "worst_period_pnl": float(timeseries["period_realized_pnl"].min()) if not timeseries.empty else np.nan,
        "max_drawdown": float(timeseries["pnl_drawdown"].max()) if not timeseries.empty else np.nan,
        "final_cumulative_pnl": float(timeseries["cumulative_realized_pnl"].iloc[-1]) if not timeseries.empty else np.nan,
    }


def build_summary(overlap: pd.DataFrame, filtered: pd.DataFrame, trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    overlap_rows = len(overlap)
    overlap_timestamps = int(overlap["forecast_datetime_utc"].nunique())
    filtered_rows = len(filtered)
    filtered_timestamps = int(filtered["forecast_datetime_utc"].nunique())
    rows: List[Dict[str, Any]] = [
        summarize_segment(
            trades,
            segment="overall",
            overlap_rows=overlap_rows,
            overlap_timestamps=overlap_timestamps,
            filtered_rows=filtered_rows,
            filtered_timestamps=filtered_timestamps,
            timeseries=timeseries,
        )
    ]

    for side, part in trades.groupby("trade_side", dropna=False):
        side_ts = build_pnl_timeseries(part)
        rows.append(
            summarize_segment(
                part,
                segment=f"trade_side={side}",
                overlap_rows=overlap_rows,
                overlap_timestamps=overlap_timestamps,
                filtered_rows=filtered_rows,
                filtered_timestamps=filtered_timestamps,
                timeseries=side_ts,
            )
        )
    for label, part in trades.groupby("contract_label", dropna=False):
        label_filtered = filtered[filtered["contract_label"] == label]
        label_ts = build_pnl_timeseries(part)
        rows.append(
            summarize_segment(
                part,
                segment=f"contract_label={label}",
                overlap_rows=int((overlap["contract_label"] == label).sum()),
                overlap_timestamps=int(overlap.loc[overlap["contract_label"] == label, "forecast_datetime_utc"].nunique()),
                filtered_rows=len(label_filtered),
                filtered_timestamps=int(label_filtered["forecast_datetime_utc"].nunique()),
                timeseries=label_ts,
            )
        )
    return pd.DataFrame(rows)


def trade_columns(trades: pd.DataFrame) -> List[str]:
    cols = [
        "shock_threshold",
        "threshold_slug",
        "strategy",
        "event_ticker",
        "event_contract_id",
        "forecast_datetime_utc",
        "event_datetime_utc",
        "minute_number",
        "minutes_to_settlement",
        "market_ticker",
        "contract_label",
        "strike",
        "q_shock",
        "trade_side",
        "p_strategy",
        "p_shock",
        "p_model_k",
        "yes_edge",
        "no_edge",
        "favorable_edge",
        "strategy_probability_selected_side",
        "model_k_price_selected_side",
        "expected_value_per_contract",
        "cost_per_contract",
        "gross_payout_per_contract",
        "realized_pnl_per_contract",
        "result_label",
        "win",
        "official_result",
        "p_reality",
        "model_n_source_file",
        "model_k_source_file",
    ]
    cols.extend([col for col in ["binance_audit_price", "binance_reference_price", "join_key_used"] if col in trades.columns])
    return [col for col in cols if col in trades.columns]


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
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower))
        elif any(token in lower for token in ["rate", "frequency", "roi", "winrate", "q_shock"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def plotly_theme(fig: Any, *, height: int, top_margin: int = 86) -> None:
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


def build_figures(timeseries: pd.DataFrame, trades: pd.DataFrame) -> List[Any]:
    if not HAS_PLOTLY or timeseries.empty or trades.empty:
        return []
    fig_top = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Cumulative Realized PnL", "Period PnL"),
        shared_xaxes=True,
        vertical_spacing=0.10,
    )
    fig_top.add_trace(
        go.Scatter(
            x=timeseries["forecast_datetime_utc"],
            y=timeseries["cumulative_realized_pnl"],
            mode="lines",
            name="Cumulative PnL",
            line=dict(color="#ff8f70", width=3),
        ),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Bar(
            x=timeseries["forecast_datetime_utc"],
            y=timeseries["period_realized_pnl"],
            name="Period PnL",
            marker_color="#6cb6ff",
            opacity=0.55,
        ),
        row=2,
        col=1,
    )
    fig_top.update_yaxes(title_text="Cumulative PnL", row=1, col=1)
    fig_top.update_yaxes(title_text="Period PnL", row=2, col=1)
    fig_top.update_xaxes(title_text="Forecast timestamp", row=2, col=1)
    plotly_theme(fig_top, height=700, top_margin=96)

    side_counts = trades.groupby("trade_side", as_index=False).size().rename(columns={"size": "trades"})
    contract_pnl = (
        trades.groupby("contract_label", as_index=False)["realized_pnl_per_contract"]
        .sum()
        .rename(columns={"realized_pnl_per_contract": "total_pnl"})
    )
    fig_bottom = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Trade Direction Counts", "PnL By Contract Label"),
        horizontal_spacing=0.12,
    )
    fig_bottom.add_trace(go.Bar(x=side_counts["trade_side"], y=side_counts["trades"], marker_color="#35c7b7"), row=1, col=1)
    fig_bottom.add_trace(
        go.Bar(x=contract_pnl["contract_label"], y=contract_pnl["total_pnl"], marker_color="#fbbf24"),
        row=1,
        col=2,
    )
    fig_bottom.update_yaxes(title_text="Trades", row=1, col=1)
    fig_bottom.update_yaxes(title_text="Total PnL", row=1, col=2)
    plotly_theme(fig_bottom, height=430, top_margin=96)
    return [fig_top, fig_bottom]


def figures_to_html(figures: List[Any]) -> str:
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


def build_dashboard_html(
    *,
    summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    trades: pd.DataFrame,
    shock_threshold: float,
    edge_threshold: float,
) -> str:
    overall = summary.loc[summary["segment"] == "overall"].iloc[0]
    cards = [
        ("Total PnL", format_number(overall["total_realized_pnl"], digits=3, signed=True)),
        ("Eligible Rows", format_number(overall["eligible_rows_q_gt_threshold"], digits=0)),
        ("Signal Rows", format_number(overall["signal_rows"], digits=0)),
        ("ROI", format_number(overall["roi_on_cost"], digits=4)),
        ("Winrate", format_number(overall["winrate"], digits=4)),
        ("Max Drawdown", format_number(overall["max_drawdown"], digits=3)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    figures_html = figures_to_html(build_figures(timeseries, trades))
    summary_table = dataframe_to_html_table(format_table(summary))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model B Shock Breakout PnL</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Standalone Breakout Experiment</p>
    <h1>Model B Shock PnL When q_shock > {shock_threshold:.2f}</h1>
    <p class="lead">
      This is isolated from the main Model N outputs. It trades only rows where
      <code>q_shock &gt; {shock_threshold:.2f}</code>, uses <code>p_shock</code> as the Model B shock
      fair value, and prices entries at Model K. BUY_YES and BUY_NO both use a selected-side edge
      threshold of <code>{edge_threshold}</code>.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>PnL Charts</h2>
  <section class="chart-wrap">{figures_html}</section>

  <h2>Summary</h2>
  <section class="panel"><div class="table-wrap">{summary_table}</div></section>

  <h2>Output Files</h2>
  <section class="panel">
    <p>
      Outputs in this folder: <code>breakout_shock_trade_minutes.csv</code>,
      <code>breakout_shock_pnl_timeseries.csv</code>, <code>breakout_shock_profitability_summary.csv</code>,
      and <code>filtered_overlap_rows.csv</code>.
    </p>
  </section>
</main>
</body>
</html>
"""


def build_comparison_figures(comparison: pd.DataFrame, combined_timeseries: pd.DataFrame) -> List[Any]:
    if not HAS_PLOTLY:
        return []

    figures: List[Any] = []
    if not combined_timeseries.empty:
        fig_overlay = go.Figure()
        for threshold, part in combined_timeseries.groupby("shock_threshold", sort=True):
            part = part.sort_values("forecast_datetime_utc")
            fig_overlay.add_trace(
                go.Scatter(
                    x=part["forecast_datetime_utc"],
                    y=part["cumulative_realized_pnl"],
                    mode="lines",
                    name=f"q > {threshold:.2f}",
                    line=dict(width=3),
                )
            )
        fig_overlay.update_layout(title="Cumulative Realized PnL By q_shock Threshold")
        fig_overlay.update_xaxes(title_text="Forecast timestamp")
        fig_overlay.update_yaxes(title_text="Cumulative PnL")
        plotly_theme(fig_overlay, height=560, top_margin=92)
        figures.append(fig_overlay)

    if not comparison.empty:
        labels = [f"q > {value:.2f}" for value in comparison["shock_threshold"]]
        fig_bars = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Total PnL", "Trade Rows", "ROI On Cost", "Max Drawdown"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )
        fig_bars.add_trace(
            go.Bar(x=labels, y=comparison["total_realized_pnl"], marker_color="#ff8f70"),
            row=1,
            col=1,
        )
        fig_bars.add_trace(
            go.Bar(x=labels, y=comparison["signal_rows"], marker_color="#6cb6ff"),
            row=1,
            col=2,
        )
        fig_bars.add_trace(
            go.Bar(x=labels, y=comparison["roi_on_cost"], marker_color="#35c7b7"),
            row=2,
            col=1,
        )
        fig_bars.add_trace(
            go.Bar(x=labels, y=comparison["max_drawdown"], marker_color="#fbbf24"),
            row=2,
            col=2,
        )
        fig_bars.update_yaxes(title_text="PnL", row=1, col=1)
        fig_bars.update_yaxes(title_text="Rows", row=1, col=2)
        fig_bars.update_yaxes(title_text="ROI", row=2, col=1)
        fig_bars.update_yaxes(title_text="Drawdown", row=2, col=2)
        plotly_theme(fig_bars, height=720, top_margin=96)
        figures.append(fig_bars)

    return figures


def build_comparison_dashboard_html(
    *,
    comparison: pd.DataFrame,
    segment_summary: pd.DataFrame,
    combined_timeseries: pd.DataFrame,
    thresholds: List[float],
    edge_threshold: float,
) -> str:
    if comparison.empty:
        card_html = "<section class='metric-card'><span>Status</span><strong>No trades</strong></section>"
    else:
        best = comparison.loc[comparison["total_realized_pnl"].idxmax()]
        worst = comparison.loc[comparison["total_realized_pnl"].idxmin()]
        cards = [
            ("Best Threshold", f"q > {best['shock_threshold']:.2f}"),
            ("Best Total PnL", format_number(best["total_realized_pnl"], digits=3, signed=True)),
            ("Worst Total PnL", format_number(worst["total_realized_pnl"], digits=3, signed=True)),
            ("Thresholds", format_number(len(thresholds), digits=0)),
            ("Max Trade Rows", format_number(comparison["signal_rows"].max(), digits=0)),
            ("Min Trade Rows", format_number(comparison["signal_rows"].min(), digits=0)),
        ]
        card_html = "".join(
            f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
            for label, value in cards
        )

    figures_html = figures_to_html(build_comparison_figures(comparison, combined_timeseries))
    comparison_table = dataframe_to_html_table(format_table(comparison))
    segment_table = dataframe_to_html_table(format_table(segment_summary))
    threshold_text = ", ".join(f"{threshold:.2f}" for threshold in thresholds)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model B Shock Breakout Threshold Comparison</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Standalone Breakout Experiment</p>
    <h1>Model B Shock PnL Across q_shock Thresholds</h1>
    <p class="lead">
      This comparison is isolated from the main Model N outputs. Each strategy uses
      <code>p_shock</code> as the Model B shock fair value, prices entries at Model K,
      and trades only rows where <code>q_shock</code> is above one of these thresholds:
      <code>{html.escape(threshold_text)}</code>. BUY_YES and BUY_NO both use a selected-side
      edge threshold of <code>{edge_threshold}</code>.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Comparison Charts</h2>
  <section class="chart-wrap">{figures_html}</section>

  <h2>Threshold Comparison</h2>
  <section class="panel"><div class="table-wrap">{comparison_table}</div></section>

  <h2>Segment Detail</h2>
  <section class="panel"><div class="table-wrap">{segment_table}</div></section>

  <h2>Output Files</h2>
  <section class="panel">
    <p>
      Root comparison outputs: <code>breakout_shock_threshold_comparison.csv</code>,
      <code>breakout_shock_threshold_segment_summary.csv</code>,
      <code>breakout_shock_threshold_pnl_timeseries.csv</code>, and
      <code>breakout_shock_threshold_trade_minutes.csv</code>. Per-threshold files are in
      <code>q_gt_0_05</code>, <code>q_gt_0_10</code>, <code>q_gt_0_15</code>,
      <code>q_gt_0_20</code>, and <code>q_gt_0_25</code>.
    </p>
  </section>
</main>
</body>
</html>
"""


def write_threshold_outputs(
    *,
    output_dir: Path,
    overlap: pd.DataFrame,
    shock_threshold: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    slug = threshold_slug(shock_threshold)
    threshold_dir = output_dir / slug
    threshold_dir.mkdir(parents=True, exist_ok=True)

    filtered = filter_breakout_rows(overlap, shock_threshold=shock_threshold)
    filtered.insert(0, "shock_threshold", shock_threshold)
    filtered.insert(1, "threshold_slug", slug)
    trades = build_trades(filtered, edge_threshold=edge_threshold, shock_threshold=shock_threshold)
    timeseries = build_pnl_timeseries(trades)
    summary = build_summary(overlap, filtered, trades, timeseries)
    summary.insert(0, "shock_threshold", shock_threshold)
    summary.insert(1, "threshold_slug", slug)

    filtered.to_csv(threshold_dir / "filtered_overlap_rows.csv", index=False)
    trades[trade_columns(trades)].to_csv(threshold_dir / "breakout_shock_trade_minutes.csv", index=False)
    timeseries.to_csv(threshold_dir / "breakout_shock_pnl_timeseries.csv", index=False)
    summary.to_csv(threshold_dir / "breakout_shock_profitability_summary.csv", index=False)

    dashboard_html = build_dashboard_html(
        summary=summary,
        timeseries=timeseries,
        trades=trades,
        shock_threshold=shock_threshold,
        edge_threshold=edge_threshold,
    )
    (threshold_dir / "breakout_shock_pnl_dashboard.html").write_text(dashboard_html, encoding="utf-8")

    if np.isclose(shock_threshold, 0.20):
        filtered.to_csv(output_dir / "filtered_overlap_rows.csv", index=False)
        trades[trade_columns(trades)].to_csv(output_dir / "breakout_shock_trade_minutes.csv", index=False)
        timeseries.to_csv(output_dir / "breakout_shock_pnl_timeseries.csv", index=False)
        summary.to_csv(output_dir / "breakout_shock_profitability_summary.csv", index=False)
        (output_dir / "breakout_shock_pnl_dashboard.html").write_text(dashboard_html, encoding="utf-8")

    return {
        "threshold": shock_threshold,
        "slug": slug,
        "output_dir": threshold_dir,
        "filtered": filtered,
        "trades": trades,
        "timeseries": timeseries,
        "summary": summary,
    }


def write_readme(output_dir: Path, *, thresholds: List[float], edge_threshold: float) -> None:
    threshold_lines = "\n".join(f"- q_shock > {threshold:.2f}" for threshold in thresholds)
    threshold_dirs = "\n".join(f"- {threshold_slug(threshold)}/" for threshold in thresholds)
    text = f"""Model B Shock Breakout PnL Experiment

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Rules:
{threshold_lines}
- Use p_shock as the strategy probability
- Compare against Model K as the entry price
- BUY_YES when p_shock - p_model_k > {edge_threshold}
- BUY_NO when (1 - p_shock) - (1 - p_model_k) > {edge_threshold}

Root comparison outputs:
- breakout_shock_threshold_comparison.csv
- breakout_shock_threshold_segment_summary.csv
- breakout_shock_threshold_pnl_timeseries.csv
- breakout_shock_threshold_trade_minutes.csv
- breakout_shock_threshold_comparison_dashboard.html

Per-threshold output folders:
{threshold_dirs}

Compatibility outputs for q_shock > 0.20 are still written in this root outputs folder.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.edge_threshold < 0.0:
        raise ValueError("--edge-threshold must be non-negative.")

    raw_thresholds = [args.shock_threshold] if args.shock_threshold is not None else args.shock_thresholds
    thresholds: List[float] = []
    for threshold in raw_thresholds:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("--shock-thresholds values must be in [0, 1].")
        if not any(np.isclose(threshold, existing) for existing in thresholds):
            thresholds.append(threshold)
    thresholds = sorted(thresholds)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    overlap = build_overlap(model_n, model_k)

    results = [
        write_threshold_outputs(
            output_dir=output_dir,
            overlap=overlap,
            shock_threshold=threshold,
            edge_threshold=args.edge_threshold,
        )
        for threshold in thresholds
    ]

    segment_summary = pd.concat([result["summary"] for result in results], ignore_index=True)
    comparison = (
        segment_summary.loc[segment_summary["segment"] == "overall"]
        .copy()
        .sort_values("shock_threshold")
        .reset_index(drop=True)
    )
    combined_timeseries = pd.concat(
        [result["timeseries"] for result in results if not result["timeseries"].empty],
        ignore_index=True,
    )
    combined_trades = pd.concat(
        [result["trades"] for result in results if not result["trades"].empty],
        ignore_index=True,
    )

    comparison.to_csv(output_dir / "breakout_shock_threshold_comparison.csv", index=False)
    segment_summary.to_csv(output_dir / "breakout_shock_threshold_segment_summary.csv", index=False)
    combined_timeseries.to_csv(output_dir / "breakout_shock_threshold_pnl_timeseries.csv", index=False)
    combined_trades[trade_columns(combined_trades)].to_csv(
        output_dir / "breakout_shock_threshold_trade_minutes.csv", index=False
    )

    comparison_html = build_comparison_dashboard_html(
        comparison=comparison,
        segment_summary=segment_summary,
        combined_timeseries=combined_timeseries,
        thresholds=thresholds,
        edge_threshold=args.edge_threshold,
    )
    (output_dir / "breakout_shock_threshold_comparison_dashboard.html").write_text(comparison_html, encoding="utf-8")
    write_readme(output_dir, thresholds=thresholds, edge_threshold=args.edge_threshold)

    print(f"Thresholds: {', '.join(f'{threshold:.2f}' for threshold in thresholds)}")
    print(f"Overlap rows: {len(overlap):,}")
    print(
        comparison[
            [
                "shock_threshold",
                "eligible_rows_q_gt_threshold",
                "signal_rows",
                "signal_timestamps",
                "total_realized_pnl",
                "roi_on_cost",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"Comparison dashboard: {output_dir / 'breakout_shock_threshold_comparison_dashboard.html'}")
    print(f"Comparison summary: {output_dir / 'breakout_shock_threshold_comparison.csv'}")


if __name__ == "__main__":
    main()
