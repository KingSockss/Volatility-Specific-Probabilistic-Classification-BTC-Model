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
except Exception:  # pragma: no cover
    go = None
    make_subplots = None
    HAS_PLOTLY = False


OUTPUT_FOLDER_NAME = "Posthoc_Regime_PnL_Chart"
DEFAULT_STRATEGY_ID = "model_n_exp_lambda_2"
REGIME_COLORS = {
    "Low Volatility": "rgba(53, 199, 183, 0.20)",
    "Standard Volatility": "rgba(108, 182, 255, 0.14)",
    "High Volatility": "rgba(255, 143, 112, 0.24)",
}
REGIME_LINE_COLORS = {
    "Low Volatility": "#35c7b7",
    "Standard Volatility": "#6cb6ff",
    "High Volatility": "#ff8f70",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    default_diag_dir = root / "Model_N" / "Model_N_Trade_Diagnostics_Outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Build a BTC price + Model N PnL chart shaded by the existing post-hoc volatility regimes."
        )
    )
    parser.add_argument(
        "--strategy-id",
        default=DEFAULT_STRATEGY_ID,
        help="Strategy from the diagnostics pnl_timeseries.csv to plot.",
    )
    parser.add_argument(
        "--pnl-timeseries",
        type=Path,
        default=default_diag_dir / "pnl_timeseries.csv",
        help="Model N trade diagnostics PnL time series.",
    )
    parser.add_argument(
        "--binance-klines",
        type=Path,
        default=root
        / "Model_K_Volatility_Decomposition_RT"
        / "Model_K_Volatility_Decomposition_RT_outputs"
        / "binance_1m_klines.csv",
        help="Binance 1m kline CSV used for BTC close prices.",
    )
    parser.add_argument(
        "--volatility-segments",
        type=Path,
        default=root
        / "Model_K_Volatility_Decomposition_RT"
        / "Model_K_Volatility_Decomposition_RT_outputs"
        / "hourly_market_volatility_segments.csv",
        help="Post-hoc hourly volatility regime table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_diag_dir / OUTPUT_FOLDER_NAME,
        help="Output folder for the chart and source CSVs.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_pnl(path: Path, strategy_id: str) -> pd.DataFrame:
    require_file(path, "PnL timeseries")
    frame = pd.read_csv(path)
    required = {
        "strategy_id",
        "strategy_label",
        "timestamp",
        "period_net_pnl",
        "period_fees",
        "period_trades",
        "cumulative_net_pnl",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ["period_net_pnl", "period_fees", "period_trades", "cumulative_net_pnl"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    strategy = frame[frame["strategy_id"] == strategy_id].copy()
    if strategy.empty:
        available = ", ".join(sorted(frame["strategy_id"].dropna().astype(str).unique()))
        raise ValueError(f"No PnL rows for strategy_id={strategy_id}. Available: {available}")
    return strategy.sort_values("timestamp").reset_index(drop=True)


def load_btc_prices(path: Path) -> pd.DataFrame:
    require_file(path, "Binance 1m klines")
    frame = pd.read_csv(path)
    required = {"open_time_utc", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[["open_time_utc", "open", "high", "low", "close", "volume"]].copy()
    frame = frame.rename(columns={"open_time_utc": "timestamp", "close": "btc_close"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in ["open", "high", "low", "btc_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["timestamp", "btc_close"]).sort_values("timestamp").reset_index(drop=True)


def load_regimes(path: Path) -> pd.DataFrame:
    require_file(path, "Hourly volatility segments")
    frame = pd.read_csv(path)
    required = {
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "primary_volatility_band",
        "realized_volatility",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "rolling_percentile_rank",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    keep = [
        "event_ticker",
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "primary_volatility_band",
        "realized_volatility",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "rolling_percentile_rank",
    ]
    frame = frame[keep].copy()
    frame["forecast_hour_start_utc"] = pd.to_datetime(frame["forecast_hour_start_utc"], utc=True)
    frame["event_datetime_utc"] = pd.to_datetime(frame["event_datetime_utc"], utc=True)
    for column in [
        "realized_volatility",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "rolling_percentile_rank",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["forecast_hour_start_utc", "event_datetime_utc", "primary_volatility_band"]).sort_values(
        "forecast_hour_start_utc"
    ).reset_index(drop=True)


def merge_chart_timeseries(pnl: pd.DataFrame, prices: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    start = pnl["timestamp"].min()
    end = pnl["timestamp"].max()
    prices = prices[(prices["timestamp"] >= start) & (prices["timestamp"] <= end)].copy()
    prices["forecast_hour_start_utc"] = prices["timestamp"].dt.floor("h")

    regime_cols = [
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "primary_volatility_band",
        "realized_volatility",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "rolling_percentile_rank",
    ]
    chart = prices.merge(regimes[regime_cols], on="forecast_hour_start_utc", how="left", validate="many_to_one")
    chart = chart.merge(
        pnl[
            [
                "timestamp",
                "strategy_id",
                "strategy_label",
                "period_net_pnl",
                "period_fees",
                "period_trades",
                "cumulative_net_pnl",
            ]
        ],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    chart[["period_net_pnl", "period_fees", "period_trades"]] = chart[
        ["period_net_pnl", "period_fees", "period_trades"]
    ].fillna(0.0)
    chart["cumulative_net_pnl"] = chart["cumulative_net_pnl"].ffill()
    chart = chart.dropna(subset=["cumulative_net_pnl"]).reset_index(drop=True)
    return chart


def build_regime_segments(chart: pd.DataFrame) -> pd.DataFrame:
    if chart.empty:
        return pd.DataFrame()
    work = chart[["timestamp", "primary_volatility_band"]].dropna().copy()
    work["next_timestamp"] = work["timestamp"].shift(-1)
    work["segment_break"] = (
        (work["primary_volatility_band"] != work["primary_volatility_band"].shift(1))
        | (work["timestamp"].diff().dt.total_seconds().fillna(60) > 60)
    )
    work["segment_id"] = work["segment_break"].cumsum()
    rows: List[Dict[str, Any]] = []
    for _segment_id, part in work.groupby("segment_id", sort=True):
        end = part["next_timestamp"].dropna().max()
        if pd.isna(end):
            end = part["timestamp"].max() + pd.Timedelta(minutes=1)
        rows.append(
            {
                "segment_start": part["timestamp"].min(),
                "segment_end": end,
                "primary_volatility_band": str(part["primary_volatility_band"].iloc[0]),
                "minutes": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def summarize_regimes(chart: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        chart.groupby("primary_volatility_band", as_index=False)
        .agg(
            minutes=("timestamp", "nunique"),
            period_net_pnl=("period_net_pnl", "sum"),
            period_fees=("period_fees", "sum"),
            period_trades=("period_trades", "sum"),
            mean_btc_close=("btc_close", "mean"),
            mean_realized_volatility=("realized_volatility", "mean"),
            mean_rolling_percentile_rank=("rolling_percentile_rank", "mean"),
        )
        .sort_values("primary_volatility_band")
        .reset_index(drop=True)
    )
    grouped["pnl_per_trade"] = grouped["period_net_pnl"] / grouped["period_trades"].replace(0, np.nan)
    grouped["trades_per_minute"] = grouped["period_trades"] / grouped["minutes"].replace(0, np.nan)
    return grouped


def plotly_theme(fig: Any, *, height: int, top_margin: int = 92) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1b26",
        plot_bgcolor="#0d1b26",
        font=dict(color="#e8f1fa"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.10,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=62, r=34, t=top_margin, b=54),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")
    fig.update_yaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")


def add_regime_shapes(fig: Any, segments: pd.DataFrame, *, rows: Sequence[int]) -> List[Dict[str, Any]]:
    shape_roles: List[Dict[str, Any]] = []
    for _, row in segments.iterrows():
        regime = str(row["primary_volatility_band"])
        color = REGIME_COLORS.get(regime, "rgba(159, 180, 199, 0.10)")
        for target_row in rows:
            before_count = len(fig.layout.shapes) if fig.layout.shapes else 0
            fig.add_vrect(
                x0=row["segment_start"],
                x1=row["segment_end"],
                fillcolor=color,
                line_width=0,
                opacity=1.0,
                layer="below",
                row=target_row,
                col=1,
                exclude_empty_subplots=False,
            )
            role = "btc_highlight" if target_row == 2 else "pnl_highlight" if target_row == 3 else "other_highlight"
            after_count = len(fig.layout.shapes) if fig.layout.shapes else 0
            for shape_index in range(before_count, after_count):
                fig.layout.shapes[shape_index].name = role
                shape_roles.append({"index": shape_index, "role": role})
    return shape_roles


def add_highlight_toggle_menu(fig: Any, shape_roles: Sequence[Dict[str, Any]]) -> None:
    if not shape_roles:
        return

    def visibility_update(*, show_btc: bool, show_pnl: bool) -> Dict[str, bool]:
        update: Dict[str, bool] = {}
        for shape in shape_roles:
            role = shape["role"]
            if role == "btc_highlight":
                visible = show_btc
            elif role == "pnl_highlight":
                visible = show_pnl
            else:
                visible = True
            update[f"shapes[{shape['index']}].visible"] = visible
        return update

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                bgcolor="#14283a",
                bordercolor="#35516c",
                borderwidth=1,
                font=dict(color="#e8f1fa", size=12),
                buttons=[
                    dict(
                        label="Highlights: BTC + PnL",
                        method="relayout",
                        args=[visibility_update(show_btc=True, show_pnl=True)],
                    ),
                    dict(
                        label="BTC only",
                        method="relayout",
                        args=[visibility_update(show_btc=True, show_pnl=False)],
                    ),
                    dict(
                        label="PnL only",
                        method="relayout",
                        args=[visibility_update(show_btc=False, show_pnl=True)],
                    ),
                    dict(
                        label="Highlights off",
                        method="relayout",
                        args=[visibility_update(show_btc=False, show_pnl=False)],
                    ),
                ],
            )
        ]
    )


def build_chart(chart: pd.DataFrame, segments: pd.DataFrame, strategy_label: str) -> Any:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.10, 0.48, 0.42],
        vertical_spacing=0.035,
        subplot_titles=(
            "Post-Hoc Volatility Regime",
            "BTCUSDT 1m Close",
            f"{strategy_label}: Cumulative Net PnL After Fees",
        ),
    )

    shape_roles = add_regime_shapes(fig, segments, rows=[2, 3])

    regime_y = {
        "Low Volatility": 0,
        "Standard Volatility": 1,
        "High Volatility": 2,
    }
    regime_names = list(regime_y.keys())
    for regime in regime_names:
        part = chart[chart["primary_volatility_band"] == regime]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=part["timestamp"],
                y=[regime_y[regime]] * len(part),
                mode="markers",
                name=regime,
                marker=dict(color=REGIME_LINE_COLORS[regime], size=5, symbol="square"),
                hovertemplate=(
                    "timestamp=%{x}<br>"
                    f"regime={regime}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scattergl(
            x=chart["timestamp"],
            y=chart["btc_close"],
            mode="lines",
            name="BTCUSDT close",
            line=dict(color="#e8f1fa", width=1.7),
            hovertemplate="BTC close=%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=chart["timestamp"],
            y=chart["cumulative_net_pnl"],
            mode="lines",
            name="Cumulative net PnL",
            line=dict(color="#fbbf24", width=2.4),
            hovertemplate="cum net PnL=%{y:,.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    period_pnl = chart[chart["period_net_pnl"] != 0.0]
    fig.add_trace(
        go.Bar(
            x=period_pnl["timestamp"],
            y=period_pnl["period_net_pnl"],
            name="Minute net PnL",
            marker_color=np.where(period_pnl["period_net_pnl"] >= 0, "#35c7b7", "#ff8f70"),
            opacity=0.34,
            hovertemplate="minute net PnL=%{y:,.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Low", "Medium", "High"],
        range=[-0.6, 2.6],
        title_text="Regime",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="BTC price", row=2, col=1)
    fig.update_yaxes(title_text="Net PnL", row=3, col=1)
    fig.update_xaxes(title_text="Timestamp", row=3, col=1)
    fig.update_layout(
        title=(
            "Model N Post-Hoc Volatility Regime Overlay: BTC Price Above Net PnL"
        )
    )
    plotly_theme(fig, height=980, top_margin=140)
    add_highlight_toggle_menu(fig, shape_roles)
    return fig


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
        if any(token in lower for token in ["minutes", "trades"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=0))
        elif any(token in lower for token in ["rank"]):
            out[column] = out[column].map(lambda v: format_number(v, percent=True))
        elif any(token in lower for token in ["pnl", "fee", "price", "volatility"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def figures_to_html(figures: Sequence[Any]) -> str:
    if not HAS_PLOTLY:
        return "<section class='panel'><p>Plotly is not available in this environment.</p></section>"
    return "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False)
        for idx, fig in enumerate(figures)
    )


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
.hero { padding: 28px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 24px; }
.eyebrow {
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0;
  font-size: 12px;
  margin: 0 0 8px;
}
h1 { font-size: clamp(30px, 4vw, 52px); line-height: 1; margin: 0 0 14px; letter-spacing: 0; }
h2 { font-size: 22px; margin: 34px 0 14px; letter-spacing: 0; }
.lead { max-width: 1040px; color: var(--muted); line-height: 1.55; margin: 0; }
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
table.data-table { border-collapse: collapse; width: 100%; min-width: 760px; font-size: 13px; }
table.data-table th {
  background: var(--panel-2);
  color: var(--text);
  text-align: left;
  font-weight: 700;
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
}
table.data-table td {
  border-bottom: 1px solid rgba(53, 81, 108, 0.55);
  padding: 8px 10px;
  color: #dbe8f3;
  vertical-align: top;
}
code { color: #bceee8; }
</style>
"""


def dashboard_cards(chart: pd.DataFrame, summary: pd.DataFrame, strategy_label: str) -> str:
    final_pnl = float(chart["cumulative_net_pnl"].iloc[-1])
    high = summary[summary["primary_volatility_band"] == "High Volatility"]
    high_pnl = float(high["period_net_pnl"].iloc[0]) if not high.empty else np.nan
    cards = [
        ("Strategy", strategy_label),
        ("Final Net PnL", format_number(final_pnl, digits=3, signed=True)),
        ("High-Regime PnL", format_number(high_pnl, digits=3, signed=True)),
        ("Chart Minutes", format_number(chart["timestamp"].nunique(), digits=0)),
        ("BTC Start", format_number(chart["btc_close"].iloc[0], digits=2)),
        ("BTC End", format_number(chart["btc_close"].iloc[-1], digits=2)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def build_dashboard(chart: pd.DataFrame, segments: pd.DataFrame, summary: pd.DataFrame, strategy_label: str) -> str:
    figures = [build_chart(chart, segments, strategy_label)]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model N Post-Hoc Regime BTC/PnL Chart</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model N Post-Hoc Regime Overlay</p>
    <h1>BTC Price Above Net PnL</h1>
    <p class="lead">
      BTC price is Binance 1-minute close. Regimes are the existing post-hoc hourly
      <code>primary_volatility_band</code> labels from the volatility decomposition, not a causal live
      classifier. PnL is cumulative after-fee net PnL from the Model N trade diagnostics time series.
    </p>
    <div class="cards">{dashboard_cards(chart, summary, strategy_label)}</div>
  </section>

  <h2>Chart</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Regime PnL Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(summary))}</div></section>
</main>
</body>
</html>
"""


def write_readme(output_dir: Path, *, strategy_id: str, strategy_label: str) -> None:
    text = f"""Model N Post-Hoc Regime BTC/PnL Chart

Strategy:
- {strategy_id}: {strategy_label}

Inputs:
- BTC price: Binance 1m close from Model_K_Volatility_Decomposition_RT_outputs/binance_1m_klines.csv
- PnL: after-fee Model N diagnostics pnl_timeseries.csv
- Regimes: post-hoc primary_volatility_band from hourly_market_volatility_segments.csv

Important:
- These volatility regimes are post-hoc labels and are not causal live trading signals.
- The chart is for visual diagnostics only.

Outputs:
- posthoc_regime_btc_pnl_dashboard.html
- posthoc_regime_btc_pnl_chart_timeseries.csv
- posthoc_regime_segments.csv
- posthoc_regime_pnl_summary.csv
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required to build the post-hoc regime PnL chart.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pnl = load_pnl(args.pnl_timeseries.resolve(), args.strategy_id)
    prices = load_btc_prices(args.binance_klines.resolve())
    regimes = load_regimes(args.volatility_segments.resolve())
    chart = merge_chart_timeseries(pnl, prices, regimes)
    if chart.empty:
        raise ValueError("Merged chart time series is empty.")
    segments = build_regime_segments(chart)
    summary = summarize_regimes(chart)
    strategy_label = str(pnl["strategy_label"].dropna().iloc[0])

    chart.to_csv(output_dir / "posthoc_regime_btc_pnl_chart_timeseries.csv", index=False)
    segments.to_csv(output_dir / "posthoc_regime_segments.csv", index=False)
    summary.to_csv(output_dir / "posthoc_regime_pnl_summary.csv", index=False)
    dashboard = build_dashboard(chart, segments, summary, strategy_label)
    (output_dir / "posthoc_regime_btc_pnl_dashboard.html").write_text(dashboard, encoding="utf-8")
    write_readme(output_dir, strategy_id=args.strategy_id, strategy_label=strategy_label)

    print(f"Post-hoc regime BTC/PnL chart output directory: {output_dir}")
    print(summary.to_string(index=False))
    print(f"Dashboard: {output_dir / 'posthoc_regime_btc_pnl_dashboard.html'}")


if __name__ == "__main__":
    main()
