from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from Exponential_Hybrid import (
    add_exponential_hybrid,
    build_all_strategy_trades,
    build_dashboard_html,
    build_eval_summary,
    build_overlap,
    build_pnl_timeseries,
    build_strategy_summary,
    dataframe_to_html_table,
    figures_to_html,
    format_number,
    format_table,
    load_raw_values,
    plotly_theme,
    repo_root_from_script,
    trade_columns,
    write_readme,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    go = None
    make_subplots = None
    HAS_PLOTLY = False


DEFAULT_LAMBDAS = [1.0, 1.5, 2.0, 2.5, 3.0]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Run an exponential-hybrid lambda sweep and compare PnL/eval metrics."
    )
    parser.add_argument(
        "--model-n-raw-values",
        type=Path,
        default=root / "Model_N" / "model_N_Evals_Outputs" / "raw_values.csv",
        help="Model N raw_values.csv containing p_normal, p_shock, q_shock, and outcomes.",
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
        default=Path(__file__).resolve().parent / "outputs",
        help="Self-contained output folder for this experiment.",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help="Lambda values to test.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum selected-side edge versus Model K required to trade.",
    )
    parser.add_argument(
        "--write-per-lambda-trades",
        action="store_true",
        help="Also write full per-lambda trade-minute CSVs. Disabled by default because these are large.",
    )
    return parser.parse_args()


def lambda_slug(lambda_value: float) -> str:
    return f"lambda_{lambda_value:g}".replace(".", "_")


def unique_lambdas(values: List[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        value = float(value)
        if value < 0.0:
            raise ValueError("--lambda-values must be non-negative.")
        if not any(abs(value - existing) < 1e-12 for existing in out):
            out.append(value)
    return sorted(out)


def run_lambda(
    *,
    model_n: pd.DataFrame,
    model_k: pd.DataFrame,
    lambda_value: float,
    edge_threshold: float,
) -> Dict[str, Any]:
    model_exp = add_exponential_hybrid(model_n, lambda_value=lambda_value)
    overlap = build_overlap(model_exp, model_k)
    eval_summary = build_eval_summary(overlap, lambda_value=lambda_value)
    trades = build_all_strategy_trades(overlap, lambda_value=lambda_value, edge_threshold=edge_threshold)
    timeseries = build_pnl_timeseries(trades)
    strategy_summary = build_strategy_summary(overlap, trades, timeseries)

    for frame in (eval_summary, trades, timeseries, strategy_summary):
        if "lambda_value" not in frame.columns:
            frame.insert(0, "lambda_value", lambda_value)
        if "lambda_slug" not in frame.columns:
            insert_at = 1 if "lambda_value" in frame.columns else 0
            frame.insert(insert_at, "lambda_slug", lambda_slug(lambda_value))

    return {
        "lambda_value": lambda_value,
        "lambda_slug": lambda_slug(lambda_value),
        "model_exp": model_exp,
        "overlap": overlap,
        "eval_summary": eval_summary,
        "trades": trades,
        "timeseries": timeseries,
        "strategy_summary": strategy_summary,
    }


def write_lambda_folder(
    *,
    result: Dict[str, Any],
    output_dir: Path,
    edge_threshold: float,
    write_trades: bool,
) -> None:
    lambda_value = float(result["lambda_value"])
    lambda_dir = output_dir / str(result["lambda_slug"])
    lambda_dir.mkdir(parents=True, exist_ok=True)

    result["eval_summary"].to_csv(lambda_dir / "exponential_hybrid_eval_summary.csv", index=False)
    result["strategy_summary"].to_csv(lambda_dir / "exponential_hybrid_strategy_summary.csv", index=False)
    result["timeseries"].to_csv(lambda_dir / "exponential_hybrid_strategy_pnl_timeseries.csv", index=False)

    if write_trades:
        result["trades"][trade_columns(result["trades"])].to_csv(lambda_dir / "exponential_hybrid_trade_minutes.csv", index=False)

    dashboard_html = build_dashboard_html(
        strategy_summary=result["strategy_summary"],
        eval_summary=result["eval_summary"],
        timeseries=result["timeseries"],
        lambda_value=lambda_value,
        edge_threshold=edge_threshold,
    )
    (lambda_dir / "exponential_hybrid_dashboard.html").write_text(dashboard_html, encoding="utf-8")

    if abs(lambda_value - 2.0) < 1e-12:
        result["model_exp"].to_csv(output_dir / "exponential_hybrid_raw_values.csv", index=False)
        result["eval_summary"].to_csv(output_dir / "exponential_hybrid_eval_summary.csv", index=False)
        result["strategy_summary"].to_csv(output_dir / "exponential_hybrid_strategy_summary.csv", index=False)
        result["timeseries"].to_csv(output_dir / "exponential_hybrid_strategy_pnl_timeseries.csv", index=False)
        result["trades"][trade_columns(result["trades"])].to_csv(output_dir / "exponential_hybrid_trade_minutes.csv", index=False)
        (output_dir / "exponential_hybrid_dashboard.html").write_text(dashboard_html, encoding="utf-8")
        write_readme(output_dir, lambda_value=lambda_value, edge_threshold=edge_threshold)


def add_baseline_deltas(lambda_comparison: pd.DataFrame, all_strategy_summary: pd.DataFrame) -> pd.DataFrame:
    out = lambda_comparison.copy()
    for strategy_id, column_name in [
        ("linear_hybrid", "delta_pnl_vs_linear_hybrid"),
        ("model_a", "delta_pnl_vs_model_a"),
        ("model_b", "delta_pnl_vs_model_b"),
    ]:
        baseline = all_strategy_summary.loc[
            all_strategy_summary["strategy_id"] == strategy_id,
            ["lambda_value", "total_realized_pnl"],
        ].rename(columns={"total_realized_pnl": f"{strategy_id}_total_realized_pnl"})
        out = out.merge(baseline, on="lambda_value", how="left", validate="one_to_one")
        out[column_name] = out["total_realized_pnl"] - out[f"{strategy_id}_total_realized_pnl"]
    return out


def build_lambda_figures(lambda_comparison: pd.DataFrame, all_timeseries: pd.DataFrame) -> List[Any]:
    if not HAS_PLOTLY:
        return []

    figures: List[Any] = []
    exp_ts = all_timeseries[all_timeseries["strategy_id"] == "exp_hybrid"].copy()
    if not exp_ts.empty:
        fig_overlay = go.Figure()
        for lambda_value, part in exp_ts.groupby("lambda_value", sort=True):
            part = part.sort_values("forecast_datetime_utc")
            fig_overlay.add_trace(
                go.Scatter(
                    x=part["forecast_datetime_utc"],
                    y=part["cumulative_realized_pnl"],
                    mode="lines",
                    name=f"lambda={lambda_value:g}",
                    line=dict(width=3),
                )
            )
        linear = all_timeseries[
            (all_timeseries["strategy_id"] == "linear_hybrid")
            & (all_timeseries["lambda_value"] == sorted(all_timeseries["lambda_value"].unique())[0])
        ].copy()
        if not linear.empty:
            fig_overlay.add_trace(
                go.Scatter(
                    x=linear["forecast_datetime_utc"],
                    y=linear["cumulative_realized_pnl"],
                    mode="lines",
                    name="Linear Hybrid baseline",
                    line=dict(width=3, dash="dash", color="#d0d7de"),
                )
            )
        fig_overlay.update_layout(title="Cumulative Realized PnL By Lambda")
        fig_overlay.update_xaxes(title_text="Forecast timestamp")
        fig_overlay.update_yaxes(title_text="Cumulative PnL")
        plotly_theme(fig_overlay, height=590, top_margin=92)
        figures.append(fig_overlay)

    if not lambda_comparison.empty:
        labels = [f"{value:g}" for value in lambda_comparison["lambda_value"]]
        fig_bars = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Total PnL", "Delta PnL Vs Linear", "ROI On Cost", "Max Drawdown"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )
        fig_bars.add_trace(go.Bar(x=labels, y=lambda_comparison["total_realized_pnl"], marker_color="#ff8f70"), row=1, col=1)
        fig_bars.add_trace(go.Bar(x=labels, y=lambda_comparison["delta_pnl_vs_linear_hybrid"], marker_color="#6cb6ff"), row=1, col=2)
        fig_bars.add_trace(go.Bar(x=labels, y=lambda_comparison["roi_on_cost"], marker_color="#35c7b7"), row=2, col=1)
        fig_bars.add_trace(go.Bar(x=labels, y=lambda_comparison["max_drawdown"], marker_color="#fbbf24"), row=2, col=2)
        fig_bars.update_xaxes(title_text="lambda", row=2, col=1)
        fig_bars.update_xaxes(title_text="lambda", row=2, col=2)
        fig_bars.update_yaxes(title_text="PnL", row=1, col=1)
        fig_bars.update_yaxes(title_text="PnL", row=1, col=2)
        fig_bars.update_yaxes(title_text="ROI", row=2, col=1)
        fig_bars.update_yaxes(title_text="Drawdown", row=2, col=2)
        plotly_theme(fig_bars, height=720, top_margin=96)
        figures.append(fig_bars)

    return figures


def build_lambda_dashboard_html(
    *,
    lambda_comparison: pd.DataFrame,
    all_strategy_summary: pd.DataFrame,
    all_eval_summary: pd.DataFrame,
    all_timeseries: pd.DataFrame,
    lambda_values: List[float],
    edge_threshold: float,
) -> str:
    best = lambda_comparison.loc[lambda_comparison["total_realized_pnl"].idxmax()]
    worst = lambda_comparison.loc[lambda_comparison["total_realized_pnl"].idxmin()]
    cards = [
        ("Best Lambda", f"{best['lambda_value']:g}"),
        ("Best PnL", format_number(best["total_realized_pnl"], digits=3, signed=True)),
        ("Best Delta Vs Linear", format_number(best["delta_pnl_vs_linear_hybrid"], digits=3, signed=True)),
        ("Worst PnL", format_number(worst["total_realized_pnl"], digits=3, signed=True)),
        ("Tested Values", format_number(len(lambda_values), digits=0)),
        ("Edge Threshold", format_number(edge_threshold, digits=4)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    lambda_text = ", ".join(f"{value:g}" for value in lambda_values)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exponential Hybrid Lambda Sweep</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Standalone Exponential Hybrid Sweep</p>
    <h1>Lambda Sweep: {html.escape(lambda_text)}</h1>
    <p class="lead">
      Each run computes <code>w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)</code>
      and <code>p_final = (1 - w(q)) * p_normal + w(q) * p_shock</code>. PnL uses the
      same favorable BUY_YES/BUY_NO rule against Model K with selected-side edge threshold
      <code>{edge_threshold}</code>.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Lambda Comparison Charts</h2>
  <section class="chart-wrap">{figures_to_html(build_lambda_figures(lambda_comparison, all_timeseries))}</section>

  <h2>Lambda PnL Comparison</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(lambda_comparison))}</div></section>

  <h2>All Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(all_strategy_summary))}</div></section>

  <h2>Eval Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(all_eval_summary))}</div></section>
</main>
</body>
</html>
"""


def dashboard_style() -> str:
    from Exponential_Hybrid import dashboard_style as base_dashboard_style

    return base_dashboard_style()


def write_sweep_readme(output_dir: Path, *, lambda_values: List[float], edge_threshold: float) -> None:
    lambda_lines = "\n".join(f"- lambda = {value:g}" for value in lambda_values)
    text = f"""Exponential Hybrid Lambda Sweep

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Formula:
- w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)
- p_final = (1 - w(q)) * p_normal + w(q) * p_shock

Lambda values:
{lambda_lines}

PnL rule:
- Compare the strategy probability against Model K as the entry price
- BUY_YES when p_strategy - p_model_k > {edge_threshold}
- BUY_NO when (1 - p_strategy) - (1 - p_model_k) > {edge_threshold}

Root sweep outputs:
- exponential_hybrid_lambda_comparison.csv
- exponential_hybrid_lambda_eval_summary.csv
- exponential_hybrid_lambda_strategy_summary.csv
- exponential_hybrid_lambda_pnl_timeseries.csv
- exponential_hybrid_lambda_comparison_dashboard.html

Per-lambda folders contain dashboard/summary/time-series files for each lambda.
The lambda=2 compatibility outputs are also written in this root outputs folder.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.edge_threshold < 0.0:
        raise ValueError("--edge-threshold must be non-negative.")

    lambda_values = unique_lambdas(args.lambda_values)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")

    results = [
        run_lambda(
            model_n=model_n,
            model_k=model_k,
            lambda_value=lambda_value,
            edge_threshold=args.edge_threshold,
        )
        for lambda_value in lambda_values
    ]

    for result in results:
        write_lambda_folder(
            result=result,
            output_dir=output_dir,
            edge_threshold=args.edge_threshold,
            write_trades=args.write_per_lambda_trades,
        )

    all_strategy_summary = pd.concat([result["strategy_summary"] for result in results], ignore_index=True)
    all_eval_summary = pd.concat([result["eval_summary"] for result in results], ignore_index=True)
    all_timeseries = pd.concat([result["timeseries"] for result in results], ignore_index=True)
    lambda_comparison = (
        all_strategy_summary.loc[all_strategy_summary["strategy_id"] == "exp_hybrid"]
        .copy()
        .sort_values("lambda_value")
        .reset_index(drop=True)
    )
    lambda_comparison = add_baseline_deltas(lambda_comparison, all_strategy_summary)

    lambda_comparison.to_csv(output_dir / "exponential_hybrid_lambda_comparison.csv", index=False)
    all_eval_summary.to_csv(output_dir / "exponential_hybrid_lambda_eval_summary.csv", index=False)
    all_strategy_summary.to_csv(output_dir / "exponential_hybrid_lambda_strategy_summary.csv", index=False)
    all_timeseries.to_csv(output_dir / "exponential_hybrid_lambda_pnl_timeseries.csv", index=False)

    dashboard_html = build_lambda_dashboard_html(
        lambda_comparison=lambda_comparison,
        all_strategy_summary=all_strategy_summary,
        all_eval_summary=all_eval_summary,
        all_timeseries=all_timeseries,
        lambda_values=lambda_values,
        edge_threshold=args.edge_threshold,
    )
    (output_dir / "exponential_hybrid_lambda_comparison_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    write_sweep_readme(output_dir, lambda_values=lambda_values, edge_threshold=args.edge_threshold)

    print(f"Lambdas: {', '.join(f'{value:g}' for value in lambda_values)}")
    print(
        lambda_comparison[
            [
                "lambda_value",
                "signal_rows",
                "signal_timestamps",
                "total_realized_pnl",
                "delta_pnl_vs_linear_hybrid",
                "roi_on_cost",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"Dashboard: {output_dir / 'exponential_hybrid_lambda_comparison_dashboard.html'}")
    print(f"Comparison summary: {output_dir / 'exponential_hybrid_lambda_comparison.csv'}")


if __name__ == "__main__":
    main()
