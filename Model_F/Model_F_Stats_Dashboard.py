from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_K.Model_K import (  # noqa: E402
    build_brier_decomposition,
    build_metrics_summary,
    build_sharpness,
    build_time_bucket_outputs,
    dataframe_to_html_table,
    expanded_calibration_error,
)

try:  # noqa: E402
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    go = None
    make_subplots = None
    HAS_PLOTLY = False


OUTPUT_FOLDER_NAME = "Model_F_Stats_Dashboard_Outputs"

COLOR_A = "#6cb6ff"
COLOR_K = "#ff8f70"
COLOR_PERFECT = "#8ea3b7"
COLOR_GRID = "#22384a"
COLOR_AXIS = "#35516c"
COLOR_PAPER = "#0d1b26"
COLOR_BG = "#07131d"
COLOR_PANEL = "#10202d"
COLOR_PANEL_ALT = "#142736"
COLOR_TEXT = "#e8f1fa"
COLOR_MUTED = "#96aabd"
COLOR_GOOD = "#35c7b7"
COLOR_WARN = "#fbbf24"


def default_model_f_eval_dir(root: Path) -> Path:
    return root / "Model_F" / "model_F_Evals_Outputs"


def default_model_k_eval_dir(root: Path) -> Path:
    return root / "Model_K" / "Model_K_outputs"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dark-mode internal and comparative dashboards for Model F using the "
            "evaluation output folders."
        )
    )
    parser.add_argument(
        "--model-f-eval-dir",
        type=Path,
        default=default_model_f_eval_dir(REPO_ROOT),
        help="Directory containing Model_F evaluation outputs.",
    )
    parser.add_argument(
        "--model-k-eval-dir",
        type=Path,
        default=default_model_k_eval_dir(REPO_ROOT),
        help="Directory containing Model_K evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where dashboard HTML files will be written.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10, help="Calibration bins for recomputed comparisons.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Classification threshold for recomputed comparison summaries.",
    )
    parser.add_argument(
        "--pit-seed",
        type=int,
        default=42,
        help="Seed used for randomized PIT generation.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_eval_dir(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "raw": load_csv(eval_dir / "raw_values.csv"),
        "metrics": load_csv(eval_dir / "metrics_summary.csv"),
        "decomposition": load_csv(eval_dir / "brier_decomposition.csv"),
        "calibration": load_csv(eval_dir / "calibration_curve.csv"),
        "ece": load_csv(eval_dir / "expanded_calibration_error_summary.csv"),
        "sharpness": load_csv(eval_dir / "sharpness.csv"),
        "coverage": load_optional_csv(eval_dir / "outcome_join_coverage.csv"),
        "mismatches": load_optional_csv(eval_dir / "resolution_mismatches.csv"),
        "time_bucket_metrics": load_optional_csv(eval_dir / "time_bucket_metrics.csv"),
        "time_bucket_accuracy": load_optional_csv(eval_dir / "time_bucket_accuracy.csv"),
        "time_bucket_brier": load_optional_csv(eval_dir / "time_bucket_brier_decomposition.csv"),
        "time_bucket_calibration": load_optional_csv(eval_dir / "time_bucket_calibration_curve.csv"),
    }


def overall_row(frame: pd.DataFrame, segment_col: str = "segment") -> pd.Series:
    overall = frame.loc[frame[segment_col] == "overall"]
    if overall.empty:
        raise ValueError("Expected an overall row in the evaluation frame.")
    return overall.iloc[0]


def build_bundle_from_raw(
    raw: pd.DataFrame,
    *,
    calibration_bins: int,
    classification_threshold: float,
    coverage: Optional[pd.DataFrame] = None,
    mismatches: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    metrics = build_metrics_summary(raw, threshold=classification_threshold)
    decomposition = build_brier_decomposition(raw, bins=calibration_bins)
    calibration, ece = expanded_calibration_error(raw, bins=calibration_bins)
    sharpness = build_sharpness(raw)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = build_time_bucket_outputs(
        raw,
        calibration_bins=calibration_bins,
        threshold=classification_threshold,
    )
    return {
        "raw": raw,
        "metrics": metrics,
        "decomposition": decomposition,
        "calibration": calibration,
        "ece": ece,
        "sharpness": sharpness,
        "coverage": coverage if coverage is not None else pd.DataFrame(),
        "mismatches": mismatches if mismatches is not None else pd.DataFrame(),
        "time_bucket_metrics": time_bucket_metrics,
        "time_bucket_accuracy": time_bucket_accuracy,
        "time_bucket_brier": time_bucket_brier,
        "time_bucket_calibration": time_bucket_calibration,
        "overall_metrics": overall_row(metrics),
        "overall_decomposition": overall_row(decomposition),
        "overall_ece": overall_row(ece),
        "overall_sharpness": overall_row(sharpness),
    }


def add_compare_key(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["compare_key"] = out["event_contract_id"].astype(str) + "|" + out["forecast_datetime_utc"].astype(str)
    return out


def intersect_raw_rows(raw_a: pd.DataFrame, raw_k: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = add_compare_key(raw_a)
    right = add_compare_key(raw_k)
    common = set(left["compare_key"]).intersection(set(right["compare_key"]))
    left = left[left["compare_key"].isin(common)].copy()
    right = right[right["compare_key"].isin(common)].copy()
    return left.drop(columns=["compare_key"]), right.drop(columns=["compare_key"])


def compute_randomized_pit(raw: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = raw["p_kalshi"].clip(0.0, 1.0).to_numpy(dtype=float)
    y = raw["outcome"].to_numpy(dtype=int)
    u = rng.random(len(raw))
    pit = np.where(y == 1, p + (1.0 - p) * u, p * u)
    return pit.clip(0.0, 1.0)


def qq_points(values: np.ndarray, max_points: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    ordered = np.sort(values)
    idx = np.linspace(0, values.size - 1, min(max_points, values.size), dtype=int)
    empirical = ordered[idx]
    theoretical = (idx + 0.5) / values.size
    return theoretical, empirical


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
        if "rows" in lower or "contracts" in lower or "count" in lower or "mismatch" in lower:
            out[column] = out[column].map(lambda v: format_number(v, digits=0))
        elif lower.endswith("_pct") or "accuracy" in lower or "rate" in lower:
            out[column] = out[column].map(lambda v: format_number(v, digits=4))
        elif any(token in lower for token in ["brier", "loss", "ece", "error", "variance", "std", "share", "frequency", "probability", "base_rate", "mean_p"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4))
        else:
            out[column] = out[column].map(lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v)
    return out


def render_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    return dataframe_to_html_table(format_table(df), max_rows=max_rows)


def dashboard_style() -> str:
    return f"""<style>
    :root {{
      color-scheme: dark;
      --bg: {COLOR_BG};
      --bg-2: #0b1823;
      --panel: {COLOR_PANEL};
      --panel-2: {COLOR_PANEL_ALT};
      --line: #22394b;
      --line-soft: rgba(255,255,255,0.05);
      --ink: {COLOR_TEXT};
      --muted: {COLOR_MUTED};
      --blue: {COLOR_A};
      --orange: {COLOR_K};
      --good: {COLOR_GOOD};
      --warn: {COLOR_WARN};
      --pill: #122736;
    }}
    html {{
      background: var(--bg);
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(108, 182, 255, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 143, 112, 0.15), transparent 28%),
        linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 34px 24px 54px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    h2 {{
      margin: 36px 0 12px;
      font-size: 18px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      color: var(--ink);
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.5;
    }}
    a {{
      color: #9aceff;
    }}
    .hero {{
      padding: 24px 24px 20px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)),
        var(--panel);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 16px 40px rgba(0,0,0,0.18);
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #a9c7e2;
    }}
    .lead {{
      max-width: 920px;
      margin: 14px 0 0;
      font-size: 15px;
    }}
    .hero-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--pill);
      font-size: 13px;
      color: var(--ink);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.012)), var(--panel-2);
    }}
    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 30px;
      line-height: 1.05;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01)),
        var(--panel);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
      padding: 20px 20px 18px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.01);
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
      background: transparent;
    }}
    .data-table th,
    .data-table td {{
      padding: 10px 11px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    .data-table th {{
      background: var(--panel-2);
      color: var(--ink);
      font-weight: 650;
      position: sticky;
      top: 0;
    }}
    .data-table td {{
      color: var(--ink);
    }}
    .section-stack {{
      display: grid;
      gap: 18px;
    }}
    .chart-wrap {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      overflow: hidden;
    }}
    .chart-note {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    code {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 5px;
      color: var(--ink);
    }}
  </style>"""


def plotly_theme(
    fig: Any,
    *,
    height: int,
    barmode: Optional[str] = None,
    legend_y: float = 1.18,
    top_margin: int = 132,
) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
        legend=dict(
            orientation="h",
            x=0.0,
            y=legend_y,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=56, r=28, t=top_margin, b=48),
        height=height,
    )
    if barmode:
        fig.update_layout(barmode=barmode)
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)


def plotly_html(figures: Sequence[Any]) -> str:
    if not figures:
        return ""
    parts: List[str] = []
    for idx, fig in enumerate(figures):
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if idx == 0 else False))
    return "".join(parts)


def internal_charts_html(bundle: Dict[str, Any], pit: np.ndarray) -> str:
    if not HAS_PLOTLY:
        return "<div class='panel'><p>Plotly is not available in this environment.</p></div>"

    calibration = bundle["calibration"]
    calibration = calibration[(calibration["segment"] == "overall") & (calibration["n_forecasts"] > 0)].copy()
    overall_decomp = bundle["overall_decomposition"]
    metrics = bundle["metrics"].copy()
    contract_metrics = metrics[metrics["segment"] != "overall"].copy()
    time_deciles = bundle["time_bucket_metrics"]
    time_deciles = time_deciles[(time_deciles["bucket_type"] == "decile") & (time_deciles["n_forecasts"] > 0)].copy()

    theoretical, empirical = qq_points(pit)

    fig_top = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration Curve", "Model F Probability Distribution"),
        horizontal_spacing=0.10,
    )
    fig_top.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=COLOR_PERFECT, dash="dash"), name="Perfect"),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Scatter(
            x=calibration["mean_p_kalshi"],
            y=calibration["observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_A, width=3),
            marker=dict(color=COLOR_A, size=8),
            name="Model F",
        ),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Histogram(
            x=bundle["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            marker_color=COLOR_GOOD,
            name="Probabilities",
        ),
        row=1,
        col=2,
    )
    fig_top.update_xaxes(range=[0, 1], title_text="Mean Model F probability", row=1, col=1)
    fig_top.update_yaxes(range=[0, 1], title_text="Observed frequency", row=1, col=1)
    fig_top.update_xaxes(range=[0, 1], title_text="Model F probability", row=1, col=2)
    fig_top.update_yaxes(title_text="Count", row=1, col=2)
    fig_top.update_annotations(y=0.98)
    plotly_theme(fig_top, height=430)

    fig_mid = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Randomized PIT Histogram", "PIT QQ Plot vs Uniform"),
        horizontal_spacing=0.10,
    )
    fig_mid.add_trace(
        go.Histogram(
            x=pit,
            xbins=dict(start=0.0, end=1.0, size=0.05),
            marker_color=COLOR_K,
            name="PIT",
        ),
        row=1,
        col=1,
    )
    fig_mid.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=COLOR_PERFECT, dash="dash"),
            name="Uniform",
        ),
        row=1,
        col=2,
    )
    fig_mid.add_trace(
        go.Scatter(
            x=theoretical,
            y=empirical,
            mode="markers",
            marker=dict(color=COLOR_A, size=5, opacity=0.7),
            name="QQ",
        ),
        row=1,
        col=2,
    )
    fig_mid.update_xaxes(range=[0, 1], title_text="PIT value", row=1, col=1)
    fig_mid.update_yaxes(title_text="Count", row=1, col=1)
    fig_mid.update_xaxes(range=[0, 1], title_text="Uniform quantile", row=1, col=2)
    fig_mid.update_yaxes(range=[0, 1], title_text="Empirical PIT quantile", row=1, col=2)
    fig_mid.update_annotations(y=0.98)
    plotly_theme(fig_mid, height=430)

    fig_bottom = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Brier Decomposition", "Contract Label Metrics", "10-Minute Bucket Trend"),
        horizontal_spacing=0.10,
    )
    categories = ["brier_score", "reliability", "resolution", "uncertainty", "brier_from_decomposition"]
    fig_bottom.add_trace(
        go.Bar(
            x=["Brier", "Reliability", "Resolution", "Uncertainty", "From Decomp"],
            y=[float(overall_decomp[c]) for c in categories],
            marker_color=[COLOR_A, COLOR_K, COLOR_GOOD, COLOR_WARN, COLOR_A],
            name="Overall",
        ),
        row=1,
        col=1,
    )
    fig_bottom.add_trace(
        go.Bar(
            x=contract_metrics["segment"],
            y=contract_metrics["brier_score"],
            marker_color=COLOR_A,
            name="Brier by contract",
        ),
        row=1,
        col=2,
    )
    fig_bottom.add_trace(
        go.Scatter(
            x=time_deciles["display_name"],
            y=time_deciles["brier_score"],
            mode="lines+markers",
            line=dict(color=COLOR_A, width=3),
            marker=dict(color=COLOR_A, size=7),
            name="Brier by bucket",
        ),
        row=1,
        col=3,
    )
    fig_bottom.add_trace(
        go.Scatter(
            x=time_deciles["display_name"],
            y=time_deciles["classification_accuracy"],
            mode="lines+markers",
            line=dict(color=COLOR_GOOD, width=3),
            marker=dict(color=COLOR_GOOD, size=7),
            name="Accuracy by bucket",
            yaxis="y2",
        ),
        row=1,
        col=3,
    )
    fig_bottom.update_yaxes(title_text="Metric value", row=1, col=1)
    fig_bottom.update_yaxes(title_text="Brier score", row=1, col=2)
    fig_bottom.update_yaxes(title_text="Brier score", row=1, col=3)
    fig_bottom.update_layout(
        yaxis4=dict(
            title="Accuracy",
            overlaying="y3",
            side="right",
            range=[0, 1],
            showgrid=False,
            color=COLOR_GOOD,
        )
    )
    fig_bottom.update_annotations(y=0.98)
    plotly_theme(fig_bottom, height=430, legend_y=1.22, top_margin=136)

    return (
        "<div class='section-stack'>"
        f"<div class='chart-wrap'>{plotly_html([fig_top, fig_mid, fig_bottom])}</div>"
        "</div>"
    )


def comparison_charts_html(bundle_a: Dict[str, Any], bundle_k: Dict[str, Any]) -> str:
    if not HAS_PLOTLY:
        return "<div class='panel'><p>Plotly is not available in this environment.</p></div>"

    cal_a = bundle_a["calibration"]
    cal_a = cal_a[(cal_a["segment"] == "overall") & (cal_a["n_forecasts"] > 0)].copy()
    cal_k = bundle_k["calibration"]
    cal_k = cal_k[(cal_k["segment"] == "overall") & (cal_k["n_forecasts"] > 0)].copy()

    dec_a = bundle_a["overall_decomposition"]
    dec_k = bundle_k["overall_decomposition"]
    time_a = bundle_a["time_bucket_metrics"]
    time_a = time_a[(time_a["bucket_type"] == "decile") & (time_a["n_forecasts"] > 0)].copy()
    time_k = bundle_k["time_bucket_metrics"]
    time_k = time_k[(time_k["bucket_type"] == "decile") & (time_k["n_forecasts"] > 0)].copy()

    fig_top = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration Overlay", "Probability Distribution Overlay"),
        horizontal_spacing=0.10,
    )
    fig_top.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=COLOR_PERFECT, dash="dash"), name="Perfect"),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Scatter(
            x=cal_a["mean_p_kalshi"],
            y=cal_a["observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_A, width=3),
            marker=dict(color=COLOR_A, size=8),
            name="Model F",
        ),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Scatter(
            x=cal_k["mean_p_kalshi"],
            y=cal_k["observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_K, width=3),
            marker=dict(color=COLOR_K, size=8),
            name="Model K",
        ),
        row=1,
        col=1,
    )
    fig_top.add_trace(
        go.Histogram(
            x=bundle_a["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=COLOR_A,
            name="Model F",
        ),
        row=1,
        col=2,
    )
    fig_top.add_trace(
        go.Histogram(
            x=bundle_k["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=COLOR_K,
            name="Model K",
        ),
        row=1,
        col=2,
    )
    fig_top.update_xaxes(range=[0, 1], title_text="Mean probability", row=1, col=1)
    fig_top.update_yaxes(range=[0, 1], title_text="Observed frequency", row=1, col=1)
    fig_top.update_xaxes(range=[0, 1], title_text="Predicted probability", row=1, col=2)
    fig_top.update_yaxes(title_text="Count", row=1, col=2)
    fig_top.update_annotations(y=0.98)
    plotly_theme(fig_top, height=430, barmode="overlay", legend_y=1.22, top_margin=136)

    fig_bottom = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Brier Decomposition Comparison", "10-Minute Bucket Brier Trend"),
        horizontal_spacing=0.10,
    )
    labels = ["Brier", "Reliability", "Resolution", "Uncertainty", "From Decomp"]
    cols = ["brier_score", "reliability", "resolution", "uncertainty", "brier_from_decomposition"]
    fig_bottom.add_trace(
        go.Bar(x=labels, y=[float(dec_a[c]) for c in cols], marker_color=COLOR_A, name="Model F"),
        row=1,
        col=1,
    )
    fig_bottom.add_trace(
        go.Bar(x=labels, y=[float(dec_k[c]) for c in cols], marker_color=COLOR_K, name="Model K"),
        row=1,
        col=1,
    )
    fig_bottom.add_trace(
        go.Scatter(
            x=time_a["display_name"],
            y=time_a["brier_score"],
            mode="lines+markers",
            line=dict(color=COLOR_A, width=3),
            marker=dict(color=COLOR_A, size=7),
            name="Model F",
        ),
        row=1,
        col=2,
    )
    fig_bottom.add_trace(
        go.Scatter(
            x=time_k["display_name"],
            y=time_k["brier_score"],
            mode="lines+markers",
            line=dict(color=COLOR_K, width=3),
            marker=dict(color=COLOR_K, size=7),
            name="Model K",
        ),
        row=1,
        col=2,
    )
    fig_bottom.update_yaxes(title_text="Metric value", row=1, col=1)
    fig_bottom.update_yaxes(title_text="Brier score", row=1, col=2)
    fig_bottom.update_annotations(y=0.98)
    plotly_theme(fig_bottom, height=430, barmode="group", legend_y=1.22, top_margin=136)

    return (
        "<div class='section-stack'>"
        f"<div class='chart-wrap'>{plotly_html([fig_top, fig_bottom])}</div>"
        "</div>"
    )


def build_internal_dashboard_html(bundle: Dict[str, Any], pit: np.ndarray, notes: Sequence[str]) -> str:
    overall = bundle["overall_metrics"]
    ece = bundle["overall_ece"]
    sharp = bundle["overall_sharpness"]
    mismatch_count = len(bundle["mismatches"])
    cards = [
        ("Scored rows", format_number(overall["n_forecasts"], digits=0)),
        ("Event contracts", format_number(overall["n_event_contracts"], digits=0)),
        ("Brier score", format_number(overall["brier_score"])),
        ("Log loss", format_number(overall["log_loss"])),
        ("ECE", format_number(ece["expected_calibration_error"])),
        ("RMSECE", format_number(ece["root_mean_squared_calibration_error"])),
        ("Forecast std", format_number(sharp["forecast_std"])),
        ("Audit mismatches", format_number(mismatch_count, digits=0)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)

    by_contract = bundle["metrics"][bundle["metrics"]["segment"] != "overall"].copy()
    decomp = bundle["decomposition"].copy()
    ece_table = bundle["ece"].copy()
    sharpness = bundle["sharpness"].copy()
    time_metrics = bundle["time_bucket_metrics"].copy()
    time_metrics = time_metrics[time_metrics["n_forecasts"] > 0]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model F Internal Dashboard</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Internal Model Fvaluation</p>
    <h1>Model F Probabilistic Diagnostics</h1>
    <p class="lead">
      This dashboard summarizes Model F as a probabilistic forecaster on its own scored output universe,
      including calibration, randomized PIT, QQ behavior versus uniformity, Brier decomposition, and
      minute-bucket performance slices.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Core Visual Diagnostics</h2>
  {internal_charts_html(bundle, pit)}

  <h2>Metric Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(bundle["metrics"])}</div>
  </section>

  <h2>Per-Contract Comparison</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(by_contract)}</div>
  </section>

  <h2>Brier Decomposition</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(decomp)}</div>
  </section>

  <h2>Expanded Calibration Error</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(ece_table)}</div>
  </section>

  <h2>Sharpness</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(sharpness)}</div>
  </section>

  <h2>Time Bucket Metrics</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(time_metrics, max_rows=20)}</div>
  </section>

  <h2>Notes</h2>
  <section class="panel">
    <ul>{notes_html}</ul>
  </section>
</main>
</body>
</html>
"""


def overview_comparison_table(bundle_a: Dict[str, Any], bundle_k: Dict[str, Any]) -> pd.DataFrame:
    oa = bundle_a["overall_metrics"]
    ok = bundle_k["overall_metrics"]
    ea = bundle_a["overall_ece"]
    ek = bundle_k["overall_ece"]
    rows = [
        ("Scored rows", float(oa["n_forecasts"]), float(ok["n_forecasts"])),
        ("Event contracts", float(oa["n_event_contracts"]), float(ok["n_event_contracts"])),
        ("Brier score", float(oa["brier_score"]), float(ok["brier_score"])),
        ("Log loss", float(oa["log_loss"]), float(ok["log_loss"])),
        ("ECE", float(ea["expected_calibration_error"]), float(ek["expected_calibration_error"])),
        ("Forecast std", float(bundle_a["overall_sharpness"]["forecast_std"]), float(bundle_k["overall_sharpness"]["forecast_std"])),
    ]
    return pd.DataFrame(
        [
            {"metric": metric, "model_f": left, "model_k": right, "delta_e_minus_k": left - right}
            for metric, left, right in rows
        ]
    )


def build_comparison_table(
    *,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    left = left_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    right = right_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    merged = left.merge(right, on=list(key_cols), how="outer", suffixes=(f"_{left_label}", f"_{right_label}"))
    ordered = list(key_cols)
    for col in value_cols:
        lcol = f"{col}_{left_label}"
        rcol = f"{col}_{right_label}"
        dcol = f"{col}_delta"
        merged[dcol] = merged[lcol] - merged[rcol]
        ordered.extend([lcol, rcol, dcol])
    return merged[ordered].reset_index(drop=True)


def build_comparative_dashboard_html(bundle_a: Dict[str, Any], bundle_k: Dict[str, Any], notes: Sequence[str]) -> str:
    overview = overview_comparison_table(bundle_a, bundle_k)
    metrics_comp = build_comparison_table(
        left_df=bundle_a["metrics"],
        right_df=bundle_k["metrics"],
        key_cols=["segment"],
        value_cols=["n_forecasts", "n_event_contracts", "base_rate", "mean_p_kalshi", "brier_score", "log_loss", "classification_accuracy"],
        left_label="a",
        right_label="k",
    )
    decomp_comp = build_comparison_table(
        left_df=bundle_a["decomposition"],
        right_df=bundle_k["decomposition"],
        key_cols=["segment"],
        value_cols=["n_forecasts", "brier_score", "reliability", "resolution", "uncertainty", "brier_from_decomposition"],
        left_label="a",
        right_label="k",
    )
    ece_comp = build_comparison_table(
        left_df=bundle_a["ece"],
        right_df=bundle_k["ece"],
        key_cols=["segment"],
        value_cols=["n_forecasts", "expected_calibration_error", "root_mean_squared_calibration_error", "max_calibration_error"],
        left_label="a",
        right_label="k",
    )
    sharp_comp = build_comparison_table(
        left_df=bundle_a["sharpness"],
        right_df=bundle_k["sharpness"],
        key_cols=["segment"],
        value_cols=["n_forecasts", "base_rate", "mean_p_kalshi", "forecast_std", "sharpness_variance_from_base_rate", "mean_abs_distance_from_0_5", "mean_predictive_variance_p_times_1_minus_p"],
        left_label="a",
        right_label="k",
    )

    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    oa = bundle_a["overall_metrics"]
    ok = bundle_k["overall_metrics"]
    common_events = bundle_a["raw"]["event_ticker"].nunique()
    common_contracts = bundle_a["raw"]["event_contract_id"].nunique()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model F vs Model K Dashboard</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Comparative Model Fvaluation</p>
    <h1>Model F vs Model K</h1>
    <p class="lead">
      This dashboard compares Model F and Model K on the exact overlapping scored-row universe,
      using the dark visual style of the volatility comparison dashboard while focusing on direct
      model-versus-model performance.
    </p>
    <div class="hero-pills">
      <span class="pill">Common scored rows: {int(oa["n_forecasts"]):,}</span>
      <span class="pill">Common event contracts: {common_contracts:,}</span>
      <span class="pill">Common events: {common_events:,}</span>
      <span class="pill">Model F Brier: {format_number(oa["brier_score"])}</span>
      <span class="pill">Model K Brier: {format_number(ok["brier_score"])}</span>
    </div>
  </section>

  <h2>Topline Comparison</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(overview)}</div>
  </section>

  <h2>Chart Comparison</h2>
  {comparison_charts_html(bundle_a, bundle_k)}

  <h2>Metric Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(metrics_comp)}</div>
  </section>

  <h2>Brier Decomposition</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(decomp_comp)}</div>
  </section>

  <h2>Expanded Calibration Error</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(ece_comp)}</div>
  </section>

  <h2>Sharpness</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(sharp_comp)}</div>
  </section>

  <h2>Notes</h2>
  <section class="panel">
    <ul>{notes_html}</ul>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_f_eval_dir = args.model_f_eval_dir.resolve()
    model_k_eval_dir = args.model_k_eval_dir.resolve()

    loaded_b = load_eval_dir(model_f_eval_dir)
    loaded_k = load_eval_dir(model_k_eval_dir)

    bundle_a_internal = build_bundle_from_raw(
        loaded_b["raw"],
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
        coverage=loaded_b["coverage"],
        mismatches=loaded_b["mismatches"],
    )
    pit_values = compute_randomized_pit(bundle_a_internal["raw"], seed=args.pit_seed)
    internal_notes = [
        f"Source directory: {model_f_eval_dir}",
        f"Randomized PIT seed: {args.pit_seed}",
        "Internal dashboard uses the full Model F scored universe currently present in raw_values.csv.",
    ]
    internal_html = build_internal_dashboard_html(bundle_a_internal, pit_values, internal_notes)
    (output_dir / "model_f_internal_dashboard.html").write_text(internal_html, encoding="utf-8")

    aligned_b_raw, aligned_k_raw = intersect_raw_rows(loaded_b["raw"], loaded_k["raw"])
    if aligned_b_raw.empty or aligned_k_raw.empty:
        raise RuntimeError("No overlapping scored rows were found between Model F and Model K.")

    bundle_a_compare = build_bundle_from_raw(
        aligned_b_raw,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
    )
    bundle_k_compare = build_bundle_from_raw(
        aligned_k_raw,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
    )
    comparative_notes = [
        f"Model F source directory: {model_f_eval_dir}",
        f"Model K source directory: {model_k_eval_dir}",
        "Comparative dashboard restricts both models to the exact overlapping scored-row universe using event_contract_id + forecast_datetime_utc.",
        f"Overlap rows: {len(aligned_b_raw):,}; overlap event contracts: {aligned_b_raw['event_contract_id'].nunique():,}.",
    ]
    comparative_html = build_comparative_dashboard_html(bundle_a_compare, bundle_k_compare, comparative_notes)
    (output_dir / "model_f_vs_model_k_dashboard.html").write_text(comparative_html, encoding="utf-8")

    print(f"Internal dashboard: {output_dir / 'model_f_internal_dashboard.html'}")
    print(f"Comparative dashboard: {output_dir / 'model_f_vs_model_k_dashboard.html'}")


if __name__ == "__main__":
    main()
