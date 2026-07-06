from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_FOLDER_NAME = "Model_M_Isotonic_Diagnostics_Outputs"
EPS = 1e-12
PROBABILITY_EDGES = np.linspace(0.0, 1.0, 11)
PROBABILITY_LABELS = [f"{PROBABILITY_EDGES[i]:.1f}-{PROBABILITY_EDGES[i + 1]:.1f}" for i in range(10)]

COLOR_RAW = "#4fd1c5"
COLOR_CAL = "#ff8f70"
COLOR_OBS = "#6cb6ff"
COLOR_WARN = "#fbbf24"
COLOR_GRID = "#22384a"
COLOR_AXIS = "#35516c"
COLOR_PAPER = "#0d1b26"
COLOR_BG = "#07131d"
COLOR_PANEL = "#10202d"
COLOR_TEXT = "#e8f1fa"
COLOR_MUTED = "#96aabd"


def default_eval_dir(root: Path) -> Path:
    return root / "Model_M" / "model_M_Evals_Outputs"


def default_diagnostics_dir(root: Path) -> Path:
    return root / "Model_M" / OUTPUT_FOLDER_NAME


def default_volatility_segments_csv(root: Path) -> Optional[Path]:
    candidates = [
        root
        / "Model_M_Volatility_Decomposition_RT"
        / "Model_M_Volatility_Decomposition_RT_outputs"
        / "hourly_market_volatility_segments.csv",
        root
        / "Model_M_Volatility_Results_Dashboard"
        / "Model_M_Volatility_Dashboards_outputs"
        / "hourly_market_volatility_segments.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Model M isotonic calibration diagnostics from raw-vs-calibrated outputs."
    )
    parser.add_argument(
        "--model-m-eval-dir",
        type=Path,
        default=default_eval_dir(REPO_ROOT),
        help="Directory containing Model_M_Eval.py outputs, especially raw_values.csv.",
    )
    parser.add_argument("--model-f-eval-dir", dest="model_m_eval_dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--isotonic-diagnostics-dir",
        type=Path,
        default=default_diagnostics_dir(REPO_ROOT),
        help="Directory containing Model_M.py isotonic diagnostic CSVs.",
    )
    parser.add_argument(
        "--volatility-segments-csv",
        type=Path,
        default=default_volatility_segments_csv(REPO_ROOT),
        help="Optional hourly_market_volatility_segments.csv used for volatility-slice diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_diagnostics_dir(REPO_ROOT),
        help="Directory where the diagnostics dashboard and summary CSVs will be written.",
    )
    return parser.parse_args()


def log_loss(probability: pd.Series, outcome: pd.Series) -> pd.Series:
    p = probability.clip(EPS, 1.0 - EPS)
    y = outcome.astype(float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def add_probability_bins(frame: pd.DataFrame, column: str = "raw_probability") -> pd.DataFrame:
    out = frame.copy()
    out["raw_probability_bin"] = pd.cut(
        out[column],
        bins=PROBABILITY_EDGES,
        labels=PROBABILITY_LABELS,
        include_lowest=True,
        right=True,
    )
    out["raw_probability_band"] = np.select(
        [
            out[column] < 0.2,
            out[column] > 0.8,
        ],
        [
            "Low raw probability (<0.2)",
            "High raw probability (>0.8)",
        ],
        default="Middle raw probability (0.2-0.8)",
    )
    return out


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def standardize_backtest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename: Dict[str, str] = {}
    if "raw_probability" in out.columns and "p_model_raw" not in out.columns:
        rename["raw_probability"] = "p_model_raw"
    if "calibrated_probability" in out.columns and "p_model_calibrated" not in out.columns:
        rename["calibrated_probability"] = "p_model_calibrated"
    if "p_raw_pre_isotonic" in out.columns and "p_model_raw" not in out.columns:
        rename["p_raw_pre_isotonic"] = "p_model_raw"
    if "p_calibrated_isotonic" in out.columns and "p_model_calibrated" not in out.columns:
        rename["p_calibrated_isotonic"] = "p_model_calibrated"
    if "scored_probability" in out.columns and "p_kalshi" not in out.columns:
        rename["scored_probability"] = "p_kalshi"
    out = out.rename(columns=rename)
    return out


def load_backtest_frame(eval_dir: Path, warnings: List[str]) -> pd.DataFrame:
    diagnostics_path = eval_dir / "isotonic_forecast_diagnostics.csv"
    raw_path = eval_dir / "raw_values.csv"
    if diagnostics_path.exists():
        raw = pd.read_csv(diagnostics_path)
        source_name = "isotonic_forecast_diagnostics.csv"
    elif raw_path.exists():
        raw = pd.read_csv(raw_path)
        source_name = "raw_values.csv"
    else:
        raise FileNotFoundError(f"Neither isotonic_forecast_diagnostics.csv nor raw_values.csv was found in {eval_dir}")

    raw = standardize_backtest_columns(raw)
    required = {"p_model_raw", "p_model_calibrated", "outcome"}
    if not required.issubset(raw.columns):
        warnings.append(
            f"{source_name} does not contain raw/calibrated/outcome columns yet. "
            "Rerun Model_M.py and Model_M_Eval.py before using the backtest raw-vs-calibrated diagnostics."
        )
        return pd.DataFrame()

    out = add_backtest_error_columns(raw)
    if out.empty:
        warnings.append(
            f"{source_name} has the expected columns, but no non-null raw/calibrated rows. "
            "This usually means the copied seed outputs have not been regenerated with Model_M.py yet."
        )
    return out


def add_backtest_error_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["p_model_raw"] = pd.to_numeric(out["p_model_raw"], errors="coerce")
    out["p_model_calibrated"] = pd.to_numeric(out["p_model_calibrated"], errors="coerce")
    out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce")
    out = out.dropna(subset=["p_model_raw", "p_model_calibrated", "outcome"]).copy()
    out = out[out["outcome"].isin([0, 1])].copy()
    out["brier_raw"] = (out["p_model_raw"] - out["outcome"]) ** 2
    out["brier_calibrated"] = (out["p_model_calibrated"] - out["outcome"]) ** 2
    out["log_loss_raw"] = log_loss(out["p_model_raw"], out["outcome"])
    out["log_loss_calibrated"] = log_loss(out["p_model_calibrated"], out["outcome"])
    out["calibration_shift"] = out["p_model_calibrated"] - out["p_model_raw"]
    out = add_probability_bins(out, column="p_model_raw")
    if "minute_number" in out.columns:
        minute = pd.to_numeric(out["minute_number"], errors="coerce") + 1
        out["market_minute"] = minute.astype("Int64")
        out["minute_bucket"] = pd.cut(
            minute,
            bins=[0, 10, 20, 30, 40, 50, 60],
            labels=["Minutes 1-10", "Minutes 11-20", "Minutes 21-30", "Minutes 31-40", "Minutes 41-50", "Minutes 51-60"],
            include_lowest=True,
        )
    return out


def add_training_error_columns(training: pd.DataFrame) -> pd.DataFrame:
    out = training.copy()
    out["raw_probability"] = pd.to_numeric(out["raw_probability"], errors="coerce")
    out["calibrated_probability"] = pd.to_numeric(out["calibrated_probability"], errors="coerce")
    out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce")
    out = out.dropna(subset=["raw_probability", "calibrated_probability", "outcome"]).copy()
    out = out[out["outcome"].isin([0, 1])].copy()
    out["brier_raw"] = (out["raw_probability"] - out["outcome"]) ** 2
    out["brier_calibrated"] = (out["calibrated_probability"] - out["outcome"]) ** 2
    out["log_loss_raw"] = log_loss(out["raw_probability"], out["outcome"])
    out["log_loss_calibrated"] = log_loss(out["calibrated_probability"], out["outcome"])
    out["calibration_shift"] = out["calibrated_probability"] - out["raw_probability"]
    out = add_probability_bins(out, column="raw_probability")
    return out


def attach_volatility_context(raw: pd.DataFrame, volatility_segments_csv: Optional[Path]) -> pd.DataFrame:
    if volatility_segments_csv is None or not volatility_segments_csv.exists() or raw.empty:
        return raw
    segments = pd.read_csv(volatility_segments_csv)
    if "event_ticker" not in segments.columns:
        return raw
    keep = [
        column
        for column in [
            "event_ticker",
            "primary_volatility_band",
            "is_low_volatility",
            "is_standard_volatility",
            "is_high_volatility",
            "is_low_volatility_extreme",
            "is_high_volatility_extreme",
        ]
        if column in segments.columns
    ]
    if len(keep) <= 1:
        return raw
    out = raw.merge(segments[keep].drop_duplicates("event_ticker"), on="event_ticker", how="left")
    out["primary_volatility_band"] = out["primary_volatility_band"].fillna("Unknown")
    return out


def summarize_calibration(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    raw_col: str,
    calibrated_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(list(group_cols), observed=False, dropna=False)
    out = grouped.agg(
        n=("outcome", "size"),
        raw_mean=(raw_col, "mean"),
        calibrated_mean=(calibrated_col, "mean"),
        observed_frequency=("outcome", "mean"),
        mean_calibration_shift=("calibration_shift", "mean"),
        brier_raw=("brier_raw", "mean"),
        brier_calibrated=("brier_calibrated", "mean"),
        log_loss_raw=("log_loss_raw", "mean"),
        log_loss_calibrated=("log_loss_calibrated", "mean"),
    ).reset_index()
    out["raw_calibration_error"] = out["raw_mean"] - out["observed_frequency"]
    out["calibrated_calibration_error"] = out["calibrated_mean"] - out["observed_frequency"]
    out["absolute_raw_calibration_error"] = out["raw_calibration_error"].abs()
    out["absolute_calibrated_calibration_error"] = out["calibrated_calibration_error"].abs()
    out["absolute_calibration_error_delta"] = (
        out["absolute_calibrated_calibration_error"] - out["absolute_raw_calibration_error"]
    )
    out["brier_delta"] = out["brier_calibrated"] - out["brier_raw"]
    out["log_loss_delta"] = out["log_loss_calibrated"] - out["log_loss_raw"]
    return out.sort_values(list(group_cols)).reset_index(drop=True)


def compare_training_to_backtest(training_bin: pd.DataFrame, backtest_bin: pd.DataFrame) -> pd.DataFrame:
    if training_bin.empty or backtest_bin.empty:
        return pd.DataFrame()
    left = training_bin[
        [
            "raw_probability_bin",
            "n",
            "raw_mean",
            "calibrated_mean",
            "observed_frequency",
            "brier_delta",
        ]
    ].rename(
        columns={
            "n": "training_n",
            "raw_mean": "training_raw_mean",
            "calibrated_mean": "training_calibrated_mean",
            "observed_frequency": "training_observed_frequency",
            "brier_delta": "training_brier_delta",
        }
    )
    right = backtest_bin[
        [
            "raw_probability_bin",
            "n",
            "raw_mean",
            "calibrated_mean",
            "observed_frequency",
            "brier_delta",
        ]
    ].rename(
        columns={
            "n": "backtest_n",
            "raw_mean": "backtest_raw_mean",
            "calibrated_mean": "backtest_calibrated_mean",
            "observed_frequency": "backtest_observed_frequency",
            "brier_delta": "backtest_brier_delta",
        }
    )
    out = left.merge(right, on="raw_probability_bin", how="outer")
    out["observed_frequency_gap_backtest_minus_training"] = (
        out["backtest_observed_frequency"] - out["training_observed_frequency"]
    )
    out["raw_probability_mean_gap_backtest_minus_training"] = out["backtest_raw_mean"] - out["training_raw_mean"]
    return out


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}).fillna(False)


def high_volatility_extreme_backtest(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[bool_series(frame, "is_high_volatility_extreme")].copy()


def high_volatility_extreme_training(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "training_is_high_volatility_extreme" in frame.columns:
        return frame[bool_series(frame, "training_is_high_volatility_extreme")].copy()
    return frame[bool_series(frame, "is_high_volatility_extreme")].copy()


def write_csv(frame: pd.DataFrame, output_dir: Path, filename: str) -> None:
    if frame.empty:
        return
    frame.to_csv(output_dir / filename, index=False)


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.4f}"
    return html.escape(str(value))


def metric_cards(metrics: Dict[str, Any]) -> str:
    cards = []
    for label, value in metrics.items():
        cards.append(
            f"<div class='metric'><span>{html.escape(label)}</span><strong>{format_value(value)}</strong></div>"
        )
    return "<div class='metric-grid'>" + "".join(cards) + "</div>"


def render_table(frame: pd.DataFrame, *, max_rows: int = 24) -> str:
    if frame.empty:
        return "<p class='muted'>No rows available.</p>"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        elif pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{int(value):,}")
    return display.to_html(index=False, escape=True, classes="data-table")


def plotly_html(fig: Any) -> str:
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def apply_theme(fig: Any, *, height: int = 420) -> Any:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
        legend=dict(orientation="h", y=1.15, x=0.0),
        margin=dict(l=54, r=28, t=84, b=48),
        height=height,
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)
    return fig


def calibration_chart(summary: pd.DataFrame, title: str) -> str:
    if not HAS_PLOTLY or summary.empty:
        return ""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#8ea3b7", dash="dash"), name="Perfect"))
    fig.add_trace(
        go.Scatter(
            x=summary["raw_mean"],
            y=summary["observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_RAW, width=3),
            marker=dict(size=8),
            name="Raw probability",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["calibrated_mean"],
            y=summary["observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_CAL, width=3),
            marker=dict(size=8),
            name="Calibrated probability",
        )
    )
    fig.update_layout(title=title, xaxis_title="Mean predicted probability", yaxis_title="Observed frequency")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return plotly_html(apply_theme(fig))


def brier_delta_chart(summary: pd.DataFrame, title: str) -> str:
    if not HAS_PLOTLY or summary.empty:
        return ""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["raw_probability_bin"].astype(str),
            y=summary["brier_delta"],
            marker_color=np.where(summary["brier_delta"] <= 0, COLOR_RAW, COLOR_CAL),
            name="Brier calibrated - raw",
        )
    )
    fig.add_hline(y=0.0, line_color="#8ea3b7", line_dash="dash")
    fig.update_layout(title=title, xaxis_title="Raw probability bin", yaxis_title="Brier delta")
    return plotly_html(apply_theme(fig))


def training_vs_backtest_chart(comparison: pd.DataFrame) -> str:
    if not HAS_PLOTLY or comparison.empty:
        return ""
    fig = go.Figure()
    x = comparison["raw_probability_bin"].astype(str)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=comparison["training_observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_RAW, width=3),
            name="Synthetic training observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=comparison["backtest_observed_frequency"],
            mode="lines+markers",
            line=dict(color=COLOR_CAL, width=3),
            name="Backtest observed",
        )
    )
    fig.update_layout(
        title="Training vs Backtest Observed Frequency",
        xaxis_title="Raw probability bin",
        yaxis_title="Observed frequency",
    )
    fig.update_yaxes(range=[0, 1])
    return plotly_html(apply_theme(fig))


def latest_curve_chart(curve: pd.DataFrame) -> str:
    if not HAS_PLOTLY or curve.empty:
        return ""
    latest_refit_id = curve["refit_id"].max()
    latest = curve[curve["refit_id"] == latest_refit_id].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#8ea3b7", dash="dash"), name="No shift"))
    fig.add_trace(
        go.Scatter(
            x=latest["raw_probability_threshold"],
            y=latest["calibrated_probability_threshold"],
            mode="lines+markers",
            line=dict(color=COLOR_CAL, width=3),
            marker=dict(size=6),
            name=f"Refit {int(latest_refit_id)}",
        )
    )
    fig.update_layout(title="Latest Isotonic Mapping", xaxis_title="Raw probability", yaxis_title="Calibrated probability")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return plotly_html(apply_theme(fig))


def build_findings(
    *,
    backtest_tail: pd.DataFrame,
    training_vs_backtest: pd.DataFrame,
    contract_bin: pd.DataFrame,
    volatility_bin: pd.DataFrame,
    minute_bin: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    low_tail = pd.DataFrame()
    if "raw_probability_band" in backtest_tail.columns:
        low_tail = backtest_tail[backtest_tail["raw_probability_band"].astype(str).str.startswith("Low")]
    if not low_tail.empty:
        row = low_tail.iloc[0]
        rows.append(
            {
                "diagnostic": "Lower-tail mapping",
                "question": "Did isotonic move low raw probabilities toward observed frequency?",
                "evidence": "backtest_tail_summary.csv",
                "n": int(row["n"]),
                "brier_delta": row["brier_delta"],
                "absolute_calibration_error_delta": row["absolute_calibration_error_delta"],
                "interpretation": "Negative deltas mean the low tail improved after isotonic.",
            }
        )

    if not training_vs_backtest.empty:
        gap = training_vs_backtest["observed_frequency_gap_backtest_minus_training"].abs().max()
        rows.append(
            {
                "diagnostic": "Synthetic-vs-backtest distribution",
                "question": "Do synthetic training labels behave like the scored backtest labels by raw bin?",
                "evidence": "training_vs_backtest_by_raw_probability_bin.csv",
                "n": int(training_vs_backtest["backtest_n"].fillna(0).sum()),
                "max_observed_frequency_gap": gap,
                "interpretation": "Large gaps suggest the synthetic calibration sample is not representative.",
            }
        )

    for name, frame, evidence in [
        ("Contract heterogeneity", contract_bin, "backtest_calibration_by_contract_and_raw_probability_bin.csv"),
        ("Volatility heterogeneity", volatility_bin, "backtest_calibration_by_volatility_and_raw_probability_bin.csv"),
        ("Minute-bucket heterogeneity", minute_bin, "backtest_calibration_by_minute_bucket_and_raw_probability_bin.csv"),
    ]:
        if frame.empty:
            continue
        rows.append(
            {
                "diagnostic": name,
                "question": "Does isotonic help some contexts and hurt others?",
                "evidence": evidence,
                "n": int(frame["n"].sum()),
                "min_brier_delta": frame["brier_delta"].min(),
                "max_brier_delta": frame["brier_delta"].max(),
                "interpretation": "A wide delta range means one global isotonic curve is probably too blunt.",
            }
        )

    return pd.DataFrame(rows)


def dashboard_style() -> str:
    return f"""<style>
    :root {{
      color-scheme: dark;
      --bg: {COLOR_BG};
      --panel: {COLOR_PANEL};
      --line: #22394b;
      --ink: {COLOR_TEXT};
      --muted: {COLOR_MUTED};
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #0b1823 0%, var(--bg) 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 34px 24px 56px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 19px;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.5;
    }}
    code {{
      color: #cfe8ff;
    }}
    .hero, .panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 22px;
      margin-bottom: 18px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: rgba(255,255,255,0.03);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric strong {{
      font-size: 22px;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    .data-table th:first-child, .data-table td:first-child {{
      text-align: left;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .warning {{
      border-color: rgba(251,191,36,0.45);
      background: rgba(251,191,36,0.08);
    }}
    .muted {{
      color: var(--muted);
    }}
  </style>"""


def section(title: str, body: str, *, warning: bool = False) -> str:
    cls = "panel warning" if warning else "panel"
    return f"<section class='{cls}'><h2>{html.escape(title)}</h2>{body}</section>"


def build_dashboard_html(
    *,
    metrics: Dict[str, Any],
    warnings: List[str],
    backtest_bin: pd.DataFrame,
    backtest_tail: pd.DataFrame,
    training_bin: pd.DataFrame,
    training_vs_backtest: pd.DataFrame,
    findings: pd.DataFrame,
    curve: pd.DataFrame,
    hve_metrics: Dict[str, Any],
    hve_backtest_bin: pd.DataFrame,
    hve_backtest_tail: pd.DataFrame,
    hve_training_bin: pd.DataFrame,
    hve_training_vs_backtest: pd.DataFrame,
    hve_backtest_contract_bin: pd.DataFrame,
    hve_backtest_minute_bin: pd.DataFrame,
) -> str:
    warning_html = ""
    if warnings:
        warning_html = section(
            "Readiness Notes",
            "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>",
            warning=True,
        )

    chart_sections = [
        section("Backtest Raw Vs Calibrated", calibration_chart(backtest_bin, "Backtest Calibration by Raw Probability Bin")),
        section("Backtest Brier Delta", brier_delta_chart(backtest_bin, "Backtest Brier Delta by Raw Probability Bin")),
        section("Training Vs Backtest", training_vs_backtest_chart(training_vs_backtest)),
        section("Latest Isotonic Curve", latest_curve_chart(curve)),
    ]
    slice_link = section(
        "Standalone Slice Dashboard",
        "<p><a href='high_volatility_extreme/index.html'>Open the high-volatility-extreme isotonic diagnostics dashboard</a>.</p>",
    )
    hve_sections = [
        section(
            "High Volatility Extreme Isotonic Slice",
            "<p>This section isolates scored rows where <code>is_high_volatility_extreme == true</code>. "
            "When Model_M.py has been rerun with the new instrumentation, the training comparison also filters "
            "synthetic isotonic examples to historical minutes that were high-volatility extreme.</p>"
            + metric_cards(hve_metrics),
        ),
        section(
            "HVE Backtest Raw Vs Calibrated",
            calibration_chart(hve_backtest_bin, "High Volatility Extreme Backtest Calibration"),
        ),
        section(
            "HVE Backtest Brier Delta",
            brier_delta_chart(hve_backtest_bin, "High Volatility Extreme Brier Delta by Raw Probability Bin"),
        ),
        section("HVE Training Vs HVE Backtest", training_vs_backtest_chart(hve_training_vs_backtest)),
        section("HVE Backtest Tail Summary", "<div class='table-wrap'>" + render_table(hve_backtest_tail, max_rows=12) + "</div>"),
        section("HVE Backtest Probability-Bin Summary", "<div class='table-wrap'>" + render_table(hve_backtest_bin, max_rows=20) + "</div>"),
        section("HVE Training Probability-Bin Summary", "<div class='table-wrap'>" + render_table(hve_training_bin, max_rows=20) + "</div>"),
        section("HVE Contract-Bin Summary", "<div class='table-wrap'>" + render_table(hve_backtest_contract_bin, max_rows=30) + "</div>"),
        section("HVE Minute-Bucket Summary", "<div class='table-wrap'>" + render_table(hve_backtest_minute_bin, max_rows=36) + "</div>"),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model M Isotonic Diagnostics</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <h1>Model M Isotonic Diagnostics</h1>
    <p>
      This dashboard compares pre-isotonic raw probabilities against calibrated Model M probabilities,
      then checks whether the isotonic training sample behaves like the scored backtest sample.
    </p>
    {metric_cards(metrics)}
  </section>
  {warning_html}
  {slice_link}
  {''.join(chart_sections)}
  {''.join(hve_sections)}
  {section("Diagnostic Findings", "<div class='table-wrap'>" + render_table(findings, max_rows=12) + "</div>")}
  {section("Backtest Tail Summary", "<div class='table-wrap'>" + render_table(backtest_tail, max_rows=12) + "</div>")}
  {section("Backtest Probability-Bin Summary", "<div class='table-wrap'>" + render_table(backtest_bin, max_rows=20) + "</div>")}
  {section("Training Probability-Bin Summary", "<div class='table-wrap'>" + render_table(training_bin, max_rows=20) + "</div>")}
  {section("Training Vs Backtest Bin Comparison", "<div class='table-wrap'>" + render_table(training_vs_backtest, max_rows=20) + "</div>")}
</main>
</body>
</html>"""


def build_hve_dashboard_html(
    *,
    hve_metrics: Dict[str, Any],
    warnings: List[str],
    hve_backtest_bin: pd.DataFrame,
    hve_backtest_tail: pd.DataFrame,
    hve_training_bin: pd.DataFrame,
    hve_training_vs_backtest: pd.DataFrame,
    hve_findings: pd.DataFrame,
    curve: pd.DataFrame,
    hve_backtest_contract_bin: pd.DataFrame,
    hve_backtest_minute_bin: pd.DataFrame,
    hve_training_contract_bin: pd.DataFrame,
) -> str:
    warning_html = ""
    if warnings:
        warning_html = section(
            "Readiness Notes",
            "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>",
            warning=True,
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model M High-Volatility-Extreme Isotonic Diagnostics</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <h1>Model M HVE Isotonic Diagnostics</h1>
    <p>
      This standalone dashboard isolates high-volatility-extreme rows and compares pre-isotonic raw
      probabilities against calibrated Model M probabilities inside that slice.
      <a href="../index.html">Open the overall dashboard</a>.
    </p>
    {metric_cards(hve_metrics)}
  </section>
  {warning_html}
  {section("HVE Backtest Raw Vs Calibrated", calibration_chart(hve_backtest_bin, "High Volatility Extreme Backtest Calibration"))}
  {section("HVE Backtest Brier Delta", brier_delta_chart(hve_backtest_bin, "High Volatility Extreme Brier Delta by Raw Probability Bin"))}
  {section("HVE Training Vs HVE Backtest", training_vs_backtest_chart(hve_training_vs_backtest))}
  {section("Latest Global Isotonic Curve", latest_curve_chart(curve))}
  {section("HVE Diagnostic Findings", "<div class='table-wrap'>" + render_table(hve_findings, max_rows=12) + "</div>")}
  {section("HVE Backtest Tail Summary", "<div class='table-wrap'>" + render_table(hve_backtest_tail, max_rows=12) + "</div>")}
  {section("HVE Backtest Probability-Bin Summary", "<div class='table-wrap'>" + render_table(hve_backtest_bin, max_rows=20) + "</div>")}
  {section("HVE Training Probability-Bin Summary", "<div class='table-wrap'>" + render_table(hve_training_bin, max_rows=20) + "</div>")}
  {section("HVE Training Vs Backtest Bin Comparison", "<div class='table-wrap'>" + render_table(hve_training_vs_backtest, max_rows=20) + "</div>")}
  {section("HVE Contract-Bin Summary", "<div class='table-wrap'>" + render_table(hve_backtest_contract_bin, max_rows=30) + "</div>")}
  {section("HVE Training Contract-Bin Summary", "<div class='table-wrap'>" + render_table(hve_training_contract_bin, max_rows=30) + "</div>")}
  {section("HVE Minute-Bucket Summary", "<div class='table-wrap'>" + render_table(hve_backtest_minute_bin, max_rows=36) + "</div>")}
</main>
</body>
</html>"""


def load_training_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    required = {"raw_probability", "calibrated_probability", "outcome"}
    wanted = [
        "refit_id",
        "schedule_idx",
        "schedule_index",
        "contract_label",
        "raw_probability",
        "calibrated_probability",
        "outcome",
        "horizon_steps",
        "synthetic_moneyness",
        "minute_price_index",
        "forecast_datetime_utc",
        "training_hour_start_utc",
        "current_spot",
        "terminal_spot",
        "reference_price",
        "strike",
        "next_variance",
        "training_primary_volatility_band",
        "primary_volatility_band",
        "training_rolling_percentile_rank",
        "rolling_percentile_rank",
        "training_is_high_volatility",
        "is_high_volatility",
        "training_is_high_volatility_extreme",
        "is_high_volatility_extreme",
    ]
    header = pd.read_csv(path, nrows=0).columns
    if not required.issubset(set(header)):
        return pd.DataFrame()
    usecols = [column for column in wanted if column in header]
    return pd.read_csv(path, usecols=usecols)


def main() -> None:
    args = parse_args()
    eval_dir = args.model_m_eval_dir.resolve()
    diagnostics_dir = args.isotonic_diagnostics_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    backtest = attach_volatility_context(
        load_backtest_frame(eval_dir, warnings),
        args.volatility_segments_csv.resolve() if args.volatility_segments_csv else None,
    )

    training_path = first_existing(
        [
            diagnostics_dir / "isotonic_training_diagnostics.csv",
            diagnostics_dir / "isotonic_training_examples.csv",
        ]
    )
    training = load_training_examples(training_path) if training_path else pd.DataFrame()
    if training.empty:
        warnings.append(
            "isotonic_training_diagnostics.csv is missing or empty. Rerun Model_M.py to save isotonic training diagnostics."
        )
    else:
        training = add_training_error_columns(training)

    curve_path = first_existing(
        [
            diagnostics_dir / "isotonic_curve_points.csv",
            diagnostics_dir / "isotonic_curve_thresholds.csv",
        ]
    )
    curve = pd.read_csv(curve_path) if curve_path and curve_path.exists() else pd.DataFrame()
    if curve.empty:
        warnings.append("isotonic_curve_points.csv is missing or empty. Latest isotonic curve chart is unavailable.")

    refit_path = diagnostics_dir / "isotonic_refit_summary.csv"
    refit_summary = pd.read_csv(refit_path) if refit_path.exists() else pd.DataFrame()
    if refit_summary.empty:
        warnings.append("isotonic_refit_summary.csv is missing or empty. Refit-level summaries are unavailable.")

    backtest_bin = summarize_calibration(
        backtest,
        group_cols=["raw_probability_bin"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    backtest_tail = summarize_calibration(
        backtest,
        group_cols=["raw_probability_band"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    backtest_contract_bin = summarize_calibration(
        backtest,
        group_cols=["contract_label", "raw_probability_bin"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    backtest_volatility_bin = (
        summarize_calibration(
            backtest,
            group_cols=["primary_volatility_band", "raw_probability_bin"],
            raw_col="p_model_raw",
            calibrated_col="p_model_calibrated",
        )
        if "primary_volatility_band" in backtest.columns
        else pd.DataFrame()
    )
    backtest_minute_bin = (
        summarize_calibration(
            backtest,
            group_cols=["minute_bucket", "raw_probability_bin"],
            raw_col="p_model_raw",
            calibrated_col="p_model_calibrated",
        )
        if "minute_bucket" in backtest.columns
        else pd.DataFrame()
    )
    training_bin = summarize_calibration(
        training,
        group_cols=["raw_probability_bin"],
        raw_col="raw_probability",
        calibrated_col="calibrated_probability",
    )
    training_contract_bin = summarize_calibration(
        training,
        group_cols=["contract_label", "raw_probability_bin"],
        raw_col="raw_probability",
        calibrated_col="calibrated_probability",
    )
    training_vs_backtest = compare_training_to_backtest(training_bin, backtest_bin)
    hve_backtest = high_volatility_extreme_backtest(backtest)
    hve_training = high_volatility_extreme_training(training)
    if not backtest.empty and hve_backtest.empty:
        warnings.append(
            "No high-volatility-extreme backtest rows were available. "
            "Run the Model_M volatility decomposition/dashboard first, or provide --volatility-segments-csv."
        )
    if not training.empty and hve_training.empty:
        warnings.append(
            "No high-volatility-extreme isotonic training examples were available. "
            "Rerun Model_M.py with the latest instrumentation, or check whether the selected training regime contains HVE minutes."
        )

    hve_backtest_bin = summarize_calibration(
        hve_backtest,
        group_cols=["raw_probability_bin"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    hve_backtest_tail = summarize_calibration(
        hve_backtest,
        group_cols=["raw_probability_band"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    hve_backtest_contract_bin = summarize_calibration(
        hve_backtest,
        group_cols=["contract_label", "raw_probability_bin"],
        raw_col="p_model_raw",
        calibrated_col="p_model_calibrated",
    )
    hve_backtest_minute_bin = (
        summarize_calibration(
            hve_backtest,
            group_cols=["minute_bucket", "raw_probability_bin"],
            raw_col="p_model_raw",
            calibrated_col="p_model_calibrated",
        )
        if "minute_bucket" in hve_backtest.columns
        else pd.DataFrame()
    )
    hve_training_bin = summarize_calibration(
        hve_training,
        group_cols=["raw_probability_bin"],
        raw_col="raw_probability",
        calibrated_col="calibrated_probability",
    )
    hve_training_contract_bin = summarize_calibration(
        hve_training,
        group_cols=["contract_label", "raw_probability_bin"],
        raw_col="raw_probability",
        calibrated_col="calibrated_probability",
    )
    hve_training_vs_backtest = compare_training_to_backtest(hve_training_bin, hve_backtest_bin)
    findings = build_findings(
        backtest_tail=backtest_tail,
        training_vs_backtest=training_vs_backtest,
        contract_bin=backtest_contract_bin,
        volatility_bin=backtest_volatility_bin,
        minute_bin=backtest_minute_bin,
    )
    hve_findings = build_findings(
        backtest_tail=hve_backtest_tail,
        training_vs_backtest=hve_training_vs_backtest,
        contract_bin=hve_backtest_contract_bin,
        volatility_bin=pd.DataFrame(),
        minute_bin=hve_backtest_minute_bin,
    )
    if not hve_findings.empty:
        hve_findings = hve_findings.copy()
        hve_findings["diagnostic"] = "HVE " + hve_findings["diagnostic"].astype(str)
        hve_findings["evidence"] = hve_findings["evidence"].replace(
            {
                "backtest_tail_summary.csv": "high_volatility_extreme_backtest_tail_summary.csv",
                "training_vs_backtest_by_raw_probability_bin.csv": (
                    "high_volatility_extreme_training_vs_backtest_by_raw_probability_bin.csv"
                ),
                "backtest_calibration_by_contract_and_raw_probability_bin.csv": (
                    "high_volatility_extreme_backtest_calibration_by_contract_and_raw_probability_bin.csv"
                ),
                "backtest_calibration_by_minute_bucket_and_raw_probability_bin.csv": (
                    "high_volatility_extreme_backtest_calibration_by_minute_bucket_and_raw_probability_bin.csv"
                ),
            }
        )

    write_csv(backtest_bin, output_dir, "backtest_calibration_by_raw_probability_bin.csv")
    write_csv(backtest_tail, output_dir, "backtest_tail_summary.csv")
    write_csv(backtest_contract_bin, output_dir, "backtest_calibration_by_contract_and_raw_probability_bin.csv")
    write_csv(backtest_volatility_bin, output_dir, "backtest_calibration_by_volatility_and_raw_probability_bin.csv")
    write_csv(backtest_minute_bin, output_dir, "backtest_calibration_by_minute_bucket_and_raw_probability_bin.csv")
    write_csv(training_bin, output_dir, "training_calibration_by_raw_probability_bin.csv")
    write_csv(training_contract_bin, output_dir, "training_calibration_by_contract_and_raw_probability_bin.csv")
    write_csv(training_vs_backtest, output_dir, "training_vs_backtest_by_raw_probability_bin.csv")
    write_csv(findings, output_dir, "diagnostic_findings.csv")
    write_csv(hve_backtest_bin, output_dir, "high_volatility_extreme_backtest_calibration_by_raw_probability_bin.csv")
    write_csv(hve_backtest_tail, output_dir, "high_volatility_extreme_backtest_tail_summary.csv")
    write_csv(
        hve_backtest_contract_bin,
        output_dir,
        "high_volatility_extreme_backtest_calibration_by_contract_and_raw_probability_bin.csv",
    )
    write_csv(
        hve_backtest_minute_bin,
        output_dir,
        "high_volatility_extreme_backtest_calibration_by_minute_bucket_and_raw_probability_bin.csv",
    )
    write_csv(hve_training_bin, output_dir, "high_volatility_extreme_training_calibration_by_raw_probability_bin.csv")
    write_csv(
        hve_training_contract_bin,
        output_dir,
        "high_volatility_extreme_training_calibration_by_contract_and_raw_probability_bin.csv",
    )
    write_csv(
        hve_training_vs_backtest,
        output_dir,
        "high_volatility_extreme_training_vs_backtest_by_raw_probability_bin.csv",
    )
    write_csv(hve_findings, output_dir, "high_volatility_extreme_diagnostic_findings.csv")

    metrics = {
        "Backtest rows": len(backtest),
        "Training examples": len(training),
        "Refits": refit_summary["refit_id"].nunique() if "refit_id" in refit_summary.columns else 0,
        "Mean backtest Brier delta": backtest["brier_calibrated"].mean() - backtest["brier_raw"].mean()
        if not backtest.empty
        else np.nan,
        "Mean training Brier delta": training["brier_calibrated"].mean() - training["brier_raw"].mean()
        if not training.empty
        else np.nan,
    }
    hve_metrics = {
        "HVE backtest rows": len(hve_backtest),
        "HVE training examples": len(hve_training),
        "HVE mean Brier delta": hve_backtest["brier_calibrated"].mean() - hve_backtest["brier_raw"].mean()
        if not hve_backtest.empty
        else np.nan,
        "HVE low-tail rows": int((hve_backtest["p_model_raw"] < 0.2).sum()) if not hve_backtest.empty else 0,
        "HVE high-tail rows": int((hve_backtest["p_model_raw"] > 0.8).sum()) if not hve_backtest.empty else 0,
        "HVE training/backtest bin rows": len(hve_training_vs_backtest),
    }
    html_out = build_dashboard_html(
        metrics=metrics,
        warnings=warnings,
        backtest_bin=backtest_bin,
        backtest_tail=backtest_tail,
        training_bin=training_bin,
        training_vs_backtest=training_vs_backtest,
        findings=findings,
        curve=curve,
        hve_metrics=hve_metrics,
        hve_backtest_bin=hve_backtest_bin,
        hve_backtest_tail=hve_backtest_tail,
        hve_training_bin=hve_training_bin,
        hve_training_vs_backtest=hve_training_vs_backtest,
        hve_backtest_contract_bin=hve_backtest_contract_bin,
        hve_backtest_minute_bin=hve_backtest_minute_bin,
    )
    (output_dir / "index.html").write_text(html_out, encoding="utf-8")
    hve_html_out = build_hve_dashboard_html(
        hve_metrics=hve_metrics,
        warnings=warnings,
        hve_backtest_bin=hve_backtest_bin,
        hve_backtest_tail=hve_backtest_tail,
        hve_training_bin=hve_training_bin,
        hve_training_vs_backtest=hve_training_vs_backtest,
        hve_findings=hve_findings,
        curve=curve,
        hve_backtest_contract_bin=hve_backtest_contract_bin,
        hve_backtest_minute_bin=hve_backtest_minute_bin,
        hve_training_contract_bin=hve_training_contract_bin,
    )
    hve_output_dir = output_dir / "high_volatility_extreme"
    hve_output_dir.mkdir(parents=True, exist_ok=True)
    (hve_output_dir / "index.html").write_text(hve_html_out, encoding="utf-8")
    (output_dir / "high_volatility_extreme_index.html").write_text(hve_html_out, encoding="utf-8")

    print(f"Model M isotonic diagnostics written to: {output_dir}")
    print(f"Model M high-volatility-extreme isotonic dashboard written to: {hve_output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
