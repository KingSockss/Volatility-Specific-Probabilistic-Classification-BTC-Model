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

from Model_B.Model_B_Eval import (
    ASSUMPTIONS as MODEL_B_ASSUMPTIONS,
    HAS_PLOTLY,
    attach_outcomes as attach_outcomes_b,
    build_brier_decomposition,
    build_metrics_summary,
    build_outcome_join_coverage,
    build_resolution_mismatches,
    build_sharpness,
    build_time_bucket_outputs,
    dataframe_to_html_table,
    default_settlement_csv,
    expanded_calibration_error,
    go,
    load_kalshi_price_outputs as load_model_b_price_outputs,
    load_settlements,
    make_subplots,
)
from Model_K.Model_K import (
    ASSUMPTIONS as MODEL_K_ASSUMPTIONS,
    attach_outcomes as attach_outcomes_k,
    build_kalshi_reality_outcomes as build_kalshi_reality_outcomes_k,
    load_kalshi_price_outputs as load_model_k_price_outputs,
)
from Model_K_Volatility_Decomposition_RT.Model_K_Volatility_Decomposition_RT import (
    ROLLING_WINDOW_HOURS,
    ROLLING_WINDOW_MINUTES,
    SEGMENTS,
    add_realtime_window_thresholds,
    build_hourly_market_state_table,
    compute_hourly_realized_volatility,
    load_or_fetch_binance_minutes,
    thresholds_table,
    volatility_assumptions,
)


OUTPUT_FOLDER_NAME = "Model_B_Volatility_Dashboards_outputs"
MODEL_B_TOTAL_LABEL = "Model B Total"
MODEL_B_RT_LABEL = "Model B RT"
MODEL_K_RT_LABEL = "Model K RT"

COLOR_MODEL_B = "#6cb6ff"
COLOR_MODEL_B_RT = "#ff8f70"
COLOR_MODEL_K = "#4fd1c5"
COLOR_PERFECT = "#8ea3b7"
COLOR_GRID = "#22384a"
COLOR_AXIS = "#35516c"
COLOR_PAPER = "#0d1b26"
COLOR_BG = "#07131d"
COLOR_PANEL = "#10202d"
COLOR_PANEL_ALT = "#142736"
COLOR_TEXT = "#e8f1fa"
COLOR_MUTED = "#96aabd"

HIST_BINS = np.linspace(0.0, 1.0, 21)

REQUIRED_HOURLY_STATE_COLUMNS = [
    "event_ticker",
    "forecast_hour_start_utc",
    "realized_variance",
    "realized_volatility",
    "rolling_window_hours",
    "rolling_window_minutes",
    "rolling_window_observations",
    "rolling_percentile_rank",
    "primary_volatility_band",
    "is_low_volatility",
    "is_standard_volatility",
    "is_high_volatility",
    "is_low_volatility_extreme",
    "is_high_volatility_extreme",
    "q10_realized_volatility",
    "q25_realized_volatility",
    "q75_realized_volatility",
    "q90_realized_volatility",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME


def model_b_rt_output_root(root: Path) -> Path:
    return root / "Model_B_Volatility_Decomposition_RT" / "Model_B_Volatility_Decomposition_RT_outputs"


def model_k_rt_output_root(root: Path) -> Path:
    return root / "Model_K_Volatility_Decomposition_RT" / "Model_K_Volatility_Decomposition_RT_outputs"


def first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def default_hourly_states_csv(root: Path) -> Optional[Path]:
    return first_existing(
        [
            model_b_rt_output_root(root) / "hourly_market_volatility_segments.csv",
            model_k_rt_output_root(root) / "hourly_market_volatility_segments.csv",
        ]
    )


def default_binance_cache_csv(root: Path) -> Optional[Path]:
    return first_existing(
        [
            model_b_rt_output_root(root) / "binance_1m_klines.csv",
            model_k_rt_output_root(root) / "binance_1m_klines.csv",
        ]
    )


def resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value)


def resolve_optional_path(root: Path, value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    return resolve_path(root, value)


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build dark-mode volatility comparison dashboards for Model B real-time volatility "
            "segments versus Model B overall and versus Model K real-time volatility segments."
        )
    )
    parser.add_argument(
        "--model-b-price-dir",
        type=Path,
        default=root / "Model_B" / "Model_B_Output_Raw_Data",
        help="Folder containing Model_B.py hourly raw forecast CSVs.",
    )
    parser.add_argument(
        "--model-k-price-dir",
        type=Path,
        default=root / "Data_Sourcing" / "Kalshi_Pricing_Fetch" / "hourly_events_price_data",
        help="Folder containing the minute-level Kalshi hourly pricing CSVs used by Model K.",
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=default_settlement_csv(root),
        help="Settlement CSV with official Kalshi outcomes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Root directory for the comparison dashboard outputs.",
    )
    parser.add_argument(
        "--hourly-market-states-csv",
        type=Path,
        default=default_hourly_states_csv(root),
        help=(
            "Optional precomputed hourly_market_volatility_segments.csv. "
            "If present and it covers the scored events, it is reused."
        ),
    )
    parser.add_argument(
        "--binance-minute-cache-csv",
        type=Path,
        default=default_binance_cache_csv(root),
        help=(
            "Optional Binance 1-minute cache used when the hourly market state table "
            "must be recomputed."
        ),
    )
    parser.add_argument(
        "--refresh-binance-cache",
        action="store_true",
        help="Force a fresh Binance minute download if the hourly state table must be recomputed.",
    )
    parser.add_argument(
        "--skip-binance-audit",
        action="store_true",
        help="Skip the diagnostic Binance-vs-Kalshi resolution audit tables.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary classification accuracy tables.",
    )
    return parser.parse_args()


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}).fillna(False)


def load_hourly_market_states_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ["forecast_hour_start_utc", "hour_start_utc"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True)
    for column in REQUIRED_HOURLY_STATE_COLUMNS:
        if column.startswith("is_") and column in df.columns:
            df[column] = coerce_bool(df[column])
    return df


def validate_hourly_market_states(hourly_market_states: pd.DataFrame, raw: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_HOURLY_STATE_COLUMNS if column not in hourly_market_states.columns]
    if missing:
        raise ValueError(
            "Hourly market state table is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    required_events = set(raw["event_ticker"].astype(str))
    available_events = set(hourly_market_states["event_ticker"].astype(str))
    missing_events = sorted(required_events - available_events)
    if missing_events:
        sample = ", ".join(missing_events[:5])
        raise ValueError(
            "Hourly market state table does not cover all scored events. "
            f"Example missing event_ticker(s): {sample}"
        )


def compute_or_load_hourly_market_states(
    *,
    raw: pd.DataFrame,
    output_root: Path,
    root: Path,
    hourly_market_states_csv: Optional[Path],
    binance_minute_cache_csv: Optional[Path],
    refresh_binance_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    notes: List[str] = []

    preferred_hourly_states = resolve_optional_path(root, hourly_market_states_csv)
    if preferred_hourly_states and preferred_hourly_states.exists():
        hourly_market_states = load_hourly_market_states_csv(preferred_hourly_states)
        validate_hourly_market_states(hourly_market_states, raw)
        notes.append(f"Reused hourly volatility segment table from {preferred_hourly_states}.")
        thresholds = thresholds_table(hourly_market_states)
        return hourly_market_states, thresholds, notes

    required_start = raw["event_datetime_utc"].min() - pd.Timedelta(hours=1 + ROLLING_WINDOW_HOURS)
    required_end = raw["event_datetime_utc"].max()

    preferred_cache = resolve_optional_path(root, binance_minute_cache_csv)
    cache_path = preferred_cache or (output_root / "binance_1m_klines.csv")
    cache_path = cache_path if cache_path.is_absolute() else (root / cache_path)

    binance_minutes = load_or_fetch_binance_minutes(
        cache_path=cache_path,
        start_utc=required_start,
        end_utc=required_end,
        refresh_cache=refresh_binance_cache,
    )
    hourly_volatility = compute_hourly_realized_volatility(binance_minutes)
    hourly_volatility = add_realtime_window_thresholds(hourly_volatility)
    hourly_market_states = build_hourly_market_state_table(raw, hourly_volatility)
    thresholds = thresholds_table(hourly_market_states)

    binance_minutes.to_csv(output_root / "binance_1m_klines.csv", index=False)
    hourly_volatility.to_csv(output_root / "binance_hourly_realized_volatility.csv", index=False)
    notes.append(f"Computed hourly volatility segment table using Binance minute cache {cache_path}.")
    return hourly_market_states, thresholds, notes


def slice_by_event_ids(frame: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["event_ticker"].astype(str).isin(event_ids)].copy()


def build_bundle(
    *,
    name: str,
    raw: pd.DataFrame,
    forecasts: pd.DataFrame,
    unmatched: pd.DataFrame,
    calibration_bins: int,
    classification_threshold: float,
    skip_binance_audit: bool,
    n_hourly_markets: int,
) -> Dict[str, Any]:
    coverage = build_outcome_join_coverage(forecasts, raw, unmatched)
    resolution_mismatches = pd.DataFrame() if skip_binance_audit else build_resolution_mismatches(raw)
    metrics = build_metrics_summary(raw, threshold=classification_threshold)
    decomposition = build_brier_decomposition(raw, bins=calibration_bins)
    calibration, expanded_summary = expanded_calibration_error(raw, bins=calibration_bins)
    sharpness = build_sharpness(raw)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = build_time_bucket_outputs(
        raw,
        calibration_bins=calibration_bins,
        threshold=classification_threshold,
    )

    overall_metrics = metrics.loc[metrics["segment"] == "overall"].iloc[0]
    overall_ece = expanded_summary.loc[expanded_summary["segment"] == "overall"].iloc[0]

    return {
        "name": name,
        "raw": raw,
        "forecasts": forecasts,
        "unmatched": unmatched,
        "coverage": coverage,
        "resolution_mismatches": resolution_mismatches,
        "metrics": metrics,
        "decomposition": decomposition,
        "calibration": calibration,
        "expanded_summary": expanded_summary,
        "sharpness": sharpness,
        "time_bucket_metrics": time_bucket_metrics,
        "time_bucket_accuracy": time_bucket_accuracy,
        "time_bucket_brier": time_bucket_brier,
        "time_bucket_calibration": time_bucket_calibration,
        "n_hourly_markets": int(n_hourly_markets),
        "overall_metrics": overall_metrics,
        "overall_ece": overall_ece,
    }


def overview_comparison_table(focus_bundle: Dict[str, Any], benchmark_bundle: Dict[str, Any]) -> pd.DataFrame:
    benchmark_overall = benchmark_bundle["overall_metrics"]
    focus_overall = focus_bundle["overall_metrics"]
    benchmark_ece = benchmark_bundle["overall_ece"]
    focus_ece = focus_bundle["overall_ece"]

    rows = [
        ("Scored rows", float(focus_overall["n_forecasts"]), float(benchmark_overall["n_forecasts"])),
        ("Event contracts", float(focus_overall["n_event_contracts"]), float(benchmark_overall["n_event_contracts"])),
        ("Brier score", float(focus_overall["brier_score"]), float(benchmark_overall["brier_score"])),
        ("Log loss", float(focus_overall["log_loss"]), float(benchmark_overall["log_loss"])),
        ("ECE", float(focus_ece["expected_calibration_error"]), float(benchmark_ece["expected_calibration_error"])),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "focus": focus_value,
                "benchmark": benchmark_value,
                "delta": focus_value - benchmark_value,
            }
            for metric, focus_value, benchmark_value in rows
        ]
    )


def build_comparison_table(
    *,
    benchmark_df: pd.DataFrame,
    focus_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    sort_cols: Optional[Sequence[str]] = None,
    hidden_key_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    hidden = set(hidden_key_cols or [])
    benchmark = benchmark_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    focus = focus_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    merged = focus.merge(benchmark, on=list(key_cols), how="outer", suffixes=("_focus", "_benchmark"))

    ordered_cols: List[str] = [column for column in key_cols if column not in hidden]
    for column in value_cols:
        focus_col = f"{column}_focus"
        benchmark_col = f"{column}_benchmark"
        ordered_cols.extend([focus_col, benchmark_col])
        if pd.api.types.is_numeric_dtype(focus[column]) and pd.api.types.is_numeric_dtype(benchmark[column]):
            delta_col = f"{column}_delta"
            merged[delta_col] = merged[focus_col] - merged[benchmark_col]
            ordered_cols.append(delta_col)

    if sort_cols:
        merged = merged.sort_values(list(sort_cols), kind="stable")

    ordered_cols = [column for column in ordered_cols if column in merged.columns]
    return merged[ordered_cols].reset_index(drop=True)


def threshold_context_table(thresholds: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "threshold_name",
        "percentile",
        "window_hours",
        "latest_realized_volatility_cutoff",
        "median_realized_volatility_cutoff",
        "min_realized_volatility_cutoff",
        "max_realized_volatility_cutoff",
    ]
    return thresholds[columns].copy()


def plotly_theme(
    fig: Any,
    *,
    height: int,
    barmode: Optional[str] = None,
    legend_y: float = 1.22,
    top_margin: int = 136,
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
            font=dict(size=11),
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


def comparison_charts_html(
    benchmark_bundle: Dict[str, Any],
    focus_bundle: Dict[str, Any],
    *,
    benchmark_label: str,
    focus_label: str,
    benchmark_color: str,
    focus_color: str,
) -> tuple[str, str]:
    if not HAS_PLOTLY:
        fallback = (
            "<div class='note-card'>"
            "<p>Interactive Plotly charts were unavailable in this environment, so only the comparison tables were generated.</p>"
            "</div>"
        )
        return fallback, fallback

    benchmark_calibration = benchmark_bundle["calibration"]
    benchmark_calibration = benchmark_calibration[
        (benchmark_calibration["segment"] == "overall") & (benchmark_calibration["n_forecasts"] > 0)
    ].copy()
    focus_calibration = focus_bundle["calibration"]
    focus_calibration = focus_calibration[
        (focus_calibration["segment"] == "overall") & (focus_calibration["n_forecasts"] > 0)
    ].copy()

    overlay = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration Curve Overlay", "Probability Distribution Overlay"),
        horizontal_spacing=0.11,
    )
    overlay.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=COLOR_PERFECT, dash="dash"),
            name="Perfect calibration",
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Scatter(
            x=benchmark_calibration["mean_p_kalshi"],
            y=benchmark_calibration["observed_frequency"],
            mode="lines+markers",
            name=benchmark_label,
            marker=dict(color=benchmark_color, size=8),
            line=dict(color=benchmark_color, width=3),
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Scatter(
            x=focus_calibration["mean_p_kalshi"],
            y=focus_calibration["observed_frequency"],
            mode="lines+markers",
            name=focus_label,
            marker=dict(color=focus_color, size=8),
            line=dict(color=focus_color, width=3),
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Histogram(
            x=benchmark_bundle["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=benchmark_color,
            name=benchmark_label,
        ),
        row=1,
        col=2,
    )
    overlay.add_trace(
        go.Histogram(
            x=focus_bundle["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=focus_color,
            name=focus_label,
        ),
        row=1,
        col=2,
    )
    overlay.update_xaxes(range=[0, 1], title_text="Mean predicted probability", row=1, col=1)
    overlay.update_yaxes(range=[0, 1], title_text="Observed frequency", row=1, col=1)
    overlay.update_xaxes(range=[0, 1], title_text="Predicted probability", row=1, col=2)
    overlay.update_yaxes(title_text="Count", row=1, col=2)
    overlay.update_annotations(y=0.98)
    plotly_theme(overlay, height=430, barmode="overlay", legend_y=1.24, top_margin=144)

    benchmark_brier = benchmark_bundle["decomposition"].loc[
        benchmark_bundle["decomposition"]["segment"] == "overall"
    ].iloc[0]
    focus_brier = focus_bundle["decomposition"].loc[focus_bundle["decomposition"]["segment"] == "overall"].iloc[0]
    categories = [
        ("Brier Score", "brier_score"),
        ("Reliability", "reliability"),
        ("Resolution", "resolution"),
        ("Uncertainty", "uncertainty"),
        ("Brier From Decomp", "brier_from_decomposition"),
    ]

    brier_chart = go.Figure()
    brier_chart.add_trace(
        go.Bar(
            x=[label for label, _ in categories],
            y=[float(benchmark_brier[column]) for _, column in categories],
            name=benchmark_label,
            marker_color=benchmark_color,
        )
    )
    brier_chart.add_trace(
        go.Bar(
            x=[label for label, _ in categories],
            y=[float(focus_brier[column]) for _, column in categories],
            name=focus_label,
            marker_color=focus_color,
        )
    )
    brier_chart.update_yaxes(title_text="Metric value")
    plotly_theme(brier_chart, height=360, barmode="group")

    return (
        overlay.to_html(full_html=False, include_plotlyjs="cdn"),
        brier_chart.to_html(full_html=False, include_plotlyjs=False),
    )


def format_number(value: Any, *, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"

    value_float = float(value)
    prefix = "+" if signed and value_float >= 0 else ""
    return f"{prefix}{value_float:,.{digits}f}"


def format_value_for_column(column: str, value: Any) -> str:
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if not isinstance(value, (np.integer, int, np.floating, float)) or pd.isna(value):
        return "n/a" if pd.isna(value) else str(value)

    lower = column.lower()
    if lower.startswith("delta") or lower.endswith("_delta"):
        digits = 2 if "pct" in lower else 4
        return format_number(value, digits=digits, signed=True)
    if lower.startswith("n_") or any(
        token in lower
        for token in ["rows", "contracts", "correct", "incorrect", "window_hours", "window_minutes", "minute_bars", "mismatch"]
    ):
        return format_number(int(round(float(value))), digits=0)
    if "pct" in lower:
        return format_number(value, digits=2)
    return format_number(value, digits=4)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if "metric" in out.columns and {"focus", "benchmark", "delta"}.issubset(out.columns):
        count_metrics = {"scored rows", "event contracts"}
        for index, metric in out["metric"].astype(str).str.lower().items():
            if metric in count_metrics:
                out.at[index, "focus"] = format_number(out.at[index, "focus"], digits=0)
                out.at[index, "benchmark"] = format_number(out.at[index, "benchmark"], digits=0)
                out.at[index, "delta"] = format_number(out.at[index, "delta"], digits=0, signed=True)
            else:
                out.at[index, "focus"] = format_number(out.at[index, "focus"], digits=4)
                out.at[index, "benchmark"] = format_number(out.at[index, "benchmark"], digits=4)
                out.at[index, "delta"] = format_number(out.at[index, "delta"], digits=4, signed=True)
    for column in out.columns:
        out[column] = out[column].map(lambda value, col=column: format_value_for_column(col, value))
    return out


def render_table(df: pd.DataFrame, *, max_rows: Optional[int] = None) -> str:
    return dataframe_to_html_table(format_table(df), max_rows=max_rows)


def side_by_side_table_html(left_title: str, left_df: pd.DataFrame, right_title: str, right_df: pd.DataFrame) -> str:
    return (
        "<div class='table-grid'>"
        f"<section><h3>{html.escape(left_title)}</h3><div class='table-wrap'>{render_table(left_df, max_rows=25)}</div></section>"
        f"<section><h3>{html.escape(right_title)}</h3><div class='table-wrap'>{render_table(right_df, max_rows=25)}</div></section>"
        "</div>"
    )


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
      --model-b: {COLOR_MODEL_B};
      --model-b-rt: {COLOR_MODEL_B_RT};
      --model-k: {COLOR_MODEL_K};
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
      letter-spacing: 0;
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
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .swatch.total {{
      background: var(--model-b);
    }}
    .swatch.vol {{
      background: var(--model-b-rt);
    }}
    .swatch.k {{
      background: var(--model-k);
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
    .note-card {{
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel-2);
    }}
    .note-card p {{
      margin: 0;
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
    .links {{
      display: grid;
      gap: 8px;
      padding-left: 18px;
    }}
    code {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 5px;
      color: var(--ink);
    }}
  </style>"""


def pill_html(text: str, swatch_class: Optional[str] = None) -> str:
    swatch = f"<span class='swatch {html.escape(swatch_class)}'></span>" if swatch_class else ""
    return f"<span class='pill'>{swatch}{text}</span>"


def comparison_key_series(frame: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_datetime(frame["forecast_datetime_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame["event_contract_id"].astype(str) + "|" + timestamps


def slice_frame_to_comparison_keys(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work["_comparison_key"] = comparison_key_series(work)
    out = work[work["_comparison_key"].isin(keys)].copy()
    return out.drop(columns=["_comparison_key"])


def overlap_comparison_keys(left_raw: pd.DataFrame, right_raw: pd.DataFrame) -> set[str]:
    return set(comparison_key_series(left_raw).tolist()) & set(comparison_key_series(right_raw).tolist())


def build_comparison_artifacts(
    *,
    focus_bundle: Dict[str, Any],
    benchmark_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "overview": overview_comparison_table(focus_bundle, benchmark_bundle),
        "metrics": build_comparison_table(
            benchmark_df=benchmark_bundle["metrics"],
            focus_df=focus_bundle["metrics"],
            key_cols=["segment"],
            value_cols=[
                "n_forecasts",
                "n_event_contracts",
                "base_rate",
                "mean_p_kalshi",
                "brier_score",
                "log_loss",
                "classification_accuracy",
            ],
        ),
        "decomposition": build_comparison_table(
            benchmark_df=benchmark_bundle["decomposition"],
            focus_df=focus_bundle["decomposition"],
            key_cols=["segment"],
            value_cols=[
                "n_forecasts",
                "brier_score",
                "reliability",
                "resolution",
                "uncertainty",
                "brier_from_decomposition",
            ],
        ),
        "calibration": build_comparison_table(
            benchmark_df=benchmark_bundle["expanded_summary"],
            focus_df=focus_bundle["expanded_summary"],
            key_cols=["segment"],
            value_cols=[
                "n_forecasts",
                "expected_calibration_error",
                "root_mean_squared_calibration_error",
                "max_calibration_error",
            ],
        ),
        "sharpness": build_comparison_table(
            benchmark_df=benchmark_bundle["sharpness"],
            focus_df=focus_bundle["sharpness"],
            key_cols=["segment"],
            value_cols=[
                "n_forecasts",
                "base_rate",
                "mean_p_kalshi",
                "forecast_std",
                "sharpness_variance_from_base_rate",
                "mean_abs_distance_from_0_5",
                "mean_predictive_variance_p_times_1_minus_p",
            ],
        ),
        "coverage": build_comparison_table(
            benchmark_df=benchmark_bundle["coverage"],
            focus_df=focus_bundle["coverage"],
            key_cols=["scope", "value"],
            value_cols=["total_forecast_rows", "matched_rows", "unmatched_rows", "match_rate"],
        ),
        "isolated_metrics": build_comparison_table(
            benchmark_df=benchmark_bundle["time_bucket_metrics"][
                (benchmark_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
                & (benchmark_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
            ],
            focus_df=focus_bundle["time_bucket_metrics"][
                (focus_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
                & (focus_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
            ],
            key_cols=["sort_order", "display_name"],
            hidden_key_cols=["sort_order"],
            sort_cols=["sort_order"],
            value_cols=[
                "n_forecasts",
                "n_event_contracts",
                "base_rate",
                "mean_p_kalshi",
                "brier_score",
                "log_loss",
                "classification_accuracy",
            ],
        ),
        "decile_metrics": build_comparison_table(
            benchmark_df=benchmark_bundle["time_bucket_metrics"][
                (benchmark_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
                & (benchmark_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
            ],
            focus_df=focus_bundle["time_bucket_metrics"][
                (focus_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
                & (focus_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
            ],
            key_cols=["sort_order", "display_name"],
            hidden_key_cols=["sort_order"],
            sort_cols=["sort_order"],
            value_cols=[
                "n_forecasts",
                "n_event_contracts",
                "base_rate",
                "mean_p_kalshi",
                "brier_score",
                "log_loss",
                "classification_accuracy",
            ],
        ),
        "time_bucket_accuracy": build_comparison_table(
            benchmark_df=benchmark_bundle["time_bucket_accuracy"][benchmark_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
            focus_df=focus_bundle["time_bucket_accuracy"][focus_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
            key_cols=["sort_order", "display_name"],
            hidden_key_cols=["sort_order"],
            sort_cols=["sort_order"],
            value_cols=[
                "n_forecasts",
                "correct_forecasts",
                "incorrect_forecasts",
                "classification_accuracy",
                "classification_accuracy_pct",
                "threshold",
            ],
        ),
        "time_bucket_brier": build_comparison_table(
            benchmark_df=benchmark_bundle["time_bucket_brier"][benchmark_bundle["time_bucket_brier"]["n_forecasts"] > 0],
            focus_df=focus_bundle["time_bucket_brier"][focus_bundle["time_bucket_brier"]["n_forecasts"] > 0],
            key_cols=["sort_order", "display_name"],
            hidden_key_cols=["sort_order"],
            sort_cols=["sort_order"],
            value_cols=[
                "n_forecasts",
                "brier_score",
                "reliability",
                "resolution",
                "uncertainty",
                "brier_from_decomposition",
            ],
        ),
        "mismatch_count": pd.DataFrame(
            [
                {"scope": focus_bundle["name"], "audit_mismatches": len(focus_bundle["resolution_mismatches"])},
                {"scope": benchmark_bundle["name"], "audit_mismatches": len(benchmark_bundle["resolution_mismatches"])},
            ]
        ),
        "focus_mismatches": focus_bundle["resolution_mismatches"],
        "benchmark_mismatches": benchmark_bundle["resolution_mismatches"],
    }


def build_segment_dashboard_html(
    *,
    page_title: str,
    eyebrow: str,
    heading: str,
    lead_html: str,
    hero_pills: Sequence[str],
    focus_bundle: Dict[str, Any],
    benchmark_bundle: Dict[str, Any],
    focus_label: str,
    benchmark_label: str,
    focus_color: str,
    benchmark_color: str,
    thresholds: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
    artifacts: Dict[str, Any],
    topline_note: str,
) -> str:
    overlay_chart_html, brier_chart_html = comparison_charts_html(
        benchmark_bundle,
        focus_bundle,
        benchmark_label=benchmark_label,
        focus_label=focus_label,
        benchmark_color=benchmark_color,
        focus_color=focus_color,
    )
    assumption_html = "".join(f"<li>{html.escape(item)}</li>" for item in assumptions)
    notes_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes)
    hero_pills_html = "".join(hero_pills)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">{html.escape(eyebrow)}</p>
    <h1>{html.escape(heading)}</h1>
    <p class="lead">{lead_html}</p>
    <div class="hero-pills">{hero_pills_html}</div>
  </section>

  <h2>Topline Comparison</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["overview"])}</div>
    <p class="chart-note">{html.escape(topline_note)}</p>
  </section>

  <h2>Chart Comparison</h2>
  <section class="section-stack">
    <div class="chart-wrap">{overlay_chart_html}</div>
    <div class="chart-wrap">{brier_chart_html}</div>
  </section>

  <h2>Metric Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["metrics"])}</div>
  </section>

  <h2>Brier Decomposition</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["decomposition"])}</div>
  </section>

  <h2>Expanded Calibration Error</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["calibration"])}</div>
  </section>

  <h2>Sharpness</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["sharpness"])}</div>
  </section>

  <h2>Outcome Join Coverage</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(artifacts["coverage"])}</div>
  </section>

  <h2>Time Bucket Comparison</h2>
  <section class="section-stack">
    <section class="panel">
      <h3>Isolated Minute Metrics</h3>
      <div class="table-wrap">{render_table(artifacts["isolated_metrics"])}</div>
    </section>
    <section class="panel">
      <h3>10-Minute Bucket Metrics</h3>
      <div class="table-wrap">{render_table(artifacts["decile_metrics"])}</div>
    </section>
    <section class="panel">
      <h3>Accuracy Evaluation</h3>
      <div class="table-wrap">{render_table(artifacts["time_bucket_accuracy"])}</div>
    </section>
    <section class="panel">
      <h3>Brier Decomposition By Bucket</h3>
      <div class="table-wrap">{render_table(artifacts["time_bucket_brier"])}</div>
    </section>
  </section>

  <h2>Resolution Audit Mismatches</h2>
  <section class="section-stack">
    <section class="panel">
      <div class="table-wrap">{render_table(artifacts["mismatch_count"])}</div>
      <p class="chart-note">
        These audit mismatches are diagnostic only. They compare official Kalshi settlement results
        against the Binance strike audit already used by the existing scripts and are not scored as truth.
      </p>
    </section>
    <section class="panel">
      {side_by_side_table_html(focus_label, artifacts["focus_mismatches"], benchmark_label, artifacts["benchmark_mismatches"])}
    </section>
  </section>

  <h2>Volatility Threshold Context</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(threshold_context_table(thresholds))}</div>
    <p class="chart-note">
      The real-time volatility segmentation uses a trailing <code>{ROLLING_WINDOW_HOURS}</code>-hour
      window, equivalent to <code>{ROLLING_WINDOW_MINUTES}</code> Binance one-minute observations.
    </p>
  </section>

  <h2>Assumptions</h2>
  <section class="panel">
    <ul>{assumption_html}</ul>
  </section>

  <h2>Run Notes</h2>
  <section class="panel">
    <ul>{notes_html}</ul>
    <p class="chart-note"><a href="../index.html">Back to dashboard index</a></p>
  </section>
</main>
</body>
</html>
"""


def build_family_index_html(
    *,
    title: str,
    eyebrow: str,
    heading: str,
    lead_html: str,
    legend_pills: Sequence[str],
    thresholds: pd.DataFrame,
    summary_rows: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
) -> str:
    assumption_html = "".join(f"<li>{html.escape(item)}</li>" for item in assumptions)
    note_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes)
    link_html = "".join(
        (
            "<li>"
            f"<a href='{html.escape(str(row['segment_name']) + '/comparison_dashboard.html')}'>"
            f"{html.escape(str(row['segment_name']))}</a>"
            f" ({html.escape(str(row['segment_rule']))})"
            "</li>"
        )
        for row in summary_rows.to_dict(orient="records")
    )
    summary_display = summary_rows.drop(columns=["segment_rule", "dashboard_path"]).copy()
    legend_html = "".join(legend_pills)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">{html.escape(eyebrow)}</p>
    <h1>{html.escape(heading)}</h1>
    <p class="lead">{lead_html}</p>
    <div class="hero-pills">{legend_html}</div>
  </section>

  <h2>Segment Dashboard Links</h2>
  <section class="panel">
    <ul class="links">{link_html}</ul>
  </section>

  <h2>Segment Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(summary_display)}</div>
  </section>

  <h2>Volatility Threshold Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(threshold_context_table(thresholds))}</div>
  </section>

  <h2>Shared Assumptions</h2>
  <section class="panel">
    <ul>{assumption_html}</ul>
  </section>

  <h2>Run Notes</h2>
  <section class="panel">
    <ul>{note_html}</ul>
  </section>
</main>
</body>
</html>
"""


def build_root_index_html() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model B Volatility Dashboards</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model B Dashboard Index</p>
    <h1>Model B Volatility Dashboard Families</h1>
    <p class="lead">
      This output contains two dark-mode dashboard families built on the same real-time volatility
      segmentation: <code>Model B RT vs Model B Total</code> and <code>Model B RT vs Model K RT</code>.
    </p>
    <div class="hero-pills">
      {pill_html(MODEL_B_TOTAL_LABEL, "total")}
      {pill_html(MODEL_B_RT_LABEL, "vol")}
      {pill_html(MODEL_K_RT_LABEL, "k")}
    </div>
  </section>

  <h2>Available Dashboard Families</h2>
  <section class="panel">
    <ul class="links">
      <li><a href="model_b_rt_vs_total/index.html">Model B RT vs Model B Total</a></li>
      <li><a href="model_b_rt_vs_model_k_rt/index.html">Model B RT vs Model K RT</a></li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def write_segment_artifacts(
    *,
    segment_dir: Path,
    page_title: str,
    eyebrow: str,
    heading: str,
    lead_html: str,
    hero_pills: Sequence[str],
    focus_bundle: Dict[str, Any],
    benchmark_bundle: Dict[str, Any],
    focus_label: str,
    benchmark_label: str,
    focus_color: str,
    benchmark_color: str,
    thresholds: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
    topline_note: str,
) -> None:
    segment_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_comparison_artifacts(focus_bundle=focus_bundle, benchmark_bundle=benchmark_bundle)

    artifacts["overview"].to_csv(segment_dir / "overview_comparison.csv", index=False)
    artifacts["metrics"].to_csv(segment_dir / "metrics_summary_comparison.csv", index=False)
    artifacts["decomposition"].to_csv(segment_dir / "brier_decomposition_comparison.csv", index=False)
    artifacts["calibration"].to_csv(segment_dir / "expanded_calibration_error_summary_comparison.csv", index=False)
    artifacts["sharpness"].to_csv(segment_dir / "sharpness_comparison.csv", index=False)
    artifacts["coverage"].to_csv(segment_dir / "outcome_join_coverage_comparison.csv", index=False)
    artifacts["isolated_metrics"].to_csv(segment_dir / "isolated_minute_metrics_comparison.csv", index=False)
    artifacts["decile_metrics"].to_csv(segment_dir / "ten_minute_bucket_metrics_comparison.csv", index=False)
    artifacts["time_bucket_accuracy"].to_csv(segment_dir / "time_bucket_accuracy_comparison.csv", index=False)
    artifacts["time_bucket_brier"].to_csv(segment_dir / "time_bucket_brier_decomposition_comparison.csv", index=False)
    artifacts["focus_mismatches"].to_csv(segment_dir / "focus_resolution_mismatches.csv", index=False)
    artifacts["benchmark_mismatches"].to_csv(segment_dir / "benchmark_resolution_mismatches.csv", index=False)

    html_document = build_segment_dashboard_html(
        page_title=page_title,
        eyebrow=eyebrow,
        heading=heading,
        lead_html=lead_html,
        hero_pills=hero_pills,
        focus_bundle=focus_bundle,
        benchmark_bundle=benchmark_bundle,
        focus_label=focus_label,
        benchmark_label=benchmark_label,
        focus_color=focus_color,
        benchmark_color=benchmark_color,
        thresholds=thresholds,
        assumptions=assumptions,
        notes=notes,
        artifacts=artifacts,
        topline_note=topline_note,
    )
    (segment_dir / "comparison_dashboard.html").write_text(html_document, encoding="utf-8")


def summary_row(
    *,
    segment: Dict[str, str],
    focus_bundle: Dict[str, Any],
    benchmark_bundle: Dict[str, Any],
    extra_fields: Dict[str, Any],
) -> Dict[str, Any]:
    focus_overall = focus_bundle["overall_metrics"]
    benchmark_overall = benchmark_bundle["overall_metrics"]
    focus_ece = focus_bundle["overall_ece"]
    benchmark_ece = benchmark_bundle["overall_ece"]
    row = {
        "segment_name": segment["name"],
        "segment_rule": segment["rule"],
        "focus_hourly_markets": focus_bundle["n_hourly_markets"],
        "benchmark_hourly_markets": benchmark_bundle["n_hourly_markets"],
        "scored_rows_focus": focus_overall["n_forecasts"],
        "scored_rows_benchmark": benchmark_overall["n_forecasts"],
        "brier_score_focus": focus_overall["brier_score"],
        "brier_score_benchmark": benchmark_overall["brier_score"],
        "log_loss_focus": focus_overall["log_loss"],
        "log_loss_benchmark": benchmark_overall["log_loss"],
        "ece_focus": focus_ece["expected_calibration_error"],
        "ece_benchmark": benchmark_ece["expected_calibration_error"],
        "audit_mismatches_focus": len(focus_bundle["resolution_mismatches"]),
        "audit_mismatches_benchmark": len(benchmark_bundle["resolution_mismatches"]),
        "dashboard_path": f"{segment['name']}/comparison_dashboard.html",
    }
    row.update(extra_fields)
    return row


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    output_root = resolve_path(root, args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_b_forecasts = load_model_b_price_outputs(resolve_path(root, args.model_b_price_dir))
    model_k_forecasts = load_model_k_price_outputs(resolve_path(root, args.model_k_price_dir))
    preferred_states_path = resolve_optional_path(root, args.hourly_market_states_csv)
    if preferred_states_path and preferred_states_path.exists():
        preferred_hourly_states = load_hourly_market_states_csv(preferred_states_path)
        allowed_events = set(preferred_hourly_states["event_ticker"].astype(str))
        model_b_forecasts = model_b_forecasts[model_b_forecasts["event_ticker"].astype(str).isin(allowed_events)].copy()
        model_k_forecasts = model_k_forecasts[model_k_forecasts["event_ticker"].astype(str).isin(allowed_events)].copy()

    settlements = load_settlements(resolve_path(root, args.settlement_csv))
    outcomes, source_notes = build_kalshi_reality_outcomes_k(
        forecasts=model_k_forecasts,
        settlements=settlements,
        output_dir=output_root,
    )
    model_b_raw, model_b_unmatched = attach_outcomes_b(model_b_forecasts, outcomes)
    model_k_raw, model_k_unmatched = attach_outcomes_k(model_k_forecasts, outcomes)

    hourly_market_states, thresholds, state_notes = compute_or_load_hourly_market_states(
        raw=model_b_raw,
        output_root=output_root,
        root=root,
        hourly_market_states_csv=args.hourly_market_states_csv,
        binance_minute_cache_csv=args.binance_minute_cache_csv,
        refresh_binance_cache=args.refresh_binance_cache,
    )
    hourly_market_states = hourly_market_states.sort_values("forecast_hour_start_utc").drop_duplicates(
        subset=["event_ticker"],
        keep="last",
    )
    thresholds.to_csv(output_root / "segment_thresholds.csv", index=False)
    hourly_market_states.to_csv(output_root / "hourly_market_volatility_segments.csv", index=False)

    total_hourly_markets = int(hourly_market_states["event_ticker"].nunique())
    model_b_total_bundle = build_bundle(
        name=MODEL_B_TOTAL_LABEL,
        raw=model_b_raw,
        forecasts=model_b_forecasts,
        unmatched=model_b_unmatched,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
        skip_binance_audit=args.skip_binance_audit,
        n_hourly_markets=total_hourly_markets,
    )

    assumptions_a = list(MODEL_B_ASSUMPTIONS) + list(volatility_assumptions(thresholds))
    assumptions_cross = list(dict.fromkeys([*MODEL_B_ASSUMPTIONS, *MODEL_K_ASSUMPTIONS, *volatility_assumptions(thresholds)]))
    shared_notes = [
        *source_notes,
        *state_notes,
        "Volatility bins use the same trailing 72-hour real-time realized-volatility definition as the existing Model_K volatility scripts.",
        "Dashboard charts and tables are rendered in dark mode and keep the same overall layout as Model_K_Volatility_Results_Dashboard.",
    ]

    family_a_root = output_root / "model_b_rt_vs_total"
    family_k_root = output_root / "model_b_rt_vs_model_k_rt"
    family_a_root.mkdir(parents=True, exist_ok=True)
    family_k_root.mkdir(parents=True, exist_ok=True)

    summary_rows_a: List[Dict[str, Any]] = []
    summary_rows_k: List[Dict[str, Any]] = []

    for segment in SEGMENTS:
        event_ids = set(hourly_market_states.loc[hourly_market_states[segment["flag"]], "event_ticker"].astype(str))

        model_b_segment_forecasts = slice_by_event_ids(model_b_forecasts, event_ids)
        model_b_segment_raw = slice_by_event_ids(model_b_raw, event_ids)
        model_b_segment_unmatched = slice_by_event_ids(model_b_unmatched, event_ids)
        model_b_segment_bundle = build_bundle(
            name=f"{MODEL_B_RT_LABEL}: {segment['name']}",
            raw=model_b_segment_raw,
            forecasts=model_b_segment_forecasts,
            unmatched=model_b_segment_unmatched,
            calibration_bins=args.calibration_bins,
            classification_threshold=args.classification_threshold,
            skip_binance_audit=args.skip_binance_audit,
            n_hourly_markets=len(event_ids),
        )

        model_k_segment_forecasts = slice_by_event_ids(model_k_forecasts, event_ids)
        model_k_segment_raw = slice_by_event_ids(model_k_raw, event_ids)
        model_k_segment_unmatched = slice_by_event_ids(model_k_unmatched, event_ids)

        segment_match_rate = model_b_segment_bundle["coverage"].loc[
            model_b_segment_bundle["coverage"]["scope"] == "overall", "match_rate"
        ].iloc[0]
        total_match_rate = model_b_total_bundle["coverage"].loc[
            model_b_total_bundle["coverage"]["scope"] == "overall", "match_rate"
        ].iloc[0]
        segment_share = (len(event_ids) / total_hourly_markets) if total_hourly_markets else np.nan

        write_segment_artifacts(
            segment_dir=family_a_root / segment["name"],
            page_title=f"Model B RT vs Total: {segment['name']}",
            eyebrow="Model B Volatility Dashboard",
            heading=f"{segment['name']} vs Model B Total",
            lead_html=(
                "This page compares the <code>Model_B.py</code> full-sample evaluation against the "
                f"<code>{html.escape(segment['name'])}</code> slice from "
                "<code>Model_B_Volatility_Decomposition_RT.py</code>. The volatility slice is defined as: "
                f"{html.escape(segment['rule'])}"
            ),
            hero_pills=[
                pill_html(f"{segment['name']} hourly markets: {model_b_segment_bundle['n_hourly_markets']:,}", "vol"),
                pill_html(f"Model B total hourly markets: {total_hourly_markets:,}", "total"),
                pill_html(f"Segment share of markets: {format_number(segment_share, digits=4)}"),
                pill_html(
                    "Outcome match rate: "
                    f"{format_number(segment_match_rate, digits=4)} vs {format_number(total_match_rate, digits=4)}"
                ),
            ],
            focus_bundle=model_b_segment_bundle,
            benchmark_bundle=model_b_total_bundle,
            focus_label=f"{MODEL_B_RT_LABEL}: {segment['name']}",
            benchmark_label=MODEL_B_TOTAL_LABEL,
            focus_color=COLOR_MODEL_B_RT,
            benchmark_color=COLOR_MODEL_B,
            thresholds=thresholds,
            assumptions=assumptions_a,
            notes=[
                *shared_notes,
                "The Model B total baseline includes the volatility segment itself. This is a subset-vs-full-sample comparison, not a disjoint holdout.",
            ],
            topline_note=(
                "The Model B total baseline includes the volatility segment itself. "
                "This dashboard shows how the subset behaves relative to the full scored Model B sample."
            ),
        )
        summary_rows_a.append(
            summary_row(
                segment=segment,
                focus_bundle=model_b_segment_bundle,
                benchmark_bundle=model_b_total_bundle,
                extra_fields={
                    "hourly_market_share": segment_share,
                    "focus_match_rate": segment_match_rate,
                    "benchmark_match_rate": total_match_rate,
                },
            )
        )

        overlap_keys = overlap_comparison_keys(model_b_segment_raw, model_k_segment_raw)
        model_b_overlap_bundle = build_bundle(
            name=f"{MODEL_B_RT_LABEL}: {segment['name']}",
            raw=slice_frame_to_comparison_keys(model_b_segment_raw, overlap_keys),
            forecasts=slice_frame_to_comparison_keys(model_b_segment_forecasts, overlap_keys),
            unmatched=slice_frame_to_comparison_keys(model_b_segment_unmatched, overlap_keys),
            calibration_bins=args.calibration_bins,
            classification_threshold=args.classification_threshold,
            skip_binance_audit=args.skip_binance_audit,
            n_hourly_markets=int(slice_frame_to_comparison_keys(model_b_segment_raw, overlap_keys)["event_ticker"].nunique()),
        )
        model_k_overlap_bundle = build_bundle(
            name=f"{MODEL_K_RT_LABEL}: {segment['name']}",
            raw=slice_frame_to_comparison_keys(model_k_segment_raw, overlap_keys),
            forecasts=slice_frame_to_comparison_keys(model_k_segment_forecasts, overlap_keys),
            unmatched=slice_frame_to_comparison_keys(model_k_segment_unmatched, overlap_keys),
            calibration_bins=args.calibration_bins,
            classification_threshold=args.classification_threshold,
            skip_binance_audit=args.skip_binance_audit,
            n_hourly_markets=int(slice_frame_to_comparison_keys(model_k_segment_raw, overlap_keys)["event_ticker"].nunique()),
        )

        overlap_contracts = int(model_b_overlap_bundle["raw"]["event_contract_id"].nunique()) if not model_b_overlap_bundle["raw"].empty else 0
        match_rate_a_overlap = model_b_overlap_bundle["coverage"].loc[
            model_b_overlap_bundle["coverage"]["scope"] == "overall", "match_rate"
        ].iloc[0]
        match_rate_k_overlap = model_k_overlap_bundle["coverage"].loc[
            model_k_overlap_bundle["coverage"]["scope"] == "overall", "match_rate"
        ].iloc[0]

        write_segment_artifacts(
            segment_dir=family_k_root / segment["name"],
            page_title=f"Model B RT vs Model K RT: {segment['name']}",
            eyebrow="Cross-Model Volatility Dashboard",
            heading=f"Model B RT vs Model K RT: {segment['name']}",
            lead_html=(
                "This page compares <code>Model_B_Volatility_Decomposition_RT.py</code> against "
                "<code>Model_K_Volatility_Decomposition_RT.py</code> on the exact overlapping scored-row universe "
                f"inside <code>{html.escape(segment['name'])}</code>. The volatility slice is defined as: "
                f"{html.escape(segment['rule'])}"
            ),
            hero_pills=[
                pill_html(f"{segment['name']} hourly markets: {len(event_ids):,}", "vol"),
                pill_html(f"Overlap scored rows: {len(overlap_keys):,}"),
                pill_html(f"Overlap event contracts: {overlap_contracts:,}"),
                pill_html(
                    "Overlap match rate: "
                    f"{format_number(match_rate_a_overlap, digits=4)} vs {format_number(match_rate_k_overlap, digits=4)}"
                ),
            ],
            focus_bundle=model_b_overlap_bundle,
            benchmark_bundle=model_k_overlap_bundle,
            focus_label=f"{MODEL_B_RT_LABEL}: {segment['name']}",
            benchmark_label=f"{MODEL_K_RT_LABEL}: {segment['name']}",
            focus_color=COLOR_MODEL_B_RT,
            benchmark_color=COLOR_MODEL_K,
            thresholds=thresholds,
            assumptions=assumptions_cross,
            notes=[
                *shared_notes,
                "Cross-model scoring metrics in this family are computed only on the exact overlapping scored rows keyed by event_contract_id + forecast_datetime_utc.",
            ],
            topline_note=(
                "All comparison metrics on this page are computed on the exact overlapping scored-row universe, "
                "so the Model B and Model K values are directly comparable inside the same volatility slice."
            ),
        )
        summary_rows_k.append(
            summary_row(
                segment=segment,
                focus_bundle=model_b_overlap_bundle,
                benchmark_bundle=model_k_overlap_bundle,
                extra_fields={
                    "segment_hourly_markets": len(event_ids),
                    "overlap_scored_rows": len(overlap_keys),
                    "overlap_event_contracts": overlap_contracts,
                    "focus_match_rate": match_rate_a_overlap,
                    "benchmark_match_rate": match_rate_k_overlap,
                },
            )
        )

    summary_df_a = pd.DataFrame(summary_rows_a)
    summary_df_k = pd.DataFrame(summary_rows_k)
    summary_df_a.to_csv(family_a_root / "segment_dashboard_summary.csv", index=False)
    summary_df_k.to_csv(family_k_root / "segment_dashboard_summary.csv", index=False)

    (family_a_root / "index.html").write_text(
        build_family_index_html(
            title="Model B RT vs Total Dashboards",
            eyebrow="Model B Dashboard Index",
            heading="Model B Total vs Volatility Segment Comparisons",
            lead_html=(
                "This output keeps the same scoring pipeline from <code>Model_B.py</code> and "
                "<code>Model_B_Volatility_Decomposition_RT.py</code>, but reorganizes the results into "
                "comparison dashboards that line up each volatility slice against the full Model B baseline."
            ),
            legend_pills=[
                pill_html(MODEL_B_TOTAL_LABEL, "total"),
                pill_html(MODEL_B_RT_LABEL, "vol"),
            ],
            thresholds=thresholds,
            summary_rows=summary_df_a,
            assumptions=assumptions_a,
            notes=shared_notes,
        ),
        encoding="utf-8",
    )
    (family_k_root / "index.html").write_text(
        build_family_index_html(
            title="Model B RT vs Model K RT Dashboards",
            eyebrow="Cross-Model Dashboard Index",
            heading="Model B RT vs Model K RT Comparisons",
            lead_html=(
                "This output compares <code>Model_B_Volatility_Decomposition_RT.py</code> against "
                "<code>Model_K_Volatility_Decomposition_RT.py</code> within each volatility segment on the exact "
                "overlapping scored-row universe."
            ),
            legend_pills=[
                pill_html(MODEL_B_RT_LABEL, "vol"),
                pill_html(MODEL_K_RT_LABEL, "k"),
            ],
            thresholds=thresholds,
            summary_rows=summary_df_k,
            assumptions=assumptions_cross,
            notes=[
                *shared_notes,
                "Cross-model metrics are overlap-restricted by event_contract_id + forecast_datetime_utc before scoring.",
            ],
        ),
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(build_root_index_html(), encoding="utf-8")

    print(f"Comparison dashboards written to: {output_root}")
    print(f"Root index: {output_root / 'index.html'}")
    print(f"Model B RT vs total index: {family_a_root / 'index.html'}")
    print(f"Model B RT vs Model K RT index: {family_k_root / 'index.html'}")


if __name__ == "__main__":
    main()
