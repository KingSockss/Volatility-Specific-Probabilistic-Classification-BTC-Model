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


OUTPUT_FOLDER_NAME = "outputs"
DEFAULT_PRIMARY_STRATEGY_ID = "model_n_exp_lambda_2"
HAR_FEATURES = ["rv_lag_1h", "rv_mean_6h", "rv_mean_24h", "rv_mean_72h"]
QUANTILE_SWEEP = tuple(value / 100 for value in range(50, 100, 5))
STRATEGY_COLORS = {
    "Model N Hybrid": "#6cb6ff",
    "Model N Exp Hybrid lambda=2": "#fbbf24",
    "Model A Normal": "#35c7b7",
    "Model B Shock": "#ff8f70",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def quantile_label(quantile: float) -> str:
    return f"q{int(round(quantile * 100))}"


def historic_rv_column(quantile: float) -> str:
    return f"historic_rv_{quantile_label(quantile)}"


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Naive causal HAR-RV high-volatility filter experiment. Trades only when the expanding-window "
            "HAR-RV forecast is above the historical q75 threshold of prior realized volatility."
        )
    )
    parser.add_argument(
        "--trade-ledger",
        type=Path,
        default=root / "Model_N" / "Model_N_Trade_Diagnostics_Outputs" / "trade_diagnostics_ledger.csv",
        help="Model N trade diagnostics ledger with per-trade net_pnl.",
    )
    parser.add_argument(
        "--hourly-volatility",
        type=Path,
        default=root
        / "Model_K_Volatility_Decomposition_RT"
        / "Model_K_Volatility_Decomposition_RT_outputs"
        / "hourly_market_volatility_segments.csv",
        help="Hourly realized-volatility table used to build HAR-RV forecasts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Output folder for HAR-RV filter experiment artifacts.",
    )
    parser.add_argument(
        "--primary-strategy-id",
        default=DEFAULT_PRIMARY_STRATEGY_ID,
        help="Strategy highlighted in the dashboard.",
    )
    parser.add_argument(
        "--min-train-hours",
        type=int,
        default=168,
        help="Minimum prior complete hourly observations required before fitting HAR-RV.",
    )
    parser.add_argument(
        "--min-q75-history",
        type=int,
        default=72,
        help="Minimum prior realized-volatility observations required before computing the historical q75 threshold.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_trade_ledger(path: Path) -> pd.DataFrame:
    require_file(path, "Trade ledger")
    frame = pd.read_csv(path)
    required = {
        "strategy_id",
        "strategy_label",
        "timestamp",
        "side",
        "selected_side",
        "event_contract_id",
        "market",
        "contract_label",
        "strike",
        "expiry",
        "market_price",
        "fee",
        "gross_edge",
        "net_expected_edge",
        "gross_pnl",
        "net_pnl",
        "win",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["forecast_hour_start_utc"] = frame["timestamp"].dt.floor("h")
    if "expiry" in frame.columns:
        frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
    numeric_cols = [
        "market_price",
        "fee",
        "gross_edge",
        "net_expected_edge",
        "gross_pnl",
        "net_pnl",
        "win",
        "time_to_expiry",
        "q_shock",
        "model_disagreement_abs",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["strategy_id", "timestamp", "event_contract_id", "side"]).reset_index(drop=True)


def load_hourly_volatility(path: Path) -> pd.DataFrame:
    require_file(path, "Hourly volatility")
    frame = pd.read_csv(path)
    required = {
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "realized_volatility",
        "primary_volatility_band",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    keep = [
        "event_ticker",
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "realized_volatility",
        "realized_variance",
        "primary_volatility_band",
        "rolling_percentile_rank",
        "q25_realized_volatility",
        "q75_realized_volatility",
    ]
    frame = frame[[col for col in keep if col in frame.columns]].copy()
    frame["forecast_hour_start_utc"] = pd.to_datetime(frame["forecast_hour_start_utc"], utc=True)
    frame["event_datetime_utc"] = pd.to_datetime(frame["event_datetime_utc"], utc=True)
    for column in frame.columns:
        if column not in {"event_ticker", "forecast_hour_start_utc", "event_datetime_utc", "primary_volatility_band"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["forecast_hour_start_utc", "realized_volatility"]).sort_values(
        "forecast_hour_start_utc"
    )
    return frame.drop_duplicates(subset=["forecast_hour_start_utc"]).reset_index(drop=True)


def add_har_features(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy()
    prior_rv = out["realized_volatility"].shift(1)
    out["rv_lag_1h"] = prior_rv
    out["rv_mean_6h"] = prior_rv.rolling(6, min_periods=6).mean()
    out["rv_mean_24h"] = prior_rv.rolling(24, min_periods=24).mean()
    out["rv_mean_72h"] = prior_rv.rolling(72, min_periods=72).mean()
    return out


def fit_predict_expanding_har(
    hourly: pd.DataFrame,
    *,
    min_train_hours: int,
    min_q75_history: int,
) -> pd.DataFrame:
    out = add_har_features(hourly)
    out["har_rv_forecast"] = np.nan
    out["har_intercept"] = np.nan
    out["har_beta_lag_1h"] = np.nan
    out["har_beta_mean_6h"] = np.nan
    out["har_beta_mean_24h"] = np.nan
    out["har_beta_mean_72h"] = np.nan
    out["har_train_rows"] = 0

    complete = out.dropna(subset=HAR_FEATURES + ["realized_volatility"]).copy()
    complete_indices = complete.index.to_numpy()
    for idx in complete_indices:
        train = out.loc[: idx - 1].dropna(subset=HAR_FEATURES + ["realized_volatility"]).copy()
        if len(train) < min_train_hours:
            continue

        x_train = np.column_stack([np.ones(len(train)), train[HAR_FEATURES].to_numpy(dtype=float)])
        y_train = train["realized_volatility"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        x_now = np.array([1.0] + [float(out.at[idx, feature]) for feature in HAR_FEATURES])
        prediction = float(max(0.0, x_now @ beta))

        out.at[idx, "har_rv_forecast"] = prediction
        out.at[idx, "har_intercept"] = beta[0]
        out.at[idx, "har_beta_lag_1h"] = beta[1]
        out.at[idx, "har_beta_mean_6h"] = beta[2]
        out.at[idx, "har_beta_mean_24h"] = beta[3]
        out.at[idx, "har_beta_mean_72h"] = beta[4]
        out.at[idx, "har_train_rows"] = len(train)

    prior_realized_volatility = out["realized_volatility"].shift(1)
    for quantile in QUANTILE_SWEEP:
        out[historic_rv_column(quantile)] = (
            prior_realized_volatility.expanding(min_periods=min_q75_history).quantile(quantile)
        )
    out["historic_har_forecast_q75"] = (
        out["har_rv_forecast"].shift(1).expanding(min_periods=min_q75_history).quantile(0.75)
    )
    out["har_rv_active"] = out["har_rv_forecast"] >= out["historic_rv_q75"]
    out.loc[out["historic_rv_q75"].isna(), "har_rv_active"] = False
    return out


def merge_trades_with_har(trades: pd.DataFrame, har: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "forecast_hour_start_utc",
        "event_datetime_utc",
        "realized_volatility",
        "primary_volatility_band",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "rv_lag_1h",
        "rv_mean_6h",
        "rv_mean_24h",
        "rv_mean_72h",
        "har_rv_forecast",
        "historic_rv_q75",
        "historic_har_forecast_q75",
        "har_rv_active",
        "har_train_rows",
    ]
    merged = trades.merge(
        har[[col for col in keep if col in har.columns]],
        on="forecast_hour_start_utc",
        how="left",
        validate="many_to_one",
        suffixes=("", "_har"),
    )
    for column in ["realized_volatility", "rolling_percentile_rank"]:
        har_column = f"{column}_har"
        if har_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[har_column])
        else:
            merged[column] = merged[har_column]
        merged = merged.drop(columns=[har_column])
    merged["har_rv_active"] = merged["har_rv_active"].fillna(False).astype(bool)
    merged["trade_filter"] = np.where(merged["har_rv_active"], "HAR-RV q75 active", "HAR-RV inactive")
    return merged


def summarize_group(frame: pd.DataFrame) -> Dict[str, Any]:
    rows = int(len(frame))
    total_cost = float(frame["market_price"].sum()) if rows else 0.0
    total_cost_after_fees = float((frame["market_price"] + frame["fee"]).sum()) if rows else 0.0
    total_net_pnl = float(frame["net_pnl"].sum()) if rows else 0.0
    return {
        "trade_count": rows,
        "timestamp_count": int(frame["timestamp"].nunique()) if rows else 0,
        "active_hour_count": int(frame["forecast_hour_start_utc"].nunique()) if rows else 0,
        "winrate": float(frame["win"].mean()) if rows else np.nan,
        "total_gross_pnl": float(frame["gross_pnl"].sum()) if rows else 0.0,
        "total_fees": float(frame["fee"].sum()) if rows else 0.0,
        "total_net_pnl": total_net_pnl,
        "mean_net_pnl": float(frame["net_pnl"].mean()) if rows else np.nan,
        "mean_gross_edge": float(frame["gross_edge"].mean()) if rows else np.nan,
        "mean_net_expected_edge": float(frame["net_expected_edge"].mean()) if rows else np.nan,
        "mean_market_price": float(frame["market_price"].mean()) if rows else np.nan,
        "mean_har_rv_forecast": float(frame["har_rv_forecast"].mean()) if rows else np.nan,
        "mean_realized_volatility": float(frame["realized_volatility"].mean()) if rows else np.nan,
        "total_cost": total_cost,
        "total_cost_after_fees": total_cost_after_fees,
        "roi_after_fees": total_net_pnl / total_cost_after_fees if total_cost_after_fees else np.nan,
    }


def build_strategy_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (strategy_id, strategy_label), strategy_trades in merged.groupby(["strategy_id", "strategy_label"], sort=False):
        baseline = summarize_group(strategy_trades)
        filtered = summarize_group(strategy_trades[strategy_trades["har_rv_active"]])
        row = {
            "strategy_id": strategy_id,
            "strategy_label": strategy_label,
            "baseline_trade_count": baseline["trade_count"],
            "filtered_trade_count": filtered["trade_count"],
            "trade_keep_rate": filtered["trade_count"] / baseline["trade_count"] if baseline["trade_count"] else np.nan,
            "baseline_total_net_pnl": baseline["total_net_pnl"],
            "filtered_total_net_pnl": filtered["total_net_pnl"],
            "delta_net_pnl_vs_baseline": filtered["total_net_pnl"] - baseline["total_net_pnl"],
            "baseline_mean_net_pnl": baseline["mean_net_pnl"],
            "filtered_mean_net_pnl": filtered["mean_net_pnl"],
            "baseline_total_fees": baseline["total_fees"],
            "filtered_total_fees": filtered["total_fees"],
            "baseline_roi_after_fees": baseline["roi_after_fees"],
            "filtered_roi_after_fees": filtered["roi_after_fees"],
            "baseline_winrate": baseline["winrate"],
            "filtered_winrate": filtered["winrate"],
            "filtered_active_hour_count": filtered["active_hour_count"],
            "filtered_timestamp_count": filtered["timestamp_count"],
            "filtered_mean_har_rv_forecast": filtered["mean_har_rv_forecast"],
            "filtered_mean_realized_volatility": filtered["mean_realized_volatility"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_quantile_sweep_summary(
    merged: pd.DataFrame,
    har: pd.DataFrame,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for quantile in quantiles:
        label = quantile_label(quantile)
        threshold_column = historic_rv_column(quantile)
        if threshold_column not in har.columns:
            continue

        forecastable = har["har_rv_forecast"].notna() & har[threshold_column].notna()
        active_hours = har.loc[
            forecastable & (har["har_rv_forecast"] >= har[threshold_column]),
            "forecast_hour_start_utc",
        ]
        active_hour_set = set(active_hours)
        active_hour_count = int(len(active_hour_set))
        forecastable_hour_count = int(forecastable.sum())
        active_hour_rate = active_hour_count / forecastable_hour_count if forecastable_hour_count else np.nan

        active_hour_frame = har[har["forecast_hour_start_utc"].isin(active_hour_set)].copy()
        for (strategy_id, strategy_label), strategy_trades in merged.groupby(["strategy_id", "strategy_label"], sort=False):
            baseline = summarize_group(strategy_trades)
            filtered = summarize_group(strategy_trades[strategy_trades["forecast_hour_start_utc"].isin(active_hour_set)])
            rows.append(
                {
                    "threshold_label": label,
                    "threshold_percentile": int(round(quantile * 100)),
                    "threshold_quantile": quantile,
                    "threshold_column": threshold_column,
                    "forecastable_hour_count": forecastable_hour_count,
                    "active_hour_count": active_hour_count,
                    "active_hour_rate": active_hour_rate,
                    "mean_active_har_rv_forecast": (
                        float(active_hour_frame["har_rv_forecast"].mean()) if active_hour_count else np.nan
                    ),
                    "mean_active_realized_volatility": (
                        float(active_hour_frame["realized_volatility"].mean()) if active_hour_count else np.nan
                    ),
                    "strategy_id": strategy_id,
                    "strategy_label": strategy_label,
                    "baseline_trade_count": baseline["trade_count"],
                    "filtered_trade_count": filtered["trade_count"],
                    "trade_keep_rate": (
                        filtered["trade_count"] / baseline["trade_count"] if baseline["trade_count"] else np.nan
                    ),
                    "baseline_total_net_pnl": baseline["total_net_pnl"],
                    "filtered_total_net_pnl": filtered["total_net_pnl"],
                    "delta_net_pnl_vs_baseline": filtered["total_net_pnl"] - baseline["total_net_pnl"],
                    "baseline_mean_net_pnl": baseline["mean_net_pnl"],
                    "filtered_mean_net_pnl": filtered["mean_net_pnl"],
                    "baseline_total_fees": baseline["total_fees"],
                    "filtered_total_fees": filtered["total_fees"],
                    "baseline_roi_after_fees": baseline["roi_after_fees"],
                    "filtered_roi_after_fees": filtered["roi_after_fees"],
                    "baseline_winrate": baseline["winrate"],
                    "filtered_winrate": filtered["winrate"],
                    "filtered_active_hour_count": filtered["active_hour_count"],
                    "filtered_timestamp_count": filtered["timestamp_count"],
                    "filtered_mean_har_rv_forecast": filtered["mean_har_rv_forecast"],
                    "filtered_mean_realized_volatility": filtered["mean_realized_volatility"],
                }
            )
    return pd.DataFrame(rows)


def build_quantile_sweep_matrix(sweep_summary: pd.DataFrame) -> pd.DataFrame:
    if sweep_summary.empty:
        return pd.DataFrame()

    base = (
        sweep_summary[
            [
                "threshold_label",
                "threshold_percentile",
                "forecastable_hour_count",
                "active_hour_count",
                "active_hour_rate",
                "mean_active_har_rv_forecast",
                "mean_active_realized_volatility",
            ]
        ]
        .drop_duplicates(subset=["threshold_label"])
        .set_index("threshold_label")
    )
    metrics = [
        "filtered_total_net_pnl",
        "filtered_trade_count",
        "trade_keep_rate",
        "filtered_mean_net_pnl",
        "filtered_roi_after_fees",
        "filtered_winrate",
        "filtered_total_fees",
    ]
    matrix = base.copy()
    for metric in metrics:
        pivot = sweep_summary.pivot(index="threshold_label", columns="strategy_id", values=metric)
        pivot.columns = [f"{strategy_id}_{metric}" for strategy_id in pivot.columns]
        matrix = matrix.join(pivot)
    return matrix.reset_index().sort_values("threshold_percentile").reset_index(drop=True)


def build_segment_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    segment_fields = [
        "har_rv_active",
        "primary_volatility_band",
        "side",
        "selected_side",
        "contract_label",
        "predicted_net_edge_bucket",
        "q_shock_bucket",
        "time_to_expiry_bucket",
        "model_disagreement_bucket",
    ]
    for (strategy_id, strategy_label), strategy_trades in merged.groupby(["strategy_id", "strategy_label"], sort=False):
        for field in segment_fields:
            if field not in strategy_trades.columns:
                continue
            work = strategy_trades.copy()
            work[field] = work[field].astype(object).where(work[field].notna(), "missing")
            for value, part in work.groupby(field, sort=False, dropna=False):
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy_label": strategy_label,
                        "segment_field": field,
                        "segment_value": str(value),
                    }
                    | summarize_group(part)
                )
    return pd.DataFrame(rows)


def build_hourly_summary(har: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    trade_hours = (
        merged.groupby("forecast_hour_start_utc", as_index=False)
        .agg(
            total_trades=("event_contract_id", "size"),
            total_net_pnl=("net_pnl", "sum"),
            active_trades=("har_rv_active", "sum"),
        )
    )
    hourly = har.merge(trade_hours, on="forecast_hour_start_utc", how="left")
    hourly[["total_trades", "total_net_pnl", "active_trades"]] = hourly[
        ["total_trades", "total_net_pnl", "active_trades"]
    ].fillna(0.0)
    return hourly


def build_pnl_timeseries(merged: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for label, part in [("baseline", merged), ("har_rv_filtered", merged[merged["har_rv_active"]])]:
        if part.empty:
            continue
        grouped = (
            part.groupby(["strategy_id", "strategy_label", "timestamp"], as_index=False)
            .agg(
                period_net_pnl=("net_pnl", "sum"),
                period_gross_pnl=("gross_pnl", "sum"),
                period_fees=("fee", "sum"),
                period_trades=("event_contract_id", "size"),
            )
            .sort_values(["strategy_id", "timestamp"])
            .reset_index(drop=True)
        )
        grouped["scenario_id"] = label
        grouped["scenario_label"] = "Baseline" if label == "baseline" else "HAR-RV q75 Filtered"
        grouped["cumulative_net_pnl"] = grouped.groupby("strategy_id")["period_net_pnl"].cumsum()
        grouped["cumulative_gross_pnl"] = grouped.groupby("strategy_id")["period_gross_pnl"].cumsum()
        grouped["cumulative_fees"] = grouped.groupby("strategy_id")["period_fees"].cumsum()
        grouped["cumulative_trades"] = grouped.groupby("strategy_id")["period_trades"].cumsum()
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["strategy_id", "scenario_id", "timestamp"]).reset_index(drop=True)


def build_confusion_summary(har: pd.DataFrame) -> pd.DataFrame:
    work = har.dropna(subset=["har_rv_forecast", "historic_rv_q75", "primary_volatility_band"]).copy()
    work["posthoc_high_vol"] = work["primary_volatility_band"] == "High Volatility"
    work["har_rv_active"] = work["har_rv_active"].astype(bool)
    rows: List[Dict[str, Any]] = []
    for (active, posthoc), part in work.groupby(["har_rv_active", "posthoc_high_vol"], dropna=False):
        rows.append(
            {
                "har_rv_active": bool(active),
                "posthoc_high_vol": bool(posthoc),
                "hours": int(len(part)),
                "mean_har_rv_forecast": float(part["har_rv_forecast"].mean()),
                "mean_realized_volatility": float(part["realized_volatility"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
        if any(token in lower for token in ["count", "rows", "trades", "hours", "timestamps"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=0))
        elif any(token in lower for token in ["rate", "roi", "winrate", "keep"]):
            out[column] = out[column].map(lambda v: format_number(v, percent=True))
        elif any(token in lower for token in ["pnl", "fee", "edge", "price", "volatility", "forecast"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower or "delta" in lower))
        else:
            out[column] = out[column].map(
                lambda v: format_number(v, digits=4) if isinstance(v, (int, float, np.integer, np.floating)) else v
            )
    return out


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def plotly_theme(fig: Any, *, height: int, top_margin: int = 92) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1b26",
        plot_bgcolor="#0d1b26",
        font=dict(color="#e8f1fa"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.12,
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


def har_forecast_figure(hourly: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["forecast_hour_start_utc"],
            y=hourly["har_rv_forecast"],
            mode="lines",
            name="HAR-RV forecast",
            line=dict(color="#fbbf24", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["forecast_hour_start_utc"],
            y=hourly["historic_rv_q75"],
            mode="lines",
            name="Historical realized-vol q75 threshold",
            line=dict(color="#ff8f70", width=1.8, dash="dash"),
        )
    )
    active = hourly[hourly["har_rv_active"]].copy()
    fig.add_trace(
        go.Scatter(
            x=active["forecast_hour_start_utc"],
            y=active["har_rv_forecast"],
            mode="markers",
            name="Active hours",
            marker=dict(color="#35c7b7", size=7),
        )
    )
    fig.update_layout(title="Causal HAR-RV Forecast And Historical Realized-Vol q75 Activation")
    fig.update_xaxes(title_text="Forecast hour")
    fig.update_yaxes(title_text="Forecast realized volatility")
    plotly_theme(fig, height=520, top_margin=96)
    return fig


def pnl_overlay_figure(timeseries: pd.DataFrame, *, primary_strategy_id: str) -> Any:
    fig = go.Figure()
    part = timeseries[timeseries["strategy_id"] == primary_strategy_id].copy()
    for scenario_id, scenario_part in part.groupby("scenario_id", sort=False):
        label = str(scenario_part["scenario_label"].iloc[0])
        color = "#9fb4c7" if scenario_id == "baseline" else "#fbbf24"
        fig.add_trace(
            go.Scatter(
                x=scenario_part["timestamp"],
                y=scenario_part["cumulative_net_pnl"],
                mode="lines",
                name=label,
                line=dict(color=color, width=3),
            )
        )
    fig.update_layout(title="Primary Strategy Cumulative Net PnL: Baseline Vs HAR-RV Filter")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Cumulative net PnL after fees")
    plotly_theme(fig, height=560, top_margin=96)
    return fig


def strategy_bar_figure(summary: pd.DataFrame) -> Any:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total Net PnL", "Trades Kept"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(x=summary["strategy_label"], y=summary["baseline_total_net_pnl"], name="Baseline", marker_color="#9fb4c7"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary["strategy_label"],
            y=summary["filtered_total_net_pnl"],
            name="HAR-RV q75 Filtered",
            marker_color="#fbbf24",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=summary["strategy_label"], y=summary["filtered_trade_count"], showlegend=False, marker_color="#35c7b7"),
        row=1,
        col=2,
    )
    fig.update_layout(title="HAR-RV Filter Strategy Comparison", barmode="group")
    fig.update_yaxes(title_text="Net PnL", row=1, col=1)
    fig.update_yaxes(title_text="Trades", row=1, col=2)
    plotly_theme(fig, height=560, top_margin=106)
    return fig


def quantile_sweep_figure(sweep_summary: pd.DataFrame) -> Any:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Filtered Net PnL After Fees", "Trade Keep Rate"),
        horizontal_spacing=0.14,
    )
    for strategy_label, part in sweep_summary.groupby("strategy_label", sort=False):
        color = STRATEGY_COLORS.get(str(strategy_label), None)
        part = part.sort_values("threshold_percentile")
        fig.add_trace(
            go.Scatter(
                x=part["threshold_percentile"],
                y=part["filtered_total_net_pnl"],
                mode="lines+markers",
                name=strategy_label,
                line=dict(color=color, width=2.4) if color else dict(width=2.4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=part["threshold_percentile"],
                y=part["trade_keep_rate"],
                mode="lines+markers",
                name=strategy_label,
                line=dict(color=color, width=2.4) if color else dict(width=2.4),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_layout(title="HAR-RV Historical Realized-Vol Quantile Sweep")
    fig.update_xaxes(title_text="Activation threshold percentile", row=1, col=1)
    fig.update_xaxes(title_text="Activation threshold percentile", row=1, col=2)
    fig.update_yaxes(title_text="Net PnL after fees", row=1, col=1)
    fig.update_yaxes(title_text="Trades kept", tickformat=".0%", row=1, col=2)
    plotly_theme(fig, height=560, top_margin=106)
    return fig


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
.lead { max-width: 1080px; color: var(--muted); line-height: 1.55; margin: 0; }
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
</style>
"""


def dashboard_cards(summary: pd.DataFrame, hourly: pd.DataFrame, *, primary_strategy_id: str) -> str:
    primary = summary[summary["strategy_id"] == primary_strategy_id]
    row = primary.iloc[0] if not primary.empty else summary.iloc[0]
    active_hours = int(hourly["har_rv_active"].sum())
    forecastable_hours = int(hourly["historic_rv_q75"].notna().sum())
    cards = [
        ("Primary Strategy", str(row["strategy_label"])),
        ("Filtered Net PnL", format_number(row["filtered_total_net_pnl"], digits=3, signed=True)),
        ("Baseline Net PnL", format_number(row["baseline_total_net_pnl"], digits=3, signed=True)),
        ("Trades Kept", format_number(row["trade_keep_rate"], percent=True)),
        ("Active Hours", f"{active_hours:,} / {forecastable_hours:,}"),
        ("Filtered PnL / Trade", format_number(row["filtered_mean_net_pnl"], digits=4, signed=True)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def build_dashboard(
    *,
    hourly: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    confusion_summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    quantile_sweep_summary: pd.DataFrame,
    quantile_sweep_matrix: pd.DataFrame,
    primary_strategy_id: str,
    min_train_hours: int,
    min_q75_history: int,
) -> str:
    primary_segments = segment_summary[
        (segment_summary["strategy_id"] == primary_strategy_id)
        & (segment_summary["segment_field"].isin(["har_rv_active", "primary_volatility_band", "side", "predicted_net_edge_bucket"]))
    ].copy()
    figures = [
        har_forecast_figure(hourly),
        pnl_overlay_figure(timeseries, primary_strategy_id=primary_strategy_id),
        strategy_bar_figure(strategy_summary),
        quantile_sweep_figure(quantile_sweep_summary),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HAR-RV q75 Filter Experiment</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model N Experiment</p>
    <h1>HAR-RV q75 Trading Filter</h1>
    <p class="lead">
      A naive expanding-window HAR-RV model forecasts next-hour realized volatility from prior hourly
      volatility only: <code>lag_1h</code>, <code>mean_6h</code>, <code>mean_24h</code>, and
      <code>mean_72h</code>. Trading is active when the HAR-RV forecast is above the historical q75 of
      prior realized volatility. Minimum training hours = <code>{min_train_hours}</code>; minimum q75 history =
      <code>{min_q75_history}</code>. Post-hoc volatility labels are shown only for attribution.
    </p>
    <div class="cards">{dashboard_cards(strategy_summary, hourly, primary_strategy_id=primary_strategy_id)}</div>
  </section>

  <h2>Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(strategy_summary))}</div></section>

  <h2>Primary Strategy Segments</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(primary_segments))}</div></section>

  <h2>HAR-RV Active Vs Post-Hoc High-Vol Hours</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(confusion_summary))}</div></section>

  <h2>q50-q95 Quantile Sweep Matrix</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(quantile_sweep_matrix))}</div></section>
</main>
</body>
</html>
"""


def write_readme(
    output_dir: Path,
    *,
    primary_strategy_id: str,
    min_train_hours: int,
    min_q75_history: int,
) -> None:
    text = f"""HAR-RV q75 Filter Experiment

Purpose:
- Test a naive causal HAR-RV high-volatility activation filter on Model N trade diagnostics.

HAR-RV setup:
- Target = next/current forecast-hour realized volatility from the hourly volatility table.
- Features use only prior realized volatility:
  - rv_lag_1h
  - rv_mean_6h
  - rv_mean_24h
  - rv_mean_72h
- Expanding OLS is refit each hour using prior complete observations only.
- min_train_hours = {min_train_hours}

Activation rule:
- Compute historical q75 from prior realized volatility only.
- Activate trading when har_rv_forecast >= historic_rv_q75.
- min_q75_history = {min_q75_history}

Primary dashboard strategy:
- {primary_strategy_id}

Outputs:
- har_rv_hourly_forecasts.csv
- har_rv_filtered_trade_ledger.csv
- har_rv_filtered_trade_ledger_active_only.csv
- har_rv_strategy_summary.csv
- har_rv_segment_summary.csv
- har_rv_pnl_timeseries.csv
- har_rv_active_vs_posthoc_high_vol.csv
- har_rv_quantile_sweep_strategy_summary.csv
- har_rv_quantile_sweep_matrix.csv
- har_rv_filter_dashboard.html
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_train_hours < 1:
        raise ValueError("--min-train-hours must be positive.")
    if args.min_q75_history < 1:
        raise ValueError("--min-q75-history must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trades = load_trade_ledger(args.trade_ledger.resolve())
    hourly = load_hourly_volatility(args.hourly_volatility.resolve())
    har = fit_predict_expanding_har(
        hourly,
        min_train_hours=args.min_train_hours,
        min_q75_history=args.min_q75_history,
    )
    merged = merge_trades_with_har(trades, har)
    active_only = merged[merged["har_rv_active"]].copy()
    hourly_summary = build_hourly_summary(har, merged)
    strategy_summary = build_strategy_summary(merged)
    segment_summary = build_segment_summary(merged)
    timeseries = build_pnl_timeseries(merged)
    confusion_summary = build_confusion_summary(har)
    quantile_sweep_summary = build_quantile_sweep_summary(merged, har, QUANTILE_SWEEP)
    quantile_sweep_matrix = build_quantile_sweep_matrix(quantile_sweep_summary)

    har.to_csv(output_dir / "har_rv_hourly_forecasts.csv", index=False)
    hourly_summary.to_csv(output_dir / "har_rv_hourly_trade_summary.csv", index=False)
    merged.to_csv(output_dir / "har_rv_filtered_trade_ledger.csv", index=False)
    active_only.to_csv(output_dir / "har_rv_filtered_trade_ledger_active_only.csv", index=False)
    strategy_summary.to_csv(output_dir / "har_rv_strategy_summary.csv", index=False)
    segment_summary.to_csv(output_dir / "har_rv_segment_summary.csv", index=False)
    timeseries.to_csv(output_dir / "har_rv_pnl_timeseries.csv", index=False)
    confusion_summary.to_csv(output_dir / "har_rv_active_vs_posthoc_high_vol.csv", index=False)
    quantile_sweep_summary.to_csv(output_dir / "har_rv_quantile_sweep_strategy_summary.csv", index=False)
    quantile_sweep_matrix.to_csv(output_dir / "har_rv_quantile_sweep_matrix.csv", index=False)

    dashboard = build_dashboard(
        hourly=hourly_summary,
        strategy_summary=strategy_summary,
        segment_summary=segment_summary,
        confusion_summary=confusion_summary,
        timeseries=timeseries,
        quantile_sweep_summary=quantile_sweep_summary,
        quantile_sweep_matrix=quantile_sweep_matrix,
        primary_strategy_id=args.primary_strategy_id,
        min_train_hours=args.min_train_hours,
        min_q75_history=args.min_q75_history,
    )
    (output_dir / "har_rv_filter_dashboard.html").write_text(dashboard, encoding="utf-8")
    write_readme(
        output_dir,
        primary_strategy_id=args.primary_strategy_id,
        min_train_hours=args.min_train_hours,
        min_q75_history=args.min_q75_history,
    )

    display_cols = [
        "strategy_label",
        "baseline_trade_count",
        "filtered_trade_count",
        "trade_keep_rate",
        "baseline_total_net_pnl",
        "filtered_total_net_pnl",
        "delta_net_pnl_vs_baseline",
        "filtered_mean_net_pnl",
        "filtered_roi_after_fees",
    ]
    print(f"HAR-RV filter output directory: {output_dir}")
    print(strategy_summary[display_cols].to_string(index=False))
    primary_sweep = quantile_sweep_summary[quantile_sweep_summary["strategy_id"] == args.primary_strategy_id]
    sweep_display_cols = [
        "threshold_label",
        "active_hour_count",
        "filtered_trade_count",
        "trade_keep_rate",
        "filtered_total_net_pnl",
        "filtered_mean_net_pnl",
        "filtered_roi_after_fees",
    ]
    if not primary_sweep.empty:
        print("\nPrimary strategy q50-q95 sweep:")
        print(primary_sweep[sweep_display_cols].to_string(index=False))
    print(f"Dashboard: {output_dir / 'har_rv_filter_dashboard.html'}")


if __name__ == "__main__":
    main()
