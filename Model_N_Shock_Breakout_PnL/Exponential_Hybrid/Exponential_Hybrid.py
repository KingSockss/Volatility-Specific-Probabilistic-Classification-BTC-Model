from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
DEFAULT_LAMBDA = 2.0
EPSILON = 1e-12


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Standalone exponential-hybrid experiment. Computes "
            "p_final = (1 - w(q)) * p_normal + w(q) * p_shock, where "
            "w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)."
        )
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
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Self-contained output folder for this experiment.",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Positive exponential shock-weight lambda. Values near zero approach linear weighting.",
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
        "p_normal",
        "p_shock",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def exponential_weight(q: pd.Series, lambda_value: float) -> pd.Series:
    q = pd.to_numeric(q, errors="coerce").clip(0.0, 1.0)
    if abs(lambda_value) < 1e-9:
        return q
    denominator = np.expm1(lambda_value)
    if abs(denominator) < EPSILON:
        return q
    return pd.Series(np.expm1(lambda_value * q) / denominator, index=q.index).clip(0.0, 1.0)


def add_exponential_hybrid(model_n: pd.DataFrame, *, lambda_value: float) -> pd.DataFrame:
    required = {"q_shock", "p_normal", "p_shock"}
    missing = required - set(model_n.columns)
    if missing:
        raise ValueError(f"Model N raw values are missing required hybrid columns: {sorted(missing)}")

    out = model_n.copy()
    out["q_shock"] = pd.to_numeric(out["q_shock"], errors="coerce").clip(0.0, 1.0)
    out["p_normal"] = pd.to_numeric(out["p_normal"], errors="coerce").clip(0.0, 1.0)
    out["p_shock"] = pd.to_numeric(out["p_shock"], errors="coerce").clip(0.0, 1.0)
    out = out.dropna(subset=["q_shock", "p_normal", "p_shock"]).copy()

    out["lambda_value"] = float(lambda_value)
    out["shock_weight_linear"] = out["q_shock"]
    out["shock_weight_exponential"] = exponential_weight(out["q_shock"], lambda_value)
    out["p_linear_hybrid_recomputed"] = (
        (1.0 - out["shock_weight_linear"]) * out["p_normal"] + out["shock_weight_linear"] * out["p_shock"]
    ).clip(0.0, 1.0)
    out["p_exponential_hybrid"] = (
        (1.0 - out["shock_weight_exponential"]) * out["p_normal"]
        + out["shock_weight_exponential"] * out["p_shock"]
    ).clip(0.0, 1.0)
    out["p_exponential_minus_linear"] = out["p_exponential_hybrid"] - out["p_kalshi"]
    out["exponential_shock_weight_minus_q"] = out["shock_weight_exponential"] - out["q_shock"]
    return out.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True)


def build_overlap(model_exp: pd.DataFrame, model_k: pd.DataFrame) -> pd.DataFrame:
    model_exp_cols = [
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
        "p_normal",
        "p_shock",
        "lambda_value",
        "shock_weight_linear",
        "shock_weight_exponential",
        "p_linear_hybrid_recomputed",
        "p_exponential_hybrid",
        "p_exponential_minus_linear",
        "exponential_shock_weight_minus_q",
    ]
    optional_cols = ["binance_audit_price", "binance_reference_price", "join_key_used"]
    model_exp_cols.extend([col for col in optional_cols if col in model_exp.columns])

    model_k_cols = ["event_contract_id", "forecast_datetime_utc", "p_kalshi", "source_file"]
    overlap = model_exp[model_exp_cols].merge(
        model_k[model_k_cols],
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_exp_source", "_model_k"),
        validate="one_to_one",
    )
    overlap = overlap.rename(
        columns={
            "p_kalshi_exp_source": "p_linear_model_n",
            "p_kalshi_model_k": "p_model_k",
            "source_file_exp_source": "model_n_source_file",
            "source_file_model_k": "model_k_source_file",
        }
    )
    overlap["p_model_k"] = pd.to_numeric(overlap["p_model_k"], errors="coerce").clip(0.0, 1.0)
    return overlap.dropna(subset=["p_model_k", "p_reality"]).sort_values(
        ["forecast_datetime_utc", "event_ticker", "contract_label"]
    ).reset_index(drop=True)


def strategy_specs(lambda_value: float) -> List[Tuple[str, str, str]]:
    return [
        ("exp_hybrid", f"Exponential Hybrid lambda={lambda_value:g}", "p_exponential_hybrid"),
        ("linear_hybrid", "Linear Hybrid Model N", "p_linear_model_n"),
        ("model_a", "Model A Normal", "p_normal"),
        ("model_b", "Model B Shock", "p_shock"),
    ]


def build_strategy_trades(
    overlap: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    edge_threshold: float,
) -> pd.DataFrame:
    if probability_col not in overlap.columns:
        raise ValueError(f"Cannot build {strategy_label}; missing {probability_col}.")

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
    return trades.sort_values(["strategy_id", "forecast_datetime_utc", "event_ticker", "contract_label", "trade_side"])


def build_all_strategy_trades(overlap: pd.DataFrame, *, lambda_value: float, edge_threshold: float) -> pd.DataFrame:
    frames = [
        build_strategy_trades(
            overlap,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            edge_threshold=edge_threshold,
        )
        for strategy_id, strategy_label, probability_col in strategy_specs(lambda_value)
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for (strategy_id, strategy_label), part in trades.groupby(["strategy_id", "strategy_label"], sort=False):
        out = (
            part.groupby("forecast_datetime_utc", as_index=False)
            .agg(
                period_realized_pnl=("realized_pnl_per_contract", "sum"),
                period_expected_value=("expected_value_per_contract", "sum"),
                period_cost=("cost_per_contract", "sum"),
                period_trades=("event_contract_id", "size"),
                period_wins=("win", "sum"),
                mean_q_shock=("q_shock", "mean"),
                mean_exponential_shock_weight=("shock_weight_exponential", "mean"),
            )
            .sort_values("forecast_datetime_utc")
            .reset_index(drop=True)
        )
        out.insert(0, "strategy_id", strategy_id)
        out.insert(1, "strategy_label", strategy_label)
        out["cumulative_realized_pnl"] = out["period_realized_pnl"].cumsum()
        out["cumulative_expected_value"] = out["period_expected_value"].cumsum()
        out["cumulative_cost"] = out["period_cost"].cumsum()
        out["cumulative_trades"] = out["period_trades"].cumsum()
        out["cumulative_wins"] = out["period_wins"].cumsum()
        out["cumulative_winrate"] = out["cumulative_wins"] / out["cumulative_trades"]
        out["cumulative_roi_on_cost"] = out["cumulative_realized_pnl"] / out["cumulative_cost"].replace(0, np.nan)
        out["running_pnl_high_water_mark"] = out["cumulative_realized_pnl"].cummax()
        out["pnl_drawdown"] = out["running_pnl_high_water_mark"] - out["cumulative_realized_pnl"]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def summarize_strategy(part: pd.DataFrame, timeseries: pd.DataFrame, *, overlap_rows: int, overlap_timestamps: int) -> Dict[str, Any]:
    signal_rows = int(len(part))
    signal_timestamps = int(part["forecast_datetime_utc"].nunique()) if signal_rows else 0
    total_cost = float(part["cost_per_contract"].sum()) if signal_rows else 0.0
    total_pnl = float(part["realized_pnl_per_contract"].sum()) if signal_rows else 0.0
    total_ev = float(part["expected_value_per_contract"].sum()) if signal_rows else 0.0
    strategy_ts = timeseries[timeseries["strategy_id"] == part["strategy_id"].iloc[0]] if signal_rows else pd.DataFrame()
    return {
        "strategy_id": str(part["strategy_id"].iloc[0]) if signal_rows else "",
        "strategy_label": str(part["strategy_label"].iloc[0]) if signal_rows else "",
        "probability_column": str(part["strategy_probability_column"].iloc[0]) if signal_rows else "",
        "overlap_rows": int(overlap_rows),
        "overlap_timestamps": int(overlap_timestamps),
        "signal_rows": signal_rows,
        "signal_timestamps": signal_timestamps,
        "row_trade_rate": signal_rows / overlap_rows if overlap_rows else np.nan,
        "timestamp_trade_rate": signal_timestamps / overlap_timestamps if overlap_timestamps else np.nan,
        "winrate": float(part["win"].mean()) if signal_rows else np.nan,
        "average_q_shock": float(part["q_shock"].mean()) if signal_rows else np.nan,
        "average_exponential_shock_weight": float(part["shock_weight_exponential"].mean()) if signal_rows else np.nan,
        "average_selected_side_probability": float(part["model_probability_selected_side"].mean()) if signal_rows else np.nan,
        "average_model_k_selected_side_price": float(part["model_k_price_selected_side"].mean()) if signal_rows else np.nan,
        "average_edge": float(part["favorable_edge"].mean()) if signal_rows else np.nan,
        "total_expected_value": total_ev,
        "average_realized_pnl": float(part["realized_pnl_per_contract"].mean()) if signal_rows else np.nan,
        "total_realized_pnl": total_pnl,
        "total_cost": total_cost,
        "roi_on_cost": total_pnl / total_cost if total_cost else np.nan,
        "best_period_pnl": float(strategy_ts["period_realized_pnl"].max()) if not strategy_ts.empty else np.nan,
        "worst_period_pnl": float(strategy_ts["period_realized_pnl"].min()) if not strategy_ts.empty else np.nan,
        "max_drawdown": float(strategy_ts["pnl_drawdown"].max()) if not strategy_ts.empty else np.nan,
        "final_cumulative_pnl": float(strategy_ts["cumulative_realized_pnl"].iloc[-1]) if not strategy_ts.empty else np.nan,
    }


def build_strategy_summary(overlap: pd.DataFrame, trades: pd.DataFrame, timeseries: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize_strategy(
            part,
            timeseries,
            overlap_rows=len(overlap),
            overlap_timestamps=int(overlap["forecast_datetime_utc"].nunique()),
        )
        for _, part in trades.groupby("strategy_id", sort=False)
    ]
    return pd.DataFrame(rows)


def build_eval_summary(overlap: pd.DataFrame, *, lambda_value: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = pd.to_numeric(overlap["p_reality"], errors="coerce")
    for strategy_id, strategy_label, probability_col in strategy_specs(lambda_value):
        p = pd.to_numeric(overlap[probability_col], errors="coerce").clip(0.0, 1.0)
        valid = ~(p.isna() | y.isna())
        p_valid = p[valid]
        y_valid = y[valid]
        p_log = p_valid.clip(EPSILON, 1.0 - EPSILON)
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                "probability_column": probability_col,
                "rows": int(valid.sum()),
                "brier": float(((p_valid - y_valid) ** 2).mean()) if len(p_valid) else np.nan,
                "log_loss": float((-(y_valid * np.log(p_log) + (1.0 - y_valid) * np.log(1.0 - p_log))).mean())
                if len(p_valid)
                else np.nan,
                "accuracy_at_0_5": float(((p_valid >= 0.5).astype(int) == y_valid.astype(int)).mean())
                if len(p_valid)
                else np.nan,
                "mean_probability": float(p_valid.mean()) if len(p_valid) else np.nan,
                "average_absolute_error": float((p_valid - y_valid).abs().mean()) if len(p_valid) else np.nan,
                "lambda_value": float(lambda_value),
            }
        )
    return pd.DataFrame(rows)


def trade_columns(trades: pd.DataFrame) -> List[str]:
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
        "lambda_value",
        "q_shock",
        "shock_weight_exponential",
        "p_normal",
        "p_shock",
        "p_linear_model_n",
        "p_exponential_hybrid",
        "p_model_k",
        "trade_side",
        "p_strategy",
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
        elif any(token in lower for token in ["roi", "winrate", "rate", "accuracy"]):
            out[column] = out[column].map(lambda v: format_number(v, percent=True))
        elif any(token in lower for token in ["pnl", "value", "cost", "drawdown", "edge", "price", "probability", "brier", "loss", "error", "weight"]):
            out[column] = out[column].map(lambda v: format_number(v, digits=4, signed="pnl" in lower))
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
            y=1.15,
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


def build_figures(
    *,
    timeseries: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    eval_summary: pd.DataFrame,
    lambda_value: float,
) -> List[Any]:
    if not HAS_PLOTLY:
        return []

    figures: List[Any] = []
    if not timeseries.empty:
        fig_pnl = go.Figure()
        colors = {
            "exp_hybrid": "#ff8f70",
            "linear_hybrid": "#6cb6ff",
            "model_a": "#35c7b7",
            "model_b": "#fbbf24",
        }
        for strategy_id, part in timeseries.groupby("strategy_id", sort=False):
            part = part.sort_values("forecast_datetime_utc")
            label = str(part["strategy_label"].iloc[0])
            fig_pnl.add_trace(
                go.Scatter(
                    x=part["forecast_datetime_utc"],
                    y=part["cumulative_realized_pnl"],
                    mode="lines",
                    name=label,
                    line=dict(color=colors.get(strategy_id, None), width=3),
                )
            )
        fig_pnl.update_layout(title="Cumulative Realized PnL")
        fig_pnl.update_xaxes(title_text="Forecast timestamp")
        fig_pnl.update_yaxes(title_text="Cumulative PnL")
        plotly_theme(fig_pnl, height=560, top_margin=92)
        figures.append(fig_pnl)

    if not strategy_summary.empty:
        fig_bars = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Total PnL", "ROI On Cost", "Trade Rows", "Max Drawdown"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )
        labels = strategy_summary["strategy_label"]
        fig_bars.add_trace(go.Bar(x=labels, y=strategy_summary["total_realized_pnl"], marker_color="#ff8f70"), row=1, col=1)
        fig_bars.add_trace(go.Bar(x=labels, y=strategy_summary["roi_on_cost"], marker_color="#35c7b7"), row=1, col=2)
        fig_bars.add_trace(go.Bar(x=labels, y=strategy_summary["signal_rows"], marker_color="#6cb6ff"), row=2, col=1)
        fig_bars.add_trace(go.Bar(x=labels, y=strategy_summary["max_drawdown"], marker_color="#fbbf24"), row=2, col=2)
        fig_bars.update_yaxes(title_text="PnL", row=1, col=1)
        fig_bars.update_yaxes(title_text="ROI", row=1, col=2)
        fig_bars.update_yaxes(title_text="Rows", row=2, col=1)
        fig_bars.update_yaxes(title_text="Drawdown", row=2, col=2)
        plotly_theme(fig_bars, height=760, top_margin=96)
        figures.append(fig_bars)

    q_grid = pd.Series(np.linspace(0.0, 1.0, 201))
    fig_weight = go.Figure()
    fig_weight.add_trace(go.Scatter(x=q_grid, y=q_grid, mode="lines", name="Linear q", line=dict(width=3)))
    fig_weight.add_trace(
        go.Scatter(
            x=q_grid,
            y=exponential_weight(q_grid, lambda_value),
            mode="lines",
            name=f"Exponential w(q), lambda={lambda_value:g}",
            line=dict(width=3),
        )
    )
    fig_weight.update_layout(title="Shock Weight Curve")
    fig_weight.update_xaxes(title_text="q_shock")
    fig_weight.update_yaxes(title_text="Shock weight")
    plotly_theme(fig_weight, height=500, top_margin=92)
    figures.append(fig_weight)

    if not eval_summary.empty:
        fig_eval = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Brier Score", "Log Loss"),
            horizontal_spacing=0.12,
        )
        fig_eval.add_trace(go.Bar(x=eval_summary["strategy_label"], y=eval_summary["brier"], marker_color="#6cb6ff"), row=1, col=1)
        fig_eval.add_trace(go.Bar(x=eval_summary["strategy_label"], y=eval_summary["log_loss"], marker_color="#ff8f70"), row=1, col=2)
        fig_eval.update_yaxes(title_text="Brier", row=1, col=1)
        fig_eval.update_yaxes(title_text="Log loss", row=1, col=2)
        plotly_theme(fig_eval, height=500, top_margin=96)
        figures.append(fig_eval)

    return figures


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


def build_dashboard_html(
    *,
    strategy_summary: pd.DataFrame,
    eval_summary: pd.DataFrame,
    timeseries: pd.DataFrame,
    lambda_value: float,
    edge_threshold: float,
) -> str:
    exp = strategy_summary.loc[strategy_summary["strategy_id"] == "exp_hybrid"].iloc[0]
    linear = strategy_summary.loc[strategy_summary["strategy_id"] == "linear_hybrid"].iloc[0]
    pnl_delta = float(exp["total_realized_pnl"] - linear["total_realized_pnl"])
    cards = [
        ("Exp Hybrid PnL", format_number(exp["total_realized_pnl"], digits=3, signed=True)),
        ("Vs Linear PnL", format_number(pnl_delta, digits=3, signed=True)),
        ("Exp ROI", format_number(exp["roi_on_cost"], percent=True, signed=True)),
        ("Trade Rows", format_number(exp["signal_rows"], digits=0)),
        ("Max Drawdown", format_number(exp["max_drawdown"], digits=3)),
        ("Avg Exp Weight", format_number(exp["average_exponential_shock_weight"], digits=4)),
    ]
    card_html = "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )

    figures_html = figures_to_html(
        build_figures(
            timeseries=timeseries,
            strategy_summary=strategy_summary,
            eval_summary=eval_summary,
            lambda_value=lambda_value,
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Exponential Hybrid Model N Experiment</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Standalone Exponential Hybrid</p>
    <h1>Model N Exponential Hybrid, lambda={lambda_value:g}</h1>
    <p class="lead">
      This experiment is isolated under <code>Model_N_Shock_Breakout_PnL/Exponential_Hybrid</code>.
      It computes <code>w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)</code>, then uses
      <code>p_final = (1 - w(q)) * p_normal + w(q) * p_shock</code>. PnL uses the same favorable
      BUY_YES/BUY_NO rule against Model K with selected-side edge threshold
      <code>{edge_threshold}</code>.
    </p>
    <div class="cards">{card_html}</div>
  </section>

  <h2>Charts</h2>
  <section class="chart-wrap">{figures_html}</section>

  <h2>Strategy PnL Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(strategy_summary))}</div></section>

  <h2>Probability Eval Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(eval_summary))}</div></section>

  <h2>Output Files</h2>
  <section class="panel">
    <p>
      Outputs: <code>exponential_hybrid_raw_values.csv</code>,
      <code>exponential_hybrid_eval_summary.csv</code>,
      <code>exponential_hybrid_strategy_summary.csv</code>,
      <code>exponential_hybrid_strategy_pnl_timeseries.csv</code>,
      <code>exponential_hybrid_trade_minutes.csv</code>, and
      <code>exponential_hybrid_dashboard.html</code>.
    </p>
  </section>
</main>
</body>
</html>
"""


def write_readme(output_dir: Path, *, lambda_value: float, edge_threshold: float) -> None:
    text = f"""Exponential Hybrid Model N Experiment

This folder is standalone and can be removed without changing the main Model N scripts or outputs.

Formula:
- w(q) = (exp(lambda * q) - 1) / (exp(lambda) - 1)
- lambda = {lambda_value}
- p_final = (1 - w(q)) * p_normal + w(q) * p_shock

PnL rule:
- Compare the strategy probability against Model K as the entry price
- BUY_YES when p_strategy - p_model_k > {edge_threshold}
- BUY_NO when (1 - p_strategy) - (1 - p_model_k) > {edge_threshold}

Outputs:
- exponential_hybrid_raw_values.csv
- exponential_hybrid_eval_summary.csv
- exponential_hybrid_strategy_summary.csv
- exponential_hybrid_strategy_pnl_timeseries.csv
- exponential_hybrid_trade_minutes.csv
- exponential_hybrid_dashboard.html
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.lambda_value < 0.0:
        raise ValueError("--lambda-value must be non-negative.")
    if args.edge_threshold < 0.0:
        raise ValueError("--edge-threshold must be non-negative.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    model_exp = add_exponential_hybrid(model_n, lambda_value=args.lambda_value)
    overlap = build_overlap(model_exp, model_k)
    eval_summary = build_eval_summary(overlap, lambda_value=args.lambda_value)
    trades = build_all_strategy_trades(overlap, lambda_value=args.lambda_value, edge_threshold=args.edge_threshold)
    timeseries = build_pnl_timeseries(trades)
    strategy_summary = build_strategy_summary(overlap, trades, timeseries)

    model_exp.to_csv(output_dir / "exponential_hybrid_raw_values.csv", index=False)
    eval_summary.to_csv(output_dir / "exponential_hybrid_eval_summary.csv", index=False)
    strategy_summary.to_csv(output_dir / "exponential_hybrid_strategy_summary.csv", index=False)
    timeseries.to_csv(output_dir / "exponential_hybrid_strategy_pnl_timeseries.csv", index=False)
    trades[trade_columns(trades)].to_csv(output_dir / "exponential_hybrid_trade_minutes.csv", index=False)

    dashboard_html = build_dashboard_html(
        strategy_summary=strategy_summary,
        eval_summary=eval_summary,
        timeseries=timeseries,
        lambda_value=args.lambda_value,
        edge_threshold=args.edge_threshold,
    )
    (output_dir / "exponential_hybrid_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    write_readme(output_dir, lambda_value=args.lambda_value, edge_threshold=args.edge_threshold)

    print(f"Lambda: {args.lambda_value:g}")
    print(f"Overlap rows: {len(overlap):,}")
    print(
        strategy_summary[
            [
                "strategy_label",
                "signal_rows",
                "signal_timestamps",
                "total_realized_pnl",
                "roi_on_cost",
                "max_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"Dashboard: {output_dir / 'exponential_hybrid_dashboard.html'}")
    print(f"Strategy summary: {output_dir / 'exponential_hybrid_strategy_summary.csv'}")


if __name__ == "__main__":
    main()
