from __future__ import annotations

import argparse
import html
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
OUTPUT_FOLDER_NAME = "Model_N_Trade_Diagnostics_Outputs"
DEFAULT_TAKER_FEE_RATE = 0.07
DEFAULT_PRIMARY_STRATEGY_ID = "model_n_exp_lambda_2"

STRATEGIES: Tuple[Tuple[str, str, str], ...] = (
    ("model_n", "Model N Hybrid", "p_model_n"),
    ("model_n_exp_lambda_2", "Model N Exp Hybrid lambda=2", "p_exponential_hybrid"),
    ("model_a", "Model A Normal", "p_normal"),
    ("model_b", "Model B Shock", "p_shock"),
)

STRATEGY_COLORS = {
    "Model N Hybrid": "#6cb6ff",
    "Model N Exp Hybrid lambda=2": "#fbbf24",
    "Model A Normal": "#35c7b7",
    "Model B Shock": "#ff8f70",
}

PREDICTED_NET_EDGE_BUCKET_ORDER = ["<=0%", "0-2%", "2-4%", "4-6%", "6-8%", "8%+"]
GROSS_EDGE_BUCKET_ORDER = ["0-2%", "2-4%", "4-8%", "8%+"]
TIME_BUCKET_ORDER = ["0-5", "5-10", "10-20", "20-40", "40-60", "60+"]
Q_SHOCK_BUCKET_ORDER = ["<5%", "5-10%", "10-15%", "15-20%", "20%+"]
DISAGREEMENT_BUCKET_ORDER = ["0-2%", "2-5%", "5-10%", "10-20%", "20%+"]

FEATURE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("predicted_net_edge_bucket", "Predicted Net Edge Bucket"),
    ("gross_edge_bucket", "Gross Edge Bucket"),
    ("q_shock_bucket", "q_shock Bucket"),
    ("time_to_expiry_bucket", "Time To Expiry Bucket"),
    ("model_disagreement_bucket", "Model Disagreement Bucket"),
    ("volatility_regime", "Volatility Regime"),
    ("selected_side", "Side"),
    ("contract_label", "Contract Label"),
    ("market_quality_bucket", "Market Quality"),
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build Model N fee-aware trade diagnostics: requested trade ledger fields, "
            "edge buckets, segment summaries, feature-combination matrices, and dashboard."
        )
    )
    parser.add_argument(
        "--model-n-raw-values",
        type=Path,
        default=root / "Model_N" / "model_N_Evals_Outputs" / "raw_values.csv",
        help="Model N eval raw_values.csv produced by Model_N_Eval.py.",
    )
    parser.add_argument(
        "--model-k-raw-values",
        type=Path,
        default=root / "Model_K" / "Model_K_outputs" / "raw_values.csv",
        help="Model K eval raw_values.csv used as the market/entry probability source.",
    )
    parser.add_argument(
        "--volatility-rows",
        type=Path,
        default=root
        / "Model_K_Volatility_Decomposition_RT"
        / "Model_K_Volatility_Decomposition_RT_outputs"
        / "all_scored_rows_with_volatility.csv",
        help="Optional scored-row volatility file with primary_volatility_band.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Directory where trade diagnostics outputs will be written.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum favorable-direction gross edge required for a trade signal.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=DEFAULT_TAKER_FEE_RATE,
        help="Taker fee rate applied to selected market price * (1 - selected market price).",
    )
    parser.add_argument(
        "--no-fee-round-up",
        action="store_true",
        help="Disable cent round-up for the fee model.",
    )
    parser.add_argument(
        "--primary-strategy-id",
        default=DEFAULT_PRIMARY_STRATEGY_ID,
        choices=[strategy_id for strategy_id, _label, _col in STRATEGIES],
        help="Strategy highlighted in the bucket/matrix dashboard tables.",
    )
    parser.add_argument(
        "--min-combo-trades",
        type=int,
        default=100,
        help="Minimum trades required when ranking feature combinations.",
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
    if "event_datetime_utc" in frame.columns:
        frame["event_datetime_utc"] = pd.to_datetime(frame["event_datetime_utc"], utc=True)
    numeric_cols = [
        "p_kalshi",
        "p_reality",
        "minute_number",
        "strike",
        "minutes_to_settlement",
        "q_shock",
        "shock_weight_exponential",
        "p_normal",
        "p_shock",
        "p_exponential_hybrid",
        "yes_ask",
        "yes_bid",
        "no_ask",
        "no_bid",
        "yes_mid",
        "no_mid",
        "volume",
        "open_interest",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=JOIN_KEYS + ["p_kalshi", "p_reality"])
    frame["p_kalshi"] = frame["p_kalshi"].clip(0.0, 1.0)
    frame["p_reality"] = frame["p_reality"].clip(0.0, 1.0)
    return frame


def load_volatility_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    required = set(JOIN_KEYS)
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    keep = JOIN_KEYS + [
        col
        for col in [
            "primary_volatility_band",
            "realized_volatility",
            "rolling_percentile_rank",
            "realized_variance",
            "hour_open",
            "hour_close",
            "hour_high",
            "hour_low",
        ]
        if col in frame.columns
    ]
    frame = frame[keep].copy()
    frame["forecast_datetime_utc"] = pd.to_datetime(frame["forecast_datetime_utc"], utc=True)
    for column in frame.columns:
        if column not in JOIN_KEYS and column != "primary_volatility_band":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(subset=JOIN_KEYS).reset_index(drop=True)


def build_overlap(model_n: pd.DataFrame, model_k: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    required_model_n = {"p_normal", "p_shock", "p_exponential_hybrid"}
    missing = required_model_n - set(model_n.columns)
    if missing:
        raise ValueError(f"Model N raw values are missing required strategy columns: {sorted(missing)}")

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
        "shock_weight_exponential",
        "p_normal",
        "p_shock",
        "p_exponential_hybrid",
    ]
    optional_model_n_cols = ["binance_audit_price", "binance_reference_price", "join_key_used"]
    model_n_cols.extend([col for col in optional_model_n_cols if col in model_n.columns])

    model_k_cols = ["event_contract_id", "forecast_datetime_utc", "p_kalshi", "source_file"]
    optional_market_cols = [
        "yes_ask",
        "yes_bid",
        "no_ask",
        "no_bid",
        "yes_mid",
        "no_mid",
        "volume",
        "open_interest",
    ]
    model_k_cols.extend([col for col in optional_market_cols if col in model_k.columns])

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
    overlap["p_model_k"] = pd.to_numeric(overlap["p_model_k"], errors="coerce").clip(0.0, 1.0)
    overlap = overlap.dropna(subset=["p_model_k", "p_reality"])

    if not volatility.empty:
        overlap = overlap.merge(volatility, on=JOIN_KEYS, how="left", validate="one_to_one")

    return overlap.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True)


def series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def fee_per_contract(price: pd.Series, *, fee_rate: float, round_up_to_cent: bool) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").clip(0.0, 1.0)
    fee = fee_rate * price * (1.0 - price)
    if round_up_to_cent:
        fee = np.ceil((fee * 100.0) - 1e-12) / 100.0
    return pd.Series(fee, index=price.index).clip(lower=0.0)


def map_volatility_regime(raw: pd.Series) -> pd.Series:
    text = raw.fillna("unknown").astype(str).str.lower()
    out = pd.Series("unknown", index=raw.index, dtype=object)
    out[text.str.contains("low", na=False)] = "low"
    out[text.str.contains("standard", na=False)] = "medium"
    out[text.str.contains("high", na=False)] = "high"
    return out


def attach_side_fields(
    frame: pd.DataFrame,
    *,
    side: str,
    selected_side: str,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    market_price: pd.Series,
    market_price_source: pd.Series,
    executable_available: pd.Series,
    spread: pd.Series,
    model_probability: pd.Series,
    p_normal_selected: pd.Series,
    p_shock_selected: pd.Series,
    realized_outcome: pd.Series,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = out["forecast_datetime_utc"]
    out["market"] = out["market_ticker"]
    out["expiry"] = out["event_datetime_utc"]
    out["side"] = side
    out["selected_side"] = selected_side
    out["strategy_id"] = strategy_id
    out["strategy_label"] = strategy_label
    out["strategy_probability_column"] = probability_col
    out["model_probability"] = model_probability
    out["market_price"] = market_price
    out["market_price_source"] = market_price_source
    out["market_price_is_executable"] = executable_available.astype(bool)
    out["fee"] = fee_per_contract(out["market_price"], fee_rate=fee_rate, round_up_to_cent=round_up_to_cent)
    out["gross_edge"] = out["model_probability"] - out["market_price"]
    out["net_expected_edge"] = out["gross_edge"] - out["fee"]
    out["realized_outcome"] = pd.to_numeric(realized_outcome, errors="coerce").clip(0.0, 1.0)
    out["yes_realized_outcome"] = out["p_reality"]
    out["gross_pnl"] = out["realized_outcome"] - out["market_price"]
    out["net_pnl"] = out["gross_pnl"] - out["fee"]
    out["time_to_expiry"] = out["minutes_to_settlement"]
    out["volatility_regime_raw"] = out.get("primary_volatility_band", pd.Series("unknown", index=out.index))
    out["volatility_regime"] = map_volatility_regime(out["volatility_regime_raw"])
    out["p_normal_selected_side"] = p_normal_selected
    out["p_shock_selected_side"] = p_shock_selected
    out["model_disagreement"] = out["p_shock_selected_side"] - out["p_normal_selected_side"]
    out["model_disagreement_abs"] = out["model_disagreement"].abs()
    out["liquidity"] = np.nan
    out["liquidity_source"] = "unavailable_from_model_k_price_file"
    if "volume" in out.columns:
        out["liquidity"] = pd.to_numeric(out["volume"], errors="coerce")
        out["liquidity_source"] = "source_volume"
    out["spread"] = pd.to_numeric(spread, errors="coerce")
    out["fee_rate"] = float(fee_rate)
    out["fee_round_up_to_cent"] = bool(round_up_to_cent)
    out["win"] = (out["realized_outcome"] == 1.0).astype(int)
    out["result_label"] = np.where(out["win"] == 1, "win", "loss")
    out["cost_after_fees"] = out["market_price"] + out["fee"]
    out["price_quality_note"] = np.where(
        out["market_price_is_executable"],
        "selected_side_ask_available",
        "using_model_k_price_proxy_no_bid_ask_in_output",
    )
    return out


def build_strategy_trades(
    overlap: pd.DataFrame,
    *,
    strategy_id: str,
    strategy_label: str,
    probability_col: str,
    edge_threshold: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    if probability_col not in overlap.columns:
        raise ValueError(f"Cannot build {strategy_label}; missing {probability_col}.")

    work = overlap.dropna(subset=[probability_col, "p_model_k", "p_reality"]).copy()
    work["p_strategy"] = pd.to_numeric(work[probability_col], errors="coerce").clip(0.0, 1.0)

    yes_ask = series_or_nan(work, "yes_ask")
    yes_bid = series_or_nan(work, "yes_bid")
    no_ask = series_or_nan(work, "no_ask")
    no_bid = series_or_nan(work, "no_bid")

    yes_market_price = yes_ask.combine_first(work["p_model_k"])
    no_market_price = no_ask.combine_first(1.0 - work["p_model_k"])
    yes_price_source = pd.Series(
        np.where(yes_ask.notna(), "yes_ask", "model_k_yes_price_proxy"),
        index=work.index,
        dtype=object,
    )
    no_price_source = pd.Series(
        np.where(no_ask.notna(), "no_ask", "one_minus_model_k_yes_price_proxy"),
        index=work.index,
        dtype=object,
    )
    yes_executable_available = yes_ask.notna()
    no_executable_available = no_ask.notna()
    yes_spread = yes_ask - yes_bid
    no_spread = no_ask - no_bid

    work["yes_gross_edge_at_entry"] = work["p_strategy"] - yes_market_price
    work["no_gross_edge_at_entry"] = (1.0 - work["p_strategy"]) - no_market_price

    buy_yes = work[work["yes_gross_edge_at_entry"] > edge_threshold].copy()
    yes_trades = attach_side_fields(
        buy_yes,
        side="BUY_YES",
        selected_side="YES",
        strategy_id=strategy_id,
        strategy_label=strategy_label,
        probability_col=probability_col,
        market_price=yes_market_price.loc[buy_yes.index],
        market_price_source=yes_price_source.loc[buy_yes.index],
        executable_available=yes_executable_available.loc[buy_yes.index],
        spread=yes_spread.loc[buy_yes.index],
        model_probability=buy_yes["p_strategy"],
        p_normal_selected=buy_yes["p_normal"],
        p_shock_selected=buy_yes["p_shock"],
        realized_outcome=buy_yes["p_reality"],
        fee_rate=fee_rate,
        round_up_to_cent=round_up_to_cent,
    )

    buy_no = work[work["no_gross_edge_at_entry"] > edge_threshold].copy()
    no_trades = attach_side_fields(
        buy_no,
        side="BUY_NO",
        selected_side="NO",
        strategy_id=strategy_id,
        strategy_label=strategy_label,
        probability_col=probability_col,
        market_price=no_market_price.loc[buy_no.index],
        market_price_source=no_price_source.loc[buy_no.index],
        executable_available=no_executable_available.loc[buy_no.index],
        spread=no_spread.loc[buy_no.index],
        model_probability=1.0 - buy_no["p_strategy"],
        p_normal_selected=1.0 - buy_no["p_normal"],
        p_shock_selected=1.0 - buy_no["p_shock"],
        realized_outcome=1.0 - buy_no["p_reality"],
        fee_rate=fee_rate,
        round_up_to_cent=round_up_to_cent,
    )

    trades = pd.concat([yes_trades, no_trades], ignore_index=True)
    if trades.empty:
        return trades
    return trades.sort_values(["strategy_id", "timestamp", "event_ticker", "contract_label", "side"]).reset_index(drop=True)


def add_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["predicted_net_edge_bucket"] = pd.cut(
        out["net_expected_edge"],
        bins=[-np.inf, 0.0, 0.02, 0.04, 0.06, 0.08, np.inf],
        labels=PREDICTED_NET_EDGE_BUCKET_ORDER,
        include_lowest=True,
        right=True,
    ).astype(object)
    out["gross_edge_bucket"] = pd.cut(
        out["gross_edge"],
        bins=[0.0, 0.02, 0.04, 0.08, np.inf],
        labels=GROSS_EDGE_BUCKET_ORDER,
        include_lowest=True,
        right=True,
    ).astype(object)
    out["time_to_expiry_bucket"] = pd.cut(
        out["time_to_expiry"],
        bins=[0.0, 5.0, 10.0, 20.0, 40.0, 60.0, np.inf],
        labels=TIME_BUCKET_ORDER,
        include_lowest=True,
        right=True,
    ).astype(object)
    out["q_shock_bucket"] = pd.cut(
        out["q_shock"],
        bins=[-np.inf, 0.05, 0.10, 0.15, 0.20, np.inf],
        labels=Q_SHOCK_BUCKET_ORDER,
        include_lowest=True,
        right=True,
    ).astype(object)
    out["model_disagreement_bucket"] = pd.cut(
        out["model_disagreement_abs"],
        bins=[-np.inf, 0.02, 0.05, 0.10, 0.20, np.inf],
        labels=DISAGREEMENT_BUCKET_ORDER,
        include_lowest=True,
        right=True,
    ).astype(object)
    out["market_quality_bucket"] = "spread_unavailable"
    has_spread = out["spread"].notna()
    if has_spread.any():
        out.loc[has_spread, "market_quality_bucket"] = pd.cut(
            out.loc[has_spread, "spread"],
            bins=[-np.inf, 0.02, 0.05, 0.10, np.inf],
            labels=["spread <=2%", "spread 2-5%", "spread 5-10%", "spread >10%"],
            include_lowest=True,
            right=True,
        ).astype(object)
    return out


def build_all_trades(
    overlap: pd.DataFrame,
    *,
    edge_threshold: float,
    fee_rate: float,
    round_up_to_cent: bool,
) -> pd.DataFrame:
    frames = [
        build_strategy_trades(
            overlap,
            strategy_id=strategy_id,
            strategy_label=strategy_label,
            probability_col=probability_col,
            edge_threshold=edge_threshold,
            fee_rate=fee_rate,
            round_up_to_cent=round_up_to_cent,
        )
        for strategy_id, strategy_label, probability_col in STRATEGIES
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return add_buckets(pd.concat(frames, ignore_index=True))


def summarize_group(frame: pd.DataFrame) -> Dict[str, Any]:
    count = int(len(frame))
    total_cost = float(frame["market_price"].sum()) if count else 0.0
    total_cost_after_fees = float(frame["cost_after_fees"].sum()) if count else 0.0
    total_net_pnl = float(frame["net_pnl"].sum()) if count else 0.0
    return {
        "trade_count": count,
        "winrate": float(frame["win"].mean()) if count else np.nan,
        "total_gross_pnl": float(frame["gross_pnl"].sum()) if count else 0.0,
        "total_fees": float(frame["fee"].sum()) if count else 0.0,
        "total_net_pnl": total_net_pnl,
        "mean_gross_pnl": float(frame["gross_pnl"].mean()) if count else np.nan,
        "mean_net_pnl": float(frame["net_pnl"].mean()) if count else np.nan,
        "realized_net_edge": float(frame["net_pnl"].mean()) if count else np.nan,
        "mean_gross_edge": float(frame["gross_edge"].mean()) if count else np.nan,
        "mean_net_expected_edge": float(frame["net_expected_edge"].mean()) if count else np.nan,
        "mean_market_price": float(frame["market_price"].mean()) if count else np.nan,
        "mean_fee": float(frame["fee"].mean()) if count else np.nan,
        "mean_q_shock": float(frame["q_shock"].mean()) if count else np.nan,
        "mean_model_disagreement_abs": float(frame["model_disagreement_abs"].mean()) if count else np.nan,
        "total_cost": total_cost,
        "total_cost_after_fees": total_cost_after_fees,
        "roi_after_fees": total_net_pnl / total_cost_after_fees if total_cost_after_fees else np.nan,
    }


def build_strategy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = trades[trades["strategy_id"] == strategy_id]
        rows.append({"strategy_id": strategy_id, "strategy_label": strategy_label} | summarize_group(part))
    return pd.DataFrame(rows)


def build_pnl_timeseries(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = (
        trades.groupby(["strategy_id", "strategy_label", "timestamp"], as_index=False)
        .agg(
            period_net_pnl=("net_pnl", "sum"),
            period_gross_pnl=("gross_pnl", "sum"),
            period_fees=("fee", "sum"),
            period_trades=("event_contract_id", "size"),
            period_wins=("win", "sum"),
            period_net_expected_edge=("net_expected_edge", "sum"),
        )
        .sort_values(["strategy_id", "timestamp"])
        .reset_index(drop=True)
    )
    grouped["cumulative_net_pnl"] = grouped.groupby("strategy_id")["period_net_pnl"].cumsum()
    grouped["cumulative_gross_pnl"] = grouped.groupby("strategy_id")["period_gross_pnl"].cumsum()
    grouped["cumulative_fees"] = grouped.groupby("strategy_id")["period_fees"].cumsum()
    grouped["cumulative_trades"] = grouped.groupby("strategy_id")["period_trades"].cumsum()
    grouped["cumulative_wins"] = grouped.groupby("strategy_id")["period_wins"].cumsum()
    grouped["cumulative_winrate"] = grouped["cumulative_wins"] / grouped["cumulative_trades"]
    running_high = grouped.groupby("strategy_id")["cumulative_net_pnl"].cummax()
    grouped["running_pnl_high_water_mark"] = np.maximum(running_high, 0.0)
    grouped["pnl_drawdown"] = grouped["running_pnl_high_water_mark"] - grouped["cumulative_net_pnl"]
    return grouped


def build_segment_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id, strategy_label, _prob_col in STRATEGIES:
        strategy_trades = trades[trades["strategy_id"] == strategy_id].copy()
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_label": strategy_label,
                "segment_field": "overall",
                "segment_label": "Overall",
                "segment_value": "all",
            }
            | summarize_group(strategy_trades)
        )
        for field, label in FEATURE_COLUMNS:
            work = strategy_trades.copy()
            work[field] = work[field].astype(object).where(work[field].notna(), "missing")
            for value, part in work.groupby(field, dropna=False, sort=False):
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy_label": strategy_label,
                        "segment_field": field,
                        "segment_label": label,
                        "segment_value": str(value),
                    }
                    | summarize_group(part)
                )
    return pd.DataFrame(rows)


def build_predicted_net_edge_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = trades[trades["strategy_id"] == strategy_id].copy()
        part["predicted_net_edge_bucket"] = pd.Categorical(
            part["predicted_net_edge_bucket"],
            categories=PREDICTED_NET_EDGE_BUCKET_ORDER,
            ordered=True,
        )
        for bucket, bucket_part in part.groupby("predicted_net_edge_bucket", observed=False, sort=True):
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "strategy_label": strategy_label,
                    "predicted_net_edge_bucket": str(bucket),
                }
                | summarize_group(bucket_part)
            )
    return pd.DataFrame(rows)


def build_feature_pair_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    feature_pairs = list(itertools.combinations(FEATURE_COLUMNS, 2))
    for strategy_id, strategy_label, _prob_col in STRATEGIES:
        strategy_trades = trades[trades["strategy_id"] == strategy_id].copy()
        for (field_x, label_x), (field_y, label_y) in feature_pairs:
            work = strategy_trades.copy()
            work[field_x] = work[field_x].astype(object).where(work[field_x].notna(), "missing")
            work[field_y] = work[field_y].astype(object).where(work[field_y].notna(), "missing")
            for (value_x, value_y), part in work.groupby([field_x, field_y], dropna=False, sort=False):
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy_label": strategy_label,
                        "feature_x": field_x,
                        "feature_x_label": label_x,
                        "feature_x_value": str(value_x),
                        "feature_y": field_y,
                        "feature_y_label": label_y,
                        "feature_y_value": str(value_y),
                    }
                    | summarize_group(part)
                )
    return pd.DataFrame(rows)


def build_ranked_combinations(pair_matrix: pd.DataFrame, *, min_trades: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eligible = pair_matrix[pair_matrix["trade_count"] >= min_trades].copy()
    sort_cols = ["total_net_pnl", "mean_net_pnl", "trade_count"]
    top = eligible.sort_values(sort_cols, ascending=[False, False, False]).head(50).reset_index(drop=True)
    bottom = eligible.sort_values(sort_cols, ascending=[True, True, False]).head(50).reset_index(drop=True)
    return top, bottom


def build_field_matrix() -> pd.DataFrame:
    rows = [
        ("timestamp", "timestamp", "forecast_datetime_utc", True, "Minute when the trade signal happened."),
        ("market / strike / expiry", "market, strike, expiry", "Model N eval raw values", True, "Contract ticker, strike, and event expiry."),
        ("side", "side, selected_side", "Derived trade rule", True, "BUY_YES or BUY_NO plus selected YES/NO side."),
        ("model probability", "model_probability", "Selected strategy probability", True, "Selected-side fair probability."),
        (
            "market price",
            "market_price",
            "Model K price file",
            True,
            "Uses selected-side ask if present; current stored files only expose the Model K price proxy.",
        ),
        ("fee", "fee", "Fee formula", True, "Taker fee on selected-side price, rounded up to cents by default."),
        ("gross edge", "gross_edge", "Derived", True, "model_probability - market_price."),
        ("net expected edge", "net_expected_edge", "Derived", True, "model_probability - market_price - fee."),
        ("realized outcome", "realized_outcome", "Official Kalshi settlement", True, "Selected-side payout, 0 or 1."),
        ("gross PnL", "gross_pnl", "Derived", True, "realized_outcome - market_price."),
        ("net PnL", "net_pnl", "Derived", True, "gross_pnl - fee."),
        ("time to expiry", "time_to_expiry", "minutes_to_settlement", True, "Minutes remaining at signal time."),
        (
            "volatility regime",
            "volatility_regime",
            "Model_K volatility decomposition",
            True,
            "Low/medium/high collapsed from primary_volatility_band when available.",
        ),
        ("q_shock", "q_shock", "Model N shock classifier", True, "Predicted shock probability."),
        (
            "model disagreement",
            "model_disagreement, model_disagreement_abs",
            "p_shock vs p_normal",
            True,
            "Selected-side shock probability minus selected-side normal probability.",
        ),
        (
            "liquidity/spread",
            "liquidity, spread, market_quality_bucket",
            "Kalshi bid/ask fields if present",
            False,
            "Current historical Model_K/Model_N files do not store bid/ask/spread/liquidity; columns are kept as unavailable.",
        ),
    ]
    return pd.DataFrame(rows, columns=["requested_field", "output_columns", "source", "available", "notes"])


def ledger_columns(trades: pd.DataFrame) -> List[str]:
    cols = [
        "strategy_id",
        "strategy_label",
        "timestamp",
        "event_ticker",
        "market",
        "contract_label",
        "strike",
        "expiry",
        "side",
        "selected_side",
        "model_probability",
        "market_price",
        "market_price_source",
        "market_price_is_executable",
        "fee",
        "gross_edge",
        "net_expected_edge",
        "predicted_net_edge_bucket",
        "gross_edge_bucket",
        "realized_outcome",
        "yes_realized_outcome",
        "gross_pnl",
        "net_pnl",
        "time_to_expiry",
        "time_to_expiry_bucket",
        "volatility_regime",
        "volatility_regime_raw",
        "q_shock",
        "q_shock_bucket",
        "p_normal_selected_side",
        "p_shock_selected_side",
        "model_disagreement",
        "model_disagreement_abs",
        "model_disagreement_bucket",
        "liquidity",
        "liquidity_source",
        "spread",
        "market_quality_bucket",
        "price_quality_note",
        "win",
        "result_label",
        "official_result",
        "p_reality",
        "p_strategy",
        "p_model_n",
        "p_exponential_hybrid",
        "p_normal",
        "p_shock",
        "p_model_k",
        "minutes_to_settlement",
        "minute_number",
        "model_n_source_file",
        "model_k_source_file",
        "event_contract_id",
        "market_ticker",
    ]
    cols.extend(
        [
            col
            for col in [
                "realized_volatility",
                "rolling_percentile_rank",
                "binance_audit_price",
                "binance_reference_price",
                "join_key_used",
            ]
            if col in trades.columns
        ]
    )
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
    out = pd.DataFrame(index=df.index)
    for column in df.columns:
        lower = column.lower()
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out[column] = series.astype(str)
        elif any(token in lower for token in ["rate", "roi", "winrate", "frequency"]):
            out[column] = series.map(lambda v: format_number(v, percent=True))
        elif any(token in lower for token in ["count", "rows", "trades", "wins"]):
            out[column] = series.map(lambda v: format_number(v, digits=0))
        elif any(token in lower for token in ["pnl", "edge", "fee", "price", "cost", "q_shock", "disagreement"]):
            out[column] = series.map(lambda v: format_number(v, digits=4, signed="pnl" in lower))
        else:
            out[column] = series.map(lambda v: format_number(v, digits=4))
    return out


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def plotly_theme(fig: Any, *, height: int, top_margin: int = 88) -> None:
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
        margin=dict(l=56, r=32, t=top_margin, b=56),
        height=height,
    )
    fig.update_xaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")
    fig.update_yaxes(gridcolor="#22384a", linecolor="#35516c", zerolinecolor="#35516c")


def cumulative_net_pnl_figure(timeseries: pd.DataFrame) -> Any:
    fig = go.Figure()
    for _strategy_id, strategy_label, _prob_col in STRATEGIES:
        part = timeseries[timeseries["strategy_label"] == strategy_label].sort_values("timestamp")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["timestamp"],
                y=part["cumulative_net_pnl"],
                mode="lines",
                name=strategy_label,
                line=dict(color=STRATEGY_COLORS.get(strategy_label), width=3),
            )
        )
    fig.update_layout(title="Cumulative Net PnL After Fees")
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Cumulative net PnL")
    plotly_theme(fig, height=540, top_margin=96)
    return fig


def predicted_edge_bar_figure(edge_summary: pd.DataFrame, *, primary_strategy_id: str) -> Any:
    part = edge_summary[edge_summary["strategy_id"] == primary_strategy_id].copy()
    part["predicted_net_edge_bucket"] = pd.Categorical(
        part["predicted_net_edge_bucket"],
        categories=PREDICTED_NET_EDGE_BUCKET_ORDER,
        ordered=True,
    )
    part = part.sort_values("predicted_net_edge_bucket")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Realized Net Edge By Predicted Net Edge Bucket", "Trade Count"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(x=part["predicted_net_edge_bucket"], y=part["realized_net_edge"], marker_color="#35c7b7"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=part["predicted_net_edge_bucket"], y=part["trade_count"], marker_color="#fbbf24"),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Mean realized net edge", row=1, col=1)
    fig.update_yaxes(title_text="Trades", row=1, col=2)
    fig.update_layout(title="Predicted Net Edge Calibration")
    plotly_theme(fig, height=520, top_margin=112)
    return fig


def top_combo_figure(top_combos: pd.DataFrame, *, primary_strategy_id: str) -> Any:
    part = top_combos[top_combos["strategy_id"] == primary_strategy_id].head(15).copy()
    if part.empty:
        fig = go.Figure()
        fig.update_layout(title="Top Feature Combinations")
        plotly_theme(fig, height=420, top_margin=92)
        return fig
    part["combo"] = (
        part["feature_x_label"]
        + "="
        + part["feature_x_value"]
        + " | "
        + part["feature_y_label"]
        + "="
        + part["feature_y_value"]
    )
    fig = go.Figure(
        go.Bar(
            x=part["total_net_pnl"],
            y=part["combo"],
            orientation="h",
            marker_color=np.where(part["total_net_pnl"] >= 0, "#35c7b7", "#ff8f70"),
        )
    )
    fig.update_layout(title="Top Feature Combinations By Total Net PnL")
    fig.update_xaxes(title_text="Total net PnL")
    fig.update_yaxes(title_text="Combination", autorange="reversed")
    plotly_theme(fig, height=660, top_margin=92)
    return fig


def heatmap_figure(
    pair_matrix: pd.DataFrame,
    *,
    primary_strategy_id: str,
    feature_x: str,
    feature_y: str,
    title: str,
    value_col: str = "mean_net_pnl",
) -> Any:
    part = pair_matrix[
        (pair_matrix["strategy_id"] == primary_strategy_id)
        & (pair_matrix["feature_x"] == feature_x)
        & (pair_matrix["feature_y"] == feature_y)
    ].copy()
    fig = go.Figure()
    if part.empty:
        fig.update_layout(title=f"{title} (no rows)")
        plotly_theme(fig, height=460, top_margin=92)
        return fig

    pivot = part.pivot_table(
        index="feature_y_value",
        columns="feature_x_value",
        values=value_col,
        aggfunc="mean",
    )
    counts = part.pivot_table(
        index="feature_y_value",
        columns="feature_x_value",
        values="trade_count",
        aggfunc="sum",
    ).reindex(index=pivot.index, columns=pivot.columns)
    hover = [
        [
            f"{feature_x}: {col}<br>{feature_y}: {idx}<br>{value_col}: {pivot.loc[idx, col]:.4f}<br>trades: {counts.loc[idx, col]:,.0f}"
            if pd.notna(pivot.loc[idx, col])
            else ""
            for col in pivot.columns
        ]
        for idx in pivot.index
    ]
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn",
            zmid=0.0,
            text=hover,
            hoverinfo="text",
            colorbar=dict(title=value_col),
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title_text=feature_x)
    fig.update_yaxes(title_text=feature_y)
    plotly_theme(fig, height=520, top_margin=96)
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
.hero {
  padding: 28px 0 18px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 24px;
}
.eyebrow {
  color: var(--accent);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0;
  font-size: 12px;
  margin: 0 0 8px;
}
h1 { font-size: clamp(30px, 4vw, 56px); line-height: 1; margin: 0 0 14px; letter-spacing: 0; }
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
@media (max-width: 760px) {
  main { width: min(100% - 24px, 1480px); padding-top: 20px; }
  .metric-card strong { font-size: 18px; }
}
</style>"""


def dashboard_cards(strategy_summary: pd.DataFrame, *, primary_strategy_id: str) -> str:
    primary = strategy_summary[strategy_summary["strategy_id"] == primary_strategy_id]
    row = primary.iloc[0] if not primary.empty else strategy_summary.iloc[0]
    best = strategy_summary.sort_values("total_net_pnl", ascending=False).iloc[0]
    cards = [
        ("Primary Strategy", str(row["strategy_label"])),
        ("Primary Net PnL", format_number(row["total_net_pnl"], digits=3, signed=True)),
        ("Primary Trades", format_number(row["trade_count"], digits=0)),
        ("Best Strategy", str(best["strategy_label"])),
        ("Best Net PnL", format_number(best["total_net_pnl"], digits=3, signed=True)),
        ("Mean Net Edge", format_number(row["mean_net_expected_edge"], percent=True, signed=True)),
    ]
    return "".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )


def build_dashboard_html(
    *,
    field_matrix: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    pair_matrix: pd.DataFrame,
    top_combos: pd.DataFrame,
    bottom_combos: pd.DataFrame,
    timeseries: pd.DataFrame,
    primary_strategy_id: str,
    min_combo_trades: int,
    fee_rate: float,
    round_up_to_cent: bool,
) -> str:
    primary_segments = segment_summary[
        (segment_summary["strategy_id"] == primary_strategy_id)
        & (segment_summary["segment_field"].isin(["predicted_net_edge_bucket", "q_shock_bucket", "time_to_expiry_bucket", "model_disagreement_bucket", "volatility_regime", "market_quality_bucket"]))
    ].copy()
    primary_edge = edge_summary[edge_summary["strategy_id"] == primary_strategy_id].copy()
    primary_top = top_combos[top_combos["strategy_id"] == primary_strategy_id].copy()
    primary_bottom = bottom_combos[bottom_combos["strategy_id"] == primary_strategy_id].copy()

    figures = [
        cumulative_net_pnl_figure(timeseries),
        predicted_edge_bar_figure(edge_summary, primary_strategy_id=primary_strategy_id),
        top_combo_figure(top_combos, primary_strategy_id=primary_strategy_id),
        heatmap_figure(
            pair_matrix,
            primary_strategy_id=primary_strategy_id,
            feature_x="predicted_net_edge_bucket",
            feature_y="q_shock_bucket",
            title="Mean Net PnL: Predicted Net Edge x q_shock",
        ),
        heatmap_figure(
            pair_matrix,
            primary_strategy_id=primary_strategy_id,
            feature_x="predicted_net_edge_bucket",
            feature_y="time_to_expiry_bucket",
            title="Mean Net PnL: Predicted Net Edge x Time To Expiry",
        ),
        heatmap_figure(
            pair_matrix,
            primary_strategy_id=primary_strategy_id,
            feature_x="predicted_net_edge_bucket",
            feature_y="model_disagreement_bucket",
            title="Mean Net PnL: Predicted Net Edge x Model Disagreement",
        ),
        heatmap_figure(
            pair_matrix,
            primary_strategy_id=primary_strategy_id,
            feature_x="predicted_net_edge_bucket",
            feature_y="volatility_regime",
            title="Mean Net PnL: Predicted Net Edge x Volatility Regime",
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model N Trade Diagnostics</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model N Diagnostics</p>
    <h1>Trade Ledger, Edge Buckets, And Feature Matrices</h1>
    <p class="lead">
      This dashboard stores the requested trade-level fields, buckets predicted net edge against realized
      net edge, and ranks pairwise feature combinations. Pairwise matrices are used instead of one large
      multi-feature cube because sparse buckets can look attractive by accident. Ranked combinations use a
      minimum of <code>{min_combo_trades}</code> trades. Fee rate is <code>{fee_rate}</code>; cent round-up is
      <code>{round_up_to_cent}</code>.
    </p>
    <div class="cards">{dashboard_cards(strategy_summary, primary_strategy_id=primary_strategy_id)}</div>
  </section>

  <h2>Charts</h2>
  <section class="chart-wrap">{figures_to_html(figures)}</section>

  <h2>Predicted Net Edge Buckets</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(primary_edge))}</div></section>

  <h2>Primary Strategy Segments</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(primary_segments))}</div></section>

  <h2>Best Feature Combinations</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(primary_top), max_rows=30)}</div></section>

  <h2>Worst Feature Combinations</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(primary_bottom), max_rows=30)}</div></section>

  <h2>Strategy Summary</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(format_table(strategy_summary))}</div></section>

  <h2>Field Matrix</h2>
  <section class="panel"><div class="table-wrap">{dataframe_to_html_table(field_matrix)}</div></section>
</main>
</body>
</html>
"""


def write_readme(
    output_dir: Path,
    *,
    fee_rate: float,
    round_up_to_cent: bool,
    edge_threshold: float,
    primary_strategy_id: str,
    min_combo_trades: int,
) -> None:
    text = f"""Model N Trade Diagnostics

Signal rule:
- Trade all favorable gross-edge rows.
- BUY_YES when selected strategy YES probability minus selected YES entry price exceeds {edge_threshold}.
- BUY_NO when selected strategy NO probability minus selected NO entry price exceeds {edge_threshold}.

Fee model:
- Taker fee = fee_rate * selected_side_price * (1 - selected_side_price)
- fee_rate = {fee_rate}
- Cent round-up enabled = {round_up_to_cent}

Execution price note:
- If selected-side ask columns are available, the script uses those as executable entry prices.
- The current Model_K/Model_N evaluated raw files only contain a single p_kalshi price from the historical price file.
- Those source files were produced with KALSHI_PRICE_FIELD = yes_mid, so the dashboard marks market_price_is_executable = False and uses the stored price as a proxy.
- Spread/liquidity fields are retained but marked unavailable unless future raw files include bid/ask/liquidity columns.

Primary dashboard strategy:
- {primary_strategy_id}

Combination matrix:
- Pairwise feature combinations are ranked with minimum trade count = {min_combo_trades}.
- This avoids sparse multi-way segment overfitting while still showing which feature combinations have persistent realized net PnL.

Outputs:
- trade_diagnostics_ledger.csv
- trade_field_matrix.csv
- strategy_summary.csv
- pnl_timeseries.csv
- segment_summary.csv
- predicted_net_edge_bucket_summary.csv
- feature_pair_matrix.csv
- top_feature_combinations.csv
- bottom_feature_combinations.csv
- diagnostics_dashboard.html
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.edge_threshold < 0:
        raise ValueError("--edge-threshold must be non-negative.")
    if args.fee_rate < 0:
        raise ValueError("--fee-rate must be non-negative.")
    if args.min_combo_trades < 1:
        raise ValueError("--min-combo-trades must be at least 1.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round_up_to_cent = not args.no_fee_round_up

    model_n = load_raw_values(args.model_n_raw_values.resolve(), model_name="Model N")
    model_k = load_raw_values(args.model_k_raw_values.resolve(), model_name="Model K")
    volatility = load_volatility_rows(args.volatility_rows.resolve())
    overlap = build_overlap(model_n, model_k, volatility)
    trades = build_all_trades(
        overlap,
        edge_threshold=args.edge_threshold,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
    )
    if trades.empty:
        raise ValueError("No trade signals were generated.")

    field_matrix = build_field_matrix()
    strategy_summary = build_strategy_summary(trades)
    timeseries = build_pnl_timeseries(trades)
    segment_summary = build_segment_summary(trades)
    edge_summary = build_predicted_net_edge_summary(trades)
    pair_matrix = build_feature_pair_matrix(trades)
    top_combos, bottom_combos = build_ranked_combinations(pair_matrix, min_trades=args.min_combo_trades)

    trades[ledger_columns(trades)].to_csv(output_dir / "trade_diagnostics_ledger.csv", index=False)
    field_matrix.to_csv(output_dir / "trade_field_matrix.csv", index=False)
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    timeseries.to_csv(output_dir / "pnl_timeseries.csv", index=False)
    segment_summary.to_csv(output_dir / "segment_summary.csv", index=False)
    edge_summary.to_csv(output_dir / "predicted_net_edge_bucket_summary.csv", index=False)
    pair_matrix.to_csv(output_dir / "feature_pair_matrix.csv", index=False)
    top_combos.to_csv(output_dir / "top_feature_combinations.csv", index=False)
    bottom_combos.to_csv(output_dir / "bottom_feature_combinations.csv", index=False)

    dashboard_html = build_dashboard_html(
        field_matrix=field_matrix,
        strategy_summary=strategy_summary,
        segment_summary=segment_summary,
        edge_summary=edge_summary,
        pair_matrix=pair_matrix,
        top_combos=top_combos,
        bottom_combos=bottom_combos,
        timeseries=timeseries,
        primary_strategy_id=args.primary_strategy_id,
        min_combo_trades=args.min_combo_trades,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
    )
    (output_dir / "diagnostics_dashboard.html").write_text(dashboard_html, encoding="utf-8")
    write_readme(
        output_dir,
        fee_rate=args.fee_rate,
        round_up_to_cent=round_up_to_cent,
        edge_threshold=args.edge_threshold,
        primary_strategy_id=args.primary_strategy_id,
        min_combo_trades=args.min_combo_trades,
    )

    display = strategy_summary[
        [
            "strategy_label",
            "trade_count",
            "total_fees",
            "total_net_pnl",
            "mean_net_expected_edge",
            "realized_net_edge",
            "roi_after_fees",
        ]
    ]
    print(f"Trade diagnostics output directory: {output_dir}")
    print(display.to_string(index=False))
    print(f"Dashboard: {output_dir / 'diagnostics_dashboard.html'}")


if __name__ == "__main__":
    main()
