from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover - Python 3.9+ is expected.
    raise SystemExit("Python 3.9+ required for zoneinfo.") from exc

try:  # Prefer SciPy when the user's environment has it.
    from scipy.optimize import minimize  # type: ignore

    HAS_SCIPY = True
except Exception:  # pragma: no cover - exercised when SciPy is unavailable.
    minimize = None
    HAS_SCIPY = False


NY = ZoneInfo("America/New_York")
UTC = timezone.utc

SYMBOL = "BTCUSDT"
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"
USER_AGENT = "model-c-high-vol-garch-studentt-builder/1.0"

OUTPUT_FOLDER_NAME = "Model_C_Output_Raw_Data"
EPS = 1e-12
MIN_VARIANCE = 1e-12
VOLATILITY_WINDOW_HOURS = 72
VOLATILITY_WINDOW_MINUTES = VOLATILITY_WINDOW_HOURS * 60

REGIME_SEGMENTS: Dict[str, Dict[str, str]] = {
    "low_volatility": {
        "flag": "is_low_volatility",
        "label": "Low Volatility",
    },
    "standard_volatility": {
        "flag": "is_standard_volatility",
        "label": "Standard Volatility",
    },
    "high_volatility": {
        "flag": "is_high_volatility",
        "label": "High Volatility",
    },
    "low_volatility_extreme": {
        "flag": "is_low_volatility_extreme",
        "label": "Low Volatility Extreme",
    },
    "high_volatility_extreme": {
        "flag": "is_high_volatility_extreme",
        "label": "High Volatility Extreme",
    },
}

CONTRACT_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("ATM", "ATM_market_ticker", "ATM_strike"),
    ("OTM+1", "OTM+1_market_ticker", "OTM+1_strike"),
    ("OTM-1", "OTM-1_market_ticker", "OTM-1_strike"),
)


@dataclass
class GarchStudentTParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float


@dataclass
class GarchFitState:
    params: GarchStudentTParams
    transformed: np.ndarray
    neg_log_likelihood: float
    next_variance: float
    fitted_at_price_index: int


@dataclass
class RegimeTrainingSample:
    blocks: List[np.ndarray]
    n_returns: int
    n_blocks: int
    n_hours: int
    lookback_start_utc: pd.Timestamp
    lookback_end_utc: pd.Timestamp


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_settlement_csv(root: Path) -> Path:
    corrected = root / "Data_Sourcing" / "Settlement_Outcomes" / "kalshi_btc_atm_settlements.csv"
    typo_legacy = root / "Data_Sourcing" / "Settlement_Outocmes" / "kalshi_btc_atm_settlements.csv"
    return corrected if corrected.exists() or not typo_legacy.exists() else typo_legacy


def default_kalshi_price_dir(root: Path) -> Path:
    return root / "Data_Sourcing" / "Kalshi_Pricing_Fetch" / "hourly_events_price_data"


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build Model C raw hourly forecast files by combining settlement-defined Kalshi "
            "contracts with a Student-t GARCH(1,1) Monte Carlo BTC model whose parameters are "
            "fit on trailing regime-filtered historical returns."
        )
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=default_settlement_csv(root),
        help="Settlement CSV used as the hourly contract manifest.",
    )
    parser.add_argument(
        "--kalshi-price-dir",
        type=Path,
        default=default_kalshi_price_dir(root),
        help="Kalshi hourly pricing directory used to define the allowed event universe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME,
        help="Directory where per-hour forecast CSV files will be written.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=SYMBOL,
        help="Binance spot symbol used for minute data downloads.",
    )
    parser.add_argument(
        "--history-minutes",
        type=int,
        default=60 * 24 * 2,
        help="Number of recent 1-minute returns used to initialize the live variance state at each refit.",
    )
    parser.add_argument(
        "--regime-lookback-days",
        type=int,
        default=90,
        help="Historical lookback window, in days, used to collect regime-specific returns for parameter MLE.",
    )
    parser.add_argument(
        "--training-volatility-segment",
        type=str,
        choices=tuple(REGIME_SEGMENTS.keys()),
        default="high_volatility",
        help=(
            "Volatility regime used to select the parameter-estimation sample. "
            "The regime labels follow the same trailing-window rules as the existing volatility scripts."
        ),
    )
    parser.add_argument(
        "--refit-every-minutes",
        type=int,
        default=60,
        help=(
            "Re-fit the GARCH/Student-t parameters on this cadence. Between refits, the script "
            "updates the conditional variance recursively each minute."
        ),
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=10_000,
        help="Monte Carlo terminal paths per forecast timestamp.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Monte Carlo reproducibility.",
    )
    parser.add_argument(
        "--binance-sleep-seconds",
        type=float,
        default=0.05,
        help="Sleep interval between Binance pagination requests.",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="Optional cap on the number of settlement hours to process, for sanity checks.",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=125,
        help="Maximum optimizer iterations for each GARCH fit.",
    )
    parser.add_argument(
        "--min-history-observations",
        type=int,
        default=240,
        help="Minimum number of realized 1-minute returns required before a forecast is emitted.",
    )
    parser.add_argument(
        "--min-regime-observations",
        type=int,
        default=240,
        help="Minimum number of regime-filtered 1-minute returns required before a Model C fit is allowed.",
    )
    return parser.parse_args()


def to_utc_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware.")
    return int(dt.astimezone(UTC).timestamp() * 1000)


def event_filename(event_ticker: str, hour_start_utc: pd.Timestamp) -> str:
    return f"{hour_start_utc.strftime('%Y%m%d_%H00UTC')}__{event_ticker}.csv"


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_utc_ms: int,
    end_utc_ms: int,
    *,
    limit: int = 1000,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    rows: List[List[Any]] = []
    cur = start_utc_ms

    while cur < end_utc_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_utc_ms,
            "limit": limit,
        }
        response = requests.get(
            BINANCE_BASE + BINANCE_KLINES,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        rows.extend(data)
        last_open_time = int(data[-1][0])
        increment = 60_000 if interval == "1m" else 60 * 60 * 1000
        next_cur = last_open_time + increment
        if next_cur <= cur:
            break
        cur = next_cur
        if len(data) < limit:
            break
        time.sleep(sleep_seconds)

    if not rows:
        raise RuntimeError(f"No Binance {interval} klines returned.")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["open_time_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    return df.sort_values("open_time_utc").reset_index(drop=True)


def load_event_tickers_from_price_dir(price_dir: Path) -> List[str]:
    files = sorted(price_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No Kalshi pricing CSV files found in {price_dir}")

    event_tickers: List[str] = []
    for path in files:
        parts = path.name.split("__", 1)
        if len(parts) != 2 or not parts[1].endswith(".csv"):
            raise ValueError(f"Unexpected Kalshi pricing filename format: {path.name}")
        event_tickers.append(parts[1][:-4])
    return event_tickers


def load_settlement_manifest(settlement_csv: Path) -> pd.DataFrame:
    if not settlement_csv.exists():
        raise FileNotFoundError(f"Settlement CSV not found: {settlement_csv}")

    df = pd.read_csv(settlement_csv)
    required_columns = {
        "event_datetime",
        "forecast_hour_start_datetime",
        "event_ticker",
        "ATM_market_ticker",
        "ATM_strike",
        "OTM+1_market_ticker",
        "OTM+1_strike",
        "OTM-1_market_ticker",
        "OTM-1_strike",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{settlement_csv} is missing columns: {sorted(missing)}")

    df["event_datetime_utc"] = pd.to_datetime(df["event_datetime"], utc=True)
    df["forecast_hour_start_datetime_utc"] = pd.to_datetime(df["forecast_hour_start_datetime"], utc=True)
    for _, ticker_col, strike_col in CONTRACT_SPECS:
        df[ticker_col] = df[ticker_col].astype("string").str.strip()
        df.loc[df[ticker_col].isin(["", "nan", "None", "<NA>"]), ticker_col] = pd.NA
        df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")

    return df.sort_values(["forecast_hour_start_datetime_utc", "event_datetime_utc", "event_ticker"]).reset_index(drop=True)


def filter_settlements_to_event_universe(
    settlements: pd.DataFrame,
    allowed_event_tickers: List[str],
    event_limit: Optional[int],
) -> pd.DataFrame:
    allowed_set = set(allowed_event_tickers)
    filtered = settlements[settlements["event_ticker"].isin(allowed_set)].copy()
    if filtered.empty:
        raise RuntimeError("No settlement rows matched the Kalshi pricing-file event universe.")

    filtered = filtered.drop_duplicates(subset=["event_ticker"], keep="first")
    filtered["event_order"] = pd.Categorical(filtered["event_ticker"], categories=allowed_event_tickers, ordered=True)
    filtered = filtered.sort_values("event_order").drop(columns=["event_order"]).reset_index(drop=True)

    missing_from_settlements = [ticker for ticker in allowed_event_tickers if ticker not in set(filtered["event_ticker"])]
    if missing_from_settlements:
        raise RuntimeError(
            "Some pricing-file events were not present in the settlement manifest. "
            f"First missing event: {missing_from_settlements[0]}"
        )

    if event_limit is not None:
        filtered = filtered.head(event_limit).copy()
    return filtered


def expand_forecast_schedule(settlements: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for settlement in settlements.to_dict(orient="records"):
        hour_start = settlement["forecast_hour_start_datetime_utc"]
        event_end = settlement["event_datetime_utc"]
        total_minutes = int((event_end - hour_start).total_seconds() / 60.0)
        if total_minutes != 60:
            raise ValueError(
                f"Expected a 60-minute forecast horizon for {settlement['event_ticker']}, found {total_minutes} minutes."
            )

        for minute_number in range(total_minutes):
            forecast_dt = hour_start + pd.Timedelta(minutes=minute_number)
            rows.append(
                {
                    "Event": settlement["event_ticker"],
                    "event_ticker": settlement["event_ticker"],
                    "hour_start_utc": hour_start,
                    "event_datetime_utc": event_end,
                    "forecast_datetime_utc": forecast_dt,
                    "minute_number": minute_number,
                    "ATM_market_ticker": settlement["ATM_market_ticker"],
                    "ATM_strike": settlement["ATM_strike"],
                    "OTM+1_market_ticker": settlement["OTM+1_market_ticker"],
                    "OTM+1_strike": settlement["OTM+1_strike"],
                    "OTM-1_market_ticker": settlement["OTM-1_market_ticker"],
                    "OTM-1_strike": settlement["OTM-1_strike"],
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["forecast_datetime_utc", "event_ticker"]).reset_index(drop=True)
    out["steps_to_expiry"] = (
        (out["event_datetime_utc"] - out["forecast_datetime_utc"]).dt.total_seconds() / 60.0
    ).round().astype(int)
    return out


def derive_download_window(
    settlements: pd.DataFrame,
    *,
    history_minutes: int,
    regime_lookback_days: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_start = settlements["forecast_hour_start_datetime_utc"].min()
    max_end = settlements["event_datetime_utc"].max()
    regime_history_minutes = regime_lookback_days * 24 * 60 + VOLATILITY_WINDOW_MINUTES
    fetch_start = min_start - pd.Timedelta(minutes=max(history_minutes, regime_history_minutes) + 1)
    fetch_end = max_end
    return fetch_start, fetch_end


def prepare_minute_market_data(
    symbol: str,
    settlements: pd.DataFrame,
    *,
    history_minutes: int,
    regime_lookback_days: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    fetch_start, fetch_end = derive_download_window(
        settlements,
        history_minutes=history_minutes,
        regime_lookback_days=regime_lookback_days,
    )
    raw = fetch_binance_klines(
        symbol=symbol,
        interval="1m",
        start_utc_ms=to_utc_millis(fetch_start.to_pydatetime()),
        end_utc_ms=to_utc_millis((fetch_end + pd.Timedelta(minutes=1)).to_pydatetime()),
        sleep_seconds=sleep_seconds,
    )

    prices = raw[["open_time_utc", "open", "close"]].copy()
    prices = prices.rename(columns={"open_time_utc": "datetime_utc", "open": "spot_open", "close": "spot_close"})
    prices = prices.dropna(subset=["datetime_utc", "spot_open"]).sort_values("datetime_utc").reset_index(drop=True)

    expected_index = pd.date_range(fetch_start, fetch_end, freq="1min", tz=UTC)
    prices = prices.set_index("datetime_utc").reindex(expected_index)
    missing_count = int(prices["spot_open"].isna().sum())
    if missing_count:
        first_missing = prices[prices["spot_open"].isna()].index[0]
        raise RuntimeError(
            "Binance minute data has gaps after download. "
            f"Missing rows: {missing_count}; first missing timestamp: {first_missing.isoformat()}"
        )

    prices = prices.reset_index().rename(columns={"index": "datetime_utc"})
    prices["spot_open"] = pd.to_numeric(prices["spot_open"], errors="coerce")
    prices["spot_close"] = pd.to_numeric(prices["spot_close"], errors="coerce")
    prices["log_spot"] = np.log(prices["spot_open"])
    prices["log_return"] = prices["log_spot"].diff()
    prices["price_index"] = np.arange(len(prices), dtype=int)
    prices["hour_start_utc"] = prices["datetime_utc"].dt.floor("h")
    return prices


def compute_hourly_realized_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    work = prices.copy()
    work = work.dropna(subset=["datetime_utc", "spot_open", "spot_close"])
    work = work[(work["spot_open"] > 0) & (work["spot_close"] > 0)].copy()
    work["minute_log_return"] = np.log(work["spot_close"] / work["spot_open"])

    hourly = (
        work.groupby("hour_start_utc", as_index=False)
        .agg(
            realized_variance=("minute_log_return", lambda values: float(np.square(values).sum())),
            realized_volatility=("minute_log_return", lambda values: float(np.sqrt(np.square(values).sum()))),
            minute_bars=("minute_log_return", "size"),
        )
        .sort_values("hour_start_utc")
        .reset_index(drop=True)
    )
    return hourly


def add_realtime_window_thresholds(hourly_volatility: pd.DataFrame) -> pd.DataFrame:
    work = hourly_volatility.sort_values("hour_start_utc").reset_index(drop=True).copy()
    rolling = work["realized_volatility"].rolling(
        window=VOLATILITY_WINDOW_HOURS,
        min_periods=VOLATILITY_WINDOW_HOURS,
    )
    work["rolling_window_hours"] = VOLATILITY_WINDOW_HOURS
    work["rolling_window_minutes"] = VOLATILITY_WINDOW_MINUTES
    work["rolling_window_observations"] = rolling.count()
    work["q10_realized_volatility"] = rolling.quantile(0.10)
    work["q25_realized_volatility"] = rolling.quantile(0.25)
    work["q75_realized_volatility"] = rolling.quantile(0.75)
    work["q90_realized_volatility"] = rolling.quantile(0.90)
    work["rolling_percentile_rank"] = work["realized_volatility"].rolling(
        window=VOLATILITY_WINDOW_HOURS,
        min_periods=VOLATILITY_WINDOW_HOURS,
    ).apply(lambda values: float(np.mean(values <= values[-1])), raw=True)

    work["is_low_volatility"] = work["realized_volatility"] <= work["q25_realized_volatility"]
    work["is_standard_volatility"] = (
        (work["realized_volatility"] > work["q25_realized_volatility"])
        & (work["realized_volatility"] < work["q75_realized_volatility"])
    )
    work["is_high_volatility"] = work["realized_volatility"] >= work["q75_realized_volatility"]
    work["is_low_volatility_extreme"] = work["realized_volatility"] <= work["q10_realized_volatility"]
    work["is_high_volatility_extreme"] = work["realized_volatility"] >= work["q90_realized_volatility"]
    work["primary_volatility_band"] = np.select(
        [
            work["is_low_volatility"],
            work["is_high_volatility"],
        ],
        [
            "Low Volatility",
            "High Volatility",
        ],
        default="Standard Volatility",
    )
    return work


def attach_hourly_regime_flags(prices: pd.DataFrame, hourly_volatility: pd.DataFrame) -> pd.DataFrame:
    flag_columns = [
        "hour_start_utc",
        "primary_volatility_band",
        "rolling_percentile_rank",
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
    return prices.merge(hourly_volatility[flag_columns], on="hour_start_utc", how="left")


def contiguous_state_returns(log_returns: np.ndarray, current_price_index: int, history_minutes: int) -> np.ndarray:
    start_idx = current_price_index - history_minutes + 1
    if start_idx < 1:
        raise RuntimeError(
            f"Insufficient history window at price_index={current_price_index} for history_minutes={history_minutes}."
        )
    history_returns = log_returns[start_idx : current_price_index + 1]
    return history_returns[np.isfinite(history_returns)]


def select_regime_training_sample(
    prices: pd.DataFrame,
    *,
    forecast_dt: pd.Timestamp,
    regime_lookback_days: int,
    segment_key: str,
) -> RegimeTrainingSample:
    segment = REGIME_SEGMENTS[segment_key]
    segment_flag = segment["flag"]
    current_hour_start = forecast_dt.floor("h")
    lookback_start = current_hour_start - pd.Timedelta(days=regime_lookback_days)

    eligible = prices[
        (prices["hour_start_utc"] >= lookback_start)
        & (prices["hour_start_utc"] < current_hour_start)
        & (prices[segment_flag].fillna(False))
    ][["price_index", "hour_start_utc", "log_return"]].copy()
    eligible = eligible[np.isfinite(eligible["log_return"])]

    if eligible.empty:
        return RegimeTrainingSample(
            blocks=[],
            n_returns=0,
            n_blocks=0,
            n_hours=0,
            lookback_start_utc=lookback_start,
            lookback_end_utc=current_hour_start,
        )

    block_ids = eligible["price_index"].diff().fillna(1).ne(1).cumsum()
    blocks = [
        block["log_return"].to_numpy(dtype=float)
        for _, block in eligible.groupby(block_ids, sort=False)
        if len(block)
    ]
    return RegimeTrainingSample(
        blocks=blocks,
        n_returns=sum(len(block) for block in blocks),
        n_blocks=len(blocks),
        n_hours=int(eligible["hour_start_utc"].nunique()),
        lookback_start_utc=lookback_start,
        lookback_end_utc=current_hour_start,
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    p = min(max(probability, 1e-8), 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


def unpack_transformed_params(x: np.ndarray) -> GarchStudentTParams:
    mu = float(x[0])
    omega = float(math.exp(x[1]))
    persistence = 0.999 * sigmoid(float(x[2]))
    beta_share = sigmoid(float(x[3]))
    alpha = persistence * (1.0 - beta_share)
    beta = persistence * beta_share
    nu = 2.01 + float(math.exp(x[4]))
    return GarchStudentTParams(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu)


def pack_transformed_params(params: GarchStudentTParams) -> np.ndarray:
    persistence = min(max(params.alpha + params.beta, 1e-8), 0.999 - 1e-8)
    beta_share = params.beta / persistence if persistence > 0.0 else 0.5
    return np.array(
        [
            params.mu,
            math.log(max(params.omega, MIN_VARIANCE)),
            logit(persistence / 0.999),
            logit(beta_share),
            math.log(max(params.nu - 2.01, 1e-8)),
        ],
        dtype=float,
    )


def initial_params_from_returns(returns: np.ndarray) -> GarchStudentTParams:
    mu = float(np.nanmean(returns))
    variance = float(np.nanvar(returns))
    variance = max(variance, 1e-8)
    alpha = 0.06
    beta = 0.92
    persistence = alpha + beta
    if persistence >= 0.995:
        beta = 0.90
        alpha = 0.07
        persistence = alpha + beta
    omega = variance * max(1.0 - persistence, 1e-4)
    return GarchStudentTParams(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=8.0)


def initial_variance_for_fit(returns: np.ndarray, params: GarchStudentTParams) -> float:
    sample_variance = float(np.nanvar(returns))
    unconditional = params.omega / max(1.0 - params.alpha - params.beta, 1e-4)
    return max(sample_variance, unconditional, MIN_VARIANCE)


def student_t_neg_log_likelihood(
    transformed: np.ndarray,
    returns: np.ndarray,
) -> float:
    params = unpack_transformed_params(transformed)
    if not np.isfinite(params.mu + params.omega + params.alpha + params.beta + params.nu):
        return float("inf")
    if params.omega <= 0.0 or params.alpha < 0.0 or params.beta < 0.0 or params.alpha + params.beta >= 1.0 or params.nu <= 2.0:
        return float("inf")

    eps = returns - params.mu
    n_obs = len(eps)
    if n_obs == 0:
        return float("inf")

    h = np.empty(n_obs, dtype=float)
    h[0] = initial_variance_for_fit(returns, params)
    for i in range(1, n_obs):
        h[i] = max(params.omega + params.alpha * (eps[i - 1] ** 2) + params.beta * h[i - 1], MIN_VARIANCE)

    scaled = eps / np.sqrt(h)
    constant = (
        math.lgamma((params.nu + 1.0) / 2.0)
        - math.lgamma(params.nu / 2.0)
        - 0.5 * math.log((params.nu - 2.0) * math.pi)
    )
    log_likelihood = constant - 0.5 * np.log(h) - ((params.nu + 1.0) / 2.0) * np.log1p((scaled ** 2) / (params.nu - 2.0))
    value = -float(np.sum(log_likelihood))
    return value if np.isfinite(value) else float("inf")


def student_t_neg_log_likelihood_blocks(
    transformed: np.ndarray,
    return_blocks: List[np.ndarray],
) -> float:
    params = unpack_transformed_params(transformed)
    if not np.isfinite(params.mu + params.omega + params.alpha + params.beta + params.nu):
        return float("inf")
    if params.omega <= 0.0 or params.alpha < 0.0 or params.beta < 0.0 or params.alpha + params.beta >= 1.0 or params.nu <= 2.0:
        return float("inf")

    total_value = 0.0
    for returns in return_blocks:
        if len(returns) == 0:
            continue
        block_value = student_t_neg_log_likelihood(pack_transformed_params(params), returns)
        if not np.isfinite(block_value):
            return float("inf")
        total_value += block_value
    return total_value if np.isfinite(total_value) else float("inf")


def coordinate_pattern_search(
    objective,
    x0: np.ndarray,
    *,
    max_iter: int,
    initial_step_scale: float,
) -> Tuple[np.ndarray, float]:
    x = x0.astype(float).copy()
    value = float(objective(x))
    steps = np.array(
        [
            max(1e-6, np.std([x[0], 0.0]) * initial_step_scale + 1e-4),
            0.50 * initial_step_scale,
            0.35 * initial_step_scale,
            0.35 * initial_step_scale,
            0.25 * initial_step_scale,
        ],
        dtype=float,
    )

    for _ in range(max_iter):
        improved = False
        for idx in range(len(x)):
            for direction in (1.0, -1.0):
                candidate = x.copy()
                candidate[idx] += direction * steps[idx]
                candidate_value = float(objective(candidate))
                if candidate_value + 1e-9 < value:
                    x = candidate
                    value = candidate_value
                    improved = True
        if improved:
            continue
        steps *= 0.5
        if float(np.max(steps)) < 1e-4:
            break

    return x, value


def garch_filter_next_variance(
    returns: np.ndarray,
    params: GarchStudentTParams,
) -> float:
    eps = returns - params.mu
    variance = initial_variance_for_fit(returns, params)
    if len(eps) == 0:
        return variance

    current_variance = variance
    for i in range(len(eps)):
        if i > 0:
            current_variance = max(
                params.omega + params.alpha * (eps[i - 1] ** 2) + params.beta * current_variance,
                MIN_VARIANCE,
            )

    next_variance = max(params.omega + params.alpha * (eps[-1] ** 2) + params.beta * current_variance, MIN_VARIANCE)
    return next_variance


def fit_garch_student_t(
    returns: np.ndarray,
    *,
    previous_state: Optional[GarchFitState],
    max_iter: int,
) -> GarchFitState:
    if len(returns) == 0:
        raise ValueError("Cannot fit GARCH with an empty return window.")

    initial_params = previous_state.params if previous_state is not None else initial_params_from_returns(returns)
    x0 = pack_transformed_params(initial_params)
    objective = lambda x: student_t_neg_log_likelihood(x, returns)

    if HAS_SCIPY and minimize is not None:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        if result.success and np.isfinite(result.fun):
            best_x = np.array(result.x, dtype=float)
            best_value = float(result.fun)
        else:
            best_x, best_value = coordinate_pattern_search(
                objective,
                x0,
                max_iter=max_iter,
                initial_step_scale=0.80,
            )
    else:
        initial_scale = 0.45 if previous_state is not None else 1.00
        best_x, best_value = coordinate_pattern_search(
            objective,
            x0,
            max_iter=max_iter,
            initial_step_scale=initial_scale,
        )

    best_params = unpack_transformed_params(best_x)
    next_variance = garch_filter_next_variance(returns, best_params)
    return GarchFitState(
        params=best_params,
        transformed=best_x,
        neg_log_likelihood=best_value,
        next_variance=next_variance,
        fitted_at_price_index=-1,
    )


def fit_garch_student_t_on_blocks(
    return_blocks: List[np.ndarray],
    *,
    previous_state: Optional[GarchFitState],
    max_iter: int,
) -> GarchFitState:
    cleaned_blocks = [np.asarray(block, dtype=float) for block in return_blocks if len(block)]
    if not cleaned_blocks:
        raise ValueError("Cannot fit GARCH with an empty regime-filtered return sample.")

    pooled_returns = np.concatenate(cleaned_blocks)
    initial_params = previous_state.params if previous_state is not None else initial_params_from_returns(pooled_returns)
    x0 = pack_transformed_params(initial_params)
    objective = lambda x: student_t_neg_log_likelihood_blocks(x, cleaned_blocks)

    if HAS_SCIPY and minimize is not None:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        if result.success and np.isfinite(result.fun):
            best_x = np.array(result.x, dtype=float)
            best_value = float(result.fun)
        else:
            best_x, best_value = coordinate_pattern_search(
                objective,
                x0,
                max_iter=max_iter,
                initial_step_scale=0.80,
            )
    else:
        initial_scale = 0.45 if previous_state is not None else 1.00
        best_x, best_value = coordinate_pattern_search(
            objective,
            x0,
            max_iter=max_iter,
            initial_step_scale=initial_scale,
        )

    return GarchFitState(
        params=unpack_transformed_params(best_x),
        transformed=best_x,
        neg_log_likelihood=best_value,
        next_variance=MIN_VARIANCE,
        fitted_at_price_index=-1,
    )


def advance_variance_state(
    next_variance: float,
    new_returns: np.ndarray,
    params: GarchStudentTParams,
) -> float:
    variance = max(next_variance, MIN_VARIANCE)
    for realized_return in new_returns:
        residual = float(realized_return) - params.mu
        variance = max(params.omega + params.alpha * (residual ** 2) + params.beta * variance, MIN_VARIANCE)
    return variance


def standardized_student_t_draws(rng: np.random.Generator, n_trials: int, nu: float) -> np.ndarray:
    half = (n_trials + 1) // 2
    base = rng.standard_t(nu, size=half) * math.sqrt((nu - 2.0) / nu)
    mirrored = -base
    if n_trials % 2 == 0:
        return np.concatenate([base, mirrored])
    return np.concatenate([base, mirrored[:-1]])


def simulate_terminal_prices(
    current_spot: float,
    initial_variance: float,
    horizon_steps: int,
    params: GarchStudentTParams,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    log_prices = np.full(n_trials, math.log(current_spot), dtype=float)
    variances = np.full(n_trials, max(initial_variance, MIN_VARIANCE), dtype=float)

    for _ in range(horizon_steps):
        z = standardized_student_t_draws(rng, n_trials=n_trials, nu=params.nu)
        step_returns = params.mu + np.sqrt(variances) * z
        log_prices += step_returns
        residual_sq = (step_returns - params.mu) ** 2
        variances = np.maximum(params.omega + params.alpha * residual_sq + params.beta * variances, MIN_VARIANCE)

    return np.exp(log_prices)


def probability_from_terminals(terminals: np.ndarray, strike: Optional[float]) -> Optional[float]:
    if strike is None or pd.isna(strike):
        return None
    return float(np.mean(terminals >= float(strike)))


def build_output_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(list(rows))
    out = out.sort_values(["datetime", "Event", "minute_number"]).reset_index(drop=True)
    ordered_columns = [
        "Event",
        "datetime",
        "minute_number",
        "ATM_market_ticker",
        "ATM_strike",
        "ATM_price",
        "OTM+1_market_ticker",
        "OTM+1_strike",
        "OTM+1_price",
        "OTM-1_market_ticker",
        "OTM-1_strike",
        "OTM-1_price",
    ]
    return out[ordered_columns]


def write_hourly_output_files(output_dir: Path, forecasts: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (event_ticker, hour_start_utc), part in forecasts.groupby(["Event", "hour_start_utc"], sort=True):
        out_path = output_dir / event_filename(event_ticker=event_ticker, hour_start_utc=hour_start_utc)
        frame = build_output_frame(part.to_dict(orient="records"))
        frame.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()

    if args.regime_lookback_days <= 0:
        raise ValueError("--regime-lookback-days must be positive.")
    if args.history_minutes < args.min_history_observations:
        raise ValueError("--history-minutes must be at least as large as --min-history-observations.")
    if args.refit_every_minutes <= 0:
        raise ValueError("--refit-every-minutes must be positive.")
    if args.mc_trials <= 1:
        raise ValueError("--mc-trials must be greater than 1.")
    if args.min_regime_observations <= 0:
        raise ValueError("--min-regime-observations must be positive.")

    settlements = load_settlement_manifest(args.settlement_csv.resolve())
    if settlements.empty:
        raise RuntimeError("No settlement rows were loaded.")
    allowed_event_tickers = load_event_tickers_from_price_dir(args.kalshi_price_dir.resolve())
    settlements = filter_settlements_to_event_universe(
        settlements,
        allowed_event_tickers=allowed_event_tickers,
        event_limit=args.event_limit,
    )
    print(
        f"Using {len(settlements)} settlement events aligned to "
        f"{len(allowed_event_tickers)} Kalshi pricing-file events from {args.kalshi_price_dir.resolve()}"
    )

    schedule = expand_forecast_schedule(settlements)
    prices = prepare_minute_market_data(
        symbol=args.symbol,
        settlements=settlements,
        history_minutes=args.history_minutes,
        regime_lookback_days=args.regime_lookback_days,
        sleep_seconds=args.binance_sleep_seconds,
    )
    hourly_volatility = add_realtime_window_thresholds(compute_hourly_realized_volatility(prices))
    prices = attach_hourly_regime_flags(prices, hourly_volatility)

    price_index_lookup = pd.Series(prices["price_index"].to_numpy(), index=prices["datetime_utc"]).to_dict()
    log_returns = prices["log_return"].to_numpy()
    spots = prices["spot_open"].to_numpy()
    training_segment = REGIME_SEGMENTS[args.training_volatility_segment]

    print(
        "Model C regime fit setup: "
        f"segment={training_segment['label']} "
        f"lookback_days={args.regime_lookback_days} "
        f"state_history_minutes={args.history_minutes} "
        f"volatility_window_hours={VOLATILITY_WINDOW_HOURS}"
    )

    rng = np.random.default_rng(args.seed)
    fit_state: Optional[GarchFitState] = None
    fitted_forecasts: List[Dict[str, Any]] = []
    last_refit_schedule_idx: Optional[int] = None

    total = len(schedule)
    for schedule_idx, forecast in enumerate(schedule.to_dict(orient="records"), start=1):
        forecast_dt = forecast["forecast_datetime_utc"]
        current_price_index = price_index_lookup.get(forecast_dt)
        if current_price_index is None:
            raise RuntimeError(f"Missing Binance minute price for forecast timestamp {forecast_dt.isoformat()}.")
        if current_price_index < args.min_history_observations:
            raise RuntimeError(
                f"Not enough price history before {forecast_dt.isoformat()}. "
                f"Need at least {args.min_history_observations} observed returns."
            )

        need_refit = fit_state is None
        if last_refit_schedule_idx is not None and schedule_idx - last_refit_schedule_idx >= args.refit_every_minutes:
            need_refit = True

        if need_refit:
            state_returns = contiguous_state_returns(
                log_returns=log_returns,
                current_price_index=current_price_index,
                history_minutes=args.history_minutes,
            )
            if len(state_returns) < args.min_history_observations:
                raise RuntimeError(
                    f"Only {len(state_returns)} returns available at {forecast_dt.isoformat()}, "
                    f"below min_history_observations={args.min_history_observations}."
                )

            regime_sample = select_regime_training_sample(
                prices,
                forecast_dt=forecast_dt,
                regime_lookback_days=args.regime_lookback_days,
                segment_key=args.training_volatility_segment,
            )
            if regime_sample.n_returns < args.min_regime_observations:
                raise RuntimeError(
                    "Insufficient regime-filtered history for Model C fit at "
                    f"{forecast_dt.isoformat()}. Needed at least {args.min_regime_observations} "
                    f"returns inside the {training_segment['label']} segment over the trailing "
                    f"{args.regime_lookback_days} days, found {regime_sample.n_returns}."
                )

            fit_state = fit_garch_student_t_on_blocks(
                regime_sample.blocks,
                previous_state=fit_state,
                max_iter=args.optimizer_maxiter,
            )
            fit_state.next_variance = garch_filter_next_variance(state_returns, fit_state.params)
            fit_state.fitted_at_price_index = current_price_index
            last_refit_schedule_idx = schedule_idx
            print(
                f"[refit {schedule_idx}/{total}] {forecast['Event']} {forecast_dt.isoformat()} "
                f"| training_segment={training_segment['label']} "
                f"returns={regime_sample.n_returns} blocks={regime_sample.n_blocks} hours={regime_sample.n_hours}"
            )
        else:
            assert fit_state is not None
            if current_price_index > fit_state.fitted_at_price_index:
                newly_realized = log_returns[fit_state.fitted_at_price_index + 1 : current_price_index + 1]
                newly_realized = newly_realized[np.isfinite(newly_realized)]
                if len(newly_realized):
                    fit_state.next_variance = advance_variance_state(
                        fit_state.next_variance,
                        newly_realized,
                        fit_state.params,
                    )
                fit_state.fitted_at_price_index = current_price_index

        assert fit_state is not None
        terminals = simulate_terminal_prices(
            current_spot=float(spots[current_price_index]),
            initial_variance=fit_state.next_variance,
            horizon_steps=int(forecast["steps_to_expiry"]),
            params=fit_state.params,
            n_trials=args.mc_trials,
            rng=rng,
        )

        fitted_forecasts.append(
            {
                "Event": forecast["Event"],
                "hour_start_utc": forecast["hour_start_utc"],
                "datetime": forecast["forecast_datetime_utc"].isoformat(),
                "minute_number": int(forecast["minute_number"]),
                "ATM_market_ticker": forecast["ATM_market_ticker"],
                "ATM_strike": forecast["ATM_strike"],
                "ATM_price": probability_from_terminals(terminals, forecast["ATM_strike"]),
                "OTM+1_market_ticker": forecast["OTM+1_market_ticker"],
                "OTM+1_strike": forecast["OTM+1_strike"],
                "OTM+1_price": probability_from_terminals(terminals, forecast["OTM+1_strike"]),
                "OTM-1_market_ticker": forecast["OTM-1_market_ticker"],
                "OTM-1_strike": forecast["OTM-1_strike"],
                "OTM-1_price": probability_from_terminals(terminals, forecast["OTM-1_strike"]),
            }
        )

        if schedule_idx == 1 or schedule_idx % 1000 == 0 or schedule_idx == total:
            params = fit_state.params
            print(
                f"[{schedule_idx}/{total}] {forecast['Event']} {forecast['forecast_datetime_utc'].isoformat()} "
                f"| mu={params.mu:.6g} omega={params.omega:.3e} alpha={params.alpha:.4f} "
                f"beta={params.beta:.4f} nu={params.nu:.3f}"
            )

    forecast_df = pd.DataFrame(fitted_forecasts)
    write_hourly_output_files(args.output_dir.resolve(), forecast_df)

    print("\nDone.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Hourly files written: {forecast_df[['Event', 'hour_start_utc']].drop_duplicates().shape[0]}")
    print(f"Forecast rows written: {len(forecast_df)}")


if __name__ == "__main__":
    main()
