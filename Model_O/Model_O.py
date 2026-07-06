from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


CONTRACT_LABELS = ("ATM", "OTM+1", "OTM-1")
RAW_OUTPUT_FOLDER_NAME = "Model_O_Output_Raw_Data"
GATE_SCENARIO = "drop_stale"
GATE_MODEL = "logistic_regression"
GATE_TIMESTAMP_COLUMN = "target_minute"
EXPONENTIAL_HYBRID_LAMBDA = 2.0


@dataclass(frozen=True)
class BuildSummary:
    normal_source_dir: Path
    shock_source_dir: Path
    shock_probability_csv: Path
    output_dir: Path
    q_rows: int
    normal_files: int
    files_with_shock_source: int
    files_written: int
    source_rows_seen: int
    rows_written: int
    rows_without_q_shock: int
    rows_without_shock_source: int
    min_q_shock: float
    max_q_shock: float
    mean_q_shock: float


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build Model O hybrid raw forecasts. The default Model O price remains the "
            "linear hybrid p_final = (1 - q_shock) * p_normal + q_shock * p_shock. "
            "The output also includes an exponential lambda=2 hybrid diagnostic column "
            "for downstream strategy analysis. p_normal comes from Model R, p_shock comes from the "
            "full Model H fair-value source used by Model M, and q_shock comes from the "
            "cloned Model M gate-prediction CSV."
        )
    )
    parser.add_argument(
        "--normal-raw-dir",
        type=Path,
        default=root / "Model_R" / "Model_R_Output_Raw_Data",
        help="Model R raw output directory, used as Model O model_a / p_normal.",
    )
    parser.add_argument(
        "--shock-raw-dir",
        type=Path,
        default=root / "Model_H" / "Model_H_Output_Raw_Data",
        help=(
            "Full shock fair-value raw output directory. Model M is a gate over this source, "
            "so this supplies p_shock for all overlap minutes."
        ),
    )
    parser.add_argument(
        "--shock-probability-csv",
        type=Path,
        default=root / "Model_O" / "Additional_Data" / "all_test_predictions.csv",
        help="Cloned Model M Additional_Data/all_test_predictions.csv used for q_shock.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / RAW_OUTPUT_FOLDER_NAME,
        help="Directory where Model O per-hour raw forecast CSV files will be written.",
    )
    parser.add_argument("--gate-scenario", default=GATE_SCENARIO, help="Shock CSV scenario value to use.")
    parser.add_argument("--gate-model", default=GATE_MODEL, help="Shock CSV model value to use.")
    parser.add_argument(
        "--gate-timestamp-column",
        choices=("minute", "target_minute"),
        default=GATE_TIMESTAMP_COLUMN,
        help="Timestamp column in the shock CSV that should align to forecast datetimes.",
    )
    parser.add_argument(
        "--missing-q-policy",
        choices=("skip", "zero"),
        default="skip",
        help=(
            "How to handle forecast minutes absent from the shock-probability CSV. "
            "'skip' keeps only q_shock-covered minutes; 'zero' falls back to p_normal."
        ),
    )
    return parser.parse_args()


def exponential_shock_weight(q: pd.Series, lambda_value: float = EXPONENTIAL_HYBRID_LAMBDA) -> pd.Series:
    q = pd.to_numeric(q, errors="coerce").clip(0.0, 1.0)
    if abs(lambda_value) < 1e-12:
        return q
    denominator = np.expm1(lambda_value)
    if abs(denominator) < 1e-12:
        return q
    return pd.Series(np.expm1(lambda_value * q) / denominator, index=q.index).clip(0.0, 1.0)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()


def load_q_shock(
    path: Path,
    *,
    scenario: str,
    model: str,
    timestamp_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Shock probability CSV not found: {path}")

    frame = pd.read_csv(path)
    required = {"scenario", "model", "minute", "target_minute", "y_prob"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame[(frame["scenario"].astype(str) == scenario) & (frame["model"].astype(str) == model)].copy()
    if frame.empty:
        raise RuntimeError(f"No q_shock rows matched scenario={scenario!r}, model={model!r}.")

    frame["minute"] = pd.to_datetime(frame["minute"], utc=True)
    frame["target_minute"] = pd.to_datetime(frame["target_minute"], utc=True)
    frame["forecast_datetime_utc"] = pd.to_datetime(frame[timestamp_column], utc=True)
    frame["q_shock"] = pd.to_numeric(frame["y_prob"], errors="coerce").clip(0.0, 1.0)
    frame = frame.dropna(subset=["forecast_datetime_utc", "q_shock"]).copy()
    if frame.empty:
        raise RuntimeError("No usable q_shock rows remained after timestamp/probability parsing.")

    frame = frame.sort_values(["forecast_datetime_utc", "q_shock"], ascending=[True, False])
    frame = frame.drop_duplicates(subset=["forecast_datetime_utc"], keep="first")
    return frame[["forecast_datetime_utc", "q_shock"]].reset_index(drop=True)


def source_files(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw source directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    return files


def read_raw(path: Path, *, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Event", "datetime", "minute_number"}
    for label in CONTRACT_LABELS:
        required.update({f"{label}_market_ticker", f"{label}_strike", f"{label}_price"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_name} file {path} is missing columns: {sorted(missing)}")

    frame["forecast_datetime_utc"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["minute_number"] = pd.to_numeric(frame["minute_number"], errors="coerce")
    return frame


def merge_normal_shock(normal: pd.DataFrame, shock: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["Event", "forecast_datetime_utc", "minute_number"]
    normal_cols = list(dict.fromkeys([*merge_cols, "datetime", *component_columns(prefix="normal")]))
    shock_cols = list(dict.fromkeys([*merge_cols, *component_columns(prefix="shock"), "refit_id"]))

    left = normal[normal_cols].copy()
    right = shock[[col for col in shock_cols if col in shock.columns]].copy()
    return left.merge(right, on=merge_cols, how="inner", suffixes=("_normal", "_shock"), validate="one_to_one")


def component_columns(*, prefix: str) -> List[str]:
    cols: List[str] = []
    for label in CONTRACT_LABELS:
        cols.extend([f"{label}_market_ticker", f"{label}_strike", f"{label}_price"])
    if prefix == "shock":
        for label in CONTRACT_LABELS:
            for optional in (f"{label}_raw_price", f"{label}_calibrated_price"):
                cols.append(optional)
    return cols


def add_hybrid_probabilities(merged: pd.DataFrame, q_shock: pd.DataFrame, *, missing_q_policy: str) -> pd.DataFrame:
    out = merged.merge(q_shock, on="forecast_datetime_utc", how="left", validate="many_to_one")
    rows_without_q = out["q_shock"].isna()
    out["missing_q_shock"] = rows_without_q
    if missing_q_policy == "zero":
        out["q_shock"] = out["q_shock"].fillna(0.0)
    else:
        out = out.loc[~rows_without_q].copy()

    q = pd.to_numeric(out["q_shock"], errors="coerce")
    out["exponential_hybrid_lambda"] = EXPONENTIAL_HYBRID_LAMBDA
    out["shock_weight_exponential"] = exponential_shock_weight(q)

    for label in CONTRACT_LABELS:
        normal_price = pd.to_numeric(out[f"{label}_price_normal"], errors="coerce")
        shock_price = pd.to_numeric(out[f"{label}_price_shock"], errors="coerce")
        out[f"{label}_normal_price"] = normal_price
        out[f"{label}_shock_price"] = shock_price
        out[f"{label}_price"] = ((1.0 - q) * normal_price + q * shock_price).clip(0.0, 1.0)
        out[f"{label}_exponential_hybrid_price"] = (
            (1.0 - out["shock_weight_exponential"]) * normal_price
            + out["shock_weight_exponential"] * shock_price
        ).clip(0.0, 1.0)

    return out


def build_output_frame(hybrid: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "Event": hybrid["Event"],
            "datetime": hybrid["datetime"],
            "minute_number": hybrid["minute_number"].astype("Int64"),
            "q_shock": hybrid["q_shock"],
            "exponential_hybrid_lambda": hybrid["exponential_hybrid_lambda"],
            "shock_weight_exponential": hybrid["shock_weight_exponential"],
        }
    )
    if "refit_id" in hybrid.columns:
        output["shock_refit_id"] = hybrid["refit_id"]

    for label in CONTRACT_LABELS:
        output[f"{label}_market_ticker"] = hybrid[f"{label}_market_ticker_normal"]
        output[f"{label}_strike"] = pd.to_numeric(hybrid[f"{label}_strike_normal"], errors="coerce")
        output[f"{label}_normal_price"] = hybrid[f"{label}_normal_price"]
        output[f"{label}_shock_price"] = hybrid[f"{label}_shock_price"]
        output[f"{label}_exponential_hybrid_price"] = hybrid[f"{label}_exponential_hybrid_price"]
        output[f"{label}_price"] = hybrid[f"{label}_price"]
        if f"{label}_raw_price" in hybrid.columns:
            output[f"{label}_shock_raw_price"] = hybrid[f"{label}_raw_price"]
        if f"{label}_calibrated_price" in hybrid.columns:
            output[f"{label}_shock_calibrated_price"] = hybrid[f"{label}_calibrated_price"]

    return output.dropna(subset=[f"{label}_price" for label in CONTRACT_LABELS]).reset_index(drop=True)


def validate_contract_alignment(merged: pd.DataFrame, source_file: str) -> None:
    problems: List[str] = []
    for label in CONTRACT_LABELS:
        ticker_same = merged[f"{label}_market_ticker_normal"].astype(str) == merged[f"{label}_market_ticker_shock"].astype(str)
        strike_same = (
            pd.to_numeric(merged[f"{label}_strike_normal"], errors="coerce").round(8)
            == pd.to_numeric(merged[f"{label}_strike_shock"], errors="coerce").round(8)
        )
        if not bool((ticker_same & strike_same).all()):
            problems.append(label)
    if problems:
        raise ValueError(f"{source_file} has normal/shock contract mismatches for labels: {problems}")


def write_summary(summary: BuildSummary) -> None:
    frame = pd.DataFrame([summary.__dict__])
    for column in ["normal_source_dir", "shock_source_dir", "shock_probability_csv", "output_dir"]:
        frame[column] = frame[column].astype(str)
    metadata_dir = summary.output_dir / "_build_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metadata_dir / "build_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    clean_output_dir(output_dir)

    q_shock = load_q_shock(
        args.shock_probability_csv.resolve(),
        scenario=args.gate_scenario,
        model=args.gate_model,
        timestamp_column=args.gate_timestamp_column,
    )
    normal_files = source_files(args.normal_raw_dir.resolve())
    shock_dir = args.shock_raw_dir.resolve()

    files_written = 0
    files_with_shock_source = 0
    source_rows_seen = 0
    rows_written = 0
    rows_without_q_shock = 0
    rows_without_shock_source = 0
    q_values: List[float] = []

    for normal_path in normal_files:
        shock_path = shock_dir / normal_path.name
        if not shock_path.exists():
            normal_count = len(pd.read_csv(normal_path, usecols=["Event"]))
            rows_without_shock_source += normal_count
            continue

        files_with_shock_source += 1
        normal = read_raw(normal_path, source_name="normal")
        shock = read_raw(shock_path, source_name="shock")
        source_rows_seen += len(normal)
        merged = merge_normal_shock(normal, shock)
        validate_contract_alignment(merged, normal_path.name)
        rows_without_shock_source += max(len(normal) - len(merged), 0)

        before_q = len(merged)
        hybrid = add_hybrid_probabilities(merged, q_shock, missing_q_policy=args.missing_q_policy)
        rows_without_q_shock += before_q - len(hybrid) if args.missing_q_policy == "skip" else int(hybrid["missing_q_shock"].sum())
        if hybrid.empty:
            continue

        output = build_output_frame(hybrid)
        if output.empty:
            continue

        q_values.extend(output["q_shock"].dropna().astype(float).tolist())
        output.to_csv(output_dir / normal_path.name, index=False)
        files_written += 1
        rows_written += len(output)

    if not files_written:
        raise RuntimeError("No Model O raw files were written; check q_shock/source overlap.")

    q_array = np.array(q_values, dtype=float)
    summary = BuildSummary(
        normal_source_dir=args.normal_raw_dir.resolve(),
        shock_source_dir=shock_dir,
        shock_probability_csv=args.shock_probability_csv.resolve(),
        output_dir=output_dir,
        q_rows=int(len(q_shock)),
        normal_files=int(len(normal_files)),
        files_with_shock_source=int(files_with_shock_source),
        files_written=int(files_written),
        source_rows_seen=int(source_rows_seen),
        rows_written=int(rows_written),
        rows_without_q_shock=int(rows_without_q_shock),
        rows_without_shock_source=int(rows_without_shock_source),
        min_q_shock=float(q_array.min()),
        max_q_shock=float(q_array.max()),
        mean_q_shock=float(q_array.mean()),
    )
    write_summary(summary)

    print(f"Model O q_shock source: {args.shock_probability_csv.resolve()}")
    print(f"Model O output directory: {output_dir}")
    print(f"Files written: {files_written:,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Rows skipped without q_shock: {rows_without_q_shock:,}")
    print(f"q_shock range: {summary.min_q_shock:.6f} to {summary.max_q_shock:.6f}")


if __name__ == "__main__":
    main()
