from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20)
RAW_OUTPUT_FOLDER_NAME = "Model_M_Output_Raw_Data"
GATE_SCENARIO = "drop_stale"
GATE_MODEL = "logistic_regression"
GATE_TIMESTAMP_COLUMN = "target_minute"


@dataclass(frozen=True)
class GateRun:
    threshold: float
    slug: str
    raw_output_dir: Path
    selected_rows: int
    selected_unique_minutes: int
    matched_source_rows: int
    unmatched_gate_minutes: int
    matched_events: int
    written_files: int


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def threshold_slug(threshold: float) -> str:
    return f"p_gt_{threshold:.2f}".replace(".", "_")


def parse_thresholds(values: Iterable[str]) -> List[float]:
    thresholds = [float(value) for value in values]
    if not thresholds:
        raise ValueError("At least one gate threshold is required.")
    bad = [value for value in thresholds if value < 0.0 or value > 1.0]
    if bad:
        raise ValueError(f"Gate thresholds must be probabilities in [0, 1]. Bad values: {bad}")
    return thresholds


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build Model M raw forecast outputs by taking Model H GARCH-t/isotonic forecast rows "
            "only at minutes admitted by the Model M logistic-regression drop-stale gate."
        )
    )
    parser.add_argument(
        "--gate-predictions-csv",
        type=Path,
        default=root / "Model_M" / "Additional_Data" / "all_test_predictions.csv",
        help="CSV containing Model M gate probabilities.",
    )
    parser.add_argument(
        "--source-model-h-raw-dir",
        type=Path,
        default=root / "Model_H" / "Model_H_Output_Raw_Data",
        help="Model H raw forecast directory used as the GARCH-t/isotonic forecast source.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / RAW_OUTPUT_FOLDER_NAME,
        help="Root directory where threshold-specific Model M raw forecast folders are written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[str(value) for value in DEFAULT_THRESHOLDS],
        help="Gate probability thresholds. Defaults to 0.10 0.15 0.20.",
    )
    parser.add_argument(
        "--gate-scenario",
        default=GATE_SCENARIO,
        help="Gate CSV scenario value to use.",
    )
    parser.add_argument(
        "--gate-model",
        default=GATE_MODEL,
        help="Gate CSV model value to use.",
    )
    parser.add_argument(
        "--gate-timestamp-column",
        choices=("minute", "target_minute"),
        default=GATE_TIMESTAMP_COLUMN,
        help=(
            "Timestamp column used as the admitted trading candle. The default target_minute "
            "uses the candle the logistic row is predicting."
        ),
    )
    parser.add_argument(
        "--keep-empty-files",
        action="store_true",
        help="Write empty per-event CSVs when a source Model H file has no selected minutes.",
    )
    return parser.parse_args()


def load_gate_rows(
    path: Path,
    *,
    scenario: str,
    model: str,
    timestamp_column: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Gate predictions CSV not found: {path}")
    gate = pd.read_csv(path)
    required = {"scenario", "model", "fold", "minute", "target_minute", "y_prob"}
    missing = required - set(gate.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    gate = gate[(gate["scenario"].astype(str) == scenario) & (gate["model"].astype(str) == model)].copy()
    if gate.empty:
        raise RuntimeError(f"No gate rows matched scenario={scenario!r}, model={model!r}.")

    gate["minute"] = pd.to_datetime(gate["minute"], utc=True)
    gate["target_minute"] = pd.to_datetime(gate["target_minute"], utc=True)
    gate["gate_timestamp_utc"] = pd.to_datetime(gate[timestamp_column], utc=True)
    gate["y_prob"] = pd.to_numeric(gate["y_prob"], errors="coerce")
    gate = gate.dropna(subset=["gate_timestamp_utc", "y_prob"]).copy()
    if gate.empty:
        raise RuntimeError("No usable gate rows remained after timestamp/probability parsing.")

    gate = gate.sort_values(["gate_timestamp_utc", "y_prob"], ascending=[True, False])
    gate = gate.drop_duplicates(subset=["gate_timestamp_utc"], keep="first").reset_index(drop=True)
    return gate


def source_files(source_raw_dir: Path) -> List[Path]:
    if not source_raw_dir.exists():
        raise FileNotFoundError(f"Source Model H raw directory not found: {source_raw_dir}")
    files = sorted(source_raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No source Model H raw CSV files found in {source_raw_dir}")
    return files


def clean_threshold_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()


def select_source_rows(
    files: List[Path],
    selected_minutes: set[pd.Timestamp],
    output_dir: Path,
    *,
    keep_empty_files: bool,
) -> Dict[str, Any]:
    matched_source_rows = 0
    matched_events: set[str] = set()
    matched_timestamps: set[pd.Timestamp] = set()
    written_files = 0

    for path in files:
        frame = pd.read_csv(path)
        if "datetime" not in frame.columns or "Event" not in frame.columns:
            raise ValueError(f"{path} is missing required Model H raw columns.")
        parsed_datetime = pd.to_datetime(frame["datetime"], utc=True)
        matched_mask = parsed_datetime.isin(selected_minutes)
        selected = frame[matched_mask].copy()
        if selected.empty and not keep_empty_files:
            continue
        selected.to_csv(output_dir / path.name, index=False)
        written_files += 1
        if not selected.empty:
            matched_source_rows += len(selected)
            matched_events.update(selected["Event"].astype(str).dropna().unique().tolist())
            matched_timestamps.update(parsed_datetime[matched_mask].tolist())

    return {
        "matched_source_rows": matched_source_rows,
        "matched_events": len(matched_events),
        "matched_timestamps": matched_timestamps,
        "written_files": written_files,
    }


def write_gate_outputs(
    gate_rows: pd.DataFrame,
    files: List[Path],
    output_root: Path,
    *,
    threshold: float,
    keep_empty_files: bool,
) -> GateRun:
    slug = threshold_slug(threshold)
    output_dir = output_root / slug
    metadata_dir = output_root / "_gate_metadata"
    clean_threshold_dir(output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    selected_gate = gate_rows[gate_rows["y_prob"] > threshold].copy()
    selected_gate = selected_gate.sort_values("gate_timestamp_utc").reset_index(drop=True)

    selected_minutes = set(selected_gate["gate_timestamp_utc"].tolist())
    match_counts = select_source_rows(
        files,
        selected_minutes,
        output_dir,
        keep_empty_files=keep_empty_files,
    )
    matched_timestamps = match_counts["matched_timestamps"]
    selected_gate["matched_model_h_source"] = selected_gate["gate_timestamp_utc"].isin(matched_timestamps)
    selected_gate.to_csv(metadata_dir / f"{slug}_gate_selected_minutes.csv", index=False)
    selected_gate.loc[~selected_gate["matched_model_h_source"]].to_csv(
        metadata_dir / f"{slug}_gate_unmatched_minutes.csv",
        index=False,
    )

    return GateRun(
        threshold=threshold,
        slug=slug,
        raw_output_dir=output_dir,
        selected_rows=int(len(selected_gate)),
        selected_unique_minutes=int(selected_gate["gate_timestamp_utc"].nunique()),
        matched_source_rows=int(match_counts["matched_source_rows"]),
        unmatched_gate_minutes=int((~selected_gate["matched_model_h_source"]).sum()),
        matched_events=int(match_counts["matched_events"]),
        written_files=int(match_counts["written_files"]),
    )


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    gate_rows = load_gate_rows(
        args.gate_predictions_csv.resolve(),
        scenario=args.gate_scenario,
        model=args.gate_model,
        timestamp_column=args.gate_timestamp_column,
    )
    files = source_files(args.source_model_h_raw_dir.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs = [
        write_gate_outputs(
            gate_rows,
            files,
            output_root,
            threshold=threshold,
            keep_empty_files=args.keep_empty_files,
        )
        for threshold in thresholds
    ]
    summary = pd.DataFrame([run.__dict__ for run in runs])
    summary["raw_output_dir"] = summary["raw_output_dir"].astype(str)
    summary.to_csv(output_root / "gate_output_summary.csv", index=False)

    print(f"Model M gate source: {args.gate_predictions_csv.resolve()}")
    print(f"Gate rows after scenario/model filtering: {len(gate_rows):,}")
    print(f"Gate timestamp column: {args.gate_timestamp_column}")
    print(f"Model M raw output root: {output_root}")
    for run in runs:
        print(
            f"{run.slug}: selected_minutes={run.selected_unique_minutes:,}, "
            f"matched_source_rows={run.matched_source_rows:,}, "
            f"unmatched_gate_minutes={run.unmatched_gate_minutes:,}, "
            f"events={run.matched_events:,}, files={run.written_files:,}"
        )


if __name__ == "__main__":
    main()
