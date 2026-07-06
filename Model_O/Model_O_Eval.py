from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_R import Model_R_Eval as base


ASSUMPTIONS = (
    "Model O forecast values are interpreted as YES probabilities. CSV values greater than 1 and no greater than 100 "
    "are treated as cents and divided by 100 before scoring.",
    "Model O uses p_final = (1 - q_shock) * p_normal + q_shock * p_shock.",
    "Model O also carries p_exponential_hybrid with lambda=2 as a downstream strategy probability: "
    "w(q) = (exp(2*q) - 1) / (exp(2) - 1), p_exp = (1 - w(q)) * p_normal + w(q) * p_shock.",
    "p_normal is Model R, which was verified to match Model A's architecture except for the 20,000 Monte Carlo path default.",
    "p_shock is the full Model H fair-value source used by Model M before Model M's hard gate selection.",
    "q_shock is the continuous y_prob value from the cloned Model M Additional_Data/all_test_predictions.csv, "
    "filtered to scenario=drop_stale and model=logistic_regression by default.",
    "Scored event outcome is the official Kalshi contract result: 1 for YES resolved, 0 for NO resolved.",
    "Forecast rows are joined to outcomes by exact Kalshi market ticker when available; legacy price files "
    "without market tickers fall back to event_ticker + contract_label + strike.",
    "Event tickers are interpreted as the contract settlement/end hour, matching the price fetcher's "
    "hour_end_utc -> New York event ticker convention.",
    "Binance prices are diagnostic only. They may be used for strike matching upstream and audit mismatch checks, "
    "but never as the scored truth label.",
)

CONTRACT_LABELS = ("ATM", "OTM+1", "OTM-1")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Evaluate Model O hybrid BTC hourly contract probabilities against official Kalshi outcomes."
    )
    parser.add_argument(
        "--kalshi-price-dir",
        type=Path,
        default=root / "Model_O" / "Model_O_Output_Raw_Data",
        help="Folder containing Model_O.py hourly raw forecast CSV outputs.",
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=base.default_settlement_csv(root),
        help="Settlement CSV with official Kalshi result/outcome columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model_O_Evals_Outputs",
        help="Directory for Model O evaluation output files.",
    )
    parser.add_argument("--skip-binance-audit", action="store_true")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model_o_price_outputs(price_dir: Path) -> pd.DataFrame:
    forecasts = base.load_kalshi_price_outputs(price_dir)
    files = sorted(price_dir.glob("*.csv"))
    diagnostic_rows = []

    for path in files:
        wide = pd.read_csv(path)
        if "q_shock" not in wide.columns:
            continue
        wide["forecast_datetime_utc"] = pd.to_datetime(wide["datetime"], utc=True)
        for label in CONTRACT_LABELS:
            required = {
                f"{label}_market_ticker",
                f"{label}_normal_price",
                f"{label}_shock_price",
            }
            if not required.issubset(wide.columns):
                continue
            columns = [
                "Event",
                "forecast_datetime_utc",
                "q_shock",
                f"{label}_market_ticker",
                f"{label}_normal_price",
                f"{label}_shock_price",
            ]
            for optional in ["shock_weight_exponential", f"{label}_exponential_hybrid_price"]:
                if optional in wide.columns:
                    columns.append(optional)
            part = wide[columns].copy()
            part = part.rename(
                columns={
                    "Event": "event_ticker",
                    f"{label}_market_ticker": "market_ticker",
                    f"{label}_normal_price": "p_normal",
                    f"{label}_shock_price": "p_shock",
                    f"{label}_exponential_hybrid_price": "p_exponential_hybrid",
                }
            )
            part["contract_label"] = label
            diagnostic_rows.append(part)

    if not diagnostic_rows:
        return forecasts

    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    diagnostics["market_ticker"] = diagnostics["market_ticker"].astype("string").str.strip()
    for column in ["q_shock", "shock_weight_exponential", "p_normal", "p_shock", "p_exponential_hybrid"]:
        if column in diagnostics.columns:
            diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce")

    merge_cols = ["event_ticker", "forecast_datetime_utc", "market_ticker", "contract_label"]
    return forecasts.merge(diagnostics, on=merge_cols, how="left", validate="one_to_one")


def write_individual_metric_files(
    *,
    output_dir: Path,
    raw: pd.DataFrame,
    metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    calibration: pd.DataFrame,
    expanded_error: pd.DataFrame,
    expanded_error_summary: pd.DataFrame,
    sharpness: pd.DataFrame,
    coverage: pd.DataFrame,
    unmatched: pd.DataFrame,
    resolution_mismatches: pd.DataFrame,
    time_bucket_metrics: pd.DataFrame,
    time_bucket_accuracy: pd.DataFrame,
    time_bucket_brier: pd.DataFrame,
    time_bucket_calibration: pd.DataFrame,
) -> None:
    raw.to_csv(output_dir / "raw_values.csv", index=False)
    coverage.to_csv(output_dir / "outcome_join_coverage.csv", index=False)
    unmatched.to_csv(output_dir / "unmatched_forecast_rows.csv", index=False)
    resolution_mismatches.to_csv(output_dir / "resolution_mismatches.csv", index=False)
    time_bucket_metrics.to_csv(output_dir / "time_bucket_metrics.csv", index=False)
    time_bucket_accuracy.to_csv(output_dir / "time_bucket_accuracy.csv", index=False)
    time_bucket_brier.to_csv(output_dir / "time_bucket_brier_decomposition.csv", index=False)
    time_bucket_calibration.to_csv(output_dir / "time_bucket_calibration_curve.csv", index=False)
    metrics[["segment", "n_forecasts", "n_event_contracts", "brier_score"]].to_csv(
        output_dir / "brier_score.csv", index=False
    )
    decomposition.to_csv(output_dir / "brier_decomposition.csv", index=False)
    metrics[["segment", "n_forecasts", "n_event_contracts", "log_loss"]].to_csv(
        output_dir / "log_loss.csv", index=False
    )
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)
    expanded_error.to_csv(output_dir / "expanded_calibration_error.csv", index=False)
    expanded_error_summary.to_csv(output_dir / "expanded_calibration_error_summary.csv", index=False)
    sharpness.to_csv(output_dir / "sharpness.csv", index=False)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "model_o_assumptions.txt").write_text("\n".join(ASSUMPTIONS) + "\n", encoding="utf-8")


def model_o_html(html_text: str) -> str:
    replacements = {
        "Model R": "Model O",
        "model_R_Evals_Outputs": "model_O_Evals_Outputs",
        "model_r_assumptions.txt": "model_o_assumptions.txt",
        "Model_R": "Model_O",
    }
    out = html_text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_model_o_price_outputs(args.kalshi_price_dir.resolve())
    settlements = base.load_settlements(args.settlement_csv.resolve())
    outcomes, messages = base.build_kalshi_reality_outcomes(
        forecasts=forecasts,
        settlements=settlements,
        output_dir=output_dir,
    )
    raw, unmatched = base.attach_outcomes(forecasts, outcomes)
    coverage = base.build_outcome_join_coverage(forecasts, raw, unmatched)
    resolution_mismatches = pd.DataFrame() if args.skip_binance_audit else base.build_resolution_mismatches(raw)

    overall_coverage = coverage.loc[coverage["scope"] == "overall"].iloc[0]
    messages.append(
        f"Matched {int(overall_coverage['matched_rows'])} of "
        f"{int(overall_coverage['total_forecast_rows'])} forecast rows to official Kalshi outcomes."
    )
    if len(resolution_mismatches):
        messages.append(f"Found {len(resolution_mismatches)} Binance audit mismatch row(s); see resolution_mismatches.csv.")

    metrics = base.build_metrics_summary(raw, threshold=args.classification_threshold)
    decomposition = base.build_brier_decomposition(raw, bins=args.calibration_bins)
    calibration, expanded_summary = base.expanded_calibration_error(raw, bins=args.calibration_bins)
    sharpness = base.build_sharpness(raw)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = base.build_time_bucket_outputs(
        raw,
        calibration_bins=args.calibration_bins,
        threshold=args.classification_threshold,
    )

    write_individual_metric_files(
        output_dir=output_dir,
        raw=raw,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_error=calibration,
        expanded_error_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        unmatched=unmatched,
        resolution_mismatches=resolution_mismatches,
        time_bucket_metrics=time_bucket_metrics,
        time_bucket_accuracy=time_bucket_accuracy,
        time_bucket_brier=time_bucket_brier,
        time_bucket_calibration=time_bucket_calibration,
    )

    original_assumptions: Any = base.ASSUMPTIONS
    try:
        base.ASSUMPTIONS = ASSUMPTIONS
        summary_html = base.build_summary_html(
            raw=raw,
            metrics=metrics,
            decomposition=decomposition,
            calibration=calibration,
            expanded_summary=expanded_summary,
            sharpness=sharpness,
            coverage=coverage,
            resolution_mismatches=resolution_mismatches,
            messages=messages,
            include_coverage=False,
        )
        summary_with_coverage_html = base.build_summary_html(
            raw=raw,
            metrics=metrics,
            decomposition=decomposition,
            calibration=calibration,
            expanded_summary=expanded_summary,
            sharpness=sharpness,
            coverage=coverage,
            resolution_mismatches=resolution_mismatches,
            messages=messages,
            include_coverage=True,
        )
    finally:
        base.ASSUMPTIONS = original_assumptions

    (output_dir / "model_o_summary.html").write_text(model_o_html(summary_html), encoding="utf-8")
    (output_dir / "model_o_summary_with_coverage.html").write_text(
        model_o_html(summary_with_coverage_html),
        encoding="utf-8",
    )
    (output_dir / "time_bucket_summary.html").write_text(
        model_o_html(
            base.build_time_bucket_summary_html(
                time_bucket_metrics=time_bucket_metrics,
                time_bucket_accuracy=time_bucket_accuracy,
                time_bucket_brier=time_bucket_brier,
                time_bucket_calibration=time_bucket_calibration,
            )
        ),
        encoding="utf-8",
    )

    print(f"Model O evaluation output directory: {output_dir}")
    print(f"Scored rows: {len(raw):,}")
    print(f"Unmatched forecast rows: {len(unmatched):,}")
    print(f"Resolution mismatches: {len(resolution_mismatches):,}")


if __name__ == "__main__":
    main()
