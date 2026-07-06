from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_R import Model_R_Stats_Dashboard as base


OUTPUT_FOLDER_NAME = "Model_O_Stats_Dashboard_Outputs"


def default_model_o_eval_dir(root: Path) -> Path:
    return root / "Model_O" / "model_O_Evals_Outputs"


def default_model_k_eval_dir(root: Path) -> Path:
    return root / "Model_K" / "Model_K_outputs"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build internal and comparative dashboards for Model O using evaluation output folders."
    )
    parser.add_argument("--model-o-eval-dir", type=Path, default=default_model_o_eval_dir(base.REPO_ROOT))
    parser.add_argument("--model-k-eval-dir", type=Path, default=default_model_k_eval_dir(base.REPO_ROOT))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--pit-seed", type=int, default=42)
    return parser.parse_args()


def model_o_html(html_text: str) -> str:
    replacements = {
        "Model R": "Model O",
        "model_r": "model_o",
        "Model_R": "Model_O",
        " R ": " O ",
        "_r_": "_o_",
    }
    out = html_text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_o_eval_dir = args.model_o_eval_dir.resolve()
    model_k_eval_dir = args.model_k_eval_dir.resolve()

    loaded_n = base.load_eval_dir(model_o_eval_dir)
    loaded_k = base.load_eval_dir(model_k_eval_dir)

    bundle_n_internal = base.build_bundle_from_raw(
        loaded_n["raw"],
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
        coverage=loaded_n["coverage"],
        mismatches=loaded_n["mismatches"],
    )
    pit_values = base.compute_randomized_pit(bundle_n_internal["raw"], seed=args.pit_seed)
    internal_notes = [
        f"Source directory: {model_o_eval_dir}",
        f"Randomized PIT seed: {args.pit_seed}",
        "Internal dashboard uses the full Model O scored universe currently present in raw_values.csv.",
    ]
    internal_html = base.build_internal_dashboard_html(bundle_n_internal, pit_values, internal_notes)
    (output_dir / "model_o_internal_dashboard.html").write_text(model_o_html(internal_html), encoding="utf-8")

    aligned_n_raw, aligned_k_raw = base.intersect_raw_rows(loaded_n["raw"], loaded_k["raw"])
    if aligned_n_raw.empty or aligned_k_raw.empty:
        raise RuntimeError("No overlapping scored rows were found between Model O and Model K.")

    bundle_n_compare = base.build_bundle_from_raw(
        aligned_n_raw,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
    )
    bundle_k_compare = base.build_bundle_from_raw(
        aligned_k_raw,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
    )
    comparative_notes = [
        f"Model O source directory: {model_o_eval_dir}",
        f"Model K source directory: {model_k_eval_dir}",
        "Comparative dashboard restricts both models to the exact overlapping scored-row universe using event_contract_id + forecast_datetime_utc.",
        f"Overlap rows: {len(aligned_n_raw):,}; overlap event contracts: {aligned_n_raw['event_contract_id'].nunique():,}.",
    ]
    comparative_html = base.build_comparative_dashboard_html(bundle_n_compare, bundle_k_compare, comparative_notes)
    (output_dir / "model_o_vs_model_k_dashboard.html").write_text(model_o_html(comparative_html), encoding="utf-8")

    print(f"Internal dashboard: {output_dir / 'model_o_internal_dashboard.html'}")
    print(f"Comparative dashboard: {output_dir / 'model_o_vs_model_k_dashboard.html'}")


if __name__ == "__main__":
    main()
