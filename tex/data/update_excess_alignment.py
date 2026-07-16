#!/usr/bin/env python3
"""
Add excess MAP-alignment statistics to the curated LSYNTH summary CSV.

Definition
----------
For generator g,

    Lambda_g =
        (mean(Upsilon_g) - mean(Upsilon_independent))
        ------------------------------------------------
        (mean(Upsilon_control) - mean(Upsilon_independent))

The bootstrap resamples GSS waves jointly, so all generators from a selected
wave remain paired within each bootstrap replicate.

Example
-------
python update_excess_alignment.py \
    --wave-csv curated_wave_level_results.csv \
    --summary-csv curated_cross_wave_summary.csv \
    --output-csv curated_cross_wave_summary_with_excess.csv \
    --split-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_KEY = "baseline"
CONTROL_KEY = "original_sample_control"

REQUIRED_WAVE_COLUMNS = {
    "year",
    "generator_key",
    "upsilon_mean",
}

REQUIRED_SUMMARY_COLUMNS = {
    "generator_key",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and append excess MAP-alignment statistics."
    )
    parser.add_argument(
        "--wave-csv",
        type=Path,
        required=True,
        help="Wave-level CSV containing year, generator_key, and upsilon_mean.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Existing cross-wave summary CSV to update.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output summary CSV. If omitted, the input summary CSV is overwritten."
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory in which summary_<generator_key>.csv files "
            "will also be written."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=20_000,
        help="Number of paired wave-bootstrap replicates. Default: 20000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260715,
        help="Random seed. Default: 20260715.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Central confidence interval width. Default: 0.95.",
    )
    return parser.parse_args()


def validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{label} is missing required columns: {missing_text}")


def compute_excess_alignment(
    wave_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    ci: float,
) -> pd.DataFrame:
    if n_boot < 1:
        raise ValueError("--bootstrap-replicates must be at least 1.")
    if not 0.0 < ci < 1.0:
        raise ValueError("--ci must lie strictly between 0 and 1.")

    # One row per wave and generator.
    duplicated = wave_df.duplicated(["year", "generator_key"])
    if duplicated.any():
        bad = wave_df.loc[duplicated, ["year", "generator_key"]]
        raise ValueError(
            "Wave CSV contains duplicate year/generator rows:\n"
            f"{bad.to_string(index=False)}"
        )

    pivot = (
        wave_df.pivot(
            index="year",
            columns="generator_key",
            values="upsilon_mean",
        )
        .sort_index()
    )

    for key in (BASELINE_KEY, CONTROL_KEY):
        if key not in pivot.columns:
            raise ValueError(f"Required generator_key '{key}' was not found.")

    if pivot.isna().any().any():
        missing = pivot.columns[pivot.isna().any()].tolist()
        raise ValueError(
            "Every bootstrap wave must contain every generator. "
            f"Missing values found for: {missing}"
        )

    years = pivot.index.to_numpy()
    values = pivot.to_numpy(dtype=float)
    keys = pivot.columns.tolist()
    key_to_col = {key: idx for idx, key in enumerate(keys)}

    baseline_col = key_to_col[BASELINE_KEY]
    control_col = key_to_col[CONTROL_KEY]

    means = values.mean(axis=0)
    denominator = means[control_col] - means[baseline_col]
    if np.isclose(denominator, 0.0):
        raise ZeroDivisionError(
            "The mean control-minus-baseline MAP-alignment difference is zero."
        )

    lambda_hat = (means - means[baseline_col]) / denominator

    # Paired bootstrap over waves.
    rng = np.random.default_rng(seed)
    n_waves = len(years)
    draw_idx = rng.integers(
        low=0,
        high=n_waves,
        size=(n_boot, n_waves),
    )

    # Shape: bootstrap replicate x generator.
    boot_means = values[draw_idx, :].mean(axis=1)
    boot_denominator = (
        boot_means[:, control_col] - boot_means[:, baseline_col]
    )

    valid = ~np.isclose(boot_denominator, 0.0)
    valid_fraction = valid.mean()
    if valid_fraction < 0.99:
        raise RuntimeError(
            "Too many bootstrap replicates have a near-zero "
            f"control-minus-baseline denominator ({valid_fraction:.3%} valid)."
        )

    baseline_boot_mean = boot_means[valid, baseline_col][:, None]
    boot_lambda = (
        boot_means[valid, :] - baseline_boot_mean
    ) / boot_denominator[valid, None]

    alpha = 1.0 - ci
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    ci_low = np.quantile(boot_lambda, lower_q, axis=0)
    ci_high = np.quantile(boot_lambda, upper_q, axis=0)

    # Preserve exact anchors by definition.
    lambda_hat[baseline_col] = 0.0
    lambda_hat[control_col] = 1.0
    ci_low[baseline_col] = 0.0
    ci_high[baseline_col] = 0.0
    ci_low[control_col] = 1.0
    ci_high[control_col] = 1.0

    return pd.DataFrame(
        {
            "generator_key": keys,
            "mean_excess_alignment": lambda_hat,
            "excess_alignment_ci_low": ci_low,
            "excess_alignment_ci_high": ci_high,
        }
    )


def main() -> None:
    args = parse_args()

    wave_df = pd.read_csv(args.wave_csv)
    summary_df = pd.read_csv(args.summary_csv)

    validate_columns(wave_df, REQUIRED_WAVE_COLUMNS, "Wave CSV")
    validate_columns(summary_df, REQUIRED_SUMMARY_COLUMNS, "Summary CSV")

    excess_df = compute_excess_alignment(
        wave_df=wave_df,
        n_boot=args.bootstrap_replicates,
        seed=args.seed,
        ci=args.ci,
    )

    # Remove prior versions of these columns so rerunning is idempotent.
    added_columns = [
        "mean_excess_alignment",
        "excess_alignment_ci_low",
        "excess_alignment_ci_high",
    ]
    summary_df = summary_df.drop(
        columns=[c for c in added_columns if c in summary_df.columns]
    )

    updated = summary_df.merge(
        excess_df,
        on="generator_key",
        how="left",
        validate="one_to_one",
    )

    if updated[added_columns].isna().any().any():
        missing_keys = updated.loc[
            updated[added_columns].isna().any(axis=1),
            "generator_key",
        ].tolist()
        raise ValueError(
            "No excess-alignment result was produced for generator keys: "
            + ", ".join(missing_keys)
        )

    output_csv = args.output_csv or args.summary_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_csv, index=False)

    if args.split_dir is not None:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        for _, row in updated.iterrows():
            key = str(row["generator_key"])
            row.to_frame().T.to_csv(
                args.split_dir / f"summary_{key}.csv",
                index=False,
            )

    display_cols = [
        "generator_key",
        "mean_excess_alignment",
        "excess_alignment_ci_low",
        "excess_alignment_ci_high",
    ]
    print(f"Wrote: {output_csv}")
    if args.split_dir is not None:
        print(f"Wrote split summary files to: {args.split_dir}")
    print()
    print(updated[display_cols].to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
