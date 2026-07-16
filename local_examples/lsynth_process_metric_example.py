#!/usr/bin/env python3
"""
Generate one-wave LSYNTH comparators and estimate the empirical process metric d_mu.

This script deliberately reuses the generator functions from the existing
``lsynth_eval_comparators_fast_chow.py`` script so that LSM, CTGAN, the
independent baseline, the restricted Chow--Liu hybrid, and the original
(control) are generated with the same implementation and argument conventions
as the current benchmark.

The procedure is:

1. Load ``gss_<year>.csv`` and the pretrained ``gss_<year>.pkl.gz`` LSM.
2. Split the available real records into a generator-source partition and a
   common reference partition E_mu.
3. Generate each comparator using only the generator-source partition, except
   LSM, which uses the pretrained wave-specific model.
4. Fit the same Qnet conditional learner to an equal number of rows from each
   generated dataset.
5. Evaluate every fitted conditional profile on the same reference rows and
   estimate

       d_mu(P_a, P_b) = mean_{x in E_mu, i} |u_a(x, i) - u_b(x, i)|.

6. Save pairwise distances, bootstrap intervals, row-level distances, and
   coordinate-level contributions. The latter identify variables contributing
   most to each generator's distance from the original (control).

Example
-------
python lsynth_process_metric_example.py 2018 \
    --generator-script ./lsynth_eval_comparators_fast_chow.py \
    --datasets-dir ./datasets \
    --output-dir process_metric_outputs \
    --num-rows 1000 \
    --metric-train-rows 1000 \
    --reference-fraction 0.50 \
    --reference-rows 500 \
    --n-workers 120 \
    --profile-workers 8 \
    --run-chow-liu

The output directory contains:

    synthetic_data/process_metric_synthetic_<generator>_<year>.csv
    models/process_metric_qnet_<generator>_<year>.joblib
    process_metric_pairwise_<year>.csv
    process_metric_matrix_<year>.csv
    process_metric_to_original_control_<year>.csv
    process_metric_row_distances_<year>.csv
    process_metric_coordinate_contributions_<year>.csv
    process_metric_reference_rows_<year>.csv
    process_metric_generator_source_rows_<year>.csv
    process_metric_metadata_<year>.json

Notes
-----
* The common reference sample is not used to generate the comparator datasets
  or fit their Qnet profile models.
* Every profile model uses the same learner, hyperparameters, feature set, and
  number of training rows.
* Bootstrap intervals condition on the fitted models and resample the common
  reference rows. They quantify evaluation-sample variation for this worked
  example, not uncertainty from refitting the profile models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

# Set thread defaults before numerical libraries are imported.
N_WORKERS_DEFAULT = 120
CTGAN_WORKERS_DEFAULT = 10
CHOW_LIU_TRAIN_ROWS_DEFAULT = 5000
CHOW_LIU_MAX_TREE_FEATURES_DEFAULT = 150
CHOW_LIU_MAX_CARDINALITY_DEFAULT = 75

os.environ.setdefault("OMP_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("MKL_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_WORKERS_DEFAULT))

import joblib
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "lsm": "LSM",
    "chow_liu": "Chow--Liu hybrid",
    "ctgan": "CTGAN",
    "baseline": "Independent baseline",
    "original_sample_control": "Original (control)",
}

CONTROL_KEY_DEFAULT = "original_sample_control"
MISSING_TOKEN_DEFAULT = "__MISSING__"


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_rows: int


def log(message: str) -> None:
    print(message, flush=True)


def slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    return text or "unnamed"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_generator_module(script_path: Path):
    """Load the existing benchmark script as a module without running main()."""
    if not script_path.exists():
        raise FileNotFoundError(f"Generator script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("lsynth_existing_generator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "get_lsm_feature_names",
        "restrict_to_lsm_features",
        "generate_lsm",
        "generate_baseline",
        "generate_ctgan",
        "generate_chow_liu",
        "generate_original_sample_control",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "The generator script is missing required functions: " + ", ".join(missing)
        )
    return module


def import_qnet_class():
    try:
        from quasinet.qnet import Qnet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import quasinet. Install the environment used to train the "
            "wave-specific LSMs before running this script."
        ) from exc
    return Qnet


def parse_max_feats(value: str) -> str | int:
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def normalize_categorical_frame(
    df: pd.DataFrame,
    *,
    missing_token: str,
) -> pd.DataFrame:
    """Convert all values to string tokens while retaining a missing category."""
    out = df.copy()
    out = out.replace("", pd.NA)
    out = out.astype("string").fillna(missing_token)
    return out.astype(str)


def choose_metric_features(
    model_feature_names: list[str],
    *,
    feature_list_file: Path | None,
    feature_limit: int,
) -> list[str]:
    if feature_list_file is not None:
        if not feature_list_file.exists():
            raise FileNotFoundError(f"Feature-list file not found: {feature_list_file}")
        requested = [
            line.strip()
            for line in feature_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = [name for name in requested if name not in model_feature_names]
        if missing:
            raise ValueError(
                "Feature-list entries absent from model.feature_names: "
                + ", ".join(missing[:20])
            )
        features = requested
    else:
        features = list(model_feature_names)

    if feature_limit > 0:
        features = features[:feature_limit]

    if not features:
        raise ValueError("The selected metric feature set is empty.")
    return features


def split_generator_source_and_reference(
    orig_df: pd.DataFrame,
    *,
    reference_fraction: float,
    reference_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("--reference-fraction must lie strictly between 0 and 1.")

    shuffled = orig_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_reference_pool = max(1, int(round(reference_fraction * len(shuffled))))
    n_reference_pool = min(n_reference_pool, len(shuffled) - 1)

    reference_pool = shuffled.iloc[:n_reference_pool].reset_index(drop=True)
    generator_source = shuffled.iloc[n_reference_pool:].reset_index(drop=True)

    if reference_rows > 0:
        n_eval = min(reference_rows, len(reference_pool))
        reference = reference_pool.sample(
            n=n_eval,
            replace=False,
            random_state=random_state + 1,
        ).reset_index(drop=True)
    else:
        reference = reference_pool

    if generator_source.empty or reference.empty:
        raise ValueError("The source/reference split produced an empty partition.")

    return generator_source, reference


def sample_training_rows(
    df: pd.DataFrame,
    *,
    n_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if n_rows <= 0 or n_rows >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, replace=False, random_state=random_state).reset_index(drop=True)


def fit_profile_qnet(
    *,
    df: pd.DataFrame,
    feature_names: list[str],
    n_workers: int,
    min_samples_split: int,
    alpha: float,
    max_depth: int,
    max_feats: str | int,
    early_stopping: bool,
    random_state: int,
    verbose: int,
):
    Qnet = import_qnet_class()
    X = df.loc[:, feature_names].to_numpy(dtype=str)
    model = Qnet(
        feature_names=feature_names,
        min_samples_split=min_samples_split,
        alpha=alpha,
        max_depth=max_depth,
        max_feats=max_feats,
        early_stopping=early_stopping,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_workers,
    )
    model.fit(X)
    return model


def normalized_profile_for_row(
    model: Any,
    row: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate the normalized conditional profile u_G(x,i).

    Unlike ``average_fidelity``, an observed value absent from a predicted
    distribution is assigned probability zero, matching the population
    definition used in the manuscript.
    """
    distributions = model.predict_distributions(row)
    profile = np.full(len(row), np.nan, dtype=float)

    for i, (distribution, observed) in enumerate(zip(distributions, row)):
        if distribution is None:
            continue
        maximum = max(distribution.values(), default=0.0)
        if maximum <= eps:
            profile[i] = 0.0
            continue
        observed_probability = float(distribution.get(observed, 0.0))
        profile[i] = observed_probability / maximum

    return profile


def evaluate_profiles(
    *,
    model: Any,
    reference_df: pd.DataFrame,
    profile_workers: int,
) -> np.ndarray:
    X = reference_df.to_numpy(dtype=str)

    # Populate the model's internal node map before any threaded predictions.
    if len(X) > 0:
        normalized_profile_for_row(model, X[0])

    if profile_workers <= 1:
        rows = [normalized_profile_for_row(model, row) for row in X]
    else:
        with ThreadPoolExecutor(max_workers=profile_workers) as executor:
            rows = list(executor.map(lambda row: normalized_profile_for_row(model, row), X))

    profiles = np.vstack(rows)
    if profiles.shape != X.shape:
        raise RuntimeError(
            f"Profile matrix has shape {profiles.shape}; expected {X.shape}."
        )
    return profiles


def bootstrap_mean(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    ci: float,
    random_state: int,
) -> BootstrapResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values were available for bootstrap estimation.")
    if n_bootstrap < 1:
        raise ValueError("--bootstrap-replicates must be at least 1.")
    if not 0.0 < ci < 1.0:
        raise ValueError("--ci must lie strictly between 0 and 1.")

    estimate = float(values.mean())
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - ci
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        n_rows=int(len(values)),
    )


def profile_distance_by_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Profile shapes differ: {a.shape} versus {b.shape}")
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(a - b), axis=1)


def coordinate_contributions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Profile shapes differ: {a.shape} versus {b.shape}")
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(a - b), axis=0)


def save_latex_table(to_control: pd.DataFrame, path: Path, year: int) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Empirical $d_\mu$ from each generator to the original (control) for the {year} GSS wave.}}",
        r"\label{tab:process_metric_example}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Generator & $\widehat d_\mu$ & 95\% bootstrap CI \\",
        r"\midrule",
    ]
    for row in to_control.itertuples(index=False):
        label = str(row.generator_label).replace("--", r"--")
        lines.append(
            f"{label} & {row.d_mu_hat:.4f} & "
            f"$[{row.ci_low:.4f},{row.ci_high:.4f}]$ \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_datasets(
    *,
    module: Any,
    args: argparse.Namespace,
    generator_source: pd.DataFrame,
    model_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    synthetic_dir = output_dir / "synthetic_data"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, pd.DataFrame] = {}

    def load_or_generate(
        key: str,
        generator: Callable[[], pd.DataFrame],
    ) -> None:
        path = synthetic_dir / f"process_metric_synthetic_{key}_{args.year}.csv"
        if args.reuse_synthetic and path.exists():
            log(f"Loading cached synthetic data: {path}")
            df = pd.read_csv(path, keep_default_na=False)
        else:
            start = time.perf_counter()
            log(f"Generating {DISPLAY_NAMES.get(key, key)} ...")
            df = generator()
            df.to_csv(path, index=False)
            log(f"Generated {key} in {time.perf_counter() - start:.2f}s: {path}")
        generated[key] = df

    if not args.no_lsm:
        load_or_generate(
            "lsm",
            lambda: module.generate_lsm(
                orig_df=generator_source,
                model_path=model_path,
                num_rows=args.num_rows,
                n_workers=args.n_workers,
            ),
        )

    if not args.no_baseline:
        load_or_generate(
            "baseline",
            lambda: module.generate_baseline(
                orig_df=generator_source,
                num_rows=args.num_rows,
                n_workers=args.n_workers,
            ),
        )

    if not args.no_ctgan:
        load_or_generate(
            "ctgan",
            lambda: module.generate_ctgan(
                orig_df=generator_source,
                num_rows=args.num_rows,
                ctgan_workers=args.ctgan_workers,
                ctgan_train_rows=args.ctgan_train_rows,
                random_state=args.random_state,
            ),
        )

    if args.run_chow_liu:
        load_or_generate(
            "chow_liu",
            lambda: module.generate_chow_liu(
                orig_df=generator_source,
                num_rows=args.num_rows,
                random_state=args.random_state,
                train_rows=args.chow_liu_train_rows,
                max_tree_features=args.chow_liu_max_tree_features,
                max_cardinality=args.chow_liu_max_cardinality,
            ),
        )

    # The process metric is reported relative to this control, so it is required.
    load_or_generate(
        CONTROL_KEY_DEFAULT,
        lambda: module.generate_original_sample_control(
            orig_df=generator_source,
            num_rows=args.num_rows,
            random_state=args.random_state,
        ),
    )

    return generated


def run(args: argparse.Namespace) -> None:
    year = int(args.year)
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = datasets_dir / f"gss_{year}.csv"
    model_path = datasets_dir / f"gss_{year}.pkl.gz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing GSS CSV: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing pretrained LSM model: {model_path}")

    generator_module = load_generator_module(Path(args.generator_script))
    model_feature_names = generator_module.get_lsm_feature_names(model_path)

    log(f"Loading {data_path}")
    orig_df = pd.read_csv(data_path, keep_default_na=False)
    orig_df = generator_module.restrict_to_lsm_features(
        orig_df,
        model_feature_names,
        name="process-metric original dataframe",
    )

    generator_source, reference_full = split_generator_source_and_reference(
        orig_df,
        reference_fraction=args.reference_fraction,
        reference_rows=args.reference_rows,
        random_state=args.random_state,
    )

    generator_source.to_csv(
        output_dir / f"process_metric_generator_source_rows_{year}.csv",
        index=False,
    )
    reference_full.to_csv(
        output_dir / f"process_metric_reference_rows_{year}.csv",
        index=False,
    )

    generated = generate_datasets(
        module=generator_module,
        args=args,
        generator_source=generator_source,
        model_path=model_path,
        output_dir=output_dir,
    )

    metric_features = choose_metric_features(
        model_feature_names,
        feature_list_file=Path(args.feature_list_file) if args.feature_list_file else None,
        feature_limit=args.feature_limit,
    )
    log(
        f"Metric feature set: {len(metric_features)}/{len(model_feature_names)} features."
    )

    reference = normalize_categorical_frame(
        reference_full.loc[:, metric_features],
        missing_token=args.missing_token,
    )

    models: dict[str, Any] = {}
    profiles: dict[str, np.ndarray] = {}
    training_sizes: dict[str, int] = {}

    max_feats = parse_max_feats(args.max_feats)

    for offset, (key, raw_df) in enumerate(generated.items()):
        metric_df = normalize_categorical_frame(
            raw_df.loc[:, metric_features],
            missing_token=args.missing_token,
        )
        train_df = sample_training_rows(
            metric_df,
            n_rows=args.metric_train_rows,
            random_state=args.random_state + 100 + offset,
        )
        training_sizes[key] = len(train_df)

        model_cache = model_dir / f"process_metric_qnet_{key}_{year}.joblib"
        if args.reuse_models and model_cache.exists():
            log(f"Loading cached profile model: {model_cache}")
            model = joblib.load(model_cache)
        else:
            start = time.perf_counter()
            log(
                f"Fitting profile Qnet for {DISPLAY_NAMES.get(key, key)} "
                f"on {len(train_df)} rows x {len(metric_features)} features ..."
            )
            model = fit_profile_qnet(
                df=train_df,
                feature_names=metric_features,
                n_workers=args.n_workers,
                min_samples_split=args.min_samples_split,
                alpha=args.alpha,
                max_depth=args.max_depth,
                max_feats=max_feats,
                early_stopping=args.early_stopping,
                random_state=args.random_state + 1000 + offset,
                verbose=args.qnet_verbose,
            )
            joblib.dump(model, model_cache, compress=args.model_compression)
            log(
                f"Fitted {key} profile model in {time.perf_counter() - start:.2f}s: "
                f"{model_cache}"
            )
        models[key] = model

        start = time.perf_counter()
        log(f"Evaluating normalized profiles for {DISPLAY_NAMES.get(key, key)} ...")
        profiles[key] = evaluate_profiles(
            model=model,
            reference_df=reference,
            profile_workers=args.profile_workers,
        )
        log(
            f"Evaluated {key} profiles in {time.perf_counter() - start:.2f}s; "
            f"shape={profiles[key].shape}"
        )

    if CONTROL_KEY_DEFAULT not in profiles:
        raise RuntimeError("The original (control) profile model was not generated.")

    pairwise_rows: list[dict[str, Any]] = []
    row_distance_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []

    keys = list(profiles.keys())
    for pair_index, (key_a, key_b) in enumerate(combinations(keys, 2)):
        row_distances = profile_distance_by_row(profiles[key_a], profiles[key_b])
        boot = bootstrap_mean(
            row_distances,
            n_bootstrap=args.bootstrap_replicates,
            ci=args.ci,
            random_state=args.random_state + 10_000 + pair_index,
        )
        pairwise_rows.append(
            {
                "year": year,
                "generator_a": key_a,
                "generator_a_label": DISPLAY_NAMES.get(key_a, key_a),
                "generator_b": key_b,
                "generator_b_label": DISPLAY_NAMES.get(key_b, key_b),
                "d_mu_hat": boot.estimate,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
                "n_reference_rows": boot.n_rows,
                "n_features": len(metric_features),
            }
        )

        for row_index, distance in enumerate(row_distances):
            row_distance_rows.append(
                {
                    "year": year,
                    "reference_row": row_index,
                    "generator_a": key_a,
                    "generator_b": key_b,
                    "row_profile_distance": float(distance),
                }
            )

        contributions = coordinate_contributions(profiles[key_a], profiles[key_b])
        order = np.argsort(np.nan_to_num(contributions, nan=-np.inf))[::-1]
        for rank, feature_index in enumerate(order, start=1):
            coordinate_rows.append(
                {
                    "year": year,
                    "generator_a": key_a,
                    "generator_b": key_b,
                    "feature": metric_features[int(feature_index)],
                    "feature_index": int(feature_index),
                    "mean_absolute_profile_difference": float(contributions[feature_index]),
                    "rank_within_pair": rank,
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_path = output_dir / f"process_metric_pairwise_{year}.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    matrix = pd.DataFrame(np.zeros((len(keys), len(keys))), index=keys, columns=keys)
    for row in pairwise_df.itertuples(index=False):
        matrix.loc[row.generator_a, row.generator_b] = row.d_mu_hat
        matrix.loc[row.generator_b, row.generator_a] = row.d_mu_hat
    matrix.index.name = "generator"
    matrix.to_csv(output_dir / f"process_metric_matrix_{year}.csv")

    row_distance_df = pd.DataFrame(row_distance_rows)
    row_distance_df.to_csv(
        output_dir / f"process_metric_row_distances_{year}.csv",
        index=False,
    )

    coordinate_df = pd.DataFrame(coordinate_rows)
    coordinate_df.to_csv(
        output_dir / f"process_metric_coordinate_contributions_{year}.csv",
        index=False,
    )

    control_rows: list[dict[str, Any]] = []
    for key in keys:
        if key == CONTROL_KEY_DEFAULT:
            continue
        match = pairwise_df[
            ((pairwise_df["generator_a"] == key) &
             (pairwise_df["generator_b"] == CONTROL_KEY_DEFAULT))
            |
            ((pairwise_df["generator_b"] == key) &
             (pairwise_df["generator_a"] == CONTROL_KEY_DEFAULT))
        ]
        if len(match) != 1:
            raise RuntimeError(f"Could not resolve unique control distance for {key}.")
        row = match.iloc[0]
        control_rows.append(
            {
                "year": year,
                "generator": key,
                "generator_label": DISPLAY_NAMES.get(key, key),
                "control": CONTROL_KEY_DEFAULT,
                "control_label": DISPLAY_NAMES[CONTROL_KEY_DEFAULT],
                "d_mu_hat": float(row["d_mu_hat"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n_reference_rows": int(row["n_reference_rows"]),
                "n_features": int(row["n_features"]),
            }
        )

    to_control_df = pd.DataFrame(control_rows).sort_values("d_mu_hat")
    to_control_path = output_dir / f"process_metric_to_original_control_{year}.csv"
    to_control_df.to_csv(to_control_path, index=False)
    save_latex_table(
        to_control_df,
        output_dir / f"process_metric_to_original_control_{year}.tex",
        year,
    )

    top_contributions = coordinate_df[
        (
            ((coordinate_df["generator_a"] == CONTROL_KEY_DEFAULT) |
             (coordinate_df["generator_b"] == CONTROL_KEY_DEFAULT))
            & (coordinate_df["rank_within_pair"] <= args.top_features)
        )
    ]
    top_contributions.to_csv(
        output_dir / f"process_metric_top_features_to_original_control_{year}.csv",
        index=False,
    )

    metadata = {
        "year": year,
        "data_path": data_path,
        "pretrained_lsm_model_path": model_path,
        "generator_script": Path(args.generator_script),
        "output_dir": output_dir,
        "generator_source_rows": len(generator_source),
        "reference_rows": len(reference),
        "reference_fraction_requested": args.reference_fraction,
        "generated_rows_per_condition": args.num_rows,
        "profile_model_training_rows": training_sizes,
        "n_model_features_available": len(model_feature_names),
        "n_metric_features": len(metric_features),
        "metric_features": metric_features,
        "generators": keys,
        "display_names": DISPLAY_NAMES,
        "control_key": CONTROL_KEY_DEFAULT,
        "bootstrap_replicates": args.bootstrap_replicates,
        "ci": args.ci,
        "random_state": args.random_state,
        "qnet": {
            "min_samples_split": args.min_samples_split,
            "alpha": args.alpha,
            "max_depth": args.max_depth,
            "max_feats": max_feats,
            "early_stopping": args.early_stopping,
            "n_jobs": args.n_workers,
        },
        "interpretation": (
            "The reported intervals condition on the fitted profile models and "
            "resample common reference rows."
        ),
    }
    (output_dir / f"process_metric_metadata_{year}.json").write_text(
        json.dumps(jsonable(metadata), indent=2),
        encoding="utf-8",
    )

    log("\nEmpirical distances to Original (control):")
    log(to_control_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    log(f"\nSaved pairwise results: {pairwise_path}")
    log(f"Saved control summary: {to_control_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-wave LSYNTH comparators and estimate the empirical "
            "normalized-profile process metric d_mu."
        )
    )

    parser.add_argument(
        "year",
        type=int,
        help="GSS year; uses gss_<year>.csv and gss_<year>.pkl.gz.",
    )
    parser.add_argument(
        "--generator-script",
        default="./lsynth_eval_comparators_fast_chow.py",
        help=(
            "Path to the existing comparator-generation script. Its generation "
            "functions are reused directly."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        default="./datasets",
        help="Directory containing the GSS CSV and pretrained LSM model.",
    )
    parser.add_argument(
        "--output-dir",
        default="process_metric_outputs",
        help="Output directory for generated data, models, and metric tables.",
    )

    parser.add_argument(
        "--num-rows",
        type=int,
        default=1000,
        help="Rows generated per condition. Default: 1000.",
    )
    parser.add_argument(
        "--metric-train-rows",
        type=int,
        default=1000,
        help=(
            "Equal number of generated rows used to fit each profile Qnet. "
            "Use 0 or a negative value for all available rows. Default: 1000."
        ),
    )
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.50,
        help=(
            "Fraction of the input GSS CSV reserved as the common E_mu reference "
            "partition before comparator generation. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--reference-rows",
        type=int,
        default=500,
        help=(
            "Rows sampled from the reserved E_mu partition. Use 0 or a negative "
            "value for all reserved rows. Default: 500."
        ),
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=N_WORKERS_DEFAULT,
        help="Workers for generation and Qnet fitting. Default: 120.",
    )
    parser.add_argument(
        "--profile-workers",
        type=int,
        default=1,
        help=(
            "Threads for evaluating reference-row profiles. Default: 1 for "
            "maximum compatibility; increase after a successful smoke run."
        ),
    )
    parser.add_argument(
        "--ctgan-workers",
        type=int,
        default=CTGAN_WORKERS_DEFAULT,
        help="Worker count passed to CTGAN generation. Default: 10.",
    )
    parser.add_argument(
        "--ctgan-train-rows",
        type=int,
        default=50,
        help="Rows from the source partition used to train CTGAN. Default: 50.",
    )
    parser.add_argument(
        "--no-ctgan",
        action="store_true",
        help="Skip CTGAN.",
    )
    parser.add_argument(
        "--no-lsm",
        action="store_true",
        help="Skip LSM generation.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the independent baseline.",
    )

    parser.add_argument(
        "--run-chow-liu",
        action="store_true",
        help="Enable the restricted categorical Chow--Liu hybrid.",
    )
    parser.add_argument(
        "--chow-liu-train-rows",
        type=int,
        default=CHOW_LIU_TRAIN_ROWS_DEFAULT,
        help=f"Rows used by Chow--Liu. Default: {CHOW_LIU_TRAIN_ROWS_DEFAULT}.",
    )
    parser.add_argument(
        "--chow-liu-max-tree-features",
        type=int,
        default=CHOW_LIU_MAX_TREE_FEATURES_DEFAULT,
        help=(
            "Maximum variables in the Chow--Liu MI tree; remaining variables "
            f"are sampled independently. Default: {CHOW_LIU_MAX_TREE_FEATURES_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--chow-liu-max-cardinality",
        type=int,
        default=CHOW_LIU_MAX_CARDINALITY_DEFAULT,
        help=(
            "Maximum cardinality for variables included in the Chow--Liu tree. "
            f"Default: {CHOW_LIU_MAX_CARDINALITY_DEFAULT}."
        ),
    )

    parser.add_argument(
        "--feature-list-file",
        default=None,
        help=(
            "Optional text file containing one model feature per line. When "
            "omitted, all LSM model features are used."
        ),
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=0,
        help=(
            "Optional deterministic limit applied after feature-list selection. "
            "Use 0 for all selected features. Default: 0."
        ),
    )
    parser.add_argument(
        "--missing-token",
        default=MISSING_TOKEN_DEFAULT,
        help=f"Categorical token used for missing values. Default: {MISSING_TOKEN_DEFAULT}.",
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Qnet conditional-tree minimum split size. Default: 2.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Qnet conditional-tree permutation-test threshold. Default: 0.05.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Qnet conditional-tree maximum depth. Default: -1.",
    )
    parser.add_argument(
        "--max-feats",
        default="-1",
        help="Qnet maximum features per split: integer, sqrt, log, or all. Default: -1.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable Qnet early stopping during conditional-tree feature selection.",
    )
    parser.add_argument(
        "--qnet-verbose",
        type=int,
        default=0,
        help="Qnet training verbosity. Default: 0.",
    )
    parser.add_argument(
        "--model-compression",
        type=int,
        default=3,
        help="Joblib compression level for profile Qnets. Default: 3.",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
        help="Reference-row bootstrap replicates. Default: 5000.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Central bootstrap interval width. Default: 0.95.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Top coordinate contributions retained in the compact output. Default: 20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed. Default: 123.",
    )
    parser.add_argument(
        "--reuse-synthetic",
        action="store_true",
        help="Reuse generated comparator CSVs already present in the output directory.",
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reuse fitted profile Qnets already present in the output directory.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
