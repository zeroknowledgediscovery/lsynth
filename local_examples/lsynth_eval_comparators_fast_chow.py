#!/usr/bin/env python3
"""
LSynth multi-year synthetic-data evaluation for GSS year-specific runs.

Usage:
    python lsynth_eval_comparators.py 2018

Expected inputs:
    ./datasets/gss_2018.csv
    ./datasets/gss_2018.pkl.gz

Outputs:
    synthetic_eval_outputs/
        synthetic_eval_results_2018.csv
        synthetic_eval_results_long_2018.csv
        synthetic_eval_results_wide_2018.csv
        synthetic_eval_generator_summary_2018.csv
        timing_2018.csv
        plot_manifest_2018.csv
        run_metadata_2018.json
        plots/*_2018.png

Notes:
    - No AUC/discriminator evaluation is run in this script.
    - CTGAN uses a separate worker count; default is 10.
    - Optional comparators are off by default: Chow-Liu, synthpop, and SDV Gaussian Copula.
    - LSM, baseline, Upsilon, and similarity calculations use the general worker count; default is 120.
    - All dataframes evaluated against the LSM are restricted to lsmmodel.feature_names.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Thread settings: set before numpy/scipy/sklearn where possible.
# ---------------------------------------------------------------------

N_WORKERS_DEFAULT = 120
CTGAN_WORKERS_DEFAULT = 10
CHOW_LIU_TRAIN_ROWS_DEFAULT = 5000
CHOW_LIU_MAX_TREE_FEATURES_DEFAULT = 150
CHOW_LIU_MAX_CARDINALITY_DEFAULT = 75
SYNTHPOP_METHODS_DEFAULT = ["cart"]
SYNTHPOP_MAXFACLEVELS_DEFAULT = 300
SYNTHPOP_HIGH_CARDINALITY_DEFAULT = [
    "SPIND10", "COOCC10", "COIND10", "MAIND10", "MAJOR1", "majorcol",
    "ISCO08", "PAISCO08", "MAISCO08", "SPISCO08", "COISCO08",
]
SMOKE_TEST_ROWS_DEFAULT = 50

os.environ.setdefault("OMP_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("MKL_NUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_WORKERS_DEFAULT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_WORKERS_DEFAULT))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Global collection buffers.
# ---------------------------------------------------------------------

PLOT_MANIFEST: list[dict[str, Any]] = []
RESULT_ROWS: list[dict[str, Any]] = []
TIMING_ROWS: list[dict[str, Any]] = []


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(msg, flush=True)


@contextlib.contextmanager
def timed_step(step_name: str):
    start_wall = now_string()
    start = time.perf_counter()
    log(f"[{start_wall}] START {step_name}")
    error = None
    try:
        yield
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        end_wall = now_string()
        elapsed = time.perf_counter() - start
        TIMING_ROWS.append(
            {
                "step": step_name,
                "start_time": start_wall,
                "end_time": end_wall,
                "elapsed_seconds": elapsed,
                "error": error,
            }
        )
        log(f"[{end_wall}] END   {step_name} | elapsed={elapsed:.3f}s")


def slugify(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s))
    out = "_".join(part for part in out.split("_") if part)
    return out or "unnamed"


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def record_result(
    *,
    year: int,
    generator: str,
    metric: str,
    value: Any,
    category: str,
    extra: dict[str, Any] | None = None,
) -> None:
    row = {
        "year": year,
        "generator": generator,
        "category": category,
        "metric": metric,
        "value": value,
    }
    if extra:
        row.update(extra)
    RESULT_ROWS.append(row)


def save_figure(fig: plt.Figure, path: Path, *, year: int, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    PLOT_MANIFEST.append(
        {
            "year": year,
            "plot_path": str(path),
            "filename": path.name,
            "description": description,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    )
    log(f"Saved plot: {path}")


def require_lsynth():
    try:
        from lsynth import compute_upsilon, generate_syndata
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import `lsynth`. Run this script in the environment where "
            "`compute_upsilon` and `generate_syndata` are available."
        ) from exc
    return compute_upsilon, generate_syndata


def make_one_hot_encoder(*, sparse_output: bool, dtype: Any | None = None):
    from sklearn.preprocessing import OneHotEncoder

    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if dtype is not None:
        kwargs["dtype"] = dtype

    try:
        return OneHotEncoder(sparse_output=sparse_output, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=sparse_output, **kwargs)


def load_lsm_model(model_path: Path) -> Any:
    """Load the serialized LSM model so its feature_names can be enforced."""
    errors: list[str] = []

    try:
        import joblib

        return joblib.load(model_path)
    except Exception as exc:
        errors.append(f"joblib.load: {exc!r}")

    try:
        with gzip.open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        errors.append(f"gzip+pickle: {exc!r}")

    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        errors.append(f"pickle: {exc!r}")

    raise RuntimeError(
        "Could not load LSM model to read feature_names from "
        f"{model_path}. Tried: " + " | ".join(errors)
    )


def get_lsm_feature_names(model_path: Path) -> list[str]:
    """Return model.feature_names, with a few defensive fallbacks."""
    model = load_lsm_model(model_path)

    feature_names = None
    for attr in ("feature_names", "feature_names_"):
        if hasattr(model, attr):
            feature_names = getattr(model, attr)
            break

    if feature_names is None and isinstance(model, dict):
        for key in ("feature_names", "feature_names_"):
            if key in model:
                feature_names = model[key]
                break

    if feature_names is None:
        raise AttributeError(
            "The loaded LSM model does not expose `feature_names` or "
            "`feature_names_`. Cannot safely align dataframes to the model."
        )

    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("LSM model feature_names is empty.")

    # Preserve exact string names used in CSV/model matching.
    feature_names = [str(c) for c in feature_names]
    return feature_names


def restrict_to_lsm_features(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    name: str,
) -> pd.DataFrame:
    """Drop non-model columns and order the dataframe exactly as model.feature_names."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        preview = missing[:20]
        raise ValueError(
            f"{name}: dataframe is missing {len(missing)} LSM model feature columns. "
            f"First missing columns: {preview}"
        )

    extra = [c for c in df.columns if c not in feature_names]
    if extra:
        preview = extra[:20]
        log(
            f"{name}: dropping {len(extra)} columns not present in "
            f"lsmmodel.feature_names. First dropped columns: {preview}"
        )

    return df.loc[:, feature_names].copy()


# ---------------------------------------------------------------------
# Synthetic-data generation.
# ---------------------------------------------------------------------

def align_columns(orig_df: pd.DataFrame, syn_df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if list(syn_df.columns) == list(orig_df.columns):
        return syn_df

    if set(syn_df.columns) == set(orig_df.columns):
        return syn_df.loc[:, list(orig_df.columns)]

    if len(syn_df.columns) == len(orig_df.columns):
        warnings.warn(
            f"{name}: generated columns did not match original names, but column "
            "counts match. Renaming generated columns to the original schema.",
            RuntimeWarning,
        )
        syn_df = syn_df.copy()
        syn_df.columns = orig_df.columns
        return syn_df

    raise ValueError(
        f"{name}: generated dataframe has incompatible columns. "
        f"orig={len(orig_df.columns)} generated={len(syn_df.columns)}"
    )


def generate_lsm(
    *,
    orig_df: pd.DataFrame,
    model_path: Path,
    num_rows: int,
    n_workers: int,
) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    df = generate_syndata(
        num=num_rows,
        model_path=str(model_path),
        gen_algorithm="LSM",
        n_workers=n_workers,
    )
    return align_columns(orig_df, df, name="LSM")


def generate_baseline(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    n_workers: int,
) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    df = generate_syndata(
        num=num_rows,
        gen_algorithm="BASELINE",
        orig_df=orig_df,
        n_workers=n_workers,
    )
    return align_columns(orig_df, df, name="baseline")


def generate_ctgan(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    ctgan_workers: int,
    ctgan_train_rows: int,
    random_state: int,
) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    n_train = min(ctgan_train_rows, len(orig_df))
    train_df = orig_df.sample(n=n_train, random_state=random_state)
    df = generate_syndata(
        num=num_rows,
        gen_algorithm="CTGAN",
        orig_df=train_df,
        n_workers=ctgan_workers,
    )
    return align_columns(orig_df, df, name="CTGAN")


def _sample_series_from_counts(counts: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    probs = counts.to_numpy(dtype=float)
    probs = probs / probs.sum()
    values = counts.index.to_numpy(dtype=object)
    return rng.choice(values, size=n, replace=True, p=probs)


def _categorical_entropy(counts: pd.Series) -> float:
    probs = counts.to_numpy(dtype=float)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _choose_chow_liu_tree_columns(
    train_df: pd.DataFrame,
    *,
    max_tree_features: int,
    max_cardinality: int,
) -> list[str]:
    """
    Choose columns for the expensive Chow-Liu pairwise-MI tree.

    GSS can have hundreds of categorical fields. A full Chow-Liu tree over 688
    columns requires ~236k pairwise MI calculations, and high-cardinality fields
    make each calculation slow. By default we tree-model a bounded subset of
    lower-cardinality, non-degenerate columns and sample the remaining columns
    independently. Set --chow-liu-max-tree-features 0 and
    --chow-liu-max-cardinality 0 to request the full tree.
    """
    rows: list[tuple[str, int, float]] = []
    for c in train_df.columns:
        counts = train_df[c].value_counts(dropna=False)
        card = int(len(counts))
        if card <= 1:
            continue
        if max_cardinality and max_cardinality > 0 and card > max_cardinality:
            continue
        rows.append((c, card, _categorical_entropy(counts)))

    if not rows:
        return []

    # Prefer variables that vary but do not explode cardinality.
    # Sorting by entropy descending gives the tree a useful dependence-bearing
    # subset; cardinality filtering controls the pain.
    rows.sort(key=lambda x: (x[2], -x[1]), reverse=True)

    if max_tree_features and max_tree_features > 0:
        rows = rows[: int(max_tree_features)]

    return [c for c, _, _ in rows]


def generate_chow_liu(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    random_state: int,
    train_rows: int,
    max_tree_features: int,
    max_cardinality: int,
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """
    Pure-Python categorical Chow-Liu tree generator.

    It learns a maximum-spanning tree over a bounded subset of categorical
    columns using pairwise mutual information, then samples the root marginal
    and child conditionals. Columns not included in the tree are sampled from
    their empirical marginals so the output still has the full LSM feature
    schema.
    """
    from sklearn.metrics import mutual_info_score

    rng = np.random.default_rng(random_state)
    if train_rows and train_rows > 0:
        n_train = min(int(train_rows), len(orig_df))
        train_df = orig_df.sample(n=n_train, random_state=random_state).reset_index(drop=True)
    else:
        train_df = orig_df.reset_index(drop=True)

    train_df = train_df.replace("", pd.NA).astype("string").fillna(missing_token).astype(object)
    all_cols = list(train_df.columns)
    if not all_cols:
        raise ValueError("Chow-Liu received a dataframe with zero columns.")

    tree_cols = _choose_chow_liu_tree_columns(
        train_df,
        max_tree_features=max_tree_features,
        max_cardinality=max_cardinality,
    )
    independent_cols = [c for c in all_cols if c not in set(tree_cols)]

    log(
        "Chow-Liu: using "
        f"{len(tree_cols)}/{len(all_cols)} columns in the pairwise-MI tree; "
        f"{len(independent_cols)} columns sampled independently. "
        f"train_rows={len(train_df)}, max_tree_features={max_tree_features}, "
        f"max_cardinality={max_cardinality}"
    )

    sampled: dict[str, np.ndarray] = {}

    # Start by sampling all non-tree columns independently. This guarantees a
    # complete dataframe even when the tree subset is intentionally bounded.
    for c in independent_cols:
        counts = train_df[c].value_counts(dropna=False)
        sampled[c] = _sample_series_from_counts(counts, num_rows, rng)

    if len(tree_cols) == 0:
        for c in all_cols:
            if c not in sampled:
                counts = train_df[c].value_counts(dropna=False)
                sampled[c] = _sample_series_from_counts(counts, num_rows, rng)
        syn = pd.DataFrame({c: sampled[c] for c in all_cols}, columns=all_cols)
        syn = syn.replace(missing_token, "")
        return align_columns(orig_df, syn, name="chow_liu")

    cols = tree_cols
    m = len(cols)

    codes: dict[str, np.ndarray] = {}
    for c in cols:
        codes[c] = pd.factorize(train_df[c], sort=False)[0]

    mi = np.zeros((m, m), dtype=float)
    total_pairs = m * (m - 1) // 2
    done_pairs = 0
    last_log = 0
    for i in range(m):
        ci = codes[cols[i]]
        for j in range(i + 1, m):
            v = float(mutual_info_score(ci, codes[cols[j]]))
            mi[i, j] = v
            mi[j, i] = v
            done_pairs += 1
        # Coarse progress reporting, useful on slow machines.
        pct = int(100 * done_pairs / max(total_pairs, 1))
        if pct >= last_log + 10:
            last_log = pct
            log(f"Chow-Liu: pairwise MI {done_pairs}/{total_pairs} pairs ({pct}%).")

    # Prim maximum-spanning tree, rooted at column 0. parent[i] gives parent of i.
    parent = np.full(m, -1, dtype=int)
    selected = np.zeros(m, dtype=bool)
    selected[0] = True
    for _ in range(m - 1):
        best_i = -1
        best_j = -1
        best_w = -np.inf
        selected_idx = np.flatnonzero(selected)
        unselected_idx = np.flatnonzero(~selected)
        for i in selected_idx:
            weights = mi[i, unselected_idx]
            k = int(np.argmax(weights))
            j = int(unselected_idx[k])
            w = float(weights[k])
            if w > best_w:
                best_w = w
                best_i = i
                best_j = j
        if best_j < 0:
            best_j = int(np.flatnonzero(~selected)[0])
            best_i = 0
        parent[best_j] = best_i
        selected[best_j] = True

    children: dict[int, list[int]] = {i: [] for i in range(m)}
    for child_idx, parent_idx in enumerate(parent):
        if parent_idx >= 0:
            children[int(parent_idx)].append(int(child_idx))

    root_col = cols[0]
    root_counts = train_df[root_col].value_counts(dropna=False)
    sampled[root_col] = _sample_series_from_counts(root_counts, num_rows, rng)

    # Sample along the tree breadth-first.
    queue = [0]
    while queue:
        p_idx = queue.pop(0)
        p_col = cols[p_idx]
        for c_idx in children[p_idx]:
            c_col = cols[c_idx]
            fallback_counts = train_df[c_col].value_counts(dropna=False)
            grouped_counts = {
                pv: sub[c_col].value_counts(dropna=False)
                for pv, sub in train_df.groupby(p_col, dropna=False, sort=False)
            }

            out = np.empty(num_rows, dtype=object)
            parent_values = sampled[p_col]
            for pv in np.unique(parent_values):
                mask = parent_values == pv
                counts = grouped_counts.get(pv, fallback_counts)
                out[mask] = _sample_series_from_counts(counts, int(mask.sum()), rng)
            sampled[c_col] = out
            queue.append(c_idx)

    # Defensive fill for any column not already sampled.
    for c in all_cols:
        if c not in sampled:
            counts = train_df[c].value_counts(dropna=False)
            sampled[c] = _sample_series_from_counts(counts, num_rows, rng)

    syn = pd.DataFrame({c: sampled[c] for c in all_cols}, columns=all_cols)
    syn = syn.replace(missing_token, "")
    return align_columns(orig_df, syn, name="chow_liu")


def generate_synthpop_method(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    random_state: int,
    synthpop_seed: int | None,
    synthpop_method: str,
    synthpop_maxfaclevels: int,
    synthpop_train_rows: int,
    synthpop_high_cardinality_exclude: list[str],
    synthpop_install_missing: bool,
    synthpop_cran_mirror: str,
) -> pd.DataFrame:
    """
    Generate synthetic rows using R synthpop for one method, e.g. cart/ranger/rf.

    This is optional and intentionally delayed because synthpop can be slow on
    high-cardinality categorical survey data.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "synthpop requires rpy2 and an R installation. Install rpy2/R/synthpop "
            "or rerun without --run-synthpop."
        ) from exc

    seed = int(random_state if synthpop_seed is None else synthpop_seed)
    if synthpop_train_rows and synthpop_train_rows > 0:
        n_train = min(int(synthpop_train_rows), len(orig_df))
        train_df = orig_df.sample(n=n_train, random_state=random_state).reset_index(drop=True)
    else:
        train_df = orig_df.reset_index(drop=True)

    log(
        "Generating rows via R synthpop "
        f"method={synthpop_method}, train_rows={len(train_df)}, seed={seed}, "
        f"maxfaclevels={synthpop_maxfaclevels}"
    )

    ro.r(f'options(repos = c(CRAN="{synthpop_cran_mirror}"))')
    if synthpop_install_missing:
        ro.r('if (!requireNamespace("synthpop", quietly=TRUE)) install.packages("synthpop")')
    ro.r("library(synthpop)")

    high_vars = [c for c in synthpop_high_cardinality_exclude if c in train_df.columns]

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(train_df)
        ro.globalenv["df"] = r_df
        ro.globalenv["high_vars"] = ro.StrVector(high_vars)
        ro.globalenv["seed_py"] = int(seed)
        ro.globalenv["method_py"] = str(synthpop_method)
        ro.globalenv["maxfaclevels_py"] = int(synthpop_maxfaclevels)

        ro.r(
            """
            tmp <- syn(df, seed=seed_py, maxfaclevels=maxfaclevels_py, method=method_py)
            pm <- tmp$predictor.matrix
            high <- intersect(high_vars, colnames(df))
            if (length(high) > 0) {
                pm[, high] <- 0
            }
            syn_obj <- syn(df, seed=seed_py, maxfaclevels=maxfaclevels_py,
                           predictor.matrix=pm, method=method_py)
            """
        )
        syn = ro.conversion.rpy2py(ro.r("syn_obj$syn"))

    syn = pd.DataFrame(syn)
    syn = align_columns(orig_df, syn, name=f"synthpop_{synthpop_method}")

    if len(syn) != num_rows:
        replace = num_rows > len(syn)
        syn = syn.sample(n=num_rows, replace=replace, random_state=random_state).reset_index(drop=True)

    return align_columns(orig_df, syn, name=f"synthpop_{synthpop_method}")


def generate_sdv_gaussian_copula(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    random_state: int,
    train_rows: int,
) -> pd.DataFrame:
    """Optional SDV Gaussian Copula comparator."""
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SDV Gaussian Copula requires the `sdv` package. Install it or rerun "
            "without --run-sdv-gaussian-copula."
        ) from exc

    if train_rows and train_rows > 0:
        n_train = min(int(train_rows), len(orig_df))
        train_df = orig_df.sample(n=n_train, random_state=random_state).reset_index(drop=True)
    else:
        train_df = orig_df.reset_index(drop=True)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_df)
    syn = synthesizer.sample(num_rows=num_rows)
    return align_columns(orig_df, syn, name="sdv_gaussian_copula")


def generate_original_sample_control(
    *,
    orig_df: pd.DataFrame,
    num_rows: int,
    random_state: int,
) -> pd.DataFrame:
    replace = num_rows > len(orig_df)
    return orig_df.sample(n=num_rows, replace=replace, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------
# Upsilon.
# ---------------------------------------------------------------------

def compute_upsilon_for_df(
    *,
    year: int,
    generator: str,
    df: pd.DataFrame,
    model_path: Path,
    feature_names: list[str],
    n_workers: int,
    out_dir: Path,
) -> np.ndarray:
    compute_upsilon, _ = require_lsynth()

    df = restrict_to_lsm_features(
        df,
        feature_names,
        name=f"{generator} dataframe before compute_upsilon",
    )

    ups, _ = compute_upsilon(
        df,
        model_path=str(model_path),
        n_workers=n_workers,
    )
    ups = np.asarray(ups, dtype=float)

    ups_path = out_dir / f"upsilon_{slugify(generator)}_{year}.csv"
    pd.DataFrame({"upsilon": ups}).to_csv(ups_path, index=False)

    stats = {
        "n": int(np.isfinite(ups).sum()),
        "mean": float(np.nanmean(ups)),
        "median": float(np.nanmedian(ups)),
        "std": float(np.nanstd(ups)),
        "p05": float(np.nanquantile(ups, 0.05)),
        "p95": float(np.nanquantile(ups, 0.95)),
    }

    log(f"{generator}: printed_nanmean_upsilon = {stats['mean']}")

    record_result(
        year=year,
        generator=generator,
        category="upsilon",
        metric="printed_nanmean_upsilon",
        value=stats["mean"],
    )

    for k, v in stats.items():
        record_result(
            year=year,
            generator=generator,
            category="upsilon",
            metric=f"upsilon_{k}",
            value=v,
        )

    return ups


def plot_upsilon_distribution(
    *,
    year: int,
    generator: str,
    ups: np.ndarray,
    plot_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ups[np.isfinite(ups)], bins=40)
    ax.set_title(f"Upsilon distribution: {generator} ({year})")
    ax.set_xlabel("Upsilon")
    ax.set_ylabel("count")
    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / f"upsilon_{slugify(generator)}_{year}.png",
        year=year,
        description=f"Upsilon distribution for {generator}",
    )


def plot_upsilon_overlay(
    *,
    year: int,
    ups_by_generator: dict[str, np.ndarray],
    plot_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for generator, ups in ups_by_generator.items():
        ax.hist(ups[np.isfinite(ups)], bins=40, histtype="step", label=generator)
    ax.set_title(f"Upsilon distributions ({year})")
    ax.set_xlabel("Upsilon")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / f"upsilon_all_generators_{year}.png",
        year=year,
        description="Overlay of Upsilon distributions across generators",
    )


# ---------------------------------------------------------------------
# Novelty / row similarity.
# ---------------------------------------------------------------------

def row_similarity_blockplot(
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    *,
    year: int,
    generator: str,
    metric: str = "cosine",
    max_rows: int | None = 2000,
    random_state: int = 123,
    missing_token: str = "__MISSING__",
    plot_path: Path,
) -> dict[str, Any]:
    from sklearn.preprocessing import normalize

    if list(orig_df.columns) != list(syn_df.columns):
        raise ValueError("orig_df and syn_df must have identical columns in the same order.")

    o = orig_df.copy()
    s = syn_df.copy()

    if max_rows is not None:
        n = min(len(o), len(s), max_rows)
        o = o.sample(n=n, random_state=random_state).reset_index(drop=True)
        s = s.sample(n=n, random_state=random_state).reset_index(drop=True)

    def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace("", pd.NA)
        df = df.astype("string").fillna(missing_token)
        return df.astype(object)

    o = normalize_frame(o)
    s = normalize_frame(s)

    if metric == "match":
        X = np.vstack([o.to_numpy(), s.to_numpy()])
        G = (X[:, None, :] == X[None, :, :]).mean(axis=2).astype(np.float32)

    elif metric == "cosine":
        comb = pd.concat([o, s], axis=0, ignore_index=True)
        enc = make_one_hot_encoder(sparse_output=True, dtype=np.float32)
        X = enc.fit_transform(comb)
        X = normalize(X, norm="l2", axis=1, copy=False)
        G = X @ X.T
        if hasattr(G, "toarray"):
            G = G.toarray()
        G = np.asarray(G, dtype=np.float32)

    else:
        raise ValueError("metric must be 'cosine' or 'match'.")

    n = len(o)

    rr = float(G[:n, :n].mean())
    rs = float(G[:n, n:].mean())
    sr = float(G[n:, :n].mean())
    ss = float(G[n:, n:].mean())

    block_means = np.array([[rr, rs], [sr, ss]], dtype=float)

    log(f"{generator}: printed_block_mean_matrix =")
    log(str(block_means))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(G, aspect="auto")
    axes[0].axhline(n - 0.5)
    axes[0].axvline(n - 0.5)
    axes[0].set_title(f"Row-row similarity ({metric})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(block_means, aspect="equal")
    axes[1].set_xticks([0, 1], labels=["Real", "Synth"])
    axes[1].set_yticks([0, 1], labels=["Real", "Synth"])
    axes[1].set_title("2x2 block mean similarity")

    for (i, j), v in np.ndenumerate(block_means):
        axes[1].text(j, i, f"{v:.3f}", ha="center", va="center")

    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()

    save_figure(
        fig,
        plot_path,
        year=year,
        description=f"Novelty row-similarity block plot for {generator}",
    )

    off_block = G[:n, n:]
    nn = off_block.max(axis=0)
    nearest_median = float(np.median(nn))

    stats = {
        "n_rows_each": int(n),
        "nearest_real_similarity_mean": float(nn.mean()),
        "nearest_real_similarity_median": nearest_median,
        "nearest_real_similarity_p95": float(np.quantile(nn, 0.95)),
        "inverse_nearest_real_similarity_median": (
            float(1.0 / nearest_median) if nearest_median > 0 else math.inf
        ),
        "missing_token_used": missing_token,
        "block_means": {
            "real_real": rr,
            "real_synth": rs,
            "synth_real": sr,
            "synth_synth": ss,
        },
    }

    log(f"{generator}: printed_stats_dictionary =")
    log(str(stats))

    record_result(
        year=year,
        generator=generator,
        category="similarity",
        metric="printed_block_mean_matrix_json",
        value=json.dumps(block_means.tolist()),
    )

    record_result(
        year=year,
        generator=generator,
        category="similarity",
        metric="printed_stats_dictionary_json",
        value=json.dumps(jsonable(stats)),
    )

    for k, v in stats.items():
        if k == "block_means":
            for kk, vv in v.items():
                record_result(
                    year=year,
                    generator=generator,
                    category="similarity",
                    metric=f"block_mean_{kk}",
                    value=vv,
                )
        elif isinstance(v, (int, float, str)):
            record_result(
                year=year,
                generator=generator,
                category="similarity",
                metric=k,
                value=v,
            )

    return stats


# ---------------------------------------------------------------------
# Marginal and generator-map plots.
# ---------------------------------------------------------------------

def plot_single_column_distribution(
    *,
    year: int,
    orig_df: pd.DataFrame,
    generated: dict[str, pd.DataFrame],
    column_index: int,
    plot_dir: Path,
) -> None:
    if column_index < 0 or column_index >= len(orig_df.columns):
        return

    col = orig_df.columns[column_index]

    fig, ax = plt.subplots(figsize=(8, 5))

    def numeric_series(x: pd.Series) -> pd.Series:
        y = x.replace("", np.nan)
        return pd.to_numeric(y, errors="coerce")

    real = numeric_series(orig_df.iloc[:, column_index])
    if real.notna().sum() >= 2:
        real.plot(kind="density", ax=ax, label="orig")

    for name, df in generated.items():
        s = numeric_series(df.iloc[:, column_index])
        if s.notna().sum() >= 2:
            s.plot(kind="density", ax=ax, label=name)

    ax.set_title(f"Column {column_index} distribution: {col} ({year})")
    ax.set_xlabel(str(col))
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / f"column_{column_index}_{slugify(col)}_{year}.png",
        year=year,
        description=f"Marginal distribution comparison for column {column_index}: {col}",
    )


def plot_generator_map(
    *,
    year: int,
    summary: dict[str, dict[str, Any]],
    plot_dir: Path,
) -> None:
    names: list[str] = []
    x: list[float] = []
    y: list[float] = []

    for generator, stats in summary.items():
        ups_median = stats.get("upsilon_median")
        inv_sim = stats.get("inverse_nearest_real_similarity_median")
        if ups_median is None or inv_sim is None:
            continue
        names.append(generator)
        x.append(float(inv_sim))
        y.append(float(ups_median))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    ax.set_xlabel("1 / nearest-real similarity median")
    ax.set_ylabel("Upsilon median")
    ax.set_title(f"Generator map: Upsilon vs novelty ({year})")

    for name, xi, yi in zip(names, x, y):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left")

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir / f"generator_map_upsilon_vs_inverse_nearest_real_similarity_{year}.png",
        year=year,
        description="Generator map using Upsilon median and inverse nearest-real similarity median",
    )


# ---------------------------------------------------------------------
# Result saving.
# ---------------------------------------------------------------------

def make_results_dataframes(year: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_df = pd.DataFrame(RESULT_ROWS)

    if long_df.empty:
        wide_df = pd.DataFrame()
        summary_df = pd.DataFrame()
    else:
        numeric_df = long_df.copy()
        numeric_df["value_numeric"] = pd.to_numeric(numeric_df["value"], errors="coerce")

        wide_df = (
            numeric_df
            .pivot_table(
                index=["year", "generator"],
                columns="metric",
                values="value_numeric",
                aggfunc="first",
            )
            .reset_index()
        )
        wide_df.columns.name = None
        summary_df = wide_df.copy()

    long_path = out_dir / f"synthetic_eval_results_long_{year}.csv"
    wide_path = out_dir / f"synthetic_eval_results_wide_{year}.csv"
    summary_path = out_dir / f"synthetic_eval_results_{year}.csv"

    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    try:
        long_df.to_parquet(out_dir / f"synthetic_eval_results_long_{year}.parquet", index=False)
        wide_df.to_parquet(out_dir / f"synthetic_eval_results_wide_{year}.parquet", index=False)
        summary_df.to_parquet(out_dir / f"synthetic_eval_results_{year}.parquet", index=False)
    except Exception as exc:
        log(f"Parquet save skipped: {exc}")

    log(f"Saved results dataframe: {summary_path}")
    log(f"Saved long results dataframe: {long_path}")
    log(f"Saved wide results dataframe: {wide_path}")

    return summary_df, long_df, wide_df


def save_timing_and_manifest(year: int, out_dir: Path) -> None:
    timing_df = pd.DataFrame(TIMING_ROWS)
    timing_path = out_dir / f"timing_{year}.csv"
    timing_df.to_csv(timing_path, index=False)
    log(f"Saved timing table: {timing_path}")

    manifest_df = pd.DataFrame(PLOT_MANIFEST)
    manifest_path = out_dir / f"plot_manifest_{year}.csv"
    manifest_df.to_csv(manifest_path, index=False)
    log(f"Saved plot manifest: {manifest_path}")


# ---------------------------------------------------------------------
# Main run.
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    year = int(args.year)

    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    synthetic_dir = out_dir / "synthetic_data"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    data_path = datasets_dir / f"gss_{year}.csv"
    model_path = datasets_dir / f"gss_{year}.pkl.gz"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data CSV: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with timed_step("load_lsm_feature_names"):
        feature_names = get_lsm_feature_names(model_path)
        log(f"Loaded {len(feature_names)} LSM feature_names from model.")

    metadata = {
        "year": year,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "n_lsm_features": len(feature_names),
        "lsm_feature_names": feature_names,
        "output_dir": str(out_dir),
        "num_rows": args.num_rows,
        "n_workers": args.n_workers,
        "ctgan_workers": args.ctgan_workers,
        "ctgan_train_rows": args.ctgan_train_rows,
        "similarity_max_rows": args.similarity_max_rows,
        "random_state": args.random_state,
        "run_ctgan": not args.no_ctgan,
        "run_chow_liu": args.run_chow_liu,
        "chow_liu_train_rows": args.chow_liu_train_rows,
        "chow_liu_max_tree_features": args.chow_liu_max_tree_features,
        "chow_liu_max_cardinality": args.chow_liu_max_cardinality,
        "run_synthpop": args.run_synthpop,
        "synthpop_methods": args.synthpop_methods,
        "run_sdv_gaussian_copula": args.run_sdv_gaussian_copula,
        "smoke_test_only": args.smoke_test_only,
        "start_time": now_string(),
    }

    effective_num_rows = min(args.num_rows, args.smoke_test_rows) if args.smoke_test_only else args.num_rows
    if args.smoke_test_only:
        log(f"Smoke-test mode: generating only {effective_num_rows} rows per selected generator and skipping metrics.")

    with timed_step("read_original_data"):
        orig_df = pd.read_csv(data_path, keep_default_na=False)
        log(f"Original data shape before LSM feature restriction: {orig_df.shape[0]} rows x {orig_df.shape[1]} columns")
        orig_df = restrict_to_lsm_features(
            orig_df,
            feature_names,
            name="original dataframe",
        )
        log(f"Original data shape after LSM feature restriction: {orig_df.shape[0]} rows x {orig_df.shape[1]} columns")

    generated: dict[str, pd.DataFrame] = {}

    if not args.no_lsm:
        with timed_step("generate_lsm"):
            generated["lsm"] = generate_lsm(
                orig_df=orig_df,
                model_path=model_path,
                num_rows=effective_num_rows,
                n_workers=args.n_workers,
            )
            generated["lsm"].to_csv(synthetic_dir / f"synthetic_lsm_{year}.csv", index=False)

    if not args.no_baseline:
        with timed_step("generate_baseline"):
            generated["baseline"] = generate_baseline(
                orig_df=orig_df,
                num_rows=effective_num_rows,
                n_workers=args.n_workers,
            )
            generated["baseline"].to_csv(synthetic_dir / f"synthetic_baseline_{year}.csv", index=False)

    if not args.no_ctgan:
        with timed_step("generate_ctgan"):
            generated["ctgan"] = generate_ctgan(
                orig_df=orig_df,
                num_rows=effective_num_rows,
                ctgan_workers=args.ctgan_workers,
                ctgan_train_rows=args.ctgan_train_rows,
                random_state=args.random_state,
            )
            generated["ctgan"].to_csv(synthetic_dir / f"synthetic_ctgan_{year}.csv", index=False)

    if args.run_chow_liu:
        with timed_step("generate_chow_liu"):
            generated["chow_liu"] = generate_chow_liu(
                orig_df=orig_df,
                num_rows=effective_num_rows,
                random_state=args.random_state,
                train_rows=args.chow_liu_train_rows,
                max_tree_features=args.chow_liu_max_tree_features,
                max_cardinality=args.chow_liu_max_cardinality,
            )
            generated["chow_liu"].to_csv(synthetic_dir / f"synthetic_chow_liu_{year}.csv", index=False)

    if args.run_sdv_gaussian_copula:
        with timed_step("generate_sdv_gaussian_copula"):
            generated["sdv_gaussian_copula"] = generate_sdv_gaussian_copula(
                orig_df=orig_df,
                num_rows=effective_num_rows,
                random_state=args.random_state,
                train_rows=args.sdv_train_rows,
            )
            generated["sdv_gaussian_copula"].to_csv(
                synthetic_dir / f"synthetic_sdv_gaussian_copula_{year}.csv",
                index=False,
            )

    if args.run_synthpop:
        for method in args.synthpop_methods:
            method_slug = slugify(method)
            label = f"synthpop_{method_slug}"
            with timed_step(f"generate_{label}"):
                generated[label] = generate_synthpop_method(
                    orig_df=orig_df,
                    num_rows=effective_num_rows,
                    random_state=args.random_state,
                    synthpop_seed=args.synthpop_seed,
                    synthpop_method=method,
                    synthpop_maxfaclevels=args.synthpop_maxfaclevels,
                    synthpop_train_rows=args.synthpop_train_rows,
                    synthpop_high_cardinality_exclude=args.synthpop_high_cardinality_exclude,
                    synthpop_install_missing=args.synthpop_install_missing,
                    synthpop_cran_mirror=args.synthpop_cran_mirror,
                )
                generated[label].to_csv(synthetic_dir / f"synthetic_{label}_{year}.csv", index=False)

    if not args.no_original_control:
        with timed_step("generate_original_sample_control"):
            generated["original_sample_control"] = generate_original_sample_control(
                orig_df=orig_df,
                num_rows=effective_num_rows,
                random_state=args.random_state,
            )
            generated["original_sample_control"].to_csv(
                synthetic_dir / f"synthetic_original_sample_control_{year}.csv",
                index=False,
            )

    if not generated:
        raise RuntimeError("No generators selected. Disable fewer --no-* flags or enable an optional comparator.")

    if args.smoke_test_only:
        smoke_rows = []
        for name, df in generated.items():
            smoke_rows.append(
                {
                    "year": year,
                    "generator": name,
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "columns_match_lsm_features": list(df.columns) == feature_names,
                    "synthetic_csv": str(synthetic_dir / f"synthetic_{slugify(name)}_{year}.csv"),
                }
            )
        smoke_df = pd.DataFrame(smoke_rows)
        smoke_path = out_dir / f"smoke_test_generators_{year}.csv"
        smoke_df.to_csv(smoke_path, index=False)
        log(f"Saved smoke-test generator table: {smoke_path}")
        metadata["end_time"] = now_string()
        metadata_path = out_dir / f"run_metadata_{year}.json"
        metadata_path.write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")
        save_timing_and_manifest(year, out_dir)
        log(f"Smoke-test complete. Outputs written to: {out_dir}")
        return

    ups_by_generator: dict[str, np.ndarray] = {}
    generator_summary: dict[str, dict[str, Any]] = {
        name: {"year": year, "n_rows": len(df), "n_cols": len(df.columns)}
        for name, df in generated.items()
    }

    for name, df in generated.items():
        with timed_step(f"compute_upsilon_{name}"):
            ups = compute_upsilon_for_df(
                year=year,
                generator=name,
                df=df,
                model_path=model_path,
                feature_names=feature_names,
                n_workers=args.n_workers,
                out_dir=out_dir,
            )
            ups_by_generator[name] = ups
            generator_summary[name]["upsilon_mean"] = float(np.nanmean(ups))
            generator_summary[name]["upsilon_median"] = float(np.nanmedian(ups))

        with timed_step(f"plot_upsilon_{name}"):
            plot_upsilon_distribution(
                year=year,
                generator=name,
                ups=ups,
                plot_dir=plot_dir,
            )

    with timed_step("plot_upsilon_overlay"):
        plot_upsilon_overlay(
            year=year,
            ups_by_generator=ups_by_generator,
            plot_dir=plot_dir,
        )

    for name, df in generated.items():
        with timed_step(f"compute_similarity_{name}"):
            sim_stats = row_similarity_blockplot(
                orig_df,
                df,
                year=year,
                generator=name,
                metric=args.similarity_metric,
                max_rows=args.similarity_max_rows,
                random_state=args.random_state,
                missing_token="__MISSING__",
                plot_path=plot_dir / f"novelty_row_similarity_{slugify(name)}_{year}.png",
            )

            for k, v in sim_stats.items():
                if k == "block_means":
                    for kk, vv in v.items():
                        generator_summary[name][f"block_mean_{kk}"] = vv
                elif isinstance(v, (int, float, str)):
                    generator_summary[name][k] = v

    with timed_step("plot_column_distribution"):
        plot_single_column_distribution(
            year=year,
            orig_df=orig_df,
            generated=generated,
            column_index=args.column_index,
            plot_dir=plot_dir,
        )

    with timed_step("plot_generator_map"):
        plot_generator_map(
            year=year,
            summary=generator_summary,
            plot_dir=plot_dir,
        )

    with timed_step("save_results_dataframes"):
        make_results_dataframes(year, out_dir)

        summary_extra_df = pd.DataFrame.from_dict(generator_summary, orient="index")
        summary_extra_df.insert(0, "generator", summary_extra_df.index)
        summary_extra_df.reset_index(drop=True, inplace=True)
        summary_extra_path = out_dir / f"synthetic_eval_generator_summary_{year}.csv"
        summary_extra_df.to_csv(summary_extra_path, index=False)

        try:
            summary_extra_df.to_parquet(
                out_dir / f"synthetic_eval_generator_summary_{year}.parquet",
                index=False,
            )
        except Exception:
            pass

        log(f"Saved generator summary dataframe: {summary_extra_path}")

    with timed_step("save_timing_and_plot_manifest"):
        save_timing_and_manifest(year, out_dir)

    metadata["end_time"] = now_string()
    metadata_path = out_dir / f"run_metadata_{year}.json"
    metadata_path.write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")
    log(f"Saved metadata: {metadata_path}")

    expected_plots = [
        plot_dir / f"upsilon_lsm_{year}.png",
        plot_dir / f"upsilon_baseline_{year}.png",
        plot_dir / f"upsilon_original_sample_control_{year}.png",
        plot_dir / f"upsilon_all_generators_{year}.png",
        plot_dir / f"novelty_row_similarity_lsm_{year}.png",
        plot_dir / f"novelty_row_similarity_baseline_{year}.png",
        plot_dir / f"novelty_row_similarity_original_sample_control_{year}.png",
        plot_dir / f"generator_map_upsilon_vs_inverse_nearest_real_similarity_{year}.png",
    ]

    if not args.no_ctgan:
        expected_plots.extend(
            [
                plot_dir / f"upsilon_ctgan_{year}.png",
                plot_dir / f"novelty_row_similarity_ctgan_{year}.png",
            ]
        )

    if args.run_chow_liu:
        expected_plots.extend(
            [
                plot_dir / f"upsilon_chow_liu_{year}.png",
                plot_dir / f"novelty_row_similarity_chow_liu_{year}.png",
            ]
        )

    if args.run_sdv_gaussian_copula:
        expected_plots.extend(
            [
                plot_dir / f"upsilon_sdv_gaussian_copula_{year}.png",
                plot_dir / f"novelty_row_similarity_sdv_gaussian_copula_{year}.png",
            ]
        )

    if args.run_synthpop:
        for method in args.synthpop_methods:
            label = f"synthpop_{slugify(method)}"
            expected_plots.extend(
                [
                    plot_dir / f"upsilon_{label}_{year}.png",
                    plot_dir / f"novelty_row_similarity_{label}_{year}.png",
                ]
            )

    missing = [str(p) for p in expected_plots if not p.exists()]
    if missing:
        log("WARNING: Missing expected plots:")
        for p in missing:
            log(f"  {p}")
    else:
        log("All expected plots were saved.")

    open_figs = plt.get_fignums()
    if open_figs:
        log(f"WARNING: closing {len(open_figs)} remaining open matplotlib figures.")
        plt.close("all")

    log(f"Done. Outputs written to: {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LSynth synthetic-data generation/evaluation for a GSS year."
    )

    p.add_argument(
        "year",
        type=int,
        help="GSS year. Uses ./datasets/gss_<year>.csv and ./datasets/gss_<year>.pkl.gz.",
    )

    p.add_argument(
        "--datasets-dir",
        default="./datasets",
        help="Directory containing gss_<year>.csv and gss_<year>.pkl.gz.",
    )

    p.add_argument(
        "--output-dir",
        default="synthetic_eval_outputs",
        help="Output directory.",
    )

    p.add_argument(
        "--num-rows",
        type=int,
        default=1000,
        help="Number of synthetic rows per generator.",
    )

    p.add_argument(
        "--n-workers",
        type=int,
        default=N_WORKERS_DEFAULT,
        help="General worker/thread count. Default: 120.",
    )

    p.add_argument(
        "--ctgan-workers",
        type=int,
        default=CTGAN_WORKERS_DEFAULT,
        help="Worker count for CTGAN only. Default: 10.",
    )

    p.add_argument(
        "--ctgan-train-rows",
        type=int,
        default=50,
        help="Number of original rows used to train CTGAN. Default: 50.",
    )

    p.add_argument(
        "--no-ctgan",
        action="store_true",
        help="Skip CTGAN generation and downstream CTGAN metrics.",
    )

    p.add_argument(
        "--no-lsm",
        action="store_true",
        help="Skip LSM generation. Useful for smoke-testing optional comparators only.",
    )

    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip independent-column baseline generation.",
    )

    p.add_argument(
        "--no-original-control",
        action="store_true",
        help="Skip the original-sample control.",
    )

    p.add_argument(
        "--run-chow-liu",
        action="store_true",
        help="Enable pure-Python categorical Chow-Liu tree comparator.",
    )

    p.add_argument(
        "--chow-liu-train-rows",
        type=int,
        default=CHOW_LIU_TRAIN_ROWS_DEFAULT,
        help=(
            "Rows used to learn the Chow-Liu tree. Use 0 or a negative value "
            f"for all rows. Default: {CHOW_LIU_TRAIN_ROWS_DEFAULT}."
        ),
    )

    p.add_argument(
        "--chow-liu-max-tree-features",
        type=int,
        default=CHOW_LIU_MAX_TREE_FEATURES_DEFAULT,
        help=(
            "Maximum number of columns included in the expensive pairwise-MI "
            "Chow-Liu tree. Remaining columns are sampled independently so the "
            "full schema is preserved. Use 0 or a negative value for all eligible "
            f"columns. Default: {CHOW_LIU_MAX_TREE_FEATURES_DEFAULT}."
        ),
    )

    p.add_argument(
        "--chow-liu-max-cardinality",
        type=int,
        default=CHOW_LIU_MAX_CARDINALITY_DEFAULT,
        help=(
            "Exclude columns with more than this many observed levels from the "
            "Chow-Liu tree and sample them independently. Use 0 or a negative "
            f"value for no cardinality cap. Default: {CHOW_LIU_MAX_CARDINALITY_DEFAULT}."
        ),
    )

    p.add_argument(
        "--run-sdv-gaussian-copula",
        action="store_true",
        help="Enable optional SDV GaussianCopulaSynthesizer comparator.",
    )

    p.add_argument(
        "--sdv-train-rows",
        type=int,
        default=5000,
        help="Rows used to fit SDV Gaussian Copula. Use 0 or negative for all rows. Default: 5000.",
    )

    p.add_argument(
        "--run-synthpop",
        action="store_true",
        help="Enable R synthpop comparators. Disabled by default.",
    )

    p.add_argument(
        "--synthpop-methods",
        nargs="+",
        default=SYNTHPOP_METHODS_DEFAULT,
        help="One or more synthpop methods, e.g. cart ranger rf. Default: cart.",
    )

    p.add_argument(
        "--synthpop-seed",
        type=int,
        default=None,
        help="Seed for synthpop. Default: use --random-state.",
    )

    p.add_argument(
        "--synthpop-maxfaclevels",
        type=int,
        default=SYNTHPOP_MAXFACLEVELS_DEFAULT,
        help=f"Maximum factor levels for synthpop. Default: {SYNTHPOP_MAXFACLEVELS_DEFAULT}.",
    )

    p.add_argument(
        "--synthpop-train-rows",
        type=int,
        default=1000,
        help=(
            "Rows used to fit synthpop before resampling to --num-rows. "
            "Use 0 or a negative value to use the full original dataframe. Default: 1000."
        ),
    )

    p.add_argument(
        "--synthpop-high-cardinality-exclude",
        nargs="*",
        default=SYNTHPOP_HIGH_CARDINALITY_DEFAULT,
        help=(
            "Column names excluded as predictors in synthpop's predictor matrix. "
            "Defaults to common high-cardinality GSS occupation/industry columns."
        ),
    )

    p.add_argument(
        "--no-synthpop-install-missing",
        dest="synthpop_install_missing",
        action="store_false",
        help="Do not attempt to install the R synthpop package if missing.",
    )
    p.set_defaults(synthpop_install_missing=True)

    p.add_argument(
        "--synthpop-cran-mirror",
        default="https://cloud.r-project.org",
        help="CRAN mirror used if synthpop installation is attempted.",
    )

    p.add_argument(
        "--smoke-test-only",
        action="store_true",
        help=(
            "Generate selected datasets, verify shapes/columns, save synthetic CSVs, "
            "and stop before Upsilon, novelty, and plotting."
        ),
    )

    p.add_argument(
        "--smoke-test-rows",
        type=int,
        default=SMOKE_TEST_ROWS_DEFAULT,
        help=f"Rows per selected generator in --smoke-test-only mode. Default: {SMOKE_TEST_ROWS_DEFAULT}.",
    )

    p.add_argument(
        "--similarity-max-rows",
        type=int,
        default=2000,
        help="Maximum rows per side for novelty row-similarity calculation.",
    )

    p.add_argument(
        "--similarity-metric",
        choices=["cosine", "match"],
        default="cosine",
        help="Novelty row-similarity metric.",
    )

    p.add_argument(
        "--column-index",
        type=int,
        default=30,
        help="Column index for marginal distribution plot.",
    )

    p.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
