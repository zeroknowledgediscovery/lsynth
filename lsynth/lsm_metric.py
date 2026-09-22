"""LSYNTH process metric evaluated with the current standalone LSM implementation.

This module replaces the QuasiNet conditional evaluator used by the historical
LSYNTH process-metric example with the standalone C++ LSM trainer and
predict_distribution binding from zeroknowledgediscovery/lsm.

For a fitted LSM G and reference state x, the normalized conditional profile is

    u_G(x,i) = phi_G^i(x_i | x_-i) / max_y phi_G^i(y | x_-i).

For two datasets/models G1 and G2 and a common reference sample E_mu,

    d_mu_hat = mean_{x in E_mu, i} |u_G1(x,i) - u_G2(x,i)|.

Missing target coordinates are excluded from the coordinate average.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import importlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_reference_rows: int
    n_features: int
    n_valid_rows: int


def read_categorical_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )


def common_features(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    feature_limit: int = 0,
) -> list[str]:
    """Shared columns, preserving dataset A's order."""
    cols = [c for c in a.columns if c in b.columns]
    if feature_limit > 0:
        cols = cols[:feature_limit]
    if not cols:
        raise ValueError("The datasets have no shared columns.")
    return cols


def split_fit_holdout(
    df: pd.DataFrame,
    *,
    fit_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie strictly between 0 and 1")
    if len(df) < 4:
        raise ValueError("Need at least four rows to split fit and holdout data")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_fit = int(round(fit_fraction * len(df)))
    n_fit = min(max(n_fit, 2), len(df) - 2)

    fit = df.iloc[order[:n_fit]].reset_index(drop=True)
    holdout = df.iloc[order[n_fit:]].reset_index(drop=True)
    return fit, holdout


def balanced_reference_sample(
    holdout_a: pd.DataFrame,
    holdout_b: pd.DataFrame,
    *,
    rows_per_dataset: int,
    seed: int,
) -> pd.DataFrame:
    """Empirical balanced reference measure mu = 1/2 holdout-A + 1/2 holdout-B."""
    rng = np.random.default_rng(seed)

    if rows_per_dataset <= 0:
        n = min(len(holdout_a), len(holdout_b))
    else:
        n = min(rows_per_dataset, len(holdout_a), len(holdout_b))
    if n < 1:
        raise ValueError("No held-out rows available for the reference sample")

    ia = rng.choice(len(holdout_a), size=n, replace=False)
    ib = rng.choice(len(holdout_b), size=n, replace=False)

    a = holdout_a.iloc[np.sort(ia)].copy()
    b = holdout_b.iloc[np.sort(ib)].copy()
    a.insert(0, "__reference_source__", "A")
    b.insert(0, "__reference_source__", "B")
    return pd.concat([a, b], ignore_index=True)


def model_ready(model_dir: str | Path) -> bool:
    model_dir = Path(model_dir)
    return (
        (model_dir / "trees" / "binary").is_dir()
        and any((model_dir / "trees" / "binary").glob("tree_*.bin"))
        and (model_dir / "source_maps" / "json_shards").is_dir()
    )


def train_lsm(
    train_df: pd.DataFrame,
    *,
    lsm_root: str | Path,
    model_dir: str | Path,
    train_csv: str | Path,
    alpha: float = 0.1,
    threads: int = 12,
    subset_mode: str = "auto",
    max_exact_levels: int = 20,
    fast_levels: int = 16,
    overwrite: bool = False,
) -> Path:
    """Train a complete standalone LSM on a categorical DataFrame."""
    lsm_root = Path(lsm_root)
    model_dir = Path(model_dir)
    train_csv = Path(train_csv)
    binary = lsm_root / "bin" / "LSM"

    if not binary.exists():
        raise FileNotFoundError(
            f"LSM trainer not found: {binary}. Build dev_ixc with ./compile.sh first."
        )

    if model_ready(model_dir) and not overwrite:
        return model_dir

    if overwrite and model_dir.exists():
        shutil.rmtree(model_dir)

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)

    cmd = [
        str(binary),
        str(train_csv),
        str(alpha),
        str(model_dir),
        "--threads",
        str(threads),
        "--subset-mode",
        str(subset_mode),
        "--max-exact-levels",
        str(max_exact_levels),
        "--fast-levels",
        str(fast_levels),
    ]
    subprocess.run(cmd, check=True)

    if not model_ready(model_dir):
        raise RuntimeError(f"Training completed but model is not inference-ready: {model_dir}")
    return model_dir


def load_predict_distribution(lsm_root: str | Path):
    bindir = str((Path(lsm_root) / "bin").resolve())
    if bindir not in sys.path:
        sys.path.insert(0, bindir)
    return importlib.import_module("predict_distribution")


def discover_tree_ids(model_dir: str | Path) -> list[int]:
    import re

    pat = re.compile(r"tree_(\d+)\.bin$")
    ids = []
    for p in (Path(model_dir) / "trees" / "binary").glob("tree_*.bin"):
        m = pat.fullmatch(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def normalized_lsm_profile_for_row(
    row: np.ndarray,
    *,
    model_dir: str | Path,
    predict_distribution,
    tree_ids: Sequence[int],
    eps_floor: float = 0.0,
) -> np.ndarray:
    """Evaluate u_G(x,i) using the standalone LSM conditional distributions."""
    model_dir = Path(model_dir)
    distributions = predict_distribution.predict_distributions(
        str(model_dir / "trees" / "binary"),
        np.asarray(row, dtype=str),
        raw=True,
        run_dir=str(model_dir),
        tree_ids=list(tree_ids),
    )

    profile = np.full(len(row), np.nan, dtype=float)

    for i in tree_ids:
        if i >= len(row):
            continue

        observed = str(row[i])
        # In standalone LSM semantics, "" is missing evidence rather than a
        # categorical target value, so it is excluded from the metric.
        if observed == "":
            continue

        distribution = distributions[i]
        if distribution is None:
            continue

        dist = {str(k): float(v) for k, v in dict(distribution).items()}
        maximum = max(dist.values(), default=0.0)

        if eps_floor > 0:
            maximum = max(maximum, eps_floor)
            observed_probability = max(dist.get(observed, 0.0), eps_floor)
        else:
            if maximum <= 0:
                continue
            observed_probability = dist.get(observed, 0.0)

        profile[i] = observed_probability / maximum

    return profile


def evaluate_lsm_profiles(
    reference_df: pd.DataFrame,
    *,
    model_dir: str | Path,
    lsm_root: str | Path,
    profile_workers: int = 1,
    eps_floor: float = 0.0,
    progress: bool = True,
) -> np.ndarray:
    """Evaluate the normalized profile matrix on one common reference sample."""
    pred = load_predict_distribution(lsm_root)
    tree_ids = discover_tree_ids(model_dir)
    if not tree_ids:
        raise RuntimeError(f"No trained trees found in {model_dir}")

    X = reference_df.to_numpy(dtype=str)

    def one(row):
        return normalized_lsm_profile_for_row(
            row,
            model_dir=model_dir,
            predict_distribution=pred,
            tree_ids=tree_ids,
            eps_floor=eps_floor,
        )

    if profile_workers <= 1:
        iterator = X
        if progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(X, total=len(X), desc=f"Profiles {Path(model_dir).name}")
            except Exception:
                pass
        rows = [one(row) for row in iterator]
    else:
        with ThreadPoolExecutor(max_workers=profile_workers) as executor:
            iterator = executor.map(one, X)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(
                        iterator,
                        total=len(X),
                        desc=f"Profiles {Path(model_dir).name}",
                    )
                except Exception:
                    pass
            rows = list(iterator)

    return np.vstack(rows)


def row_profile_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Profile shapes differ: {a.shape} vs {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    counts = valid.sum(axis=1)
    sums = np.nansum(np.abs(a - b), axis=1)
    out = np.full(len(a), np.nan, dtype=float)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def coordinate_contributions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"Profile shapes differ: {a.shape} vs {b.shape}")
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(a - b), axis=0)


def bootstrap_metric(
    row_distances: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 20260922,
) -> MetricResult:
    values = np.asarray(row_distances, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite row-level LSYNTH distances")

    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])

    return MetricResult(
        estimate=estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        n_reference_rows=len(row_distances),
        n_features=0,
        n_valid_rows=len(values),
    )
