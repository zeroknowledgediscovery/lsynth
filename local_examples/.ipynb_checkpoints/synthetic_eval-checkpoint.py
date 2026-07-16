#!/usr/bin/env python3
"""
Run synthetic tabular-data generation and evaluation from a TOML config.

This script consolidates the notebook workflow into a reproducible CLI:
  - generate LSM, BASELINE, CTGAN, synthpop, and permutation-control data
  - compute Upsilon using an lsynth model
  - train a real-vs-synthetic discriminator and report AUC
  - compute one-hot row-similarity novelty statistics
  - save CSV/JSON metrics and optional plots

Typical use:
    python synthetic_eval.py --config synthetic_eval_config.toml

TOML is used instead of YAML because it is readable, sectioned, commentable,
and avoids YAML's indentation/type coercion surprises. Python 3.11+ reads TOML
with the standard-library tomllib module. On older Python, install tomli.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------- config -------------------------------------


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "TOML config support requires Python 3.11+ or `pip install tomli`."
            ) from exc

    with path.open("rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "unnamed"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------- optional dependencies ---------------------------


def require_lsynth():
    try:
        from lsynth import compute_upsilon, generate_syndata
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import `lsynth`. Install it or run from the environment where "
            "compute_upsilon/generate_syndata are available."
        ) from exc
    return compute_upsilon, generate_syndata


def make_one_hot_encoder(*, sparse_output: bool, dtype: Any | None = None):
    from sklearn.preprocessing import OneHotEncoder

    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if dtype is not None:
        kwargs["dtype"] = dtype

    try:
        return OneHotEncoder(sparse_output=sparse_output, **kwargs)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(sparse=sparse_output, **kwargs)


def configure_matplotlib(show: bool) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)


# ------------------------------ IO ----------------------------------------


def read_original_data(cfg: dict[str, Any]) -> pd.DataFrame:
    data_csv = cfg_get(cfg, "paths.data_csv")
    if not data_csv:
        raise ValueError("Missing required config key: paths.data_csv")

    keep_default_na = bool(cfg_get(cfg, "run.keep_default_na", False))
    log(f"Reading original data: {data_csv}")
    return pd.read_csv(data_csv, keep_default_na=keep_default_na)


def ensure_output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg_get(cfg, "paths.output_dir", "synthetic_eval_out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_data").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def save_dataframe(df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    df.to_csv(path, index=False)


# ---------------------------- generation ----------------------------------


def align_columns(orig_df: pd.DataFrame, syn_df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return syn_df with original column order when safe; otherwise fail clearly."""
    if list(syn_df.columns) == list(orig_df.columns):
        return syn_df

    if set(syn_df.columns) == set(orig_df.columns):
        return syn_df.loc[:, list(orig_df.columns)]

    if len(syn_df.columns) == len(orig_df.columns):
        warnings.warn(
            f"{name}: generated columns did not match original names, but column counts "
            "match. Renaming generated columns to the original schema.",
            RuntimeWarning,
        )
        syn_df = syn_df.copy()
        syn_df.columns = orig_df.columns
        return syn_df

    raise ValueError(
        f"{name}: generated dataframe has incompatible columns. "
        f"orig={len(orig_df.columns)} generated={len(syn_df.columns)}"
    )


def maybe_cast_to_int(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    return df.astype(int)


def generate_lsm(cfg: dict[str, Any], orig_df: pd.DataFrame) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    model_path = cfg_get(cfg, "paths.model_path")
    if not model_path:
        raise ValueError("LSM generation requires paths.model_path")

    num = int(cfg_get(cfg, "run.num_rows", 1000))
    n_workers = int(cfg_get(cfg, "run.n_workers", 1))
    log(f"Generating {num} rows via LSM")
    syn = generate_syndata(
        num=num,
        model_path=model_path,
        gen_algorithm="LSM",
        n_workers=n_workers,
    )
    return align_columns(orig_df, syn, name="LSM")


def generate_baseline(cfg: dict[str, Any], orig_df: pd.DataFrame) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    num = int(cfg_get(cfg, "run.num_rows", 1000))
    n_workers = int(cfg_get(cfg, "run.n_workers", 1))
    log(f"Generating {num} rows via BASELINE")
    syn = generate_syndata(
        num=num,
        gen_algorithm="BASELINE",
        orig_df=orig_df,
        n_workers=n_workers,
    )
    return align_columns(orig_df, syn, name="BASELINE")


def generate_ctgan(cfg: dict[str, Any], orig_df: pd.DataFrame) -> pd.DataFrame:
    _, generate_syndata = require_lsynth()
    num = int(cfg_get(cfg, "run.num_rows", 1000))
    n_workers = int(cfg_get(cfg, "run.n_workers", 1))
    random_state = int(cfg_get(cfg, "run.random_state", 123))
    train_rows = positive_int_or_none(cfg_get(cfg, "generators.ctgan.train_rows", 50))

    train_df = orig_df
    if train_rows is not None:
        n = min(train_rows, len(orig_df))
        train_df = orig_df.sample(n=n, random_state=random_state)

    log(f"Generating {num} rows via CTGAN using {len(train_df)} training rows")
    syn = generate_syndata(
        num=num,
        gen_algorithm="CTGAN",
        orig_df=train_df,
        n_workers=n_workers,
    )
    return align_columns(orig_df, syn, name="CTGAN")


def generate_permutation(cfg: dict[str, Any], orig_df: pd.DataFrame) -> pd.DataFrame:
    num = int(cfg_get(cfg, "run.num_rows", 1000))
    random_state = int(cfg_get(cfg, "run.random_state", 123))
    replace = bool(cfg_get(cfg, "generators.permutation.replace", False))
    if num > len(orig_df) and not replace:
        warnings.warn(
            "permutation: requested more rows than original data has; using replace=True.",
            RuntimeWarning,
        )
        replace = True
    log(f"Generating {num} rows via permutation/original-row sample")
    return orig_df.sample(n=num, replace=replace, random_state=random_state).reset_index(drop=True)


def generate_synthpop(cfg: dict[str, Any], orig_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "synthpop generation requires rpy2 and an R installation. Disable "
            "`synthpop` in [generators].enabled if you do not need it."
        ) from exc

    seed = int(cfg_get(cfg, "generators.synthpop.seed", cfg_get(cfg, "run.random_state", 123)))
    method = str(cfg_get(cfg, "generators.synthpop.method", "cart"))
    maxfaclevels = int(cfg_get(cfg, "generators.synthpop.maxfaclevels", 300))
    install_missing = bool(cfg_get(cfg, "generators.synthpop.install_missing", True))
    cran_mirror = str(cfg_get(cfg, "generators.synthpop.cran_mirror", "https://cloud.r-project.org"))
    high_vars = list(cfg_get(cfg, "generators.synthpop.high_cardinality_exclude", []))

    log("Generating rows via R synthpop")
    ro.r(f'options(repos = c(CRAN="{cran_mirror}"))')
    if install_missing:
        ro.r('if (!requireNamespace("synthpop", quietly=TRUE)) install.packages("synthpop")')
    ro.r("library(synthpop)")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(orig_df.copy())
        ro.globalenv["df"] = r_df
        ro.globalenv["high_vars"] = ro.StrVector(high_vars)
        ro.globalenv["seed_py"] = seed
        ro.globalenv["method_py"] = method
        ro.globalenv["maxfaclevels_py"] = maxfaclevels

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
    syn = maybe_cast_to_int(syn, bool(cfg_get(cfg, "generators.synthpop.cast_to_int", False)))
    return align_columns(orig_df, syn, name="synthpop")


def generate_all(cfg: dict[str, Any], orig_df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    enabled = [str(g).lower() for g in cfg_get(cfg, "generators.enabled", ["lsm", "baseline"])]
    overwrite = bool(cfg_get(cfg, "run.overwrite", True))
    continue_on_error = bool(cfg_get(cfg, "run.continue_on_error", True))

    dispatch = {
        "lsm": generate_lsm,
        "baseline": generate_baseline,
        "ctgan": generate_ctgan,
        "synthpop": generate_synthpop,
        "pop": generate_synthpop,
        "permutation": generate_permutation,
        "perm": generate_permutation,
    }

    generated: dict[str, pd.DataFrame] = {}
    for gen in enabled:
        if gen not in dispatch:
            raise ValueError(f"Unknown generator: {gen}")
        label = "synthpop" if gen == "pop" else "permutation" if gen == "perm" else gen
        try:
            syn = dispatch[gen](cfg, orig_df)
            syn = align_columns(orig_df, syn, name=label)
            generated[label] = syn
            save_dataframe(syn, out_dir / "synthetic_data" / f"{slugify(label)}.csv", overwrite=overwrite)
        except Exception as exc:
            if not continue_on_error:
                raise
            log(f"WARNING: skipped {label}: {exc}")
    return generated


# ----------------------------- metrics ------------------------------------


def compute_upsilon_metrics(
    cfg: dict[str, Any],
    name: str,
    df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    compute_upsilon, _ = require_lsynth()
    model_path = cfg_get(cfg, "paths.model_path")
    if not model_path:
        raise ValueError("Upsilon requires paths.model_path")

    sample_rows = positive_int_or_none(cfg_get(cfg, "upsilon.sample_rows", cfg_get(cfg, "run.num_rows", 1000)))
    random_state = int(cfg_get(cfg, "run.random_state", 123))
    n_workers = int(cfg_get(cfg, "run.n_workers", 1))

    eval_df = df
    if sample_rows is not None:
        n = min(sample_rows, len(df))
        eval_df = df.sample(n=n, random_state=random_state)

    log(f"Computing Upsilon for {name} on {len(eval_df)} rows")
    ups, _ = compute_upsilon(eval_df, model_path=model_path, n_workers=n_workers)
    ups = np.asarray(ups, dtype=float)

    pd.DataFrame({"upsilon": ups}).to_csv(out_dir / f"upsilon_{slugify(name)}.csv", index=False)

    return {
        "n": int(np.isfinite(ups).sum()),
        "mean": float(np.nanmean(ups)),
        "median": float(np.nanmedian(ups)),
        "std": float(np.nanstd(ups)),
        "p05": float(np.nanquantile(ups, 0.05)),
        "p95": float(np.nanquantile(ups, 0.95)),
    }


def real_vs_synth_auc(
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 123,
    cv_folds: int | None = None,
    max_rows: int | None = None,
    max_iter: int = 2000,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline

    if list(orig_df.columns) != list(syn_df.columns):
        raise ValueError("orig_df and syn_df must have identical columns in the same order.")

    if max_rows is not None:
        n0 = min(len(orig_df), max_rows)
        n1 = min(len(syn_df), max_rows)
        orig_df = orig_df.sample(n=n0, random_state=random_state)
        syn_df = syn_df.sample(n=n1, random_state=random_state)

    X_real = orig_df.copy()
    X_synth = syn_df.copy()
    for c in X_real.columns:
        X_real[c] = X_real[c].astype("string")
        X_synth[c] = X_synth[c].astype("string")

    X = pd.concat([X_real, X_synth], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_real), dtype=int), np.ones(len(X_synth), dtype=int)])

    pipe = Pipeline(
        steps=[
            ("oh", make_one_hot_encoder(sparse_output=True)),
            ("clf", LogisticRegression(max_iter=max_iter, n_jobs=-1)),
        ]
    )

    if cv_folds is not None and cv_folds >= 2:
        safe_folds = min(int(cv_folds), len(X_real), len(X_synth))
        if safe_folds < 2:
            return {"mode": "cv", "error": "not enough rows for cross-validation"}
        cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=random_state)
        aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return {
            "mode": "cv",
            "cv_folds": safe_folds,
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "aucs": aucs,
            "n_real": len(X_real),
            "n_synth": len(X_synth),
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)

    return {
        "mode": "holdout",
        "test_size": test_size,
        "auc": float(auc),
        "n_real": len(X_real),
        "n_synth": len(X_synth),
    }


def compute_auc_metrics(cfg: dict[str, Any], orig_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict[str, Any]:
    return real_vs_synth_auc(
        orig_df,
        syn_df,
        test_size=float(cfg_get(cfg, "auc.test_size", 0.25)),
        random_state=int(cfg_get(cfg, "run.random_state", 123)),
        cv_folds=positive_int_or_none(cfg_get(cfg, "auc.cv_folds", 5)),
        max_rows=positive_int_or_none(cfg_get(cfg, "auc.max_rows", 20000)),
        max_iter=int(cfg_get(cfg, "auc.max_iter", 2000)),
    )


def row_similarity_blockplot(
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    *,
    metric: str = "cosine",
    max_rows: int | None = 2000,
    random_state: int = 123,
    missing_token: str = "__MISSING__",
    plot_path: Path | None = None,
    show_plot: bool = False,
    save_full_matrix_plot: bool = True,
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

    if plot_path is not None or show_plot:
        import matplotlib.pyplot as plt

        if save_full_matrix_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im0 = axes[0].imshow(G, aspect="auto")
            axes[0].axhline(n - 0.5)
            axes[0].axvline(n - 0.5)
            axes[0].set_title(f"Row-row similarity ({metric})")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            ax_block = axes[1]
        else:
            fig, ax_block = plt.subplots(1, 1, figsize=(5, 4))

        im1 = ax_block.imshow(block_means, aspect="equal")
        ax_block.set_xticks([0, 1], labels=["Real", "Synth"])
        ax_block.set_yticks([0, 1], labels=["Real", "Synth"])
        ax_block.set_title("2x2 block mean similarity")
        for (i, j), v in np.ndenumerate(block_means):
            ax_block.text(j, i, f"{v:.3f}", ha="center", va="center")
        fig.colorbar(im1, ax=ax_block, fraction=0.046, pad=0.04)
        fig.tight_layout()
        if plot_path is not None:
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    off_block = G[:n, n:]
    nn = off_block.max(axis=0)
    nearest_median = float(np.median(nn))

    return {
        "n_rows_each": n,
        "nearest_real_similarity_mean": float(nn.mean()),
        "nearest_real_similarity_median": nearest_median,
        "nearest_real_similarity_p95": float(np.quantile(nn, 0.95)),
        "inverse_nearest_real_similarity_median": float(1.0 / nearest_median) if nearest_median > 0 else math.inf,
        "missing_token_used": missing_token,
        "block_means": {
            "real_real": rr,
            "real_synth": rs,
            "synth_real": sr,
            "synth_synth": ss,
        },
    }


def compute_similarity_metrics(
    cfg: dict[str, Any],
    name: str,
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    plot_enabled = bool(cfg_get(cfg, "plots.enabled", True)) and bool(cfg_get(cfg, "similarity.plot", True))
    plot_path = out_dir / "plots" / f"similarity_{slugify(name)}.png" if plot_enabled else None

    return row_similarity_blockplot(
        orig_df,
        syn_df,
        metric=str(cfg_get(cfg, "similarity.metric", "cosine")),
        max_rows=positive_int_or_none(cfg_get(cfg, "similarity.max_rows", 2000)),
        random_state=int(cfg_get(cfg, "run.random_state", 123)),
        missing_token=str(cfg_get(cfg, "similarity.missing_token", "__MISSING__")),
        plot_path=plot_path,
        show_plot=bool(cfg_get(cfg, "plots.show", False)),
        save_full_matrix_plot=bool(cfg_get(cfg, "similarity.save_full_matrix_plot", True)),
    )


# ----------------------------- plots --------------------------------------


def plot_upsilon_histograms(
    ups_by_name: dict[str, np.ndarray],
    out_dir: Path,
    *,
    show: bool = False,
) -> None:
    if not ups_by_name:
        return
    import matplotlib.pyplot as plt

    for name, ups in ups_by_name.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(ups[np.isfinite(ups)], bins=40)
        ax.set_title(f"Upsilon distribution: {name}")
        ax.set_xlabel("Upsilon")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"upsilon_{slugify(name)}.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ups in ups_by_name.items():
        ax.hist(ups[np.isfinite(ups)], bins=40, histtype="step", label=name)
    ax.set_title("Upsilon distributions")
    ax.set_xlabel("Upsilon")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "upsilon_all_generators.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_generator_map(metrics: dict[str, Any], out_dir: Path, *, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    names: list[str] = []
    x: list[float] = []
    y: list[float] = []

    for name, m in metrics.items():
        ups = m.get("upsilon", {})
        sim = m.get("similarity", {})
        if "median" not in ups or "inverse_nearest_real_similarity_median" not in sim:
            continue
        names.append(name)
        x.append(float(sim["inverse_nearest_real_similarity_median"]))
        y.append(float(ups["median"]))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    ax.set_xlabel("1 / nearest-real similarity median")
    ax.set_ylabel("Upsilon median")
    ax.set_title("Generator map: Upsilon vs novelty")
    for name, xi, yi in zip(names, x, y):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "generator_map.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_column_distributions(
    cfg: dict[str, Any],
    orig_df: pd.DataFrame,
    generated: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    column_indices = list(cfg_get(cfg, "plots.column_indices", []))
    column_names = list(cfg_get(cfg, "plots.column_names", []))
    max_levels = int(cfg_get(cfg, "plots.column_max_levels", 30))
    show = bool(cfg_get(cfg, "plots.show", False))

    selected: list[str] = []
    for idx in column_indices:
        i = int(idx)
        if 0 <= i < len(orig_df.columns):
            selected.append(str(orig_df.columns[i]))
    selected.extend([str(c) for c in column_names if c in orig_df.columns])
    selected = list(dict.fromkeys(selected))

    if not selected:
        return

    import matplotlib.pyplot as plt

    for col in selected:
        freq = orig_df[col].astype("string").fillna("__MISSING__").value_counts(normalize=True).head(max_levels)
        keep = list(freq.index)
        x = np.arange(len(keep))

        fig, ax = plt.subplots(figsize=(max(8, len(keep) * 0.35), 5))
        ax.plot(x, [freq.get(v, 0.0) for v in keep], marker="o", label="real")
        for name, syn in generated.items():
            sfreq = syn[col].astype("string").fillna("__MISSING__").value_counts(normalize=True)
            ax.plot(x, [sfreq.get(v, 0.0) for v in keep], marker="o", label=name)
        ax.set_xticks(x, labels=keep, rotation=90)
        ax.set_ylabel("frequency")
        ax.set_title(f"Column distribution: {col}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"column_{slugify(col)}.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


# ------------------------------ main --------------------------------------


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    configure_matplotlib(show=bool(cfg_get(cfg, "plots.show", False)))
    out_dir = ensure_output_dir(cfg)
    orig_df = read_original_data(cfg)
    log(f"Original data shape: {orig_df.shape[0]} rows x {orig_df.shape[1]} columns")

    generated = generate_all(cfg, orig_df, out_dir)
    if not generated:
        raise RuntimeError("No synthetic datasets were generated. Check [generators].enabled.")

    metrics: dict[str, Any] = {}
    ups_arrays: dict[str, np.ndarray] = {}
    continue_on_error = bool(cfg_get(cfg, "run.continue_on_error", True))

    for name, syn_df in generated.items():
        metrics[name] = {"n_rows": len(syn_df), "n_cols": len(syn_df.columns)}

        if bool(cfg_get(cfg, "upsilon.enabled", True)):
            try:
                ups_metrics = compute_upsilon_metrics(cfg, name, syn_df, out_dir)
                metrics[name]["upsilon"] = ups_metrics
                ups_csv = out_dir / f"upsilon_{slugify(name)}.csv"
                ups_arrays[name] = pd.read_csv(ups_csv)["upsilon"].to_numpy(dtype=float)
            except Exception as exc:
                metrics[name]["upsilon_error"] = str(exc)
                if not continue_on_error:
                    raise
                log(f"WARNING: Upsilon failed for {name}: {exc}")

        if bool(cfg_get(cfg, "auc.enabled", True)):
            try:
                log(f"Computing real-vs-synthetic AUC for {name}")
                metrics[name]["auc"] = compute_auc_metrics(cfg, orig_df, syn_df)
            except Exception as exc:
                metrics[name]["auc_error"] = str(exc)
                if not continue_on_error:
                    raise
                log(f"WARNING: AUC failed for {name}: {exc}")

        if bool(cfg_get(cfg, "similarity.enabled", True)):
            try:
                log(f"Computing row-similarity novelty for {name}")
                metrics[name]["similarity"] = compute_similarity_metrics(cfg, name, orig_df, syn_df, out_dir)
            except Exception as exc:
                metrics[name]["similarity_error"] = str(exc)
                if not continue_on_error:
                    raise
                log(f"WARNING: similarity failed for {name}: {exc}")

    metrics_json = out_dir / "metrics.json"
    metrics_json.write_text(json.dumps(to_jsonable(metrics), indent=2), encoding="utf-8")

    rows = []
    for name, m in metrics.items():
        row: dict[str, Any] = {"generator": name, "n_rows": m.get("n_rows"), "n_cols": m.get("n_cols")}
        if "upsilon" in m:
            row.update({f"upsilon_{k}": v for k, v in m["upsilon"].items()})
        if "auc" in m:
            auc = m["auc"]
            if auc.get("mode") == "cv":
                row["auc"] = auc.get("auc_mean")
                row["auc_std"] = auc.get("auc_std")
            else:
                row["auc"] = auc.get("auc")
        if "similarity" in m:
            sim = m["similarity"]
            row["nearest_real_similarity_median"] = sim.get("nearest_real_similarity_median")
            row["inverse_nearest_real_similarity_median"] = sim.get("inverse_nearest_real_similarity_median")
            row["nearest_real_similarity_p95"] = sim.get("nearest_real_similarity_p95")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    if bool(cfg_get(cfg, "plots.enabled", True)):
        plot_upsilon_histograms(ups_arrays, out_dir, show=bool(cfg_get(cfg, "plots.show", False)))
        plot_generator_map(metrics, out_dir, show=bool(cfg_get(cfg, "plots.show", False)))
        plot_column_distributions(cfg, orig_df, generated, out_dir)

    log(f"Done. Wrote outputs to: {out_dir}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and evaluate synthetic categorical tabular data.")
    p.add_argument("--config", required=True, help="Path to TOML config file.")
    p.add_argument(
        "--generators",
        nargs="+",
        help="Optional override for [generators].enabled, e.g. --generators lsm baseline ctgan",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_toml(Path(args.config))
    if args.generators:
        cfg.setdefault("generators", {})["enabled"] = args.generators
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
