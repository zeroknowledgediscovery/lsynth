#!/usr/bin/env python3
"""
Streamlined synthetic-data evaluation script converted from the no-synthpop notebook.

Changes from the notebook version:
  - Replaces every notebook %%time cell magic with printed start/end timestamps.
  - Uses N_WORKERS = 120 for lsynth n_workers and sklearn n_jobs.
  - Keeps synthpop/rpy2 removed.
  - Saves plots to PLOT_DIR and never calls plt.show().
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

N_WORKERS = 120
PLOT_DIR = Path(os.environ.get("SYNTHETIC_EVAL_PLOT_DIR", "synthetic_eval_plots"))

# Force a non-interactive backend so this can run on servers/headless nodes.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)


for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_env_var] = str(N_WORKERS)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def timed_section(label: str):
    start = time.perf_counter()
    print(f"\n[{_now()}] START {label}", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{_now()}] END   {label} | elapsed={elapsed:.3f}s", flush=True)


def save_figure(fig, filename: str) -> Path:
    """Save a matplotlib/seaborn figure and close it."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[{_now()}] SAVED plot: {out_path}", flush=True)
    return out_path


def save_current_figure(filename: str) -> Path:
    import matplotlib.pyplot as plt
    return save_figure(plt.gcf(), filename)


def save_seaborn_grid(grid, filename: str) -> Path:
    return save_figure(grid.fig, filename)

with timed_section('Cell 01'):
    import pandas as pd
    import numpy as np
    from lsynth import compute_upsilon, generate_syndata
    import seaborn as sns

    #model_path='./modelHRS72p05_train.gz'
    #DATA="./HRSvar72_test.csv"
    DATA='../datasets/gss_2018.csv'
    model_path="../datasets/gss_2018.joblib"
    orig_df=pd.read_csv(DATA,keep_default_na=False)


# # Generate synthetic data

with timed_section('Cell 02: Generate synthetic data'):
    # 1. Generate synthetic data with a chosen generator
    df_lsm = generate_syndata(
        num=1000,
        model_path=model_path,
        gen_algorithm="LSM",
        n_workers=N_WORKERS,
    )


with timed_section('Cell 03'):
    # 1. Generate synthetic data with a chosen generator
    df_baseline = generate_syndata(
        num=1000,
        gen_algorithm="BASELINE",
        orig_df=orig_df,
        n_workers=N_WORKERS,
    )


with timed_section('Cell 04'):
    # 1. Generate synthetic data with a chosen generator
    df_ctgan = generate_syndata(
        num=1000,
        gen_algorithm="CTGAN",
        orig_df=orig_df.sample(50),
        n_workers=N_WORKERS,
    )


# # Compute Upsilon on synthetic dataframes

with timed_section('Cell 05: Compute Upsilon on synthetic dataframes'):
    ups_lsm, _ = compute_upsilon(
        df_lsm,
        model_path=model_path,
        n_workers=N_WORKERS,
    )
    print(np.nanmean(ups_lsm))
    save_seaborn_grid(sns.displot(ups_lsm), 'cell05_upsilon_lsm.png')


with timed_section('Cell 06'):
    df_baseline.columns=orig_df.columns


with timed_section('Cell 07'):
    ups_baseline, _ = compute_upsilon(
        df_baseline,#.round(0).astype(int),
        model_path=model_path,
        n_workers=N_WORKERS,
    )
    print(np.nanmean(ups_baseline))
    save_seaborn_grid(sns.displot(ups_baseline), 'cell07_upsilon_baseline.png')


with timed_section('Cell 08'):
    ups_ctgan, _ = compute_upsilon(
        df_ctgan,
        model_path=model_path,
        n_workers=N_WORKERS,
    )
    print(np.nanmean(ups_ctgan))
    save_seaborn_grid(sns.displot(ups_ctgan), 'cell08_upsilon_ctgan.png')


with timed_section('Cell 09'):
    ups_perm, _ = compute_upsilon(
        orig_df.sample(1000),
        model_path=model_path,
        n_workers=N_WORKERS,
    )
    print(np.nanmean(ups_perm))
    save_seaborn_grid(sns.displot(ups_perm), 'cell09_upsilon_real_control.png')


with timed_section('Cell 10'):
    import pylab as plt
    sns.distplot(ups_baseline,label='baseline')
    sns.distplot(ups_ctgan,label='ctgan')
    sns.distplot(ups_lsm,label='lsm')
    plt.legend()
    save_current_figure('cell10_upsilon_overlay.png')


with timed_section('Cell 11'):
    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score


    def real_vs_synth_auc(
        orig_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        *,
        test_size: float = 0.25,
        random_state: int = 123,
        cv_folds: int | None = None,
        max_rows: int | None = None,
    ) -> dict:
        """
        Train a classifier to separate real vs synthetic (categorical tabular data),
        and report AUC. Uses one-hot + logistic regression.

        If cv_folds is provided, returns cross-validated AUC on the full dataset.
        Otherwise returns a single train/test split AUC.

        max_rows optionally subsamples each dataset to speed things up.
        """
        # Basic checks
        if list(orig_df.columns) != list(syn_df.columns):
            raise ValueError("orig_df and syn_df must have identical columns in the same order.")

        # Optional subsampling (balanced)
        if max_rows is not None:
            n0 = min(len(orig_df), max_rows)
            n1 = min(len(syn_df), max_rows)
            orig_df = orig_df.sample(n=n0, random_state=random_state)
            syn_df = syn_df.sample(n=n1, random_state=random_state)

        # Ensure everything is treated as categorical tokens
        X_real = orig_df.copy()
        X_synth = syn_df.copy()
        for c in X_real.columns:
            X_real[c] = X_real[c].astype("string")
            X_synth[c] = X_synth[c].astype("string")

        X = pd.concat([X_real, X_synth], axis=0, ignore_index=True)
        y = np.concatenate([np.zeros(len(X_real), dtype=int), np.ones(len(X_synth), dtype=int)])

        # Model: one-hot categorical -> logistic regression
        pipe = Pipeline(
            steps=[
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ("clf", LogisticRegression(max_iter=2000, n_jobs=N_WORKERS)),
            ]
        )

        if cv_folds is not None and cv_folds >= 2:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=N_WORKERS)
            return {
                "mode": "cv",
                "cv_folds": cv_folds,
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


    # Example:
    # res = real_vs_synth_auc(orig_df, syn_df, cv_folds=5, max_rows=20000)
    # print(res)


with timed_section('Cell 12'):
    print(real_vs_synth_auc(orig_df.sample(10), df_lsm.sample(20), cv_folds=5, max_rows=20000))


with timed_section('Cell 13'):
    sns.distplot(orig_df.iloc[:,30])
    sns.distplot(df_ctgan.iloc[:,30])
    sns.distplot(df_lsm.iloc[:,30].replace('',np.nan))
    save_current_figure('cell13_column30_marginals.png')


# # How Novel Is The Generated Data
#
# One-hot cosine similarity between categorical rows; off-diagonal block quantifies synthetic novelty versus near-copying.

with timed_section('Cell 14: How Novel Is The Generated Data'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OneHotEncoder

    def row_similarity_blockplot(orig_df: pd.DataFrame,
                                 syn_df: pd.DataFrame,
                                 *,
                                 metric: str = "cosine",
                                 max_rows: int | None = 2000,
                                 random_state: int = 123,
                                 missing_token: str = "__MISSING__",
                                 show_block_means: bool = True,
                                 plot_filename: str | None = None):
        if list(orig_df.columns) != list(syn_df.columns):
            raise ValueError("orig_df and syn_df must have identical columns in the same order.")

        o = orig_df.copy()
        s = syn_df.copy()
        if max_rows is not None:
            n = min(len(o), len(s), max_rows)
            o = o.sample(n=n, random_state=random_state).reset_index(drop=True)
            s = s.sample(n=n, random_state=random_state).reset_index(drop=True)

        def normalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.replace("", pd.NA)
            df = df.astype("string").fillna(missing_token)
            return df.astype(object)

        o = normalize(o)
        s = normalize(s)

        if metric == "match":
            X = np.vstack([o.to_numpy(), s.to_numpy()])
            G = (X[:, None, :] == X[None, :, :]).mean(axis=2).astype(np.float32)

        elif metric == "cosine":
            comb = pd.concat([o, s], axis=0, ignore_index=True)
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            X = enc.fit_transform(comb)

            norms = np.linalg.norm(X, axis=1)
            norms[norms == 0] = 1.0
            G = (X @ X.T) / (norms[:, None] * norms[None, :])

        else:
            raise ValueError("metric must be 'cosine' or 'match'.")

        n = len(o)

        # Block means (RR, RS, SR, SS)
        rr = float(G[:n, :n].mean())
        rs = float(G[:n, n:].mean())
        sr = float(G[n:, :n].mean())
        ss = float(G[n:, n:].mean())
        block_means = np.array([[rr, rs],
                                [sr, ss]], dtype=float)

        # Plot full similarity + (optional) 2x2 block mean heatmap
        if show_block_means:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            im0 = axes[0].imshow(G, aspect="auto")
            axes[0].axhline(n - 0.5)
            axes[0].axvline(n - 0.5)
            axes[0].set_title(f"Row–row similarity ({metric})")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            BM = block_means#-block_means[0,1]
            print(BM)
            im1 = axes[1].imshow(BM, aspect="equal",vmin=.35,vmax=BM[0][0],cmap='Spectral_r')
            axes[1].set_xticks([0, 1], labels=["Real", "Synth"])
            axes[1].set_yticks([0, 1], labels=["Real", "Synth"])
            axes[1].set_title("2×2 block mean similarity")
            for (i, j), v in np.ndenumerate(BM):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_figure(fig, plot_filename or f'row_similarity_blockplot_{metric}.png')
        else:
            plt.figure(figsize=(7, 6))
            im = plt.imshow(G, aspect="auto")
            plt.axhline(n - 0.5)
            plt.axvline(n - 0.5)
            plt.title(f"Row–row similarity ({metric})")
            plt.colorbar(im, label="similarity")
            plt.tight_layout()
            save_current_figure(plot_filename or f'row_similarity_blockplot_{metric}.png')

        off_block = G[:n, n:]  # real x synth
        nn = off_block.max(axis=0)

        return {
            "n_rows_each": n,
            "nearest_real_similarity_mean": float(nn.mean()),
            "nearest_real_similarity_median": float(np.median(nn)),
            "nearest_real_similarity_p95": float(np.quantile(nn, 0.95)),
            "missing_token_used": missing_token,
            "block_means": {
                "real_real": rr,
                "real_synth": rs,
                "synth_real": sr,
                "synth_synth": ss,
            },
        }


# ## LSM synthetic data

with timed_section('Cell 15: LSM synthetic data'):
    statslsm = row_similarity_blockplot(orig_df, df_lsm, metric="cosine", max_rows=2000, plot_filename='cell15_row_similarity_lsm.png')
    print(statslsm)


# # CTGAN synthetic data

with timed_section('Cell 16: CTGAN synthetic data'):
    statsctgan = row_similarity_blockplot(orig_df, df_ctgan, metric="cosine", max_rows=2000, plot_filename='cell16_row_similarity_ctgan.png')
    print(statsctgan)


# # Baseline synthetic data

with timed_section('Cell 17: Baseline synthetic data'):
    statsbaseline = row_similarity_blockplot(orig_df.head(100), df_baseline, metric="cosine", max_rows=2000, plot_filename='cell17_row_similarity_baseline.png')
    print(statsbaseline)


# # Permutated Original Data as "synthetic"

with timed_section('Cell 18: Permutated Original Data as "synthetic"'):
    statsperm = row_similarity_blockplot(orig_df.head(100), orig_df.sample(100), metric="cosine", max_rows=2000, plot_filename='cell18_row_similarity_real_control.png')
    print(statsperm)


with timed_section('Cell 19'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def plot_generator_map(results, *, ups_key="upsilon_median", sim_key="nearest_real_similarity_median",
                           annotate=True, title="Generator map: Upsilon vs nearest-real similarity",
                           plot_filename: str = 'cell19_generator_map.png'):
        """
        results: dict like
          {
            "LSM": {"upsilon_median": 0.78, "nearest_real_similarity_median": 0.44},
            ...
          }

        x = nearest-real similarity (higher => closer to real rows; less movement)
        y = Upsilon (higher => better structural fidelity)
        """
        names = list(results.keys())
        x = np.array([results[n][sim_key] for n in names], dtype=float)
        y = np.array([results[n][ups_key] for n in names], dtype=float)

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y)
        plt.xlabel(sim_key.replace("_", " "))
        plt.ylabel(ups_key.replace("_", " "))
        plt.title(title)

        if annotate:
            for n, xi, yi in zip(names, x, y):
                plt.annotate(n, (xi, yi), textcoords="offset points", xytext=(6, 4), ha="left")

        plt.tight_layout()
        save_current_figure(plot_filename)


    # Example usage (fill in your Upsilon medians/means)
    results = {
        "LSM":     {"upsilon_median": np.median(ups_lsm), "nearest_real_similarity_median": 1/statslsm['nearest_real_similarity_median']},
        "baseline":     {"upsilon_median": np.median(ups_baseline), "nearest_real_similarity_median": 1/statsbaseline['nearest_real_similarity_median']},
        "CTGAN":{"upsilon_median": np.median(ups_ctgan), "nearest_real_similarity_median": 1/statsctgan['nearest_real_similarity_median']},
        "permutation":{"upsilon_median": np.median(ups_perm), "nearest_real_similarity_median": 1/statsperm['nearest_real_similarity_median']},
    }

    plot_generator_map(results, ups_key="upsilon_median", sim_key="nearest_real_similarity_median", plot_filename='cell19_generator_map.png')

