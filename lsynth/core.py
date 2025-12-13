from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from quasinet.qnet import load_qnet
from quasinet.qsampling import qsample

import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def silence_output():
    """Brute-force: silence *all* prints/warnings/logging to stdout+stderr inside the block."""
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

# ---------------------------------------------------------------------------
# Internal baseline generator (independent columns)
# ---------------------------------------------------------------------------

def _fit_independent_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            models[col] = {
                "type": "numeric",
                "mean": df[col].mean(),
                "std": df[col].std(ddof=0),
            }
        else:
            probs = df[col].value_counts(normalize=True)
            models[col] = {
                "type": "categorical",
                "values": probs.index.to_numpy(),
                "probs": probs.to_numpy(),
            }
    return models


def _sample_independent(models: Mapping[str, Mapping[str, Any]], n_rows: int) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    rng = np.random.default_rng()
    for col, m in models.items():
        if m["type"] == "numeric":
            data[col] = rng.normal(loc=m["mean"], scale=m["std"], size=n_rows)
        else:
            data[col] = rng.choice(m["values"], size=n_rows, p=m["probs"])
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Synthetic data generator → always returns a DataFrame
# ---------------------------------------------------------------------------

def generate_syndata(
    num: int,
    model: Any = None,
    model_path: Optional[str] = None,
    gen_algorithm: str = "LSM",
    orig_df: Optional[pd.DataFrame] = None,
    data_generator: Optional[Callable[..., pd.DataFrame]] = None,
    n_workers: int = 1,
    verbose: bool = True,
    feature_names: Optional[list[str]] = None,  # NEW (default None)        
) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame with `num` rows whose columns match
    `feature_names` (if provided) or else `model.feature_names`.

    Supported `gen_algorithm` values:

      - "LSM": uses `qsample` with a null symbol vector of length M; requires a QNet model
               (either `model` or `model_path`). If `feature_names` is not provided, it is
               taken from `model.feature_names`.
      - "BASELINE": independent-column Gaussian/categorical model fit to `orig_df`;
                   requires `orig_df` and `feature_names` (or a model to infer them).
      - "CTGAN": SDV `CTGANSynthesizer` fit to `orig_df`; requires `orig_df` and
                 `feature_names` (or a model to infer them).
      - otherwise: calls `data_generator(model=model, num=num, orig_df=orig_df, feature_names=feature_names)`
                   and expects a DataFrame with columns exactly equal to `feature_names`.

    For "BASELINE" and "CTGAN", `orig_df` must be provided and its columns must match
    `feature_names` exactly and in the same order.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of shape (num, M) with columns exactly equal to `feature_names`.
    """
    # Ensure we have a model ONLY if we need it (LSM or custom that wants it)
    needs_model = (gen_algorithm == "LSM") or (data_generator is not None)
    
    if needs_model and model is None:
        if model_path is None:
            raise NotImplementedError(
                "Model-free generation not implemented; provide `model` or `model_path`."
            )
        if verbose:
            print(f"Loading model from {model_path} ...")
            model = load_qnet(model_path)

            feature_names = list(model.feature_names)

    if feature_names is None:
        raise ValueError("feature_names is undefined (required for BASELINE/CTGAN column checks).")
       
    M = len(feature_names)

    # LSM: use qsample starting from null vector
    if gen_algorithm == "LSM":
        if verbose:
            print(f"Generating {num} rows via LSM (qsample).")
        N = np.array([""] * M).astype("U50")

        if n_workers == 1:
            rows = [
                qsample(N, model, M)
                for _ in tqdm(range(num), desc="qsample(LSM)")
            ]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                rows = list(
                    tqdm(
                        ex.map(lambda _: qsample(N, model, M), range(num)),
                        total=num,
                        desc=f"qsample(LSM, threads={n_workers})",
                    )
                )

        df_syn = pd.DataFrame(rows, columns=feature_names)
        return df_syn

    # BASELINE: independent columns from orig_df
    if gen_algorithm == "BASELINE":
        if orig_df is None:
            raise ValueError("orig_df must be provided for gen_algorithm='BASELINE'.")
        cols = list(orig_df.columns)
        if cols != feature_names:
            raise ValueError(
                "orig_df columns must exactly match model.feature_names "
                "in the same order for BASELINE generator.\n"
                f"Got:      {cols}\n"
                f"Expected: {feature_names}"
            )
        if verbose:
            print(f"Generating {num} rows via BASELINE (independent columns).")
        models = _fit_independent_models(orig_df)
        df_syn = _sample_independent(models, num)
        df_syn = df_syn[feature_names]
        return df_syn

    # CTGAN: SDV CTGANSynthesizer from orig_df
    if gen_algorithm == "CTGAN":
        if orig_df is None:
            raise ValueError("orig_df must be provided for gen_algorithm='CTGAN'.")
        cols = list(orig_df.columns)
        if cols != feature_names:
            raise ValueError(
                "orig_df columns must exactly match model.feature_names "
                "in the same order for CTGAN generator.\n"
                f"Got:      {cols}\n"
                f"Expected: {feature_names}"
            )
        if verbose:
            print(f"Generating {num} rows via CTGAN.")

        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer


        with silence_output():
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=orig_df)

            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(orig_df)

        df_syn = synthesizer.sample(num_rows=num)
        df_syn = df_syn[feature_names]
        return df_syn

    # Custom generator
    if data_generator is None:
        raise NotImplementedError(
            f"Generation algorithm '{gen_algorithm}' is not implemented and "
            f"no `data_generator` was provided."
        )
    if verbose:
        print(f"Generating {num} rows via custom generator '{gen_algorithm}'.")

    result = data_generator(model=model, num=num, orig_df=orig_df)

    if isinstance(result, pd.DataFrame):
        cols = list(result.columns)
        if cols != feature_names:
            raise ValueError(
                "Custom generator DataFrame columns must match model.feature_names "
                "in the same order.\n"
                f"Got:      {cols}\n"
                f"Expected: {feature_names}"
            )
        return result

    # Fallback: assume array-like
    arr = np.asarray(result)
    if arr.shape[1] != M:
        raise ValueError(
            f"Custom generator output has shape {arr.shape}, expected (*, {M})."
        )
    df_syn = pd.DataFrame(arr, columns=feature_names)
    return df_syn


# ---------------------------------------------------------------------------
# Upsilon computation from an external DataFrame only
# ---------------------------------------------------------------------------

def compute_upsilon(
    df: pd.DataFrame,
    model: Any = None,
    model_path: Optional[str] = None,
    n_workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute Upsilon = model.average_fidelity(row)[0] for each row of an
    external DataFrame.

    The DataFrame must have columns that match `model.feature_names` exactly
    and in the same order. No data generation happens here; this function
    only evaluates Upsilon on the provided `df`.

    Parameters
    ----------
    df :
        External tabular data. Each row is treated as one sample.
    model :
        Loaded QuasiNet model. If None, `model_path` must be provided.
    model_path :
        Path to a saved QuasiNet model (e.g., .joblib). Used only if `model`
        is None.
    n_workers :
        Number of threads for parallel evaluation of average_fidelity. If 1,
        computation is serial.
    verbose :
        If True, print progress messages and show tqdm bars.

    Returns
    -------
    upsilon :
        NumPy array of shape (n_rows,), containing the first component of
        model.average_fidelity(row) for each row.
    df_out :
        The same DataFrame, returned for convenience.
    """
    # Ensure we have a model
    if model is None:
        if model_path is None:
            raise NotImplementedError(
                "Model-free mode is not implemented; provide `model` or `model_path`."
            )
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = load_qnet(model_path)

    feature_names = list(model.feature_names)
    cols = list(df.columns)
    if cols != feature_names:
        raise ValueError(
            "Input DataFrame columns must exactly match model.feature_names "
            "in the same order.\n"
            f"Got:      {cols}\n"
            f"Expected: {feature_names}"
        )

    X = df.to_numpy().astype(str)
    n_rows = X.shape[0]

    if verbose:
        print(f"Computing Upsilon on DataFrame with {n_rows} rows ...")

    if n_workers == 1:
        upsilon_list = [
            model.average_fidelity(row)
            for row in tqdm(X, total=n_rows, desc="average_fidelity")
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            upsilon_list = list(
                tqdm(
                    ex.map(model.average_fidelity, X),
                    total=n_rows,
                    desc=f"average_fidelity(threads={n_workers})",
                )
            )

    upsilon = np.array([upsilon_list[i][0] for i in range(len(upsilon_list))])
    return upsilon, df
