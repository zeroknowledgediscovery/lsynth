"""
Core utilities for computing Upsilon (average fidelity) for QuasiNet LSM models.

This module provides a high–level function, :func:`compute_upsilon`, for evaluating
a QuasiNet model's ``average_fidelity`` over either externally provided synthetic
data or data generated via several supported mechanisms:

    1. LSM-based generation using ``quasinet.qsampling.qsample``.
    2. A simple independent-column baseline generator.
    3. A CTGAN-based generator from the SDV library.
    4. A user-provided custom generator.

Typical use cases
-----------------
You have a trained QuasiNet model saved to disk (e.g. ``.joblib``) and either:

    • A real dataset you used for training, from which you want to generate
      synthetic samples (baseline or CTGAN), or
    • No external data, and you want to use the LSM itself as a generative model.

In all cases, this function computes

.. math::

    \\Upsilon_i = \\text{model.average_fidelity}(x_i)[0]

for each sample :math:`x_i` in a collection of samples, and returns the vector
of first components as a NumPy array.

Dependencies
------------
This module assumes:

    • ``quasinet`` is installed and provides:
        - :func:`quasinet.qnet.load_qnet`
        - :func:`quasinet.qsampling.qsample`
    • ``pandas`` and ``numpy`` are installed.
    • ``tqdm`` is installed for progress bars.
    • ``sdv`` is required only if you use the CTGAN generator
      (``gen_algorithm="CTGAN"``). It is imported lazily.

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from quasinet.qnet import load_qnet
from quasinet.qsampling import qsample


# ---------------------------------------------------------------------------
# Internal baseline generator (independent columns)
# ---------------------------------------------------------------------------

def _fit_independent_models(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Fit independent per-column models to a tabular dataset.

    This is a very simple baseline generator that ignores all dependencies
    between columns. Each column is modeled independently as:

        • Numeric columns: Gaussian with mean and population standard deviation.
        • Non-numeric columns: Categorical with empirical frequencies.

    Parameters
    ----------
    df :
        Input pandas DataFrame. Each column is treated independently.

    Returns
    -------
    models :
        A dictionary keyed by column name. For each column, the value is a
        dictionary with keys:

            • ``"type"``: either ``"numeric"`` or ``"categorical"``.
            • For numeric columns:
                - ``"mean"``: column mean.
                - ``"std"``: population standard deviation (ddof=0).
            • For categorical columns:
                - ``"values"``: NumPy array of unique values.
                - ``"probs"``: NumPy array of corresponding probabilities.
    """
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
    """
    Sample from independent per-column models.

    This is the sampling counterpart of :func:`_fit_independent_models`. Each
    column is sampled independently:

        • If ``type == "numeric"``: draw from a Gaussian with the stored mean
          and standard deviation.
        • If ``type == "categorical"``: draw from the stored categorical
          distribution.

    Parameters
    ----------
    models :
        Dictionary produced by :func:`_fit_independent_models`.
    n_rows :
        Number of rows (samples) to generate.

    Returns
    -------
    df :
        A pandas DataFrame of shape ``(n_rows, n_columns)`` with the same
        column names as ``models.keys()``.
    """
    data: Dict[str, Any] = {}
    rng = np.random.default_rng()
    for col, m in models.items():
        if m["type"] == "numeric":
            data[col] = rng.normal(loc=m["mean"], scale=m["std"], size=n_rows)
        else:
            data[col] = rng.choice(m["values"], size=n_rows, p=m["probs"])
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_upsilon(
    num: int = 100,
    model: Any = None,
    model_path: Optional[str] = None,
    syndata: Optional[Union[pd.DataFrame, Iterable[Any]]] = None,
    generate: bool = False,
    gen_algorithm: str = "LSM",
    data_generator: Optional[Callable[..., Iterable[Any]]] = None,
    orig_df: Optional[pd.DataFrame] = None,
    n_workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, Union[np.ndarray, List[Any]]]:
    """
    Compute Upsilon = model.average_fidelity(·) for a collection of samples.

    This is the main entry point for evaluating a QuasiNet model's
    ``average_fidelity`` on synthetic or external samples. You may either
    provide an already constructed model, or a path from which it can be
    loaded. You must also provide either:

        • ``syndata`` (external samples), or
        • ``generate=True`` with a specified generation strategy.

    The function centralizes all calls to ``model.average_fidelity`` in one
    place and optionally uses multi-threading for speed.

    Parameters
    ----------
    num :
        Number of samples to generate when ``generate=True`` and no external
        ``syndata`` is provided. Ignored if ``syndata`` is not ``None``.
    model :
        A loaded QuasiNet model instance. It must expose:

            • ``feature_names`` attribute (iterable of column names).
            • ``average_fidelity(sample)`` method.

        If ``model`` is ``None`` and ``model_path`` is provided, the model
        is loaded via :func:`quasinet.qnet.load_qnet`.
    model_path :
        Path to a saved QuasiNet model (e.g., ``.joblib``). Used only when
        ``model`` is ``None``.
    syndata :
        Optional external synthetic data. Two cases are supported:

            1. ``pandas.DataFrame``: columns must match
               ``model.feature_names`` exactly and in order. Each row is
               treated as one sample.

            2. Any iterable of samples directly consumable by
               ``model.average_fidelity``. For example, a list of NumPy
               arrays or lists.

        If ``syndata`` is provided, generation options are ignored.
    generate :
        Whether to generate data when ``syndata`` is ``None``. If ``True``,
        data are generated according to ``gen_algorithm`` and the relevant
        arguments.
    gen_algorithm :
        Name of the generation algorithm to use when ``generate=True`` and
        ``syndata`` is ``None``. Supported values are:

            • ``"LSM"``:
                Use :func:`quasinet.qsampling.qsample` with the provided
                model. This treats the model as a generative LSM and draws
                ``num`` samples of length ``M = len(model.feature_names)``.

            • ``"BASELINE"``:
                Use a simple independent-column baseline. Requires
                ``orig_df`` with columns matching ``model.feature_names``.
                Columns are modeled independently using
                :func:`_fit_independent_models` and sampled via
                :func:`_sample_independent`.

            • ``"CTGAN"``:
                Use an SDV ``CTGANSynthesizer``. Requires ``orig_df`` with
                columns matching ``model.feature_names``. Imports SDV
                lazily. You must have ``sdv`` installed to use this option.

            • Any other string:
                Requires a user-provided ``data_generator`` callable with
                signature ``data_generator(model=model, num=num)`` that
                returns an iterable of samples.

    data_generator :
        Custom generator callable used when ``gen_algorithm`` is not one of
        ``{"LSM", "BASELINE", "CTGAN"}``. It must accept ``model`` and
        ``num`` as keyword arguments and return an iterable of samples.
    orig_df :
        Original real dataset used to fit the model, required for
        ``gen_algorithm="BASELINE"`` and ``gen_algorithm="CTGAN"``. Must be
        a pandas DataFrame whose columns match ``model.feature_names`` in
        both name and order.
    n_workers :
        Number of worker threads to use for parallel evaluation of
        ``model.average_fidelity`` and, in the LSM case, for parallel
        ``qsample`` calls. If ``n_workers == 1``, everything is computed
        serially.
    verbose :
        If ``True``, print high-level progress messages and show tqdm
        progress bars. If ``False``, execution is silent.

    Returns
    -------
    upsilon :
        NumPy array of shape ``(n_samples,)`` where each entry is the first
        component of the output of ``model.average_fidelity(sample)`` for
        the corresponding sample.
    samples_out :
        The collection of samples actually used for evaluation. This is:

            • A NumPy array if the data originated from a DataFrame
              (external or generated), with shape
              ``(n_samples, n_features)``; or

            • A list of samples if an arbitrary iterable was used.

    Raises
    ------
    NotImplementedError
        If neither ``model`` nor ``model_path`` is provided, or if an
        unsupported ``gen_algorithm`` is used without a custom
        ``data_generator``.
    ValueError
        If ``syndata`` is ``None`` and ``generate`` is ``False``, or if a
        generator that requires ``orig_df`` is called without it, or if
        column alignment checks fail.

    Examples
    --------
    Basic usage with LSM generation:

    >>> import pandas as pd
    >>> from upsilon_fidelity.core import compute_upsilon
    >>> df_real = pd.read_csv("../datasets/gss_2018.csv").sample(100)
    >>> ups_lsm, syn_lsm = compute_upsilon(
    ...     num=100,
    ...     model_path="../datasets/gss_2018.joblib",
    ...     generate=True,
    ...     gen_algorithm="LSM",
    ...     orig_df=df_real,
    ...     n_workers=11,
    ... )
    >>> float(ups_lsm.mean())  # doctest: +SKIP

    Baseline independent-column generator:

    >>> ups_baseline, syn_baseline = compute_upsilon(
    ...     num=100,
    ...     model_path="../datasets/gss_2018.joblib",
    ...     generate=True,
    ...     gen_algorithm="BASELINE",
    ...     orig_df=df_real,
    ...     n_workers=11,
    ... )
    >>> float(ups_baseline.mean())  # doctest: +SKIP

    CTGAN-based generator (requires ``sdv``):

    >>> ups_ctgan, syn_ctgan = compute_upsilon(
    ...     num=100,
    ...     model_path="../datasets/gss_2018.joblib",
    ...     generate=True,
    ...     gen_algorithm="CTGAN",
    ...     orig_df=df_real,
    ...     n_workers=11,
    ... )
    >>> float(ups_ctgan.mean())  # doctest: +SKIP
    """
    # ------------------------------------------------------------------
    # 1. Ensure we have a model
    # ------------------------------------------------------------------
    if model is None:
        if model_path is None:
            raise NotImplementedError(
                "Model-free mode is not implemented; provide `model` or `model_path`."
            )
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = load_qnet(model_path)

    feature_names = list(model.feature_names)

    # ------------------------------------------------------------------
    # 2. Decide how to get samples (syndata or generated)
    # ------------------------------------------------------------------
    samples_out: Union[np.ndarray, List[Any], None] = None

    # 2a. External syndata provided
    if syndata is not None:
        if isinstance(syndata, pd.DataFrame):
            cols = list(syndata.columns)
            if cols != feature_names:
                raise ValueError(
                    "syndata DataFrame columns must exactly match model.feature_names "
                    "in the same order.\n"
                    f"Got:      {cols}\n"
                    f"Expected: {feature_names}"
                )
            samples_out = syndata.to_numpy().astype(str)
            if verbose:
                print(f"Using external syndata DataFrame with {samples_out.shape[0]} rows.")
        else:
            # Assume iterable of samples acceptable to average_fidelity
            samples_list = list(syndata)
            samples_out = samples_list
            if verbose:
                print(f"Using external syndata iterable with {len(samples_list)} samples.")

    # 2b. No syndata: generate if requested
    elif generate:
        M = len(feature_names)

        if gen_algorithm == "LSM":
            if verbose:
                print(f"Generating {num} samples via LSM (qsample).")

            # Null sample vector
            N = np.array([""] * M).astype("U50")

            if n_workers == 1:
                samples_out = [
                    qsample(N, model, M)
                    for _ in tqdm(range(num), desc="qsample(LSM)")
                ]
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    samples_out = list(
                        tqdm(
                            ex.map(lambda _: qsample(N, model, M), range(num)),
                            total=num,
                            desc=f"qsample(LSM, threads={n_workers})",
                        )
                    )

        elif gen_algorithm == "BASELINE":
            if orig_df is None:
                raise ValueError(
                    "orig_df must be provided for gen_algorithm='BASELINE'."
                )
            cols = list(orig_df.columns)
            if cols != feature_names:
                raise ValueError(
                    "orig_df columns must exactly match model.feature_names "
                    "in the same order for BASELINE generator.\n"
                    f"Got:      {cols}\n"
                    f"Expected: {feature_names}"
                )
            if verbose:
                print(f"Generating {num} samples via BASELINE (independent columns).")
            models = _fit_independent_models(orig_df)
            synthetic_df = _sample_independent(models, num)
            synthetic_df = synthetic_df[feature_names]
            samples_out = synthetic_df.to_numpy().astype(str)

        elif gen_algorithm == "CTGAN":
            if orig_df is None:
                raise ValueError(
                    "orig_df must be provided for gen_algorithm='CTGAN'."
                )
            cols = list(orig_df.columns)
            if cols != feature_names:
                raise ValueError(
                    "orig_df columns must exactly match model.feature_names "
                    "in the same order for CTGAN generator.\n"
                    f"Got:      {cols}\n"
                    f"Expected: {feature_names}"
                )
            if verbose:
                print(f"Generating {num} samples via CTGAN.")

            # Local import so BASELINE/LSM don't require sdv installed
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=orig_df)

            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(orig_df)

            synthetic_df = synthesizer.sample(num_rows=num)
            synthetic_df = synthetic_df[feature_names]
            samples_out = synthetic_df.to_numpy().astype(str)

        else:
            if data_generator is None:
                raise NotImplementedError(
                    f"Generation algorithm '{gen_algorithm}' is not implemented and "
                    f"no `data_generator` was provided."
                )
            if verbose:
                print(f"Generating {num} samples via custom generator '{gen_algorithm}'.")
            samples_generated = data_generator(model=model, num=num)
            samples_out = list(samples_generated)

    else:
        # 2c. No data and no generation
        raise ValueError("No `syndata` provided and `generate` is False.")

    # ------------------------------------------------------------------
    # 3. Compute Upsilon: single place where average_fidelity is called
    # ------------------------------------------------------------------
    if verbose:
        print("Computing Upsilon via model.average_fidelity ...")

    if isinstance(samples_out, np.ndarray):
        iterator = samples_out
        total = samples_out.shape[0]
    else:
        samples_list = list(samples_out)  # type: ignore[arg-type]
        samples_out = samples_list        # normalize representation
        iterator = samples_list
        total = len(samples_list)

    if n_workers == 1:
        upsilon_list = [
            model.average_fidelity(sample)
            for sample in tqdm(iterator, total=total, desc="average_fidelity")
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            upsilon_list = list(
                tqdm(
                    ex.map(model.average_fidelity, iterator),
                    total=total,
                    desc=f"average_fidelity(threads={n_workers})",
                )
            )

    # We assume average_fidelity(sample) returns an indexable object and
    # the first component is the scalar Upsilon of interest.
    upsilon = np.array([upsilon_list[i][0] for i in range(len(upsilon_list))])

    return upsilon, samples_out
