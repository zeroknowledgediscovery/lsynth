# lsynth

**MAP-alignment fidelity and dataset distance for synthetic tabular data**

This package implements the one-sided MAP-alignment fidelity statistic
introduced by Chattopadhyay *et al.* and described in the manuscript
“How Good Is Your Synthetic Data?”.

The package cleanly separates **synthetic data generation** from **evaluation**:

* `generate_syndata(...)` produces synthetic tabular data using multiple generators.
* `compute_upsilon(df, ...)` evaluates MAP-alignment fidelity on *any* external DataFrame.

The library is fully compatible with externally generated synthetic data and is primarily aimed at **evaluation**, although several reference generators are included.

---

## Core Idea

For a synthetic record to be realistic, each coordinate should agree
with the conditional MAP prediction inferred from real data.

For a data record `x` and coordinate `i`:

```
υ(x, i) = φ_i(x_i | x_{-i}) / max_y φ_i(y | x_{-i})
```

Averaged over samples and coordinates:

```
Υ(D) ∈ [0, 1]
```

* **High Υ** ⇒ synthetic preserves *real conditional structure*
* **Low Υ** ⇒ structural distortion (even if marginals or covariance match)

---

## Installation

```bash
pip install lsynth
```

---

## Synthetic Data Generation

Synthetic data generation is handled by `generate_syndata`, which always returns a **pandas DataFrame** whose columns exactly match the target feature space.

```python
from lsynth import generate_syndata
```

### Supported Generators

* **`"LSM"`**
  Uses a trained QuasiNet model as a generative model via `qsample`.

* **`"BASELINE"`**
  Independent-column null model (Gaussian for numeric, categorical sampling for discrete).

* **`"CTGAN"`**
  Uses SDV’s `CTGANSynthesizer`.

* **Custom generators**
  Any user-defined function returning a DataFrame with the correct columns.

### Example: LSM Generation

```python
df_lsm = generate_syndata(
    num=1000,
    model_path="gss_2018.joblib",
    gen_algorithm="LSM",
    n_workers=8,
)
```

### Example: Baseline Generator

```python
df_baseline = generate_syndata(
    num=1000,
    gen_algorithm="BASELINE",
    orig_df=df_real,
    feature_names=df_real.columns.tolist(),
)
```

### Example: CTGAN Generator

```python
df_ctgan = generate_syndata(
    num=1000,
    gen_algorithm="CTGAN",
    orig_df=df_real,
    feature_names=df_real.columns.tolist(),
)
```

---

## MAP-Alignment Fidelity Evaluation

Evaluation is handled **only** by `compute_upsilon`, which operates on an external DataFrame.

```python
from lsynth import compute_upsilon
```

```python
ups, _ = compute_upsilon(
    df=df_lsm,
    model_path="gss_2018.joblib",
    n_workers=8,
)

print("Mean Υ:", ups.mean())
```

Key properties:

* No data generation occurs inside `compute_upsilon`
* Columns must match `model.feature_names` exactly and in order
* Works identically for real or synthetic data

---

## Example Notebook

See **`example2.ipynb`** for a complete, reproducible workflow:

1. Load real data
2. Generate synthetic datasets using multiple generators
3. Compute and compare MAP-alignment fidelity
4. Interpret structural degradation across generators

---

## Why MAP-Alignment?

Because **covariance matching is insufficient**.

The manuscript shows explicit counterexamples where:

* Real and synthetic data share identical means, variances, and covariance matrices
* Yet differ strongly in higher-order and conditional structure
* MAP-alignment detects the discrepancy immediately

This method:

* Detects nonlinear and higher-order dependencies
* Avoids embedding or representation artifacts
* Comes with finite-sample uncertainty control

---

## Citation

```
Chattopadhyay I, et al.
“How Good Is Your Synthetic Data?”
```
