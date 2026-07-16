import pandas as pd
from lsynth import compute_upsilon

df_real = pd.read_csv("../datasets/gss_2018.csv").sample(100)

# Baseline independent-column generator
ups_baseline, syn_baseline = compute_upsilon(
    num=100,
    model_path="../datasets/gss_2018.joblib",
    generate=True,
    gen_algorithm="BASELINE",
    orig_df=df_real,
    n_workers=11,
)
print("Baseline mean Upsilon:", ups_baseline.mean())

# LSM generator using qsample
ups_lsm, syn_lsm = compute_upsilon(
    num=100,
    model_path="../datasets/gss_2018.joblib",
    generate=True,
    gen_algorithm="LSM",
    orig_df=df_real,
    n_workers=11,
)
print("LSM mean Upsilon:", ups_lsm.mean())

# CTGAN generator (requires sdv installed)
ups_ctgan, syn_ctgan = compute_upsilon(
    num=100,
    model_path="../datasets/gss_2018.joblib",
    generate=True,
    gen_algorithm="CTGAN",
    orig_df=df_real,
    n_workers=11,
)
print("CTGAN mean Upsilon:", ups_ctgan.mean())
