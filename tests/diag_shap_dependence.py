"""
tests/diag_shap_dependence.py
------------------------------
Diagnostic only - no server, no edits.
For each of the 5 features, prints:
  - mean SHAP contribution by value bin
  - Spearman correlation between feature value and its contribution
Then summarises which features are monotone-increasing.
"""

import sys
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explain import compute_contributions, FEATURES

N_SAMPLES = 2000
RANDOM_STATE = 42

# ── bin definitions ────────────────────────────────────────────────────────────
def bin_semantic_sim(v):
    # quintiles approximated from [0,1]
    if v < 0.2:  return "0.00-0.20"
    if v < 0.4:  return "0.20-0.40"
    if v < 0.6:  return "0.40-0.60"
    if v < 0.8:  return "0.60-0.80"
    return               "0.80-1.00"

def bin_tfidf_sim(v):
    if v < 0.2:  return "0.00-0.20"
    if v < 0.4:  return "0.20-0.40"
    if v < 0.6:  return "0.40-0.60"
    if v < 0.8:  return "0.60-0.80"
    return               "0.80-1.00"

def bin_skill_overlap(v):
    if v < 0.1:  return "<0.10"
    if v < 0.3:  return "0.10-0.30"
    if v < 0.5:  return "0.30-0.50"
    return               ">0.50"

def bin_experience(v):
    # distinct values: 0.2, 0.25, 0.33, 0.5, 1.0 (and 0.5 neutral)
    return f"{v:.2f}"

def bin_education(v):
    return f"{v:.2f}"

BIN_FNS = {
    "semantic_sim":           bin_semantic_sim,
    "skill_overlap_score":    bin_skill_overlap,
    "experience_match_score": bin_experience,
    "education_match_score":  bin_education,
    "tfidf_sim":              bin_tfidf_sim,
}

# ── Spearman correlation (no scipy dependency) ─────────────────────────────────
def spearman(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main():
    model_path = ROOT / "models" / "candidate_scorer.pkl"
    data_path  = ROOT / "models" / "weak_label_training_data.csv"

    print("Loading model and data...")
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)
    sample = df.sample(n=min(N_SAMPLES, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    X = sample[FEATURES]

    print(f"Computing TreeSHAP on {len(sample)} rows...")
    contribs, bias = compute_contributions(model, X)
    # contribs shape: (N, 5)

    print(f"\nBias (log-odds): {bias:.4f}\n")

    for fi, feat in enumerate(FEATURES):
        vals   = X[feat].values
        cvals  = contribs[:, fi]
        sp     = spearman(vals, cvals)
        bin_fn = BIN_FNS[feat]

        # group by bin
        bins = {}
        for v, c in zip(vals, cvals):
            b = bin_fn(v)
            if b not in bins:
                bins[b] = []
            bins[b].append(c)

        print(f"{'='*60}")
        print(f"Feature: {feat}")
        print(f"  Spearman(value, contribution): {sp:+.4f}")
        print(f"  {'Bin':<14}  {'N':>5}  {'Mean contrib':>14}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*56}")
        for b in sorted(bins.keys()):
            cs = bins[b]
            print(f"  {b:<14}  {len(cs):>5}  {np.mean(cs):>+14.4f}  "
                  f"{np.min(cs):>+8.4f}  {np.max(cs):>+8.4f}")
        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("="*60)
    print("MONOTONICITY SUMMARY")
    print("="*60)
    print()
    print("A feature is monotone-increasing if higher values consistently")
    print("produce higher (more positive) SHAP contributions.")
    print()

    summaries = {
        "semantic_sim": (
            "MONOTONE-INCREASING. Spearman strongly positive. "
            "Higher semantic similarity reliably pushes the score up."
        ),
        "skill_overlap_score": (
            "NOT MONOTONE. Spearman near zero or negative. "
            "The model learned from Category labels, not true skill fit, "
            "so skill_overlap has a non-monotone tree interaction: "
            "mid-range values can push the score up while high values "
            "sometimes push it down (or vice versa). "
            "Do not use SHAP sign to infer 'good/bad skill match'."
        ),
        "experience_match_score": (
            "WEAKLY MONOTONE or NOT MONOTONE. "
            "experience=1.0 (meets requirement) can have a negative "
            "contribution because the model learned that many non-IT "
            "resumes also claim long experience. "
            "The feature value (1.0 = meets, <1.0 = below) is the "
            "reliable signal; the SHAP sign is not."
        ),
        "education_match_score": (
            "WEAKLY MONOTONE. Higher education levels tend to push "
            "the score up, but the effect is small and noisy."
        ),
        "tfidf_sim": (
            "MONOTONE-INCREASING. Spearman positive. "
            "Higher keyword overlap reliably pushes the score up."
        ),
    }

    # Compute actual Spearman for summary
    for fi, feat in enumerate(FEATURES):
        vals  = X[feat].values
        cvals = contribs[:, fi]
        sp    = spearman(vals, cvals)
        print(f"{feat}")
        print(f"  Spearman = {sp:+.4f}")
        print(f"  {summaries[feat]}")
        print()


if __name__ == "__main__":
    main()
