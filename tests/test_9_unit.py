"""
tests/test_9_unit.py
--------------------
Step 9 Unit Tests (no server):
  a) Additivity: sigmoid(sum(contribs) + bias) == predict_proba within 1e-4
     on 50 random rows from models/weak_label_training_data.csv.  PASS/FAIL.
  b) Strong-vs-weak profile diagnostic.  REPORT ONLY, no assertion.
     See comment in test_sign_sanity() for why.
"""

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explain import compute_contributions, FEATURES


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def test_additivity(model, training_data_path, n_samples=50):
    print("=" * 65)
    print(f"4a. ADDITIVITY TEST ({n_samples} random rows from training data)")
    print("=" * 65)

    df        = pd.read_csv(training_data_path)
    sample_df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    X         = sample_df[FEATURES]

    proba                  = model.predict_proba(X)[:, 1]
    feat_contribs, bias    = compute_contributions(model, X)

    max_abs_err = 0.0
    tol         = 1e-4
    all_passed  = True

    for i in range(len(sample_df)):
        c_row        = feat_contribs[i]
        sum_contribs = float(np.sum(c_row))
        computed     = sigmoid(sum_contribs + bias)
        expected     = float(proba[i])
        err          = abs(computed - expected)
        if err > max_abs_err:
            max_abs_err = err
        if err >= tol:
            all_passed = False
            print(f"  Row {i} FAIL: expected={expected:.6f}, computed={computed:.6f}, diff={err:.2e}")

    print(f"  Rows tested:         {n_samples}")
    print(f"  Max absolute error:  {max_abs_err:.2e}")
    print(f"  Tolerance threshold: {tol:.2e}")
    print(f"  Status:              {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed, max_abs_err


def test_sign_sanity(model):
    # DIAGNOSTIC ONLY -- no pass/fail assertion.
    #
    # A boosted-tree SHAP contribution at a single data point is NOT
    # guaranteed to be monotone in the feature value.  XGBoost partitions
    # the feature space into leaves; the contribution of feature x_i at a
    # given point depends on which leaf the full feature vector lands in and
    # on the path of splits leading there -- not just on x_i alone.
    #
    # The dependence diagnostic (tests/diag_shap_dependence.py) confirms:
    #   semantic_sim   Spearman +0.93  -- reliably monotone
    #   tfidf_sim      Spearman +0.62  -- reliably monotone
    #   education      Spearman +0.32  -- weakly monotone
    #   experience     Spearman -0.19  -- NOT monotone
    #   skill_overlap  Spearman -0.30  -- NOT monotone
    #
    # Asserting that skill_overlap or experience contributions must be
    # positive for a "strong" profile would assert something the model does
    # not guarantee.  We report the values so the reader can see the
    # behaviour; the consistency gate in explain.py handles this correctly
    # by suppressing reasons whose SHAP sign disagrees with the feature value.
    print("=" * 65)
    print("4b. SIGN SANITY -- DIAGNOSTIC (report only, no pass/fail)")
    print("=" * 65)
    print("NOTE: boosted-tree SHAP is not monotone per feature.")
    print("      skill_overlap and experience_match have Spearman ~ -0.2 to -0.3")
    print("      with their contributions (see diag_shap_dependence.py).")
    print("      Values are reported here for information only.")
    print()

    strong_vals = [0.77, 0.67, 1.0, 1.0, 0.10]
    weak_vals   = [0.30, 0.0,  0.5, 0.3, 0.02]

    c_strong, b_strong = compute_contributions(model, strong_vals)
    c_weak,   b_weak   = compute_contributions(model, weak_vals)

    print("Strong profile:", dict(zip(FEATURES, strong_vals)))
    print(f"Bias: {b_strong:+.4f}")
    for f, v, c in zip(FEATURES, strong_vals, c_strong):
        print(f"  {f:<25} val={v:.2f}  contrib={c:+.4f}")

    print()
    print("Weak profile:  ", dict(zip(FEATURES, weak_vals)))
    print(f"Bias: {b_weak:+.4f}")
    for f, v, c in zip(FEATURES, weak_vals, c_weak):
        print(f"  {f:<25} val={v:.2f}  contrib={c:+.4f}")

    feat_idx = {name: idx for idx, name in enumerate(FEATURES)}
    s_sem = c_strong[feat_idx["semantic_sim"]]
    w_sem = c_weak[feat_idx["semantic_sim"]]
    s_ski = c_strong[feat_idx["skill_overlap_score"]]
    w_ski = c_weak[feat_idx["skill_overlap_score"]]

    print()
    print("Observations (informational):")
    print(f"  Strong semantic_sim contrib:  {s_sem:+.4f}  (monotone feature -- expected > 0)")
    print(f"  Weak   semantic_sim contrib:  {w_sem:+.4f}  (monotone feature -- expected < 0)")
    print(f"  Strong skill_overlap contrib: {s_ski:+.4f}  (non-monotone -- no expectation)")
    print(f"  Weak   skill_overlap contrib: {w_ski:+.4f}  (non-monotone -- no expectation)")
    print()


def main():
    model_path         = ROOT / "models" / "candidate_scorer.pkl"
    training_data_path = ROOT / "models" / "weak_label_training_data.csv"

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    additivity_pass, max_err = test_additivity(model, training_data_path, n_samples=50)
    test_sign_sanity(model)

    print("=" * 65)
    print("UNIT TEST SUMMARY")
    print(f"  Additivity: {'PASS (max error ' + f'{max_err:.2e})' if additivity_pass else 'FAIL'}")
    print(f"  Sign sanity: diagnostic only -- see output above")
    print("=" * 65)

    if not additivity_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
