"""
explain.py
----------
TalentRank AI - Step 9: TreeSHAP explainability for XGBoost candidate scorer.

Provides:
- FEATURES list matching models/model_metadata.json
- compute_contributions(model, X) -> (feat_contribs, bias)
- build_explanation(row_values, contribs, bias, matched_skills, missing_skills)
- build_facts(row_values, matched_skills, missing_skills)

Design notes on SHAP sign vs feature value
-------------------------------------------
TreeSHAP contributions at a single data point are NOT guaranteed to be
monotone in the feature value for a boosted-tree model.  The dependence
diagnostic (tests/diag_shap_dependence.py) shows:

  semantic_sim        Spearman +0.93  -> reliably monotone
  tfidf_sim           Spearman +0.62  -> reliably monotone
  education_match     Spearman +0.32  -> weakly monotone
  experience_match    Spearman -0.19  -> NOT monotone
  skill_overlap       Spearman -0.30  -> NOT monotone (model learned
                                         Category proxy, not true skill fit)

Consequence: we must NEVER generate a reason whose wording contradicts
the feature value.  The consistency gate in build_explanation enforces:
  - positive reason only if contribution > 0 AND value is on the
    favorable side of the threshold
  - negative reason only if contribution < 0 AND value is on the
    unfavorable side
  - if SHAP sign disagrees with value, no reason is generated for that
    feature (show fewer than 3 reasons rather than a misleading one)

The "facts" list is derived from feature values only (never from SHAP)
and is always truthful.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import xgboost

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r") as f:
            _meta = json.load(f)
        FEATURES = _meta.get("features", [
            "semantic_sim", "skill_overlap_score", "experience_match_score",
            "education_match_score", "tfidf_sim",
        ])
    except Exception:
        FEATURES = [
            "semantic_sim", "skill_overlap_score", "experience_match_score",
            "education_match_score", "tfidf_sim",
        ]
else:
    FEATURES = [
        "semantic_sim", "skill_overlap_score", "experience_match_score",
        "education_match_score", "tfidf_sim",
    ]

# Thresholds that define "favorable" vs "unfavorable" for each feature.
# Used by the consistency gate: a positive reason requires value >= threshold;
# a negative reason requires value < threshold.
FAVORABLE_THRESHOLD = {
    "semantic_sim":           0.5,
    "skill_overlap_score":    0.4,
    "experience_match_score": 1.0,
    "education_match_score":  1.0,
    "tfidf_sim":              0.1,
}


def compute_contributions(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray, list],
) -> Tuple[np.ndarray, float]:
    """
    Computes TreeSHAP feature contributions and bias using XGBoost pred_contribs.

    Returns:
        (feature_contributions, bias)
        feature_contributions: shape (N, 5) for N > 1, or (5,) for a single row
        bias: float (base margin in log-odds, same for all rows)
    """
    booster = model.get_booster() if hasattr(model, "get_booster") else model

    is_single_row = False
    if isinstance(X, pd.DataFrame):
        feat_df = X[FEATURES] if all(c in X.columns for c in FEATURES) else X
        dmatrix = xgboost.DMatrix(feat_df, feature_names=FEATURES)
        if len(feat_df) == 1:
            is_single_row = True
    elif isinstance(X, pd.Series):
        is_single_row = True
        df = pd.DataFrame([X])[FEATURES] if all(c in X.index for c in FEATURES) else pd.DataFrame([X])
        dmatrix = xgboost.DMatrix(df, feature_names=FEATURES)
    else:
        arr = np.array(X, dtype=float)
        if arr.ndim == 1:
            is_single_row = True
            arr = arr.reshape(1, -1)
        dmatrix = xgboost.DMatrix(arr, feature_names=FEATURES)

    raw_contribs  = booster.predict(dmatrix, pred_contribs=True)
    feat_contribs = raw_contribs[:, :-1]
    bias          = float(raw_contribs[0, -1])

    if is_single_row:
        return feat_contribs[0], bias
    return feat_contribs, bias


# ── Reason templates ───────────────────────────────────────────────────────────

def _reason_positive(feat_name: str, val: float) -> str:
    """Plain-English positive reason from feature value. Max 80 chars."""
    if feat_name == "semantic_sim":
        return f"High semantic similarity ({val:.2f})"
    if feat_name == "skill_overlap_score":
        return f"Strong skill overlap ({val:.2f})"
    if feat_name == "experience_match_score":
        return f"Meets experience requirement ({val:.2f})"
    if feat_name == "education_match_score":
        return f"Meets education requirement ({val:.2f})"
    if feat_name == "tfidf_sim":
        return f"Good keyword similarity ({val:.2f})"
    return f"Positive {feat_name} ({val:.2f})"


def _reason_negative(feat_name: str, val: float) -> str:
    """Plain-English negative reason from feature value. Max 80 chars."""
    if feat_name == "semantic_sim":
        return f"Low semantic similarity ({val:.2f})"
    if feat_name == "skill_overlap_score":
        return f"Limited skill overlap ({val:.2f})"
    if feat_name == "experience_match_score":
        return f"Experience below requirement ({val:.2f})"
    if feat_name == "education_match_score":
        return f"Education requirement not matched ({val:.2f})"
    if feat_name == "tfidf_sim":
        return f"Low keyword similarity ({val:.2f})"
    return f"Low {feat_name} ({val:.2f})"


def _is_favorable(feat_name: str, val: float) -> bool:
    return val >= FAVORABLE_THRESHOLD[feat_name]


def build_facts(
    row_values: Dict[str, float],
    matched_skills: List[str],
    missing_skills: List[str],
) -> List[str]:
    """
    Derives always-truthful plain-English facts from feature values only.
    Never uses SHAP contributions.  Max 80 chars each.
    """
    facts = []

    sem = row_values.get("semantic_sim", 0.0)
    facts.append(f"Semantic similarity: {sem:.2f}")

    n_matched = len(matched_skills)
    n_total   = n_matched + len(missing_skills)
    if n_total > 0:
        facts.append(f"Skill overlap: {row_values.get('skill_overlap_score', 0.0):.2f} "
                     f"({n_matched} of {n_total} skills matched)")
    else:
        facts.append(f"Skill overlap: {row_values.get('skill_overlap_score', 0.0):.2f}")

    exp = row_values.get("experience_match_score", 0.5)
    if exp >= 1.0:
        facts.append("Meets experience requirement (1.00)")
    elif exp == 0.5:
        facts.append("Experience requirement not parseable (neutral 0.50)")
    else:
        facts.append(f"Experience below requirement ({exp:.2f})")

    edu = row_values.get("education_match_score", 0.5)
    if edu >= 1.0:
        facts.append("Meets education requirement (1.00)")
    elif edu == 0.5:
        facts.append("Education requirement not specified (neutral 0.50)")
    elif edu >= 0.7:
        facts.append(f"Partial education match ({edu:.2f})")
    else:
        facts.append(f"Education requirement not matched ({edu:.2f})")

    tfidf = row_values.get("tfidf_sim", 0.0)
    facts.append(f"Keyword similarity: {tfidf:.2f}")

    return facts


def build_explanation(
    row_values: Dict[str, float],
    contribs: np.ndarray,
    bias: float,
    matched_skills: List[str],
    missing_skills: List[str],
) -> Dict[str, Any]:
    """
    Constructs the explanation dict for one candidate:
    {
      "bias": float,
      "features": [{name, value, contribution, share} x5],
      "reasons": [0-3 consistency-gated plain-English strings],
      "facts":   [always-truthful value-derived strings],
      "matched_skills": [...],
      "missing_skills": [...],
    }

    Consistency gate: a reason is only generated when BOTH the SHAP
    contribution sign AND the feature value agree on direction.
    """
    total_abs = float(np.sum(np.abs(contribs)))
    features_list = []

    for i, name in enumerate(FEATURES):
        val   = float(row_values[name])
        c     = float(contribs[i])
        share = float(c / total_abs) if total_abs > 0 else 0.0
        features_list.append({
            "name":         name,
            "value":        val,
            "contribution": c,
            "share":        share,
        })

    # Consistency-gated candidate reasons
    # Positive: contribution > 0 AND value is favorable
    # Negative: contribution < 0 AND value is unfavorable
    pos_candidates = []
    neg_candidates = []
    for f in features_list:
        name = f["name"]
        val  = f["value"]
        c    = f["contribution"]
        fav  = _is_favorable(name, val)
        if c > 0 and fav:
            pos_candidates.append(f)
        elif c < 0 and not fav:
            neg_candidates.append(f)
        # if SHAP sign disagrees with value: skip (no reason generated)

    # Sort by |contribution| descending
    pos_candidates.sort(key=lambda x: x["contribution"], reverse=True)
    neg_candidates.sort(key=lambda x: x["contribution"])  # most negative first

    reasons = []
    for p in pos_candidates[:2]:
        reasons.append(_reason_positive(p["name"], p["value"]))
    if neg_candidates:
        reasons.append(_reason_negative(neg_candidates[0]["name"], neg_candidates[0]["value"]))

    facts = build_facts(row_values, matched_skills, missing_skills)

    return {
        "bias":           float(bias),
        "features":       features_list,
        "reasons":        reasons,
        "facts":          facts,
        "matched_skills": [str(s) for s in matched_skills],
        "missing_skills": [str(s) for s in missing_skills],
    }
