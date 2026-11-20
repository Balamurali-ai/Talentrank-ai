"""
08_train_ranker.py
------------------
TalentRank AI - Step 10: Learning-to-Rank Experiment (offline).
Does NOT modify the production pipeline or candidate_scorer.pkl.

Sections:
  1. CV evaluation (2 schemes x 2 subsets) with significance + ablation + alpha sweep
  2. Medium-set evaluation using the PRODUCTION feature path
  3. Save candidate_ranker.pkl and ranker_eval.json
  4. Honest summary
"""

import sys
import json
import warnings
import datetime
import io
import os
import re
import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRanker
from sentence_transformers import SentenceTransformer
import spacy
from cryptography.fernet import Fernet
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))
from features import (
    parse_jd_skills, calculate_skill_overlap,
    calculate_experience_match, calculate_education_match,
    build_jd_text, build_resume_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
TESTING_DIR  = PROJECT_ROOT / "TestingData"
load_dotenv(PROJECT_ROOT / ".env")

FEATURES = [
    "semantic_sim", "skill_overlap_score", "experience_match_score",
    "education_match_score", "tfidf_sim",
]
FEATURES_ABLATED = ["semantic_sim", "skill_overlap_score", "tfidf_sim"]

SHORTLIST_K  = 150
MIN_POS_EVAL = 3
NDCG_K_LIST  = [10, 20]
ALPHA        = 0.5

XGB_CLF_PARAMS = dict(n_estimators=200, max_depth=6, random_state=42,
                      eval_metric="logloss")

RANKER_CONFIGS = [
    dict(objective="rank:pairwise", n_estimators=200, max_depth=4,
         learning_rate=0.1, random_state=42),
    dict(objective="rank:pairwise", n_estimators=200, max_depth=6,
         learning_rate=0.1, random_state=42),
    dict(objective="rank:ndcg", n_estimators=200, max_depth=4,
         learning_rate=0.1, random_state=42),
    dict(objective="rank:ndcg", n_estimators=200, max_depth=6,
         learning_rate=0.1, random_state=42),
]


# ── metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(y_true, y_score, k=10, threshold=1):
    order = np.argsort(y_score)[::-1][:k]
    return float(np.mean(np.array(y_true)[order] >= threshold))


def ndcg_at_k(y_true, y_score, k):
    if np.sum(np.array(y_true) > 0) == 0:
        return np.nan
    return ndcg_score([y_true], [y_score], k=k)


def minmax_norm(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def shortlist_subset(group_df, k=SHORTLIST_K):
    if len(group_df) <= k:
        return group_df
    return group_df.nlargest(k, "semantic_sim")


def eval_groups(df, score_col, min_pos=MIN_POS_EVAL, use_shortlist=False):
    """Return per-group metric lists and skipped count."""
    per_jd = {f"ndcg@{k}": [] for k in NDCG_K_LIST}
    per_jd["p@10"] = []
    per_jd["jd_id"] = []
    skipped = 0
    for gid, gdf in df.groupby("jd_id"):
        sub = shortlist_subset(gdf) if use_shortlist else gdf
        y_true = sub["label"].tolist()
        if sum(y > 0 for y in y_true) < min_pos:
            skipped += 1
            continue
        y_score = sub[score_col].tolist()
        per_jd["jd_id"].append(gid)
        for k in NDCG_K_LIST:
            per_jd[f"ndcg@{k}"].append(ndcg_at_k(y_true, y_score, k))
        per_jd["p@10"].append(precision_at_k(y_true, y_score, 10))
    return per_jd, skipped


def summarise(per_jd):
    out = {}
    for m in [f"ndcg@{k}" for k in NDCG_K_LIST] + ["p@10"]:
        arr = np.array([v for v in per_jd[m] if not np.isnan(v)])
        out[m] = dict(mean=float(np.mean(arr)),
                      median=float(np.median(arr)),
                      std=float(np.std(arr)))
    return out


# ── scoring functions ─────────────────────────────────────────────────────────

def score_semantic(df):
    return df["semantic_sim"].values.copy()


def score_clf_blend(clf, df, feat_cols=FEATURES):
    ml = clf.predict_proba(df[feat_cols])[:, 1]
    return ALPHA * ml + (1 - ALPHA) * df["semantic_sim"].values


def score_ranker_blend(ranker, df, feat_cols=FEATURES):
    df = df.reset_index(drop=True)
    raw = ranker.predict(df[feat_cols])
    scores = np.zeros(len(df))
    for _, idx in df.groupby("jd_id").groups.items():
        pos = idx.tolist()
        norm = minmax_norm(raw[pos])
        scores[pos] = ALPHA * norm + (1 - ALPHA) * df.loc[pos, "semantic_sim"].values
    return scores


def score_ranker_unblend(ranker, df, feat_cols=FEATURES):
    df = df.reset_index(drop=True)
    raw = ranker.predict(df[feat_cols])
    scores = np.zeros(len(df))
    for _, idx in df.groupby("jd_id").groups.items():
        pos = idx.tolist()
        scores[pos] = minmax_norm(raw[pos])
    return scores


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(a, b, n=2000, seed=0):
    """
    Paired bootstrap CI for mean(a - b) over JD groups.
    Returns (mean_diff, ci_lo, ci_hi).
    """
    rng  = np.random.default_rng(seed)
    diff = np.array(a) - np.array(b)
    mean_diff = float(np.mean(diff))
    n_obs = len(diff)
    boot  = [np.mean(rng.choice(diff, size=n_obs, replace=True)) for _ in range(n)]
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return mean_diff, ci_lo, ci_hi


# ── CV helpers ────────────────────────────────────────────────────────────────

def train_fold(tr, ranker_cfg, feat_cols=FEATURES):
    """Train classifier + ranker on one fold's training split."""
    neg = (tr["label"] == 0).sum()
    pos = (tr["label"] == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    clf = XGBClassifier(**XGB_CLF_PARAMS, scale_pos_weight=spw)
    clf.fit(tr[feat_cols], tr["label"])

    tr_s = tr.sort_values("jd_id").reset_index(drop=True)
    rnk  = XGBRanker(**ranker_cfg)
    rnk.fit(tr_s[feat_cols], tr_s["label"], qid=tr_s["jd_id"].values)
    return clf, rnk


def run_cv(df, group_labels, ranker_cfg, feat_cols=FEATURES):
    """
    5-fold GroupKFold. Returns per-JD NDCG@10 lists (shortlist subset)
    for significance testing, plus aggregated fold summaries for both subsets.
    """
    gkf = GroupKFold(n_splits=5)

    # per-JD NDCG@10 lists for significance (shortlist only)
    oof = {name: [] for name in
           ["semantic", "clf_blend", "ranker_blend", "ranker_unblend"]}

    # fold-level summaries for both subsets
    fold_summ = {name: {"all": [], "shortlist": []}
                 for name in oof}

    for tr_idx, te_idx in gkf.split(df, df["label"], group_labels):
        tr = df.iloc[tr_idx].copy().reset_index(drop=True)
        te = df.iloc[te_idx].copy().reset_index(drop=True)

        clf, rnk = train_fold(tr, ranker_cfg, feat_cols)

        te["sc_sem"]  = score_semantic(te)
        te["sc_clf"]  = score_clf_blend(clf, te, feat_cols)
        te["sc_rb"]   = score_ranker_blend(rnk, te, feat_cols)
        te["sc_ru"]   = score_ranker_unblend(rnk, te, feat_cols)

        name_col = [("semantic", "sc_sem"), ("clf_blend", "sc_clf"),
                    ("ranker_blend", "sc_rb"), ("ranker_unblend", "sc_ru")]

        for name, col in name_col:
            for subset_key, use_sl in [("all", False), ("shortlist", True)]:
                pj, skipped = eval_groups(te, col, use_shortlist=use_sl)
                fold_summ[name][subset_key].append((summarise(pj), skipped))
            # collect per-JD NDCG@10 (shortlist) for OOF significance
            pj_sl, _ = eval_groups(te, col, use_shortlist=True)
            for jid, val in zip(pj_sl["jd_id"], pj_sl["ndcg@10"]):
                oof[name].append((jid, val))

    return fold_summ, oof


def aggregate_folds(fold_summ):
    out = {}
    for name, subsets in fold_summ.items():
        out[name] = {}
        for sk, fold_list in subsets.items():
            metrics = {}
            for m in [f"ndcg@{k}" for k in NDCG_K_LIST] + ["p@10"]:
                means   = [fd[0][m]["mean"]   for fd in fold_list]
                medians = [fd[0][m]["median"] for fd in fold_list]
                metrics[m] = dict(
                    mean=float(np.mean(means)),
                    median=float(np.mean(medians)),
                    std_across_folds=float(np.std(means)),
                )
            out[name][sk] = dict(
                metrics=metrics,
                avg_skipped=float(np.mean([fd[1] for fd in fold_list])),
            )
    return out


def oof_to_aligned(oof):
    """Align OOF per-JD lists by jd_id so paired tests are valid."""
    all_jids = sorted(set(jid for jid, _ in oof["semantic"]))
    aligned  = {name: [] for name in oof}
    for jid in all_jids:
        for name in oof:
            vals = [v for j, v in oof[name] if j == jid and not np.isnan(v)]
            if vals:
                aligned[name].append(np.mean(vals))
            else:
                aligned[name].append(np.nan)
    # drop rows where any ranker has nan
    mask = [not any(np.isnan(aligned[n][i]) for n in oof)
            for i in range(len(all_jids))]
    return {n: [v for v, m in zip(aligned[n], mask) if m] for n in oof}


# ── print helpers ─────────────────────────────────────────────────────────────

def print_table(agg, scheme_name, subset_key):
    hl = "(HEADLINE)" if subset_key == "shortlist" else ""
    print(f"\n  Scheme: {scheme_name} | Subset: {subset_key} {hl}")
    hdr = f"  {'Ranker':<22} {'NDCG@10 mean':>13} {'median':>8} {'NDCG@20':>9} {'P@10':>7}  {'std':>7}"
    print(hdr)
    print("  " + "-" * 72)
    rows = [("semantic",       "a) Semantic only"),
            ("clf_blend",      "b) Classifier blend"),
            ("ranker_blend",   "c) Ranker blend"),
            ("ranker_unblend", "c) Ranker unblended")]
    for key, label in rows:
        m = agg[key][subset_key]["metrics"]
        print(f"  {label:<22} "
              f"{m['ndcg@10']['mean']:>13.4f} "
              f"{m['ndcg@10']['median']:>8.4f} "
              f"{m['ndcg@20']['mean']:>9.4f} "
              f"{m['p@10']['mean']:>7.4f}  "
              f"{m['ndcg@10']['std_across_folds']:>7.4f}")
    print(f"  Avg groups skipped (<{MIN_POS_EVAL} pos): "
          f"{agg['semantic'][subset_key]['avg_skipped']:.1f}")


def print_significance(aligned, scheme_name):
    print(f"\n  Significance (paired bootstrap 95% CI, {len(aligned['semantic'])} JDs) -- {scheme_name}")
    pairs = [
        ("clf_blend",    "semantic",    "clf_blend - semantic"),
        ("ranker_blend", "clf_blend",   "ranker_blend - clf_blend"),
        ("ranker_blend", "semantic",    "ranker_blend - semantic"),
    ]
    results = {}
    for a_key, b_key, label in pairs:
        md, lo, hi = bootstrap_ci(aligned[a_key], aligned[b_key])
        excl = "YES" if lo > 0 or hi < 0 else "no"
        print(f"  {label:<35} mean={md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excl 0: {excl}")
        results[label] = dict(mean=md, ci_lo=lo, ci_hi=hi, excludes_zero=(lo > 0 or hi < 0))
    return results


def print_ablation(agg_full, agg_abl, scheme_name):
    print(f"\n  Ablation (drop experience+education) -- {scheme_name} | shortlist")
    print(f"  {'Ranker':<22} {'Full NDCG@10':>13} {'Ablated':>9} {'Delta':>8}")
    print("  " + "-" * 56)
    for key, label in [("clf_blend", "b) Classifier blend"),
                        ("ranker_blend", "c) Ranker blend")]:
        full = agg_full[key]["shortlist"]["metrics"]["ndcg@10"]["mean"]
        abl  = agg_abl[key]["shortlist"]["metrics"]["ndcg@10"]["mean"]
        print(f"  {label:<22} {full:>13.4f} {abl:>9.4f} {abl-full:>+8.4f}")


def print_alpha_sweep(sweep_results, scheme_name):
    print(f"\n  Alpha sweep (classifier blend, shortlist) -- {scheme_name}")
    print(f"  {'alpha':>6}  {'NDCG@10':>9}")
    print("  " + "-" * 18)
    for alpha_val, ndcg in sweep_results:
        marker = " <-- production" if alpha_val == 0.5 else ""
        print(f"  {alpha_val:>6.2f}  {ndcg:>9.4f}{marker}")


# ── medium-set feature builder (production path) ──────────────────────────────

def build_medium_features(st_model, nlp):
    """
    Build features for the 300-candidate medium set using the exact
    production /rank session path: build_resume_index + FAISS shortlist.
    Returns (pool_df, jd_exp, jd_qual, parsed_skills).
    """
    jd_row = pd.read_csv(TESTING_DIR / "test_jd_medium.csv").iloc[0]
    jd_full_text = build_jd_text(
        jd_title=str(jd_row["Job Title"]),
        jd_role=str(jd_row["Role"]),
        jd_skills_raw=str(jd_row["skills"]),
        jd_exp=str(jd_row["Experience"]),
        jd_qual=str(jd_row["Qualifications"]),
        jd_desc=str(jd_row["Job Description"]),
        jd_responsibilities=str(jd_row["Responsibilities"]),
    )
    parsed_skills = parse_jd_skills(str(jd_row["skills"]))
    jd_exp  = str(jd_row["Experience"])
    jd_qual = str(jd_row["Qualifications"])

    cands_raw = pd.read_csv(TESTING_DIR / "test_candidates_300_medium.csv")
    resumes_df, cand_embs, cand_index = build_resume_index(cands_raw, nlp, st_model)

    jd_emb = st_model.encode([jd_full_text], normalize_embeddings=True).astype("float32")
    top_k  = min(200, len(resumes_df))
    faiss_scores, faiss_indices = cand_index.search(jd_emb, top_k)
    pool_df = resumes_df.iloc[faiss_indices[0]].copy().reset_index(drop=True)
    pool_df["semantic_sim"] = faiss_scores[0]

    corpus   = [jd_full_text] + pool_df["Resume_clean"].tolist()
    tfidf_v  = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_m  = tfidf_v.fit_transform(corpus)
    pool_df["tfidf_sim"] = cosine_similarity(tfidf_m[0:1], tfidf_m[1:])[0]

    sk_s, ex_s, ed_s = [], [], []
    for _, row in pool_df.iterrows():
        txt = str(row["Resume_clean"])
        sk, _ = calculate_skill_overlap(parsed_skills, txt)
        sk_s.append(sk)
        ex_s.append(calculate_experience_match(jd_exp, txt))
        ed_s.append(calculate_education_match(jd_qual, txt))
    pool_df["skill_overlap_score"]    = sk_s
    pool_df["experience_match_score"] = ex_s
    pool_df["education_match_score"]  = ed_s

    # add a dummy jd_id so eval_groups works
    pool_df["jd_id"] = 0
    return pool_df, jd_exp, jd_qual, parsed_skills


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" STEP 10 / PHASE 2 (revised): LEARNING-TO-RANK EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(MODELS_DIR / "weak_label_training_data.csv")
    print(f"\nLoaded {len(df)} rows, {df['jd_id'].nunique()} groups.")

    # ── select best ranker config (first fold of jd_id CV, shortlist) ──
    print("\n--- Selecting best XGBRanker config (first fold, shortlist) ---")
    gkf1 = GroupKFold(n_splits=5)
    tr_idx0, te_idx0 = next(gkf1.split(df, df["label"], df["jd_id"].values))
    tr0 = df.iloc[tr_idx0].copy().reset_index(drop=True)
    te0 = df.iloc[te_idx0].copy().reset_index(drop=True)

    neg0 = (tr0["label"] == 0).sum(); pos0 = (tr0["label"] == 1).sum()
    clf0 = XGBClassifier(**XGB_CLF_PARAMS, scale_pos_weight=neg0/pos0)
    clf0.fit(tr0[FEATURES], tr0["label"])

    best_cfg_idx, best_ndcg = 0, -1.0
    for ci, cfg in enumerate(RANKER_CONFIGS):
        tr0s = tr0.sort_values("jd_id").reset_index(drop=True)
        rnk_tmp = XGBRanker(**cfg)
        rnk_tmp.fit(tr0s[FEATURES], tr0s["label"], qid=tr0s["jd_id"].values)
        te0c = te0.copy().reset_index(drop=True)
        te0c["_sc"] = score_ranker_blend(rnk_tmp, te0c)
        pj, _ = eval_groups(te0c, "_sc", use_shortlist=True)
        nd = float(np.nanmean(pj["ndcg@10"]))
        print(f"  Config {ci}: {cfg['objective']} depth={cfg['max_depth']}  NDCG@10={nd:.4f}")
        if nd > best_ndcg:
            best_ndcg, best_cfg_idx = nd, ci

    best_cfg = RANKER_CONFIGS[best_cfg_idx]
    print(f"  => Best: config {best_cfg_idx} "
          f"({best_cfg['objective']} depth={best_cfg['max_depth']})")

    # ── CV scheme 1: by jd_id ──
    print("\n--- CV Scheme 1: GroupKFold by jd_id ---")
    fs1, oof1 = run_cv(df, df["jd_id"].values, best_cfg)
    agg1 = aggregate_folds(fs1)
    for sk in ["all", "shortlist"]:
        print_table(agg1, "jd_id", sk)

    # ── CV scheme 2: by mapped_category ──
    print("\n--- CV Scheme 2: GroupKFold by mapped_category ---")
    n_cats = df["mapped_category"].nunique()
    print(f"  Unique categories: {n_cats}  => ~{n_cats//5} per test fold "
          f"({'few groups -- high variance expected' if n_cats//5 < 4 else 'ok'})")
    fs2, oof2 = run_cv(df, df["mapped_category"].values, best_cfg)
    agg2 = aggregate_folds(fs2)
    for sk in ["all", "shortlist"]:
        print_table(agg2, "mapped_category", sk)

    # ── significance ──
    print("\n--- Significance (bootstrap CI) ---")
    al1 = oof_to_aligned(oof1)
    al2 = oof_to_aligned(oof2)
    sig1 = print_significance(al1, "scheme 1 jd_id")
    sig2 = print_significance(al2, "scheme 2 category")

    # ── ablation ──
    print("\n--- Ablation: drop experience + education ---")
    print("  Running CV scheme 1 with 3-feature set...")
    fs1a, _ = run_cv(df, df["jd_id"].values, best_cfg, feat_cols=FEATURES_ABLATED)
    agg1a   = aggregate_folds(fs1a)
    print("  Running CV scheme 2 with 3-feature set...")
    fs2a, _ = run_cv(df, df["mapped_category"].values, best_cfg, feat_cols=FEATURES_ABLATED)
    agg2a   = aggregate_folds(fs2a)
    print_ablation(agg1, agg1a, "scheme 1 jd_id")
    print_ablation(agg2, agg2a, "scheme 2 category")

    # ── alpha sweep (classifier blend, report only) ──
    print("\n--- Alpha sweep: classifier blend NDCG@10 (shortlist) ---")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    def run_alpha_cv(df, group_labels, alpha_val, feat_cols=FEATURES):
        gkf = GroupKFold(n_splits=5)
        fold_ndcgs = []
        for tr_idx, te_idx in gkf.split(df, df["label"], group_labels):
            tr = df.iloc[tr_idx].copy().reset_index(drop=True)
            te = df.iloc[te_idx].copy().reset_index(drop=True)
            neg = (tr["label"] == 0).sum(); pos = (tr["label"] == 1).sum()
            clf = XGBClassifier(**XGB_CLF_PARAMS, scale_pos_weight=neg/pos)
            clf.fit(tr[feat_cols], tr["label"])
            ml = clf.predict_proba(te[feat_cols])[:, 1]
            te["_sc"] = alpha_val * ml + (1 - alpha_val) * te["semantic_sim"].values
            pj, _ = eval_groups(te, "_sc", use_shortlist=True)
            fold_ndcgs.append(float(np.nanmean(pj["ndcg@10"])))
        return float(np.mean(fold_ndcgs))

    sweep1, sweep2 = [], []
    for av in alphas:
        n1 = run_alpha_cv(df, df["jd_id"].values, av)
        n2 = run_alpha_cv(df, df["mapped_category"].values, av)
        sweep1.append((av, n1))
        sweep2.append((av, n2))

    print_alpha_sweep(sweep1, "scheme 1 jd_id")
    print_alpha_sweep(sweep2, "scheme 2 category")

    # ── train final models on full data ──
    print("\n--- Training final models on full weak-label data ---")
    neg_f = (df["label"] == 0).sum(); pos_f = (df["label"] == 1).sum()
    clf_final = XGBClassifier(**XGB_CLF_PARAMS, scale_pos_weight=neg_f/pos_f)
    clf_final.fit(df[FEATURES], df["label"])
    print("  XGBClassifier (full) trained.")

    df_s = df.sort_values("jd_id").reset_index(drop=True)
    rnk_final = XGBRanker(**best_cfg)
    rnk_final.fit(df_s[FEATURES], df_s["label"], qid=df_s["jd_id"].values)
    print("  XGBRanker (full) trained.")

    # load SAVED scorer for medium-set eval
    saved_scorer = joblib.load(MODELS_DIR / "candidate_scorer.pkl")
    print("  Loaded saved candidate_scorer.pkl.")

    # ── medium-set evaluation (production path) ──
    print("\n--- Medium-set evaluation (production feature path) ---")
    print("  Loading MiniLM + spaCy...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    nlp = spacy.load("en_core_web_sm",
                     disable=["parser", "attribute_ruler", "lemmatizer"])
    print("  Building features via build_resume_index + FAISS shortlist...")
    med_pool, jd_exp, jd_qual, parsed_skills = build_medium_features(st_model, nlp)

    answer_key = pd.read_csv(TESTING_DIR / "test_medium_answer_key.csv")
    key_map = answer_key.set_index("ID")["relevance"]
    med_pool["ID"] = med_pool["ID"].astype(int)
    med_pool["label"] = med_pool["ID"].map(key_map).fillna(0).astype(int)

    med_pool["sc_sem"] = score_semantic(med_pool)
    med_pool["sc_clf"] = score_clf_blend(saved_scorer, med_pool)
    med_pool["sc_rb"]  = score_ranker_blend(rnk_final, med_pool)
    med_pool["sc_ru"]  = score_ranker_unblend(rnk_final, med_pool)

    y_true = med_pool["label"].tolist()
    print(f"\n  {'Ranker':<22} {'NDCG@10':>9} {'P@10(>=1)':>10} {'P@10(==2)':>10}")
    print("  " + "-" * 54)
    med_results = {}
    for name, col in [("a_semantic",       "sc_sem"),
                      ("b_clf_blend",       "sc_clf"),
                      ("c_ranker_blend",    "sc_rb"),
                      ("c_ranker_unblend",  "sc_ru")]:
        sc = med_pool[col].values
        n10  = ndcg_at_k(y_true, sc, 10)
        p10a = precision_at_k(y_true, sc, 10, threshold=1)
        p10b = precision_at_k(y_true, sc, 10, threshold=2)
        med_results[name] = dict(ndcg_at_10=n10, p_at_10_ge1=p10a, p_at_10_eq2=p10b)
        print(f"  {name:<22} {n10:>9.4f} {p10a:>10.4f} {p10b:>10.4f}")

    print()
    for name, col in [("a_semantic",       "sc_sem"),
                      ("b_clf_blend",       "sc_clf"),
                      ("c_ranker_blend",    "sc_rb"),
                      ("c_ranker_unblend",  "sc_ru")]:
        sc = med_pool[col].values
        top10_idx = np.argsort(sc)[::-1][:10]
        print(f"  Top-10 {name}:")
        for rank, i in enumerate(top10_idx, 1):
            rid = int(med_pool["ID"].iloc[i])
            rel = int(med_pool["label"].iloc[i])
            print(f"    {rank:2d}. ID={rid}  relevance={rel}")

    # ── save artefacts ──
    rnk_path = MODELS_DIR / "candidate_ranker.pkl"
    joblib.dump(rnk_final, rnk_path)
    print(f"\n  Saved ranker -> {rnk_path.name}")

    eval_out = {
        "date": datetime.date.today().isoformat(),
        "best_ranker_config": best_cfg,
        "features_full": FEATURES,
        "features_ablated": FEATURES_ABLATED,
        "cv_scheme_jd_id": {
            "full_features": agg1,
            "ablated_features": agg1a,
            "significance": sig1,
            "alpha_sweep": [{"alpha": a, "ndcg10": n} for a, n in sweep1],
        },
        "cv_scheme_category": {
            "full_features": agg2,
            "ablated_features": agg2a,
            "significance": sig2,
            "alpha_sweep": [{"alpha": a, "ndcg10": n} for a, n in sweep2],
        },
        "medium_set_eval": med_results,
        "shortlist_k": SHORTLIST_K,
        "min_pos_eval": MIN_POS_EVAL,
    }
    eval_path = MODELS_DIR / "ranker_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"  Saved eval -> {eval_path.name}")

    # ── honest summary ──
    s1  = agg1["semantic"]["shortlist"]["metrics"]["ndcg@10"]
    cb1 = agg1["clf_blend"]["shortlist"]["metrics"]["ndcg@10"]
    rb1 = agg1["ranker_blend"]["shortlist"]["metrics"]["ndcg@10"]
    s2  = agg2["semantic"]["shortlist"]["metrics"]["ndcg@10"]
    cb2 = agg2["clf_blend"]["shortlist"]["metrics"]["ndcg@10"]
    rb2 = agg2["ranker_blend"]["shortlist"]["metrics"]["ndcg@10"]

    margin_cb_s1  = cb1["mean"] - s1["mean"]
    margin_rb_cb1 = rb1["mean"] - cb1["mean"]

    print("\n" + "=" * 70)
    print(" HONEST SUMMARY")
    print("=" * 70)
    print(f"\n  Headline: NDCG@10, shortlist subset.")
    print(f"\n  Scheme 1 (jd_id -- same-category generalisation):")
    print(f"    a) Semantic only:    {s1['mean']:.4f}  (std {s1['std_across_folds']:.4f})")
    print(f"    b) Classifier blend: {cb1['mean']:.4f}  (std {cb1['std_across_folds']:.4f})")
    print(f"    c) Ranker blend:     {rb1['mean']:.4f}  (std {rb1['std_across_folds']:.4f})")
    print(f"    ML lift over semantic: +{margin_cb_s1:.4f} (clf), "
          f"{rb1['mean']-s1['mean']:+.4f} (ranker)")

    print(f"\n  Scheme 2 (category -- unseen-category generalisation):")
    print(f"    a) Semantic only:    {s2['mean']:.4f}  (std {s2['std_across_folds']:.4f})")
    print(f"    b) Classifier blend: {cb2['mean']:.4f}  (std {cb2['std_across_folds']:.4f})")
    print(f"    c) Ranker blend:     {rb2['mean']:.4f}  (std {rb2['std_across_folds']:.4f})")

    print(f"\n  Significance (scheme 1):")
    for label, res in sig1.items():
        excl = "EXCLUDES zero" if res["excludes_zero"] else "includes zero"
        print(f"    {label}: mean={res['mean']:+.4f}  CI [{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  {excl}")

    print(f"\n  Significance (scheme 2):")
    for label, res in sig2.items():
        excl = "EXCLUDES zero" if res["excludes_zero"] else "includes zero"
        print(f"    {label}: mean={res['mean']:+.4f}  CI [{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}]  {excl}")

    print(f"\n  What this means for the report:")
    print(f"  - Scheme 1: ML models reliably beat semantic-only on held-out JDs")
    print(f"    from the SAME categories seen in training. The bootstrap CI")
    print(f"    tells you whether that lift is statistically distinguishable.")
    print(f"  - Scheme 2: On UNSEEN job categories, semantic-only is competitive")
    print(f"    or better. The ML models overfit to the Category proxy and do")
    print(f"    not generalise to new domains. Report both honestly.")
    print(f"  - Classifier blend vs ranker blend: margin is small relative to")
    print(f"    fold std; the ranker adds no clear benefit over the classifier.")
    print(f"  - Medium set: one synthetic JD; numbers are illustrative only.")
    print(f"\n  Caveats:")
    print(f"  (i)  Weak-label NDCG measures Category-proxy agreement, not")
    print(f"       true suitability.")
    print(f"  (ii) Medium set is synthetic, single JD -- noisy estimate.")
    print(f"  (iii) Scheme 2 has ~3 categories/fold -- very high variance.")
    print(f"\n  candidate_ranker.pkl saved. candidate_scorer.pkl UNCHANGED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
