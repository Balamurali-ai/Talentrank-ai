"""
tests/seed_stability.py
-----------------------
No-server seed stability check.

Trains XGBClassifier with the exact settings from src/04_train_model.py
on the full weak-label data using 5 different random_state values (0-4).
For each seed, scores the medium set through the production feature path
used in src/08_train_ranker.py (classifier blend, alpha=0.5) and reports
NDCG@10 and P@10(relevance==2) against the answer key.

Also reports the saved models/candidate_scorer.pkl result for comparison.
Does NOT modify any existing file.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import spacy

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TESTING_DIR = ROOT / "TestingData"

sys.path.insert(0, str(ROOT / "src"))
from features import (
    parse_jd_skills, calculate_skill_overlap,
    calculate_experience_match, calculate_education_match,
    build_jd_text, build_resume_index,
)

FEATURES = [
    "semantic_sim", "skill_overlap_score", "experience_match_score",
    "education_match_score", "tfidf_sim",
]
ALPHA = 0.5
SEEDS = [0, 1, 2, 3, 4]


# ── metric helpers ─────────────────────────────────────────────────────────────

def ndcg_at_k(y_true, y_score, k=10):
    if np.sum(np.array(y_true) > 0) == 0:
        return float("nan")
    return float(ndcg_score([y_true], [y_score], k=k))


def precision_at_k(y_true, y_score, k=10, threshold=2):
    order = np.argsort(y_score)[::-1][:k]
    return float(np.mean(np.array(y_true)[order] >= threshold))


# ── build medium-set feature pool (production path, run once) ─────────────────

def build_pool(st_model, nlp):
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
    resumes_df, _, cand_index = build_resume_index(cands_raw, nlp, st_model)

    jd_emb = st_model.encode([jd_full_text], normalize_embeddings=True).astype("float32")
    top_k  = min(200, len(resumes_df))
    faiss_scores, faiss_indices = cand_index.search(jd_emb, top_k)
    pool_df = resumes_df.iloc[faiss_indices[0]].copy().reset_index(drop=True)
    pool_df["semantic_sim"] = faiss_scores[0]

    corpus  = [jd_full_text] + pool_df["Resume_clean"].tolist()
    tfidf_v = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_m = tfidf_v.fit_transform(corpus)
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

    answer_key = pd.read_csv(TESTING_DIR / "test_medium_answer_key.csv")
    key_map = answer_key.set_index("ID")["relevance"]
    pool_df["ID"]    = pool_df["ID"].astype(int)
    pool_df["label"] = pool_df["ID"].map(key_map).fillna(0).astype(int)

    return pool_df


# ── score one classifier on the pool ──────────────────────────────────────────

def score_pool(clf, pool_df):
    ml    = clf.predict_proba(pool_df[FEATURES])[:, 1]
    final = ALPHA * ml + (1 - ALPHA) * pool_df["semantic_sim"].values
    y_true = pool_df["label"].tolist()
    return ndcg_at_k(y_true, final, k=10), precision_at_k(y_true, final, k=10, threshold=2)


# ── train one classifier with 04_train_model.py exact settings ────────────────

def train_clf(df, seed):
    """
    Replicates 04_train_model.py exactly:
      - 80/20 stratified split with random_state=seed
      - scale_pos_weight from the train split
      - XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss',
                      random_state=seed, scale_pos_weight=spw)
      - fit on X_train only (same as 04, which trains on the split)
    """
    X = df[FEATURES]
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    clf = XGBClassifier(
        n_estimators=200, max_depth=6,
        scale_pos_weight=spw,
        random_state=seed,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    return clf, spw


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" SEED STABILITY CHECK")
    print(" XGBClassifier (04_train_model.py settings), seeds 0-4")
    print(" Medium set, production feature path (08_train_ranker.py)")
    print("=" * 65)

    df = pd.read_csv(MODELS_DIR / "weak_label_training_data.csv")
    print(f"\nTraining data: {len(df)} rows, {df['label'].sum()} positives "
          f"({df['label'].mean()*100:.1f}%)")

    print("\nLoading MiniLM + spaCy (once)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    nlp = spacy.load("en_core_web_sm",
                     disable=["parser", "attribute_ruler", "lemmatizer"])

    print("Building medium-set feature pool (production path)...")
    pool_df = build_pool(st_model, nlp)
    print(f"Pool size: {len(pool_df)} candidates  "
          f"(rel=2: {(pool_df['label']==2).sum()}, "
          f"rel=1: {(pool_df['label']==1).sum()}, "
          f"rel=0: {(pool_df['label']==0).sum()})")

    print("\n" + "-" * 65)
    print(f"  {'Source':<28} {'NDCG@10':>9}  {'P@10(==2)':>10}")
    print("-" * 65)

    ndcg_vals = []
    p10_vals  = []

    for seed in SEEDS:
        clf, spw = train_clf(df, seed)
        ndcg, p10 = score_pool(clf, pool_df)
        ndcg_vals.append(ndcg)
        p10_vals.append(p10)
        print(f"  seed={seed}  spw={spw:.4f}  "
              f"{'XGBClassifier (80% split)':<18} "
              f"{ndcg:>9.4f}  {p10:>10.4f}")

    print("-" * 65)
    print(f"  {'mean (seeds 0-4)':<28} {np.mean(ndcg_vals):>9.4f}  "
          f"{np.mean(p10_vals):>10.4f}")
    print(f"  {'min  (seeds 0-4)':<28} {np.min(ndcg_vals):>9.4f}  "
          f"{np.min(p10_vals):>10.4f}")
    print(f"  {'max  (seeds 0-4)':<28} {np.max(ndcg_vals):>9.4f}  "
          f"{np.max(p10_vals):>10.4f}")
    print(f"  {'range':<28} {np.max(ndcg_vals)-np.min(ndcg_vals):>9.4f}  "
          f"{np.max(p10_vals)-np.min(p10_vals):>10.4f}")

    print("-" * 65)
    saved = joblib.load(MODELS_DIR / "candidate_scorer.pkl")
    ndcg_s, p10_s = score_pool(saved, pool_df)
    print(f"  {'saved candidate_scorer.pkl':<28} {ndcg_s:>9.4f}  {p10_s:>10.4f}")
    print(f"  (trained on 80% split, random_state=42)")
    print("-" * 65)

    print("\nNote: saved scorer uses random_state=42 on an 80% split.")
    print("      Seeds 0-4 also use 80% splits with their respective seeds.")
    print("      Variation across seeds reflects model sensitivity to")
    print("      training-set composition, not a parameter difference.")
    print("=" * 65)


if __name__ == "__main__":
    main()
