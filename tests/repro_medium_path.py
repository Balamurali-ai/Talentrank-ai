"""
tests/repro_medium_path.py
--------------------------
Reproduces the exact production /rank path (session upload) for the
300-candidate medium test set, without starting the server.

Steps mirror 06_api.py exactly:
  1. build_resume_index  (PII mask + MiniLM embed + FAISS)
  2. build_jd_text       (canonical JD string)
  3. FAISS shortlist     min(200, n)
  4. TF-IDF, skill, experience, education features
  5. candidate_scorer.pkl  predict_proba
  6. blend alpha=0.5
  7. top-10 with relevance from answer key
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import faiss
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import (
    parse_jd_skills, calculate_skill_overlap,
    calculate_experience_match, calculate_education_match,
    build_jd_text, build_resume_index,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
TESTING_DIR  = PROJECT_ROOT / "TestingData"

ALPHA = 0.5
FEATURE_COLS = [
    "semantic_sim", "skill_overlap_score", "experience_match_score",
    "education_match_score", "tfidf_sim",
]

print("=" * 65)
print(" REPRO: medium-set via production feature path (no server)")
print("=" * 65)

# ── load JD ──────────────────────────────────────────────────────────
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
print(f"\nJD text (first 120 chars): {jd_full_text[:120]}")
print(f"Parsed skills: {parsed_skills}")

# ── load models ───────────────────────────────────────────────────────
print("\nLoading MiniLM + spaCy + scorer...")
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
scorer = joblib.load(MODELS_DIR / "candidate_scorer.pkl")

# ── build_resume_index for the 300 candidates ─────────────────────────
print("Building resume index (PII mask + embed + FAISS)...")
cands_raw = pd.read_csv(TESTING_DIR / "test_candidates_300_medium.csv")
resumes_df, cand_embs, cand_index = build_resume_index(cands_raw, nlp, st_model)
print(f"  {len(resumes_df)} resumes indexed.")

# ── embed JD and FAISS shortlist ──────────────────────────────────────
jd_emb = st_model.encode([jd_full_text], normalize_embeddings=True).astype("float32")
top_k = min(200, len(resumes_df))
faiss_scores, faiss_indices = cand_index.search(jd_emb, top_k)
shortlist_indices = faiss_indices[0]
semantic_sims     = faiss_scores[0]

pool_df = resumes_df.iloc[shortlist_indices].copy().reset_index(drop=True)
pool_df["semantic_sim"] = semantic_sims

# ── TF-IDF ────────────────────────────────────────────────────────────
corpus = [jd_full_text] + pool_df["Resume_clean"].tolist()
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_mat = tfidf_vec.fit_transform(corpus)
pool_df["tfidf_sim"] = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]

# ── skill / experience / education ────────────────────────────────────
# Production uses payload.experience and payload.qualifications directly
jd_exp  = str(jd_row["Experience"])
jd_qual = str(jd_row["Qualifications"])

skill_scores, exp_scores, edu_scores = [], [], []
for _, row in pool_df.iterrows():
    txt = str(row["Resume_clean"])
    sk, _ = calculate_skill_overlap(parsed_skills, txt)
    skill_scores.append(sk)
    exp_scores.append(calculate_experience_match(jd_exp, txt))
    edu_scores.append(calculate_education_match(jd_qual, txt))

pool_df["skill_overlap_score"]    = skill_scores
pool_df["experience_match_score"] = exp_scores
pool_df["education_match_score"]  = edu_scores

# ── score with saved classifier ───────────────────────────────────────
pool_df["ML_score"]    = scorer.predict_proba(pool_df[FEATURE_COLS])[:, 1]
pool_df["final_score"] = ALPHA * pool_df["ML_score"] + (1 - ALPHA) * pool_df["semantic_sim"]

# ── top-10 ────────────────────────────────────────────────────────────
top10 = pool_df.sort_values("final_score", ascending=False).head(10).reset_index(drop=True)
answer_key = pd.read_csv(TESTING_DIR / "test_medium_answer_key.csv")
key_map = answer_key.set_index("ID")["relevance"]

print("\nTop-10 (production path, saved scorer):")
print(f"  {'Rank':>4}  {'ID':>8}  {'rel':>3}  {'sem_sim':>8}  {'ML_score':>9}  {'final':>8}")
print("  " + "-" * 55)
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    rid = int(row["ID"])
    rel = key_map.get(rid, -1)
    print(f"  {rank:>4}  {rid:>8}  {rel:>3}  "
          f"{row['semantic_sim']:>8.4f}  {row['ML_score']:>9.4f}  {row['final_score']:>8.4f}")

top10_ids = top10["ID"].astype(int).tolist()
top10_rels = [key_map.get(i, -1) for i in top10_ids]
n_rel2 = sum(r == 2 for r in top10_rels)
n_rel1 = sum(r >= 1 for r in top10_rels)
print(f"\n  relevance==2 in top-10: {n_rel2}/10")
print(f"  relevance>=1 in top-10: {n_rel1}/10")

# ── save pool_df for comparison in 08_train_ranker.py ─────────────────
pool_df.to_csv(MODELS_DIR / "_repro_medium_pool.csv", index=False)
print(f"\nFull pool saved to models/_repro_medium_pool.csv for feature comparison.")
print("Done.")
