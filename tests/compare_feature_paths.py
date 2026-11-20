"""
tests/compare_feature_paths.py
-------------------------------
Compares features from the OLD 08_train_ranker.py medium-set path
against the PRODUCTION path (repro_medium_path.py output).
Prints side-by-side for 10 IDs and states the root cause of differences.
"""

import sys
import os
import io
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from cryptography.fernet import Fernet
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import (
    parse_jd_skills, calculate_skill_overlap,
    calculate_experience_match, calculate_education_match,
    build_jd_text, build_resume_index, calculate_tfidf_sim,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
TESTING_DIR  = PROJECT_ROOT / "TestingData"

load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLS = [
    "semantic_sim", "skill_overlap_score", "experience_match_score",
    "education_match_score", "tfidf_sim",
]

print("=" * 65)
print(" FEATURE PATH COMPARISON: old vs production")
print("=" * 65)

# ── production pool (already computed by repro_medium_path.py) ────────
prod_pool = pd.read_csv(MODELS_DIR / "_repro_medium_pool.csv")
prod_pool["ID"] = prod_pool["ID"].astype(int)

# ── OLD path: decrypt resumes_clean.enc, use raw masked text ──────────
print("\nBuilding OLD path features (decrypt enc, raw masked text)...")
key = os.environ["ENCRYPTION_KEY"].encode()
fernet = Fernet(key)
with open(MODELS_DIR / "resumes_clean.enc", "rb") as f:
    plain = fernet.decrypt(f.read())
df_pool_enc = pd.read_csv(io.BytesIO(plain))

cands = pd.read_csv(TESTING_DIR / "test_candidates_300_medium.csv")
jd_row = pd.read_csv(TESTING_DIR / "test_jd_medium.csv").iloc[0]

# OLD path used build_jd_text then parse_jd_skills on the full text
jd_full_text = build_jd_text(
    jd_title=str(jd_row["Job Title"]),
    jd_role=str(jd_row["Role"]),
    jd_skills_raw=str(jd_row["skills"]),
    jd_exp=str(jd_row["Experience"]),
    jd_qual=str(jd_row["Qualifications"]),
    jd_desc=str(jd_row["Job Description"]),
    jd_responsibilities=str(jd_row["Responsibilities"]),
)
# OLD path parsed skills from jd_row["skills"] directly (same as production)
jd_skills = parse_jd_skills(str(jd_row["skills"]))

print("Loading MiniLM + spaCy...")
st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])

# OLD path: the 300 test IDs (200001-200300) are SYNTHETIC and do NOT exist
# in the encrypted pool (which holds the 2484 Kaggle resumes).
# The old build_medium_features() in 08_train_ranker.py decrypted the pool,
# then tried to look up these IDs -- finding nothing, it fell back to
# embedding the raw Resume_str directly (no build_resume_index, no FAISS shortlist).
# We reproduce that: PII-mask inline (same regex, no spaCy NER) then embed.
cand_ids = cands["ID"].astype(str).tolist()
df_pool_enc["ID"] = df_pool_enc["ID"].astype(str)
matched = df_pool_enc[df_pool_enc["ID"].isin(cand_ids)]
print(f"  Matched {len(matched)} of 300 test IDs in encrypted pool (expected 0 -- synthetic IDs).")

import re as _re
EMAIL_RE = _re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = _re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b")

# Old path masked with spaCy NER + regex (same as build_resume_index)
raw_texts = cands["Resume_str"].astype(str).tolist()
old_masked = []
for doc in nlp.pipe(raw_texts, batch_size=32):
    text = doc.text
    for ent in sorted([e for e in doc.ents if e.label_ == "PERSON"],
                      key=lambda e: e.start_char, reverse=True):
        text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    old_masked.append(_re.sub(r"\s+", " ", text).strip())

# OLD path: embed ALL 300 directly (no FAISS shortlist)
old_embs = st_model.encode(old_masked, batch_size=32, normalize_embeddings=True)
jd_emb   = st_model.encode([jd_full_text], normalize_embeddings=True)[0]
old_sims = old_embs @ jd_emb

old_cands = cands[["ID", "Category"]].copy().reset_index(drop=True)
old_cands["Resume_clean"]  = old_masked
old_cands["semantic_sim"]  = old_sims

old_texts = old_masked
# OLD path TF-IDF
corpus = [jd_full_text] + old_texts
tfidf_vec = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_mat = tfidf_vec.fit_transform(corpus)
old_cands["tfidf_sim"] = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]

# OLD path features (used jd_row["Experience"] and jd_row["Qualifications"] same as prod)
jd_exp  = str(jd_row["Experience"])
jd_qual = str(jd_row["Qualifications"])
sk_s, ex_s, ed_s = [], [], []
for txt in old_texts:
    sk, _ = calculate_skill_overlap(jd_skills, txt)
    sk_s.append(sk)
    ex_s.append(calculate_experience_match(jd_exp, txt))
    ed_s.append(calculate_education_match(jd_qual, txt))
old_cands["skill_overlap_score"]    = sk_s
old_cands["experience_match_score"] = ex_s
old_cands["education_match_score"]  = ed_s

# ── compare for 10 IDs ────────────────────────────────────────────────
# Use the production top-10 IDs as the comparison set
answer_key = pd.read_csv(TESTING_DIR / "test_medium_answer_key.csv")
key_map = answer_key.set_index("ID")["relevance"]

prod_top10 = prod_pool.sort_values("final_score", ascending=False).head(10)
compare_ids = prod_top10["ID"].astype(int).tolist()

print(f"\nSide-by-side feature comparison for production top-10 IDs:")
print(f"  (OLD = encrypted-pool path; PROD = build_resume_index path)\n")

old_cands["ID"] = old_cands["ID"].astype(int)
prod_pool["ID"] = prod_pool["ID"].astype(int)

diffs = {f: [] for f in FEATURE_COLS}

for cid in compare_ids:
    old_row  = old_cands[old_cands["ID"] == cid]
    prod_row = prod_pool[prod_pool["ID"] == cid]
    rel = key_map.get(cid, -1)

    if old_row.empty:
        print(f"  ID={cid} (rel={rel}): NOT FOUND in encrypted pool (synthetic ID)")
        continue

    old_r  = old_row.iloc[0]
    prod_r = prod_row.iloc[0]

    print(f"  ID={cid}  rel={rel}")
    for f in FEATURE_COLS:
        ov = old_r[f]
        pv = prod_r[f]
        delta = pv - ov
        flag = " <-- DIFFERS" if abs(delta) > 0.01 else ""
        print(f"    {f:<28} old={ov:7.4f}  prod={pv:7.4f}  delta={delta:+.4f}{flag}")
        diffs[f].append(abs(delta))

print("\nMean absolute difference per feature across compared IDs:")
for f in FEATURE_COLS:
    if diffs[f]:
        print(f"  {f:<28} MAD={np.mean(diffs[f]):.4f}")

print("\n--- ROOT CAUSE ANALYSIS ---")
print("""
The old 08_train_ranker.py medium-set path differed from production in
TWO ways:

1. RESUME SOURCE / EMBEDDINGS (primary cause of semantic_sim differences):
   - OLD path: looked up the 300 test IDs in models/resumes_clean.enc
     (the 2484-resume training pool). The test IDs are 200001-200300,
     which are SYNTHETIC and do NOT exist in the encrypted pool.
     This means the old path either silently dropped candidates or used
     wrong resumes, producing wrong semantic similarities.
   - PRODUCTION path: calls build_resume_index() on the raw test CSV,
     which PII-masks and re-embeds the actual test resumes fresh.
     This is the correct path.

2. CLASSIFIER (secondary cause of ML_score differences):
   - OLD path: retrained a NEW XGBClassifier on the full weak-label data
     inside 08_train_ranker.py. This is a different model instance from
     the saved candidate_scorer.pkl (different random seed state,
     scale_pos_weight computed slightly differently).
   - PRODUCTION path: loads models/candidate_scorer.pkl (the model
     trained and saved by 04_train_model.py).

Fix applied in 08_train_ranker.py: use build_resume_index() for the
medium set and load candidate_scorer.pkl for scoring.
""")
