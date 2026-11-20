"""
02_feature_engineering.py
-------------------------
TalentRank AI - Candidate Feature Engineering Script (Refined & Loose Skill Matching Version)

This script:
1. Loads Step 2 artifacts from ./models/.
2. Loads a sample recruiter Job Description filtered for "Information Technology" (aligning with candidate pool categories).
3. Performs Stage-1 FAISS vector search to retrieve the top-200 semantic shortlist.
4. Computes 5 distinct candidate features for Stage-2 re-ranking:
   - semantic_sim: FAISS cosine similarity (SentenceTransformer)
   - skill_overlap_score: Loose & partial-credit keyword matching of technical skills
   - experience_match_score: Required vs. candidate experience years (with neutral fallback)
   - education_match_score: Tiered degree hierarchy matching (PhD > Master > Bachelor > Diploma)
   - tfidf_sim: Independent lexical TF-IDF cosine similarity
5. Exports feature matrix to models/candidate_features_sample.csv.
6. Prints detailed diagnostic statistics (min/max/mean/unique counts, value_counts) for skill_overlap_score.
"""

import os
import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# UTF-8 stdout configuration for Windows console compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

# Add parent directory to sys.path for robust relative module imports
sys.path.append(str(Path(__file__).resolve().parent))

from features import (
    DEGREE_HIERARCHY,
    GENERIC_STOPWORDS,
    PRIMARY_TOOLS,
    parse_jd_skills,
    calculate_skill_overlap,
    parse_experience_years,
    extract_candidate_experience_years,
    calculate_experience_match,
    calculate_education_match,
    calculate_tfidf_sim,
)



def main():
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    print("=" * 70)
    print(" STEP 3 (REFINED & LOOSE MATCHING): FEATURE ENGINEERING")
    print("=" * 70)
    
    # 1. Load Step 2 Artifacts
    print("\n1. Loading Step 2 artifacts from ./models/...")
    clean_csv_path = models_dir / "resumes_clean.csv"
    embeddings_npy_path = models_dir / "resume_embeddings.npy"
    ids_npy_path = models_dir / "resume_ids.npy"
    faiss_index_path = models_dir / "resume_index.faiss"
    
    df_resumes = pd.read_csv(clean_csv_path)
    resume_ids = np.load(ids_npy_path, allow_pickle=True)
    embeddings = np.load(embeddings_npy_path)
    faiss_index = faiss.read_index(str(faiss_index_path))
    
    print(f"   Loaded {len(df_resumes):,} clean resumes.")
    
    # Print full list of unique resume categories in dataset
    unique_categories = sorted(df_resumes["Category"].dropna().unique().tolist())
    print(f"\n   --- UNIQUE RESUME CATEGORIES IN RESUME POOL ({len(unique_categories)} total) ---")
    print(f"   {', '.join(unique_categories)}")
    
    # 2. Load Sample Recruiter Job Description matching Information Technology
    JOB_TITLE_FILTER = "Information Technology"
    print(f"\n2. Loading sample recruiter Job Description for JOB_TITLE_FILTER = '{JOB_TITLE_FILTER}'...")
    jd_csv_path = data_dir / "job_descriptions.csv"
    if not jd_csv_path.exists():
        found_jds = list(data_dir.rglob("job_descriptions.csv"))
        if found_jds:
            jd_csv_path = found_jds[0]
        else:
            raise FileNotFoundError(f"Job descriptions dataset not found in {data_dir}")
            
    df_jds = pd.read_csv(jd_csv_path)
    
    # Filter JDs matching Information Technology or IT/Software roles existing in pool
    it_jds = df_jds[
        df_jds["Job Title"].astype(str).str.contains(r"^Software Engineer|^IT Support Specialist|^IT Manager|^Network Security Specialist|^Software Architect|^UI Developer|^IT Specialist|Information Technology", case=False, regex=True, na=False)
    ]
    if len(it_jds) > 0:
        target_jd = it_jds.iloc[0]
    else:
        target_jd = df_jds.iloc[0]
        
    jd_title = str(target_jd.get("Job Title", "Unknown Position"))
    jd_role = str(target_jd.get("Role", ""))
    jd_desc = str(target_jd.get("Job Description", ""))
    jd_skills_raw = str(target_jd.get("skills", ""))
    jd_exp = str(target_jd.get("Experience", ""))
    jd_qual = str(target_jd.get("Qualifications", ""))
    jd_responsibilities = str(target_jd.get("Responsibilities", ""))
    
    parsed_skills = parse_jd_skills(jd_skills_raw)
    jd_full_text = f"{jd_title} {jd_role}. Required Skills: {', '.join(parsed_skills)}. Experience: {jd_exp}. Qualifications: {jd_qual}. Description: {jd_desc} Responsibilities: {jd_responsibilities}"
    
    print(f"   Target Position:        '{jd_title}'")
    print(f"   Target Category Match:  '{JOB_TITLE_FILTER}'")
    print(f"   Required Experience:    '{jd_exp}'")
    print(f"   Required Qualification: '{jd_qual}'")
    print(f"\n   --- PARSED JD SKILLS ({len(parsed_skills)} skills) ---")
    for i, sk in enumerate(parsed_skills, 1):
        print(f"     [{i}] '{sk}'")
        
    # 3. FAISS Top-200 Search
    print("\n3. Embed JD with SentenceTransformer & search FAISS index (Top-200)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    jd_embedding = model.encode([jd_full_text], normalize_embeddings=True).astype('float32')
    
    top_k = 200
    faiss_scores, faiss_indices = faiss_index.search(jd_embedding, top_k)
    shortlist_indices = faiss_indices[0]
    semantic_sims = faiss_scores[0]
    
    # 4. Compute Features
    print(f"\n4. Computing candidate features for all {top_k} shortlisted resumes...")
    shortlisted_df = df_resumes.iloc[shortlist_indices].copy().reset_index(drop=True)
    shortlisted_df["semantic_sim"] = semantic_sims
    
    # TF-IDF Cosine Similarity
    corpus = [jd_full_text] + shortlisted_df["Resume_clean"].tolist()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    skill_overlap_scores = []
    matched_skills_list = []
    exp_match_scores = []
    edu_match_scores = []
    
    for idx, row in shortlisted_df.iterrows():
        resume_text = str(row["Resume_clean"])
        
        # Skill Overlap
        sk_score, sk_matched = calculate_skill_overlap(parsed_skills, resume_text)
        skill_overlap_scores.append(sk_score)
        matched_skills_list.append(sk_matched)
        
        # Experience Match
        exp_score = calculate_experience_match(jd_exp, resume_text)
        exp_match_scores.append(exp_score)
        
        # Education Match
        edu_score = calculate_education_match(jd_qual, resume_text)
        edu_match_scores.append(edu_score)
        
    shortlisted_df["skill_overlap_score"] = skill_overlap_scores
    shortlisted_df["matched_skills"] = matched_skills_list
    shortlisted_df["experience_match_score"] = exp_match_scores
    shortlisted_df["education_match_score"] = edu_match_scores
    shortlisted_df["tfidf_sim"] = tfidf_sims
    
    # Unweighted Average Sanity Score
    shortlisted_df["sanity_check_score"] = shortlisted_df[[
        "semantic_sim", "skill_overlap_score", "experience_match_score",
        "education_match_score", "tfidf_sim"
    ]].mean(axis=1)
    
    # 5. Save Artifact
    output_feature_path = models_dir / "candidate_features_sample.csv"
    save_cols = [
        "ID", "Category", "semantic_sim", "skill_overlap_score", 
        "matched_skills", "experience_match_score", "education_match_score",
        "tfidf_sim", "sanity_check_score"
    ]
    shortlisted_df[save_cols].to_csv(output_feature_path, index=False)
    
    # 6. Detailed Diagnostics & Print Output
    print("\n" + "=" * 70)
    print(" FEATURE DISTRIBUTION DIAGNOSTICS across ALL 200 SHORTLISTED CANDIDATES")
    print("=" * 70)
    
    print("\n[A] SKILL OVERLAP SCORE DISTRIBUTION:")
    sk_desc = shortlisted_df["skill_overlap_score"].describe()
    print(f"    Min:     {sk_desc['min']:.4f}")
    print(f"    Max:     {sk_desc['max']:.4f}")
    print(f"    Mean:    {sk_desc['mean']:.4f}")
    print(f"    Std:     {sk_desc['std']:.4f}")
    print(f"    Unique:  {shortlisted_df['skill_overlap_score'].nunique()}")
    print("\n    Value Counts (Top 15):")
    print(shortlisted_df["skill_overlap_score"].value_counts().head(15).to_string())
    
    print("\n[B] EDUCATION MATCH SCORE DISTRIBUTION:")
    edu_desc = shortlisted_df["education_match_score"].describe()
    print(f"    Min:     {edu_desc['min']:.4f}")
    print(f"    Max:     {edu_desc['max']:.4f}")
    print(f"    Mean:    {edu_desc['mean']:.4f}")
    print(f"    Std:     {edu_desc['std']:.4f}")
    print(f"    Unique:  {shortlisted_df['education_match_score'].nunique()}")
    print("    Value Counts:")
    print(shortlisted_df["education_match_score"].value_counts().sort_index().to_string())

    print("\n" + "=" * 70)
    print(" REFINED TOP-10 CANDIDATE PREVIEW (Sorted by Unweighted Average Sanity Score)")
    print("=" * 70)
    
    top_10 = shortlisted_df.sort_values(by="sanity_check_score", ascending=False).head(10)
    header = f"{'Rank':<5} {'ID':<8} {'Category':<24} {'Semantic':<10} {'Skills':<8} {'Exp':<6} {'Edu':<6} {'TF-IDF':<8} {'Sanity Score':<12}"
    print(header)
    print("-" * len(header))
    
    it_count_top10 = 0
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        cat_str = str(row['Category'])
        if cat_str.upper() == "INFORMATION-TECHNOLOGY":
            it_count_top10 += 1
        print(
            f"{rank:<5} {int(row['ID']):<8} {cat_str[:23]:<24} "
            f"{row['semantic_sim']:.4f}     "
            f"{row['skill_overlap_score']:.4f}   "
            f"{row['experience_match_score']:.2f}   "
            f"{row['education_match_score']:.2f}   "
            f"{row['tfidf_sim']:.4f}   "
            f"{row['sanity_check_score']:.4f}"
        )
    print("=" * 70)
    print(f"SUMMARY: {it_count_top10}/10 candidates in the refined Top-10 are Category == 'INFORMATION-TECHNOLOGY'.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
