"""
03_build_training_data.py
-------------------------
TalentRank AI - Step 4a: Training Data Builder
Constructs a weak-labeled dataset mapping Resumes (with generated features) to JDs, 
creating positive and negative examples based on Resume Category matching JD target keywords.
"""

import os
import sys
import time
import random
import re
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to sys.path for robust relative module imports
sys.path.append(str(Path(__file__).resolve().parent))

from features import (
    parse_jd_skills,
    calculate_skill_overlap,
    parse_experience_years,
    calculate_experience_match,
    calculate_education_match,
    calculate_tfidf_sim,
)

# UTF-8 stdout configuration for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

CATEGORY_KEYWORDS = {
    'ACCOUNTANT': ['accountant', 'bookkeeper', 'audit', 'finance', 'cpa', 'accounting'],
    'ADVOCATE': ['advocate', 'lawyer', 'attorney', 'legal', 'counsel', 'paralegal', 'litigation'],
    'AGRICULTURE': ['agriculture', 'farm', 'agronomist', 'crop', 'soil', 'harvest'],
    'APPAREL': ['apparel', 'fashion', 'clothing', 'textile', 'garment', 'merchandiser'],
    'ARTS': ['art', 'artist', 'painter', 'sculptor', 'curator', 'gallery'],
    'AUTOMOBILE': ['automobile', 'automotive', 'mechanic', 'vehicle', 'car'],
    'AVIATION': ['aviation', 'pilot', 'aircraft', 'aerospace', 'flight'],
    'BANKING': ['banking', 'banker', 'teller', 'loan officer', 'branch manager'],
    'BPO': ['bpo', 'customer service', 'call center', 'support', 'telesales'],
    'BUSINESS-DEVELOPMENT': ['business development', 'sales', 'bd', 'growth', 'partnerships'],
    'CHEF': ['chef', 'cook', 'culinary', 'kitchen', 'restaurant', 'food'],
    'CONSTRUCTION': ['construction', 'builder', 'contractor', 'carpenter', 'mason'],
    'CONSULTANT': ['consultant', 'adviser', 'strategy', 'management consulting'],
    'DESIGNER': ['designer', 'graphic', 'ui', 'ux', 'creative', 'visual'],
    'DIGITAL-MEDIA': ['digital media', 'social media', 'content', 'marketing', 'seo'],
    'ENGINEERING': ['engineer', 'civil', 'mechanical', 'electrical', 'chemical', 'engineering'],
    'FINANCE': ['finance', 'financial', 'analyst', 'investment', 'wealth'],
    'FITNESS': ['fitness', 'trainer', 'gym', 'health club', 'coach', 'instructor'],
    'HEALTHCARE': ['healthcare', 'nurse', 'doctor', 'physician', 'medical', 'clinical', 'hospital'],
    'HR': ['hr', 'human resources', 'recruiter', 'talent', 'personnel'],
    'INFORMATION-TECHNOLOGY': ['software', 'developer', 'it', 'programmer', 'network', 'system', 'data', 'security', 'technology', 'web', 'engineer', 'machine learning'],
    'PUBLIC-RELATIONS': ['public relations', 'pr', 'communications', 'press', 'media relations'],
    'SALES': ['sales', 'account executive', 'representative', 'account manager', 'buyer'],
    'TEACHER': ['teacher', 'educator', 'instructor', 'tutor', 'professor', 'school']
}

def main():
    print("=" * 70)
    print(" STEP 4a: BUILD WEAK-LABELED TRAINING DATA")
    print("=" * 70)
    
    print("\nCATEGORY_KEYWORDS mapping defined:")
    for cat, kws in CATEGORY_KEYWORDS.items():
        print(f"  {cat}: {kws}")
    print()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    # 1. Load data
    clean_csv_path = models_dir / "resumes_clean.csv"
    embeddings_npy_path = models_dir / "resume_embeddings.npy"
    ids_npy_path = models_dir / "resume_ids.npy"
    faiss_index_path = models_dir / "resume_index.faiss"
    jd_csv_path = data_dir / "job_descriptions.csv"
    
    df_resumes = pd.read_csv(clean_csv_path)
    resume_ids = np.load(ids_npy_path, allow_pickle=True)
    embeddings = np.load(embeddings_npy_path)
    faiss_index = faiss.read_index(str(faiss_index_path))
    df_jds = pd.read_csv(jd_csv_path)
    
    # 2. Sample JDs per category
    TARGET_PER_CATEGORY = 12
    sampled_jds = []
    
    for cat in CATEGORY_KEYWORDS.keys():
        kws = CATEGORY_KEYWORDS[cat]
        pattern = '|'.join([r'\b{}\b'.format(re.escape(k)) for k in kws])
        
        matches = df_jds[df_jds['Job Title'].astype(str).str.contains(pattern, case=False, regex=True, na=False)]
        if len(matches) > 0:
            sample_size = min(TARGET_PER_CATEGORY, len(matches))
            sampled = matches.sample(n=sample_size, random_state=42).copy()
            sampled['Mapped_Category'] = cat
            sampled_jds.append(sampled)
            
    if not sampled_jds:
        print("No JDs matched any category keywords!")
        return
        
    df_sampled_jds = pd.concat(sampled_jds, ignore_index=True)
    print(f"Sampled {len(df_sampled_jds)} JDs across categories.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    all_training_rows = []
    
    print("Processing JDs to generate training data...")
    for idx, jd_row in df_sampled_jds.iterrows():
        mapped_cat = jd_row['Mapped_Category']
        
        jd_title = str(jd_row.get("Job Title", "Unknown Position"))
        jd_role = str(jd_row.get("Role", ""))
        jd_desc = str(jd_row.get("Job Description", ""))
        jd_skills_raw = str(jd_row.get("skills", ""))
        jd_exp = str(jd_row.get("Experience", ""))
        jd_qual = str(jd_row.get("Qualifications", ""))
        jd_responsibilities = str(jd_row.get("Responsibilities", ""))
        
        parsed_skills = parse_jd_skills(jd_skills_raw)
        jd_full_text = f"{jd_title} {jd_role}. Required Skills: {', '.join(parsed_skills)}. Experience: {jd_exp}. Qualifications: {jd_qual}. Description: {jd_desc} Responsibilities: {jd_responsibilities}"
        
        # Embed and search
        jd_embedding = model.encode([jd_full_text], normalize_embeddings=True).astype('float32')
        top_k = 150
        faiss_scores, faiss_indices = faiss_index.search(jd_embedding, top_k)
        shortlist_indices = faiss_indices[0]
        semantic_sims = faiss_scores[0]
        
        # Get positive/shortlisted candidates
        shortlisted_df = df_resumes.iloc[shortlist_indices].copy().reset_index(drop=True)
        shortlisted_df['semantic_sim'] = semantic_sims
        shortlisted_df['is_shortlisted'] = True
        
        # Random negative sample (50-100 resumes from OTHER categories)
        neg_candidates = df_resumes[df_resumes['Category'] != mapped_cat]
        if not neg_candidates.empty:
            neg_sample_size = random.randint(50, 100)
            neg_sample = neg_candidates.sample(n=min(neg_sample_size, len(neg_candidates)), random_state=idx).copy()
            
            neg_indices = neg_sample.index.tolist()
            neg_embeddings = embeddings[neg_indices]
            
            # Use dot product instead of cosine_similarity since embeddings are normalized (L2) in step 1/2
            neg_sims = np.dot(neg_embeddings, jd_embedding[0])
            neg_sample['semantic_sim'] = neg_sims
            neg_sample['is_shortlisted'] = False
            
            pool_df = pd.concat([shortlisted_df, neg_sample], ignore_index=True)
        else:
            pool_df = shortlisted_df
        
        # Features computation
        # TF-IDF
        corpus = [jd_full_text] + pool_df["Resume_clean"].tolist()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            tfidf_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            pool_df['tfidf_sim'] = tfidf_sims
        except ValueError: # E.g., if corpus is empty or only stop words
            pool_df['tfidf_sim'] = 0.0
        
        skill_overlap_scores = []
        exp_match_scores = []
        edu_match_scores = []
        labels = []
        
        for p_idx, p_row in pool_df.iterrows():
            resume_text = str(p_row["Resume_clean"])
            resume_cat = p_row["Category"]
            
            # Label
            label = 1 if resume_cat == mapped_cat else 0
            labels.append(label)
            
            sk_score, _ = calculate_skill_overlap(parsed_skills, resume_text)
            skill_overlap_scores.append(sk_score)
            
            exp_score = calculate_experience_match(jd_exp, resume_text)
            exp_match_scores.append(exp_score)
            
            edu_score = calculate_education_match(jd_qual, resume_text)
            edu_match_scores.append(edu_score)
            
        pool_df["skill_overlap_score"] = skill_overlap_scores
        pool_df["experience_match_score"] = exp_match_scores
        pool_df["education_match_score"] = edu_match_scores
        pool_df["label"] = labels
        pool_df["jd_id"] = idx
        pool_df["mapped_category"] = mapped_cat
        
        # Keep columns
        keep_cols = [
            "jd_id", "ID", "Category", "mapped_category",
            "semantic_sim", "skill_overlap_score", "experience_match_score", 
            "education_match_score", "tfidf_sim", "label"
        ]
        
        all_training_rows.append(pool_df[keep_cols])

    if not all_training_rows:
        print("No training rows generated.")
        return
        
    final_train_df = pd.concat(all_training_rows, ignore_index=True)
    
    out_path = models_dir / "weak_label_training_data.csv"
    final_train_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(final_train_df)} training rows to {out_path.name}")
    
    # 6. Print Validation Output
    print("\n" + "="*50)
    print(" VALIDATION OUTPUT ")
    print("="*50)
    print(f"Total Rows: {len(final_train_df)}")
    
    labels_dist = final_train_df['label'].value_counts()
    print("\nOverall Class Balance (1=Positive, 0=Negative):")
    print(labels_dist.to_string())
    
    print("\nClass Balance by Top 5 Mapped Categories (based on sample size):")
    top_cats = final_train_df['mapped_category'].value_counts().head(5).index
    for cat in top_cats:
        subset = final_train_df[final_train_df['mapped_category'] == cat]
        pos = subset['label'].sum()
        neg = len(subset) - pos
        print(f"  {cat}:")
        print(f"    Positives (1): {pos} | Negatives (0): {neg}")
        
    print("\nSummary Stats (Mean +/- Std) of Features split by Label:")
    feature_cols = ["semantic_sim", "skill_overlap_score", "experience_match_score", "education_match_score", "tfidf_sim"]
    
    grouped = final_train_df.groupby('label')[feature_cols].agg(['mean', 'std'])
    
    for f in feature_cols:
        print(f"\n--- {f} ---")
        try:
            mean_0 = grouped.loc[0, (f, 'mean')]
            std_0 = grouped.loc[0, (f, 'std')]
            mean_1 = grouped.loc[1, (f, 'mean')]
            std_1 = grouped.loc[1, (f, 'std')]
            print(f"  Label 0 (Negative): {mean_0:.4f} +/- {std_0:.4f}")
            print(f"  Label 1 (Positive): {mean_1:.4f} +/- {std_1:.4f}")
        except KeyError:
            print("  Missing labels to compute statistics.")
        
    print("\nSanity Check: Positives (Label 1) should generally have higher feature values than Negatives (Label 0).")
    
if __name__ == "__main__":
    main()
