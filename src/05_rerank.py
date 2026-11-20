import os
import sys
import json
import numpy as np
import pandas as pd
import faiss
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from features import (
    parse_jd_skills,
    calculate_skill_overlap,
    calculate_experience_match,
    calculate_education_match,
    build_jd_text
)

def main():
    print("Loading models and data...")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    clean_csv_path = models_dir / "resumes_clean.csv"
    embeddings_npy_path = models_dir / "resume_embeddings.npy"
    faiss_index_path = models_dir / "resume_index.faiss"
    jd_csv_path = data_dir / "job_descriptions.csv"
    
    df_resumes = pd.read_csv(clean_csv_path)
    embeddings = np.load(embeddings_npy_path)
    faiss_index = faiss.read_index(str(faiss_index_path))
    df_jds = pd.read_csv(jd_csv_path)
    
    # Load model and metadata
    scorer_path = models_dir / "candidate_scorer.pkl"
    metadata_path = models_dir / "model_metadata.json"
    
    model = joblib.load(scorer_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    feature_cols = metadata['features']
    
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Select 3 Sample JDs (IT, Healthcare, Teacher)
    def find_jd(keyword):
        matches = df_jds[df_jds['Job Title'].astype(str).str.contains(keyword, case=False, na=False)]
        return matches.iloc[0] if not matches.empty else df_jds.iloc[0]
        
    jd_it = find_jd("software|developer|engineer")
    jd_health = find_jd("nurse|doctor|physician|healthcare")
    jd_teacher = find_jd("teacher|educator|instructor")
    
    jds_to_test = {
        "Information-Technology": jd_it,
        "Healthcare": jd_health,
        "Teacher": jd_teacher
    }
    
    def process_jd(jd_row):
        jd_title = str(jd_row.get("Job Title", "Unknown Position"))
        jd_role = str(jd_row.get("Role", ""))
        jd_desc = str(jd_row.get("Job Description", ""))
        jd_skills_raw = str(jd_row.get("skills", ""))
        jd_exp = str(jd_row.get("Experience", ""))
        jd_qual = str(jd_row.get("Qualifications", ""))
        jd_responsibilities = str(jd_row.get("Responsibilities", ""))
        
        parsed_skills = parse_jd_skills(jd_skills_raw)
        
        jd_full_text = build_jd_text(jd_title, jd_role, jd_skills_raw, jd_exp, jd_qual, jd_desc, jd_responsibilities)
        
        jd_embedding = st_model.encode([jd_full_text], normalize_embeddings=True).astype('float32')
        top_k = 200
        faiss_scores, faiss_indices = faiss_index.search(jd_embedding, top_k)
        
        shortlist_indices = faiss_indices[0]
        semantic_sims = faiss_scores[0]
        
        pool_df = df_resumes.iloc[shortlist_indices].copy().reset_index(drop=True)
        pool_df['semantic_sim'] = semantic_sims
        
        # Compute TF-IDF
        corpus = [jd_full_text] + pool_df["Resume_clean"].tolist()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            tfidf_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            pool_df['tfidf_sim'] = tfidf_sims
        except ValueError:
            pool_df['tfidf_sim'] = 0.0
            
        skill_overlap_scores = []
        exp_match_scores = []
        edu_match_scores = []
        
        for p_idx, p_row in pool_df.iterrows():
            resume_text = str(p_row["Resume_clean"])
            
            sk_score, _ = calculate_skill_overlap(parsed_skills, resume_text)
            skill_overlap_scores.append(sk_score)
            
            exp_score = calculate_experience_match(jd_exp, resume_text)
            exp_match_scores.append(exp_score)
            
            edu_score = calculate_education_match(jd_qual, resume_text)
            edu_match_scores.append(edu_score)
            
        pool_df["skill_overlap_score"] = skill_overlap_scores
        pool_df["experience_match_score"] = exp_match_scores
        pool_df["education_match_score"] = edu_match_scores
        
        X = pool_df[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        pool_df['ML_score'] = probs
        
        return pool_df, jd_title

    print("\n" + "="*50)
    print(" 3. CONTINUOUS PREDICT_PROBA CHECK")
    print("="*50)
    
    it_df, jd_title_it = process_jd(jds_to_test["Information-Technology"])
    
    ml_scores = it_df['ML_score']
    print(f"ML_score Min: {ml_scores.min():.4f}")
    print(f"ML_score Max: {ml_scores.max():.4f}")
    print(f"Sample values: {[round(x, 4) for x in ml_scores.head(5).tolist()]}")
    if set(ml_scores.unique()).issubset({0.0, 1.0}):
        print("WARNING: ML_scores appear to be hard labels (0/1). Check if model outputs probabilities.")
    else:
        print("CONFIRMED: ML_scores are continuous probabilities.")
        
    print("\n" + "="*50)
    print(" 5. TOP 10 RERANKING ACROSS 3 CATEGORIES (alpha=0.5)")
    print("="*50)
    
    alpha = 0.5
    for cat_name, jd_row in jds_to_test.items():
        if cat_name == "Information-Technology":
            pool_df, jd_title = it_df, jd_title_it
        else:
            pool_df, jd_title = process_jd(jd_row)
            
        pool_df['final_score'] = alpha * pool_df['ML_score'] + (1 - alpha) * pool_df['semantic_sim']
        top_10 = pool_df.sort_values(by='final_score', ascending=False).head(10)
        
        print(f"\nJD Category: {cat_name}")
        print(f"JD Title: {jd_title}")
        print("-" * 75)
        print(f"{'ID':<15} | {'Category':<25} | {'Sem Sim':<7} | {'ML Score':<8} | {'Final Score':<11}")
        print("-" * 75)
        for _, row in top_10.iterrows():
            print(f"{str(row['ID']):<15} | {str(row['Category'])[:25]:<25} | {row['semantic_sim']:.4f}  | {row['ML_score']:.4f}   | {row['final_score']:.4f}")

    print("\n" + "="*50)
    print(" 6. ALPHA SENSITIVITY TABLE FOR IT JD")
    print("="*50)
    
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    results = {}
    for a in alphas:
        it_df[f'score_{a}'] = a * it_df['ML_score'] + (1 - a) * it_df['semantic_sim']
        results[a] = it_df.sort_values(by=f'score_{a}', ascending=False).head(10)['ID'].tolist()
        
    print(f"{'Rank':<5} | {'alpha=0.0 (SemOnly)':<20} | {'alpha=0.3':<20} | {'alpha=0.5':<20} | {'alpha=0.7':<20} | {'alpha=1.0 (MLOnly)':<20}")
    print("-" * 115)
    for i in range(10):
        print(f"{i+1:<5} | {str(results[0.0][i]):<20} | {str(results[0.3][i]):<20} | {str(results[0.5][i]):<20} | {str(results[0.7][i]):<20} | {str(results[1.0][i]):<20}")

if __name__ == "__main__":
    main()
