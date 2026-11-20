"""
features.py
-----------
TalentRank AI - Shared Feature Extraction Module

Contains reusable candidate-JD feature functions for feature engineering and training set creation:
- parse_jd_skills
- calculate_skill_overlap
- parse_experience_years
- extract_candidate_experience_years
- calculate_experience_match
- calculate_education_match
- calculate_tfidf_sim
- build_resume_index
"""

import re
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standardized Degree Level Hierarchy mapping
DEGREE_HIERARCHY = {
    "phd": 4, "doctorate": 4,
    "master": 3, "masters": 3, "ms": 3, "m.s.": 3, "mtech": 3, "m.tech": 3, "mca": 3, "ma": 3, "m.a.": 3, "mba": 3, "m.b.a.": 3,
    "bachelor": 2, "bachelors": 2, "bs": 2, "b.s.": 2, "btech": 2, "b.tech": 2, "ba": 2, "b.a.": 2, "bca": 2, "be": 2, "b.e.": 2,
    "diploma": 1, "associate": 1
}

# Generic single words to never match in isolation as skills
GENERIC_STOPWORDS = {
    "data", "model", "learning", "analysis", "system", "systems", "management", 
    "development", "services", "algorithms", "preprocessing", "evaluation", 
    "engineering", "processing", "applications", "technologies", "tools", "design",
    "support", "solutions", "architecture", "software"
}

# Technical tools and framework words allowed to match as distinct terms
PRIMARY_TOOLS = {
    "python", "r", "sql", "java", "c++", "cpp", "pytorch", "tensorflow", 
    "excel", "tableau", "aws", "linux", "git", "docker", "kubernetes", "jira", 
    "agile", "scrum", "net", "c#", "react", "angular", "node"
}


def parse_jd_skills(jd_skills_raw: str) -> list:
    """Parses raw JD skill string into distinct, meaningful skill phrases."""
    if not isinstance(jd_skills_raw, str) or not jd_skills_raw.strip():
        return []

    raw = jd_skills_raw.strip()
    if any(sep in raw for sep in [',', ';', '/', '|', '\n']):
        raw_skills = [s.strip().lower() for s in re.split(r'[,;/|\n]', raw) if s.strip()]
    else:
        raw_skills = [s.strip().lower() for s in re.findall(r'[A-Z][a-z0-9]*(?:\s+[a-z0-9]+)*', raw) if len(s.strip()) > 1]

    clean_skills = []
    for sk in raw_skills:
        if sk and sk not in GENERIC_STOPWORDS:
            clean_skills.append(sk)
            
    return list(dict.fromkeys(clean_skills))


def _match_skills(jd_skills_list: list, resume_text: str):
    """
    Internal helper returning (total_credit, matched_list, missing_list).
    """
    if not jd_skills_list:
        return 0.0, [], []

    resume_lower = str(resume_text).lower()
    total_credit = 0.0
    matched = []

    for skill in jd_skills_list:
        if skill in resume_lower:
            total_credit += 1.0
            matched.append(skill)
            continue
            
        words = [w for w in re.findall(r'\b[a-z0-9+#\.]+\b', skill.lower()) if len(w) >= 2]
        keywords = [w for w in words if w not in GENERIC_STOPWORDS]
        
        if not keywords:
            keywords = [w for w in words if len(w) >= 3]
            
        if not keywords:
            continue

        matched_kws = []
        for kw in keywords:
            if kw in PRIMARY_TOOLS:
                if re.search(rf'\b{re.escape(kw)}\b', resume_lower):
                    matched_kws.append(kw)
            elif len(kw) >= 3:
                if re.search(rf'\b{re.escape(kw)}\b', resume_lower):
                    matched_kws.append(kw)
        
        if matched_kws:
            credit = len(matched_kws) / len(keywords)
            total_credit += credit
            matched.append(skill)

    unique_matched = list(dict.fromkeys(matched))
    unique_all = list(dict.fromkeys(jd_skills_list))
    missing = [s for s in unique_all if s not in unique_matched]
    return total_credit, unique_matched, missing


def calculate_skill_overlap(jd_skills_list: list, resume_text: str):
    """
    Computes fraction of required JD skills present in candidate resume text.
    Gives partial credit per skill by matching non-generic keywords while preventing
    single generic words like 'data' or 'model' from matching in isolation.
    """
    if not jd_skills_list:
        return 0.5, ""

    total_credit, unique_matched, _ = _match_skills(jd_skills_list, resume_text)
    overlap_score = total_credit / len(set(jd_skills_list))
    matched_str = ", ".join(unique_matched)
    return float(overlap_score), matched_str


def get_skill_lists(jd_skills_list: list, resume_text: str):
    """
    Returns (matched, missing) skill lists using the same matching logic
    as calculate_skill_overlap. Skill names match the input JD skills.
    """
    if not jd_skills_list:
        return [], []

    _, unique_matched, missing = _match_skills(jd_skills_list, resume_text)
    return unique_matched, missing


def parse_experience_years(text: str) -> float:
    """Extracts minimum required experience years from text."""
    if not isinstance(text, str):
        return None
    
    matches = re.findall(r'(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:|\+)?\s*(?:year|yr)s?', text.lower())
    if not matches:
        matches = re.findall(r'(\d+)\s*(?:\+)?\s*year', text.lower())
        if matches:
            return float(matches[0])
        return None

    years = []
    for m in matches:
        if isinstance(m, tuple):
            for num in m:
                if num:
                    years.append(float(num))
        else:
            years.append(float(m))
            
    return min(years) if years else None


def extract_candidate_experience_years(resume_text: str) -> float:
    """Extracts maximum candidate experience years mentioned in resume text."""
    matches = re.findall(r'(\d+)\+?\s*(?:year|yr)s?\s*(?:of\s*)?(?:exp|experience)?', str(resume_text).lower())
    if matches:
        nums = [float(x) for x in matches if float(x) <= 40]
        if nums:
            return max(nums)
    return None


def calculate_experience_match(jd_exp_raw: str, resume_text: str) -> float:
    """Calculates experience match score with neutral fallback."""
    jd_req_years = parse_experience_years(str(jd_exp_raw))
    cand_years = extract_candidate_experience_years(resume_text)
    
    if jd_req_years is None or cand_years is None:
        return 0.5
        
    if cand_years >= jd_req_years:
        return 1.0
    else:
        return max(0.2, cand_years / jd_req_years)


def calculate_education_match(jd_qual: str, resume_text: str) -> float:
    """Evaluates education match based on degree level hierarchy."""
    if not isinstance(jd_qual, str) or not jd_qual.strip():
        return 0.5

    resume_lower = str(resume_text).lower()
    qual_lower = jd_qual.lower().strip()

    req_level = 0
    for deg, lvl in DEGREE_HIERARCHY.items():
        if re.search(rf'\b{re.escape(deg)}\b', qual_lower):
            req_level = max(req_level, lvl)
            
    if req_level == 0:
        req_level = 2

    cand_level = 0
    for deg, lvl in DEGREE_HIERARCHY.items():
        if re.search(rf'\b{re.escape(deg)}\b', resume_lower):
            cand_level = max(cand_level, lvl)

    if cand_level >= req_level:
        return 1.0
    elif cand_level > 0:
        return 0.7
    else:
        return 0.3


def build_resume_index(df: pd.DataFrame, nlp, st_model):
    """
    Shared pipeline: PII masking + MiniLM embedding + FAISS IndexFlatIP build.

    Args:
        df:       DataFrame with at minimum columns [ID, Resume_str].
                  Category is used if present, otherwise filled with 'UNKNOWN'.
        nlp:      Loaded spaCy model (en_core_web_sm).
        st_model: Loaded SentenceTransformer model.

    Returns:
        (resumes_df, embeddings, faiss_index)
        resumes_df has columns [ID, Category, Resume_clean] — no raw text.
    """
    EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_REGEX = re.compile(
        r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    )

    raw_texts = df["Resume_str"].astype(str).tolist()
    masked = []
    for doc in nlp.pipe(raw_texts, batch_size=64):
        text = doc.text
        for ent in sorted(
            [e for e in doc.ents if e.label_ == "PERSON"],
            key=lambda e: e.start_char, reverse=True
        ):
            text = text[:ent.start_char] + "[NAME]" + text[ent.end_char:]
        text = EMAIL_REGEX.sub("[EMAIL]", text)
        text = PHONE_REGEX.sub("[PHONE]", text)
        masked.append(re.sub(r'\s+', ' ', text).strip())

    embeddings = st_model.encode(
        masked, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    ).astype('float32')

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    resumes_df = pd.DataFrame({
        "ID":           df["ID"].astype(str).values,
        "Category":     df["Category"].values if "Category" in df.columns else ["UNKNOWN"] * len(df),
        "Resume_clean": masked,
    })
    return resumes_df, embeddings, index


def calculate_tfidf_sim(jd_full_text: str, resume_texts: list) -> np.ndarray:
    """Computes TF-IDF cosine similarity between JD full text and list of resume texts."""
    corpus = [jd_full_text] + [str(t) for t in resume_texts]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[0:1], matrix[1:])[0]

def build_jd_text(jd_title, jd_role, jd_skills_raw, jd_exp, jd_qual, jd_desc, jd_responsibilities):
    """Canonical implementation for assembling JD features into a single semantic string."""
    jd_title = str(jd_title) if jd_title and str(jd_title) != "nan" else "Unknown Position"
    jd_role = str(jd_role) if jd_role and str(jd_role) != "nan" else ""
    jd_skills_raw = str(jd_skills_raw) if jd_skills_raw and str(jd_skills_raw) != "nan" else ""
    jd_exp = str(jd_exp) if jd_exp and str(jd_exp) != "nan" else ""
    jd_qual = str(jd_qual) if jd_qual and str(jd_qual) != "nan" else ""
    jd_desc = str(jd_desc) if jd_desc and str(jd_desc) != "nan" else ""
    jd_responsibilities = str(jd_responsibilities) if jd_responsibilities and str(jd_responsibilities) != "nan" else ""
    
    parsed_skills = parse_jd_skills(jd_skills_raw)
    jd_full_text = f"{jd_title} {jd_role}. Required Skills: {', '.join(parsed_skills)}. Experience: {jd_exp}. Qualifications: {jd_qual}. Description: {jd_desc} Responsibilities: {jd_responsibilities}"
    return jd_full_text

