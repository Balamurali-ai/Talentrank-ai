"""
src/06_api.py
TalentRank AI - Step 6: FastAPI endpoint wrapping the ranking pipeline

Setup instructions:
1. Install requirements: pip install fastapi uvicorn "python-jose[cryptography]" "passlib[bcrypt]" python-multipart python-dotenv bcrypt
2. Create a .env file with SECRET_KEY and ADMIN_PASSWORD
3. Run server: uvicorn src.06_api:app --reload

Example calls:
# 1. Login
curl -X POST http://127.0.0.1:8000/token -d "username=recruiter&password=securepassword123"

# 2. Rank (Replace <TOKEN> with the token from step 1)
curl -X POST http://127.0.0.1:8000/rank?alpha=0.5 \
     -H "Authorization: Bearer <TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"jd_text": "Looking for a Web Developer with frontend and backend experience. Required skills: HTML, CSS, JavaScript, Node.js.", "job_title": "Web Developer", "experience": "3 years", "qualifications": "Bachelors"}'
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
import uuid
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
import joblib
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
import io
from cryptography.fernet import Fernet
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from features import (
    parse_jd_skills,
    calculate_skill_overlap,
    get_skill_lists,
    calculate_experience_match,
    calculate_education_match,
    build_jd_text,
    build_resume_index,
)
from cluster import build_cluster_map
from explain import compute_contributions, build_explanation

load_dotenv()

# --- Config ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Generate one and add it to .env before starting the server."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "recruiter")
# Simplification for academic project: Single recruiter user from env.
# In production, this would be a database lookup.
# For simplicity, we read the plaintext password from env and hash it to simulate a DB.
ADMIN_PASSWORD_PLAIN = os.getenv("ADMIN_PASSWORD", "securepassword123")

ADMIN_PASSWORD_HASH = bcrypt.hashpw(ADMIN_PASSWORD_PLAIN.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

# --- Session Store ---
# { session_id: { "status": str, "processed": int, "total": int,
#                 "faiss_index": Index|None, "resumes_df": df|None,
#                 "embeddings": ndarray|None, "created_at": datetime } }
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _evict_old_sessions():
    """Remove sessions older than SESSION_TTL_SECONDS. Called on each upload/status request."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=SESSION_TTL_SECONDS)
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if s["created_at"] < cutoff]
        for sid in expired:
            del _sessions[sid]


# --- Global State ---
class AppState:
    df_resumes: pd.DataFrame = None
    embeddings: np.ndarray = None
    faiss_index: faiss.Index = None
    model = None
    metadata: dict = None
    feature_cols: list = None
    st_model: SentenceTransformer = None

state = AppState()

# --- Pydantic Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UploadResponse(BaseModel):
    session_id: str
    status: str
    total_resumes: int


class UploadStatusResponse(BaseModel):
    status: str
    processed: int
    total: int


class RankRequest(BaseModel):
    jd_text: str
    job_title: Optional[str] = ""
    role: Optional[str] = ""
    skills: Optional[str] = ""
    experience: Optional[str] = ""
    qualifications: Optional[str] = ""
    responsibilities: Optional[str] = ""
    session_id: Optional[str] = None

class FeatureExplanation(BaseModel):
    name: str
    value: float
    contribution: float
    share: float

class CandidateExplanation(BaseModel):
    bias: float
    features: List[FeatureExplanation]
    reasons: List[str]
    facts: List[str]
    matched_skills: List[str]
    missing_skills: List[str]

class CandidateResult(BaseModel):
    id: str
    category: str
    semantic_sim: float
    ML_score: float
    final_score: float
    explanation: Optional[CandidateExplanation] = None

class RankResponse(BaseModel):
    results: List[CandidateResult]
    explanation_note: Optional[str] = None

# --- Auth Utilities ---
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    if username != ADMIN_USERNAME:
        raise credentials_exception
    return username

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and FAISS index...")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    
    enc_csv_path = models_dir / "resumes_clean.enc"
    embeddings_npy_path = models_dir / "resume_embeddings.npy"
    faiss_index_path = models_dir / "resume_index.faiss"
    
    # Decrypt data into memory
    fernet = Fernet(os.getenv("ENCRYPTION_KEY").encode())
    with open(enc_csv_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    state.df_resumes = pd.read_csv(io.BytesIO(decrypted_data))
    
    state.embeddings = np.load(embeddings_npy_path)
    state.faiss_index = faiss.read_index(str(faiss_index_path))
    
    scorer_path = models_dir / "candidate_scorer.pkl"
    metadata_path = models_dir / "model_metadata.json"
    
    state.model = joblib.load(scorer_path)
    with open(metadata_path, 'r') as f:
        state.metadata = json.load(f)
        
    state.feature_cols = state.metadata['features']
    state.st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print("Models loaded successfully.")
    
    yield
    
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Endpoints ---
@app.get("/health")
def health_check():
    models_loaded = state.faiss_index is not None and state.model is not None
    return {
        "status": "ok",
        "models_loaded": models_loaded
    }

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME or not verify_password(form_data.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload-resumes", response_model=UploadResponse)
async def upload_resumes(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    _evict_old_sessions()

    contents = await file.read()
    try:
        import io as _io
        df = pd.read_csv(_io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse uploaded file as CSV.")

    missing = [c for c in ("ID", "Resume_str") if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {missing}. Required: ID, Resume_str. Category is optional.",
        )

    if len(df) > 2000:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV has {len(df):,} rows. Upload is capped at 2,000 rows for this CPU-only demo "
                "(spaCy + MiniLM embedding on larger sets would exceed acceptable response time)."
            ),
        )

    session_id = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[session_id] = {
            "status":      "processing",
            "processed":   0,
            "total":       len(df),
            "faiss_index": None,
            "resumes_df":  None,
            "embeddings":  None,
            "created_at":  datetime.now(timezone.utc),
        }

    background_tasks.add_task(_process_upload, session_id, df)
    return UploadResponse(session_id=session_id, status="processing", total_resumes=len(df))


def _process_upload(session_id: str, df: pd.DataFrame):
    """Background task: PII mask + embed + FAISS build for an uploaded CSV."""
    try:
        import spacy as _spacy
        nlp = _spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
        resumes_df, embeddings, faiss_index = build_resume_index(df, nlp, state.st_model)
        with _sessions_lock:
            _sessions[session_id].update({
                "status":      "done",
                "processed":   len(resumes_df),
                "faiss_index": faiss_index,
                "resumes_df":  resumes_df,
                "embeddings":  embeddings,
            })
    except Exception as exc:
        with _sessions_lock:
            _sessions[session_id].update({"status": "error", "error": str(exc)})


@app.get("/upload-status/{session_id}", response_model=UploadStatusResponse)
async def upload_status(session_id: str, current_user: str = Depends(get_current_user)):
    _evict_old_sessions()
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return UploadStatusResponse(
        status=session["status"],
        processed=session["processed"],
        total=session["total"],
    )


@app.post("/rank", response_model=RankResponse)
@limiter.limit("10/minute")
async def rank_candidates(request: Request, payload: RankRequest, alpha: float = 0.5, current_user: str = Depends(get_current_user)):
    # Resolve which index/pool to use
    if payload.session_id:
        with _sessions_lock:
            session = _sessions.get(payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{payload.session_id}' not found.")
        if session["status"] == "processing":
            raise HTTPException(status_code=400, detail="Session is still processing. Poll /upload-status first.")
        if session["status"] == "error":
            raise HTTPException(status_code=400, detail=f"Session processing failed: {session.get('error', 'unknown error')}")
        active_index = session["faiss_index"]
        active_df    = session["resumes_df"]
    else:
        active_index = state.faiss_index
        active_df    = state.df_resumes

    try:
        # Build full JD text using canonical logic
        jd_full_text = build_jd_text(
            jd_title=payload.job_title,
            jd_role=payload.role,
            jd_skills_raw=payload.skills,
            jd_exp=payload.experience,
            jd_qual=payload.qualifications,
            jd_desc=payload.jd_text,
            jd_responsibilities=payload.responsibilities
        )

        # Parse skills from explicit field or fallback to full text
        skills_text = payload.skills if payload.skills else jd_full_text
        parsed_skills = parse_jd_skills(skills_text)

        # Embed
        jd_embedding = state.st_model.encode([jd_full_text], normalize_embeddings=True).astype('float32')

        # Save JD embedding on session for /cluster-map (done before FAISS so it's
        # always stored even if ranking fails partway through)
        if payload.session_id:
            with _sessions_lock:
                if payload.session_id in _sessions:
                    _sessions[payload.session_id]["last_jd_embedding"] = jd_embedding

        # FAISS — cap top_k to pool size so small uploads don't error
        top_k = min(200, len(active_df))
        faiss_scores, faiss_indices = active_index.search(jd_embedding, top_k)
        shortlist_indices = faiss_indices[0]
        semantic_sims = faiss_scores[0]

        pool_df = active_df.iloc[shortlist_indices].copy().reset_index(drop=True)
        pool_df['semantic_sim'] = semantic_sims
        
        # TF-IDF
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
        
        for _, p_row in pool_df.iterrows():
            resume_text = str(p_row["Resume_clean"])
            
            sk_score, _ = calculate_skill_overlap(parsed_skills, resume_text)
            skill_overlap_scores.append(sk_score)
            
            exp_score = calculate_experience_match(payload.experience or "", resume_text)
            exp_match_scores.append(exp_score)
            
            edu_score = calculate_education_match(payload.qualifications or "", resume_text)
            edu_match_scores.append(edu_score)
            
        pool_df["skill_overlap_score"] = skill_overlap_scores
        pool_df["experience_match_score"] = exp_match_scores
        pool_df["education_match_score"] = edu_match_scores
        
        X = pool_df[state.feature_cols]
        probs = state.model.predict_proba(X)[:, 1]
        pool_df['ML_score'] = probs
        
        pool_df['final_score'] = alpha * pool_df['ML_score'] + (1 - alpha) * pool_df['semantic_sim']
        
        top_10 = pool_df.sort_values(by='final_score', ascending=False).head(10).copy().reset_index(drop=True)

        # Compute TreeSHAP explanations only for top-10 rows
        if len(top_10) > 0:
            top_10_X = top_10[state.feature_cols]
            top_10_contribs, bias = compute_contributions(state.model, top_10_X)
            if top_10_contribs.ndim == 1:
                top_10_contribs = top_10_contribs.reshape(1, -1)
        else:
            top_10_contribs, bias = np.empty((0, len(state.feature_cols))), 0.0

        results = []
        for i, (_, row) in enumerate(top_10.iterrows()):
            resume_text = str(row["Resume_clean"])
            matched, missing = get_skill_lists(parsed_skills, resume_text)

            row_feat_vals = {col: float(row[col]) for col in state.feature_cols}
            explanation_data = build_explanation(
                row_values=row_feat_vals,
                contribs=top_10_contribs[i],
                bias=bias,
                matched_skills=matched,
                missing_skills=missing,
            )

            results.append(CandidateResult(
                id=str(row['ID']),
                category=str(row['Category']),
                semantic_sim=float(row['semantic_sim']),
                ML_score=float(row['ML_score']),
                final_score=float(row['final_score']),
                explanation=explanation_data,
            ))

        # Save top-10 on session for /cluster-map majority vote
        if payload.session_id:
            with _sessions_lock:
                if payload.session_id in _sessions:
                    _sessions[payload.session_id]["last_top10"] = [
                        {"id": r.id, "final_score": r.final_score} for r in results
                    ]

        explanation_note = (
            f"Bars show the ML model's influence, learned from weak category labels, "
            f"so individual features can look counterintuitive. "
            f"Final score = {round(alpha, 4)} x ML score + {round(1 - alpha, 4)} x semantic similarity."
        )

        return RankResponse(results=results, explanation_note=explanation_note)
    except Exception as e:
        print(f"Error during ranking: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during ranking computation.")


@app.get("/model-info")
async def model_info(current_user: str = Depends(get_current_user)):
    project_root = Path(__file__).resolve().parent.parent
    metadata_path = project_root / "models" / "model_metadata.json"
    ranker_eval_path = project_root / "models" / "ranker_eval.json"

    if not ranker_eval_path.exists():
        raise HTTPException(
            status_code=404,
            detail="ranker_eval.json not found. Run src/08_train_ranker.py and src/09_closing_checks.py first."
        )

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    with open(ranker_eval_path, "r") as f:
        ranker_eval = json.load(f)

    return {
        "algorithm": meta.get("chosen_algorithm"),
        "f1": meta["metrics"]["f1"],
        "roc_auc": meta["metrics"]["roc_auc"],
        "features": meta.get("features"),
        "ranker_eval": ranker_eval,
    }


@app.get("/cluster-map/{session_id}")
async def cluster_map(session_id: str, current_user: str = Depends(get_current_user)):
    _evict_old_sessions()
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    if session["status"] == "processing":
        raise HTTPException(status_code=400, detail="Session is still processing. Poll /upload-status first.")
    if session["status"] == "error":
        raise HTTPException(status_code=400, detail=f"Session processing failed: {session.get('error', 'unknown error')}")

    jd_emb = session.get("last_jd_embedding")  # None if /rank not called yet
    try:
        return build_cluster_map(session, jd_embedding=jd_emb)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

