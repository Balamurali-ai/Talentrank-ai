# 🎯 TalentRank AI

Two-stage candidate ranking system using semantic search (FAISS) and ML scoring (XGBoost) to match resumes against job descriptions, built with Python, FastAPI and Next.js.

> **Academic prototype for controlled evaluation.** Scores are ranking signals, not hiring decisions and not validated predictions of job performance.

<!--
Add screenshots to docs/screenshots/ and uncomment:

![Login](docs/screenshots/login.png)
![Results](docs/screenshots/results.png)
![Why panel](docs/screenshots/why.png)
![Candidate map](docs/screenshots/cluster-map.png)
![Model evaluation](docs/screenshots/model-evaluation.png)
-->

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Machine learning components](#machine-learning-components)
- [Results](#results)
- [Technology](#technology)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Setup](#setup)
- [Run the application](#run-the-application)
- [Using the app](#using-the-app)
- [API](#api)
- [Testing](#testing)
- [Security and privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Future work](#future-work)

## What it does

Given a job description, TalentRank AI returns the 10 most suitable candidates from a resume pool, with a score breakdown and a per-candidate explanation. The pool is either a prebuilt default collection or a CSV the recruiter uploads (up to 2,000 resumes per session).

Features:

- **Top-10 ranking** from the default pool or an uploaded CSV.
- **Explanations** for every result: the contribution of each feature (XGBoost TreeSHAP), short reasons, and matched/missing skills.
- **Candidate map** for uploaded pools: KMeans clusters shown as an approximate 2D PCA projection, with cluster keywords and the job description's closest cluster.
- **Model evaluation panel**: model details, grouped cross-validation results and confidence intervals, and caveats, read from `models/ranker_eval.json`.
- **PII masking** of names, emails and phone numbers before resumes are embedded or stored.
- **Security basics**: JWT-protected API, Fernet-encrypted default resume file, rate limiting on `/rank`, and no resume text in API responses.
- Runs on **CPU only**; no GPU is required.

## How it works

```text
Job description -> MiniLM embedding -> FAISS cosine search -> up to 200 resumes
                                                                   |
                                    five ranking features <--------+
                                              |
                                       XGBoost score
                                              |
        final = alpha x XGBoost score + (1 - alpha) x semantic similarity
                                              |
                                           top 10  -> TreeSHAP explanations
```

The five features for each shortlisted candidate (the order is part of the model contract):

1. `semantic_sim`: cosine similarity between job description and resume embeddings.
2. `skill_overlap_score`: rule-based overlap with the requested skills.
3. `experience_match_score`: rule-based experience fit.
4. `education_match_score`: rule-based education fit.
5. `tfidf_sim`: TF-IDF keyword similarity.

The default blend is `alpha = 0.5`; `/rank` accepts an `alpha` query parameter between 0 and 1. Retrieval is capped at 200 candidates, so anyone outside the FAISS shortlist cannot appear in the results. If the feature code or feature order changes, regenerate the training data and model together. `src/features.py` holds the shared feature logic and the canonical job-description text builder used by both the API and the scripts.

## Machine learning components

| Role | Method | Notes |
|---|---|---|
| Text embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Pretrained, frozen, 384-d, normalized. No fine-tuning |
| Retrieval | FAISS `IndexFlatIP` | Exact cosine search |
| Scoring model | XGBoost classifier | Trained on weak labels; selected over Random Forest |
| Keyword similarity | TF-IDF + cosine | scikit-learn |
| PII masking | spaCy `en_core_web_sm` NER + regex | Pretrained |
| Candidate map | KMeans + PCA | Unsupervised; k chosen by silhouette score |
| Explanations | XGBoost TreeSHAP (`pred_contribs`) | Explains the XGBoost component only |
| Skill / experience / education features | Regex, keyword lists, degree hierarchy | Rule-based, not learned |

All ML algorithms come from their libraries unmodified. The project's own work is the feature engineering, the weak-label training set, the two-stage pipeline and blend, the explanation layer, and the evaluation.

**Weak labels.** No real hiring-outcome labels exist, so training labels are derived from the resume dataset's `Category` field. About 216 job descriptions were sampled and mapped to resume categories with a keyword table. A resume is labeled positive when its category matches the mapped category, otherwise negative. This gives 48,276 (job description, resume) rows with a 19.7% positive rate. `Category` is never used as a model feature, which avoids label leakage.

## Results

### Classifier (held-out split, weak labels)

| Model | F1 | ROC-AUC |
|---|---|---|
| XGBoost (selected) | 0.508 | 0.811 |
| Random Forest | 0.477 | 0.790 |

These measure agreement with noisy category labels, not hiring outcomes. `semantic_sim` accounts for about 46% of XGBoost feature importance. Per-category recall is uneven.

### Ranking quality (grouped cross-validation, NDCG@10)

Evaluated on a shortlist-like subset (top 150 candidates per job description by similarity). Differences are paired bootstrap estimates with 95% confidence intervals.

| Method | Same job families (grouped by JD) | Unseen job categories (grouped by category) |
|---|---|---|
| Semantic similarity only | 0.468 | 0.466 |
| Classifier blend (production) | **0.520** | 0.404 |
| XGBRanker blend (experiment) | 0.513 | 0.443 |

| Comparison | Same job families | Unseen categories |
|---|---|---|
| Classifier blend minus semantic only | +0.051 [0.030, 0.073], distinguishable | -0.063 [-0.120, -0.023], distinguishable |
| XGBRanker blend minus classifier blend | -0.007 [-0.023, 0.010], not distinguishable | +0.038 [0.006, 0.078], distinguishable |
| XGBRanker blend minus semantic only | +0.045 [0.031, 0.059], distinguishable | -0.026 [-0.047, -0.006], distinguishable |

What this means:

- On job families it has seen, the ML layer **improves** on semantic search alone.
- On job categories it has not seen, the ML layer **does not generalize**, and semantic search alone does better. The category-grouped evaluation has only 18 categories, so those intervals are wide.
- A learning-to-rank model (XGBRanker) showed **no distinguishable gain** over the classifier on familiar job families, so it is an offline experiment and is not used in production.
- Removing the experience and education features had no distinguishable effect in either evaluation.

### Controlled validation set

A synthetic 300-resume pool with a graded answer key (40 strong, 80 partial, 180 irrelevant fits) and one Data Analyst job description:

| Run | Precision@10 (strong fits) | NDCG@10 |
|---|---|---|
| Via the web UI | 0.90 | 0.96 |
| Via test scripts | 1.00 | 1.00 |
| Semantic similarity only | 0.30 | 0.72 |

This is one synthetic job description, so treat it as an illustration, not an accuracy estimate. The classifier is also sensitive to retraining: a model retrained with the same settings differs from the saved one by about 0.07 on average (correlation 0.93), which is why the UI and script runs differ.

## Technology

- Python, FastAPI, Uvicorn, Pydantic
- sentence-transformers, FAISS (CPU), XGBoost, scikit-learn
- spaCy (`en_core_web_sm`)
- Next.js 16, React 19, Recharts

## Repository layout

```text
src/
  00_check_data.py                Check the expected source data
  01_prepare_and_embed.py         Clean, mask PII, embed, build the FAISS index
  02_feature_engineering.py       Feature engineering demo on one job description
  03_build_training_data.py       Build the weak-label training set
  04_train_model.py               Train and select the classifier
  05_rerank.py                    Evaluate the ranking pipeline
  06_api.py                       FastAPI application
  07_encrypt_resumes.py           Encrypt the cleaned resume file
  08_train_ranker.py              Learning-to-rank experiment and evaluation
  09_closing_checks.py            Classifier parity and closing checks
  features.py                     Shared feature code and job-description text builder
  explain.py                      Candidate explanations (TreeSHAP)
  cluster.py                      Cluster map generation
frontend/                         Next.js application
tests/                            Project validation and diagnostic scripts
TestingData/                      Controlled test resume pools and job descriptions
data/                             Source datasets (not tracked by Git)
models/                           Generated artifacts (not tracked by Git)
requirements_verified.txt         Verified dependency list (Python 3.13)
TALENTRANK_HANDOFF_v2.md          Development history, decisions and caveats
```

## Data

| Dataset | Expected path | Use |
|---|---|---|
| [Resume dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | `data/Resume/Resume.csv` | 2,484 resumes in 24 categories; the default candidate pool |
| [Job description dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) | `data/job_descriptions.csv` | About 1.6 million synthetic postings; sampled to build training data and test job descriptions |

The datasets are not included in this repository. Download them from Kaggle and place them at the paths above.

Data caveats:

- The resume `Category` reflects the industry a resume was filed under, not always the person's job function. For example, a software developer can be filed under Agriculture. The ranking surfaces these cross-category matches, but the labels are noisy.
- The job description dataset is synthetic and not representative of real postings. Its `skills` field has no reliable delimiter.
- Category coverage is imbalanced. BPO has only 22 resumes, so its per-category results are not meaningful.

## Setup

Requirements: Python 3.13 (the dependency list was verified there), Node.js and npm. CPU-only execution is supported.

### 1. Python environment

From the repository root (Windows PowerShell shown; use the equivalent on macOS/Linux):

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_verified.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_sm
```

The extra index is needed because the pinned CPU build of PyTorch is not on PyPI. If PowerShell blocks activation, call `venv\Scripts\python.exe` directly.

### 2. Secrets

Create a `.env` file in the repository root. **Never commit it.**

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

```dotenv
SECRET_KEY=<generated JWT signing secret>
ENCRYPTION_KEY=<generated Fernet key>
ADMIN_USERNAME=recruiter
ADMIN_PASSWORD=<a strong password you choose>
```

- `SECRET_KEY` and `ENCRYPTION_KEY` are required. The server refuses to start without `SECRET_KEY`.
- Keep the Fernet key safe: the encrypted resume file cannot be decrypted without it.
- Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` explicitly. Do not rely on any built-in defaults.

### 3. Generate the model artifacts

Run these from the repository root, in order. Step 01 downloads the MiniLM model on first use, so it needs internet access once.

```powershell
python src/00_check_data.py
python src/01_prepare_and_embed.py
python src/02_feature_engineering.py
python src/03_build_training_data.py
python src/04_train_model.py
python src/05_rerank.py
python src/07_encrypt_resumes.py
```

Notes:

- On CPU this takes a while (PII masking runs at roughly 9 resumes per second).
- `05_rerank.py` reads `models/resumes_clean.csv`, which step 07 removes after encrypting. Run step 05 before step 07.
- The API runs in offline mode for Hugging Face (`HF_HUB_OFFLINE=1`), so the model must already be cached from step 01.
- Optional: `08_train_ranker.py` and `09_closing_checks.py` produce `models/ranker_eval.json` for the Model evaluation panel. The API works without them; only `/model-info` needs that file.

Generated artifacts:

| File | Purpose |
|---|---|
| `models/resumes_clean.enc` | Fernet-encrypted, PII-masked default resumes |
| `models/resume_embeddings.npy`, `resume_ids.npy` | Normalized vectors and their IDs |
| `models/resume_index.faiss` | FAISS cosine-search index |
| `models/weak_label_training_data.csv` | Feature rows and proxy labels |
| `models/candidate_scorer.pkl` | Selected classifier used by `/rank` |
| `models/model_metadata.json` | Feature names, selected model, metrics |
| `models/ranker_eval.json` | Evaluation results for `/model-info` (optional) |

## Run the application

Backend, from the repository root:

```powershell
venv\Scripts\python.exe -m uvicorn 06_api:app --app-dir src --host 127.0.0.1 --port 8000
```

Startup decrypts the default pool into memory and loads the models; allow 10 to 40 seconds and wait until `/health` reports the models as loaded. Interactive API docs are at http://localhost:8000/docs.

Frontend, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 and sign in with the credentials from `.env`. The API allows browser requests from `http://localhost:3000`.

## Using the app

1. **Sign in.** The access token is kept in memory only (not in local storage) and expires after 60 minutes.
2. **Choose a pool.** Use the default pool or upload a CSV with `ID` and `Resume_str` columns (`Category` is optional, maximum 2,000 rows). Uploads are indexed in the background with a progress indicator. Uploaded sessions live in server memory and expire after two hours.
3. **Describe the role.** Enter job title, skills, experience, qualifications, description and responsibilities. **Use comma-separated skills**, for example `SQL, Python, Tableau`; space-separated lists can mis-parse acronyms.
4. **Review the top 10.** Each row shows semantic similarity, ML score and final score. Expand a row to see feature contributions, reasons, and matched and missing skills.
5. **Explore the pool.** For uploaded pools of 30 or more resumes, the candidate map shows clusters, their keywords, the top-10 candidates and the closest cluster. The 2D view is an approximate projection.
6. **Open Model evaluation** to see model details, the cross-validation comparison, confidence intervals and caveats.

Error handling: an expired token (401) returns you to sign-in, an expired session (404) prompts a re-upload, and rate limiting (429) asks you to wait.

## API

The live schema at `/docs` is authoritative.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | No | Service and model readiness |
| `POST /token` | No | Issue a JWT (form fields `username`, `password`) |
| `POST /upload-resumes` | Bearer | Upload a CSV and start an in-memory indexing session |
| `GET /upload-status/{session_id}` | Bearer | Indexing progress |
| `POST /rank?alpha=0.5` | Bearer | Rank the default pool or a session |
| `GET /cluster-map/{session_id}` | Bearer | Cluster map for a completed session (30+ resumes) |
| `GET /model-info` | Bearer | Model metadata and evaluation results |

`/rank` is limited to 10 requests per minute per client IP. Its JSON body:

| Field | Required | Notes |
|---|---|---|
| `jd_text` | Yes | Job description text |
| `job_title`, `role`, `skills`, `experience`, `qualifications`, `responsibilities` | No | Combined with `jd_text` by the canonical text builder |
| `session_id` | No | Completed upload session; omit to search the default pool |

Example:

```bash
curl -X POST http://127.0.0.1:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=recruiter&password=YOUR_PASSWORD"

curl -X POST "http://127.0.0.1:8000/rank?alpha=0.5" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Backend Software Engineer",
    "skills": "Python, SQL, AWS, Docker, REST APIs",
    "experience": "3 to 10 Years",
    "qualifications": "Bachelor",
    "jd_text": "Design and build scalable REST APIs and services."
  }'
```

The response holds up to 10 records with `id`, `category`, `semantic_sim`, `ML_score`, `final_score` and an explanation. It never contains resume text.

## Testing

`TestingData/` holds controlled test files:

| File | Purpose |
|---|---|
| `test_candidates_1000_easy.csv` with `test_jd_single.csv` | 150 IT resumes among 850 unrelated ones; a scalability and smoke test, intentionally easy |
| `test_candidates_300_medium.csv` with `test_jd_medium.csv` | 300 resumes with strong, partial and irrelevant fits and look-alike decoys, plus a Data Analyst job description |
| `test_medium_answer_key.csv` | Graded relevance (0, 1, 2) for scoring only. Never upload it as candidates |

Scripts in `tests/` are project-specific checks (cluster map, explanations, model info, evaluation diagnostics). Most start the API as a subprocess, so run them from the repository root with the virtual environment's Python. A regression benchmark is also used throughout development: the default pool with the Web Developer benchmark job description should return the same top candidate and score as recorded in `TALENTRANK_HANDOFF_v2.md`. Exact values depend on the artifacts you generated locally.

```powershell
venv\Scripts\python.exe tests\test_9_unit.py
venv\Scripts\python.exe tests\test_8b_cluster.py
venv\Scripts\python.exe tests\test_9_explain.py
venv\Scripts\python.exe tests\test_10_model_info.py
```

`/rank` allows 10 requests per minute, so fast test runs can receive HTTP 429; wait a minute and retry. Synthetic test sets are not evidence of real-world accuracy.

## Security and privacy

- spaCy and regex rules mask detected names, emails and phone numbers before resumes are embedded or stored.
- The default cleaned resume file is Fernet-encrypted at rest and decrypted in memory at startup. Plaintext is removed by `07_encrypt_resumes.py`.
- Uploaded resumes stay in server memory only, are not written to disk by the upload flow, and are lost on restart or after two hours. They are not encrypted in memory.
- JWT authentication protects ranking, upload, status, cluster map and model info. `/rank` is rate-limited.
- API responses contain IDs, categories, scores and explanations, never resume text.
- `.env`, `data/`, `models/`, `logs/` and virtual environments are excluded by `.gitignore`. Keep secrets, source resumes and model artifacts out of Git.
- This is an academic prototype, not a production hiring system. Masking is best-effort and can miss identifying details. Review the limitations before using real applicant data.

## Limitations

- **No real hiring labels.** Training uses weak labels from the dataset's `Category` field, so metrics measure agreement with a noisy proxy.
- **The ML layer does not generalize to unseen job categories.** It beats semantic search alone on familiar job families and loses to it on categories it has not seen (the confidence intervals exclude zero in both cases).
- **Scores vary between retrains.** A retrained classifier differs from the saved one by about 0.07 on average, so exact scores and small ranking differences are not stable.
- **ML scores are not calibrated probabilities**, they tie often, and semantic similarity breaks the ties.
- **Explanations reflect what the model learned from weak labels.** Individual feature effects can look counterintuitive; for example, a feature can lower the score even when the candidate meets the requirement.
- **Category labels are noisy** (industry, not job function), and coverage is imbalanced. BPO is especially thin.
- **The job description dataset is synthetic.** The controlled test sets are synthetic too, and are not evidence of real-world performance.
- **Skill, experience and education parsing is heuristic.** Space-separated skill lists and acronyms can be mis-parsed; experience is taken from phrases like "N years" and education uses a simple degree hierarchy.
- **Retrieval is capped at 200 candidates.** Anyone outside the FAISS shortlist cannot appear.
- **The candidate map is a 2D PCA projection** that keeps only part of the variance, so point positions are approximate.
- **No fairness or bias audit** has been done. Human review is necessary.

## Future work

- Collect recruiter-labeled relevance judgments and evaluate on a larger independent set.
- Improve skill extraction for acronyms, varied delimiters and equivalent technologies.
- Retrain with monotonic constraints to make feature effects and explanations more intuitive.
- Study why the ML layer fails on unseen categories, for example with category-aware training and negative sampling.
- Add similar-candidate search.
- For any real deployment: identity management, data retention controls, audit logging, a security review and a bias and fairness assessment.

## Acknowledgments

Resume data from the Kaggle [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and job descriptions from the Kaggle [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset). Models and libraries: sentence-transformers (`all-MiniLM-L6-v2`), FAISS, XGBoost, scikit-learn, spaCy, FastAPI and Next.js.