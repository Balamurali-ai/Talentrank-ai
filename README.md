# TalentRank AI

TalentRank AI is a two-stage resume ranking system that matches recruiter-provided job descriptions with a resume pool. It uses semantic search with FAISS to shortlist candidates, then combines an XGBoost suitability score with semantic similarity to rank the top 10. The project includes a FastAPI backend and a Next.js recruiter interface.

> Academic prototype for controlled evaluation. Scores are ranking signals, not hiring decisions or validated predictions of job performance.

## Contents

- [How it works](#how-it-works)
- [Features](#features)
- [Technology](#technology)
- [Data](#data)
- [Model and training](#model-and-training)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the application](#run-the-application)
- [Use the API](#use-the-api)
- [Regenerate the model artifacts](#regenerate-the-model-artifacts)
- [Evaluation and test data](#evaluation-and-test-data)
- [Frontend flow](#frontend-flow)
- [Security and privacy](#security-and-privacy)
- [Limitations](#limitations)
- [Future work](#future-work)

## How it works

```text
Job description ──> MiniLM embedding ──> FAISS cosine search ──> up to 200 resumes
                                                                  │
                                  five ranking features <─────────┘
                                          │
                                   XGBoost score
                                          │
                final = alpha × XGBoost + (1 − alpha) × semantic similarity
                                          │
                                      top 10
```

The embedding model is `all-MiniLM-L6-v2` (384-dimensional, CPU inference). Resume vectors are normalized and searched with a FAISS `IndexFlatIP`, which gives cosine similarity. The XGBoost classifier scores the shortlist using these features:

1. `semantic_sim` — cosine similarity between the job description and resume.
2. `skill_overlap_score` — rule-based overlap with requested skills.
3. `experience_match_score` — rule-based experience fit.
4. `education_match_score` — rule-based education fit.
5. `tfidf_sim` — TF-IDF text similarity.

The default blend is `alpha=0.5`. The API accepts an `alpha` query parameter to adjust the balance. The response includes the component scores, final score, and a per-candidate explanation based on XGBoost TreeSHAP contributions. Explanations describe the classifier's learned feature effects; they do not establish a candidate's actual suitability.

## Data

The development pipeline uses two source datasets:

| Dataset | Expected path | Role in the project |
|---|---|---|
| Resume dataset (`snehaanbhawal/resume-dataset`) | `data/Resume/Resume.csv` | Resume text, IDs, and category labels; source data has 2,484 rows, four columns, and 24 categories. |
| Job description dataset (`ravindrasinghrana/job-description-dataset`) | `data/job_descriptions.csv` | Samples job descriptions for weak-label training and supplies skill vocabulary. The source has about 1.6 million rows and is not embedded wholesale for serving. |

The job-description collection is synthetic and intended for educational/research use. Its fields and formatting should not be treated as representative of all real job postings. The datasets are not included in this repository; obtain them from their original sources or from the project owner and place them at the paths above.

The resume `Category` field is used to construct weak training labels, but is deliberately excluded from model features to avoid label leakage. It represents the resume dataset's category or industry and does not always describe the person's job function. Some genuine skill matches therefore cross category boundaries. Category coverage is imbalanced, with BPO especially underrepresented.

## Model and training

The system combines pretrained language representations, exact vector search, engineered signals, and a supervised classifier:

1. spaCy NER and regular expressions mask detected person names, email addresses, and phone numbers. Whitespace is normalized.
2. `all-MiniLM-L6-v2` creates normalized 384-dimensional resume embeddings on CPU. The weights are frozen; this project does not fine-tune the transformer.
3. FAISS `IndexFlatIP` performs exact cosine search over the normalized vectors. Up to 200 candidates are passed to feature calculation and reranking.
4. Rule-based skill, experience, and education features are calculated alongside semantic and TF-IDF similarity.
5. Random Forest and XGBoost classifiers are compared using a held-out split of weakly labeled examples. The selected classifier is stored in `candidate_scorer.pkl`.
6. `predict_proba` supplies the classifier score, which is blended with semantic similarity and sorted to return up to 10 candidates.
7. TreeSHAP contributions are calculated for returned candidates to give feature-level explanations. The contributions explain the XGBoost component, not the semantic part of the final blend.

Weak labels are created by mapping sampled job titles/roles to resume categories. A resume receives a positive label when its category matches the mapped category and a negative label otherwise. The current training data contains about 48,000 resume/JD examples with roughly a 20% positive rate. On the documented held-out weak-label split, XGBoost scored F1 0.508 and ROC-AUC 0.811; the Random Forest comparison scored F1 0.477 and ROC-AUC 0.790. These figures measure agreement with noisy proxy labels, not hiring outcomes. Performance varies by category.

The five feature names and their order are part of the model contract. If the feature implementation or order changes, regenerate training data and model artifacts together. `src/features.py` contains shared feature logic and the canonical job-description text builder used by both the API and ranking scripts.

## Features

- Rank the top 10 candidates from the built-in resume collection or an uploaded CSV.
- Upload and index up to 2,000 resumes per session. Upload processing runs in the background and reports status.
- Explore uploaded resumes with a cluster map and cluster keywords.
- Inspect model metadata and evaluation results when the corresponding artifact exists.
- Mask detected names, email addresses, and phone numbers before embedding resumes.
- Return candidate IDs and scores without returning resume text.
- JWT-protected API, Fernet encryption for the built-in cleaned resume file, and a rate limit on ranking requests.
- Responsive recruiter interface for login, CSV upload, job description entry, ranked results, explanations, and candidate map.

## Technology

- Python, FastAPI, Uvicorn, Pydantic
- Sentence Transformers (`all-MiniLM-L6-v2`), FAISS CPU, XGBoost, scikit-learn
- spaCy (`en_core_web_sm`) for name detection
- Next.js 16, React 19, Recharts

## Repository layout

```text
src/                 Data preparation, feature engineering, model and API code
  00_check_data.py   Check expected source data
  01_prepare_and_embed.py  Clean, mask PII, embed, and build the FAISS index
  02_feature_engineering.py
  03_build_training_data.py
  04_train_model.py  Train and select the classifier
  05_rerank.py       Evaluate the ranking pipeline
  06_api.py          FastAPI application
  07_encrypt_resumes.py
  08_train_ranker.py
  09_closing_checks.py
  features.py        Shared feature and resume-index helpers
  explain.py         Candidate explanation generation
  cluster.py         Resume cluster map generation
frontend/            Next.js web application
data/                Source datasets (not tracked by Git)
models/              Generated model and index artifacts (not tracked by Git)
TestingData/         Controlled test candidate pools and job descriptions
tests/               Project validation and diagnostic scripts
PROJECT_CONTEXT.md   Earlier project decisions and implementation notes
TALENTRANK_HANDOFF.md  Detailed validation history, current status, and caveats
requirements_verified.txt
```

## Requirements

- Windows, macOS, or Linux; CPU-only execution is supported. A GPU is not required.
- Python with a working `venv`/`pip` setup. The dependency lock list was verified in the project's Python 3.13 environment.
- Node.js and npm to run the frontend.
- The source datasets and generated model artifacts described below.

The `data/` and `models/` directories are intentionally excluded from Git because they contain large datasets and generated or sensitive artifacts. Obtain the project datasets through their original sources or from the project owner. The expected source files are `data/Resume/Resume.csv` and `data/job_descriptions.csv`. The default-pool API requires generated files under `models/`.

## Setup

### 1. Create and activate a Python environment

From the repository root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_verified.txt
```

If PowerShell blocks environment activation, run the environment's Python directly, for example `.venv\Scripts\python.exe`.

### 2. Configure secrets

Create a root `.env` file. Do not commit it. Generate independent values for the JWT signing key and Fernet encryption key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Put the generated values in `.env`:

```dotenv
SECRET_KEY=replace_with_the_generated_jwt_secret
ENCRYPTION_KEY=replace_with_the_generated_fernet_key
ADMIN_USERNAME=recruiter
ADMIN_PASSWORD=replace_with_a_strong_password
```

`SECRET_KEY` and `ENCRYPTION_KEY` are required for the backend. Keep the Fernet key safe: the encrypted resume file cannot be decrypted without it. The backend currently defaults the username and password if their variables are omitted, so set both explicitly before use.

### 3. Generate the default resume index and model

Run the preparation and training scripts from the repository root, in order:

```powershell
python src/00_check_data.py
python src/01_prepare_and_embed.py
python src/02_feature_engineering.py
python src/03_build_training_data.py
python src/04_train_model.py
python src/05_rerank.py
python src/07_encrypt_resumes.py
```

These scripts create the files the backend loads, including `models/resumes_clean.enc`, `models/resume_embeddings.npy`, `models/resume_index.faiss`, `models/candidate_scorer.pkl`, and `models/model_metadata.json`. Step 7 encrypts the cleaned resume CSV and removes its plaintext copy. Do not skip encryption if preparing the default pool for API use.

The job-description dataset is used for training-data sampling and vocabulary work; serving ranks one submitted job description at a time. Training and embedding can take time on CPU.

### Generated artifacts

| Artifact | Purpose |
|---|---|
| `models/resumes_clean.enc` | Fernet-encrypted cleaned default resume data loaded by the API |
| `models/resume_embeddings.npy` | Normalized default-pool vectors |
| `models/resume_ids.npy` | IDs corresponding to the prepared resume rows/vectors |
| `models/resume_index.faiss` | FAISS cosine-search index |
| `models/weak_label_training_data.csv` | Feature rows and proxy labels used for classifier training |
| `models/candidate_scorer.pkl` | Selected fitted classifier used by `/rank` |
| `models/model_metadata.json` | Feature names, selected model, and classifier metrics |
| `models/ranker_eval.json` | Optional evaluation output written by the ranker/closing-check workflows; used by `/model-info` |

Do not put private data or keys into Git. The repository's `.gitignore` excludes `data/`, `models/`, `.env`, Python bytecode, virtual environments, and logs.

## Run the application

Start the backend from the repository root:

```powershell
python -m uvicorn src.06_api:app --host 127.0.0.1 --port 8000
```

Startup loads and decrypts the default resume pool into memory, loads model artifacts, and initializes MiniLM. Wait for `Models loaded successfully.` before sending requests. The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Sign in with the username and password configured in `.env` (the frontend does not supply credentials on its own). The API allows browser requests from `http://localhost:3000`.

The user flow is: sign in, select the default pool or upload a CSV, enter the job description fields, rank candidates, and review scores and explanations. Uploaded CSV files must include `ID` and `Resume_str`; `Category` is optional. Uploaded sessions are held in memory and expire after two hours.

## Frontend flow

The frontend is a Next.js App Router application in `frontend/`:

1. **Sign in:** submits credentials to `/token`; the access token is held in React state rather than browser local storage.
2. **Choose a pool:** use the prepared default pool or upload a CSV with `ID` and `Resume_str`. Uploads are capped at 2,000 rows, indexed asynchronously, and polled through `/upload-status/{session_id}`.
3. **Describe the role:** provide a title, skills, experience, qualifications, description, and responsibilities. Comma-separated skills are recommended.
4. **Review candidates:** view up to 10 ranked candidates and score components. Expand a candidate's explanation to inspect feature values, contribution directions, reason summaries, and matched/missing skills.
5. **Explore the pool:** the uploaded-pool cluster map shows an approximate PCA projection and cluster keywords. The cluster map requires at least 30 resumes and a completed upload session.

The backend CORS configuration allows `http://localhost:3000`. The JWT expires after 60 minutes; sessions last up to two hours. Rate-limit responses (HTTP 429), expired sessions (HTTP 404), and expired tokens (HTTP 401) require the user to retry, re-upload, or sign in again as appropriate.

## Use the API

Interactive request and schema documentation: `GET /docs`.

### Authenticate

`POST /token` accepts OAuth2 form fields:

```bash
curl -X POST http://127.0.0.1:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=recruiter&password=your-configured-password"
```

Use the returned `access_token` as a Bearer token. Tokens expire after 60 minutes.

### Rank candidates

`POST /rank?alpha=0.5` requires a Bearer token and JSON body. `jd_text` is required; other job fields and `session_id` are optional.

```bash
curl -X POST "http://127.0.0.1:8000/rank?alpha=0.5" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Backend Software Engineer",
    "skills": "Python, SQL, AWS, Docker, REST APIs",
    "experience": "3 years",
    "qualifications": "Bachelor degree",
    "jd_text": "Design and build scalable REST APIs and services.",
    "responsibilities": "Build APIs, manage databases, and deploy services."
  }'
```

The response contains up to 10 candidate records with `id`, `category`, `semantic_sim`, `ML_score`, `final_score`, and an explanation. It does not contain resume text. `alpha` is a query parameter between 0 and 1; the default is 0.5.

### Other endpoints

| Endpoint | Authentication | Purpose |
|---|---|---|
| `GET /health` | No | Report service/model readiness |
| `POST /token` | No | Issue a JWT access token |
| `POST /upload-resumes` | Bearer token | Upload CSV and create an in-memory indexing session (maximum 2,000 rows) |
| `GET /upload-status/{session_id}` | Bearer token | Check upload/indexing progress |
| `POST /rank` | Bearer token | Rank the default pool or a supplied upload session |
| `GET /cluster-map/{session_id}` | Bearer token | Get a cluster map for a completed upload session (at least 30 resumes) |
| `GET /model-info` | Bearer token | Read model and ranker evaluation metadata, if generated |

`/rank` is limited to 10 requests per minute per client IP. Uploaded sessions are process-local, are not persisted, and are lost when the server restarts.

## Regenerate the model artifacts

The normal end-to-end artifact generation order is scripts `00` through `07`, as shown in [Setup](#setup). Scripts `08_train_ranker.py` and `09_closing_checks.py` are additional ranking evaluation/closing-check workflows and may require the training artifacts and controlled test files. They write evaluation outputs used by `/model-info`. The API requires the classifier artifacts from step 4; the extra ranker evaluation files are only required for `/model-info`.

## Evaluation and test data

`TestingData/` contains controlled resume pools and sample job descriptions for functional and ranking checks. Some test sets are intentionally easy or synthetic; their scores should not be presented as real-world hiring accuracy. See `TALENTRANK_HANDOFF.md` for the current validation notes, known benchmark caveats, and project development history.

The classifier was trained on weak labels derived from resume `Category` and a keyword mapping from job titles/roles. Held-out classifier metrics therefore measure agreement with that proxy labeling process, not agreement with recruiter decisions or eventual job performance. The project also includes manually constructed controlled evaluation data; use its answer key only for evaluation, never as candidate input.

Documented controlled checks include an easy 1,000-row pool with 150 IT resumes and a backend software engineer job description, and a 300-row medium data-analyst pool with graded relevance labels. A recorded medium-pool evaluation reported precision at 10 of 0.9 for strong matches and NDCG@10 of about 0.96. These are results on constructed controlled sets and should not be presented as real-world performance estimates. See `TALENTRANK_HANDOFF.md` for the exact validation setup, regression benchmark, and interpretation notes; results can depend on the exact test files and generated artifacts present locally.

Scripts under `tests/` are project-specific validation and diagnostic utilities, not a general-purpose public test suite. Some scripts require optional evaluation files or a running local environment. Review a script's inputs before running it.

## API reference

The live OpenAPI schema at `/docs` is authoritative. The request model for `/rank` accepts:

| Field | Type | Notes |
|---|---|---|
| `jd_text` | string | Required job description text; combined with the optional fields using the canonical builder. |
| `job_title` | string | Optional title. |
| `role` | string | Optional role. |
| `skills` | string | Optional requested skills; delimit phrases with commas or semicolons. |
| `experience` | string | Optional experience requirement. |
| `qualifications` | string | Optional education/qualification requirement. |
| `responsibilities` | string | Optional responsibilities. |
| `session_id` | string | Optional completed upload session ID; omit to search the default pool. |

The `alpha` blend weight is a query parameter on `/rank`, not a field in this JSON model. Authenticated endpoints use `Authorization: Bearer <token>`. The upload endpoint accepts a multipart CSV file. Error status codes and response schemas are visible in `/docs`.

## Security and privacy

- spaCy person-entity detection and regex rules mask detected names, emails, and phone numbers before the default resume text is embedded or persisted.
- The default cleaned resume CSV is encrypted at rest with Fernet. The API decrypts it in memory during startup.
- Uploaded resume sessions are held in server memory only. They are not persisted to disk by the upload flow, but they are not encrypted in memory and are lost on restart/expiry.
- Ranking responses return IDs, categories, scores, and explanations; they do not return resume text.
- JWT authentication protects ranking, upload, upload status, model info, and cluster map endpoints. `/rank` is rate-limited.
- Keep `.env`, encryption keys, source resumes, and model artifacts private. `.env`, `data/`, and `models/` are excluded by `.gitignore`.
- This is an academic prototype, not a production hiring system. Review the security and limitations before using it with real applicant data.

## Limitations

- There are no real hiring outcome labels. Training uses weak labels based on the dataset's `Category` field.
- Resume categories describe the dataset's submission industry/category and can differ from a candidate's role or skills. The model can retrieve cross-category matches, but labels are noisy.
- Category coverage is imbalanced; BPO is particularly underrepresented, so per-category results are uneven.
- The job-description dataset used during development is synthetic and is not representative of the real job market.
- Rule-based skill parsing can mishandle space-separated skill lists and acronyms. Prefer comma-separated skills in the frontend or API.
- Experience and education extraction use heuristics and may miss context or equivalencies.
- FAISS shortlist retrieval caps the reranking pool at 200. A candidate omitted from that shortlist cannot appear in the results.
- The cluster plot is a two-dimensional PCA projection and is an approximate view of the embedding space.
- Controlled tests do not establish production accuracy, fairness, or job performance. Human review remains necessary.

## Future work

Potential next steps for an academic or engineering extension include:

- Evaluate on a larger, independently labeled set of job descriptions and resumes, with recruiter-defined relevance judgments.
- Improve skill extraction and parsing for acronyms, varied delimiters, and equivalent technologies.
- Compare learning-to-rank objectives such as XGBRanker using grouped job-description data and ranking metrics such as NDCG@10 and precision@10.
- Add cross-validation and category-aware error analysis while accounting for label noise and data scarcity.
- Explore similar-candidate search and improved explanations, then validate those features with users.
- For any real deployment, add appropriate identity management, data retention controls, auditability, security review, and bias/fairness assessment.

