# TalentRank AI — Project Context & Handoff

**Read this in full before touching any code.** This project has an established
working style and a lot of design decisions already made and validated —
don't second-guess or silently change them.

## Working style (important)

- One script at a time. Run it, show me the exact console output, wait for
  my confirmation before moving to the next step.
- CPU-only laptop (i5, no GPU). No GPU-dependent code, no model fine-tuning.
- When something looks suspicious in output (constant values, suspiciously
  perfect scores, identical numbers across unrelated categories), stop and
  diagnose before proceeding — several real bugs were caught this way
  already (see "Bugs found and fixed" below). Don't wave off odd numbers.
- Every step gets validated with printed distributions/sanity checks before
  moving forward, not just "it ran without errors."

## Goal & architecture

Given a recruiter's Job Description, return the top 10 most suitable
candidates from a resume pool, using semantic search (FAISS) + ML scoring
(RF/XGBoost), with security features to protect confidential data.

```
Recruiter JD ──► Embed (JD)
                        │
Candidate Resumes ──► Embed (batch, precomputed) ──► FAISS index ──► top-K semantic shortlist
                        │
                        ▼
              Feature Engineering (per shortlisted candidate)
                        │
                        ▼
          RF / XGBoost suitability score (0–1)
                        │
                        ▼
     Final score = α·ML_score + (1-α)·semantic_sim  ──►  sort ──► Top 10
```

Two-stage retrieval (FAISS narrows pool → ~200, then ML re-ranks) is what
keeps this fast on CPU.

## Environment

- Project folder: `talentrank-ai/` with `data/`, `models/`, `src/` subfolders
- Python venv
- Installed: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `xgboost`,
  `fastapi`, `uvicorn`, `pandas`, `numpy`, `spacy` (`en_core_web_sm`),
  `cryptography`, `python-jose[cryptography]`, `passlib`, `slowapi`

## Datasets

**Resume dataset** — `data/Resume/Resume.csv` (Kaggle: `snehaanbhawal/resume-dataset`)
- 2,484 rows × 4 columns: `ID`, `Resume_str`, `Resume_html`, `Category`
- 24 categories (avg 103.5 resumes/category): `ACCOUNTANT`, `ADVOCATE`,
  `AGRICULTURE`, `APPAREL`, `ARTS`, `AUTOMOBILE`, `AVIATION`, `BANKING`,
  `BPO`, `BUSINESS-DEVELOPMENT`, `CHEF`, `CONSTRUCTION`, `CONSULTANT`,
  `DESIGNER`, `DIGITAL-MEDIA`, `ENGINEERING`, `FINANCE`, `FITNESS`,
  `HEALTHCARE`, `HR`, `INFORMATION-TECHNOLOGY`, `PUBLIC-RELATIONS`,
  `SALES`, `TEACHER`
- **Known limitation**: `Category` reflects the industry the resume was
  submitted to, NOT job function. Confirmed multiple times — e.g. a
  software developer resume tagged `AGRICULTURE`, a UX designer tagged
  `DESIGNER` (correct) but also cross-category matches like a Java intern
  tagged `AVIATION` and a career-changing construction worker tagged
  `CONSTRUCTION` who both legitimately matched a Web Developer JD. This is
  a genuine and useful finding, not just a caveat — the model surfaces
  these correctly.
- **Known scarcity issue**: `BPO` has only 22 resumes (vs. 103.5 avg) —
  severely underrepresented. Confirmed only 9 distinct BPO resumes ever
  get labeled positive across the whole training set. Document as a
  limitation; do not try to "fix" — it's a genuine data scarcity issue,
  not a bug.

**Job description dataset** — `data/job_descriptions.csv` (Kaggle:
`ravindrasinghrana/job-description-dataset`)
- 1,615,940 rows × 23 columns. Key columns: `Job Title`, `Role`,
  `Job Description`, `skills`, `Qualifications`, `Experience`,
  `Responsibilities`, plus location/salary/company metadata.
- **Confirmed synthetic** — dataset author states it's synthetic postings
  "created for educational and research purposes." Not real job market
  data. `skills` field has no reliable delimiter (space-separated,
  TitleCase-boundary split required). `Qualifications` is often a single
  bare token (`"BA"`, `"PhD"`). `Experience` is a clean `"X to Y Years"`
  range, unlike real postings.
- Not fully embedded — used only to sample realistic-shaped JDs to test
  against and to build a skill vocabulary. Live system embeds one
  recruiter-submitted JD at a time.

## Design decisions locked in

- Embedding model: `all-MiniLM-L6-v2` — CPU-fast, frozen weights, no
  fine-tuning. Do not switch models.
- Retrieval: FAISS `IndexFlatIP` on normalized vectors (cosine similarity),
  top-200 shortlist.
- PII stripped (spaCy NER for names, regex for emails/phones) before
  anything touches disk or gets embedded/logged.
- `alpha = 0.5` for the final blend `α·ML_score + (1-α)·semantic_sim` —
  chosen after an alpha-sensitivity sweep (0.0/0.3/0.5/0.7/1.0) showed the
  ranking genuinely shifts across the range, confirming the ML layer adds
  real value over semantic-only ranking, not just noise.
- No real hiring-outcome labels exist anywhere. RF/XGBoost is trained on
  **weak labels**: a resume is labeled `1` for a given JD if the resume's
  `Category` matches a keyword-mapped category for that JD's title/role,
  `0` otherwise. `Category` itself is NEVER used as a model feature (would
  leak the label) — only as the label source.

## Steps completed (1 through 7, all done and validated)

**Step 1 — Data setup**: both datasets downloaded, shapes/columns confirmed
via `src/00_check_data.py`.

**Step 2 — Cleaning, PII masking, embeddings, FAISS index**
(`src/01_prepare_and_embed.py`): masks PII, generates 384-d MiniLM
embeddings, builds `IndexFlatIP` FAISS index. Outputs in `models/`:
`resume_embeddings.npy`, `resume_ids.npy`, `resumes_clean.csv` (now
encrypted, see Step 7), `resume_index.faiss`.

**Step 3 — Feature engineering** (`src/02_feature_engineering.py`,
shared logic in `src/features.py`): computes 5 features per shortlisted
candidate — `semantic_sim`, `skill_overlap_score`, `experience_match_score`,
`education_match_score`, `tfidf_sim`. Went through several debugging
rounds — see "Bugs found and fixed" below, all resolved.

**Step 4a — Weak-labeled training data** (`src/03_build_training_data.py`):
samples ~216 JDs across all 24 categories via a `CATEGORY_KEYWORDS`
mapping, computes features for FAISS shortlist + random negative sample
per JD, labels by Category-match. Output: `models/weak_label_training_data.csv`
(~48k rows, ~20% positive rate). Validated: positive-labeled rows show
higher `semantic_sim`/`skill_overlap_score`/`tfidf_sim` than negatives
(confirms labels correlate with features). `experience_match_score` and
`education_match_score` show near-zero separation by label — expected,
since those features aren't related to category-relevance by construction.

**Step 4b — Model training** (`src/04_train_model.py`): trained
RandomForest and XGBoost on the 5 features (Category excluded). XGBoost
selected (F1=0.508, ROC-AUC=0.811, beats RF on both metrics). Feature
importances: `semantic_sim` dominates (~46%), others contribute
meaningfully (12-15% each). Per-category recall varies widely (0.42-1.0) —
ARTS, ADVOCATE, ENGINEERING, FINANCE are weaker; BPO recall is
**statistically meaningless** (only 5 test-set positives). Saved:
`models/candidate_scorer.pkl` + `models/model_metadata.json`.

**Step 5 — Rerank formula** (`src/05_rerank.py`): blends `ML_score`
(via `predict_proba`, confirmed continuous, not hard-label `predict()`)
with `semantic_sim` at `alpha=0.5`. Validated across 3 different JD
categories (IT, Healthcare, Teacher) — Teacher performs best (most
training data), IT surfaces genuinely relevant cross-category candidates
(verified by reading actual resume text — a construction worker with
"Objective: WEB DEVELOPER", a UX/UI designer, a Java intern tagged
AVIATION — all legitimate matches despite category mismatch).

**Step 6 — FastAPI endpoint** (`src/06_api.py`): `POST /token` (JWT login,
hardcoded single recruiter user for this academic scope), `POST /rank`
(JWT-protected, runs full pipeline, returns id/category/semantic_sim/
ML_score/final_score only — no raw resume text), `GET /health`. JD text
construction was originally duplicated with slightly different phrasing
between `05_rerank.py` and `06_api.py` (caused small score drift, ~0.001-
0.05 range) — **fixed** by extracting a single canonical `build_jd_text()`
function into `src/features.py`, now imported by both. Verified
byte-identical output between script and live API after the fix.

**Step 7 — Security hardening**: `src/07_encrypt_resumes.py` encrypts
`resumes_clean.csv` → `resumes_clean.enc` via `cryptography.Fernet`
(key in `.env`), deletes plaintext. `06_api.py` decrypts into memory at
startup via lifespan event — plaintext never touches disk during normal
operation. Rate limiting added via `slowapi` (10 req/min on `/rank`).
Response fields audited — no raw text leak path found, including error
handlers (wrapped in generic 500s).

**⚠️ UNFINISHED from Step 7**: `SECRET_KEY` in `06_api.py` currently has a
hardcoded fallback: `os.getenv("SECRET_KEY", "default-secret-key-change-me")`.
This is a real security issue — if `.env` isn't loaded correctly, the
server silently signs JWTs with a public, source-visible secret instead of
failing loudly. **This needs to be fixed next**: remove the fallback,
raise a `RuntimeError` at startup if `SECRET_KEY` isn't set, and confirm
`.env` is in `.gitignore`. Not yet done — do this first.

## Bugs found and fixed during Step 3/4 (useful context, don't reintroduce)

1. Skill matching initially split JD skills string on single words,
   causing generic words (`"data"`, `"model"`) to false-match almost every
   resume → fixed with a `GENERIC_STOPWORDS` filter and phrase-aware
   splitting, then further loosened from "exact phrase required" (too
   strict, only 2 unique score values across 200 candidates) to
   per-keyword partial credit (67 unique values, real discrimination).
2. Education match initially always returned a constant `0.60` due to a
   `len(k) > 2` filter that excluded valid 2-letter degree codes like
   `"BA"` → fixed with a proper `DEGREE_HIERARCHY` mapping and word-
   boundary regex.
3. JD sample was initially "Data Scientist" — confirmed via Kaggle listing
   that this resume dataset has **no Data Science category at all**
   (closest is `INFORMATION-TECHNOLOGY`), making early validation
   meaningless. Switched to IT-relevant JD titles for testing.

## Not started

**Step 8 — Frontend (Next.js)**: explicitly deprioritized by the user as
lower priority given time constraints. Not blocking project completion.

## Immediate next task

Fix the `SECRET_KEY` fallback issue described above in `src/06_api.py`,
verify the server refuses to start without it set, then the core
technical build (Steps 1-7) is complete and tested end-to-end.

## Also useful to know

This is an **academic project**. The user has explicitly said they don't
need production-grade quality, just "reasonable" results — and has been
good about documenting limitations honestly (weak labels, synthetic JD
data, Category-as-industry-not-function, BPO data scarcity) rather than
hiding them. Keep that framing: flag issues clearly, suggest honest
disclosure in the report over silently "fixing" things that are genuine,
defensible data limitations rather than bugs.