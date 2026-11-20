# TalentRank AI: Full Project Handoff

Read this whole file before touching any code. It replaces the earlier
`PROJECT_CONTEXT.md` (which only covered Steps 1-7).

**Project:** TalentRank AI, a two-stage resume ranking system. Given a job
description (JD) and a resume pool, it returns the top-K candidates.
**Type:** academic project. "Reasonable and honestly documented" beats
"perfect". Never hide limitations; flag them.
**Machine:** Windows laptop, i5, no GPU. CPU only. No fine-tuning.

---

## 0. How to work with this user (important)

- One script / one phase at a time. Run it, show the **exact console
  output**, wait for the user's confirmation before the next phase.
- When a number looks suspicious (constant values, perfect scores, one
  category dominating), stop and diagnose before moving on. Several real
  bugs were caught this way.
- **Never loosen a test threshold to make a test pass.** If an assertion
  fails, show the failure. (A previous agent lowered a >=8 threshold to >=5
  to get a green run; it was wrong and had to be redone.)
- **Never invent substitute test data.** Another agent generated its own
  "medium" test set when the real one was missing, which invalidated the
  checks. If a required file is missing, stop and tell the user.
- **Don't claim the UI works without opening it in a browser.** API tests
  passing does not prove the frontend works.
- Lessons about the Antigravity shell on Windows (apply to any agent):
  - `python -c "..."` one-liners with quotes get mangled. Put code in `.py`
    files.
  - Backslashes in paths sometimes get stripped; prefer `cmd /c` or relative
    paths from the project root.
  - The console is cp1252: don't print unicode symbols (use ASCII "OK/PASS").
  - Never write HTML entities (`&#9679;`) in JS/JSX strings; React won't
    decode them. Use the real character, a unicode escape, or CSS.
  - Long commands (server start, 1000-resume upload ~1.5-2 min) look like
    hangs. Say what is running instead of waiting silently.
- Do not print secrets. `.env` holds `SECRET_KEY` and `ENCRYPTION_KEY`
  (Fernet). Never paste their values into chat, docs, or commits. The demo
  recruiter login is configured in the API / `.env`; ask the user for it.

---

## 1. Environment

- Project root: `d:\MyPersonalProjects\talentrank-ai` (moved from
  `C:\Users\mithu\talentrank-ai`; an interrupted cut/paste broke the venv, see
  section 9).
- Python 3.13.3, venv at `venv\`. Always run with `venv\Scripts\python.exe`.
- `requirements_verified.txt` is the verified `pip freeze` (97 packages).
  Key: torch 2.14.0+cpu, transformers 5.17.0, xgboost 3.4.1, faiss-cpu
  1.15.1, sentence-transformers 6.1.0, fastapi 0.141.1, uvicorn 0.54.0,
  scikit-learn 1.9.1, spacy 3.8.16. After installing anything new, re-freeze.
- Start the API (takes ~10-40 s to load models):
  `venv\Scripts\python.exe -m uvicorn 06_api:app --app-dir src --port 8000`
- Frontend: `cd frontend && npm run dev` (http://localhost:3000). Backend must
  be on http://localhost:8000 (CORS already allows localhost:3000).
- `.gitignore` exists (`.env` is first; also venv, models, data).

Folders: `data/`, `models/`, `src/`, `tests/`, `TestingData/`, `frontend/`.

---

## 2. Datasets

**Resumes:** `data/Resume/Resume.csv` (Kaggle `snehaanbhawal/resume-dataset`).
2,484 rows: `ID`, `Resume_str`, `Resume_html`, `Category`. 24 categories
(avg 103.5 per category): ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS,
AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION,
CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS,
HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER.
There is **no Data Science category** (closest: INFORMATION-TECHNOLOGY).

**Job descriptions:** `data/job_descriptions.csv` (Kaggle
`ravindrasinghrana/job-description-dataset`). 1,615,940 rows x 23 columns.
**Synthetic** (the author says so). `skills` has no delimiter (TitleCase
concatenated), `Qualifications` is often a single token ("BA", "PhD"),
`Experience` is "X to Y Years". Used only to sample test JDs and to build weak
labels; the live system embeds one recruiter JD at a time.

Known data limitations (document these in the report):
1. `Category` is the industry the resume was filed under, not the job
   function. Examples found: a software developer tagged AGRICULTURE; a Java
   intern tagged AVIATION; a construction worker with objective "WEB
   DEVELOPER". The ranking surfaced these correctly.
2. No real hiring-outcome labels exist anywhere, so training uses weak labels.
3. The JD dataset is synthetic.
4. BPO has only 22 resumes; only 9 distinct BPO resumes ever became positives;
   BPO test recall is based on 5 positives and is meaningless.

---

## 3. Architecture and locked design decisions

```
JD -> embed (MiniLM) -> FAISS IndexFlatIP top-200 shortlist
   -> 5 features per candidate -> XGBoost ML_score (predict_proba)
   -> final = alpha*ML_score + (1-alpha)*semantic_sim (alpha=0.5) -> top 10
```

- Embeddings: `all-MiniLM-L6-v2`, frozen, 384-d, normalized. Do not switch
  models or fine-tune.
- FAISS `IndexFlatIP` (exact cosine). PII (names via spaCy NER, emails/phones
  via regex) is masked before anything is embedded or stored; plaintext
  resumes are never written to disk.
- alpha=0.5 was chosen after a sweep (0.0/0.3/0.5/0.7/1.0) showed the ranking
  genuinely changes with alpha, so the ML layer is not a no-op.
- The 5 features (order matters, it is the training order):
  `semantic_sim`, `skill_overlap_score`, `experience_match_score`,
  `education_match_score`, `tfidf_sim`.
- `Category` is NEVER a model feature (it is the label source; using it
  would leak).

---

## 4. Where ML is used (for the report)

1. **Pretrained transformer (MiniLM)**: converts text to vectors; this does
   most of the work. Not trained by us.
2. **XGBoost classifier (trained by us)**: scores the 5 features, trained
   with weak labels. It refines, it is not the core. `semantic_sim` is ~46% of
   feature importance.
3. **Supporting:** spaCy NER (PII masking), TF-IDF (keyword similarity).
4. **Unsupervised:** KMeans + PCA cluster map (Step 8b).
5. Not ML: skill/experience/education matching are rule-based (regex,
   keyword lists, degree hierarchy).

Weak-label training: ~216 JDs sampled across the categories via a
`CATEGORY_KEYWORDS` mapping; for each JD, a FAISS shortlist plus random
negatives; label=1 if the resume's Category matches the JD's mapped category.
48,276 rows, 9,531 positives (~20%).

Model results (held-out test, weak labels): XGBoost F1 0.508, ROC-AUC 0.811
(selected); RandomForest F1 0.477, AUC 0.790. Per-category recall is uneven
(ARTS, ADVOCATE, ENGINEERING, FINANCE weak).

---

## 5. What is built (all verified unless marked)

| Step | What | Files |
|---|---|---|
| 1 | Data setup/check | `src/00_check_data.py` |
| 2 | PII masking, embeddings, FAISS | `src/01_prepare_and_embed.py` -> `models/` |
| 3 | Feature engineering | `src/02_feature_engineering.py`, `src/features.py` |
| 4a | Weak-label training data | `src/03_build_training_data.py` |
| 4b | Train RF/XGB, save best | `src/04_train_model.py` -> `models/candidate_scorer.pkl`, `model_metadata.json` |
| 5 | Rerank formula, 3-JD validation, alpha sweep | `src/05_rerank.py` |
| 6 | FastAPI + JWT | `src/06_api.py` |
| 7 | Security: Fernet-encrypted resumes, rate limit, fail-fast SECRET_KEY | `src/07_encrypt_resumes.py`, `models/resumes_clean.enc` |
| 8a | Dynamic resume upload sessions | `src/06_api.py`, `build_resume_index()` in `features.py` |
| 8b | Cluster map | `src/cluster.py`, `GET /cluster-map/{session_id}` |
| 8 | Frontend (Next.js) | `frontend/` (see section 7 for unverified parts) |

Key behaviors:
- `models/resumes_clean.enc` is decrypted into memory at server start. Scripts
  that read `models/resumes_clean.csv` (e.g. `05_rerank.py`) need a temporary
  decrypted copy; delete it afterwards.
- `SECRET_KEY` has no fallback: the server raises `RuntimeError` at import if
  it is missing (tested both ways).
- Rate limit: 10 requests/min per IP on `/rank` (tests can hit it).
- `models/` and `data/` are git-ignored; regenerate `models/` by running
  scripts 01 -> 07 in order.

### API summary (read `src/06_api.py` for exact schemas)

- `POST /token`: JWT login.
- `GET /health`: status plus models loaded.
- `POST /rank`: JWT. Optional `session_id` in the body (uploaded pool);
  default pool if omitted. Optional alpha. Returns top 10:
  `id, category, semantic_sim, ML_score, final_score`. No resume text.
  A session `/rank` also saves `last_top10` on the session.
- `POST /upload-resumes`: JWT, CSV needing `ID` and `Resume_str` (`Category`
  optional), max 2000 rows, runs as a BackgroundTask and returns a
  `session_id`. 400s for missing columns or too many rows.
- `GET /upload-status/{session_id}`: processing/done/error with counts.
- `GET /cluster-map/{session_id}`: needs >=30 resumes (else 400). Auto-k
  KMeans (best silhouette, k in 2..min(8, n//10)), PCA 2D, per-cluster TF-IDF
  keywords (domain stop-words removed), `pca_variance_explained`, ARI/NMI if
  Category present, and after a `/rank`: `jd_point`, `jd_nearest_cluster`
  (majority vote of the clusters of the actual `/rank` top 10),
  `top10_ids` (same order as `/rank`).
- Sessions are in memory only (lock-protected, 2 h TTL). Lost on restart by
  design.

---

## 6. Validation evidence and test data

**The regression anchor** (confirmed many times, including after the venv
recovery and every API change): default pool, Web Developer benchmark JD
(`build_jd_text` canonical JD text) -> top result id **91153752**,
final_score **0.7326**. Any change must keep this.

| Test | Result |
|---|---|
| Step 5 across IT / Healthcare / Teacher JDs | Teacher cleanest (9/10 TEACHER). IT surfaced cross-category candidates (verified by reading resume text). |
| Alpha sweep | Ranking shifts materially across alpha 0..1 |
| 1000-row easy set + Backend SWE JD | 10/10 INFORMATION-TECHNOLOGY |
| Edge case: IT-only session vs Teacher / Nurse JD | 10 results, valid shape, low scores; ML_score collapses toward 0 |
| Cluster map on 1000-row easy set | k=8, silhouette 0.2384; IT isolated in its own cluster (python, developer, engineer, software, aws); `/rank` top-10 10/10 in that cluster |
| Medium set (300) + Data Analyst JD | top-10: 9 strong + 1 partial; P@10(strong)=0.9, all 10 relevant, NDCG@10=0.958; no decoys in top 10 |

Caveats on those numbers:
- Do NOT cite the cluster ARI 0.79 / NMI 0.92 from the easy set; it is
  templated and artificially separable. Recompute on `Resume.csv` for the
  report.
- The medium-set #10 (id 200203, partial business analyst) had the highest
  ML_score (0.975) but the lowest semantic similarity (0.629), a small
  example of the ML layer over-trusting clean experience/education features.
- One IT resume (id 18752129) scored ML 0.79 against an unrelated Teacher
  JD: the weak-label model sometimes latches onto generic professional
  language.
- All test sets are synthetic and controlled; call them "controlled
  validation", not real-world accuracy.

### Test data inventory (`TestingData/`)

| File | Purpose |
|---|---|
| `test_candidates_1000_easy.csv` + `test_jd_single.csv` | 150 IT vs 850 clearly unrelated resumes; scalability/demo; deliberately trivial. JD skills are space-separated (see issue 1 below). |
| `test_candidates_300_medium.csv` | 300 resumes, IDs **200001-200300**: 40 strong data analysts, 80 partial fits (business analyst, ML data scientist, BI developer, financial/marketing analyst, backend engineer, junior), 180 irrelevant incl. 20 decoys (data entry clerk, data center technician, medical records clerk) |
| `test_jd_medium.csv` | Data Analyst JD, skills comma-separated: SQL, Python, Tableau, Excel, Data visualization, Statistics; 2 to 8 Years; Bachelor |
| `test_medium_answer_key.csv` | `ID, relevance (0/1/2), profile, note`. 40 x 2, 80 x 1, 180 x 0. For scoring only; never upload it. |

**Pending:** the user must copy the three medium files into `TestingData/`.
A reverted agent attempt left a substitute `test_candidates_300_medium.csv`
there (IDs 100xxx, about 120,626 bytes; the correct file is about 124,165
bytes and has IDs 200xxx). It must be overwritten.

---

## 7. Frontend status

`frontend/` (Next.js App Router, plain JS): `app/page.js`, an API helper,
`page.module.css`, `globals.css`, `layout.js`, and a `ClusterMap.js` scatter
plot (recharts). Flow: login -> upload CSV (or default pool) with progress
polling -> JD form -> results table (pool badge, error handling for
401/404/429/500) -> candidate map.

- **Backend side verified:** API flows and CORS headers tested by script.
- **Not verified:** the browser UI itself. The agent only read the code. Do
  the manual check: login, upload the 1000-row CSV, run the JD, results all
  IT, "New search", default pool shows 91153752, no console errors.
- **Prompts already given to the agent, results unknown (verify):**
  1. A "clean glassmorphism" redesign (glass only for outer frame, flat
     surfaces inside, design tokens, 3-step progress indicator, drop zone,
     clean results table, skeletons, responsive to 375 px).
  2. A bug fix: the rank column showed raw `&#9679;` text for ranks 1-3
     (HTML entity in a JS string). Fix: real character or a CSS-drawn
     circle; grep the frontend for other entities.
- The cluster map's JD star is unreliable (PCA keeps ~21% of variance).
  Rely on the highlighted top-10 dots and the "closest cluster" text, and
  label the plot as an approximate projection.

---

## 8. IN PROGRESS: Step 9, SHAP explainability

Goal: per-candidate "why" for the top 10. Status: **Phase 0 done once; no
code from Step 9 exists in the repo** (a first attempt was reverted; its
leftover `tests/test_9_explain.py` was deleted). Blocked only on the user
copying the medium test files (section 6).

Design decisions:
- Do NOT install the `shap` package (numba risk on Python 3.13). Use
  XGBoost's built-in TreeSHAP: `booster = model.get_booster()`;
  `booster.predict(xgboost.DMatrix(X, feature_names=FEATURES),
  pred_contribs=True)` returns shape (n, 6): 5 contributions plus bias in the
  last column. Verified: `sigmoid(row sum)` equals `predict_proba`.
- Contributions are in **log-odds**, never present them as probability
  points. UI shows direction and signed share plus the actual feature values.
- SHAP explains only the XGBoost part; final = alpha*ML + (1-alpha)*semantic.
  Say so in the UI (`explanation_note`).
- Weak labels mean explanations show what the model learned from the
  Category proxy, not "true suitability".
- Do NOT change the signature/return type of `calculate_skill_overlap`
  (other scripts use it). Add a new helper `get_skill_lists(jd_skills,
  resume_text) -> (matched, missing)` reusing the same logic.
- Compute SHAP only for the final top 10, in both the default-pool and
  session paths of `/rank`.
- Per result, add `explanation`: `{bias, features:[{name, value,
  contribution, share} x5], reasons:[<=3 plain-English strings], matched_skills,
  missing_skills}`. Reasons = 2 strongest positive + 1 strongest negative
  feature, templated from the feature value (e.g. "Strong skill overlap
  (0.67)"), no log-odds numbers. Top-level `explanation_note`. No resume text
  anywhere in responses; no explanation string longer than 80 chars.

Phases (stop and show output after each):
- **Phase 0:** confirm no leftovers; confirm medium files (IDs 200001-200300,
  key relevance counts 40/80/180, JD is the Data Analyst one).
- **Phase 1 (backend core, no server):** `get_skill_lists`; prove
  `calculate_skill_overlap` is unchanged on 20 sample resumes; contributions
  helper; unit test: additivity within 1e-4 on 50 training rows; sign sanity
  on a strong vs weak feature profile.
- **Phase 2 (API):** add `explanation` + `explanation_note`; update Pydantic
  models; `tests/test_9_explain.py` (server-subprocess pattern like
  `tests/test_8b_cluster.py`): additivity from the API response,
  regression (91153752 / 0.7326), medium set: >=8 of the top 10 have
  relevance 2 (a previous run got 9/10), print P@10 and NDCG@10 from the key,
  print reasons and matched/missing skills for ranks 1, 5, 10.
- **Phase 3 (frontend):** "Why" control per row opening a detail row: reason
  chips (green up / red down), diverging bar chart of the 5 shares with
  friendly names (Meaning match, Skill overlap, Experience fit, Education
  fit, Keyword match) and real values, matched/missing skill chips, the
  footnote. One row open at a time, keyboard accessible, works at 375 px.
  `npm run build`, then browser-check ranks #1 and #10 on the medium set.

What good looks like: rank 10 on the medium set (partial business analyst)
should show weaker skill overlap and missing Python/Tableau compared to rank 1.

---

## 9. Incidents and fixes already made (don't reintroduce)

1. **Skill matcher:** first too lenient (generic words like "data" matched
   everything -> 1.0 for all), then too strict (only 2 unique values). Now:
   JD skills parsed into phrases, `GENERIC_STOPWORDS`, per-keyword partial
   credit (67 unique values). Education matcher fixed (2-letter degrees like
   "BA", `DEGREE_HIERARCHY`, word-boundary regex).
2. **JD text mismatch:** `05_rerank.py` and `06_api.py` built the JD string
   differently (scores drifted ~0.001-0.05). Fixed with one canonical
   `build_jd_text()` in `features.py`.
3. **SECRET_KEY fallback:** removed; server refuses to start without it.
4. **Interrupted folder move:** venv lost `Scripts/` and many packages (torch,
   transformers, xgboost, yaml dir, spacy deps, uvicorn, bcrypt, ...). Fixed
   by recreating the venv scripts, installing CPU-only torch from the PyTorch
   CPU index, and iterating `pip check` until clean. Lesson: `pip check` plus
   an import test, not a folder listing, is the real health check.
5. **Cluster `jd_nearest_cluster`:** originally used centroid cosine and
   pointed at the designer cluster even though all top-10 were IT. Then an
   agent recomputed its own cosine top-10 (4 of 10 not IT). Final: majority
   vote over the real `/rank` top-10 saved on the session.
6. **sklearn:** `TfidfVectorizer(stop_words=...)` must be a `list` (or
   "english"), not a set/frozenset, or it errors/returns empty keywords.

---

## 10. Known issues / honest limitations

1. **Skill parser and JD formatting:** `parse_jd_skills` handles commas,
   semicolons, slashes, pipes; otherwise it splits on TitleCase boundaries,
   which breaks acronyms ("SQL AWS" -> single letters, dropped).
   `test_jd_single.csv` and `test_jds.csv` use space-separated skills, so the
   skill feature likely saw only part of the list (rankings were still good,
   driven by semantic similarity). Use **comma-separated skills** in tests and
   in the UI. Consider making the parser robust.
2. `experience_match_score` takes the max "N years" phrase anywhere in the
   resume; neutral 0.5 when unparseable. Education collapses Bachelor/Master
   into one tier when the JD wants a PhD.
3. XGBoost ML scores tie (few distinct values); semantic similarity breaks
   ties.
4. The Category label is noisy; the weak-label ceiling is real.
5. Rate limit of 10 req/min per IP will trip fast test scripts.
6. Sessions are not persisted and not encrypted (deliberate scope decision;
   state it in the report).

---

## 11. Remaining work, in priority order

1. User copies the three medium test files into `TestingData/`; re-run Step 9
   Phase 0 verification, then Phases 1-3.
2. Browser-verify the frontend (basic flow, redesign, rank-column fix, cluster
   map). Fix whatever is off.
3. Write the GitHub README. Chosen title: **TalentRank AI**. One-liner:
   "Two-stage candidate ranking system using semantic search (FAISS) and ML
   scoring (XGBoost) to match resumes against job descriptions, built with
   Python, FastAPI and Next.js." Include run steps, how to regenerate
   `models/` (scripts 01 -> 07), security notes (JWT, Fernet, rate limiting,
   `.env` never committed), screenshots.
4. Write the project report (see outline below).
5. Optional ML extras, by value: XGBRanker (`rank:pairwise`) with
   NDCG@10/P@10 on a small hand-labeled gold set (3-4 JDs scored with the
   answer-key approach); embedding-based skill matching; cross-validation and
   a model comparison; "find similar candidates" via FAISS.
6. Housekeeping: delete stray scripts in the project root (`compare.py`,
   `test_api.py`, any `_*.py` helpers), make sure no plaintext
   `models/resumes_clean.csv` exists, regenerate `requirements_verified.txt`.

### Report outline (material already collected)

1. Problem and architecture (two-stage retrieval, why CPU-friendly).
2. Data: both datasets, PII masking, the Category-vs-job-function finding,
   synthetic JD data, BPO scarcity.
3. Methods: embeddings, FAISS, 5 features, weak-label construction (and why
   `Category` is excluded from features), XGBoost vs RF, rerank formula and
   alpha sweep.
4. Results: model metrics (weak-label), qualitative cross-category discoveries
   (construction worker -> web dev, UX designer, Java intern tagged AVIATION),
   the Teacher/Nurse/IT-only edge cases, medium-set P@10 0.9 / NDCG@10 0.96,
   the cluster map, SHAP explanations (once done).
5. Security: PII masking before storage, Fernet encryption at rest, JWT,
   rate limiting, fail-fast secrets, no resume text in responses.
6. Engineering rigor: bugs found and fixed (section 9), reproducibility
   (`requirements_verified.txt`, regression anchor), the environment recovery.
7. Limitations (sections 2 and 10) and future work (real labels, XGBRanker,
   gold evaluation set, better skill extraction).

---

## 12. First actions for the new assistant

1. Read this file. Confirm you understand section 0.
2. Ask the user to confirm the three medium files are in `TestingData/`
   (candidates IDs 200001-200300). Do not create substitutes.
3. Run Step 9 Phase 0 verification, then Phase 1 only. Stop and show output.