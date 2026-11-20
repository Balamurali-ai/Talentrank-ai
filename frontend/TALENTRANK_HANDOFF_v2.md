# TalentRank AI: Full Project Handoff (v2)

Read this whole file before touching any code. It replaces
`TALENTRANK_HANDOFF.md` and `PROJECT_CONTEXT.md`.

**Project:** TalentRank AI, a two-stage resume ranking system. Given a job
description (JD) and a resume pool it returns the top-K candidates.
**Type:** academic project. "Reasonable and honestly documented" beats
"perfect". Never hide limitations; flag them.
**Machine:** Windows laptop, i5, no GPU. CPU only. No fine-tuning.

**Where we are right now (one paragraph):** Steps 1-8b and the Step 9
backend (SHAP explanations) are built and tested. An open problem remains:
the plain-English "reasons" in the explanations can contradict the
candidate's actual feature values (for example "Experience below
requirement" for a candidate whose experience score is 1.0). A fix prompt
was written (section 8, Part 2) but its result has not been seen. After
that: Step 9 frontend ("Why" row), browser verification of the whole UI,
README, report.

---

## 0. How to work with this user (important)

- One script / one phase at a time. Run it, show the **exact console
  output**, wait for the user's confirmation before the next phase.
- When a number looks suspicious (constant values, perfect scores, one
  category dominating, a reason that contradicts the data), stop and
  diagnose before moving on. Several real bugs were caught this way.
- **Never loosen a test threshold to make a test pass.** If an assertion
  fails, show the failure. (An agent once lowered a >=8 threshold to >=5
  to get green; it was wrong and had to be redone.)
- **Never invent substitute test data.** An agent once generated its own
  "medium" set when the real one was missing, which invalidated the checks.
  If a required file is missing, stop and tell the user.
- **Don't claim the UI works without opening it in a browser.** API tests
  passing does not prove the frontend works.
- **Don't call a failing check "expected and honest" and move on** unless
  you can explain the cause. (An agent did this for the SHAP sign check;
  the real cause is in section 8.)
- Budget: do NOT poll task status repeatedly or send "still running"
  messages (a previous agent burned its context that way). For anything
  slow, start it once with output redirected to a log file, wait, read the
  log once, keep messages short.
- Windows shell lessons (apply to any agent):
  - `python -c "..."` one-liners with quotes get mangled; use `.py` files.
  - Backslashes in paths sometimes get stripped; prefer `cmd /c` or relative
    paths from the project root.
  - The console is cp1252: print ASCII only ("OK/PASS").
  - Never write HTML entities (`&#9679;`) in JS/JSX strings; React won't
    decode them. Use the real character, a unicode escape, or CSS.
  - Server start takes ~10-40 s; a 1000-resume upload takes ~1.5-2 min.
- Do not print secrets. `.env` holds `SECRET_KEY` and `ENCRYPTION_KEY`
  (Fernet). Never paste values into chat, docs or commits. The demo
  recruiter login is configured in the API / `.env`; ask the user for it.

---

## 1. Environment

- Project root: `d:\MyPersonalProjects\talentrank-ai` (moved from
  `C:\Users\mithu\talentrank-ai`; an interrupted move broke the venv, see
  section 9).
- Python 3.13.3, venv at `venv\`. Always run `venv\Scripts\python.exe`.
- `requirements_verified.txt` is the verified `pip freeze` (97 packages):
  torch 2.14.0+cpu, transformers 5.17.0, xgboost 3.4.1, faiss-cpu 1.15.1,
  sentence-transformers 6.1.0, fastapi 0.141.1, uvicorn 0.54.0,
  scikit-learn 1.9.1, spacy 3.8.16. Re-freeze after installing anything.
- Start the API:
  `venv\Scripts\python.exe -m uvicorn 06_api:app --app-dir src --port 8000`
- `src/06_api.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  (via `os.environ.setdefault`, before imports) to stop a Hugging Face
  network stall at startup. This is fine because `all-MiniLM-L6-v2` is
  already cached here. **On a fresh machine the model must be downloaded
  once with offline mode off**; put that in the README.
- Frontend: `cd frontend && npm run dev` (http://localhost:3000); backend on
  http://localhost:8000 (CORS allows localhost:3000).
- `.gitignore` exists (`.env` first; also venv, models, data).

Folders: `data/`, `models/`, `src/`, `tests/`, `TestingData/`, `frontend/`.
Stray files in the project root to delete later: `compare.py`, `test_api.py`.

---

## 2. Datasets

**Resumes:** `data/Resume/Resume.csv` (Kaggle `snehaanbhawal/resume-dataset`).
2,484 rows: `ID`, `Resume_str`, `Resume_html`, `Category`. 24 categories
(avg 103.5 each): ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS,
AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF,
CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE,
FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES,
TEACHER. There is no Data Science category.

**Job descriptions:** `data/job_descriptions.csv` (Kaggle
`ravindrasinghrana/job-description-dataset`). 1,615,940 rows x 23 columns.
**Synthetic** (the author says so). `skills` has no delimiter (TitleCase
concatenated), `Qualifications` is often one token ("BA", "PhD"),
`Experience` is "X to Y Years". Used only to sample test JDs and build weak
labels; the live system embeds one recruiter JD at a time.

Data limitations (put these in the report):
1. `Category` is the industry the resume was filed under, not the job
   function (a software developer tagged AGRICULTURE; a Java intern tagged
   AVIATION; a construction worker with objective "WEB DEVELOPER"). The
   ranking surfaced these correctly.
2. No real hiring-outcome labels exist, so training uses weak labels.
3. The JD dataset is synthetic.
4. BPO has only 22 resumes; only 9 distinct BPO resumes ever became
   positives; its test recall rests on 5 positives and is meaningless.

---

## 3. Architecture and locked design decisions

```
JD -> embed (MiniLM) -> FAISS IndexFlatIP top-200 shortlist
   -> 5 features per candidate -> XGBoost ML_score (predict_proba)
   -> final = alpha*ML_score + (1-alpha)*semantic_sim (alpha=0.5) -> top 10
   -> (Step 9) TreeSHAP explanation for the top 10
```

- Embeddings: `all-MiniLM-L6-v2`, frozen, 384-d, normalized. Do not switch
  models or fine-tune.
- FAISS `IndexFlatIP` (exact cosine). PII (names via spaCy NER, emails and
  phones via regex) is masked before anything is embedded or stored;
  plaintext resumes are never written to disk.
- alpha=0.5 chosen after a sweep (0.0/0.3/0.5/0.7/1.0) showed the ranking
  genuinely changes with alpha.
- The 5 features, in training order: `semantic_sim`, `skill_overlap_score`,
  `experience_match_score`, `education_match_score`, `tfidf_sim`.
- `Category` is NEVER a model feature (it is the label source; using it
  would leak).

---

## 4. Where ML is used (for the report)

1. Pretrained transformer (MiniLM): text -> vectors; does most of the work.
   Not trained by us.
2. XGBoost classifier (trained by us): scores the 5 features, trained on
   weak labels. A refinement, not the core. `semantic_sim` is ~46% of
   feature importance.
3. Supporting: spaCy NER (PII), TF-IDF (keyword similarity).
4. Unsupervised: KMeans + PCA cluster map (Step 8b).
5. Explainability: TreeSHAP via XGBoost's built-in `pred_contribs` (Step 9).
6. Not ML: skill/experience/education matching are rule-based.

Weak-label training: ~216 JDs sampled across categories via a
`CATEGORY_KEYWORDS` mapping; per JD, a FAISS shortlist plus random
negatives; label=1 if the resume's Category matches the JD's mapped
category. 48,276 rows, 9,531 positives (~20%).

Model results (held-out, weak labels): XGBoost F1 0.508, ROC-AUC 0.811
(selected); RandomForest F1 0.477, AUC 0.790. Per-category recall is uneven
(ARTS, ADVOCATE, ENGINEERING, FINANCE weak).

---

## 5. What is built

| Step | What | Files | Status |
|---|---|---|---|
| 1 | Data setup/check | `src/00_check_data.py` | done |
| 2 | PII masking, embeddings, FAISS | `src/01_prepare_and_embed.py` | done |
| 3 | Feature engineering | `src/02_feature_engineering.py`, `src/features.py` | done |
| 4a | Weak-label training data | `src/03_build_training_data.py` | done |
| 4b | Train RF/XGB | `src/04_train_model.py` -> `models/candidate_scorer.pkl` | done |
| 5 | Rerank, 3-JD validation, alpha sweep | `src/05_rerank.py` | done |
| 6 | FastAPI + JWT | `src/06_api.py` | done |
| 7 | Fernet-encrypted resumes, rate limit, fail-fast SECRET_KEY | `src/07_encrypt_resumes.py` | done |
| 8a | Upload sessions | `src/06_api.py`, `build_resume_index()` in `features.py` | done |
| 8b | Cluster map | `src/cluster.py`, `/cluster-map/{session_id}` | done |
| 8 | Frontend (Next.js) | `frontend/` | built; browser check pending (section 7) |
| 9 backend | TreeSHAP explanations | `src/explain.py`, edits in `06_api.py`, `features.py` | works; wording bug open (section 8) |
| 9 frontend | "Why" row | `frontend/` | not started |

Key behaviors:
- `models/resumes_clean.enc` is decrypted into memory at server start.
  Scripts that read `models/resumes_clean.csv` (e.g. `05_rerank.py`) need a
  temporary decrypted copy; delete it afterwards.
- `SECRET_KEY` has no fallback: the server raises `RuntimeError` at import
  if it is missing (tested both ways).
- Rate limit: 10 requests/min per IP on `/rank`; tests can hit it (429).
- `models/` and `data/` are git-ignored; regenerate `models/` by running
  scripts 01 -> 07 in order.

### API summary (read `src/06_api.py` for exact schemas)

- `POST /token`: JWT login. `GET /health`: status plus models loaded.
- `POST /rank`: JWT. Optional `session_id` (uploaded pool) else default
  pool; optional alpha. Returns top 10: `id, category, semantic_sim,
  ML_score, final_score`, plus (Step 9) `explanation` per result and a
  top-level `explanation_note`. No resume text. A session `/rank` saves
  `last_top10` on the session.
- `POST /upload-resumes`: JWT, CSV needing `ID` and `Resume_str`
  (`Category` optional), max 2000 rows, BackgroundTask, returns
  `session_id`. 400s for missing columns / too many rows.
- `GET /upload-status/{session_id}`: processing/done/error with counts.
- `GET /cluster-map/{session_id}`: needs >=30 resumes (else 400). Auto-k
  KMeans (best silhouette, k in 2..min(8, n//10)), PCA 2D, per-cluster
  TF-IDF keywords (domain stop-words removed), `pca_variance_explained`,
  ARI/NMI if Category present; after a `/rank`: `jd_point`,
  `jd_nearest_cluster` (majority vote of the clusters of the real `/rank`
  top 10), `top10_ids` (same order as `/rank`).
- Sessions are in memory only (lock-protected, 2 h TTL), lost on restart by
  design.

---

## 6. Validation evidence and test data

**Regression anchor** (confirmed many times, including after the venv
recovery and after Step 9): default pool, Web Developer benchmark JD
(canonical `build_jd_text`) -> top result id **91153752**, final_score
**0.7326**. Any change must keep this.

| Test | Result |
|---|---|
| Step 5: IT / Healthcare / Teacher JDs | Teacher cleanest (9/10 TEACHER). IT surfaced cross-category candidates (verified by reading resume text) |
| Alpha sweep | Ranking shifts materially across alpha 0..1 |
| 1000-row easy set + Backend SWE JD | 10/10 INFORMATION-TECHNOLOGY |
| IT-only session vs Teacher / Nurse JD | 10 results, valid shape, low scores; ML_score collapses toward 0 |
| Cluster map, 1000-row easy set | k=8, silhouette 0.2384; IT isolated (python, developer, engineer, software, aws); `/rank` top-10 10/10 in that cluster |
| Medium set via the UI | top-10: 9 strong + 1 partial; P@10(strong) 0.9, NDCG@10 0.958 |
| Medium set via `tests/test_9_explain.py` | P@10 1.0, NDCG@10 1.0 (differs from the UI run, probably because of how JD fields were supplied; report both honestly) |
| Step 9 additivity | max abs error 1.77e-07 (50 training rows); ~3e-07 on API responses |

Caveats:
- Do NOT cite cluster ARI 0.79 / NMI 0.92 from the easy set (templated,
  artificially separable). Recompute on `Resume.csv` for the report.
- Medium-set #10 in the UI run (id 200203, partial business analyst) had
  the highest ML_score (0.975) but the lowest semantic similarity (0.629): a
  small example of the ML layer over-trusting clean experience/education.
- One IT resume (id 18752129) scored ML 0.79 against an unrelated Teacher
  JD: the weak-label model sometimes latches onto generic professional
  language.
- All test sets are synthetic and controlled: call it "controlled
  validation", not real-world accuracy.

### Test data (`TestingData/`)

| File | Purpose |
|---|---|
| `test_candidates_1000_easy.csv` + `test_jd_single.csv` | 150 IT vs 850 unrelated; scalability/demo; trivial. JD skills are space-separated (see section 10, issue 1) |
| `test_candidates_300_medium.csv` | IDs **200001-200300**, 124,165 bytes: 40 strong data analysts, 80 partial fits (business analyst, ML data scientist, BI developer, financial/marketing analyst, backend engineer, junior), 180 irrelevant incl. 20 decoys (data entry clerk, data center technician, medical records clerk) |
| `test_jd_medium.csv` | Data Analyst JD, comma-separated skills: SQL, Python, Tableau, Excel, Data visualization, Statistics; 2 to 8 Years; Bachelor |
| `test_medium_answer_key.csv` | `ID, relevance (0/1/2), profile, note`; 40x2, 80x1, 180x0. Scoring only; never upload |

All three medium files were verified in place (correct IDs, counts, JD).

---

## 7. Frontend status

`frontend/` (Next.js App Router, plain JS): `app/page.js`, an API helper,
`page.module.css`, `globals.css`, `layout.js`, and `ClusterMap.js`
(recharts scatter). Flow: login -> upload CSV (or default pool) with
progress polling -> JD form -> results table (pool badge, error handling for
401/404/429/500) -> candidate map.

- **Backend side verified** by script (flows, CORS headers).
- **Browser not verified.** Manual check needed: login, upload the 1000-row
  CSV, run the JD (results all IT), "New search", default pool shows
  91153752, no console errors.
- **Prompts given to the earlier agent, results unknown (verify):**
  1. A "clean glassmorphism" redesign (glass only for outer frame, flat
     surfaces inside, design tokens, 3-step progress indicator, drop zone,
     clean table, skeletons, responsive to 375 px).
  2. A bug fix: the rank column showed raw `&#9679;` for ranks 1-3 (HTML
     entity in a JS string). Fix with a real character or a CSS-drawn circle
     and grep for other entities.
- The cluster map's JD star is unreliable (PCA keeps ~21% of variance). Rely
  on the highlighted top-10 dots and the "closest cluster" text; label the
  plot an approximate projection.

---

## 8. Step 9: SHAP explainability (current work)

### What exists (audited and passing)
- `src/explain.py` (new): `compute_contributions(model, X)` returning
  per-row feature contributions plus bias, and `build_explanation()`.
- `src/features.py`: private `_match_skills` helper; `calculate_skill_overlap`
  delegates to it with its original signature and return `(float, str)`;
  new `get_skill_lists(jd_skills, resume_text) -> (matched, missing)`.
- `src/06_api.py`: SHAP computed only for the final top 10 (both default and
  session paths); Pydantic models for the explanation; `explanation_note`.
- `tests/test_9_unit.py` (no server) and `tests/test_9_explain.py` (server
  subprocess pattern like `tests/test_8b_cluster.py`).
- Design rules kept: no `shap` package (numba risk on Python 3.13); XGBoost
  `booster.predict(DMatrix(X, feature_names=FEATURES), pred_contribs=True)`
  returns (n, 6) with bias last; `sigmoid(row sum)` equals `predict_proba`.
  Contributions are **log-odds**, never present them as probability points.
- Explanation object per result: `{bias, features:[{name, value,
  contribution, share} x5], reasons:[<=3 strings], matched_skills,
  missing_skills}`; no resume text; every string <= 80 chars.

### Results so far
- Unit additivity: max abs error 1.77e-07. PASS.
- Integration: regression 91153752 / 0.7326 PASS; all results have
  explanations; strings <= 80 chars; medium-set P@10 1.0 / NDCG@10 1.0.
- Medium-set rank 1 (id 200118): all 6 skills matched; semantic_sim is the
  main driver (+1.70 log-odds, 53.5% share), skill overlap +0.65.
  Ranks 5 and 10 have identical ML scores (0.8782); semantic_sim separates
  them.

### OPEN PROBLEM
1. **Reason text contradicts the data.** Rank 1 shows "Experience below
   requirement" although `experience_match_score` is 1.0 (meets the
   requirement) while its SHAP contribution is -0.34. The wording comes from
   the feature value but the direction from the SHAP sign, so they can
   disagree.
2. **Sign sanity unit check fails:** a strong profile (skill overlap 0.67)
   gets a negative skill contribution (-0.84); a zero-skill profile gets a
   positive one (+0.59). `semantic_sim` behaves correctly.
3. Likely cause (not yet confirmed): a boosted-tree contribution at a single
   point is not guaranteed monotone in the feature, and the weak labels
   taught the model odd rules. This matches earlier evidence that experience
   and education carry 12-14% of XGBoost importance even though they barely
   separated the labels in training. The SHAP numbers are probably faithful
   to the model; the model's learned behavior is what looks odd. Treat it as
   a model limitation to disclose, and make the UI honest about it.

### NEXT: Part 2, consistency fix (prompt was given; result not seen)
Do this before any frontend work. Check first whether it was already done
(grep `explain.py` for a "facts" list and a consistency gate; check for
`tests/diag_shap_dependence.py`).
1. Diagnose (report only): `tests/diag_shap_dependence.py` (no server). On
   2000 random rows of `models/weak_label_training_data.csv` compute
   TreeSHAP and print, per feature, mean contribution by value bin
   (semantic_sim and tfidf_sim by quintile; skill_overlap <0.1, 0.1-0.3,
   0.3-0.5, >0.5; experience and education by distinct values) plus the
   Spearman correlation between value and contribution. State in plain
   words which features are monotone.
2. Fix reasons: keep SHAP ordering but add a consistency gate. A positive
   reason needs contribution > 0 AND a favorable value (semantic_sim >= 0.5,
   skill_overlap >= 0.4, experience >= 1.0, education >= 1.0, tfidf_sim >=
   0.1). A negative reason needs contribution < 0 AND an unfavorable value.
   If the SHAP sign and the value disagree, generate no reason from that
   feature. Show fewer than 3 reasons if fewer qualify; never pad.
3. Add a `facts` list per explanation derived from feature values only (never
   SHAP), e.g. "Meets experience requirement (1.00)", "Skill overlap 0.67
   (4 of 6 skills matched)". Always truthful.
4. New `explanation_note`: "Bars show the ML model's influence, learned from
   weak category labels, so individual features can look counterintuitive.
   Final score = <alpha> x ML score + <1-alpha> x semantic similarity."
5. Tests: in `test_9_explain.py`, over default-pool and medium-set results,
   assert there is no "below requirement" reason when experience >= 1.0, no
   "not matched" education reason when education >= 1.0, no "Strong skill
   overlap" when skill overlap < 0.4, and no "Low ..." reason when the value
   is on the favorable side. Do not loosen existing checks. In
   `test_9_unit.py` keep additivity as pass/fail and convert the sign check
   to report-only diagnostics, with a comment explaining why (single-point
   tree contributions are not guaranteed monotone).
6. Re-run unit and integration tests (output redirected to log files). Show
   the dependence table, the new reasons and facts for ranks 1, 5, 10 on the
   medium set, additivity numbers, the regression, and P@10 >= 0.8. Stop and
   wait for "continue".

### THEN: Phase C, frontend "Why" row
"Why" control per row opens a detail row (one open at a time, keyboard
accessible, works at 375 px): reason chips (green up / red down), the
`facts` list, a diverging bar chart of the 5 signed shares with friendly
names (semantic_sim "Meaning match", skill_overlap_score "Skill overlap",
experience_match_score "Experience fit", education_match_score "Education
fit", tfidf_sim "Keyword match") and real values, matched skills as green
chips, missing as muted red chips, footnote with `explanation_note`. Run
`npm run build`, then in the browser upload `test_candidates_300_medium.csv`,
search with `test_jd_medium.csv`, expand rows #1 and #10 and describe them.
No raw `&#...;` text, no console errors.

### Optional, only after Step 9 is finished
Retrain XGBoost with `monotone_constraints` (each feature may only push the
score up) so explanations are intuitive. This changes every ML score,
including the 0.7326 benchmark, so make it a separate, clearly labeled
change, compare AUC/F1 before and after, and update every number in the
report. Decide based on the dependence table.

---

## 9. Incidents and fixes already made (don't reintroduce)

1. **Skill matcher:** first too lenient (generic words matched everything ->
   1.0 for all), then too strict (2 unique values). Now: JD skills parsed
   into phrases, `GENERIC_STOPWORDS`, per-keyword partial credit (67 unique
   values). Education matcher fixed (2-letter degrees like "BA",
   `DEGREE_HIERARCHY`, word-boundary regex).
2. **JD text mismatch:** `05_rerank.py` and `06_api.py` built the JD string
   differently (scores drifted ~0.001-0.05). Fixed with one canonical
   `build_jd_text()` in `features.py`.
3. **SECRET_KEY fallback** removed; server refuses to start without it.
4. **Interrupted folder move:** venv lost `Scripts/` and many packages. Fixed
   by recreating venv scripts, installing CPU-only torch from the PyTorch CPU
   index, and iterating `pip check` until clean. `pip check` plus an import
   test is the real health check, not a folder listing.
5. **Cluster `jd_nearest_cluster`:** centroid cosine pointed at the designer
   cluster though all top-10 were IT; then an agent recomputed its own cosine
   top-10 (4 of 10 not IT). Final: majority vote over the real `/rank` top 10
   saved on the session.
6. **sklearn:** `TfidfVectorizer(stop_words=...)` must be a `list` (or
   "english"), not a set/frozenset.
7. **Model-load stall:** a Hugging Face network check delayed startup for
   ~40 minutes of agent time; solved with the offline env flags (section 1).

---

## 10. Known issues / honest limitations

1. **Skill parser and JD formatting:** `parse_jd_skills` handles commas,
   semicolons, slashes and pipes; otherwise it splits on TitleCase
   boundaries, which breaks acronyms ("SQL AWS" -> single letters, dropped).
   `test_jd_single.csv` and `test_jds.csv` use space-separated skills, so the
   skill feature probably saw only part of the list (rankings were still good,
   driven by semantic similarity). Use comma-separated skills in tests and in
   the UI; consider making the parser robust.
2. `experience_match_score` uses the max "N years" phrase anywhere in the
   resume; neutral 0.5 when unparseable. Education collapses Bachelor/Master
   into one tier when the JD wants a PhD.
3. XGBoost ML scores tie (few distinct values); semantic similarity breaks
   ties.
4. The Category label is noisy; the weak-label ceiling is real.
5. SHAP explanations reflect what the model learned from weak labels, so
   individual features can look counterintuitive (section 8).
6. Rate limit 10 req/min per IP trips fast test scripts.
7. Sessions are neither persisted nor encrypted (deliberate scope decision;
   state it in the report).
8. Offline Hugging Face mode requires a one-time model download on a new
   machine.

---

## 11. Remaining work, in priority order

1. Step 9 Part 2 (consistency fix), then Phase C (frontend "Why" row).
2. Browser-verify the whole frontend (basic flow, redesign, rank-column fix,
   cluster map, Why row). Fix what is off.
3. **Rotate secrets before any GitHub push.** `SECRET_KEY` and
   `ENCRYPTION_KEY` values were printed in terminal output during
   development. Generate new ones. To rotate the Fernet key: decrypt
   `models/resumes_clean.enc` to a CSV with the OLD key, put the new key in
   `.env`, run `src/07_encrypt_resumes.py`, then confirm no plaintext CSV
   remains. Also change the demo login password and keep it out of the
   README.
4. GitHub README. Title **TalentRank AI**; one-liner: "Two-stage candidate
   ranking system using semantic search (FAISS) and ML scoring (XGBoost) to
   match resumes against job descriptions, built with Python, FastAPI and
   Next.js." Include run steps, how to regenerate `models/` (scripts 01 ->
   07), the offline-model note, security notes (JWT, Fernet, rate limiting,
   `.env` never committed), screenshots.
5. Project report (outline below).
6. Optional ML extras, by value: XGBRanker (`rank:pairwise`) with NDCG@10 /
   P@10 on a small hand-labeled gold set (3-4 JDs scored with the
   answer-key approach); `monotone_constraints` retrain (section 8);
   embedding-based skill matching; cross-validation and model comparison;
   "find similar candidates" via FAISS.
7. Housekeeping: delete root `compare.py` and `test_api.py` and any `_*.py`
   helpers; ensure no plaintext `models/resumes_clean.csv`; regenerate
   `requirements_verified.txt`.

### Report outline (material already collected)
1. Problem and architecture (two-stage retrieval, CPU-friendly).
2. Data: both datasets, PII masking, Category-vs-job-function finding,
   synthetic JD data, BPO scarcity.
3. Methods: embeddings, FAISS, 5 features, weak-label construction (and why
   `Category` is excluded from features), XGBoost vs RF, rerank formula and
   alpha sweep, clustering, TreeSHAP.
4. Results: model metrics (weak labels), qualitative cross-category finds
   (construction worker -> web dev, UX designer, Java intern tagged
   AVIATION), Teacher/Nurse/IT-only edge cases, medium-set P@10 / NDCG@10
   (report both the 0.9/0.96 UI run and the 1.0/1.0 test run), cluster map,
   SHAP explanations and their caveats.
5. Security: PII masking before storage, Fernet at rest, JWT, rate limiting,
   fail-fast secrets, no resume text in responses.
6. Engineering rigor: bugs found and fixed (section 9), reproducibility
   (`requirements_verified.txt`, regression anchor), environment recovery.
7. Limitations (sections 2 and 10) and future work (real labels, XGBRanker,
   gold evaluation set, better skill extraction, monotone constraints).

---

## 12. First actions for the new assistant

1. Read this file; confirm you understand section 0.
2. Read-only audit: does `src/explain.py` already contain a `facts` list and
   a consistency gate? Does `tests/diag_shap_dependence.py` exist? Report;
   do not edit yet.
3. If the Part 2 fix is not done, run it (section 8, Part 2), showing log
   output, and stop for the user's "continue". Do not start the frontend
   before that.