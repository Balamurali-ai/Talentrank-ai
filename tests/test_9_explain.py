"""
tests/test_9_explain.py
-----------------------
Step 9: TreeSHAP explainability test suite.

Tests:
  1. Regression & Additivity on default pool with benchmark Web Developer JD:
     - Top result is 91153752 with final_score ~ 0.7326
     - All 10 have explanations with 5 features + bias + reasons + facts + skills
     - Additivity: sigmoid(sum(contribs) + bias) == ML_score within 1e-4 for all 10
     - Explanation note is present
     - Reason consistency: no reason contradicts its feature value
  2. Upload & Rank on Medium dataset:
     - Upload test_candidates_300_medium.csv (300 candidates)
     - Rank with test_jd_medium.csv
     - Additivity holds for all 10 medium results
     - Reason consistency holds for all 10 medium results
     - Precision@10: At least 8 of top 10 have relevance 2 in test_medium_answer_key.csv
     - Compute and print P@10 and NDCG@10 from the key
     - Print reasons, facts, matched/missing skills for ranks 1, 5, and 10
  3. String length constraint:
     - No explanation string field is longer than 80 characters
"""

import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

BASE    = "http://127.0.0.1:8000"
ROOT    = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"

CSV_MEDIUM = ROOT / "TestingData" / "test_candidates_300_medium.csv"
JD_MEDIUM  = ROOT / "TestingData" / "test_jd_medium.csv"
KEY_MEDIUM = ROOT / "TestingData" / "test_medium_answer_key.csv"

BENCHMARK_JD = {
    "jd_text": (
        "Frontend Web Developers design and implement user interfaces for websites, "
        "ensuring they are visually appealing and user-friendly. They collaborate with "
        "designers and backend developers to create seamless web experiences for users."
    ),
    "job_title":       "Web Developer",
    "role":            "Frontend Web Developer",
    "skills":          "HTML, CSS, JavaScript Frontend frameworks (e.g., React, Angular) User experience (UX)",
    "experience":      "2 to 12 Years",
    "qualifications":  "BCA",
    "responsibilities": (
        "Design and code user interfaces for websites, ensuring a seamless and visually "
        "appealing user experience. Collaborate with UX designers to optimize user journeys. "
        "Ensure cross-browser compatibility and responsive design."
    ),
}


# ── Server lifecycle ───────────────────────────────────────────────────────────

def start_server():
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "06_api:app",
         "--app-dir", str(ROOT / "src"), "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT),
        env=env,
    )
    print(f"[server] PID {proc.pid} -- waiting for startup...", flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200 and r.json().get("models_loaded"):
                print("[server] Up and models loaded.\n", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(3)
    proc.terminate()
    out, _ = proc.communicate(timeout=10)
    print(out.decode(errors="replace"))
    raise RuntimeError("Server did not start within 120 s.")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("\n[server] Stopped.")


def login():
    r = requests.post(f"{BASE}/token",
                      data={"username": "recruiter", "password": "securepassword123"})
    r.raise_for_status()
    return r.json()["access_token"]


def upload_and_wait(token, csv_path):
    csv_bytes = csv_path.read_bytes()
    r = requests.post(
        f"{BASE}/upload-resumes",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (csv_path.name, csv_bytes, "text/csv")},
    )
    r.raise_for_status()
    sid   = r.json()["session_id"]
    total = r.json()["total_resumes"]
    print(f"  session_id={sid}  total={total}", flush=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        s = requests.get(f"{BASE}/upload-status/{sid}",
                         headers={"Authorization": f"Bearer {token}"}).json()
        print(f"    poll: {s['status']}  {s['processed']}/{s['total']}", flush=True)
        if s["status"] == "done":
            return sid
        if s["status"] == "error":
            raise RuntimeError("Upload processing failed on server.")
        time.sleep(3)
    raise TimeoutError("Upload did not finish in time.")


def do_rank(token, payload, alpha=0.5):
    r = requests.post(
        f"{BASE}/rank?alpha={alpha}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
    )
    r.raise_for_status()
    return r.json()


# ── Assertion helpers ──────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def check_additivity(result, tol=1e-4):
    expl        = result["explanation"]
    contrib_sum = sum(f["contribution"] for f in expl["features"])
    bias        = expl["bias"]
    computed    = sigmoid(contrib_sum + bias)
    ml_score    = result["ML_score"]
    diff        = abs(computed - ml_score)
    assert diff < tol, (
        f"Additivity FAIL for id={result['id']}: "
        f"sigmoid({contrib_sum:.4f} + {bias:.4f}) = {computed:.6f} "
        f"!= ML_score {ml_score:.6f}  (diff={diff:.2e})"
    )
    return computed, diff


def check_no_resume_text(results):
    for r in results:
        expl = r.get("explanation") or {}
        for s in expl.get("reasons", []):
            assert len(s) <= 80, f"Reason too long ({len(s)} chars): {s!r}"
        for s in expl.get("facts", []):
            assert len(s) <= 80, f"Fact too long ({len(s)} chars): {s!r}"
        for s in expl.get("matched_skills", []):
            assert len(s) <= 80, f"matched_skill too long ({len(s)} chars): {s!r}"
        for s in expl.get("missing_skills", []):
            assert len(s) <= 80, f"missing_skill too long ({len(s)} chars): {s!r}"
        for feat in expl.get("features", []):
            assert len(feat["name"]) <= 80, \
                f"feature name too long ({len(feat['name'])} chars): {feat['name']!r}"


def check_reason_consistency(results, label=""):
    """Assert no reason contradicts its feature value."""
    for r in results:
        expl     = r.get("explanation") or {}
        feat_map = {f["name"]: f["value"] for f in expl.get("features", [])}
        reasons  = expl.get("reasons", [])
        rid      = r["id"]

        exp_val = feat_map.get("experience_match_score", 0.5)
        for reason in reasons:
            assert "below requirement" not in reason or exp_val < 1.0, (
                f"{label} id={rid}: 'below requirement' reason but "
                f"experience_match_score={exp_val:.2f} >= 1.0. Reason: {reason!r}"
            )

        edu_val = feat_map.get("education_match_score", 0.5)
        for reason in reasons:
            assert "not matched" not in reason or edu_val < 1.0, (
                f"{label} id={rid}: 'not matched' education reason but "
                f"education_match_score={edu_val:.2f} >= 1.0. Reason: {reason!r}"
            )

        skill_val = feat_map.get("skill_overlap_score", 0.0)
        for reason in reasons:
            assert "Strong skill overlap" not in reason or skill_val >= 0.4, (
                f"{label} id={rid}: 'Strong skill overlap' reason but "
                f"skill_overlap_score={skill_val:.2f} < 0.4. Reason: {reason!r}"
            )

        sem_val   = feat_map.get("semantic_sim", 0.0)
        tfidf_val = feat_map.get("tfidf_sim", 0.0)
        for reason in reasons:
            if "Low semantic similarity" in reason:
                assert sem_val < 0.5, (
                    f"{label} id={rid}: 'Low semantic similarity' reason but "
                    f"semantic_sim={sem_val:.2f} >= 0.5. Reason: {reason!r}"
                )
            if "Low keyword similarity" in reason:
                assert tfidf_val < 0.1, (
                    f"{label} id={rid}: 'Low keyword similarity' reason but "
                    f"tfidf_sim={tfidf_val:.2f} >= 0.1. Reason: {reason!r}"
                )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    proc = start_server()
    try:
        token = login()
        print("[auth] Login OK\n")

        # ======================================================================
        # TEST 1: Default pool -- Web Developer benchmark + additivity
        # ======================================================================
        print("=" * 70)
        print("TEST 1: Default pool -- Web Developer benchmark & TreeSHAP additivity")
        print("=" * 70)

        resp             = do_rank(token, BENCHMARK_JD, alpha=0.5)
        results          = resp["results"]
        explanation_note = resp.get("explanation_note", "")

        print(f"Top-level explanation_note:\n  '{explanation_note}'\n")
        assert explanation_note, "Missing explanation_note in rank response"

        top = results[0]
        print(f"Top result: id={top['id']}  category={top['category']}  final_score={top['final_score']:.4f}")
        assert str(top["id"]) == "91153752", f"FAIL expected top id 91153752, got {top['id']}"
        assert abs(top["final_score"] - 0.7326) < 0.0005, \
            f"FAIL final_score {top['final_score']:.4f} != 0.7326"
        print("Regression check PASS: id=91153752, final_score=0.7326\n")

        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        for r in results:
            assert r.get("explanation") is not None, f"Missing explanation for id={r['id']}"
            assert len(r["explanation"]["features"]) == 5, \
                f"Expected 5 features for id={r['id']}"
            assert "bias"          in r["explanation"], f"Missing bias for id={r['id']}"
            assert "reasons"       in r["explanation"], f"Missing reasons for id={r['id']}"
            assert "facts"         in r["explanation"], f"Missing facts for id={r['id']}"
            assert "matched_skills" in r["explanation"], f"Missing matched_skills for id={r['id']}"
            assert "missing_skills" in r["explanation"], f"Missing missing_skills for id={r['id']}"
        print("Explanations structure PASS: all 10 results have valid explanations\n")

        print("Additivity check (sigmoid(sum(contribs) + bias) vs ML_score):")
        print(f"  {'id':<12} {'ML_score':>10} {'sigmoid':>10} {'abs_diff':>12}")
        print("  " + "-" * 48)
        for r in results:
            computed, diff = check_additivity(r)
            print(f"  {r['id']:<12} {r['ML_score']:>10.6f} {computed:>10.6f} {diff:>12.2e}")
        print("Additivity check PASS: all within 1e-4 tolerance\n")

        check_no_resume_text(results)
        print("String length check PASS: no explanation field > 80 chars\n")

        check_reason_consistency(results, label="[default pool]")
        print("Reason consistency check PASS: no reason contradicts its feature value\n")

        # ======================================================================
        # TEST 2: Medium set -- Upload + Rank + P@10 & NDCG@10 from Answer Key
        # ======================================================================
        print("=" * 70)
        print("TEST 2: Medium set (300 candidates) -- Upload, Rank & Evaluation")
        print("=" * 70)

        df_jd   = pd.read_csv(JD_MEDIUM)
        jd_row  = df_jd.iloc[0]
        jd_base = {
            "jd_text":          str(jd_row.get("Job Description", "")),
            "job_title":        str(jd_row.get("Job Title", "")),
            "role":             str(jd_row.get("Role", "")),
            "skills":           str(jd_row.get("skills", "")),
            "experience":       str(jd_row.get("Experience", "")),
            "qualifications":   str(jd_row.get("Qualifications", "")),
            "responsibilities": str(jd_row.get("Responsibilities", "")),
        }
        print(f"JD Title:  {jd_base['job_title']} ({jd_base['role']})")
        print(f"JD Skills: {jd_base['skills']}\n")

        print("Uploading test_candidates_300_medium.csv...")
        sid        = upload_and_wait(token, CSV_MEDIUM)
        resp_med   = do_rank(token, {**jd_base, "session_id": sid}, alpha=0.5)
        results_med = resp_med["results"]
        assert len(results_med) == 10, f"Expected 10 results, got {len(results_med)}"

        for r in results_med:
            check_additivity(r)
        print("Medium set additivity check PASS: all within 1e-4 tolerance\n")

        check_no_resume_text(results_med)
        print("Medium set string length check PASS: no explanation field > 80 chars\n")

        check_reason_consistency(results_med, label="[medium set]")
        print("Medium set reason consistency check PASS\n")

        # Evaluate against answer key
        df_key  = pd.read_csv(KEY_MEDIUM)
        key_map = {int(row["ID"]): row for _, row in df_key.iterrows()}

        top10_ids  = [int(r["id"]) for r in results_med]
        relevances = [key_map[cid]["relevance"] for cid in top10_ids]

        rel_2_count = sum(1 for rel in relevances if rel == 2)
        p_at_10     = rel_2_count / 10.0

        dcg_10           = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))
        all_sorted_rels  = sorted(df_key["relevance"].tolist(), reverse=True)[:10]
        idcg_10          = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(all_sorted_rels))
        ndcg_10          = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

        print("-" * 70)
        print(f"{'Rank':<5} {'ID':<10} {'Final':<8} {'Rel':<5} {'Category':<22} Profile")
        print("-" * 70)
        for i, r in enumerate(results_med):
            cid  = int(r["id"])
            rel  = key_map[cid]["relevance"]
            prof = key_map[cid]["profile"]
            print(f"{i+1:<5} {cid:<10} {r['final_score']:<8.4f} {rel:<5} {r['category']:<22} {prof}")
        print("-" * 70)

        print(f"\nEvaluation Metrics from Answer Key:")
        print(f"  Relevance 2 count in top 10: {rel_2_count}/10")
        print(f"  Precision@10 (relevance=2):  {p_at_10:.4f} ({p_at_10*100:.1f}%)")
        print(f"  NDCG@10:                     {ndcg_10:.4f}")

        assert rel_2_count >= 8, \
            f"Expected at least 8/10 relevance=2 candidates, got {rel_2_count}/10"
        print(f"Answer key relevance check PASS: {rel_2_count}/10 have relevance 2 (>= 8 required)\n")

        # Print explanation details for ranks 1, 5, 10
        print("=" * 70)
        print("EXPLANATION DETAILS FOR RANKS 1, 5, AND 10")
        print("=" * 70)
        for rank in [1, 5, 10]:
            cand = results_med[rank - 1]
            expl = cand["explanation"]
            cid  = int(cand["id"])
            print(f"\n--- Rank #{rank}: ID {cid} (Category: {cand['category']}) ---")
            print(f"  Final Score:     {cand['final_score']:.4f}")
            print(f"  Semantic Sim:    {cand['semantic_sim']:.4f}")
            print(f"  ML Score:        {cand['ML_score']:.4f}")
            print(f"  Bias (log-odds): {expl['bias']:.4f}")
            print(f"  Reasons ({len(expl['reasons'])}):")
            for s in expl["reasons"]:
                print(f"    + {s}")
            print(f"  Facts ({len(expl.get('facts', []))}):")
            for s in expl.get("facts", []):
                print(f"    * {s}")
            print(f"  Matched Skills ({len(expl['matched_skills'])}): {expl['matched_skills']}")
            print(f"  Missing Skills ({len(expl['missing_skills'])}): {expl['missing_skills']}")
            print(f"  Features breakdown:")
            for f in expl["features"]:
                sign = "+" if f["contribution"] >= 0 else "-"
                print(
                    f"    {f['name']:<24} value={f['value']:.4f}  "
                    f"contrib={f['contribution']:+.4f}  share={f['share']:+.4f} [{sign}]"
                )

        print("\n" + "=" * 70)
        print("ALL STEP 9 EXPLAINABILITY TESTS PASSED SUCCESSFULLY")
        print("=" * 70)

    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
