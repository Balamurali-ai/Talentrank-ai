"""
tests/test_phase1_checkpoint.py
--------------------------------
Phase 1 checkpoint tests:
  1. Upload test_candidates_1000_easy.csv + rank with test_jd_single.csv JD
     -> all top-10 must be INFORMATION-TECHNOLOGY
  2. Default pool + Web Developer benchmark JD
     -> top result id=91153752, final_score~0.7326
  3. CORS headers present on /token and /rank
"""

import subprocess, time, io, json
from pathlib import Path
import requests
import pandas as pd

BASE     = "http://127.0.0.1:8000"
ROOT     = Path(__file__).resolve().parent.parent
VENV_PY  = ROOT / "venv" / "Scripts" / "python.exe"
CSV_1000 = ROOT / "TestingData" / "test_candidates_1000_easy.csv"
JD_CSV   = ROOT / "TestingData" / "test_jd_single.csv"

BENCHMARK_JD = {
    "jd_text": "Frontend Web Developers design and implement user interfaces for websites, ensuring they are visually appealing and user-friendly. They collaborate with designers and backend developers to create seamless web experiences for users.",
    "job_title": "Web Developer",
    "role": "Frontend Web Developer",
    "skills": "HTML, CSS, JavaScript Frontend frameworks (e.g., React, Angular) User experience (UX)",
    "experience": "2 to 12 Years",
    "qualifications": "BCA",
    "responsibilities": "Design and code user interfaces for websites, ensuring a seamless and visually appealing user experience. Collaborate with UX designers to optimize user journeys. Ensure cross-browser compatibility and responsive design.",
}

def start_server():
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "06_api:app", "--app-dir", str(ROOT / "src"), "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT),
    )
    print(f"[server] PID {proc.pid} — waiting...", flush=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200 and r.json().get("models_loaded"):
                print("[server] Up.\n"); return proc
        except Exception: pass
        time.sleep(3)
    proc.terminate(); raise RuntimeError("Server did not start.")

def stop_server(proc):
    proc.terminate()
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired: proc.kill()
    print("\n[server] Stopped.")

def login():
    r = requests.post(f"{BASE}/token", data={"username": "recruiter", "password": "securepassword123"})
    r.raise_for_status()
    return r.json()["access_token"]

def upload_and_wait(token, csv_bytes, filename="test.csv"):
    r = requests.post(f"{BASE}/upload-resumes",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, csv_bytes, "text/csv")})
    r.raise_for_status()
    sid = r.json()["session_id"]
    total = r.json()["total_resumes"]
    print(f"  session_id={sid}  total={total}")
    deadline = time.time() + 300
    while time.time() < deadline:
        s = requests.get(f"{BASE}/upload-status/{sid}",
            headers={"Authorization": f"Bearer {token}"}).json()
        print(f"  poll: {s['status']}  {s['processed']}/{s['total']}", flush=True)
        if s["status"] == "done": return sid
        if s["status"] == "error": raise RuntimeError("Upload failed.")
        time.sleep(3)
    raise TimeoutError()

def do_rank(token, payload):
    r = requests.post(f"{BASE}/rank?alpha=0.5",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload)
    r.raise_for_status()
    return r.json()["results"]

def main():
    proc = start_server()
    try:
        token = login()
        print("[auth] Login OK\n")

        # ── CHECKPOINT 1: Upload 1000-row CSV + rank with test_jd_single.csv ──
        print("=" * 60)
        print("CHECKPOINT 1: Upload flow + IT JD")

        jd_row = pd.read_csv(str(JD_CSV)).iloc[0]
        jd_payload = {
            "jd_text":        str(jd_row.get("Job Description", "")),
            "job_title":      str(jd_row.get("Job Title", "")),
            "role":           str(jd_row.get("Role", "")),
            "skills":         str(jd_row.get("skills", "")),
            "experience":     str(jd_row.get("Experience", "")),
            "qualifications": str(jd_row.get("Qualifications", "")),
            "responsibilities": str(jd_row.get("Responsibilities", "")),
        }
        print(f"  JD: {jd_payload['job_title']} / {jd_payload['role']}")

        sid = upload_and_wait(token, CSV_1000.read_bytes(), "test_1000.csv")
        jd_payload["session_id"] = sid
        results = do_rank(token, jd_payload)

        print(f"\n  Top-10 results:")
        for i, r in enumerate(results):
            print(f"    {i+1:2}. id={r['id']}  category={r['category']}  final_score={r['final_score']:.4f}")

        categories = [r["category"] for r in results]
        it_count = sum(1 for c in categories if c == "INFORMATION-TECHNOLOGY")
        print(f"\n  INFORMATION-TECHNOLOGY in top-10: {it_count}/10")
        assert it_count == 10, f"Expected all 10 to be IT, got {it_count}"
        print("  PASS OK\n")

        # ── CHECKPOINT 2: Default pool + Web Developer benchmark JD ───────────
        print("=" * 60)
        print("CHECKPOINT 2: Default pool + Web Developer benchmark JD")
        results2 = do_rank(token, BENCHMARK_JD)
        top = results2[0]
        print(f"  Top result: id={top['id']}  final_score={top['final_score']:.4f}")
        assert str(top["id"]) == "91153752", f"FAIL id={top['id']}"
        assert abs(top["final_score"] - 0.7326) < 0.0005, f"FAIL score={top['final_score']}"
        print("  id=91153752  final_score=0.7326  PASS OK\n")

        # ── CHECKPOINT 3: CORS headers ────────────────────────────────────────
        print("=" * 60)
        print("CHECKPOINT 3: CORS headers (simulating browser preflight from localhost:3000)")

        # OPTIONS preflight on /token
        r_opt = requests.options(f"{BASE}/token", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        acao = r_opt.headers.get("access-control-allow-origin", "")
        acac = r_opt.headers.get("access-control-allow-credentials", "")
        print(f"  OPTIONS /token -> {r_opt.status_code}")
        print(f"  Access-Control-Allow-Origin:      {acao}")
        print(f"  Access-Control-Allow-Credentials: {acac}")
        assert acao == "http://localhost:3000", f"CORS origin wrong: {acao}"
        assert acac.lower() == "true", f"CORS credentials wrong: {acac}"

        # Simple GET with Origin header (simulates browser fetch)
        r_get = requests.get(f"{BASE}/health", headers={"Origin": "http://localhost:3000"})
        acao2 = r_get.headers.get("access-control-allow-origin", "")
        print(f"\n  GET /health with Origin -> {r_get.status_code}")
        print(f"  Access-Control-Allow-Origin: {acao2}")
        assert acao2 == "http://localhost:3000", f"CORS origin on GET wrong: {acao2}"
        print("  CORS PASS OK\n")

        print("=" * 60)
        print("ALL CHECKPOINT TESTS PASSED OK")

    finally:
        stop_server(proc)

if __name__ == "__main__":
    main()
