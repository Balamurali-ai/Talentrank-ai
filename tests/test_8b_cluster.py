"""
src/run_8b_tests.py
-------------------
Orchestrates Step 8b testing:
  - Starts the API server as a subprocess
  - Waits for /health
  - Runs all cluster-map checks
  - Runs regression test (default pool /rank)
  - Stops the server
"""

import subprocess, sys, time, json, io, textwrap, os
from pathlib import Path
import requests
import pandas as pd

BASE      = "http://127.0.0.1:8000"
ROOT      = Path(__file__).resolve().parent.parent
VENV_PY   = ROOT / "venv" / "Scripts" / "python.exe"
CSV_1000  = ROOT / "TestingData" / "test_candidates_1000_easy.csv"

# ── server lifecycle ──────────────────────────────────────────────────────────
def start_server():
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "06_api:app", "--app-dir", str(ROOT / "src"), "--port", "8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    print(f"[server] PID {proc.pid} — waiting for startup...", flush=True)
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
    raise RuntimeError("Server did not start in time.")

def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("\n[server] Stopped.")

# ── API helpers ───────────────────────────────────────────────────────────────
def login():
    r = requests.post(f"{BASE}/token", data={"username": "recruiter", "password": "securepassword123"})
    r.raise_for_status()
    return r.json()["access_token"]

def upload_csv(token, csv_bytes, filename="test.csv"):
    r = requests.post(
        f"{BASE}/upload-resumes",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, csv_bytes, "text/csv")},
    )
    r.raise_for_status()
    return r.json()

def poll_until_done(token, sid, interval=3.0, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE}/upload-status/{sid}", headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        s = r.json()
        print(f"    poll: status={s['status']}  {s['processed']}/{s['total']}", flush=True)
        if s["status"] == "done":
            return
        if s["status"] == "error":
            raise RuntimeError("Server-side processing failed.")
        time.sleep(interval)
    raise TimeoutError("Upload did not finish in time.")

def do_rank(token, sid=None):
    payload = {
        "jd_text": "Frontend Web Developers design and implement user interfaces for websites, ensuring they are visually appealing and user-friendly. They collaborate with designers and backend developers to create seamless web experiences for users.",
        "job_title": "Web Developer",
        "role": "Frontend Web Developer",
        "skills": "HTML, CSS, JavaScript Frontend frameworks (e.g., React, Angular) User experience (UX)",
        "experience": "2 to 12 Years",
        "qualifications": "BCA",
        "responsibilities": "Design and code user interfaces for websites, ensuring a seamless and visually appealing user experience. Collaborate with UX designers to optimize user journeys. Ensure cross-browser compatibility and responsive design.",
    }
    if sid:
        payload["session_id"] = sid
    r = requests.post(
        f"{BASE}/rank",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
    )
    r.raise_for_status()
    return r.json()

def get_cluster_map(token, sid):
    return requests.get(f"{BASE}/cluster-map/{sid}", headers={"Authorization": f"Bearer {token}"})

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    proc = start_server()
    try:
        token = login()
        print("[auth] Login OK\n")

        # ── TEST 1: 404 for nonexistent session ───────────────────────────────
        print("=" * 60)
        print("TEST 1: 404 for nonexistent session_id")
        r404 = get_cluster_map(token, "00000000-0000-0000-0000-000000000000")
        assert r404.status_code == 404, f"Expected 404, got {r404.status_code}"
        print(f"  HTTP {r404.status_code}: {r404.json()['detail']}")
        print("  PASS OK\n")

        # ── TEST 2: 400 for 10-row upload ─────────────────────────────────────
        print("=" * 60)
        print("TEST 2: 400 for 10-row upload (too small to cluster)")
        df_full = pd.read_csv(str(CSV_1000))
        tiny_buf = io.BytesIO()
        df_full.head(10).to_csv(tiny_buf, index=False)
        tiny_buf.seek(0)
        up_tiny = upload_csv(token, tiny_buf.read(), "tiny.csv")
        sid_tiny = up_tiny["session_id"]
        print(f"  session_id={sid_tiny}  total={up_tiny['total_resumes']}")
        poll_until_done(token, sid_tiny, interval=2.0)
        r400 = get_cluster_map(token, sid_tiny)
        assert r400.status_code == 400, f"Expected 400, got {r400.status_code}"
        print(f"  HTTP {r400.status_code}: {r400.json()['detail']}")
        print("  PASS OK\n")

        # ── TEST 3: Main 1000-row cluster-map ────────────────────────────────
        print("=" * 60)
        print("TEST 3: Upload 1000-row CSV and call /cluster-map")
        csv_bytes = CSV_1000.read_bytes()
        up = upload_csv(token, csv_bytes, "test_1000.csv")
        sid = up["session_id"]
        print(f"  session_id={sid}  total={up['total_resumes']}")
        poll_until_done(token, sid)

        # Call /cluster-map BEFORE /rank — jd_point, jd_nearest_cluster, top10_ids absent
        r_pre = get_cluster_map(token, sid)
        assert r_pre.status_code == 200
        pre_data = r_pre.json()
        assert "jd_point"           not in pre_data, "jd_point should be absent before /rank"
        assert "jd_nearest_cluster" not in pre_data, "jd_nearest_cluster should be absent before /rank"
        assert "top10_ids"          not in pre_data, "top10_ids should be absent before /rank"
        print("  /cluster-map before /rank: jd_point/jd_nearest_cluster/top10_ids absent OK")

        # ── TEST 4: /rank seeds last_top10 ───────────────────────────────────
        print("\n  POST /rank to seed last_top10...")
        rank_result = do_rank(token, sid)
        top = rank_result["results"][0]
        rank_ids = [r["id"] for r in rank_result["results"]]
        print(f"  Top result: id={top['id']}  final_score={top['final_score']:.4f}")
        print(f"  /rank top-10 IDs: {rank_ids}")

        # ── TEST 5: /cluster-map after /rank — jd_point present ──────────────
        r_post = get_cluster_map(token, sid)
        assert r_post.status_code == 200, f"Expected 200, got {r_post.status_code}: {r_post.text}"
        data = r_post.json()

        print(f"\n  k={data['k']}  silhouette={data['silhouette']}")
        if "adjusted_rand_index" in data:
            print(f"  adjusted_rand_index={data['adjusted_rand_index']}  normalized_mutual_info={data['normalized_mutual_info']}")
        assert "jd_point" in data, "jd_point missing after /rank"
        print(f"  jd_point: x={data['jd_point']['x']:.4f}  y={data['jd_point']['y']:.4f}  OK")

        assert "jd_nearest_cluster" in data, "jd_nearest_cluster missing"
        jnc = data["jd_nearest_cluster"]
        print(f"  jd_nearest_cluster: cluster={jnc['cluster']}  top10_in_cluster={jnc['top10_in_cluster']}  centroid_similarity={jnc['centroid_similarity']}  OK")

        assert "top10_ids" in data, "top10_ids missing"
        assert len(data["top10_ids"]) == 10, f"Expected 10 top10_ids, got {len(data['top10_ids'])}"
        assert data["top10_ids"] == rank_ids, (
            f"top10_ids mismatch.\n  cluster-map: {data['top10_ids']}\n  /rank:       {rank_ids}"
        )
        print(f"  top10_ids match /rank order exactly  OK")

        assert "pca_variance_explained" in data, "pca_variance_explained missing"
        print(f"  pca_variance_explained={data['pca_variance_explained']}  OK")

        # Find the cluster whose keywords include 'python'
        python_cluster = next(
            (cl["cluster"] for cl in data["clusters"] if "python" in cl["keywords"]), None
        )
        assert python_cluster is not None, "No cluster with 'python' in keywords"

        # Build id->cluster map from points
        id_to_cluster = {str(pt["id"]): pt["cluster"] for pt in data["points"]}
        top10_in_python = sum(1 for rid in data["top10_ids"] if id_to_cluster.get(rid) == python_cluster)
        print(f"  python cluster={python_cluster}  top10 resumes in it={top10_in_python}/10")
        assert top10_in_python >= 8, f"Expected >= 8 top-10 in python cluster, got {top10_in_python}"
        assert jnc["cluster"] == python_cluster, (
            f"jd_nearest_cluster={jnc['cluster']} but python cluster={python_cluster}"
        )
        assert jnc["top10_in_cluster"] >= 8, f"top10_in_cluster={jnc['top10_in_cluster']} < 8"
        print(f"  jd_nearest_cluster == python cluster  OK")

        print("\n  Cluster labels (with domain stopwords removed):")
        for c in data["clusters"]:
            print(f"    cluster {c['cluster']}  size={c['size']:>4}  keywords={c['keywords']}")

        print(f"\n  Points sample (first 5 of {len(data['points'])}):")
        for pt in data["points"][:5]:
            cat = pt.get("category", "—")
            print(f"    id={pt['id']}  cluster={pt['cluster']}  x={pt['x']:>8.4f}  y={pt['y']:>8.4f}  category={cat}")

        # Verify no resume text in response
        resp_str = json.dumps(data)
        assert "Resume_clean" not in resp_str
        assert "Resume_str"   not in resp_str
        print("\n  No resume text in response OK")

        # Truncated JSON
        preview = {k: v for k, v in data.items() if k not in ("points", "top10_ids")}
        preview["top10_ids"] = data["top10_ids"]
        preview["points"] = data["points"][:3]
        preview["_points_note"] = f"... {len(data['points']) - 3} more points omitted"
        print("\n  Truncated JSON response:")
        print(textwrap.indent(json.dumps(preview, indent=2), "  "))
        print("  PASS OK\n")

        # ── TEST 6: Regression — default pool /rank ───────────────────────────
        print("=" * 60)
        print("TEST 6: Regression — default pool /rank")
        reg = do_rank(token)
        top_reg = reg["results"][0]
        print(f"  Top result: id={top_reg['id']}  final_score={top_reg['final_score']:.4f}")
        assert str(top_reg["id"]) == "91153752",    f"Regression FAIL: id={top_reg['id']}"
        assert abs(top_reg["final_score"] - 0.7326) < 0.0005, f"Regression FAIL: score={top_reg['final_score']}"
        print("  id=91153752  final_score=0.7326  PASS OK\n")

        print("=" * 60)
        print("ALL TESTS PASSED OK")

    finally:
        stop_server(proc)

if __name__ == "__main__":
    main()
