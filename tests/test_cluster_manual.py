"""
src/test_cluster.py
-------------------
Step 8b integration test:
  1. Login
  2. Upload 1000-row CSV  -> poll until done
  3. POST /rank (Web Developer JD) -> seeds last_jd_embedding
  4. GET /cluster-map/{session_id} -> print truncated JSON + cluster labels
  5. Upload 10-row slice  -> expect 400 from /cluster-map
"""

import sys, time, json, io, textwrap
from pathlib import Path
import requests
import pandas as pd

BASE = "http://127.0.0.1:8000"
CSV_PATH = Path(__file__).resolve().parent.parent / "TestingData" / "test_candidates_1000_easy.csv"

# ── helpers ───────────────────────────────────────────────────────────────────
def login():
    r = requests.post(f"{BASE}/token", data={"username": "recruiter", "password": "securepassword123"})
    r.raise_for_status()
    return r.json()["access_token"]

def upload(token, csv_bytes, filename="test.csv"):
    r = requests.post(
        f"{BASE}/upload-resumes",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, csv_bytes, "text/csv")},
    )
    r.raise_for_status()
    return r.json()

def poll(token, sid, interval=3.0, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(
            f"{BASE}/upload-status/{sid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        s = r.json()
        print(f"  status={s['status']}  {s['processed']}/{s['total']}", flush=True)
        if s["status"] == "done":
            return
        if s["status"] == "error":
            raise RuntimeError("Server-side processing failed.")
        time.sleep(interval)
    raise TimeoutError("Upload did not finish in time.")

def rank(token, sid):
    payload = {
        "jd_text": "Looking for a Web Developer with frontend and backend experience.",
        "job_title": "Web Developer",
        "skills": "HTML, CSS, JavaScript, Node.js",
        "experience": "3 years",
        "qualifications": "Bachelors",
        "session_id": sid,
    }
    r = requests.post(
        f"{BASE}/rank",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
    )
    r.raise_for_status()
    return r.json()

def cluster_map(token, sid):
    r = requests.get(
        f"{BASE}/cluster-map/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    return r

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Step 8b: /cluster-map integration test ===\n")

    # 1. Login
    print("[1] Logging in...")
    token = login()
    print("    OK\n")

    # 2. Upload 1000-row CSV
    print(f"[2] Uploading {CSV_PATH.name} ...")
    csv_bytes = CSV_PATH.read_bytes()
    up = upload(token, csv_bytes)
    sid = up["session_id"]
    print(f"    session_id={sid}  total={up['total_resumes']}")
    print("    Polling...")
    poll(token, sid)
    print("    Done.\n")

    # 3. POST /rank to seed last_jd_embedding
    print("[3] POST /rank (Web Developer JD) to seed jd_point...")
    rank_result = rank(token, sid)
    print(f"    Top result: id={rank_result['results'][0]['id']}  final_score={rank_result['results'][0]['final_score']:.4f}\n")

    # 4. GET /cluster-map
    print("[4] GET /cluster-map ...")
    resp = cluster_map(token, sid)
    if resp.status_code != 200:
        print(f"    FAILED {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"\n    k={data['k']}  silhouette={data['silhouette']}")
    if "adjusted_rand_index" in data:
        print(f"    adjusted_rand_index={data['adjusted_rand_index']}  normalized_mutual_info={data['normalized_mutual_info']}")
    if "jd_point" in data:
        print(f"    jd_point: x={data['jd_point']['x']:.4f}  y={data['jd_point']['y']:.4f}")

    print("\n    Cluster labels:")
    for c in data["clusters"]:
        print(f"      cluster {c['cluster']}  size={c['size']}  keywords={c['keywords']}")

    print(f"\n    Points (first 5 of {len(data['points'])}):")
    for pt in data["points"][:5]:
        cat = pt.get("category", "—")
        print(f"      id={pt['id']}  cluster={pt['cluster']}  x={pt['x']:.3f}  y={pt['y']:.3f}  category={cat}")

    # Truncated JSON dump
    preview = dict(data)
    preview["points"] = data["points"][:3]
    preview["_points_truncated"] = f"... {len(data['points']) - 3} more"
    print("\n    Truncated JSON response:")
    print(textwrap.indent(json.dumps(preview, indent=2), "    "))

    # 5. Test 400 for tiny upload
    print("\n[5] Testing 400 for 10-row upload...")
    df_full = pd.read_csv(io.BytesIO(csv_bytes))
    tiny_buf = io.BytesIO()
    df_full.head(10).to_csv(tiny_buf, index=False)
    tiny_buf.seek(0)
    up2 = upload(token, tiny_buf.read(), filename="tiny.csv")
    sid2 = up2["session_id"]
    print(f"    session_id={sid2}  total={up2['total_resumes']}")
    print("    Polling tiny upload...")
    poll(token, sid2, interval=2.0)
    resp2 = cluster_map(token, sid2)
    print(f"    HTTP {resp2.status_code}: {resp2.json()['detail']}")
    assert resp2.status_code == 400, f"Expected 400, got {resp2.status_code}"
    print("    400 confirmed. ✓\n")

    print("=== All checks passed ===")

if __name__ == "__main__":
    main()
