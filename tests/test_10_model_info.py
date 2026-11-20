"""
tests/test_10_model_info.py
---------------------------
Step 10 Part B: /model-info endpoint tests.

Tests:
  1. 401 without a token
  2. 200 with a valid token
  3. Required top-level keys present (algorithm, f1, roc_auc, features, ranker_eval)
  4. ranker_eval required keys present
  5. Regression anchor: /rank still returns id=91153752, final_score~0.7326
  6. No string value longer than 400 characters in the /model-info response
"""

import os
import subprocess
import time
from pathlib import Path

import requests

BASE    = "http://127.0.0.1:8000"
ROOT    = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"

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


def check_no_long_strings(obj, path="", max_len=400):
    """Recursively assert no string value exceeds max_len characters."""
    if isinstance(obj, str):
        assert len(obj) <= max_len, (
            f"String at '{path}' is {len(obj)} chars (max {max_len}): {obj[:80]!r}..."
        )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_long_strings(v, f"{path}.{k}", max_len)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_no_long_strings(v, f"{path}[{i}]", max_len)


def main():
    proc = start_server()
    try:
        # ── TEST 1: 401 without token ──────────────────────────────────────────
        print("=" * 60)
        print("TEST 1: 401 without token")
        print("=" * 60)
        r = requests.get(f"{BASE}/model-info")
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"
        print("PASS: 401 without token\n")

        token = login()
        print("[auth] Login OK\n")

        # ── TEST 2: 200 with token ─────────────────────────────────────────────
        print("=" * 60)
        print("TEST 2: 200 with valid token")
        print("=" * 60)
        r = requests.get(f"{BASE}/model-info",
                         headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
        data = r.json()
        print("PASS: 200 with token\n")

        # ── TEST 3: Required top-level keys ────────────────────────────────────
        print("=" * 60)
        print("TEST 3: Required top-level keys")
        print("=" * 60)
        required_top = ["algorithm", "f1", "roc_auc", "features", "ranker_eval"]
        for key in required_top:
            assert key in data, f"Missing top-level key: {key!r}"
            print(f"  PASS: key '{key}' present")
        assert isinstance(data["features"], list) and len(data["features"]) == 5, \
            f"Expected 5 features, got {data['features']}"
        print(f"  PASS: features list has 5 items: {data['features']}")
        print()

        # ── TEST 4: ranker_eval required keys ──────────────────────────────────
        print("=" * 60)
        print("TEST 4: ranker_eval required keys")
        print("=" * 60)
        re = data["ranker_eval"]
        required_re = [
            "cv_scheme_jd_id", "cv_scheme_category",
            "medium_set_eval", "classifier_parity",
            "conclusions", "caveats",
        ]
        for key in required_re:
            assert key in re, f"Missing ranker_eval key: {key!r}"
            print(f"  PASS: ranker_eval.{key} present")

        # Check significance and ablation inside each scheme
        for scheme_key in ["cv_scheme_jd_id", "cv_scheme_category"]:
            scheme = re[scheme_key]
            assert "full_features" in scheme, f"Missing full_features in {scheme_key}"
            assert "ablation_drop_exp_edu" in scheme, f"Missing ablation in {scheme_key}"
        print("  PASS: significance and ablation present in both schemes")

        # Check medium_set keys
        med = re["medium_set_eval"]
        for ranker_key in ["a_semantic", "b_clf_blend"]:
            assert ranker_key in med, f"Missing medium_set_eval.{ranker_key}"
            assert "ndcg_at_10" in med[ranker_key], f"Missing ndcg_at_10 in {ranker_key}"
        print("  PASS: medium_set_eval has required ranker keys\n")

        # ── TEST 5: Regression anchor ──────────────────────────────────────────
        print("=" * 60)
        print("TEST 5: Regression anchor (id=91153752, final_score=0.7326)")
        print("=" * 60)
        # Wait if rate-limited
        for attempt in range(3):
            rr = requests.post(
                f"{BASE}/rank?alpha=0.5",
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
                json=BENCHMARK_JD,
            )
            if rr.status_code == 429:
                print(f"  Rate limited (attempt {attempt+1}), waiting 60 s...")
                time.sleep(60)
                continue
            rr.raise_for_status()
            break
        rank_data = rr.json()
        top = rank_data["results"][0]
        print(f"  Top result: id={top['id']}  final_score={top['final_score']:.4f}")
        assert str(top["id"]) == "91153752", \
            f"FAIL: expected id 91153752, got {top['id']}"
        assert abs(top["final_score"] - 0.7326) < 0.0005, \
            f"FAIL: final_score {top['final_score']:.4f} != 0.7326"
        print("PASS: regression anchor id=91153752, final_score=0.7326\n")

        # ── TEST 6: No string longer than 400 chars ────────────────────────────
        print("=" * 60)
        print("TEST 6: No string value longer than 400 characters")
        print("=" * 60)
        check_no_long_strings(data)
        print("PASS: all string values <= 400 characters\n")

        print("=" * 60)
        print("ALL TEST 10 MODEL-INFO TESTS PASSED")
        print("=" * 60)

    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
