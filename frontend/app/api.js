const BASE = "http://localhost:8000";

async function handleResponse(res) {
  if (res.ok) return res.json();
  let detail = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    detail = body.detail || detail;
  } catch (_) {}
  const err = new Error(detail);
  err.status = res.status;
  throw err;
}

export async function login(username, password) {
  const res = await fetch(`${BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  return handleResponse(res);
}

export async function uploadResumes(token, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload-resumes`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  return handleResponse(res);
}

export async function getUploadStatus(token, sessionId) {
  const res = await fetch(`${BASE}/upload-status/${sessionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse(res);
}

export async function getClusterMap(token, sessionId) {
  const res = await fetch(`${BASE}/cluster-map/${sessionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse(res);
}

export async function getModelInfo(token) {
  const res = await fetch(`${BASE}/model-info`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return handleResponse(res);
}

export async function rank(token, payload, alpha = 0.5) {
  const safeAlpha = typeof alpha === "number" ? Math.max(0, Math.min(1, alpha)) : 0.5;
  const res = await fetch(`${BASE}/rank?alpha=${safeAlpha}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`);
  return handleResponse(res);
}
