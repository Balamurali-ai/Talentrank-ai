"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import Alert from "../components/Alert";

export default function LoginPage() {
  const router = useRouter();
  const { token, authChecked, loginUser } = useTalentRank();

  const [username, setUsername] = useState("recruiter");
  const [password, setPassword] = useState("securepassword123");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (authChecked && token) {
      router.replace("/dashboard");
    }
  }, [authChecked, token, router]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await loginUser(username, password);
      router.push("/dashboard");
    } catch (err) {
      setError(
        err.status === 401
          ? "Invalid username or password. Please verify credentials."
          : err.message || "Failed to sign in. Please verify the backend server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFillDemo = () => {
    setUsername("recruiter");
    setPassword("securepassword123");
    setError("");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundColor: "var(--bg-canvas)",
      }}
    >
      {/* Top Brand Banner */}
      <div
        style={{
          padding: "1.25rem 2rem",
          display: "flex",
          alignItems: "center",
          borderBottom: "1px solid var(--border-default)",
          backgroundColor: "var(--bg-surface)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "var(--radius-md)",
              backgroundColor: "var(--primary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#ffffff",
              fontWeight: 800,
              fontSize: "1rem",
            }}
          >
            TR
          </div>
          <span style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)" }}>
            TalentRank <span style={{ color: "var(--primary)" }}>AI</span>
          </span>
        </div>
      </div>

      {/* Main Login Body */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem 1.5rem",
        }}
      >
        <div
          className="saas-card"
          style={{
            width: "100%",
            maxWidth: "440px",
            padding: "2.25rem",
          }}
        >
          {/* Header */}
          <div style={{ textAlign: "center", marginBottom: "1.75rem" }}>
            <h1 style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
              Sign in to TalentRank AI
            </h1>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
              Enter your credentials to access the AI recruitment analytics dashboard.
            </p>
          </div>

          <Alert msg={error} onDismiss={() => setError("")} />

          {/* Form */}
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label" htmlFor="username">
                Username
              </label>
              <input
                id="username"
                type="text"
                className="form-input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                autoComplete="username"
                placeholder="recruiter"
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="password">
                Password
              </label>
              <input
                id="password"
                type="password"
                className="form-input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder="••••••••••••"
              />
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: "100%", marginTop: "0.5rem", padding: "0.65rem" }}
              disabled={loading}
            >
              {loading ? "Authenticating..." : "Sign In to Dashboard"}
            </button>
          </form>

          {/* Demo helper */}
          <div
            style={{
              marginTop: "1.5rem",
              paddingTop: "1.25rem",
              borderTop: "1px solid var(--border-default)",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
              Prototype evaluation credentials
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleFillDemo}
              style={{ fontSize: "0.75rem" }}
            >
              Prefill Demo Credentials (recruiter)
            </button>
          </div>

          {/* Academic Prototype Notice */}
          <div
            style={{
              marginTop: "1.25rem",
              backgroundColor: "var(--bg-subtle)",
              borderRadius: "var(--radius-sm)",
              padding: "0.65rem 0.85rem",
              fontSize: "0.72rem",
              color: "var(--text-secondary)",
              lineHeight: 1.35,
              textAlign: "center",
            }}
          >
            <strong>Note:</strong> TalentRank AI is an academic research prototype. Candidate scores represent ranking signals relative to proxy labels.
          </div>
        </div>
      </div>
    </div>
  );
}
