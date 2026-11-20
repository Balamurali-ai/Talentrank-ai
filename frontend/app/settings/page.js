"use client";

import { useState } from "react";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import AlphaSlider from "../components/AlphaSlider";
import Alert from "../components/Alert";

export default function SettingsPage() {
  const {
    alpha,
    setAlpha,
    sessionId,
    poolType,
    uploadedResumesCount,
    health,
    checkHealth,
    useDefaultPool,
    user,
  } = useTalentRank();

  const [healthStatus, setHealthStatus] = useState(health);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [notice, setNotice] = useState("");

  const handleHealthProbe = async () => {
    setCheckingHealth(true);
    setNotice("");
    try {
      const res = await checkHealth();
      setHealthStatus(res);
      setNotice("Backend health check completed successfully.");
    } catch (e) {
      setHealthStatus({ status: "offline", models_loaded: false });
      setNotice("Failed to reach backend server at http://localhost:8000.");
    } finally {
      setCheckingHealth(false);
    }
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="Settings & System Architecture"
        description="Configure session-level ranking parameters, inspect pipeline specifications, and review system security controls."
        badge={{ text: "Session Level", type: "neutral" }}
      />

      {notice && <Alert type="info" msg={notice} onDismiss={() => setNotice("")} />}

      <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
        
        {/* Section 1: Ranking Preferences & Alpha Parameter */}
        <div className="saas-card" style={{ padding: "1.75rem" }}>
          <div style={{ marginBottom: "1.25rem" }}>
            <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
              Ranking Preferences & Weighting Parameters
            </h2>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
              Controls the balance between Semantic Vector Similarity and the Machine Learning Classifier. Settings are stored at the session level.
            </p>
          </div>

          <div style={{ marginBottom: "1.5rem" }}>
            <AlphaSlider value={alpha} onChange={setAlpha} />
          </div>

          <div
            style={{
              backgroundColor: "var(--bg-subtle)",
              borderRadius: "var(--radius-md)",
              padding: "1rem",
              border: "1px solid var(--border-default)",
              fontSize: "0.8125rem",
              lineHeight: 1.5,
              color: "var(--text-secondary)",
            }}
          >
            <h4 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.4rem" }}>
              Understanding the Alpha (α) Weighting Formula:
            </h4>
            <div style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "var(--primary)", marginBottom: "0.5rem" }}>
              Final Score = α · ML_Score + (1 - α) · Semantic_Similarity
            </div>
            <ul style={{ paddingLeft: "1.25rem", display: "flex", flexDirection: "column", gap: "0.25rem" }}>
              <li>
                <strong>α closer to 1.0:</strong> Greater weight placed on the XGBoost candidate scorer, emphasizing feature correlations learned from category labels and tabular attributes.
              </li>
              <li>
                <strong>α closer to 0.0:</strong> Greater weight placed on direct SentenceTransformer FAISS cosine similarity, prioritizing contextual keyword and phrase matches.
              </li>
              <li>
                <strong>Default (α = 0.50):</strong> Balanced 50/50 linear combination providing both contextual semantic relevance and tabular feature scoring.
              </li>
            </ul>
          </div>
        </div>

        {/* Section 2: Active Session & Resume Pool Specifications */}
        <div className="saas-card" style={{ padding: "1.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
            <div>
              <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Resume Pool & Session State
              </h2>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                Current candidate index configuration and memory isolation rules.
              </p>
            </div>
            {poolType === "uploaded" && (
              <button className="btn btn-secondary btn-sm" onClick={useDefaultPool}>
                Reset to Default Pool
              </button>
            )}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "1.25rem" }}>
            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Active Pool Type</span>
              <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginTop: "0.2rem" }}>
                {poolType === "uploaded" ? "Custom Session Upload" : "Default Pre-Indexed Pool"}
              </div>
            </div>

            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Total Resumes in Index</span>
              <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginTop: "0.2rem" }}>
                {uploadedResumesCount?.toLocaleString() || "2,484"}
              </div>
            </div>

            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Session TTL (Time to Live)</span>
              <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginTop: "0.2rem" }}>
                2 Hours (Auto-Eviction)
              </div>
            </div>

            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Upload Ingestion Cap</span>
              <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)", marginTop: "0.2rem" }}>
                2,000 Rows (CPU Safety)
              </div>
            </div>
          </div>

          <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
            <strong>CSV Upload Contract:</strong> Uploaded CSV files must contain columns <code>ID</code> and <code>Resume_str</code>. Column <code>Category</code> is optional. Upon upload, resumes undergo PII redaction and are embedded in-memory.
          </div>
        </div>

        {/* Section 3: Model Architecture Specifications */}
        <div className="saas-card" style={{ padding: "1.75rem" }}>
          <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.25rem" }}>
            Model & ML Pipeline Architecture
          </h2>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "1.25rem" }}>
            Specifications for the two-stage hybrid retrieval and re-ranking architecture.
          </p>

          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Pipeline Component</th>
                  <th>Model / Algorithm</th>
                  <th>Configuration Details</th>
                  <th>Role in TalentRank AI</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 600 }}>Stage 1: Semantic Embedding</td>
                  <td>SentenceTransformers <code>all-MiniLM-L6-v2</code></td>
                  <td>384-dim dense vectors, L2 normalized, CPU inference</td>
                  <td>Encodes job descriptions and resumes into unified embedding space</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Stage 1: Vector Search</td>
                  <td>FAISS <code>IndexFlatIP</code></td>
                  <td>Exact cosine similarity via Inner Product, top-200 shortlist</td>
                  <td>Rapid candidate retrieval pruning from full candidate pool</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Stage 2: Candidate Scorer</td>
                  <td>XGBoost Classifier</td>
                  <td><code>binary:logistic</code>, 200 estimators, max depth 6, scale_pos_weight 4.06</td>
                  <td>Evaluates tabular engineered features on top-200 shortlist</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Interpretability Engine</td>
                  <td>TreeSHAP</td>
                  <td>Exact polynomial tree-path Shapley attribution values</td>
                  <td>Extracts feature contributions and verified signals for recruiters</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Cluster Visualizer</td>
                  <td>K-Means + PCA (2D)</td>
                  <td>Automated Silhouette k-selection (2-8), TF-IDF keyword extraction</td>
                  <td>Projects resume landscape and highlights nearest cluster to JD</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 4: Backend Telemetry & Health Probe */}
        <div className="saas-card" style={{ padding: "1.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
            <div>
              <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Backend API Status & Telemetry
              </h2>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                FastAPI service health check and active endpoint registry.
              </p>
            </div>
            <button
              className="btn btn-secondary btn-sm"
              onClick={handleHealthProbe}
              disabled={checkingHealth}
            >
              {checkingHealth ? "Probing..." : "Probe /health"}
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
            <span
              style={{
                width: "12px",
                height: "12px",
                borderRadius: "50%",
                backgroundColor: healthStatus?.models_loaded ? "var(--success)" : "var(--danger)",
              }}
            />
            <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
              FastAPI Endpoint: <code>http://localhost:8000</code>
            </span>
            <span className={`badge ${healthStatus?.models_loaded ? "badge-success" : "badge-danger"}`}>
              {healthStatus?.models_loaded ? "Models Loaded & Healthy" : "Offline / Unreachable"}
            </span>
          </div>

          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Registered API routes: <code>/token</code>, <code>/rank?alpha=...</code>, <code>/upload-resumes</code>, <code>/upload-status/:id</code>, <code>/cluster-map/:id</code>, <code>/model-info</code>, <code>/health</code>
          </div>
        </div>

        {/* Section 5: Security & Privacy Architecture */}
        <div className="saas-card" style={{ padding: "1.75rem" }}>
          <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.25rem" }}>
            Security & Candidate Privacy Architecture
          </h2>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Technical controls implemented to protect candidate personally identifiable information (PII).
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
                🔒 PII Masking via spaCy NER
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0, lineHeight: 1.4 }}>
                Candidate names, email addresses, phone numbers, and identifying tokens are automatically masked using entity recognition prior to feature computation and indexing.
              </p>
            </div>

            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
                🔑 Encryption at Rest (Fernet)
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0, lineHeight: 1.4 }}>
                Default resume datasets are encrypted on disk with symmetric Fernet (AES-128-CBC with HMAC-SHA256) and decrypted only into volatile memory upon service startup.
              </p>
            </div>

            <div style={{ backgroundColor: "var(--bg-subtle)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-default)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)", marginBottom: "0.25rem" }}>
                🛡️ Stateless JWT Authentication
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0, lineHeight: 1.4 }}>
                All sensitive endpoints require an OAuth2 Bearer token signed with HS256 and configured with 60-minute expiration.
              </p>
            </div>
          </div>
        </div>

      </div>
    </DashboardLayout>
  );
}
