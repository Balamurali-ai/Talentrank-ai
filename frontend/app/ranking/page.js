"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import AlphaSlider from "../components/AlphaSlider";
import UploadPanel from "../components/UploadPanel";
import CandidateTable from "../components/CandidateTable";
import Alert from "../components/Alert";
import EmptyState from "../components/EmptyState";
import LoadingState from "../components/LoadingState";

export default function RankingPage() {
  const router = useRouter();
  const {
    jd,
    setJd,
    alpha,
    setAlpha,
    results,
    explanationNote,
    isRanking,
    rankingError,
    runRanking,
    poolType,
    sessionId,
    uploadedResumesCount,
    comparisonCandidates,
    setIsCompareModalOpen,
  } = useTalentRank();

  const [localError, setLocalError] = useState("");
  const [activeTab, setActiveTab] = useState("form"); // "form" | "results"

  const handleInputChange = (field, value) => {
    setJd((prev) => ({ ...prev, [field]: value }));
  };

  const handleRankSubmit = async (e) => {
    e.preventDefault();
    setLocalError("");

    if (!jd.jd_text || !jd.jd_text.trim()) {
      setLocalError("Full Job Description text is required.");
      return;
    }

    try {
      await runRanking();
      setActiveTab("results");
    } catch (err) {
      setLocalError(err.message || "Ranking execution failed.");
    }
  };

  const handleLoadSample = (sampleType) => {
    if (sampleType === "developer") {
      setJd({
        job_title: "Web Developer",
        role: "Full-Stack Engineer",
        skills: "HTML, CSS, JavaScript, React, Node.js",
        experience: "3 to 5 Years",
        qualifications: "Bachelor's degree in Computer Science or related field",
        responsibilities: "Develop modern web interfaces, integrate RESTful APIs, maintain database performance, and collaborate with product teams.",
        jd_text: "Looking for an experienced Web Developer with strong front-end and back-end fundamentals. Required skills include HTML, CSS, JavaScript, React, and Node.js. Experience building responsive web applications and deploying APIs is expected.",
      });
    } else if (sampleType === "accountant") {
      setJd({
        job_title: "Senior Accountant",
        role: "Financial Reporting Specialist",
        skills: "Financial Reporting, GAAP, Ledger, Tax Compliance, Excel",
        experience: "5+ Years",
        qualifications: "Bachelor's in Accounting or CPA certification",
        responsibilities: "Manage monthly financial closes, prepare audited balance sheets, reconcile bank accounts, and oversee tax filings.",
        jd_text: "Seeking a Senior Accountant to lead general ledger maintenance and financial reporting. Must possess in-depth knowledge of GAAP standards, ledger reconciliations, tax compliance, and advanced spreadsheet modeling.",
      });
    }
    setLocalError("");
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="Talent Ranking Workspace"
        description="Configure job parameters, adjust semantic vs ML balance, and screen top candidates with explainable ranking signals."
        badge={{
          text: poolType === "uploaded" ? `Uploaded Pool (${uploadedResumesCount})` : "Default Pool (2,484)",
          type: poolType === "uploaded" ? "success" : "neutral",
        }}
        actions={
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {results.length > 0 && (
              <button
                className={`btn btn-sm ${activeTab === "results" ? "btn-primary" : "btn-secondary"}`}
                onClick={() => setActiveTab("results")}
              >
                View Results ({results.length})
              </button>
            )}
            <button
              className={`btn btn-sm ${activeTab === "form" ? "btn-primary" : "btn-secondary"}`}
              onClick={() => setActiveTab("form")}
            >
              JD Parameters
            </button>
          </div>
        }
      />

      {(localError || rankingError) && (
        <Alert type="error" msg={localError || rankingError} onDismiss={() => setLocalError("")} />
      )}

      {/* Tabs / Switcher Bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid var(--border-default)",
          marginBottom: "1.5rem",
          paddingBottom: "0.5rem",
        }}
      >
        <div style={{ display: "flex", gap: "1rem" }}>
          <button
            onClick={() => setActiveTab("form")}
            style={{
              background: "none",
              border: "none",
              padding: "0.5rem 0.25rem",
              fontSize: "0.9375rem",
              fontWeight: activeTab === "form" ? 700 : 500,
              color: activeTab === "form" ? "var(--primary)" : "var(--text-secondary)",
              borderBottom: activeTab === "form" ? "2px solid var(--primary)" : "2px solid transparent",
              cursor: "pointer",
            }}
          >
            1. Job Description & Tuning
          </button>
          <button
            onClick={() => setActiveTab("results")}
            style={{
              background: "none",
              border: "none",
              padding: "0.5rem 0.25rem",
              fontSize: "0.9375rem",
              fontWeight: activeTab === "results" ? 700 : 500,
              color: activeTab === "results" ? "var(--primary)" : "var(--text-secondary)",
              borderBottom: activeTab === "results" ? "2px solid var(--primary)" : "2px solid transparent",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "0.35rem",
            }}
          >
            2. Candidate Rankings
            {results.length > 0 && (
              <span className="badge badge-primary" style={{ fontSize: "0.7rem" }}>
                {results.length}
              </span>
            )}
          </button>
        </div>

        {activeTab === "form" && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Load Sample:</span>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => handleLoadSample("developer")}
            >
              Web Developer
            </button>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => handleLoadSample("accountant")}
            >
              Accountant
            </button>
          </div>
        )}
      </div>

      {/* TAB 1: FORM WORKSPACE */}
      {activeTab === "form" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          
          {/* Resume Pool Selection Panel */}
          <UploadPanel />

          {/* Job Description Input Form */}
          <form onSubmit={handleRankSubmit} className="saas-card" style={{ padding: "1.75rem" }}>
            <div style={{ marginBottom: "1.5rem" }}>
              <h2 style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Job Description Specifications
              </h2>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                Define the role attributes and requirements for semantic retrieval and ML scoring.
              </p>
            </div>

            {/* Section 1: Role Overview */}
            <div style={{ marginBottom: "1.5rem" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: "0.75rem" }}>
                Role Overview
              </span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="job_title">
                    Job Title
                  </label>
                  <input
                    id="job_title"
                    className="form-input"
                    value={jd.job_title}
                    onChange={(e) => handleInputChange("job_title", e.target.value)}
                    placeholder="e.g. Senior Frontend Engineer"
                  />
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="role">
                    Role Category / Function
                  </label>
                  <input
                    id="role"
                    className="form-input"
                    value={jd.role}
                    onChange={(e) => handleInputChange("role", e.target.value)}
                    placeholder="e.g. Software Development"
                  />
                </div>
              </div>
            </div>

            {/* Section 2: Candidate Requirements */}
            <div style={{ marginBottom: "1.5rem" }}>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--primary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: "0.75rem" }}>
                Candidate Qualifications & Requirements
              </span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="skills">
                    Key Required Skills
                    <span className="form-helper">Comma-separated</span>
                  </label>
                  <input
                    id="skills"
                    className="form-input"
                    value={jd.skills}
                    onChange={(e) => handleInputChange("skills", e.target.value)}
                    placeholder="e.g. Python, SQL, Docker, React"
                  />
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="experience">
                    Experience Level
                  </label>
                  <input
                    id="experience"
                    className="form-input"
                    value={jd.experience}
                    onChange={(e) => handleInputChange("experience", e.target.value)}
                    placeholder="e.g. 3 to 5 Years"
                  />
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="qualifications">
                    Minimum Qualifications
                  </label>
                  <input
                    id="qualifications"
                    className="form-input"
                    value={jd.qualifications}
                    onChange={(e) => handleInputChange("qualifications", e.target.value)}
                    placeholder="e.g. Bachelor's in CS or equivalent"
                  />
                </div>

                <div className="form-group" style={{ margin: 0 }}>
                  <label className="form-label" htmlFor="responsibilities">
                    Key Responsibilities
                  </label>
                  <input
                    id="responsibilities"
                    className="form-input"
                    value={jd.responsibilities}
                    onChange={(e) => handleInputChange("responsibilities", e.target.value)}
                    placeholder="e.g. Architecture, Code reviews, Deployment"
                  />
                </div>
              </div>
            </div>

            {/* Section 3: Full Job Description */}
            <div className="form-group">
              <label className="form-label" htmlFor="jd_text">
                Full Job Description Text <span style={{ color: "var(--danger)" }}>*</span>
                <span className="form-helper">
                  {jd.jd_text?.length || 0} characters
                </span>
              </label>
              <textarea
                id="jd_text"
                rows={5}
                className="form-textarea"
                value={jd.jd_text}
                onChange={(e) => handleInputChange("jd_text", e.target.value)}
                placeholder="Paste complete job posting here for optimal SentenceTransformer embeddings..."
                required
              />
            </div>

            {/* Section 4: Ranking Balance Slider */}
            <div style={{ marginBottom: "1.75rem" }}>
              <AlphaSlider value={alpha} onChange={setAlpha} />
            </div>

            {/* Submit Bar */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingTop: "1rem", borderTop: "1px solid var(--border-default)" }}>
              <span style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>
                Target pool: <strong>{poolType === "uploaded" ? `Uploaded CSV (${uploadedResumesCount})` : "Default 2,484 resumes"}</strong>
              </span>

              <button
                type="submit"
                className="btn btn-primary btn-lg"
                disabled={isRanking || !jd.jd_text?.trim()}
              >
                {isRanking ? "Ranking Candidates..." : "⚡ Execute Candidate Ranking"}
              </button>
            </div>
          </form>

        </div>
      )}

      {/* TAB 2: RESULTS WORKSPACE */}
      {activeTab === "results" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {isRanking ? (
            <div className="saas-card" style={{ padding: "3rem" }}>
              <LoadingState message="Computing embeddings, FAISS top-200 shortlist, and XGBoost TreeSHAP contributions..." rows={8} />
            </div>
          ) : results.length > 0 ? (
            <div className="saas-card" style={{ padding: "1.5rem" }}>
              {/* Header summary */}
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "1rem", marginBottom: "1rem" }}>
                <div>
                  <h2 style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                    Top Ranked Candidates ({results.length})
                  </h2>
                  <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                    Ranked against <strong>{jd.job_title || "Job Description"}</strong> (α = {alpha})
                  </p>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  {comparisonCandidates.length > 0 && (
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => setIsCompareModalOpen(true)}
                    >
                      Compare Selected ({comparisonCandidates.length})
                    </button>
                  )}
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => setActiveTab("form")}
                  >
                    Modify Search
                  </button>
                </div>
              </div>

              {/* Explanation Note Alert */}
              {explanationNote && (
                <div
                  style={{
                    backgroundColor: "var(--info-bg)",
                    border: "1px solid var(--info-border)",
                    borderRadius: "var(--radius-md)",
                    padding: "0.75rem 1rem",
                    fontSize: "0.8125rem",
                    color: "var(--info-text)",
                    marginBottom: "1.25rem",
                  }}
                >
                  <strong>Scoring Model:</strong> {explanationNote}
                </div>
              )}

              {/* Candidate Table */}
              <CandidateTable candidates={results} showFilters={true} />
            </div>
          ) : (
            <EmptyState
              icon="⚡"
              title="No Candidates Ranked Yet"
              description="Complete the Job Description form and click 'Execute Candidate Ranking' to retrieve and score candidates."
              action={
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => setActiveTab("form")}
                >
                  Configure Job Description
                </button>
              }
            />
          )}
        </div>
      )}
    </DashboardLayout>
  );
}
