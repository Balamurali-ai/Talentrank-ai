"use client";

import { useEffect, useCallback } from "react";
import { useTalentRank } from "../context/TalentRankContext";
import ScoreBreakdown from "./ScoreBreakdown";
import SkillMatchDisplay from "./SkillMatchDisplay";

export default function CandidateDetailDrawer() {
  const {
    selectedCandidate,
    setSelectedCandidate,
    toggleComparison,
    isCompared,
  } = useTalentRank();

  // Close on Escape
  const handleKeyDown = useCallback((e) => {
    if (e.key === "Escape") {
      setSelectedCandidate(null);
    }
  }, [setSelectedCandidate]);

  useEffect(() => {
    if (selectedCandidate) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "unset";
    };
  }, [selectedCandidate, handleKeyDown]);

  if (!selectedCandidate) return null;

  const candidate = selectedCandidate;
  const exp = candidate.explanation;
  const compared = isCompared(candidate.id);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        display: "flex",
        justifyContent: "flex-end",
        backgroundColor: "rgba(15, 23, 42, 0.45)",
        backdropFilter: "blur(2px)",
      }}
      onClick={() => setSelectedCandidate(null)}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "580px",
          height: "100%",
          backgroundColor: "var(--bg-surface)",
          boxShadow: "var(--shadow-modal)",
          display: "flex",
          flexDirection: "column",
          overflowY: "auto",
          animation: "slideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <style jsx>{`
          @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
          }
        `}</style>

        {/* Drawer Header */}
        <div
          style={{
            padding: "1.25rem 1.5rem",
            borderBottom: "1px solid var(--border-default)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            position: "sticky",
            top: 0,
            backgroundColor: "var(--bg-surface)",
            zIndex: 10,
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)" }}>
                Candidate #{candidate.id}
              </span>
              {candidate.category && (
                <span className="badge badge-primary">{candidate.category}</span>
              )}
            </div>
            <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
              Candidate profile & algorithmic explanation
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <button
              className={`btn btn-sm ${compared ? "btn-secondary" : "btn-primary"}`}
              onClick={() => toggleComparison(candidate)}
              title={compared ? "Remove from comparison" : "Add to side-by-side comparison"}
            >
              {compared ? "✓ In Compare" : "+ Compare"}
            </button>
            <button
              onClick={() => setSelectedCandidate(null)}
              style={{
                background: "transparent",
                border: "none",
                fontSize: "1.25rem",
                color: "var(--text-muted)",
                cursor: "pointer",
                padding: "0.25rem 0.5rem",
                borderRadius: "var(--radius-sm)",
              }}
              aria-label="Close drawer"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Drawer Content */}
        <div style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          
          {/* Key Scores Grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: "0.75rem",
              backgroundColor: "var(--bg-subtle)",
              padding: "1rem",
              borderRadius: "var(--radius-lg)",
              border: "1px solid var(--border-default)",
            }}
          >
            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                Final Score
              </span>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--primary)" }} className="tabular-nums">
                {candidate.final_score?.toFixed(3) || "—"}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Blended Ranking</div>
            </div>

            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                Semantic Sim
              </span>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary)" }} className="tabular-nums">
                {candidate.semantic_sim?.toFixed(3) || "—"}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>FAISS Cosine</div>
            </div>

            <div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                ML Scorer
              </span>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary)" }} className="tabular-nums">
                {candidate.ML_score?.toFixed(3) || "—"}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>XGBoost Prob</div>
            </div>
          </div>

          {/* Key Signals (TreeSHAP Reasons) */}
          {exp && exp.reasons && exp.reasons.length > 0 && (
            <div className="saas-card" style={{ padding: "1.25rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <span style={{ fontSize: "1rem" }}>💡</span>
                <h4 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
                  Algorithmic Key Signals
                </h4>
              </div>
              <ul style={{ paddingLeft: "1.25rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {exp.reasons.map((r, i) => (
                  <li key={i} style={{ fontSize: "0.8125rem", color: "var(--text-primary)", lineHeight: 1.4 }}>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Verifiable Facts */}
          {exp && exp.facts && exp.facts.length > 0 && (
            <div className="saas-card" style={{ padding: "1.25rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <span style={{ fontSize: "1rem" }}>📋</span>
                <h4 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
                  Extracted Facts
                </h4>
              </div>
              <ul style={{ paddingLeft: "1.25rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {exp.facts.map((f, i) => (
                  <li key={i} style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Skills Breakdown */}
          {exp && (exp.matched_skills?.length > 0 || exp.missing_skills?.length > 0) && (
            <div className="saas-card" style={{ padding: "1.25rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
                <span style={{ fontSize: "1rem" }}>🎯</span>
                <h4 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
                  Skills Alignment Analysis
                </h4>
              </div>
              <div style={{ marginBottom: "0.75rem" }}>
                <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--success-text)", marginBottom: "0.35rem" }}>
                  Matched Skills ({exp.matched_skills?.length || 0})
                </div>
                <SkillMatchDisplay matched={exp.matched_skills || []} missing={[]} compact={false} />
              </div>
              <div>
                <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
                  Missing / Unmatched JD Skills ({exp.missing_skills?.length || 0})
                </div>
                <SkillMatchDisplay matched={[]} missing={exp.missing_skills || []} compact={false} />
              </div>
            </div>
          )}

          {/* TreeSHAP Feature Attribution Table */}
          {exp && exp.features && exp.features.length > 0 && (
            <div className="saas-card" style={{ padding: "1.25rem" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "1rem" }}>🌲</span>
                  <h4 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
                    TreeSHAP Feature Contributions
                  </h4>
                </div>
                {exp.bias !== undefined && (
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Base Bias: {exp.bias.toFixed(3)}
                  </span>
                )}
              </div>

              <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.75rem", lineHeight: 1.4 }}>
                TreeSHAP calculates the exact marginal contribution of each engineered feature toward the model's prediction score relative to expected value.
              </p>

              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8125rem" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border-default)", textAlign: "left" }}>
                    <th style={{ padding: "0.5rem 0", color: "var(--text-secondary)", fontWeight: 600 }}>Feature</th>
                    <th style={{ padding: "0.5rem 0", color: "var(--text-secondary)", fontWeight: 600, textAlign: "right" }}>Raw Value</th>
                    <th style={{ padding: "0.5rem 0", color: "var(--text-secondary)", fontWeight: 600, textAlign: "right" }}>Contribution</th>
                  </tr>
                </thead>
                <tbody>
                  {exp.features.map((feat, i) => {
                    const isPositive = feat.contribution > 0;
                    return (
                      <tr key={i} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                        <td style={{ padding: "0.5rem 0", color: "var(--text-primary)", fontWeight: 500 }}>
                          <code>{feat.name}</code>
                        </td>
                        <td style={{ padding: "0.5rem 0", textAlign: "right", color: "var(--text-secondary)" }} className="tabular-nums">
                          {feat.value.toFixed(3)}
                        </td>
                        <td
                          style={{
                            padding: "0.5rem 0",
                            textAlign: "right",
                            fontWeight: 600,
                            color: isPositive ? "var(--success)" : "var(--danger)",
                          }}
                          className="tabular-nums"
                        >
                          {isPositive ? `+${feat.contribution.toFixed(3)}` : feat.contribution.toFixed(3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Academic Caution */}
          <div
            style={{
              backgroundColor: "var(--info-bg)",
              border: "1px solid var(--info-border)",
              borderRadius: "var(--radius-md)",
              padding: "0.75rem 1rem",
              fontSize: "0.75rem",
              color: "var(--info-text)",
              lineHeight: 1.4,
            }}
          >
            <strong>Evaluation Note:</strong> TreeSHAP feature contributions reflect patterns learned from synthetic job descriptions and proxy category labels. High scores indicate strong semantic alignment and keyword overlap, not a guarantee of candidate suitability.
          </div>
        </div>
      </div>
    </div>
  );
}
