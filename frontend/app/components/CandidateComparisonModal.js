"use client";

import { useTalentRank } from "../context/TalentRankContext";
import ScoreBreakdown from "./ScoreBreakdown";
import SkillMatchDisplay from "./SkillMatchDisplay";

export default function CandidateComparisonModal() {
  const {
    comparisonCandidates,
    clearComparison,
    toggleComparison,
    isCompareModalOpen,
    setIsCompareModalOpen,
    setSelectedCandidate,
  } = useTalentRank();

  if (!isCompareModalOpen || comparisonCandidates.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 60,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "rgba(15, 23, 42, 0.5)",
        backdropFilter: "blur(3px)",
        padding: "1rem",
      }}
      onClick={() => setIsCompareModalOpen(false)}
    >
      <div
        className="saas-card"
        style={{
          width: "100%",
          maxWidth: "1100px",
          maxHeight: "90vh",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          boxShadow: "var(--shadow-modal)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="saas-card-header">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <h2 style={{ fontSize: "1.125rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Candidate Comparison
              </h2>
              <span className="badge badge-primary">
                {comparisonCandidates.length} of 4 selected
              </span>
            </div>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
              Side-by-side comparison of candidate signals, scores, and skill alignment.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={clearComparison}
            >
              Clear All
            </button>
            <button
              onClick={() => setIsCompareModalOpen(false)}
              style={{
                background: "transparent",
                border: "none",
                fontSize: "1.25rem",
                color: "var(--text-muted)",
                cursor: "pointer",
                padding: "0.25rem 0.5rem",
              }}
              aria-label="Close comparison"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Comparison Grid */}
        <div style={{ padding: "1.5rem", overflowY: "auto", flex: 1 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `200px repeat(${comparisonCandidates.length}, minmax(200px, 1fr))`,
              gap: "1rem",
              alignItems: "stretch",
            }}
          >
            {/* Row: Candidate Header */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Candidate ID
            </div>
            {comparisonCandidates.map((c) => (
              <div
                key={c.id}
                style={{
                  backgroundColor: "var(--bg-subtle)",
                  padding: "0.75rem",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-default)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: "0.9375rem", color: "var(--text-primary)" }}>
                    #{c.id}
                  </div>
                  {c.category && (
                    <span className="badge badge-neutral" style={{ marginTop: "0.25rem", fontSize: "0.7rem" }}>
                      {c.category}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => toggleComparison(c)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--danger)",
                    cursor: "pointer",
                    fontSize: "0.875rem",
                    fontWeight: 600,
                  }}
                  title="Remove from comparison"
                >
                  ✕
                </button>
              </div>
            ))}

            {/* Row: Final Score */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Final Blended Score
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`final-${c.id}`} style={{ padding: "0.5rem" }}>
                <ScoreBreakdown value={c.final_score} accent size="md" />
              </div>
            ))}

            {/* Row: Semantic Sim */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Semantic Similarity
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`sem-${c.id}`} style={{ padding: "0.5rem" }}>
                <ScoreBreakdown value={c.semantic_sim} size="sm" />
              </div>
            ))}

            {/* Row: ML Score */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              XGBoost ML Score
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`ml-${c.id}`} style={{ padding: "0.5rem" }}>
                <ScoreBreakdown value={c.ML_score} size="sm" />
              </div>
            ))}

            {/* Row: Feature Values (if present in explanation) */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Skill Overlap Feature
            </div>
            {comparisonCandidates.map((c) => {
              const feat = c.explanation?.features?.find((f) => f.name === "skill_overlap_score");
              return (
                <div key={`skill-feat-${c.id}`} style={{ padding: "0.5rem", fontSize: "0.875rem" }} className="tabular-nums">
                  {feat ? feat.value.toFixed(3) : "—"}
                </div>
              );
            })}

            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Experience Match
            </div>
            {comparisonCandidates.map((c) => {
              const feat = c.explanation?.features?.find((f) => f.name === "experience_match_score");
              return (
                <div key={`exp-feat-${c.id}`} style={{ padding: "0.5rem", fontSize: "0.875rem" }} className="tabular-nums">
                  {feat ? feat.value.toFixed(2) : "—"}
                </div>
              );
            })}

            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0", alignSelf: "center" }}>
              Education Match
            </div>
            {comparisonCandidates.map((c) => {
              const feat = c.explanation?.features?.find((f) => f.name === "education_match_score");
              return (
                <div key={`edu-feat-${c.id}`} style={{ padding: "0.5rem", fontSize: "0.875rem" }} className="tabular-nums">
                  {feat ? feat.value.toFixed(2) : "—"}
                </div>
              );
            })}

            {/* Row: Matched Skills */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0" }}>
              Matched Skills
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`matched-${c.id}`} style={{ padding: "0.5rem" }}>
                <SkillMatchDisplay matched={c.explanation?.matched_skills || []} missing={[]} compact={false} />
              </div>
            ))}

            {/* Row: Missing Skills */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0" }}>
              Missing Skills
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`missing-${c.id}`} style={{ padding: "0.5rem" }}>
                <SkillMatchDisplay matched={[]} missing={c.explanation?.missing_skills || []} compact={false} />
              </div>
            ))}

            {/* Row: Key Signals */}
            <div style={{ fontWeight: 600, color: "var(--text-secondary)", padding: "0.5rem 0" }}>
              Key Signals
            </div>
            {comparisonCandidates.map((c) => (
              <div key={`signals-${c.id}`} style={{ padding: "0.5rem", fontSize: "0.78rem", color: "var(--text-primary)" }}>
                {c.explanation?.reasons?.length ? (
                  <ul style={{ paddingLeft: "1rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                    {c.explanation.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                ) : (
                  <span style={{ color: "var(--text-muted)" }}>None identified</span>
                )}
              </div>
            ))}

            {/* Row: Action */}
            <div style={{ padding: "0.5rem 0" }} />
            {comparisonCandidates.map((c) => (
              <div key={`act-${c.id}`} style={{ padding: "0.5rem" }}>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ width: "100%" }}
                  onClick={() => {
                    setIsCompareModalOpen(false);
                    setSelectedCandidate(c);
                  }}
                >
                  View Details
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
