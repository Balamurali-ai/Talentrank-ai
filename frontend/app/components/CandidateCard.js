"use client";

import { useTalentRank } from "../context/TalentRankContext";
import ScoreBreakdown from "./ScoreBreakdown";
import SkillMatchDisplay from "./SkillMatchDisplay";

export default function CandidateCard({ candidate, rank }) {
  const {
    setSelectedCandidate,
    toggleComparison,
    isCompared,
  } = useTalentRank();

  const compared = isCompared(candidate.id);

  return (
    <div
      className="saas-card"
      style={{
        padding: "1.25rem",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        border: compared ? "2px solid var(--primary)" : "1px solid var(--border-default)",
        backgroundColor: compared ? "var(--primary-light)" : "var(--bg-surface)",
        cursor: "pointer",
        transition: "all 0.2s ease",
      }}
      onClick={() => setSelectedCandidate(candidate)}
    >
      <div>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {rank && (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "24px",
                  height: "24px",
                  borderRadius: "50%",
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  backgroundColor: rank <= 3 ? "var(--primary-light)" : "var(--bg-subtle)",
                  color: rank <= 3 ? "var(--primary)" : "var(--text-secondary)",
                }}
              >
                #{rank}
              </span>
            )}
            <div>
              <div style={{ fontWeight: 700, fontSize: "0.9375rem", color: "var(--text-primary)" }}>
                Candidate #{candidate.id}
              </div>
              {candidate.category && (
                <span className="badge badge-neutral" style={{ fontSize: "0.7rem", marginTop: "0.2rem" }}>
                  {candidate.category}
                </span>
              )}
            </div>
          </div>

          <div onClick={(e) => e.stopPropagation()}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.75rem", cursor: "pointer", color: "var(--text-secondary)" }}>
              <input
                type="checkbox"
                checked={compared}
                onChange={() => toggleComparison(candidate)}
                style={{ accentColor: "var(--primary)" }}
              />
              Compare
            </label>
          </div>
        </div>

        {/* Scores */}
        <div
          style={{
            backgroundColor: "var(--bg-subtle)",
            padding: "0.75rem",
            borderRadius: "var(--radius-md)",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
            marginBottom: "0.75rem",
          }}
        >
          <ScoreBreakdown value={candidate.final_score} label="Final Score" accent size="md" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", paddingTop: "0.25rem", borderTop: "1px solid var(--border-default)" }}>
            <ScoreBreakdown value={candidate.semantic_sim} label="Semantic" size="sm" />
            <ScoreBreakdown value={candidate.ML_score} label="ML Scorer" size="sm" />
          </div>
        </div>

        {/* Skills */}
        <div style={{ marginBottom: "0.75rem" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            Skill Match
          </div>
          <SkillMatchDisplay
            matched={candidate.explanation?.matched_skills || []}
            missing={candidate.explanation?.missing_skills || []}
            compact={true}
            maxVisible={3}
          />
        </div>
      </div>

      {/* Footer */}
      <div style={{ paddingTop: "0.75rem", borderTop: "1px solid var(--border-default)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          {candidate.explanation?.reasons?.length || 0} signals identified
        </span>
        <button
          className="btn btn-ghost btn-sm"
          onClick={(e) => {
            e.stopPropagation();
            setSelectedCandidate(candidate);
          }}
        >
          View SHAP →
        </button>
      </div>
    </div>
  );
}
