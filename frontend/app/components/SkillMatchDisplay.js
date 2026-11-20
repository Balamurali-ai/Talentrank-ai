"use client";

export default function SkillMatchDisplay({ matched = [], missing = [], maxVisible = 6, compact = false }) {
  const allMatched = Array.isArray(matched) ? matched : [];
  const allMissing = Array.isArray(missing) ? missing : [];

  const visibleMatched = compact ? allMatched.slice(0, maxVisible) : allMatched;
  const remainingMatchedCount = allMatched.length - visibleMatched.length;

  const visibleMissing = compact ? allMissing.slice(0, Math.max(0, maxVisible - visibleMatched.length)) : allMissing;
  const remainingMissingCount = allMissing.length - visibleMissing.length;

  if (allMatched.length === 0 && allMissing.length === 0) {
    return <span style={{ fontSize: "0.8125rem", color: "var(--text-muted)" }}>—</span>;
  }

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", alignItems: "center" }}>
      {visibleMatched.map((skill, i) => (
        <span
          key={`m-${i}`}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.25rem",
            backgroundColor: "var(--success-bg)",
            color: "var(--success-text)",
            border: "1px solid var(--success-border)",
            borderRadius: "var(--radius-pill)",
            padding: compact ? "0.15rem 0.5rem" : "0.25rem 0.65rem",
            fontSize: compact ? "0.72rem" : "0.78rem",
            fontWeight: 500,
            lineHeight: 1.2,
          }}
          title="Matched Skill"
        >
          <span style={{ fontSize: "0.7rem" }}>✓</span> {skill}
        </span>
      ))}

      {visibleMissing.map((skill, i) => (
        <span
          key={`ms-${i}`}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.25rem",
            backgroundColor: "var(--bg-subtle)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border-default)",
            borderRadius: "var(--radius-pill)",
            padding: compact ? "0.15rem 0.5rem" : "0.25rem 0.65rem",
            fontSize: compact ? "0.72rem" : "0.78rem",
            fontWeight: 400,
            lineHeight: 1.2,
            textDecoration: "line-through",
            opacity: 0.85,
          }}
          title="Missing Skill"
        >
          {skill}
        </span>
      ))}

      {(remainingMatchedCount > 0 || remainingMissingCount > 0) && (
        <span
          style={{
            fontSize: "0.72rem",
            color: "var(--text-muted)",
            fontWeight: 600,
            padding: "0.1rem 0.35rem",
          }}
        >
          +{remainingMatchedCount + remainingMissingCount} more
        </span>
      )}
    </div>
  );
}
