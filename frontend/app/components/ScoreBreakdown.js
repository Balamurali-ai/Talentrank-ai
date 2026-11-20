"use client";

export default function ScoreBreakdown({ value, label, accent = false, showBar = true, size = "md" }) {
  const num = typeof value === "number" && !isNaN(value) ? value : 0;
  const pct = Math.max(0, Math.min(100, Math.round(num * 100)));

  const barColor = accent
    ? "var(--primary)"
    : num >= 0.7
    ? "var(--success)"
    : num >= 0.4
    ? "var(--info)"
    : "var(--text-muted)";

  const barBg = accent ? "var(--primary-light)" : "var(--bg-subtle)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem", width: "100%", minWidth: "90px" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "0.5rem" }}>
        {label && (
          <span style={{ fontSize: size === "sm" ? "0.75rem" : "0.8125rem", color: "var(--text-secondary)", fontWeight: 500 }}>
            {label}
          </span>
        )}
        <span
          className="tabular-nums"
          style={{
            fontSize: size === "sm" ? "0.8125rem" : size === "lg" ? "1.125rem" : "0.875rem",
            fontWeight: accent ? 700 : 600,
            color: accent ? "var(--primary)" : "var(--text-primary)",
            marginLeft: "auto",
          }}
        >
          {num.toFixed(3)}
        </span>
      </div>

      {showBar && (
        <div
          style={{
            height: size === "sm" ? "5px" : size === "lg" ? "8px" : "6px",
            backgroundColor: barBg,
            borderRadius: "var(--radius-pill)",
            overflow: "hidden",
            width: "100%",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${pct}%`,
              backgroundColor: barColor,
              borderRadius: "var(--radius-pill)",
              transition: "width 0.4s ease-out",
            }}
          />
        </div>
      )}
    </div>
  );
}
