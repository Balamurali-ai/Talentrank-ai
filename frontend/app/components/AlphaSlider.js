"use client";

export default function AlphaSlider({ value = 0.5, onChange, disabled = false, compact = false }) {
  const alphaVal = typeof value === "number" ? value : 0.5;
  const semanticPct = Math.round((1 - alphaVal) * 100);
  const mlPct = Math.round(alphaVal * 100);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        backgroundColor: compact ? "transparent" : "var(--bg-subtle)",
        padding: compact ? "0" : "1rem 1.25rem",
        borderRadius: "var(--radius-md)",
        border: compact ? "none" : "1px solid var(--border-default)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
            Ranking Balance (α = {alphaVal.toFixed(2)})
          </span>
          <span className="badge badge-primary" style={{ fontSize: "0.7rem" }}>
            {alphaVal === 0.5 ? "Balanced" : alphaVal > 0.5 ? "ML-Dominant" : "Semantic-Dominant"}
          </span>
        </div>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          Final = α·ML + (1-α)·Semantic
        </span>
      </div>

      {/* Visual Proportion Bar */}
      <div
        style={{
          display: "flex",
          height: "8px",
          width: "100%",
          borderRadius: "var(--radius-pill)",
          overflow: "hidden",
          backgroundColor: "#e2e8f0",
        }}
      >
        <div
          style={{
            width: `${semanticPct}%`,
            backgroundColor: "#3b82f6", // Blue for Semantic
            transition: "width 0.15s ease",
          }}
          title={`Semantic Similarity: ${semanticPct}%`}
        />
        <div
          style={{
            width: `${mlPct}%`,
            backgroundColor: "var(--primary)", // Indigo for ML
            transition: "width 0.15s ease",
          }}
          title={`XGBoost ML Score: ${mlPct}%`}
        />
      </div>

      {/* Slider input */}
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={alphaVal}
        onChange={(e) => onChange && onChange(parseFloat(e.target.value))}
        disabled={disabled}
        style={{
          width: "100%",
          cursor: disabled ? "not-allowed" : "pointer",
          accentColor: "var(--primary)",
        }}
        aria-label="Alpha weighting slider"
      />

      {/* Labels below */}
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#3b82f6" }} />
          <span><strong>Semantic (1-α):</strong> {semanticPct}%</span>
          <span style={{ color: "var(--text-muted)" }}>(Sentence-Transformers FAISS)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--primary)" }} />
          <span><strong>XGBoost ML (α):</strong> {mlPct}%</span>
          <span style={{ color: "var(--text-muted)" }}>(TreeSHAP Scorer)</span>
        </div>
      </div>
    </div>
  );
}
