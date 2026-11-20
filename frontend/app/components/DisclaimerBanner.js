"use client";

import { useState } from "react";

export default function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <div
      style={{
        backgroundColor: "#fffbeb",
        borderBottom: "1px solid #fde68a",
        padding: "0.5rem 1.25rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontSize: "0.8125rem",
        color: "#92400e",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span style={{ fontSize: "1rem" }}>⚠️</span>
        <span>
          <strong>Academic Research Prototype:</strong> Candidate scores represent ranking signals relative to the provided job description and proxy labels, not hiring probabilities or validated predictions of job performance.
        </span>
      </div>
      <button
        onClick={() => setDismissed(true)}
        style={{
          background: "none",
          border: "none",
          color: "#b45309",
          cursor: "pointer",
          fontSize: "1rem",
          padding: "0 0.25rem",
          lineHeight: 1,
        }}
        aria-label="Dismiss disclaimer"
      >
        ✕
      </button>
    </div>
  );
}
