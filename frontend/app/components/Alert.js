"use client";

export default function Alert({ type = "error", msg, onDismiss }) {
  if (!msg) return null;

  const stylesMap = {
    error: {
      bg: "var(--danger-bg)",
      border: "var(--danger-border)",
      text: "var(--danger-text)",
      icon: "⚠️",
    },
    warning: {
      bg: "var(--warning-bg)",
      border: "var(--warning-border)",
      text: "var(--warning-text)",
      icon: "⚡",
    },
    success: {
      bg: "var(--success-bg)",
      border: "var(--success-border)",
      text: "var(--success-text)",
      icon: "✓",
    },
    info: {
      bg: "var(--info-bg)",
      border: "var(--info-border)",
      text: "var(--info-text)",
      icon: "ℹ️",
    },
  };

  const current = stylesMap[type] || stylesMap.info;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "0.625rem",
        backgroundColor: current.bg,
        border: `1px solid ${current.border}`,
        color: current.text,
        padding: "0.75rem 1rem",
        borderRadius: "var(--radius-md)",
        fontSize: "0.875rem",
        lineHeight: "1.375rem",
        marginBottom: "1rem",
      }}
    >
      <span style={{ fontSize: "1rem", flexShrink: 0 }}>{current.icon}</span>
      <div style={{ flex: 1, wordBreak: "break-word" }}>{msg}</div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          style={{
            background: "transparent",
            border: "none",
            color: "currentColor",
            opacity: 0.7,
            cursor: "pointer",
            fontSize: "1rem",
            padding: "0 0.25rem",
            lineHeight: 1,
            marginLeft: "0.5rem",
          }}
          aria-label="Dismiss alert"
        >
          ✕
        </button>
      )}
    </div>
  );
}
