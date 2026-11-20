"use client";

export default function StatCard({ title, value, subtext, icon, badge, trend, onClick }) {
  return (
    <div
      className="saas-card"
      onClick={onClick}
      style={{
        padding: "1.25rem 1.5rem",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        cursor: onClick ? "pointer" : "default",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
          {title}
        </span>
        {icon && (
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "var(--radius-md)",
              backgroundColor: "var(--primary-light)",
              color: "var(--primary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.1rem",
            }}
          >
            {icon}
          </div>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem", marginBottom: "0.35rem" }}>
        <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }} className="tabular-nums">
          {value}
        </span>
        {badge && (
          <span className={`badge ${badge.type ? `badge-${badge.type}` : 'badge-primary'}`}>
            {badge.text}
          </span>
        )}
      </div>

      {(subtext || trend) && (
        <div style={{ fontSize: "0.8125rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          {trend && (
            <span style={{ color: trend.positive ? "var(--success)" : "var(--danger)", fontWeight: 600 }}>
              {trend.positive ? "↑" : "↓"} {trend.text}
            </span>
          )}
          <span>{subtext}</span>
        </div>
      )}
    </div>
  );
}
