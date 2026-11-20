"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: "📊" },
  { href: "/ranking", label: "Talent Ranking", icon: "⚡" },
  { href: "/candidates", label: "Candidate Explorer", icon: "👥" },
  { href: "/clustering", label: "Clustering & Insights", icon: "🌐" },
  { href: "/evaluation", label: "Model Evaluation", icon: "📈" },
  { href: "/settings", label: "Settings", icon: "⚙️" },
];

export default function Sidebar({ collapsed, onToggleCollapse, mobileOpen, onCloseMobile }) {
  const pathname = usePathname();
  const { user, logoutUser, results, comparisonCandidates } = useTalentRank();

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(15, 23, 42, 0.4)",
            backdropFilter: "blur(2px)",
            zIndex: 40,
          }}
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar Container */}
      <aside
        style={{
          width: collapsed ? "var(--sidebar-collapsed-width)" : "var(--sidebar-width)",
          minWidth: collapsed ? "var(--sidebar-collapsed-width)" : "var(--sidebar-width)",
          height: "100vh",
          position: "sticky",
          top: 0,
          backgroundColor: "var(--bg-surface)",
          borderRight: "1px solid var(--border-default)",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          zIndex: 45,
          transition: "width var(--transition-normal), transform var(--transition-normal)",
          transform: mobileOpen ? "translateX(0)" : undefined,
          overflow: "hidden",
        }}
        className="sidebar-responsive"
      >
        <style jsx>{`
          @media (max-width: 1024px) {
            .sidebar-responsive {
              position: fixed !important;
              left: 0;
              top: 0;
              bottom: 0;
              transform: ${mobileOpen ? "translateX(0)" : "translateX(-100%)"};
              width: 260px !important;
              min-width: 260px !important;
              box-shadow: ${mobileOpen ? "var(--shadow-modal)" : "none"};
            }
          }
        `}</style>

        {/* Top: Logo & Brand */}
        <div>
          <div
            style={{
              height: "var(--topbar-height)",
              padding: collapsed ? "0 1rem" : "0 1.25rem",
              display: "flex",
              alignItems: "center",
              justifyContent: collapsed ? "center" : "space-between",
              borderBottom: "1px solid var(--border-default)",
            }}
          >
            <Link
              href="/dashboard"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.625rem",
                textDecoration: "none",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "var(--radius-md)",
                  backgroundColor: "var(--primary)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#ffffff",
                  fontWeight: 800,
                  fontSize: "1rem",
                  flexShrink: 0,
                  boxShadow: "0 2px 4px rgba(79, 70, 229, 0.3)",
                }}
              >
                TR
              </div>
              {!collapsed && (
                <div style={{ display: "flex", alignItems: "baseline", gap: "2px" }}>
                  <span style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
                    TalentRank
                  </span>
                  <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "var(--primary)", marginLeft: "2px" }}>
                    AI
                  </span>
                </div>
              )}
            </Link>

            {!collapsed && onToggleCollapse && (
              <button
                onClick={onToggleCollapse}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                  padding: "0.25rem",
                  fontSize: "0.9rem",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                title="Collapse sidebar"
              >
                ‹
              </button>
            )}
          </div>

          {/* Navigation Links */}
          <nav style={{ padding: "0.75rem 0.5rem", display: "flex", flexDirection: "column", gap: "0.25rem" }}>
            {NAV_ITEMS.map((item) => {
              const isActive = pathname === item.href;
              const hasCandidateResults = item.href === "/candidates" && results.length > 0;
              const hasCompareCandidates = item.href === "/candidates" && comparisonCandidates.length > 0;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={onCloseMobile}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.75rem",
                    padding: collapsed ? "0.625rem 0" : "0.625rem 0.875rem",
                    justifyContent: collapsed ? "center" : "flex-start",
                    borderRadius: "var(--radius-md)",
                    fontSize: "0.875rem",
                    fontWeight: isActive ? 600 : 500,
                    color: isActive ? "var(--primary)" : "var(--text-secondary)",
                    backgroundColor: isActive ? "var(--primary-light)" : "transparent",
                    textDecoration: "none",
                    transition: "all var(--transition-fast)",
                    position: "relative",
                  }}
                  title={collapsed ? item.label : undefined}
                >
                  <span style={{ fontSize: "1.1rem", flexShrink: 0 }}>{item.icon}</span>
                  {!collapsed && (
                    <span style={{ flex: 1, whiteSpace: "nowrap" }}>{item.label}</span>
                  )}

                  {!collapsed && hasCandidateResults && (
                    <span className="badge badge-primary" style={{ fontSize: "0.7rem", padding: "0.1rem 0.4rem" }}>
                      {results.length}
                    </span>
                  )}

                  {collapsed && hasCandidateResults && (
                    <span
                      style={{
                        position: "absolute",
                        top: "6px",
                        right: "12px",
                        width: "8px",
                        height: "8px",
                        borderRadius: "50%",
                        backgroundColor: "var(--primary)",
                      }}
                    />
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Bottom Section: Collapse Toggle & User Profile */}
        <div style={{ borderTop: "1px solid var(--border-default)", padding: "0.75rem" }}>
          {/* Collapse Toggle for Collapsed state */}
          {collapsed && (
            <button
              onClick={onToggleCollapse}
              style={{
                width: "100%",
                padding: "0.5rem 0",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "transparent",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                fontSize: "1.1rem",
              }}
              title="Expand sidebar"
            >
              ›
            </button>
          )}

          {/* User Profile Mini Card */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.625rem",
              padding: collapsed ? "0.5rem 0" : "0.5rem",
              justifyContent: collapsed ? "center" : "flex-start",
              borderRadius: "var(--radius-md)",
              backgroundColor: "var(--bg-subtle)",
            }}
          >
            <div
              style={{
                width: "32px",
                height: "32px",
                borderRadius: "50%",
                backgroundColor: "var(--primary)",
                color: "#ffffff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: 600,
                fontSize: "0.8125rem",
                flexShrink: 0,
              }}
            >
              {user?.username ? user.username[0].toUpperCase() : "R"}
            </div>

            {!collapsed && (
              <div style={{ flex: 1, minWidth: 0, overflow: "hidden" }}>
                <div style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {user?.username || "Recruiter"}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                  Recruiter Admin
                </div>
              </div>
            )}

            {!collapsed && (
              <button
                onClick={logoutUser}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                  padding: "0.25rem",
                  fontSize: "0.875rem",
                }}
                title="Sign out"
              >
                ⏻
              </button>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
