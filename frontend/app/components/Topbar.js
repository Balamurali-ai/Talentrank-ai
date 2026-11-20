"use client";

import { useState, useRef, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";

const BREADCRUMB_MAP = {
  "/dashboard": { title: "Executive Dashboard", section: "Overview" },
  "/ranking": { title: "Talent Ranking Workspace", section: "Screening" },
  "/candidates": { title: "Candidate Explorer", section: "Talent Pool" },
  "/clustering": { title: "Resume Clustering & Insights", section: "Analytics" },
  "/evaluation": { title: "ML Model Evaluation", section: "ML Performance" },
  "/settings": { title: "System & Ranking Settings", section: "Configuration" },
};

export default function Topbar({ onOpenMobile }) {
  const pathname = usePathname();
  const router = useRouter();
  const {
    user,
    logoutUser,
    poolType,
    sessionId,
    uploadedResumesCount,
    notifications,
    hasUnreadNotifications,
    markNotificationsRead,
    health,
  } = useTalentRank();

  const [globalSearch, setGlobalSearch] = useState("");
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const notifRef = useRef(null);
  const userRef = useRef(null);

  const pageInfo = BREADCRUMB_MAP[pathname] || { title: "Dashboard", section: "Overview" };

  // Click outside to close dropdowns
  useEffect(() => {
    function handleClickOutside(e) {
      if (notifRef.current && !notifRef.current.contains(e.target)) {
        setShowNotifications(false);
      }
      if (userRef.current && !userRef.current.contains(e.target)) {
        setShowUserMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (globalSearch.trim()) {
      router.push(`/candidates?q=${encodeURIComponent(globalSearch.trim())}`);
    }
  };

  return (
    <header
      style={{
        height: "var(--topbar-height)",
        backgroundColor: "var(--bg-surface)",
        borderBottom: "1px solid var(--border-default)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 1.5rem",
        position: "sticky",
        top: 0,
        zIndex: 30,
      }}
    >
      {/* Left: Mobile hamburger + Breadcrumbs */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <button
          onClick={onOpenMobile}
          style={{
            display: "none",
            background: "transparent",
            border: "none",
            fontSize: "1.25rem",
            color: "var(--text-primary)",
            cursor: "pointer",
            padding: "0.25rem",
          }}
          className="mobile-hamburger"
          aria-label="Open navigation menu"
        >
          ☰
        </button>

        <style jsx>{`
          @media (max-width: 1024px) {
            .mobile-hamburger {
              display: block !important;
            }
          }
        `}</style>

        <div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <span>TalentRank AI</span>
            <span>/</span>
            <span>{pageInfo.section}</span>
          </div>
          <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>
            {pageInfo.title}
          </div>
        </div>
      </div>

      {/* Right: Global Search + Pool Badge + Notifications + User Menu */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        
        {/* Global Candidate Search */}
        <form onSubmit={handleSearchSubmit} style={{ position: "relative" }} className="topbar-search">
          <input
            type="text"
            placeholder="Search candidate ID..."
            value={globalSearch}
            onChange={(e) => setGlobalSearch(e.target.value)}
            className="form-input"
            style={{
              paddingLeft: "2rem",
              paddingTop: "0.35rem",
              paddingBottom: "0.35rem",
              fontSize: "0.8125rem",
              width: "180px",
              height: "34px",
            }}
          />
          <span
            style={{
              position: "absolute",
              left: "0.625rem",
              top: "50%",
              transform: "translateY(-50%)",
              fontSize: "0.8125rem",
              color: "var(--text-muted)",
              pointerEvents: "none",
            }}
          >
            🔍
          </span>
        </form>

        <style jsx>{`
          @media (max-width: 640px) {
            .topbar-search {
              display: none !important;
            }
          }
        `}</style>

        {/* Active Pool Indicator Badge */}
        <div style={{ display: "flex", alignItems: "center" }}>
          {poolType === "uploaded" && sessionId ? (
            <span
              className="badge badge-success"
              title={`Uploaded Session: ${sessionId}`}
              style={{ fontSize: "0.75rem" }}
            >
              Session: {uploadedResumesCount} Resumes
            </span>
          ) : (
            <span
              className="badge badge-neutral"
              title="Default Kaggle Pre-embedded Dataset"
              style={{ fontSize: "0.75rem" }}
            >
              Default Pool: 2,484
            </span>
          )}
        </div>

        {/* Notifications Popover */}
        <div style={{ position: "relative" }} ref={notifRef}>
          <button
            onClick={() => {
              setShowNotifications(!showNotifications);
              markNotificationsRead();
            }}
            style={{
              background: "transparent",
              border: "1px solid var(--border-default)",
              borderRadius: "var(--radius-md)",
              width: "36px",
              height: "36px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              position: "relative",
              fontSize: "1rem",
            }}
            aria-label="View notifications"
          >
            🔔
            {hasUnreadNotifications && (
              <span
                style={{
                  position: "absolute",
                  top: "6px",
                  right: "6px",
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  backgroundColor: "var(--primary)",
                }}
              />
            )}
          </button>

          {showNotifications && (
            <div
              style={{
                position: "absolute",
                right: 0,
                top: "calc(100% + 8px)",
                width: "320px",
                backgroundColor: "var(--bg-surface)",
                borderRadius: "var(--radius-lg)",
                boxShadow: "var(--shadow-dropdown)",
                border: "1px solid var(--border-default)",
                zIndex: 50,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  padding: "0.75rem 1rem",
                  borderBottom: "1px solid var(--border-default)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span style={{ fontSize: "0.875rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  Activity & System Alerts
                </span>
                <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                  {notifications.length} events
                </span>
              </div>

              <div style={{ maxHeight: "280px", overflowY: "auto" }}>
                {notifications.map((n) => (
                  <div
                    key={n.id}
                    style={{
                      padding: "0.75rem 1rem",
                      borderBottom: "1px solid var(--border-subtle)",
                      fontSize: "0.8125rem",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                      <strong style={{ color: "var(--text-primary)" }}>{n.title}</strong>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{n.time}</span>
                    </div>
                    <div style={{ color: "var(--text-secondary)", lineHeight: 1.3 }}>
                      {n.message}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* User Profile Avatar Dropdown */}
        <div style={{ position: "relative" }} ref={userRef}>
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              background: "transparent",
              border: "1px solid var(--border-default)",
              borderRadius: "var(--radius-md)",
              padding: "0.25rem 0.5rem 0.25rem 0.25rem",
              cursor: "pointer",
            }}
            aria-label="User menu"
          >
            <div
              style={{
                width: "28px",
                height: "28px",
                borderRadius: "50%",
                backgroundColor: "var(--primary)",
                color: "#ffffff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: 700,
                fontSize: "0.75rem",
              }}
            >
              {user?.username ? user.username[0].toUpperCase() : "R"}
            </div>
            <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--text-primary)" }}>
              {user?.username || "Recruiter"}
            </span>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>▼</span>
          </button>

          {showUserMenu && (
            <div
              style={{
                position: "absolute",
                right: 0,
                top: "calc(100% + 8px)",
                width: "200px",
                backgroundColor: "var(--bg-surface)",
                borderRadius: "var(--radius-md)",
                boxShadow: "var(--shadow-dropdown)",
                border: "1px solid var(--border-default)",
                zIndex: 50,
                padding: "0.5rem 0",
              }}
            >
              <div style={{ padding: "0.5rem 1rem", borderBottom: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.8125rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  {user?.username || "Recruiter Admin"}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                  Academic Evaluator
                </div>
              </div>

              <div style={{ padding: "0.25rem 0" }}>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    router.push("/settings");
                  }}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "0.5rem 1rem",
                    fontSize: "0.8125rem",
                    color: "var(--text-primary)",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  ⚙️ Settings & System
                </button>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    router.push("/evaluation");
                  }}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "0.5rem 1rem",
                    fontSize: "0.8125rem",
                    color: "var(--text-primary)",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  📈 Model Evaluation
                </button>
              </div>

              <div style={{ borderTop: "1px solid var(--border-subtle)", padding: "0.25rem 0" }}>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    logoutUser();
                    router.push("/login");
                  }}
                  style={{
                    width: "100%",
                    textAlign: "left",
                    padding: "0.5rem 1rem",
                    fontSize: "0.8125rem",
                    color: "var(--danger)",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  Sign Out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
