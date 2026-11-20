"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import ScoreBreakdown from "../components/ScoreBreakdown";
import EmptyState from "../components/EmptyState";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function DashboardPage() {
  const router = useRouter();
  const {
    poolType,
    sessionId,
    uploadedResumesCount,
    results,
    rankingHistory,
    modelData,
    health,
    setSelectedCandidate,
  } = useTalentRank();

  // Compute stats from current results
  const currentStats = useMemo(() => {
    if (!results || results.length === 0) return null;
    const count = results.length;
    const avgFinal = results.reduce((acc, r) => acc + (r.final_score || 0), 0) / count;
    const avgSem = results.reduce((acc, r) => acc + (r.semantic_sim || 0), 0) / count;
    const avgML = results.reduce((acc, r) => acc + (r.ML_score || 0), 0) / count;
    const topFinal = results[0]?.final_score || 0;
    return { count, avgFinal, avgSem, avgML, topFinal };
  }, [results]);

  // Candidate distribution by category from active results
  const categoryData = useMemo(() => {
    if (!results || results.length === 0) return [];
    const counts = {};
    results.forEach((r) => {
      const cat = r.category || "Unassigned";
      counts[cat] = (counts[cat] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count);
  }, [results]);

  const COLORS = ["#4f46e5", "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"];

  return (
    <DashboardLayout>
      <PageHeader
        title="Recruitment Analytics Dashboard"
        description="Real-time candidate ranking signals, semantic resume indexing, and machine learning pipeline telemetry."
        badge={{ text: "Active", type: "success" }}
        actions={
          <>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => router.push("/candidates")}
            >
              👥 Explore Candidates
            </button>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => router.push("/ranking")}
            >
              ⚡ Start New Ranking
            </button>
          </>
        }
      />

      {/* KPI Cards Row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "1rem",
          marginBottom: "1.75rem",
        }}
      >
        {/* Total Resumes */}
        <StatCard
          title="Resume Pool"
          value={uploadedResumesCount ? uploadedResumesCount.toLocaleString() : "2,484"}
          subtext={
            poolType === "uploaded" && sessionId
              ? `Custom CSV (${sessionId.slice(0, 8)}...)`
              : "Default Pre-Indexed Pool"
          }
          icon="📄"
          badge={{
            text: poolType === "uploaded" ? "Custom Session" : "Default Pool",
            type: poolType === "uploaded" ? "success" : "neutral",
          }}
          onClick={() => router.push("/ranking")}
        />

        {/* Total Ranking Sessions */}
        <StatCard
          title="Ranking Sessions"
          value={rankingHistory.length}
          subtext={
            rankingHistory.length > 0
              ? `Latest: ${rankingHistory[0].job_title || "Search"}`
              : "No queries submitted yet"
          }
          icon="⚡"
          badge={{
            text: rankingHistory.length > 0 ? "Active" : "Ready",
            type: "primary",
          }}
          onClick={() => router.push("/ranking")}
        />

        {/* Average Candidate Similarity */}
        <StatCard
          title="Avg Candidate Sim"
          value={currentStats ? currentStats.avgSem.toFixed(3) : "—"}
          subtext={
            currentStats
              ? `Top candidate: ${currentStats.topFinal.toFixed(3)} final score`
              : "Calculated upon ranking execution"
          }
          icon="🎯"
          badge={{
            text: currentStats ? "Top 10" : "Pending",
            type: currentStats ? "info" : "neutral",
          }}
        />

        {/* Model Pipeline Status */}
        <StatCard
          title="Model Pipeline"
          value={health?.models_loaded ? "Online" : "Loaded"}
          subtext={
            modelData?.algorithm
              ? `${modelData.algorithm} (F1: ${modelData.f1?.toFixed(3) || "0.508"})`
              : "MiniLM-L6-v2 + XGBoost"
          }
          icon="🧠"
          badge={{
            text: "XGBoost + FAISS",
            type: "success",
          }}
          onClick={() => router.push("/evaluation")}
        />
      </div>

      {/* Main Grid: Category Distribution & Quick Actions */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(450px, 1fr))",
          gap: "1.5rem",
          marginBottom: "1.75rem",
        }}
      >
        {/* Category Breakdown Card */}
        <div className="saas-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
            <div>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                Top Candidates Category Breakdown
              </h3>
              <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                Distribution of current top candidates across professional categories.
              </p>
            </div>
            {categoryData.length > 0 && (
              <span className="badge badge-primary">{categoryData.length} Categories</span>
            )}
          </div>

          {categoryData.length > 0 ? (
            <div style={{ height: "260px", width: "100%" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fontSize: 11, fill: "var(--text-secondary)" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--bg-surface)",
                      borderRadius: "var(--radius-md)",
                      border: "1px solid var(--border-default)",
                      fontSize: "0.75rem",
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ padding: "2rem 0" }}>
              <EmptyState
                icon="📊"
                title="No Active Candidate Results"
                description="Run a job description search in the Talent Ranking workspace to generate candidate distributions."
                action={
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => router.push("/ranking")}
                  >
                    Go to Talent Ranking
                  </button>
                }
              />
            </div>
          )}
        </div>

        {/* Quick Launch & System Status */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          
          {/* Quick Actions Card */}
          <div className="saas-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
              Quick Workspaces
            </h3>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
              Direct access to primary recruitment intelligence tools.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              <button
                className="btn btn-secondary"
                style={{
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: "0.35rem",
                  textAlign: "left",
                  height: "auto",
                }}
                onClick={() => router.push("/ranking")}
              >
                <span style={{ fontSize: "1.25rem" }}>⚡</span>
                <span style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)" }}>
                  Talent Ranking
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Submit JD and rank top 10 candidates
                </span>
              </button>

              <button
                className="btn btn-secondary"
                style={{
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: "0.35rem",
                  textAlign: "left",
                  height: "auto",
                }}
                onClick={() => router.push("/candidates")}
              >
                <span style={{ fontSize: "1.25rem" }}>👥</span>
                <span style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)" }}>
                  Candidate Explorer
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Filter, inspect, and compare profiles
                </span>
              </button>

              <button
                className="btn btn-secondary"
                style={{
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: "0.35rem",
                  textAlign: "left",
                  height: "auto",
                }}
                onClick={() => router.push("/clustering")}
              >
                <span style={{ fontSize: "1.25rem" }}>🌐</span>
                <span style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)" }}>
                  Cluster Map
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  2D PCA embedding landscape & keywords
                </span>
              </button>

              <button
                className="btn btn-secondary"
                style={{
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  gap: "0.35rem",
                  textAlign: "left",
                  height: "auto",
                }}
                onClick={() => router.push("/evaluation")}
              >
                <span style={{ fontSize: "1.25rem" }}>📈</span>
                <span style={{ fontWeight: 600, fontSize: "0.875rem", color: "var(--text-primary)" }}>
                  Model Evaluation
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  NDCG@10, F1, bootstrap CIs & ablation
                </span>
              </button>
            </div>
          </div>

          {/* Model Telemetry Summary */}
          <div className="saas-card" style={{ padding: "1.25rem 1.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
              <span style={{ fontSize: "0.8125rem", fontWeight: 700, color: "var(--text-primary)" }}>
                Pipeline Telemetry
              </span>
              <span className="badge badge-success">FAISS + MiniLM</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", fontSize: "0.75rem" }}>
              <div style={{ backgroundColor: "var(--bg-subtle)", padding: "0.6rem", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-muted)", display: "block" }}>Classifier F1</span>
                <strong style={{ fontSize: "0.95rem", color: "var(--text-primary)" }}>
                  {modelData?.f1 ? modelData.f1.toFixed(3) : "0.508"}
                </strong>
              </div>

              <div style={{ backgroundColor: "var(--bg-subtle)", padding: "0.6rem", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-muted)", display: "block" }}>ROC-AUC</span>
                <strong style={{ fontSize: "0.95rem", color: "var(--text-primary)" }}>
                  {modelData?.roc_auc ? modelData.roc_auc.toFixed(3) : "0.811"}
                </strong>
              </div>

              <div style={{ backgroundColor: "var(--bg-subtle)", padding: "0.6rem", borderRadius: "var(--radius-sm)" }}>
                <span style={{ color: "var(--text-muted)", display: "block" }}>Features</span>
                <strong style={{ fontSize: "0.95rem", color: "var(--text-primary)" }}>
                  {modelData?.features?.length || 5} Engineered
                </strong>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Recent Ranking Activity Section */}
      <div className="saas-card" style={{ padding: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
              Recent Ranking Activity
            </h3>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
              History of candidate searches executed during this session.
            </p>
          </div>
          {rankingHistory.length > 0 && (
            <span className="badge badge-neutral">{rankingHistory.length} Sessions</span>
          )}
        </div>

        {rankingHistory.length > 0 ? (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Job Title</th>
                  <th>Pool Used</th>
                  <th>Candidates Ranked</th>
                  <th>Top Score</th>
                  <th>Avg Score</th>
                  <th>Alpha (α)</th>
                  <th>Timestamp</th>
                  <th style={{ textAlign: "right" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {rankingHistory.map((item) => (
                  <tr key={item.id}>
                    <td style={{ fontWeight: 600 }}>{item.job_title}</td>
                    <td>
                      <span className={`badge ${item.poolType === "uploaded" ? "badge-success" : "badge-neutral"}`}>
                        {item.poolType === "uploaded" ? "Custom CSV" : "Default 2,484"}
                      </span>
                    </td>
                    <td>{item.count} returned</td>
                    <td className="tabular-nums font-semibold" style={{ color: "var(--primary)" }}>
                      {item.topScore}
                    </td>
                    <td className="tabular-nums">{item.avgScore}</td>
                    <td className="tabular-nums">α = {item.alpha}</td>
                    <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => router.push("/ranking")}
                      >
                        Open Workspace →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: "1.5rem 0" }}>
            <EmptyState
              icon="⏱️"
              title="No Ranking History Yet"
              description="Searches performed in the Talent Ranking workspace will be logged here for quick reference."
              action={
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => router.push("/ranking")}
                >
                  Start First Ranking Search
                </button>
              }
            />
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
