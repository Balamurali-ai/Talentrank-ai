"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import ClusterPlot from "../components/ClusterPlot";
import ClusterSidePanel from "../components/ClusterSidePanel";
import UploadPanel from "../components/UploadPanel";
import Alert from "../components/Alert";
import EmptyState from "../components/EmptyState";
import LoadingState from "../components/LoadingState";

const CLUSTER_COLORS = [
  "#6366f1", // Indigo
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#3b82f6", // Blue
  "#ec4899", // Pink
  "#8b5cf6", // Violet
  "#14b8a6", // Teal
  "#ef4444", // Red
];

export default function ClusteringPage() {
  const router = useRouter();
  const {
    sessionId,
    poolType,
    uploadedResumesCount,
    clusterMap,
    clusterMapState,
    clusterMapError,
    loadClusterMap,
  } = useTalentRank();

  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);

  // If a session exists and cluster map is not yet loaded, load it
  useEffect(() => {
    if (sessionId && clusterMapState === "idle" && !clusterMap) {
      loadClusterMap(sessionId);
    }
  }, [sessionId, clusterMapState, clusterMap, loadClusterMap]);

  const handleSelectCluster = (cId) => {
    setSelectedCluster(cId === selectedCluster ? null : cId);
  };

  const handleSelectPoint = (pt) => {
    setSelectedPoint(pt);
    if (pt.cluster !== undefined && pt.cluster !== null) {
      setSelectedCluster(pt.cluster);
    }
  };

  return (
    <DashboardLayout>
      <PageHeader
        title="Resume Clustering & Embedding Insights"
        description="2D Principal Component Analysis (PCA) projection of SentenceTransformer resume embeddings with TF-IDF keyword extraction."
        badge={{
          text: clusterMap ? `${clusterMap.k} Clusters Identified` : "PCA Projection",
          type: clusterMap ? "primary" : "neutral",
        }}
        actions={
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {sessionId && (
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => loadClusterMap(sessionId)}
                disabled={clusterMapState === "loading"}
              >
                🔄 Refresh Cluster Map
              </button>
            )}
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setShowUploadModal(!showUploadModal)}
            >
              {showUploadModal ? "Close Upload" : "Upload Resume Pool"}
            </button>
          </div>
        }
      />

      {showUploadModal && (
        <div style={{ marginBottom: "1.5rem" }}>
          <UploadPanel onUploaded={(sid) => {
            setShowUploadModal(false);
            loadClusterMap(sid);
          }} />
        </div>
      )}

      {clusterMapError && (
        <Alert type="error" msg={clusterMapError} />
      )}

      {/* Main Clustering Workspace */}
      {clusterMapState === "loading" ? (
        <div className="saas-card" style={{ padding: "3rem" }}>
          <LoadingState message="Executing k-means clustering on embeddings and computing 2D PCA projection..." rows={6} />
        </div>
      ) : clusterMapState === "tooFew" ? (
        <div className="saas-card" style={{ padding: "2.5rem" }}>
          <EmptyState
            icon="⚠️"
            title="Insufficient Resumes for Clustering"
            description="The active session contains fewer than 30 resumes. At least 30 resumes are required to construct a statistically meaningful embedding cluster map."
            action={
              <button
                className="btn btn-primary btn-sm"
                onClick={() => setShowUploadModal(true)}
              >
                Upload Pool (≥ 30 Resumes)
              </button>
            }
          />
        </div>
      ) : clusterMap ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 2fr) minmax(320px, 1fr)",
            gap: "1.5rem",
            alignItems: "start",
          }}
        >
          {/* Left Column: Interactive Scatter Plot & Cluster Filter Chips */}
          <div className="saas-card" style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            {/* Header & Cluster Legend Filter Chips */}
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  2D Embedding Landscape
                </h3>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {clusterMap.points?.length || 0} resumes mapped
                </span>
              </div>

              {/* Cluster Filter Buttons */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", alignItems: "center" }}>
                <button
                  className={`btn btn-xs ${selectedCluster === null ? "btn-primary" : "btn-secondary"}`}
                  onClick={() => setSelectedCluster(null)}
                  style={{ fontSize: "0.75rem", padding: "0.2rem 0.6rem" }}
                >
                  All Clusters
                </button>
                {clusterMap.clusters.map((c) => {
                  const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];
                  const isSelected = selectedCluster === c.cluster;
                  const label = c.keywords?.slice(0, 2).join(", ") || `Cluster ${c.cluster}`;

                  return (
                    <button
                      key={c.cluster}
                      onClick={() => handleSelectCluster(c.cluster)}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "0.35rem",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "var(--radius-pill)",
                        fontSize: "0.75rem",
                        fontWeight: isSelected ? 700 : 500,
                        backgroundColor: isSelected ? "var(--bg-subtle)" : "var(--bg-surface)",
                        border: isSelected ? `2px solid ${color}` : "1px solid var(--border-default)",
                        color: "var(--text-primary)",
                        cursor: "pointer",
                      }}
                    >
                      <span
                        style={{
                          width: "8px",
                          height: "8px",
                          borderRadius: "50%",
                          backgroundColor: color,
                        }}
                      />
                      <span>{label}</span>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                        ({c.size})
                      </span>
                    </button>
                  );
                })}

                {clusterMap.jd_point && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.35rem",
                      padding: "0.2rem 0.6rem",
                      borderRadius: "var(--radius-pill)",
                      fontSize: "0.75rem",
                      fontWeight: 600,
                      backgroundColor: "var(--primary-light)",
                      color: "var(--primary-text)",
                      border: "1px solid var(--primary-border)",
                    }}
                  >
                    ★ Target JD
                  </span>
                )}
              </div>
            </div>

            {/* Interactive Scatter Plot */}
            <div style={{ backgroundColor: "var(--bg-canvas)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
              <ClusterPlot
                data={clusterMap}
                selectedCluster={selectedCluster}
                onSelectCluster={setSelectedCluster}
                onSelectPoint={handleSelectPoint}
              />
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <span>• Click any dot to inspect candidate belonging to cluster</span>
              <span>• Solid outline = Top 10 candidate</span>
            </div>
          </div>

          {/* Right Column: Cluster Detail Side Panel */}
          <div>
            <ClusterSidePanel
              clusterData={clusterMap}
              selectedClusterId={selectedCluster}
              selectedPoint={selectedPoint}
              onClearSelection={() => {
                setSelectedCluster(null);
                setSelectedPoint(null);
              }}
            />
          </div>
        </div>
      ) : (
        /* Empty State when no upload session is active */
        <div className="saas-card" style={{ padding: "2.5rem" }}>
          <EmptyState
            icon="🌐"
            title="Cluster Map Requires Uploaded Resume Pool"
            description="The clustering engine requires a dynamic resume session (minimum 30 resumes) to perform k-means grouping, compute silhouette metrics, and project high-dimensional embeddings via PCA."
            action={
              <button
                className="btn btn-primary"
                onClick={() => setShowUploadModal(true)}
              >
                Upload Resume Pool CSV
              </button>
            }
          />

          <div
            style={{
              marginTop: "2rem",
              padding: "1.25rem",
              backgroundColor: "var(--bg-subtle)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-default)",
              maxWidth: "680px",
              margin: "2rem auto 0",
              fontSize: "0.8125rem",
              color: "var(--text-secondary)",
              lineHeight: 1.5,
            }}
          >
            <h4 style={{ fontSize: "0.875rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
              How Resume Clustering Works in TalentRank AI:
            </h4>
            <ol style={{ paddingLeft: "1.25rem", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              <li>
                <strong>Embedding Generation:</strong> Each resume is transformed into a dense 384-dimensional vector using <code>all-MiniLM-L6-v2</code>.
              </li>
              <li>
                <strong>K-Means Auto-Selection:</strong> Candidate k values (2 through 8) are evaluated with Silhouette Score optimization.
              </li>
              <li>
                <strong>2D PCA Projection:</strong> Dimensionality is reduced to two principal components for spatial scatter visualization.
              </li>
              <li>
                <strong>TF-IDF Keyword Extraction:</strong> Domain-specific stopwords are pruned, and prominent n-grams are extracted per cluster.
              </li>
            </ol>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
