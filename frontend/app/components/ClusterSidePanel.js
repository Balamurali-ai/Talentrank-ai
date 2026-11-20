"use client";

export default function ClusterSidePanel({ clusterData, selectedClusterId, selectedPoint, onClearSelection }) {
  if (!clusterData) return null;

  const {
    k,
    clusters = [],
    points = [],
    jd_nearest_cluster,
    pca_variance_explained,
    adjusted_rand_index,
    normalized_mutual_info,
    silhouette,
  } = clusterData;

  const currentCluster = clusters.find((c) => c.cluster === selectedClusterId);
  const clusterPoints = selectedClusterId !== null
    ? points.filter((p) => p.cluster === selectedClusterId)
    : [];

  const isNearestToJd = jd_nearest_cluster && jd_nearest_cluster.cluster === selectedClusterId;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1.25rem",
        height: "100%",
      }}
    >
      {/* Projection Disclaimer & Info Box */}
      <div
        style={{
          backgroundColor: "var(--info-bg)",
          border: "1px solid var(--info-border)",
          borderRadius: "var(--radius-md)",
          padding: "0.875rem 1rem",
          fontSize: "0.75rem",
          color: "var(--info-text)",
          lineHeight: 1.45,
        }}
      >
        <strong>Dimensionality Reduction Note:</strong> This plot represents a 2D PCA projection of the 384-dimensional SentenceTransformer (MiniLM) embedding space.
        It captures <strong>{Math.round((pca_variance_explained || 0) * 100)}%</strong> of embedding variance. Distances are approximate.
      </div>

      {/* Selected Point Inspector (if user clicked a dot) */}
      {selectedPoint && (
        <div className="saas-card" style={{ padding: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.8125rem", fontWeight: 700, color: "var(--text-primary)" }}>
              Inspected Candidate #{selectedPoint.id}
            </span>
            <span className="badge badge-primary">Cluster {selectedPoint.cluster}</span>
          </div>
          {selectedPoint.category && (
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.25rem" }}>
              Category: <strong>{selectedPoint.category}</strong>
            </div>
          )}
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Coordinates: ({selectedPoint.x.toFixed(2)}, {selectedPoint.y.toFixed(2)})
          </div>
        </div>
      )}

      {/* Cluster Details */}
      {currentCluster ? (
        <div className="saas-card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <h4 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  Cluster {currentCluster.cluster}
                </h4>
                {isNearestToJd && (
                  <span className="badge badge-success" style={{ fontSize: "0.7rem" }}>
                    ★ Closest to JD
                  </span>
                )}
              </div>
              <div style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginTop: "0.2rem" }}>
                {currentCluster.size} candidates ({Math.round((currentCluster.size / points.length) * 100)}% of pool)
              </div>
            </div>

            <button
              className="btn btn-ghost btn-sm"
              onClick={onClearSelection}
              style={{ fontSize: "0.75rem" }}
            >
              Show All
            </button>
          </div>

          {/* Top Keywords */}
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
              Top Defining TF-IDF Keywords:
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
              {currentCluster.keywords && currentCluster.keywords.map((kw, i) => (
                <span key={i} className="badge badge-neutral" style={{ fontSize: "0.75rem" }}>
                  {kw}
                </span>
              ))}
            </div>
          </div>

          {/* JD Overlap Info if applicable */}
          {isNearestToJd && jd_nearest_cluster && (
            <div
              style={{
                backgroundColor: "var(--primary-light)",
                border: "1px solid var(--primary-border)",
                borderRadius: "var(--radius-sm)",
                padding: "0.65rem 0.75rem",
                fontSize: "0.75rem",
                color: "var(--primary-text)",
                marginBottom: "1rem",
              }}
            >
              <strong>{jd_nearest_cluster.top10_in_cluster} of the Top 10</strong> ranked candidates reside in this cluster, confirming high semantic alignment with your JD requirements.
            </div>
          )}

          {/* Candidates in Cluster List */}
          <div>
            <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
              Candidates in this cluster ({clusterPoints.length}):
            </div>
            <div
              style={{
                maxHeight: "180px",
                overflowY: "auto",
                border: "1px solid var(--border-default)",
                borderRadius: "var(--radius-sm)",
                padding: "0.5rem",
                backgroundColor: "var(--bg-subtle)",
                display: "flex",
                flexWrap: "wrap",
                gap: "0.35rem",
              }}
            >
              {clusterPoints.slice(0, 80).map((pt) => (
                <span
                  key={pt.id}
                  style={{
                    fontSize: "0.7rem",
                    padding: "0.15rem 0.4rem",
                    backgroundColor: "var(--bg-surface)",
                    borderRadius: "var(--radius-xs)",
                    border: "1px solid var(--border-default)",
                    fontFamily: "monospace",
                  }}
                >
                  #{pt.id}
                </span>
              ))}
              {clusterPoints.length > 80 && (
                <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", alignSelf: "center" }}>
                  +{clusterPoints.length - 80} more
                </span>
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Summary of All Clusters */
        <div className="saas-card" style={{ padding: "1.25rem" }}>
          <h4 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
            Cluster Overview ({k} Clusters)
          </h4>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
            Click on any cluster below or point in the scatter plot to inspect its composition and keywords.
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            {clusters.map((c) => (
              <div
                key={c.cluster}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.5rem 0.75rem",
                  borderRadius: "var(--radius-sm)",
                  backgroundColor: "var(--bg-subtle)",
                  fontSize: "0.8125rem",
                  border: "1px solid var(--border-default)",
                }}
              >
                <div>
                  <strong>Cluster {c.cluster}</strong>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                    {c.keywords?.slice(0, 3).join(", ") || "No keywords"}
                  </div>
                </div>
                <span className="badge badge-neutral">
                  {c.size} resumes
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cluster Metrics Card */}
      <div className="saas-card" style={{ padding: "1rem 1.25rem" }}>
        <h5 style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: "0.75rem" }}>
          Cluster Validation Metrics
        </h5>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.75rem" }}>
          <div>
            <span style={{ color: "var(--text-muted)" }}>Silhouette Score:</span>
            <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{silhouette ?? "—"}</div>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>PCA Variance:</span>
            <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>
              {pca_variance_explained ? `${Math.round(pca_variance_explained * 100)}%` : "—"}
            </div>
          </div>
          {adjusted_rand_index !== undefined && (
            <div>
              <span style={{ color: "var(--text-muted)" }}>Adjusted Rand (ARI):</span>
              <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{adjusted_rand_index}</div>
            </div>
          )}
          {normalized_mutual_info !== undefined && (
            <div>
              <span style={{ color: "var(--text-muted)" }}>Mutual Info (NMI):</span>
              <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{normalized_mutual_info}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
