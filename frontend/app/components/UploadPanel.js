"use client";

import { useState, useCallback } from "react";
import { useTalentRank } from "../context/TalentRankContext";
import Alert from "./Alert";

function formatBytes(bytes) {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function UploadPanel({ onUploaded }) {
  const {
    sessionId,
    poolType,
    uploadedFileName,
    uploadedResumesCount,
    uploadProgress,
    isUploading,
    uploadError,
    uploadResumeFile,
    useDefaultPool,
  } = useTalentRank();

  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [localError, setLocalError] = useState("");

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith(".csv")) {
      setSelectedFile(file);
      setLocalError("");
    } else {
      setLocalError("Please select a valid .csv file.");
    }
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.name.toLowerCase().endsWith(".csv")) {
        setSelectedFile(file);
        setLocalError("");
      } else {
        setLocalError("Please select a valid .csv file.");
      }
    }
  };

  const handleStartUpload = async () => {
    if (!selectedFile) return;
    try {
      setLocalError("");
      const sid = await uploadResumeFile(selectedFile);
      if (onUploaded) onUploaded(sid);
    } catch (err) {
      setLocalError(err.message || "Upload failed");
    }
  };

  return (
    <div className="saas-card" style={{ padding: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <div>
          <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--text-primary)", margin: 0 }}>
            Resume Candidate Pool
          </h3>
          <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
            Choose the candidate pool to search and rank against.
          </p>
        </div>
        <div>
          {poolType === "uploaded" && sessionId ? (
            <span className="badge badge-success">
              ✓ Custom Pool ({uploadedResumesCount} Resumes)
            </span>
          ) : (
            <span className="badge badge-neutral">
              Default Pool (2,484 Resumes)
            </span>
          )}
        </div>
      </div>

      {/* Pool Switcher Buttons */}
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
        <button
          type="button"
          className={`btn btn-sm ${poolType === "default" ? "btn-primary" : "btn-secondary"}`}
          onClick={useDefaultPool}
          disabled={isUploading}
        >
          Use Default Pre-Indexed Pool (2,484)
        </button>
        {poolType === "uploaded" && sessionId && (
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={useDefaultPool}
          >
            Switch back to Default Pool
          </button>
        )}
      </div>

      {/* Active Pool Info Alert if Uploaded */}
      {poolType === "uploaded" && sessionId && (
        <div
          style={{
            backgroundColor: "var(--success-bg)",
            border: "1px solid var(--success-border)",
            borderRadius: "var(--radius-md)",
            padding: "0.75rem 1rem",
            marginBottom: "1rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "0.8125rem",
            color: "var(--success-text)",
          }}
        >
          <div>
            <strong>Active Session:</strong> {sessionId.slice(0, 12)}... ({uploadedFileName || "Uploaded CSV"}, {uploadedResumesCount} candidates indexed with FAISS)
          </div>
          <span style={{ fontSize: "0.75rem", opacity: 0.85 }}>Session TTL: ~2 hours</span>
        </div>
      )}

      {/* Upload Drag & Drop Area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragOver ? "var(--primary)" : "var(--border-strong)"}`,
          borderRadius: "var(--radius-lg)",
          backgroundColor: dragOver ? "var(--primary-light)" : "var(--bg-canvas)",
          padding: "1.75rem 1.5rem",
          textAlign: "center",
          transition: "all 0.2s ease",
          cursor: "pointer",
          position: "relative",
          marginBottom: "1rem",
        }}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          style={{
            position: "absolute",
            inset: 0,
            opacity: 0,
            cursor: "pointer",
            width: "100%",
            height: "100%",
          }}
          disabled={isUploading}
          aria-label="Upload CSV file"
        />

        <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>📄</div>
        <div style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--text-primary)" }}>
          {selectedFile ? selectedFile.name : "Drop your resume CSV here or browse"}
        </div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
          {selectedFile
            ? `${formatBytes(selectedFile.size)} selected`
            : "Required columns: ID, Resume_str. Optional: Category. Max 2,000 rows."}
        </div>
      </div>

      {/* Progress Bar */}
      {uploadProgress && (
        <div style={{ marginBottom: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>
            <span>
              {uploadProgress.status === "processing"
                ? `Indexing with SentenceTransformers (MiniLM) & FAISS: ${uploadProgress.processed} / ${uploadProgress.total}...`
                : `Completed — ${uploadProgress.processed} resumes indexed`}
            </span>
            <span className="tabular-nums font-semibold">
              {uploadProgress.total ? `${Math.round((uploadProgress.processed / uploadProgress.total) * 100)}%` : ""}
            </span>
          </div>
          <div style={{ height: "6px", backgroundColor: "var(--bg-subtle)", borderRadius: "var(--radius-pill)", overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                width: uploadProgress.total ? `${Math.round((uploadProgress.processed / uploadProgress.total) * 100)}%` : "0%",
                backgroundColor: "var(--primary)",
                borderRadius: "var(--radius-pill)",
                transition: "width 0.3s ease",
              }}
            />
          </div>
        </div>
      )}

      {(localError || uploadError) && (
        <Alert type="error" msg={localError || uploadError} onDismiss={() => setLocalError("")} />
      )}

      {selectedFile && !isUploading && (
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => setSelectedFile(null)}
          >
            Clear
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={handleStartUpload}
          >
            Upload & Index Resumes
          </button>
        </div>
      )}
    </div>
  );
}
