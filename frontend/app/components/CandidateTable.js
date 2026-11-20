"use client";

import { useState, useMemo } from "react";
import { useTalentRank } from "../context/TalentRankContext";
import ScoreBreakdown from "./ScoreBreakdown";
import SkillMatchDisplay from "./SkillMatchDisplay";

export default function CandidateTable({ candidates = [], showFilters = true }) {
  const {
    setSelectedCandidate,
    toggleComparison,
    isCompared,
    comparisonCandidates,
    setIsCompareModalOpen,
  } = useTalentRank();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [sortField, setSortField] = useState("final_score");
  const [sortAsc, setSortAsc] = useState(false);

  // Extract unique categories
  const categories = useMemo(() => {
    const set = new Set();
    candidates.forEach((c) => {
      if (c.category) set.add(c.category);
    });
    return Array.from(set).sort();
  }, [candidates]);

  // Filter and sort candidates
  const processedCandidates = useMemo(() => {
    return candidates
      .filter((c) => {
        const matchesCat = selectedCategory === "all" || c.category === selectedCategory;
        const matchesSearch =
          !searchQuery.trim() ||
          String(c.id).toLowerCase().includes(searchQuery.toLowerCase()) ||
          (c.category && c.category.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (c.explanation?.matched_skills &&
            c.explanation.matched_skills.some((s) => s.toLowerCase().includes(searchQuery.toLowerCase())));
        return matchesCat && matchesSearch;
      })
      .sort((a, b) => {
        let valA = a[sortField] || 0;
        let valB = b[sortField] || 0;
        if (typeof valA === "string") valA = valA.toLowerCase();
        if (typeof valB === "string") valB = valB.toLowerCase();
        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
      });
  }, [candidates, searchQuery, selectedCategory, sortField, sortAsc]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const getSortIcon = (field) => {
    if (sortField !== field) return "↕";
    return sortAsc ? "↑" : "↓";
  };

  if (!candidates || candidates.length === 0) {
    return null;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Search & Filter Toolbar */}
      {showFilters && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.75rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flex: 1, minWidth: "260px" }}>
            <input
              type="text"
              placeholder="Search candidate ID, category, or skill..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="form-input"
              style={{ maxWidth: "320px", fontSize: "0.8125rem" }}
            />

            {categories.length > 0 && (
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="form-select"
                style={{ maxWidth: "200px", fontSize: "0.8125rem" }}
              >
                <option value="all">All Categories ({categories.length})</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            )}
          </div>

          {comparisonCandidates.length > 0 && (
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setIsCompareModalOpen(true)}
            >
              Compare Selected ({comparisonCandidates.length})
            </button>
          )}
        </div>
      )}

      {/* Table Container */}
      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: "40px", textAlign: "center" }}>#</th>
              <th style={{ width: "36px", textAlign: "center" }}>Compare</th>
              <th>Candidate ID</th>
              <th>Category</th>
              <th
                onClick={() => handleSort("semantic_sim")}
                style={{ cursor: "pointer", userSelect: "none" }}
              >
                Semantic Sim {getSortIcon("semantic_sim")}
              </th>
              <th
                onClick={() => handleSort("ML_score")}
                style={{ cursor: "pointer", userSelect: "none" }}
              >
                ML Score {getSortIcon("ML_score")}
              </th>
              <th
                onClick={() => handleSort("final_score")}
                style={{ cursor: "pointer", userSelect: "none" }}
              >
                Final Score {getSortIcon("final_score")}
              </th>
              <th>Matched Skills</th>
              <th style={{ textAlign: "right" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {processedCandidates.map((c, idx) => {
              const compared = isCompared(c.id);
              const rankNum = idx + 1;

              return (
                <tr
                  key={c.id}
                  style={{
                    backgroundColor: compared ? "var(--primary-light)" : undefined,
                    cursor: "pointer",
                  }}
                  onClick={() => setSelectedCandidate(c)}
                >
                  <td style={{ textAlign: "center" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: "24px",
                        height: "24px",
                        borderRadius: "50%",
                        fontSize: "0.75rem",
                        fontWeight: 700,
                        backgroundColor:
                          rankNum === 1
                            ? "#fef3c7"
                            : rankNum === 2
                            ? "#f1f5f9"
                            : rankNum === 3
                            ? "#ffedd5"
                            : "transparent",
                        color:
                          rankNum === 1
                            ? "#b45309"
                            : rankNum === 2
                            ? "#475569"
                            : rankNum === 3
                            ? "#c2410c"
                            : "var(--text-secondary)",
                      }}
                    >
                      {rankNum}
                    </span>
                  </td>

                  <td style={{ textAlign: "center" }} onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={compared}
                      onChange={() => toggleComparison(c)}
                      style={{ cursor: "pointer", accentColor: "var(--primary)" }}
                      title="Select for comparison"
                    />
                  </td>

                  <td style={{ fontWeight: 600 }}>
                    <code style={{ fontSize: "0.8125rem", color: "var(--primary-text)" }}>
                      #{c.id}
                    </code>
                  </td>

                  <td>
                    {c.category ? (
                      <span className="badge badge-neutral">{c.category}</span>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>—</span>
                    )}
                  </td>

                  <td style={{ width: "130px" }}>
                    <ScoreBreakdown value={c.semantic_sim} size="sm" />
                  </td>

                  <td style={{ width: "130px" }}>
                    <ScoreBreakdown value={c.ML_score} size="sm" />
                  </td>

                  <td style={{ width: "140px" }}>
                    <ScoreBreakdown value={c.final_score} accent size="md" />
                  </td>

                  <td style={{ maxWidth: "240px" }}>
                    <SkillMatchDisplay
                      matched={c.explanation?.matched_skills || []}
                      missing={c.explanation?.missing_skills || []}
                      compact={true}
                      maxVisible={3}
                    />
                  </td>

                  <td style={{ textAlign: "right" }} onClick={(e) => e.stopPropagation()}>
                    <button
                      className="btn btn-ghost btn-sm"
                      onClick={() => setSelectedCandidate(c)}
                      title="View TreeSHAP explanation & candidate details"
                    >
                      Inspect →
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-secondary)", padding: "0 0.5rem" }}>
        <span>Showing {processedCandidates.length} candidate{processedCandidates.length !== 1 ? "s" : ""}</span>
        <span>Click any row to inspect TreeSHAP feature attributions and signals</span>
      </div>
    </div>
  );
}
