"use client";

import { useState, useMemo, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import CandidateTable from "../components/CandidateTable";
import CandidateCard from "../components/CandidateCard";
import EmptyState from "../components/EmptyState";

function CandidateExplorerContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";

  const {
    results,
    comparisonCandidates,
    clearComparison,
    setIsCompareModalOpen,
    poolType,
    uploadedResumesCount,
  } = useTalentRank();

  const [searchQuery, setSearchQuery] = useState(initialQuery);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [minScoreFilter, setMinScoreFilter] = useState("all"); // "all", "0.5", "0.7"
  const [viewMode, setViewMode] = useState("table"); // "table" | "grid"

  useEffect(() => {
    if (initialQuery) setSearchQuery(initialQuery);
  }, [initialQuery]);

  // Unique categories
  const categories = useMemo(() => {
    const set = new Set();
    results.forEach((c) => {
      if (c.category) set.add(c.category);
    });
    return Array.from(set).sort();
  }, [results]);

  // Filtered candidate list
  const filteredCandidates = useMemo(() => {
    return results.filter((c) => {
      const matchesCategory = selectedCategory === "all" || c.category === selectedCategory;
      const matchesSearch =
        !searchQuery.trim() ||
        String(c.id).toLowerCase().includes(searchQuery.toLowerCase()) ||
        (c.category && c.category.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (c.explanation?.matched_skills &&
          c.explanation.matched_skills.some((s) => s.toLowerCase().includes(searchQuery.toLowerCase())));

      const score = c.final_score || 0;
      const matchesScore =
        minScoreFilter === "all" ||
        (minScoreFilter === "0.7" && score >= 0.7) ||
        (minScoreFilter === "0.5" && score >= 0.5);

      return matchesCategory && matchesSearch && matchesScore;
    });
  }, [results, searchQuery, selectedCategory, minScoreFilter]);

  return (
    <DashboardLayout>
      <PageHeader
        title="Candidate Explorer"
        description="Search, filter, inspect TreeSHAP explanations, and compare candidate profiles side by side."
        badge={{
          text: `${filteredCandidates.length} Active Candidates`,
          type: "primary",
        }}
        actions={
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {comparisonCandidates.length > 0 && (
              <button
                className="btn btn-primary btn-sm"
                onClick={() => setIsCompareModalOpen(true)}
              >
                Compare Selected ({comparisonCandidates.length})
              </button>
            )}
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => router.push("/ranking")}
            >
              ⚡ New Ranking Search
            </button>
          </div>
        }
      />

      {/* Comparison Action Floating Pill / Banner */}
      {comparisonCandidates.length > 0 && (
        <div
          style={{
            backgroundColor: "var(--primary-light)",
            border: "1px solid var(--primary-border)",
            borderRadius: "var(--radius-md)",
            padding: "0.75rem 1.25rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "1.25rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "1.1rem" }}>👥</span>
            <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--primary-text)" }}>
              {comparisonCandidates.length} of 4 candidates selected for comparison
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              ({comparisonCandidates.map((c) => `#${c.id}`).join(", ")})
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <button
              className="btn btn-ghost btn-sm"
              onClick={clearComparison}
              style={{ fontSize: "0.75rem" }}
            >
              Clear Selection
            </button>
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setIsCompareModalOpen(true)}
            >
              Open Side-by-Side Comparison →
            </button>
          </div>
        </div>
      )}

      {/* Filter and Control Bar */}
      <div
        className="saas-card"
        style={{
          padding: "1rem 1.25rem",
          marginBottom: "1.5rem",
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1rem",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.75rem", flex: 1 }}>
          {/* Search Box */}
          <input
            type="text"
            placeholder="Search candidate ID, category, or skills..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="form-input"
            style={{ maxWidth: "300px", fontSize: "0.8125rem" }}
          />

          {/* Category Dropdown */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="form-select"
            style={{ maxWidth: "220px", fontSize: "0.8125rem" }}
          >
            <option value="all">All Categories ({categories.length || "0"})</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>

          {/* Min Score Filter */}
          <select
            value={minScoreFilter}
            onChange={(e) => setMinScoreFilter(e.target.value)}
            className="form-select"
            style={{ maxWidth: "180px", fontSize: "0.8125rem" }}
          >
            <option value="all">All Final Scores</option>
            <option value="0.7">Score ≥ 0.700</option>
            <option value="0.5">Score ≥ 0.500</option>
          </select>
        </div>

        {/* View Mode Toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.25rem", backgroundColor: "var(--bg-subtle)", padding: "0.25rem", borderRadius: "var(--radius-sm)" }}>
          <button
            onClick={() => setViewMode("table")}
            style={{
              background: viewMode === "table" ? "var(--bg-surface)" : "transparent",
              border: "none",
              borderRadius: "var(--radius-xs)",
              padding: "0.25rem 0.5rem",
              fontSize: "0.8125rem",
              fontWeight: viewMode === "table" ? 600 : 500,
              color: viewMode === "table" ? "var(--primary)" : "var(--text-secondary)",
              boxShadow: viewMode === "table" ? "var(--shadow-sm)" : "none",
              cursor: "pointer",
            }}
          >
            ☰ Table
          </button>
          <button
            onClick={() => setViewMode("grid")}
            style={{
              background: viewMode === "grid" ? "var(--bg-surface)" : "transparent",
              border: "none",
              borderRadius: "var(--radius-xs)",
              padding: "0.25rem 0.5rem",
              fontSize: "0.8125rem",
              fontWeight: viewMode === "grid" ? 600 : 500,
              color: viewMode === "grid" ? "var(--primary)" : "var(--text-secondary)",
              boxShadow: viewMode === "grid" ? "var(--shadow-sm)" : "none",
              cursor: "pointer",
            }}
          >
            ⊞ Cards
          </button>
        </div>
      </div>

      {/* Main Content: Table or Cards or Empty State */}
      {filteredCandidates.length > 0 ? (
        viewMode === "table" ? (
          <div className="saas-card" style={{ padding: "1.25rem" }}>
            <CandidateTable candidates={filteredCandidates} showFilters={false} />
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
              gap: "1rem",
            }}
          >
            {filteredCandidates.map((c, idx) => (
              <CandidateCard key={c.id} candidate={c} rank={idx + 1} />
            ))}
          </div>
        )
      ) : results.length > 0 ? (
        <EmptyState
          icon="🔍"
          title="No Candidates Match Filters"
          description="Try broadening your search term or resetting the category filter."
          action={
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setSearchQuery("");
                setSelectedCategory("all");
                setMinScoreFilter("all");
              }}
            >
              Reset Filters
            </button>
          }
        />
      ) : (
        <EmptyState
          icon="👥"
          title="No Candidates Ranked Yet"
          description="Candidate profiles and feature explanations appear here once you run a search in the Talent Ranking workspace."
          action={
            <button
              className="btn btn-primary btn-sm"
              onClick={() => router.push("/ranking")}
            >
              ⚡ Go to Talent Ranking
            </button>
          }
        />
      )}
    </DashboardLayout>
  );
}

export default function CandidatesPage() {
  return (
    <Suspense fallback={<div>Loading explorer...</div>}>
      <CandidateExplorerContent />
    </Suspense>
  );
}
