"use client";

import { useEffect } from "react";
import { useTalentRank } from "../context/TalentRankContext";
import DashboardLayout from "../components/DashboardLayout";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import Alert from "../components/Alert";
import EmptyState from "../components/EmptyState";
import LoadingState from "../components/LoadingState";

function fmt(v, decimals = 4) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Number(v).toFixed(decimals);
}

function fmtPct(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function VerdictChip({ excludesZero }) {
  return (
    <span
      className={`badge ${excludesZero ? "badge-success" : "badge-neutral"}`}
      style={{ fontSize: "0.75rem" }}
    >
      {excludesZero ? "✓ Statistically Distinguishable" : "— Indistinguishable from 0"}
    </span>
  );
}

export default function EvaluationPage() {
  const { modelData, modelLoading, modelError, loadModelData } = useTalentRank();

  useEffect(() => {
    if (!modelData && !modelLoading) {
      loadModelData();
    }
  }, [modelData, modelLoading, loadModelData]);

  const re = modelData?.ranker_eval;

  return (
    <DashboardLayout>
      <PageHeader
        title="ML Model Evaluation & Benchmarking"
        description="Empirical validation of the candidate scoring pipeline against semantic baselines, bootstrap significance testing, and cross-validation ablation."
        badge={{
          text: modelData?.algorithm ? `Model: ${modelData.algorithm}` : "Offline Eval",
          type: "primary",
        }}
        actions={
          <button
            className="btn btn-secondary btn-sm"
            onClick={loadModelData}
            disabled={modelLoading}
          >
            🔄 Refresh Metrics
          </button>
        }
      />

      {/* Prominent Academic Weak-Label Disclaimer */}
      <div
        style={{
          backgroundColor: "#fffbeb",
          border: "1px solid #fde68a",
          borderRadius: "var(--radius-md)",
          padding: "1rem 1.25rem",
          marginBottom: "1.5rem",
          display: "flex",
          alignItems: "flex-start",
          gap: "0.75rem",
          fontSize: "0.8125rem",
          color: "#92400e",
          lineHeight: 1.5,
        }}
      >
        <span style={{ fontSize: "1.25rem", flexShrink: 0 }}>⚠️</span>
        <div>
          <strong>Evaluation Methodology & Limitations Disclaimer:</strong>
          <p style={{ margin: "0.25rem 0 0" }}>
            The classification and ranking metrics presented on this dashboard are evaluated against <strong>weak proxy labels</strong> (derived from dataset category overlap and synthetic job descriptions).
            They measure algorithmic agreement with proxy signals and <strong>do not represent validated predictions of human job performance, interviewing success, or hiring outcomes</strong>.
          </p>
        </div>
      </div>

      {modelError && <Alert type="error" msg={modelError} />}

      {modelLoading ? (
        <div className="saas-card" style={{ padding: "3rem" }}>
          <LoadingState message="Loading model evaluation artifacts (model_metadata.json & ranker_eval.json)..." rows={6} />
        </div>
      ) : !modelData ? (
        <EmptyState
          icon="📈"
          title="Evaluation Data Unavailable"
          description="Model evaluation artifacts were not found on the backend. Please ensure src/08_train_ranker.py and src/09_closing_checks.py have been executed."
          action={
            <button className="btn btn-primary btn-sm" onClick={loadModelData}>
              Retry Loading Model Info
            </button>
          }
        />
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
          
          {/* Section 1: Classifier Performance Metrics */}
          <div>
            <h2 style={{ fontSize: "1.0625rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.875rem" }}>
              1. Candidate Scorer Classifier Metrics (Stage 2 XGBoost)
            </h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
                gap: "1rem",
              }}
            >
              <StatCard
                title="Primary Algorithm"
                value={modelData.algorithm || "XGBoost"}
                subtext="Binary Logistic Classifier"
                icon="🤖"
                badge={{ text: "Stage 2 Scorer", type: "primary" }}
              />

              <StatCard
                title="ROC-AUC Score"
                value={fmt(modelData.roc_auc, 3)}
                subtext="Discrimination capacity on held-out test"
                icon="📊"
                badge={{ text: "Good", type: "success" }}
              />

              <StatCard
                title="F1 Score"
                value={fmt(modelData.f1, 3)}
                subtext="Harmonic mean of precision & recall"
                icon="🎯"
                badge={{ text: "Weak Labels", type: "info" }}
              />

              <StatCard
                title="Engineered Features"
                value={modelData.features?.length || 5}
                subtext="Semantic, Skill, Experience, Education, TF-IDF"
                icon="🧩"
                badge={{ text: "SHAP Explainable", type: "neutral" }}
              />
            </div>
          </div>

          {/* Section 2: Features List & Explanations */}
          <div className="saas-card" style={{ padding: "1.5rem" }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
              Engineered Model Features & Monotonicity Properties
            </h3>
            <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
              The 5 candidate features evaluated by XGBoost and TreeSHAP explainability gates:
            </p>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature Column</th>
                    <th>Extraction Source</th>
                    <th>Monotonicity Behavior</th>
                    <th>TreeSHAP Consistency Gate</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>semantic_sim</code></td>
                    <td>SentenceTransformers <code>all-MiniLM-L6-v2</code> (Cosine Sim)</td>
                    <td><span className="badge badge-success">Spearman +0.93 (Strongly Monotone)</span></td>
                    <td>Positive signal if &gt; 0.50</td>
                  </tr>
                  <tr>
                    <td><code>tfidf_sim</code></td>
                    <td>Unigram/Bigram Term Frequency (Cosine Sim)</td>
                    <td><span className="badge badge-success">Spearman +0.62 (Monotone)</span></td>
                    <td>Positive signal if &gt; 0.10</td>
                  </tr>
                  <tr>
                    <td><code>education_match_score</code></td>
                    <td>Regex Qualification Ladder (BA, MA, PhD)</td>
                    <td><span className="badge badge-warning">Spearman +0.32 (Weakly Monotone)</span></td>
                    <td>Positive signal if &ge; 1.00</td>
                  </tr>
                  <tr>
                    <td><code>experience_match_score</code></td>
                    <td>Extracted Years of Experience match</td>
                    <td><span className="badge badge-danger">Spearman -0.19 (Non-Monotone)</span></td>
                    <td>Guarded: never generates contradictory reason</td>
                  </tr>
                  <tr>
                    <td><code>skill_overlap_score</code></td>
                    <td>JD Skills overlap ratio in resume text</td>
                    <td><span className="badge badge-danger">Spearman -0.30 (Proxy artifact)</span></td>
                    <td>Guarded: checks true skill count before phrasing</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3: Ranking Evaluation Metrics (NDCG & Precision@10) */}
          {re && (
            <div className="saas-card" style={{ padding: "1.5rem" }}>
              <div style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  2. Headline Ranking Benchmark: Same-Category Generalisation (Scheme 1)
                </h3>
                <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                  5-fold GroupKFold by job description ID across 211 aligned test JDs. Comparing Semantic-only vs Classifier Blend vs Ranker.
                </p>
              </div>

              {re.cv_scheme_jd_id?.full_features && (
                <div className="data-table-container" style={{ marginBottom: "1.25rem" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Ranking Method</th>
                        <th>NDCG@10 (Mean)</th>
                        <th>Fold Std Dev</th>
                        <th>NDCG@20</th>
                        <th>Precision@10 (Mean)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Semantic Retrieval Only (Baseline)</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.semantic?.["ndcg@10"]?.mean)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.semantic?.["ndcg@10"]?.std)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.semantic?.["ndcg@20"]?.mean)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.semantic?.["p@10"]?.mean)}</td>
                      </tr>
                      <tr style={{ backgroundColor: "var(--primary-light)" }}>
                        <td style={{ fontWeight: 700, color: "var(--primary)" }}>
                          ★ Classifier Blend (Production Pipeline)
                        </td>
                        <td className="tabular-nums" style={{ fontWeight: 700, color: "var(--primary)" }}>
                          {fmt(re.cv_scheme_jd_id.full_features.clf_blend?.["ndcg@10"]?.mean)}
                        </td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.clf_blend?.["ndcg@10"]?.std)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.clf_blend?.["ndcg@20"]?.mean)}</td>
                        <td className="tabular-nums" style={{ fontWeight: 700 }}>
                          {fmt(re.cv_scheme_jd_id.full_features.clf_blend?.["p@10"]?.mean)}
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>XGBRanker Pairwise Blend</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.ranker_blend?.["ndcg@10"]?.mean)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.ranker_blend?.["ndcg@10"]?.std)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.ranker_blend?.["ndcg@20"]?.mean)}</td>
                        <td className="tabular-nums">{fmt(re.cv_scheme_jd_id.full_features.ranker_blend?.["p@10"]?.mean)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}

              {/* Paired Differences & 95% Bootstrap CIs */}
              {re.cv_scheme_jd_id?.significance && (
                <div>
                  <h4 style={{ fontSize: "0.875rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
                    Paired Bootstrap Differences (2,000 resamples, 95% Confidence Intervals)
                  </h4>
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Comparison Hypothesis</th>
                          <th>Mean Difference</th>
                          <th>95% Confidence Interval</th>
                          <th>Statistical Verdict</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(re.cv_scheme_jd_id.significance).map(([comp, sig]) => (
                          <tr key={comp}>
                            <td style={{ fontWeight: 600 }}><code>{comp}</code></td>
                            <td className="tabular-nums">{fmt(sig.mean, 4)}</td>
                            <td className="tabular-nums">[{fmt(sig.ci_lo, 4)}, {fmt(sig.ci_hi, 4)}]</td>
                            <td><VerdictChip excludesZero={sig.excludes_zero} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Section 4: Controlled Validation Set (300 candidates) */}
          {re?.medium_set_eval && (
            <div className="saas-card" style={{ padding: "1.5rem" }}>
              <div style={{ marginBottom: "1rem" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  3. Controlled Validation Set Benchmark (300 candidates, Hand-graded Answer Key)
                </h3>
                <p style={{ fontSize: "0.8125rem", color: "var(--text-secondary)", margin: "0.25rem 0 0" }}>
                  Gold-standard evaluation graded 0 (irrelevant), 1 (relevant), 2 (strong fit) on a realistic Web Developer JD.
                </p>
              </div>

              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Method</th>
                      <th>NDCG@10</th>
                      <th>Precision@10 (Relevant, &ge;1)</th>
                      <th>Precision@10 (Strong Fit, =2)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Semantic Only</td>
                      <td className="tabular-nums">{fmt(re.medium_set_eval.a_semantic?.ndcg_at_10)}</td>
                      <td className="tabular-nums">{fmtPct(re.medium_set_eval.a_semantic?.p_at_10_ge1)}</td>
                      <td className="tabular-nums">{fmtPct(re.medium_set_eval.a_semantic?.p_at_10_eq2)}</td>
                    </tr>
                    <tr style={{ backgroundColor: "var(--primary-light)" }}>
                      <td style={{ fontWeight: 700, color: "var(--primary)" }}>
                        Classifier Blend (Deployed)
                      </td>
                      <td className="tabular-nums" style={{ fontWeight: 700, color: "var(--primary)" }}>
                        {fmt(re.medium_set_eval.b_clf_blend?.ndcg_at_10)}
                      </td>
                      <td className="tabular-nums" style={{ fontWeight: 700 }}>
                        {fmtPct(re.medium_set_eval.b_clf_blend?.p_at_10_ge1)}
                      </td>
                      <td className="tabular-nums" style={{ fontWeight: 700 }}>
                        {fmtPct(re.medium_set_eval.b_clf_blend?.p_at_10_eq2)}
                      </td>
                    </tr>
                    <tr>
                      <td>Ranker Blend</td>
                      <td className="tabular-nums">{fmt(re.medium_set_eval.c_ranker_blend?.ndcg_at_10)}</td>
                      <td className="tabular-nums">{fmtPct(re.medium_set_eval.c_ranker_blend?.p_at_10_ge1)}</td>
                      <td className="tabular-nums">{fmtPct(re.medium_set_eval.c_ranker_blend?.p_at_10_eq2)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Section 5: Caveats and Limitations */}
          {re?.caveats && re.caveats.length > 0 && (
            <div className="saas-card" style={{ padding: "1.5rem" }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "0.5rem" }}>
                Documented Empirical Caveats & Model Limitations
              </h3>
              <ul style={{ paddingLeft: "1.25rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.8125rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                {re.caveats.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </div>
          )}

        </div>
      )}
    </DashboardLayout>
  );
}
