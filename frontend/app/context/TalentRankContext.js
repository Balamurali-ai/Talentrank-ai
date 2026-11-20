"use client";

import { createContext, useContext, useState, useEffect, useRef, useCallback } from "react";
import { login as apiLogin, uploadResumes, getUploadStatus, rank as apiRank, getClusterMap, getModelInfo, getHealth } from "../api";

const TalentRankContext = createContext(null);

export function TalentRankProvider({ children }) {
  // ── Authentication ──────────────────────────────────────────────────────────
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);

  useEffect(() => {
    try {
      const savedToken = localStorage.getItem("talentrank_token");
      const savedUser = localStorage.getItem("talentrank_user");
      if (savedToken) {
        setToken(savedToken);
        if (savedUser) setUser(JSON.parse(savedUser));
      }
    } catch (_) {}
    setAuthChecked(true);
  }, []);

  const loginUser = async (username, password) => {
    const data = await apiLogin(username, password);
    setToken(data.access_token);
    const userInfo = { username, role: "Recruiter Admin" };
    setUser(userInfo);
    try {
      localStorage.setItem("talentrank_token", data.access_token);
      localStorage.setItem("talentrank_user", JSON.stringify(userInfo));
    } catch (_) {}
    addNotification({
      id: Date.now(),
      title: "Signed In",
      message: `Welcome back, ${username}. Ready for candidate ranking.`,
      time: "Just now",
      type: "success",
    });
    return data;
  };

  const logoutUser = () => {
    clearInterval(pollRef.current);
    setToken(null);
    setUser(null);
    try {
      localStorage.removeItem("talentrank_token");
      localStorage.removeItem("talentrank_user");
    } catch (_) {}
  };

  // ── Notifications ───────────────────────────────────────────────────────────
  const [notifications, setNotifications] = useState([
    {
      id: 1,
      title: "System Online",
      message: "SentenceTransformers and FAISS index loaded (2,484 resumes available).",
      time: "System start",
      type: "info",
    },
  ]);
  const [hasUnreadNotifications, setHasUnreadNotifications] = useState(true);

  const addNotification = (notif) => {
    setNotifications((prev) => [notif, ...prev.slice(0, 9)]);
    setHasUnreadNotifications(true);
  };

  const markNotificationsRead = () => {
    setHasUnreadNotifications(false);
  };

  // ── Resume Pool / Session ───────────────────────────────────────────────────
  const [sessionId, setSessionId] = useState(null);
  const [poolType, setPoolType] = useState("default"); // "default" | "uploaded"
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [uploadedResumesCount, setUploadedResumesCount] = useState(2484);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  const uploadResumeFile = async (file) => {
    if (!token) throw new Error("Authentication required.");
    setUploadError("");
    setIsUploading(true);
    setUploadProgress(null);
    setUploadedFileName(file.name);

    try {
      const data = await uploadResumes(token, file);
      const sid = data.session_id;
      setUploadProgress({ processed: 0, total: data.total_resumes, status: "processing" });

      return new Promise((resolve, reject) => {
        clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
          try {
            const s = await getUploadStatus(token, sid);
            setUploadProgress({ processed: s.processed, total: s.total, status: s.status });

            if (s.status === "done") {
              clearInterval(pollRef.current);
              setSessionId(sid);
              setPoolType("uploaded");
              setUploadedResumesCount(s.processed || data.total_resumes);
              setIsUploading(false);
              addNotification({
                id: Date.now(),
                title: "Resume Pool Indexed",
                message: `Successfully processed ${s.processed || data.total_resumes} resumes from "${file.name}".`,
                time: "Just now",
                type: "success",
              });
              resolve(sid);
            } else if (s.status === "error") {
              clearInterval(pollRef.current);
              setIsUploading(false);
              setUploadError("Server-side indexing failed. Please verify CSV formatting.");
              reject(new Error("Upload processing error"));
            }
          } catch (pollErr) {
            clearInterval(pollRef.current);
            setIsUploading(false);
            if (pollErr.status === 401) logoutUser();
            setUploadError(pollErr.message);
            reject(pollErr);
          }
        }, 2000);
      });
    } catch (err) {
      setIsUploading(false);
      setUploadError(err.message || "Failed to initiate resume upload.");
      if (err.status === 401) logoutUser();
      throw err;
    }
  };

  const useDefaultPool = () => {
    clearInterval(pollRef.current);
    setSessionId(null);
    setPoolType("default");
    setUploadedFileName("");
    setUploadedResumesCount(2484);
    setUploadProgress(null);
    setUploadError("");
    setClusterMap(null);
    setClusterMapState("idle");
  };

  // ── Job Description & Ranking ───────────────────────────────────────────────
  const [jd, setJd] = useState({
    job_title: "Web Developer",
    role: "Full-Stack Engineer",
    skills: "HTML, CSS, JavaScript, React, Node.js",
    experience: "3 to 5 Years",
    qualifications: "Bachelor's degree in Computer Science or related field",
    responsibilities: "Develop modern web interfaces, integrate RESTful APIs, maintain database performance, and collaborate with product teams.",
    jd_text: "Looking for an experienced Web Developer with strong front-end and back-end fundamentals. Required skills include HTML, CSS, JavaScript, React, and Node.js. Experience building responsive web applications and deploying APIs is expected.",
  });

  const [alpha, setAlpha] = useState(0.5);
  const [results, setResults] = useState([]);
  const [explanationNote, setExplanationNote] = useState("");
  const [isRanking, setIsRanking] = useState(false);
  const [rankingError, setRankingError] = useState("");
  const [rankingHistory, setRankingHistory] = useState([]);

  const runRanking = async (overrideJd = null, overrideAlpha = null) => {
    if (!token) throw new Error("Authentication required.");
    const targetJd = overrideJd || jd;
    const targetAlpha = overrideAlpha !== null && overrideAlpha !== undefined ? overrideAlpha : alpha;

    if (!targetJd.jd_text || !targetJd.jd_text.trim()) {
      setRankingError("Job description text is required.");
      throw new Error("Job description text is required.");
    }

    setRankingError("");
    setIsRanking(true);

    try {
      const payload = { ...targetJd };
      if (sessionId) payload.session_id = sessionId;

      const data = await apiRank(token, payload, targetAlpha);
      const rankedResults = data.results || [];
      setResults(rankedResults);
      setExplanationNote(data.explanation_note || "");

      // Record in history
      const avgScore = rankedResults.length > 0
        ? (rankedResults.reduce((acc, r) => acc + (r.final_score || 0), 0) / rankedResults.length).toFixed(3)
        : "0.000";
      const topScore = rankedResults.length > 0 ? (rankedResults[0]?.final_score || 0).toFixed(3) : "0.000";

      const historyEntry = {
        id: `rank-${Date.now()}`,
        timestamp: new Date().toISOString(),
        job_title: targetJd.job_title || "Untitled Search",
        skills: targetJd.skills,
        count: rankedResults.length,
        topScore,
        avgScore,
        alpha: targetAlpha,
        poolType,
        sessionId,
      };

      setRankingHistory((prev) => [historyEntry, ...prev.slice(0, 19)]);

      addNotification({
        id: Date.now(),
        title: "Ranking Complete",
        message: `Ranked top ${rankedResults.length} candidates for "${targetJd.job_title || 'Job'}" (alpha=${targetAlpha}).`,
        time: "Just now",
        type: "success",
      });

      // If active session, fetch or refresh cluster map
      if (sessionId) {
        loadClusterMap(sessionId);
      }

      return rankedResults;
    } catch (err) {
      if (err.status === 401) {
        logoutUser();
        throw new Error("Session expired. Please sign in again.");
      }
      const msg = err.status === 429
        ? "Rate limit exceeded (10 requests/minute). Please wait a moment."
        : (err.message || "Candidate ranking failed.");
      setRankingError(msg);
      throw new Error(msg);
    } finally {
      setIsRanking(false);
    }
  };

  // ── Cluster Map ─────────────────────────────────────────────────────────────
  const [clusterMap, setClusterMap] = useState(null);
  const [clusterMapState, setClusterMapState] = useState("idle"); // idle, loading, ok, tooFew, error
  const [clusterMapError, setClusterMapError] = useState("");

  const loadClusterMap = useCallback(async (sid) => {
    const activeSid = sid || sessionId;
    if (!token || !activeSid) return;
    setClusterMapState("loading");
    setClusterMapError("");

    try {
      const data = await getClusterMap(token, activeSid);
      setClusterMap(data);
      setClusterMapState("ok");
    } catch (err) {
      if (err.status === 400) {
        setClusterMapState("tooFew");
      } else {
        setClusterMapState("error");
        setClusterMapError(err.message || "Failed to load cluster projection.");
      }
    }
  }, [token, sessionId]);

  // ── Model Info ──────────────────────────────────────────────────────────────
  const [modelData, setModelData] = useState(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");

  const loadModelData = useCallback(async () => {
    if (!token) return;
    setModelLoading(true);
    setModelError("");

    try {
      const data = await getModelInfo(token);
      setModelData(data);
    } catch (err) {
      if (err.status === 404) {
        setModelError("Evaluation artifacts (model_metadata.json / ranker_eval.json) not found on backend.");
      } else {
        setModelError(err.message || "Failed to retrieve model evaluation metrics.");
      }
    } finally {
      setModelLoading(false);
    }
  }, [token]);

  // ── Health Status ───────────────────────────────────────────────────────────
  const [health, setHealth] = useState(null);

  const checkHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth(data);
      return data;
    } catch (_) {
      setHealth({ status: "offline", models_loaded: false });
      return null;
    }
  }, []);

  // Poll health on mount
  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Load model info whenever token is present
  useEffect(() => {
    if (token) {
      loadModelData();
    }
  }, [token, loadModelData]);

  // ── Candidate Selection & Comparison ────────────────────────────────────────
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [comparisonCandidates, setComparisonCandidates] = useState([]);
  const [isCompareModalOpen, setIsCompareModalOpen] = useState(false);

  const toggleComparison = (candidate) => {
    setComparisonCandidates((prev) => {
      const exists = prev.some((c) => c.id === candidate.id);
      if (exists) {
        return prev.filter((c) => c.id !== candidate.id);
      }
      if (prev.length >= 4) {
        alert("You can compare up to 4 candidates side by side.");
        return prev;
      }
      return [...prev, candidate];
    });
  };

  const clearComparison = () => {
    setComparisonCandidates([]);
  };

  const isCompared = (candidateId) => {
    return comparisonCandidates.some((c) => c.id === candidateId);
  };

  return (
    <TalentRankContext.Provider
      value={{
        token,
        user,
        authChecked,
        loginUser,
        logoutUser,
        notifications,
        hasUnreadNotifications,
        markNotificationsRead,
        sessionId,
        poolType,
        uploadedFileName,
        uploadedResumesCount,
        uploadProgress,
        isUploading,
        uploadError,
        uploadResumeFile,
        useDefaultPool,
        jd,
        setJd,
        alpha,
        setAlpha,
        results,
        explanationNote,
        isRanking,
        rankingError,
        rankingHistory,
        runRanking,
        clusterMap,
        clusterMapState,
        clusterMapError,
        loadClusterMap,
        modelData,
        modelLoading,
        modelError,
        loadModelData,
        health,
        checkHealth,
        selectedCandidate,
        setSelectedCandidate,
        comparisonCandidates,
        toggleComparison,
        clearComparison,
        isCompared,
        isCompareModalOpen,
        setIsCompareModalOpen,
      }}
    >
      {children}
    </TalentRankContext.Provider>
  );
}

export function useTalentRank() {
  const context = useContext(TalentRankContext);
  if (!context) {
    throw new Error("useTalentRank must be used within a TalentRankProvider");
  }
  return context;
}
