"""
src/cluster.py
--------------
TalentRank AI - Step 8b: Clustering / visualization logic.

build_cluster_map(session, pca_override=None) -> dict
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

MIN_RESUMES = 30
MAX_K = 8
RANDOM_STATE = 42


def _best_k(embeddings: np.ndarray):
    """Try k in [2, min(MAX_K, n//10)], return (best_k, best_silhouette)."""
    n = len(embeddings)
    k_max = min(MAX_K, n // 10)
    if k_max < 2:
        k_max = 2

    best_k, best_sil, best_labels = 2, -1.0, None
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=min(2000, n), random_state=RANDOM_STATE)
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels

    return best_k, best_sil, best_labels


_DOMAIN_STOPWORDS = {
    "experience", "year", "years", "summary", "current", "company", "city", "state",
}


def _top_keywords(texts, n: int = 5):
    """Return top-n TF-IDF keywords from a list of texts, excluding domain stop words."""
    if not texts:
        return []
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = list(frozenset(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS)
    vec = TfidfVectorizer(stop_words=stop, max_features=5000, min_df=1)
    try:
        mat = vec.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(mat.sum(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:n]
    terms = vec.get_feature_names_out()
    return [terms[i] for i in top_idx]


def build_cluster_map(session: dict, jd_embedding=None) -> dict:
    """
    Build the cluster-map response dict from a completed session.

    Parameters
    ----------
    session      : session dict from _sessions store (status must be 'done')
    jd_embedding : optional (1, D) float32 array — last JD embedding to project

    Returns
    -------
    dict matching the /cluster-map response schema
    """
    resumes_df = session["resumes_df"]
    embeddings = session["embeddings"]          # (n, D) float32
    n = len(resumes_df)

    if n < MIN_RESUMES:
        raise ValueError(
            f"Only {n} resumes in this session. At least {MIN_RESUMES} are required "
            "to produce a meaningful cluster map."
        )

    # ── 1. Auto-select k and cluster on original embeddings ──────────────────
    k, silhouette, cluster_labels = _best_k(embeddings)

    # ── 2. PCA 2D projection ─────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca.fit_transform(embeddings)   # (n, 2)

    # ── 3. Build points list ──────────────────────────────────────────────────
    has_category = "Category" in resumes_df.columns and not (
        resumes_df["Category"].eq("UNKNOWN").all()
    )

    points = []
    for i, row in resumes_df.iterrows():
        pt = {
            "id":      row["ID"],
            "x":       float(coords_2d[i, 0]),
            "y":       float(coords_2d[i, 1]),
            "cluster": int(cluster_labels[i]),
        }
        if has_category:
            pt["category"] = str(row["Category"])
        points.append(pt)

    # ── 4. Per-cluster keyword labels ─────────────────────────────────────────
    clusters_info = []
    for c in range(k):
        mask = cluster_labels == c
        texts = resumes_df.loc[mask, "Resume_clean"].tolist()
        clusters_info.append({
            "cluster":  c,
            "size":     int(mask.sum()),
            "keywords": _top_keywords(texts, n=5),
        })

    result = {
        "k":          k,
        "silhouette": round(float(silhouette), 4),
        "points":     points,
        "clusters":   clusters_info,
    }

    # ── 5. Optional: ARI + NMI if Category present ───────────────────────────
    if has_category:
        le = LabelEncoder()
        true_labels = le.fit_transform(resumes_df["Category"].values)
        result["adjusted_rand_index"] = round(
            float(adjusted_rand_score(true_labels, cluster_labels)), 4
        )
        result["normalized_mutual_info"] = round(
            float(normalized_mutual_info_score(true_labels, cluster_labels)), 4
        )

    # ── 6. PCA variance explained ─────────────────────────────────────────────
    result["pca_variance_explained"] = round(float(pca.explained_variance_ratio_.sum()), 4)

    # ── 7. Optional: jd_point + jd_nearest_cluster + top10_ids from /rank ──────
    last_top10    = session.get("last_top10")     # [{id, final_score}] or None
    jd_embedding  = session.get("last_jd_embedding")

    if last_top10 and jd_embedding is not None:
        jd_vec  = jd_embedding.reshape(1, -1)
        jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-10)

        # Build id -> position index in resumes_df for fast cluster lookup
        id_to_idx = {str(rid): i for i, rid in enumerate(resumes_df["ID"].tolist())}

        # Majority vote over /rank top-10 cluster assignments
        # Ties broken by higher mean final_score of that cluster's members
        vote_counts = {}
        vote_scores = {}
        for entry in last_top10:
            rid = str(entry["id"])
            idx = id_to_idx.get(rid)
            if idx is None:
                continue
            c = int(cluster_labels[idx])
            vote_counts[c] = vote_counts.get(c, 0) + 1
            vote_scores[c] = vote_scores.get(c, []) + [float(entry["final_score"])]

        best_c = max(
            vote_counts,
            key=lambda c: (vote_counts[c], np.mean(vote_scores[c]))
        )

        # Centroid cosine sim for the winning cluster (extra info)
        centroid     = embeddings[cluster_labels == best_c].mean(axis=0)
        cent_norm    = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroid_sim = float(np.dot(cent_norm, jd_norm.ravel()))

        jd_2d = pca.transform(jd_vec)
        result["jd_point"] = {
            "x": float(jd_2d[0, 0]),
            "y": float(jd_2d[0, 1]),
        }
        result["jd_nearest_cluster"] = {
            "cluster":             best_c,
            "top10_in_cluster":    vote_counts[best_c],
            "centroid_similarity": round(centroid_sim, 4),
        }
        result["top10_ids"] = [str(e["id"]) for e in last_top10]

    return result
