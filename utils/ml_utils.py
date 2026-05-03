"""
utils/ml_utils.py — ML helper functions: link prediction, PCA, fraud scoring.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def normalise_fraud_score(score: float, score_min: float, score_max: float) -> float:
    if score_max == score_min:
        return 50.0
    return float(np.clip((score_max - score) / (score_max - score_min) * 100, 0, 100))


def risk_label(risk_score: float) -> str:
    if risk_score >= 80: return "Critical"
    if risk_score >= 60: return "High"
    if risk_score >= 35: return "Medium"
    return "Low"


def risk_color_hex(label: str) -> str:
    return {"Critical": "#f85149", "High": "#ff7b72",
            "Medium": "#e3b341", "Low": "#3fb950"}.get(label, "#8b949e")


def compute_link_probability(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Multi-signal link probability: weighted dot + cosine + L2."""
    dot    = float(np.clip(np.dot(emb_a, emb_b), -500.0, 500.0))
    norm_a = np.linalg.norm(emb_a) + 1e-9
    norm_b = np.linalg.norm(emb_b) + 1e-9
    cosine = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    l2_sim = 1.0 / (1.0 + float(np.linalg.norm(emb_a - emb_b)))
    combined = np.clip(
        0.5 * dot + 0.3 * cosine * abs(dot) + 0.2 * l2_sim * abs(dot),
        -500.0, 500.0,
    )
    return float(1.0 / (1.0 + np.exp(-combined)))


def compute_pca_projection(
    embeddings: np.ndarray,
    sample_size: int = 6000,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce embeddings to n_components via PCA.
    Returns (coords, sampled_indices, explained_variance_ratio).
    """
    n = min(sample_size, embeddings.shape[0])
    idx = np.random.RandomState(42).choice(embeddings.shape[0], n, replace=False)
    sample = embeddings[idx]
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(sample)
    return coords, idx, pca.explained_variance_ratio_


def find_top_k_similar(
    target_emb: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
    exclude_id: int = -1,
    sample_size: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find top-k wallet IDs most similar (cosine) to target_emb.
    Returns (top_ids, cosine_scores).
    """
    rng = np.random.RandomState(0)
    sample_ids = rng.choice(embeddings.shape[0], min(sample_size, embeddings.shape[0]), replace=False)
    sample_embs = embeddings[sample_ids]
    t_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
    s_norms = np.linalg.norm(sample_embs, axis=1, keepdims=True) + 1e-9
    cosines = sample_embs @ t_norm / s_norms.squeeze()
    if exclude_id >= 0:
        mask = sample_ids != exclude_id
        cosines = cosines * mask
    top_local = np.argsort(cosines)[::-1][:k]
    return sample_ids[top_local], cosines[top_local]


def create_graph_statistics(edges_df: pd.DataFrame) -> dict:
    out_deg    = edges_df.groupby("from_id").size()
    in_deg     = edges_df.groupby("to_id").size()
    all_active = set(edges_df["from_id"].unique()) | set(edges_df["to_id"].unique())
    n_edges    = len(edges_df)
    top_out_id = int(out_deg.idxmax())
    top_in_id  = int(in_deg.idxmax())
    return {
        "total_transactions":  n_edges,
        "unique_senders":      edges_df["from_id"].nunique(),
        "unique_receivers":    edges_df["to_id"].nunique(),
        "active_nodes":        len(all_active),
        # avg over active senders/receivers respectively
        "avg_out_degree":      float(out_deg.mean()),
        "avg_in_degree":       float(in_deg.mean()),
        "max_out_degree":      int(out_deg.max()),
        "max_in_degree":       int(in_deg.max()),
        "median_out_degree":   float(out_deg.median()),
        "median_in_degree":    float(in_deg.median()),
        "top_sender_id":       top_out_id,
        "top_receiver_id":     top_in_id,
        # single-tx wallets (power-law indicator)
        "single_tx_senders":   int((out_deg == 1).sum()),
        "single_tx_pct":       float((out_deg == 1).mean() * 100),
        # hub wallets with > 10 txs
        "hub_senders":         int((out_deg > 10).sum()),
        "hub_receivers":       int((in_deg  > 10).sum()),
    }
