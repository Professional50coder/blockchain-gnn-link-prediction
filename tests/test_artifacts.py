"""Consistency checks on the committed dashboard artifacts."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def emb():
    return np.load(ROOT / "node_embeddings.npy")


def test_embeddings_are_finite_64d(emb):
    assert emb.ndim == 2 and emb.shape[1] == 64
    assert np.isfinite(emb).all()


def test_edges_index_into_embeddings(emb):
    edges = pd.read_csv(ROOT / "edges.csv")
    assert list(edges.columns) == ["from_id", "to_id"]
    assert edges.min().min() >= 0
    assert edges.max().max() < emb.shape[0]


def test_flagged_wallets_are_valid(emb):
    f = pd.read_csv(ROOT / "fraudulent_wallets.csv")
    assert {"wallet_id", "fraud_score", "wallet_address"} <= set(f.columns)
    assert f["wallet_id"].between(0, emb.shape[0] - 1).all()
    assert f["wallet_id"].is_unique


def test_label_encoder_matches_embedding_rows(emb):
    with open(ROOT / "label_encoder.pkl", "rb") as fh:
        le = pickle.load(fh)  # first-party artifact committed to this repo
    assert len(le.classes_) == emb.shape[0]


def test_loss_history_and_roc_present():
    loss = np.load(ROOT / "loss_history.npy")
    assert loss.shape == (100,)
    roc = np.load(ROOT / "roc_data.npz")
    assert {"fpr", "tpr", "auc"} <= set(roc.files)
    assert len(roc["fpr"]) == len(roc["tpr"])


def test_leakage_free_metrics_file_is_consistent():
    import json
    m = json.loads((ROOT / "metrics.json").read_text())
    assert sum(m["train/val/test"]) == m["unique_edges"]
    assert 0.0 <= m["graphsage_test_auc"] <= 1.0
