import numpy as np
import pandas as pd
import pytest

from utils import ml_utils as m


def test_risk_label_boundaries():
    assert m.risk_label(80) == "Critical"
    assert m.risk_label(79.99) == "High"
    assert m.risk_label(60) == "High"
    assert m.risk_label(35) == "Medium"
    assert m.risk_label(34.99) == "Low"
    assert m.risk_label(0) == "Low"


def test_risk_color_unknown_label_falls_back():
    assert m.risk_color_hex("Nope") == "#8b949e"
    assert m.risk_color_hex("Critical").startswith("#")


def test_normalise_score_orientation_and_bounds():
    # lower Isolation Forest score = more anomalous = higher risk
    assert m.normalise_fraud_score(-0.5, -0.5, 0.0) == pytest.approx(100.0)
    assert m.normalise_fraud_score(0.0, -0.5, 0.0) == pytest.approx(0.0)
    assert m.normalise_fraud_score(-0.25, -0.5, 0.0) == pytest.approx(50.0)
    assert 0 <= m.normalise_fraud_score(5.0, -0.5, 0.0) <= 100  # clipped


def test_normalise_score_degenerate_range():
    assert m.normalise_fraud_score(0.1, 0.1, 0.1) == 50.0


def test_link_probability_is_probability_and_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    p = m.compute_link_probability(a, b)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(m.compute_link_probability(b, a))


def test_link_probability_extreme_values_do_not_overflow():
    big = np.full(64, 1e6)
    assert m.compute_link_probability(big, big) == pytest.approx(1.0)
    assert m.compute_link_probability(big, -big) == pytest.approx(0.0, abs=1e-12)


def test_link_probability_zero_vectors_is_half():
    z = np.zeros(8)
    assert m.compute_link_probability(z, z) == pytest.approx(0.5)


def test_pca_projection_shapes_and_determinism():
    emb = np.random.default_rng(1).normal(size=(200, 16))
    c1, idx1, var = m.compute_pca_projection(emb, sample_size=50, n_components=3)
    c2, idx2, _ = m.compute_pca_projection(emb, sample_size=50, n_components=3)
    assert c1.shape == (50, 3) and len(set(idx1)) == 50
    assert np.array_equal(idx1, idx2) and np.allclose(c1, c2)
    assert var.sum() <= 1.0 + 1e-9


def test_pca_sample_size_capped_to_population():
    emb = np.random.default_rng(1).normal(size=(10, 5))
    coords, idx, _ = m.compute_pca_projection(emb, sample_size=6000, n_components=2)
    assert coords.shape == (10, 2) and len(idx) == 10


def test_top_k_similar_finds_identical_vector_and_excludes_self():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(100, 8))
    emb[7] = emb[3] * 2.0  # same direction as node 3
    ids, cos = m.find_top_k_similar(emb[3], emb, k=3, exclude_id=3, sample_size=100)
    assert 3 not in ids
    assert ids[0] == 7 and cos[0] == pytest.approx(1.0, abs=1e-6)
    assert list(cos) == sorted(cos, reverse=True)


def test_graph_statistics_on_tiny_graph():
    edges = pd.DataFrame({"from_id": [0, 0, 0, 1, 2], "to_id": [1, 2, 3, 3, 3]})
    s = m.create_graph_statistics(edges)
    assert s["total_transactions"] == 5
    assert s["active_nodes"] == 4
    assert s["max_out_degree"] == 3 and s["top_sender_id"] == 0
    assert s["max_in_degree"] == 3 and s["top_receiver_id"] == 3
    assert s["single_tx_senders"] == 2


def test_top_k_excluded_node_never_returned_even_if_all_cosines_negative():
    target = np.array([1.0, 0.0])
    emb = np.array([[1.0, 0.0], [-1.0, 0.1], [-1.0, -0.1], [-0.9, 0.2]])
    ids, _ = m.find_top_k_similar(target, emb, k=3, exclude_id=0, sample_size=4)
    assert 0 not in ids
