import numpy as np
import pytest
from src.utils.metrics import evaluate_metrics, compute_eer


def _binary_case():
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    return labels, scores


def test_evaluate_metrics_keys():
    labels, scores = _binary_case()
    m = evaluate_metrics(labels, scores)
    assert set(m.keys()) == {"auc", "ap", "eer"}


def test_auc_perfect():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    m = evaluate_metrics(labels, scores)
    assert m["auc"] == pytest.approx(1.0)


def test_eer_range():
    labels, scores = _binary_case()
    eer = compute_eer(labels, scores)
    assert 0.0 <= eer <= 1.0


def test_ap_range():
    labels, scores = _binary_case()
    m = evaluate_metrics(labels, scores)
    assert 0.0 <= m["ap"] <= 1.0
