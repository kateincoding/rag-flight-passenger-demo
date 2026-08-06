"""Tests for retrieval vector math (no network — pure numpy)."""
import numpy as np

from agents.retrieval import _normalize


def test_normalize_returns_unit_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)   # norm 5
    assert abs(float(np.linalg.norm(_normalize(v))) - 1.0) < 1e-6


def test_normalize_preserves_direction():
    v = np.array([2.0, 0.0], dtype=np.float32)
    assert np.allclose(_normalize(v), [1.0, 0.0])


def test_inner_product_of_normalized_equals_cosine():
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    ip = float(np.dot(_normalize(a), _normalize(b)))
    assert abs(ip - (1.0 / np.sqrt(2))) < 1e-6   # cos(45°)
