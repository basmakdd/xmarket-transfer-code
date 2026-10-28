"""Tests of the superseded v3 analysis code (legacy_v3/src/analyze.py, verify.py).

Run with:  python -m pytest -q legacy_v3/tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "legacy_v3" / "src"))


def test_paired_user_bootstrap_uses_the_difference_not_the_arms(tmp_path, monkeypatch):
    """The interval must describe the EFFECT. Two noisy arms with a small, consistent
    per-user gap have wide marginals but a tight paired interval - if the code bootstrapped
    the arms separately it would report the wide one and hide a real effect."""
    import analyze as A
    rng = np.random.default_rng(0)
    base = rng.normal(0.5, 0.30, 4000)          # very noisy across users
    borrowed = base + 0.02                       # small, perfectly consistent gain

    calls = {"a": borrowed, "b": base}
    monkeypatch.setattr(A, "load_per_user",
                        lambda rec, metric=A.METRIC, out_stem="full_random":
                        calls["a"] if rec["which"] == "a" else calls["b"])
    out = A.paired_user_bootstrap({"which": "a"}, {"which": "b"}, n_boot=800)
    assert out["delta"] == pytest.approx(0.02, abs=1e-9)
    assert out["lo"] > 0 and out["hi"] < 0.04       # tight, excludes zero
    assert out["share_users_improved"] == 1.0


def test_paired_user_bootstrap_returns_none_on_missing_or_mismatched_vectors(monkeypatch):
    import analyze as A
    monkeypatch.setattr(A, "load_per_user",
                        lambda rec, metric=A.METRIC, out_stem="full_random": None)
    assert A.paired_user_bootstrap({}, {}, n_boot=10) is None

    vecs = {"a": np.zeros(10), "b": np.zeros(7)}
    monkeypatch.setattr(A, "load_per_user",
                        lambda rec, metric=A.METRIC, out_stem="full_random":
                        vecs[rec["which"]])
    assert A.paired_user_bootstrap({"which": "a"}, {"which": "b"}, n_boot=10) is None


def test_paired_user_bootstrap_brackets_zero_for_a_null_effect(monkeypatch):
    import analyze as A
    rng = np.random.default_rng(1)
    a_vec = rng.normal(0.4, 0.2, 3000)
    b_vec = a_vec + rng.normal(0.0, 0.05, 3000)
    vecs = {"a": a_vec, "b": b_vec}
    monkeypatch.setattr(A, "load_per_user",
                        lambda rec, metric=A.METRIC, out_stem="full_random":
                        vecs[rec["which"]])
    out = A.paired_user_bootstrap({"which": "a"}, {"which": "b"}, n_boot=800)
    assert out["lo"] < 0 < out["hi"]


def test_reference_metrics_batching_does_not_change_the_answer():
    """The verifier batches to survive the large contexts; batching must be arithmetic-
    neutral or the gate it feeds is meaningless."""
    import verify
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(50, 120))
    truth = [set(rng.choice(120, size=2, replace=False).tolist()) for _ in range(50)]
    whole = verify.reference_metrics(scores, truth, cutoff=10)
    n_all, r_all = [], []
    for start in range(0, 50, 7):
        n, r = verify.reference_metrics(scores[start:start + 7],
                                        truth[start:start + 7], cutoff=10, reduce=False)
        n_all += n
        r_all += r
    assert float(np.mean(n_all)) == pytest.approx(whole[0])
    assert float(np.mean(r_all)) == pytest.approx(whole[1])
