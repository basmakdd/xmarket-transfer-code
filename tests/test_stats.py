"""Statistical machinery against known answers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import stats as S  # noqa: E402


def test_bh_matches_a_worked_example():
    # Benjamini & Hochberg's own ordering example.
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    adj = S.benjamini_hochberg(p)
    assert adj == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05])


def test_bh_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    adj = S.benjamini_hochberg(p)
    assert np.all(adj >= p - 1e-12)
    assert np.all((adj >= 0) & (adj <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bh_preserves_input_order_and_ignores_nan():
    p = np.array([0.5, np.nan, 0.001])
    adj = S.benjamini_hochberg(p)
    assert np.isnan(adj[1])
    assert adj[2] < adj[0]


def test_cluster_bootstrap_recovers_the_mean():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "delta": rng.normal(0.05, 0.01, 400),
        "target_market": rng.choice(list("abcdefg"), 400),
    })
    res = S.cluster_bootstrap(df, "delta", "target_market", n_boot=500)
    assert res["mean"] == pytest.approx(df["delta"].mean())
    assert res["lo"] < res["mean"] < res["hi"]
    assert res["n_clusters"] == 7


def test_cluster_bootstrap_is_wider_than_ignoring_clusters():
    """Clustered data really does carry less information than n suggests."""
    rng = np.random.default_rng(2)
    # Strong market-level effects: within a market, observations are near-identical.
    offsets = dict(zip("abcdefg", rng.normal(0, 0.1, 7)))
    markets = rng.choice(list("abcdefg"), 700)
    df = pd.DataFrame({
        "delta": [offsets[m] + rng.normal(0, 0.001) for m in markets],
        "target_market": markets,
    })
    clustered = S.cluster_bootstrap(df, "delta", "target_market", n_boot=800)
    naive = df["delta"].std(ddof=1) / np.sqrt(len(df)) * 1.96
    assert (clustered["hi"] - clustered["lo"]) / 2 > naive * 3


def test_cluster_bootstrap_handles_empty_input():
    df = pd.DataFrame({"delta": [np.nan], "target_market": ["a"]})
    res = S.cluster_bootstrap(df, "delta", "target_market", n_boot=10)
    assert res["n"] == 0 and np.isnan(res["mean"])


def test_paired_summary_signs_and_effect_size():
    df = pd.DataFrame({
        "delta": [0.1, 0.2, 0.3, -0.05, 0.15, 0.25, 0.05, 0.2],
        "target_market": list("aabbccdd"),
    })
    out = S.paired_summary(df, "delta")
    assert out["share_positive"] == pytest.approx(7 / 8)
    assert out["rank_biserial"] > 0.5  # mostly positive ranks
    assert out["wilcoxon_p"] < 0.05


def test_paired_summary_on_a_null_effect():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"delta": rng.normal(0, 0.01, 60),
                       "target_market": rng.choice(list("abcd"), 60)})
    out = S.paired_summary(df, "delta")
    assert out["wilcoxon_p"] > 0.05
    assert abs(out["rank_biserial"]) < 0.4


def test_spearman_ci_detects_a_real_relationship():
    rng = np.random.default_rng(4)
    x = rng.normal(size=120)
    y = -x + rng.normal(scale=0.3, size=120)
    res = S.spearman_with_ci(x, y, n_boot=400)
    assert res["rho"] < -0.8
    assert res["hi"] < 0  # interval excludes zero


def test_spearman_ci_brackets_zero_when_unrelated():
    rng = np.random.default_rng(5)
    res = S.spearman_with_ci(rng.normal(size=100), rng.normal(size=100), n_boot=400)
    assert res["lo"] < 0 < res["hi"]


def test_spearman_ci_refuses_tiny_samples():
    assert np.isnan(S.spearman_with_ci([1, 2], [3, 4])["rho"])


def test_dose_model_recovers_a_known_slope():
    rng = np.random.default_rng(6)
    n = 600
    market = rng.choice(list("abcdef"), n)
    category = rng.choice(list("xyz"), n)
    dose = rng.normal(size=n)
    delta = -0.03 * dose + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"delta": delta, "log_borrowed": dose,
                       "target_market": market, "target_category": category})
    out = S.dose_model(df)
    row = out[out.term == "log_borrowed"].iloc[0]
    assert row["coef"] == pytest.approx(-0.03, abs=0.01)
    assert row["hi"] < 0


def test_dose_model_returns_empty_without_variation():
    df = pd.DataFrame({"delta": [0.1] * 5, "log_borrowed": [1.0] * 5,
                       "target_market": ["a"] * 5, "target_category": ["x"] * 5})
    assert S.dose_model(df).empty
