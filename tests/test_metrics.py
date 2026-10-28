"""Metric correctness against hand-computed values."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics as M  # noqa: E402


def test_dcg_weights():
    assert M.dcg_weights(3) == pytest.approx([1.0, 1 / math.log2(3), 0.5])


def test_single_user_hand_computed():
    # Five items, scores descending by index: the ranking is 0,1,2,3,4.
    scores = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    # Held-out items at ranks 2 and 4 (0-indexed positions 1 and 3).
    truth = [np.array([1, 3])]
    res = M.rank_metrics(scores, truth, cutoffs=(2, 5))

    # DCG@2 counts only the rank-2 hit: 1/log2(3).
    idcg2 = 1.0 + 1 / math.log2(3)
    assert res["ndcg@2"][0] == pytest.approx((1 / math.log2(3)) / idcg2)
    assert res["recall@2"][0] == pytest.approx(0.5)

    # DCG@5 counts ranks 2 and 4: 1/log2(3) + 1/log2(5).
    dcg5 = 1 / math.log2(3) + 1 / math.log2(5)
    assert res["ndcg@5"][0] == pytest.approx(dcg5 / idcg2)
    assert res["recall@5"][0] == pytest.approx(1.0)
    # First hit is at rank 2.
    assert res["mrr@5"][0] == pytest.approx(0.5)


def test_perfect_and_empty_rankings():
    scores = np.array([[3.0, 2.0, 1.0, 0.0]])
    assert M.rank_metrics(scores, [np.array([0])], cutoffs=(4,))["ndcg@4"][0] == 1.0
    # A held-out item ranked last inside a top-4 cutoff still counts for recall.
    res = M.rank_metrics(scores, [np.array([3])], cutoffs=(4,))
    assert res["recall@4"][0] == pytest.approx(1.0)
    assert res["mrr@4"][0] == pytest.approx(0.25)


def test_masked_items_never_retrieved():
    """A model that loves training items must score zero once they are masked."""
    scores = np.array([[9.0, 9.0, 1.0, 2.0]])
    scores[0, :2] = M.MASK_VALUE  # items 0,1 are training positives
    res = M.rank_metrics(scores, [np.array([0])], cutoffs=(2,))
    assert res["recall@2"][0] == 0.0
    assert res["ndcg@2"][0] == 0.0


def test_ideal_dcg_caps_at_cutoff():
    # A user with 10 held-out items cannot exceed IDCG over the top 3.
    assert M.ideal_dcg(np.array([10]), 3)[0] == pytest.approx(
        1.0 + 1 / math.log2(3) + 0.5)


def test_candidate_sets_preserve_every_positive():
    rng = np.random.default_rng(0)
    n_items = 500
    scores = rng.normal(size=(20, n_items))
    truth = [rng.choice(n_items, size=3, replace=False) for _ in range(20)]
    sets = M.build_candidate_sets(scores, truth, [50], np.random.default_rng(1))
    sub, sub_truth = sets[50]
    assert sub.shape == (20, 50)
    assert all(len(t) == 3 for t in sub_truth)
    # Each recorded positive column really does hold that user's positive score.
    for r in range(20):
        assert sorted(sub[r, sub_truth[r]]) == sorted(scores[r, truth[r]])


def test_candidate_sets_do_not_hand_ties_to_the_positives():
    """The defect this guards inflated NDCG@10 29x purely from column position.

    Every held-out item ties with every negative at exactly 0.0, so a correct
    implementation must score near the chance rate of 10/size, not near 1.
    """
    n_users, n_items, size = 150, 400, 100
    rng = np.random.default_rng(3)
    scores = np.zeros((n_users, n_items), dtype=np.float64)
    truth = [rng.choice(n_items, size=2, replace=False) for _ in range(n_users)]
    sets = M.build_candidate_sets(scores, truth, [size], np.random.default_rng(4))
    sub, sub_truth = sets[size]
    got = M.rank_metrics(sub, sub_truth, cutoffs=(10,))["ndcg@10"].mean()
    assert got < 0.25, f"positives are winning ties by position: NDCG@10 = {got:.3f}"


def test_candidate_sets_are_actually_nested():
    """The smaller set must be a subset of the larger, or the across-size comparison
    is not a controlled contrast."""
    rng = np.random.default_rng(5)
    n_items = 800
    scores = rng.normal(size=(12, n_items))
    truth = [rng.choice(n_items, size=2, replace=False) for _ in range(12)]
    sets = M.build_candidate_sets(scores, truth, [50, 200], np.random.default_rng(6))
    small, _ = sets[50]
    large, _ = sets[200]
    for r in range(12):
        s_vals = sorted(v for v in small[r] if np.isfinite(v))
        l_vals = sorted(v for v in large[r] if np.isfinite(v))
        # Every score in the small set appears in the large set.
        for v in s_vals:
            assert v in l_vals


def test_candidate_sets_shrinking_still_makes_the_task_easier():
    rng = np.random.default_rng(7)
    n_items = 500
    scores = rng.normal(size=(30, n_items))
    truth = [rng.choice(n_items, size=3, replace=False) for _ in range(30)]
    full = M.rank_metrics(scores, truth, cutoffs=(10,))["ndcg@10"].mean()
    sub, sub_truth = M.build_candidate_sets(scores, truth, [50],
                                            np.random.default_rng(8))[50]
    reduced = M.rank_metrics(sub, sub_truth, cutoffs=(10,))["ndcg@10"].mean()
    assert reduced >= full


def test_candidate_sets_are_a_noop_when_larger_than_the_catalog():
    scores = np.random.default_rng(2).normal(size=(4, 10))
    truth = [np.array([0]) for _ in range(4)]
    out = M.build_candidate_sets(scores, truth, [99], np.random.default_rng(3))
    sub, sub_truth = out[99]
    assert sub is scores and sub_truth is truth


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(4).normal(loc=0.3, scale=0.1, size=500)
    mean, lo, hi = M.bootstrap_ci(values, n_boot=500, seed=7)
    assert lo < mean < hi
    assert mean == pytest.approx(values.mean())


def test_bootstrap_is_deterministic_under_seed():
    values = np.random.default_rng(5).normal(size=100)
    assert M.bootstrap_ci(values, n_boot=200, seed=1) == \
           M.bootstrap_ci(values, n_boot=200, seed=1)


def test_coverage_and_novelty():
    top = np.array([[0, 1], [0, 2]])
    assert M.catalog_coverage(top, 10) == pytest.approx(0.3)
    # A uniformly popular catalog of 4 items gives every item -log2(1/4) = 2 bits.
    assert M.mean_novelty(top, np.array([5.0, 5.0, 5.0, 5.0])) == pytest.approx(2.0)


def test_ties_are_broken_by_lower_catalog_index():
    """Without this, which of several equally-scored items reaches the top ten is decided
    by partition order rather than by the data - and MostPop scores are integer counts,
    so large tie blocks are the normal case, not an edge case."""
    scores = np.array([[1.0, 1.0, 1.0, 1.0, 0.0]])
    res = M.rank_metrics(scores, [np.array([0])], cutoffs=(1,))
    assert res["_top"][0, 0] == 0
    res = M.rank_metrics(scores, [np.array([3])], cutoffs=(1,))
    assert res["_top"][0, 0] == 0  # index 3 loses the tie to index 0


def test_ranking_matches_an_explicit_sort_reference_under_heavy_ties():
    rng = np.random.default_rng(0)
    # Integer scores over a small catalog: ties everywhere, as with popularity.
    scores = rng.integers(0, 4, size=(25, 40)).astype(np.float32)
    res = M.rank_metrics(scores, [np.array([0]) for _ in range(25)], cutoffs=(10,))
    for r in range(25):
        expected = sorted(range(40), key=lambda j: (-scores[r, j], j))[:10]
        assert res["_top"][r].tolist() == expected


def test_tie_break_does_not_reorder_genuinely_distinct_scores():
    scores = np.array([[0.5000001, 0.5, 0.4999999]], dtype=np.float64)
    res = M.rank_metrics(scores, [np.array([0])], cutoffs=(3,))
    assert res["_top"][0].tolist() == [0, 1, 2]


def test_masked_entries_never_outrank_real_ones_after_tie_break():
    scores = np.array([[1.0, 1.0, 1.0]])
    scores[0, 0] = M.MASK_VALUE
    res = M.rank_metrics(scores, [np.array([0])], cutoffs=(3,))
    assert res["_top"][0, 0] != 0
