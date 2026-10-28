"""Model correctness and the borrowing contract.

The decisive tests here are the *provenance* ones: donor data must never reach evaluation,
`k = 0` must reproduce target-only exactly, and each model must move only through the
coupling it can physically use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import experiment as E  # noqa: E402
import models as Mod  # noqa: E402


def toy(n_users=40, n_items=12, density=0.25, seed=0):
    rng = np.random.default_rng(seed)
    R = (rng.random((n_users, n_items)) < density).astype(np.float32)
    R[:, 0] = 1.0  # one guaranteed-popular item
    return sp.csr_matrix(R)


ALL_MODELS = ["MostPop", "ItemKNN", "ItemKNN-ext", "PureSVD", "EASE"]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_score_shape_and_finiteness(name):
    R = toy()
    cols = np.arange(R.shape[1])
    model = Mod.build(name, R, cols, **Mod.GRIDS[name][0])
    rows = np.array([0, 1, 2])
    scores = model.score(rows)
    assert scores.shape == (3, len(cols))
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", ALL_MODELS)
def test_scoring_is_deterministic(name):
    R = toy()
    cols = np.arange(R.shape[1])
    rows = np.array([0, 5, 9])
    a = Mod.build(name, R, cols, **Mod.GRIDS[name][0]).score(rows)
    b = Mod.build(name, R, cols, **Mod.GRIDS[name][0]).score(rows)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_mostpop_ranks_by_count():
    R = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32))
    model = Mod.MostPop(R, np.arange(3))
    scores = model.score(np.array([0]))[0]
    assert scores[0] > scores[1] > scores[2]


def test_itemknn_has_no_self_similarity():
    R = toy()
    cols = np.arange(R.shape[1])
    model = Mod.ItemKNN(R, cols, topk=100, shrink=0.0)
    assert np.allclose(model.sim.toarray().diagonal(), 0.0)


def test_itemknn_topk_truncation_is_respected():
    R = toy(n_items=20, density=0.6)
    cols = np.arange(R.shape[1])
    model = Mod.ItemKNN(R, cols, topk=3, shrink=0.0)
    per_column = np.diff(model.sim.tocsc().indptr)
    assert per_column.max() <= 3


def test_itemknn_topk_keeps_the_strongest_neighbours():
    R = toy(n_items=15, density=0.5)
    cols = np.arange(R.shape[1])
    full = Mod.ItemKNN(R, cols, topk=10_000, shrink=0.0).sim.toarray()
    trunc = Mod.ItemKNN(R, cols, topk=2, shrink=0.0).sim.toarray()
    for c in range(full.shape[1]):
        kept = np.flatnonzero(trunc[:, c])
        if len(kept) == 0:
            continue
        threshold = full[kept, c].min()
        dropped = full[np.setdiff1d(np.arange(full.shape[0]), kept), c]
        assert dropped.max(initial=-np.inf) <= threshold + 1e-6


def test_ease_diagonal_is_zero_and_refuses_large_catalogs():
    R = toy()
    model = Mod.EASE(R, np.arange(R.shape[1]), lam=10.0)
    assert np.allclose(np.diag(model.B), 0.0)
    with pytest.raises(MemoryError):
        Mod.EASE(R, np.arange(Mod.EASE_MAX_ITEMS + 1))


def test_puresvd_rank_is_capped_by_matrix_size():
    R = toy(n_users=8, n_items=6)
    model = Mod.PureSVD(R, np.arange(6), rank=256)
    assert model.rank <= min(R.shape) - 1


# ------------------------------------------------------- the borrowing contract
def _blocks():
    """Target plus one item-coupled donor and one user-coupled donor."""
    rng = np.random.default_rng(3)
    t_users = np.arange(0, 30)
    t_items = np.arange(100, 115)
    tu = rng.choice(t_users, 200)
    ti = rng.choice(t_items, 200)

    # XM donor: different users, overlapping items.
    xm_u = rng.choice(np.arange(30, 60), 150)
    xm_i = rng.choice(t_items[:8], 150)

    # XC donor: the SAME users, disjoint items.
    xc_u = rng.choice(t_users, 150)
    xc_i = rng.choice(np.arange(200, 215), 150)
    return (tu, ti), (xm_u, xm_i), (xc_u, xc_i), t_items


def test_build_matrix_merges_shared_users_into_one_row():
    (tu, ti), _, (xc_u, xc_i), t_items = _blocks()
    R_own, _, _ = E.build_matrix([(tu, ti)], t_items, np.unique(tu))
    R_pool, _, _ = E.build_matrix([(tu, ti), (xc_u, xc_i)], t_items, np.unique(tu))
    # The cross-category donor shares every user, so no new rows may appear.
    assert R_pool.shape[0] == R_own.shape[0]
    # ...but its disjoint items must add columns.
    assert R_pool.shape[1] > R_own.shape[1]


def test_build_matrix_is_binary():
    (tu, ti), _, _, t_items = _blocks()
    R, _, _ = E.build_matrix([(tu, ti), (tu, ti)], t_items, np.unique(tu))
    assert set(np.unique(R.data)) == {1.0}


def test_target_catalog_is_unchanged_by_any_donor():
    """Evaluation ranks over the target catalog whatever is borrowed."""
    (tu, ti), (xm_u, xm_i), (xc_u, xc_i), t_items = _blocks()
    _, cols_own, _ = E.build_matrix([(tu, ti)], t_items, np.unique(tu))
    for donor in [(xm_u, xm_i), (xc_u, xc_i)]:
        R, cols, _ = E.build_matrix([(tu, ti), donor], t_items, np.unique(tu))
        assert len(cols) == len(cols_own) == len(t_items)


def test_cross_category_donors_cannot_move_target_catalog_gram():
    """A user-coupled donor adds only columns outside the catalog, so ItemKNN `basic`
    and EASE are structurally blind to it. This is a mechanism claim the experiment
    depends on, so it is asserted rather than assumed."""
    (tu, ti), _, (xc_u, xc_i), t_items = _blocks()
    R_own, cols_own, _ = E.build_matrix([(tu, ti)], t_items, np.unique(tu))
    R_pool, cols_pool, _ = E.build_matrix([(tu, ti), (xc_u, xc_i)], t_items, np.unique(tu))
    g_own = (R_own[:, cols_own].T @ R_own[:, cols_own]).toarray()
    g_pool = (R_pool[:, cols_pool].T @ R_pool[:, cols_pool]).toarray()
    np.testing.assert_allclose(g_own, g_pool)


def test_cross_market_donors_do_move_target_catalog_gram():
    (tu, ti), (xm_u, xm_i), _, t_items = _blocks()
    R_own, cols_own, _ = E.build_matrix([(tu, ti)], t_items, np.unique(tu))
    R_pool, cols_pool, _ = E.build_matrix([(tu, ti), (xm_u, xm_i)], t_items, np.unique(tu))
    g_own = (R_own[:, cols_own].T @ R_own[:, cols_own]).toarray()
    g_pool = (R_pool[:, cols_pool].T @ R_pool[:, cols_pool]).toarray()
    assert not np.allclose(g_own, g_pool)


def test_extended_itemknn_opens_the_cross_category_channel():
    """`extended=True` is what gives XC any pathway at all; without it the channel is
    closed by definition rather than by evidence."""
    (tu, ti), _, (xc_u, xc_i), t_items = _blocks()
    users = np.unique(tu)
    R_own, cols_own, rows_own = E.build_matrix([(tu, ti)], t_items, users)
    R_pool, cols_pool, rows_pool = E.build_matrix([(tu, ti), (xc_u, xc_i)], t_items, users)

    basic_own = Mod.ItemKNN(R_own, cols_own, topk=50, shrink=0.0, extended=False)
    basic_pool = Mod.ItemKNN(R_pool, cols_pool, topk=50, shrink=0.0, extended=False)
    np.testing.assert_allclose(basic_own.score(rows_own), basic_pool.score(rows_pool),
                               atol=1e-6)

    ext_own = Mod.ItemKNN(R_own, cols_own, topk=50, shrink=0.0, extended=True)
    ext_pool = Mod.ItemKNN(R_pool, cols_pool, topk=50, shrink=0.0, extended=True)
    assert not np.allclose(ext_own.score(rows_own), ext_pool.score(rows_pool), atol=1e-6)


def test_mostpop_is_blind_to_cross_category_donors():
    (tu, ti), _, (xc_u, xc_i), t_items = _blocks()
    users = np.unique(tu)
    R_own, cols_own, rows_own = E.build_matrix([(tu, ti)], t_items, users)
    R_pool, cols_pool, rows_pool = E.build_matrix([(tu, ti), (xc_u, xc_i)], t_items, users)
    np.testing.assert_allclose(
        Mod.MostPop(R_own, cols_own).score(rows_own),
        Mod.MostPop(R_pool, cols_pool).score(rows_pool))


def test_random_control_is_near_chance_and_reproducible():
    """The control must be uninformative: it is the reference the floor is read against."""
    R = toy(n_users=30, n_items=40)
    cols = np.arange(R.shape[1])
    a = Mod.RandomRec(R, cols).score(np.array([0, 1, 2]))
    b = Mod.RandomRec(R, cols).score(np.array([0, 1, 2]))
    np.testing.assert_allclose(a, b)
    assert a.shape == (3, 40)
    # It must not depend on the interaction data at all.
    R2 = toy(n_users=30, n_items=40, seed=99)
    c = Mod.RandomRec(R2, cols).score(np.array([0, 1, 2]))
    np.testing.assert_allclose(a, c)


def test_extended_itemknn_equals_basic_when_only_items_are_shared():
    """With an item-coupled donor the target's users hold no new items, so the extended
    profile space collapses to the basic one. If these ever diverge, the extended variant
    is spending neighbour slots on unreachable items."""
    (tu, ti), (xm_u, xm_i), _, t_items = _blocks()
    users = np.unique(tu)
    R, cols, rows = E.build_matrix([(tu, ti), (xm_u, xm_i)], t_items, users)
    basic = Mod.build("ItemKNN", R, cols, user_rows=rows, topk=50, shrink=0.0,
                      extended=False)
    ext = Mod.build("ItemKNN-ext", R, cols, user_rows=rows, topk=50, shrink=0.0,
                    extended=True)
    np.testing.assert_allclose(basic.score(rows), ext.score(rows), atol=1e-6)


def test_extended_itemknn_profile_space_excludes_untouched_donor_items():
    (tu, ti), (xm_u, xm_i), _, t_items = _blocks()
    users = np.unique(tu)
    R, cols, rows = E.build_matrix([(tu, ti), (xm_u, xm_i)], t_items, users)
    ext = Mod.build("ItemKNN-ext", R, cols, user_rows=rows, topk=50, shrink=0.0,
                    extended=True)
    # Nothing outside what the target's users hold, plus the catalog itself.
    touched = np.unique(R[rows].indices)
    assert set(ext.profile_cols) == set(np.union1d(touched, cols))


def test_extended_itemknn_still_differs_when_users_are_shared():
    (tu, ti), _, (xc_u, xc_i), t_items = _blocks()
    users = np.unique(tu)
    R, cols, rows = E.build_matrix([(tu, ti), (xc_u, xc_i)], t_items, users)
    basic = Mod.build("ItemKNN", R, cols, user_rows=rows, topk=50, shrink=0.0,
                      extended=False)
    ext = Mod.build("ItemKNN-ext", R, cols, user_rows=rows, topk=50, shrink=0.0,
                    extended=True)
    assert not np.allclose(basic.score(rows), ext.score(rows), atol=1e-6)


def test_blocked_similarity_matches_an_unblocked_reference():
    """Blocking is a memory optimisation and must not change a single value."""
    R = toy(n_users=120, n_items=60, density=0.3, seed=11)
    cols = np.arange(R.shape[1])
    reference = Mod._cosine_similarity_topk(R, cols, cols, shrink=5.0, topk=7)
    saved = Mod.SIM_BLOCK_BUDGET
    try:
        Mod.SIM_BLOCK_BUDGET = 64  # forces many tiny blocks
        blocked = Mod._cosine_similarity_topk(R, cols, cols, shrink=5.0, topk=7)
    finally:
        Mod.SIM_BLOCK_BUDGET = saved
    np.testing.assert_allclose(reference.toarray(), blocked.toarray(), atol=0, rtol=0)


def test_similarity_shape_and_dtype():
    R = toy(n_users=50, n_items=30)
    target = np.arange(10)
    sim = Mod._cosine_similarity_topk(R, np.arange(30), target, shrink=0.0, topk=5)
    assert sim.shape == (30, 10)
    assert sim.dtype == np.float32
    assert np.diff(sim.tocsc().indptr).max() <= 5


def test_itemknn_scores_stay_float32():
    R = toy(n_users=40, n_items=20)
    cols = np.arange(20)
    scores = Mod.ItemKNN(R, cols, topk=5, shrink=0.0).score(np.array([0, 1]))
    assert scores.dtype == np.float32


def test_random_control_does_not_depend_on_batch_boundaries():
    """A model whose scores change with the evaluation batch size disagrees with itself."""
    R = toy(n_users=60, n_items=25)
    cols = np.arange(25)
    model = Mod.RandomRec(R, cols)
    rows = np.arange(20)
    whole = model.score(rows)
    halves = np.vstack([model.score(rows[:7]), model.score(rows[7:])])
    np.testing.assert_allclose(whole, halves)


def test_puresvd_rank_stays_well_below_the_catalog_dimension():
    """At rank ~ catalog size the SVD reconstructs the data and the ranking degenerates
    into tie-breaking. The cap is what stops validation selection from choosing that."""
    R = toy(n_users=200, n_items=100, density=0.2)
    model = Mod.PureSVD(R, np.arange(100), rank=256)
    assert model.rank <= 50
    scores = model.score(np.arange(20))
    # A genuine low-rank model gives a user many distinct scores, not two or three.
    distinct = np.median([len(np.unique(np.round(r, 6))) for r in scores])
    assert distinct > 20, f"degenerate reconstruction: {distinct} distinct scores"


def test_puresvd_rank_cap_handles_tiny_matrices():
    R = toy(n_users=6, n_items=4, density=0.5)
    assert Mod.PureSVD(R, np.arange(4), rank=64).rank >= 1


def test_shared_factors_reproduce_per_rank_decompositions():
    """Slicing one decomposition must give exactly the model a per-rank solve would.

    This is what makes the rank grid affordable; if it were only approximately true the
    saving would be bought with silent error.
    """
    R = toy(n_users=200, n_items=80, density=0.25, seed=4)
    cols = np.arange(80)
    rows = np.arange(30)
    top = 40
    shared = Mod.PureSVD.factorize(R, top, seed=0)
    for rank in (8, 16, 32):
        alone = Mod.PureSVD(R, cols, rank=rank, seed=0)
        sliced = Mod.PureSVD(R, cols, rank=rank, seed=0, factors=shared)
        assert alone.rank == sliced.rank == rank
        # Singular vectors are sign-ambiguous, so compare the scores they produce.
        np.testing.assert_allclose(alone.score(rows), sliced.score(rows),
                                   rtol=1e-3, atol=1e-4)


def test_factorize_orders_columns_by_singular_value():
    R = toy(n_users=150, n_items=60, density=0.3, seed=5)
    V = Mod.PureSVD.factorize(R, 20, seed=0)
    energies = np.linalg.norm(R.toarray() @ V, axis=0)
    assert np.all(np.diff(energies) <= 1e-3), "columns are not in descending order"


def test_dense_and_sparse_factorization_routes_agree():
    """The dense Gram route is an exact shortcut, not an approximation: PureSVD scores
    must match whichever path produced the factors."""
    R = toy(n_users=300, n_items=120, density=0.2, seed=8)
    cols = np.arange(120)
    rows = np.arange(40)
    saved = Mod.DENSE_GRAM_MAX_ITEMS
    try:
        Mod.DENSE_GRAM_MAX_ITEMS = 10_000          # force the dense route
        dense = Mod.PureSVD(R, cols, rank=20, seed=0)
        Mod.DENSE_GRAM_MAX_ITEMS = 0               # force the iterative route
        sparse = Mod.PureSVD(R, cols, rank=20, seed=0)
    finally:
        Mod.DENSE_GRAM_MAX_ITEMS = saved
    np.testing.assert_allclose(dense.score(rows), sparse.score(rows),
                               rtol=1e-2, atol=1e-3)


def test_dense_route_orders_columns_by_singular_value():
    R = toy(n_users=200, n_items=90, density=0.25, seed=9)
    saved = Mod.DENSE_GRAM_MAX_ITEMS
    try:
        Mod.DENSE_GRAM_MAX_ITEMS = 10_000
        V = Mod.PureSVD.factorize(R, 15, seed=0)
    finally:
        Mod.DENSE_GRAM_MAX_ITEMS = saved
    energies = np.linalg.norm(R.toarray() @ V, axis=0)
    assert np.all(np.diff(energies) <= 1e-3)


def test_random_control_is_invariant_to_the_donor_set():
    """The control ignores the data, so borrowing must not move it by a single digit.

    It previously did: scores were keyed on pooled row position, which shifts when donors
    add users, so a model that reads nothing showed transfer effects up to 0.04 NDCG@10.
    """
    (tu, ti), (xm_u, xm_i), (xc_u, xc_i), t_items = _blocks()
    users = np.unique(tu)
    out = []
    for donors in ([], [(xm_u, xm_i)], [(xm_u, xm_i), (xc_u, xc_i)]):
        R, cols, rows = E.build_matrix([(tu, ti)] + donors, t_items, users)
        model = Mod.build("Random", R, cols,
                          user_ids=E.row_to_user_id(rows, users))
        out.append(model.score(rows))
    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], out[2])


def test_row_to_user_id_round_trips():
    rows = np.array([3, 7, 11])
    users = np.array([100, 200, 300])
    table = E.row_to_user_id(rows, users)
    np.testing.assert_array_equal(table[rows], users)


def test_puresvd_capacity_is_fixed_by_the_target_not_the_pooled_matrix():
    """The reachable rank grid must be identical with and without donors.

    Capping by pooled dimensions let de Grocery reach [16,31,31,31] target-only and
    [16,32,64,128] with sixteen donors, so the two arms compared different model
    capacities - the exact confound the closed-form design exists to remove.
    """
    small = toy(n_users=70, n_items=60, density=0.3, seed=12)
    pooled = toy(n_users=5000, n_items=4000, density=0.01, seed=13)
    capacity = min(small.shape)
    grid = (16, 32, 64, 128)
    alone = [Mod.PureSVD.max_rank(small, r, capacity) for r in grid]
    with_donors = [Mod.PureSVD.max_rank(pooled, r, capacity) for r in grid]
    assert alone == with_donors
    # Without the capacity argument the two diverge, which is the bug this guards.
    assert [Mod.PureSVD.max_rank(small, r) for r in grid] != \
           [Mod.PureSVD.max_rank(pooled, r) for r in grid]


def test_puresvd_numerical_route_can_be_pinned_across_arms():
    """The dense and iterative routes must not be chosen by pooled catalog size, or a
    target-only fit takes the exact path while every donor fit takes the iterative one."""
    R = toy(n_users=200, n_items=100, density=0.25, seed=14)
    forced_sparse = Mod.PureSVD.factorize(R, 12, seed=0, dense=False)
    forced_dense = Mod.PureSVD.factorize(R, 12, seed=0, dense=True)
    assert forced_sparse.shape == forced_dense.shape == (100, 12)
