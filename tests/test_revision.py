"""Tests of the null model, the bootstrap weights, the split rules and the foreign-data weighting.

The calendar-split test needs the preprocessed calendar data (python run.py data --only calendar) and is skipped
without it; every other test runs without data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "paper" / "code"))

import boot  # noqa: E402
import data as D  # noqa: E402
import experiment as E  # noqa: E402
import mustadd  # noqa: E402
import rq1_agree  # noqa: E402


# ------------------------------------------------------------------------------ curveball null model
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_curveball_preserves_margins_and_creates_no_duplicates(seed):
    rng = np.random.default_rng(seed)
    M = rng.random((40, 25)) < 0.2
    M[np.arange(40), rng.integers(0, 25, 40)] = True          # every user has at least one product
    uu, ii = np.nonzero(M)
    new_ii, trades = rq1_agree.curveball(uu, ii, np.random.default_rng(100 + seed))
    assert trades > 0
    assert len(new_ii) == len(ii)
    # row sums: every edge keeps its user; column sums: every product keeps its count
    assert np.array_equal(np.bincount(new_ii, minlength=25), np.bincount(ii, minlength=25))
    N = np.zeros_like(M)
    N[uu, new_ii] = True
    assert np.array_equal(N.sum(axis=1), M.sum(axis=1))
    assert np.array_equal(N.sum(axis=0), M.sum(axis=0))
    # no duplicate user-product pairs
    assert len(set(zip(uu.tolist(), new_ii.tolist()))) == len(uu)
    # the matrix is actually randomised
    assert not np.array_equal(N, M)


# ------------------------------------------------------------------------------ cluster bootstrap
def _indices(markets, categories, donors=None):
    countries = sorted(set(markets) | (set(donors) if donors is not None else set()))
    cats = sorted(set(categories))
    mi = np.array([countries.index(x) for x in markets])
    ci = np.array([cats.index(x) for x in categories])
    di = None if donors is None else np.array([countries.index(x) for x in donors])
    return countries, cats, mi, ci, di


def test_cboot_pair_weights_share_the_country_draw():
    markets = ["ca", "de", "ca", "fr", "de"]
    donors = ["de", "ca", "fr", "ca", "fr"]
    cats = ["Books", "Books", "Toys", "Toys", "Books"]
    countries, cl, mi, ci, di = _indices(markets, cats, donors)
    gen = boot.replicate_weights(np.random.default_rng(1), len(countries), len(cl), mi, ci, di)
    for _ in range(300):
        w, wm, wc = next(gen)
        # one country draw serves both the target and the donor role
        assert np.array_equal(w, wm[mi] * wm[di] * wc[ci])
        # A->B and B->A (same category) always get the same weight
        assert w[0] == w[1]


def test_cboot_context_weights_and_zero_weight_redraw():
    markets = ["ca", "de"]
    cats = ["Books", "Toys"]
    countries, cl, mi, ci, _ = _indices(markets, cats)
    gen = boot.replicate_weights(np.random.default_rng(2), len(countries), len(cl), mi, ci)
    for _ in range(500):
        w, wm, wc = next(gen)
        assert np.array_equal(w, wm[mi] * wc[ci])
        assert w.sum() > 0                                  # zero-weight replicates are never returned
    # with 2 countries x 2 categories and observations on the diagonal only, many raw draws have zero weight;
    # cboot must still return an interval built from B valid replicates
    lo, hi = boot.cboot([1.0, 2.0], markets, cats, stat="mean", label="t", B=200)
    assert 1.0 <= lo <= hi <= 2.0


def test_cboot_is_deterministic_per_label():
    rng = np.random.default_rng(3)
    vals = rng.normal(size=30)
    mk = rng.choice(["ca", "de", "fr", "in", "jp", "mx", "uk"], 30)
    ct = rng.choice(["Books", "Toys", "Office"], 30)
    a = boot.cboot(vals, mk, ct, stat="mean", label="x", B=300)
    b = boot.cboot(vals, mk, ct, stat="mean", label="x", B=300)
    c = boot.cboot(vals, mk, ct, stat="mean", label="y", B=300)
    assert a == b
    assert a != c                                            # another label gives other draws
    # the order of other calls does not matter
    boot.cboot(vals, mk, ct, stat="mean", label="z", B=50)
    assert boot.cboot(vals, mk, ct, stat="mean", label="x", B=300) == a


# ------------------------------------------------------------------------------ per-user split sizes
@pytest.mark.parametrize("n", range(1, 11))
def test_cut_points(n):
    n_train, n_val = D.cut_points(n)
    n_test = n - n_train - n_val
    assert n_train >= 1 and n_val >= 0 and n_test >= 0
    if n > 1:
        assert n_test >= 1
    else:
        assert (n_train, n_val, n_test) == (1, 0, 0)
    # a validation interaction exists only for users with at least 8 interactions
    assert (n_val >= 1) == (n >= 8)


# ------------------------------------------------------------------------------ calendar split
CAL = D.OUT_DIR / "interactions_calendar.parquet"
CUTS = D.RESULTS_DIR / "calendar_cutoffs.csv"


@pytest.mark.skipif(not (CAL.exists() and CUTS.exists()),
                    reason="needs data/interactions_calendar.parquet (python run.py data --only calendar)")
def test_calendar_split_respects_the_category_cutoffs():
    d = pd.read_parquet(CAL, columns=["category", "date", "split_calendar"])
    c = pd.read_csv(CUTS)
    d = d.merge(c, on="category", how="left")
    assert d.t_val.notna().all()
    date = pd.to_datetime(d.date)
    t_val, t_test = pd.to_datetime(d.t_val), pd.to_datetime(d.t_test)
    s = d.split_calendar
    assert set(s.unique()) <= {"train", "val", "test"}
    assert (date[s == "train"] < t_val[s == "train"]).all()
    v = s == "val"
    assert ((date[v] >= t_val[v]) & (date[v] < t_test[v])).all()
    assert (date[s == "test"] >= t_test[s == "test"]).all()


# ------------------------------------------------------------------------------ foreign-data weighting
def test_weight_rows_scales_only_foreign_rows():
    rng = np.random.default_rng(4)
    local = (np.array([0, 0, 1, 2, 2, 3]), np.array([0, 1, 1, 2, 3, 0]))
    foreign1 = (np.array([10, 10, 11]), np.array([0, 1, 4]))
    foreign2 = (np.array([20, 21, 21]), np.array([2, 3, 4]))
    el = [local, foreign1, foreign2]
    catalog = np.arange(5)
    eval_users = np.array([0, 1, 2, 3])
    R, cols, rows = E.build_matrix(el, catalog, eval_users)
    alpha = float(rng.choice([0.1, 0.25, 0.5]))
    W = mustadd.weight_rows(R, el, eval_users, [1, alpha, alpha]).toarray()
    R = R.toarray()
    all_u = np.unique(np.concatenate([e[0] for e in el] + [eval_users]))
    is_foreign = np.isin(all_u, np.concatenate([foreign1[0], foreign2[0]]))
    assert np.array_equal(W[~is_foreign], R[~is_foreign])               # local rows unchanged
    assert np.allclose(W[is_foreign], alpha * R[is_foreign])             # foreign rows times alpha
    # weights [1, 1, 1] (and None) return the matrix itself
    assert mustadd.weight_rows(sp.csr_matrix(R), el, eval_users, [1, 1, 1]).toarray().tolist() == R.tolist()
    assert mustadd.weight_rows(sp.csr_matrix(R), el, eval_users, None).toarray().tolist() == R.tolist()
    # per-donor weights: only that donor's rows change
    W2 = mustadd.weight_rows(sp.csr_matrix(R), el, eval_users, [1, alpha, 1]).toarray()
    f1 = np.isin(all_u, foreign1[0])
    assert np.allclose(W2[f1], alpha * R[f1]) and np.array_equal(W2[~f1], R[~f1])
