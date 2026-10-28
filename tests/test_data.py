"""Checks on the preprocessed grid."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D  # noqa: E402

pytestmark = pytest.mark.skipif(not (D.OUT_DIR / "interactions.parquet").exists(),
                                reason="needs data/interactions.parquet (python run.py data; see raw_data/README.md)")


@pytest.fixture(scope="module")
def grid():
    return D.load(), D.load_contexts()


# ----------------------------------------------------------------- structure of the grid
def test_grid_reproduces_the_published_context_set(grid):
    _, contexts = grid
    assert len(contexts) == 74
    assert contexts["category"].nunique() == 14
    assert contexts["market"].nunique() == 7
    # Digital Music and Kindle Store are the two categories the paper eliminates.
    assert "Kindle_Store" not in set(contexts["category"])
    assert "Digital_Music" not in set(contexts["category"])


# Spot values from the paper's dataset table, as (#users, #items).
TABLE1 = {
    ("de", "Arts_Crafts_and_Sewing"): (877, 102),
    ("in", "Arts_Crafts_and_Sewing"): (749, 42),
    ("uk", "Books"): (104881, 35324),
    ("ca", "Books"): (5790, 7564),
    ("uk", "Cell_Phones_and_Accessories"): (2555, 163),
    ("ca", "Grocery_and_Gourmet_Food"): (458, 221),
    ("mx", "Industrial_and_Scientific"): (334, 108),
    ("uk", "Movies_and_TV"): (24295, 3566),
    ("jp", "Musical_Instruments"): (227, 48),
    ("ca", "Electronics"): (17916, 3934),
}


@pytest.mark.parametrize("key,expected", TABLE1.items())
def test_counts_match_dataset_table(grid, key, expected):
    _, contexts = grid
    row = contexts[(contexts.market == key[0]) & (contexts.category == key[1])]
    assert len(row) == 1, f"context {key} missing"
    assert (int(row.iloc[0]["users"]), int(row.iloc[0]["items"])) == expected


def test_item_identities_are_category_bound(grid):
    """An item belongs to exactly one category."""
    data, _ = grid
    per_item = data.groupby("i")["category"].nunique()
    assert per_item.max() == 1


def test_user_identities_are_market_bound(grid):
    """A user lives in exactly one market."""
    data, _ = grid
    per_user = data.groupby("u")["market"].nunique()
    assert per_user.max() == 1


def test_no_duplicate_interactions(grid):
    data, _ = grid
    assert not data.duplicated(["market", "category", "u", "i"]).any()


# ------------------------------------------------------------------------------- splits
@pytest.mark.parametrize("split", ["split_random", "split_temporal"])
def test_every_interaction_has_exactly_one_split(grid, split):
    data, _ = grid
    assert set(data[split].unique()) <= {"train", "val", "test"}
    assert data[split].notna().all()


@pytest.mark.parametrize("split", ["split_random", "split_temporal"])
def test_every_context_has_train_and_test(grid, split):
    data, contexts = grid
    counts = data.groupby(["market", "category"])[split].value_counts().unstack(fill_value=0)
    assert (counts.get("train", 0) > 0).all()
    assert (counts.get("test", 0) > 0).all()


@pytest.mark.parametrize("split", ["split_random", "split_temporal"])
def test_split_proportions_are_close_to_80_10_10(grid, split):
    data, _ = grid
    share = data[split].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.8, abs=0.06)
    assert share.get("test", 0) == pytest.approx(0.1, abs=0.05)


def test_temporal_split_respects_time_ordering(grid):
    """For a sample of users, no training item is newer than a test item."""
    data, _ = grid
    rng = np.random.default_rng(0)
    multi = data.groupby(["market", "category", "u"]).size()
    candidates = multi[multi >= 5].index.to_numpy()
    picks = rng.choice(len(candidates), size=min(200, len(candidates)), replace=False)
    for idx in picks:
        market, category, user = candidates[idx]
        block = data[(data.market == market) & (data.category == category)
                     & (data.u == user)]
        train_max = block.loc[block.split_temporal == "train", "date"].max()
        test_min = block.loc[block.split_temporal == "test", "date"].min()
        if pd.isna(train_max) or pd.isna(test_min):
            continue
        assert train_max <= test_min


def test_random_and_temporal_splits_agree_on_sizes(grid):
    """Both variants use the same cut points, so per-user split sizes must match."""
    data, _ = grid
    a = data.groupby(["market", "category", "u"])["split_random"].value_counts().unstack(
        fill_value=0)
    b = data.groupby(["market", "category", "u"])["split_temporal"].value_counts().unstack(
        fill_value=0)
    assert a.equals(b)


def test_cut_points_guarantee_a_test_item():
    for n in range(2, 60):
        n_train, n_val = D.cut_points(n)
        assert n_train >= 1
        assert n - n_train - n_val >= 1, f"no test item for a user with {n} interactions"


def test_single_interaction_user_goes_to_train():
    n_train, n_val = D.cut_points(1)
    assert (n_train, n_val) == (1, 0)


# ------------------------------------------------------------------------- k-core logic
def test_k_core_is_idempotent_and_meets_the_threshold():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "user": rng.integers(0, 60, 1200).astype(str),
        "item": rng.integers(0, 40, 1200).astype(str),
    }).drop_duplicates()
    core = D.k_core(df, 5)
    assert D.k_core(core, 5).equals(core)
    if len(core):
        assert core["user"].value_counts().min() >= 5
        assert core["item"].value_counts().min() >= 5
