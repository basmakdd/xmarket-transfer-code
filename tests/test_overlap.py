"""Checks on the structural overlap map.

These tests do more than guard the code: they *empirically verify* the coupling structure
the paper relies on (shared products across markets, shared users across categories).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import overlap as O  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def pairs():
    return pd.read_parquet(RESULTS / "overlap_random.parquet")


def test_every_ordered_pair_present(pairs):
    assert len(pairs) == 74 * 73


def test_channel_labels_are_exhaustive(pairs):
    assert set(pairs["channel"]) == {"XM", "XC", "XX"}


def test_xm_pairs_share_items_and_never_users(pairs):
    """Items are shared across markets within a category."""
    xm = pairs[pairs.channel == "XM"]
    assert (xm["shared_users"] == 0).all()
    assert (xm["shared_items"] > 0).mean() > 0.9


def test_xc_pairs_share_users_and_never_items(pairs):
    """Users are shared across categories within a market."""
    xc = pairs[pairs.channel == "XC"]
    assert (xc["shared_items"] == 0).all()
    assert (xc["shared_users"] > 0).mean() > 0.8


def test_xx_pairs_share_nothing(pairs):
    """A context that differs in both market and category has no direct coupling at all.

    So not every auxiliary context is connected through exactly one coupling: 4,298 of the
    5,402 ordered pairs share neither users nor products.
    """
    xx = pairs[pairs.channel == "XX"]
    assert (xx["shared_items"] == 0).all()
    assert (xx["shared_users"] == 0).all()
    assert len(xx) == 4298


def test_uncoupled_pairs_dominate_the_all_composition(pairs):
    """rho_ALL is mostly donors that share nothing with the target."""
    assert (pairs["channel"] == "XX").mean() > 0.75


def test_overlap_shares_are_proportions(pairs):
    for col in ["item_overlap_target", "user_overlap_target",
                "shared_item_interaction_cover", "shared_user_interaction_cover",
                "shared_test_item_cover"]:
        assert pairs[col].between(0, 1).all(), col


def test_channel_function_is_symmetric_in_label():
    assert O.channel(("uk", "Books"), ("de", "Books")) == "XM"
    assert O.channel(("uk", "Books"), ("uk", "Electronics")) == "XC"
    assert O.channel(("uk", "Books"), ("de", "Electronics")) == "XX"


def test_gini_bounds():
    assert O.gini(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    assert O.gini(np.array([0.0, 0.0, 9.0])) > 0.5


def test_js_divergence_properties():
    p = np.array([1.0, 1.0, 2.0])
    assert O.js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    # Disjoint supports give the maximum of 1 bit.
    assert O.js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data" / "interactions.parquet").exists(),
                    reason="needs data/interactions.parquet (python run.py data)")
def test_donor_volume_uses_training_split_only(pairs):
    """A donor may never contribute its validation or test interactions."""
    import data as D
    data = D.load()
    train = data[data.split_random == "train"]
    counts = train.groupby(["market", "category"]).size()
    row = pairs.iloc[0]
    assert row["donor_train"] == counts[(row["donor_market"], row["donor_category"])]
    assert row["target_train"] == counts[(row["target_market"], row["target_category"])]


def test_temporal_and_random_maps_agree_on_structure(pairs):
    other = pd.read_parquet(RESULTS / "overlap_temporal.parquet")
    key = ["target_market", "target_category", "donor_market", "donor_category"]
    merged = pairs.merge(other, on=key, suffixes=("_r", "_t"))
    assert (merged["channel_r"] == merged["channel_t"]).all()
    assert (merged["target_train_r"] == merged["target_train_t"]).all()
