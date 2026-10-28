"""The borrowing design enumerates what it claims to enumerate."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import design as Design  # noqa: E402
import runner as R  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGETS = [("uk", "Musical_Instruments"), ("mx", "Industrial_and_Scientific")]
MODELS = ["ItemKNN", "PureSVD"]


@pytest.fixture(scope="module")
def overlap():
    return pd.read_parquet(RESULTS / "overlap_random.parquet")


@pytest.fixture(scope="module")
def units(overlap):
    return Design.enumerate_units(overlap, TARGETS, MODELS, "random", n_random=2)


def test_units_are_unique(units):
    keys = [u.key() for u in units]
    assert len(keys) == len(set(keys))


def test_exactly_one_target_only_unit_per_target_and_model(units):
    zero = [u for u in units if u.k == 0]
    assert len(zero) == len(TARGETS) * len(MODELS)
    assert all(u.donors == () and u.channel == "NONE" for u in zero)


def test_donor_count_matches_k(units):
    for u in units:
        assert len(u.donors) == u.k


def test_donors_belong_to_the_declared_channel(units, overlap):
    for u in units:
        if u.channel in ("NONE", "ALL"):
            continue
        for donor in u.donors:
            same_market = donor[0] == u.target[0]
            same_category = donor[1] == u.target[1]
            if u.channel == "XM":
                assert same_category and not same_market
            elif u.channel == "XC":
                assert same_market and not same_category
            elif u.channel == "XX":
                assert not same_market and not same_category


def test_target_is_never_its_own_donor(units):
    for u in units:
        assert u.target not in u.donors


def test_orders_are_prefix_nested(overlap):
    """Adding a donor must extend the previous set, or the dose axis is not a dose."""
    units = Design.enumerate_units(overlap, [("uk", "Musical_Instruments")], ["ItemKNN"],
                                   "random", n_random=2)
    by_order: dict = {}
    for u in units:
        if u.channel == "XM" and u.order not in ("full", "subsample"):
            by_order.setdefault((u.order, u.rep), {})[u.k] = u.donors
    assert by_order
    for steps in by_order.values():
        for k in sorted(steps):
            if k - 1 in steps:
                assert steps[k][:k - 1] == steps[k - 1]


def test_full_donor_set_is_shared_across_orders(overlap):
    """The complete channel is one donor set, so it must be fitted once, not per order."""
    units = Design.enumerate_units(overlap, [("uk", "Musical_Instruments")], ["ItemKNN"],
                                   "random", n_random=3)
    xm_full = [u for u in units if u.channel == "XM" and u.order == "full"]
    assert len(xm_full) == 1


def test_subsample_units_use_the_whole_channel(overlap):
    units = Design.enumerate_units(overlap, TARGETS, ["ItemKNN"], "random", n_random=2)
    for u in units:
        if u.order == "subsample":
            assert u.frac < 1.0
            assert u.k == len(u.donors) > 0


def test_all_composite_is_the_union_of_the_three_channels(overlap):
    units = Design.enumerate_units(overlap, TARGETS, ["ItemKNN"], "random", n_random=1)
    for target in TARGETS:
        comp = [u for u in units if u.channel == "ALL" and u.target == target]
        if not comp:
            continue
        got = set(comp[0].donors)
        expected = set()
        for ch in Design.CHANNELS:
            sub = Design.donor_table(overlap, target, ch)
            expected |= set(zip(sub["donor_market"], sub["donor_category"]))
        assert got == expected


def test_orders_are_reproducible(overlap):
    a = Design.enumerate_units(overlap, TARGETS, MODELS, "random", n_random=2)
    b = Design.enumerate_units(overlap, TARGETS, MODELS, "random", n_random=2)
    assert [u.key() for u in a] == [u.key() for u in b]
    assert [u.donors for u in a] == [u.donors for u in b]


def test_xx_steps_are_log_spaced_and_end_at_the_full_set(overlap):
    steps = Design.steps_for("XX", 58)
    assert steps == list(Design.XX_STEPS) + [58]
    assert steps == sorted(set(steps))
    assert Design.steps_for("XM", 4) == [1, 2, 3, 4]


def test_xx_uses_only_non_degenerate_orders(overlap):
    """Overlap ordering is meaningless for a channel with no coupling."""
    units = Design.enumerate_units(overlap, [("uk", "Musical_Instruments")],
                                   ["PureSVD"], "random", n_random=3)
    xx_orders = {u.order for u in units if u.channel == "XX"}
    assert not any(o.startswith("overlap") for o in xx_orders)
    assert "volume-desc" in xx_orders


def test_channel_random_reps_are_respected(overlap):
    units = Design.enumerate_units(overlap, [("uk", "Musical_Instruments")],
                                   ["PureSVD"], "random", n_random=3)
    for channel, cap in Design.CHANNEL_RANDOM_REPS.items():
        reps = {u.rep for u in units if u.channel == channel and u.order == "random"}
        assert reps <= set(range(cap)), (channel, reps)


# ------------------------------------------------------------------- runner mechanics
def test_subsample_is_deterministic_and_proportional():
    import numpy as np
    u = np.arange(1000, dtype=np.int32)
    i = np.arange(1000, dtype=np.int32)
    a = R.subsample((u, i), 0.25, seed=7)
    b = R.subsample((u, i), 0.25, seed=7)
    assert len(a[0]) == 250
    np.testing.assert_array_equal(a[0], b[0])
    assert len(R.subsample((u, i), 1.0, seed=7)[0]) == 1000


def test_subsample_draws_without_replacement():
    import numpy as np
    u = np.arange(500, dtype=np.int32)
    out = R.subsample((u, u), 0.5, seed=1)
    assert len(np.unique(out[0])) == len(out[0])



def test_donor_only_arm_is_not_enumerated(overlap):
    """It was removed: with the target's own edges withheld, every
    evaluation user had an empty profile, so all profile-based models returned the
    index tie-break rather than a transfer measurement."""
    units = Design.enumerate_units(overlap, TARGETS, ["PureSVD", "ItemKNN"], "random",
                                   n_random=2)
    assert not [u for u in units if u.order == "donor-only"]


def test_every_unit_includes_the_target_own_training_edges():
    import numpy as np
    import experiment as E
    target = ("uk", "Musical_Instruments")
    donor = ("de", "Musical_Instruments")
    blocks = {
        target: E.ContextBlock(np.array([1, 2], np.int32), np.array([7, 8], np.int32),
                               np.array([], np.int32), np.array([], np.int32),
                               np.array([1], np.int32), np.array([9], np.int32),
                               np.array([7, 8, 9], np.int32)),
        donor: E.ContextBlock(np.array([5, 6], np.int32), np.array([7, 9], np.int32),
                              np.array([], np.int32), np.array([], np.int32),
                              np.array([], np.int32), np.array([], np.int32),
                              np.array([7, 9], np.int32)),
    }
    unit = Design.Unit("random", "PureSVD", target, "XM", "full", 0, 1, 1.0, (donor,))
    edges = R.unit_edge_lists(unit, blocks)
    assert len(edges) == 2
    np.testing.assert_array_equal(edges[0][0], np.array([1, 2]))


# ------------------------------------------------------------------- runner mechanics
def test_subsample_is_deterministic_and_proportional():
    import numpy as np
    u = np.arange(1000, dtype=np.int32)
    i = np.arange(1000, dtype=np.int32)
    a = R.subsample((u, i), 0.25, seed=7)
    b = R.subsample((u, i), 0.25, seed=7)
    assert len(a[0]) == 250
    np.testing.assert_array_equal(a[0], b[0])
    assert len(R.subsample((u, i), 1.0, seed=7)[0]) == 1000


def test_subsample_draws_without_replacement():
    import numpy as np
    u = np.arange(500, dtype=np.int32)
    out = R.subsample((u, u), 0.5, seed=1)
    assert len(np.unique(out[0])) == len(out[0])


def test_unit_key_distinguishes_different_donor_sets():
    """Two units differing only in WHICH donors they use must not share a key."""
    t = ("uk", "Musical_Instruments")
    a = Design.Unit("random", "ItemKNN", t, "XM", "random", 0, 1, 1.0, (("de", "X"),))
    b = Design.Unit("random", "ItemKNN", t, "XM", "random", 0, 1, 1.0, (("fr", "X"),))
    assert a.key() != b.key()


def test_unit_key_is_insensitive_to_donor_ordering():
    t = ("uk", "Musical_Instruments")
    d1, d2 = ("de", "X"), ("fr", "X")
    a = Design.Unit("random", "ItemKNN", t, "XM", "full", 0, 2, 1.0, (d1, d2))
    b = Design.Unit("random", "ItemKNN", t, "XM", "full", 0, 2, 1.0, (d2, d1))
    assert a.key() == b.key()


def test_unit_key_carries_the_code_version():
    t = ("uk", "Musical_Instruments")
    u = Design.Unit("random", "ItemKNN", t, "XM", "full", 0, 1, 1.0, (("de", "X"),))
    assert u.key()[-1] == Design.CODE_VERSION


def test_record_key_matches_unit_key_including_donors():
    t = ("uk", "Musical_Instruments")
    u = Design.Unit("random", "ItemKNN", t, "XM", "full", 0, 2, 1.0,
                    (("de", "X"), ("fr", "X")))
    rec = {
        "split": u.split, "model": u.model, "target_market": u.target[0],
        "target_category": u.target[1], "channel": u.channel, "order": u.order,
        "rep": u.rep, "k": u.k, "frac": u.frac,
        "donors": [list(x) for x in u.donors], "code_version": Design.CODE_VERSION,
    }
    assert R.record_key(rec) == u.key()


def test_stale_code_version_is_not_treated_as_complete():
    t = ("uk", "Musical_Instruments")
    u = Design.Unit("random", "ItemKNN", t, "XM", "full", 0, 1, 1.0, (("de", "X"),))
    stale = {
        "split": u.split, "model": u.model, "target_market": u.target[0],
        "target_category": u.target[1], "channel": u.channel, "order": u.order,
        "rep": u.rep, "k": u.k, "frac": u.frac,
        "donors": [list(x) for x in u.donors],
        "code_version": Design.CODE_VERSION - 1,
    }
    assert R.record_key(stale) != u.key()
