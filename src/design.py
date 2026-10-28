"""Stage 1d - the borrowing design: which donor sets to fit, in what order, at what dose.

Two dose axes, kept independent so that *how much* is borrowed can be separated from
*which* donors are borrowed - two quantities that a single correlation between volume
and gain cannot separate.

  Axis 1, donor count : add donors one at a time along a fixed order.
                        `overlap-*` orders vary relevance at roughly matched volume;
                        `volume-*` orders vary volume at unmanaged relevance;
                        `random` repeats give the spread.
  Axis 2, donor volume: hold the donor set at the whole channel and subsample each donor
                        to a fraction of its training interactions.

`k = 0` is target-only and is the reference for every reported effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from util import stable_seed

# Bump whenever a change alters the numbers a unit produces. Resumption compares this,
# so stale records are re-run instead of being mistaken for completed work.
CODE_VERSION = 3

CHANNELS = ("XM", "XC", "XX")
ORDERS = ("overlap-desc", "overlap-asc", "volume-desc", "volume-asc")
FRACTIONS = (0.05, 0.1, 0.25, 0.5)
# Enumerating 58 XX donors one at a time is pointless; a log schedule covers the same
# range at a fraction of the cost.
XX_STEPS = (1, 4, 16)

# Orders that are meaningful per channel. For XX every donor has zero coupling strength by
# definition, so `overlap-desc` and `overlap-asc` degenerate to the same name-sorted
# sequence and would only duplicate work under two different labels.
CHANNEL_ORDERS = {
    "XM": ("overlap-desc", "overlap-asc", "volume-desc", "volume-asc"),
    "XC": ("overlap-desc", "overlap-asc", "volume-desc", "volume-asc"),
    "XX": ("volume-desc", "volume-asc"),
}
# Each XX step pools most of the grid, so repeats there are the most expensive units in
# the design and buy the least: with no coupling there is nothing for donor identity to
# vary, only volume.
CHANNEL_RANDOM_REPS = {"XM": 3, "XC": 3, "XX": 2}

# PureSVD costs 20-60x per unit what the other models cost, and accounts for ~93% of the
# whole grid's runtime. Repeats are therefore allocated by whether a cheaper instrument
# already covers the channel, rather than uniformly:
#
#   XM  - MostPop, ItemKNN and EASE all carry this channel at ~0.07 s/unit and supply the
#         donor-identity spread, so PureSVD needs only the deterministic orders plus one
#         random draw to confirm the latent-factor family behaves the same way.
#   XC  - ItemKNN-ext carries this channel cheaply, so the same argument applies.
#   XX  - PureSVD is the ONLY model that responds here, so its depth is preserved in full.
#
# This spends the expensive model where it is the sole instrument and economises where
# cheap instruments already measure the same thing.
MODEL_CHANNEL_REPS = {
    ("PureSVD", "XM"): 1,
    ("PureSVD", "XC"): 1,
}

# Which (model, channel) combinations can move the metric *at all*.
#
# This is not a heuristic. `tests/test_models.py` proves that a cross-category or
# doubly-uncoupled donor leaves the target-catalog Gram bit-identical, so MostPop,
# ItemKNN(basic) and EASE are mathematically blind to XC and XX. ItemKNN(extended) opens
# XC through shared users but is still blind to XX, where no user touches both sides.
# PureSVD responds everywhere - including XX, where the donor block shares no information
# at all and can only compete for a fixed rank budget. That makes XX x PureSVD a clean
# isolation of capacity-driven negative transfer with the information channel closed.
#
# Enumerating a full dose curve for a provably null combination would burn CPU to confirm
# a theorem, so those combinations get NULL_CHECK_STEPS units instead: enough to falsify
# the claim empirically, not enough to dominate the budget.
RESPONSIVE = {
    "XM": ("MostPop", "ItemKNN", "PureSVD", "EASE"),
    "XC": ("ItemKNN-ext", "PureSVD"),
    "XX": ("PureSVD",),
    "ALL": ("MostPop", "ItemKNN", "ItemKNN-ext", "PureSVD", "EASE"),
}
# Two different reasons a (model, channel) pair is left off the full dose curve, which the
# analysis must not conflate.
#
# BLIND: the pair is mathematically incapable of responding - a cross-category or
# doubly-uncoupled donor leaves the target-catalog Gram bit-identical. Expected effect is
# EXACTLY zero, and a non-zero value falsifies the mechanism model.
#
# REDUNDANT: the pair responds, but reproduces a cheaper model exactly. ItemKNN-ext differs
# from ItemKNN only through items the target's own users hold, so on an item-coupled donor
# the two are the same estimator. Expected effect equals that model's, NOT zero.
BLIND = {
    ("MostPop", "XC"), ("MostPop", "XX"),
    ("ItemKNN", "XC"), ("ItemKNN", "XX"),
    ("EASE", "XC"), ("EASE", "XX"),
    ("ItemKNN-ext", "XX"),
    ("Random", "XM"), ("Random", "XC"), ("Random", "XX"), ("Random", "ALL"),
}
REDUNDANT = {("ItemKNN-ext", "XM"): "ItemKNN"}

NULL_CHECK_STEPS = (1, 4)


@dataclass(frozen=True)
class Unit:
    split: str
    model: str
    target: tuple[str, str]
    channel: str
    order: str
    rep: int
    k: int
    frac: float
    donors: tuple[tuple[str, str], ...] = field(default=())

    def key(self) -> tuple:
        """Identity of a run unit, including WHICH donors and under which code version.

        Both additions guard resumption. Without the donor set, a `random:r` ordering is
        only as stable as the row order of `overlap_*.parquet`: regenerate that file and
        the same key silently maps onto a different donor set, while a resumed run keeps
        the stale record. Without the version, a resumed run happily treats records
        produced under an earlier model definition as complete - which is exactly what
        happened when 4,084 records survived the PureSVD capacity fix and had to be
        discarded by hand.
        """
        return (self.split, self.model, self.target, self.channel,
                self.order, self.rep, self.k, round(self.frac, 4),
                stable_seed(tuple(sorted(self.donors))), CODE_VERSION)


def overlap_strength(row: pd.Series) -> float:
    """How much of the target a donor can reach through its own coupling."""
    if row["channel"] == "XM":
        return float(row["shared_item_interaction_cover"])
    if row["channel"] == "XC":
        return float(row["shared_user_interaction_cover"])
    return 0.0


def donor_table(overlap: pd.DataFrame, target: tuple[str, str],
                channel: str) -> pd.DataFrame:
    sub = overlap[(overlap.target_market == target[0])
                  & (overlap.target_category == target[1])
                  & (overlap.channel == channel)].copy()
    sub["strength"] = sub.apply(overlap_strength, axis=1)
    return sub


def donor_orders(sub: pd.DataFrame, n_random: int, seed: int,
                 allowed=None) -> dict:
    """Named donor orderings; ties broken by context name so every order is reproducible."""
    names = list(zip(sub["donor_market"], sub["donor_category"]))
    strength = sub["strength"].to_numpy()
    volume = sub["donor_train"].to_numpy()
    tiebreak = np.array([f"{m}|{c}" for m, c in names])

    candidates = {
        "overlap-desc": np.lexsort((tiebreak, -strength)),
        "overlap-asc": np.lexsort((tiebreak, strength)),
        "volume-desc": np.lexsort((tiebreak, -volume)),
        "volume-asc": np.lexsort((tiebreak, volume)),
    }
    allowed = set(allowed) if allowed is not None else set(candidates)
    out = {k: v for k, v in candidates.items() if k in allowed}
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        out[f"random:{r}"] = rng.permutation(len(names))
    return {k: [names[j] for j in v] for k, v in out.items()}


def steps_for(channel: str, n_donors: int) -> list[int]:
    if channel == "XX":
        ks = [k for k in XX_STEPS if k < n_donors]
        return ks + [n_donors]
    return list(range(1, n_donors + 1))


def enumerate_units(overlap: pd.DataFrame, targets, models, split: str,
                    n_random: int = 3, seed: int = 0,
                    channels=CHANNELS, fractions=FRACTIONS) -> list[Unit]:
    """All run units for one split. `k = 0` appears once per (target, model)."""
    units: list[Unit] = []
    seen: set[tuple] = set()

    def add(unit: Unit):
        k = unit.key()
        if k not in seen:
            seen.add(k)
            units.append(unit)

    for target in targets:
        for model in models:
            add(Unit(split, model, target, "NONE", "none", 0, 0, 1.0, ()))

            full_sets: dict[str, tuple] = {}
            for channel in channels:
                sub = donor_table(overlap, target, channel)
                if sub.empty:
                    continue
                reps = min(n_random, CHANNEL_RANDOM_REPS.get(channel, n_random),
                           MODEL_CHANNEL_REPS.get((model, channel), n_random))
                orders = donor_orders(sub, reps, seed=stable_seed(target, channel),
                                      allowed=CHANNEL_ORDERS.get(channel))
                n = len(sub)
                ranked = orders.get("overlap-desc") or orders["volume-desc"]
                full_sets[channel] = tuple(ranked)
                responsive = model in RESPONSIVE.get(channel, ())

                if not responsive:
                    # Provably null: spot-check a few points instead of the whole curve.
                    for k in NULL_CHECK_STEPS:
                        if k < n:
                            add(Unit(split, model, target, channel, "null-check", 0, k,
                                     1.0, tuple(ranked[:k])))
                    add(Unit(split, model, target, channel, "full", 0, n, 1.0,
                             tuple(sorted(ranked))))
                    continue

                for order_name, sequence in orders.items():
                    rep = int(order_name.split(":")[1]) if ":" in order_name else 0
                    base = order_name.split(":")[0]
                    for k in steps_for(channel, n):
                        donors = tuple(sequence[:k])
                        if k == n:
                            # The complete donor set is identical under every order.
                            add(Unit(split, model, target, channel, "full", 0, k, 1.0,
                                     tuple(sorted(donors))))
                        else:
                            add(Unit(split, model, target, channel, base, rep, k, 1.0,
                                     donors))

                # Axis 2: whole channel, each donor subsampled.
                for frac in fractions:
                    for rep in range(reps):
                        add(Unit(split, model, target, channel, "subsample", rep, n,
                                 frac, tuple(sorted(ranked))))

                # NOTE - a `donor-only` arm was removed here. Withholding the
                # target's training edges left every evaluation user with an empty profile
                # row, so all profile-based models scored a zero vector and the ranking
                # fell entirely to the deterministic index tie-break: six different
                # (model, channel) combinations returned bit-identical NDCG per target.
                # It measured catalog index order, not transfer.
                #
                # The question it was meant to answer - how far borrowed data carries a
                # target on its own - is worth answering, but requires fitting parameters
                # on donor edges while still scoring each target user from THEIR OWN
                # profile (true zero-shot transfer). That needs the fit matrix and the
                # scoring matrix separated throughout `models.py`, and is deliberately
                # left as a separate change rather than shipped broken.

            if len(full_sets) == len(channels):
                merged = tuple(sorted(sum((list(v) for v in full_sets.values()), [])))
                add(Unit(split, model, target, "ALL", "full", 0, len(merged), 1.0, merged))
    return units
