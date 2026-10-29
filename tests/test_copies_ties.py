"""Cross-market review copies, the temporal tie-break and month-precision dates (src/data.py, src/calendar_split.py).

Synthetic raw files only; no XMarket data needed.
"""
from __future__ import annotations

import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D  # noqa: E402
import calendar_split as CS  # noqa: E402

CAT = "Books"


def _write(tmp_path, ratings, reviews):
    """ratings: {market: [(user, item, rating, date)]}; reviews: {market: [(user, item, text, rating, date)]}."""
    zp = tmp_path / "raw.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for m, rows in ratings.items():
            z.writestr(f"xmarket_categories/ratings_{m}_{CAT}.txt.gz",
                       gzip.compress("".join(f"{u} {i} {r} {d}\n" for u, i, r, d in rows).encode()))
    for m, rows in reviews.items():
        p = tmp_path / "full" / m / CAT
        p.mkdir(parents=True)
        with gzip.open(p / f"reviews_{m}_{CAT}.json.gz", "wt", encoding="utf-8") as f:
            for u, i, t, r, d in rows:
                f.write(json.dumps([{"reviewerID": u, "asin": i, "reviewText": t, "summary": "", "overall": r,
                                     "cleanReviewTime": d}]) + "\n")
    return zp


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "FULL_DIR", tmp_path / "full")
    monkeypatch.setattr(D, "OUT_DIR", tmp_path / "data")
    monkeypatch.setattr(D, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(D, "COPIES_PATH", tmp_path / "data" / "copies.parquet")
    monkeypatch.setattr(D, "CATEGORIES", [CAT])
    monkeypatch.setattr(D, "KEEP_XMARKET_COPIES", False)
    monkeypatch.setattr(D, "_COPY_KEYS", None)
    return tmp_path


def test_a_copied_review_is_kept_once_in_the_earliest_market(env):
    ratings = {"uk": [("A", "X", 5.0, "2019-01-10"), ("A", "Y", 4.0, "2019-02-01"), ("B", "X", 3.0, "2019-01-01")],
               "ca": [("A", "X", 5.0, "2019-01-12"), ("A", "Z", 2.0, "2019-03-01")],
               "de": [("A", "Y", 1.0, "2019-02-01")]}
    reviews = {"uk": [("A", "X", "great book", 5.0, "2019-01-10"), ("A", "Y", "fine", 4.0, "2019-02-01"),
                      ("B", "X", "meh", 3.0, "2019-01-01")],
               "ca": [("A", "X", "great  book ", 5.0, "2019-01-12"), ("A", "Z", "bad", 2.0, "2019-03-01")],
               "de": [("A", "Y", "ganz anders", 1.0, "2019-02-01")]}
    zp = _write(env, ratings, reviews)
    with zipfile.ZipFile(zp) as zf:
        drops = D.xmarket_copies(zf)
        # A-X: same text (whitespace normalised) in uk and ca -> kept in uk (earlier), dropped in ca.
        # A-Y: different texts in uk and de -> two reviews, nothing dropped. B is in one market only.
        assert sorted(map(tuple, drops[["market", "user", "item"]].values)) == [("ca", "A", "X")]
        ca = D.read_context(zf, "ca", CAT)
        uk = D.read_context(zf, "uk", CAT)
    assert list(ca["item"]) == ["Z"] and len(uk) == 3
    st = json.loads((env / "results" / "xmarket_copies.json").read_text(encoding="utf-8"))
    assert st["pairs_dropped"] == 1


def test_copy_date_ties_go_to_the_reviewers_main_market(env):
    ratings = {"uk": [("A", "X", 5.0, "2019-01-10")],
               "ca": [("A", "X", 5.0, "2019-01-10"), ("A", "Q", 4.0, "2018-01-01"), ("A", "R", 4.0, "2018-02-01")]}
    reviews = {"uk": [("A", "X", "same", 5.0, "2019-01-10")],
               "ca": [("A", "X", "same", 5.0, "2019-01-10"), ("A", "Q", "q", 4.0, "2018-01-01"),
                      ("A", "R", "r", 4.0, "2018-02-01")]}
    zp = _write(env, ratings, reviews)
    with zipfile.ZipFile(zp) as zf:
        drops = D.xmarket_copies(zf)
    assert sorted(map(tuple, drops[["market", "user", "item"]].values)) == [("uk", "A", "X")]   # ca has more rows of A


def test_temporal_tie_break_is_seeded_and_leaves_the_random_split_unchanged():
    df = pd.DataFrame({"u": [0] * 10, "i": np.arange(10), "date": ["2020-01-24"] * 10})
    a = D.assign_splits(df, np.random.default_rng(0), np.random.default_rng([0, 1]))
    b = D.assign_splits(df, np.random.default_rng(0), np.random.default_rng([0, 1]))
    c = D.assign_splits(df, np.random.default_rng(0))                     # v1.0 rule: row position
    assert (a.split_temporal == b.split_temporal).all()
    assert (a.split_random == c.split_random).all()
    assert list(c.split_temporal) == ["train"] * 8 + ["val", "test"]
    assert (a.split_temporal.value_counts().to_dict() == {"train": 8, "val": 1, "test": 1})
    # chronological order still decides when dates differ
    d = df.assign(date=[f"2020-01-{k + 10}" for k in range(10)])
    e = D.assign_splits(d, np.random.default_rng(0), np.random.default_rng([0, 1]))
    assert e.split_temporal.iloc[-1] == "test" and e.split_temporal.iloc[-2] == "val"


def test_month_precision_rows_count_as_month_end(monkeypatch):
    monkeypatch.setattr(D, "MONTH_PRECISION", {"mx"})
    df = pd.DataFrame({"date": ["2019-08-24", "2019-02-24"]})
    mx = CS.effective_dates(df, "mx")
    uk = CS.effective_dates(df, "uk")
    assert list(mx.dt.strftime("%Y-%m-%d")) == ["2019-08-31", "2019-02-28"]
    assert list(uk.dt.strftime("%Y-%m-%d")) == ["2019-08-24", "2019-02-24"]
    # an mx review of August 2019 is not training data for a cutoff of 2019-08-28
    assert not (mx.iloc[0] < pd.Timestamp("2019-08-28"))


def test_cal_tag():
    assert D.CAL_TAG == "" and D.CAL_Q == (0.8, 0.9)
