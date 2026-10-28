"""Two-way cluster bootstrap over countries and categories (the interval procedure of the paper, Section 4.6).

Each replicate draws the 7 countries and the categories with replacement, independently.
Let w_m be how often country m was drawn and w_c how often category c was drawn.

* Context-level observation (target market m, category c):      weight w_m * w_c
* Pair-level observation (target m, donor d, category c):        weight w_m * w_d * w_c
  (one country draw serves both roles, so A->B and B->A share their country draws)
* Results with several pooled donors are context-level observations.

Repeated splits, subsamples and launch sets must be averaged within each observation before
calling these functions. Replicates with zero total weight are redrawn. Every call uses its
own seed derived from `label`, so intervals do not depend on the order of calls.
"""
from __future__ import annotations

import hashlib

import numpy as np

B_DEFAULT = 2000


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


def stat_value(vals, stat="median"):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return float(np.median(vals)) if stat == "median" else float(np.mean(vals))


def cboot(vals, markets, categories, donors=None, stat="median", label="", B=B_DEFAULT, alpha=0.05):
    """Percentile CI of a weighted median/mean under the two-way cluster bootstrap.

    vals, markets, categories: equal-length sequences (target market, category per observation).
    donors: donor market per observation for pair-level data (None for context-level data).
    Returns (lo, hi).
    """
    vals = np.asarray(vals, float)
    mk = np.asarray(markets, object)
    ct = np.asarray(categories, object)
    dn = None if donors is None else np.asarray(donors, object)
    ok = ~np.isnan(vals)
    vals, mk, ct = vals[ok], mk[ok], ct[ok]
    if dn is not None:
        dn = dn[ok]
    countries = sorted(set(mk) | (set(dn) if dn is not None else set()))
    cats = sorted(set(ct))
    mi = np.array([countries.index(x) for x in mk])
    ci = np.array([cats.index(x) for x in ct])
    di = None if dn is None else np.array([countries.index(x) for x in dn])
    rng = np.random.default_rng(_seed(f"cboot|{label}|{stat}"))
    draws = replicate_weights(rng, len(countries), len(cats), mi, ci, di)
    out = []
    while len(out) < B:
        w = next(draws)[0]
        out.append(weighted_median(vals, w) if stat == "median" else float(np.average(vals, weights=w)))
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def replicate_weights(rng, n_countries, n_cats, mi, ci, di=None):
    """Generator of bootstrap replicates: yields (w, wm, wc) with observation weights w and the draw counts per
    country (wm) and per category (wc). mi, ci, di: country index of the target, category index and (pair-level
    data) country index of the donor per observation. One country draw wm serves both the target and the donor
    role. Replicates whose weights sum to zero are redrawn (not yielded)."""
    while True:
        wm = np.bincount(rng.integers(0, n_countries, n_countries), minlength=n_countries)
        wc = np.bincount(rng.integers(0, n_cats, n_cats), minlength=n_cats)
        w = wm[mi] * wc[ci]
        if di is not None:
            w = w * wm[di]
        if w.sum() == 0:
            continue
        yield w, wm, wc


def summarize(vals, markets, categories, donors=None, stat="median", label=""):
    """Point estimate (ordinary median or mean over all observations) plus the cluster-bootstrap CI."""
    lo, hi = cboot(vals, markets, categories, donors, stat, label)
    return stat_value(vals, stat), lo, hi
