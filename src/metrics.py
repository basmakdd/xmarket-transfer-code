"""Ranking metrics under the full-catalog protocol, with nested candidate sets.

Everything returns *per-user* arrays rather than context means, so uncertainty can be
obtained by bootstrapping over test users rather than from three-seed per-context
p-values, which are indicative only.

Conventions (identical to the original pipeline's `cat_evaluate`):
* ranking is over the target context catalog only;
* a user positives seen during training (and validation, when scoring the test split)
  are masked out before ranking;
* NDCG uses the ideal DCG over `min(|T_u|, K)` held-out items.
"""
from __future__ import annotations

import numpy as np

MASK_VALUE = -np.inf


def dcg_weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def ideal_dcg(n_relevant: np.ndarray, k: int) -> np.ndarray:
    """IDCG for each user given how many held-out items they have."""
    cum = np.concatenate([[0.0], np.cumsum(dcg_weights(k))])
    return cum[np.minimum(n_relevant, k)]


def rank_metrics(scores: np.ndarray, truth: list[np.ndarray], cutoffs=(5, 10, 20)) -> dict:
    """Per-user NDCG@K, Recall@K and MRR from a (n_users, n_items) score matrix.

    `truth[r]` holds the column indices of user r held-out items. Masked entries must
    already be set to `MASK_VALUE` by the caller.
    """
    n_users, n_items = scores.shape
    k_max = min(max(cutoffs), n_items)

    # Ties must be broken deterministically, or the metric depends on partition order
    # rather than on the data. This is not hypothetical: MostPop scores are interaction
    # counts, so a 168-item catalog can have 33 items sharing one score and only 26
    # distinct scores in total - which items reach the top ten would otherwise be
    # arbitrary, and would differ between two runs of the same model.
    #
    # Ties are resolved in favour of the lower catalog index. The offset is built in
    # float64 from the row's own scale, so it separates exact ties while staying far
    # below the gap between any two genuinely different scores.
    keys = scores.astype(np.float64, copy=True)
    finite = keys[np.isfinite(keys)]
    scale = np.abs(finite).max() if finite.size else 1.0
    step = np.spacing(max(scale, 1.0)) * 4.0
    keys += (n_items - np.arange(n_items, dtype=np.float64)) * step

    # argpartition + a partial sort is materially cheaper than a full argsort here.
    part = np.argpartition(-keys, k_max - 1, axis=1)[:, :k_max]
    part_keys = np.take_along_axis(keys, part, axis=1)
    order = np.argsort(-part_keys, axis=1, kind="stable")
    top = np.take_along_axis(part, order, axis=1)
    del keys, part_keys

    hits = np.zeros((n_users, k_max), dtype=np.float64)
    n_rel = np.zeros(n_users, dtype=np.int64)
    for r in range(n_users):
        gt = truth[r]
        n_rel[r] = len(gt)
        if len(gt):
            hits[r] = np.isin(top[r], gt)

    weights = dcg_weights(k_max)
    out = {}
    for k in cutoffs:
        kk = min(k, k_max)
        dcg = (hits[:, :kk] * weights[:kk]).sum(axis=1)
        idcg = ideal_dcg(n_rel, kk)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"ndcg@{k}"] = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)
            out[f"recall@{k}"] = np.where(
                n_rel > 0, hits[:, :kk].sum(axis=1) / np.maximum(n_rel, 1), 0.0)

    first = np.argmax(hits > 0, axis=1)
    any_hit = hits.any(axis=1)
    # Truncated at k_max, so name it for what it is rather than bare "mrr".
    out[f"mrr@{k_max}"] = np.where(any_hit, 1.0 / (first + 1.0), 0.0)
    out["_top"] = top
    return out


def build_candidate_sets(scores: np.ndarray, truth: list[np.ndarray],
                         sizes, rng: np.random.Generator) -> dict:
    """Genuinely nested candidate sets, with candidate order randomised.

    Two defects this replaces, both of which corrupted the reduced-catalog numbers:

    1. **Positives were placed at the lowest candidate columns.** `rank_metrics` breaks
       ties toward the lower index, so every tie went to a held-out item. Ties are the
       normal case here - MostPop scores are integer counts, and an ItemKNN item with no
       neighbour in the profile scores exactly 0.0 (83.8% of finite scores on one real
       context). On a synthetic case where every positive tied at 0.0 this inflated
       NDCG@10 from 0.014 to 0.408, a factor of 29, purely from column position. It did
       not cancel in the paired difference either, because tie density changes with the
       donor set - it reversed the sign of a measured effect on real data.

    2. **The sets were not nested.** Each size drew its own pool, so the 100-item set was
       not a subset of the 1,000-item set and the across-size comparison was not the
       controlled contrast it was documented to be.

    The negative pool is now drawn once at the largest size and truncated for smaller
    ones, so the sets nest by construction; candidate columns are then shuffled per user
    so column position carries no information about relevance.
    """
    n_users, n_items = scores.shape
    sizes = sorted({int(s) for s in sizes}, reverse=True)
    if not sizes:
        return {}
    eligible = np.isfinite(scores)

    pools, gts = [], []
    for r in range(n_users):
        gt = np.asarray(truth[r])
        pool = np.flatnonzero(eligible[r])
        if gt.size:
            pool = pool[~np.isin(pool, gt)]
        rng.shuffle(pool)
        pools.append(pool)
        gts.append(gt)

    out = {}
    for size in sizes:
        if size >= n_items:
            out[size] = (scores, truth)
            continue
        width = max(size, max((g.size for g in gts), default=0))
        sub = np.full((n_users, width), MASK_VALUE, dtype=scores.dtype)
        sub_truth = []
        for r in range(n_users):
            gt = gts[r]
            n_fill = max(0, size - gt.size)
            # A prefix of the same shuffled pool, so smaller sets nest inside larger ones.
            cand = np.concatenate([gt, pools[r][:n_fill]])
            order = rng.permutation(cand.size)
            cand = cand[order]
            sub[r, :cand.size] = scores[r, cand]
            sub_truth.append(np.flatnonzero(np.isin(cand, gt)) if gt.size
                             else np.zeros(0, dtype=np.int64))
        out[size] = (sub, sub_truth)
    return out


def aggregate(per_user: dict, keys=None) -> dict:
    keys = keys or [k for k in per_user if not k.startswith("_")]
    return {k: float(np.mean(per_user[k])) for k in keys}


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    """Percentile bootstrap over users for a per-user metric vector."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def catalog_coverage(top: np.ndarray, n_items: int) -> float:
    """Share of the target catalog that appears in any user top-list."""
    return float(len(np.unique(top)) / max(n_items, 1))


def mean_novelty(top: np.ndarray, popularity: np.ndarray) -> float:
    """Mean self-information of recommended items; higher means less popularity-biased."""
    total = popularity.sum()
    if total <= 0:
        return float("nan")
    p = np.clip(popularity / total, 1e-12, None)
    return float(np.mean(-np.log2(p[top])))
