"""Stage 1 - the borrowing harness.

One rule, applied to every model and every donor set, matching the `rho` semantics of
the published study so the results stay comparable with it:

    fit on the union of the target's and the donors' TRAINING interactions;
    score and evaluate only over the target catalog and the target's own test users,
    with that user's training and validation items masked.

A donor never contributes validation or test interactions, and never contributes
evaluation users. `k = 0` donors is target-only training and is the reference for every
reported effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

import metrics as M
import models as Mod
from util import stable_seed

USER_BATCH = 512
# Bytes of float64 ranking key held at once during evaluation.
RANK_KEY_BUDGET = 8_000_000
CUTOFFS = (5, 10, 20)
# Nested candidate-set metrics (ndcg@10|c<size>) are not reported in the paper and are off by default.
# The function path is kept (metrics.build_candidate_sets, tested in tests/test_metrics.py): pass
# candidate_sizes=LEGACY_CANDIDATE_SIZES to evaluate() to add them. The shipped core grid
# (results/full_random.jsonl) and paper/raw/mkt_runs.jsonl were written with (100, 1000); those extra
# fields are never read. Candidate sets use their own random generator, so switching them off leaves
# every other metric unchanged.
CANDIDATE_SIZES = ()
LEGACY_CANDIDATE_SIZES = (100, 1000)

# Evaluation cost is linear in (test users x catalog size), and one context - UK Books,
# with 104,881 test users over a 35,324-item catalog - accounts for 90% of the whole
# design. Capping the evaluation set at a deterministic sample keeps the grid affordable
# on this machine; with 20,000 users the standard error of a context mean is under 0.004,
# far below any effect the study reports, and every figure carries a bootstrap interval
# over exactly these users. The sample is drawn once per (target, split) and reused by
# every donor set, so all within-target comparisons remain paired on identical users.
MAX_EVAL_USERS = 20_000
EVAL_SAMPLE_SEED = 20260922


@dataclass
class ContextBlock:
    """Training, validation and test edges of one market-category context."""
    train_u: np.ndarray
    train_i: np.ndarray
    val_u: np.ndarray
    val_i: np.ndarray
    test_u: np.ndarray
    test_i: np.ndarray
    catalog: np.ndarray


def load_blocks(data: pd.DataFrame, split: str) -> dict[tuple[str, str], ContextBlock]:
    """Slice the interaction table into per-context edge blocks once, up front."""
    col = f"split_{split}"
    blocks = {}
    for key, g in data.groupby(["market", "category"], sort=True):
        part = {name: sub for name, sub in g.groupby(col, observed=True)}

        def arrs(name):
            sub = part.get(name)
            if sub is None:
                return np.array([], np.int32), np.array([], np.int32)
            return sub["u"].to_numpy(np.int32), sub["i"].to_numpy(np.int32)

        tr_u, tr_i = arrs("train")
        va_u, va_i = arrs("val")
        te_u, te_i = arrs("test")
        blocks[key] = ContextBlock(
            tr_u, tr_i, va_u, va_i, te_u, te_i,
            catalog=np.unique(g["i"].to_numpy(np.int32)),
        )
    return blocks


@dataclass
class TargetSession:
    """Everything about one target context that does not depend on the donor set."""
    key: tuple[str, str]
    block: ContextBlock
    catalog: np.ndarray = field(init=False)
    eval_users: np.ndarray = field(init=False)
    truth: list = field(init=False)
    masked: list = field(init=False)
    popularity: np.ndarray = field(init=False)

    def __post_init__(self):
        b = self.block
        self.catalog = b.catalog
        col_of = {int(i): j for j, i in enumerate(self.catalog)}

        by_user_test: dict[int, list[int]] = {}
        for u, i in zip(b.test_u, b.test_i):
            by_user_test.setdefault(int(u), []).append(col_of[int(i)])

        seen: dict[int, list[int]] = {}
        for u, i in zip(np.concatenate([b.train_u, b.val_u]),
                        np.concatenate([b.train_i, b.val_i])):
            j = col_of.get(int(i))
            if j is not None:
                seen.setdefault(int(u), []).append(j)

        users = np.array(sorted(by_user_test), dtype=np.int32)
        self.n_eval_users_available = len(users)
        if len(users) > MAX_EVAL_USERS:
            # Deterministic in the target key, so every donor set for this target is
            # scored on exactly the same users and the comparisons stay paired.
            rng = np.random.default_rng(stable_seed(EVAL_SAMPLE_SEED, self.key))
            users = np.sort(rng.choice(users, size=MAX_EVAL_USERS, replace=False))
        self.eval_users = users
        self.truth = [np.unique(by_user_test[int(u)]) for u in self.eval_users]
        self.masked = [np.unique(seen.get(int(u), [])) for u in self.eval_users]

        pop = np.zeros(len(self.catalog), dtype=np.float64)
        idx = np.array([col_of[int(i)] for i in b.train_i], dtype=np.int64) \
            if len(b.train_i) else np.array([], np.int64)
        np.add.at(pop, idx, 1.0)
        self.popularity = pop

    def validation_view(self) -> "TargetSession":
        """A session whose evaluation target is the validation split, for tuning."""
        b = self.block
        swapped = ContextBlock(b.train_u, b.train_i,
                               np.array([], np.int32), np.array([], np.int32),
                               b.val_u, b.val_i, b.catalog)
        return TargetSession(self.key, swapped)


def build_matrix(edge_lists, catalog: np.ndarray, eval_users: np.ndarray):
    """Pooled binary user x item matrix over local indices.

    Users shared between the target and a cross-category donor are merged into a single
    row, so the co-occurrence a shared user creates is preserved. Target catalog items
    with no training interaction still receive a column so they remain rankable.
    """
    us = [e[0] for e in edge_lists] + [eval_users]
    it = [e[1] for e in edge_lists] + [catalog]
    all_u = np.unique(np.concatenate(us)) if us else np.array([], np.int32)
    all_i = np.unique(np.concatenate(it)) if it else np.array([], np.int32)

    rows, cols = [], []
    for eu, ei in edge_lists:
        if len(eu) == 0:
            continue
        rows.append(np.searchsorted(all_u, eu))
        cols.append(np.searchsorted(all_i, ei))
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
    else:
        r = np.array([], np.int64)
        c = np.array([], np.int64)

    R = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                      shape=(len(all_u), len(all_i)))
    R.sum_duplicates()
    R.data[:] = 1.0  # binary implicit feedback
    target_cols = np.searchsorted(all_i, catalog)
    user_rows = np.searchsorted(all_u, eval_users)
    return R, target_cols, user_rows


def row_to_user_id(rows: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
    """Lookup from pooled row position to stable global user id.

    Pooled row positions move as donors add users, so anything keyed on them silently
    changes meaning between donor sets.
    """
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64)
    table = np.zeros(int(rows.max()) + 1, dtype=np.int64)
    table[rows] = user_ids
    return table


def evaluate(model: Mod.Recommender, session: TargetSession, user_rows: np.ndarray,
             candidate_sizes=CANDIDATE_SIZES, seed: int = 0) -> dict:
    """Full-catalog metrics; optionally also the same fitted model on nested candidate sets."""
    n_items = len(session.catalog)
    per_user = {f"{m}@{k}": [] for m in ("ndcg", "recall") for k in CUTOFFS}
    mrr_key = f"mrr@{min(max(CUTOFFS), n_items)}"
    per_user[mrr_key] = []
    nested = {f"ndcg@10|c{c}": [] for c in candidate_sizes}
    tops = []

    # Ranking keys are built in float64 for deterministic tie-breaking, so the batch is
    # sized by bytes rather than by row count: a 512-row block over the 35k-item catalog
    # would hold 145 MB at once, which this machine cannot spare.
    batch = int(max(32, min(USER_BATCH, RANK_KEY_BUDGET // max(n_items, 1))))
    rng = np.random.default_rng(seed)
    for start in range(0, len(user_rows), batch):
        sl = slice(start, start + batch)
        scores = model.score(user_rows[sl]).astype(np.float32)
        truth = session.truth[sl]
        for r, mask in enumerate(session.masked[sl]):
            if len(mask):
                scores[r, mask] = M.MASK_VALUE

        res = M.rank_metrics(scores, truth, cutoffs=CUTOFFS)
        for k in per_user:
            per_user[k].append(res[k])
        tops.append(res["_top"])

        for size, (sub, sub_truth) in M.build_candidate_sets(
                scores, truth, candidate_sizes, rng).items():
            sub_res = M.rank_metrics(sub, sub_truth, cutoffs=(10,))
            nested[f"ndcg@10|c{size}"].append(sub_res["ndcg@10"])

    out = {}
    per_user = {k: np.concatenate(v) if v else np.array([]) for k, v in per_user.items()}
    for k, v in per_user.items():
        out[k] = float(v.mean()) if v.size else float("nan")
    for k, v in nested.items():
        arr = np.concatenate(v) if v else np.array([])
        out[k] = float(arr.mean()) if arr.size else float("nan")

    top = np.concatenate(tops) if tops else np.zeros((0, 1), np.int64)
    out["coverage"] = M.catalog_coverage(top, n_items)
    out["novelty"] = M.mean_novelty(top, session.popularity) if top.size else float("nan")
    out["n_eval_users"] = int(len(user_rows))
    out["n_target_items"] = int(n_items)
    return out, per_user


def target_capacity(session: TargetSession, block: ContextBlock) -> tuple[int, bool]:
    """Capacity ceiling and numerical route, fixed by the TARGET alone.

    Both must be identical across donor sets, or the borrowed and unborrowed arms differ
    in model capacity and in numerical method as well as in data.
    """
    n_users = int(len(np.unique(block.train_u))) or 1
    n_items = int(len(session.catalog)) or 1
    return min(n_users, n_items), n_items <= Mod.DENSE_GRAM_MAX_ITEMS


def fit_and_select(model: str, session: TargetSession, edge_lists, seed: int = 0,
                   capacity: int | None = None, dense: bool | None = None,
                   base_params: dict | None = None):
    """Select hyperparameters on the target validation split, then refit for test.

    Tuning per (target, donor set, model) rules out that a conclusion only says
    'pooling hurts under settings tuned for target-only'. It is affordable only because
    these models are closed form.

    It also has a cost: the tuned contrast estimates "best-of-grid with donors minus
    best-of-grid without", so model capacity moves alongside the data. `base_params`
    therefore asks for a SECOND test-side fit at the hyperparameters the target-only run
    selected. Reporting both turns that design choice from an unfalsifiable assumption
    into a measured sensitivity: where the two agree the causal reading is clean, and
    where they diverge the divergence is itself the finding.
    """
    val_session = session.validation_view()
    grid = Mod.GRIDS[model]

    best_params, best_score = None, -np.inf
    # The test-item clamp in data.cut_points gives users with n <= 7 no validation items, and `adaptive_core`
    # relaxes to a 2- or 3-core in 39 of 74 contexts, so several contexts have NO
    # validation edges at all and silently fall through to grid[0]. That is not leakage
    # and it is identical in both arms, but the claim that hyperparameters are tuned per
    # donor set is false there, so it is recorded per unit rather than assumed.
    tuned = bool(len(grid) > 1 and len(val_session.eval_users))
    if tuned:
        R_val, cols_val, rows_val = build_matrix(
            edge_lists, val_session.catalog, val_session.eval_users)
        # One decomposition serves the whole rank grid (see PureSVD.factorize).
        shared = None
        if model in Mod.SHARED_FACTORS:
            top = max(p["rank"] for p in grid)
            shared = Mod.PureSVD.factorize(R_val, top, seed, capacity=capacity,
                                           dense=dense)
        for params in grid:
            try:
                candidate = Mod.build(model, R_val, cols_val,
                                      user_rows=rows_val, factors=shared,
                                      user_ids=row_to_user_id(rows_val,
                                                              val_session.eval_users),
                                      capacity=capacity, dense=dense, **params)
            except MemoryError as exc:
                # Only EASE has a documented catalog ceiling. Labelling any other
                # model's MemoryError as `ease_too_large` would report a genuine
                # failure as an expected, documented skip.
                if model != "EASE":
                    raise
                return None, None, "ease_too_large"
            res, _ = evaluate(candidate, val_session, rows_val,
                              candidate_sizes=(), seed=seed)
            if res["ndcg@10"] > best_score:
                best_score, best_params = res["ndcg@10"], params
            del candidate
        del R_val, cols_val, rows_val, shared
    else:
        best_params = grid[0]

    R, cols, rows = build_matrix(edge_lists, session.catalog, session.eval_users)
    ids = row_to_user_id(rows, session.eval_users)

    def fit_at(params):
        return Mod.build(model, R, cols, user_rows=rows, user_ids=ids,
                         capacity=capacity, dense=dense, **params)

    try:
        fitted = fit_at(best_params)
    except MemoryError:
        if model != "EASE":
            raise
        return None, None, "ease_too_large"
    res, per_user = evaluate(fitted, session, rows, seed=seed)
    res["params"] = best_params
    res["tuned"] = tuned
    res["n_val_users"] = int(len(val_session.eval_users))
    # MAX_EVAL_USERS changes the estimand for the largest contexts from "the target's
    # test users" to "a fixed sample of them". Deterministic and paired, but a reader of
    # the results file could not previously tell which contexts were capped.
    res["n_eval_users_available"] = int(session.n_eval_users_available)
    res["n_pooled_users"] = int(R.shape[0])
    res["n_pooled_items"] = int(R.shape[1])
    res["n_pooled_edges"] = int(R.nnz)

    # The frozen-hyperparameter arm: same donor data, capacity held at the target-only
    # choice, so the contrast isolates the data.
    if base_params is not None and base_params != best_params:
        try:
            frozen = fit_at(base_params)
            frozen_res, _ = evaluate(frozen, session, rows, candidate_sizes=(), seed=seed)
            res["frozen_ndcg@10"] = frozen_res["ndcg@10"]
            res["frozen_recall@10"] = frozen_res["recall@10"]
            del frozen
        except MemoryError:
            res["frozen_ndcg@10"] = float("nan")
            res["frozen_recall@10"] = float("nan")
    elif base_params is not None:
        res["frozen_ndcg@10"] = res["ndcg@10"]
        res["frozen_recall@10"] = res["recall@10"]
    res["frozen_params"] = base_params
    return res, per_user, None
