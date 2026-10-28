"""Stage 1c - closed-form recommenders.

Why closed form. Confounds such as the negative sampler, the update budget and shared
embedding tables are properties of *iterative* training. These four models have
no learning rate, no epoch count and no negative sampler, so fitting on target-only versus
target+donors changes exactly one thing: the data. That is what makes the borrowing effect
identifiable, and it is the scientific argument of Stage 1 rather than a convenience.

The four families also differ in which coupling can physically carry signal, which turns
the choice of model into a mechanism probe:

  MostPop   XM shifts item counts; XC cannot reach the target catalog at all.
  ItemKNN   `basic` builds item-item co-occurrence over the target catalog only, so only
            XM can move it; `extended` lets donor items act as neighbours feeding the user
            profile, which opens the XC channel through shared users.
  PureSVD   both couplings act, through a shared latent space.
  EASE      closed-form item-item ridge over the target catalog; like ItemKNN `basic`,
            reachable only through shared items.

Common contract
---------------
Every model is fitted on a pooled binary user x item matrix `R` (CSR, local indices) and
exposes `score(rows)` returning a dense (len(rows), n_target_items) array over the target
catalog, in the column order given by `target_cols`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Dense inversion of an n x n Gram costs ~8n^2 bytes plus LAPACK workspace. On an 8 GB
# machine with ~1.5 GB free, 4000 is the safe ceiling (128 MB Gram, ~400 MB peak);
# larger catalogs are recorded as skipped rather than silently dropped.
EASE_MAX_ITEMS = 4000

# Below this catalog size PureSVD factorises through a dense eigendecomposition of the
# item Gram rather than an iterative sparse solve. The Gram is n_items^2 float64, so 1500
# items costs 18 MB and roughly half a second, while ARPACK spends most of its time on
# per-call overhead at these sizes.
DENSE_GRAM_MAX_ITEMS = 1500


class Recommender:
    name = "base"

    def score(self, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomRec(Recommender):
    """Chance-level control.

    Not filler. A recommender that ignores the data entirely pins the floor of every
    table, so an unexpectedly high score elsewhere can be read as signal rather than
    leakage, and a near-zero score can be recognised as genuinely hard rather than
    broken.
    """

    name = "Random"

    def __init__(self, R: sp.csr_matrix, target_cols: np.ndarray, seed: int = 0,
                 user_ids=None):
        self.n_items = len(target_cols)
        self.seed = seed
        # Seeded on the GLOBAL user identity, not on the row position. Row positions shift
        # whenever donors add users to the pooled matrix, which made the control's scores
        # depend on the donor set - so a model that ignores the data entirely was showing
        # apparent transfer effects of up to 0.04 NDCG@10. Keyed on user id it is
        # genuinely invariant, which is the whole point of a control.
        self.user_ids = None if user_ids is None else np.asarray(user_ids)

    def score(self, rows: np.ndarray) -> np.ndarray:
        # `user_ids` maps a pooled row position to a stable global user id.
        ids = rows if self.user_ids is None else self.user_ids[rows]
        out = np.empty((len(rows), self.n_items), dtype=np.float32)
        for j, uid in enumerate(ids):
            out[j] = np.random.default_rng(
                (self.seed, int(uid))).random(self.n_items, dtype=np.float32)
        return out


class MostPop(Recommender):
    """Popularity over the pooled training data, ranked within the target catalog."""

    name = "MostPop"

    def __init__(self, R: sp.csr_matrix, target_cols: np.ndarray):
        pop = np.asarray(R.sum(axis=0)).ravel()
        self._scores = pop[target_cols].astype(np.float32)

    def score(self, rows: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(rows), self._scores.size)).copy()


def _keep_top_k_per_column(sim: sp.csc_matrix, k: int) -> sp.csr_matrix:
    """Retain each target item's k strongest neighbours.

    Vectorised: a per-column Python loop costs minutes on the 35k-item catalogs, and this
    runs once per block, per hyperparameter setting, per dose step.
    """
    sim = sim.tocsc()
    n_cols = sim.shape[1]
    counts = np.diff(sim.indptr)
    if counts.size == 0 or counts.max() <= k:
        return sim.tocsr()

    col_of_nnz = np.repeat(np.arange(n_cols), counts)
    # Primary key column, secondary key descending similarity.
    order = np.lexsort((-sim.data, col_of_nnz))
    rank_in_col = np.arange(sim.nnz) - sim.indptr[col_of_nnz[order]]
    keep = order[rank_in_col < k]

    mask = np.zeros(sim.nnz, dtype=bool)
    mask[keep] = True
    sim.data[~mask] = 0.0
    sim.eliminate_zeros()
    return sim.tocsr()


# Peak memory of the similarity build is set by the number of co-occurrence entries held
# at once, not by the catalog. UK Books generates 14.7M of them, which at float64 plus the
# COO index arrays and a few temporaries costs ~0.7 GB per fit - and the ItemKNN grid runs
# twelve fits in sequence, before any donor is added. Blocking over target columns bounds
# that regardless of catalog size, and top-k pruning is applied per block so only the
# retained neighbours survive into the assembled matrix.
SIM_BLOCK_BUDGET = 8_000_000


def _cosine_similarity_topk(R: sp.csr_matrix, profile_cols: np.ndarray,
                            target_cols: np.ndarray, shrink: float,
                            topk: int) -> sp.csr_matrix:
    """Shrunk cosine between profile items and target items, pruned to top-k per column."""
    counts = np.asarray(R.sum(axis=0)).ravel()
    norms = np.sqrt(counts).astype(np.float32)
    profile_norms = norms[profile_cols]

    left = R[:, profile_cols].T.tocsr()
    n_profile = len(profile_cols)
    block = int(max(64, min(4096, SIM_BLOCK_BUDGET // max(n_profile, 1))))

    pieces = []
    for start in range(0, len(target_cols), block):
        cols = target_cols[start:start + block]
        co = (left @ R[:, cols]).tocoo()
        if co.nnz:
            denom = profile_norms[co.row] * norms[cols][co.col] + np.float32(shrink)
            vals = np.where(denom > 0, co.data / np.maximum(denom, 1e-12), 0.0)
            vals = vals.astype(np.float32, copy=False)
            # An item is never its own neighbour.
            vals[profile_cols[co.row] == cols[co.col]] = 0.0
        else:
            vals = co.data.astype(np.float32, copy=False)
        piece = sp.coo_matrix((vals, (co.row, co.col)),
                              shape=(n_profile, len(cols))).tocsc()
        piece.eliminate_zeros()
        pieces.append(_keep_top_k_per_column(piece, topk))
        del co, piece
    if not pieces:
        return sp.csr_matrix((n_profile, 0), dtype=np.float32)
    return sp.hstack(pieces, format="csr")


class ItemKNN(Recommender):
    """Item-based cosine k-NN.

    `extended=False` restricts both neighbours and profile to the target catalog, so a
    donor can only act through items it shares with the target. `extended=True` lets the
    whole pooled catalog act as neighbours, which is the only way the cross-category
    channel can carry anything, and is included so XC is tested rather than defined away.
    """

    name = "ItemKNN"

    def __init__(self, R: sp.csr_matrix, target_cols: np.ndarray, topk: int = 200,
                 shrink: float = 10.0, extended: bool = False, user_rows=None):
        self.R = R
        if extended:
            # Only items the target's own users actually hold can contribute to their
            # profile. Admitting the entire pooled catalog would let donor-only items
            # occupy top-k neighbour slots while multiplying by a zero profile entry,
            # which would weaken the extended variant for no reason and make its
            # comparison against the basic variant an artefact rather than a result.
            if user_rows is not None and len(user_rows):
                touched = np.unique(R[user_rows].indices)
            else:
                touched = np.arange(R.shape[1])
            self.profile_cols = np.union1d(touched, target_cols).astype(np.int64)
        else:
            self.profile_cols = np.asarray(target_cols)
        self.sim = _cosine_similarity_topk(R, self.profile_cols, target_cols,
                                           shrink, topk)

    def score(self, rows: np.ndarray) -> np.ndarray:
        profile = self.R[rows][:, self.profile_cols]
        # Both operands are float32, so the product never materialises as float64.
        return np.asarray((profile @ self.sim).toarray(), dtype=np.float32)


class PureSVD(Recommender):
    """Truncated SVD of the pooled matrix; both couplings act through the latent space.

    The rank grid is evaluated from a SINGLE decomposition. Right singular vectors come
    back ordered by singular value, so the rank-r model is exactly the leading r columns
    of a rank-R decomposition for any r <= R. Running one ARPACK solve per grid point
    instead was costing 98% of the entire experiment's runtime - 102x per unit more than
    every other model combined - for factors that were already contained in the largest
    solve. Pass `factors` to reuse a decomposition across the grid.
    """

    name = "PureSVD"

    @staticmethod
    def max_rank(R: sp.csr_matrix, requested: int, capacity: int | None = None) -> int:
        """Largest usable rank, capped by `capacity` rather than by the pooled matrix.

        Capping by the POOLED dimensions was a confound, not a detail. A target-only fit
        of de Grocery could reach rank 31, so the grid {16,32,64,128} collapsed to
        [16,31,31,31]; adding sixteen XX donors raised the cap to 128 and opened
        [16,32,64,128]. The borrowed and unborrowed arms were then comparing different
        model capacities, which is exactly the confound the closed-form design exists to
        remove. The
        capacity ceiling must be a property of the TARGET, identical across donor sets.
        """
        ceiling = min(R.shape) if capacity is None else capacity
        return max(1, int(min(requested, max(1, ceiling // 2))))

    @staticmethod
    def factorize(R: sp.csr_matrix, rank: int, seed: int = 0,
                  capacity: int | None = None, dense: bool | None = None) -> np.ndarray:
        """Right singular vectors, columns ordered by descending singular value.

        Two exact routes. The right singular vectors of R are the eigenvectors of the
        item Gram R^T R, so on a small catalog a dense symmetric eigendecomposition is
        both exact and far cheaper than an iterative solve - ARPACK's per-call overhead
        dominates there, and most contexts in this grid have catalogs of a few hundred
        items. Large catalogs keep the sparse iterative path.
        """
        rank = PureSVD.max_rank(R, rank, capacity)
        # The numerical route must be pinned by the caller, not chosen from the pooled
        # catalog size. Otherwise a target-only fit takes the exact dense path while every
        # donor fit takes the iterative one, so the two arms differ in method as well as
        # in data.
        use_dense = (R.shape[1] <= DENSE_GRAM_MAX_ITEMS) if dense is None else dense

        if use_dense and R.shape[1] <= DENSE_GRAM_MAX_ITEMS:
            gram = np.asarray((R.T @ R).todense(), dtype=np.float64)
            # eigh returns ascending eigenvalues; reverse for descending singular values.
            _, vectors = np.linalg.eigh(gram)
            vectors = vectors[:, ::-1][:, :rank]
            return np.ascontiguousarray(vectors, dtype=np.float32)

        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=min(R.shape))
        # `return_singular_vectors="vh"` avoids materialising the large user factors.
        _, sigma, vh = spl.svds(R.astype(np.float32, copy=False), k=rank, v0=v0,
                                return_singular_vectors="vh")
        order = np.argsort(-sigma)
        return np.ascontiguousarray(vh[order].T, dtype=np.float32)

    def __init__(self, R: sp.csr_matrix, target_cols: np.ndarray, rank: int = 64,
                 seed: int = 0, factors: np.ndarray | None = None,
                 capacity: int | None = None, dense: bool | None = None):
        self.R = R
        # The rank must stay well below the catalog dimension. Allowed to approach it,
        # the truncated SVD reconstructs the interaction matrix almost exactly: on a
        # 100-item catalog a rank-99 fit gave every user only 2-3 distinct scores, so the
        # ranking was decided by tie-breaking rather than by the model, and validation
        # selection happily picked that degenerate setting. Capping at half of the
        # smaller dimension keeps it a genuine low-rank model.
        self.rank = PureSVD.max_rank(R, rank, capacity)
        if factors is None:
            factors = PureSVD.factorize(R, self.rank, seed, capacity=capacity,
                                        dense=dense)
        self.rank = min(self.rank, factors.shape[1])
        self.V = np.ascontiguousarray(factors[:, :self.rank], dtype=np.float32)
        self.Vt = np.ascontiguousarray(self.V[target_cols].T, dtype=np.float32)

    def score(self, rows: np.ndarray) -> np.ndarray:
        latent = self.R[rows] @ self.V
        return np.asarray(latent @ self.Vt, dtype=np.float32)


class EASE(Recommender):
    """Closed-form item-item ridge (Steck, WWW'19), solved over the target catalog."""

    name = "EASE"

    def __init__(self, R: sp.csr_matrix, target_cols: np.ndarray, lam: float = 250.0):
        if len(target_cols) > EASE_MAX_ITEMS:
            raise MemoryError(
                f"EASE skipped: {len(target_cols)} target items exceeds the "
                f"{EASE_MAX_ITEMS}-item ceiling for dense inversion on this machine")
        self.R = R
        self.target_cols = target_cols
        block = R[:, target_cols]
        gram = np.asarray((block.T @ block).todense(), dtype=np.float64)
        diag = np.diag_indices(gram.shape[0])
        gram[diag] += lam
        precision = np.linalg.inv(gram)
        weights = precision / (-np.diag(precision))
        weights[diag] = 0.0
        self.B = weights.astype(np.float32)

    def score(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.R[rows][:, self.target_cols] @ self.B, dtype=np.float32)


# Hyperparameter grids. Selection happens on the target validation split, per
# (target, donor set, model), so a conclusion cannot simply reflect "pooling is harmful
# under target-tuned settings".
GRIDS = {
    "Random": [{}],
    "MostPop": [{}],
    "ItemKNN": [{"topk": k, "shrink": s, "extended": False}
                for k in (50, 200, 1000) for s in (0.0, 10.0)],
    "ItemKNN-ext": [{"topk": k, "shrink": s, "extended": True}
                    for k in (50, 200, 1000) for s in (0.0, 10.0)],
    "PureSVD": [{"rank": r} for r in (16, 32, 64, 128)],
    "EASE": [{"lam": l} for l in (10.0, 50.0, 250.0, 1000.0)],
}

BUILDERS = {
    "Random": RandomRec,
    "MostPop": MostPop,
    "ItemKNN": ItemKNN,
    "ItemKNN-ext": ItemKNN,
    "PureSVD": PureSVD,
    "EASE": EASE,
}


# Models that need to know which rows are the target's evaluation users.
NEEDS_USER_ROWS = {"ItemKNN-ext"}
# Models that must be keyed on stable user identity rather than pooled row position.
NEEDS_USER_IDS = {"Random"}
# Models whose whole hyperparameter grid is served by one shared decomposition.
SHARED_FACTORS = {"PureSVD"}
# Models whose capacity ceiling and numerical route must be fixed by the TARGET, so they
# do not drift between the borrowed and unborrowed arms.
NEEDS_CAPACITY = {"PureSVD"}


def build(model: str, R: sp.csr_matrix, target_cols: np.ndarray,
          user_rows=None, factors=None, user_ids=None, capacity=None, dense=None,
          **kwargs) -> Recommender:
    if model in NEEDS_USER_ROWS:
        kwargs["user_rows"] = user_rows
    if model in NEEDS_USER_IDS:
        kwargs["user_ids"] = user_ids
    if model in NEEDS_CAPACITY:
        kwargs["capacity"] = capacity
        kwargs["dense"] = dense
    if model in SHARED_FACTORS and factors is not None:
        kwargs["factors"] = factors
    return BUILDERS[model](R, target_cols, **kwargs)
