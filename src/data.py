"""Stage 1a - preprocessing.

Streams the raw XMarket per-category ratings out of the source zip, reproduces the
adaptive k-core filtering of the original pipeline, assigns the global ID space, and
builds two split variants (random and temporal).

Design notes
------------
* Item identities are ASINs and are shared across markets within a category; user
  identities are prefixed by market and are therefore disjoint across markets. This is
  the structural coupling the borrowing study exploits, and Stage 1b tests it
  empirically rather than assuming it.
* The split is per user *within a context*, as described in the paper. (The original
  pipeline split per user over that user pooled across every context; the deviation is
  deliberate.)
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import os
REPO = Path(__file__).resolve().parents[1]
# Raw XMarket category files (ratings_<market>_<category>.txt.gz) packed in ONE zip whose members sit under
# "xmarket_categories/". Default location raw_data/xmarket_categories.zip; override with XMARKET_ZIP.
ZIP_PATH = Path(os.environ.get("XMARKET_ZIP", REPO / "raw_data" / "xmarket_categories.zip"))
OUT_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"

MARKETS = ["de", "jp", "in", "fr", "ca", "mx", "uk"]
CATEGORIES = [
    "Electronics", "Home_and_Kitchen", "Cell_Phones_and_Accessories",
    "Movies_and_TV", "Sports_and_Outdoors", "Toys_and_Games",
    "Office_Products", "CDs_and_Vinyl", "Musical_Instruments", "Books",
    "Arts_Crafts_and_Sewing", "Automotive", "Digital_Music",
    "Grocery_and_Gourmet_Food", "Industrial_and_Scientific", "Kindle_Store",
]

# Filtering thresholds, identical to those of the original pipeline.
MIN_CORE = 5
MIN_PAIR_INTER = 800
MIN_PAIR_ITEMS = 40

# Variants for robustness runs (defaults reproduce the paper):
#   XM_SPLIT_SEED  seed of the per-user random/temporal split assignment (default 0)
#   XM_MIN_RATING  keep only ratings >= this value before deduplication (default: keep all)
#   XM_DATA_TAG    suffix for the output/input files, e.g. "_seed1" -> data/interactions_seed1.parquet
SPLIT_SEED = int(os.environ.get("XM_SPLIT_SEED", "0"))
MIN_RATING = float(os.environ["XM_MIN_RATING"]) if os.environ.get("XM_MIN_RATING") else None
DATA_TAG = os.environ.get("XM_DATA_TAG", "")
#   XM_MERGE_VARIANTS=1      merge product variants (see variant_map) before the core filter
#   XM_DROP_XMARKET_USERS=1  drop every interaction of a raw reviewer ID found in > 1 market (any category)
MERGE_VARIANTS = os.environ.get("XM_MERGE_VARIANTS") == "1"
#   XM_MERGE_VARIANTS=context  same rule, but users/products linked per market x category context (not pooled over
#                              markets); the representative ASIN is the shared cross-market identifier
MERGE_VARIANTS_CONTEXT = os.environ.get("XM_MERGE_VARIANTS") == "context"
DROP_XMARKET_USERS = os.environ.get("XM_DROP_XMARKET_USERS") == "1"
VAR_MIN_SHARED, VAR_JACCARD_NUM, VAR_JACCARD_DEN = 3, 4, 5   # >= 3 shared users and Jaccard >= 4/5
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1


# --------------------------------------------------------------------------- loading
def read_context(zf: zipfile.ZipFile, market: str, category: str):
    """Read one raw (market, category) ratings file; None when absent or empty."""
    name = f"xmarket_categories/ratings_{market}_{category}.txt.gz"
    try:
        blob = zf.read(name)
    except KeyError:
        return None
    text = gzip.decompress(blob)
    if not text.strip():
        return None
    df = pd.read_csv(
        io.BytesIO(text), sep=" ", header=None,
        names=["user", "item", "rating", "date"],
        dtype={"user": "string", "item": "string", "rating": "float32", "date": "string"},
    )
    if df.empty:
        return None
    if MIN_RATING is not None:
        df = df[df["rating"] >= MIN_RATING]
        if df.empty:
            return None
    # Collapse repeated user-item pairs, keeping the earliest observation.
    df = df.sort_values("date", kind="stable").drop_duplicates(["user", "item"], keep="first")
    return df[["user", "item", "date"]]


# --------------------------------------------------------------------------- sensitivity variants (off by default)
def multi_market_reviewers(zf: zipfile.ZipFile) -> set:
    """Raw reviewer IDs (as in the files, all rows) occurring in more than one market, any category."""
    per_market = []
    for market in MARKETS:
        ids = []
        for category in CATEGORIES:
            try:
                text = gzip.decompress(zf.read(f"xmarket_categories/ratings_{market}_{category}.txt.gz"))
            except KeyError:
                continue
            if not text.strip():
                continue
            u = pd.read_csv(io.BytesIO(text), sep=" ", header=None, usecols=[0], names=["user"],
                            dtype={"user": "string"})["user"]
            ids.append(pd.Series(u.unique()))
            del text, u
        if ids:
            per_market.append(pd.Series(pd.concat(ids).unique()))
    vc = pd.concat(per_market).value_counts()
    return set(vc.index[vc > 1].tolist())


def variant_map(frames: dict):
    """Link products of one category sharing >= VAR_MIN_SHARED distinct market-prefixed users with Jaccard
    of their user sets >= 0.8; merge connected components; representative = smallest ASIN.
    frames: {market: DataFrame(user, item)}. Exact all-pairs search with prefix filtering (users ordered
    rarest first; a product with n users indexes its first n - ceil(0.8 n) + 1 users), then verification.
    Returns {asin: representative} for products in components of size >= 2, and a stats dict."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    uk = pd.concat([("" + m + "::") + f["user"].astype(str) for m, f in frames.items()], ignore_index=True)
    it = pd.concat([f["item"].astype(str) for f in frames.values()], ignore_index=True)
    uc, _ = pd.factorize(uk); ic, inames = pd.factorize(it)
    del uk, it
    key = np.unique(uc.astype(np.int64) * (len(inames) + 1) + ic)
    uc, ic = (key // (len(inames) + 1)).astype(np.int64), (key % (len(inames) + 1)).astype(np.int64)
    deg = np.bincount(ic, minlength=len(inames))
    ok = deg[ic] >= VAR_MIN_SHARED
    uc, ic = uc[ok], ic[ok]
    stats = {"products": int(len(inames)), "products_deg3": int((deg >= VAR_MIN_SHARED).sum())}
    if len(ic) == 0:
        return {}, dict(stats, candidates=0, links=0)
    ufreq = np.bincount(uc)
    o = np.lexsort((uc, ufreq[uc], ic))                     # per product: users rarest first
    uc, ic = uc[o], ic[o]
    n = deg[ic]
    start = np.r_[0, np.flatnonzero(np.diff(ic)) + 1]
    pos = np.arange(len(ic)) - np.repeat(start, np.diff(np.r_[start, len(ic)]))
    plen = n - (VAR_JACCARD_NUM * n + VAR_JACCARD_DEN - 1) // VAR_JACCARD_DEN + 1
    pre = pd.DataFrame({"u": uc[pos < plen], "i": ic[pos < plen]})
    gs = pre.groupby("u").i.transform("size")
    pre = pre[gs > 1]
    stats["candidate_pairs_upper_bound"] = int((gs[gs > 1] ** 2).sum())
    cand = pre.merge(pre, on="u")
    cand = cand[cand.i_x < cand.i_y][["i_x", "i_y"]].drop_duplicates().to_numpy()
    del pre
    a, b = cand[:, 0], cand[:, 1]
    na, nb = deg[a], deg[b]
    keep = VAR_JACCARD_DEN * np.minimum(na, nb) >= VAR_JACCARD_NUM * np.maximum(na, nb)
    a, b = a[keep], b[keep]
    stats["candidates"] = int(len(a))
    X = sp.csr_matrix((np.ones(len(ic), np.float32), (ic, uc)), shape=(len(inames), int(uc.max()) + 1))
    inter = np.zeros(len(a), np.int64)
    for s in range(0, len(a), 200000):
        inter[s:s + 200000] = np.asarray(X[a[s:s + 200000]].multiply(X[b[s:s + 200000]]).sum(axis=1)).ravel().round()
    uni = deg[a] + deg[b] - inter
    link = (inter >= VAR_MIN_SHARED) & (VAR_JACCARD_DEN * inter >= VAR_JACCARD_NUM * uni)
    a, b = a[link], b[link]
    stats["links"] = int(len(a))
    if len(a) == 0:
        return {}, stats
    G = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(len(inames), len(inames)))
    _, lab = connected_components(G, directed=False)
    members = np.unique(np.r_[a, b])
    comp = pd.DataFrame({"asin": np.asarray(inames)[members], "lab": lab[members]})
    comp["rep"] = comp.groupby("lab").asin.transform("min")
    return dict(zip(comp.asin, comp.rep)), stats


def collapse_variants(df: pd.DataFrame, vmap: dict) -> pd.DataFrame:
    """Replace merged ASINs by their representative; keep the earliest (user, product) interaction."""
    s = df["item"].astype(str)
    df = df.assign(item=s.map(vmap).fillna(s).astype("string"))
    df = df.sort_values("date", kind="stable").drop_duplicates(["user", "item"], keep="first")
    return df


# --------------------------------------------------------------------------- filtering
def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iterative k-core on users and items (mirrors `five_core` of the original pipeline)."""
    while True:
        n0 = len(df)
        vc = df["user"].value_counts()
        df = df[df["user"].isin(vc.index[vc >= k])]
        vc = df["item"].value_counts()
        df = df[df["item"].isin(vc.index[vc >= k])]
        if len(df) == n0:
            return df


def adaptive_core(df: pd.DataFrame):
    """Relax the core level until the context survives the size thresholds."""
    for kk in range(MIN_CORE, 1, -1):
        d = k_core(df, kk)
        if len(d) >= MIN_PAIR_INTER and d["item"].nunique() >= MIN_PAIR_ITEMS:
            return d, kk
    return df.iloc[0:0], None


# --------------------------------------------------------------------------- splitting
def cut_points(n: int):
    """Split sizes per user, guaranteeing a held-out test item whenever n > 1.

    The original pipeline's `split_user` caps the training block at `n - 1` but lets
    the validation block consume whatever is left, so users with exactly 6 or 7
    interactions receive *zero* test items and drop out of evaluation silently. That hits
    27.9% of (context, user) pairs - 29,115 users in UK Books alone - and it is not
    random: it removes one specific activity band. The extra clamp below
    reserves the test item instead.
    """
    n_train = max(1, int(round(TRAIN_FRAC * n)))
    n_val = max(0, int(round(VAL_FRAC * n)))
    if n > 1:
        n_train = min(n_train, n - 1)
        n_val = min(n_val, n - n_train - 1)
    return n_train, n_val


def assign_splits(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add `split_random` and `split_temporal`, both per user within this context."""
    df = df.reset_index(drop=True)
    rnd = np.full(len(df), "train", dtype=object)
    tmp = np.full(len(df), "train", dtype=object)
    # Row position breaks date ties deterministically in the temporal split.
    order_key = df["date"].to_numpy().astype(str)
    for pos in df.groupby("u", sort=True).indices.values():
        pos = np.asarray(pos)
        n = len(pos)
        n_train, n_val = cut_points(n)

        shuffled = pos[rng.permutation(n)]
        rnd[shuffled[n_train:n_train + n_val]] = "val"
        rnd[shuffled[n_train + n_val:]] = "test"

        chrono = pos[np.lexsort((pos, order_key[pos]))]
        tmp[chrono[n_train:n_train + n_val]] = "val"
        tmp[chrono[n_train + n_val:]] = "test"
    df["split_random"] = rnd
    df["split_temporal"] = tmp
    return df


# --------------------------------------------------------------------------- pipeline
def build():
    frames, core_used, dropped = [], {}, []
    vrep, vstats, vctx = [], {}, []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        xm_users = multi_market_reviewers(zf) if DROP_XMARKET_USERS else None
        if xm_users is not None:
            print(f"  dropping {len(xm_users)} raw reviewer IDs found in more than one market", flush=True)

        def read(market, category):
            raw = read_context(zf, market, category)
            if raw is not None and xm_users is not None:
                n0 = len(raw)
                raw = raw[~raw["user"].isin(xm_users)]
                vctx.append({"market": market, "category": category, "step": "drop_xmarket_users",
                             "rows_before": n0, "rows_after": len(raw)})
            return raw

        for category in CATEGORIES:
            vmap = None
            if MERGE_VARIANTS:
                fr = {m: r for m in MARKETS if (r := read(m, category)) is not None and not r.empty}
                vctx[:] = [v for v in vctx if v["category"] != category]   # counted again below
                vmap, st = variant_map(fr) if fr else ({}, {})
                vstats[category] = st
                vrep.append(pd.DataFrame({"category": category, "asin": list(vmap.keys()), "rep": list(vmap.values())}))
                del fr
            for market in MARKETS:
                raw = read(market, category)
                if raw is not None and vmap is not None:
                    n0 = len(raw)
                    raw = collapse_variants(raw, vmap)
                    vctx.append({"market": market, "category": category, "step": "merge_variants",
                                 "rows_before": n0, "rows_after": len(raw)})
                elif raw is not None and MERGE_VARIANTS_CONTEXT and not raw.empty:
                    cmap, st = variant_map({market: raw})
                    vstats[f"{market} {category}"] = st
                    vrep.append(pd.DataFrame({"market": market, "category": category,
                                              "asin": list(cmap.keys()), "rep": list(cmap.values())}))
                    n0 = len(raw)
                    raw = collapse_variants(raw, cmap)
                    vctx.append({"market": market, "category": category, "step": "merge_variants_context",
                                 "rows_before": n0, "rows_after": len(raw)})
                if raw is None or raw.empty:
                    dropped.append([market, category, "absent/empty"])
                    continue
                kept, kk = adaptive_core(raw)
                if kk is None:
                    dropped.append([market, category, "below thresholds"])
                    continue
                core_used[(market, category)] = kk
                frames.append(kept.assign(market=market, category=category))
                print(f"  {market} x {category:<28} {len(kept):>8} inter | "
                      f"{kept['user'].nunique():>7} users | {kept['item'].nunique():>6} items "
                      f"| {kk}-core", flush=True)

    data = pd.concat(frames, ignore_index=True)

    # Global ID space: users keyed by market (disjoint), items keyed by ASIN (shared).
    data["user_key"] = data["market"] + "::" + data["user"]
    data["u"] = pd.factorize(data["user_key"], sort=True)[0].astype(np.int32)
    data["i"] = pd.factorize(data["item"], sort=True)[0].astype(np.int32)

    rng = np.random.default_rng(SPLIT_SEED)
    blocks = []
    for _, block in data.groupby(["market", "category"], sort=True):
        blocks.append(assign_splits(block, rng))
    data = pd.concat(blocks, ignore_index=True)

    data = data[["market", "category", "u", "i", "date", "split_random", "split_temporal"]]

    rows = []
    for (market, category), block in data.groupby(["market", "category"], sort=True):
        rec = {
            "market": market, "category": category,
            "core": core_used[(market, category)],
            "users": int(block["u"].nunique()), "items": int(block["i"].nunique()),
            "interactions": int(len(block)),
        }
        for split in ("random", "temporal"):
            counts = block[f"split_{split}"].value_counts()
            for part in ("train", "val", "test"):
                rec[f"n_{split}_{part}"] = int(counts.get(part, 0))
        rec["median_profile"] = float(block.groupby("u").size().median())
        rows.append(rec)
    contexts = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT_DIR / f"interactions{DATA_TAG}.parquet", index=False)
    contexts.to_parquet(RESULTS_DIR / f"contexts{DATA_TAG}.parquet", index=False)
    (RESULTS_DIR / f"dropped_contexts{DATA_TAG}.json").write_text(json.dumps(dropped, indent=1), encoding='utf-8', newline='\n')
    if MERGE_VARIANTS or MERGE_VARIANTS_CONTEXT or DROP_XMARKET_USERS:   # diagnostics of the sensitivity variants
        if vrep:
            pd.concat(vrep, ignore_index=True).to_parquet(RESULTS_DIR / f"variant_map{DATA_TAG}.parquet", index=False)
        (RESULTS_DIR / f"variant_stats{DATA_TAG}.json").write_text(json.dumps(
            {"settings": {"merge_variants": "context" if MERGE_VARIANTS_CONTEXT else MERGE_VARIANTS,
                          "drop_xmarket_users": DROP_XMARKET_USERS,
                          "n_xmarket_reviewer_ids": None if xm_users is None else len(xm_users)},
             "link_stats": vstats, "per_context": vctx}, indent=1), encoding='utf-8', newline='\n')
    return data, contexts


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / f"interactions{DATA_TAG}.parquet")


def load_for(split: str) -> pd.DataFrame:
    """Interaction table for a split: the calendar-cutoff table for 'calendar', else the main one."""
    if split == "calendar":
        return pd.read_parquet(OUT_DIR / "interactions_calendar.parquet")
    return load()


def load_contexts() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"contexts{DATA_TAG}.parquet")


if __name__ == "__main__":
    data, contexts = build()
    print(f"\n{len(contexts)} contexts | {len(data):,} interactions | "
          f"{data['u'].nunique():,} users | {data['i'].nunique():,} items")
    print(f"categories: {contexts['category'].nunique()}  markets: {contexts['market'].nunique()}")
