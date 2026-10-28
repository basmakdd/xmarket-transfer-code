"""One pass over the RAW XMarket files (before any prefixing), shared by identity_checks.py and s2_accounting.py.

For every (market, category) name the pipeline tries (src/data.py: 7 markets x 16 categories), records
  - whether the zip member exists and whether it is empty,
  - raw rows, distinct (user, item) pairs (= rows after src/data.py's deduplication), duplicate structure,
  - rating distribution, date range/format,
  - the adaptive k-core result, recomputed with src/data.py's own k_core()/adaptive_core() on the deduplicated
    rows (deduplication re-implemented with the same rule as data.read_context: stable sort by date, keep the
    first row of each (user, item); categorical columns are used only to save memory; s2_accounting.py checks
    that the retained counts equal results/contexts.parquet for all 74 contexts),
  - 64-bit hashes (pandas.util.hash_array) of the raw reviewer IDs / ASINs before and after filtering, for the
    identity checks (hashes instead of strings to fit in memory; collision probability for ~1e7 distinct
    strings is ~3e-6).
Per-file results are cached in paper/raw/aud_rawcache/ (resumable; delete the folder to recompute).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, io, gzip, zipfile, pickle, time, gc
sys.path.insert(0, SRC)
import numpy as np, pandas as pd
import data as D

CDIR = f'{PAPER_RAW}/aud_rawcache'


def H(values):
    return np.unique(pd.util.hash_array(np.asarray(values, dtype=object)))


def one(zf, market, category):
    name = f'xmarket_categories/ratings_{market}_{category}.txt.gz'
    rec = dict(market=market, category=category, file=name.split('/')[-1])
    try:
        blob = zf.read(name)
    except KeyError:
        rec['status'] = 'absent'; return rec, None
    text = gzip.decompress(blob); del blob
    if not text.strip():
        rec['status'] = 'empty'; return rec, None
    df = pd.read_csv(io.BytesIO(text), sep=' ', header=None, names=['user', 'item', 'rating', 'date'],
                     dtype={'user': 'category', 'item': 'category', 'rating': 'float32', 'date': 'category'})
    del text; gc.collect()
    if df.empty:
        rec['status'] = 'empty'; return rec, None
    rec['status'] = 'ok'
    dates = np.asarray(df.date.cat.categories, dtype=object)
    rec['date_iso'] = bool(pd.Series(dates).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all())
    rec['date_min'], rec['date_max'] = str(min(dates)), str(max(dates))
    # chronological order of the date strings (ISO dates: lexicographic == chronological), as data.read_context sorts strings
    order = np.argsort(np.argsort(dates.astype(str), kind='stable'))
    dkey = order[df.date.cat.codes.to_numpy()]
    uc = df.user.cat.codes.to_numpy().astype(np.int64); ic = df.item.cat.codes.to_numpy().astype(np.int64)
    key = uc * (ic.max() + 1) + ic
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    rec.update(raw_rows=int(len(df)), raw_pairs=int(len(uk)), dup_pairs=int((cnt > 1).sum()),
               raw_users=int(len(np.unique(uc))), raw_items=int(len(np.unique(ic))),
               **{f'rating_{int(k)}': int(v) for k, v in df.rating.value_counts().items()})
    if (cnt > 1).any():
        dmin = np.full(len(uk), np.iinfo(np.int64).max); np.minimum.at(dmin, inv, dkey)
        dmax = np.full(len(uk), -1); np.maximum.at(dmax, inv, dkey)
        at_min = np.bincount(inv, weights=(dkey == dmin[inv]), minlength=len(uk))
        rec['dup_pairs_diff_dates'] = int(((cnt > 1) & (dmax > dmin)).sum())
        rec['dup_pairs_tie_at_min'] = int(((cnt > 1) & (at_min > 1)).sum())
    else:
        rec['dup_pairs_diff_dates'] = rec['dup_pairs_tie_at_min'] = 0
    # data.read_context: df.sort_values("date", kind="stable").drop_duplicates(["user","item"], keep="first")
    o = np.argsort(dkey, kind='stable')
    k_sorted = key[o]
    _, first = np.unique(k_sorted, return_index=True)
    keep_rows = o[np.sort(first)]
    ded = pd.DataFrame({'user': uc[keep_rows], 'item': ic[keep_rows]})
    rec['dedup_rows'] = int(len(ded)); assert len(ded) == len(uk)
    kept, kk = D.adaptive_core(ded)
    k2 = D.k_core(ded, 2)
    rec.update(core=kk, kept_rows=int(len(kept)), kept_users=int(kept.user.nunique()) if kk else 0,
               kept_items=int(kept.item.nunique()) if kk else 0, k2_rows=int(len(k2)), k2_items=int(k2.item.nunique()))
    ucat = np.asarray(df.user.cat.categories, dtype=object); icat = np.asarray(df.item.cat.categories, dtype=object)
    ids = dict(raw_users=H(ucat[np.unique(uc)]), raw_items=H(icat[np.unique(ic)]),
               kept_users=H(ucat[np.unique(kept.user.to_numpy())]) if kk else np.array([], np.uint64),
               kept_items=H(icat[np.unique(kept.item.to_numpy())]) if kk else np.array([], np.uint64))
    return rec, ids


def compute(fresh=None):
    """Per-file raw summaries, cached in paper/raw/aud_rawcache/. With --fresh on the command line (or fresh=True) the cache is
    ignored and every file is recomputed (and the cache rewritten)."""
    if fresh is None: fresh = '--fresh' in sys.argv[1:]
    os.makedirs(CDIR, exist_ok=True)
    rows, ids = [], {}
    t0 = time.time()
    with zipfile.ZipFile(D.ZIP_PATH) as zf:
        members = sorted(zf.namelist())
        for category in D.CATEGORIES:
            for market in D.MARKETS:
                fn = f'{CDIR}/{market}_{category}.pkl'
                if os.path.exists(fn) and not fresh:
                    rec, idd = pickle.load(open(fn, 'rb'))
                else:
                    rec, idd = one(zf, market, category)
                    pickle.dump((rec, idd), open(fn + '.tmp', 'wb')); os.replace(fn + '.tmp', fn)
                    print(market, category, rec['status'], rec.get('raw_rows'), rec.get('core'), round(time.time() - t0), flush=True)
                    gc.collect()
                rows.append(rec)
                if idd is not None: ids[(market, category)] = idd
    return dict(table=pd.DataFrame(rows), ids=ids, members=members)


if __name__ == '__main__':
    from common import already_done
    if already_done(*[f'{CDIR}/{m}_{c}.pkl' for c in D.CATEGORIES for m in D.MARKETS]): sys.exit(0)
    r = compute(); print(r['table'].to_string())
