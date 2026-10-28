"""Identity checks on the RAW XMarket files (reviewer IDs and ASINs exactly as in the files, before any prefixing),
and on the retained (filtered) rows, using the raw-file pass of aud_raw.py.

  1. Reviewer IDs occurring in more than one market (any category), and in more than one market within one category.
  2. Reviewer IDs occurring in more than one category within a market (the XC coupling; expected).
  3. ASINs occurring in more than one category: within a market, and anywhere (category files of any markets).
  4. ASINs occurring in more than one market within a category (the XM coupling; expected).
  5. The only identifier transformation: src/data.py sets user_key = market + '::' + reviewer_id, keeps the ASIN unchanged,
     and integer-codes both with pd.factorize(sort=True). Verified here: #distinct u in data/interactions.parquet ==
     #distinct (market, reviewer_id) among retained rows; #distinct i == #distinct ASINs among retained rows; per context the
     retained users/products sets have the same sizes.
Writes paper/raw/aud_identity.json.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, already_done
if already_done(f'{PAPER_RAW}/aud_identity.json', json_key='cross_market_copies'): _sys.exit(0)

import sys, json
sys.path.insert(0, SRC); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import data as D
import aud_raw

R = aud_raw.compute(); ids = R['ids']
out = {}
for stage in ('raw', 'kept'):
    U = pd.concat([pd.DataFrame({'id': v[f'{stage}_users'], 'market': m, 'category': c}) for (m, c), v in ids.items() if len(v[f'{stage}_users'])])
    I = pd.concat([pd.DataFrame({'id': v[f'{stage}_items'], 'market': m, 'category': c}) for (m, c), v in ids.items() if len(v[f'{stage}_items'])])
    r = {}
    nm = U.groupby('id').market.nunique()
    r['reviewer_ids'] = int(len(nm)); r['reviewer_ids_in_gt1_market_any_category'] = int((nm > 1).sum())
    nmc = U.groupby(['category', 'id']).market.nunique()
    r['reviewer_ids_in_gt1_market_same_category'] = int(nmc[nmc > 1].index.get_level_values('id').nunique())
    r['reviewer_ids_in_gt1_market_same_category_by_category'] = {k: int(v) for k, v in (nmc > 1).groupby(level='category').sum().items() if v}
    ncm = U.groupby(['market', 'id']).category.nunique()
    r['market_user_keys'] = int(len(ncm)); r['market_user_keys_in_gt1_category'] = int((ncm > 1).sum())
    r['market_user_keys_in_gt1_category_by_market'] = {k: int(v) for k, v in (ncm > 1).groupby(level='market').sum().items()}
    # no example identifiers are written (only counts)
    nc = I.groupby('id').category.nunique()
    r['asins'] = int(len(nc)); r['asins_in_gt1_category_anywhere'] = int((nc > 1).sum())
    ncw = I.groupby(['market', 'id']).category.nunique()
    r['asins_in_gt1_category_within_a_market'] = int(ncw[ncw > 1].index.get_level_values('id').nunique())
    r['market_asin_pairs_in_gt1_category'] = int((ncw > 1).sum())
    r['asins_in_gt1_category_within_a_market_by_market'] = {k: int(v) for k, v in (ncw > 1).groupby(level='market').sum().items() if v}
    if r['asins_in_gt1_category_anywhere']:
        cc = I[I.id.isin(nc[nc > 1].index)].groupby('id').category.agg(lambda s: '|'.join(sorted(set(s))))
        r['category_combinations_of_multi_category_asins'] = {k: int(v) for k, v in cc.value_counts().head(15).items()}
    nmk = I.groupby(['category', 'id']).market.nunique()
    r['asins_in_gt1_market_within_category'] = int((nmk > 1).sum())
    out[stage] = r
    print(f'--- {stage} ---'); print(json.dumps(r, indent=1))

# transformation check against the paper data
d = D.load()
ku = sum(len(v['kept_users']) for v in ids.values())        # (market, category, reviewer) - per context
mu = pd.concat([pd.DataFrame({'id': v['kept_users'], 'market': m}) for (m, c), v in ids.items() if len(v['kept_users'])]).drop_duplicates()
ki = pd.unique(np.concatenate([v['kept_items'] for v in ids.values() if len(v['kept_items'])]))
per = d.groupby(['market', 'category']).agg(u=('u', 'nunique'), i=('i', 'nunique'))
same_ctx = all(per.loc[k, 'u'] == len(v['kept_users']) and per.loc[k, 'i'] == len(v['kept_items']) for k, v in ids.items() if len(v['kept_users']))
chk = dict(distinct_u_in_paper_data=int(d.u.nunique()), distinct_market_reviewer_pairs_retained=int(len(mu)),
           distinct_i_in_paper_data=int(d.i.nunique()), distinct_asins_retained=int(len(ki)), per_context_sizes_equal=bool(same_ctx),
           u_is_contiguous_0_to_n=bool(d.u.min() == 0 and d.u.max() == d.u.nunique() - 1), i_is_contiguous_0_to_n=bool(d.i.min() == 0 and d.i.max() == d.i.nunique() - 1))
chk['only_transformation_market_prefix'] = bool(chk['distinct_u_in_paper_data'] == chk['distinct_market_reviewer_pairs_retained'] and chk['distinct_i_in_paper_data'] == chk['distinct_asins_retained'] and same_ctx)
# what the un-prefixed alternative would have merged
chk['users_that_would_merge_without_prefix'] = int(len(mu) - mu.id.nunique())
out['transformation'] = chk
print('--- transformation ---'); print(json.dumps(chk, indent=1))
json.dump(out, open(f'{PAPER_RAW}/aud_identity.json', 'w', encoding='utf-8', newline='\n'), indent=1)

# ------------------------------------------------------------------ 6. same reviewer ID in several markets: copies of one review, and test leakage
# Reviewer IDs found in >1 market are mapped back to the paper data (u = factorize(sorted market::reviewer), i = factorize(sorted ASIN)).
# For every retained row of such a reviewer we ask whether the same (reviewer, ASIN) is also retained in another market of the same
# category (a cross-listed copy of one review), with the same date / rating, and - for each split - whether a TARGET test (or val)
# row has its copy in the TRAINING rows of a same-category donor market (then pooling that donor puts the held-out interaction, under
# another user key, into the training data).
import zipfile, io, gzip
multi_h = set()
U = pd.concat([pd.DataFrame({'id': v['kept_users'], 'market': m}) for (m, c), v in ids.items() if len(v['kept_users'])]).drop_duplicates()
mm = U.groupby('id').market.nunique(); multi_h = set(mm[mm > 1].index.tolist())
rows = []; all_keys = []; all_items = []
with zipfile.ZipFile(D.ZIP_PATH) as zf:
    for (m, c), v in sorted(ids.items()):
        if not len(v['kept_users']): continue
        ded = D.read_context(zf, m, c)                       # pipeline's own dedup (strings)
        kept, kk = D.adaptive_core(ded)
        all_keys.append((m + '::' + kept.user.astype(str)).unique()); all_items.append(kept.item.astype(str).unique())
        h = pd.util.hash_array(kept.user.astype(str).to_numpy(dtype=object))
        sel = kept[np.isin(h, list(multi_h))]
        rows.append(pd.DataFrame({'market': m, 'category': c, 'user': sel.user.astype(str).values, 'item': sel.item.astype(str).values,
                                  'date': sel.date.astype(str).values}))
        del ded, kept
ukeys = np.sort(pd.unique(np.concatenate(all_keys))); iks = np.sort(pd.unique(np.concatenate(all_items)))
M = pd.concat(rows, ignore_index=True)
M['u'] = np.searchsorted(ukeys, (M.market + '::' + M.user).values).astype(np.int32); M['i'] = np.searchsorted(iks, M.item.values).astype(np.int32)
assert len(ukeys) == d.u.nunique() and len(iks) == d.i.nunique()
M = M.merge(d[['market', 'category', 'u', 'i', 'date', 'split_random', 'split_temporal']], on=['market', 'category', 'u', 'i'], how='left', suffixes=('', '_pd'))
assert M.split_random.notna().all() and (M.date == M.date_pd).all(), 'mapping to paper data failed'
g = M.groupby(['category', 'user', 'item'])
M['n_mk'] = g.market.transform('nunique'); M['n_dates'] = g.date.transform('nunique')
cp = M[M.n_mk > 1]
leak = {'multi_market_reviewers_retained': len(multi_h), 'retained_rows_of_these_reviewers': int(len(M)),
        'retained_rows_with_a_copy_in_another_market_same_category': int(len(cp)),
        'distinct_reviewer_item_pairs_copied': int(cp.groupby(['category', 'user', 'item']).ngroups),
        'copies_with_identical_date': int(cp[cp.n_dates == 1].groupby(['category', 'user', 'item']).ngroups),
        'share_of_all_retained_rows': float(len(cp) / len(d))}
for split in ('random', 'temporal'):
    col = f'split_{split}'
    tr = cp[cp[col] == 'train'][['category', 'user', 'item', 'market']].rename(columns={'market': 'donor_market'})
    ho = cp[cp[col].isin(['test', 'val'])][['category', 'user', 'item', 'market', col]]
    j = ho.merge(tr, on=['category', 'user', 'item']); j = j[j.market != j.donor_market]
    tst = j[j[col] == 'test']
    nt = d[d[col] == 'test'].groupby(['market', 'category']).size()
    per = tst.drop_duplicates(['market', 'category', 'user', 'item']).groupby(['market', 'category']).size()
    leak[split] = {'target_test_rows_with_copy_in_donor_train': int(tst.drop_duplicates(['market', 'category', 'user', 'item']).shape[0]),
                   'target_val_rows_with_copy_in_donor_train': int(j[j[col] == 'val'].drop_duplicates(['market', 'category', 'user', 'item']).shape[0]),
                   'ordered_target_donor_pairs_affected': int(tst.groupby(['market', 'category', 'donor_market']).ngroups),
                   'target_contexts_affected': int(len(per)),
                   'share_of_all_test_rows': float(per.sum() / nt.sum()),
                   'per_context_share_of_test_rows': {f'{a} {b}': round(float(v / nt[(a, b)]), 5) for (a, b), v in per.sort_values(ascending=False).items()}}
out['cross_market_copies'] = leak
print('--- cross-market copies / leakage ---'); print(json.dumps(leak, indent=1))
M.to_csv(f'{PAPER_RAW}/aud_identity_multimarket_rows.csv', index=False, lineterminator='\n')
json.dump(out, open(f'{PAPER_RAW}/aud_identity.json', 'w', encoding='utf-8', newline='\n'), indent=1)
