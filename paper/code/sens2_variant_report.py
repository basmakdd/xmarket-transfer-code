"""Merge report for the within-context variant merge _nvc (src/data.py, XM_MERGE_VARIANTS=context), compared with the
paper data and the pooled-market merge _nv.

Reads results/variant_stats_nvc.json, variant_map_nvc.parquet (market, category, asin, rep), contexts{,_nv,_nvc}.parquet,
dropped_contexts{,_nv,_nvc}.json, overlap_random{,_nv,_nvc}.parquet, paper/results/sens_diag_{orig,nv,nvc}.csv
(variant_diagnostics.py) and the raw-item hashes in paper/raw/aud_rawcache/ (aud_raw.py; pandas.util.hash_array of the
raw ASINs per file) for the cross-market identifier check.
Writes paper/results/sens2_variant_components.csv, sens2_variant_top10.csv, sens2_context_sizes.csv, sens2_diag_compare.csv,
sens2_shared_products.csv and prints everything.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json, glob
import numpy as np, pandas as pd

NOTES = f'{REPO}/paper/results'
RAW = PAPER_RAW   # paper/raw unless the environment variable PAPER_RAW is set
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
TAGS = ['', '_nv', '_nvc']
NAME = {'': 'orig', '_nv': '_nv', '_nvc': '_nvc'}


def ctx(tag):
    c = pd.read_parquet(f'{RESULTS}/contexts{tag}.parquet'); c['t'] = c.market + ' ' + c.category
    return c.set_index('t')


st = json.load(open(f'{RESULTS}/variant_stats_nvc.json', encoding='utf-8'))
print('settings:', st['settings'])
pc = pd.DataFrame(st['per_context']); pc['t'] = pc.market + ' ' + pc.category
pc['share'] = (pc.rows_before - pc.rows_after) / pc.rows_before
print(f'-- merge_variants_context: deduplicated raw rows {pc.rows_before.sum():,} -> {pc.rows_after.sum():,} '
      f'(removed {(pc.rows_before - pc.rows_after).sum():,} = {100 * (1 - pc.rows_after.sum() / pc.rows_before.sum()):.2f}%); '
      f'per file share removed: median {100 * pc.share.median():.2f}%, max {100 * pc.share.max():.2f}% ({pc.loc[pc.share.idxmax(), "t"]})')
M = pd.read_parquet(f'{RESULTS}/variant_map_nvc.parquet')
comp = M.groupby(['market', 'category', 'rep']).size().rename('size').reset_index()
comp['t'] = comp.market + ' ' + comp.category
print(f'-- variant linking (per market x category file): {len(comp)} components (size >= 2), {len(M)} products in them, '
      f'{len(M) - len(comp)} products removed by merging; files with >= 1 component: {comp.t.nunique()} of {len(pc)}')
ls = pd.DataFrame(st['link_stats']).T
print(f'   links in total {int(ls.links.sum()):,}; candidates verified {int(ls.candidates.sum()):,}')
bins = pd.cut(comp['size'], [1, 2, 3, 5, 10, 20, 50, 10 ** 6], labels=['2', '3', '4-5', '6-10', '11-20', '21-50', '>50'])
print('   component size distribution:', bins.value_counts().sort_index().to_dict(),
      f'| median {comp["size"].median():.0f}, mean {comp["size"].mean():.2f}, max {comp["size"].max()}')
print('   components per category:', comp.groupby('category').size().to_dict())
print('   components per market:', comp.groupby('market').size().to_dict())
comp.to_csv(f'{NOTES}/sens2_variant_components.csv', index=False, lineterminator='\n')
top = comp.sort_values(['size', 'rep'], ascending=[False, True]).head(10)
# product identifiers (ASINs) are not written: only the context and the size of each component
T = pd.DataFrame([dict(context=r.t, size=r['size']) for _, r in top.iterrows()]); T.to_csv(f'{NOTES}/sens2_variant_top10.csv', index=False, lineterminator='\n')
print('   10 largest components:'); print(T.to_string(index=False))

# ------------------------------------------------------------- cross-market identifiers
# per category: ASINs touched by the merge in some market; for every market whose raw file contains the ASIN, its identifier
# after merging is rep_m(a) = map_m.get(a, a). An ASIN is 'split' if it is present in >= 2 markets and these differ.
print('\n-- cross-market identifiers after the within-context merge (raw presence from aud_rawcache hashes)')
H = {}
for fn in glob.glob(f'{RAW}/aud_rawcache/*.pkl'):
    b = os.path.basename(fn)[:-4]; m, c = b.split('_', 1)
    x = pd.read_pickle(fn)
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], dict) and 'raw_items' in x[1]:
        H[(m, c)] = set(x[1]['raw_items'].tolist())
out = []
for c, g in M.groupby('category'):
    mk = sorted({m for (m, cc) in H if cc == c})
    maps = {m: dict(zip(gg.asin, gg.rep)) for m, gg in g.groupby('market')}
    asins = sorted(set(g.asin))
    hv = pd.util.hash_array(np.asarray(asins, dtype=object))
    for a, h in zip(asins, hv):
        pres = [m for m in mk if int(h) in H[(m, c)]]
        ids = {m: maps.get(m, {}).get(a, a) for m in pres}
        out.append(dict(category=c, asin=a, markets_present=len(pres), markets_merged=sum(a in maps.get(m, {}) for m in pres),
                        distinct_ids=len(set(ids.values())), self_kept_somewhere=any(v == a for v in ids.values())))
X = pd.DataFrame(out)
multi = X[X.markets_present >= 2]
print(f'   ASINs in a merged component in >= 1 market: {len(X):,}; present in >= 2 markets (raw): {len(multi):,}; '
      f'of these with different identifiers across markets: {int((multi.distinct_ids > 1).sum()):,} '
      f'({100 * (multi.distinct_ids > 1).mean():.1f}%); merged in every market where present: {int((multi.markets_merged == multi.markets_present).sum()):,}')

# ------------------------------------------------------------- contexts surviving, sizes
C = {t: ctx(t) for t in TAGS}
C0 = C['']
d = {t: {f'{a} {b}': r for a, b, r in json.load(open(f'{RESULTS}/dropped_contexts{t}.json', encoding='utf-8'))} for t in TAGS}
sizes = pd.DataFrame({'orig_inter': C0.interactions, 'orig_core': C0.core, 'orig_items': C0['items']})
for t in ('_nv', '_nvc'):
    Ct = C[t]
    sizes[f'{t}_inter'] = Ct.interactions; sizes[f'{t}_core'] = Ct.core; sizes[f'{t}_items'] = Ct['items']
    lost = sorted(set(C0.index) - set(Ct.index)); new = sorted(set(Ct.index) - set(C0.index))
    print(f'\n-- {t}: contexts surviving the core filter: {len(Ct)} (paper data {len(C0)}); lost: {[(x, d[t].get(x)) for x in lost]}; new: {new}')
    both = C0.index.intersection(Ct.index)
    ch = Ct.loc[both, 'interactions'] / C0.loc[both, 'interactions']
    print(f'   retained interactions: {Ct.interactions.sum():,} vs {C0.interactions.sum():,} ({100 * (Ct.interactions.sum() / C0.interactions.sum() - 1):+.1f}%); '
          f'per common context ratio: median {ch.median():.3f}, quartiles {ch.quantile(.25):.3f}-{ch.quantile(.75):.3f}, '
          f'min {ch.min():.3f} ({ch.idxmin()}), max {ch.max():.3f} ({ch.idxmax()})')
    cc = C0.loc[both, 'core'] != Ct.loc[both, 'core']
    print(f'   core level changed in {int(cc.sum())} of {len(both)} common contexts: '
          f'{[(x, int(C0.loc[x, "core"]), int(Ct.loc[x, "core"])) for x in both[cc.values]]}')
print('\n-- _nvc vs _nv: in _nv not _nvc:', sorted(set(C['_nv'].index) - set(C['_nvc'].index)),
      '| in _nvc not _nv:', sorted(set(C['_nvc'].index) - set(C['_nv'].index)))
pcs = pc.set_index('t').share
sizes['_nvc_raw_share_removed'] = pcs
sizes.to_csv(f'{NOTES}/sens2_context_sizes.csv', lineterminator='\n')
z = sizes.assign(ratio_nvc=sizes['_nvc_inter'] / sizes.orig_inter)
print('\n10 contexts with the largest relative loss of retained interactions under _nvc:')
print(z.sort_values('ratio_nvc')[['orig_inter', 'orig_core', '_nv_inter', '_nvc_inter', '_nvc_core', 'ratio_nvc', '_nvc_raw_share_removed']].head(10).round(3).to_string())
print('\ncontexts whose retained interactions grow under _nvc:')
print(z[z.ratio_nvc > 1][['orig_inter', 'orig_core', 'orig_items', '_nvc_inter', '_nvc_core', '_nvc_items', 'ratio_nvc']].round(3).to_string())

# ------------------------------------------------------------- diagnostics before / after
D = {}
for t, fn in (('orig', 'sens_diag_orig.csv'), ('_nv', 'sens_diag_nv.csv'), ('_nvc', 'sens_diag_nvc.csv')):
    x = pd.read_csv(f'{NOTES}/{fn}'); x['t'] = x.market + ' ' + x.category; D[t] = x.set_index('t')
print('\n-- within-context diagnostics (variant_diagnostics.py; retained data, all splits)')
rows = []
for setname in ('all', 'common'):
    com = set.intersection(*[set(x.index) for x in D.values()])
    for t, x in D.items():
        y = x if setname == 'all' else x.loc[sorted(com)]
        pooled = (y.same_date_share * y.interactions).sum() / y.interactions.sum()
        r = dict(set=setname, data=t, contexts=len(y), same_date_median=y.same_date_share.median(), same_date_q1=y.same_date_share.quantile(.25),
                 same_date_q3=y.same_date_share.quantile(.75), same_date_pooled=pooled, jac_mass_median=y.jac_mass.median(),
                 jac_mass_q1=y.jac_mass.quantile(.25), jac_mass_q3=y.jac_mass.quantile(.75), jac_mass_max=y.jac_mass.max(),
                 jac_mass_ge033=int((y.jac_mass >= 0.33).sum()), jac_mass_ge010=int((y.jac_mass >= 0.10).sum()))
        rows.append(r)
DC = pd.DataFrame(rows); DC.to_csv(f'{NOTES}/sens2_diag_compare.csv', index=False, lineterminator='\n')
print(DC.round(3).to_string(index=False))
print('   _nvc contexts with the largest remaining Jaccard>=0.8 mass:')
print(D['_nvc'].sort_values('jac_mass', ascending=False)[['interactions', 'same_date_share', 'jac_mass']].head(5).round(3).to_string())
print('   _nvc contexts with the largest same user-date share:')
print(D['_nvc'].sort_values('same_date_share', ascending=False)[['interactions', 'same_date_share', 'jac_mass']].head(5).round(3).to_string())

# ------------------------------------------------------------- XM pairs and shared products
O = {}
for t in TAGS:
    o = pd.read_parquet(f'{RESULTS}/overlap_random{t}.parquet'); o = o[o.channel == 'XM'].copy()
    o['t'] = o.target_market + ' ' + o.target_category; O[NAME[t]] = o
print('\n-- XM pairs and shared products')
for k, o in O.items():
    print(f'   {k}: XM pairs {len(o)}, targets with an XM donor {o.t.nunique()}, median shared products {o.shared_items.median():.0f}, '
          f'pairs with >= 10 shared products {int((o.shared_items >= 10).sum())}, median shared-item interaction cover {o.shared_item_interaction_cover.median():.3f}')
key = ['t', 'source_market'] if 'source_market' in O['orig'].columns else ['t', 'donor_market']
S = pd.concat({k: o.set_index(key).shared_items for k, o in O.items()}, axis=1)
S.to_csv(f'{NOTES}/sens2_shared_products.csv', lineterminator='\n')
SS = S.dropna()
for a in ('_nv', '_nvc'):
    r = SS[a] / SS['orig']
    print(f'   pairs common to orig/_nv/_nvc ({len(SS)}): shared products {a}/orig median ratio {r.median():.3f}, quartiles {r.quantile(.25):.3f}-{r.quantile(.75):.3f}; '
          f'fewer shared than orig in {int((SS[a] < SS["orig"]).sum())} pairs')
r = SS['_nvc'] / SS['_nv']
print(f'   _nvc/_nv median ratio {r.median():.3f}; _nvc fewer than _nv in {int((SS["_nvc"] < SS["_nv"]).sum())}, more in {int((SS["_nvc"] > SS["_nv"]).sum())} pairs')
