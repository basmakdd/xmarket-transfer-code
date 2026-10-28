"""Report on the data variants built by src/data.py with XM_MERGE_VARIANTS / XM_DROP_XMARKET_USERS.

Reads results/variant_stats{TAG}.json, results/variant_map{TAG}.parquet, results/contexts{,TAG}.parquet,
results/dropped_contexts{,TAG}.json, results/overlap_random{,TAG}.parquet.
Writes paper/results/sens_variant_components.csv, sens_variant_top10.csv, sens_context_sizes.csv and prints everything.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json
import numpy as np, pandas as pd

NOTES = f'{REPO}/paper/results'
TAGS = ['_nv', '_xd', '_nvxd']
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)


def ctx(tag):
    c = pd.read_parquet(f'{RESULTS}/contexts{tag}.parquet')
    c['t'] = c.market + ' ' + c.category
    return c.set_index('t')


C0 = ctx('')
sizes = pd.DataFrame({'orig_inter': C0.interactions, 'orig_core': C0.core, 'orig_items': C0['items']})
for tag in TAGS:
    st = json.load(open(f'{RESULTS}/variant_stats{tag}.json', encoding='utf-8'))
    print(f'\n################ {tag}  settings: {st["settings"]}')
    pc = pd.DataFrame(st['per_context'])
    if len(pc):
        pc['t'] = pc.market + ' ' + pc.category
        for step, g in pc.groupby('step'):
            g = g.assign(removed=g.rows_before - g.rows_after, share=(g.rows_before - g.rows_after) / g.rows_before)
            print(f'-- {step}: deduplicated raw rows {g.rows_before.sum():,} -> {g.rows_after.sum():,} '
                  f'(removed {g.removed.sum():,} = {100 * g.removed.sum() / g.rows_before.sum():.2f}%); per file share removed: '
                  f'median {100 * g.share.median():.2f}%, max {100 * g.share.max():.2f}% ({g.loc[g.share.idxmax(), "t"]})')
            sizes[f'{tag}_{step}_raw_share_removed'] = g.set_index('t').share
    if os.path.exists(f'{RESULTS}/variant_map{tag}.parquet'):
        M = pd.read_parquet(f'{RESULTS}/variant_map{tag}.parquet')
        comp = M.groupby(['category', 'rep']).size().rename('size').reset_index()
        print(f'-- variant linking: {len(comp)} components (size >= 2), {len(M)} products in them, '
              f'{len(M) - len(comp)} products removed by merging')
        print('   link statistics per category:', json.dumps(st['link_stats']))
        print('   components per category:', comp.groupby('category').size().to_dict())
        print('   products in components per category:', comp.groupby('category')['size'].sum().to_dict())
        vc = comp['size'].value_counts().sort_index()
        bins = pd.cut(comp['size'], [1, 2, 3, 5, 10, 20, 50, 10 ** 6], labels=['2', '3', '4-5', '6-10', '11-20', '21-50', '>50'])
        print('   component size distribution:', bins.value_counts().sort_index().to_dict(),
              f'| median {comp["size"].median():.0f}, mean {comp["size"].mean():.2f}, max {comp["size"].max()}')
        if tag == '_nv':
            comp.to_csv(f'{NOTES}/sens_variant_components.csv', index=False, lineterminator='\n')
            top = comp.sort_values(['size', 'rep'], ascending=[False, True]).head(10)
            # product identifiers (ASINs) are not written: only the category and the size of each component
            T = pd.DataFrame([dict(category=r.category, size=r['size']) for _, r in top.iterrows()]); T.to_csv(f'{NOTES}/sens_variant_top10.csv', index=False, lineterminator='\n')
            print('   10 largest components:'); print(T.to_string(index=False))
    C = ctx(tag)
    sizes[f'{tag}_inter'] = C.interactions; sizes[f'{tag}_core'] = C.core; sizes[f'{tag}_items'] = C['items']
    lost = sorted(set(C0.index) - set(C.index)); new = sorted(set(C.index) - set(C0.index))
    d0 = {f'{a} {b}': r for a, b, r in json.load(open(f'{RESULTS}/dropped_contexts.json', encoding='utf-8'))}
    d1 = {f'{a} {b}': r for a, b, r in json.load(open(f'{RESULTS}/dropped_contexts{tag}.json', encoding='utf-8'))}
    print(f'-- contexts surviving the core filter: {len(C)} (paper data {len(C0)}); lost: {[(t, d1.get(t)) for t in lost]}; new: {new}')
    both = C0.index.intersection(C.index)
    ch = (C.loc[both, 'interactions'] / C0.loc[both, 'interactions'])
    print(f'-- retained interactions: {C.interactions.sum():,} vs {C0.interactions.sum():,} ({100 * (C.interactions.sum() / C0.interactions.sum() - 1):+.1f}%); '
          f'per common context ratio: median {ch.median():.3f}, quartiles {ch.quantile(.25):.3f}-{ch.quantile(.75):.3f}, min {ch.min():.3f} ({ch.idxmin()}), max {ch.max():.3f} ({ch.idxmax()})')
    cc = (C.loc[both, 'core'] != C0.loc[both, 'core'])
    print(f'-- core level changed in {int(cc.sum())} of {len(both)} common contexts: '
          f'{[(t, int(C0.loc[t, "core"]), int(C.loc[t, "core"])) for t in both[cc.values]]}')
    o0 = pd.read_parquet(f'{RESULTS}/overlap_random.parquet'); o1 = pd.read_parquet(f'{RESULTS}/overlap_random{tag}.parquet')
    x0, x1 = o0[o0.channel == 'XM'], o1[o1.channel == 'XM']
    print(f'-- XM pairs: {len(x1)} (paper data {len(x0)}); targets with an XM donor: {x1.groupby(["target_market", "target_category"]).ngroups}; '
          f'median shared products {x1.shared_items.median():.0f} (paper {x0.shared_items.median():.0f}); '
          f'median shared-item interaction cover {x1.shared_item_interaction_cover.median():.3f} (paper {x0.shared_item_interaction_cover.median():.3f})')
sizes.to_csv(f'{NOTES}/sens_context_sizes.csv', lineterminator='\n')
print('\nper-context sizes written to paper/results/sens_context_sizes.csv')
print(sizes.round(3).to_string())
# most affected contexts under _nv (retained interactions relative to the paper data)
z = sizes.assign(ratio_nv=sizes['_nv_inter'] / sizes.orig_inter, ratio_xd=sizes['_xd_inter'] / sizes.orig_inter)
print('\n10 contexts with the largest relative loss of retained interactions under _nv:')
print(z.sort_values('ratio_nv')[['orig_inter', 'orig_core', '_nv_inter', '_nv_core', 'ratio_nv', '_nv_merge_variants_raw_share_removed']].head(10).round(3).to_string())
print('\ncontexts whose retained interactions grow under _nv (core level relaxed):')
print(z[z.ratio_nv > 1][['orig_inter', 'orig_core', 'orig_items', '_nv_inter', '_nv_core', '_nv_items', 'ratio_nv']].round(3).to_string())
print('\n10 contexts with the largest relative loss under _xd:')
print(z.sort_values('ratio_xd')[['orig_inter', '_xd_inter', 'ratio_xd', '_xd_drop_xmarket_users_raw_share_removed']].head(10).round(3).to_string())
