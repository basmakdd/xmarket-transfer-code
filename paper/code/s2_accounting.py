"""Supplementary Table S2: dataset accounting per input file.

Uses the raw-file pass of aud_raw.py (src/data.py's own read_context/adaptive_core re-applied to every file) and
cross-checks the retained counts against results/contexts.parquet and data/interactions.parquet.
Per (market, category) file: raw rows, rows after collapsing repeated user-product pairs (= distinct pairs; the
pipeline keeps the earliest-dated row), retained interactions, retention rate (retained / after dedup and
retained / raw), raw and retained users and products, selected core level k, validation users (random split).
Dropped files: the size of their 2-core (the most permissive level) shows which threshold failed.
Outputs: paper/tables_S2.tex (longtable), paper/raw/aud_S2.csv, paper/raw/aud_input_files.txt.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, json
sys.path.insert(0, SRC); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import data as D
import aud_raw

R = aud_raw.compute(); T = R['table'].copy()
print('zip members:', len(R['members']), '| names tried by src/data.py:', len(T))
print(T.status.value_counts().to_dict())
present = T[T.status == 'ok'].copy()
absent = T[T.status != 'ok']
print('absent/empty:', [(m, c, s) for m, c, s in zip(absent.market, absent.category, absent.status)])
open(f'{PAPER_RAW}/aud_input_files.txt', 'w', encoding='utf-8', newline='\n').write('\n'.join(sorted(present.file)) + '\n')
print('non-empty input files:', len(present))

ctx = pd.read_parquet(f'{RESULTS}/contexts.parquet')
d = D.load()
chk = d.groupby(['market', 'category']).agg(n=('u', 'size'), pu=('u', 'nunique'), pi=('i', 'nunique')).reset_index()
present = present.merge(ctx[['market', 'category', 'core', 'users', 'items', 'interactions', 'n_random_val', 'n_temporal_val']], on=['market', 'category'], how='left', suffixes=('', '_ctx'))
present = present.merge(chk, on=['market', 'category'], how='left')
kept = present[present.core.notna()]
assert (kept.kept_rows == kept.interactions).all() and (kept.kept_rows == kept.n).all(), 'retained rows differ from contexts.parquet'
assert (kept.kept_users == kept.users).all() and (kept.kept_items == kept['items']).all() and (kept.pu == kept.users).all() and (kept.pi == kept['items']).all() and (kept.core == kept.core_ctx).all()
print('cross-check passed: recomputed retained rows/users/products/k equal results/contexts.parquet and data/interactions.parquet for all',
      len(kept), 'kept contexts')
# validation users per context (random split; temporal has identical sizes)
vu = d[d.split_random == 'val'].groupby(['market', 'category']).u.nunique()
present['val_users'] = [int(vu.get((m, c), 0)) if k == k else np.nan for m, c, k in zip(present.market, present.category, present.core)]
present['ret_dedup'] = present.kept_rows / present.dedup_rows
present['ret_raw'] = present.kept_rows / present.raw_rows
dropped = present[present.core.isna()]; kept = present[present.core.notna()]
assert len(kept) == 74 and len(dropped) == 24

tot = dict(raw=int(present.raw_rows.sum()), dedup=int(present.dedup_rows.sum()), kept=int(kept.kept_rows.sum()),
           raw_kept_files=int(kept.raw_rows.sum()), dedup_kept_files=int(kept.dedup_rows.sum()))
print('totals:', tot, '| retention of all raw rows %.4f, of kept files raw rows %.4f, of kept files after dedup %.4f' %
      (tot['kept'] / tot['raw'], tot['kept'] / tot['raw_kept_files'], tot['kept'] / tot['dedup_kept_files']))
print('core levels:', kept.core.astype(int).value_counts().sort_index(ascending=False).to_dict())
print('retention after dedup per kept context: median %.3f, min %.3f (%s), max %.3f (%s)' % (kept.ret_dedup.median(), kept.ret_dedup.min(),
      ' '.join(kept.loc[kept.ret_dedup.idxmin(), ['market', 'category']]), kept.ret_dedup.max(), ' '.join(kept.loc[kept.ret_dedup.idxmax(), ['market', 'category']])))
print('retained products per context: median', kept.kept_items.median())
nov = kept[(kept.n_random_val == 0)]
print('contexts without validation users (random split):', [f'{m} {c}' for m, c in zip(nov.market, nov.category)],
      '| also none on temporal split:', bool((nov.n_temporal_val == 0).all()), '| total with zero val on either split:', int(((kept.n_random_val == 0) | (kept.n_temporal_val == 0)).sum()))
print('duplicates: rows removed by dedup', int((present.raw_rows - present.dedup_rows).sum()), '| duplicated pairs', int(present.dup_pairs.sum()),
      '| with differing dates', int(present.dup_pairs_diff_dates.sum()), '| with >1 row at the earliest date', int(present.dup_pairs_tie_at_min.sum()))
rc = [c for c in present.columns if c.startswith('rating_')]
rs = present[rc].fillna(0).sum(); print('raw rating distribution (all non-empty files):', {k: int(v) for k, v in rs.items()}, '| share <=3: %.4f' % (rs[[c for c in rc if int(c.split('_')[1]) <= 3]].sum() / rs.sum()))
print('dates ISO YYYY-MM-DD in all files:', bool(present.date_iso.all()), '| range', present.date_min.min(), present.date_max.max())
# dropped: which threshold failed at k=2
dropped = dropped.assign(fail=np.where(dropped.k2_rows < D.MIN_PAIR_INTER, np.where(dropped.k2_items < D.MIN_PAIR_ITEMS, 'both', 'interactions'), 'products'))
print('dropped files, failing threshold at k=2:', dropped.fail.value_counts().to_dict())
cols = ['market', 'category', 'file', 'status', 'raw_rows', 'dedup_rows', 'kept_rows', 'ret_dedup', 'ret_raw', 'raw_users', 'kept_users', 'raw_items', 'kept_items',
        'core', 'val_users', 'k2_rows', 'k2_items', 'date_min', 'date_max'] + rc
present.assign(kept=present.core.notna())[cols + ['kept']].to_csv(f'{PAPER_RAW}/aud_S2.csv', index=False, lineterminator='\n')

# ------------------------------------------------------------------ LaTeX longtable
def nm(c): return c.replace('_and_', ' \\& ').replace('_', ' ')
def i(x): return f'{int(x):,}'
L = ['% Auto-generated by paper/code/s2_accounting.py. Requires booktabs, longtable.', '{\\scriptsize\\setlength{\\tabcolsep}{3pt}',
     '\\begin{longtable}{llrrrrrrrrcr}',
     '\\caption{Dataset accounting per input file (Supplementary Table S2). Raw: rows in the file; Dedup: after collapsing repeated user--product pairs; '
     'Kept: interactions retained by the adaptive core filter; Ret.: Kept/Dedup; $k$: selected core level; Val.: users with at least one validation interaction (random split).}\\label{tab:S2}\\\\',
     '\\toprule', 'Market & Category & Raw & Dedup & Kept & Ret. (\\%) & Users raw & Users kept & Products raw & Products kept & $k$ & Val. \\\\', '\\midrule', '\\endfirsthead',
     '\\toprule', 'Market & Category & Raw & Dedup & Kept & Ret. (\\%) & Users raw & Users kept & Products raw & Products kept & $k$ & Val. \\\\', '\\midrule', '\\endhead',
     '\\multicolumn{12}{l}{\\textit{(a) 74 retained contexts}} \\\\']
for _, r in kept.sort_values(['category', 'market']).iterrows():
    L.append(f'{r.market.upper()} & {nm(r.category)} & {i(r.raw_rows)} & {i(r.dedup_rows)} & {i(r.kept_rows)} & {100 * r.ret_dedup:.1f} & {i(r.raw_users)} & {i(r.kept_users)} & {i(r.raw_items)} & {i(r.kept_items)} & {int(r.core)} & {i(r.val_users)}{"$^*$" if r.val_users == 0 else ""} \\\\')
L.append(f'\\midrule All retained & & {i(kept.raw_rows.sum())} & {i(kept.dedup_rows.sum())} & {i(kept.kept_rows.sum())} & {100 * kept.kept_rows.sum() / kept.dedup_rows.sum():.1f} & & & & & & \\\\')
L += ['\\midrule', '\\multicolumn{12}{l}{\\textit{(b) 24 dropped files (below 800 interactions or 40 products even at $k=2$); Kept and Products kept show the 2-core}} \\\\']
for _, r in dropped.sort_values(['category', 'market']).iterrows():
    L.append(f'{r.market.upper()} & {nm(r.category)} & {i(r.raw_rows)} & {i(r.dedup_rows)} & ({i(r.k2_rows)}) & -- & {i(r.raw_users)} & -- & {i(r.raw_items)} & ({i(r.k2_items)}) & -- & -- \\\\')
L.append(f'\\midrule All 98 files & & {i(present.raw_rows.sum())} & {i(present.dedup_rows.sum())} & {i(kept.kept_rows.sum())} & {100 * kept.kept_rows.sum() / present.dedup_rows.sum():.1f} & & & & & & \\\\')
L += ['\\bottomrule',
      '\\multicolumn{12}{p{0.97\\linewidth}}{$^*$ No validation users; the first grid setting is used. '
      f'Input files: \\texttt{{ratings\\_$\\langle$market$\\rangle$\\_$\\langle$category$\\rangle$.txt.gz}} of XMarket (7 markets $\\times$ 16 categories = 112 names; {len(absent)} absent or empty: '
      + ', '.join(f'{m.upper()} {nm(c)}' for m, c in zip(absent.market, absent.category)) + '). All ratings (1--5) are kept as positive implicit interactions.} \\\\',
      '\\end{longtable}}']
open(f'{REPO}/paper/tables_S2.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('wrote paper/tables_S2.tex')
