"""Split-rule facts from src/data.py and data/interactions.parquet (paper data, seed 0).

1. Allocation rule: tabulates src/data.py cut_points(n) for n = 1..30 (train, val, test sizes).
2. Share of (context, user) profiles by n (1..7, 8+) and validation eligibility (n_val > 0).
3. Exact date ties within a user's profile in the chronological (temporal) split: share of
   (context, user) profiles with >= 1 tie at a train/val, val/test or train/test boundary
   (last item of the earlier block has the same date as the first item of the next block),
   plus the share with any tied date at all. Ties are broken by row position (src/data.py
   assign_splits: np.lexsort((pos, date))), i.e. by order after the date-sorted deduplication,
   which itself is stable in the raw file order.
4. Check that the stored split equals a re-application of the rule (sizes per user).
Writes paper/raw/aud_split_facts.json and prints everything.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, already_done
if already_done(f'{PAPER_RAW}/aud_split_facts.json'): _sys.exit(0)

import sys, json
sys.path.insert(0, SRC)
import numpy as np, pandas as pd
import data as D

out = {}
rule = pd.DataFrame([(n,) + D.cut_points(n) for n in range(1, 31)], columns=['n', 'train', 'val'])
rule['test'] = rule.n - rule.train - rule.val
print('Allocation rule (src/data.py cut_points, TRAIN_FRAC=0.8, VAL_FRAC=0.1, Python round = round-half-to-even):')
print(rule.to_string(index=False))
out['rule'] = rule.to_dict('records')
first_val = int(rule[rule.val > 0].n.min()); out['first_n_with_val'] = first_val
print('smallest n with a validation item:', first_val, '; n values with val==0 above it:', rule[(rule.n > first_val) & (rule.val == 0)].n.tolist())

d = D.load()
prof = d.groupby(['market', 'category', 'u'], sort=False).size().rename('n').reset_index()
N = len(prof); out['n_profiles'] = N
bins = prof.n.clip(upper=8)
share = bins.value_counts().sort_index() / N
out['share_by_n'] = {('8+' if k == 8 else str(k)): float(v) for k, v in share.items()}
out['count_by_n'] = {('8+' if k == 8 else str(k)): int(v) for k, v in bins.value_counts().sort_index().items()}
print('\n(context, user) profiles:', N)
print(pd.DataFrame({'count': bins.value_counts().sort_index(), 'share': share.round(4)}).rename(index={8: '8+'}).to_string())
prof['val'] = [D.cut_points(int(n))[1] for n in prof.n]
out['share_val_eligible'] = float((prof.val > 0).mean())
print('share of profiles with >=1 validation item:', round(out['share_val_eligible'], 4))
# per context: users with a validation item
ctx = prof.assign(v=prof.val > 0).groupby(['market', 'category']).v.agg(['sum', 'mean']).reset_index()
nov = ctx[ctx['sum'] == 0]
out['contexts_without_val_users'] = [f'{m} {c}' for m, c in zip(nov.market, nov.category)]
print('contexts with no validation users:', out['contexts_without_val_users'])

# consistency of stored split sizes with the rule
for split in ('random', 'temporal'):
    cnt = d.groupby(['market', 'category', 'u'], sort=False)[f'split_{split}'].value_counts().unstack(fill_value=0)
    cnt = cnt.reindex(columns=['train', 'val', 'test'], fill_value=0)
    n = cnt.sum(axis=1).values
    exp = np.array([D.cut_points(int(x)) for x in n])
    ok = (cnt.train.values == exp[:, 0]) & (cnt.val.values == exp[:, 1])
    out[f'stored_matches_rule_{split}'] = bool(ok.all())
    print(f'stored {split} split sizes match cut_points for all profiles:', bool(ok.all()))

# date ties in the temporal split
d = d.reset_index(drop=True)
d['pos'] = np.arange(len(d))
rank = {'train': 0, 'val': 1, 'test': 2}
d['blk'] = d.split_temporal.map(rank).astype(int)
d = d.sort_values(['market', 'category', 'u', 'date', 'pos'], kind='stable')
# check: within a user, temporal blocks are non-decreasing in (date, pos)
g = d.groupby(['market', 'category', 'u'], sort=False)
mono = (g.blk.diff().fillna(0) >= 0).all()
out['temporal_blocks_monotone_in_date'] = bool(mono)
print('temporal train<val<test ordered by (date, row position) for every user:', bool(mono))
prev_date = g.date.shift(); prev_blk = g.blk.shift()
boundary = (d.blk != prev_blk) & prev_blk.notna()
tie_b = boundary & (d.date == prev_date)
anytie = (d.date == prev_date) & prev_date.notna()
d['tie_b'] = tie_b; d['anytie'] = anytie
pu = d.groupby(['market', 'category', 'u'], sort=False).agg(n=('blk', 'size'), tie_b=('tie_b', 'any'), anytie=('anytie', 'any'))
# boundary type split
d['btype'] = np.where(tie_b, prev_blk.fillna(-1).astype(int).astype(str) + '>' + d.blk.astype(str), '')
out['share_profiles_any_date_tie'] = float(pu.anytie.mean())
out['share_profiles_boundary_tie'] = float(pu.tie_b.mean())
out['share_profiles_boundary_tie_by_n'] = {('8+' if k == 8 else str(k)): float(v) for k, v in pu.groupby(pu.n.clip(upper=8)).tie_b.mean().items()}
bt = d[d.btype != ''].groupby(['market', 'category', 'u']).btype.agg(lambda s: tuple(sorted(set(s))))
out['boundary_tie_types'] = {'|'.join(k): int(v) for k, v in bt.value_counts().items()}
out['share_interactions_tied_with_previous'] = float(anytie.mean())
print('share of profiles with any repeated date:', round(out['share_profiles_any_date_tie'], 4))
print('share of profiles with a date tie at a temporal block boundary:', round(out['share_profiles_boundary_tie'], 4))
print('by n:', {k: round(v, 4) for k, v in out['share_profiles_boundary_tie_by_n'].items()})
print('boundary types (0=train,1=val,2=test; counts of profiles):', out['boundary_tie_types'])
# per-context share of boundary ties (max/median)
pc = pu.groupby(level=[0, 1]).tie_b.mean()
out['boundary_tie_share_per_context'] = {'median': float(pc.median()), 'min': float(pc.min()), 'max': float(pc.max()),
                                         'argmax': ' '.join(pc.idxmax())}
print('per-context share with boundary tie: median %.4f, min %.4f, max %.4f (%s)' % (pc.median(), pc.min(), pc.max(), ' '.join(pc.idxmax())))
json.dump(out, open(f'{PAPER_RAW}/aud_split_facts.json', 'w', encoding='utf-8', newline='\n'), indent=1)
