"""Experiment-by-context accounting for the RQ2 runs: which eligible contexts are missing from each
(variant, model, arm) and why. Eligible = target contexts with >= 1 same-category foreign donor in the
variant's overlap file. Writes paper/results/rq2_accounting_detail.csv."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, PAPER_RAW
import os, json, glob
import pandas as pd
rows = []
ctx = {'': pd.read_parquet(f'{REPO}/results/contexts.parquet')}
allctx = set(ctx[''].market + ' ' + ctx[''].category)
for f in sorted(glob.glob(f'{PAPER_RAW}/rq2_*.jsonl')):
    if f.endswith('_grid.jsonl'): continue
    recs = [json.loads(l) for l in open(f, encoding='utf-8')]
    r0 = recs[0]; split, tag, model = r0['split'], r0['tag'], r0['model']
    ov = 'overlap_calendar.parquet' if split == 'calendar' else f'overlap_{split}{tag}.parquet'
    o = pd.read_parquet(f'{REPO}/results/{ov}'); o = o[o.channel == 'XM']
    elig = set(o.target_market + ' ' + o.target_category)
    got = {r['t']: r for r in recs}
    items = {r['t']: r['n_items'] for r in recs}
    for arm in ('local', 'pool', 'weight', 'greedy'):
        have = {t for t, r in got.items() if arm in r}
        if not have and arm != 'local': continue
        miss = sorted(elig - have)
        rows.append(dict(split=split, tag=tag, model=model, arm=arm, eligible=len(elig), included=len(have & elig), missing=';'.join(miss)))
D = pd.DataFrame(rows); D.to_csv(f'{REPO}/paper/results/rq2_accounting_detail.csv', index=False, lineterminator='\n')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(D.to_string())
c = ctx['']; print('contexts in data:', len(c), '; without any same-category donor (random overlap):',
      sorted(allctx - set(pd.read_parquet(f'{REPO}/results/overlap_random.parquet').query("channel=='XM'").pipe(lambda o: o.target_market + ' ' + o.target_category))))
print('catalogs > 4000 items (EASE skipped):', sorted((c.market + ' ' + c.category)[c['items'] > 4000]))
r4 = pd.read_parquet(f'{REPO}/results/contexts_r4.parquet'); print('r4 contexts in data:', len(r4))
