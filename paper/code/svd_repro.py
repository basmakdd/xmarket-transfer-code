"""Re-runs PureSVD local-only (k=0) units of the core grid in the current software stack and compares NDCG@10 with
paper/raw/df.pkl (results/full_random.jsonl). Contexts: the 4 smallest catalogs (dense eigendecomposition route) and the
2 smallest catalogs above models.DENSE_GRAM_MAX_ITEMS = 1500 (ARPACK route). Same call as src/runner.py
(E.fit_and_select with the target's capacity and numerical route). Writes paper/raw/aud_svd_repro.csv."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, already_done
if already_done(f'{PAPER_RAW}/aud_svd_repro.csv'): _sys.exit(0)
import sys
sys.path.insert(0, SRC)
import numpy as np, pandas as pd, scipy, numpy
import data as D, experiment as E, models as Mod

df = pd.read_pickle(f'{PAPER_RAW}/df.pkl')
k0 = df[(df.model == 'PureSVD') & (df.k == 0)].set_index('t')
c = pd.read_parquet(f'{RESULTS}/contexts.parquet'); c['t'] = c.market + ' ' + c.category
c = c[c.t.isin(k0.index)].sort_values('items')
pick = c[c['items'] <= Mod.DENSE_GRAM_MAX_ITEMS].t.head(4).tolist() + c[c['items'] > Mod.DENSE_GRAM_MAX_ITEMS].t.head(2).tolist()
blocks = E.load_blocks(D.load(), 'random')
rows = []
for t in pick:
    key = tuple(t.split()); s = E.TargetSession(key, blocks[key]); cap, dense = E.target_capacity(s, blocks[key])
    res, _, err = E.fit_and_select('PureSVD', s, [(blocks[key].train_u, blocks[key].train_i)], capacity=cap, dense=dense)
    old = float(k0.loc[t, 'ndcg@10'])
    rows.append(dict(t=t, items=int(c.set_index('t').loc[t, 'items']), route='dense' if dense else 'ARPACK', old=old, new=res['ndcg@10'],
                     diff=res['ndcg@10'] - old, params_old=k0.loc[t, 'params'], params_new=res['params']))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{PAPER_RAW}/aud_svd_repro.csv', index=False, lineterminator='\n')
print(R.to_string()); print('numpy', numpy.__version__, 'scipy', scipy.__version__)
try:
    numpy.show_config()
except Exception:
    pass
