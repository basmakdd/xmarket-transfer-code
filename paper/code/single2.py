"""RQ4 single-donor gains (replaces legacy_v3 run_single.py / g.py / h.py).

For EVERY same-category foreign (XM) target-donor pair of the random split
(results/overlap_random.parquet, channel == 'XM'), ItemKNN is fitted on
  arm 'local'  : the target's own training interactions only, and
  arm 'donor'  : the target's training interactions + ALL training interactions of that ONE donor,
each re-tuned on the target's validation users (mustadd.run_unit, regime 'F' = complete data;
contexts without validation users use grid[0] in both arms). The local arm is fitted once per target
and shared by all of that target's donors. UK Books is INCLUDED (largest context: run last).

Usage: single2.py LIM_SECONDS W NW   (worker W of NW; env ONLY="uk Books" restricts targets;
        env MAXUNITS=n stops after n new units, used to time one unit)
Output: paper/raw/aud_single.jsonl, one record per (target, arm[, donor]); resumable.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time
sys.path.insert(0, SRC); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import data as D, experiment as E
import mustadd as MA

MODEL = 'ItemKNN'; SPLIT = 'random'
LIM = float(sys.argv[1]); W = int(sys.argv[2]); NW = int(sys.argv[3])
MAXUNITS = int(os.environ.get('MAXUNITS', '0')) or None
out = f'{PAPER_RAW}/aud_single.jsonl'
T0 = time.time(); done = set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'):
        r = json.loads(l); done.add((r['t'], r['donor_market']))
o = pd.read_parquet(f'{RESULTS}/overlap_{SPLIT}.parquet'); o = o[o.channel == 'XM']
c = pd.read_parquet(f'{RESULTS}/contexts.parquet')
cost = {(m, cc): int(u) * int(i) for m, cc, u, i in zip(c['market'], c['category'], c['users'], c['items'])}
targets = sorted(set(zip(o.target_market, o.target_category)), key=lambda t: (cost.get(t, 0), t))[W::NW]
if os.environ.get('ONLY'): targets = [t for t in targets if ' '.join(t) in os.environ['ONLY'].split(',')]
data = D.load_for(SPLIT); blocks = E.load_blocks(data, SPLIT)
print('loaded', round(time.time() - T0, 1), 'targets', len(targets), flush=True)
n_new = 0
for target in targets:
    t = ' '.join(target); b = blocks[target]
    donors = sorted(tuple(x) for x in o[(o.target_market == target[0]) & (o.target_category == target[1])][['donor_market', 'donor_category']].values)
    plan = [None] + donors
    s = vs = None
    for donor in plan:
        dm = '' if donor is None else donor[0]
        if (t, dm) in done: continue
        if time.time() - T0 > LIM: print('deadline', flush=True); sys.exit()
        if MAXUNITS is not None and n_new >= MAXUNITS: print('maxunits', flush=True); sys.exit()
        if s is None: s = E.TargetSession(target, b); vs = s.validation_view()
        el = [(b.train_u, b.train_i)] + ([] if donor is None else [(blocks[donor].train_u, blocks[donor].train_i)])
        keep = np.ones(len(b.train_u), bool)
        t1 = time.time()
        try:
            sc, bp = MA.run_unit(MODEL, s, vs, el, keep, 'F'); err = None
        except MemoryError:
            sc, bp, err = None, None, 'memory'
        rec = {'t': t, 'target_market': target[0], 'category': target[1], 'arm': 'local' if donor is None else 'donor',
               'donor_market': dm, 'split': SPLIT, 'model': MODEL, 'ndcg': sc, 'params': bp, 'err': err,
               'tuned': bool(len(vs.eval_users)), 'n_val_users': int(len(vs.eval_users)),
               'n_eval_users': int(len(s.eval_users)), 'n_dom': int(len(b.train_u)),
               'n_for': 0 if donor is None else int(len(blocks[donor].train_u)), 'sec': round(time.time() - t1, 2)}
        open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n'); n_new += 1
        print(t, dm or 'LOCAL', sc, bp, rec['sec'], flush=True)
print('ALL DONE', flush=True)
