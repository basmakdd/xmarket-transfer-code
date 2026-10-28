"""Time single fitted units (fit + test evaluation) for UK Books, to decide its inclusion (RQ2)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import sys, os, time, json
sys.path.insert(0, SRC); sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import data as D, experiment as E, mustadd as MA
MODELS = sys.argv[1].split(','); SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'random'
T0 = time.time(); blocks = E.load_blocks(D.load_for(SPLIT), SPLIT); target = ('uk', 'Books'); b = blocks[target]
s = E.TargetSession(target, b); vs = s.validation_view(); print('session', round(time.time() - T0, 1), len(s.eval_users), len(vs.eval_users), flush=True)
o = pd.read_parquet(f'{REPO}/results/overlap_{SPLIT}.parquet'); o = o[(o.channel == 'XM') & (o.target_market == 'uk') & (o.target_category == 'Books')]
donors = [tuple(x) for x in o[['donor_market', 'donor_category']].values]
for model in MODELS:
    for arm, el in (('local', [(b.train_u, b.train_i)]), ('pool', [(b.train_u, b.train_i)] + [(blocks[d].train_u, blocks[d].train_i) for d in donors])):
        t1 = time.time(); keep = np.ones(len(b.train_u), bool)
        R, Rs, cols, rows = MA.mats(el, s.catalog, s.eval_users, keep)
        m = MA.build(model, R, cols)(MA.GRIDS[model][0]); MA.set_prof(m, model, R)
        r = E.evaluate(m, s, rows, candidate_sizes=())[0]
        rec = {'model': model, 'split': SPLIT, 'arm': arm, 'params': MA.GRIDS[model][0], 'ndcg': r['ndcg@10'], 'sec': round(time.time() - t1, 1)}
        print(json.dumps(rec), flush=True)
        open(f'{REPO}/paper/logs/rq2_timeuk.jsonl', 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')
