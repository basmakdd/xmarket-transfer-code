"""Scarcity runs on a data variant, reusing scarcity2.run_context unchanged (only the unit plan is restricted).

Run through the wrapper so that overlap/contexts files are the tagged ones:
  XM_DATA_TAG=_nv PAPER_RAW=paper/raw/sens_nv python paper/code/sens_wrap.py paper/code/sens_scarcity.py JOB LIMIT_S
JOB = itemknn : ItemKNN, random split, all XM targets: F (foreign 0/1) + arms P/R/B at f=0.10, seed 0, foreign 0/1, no fallback.
JOB = userknn : UserKNN, random split, the contexts named in env SC_CONTEXTS (comma-separated 'mk Category'):
                F (foreign 0) and arm P at f=0.50, seeds 0,1,2, foreign 0 (the monotonicity check).
Output: PAPER_RAW/sc_<MODEL>_random.jsonl (scarcity2 record format).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, time
import scarcity2 as S2

JOB = sys.argv[1]; LIM = float(sys.argv[2])
if JOB == 'itemknn':
    MODEL = 'ItemKNN'

    def plan(model, split, seeds):
        u = []
        for foreign in (0, 1):
            u.append(('F', 0, 1.0, foreign, 0))
            for arm in S2.ARMS:
                u.append((arm, 0, 0.10, foreign, 0))
        return u
    targets = None
else:
    MODEL = 'UserKNN'

    def plan(model, split, seeds):
        return [('F', 0, 1.0, 0, 0)] + [('P', s, 0.50, 0, 0) for s in (0, 1, 2)]
    targets = os.environ['SC_CONTEXTS'].split(',')

S2.plan_units = plan
T0 = time.time()
cost = S2.context_costs('random'); tg = S2.targets_of('random')
work = sorted(t for t in tg if targets is None or t in targets)
work.sort(key=lambda t: cost[t])
if targets is not None:
    missing = [t for t in targets if t not in tg]
    print('requested contexts without XM donors / absent in this variant:', missing, flush=True)
for t in work:
    t1 = time.time()
    st = S2.run_context(MODEL, 'random', t, tg[t], (0,), T0, LIM)
    print(f'{MODEL} {t} cost={cost[t]} -> {st} ({time.time() - t1:.1f}s)', flush=True)
    if st == 'deadline':
        print('deadline', flush=True); sys.exit()
print('ALL DONE', flush=True)
