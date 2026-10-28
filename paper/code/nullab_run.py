"""Model-level null ablation for RQ2: local / pool / pool_null (curveball-randomised donor data).

Usage:  python nullab_run.py LIM MODEL
  LIM    wall-clock budget in seconds (no new context is started after it)
  MODEL  MostPop | ItemKNN | EASE
  env    XM_DATA_TAG (e.g. _nvc) selects the preprocessed variant; this script reads the tagged
         data, overlap_random{TAG}.parquet and contexts{TAG}.parquet itself (no wrapper needed).
         ONLY / SKIP = "mk Category,mk Category" restrict / exclude contexts (as in rq2_run.py).

Random split, donors = the XM donor list of overlap_random{TAG}.parquet (same as rq2_run.py).
Arms, each tuned exactly like rq2_run.py (regime F; validation NDCG@10 over the model grid, first
strictly-best setting wins; grid[0] when the context has no validation users):
  local     : local training data only
  pool      : local + all same-category foreign training data (alpha = 1)
  pool_null : as pool, but each donor's training user x product matrix (all of that donor's training
              interactions, binarised) is replaced by a curveball randomisation of it
              (rq1_agree.curveball: >= 5 x (number of donor users) trades), which keeps every product's
              interaction count and every user's activity within that donor and disrupts which
              products co-occur in a user's history. 3 independent draws per context
              (rng = default_rng(util.stable_seed('nullab', t, draw)), donors processed in sorted order);
              every draw is retuned separately; the context score is the mean of the 3 draws' test scores.
Output: paper/raw/nullab_<MODEL>_random<TAG>.jsonl, one record per context; finished contexts are skipped.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time
sys.path.insert(0, os.path.expanduser(SRC)); sys.path.insert(0, os.path.expanduser(CODE))
import pandas as pd, numpy as np
import data as D, experiment as E
import mustadd as MA
from rq1_agree import curveball, TRADES_PER_USER
from util import stable_seed

MA.GRIDS['MostPop'] = [{}]
N_DRAWS = 3
SPLIT = 'random'

LIM = float(sys.argv[1]); MODEL = sys.argv[2]
TAG = D.DATA_TAG
GRID = MA.GRIDS[MODEL]
out = os.path.join(PAPER_RAW, f'nullab_{MODEL}_{SPLIT}{TAG}.jsonl')
T0 = time.time()
done = set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'): done.add(json.loads(l)['t'])

o = pd.read_parquet(os.path.join(RESULTS, f'overlap_{SPLIT}{TAG}.parquet')); o = o[o.channel == 'XM']
c = pd.read_parquet(os.path.join(RESULTS, f'contexts{TAG}.parquet'))
cost = {(m, cc): int(u) * int(i) for m, cc, u, i in zip(c['market'], c['category'], c['users'], c['items'])}
data = D.load_for(SPLIT); blocks = E.load_blocks(data, SPLIT); del data
targets = sorted(set(zip(o.target_market, o.target_category)), key=lambda t: (cost.get(t, 0), t))
targets = [t for t in targets if t in blocks]
if os.environ.get('ONLY'): targets = [t for t in targets if ' '.join(t) in os.environ['ONLY'].split(',')]
if os.environ.get('SKIP'): targets = [t for t in targets if ' '.join(t) not in os.environ['SKIP'].split(',')]
print(f'nullab {MODEL} {SPLIT}{TAG}: {len(targets)} contexts, {len(done)} done', flush=True)


def fitscore(sess, el, params):
    """rq2_run.fitscore (regime F, no weights)."""
    keep = np.ones(len(el[0][0]), bool)
    R, Rs, cols, rows = MA.mats(el, sess.catalog, sess.eval_users, keep, None)
    m = MA.build(MODEL, R, cols)(params); MA.set_prof(m, MODEL, R)
    res, pu = E.evaluate(m, sess, rows, candidate_sizes=(), seed=0)
    return res, pu


def tune(vs, el):
    scores = []
    for p in GRID:
        try: scores.append(fitscore(vs, el, p)[0]['ndcg@10'])
        except MemoryError: scores.append(None)
    return scores


def best(scores):
    bi, bs = None, -1
    for k, sc in enumerate(scores):
        if sc is not None and sc > bs: bi, bs = k, sc
    return 0 if bi is None else bi


def null_donor(u, i, rng):
    """Curveball randomisation of one donor's binarised training matrix. Returns (u, i) edge arrays."""
    df = pd.DataFrame({'u': u, 'i': i}).drop_duplicates()
    uc, uval = pd.factorize(df.u.values); ic, ival = pd.factorize(df.i.values)
    uc = uc.astype(np.int64); ic = ic.astype(np.int64)
    nic, trades = curveball(uc, ic, rng)
    # checks (user activity is kept by construction: edges keep their user): product counts, binary (no duplicate user-product pairs)
    ok_i = np.array_equal(np.bincount(ic, minlength=len(ival)), np.bincount(nic, minlength=len(ival)))
    dup = int(pd.Series(uc * len(ival) + nic).duplicated().sum())
    changed = float(np.mean(nic != ic)) if len(ic) else 0.0
    # share of the original user-product pairs that survive the randomisation
    orig = set((uc * len(ival) + ic).tolist()); new = uc * len(ival) + nic
    kept = float(np.mean([x in orig for x in new.tolist()])) if len(new) else 1.0
    info = {'edges': int(len(ic)), 'users': int(len(uval)), 'trades': int(trades), 'ok_counts': bool(ok_i),
            'dups': dup, 'kept_pairs': round(kept, 4), 'slot_changed': round(changed, 4)}
    return uval[uc].astype(np.int32), ival[nic].astype(np.int32), info


for target in targets:
    t = ' '.join(target); b = blocks[target]
    if t in done: continue
    if MODEL == 'EASE' and len(b.catalog) > 4000: continue
    if time.time() - T0 > LIM: print('deadline', flush=True); sys.exit()
    t1 = time.time()
    donors = sorted(tuple(x) for x in o[(o.target_market == target[0]) & (o.target_category == target[1])][['donor_market', 'donor_category']].values)
    s = E.TargetSession(target, b); vs = s.validation_view()
    tuned = bool(len(vs.eval_users))
    loc = [(b.train_u, b.train_i)]
    dl = {d: (blocks[d].train_u, blocks[d].train_i) for d in donors}
    pool_el = loc + [dl[d] for d in donors]
    rec = {'t': t, 'split': SPLIT, 'tag': TAG, 'model': MODEL, 'donors': [d[0] for d in donors],
           'tuned': tuned, 'n_val_users': int(len(vs.eval_users)), 'n_eval_users': int(len(s.eval_users)),
           'n_items': int(len(s.catalog)), 'n_local_train': int(len(b.train_u)),
           'n_foreign_train': int(sum(len(dl[d][0]) for d in donors)), 'trades_per_user': TRADES_PER_USER}
    do_tune = tuned and len(GRID) > 1

    def arm(el):
        val = tune(vs, el) if do_tune else None
        k = best(val) if do_tune else 0
        r, _ = fitscore(s, el, GRID[k])
        return {'ndcg': r['ndcg@10'], 'recall': r['recall@10'], 'k': k, 'params': GRID[k], 'val': val}

    rec['local'] = arm(loc)
    rec['pool'] = arm(pool_el)
    draws = []
    for draw in range(N_DRAWS):
        rng = np.random.default_rng(stable_seed('nullab', t, draw))
        nel, infos = [], []
        for d in donors:
            nu, ni, info = null_donor(dl[d][0], dl[d][1], rng)
            nel.append((nu, ni)); infos.append(dict(info, donor=d[0]))
        a = arm(loc + nel); a['null_info'] = infos
        draws.append(a)
    rec['pool_null_draws'] = draws
    rec['pool_null'] = {'ndcg': float(np.mean([x['ndcg'] for x in draws])),
                        'recall': float(np.mean([x['recall'] for x in draws])),
                        'ndcg_sd': float(np.std([x['ndcg'] for x in draws], ddof=1))}
    rec['sec'] = round(time.time() - t1, 1)
    open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')
    print(t, rec['sec'], {k: round(rec[k]['ndcg'], 4) for k in ('local', 'pool', 'pool_null')}, flush=True)
print('ALL DONE', flush=True)
