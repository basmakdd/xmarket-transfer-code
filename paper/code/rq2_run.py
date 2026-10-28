"""RQ2 complete-data experiment: local / pool / weight / greedy arms.

Usage:  python rq2_run.py LIM MODEL SPLIT ARMS [MAXCOST]
  LIM    wall-clock budget in seconds (no new context is started after it)
  MODEL  MostPop | ItemKNN | EASE | UserKNN | Random
  SPLIT  random | temporal | calendar
  ARMS   full  = local, pool, weight (alpha grid), greedy
         lpw   = local, pool, weight (no greedy)
         lp    = local and pool only
         local = local only (chance control)
         grid  = every grid setting, local and pool, test scores (sensitivity runs; use ONLY=)
  env    XM_DATA_TAG (e.g. _seed1, _r4) selects the preprocessed variant (data + overlap file)
         ONLY="mk Category,mk Category" restricts the contexts; SKIP likewise excludes contexts.

Every arm is tuned exactly like mustadd.run_unit (regime F): hyperparameters chosen by
validation NDCG@10 over the model grid (first strictly-best setting wins), grid[0] when the
context has no validation users. `weights` are per-edge-list row weights (mustadd.mats).
  local : local training data only (alpha = 0)
  pool  : local + all same-category foreign training data, alpha = 1
  weight: alpha in {0, .1, .25, .5, 1} chosen jointly with the hyperparameters by validation
          NDCG@10 (alpha = 0 is the local arm itself). No validation users -> alpha = 1 (flagged).
  greedy: forward selection of donor countries at the local arm's hyperparameters (fixed),
          adding the donor that most improves validation NDCG@10 while it strictly improves;
          then the grid is retuned on the selected set. No validation users -> all donors.
Output: paper/raw/rq2_<MODEL>_<SPLIT><TAG>[_grid].jsonl, one record per context;
per-user NDCG@10 of local and pool (random split, untagged) in paper/raw/rq2_peruser/.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time
sys.path.insert(0, os.path.expanduser(SRC)); sys.path.insert(0, os.path.expanduser(CODE))
import pandas as pd, numpy as np
import data as D, experiment as E
import mustadd as MA

MA.GRIDS['MostPop'] = [{}]
MA.GRIDS['Random'] = [{}]
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)

LIM = float(sys.argv[1]); MODEL = sys.argv[2]; SPLIT = sys.argv[3]; ARMS = sys.argv[4]
MAXCOST = float(sys.argv[5]) if len(sys.argv) > 5 else 1e18
TAG = D.DATA_TAG
GRID = MA.GRIDS[MODEL]
out = os.path.expanduser(f'{PAPER_RAW}/rq2_{MODEL}_{SPLIT}{TAG}' + ('_grid' if ARMS == 'grid' else '') + '.jsonl')
PU = os.path.join(PAPER_RAW, 'rq2_peruser'); os.makedirs(PU, exist_ok=True)
SAVE_PU = (SPLIT == 'random' and TAG == '' and ARMS in ('full', 'lpw', 'lp'))
T0 = time.time()
done = set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'): done.add(json.loads(l)['t'])

ov = 'overlap_calendar.parquet' if SPLIT == 'calendar' else f'overlap_{SPLIT}{TAG}.parquet'
o = pd.read_parquet(os.path.join(REPO, 'results', ov)); o = o[o.channel == 'XM']
c = pd.read_parquet(os.path.join(REPO, 'results', 'contexts.parquet'))
cost = {(m, cc): int(u) * int(i) for m, cc, u, i in zip(c['market'], c['category'], c['users'], c['items'])}
data = D.load_for(SPLIT); blocks = E.load_blocks(data, SPLIT); del data
targets = sorted(set(zip(o.target_market, o.target_category)), key=lambda t: (cost.get(t, 0), t))
targets = [t for t in targets if cost.get(t, 0) <= MAXCOST and t in blocks]
if os.environ.get('ONLY'): targets = [t for t in targets if ' '.join(t) in os.environ['ONLY'].split(',')]
if os.environ.get('SKIP'): targets = [t for t in targets if ' '.join(t) not in os.environ['SKIP'].split(',')]
print(f'{MODEL} {SPLIT}{TAG} {ARMS}: {len(targets)} contexts, {len(done)} done', flush=True)


def fitscore(sess, el, params, weights=None):
    """Same fit as mustadd.run_unit's inner fitscore (regime F), returning per-user arrays too."""
    keep = np.ones(len(el[0][0]), bool)
    R, Rs, cols, rows = MA.mats(el, sess.catalog, sess.eval_users, keep, weights)
    if MODEL == 'Random':
        import models as Mod
        m = Mod.RandomRec(R, cols, user_ids=E.row_to_user_id(rows, sess.eval_users))
    else:
        m = MA.build(MODEL, R, cols)(params); MA.set_prof(m, MODEL, R)
    res, pu = E.evaluate(m, sess, rows, candidate_sizes=(), seed=0)
    return res, pu


def tune(vs, el, weights=None):
    """Validation NDCG@10 per grid setting (None where the fit failed)."""
    scores = []
    for p in GRID:
        try: scores.append(fitscore(vs, el, p, weights)[0]['ndcg@10'])
        except MemoryError: scores.append(None)
    return scores


def best(scores):
    """Index of the first strictly-best score (run_unit's rule); 0 when nothing is valid."""
    bi, bs = None, -1
    for k, sc in enumerate(scores):
        if sc is not None and sc > bs: bi, bs = k, sc
    return 0 if bi is None else bi


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
    rec = {'t': t, 'split': SPLIT, 'tag': TAG, 'model': MODEL, 'arms': ARMS, 'donors': [d[0] for d in donors],
           'tuned': tuned, 'n_val_users': int(len(vs.eval_users)), 'n_eval_users': int(len(s.eval_users)),
           'n_items': int(len(s.catalog)), 'n_local_train': int(len(b.train_u)),
           'n_foreign_train': int(sum(len(dl[d][0]) for d in donors))}

    if ARMS == 'grid':
        rows = []
        for k, p in enumerate(GRID):
            for arm, el in (('local', loc), ('pool', pool_el)):
                try: r, _ = fitscore(s, el, p); rows.append({'arm': arm, 'k': k, 'params': p, 'ndcg': r['ndcg@10'], 'recall': r['recall@10']})
                except MemoryError: rows.append({'arm': arm, 'k': k, 'params': p, 'ndcg': None, 'recall': None})
        rec['grid'] = rows
        if MODEL in ('ItemKNN', 'EASE', 'UserKNN'):   # popularity fallback for the local model: rank by local popularity
            import models as Mod
            R, cols, rows_ = E.build_matrix(loc, s.catalog, s.eval_users)
            r, _ = E.evaluate(Mod.MostPop(R, cols), s, rows_, candidate_sizes=())
            rec['local_popularity_fallback'] = r['ndcg@10']
        rec['sec'] = round(time.time() - t1, 1)
        open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n'); print(t, rec['sec'], flush=True)
        continue

    # ---- validation scores: alpha grid (local = alpha 0, pool = alpha 1)
    alphas = {'full': ALPHAS, 'lpw': ALPHAS, 'lp': (0.0, 1.0), 'local': (0.0,)}[ARMS]
    val = {}
    if tuned and len(GRID) > 1:
        for a in alphas:
            if a == 0: val[a] = tune(vs, loc)
            elif a == 1: val[a] = tune(vs, pool_el)
            else: val[a] = tune(vs, pool_el, [1.0] + [a] * len(donors))
    elif tuned and ARMS in ('full', 'lpw'):   # single-setting models still need alpha validation scores
        for a in alphas:
            val[a] = tune(vs, loc if a == 0 else pool_el, None if a in (0, 1) else [1.0] + [a] * len(donors))
    rec['val'] = {str(a): v for a, v in val.items()}

    def test(el, k, weights=None):
        r, pu = fitscore(s, el, GRID[k], weights)
        return {'ndcg': r['ndcg@10'], 'recall': r['recall@10'], 'k': k, 'params': GRID[k]}, pu['ndcg@10']

    kl = best(val[0.0]) if 0.0 in val else 0
    rec['local'], pu_l = test(loc, kl)
    if ARMS != 'local':
        kp = best(val[1.0]) if 1.0 in val else 0
        rec['pool'], pu_p = test(pool_el, kp)
        if SAVE_PU:
            cnt = pd.Series(b.train_u).value_counts().reindex(s.eval_users).fillna(0).to_numpy(np.int32)
            np.savez_compressed(os.path.join(PU, f'{MODEL}_{target[0]}_{target[1]}.npz'),
                                users=s.eval_users, hist=cnt, local=pu_l.astype(np.float32), pool=pu_p.astype(np.float32))
    if ARMS in ('full', 'lpw'):
        # weight arm: joint (alpha, setting) selection; ties -> smaller alpha, then earlier setting
        if tuned:
            ba, bk, bs = 0.0, kl, -1
            for a in ALPHAS:
                for k, sc in enumerate(val[a]):
                    if sc is not None and sc > bs: ba, bk, bs = a, k, sc
        else:
            ba, bk = 1.0, 0
        if ba == 0: w = dict(rec['local'])
        elif ba == 1 and bk == kp: w = dict(rec['pool'])
        else: w, _ = test(pool_el, bk, None if ba == 1 else [1.0] + [ba] * len(donors))
        w['alpha'] = ba; w['flag_untuned'] = not tuned; rec['weight'] = w
    if ARMS == 'full':
        # greedy forward selection at the local arm's setting, then retune on the selected set
        steps = []
        if tuned:
            cur = val[0.0][kl]; sel = []
            while len(sel) < len(donors):
                cand = {}
                for d in donors:
                    if d in sel: continue
                    try: cand[d] = fitscore(vs, loc + [dl[x] for x in sel + [d]], GRID[kl])[0]['ndcg@10']
                    except MemoryError: pass
                if not cand: break
                d_best = max(sorted(cand), key=lambda d: cand[d])   # ties -> alphabetical first
                steps.append({'cand': {x[0]: v for x, v in cand.items()}, 'cur': cur})
                if cand[d_best] > cur: sel.append(d_best); cur = cand[d_best]
                else: break
        else:
            sel = list(donors)
        sel = sorted(sel)
        if not sel: g = dict(rec['local'])
        elif sel == donors: g = dict(rec['pool'])
        else:
            gel = loc + [dl[d] for d in sel]
            vk = best(tune(vs, gel)) if (tuned and len(GRID) > 1) else 0
            g, _ = test(gel, vk)
        g['selected'] = [d[0] for d in sel]; g['steps'] = steps; g['flag_untuned'] = not tuned
        rec['greedy'] = g
    rec['sec'] = round(time.time() - t1, 1)
    open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')
    print(t, rec['sec'], {k: round(rec[k]['ndcg'], 4) for k in ('local', 'pool', 'weight', 'greedy') if k in rec}, flush=True)
print('ALL DONE', flush=True)
