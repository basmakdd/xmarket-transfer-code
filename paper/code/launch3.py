"""Product-launch experiment, redesigned (replaces legacy_v3/paper_code/launch2.py).

Per-user splits (`random`, `temporal`; retrospective simulation):
  * Eligible products are chosen WITHOUT looking at the test set: products in the local catalog that have local
    TRAINING interactions and appear in the training data of at least one same-category foreign donor.
  * Launched = random 30% of eligible (at least 3); 3 launch sets per context (stable_seed('launch3', t, seed)).
    Contexts with < 5 eligible products are skipped.
  * All local TRAIN and VALIDATION interactions with launched products are hidden: they are removed from model
    fitting and from the users' scoring profiles (and from the masks). The test set is unchanged.
Calendar split (`calendar`; prospective): launched products = launch candidates from data/launch_calendar.parquet
  (products outside the pre-cutoff local catalog, i.e. not retained by the pre-cutoff core filter, with foreign
  pre-cutoff interactions and a local interaction after T_val; see src/calendar_split.py) that occur in some
  donor's calendar training data. Catalog = local catalog + launched products. Training = local train
  (before T_val) [+ donors' calendar train, before the common category cutoff T_val].

Outcomes (test set): (a) launched-item outcome: users whose test interactions include a launched product, truth =
those launched test products only; (b) overall outcome: all test users (capped at 20,000, as everywhere) with their
full test truth. NDCG@10 and Recall@10.
Tuning settings (both run, both recorded):
  zf  strict zero local feedback: tune on validation interactions with NON-launched products (users whose remaining
      validation truth is empty are dropped); needs >= 10 validation users, else grid[0].
  vf  validation feedback (the procedure of the v3 study): tune on the hidden validation interactions with launched products;
      needs >= 10 validation users, else grid[0].
Arms: foreign = 0 (local only) / 1 (+ all same-category foreign training data, equal weight).

Skip rules (written to l3_skip_*.jsonl with the reason):
  * a launch set with fewer than MIN_TEST_USERS = 10 launched-item test users (users whose TEST interactions include
    a launched product) is skipped; a context enters a table only through its non-skipped launch sets, so a context
    whose launch sets all have < 10 such users is excluded. Inclusion therefore depends on the test-set counts
    (the number of launched-item test users), not only on training data; the rule is the same for every model and arm.
  * per-user splits: contexts with fewer than MIN_ELIG = 5 eligible products; calendar split: launch sets with fewer
    than MIN_LAUNCHED = 3 launched products; EASE: catalogs over models.EASE_MAX_ITEMS products; a unit running
    longer than UNIT_TIMEOUT (30 min) excludes the context for that model (POSIX only, see set_alarm below).

Usage: launch3.py LIMIT_SEC MODEL SPLIT [MAXCOST]   (resumable; ONLY='uk Books,...' restricts targets)
Outputs: paper/raw/l3_<MODEL>_<SPLIT>.jsonl (units), paper/raw/l3_skip_<MODEL>_<SPLIT>.jsonl (skips with reasons).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time, signal
sys.path.insert(0, os.path.expanduser(SRC)); sys.path.insert(0, os.path.expanduser(CODE))
import pandas as pd, numpy as np
import data as D, experiment as E, models as Mod
from util import stable_seed
from mustadd import UserKNN, GRIDS

GR = dict(GRIDS); GR['MostPop'] = [{}]; GR['Random'] = [{}]
MIN_ELIG = 5          # per-user splits: skip context below this many eligible products
LAUNCH_FRAC = 0.3     # share of eligible products launched
MIN_LAUNCHED = 3      # at least this many launched products (calendar: skip context below)
MIN_TEST_USERS = 10   # launched-item outcome: skip launch set / context below this many users
MIN_VAL_USERS = 10    # tuning needs this many validation users, else grid[0]
UNIT_TIMEOUT = 1800   # one unit (target x launch set x arm) > 30 min -> exclude the context for this model
SEEDS = (0, 1, 2)
EMPTY = np.array([], np.int32)

LIM = float(sys.argv[1]); MODEL = sys.argv[2]; SPLIT = sys.argv[3]
MAXCOST = float(sys.argv[4]) if len(sys.argv) > 4 else 1e30
out = os.path.expanduser(f'{PAPER_RAW}/l3_{MODEL}_{SPLIT}.jsonl')
skipf = os.path.expanduser(f'{PAPER_RAW}/l3_skip_{MODEL}_{SPLIT}.jsonl')
T0 = time.time()
done = set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'):
        r = json.loads(l); done.add((r['t'], r['seed'], r['foreign']))
skipped = {}
if os.path.exists(skipf):
    for l in open(skipf, encoding='utf-8'):
        r = json.loads(l); skipped[(r['t'], r['seed'])] = r


def skip(t, seed, reason, **kw):
    if (t, seed) in skipped: return
    rec = {'t': t, 'split': SPLIT, 'model': MODEL, 'seed': seed, 'reason': reason, **kw}
    skipped[(t, seed)] = rec
    open(skipf, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')


class Timeout(Exception): pass
def _alarm(signum, frame): raise Timeout()
# The per-unit timeout uses SIGALRM, which exists only on POSIX systems. On Windows the timeout is skipped:
# every unit runs to completion, so a context that hit the 30-min limit in the paper run is not excluded there.
HAS_ALARM = hasattr(signal, 'SIGALRM')
if HAS_ALARM: signal.signal(signal.SIGALRM, _alarm)
else: print(f'note: no SIGALRM on this platform; the {UNIT_TIMEOUT // 60}-min per-unit timeout is not applied', flush=True)
def set_alarm(sec):
    if HAS_ALARM: signal.alarm(sec)

o = pd.read_parquet(os.path.expanduser(f'{REPO}/results/overlap_{SPLIT}.parquet')); o = o[o.channel == 'XM']
if SPLIT == 'calendar':
    c = pd.read_parquet(os.path.expanduser(REPO + '/results/contexts_calendar.parquet'))
else:
    c = pd.read_parquet(os.path.expanduser(REPO + '/results/contexts.parquet'))
cost = {(m, cc): int(u) * int(i) for m, cc, u, i in zip(c['market'], c['category'], c['users'], c['items'])}
data = D.load_for(SPLIT); blocks = E.load_blocks(data, SPLIT)
if SPLIT == 'calendar':
    import calendar_split as CS
    lc = CS.load(launch=True)
    lcg = {k: g for k, g in lc.groupby(['market', 'category'])}
targets = sorted(set(zip(o.target_market, o.target_category)), key=lambda t: cost.get(t, 0))
targets = [t for t in targets if cost.get(t, 0) <= MAXCOST]
if _os.environ.get('ONLY'): targets = [t for t in targets if ' '.join(t) in _os.environ['ONLY'].split(',')]


def make(model, R, cols, rows, ids, p):
    """Same model construction as launch2.make (UserKNN from mustadd, Random keyed on global user ids)."""
    if model == 'UserKNN': return UserKNN(R, cols, **p)
    if model == 'Random': return Mod.RandomRec(R, cols, user_ids=ids)
    return Mod.build(model, R, cols, user_rows=rows, **p)


def score(m, sess, rows_of):
    res, _ = E.evaluate(m, sess, rows_of(sess), candidate_sizes=(), seed=0)
    return res


def run_unit(el, catalog, launch_s, all_s, views):
    """One fit set for a (target, launch set, arm). views: {'zf': session|None, 'vf': session|None}.
    Matrix built once over the union of all evaluation users (empty rows do not change any of the models)."""
    sess = [launch_s, all_s] + [v for v in views.values() if v is not None]
    U = np.unique(np.concatenate([s.eval_users for s in sess]))
    R, cols, rows_all = E.build_matrix(el, catalog, U)
    ids = E.row_to_user_id(rows_all, U)
    rows_of = lambda s: rows_all[np.searchsorted(U, s.eval_users)]
    grid = GR[MODEL]
    nval = {k: (0 if v is None else int(len(v.eval_users))) for k, v in views.items()}
    tune = {k: (len(grid) > 1 and nval[k] >= MIN_VAL_USERS) for k in views}
    best = {k: (-1.0, grid[0]) for k in views}
    if any(tune.values()):
        for p in grid:
            m = make(MODEL, R, cols, rows_all, ids, p)
            for k, v in views.items():
                if tune[k]:
                    sc = score(m, v, rows_of)['ndcg@10']
                    if sc > best[k][0]: best[k] = (sc, p)
            del m
    rec = {}
    cache = {}
    for k in views:
        bp = best[k][1]; key = json.dumps(bp, sort_keys=True)
        if key not in cache:
            m = make(MODEL, R, cols, rows_all, ids, bp)
            rl = score(m, launch_s, rows_of); ra = score(m, all_s, rows_of)
            cache[key] = (rl, ra); del m
        rl, ra = cache[key]
        rec[k] = {'params': bp, 'tuned': bool(tune[k]), 'n_val_users': nval[k],
                  'val_ndcg@10': (best[k][0] if tune[k] else None),
                  'launch_ndcg@10': rl['ndcg@10'], 'launch_recall@10': rl['recall@10'],
                  'all_ndcg@10': ra['ndcg@10'], 'all_recall@10': ra['recall@10']}
    return rec


def sub(u, i, keep):
    return u[keep], i[keep]


def setups(target, t, b, donors):
    """Yield (seed, catalog, train edges, launch session, overall session, views, info) per launch set."""
    if SPLIT != 'calendar':
        abroad = np.unique(np.concatenate([blocks[d].train_i for d in donors]))
        elig = np.intersect1d(np.intersect1d(np.unique(b.train_i), abroad), b.catalog)
        if len(elig) < MIN_ELIG:
            skip(t, -1, f'fewer than {MIN_ELIG} eligible products', n_elig=int(len(elig))); return
        for seed in SEEDS:
            if all((t, seed, f) in done for f in (0, 1)) or (t, seed) in skipped: continue
            rng = np.random.default_rng(stable_seed('launch3', t, seed))
            launched = np.sort(rng.choice(elig, size=max(MIN_LAUNCHED, int(round(LAUNCH_FRAC * len(elig)))), replace=False))
            kt = ~np.isin(b.train_i, launched); kv = ~np.isin(b.val_i, launched); lt = np.isin(b.test_i, launched)
            tu, ti = sub(b.train_u, b.train_i, kt); vu, vi = sub(b.val_u, b.val_i, kv)
            launch_s = E.TargetSession(target, E.ContextBlock(tu, ti, vu, vi, b.test_u[lt], b.test_i[lt], b.catalog))
            if len(launch_s.eval_users) < MIN_TEST_USERS:
                skip(t, seed, f'fewer than {MIN_TEST_USERS} launched-item test users', n_users=int(len(launch_s.eval_users)),
                     n_elig=int(len(elig)), n_launched=int(len(launched))); continue
            all_s = E.TargetSession(target, E.ContextBlock(tu, ti, vu, vi, b.test_u, b.test_i, b.catalog))
            zf = E.TargetSession(target, E.ContextBlock(tu, ti, EMPTY, EMPTY, vu, vi, b.catalog))
            hv = ~kv
            vf = E.TargetSession(target, E.ContextBlock(tu, ti, EMPTY, EMPTY, b.val_u[hv], b.val_i[hv], b.catalog))
            info = {'n_elig': int(len(elig)), 'n_launched': int(len(launched)),
                    'n_hidden_train': int((~kt).sum()), 'n_hidden_val': int(hv.sum()), 'n_launch_test': int(lt.sum())}
            yield seed, b.catalog, (tu, ti), launch_s, all_s, {'zf': zf, 'vf': vf}, info
    else:
        seed = 0
        if all((t, seed, f) in done for f in (0, 1)) or (t, seed) in skipped: return
        g = lcg.get(target)
        abroad = np.unique(np.concatenate([blocks[d].train_i for d in donors]))
        cand = np.unique(g['i'].to_numpy(np.int32)) if g is not None else EMPTY
        launched = np.intersect1d(cand, abroad)
        launched = launched[~np.isin(launched, b.catalog)]
        if len(launched) < MIN_LAUNCHED:
            skip(t, seed, f'fewer than {MIN_LAUNCHED} launched products', n_launched=int(len(launched)),
                 n_candidates=int(len(cand))); return
        g = g[g['i'].isin(launched)]
        gv = g[g.split_calendar == 'val']; gt = g[g.split_calendar == 'test']
        lvu, lvi = gv['u'].to_numpy(np.int32), gv['i'].to_numpy(np.int32)
        ltu, lti = gt['u'].to_numpy(np.int32), gt['i'].to_numpy(np.int32)
        cat = np.union1d(b.catalog, launched).astype(np.int32)
        launch_s = E.TargetSession(target, E.ContextBlock(b.train_u, b.train_i, b.val_u, b.val_i, ltu, lti, cat))
        if len(launch_s.eval_users) < MIN_TEST_USERS:
            skip(t, seed, f'fewer than {MIN_TEST_USERS} launched-item test users', n_users=int(len(launch_s.eval_users)),
                 n_launched=int(len(launched))); return
        all_s = E.TargetSession(target, E.ContextBlock(b.train_u, b.train_i, b.val_u, b.val_i,
                                                       np.concatenate([b.test_u, ltu]), np.concatenate([b.test_i, lti]), cat))
        zf = E.TargetSession(target, E.ContextBlock(b.train_u, b.train_i, EMPTY, EMPTY, b.val_u, b.val_i, cat))
        vf = E.TargetSession(target, E.ContextBlock(b.train_u, b.train_i, EMPTY, EMPTY, lvu, lvi, cat))
        info = {'n_candidates': int(len(cand)), 'n_launched': int(len(launched)), 'n_hidden_val': int(len(lvu)),
                'n_launch_test': int(len(ltu))}
        yield seed, cat, (b.train_u, b.train_i), launch_s, all_s, {'zf': zf, 'vf': vf}, info


for target in targets:
    t = ' '.join(target)
    if (t, -2) in skipped: continue
    donors = [tuple(x) for x in o[(o.target_market == target[0]) & (o.target_category == target[1])][['donor_market', 'donor_category']].values]
    b = blocks[target]
    ncat = len(b.catalog) if SPLIT != 'calendar' else None
    if MODEL == 'EASE':
        if SPLIT == 'calendar':
            g = lcg.get(target)
            ncat = len(np.union1d(b.catalog, g['i'].unique())) if g is not None else len(b.catalog)
        if ncat > Mod.EASE_MAX_ITEMS:
            skip(t, -2, f'EASE: catalog over {Mod.EASE_MAX_ITEMS} products', n_items=int(ncat)); continue
    timed_out = False
    for seed, cat, own, launch_s, all_s, views, info in setups(target, t, b, donors):
        for foreign in (0, 1):
            if (t, seed, foreign) in done: continue
            if time.time() - T0 > LIM: print('deadline', flush=True); sys.exit()
            el = [own] + ([(blocks[d].train_u, blocks[d].train_i) for d in donors] if foreign else [])
            t1 = time.time()
            set_alarm(UNIT_TIMEOUT)
            try:
                res = run_unit(el, cat, launch_s, all_s, views)
            except Timeout:
                timed_out = True
            finally:
                set_alarm(0)
            if timed_out:
                skip(t, -2, f'one unit exceeded {UNIT_TIMEOUT // 60} min (seed {seed}, foreign {foreign})')
                break
            rec = {'t': t, 'split': SPLIT, 'seed': seed, 'foreign': foreign, 'model': MODEL, **info,
                   'n_donors': len(donors), 'n_items': int(len(cat)),
                   'n_users_launch': int(len(launch_s.eval_users)), 'n_users_all': int(len(all_s.eval_users)),
                   'n_users_all_available': int(all_s.n_eval_users_available),
                   'zf': res['zf'], 'vf': res['vf'], 'sec': round(time.time() - t1, 2)}
            open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n'); done.add((t, seed, foreign))
            print(t, seed, foreign, rec['sec'], flush=True)
        if timed_out: break
print('ALL DONE', flush=True)
