"""Scarcity experiment (RQ3; replaces the v3 scarcity runner in the main block of mustadd.py).

Arms (names as used in the manuscript; mustadd.py uses the opposite letters for P and R):
  F  full reference     : fit = 100% local (+foreign), scoring histories = 100% local       (mustadd 'F')
  P  patterns scarce    : fit = f-subsample of local (+foreign), histories = 100% local      (mustadd 'R')
  R  histories scarce   : fit = 100% local (+foreign), histories = f-subsample of local      (mustadd 'P')
  B  both scarce        : fit = f-subsample (+foreign), histories = the same f-subsample     (mustadd 'B')
Retention levels f in {0.05, 0.10, 0.25, 0.50}; subsamples seed 0,1,2 drawn exactly as in mustadd
(permutation from stable_seed('mustadd', context, seed), keep the first round(f*n) local training
interactions, so subsamples are nested across f within a seed and shared by the three arms).
Foreign data (all same-category XM donors) enter model fitting only; foreign users never have
scoring histories in the target context because users are disjoint across countries.

Fallback variant (fb=1): users whose scoring history is empty are ranked by item popularity in the
model-fitting matrix (local subsample or full local, plus foreign when foreign=1), over the target
catalog; the rest are scored by the model as usual. Run for f=0.10, all arms, ItemKNN/EASE random.
Without the fallback, an empty history gives an all-zero score row for ItemKNN, EASE and UserKNN;
after masking the user's own training/validation items, metrics.rank_metrics breaks the ties in
favour of the lower catalog index, i.e. in ASIN order (item ids are factorised from sorted ASINs).

Semantics are those of mustadd.run_unit (tuning on the validation view with the same subsample when
it has >=1 validation user, else grid[0]; MemoryError on a grid point skips that point). To save
compute, a fitted model is cached and reused across arms that share the same fitting matrix
(F and R share the full fit; P and B share the subsample fit); only the scoring-history matrix is
swapped via mustadd.set_prof. `--check` verifies equality with mustadd.run_unit on sample units.

Per unit we record overall NDCG@10 and, per evaluated test user, the bucket of retained
scoring-history interactions in the target context (0, 1, 2+): user counts and mean NDCG@10.

Inclusion rule: every context with an XM donor and users x items <= 2.3e6 (from
results/contexts{,_calendar}.parquet) runs with all 3 subsamples (phase A). Larger contexts are then
tried with subsample 0 only, cheapest first (phase B), and with subsamples 1,2 if the compute budget
allows (phase C). A unit that exceeds 20 minutes is aborted, recorded (err='timeout') and the
context is excluded for that model x split. EASE skips catalogs > 4000 items (models.EASE_MAX_ITEMS).

Usage: python scarcity2.py LIMIT_SECONDS [phases e.g. A,B,C] [--check]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time, signal
sys.path.insert(0, os.path.expanduser(SRC)); sys.path.insert(0, os.path.expanduser(CODE))
import numpy as np, pandas as pd, scipy.sparse as sp
import data as D, experiment as E, models as Mod
import mustadd as MA
from util import stable_seed

LEVELS = (0.05, 0.10, 0.25, 0.50)
SEEDS = (0, 1, 2)
ARMS = ('P', 'R', 'B')
MA_REGIME = {'F': 'F', 'P': 'R', 'R': 'P', 'B': 'B'}   # manuscript arm -> mustadd regime letter
MODEL_SPLITS = [('ItemKNN', 'random'), ('EASE', 'random'), ('UserKNN', 'random'),
                ('ItemKNN', 'temporal'), ('ItemKNN', 'calendar'), ('EASE', 'calendar')]
FB_MODEL_SPLITS = {('ItemKNN', 'random'), ('EASE', 'random')}
FB_LEVEL = 0.10
SMALL = 2.3e6
UNIT_TIMEOUT = 1200   # 20 minutes
# Budget decision (taken while the run was in progress, before any results were analysed): the
# design did not fit the ~4 h compute guide, so f=0.05 is dropped for UserKNN for every context not
# yet started when this switch was turned on (UserKNN f=0.05 results that exist are kept in the raw
# file but reported only as a partial run). Turn on with env SC_DROP_UKNN005=1.
DROP_UKNN005 = os.environ.get('SC_DROP_UKNN005') == '1'


def levels_for(model):
    return tuple(f for f in LEVELS if not (DROP_UKNN005 and model == 'UserKNN' and f == 0.05))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# The per-unit timeout uses SIGALRM, which exists only on POSIX systems. On Windows the timeout is skipped:
# every unit runs to completion, so a context that hit the 20-min limit in the paper run is not excluded there.
HAS_ALARM = hasattr(signal, 'SIGALRM')
if HAS_ALARM:
    signal.signal(signal.SIGALRM, _alarm)
else:
    print(f'note: no SIGALRM on this platform; the {UNIT_TIMEOUT // 60}-min per-unit timeout is not applied', flush=True)


def set_alarm(sec):
    if HAS_ALARM:
        signal.alarm(sec)


class Fallback(Mod.Recommender):
    """Popularity of the fitting matrix for users with an empty scoring history."""
    name = 'fallback'

    def __init__(self, base, prof, pop):
        self.base, self.prof, self.pop = base, prof, pop

    def score(self, rows):
        sc = self.base.score(rows)
        empty = self.prof[rows].getnnz(axis=1) == 0
        if empty.any():
            sc[empty] = self.pop
        return sc


def keep_mask(t, n, seed, f):
    perm = np.random.default_rng(stable_seed('mustadd', t, seed)).permutation(n)
    keep = np.zeros(n, bool); keep[perm[:int(round(f * n))]] = True
    return keep


class Engine:
    """Fits cached per (session, fitting-matrix key, grid index); profiles swapped per arm."""

    def __init__(self, model, s, vs):
        self.model, self.s, self.vs = model, s, vs
        self.grid = MA.GRIDS[model]
        self.tune = len(self.grid) > 1 and len(vs.eval_users) > 0
        self.cache = {}

    def fitted(self, which, key, fitmat, cols, j):
        k = (which, key, j)
        if k not in self.cache:
            try:
                self.cache[k] = MA.build(self.model, fitmat, cols)(self.grid[j])
            except MemoryError:
                self.cache[k] = None
        return self.cache[k]

    def drop(self, pred):
        for k in [k for k in self.cache if pred(k[1])]:
            del self.cache[k]

    def run(self, key, mats_val, mats_test, arm, fb):
        """mats_* = (R, Rs, cols, rows). Returns (res, per_user ndcg, hist counts, best params, tuned)."""
        def pick(m):
            R, Rs, cols, rows = m
            fit, prof = {'F': (R, R), 'P': (Rs, R), 'R': (R, Rs), 'B': (Rs, Rs)}[arm]
            return fit, prof, cols, rows

        def ev(which, m, j):
            fit, prof, cols, rows = pick(m)
            fkey = key + ('full' if fit is m[0] else 'sub',)
            mod = self.fitted(which, fkey, fit, cols, j)
            if mod is None:
                return None
            MA.set_prof(mod, self.model, prof)
            if fb:
                pop = np.asarray(fit[:, cols].sum(axis=0)).ravel().astype(np.float32)
                mod = Fallback(mod, prof, pop)
            sess = self.vs if which == 'val' else self.s
            return E.evaluate(mod, sess, rows, candidate_sizes=(), seed=0)

        best, bj = -1, None
        if self.tune:
            for j in range(len(self.grid)):
                out = ev('val', mats_val, j)
                if out is None:
                    continue
                if out[0]['ndcg@10'] > best:
                    best, bj = out[0]['ndcg@10'], j
        if bj is None:
            bj = 0
        out = ev('test', mats_test, bj)
        if out is None:
            raise MemoryError('test fit failed')
        _, prof, _, rows = pick(mats_test)
        hist = prof[rows].getnnz(axis=1)
        return out[0], out[1]['ndcg@10'], hist, self.grid[bj]


def buckets(hist, nd):
    b = np.minimum(hist, 2)
    n = [int((b == k).sum()) for k in range(3)]
    m = [float(nd[b == k].mean()) if n[k] else None for k in range(3)]
    return n, m


def context_costs(split):
    tag = '_calendar' if split == 'calendar' else ''
    c = pd.read_parquet(f'{REPO}/results/contexts{tag}.parquet')
    return {f'{m} {cc}': int(u) * int(i) for m, cc, u, i in zip(c['market'], c['category'], c['users'], c['items'])}


def targets_of(split):
    o = pd.read_parquet(f'{REPO}/results/overlap_{split}.parquet'); o = o[o.channel == 'XM']
    out = {}
    for (tm, tc), g in o.groupby(['target_market', 'target_category']):
        out[f'{tm} {tc}'] = [tuple(x) for x in g[['donor_market', 'donor_category']].values]
    return out


def out_path(model, split):
    return f'{PAPER_RAW}/sc_{model}_{split}.jsonl'


def load_done(model, split):
    done, dead = set(), set()
    fn = out_path(model, split)
    if os.path.exists(fn):
        for l in open(fn, encoding='utf-8'):
            r = json.loads(l)
            if r.get('err') in ('timeout', 'ease_too_large'):
                dead.add(r['t'])
            else:
                done.add((r['t'], r['arm'], r['seed'], r['f'], r['foreign'], r['fb']))
    return done, dead


def plan_units(model, split, seeds):
    """Ordered unit list for one context: F first, then per foreign, seed, level."""
    units = []
    for foreign in (0, 1):
        units.append(('F', 0, 1.0, foreign, 0))
        for seed in seeds:
            for f in levels_for(model):
                for arm in ARMS:
                    units.append((arm, seed, f, foreign, 0))
                    if (model, split) in FB_MODEL_SPLITS and f == FB_LEVEL:
                        units.append((arm, seed, f, foreign, 1))
    return units


_DATA = {}


def blocks_for(split):
    if split not in _DATA:
        _DATA[split] = E.load_blocks(D.load_for(split), split)
    return _DATA[split]


def run_context(model, split, t, donors, seeds, T0, LIM):
    out = out_path(model, split)
    done, dead = load_done(model, split)
    if t in dead:
        return 'dead'
    units = [u for u in plan_units(model, split, seeds) if (t,) + u not in done]
    if not units:
        return 'done'
    blocks = blocks_for(split)
    target = tuple(t.split(' '))
    b = blocks[target]
    if model == 'EASE' and len(b.catalog) > Mod.EASE_MAX_ITEMS:
        open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps({'t': t, 'split': split, 'model': model, 'arm': 'skip', 'err': 'ease_too_large',
                                         'n_items': int(len(b.catalog))}) + '\n')
        return 'dead'
    s = E.TargetSession(target, b); vs = s.validation_view()
    eng = Engine(model, s, vs)
    n = len(b.train_u)
    els = {0: [(b.train_u, b.train_i)]}
    els[1] = els[0] + [(blocks[d].train_u, blocks[d].train_i) for d in donors]
    matcache = {}

    def mats_for(foreign, seed, f):
        k = (foreign, seed, f)
        if k not in matcache:
            matcache.clear()
            keep = np.ones(n, bool) if f == 1.0 else keep_mask(t, n, seed, f)
            matcache[k] = (MA.mats(els[foreign], vs.catalog, vs.eval_users, keep),
                           MA.mats(els[foreign], s.catalog, s.eval_users, keep))
        return matcache[k]

    for (arm, seed, f, foreign, fb) in units:
        if time.time() - T0 > LIM:
            return 'deadline'
        # free fits that later units cannot use: other foreign arm's full fit, other subsamples
        eng.drop(lambda k: (k[0] == 'fullfit' and k[1] != foreign) or
                 (k[0] != 'fullfit' and k[:3] != (foreign, seed, f)))
        t1 = time.time()
        set_alarm(UNIT_TIMEOUT)
        err = None
        try:
            mv, mt = mats_for(foreign, seed, f)
            # full fits are keyed by foreign only, so F and every R-arm unit share them;
            # subsample fits by (foreign, seed, f), so P and B (and their fb variants) share them
            key = ('fullfit', foreign) if arm in ('F', 'R') else (foreign, seed, f)
            res, nd, hist, bp = eng.run(key, mv, mt, arm, fb)
        except Timeout:
            err = 'timeout'
        except MemoryError:
            err = 'memory'
        finally:
            set_alarm(0)
        rec = {'t': t, 'split': split, 'model': model, 'arm': arm, 'seed': seed, 'f': f, 'foreign': foreign, 'fb': fb,
               'err': err, 'sec': round(time.time() - t1, 2), 'n_dom': int(n),
               'n_for': int(sum(len(e[0]) for e in els[foreign][1:])) if foreign else 0}
        if err is None:
            hn, hm = buckets(hist, nd)
            rec.update({'ndcg': res['ndcg@10'], 'recall': res['recall@10'], 'params': bp, 'tuned': eng.tune,
                        'n_eval': int(len(nd)), 'h_n': hn, 'h_ndcg': hm})
        open(out, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')
        if err == 'timeout':
            return 'dead'
    return 'done'


def check():
    """Equality with mustadd.run_unit on a few units (manuscript arm -> mustadd regime)."""
    blocks = blocks_for('random'); tg = targets_of('random')
    for model, t in [('ItemKNN', 'de Office_Products'), ('EASE', 'fr Cell_Phones_and_Accessories'),
                     ('UserKNN', 'jp Office_Products')]:
        target = tuple(t.split(' ')); b = blocks[target]
        s = E.TargetSession(target, b); vs = s.validation_view(); eng = Engine(model, s, vs)
        n = len(b.train_u)
        for foreign in (0, 1):
            el = [(b.train_u, b.train_i)] + ([(blocks[d].train_u, blocks[d].train_i) for d in tg[t]] if foreign else [])
            for arm in ('F', 'P', 'R', 'B'):
                f = 1.0 if arm == 'F' else 0.1
                keep = np.ones(n, bool) if arm == 'F' else keep_mask(t, n, 1, f)
                ref, rp = MA.run_unit(model, s, vs, el, keep, MA_REGIME[arm])
                mv = MA.mats(el, vs.catalog, vs.eval_users, keep); mt = MA.mats(el, s.catalog, s.eval_users, keep)
                key = ('fullfit', foreign) if arm in ('F', 'R') else (foreign, 1, f)
                res, nd, hist, bp = eng.run(key, mv, mt, arm, 0)
                print(model, t, foreign, arm, ref, res['ndcg@10'], rp, bp, 'OK' if (ref == res['ndcg@10'] and rp == bp) else 'MISMATCH', flush=True)


if __name__ == '__main__':
    if '--check' in sys.argv:
        check(); sys.exit()
    LIM = float(sys.argv[1]); PHASES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['A', 'B', 'C']
    T0 = time.time()
    work = []   # (phase, cost, model, split, t, donors, seeds)
    for model, split in MODEL_SPLITS:
        cost = context_costs(split); tg = targets_of(split)
        for t, donors in tg.items():
            c = cost[t]
            if c <= SMALL:
                work.append(('A', c, model, split, t, donors, SEEDS))
            else:
                work.append(('B', c, model, split, t, donors, (0,)))
                work.append(('C', c, model, split, t, donors, SEEDS))
    order = {'A': 0, 'B': 1, 'C': 2}
    # group by split (one dataset in memory at a time) within a phase, cheapest contexts first
    split_rank = {'random': 0, 'temporal': 1, 'calendar': 2}
    # phase A: grouped by split; phases B/C: cheapest context first across all model x splits
    work.sort(key=lambda w: (order[w[0]], split_rank[w[3]] if w[0] == 'A' else 0, w[1], split_rank[w[3]], w[2]))
    for ph, c, model, split, t, donors, seeds in work:
        if ph not in PHASES:
            continue
        t1 = time.time()
        st = run_context(model, split, t, donors, seeds, T0, LIM)
        print(f'{ph} {model} {split} {t} cost={c} -> {st} ({time.time() - t1:.1f}s, total {time.time() - T0:.0f}s)', flush=True)
        if st == 'deadline':
            print('deadline', flush=True); sys.exit()
    print('ALL DONE', flush=True)
