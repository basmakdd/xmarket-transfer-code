"""Market-Aware GMF (Bhargav, Aliannejadi & Kanoulas, ECIR 2023) and its two GMF controls.

Variants (identical architecture, loss, sampler, optimiser, budget and tuning; only the
training data / market embedding differ):
  local  : GMF on the target context's training interactions only
  pooled : GMF on target + all same-category foreign training interactions (no market embedding)
  ma     : MA-GMF on the same pooled data; logit = h^T (p_u * (o_m * q_i)) + b, o_m = market embedding
Training: BCE, 4 uniformly sampled negatives per positive (re-drawn every epoch, from the item
catalog of the positive's own market-category context), Adam (weight_decay = L2), dim 32,
batch 1024, grid lr in {1e-3, 5e-3} x L2 in {0, 1e-6}; up to 30 epochs, validation NDCG@10
(full catalog, experiment.TargetSession.validation_view) after every epoch, patience 3;
configuration + epoch chosen by best validation NDCG@10. Targets without validation users:
lr 1e-3, L2 0, state after exactly 10 epochs (flag tuned=False).
Evaluation: experiment.evaluate on TargetSession (target catalog, <=20k test users, train+val masked).

Pooled and MA data for a category are the same for every target of that category (the donor
lists hold all same-category foreign markets), and the training seed depends on (variant,
category, config) only. One training run per (category, variant, config) therefore produces
exactly the trajectory each target would get if trained alone; per-target early stopping is
applied by tracking each target's validation curve separately.

Excluded categories (pre-specified compute rule): a (category, variant) whose first epoch (training plus
validation evaluation of all targets of the category) takes longer than MAX_EPOCH_SEC = 15 s is skipped and
written to paper/raw/mkt_skips.jsonl; if the pooled or the MA variant of a category is skipped, the whole
category is dropped (no variant, including the local controls, is reported for it). In the paper run this excluded three categories, all at the pooled variant: Electronics (first epoch
59.9 s), Movies_and_TV (46.6 s) and Books (399.3 s), leaving 52 contexts. The rule is hardware-dependent: on a
faster (or slower) machine the same 15-s threshold can admit (or exclude) different categories. To reproduce the
paper's context set exactly, keep the shipped mkt_skips.jsonl (the run is resumable and honours it), or set the
environment variable MKT_MAX_EPOCH_SEC to a threshold suited to the machine.

usage: mkt_run.py LIMIT_SECONDS [CATEGORY,...]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time
sys.path.insert(0, os.path.expanduser(SRC))
import numpy as np, pandas as pd, torch
import data as D, experiment as E, models as Mod
from util import stable_seed

torch.set_num_threads(1)
SPLIT = 'random'
DIM, BATCH, NEG, MAXEP, PATIENCE, FIXED_EP = 32, 1024, 4, 30, 3, 10
GRID = [(lr, l2) for lr in (1e-3, 5e-3) for l2 in (0.0, 1e-6)]
DEFAULT = (1e-3, 0.0)
# Pre-specified exclusion rule: a (category, variant) whose first epoch (training + validation
# evaluation of all targets) exceeds 15 s would take > 30 min in the worst case (4 configs x 30 epochs).
MAX_EPOCH_SEC = float(os.environ.get('MKT_MAX_EPOCH_SEC', 15))
OUT = f'{PAPER_RAW}/mkt_runs.jsonl'
LOG = f'{PAPER_RAW}/mkt_skips.jsonl'


class GMF(torch.nn.Module):
    def __init__(self, n_users, n_items, n_markets=0):
        super().__init__()
        # Initialisation (not specified in the MA paper): user/item embeddings N(0, 0.01) as in
        # NCF (He et al., 2017); prediction layer h with the PyTorch Linear default; market
        # embeddings start at 1, so MA-GMF starts from exactly the pooled GMF function and the
        # market embedding learns only market-specific rescaling of the item factors.
        self.P = torch.nn.Embedding(n_users, DIM); torch.nn.init.normal_(self.P.weight, 0.0, 0.01)
        self.Q = torch.nn.Embedding(n_items, DIM); torch.nn.init.normal_(self.Q.weight, 0.0, 0.01)
        self.O = None
        if n_markets:
            self.O = torch.nn.Embedding(n_markets, DIM); torch.nn.init.ones_(self.O.weight)
        self.h = torch.nn.Linear(DIM, 1)

    def forward(self, u, i, m):
        q = self.Q(i)
        if self.O is not None:
            q = q * self.O(m)
        return self.h(self.P(u) * q).squeeze(-1)

    @torch.no_grad()
    def scorer(self, users, items, m):
        """Frozen copy of what is needed to score `users` x `items` in market m."""
        q = self.Q.weight[items]
        if self.O is not None:
            q = q * self.O.weight[m]
        w = self.h.weight[0]
        return Scorer((self.P.weight[users] * w).numpy().copy(), q.numpy().copy())


class Scorer(Mod.Recommender):
    """experiment.evaluate-compatible: score(rows) -> (len(rows), n_target_items)."""
    name = 'GMF'

    def __init__(self, Pw, Qc):
        self.Pw, self.Qc = Pw, Qc   # bias is constant per row and does not change the ranking

    def score(self, rows):
        return (self.Pw[rows] @ self.Qc.T).astype(np.float32)


def val_ndcg(sc, vs):
    return E.evaluate(sc, vs, np.arange(len(vs.eval_users)), candidate_sizes=(), seed=0)[0]['ndcg@10']


def train(variant, seed_key, ctxs, targets, sessions, cfg, blocks, deadline, mult=1):
    """ctxs: contexts whose train edges are used. targets: contexts to select/evaluate.
    Returns {target: dict(best_val, best_ep, scorer_test, epochs_run)} and seconds per epoch."""
    lr, l2 = cfg
    users = np.unique(np.concatenate([blocks[c].train_u for c in ctxs] +
                                     [sessions[t][0].eval_users for t in targets] +
                                     [sessions[t][1].eval_users for t in targets]))
    items = np.unique(np.concatenate([blocks[c].catalog for c in ctxs]))
    mk = sorted({c[0] for c in ctxs}); mid = {m: j for j, m in enumerate(mk)}
    U = np.concatenate([np.searchsorted(users, blocks[c].train_u) for c in ctxs]).astype(np.int64)
    I = np.concatenate([np.searchsorted(items, blocks[c].train_i) for c in ctxs]).astype(np.int64)
    Mk = np.concatenate([np.full(len(blocks[c].train_u), mid[c[0]]) for c in ctxs]).astype(np.int64)
    cats = {mid[c[0]]: np.searchsorted(items, blocks[c].catalog) for c in ctxs}
    seed = stable_seed('mkt', variant, seed_key, cfg)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GMF(len(users), len(items), len(mk) if variant == 'ma' else 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    lossf = torch.nn.BCEWithLogitsLoss()
    # per-target bookkeeping
    st = {}
    for t in targets:
        s, vs = sessions[t]
        st[t] = dict(tuned=len(vs.eval_users) > 0, best_val=-1.0, best_ep=0, bad=0, done=False,
                     tu=np.searchsorted(users, s.eval_users), vu=np.searchsorted(users, vs.eval_users),
                     ti=np.searchsorted(items, s.catalog), m=mid[t[0]], scorer=None)
    if cfg != DEFAULT:   # untuned targets only use the default config
        for t in targets:
            if not st[t]['tuned']: st[t]['done'] = True
    n = len(U); ep_secs = []
    for ep in range(1, MAXEP + 1):
        if all(v['done'] for v in st.values()): break
        if time.time() > deadline: raise TimeoutError
        t0 = time.time()
        for _rep in range(mult):   # mult > 1 only for local_steps (update-matched control)
            negs = np.empty((n, NEG), np.int64)
            for m, cat in cats.items():
                idx = np.where(Mk == m)[0]
                negs[idx] = cat[rng.integers(0, len(cat), (len(idx), NEG))]
            uu = np.concatenate([U, np.repeat(U, NEG)]); ii = np.concatenate([I, negs.ravel()])
            mm = np.concatenate([Mk, np.repeat(Mk, NEG)]); yy = np.concatenate([np.ones(n), np.zeros(n * NEG)])
            perm = rng.permutation(len(uu))
            uu, ii, mm, yy = (torch.from_numpy(a[perm]) for a in (uu, ii, mm, yy.astype(np.float32)))
            model.train()
            for b in range(0, len(uu), BATCH):
                sl = slice(b, b + BATCH)
                opt.zero_grad()
                loss = lossf(model(uu[sl], ii[sl], mm[sl]), yy[sl])
                loss.backward(); opt.step()
        model.eval()
        for t, v in st.items():
            if v['done']: continue
            if v['tuned']:
                vs = sessions[t][1]
                sc = model.scorer(torch.from_numpy(v['vu']), torch.from_numpy(v['ti']), v['m'])
                val = val_ndcg(sc, vs)
                if val > v['best_val']:
                    v.update(best_val=val, best_ep=ep, bad=0,
                             scorer=model.scorer(torch.from_numpy(v['tu']), torch.from_numpy(v['ti']), v['m']))
                else:
                    v['bad'] += 1
                    if v['bad'] >= PATIENCE: v['done'] = True
            elif ep == FIXED_EP:
                v.update(best_val=float('nan'), best_ep=ep, done=True,
                         scorer=model.scorer(torch.from_numpy(v['tu']), torch.from_numpy(v['ti']), v['m']))
        ep_secs.append(time.time() - t0)
        print(f'  {variant} {seed_key} cfg={cfg} ep={ep} loss={loss.item():.4f} {ep_secs[-1]:.1f}s', flush=True)
        if ep == 1 and ep_secs[0] > MAX_EPOCH_SEC:
            return None, ep_secs, dict(n_users=len(users), n_items=len(items), n_edges=n)
    return st, ep_secs, dict(n_users=len(users), n_items=len(items), n_edges=n)


def main():
    LIM = float(sys.argv[1]); deadline = time.time() + LIM
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    done = set(); skipped = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'): r = json.loads(l); done.add((r['t'], r['variant']))
    if os.path.exists(LOG):
        for l in open(LOG, encoding='utf-8'): r = json.loads(l); skipped.add((r['unit'], r['variant']))
    o = pd.read_parquet(f'{REPO}/results/overlap_{SPLIT}.parquet'); o = o[o.channel == 'XM']
    data = D.load_for(SPLIT); blocks = E.load_blocks(data, SPLIT)
    tg = sorted(set(zip(o.target_market, o.target_category)))
    bycat = {}
    for t in tg: bycat.setdefault(t[1], []).append(t)
    donors = {t: [tuple(x) for x in o[(o.target_market == t[0]) & (o.target_category == t[1])][['donor_market', 'donor_category']].values] for t in tg}
    size = {c: sum(len(blocks[x].train_u) for x in set(ts) | {d for t in ts for d in donors[t]}) for c, ts in bycat.items()}
    for cat in sorted(bycat, key=lambda c: size[c]):
        if only and cat not in only: continue
        ts = sorted(bycat[cat], key=lambda t: len(blocks[t].train_u))
        pool = sorted(set(ts) | {d for t in ts for d in donors[t]})
        for t in ts:   # pooled data must be identical for all targets of the category
            assert sorted(set(donors[t]) | {t}) == pool, t
        sessions = {}
        def sess(t):
            if t not in sessions:
                s = E.TargetSession(t, blocks[t]); sessions[t] = (s, s.validation_view())
            return sessions[t]
        # pooled/MA first: if either is excluded the category is dropped and local is not run
        units = [(v, cat, pool, ts) for v in ('pooled', 'ma')] + \
                [('local', ' '.join(t), [t], [t]) for t in ts] + \
                [('local_steps', ' '.join(t), [t], [t]) for t in ts]
        # local_steps (sensitivity added after the main run): local data, but each of the 30 'epochs'
        # makes as many gradient updates as one pooled epoch (mult passes over the local data,
        # fresh negatives per pass); validation, patience and the 10-epoch default count these
        # pooled-equivalent epochs, so the update budget and stopping granularity match pooled/MA.
        steps = lambda nn: int(np.ceil(nn * (1 + NEG) / BATCH))
        n_pool = sum(len(blocks[x].train_u) for x in pool)
        mults = {' '.join(t): max(1, int(round(steps(n_pool) / steps(len(blocks[t].train_u))))) for t in ts}
        for variant, key, ctxs, targets in units:
            if (cat, 'pooled') in skipped or (cat, 'ma') in skipped: break
            if all((' '.join(t), variant) in done for t in targets) or (key, variant) in skipped: continue
            if time.time() > deadline: print('deadline', flush=True); sys.exit()
            for t in targets: sess(t)
            t1 = time.time(); res = {}; meta = None; skip = None
            try:
                for cfg in GRID:
                    if cfg != DEFAULT and not any(len(sessions[t][1].eval_users) for t in targets): continue
                    st, secs, meta = train(variant, key, ctxs, targets, sessions, cfg, blocks, deadline,
                                           mult=mults[key] if variant == 'local_steps' else 1)
                    if st is None:
                        skip = dict(unit=key, variant=variant, reason=f'first epoch {secs[0]:.1f}s > {MAX_EPOCH_SEC}s', **meta); break
                    res[cfg] = (st, secs)
            except TimeoutError:
                print('deadline inside unit', flush=True); sys.exit()
            if skip:
                open(LOG, 'a', encoding='utf-8', newline='\n').write(json.dumps(skip) + '\n'); print('SKIP', skip, flush=True)
                skipped.add((key, variant)); continue
            for t in targets:
                tn = ' '.join(t); s, vs = sessions[t]
                grid = [dict(lr=c[0], l2=c[1], best_val=r[0][t]['best_val'], best_ep=r[0][t]['best_ep'])
                        for c, r in res.items() if r[0][t]['scorer'] is not None]
                cands = [(r[0][t]['best_val'], -i, c) for i, (c, r) in enumerate(res.items()) if r[0][t]['scorer'] is not None]
                tuned = len(vs.eval_users) > 0
                c = max(cands)[2] if tuned else DEFAULT
                sv = res[c][0][t]
                m, _ = E.evaluate(sv['scorer'], s, np.arange(len(s.eval_users)), seed=0)
                rec = dict(t=tn, split=SPLIT, variant=variant, lr=c[0], l2=c[1], best_ep=sv['best_ep'],
                           val_ndcg=sv['best_val'], tuned=tuned, n_val_users=int(len(vs.eval_users)), grid=grid,
                           train_contexts=[' '.join(x) for x in ctxs], unit_sec=round(time.time() - t1, 1),
                           sec_per_epoch=round(float(np.mean(np.concatenate([r[1] for r in res.values()]))), 2),
                           mult=mults[tn] if variant == 'local_steps' else 1, **meta, **{k: v for k, v in m.items()})
                open(OUT, 'a', encoding='utf-8', newline='\n').write(json.dumps(rec) + '\n')
            print(f'DONE {variant} {key} {time.time()-t1:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
