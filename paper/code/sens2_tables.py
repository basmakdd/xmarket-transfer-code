"""sens2: side-by-side comparison tables, paper data vs _xd (cross-market reviewers removed), _nv (variants merged, users
pooled over markets) and _nvc (variants merged within each market x category context). Derived from sens_tables.py
(same statistics and bootstrap labels, so rows for orig/_xd/_nv on 'all' sets reproduce sens_tables.md exactly).
Inputs: paper/raw/* (paper data) and paper/raw/sens{TAG}/* (variants). Outputs: paper/results/sens2_*.csv, sens2_tables.md.

Statistics (fixed before looking at results, same definitions as the main analyses):
  RQ1  rq1_tables.measure_rows / paired_rows (pair-level two-way cluster bootstrap), measures sp, top10, cos, nbr5;
       all foreign data ('all') and two-sided matched ('mat').
  RQ2  per context gain = arm - local (NDCG@10); median and mean over contexts with boot.summarize (context level);
       counts of contexts with gain > +0.01 / < -0.01; mean excluding IN Arts & Crafts.
  Launch (launch3, random split, zf tuning): per context (3 launch sets averaged) dF = foreign - chance (Random ranker),
       gain = foreign - local on launched-item NDCG@10, dAll = overall NDCG@10 change; medians + CI; counts > 0.01.
  Scarcity (ItemKNN random, f = 0.10, seed 0): gain = (+foreign) - local per arm P/R/B; G = gain / (Q_full - local)
       over contexts with Q_full - local > 0.01 (Q_full = complete local data, foreign 0).
Each comparison is shown on all available contexts/pairs of each variant AND on the set common to all four.
(Outputs listed above.)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, sys, json, glob, io
import numpy as np, pandas as pd
import boot
import rq1_tables as R1

NOTES = f'{REPO}/paper/results'
RAW = PAPER_RAW   # paper/raw unless the environment variable PAPER_RAW is set
VAR = ['orig', '_xd', '_nv', '_nvc']
VNAME = {'orig': 'paper data', '_nv': '_nv variants merged', '_xd': '_xd x-market reviewers dropped', '_nvc': '_nvc variants merged within context'}
IN_AC = 'in Arts_Crafts_and_Sewing'
MD = io.StringIO()


def md(s=''):
    print(s); MD.write(s + '\n')


def src(v, name):
    return f'{RAW}/{name}' if v == 'orig' else f'{RAW}/sens{v}/{name}'


def tagof(v):
    return '' if v == 'orig' else v


def f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '--'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}')


def fci(t, d=3, sign=True):
    return '--' if np.isnan(t[0]) else f'{f(t[0], d, sign)} [{f(t[1], d, sign)}, {f(t[2], d, sign)}]'


def csum(vals, ts, label, stat='median'):
    vals = np.asarray(vals, float); ok = ~np.isnan(vals)
    if ok.sum() < 2: return (np.nan, np.nan, np.nan)
    mk = [t.split(' ')[0] for t in ts]; ct = [t.split(' ')[1] for t in ts]
    return boot.summarize(vals, mk, ct, None, stat, label)


# ============================================================================ RQ1
def rq1_load(v):
    fn = src(v, f'rq1_agree{tagof(v)}.jsonl')
    if not os.path.exists(fn): return None
    X = pd.DataFrame([json.loads(l) for l in open(fn, encoding='utf-8')]).drop_duplicates(['t', 'donor_market'])
    X = X.sort_values(['t', 'donor_market']).reset_index(drop=True)
    for c in X.columns:
        if X[c].dtype == object and not c.startswith('splits') and c not in ('t', 'target_market', 'donor_market', 'category'):
            X[c] = pd.to_numeric(X[c])
    return X


def rq1():
    md('\n## RQ1: cross-market agreement on shared products\n')
    XS = {v: rq1_load(v) for v in VAR}
    XS = {v: x for v, x in XS.items() if x is not None}
    keys = [set(zip(x.t, x.donor_market)) for x in XS.values()]
    common = set.intersection(*keys)
    md(f'Pairs per variant: ' + ', '.join(f'{v}: {len(x)} ({x.t.nunique()} targets)' for v, x in XS.items()) + f'; common to all: {len(common)}')
    rows, prow = [], []
    for setname in ('all', 'common'):
        for v, X in XS.items():
            Z = X if setname == 'all' else X[[k in common for k in zip(X.t, X.donor_market)]].reset_index(drop=True)
            lab = f'sens{tagof(v) or "_orig"}_{setname}'
            M = R1.measure_rows(Z, ['sp', 'top10', 'cos', 'nbr5'], lab); M['set'] = setname; M['var'] = v; rows.append(M)
            P = R1.paired_rows(Z, [('cos', 'sp'), ('nbr5', 'top10')], lab); P['set'] = setname; P['var'] = v; prow.append(P)
    M = pd.concat(rows); P = pd.concat(prow)
    M.to_csv(f'{NOTES}/sens2_rq1_measures.csv', index=False, lineterminator='\n'); P.to_csv(f'{NOTES}/sens2_rq1_paired.csv', index=False, lineterminator='\n')
    for setname in ('all', 'common'):
        md(f'\n### RQ1 measures ({"all pairs of each variant" if setname == "all" else "pairs common to all variants"}; all foreign data)\n')
        md('| Measure | Data | Pairs | Local L | Foreign F | Null F0 | Ratio F/L, L>=0.10 (n) | D = F-L [CI] | D>0 | F - chance [CI] |')
        md('|---|---|---|---|---|---|---|---|---|---|')
        for m in ['sp', 'top10', 'cos', 'nbr5']:
            for v in XS:
                r = M[(M.set == setname) & (M['var'] == v) & (M.measure == m) & (M.arm == 'all')].iloc[0]
                md(f"| {m} | {v} | {r.pairs} | {f(r.local, 2, False)} | {f(r.foreign, 2, False)} | {f(r.null_foreign, 2, False)} | "
                   f"{fci((r['ratio_0.10'], r['ratio_0.10_lo'], r['ratio_0.10_hi']), 2, False)} ({r['ratio_0.10_n']}) | "
                   f"{fci((r.D, r.D_lo, r.D_hi), 2)} | {100 * r.D_pos_share:.0f}% | {fci((r.excess, r.excess_lo, r.excess_hi), 2)} |")
        md(f'\n### RQ1 paired comparison, patterns vs popularity ({setname})\n')
        md('| Comparison | Scale | Data | Pairs (all) | Patterns higher (all) | Median diff (all) | Pairs (matched) | Patterns higher (matched) | Median diff (matched) |')
        md('|---|---|---|---|---|---|---|---|---|')
        for (pa, po) in [('cos', 'sp'), ('nbr5', 'top10')]:
            for sc, th in [('D (F-L)', None), ('chance-corrected', 0.05), ('ratio', 0.10)]:
                for v in XS:
                    q = P[(P.set == setname) & (P['var'] == v) & (P.pattern == pa) & (P.scale == sc)]
                    if th is not None: q = q[np.isclose(q.threshold, th)]
                    cells = []
                    for arm in ('all', 'mat'):
                        r = q[q.arm == arm].iloc[0]
                        cells.append(f"{r.pairs} | {100 * r.share_pattern_higher:.0f}% [{100 * r.share_lo:.0f}, {100 * r.share_hi:.0f}] | {fci((r['median'], r.lo, r.hi), 2)}")
                    md(f"| {pa} vs {po} | {sc}{'' if th is None else f' ({th:.2f})'} | {v} | " + ' | '.join(cells) + ' |')
    # chance-corrected recovery of cos and sp (as quoted in the manuscript wording)
    md('\n### RQ1 chance-corrected agreement (F-c)/(L-c), L-c >= 0.05, all foreign data\n')
    md('| Data | Set | cos | sp | top10 | nbr5 |'); md('|---|---|---|---|---|---|')
    for setname in ('all', 'common'):
        for v in XS:
            cells = []
            for m in ['cos', 'sp', 'top10', 'nbr5']:
                r = M[(M.set == setname) & (M['var'] == v) & (M.measure == m) & (M.arm == 'all')].iloc[0]
                cells.append(f"{fci((r['cc_0.05'], r['cc_0.05_lo'], r['cc_0.05_hi']), 2, False)} ({r['cc_0.05_n']})")
            md(f'| {v} | {setname} | ' + ' | '.join(cells) + ' |')


# ============================================================================ RQ2
def rq2_load(v, model, split):
    fn = src(v, f'rq2_{model}_{split}{tagof(v)}.jsonl')
    if not os.path.exists(fn): return None
    rows, seen = [], set()
    for l in open(fn, encoding='utf-8'):
        r = json.loads(l)
        if r['t'] in seen: continue
        seen.add(r['t'])
        d = {'t': r['t'], 'local': r['local']['ndcg'], 'tuned': r['tuned'], 'n_items': r['n_items']}
        for a in ('pool', 'weight'):
            if a in r: d[a] = r[a]['ndcg']
        if 'weight' in r: d['alpha'] = r['weight']['alpha']
        rows.append(d)
    return pd.DataFrame(rows)


def rq2():
    md('\n## RQ2: complete local data, pooling foreign data (NDCG@10)\n')
    out = []
    jobs = [('MostPop', 'random', ['pool']), ('ItemKNN', 'random', ['pool', 'weight']), ('EASE', 'random', ['pool']),
            ('ItemKNN', 'temporal', ['pool'])]
    for model, split, arms in jobs:
        D = {v: rq2_load(v, model, split) for v in VAR}
        D = {v: d for v, d in D.items() if d is not None}
        common = set.intersection(*[set(d.t) for d in D.values()])
        for setname in ('all', 'common'):
            for v, d in D.items():
                g = d if setname == 'all' else d[d.t.isin(common)]
                for a in arms:
                    gain = (g[a] - g.local).to_numpy(); ts = g.t.tolist()
                    lab = f'sens|rq2|{model}|{split}|{a}|{v}|{setname}'
                    med = csum(gain, ts, lab + '|med'); mean = csum(gain, ts, lab + '|mean', 'mean')
                    x = g[g.t != IN_AC]
                    mx = csum((x[a] - x.local).to_numpy(), x.t.tolist(), lab + '|meanexIN', 'mean')
                    r = dict(model=model, split=split, arm=a, var=v, set=setname, contexts=len(g), local_median=float(g.local.median()),
                             arm_median=float(g[a].median()), gain_median=med[0], gain_median_lo=med[1], gain_median_hi=med[2],
                             gain_mean=mean[0], gain_mean_lo=mean[1], gain_mean_hi=mean[2],
                             gain_mean_exIN=mx[0], gain_mean_exIN_lo=mx[1], gain_mean_exIN_hi=mx[2],
                             better=int((gain > 0.01).sum()), worse=int((gain < -0.01).sum()),
                             in_ac_gain=float((g[a] - g.local)[g.t == IN_AC].iloc[0]) if (g.t == IN_AC).any() else np.nan)
                    if a == 'weight':
                        r['alpha0'] = int((g.alpha == 0).sum()); r['alpha1'] = int((g.alpha == 1).sum())
                    out.append(r)
    S = pd.DataFrame(out); S.to_csv(f'{NOTES}/sens2_rq2.csv', index=False, lineterminator='\n')
    for setname in ('all', 'common'):
        md(f'\n### RQ2 ({"all contexts of each variant" if setname == "all" else "contexts common to all four data versions"})\n')
        md('| Model | Split | Arm | Data | Contexts | Local median | Gain median [CI] | Gain mean [CI] | Mean excl. IN A&C [CI] | >+0.01 | <-0.01 |')
        md('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in S[S.set == setname].iterrows():
            md(f"| {r.model} | {r.split} | {r.arm} | {r['var']} | {r.contexts} | {f(r.local_median, 3, False)} | "
               f"{fci((r.gain_median, r.gain_median_lo, r.gain_median_hi), 4)} | {fci((r.gain_mean, r.gain_mean_lo, r.gain_mean_hi), 4)} | "
               f"{fci((r.gain_mean_exIN, r.gain_mean_exIN_lo, r.gain_mean_exIN_hi), 4)} | {r.better} | {r.worse} |")
    w = S[(S.arm == 'weight') & (S.set == 'all')]
    md('\nWeight arm alpha choices (all contexts): ' + '; '.join(f"{r['var']}: alpha=0 in {int(r.alpha0)}, alpha=1 in {int(r.alpha1)} of {r.contexts}" for _, r in w.iterrows()))
    # per-context local scores, paper data vs variants (common contexts), ItemKNN random
    D = {v: rq2_load(v, 'ItemKNN', 'random') for v in VAR}
    W = pd.concat({v: d.set_index('t')[['local', 'pool']] for v, d in D.items() if d is not None}, axis=1)
    W.to_csv(f'{NOTES}/sens2_rq2_itemknn_random_per_context.csv', lineterminator='\n')
    md('\nItemKNN random, per-context local NDCG@10 change vs paper data (common contexts): ' + '; '.join(
        f"{v}: median {np.nanmedian(W[(v, 'local')] - W[('orig', 'local')]):+.4f}, largest drop {np.nanmin(W[(v, 'local')] - W[('orig', 'local')]):+.3f} "
        f"({(W[(v, 'local')] - W[('orig', 'local')]).idxmin()})" for v in VAR[1:] if (v, 'local') in W))
    G = pd.DataFrame({v: W[(v, 'pool')] - W[(v, 'local')] for v in VAR if (v, 'local') in W})
    G.to_csv(f'{NOTES}/sens2_rq2_itemknn_random_gain_per_context.csv', lineterminator='\n')
    md('Largest ItemKNN random pooling gains per data version (top 5): ')
    for v in G:
        md(f'- {v}: ' + ', '.join(f'{t} {x:+.3f}' for t, x in G[v].dropna().sort_values(ascending=False).head(5).items()))
    md('Category medians of the ItemKNN random pooling gain:')
    G2 = G.copy(); G2['category'] = [t.split(' ')[1] for t in G2.index]
    md('```\n' + G2.groupby('category').median().round(4).to_string() + '\n```')


# ============================================================================ UserKNN check on _nv and _nvc
def userknn():
    md('\n## UserKNN anomaly check on _nv and _nvc\n')
    VV = ['orig', '_nv', '_nvc']
    rq = {v: rq2_load(v, 'UserKNN', 'random') for v in VV}
    scs = {v: (pd.DataFrame([json.loads(l) for l in open(src(v, 'sc_UserKNN_random.jsonl'), encoding='utf-8')]) if os.path.exists(src(v, 'sc_UserKNN_random.jsonl')) else None) for v in VV}
    dgf = {'orig': 'sens_diag_orig.csv', '_nv': 'sens_diag_nv.csv', '_nvc': 'sens_diag_nvc.csv'}
    ctxs = ['de Arts_Crafts_and_Sewing', 'de Cell_Phones_and_Accessories', 'de Grocery_and_Gourmet_Food', 'de Industrial_and_Scientific',
            'de Toys_and_Games', 'in Musical_Instruments', 'jp Cell_Phones_and_Accessories', 'mx Office_Products', 'uk Electronics',
            'uk Industrial_and_Scientific', 'uk Office_Products']
    rows = []
    for t in ctxs:
        r = {'t': t}
        for v in VV:
            sc = scs[v]
            s = sc[(sc.t == t) & sc.err.isna() & (sc.foreign == 0) & (sc.fb == 0)] if sc is not None else pd.DataFrame(columns=['arm', 'f', 'ndcg'])
            full = s[s.arm == 'F'].ndcg; p5 = s[(s.arm == 'P') & (s.f == 0.5)]
            r[f'{v}_full'] = float(full.iloc[0]) if len(full) else np.nan
            r[f'{v}_P50'] = float(p5.ndcg.mean()) if len(p5) else np.nan
            r[f'{v}_P50_seeds'] = int(len(p5))
            o = rq[v]
            r[f'{v}_rq2_local'] = float(o[o.t == t].local.iloc[0]) if o is not None and (o.t == t).any() else np.nan
            dg = pd.read_csv(f'{NOTES}/{dgf[v]}'); dg = dg[(dg.market + ' ' + dg.category) == t]
            r[f'{v}_jac'] = float(dg.jac_mass.iloc[0]) if len(dg) else np.nan
        rows.append(r)
    U = pd.DataFrame(rows); U.to_csv(f'{NOTES}/sens2_userknn_check.csv', index=False, lineterminator='\n')
    md('| Context | ' + ' | '.join(f'{v}: complete | {v}: 50% patterns (seeds) | {v}: Jaccard>=0.8 mass' for v in VV) + ' |')
    md('|---|' + '---|---|---|' * len(VV))
    for _, r in U.iterrows():
        md(f'| {r.t} | ' + ' | '.join(f"{f(r[v + '_full'], 3, False)} | {f(r[v + '_P50'], 3, False)} ({int(r[v + '_P50_seeds'])}) | {f(r[v + '_jac'], 2, False)}" for v in VV) + ' |')
    for v in VV:
        ok = U.dropna(subset=[f'{v}_full', f'{v}_P50'])
        md(f"Anomaly (50% subsample > complete + 0.01), {v}: {int((ok[v + '_P50'] - ok[v + '_full'] > 0.01).sum())} of {len(ok)} contexts present"
           f" ({', '.join(ok[ok[v + '_P50'] - ok[v + '_full'] > 0.01].t)}); median complete-data local over these: {ok[v + '_full'].median():.3f}")
    a = rq['orig'].set_index('t').local
    for v in ('_nv', '_nvc'):
        if rq[v] is None: continue
        b = rq[v].set_index('t').local; j = a.index.intersection(b.index)
        md(f"UserKNN local (random), orig vs {v} on {len(j)} common contexts: median {a[j].median():.3f} vs {b[j].median():.3f}; "
           f"median change {(b[j] - a[j]).median():+.3f}; contexts dropping by > 0.10: {int(((b[j] - a[j]) < -0.10).sum())}; all {v} contexts: median {b.median():.3f} ({len(b)})")


# ============================================================================ Launch
def launch_load(v):
    rows = []
    for m in ('ItemKNN', 'EASE', 'MostPop', 'Random'):
        fn = src(v, f'l3_{m}_random.jsonl')
        if not os.path.exists(fn): continue
        for l in open(fn, encoding='utf-8'):
            r = json.loads(l); x = r['zf']
            rows.append({'model': m, 't': r['t'], 'seed': r['seed'], 'foreign': r['foreign'], 'L': x['launch_ndcg@10'],
                         'A': x['all_ndcg@10'], 'tuned': x['tuned']})
    if not rows: return None
    df = pd.DataFrame(rows).drop_duplicates(['model', 't', 'seed', 'foreign'])
    df = df[df.groupby(['model', 't', 'seed']).foreign.transform('nunique') == 2]
    W = df.pivot_table(index=['model', 't', 'seed'], columns='foreign', values=['L', 'A'])
    W.columns = [f'{a}_{"F" if b else "L"}' for a, b in W.columns]; W = W.reset_index()
    ch = df[(df.model == 'Random') & (df.foreign == 0)][['t', 'seed', 'L']].rename(columns={'L': 'chance'})
    W = W.merge(ch, on=['t', 'seed'])
    W['dF'] = W.L_F - W.chance; W['dL'] = W.L_L - W.chance; W['gain'] = W.L_F - W.L_L; W['dAll'] = W.A_F - W.A_L
    return W.groupby(['model', 't'], as_index=False).mean(numeric_only=True)


def launch():
    md('\n## Product launch (launch3, random split, zero-feedback tuning)\n')
    C = {v: launch_load(v) for v in VAR}
    C = {v: c for v, c in C.items() if c is not None}
    out = []
    for setname in ('all', 'common'):
        for m in ('ItemKNN', 'EASE', 'MostPop', 'Random'):
            sets = [set(c[c.model == m].t) for c in C.values()]
            common = set.intersection(*sets)
            for v, c in C.items():
                s = c[c.model == m]
                if setname == 'common': s = s[s.t.isin(common)]
                lab = f'sens|l3|{m}|{v}|{setname}'
                d = csum(s.dF, s.t, lab + '|dF'); g = csum(s.gain, s.t, lab + '|gain'); a = csum(s.dAll, s.t, lab + '|dAll')
                out.append(dict(model=m, var=v, set=setname, contexts=len(s), ndcg_foreign=float(s.L_F.median()), ndcg_local=float(s.L_L.median()),
                                chance=float(s.chance.median()), dF=d[0], dF_lo=d[1], dF_hi=d[2], above_F=int((s.dF > 0.01).sum()),
                                above_L=int((s.dL > 0.01).sum()), gain=g[0], gain_lo=g[1], gain_hi=g[2], dAll=a[0], dAll_lo=a[1], dAll_hi=a[2]))
    S = pd.DataFrame(out); S.to_csv(f'{NOTES}/sens2_launch.csv', index=False, lineterminator='\n')
    for setname in ('all', 'common'):
        md(f'\n### Launch ({"all contexts of each variant" if setname == "all" else "contexts common to all data versions, per model"})\n')
        md('| Model | Data | Contexts | NDCG@10 launched, +foreign | local only | chance | +foreign - chance [CI] | >chance+0.01: foreign (local) | Gain over local [CI] | Overall change [CI] |')
        md('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in S[S.set == setname].iterrows():
            if r.model == 'Random': continue
            md(f"| {r.model} | {r['var']} | {r.contexts} | {f(r.ndcg_foreign, 3, False)} | {f(r.ndcg_local, 3, False)} | {f(r.chance, 3, False)} | "
               f"{fci((r.dF, r.dF_lo, r.dF_hi))} | {r.above_F} ({r.above_L}) | {fci((r.gain, r.gain_lo, r.gain_hi))} | {fci((r.dAll, r.dAll_lo, r.dAll_hi))} |")


# ============================================================================ Scarcity
def sc_load(v):
    fn = src(v, 'sc_ItemKNN_random.jsonl')
    if not os.path.exists(fn): return None
    r = pd.DataFrame([json.loads(l) for l in open(fn, encoding='utf-8')])
    r = r[r.err.isna() & (r.fb == 0)]
    F = r[(r.arm == 'F') & (r.foreign == 0)].set_index('t').ndcg.rename('full')
    s = r[(r.seed == 0) & (r.f == 0.1) & r.arm.isin(['P', 'R', 'B'])]
    p = s.pivot_table(index=['t', 'arm'], columns='foreign', values='ndcg').dropna().reset_index().rename(columns={0: 'y0', 1: 'y1'})
    p = p.merge(F, left_on='t', right_index=True)
    p['gain'] = p.y1 - p.y0; p['gap'] = p.full - p.y0
    p['G'] = np.where(p.gap > 0.01, p.gain / p.gap, np.nan)
    return p


def scarcity():
    md('\n## Scarcity (ItemKNN, random split, f = 0.10, subsample seed 0)\n')
    P = {v: sc_load(v) for v in VAR}
    P = {v: p for v, p in P.items() if p is not None}
    out = []
    for setname in ('all', 'common'):
        for arm in ('P', 'R', 'B'):
            common = set.intersection(*[set(p[p.arm == arm].t) for p in P.values()])
            for v, p in P.items():
                s = p[p.arm == arm]
                if setname == 'common': s = s[s.t.isin(common)]
                lab = f'sens|sc|{arm}|{v}|{setname}'
                g = csum(s.gain, s.t, lab + '|gain'); gm = csum(s.gain, s.t, lab + '|gainmean', 'mean')
                sg = s.dropna(subset=['G']); G = csum(sg.G, sg.t, lab + '|G')
                out.append(dict(arm=arm, var=v, set=setname, contexts=len(s), local=float(s.y0.median()), foreign=float(s.y1.median()),
                                full=float(s.full.median()), gain=g[0], gain_lo=g[1], gain_hi=g[2], gain_mean=gm[0], gain_mean_lo=gm[1], gain_mean_hi=gm[2],
                                G=G[0], G_lo=G[1], G_hi=G[2], G_n=len(sg), better=int((s.gain > 0.01).sum()), worse=int((s.gain < -0.01).sum())))
    S = pd.DataFrame(out); S.to_csv(f'{NOTES}/sens2_scarcity.csv', index=False, lineterminator='\n')
    for setname in ('all', 'common'):
        md(f'\n### Scarcity ({"all contexts of each variant" if setname == "all" else "contexts common to all data versions"})\n')
        md('| Arm | Data | Contexts | Local (10%) | +foreign | Complete local | Gain median [CI] | Gain mean [CI] | G median [CI] (n) | >+0.01 | <-0.01 |')
        md('|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in S[S.set == setname].iterrows():
            md(f"| {r.arm} | {r['var']} | {r.contexts} | {f(r.local, 3, False)} | {f(r.foreign, 3, False)} | {f(r.full, 3, False)} | "
               f"{fci((r.gain, r.gain_lo, r.gain_hi))} | {fci((r.gain_mean, r.gain_mean_lo, r.gain_mean_hi))} | {fci((r.G, r.G_lo, r.G_hi), 2)} ({r.G_n}) | {r.better} | {r.worse} |")


if __name__ == '__main__':
    parts = sys.argv[1:] if len(sys.argv) > 1 else ['rq1', 'rq2', 'userknn', 'launch', 'scarcity']
    for p in parts:
        globals()[p]()
    open(f'{NOTES}/sens2_tables{"" if len(sys.argv) == 1 else "_" + "_".join(parts)}.md', 'w', encoding='utf-8', newline='\n').write(MD.getvalue())
