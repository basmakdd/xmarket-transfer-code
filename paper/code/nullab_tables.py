"""Tables for the model-level null ablation of RQ2 (nullab_run.py).

Reads paper/raw/nullab_<MODEL>_random{,_nvc}.jsonl.
Writes paper/tables_nullab.tex, paper/results/nullab_summary.csv, nullab_per_context.csv,
nullab_null_diagnostics.csv, nullab_rq2_consistency.csv, and prints every reported number
(also to paper/results/nullab_tables_output.txt).

Fixed in advance (task specification, before any result was looked at):
* Per context (random split, XM donors): NDCG@10 of local, pool and pool_null (mean of 3 null draws).
  gain_pool = pool - local, gain_null = pool_null - local, relational = pool - pool_null.
* Context-level 95% CIs: boot.summarize (two-way cluster bootstrap over countries x categories), for
  both the mean and the median of every quantity; one label per (data version, model), so the three
  quantities of a cell share their bootstrap replicates (boot.cboot adds the statistic to the seed).
* 'Local only' column: median over contexts of the local NDCG@10 (as in the RQ2 table).
* Counts: contexts with relational > +0.01 and < -0.01 (also for the two gains).
* Reporting rule: relational information is "supported" for a (data version, model) cell iff the 95% CI
  of mean(pool - pool_null) excludes 0 and is positive (lower bound > 0). Nothing else is used for the verdict.
* MostPop is a sanity check only: pool_null must equal pool up to ties (popularity counts are preserved).
* Also reported (as in the RQ2 table): the means without IN Arts & Crafts.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import boot

RES = os.path.join(REPO, 'paper', 'results'); os.makedirs(RES, exist_ok=True)
TEX = os.path.join(REPO, 'paper', 'tables_nullab.tex')
MODELS = ['ItemKNN', 'EASE', 'MostPop']
VERSIONS = [('', 'Original'), ('_nvc', 'Variants merged')]
IN_AC = 'in Arts_Crafts_and_Sewing'
THR = 0.01
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def load():
    rows, diag = [], []
    for tag, _ in VERSIONS:
        for m in MODELS:
            f = os.path.join(PAPER_RAW, f'nullab_{m}_random{tag}.jsonl')
            if not os.path.exists(f): say('missing', f); continue
            seen = set()
            for l in open(f, encoding='utf-8'):
                r = json.loads(l)
                if r['t'] in seen: continue
                seen.add(r['t'])
                mk, c = r['t'].split(' ', 1)
                rows.append({'tag': tag, 'model': m, 't': r['t'], 'market': mk, 'category': c, 'tuned': r['tuned'],
                             'n_items': r['n_items'], 'n_donors': len(r['donors']), 'sec': r['sec'],
                             'local': r['local']['ndcg'], 'pool': r['pool']['ndcg'], 'pool_null': r['pool_null']['ndcg'],
                             'null_sd': r['pool_null']['ndcg_sd'], 'k_local': r['local']['k'], 'k_pool': r['pool']['k'],
                             'k_null': ','.join(str(x['k']) for x in r['pool_null_draws'])})
                for dr, x in enumerate(r['pool_null_draws']):
                    for inf in x['null_info']:
                        diag.append(dict(inf, tag=tag, model=m, t=r['t'], draw=dr))
    d = pd.DataFrame(rows)
    d['gain_pool'] = d.pool - d.local
    d['gain_null'] = d.pool_null - d.local
    d['relational'] = d.pool - d.pool_null
    return d, pd.DataFrame(diag)


def fmt(x, nd=4):
    if abs(x) < 0.5 * 10 ** -nd: x = 0.0
    return f'{x:+.{nd}f}'.replace('-', '$-$')


def main():
    d, diag = load()
    d.to_csv(os.path.join(RES, 'nullab_per_context.csv'), index=False, lineterminator='\n')
    S = []
    for tag, vname in VERSIONS:
        for m in MODELS:
            g = d[(d.tag == tag) & (d.model == m)]
            if not len(g): continue
            base = {'data': vname, 'tag': tag, 'model': m, 'contexts': len(g), 'local_median': g.local.median(), 'in_ac_present': bool((g.t == IN_AC).any()),
                    'untuned_contexts': int((~g.tuned).sum())}
            for q in ('gain_pool', 'gain_null', 'relational'):
                r = dict(base, quantity=q)
                for st in ('mean', 'median'):
                    v, lo, hi = boot.summarize(g[q].values, g.market.values, g.category.values, stat=st,
                                               label=f'nullab|{tag}|{m}')
                    r[st], r[st + '_lo'], r[st + '_hi'] = v, lo, hi
                ex = g[g.t != IN_AC]
                r['mean_excl_in_ac'] = ex[q].mean()
                r['n_above'] = int((g[q] > THR).sum()); r['n_below'] = int((g[q] < -THR).sum())
                if q == 'relational':
                    r['supported'] = bool(r['mean_lo'] > 0)
                S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(RES, 'nullab_summary.csv'), index=False, lineterminator='\n')
    say('=== summary (NDCG@10; gains per context; CIs = context-level two-way cluster bootstrap, 95%)')
    for _, r in S.iterrows():
        say(f"{r.data:16s} {r.model:8s} {r.quantity:10s} n={r.contexts:2d} local_med={r.local_median:.4f} "
            f"median={r['median']:+.4f} [{r.median_lo:+.4f},{r.median_hi:+.4f}] mean={r['mean']:+.4f} [{r.mean_lo:+.4f},{r.mean_hi:+.4f}] "
            f"mean_exIN={r.mean_excl_in_ac:+.4f} >+{THR}:{r.n_above} <-{THR}:{r.n_below}"
            + (f"  supported={r.supported}" if r.quantity == 'relational' else ''))
    # share of the pooling gain kept by the null
    say('=== share of the mean pooling gain retained by the null (mean gain_null / mean gain_pool)')
    for (vn, m), g in S.groupby(['data', 'model'], sort=False):
        gp = g[g.quantity == 'gain_pool']['mean'].iloc[0]; gn = g[g.quantity == 'gain_null']['mean'].iloc[0]
        say(f'{vn:16s} {m:8s} {gn / gp:.2f}' if gp != 0 else f'{vn} {m} n/a')
    # MostPop sanity
    mp = d[d.model == 'MostPop']
    if len(mp):
        say('=== MostPop sanity: max |pool_null - pool| per data version:',
            {t: float(g.relational.abs().max()) for t, g in mp.groupby('tag')},
            ' contexts with |diff|>1e-9:', {t: int((g.relational.abs() > 1e-9).sum()) for t, g in mp.groupby('tag')})
    # null diagnostics
    if len(diag):
        dg = diag.groupby(['tag', 'model']).apply(lambda x: pd.Series({
            'donor_matrices': len(x), 'all_counts_preserved': bool(x.ok_counts.all()), 'max_dups': int(x.dups.max()),
            'min_trades_per_user': float((x.trades / x.users).min()),
            'kept_pairs_edge_weighted': float(np.average(x.kept_pairs, weights=x.edges)),
            'kept_pairs_median': float(x.kept_pairs.median())}), include_groups=False).reset_index()
        dg.to_csv(os.path.join(RES, 'nullab_null_diagnostics.csv'), index=False, lineterminator='\n')
        say('=== null diagnostics'); say(dg.to_string(index=False))
        say('=== draw-to-draw SD of pool_null NDCG@10 (median over contexts):',
            d.groupby(['tag', 'model']).null_sd.median().round(5).to_dict())
    # consistency of local/pool with the RQ2 runs (same code path, deterministic)
    C = []
    for tag, _ in VERSIONS:
        for m in MODELS:
            f = os.path.join(PAPER_RAW, f'rq2_{m}_random.jsonl') if tag == '' else os.path.join(PAPER_RAW, 'sens_nvc', f'rq2_{m}_random_nvc.jsonl')
            if not os.path.exists(f): continue
            q = {}
            for l in open(f, encoding='utf-8'):
                r = json.loads(l)
                if 'pool' in r: q[r['t']] = (r['local']['ndcg'], r['pool']['ndcg'])
            g = d[(d.tag == tag) & (d.model == m)]
            both = [t for t in g.t if t in q]
            dl = max([abs(g[g.t == t].local.iloc[0] - q[t][0]) for t in both] or [np.nan])
            dp = max([abs(g[g.t == t].pool.iloc[0] - q[t][1]) for t in both] or [np.nan])
            C.append({'tag': tag, 'model': m, 'contexts_nullab': len(g), 'contexts_rq2': len(q), 'matched': len(both),
                      'only_rq2': ';'.join(sorted(set(q) - set(g.t))), 'only_nullab': ';'.join(sorted(set(g.t) - set(q))),
                      'max_abs_diff_local': dl, 'max_abs_diff_pool': dp})
    C = pd.DataFrame(C); C.to_csv(os.path.join(RES, 'nullab_rq2_consistency.csv'), index=False, lineterminator='\n')
    say('=== consistency with RQ2 runs'); say(C.assign(only_rq2=C.only_rq2.str.count(';') + (C.only_rq2 != ''), only_nullab=C.only_nullab.str.count(';') + (C.only_nullab != '')).to_string(index=False))
    for tag, _ in VERSIONS:
        a = set(d[(d.tag == tag) & (d.model == 'ItemKNN')].t); e = set(d[(d.tag == tag) & (d.model == 'EASE')].t)
        say(f'=== {tag or "orig"}: ItemKNN contexts not in EASE (catalogue > 4000):', sorted(a - e))
    say('=== runtime (s): per (tag, model) total and max', d.groupby(['tag', 'model']).sec.agg(['sum', 'max']).round(0).to_dict('index'))
    tex(S, mp)
    open(os.path.join(RES, 'nullab_tables_output.txt'), 'w', encoding='utf-8', newline='\n').write('\n'.join(OUT) + '\n')


QN = {'gain_pool': 'Pool $-$ local', 'gain_null': 'Null pool $-$ local', 'relational': 'Pool $-$ null pool'}


def tex(S, mp):
    L = ['% Generated by paper/code/nullab_tables.py -- do not edit by hand', '',
         r'\begin{table}[H]', r'\footnotesize',
         r'\caption{Null-model ablation of pooling (random split, NDCG@10): foreign data with intact co-interaction patterns versus foreign data randomised to keep only product popularity and user activity.\label{tab:nullab}}',
         r'\begin{adjustwidth}{-\extralength}{0cm}', r'\centering', r'\begin{threeparttable}',
         r'\begin{tabular}{lllcccccccc}', r'\toprule',
         r'Data & Model & Difference & Contexts & Local only & Median [95\% CI] & Mean [95\% CI] & Mean excl.\ IN A\&C & $>$+0.01 & $<-$0.01 & Supported \\',
         r'\midrule']
    first = True
    for vn in [v for _, v in VERSIONS]:
        g0 = S[S.data == vn]
        if not len(g0): continue
        if not first: L.append(r'\midrule')
        first = False
        for m in MODELS:
            g = g0[g0.model == m]
            for _, r in g.iterrows():
                sup = ('--' if m == 'MostPop' else ('yes' if r.supported else 'no')) if r.quantity == 'relational' else ''
                L.append(f'{vn} & {m} & {QN[r.quantity]} & {r.contexts} & {r.local_median:.3f} & '
                         f"{fmt(r['median'])} [{fmt(r.median_lo, 3)}, {fmt(r.median_hi, 3)}] & "
                         f"{fmt(r['mean'])} [{fmt(r.mean_lo, 3)}, {fmt(r.mean_hi, 3)}] & {fmt(r.mean_excl_in_ac) if r.in_ac_present else '--'} & "
                         f'{r.n_above} & {r.n_below} & {sup} \\\\')
    mpd = {t: float(g.relational.abs().max()) for t, g in mp.groupby('tag')} if len(mp) else {}
    mps = ', '.join(f'{v}: ' + (r'$<10^{-15}$' if mpd[t] < 1e-15 else f'{mpd[t]:.1e}') for t, v in VERSIONS if t in mpd)
    L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}\footnotesize',
          r'\item Per target context, the local model, the pooled model (local plus all same-category foreign training data) and the null pooled model are each retuned on the validation users. '
          r"In the null pooled model each donor country's training user $\times$ product matrix is randomised with the curveball algorithm (at least five trades per user), "
          r"which keeps every product's interaction count and every user's activity within that donor but breaks which products co-occur in a user's history; "
          r"its score is the mean over three independent randomisations. ``Pool $-$ null pool'' is the part of the pooling gain that depends on the foreign co-interaction patterns. "
          r'CIs: two-way cluster bootstrap over countries and categories, 95\%. Rule fixed in advance: the contribution of co-interaction patterns is \emph{supported} if the 95\% CI of the mean of pool $-$ null pool lies entirely above 0. '
          r'MostPop is a sanity check (largest absolute difference between pool and null pool: ' + mps + r'). '
          r'Variants merged: product variants merged within each market $\times$ category context. EASE is not fitted on catalogues of more than 4000 products; IN Arts \& Crafts is not a context of the variant-merged data. MostPop: no verdict (sanity check).',
          r'\end{tablenotes}', r'\end{threeparttable}', r'\end{adjustwidth}', r'\end{table}', '']
    open(TEX, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
    say('wrote', os.path.relpath(TEX, REPO))


if __name__ == '__main__':
    main()
