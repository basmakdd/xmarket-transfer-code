"""RQ2 tables: complete-data transfer with local / pool / weight / greedy arms.

Reads paper/raw/rq2_*.jsonl (rq2_run.py) and paper/raw/rq2_peruser/*.npz.
Writes paper/tables_rq2.tex, paper/results/rq2_*.csv, and prints every reported number.

Fixed in advance (task specification):
* gain = arm - local NDCG@10 per context; better/worse = gain > +thr / < -thr, thr in {0.005, 0.01, 0.02}
  (0.01 is the headline threshold).
* CIs: boot.summarize, context-level two-way cluster bootstrap (countries x categories), 95%.
  Repeated split seeds are averaged within a context before any CI.
* IN Arts & Crafts: every mean is reported with and without it.
* History: buckets 1, 2, 3-4, 5+ local training interactions of the test user; a (context,
  bucket) cell is dropped if it has < 20 users. Cell value = mean per-user gain; table rows give
  the mean over contexts of the per-context mean gain (with CI) and the median over contexts.
* Short-history test: per context d = mean gain(users with 1-2) - mean gain(users with 3+),
  each group needing >= 20 users. Rule fixed in advance: the short-history advantage is called
  "supported" only if the 95% cluster-bootstrap CI of the MEAN of d excludes 0 (the CI of the
  median of d is reported as well). The old Wilcoxon signed-rank p is shown for comparison only.
* Common-context set: contexts with results for every main split (random, temporal, calendar)
  and every main model (MostPop, ItemKNN, EASE, UserKNN).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, glob, math
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from scipy import stats
import boot

NOTES = os.path.join(REPO, 'paper', 'results'); os.makedirs(NOTES, exist_ok=True)
TEX = os.path.join(REPO, 'paper', 'tables_rq2.tex')
IN_AC = 'in Arts_Crafts_and_Sewing'
MODELS = ['MostPop', 'ItemKNN', 'EASE', 'UserKNN', 'Random']
ARMS = ['pool', 'weight', 'greedy']
THR = (0.005, 0.01, 0.02)
VARIANTS = [('random', ''), ('temporal', ''), ('calendar', ''), ('random', '_seed1'), ('random', '_seed2'),
            ('random', '_r4'), ('temporal', '_r4')]
VNAME = {('random', ''): 'Random', ('temporal', ''): 'Temporal', ('calendar', ''): 'Calendar',
         ('random', '_seed1'): 'Random (seed 1)', ('random', '_seed2'): 'Random (seed 2)',
         ('random', '_r4'): 'Random, ratings>=4', ('temporal', '_r4'): 'Temporal, ratings>=4'}
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(PAPER_RAW, 'rq2_*.jsonl'))):
        if f.endswith('_grid.jsonl'): continue
        seen = set()
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            if r['t'] in seen: continue   # resumable runs never duplicate, but be safe
            seen.add(r['t'])
            m, c = r['t'].split(' ', 1)
            d = {'t': r['t'], 'market': m, 'category': c, 'split': r['split'], 'tag': r['tag'], 'model': r['model'],
                 'tuned': r['tuned'], 'n_items': r['n_items'], 'n_eval': r['n_eval_users'], 'n_donors': len(r['donors']),
                 'local': r['local']['ndcg'], 'local_recall': r['local']['recall'], 'sec': r['sec']}
            for a in ARMS:
                if a in r:
                    d[a] = r[a]['ndcg']; d[a + '_recall'] = r[a]['recall']
            if 'weight' in r: d['alpha'] = r['weight']['alpha']
            if 'greedy' in r: d['n_selected'] = len(r['greedy']['selected']); d['selected'] = ','.join(r['greedy']['selected'])
            rows.append(d)
    df = pd.DataFrame(rows)
    for a in ARMS:
        if a in df: df['g_' + a] = df[a] - df['local']
    return df


def ci(vals, mk, ct, stat, label):
    vals = np.asarray(vals, float)
    if np.sum(~np.isnan(vals)) < 2: return (float(np.nanmean(vals)) if len(vals) else np.nan, np.nan, np.nan)
    return boot.summarize(vals, mk, ct, stat=stat, label=label)


def fmt(x, d=4, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)): return '--'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def summary_rows(df, label):
    """One row per (variant, model, arm) with all Table-2 statistics."""
    out = []
    for (sp, tag) in VARIANTS:
        for model in MODELS:
            g = df[(df.split == sp) & (df.tag == tag) & (df.model == model)]
            if not len(g): continue
            base = {'variant': VNAME[(sp, tag)], 'split': sp, 'tag': tag, 'model': model, 'contexts': len(g),
                    'local_median': float(g.local.median())}
            arms = [a for a in ARMS if 'g_' + a in g and g['g_' + a].notna().any()]
            if model == 'Random' or not arms:
                out.append(dict(base, arm='local only')); continue
            for a in arms:
                h = g[g['g_' + a].notna()]; v = h['g_' + a].to_numpy()
                lab = f'rq2|{label}|{sp}{tag}|{model}|{a}'
                med = ci(v, h.market, h.category, 'median', lab + '|all')
                mean = ci(v, h.market, h.category, 'mean', lab + '|all')
                x = h[h.t != IN_AC]; vx = x['g_' + a].to_numpy()
                mean_x = ci(vx, x.market, x.category, 'mean', lab + '|exIN')
                r = dict(base, arm=a, contexts=len(h), local_median=float(h.local.median()),
                         gain_median=med[0], gain_median_lo=med[1], gain_median_hi=med[2],
                         gain_mean=mean[0], gain_mean_lo=mean[1], gain_mean_hi=mean[2],
                         contexts_exIN=len(x), gain_mean_exIN=mean_x[0], gain_mean_exIN_lo=mean_x[1], gain_mean_exIN_hi=mean_x[2],
                         in_ac_gain=float(h.loc[h.t == IN_AC, 'g_' + a].iloc[0]) if (h.t == IN_AC).any() else np.nan)
                for th in THR:
                    r[f'better_{th}'] = int((v > th).sum()); r[f'worse_{th}'] = int((v < -th).sum())
                if a == 'weight':
                    for al in (0.0, 0.1, 0.25, 0.5, 1.0): r[f'alpha_{al}'] = int((h.alpha == al).sum())
                    r['alpha_untuned'] = int((~h.tuned).sum())
                if a == 'greedy':
                    r['greedy_none'] = int((h.n_selected == 0).sum()); r['greedy_all'] = int((h.n_selected == h.n_donors).sum())
                    r['greedy_median_selected'] = float(h.n_selected.median())
                out.append(r)
    return pd.DataFrame(out)


def print_summary(S, title):
    say(f'\n=== {title} ===')
    for _, r in S.iterrows():
        if r.arm == 'local only':
            say(f"{r.variant:22s} {r.model:8s} n={r.contexts:3d} local_med={r.local_median:.3f} (local only)"); continue
        s = (f"{r.variant:22s} {r.model:8s} {r.arm:6s} n={r.contexts:3d} local_med={r.local_median:.3f} "
             f"gain med {fmt(r.gain_median)} mean {fmt(r.gain_mean)} [{fmt(r.gain_mean_lo,3)}, {fmt(r.gain_mean_hi,3)}] "
             f"exIN {fmt(r.gain_mean_exIN)} [{fmt(r.gain_mean_exIN_lo,3)}, {fmt(r.gain_mean_exIN_hi,3)}] "
             f"b/w .005:{int(r['better_0.005'])}/{int(r['worse_0.005'])} .01:{int(r['better_0.01'])}/{int(r['worse_0.01'])} .02:{int(r['better_0.02'])}/{int(r['worse_0.02'])}")
        if r.arm == 'weight':
            s += ' alpha ' + ' '.join(f"{al}:{int(r[f'alpha_{al}'])}" for al in (0.0, 0.1, 0.25, 0.5, 1.0)) + f" (untuned->1: {int(r.alpha_untuned)})"
        if r.arm == 'greedy':
            s += f" greedy none/all={int(r.greedy_none)}/{int(r.greedy_all)} med_sel={r.greedy_median_selected}"
        say(s)


def tex_table2(S, caption, label):
    L = [r'\begin{table}[H]', r'\footnotesize', r'\caption{' + caption + r'\label{' + label + r'}}',
         r'\begin{adjustwidth}{-\extralength}{0cm}', r'\centering', r'\begin{threeparttable}',
         r'\begin{tabular}{lllccccccc}', r'\toprule',
         r'Split & Model & Arm & Contexts & Local only & Gain (median) & Gain (mean, 95\% CI) & Mean excl.\ IN A\&C & Better $>$0.01 & Worse $>$0.01 \\',
         r'\midrule']
    prev = None
    for _, r in S.iterrows():
        if prev is not None and r.variant != prev: L.append(r'\midrule')
        prev = r.variant
        v = r.variant.replace('>=', r'$\geq$')
        if r.arm == 'local only':
            L.append(f'{v} & {r.model} & local only & {r.contexts} & {r.local_median:.3f} & -- & -- & -- & -- & -- \\\\'); continue
        L.append(f"{v} & {r.model} & {r.arm} & {r.contexts} & {r.local_median:.3f} & {fmt(r.gain_median)} & "
                 f"{fmt(r.gain_mean)} [{fmt(r.gain_mean_lo,3)}, {fmt(r.gain_mean_hi,3)}] & {fmt(r.gain_mean_exIN)} & "
                 f"{int(r['better_0.01'])} & {int(r['worse_0.01'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\begin{tablenotes}\footnotesize',
          r'\item Gain = arm minus local-only NDCG@10 per context. pool: all same-category foreign training data at equal weight; '
          r'weight: foreign rows weighted by $\alpha\in\{0,0.1,0.25,0.5,1\}$, chosen with the hyperparameters on validation NDCG@10; '
          r'greedy: forward selection of donor countries on validation NDCG@10, then retuned. Every arm is retuned on validation data; '
          r'contexts without validation users use the first grid setting, $\alpha=1$ and all donors. 95\% two-way cluster-bootstrap intervals (countries $\times$ categories). '
          r'Counts at thresholds 0.005 and 0.02 are in the CSV files.',
          r'\end{tablenotes}', r'\end{threeparttable}', r'\end{adjustwidth}', r'\end{table}', '']
    return '\n'.join(L)


def history(tex):
    say('\n=== History analysis (random split, seed 0; local vs pool) ===')
    files = sorted(glob.glob(os.path.join(PAPER_RAW, 'rq2_peruser', '*.npz')))
    cells, drows = [], []
    for f in files:
        model, rest = os.path.basename(f)[:-4].split('_', 1)
        if model not in ('ItemKNN', 'EASE'): continue
        mk, cat = rest.split('_', 1); t = f'{mk} {cat}'
        z = np.load(f); h = z['hist']; g = z['pool'].astype(float) - z['local'].astype(float)
        b = np.select([h == 1, h == 2, (h >= 3) & (h <= 4), h >= 5], ['1', '2', '3-4', '5+'], 'other')
        for bk in ('1', '2', '3-4', '5+'):
            m = b == bk
            if m.sum() >= 20: cells.append((model, t, mk, cat, bk, int(m.sum()), float(g[m].mean())))
        s, l = (h >= 1) & (h <= 2), h >= 3
        if s.sum() >= 20 and l.sum() >= 20:
            drows.append((model, t, mk, cat, int(s.sum()), int(l.sum()), float(g[s].mean() - g[l].mean())))
        if (h < 1).any(): say('WARNING users with 0 local training interactions in', t, int((h < 1).sum()))
    C = pd.DataFrame(cells, columns=['model', 't', 'market', 'category', 'bucket', 'users', 'mean_gain'])
    Dd = pd.DataFrame(drows, columns=['model', 't', 'market', 'category', 'n_short', 'n_long', 'd'])
    C.to_csv(os.path.join(NOTES, 'rq2_history_cells.csv'), index=False, lineterminator='\n'); Dd.to_csv(os.path.join(NOTES, 'rq2_history_d.csv'), index=False, lineterminator='\n')
    rows = []
    for model in ('ItemKNN', 'EASE'):
        for bk in ('1', '2', '3-4', '5+'):
            x = C[(C.model == model) & (C.bucket == bk)]
            if not len(x): continue
            mean = ci(x.mean_gain, x.market, x.category, 'mean', f'rq2hist|{model}|{bk}|mean')
            med = ci(x.mean_gain, x.market, x.category, 'median', f'rq2hist|{model}|{bk}|median')
            xx = x[x.t != IN_AC]
            mx = ci(xx.mean_gain, xx.market, xx.category, 'mean', f'rq2hist|{model}|{bk}|mean|exIN')
            rows.append(dict(model=model, bucket=bk, contexts=len(x), users=int(x.users.sum()), median=med[0], median_lo=med[1], median_hi=med[2],
                             mean=mean[0], mean_lo=mean[1], mean_hi=mean[2], mean_exIN=mx[0], mean_exIN_lo=mx[1], mean_exIN_hi=mx[2]))
            say(f'{model:8s} bucket {bk:4s} contexts={len(x):3d} users={int(x.users.sum()):7d} median-over-contexts={fmt(med[0])} '
                f'mean-over-contexts-of-per-context-mean={fmt(mean[0])} [{fmt(mean[1],3)}, {fmt(mean[2],3)}]  excl IN A&C {fmt(mx[0])} [{fmt(mx[1],3)}, {fmt(mx[2],3)}]')
    H = pd.DataFrame(rows); H.to_csv(os.path.join(NOTES, 'rq2_fig6_history.csv'), index=False, lineterminator='\n')
    test = []
    for model in ('ItemKNN', 'EASE'):
        x = Dd[Dd.model == model]
        if len(x) < 2: continue
        mean = ci(x.d, x.market, x.category, 'mean', f'rq2d|{model}|mean')
        med = ci(x.d, x.market, x.category, 'median', f'rq2d|{model}|median')
        xx = x[x.t != IN_AC]; mx = ci(xx.d, xx.market, xx.category, 'mean', f'rq2d|{model}|mean|exIN')
        w = stats.wilcoxon(x.d)
        sup = 'supported' if (mean[1] > 0 or mean[2] < 0) else 'not supported'
        test.append(dict(model=model, contexts=len(x), short_more=int((x.d > 0).sum()), mean_d=mean[0], mean_lo=mean[1], mean_hi=mean[2],
                         median_d=med[0], median_lo=med[1], median_hi=med[2], mean_exIN=mx[0], mean_exIN_lo=mx[1], mean_exIN_hi=mx[2],
                         wilcoxon_p=float(w.pvalue), verdict=sup))
        say(f'{model:8s} d = gain(1-2) - gain(3+): contexts={len(x)} short gains more in {int((x.d > 0).sum())}/{len(x)}; '
            f'mean d {fmt(mean[0])} [{fmt(mean[1],4)}, {fmt(mean[2],4)}]; median d {fmt(med[0])} [{fmt(med[1],4)}, {fmt(med[2],4)}]; '
            f'excl IN A&C mean {fmt(mx[0])} [{fmt(mx[1],4)}, {fmt(mx[2],4)}]; Wilcoxon p={w.pvalue:.3f} (comparison only) -> {sup}')
    T = pd.DataFrame(test); T.to_csv(os.path.join(NOTES, 'rq2_history_test.csv'), index=False, lineterminator='\n')
    L = [r'\begin{table}[H]', r'\footnotesize', r'\caption{Gain from foreign data (pool minus local) by number of local training interactions of the test user (random split).\label{tab:history}}',
         r'\begin{adjustwidth}{-\extralength}{0cm}', r'\centering', r'\begin{threeparttable}', r'\begin{tabular}{lccccc}', r'\toprule',
         r'Model & Local interactions & Contexts & Users & Median over contexts & Mean over contexts of per-context mean (95\% CI) \\', r'\midrule']
    for model in ('ItemKNN', 'EASE'):
        for _, r in H[H.model == model].iterrows():
            L.append(f"{model} & {r.bucket.replace('-', '--')} & {r.contexts} & {r.users:,} & {fmt(r['median'])} & {fmt(r['mean'])} [{fmt(r.mean_lo,3)}, {fmt(r.mean_hi,3)}] \\\\")
        for _, r in T[T.model == model].iterrows():
            L.append(r'\multicolumn{6}{l}{' + f"{model}, within context, $d$ = gain(1--2) $-$ gain(3+): mean {fmt(r.mean_d)} [{fmt(r.mean_lo,4)}, {fmt(r.mean_hi,4)}], "
                     f"median {fmt(r.median_d)}; larger in {r.short_more}/{r.contexts} contexts; {r.verdict}" + r'} \\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}', r'\begin{tablenotes}\footnotesize',
          r'\item Per-user NDCG@10 difference, averaged within each (context, bucket) cell; cells with fewer than 20 users are dropped. '
          r'95\% two-way cluster-bootstrap intervals over contexts. The short-history advantage is called supported only if the interval for the mean of $d$ excludes zero (rule fixed in advance).',
          r'\end{tablenotes}', r'\end{threeparttable}', r'\end{adjustwidth}', r'\end{table}', '']
    tex.append('\n'.join(L))
    return H, T


def main():
    df = load()
    df.to_csv(os.path.join(NOTES, 'rq2_per_context.csv'), index=False, lineterminator='\n')
    tex = ['% Generated by paper/code/rq2_tables.py -- do not edit by hand', '']

    # ---------- accounting
    say('=== Accounting: contexts per variant x model ===')
    ctx = pd.read_parquet(os.path.join(REPO, 'results', 'contexts.parquet'))
    acc = df.groupby(['split', 'tag', 'model']).agg(contexts=('t', 'size'), untuned=('tuned', lambda s: int((~s).sum())),
                                                    has_ukbooks=('t', lambda s: 'uk Books' in set(s)), max_items=('n_items', 'max'),
                                                    total_sec=('sec', 'sum')).reset_index()
    say(acc.to_string()); acc.to_csv(os.path.join(NOTES, 'rq2_accounting.csv'), index=False, lineterminator='\n')

    # ---------- (1) Table 2 (complete local data)
    main_df = df[df.tag.isin(['', '_r4'])]
    S = summary_rows(df, 'all'); S.to_csv(os.path.join(NOTES, 'rq2_table2_all.csv'), index=False, lineterminator='\n')
    print_summary(S[S.tag == ''], 'Table 2, complete local data (all available contexts)')
    tex.append(tex_table2(S[S.tag == ''], 'Adding same-category foreign data with complete local data (NDCG@10): pooling, validation-tuned weighting and greedy source selection.', 'tab:full'))
    # Figure 5 data
    S[S.tag == ''].to_csv(os.path.join(NOTES, 'rq2_fig5_summary.csv'), index=False, lineterminator='\n')

    # common-context set
    main = df[(df.tag == '') & df.model.isin(['MostPop', 'ItemKNN', 'EASE', 'UserKNN'])]
    sets = [set(g.t) for _, g in main.groupby(['split', 'model'])]
    common = set.intersection(*sets) if sets else set()
    say(f'\nCommon-context set (all of random/temporal/calendar x MostPop/ItemKNN/EASE/UserKNN): {len(common)} contexts')
    Sc = summary_rows(df[df.t.isin(common) & (df.tag == '')], 'common')
    Sc.to_csv(os.path.join(NOTES, 'rq2_table2_common.csv'), index=False, lineterminator='\n')
    print_summary(Sc, f'Table 2, common-context set ({len(common)} contexts)')
    tex.append(tex_table2(Sc, f'As Table~\\ref{{tab:full}}, restricted to the {len(common)} contexts with results for every split and model.', 'tab:full_common'))

    # ---------- (5) contexts without validation users
    unt = sorted(set(df[(df.tag == '') & (~df.tuned)].t))
    say('\nContexts without validation users (tag ""):', unt)
    for (sp, tag), g in df[df.tag == ''].groupby(['split', 'tag']):
        say(f'  {sp}: untuned contexts', sorted(set(g[~g.tuned].t)))
    Su = summary_rows(df[(df.tag == '') & df.tuned], 'tunedonly')
    Su.to_csv(os.path.join(NOTES, 'rq2_table2_tuned_only.csv'), index=False, lineterminator='\n')
    print_summary(Su, 'Table 2 excluding contexts without validation users')
    grid_section(tex)

    # ---------- (2) split-seed variability
    seeds(df, tex)

    # ---------- (6) r4
    print_summary(S[S.tag == '_r4'], 'Ratings >= 4 variant (local vs pool)')
    tex.append(tex_table2(S[S.tag == '_r4'], 'Ratings $\\geq$4 only (random split): local versus pool, NDCG@10.', 'tab:full_r4'))

    # ---------- (3) history
    history(tex)

    # ---------- per-context matrix (Table A3) for ItemKNN pool / weight / greedy, random split
    for arm in ARMS:
        g = df[(df.split == 'random') & (df.tag == '') & (df.model == 'ItemKNN')]
        if 'g_' + arm not in g or g['g_' + arm].isna().all(): continue
        M = g.pivot(index='category', columns='market', values='g_' + arm)
        M.to_csv(os.path.join(NOTES, f'rq2_matrix_ItemKNN_{arm}.csv'), lineterminator='\n')
        L = [r'\begin{table}[H]', r'\caption{ItemKNN gain in NDCG@10 with complete local data, per context (random split, ' + arm + r' arm).}',
             r'\label{tab:mat_full' + ('' if arm == 'pool' else '_' + arm) + '}', r'\centering', r'\small',
             r'\begin{tabular}{l' + 'r' * len(M.columns) + '}', r'\toprule', 'Category & ' + ' & '.join(c.upper() for c in M.columns) + r' \\', r'\midrule']
        for cat, row in M.iterrows():
            L.append(cat.replace('_', ' ').replace(' and ', r' \& ') + ' & ' + ' & '.join('--' if pd.isna(v) else f'{v:+.3f}' for v in row) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
        tex.append('\n'.join(L))

    # ---------- (7) threshold justification
    say('\n=== Threshold justification (G5) ===')
    g = df[(df.split == 'random') & (df.tag == '') & (df.model == 'ItemKNN')]
    if len(g):
        med = g.local.median()
        say(f'0.01 as % of the median local ItemKNN NDCG@10 (random split, {len(g)} contexts, median {med:.4f}): {100 * 0.01 / med:.2f}%')
        lo, hi = 0.01 / 1.0 * 100, 0.01 / (1 / math.log2(11)) * 100
        say(f'Assumption: each user has one relevant test item and a gain moves it from outside the top 10 into the top 10 '
            f'(NDCG@10 of a hit at rank r is 1/log2(r+1)). +0.01 mean NDCG@10 then equals {lo:.1f} (all new hits at rank 1) to '
            f'{hi:.1f} (all at rank 10) extra top-10 hits per 100 users.')
        if 'pool_recall' in g:
            dr = g.pool_recall - g.local_recall; dn = g.pool - g.local
            slope = float(np.sum(dr * dn) / np.sum(dn * dn))
            say(f'Empirical (ItemKNN pool vs local, random split): least-squares slope through 0 of delta Recall@10 on delta NDCG@10 = {slope:.2f}, '
                f'i.e. +0.01 NDCG@10 ~ {100 * 0.01 * slope:.2f} extra percentage points of Recall@10 (~ extra hits per 100 single-item users).')

    open(TEX, 'w', encoding='utf-8', newline='\n').write('\n'.join(tex))
    open(os.path.join(NOTES, 'rq2_tables_output.txt'), 'w', encoding='utf-8', newline='\n').write('\n'.join(OUT) + '\n')
    say(f'\nwrote {TEX}')


def grid_section(tex):
    say('\n=== Every-grid-setting sensitivity (random split): IN Arts & Crafts and contexts without validation users ===')
    rows = []
    for f in sorted(glob.glob(os.path.join(PAPER_RAW, 'rq2_*_grid.jsonl'))):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            loc = {x['k']: x['ndcg'] for x in r['grid'] if x['arm'] == 'local'}
            pool = {x['k']: x['ndcg'] for x in r['grid'] if x['arm'] == 'pool'}
            for x in r['grid']:
                if x['arm'] == 'local':
                    k = x['k']
                    rows.append(dict(model=r['model'], split=r['split'], t=r['t'], k=k, params=json.dumps(x['params']), local=loc[k], pool=pool.get(k),
                                     gain_same=None if (pool.get(k) is None or loc[k] is None) else pool[k] - loc[k],
                                     local_pop_fallback=r.get('local_popularity_fallback')))
    if not rows: say('  (no grid runs yet)'); return
    G = pd.DataFrame(rows); G.to_csv(os.path.join(NOTES, 'rq2_grid_sensitivity.csv'), index=False, lineterminator='\n')
    for (model, t), g in G.groupby(['model', 't']):
        lv, pv = g.local.dropna(), g.pool.dropna()
        allpairs = [p - q for p in pv for q in lv]
        say(f'{model:8s} {t:30s} local {lv.min():.4f}..{lv.max():.4f}  pool {pv.min():.4f}..{pv.max():.4f}  '
            f'gain same setting {g.gain_same.min():+.4f}..{g.gain_same.max():+.4f}  any pair {min(allpairs):+.4f}..{max(allpairs):+.4f}  '
            f'grid[0] gain {g[g.k == 0].gain_same.iloc[0]:+.4f}  local popularity fallback {g.local_pop_fallback.iloc[0]:.4f} '
            f'(pool minus fallback, best pool {pv.max() - g.local_pop_fallback.iloc[0]:+.4f}, grid[0] pool {g[g.k == 0].pool.iloc[0] - g.local_pop_fallback.iloc[0]:+.4f})')
    L = [r'\begin{table}[H]', r'\footnotesize', r'\caption{Contexts without validation users (random split): pooling gain (pool minus local NDCG@10) over every grid setting.\label{tab:untuned_grid}}',
         r'\centering', r'\begin{tabular}{llcccc}', r'\toprule', r'Model & Context & Gain at grid[0] & Gain, same setting (min..max) & Gain, any pair of settings (min..max) & Local popularity fallback \\', r'\midrule']
    for (model, t), g in G.groupby(['model', 't']):
        lv, pv = g.local.dropna(), g.pool.dropna(); allpairs = [p - q for p in pv for q in lv]
        L.append(f"{model} & {t.replace('_', ' ').replace(' and ', ' \\& ')} & {g[g.k == 0].gain_same.iloc[0]:+.3f} & {g.gain_same.min():+.3f}..{g.gain_same.max():+.3f} & "
                 f"{min(allpairs):+.3f}..{max(allpairs):+.3f} & {g.local_pop_fallback.iloc[0]:.3f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    tex.append('\n'.join(L))
    x = G[G.t == IN_AC]
    if len(x):
        L = [r'\begin{table}[H]', r'\footnotesize', r'\caption{IN Arts \& Crafts (random split): NDCG@10 at every grid setting, and the local popularity fallback.\label{tab:in_ac}}',
             r'\centering', r'\begin{tabular}{llccc}', r'\toprule', r'Model & Setting & Local only & Pool & Gain \\', r'\midrule']
        for _, r in x.iterrows():
            p = json.loads(r.params); ps = ', '.join(f'{k}={v}' for k, v in p.items() if k != 'extended')
            L.append(f'{r.model} & {ps} & {r.local:.3f} & {r.pool:.3f} & {r.gain_same:+.3f} \\\\')
        L.append(r'\midrule')
        L.append(f'Local popularity fallback & -- & {x.local_pop_fallback.iloc[0]:.3f} & -- & -- \\\\')
        L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
        tex.append('\n'.join(L))


def seeds(df, tex):
    say('\n=== Split-seed variability (random split, seeds 0,1,2; local vs pool) ===')
    rows, per = [], []
    for model in ['MostPop', 'ItemKNN', 'EASE', 'UserKNN']:
        gs = {sd: df[(df.split == 'random') & (df.tag == tag) & (df.model == model)].set_index('t')
              for sd, tag in ((0, ''), (1, '_seed1'), (2, '_seed2'))}
        if any(len(g) == 0 for g in gs.values()): say(f'{model}: seeds incomplete', {k: len(v) for k, v in gs.items()}); continue
        common = sorted(set.intersection(*[set(g.index) for g in gs.values()]))
        G = pd.DataFrame({sd: gs[sd].loc[common, 'g_pool'] for sd in gs})
        Lc = pd.DataFrame({sd: gs[sd].loc[common, 'local'] for sd in gs})
        sd_ctx = G.std(axis=1, ddof=1)
        stats_by_seed = {}
        for sd in G:
            v = G[sd]
            stats_by_seed[sd] = dict(median=v.median(), mean=v.mean(), better=int((v > 0.01).sum()), worse=int((v < -0.01).sum()), local_median=Lc[sd].median())
        sbs = pd.DataFrame(stats_by_seed).T
        avg = G.mean(axis=1); mk = [t.split(' ', 1)[0] for t in common]; ct = [t.split(' ', 1)[1] for t in common]
        m = ci(avg, mk, ct, 'mean', f'rq2seed|{model}|mean'); md = ci(avg, mk, ct, 'median', f'rq2seed|{model}|median')
        say(f'{model:8s} common contexts={len(common)}; per-context between-seed SD of the gain: median {sd_ctx.median():.4f}, mean {sd_ctx.mean():.4f}, max {sd_ctx.max():.4f}')
        for sd, r in sbs.iterrows():
            say(f'    seed {sd}: local median {r.local_median:.4f} gain median {r["median"]:+.4f} mean {r["mean"]:+.4f} better/worse>0.01 {int(r.better)}/{int(r.worse)}')
        say(f'    between-seed SD: median gain {sbs["median"].std(ddof=1):.4f}, mean gain {sbs["mean"].std(ddof=1):.4f}, '
            f'better {sbs.better.std(ddof=1):.2f}, worse {sbs.worse.std(ddof=1):.2f}, local median {sbs.local_median.std(ddof=1):.4f}')
        say(f'    seed-averaged gain: median {fmt(md[0])} [{fmt(md[1],4)}, {fmt(md[2],4)}], mean {fmt(m[0])} [{fmt(m[1],4)}, {fmt(m[2],4)}]')
        rows.append(dict(model=model, contexts=len(common), sd_ctx_median=sd_ctx.median(), sd_ctx_mean=sd_ctx.mean(), sd_ctx_max=sd_ctx.max(),
                         **{f'seed{sd}_{k}': v for sd, r in sbs.iterrows() for k, v in r.items()},
                         sd_median=sbs['median'].std(ddof=1), sd_mean=sbs['mean'].std(ddof=1), sd_better=sbs.better.std(ddof=1), sd_worse=sbs.worse.std(ddof=1),
                         avg_median=md[0], avg_median_lo=md[1], avg_median_hi=md[2], avg_mean=m[0], avg_mean_lo=m[1], avg_mean_hi=m[2]))
        per.append(G.assign(model=model, sd=sd_ctx))
    if not rows: return
    R = pd.DataFrame(rows); R.to_csv(os.path.join(NOTES, 'rq2_seed_variability.csv'), index=False, lineterminator='\n')
    pd.concat(per).to_csv(os.path.join(NOTES, 'rq2_seed_per_context.csv'), lineterminator='\n')
    L = [r'\begin{table}[H]', r'\footnotesize', r'\caption{Split-seed variability of the pooling gain (random split, seeds 0, 1, 2).\label{tab:seeds}}',
         r'\centering', r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Model & Contexts & SD of context gain (median) & Mean gain per seed & Median gain per seed & SD of mean & Seed-averaged mean (95\% CI) \\', r'\midrule']
    for _, r in R.iterrows():
        L.append(f"{r.model} & {r.contexts} & {r.sd_ctx_median:.4f} & {', '.join(fmt(r[f'seed{s}_mean']) for s in (0, 1, 2))} & "
                 f"{', '.join(fmt(r[f'seed{s}_median']) for s in (0, 1, 2))} & {r.sd_mean:.4f} & {fmt(r.avg_mean)} [{fmt(r.avg_mean_lo,3)}, {fmt(r.avg_mean_hi,3)}] \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    tex.append('\n'.join(L))


if __name__ == '__main__':
    main()
