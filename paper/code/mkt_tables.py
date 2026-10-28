"""Tables for the Market-Aware GMF baseline (mkt_run.py output paper/raw/mkt_runs.jsonl).

Thresholds fixed in advance: better/worse = difference in NDCG@10 > 0.01 / < -0.01.
CIs: boot.summarize (two-way cluster bootstrap over countries x categories, context level).
Closed-form references: local (k=0) and all-donor pooled ItemKNN / EASE from paper/raw/rq2_<model>_random.jsonl
(fields local.ndcg / pool.ndcg); if that file is missing, from paper/raw/df.pkl (k=0 NONE and XM full).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json
import numpy as np, pandas as pd
import boot

THR = 0.01
r = pd.read_json(f'{PAPER_RAW}/mkt_runs.jsonl', lines=True)
r = r.drop_duplicates(['t', 'variant'], keep='last')
W = r.pivot(index='t', columns='variant', values='ndcg@10')
W = W.dropna(subset=['local', 'pooled', 'ma'])
tuned = r[r.variant == 'local'].set_index('t')['tuned'].reindex(W.index)
mk = np.array([t.split(' ')[0] for t in W.index]); ct = np.array([t.split(' ', 1)[1] for t in W.index])


def ref(model):
    p = f'{PAPER_RAW}/rq2_{model}_random.jsonl'
    if os.path.exists(p):
        d = pd.read_json(p, lines=True)
        d = d[(d.tag == '') & (d.arms == 'full')].drop_duplicates('t', keep='last').set_index('t')
        return (d['local'].map(lambda x: x['ndcg']), d['pool'].map(lambda x: x['ndcg']), os.path.basename(p))
    d = pd.read_pickle(f'{PAPER_RAW}/df.pkl'); d = d[d.model == model]
    lo = d[(d.k == 0) & (d.channel == 'NONE')].drop_duplicates('t').set_index('t')['ndcg@10']
    po = d[(d.channel == 'XM') & (d.order == 'full')].drop_duplicates('t').set_index('t')['ndcg@10']
    return lo, po, 'df.pkl'


def s(vals, sub, stat, label):
    v, lo, hi = boot.summarize(np.asarray(vals, float), mk[sub], ct[sub], stat=stat, label=label)
    return v, lo, hi


def fmt(x):
    return f'{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]'


ALL = np.ones(len(W), bool)
print(f'contexts with all three variants: {len(W)}  (untuned local: {int((~tuned.astype(bool)).sum())})')
print('categories:', sorted(set(ct)))
skips = f'{PAPER_RAW}/mkt_skips.jsonl'
if os.path.exists(skips):
    for l in open(skips, encoding='utf-8'): print('excluded:', l.strip())

rows = []
names = {'local': 'GMF, local only', 'pooled': 'GMF, pooled (no market embedding)', 'ma': 'MA-GMF, pooled + market embedding'}
for v in ('local', 'pooled', 'ma'):
    med = s(W[v], ALL, 'median', f'mkt|{v}|level')
    row = dict(name=names[v], n=len(W), med=med)
    if v != 'local':
        d = (W[v] - W['local']).to_numpy()
        row.update(gmed=s(d, ALL, 'median', f'mkt|{v}-local|med'), gmean=s(d, ALL, 'mean', f'mkt|{v}-local|mean'),
                   better=int((d > THR).sum()), worse=int((d < -THR).sum()))
    rows.append(row)
d = (W['ma'] - W['pooled']).to_numpy()
mvp = dict(name='MA-GMF minus pooled GMF', n=len(W), gmed=s(d, ALL, 'median', 'mkt|ma-pooled|med'),
           gmean=s(d, ALL, 'mean', 'mkt|ma-pooled|mean'), better=int((d > THR).sum()), worse=int((d < -THR).sum()))

print('\n== GMF family (NDCG@10, test, full-catalog) ==')
for row in rows + [mvp]:
    out = f"{row['name']:38s} n={row['n']}"
    if 'med' in row: out += f"  median={fmt(row['med'])}"
    if 'gmed' in row:
        out += f"  gain median={fmt(row['gmed'])} mean={fmt(row['gmean'])}  better={row['better']} worse={row['worse']}"
    print(out)

# update-matched local control (local_steps): same number of gradient updates and stopping granularity as pooled
srows = []
if 'local_steps' in W.columns and W['local_steps'].notna().all():
    ls = W['local_steps'].to_numpy()
    srows.append(dict(name='GMF, local only, update-matched', n=len(W), med=s(ls, ALL, 'median', 'mkt|local_steps|level')))
    for v in ('local', 'pooled', 'ma'):
        dd = W[v].to_numpy() - ls
        srows.append(dict(name=f'{names[v]} minus update-matched local GMF', n=len(W),
                          gmed=s(dd, ALL, 'median', f'mkt|{v}-lsteps|med'), gmean=s(dd, ALL, 'mean', f'mkt|{v}-lsteps|mean'),
                          better=int((dd > THR).sum()), worse=int((dd < -THR).sum())))
    print('\n== update-matched local GMF ==')
    for row in srows:
        out = f"{row['name']:60s} n={row['n']}"
        if 'med' in row: out += f"  median={fmt(row['med'])}"
        if 'gmed' in row: out += f"  diff median={fmt(row['gmed'])} mean={fmt(row['gmean'])} better={row['better']} worse={row['worse']}"
        print(out)
    for model in ('ItemKNN', 'EASE'):
        lo = ref(model)[0].reindex(W.index).to_numpy(float)
        dd = ls - lo
        print(f'update-matched local GMF minus {model} local: median={np.nanmedian(dd):.3f} mean={np.nanmean(dd):.3f} better={(dd>THR).sum()} worse={(dd<-THR).sum()}')
else:
    print('\nlocal_steps incomplete')

# tuned-only sensitivity
T = tuned.astype(bool).to_numpy()
print(f'\n== tuned-only subset (n={T.sum()}) ==')
for a, b in (('pooled', 'local'), ('ma', 'local'), ('ma', 'pooled')):
    dd = (W[a] - W[b]).to_numpy()[T]
    print(f'{a}-{b}: median={np.median(dd):.3f} mean={dd.mean():.3f} better={(dd>THR).sum()} worse={(dd<-THR).sum()}')

# per category
print('\n== per category (median NDCG@10 / median MA-local, MA-pooled) ==')
C = W.copy(); C['cat'] = ct
for c, g in C.groupby('cat'):
    print(f"{c:30s} n={len(g)} local={g.local.median():.3f} pooled={g.pooled.median():.3f} ma={g.ma.median():.3f} "
          f"ma-local={(g.ma-g.local).median():+.3f} ma-pooled={(g.ma-g.pooled).median():+.3f}")

# closed-form reference
print('\n== closed-form references on the same contexts ==')
refrows = []
for model in ('ItemKNN', 'EASE'):
    lo, po, src = ref(model)
    idx = [t for t in W.index if t in lo.index and pd.notna(lo[t])]
    sub = np.isin(W.index, idx)
    L = lo.reindex(W.index)[sub].to_numpy(float); P = po.reindex(W.index)[sub].to_numpy(float)
    for lab, vec in ((f'{model}, local', L), (f'{model}, pooled', P)):
        med = s(vec, sub, 'median', f'mkt|{lab}|level')
        refrows.append(dict(name=lab, n=int(sub.sum()), med=med, src=src))
    for v in ('local', 'pooled', 'ma'):
        dd = W[v].to_numpy()[sub] - L
        e = dict(name=f'{names[v]} minus {model} local', n=int(sub.sum()),
                 gmed=s(dd, sub, 'median', f'mkt|{v}-{model}L|med'), gmean=s(dd, sub, 'mean', f'mkt|{v}-{model}L|mean'),
                 better=int((dd > THR).sum()), worse=int((dd < -THR).sum()), src=src)
        refrows.append(e)
    print(f'[{model} from {src}; {sub.sum()} of {len(W)} contexts]')
for row in refrows:
    out = f"{row['name']:60s} n={row['n']}"
    if 'med' in row: out += f"  median={fmt(row['med'])}"
    if 'gmed' in row: out += f"  diff median={fmt(row['gmed'])} mean={fmt(row['gmean'])} better={row['better']} worse={row['worse']}"
    print(out)

# timing
print('\n== runtime ==')
u = r.drop_duplicates(['variant', 'unit_sec', 'sec_per_epoch'])
print(r.groupby('variant')[['sec_per_epoch']].describe().round(2).to_string())
r['_unit'] = r['variant'] + '|' + r['train_contexts'].map(lambda x: ','.join(x))
print('total training+selection seconds (one per unit):', round(r.drop_duplicates('_unit')['unit_sec'].sum()))
print('seconds by variant:', r.drop_duplicates('_unit').groupby('variant')['unit_sec'].sum().round().to_dict())
# gradient updates up to the selected epoch (epoch budget is equal; updates scale with training data)
rr = r[r.t.isin(W.index)].copy()
rr['steps'] = np.ceil(rr.n_edges * 5 / 1024) * rr.best_ep * rr['mult'].fillna(1)
print('gradient updates at selected epoch, median by variant:', rr.groupby('variant').steps.median().to_dict())
print('training edges, median by variant:', rr.groupby('variant').n_edges.median().to_dict())
print('selected configs:\n', r[r.t.isin(W.index)].groupby(['variant', 'lr', 'l2']).size().to_string())
print('best epoch median by variant:\n', r[r.t.isin(W.index) & r.tuned].groupby('variant').best_ep.median().to_string())


# machine-readable summary (input of paper/code/supp_tables.py, Table S5b)
def flat(row, group):
    out = dict(group=group, name=row['name'], n=row['n'])
    for k in ('med', 'gmed', 'gmean'):
        v = row.get(k, (np.nan, np.nan, np.nan))
        out[k], out[k + '_lo'], out[k + '_hi'] = v
    out['better'] = row.get('better', np.nan); out['worse'] = row.get('worse', np.nan)
    return out
pd.DataFrame([flat(x, 'gmf') for x in rows + [mvp]] + [flat(x, 'update_matched') for x in srows] +
             [flat(x, 'closed_form') for x in refrows]).to_csv(f'{REPO}/paper/results/mkt_summary.csv', index=False, encoding='utf-8', lineterminator='\n')


# LaTeX
def tex_ci(x): return f'{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]'
L = [r'\begin{table}[t]', r'\centering\small',
     r'\caption{Market-Aware GMF \citep{bhargav2023market} and two GMF controls with identical architecture, loss, '
     r'negative sampler (4 per positive), optimiser, epoch budget and validation-based tuning, on the random split. '
     r'NDCG@10, full-catalog ranking over the target catalog. Gains are paired differences per context; '
     r'brackets: 95\% two-way cluster-bootstrap intervals (countries $\times$ categories). '
     r'Better/worse: difference above $+0.01$ / below $-0.01$.}',
     r'\label{tab:mkt}', r'\begin{tabular}{lrcccrr}', r'\toprule',
     r'Model & $n$ & Median NDCG@10 & Median gain & Mean gain & Better & Worse \\', r'\midrule',
     r'\multicolumn{7}{l}{\emph{Gain over GMF trained on the target context only}}\\']
for row in rows:
    if 'gmed' in row:
        L.append(f"{row['name']} & {row['n']} & {tex_ci(row['med'])} & {tex_ci(row['gmed'])} & {tex_ci(row['gmean'])} & {row['better']} & {row['worse']} \\\\")
    else:
        L.append(f"{row['name']} & {row['n']} & {tex_ci(row['med'])} & -- & -- & -- & -- \\\\")
L.append(r'\midrule')
L.append(f"MA-GMF $-$ pooled GMF & {mvp['n']} & -- & {tex_ci(mvp['gmed'])} & {tex_ci(mvp['gmean'])} & {mvp['better']} & {mvp['worse']} \\\\")
if srows:
    L.append(r'\midrule')
    L.append(r'\multicolumn{7}{l}{\emph{Update-matched local control: local GMF given as many gradient updates per epoch as pooled GMF}}\\')
    for row in srows:
        nm = row['name'].replace('GMF, local only, update-matched', 'GMF local, update-matched').replace('GMF, local only', 'GMF local') \
            .replace('GMF, pooled (no market embedding)', 'GMF pooled').replace('MA-GMF, pooled + market embedding', 'MA-GMF') \
            .replace('update-matched local GMF', 'update-matched local')
        if 'med' in row:
            L.append(f"{nm} & {row['n']} & {tex_ci(row['med'])} & -- & -- & -- & -- \\\\")
        else:
            L.append(f"{nm} & {row['n']} & -- & {tex_ci(row['gmed'])} & {tex_ci(row['gmean'])} & {row['better']} & {row['worse']} \\\\")
L.append(r'\midrule')
L.append(r'\multicolumn{7}{l}{\emph{Closed-form references on the same contexts; differences are GMF-family minus the local closed-form model}}\\')
for row in refrows:
    nm = row['name'].replace('GMF, local only', 'GMF local').replace('GMF, pooled (no market embedding)', 'GMF pooled') \
        .replace('MA-GMF, pooled + market embedding', 'MA-GMF')
    if 'med' in row:
        L.append(f"{nm} & {row['n']} & {tex_ci(row['med'])} & -- & -- & -- & -- \\\\")
    else:
        L.append(f"{nm} & {row['n']} & -- & {tex_ci(row['gmed'])} & {tex_ci(row['gmean'])} & {row['better']} & {row['worse']} \\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(f'{REPO}/paper/tables_mkt.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L).replace(' minus ', ' $-$ ') + '\n')
print('\nwrote paper/tables_mkt.tex and paper/results/mkt_summary.csv')
