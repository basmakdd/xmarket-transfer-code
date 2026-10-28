"""RQ4 ratio-free version: country distances vs the foreign-minus-local agreement difference D = F - L of RQ1.

Why: the agreement ratios F/L of Table 6 are sensitive to small local denominators L (editor, round 2). RQ1 already uses
D = F - L as its main outcome; this script runs the RQ4 permutation tests on D as well, so RQ4 no longer depends on ratios.

Everything except the outcome definition is identical to paper/code/rq4_tables.py and paper/code/sens2_rq4.py
(country features, centring within target context, averaging to ordered country pairs, exact 7! = 5,040 permutation
test, Bonferroni 0.05/15). As a self-check, the script also recomputes the ratio outcomes and verifies that they
reproduce paper/raw/aud_rq4_results.csv (rq4_tables.py --rq1 new) exactly.

Outcomes per source (one row per ordered target-context/donor-country pair, from the RQ1 records):
  D_m = for_m - loc_m for m in {sp (rank correlation), top10, cos, nbr5}; defined wherever both values exist
  (no positivity restriction is needed for a difference). The single-donor gain row is unchanged (it is not a ratio).
Sources: new = paper/raw/rq1_agree.jsonl (original data); _nv and _nvc = variant-merged RQ1 reruns (as in sens2_rq4.py).
Each source is run with IN Arts & Crafts included (as in Table 6 for agreement rows) and excluded; the gain row always
excludes IN Arts & Crafts (as pre-specified), so it is repeated unchanged under both labels.

Outputs: paper/results/rq4_ratiofree.csv, paper/tables_rq4_ratiofree.tex, paper/logs/aud_rq4_ratiofree.log
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, PAPER_RAW

import os, sys, json, math, itertools
import numpy as np, pandas as pd

RAW = PAPER_RAW
LOG = f'{REPO}/paper/logs/aud_rq4_ratiofree.log'


class _Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True); self.f = open(path, 'w', encoding='utf-8', newline='\n'); self.o = sys.stdout
    def write(self, x): self.o.write(x); self.f.write(x)
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = _Tee(LOG)

# country features: verbatim from rq4_tables.py (checked by tests/test_round3.py)
hof={'ca':[39,80,52,48,36,68],'de':[35,67,66,65,83,40],'fr':[68,71,43,86,63,48],'in':[77,48,56,40,51,26],'jp':[54,46,95,92,88,42],'mx':[81,30,69,82,24,97],'uk':[35,89,66,35,51,69]}
Mh=np.array(list(hof.values()),float); var=Mh.var(axis=0,ddof=1)
def ks(a,b): return float(np.mean((np.array(hof[a])-np.array(hof[b]))**2/var))
lang={'ca':{'en','fr'},'uk':{'en'},'in':{'en'},'fr':{'fr'},'de':{'de'},'mx':{'es'},'jp':{'ja'}}
def sl(a,b): return float(bool(lang[a]&lang[b]))
cap={'ca':(45.42,-75.70),'de':(52.52,13.40),'fr':(48.86,2.35),'in':(28.61,77.21),'jp':(35.68,139.69),'mx':(19.43,-99.13),'uk':(51.51,-0.13)}
def geo(a,b):
    (la1,lo1),(la2,lo2)=[(math.radians(p[0]),math.radians(p[1])) for p in (cap[a],cap[b])]
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2; return 6371*2*math.asin(math.sqrt(h))
mk = sorted(hof); perms = list(itertools.permutations(mk))
DIST = [('Hofstede distance', ks), ('Shared language', sl), ('Geographic distance', geo)]
INART = 'in Arts_Crafts_and_Sewing'
ALPHA = 0.05 / 15
SOURCES = {'new': f'{RAW}/rq1_agree.jsonl', '_nv': f'{RAW}/sens_nv/rq1_agree_nv.jsonl', '_nvc': f'{RAW}/sens_nvc/rq1_agree_nvc.jsonl'}
MEAS = [('sp', 'Popularity: rank correlation'), ('top10', 'Popularity: top-10 overlap'),
        ('cos', 'Patterns: co-interaction cosine'), ('nbr5', 'Patterns: 5-NN overlap')]


def pair_avg(z, col):
    z = z.dropna(subset=[col]).copy(); z['res'] = z[col] - z.groupby('t')[col].transform('mean')
    pm = z.groupby(['target_market', 'donor_market']).agg(res=('res', 'mean'), ncat=('category', 'nunique'), npairs=('res', 'size')).reset_index()
    return z, pm


def perm_test(pm, fn):
    a = pm.target_market.values; b = pm.donor_market.values; y = pm.res.values
    def st(p):
        mp = dict(zip(mk, p)); v = np.array([fn(mp[x], mp[w]) for x, w in zip(a, b)]); return np.corrcoef(v, y)[0, 1]
    obs = st(mk); null = np.array([st(p) for p in perms])
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


S = pd.DataFrame([json.loads(l) for l in open(f'{RAW}/aud_single.jsonl', encoding='utf-8')]).drop_duplicates(['t', 'donor_market'])
loc = S[S.arm == 'local'].set_index('t')['ndcg']
G = S[S.arm == 'donor'].copy(); G['d'] = G.ndcg - G.t.map(loc); G = G.dropna(subset=['d'])

rows = []
for name, fn in SOURCES.items():
    if not os.path.exists(fn):
        print('missing', fn); continue
    X = pd.read_json(fn, lines=True).drop_duplicates(['t', 'donor_market']).sort_values(['t', 'donor_market']).reset_index(drop=True)
    Y = X.merge(G[['t', 'donor_market', 'd']], on=['t', 'donor_market'], how='left')
    for m, _ in MEAS:
        Y['D_' + m] = Y['for_' + m] - Y['loc_' + m]
        Y['R_' + m] = (Y['for_' + m] / Y['loc_' + m]).where(Y['loc_' + m] > 0)
    for inc in (True, False):
        Z = Y if inc else Y[Y.t != INART]
        for kind in ('D', 'R'):
            for m, lab in MEAS:
                col = f'{kind}_{m}'
                z, pm = pair_avg(Z, col)
                for dl, f in DIST:
                    r, p = perm_test(pm, f)
                    rows.append(dict(source=name, in_ac='incl.' if inc else 'excl.', kind=kind, measure=m, label=lab, distance=dl,
                                     pairs=len(z), country_pairs=len(pm), r=r, p=p, bonferroni=bool(p < ALPHA)))
        z, pm = pair_avg(Y[Y.t != INART], 'd')
        for dl, f in DIST:
            r, p = perm_test(pm, f)
            rows.append(dict(source=name, in_ac='incl.' if inc else 'excl.', kind='gain', measure='d', label='Gain from one foreign donor (ItemKNN), excl. IN Arts & Crafts',
                             distance=dl, pairs=len(z), country_pairs=len(pm), r=r, p=p, bonferroni=bool(p < ALPHA)))
        print(f'done {name} IN A&C {"incl." if inc else "excl."}', flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{REPO}/paper/results/rq4_ratiofree.csv', index=False, lineterminator='\n')

# ---------------------------------------------------------------- self-check: ratio rows reproduce rq4_tables.py --rq1 new
A = pd.read_csv(f'{RAW}/aud_rq4_results.csv')
mp = {'pop_rel': 'sp', 'pop10_rel': 'top10', 'pat_rel': 'cos', 'nbr_rel': 'nbr5'}
diffs = []
for _, a in A[A.outcome.isin(mp)].iterrows():
    for dl, _ in DIST:
        q = R[(R.source == 'new') & (R.in_ac == 'incl.') & (R.kind == 'R') & (R.measure == mp[a.outcome]) & (R.distance == dl)].iloc[0]
        diffs += [abs(q.r - a[f'r|{dl}']), abs(q.p - a[f'p|{dl}'])]
a = A[A.outcome == 'd_excl'].iloc[0]
for dl, _ in DIST:
    q = R[(R.source == 'new') & (R.in_ac == 'incl.') & (R.kind == 'gain') & (R.distance == dl)].iloc[0]
    diffs += [abs(q.r - a[f'r|{dl}']), abs(q.p - a[f'p|{dl}'])]
mx = max(diffs)
print(f'CHECK ratio and gain rows vs paper/raw/aud_rq4_results.csv: max |diff| over {len(diffs)} values = {mx:.2e}')
if mx > 1e-9: raise SystemExit('ratio rows do not reproduce rq4_tables.py')

# ---------------------------------------------------------------- summary and LaTeX
pd.set_option('display.width', 250)
for name in R.source.unique():
    for inc in ('incl.', 'excl.'):
        q = R[(R.source == name) & (R.in_ac == inc) & (R.kind.isin(['D', 'gain']))]
        pv = q.p.values
        print(f'{name} IN A&C {inc}: ratio-free 15 tests: nominal p<0.05: {(pv < 0.05).sum()}; Bonferroni: {(pv < ALPHA).sum()}; min p {pv.min():.4f}')
        print(q[['label', 'distance', 'pairs', 'country_pairs', 'r', 'p']].round(4).to_string(index=False))

def cell(r, p, star): return f'{r:+.2f}'.replace('-', '$-$') + f' ({p:.4f})' + ('$^{*}$' if star else '')
L = ['% Auto-generated by paper/code/rq4_ratiofree.py. Requires booktabs, threeparttable, adjustbox.',
     '\\begin{table}[H]', '\\begin{threeparttable}', '\\caption{Country distances vs ratio-free cross-market agreement $D=F-L$ and single-donor transfer: correlation $r$ (exact permutation $p$).}',
     '\\label{tab:culture_ratiofree}', '\\centering', '\\small', '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{llcccc}', '\\toprule',
     'Data & Outcome & Pairs & Hofstede distance & Shared language & Geographic distance \\\\', '\\midrule']
SRCLAB = {'new': 'Original', '_nvc': 'Variants merged (store)', '_nv': 'Variants merged (markets)'}
for name in ['new', '_nvc', '_nv']:
    q = R[(R.source == name) & (R.in_ac == 'incl.')]
    if q.empty: continue
    for kind, m, lab in [('D', m, f'$D$, {lab}') for m, lab in MEAS] + [('gain', 'd', 'Single-donor gain (ItemKNN)$^{a}$')]:
        x = q[(q.kind == kind) & (q.measure == m)].set_index('distance')
        L.append(f'{SRCLAB[name]} & {lab} & {x.pairs.iloc[0]} & ' + ' & '.join(cell(x.loc[d, "r"], x.loc[d, "p"], x.loc[d, "bonferroni"]) for d, _ in DIST) + ' \\\\')
    if name != '_nv': L.append('\\midrule')
L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\begin{tablenotes}\\footnotesize',
      '\\item $D=F-L$: foreign minus local agreement with local half A (Table~\\ref{tab:travel}); no positivity restriction on $L$ is needed. '
      'Outcomes are centred within each target context, averaged to ordered country pairs and tested against all 5,040 relabellings of the 7 countries. '
      '15 tests per data version; $^{*}$ $p<0.05/15$. $^{a}$ Excludes IN Arts \\& Crafts; original-data single-donor gains joined to each version\'s pair set.',
      '\\end{tablenotes}', '\\end{threeparttable}', '\\end{table}']
open(f'{REPO}/paper/tables_rq4_ratiofree.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('wrote paper/results/rq4_ratiofree.csv, paper/tables_rq4_ratiofree.tex')
sys.stdout.flush()
