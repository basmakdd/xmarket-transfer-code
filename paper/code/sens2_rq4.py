"""sens2: RQ4 permutation tests (country distances vs RQ1 agreement ratios and single-donor gains) on several RQ1 sources,
each with and without the target context IN Arts & Crafts on ALL outcomes.

Country features, outcome definitions, centring, pair averaging and the exact 7! permutation test are copied verbatim from
paper/code/rq4_tables.py (which cannot be imported: it parses arguments and writes files at import). The 'with IN A&C'
rows of source 'new' must reproduce rq4_tables.py --rq1 new (checked below against paper/raw/aud_rq4_results.csv).

Sources (RQ1 records, one per ordered pair, keyed by t and donor_market):
  new   paper/raw/rq1_agree.jsonl               (rebuilt RQ1, paper data)
  _nv   paper/raw/sens_nv/rq1_agree_nv.jsonl    (RQ1 rerun on the pooled-market variant merge)
  _nvc  paper/raw/sens_nvc/rq1_agree_nvc.jsonl  (RQ1 rerun on the within-context variant merge)
Single-donor gains: ALWAYS paper/raw/aud_single.jsonl (single2.py on the ORIGINAL paper data; no single-donor run exists
on the variants), joined on (t, donor_market); so for _nv/_nvc the gain rows use variant RQ1 pair inclusion but paper-data gains.
15 pre-specified tests per version = 5 outcomes (4 ratios + gain excl. IN A&C) x 3 distances; Bonferroni alpha = 0.05/15.
Output: paper/results/sens2_rq4.csv; printed table.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json, math, itertools
import numpy as np, pandas as pd

RAW = PAPER_RAW   # paper/raw unless the environment variable PAPER_RAW is set
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
OUT = [('pop_rel', 'pop ratio (rank corr.)'), ('pop10_rel', 'pop ratio (top-10)'), ('pat_rel', 'pattern ratio (cosine)'),
       ('nbr_rel', 'pattern ratio (nbr overlap)'), ('d', 'single-donor gain (ItemKNN)')]


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
    for m, col in [('sp', 'pop_rel'), ('top10', 'pop10_rel'), ('cos', 'pat_rel'), ('nbr5', 'nbr_rel')]:
        Y[col] = (Y['for_' + m] / Y['loc_' + m]).where(Y['loc_' + m] > 0)
    for inc in (True, False):
        Z = Y if inc else Y[Y.t != INART]
        for col, lab in OUT:
            z, pm = pair_avg(Z, col)
            for dl, f in DIST:
                r, p = perm_test(pm, f)
                rows.append(dict(source=name, in_ac=('incl.' if inc else 'excl.'), outcome=col, label=lab, distance=dl, pairs=len(z),
                                 targets=z.t.nunique(), country_pairs=len(pm), r=r, p=p, bonferroni=bool(p < ALPHA)))
        print(f'done {name} IN A&C {"incl." if inc else "excl."}', flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{REPO}/paper/results/sens2_rq4.csv', index=False, lineterminator='\n')

pd.set_option('display.width', 250)
for name in R.source.unique():
    for inc in ('incl.', 'excl.'):
        q = R[(R.source == name) & (R.in_ac == inc)]
        print(f'\n### RQ1 source {name}, IN Arts & Crafts {inc} (single-donor gains from paper/raw/aud_single.jsonl)')
        print('| Outcome | Pairs | Country pairs | ' + ' | '.join(d for d, _ in DIST) + ' |')
        print('|---|---|---|---|---|---|')
        for col, lab in OUT:
            x = q[q.outcome == col].set_index('distance')
            print(f'| {lab} | {x.pairs.iloc[0]} | {x.country_pairs.iloc[0]} | ' +
                  ' | '.join(f"{x.loc[d, 'r']:+.3f} (p={x.loc[d, 'p']:.4f}){' *' if x.loc[d, 'bonferroni'] else ''}" for d, _ in DIST) + ' |')
        pv = q.p.values
        print(f'15 tests: nominal p<0.05: {(pv < 0.05).sum()}; p<0.05/15: {(pv < ALPHA).sum()}; min p {pv.min():.4f} '
              f'({q.loc[q.p.idxmin(), "label"]} x {q.loc[q.p.idxmin(), "distance"]})')

# reproduction check against rq4_tables.py --rq1 new (its ratio rows include IN A&C, its gain row 'd_excl' excludes it)
A = pd.read_csv(f'{RAW}/aud_rq4_results.csv')
q = R[R.source == 'new']; diffs = []
for col in ('pop_rel', 'pop10_rel', 'pat_rel', 'nbr_rel', 'd_excl', 'd'):
    a = A[A.outcome == col].iloc[0]
    mine = q[(q.outcome == ('d' if col.startswith('d') else col)) & (q.in_ac == ('excl.' if col == 'd_excl' else 'incl.'))].set_index('distance')
    for dl, _ in DIST:
        diffs.append(max(abs(a[f'r|{dl}'] - mine.loc[dl, 'r']), abs(a[f'p|{dl}'] - mine.loc[dl, 'p'])))
print(f'\nCHECK vs paper/raw/aud_rq4_results.csv (rq4_tables.py --rq1 new): max |diff| of r and p over 18 cells = {max(diffs):.2e}')
