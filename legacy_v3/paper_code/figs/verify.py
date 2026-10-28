"""Cross-check the summary tables (tables.tex) against the per-context tables (tables_detailed.tex)."""
import numpy as np, re, texparse as T
_sh=[int(v) for v in re.findall(r'patterns higher in (\d+)\\% of pairs', open(T.ROOT + '/tables.tex', encoding='utf-8').read())]  # shares printed in tab:travel paired rows (all foreign, matched, all foreign, matched)
ok = []; bad = []
def chk(name, got, want, tol):
    (ok if abs(got - want) <= tol else bad).append(f'{name}: detailed={got:.4f} summary={want:.4f}')
def chki(name, got, want):
    (ok if got == want else bad).append(f'{name}: detailed={got} summary={want}')
def chkcount(name, g, thr, want, sign=1):
    # detailed tables are rounded to 3 dp, so values printed as exactly +/-0.010 are ambiguous
    g = sign*g; strict = int((g > thr + 1e-9).sum()); amb = int((np.abs(g - thr) < 1e-9).sum())
    good = strict <= want <= strict + amb
    (ok if good else bad).append(f'{name}: detailed={strict} (+{amb} printed as exactly {thr:.3f}) summary={want}')

# ---- travel
d = np.array([r[3:] for r in T.detail('tab:travel_detail')], float)  # shared, poprank, top10, cos, nbr, diff
S = T.main('tab:travel')
for k, col in enumerate([1, 2, 3, 4]):
    x = d[:, col]; x = x[~np.isnan(x)]
    chki(f'travel {S[k][0]} n', len(x), int(S[k][1]))
    chk(f'travel {S[k][0]} median ratio', np.median(x), T.ci(S[k][5])[0], 0.006)
m = ~np.isnan(d[:,1]) & ~np.isnan(d[:,3])
chki('travel paired cos-rank n', int(m.sum()), int(S[4][1]))
chk('travel paired cos-rank median', np.median(d[m,3]-d[m,1]), T.ci(S[4][3])[0], 0.011)
chk('travel paired share pattern>pop (%)', 100*np.mean(d[m,3] > d[m,1]), _sh[0], 0.6)
m2 = ~np.isnan(d[:,2]) & ~np.isnan(d[:,4])
chki('travel paired nbr-top10 n', int(m2.sum()), int(S[6][1]))
chk('travel paired nbr-top10 median', np.median(d[m2,4]-d[m2,2]), T.ci(S[6][3])[0], 0.011)
chk('travel paired share nbr>top10 (%)', 100*np.mean(d[m2,4] > d[m2,2]), _sh[2], 0.6)
chk('travel diff column = cos - rank (max abs err)', np.nanmax(np.abs(d[m,5]-(d[m,3]-d[m,1]))), 0, 0.011)

# ---- full
F = T.main('tab:full')
for split in ['random', 'temporal']:
    D = np.array([r[3:] for r in T.detail(f'tab:full_detail_{split}')], float)
    for j, model in enumerate(['MostPop', 'ItemKNN', 'EASE', 'UserKNN']):
        loc, g = D[:, 2*j], D[:, 2*j+1]; mm = ~np.isnan(g)
        row = [r for r in F if r[0].lower() == split and r[1] == model][0]
        tag = f'full {split} {model}'
        chki(tag+' contexts', int(mm.sum()), int(row[2]))
        chk(tag+' local-only (median)', np.median(loc[mm]), row[3], 0.0015)
        chk(tag+' gain median', np.median(g[mm]), row[4], 0.0006)
        chk(tag+' gain mean', np.mean(g[mm]), T.ci(row[5])[0], 0.0006)
        chkcount(tag+' better>0.01', g[mm], 0.01, int(row[6]))
        chkcount(tag+' worse>0.01', g[mm], 0.01, int(row[7]), -1)

# ---- scarcity
SC = T.main('tab:scarcity')
arms = ['Shopping patterns scarce', 'Both scarce (new market)', 'Shopper histories scarce']
for split, model in [('random','ItemKNN'),('random','EASE'),('random','UserKNN'),('temporal','ItemKNN')]:
    D = np.array([r[2:-1] for r in T.detail(f'tab:scarcity_detail_{model}_{split}')], float)
    for a, arm in enumerate(arms):
        lo, fo, gap = D[:,1+3*a], D[:,2+3*a], D[:,3+3*a]; mm = ~np.isnan(gap)
        row = [r for r in SC if r[0].lower()==split and r[1]==model and r[2]==arm][0]
        tag = f'scarcity {split} {model} {arm}'
        chki(tag+' contexts', int(mm.sum()), int(row[3]))
        chk(tag+' gap median', np.median(gap[mm]), T.ci(row[7])[0], 0.006)
        chk(tag+' local10 median', np.median(lo[mm]), row[4], 0.0015)
        chk(tag+' +foreign median', np.median(fo[mm]), row[5], 0.0015)
        chk(tag+' gain median', np.median(fo[mm]-lo[mm]), row[6], 0.0015)

# ---- launch
L = T.main('tab:launch')
for split in ['random','temporal']:
    D = np.array([r[2:] for r in T.detail(f'tab:launch_detail_{split}')], float)
    for j, model in enumerate(['ItemKNN','EASE','UserKNN','MostPop']):
        lo, fo = D[:,2+2*j], D[:,3+2*j]; mm = ~np.isnan(lo) & ~np.isnan(fo)
        row = [r for r in L if r[0].lower()==split and r[1]==model][0]
        tag = f'launch {split} {model}'
        chki(tag+' contexts', int(mm.sum()), int(row[2]))
        chk(tag+' local median', np.median(lo[mm]), row[3], 0.0015)
        chk(tag+' +foreign median', np.median(fo[mm]), row[4], 0.0015)
        chk(tag+' gain mean', np.mean(fo[mm]-lo[mm]), T.ci(row[5])[0], 0.0015)
        chkcount(tag+' improved (gain>0.01, as in final_tables.py)', fo[mm]-lo[mm], 0.01, int(str(row[6]).split('/')[0]))
    rr = D[:,10]; row = [r for r in L if r[0].lower()==split and r[1].startswith('Random')][0]
    chk(f'launch {split} chance median', np.nanmedian(rr), row[3], 0.0015)

# ---- matrices vs detailed
M = {r[0]: r[1:] for r in T.detail('tab:mat_full')}
cols = ['CA','DE','FR','IN','JP','MX','UK']
D = T.detail('tab:full_detail_random'); mx=0
for r in D:
    cat = r[0].replace('Arts & Crafts','Arts & Crafts')
    if cat in M:
        v = M[cat][cols.index(r[1])]; mx = max(mx, abs(v - r[6])) if not np.isnan(v) else mx
chk('mat_full vs full_detail ItemKNN gain (max abs diff)', mx, 0, 0.0006)

# ---- history: shoppers add up / monotone check only (no per-context file)
H = T.main('tab:history')
for r in H:
    lo, a, b = T.ci(r[5])
    (ok if a <= lo <= b else bad).append(f'history {r[0]} {r[1]}: mean {lo} inside CI [{a},{b}]')

# ---- history: recompute from raw/hist.pkl
import pandas as pd
from scipy.stats import wilcoxon
hh = pd.read_pickle(T.RAW + '/hist.pkl'); hh['g'] = hh.y1 - hh.y0
hh['bucket'] = pd.cut(hh['hist'], [0,1,2,4,1e9], labels=['1','2','3--4','5+'])
for r in H:
    x = hh[hh.model == r[0]]
    pc = x.groupby(['t','bucket'], observed=True).agg(g=('g','mean'), n=('g','size')).reset_index(); pc = pc[pc.n >= 20]
    b = pc[pc.bucket.astype(str) == str(r[1]).replace('.0','')]
    tag = f'history {r[0]} bucket {r[1]}'
    chki(tag+' contexts', b.t.nunique(), int(r[2])); chki(tag+' shoppers', int(b.n.sum()), int(r[3]))
    chk(tag+' median', b.g.median(), r[4], 6e-5); chk(tag+' mean', b.g.mean(), T.ci(r[5])[0], 6e-5)
txt = open(T.ROOT + '/tables.tex', encoding='utf-8').read()
for m, (k, n, p) in {'ItemKNN': (26, 36, 0.003), 'EASE': (18, 36, 0.581)}.items():
    x = hh[hh.model == m]
    x2 = x.assign(s=np.where(x['hist'] <= 2, 's', 'l')).groupby(['t','s']).agg(g=('g','mean'), n=('g','size')).reset_index(); x2 = x2[x2.n >= 20]
    w = x2.pivot(index='t', columns='s', values='g').dropna(); dd = w.s - w.l
    chki(f'history {m} short>long contexts', f'{(dd>0).sum()}/{len(dd)}', f'{k}/{n}')
    chk(f'history {m} Wilcoxon p', wilcoxon(dd).pvalue, p, 6e-4)

print(f'PASSED {len(ok)}  FAILED {len(bad)}')
for s in bad: print('  FAIL', s)
with open('verify_log.txt','w', encoding='utf-8') as f:
    f.write(f'PASSED {len(ok)}  FAILED {len(bad)}\n'); f.write('\n'.join(['OK   '+s for s in ok]+['FAIL '+s for s in bad]))
