"""RQ4 analysis: do country distances predict cross-market agreement or single-donor transfer?

Inputs
  * Single-donor gains: paper/raw/aud_single.jsonl (single2.py; ItemKNN, random split, regime F, retuned per arm).
    gain(pair) = NDCG@10(local + that one donor) - NDCG@10(local only).
  * Agreement ratios (RQ1), selected with --rq1:
      new (default)      paper/raw/rq1_*.parquet, or if none exists paper/raw/rq1_agree.jsonl (current RQ1, Table 6)
      travel3            paper/raw/travel3_*.pkl (shipped files of the earlier v3 study version, travel3.py); only with
                         an explicit --rq1 travel3 (run.py uses it with --tag _oldrq1)
    Expected columns (one row per ordered pair = target context t and donor market, already averaged over the
    repeated local half-splits): t ('<market> <Category>'), target_market, donor_market, category,
    n_shared (optional), loc_sp, for_sp, loc_top10, for_top10, loc_cos, for_cos, loc_nbr5, for_nbr5.
    ratio_m = for_m / loc_m, defined only where loc_m > 0 (NaN otherwise, and NaN where for_m is NaN). With
    --min-local T (sensitivity check of Section 4.5), the ratio is defined only where loc_m >= T as well.

Pair-inclusion rule (as in final_tables.py, reproduced here): a pair enters an outcome row if it is one of the
RQ1 pairs (travel3.py computes a pair only if target and donor contexts share >= 10 products, counted over ALL
interactions of both contexts, i.e. train+val+test) and the outcome is defined for it (ratio: loc_m > 0 and
for_m not NaN; nbr5 additionally needs >= 5 products with >= 5 co-interaction neighbours in both samples;
gain: a single-donor result exists for the pair). The gain row is run in two versions, excluding the target
context 'in Arts_Crafts_and_Sewing' (as pre-specified in the v3 study) and including it.

Country features (verbatim from i.py / final_tables.py):
  Hofstede 2015 scores (PDI, IDV, MAS, UAI, LTO, IVR); Kogut-Singh index KS(a,b) = mean_d (h_ad - h_bd)^2 / V_d,
  V_d = sample variance (ddof=1) of dimension d over the 7 study countries;
  shared language = 1 if the sets of official languages coded below intersect;
  geographic distance = haversine great-circle distance between capitals, Earth radius 6371 km.

Test: outcomes centred within each target context (value - mean over that context's included donors), averaged
to ordered country pairs (target market, donor market); Pearson r between the pair-averaged residual and the
distance; exact permutation test over all 7! = 5,040 relabellings of the countries (the distance matrix is
relabelled, the pair-averaged outcomes stay attached to their ordered pair, so each pair keeps its own set of
categories); two-sided p = share of relabellings with |r| >= |r_obs| (identity included). 15 tests (5 outcomes
x 3 distances), Bonferroni alpha = 0.05/15.

Outputs: paper/tables_rq4.tex (country-closeness table), paper/tables_S4.tex (Supplementary Table S4),
         paper/raw/aud_rq4_results.csv, paper/raw/aud_rq4_pairs.csv, paper/raw/aud_rq4_gain_summary.csv,
         paper/results/rq4_country_features.csv, paper/results/rq4_country_distances.csv, and the log
         paper/logs/aud_rq4.log; with --tag X every file name above except the two country files gets the suffix X.
         With --min-local T (e.g. 0.05): paper/tables_rq4_minlocal005.tex, paper/results/rq4_minlocal005_results.csv,
         paper/results/rq4_minlocal005_pairs.csv, paper/results/rq4_minlocal005_gain_summary.csv and
         paper/logs/aud_rq4_minlocal005.log (the main outputs are not touched; Table S4 does not depend on T).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, glob, json, math, itertools, argparse
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import boot

ap = argparse.ArgumentParser(); ap.add_argument('--rq1', default='new', choices=['new', 'travel3'],
                                                  help='RQ1 source: new (current, default) or travel3 (earlier v3 study version)')
ap.add_argument('--tag', default='', help='suffix for all output files (e.g. _oldrq1)')
ap.add_argument('--min-local', type=float, default=0.0,
                help='sensitivity check: ratios only where the local agreement is >= this value (e.g. 0.05); separate outputs')
ap.add_argument('--check-old', action='store_true', help='also reproduce the v3 Table 6 gain row from pairs2.pkl')
args = ap.parse_args()
if args.min_local:
    if args.tag: raise SystemExit('--min-local writes its own outputs (rq4_minlocal<T>_*); do not combine it with --tag')
    ML = 'minlocal' + f'{args.min_local:g}'.replace('.', '')         # 0.05 -> minlocal005
    OUT_RES, OUT_PAIRS, OUT_GAIN = (f'{REPO}/paper/results/rq4_{ML}_{n}.csv' for n in ('results', 'pairs', 'gain_summary'))
    OUT_TEX, OUT_S4, LOG = f'{REPO}/paper/tables_rq4_{ML}.tex', None, f'{REPO}/paper/logs/aud_rq4_{ML}.log'
else:
    ML = ''
    OUT_RES, OUT_PAIRS, OUT_GAIN = (f'{PAPER_RAW}/aud_rq4_{n}{args.tag}.csv' for n in ('results', 'pairs', 'gain_summary'))
    OUT_TEX, OUT_S4, LOG = f'{REPO}/paper/tables_rq4{args.tag}.tex', f'{REPO}/paper/tables_S4{args.tag}.tex', f'{REPO}/paper/logs/aud_rq4{args.tag}.log'

class _Tee:
    """Print to the console and to this run's log (paper/logs/aud_rq4<variant>.log)."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True); self.f = open(path, 'w', encoding='utf-8', newline='\n'); self.o = sys.stdout
    def write(self, x): self.o.write(x); self.f.write(x)
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = _Tee(LOG)
def rel(p): return os.path.relpath(p, REPO).replace(os.sep, '/')

# ------------------------------------------------------------------ country features (verbatim from i.py / final_tables.py)
hof={'ca':[39,80,52,48,36,68],'de':[35,67,66,65,83,40],'fr':[68,71,43,86,63,48],'in':[77,48,56,40,51,26],'jp':[54,46,95,92,88,42],'mx':[81,30,69,82,24,97],'uk':[35,89,66,35,51,69]}
dims=['PDI','IDV','MAS','UAI','LTO','IVR']
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
NAME = {'ca': 'Canada', 'de': 'Germany', 'fr': 'France', 'in': 'India', 'jp': 'Japan', 'mx': 'Mexico', 'uk': 'United Kingdom'}
CAPN = {'ca': 'Ottawa', 'de': 'Berlin', 'fr': 'Paris', 'in': 'New Delhi', 'jp': 'Tokyo', 'mx': 'Mexico City', 'uk': 'London'}

# ------------------------------------------------------------------ data
need = ['t', 'target_market', 'donor_market', 'category'] + [f'{a}_{m}' for m in ('sp', 'top10', 'cos', 'nbr5') for a in ('loc', 'for')]
if args.rq1 == 'travel3':
    X = pd.concat([pd.read_pickle(fn) for fn in sorted(glob.glob(f'{PAPER_RAW}/travel3_*.pkl'))]); src = 'paper/raw/travel3_*.pkl'
else:
    fs = sorted(glob.glob(f'{PAPER_RAW}/rq1_*.parquet'))
    if fs: X = pd.concat([pd.read_parquet(fn) for fn in fs]); src = ', '.join(os.path.basename(f) for f in fs)
    else: X = pd.read_json(f'{PAPER_RAW}/rq1_agree.jsonl', lines=True); src = 'paper/raw/rq1_agree.jsonl'
miss = [c for c in need if c not in X.columns]
if miss: raise SystemExit(f'RQ1 source {src} lacks columns {miss}')
X = X.drop_duplicates(['t', 'donor_market']).sort_values(['t', 'donor_market']).reset_index(drop=True)
print('RQ1 source:', src, '| pairs:', len(X) , *([f'| ratios only where local agreement >= {args.min_local:g}'] if args.min_local else []))

S = pd.DataFrame([json.loads(l) for l in open(f'{PAPER_RAW}/aud_single.jsonl', encoding='utf-8')]).drop_duplicates(['t', 'donor_market'])
loc = S[S.arm == 'local'].set_index('t')['ndcg']
G = S[S.arm == 'donor'].copy(); G['own'] = G.t.map(loc); G['d'] = G.ndcg - G.own
G = G.dropna(subset=['d'])
o = pd.read_parquet(f'{RESULTS}/overlap_random.parquet'); o = o[o.channel == 'XM']
print('single-donor results: local arms', S[S.arm == 'local'].ndcg.notna().sum(), '| donor arms with gain', len(G), 'of', len(o), 'XM pairs')

Y = X.merge(G[['t', 'donor_market', 'd', 'own', 'ndcg']], on=['t', 'donor_market'], how='left')
for m, col in [('sp', 'pop_rel'), ('top10', 'pop10_rel'), ('cos', 'pat_rel'), ('nbr5', 'nbr_rel')]:
    Y[col] = (Y['for_' + m] / Y['loc_' + m]).where((Y['loc_' + m] > 0) & (Y['loc_' + m] >= args.min_local))
INART = 'in Arts_Crafts_and_Sewing'
Y['d_excl'] = Y.d.where(Y.t != INART)
OUT = [('pop_rel', 'Popularity agreement ratio (rank correlation)'), ('pop10_rel', 'Popularity agreement ratio (top-10 overlap)'),
       ('pat_rel', 'Pattern agreement ratio (cosine)'), ('nbr_rel', 'Pattern agreement ratio (neighbour overlap)'),
       ('d_excl', 'Gain from one foreign donor (ItemKNN), excl.\\ IN Arts \\& Crafts'), ('d', 'Gain from one foreign donor (ItemKNN), incl.\\ IN Arts \\& Crafts')]

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

def automorphisms(fn):
    D0 = {(x, w): fn(x, w) for x in mk for w in mk if x != w}
    return sum(all(abs(fn(dict(zip(mk, p))[x], dict(zip(mk, p))[w]) - D0[(x, w)]) < 1e-12 for (x, w) in D0) for p in perms)

res, pairs_out = [], []
ALPHA = 0.05 / 15
for col, lab in OUT:
    z, pm = pair_avg(Y, col)
    rec = dict(outcome=col, label=lab, pairs=len(z), targets=z.t.nunique(), country_pairs=len(pm),
               single_donor_targets=int((z.groupby('t').size() == 1).sum()))
    for dl, fn in DIST:
        r, p = perm_test(pm, fn); rec[f'r|{dl}'] = r; rec[f'p|{dl}'] = p
    res.append(rec)
    pairs_out.append(pm.assign(outcome=col))
    print(f"{lab}: pairs {len(z)}, country pairs {len(pm)}, " + ', '.join(f"{dl} r={rec['r|'+dl]:+.3f} p={rec['p|'+dl]:.4f}" for dl, _ in DIST))
R = pd.DataFrame(res); R.to_csv(OUT_RES, index=False, lineterminator='\n')
P = pd.concat(pairs_out); P.to_csv(OUT_PAIRS, index=False, lineterminator='\n')
main = R[R.outcome != 'd']
pv = main[[f'p|{dl}' for dl, _ in DIST]].values.ravel()
print(f'15 pre-specified tests: nominal p<0.05: {(pv < 0.05).sum()}; p<0.05/15: {(pv < ALPHA).sum()}; min p {pv.min():.4f}')
aut = {dl: automorphisms(fn) for dl, fn in DIST}
print('relabellings leaving each distance matrix unchanged (lower bound of attainable p = count/5040):', aut)

if args.check_old:
    P2 = pd.read_pickle(f'{PAPER_RAW}/pairs2.pkl')[['t', 'donor_market', 'd']]
    Yo = X.merge(P2, on=['t', 'donor_market'], how='left'); zo, pmo = pair_avg(Yo, 'd')
    print('CHECK old Table 6 gain row (pairs2.pkl):', len(zo), 'pairs;', ', '.join(f'{dl} r={perm_test(pmo, fn)[0]:+.2f} p={perm_test(pmo, fn)[1]:.2f}' for dl, fn in DIST))
    cmp = Yo.dropna(subset=['d']).merge(Y[['t', 'donor_market', 'd']], on=['t', 'donor_market'], suffixes=('_old', '_new')).dropna(subset=['d_old', 'd_new'])
    dd = (cmp.d_new - cmp.d_old).abs()
    print(f'CHECK new vs old single-donor gains on {len(cmp)} common pairs: identical (|diff|<1e-12) {int((dd < 1e-12).sum())}, max |diff| {dd.max():.2e}')
    newp = set(zip(Y.dropna(subset=["d_excl"]).t, Y.dropna(subset=["d_excl"]).donor_market)) - set(zip(zo.t, zo.donor_market))
    print('pairs in the new gain row (excl. IN Arts) but not in the old one:', sorted(newp))

# ------------------------------------------------------------------ single-donor gain summaries (pair level, boot.summarize with donors)
summ = []
for lab, z in [('all XM pairs with a result', G), ('all XM pairs, excl. IN Arts & Crafts', G[G.t != INART]),
               ('RQ4 pairs (>=10 shared products)', Y.dropna(subset=['d'])), ('RQ4 pairs, excl. IN Arts & Crafts', Y.dropna(subset=['d_excl']))]:
    tm = z.target_market.values; ct = (z.category if 'category' in z else z.t.str.split().str[1]).values; dm = z.donor_market.values
    for stt in ('median', 'mean'):
        v, lo, hi = boot.summarize(z.d.values, tm, ct, donors=dm, stat=stt, label=f'aud_rq4|gain|{lab}|{stt}')
        summ.append(dict(set=lab, stat=stt, n=len(z), value=v, lo=lo, hi=hi, better=int((z.d > 0.01).sum()), worse=int((z.d < -0.01).sum())))
SM = pd.DataFrame(summ); print(SM.round(4).to_string())
SM.to_csv(OUT_GAIN, index=False, lineterminator='\n')

# ------------------------------------------------------------------ LaTeX: country-closeness table (tab:culture)
def cell(r, p): return f'{r:+.2f} ({p:.3f})'
L = ['% Auto-generated by paper/code/rq4_tables.py (RQ1 source: ' + src.replace('_', '\\_') + '). Requires booktabs, threeparttable, adjustbox.',
     '\\begin{table}[H]', '\\caption{Country distances vs cross-market agreement and single-donor transfer: correlation $r$ (exact permutation $p$).}',
     '\\label{tab:culture' + (f'_{ML}' if ML else '') + '}', '\\centering', '\\small', '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{lccccc}', '\\toprule',
     'Outcome & Pairs & Country pairs & Hofstede distance & Shared language & Geographic distance \\\\', '\\midrule']
for _, r in R.iterrows():
    if r.outcome == 'd': L.append('\\midrule')
    L.append(f'{r.label}{"$^{a}$" if r.outcome == "d" else ""} & {r.pairs} & {r.country_pairs} & ' + ' & '.join(cell(r[f"r|{dl}"], r[f"p|{dl}"]) for dl, _ in DIST) + ' \\\\')
gsum = SM[(SM.set == 'RQ4 pairs, excl. IN Arts & Crafts') & (SM.stat == 'median')].iloc[0]
L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\begin{tablenotes}\\footnotesize',
      '\\item Pairs: target-context/donor-country pairs that share at least 10 products (counted over all interactions of both contexts) and for which the outcome is defined. '
      'Outcomes are centred within each target context, averaged to ordered country pairs, and correlated (Pearson) with the distance; '
      'two-sided $p$ from all 5,040 relabellings of the 7 countries (the distance matrix is relabelled; each ordered country pair keeps its own outcome and set of categories). '
      'Ratios: foreign/local agreement as in Table~\\ref{tab:travel}, pairs with '
      + (f'local agreement below {args.min_local:g} excluded (sensitivity check; the main table excludes only non-positive local agreement). ' if args.min_local else 'non-positive local agreement excluded. ') +
      'Gain: NDCG@10 of ItemKNN trained on local data plus all same-category data of one foreign country, minus local data only; random split; hyperparameters retuned on validation users in both arms '
      f'(median gain over the {int(gsum.n)} pairs {gsum.value:+.4f}, 95\\% CI [{gsum.lo:+.4f}, {gsum.hi:+.4f}]). '
      'Hofstede distance: Kogut--Singh index over six dimensions; shared language: the countries share an official language; geographic distance: great-circle distance between capitals (Supplementary Table S4). '
      f'The first five rows are the 15 pre-specified tests: {(pv < 0.05).sum()} of 15 with $p<0.05$, {(pv < ALPHA).sum()} below the Bonferroni threshold ($\\alpha=0.05/15$). '
      f'$^{{a}}$ Sensitivity row: IN Arts \\& Crafts, whose local-only ItemKNN scores below chance, retained. '
      f'Because several countries are interchangeable on a distance, the smallest attainable $p$ is {aut["Hofstede distance"]}/5040 for Hofstede, {aut["Shared language"]}/5040 for language and {aut["Geographic distance"]}/5040 for geography.',
      '\\end{tablenotes}', '\\end{table}']
open(OUT_TEX, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ LaTeX: Supplementary Table S4
def mat(fn, fmt):
    rows = []
    for a in mk:
        rows.append(f'{a.upper()} & ' + ' & '.join('--' if a == b else fmt(fn(a, b)) for b in mk) + ' \\\\')
    return rows
head = ' & ' + ' & '.join(x.upper() for x in mk) + ' \\\\'
gz, gpm = pair_avg(Y, 'd_excl')
ncat = gpm.set_index(['target_market', 'donor_market']).ncat
L = ['% Auto-generated by paper/code/rq4_tables.py. Requires booktabs, threeparttable, adjustbox.',
     '\\begin{table}[H]', '\\caption{Inputs of the country-distance analysis (Supplementary Table S4).}', '\\label{tab:S4}', '\\centering', '\\small',
     '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{l' + 'c' * 8 + '}', '\\toprule',
     '\\multicolumn{9}{l}{\\textit{(a) Hofstede scores (2015 release) and capitals}} \\\\',
     'Country & ' + ' & '.join(dims) + ' & Capital & Lat., Lon. \\\\', '\\midrule']
for a in mk:
    L.append(f'{NAME[a]} ({a.upper()}) & ' + ' & '.join(str(v) for v in hof[a]) + f' & {CAPN[a]} & {cap[a][0]:.2f}, {cap[a][1]:.2f} \\\\')
L.append('Variance $V_d$ (ddof=1, 7 countries) & ' + ' & '.join(f'{v:.1f}' for v in var) + ' & & \\\\')
L += ['\\midrule', '\\multicolumn{9}{l}{\\textit{(b) Kogut--Singh distance $\\frac{1}{6}\\sum_d (h_{ad}-h_{bd})^2/V_d$}} \\\\', head, '\\midrule'] + mat(ks, lambda v: f'{v:.2f}')
L += ['\\midrule', '\\multicolumn{9}{l}{\\textit{(c) Shared official language (1 = yes); coded languages: ' + '; '.join(f'{a.upper()} ' + '/'.join(sorted(lang[a])) for a in mk) + '}} \\\\', head, '\\midrule'] + mat(sl, lambda v: f'{int(v)}')
L += ['\\midrule', '\\multicolumn{9}{l}{\\textit{(d) Great-circle (haversine) distance between capitals, km ($R=6371$ km)}} \\\\', head, '\\midrule'] + mat(geo, lambda v: f'{v:,.0f}')
L += ['\\midrule', '\\multicolumn{9}{l}{\\textit{(e) Categories contributing to each ordered pair, gain row (row: target, column: donor)}} \\\\', head, '\\midrule']
for a in mk:
    L.append(f'{a.upper()} & ' + ' & '.join('--' if a == b else str(int(ncat.get((a, b), 0))) for b in mk) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\begin{tablenotes}\\footnotesize',
      '\\item Hofstede scores: PDI power distance, IDV individualism, MAS masculinity, UAI uncertainty avoidance, LTO long-term orientation, IVR indulgence. '
      'Variances are sample variances over the seven study countries. Panel (e) shows that country pairs are averaged over different, incomplete sets of categories; '
      'the permutation test keeps these sets attached to their ordered pair.', '\\end{tablenotes}', '\\end{table}']
if OUT_S4: open(OUT_S4, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
# machine-readable country features (input of paper/code/supp_tables.py, Table S4; the same for every --rq1/--tag)
pd.DataFrame([dict(market=a, country=NAME[a], **dict(zip(dims, hof[a])), capital=CAPN[a], lat=cap[a][0], lon=cap[a][1],
                   languages='/'.join(sorted(lang[a]))) for a in mk]).to_csv(f'{REPO}/paper/results/rq4_country_features.csv', index=False, encoding='utf-8', lineterminator='\n')
pd.DataFrame([dict(market_a=a, market_b=b, kogut_singh=ks(a, b), shared_language=int(sl(a, b)), geo_km=geo(a, b))
              for a in mk for b in mk if a != b]).to_csv(f'{REPO}/paper/results/rq4_country_distances.csv', index=False, encoding='utf-8', lineterminator='\n')
print('wrote', ', '.join(rel(f) for f in (OUT_TEX, OUT_S4, OUT_RES, OUT_PAIRS, OUT_GAIN) if f))
sys.stdout.flush()
