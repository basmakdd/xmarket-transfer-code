"""Unreported variants of the stopped core grid (results/full_random.jsonl -> paper/raw/df.pkl), for an appendix.

(a) PureSVD, XM channel (same category, other markets): all donors pooled vs local only.
(b) ItemKNN-ext on the XC channel (same market, other categories; users shared).
(c) XC and XX (other market AND other category; nothing shared) for every model, all donors pooled vs local only.
    XM rows of all models are added for reference.
(d) Coverage: which contexts lack the local-only or the full-pooling unit for each (model, channel), and why
    (error recorded in results/full_random.jsonl, or no record because the run was stopped).

Random split, NDCG@10 on the full catalog, hyperparameters re-tuned per arm on validation users (src/experiment.py
fit_and_select). Gain = NDCG@10(pooled) - NDCG@10(local only, same model). 'Better'/'Worse': gain > +0.01 / < -0.01.
CIs: boot.summarize (two-way cluster bootstrap, context level).
Outputs paper/tables_variants.tex and paper/raw/aud_variants.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, json
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import boot

raw = pd.DataFrame([json.loads(l) for l in open(f'{RESULTS}/full_random.jsonl', encoding='utf-8') if l.strip()])
raw['t'] = raw.target_market + ' ' + raw.target_category
df = pd.read_pickle(f'{PAPER_RAW}/df.pkl')
ctx = pd.read_parquet(f'{RESULTS}/contexts.parquet'); ALL = sorted(ctx.market + ' ' + ctx.category)
print('records in results/full_random.jsonl:', len(raw), '| with error:', int(raw.error.notna().sum()),
      raw.error.value_counts().to_dict(), '| df.pkl records (error-free):', len(df))
ok = raw[raw.error.isna()]
ov = pd.read_parquet(f'{RESULTS}/overlap_random.parquet')
HAS = {ch: set(g.target_market + ' ' + g.target_category) for ch, g in ov.groupby('channel')}

rows, cov = [], []
MODELS = ['MostPop', 'ItemKNN', 'ItemKNN-ext', 'EASE', 'PureSVD']
for ch in ['XM', 'XC', 'XX']:
    for m in MODELS:
        base = ok[(ok.model == m) & (ok.k == 0)].drop_duplicates('t').set_index('t')['ndcg@10']
        full = ok[(ok.model == m) & (ok.channel == ch) & (ok.order == 'full') & (ok.frac == 1)].drop_duplicates('t').set_index('t')['ndcg@10']
        z = pd.DataFrame({'base': base, 'full': full}).dropna(); z['d'] = z.full - z.base
        mk = [t.split()[0] for t in z.index]; ct = [t.split()[1] for t in z.index]
        med, mlo, mhi = boot.summarize(z.d.values, mk, ct, stat='median', label=f'aud_variants|{m}|{ch}|median')
        mea, lo, hi = boot.summarize(z.d.values, mk, ct, stat='mean', label=f'aud_variants|{m}|{ch}|mean')
        status = 'blind (gain exactly 0 by construction)' if (m, ch) in {('MostPop', 'XC'), ('MostPop', 'XX'), ('ItemKNN', 'XC'), ('ItemKNN', 'XX'), ('EASE', 'XC'), ('EASE', 'XX'), ('ItemKNN-ext', 'XX')} else \
                 ('redundant with ItemKNN' if (m, ch) == ('ItemKNN-ext', 'XM') else 'responsive')
        rows.append(dict(channel=ch, model=m, status=status, contexts=len(z), local_median=z.base.median(), gain_median=med, gain_median_lo=mlo, gain_median_hi=mhi,
                         gain_mean=mea, gain_mean_lo=lo, gain_mean_hi=hi, better=int((z.d > 0.01).sum()), worse=int((z.d < -0.01).sum()),
                         exact_zero=int((z.d == 0).sum()), max_abs=float(z.d.abs().max())))
        # coverage
        er = raw[(raw.model == m) & raw.error.notna()]
        for t in ALL:
            why = []
            if t not in base.index:
                e = er[(er.t == t) & (er.k == 0)].error
                why.append('local: ' + (e.iloc[0] if len(e) else 'not run (grid stopped)'))
            if t not in full.index:
                e = er[(er.t == t) & (er.channel == ch) & (er.order == 'full')].error
                why.append('pooled: ' + (e.iloc[0] if len(e) else ('no donor in this channel' if t not in HAS[ch] else 'not run (grid stopped)')))
            if why: cov.append(dict(model=m, channel=ch, t=t, reason='; '.join(why)))
R = pd.DataFrame(rows); C = pd.DataFrame(cov)
# check: ItemKNN-ext == ItemKNN on XM (same contexts)
a = ok[(ok.channel == 'XM') & (ok.order == 'full') & (ok.frac == 1)].pivot_table(index='t', columns='model', values='ndcg@10')
b = a[['ItemKNN', 'ItemKNN-ext']].dropna()
print('ItemKNN-ext vs ItemKNN, XM full pooling: contexts', len(b), '| identical NDCG@10 in', int((b['ItemKNN'] == b['ItemKNN-ext']).sum()),
      '| max |diff|', float((b['ItemKNN'] - b['ItemKNN-ext']).abs().max()))
pd.set_option('display.width', 250)
print(R.round(4).to_string())
print('\nMissing contexts per (model, channel):')
for (m, ch), g in C.groupby(['model', 'channel'], sort=False): print(m, ch, len(g), '|', '; '.join(f'{t} [{r}]' for t, r in zip(g.t, g.reason)))
R.to_csv(f'{PAPER_RAW}/aud_variants.csv', index=False, lineterminator='\n'); C.to_csv(f'{PAPER_RAW}/aud_variants_coverage.csv', index=False, lineterminator='\n')

# ---- LaTeX
def f(x, d=3, s=False): return ('{:+.%df}' % d if s else '{:.%df}' % d).format(x)
nice = {'XM': 'XM (same category, other markets)', 'XC': 'XC (same market, other categories)', 'XX': 'XX (other market and category)'}
L = ['% Auto-generated by paper/code/variants.py. Requires booktabs, threeparttable, adjustbox.',
     '\\begin{table}[H]', '\\caption{Variants not reported in the main text: all donors of one channel pooled vs local data only (random split, NDCG@10, complete local data).}',
     '\\label{tab:variants}', '\\centering', '\\small', '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{llcccccc}', '\\toprule',
     'Channel & Model & Contexts & Local only (median) & Gain (median, 95\\% CI) & Gain (mean, 95\\% CI) & Better $>$0.01 & Worse $>$0.01 \\\\', '\\midrule']
for ch in ['XM', 'XC', 'XX']:
    L.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{nice[ch]}}}}} \\\\')
    for _, r in R[R.channel == ch].iterrows():
        tag = {'responsive': '', 'redundant with ItemKNN': '$^b$'}.get(r.status, '$^a$')
        L.append(f' & {r.model}{tag} & {r.contexts} & {f(r.local_median)} & {f(r.gain_median, 4, True)} [{f(r.gain_median_lo, 4, True)}, {f(r.gain_median_hi, 4, True)}] & '
                 f'{f(r.gain_mean, 4, True)} [{f(r.gain_mean_lo, 4, True)}, {f(r.gain_mean_hi, 4, True)}] & {r.better} & {r.worse} \\\\')
    L.append('\\midrule')
L = L[:-1]
miss = C.groupby(['model', 'channel']).size()
L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\begin{tablenotes}\\footnotesize',
      '\\item Gain = NDCG@10 with all donors of the channel pooled minus NDCG@10 with local data only (same model), both re-tuned on validation users. '
      'ItemKNN-ext is ItemKNN whose neighbour set and user profiles may include any product the target\'s evaluation users interacted with in the pooled data '
      '(not only target-catalog products); through users shared between categories of one market it is the only item-based model that can use the XC channel. '
      '$^a$ Provably blind combination: a donor that shares no product with the target catalog leaves the model unchanged (gain exactly 0; '
      f'exactly-zero gains: {", ".join(f"{r.model}/{r.channel} {r.exact_zero}/{r.contexts}" for _, r in R[R.status.str.startswith("blind")].iterrows())}). '
      '$^b$ Identical estimator to ItemKNN on XM, because target users hold no donor-only products. '
      f'Results come from the core grid, which was stopped before completion: {len(raw):,} unit records ({int(raw.error.notna().sum())} with an error); '
      'of the 74 contexts, a row lacks ' + '; '.join(f'{t.replace("_", " ").replace(" and ", " \\& ")} ({", ".join(sorted(set(g.model + "/" + g.channel)))}; {"/".join(sorted(set(g.r)))})' for t, g in C.assign(r=C.reason.str.split(';').str[0].str.split(': ').str[-1].replace({'ease_too_large': 'EASE skips catalogs over 4,000 products'})).groupby('t')) + '. '
      'PureSVD uses an iterative sparse SVD (ARPACK) on catalogs over 1,500 products and a dense eigendecomposition otherwise; its scores are deterministic for a fixed '
      'software stack but are not bit-reproducible across numerical libraries. Better/Worse: gain above $+0.01$ / below $-0.01$. '
      'CIs: two-way cluster bootstrap over countries and categories.', '\\end{tablenotes}', '\\end{table}']
open(f'{REPO}/paper/tables_variants.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('\n'.join(L))
