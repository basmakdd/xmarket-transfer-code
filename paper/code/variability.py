"""Variability of the scarcity and launch results across repetitions, and the reach of the test-user cap.

Answers the editor's round-2 point that closed-form fitting removes optimisation noise but not the variability from
scarcity subsamples, launch sets and the test-user sample. The headline tables average repetitions within each context
first; this script reports how much the per-context result moves between repetitions, next to the spread between
contexts, so the reader can see that repetition noise is small relative to context heterogeneity.

Inputs (shipped records only; no raw data needed):
  paper/raw/sc_<model>_<split>.jsonl   scarcity runs (arm P/B/R, retention f, seed = subsample, foreign 0/1, fb = fallback)
  paper/raw/l3_<model>_<split>.jsonl   launch runs (seed = launch set; zero-feedback 'zf' block)
  paper/results/rq2_seed_variability.csv (split-seed variability of the complete-data result, from rq2_tables.py)
Definitions (per context c and repetition s):
  scarcity gain g_cs = NDCG@10(10% local + foreign) - NDCG@10(10% local), fallback off, f = 0.10
  launch excess  e_cs = launched-item NDCG@10 with foreign data (zero feedback) - random ranker on the same launch set
  within-context SD = SD over repetitions s (ddof=1), contexts with >= 2 repetitions;
  between-context SD = SD over contexts of the repetition-averaged value; sign flips = contexts whose repetitions
  disagree in sign by more than 0.01 on both sides.
Test-user cap: contexts with more than 20,000 available test users (src/experiment.py MAX_EVAL_USERS) use a fixed sample.
Outputs: paper/results/variability.csv, paper/supp/S5d.tex, paper/logs/aud_variability.log
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, PAPER_RAW

import os, sys, json
import numpy as np, pandas as pd

RAW = PAPER_RAW
CAP = 20_000
LOG = f'{REPO}/paper/logs/aud_variability.log'
os.makedirs(os.path.dirname(LOG), exist_ok=True)
_log = open(LOG, 'w', encoding='utf-8', newline='\n')
def out(*a):
    s = ' '.join(str(x) for x in a); print(s); _log.write(s + '\n')

def rd(fn):
    p = f'{RAW}/{fn}'
    return pd.DataFrame([json.loads(l) for l in open(p, encoding='utf-8')]) if os.path.exists(p) else None

def summarise(v, exp, split, model, arm):
    """v: DataFrame with columns t, seed, val (one value per context and repetition)."""
    g = v.groupby('t').val
    n = g.size(); m = g.mean(); sd = g.std(ddof=1)
    multi = n[n >= 2].index
    flips = int(sum(((v[v.t == t].val > 0.01).any() and (v[v.t == t].val < -0.01).any()) for t in multi)) if len(multi) else np.nan
    return dict(experiment=exp, split=split, model=model, arm=arm, contexts=len(n), contexts_multi=len(multi),
                reps_median=float(n.median()), within_sd_median=float(sd[multi].median()), within_sd_p90=float(sd[multi].quantile(0.9)),
                between_sd=float(m.std(ddof=1)), ratio=float(sd[multi].median() / m.std(ddof=1)), median_of_means=float(m.median()), sign_flips=flips)

SC = pd.read_csv(f'{REPO}/paper/results/sc_results.csv')
rows = []
for model, split in [('ItemKNN', 'random'), ('EASE', 'random'), ('UserKNN', 'random'), ('ItemKNN', 'temporal'), ('ItemKNN', 'calendar'), ('EASE', 'calendar')]:
    d = rd(f'sc_{model}_{split}.jsonl')
    if d is None: continue
    d = d[(d.f == 0.1) & (d.fb == 0) & d.err.isna() & d.ndcg.notna()]
    for arm, lab in [('P', 'patterns scarce'), ('B', 'both scarce'), ('R', 'histories scarce')]:
        x = d[d.arm == arm].pivot_table(index=['t', 'seed'], columns='foreign', values='ndcg', aggfunc='first').dropna()
        if x.empty: continue
        v = pd.DataFrame({'t': x.index.get_level_values(0), 'seed': x.index.get_level_values(1), 'val': (x[1] - x[0]).values})
        rec = summarise(v, 'scarcity gain (f = 10%)', split, model, lab)
        ref = SC[(SC.model == model) & (SC.split == split) & (SC.f == 0.1) & (SC.arm == arm)]
        if len(ref):
            dev = abs(rec['median_of_means'] - ref.gain_med.iloc[0])
            out(f'CHECK scarcity {split} {model} {arm}: median gain {rec["median_of_means"]:.6f} vs sc_results.csv {ref.gain_med.iloc[0]:.6f} (|diff| {dev:.1e})')
            if dev > 1e-9: raise SystemExit('scarcity values do not reproduce scarcity2_tables.py')
        rows.append(rec)

CM = pd.read_csv(f'{REPO}/paper/results/l3_context.csv')   # launch3_tables.py: per-context values and common-set flags via l3_fig8.csv
F8 = pd.read_csv(f'{REPO}/paper/results/l3_fig8.csv')
MC = pd.read_csv(f'{REPO}/paper/results/l3_main_common.csv')
for split in ('random', 'temporal', 'calendar'):
    ch = rd(f'l3_Random_{split}.jsonl')
    if ch is None: continue
    # chance exactly as in launch3_tables.py: random ranker, local-only arm, zero-feedback block
    chance = {(r.t, r.seed): r.zf['launch_ndcg@10'] for r in ch[ch.foreign == 0].itertuples()}
    for model in ('ItemKNN', 'EASE', 'UserKNN', 'MostPop'):
        d = rd(f'l3_{model}_{split}.jsonl')
        if d is None: continue
        both = d.groupby(['t', 'seed']).foreign.transform('nunique') == 2      # launch sets run in both arms only
        d = d[both & (d.foreign == 1)].drop_duplicates(['t', 'seed'])
        com = set(F8[(F8.split == split) & (F8.model == model) & F8.in_common].t) if 'in_common' in F8 else None
        if not com: raise SystemExit(f'empty common context set for {split} {model} (l3_fig8.csv)')
        d = d[d.t.isin(com)]
        v = pd.DataFrame([dict(t=r.t, seed=r.seed, val=r.zf['launch_ndcg@10'] - chance[(r.t, r.seed)]) for r in d.itertuples() if (r.t, r.seed) in chance])
        if v.empty: continue
        rec = summarise(v, 'launch, NDCG@10 minus chance (zero feedback)', split, model, 'launched items')
        ref = MC[(MC.split == split) & (MC.model == model) & (MC.tuning == 'zf')]
        if len(ref):
            dev = abs(rec['median_of_means'] - ref.dchance_med.iloc[0])
            out(f'CHECK launch {split} {model}: median over common contexts {rec["median_of_means"]:.6f} vs Table 5 {ref.dchance_med.iloc[0]:.6f} (|diff| {dev:.1e})')
            if dev > 1e-9: raise SystemExit('launch values do not reproduce launch3_tables.py')
        rows.append(rec)

R = pd.DataFrame(rows)
sv = pd.read_csv(f'{REPO}/paper/results/rq2_seed_variability.csv')
for r in sv.itertuples():
    R = pd.concat([R, pd.DataFrame([dict(experiment='complete data, pooling gain (3 split seeds)', split='random', model=r.model, arm='pool',
                                         contexts=r.contexts, contexts_multi=r.contexts, reps_median=3.0, within_sd_median=r.sd_ctx_median,
                                         within_sd_p90=np.nan, between_sd=np.nan, ratio=np.nan, median_of_means=r.avg_median, sign_flips=np.nan)])], ignore_index=True)

# test-user cap
cap = []
for split in ('random', 'temporal', 'calendar'):
    d = rd(f'l3_Random_{split}.jsonl')
    if d is None: continue
    a = d.groupby('t').n_users_all_available.max()
    cap.append(dict(split=split, contexts=len(a), above_cap=int((a > CAP).sum()), names='; '.join(sorted(a[a > CAP].index))))
C = pd.DataFrame(cap)
R.to_csv(f'{REPO}/paper/results/variability.csv', index=False, lineterminator='\n')
pd.set_option('display.width', 250)
out(R.round(4).to_string(index=False)); out(C.to_string(index=False))

# LaTeX (file S5d.tex = Supplementary Table S5c of the paper)
def f(x, k=3): return '--' if pd.isna(x) else f'{x:.{k}f}'
L = ['% Auto-generated by paper/code/variability.py. Requires booktabs, adjustbox.', '\\begin{table}[H]',
     '\\caption{Table S5c. Variability across repetitions.}', '\\label{tab:S5d}', '\\centering', '\\small',
     '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{lllrrrrrr}', '\\toprule',
     'Quantity & Split & Model & Contexts & Reps & Within SD (median) & Within SD (P90) & Between SD & Sign flips \\\\', '\\midrule']
SHORT = {'patterns scarce': 'Scarcity gain, patterns scarce', 'both scarce': 'Scarcity gain, both scarce',
         'histories scarce': 'Scarcity gain, histories scarce', 'launched items': 'Launch lift over chance', 'pool': 'Pooling gain, complete data'}
SPL = {'random': 'R', 'temporal': 'Ch', 'calendar': 'Ca'}
SPLN = {'random': 'random', 'temporal': 'chronological', 'calendar': 'calendar'}
for r in R.itertuples():
    L.append(f'{SHORT[r.arm]} & {SPL[r.split]} & {r.model} & {r.contexts_multi} & {r.reps_median:.0f} & {f(r.within_sd_median, 4)} & {f(r.within_sd_p90, 4)} & {f(r.between_sd, 4)} & {"--" if pd.isna(r.sign_flips) else int(r.sign_flips)} \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\par\\vspace{2pt}{\\footnotesize',
      'Split: R random, Ch chronological, Ca calendar. Scarcity gains at $f=0.10$; launch lift: launched-item NDCG@10 with foreign data minus the random ranker, zero-feedback tuning, common context set. Contexts: contexts with at least two repetitions. Within SD: standard deviation of the per-context result over repetitions (scarcity subsamples, launch sets or split seeds), median and 90th percentile over contexts with at least two repetitions. '
      'Between SD: standard deviation over contexts of the repetition-averaged result. Sign flips: contexts whose repetitions include values above $+0.01$ and below $-0.01$. '
      'Test-user sample: ' + '; '.join(f'{SPLN[c.split]} split, {c.above_cap} of {c.contexts} contexts above the cap of 20,000' for c in C.itertuples()) +
      '; all other contexts evaluate every test user, so the sample adds no variability there.\\par}', '\\end{table}']
open(f'{REPO}/paper/supp/S5d.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
out('wrote paper/results/variability.csv, paper/supp/S5d.tex')
