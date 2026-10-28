"""Tables for the redesigned product-launch experiment (launch3.py).

All CIs: boot.summarize at context level (launch sets averaged within context first).
Outputs: paper/tables_launch.tex (main table on a common context set + full-set version),
paper/tables_launch_S1.tex (per-context longtable, random + calendar splits),
paper/results/l3_*.csv (main tables, per-context values, Figure 8 data, skips, tuning shares).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json, glob
import numpy as np, pandas as pd
import boot

PAPER = os.path.join(REPO, 'paper'); NOTES = os.path.join(PAPER, 'results'); os.makedirs(NOTES, exist_ok=True)
SPLITS = ['random', 'temporal', 'calendar']
MODELS = ['ItemKNN', 'EASE', 'UserKNN', 'MostPop']
ALLM = MODELS + ['Random']
TUNINGS = ['zf', 'vf']
THR = 0.01  # practically meaningful above-chance margin (fixed in the task)
SPLITNAME = {'random': 'Random', 'temporal': 'Temporal', 'calendar': 'Calendar'}
TUNNAME = {'zf': 'zero feedback', 'vf': 'val. feedback'}

# ---------------------------------------------------------------- load
rows = []
for fn in sorted(glob.glob(f'{PAPER_RAW}/l3_*.jsonl')):
    if os.path.basename(fn).startswith('l3_skip_'): continue
    for l in open(fn, encoding='utf-8'):
        r = json.loads(l)
        for tu in TUNINGS:
            x = r[tu]
            rows.append({'split': r['split'], 'model': r['model'], 't': r['t'], 'seed': r['seed'], 'foreign': r['foreign'],
                         'tuning': tu, 'tuned': x['tuned'], 'n_val_users': x['n_val_users'],
                         'L_ndcg': x['launch_ndcg@10'], 'L_rec': x['launch_recall@10'],
                         'A_ndcg': x['all_ndcg@10'], 'A_rec': x['all_recall@10'],
                         'n_users_launch': r['n_users_launch'], 'n_users_all': r['n_users_all'],
                         'n_launched': r['n_launched'], 'n_items': r['n_items'], 'sec': r['sec']})
df = pd.DataFrame(rows).drop_duplicates(['split', 'model', 't', 'seed', 'foreign', 'tuning'])
# only launch sets run in both arms
df = df[df.groupby(['split', 'model', 't', 'seed', 'tuning']).foreign.transform('nunique') == 2]
skips = []
for fn in sorted(glob.glob(f'{PAPER_RAW}/l3_skip_*.jsonl')):
    for l in open(fn, encoding='utf-8'): skips.append(json.loads(l))
sk = pd.DataFrame(skips)

# ---------------------------------------------------------------- per launch set, paired with chance
W = df.pivot_table(index=['split', 'model', 'tuning', 't', 'seed'], columns='foreign',
                   values=['L_ndcg', 'L_rec', 'A_ndcg', 'A_rec'])
W.columns = [f'{a}_{"F" if b else "L"}' for a, b in W.columns]
W = W.reset_index()
ch = df[(df.model == 'Random') & (df.foreign == 0) & (df.tuning == 'zf')][['split', 't', 'seed', 'L_ndcg', 'A_ndcg']] \
    .rename(columns={'L_ndcg': 'chance', 'A_ndcg': 'chance_all'})
W = W.merge(ch, on=['split', 't', 'seed'], how='inner')
W['dF'] = W['L_ndcg_F'] - W['chance']        # model + foreign minus chance (launched-item outcome)
W['dL'] = W['L_ndcg_L'] - W['chance']        # local only minus chance
W['gain'] = W['L_ndcg_F'] - W['L_ndcg_L']     # gain over local only (secondary)
W['dAll'] = W['A_ndcg_F'] - W['A_ndcg_L']     # overall-outcome change (foreign - local)
C = W.groupby(['split', 'model', 'tuning', 't'], as_index=False).mean(numeric_only=True)  # average launch sets
NU = df[(df.model == 'Random') & (df.foreign == 0) & (df.tuning == 'zf')].groupby(['split', 't']).n_users_launch.mean().rename('users_launch').reset_index()
C = C.merge(NU, on=['split', 't'], how='left')
C['n_sets'] = W.groupby(['split', 'model', 'tuning', 't']).size().values
C['market'] = C.t.str.split(' ').str[0]; C['category'] = C.t.str.split(' ').str[1]
C.drop(columns=['seed']).to_csv(f'{NOTES}/l3_context.csv', index=False, lineterminator='\n')

# common context set per split: contexts where all five models have results
common = {}
for s in SPLITS:
    sets = [set(C[(C.split == s) & (C.model == m) & (C.tuning == 'zf')].t) for m in ALLM]
    common[s] = set.intersection(*sets) if sets else set()


def summ(sub, col, label):
    if len(sub) == 0: return (np.nan, np.nan, np.nan)
    return boot.summarize(sub[col].values, sub.market.values, sub.category.values, stat='median', label=label)


def main_table(which):
    out = []
    for s in SPLITS:
        for m in ALLM:
            for tu in (TUNINGS if m != 'Random' else ['zf']):
                sub = C[(C.split == s) & (C.model == m) & (C.tuning == tu)]
                if which == 'common': sub = sub[sub.t.isin(common[s])]
                if len(sub) == 0: continue
                lab = f'l3|{which}|{s}|{m}|{tu}'
                d = summ(sub, 'dF', lab + '|dF'); g = summ(sub, 'gain', lab + '|gain'); a = summ(sub, 'dAll', lab + '|dAll')
                out.append({'split': s, 'model': m, 'tuning': tu, 'contexts': len(sub),
                            'ndcg_foreign': float(np.median(sub.L_ndcg_F)), 'ndcg_local': float(np.median(sub.L_ndcg_L)),
                            'chance': float(np.median(sub.chance)),
                            'dchance_med': d[0], 'dchance_lo': d[1], 'dchance_hi': d[2],
                            'above_chance_F': int((sub.dF > THR).sum()), 'above_chance_L': int((sub.dL > THR).sum()),
                            'recall_foreign': float(np.median(sub.L_rec_F)),
                            'dall_med': a[0], 'dall_lo': a[1], 'dall_hi': a[2],
                            'all_ndcg_local': float(np.median(sub.A_ndcg_L)), 'all_ndcg_foreign': float(np.median(sub.A_ndcg_F)),
                            'gain_med': g[0], 'gain_lo': g[1], 'gain_hi': g[2],
                            'gain_gt_thr': int((sub.gain > THR).sum())})
    return pd.DataFrame(out)


MT = {w: main_table(w) for w in ('common', 'full')}
for w, t in MT.items():
    t.to_csv(f'{NOTES}/l3_main_{w}.csv', index=False, lineterminator='\n')

# Figure 8 data: difference from chance per context, foreign vs local
C[['split', 'model', 'tuning', 't', 'market', 'category', 'n_sets', 'chance', 'L_ndcg_L', 'L_ndcg_F', 'dL', 'dF', 'dAll']] \
    .assign(in_common=lambda x: [tt in common[s] for s, tt in zip(x.split, x.t)]) \
    .to_csv(f'{NOTES}/l3_fig8.csv', index=False, lineterminator='\n')

# tuning shares (runs = launch set x arm; models with a grid only)
g = df[df.model.isin(['ItemKNN', 'EASE', 'UserKNN'])]
TS = g.groupby(['split', 'model', 'tuning']).tuned.agg(['mean', 'size']).reset_index().rename(columns={'mean': 'share_tuned', 'size': 'runs'})
TSall = g.groupby(['tuning']).tuned.agg(['mean', 'size']).reset_index()
TSsplit = g.groupby(['split', 'tuning']).tuned.agg(['mean', 'size']).reset_index()
TS.to_csv(f'{NOTES}/l3_tuning_shares.csv', index=False, lineterminator='\n')

# skips
if len(sk):
    sk['level'] = np.where(sk.seed == -2, 'context (model-specific)', np.where(sk.seed == -1, 'context', 'launch set'))
    sk.to_csv(f'{NOTES}/l3_skips.csv', index=False, lineterminator='\n')
    SK = sk.groupby(['split', 'model', 'level', 'reason']).t.agg(['size', 'nunique']).reset_index() \
        .rename(columns={'size': 'n_records', 'nunique': 'n_contexts'})
    SK.to_csv(f'{NOTES}/l3_skips_summary.csv', index=False, lineterminator='\n')

# descriptives
DS = df[(df.foreign == 0) & (df.tuning == 'zf') & (df.model == 'Random')].groupby('split').agg(
    contexts=('t', 'nunique'), launch_sets=('t', 'size'), med_launched=('n_launched', 'median'),
    med_launch_users=('n_users_launch', 'median'), med_all_users=('n_users_all', 'median')).reset_index()
DS.to_csv(f'{NOTES}/l3_descriptives.csv', index=False, lineterminator='\n')


# ---------------------------------------------------------------- LaTeX
def f3(x): return f'{x:.3f}'
def sg(x): return f'{x:+.3f}'
def ci(m, lo, hi): return f'{sg(m)} [{sg(lo)}, {sg(hi)}]'


def latex_main(t, which, caption, label):
    L = [r'\begin{table}[H]', r'\footnotesize', rf'\caption{{{caption}\label{{{label}}}}}',
         r'\begin{adjustwidth}{-\extralength}{0cm}', r'\centering', r'\begin{threeparttable}',
         r'\begin{tabular}{lllccccccc}', r'\toprule',
         r'Split & Model & Tuning & Ctx & +Foreign & $-$Chance (median, 95\% CI) & $>$Chance+0.01 & Recall@10 & Overall change (median, 95\% CI) & Gain vs local \\',
         r'\midrule']
    for s in SPLITS:
        ts = t[t.split == s]
        if not len(ts): continue
        for _, r in ts.iterrows():
            mname = 'Random (chance)' if r.model == 'Random' else r.model
            if r.model == 'Random':
                L.append(f'{SPLITNAME[s]} & {mname} & -- & {r.contexts} & {f3(r.ndcg_foreign)} & -- & -- & {r.recall_foreign:.2f} & -- & -- \\\\')
            else:
                L.append(f'{SPLITNAME[s]} & {mname} & {r.tuning} & {r.contexts} & {f3(r.ndcg_foreign)} & '
                         f'{ci(r.dchance_med, r.dchance_lo, r.dchance_hi)} & {r.above_chance_F}/{r.contexts} ({r.above_chance_L}) & '
                         f'{r.recall_foreign:.2f} & {ci(r.dall_med, r.dall_lo, r.dall_hi)} & {sg(r.gain_med)} \\\\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    cs = ', '.join(f'{SPLITNAME[s]} {len(common[s])}' for s in SPLITS)
    note = (r'Launched-item outcome: NDCG@10 and Recall@10 over users whose test interactions include a launched product, '
            r'truth restricted to launched products; values are medians over contexts after averaging launch sets within a context. '
            r'+Foreign: model fitted on local plus all same-category foreign training data. $-$Chance: paired difference model+foreign minus the random ranker, '
            r'median over contexts with a two-way (country, category) cluster-bootstrap 95\% CI. $>$Chance+0.01: contexts where that difference exceeds 0.01 '
            r'(in parentheses: the same count for the local-only model). Overall change: NDCG@10 over all test users (full test truth, at most 20,000 users), '
            r'foreign minus local only. Gain vs local: launched-item NDCG@10, foreign minus local only (median; local-only scores are near zero by construction). '
            r'Tuning zf: hyperparameters selected on validation interactions with non-launched products only (strictly zero local feedback on launched products); '
            r'vf: selected on the hidden validation interactions with launched products. Either needs $\geq$10 validation users, otherwise the first grid setting is used. '
            r'Random/temporal: 3 launch sets per context, 30\% of eligible products (local training interactions and foreign training interactions, chosen without the test set). '
            r'Calendar: products outside the pre-cutoff local catalog (not retained by the pre-cutoff core filter) that are in a foreign pre-cutoff catalog and have a local interaction after the validation cutoff. EASE skips catalogs over 4,000 products.')
    if which == 'common':
        note += f' Common context set (all five models available): {cs}.'
    else:
        note += ' All contexts available for each model (sets differ across models).'
    L += [r'\end{tabular}', r'\begin{tablenotes}\footnotesize', r'\item ' + note, r'\end{tablenotes}',
          r'\end{threeparttable}', r'\end{adjustwidth}', r'\end{table}']
    return '\n'.join(L)


tex = ['% generated by paper/code/launch3_tables.py -- do not edit by hand',
       latex_main(MT['common'], 'common', 'Product launch: launched-item and overall NDCG@10, with paired differences from chance (common context set).', 'tab:launch'),
       '',
       latex_main(MT['full'], 'full', 'Product launch, all available contexts per model.', 'tab:launch_full')]
open(f'{PAPER}/tables_launch.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(tex) + '\n')


# S1: per context, random + calendar, zf tuning
def esc(x): return x.replace('_', r'\_')


S1 = [r'% generated by paper/code/launch3_tables.py -- do not edit by hand', r'\begingroup\scriptsize', r'\setlength{\tabcolsep}{2.5pt}']
for part, s in (('A', 'random'), ('B', 'calendar')):
    sub = C[(C.split == s) & (C.tuning == 'zf')]
    ctx = sorted(set(sub[sub.model == 'Random'].t))
    head = r'Context & Users & Chance & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{m}}}' for m in MODELS) + r' \\'
    sub2 = r' & & & ' + ' & '.join([r'Local & Foreign & $\Delta$'] * len(MODELS)) + r' \\'
    S1 += [r'\begin{longtable}{lrr' + 'rrr' * len(MODELS) + '}',
           rf'\caption{{Table S1{part}: Product launch per context, {s} split (launched-item NDCG@10; zero-local-feedback tuning; $\Delta$ = foreign minus chance).}}\\',
           r'\toprule', head, sub2, r'\midrule', r'\endfirsthead', r'\toprule', head, sub2, r'\midrule', r'\endhead',
           r'\bottomrule', r'\endfoot']
    for t in ctx:
        r0 = sub[(sub.model == 'Random') & (sub.t == t)].iloc[0]
        cells = [esc(t), f'{r0.users_launch:.0f}', f'{r0.chance:.3f}']
        for m in MODELS:
            q = sub[(sub.model == m) & (sub.t == t)]
            if len(q): q = q.iloc[0]; cells += [f'{q.L_ndcg_L:.3f}', f'{q.L_ndcg_F:.3f}', f'{q.dF:+.3f}']
            else: cells += ['--', '--', '--']
        S1.append(' & '.join(cells) + r' \\')
    S1.append(r'\end{longtable}')
S1.append(r'\endgroup')
open(f'{PAPER}/tables_launch_S1.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(S1) + '\n')

# ---------------------------------------------------------------- print
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
for w in ('common', 'full'):
    print(f'\n=== MAIN TABLE ({w})')
    print(MT[w].round(3).to_string(index=False))
print('\ncommon context sets:', {s: len(v) for s, v in common.items()})
print('\n=== tuning shares (ItemKNN/EASE/UserKNN runs)'); print(TS.round(3).to_string(index=False))
print(TSsplit.round(3).to_string(index=False)); print(TSall.round(3).to_string(index=False))
if len(sk): print('\n=== skips'); print(SK.to_string(index=False))
print('\n=== descriptives'); print(DS.to_string(index=False))
