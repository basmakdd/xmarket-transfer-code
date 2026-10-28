"""Tables and figure data for the scarcity experiment (records from scarcity2.py).

All thresholds fixed in advance: levels f in {0.05,0.10,0.25,0.50}; gap rule Q_full - Q_f > 0.01;
history buckets 0 / 1 / 2+ retained scoring-history interactions; fallback at f = 0.10.
Every CI: boot.summarize, context level (subsamples averaged within context first).
Outputs: paper/tables_scarcity.tex, paper/results/sc_*.csv.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, glob
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import boot

NOTES = f'{REPO}/paper/results'; os.makedirs(NOTES, exist_ok=True)
TEX = f'{REPO}/paper/tables_scarcity.tex'
LEVELS = (0.05, 0.10, 0.25, 0.50)
ARMS = ('P', 'R', 'B')
ARMLAB = {'P': 'Patterns scarce', 'R': 'Histories scarce', 'B': 'Both scarce'}
MS = [('ItemKNN', 'random'), ('EASE', 'random'), ('UserKNN', 'random'),
      ('ItemKNN', 'temporal'), ('ItemKNN', 'calendar'), ('EASE', 'calendar')]
GAPMIN = 0.01
CAT = {'Arts_Crafts_and_Sewing': 'Arts \\& Crafts', 'Automotive': 'Automotive', 'Books': 'Books', 'CDs_and_Vinyl': 'CDs \\& Vinyl',
       'Cell_Phones_and_Accessories': 'Cell Phones', 'Digital_Music': 'Digital Music', 'Electronics': 'Electronics',
       'Grocery_and_Gourmet_Food': 'Grocery', 'Home_and_Kitchen': 'Home \\& Kitchen', 'Industrial_and_Scientific': 'Industrial',
       'Movies_and_TV': 'Movies \\& TV', 'Musical_Instruments': 'Musical Instr.', 'Office_Products': 'Office',
       'Sports_and_Outdoors': 'Sports', 'Toys_and_Games': 'Toys'}
MK = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk']


def load():
    rows = []
    for fn in sorted(glob.glob(f'{PAPER_RAW}/sc_*_*.jsonl')):
        rows += [json.loads(l) for l in open(fn, encoding='utf-8')]
    r = pd.DataFrame(rows)
    bad = r[r.err.notna()]
    r = r[r.err.isna()].drop_duplicates(['model', 'split', 't', 'arm', 'seed', 'f', 'foreign', 'fb'])
    # keep only complete units: F in both foreign arms, and subsamples whose whole design (all levels x arms x
    # foreign, no-fallback) finished; a run interrupted by the budget deadline leaves partial contexts out
    keep = []
    for (model, split, t), g in r.groupby(['model', 'split', 't']):
        if g[g.arm == 'F'].foreign.nunique() < 2:
            continue
        lv = [f for f in LEVELS if not (model == 'UserKNN' and f == 0.05)]
        sc = g[(g.arm != 'F') & (g.fb == 0) & g.f.isin(lv)].groupby('seed').size()
        seeds = set(sc[sc == len(lv) * len(ARMS) * 2].index)
        if seeds:
            keep.append(g[(g.arm == 'F') | g.seed.isin(seeds)])
    r = pd.concat(keep)
    return r, bad


def mc(ts):
    return [t.split(' ')[0] for t in ts], [t.split(' ')[1] for t in ts]


def summ(vals, ts, label, stat='median'):
    vals = np.asarray(vals, float)
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return np.nan, np.nan, np.nan
    m, c = mc(ts)
    return boot.summarize(vals, m, c, None, stat, label)


def paired(r, model, split, arm, f, fb=0):
    """Per context: seed-averaged local (foreign 0) and +foreign (foreign 1) on seeds run in both arms."""
    g = r[(r.model == model) & (r.split == split) & (r.arm == arm) & (r.f == f) & (r.fb == fb)]
    p = g.pivot_table(index=['t', 'seed'], columns='foreign', values='ndcg')
    if p.empty or 0 not in p or 1 not in p:
        return pd.DataFrame(columns=['y0', 'y1', 'nseeds'])
    p = p.dropna()
    out = p.groupby('t').mean().rename(columns={0: 'y0', 1: 'y1'})
    out['nseeds'] = p.groupby('t').size()
    return out


def full_ref(r, model, split):
    g = r[(r.model == model) & (r.split == split) & (r.arm == 'F')]
    return g.pivot_table(index='t', columns='foreign', values='ndcg')


def fmt(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return ('{:+.%df}' % d if sign else '{:.%df}' % d).format(x)


def ci(lo, hi, d=3, sign=False):
    return f'[{fmt(lo, d, sign)}, {fmt(hi, d, sign)}]'


def table(caption, label, cols, head, rows, notes):
    s = ['\\begin{table}[H]', '\\caption{' + caption + '}', '\\label{' + label + '}', '\\centering', '\\small',
         '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{' + cols + '}', '\\toprule', head, '\\midrule']
    s += rows + ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}']
    if notes:
        s += ['\\begin{tablenotes}\\footnotesize', '\\item ' + notes, '\\end{tablenotes}']
    s += ['\\end{table}']
    return '\n'.join(s)


def main():
    r, bad = load()
    tex = []
    # ---------------- coverage: completed seeds per context (F + all 24 scarce units per seed)
    cov = []
    for model, split in MS:
        g = r[(r.model == model) & (r.split == split)]
        fdone = set(g[g.arm == 'F'].groupby('t').foreign.nunique().loc[lambda x: x == 2].index)
        lv = [f for f in LEVELS if not (model == 'UserKNN' and f == 0.05)]   # UserKNN f=0.05 dropped for budget
        sc = g[(g.arm != 'F') & (g.fb == 0) & g.f.isin(lv)].groupby(['t', 'seed']).size()
        for t in sorted(set(g.t)):
            seeds = [int(s) for (tt, s), n in sc.items() if tt == t and n == len(lv) * len(ARMS) * 2]
            cov.append({'model': model, 'split': split, 't': t, 'F_done': t in fdone, 'complete_seeds': ','.join(map(str, seeds))})
        for _, b in bad[(bad.model == model) & (bad.split == split)].iterrows():
            cov.append({'model': model, 'split': split, 't': b.t, 'F_done': False, 'complete_seeds': '',
                        'excluded': b.err, 'note': f"arm={b.get('arm')} f={b.get('f')} foreign={b.get('foreign')} sec={b.get('sec')}"})
    cov = pd.DataFrame(cov)
    cov.to_csv(f'{NOTES}/sc_coverage.csv', index=False, lineterminator='\n')

    # ---------------- main results: all model x split x level x arm
    res = []
    for model, split in MS:
        F = full_ref(r, model, split)
        if F.empty:
            continue
        Fb = F.dropna()
        # full-data reference row
        if len(Fb):
            gain = (Fb[1] - Fb[0]).values
            m, lo, hi = summ(gain, Fb.index, f'sc|full|{model}|{split}')
            res.append({'model': model, 'split': split, 'f': 1.0, 'arm': 'F', 'n_all': len(Fb), 'local_med': Fb[0].median(),
                        'foreign_med': Fb[1].median(), 'gain_med': m, 'gain_lo': lo, 'gain_hi': hi})
        for f in LEVELS:
            for arm in ARMS:
                Z = paired(r, model, split, arm, f)
                Z = Z.join(F[0].rename('full'), how='inner').dropna(subset=['y0', 'y1', 'full'])
                if Z.empty:
                    continue
                gain = Z.y1 - Z.y0
                gm, glo, ghi = summ(gain.values, Z.index, f'sc|gain|{model}|{split}|{f}|{arm}')
                gap = Z.full - Z.y0
                Zg = Z[gap > GAPMIN]
                G = (Zg.y1 - Zg.y0) / (Zg.full - Zg.y0)
                Gm, Glo, Ghi = summ(G.values, Zg.index, f'sc|G|{model}|{split}|{f}|{arm}')
                res.append({'model': model, 'split': split, 'f': f, 'arm': arm, 'n_all': len(Z),
                            'seeds_min': int(Z.nseeds.min()), 'n_3seeds': int((Z.nseeds == 3).sum()),
                            'full_med': Z.full.median(), 'local_med': Z.y0.median(), 'foreign_med': Z.y1.median(),
                            'gain_med': gm, 'gain_lo': glo, 'gain_hi': ghi, 'gain_mean': gain.mean(),
                            'n_pos': int((gain > 0.01).sum()), 'n_neg': int((gain < -0.01).sum()),
                            'n_gap': len(Zg), 'n_removed': len(Z) - len(Zg), 'G_med': Gm, 'G_lo': Glo, 'G_hi': Ghi,
                            'gap_med': gap.median()})
    res = pd.DataFrame(res)
    res['partial'] = (res.model == 'UserKNN') & (res.f == 0.05)
    res.to_csv(f'{NOTES}/sc_results.csv', index=False, lineterminator='\n')
    fig = res[(res.arm != 'F') & ~res.partial][['model', 'split', 'f', 'arm', 'n_all', 'gain_med', 'gain_lo', 'gain_hi', 'n_gap', 'G_med', 'G_lo', 'G_hi']]
    fig.to_csv(f'{NOTES}/sc_figure7_data.csv', index=False, lineterminator='\n')

    def main_rows(sub):
        rows = []
        for (model, split), g in sub.groupby(['model', 'split'], sort=False):
            for _, x in g.iterrows():
                if x.arm == 'F':
                    rows.append(f'{split.capitalize()} & {model} & 1.00 & Complete data & {x.n_all} & {fmt(x.local_med)} & {fmt(x.foreign_med)} & '
                                f'{fmt(x.gain_med, 3, True)} {ci(x.gain_lo, x.gain_hi, 3, True)} & -- & -- \\\\')
                else:
                    rows.append(f'{split.capitalize()} & {model} & {x.f:.2f} & {ARMLAB[x.arm]} & {x.n_all} & {fmt(x.local_med)} & {fmt(x.foreign_med)} & '
                                f'{fmt(x.gain_med, 3, True)} {ci(x.gain_lo, x.gain_hi, 3, True)} & {int(x.n_gap)} ({int(x.n_removed)}) & '
                                f'{fmt(x.G_med, 2)} {ci(x.G_lo, x.G_hi, 2)} \\\\')
            rows.append('\\midrule')
        return rows[:-1]

    order = {(m, s): k for k, (m, s) in enumerate(MS)}
    res['o'] = res.apply(lambda x: order[(x.model, x.split)], axis=1)
    res['ao'] = res.arm.map({'F': 0, 'P': 1, 'B': 2, 'R': 3})
    res = res.sort_values(['o', 'ao', 'f'])   # complete-data row first, then P, B, R by level
    head = ('Split & Model & $f$ & Local data scarce & Contexts & Local & +Foreign & Gain, median [95\\% CI] & '
            'Contexts with gap $>$0.01 (removed) & Gap closed $G$, median [95\\% CI] \\\\')
    note_rule = ('Local = local data only, +Foreign = with all same-category foreign data in model fitting; both medians over '
                 'all completed contexts. Gain = +Foreign $-$ Local per context (subsamples averaged first), median over all '
                 'completed contexts. $G$ = Gain / ($Q_{100\\%}$ $-$ Local) over contexts with $Q_{100\\%}$ $-$ Local $>$ 0.01; '
                 'the number in parentheses is how many completed contexts this rule removes. CIs: two-way cluster bootstrap over '
                 'countries and categories (context level). Hyperparameters re-tuned on validation data in every arm.')
    main = res[res.f.isin([1.0, 0.10])]
    tex.append(table('Value of foreign data when 10\\% of local training interactions are kept.', 'tab:scarcity2',
                     'llcllccccc', head, main_rows(main), note_rule))
    tex.append(table('Value of foreign data at every retention level $f$ (all model--split runs).', 'tab:scarcity2_levels',
                     'llcllccccc', head, main_rows(res[(res.f != 1.0) & ~res.partial]),
                     note_rule + ' UserKNN at $f=0.05$ was dropped to fit the compute budget (only a partial run exists; see notes).'))

    # ---------------- history buckets
    bk = []
    for model, split in MS:
        g = r[(r.model == model) & (r.split == split) & (r.fb == 0) & (r.arm != 'F')]
        for f in LEVELS:
            for arm in ARMS:
                q = g[(g.f == f) & (g.arm == arm)]
                if q.empty:
                    continue
                q0 = q[q.foreign == 0].set_index(['t', 'seed']); q1 = q[q.foreign == 1].set_index(['t', 'seed'])
                idx = q0.index.intersection(q1.index)
                per = []
                for k in idx:
                    hn = q0.at[k, 'h_n']; m0 = q0.at[k, 'h_ndcg']; m1 = q1.at[k, 'h_ndcg']; tot = sum(hn)
                    for b in range(3):
                        per.append({'t': k[0], 'seed': k[1], 'b': b, 'share': hn[b] / tot if tot else np.nan, 'n': hn[b],
                                    'y0': m0[b], 'y1': m1[b]})
                per = pd.DataFrame(per)
                if per.empty:
                    continue
                per['gain'] = per.y1.astype(float) - per.y0.astype(float)
                for b in range(3):
                    pb = per[per.b == b]
                    share = pb.groupby('t').share.mean()
                    pg = pb[pb.n > 0].groupby('t')[['gain', 'y0', 'y1']].mean()
                    gm, glo, ghi = summ(pg.gain.values, pg.index, f'sc|bucket|{model}|{split}|{f}|{arm}|{b}')
                    pooled = pb.n.sum() / per.groupby(['t', 'seed']).n.sum().sum()
                    bk.append({'model': model, 'split': split, 'f': f, 'arm': arm, 'bucket': ['0', '1', '2+'][b],
                               'share_med': share.median(), 'share_mean': share.mean(), 'share_pooled': pooled,
                               'n_ctx_nonempty': len(pg), 'local_med': pg.y0.median() if len(pg) else np.nan,
                               'foreign_med': pg.y1.median() if len(pg) else np.nan,
                               'gain_med': gm, 'gain_lo': glo, 'gain_hi': ghi, 'gain_mean': pg.gain.mean() if len(pg) else np.nan})
    bk = pd.DataFrame(bk)
    bk.to_csv(f'{NOTES}/sc_buckets.csv', index=False, lineterminator='\n')
    rows = []
    for (model, split), g in bk[bk.f == 0.10].groupby(['model', 'split'], sort=False):
        for arm in ('P', 'B', 'R'):
            for _, x in g[g.arm == arm].iterrows():
                rows.append(f'{split.capitalize()} & {model} & {ARMLAB[arm]} & {x.bucket} & {fmt(x.share_med, 2)} ({fmt(x.share_pooled, 2)}) & '
                            f'{x.n_ctx_nonempty} & {fmt(x.local_med)} & {fmt(x.foreign_med)} & {fmt(x.gain_med, 3, True)} {ci(x.gain_lo, x.gain_hi, 3, True)} \\\\')
        rows.append('\\midrule')
    tex.append(table('Results by retained scoring history at $f=0.10$ (0, 1 or 2+ retained local training interactions of the test user).',
                     'tab:scarcity2_buckets', 'lllcccccc',
                     'Split & Model & Local data scarce & History & Share of users, median (pooled) & Contexts & Local & +Foreign & Gain, median [95\\% CI] \\\\',
                     rows[:-1],
                     'Share = share of evaluated test users in the bucket (median over contexts; pooled over all users in parentheses). '
                     'Local, +Foreign and Gain = mean NDCG@10 of the users in the bucket, per context (subsamples averaged), then median over '
                     'contexts with at least one such user. The bucket of a user is the same with and without foreign data, so gains are paired. '
                     'In the patterns-scarce arm histories are complete, so its buckets describe complete histories. Without a fallback, '
                     'a user with an empty history receives an all-zero score row and is ranked in catalog (ASIN) order after masking.'))

    # ---------------- fallback
    fbr = []
    for model, split in [('ItemKNN', 'random'), ('EASE', 'random')]:
        F = full_ref(r, model, split)
        for arm in ARMS:
            Zn = paired(r, model, split, arm, 0.10, 0); Zf = paired(r, model, split, arm, 0.10, 1)
            Z = Zn.join(Zf, rsuffix='_fb', how='inner').join(F[0].rename('full'), how='inner').dropna(subset=['y0', 'y1', 'y0_fb', 'y1_fb', 'full'])
            if Z.empty:
                continue
            d = {'model': model, 'split': split, 'arm': arm, 'n_all': len(Z),
                 'local_med': Z.y0.median(), 'local_fb_med': Z.y0_fb.median(),
                 'foreign_med': Z.y1.median(), 'foreign_fb_med': Z.y1_fb.median()}
            for nm, v in [('fbeffect_local', Z.y0_fb - Z.y0), ('fbeffect_foreign', Z.y1_fb - Z.y1), ('gain_nofb', Z.y1 - Z.y0), ('gain_fb', Z.y1_fb - Z.y0_fb)]:
                m, lo, hi = summ(v.values, Z.index, f'sc|fb|{nm}|{model}|{arm}')
                d.update({nm + '_med': m, nm + '_lo': lo, nm + '_hi': hi})
            Zg = Z[(Z.full - Z.y0_fb) > GAPMIN]
            G = (Zg.y1_fb - Zg.y0_fb) / (Zg.full - Zg.y0_fb)
            m, lo, hi = summ(G.values, Zg.index, f'sc|fb|G|{model}|{arm}')
            d.update({'n_gap_fb': len(Zg), 'n_removed_fb': len(Z) - len(Zg), 'G_fb_med': m, 'G_fb_lo': lo, 'G_fb_hi': hi})
            fbr.append(d)
    fbr = pd.DataFrame(fbr)
    fbr.to_csv(f'{NOTES}/sc_fallback.csv', index=False, lineterminator='\n')
    rows = []
    for _, x in fbr.iterrows():
        rows.append(f'{x.model} & {ARMLAB[x.arm]} & {x.n_all} & {fmt(x.local_med)} & {fmt(x.local_fb_med)} & '
                    f'{fmt(x.fbeffect_local_med, 3, True)} {ci(x.fbeffect_local_lo, x.fbeffect_local_hi, 3, True)} & {fmt(x.foreign_fb_med)} & '
                    f'{fmt(x.gain_fb_med, 3, True)} {ci(x.gain_fb_lo, x.gain_fb_hi, 3, True)} & {x.n_gap_fb} ({x.n_removed_fb}) & {fmt(x.G_fb_med, 2)} {ci(x.G_fb_lo, x.G_fb_hi, 2)} \\\\')
    tex.append(table('Popularity fallback for users with an empty scoring history ($f=0.10$, random split).', 'tab:scarcity2_fallback',
                     'llcccccccc',
                     'Model & Local data scarce & Contexts & Local & Local + fallback & Fallback effect & +Foreign + fallback & Gain with fallback & Contexts with gap $>$0.01 (removed) & $G$ with fallback \\\\',
                     rows,
                     'Fallback: a user with no retained scoring-history interaction is ranked by item popularity in the model-fitting data '
                     '(local, plus foreign data in the +Foreign arm) instead of in catalog order; other users are unchanged. Fallback effect = '
                     'Local + fallback $-$ Local. Gain with fallback = (+Foreign + fallback) $-$ (Local + fallback). $G$ uses the same complete-data '
                     'reference $Q_{100\\%}$ (no fallback needed there). Medians over contexts, CIs from the two-way cluster bootstrap.'))

    # ---------------- per-context G matrix, f=0.10, ItemKNN random, patterns scarce
    F = full_ref(r, 'ItemKNN', 'random'); Z = paired(r, 'ItemKNN', 'random', 'P', 0.10).join(F[0].rename('full'), how='inner')
    Z['gap'] = Z.full - Z.y0; Z['gain'] = Z.y1 - Z.y0; Z['G'] = np.where(Z.gap > GAPMIN, Z.gain / Z.gap, np.nan)
    Z.to_csv(f'{NOTES}/sc_matrix_itemknn_random_f010.csv', lineterminator='\n')
    print('A4 matrix: contexts', len(Z), 'G defined', int(Z.G.notna().sum()), 'G>0', int((Z.G > 0).sum()), 'G>1', int((Z.G > 1).sum()),
          'G<0', int((Z.G < 0).sum()), 'seed-0 only', int((Z.nseeds < 3).sum()))
    print('A4 G>1 contexts:', Z[Z.G > 1].assign(full=Z.full.round(3), G=Z.G.round(2))[['full', 'G']].to_dict('index'))
    print('A4 G<0 contexts:', Z[Z.G < 0].assign(full=Z.full.round(3), G=Z.G.round(2), gain=Z.gain.round(3))[['full', 'gain', 'G']].to_dict('index'))
    cats = sorted({t.split(' ')[1] for t in Z.index})
    rows = []
    for cat in cats:
        cells = []
        for m in MK:
            t = f'{m} {cat}'
            if t not in Z.index:
                cells.append('')
            else:
                x = Z.loc[t]
                s = fmt(x.G, 2) if not np.isnan(x.G) else f'({fmt(x.gain, 3, True)})'
                if x.nseeds < 3:
                    s += '$^{\\dagger}$'
                cells.append(s)
        rows.append(f'{CAT.get(cat, cat)} & ' + ' & '.join(cells) + ' \\\\')
    tex.append(table('Gap closed under pattern scarcity per context ($f=0.10$, ItemKNN, random split).', 'tab:mat_scarcity2',
                     'lrrrrrrr', 'Category & CA & DE & FR & IN & JP & MX & UK \\\\', rows,
                     f'Cells give $G$; where the gap $Q_{{100\\%}} - Q_{{10\\%}}$ is at most {GAPMIN} the absolute gain is shown in parentheses instead. '
                     'Empty cell: context has no same-category foreign data or was not run. $^{\\dagger}$ fewer than 3 subsamples (larger contexts, see coverage notes).'))
    open(TEX, 'w', encoding='utf-8', newline='\n').write('% Generated by paper/code/scarcity2_tables.py -- do not edit by hand\n' + '\n\n'.join(tex) + '\n')

    # ---------------- monotonicity check: local score with a subsample above the complete-data score
    mono = []
    for model, split in MS:
        F = full_ref(r, model, split)
        for f in LEVELS:
            for arm in ARMS:
                Z = paired(r, model, split, arm, f).join(F[0].rename('full'), how='inner').dropna()
                if len(Z):
                    mono.append({'model': model, 'split': split, 'f': f, 'arm': arm, 'n': len(Z),
                                 'n_local_above_full_by_0.01': int((Z.y0 - Z.full > 0.01).sum())})
    mono = pd.DataFrame(mono); mono.to_csv(f'{NOTES}/sc_monotonicity.csv', index=False, lineterminator='\n')
    print(mono.pivot_table(index=['model', 'split'], columns=['arm', 'f'], values='n_local_above_full_by_0.01').to_string())
    # ---------------- console summary
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
    print('records', len(r), 'errors', len(bad)); print(bad[['model', 'split', 't', 'arm', 'f', 'err', 'sec']].to_string() if len(bad) else '')
    print(res.drop(columns=['o', 'ao']).round(3).to_string(index=False))
    print(bk[bk.f == 0.10].round(3).to_string(index=False))
    print(fbr.round(3).T.to_string())


if __name__ == '__main__':
    main()
