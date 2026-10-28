"""RQ1 tables (T1-T5), Figure-4 data and supporting statistics, from paper/raw/rq1_agree{,_r4}.jsonl.

All thresholds and bins are fixed here before looking at results:
  * ratio foreign/local: local agreement >= 0.05, 0.10, 0.20 (all reported);
  * chance level: degree-preserving curveball null for pattern measures (cos, nbr5, ppmi, resid);
    random-ranking expectation for popularity measures (Spearman: 0; top-10 overlap: 10/n), because the
    null preserves product counts and therefore leaves popularity agreement unchanged (checked below);
  * chance-corrected agreement (F - chance) / (L - chance), defined where L - chance >= 0.05 (also 0.10, 0.20 in CSV);
  * top-10 catalog bins: 10-19, 20-49, 50-199, >=200 shared products; sensitivity excludes pairs with < 20.
CIs: boot.summarize (two-way cluster bootstrap, pair level: target market, donor market, category).
Outputs: paper/tables_rq1.tex, paper/results/rq1_T*.csv, paper/results/rq1_fig4_data.csv, paper/results/rq1_stats.txt
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, json, glob, io
import numpy as np, pandas as pd
import boot

NOTES = f'{REPO}/paper/results'; os.makedirs(NOTES, exist_ok=True)
THR = [0.05, 0.10, 0.20]
CC_THR = 0.05
BINS = [(10, 19), (20, 49), (50, 199), (200, 10 ** 9)]
POP, PAT, ADJ = ['sp', 'top10'], ['cos', 'nbr5'], ['ppmi', 'resid']
NAME = {'sp': 'Popularity: rank correlation (Spearman)', 'top10': 'Popularity: top-10 overlap',
        'cos': 'Patterns: co-interaction cosine', 'nbr5': 'Patterns: 5-NN neighbour overlap',
        'ppmi': 'Patterns, popularity-adjusted: PPMI cosine', 'resid': 'Patterns, popularity-adjusted: residual correlation'}
ARM = {'all': ('loc_', 'for_', 'null_'), 'mat': ('mloc_', 'mfor_', 'mnull_')}
ARMNAME = {'all': 'all foreign', 'mat': 'two-sided matched'}
LOG = io.StringIO()


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n')


def load(tag):
    fn = f'{PAPER_RAW}/rq1_agree{tag}.jsonl'
    X = pd.DataFrame([json.loads(l) for l in open(fn, encoding='utf-8')]).drop_duplicates(['t', 'donor_market'])
    X = X.sort_values(['t', 'donor_market']).reset_index(drop=True)
    for c in X.columns:
        if X[c].dtype == object and not c.startswith('splits') and c not in ('t', 'target_market', 'donor_market', 'category'):
            X[c] = pd.to_numeric(X[c])
    return X


def S(v, z, label, stat='median'):
    """median (or mean) + pair-level cluster-bootstrap CI"""
    v = np.asarray(v, float)
    if np.isnan(v).all() or len(v) == 0:
        return (np.nan, np.nan, np.nan)
    return boot.summarize(v, z.target_market.values, z.category.values, z.donor_market.values, stat=stat, label=label)


def chance(X, m, arm):
    if m == 'sp':
        return pd.Series(0.0, index=X.index)
    if m == 'top10':
        return 10.0 / X.n_shared
    return X[ARM[arm][2] + m]


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return ('{:+.%df}' % d if sign else '{:.%df}' % d).format(x)


def fci(t, d=2, sign=False):
    m, lo, hi = t
    if np.isnan(m):
        return '--'
    return f'{fmt(m, d, sign)} [{fmt(lo, d, sign)}, {fmt(hi, d, sign)}]'


def table(caption, label, cols, header, rows, notes):
    s = ['\\begin{table}[H]', '\\caption{' + caption + '}', '\\label{' + label + '}', '\\centering', '\\small',
         '\\begin{adjustbox}{max width=\\linewidth}', '\\begin{tabular}{' + cols + '}', '\\toprule', header, '\\midrule']
    s += rows + ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}']
    if notes:
        s += ['\\begin{tablenotes}\\footnotesize', '\\item ' + notes, '\\end{tablenotes}']
    s += ['\\end{table}']
    return '\n'.join(s)


# ----------------------------------------------------------------------------- T1 / T4 / T5 rows
def measure_rows(X, measures, tag):
    recs = []
    for m in measures:
        for arm in ['all', 'mat']:
            L, F, N = (p + m for p in ARM[arm])
            z = X.dropna(subset=[L, F]).copy()
            ch = chance(z, m, arm)
            r = dict(variant=tag or 'main', measure=m, arm=arm, pairs=len(z),
                     local=z[L].median(), foreign=z[F].median(), null_foreign=z[N].median(),
                     chance_median=float(np.nanmedian(ch)))
            for th in THR:
                zz = z[z[L] >= th]
                r[f'ratio_{th:.2f}'], r[f'ratio_{th:.2f}_lo'], r[f'ratio_{th:.2f}_hi'] = S(zz[F] / zz[L], zz, f'rq1{tag}|ratio|{m}|{arm}|{th}')
                r[f'ratio_{th:.2f}_n'] = len(zz)
            r['D'], r['D_lo'], r['D_hi'] = S(z[F] - z[L], z, f'rq1{tag}|D|{m}|{arm}')
            r['D_pos_share'] = float((z[F] - z[L] > 0).mean())
            ex = z[F] - ch
            r['excess'], r['excess_lo'], r['excess_hi'] = S(ex, z, f'rq1{tag}|excess|{m}|{arm}')
            r['excess_n'] = int(ex.notna().sum())
            r['local_excess'] = float(np.nanmedian(z[L] - ch))
            for th in THR:
                den = z[L] - ch; ok = den >= th
                cc = ((z[F] - ch) / den)[ok]
                r[f'cc_{th:.2f}'], r[f'cc_{th:.2f}_lo'], r[f'cc_{th:.2f}_hi'] = S(cc, z[ok], f'rq1{tag}|cc|{m}|{arm}|{th}')
                r[f'cc_{th:.2f}_n'] = int(ok.sum())
            recs.append(r)
    return pd.DataFrame(recs)


SHORT = {'sp': 'Rank correlation', 'top10': 'Top-10 overlap', 'cos': 'Co-interaction cosine', 'nbr5': '5-NN neighbour overlap',
         'ppmi': 'PPMI cosine', 'resid': 'Residual correlation'}
KIND = {'sp': 'Popularity', 'top10': 'Popularity', 'cos': 'Patterns', 'nbr5': 'Patterns', 'ppmi': 'Patterns (adjusted)', 'resid': 'Patterns (adjusted)'}


def tex_raw_rows(R):
    """raw agreement, D and excess over chance; one row per measure x arm"""
    rows = []
    for m in R.measure.unique():
        for _, r in R[R.measure == m].iterrows():
            lab = f'{KIND[m]}: {SHORT[m]}' if r.arm == 'all' else ''
            rows.append(f"{lab} & {ARMNAME[r.arm]} & {r.pairs} & {fmt(r.local)} & {fmt(r.foreign)} & {fmt(r.null_foreign)} & "
                        f"{fci((r.D, r.D_lo, r.D_hi), 2, True)} & {100 * r.D_pos_share:.0f}\\% & {fci((r.excess, r.excess_lo, r.excess_hi), 2, True)} \\\\")
        rows.append('\\midrule')
    return rows[:-1]


RAWHEAD = ('Measure & Foreign data & Pairs & Local $L$ & Foreign $F$ & Null $F_0$ & $D=F-L$ (95\\% CI) & $D>0$ & $F-c$ (95\\% CI) \\\\')


def tex_ratio_rows(R):
    rows = []
    for m in R.measure.unique():
        for _, r in R[R.measure == m].iterrows():
            lab = f'{KIND[m]}: {SHORT[m]}' if r.arm == 'all' else ''
            ratios = ' & '.join(f"{fci((r[f'ratio_{t:.2f}'], r[f'ratio_{t:.2f}_lo'], r[f'ratio_{t:.2f}_hi']))} ({r[f'ratio_{t:.2f}_n']})" for t in THR)
            rows.append(f"{lab} & {ARMNAME[r.arm]} & {ratios} & "
                        f"{fci((r[f'cc_{CC_THR:.2f}'], r[f'cc_{CC_THR:.2f}_lo'], r[f'cc_{CC_THR:.2f}_hi']))} ({r[f'cc_{CC_THR:.2f}_n']}) \\\\")
        rows.append('\\midrule')
    return rows[:-1]


RATIOHEAD = ('Measure & Foreign data & $F/L$, $L\\ge0.05$ & $F/L$, $L\\ge0.10$ & $F/L$, $L\\ge0.20$ & $(F-c)/(L-c)$, $L-c\\ge0.05$ \\\\')


# ----------------------------------------------------------------------------- T2 paired comparisons
def paired_rows(X, comps, tag):
    recs = []
    for pa, po in comps:
        for arm in ['all', 'mat']:
            L, F, _ = ARM[arm]
            # (1) difference scale
            z = X.dropna(subset=[L + pa, F + pa, L + po, F + po]).copy()
            dd = (z[F + pa] - z[L + pa]) - (z[F + po] - z[L + po])
            r = dict(variant=tag or 'main', pattern=pa, popularity=po, arm=arm, scale='D (F-L)', threshold=np.nan, pairs=len(z),
                     share_pattern_higher=float((dd > 0).mean()))
            r['share_lo'], r['share_hi'] = S((dd > 0).astype(float), z, f'rq1{tag}|share|D|{pa}|{po}|{arm}', 'mean')[1:]
            r['median'], r['lo'], r['hi'] = S(dd, z, f'rq1{tag}|pairD|{pa}|{po}|{arm}')
            recs.append(r)
            # (2) chance-corrected scale
            for th in THR:
                cpa, cpo = chance(z, pa, arm), chance(z, po, arm)
                ok = ((z[L + pa] - cpa) >= th) & ((z[L + po] - cpo) >= th)
                dd = ((z[F + pa] - cpa) / (z[L + pa] - cpa) - (z[F + po] - cpo) / (z[L + po] - cpo))[ok]
                zz = z[ok]
                r = dict(variant=tag or 'main', pattern=pa, popularity=po, arm=arm, scale='chance-corrected', threshold=th, pairs=int(ok.sum()),
                         share_pattern_higher=float((dd > 0).mean()))
                r['share_lo'], r['share_hi'] = S((dd > 0).astype(float), zz, f'rq1{tag}|share|cc|{pa}|{po}|{arm}|{th}', 'mean')[1:]
                r['median'], r['lo'], r['hi'] = S(dd, zz, f'rq1{tag}|paircc|{pa}|{po}|{arm}|{th}')
                recs.append(r)
            # (3) ratio scale (the ratio of the v3 study, now with thresholds)
            for th in THR:
                ok = (z[L + pa] >= th) & (z[L + po] >= th)
                dd = (z[F + pa] / z[L + pa] - z[F + po] / z[L + po])[ok]; zz = z[ok]
                r = dict(variant=tag or 'main', pattern=pa, popularity=po, arm=arm, scale='ratio', threshold=th, pairs=int(ok.sum()),
                         share_pattern_higher=float((dd > 0).mean()))
                r['share_lo'], r['share_hi'] = S((dd > 0).astype(float), zz, f'rq1{tag}|share|ratio|{pa}|{po}|{arm}|{th}', 'mean')[1:]
                r['median'], r['lo'], r['hi'] = S(dd, zz, f'rq1{tag}|pairratio|{pa}|{po}|{arm}|{th}')
                recs.append(r)
    return pd.DataFrame(recs)


def tex_paired_rows(P, scales=(('D (F-L)', None), ('chance-corrected', CC_THR), ('ratio', 0.05), ('ratio', 0.10), ('ratio', 0.20))):
    rows = []
    for (pa, po), g in P.groupby(['pattern', 'popularity'], sort=False):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{SHORT[pa]} vs {SHORT[po].lower()}}}}} \\\\")
        for sc, th in scales:
            q = g[g.scale == sc]
            if th is not None:
                q = q[np.isclose(q.threshold, th)]
            sl = '$D_{pat}-D_{pop}$' if sc == 'D (F-L)' else (f'Chance-corr., $L-c\\ge{th:.2f}$' if sc == 'chance-corrected' else f'Ratio, $L\\ge{th:.2f}$')
            cells = []
            for arm in ['all', 'mat']:
                r = q[q.arm == arm].iloc[0]
                cells.append(f"{r.pairs} & {100 * r.share_pattern_higher:.0f}\\% [{100 * r.share_lo:.0f}, {100 * r.share_hi:.0f}] & {fci((r['median'], r.lo, r.hi), 2, True)}")
            rows.append(f"\\quad {sl} & " + ' & '.join(cells) + ' \\\\')
        rows.append('\\midrule')
    return rows[:-1]


PAIRHEAD = ('& \\multicolumn{3}{c}{All foreign data} & \\multicolumn{3}{c}{Two-sided matched} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n'
            'Scale & Pairs & Patterns higher & Median difference & Pairs & Patterns higher & Median difference \\\\')


# ----------------------------------------------------------------------------- T3 top-10 by catalog size
def top10_rows(X, tag):
    recs = []
    groups = [(f'{a}--{b}' if b < 10 ** 8 else f'$\\ge${a}', (X.n_shared >= a) & (X.n_shared <= b)) for a, b in BINS]
    groups += [('all pairs', X.n_shared >= 10), ('pairs with $\\ge$20', X.n_shared >= 20)]
    for lab, sel in groups:
        for arm in ['all', 'mat']:
            L, F, _ = ARM[arm]
            z = X[sel].dropna(subset=[L + 'top10', F + 'top10']).copy()
            c = 10.0 / z.n_shared
            r = dict(variant=tag or 'main', bin=lab, arm=arm, pairs=len(z), median_shared=float(z.n_shared.median()) if len(z) else np.nan,
                     chance=float(c.median()) if len(z) else np.nan, local=z[L + 'top10'].median(), foreign=z[F + 'top10'].median())
            r['local_excess'], r['local_excess_lo'], r['local_excess_hi'] = S(z[L + 'top10'] - c, z, f'rq1{tag}|t10|lex|{lab}|{arm}')
            r['foreign_excess'], r['foreign_excess_lo'], r['foreign_excess_hi'] = S(z[F + 'top10'] - c, z, f'rq1{tag}|t10|fex|{lab}|{arm}')
            r['D'], r['D_lo'], r['D_hi'] = S(z[F + 'top10'] - z[L + 'top10'], z, f'rq1{tag}|t10|D|{lab}|{arm}')
            ok = (z[L + 'top10'] - c) >= CC_THR
            r['cc'], r['cc_lo'], r['cc_hi'] = S(((z[F + 'top10'] - c) / (z[L + 'top10'] - c))[ok], z[ok], f'rq1{tag}|t10|cc|{lab}|{arm}')
            r['cc_n'] = int(ok.sum())
            ok = z[L + 'top10'] >= 0.10
            r['ratio_0.10'], r['ratio_0.10_lo'], r['ratio_0.10_hi'] = S((z[F + 'top10'] / z[L + 'top10'])[ok], z[ok], f'rq1{tag}|t10|ratio|{lab}|{arm}')
            r['ratio_0.10_n'] = int(ok.sum())
            # paired nbr5 - top10 on the D scale within the bin
            zz = z.dropna(subset=[L + 'nbr5', F + 'nbr5'])
            dd = (zz[F + 'nbr5'] - zz[L + 'nbr5']) - (zz[F + 'top10'] - zz[L + 'top10'])
            r['pair_nbr_minus_top10_D'], r['pair_lo'], r['pair_hi'] = S(dd, zz, f'rq1{tag}|t10|pair|{lab}|{arm}')
            r['pair_n'] = len(zz); r['pair_share'] = float((dd > 0).mean()) if len(zz) else np.nan
            recs.append(r)
    return pd.DataFrame(recs)


def main():
    X = load('')
    say(f'main variant: {len(X)} pairs, {X.t.nunique()} target contexts')
    o = pd.read_parquet(f'{RESULTS}/overlap_random.parquet'); xm = o[o.channel == 'XM']
    say(f'XM pairs in overlap file: {len(xm)}; with >=10 shared products (RQ1 eligible): {len(X)}')

    # ---------------- sanity checks on the null model
    say('\n== Null-model sanity checks ==')
    say('max |null product counts - foreign product counts| over all pairs/draws:', X.null_pop_maxdiff.max())
    say('duplicate user-product edges after curveball (max):', X.null_dup_edges.max())
    for m in POP:
        d = (X['null_' + m] - X['for_' + m]).abs().max()
        say(f'max |null_{m} - for_{m}| = {d:.3g}  (popularity agreement unchanged by the null)')
    say('null trades per pair: median', X.null_trades.median(), '; ratio trades/foreign users: min', (X.null_trades / X.n_foreign_users).min().round(3))
    say('null sd of cosine across 10 draws (median over pairs):', round(X.null_sd_cos.median(), 4))

    # ---------------- pairs per measure, non-positive local agreement, denominators
    say('\n== Pairs per measure (local and foreign both defined) ==')
    for m in POP + PAT + ADJ:
        say(f'{m}: all foreign {X[["loc_" + m, "for_" + m]].notna().all(1).sum()}, matched {X[["mloc_" + m, "mfor_" + m]].notna().all(1).sum()}, '
            f'null {X[["loc_" + m, "null_" + m]].notna().all(1).sum()}')
    say('nbr5 undefined reason: fewer than 5 products with >=5 co-interaction neighbours in both samples')
    say('\n== Pairs with non-positive or undefined local agreement ==')
    bad = []
    for m in POP + PAT + ADJ:
        q = X[~(X['loc_' + m] > 0)]
        for _, r in q.iterrows():
            bad.append(dict(measure=m, t=r.t, donor=r.donor_market, n_shared=r.n_shared, local=r['loc_' + m], foreign=r['for_' + m]))
    B = pd.DataFrame(bad)
    for m in POP + PAT + ADJ:
        q = B[B.measure == m] if len(B) else B
        q1 = q[q.local.notna()] if len(q) else q
        say(f'{m}: non-positive {len(q1)}; undefined {len(q) - len(q1)}')
        if 0 < len(q1) <= 12:
            for _, r in q1.iterrows():
                say(f'    {r.t} <- {r.donor}: n_shared={r.n_shared}, local={r.local:.3f}, foreign={r.foreign:.3f}')
    B.to_csv(f'{NOTES}/rq1_nonpositive_local.csv', index=False, lineterminator='\n')
    say('\n== Denominator (local B vs A) distributions ==')
    den = []
    for m in POP + PAT + ADJ:
        for arm, L in [('all', 'loc_'), ('mat', 'mloc_')]:
            v = X[L + m].dropna()
            row = dict(measure=m, arm=arm, n=len(v), min=v.min(), p05=v.quantile(.05), p25=v.quantile(.25), median=v.median(), p75=v.quantile(.75), max=v.max())
            row.update({f'n_below_{t:.2f}': int((v < t).sum()) for t in THR}); row['n_nonpositive'] = int((v <= 0).sum())
            den.append(row)
    Dn = pd.DataFrame(den); Dn.to_csv(f'{NOTES}/rq1_denominators.csv', index=False, lineterminator='\n'); say(Dn.round(3).to_string(index=False))

    # ---------------- volumes
    say('\n== Volume matching (two-sided) ==')
    say(f'median shared-product interactions: half B {X.nB.median():.0f}, all foreign {X.nF.median():.0f}, target V {X.V.median():.0f}')
    say(f'pairs where all foreign data < half B (nF < nB): {(X.nF < X.nB).sum()} of {len(X)} ({100 * (X.nF < X.nB).mean():.0f}%)')
    say(f'achieved volume / V: half B median {np.median(X.mVB / X.V):.4f} (min {np.min(X.mVB / X.V):.4f}); '
        f'foreign median {np.median(X.mVF / X.V):.4f} (min {np.min(X.mVF / X.V):.4f})')
    say(f'achieved |B - F| / V: median {np.median(np.abs(X.mVB - X.mVF) / X.V):.4f}, 95th pct {np.quantile(np.abs(X.mVB - X.mVF) / X.V, .95):.4f}, max {np.max(np.abs(X.mVB - X.mVF) / X.V):.4f}')
    say(f'empty draws (first user exceeds V): pairs with any {(X.m_empty > 0).sum()}, mean empty draws per split where any {X.m_empty[X.m_empty > 0].mean() if (X.m_empty > 0).any() else 0}')

    # ---------------- shared-product statistic
    say('\n== Share of local interactions on products the foreign store also has ==')
    say(f'overlap_random.parquet (training interactions, all {len(xm)} XM pairs): median {xm.shared_item_interaction_cover.median():.4f}')
    xm10 = xm.merge(X[['target_market', 'donor_market', 'category']], left_on=['target_market', 'donor_market', 'target_category'],
                    right_on=['target_market', 'donor_market', 'category'])
    say(f'same, restricted to the {len(xm10)} RQ1 pairs: median {xm10.shared_item_interaction_cover.median():.4f}')
    say(f'RQ1 definition (all rows, shared = products present in both contexts), {len(X)} pairs: median {X.local_share_on_shared.median():.4f}; '
        f'quartiles {X.local_share_on_shared.quantile(.25):.4f}-{X.local_share_on_shared.quantile(.75):.4f}')
    say(f'shared products per pair: median {X.n_shared.median():.0f}, quartiles {X.n_shared.quantile(.25):.0f}-{X.n_shared.quantile(.75):.0f}, min {X.n_shared.min()}, max {X.n_shared.max()}')

    # ---------------- CA Grocery (+20.74 in Table A2 of the v3 study)
    say('\n== CA Grocery (v3 study, Table A2: pattern-minus-popularity ratio +20.74) ==')
    old = sorted(glob.glob(f'{PAPER_RAW}/travel3_*.pkl'))
    if old:
        Y = pd.concat([pd.read_pickle(f) for f in old]).drop_duplicates(['t', 'donor_market'])
        Y['r_sp'] = (Y.for_sp / Y.loc_sp).where(Y.loc_sp > 0); Y['r_cos'] = (Y.for_cos / Y.loc_cos).where(Y.loc_cos > 0)
        Y['gap'] = Y.r_cos - Y.r_sp
        g = Y[Y.t.str.startswith('ca Grocery')]
        for _, r in g.iterrows():
            say(f'  old travel3 {r.t} <- {r.donor_market}: n_shared={r.n_shared}, loc_sp={r.loc_sp:.4f}, for_sp={r.for_sp:.4f}, ratio_sp={r.r_sp:.2f}, '
                f'loc_cos={r.loc_cos:.4f}, for_cos={r.for_cos:.4f}, ratio_cos={r.r_cos:.2f}, gap={r.gap:+.2f}, nB={r.nB:.0f}, nF={r.nF:.0f}')
        say(f'  v3 Table A2 cell = median over the CA Grocery donors of the gap: {g.gap.median():+.2f}')
        say(f'  old travel3: pairs with |gap| > 3: {(Y.gap.abs() > 3).sum()}; with loc_sp in (0, 0.05): {((Y.loc_sp > 0) & (Y.loc_sp < 0.05)).sum()}')
        say('  old travel3 non-positive local agreement:')
        for m in ['sp', 'top10', 'cos', 'nbr5']:
            q = Y[Y['loc_' + m] <= 0]
            for _, r in q.iterrows():
                say(f'    {m}: {r.t} <- {r.donor_market} local={r["loc_" + m]:.3f}')
        q = Y[Y.loc_cos.isna()]
        for _, r in q.iterrows():
            say(f'    cos undefined: {r.t} <- {r.donor_market}')
    for _, r in X[X.t.str.startswith('ca Grocery')].iterrows():
        say(f'  new {r.t} <- {r.donor_market}: n_shared={r.n_shared}, local interactions on shared products={r.nT_shared}, half B={r.nB:.0f}, foreign={r.nF}; '
            f'loc_sp per split={np.round(r.splits_loc_sp, 3).tolist()}, mean {r.loc_sp:.4f}; for_sp={r.for_sp:.4f}; '
            f'loc_cos={r.loc_cos:.4f}, for_cos={r.for_cos:.4f}, null_cos={r.null_cos:.4f}; D_cos-D_sp={(r.for_cos - r.loc_cos) - (r.for_sp - r.loc_sp):+.3f}')

    # ---------------- tables
    tex = ['% RQ1 tables generated by paper/code/rq1_tables.py -- do not edit by hand']
    R1 = measure_rows(X, POP + PAT, ''); R1.to_csv(f'{NOTES}/rq1_T1.csv', index=False, lineterminator='\n')
    note1 = (f'Local users are split into random halves A and B (3 splits, averaged); half A is the reference of every comparison. '
             f'Two-sided matched: half B and the foreign data are both reduced to $V=\\min(n_B, n_F)$ shared-product interactions by adding whole users '
             f'in random order and stopping before the user who would exceed $V$ (10 draws per split). Null foreign: the foreign user$\\times$product matrix '
             f'randomised by curveball trades ($5\\times$ users per draw, 10 draws), preserving every product\'s and user\'s interaction count; '
             f'the matched null uses the same users as the matched foreign sample. In matched rows, Local is half B after downsampling (equal to half B when half B is the smaller side). Ratio = $F/L$ over pairs with local agreement $L$ at or above the threshold '
             f'(pairs in parentheses). $D=F-L$ uses all pairs; $D>0$ = share of pairs where foreign agreement exceeds local agreement. Chance level $c$: the null agreement for pattern measures; 0 for rank correlation and $10/n$ '
             f'for top-10 overlap ($n$ = shared products), because the null leaves popularity agreement unchanged. Chance-corrected values use pairs with '
             f'$L-c\\ge{CC_THR:.2f}$. Medians over pairs; 95\\% CIs from the two-way cluster bootstrap (countries, with target and donor sharing a draw, and categories).')
    tex.append(table('Cross-market agreement on shared products: raw agreement, foreign-minus-local difference and excess over chance (RQ1).', 'tab:rq1_main',
                     'llccccccc', RAWHEAD, tex_raw_rows(R1), note1))
    tex.append(table('Foreign-to-local agreement ratios under minimum-denominator thresholds, and chance-corrected agreement (RQ1).', 'tab:rq1_ratio',
                     'llcccc', RATIOHEAD, tex_ratio_rows(R1),
                     'Ratio $F/L$: median over pairs whose local agreement $L$ reaches the threshold; chance-corrected $(F-c)/(L-c)$: median over pairs with '
                     f'$L-c\\ge{CC_THR:.2f}$ (values for 0.10 and 0.20 in the CSV file). Pairs in parentheses; 95\\% cluster-bootstrap CIs in brackets. '
                     'Definitions as in Table~\\ref{tab:rq1_main}.'))
    P = paired_rows(X, [('cos', 'sp'), ('nbr5', 'top10'), ('ppmi', 'sp'), ('resid', 'sp')], ''); P.to_csv(f'{NOTES}/rq1_T2.csv', index=False, lineterminator='\n')
    tex.append(table('Paired comparison of pattern and popularity agreement within each country pair (RQ1).', 'tab:rq1_paired', 'lcccccc', PAIRHEAD, tex_paired_rows(P),
                     '$D_{pat}-D_{pop}$: (foreign $-$ local agreement) for the pattern measure minus the same for the popularity measure; no ratio and no threshold. '
                     'Chance-corrected: $(F-c)/(L-c)$ for each measure (chance $c$ as in Table~\\ref{tab:rq1_main}), pattern minus popularity, pairs where both '
                     f'denominators are $\\ge{CC_THR:.2f}$. Ratio: $F/L$ pattern minus popularity, pairs where both local values reach the threshold. '
                     'Shares: share of pairs where the pattern value exceeds the popularity value; CI from the cluster bootstrap of the mean of the indicator. Median difference: median of the paired difference, 95\\% cluster-bootstrap CI.'))
    R3 = top10_rows(X, ''); R3.to_csv(f'{NOTES}/rq1_T3.csv', index=False, lineterminator='\n')
    rows = []
    for _, r in R3.iterrows():
        if r.arm != 'all':
            continue
        rows.append(f"{r.bin} & {r.pairs} & {fmt(r.median_shared, 0)} & {fmt(r.chance)} & {fmt(r.local)} & {fmt(r.foreign)} & "
                    f"{fci((r.local_excess, r.local_excess_lo, r.local_excess_hi), 2, True)} & {fci((r.foreign_excess, r.foreign_excess_lo, r.foreign_excess_hi), 2, True)} & "
                    f"{fci((r.cc, r.cc_lo, r.cc_hi))} ({r.cc_n}) & {fci((r.pair_nbr_minus_top10_D, r.pair_lo, r.pair_hi), 2, True)} ({r.pair_n}) \\\\")
        if r.bin == '$\\ge$200':
            rows.append('\\midrule')
    tex.append(table('Top-10 overlap by size of the shared catalog (all foreign data).', 'tab:rq1_top10', 'lccccccccc',
                     'Shared products & Pairs & Median $n$ & Chance $10/n$ & Local & Foreign & Local $-$ chance & Foreign $-$ chance & Chance-corr. & '
                     '$D_{nbr5}-D_{top10}$ \\\\', rows,
                     'Chance: expected overlap of two random top-10 lists drawn from $n$ shared products ($10/n$; median shown). Chance-corrected '
                     f'$(F-c)/(L-c)$ over pairs with $L-c\\ge{CC_THR:.2f}$ (pairs in parentheses). Last column: paired difference of the 5-NN '
                     'and top-10 measures on the $D=F-L$ scale, pairs where 5-NN overlap is defined. Matched-volume rows are in the CSV file.'))
    R4 = measure_rows(X, ADJ, ''); R4.to_csv(f'{NOTES}/rq1_T4.csv', index=False, lineterminator='\n')
    tex.append(table('Popularity-adjusted pattern measures: thresholded ratios and chance-corrected agreement (RQ1).', 'tab:rq1_adjusted_ratio',
                     'llcccc', RATIOHEAD, tex_ratio_rows(R4), 'As Table~\\ref{tab:rq1_ratio}; the chance level is the curveball null.'))
    tex.append(table('Popularity-adjusted pattern measures (RQ1).', 'tab:rq1_adjusted', 'llccccccc', RAWHEAD, tex_raw_rows(R4),
                     'PPMI: $\\max(0,\\log(c_{ij}N/(n_in_j)))$ with $c_{ij}$ the number of users interacting with both products, $n_i$ the product\'s interaction '
                     'count and $N$ the number of users in the sample; agreement = cosine between upper triangles. Residual: $c_{ij}-n_in_j/N$ over all product '
                     'pairs $i<j$ (zeros included); agreement = Pearson correlation of upper triangles. Columns as in Table~\\ref{tab:rq1_main}; the chance level is '
                     'the curveball null. Paired comparisons with popularity are in Table~\\ref{tab:rq1_paired}.'))

    # ---------------- r4 variant
    if os.path.exists(f'{PAPER_RAW}/rq1_agree_r4.jsonl'):
        X4 = load('_r4')
        o4 = pd.read_parquet(f'{RESULTS}/overlap_random_r4.parquet'); x4 = o4[o4.channel == 'XM']
        say(f'\n== Ratings >= 4 variant: {len(X4)} pairs ({X4.t.nunique()} targets); XM pairs in overlap_random_r4: {len(x4)} ==')
        say(f'r4 null checks: pop maxdiff {X4.null_pop_maxdiff.max()}, dup {X4.null_dup_edges.max()}')
        say(f'r4 share of local interactions on shared products: median {X4.local_share_on_shared.median():.4f} (RQ1 definition); '
            f'{x4.shared_item_interaction_cover.median():.4f} (training, all XM pairs)')
        R5 = measure_rows(X4, POP + PAT + ADJ, '_r4'); R5.to_csv(f'{NOTES}/rq1_T5.csv', index=False, lineterminator='\n')
        P5 = paired_rows(X4, [('cos', 'sp'), ('nbr5', 'top10'), ('ppmi', 'sp'), ('resid', 'sp')], '_r4'); P5.to_csv(f'{NOTES}/rq1_T5_paired.csv', index=False, lineterminator='\n')
        rows = []
        for _, r in R5.iterrows():
            lab = NAME[r.measure] + (', all foreign' if r.arm == 'all' else ', two-sided matched')
            rows.append(f"{lab} & {r.pairs} & {fmt(r.local)} & {fmt(r.foreign)} & {fmt(r.null_foreign)} & "
                        f"{fci((r['ratio_0.10'], r['ratio_0.10_lo'], r['ratio_0.10_hi']))} ({r['ratio_0.10_n']}) & {fci((r.D, r.D_lo, r.D_hi), 2, True)} & "
                        f"{fci((r.excess, r.excess_lo, r.excess_hi), 2, True)} \\\\")
        rows.append('\\midrule')
        for _, r in P5[(P5.scale == 'D (F-L)')].iterrows():
            rows.append(f"Paired $D$: {NAME[r.pattern].split(': ')[1]} vs {NAME[r.popularity].split(': ')[1]}, {ARMNAME[r.arm]} & {r.pairs} & "
                        f"\\multicolumn{{4}}{{c}}{{patterns higher in {100 * r.share_pattern_higher:.0f}\\% [{100 * r.share_lo:.0f}, {100 * r.share_hi:.0f}] of pairs}} & "
                        f"\\multicolumn{{2}}{{c}}{{{fci((r['median'], r.lo, r.hi), 2, True)}}} \\\\")
        tex.append(table('RQ1 with only ratings of 4 or 5 treated as interactions.', 'tab:rq1_r4', 'lccccccc',
                         'Measure & Pairs & Local B vs A & Foreign vs A & Null foreign vs A & Ratio, $L\\ge0.10$ & $D=F-L$ & Excess over chance \\\\', rows,
                         f'Data rebuilt with ratings $\\ge4$ before deduplication and the adaptive core filter ({X4.t.nunique()} target contexts, {len(X4)} pairs with $\\ge$10 shared products). '
                         'Same design as Tables~\\ref{tab:rq1_main}--\\ref{tab:rq1_adjusted}; all thresholds in the CSV files.'))
    else:
        say('\n(r4 results not available yet)')

    # ---------------- Figure 4 data: raw agreement, no clipping
    cols = ['t', 'target_market', 'donor_market', 'category', 'n_shared', 'nB', 'nF', 'V', 'mVB', 'mVF']
    F4 = X[cols].copy()
    for m in POP + PAT + ADJ:
        for p in ['loc_', 'for_', 'mloc_', 'mfor_', 'null_', 'mnull_']:
            F4[p + m] = X[p + m]
        F4['D_' + m] = X['for_' + m] - X['loc_' + m]
        F4['Dmat_' + m] = X['mfor_' + m] - X['mloc_' + m]
        F4['ratio_' + m] = (X['for_' + m] / X['loc_' + m]).where(X['loc_' + m] > 0)
    F4['chance_top10'] = 10 / X.n_shared
    F4.to_csv(f'{NOTES}/rq1_fig4_data.csv', index=False, lineterminator='\n')
    say(f'\nFigure 4 data: {len(F4)} pairs -> paper/results/rq1_fig4_data.csv (raw values, no clipping)')

    open(f'{REPO}/paper/tables_rq1.tex', 'w', encoding='utf-8', newline='\n').write('\n\n'.join(tex) + '\n')
    say('\n== T1 (main) ==')
    show = ['measure', 'arm', 'pairs', 'local', 'foreign', 'null_foreign'] + [f'ratio_{t:.2f}' for t in THR] + [f'ratio_{t:.2f}_n' for t in THR] + \
           ['D', 'D_lo', 'D_hi', 'D_pos_share', 'excess', 'excess_lo', 'excess_hi', f'cc_{CC_THR:.2f}', f'cc_{CC_THR:.2f}_lo', f'cc_{CC_THR:.2f}_hi', f'cc_{CC_THR:.2f}_n']
    say(pd.concat([R1, R4])[show].round(3).to_string(index=False))
    say('\n== T1 ratio CIs ==')
    for _, r in pd.concat([R1, R4]).iterrows():
        say(r.measure, r.arm, ' | '.join(f"{t:.2f}: {fci((r[f'ratio_{t:.2f}'], r[f'ratio_{t:.2f}_lo'], r[f'ratio_{t:.2f}_hi']))} (n={r[f'ratio_{t:.2f}_n']})" for t in THR),
            '| cc:', ' | '.join(f"{t:.2f}: {fci((r[f'cc_{t:.2f}'], r[f'cc_{t:.2f}_lo'], r[f'cc_{t:.2f}_hi']))} (n={r[f'cc_{t:.2f}_n']})" for t in THR))
    say('\n== T2 (paired) ==')
    say(P.round(3).to_string(index=False))
    say('\n== T3 (top-10 by bin) ==')
    say(R3.round(3).to_string(index=False))
    if os.path.exists(f'{PAPER_RAW}/rq1_agree_r4.jsonl'):
        say('\n== T5 (r4) ==')
        say(R5[show].round(3).to_string(index=False))
        say(P5[P5.scale != 'ratio'].round(3).to_string(index=False))
    open(f'{NOTES}/rq1_stats.txt', 'w', encoding='utf-8', newline='\n').write(LOG.getvalue())


if __name__ == '__main__':
    main()
