"""Tables of the Supplementary Material (S1-S5) and the per-context RQ1 result cited in the main text.

Reads only machine-readable outputs of the other scripts (paper/results/*.csv, paper/raw/*.csv, results/*.parquet);
run it after the table scripts (`python run.py tables` runs it last). Writes:
  paper/supp/S1.tex   product launch per context (random split, zero-feedback tuning)       <- results/l3_context.csv
  paper/supp/S2.tex   dataset accounting, 74 retained contexts                              <- raw/aud_S2.csv
  paper/supp/S3.tex   experiment accounting per experiment                                 <- raw/aud_accounting_summary.csv, results/sc_results.csv
  paper/supp/S4.tex   inputs of the country-distance analysis                              <- results/rq4_country_*.csv, raw/aud_rq4_pairs.csv
  paper/supp/S5a.tex  main results on the original data and the four cleaned data versions <- results/sens_*.csv, results/sens2_*.csv, results/rq4_ratiofree.csv, results/contexts*.parquet
  paper/supp/S5b.tex  null-pool ablation and Market-Aware GMF                              <- results/nullab_summary.csv, results/mkt_summary.csv
  paper/supp/S5c.tex  further robustness checks                                            <- results/loo_robustness.csv, rq2_seed_variability.csv,
                                                                                               rq2_catalog_strata.csv, sc_results.csv, rq2_table2_all.csv
  paper/results/rq1_per_context.csv  per target context: median over its foreign donors of the paired difference
                      D_pat - D_pop (cosine vs rank correlation, all foreign data, original data); the number of
                      contexts with a positive median is printed (main text, Section 5.1)        <- results/rq1_fig4_data.csv
Every number in the table cells is formatted from these files; nothing is typed in (the table notes contain fixed
descriptive facts of the study design, each checked against these files and the code). The S-tables are \\input by the Supplementary
Material (Supplementary.tex) of the paper.
"""
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(REPO, 'paper', 'results')
RAW = os.environ.get('PAPER_RAW', os.path.join(REPO, 'paper', 'raw'))
if not os.path.isabs(RAW):
    RAW = os.path.join(REPO, RAW)
CTX = os.path.join(REPO, 'results')
OUT = os.path.join(REPO, 'paper', 'supp')
os.makedirs(OUT, exist_ok=True)


def res(name):
    return pd.read_csv(os.path.join(RES, name), encoding='utf-8')


def write(name, lines):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('wrote', os.path.relpath(os.path.join(OUT, name), REPO).replace(os.sep, '/'))


CAT = {'Arts_Crafts_and_Sewing': 'Arts', 'Automotive': 'Auto', 'Books': 'Books', 'CDs_and_Vinyl': 'CDs',
       'Cell_Phones_and_Accessories': 'Phones', 'Electronics': 'Electr.', 'Grocery_and_Gourmet_Food': 'Grocery',
       'Home_and_Kitchen': 'Home', 'Industrial_and_Scientific': 'Indust.', 'Movies_and_TV': 'Movies',
       'Musical_Instruments': 'Music', 'Office_Products': 'Office', 'Sports_and_Outdoors': 'Sports', 'Toys_and_Games': 'Toys'}
MK = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk']


def ctx(t):
    m, c = t.split(' ', 1)
    return m.upper() + ' ' + CAT[c]


def sg(v, d=3, sign=True, zero_plus=False):
    """Number with d decimals, '+' sign (if sign) and LaTeX minus. A small negative value that rounds to zero keeps its
    minus sign ($-$0.000), except with zero_plus=True (Table S1), where it is written +0.000."""
    if v is None or pd.isna(v):
        return '--'
    s = ('%+.*f' if sign else '%.*f') % (d, v)
    if zero_plus and float(s) == 0:
        s = s.replace('-', '+' if sign else '')
    return s.replace('-', '$-$')


def ci(lo, hi, d=3, sign=True):
    return f'[{sg(lo, d, sign)}, {sg(hi, d, sign)}]'


def i(x):
    return f'{int(x):,}'


def panels(rows, ncol, n_left=None):
    """Two side-by-side panels: the first half of the rows on the left, the rest on the right."""
    h = (len(rows) + 1) // 2 if n_left is None else n_left
    left, right = rows[:h], rows[h:]
    return [left[k] + ' & ' + (right[k] if k < len(right) else ' & '.join([''] * ncol)) + ' \\\\' for k in range(h)]


# ================================================================================ S1
def s1():
    d = res('l3_context.csv')
    r = d[(d.split == 'random') & (d.tuning == 'zf')]
    dF = r.pivot(index='t', columns='model', values='dF')
    dL = r.pivot(index='t', columns='model', values='dL')
    ch = r.groupby('t').chance.first()
    rows = []
    for t in sorted(dF.index, key=lambda x: (x.split(' ', 1)[1], x.split(' ', 1)[0])):
        cells = [ctx(t), '%.3f' % ch[t]]
        for m in ['ItemKNN', 'EASE', 'MostPop']:
            cells += [sg(dF.loc[t, m], zero_plus=True), sg(dL.loc[t, m], zero_plus=True)]
        rows.append(' & '.join(cells))
    hdr = 'Context & Chance & \\multicolumn{2}{c}{ItemKNN} & \\multicolumn{2}{c}{EASE} & \\multicolumn{2}{c}{MostPop}'
    sub = ' & & F & L & F & L & F & L'
    L = ['\\begin{table}[H]',
         '\\caption{Table S1. Product launch per context, random split: launched-item NDCG@10 minus chance, with foreign data (F) and with local data only (L); zero-local-feedback tuning; mean over three launch sets.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}',
         '\\begin{tabular}{lrrrrrrr@{\\hspace{10pt}}lrrrrrrr}', '\\toprule',
         hdr + ' & ' + hdr + ' \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\\cmidrule(lr){15-16}',
         sub + ' &' + sub + ' \\\\', '\\midrule']
    L += panels(rows, 8)
    L += ['\\bottomrule', '\\end{tabular}',
          '\\par\\vspace{2pt}{\\footnotesize\\noindent Original data, %d contexts. Chance: NDCG@10 of the random ranker on the same launch sets. '
          'F: model with all same-category foreign data minus chance; L: local-only model minus chance (local-only scores are near zero by design, so L is close to $-$chance). '
          '-- = not run (EASE skips catalogs over 4,000 products). Not in the table: CA CDs (no foreign donor) and CA Toys (fewer than 10 launched-item test users). '
          'Summary statistics: Table 5 of the main text.\\par}' % len(rows),
          '\\end{table}']
    write('S1.tex', L)


# ================================================================================ S2
def s2():
    a = pd.read_csv(os.path.join(RAW, 'aud_S2.csv'), encoding='utf-8')
    kept, present = a[a.kept], a
    assert len(kept) == 74, len(kept)
    rows = []
    for _, r in kept.sort_values(['category', 'market']).iterrows():
        rows.append(' & '.join([r.market.upper() + ' ' + CAT[r.category], i(r.dedup_rows), i(r.kept_rows), f'{100 * r.ret_dedup:.1f}',
                                i(r.kept_users), i(r.kept_items), str(int(r.core)), i(r.val_users)]))
    hdr = 'Context & Dedup & Kept & Ret.\\,\\% & Users & Prod. & $k$ & Val.'
    L = ['\\begin{table}[H]', '\\caption{Table S2. Dataset accounting for the 74 retained contexts.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}',
         '\\begin{tabular}{lrrrrrcr@{\\hspace{10pt}}lrrrrrcr}', '\\toprule', hdr + ' & ' + hdr + ' \\\\', '\\midrule']
    L += panels(rows, 8)
    tk = (i(kept.dedup_rows.sum()), i(kept.kept_rows.sum()), f'{100 * kept.kept_rows.sum() / kept.dedup_rows.sum():.1f}')
    ta = (i(present.raw_rows.sum()), i(present.dedup_rows.sum()), i(kept.kept_rows.sum()), f'{100 * kept.kept_rows.sum() / present.dedup_rows.sum():.1f}')
    dropped = a[~a.kept]
    L += ['\\bottomrule', '\\end{tabular}',
          ('\\par\\vspace{2pt}{\\footnotesize\\noindent Dedup: interactions after collapsing repeated user--product pairs; Kept: interactions retained by the adaptive core filter; '
           'Ret.: Kept/Dedup; Users, Prod.: users and products kept; $k$: selected core level; Val.: users with a validation interaction (random split; 0 = no validation users, first grid setting used). '
           'All 74 contexts: %s deduplicated and %s kept interactions (%s\\%%). '
           'All %d non-empty files: %s raw rows and %s deduplicated interactions, of which %s are kept (%s\\%% of the deduplicated interactions). '
           'The %d dropped files (below 800 interactions or 40 products even at $k=2$) are CA and JP Arts \\& Crafts; CA, FR, IN and JP Automotive; Digital Music in all seven markets; IN, JP and MX Grocery; JP Home \\& Kitchen; '
           'CA, FR, IN and JP Industrial; IN Sports; JP and UK Toys \\& Games. Fourteen of the 112 file names are absent or empty (CDs \\& Vinyl outside CA, FR Grocery, and Kindle Store in all markets). '
           'Raw rows and raw user and product counts per file are in \\texttt{tables\\_S2.tex} of the code repository.\\par}') % (tk + (len(present),) + ta + (len(dropped),)),
          '\\end{table}']
    write('S2.tex', L)


# ================================================================================ S3
CODE = [('n_nodonor', 'D'), ('n_rq1rule', 'S'), ('n_catalog', 'C'), ('n_launch', 'U'), ('n_launchp', 'P'), ('n_costrule', 'R'),
        ('n_cost', 'B'), ('n_incomplete', 'I'), ('n_merge', 'M'), ('n_subset', 'O'), ('n_error', 'E')]
SPL = {'random': 'R', 'temporal': 'Ch', 'calendar': 'Ca'}


def reps(s):
    s = str(s)
    if s.startswith('pairs'):
        return s.replace('pairs ', '').replace(' XM', ' pairs')
    out = []
    for p in [p.strip() for p in s.split(';')]:
        m = re.match(r'(\d) rep: (\d+)', p)
        out.append('%s$\\times$%s' % (m.group(2), m.group(1)))
    return ', '.join(out)


def s3():
    sm = pd.read_csv(os.path.join(RAW, 'aud_accounting_summary.csv'), encoding='utf-8').set_index('experiment')
    V = [('xd', 'V1'), ('nv', 'V2'), ('nvxd', 'V3'), ('nvc', 'V4')]
    groups = [
        ('RQ1 agreement', [('rq1_agree', 'all ratings', 'R'), ('rq1_agree_r4', 'all, ratings $\\geq$4', 'R')]),
        ('RQ2 complete data', [('rq2_%s_%s' % (m, sp), m, SPL[sp]) for sp in ['random', 'temporal', 'calendar'] for m in ['MostPop', 'ItemKNN', 'EASE', 'UserKNN', 'Random']]
         + [('rq2_%s_random_seed1' % m, m + ', seeds 1, 2', 'R') for m in ['MostPop', 'ItemKNN', 'EASE', 'UserKNN']]
         + [('rq2_%s_random_r4' % m, m + ', ratings $\\geq$4', 'R') for m in ['ItemKNN', 'EASE']]
         + [('mkt_runs', 'MA-GMF, GMF', 'R')]
         + [('nullab_%s_random%s' % (m, t), 'Null pool, %s%s' % (m, n), 'R') for t, n in [('', ''), ('_nvc', ', V4')] for m in ['ItemKNN', 'EASE', 'MostPop']]),
        ('RQ3 scarcity', [('sc_ItemKNN_random|arm=P', 'ItemKNN', 'R'), ('sc_EASE_random|arm=P', 'EASE', 'R'), ('sc_UserKNN_random|arm=P', 'UserKNN', 'R'),
                          ('sc_ItemKNN_temporal|arm=P', 'ItemKNN', 'Ch'), ('sc_ItemKNN_calendar|arm=P', 'ItemKNN', 'Ca'), ('sc_EASE_calendar|arm=P', 'EASE', 'Ca')]),
        ('RQ3 launch', [('l3_%s_%s' % (m, sp), m, SPL[sp]) for sp in ['random', 'temporal', 'calendar'] for m in ['ItemKNN', 'EASE', 'UserKNN', 'MostPop', 'Random']]),
        ('RQ4 single donor', [('aud_single', 'ItemKNN', 'R')]),
        ('Cleaned data versions', [('sens_%s/rq1_agree_%s' % (v, v), 'RQ1 agreement, ' + n, 'R') for v, n in V]
         + [('sens_%s/rq2_ItemKNN_random_%s' % (v, v), 'RQ2 ItemKNN, ' + n, 'R') for v, n in V]
         + [('sens_%s/sc_ItemKNN_random|arm=P' % v, 'Scarcity ItemKNN, ' + n, 'R') for v, n in [('nv', 'V2'), ('nvc', 'V4')]]
         + [('sens_%s/l3_ItemKNN_random' % v, 'Launch ItemKNN, ' + n, 'R') for v, n in [('xd', 'V1'), ('nv', 'V2'), ('nvc', 'V4')]]),
    ]
    sc = res('sc_results.csv')
    rows = []
    for g, items in groups:
        rows.append('\\multicolumn{6}{l}{\\emph{%s}}' % g)
        for exp, lab, sp in items:
            r = sm.loc[exp]
            exc = ' '.join('%s%d' % (c, r[k]) for k, c in CODE if r[k])
            unt, gap = int(r.included_without_val_users), int(r.gap_rule)
            extra = str(unt) + ('; %d' % gap if gap else '')
            inc_s, rep_s = '%d/%d' % (r.included, r.universe), reps(r.reps)
            if exp == 'sc_UserKNN_random|arm=P':
                # the accounting counts f=0.05 (stopped for budget after 50 contexts); use the f=0.10 coverage of sc_results.csv
                q = sc[(sc.model == 'UserKNN') & (sc.split == 'random') & (sc.f == 0.10) & (sc.arm == 'P')].iloc[0]
                inc_s = '%d/%d' % (q.n_all, r.universe)
                rep_s = '%d$\\times$1, %d$\\times$3' % (q.n_all - q.n_3seeds, q.n_3seeds)
                exc = 'D%d B%d' % (r.n_nodonor, r.n_cost)
                extra = '%d; %d' % (unt, q.n_removed)
                lab = 'UserKNN$^a$'
            rows.append(' & '.join([lab, sp, inc_s, rep_s, exc or '--', extra]))
    hdr = 'Experiment & Split & Incl. & Repetitions & Excluded & Unt.'
    h = (len(rows) + 1) // 2
    while rows[h - 1].startswith('\\multicolumn'):     # do not end the left panel with a group header
        h -= 1
    L = ['\\begin{table}[H]', '\\caption{Table S3. Experiment accounting: included contexts, exclusions by reason and completed repetitions, per experiment.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{1.9pt}',
         '\\begin{tabular}{llrlll@{\\hspace{8pt}}llrlll}', '\\toprule', hdr + ' & ' + hdr + ' \\\\', '\\midrule']
    left, right = rows[:h], rows[h:]
    for k in range(max(len(left), len(right))):
        L.append((left[k] if k < len(left) else ' & & & & & ') + ' & ' + (right[k] if k < len(right) else ' & & & & & ') + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}',
          ('\\par\\vspace{2pt}{\\footnotesize\\noindent Split: R random, Ch chronological, Ca calendar. Incl.: contexts with at least one complete repetition (all arms present), out of 74 contexts (calendar split: 69 contexts; ratings $\\geq$4: 69). '
           'Repetitions: included contexts $\\times$ completed repetitions (split seeds, scarcity subsamples or launch sets); for pair experiments, completed target--donor pairs out of all same-category pairs. '
           'Excluded (number of contexts): D no same-category foreign donor; S no donor shares $\\geq$10 products (RQ1 rule); C catalog over 4,000 products (EASE); U fewer than 10 launched-item test users; P too few launched or eligible products; '
           'R compute rule fixed before the run (MA-GMF: first pooled epoch over 15 s; Electronics, Movies \\& TV and Books); B not run within the compute budget (UK Books in most runs; the largest scarcity contexts); I incomplete (not all arms of any repetition); M fails the core thresholds after merging variants. Data versions (Section S5): V1 reviewers dropped, V2 variants merged across markets, V3 both, V4 variants merged within each store (the variant-merged data). '
           'Unt.: included contexts without validation users (first grid setting); after the semicolon, contexts removed from $G$ by the gap rule ($f=0.10$). '
           'Scarcity rows give the patterns-scarce arm (the other arms have the same contexts; the complete-data reference has one repetition). Seed rows: seeds 1 and 2 have identical coverage. $^a$ $f\\geq0.10$; UserKNN at $f=0.05$ was stopped for the compute budget after 50 contexts and is not reported. '
           'Null pool: the null-pool ablation (pooling with curveball-randomised foreign data), original data and V4. '
           'Runs of the v3 study version and the per-context accounting (which context is excluded for which reason) are in \\texttt{tables\\_S3.tex} and \\texttt{paper/results} of the code repository.\\par}'),
          '\\end{table}']
    write('S3.tex', L)


# ================================================================================ S4
def s4():
    F = res('rq4_country_features.csv').set_index('market').loc[MK]
    D = res('rq4_country_distances.csv').set_index(['market_a', 'market_b'])
    P = pd.read_csv(os.path.join(RAW, 'aud_rq4_pairs.csv'), encoding='utf-8')
    ncat = P[P.outcome == 'd_excl'].set_index(['target_market', 'donor_market']).ncat
    dims = ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']
    var = F[dims].astype(float).var(axis=0, ddof=1)

    def mat(col, fmt, row):
        return [f'{a.upper()} & ' + ' & '.join('--' if a == b else fmt(row(a, b)) for b in MK) for a in MK]
    head = ' & ' + ' & '.join(x.upper() for x in MK)
    ks = mat(None, lambda v: f'{v:.2f}', lambda a, b: D.loc[(a, b), 'kogut_singh'])
    sl = mat(None, lambda v: f'{int(v)}', lambda a, b: D.loc[(a, b), 'shared_language'])
    geo = mat(None, lambda v: f'{v:,.0f}', lambda a, b: D.loc[(a, b), 'geo_km'])
    nc = mat(None, lambda v: f'{int(v)}', lambda a, b: ncat.get((a, b), 0))
    L = ['\\begin{table}[H]', '\\caption{Table S4. Inputs of the country-distance analysis.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{3.2pt}', '\\begin{tabular}{lcccccccc}', '\\toprule',
         '\\multicolumn{9}{l}{\\textit{(a) Hofstede scores (2015 release), variances and capitals}} \\\\',
         'Country & ' + ' & '.join(dims) + ' & Capital & Lat., Lon. \\\\', '\\midrule']
    for a, r in F.iterrows():
        L.append(f'{r.country} ({a.upper()}) & ' + ' & '.join(str(int(r[d])) for d in dims) + f' & {r.capital} & {sg(r.lat, 2, False)}, {sg(r.lon, 2, False)} \\\\')
    L.append('Variance $V_d$ (7 countries) & ' + ' & '.join(f'{v:.1f}' for v in var) + ' & & \\\\')
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{lccccccc@{\\hspace{14pt}}lccccccc}', '\\toprule',
          '\\multicolumn{8}{l}{\\textit{(b) Kogut--Singh cultural distance}} & \\multicolumn{8}{l}{\\textit{(c) Shared official language}} \\\\',
          head + ' & ' + head + ' \\\\', '\\midrule']
    L += [x + ' & ' + y + ' \\\\' for x, y in zip(ks, sl)]
    L += ['\\midrule', '\\multicolumn{8}{l}{\\textit{(d) Great-circle distance between capitals (km)}} & \\multicolumn{8}{l}{\\textit{(e) Categories per ordered pair}} \\\\',
          head + ' & ' + head + ' \\\\', '\\midrule']
    L += [x + ' & ' + y + ' \\\\' for x, y in zip(geo, nc)]
    langs = '; '.join(f'{a.upper()} {F.loc[a, "languages"]}' for a in MK)
    L += ['\\bottomrule', '\\end{tabular}',
          '\\par\\vspace{2pt}{\\footnotesize\\noindent PDI power distance, IDV individualism, MAS masculinity, UAI uncertainty avoidance, LTO long-term orientation, IVR indulgence. '
          f'(c) 1 = the two countries share an official language; coded languages: {langs}. '
          '(e) Categories contributing to each ordered pair in the single-donor gain outcome (row: target, column: donor); ordered country pairs are averaged over different, incomplete sets of categories; the permutation test keeps these sets attached to their pair.\\par}',
          '\\end{table}']
    write('S4.tex', L)


# ================================================================================ S5a
VERS = ['orig', '_xd', '_nv', '_nvxd', '_nvc']                 # Original, reviewers dropped, merged across markets, both, merged within store


def s5a():
    def both(name):                                            # sens2_* holds orig/_xd/_nv/_nvc, sens_* holds _nvxd (same labels)
        return res('sens2_' + name), res('sens_' + name)

    def pick(name, v, cond):
        a, b = both(name)
        d = b if v == '_nvxd' else a
        d = cond(d)
        if 'var' in d.columns:
            d = d[d['var'] == v]
        return d.iloc[0] if len(d) else None

    def rq1(name, v, cond):
        a, b = both(name)
        d = b if v == '_nvxd' else a
        lab = f'sens{"_orig" if v == "orig" else v}_all'
        d = cond(d[d.variant == lab])
        return d.iloc[0] if len(d) else None
    nr = lambda: 'not run'
    rows = []

    def row(label, cells, ci_cells=None):
        rows.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
        if ci_cells is not None:
            rows.append('\\quad 95\\% CI & ' + ' & '.join(ci_cells) + ' \\\\[1pt]')
    nctx = [len(pd.read_parquet(os.path.join(CTX, f'contexts{"" if v == "orig" else v}.parquet'))) for v in VERS]
    row('Contexts after core filter', [str(n) for n in nctx])
    jm = [res(f'sens_diag_{v.strip("_")}.csv').jac_mass.median() for v in VERS]
    row('Co-occurrence mass on Jaccard $\\geq$0.8 pairs (median)', [f'{100 * x:.1f}\\%' for x in jm])
    row('RQ1 pairs ($\\geq$10 shared products)', [str(int(rq1('rq1_measures.csv', v, lambda d: d[(d.measure == 'sp') & (d.arm == 'all')]).pairs)) for v in VERS])
    PP = {v: rq1('rq1_paired.csv', v, lambda d: d[(d.pattern == 'cos') & (d.popularity == 'sp') & (d.arm == 'all') & (d.scale == 'D (F-L)')]) for v in VERS}
    row('RQ1: cosine vs rank, patterns lose less', [f'{100 * PP[v].share_pattern_higher:.0f}\\%' for v in VERS])
    row('RQ1: cosine vs rank, paired $D$, median', [sg(PP[v]['median'], 2) for v in VERS], [ci(PP[v].lo, PP[v].hi, 2) for v in VERS])
    C = {v: rq1('rq1_measures.csv', v, lambda d: d[(d.measure == 'cos') & (d.arm == 'all')]) for v in VERS}
    row('RQ1: cosine, local $L$ / foreign $F$ / null $F_0$', [f'{C[v].local:.2f} / {C[v].foreign:.2f} / {C[v].null_foreign:.2f}' for v in VERS])
    row('RQ1: cosine, $F-c$', [sg(C[v].excess, 2) for v in VERS], [ci(C[v].excess_lo, C[v].excess_hi, 2) for v in VERS])

    def q2(model, split, arm, v):
        return pick('rq2.csv', v, lambda d: d[(d.model == model) & (d.split == split) & (d.arm == arm) & (d.set == 'all')])
    IP = {v: q2('ItemKNN', 'random', 'pool', v) for v in VERS}
    IW = {v: q2('ItemKNN', 'random', 'weight', v) for v in VERS}
    EP = {v: q2('EASE', 'random', 'pool', v) for v in VERS}
    MP = {v: q2('MostPop', 'random', 'pool', v) for v in VERS}
    IT = {v: q2('ItemKNN', 'temporal', 'pool', v) for v in VERS}
    row('RQ2: ItemKNN local only (median)', [f'{IP[v].local_median:.3f}' for v in VERS])
    row('RQ2: ItemKNN pool, mean gain', [sg(IP[v].gain_mean, 4) for v in VERS], [ci(IP[v].gain_mean_lo, IP[v].gain_mean_hi, 4) for v in VERS])
    row('RQ2: ItemKNN pool, better/worse $>$0.01', [f'{IP[v].better}/{IP[v].worse}' for v in VERS])
    row('RQ2: ItemKNN weight, mean gain', [sg(IW[v].gain_mean, 4) for v in VERS], [ci(IW[v].gain_mean_lo, IW[v].gain_mean_hi, 4) for v in VERS])
    row('RQ2: ItemKNN weight, better/worse $>$0.01', [f'{IW[v].better}/{IW[v].worse}' for v in VERS])
    row('RQ2: EASE local only (median) / pool mean gain', [f'{EP[v].local_median:.3f} / {sg(EP[v].gain_mean, 4)}' for v in VERS])
    row('RQ2: MostPop pool, mean gain', [sg(MP[v].gain_mean, 4) for v in VERS], [ci(MP[v].gain_mean_lo, MP[v].gain_mean_hi, 4) for v in VERS])
    row('RQ2: ItemKNN chronological pool, mean gain', [sg(IT[v].gain_mean, 4) for v in VERS], [ci(IT[v].gain_mean_lo, IT[v].gain_mean_hi, 4) for v in VERS])
    SC = {v: pick('scarcity.csv', v, lambda d: d[(d.arm == 'P') & (d.set == 'all')]) for v in VERS}
    row('RQ3 scarcity, ItemKNN patterns, gain (seed 0)', [sg(SC[v].gain, 3) if SC[v] is not None else nr() for v in VERS],
        [ci(SC[v].gain_lo, SC[v].gain_hi, 3) if SC[v] is not None else '' for v in VERS])
    row('RQ3 scarcity, ItemKNN patterns, $G$ (seed 0)', [f'{SC[v].G:.2f}' if SC[v] is not None else nr() for v in VERS],
        [ci(SC[v].G_lo, SC[v].G_hi, 2, False) if SC[v] is not None else '' for v in VERS])
    U = res('sens2_userknn_check.csv')

    def nonmono(v):                                             # contexts where 50% of the local data beat 100% (UserKNN, patterns scarce)
        p = 'orig' if v == 'orig' else v
        if f'{p}_full' not in U:
            return nr()
        x = U[[f'{p}_full', f'{p}_P50']].dropna()
        return f'{int((x[f"{p}_P50"] > x[f"{p}_full"]).sum())} of {len(x)}'
    row('RQ3 scarcity, UserKNN non-monotone contexts', [nonmono(v) for v in VERS])
    LA = {v: pick('launch.csv', v, lambda d: d[(d.model == 'ItemKNN') & (d.set == 'all')]) for v in VERS}
    LM = {v: pick('launch.csv', v, lambda d: d[(d.model == 'MostPop') & (d.set == 'all')]) for v in VERS}
    lp = res('sens2_launch_paired.csv')
    LP = {v: (lp[(lp.data == v) & (lp.comparison == 'ItemKNN - MostPop')].iloc[0] if ((lp.data == v) & (lp.comparison == 'ItemKNN - MostPop')).any() else None) for v in VERS}
    row('RQ3 launch, ItemKNN +foreign / chance', [f'{LA[v].ndcg_foreign:.3f} / {LA[v].chance:.3f}' if LA[v] is not None else nr() for v in VERS])
    row('RQ3 launch, ItemKNN $-$chance', [sg(LA[v].dF) if LA[v] is not None else nr() for v in VERS],
        [ci(LA[v].dF_lo, LA[v].dF_hi) if LA[v] is not None else '' for v in VERS])
    row('RQ3 launch, contexts above chance+0.01', [f'{int(LA[v].above_F)}/{int(LA[v].contexts)}' if LA[v] is not None else nr() for v in VERS])
    row('RQ3 launch, MostPop $-$chance', [sg(LM[v].dF) if LM[v] is not None else nr() for v in VERS],
        [ci(LM[v].dF_lo, LM[v].dF_hi) if LM[v] is not None else '' for v in VERS])
    row('RQ3 launch, ItemKNN $-$ MostPop', [sg(LP[v]['median']) if LP[v] is not None else nr() for v in VERS],
        [ci(LP[v].lo, LP[v].hi) if LP[v] is not None else '' for v in VERS])
    # Table 6 tests: ratio-free D = F - L rows (IN Arts & Crafts kept) and the gain row (IN Arts & Crafts excluded), rq4_ratiofree.py
    q4 = res('rq4_ratiofree.csv')
    q4 = q4[(q4.in_ac == 'incl.') & q4.kind.isin(['D', 'gain'])]
    src = {'orig': 'new', '_nv': '_nv', '_nvc': '_nvc'}
    row('RQ4: Table 6 tests below the Bonferroni threshold (of 15)',
        [str(int(q4[q4.source == src[v]].bonferroni.sum())) if v in src and (q4.source == src[v]).any() else nr() for v in VERS])
    assert all(len(q4[q4.source == s]) == 15 for s in src.values())
    # 'common' in sens2_rq2.csv: the contexts present in all four sens2 versions (original, reviewers dropped, merged across markets, merged within store)
    cm = pick('rq2.csv', 'orig', lambda d: d[(d.model == 'ItemKNN') & (d.split == 'random') & (d.arm == 'pool') & (d.set == 'common')])
    L = ['\\begin{table}[H]', '\\caption{Table S5a. Main results on the original data and on the four cleaned data versions.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}', '\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.8cm}ccccc}', '\\toprule',
         'Result & Original & Reviewers dropped & Merged across markets & Both & Merged within store \\\\', '\\midrule']
    L += rows
    L += ['\\bottomrule', '\\end{tabular}',
          '\\par\\vspace{2pt}{\\footnotesize\\noindent All contexts (pairs) of each version; random split unless stated. Intervals: 95\\% two-way cluster bootstrap from one run over all versions with its own seeds, '
          'so original-data intervals can differ slightly from the main tables (point estimates are identical). '
          f'On the {int(cm.contexts)} contexts common to all versions, the original data give an ItemKNN pooling gain of {sg(cm.gain_mean, 4)} {ci(cm.gain_mean_lo, cm.gain_mean_hi, 4)}. '
          'Launch: zero-feedback tuning, all contexts per model. Scarcity: $f=0.10$, subsample 0 only. RQ4 on cleaned data uses the original-data single-donor gains.\\par}',
          '\\end{table}']
    write('S5a.tex', L)


# ================================================================================ S5b
def s5b():
    n = res('nullab_summary.csv')
    rows = []
    for data, lab in [('Original', 'Original'), ('Variants merged', 'Merged')]:
        for m in ['ItemKNN', 'EASE']:
            g = n[(n.data == data) & (n.model == m)].set_index('quantity')
            p, z, r = g.loc['gain_pool'], g.loc['gain_null'], g.loc['relational']
            exin = sg(r.mean_excl_in_ac, 4) if bool(r.in_ac_present) else '--'
            rows.append(f'{lab} & {m} & {int(p.contexts)} & {p.local_median:.3f} & {sg(p["mean"], 4)} & {sg(z["mean"], 4)} {ci(z.mean_lo, z.mean_hi)} & '
                        f'{sg(r["mean"], 4)} {ci(r.mean_lo, r.mean_hi)} & {exin} & {int(r.n_above)}/{int(r.n_below)} & {"yes" if str(r.supported) == "True" else "no"} \\\\')
    mp = n[n.model == 'MostPop'].set_index(['data', 'quantity'])
    same = all(abs(mp.loc[(d, 'relational'), 'mean']) < 1e-12 and mp.loc[(d, 'relational'), 'n_above'] == 0 and mp.loc[(d, 'relational'), 'n_below'] == 0
               for d in n.data.unique())
    assert same, 'MostPop: pool and null pool should coincide'
    k = res('mkt_summary.csv').set_index('name')

    def lvl(x):
        return f'{x.med:.3f} {ci(x.med_lo, x.med_hi, 3, False)}'

    def gn(x, key):
        return f'{sg(x[key], 3, False)} {ci(x[key + "_lo"], x[key + "_hi"], 3, False)}'
    G = [('GMF pooled (no market embedding) $-$ GMF local', 'GMF, pooled (no market embedding)', True),
         ('MA-GMF $-$ GMF local', 'MA-GMF, pooled + market embedding', True),
         ('MA-GMF $-$ GMF pooled', 'MA-GMF minus pooled GMF', False),
         ('GMF pooled $-$ update-matched local GMF', 'GMF, pooled (no market embedding) minus update-matched local GMF', False),
         ('MA-GMF $-$ update-matched local GMF', 'MA-GMF, pooled + market embedding minus update-matched local GMF', False),
         ('MA-GMF $-$ ItemKNN local (median %.3f)' % k.loc['ItemKNN, local'].med, 'MA-GMF, pooled + market embedding minus ItemKNN local', False),
         ('MA-GMF $-$ EASE local (median %.3f)' % k.loc['EASE, local'].med, 'MA-GMF, pooled + market embedding minus EASE local', False)]
    loc = k.loc['GMF, local only']
    nm = int(loc.n)
    mrows = [f'GMF, local only & {nm} & {lvl(loc)} & -- & -- & -- & -- \\\\']
    for lab, key, level in G:
        x = k.loc[key]
        mrows.append(f'{lab} & {int(x.n)} & {lvl(x) if level else "--"} & {gn(x, "gmed")} & {gn(x, "gmean")} & {int(x.better)} & {int(x.worse)} \\\\')
    um = k.loc['GMF, local only, update-matched']
    L = ['\\begin{table}[H]', '\\caption{Table S5b. Null-pool ablation and Market-Aware GMF (random split, NDCG@10).}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{llcccccccc}', '\\toprule',
         '\\multicolumn{10}{l}{\\textit{(a) Null-pool ablation: real foreign data vs foreign data that keep only popularity and user activity}} \\\\',
         'Data & Model & Contexts & \\makecell{Local\\\\only} & \\makecell{Pool $-$ local,\\\\mean} & \\makecell{Null pool $-$ local,\\\\mean [95\\% CI]} & '
         '\\makecell{Pool $-$ null pool,\\\\mean [95\\% CI]} & \\makecell{Excl.\\\\IN A\\&C} & \\makecell{$>$+0.01/\\\\$<-$0.01} & Supported \\\\', '\\midrule']
    L += rows
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{lrcccrr}', '\\toprule',
          f'\\multicolumn{{7}}{{l}}{{\\textit{{(b) Market-Aware GMF and GMF controls (original data, {nm} contexts; paired differences per context)}}}} \\\\',
          'Comparison & $n$ & \\makecell{Median NDCG@10\\\\of first model} & Median gain [95\\% CI] & Mean gain [95\\% CI] & Better & Worse \\\\', '\\midrule']
    L += mrows
    L += ['\\bottomrule', '\\end{tabular}',
          '\\par\\vspace{2pt}{\\footnotesize\\noindent Brackets: 95\\% two-way cluster-bootstrap intervals. Better/worse and $>$+0.01/$<-$0.01: contexts with a difference above $+0.01$ / below $-0.01$. '
          'Excl.\\ IN A\\&C: mean of pool $-$ null pool without IN Arts \\& Crafts (not a context of the merged data). For MostPop, pool and null pool coincide in every context, as they must (sanity check). '
          f'In (b), the update-matched local GMF has median {lvl(um)}.\\par}}',
          '\\end{table}']
    write('S5b.tex', L)


# ================================================================================ S5c
def s5c():
    lo = res('loo_robustness.csv').set_index('statistic')
    A = [('RQ1: median paired difference $D_{\\mathrm{cosine}}-D_{\\mathrm{rank}}$, all pairs', 'RQ1: median paired difference D(cosine) - D(rank), all pairs')]
    for m in ['ItemKNN', 'EASE', 'MostPop']:
        for a in ['pool', 'weight']:
            A.append((f'RQ2: mean gain, {a}, {m}', f'RQ2 random: mean gain, {a}, {m}'))
    A[1:] = sorted(A[1:], key=lambda x: (['ItemKNN', 'EASE', 'MostPop'].index(x[0].split(', ')[-1]), x[0]))
    A += [('RQ2, variant-merged data: mean gain, pool, ItemKNN', 'RQ2 random, variant-merged (_nvc): mean gain, pool, ItemKNN'),
          ('RQ3 scarcity (patterns scarce, $f=0.10$, ItemKNN): median gain', 'Scarcity (patterns scarce, 10%, ItemKNN random): median gain'),
          ('RQ3 launch (ItemKNN, zero feedback): median foreign $-$ chance', 'Launch (random, ItemKNN, zero feedback): median foreign - chance')]
    ra = []
    for lab, key in A:
        r = lo.loc[key]
        ra.append(f'{lab} & {int(r.n)} & {sg(r.full)} & {ci(r.loco_min, r.loco_max)} & {ci(r.loko_min, r.loko_max)} \\\\')
    sv = res('rq2_seed_variability.csv').set_index('model')
    rb = []
    for m in ['MostPop', 'ItemKNN', 'EASE', 'UserKNN']:
        r = sv.loc[m]
        rb.append(f'{m} & {int(r.contexts)} & {r.sd_ctx_median:.4f} & ' + ', '.join(sg(r[f"seed{s}_mean"], 4) for s in range(3)) + ' & '
                  + ', '.join(sg(r[f"seed{s}_median"], 4) for s in range(3)) + f' & {r.sd_mean:.4f} & {sg(r.avg_mean, 4)} {ci(r.avg_mean_lo, r.avg_mean_hi)} \\\\')
    st = res('rq2_catalog_strata.csv').set_index(['model', 'catalog'])
    rc = []
    for c, lab in [('<200', '$<$200'), ('200-1,000', '200--1,000'), ('>1,000', '$>$1,000')]:
        a, e, p = st.loc[('ItemKNN', c)], st.loc[('EASE', c)], st.loc[('MostPop', c)]
        rc.append(f'{lab} & {int(a.contexts)} & {a.median_local:.3f} & {sg(a.mean_gain_pool)} & {int(a.better)}/{int(a.worse)} & {sg(e.mean_gain_pool)} & '
                  f'{int(e.better)}/{int(e.worse)} & {p.median_local:.3f} & {sg(p.mean_gain_pool)} & {int(p.better)}/{int(p.worse)} \\\\')
    sc = res('sc_results.csv')
    sc = sc[sc.arm == 'P']
    rd = []
    for sp, m in [('random', 'ItemKNN'), ('random', 'EASE'), ('random', 'UserKNN'), ('temporal', 'ItemKNN'), ('calendar', 'ItemKNN'), ('calendar', 'EASE')]:
        cells = []
        for f in (0.05, 0.10, 0.25, 0.50):
            q = sc[(sc.split == sp) & (sc.model == m) & np.isclose(sc.f, f)]
            cells.append('not run' if not len(q) or bool(q.iloc[0].partial) else f'{sg(q.iloc[0].gain_med)} ({q.iloc[0].G_med:.2f})')
        rd.append(f'{ {"random": "Random", "temporal": "Chronological", "calendar": "Calendar"}[sp]} & {m} & ' + ' & '.join(cells) + ' \\\\')
    nctx = {k: int(sc[(sc.split == s) & (sc.model == m) & np.isclose(sc.f, 0.10)].iloc[0].n_all) for k, (s, m) in
            {'ri': ('random', 'ItemKNN'), 'ru': ('random', 'UserKNN'), 're': ('random', 'EASE'), 'ti': ('temporal', 'ItemKNN'),
             'ci': ('calendar', 'ItemKNN'), 'ce': ('calendar', 'EASE')}.items()}
    assert nctx['ri'] == nctx['ru'] and nctx['ci'] == nctx['ce']
    t2 = res('rq2_table2_all.csv')
    t2 = t2[(t2.split == 'calendar') & t2.tag.isna()]
    re_ = []
    for m, a in [('MostPop', 'pool'), ('MostPop', 'weight'), ('MostPop', 'greedy'), ('UserKNN', 'pool')]:
        r = t2[(t2.model == m) & (t2.arm == a)].iloc[0]
        re_.append(f'{m} & {a} & {int(r.contexts)} & {r.local_median:.3f} & {sg(r.gain_median, 4)} & {sg(r.gain_mean, 4)} {ci(r.gain_mean_lo, r.gain_mean_hi)} & '
                   f'{int(r["better_0.01"])} & {int(r["worse_0.01"])} \\\\')
    r = t2[(t2.model == 'Random')].iloc[0]
    re_.append(f'Random & local only & {int(r.contexts)} & {r.local_median:.3f} & -- & -- & -- & -- \\\\')
    L = ['\\begin{table}[H]', '\\caption{Table S5c. Further robustness checks cited in the main text.}',
         '\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{>{\\raggedright\\arraybackslash}p{9.6cm}rrcc}', '\\toprule',
         '\\multicolumn{5}{l}{\\textit{(a) Leave-one-country-out (LOCO) and leave-one-category-out (LOKO) ranges of the headline statistics}} \\\\',
         'Statistic & $n$ & All & LOCO range & LOKO range \\\\', '\\midrule'] + ra
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{lcccccc}', '\\toprule',
          '\\multicolumn{7}{l}{\\textit{(b) Split-seed variability of the pooling gain (random split, seeds 0, 1, 2)}} \\\\',
          'Model & Contexts & \\makecell{SD of context\\\\gain (median)} & Mean gain per seed & Median gain per seed & \\makecell{SD of\\\\mean} & \\makecell{Seed-averaged\\\\mean [95\\% CI]} \\\\',
          '\\midrule'] + rb
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{lrcccccccc}', '\\toprule',
          '\\multicolumn{10}{l}{\\textit{(c) Pooling gain by catalog size (products in the context catalog; mean gain; better/worse: above $+0.01$ / below $-0.01$)}} \\\\',
          ' & & \\multicolumn{3}{c}{ItemKNN} & \\multicolumn{2}{c}{EASE} & \\multicolumn{3}{c}{MostPop} \\\\', '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}',
          'Catalog & Contexts & Local (median) & Pool gain & Better/worse & Pool gain & Better/worse & Local (median) & Pool gain & Better/worse \\\\', '\\midrule'] + rc
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{llcccc}', '\\toprule',
          '\\multicolumn{6}{l}{\\textit{(d) Scarcity, patterns scarce: median gain from foreign data ($G$)}} \\\\',
          'Split & Model & $f=0.05$ & $f=0.10$ & $f=0.25$ & $f=0.50$ \\\\', '\\midrule'] + rd
    L += ['\\bottomrule', '\\end{tabular}\\par\\vspace{4pt}', '\\begin{tabular}{llcccccc}', '\\toprule',
          '\\multicolumn{8}{l}{\\textit{(e) Calendar split, complete local data: rows not shown in Table 2 of the main text}} \\\\',
          'Model & Arm & Contexts & Local only & Gain (median) & Gain (mean, 95\\% CI) & Better $>$0.01 & Worse $>$0.01 \\\\', '\\midrule'] + re_
    L += ['\\bottomrule', '\\end{tabular}',
          '\\par\\vspace{2pt}{\\footnotesize\\noindent (a) Minimum and maximum after removing one country (as target and, for pairs, as donor) or one category; $n$: observations with all data; original data unless stated. '
          '(b) Contexts present in all three seed runs; UK Books is not in the seed runs, so seed-0 means differ slightly from Table 2 of the main text. '
          '(d) Patterns-scarce arm at every retention level $f$; in parentheses, $G$, the median share of the gap closed. '
          f'Contexts: random ItemKNN and UserKNN {nctx["ri"]}, EASE {nctx["re"]}, chronological {nctx["ti"]}, calendar {nctx["ci"]}; three subsamples for contexts up to 2.3 million user--product cells, one for larger ones; $G$ over contexts with a gap above 0.01. '
          'Both-scarce and histories-scarce arms at every level are in the code repository. (e) Random ranker: local only by definition.\\par}',
          '\\end{table}']
    write('S5c.tex', L)


# ================================================================================ RQ1 per context (main text, Section 5.1)
def rq1_per_context():
    d = res('rq1_fig4_data.csv')
    d['paired'] = d['D_cos'] - d['D_sp']
    g = d.groupby(['t', 'target_market', 'category']).agg(donors=('paired', 'size'), median_paired_cos_vs_rank=('paired', 'median'),
                                                         share_pairs_positive=('paired', lambda s: float((s > 0).mean()))).reset_index()
    g['positive'] = g.median_paired_cos_vs_rank > 0
    g.to_csv(os.path.join(RES, 'rq1_per_context.csv'), index=False, encoding='utf-8', lineterminator='\n')
    print(f'RQ1 per target context (paired D_pat - D_pop, cosine vs rank correlation, all foreign data, original data): '
          f'median positive in {int(g.positive.sum())} of {len(g)} target contexts; non-positive: '
          + ', '.join(f'{r.t} ({r.median_paired_cos_vs_rank:+.3f})' for r in g[~g.positive].itertuples()))
    print('wrote paper/results/rq1_per_context.csv')


if __name__ == '__main__':
    rq1_per_context()
    s1(); s2(); s3(); s4(); s5a(); s5b(); s5c()
