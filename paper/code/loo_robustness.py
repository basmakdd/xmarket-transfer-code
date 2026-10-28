"""Leave-one-country-out / leave-one-category-out robustness of the headline statistics,
and RQ2 gains by catalog size. Reads result CSVs in paper/results only; writes
paper/results/loo_robustness.csv, paper/results/rq2_catalog_strata.csv and paper/tables_loo.tex."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
OUT_TEX = os.path.join(HERE, '..', 'tables_loo.tex')


def rd(name):
    return pd.read_csv(os.path.join(RES, name), encoding='utf-8')


stats = []  # (label, frame, value column, statistic, has donor column)
rq1 = rd('rq1_fig4_data.csv')
rq1['paired'] = rq1['D_cos'] - rq1['D_sp']
stats.append(('RQ1: median paired difference D(cosine) - D(rank), all pairs', rq1, 'paired', 'median', True))

r2 = rd('rq2_per_context.csv')
main = r2[r2['tag'].isna() & (r2['split'] == 'random')]
for model in ('ItemKNN', 'EASE', 'MostPop'):
    m = main[main['model'] == model]
    stats.append((f'RQ2 random: mean gain, pool, {model}', m, 'g_pool', 'mean', False))
    stats.append((f'RQ2 random: mean gain, weight, {model}', m, 'g_weight', 'mean', False))

nvc = rd('sens2_rq2_itemknn_random_gain_per_context.csv')
nvc = nvc.dropna(subset=['_nvc']).copy()
nvc['market'] = nvc['t'].str.split(' ').str[0]
nvc['category'] = nvc['t'].str.split(' ').str[1]
stats.append(('RQ2 random, variant-merged (_nvc): mean gain, pool, ItemKNN', nvc, '_nvc', 'mean', False))

sc = rd('sc_matrix_itemknn_random_f010.csv')
sc['market'] = sc['t'].str.split(' ').str[0]
sc['category'] = sc['t'].str.split(' ').str[1]
stats.append(('Scarcity (patterns scarce, 10%, ItemKNN random): median gain', sc, 'gain', 'median', False))

l3 = rd('l3_context.csv')
l3 = l3[(l3['split'] == 'random') & (l3['model'] == 'ItemKNN') & (l3['tuning'] == 'zf')]
stats.append(('Launch (random, ItemKNN, zero feedback): median foreign - chance', l3, 'dF', 'median', False))


def value(df, col, stat):
    v = df[col].dropna().to_numpy(float)
    return (np.median(v) if stat == 'median' else np.mean(v)), len(v)


rows = []
for label, df, col, stat, pair in stats:
    full, n = value(df, col, stat)
    country_col = 'target_market' if pair else 'market'
    loc_c, loc_k = [], []
    for c in sorted(set(df[country_col])):
        keep = df[country_col] != c
        if pair:
            keep &= df['donor_market'] != c
        loc_c.append((value(df[keep], col, stat)[0], c))
    for k in sorted(set(df['category'])):
        loc_k.append((value(df[df['category'] != k], col, stat)[0], k))
    vc = [x for x, _ in loc_c]
    vk = [x for x, _ in loc_k]
    rows.append({
        'statistic': label, 'n': n, 'full': full,
        'loco_min': min(vc), 'loco_min_drop': min(loc_c)[1], 'loco_max': max(vc), 'loco_max_drop': max(loc_c)[1],
        'loko_min': min(vk), 'loko_min_drop': min(loc_k)[1], 'loko_max': max(vk), 'loko_max_drop': max(loc_k)[1],
        'sign_changes': int(sum(np.sign(x) != np.sign(full) for x in vc + vk)),
    })
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'loo_robustness.csv'), index=False, encoding='utf-8', lineterminator='\n')

# Catalog-size strata (RQ2, random split, paper data)
bins = [0, 200, 1000, np.inf]
labels = ['<200', '200-1,000', '>1,000']
srows = []
for model in ('ItemKNN', 'EASE', 'MostPop'):
    m = main[main['model'] == model].copy()
    m['stratum'] = pd.cut(m['n_items'], bins, right=False, labels=labels)
    for s in labels:
        g = m[m['stratum'] == s]
        srows.append({'model': model, 'catalog': s, 'contexts': len(g),
                      'median_local': g['local'].median(), 'mean_gain_pool': g['g_pool'].mean(),
                      'median_gain_pool': g['g_pool'].median(), 'mean_gain_weight': g['g_weight'].mean(),
                      'better': int((g['g_pool'] > 0.01).sum()), 'worse': int((g['g_pool'] < -0.01).sum())})
strata = pd.DataFrame(srows)
strata.to_csv(os.path.join(RES, 'rq2_catalog_strata.csv'), index=False, encoding='utf-8', lineterminator='\n')


def f(x):
    return f'{x:+.3f}'


def tex(s):
    """Escape the LaTeX special characters that occur in the labels (_ % & < >)."""
    s = str(s).replace('\\', r'\textbackslash{}')
    for a, b in (('_', r'\_'), ('%', r'\%'), ('&', r'\&'), ('<', r'$<$'), ('>', r'$>$')):
        s = s.replace(a, b)
    return s


lines = [r'\begin{table}[H]', r'\footnotesize',
         r'\caption{Leave-one-country-out (LOCO) and leave-one-category-out (LOKO) ranges of the headline statistics.\label{tab:loo}}',
         r'\begin{adjustwidth}{-\extralength}{0cm}', r'\centering',
         r'\begin{tabular}{>{\raggedright\arraybackslash}p{6.2cm}rrcc}', r'\toprule',
         r'Statistic & $n$ & All & LOCO range & LOKO range \\', r'\midrule']
for r in rows:
    lines.append(f"{tex(r['statistic'])} & {r['n']} & {f(r['full'])} & "
                 f"[{f(r['loco_min'])}, {f(r['loco_max'])}] & [{f(r['loko_min'])}, {f(r['loko_max'])}] \\\\")
lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustwidth}',
          r'\par\footnotesize Each range is the minimum and maximum of the statistic after removing one country (as target and, for pairs, as donor) or one category. $n$ = observations with all data. Source: paper/results/loo\_robustness.csv.',
          r'\end{table}', '',
          r'\begin{table}[H]', r'\footnotesize',
          r'\caption{Gain from pooling foreign data by catalog size (random split, original data).\label{tab:strata}}',
          r'\centering', r'\begin{tabular}{llrrrrrr}', r'\toprule',
          r'Model & Catalog & Contexts & Local (median) & Pool gain (mean) & Pool gain (median) & Weight gain (mean) & Better/worse \\', r'\midrule']
for r in srows:
    lines.append(f"{r['model']} & {tex(r['catalog'])} & {r['contexts']} & {r['median_local']:.3f} & {f(r['mean_gain_pool'])} & "
                 f"{f(r['median_gain_pool'])} & {f(r['mean_gain_weight'])} & {r['better']}/{r['worse']} \\\\")
lines += [r'\bottomrule', r'\end{tabular}',
          r'\par\footnotesize Catalog = products in the context catalog. Better/worse: pool gain above $+0.01$ / below $-0.01$. Source: paper/results/rq2\_catalog\_strata.csv.',
          r'\end{table}']
open(OUT_TEX, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')
print(out.to_string())
print(strata.to_string())
