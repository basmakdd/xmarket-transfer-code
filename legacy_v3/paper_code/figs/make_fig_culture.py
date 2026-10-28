"""RQ4 figure: do country-level distances predict transfer? Data: tab:culture (r, exact permutation p)
and figures/data/culture_pairs.csv (the 42 country-pair values, rebuilt by culture_data.py, which checks all 15 r)."""
import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import texparse as T
from style import *
setup()
OUT = os.path.join(T.ROOT, 'figures')
rows = T.main('tab:culture')
dists = [('Hofstede distance', VIOLET, 's'), ('Shared language', AQUA, '^'), ('Geographic distance', BLUE, 'o')]
short = {'Popularity agreement (rank correlation)': 'Popularity:\nbest-seller ranking', 'Popularity agreement (top-10 overlap)': 'Popularity:\ntop-10 best-sellers',
         'Pattern agreement (cosine)': 'Patterns:\nco-purchase similarity', 'Pattern agreement (neighbour overlap)': 'Patterns:\nnearest neighbours',
         'Gain from one foreign donor (ItemKNN)': 'NDCG@10 gain from\none foreign country'}
ntests = len(rows) * 3
fig = plt.figure(figsize=(FULL_W, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.3, left=0.2, right=0.98, top=0.94, bottom=0.08)
a = fig.add_subplot(gs[0, :])
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    for k, (dname, c, mk) in enumerate(dists):
        rv, p = [float(v) for v in re.findall(r'[-+]?\d*\.\d+', str(r[2 + k]))]
        yy = y + (1 - k) * 0.22
        sig = p < 0.05
        a.plot([0, rv], [yy, yy], color=c, lw=1.2, alpha=0.5)
        a.plot(rv, yy, mk, ms=8, **(dict(mfc=c, mec='white', mew=0.8) if sig else dict(mfc='white', mec=c, mew=1.6)))
        if sig: a.text(rv - 0.04, yy, f'$p$ = {p:.2f}', ha='right', va='center', fontsize=8.5, color=INK2)
a.axvline(0, color=INK2, lw=1)
a.set_yticks(range(len(rows))[::-1]); a.set_yticklabels([short[r[0]] for r in rows], fontsize=9)
a.set_xlim(-0.8, 0.8); a.set_ylim(-0.5, len(rows) - 0.5); a.grid(axis='y', visible=False)
a.set_xlabel('Correlation $r$ across the 42 ordered country pairs')
a.legend(handles=[Line2D([], [], marker=mk, ls='', ms=7.5, mfc=c, mec='white', label=d) for d, c, mk in dists] +
                  [Line2D([], [], marker='o', ls='', ms=7.5, mfc='white', mec=INK2, mew=1.5, label='$p$ ≥ 0.05 (hollow)')],
         loc='upper center', bbox_to_anchor=(0.45, 1.13), ncol=4, fontsize=8.5, handletextpad=0.2, columnspacing=1.0)
a.text(0.99, 0.97, f'filled = nominal $p$ < 0.05\nnone survives Bonferroni ($\\alpha$ = 0.05/{ntests})', transform=a.transAxes,
       ha='right', va='top', fontsize=8, color=INK2)
a.set_title('(a) Correlation of country distance with agreement and gain', loc='left', pad=26)

D = pd.read_csv(os.path.join(OUT, 'data', 'culture_pairs.csv'))
rng = np.random.default_rng(3)
b = fig.add_subplot(gs[1, 0])
for v, lab, c in [(0, 'no', LOCAL), (1, 'yes', AQUA)]:
    s = D[D.shared_language == v].pop10_rel.dropna()
    b.scatter(v + rng.uniform(-0.12, 0.12, len(s)), s, s=22, color=c if v else INK2, alpha=0.75 if v else 0.55, lw=0)
    b.plot([v - 0.25, v + 0.25], [s.median()] * 2, color=INK, lw=2)
    b.text(v + 0.3, s.median(), f'median\n{s.median():+.2f}', va='center', fontsize=8, color=INK2)
b.axhline(0, color=GRID, lw=1.2, zorder=0)
b.set_xticks([0, 1]); b.set_xticklabels([f'no\n({(D.shared_language == 0).sum()} pairs)', f'yes\n({(D.shared_language == 1).sum()} pairs)'])
b.set_xlim(-0.5, 1.75); b.set_xlabel('Countries share an official language'); b.grid(axis='x', visible=False)
b.set_ylabel('Top-10 best-seller overlap\n(ratio, centred within context)', fontsize=9)
rt = [re.findall(r'[-+]?\d*\.\d+', str(r[3])) for r in rows if r[0] == 'Popularity agreement (top-10 overlap)'][0]
b.set_title(f'(b) Language vs best-sellers ($r$ = {float(rt[0]):+.2f}, $p$ = {float(rt[1]):.2f})', loc='left', pad=6, fontsize=9.5)
c = fig.add_subplot(gs[1, 1])
c.scatter(D.geo_km / 1000, D.pat_rel, s=22, color=BLUE, alpha=0.7, lw=0)
m, q = np.polyfit(D.geo_km / 1000, D.pat_rel, 1); xs = np.linspace(0, D.geo_km.max() / 1000, 50)
c.plot(xs, m * xs + q, color=INK2, lw=1.4, ls='--')
c.axhline(0, color=GRID, lw=1.2, zorder=0)
c.set_xlabel('Distance between capitals (1,000 km)'); c.set_ylabel('Co-purchase similarity\n(ratio, centred within context)', fontsize=9)
rg = [re.findall(r'[-+]?\d*\.\d+', str(r[4])) for r in rows if r[0] == 'Pattern agreement (cosine)'][0]
c.set_title(f'(c) Distance vs patterns ($r$ = {float(rg[0]):+.2f}, $p$ = {float(rg[1]):.2f})', loc='left', pad=6, fontsize=9.5)
save(fig, 'fig6_culture', OUT)
print('done')
