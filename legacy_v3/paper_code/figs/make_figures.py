"""Five main figures, drawn only from tables.tex / tables_detailed.tex (and raw/hist.pkl for Fig. 5 spread).
Run: python make_figures.py   -> ../../figures/fig{1..5}_*.pdf/png"""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import texparse as T
from style import *
setup()
OUT = os.path.join(T.ROOT, 'figures')
rng = np.random.default_rng(7)
def hollow(c): return dict(mfc='white', mec=c, mew=1.6)
def filled(c): return dict(mfc=c, mec='white', mew=0.8)

# =====================================================================  Fig 1: what travels
S = T.main('tab:travel')
d = np.array([r[3:] for r in T.detail('tab:travel_detail')], float)
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.32))
labels = ['Best-seller ranking\n(rank correlation)', 'Top-10 best-sellers\n(overlap)',
          'Co-purchase similarity\n(cosine)', 'Nearest co-purchase\nneighbours (overlap)']
cols = [POP, POP, PAT, PAT]
for i, r in enumerate(S[:4]):
    y = 3 - i
    for off, cell, sty in [(0.14, r[5], filled), (-0.14, r[6], hollow)]:
        m, lo, hi = T.ci(cell)
        a.plot([lo, hi], [y+off]*2, color=cols[i], lw=2, solid_capstyle='round')
        a.plot(m, y+off, 'o', ms=8, **sty(cols[i]))
        a.text(lo - 0.025, y+off, f'{m:.2f}', va='center', ha='right', fontsize=9, color=INK2)
a.axvline(1, color=INK2, lw=1, ls='--'); a.text(0.985, 3.62, 'as good as\nlocal shoppers', ha='right', va='top', fontsize=8.5, color=INK2)
a.set_yticks(range(4)[::-1]); a.set_yticklabels(labels)
for t, c in zip(a.get_yticklabels(), cols): t.set_color(c)
a.axhline(1.5, color=GRID, lw=1)
a.text(0.02, 3.62, 'POPULARITY', color=POP, fontsize=8.5, fontweight='bold', va='top')
a.text(0.02, 1.38, 'SHOPPING PATTERNS', color=PAT, fontsize=8.5, fontweight='bold', va='top')
a.set_xlim(0, 1.18); a.set_ylim(-0.5, 3.7); a.set_xlabel('Foreign agreement / local agreement\n(median over country pairs, 95% CI)')
a.grid(axis='y', visible=False)
a.legend(handles=[Line2D([], [], marker='o', ls='', color=INK2, ms=7, mec='white', label='all foreign data'),
                  Line2D([], [], marker='o', ls='', ms=7, mfc='white', mec=INK2, mew=1.5, label='volume-matched')],
         loc='upper right', bbox_to_anchor=(0.985, 0.8), handletextpad=0.2, borderaxespad=0.1, fontsize=8.5, frameon=True, framealpha=1, edgecolor='white')
panel(a, 'a', 'Share of local agreement kept abroad')
# (b) paired scatter
m = ~np.isnan(d[:, 1]) & ~np.isnan(d[:, 3]); x, y = d[m, 1], d[m, 3]
XL, YL = (-1.0, 2.0), (0.0, 2.0)
xc, yc = np.clip(x, *XL), np.clip(y, *YL); out = int(((x != xc) | (y != yc)).sum())
above = y > x
b.scatter(xc[above], yc[above], s=16, color=PAT, alpha=0.55, lw=0, label=f'patterns travel better ({above.mean()*100:.0f}%)')
b.scatter(xc[~above], yc[~above], s=18, color=POP, alpha=0.9, lw=0, label=f'popularity travels better ({(~above).mean()*100:.0f}%)')
b.plot([0, 2], [0, 2], color=INK2, lw=1, ls='--', label='equal ratio')
b.axvline(1, color=GRID, lw=1.2); b.axhline(1, color=GRID, lw=1.2)
b.set_xlim(XL[0]-0.05, XL[1]+0.05); b.set_ylim(YL[0]-0.05, YL[1]+0.05)
b.set_xlabel('Popularity ratio (rank correlation)'); b.set_ylabel('Pattern ratio (cosine)')
b.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.1, markerscale=1.3, fontsize=8.5)
panel(b, 'b', f'Each country pair (n = {m.sum()})')
b.text(1.0, -0.3, f'{out} pairs outside the range are drawn at the axis edge', transform=b.transAxes, fontsize=8, color=INK2, va='top', ha='right')
save(fig, 'fig1_what_travels', OUT)

# =====================================================================  Fig 2: complete local data
F = T.main('tab:full'); models = ['MostPop', 'ItemKNN', 'EASE', 'UserKNN']
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.45))
for i, mo in enumerate(models):
    yy = len(models) - 1 - i
    for off, split, sty in [(0.15, 'Random', filled), (-0.15, 'Temporal', hollow)]:
        r = [q for q in F if q[0] == split and q[1] == mo][0]; mu, lo, hi = T.ci(r[5])
        c = MODEL_C[mo]
        a.plot([lo, hi], [yy+off]*2, color=c, lw=2, solid_capstyle='round')
        a.plot(mu, yy+off, MODEL_M[mo], ms=7.5, **sty(c))
        a.plot(r[4], yy+off, '|', ms=11, mew=2, color=INK)
a.axvline(0, color=INK2, lw=1)
a.set_yticks(range(4)[::-1]); a.set_yticklabels(models); a.grid(axis='y', visible=False)
a.set_xlim(-0.17, 0.10); a.set_xlabel('Change in NDCG@10 from adding foreign data')
a.legend(handles=[Line2D([], [], marker='o', ls='', color=INK2, mec='white', ms=7, label='mean, random split'),
                  Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, mew=1.5, ms=7, label='mean, temporal split'),
                  Line2D([], [], marker='|', ls='', color=INK, ms=10, mew=2, label='median')],
         loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=3, handletextpad=0.1, columnspacing=0.7, fontsize=8.5)
panel(a, 'a', 'Gain is near zero (95% CI)')
# (b) contexts better / unchanged / worse
yl = []; k = 0
for split in ['Random', 'Temporal']:
    for mo in models:
        r = [q for q in F if q[0] == split and q[1] == mo][0]
        n, bt, wr = int(r[2]), int(r[6]), int(r[7]); ne = n - bt - wr
        yv = 8.6 - k - (0.6 if split == 'Temporal' else 0)
        for left, w, c, tc in [(0, wr, DARK, 'white'), (wr, ne, GRID, INK2), (wr+ne, bt, BLUE, 'white')]:
            b.barh(yv, w, left=left, height=0.78, color=c, edgecolor='white', lw=1.2)
            if w >= 6: b.text(left + w/2, yv, str(w), ha='center', va='center', fontsize=8.5, color=tc)
        yl.append((yv, mo)); k += 1
b.set_yticks([p for p, _ in yl]); b.set_yticklabels([q for _, q in yl]); b.grid(False)
b.text(0, 8.6+0.5, 'Random split', fontsize=8.5, fontweight='bold', color=INK2, ha='left')
b.text(0, 4.0+0.5, 'Temporal split', fontsize=8.5, fontweight='bold', color=INK2, ha='left')
b.set_xlim(0, 74); b.set_ylim(0.5, 9.45); b.set_xlabel('Number of market–category contexts')
b.legend(handles=[Patch(color=DARK, label='worse by >0.01'), Patch(color=GRID, label='within ±0.01'), Patch(color=BLUE, label='better by >0.01')],
         loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=3, handlelength=1, columnspacing=0.8, fontsize=8.5)
panel(b, 'b', 'Contexts helped vs hurt')
save(fig, 'fig2_complete_data', OUT)

# =====================================================================  Fig 3: scarcity
SC = T.main('tab:scarcity')
arms = ['Shopping patterns scarce', 'Both scarce (new market)', 'Shopper histories scarce']
armlab = ['Shopping patterns\nscarce', 'Both scarce\n(new market)', 'Shopper histories\nscarce (control)']
series = [('Random', 'ItemKNN', filled), ('Random', 'EASE', filled), ('Random', 'UserKNN', filled), ('Temporal', 'ItemKNN', hollow)]
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 3.2), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.42))
offs = [0.27, 0.09, -0.09, -0.27]
for i, arm in enumerate(arms):
    yy = 2 - i
    for (split, mo, sty), off in zip(series, offs):
        r = [q for q in SC if q[0] == split and q[1] == mo and q[2] == arm][0]; mu, lo, hi = T.ci(r[7])
        a.plot([lo, hi], [yy+off]*2, color=MODEL_C[mo], lw=2, solid_capstyle='round')
        a.plot(mu, yy+off, MODEL_M[mo], ms=7, **sty(MODEL_C[mo]))
a.axvline(0, color=INK2, lw=1); a.axvline(1, color=INK2, lw=1, ls='--')
a.text(0.98, 2.62, 'full local\nlevel', ha='right', va='top', fontsize=8.5, color=INK2)
a.set_yticks([2, 1, 0]); a.set_yticklabels(armlab); a.grid(axis='y', visible=False)
a.set_xlim(-0.05, 1.05); a.set_ylim(-0.5, 2.62); a.set_xlabel('Share of the gap to 100% local data closed\nby foreign data (median, 95% CI)')
a.legend(handles=[Line2D([], [], marker=MODEL_M[mo], ls='', ms=7, **sty(MODEL_C[mo]), label=f'{mo}' + (' (temporal)' if s == 'Temporal' else ''))
                  for s, mo, sty in series], loc='lower right', handletextpad=0.1, fontsize=8.5, borderaxespad=0.1)
panel(a, 'a', 'Foreign data replaces patterns, not profiles')
# (b) ItemKNN random: NDCG levels
D = np.array([r[2:-1] for r in T.detail('tab:scarcity_detail_ItemKNN_random')], float)
w = 0.28; xs = np.arange(3)
for i in range(3):
    lo, fo, gap = D[:, 1+3*i], D[:, 2+3*i], D[:, 3+3*i]; mm = ~np.isnan(gap)
    vals = [np.median(lo[mm]), np.median(fo[mm]), np.median(D[mm, 0])]
    for j, (v, c) in enumerate(zip(vals, [LOCAL, BLUE, INK2])):
        b.bar(xs[i] + (j-1)*w, v, w-0.03, color=c, edgecolor='white', lw=0)
        b.text(xs[i] + (j-1)*w, v + 0.012, f'{v:.2f}', ha='center', fontsize=7, color=INK2)
b.set_xticks(xs); b.set_xticklabels(['Patterns\nscarce', 'Both\nscarce', 'Histories\nscarce']); b.grid(axis='x', visible=False)
b.set_ylabel('NDCG@10 (median over contexts)'); b.set_ylim(0, 0.68)
b.legend(handles=[Patch(color=LOCAL, label='10% local'), Patch(color=BLUE, label='10% local + foreign'), Patch(color=INK2, label='100% local')],
         loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5, handlelength=1, columnspacing=0.7)
panel(b, 'b', 'ItemKNN, random split')
save(fig, 'fig3_scarcity', OUT)

# =====================================================================  Fig 4: launch
L = T.main('tab:launch'); lm = ['ItemKNN', 'EASE', 'UserKNN', 'MostPop']
fig, axs = plt.subplots(1, 2, figsize=(FULL_W, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, split, let in zip(axs, ['Random', 'Temporal'], 'ab'):
    D = np.array([r[2:] for r in T.detail(f'tab:launch_detail_{split.lower()}')], float)
    chance = [q for q in L if q[0] == split and q[1].startswith('Random')][0][3]
    ax.axvspan(0, chance, color=GRID, alpha=0.8, lw=0)
    for i, mo in enumerate(lm):
        yy = 3 - i; c = MODEL_C[mo]
        lo, fo = D[:, 2+2*i], D[:, 3+2*i]; mm = ~np.isnan(lo) & ~np.isnan(fo)
        ax.scatter(fo[mm], yy + rng.uniform(-0.18, 0.18, mm.sum()), s=9, color=c, alpha=0.28, lw=0)
        r = [q for q in L if q[0] == split and q[1] == mo][0]
        ax.plot([r[3], r[4]], [yy, yy], color=INK2, lw=1.4)
        ax.plot(r[3], yy, 'o', ms=7, mfc=LOCAL, mec=INK2, mew=0.8)
        ax.plot(r[4], yy, MODEL_M[mo], ms=8.5, mfc=c, mec='white', mew=1)
        ax.text(0.99, yy + 0.27, f'improved {r[6]}', transform=ax.get_yaxis_transform(), ha='right', fontsize=8, color=INK2)
    ax.text(chance + 0.008, -0.62, f'chance\n({chance:.3f})', fontsize=8, color=INK2, va='bottom')
    ax.set_xlim(0, 0.92); ax.set_ylim(-0.7, 3.62); ax.grid(axis='y', visible=False)
    ax.set_xlabel('NDCG@10 on launched products')
    panel(ax, let, f'{split} split')
axs[0].set_yticks(range(4)[::-1]); axs[0].set_yticklabels(lm)
fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=7, mfc=LOCAL, mec=INK2, label='local data only (median)'),
                    Line2D([], [], marker='o', ls='', ms=8, mfc=INK2, mec='white', label='+ foreign data (median)'),
                    Line2D([], [], marker='o', ls='', ms=4, color=INK2, alpha=0.35, label='+ foreign, each context')],
           loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5, handletextpad=0.2)
save(fig, 'fig4_launch', OUT)

# =====================================================================  Fig 5: history length
H = T.main('tab:history'); bk = ['1', '2', '3--4', '5+']; bkl = ['1', '2', '3–4', '5+']
fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 3.0), gridspec_kw=dict(wspace=0.3))
xs = np.arange(4)
for mo, off, sty in [('ItemKNN', -0.1, filled), ('EASE', 0.1, filled)]:
    rows = [[q for q in H if q[0] == mo and str(q[1]).replace('.0', '') == k][0] for k in bk]
    mu = np.array([T.ci(r[5]) for r in rows]); med = np.array([r[4] for r in rows]); c = MODEL_C[mo]
    for j in range(4):
        a.plot([xs[j]+off]*2, mu[j, 1:], color=c, lw=2, solid_capstyle='round')
    a.plot(xs+off, mu[:, 0], '-', color=c, lw=1.2, alpha=0.6)
    a.plot(xs+off, mu[:, 0], MODEL_M[mo], ms=7, **sty(c), label=mo)
    b.plot(xs+off, med, '-', color=c, lw=1.2, alpha=0.6)
    b.plot(xs+off, med, MODEL_M[mo], ms=7, **sty(c), label=mo)
nshop = [int([q for q in H if q[0] == 'ItemKNN' and str(q[1]).replace('.0', '') == k][0][3]) for k in bk]
for ax in (a, b):
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(xs); ax.grid(axis='x', visible=False)
    ax.set_xticklabels([f'{l}\n{n:,}' for l, n in zip(bkl, nshop)])
    ax.set_xlabel("Shopper's past local purchases\n(ItemKNN shoppers per group)")
a.set_ylabel('Change in NDCG@10'); a.legend(loc='upper right')
panel(a, 'a', 'Mean gain per shopper (95% CI)')
panel(b, 'b', 'Median over contexts')
import re
wil = dict(re.findall(r'\{(ItemKNN|EASE), within context:.*?Wilcoxon \$p=([0-9.]+)\$', open(T.ROOT + '/tables.tex', encoding='utf-8').read()))
b.text(0.97, 0.95, f"1–2 vs 3+ purchases, within context:\nItemKNN: $p$ = {float(wil['ItemKNN']):.3f}\nEASE: $p$ = {float(wil['EASE']):.2f}", transform=b.transAxes,
       ha='right', va='top', fontsize=8.5, color=INK2)
save(fig, 'fig5_history', OUT)
print('done')
