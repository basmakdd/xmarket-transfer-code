"""Data figure: purchases per market x category before and after preprocessing.
Source: figures/data/data_distribution.csv (built by data_stats.py + data_table.py from the raw XMarket zip
and analysis/results/contexts.parquet; data_table.py checks the k-core re-run matches the pipeline exactly)."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from style import *
setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', '..', 'paper', 'figures')
D = pd.read_csv(os.path.join(OUT, 'data', 'data_distribution.csv'))
NAMES = {'Arts_Crafts_and_Sewing': 'Arts & Crafts', 'Automotive': 'Automotive', 'Books': 'Books', 'CDs_and_Vinyl': 'CDs & Vinyl',
         'Cell_Phones_and_Accessories': 'Cell Phones', 'Digital_Music': 'Digital Music', 'Electronics': 'Electronics',
         'Grocery_and_Gourmet_Food': 'Grocery', 'Home_and_Kitchen': 'Home & Kitchen', 'Industrial_and_Scientific': 'Industrial',
         'Kindle_Store': 'Kindle Store', 'Movies_and_TV': 'Movies & TV', 'Musical_Instruments': 'Musical Instr.',
         'Office_Products': 'Office', 'Sports_and_Outdoors': 'Sports', 'Toys_and_Games': 'Toys'}
MK = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk']
D = D[D.present]
cats = D.groupby('category').raw_purchases.sum().sort_values(ascending=False).index.tolist()
before = D.pivot(index='category', columns='market', values='raw_purchases').reindex(index=cats, columns=MK)
after = D.pivot(index='category', columns='market', values='purchases_after').reindex(index=cats, columns=MK)
core = D.pivot(index='category', columns='market', values='core').reindex(index=cats, columns=MK)
cmap = LinearSegmentedColormap.from_list('seq', SEQ[1:])
norm = LogNorm(vmin=10, vmax=before.max().max())

def fmt(v):
    return f'{v/1e6:.1f}M' if v >= 1e6 else (f'{v/1e3:.0f}k' if v >= 1e4 else (f'{v/1e3:.1f}k' if v >= 1e3 else f'{int(v)}'))

fig = plt.figure(figsize=(FULL_W, 9.6))
gt = fig.add_gridspec(1, 2, wspace=0.08, left=0.15, right=0.98, top=0.955, bottom=0.465)
gb = fig.add_gridspec(1, 2, wspace=0.62, left=0.1, right=0.98, top=0.325, bottom=0.05)
axs = [fig.add_subplot(gt[0, 0]), fig.add_subplot(gt[0, 1])]
for ax, M, let, title in [(axs[0], before, 'a', 'Raw XMarket purchases'), (axs[1], after, 'b', 'After preprocessing')]:
    ax.grid(False); ax.set_xlim(-0.5, 6.5); ax.set_ylim(len(cats) - 0.5, -0.5)
    for i, c in enumerate(cats):
        for j, m in enumerate(MK):
            v = M.loc[c, m]; raw = before.loc[c, m]
            if np.isnan(raw):
                continue                                           # no file for this market
            if np.isnan(v) or v == 0:                               # dropped by preprocessing
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc='white', ec=GRID, hatch='////', lw=0))
                continue
            col = cmap(norm(v)); ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fc=col, ec='white', lw=1.5))
            dark = norm(v) > 0.6
            ax.text(j, i, fmt(v), ha='center', va='center', fontsize=7.5, color='white' if dark else INK)
    ax.set_xticks(range(7)); ax.set_xticklabels([m.upper() for m in MK]); ax.xaxis.tick_top()
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([NAMES[c] for c in cats] if let == 'a' else [])
    ax.set_title(f'({let}) {title}', loc='left', pad=22)
cb_ax = fig.add_axes([0.15, 0.415, 0.4, 0.011])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='horizontal')
cb.set_label('Purchases per market–category context (log scale)', fontsize=8.5); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
fig.legend(handles=[Patch(fc='white', ec=GRID, hatch='////', label='dropped: too small after k-core'),
                    Patch(fc=LOCAL, label='raw purchases'), Patch(fc=BLUE, label='after preprocessing')],
           loc='upper left', bbox_to_anchor=(0.6, 0.435), fontsize=8.5, handlelength=1.6, borderaxespad=0, labelspacing=0.35)

# (c) per market and (d) per category totals
for pos, key, order, labels, let, title in [
        (gb[0, 0], 'market', MK, [m.upper() for m in MK], 'c', 'Purchases per market'),
        (gb[0, 1], 'category', cats, [NAMES[c] for c in cats], 'd', 'Purchases per category')]:
    ax = fig.add_subplot(pos)
    g = D.groupby(key)[['raw_purchases', 'purchases_after']].sum().reindex(order).fillna(0)
    n_ctx = D.groupby(key).purchases_after.apply(lambda s: s.notna().sum()).reindex(order).fillna(0).astype(int)
    y = np.arange(len(order)); h = 0.4
    ax.barh(y - h/2, g.raw_purchases, h, color=LOCAL, lw=0, label='raw')
    ax.barh(y + h/2, g.purchases_after.replace(0, np.nan), h, color=BLUE, lw=0, label='after preprocessing')
    for yi, (b, a, n) in enumerate(zip(g.raw_purchases, g.purchases_after, n_ctx)):
        fs = 7 if key == 'category' else 8
        if key == 'market': ax.text(b * 1.15, yi - h/2, fmt(b), va='center', fontsize=fs, color=MUTED)
        ax.text(a * 1.15 if a > 0 else 1.2e2, yi + h/2, fmt(a) if a > 0 else 'none kept', va='center', fontsize=fs, color=INK2)
    ax.set_xscale('log'); ax.set_xlim(1e2, 6e7 if key == 'market' else 2e7); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5 if key == 'category' else 9.5)
    ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Purchases (log scale)')
    ax.set_title(f'({let}) {title}', loc='left', pad=6)
save(fig, 'fig0_data_distribution', OUT)
t = D[['raw_purchases', 'purchases_after']].sum()
print('raw', int(t.raw_purchases), 'after', int(t.purchases_after), 'contexts kept', int(D.purchases_after.notna().sum()), 'of', len(D))
