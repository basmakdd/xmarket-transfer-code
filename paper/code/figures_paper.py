"""Figures 1-9 of the paper, drawn only from result files (paper/results/*.csv, paper/raw/*.csv) and constants.

Usage:  python paper/code/figures_paper.py [--only 1,4,9] [--copy-to DIR]
Output: paper/figures/figure{N}.pdf and .png, and the exact values each figure draws in paper/figures/data/figureN.csv.
Style:  paper/code/figs/style.py (colour-blind-safe palette; every colour is paired with a marker, hatch or label).
"""
import argparse
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'figs'))
from style import *  # noqa: E402,F401,F403  (setup, colours, save)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(REPO, 'paper', 'results')
RAW = os.environ.get('PAPER_RAW', os.path.join(REPO, 'paper', 'raw'))   # as in common.py
RAW = RAW if os.path.isabs(RAW) else os.path.join(REPO, RAW)
FIG = os.path.join(REPO, 'paper', 'figures')
FDATA = os.path.join(FIG, 'data')
os.makedirs(FDATA, exist_ok=True)
setup()
plt.rcParams['pdf.compression'] = 9
ORANGE = '#d97a2b'            # second categorical slot (validated with blue, aqua, violet; CVD-safe)
ORIG, CLEAN, MID = BLUE, ORANGE, VIOLET    # data versions: original, variants merged within store (_nvc), across markets (_nv)
V_ORIG, V_NVC, V_NV = 'original data', 'variants merged within store', 'variants merged across markets'
SPLITS = [('random', 'Random (retrospective)'), ('temporal', 'Chronological (retrospective)'), ('calendar', 'Calendar cutoff (prospective)')]
ARMNAME = {'pool': 'pool', 'weight': 'weight', 'greedy': 'greedy'}


def rd(name, where=RES):
    return pd.read_csv(os.path.join(where, name), encoding='utf-8')


def out_csv(df, n):
    df.to_csv(os.path.join(FDATA, f'figure{n}.csv'), index=False, encoding='utf-8', lineterminator='\n')


def fsave(fig, n):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'figure{n}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.04,
                    metadata={'CreationDate': None} if ext == 'pdf' else {'Software': None})
    plt.close(fig)


def ci_line(ax, y, m, lo, hi, color, marker='o', ms=6.5, filled=True, lw=1.8, zorder=3, horizontal=True):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='round', zorder=zorder)
        ax.plot(m, y, marker, ms=ms, mfc=color if filled else 'white', mec='white' if filled else color,
                mew=0.8 if filled else 1.5, zorder=zorder + 1)
    else:
        ax.plot([y, y], [lo, hi], color=color, lw=lw, solid_capstyle='round', zorder=zorder)
        ax.plot(y, m, marker, ms=ms, mfc=color if filled else 'white', mec='white' if filled else color,
                mew=0.8 if filled else 1.5, zorder=zorder + 1)


# ============================================================================ pixel canvas for the two diagrams
class Canvas:
    """Draw in pixel coordinates (origin top-left, y down), like the SVG the diagrams were designed in."""

    def __init__(self, W, H, width_in=FULL_W):
        self.W, self.H = W, H
        self.k = width_in * 72 / W                     # points per pixel (font sizes, line widths)
        self.fig = plt.figure(figsize=(width_in, width_in * H / W))
        ax = self.fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
        ax.set_aspect('equal'); self.ax = ax

    def rect(self, x, y, w, h, fc='none', ec='none', lw=0, r=0, alpha=1, z=1, ls='-'):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}' if r else 'square,pad=0',
                                         fc=fc, ec=ec, lw=lw * self.k, alpha=alpha, zorder=z, ls=ls))

    def circle(self, cx, cy, r, fc='none', ec='none', lw=0, z=2, ls='-'):
        self.ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw * self.k, zorder=z, ls=ls))

    def line(self, pts, color, lw, z=2, ls='-', cap='round', alpha=1):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw * self.k, solid_capstyle=cap, dash_capstyle=cap, ls=ls, zorder=z, alpha=alpha)

    def path(self, verts, codes, fc='none', ec='none', lw=0, z=2, alpha=1):
        self.ax.add_patch(PathPatch(Path(verts, codes), fc=fc, ec=ec, lw=lw * self.k, zorder=z, alpha=alpha, capstyle='round', joinstyle='round'))

    def poly(self, pts, fc, ec='none', lw=0, z=2):
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw * self.k, zorder=z))

    def text(self, x, y, s, size=19, color=INK, weight='normal', ha='center', style='normal', z=5, va='baseline'):
        self.ax.text(x, y, s, fontsize=size * self.k, color=color, fontweight=weight, ha=ha, va=va, fontstyle=style, zorder=z)

    def arrow(self, x0, y0, x1, y1, color, lw=2.5, ls='-'):
        self.ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle='-|>', color=color, lw=lw * self.k, ls=ls, mutation_scale=14, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- icons (the 60 x 60 symbols of the original SVG)
def icon(cv, name, cx, cy, s=40, ring=True):
    if ring:
        cv.circle(cx, cy, s * 0.6, fc='white', ec='#D2D7DE', lw=1.6, z=3)
    f = s / 60.0; X = lambda u: cx - s / 2 + u * f; Y = lambda v: cy - s / 2 + v * f   # noqa: E731
    R = lambda x, y, w, h, c, r=0: cv.rect(X(x), Y(y), w * f, h * f, fc=c, r=r * f, z=4)  # noqa: E731
    if name == 'camera':
        R(20, 11, 14, 9, '#35598A', 2); R(7, 17, 46, 32, '#4C78A8', 6)
        cv.circle(X(30), Y(33), 11 * f, fc='#fff', z=5); cv.circle(X(30), Y(33), 7 * f, fc='#35598A', z=6); R(43, 21, 6, 4, '#fff', 1)
    elif name == 'card':
        cv.poly([(X(16), Y(7)), (X(39), Y(7)), (X(46), Y(14)), (X(46), Y(53)), (X(16), Y(53))], fc='#8E6BD1', z=4)
        for x in (20, 26, 32): cv.rect(X(x), Y(11), 3 * f, 9 * f, fc='#fff', z=5)
        cv.rect(X(21), Y(34), 20 * f, 12 * f, fc='#B9A3E6', r=2 * f, z=5)
    elif name == 'tripod':
        R(19, 6, 22, 10, '#5E6B78', 3)
        for x2 in (13, 30, 47): cv.line([(X(30), Y(17)), (X(x2), Y(54))], '#5E6B78', 4.5 * f, z=4)
    elif name == 'case':
        R(17, 5, 26, 50, '#E45F8B', 7); cv.rect(X(21), Y(9), 9 * f, 12 * f, fc='#F7B7CC', r=3 * f, z=5)
        cv.circle(X(30), Y(40), 5 * f, fc='#F7B7CC', z=5)
    elif name == 'cable':
        cv.path([(X(14), Y(16)), (X(14), Y(40)), (X(46), Y(20)), (X(46), Y(44))], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                ec='#E8A21A', lw=4.5 * f, z=4)
        R(8, 5, 12, 13, '#C98A10', 2); R(11, 1, 6, 5, '#9AA3AD'); R(40, 42, 12, 13, '#C98A10', 2); R(43, 54, 6, 5, '#9AA3AD')


def person(cv, x, y, s, col):
    """The 'user' symbol: head + shoulders inside an s x s box with top-left corner (x, y)."""
    f = s / 60.0
    cv.circle(x + 30 * f, y + 18 * f, 11 * f, fc=col, z=4)
    cv.path([(x + 8 * f, y + 58 * f), (x + 8 * f, y + 36 * f), (x + 52 * f, y + 36 * f), (x + 52 * f, y + 58 * f), (x + 8 * f, y + 58 * f)],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY], fc=col, z=4)


def badge(cv, x, y, ok, r=19):
    cv.circle(x, y, r, fc='#2E9E6A' if ok else '#D64545', z=5); k = r / 22
    if ok:
        cv.line([(x - 10 * k, y), (x - 3 * k, y + 8 * k), (x + 11 * k, y - 8 * k)], '#fff', 4, z=6)
    else:
        cv.line([(x - 8 * k, y - 8 * k), (x + 8 * k, y + 8 * k)], '#fff', 4, z=6); cv.line([(x + 8 * k, y - 8 * k), (x - 8 * k, y + 8 * k)], '#fff', 4, z=6)


def flag(cv, x, y, c, w=27, h=18):
    if c == 'de':
        for i, col in enumerate(['#1F1F1F', '#D52B1E', '#F2C200']): cv.rect(x, y + i * h / 3, w, h / 3, fc=col, z=4)
    else:
        for i, col in enumerate(['#1F8A4C', '#FFFFFF', '#C8102E']): cv.rect(x + i * w / 3, y, w / 3, h, fc=col, z=4)
        cv.circle(x + w / 2, y + h / 2, 2.5, fc='#8A6D3B', z=5)
    cv.rect(x, y, w, h, ec='#C9CED6', lw=1, z=6)


def store(cv, x, y, col, w=112, sh=64, aw=22, n=6, light=None):
    """Shop front: body with two windows and a striped awning with scalloped edge."""
    cv.rect(x, y + aw + 2, w, sh, fc='white', ec=col, lw=2.5, r=4, z=3)
    sw = w / n
    for i in range(n):
        x0 = x + i * sw; th = np.linspace(0, np.pi, 20)
        pts = [(x0, y), (x0 + sw, y), (x0 + sw, y + aw)] + [(x0 + sw / 2 + sw / 2 * np.cos(t), y + aw + sw / 4 * np.sin(t)) for t in th]
        cv.poly(pts, fc=col if i % 2 == 0 else 'white', ec=col, lw=1.6, z=4)
    cv.rect(x + w * 0.12, y + aw + 20, w * 0.38, sh - 18, fc=light or col, z=4, alpha=1 if light else 0.18)
    cv.rect(x + w * 0.58, y + aw + 20, w * 0.30, min(24, (sh - 18) * 0.55), fc=light or col, z=4, alpha=1 if light else 0.18)


# ============================================================================ Figure 1: toy example
def figure1():
    PW, PH, M, G = 485, 340, 8, 14
    W = M * 2 + PW * 2 + G; H = M * 2 + PH * 2 + G
    C = dict(de='#3B6FB6', mx='#E07A2E', ink='#2B2F36', sub='#5F6672', panel='#F3F4F7', line='#D2D7DE')
    cv = Canvas(W, H)
    texts = []
    def T(x, y, s, size=19, fill=C['ink'], w='normal', anchor='middle', it=False):  # noqa: E306
        cv.text(ox + x, oy + y, s, size=size, color=fill, weight='bold' if w == 'bold' else 'normal',
                ha={'middle': 'center', 'start': 'left', 'end': 'right'}[anchor], style='italic' if it else 'normal')
        texts.append(s)
    def P(i, title):  # noqa: E306
        nonlocal ox, oy
        ox = M + (i % 2) * (PW + G); oy = M + (i // 2) * (PH + G)
        cv.rect(ox, oy, PW, PH, fc=C['panel'], r=14, z=0); T(18, 34, title, 22, w='bold', anchor='start')
    ox = oy = 0
    # (a)
    P(0, '(a) Two stores, one catalogue')
    store(cv, ox + 40, oy + 54, C['de']); store(cv, ox + PW - 40 - 112, oy + 54, C['mx'])
    cv.arrow(ox + 166, oy + 108, ox + PW - 172, oy + 108, '#6B7280', lw=2.5, ls=(0, (2.4, 2)))
    T(PW / 2, 96, 'borrow?', 18, fill=C['sub'], it=True)
    flag(cv, ox + 40, oy + 158, 'de'); T(74, 174, 'Germany', 19, w='bold', anchor='start'); T(96, 198, 'many interactions', 18, fill=C['sub'])
    flag(cv, ox + PW - 152, oy + 158, 'mx'); T(PW - 118, 174, 'Mexico', 19, w='bold', anchor='start'); T(PW - 96, 198, 'few interactions', 18, fill=C['sub'])
    cv.rect(ox + 14, oy + 214, PW - 28, 56, fc='white', ec=C['line'], lw=1.6, r=10, z=1)
    T(28, 248, 'Same products', 19, w='bold', anchor='start')
    for i, n in enumerate(['camera', 'card', 'tripod', 'case', 'cable']): icon(cv, n, ox + 203 + i * 58, oy + 242, 34)
    cv.rect(ox + 14, oy + 278, PW - 28, 50, fc='white', ec=C['line'], lw=1.6, r=10, z=1)
    T(28, 309, 'Users keyed per store', 19, w='bold', anchor='start')
    for i in range(3):
        person(cv, ox + 262 + i * 26, oy + 290, 25, C['de']); person(cv, ox + 372 + i * 26, oy + 290, 25, C['mx'])
    cv.line([(ox + 352, oy + 288), (ox + 352, oy + 318)], C['line'], 2, z=5)
    # (b)
    P(1, '(b) Most-reviewed products differ')
    def bestlist(x, c, items, label):  # noqa: E306
        flag(cv, ox + x, oy + 58, c); T(x + 34, 74, label, 19, w='bold', anchor='start')
        for i, (n, v) in enumerate(items):
            yy = 118 + i * 62
            T(x + 12, yy + 7, f'#{i + 1}', 19, w='bold', fill=C['sub']); icon(cv, n, ox + x + 62, oy + yy, 38)
            cv.rect(ox + x + 94, oy + yy - 10, v, 20, fc=C[c], r=4, z=3)
    bestlist(24, 'de', [('camera', 96), ('tripod', 74), ('card', 54)], 'Germany')
    bestlist(276, 'mx', [('case', 96), ('cable', 74), ('card', 54)], 'Mexico')
    cv.line([(ox + PW / 2 + 6, oy + 60), (ox + PW / 2 + 6, oy + 262)], C['line'], 2, z=1); badge(cv, ox + PW / 2 + 6, oy + 180, False)
    T(PW / 2, 300, 'Most-reviewed products differ:', 19, w='bold')
    T(PW / 2, 324, 'German top products mislead in Mexico', 18, fill=C['sub'])
    # (c)
    P(2, '(c) Co-interaction links largely shared')
    def graph(x, c, label):  # noqa: E306
        flag(cv, ox + x + 50, oy + 58, c); T(x + 84, 74, label, 19, w='bold', anchor='start')
        cx = x + 112
        pos = {'camera': (cx, 112), 'card': (cx - 62, 172), 'tripod': (cx + 62, 172), 'case': (cx - 44, 238), 'cable': (cx + 44, 238)}
        for u, v in [('camera', 'card'), ('camera', 'tripod'), ('case', 'cable')]:
            cv.line([(ox + pos[u][0], oy + pos[u][1]), (ox + pos[v][0], oy + pos[v][1])], C[c], 5.5, z=2, alpha=0.85)
        for n, (xx, yy) in pos.items(): icon(cv, n, ox + xx, oy + yy, 36)
    graph(14, 'de', 'Germany'); graph(PW - 14 - 224, 'mx', 'Mexico')
    cv.line([(ox + PW / 2, oy + 60), (ox + PW / 2, oy + 262)], C['line'], 2, z=1); badge(cv, ox + PW / 2, oy + 205, True)
    T(PW / 2, 300, 'Co-interaction links mostly match:', 19, w='bold')
    T(PW / 2, 324, 'camera – card, tripod;  phone case – cable', 18, fill=C['sub'])
    # (d)
    P(3, '(d) What should Mexico borrow?')
    person(cv, ox + 22, oy + 52, 40, C['mx'])
    T(72, 70, 'A Mexican user just reviewed', 19, anchor='start'); T(72, 93, 'a phone case. Using German data:', 19, anchor='start')
    icon(cv, 'case', ox + PW - 52, oy + 74, 38)
    def row(y, title, note, items, ok):  # noqa: E306
        cv.rect(ox + 14, oy + y, PW - 28, 100, fc='white', ec='#2E9E6A' if ok else '#D64545', lw=2.2, r=12, z=1)
        T(32, y + 38, title, 19, w='bold', anchor='start'); T(32, y + 68, note[0], 18, anchor='start', fill=C['sub'])
        T(32, y + 89, note[1], 18, anchor='start', fill=C['sub'])
        for i, n in enumerate(items): icon(cv, n, ox + 334 + i * 52, oy + y + 62, 36)
        badge(cv, ox + PW - 44, oy + y + 50, ok)
    row(116, 'Borrow most-reviewed products', ('Suggests camera, tripod:', 'popular abroad, unrelated here'), ['camera', 'tripod'], False)
    row(226, 'Borrow co-interaction patterns', ('Suggests charging cable:', 'a link that also holds in Mexico'), ['cable'], True)
    fsave(cv.fig, 1)
    out_csv(pd.DataFrame({'element': ['text'] * len(texts), 'value': texts,
                          'note': 'illustrative toy example; no data from the study'}), 1)


# ============================================================================ Figure 2: data per market and category
NAMES = {'Arts_Crafts_and_Sewing': 'Arts & Crafts', 'Automotive': 'Automotive', 'Books': 'Books', 'CDs_and_Vinyl': 'CDs & Vinyl',
         'Cell_Phones_and_Accessories': 'Cell Phones', 'Digital_Music': 'Digital Music', 'Electronics': 'Electronics',
         'Grocery_and_Gourmet_Food': 'Grocery', 'Home_and_Kitchen': 'Home & Kitchen', 'Industrial_and_Scientific': 'Industrial',
         'Kindle_Store': 'Kindle Store', 'Movies_and_TV': 'Movies & TV', 'Musical_Instruments': 'Musical Instr.',
         'Office_Products': 'Office', 'Sports_and_Outdoors': 'Sports', 'Toys_and_Games': 'Toys & Games'}
MK = ['ca', 'de', 'fr', 'in', 'jp', 'mx', 'uk']


def fmt_n(v):
    return f'{v / 1e6:.1f}M' if v >= 1e6 else (f'{v / 1e3:.0f}k' if v >= 1e4 else (f'{v / 1e3:.1f}k' if v >= 1e3 else f'{int(v)}'))


def figure2():
    D = rd('data_distribution.csv', FDATA); D = D[D.present]
    J = rd('sens_diag_orig.csv')[['market', 'category', 'jac_mass']]
    D = D.merge(J, on=['market', 'category'], how='left')
    cats = D.groupby('category').raw_interactions.sum().sort_values(ascending=False).index.tolist()
    piv = lambda c: D.pivot(index='category', columns='market', values=c).reindex(index=cats, columns=MK)  # noqa: E731
    before, after, jac = piv('raw_interactions'), piv('interactions_after'), piv('jac_mass')
    cmap = LinearSegmentedColormap.from_list('seq', SEQ[1:]); norm = LogNorm(vmin=500, vmax=before.max().max())
    cmapj = LinearSegmentedColormap.from_list('seqj', ['#fbf3ea', '#f3d3b0', '#e8a96b', ORANGE, '#a3521a']); normj = Normalize(0, 1)
    fig = plt.figure(figsize=(FULL_W, 7.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.22, 0.78], height_ratios=[1, 1], wspace=0.1, hspace=0.42,
                          left=0.155, right=0.985, top=0.95, bottom=0.075)
    def heat(ax, M, let, title, norm_, cmap_, txt, show_y):  # noqa: E306
        ax.grid(False); ax.set_xlim(-0.5, 6.5); ax.set_ylim(len(cats) - 0.5, -0.5)
        for i, c in enumerate(cats):
            for j, m in enumerate(MK):
                raw = before.loc[c, m]
                if np.isnan(raw): continue                                   # no non-empty file for this market
                v = M.loc[c, m]
                if np.isnan(after.loc[c, m]):                                # dropped by the core filter
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc='white', ec=MUTED, hatch='////', lw=0)); continue
                col = cmap_(norm_(v)); ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=col, ec='white', lw=1.5))
                ax.text(j, i, txt(v), ha='center', va='center', fontsize=6.8, color='white' if norm_(v) > 0.6 else INK)
        ax.set_xticks(range(7)); ax.set_xticklabels([m.upper() for m in MK]); ax.xaxis.tick_top(); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels([NAMES[c] for c in cats] if show_y else [], fontsize=8.5)
        ax.set_title(f'({let}) {title}', loc='left', pad=20, fontsize=10)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); t = fig.add_subplot(gs[0, 3])
    heat(a, before, 'a', 'Raw interactions', norm, cmap, fmt_n, True)
    heat(b, after, 'b', 'Retained interactions', norm, cmap, fmt_n, False)
    # (b, right) totals per market, raw vs retained, with the number of contexts kept
    g = D.groupby('market')[['raw_interactions', 'interactions_after']].sum().reindex(MK)
    nk = D.groupby('market').interactions_after.apply(lambda s: s.notna().sum()).reindex(MK)
    y = np.arange(7); h = 0.38
    t.barh(y - h / 2, g.raw_interactions, h, color=LOCAL, lw=0, label='raw')
    t.barh(y + h / 2, g.interactions_after, h, color=BLUE, lw=0, label='retained')
    for yi, (r, af, n) in enumerate(zip(g.raw_interactions, g.interactions_after, nk)):
        t.text(r * 1.12, yi - h / 2, fmt_n(r), va='center', fontsize=7.5, color=INK2)
        t.text(af * 1.12, yi + h / 2, f'{fmt_n(af)} ({n})', va='center', fontsize=7.5, color=INK)
    t.set_xscale('log'); t.set_xlim(1e4, 5e8); t.set_yticks(y); t.set_yticklabels([m.upper() for m in MK]); t.invert_yaxis()
    t.grid(axis='y', visible=False); t.set_xlabel('Interactions (log scale)', fontsize=9)
    t.set_ylim(6.6, -1.5); t.legend(loc='upper right', fontsize=7.5, handlelength=1, borderaxespad=0.1, ncol=2, columnspacing=0.8)
    t.set_title('Totals per market\n(contexts kept)', loc='left', pad=6, fontsize=9, fontweight='normal')
    c = fig.add_subplot(gs[1, 0])
    heat(c, jac, 'c', 'Co-occurrence mass on near-identical\nproduct pairs (Jaccard ≥ 0.8)', normj, cmapj, lambda v: f'{100 * v:.0f}%', True)
    # colour bars and legend
    cb1 = fig.add_axes([0.53, 0.40, 0.42, 0.012]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb1, orientation='horizontal')
    cb.set_label('Interactions per market–category context, panels a–b (log scale)', fontsize=8); cb.ax.tick_params(labelsize=7.5); cb.outline.set_visible(False)
    cb2 = fig.add_axes([0.53, 0.29, 0.42, 0.012]); cbj = fig.colorbar(plt.cm.ScalarMappable(norm=normj, cmap=cmapj), cax=cb2, orientation='horizontal')
    cbj.set_label('Share of co-occurrence mass, panel c', fontsize=8); cbj.ax.tick_params(labelsize=7.5); cbj.outline.set_visible(False)
    cbj.set_ticks([0, .25, .5, .75, 1]); cbj.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    fig.legend(handles=[Patch(fc='white', ec=MUTED, hatch='////', label='dropped by the core filter (too small)'),
                        Patch(fc='white', ec=GRID, label='blank: no non-empty file for this market')],
               loc='upper left', bbox_to_anchor=(0.525, 0.215), fontsize=8, handlelength=1.6, labelspacing=0.5)
    fsave(fig, 2)
    o = D[['market', 'category', 'raw_interactions', 'core', 'interactions_after', 'jac_mass']].rename(
        columns={'interactions_after': 'retained_interactions', 'jac_mass': 'jaccard08_cooc_mass_share'})
    o['kept'] = o.retained_interactions.notna()
    out_csv(o.sort_values(['category', 'market']), 2)


# ============================================================================ Figure 3: RQ1 method diagram
def figure3():
    W = 750
    OR, ORL, ORD = '#E8822F', '#FBE6D3', '#C4641A'      # local store
    BL, BLL, BLD = '#3F6FB5', '#DCE6F4', '#2D548F'      # foreign store
    GR = '#2E9E62'; MUT, PAN, LINE = '#5a5a5a', '#EFF0F3', '#C9CDD4'
    TOP, H1, G = 4, 276, 12
    ax_, aw = 4, 200
    bx, bw = ax_ + aw + G, 272
    cx_ = bx + bw + G; cw = W - 4 - cx_
    dy, H2 = TOP + H1 + 10, 150
    H = dy + H2 + 4
    cv = Canvas(W, H)
    texts = []
    def T(x, y, s, size=17, w='normal', fill=INK, anchor='start', st='normal'):  # noqa: E306
        cv.text(x, y, s, size=size * 0.86, color=fill, weight=w, ha={'start': 'left', 'middle': 'center', 'end': 'right'}[anchor], style=st); texts.append(s)
    def panel(x, y, w, h, title):  # noqa: E306
        cv.rect(x, y, w, h, fc=PAN, r=12, z=0); T(x + 14, y + 27, title, 18, 'bold')
    def shop(cx, y, col, light):  # noqa: E306
        store(cv, cx - 32, y, col, w=64, sh=36, aw=10, n=4, light=light)
    def badge2(cx, cy):  # noqa: E306
        cv.circle(cx, cy, 11, fc='white', ec=LINE, lw=1.5, z=5)
        cv.line([(cx - 3, cy - 5), (cx + 3, cy), (cx - 3, cy + 5)], MUT, 2.4, z=6)
    # (a) local vs foreign store on shared products
    panel(ax_, TOP, aw, H1, '(a) Pair two stores')
    sl, sr = ax_ + 50, ax_ + aw - 50
    shop(sl, TOP + 42, OR, ORL); shop(sr, TOP + 42, BL, BLL)
    T(sl, TOP + 108, 'Local', 16, 'bold', OR, 'middle'); T(sr, TOP + 108, 'Foreign', 16, 'bold', BL, 'middle')
    cv.line([(sl + 36, TOP + 70), (sr - 36, TOP + 70)], MUT, 1.6, ls=(0, (2.5, 2)))
    T(ax_ + aw / 2, TOP + 132, 'same category,', 15, fill=MUT, anchor='middle')
    T(ax_ + aw / 2, TOP + 150, 'different country', 15, fill=MUT, anchor='middle')
    vy, r = TOP + 202, 30; vc = ax_ + 62; c1, c2 = vc - 20, vc + 20
    cv.circle(c1, vy, r, fc=ORL, z=2); cv.circle(c2, vy, r, fc=BLL, z=2)
    # lens (intersection) filled light yellow
    th1 = np.arccos(20 / r); a1 = np.linspace(-th1, th1, 40); a2 = np.linspace(np.pi - th1, np.pi + th1, 40)
    lens = [(c1 + r * np.cos(t), vy + r * np.sin(t)) for t in a1] + [(c2 + r * np.cos(t), vy + r * np.sin(t)) for t in a2]
    cv.poly(lens, fc='#FFF6C9', z=3)
    cv.circle(c1, vy, r, ec=OR, lw=2, z=4); cv.circle(c2, vy, r, ec=BL, lw=2, z=4)
    for d in (-13, 0, 13): cv.rect(vc - 5, vy + d - 5, 10, 10, fc=INK, r=2, z=5)
    for px, py in [(c1 - 17, vy - 10), (c1 - 20, vy + 10), (c2 + 17, vy - 10), (c2 + 20, vy + 10)]:
        cv.rect(px - 4, py - 4, 8, 8, fc='#A7ADB5', r=2, z=5)
    tx = ax_ + 122
    T(tx, vy - 14, 'keep only', 15, fill=MUT); T(tx, vy + 4, 'shared', 15, 'bold'); T(tx, vy + 22, 'products', 15, 'bold')
    # (b) samples
    panel(bx, TOP, bw, H1, '(b) Build samples')
    T(bx + 14, TOP + 52, 'Local users: random halves', 15, 'bold', OR); T(bx + bw - 14, TOP + 52, '×3', 15, fill=MUT, anchor='end')
    ry = TOP + 80; gA = bx + 14; gB = bx + bw - 14 - 118
    for gx, lab, fill_ in [(gA, 'A', ORD), (gB, 'B', OR)]:
        cv.rect(gx, ry - 20, 118, 38, fc='white', ec=fill_, lw=2.6 if lab == 'A' else 1.5, r=8, z=2)
        T(gx + 12, ry + 7, lab, 19, 'bold', fill_)
        for k in range(3): person(cv, gx + 38 + k * 24, ry - 13, 24, fill_)
    T(gA + 59, ry + 36, 'reference', 15, 'bold', ORD, 'middle'); T(gB + 59, ry + 36, 'local sample (L)', 15, 'normal', OR, 'middle')
    T(bx + 14, TOP + 146, 'Foreign users', 15, 'bold', BL); T(bx + bw - 14, TOP + 146, 'all (F)', 15, 'bold', BL, 'end')
    fy = TOP + 172
    cv.rect(bx + 14, fy - 20, bw - 28, 38, fc='white', ec=BL, lw=1.5, r=8, z=2)
    sel = {1, 4, 6}
    for k in range(9):
        px = bx + 36 + k * 25
        person(cv, px - 12, fy - 13, 24, BL if k in sel else '#AFC3E3')
        if k in sel: cv.circle(px, fy - 1, 14, ec=BLD, lw=1.6, z=5, ls=(0, (2, 1.5)))
    cv.circle(bx + 22, TOP + 205, 7, ec=BLD, lw=1.6, z=5, ls=(0, (2, 1.5)))
    T(bx + 36, TOP + 211, 'matched: B and foreign cut to', 15, fill=BLD); T(bx + 36, TOP + 227, 'the same interaction volume', 15, fill=BLD)
    # null: shuffle symbol
    sx, sy = bx + 22, TOP + 245
    cv.line([(sx - 7, sy - 5), (sx + 7, sy + 5)], MUT, 1.8, z=5); cv.line([(sx - 7, sy + 5), (sx + 7, sy - 5)], MUT, 1.8, z=5)
    T(bx + 36, sy + 6, 'null: foreign data shuffled,', 15, fill=MUT); T(bx + 36, sy + 22, 'interaction counts kept', 15, fill=MUT)
    # (c) compare with half A
    panel(cx_, TOP, cw, H1, '(c) Compare each with A')
    def kbox(y, head, lines, drawicon, col):  # noqa: E306
        cv.rect(cx_ + 12, y, cw - 24, 104, fc='white', ec=LINE, lw=1.5, r=8, z=1)
        drawicon(cx_ + 24, y + 8)
        for q, h_ in enumerate(head.split('\n')): T(cx_ + 60, y + (26 if '\n' not in head else 19 + 17 * q), h_, 16, 'bold', col)
        for i, l in enumerate(lines): T(cx_ + 22, y + (52 if '\n' not in head else 60) + i * 19, ('• ' + l) if not l.startswith(' ') else '   ' + l.strip(), 14.5) if True else None
    def bars(x, y):  # noqa: E306
        for i, hh in enumerate([22, 15, 9]): cv.rect(x + i * 9, y + 24 - hh, 7, hh, fc=MUT, r=1.5, z=3)
    def graph(x, y):  # noqa: E306
        pts = [(x + 3, y + 18), (x + 14, y + 3), (x + 26, y + 16), (x + 16, y + 24)]
        for i, j in [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]: cv.line([pts[i], pts[j]], MUT, 1.6, z=3)
        for p in pts: cv.circle(p[0], p[1], 3.8, fc=GR, z=4)
    kbox(TOP + 42, 'Popularity', ['interaction-count rank', '  correlation (Spearman)', 'top-10 overlap'], bars, INK)
    kbox(TOP + 156, 'Co-interaction\npatterns', ['co-interaction cosine', '5-nearest-neighbour', '  overlap'], graph, INK)
    badge2(bx - G / 2, TOP + H1 / 2); badge2(cx_ - G / 2, TOP + H1 / 2)
    # (d) reported quantities
    panel(ax_, dy, W - 8, H2, '(d) Reported quantities (per target–donor pair; median over pairs, 95% CI)')
    cols = [ax_ + 14, ax_ + 262, ax_ + 504]
    for xd in cols[1:]: cv.line([(xd - 14, dy + 42), (xd - 14, dy + H2 - 10)], LINE, 1.5, z=1)
    y1, y2, y3, y4 = dy + 66, dy + 92, dy + 114, dy + 136
    x = cols[0]
    T(x, y1, 'Difference  D = F − L', 16, 'bold')
    T(x, y2, 'F: foreign vs A', 14.5, fill=BL); T(x, y3, 'L: local B vs A', 14.5, fill=OR); T(x, y4, '0 = as good as local', 14.5, fill=MUT)
    x = cols[1]
    T(x, y1, 'Excess over chance  F − c', 16, 'bold'); T(x, y2, 'c: null sample (patterns) or', 14.5)
    T(x, y3, 'random ranking (popularity)', 14.5); T(x, y4, '> 0: more than chance', 14.5, fill=MUT)
    x = cols[2]
    T(x, y1, r'Paired  $D_{\rm pat} - D_{\rm pop}$', 16, 'bold'); T(x, y2, 'cosine vs rank correlation;', 14.5)
    T(x, y3, '5-NN vs top-10', 14.5); T(x, y4, '> 0: patterns lose less', 14.5, 'bold', GR)
    fsave(cv.fig, 3)
    out_csv(pd.DataFrame({'element': 'text', 'value': texts, 'note': 'method diagram; no data values'}), 3)


# ============================================================================ Figure 4: RQ1 results
MEAS = [('sp', 'Rank correlation', POP), ('top10', 'Top-10 overlap', POP), ('cos', 'Co-interaction cosine', PAT), ('nbr5', '5-NN overlap', PAT)]


def figure4():
    o = rd('rq1_T1.csv'); o = o[(o.variant == 'main') & (o.arm == 'all')].assign(version=V_ORIG, source='paper/results/rq1_T1.csv')
    n = rd('sens2_rq1_measures.csv'); n = n[(n['var'] == '_nvc') & (n['set'] == 'all') & (n.arm == 'all')]
    n = n.assign(version=V_NVC, source='paper/results/sens2_rq1_measures.csv')
    X = pd.concat([o, n])[['version', 'measure', 'pairs', 'local', 'foreign', 'null_foreign', 'chance_median', 'D', 'D_lo', 'D_hi',
                           'excess', 'excess_lo', 'excess_hi', 'source']]
    X = X[X.measure.isin([m for m, _, _ in MEAS])]
    fig = plt.figure(figsize=(FULL_W, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.75, wspace=0.12, left=0.21, right=0.985, top=0.895, bottom=0.08)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for k, (ax, q, xl) in enumerate([(axs[0], 'D', 'Foreign − local agreement, D = F − L'),
                                     (axs[1], 'excess', 'Foreign − chance, F − c')]):
        for i, (m, lab, col) in enumerate(MEAS):
            for off, ver, colr, mk, fill in [(0.16, V_ORIG, ORIG, 'o', True), (-0.16, V_NVC, CLEAN, 's', False)]:
                r = X[(X.measure == m) & (X.version == ver)].iloc[0]
                ci_line(ax, 3 - i + off, r[q], r[f'{q}_lo'], r[f'{q}_hi'], colr, mk, filled=fill)
                ax.text(max(r[f'{q}_hi'], r[q]) + 0.015, 3 - i + off, f'{r[q]:+.2f}', va='center', fontsize=7.5, color=INK2, zorder=8,
                        bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85))
        ax.axvline(0, color=INK2, lw=1); ax.axhline(1.5, color=GRID, lw=1)
        ax.set_ylim(-0.6, 3.6); ax.grid(axis='y', visible=False); ax.set_xlabel(xl, fontsize=9)
        ax.set_yticks(range(4)[::-1])
        ax.set_yticklabels([f'{lab}' for _, lab, _ in MEAS] if k == 0 else [])
        if k == 0:
            for tl, (_, _, col) in zip(ax.get_yticklabels(), MEAS): tl.set_color(col)
            ax.text(-0.33, 3.55, 'POPULARITY', transform=ax.get_yaxis_transform(), color=POP, fontsize=7.5, fontweight='bold', ha='left', va='top')
            ax.text(-0.33, 1.4, 'PATTERNS', transform=ax.get_yaxis_transform(), color=PAT, fontsize=7.5, fontweight='bold', ha='left', va='top')
    axs[0].set_xlim(-0.72, 0.22); axs[1].set_xlim(-0.12, 0.55)
    fig.text(0.015, 0.975, '(a) Agreement with local half A: median over target–donor pairs (95% CI)', fontsize=10.5, fontweight='bold')
    fig.legend(handles=[Line2D([], [], marker='o', ls='-', color=ORIG, mfc=ORIG, mec='white', ms=6.5, label='original data (324 pairs; 5-NN 221)'),
                        Line2D([], [], marker='s', ls='-', color=CLEAN, mfc='white', mec=CLEAN, mew=1.5, ms=6.5,
                               label='variants merged within store (338–340 pairs; 5-NN 277)')],
               loc='upper left', bbox_to_anchor=(0.01, 0.962), ncol=2, fontsize=7.6, handletextpad=0.4, columnspacing=1.2)
    # (b) raw local vs foreign agreement per pair
    F = rd('rq1_fig4_data.csv')
    bx = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    rows = []
    for ax, m, lab, col in [(bx[0], 'sp', 'Rank correlation (popularity)', POP), (bx[1], 'cos', 'Co-interaction cosine (patterns)', PAT)]:
        x, y = F[f'loc_{m}'], F[f'for_{m}']; ok = x.notna() & y.notna()
        ax.scatter(x[ok], y[ok], s=13, color=col, alpha=0.55, lw=0, zorder=3)
        lo = min(-0.2, float(np.floor(min(x[ok].min(), y[ok].min()) * 10) / 10)); hi = 1.0
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--', zorder=2)
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        ax.set_xlim(lo - 0.03, hi + 0.03); ax.set_ylim(lo - 0.03, hi + 0.03); ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Local agreement L (half B vs A)', fontsize=9)
        ax.set_title(f'{lab}, {int(ok.sum())} pairs', loc='left', fontsize=9, fontweight='normal', pad=4)
        ax.text(0.03, 0.97, 'y = x: foreign as good as local', transform=ax.transAxes, fontsize=7.5, color=INK2, va='top')
        rows.append(pd.DataFrame({'measure': m, 't': F.t, 'donor_market': F.donor_market, 'L': x, 'F': y}))
    bx[0].set_ylabel('Foreign agreement F (vs A)', fontsize=9)
    bx[0].text(0, 1.02, 'Rank correlation (popularity), 324 pairs', transform=bx[0].transAxes, fontsize=9, va='bottom')
    bx[0].set_title(''); bx[1].set_title('')
    bx[1].text(0, 1.02, 'Co-interaction cosine (patterns), 324 pairs', transform=bx[1].transAxes, fontsize=9, va='bottom')
    fig.text(0.015, 0.46, '(b) Each target–donor pair, original data: local vs foreign agreement (raw values, not clipped)', fontsize=10.5, fontweight='bold')
    fsave(fig, 4)
    a = X.assign(panel='a'); b = pd.concat(rows).assign(panel='b', version=V_ORIG, source='paper/results/rq1_fig4_data.csv')
    out_csv(pd.concat([a, b], ignore_index=True), 4)


# ============================================================================ Figure 5: RQ2 complete local data
def figure5():
    S = rd('rq2_fig5_summary.csv'); S = S[S.arm.isin(['pool', 'weight', 'greedy'])].copy()
    C = rd('sens2_rq2.csv'); C = C[(C['var'] == '_nvc') & (C['set'] == 'all') & (C.arm == 'pool') & C.model.isin(['ItemKNN', 'EASE', 'MostPop'])]
    rows = [(m, a) for m in ['MostPop', 'ItemKNN', 'EASE', 'UserKNN'] for a in ['pool', 'weight', 'greedy']
            if ((S.model == m) & (S.arm == a)).any()]
    ypos = {}; y = 0
    for i, (m, a) in enumerate(rows):
        if i and rows[i - 1][0] != m: y += 0.6
        ypos[(m, a)] = y; y += 1
    ymax = y - 1
    fig = plt.figure(figsize=(FULL_W, 7.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.5, wspace=0.08, left=0.16, right=0.99, top=0.87, bottom=0.07)
    AM = {'pool': 'o', 'weight': 's', 'greedy': '^'}
    for j, (sp, title) in enumerate(SPLITS):
        ax = fig.add_subplot(gs[0, j])
        for (m, a), yy in ypos.items():
            r = S[(S.split == sp) & (S.model == m) & (S.arm == a)]
            if r.empty:
                ax.text(-0.155, yy, 'not run', fontsize=7, color=MUTED, va='center'); continue
            r = r.iloc[0]
            ci_line(ax, yy, r.gain_mean, r.gain_mean_lo, r.gain_mean_hi, MODEL_C[m], AM[a], ms=6)
            ax.plot(r.gain_median, yy, '|', ms=9, mew=1.6, color=INK, zorder=6)
            if sp == 'random' and not np.isnan(r.gain_mean_exIN):
                ax.plot(r.gain_mean_exIN, yy, 'x', ms=6, mew=1.5, color=INK, zorder=7)
            c = C[(C.split == sp) & (C.model == m) & (C.arm == a)]
            if len(c):
                c = c.iloc[0]; ax.plot(c.gain_mean, yy - 0.38, 'D', ms=5, mfc='white', mec=CLEAN, mew=1.5, zorder=7)
                ax.plot([c.gain_mean_lo, c.gain_mean_hi], [yy - 0.38] * 2, color=CLEAN, lw=1.2, zorder=6)
        ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-0.165, 0.095); ax.set_ylim(ymax + 0.7, -0.7)
        ax.grid(axis='y', visible=False); ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([f'{m} · {a}' for m, a in ypos] if j == 0 else [], fontsize=8.5)
        ax.set_xticks([-0.15, -0.1, -0.05, 0, 0.05]); ax.tick_params(axis='x', labelsize=8)
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='normal', pad=4)
        if j == 1: ax.set_xlabel('Mean change in NDCG@10 from foreign data (95% CI)', fontsize=9)
    fig.text(0.16, 0.975, '(a) Complete local data: mean gain over local only, per model, arm and split', fontsize=10.5, fontweight='bold')
    fig.legend(handles=[Line2D([], [], marker=AM[a], ls='-', color=INK2, mfc=INK2, mec='white', ms=6, label=f'{a}, {V_ORIG}') for a in AM] +
               [Line2D([], [], marker='|', ls='', color=INK, ms=9, mew=1.6, label='median'),
                Line2D([], [], marker='x', ls='', color=INK, ms=6, mew=1.5, label='mean without IN Arts & Crafts (random)'),
                Line2D([], [], marker='D', ls='-', color=CLEAN, mfc='white', mec=CLEAN, mew=1.5, ms=5, label=f'pool, {V_NVC} (open diamond below the row)')],
               loc='upper left', bbox_to_anchor=(0.155, 0.968), ncol=3, fontsize=7.3, handletextpad=0.5, columnspacing=1.2)
    # (b) better / worse counts (> 0.01 / < -0.01)
    for j, (sp, title) in enumerate(SPLITS):
        ax = fig.add_subplot(gs[1, j])
        for (m, a), yy in ypos.items():
            r = S[(S.split == sp) & (S.model == m) & (S.arm == a)]
            if r.empty: continue
            r = r.iloc[0]
            ax.barh(yy, r['better_0.01'], 0.7, color=MODEL_C[m], lw=0); ax.barh(yy, -r['worse_0.01'], 0.7, color=LOCAL, lw=0, hatch='///', ec='white')
            ax.text(r['better_0.01'] + 1, yy, str(int(r['better_0.01'])), va='center', fontsize=7, color=INK)
            ax.text(-r['worse_0.01'] - 1, yy, str(int(r['worse_0.01'])), va='center', ha='right', fontsize=7, color=INK)
        ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-64, 44); ax.set_ylim(ymax + 0.7, -0.7)
        ax.grid(axis='y', visible=False); ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([f'{m} · {a}' for m, a in ypos] if j == 0 else [], fontsize=8.5)
        ax.set_xticks([-60, -40, -20, 0, 20, 40]); ax.set_xticklabels(['60', '40', '20', '0', '20', '40'], fontsize=8)
        ax.set_title(title, loc='left', fontsize=9.5, fontweight='normal', pad=4)
        if j == 1: ax.set_xlabel('Contexts losing (left, hatched) / gaining (right) more than 0.01', fontsize=9)
    fig.text(0.16, 0.425, '(b) Contexts better or worse than local only (original data)', fontsize=10.5, fontweight='bold')
    fsave(fig, 5)
    o = S[['split', 'model', 'arm', 'contexts', 'local_median', 'gain_median', 'gain_mean', 'gain_mean_lo', 'gain_mean_hi',
           'gain_mean_exIN', 'better_0.01', 'worse_0.01']].assign(data=V_ORIG, source='paper/results/rq2_fig5_summary.csv')
    c = C[['split', 'model', 'arm', 'contexts', 'local_median', 'gain_median', 'gain_mean', 'gain_mean_lo', 'gain_mean_hi', 'better', 'worse']]
    c = c.rename(columns={'better': 'better_0.01', 'worse': 'worse_0.01'}).assign(data=V_NVC, source='paper/results/sens2_rq2.csv')
    out_csv(pd.concat([o, c], ignore_index=True), 5)


# ============================================================================ Figure 6: gain by local history
def figure6():
    H = rd('rq2_fig6_history.csv'); Dd = rd('rq2_history_d.csv'); Tt = rd('rq2_history_test.csv')
    B = ['1', '2', '3-4', '5+']
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.8], wspace=0.42))
    OFF = {'ItemKNN': -0.12, 'EASE': 0.12}
    for ax, q, title, let in [(axs[0], 'mean', 'Mean over contexts', 'a'), (axs[1], 'median', 'Median over contexts', 'b')]:
        for m in ['ItemKNN', 'EASE']:
            h = H[H.model == m].set_index('bucket').reindex(B)
            for i, bk in enumerate(B):
                r = h.loc[bk]
                ci_line(ax, i + OFF[m], r[q], r[f'{q}_lo'], r[f'{q}_hi'], MODEL_C[m], MODEL_M[m], ms=6, horizontal=False)
                if q == 'mean': ax.plot(i + OFF[m] + 0.11, r.mean_exIN, 'x', ms=5, mew=1.3, color=MODEL_C[m], zorder=6)
        ax.axhline(0, color=INK2, lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(['1', '2', '3–4', '5+'])
        ax.set_xlabel('Local training interactions\nof the test user', fontsize=9); ax.grid(axis='x', visible=False)
        ax.set_title(f'({let}) {title}', loc='left', fontsize=10, pad=6)
    axs[0].set_ylabel('Gain in NDCG@10 (pool − local)', fontsize=9)
    axs[0].set_ylim(min(-0.03, H.mean_lo.min() - 0.01), H.mean_hi.max() + 0.015)
    axs[1].set_ylim(H.median_lo.min() - 0.008, H.median_hi.max() + 0.008)
    axs[0].legend(handles=[Line2D([], [], marker=MODEL_M[m], ls='-', color=MODEL_C[m], mec='white', ms=6, label=m) for m in OFF] +
                  [Line2D([], [], marker='x', ls='', color=INK2, ms=5, mew=1.3, label='mean without\nIN Arts & Crafts')],
                  loc='upper right', fontsize=7.5, handletextpad=0.3, borderaxespad=0.1)
    ax = axs[2]; rng = np.random.default_rng(6)
    for i, m in enumerate(['ItemKNN', 'EASE']):
        d = Dd[Dd.model == m].d.values
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(d)), d, s=10, color=MODEL_C[m], alpha=0.5, lw=0, zorder=2)
        t = Tt[Tt.model == m].iloc[0]
        ax.plot([i - 0.28, i + 0.28], [t.mean_d] * 2, color=INK, lw=2, zorder=4)
        ax.plot([i + 0.3] * 2, [t.mean_lo, t.mean_hi], color=INK, lw=2, zorder=4, solid_capstyle='round')
        ax.text(i, Dd.d.max() + 0.012, f'{int(t.short_more)}/{int(t.contexts)}\nd > 0', ha='center', fontsize=7.2, color=INK2)
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks([0, 1]); ax.set_xticklabels(['ItemKNN', 'EASE']); ax.grid(axis='x', visible=False)
    ax.set_ylim(Dd.d.min() - 0.035, Dd.d.max() + 0.045); ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel('d = gain(1–2) − gain(3+)', fontsize=9)
    ax.text(0.5, 0.012, 'dots: contexts\nline: mean; bar: 95% CI', transform=ax.transAxes, ha='center', fontsize=6.6, color=INK2)
    ax.set_title('(c) Within context', loc='left', fontsize=10, pad=6)
    fsave(fig, 6)
    a = H.assign(panel='a,b', source='paper/results/rq2_fig6_history.csv')
    c = Dd.assign(panel='c', source='paper/results/rq2_history_d.csv')
    t = Tt.assign(panel='c (mean and CI)', source='paper/results/rq2_history_test.csv')
    out_csv(pd.concat([a, c, t], ignore_index=True), 6)


# ============================================================================ Figure 7: scarcity
ARMS3 = [('P', 'patterns scarce', BLUE, 'o'), ('B', 'both scarce', VIOLET, 's'), ('R', 'histories scarce', INK2, '^')]


def figure7():
    S = rd('sc_figure7_data.csv'); K = rd('sc_buckets.csv')
    fig = plt.figure(figsize=(FULL_W, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.72], hspace=0.62, wspace=0.28, left=0.09, right=0.985, top=0.81, bottom=0.07)
    F = [0.05, 0.10, 0.25, 0.50]; xs = np.arange(4)
    others = [('EASE', 'random', 'D'), ('UserKNN', 'random', 'v'), ('ItemKNN', 'temporal', 'P'), ('ItemKNN', 'calendar', 'X')]
    OFFO = [-0.24, -0.12, 0.12, 0.24]
    for k, (q, ylab, let, title) in enumerate([('gain', 'Gain in NDCG@10\n(median, 95% CI)', 'a', 'Absolute gain from foreign data'),
                                                ('G', 'Share of the gap closed\n(median, 95% CI)', 'b', 'Gap to complete local data closed')]):
        ax = fig.add_subplot(gs[0, k])
        for arm, lab, col, mk in ARMS3:
            s = S[(S.model == 'ItemKNN') & (S.split == 'random') & (S.arm == arm)].set_index('f').reindex(F)
            ax.fill_between(xs, s[f'{q}_lo'], s[f'{q}_hi'], color=col, alpha=0.13, lw=0)
            ax.plot(xs, s[f'{q}_med'], '-', color=col, lw=1.8, marker=mk, ms=6.5, mec='white', mew=0.8, zorder=4)
            for (mo, sp, om), off in zip(others, OFFO):
                o = S[(S.model == mo) & (S.split == sp) & (S.arm == arm)].set_index('f').reindex(F)
                ax.plot(xs + off, o[f'{q}_med'], om, ls='', ms=3.6, color=col, alpha=0.85, mew=0.5, mec=col, zorder=3)
        ax.axhline(0, color=INK2, lw=1); ax.set_xticks(xs); ax.set_xticklabels(['5%', '10%', '25%', '50%'])
        ax.set_xlabel('Share f of local training interactions kept', fontsize=9); ax.set_ylabel(ylab, fontsize=9)
        ax.grid(axis='x', visible=False); ax.set_title(f'({let}) {title}', loc='left', fontsize=10, pad=6)
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ms=6.5, mec='white', label=lab) for _, lab, c, mk in ARMS3] +
               [Line2D([], [], ls='', marker=om, ms=4.5, color=INK2, label={'temporal': f'{mo} chronological', 'calendar': f'{mo} calendar cutoff'}.get(sp, mo))
                for mo, sp, om in others],
               loc='upper left', bbox_to_anchor=(0.085, 0.985), ncol=4, fontsize=7.6, handletextpad=0.3, columnspacing=1.0)
    fig.text(0.088, 0.858, 'Original data. Lines and bands: ItemKNN, random (retrospective) split. Small markers: EASE and UserKNN (random split),\n'
             'ItemKNN chronological (retrospective) and ItemKNN calendar cutoff (prospective); UserKNN not run at 5%.', fontsize=7.3, color=INK2, va='bottom')
    # (c) buckets at f = 0.10
    ax = fig.add_subplot(gs[1, :])
    k = K[(K.model == 'ItemKNN') & (K.split == 'random') & (K.f == 0.1)]
    BK = ['0', '1', '2+']; BC = {'0': LOCAL, '1': AQUA, '2+': BLUE}; BH = {'0': '', '1': '', '2+': ''}
    w = 0.24
    for i, (arm, lab, _, _) in enumerate(ARMS3):
        for j, bk in enumerate(BK):
            r = k[(k.arm == arm) & (k.bucket.astype(str) == bk)].iloc[0]
            x = i + (j - 1) * w
            if r.n_ctx_nonempty == 0 or np.isnan(r.gain_med):
                ax.text(x, 0.008, 'no\nusers', ha='center', fontsize=6.5, color=MUTED, va='bottom'); continue
            if r.gain_med == 0 and r.gain_lo == 0 and r.gain_hi == 0:
                ax.plot(x, 0, '_', ms=12, mew=2.5, color=INK2); ax.text(x, 0.008, 'no gain\n(no history)', ha='center', fontsize=6.5, color=INK2, va='bottom')
            ax.bar(x, r.gain_med, w - 0.03, color=BC[bk], lw=0, hatch=BH[bk])
            if r.gain_hi > r.gain_lo: ax.plot([x, x], [r.gain_lo, r.gain_hi], color=INK, lw=1.2)
            ax.text(x, -0.045, f'{100 * r.share_med:.0f}%', ha='center', fontsize=7.2, color=INK2)
    ax.text(2.47, 0.315, 'Percent below each bar: share of test users\nin that group, median over contexts', ha='right', va='top', fontsize=7.2, color=INK2)
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(range(3)); ax.set_xticklabels([lab for _, lab, _, _ in ARMS3])
    ax.tick_params(axis='x', pad=13, length=0); ax.set_xlim(-0.75, 2.5); ax.set_ylim(-0.055, 0.40); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Gain in NDCG@10\n(median, 95% CI)', fontsize=9)
    ax.set_title('(c) At f = 10%, by retained local interactions of the test user (ItemKNN, random (retrospective) split)', loc='left', fontsize=10, pad=6)
    ax.legend(handles=[Patch(color=BC[b], label=f'{b} retained') for b in BK], loc='upper right', fontsize=7.6, ncol=3,
              handlelength=1, columnspacing=0.8)
    fsave(fig, 7)
    out_csv(pd.concat([S.assign(panel='a,b', source='paper/results/sc_figure7_data.csv'),
                       k.assign(panel='c', source='paper/results/sc_buckets.csv')], ignore_index=True), 7)


# ============================================================================ Figure 8: launch
def figure8():
    L = rd('l3_fig8.csv'); L = L[(L.tuning == 'zf') & L.model.isin(['ItemKNN', 'EASE', 'MostPop']) & L.in_common]   # common context sets (all five models have results)
    MODS = ['ItemKNN', 'EASE', 'MostPop']
    fig, axs = plt.subplots(3, 3, figsize=(FULL_W, 7.2), gridspec_kw=dict(hspace=0.5, wspace=0.32, top=0.86))
    for j, (sp, title) in enumerate(SPLITS):
        sub = L[L.split == sp]
        padx = 0.08 * (sub.dL.max() - sub.dL.min()); pady = 0.06 * (sub.dF.max() - sub.dF.min())
        xl = (sub.dL.min() - padx, max(sub.dL.max(), 0) + padx); yl = (min(sub.dF.min(), 0) - pady, sub.dF.max() + pady)
        for i, m in enumerate(MODS):
            ax = axs[i, j]; d = sub[sub.model == m]
            ax.axhline(0, color=INK2, lw=1, zorder=1); ax.axvline(0, color=INK2, lw=1, zorder=1)
            lo_, hi_ = max(xl[0], yl[0]), min(xl[1], yl[1])
            ax.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=0.9, ls='--', zorder=1)
            ax.scatter(d.dL, d.dF, s=12, color=MODEL_C[m], marker=MODEL_M[m], alpha=0.65, lw=0, zorder=3)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.tick_params(labelsize=7.5)
            ax.set_title(f'{m}, n = {len(d)} contexts', loc='left', fontsize=8.5, fontweight='normal', pad=3)
            if i == 0: ax.text(0, 1.2, title, transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='bottom')
            if j == 0: ax.set_ylabel('With foreign data\n\u2212 chance', fontsize=8.5)
            if i == 2: ax.set_xlabel('Local data only \u2212 chance', fontsize=8.5)
    fig.suptitle('Launched-product NDCG@10 minus the random ranker, per context (zero-feedback tuning, original data)', x=0.08, ha='left', y=0.985,
                 fontsize=10, fontweight='bold')
    fig.text(0.08, 0.962, 'Common context sets (all five models have results). Lines at zero mark chance; dashed line: no change from adding\n'
             'foreign data. Axis ranges differ between splits.',
             fontsize=7.6, color=INK2, va='top')
    fsave(fig, 8)
    out_csv(L.assign(source='paper/results/l3_fig8.csv'), 8)


# ============================================================================ Figure 9: country closeness
# Country features: copied verbatim from paper/code/rq4_tables.py (`python run.py check` verifies that they match).
LANG = {'ca': {'en', 'fr'}, 'uk': {'en'}, 'in': {'en'}, 'fr': {'fr'}, 'de': {'de'}, 'mx': {'es'}, 'jp': {'ja'}}
CAP = {'ca': (45.42, -75.70), 'de': (52.52, 13.40), 'fr': (48.86, 2.35), 'in': (28.61, 77.21), 'jp': (35.68, 139.69), 'mx': (19.43, -99.13), 'uk': (51.51, -0.13)}


def geo(a, b):
    (la1, lo1), (la2, lo2) = [(math.radians(p[0]), math.radians(p[1])) for p in (CAP[a], CAP[b])]
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def figure9():
    # the 15 tests per data version on the ratio-free outcomes D = F - L (paper/code/rq4_ratiofree.py; Table 6), gain without IN A&C
    R = rd('rq4_ratiofree.csv')
    R = R[(R.in_ac == 'incl.') & R.kind.isin(['D', 'gain'])].copy()
    R['outcome'] = np.where(R.kind == 'gain', 'd', 'D_' + R.measure)
    OUT = [('D_sp', 'Popularity: rank\ncorrelation, D = F − L'), ('D_top10', 'Popularity: top-10\noverlap, D = F − L'),
           ('D_cos', 'Patterns: co-interaction\ncosine, D = F − L'), ('D_nbr5', 'Patterns: 5-NN\noverlap, D = F − L'),
           ('d', 'Single-donor ItemKNN\ngain (excl. IN A&C)')]
    DIS = ['Hofstede distance', 'Shared language', 'Geographic distance']
    VER = [('new', V_ORIG, ORIG, 'o'), ('_nvc', V_NVC, CLEAN, 's'), ('_nv', V_NV, MID, '^')]   # same order as Table 6
    fig = plt.figure(figsize=(FULL_W, 6.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.5, wspace=0.12, left=0.2, right=0.985, top=0.86, bottom=0.08)
    for j, dname in enumerate(DIS):
        ax = fig.add_subplot(gs[0, j])
        for i, (o, lab) in enumerate(OUT):
            for (src, vl, col, mk), off in zip(VER, [0.25, 0, -0.25]):
                r = R[(R.source == src) & (R.outcome == o) & (R.distance == dname)]
                if r.empty: continue
                r = r.iloc[0]; yy = 4 - i + off; sig = r.p < 0.05
                ax.plot([0, r.r], [yy, yy], color=col, lw=1, alpha=0.5)
                ax.plot(r.r, yy, mk, ms=6.5, mfc=col if sig else 'white', mec='white' if sig else col, mew=0.8 if sig else 1.4, zorder=4)
                if sig: ax.text(r.r - 0.08 if r.r < 0 else r.r + 0.08, yy, f'{r.p:.4f}' + ('*' if r.bonferroni else ''),
                                ha='right' if r.r < 0 else 'left', va='center', fontsize=6.6, color=INK)
        ax.axvline(0, color=INK2, lw=1); ax.set_xlim(-1.25, 0.95);   # room for p-value labels left of negative r ax.set_ylim(-0.5, 4.5); ax.grid(axis='y', visible=False)
        ax.set_yticks(range(5)[::-1]); ax.set_yticklabels([lab for _, lab in OUT] if j == 0 else [], fontsize=8)
        ax.set_title(dname, loc='left', fontsize=9.5, fontweight='normal', pad=4); ax.tick_params(axis='x', labelsize=8)
        if j == 1: ax.set_xlabel('Correlation r over the 42 ordered country pairs', fontsize=9)
    fig.text(0.2, 0.955, '(a) Country distance vs agreement and transfer: 15 tests per data version', fontsize=10.5, fontweight='bold')
    fig.legend(handles=[Line2D([], [], marker=mk, ls='', ms=6.5, mfc=c, mec='white', label=vl) for _, vl, c, mk in VER] +
               [Line2D([], [], marker='o', ls='', ms=6.5, mfc='white', mec=INK2, mew=1.4, label='hollow: permutation p ≥ 0.05'),
                Line2D([], [], ls='', label='filled: p shown; * below Bonferroni 0.05/15')],
               loc='upper left', bbox_to_anchor=(0.195, 0.948), ncol=3, fontsize=7.3, handletextpad=0.3, columnspacing=1.0)
    # (b), (c): single-donor gain per ordered country pair (original data, centred within target context, pair-averaged)
    P = rd('aud_rq4_pairs.csv', RAW); P = P[P.outcome == 'd_excl'].copy()
    P['shared_language'] = [float(bool(LANG[a] & LANG[b])) for a, b in zip(P.target_market, P.donor_market)]
    P['capital_km'] = [geo(a, b) for a, b in zip(P.target_market, P.donor_market)]
    rr = R[(R.source == 'new') & (R.outcome == 'd')].set_index('distance')
    b = fig.add_subplot(gs[1, 0:1]); rng = np.random.default_rng(3)
    for v, lab in [(0, 'no'), (1, 'yes')]:
        s = P[P.shared_language == v]
        b.scatter(v + rng.uniform(-0.13, 0.13, len(s)), s.res, s=s.ncat * 4, color=ORIG if v else INK2, alpha=0.6, lw=0)
        b.plot([v - 0.28, v + 0.28], [s.res.median()] * 2, color=INK, lw=2)
    b.axhline(0, color=GRID, lw=1.2, zorder=0); b.set_xticks([0, 1])
    b.set_xticklabels([f'no\n({(P.shared_language == 0).sum()} pairs)', f'yes\n({(P.shared_language == 1).sum()} pairs)'], fontsize=8)
    b.set_xlim(-0.5, 1.5); b.grid(axis='x', visible=False); b.set_xlabel('Shared official language', fontsize=9)
    b.set_ylabel('Single-donor ItemKNN gain, original\ndata, centred within target context', fontsize=9); b.tick_params(axis='y', labelsize=8)
    lr = rr.loc['Shared language']
    b.set_title(f'(b) By shared language\n(r = {lr.r:+.2f}, p = {lr.p:.2f})', loc='left', fontsize=9.5, pad=4)
    c = fig.add_subplot(gs[1, 1:], sharey=b); plt.setp(c.get_yticklabels(), visible=False)
    c.scatter(P.capital_km / 1000, P.res, s=P.ncat * 4, color=ORIG, alpha=0.6, lw=0)
    gr = rr.loc['Geographic distance']
    c.axhline(0, color=GRID, lw=1.2, zorder=0); c.set_xlabel('Distance between capitals (1,000 km)', fontsize=9); c.tick_params(labelsize=8)
    c.set_title(f'(c) By capital distance\n(r = {gr.r:+.2f}, p = {gr.p:.2f}; point size: shared categories)', loc='left', fontsize=9.5, pad=4)
    for n in (2, 6, 12):
        c.scatter([], [], s=n * 4, color=ORIG, alpha=0.6, lw=0, label=f'{n}')
    c.legend(title='shared categories', title_fontsize=7.5, fontsize=7.5, loc='lower right', ncol=3, handletextpad=0.2, columnspacing=0.6)
    fsave(fig, 9)
    out_csv(pd.concat([R.assign(panel='a', source_file='paper/results/rq4_ratiofree.csv'),
                       P.assign(panel='b,c', source_file='paper/raw/aud_rq4_pairs.csv')], ignore_index=True), 9)


FIGS = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6, 7: figure7, 8: figure8, 9: figure9}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', default=''); ap.add_argument('--copy-to', default='')
    a = ap.parse_args()
    todo = [int(x) for x in a.only.split(',') if x] or list(FIGS)
    for n in todo:
        FIGS[n](); print(f'figure{n}: paper/figures/figure{n}.pdf/.png, data paper/figures/data/figure{n}.csv', flush=True)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        for n in todo: shutil.copy2(os.path.join(FIG, f'figure{n}.pdf'), os.path.join(a.copy_to, f'figure{n}.pdf'))
        print('PDFs copied to', a.copy_to)
