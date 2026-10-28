"""One shared look for every figure in the paper (palette of Fig. 3: blue, violet, green, light and dark grey)."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, SURF = '#1f1f1e', '#52514e', '#8a8985', '#e4e3df', '#ffffff'
BLUE, AQUA, VIOLET = '#2a78d6', '#1baf7a', '#4a3aa7'
LOCAL = '#b4b2ab'            # "local data only" everywhere
DARK = INK2                  # dark grey: popularity / MostPop / 'worse'
POP, PAT = DARK, BLUE        # popularity vs shopping patterns
MODEL_C = {'ItemKNN': BLUE, 'EASE': VIOLET, 'UserKNN': AQUA, 'MostPop': DARK, 'Chance': MUTED}
MODEL_M = {'ItemKNN': 'o', 'EASE': 's', 'UserKNN': '^', 'MostPop': 'D', 'Chance': 'x'}
SEQ = ['#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # blue sequential ramp
FULL_W = 7.2   # inches, two-column width

def setup():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
        'axes.axisbelow': True, 'legend.frameon': False,
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'lines.linewidth': 1.6,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    })

def panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', pad=8)

def save(fig, name, outdir):
    import os
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'{name}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
