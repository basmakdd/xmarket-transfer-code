"""Per-context diagnostics of product variants (appendix table), on the retained data (all splits).

  same_date_share : share of a context's retained interactions whose (user, date) occurs at least twice
                    for that user in the context (one review attached to several products shows up this way).
  jac_mass        : co-occurrence mass on near-identical products: with c_ij = number of the context's users
                    interacting with both products i and j (i < j) and J_ij = c_ij / (n_i + n_j - c_ij),
                    jac_mass = sum_{J_ij >= 0.8} c_ij / sum c_ij.
  jac_pairs_share : share of co-occurring product pairs (c_ij >= 1) with J_ij >= 0.8.
Thresholds fixed in advance (Jaccard 0.8, as for the merge rule in src/data.py).

Usage: python variant_diagnostics.py [TAG ...]   ('' = paper data; e.g. _nv _xd _nvxd)
Outputs: paper/results/sens_diag{TAG}.csv ('' -> sens_diag_orig.csv) and, for the paper data, paper/tables_sens_diag.tex.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys
import numpy as np, pandas as pd, scipy.sparse as sp

NOTES = f'{REPO}/paper/results'
JAC = 0.8


def diag(g):
    n = len(g)
    ud = g.groupby(['u', 'date']).size()
    same = float((ud[ud > 1]).sum() / n)
    uc = pd.factorize(g.u)[0]; ic = pd.factorize(g.i)[0]
    R = sp.csr_matrix((np.ones(n, np.float32), (uc, ic)))
    deg = np.asarray(R.sum(0)).ravel()
    C = sp.triu((R.T @ R).tocsr(), k=1).tocoo()
    c = C.data.astype(np.float64)
    J = c / (deg[C.row] + deg[C.col] - c)
    hi = J >= JAC - 1e-12
    return dict(interactions=n, users=int(uc.max() + 1), products=int(ic.max() + 1), same_date_share=same,
                cooc_pairs=int(len(c)), jac_pairs_share=float(hi.mean()) if len(c) else np.nan,
                jac_mass=float(c[hi].sum() / c.sum()) if len(c) else np.nan)


def run(tag):
    d = pd.read_parquet(f'{DATA}/interactions{tag}.parquet', columns=['market', 'category', 'u', 'i', 'date'])
    rows = []
    for (m, c), g in d.groupby(['market', 'category'], sort=True):
        rows.append(dict(market=m, category=c, **diag(g)))
    X = pd.DataFrame(rows)
    X.to_csv(f'{NOTES}/sens_diag{tag or "_orig"}.csv', index=False, lineterminator='\n')
    print(f'== {tag or "paper data"}: {len(X)} contexts ==')
    for col in ['same_date_share', 'jac_mass', 'jac_pairs_share']:
        v = X[col]
        print(f'{col}: median {v.median():.3f}, quartiles {v.quantile(.25):.3f}-{v.quantile(.75):.3f}, min {v.min():.3f}, max {v.max():.3f}; '
              f'contexts >= 0.33: {(v >= 0.33).sum()}')
    pooled = (X.same_date_share * X.interactions).sum() / X.interactions.sum()
    print(f'same_date_share pooled over all interactions: {pooled:.3f}')
    print(X.groupby('category')[['same_date_share', 'jac_mass']].median().round(3).to_string())
    return X


def tex(X):
    L = [r'\begin{longtable}{llrrrr}', r'\caption{Product-variant diagnostics per context (retained data, all splits).}\label{tab:variant_diag}\\',
         r'\toprule', r'Category & Country & Interactions & Same user--date & Jaccard$\ge$0.8 pairs & Co-occurrence mass, Jaccard$\ge$0.8 \\', r'\midrule\endhead']
    for _, r in X.sort_values(['category', 'market']).iterrows():
        L.append(f"{r.category.replace('_and_', ' & ').replace('_', ' ').replace('&', chr(92) + '&')} & {r.market.upper()} & {r.interactions:,} & "
                 f"{100 * r.same_date_share:.1f}\\% & {100 * r.jac_pairs_share:.1f}\\% & {100 * r.jac_mass:.1f}\\% \\\\")
    L += [r'\bottomrule',
          r'\multicolumn{6}{p{0.95\linewidth}}{\footnotesize Same user--date: share of the context\textquotesingle s interactions whose user has another interaction '
          r'on the same date in the context. Jaccard of two products = shared users / users of either product (within the context). Co-occurrence mass: '
          r'share of all co-occurrences (user pairs of products) that fall on product pairs with Jaccard $\ge 0.8$.}', r'\end{longtable}']
    open(f'{REPO}/paper/tables_sens_diag.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    tags = sys.argv[1:] or ['']
    for t in tags:
        X = run('' if t in ('', 'orig') else t)
        if t in ('', 'orig'):
            tex(X)
