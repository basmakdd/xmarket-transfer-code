"""sens2 extra: launch, patterns vs popularity head-to-head (needed because on _nvc foreign MostPop is also above chance).

Per context (launch3, random split, zf tuning, 3 launch sets averaged; sens2_tables.launch_load), the paired difference
of launched-item NDCG@10 with foreign data: ItemKNN - MostPop and EASE - MostPop (contexts where both models ran).
Median with the context-level two-way cluster bootstrap (boot.summarize via sens2_tables.csum), share of contexts where the
pattern model is higher by > 0.01. Output: paper/results/sens2_launch_paired.csv and printed markdown.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO
import numpy as np, pandas as pd
import sens2_tables as T

rows = []
for v in T.VAR:
    C = T.launch_load(v)
    if C is None: continue
    W = C.pivot_table(index='t', columns='model', values='L_F')
    for m in ('ItemKNN', 'EASE'):
        w = W[[m, 'MostPop']].dropna()
        d = (w[m] - w.MostPop).to_numpy()
        med = T.csum(d, w.index.tolist(), f'sens2|l3pair|{m}-MostPop|{v}')
        rows.append(dict(data=v, comparison=f'{m} - MostPop', contexts=len(w), median=med[0], lo=med[1], hi=med[2],
                         higher=int((d > 0.01).sum()), lower=int((d < -0.01).sum())))
R = pd.DataFrame(rows); R.to_csv(f'{REPO}/paper/results/sens2_launch_paired.csv', index=False, lineterminator='\n')
print('| Comparison | Data | Contexts | Median difference, launched-item NDCG@10 with foreign data [CI] | pattern model higher by > 0.01 | lower by > 0.01 |')
print('|---|---|---|---|---|---|')
for _, r in R.sort_values(['comparison', 'data'], key=lambda s: s.map({v: i for i, v in enumerate(T.VAR)}) if s.name == 'data' else s).iterrows():
    print(f'| {r.comparison} | {r.data} | {r.contexts} | {r["median"]:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.higher} | {r.lower} |')
