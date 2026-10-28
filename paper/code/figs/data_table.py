"""Merge raw counts (data_stats.py) with the pipeline's contexts.parquet and check they agree."""
import json, os, pandas as pd
P = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'data')
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
raw = json.load(open(f'{P}/data_distribution_partial.json', encoding='utf-8'))
rows = []
for k, v in raw.items():
    mk, cat = k.split('|'); r = dict(market=mk, category=cat, present=v is not None and v.get('raw', 0) > 0)
    if v: r.update(v)
    rows.append(r)
R = pd.DataFrame(rows)
C = pd.read_parquet(f'{REPO}/results/contexts.parquet')[['market','category','core','users','items','interactions']]
M = R.merge(C, on=['market','category'], how='left')
kept = M.interactions.notna()
chk = {
 'contexts kept (pipeline)': int(kept.sum()),
 'contexts re-derived by k-core': int(M.rec_core.notna().sum()),
 'kept sets identical': bool((M.rec_core.notna() == kept).all()),
 'interactions match': bool((M.loc[kept,'rec_after'] == M.loc[kept,'interactions']).all()),
 'users match': bool((M.loc[kept,'rec_users'] == M.loc[kept,'users']).all()),
 'items match': bool((M.loc[kept,'rec_items'] == M.loc[kept,'items']).all()),
 'core match': bool((M.loc[kept,'rec_core'] == M.loc[kept,'core']).all()),
 'raw files present': int(M.present.sum()),
 'raw interactions total': int(M.raw.fillna(0).sum()),
 'after dedup total': int(M.dedup.fillna(0).sum()),
 'after preprocessing total': int(M.interactions.fillna(0).sum()),
 'core levels': M.loc[kept,'core'].value_counts().sort_index().to_dict(),
}
for k, v in chk.items(): print(f'{k}: {v}')
M.drop(columns=[c for c in M if c.startswith('rec_')]).rename(columns={'raw':'raw_interactions','dedup':'interactions_dedup','interactions':'interactions_after','users':'users_after','items':'items_after'}).to_csv(f'{P}/data_distribution.csv', index=False, lineterminator='\n')
json.dump({k: (v if not isinstance(v, dict) else {int(a): int(b) for a, b in v.items()}) for k, v in chk.items()}, open(f'{P}/data_distribution_checks.json', 'w', encoding='utf-8', newline='\n'), indent=1)
