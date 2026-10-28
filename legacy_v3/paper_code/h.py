import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import pandas as pd,numpy as np,os,json,itertools
H=os.path.expanduser
df=pd.read_pickle(H(PAPER_RAW+'/df.pkl'))
o=pd.read_parquet(H(REPO+'/results/overlap_random.parquet'))
c=pd.read_parquet(H(REPO+'/results/contexts.parquet'))
x=o[o.channel=='XM'].copy(); x['t']=x.target_market+' '+x.target_category; x['donor']=list(zip(x.donor_market,x.donor_category))
s=df[(df.channel=='XM')&(df.k==1)&(df.frac==1)&(df.model=='ItemKNN')].copy()
s['donor']=s.donors.apply(lambda d:tuple(d[0]))
s=s.drop_duplicates(['t','donor'])[['t','donor','ndcg@10']]
new=[json.loads(l) for l in open(H(PAPER_RAW+'/single.jsonl'), encoding='utf-8')]
chk=new[:3]
for r in chk:
    old=s[(s.t==r['t'])&(s.donor==tuple(r['donor']))]['ndcg@10'].values
    print('repro check',r['t'],r['donor'],old,r['ndcg@10'])
n=pd.DataFrame([{'t':r['t'],'donor':tuple(r['donor']),'ndcg@10':r['ndcg@10']} for r in new[3:]])
s=pd.concat([s,n]).drop_duplicates(['t','donor'])
base=df[(df.k==0)&(df.model=='ItemKNN')].set_index('t')['ndcg@10']
x=x.merge(s,on=['t','donor'],how='left'); x['own']=x.t.map(base); x['d']=x['ndcg@10']-x.own
print('pairs with results',x.d.notna().sum(),'of',len(x))
x.to_pickle(H(PAPER_RAW+'/pairs.pkl'))
size=c.assign(t=c.market+' '+c.category).set_index('t')['n_random_train']
out=[]
for cat,g in x.groupby('target_category'):
    mk=sorted(set(g.target_market)|set(g.donor_market),key=lambda m:size.get(f'{m} {cat}',1e12))
    tab=g.pivot_table(index='target_market',columns='donor_market',values='d')
    tab=tab.reindex(index=[m for m in mk if m in tab.index],columns=mk)
    own=g.groupby('target_market').own.first()
    lines=[f'### {cat}',"",'| target (train size, own score) | '+' | '.join(mk)+' |','|---|'+'---|'*len(mk)]
    for m in tab.index:
        cells=[]
        for dmk in mk:
            v=tab.loc[m,dmk]
            if dmk==m: cells.append('·')
            elif pd.isna(v): cells.append('n/a' if f'{dmk} {cat}' in size.index else '')
            else:
                t=f'{v:+.3f}'; cells.append(f'**{t}**' if v>=0.01 else t)
        lines.append(f'| {m} ({int(size[f"{m} {cat}"]):,}, {own[m]:.2f}) | '+' | '.join(cells)+' |')
    out.append('\n'.join(lines))
open(H(PAPER_RAW+'/tables.md'),'w', encoding='utf-8').write('\n\n'.join(out))
print(x.groupby('target_category').d.agg(['count','median','mean','max']).sort_values('mean',ascending=False).round(3))
