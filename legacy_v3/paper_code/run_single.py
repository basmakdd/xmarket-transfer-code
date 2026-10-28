import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import sys,os,json,time
sys.path.insert(0,os.path.expanduser(SRC))
import pandas as pd, numpy as np
import data as D, experiment as E, runner as Rn, design as Design
T0=time.time(); LIM=float(sys.argv[1]) if len(sys.argv)>1 else 150
skip_big = '--all' not in sys.argv
out=os.path.expanduser(PAPER_RAW+'/single.jsonl')
done=set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'):
        r=json.loads(l); done.add((r['t'],tuple(r['donor'])))
miss=pd.read_csv(os.path.expanduser(PAPER_RAW+'/miss.csv'))
# add 3 already-recorded pairs as a reproducibility check
chk=[('de','Grocery_and_Gourmet_Food','uk','Grocery_and_Gourmet_Food'),('in','Arts_Crafts_and_Sewing','uk','Arts_Crafts_and_Sewing'),('mx','Electronics','uk','Electronics')]
rows=[tuple(r) for r in chk]+[tuple(r) for r in miss.itertuples(index=False)]
df=pd.read_pickle(os.path.expanduser(PAPER_RAW+'/df.pkl'))
bp={(t):p for t,p in zip(df[(df.k==0)&(df.model=='ItemKNN')].t,df[(df.k==0)&(df.model=='ItemKNN')].params)}
data=D.load(); blocks=E.load_blocks(data,'random')
print('loaded',round(time.time()-T0,1),flush=True)
sess={}
for tm,tc,dm,dc in rows:
    t=f'{tm} {tc}'; donor=(dm,dc)
    if (t,donor) in done: continue
    if skip_big and t=='uk Books': continue
    if time.time()-T0>LIM: print('deadline'); break
    target=(tm,tc)
    if target not in sess:
        s=E.TargetSession(target,blocks[target]); cap,dense=E.target_capacity(s,blocks[target]); sess={target:(s,cap,dense)}
    s,cap,dense=sess[target]
    unit=Design.Unit('random','ItemKNN',target,'XM','single',0,1,1.0,(donor,))
    el=Rn.unit_edge_lists(unit,blocks)
    t1=time.time()
    res,_,err=E.fit_and_select('ItemKNN',s,el,capacity=cap,dense=dense,base_params=bp.get(t))
    rec={'t':t,'donor':list(donor),'ndcg@10':res['ndcg@10'] if res else None,'frozen':res.get('frozen_ndcg@10') if res else None,'err':err,'sec':round(time.time()-t1,1)}
    open(out,'a', encoding='utf-8').write(json.dumps(rec)+'\n'); print(rec,flush=True)
