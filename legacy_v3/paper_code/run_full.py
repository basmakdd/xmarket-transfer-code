"""Complete local data, with vs without all same-category foreign data, re-tuned per arm (regime F)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys,os,json,time
sys.path.insert(0,os.path.expanduser(SRC)); sys.path.insert(0,os.path.expanduser(CODE))
import pandas as pd,numpy as np
import data as D, experiment as E
import mustadd as MA
MA.GRIDS['MostPop']=[{}]
LIM=float(sys.argv[1]); MODEL=sys.argv[2]; SPLIT=sys.argv[3]; W=int(sys.argv[4]); NW=int(sys.argv[5]); MAXCOST=float(sys.argv[6])
out=os.path.expanduser(f'{PAPER_RAW}/fr_{MODEL}_{SPLIT}_{W}.jsonl'); T0=time.time(); done=set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'): r=json.loads(l); done.add((r['t'],r['foreign']))
o=pd.read_parquet(os.path.expanduser(f'{REPO}/results/overlap_{SPLIT}.parquet')); o=o[o.channel=='XM']
c=pd.read_parquet(os.path.expanduser(REPO+'/results/contexts.parquet'))
cost={(m,cc):int(u)*int(i) for m,cc,u,i in zip(c['market'],c['category'],c['users'],c['items'])}
data=D.load_for(SPLIT); blocks=E.load_blocks(data,SPLIT)
targets=sorted(set(zip(o.target_market,o.target_category)),key=lambda t:cost.get(t,0))
targets=[t for t in targets if cost.get(t,0)<=MAXCOST][W::NW]
if _os.environ.get('ONLY'): targets=[t for t in targets if ' '.join(t) in _os.environ['ONLY'].split(',')]
for target in targets:
    t=' '.join(target); b=blocks[target]
    if MODEL=='EASE' and len(b.catalog)>4000: continue
    donors=[tuple(x) for x in o[(o.target_market==target[0])&(o.target_category==target[1])][['donor_market','donor_category']].values]
    s=vs=None
    for foreign in (0,1):
        if (t,foreign) in done: continue
        if time.time()-T0>LIM: print('deadline',flush=True); sys.exit()
        if s is None: s=E.TargetSession(target,b); vs=s.validation_view()
        el=[(b.train_u,b.train_i)]+([(blocks[d].train_u,blocks[d].train_i) for d in donors] if foreign else [])
        keep=np.ones(len(b.train_u),bool)
        sc,bp=MA.run_unit(MODEL,s,vs,el,keep,'F')
        open(out,'a', encoding='utf-8').write(json.dumps({'t':t,'split':SPLIT,'model':MODEL,'foreign':foreign,'ndcg':sc,'params':bp,'tuned':bool(len(vs.eval_users))})+'\n')
print('ALL DONE',flush=True)
