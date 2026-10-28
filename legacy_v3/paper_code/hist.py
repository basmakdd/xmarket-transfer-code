import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import sys,os,json
sys.path.insert(0,os.path.expanduser(SRC))
import pandas as pd,numpy as np
import data as D, experiment as E, design as Design, runner as Rn
df=pd.read_pickle(os.path.expanduser(PAPER_RAW+'/df.pkl'))
out=[]
data=D.load(); blocks=E.load_blocks(data,'random')
base=os.path.expanduser(REPO+'/results/full_random.jsonl')
P=os.path.expanduser(REPO+'/results/full_random_peruser')
def path(rec):
    donors=tuple(tuple(x) for x in rec['donors'])
    u=Design.Unit('random',rec['model'],(rec['target_market'],rec['target_category']),rec['channel'],rec['order'],rec['rep'],rec['k'],rec['frac'],donors)
    return Rn.per_user_path(Rn.Path(base),u) if hasattr(Rn,'Path') else Rn.per_user_path(__import__('pathlib').Path(base),u)
for model in ['ItemKNN','EASE']:
    k0=df[(df.model==model)&(df.k==0)&df.error.isna()]
    xm=df[(df.model==model)&(df.channel=='XM')&(df.order=='full')&(df.frac==1)&df.error.isna()].set_index('t')
    for _,r0 in k0.iterrows():
        t=r0.t
        if t not in xm.index: continue
        r1=xm.loc[t]
        p0,p1=path(r0.to_dict()),path(r1.to_dict())
        if not (os.path.exists(p0) and os.path.exists(p1)): continue
        target=(r0.target_market,r0.target_category); b=blocks[target]
        users=E.TargetSession.__new__(E.TargetSession)  # avoid heavy init: recompute eval users cheaply
        te=np.unique(b.test_u)
        if len(te)>E.MAX_EVAL_USERS: 
            s=E.TargetSession(target,b); te=s.eval_users
        cnt=pd.Series(b.train_u).value_counts()
        L=cnt.reindex(te).fillna(0).values
        a0=np.load(p0)['ndcg@10']; a1=np.load(p1)['ndcg@10']
        if len(a0)!=len(te): print('mismatch',t); continue
        for l,x0,x1 in zip(L,a0,a1): out.append((model,t,int(l),float(x0),float(x1)))
o=pd.DataFrame(out,columns=['model','t','hist','y0','y1'])
o.to_pickle(os.path.expanduser(PAPER_RAW+'/hist.pkl')); print(len(o), o.t.nunique())
