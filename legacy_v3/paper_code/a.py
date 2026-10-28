import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import json,pandas as pd,numpy as np
import os
base=os.path.expanduser(REPO+'/results/')
recs=[json.loads(l) for l in open(base+'full_random.jsonl', encoding='utf-8') if l.strip()]
df=pd.DataFrame(recs); df=df[df.error.isna()]
df['t']=df.target_market+' '+df.target_category
b=df[df.k==0].set_index(['model','t'])['ndcg@10']
df['base']=[b.get((m,t),np.nan) for m,t in zip(df.model,df.t)]
df['d']=df['ndcg@10']-df.base
df.to_pickle(os.path.expanduser(PAPER_RAW+'/df.pkl'))
full=df[(df.order=='full')&(df.frac==1.0)]
print(full.groupby(['channel','model']).apply(lambda g: pd.Series({'n':len(g),'base_med':g.base.median(),'d_med':g.d.median(),'d_mean':g.d.mean(),'rel_med%':100*(g.d/g.base.replace(0,np.nan)).median()})).round(4))
