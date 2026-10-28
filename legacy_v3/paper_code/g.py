import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import pandas as pd,os
df=pd.read_pickle(os.path.expanduser(PAPER_RAW+'/df.pkl'))
o=pd.read_parquet(os.path.expanduser(REPO+'/results/overlap_random.parquet'))
x=o[o.channel=='XM'].copy(); x['t']=x.target_market+' '+x.target_category; x['donor']=list(zip(x.donor_market,x.donor_category))
s=df[(df.channel=='XM')&(df.k==1)&(df.frac==1)&(df.model=='ItemKNN')]
have=set(zip(s.t,s.donors.apply(lambda d:tuple(d[0]))))
miss=x[[ (t,d) not in have for t,d in zip(x.t,x.donor)]]
print(len(miss)); print(miss.groupby('t').size().sort_values().to_string())
miss[['target_market','target_category','donor_market','donor_category']].to_csv(os.path.expanduser(PAPER_RAW+'/miss.csv'),index=False)
