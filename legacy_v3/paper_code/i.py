import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE
import pandas as pd,numpy as np,os,itertools,math
H=os.path.expanduser
x=pd.read_pickle(H(PAPER_RAW+'/pairs.pkl')).dropna(subset=['d'])
x=x[~((x.t=='in Arts_Crafts_and_Sewing'))].dropna(subset=['popularity_js'])
hof={'ca':[39,80,52,48,36,68],'de':[35,67,66,65,83,40],'fr':[68,71,43,86,63,48],'in':[77,48,56,40,51,26],'jp':[54,46,95,92,88,42],'mx':[81,30,69,82,24,97],'uk':[35,89,66,35,51,69]}
dims=['PDI','IDV','MAS','UAI','LTO','IVR']
M=np.array(list(hof.values()),float); var=M.var(axis=0,ddof=1)
def ks(a,b): return float(np.mean((np.array(hof[a])-np.array(hof[b]))**2/var))
lang={'ca':{'en','fr'},'uk':{'en'},'in':{'en'},'fr':{'fr'},'de':{'de'},'mx':{'es'},'jp':{'ja'}}
cap={'ca':(45.42,-75.70),'de':(52.52,13.40),'fr':(48.86,2.35),'in':(28.61,77.21),'jp':(35.68,139.69),'mx':(19.43,-99.13),'uk':(51.51,-0.13)}
def geo(a,b):
    (la1,lo1),(la2,lo2)=[(math.radians(p[0]),math.radians(p[1])) for p in (cap[a],cap[b])]
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 6371*2*math.asin(math.sqrt(h))
x['KS']=[ks(a,b) for a,b in zip(x.target_market,x.donor_market)]
for i,dn in enumerate(dims): x['|'+dn+'|']=[abs(hof[a][i]-hof[b][i]) for a,b in zip(x.target_market,x.donor_market)]
x['same_lang']=[int(bool(lang[a]&lang[b])) for a,b in zip(x.target_market,x.donor_market)]
x['geo_km']=[geo(a,b) for a,b in zip(x.target_market,x.donor_market)]
x['log_target']=np.log(x.target_train); x['log_ratio']=np.log(x.donor_train/x.target_train)
preds=['log_target','shared_item_interaction_cover','item_overlap_target','popularity_js','log_ratio','KS','same_lang','geo_km']+['|'+d+'|' for d in dims]
print('n pairs',len(x))
r=x[preds+['d']].corr('spearman')['d'].drop('d').round(3); print(r.to_string())
# partial: residualize ranks of d on data-side predictors
from numpy.linalg import lstsq
rk=x[preds+['d']].rank()
Xc=np.column_stack([np.ones(len(x)),rk[['log_target','shared_item_interaction_cover','popularity_js','log_ratio']].values])
res=rk.d.values-Xc@lstsq(Xc,rk.d.values,rcond=None)[0]
x['res']=res
print('R2 of data-side predictors on rank(d): %.3f'%(1-res.var()/rk.d.var()))
for p in ['KS','same_lang','geo_km']+['|'+d+'|' for d in dims]:
    print('partial rank corr',p,round(np.corrcoef(res,rk[p])[0,1],3))
# exact permutation of market labels on market-pair mean residual
pm=x.groupby(['target_market','donor_market']).res.mean().reset_index()
mk=sorted(hof)
def stat(perm,fn):
    mp=dict(zip(mk,perm)); v=np.array([fn(mp[a],mp[b]) for a,b in zip(pm.target_market,pm.donor_market)])
    return np.corrcoef(v,pm.res)[0,1]
for name,fn in [('KS',ks),('geo',geo),('lang',lambda a,b:float(bool(lang[a]&lang[b])))]:
    obs=stat(mk,fn); null=np.array([stat(p,fn) for p in itertools.permutations(mk)])
    print(f'{name}: market-pair corr {obs:.3f}, exact perm p(neg side)={np.mean(null<=obs):.3f}, p(two-sided)={np.mean(np.abs(null)>=abs(obs)):.3f}')
# donor market helpfulness overall & target market
print(x.groupby('donor_market').d.agg(['count','median','mean']).round(4).sort_values('mean'))
print(x.groupby('target_market').d.agg(['count','median','mean']).round(4).sort_values('mean'))
x.to_pickle(H(PAPER_RAW+'/pairs2.pkl'))
