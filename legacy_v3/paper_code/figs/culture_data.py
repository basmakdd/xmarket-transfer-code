"""Pair-level data behind tab:culture, rebuilt with the exact steps of final_tables.py (Table 7 block).
Writes ../../figures/data/culture_pairs.csv and checks every r in tables.tex is reproduced."""
import os, glob, math, re, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'raw')
LP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper')   # legacy_v3/paper
X = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f'{R}/travel3_*.pkl'))]).drop_duplicates(['t', 'donor_market']).sort_values(['t', 'donor_market']).reset_index(drop=True)
hof={'ca':[39,80,52,48,36,68],'de':[35,67,66,65,83,40],'fr':[68,71,43,86,63,48],'in':[77,48,56,40,51,26],'jp':[54,46,95,92,88,42],'mx':[81,30,69,82,24,97],'uk':[35,89,66,35,51,69]}
Mh=np.array(list(hof.values()),float); var=Mh.var(axis=0,ddof=1)
def ks(a,b): return float(np.mean((np.array(hof[a])-np.array(hof[b]))**2/var))
lang={'ca':{'en','fr'},'uk':{'en'},'in':{'en'},'fr':{'fr'},'de':{'de'},'mx':{'es'},'jp':{'ja'}}
def sl(a,b): return float(bool(lang[a]&lang[b]))
cap={'ca':(45.42,-75.70),'de':(52.52,13.40),'fr':(48.86,2.35),'in':(28.61,77.21),'jp':(35.68,139.69),'mx':(19.43,-99.13),'uk':(51.51,-0.13)}
def geo(a,b):
    (la1,lo1),(la2,lo2)=[(math.radians(p[0]),math.radians(p[1])) for p in (cap[a],cap[b])]
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2; return 6371*2*math.asin(math.sqrt(h))
P2=pd.read_pickle(f'{R}/pairs2.pkl')[['t','donor_market','d']]
Y=X.merge(P2,on=['t','donor_market'],how='left')
for m,col in [('sp','pop_rel'),('top10','pop10_rel'),('cos','pat_rel'),('nbr5','nbr_rel')]:
    Y[col]=(Y['for_'+m]/Y['loc_'+m]).where(Y['loc_'+m]>0)
out=None; tex=open(os.path.join(LP,'tables.tex'), encoding='utf-8').read()
block=tex[tex.index('label{tab:culture}'):]; block=block[:block.index('bottomrule')]
nok=0
for col,lab in [('pop_rel','Popularity agreement (rank correlation)'),('pop10_rel','Popularity agreement (top-10 overlap)'),('pat_rel','Pattern agreement (cosine)'),('nbr_rel','Pattern agreement (neighbour overlap)'),('d','Gain from one foreign donor (ItemKNN)')]:
    z=Y.dropna(subset=[col]).copy(); z['res']=z[col]-z.groupby('t')[col].transform('mean')
    pm=z.groupby(['target_market','donor_market']).res.mean().rename(col).reset_index()
    out=pm if out is None else out.merge(pm,on=['target_market','donor_market'],how='outer')
    line=[l for l in block.split('\n') if l.startswith(lab)][0]
    want=[float(v) for v in re.findall(r'([+-]\d\.\d\d) \(',line)]
    got=[np.corrcoef([fn(a,b) for a,b in zip(pm.target_market,pm.donor_market)],pm[col])[0,1] for fn in (ks,sl,geo)]
    ok=all(abs(g-w)<0.006 for g,w in zip(got,want)); nok+=ok
    print(f'{lab}: n={len(z)} pairs={len(pm)} r={[round(g,2) for g in got]} table={want} {"OK" if ok else "MISMATCH"}')
out['hofstede']=[ks(a,b) for a,b in zip(out.target_market,out.donor_market)]
out['shared_language']=[sl(a,b) for a,b in zip(out.target_market,out.donor_market)]
out['geo_km']=[geo(a,b) for a,b in zip(out.target_market,out.donor_market)]
out.to_csv(os.path.join(LP,'figures','data','culture_pairs.csv'),index=False)
print('rows reproduced OK:',nok,'/ 5')
