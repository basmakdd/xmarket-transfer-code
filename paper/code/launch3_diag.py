"""Diagnostic for the calendar launch result: foreign coverage of launched products and local profile sizes of launched-item test users. Usage: launch3_diag.py "uk Electronics"."""
import sys,os
import os; REPO=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
sys.path.insert(0,REPO+'/src')
import numpy as np,pandas as pd, data as D, experiment as E, models as Mod, calendar_split as CS
T=('ca','Books') if len(sys.argv)<2 else tuple(sys.argv[1].split())
data=D.load_for('calendar'); blocks=E.load_blocks(data,'calendar'); lc=CS.load(launch=True)
o=pd.read_parquet(REPO+'/results/overlap_calendar.parquet'); o=o[o.channel=='XM']
donors=[tuple(x) for x in o[(o.target_market==T[0])&(o.target_category==T[1])][['donor_market','donor_category']].values]
b=blocks[T]; g=lc[(lc.market==T[0])&(lc.category==T[1])]
abroad=np.unique(np.concatenate([blocks[d].train_i for d in donors]))
L=np.intersect1d(g.i.unique(),abroad); cat=np.union1d(b.catalog,L)
print('donors',donors,'local items',len(b.catalog),'launched',len(L))
el=[(b.train_u,b.train_i)]+[(blocks[d].train_u,blocks[d].train_i) for d in donors]
R,cols,_=E.build_matrix(el,cat,np.array([],np.int32))
Rt=R[:,cols]; co=(Rt.T@Rt).tocsr()
isL=np.isin(cat,L)
cntL=np.asarray(Rt[:,isL].sum(0)).ravel(); print('foreign interactions per launched item: median',np.median(cntL),'zero',(cntL==0).sum())
# co-occurrence of launched items with local items
sub=co[isL][:,~isL]; nn=np.diff(sub.tocsr().indptr); print('launched items with any local-item co-occurrence:',(nn>0).mean(),'median #local neighbours',np.median(nn))
# local items popular in foreign? share of local catalog present in donors
print('local items present abroad',np.isin(b.catalog,abroad).mean())
gt=g[(g.split_calendar=='test')&g.i.isin(L)]
# test launched users' local profile size
prof=pd.Series(b.train_u).value_counts()
pu=prof.reindex(gt.u.unique()).fillna(0); print('launch test users profile size median',pu.median(), 'share with profile items abroad:')
tr=pd.DataFrame({'u':b.train_u,'i':b.train_i}); tr['abroad']=np.isin(tr.i,abroad)
print(tr[tr.u.isin(gt.u.unique())].groupby('u').abroad.max().mean())
