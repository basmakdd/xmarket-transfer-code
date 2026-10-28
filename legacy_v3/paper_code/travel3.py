"""Fair benchmark. Split local shoppers into halves A and B. Reference = half A.
Local benchmark = agree(A, B). Foreign = agree(A, foreign). Volume-matched foreign = agree(A, random foreign
shoppers subsampled until they hold as many purchases of shared products as half B). 3 splits, averaged."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import pandas as pd,numpy as np,os,sys,zlib
import scipy.sparse as sp
from scipy.stats import spearmanr
H=os.path.expanduser
d=pd.read_parquet(H(REPO+'/data/interactions.parquet'),columns=['market','category','u','i'])
o=pd.read_parquet(H(REPO+'/results/overlap_random.parquet')); o=o[o.channel=='XM']
G={k:g for k,g in d.groupby(['market','category'])}
def stats(g,items):
    """Sparse: never builds an item x item dense matrix, so every pair fits in memory.
    Gives the same numbers as the earlier dense version (checked to 1e-15 on 33 pairs)."""
    g=g[g.i.isin(items)]
    if g.empty: return None
    uu=pd.factorize(g.u)[0]; ii=np.searchsorted(items,g.i.values)
    R=sp.csr_matrix((np.ones(len(g)),(uu,ii)),shape=(uu.max()+1,len(items)))
    C=(R.T@R).tocsr(); pop=C.diagonal().astype(float); C.setdiag(0); C.eliminate_zeros()   # co-purchase counts, no self-pairs
    d=sp.diags(1/np.maximum(np.sqrt(pop),1e-12))
    S=sp.triu(d@C@d,k=1).tocsr()                                                        # cosine similarities, upper triangle
    return dict(pop=pop,S=S,C=C)
def _top5(C):
    out=[]
    for j in range(C.shape[0]):
        s,e=C.indptr[j],C.indptr[j+1]; idx=C.indices[s:e]; val=C.data[s:e]
        out.append(set(idx[np.lexsort((idx,-val))[:5]]))   # highest counts first; ties to the lower product index
    return out
def agree(a,b):
    if a is None or b is None: return dict(sp=np.nan,top10=np.nan,cos=np.nan,nbr5=np.nan)
    k=min(10,len(a['pop']))
    t10=len(set(np.argsort(-a['pop'],kind='stable')[:k])&set(np.argsort(-b['pop'],kind='stable')[:k]))/k
    aa=a['S'].multiply(a['S']).sum(); bb=b['S'].multiply(b['S']).sum()
    cs=float(a['S'].multiply(b['S']).sum()/np.sqrt(aa*bb)) if aa>0 and bb>0 else np.nan
    ok=(np.diff(a['C'].indptr)>=5)&(np.diff(b['C'].indptr)>=5)
    if ok.sum()>=5:
        ta,tb=_top5(a['C']),_top5(b['C']); nb=float(np.mean([len(ta[j]&tb[j])/5 for j in np.flatnonzero(ok)]))
    else: nb=np.nan
    return dict(sp=spearmanr(a['pop'],b['pop']).correlation,top10=t10,cos=cs,nbr5=nb)
rows=[]
W,NW=int(sys.argv[1]),int(sys.argv[2])
keys=sorted(o.groupby(['target_market','target_category']).groups)[W::NW]
if _os.environ.get('ONLY'): keys=[k for k in keys if ' '.join(k) in _os.environ['ONLY'].split(',')]
for (tm,tc) in keys:
    T=G[(tm,tc)]
    for dm in o[(o.target_market==tm)&(o.target_category==tc)].donor_market:
        Dn=G[(dm,tc)]; items=np.intersect1d(T.i.unique(),Dn.i.unique())
        if len(items)<10: continue
        rng=np.random.default_rng(zlib.crc32(f'{tm}|{tc}|{dm}'.encode()))   # per-pair seed: results do not depend on how pairs are split across workers
        Fs=stats(Dn,items); users=T.u.unique(); du=Dn[Dn.i.isin(items)].u.unique(); acc=[]
        for rep in range(3):
            h=rng.random(len(users))<0.5; A=T[T.u.isin(users[h])]; B=T[T.u.isin(users[~h])]
            sa=stats(A,items); sb=stats(B,items)
            nB=int(B.i.isin(items).sum()); perm=rng.permutation(du); Dsh=Dn[Dn.i.isin(items)]
            cnt=Dsh.groupby('u').size().reindex(perm).cumsum().values; take=perm[:max(1,np.searchsorted(cnt,nB)+1)]
            sm=stats(Dsh[Dsh.u.isin(take)],items)
            acc.append({**{'loc_'+k:v for k,v in agree(sa,sb).items()},**{'for_'+k:v for k,v in agree(sa,Fs).items()},**{'mat_'+k:v for k,v in agree(sa,sm).items()},
                        'nB':nB,'nF':int(len(Dsh)),'nM':int(Dsh.u.isin(take).sum())})
        m=pd.DataFrame(acc).mean(numeric_only=True).to_dict()
        rows.append(dict(t=f'{tm} {tc}',target_market=tm,donor_market=dm,category=tc,n_shared=len(items),**m))
pd.DataFrame(rows).to_pickle(H(f'{PAPER_RAW}/travel3_{W}.pkl')); print('done',len(rows))
