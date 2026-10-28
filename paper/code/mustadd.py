"""Shared fitting helpers (UserKNN, GRIDS, build, set_prof, weight_rows, mats, run_unit) and the v3 scarcity runner.

The functions are used by rq2_run.py, scarcity2.py, launch3.py and other scripts in paper/code. The main block is the
scarcity runner of the v3 study (records paper/raw/ma_*.jsonl; superseded by scarcity2.py, which reuses run_unit's
semantics): 3 scarcity regimes x seeds x re-tuning, ItemKNN/EASE/UserKNN, random|temporal split.
Regimes (domestic training edges subsampled to share f, nested across f within a seed):
  B  both scarce     : model fitted on subsampled data, users' profiles subsampled
  R  relations scarce: model fitted on subsampled domestic (+foreign), profiles complete
  P  profiles scarce : model fitted on complete domestic (+foreign), profiles subsampled
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys,os,json,time
sys.path.insert(0,os.path.expanduser(SRC))
import pandas as pd, numpy as np, scipy.sparse as sp
import data as D, experiment as E, models as Mod
from util import stable_seed

class UserKNN(Mod.Recommender):
    name='UserKNN'
    def __init__(self,R,target_cols,k=100,shrink=0.0):
        self.Rf=R.tocsr(); self.cols=np.asarray(target_cols); self.k=k; self.shrink=shrink
        self.RfT=self.Rf.T.tocsr(); self.Rt=self.Rf[:,self.cols].tocsr()
        self.nf=np.sqrt(np.asarray(self.Rf.sum(axis=1)).ravel()).astype(np.float32)
        self.Rp=self.Rf
    def score(self,rows):
        P=self.Rp[rows]
        np_=np.sqrt(np.asarray(P.sum(axis=1)).ravel()).astype(np.float32)
        C=(P@self.RfT).tocoo()
        den=np_[C.row]*self.nf[C.col]+np.float32(self.shrink)
        v=np.where(den>0,C.data/np.maximum(den,1e-12),0).astype(np.float32)
        v[C.col==rows[C.row]]=0.0   # never your own neighbour
        C=sp.coo_matrix((v,(C.row,C.col)),shape=(len(rows),self.Rf.shape[0])).tocsr(); C.eliminate_zeros()
        S=Mod._keep_top_k_per_column(C.T.tocsc(),self.k).T.tocsr()
        return np.asarray((S@self.Rt).toarray(),dtype=np.float32)

GRIDS={'ItemKNN':Mod.GRIDS['ItemKNN'],'EASE':Mod.GRIDS['EASE'],
       'UserKNN':[{'k':k,'shrink':s} for k in (20,100,500) for s in (0.0,10.0)]}
def build(model,Rfit,cols):
    if model=='UserKNN': return lambda p: UserKNN(Rfit,cols,**p)
    cls=Mod.BUILDERS[model]; return lambda p: cls(Rfit,cols,**p)
def set_prof(m,model,Rprof):
    if model=='UserKNN': m.Rp=Rprof
    else: m.R=Rprof

def weight_rows(R,edge_lists,eval_users,weights):
    """Scale the rows of users from edge list k by weights[k] (users are disjoint across markets).

    R is the binary user x product interaction matrix of the pooled training data (rows: local and foreign users).
    With weights = [1, a, a, ...] (edge list 0 = local data, the others = foreign donors), every FOREIGN user's row
    is multiplied by a (alpha); local rows, and hence the scoring profiles of the local evaluation users, are
    unchanged. The models then see alpha as follows (no model is changed; this is what the scaling implies):
      MostPop  item popularity = column sums: local counts + alpha x foreign counts.
      EASE     Gram matrix R^T R: each foreign user's contribution is multiplied by alpha^2
               (local + alpha^2 x foreign co-occurrences on the diagonal and off the diagonal).
      ItemKNN  co-occurrence R^T R: foreign contributions multiplied by alpha^2; the item norms are computed from the
               scaled matrix, sqrt(column sum) = sqrt(local count + alpha x foreign count), so the cosine is
               (C_local + alpha^2 C_foreign) / (sqrt(n_i,local + alpha n_i,foreign) sqrt(n_j,local + alpha n_j,foreign) + shrink).
      UserKNN  (mustadd.UserKNN) the similarity of a local user u to a foreign user v is
               alpha c_uv / (sqrt(n_u) sqrt(alpha n_v) + shrink), i.e. sqrt(alpha) x cosine when shrink = 0, and the
               neighbour's vote is its scaled row (alpha); with shrink = 0 a foreign neighbour's vote is alpha^1.5 x the
               unweighted one, and the top-k neighbour selection uses the scaled similarities.
      Random   unaffected.
    alpha = 0 removes the foreign data (equal to the local-only arm), alpha = 1 is equal-weight pooling.
    """
    if weights is None or all(w==1 for w in weights): return R
    all_u=np.unique(np.concatenate([e[0] for e in edge_lists]+[eval_users]))
    w=np.ones(R.shape[0],np.float32)
    for k,(eu,_) in enumerate(edge_lists):
        if weights[k]!=1: w[np.searchsorted(all_u,np.unique(eu))]=weights[k]
    return sp.diags(w).dot(R).tocsr()

def mats(edge_lists,catalog,eval_users,keep_mask,weights=None):
    """Full pooled matrix and the same matrix with dropped DOMESTIC edges removed.
    weights: optional per-edge-list row weights (e.g. [1, a, a, ...] to down-weight foreign users)."""
    R,cols,rows=E.build_matrix(edge_lists,catalog,eval_users)
    R=weight_rows(R,edge_lists,eval_users,weights)
    tu,ti=edge_lists[0]
    all_u=np.unique(np.concatenate([e[0] for e in edge_lists]+[eval_users]))
    all_i=np.unique(np.concatenate([e[1] for e in edge_lists]+[catalog]))
    dr=~keep_mask
    Dm=sp.csr_matrix((np.ones(dr.sum(),np.float32),(np.searchsorted(all_u,tu[dr]),np.searchsorted(all_i,ti[dr]))),shape=R.shape)
    Rs=(R-Dm).tocsr(); Rs.data=np.clip(Rs.data,0,None); Rs.eliminate_zeros()
    return R,Rs,cols,rows

def run_unit(model,s,vs,edge_lists,keep,regime,weights=None):
    best,bp=-1,None
    grid=GRIDS[model]
    def fitscore(sess,params):
        R,Rs,cols,rows=mats(edge_lists,sess.catalog,sess.eval_users,keep,weights)
        fit,prof={'B':(Rs,Rs),'R':(Rs,R),'P':(R,Rs),'F':(R,R)}[regime]
        m=build(model,fit,cols)(params); set_prof(m,model,prof)
        return E.evaluate(m,sess,rows,candidate_sizes=(),seed=0)[0]['ndcg@10']
    if len(grid)>1 and len(vs.eval_users):
        for p in grid:
            try: sc=fitscore(vs,p)
            except MemoryError: continue
            if sc>best: best,bp=sc,p
    if bp is None: bp=grid[0]
    return fitscore(s,bp),bp

if __name__=='__main__':
    LIM=float(sys.argv[1]); MODEL=sys.argv[2]; SPLIT=sys.argv[3]; W=int(sys.argv[4]); NW=int(sys.argv[5])
    SEEDS=[int(x) for x in sys.argv[6].split(',')]; REGIMES=sys.argv[7].split(',')
    MAXCOST=float(sys.argv[8]) if len(sys.argv)>8 else 1e18
    T0=time.time()
    out=os.path.expanduser(f'{PAPER_RAW}/ma_{MODEL}_{SPLIT}_{W}.jsonl')
    done=set()
    if os.path.exists(out):
        for l in open(out, encoding='utf-8'): r=json.loads(l); done.add((r['t'],r['regime'],r['seed'],r['own_frac'],r['foreign']))
    o=pd.read_parquet(os.path.expanduser(f'{REPO}/results/overlap_{SPLIT}.parquet')); o=o[o.channel=='XM']
    c=pd.read_parquet(os.path.expanduser(REPO+'/results/contexts.parquet'))
    cost={(m,cc):int(u)*int(i) for m,cc,u,i in zip(c["market"],c["category"],c["users"],c["items"])}
    data=D.load_for(SPLIT); blocks=E.load_blocks(data,SPLIT)
    targets=sorted(set(zip(o.target_market,o.target_category)),key=lambda t:cost.get(t,0))
    targets=[t for t in targets if cost.get(t,0)<=MAXCOST][W::NW]
    if os.environ.get('ONLY'): targets=[t for t in targets if ' '.join(t) in os.environ['ONLY'].split(',')]
    for target in targets:
        t=' '.join(target)
        donors=[tuple(x) for x in o[(o.target_market==target[0])&(o.target_category==target[1])][['donor_market','donor_category']].values]
        b=blocks[target]; n=len(b.train_u); s=vs=None
        plan=[('F',0,1.0,0),('F',0,1.0,1)]
        for reg in REGIMES:
            for seed in SEEDS:
                for f in (0.1,0.25,0.5): plan.append((reg,seed,f,0))
                for f in (0.1,0.25): plan.append((reg,seed,f,1))
        for reg,seed,f,foreign in plan:
            if (t,reg,seed,f,foreign) in done: continue
            if time.time()-T0>LIM: print('deadline',flush=True); sys.exit()
            if s is None: s=E.TargetSession(target,b); vs=s.validation_view()
            perm=np.random.default_rng(stable_seed('mustadd',t,seed)).permutation(n)
            keep=np.zeros(n,bool); keep[perm[:int(round(f*n))]]=True
            el=[(b.train_u,b.train_i)]+([(blocks[d].train_u,blocks[d].train_i) for d in donors] if foreign else [])
            t1=time.time()
            try: sc,bp=run_unit(MODEL,s,vs,el,keep,reg); err=None
            except MemoryError as e: sc,bp,err=None,None,'memory'
            rec={'t':t,'split':SPLIT,'model':MODEL,'regime':reg,'seed':seed,'own_frac':f,'foreign':foreign,'ndcg':sc,'params':bp,'err':err,
                 'n_dom':int(n),'n_for':int(sum(len(e[0]) for e in el[1:])),'sec':round(time.time()-t1,2)}
            open(out,'a', encoding='utf-8', newline='\n').write(json.dumps(rec)+'\n')
    print('ALL DONE',flush=True)
