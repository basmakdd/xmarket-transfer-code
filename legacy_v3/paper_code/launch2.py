"""Product-launch test (re-tuned). A random 30% of products that are sold locally AND abroad (and have a local test
purchase) are 'launched': all their local train+val purchases are hidden from training. Hyperparameters are tuned on the
hidden VALIDATION purchases of launched products; results are reported on their hidden TEST purchases."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys,os,json,time,glob
sys.path.insert(0,os.path.expanduser(SRC)); sys.path.insert(0,os.path.expanduser(CODE))
import pandas as pd,numpy as np
import data as D, experiment as E, models as Mod
from util import stable_seed
from mustadd import UserKNN, GRIDS
GR=dict(GRIDS); GR['MostPop']=[{}]; GR['Random']=[{}]
LIM=float(sys.argv[1]); MODEL=sys.argv[2]; SPLIT=sys.argv[3]; W=int(sys.argv[4]); NW=int(sys.argv[5]); MAXCOST=float(sys.argv[6])
out=os.path.expanduser(f'{PAPER_RAW}/l2_{MODEL}_{SPLIT}_{W}.jsonl'); T0=time.time()
done=set()
if os.path.exists(out):
    for l in open(out, encoding='utf-8'): r=json.loads(l); done.add((r['t'],r['seed'],r['foreign']))
o=pd.read_parquet(os.path.expanduser(f'{REPO}/results/overlap_{SPLIT}.parquet')); o=o[o.channel=='XM']
c=pd.read_parquet(os.path.expanduser(REPO+'/results/contexts.parquet'))
cost={(m,cc):int(u)*int(i) for m,cc,u,i in zip(c['market'],c['category'],c['users'],c['items'])}
data=D.load_for(SPLIT); blocks=E.load_blocks(data,SPLIT)
targets=sorted(set(zip(o.target_market,o.target_category)),key=lambda t:cost.get(t,0))
targets=[t for t in targets if cost.get(t,0)<=MAXCOST][W::NW]
if _os.environ.get('ONLY'): targets=[t for t in targets if ' '.join(t) in _os.environ['ONLY'].split(',')]
def restrict(s,launched):
    lc=set(np.searchsorted(s.catalog,launched).tolist())
    keep=[j for j,tr in enumerate(s.truth) if lc.intersection(tr.tolist())]
    s.eval_users=s.eval_users[keep]; s.truth=[np.array(sorted(lc.intersection(s.truth[j].tolist()))) for j in keep]; s.masked=[s.masked[j] for j in keep]
    return len(keep)
def make(model,R,cols,rows,s,cap,dense,p):
    if model=='UserKNN': return UserKNN(R,cols,**p)
    if model=='Random': return Mod.RandomRec(R,cols,user_ids=E.row_to_user_id(rows,s.eval_users))
    return Mod.build(model,R,cols,user_rows=rows,capacity=cap,dense=dense,**p)
for target in targets:
    t=' '.join(target)
    donors=[tuple(x) for x in o[(o.target_market==target[0])&(o.target_category==target[1])][['donor_market','donor_category']].values]
    b=blocks[target]
    if MODEL=='EASE' and len(b.catalog)>Mod.EASE_MAX_ITEMS: continue
    abroad=np.unique(np.concatenate([blocks[d].train_i for d in donors]))
    elig=np.intersect1d(np.intersect1d(np.unique(b.test_i),abroad),b.catalog)
    if len(elig)<5: continue
    for seed in (0,1,2):
        if all((t,seed,f) in done for f in (0,1)): continue
        rng=np.random.default_rng(stable_seed('launch',t,seed))
        launched=np.sort(rng.choice(elig,size=max(3,int(round(0.3*len(elig)))),replace=False))
        kt=~np.isin(b.train_i,launched); kv=~np.isin(b.val_i,launched)
        nb=E.ContextBlock(b.train_u[kt],b.train_i[kt],b.val_u[kv],b.val_i[kv],b.test_u,b.test_i,b.catalog)
        s=E.TargetSession(target,nb)
        if restrict(s,launched)<10: continue
        vb=E.ContextBlock(b.train_u[kt],b.train_i[kt],np.array([],np.int32),np.array([],np.int32),b.val_u,b.val_i,b.catalog)
        vs=E.TargetSession(target,vb); nval=restrict(vs,launched)
        cap,dense=E.target_capacity(s,nb)
        for foreign in (0,1):
            if (t,seed,foreign) in done: continue
            if time.time()-T0>LIM: print('deadline',flush=True); sys.exit()
            el=[(nb.train_u,nb.train_i)]+([(blocks[d].train_u,blocks[d].train_i) for d in donors] if foreign else [])
            grid=GR[MODEL]; bp=grid[0]; tuned=False
            if len(grid)>1 and nval>=10:
                Rv,cv,rv=E.build_matrix(el,vs.catalog,vs.eval_users); best=-1
                for p in grid:
                    sc=E.evaluate(make(MODEL,Rv,cv,rv,vs,cap,dense,p),vs,rv,candidate_sizes=(),seed=0)[0]['ndcg@10']
                    if sc>best: best,bp=sc,p
                tuned=True
            R,cols,rows=E.build_matrix(el,s.catalog,s.eval_users)
            res,_=E.evaluate(make(MODEL,R,cols,rows,s,cap,dense,bp),s,rows,candidate_sizes=(),seed=0)
            rec={'t':t,'split':SPLIT,'seed':seed,'foreign':foreign,'model':MODEL,'params':bp,'tuned':tuned,'n_val_users':nval,
                 'n_launched':int(len(launched)),'n_items':int(len(s.catalog)),'n_users':int(len(s.eval_users)),
                 'ndcg@10':res['ndcg@10'],'recall@10':res['recall@10']}
            open(out,'a', encoding='utf-8').write(json.dumps(rec)+'\n')
print('ALL DONE',flush=True)
