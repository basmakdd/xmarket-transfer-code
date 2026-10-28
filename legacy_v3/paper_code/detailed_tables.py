"""Per-context / per-pair detail behind every aggregate table, plus compact market x category matrices.
Writes paper/tables_detailed.tex. Needs booktabs, longtable."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, LEGACY

import pandas as pd,numpy as np,json,glob,os
H=os.path.expanduser
S=PAPER_RAW
A=REPO
CAT={'Arts_Crafts_and_Sewing':'Arts \\& Crafts','Automotive':'Automotive','Books':'Books','CDs_and_Vinyl':'CDs \\& Vinyl',
 'Cell_Phones_and_Accessories':'Cell Phones','Digital_Music':'Digital Music','Electronics':'Electronics','Grocery_and_Gourmet_Food':'Grocery',
 'Home_and_Kitchen':'Home \\& Kitchen','Industrial_and_Scientific':'Industrial','Movies_and_TV':'Movies \\& TV','Musical_Instruments':'Musical Instr.',
 'Office_Products':'Office','Sports_and_Outdoors':'Sports','Toys_and_Games':'Toys'}
MK=['ca','de','fr','in','jp','mx','uk']
c=pd.read_parquet(f'{A}/results/contexts.parquet'); c['t']=c.market+' '+c.category
size=c.set_index('t')['n_random_train']
def order(ts):
    return sorted(ts,key=lambda t:(t.split()[1],size.get(t,0)))
def n(x,d=3,sign=False):
    if x is None or (isinstance(x,float) and np.isnan(x)): return '--'
    return ('{:+.%df}'%d if sign else '{:.%df}'%d).format(x)
def longtable(caption,label,cols,header,rows,note):
    k=len(cols)
    if not header.rstrip().endswith('\\\\'): header=header+' \\\\'
    s=['\\begin{landscape}','{\\scriptsize','\\setlength{\\tabcolsep}{3pt}','\\begin{longtable}{'+cols+'}',
       '\\caption{'+caption+'}\\label{'+label+'}\\\\','\\toprule',header,'\\midrule','\\endfirsthead',
       '\\multicolumn{'+str(k)+'}{l}{\\emph{(continued)}}\\\\','\\toprule',header,'\\midrule','\\endhead',
       '\\endlastfoot']
    s+=rows+['\\bottomrule','\\end{longtable}','\\vspace{-1em}\\par\\noindent\\parbox{\\linewidth}{'+note+'}','}','\\end{landscape}']
    return '\n'.join(s)
def lab(t): m,cat=t.split(); return CAT.get(cat,cat), m.upper()
OUT=[]
# ---------- D1: agreement per (target, donor) pair
X=pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f'{S}/travel3_*.pkl'))]).drop_duplicates(['t','donor_market']).sort_values(['t','donor_market']).reset_index(drop=True)
for m in ['sp','top10','cos','nbr5']: X['r_'+m]=(X['for_'+m]/X['loc_'+m]).where(X['loc_'+m]>0)
rows=[]
for t in order(X.t.unique()):
    for _,r in X[X.t==t].sort_values('donor_market').iterrows():
        cat,m=lab(t)
        rows.append(f'{cat} & {m} & {r.donor_market.upper()} & {int(r.n_shared)} & {n(r.r_sp,2)} & {n(r.r_top10,2)} & {n(r.r_cos,2)} & {n(r.r_nbr5,2)} & {n(r.r_cos-r.r_sp,2,True)} \\\\')
OUT.append(longtable('Cross-market agreement for every target--foreign country pair, as a share of local agreement (foreign vs half A divided by half B vs half A).','tab:travel_detail','lllrrrrrr',
 'Category & Target & Foreign & Shared & Pop.\\ rank & Pop.\\ top-10 & Pattern cos & Pattern nbr & Pattern$-$pop \\\\',rows,
 'Ratios of 1 mean foreign shoppers agree with local shoppers as well as other local shoppers do. Pop.\\ = popularity, nbr = 5-nearest-neighbour overlap. Last column: pattern (cosine) ratio minus popularity (rank) ratio. -- = undefined (non-positive local agreement or too few products).'))
# ---------- D2: complete local data per context (random + temporal)
df=pd.read_pickle(f'{S}/df.pkl')
def fr(model,split):
    fs=sorted(glob.glob(f'{S}/fr_{model}_{split}_*.jsonl'))
    if not fs: return None
    r=pd.DataFrame([json.loads(x) for f in fs for x in open(f, encoding='utf-8')]).drop_duplicates(['t','foreign'])
    return r.pivot(index='t',columns='foreign',values='ndcg')
def full_random(model):
    if model=='UserKNN': return fr('UserKNN','random')
    k0=df[(df.k==0)&(df.model==model)&df.error.isna()].set_index('t')['ndcg@10']
    xf=df[(df.model==model)&(df.channel=='XM')&(df.order=='full')&(df.frac==1)&df.error.isna()].set_index('t')['ndcg@10']
    return pd.DataFrame({0:k0,1:xf})
models=['MostPop','ItemKNN','EASE','UserKNN']
for split in ['random','temporal']:
    P={m:(full_random(m) if split=='random' else fr(m,split)) for m in models}
    ts=order(set().union(*[set(p.dropna().index) for p in P.values() if p is not None]))
    rows=[]
    for t in ts:
        cat,m=lab(t); cells=[]
        for mo in models:
            p=P[mo]
            if p is None or t not in p.index or p.loc[t].isna().any(): cells+=['--','--']
            else: cells+=[n(p.loc[t,0]),n(p.loc[t,1]-p.loc[t,0],3,True)]
        rows.append(f'{cat} & {m} & {int(size[t]):,} & '+' & '.join(cells)+' \\\\')
    OUT.append(longtable(f'Complete local data: NDCG@10 without foreign data and gain from adding all same-category foreign data, per context ({split} split).',f'tab:full_detail_{split}','llrrrrrrrrr',
     'Category & Market & Train & \\multicolumn{2}{c}{MostPop} & \\multicolumn{2}{c}{ItemKNN} & \\multicolumn{2}{c}{EASE} & \\multicolumn{2}{c}{UserKNN} \\\\ & & & Local & Gain & Local & Gain & Local & Gain & Local & Gain',rows,
     'Local = local data only; Gain = with all foreign data minus local only. Hyperparameters tuned per arm on validation data. -- = not run (EASE: catalogs over 4,000 products; uk Books too large for some runs).'))
# ---------- D3: scarcity per context
r=pd.concat([pd.DataFrame([json.loads(x) for x in open(f, encoding='utf-8')]) for f in sorted(glob.glob(f'{S}/ma_*.jsonl'))])
r=r[r.err.isna()].drop_duplicates(['t','split','model','regime','seed','own_frac','foreign'])
for split,model in [('random','ItemKNN'),('random','EASE'),('random','UserKNN'),('temporal','ItemKNN')]:
    g=r[(r.split==split)&(r.model==model)]
    F=g[g.regime=='F'].pivot_table(index='t',columns='foreign',values='ndcg')
    rows=[]
    for t in order(g.t.unique()):
        cat,m=lab(t); cells=[]
        full=F.loc[t,0] if t in F.index else np.nan
        for reg in ['R','B','P']:
            q=g[(g.t==t)&(g.regime==reg)&(g.own_frac==0.1)]
            y0=q[q.foreign==0].ndcg.mean(); y1=q[q.foreign==1].ndcg.mean(); den=full-y0
            cells+=[n(y0),n(y1),n((y1-y0)/den,2) if den>0.01 else '--']
        rows.append(f'{cat} & {m} & {n(full)} & '+' & '.join(cells)+f' & {int(g[g.t==t].seed.nunique())} \\\\')
    OUT.append(longtable(f'Scarcity experiment per context ({model}, {split} split): NDCG@10 with 10\\% of local data, with foreign data added, and share of the gap to 100\\% local closed.',f'tab:scarcity_detail_{model}_{split}','llrrrrrrrrrrr',
     'Category & Market & 100\\% & \\multicolumn{3}{c}{Patterns scarce} & \\multicolumn{3}{c}{Both scarce} & \\multicolumn{3}{c}{Histories scarce} & Seeds \\\\ & & local & 10\\% & +For. & Gap & 10\\% & +For. & Gap & 10\\% & +For. & Gap &',rows,
     'Values averaged over the listed number of random subsamples. Gap = (+For.\\ $-$ 10\\%)/(100\\% $-$ 10\\%), shown only when the denominator exceeds 0.01.'))
# ---------- D4: launch per context
l=pd.concat([pd.DataFrame([json.loads(x) for x in open(f, encoding='utf-8')]) for f in sorted(glob.glob(f'{S}/l2_*.jsonl'))]).drop_duplicates(['t','split','model','seed','foreign'])
l=l[l.groupby(['split','model','t','seed']).foreign.transform('nunique')==2]
for split in ['random','temporal']:
    g=l[l.split==split]; P=g.groupby(['t','model','foreign'])['ndcg@10'].mean().unstack(['model','foreign'])
    meta=g.groupby('t')[['n_launched','n_users']].mean()
    rows=[]
    for t in order(P.index):
        cat,m=lab(t); cells=[]
        for mo in ['ItemKNN','EASE','UserKNN','MostPop']:
            a=P.get((mo,0),pd.Series(dtype=float)).get(t,np.nan); b=P.get((mo,1),pd.Series(dtype=float)).get(t,np.nan)
            cells+=[n(a),n(b)]
        rows.append(f'{cat} & {m} & {int(round(meta.loc[t,"n_launched"]))} & {int(round(meta.loc[t,"n_users"]))} & '+' & '.join(cells)+f' & {n(P.get(("Random",0),pd.Series(dtype=float)).get(t,np.nan))} \\\\')
    OUT.append(longtable(f'Product launch per context ({split} split): NDCG@10 on shoppers whose test purchase is a launched product, with local data only and with foreign data.',f'tab:launch_detail_{split}','llrrrrrrrrrrr',
     'Category & Market & Launched & Shoppers & \\multicolumn{2}{c}{ItemKNN} & \\multicolumn{2}{c}{EASE} & \\multicolumn{2}{c}{UserKNN} & \\multicolumn{2}{c}{MostPop} & Random \\\\ & & & & Local & +For. & Local & +For. & Local & +For. & Local & +For. &',rows,
     'Means over up to 3 launch sets (only sets run in both arms). Launched = products whose local purchases are hidden; Shoppers = test shoppers who bought one of them.'))
# ---------- compact market x category matrices (candidates for the main paper)
def matrix(caption,label,vals,d=2,sign=True,note=''):
    cats=sorted(set(t.split()[1] for t in vals.index)); rows=[]
    for cat in cats:
        cells=[n(vals.get(f'{m} {cat}',np.nan),d,sign) for m in MK]
        if all(x=='--' for x in cells): continue
        rows.append(f'{CAT.get(cat,cat)} & '+' & '.join(cells)+' \\\\')
    s=['\\begin{table}[H]','\\caption{'+caption+'}','\\label{'+label+'}','\\centering','\\small','\\begin{tabular}{l'+'r'*7+'}','\\toprule',
       'Category & '+' & '.join(x.upper() for x in MK)+' \\\\','\\midrule']+rows+['\\bottomrule','\\end{tabular}']
    if note: s+=['\\par\\footnotesize '+note]
    s+=['\\end{table}']; return '\n'.join(s)
dd=(X.r_cos-X.r_sp).groupby(X.t).median()
OUT.append(matrix('How much better shopping patterns travel than popularity, per context (median over foreign countries of pattern-cosine ratio minus popularity-rank ratio).','tab:mat_travel',dd,note='Positive = patterns travel better than popularity. -- = no eligible pair. Very large values (e.g.\\ CA Grocery) arise when local popularity agreement is close to zero.'))
P=full_random('ItemKNN'); OUT.append(matrix('Gain in NDCG@10 from adding all foreign data with complete local data (ItemKNN, random split).','tab:mat_full',(P[1]-P[0]),d=3,note='Mostly close to zero: with complete local data, foreign data adds little. IN Arts \\& Crafts (+0.68) is an outlier: its local-only ItemKNN scores below chance, so foreign data repairs a failed model rather than improving a working one.'))
g=r[(r.split=='random')&(r.model=='ItemKNN')]; F=g[g.regime=='F'].pivot_table(index='t',columns='foreign',values='ndcg')[0]
q=g[(g.regime=='R')&(g.own_frac==0.1)].groupby(['t','foreign']).ndcg.mean().unstack(); den=F.reindex(q.index)-q[0]
OUT.append(matrix('Share of the gap closed by foreign data when local shopping-pattern data is scarce (ItemKNN, random split, 10\\% local).','tab:mat_scarcity',((q[1]-q[0])/den).where(den>0.01),sign=False))
P=l[(l.split=='random')&(l.model=='ItemKNN')].groupby(['t','foreign'])['ndcg@10'].mean().unstack()
OUT.append(matrix('Product launch: NDCG@10 with foreign data for launched products (ItemKNN, random split); local-only is 0.00 everywhere.','tab:mat_launch',P[1],sign=False))
open(f'{LEGACY}/paper/tables_detailed.tex','w', encoding='utf-8').write('% Auto-generated by detailed_tables.py. Requires \\usepackage{booktabs,longtable,float,pdflscape}.\n\n'+'\n\n'.join(OUT)+'\n')
print('tables:',len(OUT))
