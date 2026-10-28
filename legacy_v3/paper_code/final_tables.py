"""Builds every LaTeX table for the manuscript from the raw result files. Numbers are computed, never typed."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, LEGACY

import pandas as pd,numpy as np,json,glob,os,itertools,math
H=os.path.expanduser; S=PAPER_RAW; A=REPO
rng=np.random.default_rng(2026)
def cboot(vals,ts,stat=np.median,B=2000):
    """Two-way cluster bootstrap: resample markets and categories independently."""
    vals=np.asarray(vals,float); mk=np.array([t.split()[0] for t in ts]); ct=np.array([t.split()[1] for t in ts])
    ok=~np.isnan(vals); vals,mk,ct=vals[ok],mk[ok],ct[ok]
    mi,M=pd.factorize(mk); ci,C=pd.factorize(ct); o=np.argsort(vals); v=vals[o]; mi=mi[o]; ci=ci[o]; out=[]
    for _ in range(B):
        w=np.bincount(rng.integers(0,len(M),len(M)),minlength=len(M))[mi]*np.bincount(rng.integers(0,len(C),len(C)),minlength=len(C))[ci]
        if w.sum()==0: continue
        if stat is np.median: c=np.cumsum(w); out.append(v[np.searchsorted(c,c[-1]/2)])
        else: out.append(np.average(v,weights=w))
    return np.percentile(out,[2.5,97.5])
def f(x,d=3,sign=False): return ('{:+.%df}'%d if sign else '{:.%df}'%d).format(x)
def ci(lo,hi,d=2,sign=False): return f'[{f(lo,d,sign)}, {f(hi,d,sign)}]'
def table(caption,label,cols,rows,notes):
    s=['\\begin{table}[H]','\\caption{'+caption+'}','\\label{'+label+'}','\\centering','\\small',
       '\\begin{adjustbox}{max width=\\linewidth}','\\begin{tabular}{'+cols+'}','\\toprule']
    s+=rows+['\\bottomrule','\\end{tabular}','\\end{adjustbox}']
    if notes: s+=['\\begin{tablenotes}\\footnotesize','\\item '+notes,'\\end{tablenotes}']
    s+=['\\end{table}']; return '\n'.join(s)
T={}
# ---------------- Table 1: data
c=pd.read_parquet(f'{A}/results/contexts.parquet'); o=pd.read_parquet(f'{A}/results/overlap_random.parquet'); xm=o[o.channel=='XM']
T['tab_data']=table('Dataset after preprocessing (XMarket, random split).','tab:data','lr',
 [f'Markets (countries) & {c.market.nunique()} \\\\', f'Categories & {c.category.nunique()} \\\\', f'Market--category contexts & {len(c)} \\\\',
  f'Training purchases per context (median [min, max]) & {int(c.n_random_train.median()):,} [{int(c.n_random_train.min()):,}, {int(c.n_random_train.max()):,}] \\\\',
  f'Products per context (median) & {int(c["items"].median()):,} \\\\',
  f'Same-category foreign donor pairs & {len(xm)} \\\\',
  f'Shoppers shared between markets & {int(xm.shared_users.sum())} \\\\',
  f'Share of local purchases on products also sold abroad (median per pair) & {xm.shared_item_interaction_cover.median():.2f} \\\\'],
 'A context is one country--category store, e.g.\\ Germany Grocery. No shopper appears in two countries.')
# ---------------- Table 2: popularity vs patterns (fair benchmark: every comparison uses local half A as reference)
X=pd.concat([pd.read_pickle(fn) for fn in sorted(glob.glob(f'{S}/travel3_*.pkl'))]).drop_duplicates(['t','donor_market']).sort_values(['t','donor_market']).reset_index(drop=True)
names={'sp':'Popularity: rank correlation (Spearman)','top10':'Popularity: top-10 best-seller overlap','cos':'Patterns: co-purchase similarity (cosine)','nbr5':'Patterns: 5-nearest-neighbour overlap'}
rows=[]; mat_note=[]
for m in ['sp','top10','cos','nbr5']:
    # all-foreign columns use every pair where the foreign ratio exists; matched columns every pair where the matched ratio exists
    z=X[X['loc_'+m]>0].dropna(subset=['loc_'+m,'for_'+m]); zm=X[X['loc_'+m]>0].dropna(subset=['loc_'+m,'mat_'+m])
    r1=z['for_'+m]/z['loc_'+m]; r2=zm['mat_'+m]/zm['loc_'+m]
    l1,h1=cboot(r1.values,z.t.values); l2,h2=cboot(r2.values,zm.t.values)
    if len(zm)!=len(z): mat_note.append(f'{names[m].split(": ")[1]}: {len(zm)} pairs')
    rows.append(f'{names[m]} & {len(z)} & {z["loc_"+m].median():.2f} & {z["for_"+m].median():.2f} & {zm["mat_"+m].median():.2f} & {r1.median():.2f} {ci(l1,h1)} & {r2.median():.2f} {ci(l2,h2)} \\\\')
rows.append('\\midrule')
for lab,(pa,po) in [('Paired: patterns $-$ popularity (cosine vs rank)',('cos','sp')),('Paired: patterns $-$ popularity (neighbours vs top-10)',('nbr5','top10'))]:
    for arm in ['for','mat']:
        z=X[(X['loc_'+pa]>0)&(X['loc_'+po]>0)].copy(); z['dd']=z[f'{arm}_{pa}']/z['loc_'+pa]-z[f'{arm}_{po}']/z['loc_'+po]; z=z.dropna(subset=['dd'])
        lo,hi=cboot(z.dd.values,z.t.values)
        rows.append(f'{lab}, {"all foreign" if arm=="for" else "volume-matched"} & {len(z)} & \\multicolumn{{3}}{{c}}{{patterns higher in {100*(z.dd>0).mean():.0f}\\% of pairs}} & \\multicolumn{{2}}{{c}}{{{z.dd.median():+.2f} {ci(lo,hi,2,True)}}} \\\\')
T['tab_travel']=table('Cross-market agreement on shared products, relative to agreement between two halves of the local shoppers.','tab:travel','lcccccc',
 ['Measure & Pairs & Local B vs A & Foreign vs A & Matched foreign vs A & Ratio foreign (95\\% CI) & Ratio matched (95\\% CI) \\\\','\\midrule']+rows,
 f'Local shoppers are split into random halves A and B (3 splits, averaged). Every comparison uses half A as the reference, so local and foreign data are compared against the same thing. Matched foreign = random foreign shoppers subsampled until they hold up to as many purchases of shared products as half B (median {int(X.nB.median())} vs {int(X.nM.median())}; all foreign data: {int(X.nF.median())}); in {100*(X.nF<X.nB).mean():.0f}\\% of pairs the foreign data is smaller than half B and is used in full. Ratio = foreign agreement / local agreement, median over pairs; pairs with non-positive local agreement are excluded.{(" Matched columns use fewer pairs where the matched sample is too small for the measure ("+"; ".join(mat_note)+").") if mat_note else ""} CIs: two-way cluster bootstrap over markets and categories.')
# ---------------- Table 3: borrowing at full data (random + temporal)
df=pd.read_pickle(f'{S}/df.pkl'); full=df[(df.channel=='XM')&(df.order=='full')&(df.frac==1)&df.error.isna()]
def frp(model,split):
    fs=sorted(glob.glob(f'{S}/fr_{model}_{split}_*.jsonl'))
    q=pd.DataFrame([json.loads(x) for fn in fs for x in open(fn, encoding='utf-8')]).drop_duplicates(['t','foreign'])
    p=q.pivot(index='t',columns='foreign',values='ndcg').dropna(); return p[0],p[1]-p[0]
rows=[]
for split,models in [('Random',['MostPop','ItemKNN','EASE','UserKNN','PureSVD','Random']),('Temporal',['MostPop','ItemKNN','EASE','UserKNN'])]:
    for m in models:
        if split=='Random' and m!='UserKNN':
            g=full[full.model==m]; base,gain,ts=g.base.values,g.d.values,g.t.values
        else:
            b0,gg=frp(m,split.lower()); base,gain,ts=b0.values,gg.values,gg.index.values
        lo,hi=cboot(gain,ts,stat=np.mean)
        rows.append(f'{split} & {m} & {len(gain)} & {np.median(base):.3f} & {np.median(gain):+.4f} & {np.mean(gain):+.4f} {ci(lo,hi,3,True)} & {(gain>0.01).sum()} & {(gain<-0.01).sum()} \\\\')
    rows.append('\\midrule')
T['tab_full']=table('Adding all same-category foreign data to a context with complete local data (NDCG@10, full catalog).','tab:full','llcccccc',
 ['Split & Model & Contexts & Local only & Gain (median) & Gain (mean, 95\\% CI) & Better $>$0.01 & Worse $>$0.01 \\\\','\\midrule']+rows[:-1],
 'Hyperparameters tuned on validation data separately with and without foreign data; six contexts without validation shoppers use the first grid setting in both arms. UserKNN and all temporal-split rows come from dedicated runs (uk Books excluded for cost; EASE skips catalogs over 4,000 products). PureSVD is reported for completeness only; its results reflect a rank budget shared by all data rather than transfer.')
# ---------------- Table 4: scarcity regimes
rows_all=[]
for fn in sorted(glob.glob(f'{S}/ma_*_*_*.jsonl')): rows_all+=[json.loads(l) for l in open(fn, encoding='utf-8')]
r=pd.DataFrame(rows_all); r=r[r.err.isna()].drop_duplicates(['t','split','model','regime','seed','own_frac','foreign'])
lab={'R':'Shopping patterns scarce','B':'Both scarce (new market)','P':'Shopper histories scarce'}
rows=[]
for split in ['random','temporal']:
    for model in ['ItemKNN','EASE','UserKNN']:
        g=r[(r.split==split)&(r.model==model)]
        if g.empty: continue
        F=g[g.regime=='F'].pivot_table(index='t',columns='foreign',values='ndcg')
        for reg in ['R','B','P']:
            q=g[g.regime==reg]; mm=q.groupby(['t','own_frac','foreign']).ndcg.mean().unstack(['own_frac','foreign'])
            Z=pd.DataFrame({'y0':mm[(0.1,0)],'y1':mm[(0.1,1)]}); Z['full']=F[0].reindex(Z.index); Z=Z.dropna()
            den=Z.full-Z.y0; Z=Z[den>0.01]; gain=Z.y1-Z.y0; gap=gain/(Z.full-Z.y0)
            lo,hi=cboot(gap.values,gap.index.values)
            rows.append(f'{split.capitalize()} & {model} & {lab[reg]} & {int(gap.notna().sum())} & {Z.y0.median():.3f} & {Z.y1.median():.3f} & {gain.median():+.3f} & {gap.median():.2f} {ci(lo,hi)} \\\\')
        rows.append('\\midrule')
T['tab_scarcity']=table('Value of foreign data when only 10\\% of one kind of local knowledge is kept.','tab:scarcity','lllccccc',
 ['Split & Model & Local knowledge removed & Contexts & Local 10\\% & +Foreign & Gain & Gap closed (95\\% CI) \\\\','\\midrule']+rows[:-1],
 'Gap closed = (score with foreign data $-$ score with 10\\% local) / (score with 100\\% local $-$ score with 10\\% local), median over contexts. 3 random subsamples per context, hyperparameters re-tuned on validation data in every arm. Contexts = contexts with a defined gap (100\\% local $-$ 10\\% local $>$ 0.01); all columns use these contexts. Contexts with no validation shoppers use the first grid setting. Contexts with more than 2.3M user--item cells are excluded for cost, except that the ItemKNN random-split run adds 8 of them (6 with one subsample, fr Movies and mx Electronics with 3); ca Books, ca Electronics, ca Home \\& Kitchen, de Movies, uk Books, uk Electronics and uk Movies are never included.')
# ---------------- Table 5: launch test
l=pd.concat([pd.DataFrame([json.loads(x) for x in open(fn, encoding='utf-8')]) for fn in sorted(glob.glob(f'{S}/l2_*.jsonl'))]).drop_duplicates(['t','split','model','seed','foreign'])
l=l[l.groupby(['split','model','t','seed']).foreign.transform('nunique')==2]   # only launch sets run in both arms
rows=[]
for split in ['random','temporal']:
    rnd=l[(l.split==split)&(l.model=='Random')].groupby('t')['ndcg@10'].mean()
    for m in ['ItemKNN','EASE','UserKNN','MostPop']:
        g=l[(l.split==split)&(l.model==m)]
        P=g.groupby(['t','foreign'])[['ndcg@10','recall@10']].mean().unstack('foreign').dropna()
        gain=P[('ndcg@10',1)]-P[('ndcg@10',0)]; lo,hi=cboot(gain.values,gain.index.values,stat=np.mean)
        rows.append(f'{split.capitalize()} & {m} & {len(P)} & {P[("ndcg@10",0)].median():.3f} & {P[("ndcg@10",1)].median():.3f} & {gain.mean():+.3f} {ci(lo,hi,3,True)} & {(gain>0.01).sum()}/{len(P)} & {P[("recall@10",1)].median():.2f} \\\\')
    rows.append(f'{split.capitalize()} & Random (chance) & {len(rnd)} & {rnd.median():.3f} & -- & -- & -- & -- \\\\'); rows.append('\\midrule')
tuned=l[l.model.isin(['ItemKNN','EASE','UserKNN'])].tuned.mean()
T['tab_launch']=table('Launching products: all local purchases of 30\\% of the products that are also sold abroad are hidden (NDCG@10 on shoppers whose test purchase is a launched product).','tab:launch','llcccccc',
 ['Split & Model & Contexts & Local only & +Foreign & Gain (mean, 95\\% CI) & Improved & Recall@10 \\\\','\\midrule']+rows[:-1],
 f'3 random launch sets per context. Hyperparameters tuned on the hidden validation purchases of launched products where at least 10 validation shoppers exist ({100*tuned:.0f}\\% of runs); otherwise the first grid setting is used. Only launch sets run both with and without foreign data are used. Launch sets with fewer than 10 test shoppers are skipped. uk Books is excluded for cost; EASE skips catalogs over 4,000 products.')
# ---------------- Table 6: short histories
hh=pd.read_pickle(f'{S}/hist.pkl'); hh['g']=hh.y1-hh.y0
hh['bucket']=pd.cut(hh['hist'],[0,1,2,4,1e9],labels=['1','2','3--4','5+'])
rows=[]
for m in ['ItemKNN','EASE']:
    x=hh[hh.model==m]; pc=x.groupby(['t','bucket'],observed=True).agg(g=('g','mean'),n=('g','size')).reset_index(); pc=pc[pc.n>=20]
    for bk,g2 in pc.groupby('bucket',observed=True):
        lo,hi=cboot(g2.g.values,g2.t.values,stat=np.mean)
        rows.append(f'{m} & {bk} & {g2.t.nunique()} & {int(g2.n.sum()):,} & {g2.g.median():+.4f} & {g2.g.mean():+.4f} {ci(lo,hi,3,True)} \\\\')
    x2=x.assign(s=np.where(x['hist']<=2,'s','l')).groupby(['t','s']).agg(g=('g','mean'),n=('g','size')).reset_index(); x2=x2[x2.n>=20]
    w=x2.pivot(index='t',columns='s',values='g').dropna(); dd=w.s-w.l
    from scipy.stats import wilcoxon
    rows.append(f'\\multicolumn{{6}}{{l}}{{{m}, within context: short (1--2) gains more in {(dd>0).sum()}/{len(dd)} contexts, Wilcoxon $p={wilcoxon(dd).pvalue:.3f}$}} \\\\'); rows.append('\\midrule')
T['tab_history']=table('Gain from foreign data by the shopper\'s number of past local purchases (complete local data).','tab:history','lccccc',
 ['Model & Past purchases & Contexts & Shoppers & Gain (median) & Gain (mean, 95\\% CI) \\\\','\\midrule']+rows[:-1],
 'Per-shopper NDCG@10 difference between the target-only and all-foreign-donor runs, averaged within context (buckets with $\\geq$20 shoppers).')
# ---------------- Table 7: culture
hof={'ca':[39,80,52,48,36,68],'de':[35,67,66,65,83,40],'fr':[68,71,43,86,63,48],'in':[77,48,56,40,51,26],'jp':[54,46,95,92,88,42],'mx':[81,30,69,82,24,97],'uk':[35,89,66,35,51,69]}
Mh=np.array(list(hof.values()),float); var=Mh.var(axis=0,ddof=1)
def ks(a,b): return float(np.mean((np.array(hof[a])-np.array(hof[b]))**2/var))
lang={'ca':{'en','fr'},'uk':{'en'},'in':{'en'},'fr':{'fr'},'de':{'de'},'mx':{'es'},'jp':{'ja'}}
def sl(a,b): return float(bool(lang[a]&lang[b]))
cap={'ca':(45.42,-75.70),'de':(52.52,13.40),'fr':(48.86,2.35),'in':(28.61,77.21),'jp':(35.68,139.69),'mx':(19.43,-99.13),'uk':(51.51,-0.13)}
def geo(a,b):
    (la1,lo1),(la2,lo2)=[(math.radians(p[0]),math.radians(p[1])) for p in (cap[a],cap[b])]
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2; return 6371*2*math.asin(math.sqrt(h))
P2=pd.read_pickle(f'{S}/pairs2.pkl')[['t','donor_market','d']]
Y=X.merge(P2,on=['t','donor_market'],how='left')
for m,col in [('sp','pop_rel'),('top10','pop10_rel'),('cos','pat_rel'),('nbr5','nbr_rel')]:
    Y[col]=(Y['for_'+m]/Y['loc_'+m]).where(Y['loc_'+m]>0)
mk=sorted(hof); perms=list(itertools.permutations(mk)); rows=[]
for col,lab in [('pop_rel','Popularity agreement (rank correlation)'),('pop10_rel','Popularity agreement (top-10 overlap)'),('pat_rel','Pattern agreement (cosine)'),('nbr_rel','Pattern agreement (neighbour overlap)'),('d','Gain from one foreign donor (ItemKNN)')]:
    z=Y.dropna(subset=[col]).copy(); z['res']=z[col]-z.groupby('t')[col].transform('mean')
    pm=z.groupby(['target_market','donor_market']).res.mean().reset_index(); cells=[]
    for fn in (ks,sl,geo):
        def st(p):
            mp=dict(zip(mk,p)); v=np.array([fn(mp[a],mp[b]) for a,b in zip(pm.target_market,pm.donor_market)]); return np.corrcoef(v,pm.res)[0,1]
        obs=st(mk); null=np.array([st(p) for p in perms]); cells.append(f'{obs:+.2f} ({np.mean(np.abs(null)>=abs(obs)-1e-12):.2f})')
    rows.append(f'{lab} & {len(z)} & '+' & '.join(cells)+' \\\\')
T['tab_culture']=table('Do country-level distances predict cross-market agreement or transfer? Correlation $r$ (exact permutation $p$).','tab:culture','lcccc',
 ['Outcome & Pairs & Hofstede distance & Shared language & Geographic distance \\\\','\\midrule']+rows,
 'Outcomes are centred within each target context, averaged to 42 ordered country pairs, and tested against all 5,040 relabellings of the 7 countries. Hofstede distance: Kogut--Singh index over six dimensions. Ratios use the fair benchmark of Table~\\ref{tab:travel}, excluding pairs with non-positive local agreement. The gain row excludes in Arts\\_Crafts\\_and\\_Sewing, whose local-only ItemKNN scores below chance. 15 tests; none survives a Bonferroni correction ($\\alpha=0.05/15$).')
with open(f'{LEGACY}/paper/tables.tex','w', encoding='utf-8') as fh:
    fh.write('% Auto-generated by final_tables.py. Requires \\usepackage{booktabs,threeparttable,float,adjustbox}.\n\n'+'\n\n'.join(T.values())+'\n')
print('\n\n'.join(T.values()))
