"""Per market x category counts before and after preprocessing -> ../../figures/data/data_distribution.csv
before = raw XMarket file; after = analysis/results/contexts.parquet (the files the experiments used).
Also re-runs the pipeline's adaptive k-core (analysis/src/data.py) on each raw file to check 'after'."""
import sys, os, glob, gzip, io, zipfile, json, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO + '/src'); import data as D
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'data'); os.makedirs(OUT, exist_ok=True)
cache = os.path.join(OUT, 'data_distribution_partial.json')
done = json.load(open(cache, encoding='utf-8')) if os.path.exists(cache) else {}
cats = sys.argv[1:] or D.CATEGORIES
z = zipfile.ZipFile(D.ZIP_PATH)
for cat in cats:
    for mk in D.MARKETS:
        key = f'{mk}|{cat}'
        if key in done: continue
        name = f'xmarket_categories/ratings_{mk}_{cat}.txt.gz'
        if name not in z.namelist(): done[key] = None; continue
        txt = gzip.decompress(z.read(name))
        if not txt.strip(): done[key] = dict(raw=0); continue
        df = pd.read_csv(io.BytesIO(txt), sep=' ', header=None, names=['user','item','rating','date'], dtype=str)
        rec = dict(raw=len(df), raw_users=df.user.nunique(), raw_items=df.item.nunique())
        dd = df.sort_values('date', kind='stable').drop_duplicates(['user','item'])
        rec['dedup'] = len(dd)
        kept, kk = D.adaptive_core(dd[['user','item','date']])
        rec.update(rec_core=kk, rec_after=len(kept), rec_users=kept.user.nunique() if kk else 0, rec_items=kept.item.nunique() if kk else 0)
        done[key] = rec; json.dump(done, open(cache, 'w', encoding='utf-8', newline='\n'))
        print(key, rec, flush=True)
