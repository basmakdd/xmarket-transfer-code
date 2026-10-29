"""Which repeated user-product rows are real duplicates? Classification with the review texts (reviewer comment).

src/data.py collapses every repeated (reviewer, ASIN) pair of a ratings file to its earliest row, whatever the dates
and ratings. A shared date and rating alone do not show that two rows are one review, so every repeated pair of the
RAW ratings files (before the cross-market copy removal) is matched with its rows in the review file of the same
market and category (reviews_<m>_<c>.json.gz, one row per ratings row; text = review text + summary, whitespace
normalised) and classified:
  identical_text_rating_date   one review stored more than once (exact copy)
  identical_text_other         same text, rating or date differ (an edited copy; still one review)
  distinct_same_rating_date    different texts with the same rating and date (distinct reviews)
  distinct_other               different texts and ratings or dates (repeat reviews)
  unverifiable                 empty texts, or fewer than two review rows found
The first two are duplicates; the last two are separate reviews of the same product, which the binary user x product
matrix of the experiments records as one interaction either way. Collapsing therefore changes no retained pair; it
only fixes the date of the pair (the earliest, which the temporal split uses).

Also counts the cross-market copies (one reviewer ID, ASIN and text in >= 2 study markets of a category) that
src/data.py now keeps in a single market (results/xmarket_copies.json has the pairs dropped).
Writes paper/raw/aud_dups.csv (per file) and paper/raw/aud_dups.json (totals). Needs the raw zip and XMARKET_FULL.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE, already_done
if already_done(f'{PAPER_RAW}/aud_dups.json', json_key='totals'): _sys.exit(0)

import sys, json, gzip, io, zipfile, hashlib, collections
sys.path.insert(0, SRC)
import pandas as pd
import data as D

CLASSES = ['identical_text_rating_date', 'identical_text_other', 'distinct_same_rating_date', 'distinct_other', 'unverifiable']


def text_hash(rv):
    t = ' '.join((rv.get('reviewText') or '').split()) + '\x1f' + ' '.join((rv.get('summary') or '').split())
    return None if t == '\x1f' else hashlib.sha1(t.encode('utf-8')).hexdigest()[:16]


def classify(rows):
    if len(rows) < 2 or any(r[0] is None for r in rows): return 'unverifiable'
    texts = {r[0] for r in rows}; same_rd = len({r[1:] for r in rows}) == 1
    if len(texts) == 1: return 'identical_text_rating_date' if same_rd else 'identical_text_other'
    return 'distinct_same_rating_date' if same_rd else 'distinct_other'


per_file, xm = [], collections.Counter()
with zipfile.ZipFile(D.ZIP_PATH) as zf:
    for category in D.CATEGORIES:
        multi = D._multi_market_counts(zf, category); musers = set(multi['user'])
        texts = {}                                           # (category, user, item) -> {market: set(text)}
        for market in D.MARKETS:
            try: raw = gzip.decompress(zf.read(f'xmarket_categories/ratings_{market}_{category}.txt.gz'))
            except KeyError: continue
            if not raw.strip(): continue
            df = pd.read_csv(io.BytesIO(raw), sep=' ', header=None, usecols=[0, 1], names=['user', 'item'], dtype='category')
            del raw
            dupmask = df.duplicated(['user', 'item'], keep=False)
            rep = df[dupmask]
            want = set(rep.user.astype(str) + ' ' + rep.item.astype(str))
            n_rows, n_removed = len(df), int(dupmask.sum() - len(want))
            del df, rep, dupmask
            found = collections.defaultdict(list)
            path = D.FULL_DIR / market / category / f'reviews_{market}_{category}.json.gz'
            n_rev = 0
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    for rv in json.loads(line):
                        n_rev += 1
                        u, a = rv.get('reviewerID'), rv.get('asin'); k = f'{u} {a}'
                        h = None
                        if k in want:
                            h = text_hash(rv); found[k].append((h, rv.get('overall'), rv.get('cleanReviewTime')))
                        if u in musers:
                            h = h if h is not None else text_hash(rv)
                            if h is not None: texts.setdefault((u, a), {}).setdefault(market, set()).add(h)
            cls = collections.Counter(classify(found.get(k, [])) for k in want)
            per_file.append(dict(market=market, category=category, raw_rows=n_rows, review_rows=n_rev,
                                 repeated_pairs=len(want), rows_removed_by_dedup=n_removed,
                                 **{c: int(cls.get(c, 0)) for c in CLASSES}))
            print(market, category, dict(cls), flush=True)
        for (u, a), by_m in texts.items():
            if len(by_m) < 2: continue
            xm['pairs_in_ge2_markets'] += 1
            common = collections.Counter(h for s in by_m.values() for h in s)
            xm['with_a_text_shared_by_ge2_markets' if any(v > 1 for v in common.values()) else 'only_distinct_texts'] += 1

T = pd.DataFrame(per_file)
assert (T.raw_rows == T.review_rows).all(), 'review files and ratings files differ in row counts'
tot = {c: int(T[c].sum()) for c in ['raw_rows', 'repeated_pairs', 'rows_removed_by_dedup'] + CLASSES}
tot['share_duplicates'] = (tot['identical_text_rating_date'] + tot['identical_text_other']) / tot['repeated_pairs']
tot['share_same_rating_date_but_distinct_text'] = tot['distinct_same_rating_date'] / tot['repeated_pairs']
out = {'totals': tot, 'cross_market_same_category': dict(xm)}
T.to_csv(f'{PAPER_RAW}/aud_dups.csv', index=False, lineterminator='\n')
json.dump(out, open(f'{PAPER_RAW}/aud_dups.json', 'w', encoding='utf-8', newline='\n'), indent=1)
print(json.dumps(out, indent=1))
