"""RQ1: popularity vs co-interaction pattern agreement across markets (null model, popularity-adjusted measures,
two-sided volume matching; also run on the ratings >= 4 variant).

Keeps the definitions of legacy_v3/paper_code/travel3.py:
  * target-donor pairs of the same category (results/overlap_random{TAG}.parquet, channel XM);
  * shared products = products with at least one interaction in both contexts (all rows of the
    interaction table, train+val+test, exactly as travel3.py reads it); pairs need >= 10 shared products;
  * local users split into halves A/B by Bernoulli(0.5) per user, 3 half-splits per pair; half A is the
    reference of every comparison;
  * popularity measures: Spearman correlation of per-product interaction counts; top-10 overlap;
  * pattern measures: co-interaction cosine (cosine between the upper triangles of the cosine-normalised
    co-occurrence matrices, diagonal removed); 5-NN neighbour overlap on raw co-occurrence counts
    (products with >= 5 co-occurring products in both samples; pair needs >= 5 such products).

New (per pair, averaged over the 3 half-splits and over the draws within a split):
  loc_*     half B vs A                          for_*     all foreign data vs A
  mloc_*    half B downsampled to V vs A         mfor_*    foreign downsampled to V vs A   (two-sided matching)
  null_*    curveball-randomised foreign vs A    mnull_*   randomised foreign, same users as mfor_ (matched null)
  Measures: sp, top10 (popularity), cos, nbr5 (patterns), ppmi (cosine of PPMI upper triangles),
            resid (Pearson of upper triangles of c_ij - n_i n_j / N)

Volume matching: V = min(shared-product interactions of half B, of all foreign data). Both half B and the
foreign data are reduced to at most V by adding whole users in random order and stopping BEFORE the first user
who would push the total above V (fixes the off-by-one of travel3.py:58, which kept that user). The side whose
total equals V is used in full. If the first user alone exceeds V the draw is empty and is skipped (NaN).
10 draws per half-split.

Null model: degree-preserving randomisation of the foreign user x shared-product binary matrix with the
curveball algorithm (Strona et al. 2014). Trades are executed in rounds of disjoint random user pairs
(the 'global curveball' schedule of Carstens, Berger & Strona 2018); the number of rounds is the smallest that
gives >= 5 x (number of users) trades. Each trade pools the products held by only one of the two users and
redistributes them at random, keeping both users' counts; this preserves every user's activity and every
product's interaction count. 10 draws per pair (fixed seeds), shared across the 3 half-splits.

Usage: python rq1_agree.py TAG [TIME_LIMIT_S]      TAG = '' (paper data) or '_r4' (ratings >= 4)
Output: paper/raw/rq1_agree{TAG}.jsonl, one record per pair; finished pairs are skipped on restart.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import sys, os, json, time
sys.path.insert(0, SRC)
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from util import stable_seed

N_SPLITS, N_MATCH, N_NULL, TRADES_PER_USER = 3, 10, 10, 5
MIN_SHARED, NBR_K, NBR_MIN_DEG, NBR_MIN_PRODUCTS = 10, 5, 5, 5
MEASURES = ['sp', 'top10', 'cos', 'nbr5', 'ppmi', 'resid']


# ----------------------------------------------------------------------------- per-sample statistics
def stats(uu, ii, n_items):
    """uu: user codes, ii: product indices (0..n_items-1) of one sample's shared-product interactions (binary)."""
    if len(uu) == 0:
        return None
    uu = pd.factorize(uu)[0]
    N = int(uu.max()) + 1
    R = sp.csr_matrix((np.ones(len(uu)), (uu, ii)), shape=(N, n_items))
    C = (R.T @ R).tocsr()
    pop = C.diagonal().astype(float)
    C.setdiag(0); C.eliminate_zeros(); C.sort_indices()
    # cosine-normalised co-occurrence, upper triangle
    dg = sp.diags(1 / np.maximum(np.sqrt(pop), 1e-12))
    S = sp.triu(dg @ C @ dg, k=1).tocsr()
    U = sp.triu(C, k=1).tocoo()                        # raw co-occurrence, upper triangle
    r, c, v = U.row, U.col, U.data.astype(float)
    ppmi = np.maximum(0.0, np.log(v * N / (pop[r] * pop[c])))
    P = sp.csr_matrix((ppmi, (r, c)), shape=(n_items, n_items)); P.eliminate_zeros()
    U = U.tocsr()
    # top-5 neighbours per product on raw counts: highest count first, ties to the lower product index
    deg = np.diff(C.indptr)
    rows = np.repeat(np.arange(n_items), deg)
    o = np.lexsort((C.indices, -C.data, rows))
    rank = np.arange(len(o)) - C.indptr[rows[o]]
    keep = o[rank < NBR_K]
    top5 = rows[keep].astype(np.int64) * n_items + C.indices[keep]
    return dict(pop=pop, N=N, S=S, U=U, P=P, deg=deg, top5=top5, n=n_items,
                ss=float(S.multiply(S).sum()), pp=float(P.multiply(P).sum()))


def _T(v):
    """sum_{i<j} v_i v_j"""
    return (v.sum() ** 2 - (v * v).sum()) / 2.0


def _quad(U, x):
    """sum over upper-triangle entries of U_ij * x_i * x_j"""
    Uc = U.tocoo()
    return float((Uc.data * x[Uc.row] * x[Uc.col]).sum())


def resid_corr(a, b):
    """Pearson correlation over all i<j of (c_ij - n_i n_j / N) between two samples, computed in closed form
    (the residual matrix is dense; this avoids building it)."""
    n = a['n']; M = n * (n - 1) / 2.0
    na, nb, Na, Nb = a['pop'], b['pop'], a['N'], b['N']
    Sx = a['U'].sum() - _T(na) / Na
    Sy = b['U'].sum() - _T(nb) / Nb
    Sxx = a['U'].multiply(a['U']).sum() - 2 * _quad(a['U'], na) / Na + _T(na * na) / Na ** 2
    Syy = b['U'].multiply(b['U']).sum() - 2 * _quad(b['U'], nb) / Nb + _T(nb * nb) / Nb ** 2
    Sxy = (a['U'].multiply(b['U']).sum() - _quad(a['U'], nb) / Nb - _quad(b['U'], na) / Na
           + _T(na * nb) / (Na * Nb))
    vx, vy = Sxx - Sx * Sx / M, Syy - Sy * Sy / M
    if vx <= 1e-12 * max(1.0, Sxx) or vy <= 1e-12 * max(1.0, Syy):
        return np.nan
    return float((Sxy - Sx * Sy / M) / np.sqrt(vx * vy))


NAN = {m: np.nan for m in MEASURES}


def agree(a, b):
    if a is None or b is None:
        return dict(NAN)
    k = min(10, a['n'])
    t10 = len(set(np.argsort(-a['pop'], kind='stable')[:k]) & set(np.argsort(-b['pop'], kind='stable')[:k])) / k
    cs = float(a['S'].multiply(b['S']).sum() / np.sqrt(a['ss'] * b['ss'])) if a['ss'] > 0 and b['ss'] > 0 else np.nan
    pm = float(a['P'].multiply(b['P']).sum() / np.sqrt(a['pp'] * b['pp'])) if a['pp'] > 0 and b['pp'] > 0 else np.nan
    ok = (a['deg'] >= NBR_MIN_DEG) & (b['deg'] >= NBR_MIN_DEG)
    if ok.sum() >= NBR_MIN_PRODUCTS:
        common = np.intersect1d(a['top5'], b['top5'])
        cnt = np.bincount(common // a['n'], minlength=a['n'])
        nb = float(np.mean(cnt[ok] / NBR_K))
    else:
        nb = np.nan
    with np.errstate(all='ignore'):
        rho = spearmanr(a['pop'], b['pop']).correlation
    return dict(sp=float(rho), top10=t10, cos=cs, nbr5=nb, ppmi=pm, resid=resid_corr(a, b))


# ----------------------------------------------------------------------------- curveball null
def curveball(uu, ii, rng):
    """Degree-preserving randomisation of a binary user x product edge list (uu, ii are 0-based codes).
    Returns the new product index for every edge, edges keep their user (row sums fixed)."""
    nu = int(uu.max()) + 1
    if nu < 2:
        return ii.copy(), 0
    per_round = nu // 2
    rounds = int(np.ceil(TRADES_PER_USER * nu / per_round))
    ii = ii.copy()
    for _ in range(rounds):
        perm = rng.permutation(nu)
        pair = np.full(nu, -1); side = np.zeros(nu, np.int8)
        pair[perm[:2 * per_round:2]] = np.arange(per_round); pair[perm[1:2 * per_round:2]] = np.arange(per_round)
        side[perm[1:2 * per_round:2]] = 1
        ep = pair[uu]; es = side[uu]
        act = ep >= 0
        # a product held by both users of a pair stays; products held by one user are tradable
        key = ep[act].astype(np.int64) * (ii.max() + 1) + ii[act]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        trad = np.flatnonzero(act)[cnt[inv] == 1]
        if len(trad) == 0:
            continue
        tp, ts = ep[trad], es[trad]
        # shuffle the tradable products within each pair, then refill side-0 slots first, side-1 slots next
        slots = trad[np.lexsort((ts, tp))]                        # edge slots grouped by pair, side 0 first
        prods = ii[trad][np.lexsort((rng.random(len(trad)), tp))]  # products grouped by pair, random order
        ii[slots] = prods
    return ii, rounds * per_round


# ----------------------------------------------------------------------------- matching helper
def take_users(user_order, counts, V):
    """Users added in random order; stop before the first user who would push the total above V."""
    cs = np.cumsum(counts)
    k = int(np.searchsorted(cs, V, side='right'))          # number of users with cumulative total <= V
    return user_order[:k], int(cs[k - 1]) if k else 0


def run_pair(tag, tm, tc, dm, T, Dn):
    items = np.intersect1d(T.i.unique(), Dn.i.unique())
    if len(items) < MIN_SHARED:
        return None
    n = len(items)
    Tsh = T[T.i.isin(items)]; Dsh = Dn[Dn.i.isin(items)]
    tu = Tsh.u.values; ti = np.searchsorted(items, Tsh.i.values)
    fu_codes = pd.factorize(Dsh.u.values)[0]; fi = np.searchsorted(items, Dsh.i.values)
    nF = len(fi); n_fu = int(fu_codes.max()) + 1
    Fs = stats(fu_codes, fi, n)
    # ---- null draws (shared across half-splits)
    null_ii, null_S, trades = [], [], 0
    for k in range(N_NULL):
        rng = np.random.default_rng(stable_seed('rq1', tag, tm, tc, dm, 'null', k))
        nii, tr = curveball(fu_codes, fi, rng); trades = tr
        null_ii.append(nii); null_S.append(stats(fu_codes, nii, n))
    # sanity: popularity and user activity preserved
    chk_pop = max(float(np.abs(s['pop'] - Fs['pop']).max()) for s in null_S)
    chk_dup = max(int(pd.Series(fu_codes.astype(np.int64) * n + x).duplicated().sum()) for x in null_ii)
    fcount = np.bincount(fu_codes, minlength=n_fu)
    users = T.u.unique()
    rec_splits = []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(stable_seed('rq1', tag, tm, tc, dm, 'split', s))
        h = rng.random(len(users)) < 0.5
        inA = np.isin(tu, users[h])
        sa = stats(tu[inA], ti[inA], n); sb = stats(tu[~inA], ti[~inA], n)
        nA, nB = int(inA.sum()), int((~inA).sum())
        out = {'nA': nA, 'nB': nB}
        for pre, b in [('loc_', sb), ('for_', Fs)]:
            out.update({pre + m: v for m, v in agree(sa, b).items()})
        nl = pd.DataFrame([agree(sa, x) for x in null_S]).mean()
        out.update({'null_' + m: nl[m] for m in MEASURES})
        out['null_sd_cos'] = float(pd.DataFrame([agree(sa, x) for x in null_S])['cos'].std()) if N_NULL > 1 else np.nan
        # ---- two-sided volume matching
        V = min(nB, nF)
        bu_codes = pd.factorize(tu[~inA])[0]; bi = ti[~inA]
        bcount = np.bincount(bu_codes)
        acc, vb, vf = [], [], []
        for k in range(N_MATCH):
            r2 = np.random.default_rng(stable_seed('rq1', tag, tm, tc, dm, 'match', s, k))
            ob = r2.permutation(len(bcount)); of = r2.permutation(n_fu)
            tb, ab = take_users(ob, bcount[ob], V); tf, af = take_users(of, fcount[of], V)
            mb = np.isin(bu_codes, tb); mf = np.isin(fu_codes, tf)
            sbm = stats(bu_codes[mb], bi[mb], n) if ab > 0 else None
            sfm = stats(fu_codes[mf], fi[mf], n) if af > 0 else None
            snm = stats(fu_codes[mf], null_ii[k][mf], n) if af > 0 else None
            row = {}
            for pre, b in [('mloc_', sbm), ('mfor_', sfm), ('mnull_', snm)]:
                row.update({pre + m: v for m, v in agree(sa, b).items()})
            acc.append(row); vb.append(ab); vf.append(af)
        mm = pd.DataFrame(acc).mean()   # NaN draws (empty sample or undefined measure) are skipped
        out.update(mm.to_dict())
        out.update({'V': V, 'mVB': float(np.mean(vb)), 'mVF': float(np.mean(vf)),
                    'mVB_min': int(np.min(vb)), 'mVF_min': int(np.min(vf)),
                    'm_empty': int(sum(1 for a, b in zip(vb, vf) if a == 0 or b == 0))})
        rec_splits.append(out)
    df = pd.DataFrame(rec_splits)
    rec = dict(t=f'{tm} {tc}', target_market=tm, donor_market=dm, category=tc, n_shared=n,
               nF=nF, n_foreign_users=n_fu, n_local_users=int(len(users)), n_local_shared_users=int(len(np.unique(tu))),
               nT_shared=int(len(Tsh)), nT=int(len(T)), local_share_on_shared=float(len(Tsh) / len(T)),
               null_trades=int(trades), null_pop_maxdiff=chk_pop, null_dup_edges=chk_dup)
    rec.update({k: (None if pd.isna(v) else float(v)) for k, v in df.mean().items()})
    # per-split raw values of the main measures, for transparency
    for pre in ['loc_', 'for_', 'mloc_', 'mfor_', 'null_']:
        for m in MEASURES:
            rec['splits_' + pre + m] = [None if pd.isna(x) else float(x) for x in df[pre + m]]
    return rec


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ''
    lim = float(sys.argv[2]) if len(sys.argv) > 2 else 1e12
    T0 = time.time()
    out = f'{PAPER_RAW}/rq1_agree{tag}.jsonl'
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding='utf-8'):
            r = json.loads(l); done.add((r['t'], r['donor_market']))
    d = pd.read_parquet(f'{DATA}/interactions{tag}.parquet', columns=['market', 'category', 'u', 'i'])
    o = pd.read_parquet(f'{RESULTS}/overlap_random{tag}.parquet'); o = o[o.channel == 'XM']
    G = {k: g for k, g in d.groupby(['market', 'category'])}
    pairs = sorted(zip(o.target_market, o.target_category, o.donor_market),
                   key=lambda p: (len(G[(p[0], p[1])]) + len(G[(p[2], p[1])]), p))
    if os.environ.get('ONLY'):
        pairs = [p for p in pairs if f'{p[0]} {p[1]} {p[2]}' in os.environ['ONLY'].split(',')]
    nskip = 0
    with open(out, 'a', encoding='utf-8', newline='\n') as fh:
        for tm, tc, dm in pairs:
            if (f'{tm} {tc}', dm) in done:
                continue
            if time.time() - T0 > lim:
                print('deadline', flush=True); return
            t1 = time.time()
            rec = run_pair(tag, tm, tc, dm, G[(tm, tc)], G[(dm, tc)])
            if rec is None:
                nskip += 1; continue
            rec['sec'] = round(time.time() - t1, 2)
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            print(f'{tm} {tc} <- {dm}: n={rec["n_shared"]} {rec["sec"]}s', flush=True)
    print('finished; skipped (<10 shared products):', nskip, flush=True)


if __name__ == '__main__':
    main()
