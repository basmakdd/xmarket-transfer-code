"""Experiment-by-context accounting (Supplementary Table S3). Generic and re-runnable: re-run whenever result files change.

Reads every result file in paper/raw (*.jsonl, *.parquet, travel3_*.pkl, df.pkl) and reports, per experiment x context:
included?, exclusion reason, completed repetitions, number of records / errors.

Experiment identity = file stem with a trailing worker index removed (ma_ItemKNN_random_0/_1/_9 -> ma_ItemKNN_random), split further by
'regime' (ma_*) or 'arm' (sc_*) when present. Old families (fr_, l2_, single, travel3, df.pkl core grid) are included as well.
Per-family conventions (fields that define a repetition and an arm):
  ma   rep=seed, arms=(own_frac, foreign), sub-experiment=regime   (mustadd.py)
  sc   rep=seed, arms=(f, foreign, fb),   sub-experiment=arm       (scarcity2.py; arm='skip' records = exclusions)
  l2,l3 rep=seed, arms=(foreign,)                                  (launch2.py / launch3.py; l3_skip_* = exclusions with reasons)
  fr   arms=(foreign,)                                             (run_full.py)
  aud  arms=(arm,), pairs=donor_market                             (single2.py)
  rq1, travel3, single  pair-level (one record per target-donor pair)
  rq2  one record per context (rq2_run.py)
  nullab one record per context (nullab_run.py); files ending in _nvc are runs on the _nvc data version (universe =
       the 74 paper-data contexts, contexts missing from _nvc count as dropped by the variant merge)
  core df.pkl: local = k==0, pooled = order=='full' & frac==1, per (model, channel)
  other families: rep=seed|rep if present, arms=(foreign,) if present.
A record counts as completed if it has no 'err'/'error' value and an outcome (ndcg, ndcg@10, zf, any of the RQ1 measures loc_sp/loc_top10/
loc_cos/loc_ppmi/loc_resid, or any record for rq2)
is not missing. A repetition is complete for a context if the context has every arm that appears (in any context) for that
repetition value of the same experiment. A context is INCLUDED if it has >= 1 complete repetition (pair experiments: >= 1 completed
pair; rq2: a completed record). Otherwise the exclusion reason is taken from skip records / error values, else inferred:
'no same-category foreign donor', 'EASE skips catalogs > 4,000 products', or 'not run (cost cap / budget / still running)';
'incomplete' if some but not all arms exist. Contexts without validation users are flagged (they use grid[0]).
Context universe: the 74 contexts of results/contexts.parquet (calendar split: results/contexts_calendar.parquet).
Outputs: paper/raw/aud_accounting.csv (long), paper/raw/aud_accounting_summary.csv, paper/tables_S3.tex.

It also reads the sensitivity runs in paper/raw/sens_<tag>/ (experiment name 'sens_<tag>/<stem>'; universe = the 74
paper-data contexts; overlap/contexts/interactions files of tag _<tag>), mkt_skips.jsonl (category-level skips of mkt_run.py,
expanded to the category's targets) and gives every exclusion one reason code:
  merge     dropped by the variant merge (context fails the core thresholds in that data version)
  nodonor   no same-category foreign donor
  rq1rule   no donor shares >= 10 products (RQ1 pair rule)
  catalog   catalog > 4,000 products (EASE)
  launch    fewer than 10 launched-item test users in every launch set (launch2/3 skip records)
  launchp   too few launched / eligible products (launch3 skip records)
  costrule  pre-specified cost rule (mkt: first pooled epoch > 15 s)
  subset    outside the experiment's pre-specified context subset (rq2_*_grid: IN A&C + contexts without validation users;
            sens_*/sc_UserKNN: the 11 UserKNN anomaly contexts)
  error     error records only
  incomplete some but not all arms of every repetition
  cost      not run: cost cap / compute budget (all experiments have finished, so a context without any record that
            matches none of the codes above was not run for cost)
Not exclusions but reported per experiment: 'untuned' = included contexts without validation users (grid[0]); for scarcity
experiments, 'gap' = included contexts whose gap-closed share G is undefined at f = 0.10 (gap = complete-data local - mean
local at f = 0.10 <= 0.01, foreign 0, no fallback), i.e. dropped from G summaries by the gap rule.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, SRC, RESULTS, DATA, PAPER_RAW, CODE

import os, re, glob, json, collections
import numpy as np, pandas as pd

SKIP_FILES = {'aud_rawcache.pkl', 'pairs.pkl', 'pairs2.pkl', 'hist.pkl'}
FAM = {'ma': dict(rep='seed', arms=['own_frac', 'foreign'], sub='regime'),
       'sc': dict(rep='seed', arms=['f', 'foreign', 'fb'], sub='arm'),
       'l2': dict(rep='seed', arms=['foreign']), 'l3': dict(rep='seed', arms=['foreign']),
       'fr': dict(arms=['foreign']), 'aud': dict(arms=['arm'], pair='donor_market'),
       'rq1': dict(pair='donor_market'), 'travel3': dict(pair='donor_market'), 'single': dict(pair='donor'),
       'rq2': dict(per_context=True), 'nullab': dict(per_context=True), 'core': dict(arms=['arm']), 'mkt': dict(arms=['variant'])}
VARIANT_SUFFIX = re.compile(r'_(nvxd|nvc|nv|xd)$')   # data-version suffix of result files kept in paper/raw itself (nullab_*_nvc)
REASON = {'merge': 'dropped by variant merge', 'nodonor': 'no same-category foreign donor', 'rq1rule': 'no donor shares at least 10 products (RQ1 rule)',
          'catalog': 'catalog over 4,000 products (EASE)', 'launch': 'fewer than 10 launched-item test users',
          'launchp': 'too few launched/eligible products', 'costrule': 'cost rule (first pooled epoch over 15 s)',
          'subset': 'outside pre-specified subset', 'error': 'error', 'incomplete': 'incomplete (not all arms of any repetition)',
          'cost': 'not run: cost cap / compute budget'}
SUBSETS = [(re.compile(r'^rq2_.*_grid$'), 'subset'), (re.compile(r'^sens_[a-z]+/sc_UserKNN'), 'subset')]


def skip_code(txt):
    t = str(txt).lower()
    if 'catalog' in t or 'ease_too_large' in t: return 'catalog'
    if 'launched-item test users' in t: return 'launch'
    if 'launched products' in t or 'eligible products' in t: return 'launchp'
    if 'first epoch' in t: return 'costrule'
    return 'error'
# RQ1 pair records carry several agreement measures; any one of them marks the pair as computed (loc_cos alone is
# missing (NaN) for some pairs, e.g. ca Toys_and_Games <- in in the _nv/_nvxd/_nvc data versions, whose other measures exist).
OUTCOME = ['ndcg', 'ndcg@10', 'zf', 'loc_sp', 'loc_top10', 'loc_cos', 'loc_ppmi', 'loc_resid']


def ctx_universe(split, tag=''):
    fn = f'{RESULTS}/contexts_calendar.parquet' if split == 'calendar' else f'{RESULTS}/contexts{tag}.parquet'
    if not os.path.exists(fn): fn = f'{RESULTS}/contexts.parquet'
    c = pd.read_parquet(fn); c['t'] = c.market + ' ' + c.category
    return c


_SH = {}
def max_shared(tag=''):
    """Per target context: the largest number of products shared with a same-category foreign context (all interactions),
    the quantity behind the RQ1 rule (travel3.py / rq1 scripts compute a pair only if it is >= 10)."""
    if tag not in _SH:
        fn = f'{DATA}/interactions{tag}.parquet'
        cache = f'{REPO}/paper/results/max_shared_products{tag or "_orig"}.csv'   # shipped, so S3 rebuilds without data/
        if os.path.exists(fn):
            d = pd.read_parquet(fn, columns=['market', 'category', 'i'])
            it = {k: set(g.i.unique()) for k, g in d.groupby(['market', 'category'])}
            _SH[tag] = {f'{m} {c}': max([len(it[(m, c)] & it[(m2, c2)]) for (m2, c2) in it if c2 == c and m2 != m] or [0]) for (m, c) in it}
            pd.DataFrame(sorted(_SH[tag].items()), columns=['t', 'max_shared']).to_csv(cache, index=False, encoding='utf-8', lineterminator='\n')
        elif os.path.exists(cache):
            c = pd.read_csv(cache, encoding='utf-8'); _SH[tag] = dict(zip(c.t, c.max_shared.astype(int)))
        else:
            _SH[tag] = {}
    return _SH[tag]


def load_records():
    recs = []
    files = sorted(glob.glob(f'{PAPER_RAW}/*.jsonl') + glob.glob(f'{PAPER_RAW}/*.parquet') + glob.glob(f'{PAPER_RAW}/travel3_*.pkl') + glob.glob(f'{PAPER_RAW}/df.pkl'))
    files += sorted(glob.glob(f'{PAPER_RAW}/sens_*/*.jsonl'))
    used, skipped = [], []
    for fn in files:
        b = os.path.basename(fn)
        sd = os.path.basename(os.path.dirname(fn))
        vtag = '_' + sd[5:] if sd.startswith('sens_') else ''
        pre = f'{sd}/' if vtag else ''
        if b == 'mkt_skips.jsonl':
            df = pd.read_json(fn, lines=True, dtype=False)
            c = ctx_universe('random')
            df = df.merge(c[['t', 'category']], left_on='unit', right_on='category')
            df = df.assign(exp='mkt_runs', family='mkt', skiprec=True, file=b, vtag='', ok=False)
            recs.append(df); used.append(b); continue
        if b in SKIP_FILES: continue
        stem = re.sub(r'\.(jsonl|parquet|pkl)$', '', b)
        is_skip = '_skip_' in stem
        stem2 = stem.replace('_skip_', '_')
        if b == 'df.pkl':
            fam = 'core'
        else:
            fam = stem2.split('_')[0]
        base = re.sub(r'_\d+$', '', stem2)
        if fam == 'nullab' and VARIANT_SUFFIX.search(base):   # e.g. nullab_ItemKNN_random_nvc: data version _nvc
            vtag = '_' + VARIANT_SUFFIX.search(base).group(1)
        try:
            if fn.endswith('.jsonl'): df = pd.read_json(fn, lines=True, dtype=False) if os.path.getsize(fn) else pd.DataFrame()
            elif fn.endswith('.parquet'): df = pd.read_parquet(fn)
            else: df = pd.read_pickle(fn)
        except Exception as e:
            skipped.append((b, f'unreadable: {e}')); continue
        if df.empty: skipped.append((b, 'empty')); continue
        if 't' not in df.columns:
            if 'target_market' in df.columns and ('target_category' in df.columns or 'category' in df.columns):
                df['t'] = df.target_market + ' ' + df.get('target_category', df.get('category'))
            else:
                skipped.append((b, 'no context field')); continue
        if fam == 'core':
            df = df[df.error.isna()] if 'error' in df else df
            loc = df[df.k == 0].assign(arm='local')
            parts = []
            for ch in sorted(set(df.channel) - {'NONE'}):
                pool = df[(df.channel == ch) & (df.order == 'full') & (df.frac == 1)].assign(arm='pool')
                parts.append(pd.concat([loc, pool]).assign(exp=lambda x: 'core_' + x.model + '_random_' + ch))
            df = pd.concat(parts)
        else:
            df = df.copy(); df['exp'] = pre + base
            if 'model' in df.columns and base in ('single', 'travel3'): pass
            sub = FAM.get(fam, {}).get('sub')
            if sub and sub in df.columns:
                m = df[sub].notna() & (df[sub] != 'skip')
                df.loc[m, 'exp'] = pre + base + '|' + sub + '=' + df.loc[m, sub].astype(str)
                if sub == 'arm' and (df[sub] == 'skip').any():
                    df.loc[df[sub] == 'skip', 'exp'] = pre + base + '|*'
        df['family'] = fam; df['skiprec'] = is_skip | (df.get('arm', pd.Series(index=df.index, dtype=object)) == 'skip')
        df['file'] = pre + b; df['vtag'] = vtag
        df['ok'] = is_ok(df)              # per file, before columns of other families are mixed in
        recs.append(df); used.append(b)
    return pd.concat(recs, ignore_index=True, sort=False), used, skipped


def is_ok(df):
    ok = pd.Series(True, index=df.index)
    for c in ('err', 'error'):
        if c in df: ok &= df[c].isna() | (df[c].astype(str).isin(['None', 'nan', '']))
    has = pd.Series(False, index=df.index); anyc = False
    for c in OUTCOME:
        if c in df: has |= df[c].notna(); anyc = True
    if anyc: ok &= has
    return ok & ~df.skiprec


def main():
    R, used, skipped = load_records()
    print('files used:', len(used)); print('files skipped:', skipped)
    ov = {s: pd.read_parquet(f'{RESULTS}/overlap_{s}.parquet') for s in ('random', 'temporal', 'calendar') if os.path.exists(f'{RESULTS}/overlap_{s}.parquet')}
    for fn in glob.glob(f'{RESULTS}/overlap_*_*.parquet'):   # tagged variants, e.g. overlap_random_r4
        ov[os.path.basename(fn)[8:-8]] = pd.read_parquet(fn)
    xm = {s: o[o.channel == 'XM'].assign(t=lambda x: x.target_market + ' ' + x.target_category).groupby('t').size() for s, o in ov.items()}
    rows = []
    # expand sub-experiment skip records ("base|*") to every sub-experiment of that base
    star = R[R.exp.str.endswith('|*')]
    for exp, g in R[~R.exp.str.endswith('|*')].groupby('exp'):
        fam = g.family.iloc[0]; cfg = FAM.get(fam, {})
        split = g.split.dropna().iloc[0] if 'split' in g and g.split.notna().any() else 'random'
        model = g.model.dropna().iloc[0] if 'model' in g and g.model.notna().any() else ''
        mt = re.search(r'(_r4|_seed\d+)(?=$|_|\|)', exp)
        vtag = g.vtag.iloc[0] if 'vtag' in g and isinstance(g.vtag.iloc[0], str) else ''
        tag = vtag or (mt.group(1) if mt else '')
        U = ctx_universe(split, '' if vtag else tag)
        Uv = ctx_universe(split, tag) if vtag else U
        vset = set(Uv.t)
        items = dict(zip(Uv.t, Uv['items'])); nval = dict(zip(Uv.t, Uv.get(f'n_{split}_val', Uv.get('n_random_val', Uv.get('val')))))   # calendar: 'val' = validation rows
        split_key = split + tag if (split + tag) in xm else split
        subset_code = next((c for rx, c in SUBSETS if rx.search(exp)), None)
        gapset = None
        if fam == 'sc' and '|arm=' in exp and not exp.endswith('|arm=F'):
            Rb = R[(R.exp == exp.split('|')[0] + '|arm=F') & R.ok]
            Rb = Rb[(Rb.foreign.astype(float) == 0) & (Rb.fb.astype(float) == 0)] if 'fb' in Rb else Rb[Rb.foreign.astype(float) == 0]
            full = Rb.groupby('t').ndcg.first()
            q = g[g.ok & (g.f.astype(float).round(3) == 0.1) & (g.foreign.astype(float) == 0) & (g.fb.astype(float) == 0)]
            y0 = q.groupby('t').ndcg.mean()
            gap = (full - y0).dropna()
            gapset = set(gap.index[gap <= 0.01])
        sk = pd.concat([g[g.skiprec], star[star.exp.str.split('|').str[0] == exp.split('|')[0]]])
        okg = g[g.ok]
        repc = cfg.get('rep') if cfg.get('rep') in g.columns else ('seed' if 'seed' in g.columns and fam not in FAM else None)
        arms = [a for a in cfg.get('arms', ['foreign']) if a in g.columns]
        pair = cfg.get('pair') if cfg.get('pair') in g.columns else None
        if pair == 'donor': okg = okg.assign(donor=okg.donor.astype(str))
        sig = (okg[arms].astype(str).agg('|'.join, axis=1) if arms else pd.Series('', index=okg.index))
        rep = okg[repc].astype(str) if repc else pd.Series('0', index=okg.index)
        okg = okg.assign(_sig=sig, _rep=rep)
        if pair:  # arms of a pair experiment: local/donor; count pairs separately
            exp_arms = {r: set(x._sig) for r, x in okg.groupby('_rep')}
        else:
            exp_arms = {r: set(x._sig) for r, x in okg.groupby('_rep')}
        for t in U.t:
            gt = g[g.t == t]; ot = okg[okg.t == t]; st = sk[sk.t == t]
            reps_done = sorted(r for r, x in ot.groupby('_rep') if (set(x._sig) >= exp_arms.get(r, set())) and len(x)) if not pair else []
            n_pairs = int(ot[ot.get('arm', pd.Series('donor', index=ot.index)) != 'local'][pair].nunique()) if pair else None
            if fam == 'rq2' or cfg.get('per_context'):
                inc = len(ot) > 0; reps_done = ['0'] if inc else []
            elif pair:
                inc = n_pairs > 0 and (('arm' not in ot) or (ot.get('arm') == 'local').any() or fam != 'aud')
            else:
                inc = len(reps_done) > 0
            reason = ''; code = ''
            if not inc:
                errs = collections.Counter(str(e) for c in ('err', 'error') if c in gt for e in gt[c].dropna() if str(e) not in ('None', 'nan', ''))
                if vtag and t not in vset: code = 'merge'; reason = REASON['merge'] + ' (fails core thresholds in ' + vtag + ')'
                elif len(st):
                    rs = st.get('reason', pd.Series(index=st.index, dtype=object)).fillna(st.get('err', pd.Series(index=st.index, dtype=object))).dropna().astype(str)
                    codes = sorted({skip_code(x) for x in rs}) or ['error']
                    code = codes[0] if len(codes) == 1 else '+'.join(codes)
                    reason = '; '.join(REASON.get(c, c) for c in codes)
                elif errs: code = 'error'; reason = 'error: ' + ', '.join(f'{k} x{v}' for k, v in errs.items())
                elif split_key in xm and t not in xm[split_key].index and not exp.endswith(('_XC', '_XX')): code = 'nodonor'
                elif fam in ('rq1', 'travel3') and max_shared(tag).get(t, 99) < 10: code = 'rq1rule'
                elif model == 'EASE' and items.get(t, 0) > 4000: code = 'catalog'
                elif len(gt): code = 'incomplete'
                elif subset_code: code = subset_code
                else: code = 'cost'
                reason = reason or REASON[code]
            rows.append(dict(experiment=exp, family=fam, model=model, split=split, context=t, included=bool(inc), reason=reason, reason_code=code,
                             gap_rule=bool(inc and gapset is not None and t in gapset), data_version=vtag or tag or 'paper',
                             completed_reps=len(reps_done) if not pair else None, reps=','.join(reps_done) if not pair else '',
                             pairs_done=n_pairs, pairs_xm=int(xm.get(split_key, pd.Series(dtype=int)).get(t, 0)) if pair else None,
                             n_records=len(gt), n_ok=len(ot), n_skip_records=len(st), no_val_users=bool(nval.get(t, 1) == 0),
                             files=','.join(sorted(set(gt.file))) if len(gt) else ''))
    A = pd.DataFrame(rows)
    A.to_csv(f'{PAPER_RAW}/aud_accounting.csv', index=False, lineterminator='\n')
    S = []
    for exp, g in A.groupby('experiment'):
        inc = g[g.included]
        rc = inc.completed_reps.dropna().astype(int).value_counts().sort_index().to_dict() if inc.completed_reps.notna().any() else {}
        ex = g[~g.included].groupby('reason').context.apply(lambda s: ', '.join(s)).to_dict()
        rec = dict(experiment=exp, split=g.split.iloc[0], model=g.model.iloc[0], data_version=g.data_version.iloc[0], universe=len(g), included=len(inc),
                   reps=('; '.join(f'{k} rep: {v}' for k, v in rc.items()) if rc else (f'pairs {int(inc.pairs_done.sum())}/{int(g.pairs_xm.sum())} XM' if g.pairs_done.notna().any() else '')),
                   excluded='; '.join(f'{r} ({len(c.split(", "))}): {c}' for r, c in ex.items()),
                   included_without_val_users=int(inc.no_val_users.sum()), untuned_contexts=', '.join(inc[inc.no_val_users].context),
                   gap_rule=int(inc.gap_rule.sum()), gap_rule_contexts=', '.join(inc[inc.gap_rule].context))
        for c in REASON: rec[f'n_{c}'] = int((g.reason_code == c).sum())
        S.append(rec)
    S = pd.DataFrame(S); S.to_csv(f'{PAPER_RAW}/aud_accounting_summary.csv', index=False, lineterminator='\n')
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
    print(S[['experiment', 'universe', 'included', 'reps', 'included_without_val_users', 'gap_rule']].to_string())
    for _, r in S.iterrows():
        if r.excluded: print(f'  {r.experiment}: {r.excluded}')
    cnt = A[~A.included].reason_code.value_counts().to_dict()
    print('exclusions by reason code (experiment x context rows):', cnt)
    # LaTeX
    def esc(s): return str(s).replace('_', '\\_').replace('&', '\\&').replace('>', '$>$').replace('|', ' $\\mid$ ')
    L = ['% Auto-generated by paper/code/accounting.py from the result files present at run time; re-run when experiments finish.',
         '{\\scriptsize', '\\begin{longtable}{p{3.6cm}cp{2.4cm}p{7.6cm}}',
         '\\caption{Experiment-by-context accounting (Supplementary Table S3). Included: contexts with at least one complete repetition (all arms present), out of the 74 paper-data contexts '
         '(calendar split: its own contexts). Repetitions: number of included contexts by completed repetitions (pair experiments: completed target--donor pairs / same-category foreign pairs); '
         'untuned: included contexts without validation users (first grid value); gap rule: included contexts left out of the gap-closed share $G$ at $f=0.10$ (gap $\\le 0.01$). '
         'Exclusion reasons: dropped by variant merge (sensitivity data versions sens\\_nv, sens\\_nvxd, sens\\_nvc: the context fails the core thresholds); no same-category foreign donor; '
         'no donor shares $\\ge$10 products (RQ1 rule); catalog $>$4,000 products (EASE); fewer than 10 launched-item test users / too few launched or eligible products (launch experiments); '
         'cost rule (pre-specified per-unit time limit); outside pre-specified subset; incomplete; not run: cost cap / compute budget (all experiments have finished, so remaining unrun contexts were excluded for cost).}\\label{tab:S3}\\\\',
         '\\toprule', 'Experiment & Included & Repetitions & Excluded contexts (reason) \\\\', '\\midrule', '\\endfirsthead', '\\toprule',
         'Experiment & Included & Repetitions & Excluded contexts (reason) \\\\', '\\midrule', '\\endhead']
    for _, r in S.iterrows():
        extra = []
        if r.included_without_val_users: extra.append(f'untuned {r.included_without_val_users}')
        if r.gap_rule: extra.append(f'gap rule {r.gap_rule}: {r.gap_rule_contexts}')
        reps = r.reps + (('; ' if r.reps else '') + '; '.join(extra) if extra else '')
        L.append(f'{esc(r.experiment)} & {r.included}/{r.universe} & {esc(reps)} & {esc(r.excluded) if r.excluded else "--"} \\\\')
    L += ['\\bottomrule', '\\end{longtable}}']
    open(f'{REPO}/paper/tables_S3.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote paper/tables_S3.tex,', len(S), 'experiments')


if __name__ == '__main__':
    main()
