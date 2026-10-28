#!/usr/bin/env python
"""One driver for the whole repository (works the same on Windows, macOS and Linux).

    python run.py data        [--only VARIANT ...] [--force]   preprocess the raw XMarket files (needs raw_data/)
    python run.py experiments [--only NAME ...] [--fresh]      re-run the experiments with the arguments used in the paper
    python run.py tables                                      rebuild every table (paper/tables_*.tex) from the shipped results
    python run.py figures                                     rebuild Figures 1-9 (paper/figures/) from the shipped results
    python run.py check       [--tol 1e-9]                    unit tests + consistency checks
    python run.py list                                        show the data variants and experiment groups

Every command can be started from any working directory. Experiment scripts are resumable: if a run is
interrupted, start the same command again and finished units are skipped.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CODE = os.path.join('paper', 'code')
UK = 'uk Books'
G6 = ('ca Grocery_and_Gourmet_Food,de Automotive,de Grocery_and_Gourmet_Food,de Toys_and_Games,'
      'in Arts_Crafts_and_Sewing,in Home_and_Kitchen')           # IN Arts & Crafts + the contexts without validation users
UK11 = ('de Arts_Crafts_and_Sewing,de Cell_Phones_and_Accessories,de Grocery_and_Gourmet_Food,de Industrial_and_Scientific,'
        'de Toys_and_Games,in Musical_Instruments,jp Cell_Phones_and_Accessories,mx Office_Products,uk Electronics,'
        'uk Industrial_and_Scientific,uk Office_Products')        # the 11 contexts of the UserKNN subsample check

# --------------------------------------------------------------------------------------------- helpers
def run(script, *args, env=None, cwd=None, check=True):
    """Run `python script args...` from the repository root (or cwd), with extra environment variables."""
    e = dict(os.environ)
    e.update({'PYTHONIOENCODING': 'utf-8', 'MPLBACKEND': 'Agg'})
    e.update(env or {})
    path = script if os.path.isabs(script) else os.path.join(ROOT, script)
    shown = ' '.join(f'{k}={v!r}' for k, v in (env or {}).items())
    print(f'\n>>> {shown + " " if shown else ""}python {os.path.relpath(path, ROOT)} {" ".join(map(str, args))}', flush=True)
    t = time.time()
    r = subprocess.run([PY, path, *map(str, args)], cwd=cwd or ROOT, env=e)
    print(f'<<< exit {r.returncode}, {time.time() - t:.0f} s', flush=True)
    if check and r.returncode != 0:
        sys.exit(f'command failed: {script} {args}')
    return r.returncode


def have_data(tag=''):
    return os.path.exists(os.path.join(ROOT, 'data', f'interactions{tag}.parquet'))


def have_zip():
    z = os.environ.get('XMARKET_ZIP', os.path.join(ROOT, 'raw_data', 'xmarket_categories.zip'))
    return os.path.exists(z)


def paper_raw():
    """Folder of the experiment records: paper/raw, or the folder named by the environment variable PAPER_RAW
    (a relative path is taken relative to the repository root, as in paper/code/common.py)."""
    p = os.environ.get('PAPER_RAW', os.path.join('paper', 'raw'))
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


# Requirement tags of table scripts and experiment groups -> (test, what to do if it is missing)
def _variant_of(tag):
    return next((n for n, v in VARIANTS.items() if v[0] == tag), tag)


def requirement(tag):
    """(present?, description) of one requirement tag."""
    if tag.startswith('data:'):                     # preprocessed interactions of a data variant (VARIANTS name)
        n = tag[5:]
        t = '_calendar' if n == 'calendar' else VARIANTS[n][0]
        return have_data(t), f'data/interactions{t}.parquet (python run.py data --only {n})'
    if tag == 'zip':
        return have_zip(), 'the raw XMarket zip (raw_data/xmarket_categories.zip or XMARKET_ZIP; see raw_data/README.md)'
    if tag == 'rawcache':
        return os.path.isdir(os.path.join(paper_raw(), 'aud_rawcache')), 'paper/raw/aud_rawcache/ (python run.py experiments --only audit)'
    if tag.startswith('varmap:'):
        n = tag[7:]
        return (os.path.exists(os.path.join(ROOT, 'results', f'variant_map{VARIANTS[n][0]}.parquet')),
                f'results/variant_map{VARIANTS[n][0]}.parquet (python run.py data --only {n})')
    if tag == 'torch':
        try:
            import importlib.util
            ok = importlib.util.find_spec('torch') is not None
        except Exception:
            ok = False
        return ok, 'PyTorch (pip install torch --index-url https://download.pytorch.org/whl/cpu)'
    raise ValueError(tag)


def missing(tags):
    return [d for ok, d in map(requirement, tags) if not ok]


# --------------------------------------------------------------------------------------------- data
# tag -> (environment of src/data.py, description). Every variant writes only files carrying its tag.
VARIANTS = {
    'paper':    ('', {}, 'paper data (random and chronological per-user splits, seed 0)'),
    'calendar': (None, {}, 'calendar-cutoff (prospective) split, plus launch candidates (src/calendar_split.py)'),
    'seed1':    ('_seed1', {'XM_SPLIT_SEED': '1'}, 'per-user split with seed 1'),
    'seed2':    ('_seed2', {'XM_SPLIT_SEED': '2'}, 'per-user split with seed 2'),
    'r4':       ('_r4', {'XM_MIN_RATING': '4'}, 'only ratings 4-5 count as interactions'),
    'nv':       ('_nv', {'XM_MERGE_VARIANTS': '1'}, 'product variants merged, users pooled over markets'),
    'xd':       ('_xd', {'XM_DROP_XMARKET_USERS': '1'}, 'reviewer IDs found in more than one market dropped'),
    'nvxd':     ('_nvxd', {'XM_MERGE_VARIANTS': '1', 'XM_DROP_XMARKET_USERS': '1'}, 'both of the above'),
    'nvc':      ('_nvc', {'XM_MERGE_VARIANTS': 'context'}, 'product variants merged within each market x category context'),
    'figdata':  (None, {}, 'counts per market x category before/after preprocessing (Figure 2 input)'),
}


def cmd_data(a):
    if not have_zip():
        sys.exit('raw data not found: put xmarket_categories.zip into raw_data/ (see raw_data/README.md) '
                 'or set XMARKET_ZIP=/path/to/zip')
    names = a.only or list(VARIANTS)
    for n in names:
        if n not in VARIANTS: sys.exit(f'unknown variant {n}; choose from {", ".join(VARIANTS)}')
    for n in names:
        tag, env, desc = VARIANTS[n]
        print(f'\n===== data variant {n}: {desc}')
        if n == 'calendar':
            if have_data('_calendar') and not a.force: print('exists, skipped (use --force to rebuild)'); continue
            run('src/calendar_split.py', cwd=os.path.join(ROOT, 'src'))
        elif n == 'figdata':
            run(os.path.join(CODE, 'figs', 'data_stats.py')); run(os.path.join(CODE, 'figs', 'data_table.py'))
        else:
            env = dict(env, XM_DATA_TAG=tag)
            if not have_data(tag) or a.force:
                run('src/data.py', env=env, cwd=os.path.join(ROOT, 'src'))
            else:
                print('data file exists, skipped (use --force to rebuild); overlap statistics are recomputed')
            run('src/overlap.py', env={'XM_DATA_TAG': tag}, cwd=os.path.join(ROOT, 'src'))


# --------------------------------------------------------------------------------------------- experiments
def rq2(model, split, arms, lim=30000, env=None):
    return (os.path.join(CODE, 'rq2_run.py'), [lim, model, split, arms], env or {})


def sens_runs(tag, rq2_jobs, launch=True, userknn=False, scarcity=False):
    """Runs on a data variant, through sens_wrap.py (tagged overlap/contexts files). Outputs go to <PAPER_RAW>/sens<tag>/,
    i.e. paper/raw/sens<tag>/ unless the environment variable PAPER_RAW names another folder."""
    W = os.path.join(CODE, 'sens_wrap.py')
    env = {'XM_DATA_TAG': tag, 'PAPER_RAW': os.path.join(paper_raw(), f'sens{tag}')}
    J = [(os.path.join(CODE, 'rq1_agree.py'), [tag], env)]
    J += [(W, [os.path.join(CODE, 'rq2_run.py'), '1e9', m, s, arms], env) for m, s, arms in rq2_jobs]
    if userknn:
        J += [(W, [os.path.join(CODE, 'rq2_run.py'), '1e9', 'UserKNN', 'random', 'local'], env),
              (W, [os.path.join(CODE, 'sens_scarcity.py'), 'userknn', '1e9'], dict(env, SC_CONTEXTS=UK11))]
    if scarcity:
        J += [(W, [os.path.join(CODE, 'sens_scarcity.py'), 'itemknn', '1e9'], env)]
    if launch:
        J += [(W, [os.path.join(CODE, 'launch3.py'), '1e9', m, 'random'], env) for m in ('MostPop', 'Random', 'ItemKNN', 'EASE')]
    return J


SENS_RQ2 = [('MostPop', 'random', 'lp'), ('ItemKNN', 'random', 'lpw'), ('EASE', 'random', 'lp'), ('ItemKNN', 'temporal', 'lp')]


def experiments():
    """Experiment groups, in the order and with the arguments used for the paper. Each entry: (script, args, env)."""
    X = {}
    X['core'] = ('Core borrowing grid of the v3 study (random split); input of the variants table and the S3 accounting. '
                 'The shipped results/full_random.jsonl is the first 24,754 records of this run (it was stopped there).',
                 ['data:paper'], [('src/runner.py', ['--stage', 'full', '--split', 'random', '--cheap-first'], {}),
                                 (os.path.join('legacy_v3', 'paper_code', 'a.py'), [], {})])
    X['rq1'] = ('RQ1: popularity vs co-interaction agreement (Tables rq1_*, Figure 4)', ['data:paper', 'data:r4'],
                [(os.path.join(CODE, 'rq1_agree.py'), [''], {}), (os.path.join(CODE, 'rq1_agree.py'), ['_r4'], {})])
    S = {'SKIP': UK}
    J = [rq2('ItemKNN', 'random', 'full', env=S), rq2('EASE', 'random', 'full', env=S),
         rq2('MostPop', 'random', 'full'), rq2('Random', 'random', 'local')]
    J += [rq2(m, 'random', 'grid', env={'ONLY': G6}) for m in ('ItemKNN', 'EASE', 'UserKNN')]
    J += [rq2('UserKNN', 'random', 'lpw', env=S)]
    for sp in ('temporal', 'calendar'):
        J += [rq2('ItemKNN', sp, 'full', env=S), rq2('EASE', sp, 'full', env=S), rq2('MostPop', sp, 'full'), rq2('Random', sp, 'local')]
    for sd in (1, 2):
        J += [rq2(m, 'random', 'lp', env={'SKIP': UK, 'XM_DATA_TAG': f'_seed{sd}'}) for m in ('ItemKNN', 'EASE', 'MostPop')]
    J += [rq2(m, 'random', 'lp', env={'SKIP': UK, 'XM_DATA_TAG': '_r4'}) for m in ('ItemKNN', 'EASE')]
    J += [rq2('ItemKNN', 'random', 'lpw', env={'ONLY': UK}), rq2('UserKNN', 'temporal', 'lp', env=S),
          rq2('UserKNN', 'calendar', 'lp', lim=1500, env=S)]
    J += [rq2('UserKNN', 'random', 'lp', lim=1500, env={'SKIP': UK, 'XM_DATA_TAG': f'_seed{sd}'}) for sd in (1, 2)]
    X['rq2'] = ('RQ2: complete local data, pool / weight / greedy (Tables rq2_*, Figures 5-6)', ['data:paper', 'data:calendar', 'data:seed1', 'data:seed2', 'data:r4'], J)
    J = []
    for p in (1, 2):                      # pass 1: every context except UK Books (cost cap); pass 2: UK Books
        for sp in ('random', 'calendar', 'temporal'):
            for m in ('ItemKNN', 'EASE', 'MostPop', 'Random', 'UserKNN'):
                mc = ('2e9' if sp == 'calendar' else '1e9') if p == 1 else '1e30'
                J.append((os.path.join(CODE, 'launch3.py'), ['1e9', m, sp, mc], {}))
    X['launch'] = ('Product launch (Tables launch*, Figure 8)', ['data:paper', 'data:calendar'], J)
    X['scarcity'] = ('RQ3: scarce local data (Table scarcity, Figure 7). The time limits are those of the paper run; '
                     'they decide which large contexts are reached (see README).', ['data:paper', 'data:calendar'],
                     [(os.path.join(CODE, 'scarcity2.py'), ['12600', 'A,B,C'], {}),
                      (os.path.join(CODE, 'scarcity2.py'), ['11000', 'A,B,C'], {'SC_DROP_UKNN005': '1'}),
                      (os.path.join(CODE, 'scarcity2.py'), ['5400', 'B,C'], {'SC_DROP_UKNN005': '1'})])
    X['mkt'] = ('Market-Aware GMF baseline and GMF controls (Table mkt; needs PyTorch)', ['data:paper', 'torch'],
                [(os.path.join(CODE, 'mkt_run.py'), ['12000'], {'OMP_NUM_THREADS': '1'}),
                 (os.path.join(CODE, 'mkt_run.py'), ['3000'], {'OMP_NUM_THREADS': '1'})])
    X['rq4'] = ('RQ4: single-donor ItemKNN gains (Tables rq4, S4; Figure 9)', ['data:paper'],
                [(os.path.join(CODE, 'single2.py'), ['40000', '0', '1'], {})])
    X['audit'] = ('Dataset accounting and checks (Table S2, split facts, identity checks, PureSVD reproducibility)', ['data:paper', 'zip'],
                  [(os.path.join(CODE, n), [], {}) for n in ('aud_raw.py', 'identity_checks.py', 'split_facts.py', 'svd_repro.py')])
    X['sens'] = ('Data-artefact sensitivity on _nv, _xd, _nvxd (Table sens)', ['data:paper', 'data:nv', 'data:xd', 'data:nvxd'],
                 sens_runs('_nv', SENS_RQ2, launch=True, userknn=True, scarcity=True)
                 + sens_runs('_xd', SENS_RQ2, launch=True) + sens_runs('_nvxd', SENS_RQ2, launch=False))
    X['sens2'] = ('Within-context variant merge _nvc (Table sens, Figures 4, 5, 9)', ['data:paper', 'data:nvc'],
                  sens_runs('_nvc', SENS_RQ2, launch=True, userknn=True, scarcity=True))
    X['nullab'] = ('RQ2 null-model ablation: pooling with curveball-randomised foreign data (Table nullab)', ['data:paper', 'data:nvc'],
                   [(os.path.join(CODE, 'nullab_run.py'), ['1e9', m], {'XM_DATA_TAG': tag} if tag else {})
                    for tag in ('', '_nvc') for m in ('MostPop', 'ItemKNN', 'EASE')])
    return X


KEEP_RAW = ['mkt_skips.jsonl']   # hardware-dependent MA-GMF exclusions of the paper run; mkt_run.py honours them (README 7.)
LEGACY_RAW = ['df.pkl', 'travel3_*.pkl', 'pairs.pkl', 'pairs2.pkl', 'hist.pkl', 'single.jsonl', 'ma_*.jsonl', 'fr_*.jsonl', 'l2_*.jsonl']


def cmd_experiments(a):
    X = experiments()
    names = a.only or [n for n in X if n != 'core']
    for n in names:
        if n not in X: sys.exit(f'unknown experiment {n}; choose from {", ".join(X)}')
    need = {}
    for n in names:
        for d in missing(X[n][1]): need.setdefault(d, []).append(n)
    if need:
        sys.exit('missing inputs (nothing was run):\n' + '\n'.join(f'  {d}  [needed by: {", ".join(g)}]' for d, g in need.items()))
    if a.fresh:
        raw, keep = os.path.join(ROOT, 'paper', 'raw'), os.path.join(ROOT, 'paper', 'raw_precomputed')
        if os.path.exists(keep): sys.exit('paper/raw_precomputed exists already; move it away first')
        shutil.move(raw, keep); os.makedirs(raw)
        for pat in LEGACY_RAW + KEEP_RAW:  # outputs of the superseded v3 scripts are not re-created by run.py; the MA-GMF
            for f in glob.glob(os.path.join(keep, pat)): shutil.copy2(f, raw)   # skip list keeps the paper's context set
        print('shipped results moved to paper/raw_precomputed/ (v3-only files and mkt_skips.jsonl copied back)')
    t0 = time.time()
    for n in names:
        desc, needs, jobs = X[n]
        print(f'\n===== experiment group {n}: {desc} [needs: {", ".join(needs)}]')
        for script, args, env in jobs:
            cwd = os.path.join(ROOT, 'src') if script.startswith('src') else None
            run(script, *args, env=env, cwd=cwd)
    print(f'\nexperiments done in {(time.time() - t0) / 3600:.1f} h')


# --------------------------------------------------------------------------------------------- tables
TABLES = [  # (script, args, requirement tags beyond the shipped records, outputs); run in this order
    ('rq1_tables.py', [], [], 'tables_rq1.tex, results/rq1_*.csv'),
    ('rq2_tables.py', [], [], 'tables_rq2.tex, results/rq2_*.csv'),
    ('rq2_accounting.py', [], [], 'results/rq2_accounting_detail.csv'),
    ('scarcity2_tables.py', [], [], 'tables_scarcity.tex, results/sc_*.csv'),
    ('launch3_tables.py', [], [], 'tables_launch.tex, tables_launch_S1.tex, results/l3_*.csv'),
    ('mkt_tables.py', [], [], 'tables_mkt.tex, results/mkt_summary.csv'),
    ('rq4_tables.py', ['--rq1', 'new'], [], 'tables_rq4.tex, tables_S4.tex, results/rq4_country_*.csv, logs/aud_rq4.log'),
    ('rq4_tables.py', ['--rq1', 'new', '--min-local', '0.05'], [],
     'tables_rq4_minlocal005.tex, results/rq4_minlocal005_*.csv (ratios only where local agreement >= 0.05)'),
    ('rq4_tables.py', ['--rq1', 'travel3', '--tag', '_oldrq1', '--check-old'], [], 'tables_rq4_oldrq1.tex, tables_S4_oldrq1.tex (v3 RQ1 outcomes)'),
    ('rq4_ratiofree.py', [], [], 'tables_rq4_ratiofree.tex, results/rq4_ratiofree.csv (RQ4 on D = F - L, all data versions)'),
    ('variants.py', [], [], 'tables_variants.tex'),
    ('s2_accounting.py', [], ['data:paper', 'zip'], 'tables_S2.tex, raw/aud_S2.csv'),
    ('variant_diagnostics.py', ['orig', '_nv', '_xd', '_nvxd', '_nvc'], ['data:paper', 'data:nv', 'data:xd', 'data:nvxd', 'data:nvc'],
     'tables_sens_diag.tex, results/sens_diag_*.csv'),
    ('sens_variant_report.py', [], ['varmap:nv'], 'results/sens_variant_*.csv'),
    ('sens2_variant_report.py', [], ['varmap:nvc', 'rawcache'], 'results/sens2_variant_*.csv'),
    ('sens_tables.py', [], [], 'results/sens_*.csv, sens_tables.md'),
    ('sens2_tables.py', [], [], 'results/sens2_*.csv, sens2_tables.md'),
    ('sens2_extra.py', [], [], 'results/sens2_launch_paired.csv'),
    ('sens2_rq4.py', [], [], 'results/sens2_rq4.csv'),
    ('sens_accounting.py', [], [], '(prints run times of the sensitivity runs)'),
    ('accounting.py', [], [], 'tables_S3.tex, raw/aud_accounting*.csv'),
    ('nullab_tables.py', [], [], 'tables_nullab.tex, results/nullab_*.csv'),
    ('loo_robustness.py', [], [], 'tables_loo.tex, results/loo_robustness.csv, results/rq2_catalog_strata.csv'),
    ('supp_tables.py', [], [], 'supp/S1-S5c.tex (Supplementary Tables), results/rq1_per_context.csv'),
    ('variability.py', [], [], 'supp/S5d.tex, results/variability.csv (variability across repetitions)'),
    ('supp_extra.py', [], [], 'supp/S6.tex (per-context accounting), supp/S7.tex (RQ1 diagnostics, launch skips)'),
]


def cmd_tables(a):
    t0 = time.time(); skipped = []
    for script, args, needs, out in TABLES:
        miss = missing(needs)
        if miss:
            print(f'\n--- skipped {script} {" ".join(args)} -> {out}: missing ' + '; '.join(miss), flush=True)
            skipped.append(f'{script} -> {out} (missing: {"; ".join(miss)})'); continue
        run(os.path.join(CODE, script), *args)
    print(f'\ntables done in {time.time() - t0:.0f} s; outputs in paper/, paper/supp/ and paper/results/')
    if skipped:
        print('Not rebuilt (they read the preprocessed or raw data); the shipped versions are kept:\n  ' + '\n  '.join(skipped))


def cmd_figures(a):
    run(os.path.join(CODE, 'figures_paper.py'), *(['--copy-to', a.copy_to] if a.copy_to else []))


# --------------------------------------------------------------------------------------------- check
ID_RE = re.compile(r'\bA[A-Z0-9]{13,}\b')      # XMarket reviewer IDs look like this; none may be shipped


IGNORED = ('data/', 'raw_data/xmarket_categories/', 'paper/raw/aud_rawcache/', 'paper/raw_precomputed/', '.venv/', '.pytest_cache/', 'scratch/',
           '.git/')                                   # folders that are .gitignore'd (never part of a release)
IGNORED_FILES = re.compile(r'(^|/)__pycache__/|\.pyc$|\.pkl\.tmp$|^raw_data/[^/]*\.zip$|^results/variant_map_[^/]*\.parquet$|'
                           r'^results/.*_precomputed|^paper/raw/aud_identity_multimarket_rows\.csv$|^paper/raw/hist\.pkl$|'
                           r'^paper/results/sens2?_variant_components\.csv$|^legacy_v3/paper/figures/[^/]*\.png$')


def tracked_files():
    """Files of the release: `git ls-files` in a git checkout, otherwise every file except the .gitignore'd ones."""
    try:
        out = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split('\n')
        out = [f for f in out if f]
        if out:
            return out, True
    except Exception:
        pass
    fs = []
    for d, dirs, names in os.walk(ROOT):
        rel = os.path.relpath(d, ROOT).replace(os.sep, '/')
        rel = '' if rel == '.' else rel + '/'
        dirs[:] = [x for x in dirs if not (rel + x + '/').startswith(IGNORED) and x != '__pycache__']
        fs += [rel + n for n in names if not IGNORED_FILES.search(rel + n)]
    return fs, False


NUM_RE = re.compile(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')


def same_as_committed(f, tol=0.0):
    """True if the working-tree file f equals its committed version up to line endings (CRLF = LF), and, for text tables
    with tol > 0, up to a numeric tolerance (relative or absolute) on every number; everything else must be identical."""
    path = os.path.join(ROOT, f)
    if not os.path.isfile(path):
        return False
    r = subprocess.run(['git', 'show', f'HEAD:{f}'], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        return False
    new, old = (open(path, 'rb').read().replace(b'\r\n', b'\n'), r.stdout.replace(b'\r\n', b'\n'))
    if new == old:
        return True
    if not tol or not f.endswith(('.csv', '.tex', '.txt', '.md')):
        return False
    try:
        A, B = NUM_RE.split(new.decode('utf-8')), NUM_RE.split(old.decode('utf-8'))
    except UnicodeDecodeError:
        return False
    if len(A) != len(B):
        return False
    import math
    for i, (x, y) in enumerate(zip(A, B)):
        if i % 2 == 0:
            if x != y: return False
        elif x != y and not math.isclose(float(x), float(y), rel_tol=tol, abs_tol=tol):
            return False
    return True


def _strings(o, depth=0):
    """Every string held by a loaded result object (DataFrame/Series cells, index and column labels, dict keys and
    values, list items, numpy string/object arrays)."""
    import numpy as np, pandas as pd
    if depth > 6: return
    if isinstance(o, str): yield o
    elif isinstance(o, bytes): yield o.decode('utf-8', 'replace')
    elif isinstance(o, pd.DataFrame):
        yield from (str(c) for c in o.columns)
        yield from _strings(o.index, depth + 1)
        for c in o.columns:
            if not (pd.api.types.is_numeric_dtype(o[c]) or pd.api.types.is_bool_dtype(o[c])):
                yield from _strings(o[c].to_numpy(), depth + 1)
    elif isinstance(o, (pd.Series, pd.Index)):
        if not pd.api.types.is_numeric_dtype(o): yield from _strings(o.to_numpy(), depth + 1)
    elif isinstance(o, np.ndarray):
        if o.dtype.kind in 'USO':
            for v in o.ravel(): yield from _strings(v, depth + 1)
    elif isinstance(o, dict):
        for k, v in o.items(): yield from _strings(k, depth + 1); yield from _strings(v, depth + 1)
    elif isinstance(o, (list, tuple, set)):
        for v in o: yield from _strings(v, depth + 1)


def binary_strings(path):
    """The strings stored in a shipped .pkl, .parquet or .npz result file (loaded with pandas / numpy)."""
    import numpy as np, pandas as pd
    if path.endswith('.parquet'): yield from _strings(pd.read_parquet(path))
    elif path.endswith('.pkl'): yield from _strings(pd.read_pickle(path))
    else:
        with np.load(path, allow_pickle=False) as z:
            for k in z.files: yield k; yield from _strings(z[k])


def cmd_check(a):
    ok = True
    rc = subprocess.run([PY, '-m', 'pytest', '-q', os.path.join(ROOT, 'tests')], cwd=ROOT).returncode
    print('unit tests:', 'passed' if rc == 0 else 'FAILED'); ok &= rc == 0
    files, in_git = tracked_files()
    # 1. sizes and raw identifiers
    sizes = {f: os.path.getsize(os.path.join(ROOT, f)) for f in files if os.path.isfile(os.path.join(ROOT, f))}
    big = [f for f, s in sizes.items() if s > 50e6]
    total = sum(sizes.values())
    print(f'repository files: {len(sizes)}, {total / 1e6:.1f} MB (limit 100 MB); files over 50 MB: {big or "none"}')
    if 95e6 < total <= 100e6:
        print('WARNING: the repository is above 95 MB, close to the 100 MB limit')
    ok &= not big and total <= 100e6
    leaks, nbin = [], 0
    for f in files:
        path = os.path.join(ROOT, f)
        if not os.path.isfile(path): continue
        if f.endswith(('.csv', '.jsonl', '.json', '.txt', '.tex', '.md', '.log', '.ipynb', '.py', '.cff')):
            with open(path, encoding='utf-8', errors='replace') as fh:
                if ID_RE.search(fh.read()): leaks.append(f)
        elif f.endswith(('.pkl', '.parquet', '.npz')):
            nbin += 1
            if any(ID_RE.search(s) for s in binary_strings(path)): leaks.append(f)
    print(f'files with raw reviewer IDs (text files and the string contents of {nbin} .pkl/.parquet/.npz files):', leaks or 'none')
    ok &= not leaks
    # 2. every table and figure of the paper exists
    need = ['paper/tables_rq1.tex', 'paper/tables_rq2.tex', 'paper/tables_scarcity.tex', 'paper/tables_launch.tex',
            'paper/tables_launch_S1.tex', 'paper/tables_mkt.tex', 'paper/tables_rq4.tex', 'paper/tables_S2.tex', 'paper/tables_S3.tex',
            'paper/tables_S4.tex', 'paper/tables_rq4_ratiofree.tex', 'paper/tables_variants.tex', 'paper/tables_sens_diag.tex', 'paper/tables_nullab.tex', 'paper/tables_loo.tex']
    need += [f'paper/supp/{n}.tex' for n in ('S1', 'S2', 'S3', 'S4', 'S5a', 'S5b', 'S5c', 'S5d', 'S6', 'S7')]
    need += [f'paper/figures/figure{i}.{e}' for i in range(1, 10) for e in ('pdf', 'png')]
    need += [f'paper/figures/data/figure{i}.csv' for i in range(1, 10)]
    miss = [f for f in need if not os.path.exists(os.path.join(ROOT, f))]
    print('missing tables/figures:', miss or 'none'); ok &= not miss
    # 3. numbers in the tables equal the machine-readable results (a sample per table)
    import pandas as pd
    def tex(name): return open(os.path.join(ROOT, 'paper', name), encoding='utf-8').read()
    R = os.path.join(ROOT, 'paper', 'results')
    probs = []
    t2 = pd.read_csv(os.path.join(R, 'rq2_table2_all.csv'))
    for _, r in t2[(t2.split == 'random') & (t2.arm == 'pool') & t2.model.isin(['ItemKNN', 'EASE', 'MostPop'])].iterrows():
        if f'{r.gain_mean:+.4f}' not in tex('tables_rq2.tex'): probs.append(f'rq2 {r.model} {r.gain_mean:+.4f}')
    t1 = pd.read_csv(os.path.join(R, 'rq1_T1.csv')); t1 = t1[t1.arm == 'all']
    for _, r in t1.iterrows():
        if f'{r.D:+.2f}'.replace('-', '$-$') not in tex('tables_rq1.tex') and f'{r.D:+.2f}' not in tex('tables_rq1.tex'):
            probs.append(f'rq1 D {r.measure} {r.D:+.2f}')
    q4 = pd.read_csv(os.path.join(ROOT, 'paper', 'raw', 'aud_rq4_results.csv'))
    for _, r in q4[q4.outcome != 'd'].iterrows():
        for d in ('Hofstede distance', 'Shared language', 'Geographic distance'):
            s = f"{r['r|' + d]:+.2f} ({r['p|' + d]:.3f})"
            if s not in tex('tables_rq4.tex'): probs.append(f'rq4 {r.outcome} {d} {s}')
    q4f = pd.read_csv(os.path.join(R, 'rq4_ratiofree.csv'))
    for _, r in q4f[(q4f.in_ac == 'incl.') & q4f.kind.isin(['D', 'gain'])].iterrows():
        s = f"{r.r:+.2f}".replace('-', '$-$') + f" ({r.p:.4f})"
        if s not in tex('tables_rq4_ratiofree.tex'): probs.append(f'rq4 ratio-free {r.source} {r.measure} {r.distance} {s}')
    print('table/result cross-checks:', 'all agree' if not probs else probs); ok &= not probs
    # 4. figure data were drawn from the result files (spot checks)
    f5 = pd.read_csv(os.path.join(ROOT, 'paper', 'figures', 'data', 'figure5.csv'))
    f5 = f5[f5['data'].isin(['paper data', 'original data'])]
    m = f5.merge(pd.read_csv(os.path.join(R, 'rq2_fig5_summary.csv')), on=['split', 'model', 'arm'], suffixes=('', '_src'))
    bad5 = int((m.gain_mean - m.gain_mean_src).abs().gt(1e-12).sum()) if 'gain_mean' in m and len(m) else -1
    print('figure 5 data = rq2_fig5_summary.csv:', 'yes' if bad5 == 0 else f'NO ({bad5})'); ok &= bad5 == 0
    # 5. country features used by Figure 9 are the ones of the RQ4 analysis
    src = open(os.path.join(ROOT, CODE, 'rq4_tables.py'), encoding='utf-8').read().replace(' ', '')
    figsrc = open(os.path.join(ROOT, CODE, 'figures_paper.py'), encoding='utf-8').read()
    same = all(('lang=' + m.group(1)).replace(' ', '') in src if n == 'LANG' else ('cap=' + m.group(1)).replace(' ', '') in src
               for n, m in [(k, re.search(rf'^{k} = (\{{.*\}})$', figsrc, re.M)) for k in ('LANG', 'CAP')])
    print('Figure 9 country features = rq4_tables.py:', 'yes' if same else 'NO'); ok &= same
    # 6. regenerated outputs equal the committed ones (CRLF and LF line endings count as equal; with --tol, numbers in
    #    .csv/.tex/.txt/.md files may differ by that relative/absolute tolerance)
    if in_git:
        st = subprocess.run(['git', 'diff', '--name-only', '--', 'paper', 'results'], cwd=ROOT, capture_output=True, text=True).stdout
        changed = [l for l in st.split('\n') if l.strip() and not l.startswith('paper/figures/')]
        changed = [f for f in changed if not same_as_committed(f, a.tol)]
        print('tables/results differing from the committed versions'
              + (f' (numeric tolerance {a.tol:g})' if a.tol else '') + ':', changed or 'none'); ok &= not changed
    else:
        print('tables/results vs committed versions: not checked (not a git checkout)')
    print('\nCHECK', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


def shell_command(script, args, env):
    """The command line of one experiment run, as typed from the repository root (POSIX shell syntax)."""
    env = {k: (os.path.relpath(v, ROOT) if k == 'PAPER_RAW' and os.path.isabs(v) else v) for k, v in env.items()}
    q = lambda v: f'"{v}"' if (str(v) == '' or any(c in str(v) for c in ' ,')) else str(v)
    pre = ' '.join(f'{k}={q(v)}' for k, v in env.items())
    cd = 'cd src && ' if script.startswith('src') else ''
    path = os.path.relpath(script, 'src') if cd else script
    return f'{cd}{pre + " " if pre else ""}python {path.replace(os.sep, "/")} {" ".join(q(str(x).replace(os.sep, "/")) for x in args)}'.rstrip()


def cmd_list(a):
    print('Data variants (python run.py data --only NAME):')
    for n, (tag, env, d) in VARIANTS.items(): print(f'  {n:9s} {d}')
    print('\nExperiment groups (python run.py experiments --only NAME; `core` runs only when named).')
    print('Below each group: the exact command of every run, as typed from the repository root (POSIX shell syntax;')
    print('in PowerShell set each variable first, e.g. $env:SKIP = "uk Books").')
    for n, (d, needs, jobs) in experiments().items():
        print(f'\n  {n:9s} {len(jobs):3d} {"run " if len(jobs) == 1 else "runs"}  [needs: {", ".join(needs)}]  {d}')
        for script, args, env in jobs:
            print('      ' + shell_command(script, args, env))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('data'); p.add_argument('--only', nargs='+'); p.add_argument('--force', action='store_true')
    p = sub.add_parser('experiments'); p.add_argument('--only', nargs='+')
    p.add_argument('--fresh', action='store_true', help='move the shipped results to paper/raw_precomputed/ first')
    sub.add_parser('tables')
    p = sub.add_parser('figures'); p.add_argument('--copy-to', help='also copy the PDFs to this folder')
    p = sub.add_parser('check')
    p.add_argument('--tol', type=float, default=0.0, help='numeric tolerance when comparing regenerated tables/results with '
                   'the committed ones (default 0: exact; CRLF and LF line endings always count as equal), e.g. --tol 1e-9')
    sub.add_parser('list')
    a = ap.parse_args()
    {'data': cmd_data, 'experiments': cmd_experiments, 'tables': cmd_tables, 'figures': cmd_figures,
     'check': cmd_check, 'list': cmd_list}[a.cmd](a)


if __name__ == '__main__':
    main()
