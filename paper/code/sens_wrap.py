"""Thin wrapper: run an existing experiment script on a tagged data variant without editing it.

Usage:  XM_DATA_TAG=_nv PAPER_RAW=paper/raw/sens_nv python paper/code/sens_wrap.py SCRIPT.py [args...]

src/data.py already reads data/interactions{XM_DATA_TAG}.parquet. Several experiment scripts, however, read the
untagged results/overlap_{random,temporal}.parquet and results/contexts.parquet directly. This wrapper redirects
exactly those two file names (when untagged) to their tagged versions by wrapping pandas.read_parquet, then runs the
script as __main__. Outputs go to PAPER_RAW (set it to a per-tag directory so untagged output names do not collide).
"""
import os, sys, re, runpy
import pandas as pd

TAG = os.environ.get('XM_DATA_TAG', '')
assert TAG, 'set XM_DATA_TAG'
_orig = pd.read_parquet
PAT = re.compile(r'^(overlap_(?:random|temporal)|contexts)\.parquet$')
REDIRECTS = []


def _rp(path, *a, **k):
    p = str(path)
    base = os.path.basename(p)
    if PAT.match(base) and os.path.basename(os.path.dirname(os.path.abspath(p))) == 'results':
        new = os.path.join(os.path.dirname(p), base.replace('.parquet', f'{TAG}.parquet'))
        if not os.path.exists(new):
            raise FileNotFoundError(new)
        REDIRECTS.append((base, os.path.basename(new)))
        print(f'[sens_wrap] {base} -> {os.path.basename(new)}', file=sys.stderr, flush=True)
        path = new
    return _orig(path, *a, **k)


pd.read_parquet = _rp
script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name='__main__')
