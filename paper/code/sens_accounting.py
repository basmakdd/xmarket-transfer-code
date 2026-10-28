"""Run accounting for the sensitivity runs: records, contexts, errors/skips and summed unit time per output file."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import REPO, PAPER_RAW
import json, glob, os
for d in sorted(glob.glob(f'{PAPER_RAW}/sens_*')):
    for f in sorted(glob.glob(f'{d}/*.jsonl')):
        R = [json.loads(l) for l in open(f, encoding='utf-8')]
        sec = sum((r.get('sec') or 0) for r in R)
        err = sum(1 for r in R if r.get('err'))
        print(f"{os.path.relpath(f, PAPER_RAW).replace(os.sep, '/'):45s} records {len(R):5d} contexts {len({r['t'] for r in R}):3d} errors/skips {err:3d} unit time {sec / 60:6.1f} min")
