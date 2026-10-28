"""Shared paths for the paper experiments (all relative to the repository root)."""
import os
CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CODE, '..', '..'))
SRC = os.path.join(REPO, 'src')          # core pipeline (data, models, metrics, experiment)
RESULTS = os.path.join(REPO, 'results')  # core pipeline outputs
DATA = os.path.join(REPO, 'data')        # preprocessed interactions.parquet
PAPER_RAW = os.environ.get('PAPER_RAW', os.path.join(REPO, 'paper', 'raw'))   # paper experiment outputs
if not os.path.isabs(PAPER_RAW):          # a relative PAPER_RAW is taken relative to the repository root
    PAPER_RAW = os.path.join(REPO, PAPER_RAW)
os.makedirs(PAPER_RAW, exist_ok=True)


def already_done(*paths, json_key=None):
    """Resumability of the audit scripts: True (and a message) if every output in `paths` exists and, for a JSON
    output, contains `json_key` (written only at the very end). The command-line flag --fresh forces a recomputation;
    `python run.py experiments --fresh` moves paper/raw away first, so the outputs are recomputed as well."""
    import sys, json
    if '--fresh' in sys.argv[1:]:
        return False
    for p in paths:
        if not os.path.exists(p):
            return False
        if json_key and p.endswith('.json'):
            try:
                if json_key not in json.load(open(p, encoding='utf-8')): return False
            except ValueError:
                return False
    names = [os.path.relpath(p, REPO).replace(os.sep, '/') for p in paths]
    shown = ', '.join(names) if len(names) <= 3 else f'{len(names)} files in {os.path.dirname(names[0])}/'
    print(f'finished already (outputs exist): {shown}; skipped (use --fresh to recompute)')
    return True
