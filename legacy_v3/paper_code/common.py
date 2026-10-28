"""Shared paths for the superseded v3 scripts (legacy_v3/paper_code). Inputs are read from the main tree
(results/, paper/raw/); outputs of the table and figure scripts go to legacy_v3/paper/."""
import os, sys
CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CODE, '..', '..'))
LEGACY = os.path.join(REPO, 'legacy_v3')
SRC = os.path.join(REPO, 'src')
RESULTS = os.path.join(REPO, 'results')
DATA = os.path.join(REPO, 'data')
PAPER_RAW = os.environ.get('PAPER_RAW', os.path.join(REPO, 'paper', 'raw'))
if not os.path.isabs(PAPER_RAW):
    PAPER_RAW = os.path.join(REPO, PAPER_RAW)
os.makedirs(PAPER_RAW, exist_ok=True)
for p in (os.path.join(REPO, 'paper', 'code'), os.path.join(LEGACY, 'src'), SRC):   # mustadd, boot; analyze, verify; core
    if p not in sys.path:
        sys.path.append(p)
