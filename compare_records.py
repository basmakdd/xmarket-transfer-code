#!/usr/bin/env python
"""Compare re-run experiment records with the shipped ones.

    python compare_records.py SCRATCH_RAW [SHIPPED_RAW] [--tol 1e-9]

Every *.jsonl file under SCRATCH_RAW (for example scratch/raw, written by an experiment script run with
PAPER_RAW=scratch/raw) is matched with the file of the same relative path under SHIPPED_RAW (default: paper/raw of
this repository). A re-run usually covers only some contexts, so each re-run record is looked up among the shipped
records with the same identifying fields (its text, integer and boolean fields, e.g. context, model, split, arm, seed;
count fields n_* and the error field are not used for matching) and must equal one of them. Timing fields (sec,
unit_sec, sec_per_epoch, elapsed_s) are ignored. Numbers must agree exactly, or within the relative/absolute tolerance
--tol (other platforms or linear-algebra libraries may differ in the last digits of floating-point results).

Prints one line per file and a final line "RECORDS EQUAL: n/n". Exit code 0 if every re-run record equals a shipped
one, 1 otherwise (a difference, a file without a shipped counterpart, or no records found).
"""
import argparse
import glob
import json
import math
import os
import sys

TIMING = {'sec', 'unit_sec', 'sec_per_epoch', 'elapsed_s'}


def strip(x):
    """Drop the timing fields, at any depth."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def same(a, b, tol):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            return True
        return tol > 0 and math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def key(r):
    """Identifying fields of a record."""
    return tuple(sorted((k, str(v)) for k, v in r.items()
                        if isinstance(v, (str, int, bool)) and not k.startswith('n_') and k != 'err'))


def read(path):
    with open(path, encoding='utf-8') as fh:
        return [strip(json.loads(l)) for l in fh if l.strip()]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scratch', metavar='SCRATCH_RAW', help='folder with the re-run records (e.g. scratch/raw)')
    ap.add_argument('shipped', metavar='SHIPPED_RAW', nargs='?', default=os.path.join(here, 'paper', 'raw'),
                    help='folder with the shipped records (default: paper/raw of this repository)')
    ap.add_argument('--tol', type=float, default=0.0,
                    help='relative/absolute tolerance for numbers (default 0: exact), e.g. --tol 1e-9')
    a = ap.parse_args()
    if not os.path.isdir(a.scratch):
        sys.exit(f'not a folder: {a.scratch}')
    total = ok = 0; problems = False
    files = sorted(glob.glob(os.path.join(a.scratch, '**', '*.jsonl'), recursive=True))
    for f in files:
        rel = os.path.relpath(f, a.scratch); g = os.path.join(a.shipped, rel)
        if not os.path.exists(g):
            print(f'{rel}: no shipped file {g}'); problems = True; continue
        idx = {}
        for r in read(g):
            idx.setdefault(key(r), []).append(r)
        A = read(f)
        n = sum(any(same(r, s, a.tol) for s in idx.get(key(r), [])) for r in A)
        total += len(A); ok += n
        print(f'{rel}: {n}/{len(A)} records equal to the shipped ones')
    if not total:
        print(f'no records found under {a.scratch}'); problems = True
    print(f'RECORDS EQUAL: {ok}/{total}' + (f' (tolerance {a.tol:g})' if a.tol else '')
          + ('' if ok == total else '  <-- DIFFERENCES'))
    sys.exit(0 if ok == total and not problems else 1)


if __name__ == '__main__':
    main()
