"""Stage 1d - resumable runner.

Every completed unit is appended to a JSONL file immediately and skipped on a rerun, so
an overnight job survives interruption. Unit keys are deduplicated on read as well as on
write, because duplicate appends would silently produce cells averaged over unequal
seed counts.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import design as Design
import experiment as E
from util import stable_hex, stable_seed

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_MODELS = ["Random", "MostPop", "ItemKNN", "ItemKNN-ext", "PureSVD", "EASE"]


def subsample(edges: tuple[np.ndarray, np.ndarray], frac: float, seed: int):
    u, i = edges
    if frac >= 1.0 or len(u) == 0:
        return u, i
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * len(u))))
    sel = rng.choice(len(u), size=n, replace=False)
    return u[sel], i[sel]


def unit_edge_lists(unit: Design.Unit, blocks) -> list:
    """Training edges for a unit: the target's own, plus each donor's."""
    target_block = blocks[unit.target]
    edge_lists = [(target_block.train_u, target_block.train_i)]
    for donor in unit.donors:
        b = blocks[donor]
        seed = stable_seed(unit.target, donor, unit.rep, round(unit.frac, 4))
        edge_lists.append(subsample((b.train_u, b.train_i), unit.frac, seed))
    return edge_lists


def per_user_path(out_path: Path, unit: Design.Unit) -> Path:
    """One compressed file of per-user metrics per unit, keyed by the unit's identity."""
    root = out_path.parent / (out_path.stem + "_peruser")
    root.mkdir(parents=True, exist_ok=True)
    tag = stable_hex(*unit.key())
    return root / f"{unit.model}_{unit.target[0]}_{unit.target[1]}_{tag}.npz"


def existing_params(out_path: Path, split: str):
    """Recover each target-only hyperparameter choice so a resumed run can freeze it.

    Filtered on split AND code version: without that, a resume against a file holding
    records from an earlier model generation would freeze donor units at hyperparameters
    chosen by a model that no longer exists.
    """
    if not out_path.exists():
        return
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if (rec.get("k") == 0 and rec.get("params") is not None
                    and rec.get("split") == split
                    and rec.get("code_version") == Design.CODE_VERSION):
                yield ((rec["model"],
                        (rec["target_market"], rec["target_category"])), rec["params"])


def record_key(rec: dict) -> tuple:
    donors = tuple(sorted(tuple(x) for x in rec.get("donors", [])))
    return (rec["split"], rec["model"], (rec["target_market"], rec["target_category"]),
            rec["channel"], rec["order"], rec["rep"], rec["k"], round(rec["frac"], 4),
            stable_seed(donors), rec.get("code_version"))


def load_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                done.add(record_key(json.loads(line)))
    return done


def run(units, blocks, out_path: Path, verbose: bool = True,
        max_seconds: float | None = None):
    done = load_done(out_path)
    pending = [u for u in units if u.key() not in done]
    print(f"{len(done)} units already recorded | {len(pending)} pending "
          f"of {len(units)} total", flush=True)

    sessions: dict = {}
    base_params: dict = {}
    split = units[0].split if units else "random"
    for rec in existing_params(out_path, split):
        base_params[rec[0]] = rec[1]
    t_start = time.time()
    stopped_early = False
    for n, unit in enumerate(pending, 1):
        # A clean deadline lets the job run in the foreground and exit by itself, instead
        # of becoming an idle background process. On a memory-constrained machine such a
        # process gets reaped mid-unit; every completed unit is already on disk, but a
        # bounded run that returns control is simpler to drive and loses nothing.
        if max_seconds is not None and time.time() - t_start > max_seconds:
            stopped_early = True
            print(f"reached the {max_seconds:.0f}s deadline after {n - 1} units; "
                  f"rerun the same command to continue", flush=True)
            break
        if unit.target not in sessions:
            session = E.TargetSession(unit.target, blocks[unit.target])
            capacity, dense = E.target_capacity(session, blocks[unit.target])
            sessions = {unit.target: (session, capacity, dense)}
        session, capacity, dense = sessions[unit.target]

        t0 = time.time()
        edge_lists = unit_edge_lists(unit, blocks)
        try:
            res, per_user, err = E.fit_and_select(
                unit.model, session, edge_lists, capacity=capacity, dense=dense,
                base_params=base_params.get((unit.model, unit.target)))
        except Exception as exc:  # a failed unit must not lose the rest of the run
            res, per_user, err = None, None, f"{type(exc).__name__}: {exc}"

        rec = {
            "split": unit.split, "model": unit.model,
            "target_market": unit.target[0], "target_category": unit.target[1],
            "channel": unit.channel, "order": unit.order, "rep": unit.rep,
            "k": unit.k, "frac": unit.frac,
            "donors": [list(d) for d in unit.donors],
            "borrowed_edges": int(sum(len(e[0]) for e in edge_lists[1:])),
            "elapsed_s": round(time.time() - t0, 2),
            "code_version": Design.CODE_VERSION,
            "error": err,
        }
        if res is not None:
            params = res.pop("params", {})
            rec["params"] = params
            rec.update(res)
            if unit.k == 0:
                # The target-only selection becomes the frozen reference for this
                # target's donor units, so both must be produced in the same pass.
                base_params[(unit.model, unit.target)] = params
            if per_user is not None:
                # Per-user vectors are what the per-user bootstrap intervals are built
                # from, and they cannot be recovered from the recorded means afterwards.
                np.savez_compressed(
                    per_user_path(out_path, unit),
                    **{k: v.astype(np.float32) for k, v in per_user.items()})

        with out_path.open("a", encoding="utf-8", newline='\n') as fh:
            fh.write(json.dumps(rec) + "\n")

        if verbose:
            rate = (time.time() - t_start) / n
            eta = rate * (len(pending) - n) / 60
            label = f"{unit.target[0]} {unit.target[1][:18]}"
            status = err if err else f"ndcg@10 {rec.get('ndcg@10', float('nan')):.4f}"
            print(f"[{n}/{len(pending)}] {unit.model:<11} {label:<24} "
                  f"{unit.channel:<4} k={unit.k:<3} f={unit.frac:<5} {status} "
                  f"({rec['elapsed_s']}s, eta {eta:.0f}m)", flush=True)

    if not stopped_early:
        print(f"run complete: {len(pending)} units", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "pilot", "full"], default="baseline")
    ap.add_argument("--split", default="random")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="market:Category tokens; default is every context")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--cheap-first", action="store_true",
                    help="order targets by ascending evaluation cost")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="exit cleanly after this many seconds; rerun to continue")
    args = ap.parse_args()

    data = D.load()
    blocks = E.load_blocks(data, args.split)
    all_targets = sorted(blocks)

    if args.targets:
        targets = [tuple(t.split(":", 1)) for t in args.targets]
    else:
        targets = all_targets

    if args.cheap_first:
        # Evaluation cost is roughly (test users x catalog), and it spans four orders of
        # magnitude across the grid. Running cheap contexts first means most of the design
        # is on disk within the first hour, and an interrupted overnight run still leaves
        # a usable, analysable majority rather than a random slice.
        contexts = pd.read_parquet(RESULTS_DIR / "contexts.parquet")
        cost = {(m, c): int(u) * int(i) for m, c, u, i
                in zip(contexts["market"], contexts["category"],
                       contexts["users"], contexts["items"])}
        targets = sorted(targets, key=lambda t: cost.get(t, 0))

    if args.stage == "baseline":
        units = [Design.Unit(args.split, m, t, "NONE", "none", 0, 0, 1.0, ())
                 for t in targets for m in args.models]
        out = RESULTS_DIR / (args.out or f"baseline_{args.split}.jsonl")
    else:
        overlap = pd.read_parquet(RESULTS_DIR / f"overlap_{args.split}.parquet")
        units = Design.enumerate_units(overlap, targets, args.models, args.split,
                                       n_random=args.n_random)
        out = RESULTS_DIR / (args.out or f"{args.stage}_{args.split}.jsonl")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(units, blocks, out, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
