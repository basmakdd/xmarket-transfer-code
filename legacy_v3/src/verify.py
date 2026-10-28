"""Verifier role - recompute recorded metrics through a deliberately different code path.

`metrics.rank_metrics` is vectorised, batched and uses `argpartition`; the reference below
is a plain Python loop that sorts every user's scores and accumulates gains term by term.
Agreement between the two is meaningful precisely because they share no machinery beyond
the fitted model. The reference also rebuilds the mask and the ground truth from the raw
interaction table rather than from a `TargetSession`, so the session construction is
checked too.

    python verify.py --file results/baseline_random.jsonl --sample 0.02
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

import data as D
import experiment as E
import models as Mod
import runner
from util import stable_seed

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def reference_metrics(scores, truth_sets, cutoff=10, reduce=True):
    """Deliberately naive: sort, walk the list, accumulate. No numpy tricks.

    `reduce=False` returns the per-user lists so a caller can batch without changing the
    arithmetic - the mean of concatenated per-user values equals the unbatched mean.
    """
    ndcgs, recalls = [], []
    for row, truth in zip(scores, truth_sets):
        ranked = sorted(range(len(row)), key=lambda j: (-row[j], j))[:cutoff]
        dcg = 0.0
        hits = 0
        for pos, item in enumerate(ranked):
            if item in truth:
                dcg += 1.0 / math.log2(pos + 2)
                hits += 1
        idcg = sum(1.0 / math.log2(p + 2) for p in range(min(len(truth), cutoff)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        recalls.append(hits / len(truth) if truth else 0.0)
    if not reduce:
        return ndcgs, recalls
    return float(np.mean(ndcgs)), float(np.mean(recalls))


def rebuild_from_raw(data, target, split):
    """Ground truth and mask built straight from the interaction table."""
    block = data[(data.market == target[0]) & (data.category == target[1])]
    col = f"split_{split}"
    catalog = sorted(block["i"].unique().tolist())
    col_of = {int(v): j for j, v in enumerate(catalog)}

    truth, masked = {}, {}
    for u, i, s in zip(block["u"], block["i"], block[col]):
        u, i = int(u), int(i)
        if s == "test":
            truth.setdefault(u, set()).add(col_of[i])
        else:
            masked.setdefault(u, set()).add(col_of[i])
    users = sorted(truth)
    return np.array(catalog, dtype=np.int32), users, truth, masked


def verify_record(rec, data, blocks, tol=1e-9):
    target = (rec["target_market"], rec["target_category"])
    split = rec["split"]

    catalog, users, truth, masked = rebuild_from_raw(data, target, split)
    session = E.TargetSession(target, blocks[target])

    assert np.array_equal(catalog, session.catalog), "catalog mismatch"
    # The session subsamples evaluation users above MAX_EVAL_USERS. Asserting equality
    # with the full raw set crashed the verifier on exactly the three largest and most
    # influential contexts, so it could never check them. Restrict the raw rebuild to the
    # sampled users instead, and keep the containment check that actually matters.
    sampled = session.eval_users.tolist()
    assert set(sampled) <= set(users), "session evaluates users absent from the raw data"
    users = sampled
    truth = {u: truth[u] for u in users}

    # The verifier must build the SAME model the runner built, or a mismatch means only
    # that the two disagree about the model. Each of these was previously omitted and
    # produced false failures: `user_ids` (Random fell back to pooled row positions),
    # `capacity`/`dense` (PureSVD fell back to the pooled rank cap the fix removed), and
    # `frac` (every subsample unit was rebuilt at full donor volume).
    unit_donors = [tuple(d) for d in rec["donors"]]
    frac = float(rec.get("frac", 1.0))
    edge_lists = [(blocks[target].train_u, blocks[target].train_i)]
    for donor in unit_donors:
        b = blocks[donor]
        seed = stable_seed(target, donor, rec.get("rep", 0), round(frac, 4))
        edge_lists.append(runner.subsample((b.train_u, b.train_i), frac, seed))

    R, cols, rows = E.build_matrix(edge_lists, session.catalog, session.eval_users)
    capacity, dense = E.target_capacity(session, blocks[target])
    params = dict(rec.get("params") or {})
    model = Mod.build(rec["model"], R, cols, user_rows=rows,
                      user_ids=E.row_to_user_id(rows, session.eval_users),
                      capacity=capacity, dense=dense, **params)

    # Batched, for the same reason `experiment.evaluate` batches: scoring all 20,000
    # evaluation users of `uk Books` at once is 20,000 x 35,324, which is 2.8 GB in
    # float32 and 5.7 GB promoted to float64 - a MemoryError on this machine. The
    # stratified sampler deliberately draws from the most expensive quartile, so the
    # verifier could not run on precisely the contexts the conclusions lean on.
    n_items = len(session.catalog)
    batch = int(max(16, E.RANK_KEY_BUDGET // max(n_items, 1)))
    ndcgs, recalls = [], []
    for start in range(0, len(users), batch):
        block = users[start:start + batch]
        block_rows = rows[start:start + batch]
        scores = model.score(block_rows).astype(np.float64)
        for r, u in enumerate(block):
            for j in masked.get(u, ()):
                scores[r, j] = -np.inf
        n, rc = reference_metrics(scores, [truth[u] for u in block], cutoff=10,
                                  reduce=False)
        ndcgs += n
        recalls += rc
        del scores
    ref_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    ref_recall = float(np.mean(recalls)) if recalls else 0.0

    deltas = {
        "ndcg@10": abs(ref_ndcg - rec["ndcg@10"]),
        "recall@10": abs(ref_recall - rec["recall@10"]),
    }
    ok = all(v <= tol for v in deltas.values())
    return ok, deltas, ref_ndcg, ref_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="baseline_random.jsonl")
    ap.add_argument("--sample", type=float, default=0.02)
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = RESULTS_DIR / args.file
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if not r.get("error") and "ndcg@10" in r]
    # Subsampling favours cheap contexts so the verifier itself stays overnight-sized.
    # Stratify by cost so the largest contexts - previously excluded outright by taking
    # only the cheapest 90% - are represented. C2 asks for agreement on a sample of the
    # FULL run, and the biggest contexts are the ones the conclusions lean on.
    records.sort(key=lambda r: r.get("n_eval_users", 0) * r.get("n_target_items", 0))
    rng = random.Random(args.seed)
    want = min(args.max, max(1, int(len(records) * args.sample)))
    n_strata = max(1, min(4, want))
    stride = max(1, len(records) // n_strata)
    picks = []
    for s_i in range(n_strata):
        stratum = records[s_i * stride:(s_i + 1) * stride] or records
        picks += rng.sample(stratum, k=min(len(stratum), max(1, want // n_strata)))

    data = D.load()
    blocks_cache = {}
    failures = 0
    for n, rec in enumerate(picks, 1):
        split = rec["split"]
        if split not in blocks_cache:
            blocks_cache[split] = E.load_blocks(data, split)
        ok, deltas, ref_n, ref_r = verify_record(rec, data, blocks_cache[split],
                                                 tol=args.tol)
        flag = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{n}/{len(picks)}] {flag} {rec['model']:<11} "
              f"{rec['target_market']} {rec['target_category'][:20]:<20} "
              f"k={rec['k']:<3} recorded {rec['ndcg@10']:.6f} "
              f"reference {ref_n:.6f} delta {deltas['ndcg@10']:.2e}", flush=True)

    print(f"\n{len(picks) - failures}/{len(picks)} agree within {args.tol:g}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
