"""Calendar-cutoff (prospective) variant of the preprocessing.

Unlike the per-user splits in data.py, every split here respects one calendar per category:

* For each category, two cutoff dates are fixed from the raw (deduplicated) review dates of
  all markets that sell it: T_val = 80th percentile, T_test = 90th percentile. Because the
  dates are common to all markets of a category, foreign training data are never later
  than the target's own cutoff.
* The adaptive core filter (same thresholds as data.py) is applied to interactions dated
  before T_val only, so later activity cannot influence which products enter. It fixes the
  product catalog and the context's core level k. Users are NOT filtered: returning users
  are exactly the population a prospective evaluation can score, and a user-side k-core
  would remove most of them (median test users per context: 7.5 with user filtering vs
  44 without).
* train = all interactions before T_val with catalog products; val = interactions in
  [T_val, T_test); test = interactions on or after T_test. Validation and test rows are kept
  only for users with at least one training interaction and for catalog products (cold
  users and products unseen before the cutoff cannot be scored; both are counted in
  contexts_calendar.parquet).
* Models are fitted on train only (as in the per-user splits); validation selects
  hyperparameters; test is scored with the training-only fit.

Launch candidates are products outside the pre-cutoff local catalog (not retained by the
pre-cutoff core filter of the target context; such a product may still have a few local
interactions before T_val that the filter removed), with foreign pre-cutoff interactions
(the product is in the pre-cutoff core catalog of at least one other market of the same
category) and a local interaction after T_val. The candidate rows are those local
interactions on or after T_val, by users with at least one local training interaction; they
are written separately to data/launch_calendar.parquet for the calendar launch experiment
(paper/code/launch3.py, split `calendar`).

Outputs: data/interactions_calendar.parquet, data/launch_calendar.parquet,
results/contexts_calendar.parquet, results/dropped_contexts_calendar.json,
results/calendar_cutoffs.csv
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

import data as D

Q_VAL, Q_TEST = 0.80, 0.90


def build():
    raw = {}
    with zipfile.ZipFile(D.ZIP_PATH) as zf:
        for category in D.CATEGORIES:
            for market in D.MARKETS:
                df = D.read_context(zf, market, category)
                if df is not None and not df.empty:
                    raw[(market, category)] = df
    cut = {}
    for category in D.CATEGORIES:
        dates = [pd.to_datetime(df["date"]) for (m, c), df in raw.items() if c == category]
        if not dates:
            continue
        allx = pd.concat(dates)
        cut[category] = (allx.quantile(Q_VAL).normalize(), allx.quantile(Q_TEST).normalize())

    kept, dropped, stats, pre_items = {}, [], [], {}
    for (market, category), df in sorted(raw.items()):
        t_val, t_test = cut[category]
        dt = pd.to_datetime(df["date"])
        pre = df[dt < t_val]
        core, kk = D.adaptive_core(pre) if len(pre) else (pre, None)
        if kk is None:
            dropped.append([market, category, "below thresholds before cutoff"])
            continue
        items = set(core["item"])
        core = pre[pre["item"].isin(items)]
        users = set(core["user"])
        post = df[dt >= t_val].copy()
        post_dt = pd.to_datetime(post["date"])
        post["split_calendar"] = np.where(post_dt < t_test, "val", "test")
        in_u = post["user"].isin(users)
        in_i = post["item"].isin(items)
        keep_post = post[in_u & in_i]
        tr = core.assign(split_calendar="train")
        kept[(market, category)] = (pd.concat([tr, keep_post]), post[in_u & ~in_i])
        pre_items[(market, category)] = items
        stats.append({
            "market": market, "category": category, "core": kk,
            "t_val": t_val.date().isoformat(), "t_test": t_test.date().isoformat(),
            "raw": len(df), "raw_pre_cutoff": len(pre), "train": len(core),
            "val": int((keep_post.split_calendar == "val").sum()),
            "test": int((keep_post.split_calendar == "test").sum()),
            "post_dropped_cold_user": int((~in_u).sum()),
            "post_dropped_unseen_item": int((in_u & ~in_i).sum()),
            "users": len(users), "items": len(items),
        })

    frames, launch = [], []
    for (market, category), (tbl, newitem) in kept.items():
        frames.append(tbl.assign(market=market, category=category))
        foreign = set().union(*[pre_items[k] for k in pre_items if k[1] == category and k[0] != market])
        cand = newitem[newitem["item"].isin(foreign)]
        if len(cand):
            launch.append(cand.assign(market=market, category=category))
    data = pd.concat(frames, ignore_index=True)
    lc = pd.concat(launch, ignore_index=True) if launch else data.iloc[0:0]
    both = pd.concat([data[["market", "user", "item"]], lc[["market", "user", "item"]]])
    ukeys = pd.Index(sorted(set(both["market"] + "::" + both["user"])))
    ikeys = pd.Index(sorted(set(both["item"])))
    for t in (data, lc):
        t["u"] = ukeys.get_indexer(t["market"] + "::" + t["user"]).astype(np.int32)
        t["i"] = ikeys.get_indexer(t["item"]).astype(np.int32)
    cols = ["market", "category", "u", "i", "date", "split_calendar"]
    data, lc = data[cols], lc[cols]
    D.OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(D.OUT_DIR / "interactions_calendar.parquet", index=False)
    lc.to_parquet(D.OUT_DIR / "launch_calendar.parquet", index=False)
    st = pd.DataFrame(stats)
    st.to_parquet(D.RESULTS_DIR / "contexts_calendar.parquet", index=False)
    (D.RESULTS_DIR / "dropped_contexts_calendar.json").write_text(json.dumps(dropped, indent=1), encoding='utf-8', newline='\n')
    pd.DataFrame([{"category": c, "t_val": a.date(), "t_test": b.date()} for c, (a, b) in cut.items()]) \
        .to_csv(D.RESULTS_DIR / "calendar_cutoffs.csv", index=False, lineterminator='\n')
    return data, lc, st, dropped


def load(launch: bool = False) -> pd.DataFrame:
    name = "launch_calendar.parquet" if launch else "interactions_calendar.parquet"
    return pd.read_parquet(D.OUT_DIR / name)


if __name__ == "__main__":
    data, lc, st, dropped = build()
    print(f"{len(st)} contexts kept, {len(dropped)} dropped; {len(data):,} interactions; "
          f"{len(lc):,} launch-candidate interactions")
    print(st[["train", "val", "test", "post_dropped_cold_user", "post_dropped_unseen_item"]].sum().to_string())
    import overlap as O
    df = O.build_from(data, "calendar")
    df.to_parquet(D.RESULTS_DIR / "overlap_calendar.parquet", index=False)
    print(f"overlap_calendar: {len(df)} ordered pairs")
