"""Stage 1b - structural overlap map over all ordered (target, donor) context pairs.

No model is fitted here. The question is prior to any model: *can* a donor context carry
anything to a target at all, and through which coupling? The answer supplies the
covariates for the dose model in Stage 1e and for any donor-selection rule in Stage 2.

Channels:
  XM  same category, different market  -> items may be shared, users cannot be
  XC  same market, different category  -> users may be shared, items cannot be
  XX  both differ                      -> neither is shared (control channel)

Only *training* interactions of a donor are ever counted, here and in the experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data as D

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def channel(target: tuple[str, str], donor: tuple[str, str]) -> str:
    same_market = target[0] == donor[0]
    same_category = target[1] == donor[1]
    if same_category and not same_market:
        return "XM"
    if same_market and not same_category:
        return "XC"
    return "XX"


def gini(counts: np.ndarray) -> float:
    if counts.size == 0:
        return float("nan")
    x = np.sort(counts.astype(np.float64))
    n = x.size
    total = x.sum()
    if total <= 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two popularity distributions."""
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def build(split: str = "random") -> pd.DataFrame:
    return build_from(D.load(), split)


def build_from(data: pd.DataFrame, split: str) -> pd.DataFrame:
    train = data[data[f"split_{split}"] == "train"]
    test = data[data[f"split_{split}"] == "test"]

    keys = sorted({(m, c) for m, c in zip(data["market"], data["category"])})
    train_groups = {k: g for k, g in train.groupby(["market", "category"], sort=True)}
    test_groups = {k: g for k, g in test.groupby(["market", "category"], sort=True)}

    items, users, item_pop, test_items, profile = {}, {}, {}, {}, {}
    for k in keys:
        g = train_groups[k]
        i_vals, i_counts = np.unique(g["i"].to_numpy(), return_counts=True)
        items[k] = i_vals
        item_pop[k] = dict(zip(i_vals.tolist(), i_counts.tolist()))
        users[k] = np.unique(g["u"].to_numpy())
        profile[k] = g.groupby("u").size().to_numpy()
        tg = test_groups.get(k)
        test_items[k] = np.unique(tg["i"].to_numpy()) if tg is not None else np.array([], np.int32)

    rows = []
    for target in keys:
        t_items, t_users = items[target], users[target]
        t_train = train_groups[target]
        t_item_counts = t_train["i"].to_numpy()
        t_user_counts = t_train["u"].to_numpy()
        for donor in keys:
            if donor == target:
                continue
            ch = channel(target, donor)
            d_items, d_users = items[donor], users[donor]

            shared_items = np.intersect1d(t_items, d_items, assume_unique=True)
            shared_users = np.intersect1d(t_users, d_users, assume_unique=True)

            n_si, n_su = len(shared_items), len(shared_users)
            rec = {
                "target_market": target[0], "target_category": target[1],
                "donor_market": donor[0], "donor_category": donor[1],
                "channel": ch,
                "target_train": len(t_train), "donor_train": len(train_groups[donor]),
                "target_items": len(t_items), "donor_items": len(d_items),
                "target_users": len(t_users), "donor_users": len(d_users),
                "volume_ratio": len(train_groups[donor]) / max(len(t_train), 1),
                "shared_items": n_si, "shared_users": n_su,
                # What share of the target is reachable through the shared identities?
                "item_overlap_target": n_si / max(len(t_items), 1),
                "user_overlap_target": n_su / max(len(t_users), 1),
                "shared_item_interaction_cover":
                    float(np.isin(t_item_counts, shared_items).mean()) if n_si else 0.0,
                "shared_user_interaction_cover":
                    float(np.isin(t_user_counts, shared_users).mean()) if n_su else 0.0,
                "shared_test_item_cover":
                    float(np.isin(test_items[target], shared_items).mean())
                    if n_si and len(test_items[target]) else 0.0,
                "donor_gini": gini(np.fromiter(item_pop[donor].values(), float)),
                "donor_median_profile": float(np.median(profile[donor])),
            }
            if n_si:
                p = np.array([item_pop[target].get(int(i), 0) for i in shared_items], float)
                q = np.array([item_pop[donor].get(int(i), 0) for i in shared_items], float)
                rec["popularity_js"] = js_divergence(p, q)
            else:
                rec["popularity_js"] = float("nan")
            rows.append(rec)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("random", "temporal"):
        df = build(split)
        df.to_parquet(RESULTS_DIR / f"overlap_{split}{D.DATA_TAG}.parquet", index=False)
        print(f"\n=== split: {split} | {len(df)} ordered pairs ===")
        summary = df.groupby("channel").agg(
            pairs=("channel", "size"),
            shared_items_median=("shared_items", "median"),
            shared_users_median=("shared_users", "median"),
            pct_with_shared_items=("shared_items", lambda s: 100 * (s > 0).mean()),
            pct_with_shared_users=("shared_users", lambda s: 100 * (s > 0).mean()),
        ).round(2)
        print(summary.to_string())
