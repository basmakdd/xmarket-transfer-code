"""Stage 1e - turn the run records into effects, tables and figures.

Every reported quantity is an effect *against that context's own target-only fit* under
the identical model, split, evaluation users and catalog, so the only thing that differs
between a record and its reference is the borrowed data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats as S

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
METRIC = "ndcg@10"


def load_records(*files) -> pd.DataFrame:
    rows = []
    for f in files:
        path = RESULTS / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["target"] = list(zip(df["target_market"], df["target_category"]))
    # Guard against the duplicate-append defect that affected the original results file.
    # code_version included: records from different model definitions must never be
    # collapsed onto one another by the dedup.
    key = ["split", "model", "target_market", "target_category", "channel",
           "order", "rep", "k", "frac"]
    if "code_version" in df.columns:
        key = key + ["code_version"]
    before = len(df)
    df = df.drop_duplicates(key, keep="last").reset_index(drop=True)
    if len(df) != before:
        print(f"  note: dropped {before - len(df)} duplicate unit records")
    return df


def attach_baseline(df: pd.DataFrame, metrics=(METRIC, "ndcg@5", "recall@10",
                                               "coverage", "novelty")) -> pd.DataFrame:
    """Join every record to its own k=0 reference and compute effects.

    The reduced-catalog columns are included so that C6 robustness applies to the
    BORROWING conclusions, not merely to target-only levels. Previously `ndcg@10|c*` was
    recorded on every unit and differenced on none, so "conclusions are checked against
    the nested candidate-set results" had no code path at all. The frozen-hyperparameter
    metric is differenced for the same reason: without it the headline contrast is only
    ever reported at re-tuned capacity.
    """
    base = df[(df["k"] == 0) & df["error"].isna()]
    metrics = tuple(metrics) + tuple(
        c for c in df.columns if c.startswith("ndcg@10|c"))
    keep = ["split", "model", "target_market", "target_category"]
    base = base[keep + list(metrics)].rename(
        columns={m: f"base_{m}" for m in metrics})
    out = df.merge(base, on=keep, how="left", validate="many_to_one")
    for m in metrics:
        out[f"delta_{m}"] = out[m] - out[f"base_{m}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_delta"] = np.where(
            out[f"base_{METRIC}"] > 0,
            out[f"delta_{METRIC}"] / out[f"base_{METRIC}"], np.nan)
    # The frozen contrast differences against the SAME k=0 reference. The target-only fit
    # already sits at the target-only hyperparameters - that is what "frozen" means - so
    # `frozen_ndcg@10 - base_ndcg@10` holds capacity fixed across both arms, while
    # `delta_ndcg@10` lets it move with the data. Differencing frozen against a frozen
    # baseline column would be wrong twice over: k=0 records carry no frozen column at
    # all, so every such difference came out NaN and the table silently vanished.
    if "frozen_ndcg@10" in out.columns:
        out["delta_frozen_ndcg@10"] = out["frozen_ndcg@10"] - out[f"base_{METRIC}"]
    if "frozen_recall@10" in out.columns:
        out["delta_frozen_recall@10"] = (out["frozen_recall@10"]
                                         - out["base_recall@10"])
    out["log_borrowed"] = np.log10(out["borrowed_edges"].clip(lower=1))
    out["borrow_ratio"] = out["borrowed_edges"] / out["n_pooled_edges"].where(
        out["n_pooled_edges"] > 0)
    return out


def donor_covariates(overlap: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Coupling bandwidth of a record's donor set: the MAXIMUM over its donors.

    Max, not sum: the question these covariates answer is whether ANY donor reaches far
    enough into the target to matter, and a sum over many weak donors would masquerade as
    one strong one.
    """
    idx = overlap.set_index(["target_market", "target_category",
                             "donor_market", "donor_category"])
    item_cov = idx["shared_item_interaction_cover"].to_dict()
    user_cov = idx["shared_user_interaction_cover"].to_dict()
    test_cov = idx["shared_test_item_cover"].to_dict()

    best_item, best_user, best_test = [], [], []
    for tm, tc, donors in zip(df["target_market"], df["target_category"], df["donors"]):
        ic = [item_cov.get((tm, tc, d[0], d[1]), 0.0) for d in donors]
        uc = [user_cov.get((tm, tc, d[0], d[1]), 0.0) for d in donors]
        xc = [test_cov.get((tm, tc, d[0], d[1]), 0.0) for d in donors]
        best_item.append(max(ic) if ic else 0.0)
        best_user.append(max(uc) if uc else 0.0)
        best_test.append(max(xc) if xc else 0.0)
    df = df.copy()
    df["max_item_cover"] = best_item
    df["max_user_cover"] = best_user
    df["max_test_cover"] = best_test
    df["max_cover"] = np.maximum(df["max_item_cover"], df["max_user_cover"])
    return df



# --------------------------------------------------------------------- per-user effects
def out_stem_for(record: dict) -> str:
    """Which results file a record belongs to - and therefore which per-user directory.

    Hard-coding `full_random` meant a temporal-split run's vectors were looked up in the
    random split's directory, missed, and silently dropped: the whole temporal half of
    criterion C3 would have vanished without an error or a count.
    """
    return f"full_{record['split']}"


def per_user_file(record: dict, out_stem: str | None = None) -> Path:
    """Locate the per-user metric vectors saved alongside a unit."""
    import design as Design
    from util import stable_hex
    out_stem = out_stem or out_stem_for(record)
    unit = Design.Unit(
        record["split"], record["model"],
        (record["target_market"], record["target_category"]),
        record["channel"], record["order"], int(record["rep"]),
        int(record["k"]), float(record["frac"]),
        tuple(tuple(d) for d in record.get("donors", [])),
    )
    tag = stable_hex(*unit.key())
    root = RESULTS / f"{out_stem}_peruser"
    return root / (f"{record['model']}_{record['target_market']}_"
                   f"{record['target_category']}_{tag}.npz")


def load_per_user(record: dict, metric: str = METRIC, out_stem: str | None = None):
    path = per_user_file(record, out_stem)
    if not path.exists():
        return None
    with np.load(path) as data:
        if metric not in data:
            return None
        return data[metric].astype(np.float64)


def paired_user_bootstrap(record: dict, baseline: dict, metric: str = METRIC,
                          n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap the PAIRED per-user difference between a unit and its target-only fit.

    Resampling the difference, not the two arms separately, is what makes the interval a
    statement about the effect. Both arms score the identical evaluation users in the
    identical order - `TargetSession` fixes them per target and every donor set reuses
    them - so the vectors align element-wise and the pairing is exact.

    This is criterion C3's per-context interval. Until it existed the per-user vectors
    were being written and never read, and C3 was not satisfiable from the results files.
    """
    a = load_per_user(record, metric)
    b = load_per_user(baseline, metric)
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return None
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "p": bootstrap_p_value(diff, n_boot=n_boot, seed=seed),
            "n_users": int(diff.size),
            "share_users_improved": float((diff > 0).mean())}



def bootstrap_p_value(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> float:
    """Two-sided bootstrap p for "the mean paired difference is zero".

    The per-context intervals alone cannot carry FDR control, and C3 asks for per-context
    tests with raw and adjusted values. This inverts the same user-level bootstrap the
    interval uses, so the p and the interval always tell the same story.
    """
    if diff.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    centred = means - means.mean()
    p = float((np.abs(centred) >= abs(diff.mean())).mean())
    return min(1.0, max(p, 1.0 / n_boot))


def table_chance_comparable(df: pd.DataFrame, margin: float = 2.0) -> pd.DataFrame:
    """Contexts where a model is not clearly above the chance-level control.

    C3 forbids resting a conclusion on such a context and requires each to be named.
    Without this the reader cannot tell a real near-zero effect from a context where no
    method has anything to work with - which is exactly the confusion that let the
    original study's corrupted markets pass as a scientific finding.
    """
    base = df[(df["k"] == 0) & df["error"].isna()]
    chance = {(r["split"], r["target_market"], r["target_category"]): r[METRIC]
              for r in base[base["model"] == "Random"].to_dict("records")}
    rows = []
    for r in base[base["model"] != "Random"].to_dict("records"):
        ref = chance.get((r["split"], r["target_market"], r["target_category"]))
        if ref is None or not np.isfinite(ref):
            continue
        rows.append({"model": r["model"], "market": r["target_market"],
                     "category": r["target_category"], "model_ndcg10": r[METRIC],
                     "chance_ndcg10": ref,
                     "ratio": r[METRIC] / ref if ref > 0 else np.inf})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["comparable_to_chance"] = out["ratio"] < margin
    return out.round(5).sort_values("ratio")


def table_per_context_effects(df: pd.DataFrame, channel: str = "XM",
                              order: str = "full", metric: str = METRIC,
                              n_boot: int = 2000) -> pd.DataFrame:
    """Per-context effect of a whole channel, each with its own user-level interval."""
    base = {(r["model"], r["target_market"], r["target_category"]): r
            for r in df[(df["k"] == 0) & df["error"].isna()].to_dict("records")}
    rows, skipped_no_ref, skipped_no_vectors = [], 0, 0
    sub = df[(df["channel"] == channel) & (df["order"] == order) & df["error"].isna()]
    for rec in sub.to_dict("records"):
        ref = base.get((rec["model"], rec["target_market"], rec["target_category"]))
        if ref is None:
            skipped_no_ref += 1
            continue
        boot = paired_user_bootstrap(rec, ref, metric, n_boot=n_boot)
        if boot is None:
            # A missing, truncated or mismatched per-user file. Counted, never silent:
            # an unreported skip is indistinguishable from a context that had no effect.
            skipped_no_vectors += 1
            continue
        rows.append({"model": rec["model"], "market": rec["target_market"],
                     "category": rec["target_category"], "k": rec["k"], **boot})
    if skipped_no_ref or skipped_no_vectors:
        print(f"  [{channel}] skipped {skipped_no_ref} without a k=0 reference, "
              f"{skipped_no_vectors} without usable per-user vectors, "
              f"of {len(sub)} attempted")
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # An interval that excludes zero is a real effect for that context.
    out["excludes_zero"] = (out["lo"] > 0) | (out["hi"] < 0)
    # Per-context multiplicity control. This is the family C3 names; the cross-context
    # family is corrected separately in `table_channel_effects`.
    out["p_fdr"] = S.benjamini_hochberg(out["p"].to_numpy())
    out["significant_fdr"] = out["p_fdr"] < 0.05
    return out.round(5)


# ------------------------------------------------------------------------------ tables
def table_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """The 'own data only' reference: what each model achieves with no borrowing."""
    base = df[(df["k"] == 0) & df["error"].isna()]
    out = base.groupby("model").agg(
        contexts=(METRIC, "size"),
        mean_ndcg10=(METRIC, "mean"),
        median_ndcg10=(METRIC, "median"),
        mean_recall10=("recall@10", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_seconds=("elapsed_s", "mean"),
    ).round(4)
    return out.sort_values("mean_ndcg10", ascending=False)


def table_channel_effects(df: pd.DataFrame, only_full: bool = True) -> pd.DataFrame:
    """Effect of taking a whole channel, per model, with cluster intervals."""
    sub = df[df["error"].isna() & (df["k"] > 0)]
    if only_full:
        sub = sub[sub["order"] == "full"]
    rows = []
    for (model, channel), g in sub.groupby(["model", "channel"]):
        rec = {"model": model, "channel": channel}
        rec.update(S.paired_summary(g, f"delta_{METRIC}", cluster="target_market"))
        # C3 asks for BOTH clusterings: contexts share markets AND categories, and
        # either dependence on its own understates the uncertainty.
        by_cat = S.cluster_bootstrap(g, f"delta_{METRIC}", "target_category")
        rec["cat_lo"], rec["cat_hi"] = by_cat["lo"], by_cat["hi"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = S.benjamini_hochberg(out["wilcoxon_p"].to_numpy())
    cols = ["model", "channel", "n", "mean", "median", "share_positive",
            "rank_biserial", "cluster_lo", "cluster_hi", "cat_lo", "cat_hi",
            "wilcoxon_p", "p_fdr"]
    return out[cols].round(5).sort_values(["channel", "model"])


def table_dose(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Mean effect as donors accumulate, by ordering rule."""
    sub = df[(df["channel"] == channel) & df["error"].isna()
             & (~df["order"].isin(["subsample"]))]
    rows = []
    for (model, order, k), g in sub.groupby(["model", "order", "k"]):
        rows.append({
            "model": model, "order": order, "k": k, "n": len(g),
            "mean_delta": g[f"delta_{METRIC}"].mean(),
            "median_delta": g[f"delta_{METRIC}"].median(),
            "share_positive": (g[f"delta_{METRIC}"] > 0).mean(),
        })
    return pd.DataFrame(rows).round(5)


def table_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Effect as each donor's volume is scaled, with the donor set held fixed."""
    sub = df[(df["order"].isin(["subsample", "full"])) & df["error"].isna()
             & (df["k"] > 0)]
    rows = []
    for (model, channel, frac), g in sub.groupby(["model", "channel", "frac"]):
        rows.append({
            "model": model, "channel": channel, "frac": frac, "n": len(g),
            "mean_delta": g[f"delta_{METRIC}"].mean(),
            "median_delta": g[f"delta_{METRIC}"].median(),
            "share_positive": (g[f"delta_{METRIC}"] > 0).mean(),
            "mean_borrowed": g["borrowed_edges"].mean(),
        })
    return pd.DataFrame(rows).round(5)


def table_null_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Falsification test for the mechanism argument the design's pruning rests on.

    `design.BLIND` lists (model, channel) pairs that *cannot* move the metric: the donor
    leaves the target-catalog Gram bit-identical, or the model ignores the data entirely.
    Every one of these must come out at exactly zero. A non-zero entry means the mechanism
    model is wrong and the design has to be re-run rather than reported (criterion C2).

    Pairs that merely duplicate a cheaper model are NOT listed here - they respond, and
    checking them against zero would raise a false alarm. `table_redundancy` covers those.
    """
    import design as Design
    sub = df[df["error"].isna() & (df["k"] > 0)].copy()
    sub["pair"] = list(zip(sub["model"], sub["channel"]))
    blind = sub[sub["pair"].isin(Design.BLIND)]
    if blind.empty:
        return pd.DataFrame()
    out = blind.groupby(["model", "channel"]).agg(
        n=(f"delta_{METRIC}", "size"),
        max_abs_delta=(f"delta_{METRIC}", lambda s: s.abs().max()),
    ).reset_index()
    out["passes"] = out["max_abs_delta"] == 0.0
    return out


def table_redundancy(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs that should reproduce a cheaper model exactly, rather than be null."""
    import design as Design
    key = ["split", "target_market", "target_category", "channel", "order", "rep",
           "k", "frac"]
    rows = []
    for (model, channel), twin in Design.REDUNDANT.items():
        a = df[(df.model == twin) & (df.channel == channel)][key + [METRIC]]
        b = df[(df.model == model) & (df.channel == channel)][key + [METRIC]]
        merged = a.merge(b, on=key, suffixes=("_twin", "_model"))
        if merged.empty:
            continue
        diff = (merged[f"{METRIC}_twin"] - merged[f"{METRIC}_model"]).abs()
        rows.append({"model": model, "channel": channel, "equals": twin,
                     "n": len(merged), "max_abs_diff": float(diff.max()),
                     "passes": bool(diff.max() == 0.0)})
    return pd.DataFrame(rows)


def table_candidate_sets(df: pd.DataFrame) -> pd.DataFrame:
    """The same fitted models scored against nested candidate sets.

    This is what RQ4 asked and could not answer, because the original compared regimes
    that also differed in training budget.
    """
    cols = [c for c in df.columns if c.startswith("ndcg@10|c")] + [METRIC]
    sub = df[df["error"].isna() & (df["k"] == 0)]
    if not cols or sub.empty:
        return pd.DataFrame()
    out = sub.groupby("model")[cols].mean().round(4)
    return out.rename(columns={METRIC: "full catalog"})



def table_dose_shape(df: pd.DataFrame, channel: str = "XM") -> pd.DataFrame:
    """Quantify the dose curve: slope, monotonicity and any turning point.

    C4 asks for a reported SHAPE, not a drawn one. The second half of the research
    question - how the effect changes as more is borrowed - was previously only plotted,
    never estimated.
    """
    sub = df[(df["channel"] == channel) & df["error"].isna()
             & (~df["order"].isin(["subsample", "null-check"]))]
    rows = []
    for model, g in sub.groupby("model"):
        curve = g.groupby("k")[f"delta_{METRIC}"].mean().sort_index()
        if len(curve) < 3:
            continue
        ks = curve.index.to_numpy(dtype=float)
        vals = curve.to_numpy()
        slope = float(np.polyfit(np.log10(np.maximum(ks, 1)), vals, 1)[0])
        diffs = np.diff(vals)
        shape = ("increasing" if np.all(diffs >= -1e-9)
                 else "decreasing" if np.all(diffs <= 1e-9) else "non-monotone")
        turns = np.flatnonzero(np.sign(diffs[:-1]) * np.sign(diffs[1:]) < 0)
        rows.append({
            "model": model, "channel": channel, "n_steps": len(curve),
            "delta_at_k1": float(vals[0]), "delta_at_kmax": float(vals[-1]),
            "slope_per_decade": slope, "shape": shape,
            "turning_k": int(ks[turns[0] + 1]) if turns.size else None,
            "rho_with_dose": S.spearman_with_ci(
                g["log_borrowed"], g[f"delta_{METRIC}"], n_boot=500)["rho"],
        })
    return pd.DataFrame(rows).round(5)


def table_predictability(df: pd.DataFrame, channel: str = "XM") -> pd.DataFrame:
    """C5: can the SIGN of the effect be predicted before fitting anything?

    Only covariates a platform could compute in advance are admitted. `max_test_cover` is
    deliberately excluded: it derives from the test split, so predicting the effect with
    it would be circular - a platform cannot know its future held-out items.
    """
    pre_fit = ["max_item_cover", "max_user_cover", "max_cover", "log_borrowed",
               "borrow_ratio"]
    sub = df[(df["channel"] == channel) & df["error"].isna() & (df["k"] > 0)]
    rows = []
    for model, g in sub.groupby("model"):
        y = g[f"delta_{METRIC}"]
        if y.notna().sum() < 10 or y.nunique() < 3:
            continue
        for cov in pre_fit:
            if cov not in g.columns or g[cov].nunique() < 3:
                continue
            res = S.spearman_with_ci(g[cov], y, n_boot=1000)
            rows.append({"model": model, "channel": channel, "covariate": cov,
                         "rho": res["rho"], "lo": res["lo"], "hi": res["hi"],
                         "n": res["n"],
                         "excludes_zero": bool(res["lo"] > 0 or res["hi"] < 0),
                         "sign_accuracy": float(
                             ((g[cov] > g[cov].median()) == (y > 0)).mean())})
    return pd.DataFrame(rows).round(4)


def table_frozen_vs_tuned(df: pd.DataFrame) -> pd.DataFrame:
    """Does re-tuning per donor set change the measured effect?

    The tuned contrast estimates best-of-grid-with-donors minus best-of-grid-without, so
    capacity moves with the data. The frozen arm holds hyperparameters at the target-only
    choice. Where the two agree the causal reading is clean; where they diverge, the
    divergence is itself the finding rather than a confound to argue about.
    """
    if "delta_frozen_ndcg@10" not in df.columns:
        return pd.DataFrame()
    sub = df[df["error"].isna() & (df["k"] > 0) & df["delta_frozen_ndcg@10"].notna()]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (model, channel), g in sub.groupby(["model", "channel"]):
        tuned = g[f"delta_{METRIC}"]
        frozen = g["delta_frozen_ndcg@10"]
        rows.append({
            "model": model, "channel": channel, "n": len(g),
            "tuned_mean": float(tuned.mean()), "frozen_mean": float(frozen.mean()),
            "mean_gap": float((tuned - frozen).mean()),
            "max_abs_gap": float((tuned - frozen).abs().max()),
            "sign_agrees": float((np.sign(tuned) == np.sign(frozen)).mean()),
        })
    return pd.DataFrame(rows).round(5)


def table_candidate_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """C6: does the borrowing conclusion survive a reduced candidate set?"""
    cols = [c for c in df.columns if c.startswith("delta_ndcg@10|c")]
    if not cols:
        return pd.DataFrame()
    sub = df[df["error"].isna() & (df["k"] > 0) & (df["order"] == "full")]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (model, channel), g in sub.groupby(["model", "channel"]):
        rec = {"model": model, "channel": channel, "n": len(g),
               "delta_full": float(g[f"delta_{METRIC}"].mean())}
        for c in cols:
            size = c.split("|")[-1]
            rec["delta_" + size] = float(g[c].mean())
            rec["sign_agrees_" + size] = float(
                (np.sign(g[c]) == np.sign(g[f"delta_{METRIC}"])).mean())
        rows.append(rec)
    return pd.DataFrame(rows).round(5)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_records("baseline_random.jsonl", "full_random.jsonl",
                      "baseline_temporal.jsonl", "full_temporal.jsonl")
    if df.empty:
        print("no records yet")
        return
    df = attach_baseline(df)
    overlap = pd.read_parquet(RESULTS / "overlap_random.parquet")
    df = donor_covariates(overlap, df)
    df.to_parquet(RESULTS / "effects.parquet", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Target-only baseline: what each model achieves with its own data ===")
    print(table_baseline(df).to_string())

    print("\n=== Nested candidate sets (same fitted model) ===")
    print(table_candidate_sets(df).to_string())

    ch = table_channel_effects(df)
    if not ch.empty:
        print("\n=== Effect of borrowing a whole channel ===")
        print(ch.to_string(index=False))

    for channel in ("XM", "XC"):
        pc = table_per_context_effects(df, channel=channel)
        if not pc.empty:
            n_sig = int(pc["excludes_zero"].sum())
            print(f"\n=== C3 per-context effects, {channel}, user-level bootstrap ===")
            print(f"{len(pc)} context-model pairs | {n_sig} with intervals excluding zero")
            print(pc.head(12).to_string(index=False))

    chance = table_chance_comparable(df)
    if not chance.empty:
        n_bad = int(chance["comparable_to_chance"].sum())
        print(f"\n=== C3 contexts within 2x of chance ({n_bad} of {len(chance)}) ===")
        if n_bad:
            print(chance[chance["comparable_to_chance"]].head(15).to_string(index=False))

    for channel in ("XM", "XC", "XX"):
        shape = table_dose_shape(df, channel)
        if not shape.empty:
            print(f"\n=== C4 dose shape, {channel} ===")
            print(shape.to_string(index=False))

    frozen = table_frozen_vs_tuned(df)
    if not frozen.empty:
        print("\n=== Tuned vs frozen hyperparameters (identification check) ===")
        print(frozen.to_string(index=False))

    cand = table_candidate_robustness(df)
    if not cand.empty:
        print("\n=== C6 borrowing effect under reduced candidate sets ===")
        print(cand.to_string(index=False))

    for channel in ("XM", "XC"):
        pred = table_predictability(df, channel)
        if not pred.empty:
            print(f"\n=== C5 predictability from pre-fit covariates, {channel} ===")
            print(pred.to_string(index=False))

    red = table_redundancy(df)
    if not red.empty:
        print("\n=== C2 redundancy checks ===")
        print(red.to_string(index=False))

    nulls = table_null_checks(df)
    if not nulls.empty:
        print("\n=== Null checks (provably blind model x channel combinations) ===")
        print(nulls.to_string(index=False))


if __name__ == "__main__":
    main()
