"""Stage 1e - statistics.

Three choices for how uncertainty is reported:

* **Effect sizes with intervals, not three-seed p-values**. These models
  are deterministic given the data, so seeds carry no information; uncertainty comes from
  a bootstrap over *test users*, which is the population the metric actually averages
  over and gives far more resolution than three runs.
* **Dependence between contexts is modelled, not ignored**. Contexts share
  markets and categories, so a naive paired test across 74 contexts overstates precision.
  The cluster bootstrap resamples whole markets, or whole categories, instead of contexts.
* **Multiplicity is controlled**. Per-context tests get Benjamini-Hochberg
  FDR, and both the raw and adjusted values are reported.

Nothing here dichotomises on a metric floor: effects are reported
continuously against the baseline level, with thresholds available only as an overlay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    q = p[ok]
    n = q.size
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    out[ok] = adjusted
    return out


def cluster_bootstrap(frame: pd.DataFrame, value: str, cluster: str,
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of `value` with an interval from resampling whole clusters.

    Resampling markets (or categories) rather than contexts respects the dependence the
    original analysis assumed away.
    """
    frame = frame.dropna(subset=[value])
    if frame.empty:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "n_clusters": 0}
    groups = [g[value].to_numpy() for _, g in frame.groupby(cluster, sort=True)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n_groups = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        means[b] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(frame[value].mean()), "lo": float(lo), "hi": float(hi),
            "n": int(len(frame)), "n_clusters": n_groups}


def paired_summary(frame: pd.DataFrame, value: str, cluster: str = "target_market",
                   seed: int = 0) -> dict:
    """Effect size, sign test and cluster interval for a set of paired differences."""
    x = frame[value].dropna().to_numpy()
    out = {"n": int(x.size), "mean": float(np.mean(x)) if x.size else np.nan,
           "median": float(np.median(x)) if x.size else np.nan,
           "share_positive": float((x > 0).mean()) if x.size else np.nan,
           "share_negative": float((x < 0).mean()) if x.size else np.nan}
    if x.size >= 6 and np.any(x != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(x).pvalue)
        # Matched-pairs rank-biserial correlation: an effect size, not just a verdict.
        ranks = stats.rankdata(np.abs(x[x != 0]))
        signs = np.sign(x[x != 0])
        out["rank_biserial"] = float(np.sum(ranks * signs) / np.sum(ranks))
    else:
        out["wilcoxon_p"] = np.nan
        out["rank_biserial"] = np.nan
    out.update({f"cluster_{k}": v
                for k, v in cluster_bootstrap(frame, value, cluster, seed=seed).items()})
    return out


def dose_model(frame: pd.DataFrame, value: str = "delta",
               dose: str = "log_borrowed", extra=()) -> pd.DataFrame:
    """Mixed-effects dose-response with random intercepts for market and category.

    This is the quantitative answer to "what happens when more data is borrowed": a slope
    on borrowed volume, adjusted for coupling bandwidth, with the market and category
    dependence absorbed into random effects rather than ignored.
    """
    import statsmodels.formula.api as smf

    df = frame.dropna(subset=[value, dose]).copy()
    if df.empty or df[dose].nunique() < 3:
        return pd.DataFrame()
    df["grp"] = df["target_market"].astype(str)
    terms = [dose] + [t for t in extra if t in df.columns]
    formula = f"{value} ~ " + " + ".join(terms)
    try:
        model = smf.mixedlm(formula, df, groups=df["grp"],
                            re_formula="1", vc_formula={"cat": "0 + C(target_category)"})
        fit = model.fit(method="lbfgs", maxiter=200)
    except Exception:
        model = smf.mixedlm(formula, df, groups=df["grp"])
        fit = model.fit(method="lbfgs", maxiter=200)
    out = pd.DataFrame({
        "term": fit.params.index, "coef": fit.params.values,
        "se": fit.bse.reindex(fit.params.index).values,
        "p": fit.pvalues.reindex(fit.params.index).values,
    })
    out["lo"] = out["coef"] - 1.96 * out["se"]
    out["hi"] = out["coef"] + 1.96 * out["se"]
    # Carried through rather than swallowed: a boundary fit (zero random-effect variance)
    # still returns usable fixed effects, but the reader is entitled to know.
    out["converged"] = bool(getattr(fit, "converged", True))
    out["n_obs"] = int(len(df))
    return out


def spearman_with_ci(x, y, n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman correlation with a bootstrap interval.

    A bare correlation carries no uncertainty; an interval is the minimum needed to judge one.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return {"rho": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": int(x.size)}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        if np.all(x[idx] == x[idx][0]) or np.all(y[idx] == y[idx][0]):
            boots[b] = np.nan
            continue
        boots[b] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return {"rho": float(rho), "lo": float(lo), "hi": float(hi), "p": float(p),
            "n": int(x.size)}
