"""Round-3 additions: ratio-free RQ4 (paper/code/rq4_ratiofree.py), variability across repetitions
(paper/code/variability.py) and the per-context accounting / RQ1 diagnostics (paper/code/supp_extra.py).

The tests read the shipped outputs; they are skipped if `python run.py tables` has not produced them."""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "paper" / "code"
RES = ROOT / "paper" / "results"
RAW = ROOT / "paper" / "raw"
SUPP = ROOT / "paper" / "supp"


def need(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing {p.relative_to(ROOT)} (run `python run.py tables`)")


def _block(src: str, name: str) -> str:
    m = re.search(rf"^{name}\s*=\s*(.+)$", src, re.M)
    assert m, name
    return m.group(1).replace(" ", "")


@pytest.mark.parametrize("name", ["hof", "lang", "cap"])
def test_ratiofree_country_features_equal_rq4_tables(name):
    a = (CODE / "rq4_tables.py").read_text(encoding="utf-8")
    b = (CODE / "rq4_ratiofree.py").read_text(encoding="utf-8")
    assert _block(a, name) == _block(b, name)


def test_ratiofree_ratio_rows_reproduce_table6_ratios():
    need(RES / "rq4_ratiofree.csv", RAW / "aud_rq4_results.csv")
    R = pd.read_csv(RES / "rq4_ratiofree.csv")
    A = pd.read_csv(RAW / "aud_rq4_results.csv")
    mp = {"pop_rel": "sp", "pop10_rel": "top10", "pat_rel": "cos", "nbr_rel": "nbr5"}
    for _, a in A[A.outcome.isin(mp)].iterrows():
        for d in ("Hofstede distance", "Shared language", "Geographic distance"):
            q = R[(R.source == "new") & (R.in_ac == "incl.") & (R.kind == "R") & (R.measure == mp[a.outcome]) & (R.distance == d)]
            assert len(q) == 1
            assert abs(q.r.iloc[0] - a[f"r|{d}"]) < 1e-9 and abs(q.p.iloc[0] - a[f"p|{d}"]) < 1e-12


def test_ratiofree_outcome_is_foreign_minus_local():
    """Recompute one D row by hand from the RQ1 records: centring, pair averaging and Pearson r."""
    need(RES / "rq4_ratiofree.csv", RAW / "rq1_agree.jsonl")
    X = pd.read_json(RAW / "rq1_agree.jsonl", lines=True).drop_duplicates(["t", "donor_market"])
    X["D"] = X.for_sp - X.loc_sp
    z = X.dropna(subset=["D"]).copy()
    z["res"] = z.D - z.groupby("t").D.transform("mean")
    pm = z.groupby(["target_market", "donor_market"]).res.mean().reset_index()
    cap = {'ca': (45.42, -75.70), 'de': (52.52, 13.40), 'fr': (48.86, 2.35), 'in': (28.61, 77.21), 'jp': (35.68, 139.69),
           'mx': (19.43, -99.13), 'uk': (51.51, -0.13)}
    def geo(a, b):
        (la1, lo1), (la2, lo2) = [(math.radians(p[0]), math.radians(p[1])) for p in (cap[a], cap[b])]
        h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(h))
    r = np.corrcoef([geo(a, b) for a, b in zip(pm.target_market, pm.donor_market)], pm.res)[0, 1]
    R = pd.read_csv(RES / "rq4_ratiofree.csv")
    q = R[(R.source == "new") & (R.in_ac == "incl.") & (R.kind == "D") & (R.measure == "sp") & (R.distance == "Geographic distance")]
    assert abs(q.r.iloc[0] - r) < 1e-12
    assert len(pm) == 42


def test_permutation_p_is_valid():
    need(RES / "rq4_ratiofree.csv")
    R = pd.read_csv(RES / "rq4_ratiofree.csv")
    assert R.p.between(1 / 5040 - 1e-12, 1).all()
    assert ((R.p * 5040) - (R.p * 5040).round()).abs().max() < 1e-6   # p is a count over the 5,040 relabellings
    assert (R.bonferroni == (R.p < 0.05 / 15)).all()


def test_ratiofree_table_matches_csv():
    need(RES / "rq4_ratiofree.csv", ROOT / "paper" / "tables_rq4_ratiofree.tex")
    R = pd.read_csv(RES / "rq4_ratiofree.csv")
    tex = (ROOT / "paper" / "tables_rq4_ratiofree.tex").read_text(encoding="utf-8")
    for _, r in R[(R.in_ac == "incl.") & R.kind.isin(["D", "gain"])].iterrows():
        assert f"{r.r:+.2f}".replace("-", "$-$") + f" ({r.p:.4f})" in tex


def test_variability_reproduces_headline_medians():
    need(RES / "variability.csv", RES / "sc_results.csv", RES / "l3_main_common.csv")
    V = pd.read_csv(RES / "variability.csv")
    SC = pd.read_csv(RES / "sc_results.csv")
    arm = {"patterns scarce": "P", "both scarce": "B", "histories scarce": "R"}
    sv = V[V.experiment.str.startswith("scarcity")]
    assert len(sv) >= 12
    for _, v in sv.iterrows():
        ref = SC[(SC.model == v.model) & (SC.split == v.split) & (SC.f == 0.1) & (SC.arm == arm[v.arm])]
        assert abs(v.median_of_means - ref.gain_med.iloc[0]) < 1e-9
    MC = pd.read_csv(RES / "l3_main_common.csv")
    for _, v in V[V.experiment.str.startswith("launch")].iterrows():
        ref = MC[(MC.split == v.split) & (MC.model == v.model) & (MC.tuning == "zf")]
        assert abs(v.median_of_means - ref.dchance_med.iloc[0]) < 1e-9
    assert (V.within_sd_median.dropna() >= 0).all()


def test_s6_covers_every_context_once():
    need(SUPP / "S6.tex", RAW / "aud_accounting.csv")
    tex = (SUPP / "S6.tex").read_text(encoding="utf-8")
    body = tex.split("\\endfoot", 1)[1]
    rows = [l for l in body.splitlines() if l.endswith("\\\\")]
    A = pd.read_csv(RAW / "aud_accounting.csv")
    assert len(rows) == A.context.nunique() == 74
    cells = {c.strip() for l in rows for c in l[:-2].split("&")[1:]}
    allowed = {"\\checkmark", "--", "D", "S", "C", "U", "P", "R", "B", "I", "M", "O", "E"}
    assert all(c in allowed or c.isdigit() for c in cells), cells - allowed
    assert "?" not in cells


def test_s7_top10_bins_partition_pairs():
    need(RES / "rq1_T3.csv")
    T = pd.read_csv(RES / "rq1_T3.csv")
    T = T[(T.variant == "main") & (T.arm == "all")].set_index("bin")
    bins = ["10--19", "20--49", "50--199", "$\\ge$200"]
    assert T.loc[bins, "pairs"].sum() == T.loc["all pairs", "pairs"] == 324


def test_rq1_sparsity_bins_partition_and_match_table1():
    need(RES / "rq1_sparsity.csv", RES / "rq1_T2.csv")
    R = pd.read_csv(RES / "rq1_sparsity.csv")
    allp = R[R.kind == "all pairs"].iloc[0]
    for kind in ("shared", "density"):
        assert R[R.kind == kind].cos_vs_rank_pairs.sum() == allp.cos_vs_rank_pairs == 324
        assert R[R.kind == kind].nn_vs_top10_pairs.sum() == allp.nn_vs_top10_pairs
    assert R.cos_vs_rank_share.between(0, 1).all()
    assert (R.cos_vs_rank_lo <= R.cos_vs_rank_median).all() and (R.cos_vs_rank_median <= R.cos_vs_rank_hi).all()
