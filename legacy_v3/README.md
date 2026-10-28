# legacy_v3: code of the earlier manuscript version (superseded)

This folder keeps the version of the code used for the earlier manuscript version (v3): the scripts that produced
its tables and figures. They are superseded by the scripts in `paper/code/` and `run.py` and are **not** used for the
current version of the paper. They are kept so that the numbers of the earlier version remain reproducible.

**Terminology.** These scripts, their comments and their figure labels use the earlier wording *purchases*, *shoppers*,
*sales* and *co-purchase*. XMarket records ratings and reviews, not verified purchases; the current paper and the code
outside this folder therefore speak of *interactions*, *users* and *co-interaction* patterns. The data are the same.

| File | Was used for | Replaced by |
|---|---|---|
| `src/analyze.py`, `src/verify.py` | analysis and independent verification of the core grid | `paper/code/*_tables.py`, `paper/code/boot.py` |
| `paper_code/a.py` | results frame `paper/raw/df.pkl` from the core grid | still used by `run.py experiments --only core` |
| `paper_code/travel3.py` | popularity vs pattern agreement | `paper/code/rq1_agree.py` |
| `paper_code/run_full.py` | complete-data runs (UserKNN, temporal) | `paper/code/rq2_run.py` |
| `paper_code/launch2.py` | product-launch experiment | `paper/code/launch3.py` |
| `paper_code/g.py`, `run_single.py`, `h.py`, `i.py` | single-donor pairs and country features | `paper/code/single2.py`, `rq4_tables.py` |
| `paper_code/hist.py` | gain by user history | `paper/code/rq2_tables.py` (per-user records of `rq2_run.py`) |
| `paper_code/final_tables.py`, `detailed_tables.py` | `paper/tables.tex`, `paper/tables_detailed.tex` | `paper/code/*_tables.py` |
| `paper_code/make_fig_intro_example.py`, `figs/*` | figures of the earlier version | `paper/code/figures_paper.py` |

The scarcity runs of the earlier version (`ma_*.jsonl`) came from the main block of `paper/code/mustadd.py`, which
stays in `paper/code/` because the current experiments import its model and fitting functions.

## Rebuilding the tables of the earlier version

The scripts read the shipped records in `paper/raw/` and `results/` and write into `legacy_v3/paper/`:

```bash
python legacy_v3/paper_code/final_tables.py      # -> legacy_v3/paper/tables.tex
python legacy_v3/paper_code/detailed_tables.py   # -> legacy_v3/paper/tables_detailed.tex
python legacy_v3/paper_code/figs/verify.py       # 210 cross-checks between those tables and the records
python -m pytest -q legacy_v3/tests              # tests of analyze.py / verify.py
```

`final_tables.py` and `figs/verify.py` also read `paper/raw/hist.pkl` (per-user gains by history length), which is
not shipped to keep the repository small. Rebuild it first with `python legacy_v3/paper_code/hist.py` (needs the
preprocessed data, `python run.py data --only paper`). With it, both tables are rebuilt byte-identical to the shipped `legacy_v3/paper/tables*.tex`. The old figure scripts in
`paper_code/figs/` write to `legacy_v3/paper/figures/` (only the PDFs are shipped).
