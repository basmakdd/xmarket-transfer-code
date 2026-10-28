# When Does Cross-Market Transfer Help? Evidence from E-Commerce Markets

Code, experiment results and figure/table builders for the paper
*When Does Cross-Market Transfer Help? Evidence from E-Commerce Markets* (Basma Alharbi).

The study asks when a recommender for one country store (for example Mexico, Electronics) benefits from the
interactions (ratings and reviews) of users in other countries that list the same products. It uses the XMarket
dataset (7 countries, 74 market-category contexts) and compares what travels between countries
(product popularity vs co-interaction patterns), the value of foreign data with complete local data, with scarce
local data and for newly launched products, and whether country closeness predicts transfer.

Every number in the paper's tables and figures is computed by the code in this repository from the experiment records.
The manuscript tables are formatted from the generated LaTeX tables (`paper/tables_*.tex`), the generated
Supplementary tables (`paper/supp/*.tex`) and the machine-readable results (`paper/results/*.csv`); the figures are
the generated `paper/figures/figureN.pdf`.

| What you want | Command | Needs the raw data? | Time (2 CPUs) |
|---|---|---|---|
| Rebuild all tables and figures from the shipped results | `python run.py tables` and `python run.py figures` | No (four outputs need it, see Section 2) | about 7 min + 10 s to 1 min |
| Check the repository | `python run.py check` | No | about 1 min |
| Check and rebuild everything in Google Colab, without installing anything | `reproduce_in_colab.ipynb` (Section 2.1) | No (optional Part B: yes) | about 10 min (+ 15 min) |
| Re-run a few contexts and compare with the shipped records | Section 3.3 | Yes | minutes |
| Re-run everything from the raw XMarket files | `python run.py data`, then `python run.py experiments` | Yes | about 1-2 days |

---

## 1. Install

You need **Python 3.10–3.13** (tested with 3.13); the pinned numpy 2.2.6 has no wheels for Python 3.14+, so create the
virtual environment with a 3.10–3.13 interpreter even if a newer Python is installed. The requirements are pinned
(`requirements.txt`); the code uses only cross-platform packages.

The repository folder is `xmarket-transfer` (a GitHub "Download ZIP" unpacks to `xmarket-transfer-main`; use that
name in the `cd` commands below).

**Windows (PowerShell)**

```powershell
cd xmarket-transfer
py -3.13 -m venv .venv               # any installed 3.10-3.13 works, e.g. py -3.12; `py -0` lists the installed versions
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd xmarket-transfer
python3.13 -m venv .venv             # or another 3.10-3.13 interpreter; with pyenv: pyenv install 3.13 && pyenv local 3.13
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional: PyTorch, CPU build (the GPU build is not needed and much larger):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

It is only needed to re-run the Market-Aware GMF baseline (`python run.py experiments --only mkt`); tables and figures
do not need it.

**Windows notes.** Everything runs on Windows (paths are built with `os.path`, all files are read and written as UTF-8).
Two differences: (1) the per-unit time limits of `launch3.py` (30 min) and `scarcity2.py` (20 min) use `SIGALRM`,
which Windows lacks, so there the limit is skipped and every unit runs to completion (a context that hit the limit in
the paper run may then be included); (2) set environment variables with `$env:NAME = "value"` in PowerShell
(`set NAME=value` in cmd) instead of `NAME=value command`.

All commands below can be started from any folder (for example `python C:\path\to\xmarket-transfer\run.py tables`).

## 2. Quick path: rebuild tables and figures (no raw data needed)

```bash
python run.py tables      # about 7 minutes       -> paper/tables_*.tex, paper/supp/*.tex, paper/results/*.csv
python run.py figures     # 10 s to 1 minute      -> paper/figures/figure1..9.pdf/.png, paper/figures/data/figure1..9.csv
python run.py check       # unit tests + consistency checks (about 1 minute)
```

`tables` rebuilds every LaTeX table from the experiment records in `paper/raw/` and `results/`, then the
Supplementary tables (`paper/code/supp_tables.py`, last). Each table script declares the inputs it needs beyond the
shipped records; a script whose inputs are missing is skipped with a message naming them, the others still run, and
the shipped outputs of the skipped script are kept. Only four scripts need more: `s2_accounting.py` (raw zip and
`data/interactions.parquet`), `variant_diagnostics.py` (the preprocessed data variants paper, nv, xd, nvxd, nvc),
`sens_variant_report.py` (`results/variant_map_nv.parquet`) and `sens2_variant_report.py`
(`results/variant_map_nvc.parquet`, `paper/raw/aud_rawcache/`).
`check` runs the unit tests (tests that need the preprocessed data are skipped when it is absent), cross-checks numbers
in the tables against the result files, checks file sizes (no file over 50 MB, at most 100 MB in total, with a warning
above 95 MB) and that no raw reviewer ID appears in the shipped text files (binary result files hold integer indices only;
the strings stored in the `.pkl`, `.parquet` and `.npz` files are scanned as well), and, in a git checkout, confirms that the rebuilt tables are
identical to the committed ones (all files are written with `\n` line endings on every platform; a file that differs
only by CRLF vs LF line endings counts as identical). `python run.py check --tol 1e-9` additionally accepts numbers in
the rebuilt `.csv`/`.tex` files that differ from the committed ones by at most that relative/absolute tolerance (the
default is an exact comparison). Outside a git checkout (e.g. the files-only release zip) it checks every file
except the `.gitignore`d ones (data, caches); the comparison with the committed versions is then reported as
"not checked (not a git checkout)".

### 2.1 Run in Google Colab

The notebook `reproduce_in_colab.ipynb` (in the repository root) runs the quick path, and optionally a small re-run
from the raw data, in Google Colab. It installs the pinned environment in a separate virtual environment, so Colab's
own packages do not interfere; no GPU is needed.

1. Open https://colab.research.google.com and choose **File → Upload notebook** (select `reproduce_in_colab.ipynb`),
   or open the notebook directly from GitHub (**File → Open notebook → GitHub**).
2. In the first code cell set `CODE_SOURCE` to the GitHub address of this repository, or to the Google Drive path of
   the files zip (e.g. `/content/drive/MyDrive/xmarket-transfer-v1.0-files.zip`), and, optionally, `DATA_ZIP` to the
   path of the raw XMarket zip (Section 3.1), e.g. `/content/drive/MyDrive/xmarket_categories.zip`. `DATA_ZIP` is
   empty by default, and then Part B is skipped.
3. **Runtime → Run all**. Google Drive is connected (allow the connection) only if `CODE_SOURCE` or `DATA_ZIP` is a
   path in Drive; Part A from a GitHub address needs no Drive access.

Success: the output shows `PART A PASSED` (unit tests, `run.py check`, all tables and figures rebuilt; about 10 min)
and, if `DATA_ZIP` is set, `PART B PASSED` (data rebuilt from the raw files, two stores re-run for the complete-data
and product-launch experiments with the arguments of the paper run, every record equal to the shipped one; about
15 min). Part A fingerprints the shipped tables (`paper/tables*.tex`, `paper/supp/*.tex`), result files
(`paper/results/*.csv|txt|md` and the `paper/raw/aud_{accounting,rq4,variants,S2}*.csv` files written by the table
builders) and figure data (`paper/figures/data/figure*.csv`) before rebuilding and checks that the rebuilt files are byte-identical, from a GitHub
clone and from the files zip alike. The figure PDFs/PNGs are redrawn from the figure data but not byte-compared.

## 3. Full path: re-run the experiments

### 3.1 Get the raw data

The XMarket ratings files are not redistributed here. Download them as described in
[`raw_data/README.md`](raw_data/README.md) and pack them into `raw_data/xmarket_categories.zip`
(or set the environment variable `XMARKET_ZIP` to the zip's path).

### 3.2 Run

```bash
python run.py data                    # all data variants (about 1 h); or e.g. --only paper calendar seed1 (three variants: about 5 min)
python run.py experiments             # every experiment group, with the arguments used for the paper
python run.py experiments --only rq1  # one group; `python run.py list` shows all groups
python run.py tables
python run.py figures
```

Every experiment script is resumable: if a run stops, start the same command again and finished units are skipped.
Because the shipped records are in `paper/raw/`, `experiments` on a fresh copy finds everything finished and returns
quickly. This includes the `audit` group: `identity_checks.py`, `split_facts.py` and `svd_repro.py` skip when their
outputs (`paper/raw/aud_identity.json`, `aud_split_facts.json`, `aud_svd_repro.csv`) exist, and `aud_raw.py` skips when
its raw-file cache `paper/raw/aud_rawcache/` (not shipped) is complete; on a fresh copy it builds that cache once
(raw zip needed, about 5 minutes). Each of the four scripts accepts `--fresh` to recompute anyway (for `aud_raw.py`,
`--fresh` ignores and rewrites `paper/raw/aud_rawcache/`). Use
`python run.py experiments --fresh` to move the shipped records to `paper/raw_precomputed/` first and
recompute them (the audit checks included). The file `paper/raw/mkt_skips.jsonl` (the hardware-dependent Market-Aware GMF exclusions of the paper
run, see Section 7) is copied back, so the re-run uses the paper's context set.

`experiments` first checks that every selected group has its inputs (data variants, raw zip, PyTorch) and stops,
without running anything, with a list of what is missing and the command that builds it.

| Group | What it runs | Data variants needed (`python run.py data --only ...`) | Wall time on a shared 2-CPU machine |
|---|---|---|---|
| `rq1` | `paper/code/rq1_agree.py` on the paper data and the ratings>=4 variant | paper, r4 | 43 + 27 min |
| `rq2` | `paper/code/rq2_run.py`: local / pool / weight / greedy for MostPop, ItemKNN, EASE, UserKNN, Random on the random, chronological and calendar splits, split seeds 1-2, ratings>=4, grid sensitivity | paper, calendar, seed1, seed2, r4 | 5.9 h |
| `launch` | `paper/code/launch3.py`: product launch, 5 models x 3 splits | paper, calendar | 6.7 h |
| `scarcity` | `paper/code/scarcity2.py`: scarce local data, 4 retention levels x 3 arms | paper, calendar | 4.0 h |
| `mkt` | `paper/code/mkt_run.py`: Market-Aware GMF and GMF controls | paper (+ PyTorch) | 48 min |
| `rq4` | `paper/code/single2.py`: single-donor ItemKNN gains for the 370 target--donor pairs | paper | 1.2 h |
| `audit` | raw-file accounting, identity checks, split facts, PureSVD reproducibility | paper (+ raw zip) | 15 min |
| `sens` | RQ1 and RQ2 on `_nv`, `_xd`, `_nvxd`; launch on `_nv` and `_xd`; scarcity and UserKNN on `_nv` only | paper, nv, xd, nvxd | 3 h |
| `sens2` | the same on the `_nvc` data variant | paper, nvc | 1.3 h |
| `nullab` | `paper/code/nullab_run.py`: RQ2 pooling with curveball-randomised foreign data (null model), paper and `_nvc` data | paper, nvc | 40 min |
| `core` | the core borrowing grid (`src/runner.py`, run only when named) | paper | many hours (PureSVD dominates) |

The run times come from the logs in `paper/logs/` (several experiments shared the machine; one process at a time per
group, niced). `python run.py list` prints the exact command line (script, arguments, environment variables) of every
run of every group; they are defined in `run.py` (function `experiments`).

### 3.3 Environment variables; re-running a few contexts into a scratch folder

| Variable | Read by | Effect |
|---|---|---|
| `ONLY` | `rq2_run.py`, `launch3.py`, `nullab_run.py`, `single2.py`, `mustadd.py`, `rq1_agree.py` | comma-separated list of contexts to run, e.g. `ONLY="jp Musical_Instruments,mx Office_Products"` (market code, a space, the category as in the file names); for `rq1_agree.py` the entries are target--donor pairs, `"jp Musical_Instruments de"` |
| `SKIP` | `rq2_run.py`, `nullab_run.py` | comma-separated contexts to leave out (the paper runs use `SKIP="uk Books"` where stated in `run.py`) |
| `PAPER_RAW` | every script in `paper/code/` (via `common.py`), `figures_paper.py`, `run.py` | folder for the experiment records instead of `paper/raw/`; a relative path is relative to the repository root. The sensitivity groups (`sens`, `sens2`) write to `<PAPER_RAW>/sens_<tag>/` (by default `paper/raw/sens_<tag>/`) |
| `XMARKET_ZIP` | `src/data.py`, `run.py` | path of the raw XMarket zip (default `raw_data/xmarket_categories.zip`) |
| `XM_DATA_TAG` | most experiment scripts | selects a preprocessed data variant (`_seed1`, `_r4`, `_nvc`, ...); set by `run.py` |
| `MKT_MAX_EPOCH_SEC` | `mkt_run.py` | first-epoch limit of the MA-GMF compute rule (default 15 s; see 7.) |

Experiment scripts skip every unit already present in their output file, so to recompute a few contexts without
touching the shipped records, point `PAPER_RAW` at an empty scratch folder (`scratch/` is `.gitignore`d) and call the
experiment script directly with **the arguments of the paper run**, as printed by `python run.py list`, adding `ONLY`
to restrict the contexts. Then compare the new records with the shipped ones using `compare_records.py`
(repository root):

```bash
# macOS / Linux  (paper run: SKIP="uk Books" python paper/code/rq2_run.py 30000 ItemKNN random full)
PAPER_RAW=scratch/raw ONLY="jp Musical_Instruments,mx Office_Products" python paper/code/rq2_run.py 30000 ItemKNN random full
PAPER_RAW=scratch/raw ONLY="jp Musical_Instruments" python paper/code/launch3.py 1e9 ItemKNN random 1e9
python compare_records.py scratch/raw paper/raw
```

```powershell
# Windows PowerShell
$env:PAPER_RAW = "scratch/raw"; $env:ONLY = "jp Musical_Instruments,mx Office_Products"
python paper/code/rq2_run.py 30000 ItemKNN random full
Remove-Item Env:ONLY; Remove-Item Env:PAPER_RAW
python compare_records.py scratch/raw paper/raw
```

`compare_records.py` matches every record under `scratch/raw/` with the shipped record of the same file, context,
model, split, arm and seed, ignores timing fields (`sec`, ...), prints one line per file and finally
`RECORDS EQUAL: n/n` when all `n` re-run records equal shipped ones (exit code 0; otherwise the line is marked
`<-- DIFFERENCES` and the exit code is 1). Numbers are compared exactly; `--tol 1e-9` allows a relative/absolute
tolerance (another platform or linear-algebra library can change the last digits of floating-point results).

`ONLY` does not apply to two scripts:

- `scarcity2.py` has no context selection: it runs a fixed list of units, small contexts first and then the larger ones
  cheapest first, until its time limit (first argument, seconds) is reached. A short time limit therefore re-runs the
  cheapest contexts: `PAPER_RAW=scratch/raw python paper/code/scarcity2.py 900 A` (phase A only, 15 minutes).
- `mkt_run.py` takes a time limit and an optional comma-separated list of **categories** (all markets of a category are
  trained together): `PAPER_RAW=scratch/raw OMP_NUM_THREADS=1 python paper/code/mkt_run.py 12000 Musical_Instruments`.
  Copy `paper/raw/mkt_skips.jsonl` into the scratch folder first if the comparison should use the paper's exclusions.

Do not run `python run.py tables` with `PAPER_RAW` pointing at a partial scratch folder: the table
scripts then read the scratch records and overwrite `paper/tables_*.tex` and `paper/results/` (restore them with
`git checkout -- paper`). Note that `run.py experiments` sets `ONLY`/`SKIP` itself for some runs (e.g. the grid
sensitivity runs), overriding a value set by the user; use the scripts directly to restrict contexts.

## 4. Where every table and figure comes from

All scripts are in `paper/code/`. Table scripts write `paper/tables_*.tex` and machine-readable results in
`paper/results/`; `supp_tables.py` writes the Supplementary tables to `paper/supp/`; `figures_paper.py` writes the
figures to `paper/figures/` and the data each figure draws to `paper/figures/data/figureN.csv`. The manuscript tables
are formatted from these files (same numbers, journal layout).

**Main text**

| Item | Label | Script | Output (formatted into the manuscript) | Experiment records |
|---|---|---|---|---|
| Table 1 (cross-market agreement) | `tab:travel` | `rq1_tables.py` | `paper/tables_rq1.tex` (`tab:rq1_main`, `tab:rq1_paired`), `paper/results/rq1_T1.csv`, `paper/results/rq1_T2.csv` | `paper/raw/rq1_agree.jsonl` |
| Table 2 (complete local data) | `tab:full` | `rq2_tables.py` | `paper/tables_rq2.tex` (`tab:full`), `paper/results/rq2_table2_all.csv` | `paper/raw/rq2_*.jsonl` |
| Table 3 (gain by user history) | `tab:history` | `rq2_tables.py` | `paper/tables_rq2.tex` (`tab:history`), `paper/results/rq2_history_cells.csv`, `paper/results/rq2_history_test.csv` | `paper/raw/rq2_peruser/` |
| Table 4 (scarce local data) | `tab:scarcity` | `scarcity2_tables.py` | `paper/tables_scarcity.tex` (`tab:scarcity2`), `paper/results/sc_results.csv` | `paper/raw/sc_*.jsonl` |
| Table 5 (product launch) | `tab:launch` | `launch3_tables.py` | `paper/tables_launch.tex` (`tab:launch`), `paper/results/l3_main_common.csv` | `paper/raw/l3_*.jsonl` |
| Table 6 (country closeness, ratio-free $D=F-L$, three data versions) | `tab:culture` (generated label `tab:culture_ratiofree`) | `rq4_ratiofree.py` | `paper/tables_rq4_ratiofree.tex`, `paper/results/rq4_ratiofree.csv` (also the ratio rows, which reproduce `rq4_tables.py`) | `paper/raw/aud_single.jsonl`, `paper/raw/rq1_agree.jsonl`, `paper/raw/sens_nv*/rq1_agree_*.jsonl` |
| Table 6 ratio version (sensitivity check, earlier main table) | `tab:culture` in `tables_rq4.tex` (repository only) | `rq4_tables.py --rq1 new` | `paper/tables_rq4.tex`, `paper/raw/aud_rq4_results.csv`, `paper/raw/aud_rq4_gain_summary.csv` | as above |
| Table A1 (hyperparameter grids) | `tab:grids` | `src/models.py` (`GRIDS`), `paper/code/mustadd.py` (`GRIDS`, UserKNN); other grids, not in Table A1: foreign-data weights `paper/code/rq2_run.py` (`ALPHAS`), MA-GMF `paper/code/mkt_run.py` (`GRID`) | - | - |
| Figure 1 (illustrative example) | `fig:intro` | `figures_paper.py` | `paper/figures/figure1.pdf`, `paper/figures/data/figure1.csv` | none (illustration, no study data) |
| Figure 2 (data per market and category) | `fig:data` | `figures_paper.py` | `paper/figures/figure2.pdf` | `paper/figures/data/data_distribution.csv` (`figs/data_stats.py`, `figs/data_table.py`), `paper/results/sens_diag_orig.csv` |
| Figure 3 (measuring what travels) | `fig:rq1method` | `figures_paper.py` | `paper/figures/figure3.pdf` | none (diagram) |
| Figure 4 (RQ1 results) | `fig:travel` | `figures_paper.py` | `paper/figures/figure4.pdf` | `paper/results/rq1_T1.csv`, `paper/results/sens2_rq1_measures.csv`, `paper/results/rq1_fig4_data.csv` |
| Figure 5 (complete local data) | `fig:full` | `figures_paper.py` | `paper/figures/figure5.pdf` | `paper/results/rq2_fig5_summary.csv`, `paper/results/sens2_rq2.csv` |
| Figure 6 (gain by history length) | `fig:history` | `figures_paper.py` | `paper/figures/figure6.pdf` | `paper/results/rq2_fig6_history.csv`, `paper/results/rq2_history_d.csv`, `paper/results/rq2_history_test.csv` |
| Figure 7 (scarce local knowledge) | `fig:scarcity` | `figures_paper.py` | `paper/figures/figure7.pdf` | `paper/results/sc_figure7_data.csv`, `paper/results/sc_buckets.csv` |
| Figure 8 (product launch per context) | `fig:launch` | `figures_paper.py` | `paper/figures/figure8.pdf` | `paper/results/l3_fig8.csv` |
| Figure 9 (country closeness) | `fig:culture` | `figures_paper.py` | `paper/figures/figure9.pdf` | `paper/results/rq4_ratiofree.csv` (`rq4_ratiofree.py`), `paper/raw/aud_rq4_pairs.csv` |
| Numbers quoted in the text with "(code repository)" | - | `supp_tables.py`, `rq1_tables.py`, `scarcity2_tables.py`, ... | e.g. `paper/results/rq1_per_context.csv` (paired difference positive in 69 of 70 target contexts), `paper/results/sc_matrix_itemknn_random_f010.csv` (gap closed per context), `paper/results/rq1_stats.txt` | as above |

**Supplementary Material** (`Supplementary.tex` `\input`s the files of `paper/supp/`, written by `supp_tables.py`, `variability.py` and `supp_extra.py`)

| Item | Output | Inputs (written by) |
|---|---|---|
| Table S1 (product launch per context) | `paper/supp/S1.tex` | `paper/results/l3_context.csv` (`launch3_tables.py`) |
| Table S2 (dataset accounting) | `paper/supp/S2.tex` | `paper/raw/aud_S2.csv` (`s2_accounting.py`; full per-file table: `paper/tables_S2.tex`) |
| Table S3 (experiment accounting) | `paper/supp/S3.tex` | `paper/raw/aud_accounting_summary.csv` (`accounting.py`; per-context accounting: `paper/tables_S3.tex`, `paper/raw/aud_accounting.csv`), `paper/results/sc_results.csv` |
| Table S4 (country-distance inputs) | `paper/supp/S4.tex` | `paper/results/rq4_country_features.csv`, `paper/results/rq4_country_distances.csv`, `paper/raw/aud_rq4_pairs.csv` (`rq4_tables.py`; also `paper/tables_S4.tex`) |
| Table S5a (main results on the five data versions) | `paper/supp/S5a.tex` | `paper/results/sens_*.csv` (`sens_tables.py`), `paper/results/sens2_*.csv` (`sens2_tables.py`, `sens2_extra.py`, `sens2_rq4.py`), `paper/results/sens_diag_*.csv` (`variant_diagnostics.py`), `results/contexts*.parquet` |
| Table S5b (null-pool ablation, Market-Aware GMF) | `paper/supp/S5b.tex` | `paper/results/nullab_summary.csv` (`nullab_tables.py`; also `paper/tables_nullab.tex`), `paper/results/mkt_summary.csv` (`mkt_tables.py`; also `paper/tables_mkt.tex`) |
| Table S5d (variability across scarcity subsamples, launch sets and split seeds; test-user cap) | `paper/supp/S5d.tex` | `paper/raw/sc_*.jsonl`, `paper/raw/l3_*.jsonl`, `paper/results/rq2_seed_variability.csv` (`variability.py`; also `paper/results/variability.csv`; checks its medians against Tables 4 and 5) |
| Table S6 (per-context experiment accounting) | `paper/supp/S6.tex` | `paper/raw/aud_accounting.csv` (`accounting.py`), `paper/raw/sc_UserKNN_random.jsonl` (`supp_extra.py`) |
| Table S7 (top-10 overlap by shared products, RQ1 denominators, launch skips) | `paper/supp/S7.tex` | `paper/results/rq1_T3.csv`, `paper/results/rq1_denominators.csv` (`rq1_tables.py`), `paper/results/l3_skips_summary.csv` (`launch3_tables.py`) (`supp_extra.py`) |
| Table S5c (further robustness checks) | `paper/supp/S5c.tex` | `paper/results/loo_robustness.csv`, `paper/results/rq2_catalog_strata.csv` (`loo_robustness.py`; also `paper/tables_loo.tex`), `paper/results/rq2_seed_variability.csv`, `paper/results/rq2_table2_all.csv` (`rq2_tables.py`), `paper/results/sc_results.csv` (`scarcity2_tables.py`) |

**Further tables in the repository** (per-context results and checks referred to as "code repository"):
`paper/tables_rq1.tex` (ratios, popularity-adjusted measures, top-10 bins, ratings >= 4), `paper/tables_rq2.tex`
(common contexts, untuned contexts, seeds, ratings >= 4, per-context matrices), `paper/tables_scarcity.tex` (all
levels and arms, history buckets, fallback, per-context gap closed), `paper/tables_launch.tex` and
`paper/tables_launch_S1.tex` (all contexts, all splits, both tunings), `paper/tables_variants.tex` (unreported
variants of the core grid, including PureSVD; `variants.py`), `paper/tables_sens_diag.tex` (product-variant
diagnostics), `paper/results/sens_tables.md` and `paper/results/sens2_tables.md` (all sensitivity comparisons).
`tables_rq4_oldrq1.tex` and `tables_S4_oldrq1.tex` repeat the country-closeness analysis with the agreement outcomes
of the v3 study version (kept for comparison; `rq4_tables.py --rq1 travel3 --tag _oldrq1 --check-old`). `tables_rq4_minlocal005.tex`
and `paper/results/rq4_minlocal005_{results,pairs,gain_summary}.csv` are the threshold check of Section 4.5
(`rq4_tables.py --rq1 new --min-local 0.05`: agreement ratios only for pairs whose local agreement is at least 0.05).
Each `rq4_tables.py` run writes its log to `paper/logs/aud_rq4.log`, `aud_rq4_minlocal005.log` or `aud_rq4_oldrq1.log`.

## 5. Data variants

All variants are built by `src/data.py` (or `src/calendar_split.py`) and write only files carrying their tag,
so they never overwrite the paper data.

| Variant | Tag | How it differs from the paper data |
|---|---|---|
| paper | (none) | 74 market-category contexts after removing repeated user-product pairs and an adaptive k-core filter; per-user random and chronological 80/10/10 splits (seed 0) |
| calendar | `_calendar` | prospective split: one pair of calendar cutoffs per category (80th/90th percentile of dates), so foreign training data are never later than the local cutoff; the core filter is applied to pre-cutoff interactions and fixes the catalog, users are not core-filtered. Also lists the calendar launch candidates: products outside the pre-cutoff local catalog (not retained by the pre-cutoff core filter), with foreign pre-cutoff interactions and a local interaction after T_val (`src/calendar_split.py`) |
| seeds | `_seed1`, `_seed2` | the per-user random split drawn with seeds 1 and 2 |
| ratings >= 4 | `_r4` | only ratings of 4 or 5 stars count as interactions |
| variant merge | `_nv` | listings of the same product (variants that share >= 3 users and Jaccard >= 0.8) merged; users pooled over markets for the linking |
| cross-market reviewers | `_xd` | interactions of reviewer IDs that appear in more than one market are removed |
| both | `_nvxd` | `_nv` and `_xd` together |
| within-context variant merge | `_nvc` | product variants merged within each market-category context (the "cleaned" data in Figures 4, 5 and 9) |

## 6. Repository layout

```
run.py              one driver: data, experiments, tables, figures, check, list
reproduce_in_colab.ipynb  Google Colab notebook: check, rebuild, optional re-run from the raw data (Section 2.1)
compare_records.py  compares re-run experiment records with the shipped ones (Section 3.3)
src/                preprocessing (data.py, calendar_split.py), overlap statistics, models, metrics,
                    evaluation harness (experiment.py), core-grid runner, statistics
paper/code/         experiments, table builders, figures_paper.py (Figures 1-9), figs/style.py
paper/raw/          shipped experiment records (inputs of the table builders)
paper/results/      machine-readable outputs of the table builders (CSV), used by the figures
paper/tables_*.tex  generated LaTeX tables
paper/supp/         generated Supplementary tables S1-S7 (supp_tables.py, variability.py, supp_extra.py)
paper/figures/      generated figures and the data each one draws (data/figureN.csv)
paper/logs/         run logs (run times)
results/            preprocessing outputs (contexts, overlap statistics, core grid)
tests/              unit tests
legacy_v3/          superseded code of the v3 study version (see legacy_v3/README.md; it uses the earlier
                    'purchase/shopper' wording, which the current paper replaced by 'interaction/user')
raw_data/           where the XMarket files go (not included)
```

## 7. Reproducibility notes

- All randomness is seeded (splits, subsamples, launch sets, half-splits, bootstraps), so re-runs give identical numbers.
- Exception: **PureSVD** (core grid and the variants table only) uses an eigendecomposition whose result can depend on the
  linear-algebra library when singular values are tied; `paper/code/svd_repro.py` documents the check.
- The Market-Aware GMF baseline is trained with PyTorch on CPU (one thread); other models are closed form.
- Run times above were measured on a shared machine with 2 CPUs. Several experiments stop at a time or cost budget
  (stated in the table notes); in particular the scarcity run (`scarcity2.py`) completed contexts up to a cost of
  5.9e7 users x products (UK Electronics) within its time limits (CA Electronics, 7.05e7, was started for EASE but
  not completed), so on a faster machine the same command may cover more large contexts.
- **Product launch inclusion.** A launch set with fewer than 10 launched-item test users (users whose test interactions
  include a launched product) is skipped, and a context without any remaining launch set is excluded for that split
  (`launch3.py`, written to `paper/raw/l3_skip_*.jsonl`). Inclusion therefore depends on the test-set counts; the rule
  is the same for every model and arm.
- **Market-Aware GMF exclusions.** `mkt_run.py` skips a category whose first pooled (or MA) epoch takes longer than
  15 s; in the paper run this excluded Electronics, Movies & TV and Books (52 contexts remain). The rule is
  hardware-dependent: on another machine the same threshold can admit or exclude other categories. Keep the shipped
  `paper/raw/mkt_skips.jsonl` (the run honours it) or set `MKT_MAX_EPOCH_SEC` to reproduce the paper's context set.
- **Per-unit time limits** (`launch3.py` 30 min, `scarcity2.py` 20 min) use `SIGALRM` and apply on Linux/macOS only;
  on Windows they are skipped (see Section 1).
- **Foreign-data weighting** (`weight` arm, `paper/code/mustadd.py: weight_rows`): every foreign user's row of the
  binary user x product matrix is multiplied by alpha (local rows unchanged). Hence, per model: MostPop counts foreign
  interactions with weight alpha; EASE's Gram matrix receives foreign contributions times alpha^2; ItemKNN's
  co-occurrence counts receive foreign contributions times alpha^2, with item norms computed from the scaled matrix
  (sqrt(local count + alpha x foreign count)); UserKNN (as implemented in `mustadd.UserKNN`) gives a foreign
  neighbour the similarity alpha c / (sqrt(n_u) sqrt(alpha n_v) + shrink) and a vote from its scaled row, i.e.
  alpha^1.5 times the unweighted contribution when shrink = 0. alpha = 0 is the local-only arm, alpha = 1 equal pooling.
- The core grid in `results/full_random.jsonl` is the first 24,754 records of a stopped run; a complete re-run adds
  records for the largest contexts.
- Candidate-set metrics (`ndcg@10|c100`, `|c1000`) in `results/full_random.jsonl` and `paper/raw/mkt_runs.jsonl` are not
  used in the paper; they are now off by default (`src/experiment.py`, `CANDIDATE_SIZES`).

## 8. Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff); the article is under review at the
*Journal of Theoretical and Applied Electronic Commerce Research*, so year, volume and DOI are added on publication)
and the XMarket dataset:
H. Bonab, M. Aliannejadi, A. Vardasbi, E. Kanoulas, J. Allan. *Cross-Market Product Recommendation.* CIKM 2021.
Hofstede country scores are from the official 2015 file (<https://geerthofstede.com/research-and-vsm/dimension-data-matrix/>)
and are embedded in the code.

## 9. Licence

MIT, see [`LICENSE`](LICENSE). The XMarket data are not part of this repository and keep their own terms.
