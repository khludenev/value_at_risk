# WTI 95% Value at Risk

Open **[value-at-risk-modelling.ipynb](value-at-risk-modelling.ipynb)** with the
project's `.venv` interpreter. It contains the requested stages 0–16; reusable
estimators, tests, losses, and plotting functions are in `var_utils.py`.

## Data and agreed dates

The project uses the **daily WTI close from TankerMap**, in USD/barrel:
<https://tankermap.com/market-data/wti>.
Its public MAX endpoint starts on **2006-05-31**, so the agreed revised splits are:

| Split | Price/return dates |
|---|---|
| Training | 2006-05-31–2008-12-31 (first return on the next observed date) |
| Validation | 2009-01-01–2017-12-31 |
| Test | 2018-01-01–2025-12-31 |

The raw JSON snapshot and SHA-256 provenance record are included in `data/`.
Only prices through 2025 enter the analysis. Data are the snapshot downloaded on
2026-10-25, not an archive of what was known at each historical publication date.
TankerMap identifies KuzTerm as its upstream source. Contract-roll construction
is not documented on the price page; these are published price-series returns,
not a reconstructed futures investment strategy.

The source includes WTI's **negative price on 2020-04-20**. The exact requested
raw-return formula retains that observation and produces returns below −100%.
No interpolation, forward filling, winsorizing, or replacement data are used.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab value-at-risk-modelling.ipynb
```

Select `.venv/bin/python` as the notebook interpreter. `requirements-lock.txt`
records the verification environment (Python 3.14/macOS ARM); `requirements.txt`
provides broader compatible dependency ranges.

Run all notebook cells, or precompute the three rolling model families concurrently:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python precompute.py
.venv/bin/python execute_notebook.py
```

The full study makes **6,039 daily model refits**. Runtime depends on hardware and
optimizer difficulty. The notebook uses four local GH workers; set `GH_WORKERS=1`
if local multiprocessing is unavailable. The same independent GH starts are
used regardless of worker count.

Each model writes a checkpoint every 25 dates. Rerunning resumes automatically.
Fingerprints include returns, dates, settings, library versions, and utility source
code; changes trigger a new run. Failed dates remain missing with diagnostics,
and block model comparison until resolved. Inspect `*_forecasts.csv` and the
checkpoint JSON. To retry failures after resolving the numerical issue, remove
the corresponding checkpoint or change the fitting implementation. Never fill
missing forecasts using a different model or silently remove test dates.

Tables, forecasts, metadata and selected PNG plots are saved in `results/`.
The notebook already contains the formulas and methodological qualifications.

## Statistical conventions

- 95% VaR is the **lower 5% return quantile**, plotted below the return series.
  Positive loss VaR is its negative. Exceptions are `return < quantile`.
- Rolling windows contain **504 previous observations**, excluding the forecast
  date. The original three-year training label is distinct from a two-year window.
- The requested “historical GH” method is called **rolling parametric GH VaR**;
  an empirical historical quantile is included as a reference.
- Density and GARCH comparisons use the pooled development period through 2017.
  Model families are fixed by the study design; there is no claim of an independent
  validation-based model-selection exercise.
- GARCH models use a constant mean and standardized Student-t innovations.
- CAViaR SAV, AS and INDGARCH use quantile loss and explicit numerical constraints.
  Their AIC/BIC are **asymmetric-Laplace pseudo-criteria**, with the estimated scale
  counted as a parameter. Compare them within CAViaR, not with full-density models.
- In-sample DQ includes a parameter-estimation covariance adjustment; test-period
  DQ uses predetermined instruments. Both rely on asymptotic assumptions.
- Pairwise DM tests use quantile loss, a HAC variance, a one-step finite-sample
  correction, and Holm adjustment for the three comparisons.

## Verify / edit

```bash
.venv/bin/python -m pytest -q
.venv/bin/python execute_notebook.py --development
```

`build_notebook.py` generates the notebook from editable cell strings. Running it
**replaces notebook cells and clears outputs**; edit the notebook directly for
normal experimentation, or keep changes in the generator if regenerating it.
The tests cover loss signs, boundary exception counts, clustered hits, DQ algebra,
DM direction, date alignment, negative prices, parsing, coverage, CAViaR dynamics,
rolling-window leakage, resumability, and explicit fit failures.
