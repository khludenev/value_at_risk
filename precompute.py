"""Compute independent rolling model families concurrently, then open the notebook.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python precompute.py
This performs the full 2018–2025 study; it does not change the notebook settings.
"""
from concurrent.futures import ProcessPoolExecutor
from var_utils import load_tankermap, raw_returns, require_coverage, rolling_forecasts


def compute(model):
    prices, _ = load_tankermap()
    prices = prices.loc['2006-05-31':'2025-12-31']
    require_coverage(prices)
    result = rolling_forecasts(raw_returns(prices), model, gh_workers=4 if model == 'GH' else 1)
    return model, result.status.value_counts().to_dict()


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(compute, ['GH', 'GJR-GARCH', 'AS-CAViaR']):
            print(result, flush=True)
