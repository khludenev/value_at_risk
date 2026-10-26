"""Reusable estimators and diagnostics for the WTI 95% VaR notebook.

All public return/quantile inputs use decimal returns. VaR lines are signed lower
tail quantiles: an exception is r_t < q_t. No missing forecasts are imputed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import requests
from scipy import optimize, special, stats
from arch import arch_model
from arch.univariate import SkewStudent
from numba import njit
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

ALPHA = 0.05
SEED = 2026


def paired(y, q):
    if isinstance(y, pd.Series) and isinstance(q, pd.Series):
        if not y.index.equals(q.index):
            raise ValueError("Returns and quantiles must have identical date indices.")
    y, q = np.asarray(y, float), np.asarray(q, float)
    if y.ndim != 1 or y.shape != q.shape or len(y) == 0:
        raise ValueError("Expected equal, nonempty one-dimensional arrays.")
    if not (np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("Missing/nonfinite forecasts: inspect fit diagnostics first.")
    return y, q


def quantile_loss(y, q, alpha=ALPHA):
    y, q = paired(y, q)
    return (alpha - (y < q)) * (y - q)


def lopez_loss(y, q):
    y, q = paired(y, q)
    return (y < q) * (1.0 + (y - q) ** 2)


def blanco_ihle_loss(y, q):
    y, q = paired(y, q)
    if np.any(q >= 0):
        raise ValueError("Blanco–Ihle requires strictly negative return quantiles.")
    return np.where(y < q, (y - q) / q, 0.0)


def _binom_ll(k, n, p):
    return float(special.xlogy(k, p) + special.xlogy(n - k, 1 - p))


def kupiec_test(y, q, alpha=ALPHA):
    y, q = paired(y, q)
    n, k = len(y), int(np.sum(y < q))
    lr = max(0.0, 2 * (_binom_ll(k, n, k / n) - _binom_ll(k, n, alpha)))
    return dict(n=n, exceptions=k, rate=k / n, LR_uc=lr,
                p_uc=float(stats.chi2.sf(lr, 1)),
                p_binomial=float(stats.binomtest(k, n, alpha).pvalue))


def christoffersen_test(y, q, alpha=ALPHA):
    y, q = paired(y, q)
    hit = (y < q).astype(int)
    if len(hit) < 3:
        raise ValueError("At least three consecutive forecasts are required.")
    counts = np.bincount(2 * hit[:-1] + hit[1:], minlength=4)
    n00, n01, n10, n11 = counts
    n0, n1 = n00 + n01, n10 + n11
    result = dict(zip(('n00', 'n01', 'n10', 'n11'), map(int, counts)))
    # With no origin observations in a state the transition model is unidentified.
    if n0 == 0 or n1 == 0:
        return result | dict(LR_ind=np.nan, p_ind=np.nan, LR_cc=np.nan,
                             p_cc=np.nan, status="unidentified transition row")
    pooled = (n01 + n11) / (n0 + n1)
    ll0 = _binom_ll(n01 + n11, n0 + n1, pooled)
    ll1 = _binom_ll(n01, n0, n01 / n0) + _binom_ll(n11, n1, n11 / n1)
    lr = max(0.0, 2 * (ll1 - ll0))
    cc = lr + kupiec_test(y, q, alpha)['LR_uc']
    sparse = np.any(counts < 5)
    return result | dict(LR_ind=lr, p_ind=float(stats.chi2.sf(lr, 1)),
                         LR_cc=cc, p_cc=float(stats.chi2.sf(cc, 2)),
                         status="sparse transitions; asymptotic p-values" if sparse else "ok")


def dq_test(y, q, alpha=ALPHA, lags=4, gradient=None, bandwidth=None):
    """Engle–Manganelli DQ; supplied gradient enables in-sample adjustment.

    In-sample instruments: lagged centered hits. Out-of-sample: constant,
    lagged hits, and the current (predetermined) quantile. In-sample covariance
    uses the rectangular residual kernel in Theorems 3–4 of the original paper.
    """
    y, q = paired(y, q)
    if len(y) <= lags + 10:
        raise ValueError("Too few observations for the DQ regression.")
    hits = (y < q).astype(float) - alpha
    x = np.column_stack([hits[lags - j:-j] for j in range(1, lags + 1)])
    h = hits[lags:]
    extra = {}
    if gradient is None:
        x = np.column_stack([np.ones(len(h)), x, q[lags:]])
        m = x.T
    else:
        g = np.asarray(gradient, float)
        if g.ndim != 2 or g.shape[0] != len(y) or not np.isfinite(g).all():
            raise ValueError("Gradient must be finite with one row per fitted observation.")
        g = g[lags:]
        residual = (y - q)[lags:]
        if bandwidth is None:
            # Bandwidth tends to zero, while n*bandwidth tends to infinity.
            robust_scale = min(np.std(residual), stats.iqr(residual) / 1.349)
            bandwidth = max(1e-8, 1.06 * robust_scale * len(h) ** (-1 / 5))
        weights = (np.abs(residual) < bandwidth) / (2 * bandwidth)
        d = g.T @ (weights[:, None] * g)
        if np.linalg.matrix_rank(d) < g.shape[1]:
            return dict(DQ=np.nan, p_DQ=np.nan, df=0,
                        status="singular in-sample density/gradient matrix")
        m = x.T - (x.T @ (weights[:, None] * g)) @ np.linalg.solve(d, g.T)
        extra = dict(bandwidth=bandwidth, kernel_observations=int(np.sum(weights > 0)))
    covariance = m @ m.T
    rank = np.linalg.matrix_rank(covariance)
    if rank < x.shape[1]:
        return dict(DQ=np.nan, p_DQ=np.nan, df=rank, status="singular DQ instruments")
    score = x.T @ h
    statistic = float(score @ np.linalg.solve(covariance, score) / (alpha * (1 - alpha)))
    return dict(DQ=statistic, p_DQ=float(stats.chi2.sf(statistic, rank)),
                df=rank, status="asymptotic", **extra)


def backtest(y, q, alpha=ALPHA):
    y, q = paired(y, q)
    result = kupiec_test(y, q, alpha) | christoffersen_test(y, q, alpha)
    result |= {f"DQ_{k}": v for k, v in dq_test(y, q, alpha).items()}
    result |= dict(Lopez_mean=float(lopez_loss(y, q).mean()),
                   Quantile_mean=float(quantile_loss(y, q, alpha).mean()))
    try:
        result['Blanco_Ihle_mean'] = float(blanco_ihle_loss(y, q).mean())
        result['Blanco_Ihle_status'] = 'ok'
    except ValueError as exc:
        result['Blanco_Ihle_mean'] = np.nan
        result['Blanco_Ihle_status'] = str(exc)
    return pd.Series(result)


def diebold_mariano(loss_a, loss_b, bandwidth=None):
    """Two-sided one-step DM with Bartlett/Newey–West long-run variance.

    Negative statistic favors A. Fixed one-step horizon; HLN finite-sample
    correction sqrt((n-1)/n), with a t(n-1) reference distribution.
    """
    a, b = paired(loss_a, loss_b)
    d = a - b
    n = len(d)
    if n < 10:
        raise ValueError("DM requires at least 10 observations.")
    lag = int(np.floor(4 * (n / 100) ** (2 / 9))) if bandwidth is None else int(bandwidth)
    if not 0 <= lag < n:
        raise ValueError("Invalid HAC bandwidth.")
    centered = d - d.mean()
    lrv = centered @ centered / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    if lrv <= np.finfo(float).tiny:
        return dict(DM=np.nan, p_value=np.nan, mean_difference=float(d.mean()),
                    hac_lags=lag, status="degenerate loss differential")
    dm = float(d.mean() / np.sqrt(lrv / n) * np.sqrt((n - 1) / n))
    return dict(DM=dm, p_value=float(2 * stats.t.sf(abs(dm), n - 1)),
                mean_difference=float(d.mean()), hac_lags=lag, status="ok")


def pairwise_dm(y, forecasts, alpha=ALPHA):
    if not forecasts.index.equals(y.index) or forecasts.isna().any().any():
        raise ValueError("DM requires complete forecasts on identical dates.")
    rows = []
    for a, b in itertools.combinations(forecasts.columns, 2):
        rows.append(dict(model_A=a, model_B=b, **diebold_mariano(
            quantile_loss(y, forecasts[a], alpha), quantile_loss(y, forecasts[b], alpha))))
    result = pd.DataFrame(rows)
    valid = result.p_value.notna()
    result['p_Holm'] = np.nan
    if valid.any():
        result.loc[valid, 'p_Holm'] = multipletests(result.loc[valid, 'p_value'], method='holm')[1]
    return result


def parse_tankermap(payload):
    if payload.get('ticker', '').upper() != 'WTI':
        raise ValueError("This project expects the WTI feed.")
    frame = pd.DataFrame(payload['bars'])
    if not {'time', 'close'} <= set(frame):
        raise ValueError("Unexpected TankerMap schema: expected bars/time/close.")
    return clean_prices(frame.rename(columns={'time': 'date'})[['date', 'close']])


def clean_prices(frame):
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
    frame['close'] = pd.to_numeric(frame['close'], errors='raise')
    if frame['date'].isna().any() or frame['date'].duplicated().any():
        raise ValueError("Missing or duplicated price dates; resolve at the source.")
    if not np.isfinite(frame['close']).all():
        raise ValueError("Nonfinite prices; explicitly remove source missing-value rows first.")
    return frame.set_index('date')['close'].sort_index().rename('WTI_USD_per_barrel')


def load_tankermap(cache='data/wti_tankermap_max.json', refresh=False):
    path = Path(cache)
    if refresh or not path.exists():
        response = requests.get('https://tankermap.com/api/market-data/wti',
                                params={'period': 'MAX'}, timeout=90)
        response.raise_for_status()
        payload = response.json()
        parse_tankermap(payload)  # Validate before replacing a working snapshot.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        path.with_suffix('.provenance.json').write_text(json.dumps({
            'url': response.url, 'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
    payload = json.loads(path.read_text())
    return parse_tankermap(payload), payload.get('meta', {})


def coverage_table(prices, start=2006, end=2025):
    p = prices.loc[f'{start}':f'{end}']
    rows = []
    for year in range(start, end + 1):
        yearly = p[p.index.year == year]
        rows.append(dict(year=year, observations=len(yearly),
                         first=yearly.index.min(), last=yearly.index.max()))
    return pd.DataFrame(rows).set_index('year')


def require_coverage(prices, start='2006-05-31', end='2025-12-31'):
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    table = coverage_table(prices, first.year, last.year)
    bad = []
    for year, row in table.iterrows():
        left, right = max(first, pd.Timestamp(year, 1, 1)), min(last, pd.Timestamp(year, 12, 31))
        expected = len(pd.bdate_range(left, right))
        if (row.observations < .75 * expected or pd.isna(row['first'])
                or row['first'] > left + pd.Timedelta(days=14)
                or row['last'] < right - pd.Timedelta(days=14)):
            bad.append(year)
    if bad:
        raise ValueError(f"Insufficient {start}–{end} daily coverage in years {bad}. "
                         f"Source range: {prices.index.min().date()}–{prices.index.max().date()}. "
                         "Supply earlier WTI data or explicitly revise the study. No data were fabricated.")


def raw_returns(prices):
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Prices must be date-ordered with unique dates.")
    if not np.isfinite(prices).all():
        raise ValueError("Missing prices must be resolved before calculating returns.")
    if (prices.shift(1) == 0).any():
        raise ValueError("Raw percentage returns are undefined after a zero price.")
    # Negative prices are retained; they are economically material for WTI.
    return ((prices - prices.shift(1)) / prices.shift(1)).dropna().rename('return')


def information_criteria(loglik, k, n):
    return dict(loglik=float(loglik), k=k, n=n,
                AIC=float(2 * k - 2 * loglik), BIC=float(np.log(n) * k - 2 * loglik))


def fit_ghyp(y, warm=None):
    """Multistart, bounded MLE; normalize data, then restore original units.

    Numerical bounds are explicit and any boundary estimate is reported.
    They are safeguards, not claims of a global optimum.
    """
    y = np.asarray(y, float)
    center, scale = np.mean(y), np.std(y)
    if not np.isfinite(y).all() or scale < 1e-12:
        raise ValueError("GH requires finite, nonconstant observations.")
    z = (y - center) / scale
    bounds = [(-15, 15), (-7, 7), (-0.995, 0.995), (-5, 5), (-7, 4)]

    def decode(v):
        p, loga, ratio, loc, logs = v
        a = np.exp(loga)
        return p, a, a * ratio, loc, np.exp(logs)

    def objective(v):
        with np.errstate(all='ignore'):
            ll = stats.genhyperbolic.logpdf(z, *decode(v))
        return -float(ll.sum()) if np.isfinite(ll).all() else 1e100

    starts = [[-0.5, 0, 0, 0, 0], [-2, -1, 0, 0, 0], [1, 0.5, 0, 0, 0]]
    if warm is not None:
        p, a, b, loc, s = warm
        starts.insert(0, [p, np.log(a), b / a, (loc - center) / scale, np.log(s / scale)])
    fits = []
    for start in starts:
        initial = np.clip(start, np.array(bounds)[:, 0] + 1e-6, np.array(bounds)[:, 1] - 1e-6)
        res = optimize.minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 800, 'ftol': 1e-10})
        if res.success and np.isfinite(res.fun) and res.fun < 1e90:
            fits.append(res)
    if not fits:
        raise RuntimeError("All GH optimization starts failed.")
    best = min(fits, key=lambda x: x.fun)
    p, a, b, loc, s = decode(best.x)
    params = np.array([p, a, b, center + scale * loc, scale * s])
    boundary = any(min(abs(v - lo), abs(v - hi)) < 1e-4
                   for v, (lo, hi) in zip(best.x, bounds))
    return dict(params=params, **information_criteria(-best.fun - len(y) * np.log(scale), 5, len(y)),
                converged=True, boundary=boundary, optimizer_message=str(best.message))


def fit_distributions(y):
    y = np.asarray(y, float)
    result = {}
    for name, distribution in [('Normal', stats.norm), ('Student-t', stats.t)]:
        params = distribution.fit(y)
        result[name] = dict(params=np.array(params), converged=True,
                            **information_criteria(distribution.logpdf(y, *params).sum(), len(params), len(y)))
    # Hansen skew-t has eta, lambda, location and scale (4 free parameters).
    center, scale = np.mean(y), np.std(y)
    z = (y - center) / scale
    skew = SkewStudent()

    def skew_objective(v):
        eta, lam, loc, logs = v
        with np.errstate(all='ignore'):
            ll = skew.loglikelihood([eta, lam], z - loc, np.full(len(z), np.exp(2 * logs)))
        return -float(ll) if np.isfinite(ll) else 1e100

    fits = [optimize.minimize(skew_objective, [eta, 0, 0, 0], method='L-BFGS-B',
                             bounds=[(2.05, 200), (-.99, .99), (-5, 5), (-7, 4)])
            for eta in (4, 8, 20)]
    fits = [r for r in fits if r.success and np.isfinite(r.fun)]
    if not fits:
        raise RuntimeError("Skew-t estimation did not converge.")
    fit = min(fits, key=lambda r: r.fun)
    eta, lam, loc, logs = fit.x
    result['Hansen skew-t'] = dict(params=np.array([eta, lam, center + scale * loc, scale * np.exp(logs)]),
        converged=True, **information_criteria(-fit.fun - len(y) * np.log(scale), 4, len(y)))
    result['GH'] = fit_ghyp(y)
    table = pd.DataFrame({k: {key: value for key, value in v.items() if key != 'params'}
                          for k, v in result.items()}).T
    return result, table


GARCH_SPECS = {'GARCH(1,1)': (1, 0, 1), 'GARCH(2,2)': (2, 0, 2), 'GJR-GARCH(1,1)': (1, 1, 1)}


def fit_garch(y, spec='GJR-GARCH(1,1)', warm=None):
    p, o, q = GARCH_SPECS[spec]
    model = arch_model(100 * np.asarray(y, float), mean='Constant', vol='GARCH',
                       p=p, o=o, q=q, dist='t', rescale=False)
    with warnings.catch_warnings(record=True) as caught:
        fit = model.fit(disp='off', starting_values=warm, options={'maxiter': 1500})
        if fit.convergence_flag != 0 and warm is not None:
            fit = model.fit(disp='off', options={'maxiter': 3000})
    if fit.convergence_flag != 0:
        raise RuntimeError(f"GARCH optimizer failed: {fit.optimization_result.message}")
    fit.var_project_warnings = '; '.join(str(w.message) for w in caught)
    return fit


def garch_quantile(fit, alpha=ALPHA):
    forecast = fit.forecast(horizon=1, reindex=False)
    mu = float(forecast.mean.iloc[-1, 0])
    sigma = float(np.sqrt(forecast.residual_variance.iloc[-1, 0]))
    # arch's t quantile is standardized to unit variance.
    distribution = fit.model.distribution
    pars = fit.params[distribution.parameter_names()].to_numpy()
    z = float(np.asarray(distribution.ppf(alpha, pars)).item())
    return (mu + sigma * z) / 100


@njit(cache=True)
def caviar_path(beta, y, q0, kind):
    """n fitted quantiles plus the n+1 one-step forecast, in input units."""
    q = np.empty(len(y) + 1)
    q[0] = q0
    for t in range(1, len(y) + 1):
        lag = y[t - 1]
        if kind == 0:  # SAV
            q[t] = beta[0] + beta[1] * q[t - 1] + beta[2] * abs(lag)
        elif kind == 1:  # AS: negative-part MAGNITUDE, not signed min(lag, 0)
            q[t] = beta[0] + beta[1] * q[t - 1] + beta[2] * max(lag, 0) + beta[3] * max(-lag, 0)
        else:  # Indirect GARCH: negative root for a lower-tail quantile.
            q[t] = -np.sqrt(beta[0] + beta[1] * q[t - 1] ** 2 + beta[2] * lag ** 2)
    return q


def fit_caviar(y, spec='AS', alpha=ALPHA, seed=SEED, starts=8, warm=None, burn=75):
    y = 100 * np.asarray(y, float)
    if len(y) < burn + 30 or not np.isfinite(y).all() or np.std(y) < 1e-10:
        raise ValueError("CAViaR needs finite variable observations beyond burn-in.")
    kind = {'SAV': 0, 'AS': 1, 'INDGARCH': 2}[spec]
    q0 = float(np.quantile(y[:burn], alpha))
    scale = max(np.std(y), 0.1)
    if kind < 2:
        bounds = [(-10 * scale, 10 * scale), (0, .999), (-10, 10)]
        base = [q0 * .1, .8, -.1]
        if kind == 1:
            bounds += [(-10, 10)]
            base += [-.1]
    else:
        bounds = [(1e-8, 100 * scale ** 2), (0, .999), (0, 20)]
        base = [max(q0 ** 2 * .05, .01), .8, .1]

    def objective(beta):
        q = caviar_path(beta, y, q0, kind)[:-1]
        loss = (alpha - (y[burn:] < q[burn:])) * (y[burn:] - q[burn:])
        return float(np.mean(loss)) if np.isfinite(loss).all() else 1e100

    rng = np.random.default_rng(seed)
    candidates = [np.array(base, float)]
    if warm is not None:
        candidates.insert(0, np.asarray(warm, float))
    while len(candidates) < starts:
        b = np.array(base, float)
        b[1] = rng.uniform(.1, .97)
        b[0] = abs(q0) ** 2 * rng.uniform(.01, .5) if kind == 2 else q0 * (1 - b[1])
        b[2:] = rng.uniform(.01, .6, len(b) - 2) * (1 if kind == 2 else -1)
        candidates.append(b)
    successful = []
    for candidate in candidates:
        candidate = np.clip(candidate, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        res = optimize.minimize(objective, candidate, method='Nelder-Mead', bounds=bounds,
                                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.success and np.isfinite(res.fun):
            successful.append(res)
    if not successful:
        raise RuntimeError(f"All {spec} CAViaR optimization starts failed.")
    fit = min(successful, key=lambda r: r.fun)
    q = caviar_path(fit.x, y, q0, kind) / 100
    # ALD working likelihood, on decimal returns; estimated scale counts as a parameter.
    loss = quantile_loss(y[burn:] / 100, q[burn:-1], alpha)
    ald_scale = float(loss.mean())
    loglik = len(loss) * (np.log(alpha * (1 - alpha)) - np.log(ald_scale) - 1)
    boundary = any(min(abs(v - lo), abs(v - hi)) < 1e-5 for v, (lo, hi) in zip(fit.x, bounds))
    return dict(spec=spec, kind=kind, params=fit.x, q=q[:-1], forecast=float(q[-1]),
                q0=q0, burn=burn, quantile_loss=ald_scale, converged=True, boundary=boundary,
                successful_starts=len(successful), **information_criteria(loglik, len(fit.x) + 1, len(loss)))


def caviar_gradient(y, fit):
    y = np.asarray(y, float) * 100
    beta = fit['params']
    columns = []
    for j in range(len(beta)):
        step = 1e-5 * max(1, abs(beta[j]))
        plus, minus = beta.copy(), beta.copy()
        plus[j] += step
        minus[j] -= step
        if fit['kind'] == 2 and minus[j] < 0:
            minus[j] = beta[j]
        qp = caviar_path(plus, y, fit['q0'], fit['kind'])[:-1]
        qm = caviar_path(minus, y, fit['q0'], fit['kind'])[:-1]
        columns.append((qp - qm) / (plus[j] - minus[j]) / 100)
    return np.column_stack(columns)


def _gh_forecast_job(arguments):
    """Independent deterministic GH refit, suitable for local process workers."""
    values, alpha = arguments
    try:
        fit = fit_ghyp(values)
        value = float(stats.genhyperbolic.ppf(alpha, *fit['params']))
        if not np.isfinite(value):
            raise RuntimeError('Nonfinite GH quantile.')
        return {'fit': fit, 'value': value}
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def rolling_forecasts(returns, model, start='2018-01-01', end='2025-12-31',
                      window=504, alpha=ALPHA, output_dir='results', progress=True,
                      caviar_starts=5, seed=SEED, gh_workers=1):
    """Strict one-step rolling refits, checkpointed after each 25 dates.

    Every fit sees exactly returns.iloc[i-window:i]. Failed dates are recorded
    as NaN; they are never silently dropped or filled with another model.
    Checkpoints are invalidated by input, source-code, config or package changes.
    """
    if model not in {'GH', 'GJR-GARCH', 'AS-CAViaR'}:
        raise ValueError("Unknown rolling model.")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing or not np.isfinite(returns).all():
        raise ValueError("Returns must be finite and uniquely date-ordered.")
    selected = returns.loc[start:end]
    if selected.empty:
        raise ValueError("No observations in the forecast period.")
    if returns.index.get_loc(selected.index[0]) < window:
        raise ValueError("Insufficient pre-forecast history for a full window.")
    config = dict(model=model, start=start, end=end, window=window, alpha=alpha,
                  caviar_starts=caviar_starts, seed=seed,
                  versions={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'arch', 'numba']})
    digest = hashlib.sha256(pd.util.hash_pandas_object(returns, index=True).values.tobytes()
                            + json.dumps(config, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f'{model}_{digest[:16]}.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'fingerprint': digest, 'config': config, 'rows': []}
    if saved['fingerprint'] != digest:
        raise ValueError("Checkpoint fingerprint mismatch.")
    rows = saved['rows']
    warm = np.array(rows[-1]['params']) if rows and rows[-1]['status'] == 'ok' else None
    dates = selected.index[len(rows):]
    executor = None
    gh_results = None
    if model == 'GH':
        arguments = [(returns.iloc[returns.index.get_loc(d) - window:returns.index.get_loc(d)].to_numpy(), alpha)
                     for d in dates]
        if gh_workers > 1 and len(dates):
            executor = ProcessPoolExecutor(max_workers=gh_workers)
            gh_results = executor.map(_gh_forecast_job, arguments, chunksize=1)
        else:
            gh_results = map(_gh_forecast_job, arguments)

    def save():
        saved['rows'] = rows
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(saved, indent=2, allow_nan=False))
        temporary.replace(checkpoint)

    try:
        for date in tqdm(dates, desc=f'{model}: daily refits', disable=not progress):
            i = returns.index.get_loc(date)
            history = returns.iloc[i - window:i]
            row = dict(date=date.isoformat(), train_start=history.index[0].isoformat(),
                       train_end=history.index[-1].isoformat(), n_train=len(history),
                       q=None, status='failed', params=None, detail='', boundary=False)
            try:
                if model == 'GH':
                    outcome = next(gh_results)
                    if 'error' in outcome:
                        raise RuntimeError(outcome['error'])
                    fit, value = outcome['fit'], outcome['value']
                    params = fit['params']
                    row['boundary'] = fit['boundary']
                    row['detail'] = fit['optimizer_message']
                elif model == 'GJR-GARCH':
                    fit = fit_garch(history, warm=warm)
                    value, params = garch_quantile(fit, alpha), fit.params.to_numpy()
                    row['detail'] = fit.var_project_warnings
                else:
                    fit = fit_caviar(history, alpha=alpha, seed=seed + i, starts=caviar_starts, warm=warm)
                    value, params = fit['forecast'], fit['params']
                    row['boundary'] = fit['boundary']
                    row['detail'] = f"{fit['successful_starts']} converged starts"
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite forecast quantile.")
                row.update(q=value, status='ok', params=params.tolist())
                warm = params
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                row['detail'] = f'{type(exc).__name__}: {exc}'
                warm = None
            rows.append(row)
            if len(rows) % 25 == 0:
                save()
    finally:
        save()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    frame = pd.DataFrame(rows).set_index('date')
    frame.index = pd.to_datetime(frame.index)
    frame['q'] = pd.to_numeric(frame['q'], errors='coerce')
    if not frame.index.equals(selected.index):
        raise ValueError("Forecast dates do not match target returns.")
    frame.to_csv(directory / f'{model}_forecasts.csv')
    return frame


def complete_quantiles(frame):
    bad = frame[(frame.status != 'ok') | frame.q.isna()]
    if len(bad):
        raise ValueError(f"{len(bad)} daily fits failed. Inspect diagnostics before backtesting. "
                         "Resolve the fits and invalidate the checkpoint to retry.")
    return frame.q


def plot_var(y, q, title, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    paired(y, q)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y.index, y, color='#7993aa', lw=.65, label='Daily raw return')
    ax.plot(q.index, q, color='#b64a3a', lw=1.1, label='95% VaR: lower 5% return quantile')
    violation = y < q
    ax.scatter(y.index[violation], y[violation], s=27, color='#9e192c',
               edgecolor='black', linewidth=.5, zorder=4, label=f'Exceptions ({violation.sum()})')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title=title, xlabel='Date', ylabel='Return')
    ax.legend(loc='best')
    return ax
