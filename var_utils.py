"""Reusable estimators and diagnostics for the WTI 95% VaR notebook.

All public return/quantile inputs use decimal returns. VaR lines are signed lower
tail quantiles: an exception is r_t < q_t. No missing forecasts are imputed.
"""
from __future__ import annotations

import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from arch import arch_model
from arch.univariate import SkewStudent
from numba import njit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
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
    loss = quantile_loss(y[burn:] / 100, q[burn:-1], alpha)
    return dict(spec=spec, kind=kind, params=fit.x, q=q[:-1], forecast=float(q[-1]),
                q0=q0, burn=burn, quantile_loss=float(loss.mean()), n=len(loss))


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
                      window=504, alpha=ALPHA, progress=False,
                      caviar_starts=5, seed=SEED, gh_workers=1):
    """Strict one-step rolling forecasts, using only the prior window."""
    if model not in {'GH', 'GJR-GARCH', 'AS-CAViaR'}:
        raise ValueError("Unknown rolling model.")
    if not returns.index.is_unique or not returns.index.is_monotonic_increasing or not np.isfinite(returns).all():
        raise ValueError("Returns must be finite and uniquely date-ordered.")
    selected = returns.loc[start:end]
    if selected.empty:
        raise ValueError("No observations in the forecast period.")
    if returns.index.get_loc(selected.index[0]) < window:
        raise ValueError("Insufficient pre-forecast history for a full window.")
    values = []
    warm = None
    dates = selected.index
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

    try:
        for date in tqdm(dates, desc=f'{model}: daily refits', disable=not progress):
            i = returns.index.get_loc(date)
            history = returns.iloc[i - window:i]
            try:
                if model == 'GH':
                    outcome = next(gh_results)
                    if 'error' in outcome:
                        raise RuntimeError(outcome['error'])
                    fit, value = outcome['fit'], outcome['value']
                    params = fit['params']
                elif model == 'GJR-GARCH':
                    fit = fit_garch(history, warm=warm)
                    value, params = garch_quantile(fit, alpha), fit.params.to_numpy()
                else:
                    fit = fit_caviar(history, alpha=alpha, seed=seed + i, starts=caviar_starts, warm=warm)
                    value, params = fit['forecast'], fit['params']
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite forecast quantile.")
                values.append(value)
                warm = params
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(f'{model} could not estimate the forecast for {date.date()}: {exc}') from exc
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return pd.Series(values, index=selected.index, name=model)


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


def plot_var_periods(y, q, model_name):
    """Two readable test-period panels with exceptions marked."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    paired(y, q)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    periods = [('2018', '2021'), ('2022', '2025')]
    for ax, (start, end) in zip(axes, periods):
        sample_y, sample_q = y.loc[start:end], q.loc[start:end]
        hit = sample_y < sample_q
        ax.plot(sample_y.index, sample_y, color='#89949c', lw=.65, label='Return')
        ax.plot(sample_q.index, sample_q, color='#b64a3a', lw=1.15, label='95% VaR')
        ax.scatter(sample_y.index[hit], sample_y[hit], s=24, color='#9e192c', edgecolor='black',
                   linewidth=.4, zorder=4, label=f'Exceptions ({int(hit.sum())})')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=f'{model_name}: {start}–{end}', ylabel='Return')
        ax.legend(loc='best')
    axes[-1].set_xlabel('Date')
    fig.tight_layout()
    return fig


# --- Compact notebook summaries -------------------------------------------------

WIDTH = 76


def _line(char='='):
    print(char * WIDTH)


def _num(value):
    if value is None or not np.isfinite(value):
        return 'NA'
    if value != 0 and abs(value) < 0.0005:
        return f'{value:.2e}'
    return f'{value:.3f}'


def _pvalue(value):
    if value is None or not np.isfinite(value):
        return 'NA'
    return '<0.001' if value < 0.001 else f'{value:.3f}'


def _decision(pvalue, alpha=.05):
    if pvalue is None or not np.isfinite(pvalue):
        return 'NA'
    return 'FAIL' if pvalue < alpha else 'PASS'


def _stars(pvalue):
    if not np.isfinite(pvalue):
        return ''
    if pvalue < .001:
        return '***'
    if pvalue < .01:
        return '**'
    if pvalue < .05:
        return '*'
    return ''


def print_download_summary(info):
    """Print a short, human-readable record of the one WTI download."""
    print('Download Summary')
    _line('-')
    for label, value in info.items():
        print(f'{label:<30}{value:>24}')
    _line('-')


def print_split_summary(splits):
    print('Sample Coverage')
    _line('=')
    print(f"{'Sample':<16}{'First return':<16}{'Last return':<16}{'Observations':>14}")
    _line('-')
    for name, values in splits.items():
        print(f'{name:<16}{values.index.min():%Y-%m-%d}      {values.index.max():%Y-%m-%d}{len(values):>14,}')
    _line('=')


def print_return_summary(returns, asset='WTI', alpha=.05):
    """R-style descriptive, normality and Ljung–Box report."""
    x = np.asarray(returns, float)
    missing = int(pd.isna(returns).sum())
    descriptive = [
        ('Mean', np.mean(x)), ('Std. deviation', np.std(x, ddof=1)),
        ('Variance', np.var(x, ddof=1)), ('Minimum', np.min(x)),
        ('1st Quartile', np.quantile(x, .25)), ('Median', np.median(x)),
        ('3rd Quartile', np.quantile(x, .75)), ('Maximum', np.max(x)),
        ('Skewness', stats.skew(x, bias=False)), ('Excess kurtosis', stats.kurtosis(x, bias=False)),
    ]
    sw = stats.shapiro(x)
    ad_stat, ad_p = normal_ad(x)
    jb = stats.jarque_bera(x)
    lb_r = acorr_ljungbox(x, lags=[3, 5, 10], return_df=True)
    lb_sq = acorr_ljungbox(x ** 2, lags=[3, 5, 10], return_df=True)
    print('Return Distribution Summary')
    _line('=')
    print(f'Asset:          {asset}')
    print('Variable:       Winsorised arithmetic returns')
    print('Frequency:      Daily')
    print(f'Sample:         {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}')
    print(f'Observations:   {len(returns):,}')
    print(f'Missing:        {missing:,}')
    print('\nDescriptive Statistics')
    _line('-')
    print(f"{'Statistic':<28}{'Value':>12}")
    _line('-')
    for label, value in descriptive:
        print(f'{label:<28}{_num(value):>12}')
    print('\nNormality Tests')
    _line('=')
    print(f"{'Test':<24}{'Statistic':>12}{'p-value':>12}{'Decision':>12}")
    _line('-')
    for label, result in [('Shapiro-Wilk', sw), ('Anderson-Darling', (ad_stat, ad_p)), ('Jarque-Bera', jb)]:
        print(f'{label:<24}{_num(result.statistic if hasattr(result, "statistic") else result[0]):>12}'
              f'{_pvalue(result.pvalue if hasattr(result, "pvalue") else result[1]):>12}'
              f'{_decision(result.pvalue if hasattr(result, "pvalue") else result[1], alpha):>12}')
    print('\nLjung-Box Tests')
    _line('=')
    print(f"{'Lag':<8}{'Returns Q':>12}{'p-value':>12}{'Squared Q':>14}{'p-value':>12}")
    _line('-')
    for lag in [3, 5, 10]:
        print(f'{lag:<8}{_num(lb_r.loc[lag, "lb_stat"]):>12}{_pvalue(lb_r.loc[lag, "lb_pvalue"]):>12}'
              f'{_num(lb_sq.loc[lag, "lb_stat"]):>14}{_pvalue(lb_sq.loc[lag, "lb_pvalue"]):>12}')
    print('\nDiagnostics')
    _line('-')
    print(f"Normality:              {'Rejected' if min(sw.pvalue, ad_p, jb.pvalue) < alpha else 'Not rejected'}")
    print(f"Serial correlation:     {'Detected' if (lb_r.lb_pvalue < alpha).any() else 'Not detected'}")
    print(f"Volatility clustering:  {'Detected' if (lb_sq.lb_pvalue < alpha).any() else 'Not detected'}")
    print(f'Significance level:     {alpha:.3f}')
    _line('=')


def print_distribution_summary(fits):
    print('Distribution Fit Comparison')
    _line('=')
    print(f"{'Distribution':<22}{'Observations':>14}{'Log-likelihood':>18}{'AIC':>12}{'BIC':>12}")
    _line('-')
    for name, fit in fits.items():
        print(f'{name:<22}{fit["n"]:>14,}{_num(fit["loglik"]):>18}{_num(fit["AIC"]):>12}{_num(fit["BIC"]):>12}')
    _line('=')
    print('Lower AIC and BIC indicate preferred in-sample fit.')


def _garch_scale(name):
    return 1 / 10_000 if name == 'omega' else 1 / 100 if name == 'mu' else 1


def print_garch_summary(fits):
    names = list(fits)
    parameters = ['mu', 'omega', 'alpha[1]', 'alpha[2]', 'gamma[1]', 'beta[1]', 'beta[2]', 'nu']
    print('GARCH Model Comparison')
    _line('=')
    print(f"{'Parameter':<16}" + ''.join(f'{name:>19}' for name in names))
    _line('-')
    for parameter in parameters:
        values = []
        for fit in fits.values():
            if parameter not in fit.params.index:
                values.append('—')
                continue
            scale = _garch_scale(parameter)
            estimate = fit.params[parameter] * scale
            pvalue = fit.pvalues[parameter]
            values.append(_num(estimate) + _stars(pvalue))
        print(f'{parameter:<16}' + ''.join(f'{value:>19}' for value in values))
    _line('-')
    print(f"{'Log-likelihood':<16}" + ''.join(f'{_num(fit.loglikelihood + fit.nobs*np.log(100)):>19}' for fit in fits.values()))
    print(f"{'AIC':<16}" + ''.join(f'{_num((2*fit.num_params - 2*(fit.loglikelihood + fit.nobs*np.log(100)))/fit.nobs):>19}' for fit in fits.values()))
    print(f"{'BIC':<16}" + ''.join(f'{_num((np.log(fit.nobs)*fit.num_params - 2*(fit.loglikelihood + fit.nobs*np.log(100)))/fit.nobs):>19}' for fit in fits.values()))
    _line('=')
    print('Innovations: standardized Student-t. AIC/BIC are reported per observation.')
    print('Significance codes: *** p<0.001  ** p<0.01  * p<0.05')


def caviar_inference(y, fit, alpha=ALPHA):
    """Kernel sandwich standard errors for the fitted CAViaR recursion."""
    burn = fit['burn']
    gradient = caviar_gradient(y, fit)[burn:]
    residual = np.asarray(y, float)[burn:] - fit['q'][burn:]
    n = len(residual)
    robust_scale = min(np.std(residual), stats.iqr(residual) / 1.349)
    bandwidth = max(1e-8, 1.06 * robust_scale * n ** (-1 / 5))
    density_weights = (np.abs(residual) < bandwidth) / (2 * bandwidth)
    a = gradient.T @ (density_weights[:, None] * gradient) / n
    b = gradient.T @ gradient / n
    covariance = alpha * (1 - alpha) * np.linalg.pinv(a) @ b @ np.linalg.pinv(a) / n
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    z = np.divide(fit['params'], se, out=np.full_like(se, np.nan), where=se > 0)
    pvalue = 2 * stats.norm.sf(np.abs(z))
    return dict(se=se, lower=fit['params'] - 1.96 * se, upper=fit['params'] + 1.96 * se,
                pvalue=pvalue, bandwidth=bandwidth)


def print_caviar_summary(fits, y, alpha=ALPHA):
    labels = {'SAV': ['Beta 0', 'Beta 1', 'Gamma 1'],
              'AS': ['Beta 0', 'Beta 1', 'Gamma 1', 'Gamma 2'],
              'INDGARCH': ['Beta 0', 'Beta 1', 'Gamma 1']}
    inference = {name: caviar_inference(y, fit, alpha) for name, fit in fits.items()}
    rows = ['Beta 0', 'Beta 1', 'Gamma 1', 'Gamma 2']
    print('CAViaR Model Comparison')
    _line('=')
    print(f"{'Parameter':<16}" + ''.join(f'{name:>19}' for name in fits))
    _line('-')
    for row in rows:
        values = []
        for name, fit in fits.items():
            if row not in labels[name]:
                values.append('—')
                continue
            index = labels[name].index(row)
            estimate = fit['params'][index]
            values.append(_num(estimate) + _stars(inference[name]['pvalue'][index]))
        print(f'{row:<16}' + ''.join(f'{value:>19}' for value in values))
    print('\n95% confidence intervals')
    _line('-')
    print(f"{'Parameter':<16}" + ''.join(f'{name:>19}' for name in fits))
    for row in rows:
        values = []
        for name, fit in fits.items():
            if row not in labels[name]:
                values.append('—')
                continue
            index = labels[name].index(row)
            values.append(f'[{_num(inference[name]["lower"][index])}, {_num(inference[name]["upper"][index])}]')
        print(f'{row:<16}' + ''.join(f'{value:>19}' for value in values))
    _line('-')
    print(f"{'Mean quantile loss':<16}" + ''.join(f'{_num(fit["quantile_loss"]):>19}' for fit in fits.values()))
    print(f"{'In-sample coverage':<16}" + ''.join(f'{_num(np.mean(np.asarray(y)[fit["burn"]:] < fit["q"][fit["burn"]:])):>19}' for fit in fits.values()))
    _line('=')
    print('Returns are expressed in percent inside the CAViaR recursion.')
    print('95% confidence intervals use kernel sandwich standard errors.')
    print('Significance codes: *** p<0.001  ** p<0.01  * p<0.05')


def print_backtest_summary(name, result, alpha=ALPHA):
    expected = result['n'] * alpha
    print('VaR Backtesting Results')
    _line('=')
    print(f'Model:          {name}')
    print(f'VaR level:      {alpha:.1%}')
    print(f'Observations:   {int(result["n"]):,}')
    print('\nCoverage')
    _line('-')
    print(f'Exceptions:          {int(result["exceptions"])} / {int(result["n"])}')
    print(f'Expected rate:       {alpha:.3%}')
    print(f'Observed rate:       {result["rate"]:.3%}')
    print(f'Expected exceptions: {expected:.3f}')
    print('\nTransition Matrix')
    _line('-')
    print(f"{'':<20}{'I(t) = 0':>14}{'I(t) = 1':>14}{'Total':>14}")
    print(f"{'I(t-1) = 0':<20}{int(result['n00']):>14}{int(result['n01']):>14}{int(result['n00']+result['n01']):>14}")
    print(f"{'I(t-1) = 1':<20}{int(result['n10']):>14}{int(result['n11']):>14}{int(result['n10']+result['n11']):>14}")
    print(f"{'Total':<20}{int(result['n00']+result['n10']):>14}{int(result['n01']+result['n11']):>14}{int(result['n']-1):>14}")
    print(f"P(0 -> 1) = {result['n01']/(result['n00']+result['n01']):.3f}    P(1 -> 1) = {result['n11']/(result['n10']+result['n11']):.3f}")
    print('\nStatistical Tests')
    _line('=')
    print(f"{'Test':<26}{'Statistic':>12}{'p-value':>12}{'Decision':>12}")
    _line('-')
    tests = [('Kupiec uncond. coverage', result['LR_uc'], result['p_uc']), ('Christoffersen independence', result['LR_ind'], result['p_ind']),
             ('Christoffersen cond. cov.', result['LR_cc'], result['p_cc']), ('Dynamic Quantile', result['DQ_DQ'], result['DQ_p_DQ']),
             ('Exact Binomial', np.nan, result['p_binomial'])]
    for label, statistic, pvalue in tests:
        print(f'{label:<26}{_num(statistic):>12}{_pvalue(pvalue):>12}{_decision(pvalue, alpha):>12}')
    print('\nLoss Functions')
    _line('-')
    for label, key in [('Lopez', 'Lopez_mean'), ('Quantile', 'Quantile_mean'), ('Blanco-Ihle', 'Blanco_Ihle_mean')]:
        print(f'{label:<26}{_num(result[key]):>12}')
    _line('=')
    print('PASS means the null hypothesis is not rejected at the stated level.')


def print_final_backtest_summary(results):
    print('VaR Model Comparison')
    _line('=')
    print(f"{'Model':<22}{'Observations':>14}{'Exceptions':>14}{'Rate':>12}{'Rejected tests':>16}")
    _line('-')
    for name, result in results.items():
        rejected = sum(result[p] < .05 for p in ['p_uc', 'p_ind', 'p_cc', 'DQ_p_DQ', 'p_binomial'])
        print(f'{name:<22}{int(result["n"]):>14,}{int(result["exceptions"]):>14}{result["rate"]:>12.3%}{rejected:>16}')
    _line('-')
    print(f"{'Model':<22}{'Kupiec p':>14}{'IND p':>14}{'CC p':>14}{'DQ p':>16}")
    for name, result in results.items():
        print(f'{name:<22}{_pvalue(result["p_uc"]):>14}{_pvalue(result["p_ind"]):>14}{_pvalue(result["p_cc"]):>14}{_pvalue(result["DQ_p_DQ"]):>16}')
    _line('-')
    print(f"{'Model':<22}{'Lopez':>14}{'Quantile':>14}{'Blanco-Ihle':>14}")
    for name, result in results.items():
        print(f'{name:<22}{_num(result["Lopez_mean"]):>14}{_num(result["Quantile_mean"]):>14}{_num(result["Blanco_Ihle_mean"]):>14}')
    _line('=')


def print_dm_summary(results):
    print('Diebold-Mariano Test Results')
    _line('=')
    print('Loss function: Quantile loss     Variance estimator: HAC     Significance: 0.05')
    _line('-')
    print(f"{'Model A':<20}{'Model B':<20}{'DM stat.':>12}{'p-value':>12}{'Holm p':>12}{'Mean diff.':>14}")
    _line('-')
    for _, row in results.iterrows():
        print(f'{row.model_A:<20}{row.model_B:<20}{_num(row.DM):>12}{_pvalue(row.p_value):>12}'
              f'{_pvalue(row.p_Holm):>12}{_num(row.mean_difference):>14}')
    _line('=')
    print('A negative statistic favours Model A. Holm p-values adjust for three comparisons.')
