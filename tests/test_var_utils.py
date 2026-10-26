import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import var_utils as v


def test_loss_signs_and_scaling():
    y, q = np.array([-.10, -.03, .02]), np.array([-.05, -.05, -.05])
    np.testing.assert_allclose(v.quantile_loss(y, q), [.0475, .001, .0035])
    np.testing.assert_allclose(v.lopez_loss(y, q), [1.0025, 0, 0])
    np.testing.assert_allclose(v.blanco_ihle_loss(y, q), [1, 0, 0])
    with pytest.raises(ValueError):
        v.blanco_ihle_loss([0], [0])


def test_kupiec_known_coverage_and_extremes():
    q = np.zeros(100)
    y = np.r_[-np.ones(5), np.ones(95)]
    assert v.kupiec_test(y, q)['LR_uc'] == pytest.approx(0)
    for y in [np.ones(100), -np.ones(100)]:
        assert np.isfinite(v.kupiec_test(y, q)['LR_uc'])
        assert np.isnan(v.christoffersen_test(y, q)['p_ind'])


def test_christoffersen_clustering():
    hit = np.tile(np.r_[np.zeros(95), np.ones(5)], 20)
    result = v.christoffersen_test(.5 - hit, np.zeros(len(hit)))
    assert result['p_ind'] < .001
    assert sum(result[k] for k in ['n00', 'n01', 'n10', 'n11']) == len(hit) - 1


def test_dq_independent_reference_matrix():
    rng = np.random.default_rng(42)
    y = rng.normal(size=800)
    q = np.full(len(y), stats.norm.ppf(.05)) + .05 * np.sin(np.arange(len(y)))
    hits = (y < q) - .05
    x = np.column_stack([np.ones(796), *[hits[4-j:-j] for j in range(1, 5)], q[4:]])
    projection = x @ np.linalg.lstsq(x, hits[4:], rcond=None)[0]
    expected = projection @ projection / (.05 * .95)
    assert v.dq_test(y, q)['DQ'] == pytest.approx(expected)


def test_dm_direction_and_holm():
    rng = np.random.default_rng(34)
    a = rng.uniform(0, .01, 300)
    b = a + rng.uniform(.01, .02, 300)
    result = v.diebold_mariano(a, b)
    assert result['DM'] < 0 and result['p_value'] < .01
    assert np.isnan(v.diebold_mariano(a, a)['p_value'])


def test_no_silent_alignment_or_missing_forecasts():
    a = pd.Series([1, 2], index=[1, 2])
    b = pd.Series([1, 2], index=[2, 3])
    with pytest.raises(ValueError):
        v.quantile_loss(a, b)
    with pytest.raises(ValueError):
        v.kupiec_test([1, np.nan], [0, 0])


def test_price_returns_preserve_negative_and_do_not_fill():
    idx = pd.date_range('2020-04-17', periods=3)
    p = pd.Series([20., -30., 10.], index=idx)
    np.testing.assert_allclose(v.raw_returns(p), [-2.5, -4/3])
    with pytest.raises(ValueError):
        v.raw_returns(pd.Series([1., 0., 2.], index=idx))


def test_parser_rejects_wrong_series_duplicates():
    with pytest.raises(ValueError):
        v.parse_tankermap({'ticker': 'URALS', 'bars': []})
    payload = {'ticker': 'WTI', 'bars': [{'time': '2020-01-01', 'close': 40}] * 2}
    with pytest.raises(ValueError):
        v.parse_tankermap(payload)


def test_coverage_detects_missing_training_years():
    p, _ = v.load_tankermap()
    v.require_coverage(p)
    with pytest.raises(ValueError, match='2000'):
        v.require_coverage(p, start='2000-01-01')


def test_caviar_recursion_is_predetermined():
    y = np.array([1., -2., 3.])
    beta = np.array([-.1, .8, -.2, -.4])
    q = v.caviar_path(beta, y, -1, 1)
    np.testing.assert_allclose(q[:3], [-1, -1.1, -1.78])
    altered = y.copy(); altered[1] = 999
    np.testing.assert_allclose(q[:2], v.caviar_path(beta, altered, -1, 1)[:2])
    assert np.all(v.caviar_path(np.array([.1, .8, .2]), y, -1, 2) < 0)


def test_rolling_window_is_past_only_and_resume_is_exact(tmp_path, monkeypatch):
    idx = pd.bdate_range('2017-01-01', periods=130)
    returns = pd.Series(np.linspace(-.02, .03, len(idx)), index=idx)
    seen = []

    def fake_fit(y, **kwargs):
        seen.append(y.copy())
        return dict(forecast=float(y.mean()), params=np.array([1., 2., 3., 4.]),
                    boundary=False, successful_starts=1)

    monkeypatch.setattr(v, 'fit_caviar', fake_fit)
    kwargs = dict(model='AS-CAViaR', start=str(idx[125].date()), end=str(idx[-1].date()),
                  window=100, output_dir=tmp_path, progress=False)
    result = v.rolling_forecasts(returns, **kwargs)
    for j, history in enumerate(seen):
        pd.testing.assert_series_equal(history, returns.iloc[25+j:125+j])
    assert (pd.to_datetime(result.train_end) < result.index).all()
    again = v.rolling_forecasts(returns, **kwargs)
    pd.testing.assert_frame_equal(result, again)
    assert len(seen) == 5
    # Changing a future return cannot change the first forecast (cold seed identical).
    changed = returns.copy(); changed.iloc[-1] = 999
    new = v.rolling_forecasts(changed, **(kwargs | {'output_dir': tmp_path / 'changed'}))
    assert result.q.iloc[0] == new.q.iloc[0]


def test_fit_failure_is_visible_and_blocks_backtest(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('deliberate test failure')
    monkeypatch.setattr(v, 'fit_caviar', failed)
    idx = pd.bdate_range('2017-01-01', periods=110)
    y = pd.Series(np.linspace(-.02, .03, len(idx)), index=idx)
    result = v.rolling_forecasts(y, 'AS-CAViaR', start=str(idx[-1].date()),
                                window=100, output_dir=tmp_path, progress=False)
    assert result.q.isna().all()
    with pytest.raises(ValueError, match='failed'):
        v.complete_quantiles(result)


def test_garch_standardized_student_quantile():
    class Forecast:
        mean = pd.DataFrame([[.2]])
        residual_variance = pd.DataFrame([[4.]])
    class Fit:
        from arch.univariate import StudentsT
        class Model:
            from arch.univariate import StudentsT
            distribution = StudentsT()
        model = Model()
        params = pd.Series({'nu': 6.})
        def forecast(self, **kwargs):
            return Forecast()
    expected = (.2 + 2 * stats.t.ppf(.05, 6) * np.sqrt(4/6)) / 100
    assert v.garch_quantile(Fit()) == pytest.approx(expected)


def test_caviar_fits_and_gradient_are_finite():
    y = np.random.default_rng(2).standard_t(6, 220) * .01
    for spec in ['SAV', 'AS', 'INDGARCH']:
        fit = v.fit_caviar(y, spec=spec, starts=2)
        assert fit['converged'] and np.isfinite(fit['forecast'])
        assert fit['forecast'] < 0
        g = v.caviar_gradient(y, fit)
        assert g.shape == (len(y), len(fit['params'])) and np.isfinite(g).all()
        assert fit['k'] == len(fit['params']) + 1
