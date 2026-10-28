import numpy as np
import pandas as pd
import pytest

from ninecat.model.baselines import RookiePrior, prior_rates
from ninecat.model.dataset import RATE_STATS
from ninecat.model.shrinkage import (
    MINUTES_FEATURES,
    AgeCurve,
    MinutesModel,
    equivalent_games,
    fit_k,
    fit_shrinkage,
    shrink,
)
from tests.test_model_baselines import frame


def test_shrink_formula_and_edges():
    s, n, p = pd.Series([0.6, 0.6, 0.6, np.nan]), pd.Series([100.0, 0.0, 100.0, 0.0]), pd.Series([0.3, 0.3, np.nan, 0.3])
    got = shrink(s, n, p, 300.0)
    assert got[0] == pytest.approx((100 * 0.6 + 300 * 0.3) / 400)
    assert got[1] == pytest.approx(0.3)            # no evidence: the prior
    assert got[2] == pytest.approx(0.6)            # no prior: the evidence
    assert got[3] == pytest.approx(0.3)


@pytest.mark.parametrize("k_true", [25.0, 300.0, 4000.0])
def test_fit_k_recovers_a_known_k(k_true):
    rng = np.random.default_rng(1)
    n = pd.Series(rng.uniform(20, 1500, 2000))
    so_far, prior = pd.Series(rng.uniform(0.1, 0.9, 2000)), pd.Series(rng.uniform(0.1, 0.9, 2000))
    actual = (n * so_far + k_true * prior) / (n + k_true)
    k = fit_k(so_far, n, prior, actual, pd.Series(rng.uniform(50, 2000, 2000)))
    assert k == pytest.approx(k_true, rel=0.03)


def test_fit_k_is_bayes_optimal_on_simulated_players():
    """True talent ~ N(prior, tau^2), so-far rate ~ N(talent, sigma^2 / n): the best k is sigma^2 / tau^2."""
    rng = np.random.default_rng(7)
    m, prior, tau, sigma = 20000, 0.5, 0.05, 1.0
    talent = rng.normal(prior, tau, m)
    n = rng.uniform(30, 800, m)
    so_far = talent + rng.normal(0, sigma / np.sqrt(n))
    actual = talent + rng.normal(0, sigma / np.sqrt(2000), m)
    k = fit_k(pd.Series(so_far), pd.Series(n), pd.Series(np.full(m, prior)), pd.Series(actual), pd.Series(np.ones(m)))
    assert k == pytest.approx(sigma ** 2 / tau ** 2, rel=0.15)


def test_minutes_model_recovers_linear_coefficients_per_checkpoint():
    rng = np.random.default_rng(3)
    rows = []
    for k in (5, 20):
        g = np.full(400, float(k))
        cur = rng.uniform(10, 36, 400)
        rows.append(frame(cur=list(zip(g, cur * g, np.full(400, 0.5))),
                          p1=list(zip(np.full(400, 60.0), rng.uniform(10, 36, 400) * 60, np.full(400, 0.5)))).assign(k=k))
    df = pd.concat(rows, ignore_index=True)
    df["l5_G"], df["l5_MIN"] = 5.0, df["cur_MIN"] / df["cur_G"] * 5 + rng.uniform(-10, 10, len(df))
    df["l5_GS"], df["cur_GS"] = 5.0, df["cur_G"]
    df["age"] = rng.uniform(20, 34, len(df))
    prior = df["p1_MIN"] / df["p1_G"]
    beta = {5: np.array([2, 0.5, 0.2, 0.3, 0, 0, -0.1]), 20: np.array([1, 0.7, 0.1, 0.2, 0, 0, -0.2])}
    x = MinutesModel.features(df, prior)
    truth = np.array([x[i] @ beta[k] for i, k in enumerate(df["k"])])
    df["y_G"], df["y_MIN"] = 30.0, truth * 30
    model = MinutesModel.fit(df, prior)
    for k in (5, 20):
        est, true = dict(zip(MINUTES_FEATURES, model.coef[k])), dict(zip(MINUTES_FEATURES, beta[k]))
        for f in ("so_far", "last5", "prior", "age"):
            assert est[f] == pytest.approx(true[f], abs=1e-6)
        # everyone here has a prior season, so only the intercept's total is identified
        assert est["const"] + est["has_prior"] == pytest.approx(true["const"] + true["has_prior"], abs=1e-6)
    assert model.predict(df, prior).to_numpy() == pytest.approx(np.clip(truth, 0, 48))
    assert model.checkpoint_for(pd.Series([1.0, 5.0, 12.0, 60.0])).tolist() == [5, 5, 5, 20]


def test_age_curve_is_one_when_the_prior_was_right_and_scales_rates():
    df = frame(cur=[(10, 300, 0.5)] * 4, p1=[(70, 2100, 0.5)] * 4, p2=[(70, 2100, 0.5)] * 4,
               y=[(40, 1200, 0.5), (40, 1200, 0.5), (40, 1200, 0.6), (40, 1200, 0.6)])
    df["age"] = [21.0, 21.5, 33.0, 34.0]
    rookie = RookiePrior.fit(df)
    curve = AgeCurve.fit(df, rookie)
    assert curve.factors.loc[0, "PTS"] == pytest.approx(1.0) and curve.factors.loc[4, "PTS"] == pytest.approx(1.2)
    adj = curve.apply(df, prior_rates(df, rookie))
    assert adj.loc[2, "PTS"] == pytest.approx(0.6) and adj.loc[0, "PTS"] == pytest.approx(0.5)


def test_fitted_model_predicts_a_full_line_and_reports_equivalent_games():
    rng = np.random.default_rng(5)
    n = 300
    spec = lambda g, mpg: list(zip(g, g * mpg, rng.uniform(0.05, 0.6, n)))
    df = frame(cur=spec(np.full(n, 10.0), rng.uniform(12, 36, n)), p1=spec(np.full(n, 60.0), rng.uniform(12, 36, n)),
               p2=spec(np.full(n, 60.0), rng.uniform(12, 36, n)), y=spec(np.full(n, 50.0), rng.uniform(12, 36, n)),
               l5=spec(np.full(n, 5.0), rng.uniform(12, 36, n)))
    df["k"], df["age"] = 10, rng.uniform(20, 35, n)
    model = fit_shrinkage(df)
    line = model.predict(df)
    assert set(RATE_STATS) <= set(line.columns) and {"FG%", "FT%", "FGM", "FTM"} <= set(line.columns)
    assert line.notna().all().all() and ((line["MIN"] >= 0) & (line["MIN"] <= 48)).all()
    eq = equivalent_games(model, df)
    assert eq["PTS"] == pytest.approx(model.k["PTS"] / 30)
    assert set(eq) == set(RATE_STATS) | {"FG%", "FT%"}
    assert set(model.k) == set(eq)
