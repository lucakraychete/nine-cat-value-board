import numpy as np
import pandas as pd
import pytest

from ninecat.model.boosting import TARGETS, compare_models, features, fit_boosted
from ninecat.model.dataset import target_line, window_rates
from ninecat.model.shrinkage import fit_shrinkage
from tests.test_model_baselines import frame

FAST = {"max_iter": 40, "learning_rate": 0.2, "max_leaf_nodes": 7, "min_samples_leaf": 10,
        "early_stopping": False, "random_state": 0}


def synthetic(n: int = 600, seed: int = 2) -> pd.DataFrame:
    """Checkpoints where young players beat their numbers by a margin shrinkage can't see."""
    rng = np.random.default_rng(seed)

    def spec(g, rate):
        mpg = rng.uniform(15, 35, n)
        return list(zip(np.full(n, g), mpg * g, rate))

    talent = rng.uniform(0.2, 0.6, n)
    age = rng.uniform(20, 35, n)
    df = frame(cur=spec(10.0, talent + rng.normal(0, 0.05, n)), l5=spec(5.0, talent), l10=spec(10.0, talent),
               p1=spec(70.0, talent), p2=spec(70.0, talent),
               y=spec(50.0, talent + np.where(age < 25, 0.15, 0.0)))
    df["k"], df["age"], df["games_share"] = 10, age, 1.0
    df["nba_id"] = np.arange(n) // 2           # two checkpoints per player
    return df


def test_features_never_include_the_target():
    df = synthetic(50)
    m2 = fit_shrinkage(df)
    x = features(df, m2.rates(df))
    assert not [c for c in x.columns if c.startswith("y_")]
    changed = df.copy()
    changed[[c for c in df.columns if c.startswith("y_")]] *= 3
    pd.testing.assert_frame_equal(x, features(changed, m2.rates(changed)))


def test_boosting_learns_what_shrinkage_misses():
    train, val = synthetic(seed=2), synthetic(seed=9)
    m2 = fit_shrinkage(train)
    m3 = fit_boosted(train, m2, FAST, targets=["PTS"])    # one target: weighted fits are slow
    assert set(m3.models) == {"PTS"} and set(TARGETS) > {"PTS"}
    actual = window_rates(val, "y")["PTS"]
    err2 = (m2.rates(val)["PTS"] - actual).abs().mean()
    err3 = (m3.rates(val)["PTS"] - actual).abs().mean()
    assert err3 < 0.7 * err2
    assert m3.predict(val)["MIN"].between(0, 48).all()


def test_compare_models_needs_more_than_noise():
    df = synthetic(400)
    truth = target_line(df)
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 1, truth.shape)
    same = compare_models(df, noisy, noisy, n_boot=200)
    assert same.overall[0] == pytest.approx(0) and not same.challenger_wins
    better = compare_models(df, noisy, truth + rng.normal(0, 0.3, truth.shape), n_boot=200)
    assert better.challenger_wins and (better.by_cat["high"] < 0).all()
    worse = compare_models(df, truth + rng.normal(0, 0.3, truth.shape), noisy, n_boot=200)
    assert not worse.challenger_wins and worse.overall[1] > 0
