import numpy as np
import pandas as pd
import pytest

from ninecat.model.games import (
    AvailabilityModel,
    actual_availability,
    prior_availability,
    projected_games,
    so_far_availability,
)


def rows(**cols) -> pd.DataFrame:
    base = {"cur_G": [10.0], "team_games": [12.0], "p1_G": [60.0], "p1_team_games": [82.0], "p2_G": [72.0],
            "p2_team_games": [72.0], "y_G": [50.0], "team_left": [70.0]}
    base.update(cols)
    return pd.DataFrame(base)


def test_availability_uses_each_seasons_own_length():
    df = rows()
    assert prior_availability(df)[0] == pytest.approx((60 + 72) / (82 + 72))   # the 72-game season counts fully
    assert so_far_availability(df)[0] == pytest.approx(10 / 12)
    assert actual_availability(df)[0] == pytest.approx(50 / 70)
    assert np.isnan(prior_availability(rows(p1_G=[np.nan], p2_G=[np.nan]))[0])


def test_availability_model_shrinks_and_fills_newcomers():
    rng = np.random.default_rng(0)
    n = 3000
    true = rng.uniform(0.5, 1.0, n)
    team = np.full(n, 20.0)
    df = pd.DataFrame({"cur_G": rng.binomial(20, true).astype(float), "team_games": team,
                       "p1_G": np.where(np.arange(n) % 10 == 0, np.nan, true * 82), "p1_team_games": 82.0,
                       "p2_G": np.nan, "p2_team_games": 82.0, "team_left": 60.0})
    df["y_G"] = rng.binomial(60, true).astype(float)
    m = AvailabilityModel.fit(df)
    assert m.k > 0 and 0.5 < m.newcomer < 1.0
    pred = m.predict(df)
    assert pred.between(0, 1).all()
    err_model = (pred * 60 - df["y_G"]).abs().mean()
    err_so_far = (so_far_availability(df) * 60 - df["y_G"]).abs().mean()
    assert err_model < err_so_far


def test_projected_games_count_the_injury():
    avail = pd.Series([0.9, 0.9, 0.9, 0.9])
    remaining = pd.Series([60.0, 60.0, 60.0, 0.0])
    out = pd.Series([False, True, True, False])
    before = pd.Series([np.nan, 10.0, np.nan, np.nan])
    week = pd.Series([3.0, 3.0, 3.0, 0.0])
    got = projected_games(avail, remaining, out, before, week)
    assert got.tolist() == pytest.approx([54.0, 45.0, 51.3, 0.0])   # healthy; back after 10; OUT, no date: a week
