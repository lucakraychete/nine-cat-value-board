import numpy as np
import pandas as pd
import pytest

from ninecat.config import ROS_PRIOR_GAMES
from ninecat.model.baselines import (
    BLEND_PROJECTION,
    RookiePrior,
    blend,
    interim,
    last_season,
    mix,
    per_game_columns,
    preseason_prior,
    season_to_date,
)
from ninecat.model.dataset import STATS


def frame(**windows) -> pd.DataFrame:
    """One checkpoint row per entry; each window is (games, minutes, per-minute rate for every stat)."""
    n = len(next(iter(windows.values())))
    df = pd.DataFrame(index=range(n))
    for w in ("cur", "p1", "p2", "y", "l5", "l10"):
        spec = windows.get(w, [(np.nan, np.nan, np.nan)] * n)
        df[f"{w}_G"] = [g for g, _, _ in spec]
        df[f"{w}_MIN"] = [m for _, m, _ in spec]
        df[f"{w}_GS"] = 0.0
        for s in STATS:
            df[f"{w}_{s}"] = [m * r for _, m, r in spec]
        df[f"{w}_FGM"] = df[f"{w}_FGA"] / 2
        df[f"{w}_FTM"] = df[f"{w}_FTA"] * 0.8
    return df


def test_mix_weights_and_falls_back_to_the_known_side():
    a = pd.DataFrame({c: [10.0, np.nan] for c in per_game_columns()})
    b = pd.DataFrame({c: [20.0, 4.0] for c in per_game_columns()})
    a["FGM"], a["FGA"] = [4.0, np.nan], [10.0, np.nan]
    b["FGM"], b["FGA"] = [6.0, 1.0], [10.0, 2.0]
    m = mix(a, b, 0.25)
    assert m.loc[0, "PTS"] == pytest.approx(0.25 * 10 + 0.75 * 20) and m.loc[1, "PTS"] == 4
    assert m.loc[0, "FG%"] == pytest.approx((0.25 * 4 + 0.75 * 6) / 10)    # from mixed makes and attempts


def test_preseason_prior_weights_last_season_twice_and_uses_its_minutes():
    df = frame(cur=[(5, 100, 0.5)], p1=[(50, 1500, 0.6)], p2=[(60, 1200, 0.3)])
    prior = preseason_prior(df)
    expected = (2 * 1500 * 0.6 + 1200 * 0.3) / (2 * 1500 + 1200)
    assert prior.loc[0, "MIN"] == pytest.approx(30.0)
    assert prior.loc[0, "PTS"] == pytest.approx(30.0 * expected)


def test_rookies_get_the_pool_rate_at_their_minutes():
    train = frame(cur=[(10, 120, 0.2), (10, 320, 0.5)], y=[(40, 480, 0.25), (40, 1280, 0.6)])
    rookie = RookiePrior.fit(train)
    df = frame(cur=[(5, 60, 0.9), (5, 160, 0.1)])        # 12 and 32 minutes a game, no prior seasons
    prior = preseason_prior(df, rookie)
    assert prior.loc[0, "MIN"] == pytest.approx(12) and prior.loc[0, "PTS"] == pytest.approx(12 * 0.25)
    assert prior.loc[1, "PTS"] == pytest.approx(32 * 0.6)


def test_blend_is_sixty_forty_projection_and_last_season():
    df = frame(cur=[(5, 150, 0.5)], p1=[(50, 1500, 0.6)], p2=[(60, 1200, 0.3)])
    got = blend(df)
    assert got.loc[0, "PTS"] == pytest.approx(BLEND_PROJECTION * preseason_prior(df).loc[0, "PTS"]
                                              + (1 - BLEND_PROJECTION) * last_season(df).loc[0, "PTS"])


@pytest.mark.parametrize("g", [0.0, 20.0, 400.0])
def test_interim_is_the_apps_ros_line(g):
    df = frame(cur=[(g, 30 * g, 0.5)], p1=[(50, 1500, 0.6)], p2=[(60, 1200, 0.3)])
    w = g / (g + ROS_PRIOR_GAMES)
    got, prior, now = interim(df).loc[0, "PTS"], preseason_prior(df).loc[0, "PTS"], season_to_date(df).loc[0, "PTS"]
    assert got == pytest.approx(prior if g == 0 else w * now + (1 - w) * prior)


def test_last_season_falls_back_to_this_season():
    df = frame(cur=[(5, 150, 0.5), (5, 150, 0.5)], p1=[(50, 1500, 0.6), (np.nan, np.nan, np.nan)])
    got = last_season(df)
    assert got["PTS"].tolist() == pytest.approx([18.0, 15.0])
