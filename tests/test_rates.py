import numpy as np
import pandas as pd
import pytest

from ninecat.rates import per, possessions, rate_stats


def team(**kw):
    base = dict(MIN=3936.0, FGM=3400, FGA=7300, FG3M=1100, FG3A=3000, FTM=1450, FTA=1900, OREB=900,
                DREB=2700, REB=3600, AST=2100, STL=640, BLK=410, TOV=1150, PF=1600, PTS=9350,
                OPP_FGM=3400, OPP_FGA=7300, OPP_FG3M=1100, OPP_FG3A=3000, OPP_FTM=1450, OPP_FTA=1900,
                OPP_OREB=900, OPP_DREB=2700, OPP_REB=3600, OPP_AST=2100, OPP_STL=640, OPP_BLK=410,
                OPP_TOV=1150, OPP_PF=1600, OPP_PTS=9350, PACE=100.0)
    base.update(kw)
    return base


@pytest.fixture
def teams():
    return pd.DataFrame([team(), team(PACE=104.0, AST=2300, PTS=9600)], index=["AAA", "BBB"])


@pytest.fixture
def two_players():
    return pd.DataFrame([
        dict(team="AAA", MIN=2400.0, FGM=600, FGA=1250, FG3M=150, FTM=300, FTA=380, OREB=80, DREB=320,
             REB=400, AST=420, STL=90, BLK=40, TOV=210, PF=150, PTS=1650),
        dict(team="BBB", MIN=1500.0, FGM=250, FGA=480, FG3M=10, FTM=90, FTA=150, OREB=180, DREB=400,
             REB=580, AST=90, STL=40, BLK=120, TOV=80, PF=200, PTS=600),
    ])


def test_true_shooting(two_players, teams):
    r = rate_stats(two_players, teams)
    assert r["ts"][0] == pytest.approx(1650 / (2 * (1250 + 0.44 * 380)))


def test_usage_rate(two_players, teams):
    t = team()
    plays = 1250 + 0.44 * 380 + 210
    expected = 100 * plays * t["MIN"] / (2400 * (t["FGA"] + 0.44 * t["FTA"] + t["TOV"]))
    assert rate_stats(two_players, teams)["usg"][0] == pytest.approx(expected)


def test_assist_rebound_and_turnover_rates(two_players, teams):
    t = team()
    r = rate_stats(two_players, teams)
    assert r["ast"][0] == pytest.approx(100 * 420 / ((2400 / t["MIN"]) * t["FGM"] - 600))
    assert r["trb"][0] == pytest.approx(100 * 400 * t["MIN"] / (2400 * (t["REB"] + t["OPP_REB"])))
    assert r["tov"][0] == pytest.approx(100 * 210 / (1250 + 0.44 * 380 + 210))


def test_steal_and_block_rates(two_players, teams):
    t = team()
    r = rate_stats(two_players, teams)
    poss = possessions(pd.DataFrame([t])).iloc[0]
    assert r["stl"][0] == pytest.approx(100 * 90 * t["MIN"] / (2400 * poss))
    assert r["blk"][0] == pytest.approx(100 * 40 * t["MIN"] / (2400 * (t["OPP_FGA"] - t["OPP_FG3A"])))


def test_possessions_matches_bbr_formula():
    t = team()
    side = t["FGA"] + 0.4 * t["FTA"] - 1.07 * (t["OREB"] / (t["OREB"] + t["OPP_DREB"])) * (t["FGA"] - t["FGM"]) + t["TOV"]
    assert possessions(pd.DataFrame([t])).iloc[0] == pytest.approx(side)  # symmetric team


def test_per_is_15_on_average_weighted_by_minutes(two_players, teams):
    p = per(two_players, teams)
    assert np.average(p, weights=two_players["MIN"]) == pytest.approx(15.0)


def test_per_rewards_production(two_players, teams):
    better = two_players.copy()
    better.loc[1, ["PTS", "FGM", "REB"]] += [200, 100, 100]
    assert per(better, teams)[1] - per(better, teams)[0] > per(two_players, teams)[1] - per(two_players, teams)[0]


def test_no_minutes_gives_nan(two_players, teams):
    two_players.loc[1, "MIN"] = 0
    r = rate_stats(two_players, teams)
    assert r.loc[1].isna().all() and r.loc[0].notna().all()
