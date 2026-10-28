import numpy as np
import pandas as pd
import pytest

from ninecat.lines import (
    BLEND_WEIGHT,
    attach_by_name,
    name_key,
    per36,
    per100,
    projection_line,
    stat_line,
)


def one(**cols):
    base = {"player_id": 1, "name": "A", "pro_team": "DEN", "stat_source": "projected",
            "MIN": 30.0, "PTS": 20.0, "REB": 5.0, "AST": 4.0, "STL": 1.0, "BLK": 0.5, "3PM": 2.0,
            "TO": 2.0, "FGM": 8.0, "FGA": 16.0, "FTM": 3.0, "FTA": 4.0, "GP": 70.0,
            "L_G": 60.0, "L_MIN": 25.0, "L_PTS": 10.0, "L_REB": 3.0, "L_AST": 2.0, "L_STL": 0.5,
            "L_BLK": 0.5, "L_3PM": 1.0, "L_TO": 1.0, "L_FGM": 4.0, "L_FGA": 10.0, "L_FTM": 1.0,
            "L_FTA": 2.0}
    base.update(cols)
    return pd.DataFrame([base])


def test_name_key_strips_accents_suffixes_and_punctuation():
    assert name_key("Nikola Jokić") == name_key("Nikola Jokic") == "nikola jokic"
    assert name_key("Jaren Jackson Jr.") == "jaren jackson"
    assert name_key("Jimmy Butler III") == "jimmy butler"
    assert name_key("D'Angelo  Russell") == "dangelo russell"
    assert name_key("P.J. Washington") == "pj washington"


def test_name_key_aliases_known_nicknames():
    assert name_key("Herbert Jones") == name_key("Herb Jones")


def test_attach_by_name_joins_by_name_without_clobbering():
    players = pd.DataFrame({"name": ["Nikola Jokić", "Rookie"], "PTS": [28.0, 10.0]})
    last = pd.DataFrame({"name_key": ["nikola jokic"], "L_PTS": [29.6], "PTS": [999.0]})
    out = attach_by_name(players, last)
    assert out["L_PTS"].tolist()[0] == 29.6 and np.isnan(out["L_PTS"].tolist()[1])
    assert out["PTS"].tolist() == [28.0, 10.0]
    assert attach_by_name(players, None).equals(players)


def test_projection_source_uses_espn_line():
    out = stat_line(one(), "proj")
    assert out["PTS"].iloc[0] == 20.0 and out["GP"].iloc[0] == 70.0 and out["has_line"].iloc[0]


def test_projection_falls_back_to_last_season_without_projection():
    out = stat_line(one(stat_source=None, PTS=0.0, MIN=0.0), "proj")
    assert out["PTS"].iloc[0] == 10.0 and out["GP"].iloc[0] == 60.0


def test_last_source_uses_last_season_games():
    out = stat_line(one(), "last")
    assert out["PTS"].iloc[0] == 10.0 and out["GP"].iloc[0] == 60.0 and out["MIN"].iloc[0] == 25.0


def test_last_source_without_last_season_has_no_line():
    out = stat_line(one(L_G=np.nan), "last")
    assert not out["has_line"].iloc[0] and np.isnan(out["PTS"].iloc[0])


def test_blend_mixes_60_40_and_keeps_projected_games():
    out = stat_line(one(), "blend").iloc[0]
    w = BLEND_WEIGHT
    assert out["PTS"] == pytest.approx(w * 20 + (1 - w) * 10)
    assert out["MIN"] == pytest.approx(w * 30 + (1 - w) * 25)
    assert out["GP"] == 70.0


def test_blend_mixes_makes_and_attempts_not_percentages():
    out = stat_line(one(), "blend").iloc[0]
    fgm, fga = 0.6 * 8 + 0.4 * 4, 0.6 * 16 + 0.4 * 10
    assert out["FGM"] / out["FGA"] == pytest.approx(fgm / fga)
    assert fgm / fga != pytest.approx(0.6 * 0.5 + 0.4 * 0.4)


def test_blend_ignores_small_last_season_samples():
    out = stat_line(one(L_G=14.0), "blend").iloc[0]
    assert out["PTS"] == 20.0
    assert stat_line(one(L_G=15.0), "blend").iloc[0]["PTS"] != 20.0


def test_blend_uses_last_season_when_no_projection():
    out = stat_line(one(stat_source=None, PTS=0.0, MIN=0.0), "blend").iloc[0]
    assert out["PTS"] == 10.0


def season_cols(prefix="S_", games=5.0, **kw):
    """This season's line under nba.com's S_ prefix (games in S_G) or ESPN's T_ (games in T_GP)."""
    base = dict(MIN=34.0, PTS=30.0, REB=8.0, AST=6.0, STL=2.0, BLK=1.0, TO=3.0, FGM=11.0, FGA=20.0,
                FTM=5.0, FTA=6.0, **{"3PM": 3.0})
    base.update(kw)
    cols = {prefix + k: v for k, v in base.items()}
    cols[prefix + ("G" if prefix == "S_" else "GP")] = games
    return cols


def test_season_source_uses_this_seasons_line_and_projected_games():
    out = stat_line(one(**season_cols()), "season").iloc[0]
    assert (out["PTS"], out["FGA"], out["3PM"], out["GP"]) == (30.0, 20.0, 3.0, 70.0)
    assert out["has_line"]


def test_season_source_falls_back_to_espn_then_to_the_blend():
    espn = stat_line(one(**season_cols(games=2.0), **season_cols("T_", 4.0, PTS=26.0)), "season").iloc[0]
    assert espn["PTS"] == 26.0
    blend = stat_line(one(), "blend").iloc[0]
    for few in (one(**season_cols(games=2.0)), one(**season_cols("T_", 2.0)), one()):
        out = stat_line(few, "season").iloc[0]
        assert out["PTS"] == pytest.approx(blend["PTS"]) and out["GP"] == blend["GP"]


def test_season_source_without_a_projection_keeps_its_own_line():
    out = stat_line(one(stat_source=None, PTS=0.0, MIN=0.0, GP=0.0, **season_cols(games=4.0)), "season").iloc[0]
    assert out["PTS"] == 30.0 and out["has_line"]


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        stat_line(one(), "vibes")


def test_missing_projected_attempts_are_estimated_from_last_season():
    line = projection_line(one(FGA=0.0, FGM=0.0)).iloc[0]
    # 10 FGA per 10 last-season points -> 1 FGA per point, at last season's 40%.
    assert line["FGA"] == pytest.approx(20.0)
    assert line["FGM"] == pytest.approx(8.0)


def test_stat_line_does_not_mutate_input():
    df = one()
    before = df.copy()
    stat_line(df, "blend")
    pd.testing.assert_frame_equal(df, before)


def test_per36_scales_counting_stats_and_keeps_percentages():
    line = stat_line(one(), "proj")
    out = per36(line).iloc[0]
    assert out["PTS"] == pytest.approx(20 * 36 / 30)
    assert out["FGM"] / out["FGA"] == pytest.approx(0.5)
    assert out["MIN"] == 30.0


def test_per100_uses_team_pace_and_league_default():
    line = stat_line(one(), "proj")
    poss = 104.0 * 30 / 48
    assert per100(line, {"DEN": 104.0}).iloc[0]["PTS"] == pytest.approx(20 * 100 / poss)
    assert per100(line, {}).iloc[0]["PTS"] == pytest.approx(20 * 100 / (100 * 30 / 48))


def test_zero_minutes_scale_to_nan_not_inf():
    line = stat_line(one(MIN=0.0), "proj")
    assert np.isnan(per36(line).iloc[0]["PTS"])
    assert np.isnan(per100(line, {}).iloc[0]["PTS"])


def test_model_source_reads_the_model_line_and_falls_back_to_the_blend(players):
    from tests.conftest import with_model
    m = with_model(players)
    m.loc[3, [c for c in m.columns if c.startswith("M_")]] = np.nan
    line = stat_line(m, "model")
    blend = stat_line(players, "blend")
    assert line.loc[0, "PTS"] == pytest.approx(players.loc[0, "PTS"] * 1.1) and line.loc[0, "GP"] == m.loc[0, "M_GP"]
    assert line.loc[3, "PTS"] == pytest.approx(blend.loc[3, "PTS"])
    assert stat_line(players, "model")["PTS"].equals(blend["PTS"])             # no model trained: the blend
