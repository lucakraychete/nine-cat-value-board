import numpy as np
import pandas as pd
import pytest

from ninecat.config import categories
from ninecat.valuation import (
    apply_games_discount,
    availability,
    category_zscores,
    percentage_impact,
    replacement_level,
    team_profile,
    value_players,
)

NINE = categories(exclude_to=False)
EIGHT = categories(exclude_to=True)


def make_player(name, **stats):
    base = {
        "name": name,
        "PTS": 15.0, "REB": 5.0, "AST": 3.0, "STL": 1.0, "BLK": 0.5,
        "3PM": 1.5, "TO": 2.0, "FGM": 6.0, "FGA": 13.0, "FTM": 2.0,
        "FTA": 2.5, "GP": 82.0, "fantasy_team": "FA",
    }
    base.update(stats)
    return base


@pytest.fixture
def random_league():
    """300 players with plausible, varied per-game lines."""
    rng = np.random.default_rng(7)
    n = 300
    fga = rng.uniform(3, 20, n)
    fta = rng.uniform(0.5, 8, n)
    df = pd.DataFrame({
        "name": [f"P{i}" for i in range(n)],
        "PTS": rng.uniform(4, 30, n),
        "REB": rng.uniform(1, 12, n),
        "AST": rng.uniform(0.5, 9, n),
        "STL": rng.uniform(0.2, 2, n),
        "BLK": rng.uniform(0.1, 2.5, n),
        "3PM": rng.uniform(0, 4, n),
        "TO": rng.uniform(0.5, 4, n),
        "FGA": fga,
        "FGM": fga * rng.uniform(0.38, 0.62, n),
        "FTA": fta,
        "FTM": fta * rng.uniform(0.55, 0.92, n),
        "GP": rng.integers(40, 83, n).astype(float),
        "fantasy_team": rng.choice(["A", "B", "C", "FA"], n),
    })
    return df


def test_categories_respect_exclude_to():
    assert len(NINE) == 9 and "TO" in NINE
    assert len(EIGHT) == 8 and "TO" not in EIGHT


def test_counting_zscores_are_standardized_over_pool():
    df = pd.DataFrame([make_player(str(i), PTS=float(i)) for i in range(10)])
    pool = pd.Series(True, index=df.index)
    z = category_zscores(df, ["PTS"], pool)["z_PTS"]
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=0) == pytest.approx(1.0)
    assert z.is_monotonic_increasing


def test_zscores_use_pool_stats_not_everyone():
    df = pd.DataFrame([make_player(str(i), PTS=float(i)) for i in range(10)])
    pool = pd.Series(df.index >= 5, index=df.index)  # PTS 5..9, mean 7
    z = category_zscores(df, ["PTS"], pool)["z_PTS"]
    assert z[7] == pytest.approx(0.0)
    assert z[0] < -2  # far below a pool he isn't part of


def test_turnovers_are_negative():
    df = pd.DataFrame([make_player("careful", TO=1.0), make_player("sloppy", TO=4.0)])
    z = category_zscores(df, ["TO"], pd.Series(True, index=df.index))["z_TO"]
    assert z[0] > 0 > z[1]


def test_constant_category_gives_zero_not_nan():
    df = pd.DataFrame([make_player(str(i)) for i in range(5)])
    z = category_zscores(df, NINE, pd.Series(True, index=df.index))
    assert not z.isna().any().any()
    assert (z == 0).all().all()


def test_percentage_impact_weights_by_volume():
    df = pd.DataFrame([
        # League FG% across these three is exactly 50%.
        make_player("volume_bad", FGM=7.0, FGA=20.0),   # 35% on 20
        make_player("tiny_great", FGM=2.0, FGA=2.0),    # 100% on 2
        make_player("volume_good", FGM=12.0, FGA=20.0),  # 60% on 20
    ])
    pool = pd.Series(True, index=df.index)
    impact = percentage_impact(df, "FG%", pool)
    assert impact.tolist() == pytest.approx([-3.0, 1.0, 2.0])
    # 100% on 2 shots helps less than 60% on 20.
    z = category_zscores(df, ["FG%"], pool)["z_FG%"]
    assert z[2] > z[1] > 0 > z[0]


def test_league_average_shooter_scores_zero():
    df = pd.DataFrame([
        make_player("a", FTM=9.0, FTA=10.0),
        make_player("b", FTM=5.0, FTA=10.0),
        make_player("avg", FTM=3.5, FTA=5.0),  # 70%, same as the pool
    ])
    z = category_zscores(df, ["FT%"], pd.Series(True, index=df.index))["z_FT%"]
    assert z[2] == pytest.approx(0.0)


def test_zero_attempts_is_handled():
    df = pd.DataFrame([make_player("a", FTA=0.0, FTM=0.0), make_player("b")])
    out = value_players(df, NINE, pool_size=2)
    assert out["z_FT%"].notna().all()
    assert np.isnan(out.loc[out["name"] == "a", "FT%"].iloc[0])


def test_exclude_to_drops_turnover_column(random_league):
    out = value_players(random_league, EIGHT)
    assert "z_TO" not in out.columns
    assert "z_TO" in value_players(random_league, NINE).columns


def test_punting_removes_category_from_total(random_league):
    full = value_players(random_league, NINE).set_index("name")
    punted = value_players(random_league, NINE, punt=["FT%", "TO"]).set_index("name")
    expected = full["z_total"] - full["z_FT%"] - full["z_TO"]
    pd.testing.assert_series_equal(
        punted["z_total"], expected.loc[punted.index], check_names=False
    )


def test_punting_does_not_change_the_pool(random_league):
    full = value_players(random_league, NINE).set_index("name")
    punted = value_players(random_league, NINE, punt=["AST"]).set_index("name")
    assert set(full.index[full["in_pool"]]) == set(punted.index[punted["in_pool"]])


def test_punting_unknown_category_raises(random_league):
    with pytest.raises(ValueError):
        value_players(random_league, EIGHT, punt=["TO"])


def test_punting_changes_rankings_toward_punt_profile():
    df = pd.DataFrame(
        [make_player(f"filler{i}", PTS=10.0 + i % 5, AST=2.0 + i % 3) for i in range(20)]
        + [
            make_player("big", REB=13.0, BLK=2.5, AST=1.0, FTM=2.0, FTA=4.0),
            make_player("guard", AST=10.0, REB=3.0, BLK=0.1, FTM=4.5, FTA=5.0),
        ]
    )
    ranks = value_players(df, EIGHT, pool_size=10).set_index("name")["rank"]
    punt_ranks = value_players(df, EIGHT, punt=["AST", "FT%"], pool_size=10).set_index(
        "name"
    )["rank"]
    assert punt_ranks["big"] < punt_ranks["guard"]
    assert punt_ranks["big"] <= ranks["big"]


def test_availability_is_clipped_and_defaults_missing_games():
    gp = pd.Series([41.0, 82.0, 90.0])
    assert availability(gp).tolist() == [0.5, 1.0, 1.0]
    assert availability(pd.Series([0.0, np.nan])).tolist() == [70 / 82, 70 / 82]


def test_replacement_level_is_last_player_in_pool():
    per_game = pd.Series([5.0, -1.0, 3.0, 0.5])
    assert replacement_level(per_game, 3) == 0.5
    assert replacement_level(per_game, 10) == -1.0  # pool larger than the list


def test_games_discount_scales_value_above_replacement():
    per_game = pd.Series([4.0, 1.0, -2.0])
    half = pd.Series([0.5, 0.5, 0.5])
    # Replacement 1: a star keeps it plus half his edge; a player below
    # replacement moves halfway up toward it.
    assert apply_games_discount(per_game, half, 1.0).tolist() == [2.5, 1.0, -0.5]
    assert apply_games_discount(per_game, half * 2, 1.0).tolist() == [4.0, 1.0, -2.0]


def test_games_missed_halves_edge_over_replacement():
    df = pd.DataFrame(
        [make_player(f"filler{i}", PTS=8.0 + i) for i in range(10)]
        + [
            make_player("healthy", PTS=30.0, GP=82.0),
            make_player("hurt", PTS=30.0, GP=41.0),
        ]
    )
    out = value_players(df, EIGHT, pool_size=12).set_index("name")
    repl = replacement_level(out["z_total"], 12)
    assert out.loc["hurt", "z_total"] == pytest.approx(out.loc["healthy", "z_total"])
    assert out.loc["healthy", "value"] == pytest.approx(out.loc["healthy", "z_total"])
    assert out.loc["hurt", "value"] - repl == pytest.approx(
        (out.loc["healthy", "value"] - repl) / 2
    )

    no_discount = value_players(df, EIGHT, pool_size=12, games_discount=False)
    no_discount = no_discount.set_index("name")
    assert no_discount.loc["hurt", "value"] == pytest.approx(
        no_discount.loc["healthy", "value"]
    )


def test_replacement_player_is_unchanged_by_missed_games():
    df = pd.DataFrame([make_player(f"p{i}", PTS=10.0 + i, GP=41.0) for i in range(10)])
    out = value_players(df, EIGHT, pool_size=5)
    repl = replacement_level(out["z_total"], 5)
    last_in_pool = out[out["z_total"] == repl].iloc[0]
    assert last_in_pool["value"] == pytest.approx(repl)


def test_missing_gp_is_treated_as_70_games(random_league):
    random_league["GP"] = 0.0
    out = value_players(random_league, NINE)
    assert out["availability"].eq(round(70 / 82, 3)).all()
    repl = replacement_level(out["z_total"], 130)
    expected = repl + (out["z_total"] - repl) * 70 / 82
    pd.testing.assert_series_equal(out["value"], expected, check_names=False)


def test_punts_move_replacement_level_but_not_the_pool(random_league):
    full = value_players(random_league, NINE)
    punted = value_players(random_league, NINE, punt=["AST", "PTS"])
    assert set(full.loc[full["in_pool"], "name"]) == set(punted.loc[punted["in_pool"], "name"])
    assert replacement_level(punted["z_total"], 130) != replacement_level(full["z_total"], 130)


def test_pool_is_top_n_and_a_fixed_point(random_league):
    out = value_players(random_league, NINE, pool_size=130)
    assert out["in_pool"].sum() == 130
    # The pool is exactly the 130 best per-game values (before the discount)...
    by_per_game = out.sort_values("z_total", ascending=False)
    assert by_per_game["in_pool"].tolist() == [True] * 130 + [False] * (len(out) - 130)
    # ...and pool z-scores are centered, i.e. stats came from the pool.
    pool = out[out["in_pool"]]
    for cat in ["PTS", "REB", "AST"]:
        assert pool[f"z_{cat}"].mean() == pytest.approx(0.0, abs=1e-9)
        assert pool[f"z_{cat}"].std(ddof=0) == pytest.approx(1.0)


def test_iterative_pool_differs_from_single_pass(random_league):
    out = value_players(random_league, NINE, pool_size=130)
    single = value_players(random_league, NINE, pool_size=len(random_league))
    # Against a stronger pool, the average player looks worse.
    assert out["z_total"].mean() < single["z_total"].mean()


def test_pool_larger_than_player_count():
    df = pd.DataFrame([make_player(str(i), PTS=float(i)) for i in range(5)])
    out = value_players(df, EIGHT, pool_size=130)
    assert out["in_pool"].all()


def test_output_is_ranked(random_league):
    out = value_players(random_league, NINE)
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert out["value"].is_monotonic_decreasing
    assert len(out) == len(random_league)


def test_input_frame_is_not_mutated(random_league):
    before = random_league.copy()
    value_players(random_league, NINE, punt=["PTS"])
    pd.testing.assert_frame_equal(random_league, before)


def test_team_profile_sums_zscores(random_league):
    out = value_players(random_league, EIGHT)
    profile = team_profile(out, EIGHT)
    assert list(profile.columns) == [*EIGHT, "total"]
    team_a = out[out["fantasy_team"] == "A"]
    assert profile.loc["A", "REB"] == pytest.approx(team_a["z_REB"].sum())
    assert profile["total"].is_monotonic_decreasing
