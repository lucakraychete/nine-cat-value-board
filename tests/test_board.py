import numpy as np
import pandas as pd
import pytest

from ninecat.board import (
    advanced_zscores,
    build_board,
    fantasy_tier,
    market_ranks,
    role_tier,
)
from ninecat.config import categories
from ninecat.lines import per36, stat_line
from ninecat.valuation import value_players

EIGHT = categories(exclude_to=True)


@pytest.mark.parametrize("rank,tier", [(1, "Superstar"), (10, "Superstar"), (11, "Star"), (30, "Star"),
                                       (70, "Starter"), (110, "Rotation"), (150, "Bench"), (151, "Waiver")])
def test_fantasy_tiers(rank, tier):
    assert fantasy_tier(rank) == tier


def test_role_tiers_need_usage_for_franchise_when_known():
    assert role_tier(34, 30) == "Franchise"
    assert role_tier(34, None) == "Franchise"
    assert role_tier(34, 22) == "Starter"
    assert role_tier(29, 30) == "Starter"
    assert [role_tier(m) for m in (23, 16, 15.9)] == ["Rotation", "Bench", "Fringe"]


def test_market_ranks_order_by_adp_and_share_last_slot():
    rank = pd.Series([1, 2, 3, 4])
    adp = pd.Series([12.0, np.nan, 3.0, 0.0])
    out = market_ranks(rank, adp)
    assert out["mkt_rank"].tolist() == [2, 3, 1, 3]
    assert out["gap"].tolist() == [1, 1, -2, -1]


def test_advanced_zscores_use_top_of_board_and_negate_turnovers():
    board = pd.DataFrame({"rank": [1, 2, 3, 4], "adv_usg": [30.0, 20.0, np.nan, 40.0],
                          "adv_tov": [10.0, 20.0, 15.0, 30.0]})
    z = advanced_zscores(board, pool_size=3)
    assert z["az_usg"].iloc[:2].tolist() == pytest.approx([1.0, -1.0])
    assert np.isnan(z["az_usg"].iloc[2])
    assert z["az_tov"].iloc[0] > 0 > z["az_tov"].iloc[1]
    assert z["az_bpm"].isna().all()


def test_build_board_drops_players_without_a_line_or_minutes(players):
    players.loc[0, "MIN"] = 9.0
    players.loc[1, ["stat_source"]] = None
    players.loc[1, ["PTS", "MIN"]] = 0.0
    players.loc[1, "L_G"] = np.nan
    board = build_board(players, EIGHT)
    assert 1000 not in board["player_id"].values and 1001 not in board["player_id"].values
    assert board["rank"].tolist() == list(range(1, len(board) + 1))


def test_build_board_value_matches_value_players_on_the_line(players):
    board = build_board(players, EIGHT, source="proj", pool_size=40).set_index("player_id")
    line = stat_line(players, "proj")
    direct = value_players(line[line["MIN"] >= 10], EIGHT, pool_size=40).set_index("player_id")
    pd.testing.assert_series_equal(board["value"], direct.loc[board.index, "value"])


def test_per36_layer_only_for_15_minutes_and_without_games_discount(players):
    board = build_board(players, EIGHT, punt=["AST"], pool_size=40).set_index("player_id")
    line = stat_line(players, "blend")
    line = line[line["MIN"] >= 10]
    rate = value_players(per36(line[line["MIN"] >= 15]), EIGHT, punt=["AST"], pool_size=40,
                         games_discount=False).set_index("player_id")
    assert board["p36_rank"].notna().sum() == len(rate)
    pd.testing.assert_series_equal(board.loc[rate.index, "p36_value"], rate["value"], check_names=False)
    assert board.loc[board["MIN"] < 15, "p36_rank"].isna().all()


def test_per100_differs_from_per36_by_pace(players):
    flat = build_board(players, EIGHT, pace={})
    paced = build_board(players, EIGHT, pace={"DEN": 108.0, "BOS": 94.0})
    assert not np.allclose(flat["p100_value"].dropna(), paced["p100_value"].dropna())


def test_minutes_change_and_tiers(players):
    board = build_board(players, EIGHT).set_index("player_id")
    p = players.set_index("player_id")
    some = board.index[board["d_min"].notna()][0]
    assert board.loc[some, "d_min"] == pytest.approx(p.loc[some, "MIN"] - p.loc[some, "L_MIN"])
    assert board.loc[some, "proj_MIN"] == pytest.approx(p.loc[some, "MIN"])
    assert board["d_min"][board["L_MIN"].isna()].isna().all()
    assert (board["fantasy_tier"] == board["rank"].map(fantasy_tier)).all()


def test_last_season_source_values_on_last_season_games(players):
    board = build_board(players, EIGHT, source="last").set_index("player_id")
    p = players.set_index("player_id")
    assert (board["GP"] == p.loc[board.index, "L_G"]).all()
    assert board.index.isin(p.index[p["L_G"].isna()]).sum() == 0


def test_market_gap_is_market_minus_value_rank(players):
    board = build_board(players, EIGHT)
    assert (board["gap"] == board["mkt_rank"] - board["rank"]).all()
    undrafted = board["ADP"].isna()
    assert (board.loc[undrafted, "mkt_rank"] == (~undrafted).sum() + 1).all()


def test_build_board_does_not_mutate_input(players):
    before = players.copy()
    build_board(players, EIGHT, punt=["PTS"], source="blend")
    pd.testing.assert_frame_equal(players, before)


def test_no_minutes_column_skips_minute_filter(players):
    old = players.drop(columns=["MIN", "L_MIN"])
    board = build_board(old, EIGHT, source="proj")
    assert len(board) == len(players)
