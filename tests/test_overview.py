from datetime import date

import numpy as np
import pandas as pd
import pytest

from ninecat.board import build_board
from ninecat.config import ROS_PRIOR_GAMES, categories
from ninecat.lines import LINE_STATS
from ninecat.overview import (
    game_values,
    games_played,
    overview,
    playoff_windows,
    preseason_snapshot,
    recent_minutes,
    ros_line,
    season_rate_line,
    snapshot_line,
    week_windows,
)
from tests.conftest import TEAMS

EIGHT = categories(exclude_to=True)
OPENING = date(2026, 10, 20)


def snapshot(players: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """A preseason snapshot: everyone's projection, scaled."""
    snap = players[["player_id", "name", *LINE_STATS, "GP"]].copy()
    snap[LINE_STATS] = snap[LINE_STATS] * scale
    return snap


# ---------------------------------------------------------------- snapshot choice

def test_preseason_snapshot_is_the_earliest_on_or_before_opening_night():
    days = [date(2026, 10, 23), date(2026, 10, 18), date(2026, 10, 12), date(2026, 10, 20)]
    pick = preseason_snapshot(days, OPENING)
    assert pick.day == date(2026, 10, 12) and pick.before_opening


def test_preseason_snapshot_falls_back_to_the_earliest_after_opening():
    pick = preseason_snapshot([date(2026, 10, 25), date(2026, 10, 23)], OPENING)
    assert pick.day == date(2026, 10, 23) and not pick.before_opening
    assert preseason_snapshot([date(2026, 10, 23)], None).before_opening is False


def test_no_snapshot_at_all():
    pick = preseason_snapshot([], OPENING)
    assert pick.day is None and not pick.before_opening


# ---------------------------------------------------------------- lines

def test_games_played_prefers_nba_then_espn_then_zero(players):
    p = players.head(4).copy()
    p["S_G"] = [5.0, np.nan, 0.0, np.nan]
    p["T_GP"] = [4.0, 3.0, 2.0, np.nan]
    assert games_played(p).tolist() == [5.0, 3.0, 2.0, 0.0]


def test_season_rate_line_is_full_availability_and_blank_without_games(players):
    line = season_rate_line(players, 82)
    has = players["S_G"].fillna(0) > 0
    assert (line.loc[has, "GP"] == 82).all()
    assert line.loc[has, "PTS"].equals(players.loc[has, "S_PTS"])
    assert line.loc[~has, "PTS"].isna().all()


def test_snapshot_line_matches_by_id_and_leaves_newcomers_blank(players):
    snap = snapshot(players.iloc[:50], scale=2.0)
    line = snapshot_line(players, snap)
    assert line.loc[0, "PTS"] == pytest.approx(players.loc[0, "PTS"] * 2)
    assert line.loc[60, LINE_STATS].isna().all()


@pytest.mark.parametrize("gp,weight", [(0.0, 0.0), (20.0, 0.5), (980.0, 0.98)])
def test_ros_line_weights_this_season_by_games(players, gp, weight):
    p = players.head(1).copy()
    p["S_G"] = gp
    p["S_PTS"] = 30.0
    pre = snapshot(p)
    pre["PTS"] = 10.0
    line = ros_line(p, pre, k=ROS_PRIOR_GAMES)
    expected = 10.0 if gp == 0 else weight * 30.0 + (1 - weight) * 10.0
    assert line.loc[0, "PTS"] == pytest.approx(expected)
    assert line.loc[0, "GP"] == p.loc[0, "GP"]            # availability from the projection


def test_ros_line_falls_back_to_the_current_projection(players):
    p = players.head(2).copy()
    p["S_G"] = 0.0
    line = ros_line(p, None)
    assert line["PTS"].tolist() == pytest.approx(p["PTS"].tolist())


# ---------------------------------------------------------------- overview

@pytest.fixture
def board(players):
    return build_board(players, EIGHT)


def run(players, board, **kw):
    return overview(players, board, EIGHT, TEAMS, preseason=snapshot(players), **kw)


def test_so_far_is_rate_times_games_and_zero_without_games(players, board):
    ov = run(players, board)
    keyed = players.set_index("player_id")
    gp = board["player_id"].map(keyed["S_G"]).fillna(0)
    assert (ov.loc[gp == 0, "so_far"] == 0).all() and (ov.loc[gp == 0, "gp"] == 0).all()
    # doubling everyone's games doubles everyone's so far
    more = players.assign(S_G=players["S_G"] * 2)
    ov2 = run(more, board)
    assert ov2["so_far"].to_numpy() == pytest.approx(2 * ov["so_far"].to_numpy())


def test_per_category_so_far_sums_to_the_total(players, board):
    ov = run(players, board)
    assert ov[[f"sf_{c}" for c in EIGHT]].sum(axis=1).to_numpy() == pytest.approx(ov["so_far"].to_numpy())
    punted = run(players, board, punt=["AST", "FT%"])
    scored = [c for c in EIGHT if c not in ("AST", "FT%")]
    assert punted[[f"sf_{c}" for c in scored]].sum(axis=1).to_numpy() == pytest.approx(punted["so_far"].to_numpy())
    assert punted["sf_AST"].abs().sum() > 0      # still reported, just not counted


def test_ranks_and_rank_change(players, board):
    ov = run(players, board)
    top = ov["so_far"].idxmax()
    assert ov.loc[top, "so_far_rank"] == 1 and ov["pre_rank"].min() == 1
    assert (ov["rank_delta"] == ov["pre_rank"] - ov["so_far_rank"]).all()


def test_ros_scales_with_remaining_games(players, board):
    full = run(players, board)
    half = run(players, board, remaining={t: 41 for t in players["pro_team"].unique()})
    both = full["ros"].notna()
    assert half.loc[both, "ros"].to_numpy() == pytest.approx(full.loc[both, "ros"].to_numpy() / 2)


def test_pre_is_blank_for_players_missing_from_the_snapshot(players, board):
    ov = overview(players, board, EIGHT, TEAMS, preseason=snapshot(players.iloc[:40]))
    missing = ~board["player_id"].isin(players["player_id"].iloc[:40])
    assert ov.loc[missing, "pre"].isna().all() and ov.loc[~missing, "pre"].notna().any()


# ---------------------------------------------------------------- game logs and schedule windows

LOGS = pd.DataFrame({
    "name_key": ["a", "a", "a", "b"],
    "date": ["2026-10-01", "2026-10-20", "2026-10-25", "2026-10-24"],
    "MIN": [10.0, 30.0, 34.0, 20.0],
    "PTS": [5.0, 20.0, 30.0, 10.0], "REB": 4.0, "AST": 3.0, "STL": 1.0, "BLK": 0.5, "3PM": 1.0, "TO": 2.0,
    "FGM": [2.0, 8.0, 11.0, 4.0], "FGA": 10.0, "FTM": 2.0, "FTA": 2.0,
})


def test_recent_minutes_window():
    m = recent_minutes(LOGS, date(2026, 10, 26), 14)
    assert m.to_dict() == {"a": 32.0, "b": 20.0}
    assert recent_minutes(pd.DataFrame(), date(2026, 10, 26), 14).empty


def test_game_values_last_games_oldest_first(board):
    vals = game_values(LOGS, board, EIGHT, last=2)
    assert len(vals["a"]) == 2 and vals["a"][1] > vals["a"][0]     # the 30-point game scores higher
    assert len(vals["b"]) == 1


CAL = pd.DataFrame({"period": [1, 2, 3], "start": ["2026-10-20", "2026-10-26", "2027-03-08"],
                    "end": ["2026-10-25", "2026-11-01", "2027-03-21"], "playoff": [False, False, True]})


def test_week_windows_use_matchup_periods_then_calendar_weeks():
    this, nxt = week_windows(CAL, 1, date(2026, 10, 23))
    assert this == (date(2026, 10, 20), date(2026, 10, 25)) and nxt == (date(2026, 10, 26), date(2026, 11, 1))
    this, nxt = week_windows(pd.DataFrame(), None, date(2026, 10, 28))   # a Wednesday
    assert this == (date(2026, 10, 26), date(2026, 11, 1)) and nxt == (date(2026, 11, 2), date(2026, 11, 8))


def test_playoff_windows():
    assert playoff_windows(CAL) == [(date(2027, 3, 8), date(2027, 3, 21))]
    assert playoff_windows(pd.DataFrame()) == []
