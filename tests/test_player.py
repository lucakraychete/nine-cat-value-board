import numpy as np
import pandas as pd
import pytest

from ninecat.board import build_board
from ninecat.config import categories
from ninecat.player import (
    category_profile,
    comparables,
    efficiency_items,
    minutes_history,
    minutes_trend,
    pct3,
    percentile,
    position_group,
    primary_position,
    rate_items,
    rolling_minutes,
    season_so_far,
    stat_lines,
)

EIGHT = categories(exclude_to=True)
SEASONS = {"current": "2026-27", "last": "2025-26", "prev": "2024-25"}
LABELS = {"proj": "proj", "last": "last", "board": "board", "per100": "per100"}


@pytest.fixture
def board(players):
    return build_board(players, EIGHT)


def test_percentile_is_share_strictly_below():
    assert percentile(3, [1, 2, 3, 4]) == 50.0
    assert percentile(0, [1, 2]) == 0.0 and percentile(9, [1, 2]) == 100.0
    assert percentile(np.nan, [1, 2]) == 50.0 and percentile(1, [np.nan]) == 50.0


def test_position_group_uses_eligibility(board):
    group = position_group(board, "PF")
    assert group["eligible"].str.split("/").apply(lambda s: "PF" in s).all()
    assert group["rank"].is_monotonic_increasing


def test_category_profile(board):
    idx = board.index[0]
    prof = category_profile(board, idx, EIGHT)
    assert prof["cat"].tolist() == EIGHT
    assert prof["pct"].between(0, 100).all() and prof["pos_avg_pct"].between(0, 100).all()
    group = position_group(board, primary_position(board.loc[idx]))
    pts_rank = (group["z_PTS"] > board.loc[idx, "z_PTS"]).sum() + 1
    assert prof.set_index("cat").loc["PTS", "pos_rank"] == pts_rank


def test_efficiency_items_rank_and_invert_turnovers(board):
    idx = board.index[0]
    items = {i.label: i for i in efficiency_items(board, idx)}
    assert items["Fantasy value / game"].rank.startswith("#1 ·")
    assert items["Fantasy value / game"].pct_all == 100 * (len(board[board["rank"] <= 200]) - 1) / len(board[board["rank"] <= 200])
    tov = board["adv_tov"]
    low = tov.idxmin()
    assert {i.label: i for i in efficiency_items(board, low)}["Turnover rate"].pct_all == 100.0


def test_rate_items_cover_the_share_stats(board):
    labels = [i.label for i in rate_items(board, board.index[0])]
    assert labels == ["AST%", "TRB%", "STL%", "BLK%"]


def test_missing_advanced_shows_a_dash(board):
    rookie = board.index[board["adv_usg"].isna()][0]
    item = {i.label: i for i in efficiency_items(board, rookie)}["Usage rate"]
    assert item.value == "–" and item.pct_all is None and item.rank == ""


def test_comparables_are_nearest_in_value_at_position(board):
    idx = board.index[5]
    comps = comparables(board, idx, n=6)
    assert idx in comps.index and len(comps) == 7
    assert comps["value"].is_monotonic_decreasing
    group = position_group(board, primary_position(board.loc[idx])).drop(index=idx)
    worst = (comps.drop(index=idx)["value"] - board.loc[idx, "value"]).abs().max()
    assert ((group["value"] - board.loc[idx, "value"]).abs() < worst).sum() <= 6


def test_minutes_history_and_trend(board):
    row = board[board["prev_MIN"].notna() & board["S_G"].isna()].iloc[0]
    hist = minutes_history(row, SEASONS)
    assert [h[0] for h in hist] == ["2024-25", "2025-26", "2026-27 proj"]
    assert hist[-1][2] and not hist[0][2]
    now = board[board["prev_MIN"].notna() & (board["S_G"] > 0)].iloc[0]
    hist = minutes_history(now, SEASONS)
    assert [h[0] for h in hist] == ["2024-25", "2025-26", "2026-27 so far", "2026-27 proj"]
    assert hist[2][1] == pytest.approx(now["S_MIN"]) and not hist[2][2]
    assert "more minutes" in minutes_trend(3.0) and "fewer" in minutes_trend(-2.5)
    assert "same" in minutes_trend(0.5) and "No prior" in minutes_trend(np.nan)


def test_rolling_minutes_window():
    logs = pd.DataFrame({"date": [f"2026-01-{d:02d}" for d in range(30, 0, -1)], "MIN": range(30)})
    g = rolling_minutes(logs, games=12)
    assert len(g) == 12 and g["date"].is_monotonic_increasing
    assert g["r5"].iloc[-1] == pytest.approx(np.mean(g["MIN"].iloc[-5:]))
    assert g["r10"].iloc[0] == g["MIN"].iloc[0]


def test_stat_lines(players, board):
    idx = board.index[(board["MIN"] >= 15) & (board["L_G"] > 0)][0]
    row = board.loc[idx]
    raw = players[players["player_id"] == row["player_id"]]
    lines = stat_lines(raw, row, {"DEN": 100.0}, LABELS)
    assert lines["Line"].tolist() == ["proj", "last", "board", "Per 36 minutes", "per100"]
    assert lines.loc[3, "PTS"] == pytest.approx(row["PTS"] * 36 / row["MIN"])
    assert lines.loc[2, "FG%"] == pytest.approx(row["FGM"] / row["FGA"])
    assert lines.loc[0, "G"] == row["GP"] and np.isnan(lines.loc[3, "G"])


def test_stat_lines_add_this_season_and_last_15_days(players, board):
    idx = board.index[(board["MIN"] >= 15) & (board["S_G"] > 0)][0]
    row = board.loc[idx]
    raw = players[players["player_id"] == row["player_id"]].assign(R15_GP=3.0, **{f"R15_{s}": 2.0 for s in (
        "MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA")})
    labels = {**LABELS, "season": "season", "recent": "l15"}
    lines = stat_lines(raw, row, {"DEN": 100.0}, labels).set_index("Line")
    assert lines.index[:3].tolist() == ["season", "l15", "proj"]
    assert lines.loc["season", "PTS"] == pytest.approx(raw["S_PTS"].iloc[0])
    assert lines.loc["season", "G"] == raw["S_G"].iloc[0]
    assert lines.loc["season", "FG%"] == pytest.approx(raw["S_FGM"].iloc[0] / raw["S_FGA"].iloc[0])
    assert (lines.loc["l15", "G"], lines.loc["l15", "FT%"]) == (3.0, 1.0)


def test_season_so_far_falls_back_to_espn_and_waits_for_a_game():
    row = pd.Series({"S_G": np.nan, "T_GP": 2.0, "T_PTS": 21.0, "T_MIN": 30.0})
    now = season_so_far(row)
    assert (now["PTS"], now["G"]) == (21.0, 2.0)
    assert season_so_far(pd.Series({"S_G": 0.0, "T_GP": np.nan})) is None


def test_pct3_formats_like_a_box_score():
    assert pct3(0.4567) == ".457" and pct3(1.0) == "1.000"
