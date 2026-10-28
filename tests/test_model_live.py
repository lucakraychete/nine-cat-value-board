from datetime import date

import numpy as np
import pandas as pd
import pytest

from ninecat.model.live import actual_since, brier, calibration, matchup_probabilities, score_snapshots, update_matchup_log
from tests.conftest import TEAMS

STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]


def logs() -> pd.DataFrame:
    rows = []
    for d in range(1, 9):
        for who, pts in (("a", 20.0), ("b", 10.0)):
            rows.append({"name_key": who, "date": f"2026-11-{d:02d}", "MIN": 30.0, **{s: 1.0 for s in STATS[1:]},
                         "PTS": pts, "FGM": 5.0, "FGA": 10.0, "FTM": 4.0, "FTA": 5.0})
    return pd.DataFrame(rows)


def snapshot(kind: str, pts: dict[str, float]) -> pd.DataFrame:
    if kind == "model":
        return pd.DataFrame({"name_key": list(pts), **{f"M_{s}": 1.0 for s in STATS}, "M_MIN": 30.0,
                             "M_PTS": list(pts.values()), "M_FGM": 5.0, "M_FGA": 10.0, "M_FTM": 4.0, "M_FTA": 5.0})
    return pd.DataFrame({"name": [k.upper() for k in pts], **{s: 1.0 for s in STATS}, "MIN": 30.0,
                         "PTS": list(pts.values()), "FGM": 5.0, "FGA": 10.0, "FTM": 4.0, "FTA": 5.0})


def test_actual_since_counts_only_later_games():
    a = actual_since(logs(), date(2026, 11, 5))
    assert a.loc["a", "G"] == 3 and a.loc["a", "PTS"] == 20 and a.loc["a", "FG%"] == pytest.approx(0.5)
    assert actual_since(logs(), date(2026, 11, 6)).empty      # only 2 games after: not enough


def test_snapshots_are_scored_on_common_players():
    snaps = [("model", date(2026, 11, 1), snapshot("model", {"a": 22.0, "b": 10.0})),
             ("espn", date(2026, 11, 1), snapshot("espn", {"a": 26.0}))]
    table = score_snapshots(snaps, logs()).set_index("source")
    assert (table["players"] == 1).all()                       # only "a" is in both
    assert table.loc["model", "PTS"] == pytest.approx(2.0) and table.loc["espn", "PTS"] == pytest.approx(6.0)
    alone = score_snapshots(snaps[:1], logs())
    assert alone.loc[0, "players"] == 2 and alone.loc[0, "PTS"] == pytest.approx(1.0)


def live(period: int, wins: tuple[int, int]) -> pd.DataFrame:
    return pd.DataFrame({"period": period, "team_id": [1, 2], "opponent_id": [2, 1],
                         "wins": [wins[0], wins[1]], "losses": [wins[1], wins[0]], "ties": 0})


def test_matchup_log_records_once_then_finalizes():
    log = update_matchup_log(None, live(3, (2, 1)), {1: 0.7, 2: 0.3}, 3, date(2026, 11, 2), "2026-11-01T10:00:00")
    assert log["p_win"].tolist() == [0.7, 0.3] and log["pre_week"].all() and not log["final"].any()
    log = update_matchup_log(log, live(3, (5, 3)), {1: 0.1, 2: 0.9}, 3, date(2026, 11, 2), "2026-11-04T10:00:00")
    assert log["p_win"].tolist() == [0.7, 0.3]                 # the first sighting's chance is kept
    assert log["wins"].tolist() == [5, 3]
    log = update_matchup_log(log, live(4, (0, 0)), {}, 4, date(2026, 11, 9), "2026-11-10T10:00:00")
    done = log[log["period"] == 3]
    assert done["final"].all() and done["result"].tolist() == [1.0, 0.0]
    assert not log.loc[log["period"] == 4, "pre_week"].any()   # first seen after the week started
    score, n = brier(log)
    assert n == 2 and score == pytest.approx(((0.7 - 1) ** 2 + (0.3 - 0) ** 2) / 2)
    cal = calibration(log)
    assert cal["n"].sum() == 2 and set(cal["won"]) == {0.0, 1.0}


def test_brier_with_nothing_final():
    assert brier(pd.DataFrame())[1] == 0 and calibration(None).empty


def test_matchup_probabilities_are_complementary(players):
    from ninecat.board import build_board
    from ninecat.config import categories
    cats = categories(exclude_to=True)
    board = build_board(players, cats)
    schedule = pd.DataFrame({"date": ["2026-11-03", "2026-11-04"], "game_id": ["g1", "g2"],
                             "home": ["DEN", "BOS"], "away": ["PHL", "LAL"]})
    lv = pd.DataFrame({"period": 3, "team_id": [1, 2], "opponent_id": [2, 1], "wins": 0, "losses": 0, "ties": 0})
    p = matchup_probabilities(board, cats, {1: TEAMS[0], 2: TEAMS[1]}, lv, date(2026, 11, 3), date(2026, 11, 9),
                              schedule)
    assert set(p) == {1, 2} and p[1] + p[2] == pytest.approx(1.0, abs=0.08) and 0 < p[1] < 1
    assert not np.isnan(p[1])
