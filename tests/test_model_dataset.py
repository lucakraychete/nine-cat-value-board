import numpy as np
import pandas as pd
import pytest

from ninecat.model.dataset import (
    PCT_STATS,
    RATE_STATS,
    STATS,
    build_dataset,
    checkpoints,
    per_game,
    season_totals,
    split_of,
    started,
    team_games_before,
    window_rates,
)


def make_logs(games: dict[int, int], season: str = "2024-25", team_of=lambda pid: "AAA") -> pd.DataFrame:
    """Player pid plays games[pid] games on consecutive days; game j's stats all equal j + 1."""
    rows = []
    for pid, n in games.items():
        for j in range(n):
            rows.append({"name_key": f"p{pid}", "nba_id": pid, "game_id": f"g{j}", "date": f"2024-11-{j + 1:02d}",
                         "team": team_of(pid), "MIN": 20.0 + j, "season": season,
                         **{s: float(j + 1) for s in STATS}})
    return pd.DataFrame(rows)


def test_checkpoint_features_use_only_games_before_the_cut():
    logs = make_logs({1: 30})
    cp = checkpoints(logs, ks=(10,)).iloc[0]
    assert cp["cur_PTS"] == sum(range(1, 11)) and cp["cur_MIN"] == sum(20.0 + j for j in range(10))
    assert cp["l5_PTS"] == sum(range(6, 11)) and cp["l10_PTS"] == cp["cur_PTS"]
    assert cp["y_PTS"] == sum(range(11, 31)) and cp["y_G"] == 20
    assert cp["cut_date"] == "2024-11-11"          # the first game after the cut
    # whatever happens from the cut on, no feature moves
    later = logs.copy()
    later.loc[later["game_id"].isin([f"g{j}" for j in range(10, 30)]), [*STATS, "MIN"]] *= 100
    again = checkpoints(later, ks=(10,)).iloc[0]
    features = [c for c in cp.index if c.startswith(("cur_", "l5_", "l10_"))]
    pd.testing.assert_series_equal(cp[features], again[features])
    assert again["y_PTS"] != cp["y_PTS"]


def test_short_windows_and_minimum_remaining_games():
    cp = checkpoints(make_logs({1: 3 + 5, 2: 3 + 4}), ks=(3,))
    assert cp["nba_id"].tolist() == [1]                     # player 2 has only 4 games left
    row = cp.iloc[0]
    assert row["l5_G"] == 3 and row["l5_PTS"] == row["cur_PTS"]


def test_started_is_the_top_five_in_minutes_per_team_game():
    logs = pd.DataFrame({"game_id": "g", "team": ["A"] * 7 + ["B"] * 2, "MIN": [30, 10, 35, 5, 25, 20, 40, 1, 2]})
    assert started(logs).tolist() == [1, 0, 1, 0, 1, 1, 1, 1, 1]


def test_team_games_before_each_day():
    logs = pd.DataFrame({"game_id": ["a", "a", "b", "b", "c", "c"], "team": ["X", "Y", "X", "Z", "Y", "Z"],
                         "date": ["d1", "d1", "d2", "d2", "d3", "d3"]})
    assert team_games_before(logs).to_dict() == {"d1": 0.0, "d2": 2 / 3, "d3": 4 / 3}


def test_priors_come_from_earlier_seasons_by_player():
    logs = {"2023-24": make_logs({1: 12}, "2023-24"), "2024-25": make_logs({1: 20, 2: 20}, "2024-25")}
    totals = {"2024-25": pd.DataFrame({"nba_id": [1, 2], "age": [25, 21]})}
    df = build_dataset(logs, totals, ["2024-25"], ks=(5,)).set_index("nba_id")
    prev = season_totals(logs["2023-24"]).loc[1]
    assert df.loc[1, "p1_G"] == 12 and df.loc[1, "p1_PTS"] == prev["PTS"]
    assert np.isnan(df.loc[2, "p1_G"]) and df[["p2_G"]].isna().all().all()
    assert df.loc[1, "age"] == 25 and df.loc[2, "age"] == 21
    assert split_of("2024-25") == "validate" and split_of("2017-18") == "train" and split_of("2016-17") == "other"


def test_rates_and_recombination_round_trip():
    df = pd.DataFrame({"w_G": [4.0, 2.0], "w_MIN": [120.0, 0.0], "w_GS": [4.0, 0.0],
                       **{f"w_{s}": [12.0, 0.0] for s in STATS}})
    df["w_FGA"], df["w_FGM"] = [24.0, 0.0], [12.0, 0.0]
    r = window_rates(df, "w")
    assert r.loc[0, "MPG"] == 30 and r.loc[0, "PTS"] == pytest.approx(0.1)
    assert r.loc[0, "FG%"] == pytest.approx(0.5) and np.isnan(r.loc[1, "FG%"]) and np.isnan(r.loc[1, "PTS"])
    line = per_game(r)
    for s in RATE_STATS:
        assert line.loc[0, s] == pytest.approx(df.loc[0, f"w_{s}"] / 4)
    for c, (made, att) in PCT_STATS.items():
        assert line.loc[0, made] == pytest.approx(df.loc[0, f"w_{made}"] / 4)
        assert line.loc[0, c] == pytest.approx(line.loc[0, made] / line.loc[0, att])
