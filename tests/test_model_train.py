"""The training pipeline end to end on synthetic seasons: report shape, the saved
model's live predictions, and the M_ columns the app reads."""

import json

import numpy as np
import pandas as pd
import pytest

from ninecat.lines import MODEL_PREFIX as M
from ninecat.model.dataset import STATS, build_dataset, split_of
from ninecat.model.evaluate import INTERVAL_CATS
from ninecat.model.predict import live_rows, with_games
from ninecat.model.train import train_and_report

FAST = {"max_iter": 20, "learning_rate": 0.3, "max_leaf_nodes": 7, "min_samples_leaf": 10,
        "early_stopping": False, "random_state": 0}
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2026)]


def season_logs(season: str, n_players: int = 60, games: int = 50, seed: int = 0) -> pd.DataFrame:
    """Each player has a steady talent and minutes, plus game-to-game noise; 6 teams."""
    rng = np.random.default_rng(seed)
    talent = np.random.default_rng(1).uniform(0.3, 0.9, n_players)
    mpg = np.random.default_rng(2).uniform(12, 36, n_players)
    rows = []
    for g in range(games):
        day = pd.Timestamp(f"{season[:4]}-11-01") + pd.Timedelta(days=g)
        for i in range(n_players):
            if rng.random() < 0.1:
                continue
            minutes = max(1.0, mpg[i] + rng.normal(0, 4))
            line = {s: max(0.0, minutes * talent[i] * w + rng.normal(0, 1)) for s, w in
                    {"PTS": 0.6, "REB": 0.25, "AST": 0.15, "STL": 0.03, "BLK": 0.03, "3PM": 0.06, "TO": 0.05,
                     "FGA": 0.5, "FTA": 0.15}.items()}
            line["FGM"], line["FTM"] = line["FGA"] * 0.47, line["FTA"] * 0.78
            rows.append({"name_key": f"p{i}", "nba_id": i, "game_id": f"{season}-{g}-{i % 6}", "date": day.date().isoformat(),
                         "team": f"T{i % 6}", "MIN": minutes, "season": season, **line})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    logs = {s: season_logs(s, seed=k) for k, s in enumerate(SEASONS)}
    totals = {s: pd.DataFrame({"nba_id": range(60), "age": np.linspace(20, 34, 60)}) for s in SEASONS}
    df = build_dataset(logs, totals, [s for s in SEASONS if split_of(s) != "other"])
    model, report = train_and_report(df, boost_params=FAST, boost_targets=["PTS"], n_boot=50)
    return model, report, logs


def test_report_shape(trained):
    _, report, _ = trained
    json.dumps(report)                                   # serializable as written to report.json
    for key in ("created", "seasons", "checkpoints", "chosen", "boosting_vs_shrinkage", "k", "k_units",
                "availability", "age_factors", "minutes_coef", "backtest", "value_rank", "intervals",
                "coverage_test", "test_verdict", "model_beats_blend"):
        assert key in report, key
    assert report["chosen"] in ("boosted", "shrinkage")
    test = pd.DataFrame(report["backtest"]["test"])
    assert {"Season to date", "Last season", "Blend 60/40", "Interim (app today)", "Shrinkage", "Boosted"} \
        == set(test["model"])
    assert set(test["k"]) == {3, 5, 10, 20, 40} and {"model", "k", "cat", "mae", "rmse", "n"} <= set(test.columns)
    assert set(pd.DataFrame(report["coverage_test"])["cat"]) == set(INTERVAL_CATS)
    assert set(report["k"]) == set(report["k_units"]) and all("games_at_30" in v for v in report["k"].values())
    assert set(report["test_verdict"]) == {"vs Blend 60/40", "vs Interim (app today)"}


def test_live_predictions_have_every_column_the_app_reads(trained):
    model, _, logs = trained
    current = logs["2025-26"]
    current = current[current["date"] < "2025-11-08"].assign(season="2026-27")
    rows = live_rows(current, logs["2025-26"], logs["2024-25"], {"p1": 25.0}, ["p0", "p1"])
    pred = model.predict(rows)
    for c in ["MIN", *STATS, "avail", "GP", "k"]:
        assert M + c in pred, c
    for c in INTERVAL_CATS:
        assert (pred[M + "lo_" + c] <= pred[M + "hi_" + c]).all(), c
    assert pred[M + "avail"].between(0, 1).all() and pred[M + "MIN"].between(0, 48).all()
    assert pred["name_key"].is_unique

    team = pd.Series(["T0"] * len(pred))
    out = pd.Series([True] + [False] * (len(pred) - 1))
    games = with_games(pred, team, {"T0": 40.0}, out, pd.Series(np.nan, index=pred.index),
                       pd.Series(3.0, index=pred.index))
    assert games.loc[1, M + "G"] == pytest.approx(40 * pred.loc[1, M + "avail"])
    assert games.loc[0, M + "G"] == pytest.approx(37 * pred.loc[0, M + "avail"])      # OUT: loses this week
    assert games.loc[0, M + "GP"] == pytest.approx(games.loc[0, M + "G"] / 40 * 82)


def test_live_rows_include_players_without_games_yet(trained):
    _, _, logs = trained
    current = logs["2025-26"]
    current = current[(current["date"] < "2025-11-05") & (current["nba_id"] != 7)].assign(season="2026-27")
    rows = live_rows(current, logs["2025-26"], logs["2024-25"], {}, ["p7"])
    row = rows.set_index("name_key").loc["p7"]
    assert row["cur_G"] == 0 and row["k"] == 0 and row["p1_G"] > 0
