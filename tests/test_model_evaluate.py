import numpy as np
import pandas as pd
import pytest

from ninecat.model.dataset import STATS, target_line
from ninecat.model.evaluate import EVAL_CATS, backtest_table, mae_grid, weighted_errors


def test_weighted_errors_known_answer():
    mae, rmse, n = weighted_errors(pd.Series([1.0, 3.0, np.nan]), pd.Series([0.0, 0.0, 5.0]),
                                   pd.Series([3.0, 1.0, 1.0]))
    assert n == 2 and mae == pytest.approx((3 * 1 + 1 * 3) / 4) and rmse == pytest.approx(np.sqrt((3 + 9) / 4))
    assert weighted_errors(pd.Series([np.nan]), pd.Series([1.0]), pd.Series([1.0]))[2] == 0


def test_backtest_table_shape_and_a_perfect_prediction():
    df = pd.DataFrame({"k": [3, 3, 10], "y_G": 10.0, "y_MIN": [300.0, 200.0, 100.0], "y_GS": 0.0,
                       **{f"y_{s}": [50.0, 20.0, 10.0] for s in STATS}})
    perfect = target_line(df)
    off = perfect + 1.0
    table = backtest_table(df, {"perfect": perfect, "off": off})
    assert len(table) == 2 * 2 * len(EVAL_CATS)
    assert set(table.columns) == {"model", "k", "cat", "mae", "rmse", "n"}
    assert (table.loc[table["model"] == "perfect", "mae"] == 0).all()
    grid = mae_grid(table)
    assert list(grid.columns) == EVAL_CATS and grid.loc["off", "PTS"] == pytest.approx(1.0)


from ninecat.model.evaluate import coverage, fit_intervals, interval_bounds, spearman, value_rank_check


def checkpoints(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    talent = rng.uniform(0.2, 1.0, n)
    df = pd.DataFrame({"season": "2024-25", "k": np.where(np.arange(n) % 2 == 0, 5, 20), "y_G": 40.0,
                       "y_GS": 0.0, "nba_id": np.arange(n)})
    df["y_MIN"] = 40 * rng.uniform(15, 35, n)
    for s, w in {"PTS": 0.6, "REB": 0.25, "AST": 0.15, "STL": 0.03, "BLK": 0.03, "3PM": 0.06, "TO": 0.05,
                 "FGA": 0.5, "FTA": 0.15}.items():
        df[f"y_{s}"] = df["y_MIN"] * talent * w
    df["y_FGM"], df["y_FTM"] = df["y_FGA"] * rng.uniform(0.4, 0.55, n), df["y_FTA"] * rng.uniform(0.65, 0.9, n)
    return df


def test_spearman_known_answers():
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0) and spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0])))


def test_value_rank_check_rewards_the_truth():
    df = checkpoints()
    truth = target_line(df)
    rng = np.random.default_rng(1)
    noisy = truth * rng.uniform(0.5, 1.5, truth.shape)
    table = value_rank_check(df, {"truth": truth, "noisy": noisy}, top=100).set_index(["model", "k"])
    assert (table.loc["truth", "spearman"] > 0.999).all()
    assert (table.loc["noisy", "spearman"] < table.loc["truth", "spearman"]).all() and (table["n"] == 100).all()


def test_intervals_cover_about_eighty_percent_and_fall_back_by_checkpoint():
    df = checkpoints(1200)
    truth = target_line(df)
    pred = truth + np.random.default_rng(2).normal(0, 1, truth.shape) * truth.std().to_numpy()
    table = fit_intervals(df, pred)
    assert set(table["band"]) >= {-1}
    cov = coverage(df, pred, table)
    overall = cov[cov["k"] == 0].set_index("cat")["coverage"]
    assert overall.drop("VALUE").between(0.75, 0.85).all()
    lo, hi = interval_bounds(table, pd.Series([0.0, 7.0, 99.0]), pd.Series([np.nan, 30.0, 30.0]), "PTS")
    assert (lo < hi).all()                                         # unseen k and NaN minutes still get a range
