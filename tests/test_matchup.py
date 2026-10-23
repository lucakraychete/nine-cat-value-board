import math

import numpy as np
import pandas as pd
import pytest

from ninecat.config import categories
from ninecat.matchup import (
    StatMatrix,
    Week,
    category_odds,
    change,
    corr_matrix,
    matchup,
    ncdf,
    outlook,
    slot_weights,
    win_probability,
)

EIGHT = categories(exclude_to=True)
NINE = categories(exclude_to=False)


def player(value, gp=82.0, **stats):
    base = {"value": value, "GP": gp, "PTS": 20.0, "REB": 5.0, "AST": 4.0, "STL": 1.0, "BLK": 0.5,
            "3PM": 2.0, "TO": 2.0, "FGM": 8.0, "FGA": 16.0, "FTM": 3.0, "FTA": 4.0}
    base.update(stats)
    return base


def test_ncdf_matches_the_error_function():
    for x in (-3, -1.2, -0.3, 0, 0.5, 1.96, 4):
        assert ncdf(x) == pytest.approx(0.5 * (1 + math.erf(x / math.sqrt(2))), abs=1e-6)
    assert ncdf(0) == pytest.approx(0.5, abs=1e-6)


def test_corr_matrix_is_symmetric_with_zero_diagonal():
    r = corr_matrix(EIGHT)
    assert np.allclose(r, r.T) and np.all(np.diag(r) == 0)
    assert r[EIGHT.index("REB"), EIGHT.index("BLK")] == 0.35
    assert r[EIGHT.index("STL"), EIGHT.index("BLK")] == 0


def test_slot_weights_top_ten_full_next_three_bench():
    assert slot_weights(15).tolist() == [1.0] * 10 + [0.75] * 3 + [0.4] * 2
    assert slot_weights(4).tolist() == [1.0] * 4


def test_week_sums_games_by_slot_and_availability():
    board = pd.DataFrame([player(5.0, PTS=30.0), player(1.0, gp=41.0, PTS=10.0, FGM=2.0, FGA=10.0)])
    stats = StatMatrix(board, EIGHT)
    w = stats.week([1, 0], games=3.0)
    k = EIGHT.index
    g = np.array([3.0, 3.0 * 0.5])  # second player available half the season
    assert w.m[k("PTS")] == pytest.approx(g @ [30.0, 10.0])
    assert w.v[k("PTS")] == pytest.approx(g @ [30 + 0.09 * 900, 10 + 0.09 * 100])
    att, made = g @ [16.0, 10.0], g @ [8.0, 2.0]
    p = made / att
    assert w.m[k("FG%")] == pytest.approx(p)
    assert w.v[k("FG%")] == pytest.approx(1.6 * p * (1 - p) / att)


def test_lineup_slots_follow_value_order():
    board = pd.DataFrame([player(float(v), PTS=1.0) for v in range(12)])
    stats = StatMatrix(board, EIGHT)
    w = stats.week(range(12), games=1.0)
    # 10 at full weight, the two lowest-valued at 0.75
    assert w.m[EIGHT.index("PTS")] == pytest.approx(10 + 2 * 0.75)


def test_missing_games_count_as_70():
    board = pd.DataFrame([player(1.0, gp=0.0)])
    assert StatMatrix(board, EIGHT).avail[0] == pytest.approx(70 / 82)


def test_empty_roster_week():
    stats = StatMatrix(pd.DataFrame([player(1.0)]), EIGHT)
    w = stats.week([])
    assert w.m.sum() == 0 and w.v[EIGHT.index("FG%")] == 1.0


def test_category_odds_even_and_turnovers_reversed():
    stats = StatMatrix(pd.DataFrame([player(1.0)]), NINE)
    a = Week(np.ones(9), np.ones(9))
    assert category_odds(a, np.ones(9), np.ones(9), stats.sign) == pytest.approx(np.full(9, 0.5))
    b_m = np.zeros(9)
    p = category_odds(a, b_m, np.ones(9), stats.sign)
    assert p[NINE.index("PTS")] > 0.5 > p[NINE.index("TO")]


def test_win_probability():
    corr = corr_matrix(EIGHT)
    assert win_probability(np.full(8, 0.5), corr) == pytest.approx(0.5)
    assert win_probability(np.full(8, 0.999), corr) > 0.99
    # Correlated categories widen the spread, pulling a favourite back toward 50%.
    p = np.full(8, 0.6)
    assert 0.5 < win_probability(p, corr) < win_probability(p, np.zeros((8, 8)))


def test_outlook_and_change():
    board = pd.DataFrame([player(3.0, PTS=30.0), player(2.0), player(1.0, PTS=10.0)])
    stats = StatMatrix(board, EIGHT)
    weeks = {"A": stats.week([0]), "B": stats.week([1]), "C": stats.week([2])}
    o = outlook("A", weeks, stats)
    assert o.opponents == ["B", "C"]
    assert o.by_cat == pytest.approx(o.p.mean(0))
    assert o.p[1, EIGHT.index("PTS")] > o.p[0, EIGHT.index("PTS")] > 0.5
    p, cats, win = matchup(weeks["A"], weeks["C"], stats)
    assert cats == pytest.approx(o.cats_won[1]) and win == pytest.approx(o.win[1])
    worse = outlook("C", weeks, stats)
    ch = change(worse, o, weeks=20)
    assert ch.cats == pytest.approx((o.avg_cats - worse.avg_cats) * 20)
    assert ch.by_cat == pytest.approx((o.by_cat - worse.by_cat) * 20)
