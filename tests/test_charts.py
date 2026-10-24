import pandas as pd
import plotly.graph_objects as go
import pytest

from ninecat import charts
from ninecat.player import RankItem
from ninecat.theme import tokens

T = tokens("light")


@pytest.mark.parametrize("rank,tier,x", [(1, 0, 0.1), (10, 0, 1.0), (20, 1, 1.5), (200, 5, 6.0), (500, 5, 6.0)])
def test_ladder_position(rank, tier, x):
    assert charts.ladder_position(rank) == (tier, pytest.approx(x))


def test_beeswarm_levels_never_overlap_on_the_same_level():
    values = [0.0, 0.01, 0.02, 0.03, 5.0, 10.0]
    levels = charts.beeswarm_levels(values)
    xs = [10 + v / 10 * 540 for v in values]
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if levels[i] == levels[j]:
                assert abs(xs[i] - xs[j]) >= 9
    assert sorted(levels[:4]) == [-1, 0, 1, 2]


def test_figures_build():
    item = RankItem("Usage", "share", "30.0%", 80.0, None, "#3 · C #1")
    games = pd.DataFrame({"date": ["2026-01-01", "2026-01-03"], "MIN": [30.0, 34.0],
                          "r5": [30.0, 32.0], "r10": [30.0, 32.0]})
    figs = [
        charts.tier_ladder(12, T),
        charts.minutes_chart([("2024-25", 30.0, False), ("2026-27 proj", 33.0, True)], T),
        charts.rolling_minutes_chart(games, T),
        charts.radar(["PTS", "REB", "AST"], [90, 50, 10], [50, 50, 50], T),
        charts.value_strip([-1.0, 0.5, 2.0], ["a", "b", "c"], [3, 2, 1], 2, T),
        charts.percentile_bars([item, item], T, "C"),
    ]
    figs += [
        charts.category_delta(["PTS", "REB"], [1.2, -0.4], T, league=[0.5, 0.1]),
        charts.win_bars(["A", "B"], [0.7, 0.4], [5.1, 3.2], T),
        charts.win_change(["A", "B"], [0.5, 0.4], [0.6, 0.35], T),
        charts.category_strength(["PTS", "REB"], [0.8, 0.3], [1, 7], 20, ["REB"], T),
        charts.worth_bars(["A", "B"], [2.0, -1.0], "A", "B", T),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
    # gains green, losses red; a punted category is faded
    assert list(figs[6].data[0].marker.color) == [T["bad"], T["good"]]
    assert figs[9].data[0].marker.color[0].endswith("0.35)")
    # win bars run worst to best, bottom to top
    assert list(figs[7].data[0].y) == ["B", "A"]
    # projection segment is dashed, the real season solid
    assert figs[1].data[0].line.dash == "dash"
    # percentile bars: only the overall dots exist when position percentiles are missing
    assert len(figs[5].data) == 1 and list(figs[5].data[0].x) == [80.0, 80.0]
