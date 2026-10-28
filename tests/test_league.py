import numpy as np
import pandas as pd
import pytest

from ninecat.board import build_board
from ninecat.config import categories
from ninecat.league import (
    FREE_AGENT,
    build_league,
    drop_options,
    fit_note,
    team_category_totals,
    uncertainty,
    waivers,
)
from ninecat.matchup import change, outlook
from tests.conftest import TEAMS

EIGHT = categories(exclude_to=True)
NINE = categories(exclude_to=False)


@pytest.fixture(scope="module")
def board():
    from tests.conftest import synthetic_players
    return build_board(synthetic_players(), EIGHT)


@pytest.fixture(scope="module")
def lg(board):
    return build_league(board, EIGHT, TEAMS)


def test_team_totals_use_top_ten_and_rank_turnovers_low(board):
    totals = team_category_totals(board, NINE, TEAMS)
    team = board[board["fantasy_team"] == "Alpha"].sort_values("value", ascending=False).head(10)
    assert totals.loc["Alpha", "PTS"] == pytest.approx(team["PTS"].sum())
    assert totals.loc["Alpha", "FG%"] == pytest.approx(team["FGM"].sum() / team["FGA"].sum())
    best_to = totals["TO"].idxmin()
    assert totals.loc[best_to, "r_TO"] == 1
    assert totals.loc[totals["PTS"].idxmax(), "r_PTS"] == 1


def test_rosters_follow_value_order(lg):
    for t in TEAMS:
        vals = lg.stats.value[lg.rosters[t]]
        assert (np.diff(vals) <= 0).all()
        assert (lg.owner[lg.rosters[t]] == t).all()


def test_drop_for_prefers_lowest_value_at_shared_position(lg):
    roster = list(lg.rosters["Alpha"])
    incoming = next(i for i in np.flatnonzero(lg.owner == FREE_AGENT))
    d = lg.drop_for(roster, incoming)
    same = [x for x in roster if lg.positions[x] & lg.positions[incoming]]
    expected = min(same or roster, key=lambda x: lg.stats.value[x])
    assert d == expected
    assert lg.drop_for(roster, incoming, keep=[expected]) != expected


def test_best_fa_at_prefers_position_then_value(lg):
    r = lg.rosters["Bravo"][0]
    f = lg.best_fa_at(r)
    fas = [x for x in np.flatnonzero(lg.owner == FREE_AGENT)]
    same = [x for x in fas if lg.positions[x] & lg.positions[r]]
    assert f == max(same or fas, key=lambda x: lg.stats.value[x])


def test_worth_for_another_team_is_adding_him_with_a_drop(lg):
    r = lg.rosters["Alpha"][0]
    j = TEAMS.index("Bravo")
    roster = list(lg.rosters["Bravo"])
    d = lg.drop_for(roster, r)
    ch = change(lg.base["Bravo"], lg.outlook_with("Bravo", [x for x in roster if x != d] + [r]))
    assert lg.worth[r, j] == pytest.approx(ch.cats)
    assert lg.worth_by_cat[r, j] == pytest.approx(ch.by_cat)


def test_worth_to_own_team_is_versus_best_free_agent(lg):
    r = lg.rosters["Alpha"][0]
    j = TEAMS.index("Alpha")
    f = lg.best_fa_at(r)
    roster = [x for x in lg.rosters["Alpha"] if x != r] + [f]
    ch = change(lg.base["Alpha"], lg.outlook_with("Alpha", roster))
    assert lg.worth[r, j] == pytest.approx(-ch.cats)
    assert lg.worth[r, j] > 0  # a team's best player beats the wire


def test_impact_averages_the_other_teams(lg):
    r = lg.rosters["Alpha"][0]
    others = [TEAMS.index(t) for t in TEAMS if t != "Alpha"]
    assert lg.impact_by_cat[r] == pytest.approx(lg.worth_by_cat[r, others].mean(0))
    assert lg.impact[r] == pytest.approx(lg.worth[r, others].mean())
    fa = np.flatnonzero(lg.owner == FREE_AGENT)[0]
    assert lg.impact[fa] == pytest.approx(lg.worth[fa].mean())


def test_impact_rank_and_replacement(lg):
    order = np.argsort(lg.impact_rank)
    assert (np.diff(lg.impact[order]) <= 1e-12).all()
    r = lg.rosters["Charlie"][0]
    assert lg.vs_repl[r] == pytest.approx(lg.impact[r] - lg.impact[lg.repl[r]])
    for p, ranks in lg.pos_impact_rank.items():
        members = [i for i in range(len(ranks)) if p in lg.positions[i]]
        assert sorted(ranks[members]) == list(range(1, len(members) + 1))


def test_stars_matter_more_than_scrubs(lg):
    best = lg.rosters["Alpha"][0]
    worst = lg.rosters["Alpha"][-1]
    assert lg.impact[best] > lg.impact[worst]


def test_drop_options_cover_the_roster_best_first(lg):
    add = np.flatnonzero(lg.owner == FREE_AGENT)[0]
    opts = drop_options(lg, "Alpha", add)
    assert {o.drop for o in opts} == set(lg.rosters["Alpha"])
    assert [o.change.cats for o in opts] == sorted((o.change.cats for o in opts), reverse=True)


def test_waivers_rank_free_agents_by_their_best_drop(lg):
    picks, drops, out_player = waivers(lg, "Bravo")
    assert len(picks) == (lg.owner == FREE_AGENT).sum()
    assert [p.change.cats for p in picks] == sorted((p.change.cats for p in picks), reverse=True)
    top = picks[0]
    assert top.change.cats == pytest.approx(drop_options(lg, "Bravo", top.player)[0].change.cats)
    assert {d.player for d in drops} == set(lg.rosters["Bravo"])
    if out_player is not None:
        assert lg.injury[out_player] == "OUT" and all(p.il == out_player for p in picks)


def test_swap_weeks_send_the_outgoing_player_back_in_a_trade(lg):
    incoming, outgoing = lg.rosters["Bravo"][0], lg.rosters["Alpha"][-1]
    weeks = lg.swap_weeks("Alpha", incoming, outgoing)
    expected = lg.stats.week([x for x in lg.rosters["Bravo"] if x != incoming] + [outgoing])
    assert weeks["Bravo"].m == pytest.approx(expected.m)
    assert weeks["Charlie"] is lg.weeks["Charlie"]
    assert outlook("Alpha", weeks, lg.stats).avg_cats > lg.base["Alpha"].avg_cats


def test_standings_and_bands(lg):
    s = lg.standings()
    assert list(s["win"]) == sorted(s["win"], reverse=True) and set(s["team"]) == set(TEAMS)
    for c in EIGHT:
        p = lg.base["Alpha"].by_cat[EIGHT.index(c)]
        assert lg.band("Alpha", c) == ("lock" if p >= 0.7 else "swing" if p >= 0.3 else "weak")


class StubLeague:
    cats = ["PTS", "REB", "AST"]
    cat_rank = pd.DataFrame({"PTS": [1], "REB": [5], "AST": [9]}, index=["T"])
    bands = {"PTS": "lock", "REB": "swing", "AST": "weak"}

    def band(self, team, cat):
        return self.bands[cat]


@pytest.mark.parametrize("by_cat,text,good", [
    ([0.0, 0.5, 0.0], "Fills swing: REB", True),
    ([0.5, 0.0, 0.0], "Piles onto categories already won", False),
    ([0.0, 0.0, 0.5], "Helps categories mostly lost", False),
    ([0.0, 0.0, 0.0], "No meaningful category gain", False),
    ([-1.0, 0.4, 0.0], "Net loss: the costs outweigh swing gains in REB", False),
])
def test_fit_note(by_cat, text, good):
    f = fit_note(StubLeague(), np.array(by_cat), "T")
    assert f.text == text and f.good == good


def test_fit_note_details():
    f = fit_note(StubLeague(), np.array([-0.3, 0.5, 0.2]), "T")
    assert f.gains == "REB +0.5 (swing, #5), AST +0.2 (weak, #9)"
    assert f.losses == "PTS -0.3"


@pytest.mark.parametrize("games,injury,label,spread", [
    (None, "ACTIVE", "wide", 0.4), (20, "ACTIVE", "moderate", 0.25),
    (60, "OUT", "moderate", 0.25), (60, "ACTIVE", "narrow", 0.12)])
def test_uncertainty(games, injury, label, spread):
    u = uncertainty(10.0, games, injury)
    assert u.label == label and u.high - 10.0 == pytest.approx(max(0.6, 10 * spread))
    assert uncertainty(0.5, 60, "ACTIVE").high == pytest.approx(1.1)  # floor of 0.6


def test_uncertainty_uses_the_models_measured_range():
    from ninecat.league import uncertainty
    u = uncertainty(5.0, 70, "ACTIVE", value_range=(-1.0, 2.0), slope=1.5, games=12)
    assert (u.low, u.high) == pytest.approx((3.5, 8.0)) and "12 games" in u.why
    assert u.label == "wide"                                      # half-width 2.25 is 45% of 5
    assert uncertainty(5.0, 70, None, value_range=(-0.4, 0.4), slope=1.5).label == "narrow"     # 12%
    assert uncertainty(5.0, 70, None, value_range=(-1.0, 1.0), slope=1.5).label == "moderate"   # 30%
    hand = uncertainty(5.0, 0, None)
    assert hand.label == "wide" and "Projection only" in hand.why
