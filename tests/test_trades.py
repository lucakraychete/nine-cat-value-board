import numpy as np
import pandas as pd
import pytest

from ninecat.config import categories
from ninecat.lines import LINE_STATS
from ninecat.matchup import GAMES_PER_WEEK, WEEKS
from ninecat.overview import ros_line
from ninecat.trades import (
    FINDER_POOL,
    MIN_GAIN,
    VARIETY,
    Look,
    apply_trade,
    build_trade_model,
    evaluate,
    find_trades,
    roster_fits,
    season_phase,
    seen_line,
    surpluses,
    trade_board,
    verdict,
    weekly_games,
)
from tests.conftest import TEAMS

EIGHT = categories(exclude_to=True)
NBA = ["DEN", "BOS", "PHL", "LAL", "MIA"]


@pytest.fixture
def healthy(players):
    return players.assign(injury_status="ACTIVE")


@pytest.fixture
def model(healthy):
    return build_trade_model(healthy, EIGHT, TEAMS)


# ---------------------------------------------------------------- the rest-of-season line

def test_trades_use_the_ros_line_whatever_the_board_shows(healthy):
    board = trade_board(healthy, EIGHT)
    ros = ros_line(healthy, None).set_index("player_id")
    got = board.set_index("player_id")
    assert got[LINE_STATS].to_numpy() == pytest.approx(ros.loc[got.index, LINE_STATS].to_numpy())


def test_every_rostered_player_stays_on_his_roster(healthy):
    board = trade_board(healthy, EIGHT)
    assert (board["fantasy_team"] != "FA").sum() == (healthy["fantasy_team"] != "FA").sum()


def test_weekly_games_follow_the_schedule_left():
    g, w = weekly_games(["DEN", "BOS"], None)
    assert g.tolist() == [GAMES_PER_WEEK] * 2 and w == WEEKS
    g, w = weekly_games(["DEN", "BOS", "XXX"], {"DEN": 60, "BOS": 20}, season_games=80)
    assert g.tolist() == pytest.approx([GAMES_PER_WEEK * 1.5, GAMES_PER_WEEK * 0.5, GAMES_PER_WEEK])
    assert w == pytest.approx(WEEKS * 40 / 80)
    # games x weeks is the Overview's ROS scaling: 3.5 x 20 x his team's games left / season games
    assert g[0] * w == pytest.approx(GAMES_PER_WEEK * WEEKS * 60 / 80)
    g, w = weekly_games(["DEN"], {"DEN": 0, "BOS": 0})
    assert g.tolist() == [0.0] and w == 0


def test_a_team_with_no_games_left_contributes_nothing(healthy):
    m = build_trade_model(healthy, EIGHT, TEAMS, remaining={**{t: 40 for t in NBA}, "DEN": 0})
    den = [i for i, t in enumerate(m.board["pro_team"]) if t == "DEN"]
    week = m.lg.week(den)
    assert week.m[m.lg.stats.count_ix] == pytest.approx(0)
    for i in den:   # adding him only costs the team whoever he replaces
        assert all(m.lg.worth[i, k] <= 1e-9 for k, t in enumerate(m.lg.teams) if t != m.lg.owner[i])
    over = build_trade_model(healthy, EIGHT, TEAMS, remaining={t: 0 for t in NBA})
    assert np.abs(over.lg.worth).max() == 0


# ---------------------------------------------------------------- settlement and legality

def roster_of(m, team):
    return [int(x) for x in m.lg.rosters[team]]


@pytest.mark.parametrize("n_give,n_get", [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (3, 3), (3, 2)])
def test_settlement_keeps_roster_sizes(model, n_give, n_get):
    a, b = TEAMS[0], TEAMS[1]
    give, get = roster_of(model, a)[:n_give], roster_of(model, b)[:n_get]
    sa, sb = apply_trade(model.lg, a, give, b, get)
    assert len(sa.roster) == len(model.lg.rosters[a]) and len(sb.roster) == len(model.lg.rosters[b])
    assert set(get) <= set(sa.roster) and set(give) <= set(sb.roster)       # arrivals are never dropped
    assert not set(give) & set(sa.roster) and not set(get) & set(sb.roster)
    assert len(sa.drops) == max(0, n_get - n_give) and len(sb.drops) == max(0, n_give - n_get)
    assert len(sa.adds) == max(0, n_give - n_get) and len(sb.adds) == max(0, n_get - n_give)
    assert not set(sa.adds) & set(sb.adds)                                  # no free agent picked up twice
    assert all(model.lg.owner[f] == "FA" for f in sa.adds + sb.adds)


def test_three_for_one_keeps_both_rosters_legal(model):
    a, b = TEAMS[0], TEAMS[2]
    assert roster_fits(model.lg, model.lg.rosters[a]) and roster_fits(model.lg, model.lg.rosters[b])
    res = evaluate(model.lg, a, roster_of(model, a)[:3], b, roster_of(model, b)[:1])
    assert res.fits and roster_fits(model.lg, res.a.roster) and roster_fits(model.lg, res.b.roster)


def test_roster_fits_needs_a_healthy_player_everywhere(model):
    lg = model.lg
    roster = roster_of(model, TEAMS[0])
    no_center = [x for x in roster if "C" not in lg.positions[x]]
    assert not roster_fits(lg, no_center)
    centers = [x for x in roster if "C" in lg.positions[x]]
    saved = lg.injury.copy()
    lg.injury[centers] = "OUT"
    assert not roster_fits(lg, roster)
    lg.injury[:] = saved


def test_evaluate_reports_both_sides_and_adds_up(model):
    a, b = TEAMS[0], TEAMS[1]
    res = evaluate(model.lg, a, roster_of(model, a)[:1], b, roster_of(model, b)[:1])
    assert res.me.by_cat.sum() == pytest.approx(res.me.cats)
    assert res.them.by_cat.sum() == pytest.approx(res.them.cats)
    assert res.gap == pytest.approx(res.me.cats - res.them.cats)
    assert -1 <= res.me.win <= 1 and -1 <= res.them.win <= 1
    nothing = evaluate(model.lg, a, [], b, [])
    assert nothing.me.cats == pytest.approx(0) and nothing.them.cats == pytest.approx(0)


def test_surpluses_rank_by_what_another_team_gains(model):
    me = TEAMS[0]
    s = surpluses(model.lg, me)
    assert len(s) == len(model.lg.rosters[me])
    assert [x.surplus for x in s] == sorted((x.surplus for x in s), reverse=True)
    for x in s:
        assert x.best != me and x.best_worth >= (x.second_worth if x.second is not None else -np.inf)
        assert x.by_cat.sum() == pytest.approx(x.best_worth)


# ---------------------------------------------------------------- perception

def test_season_phase_blends_from_draft_position_to_this_season():
    assert season_phase([0, 0, np.nan]) == 0
    assert season_phase([2, 6, 10, 0]) == pytest.approx(6 / 15)
    assert season_phase([20, 30, 40]) == 1


def test_seen_line_shifts_to_this_season_with_the_phase():
    proj = pd.DataFrame({s: [10.0, 10.0] for s in ("PTS", "REB", "AST", "3PM", "STL", "BLK", "MIN")})
    cur = proj * 2
    gp = pd.Series([5.0, 0.0])
    assert seen_line(proj, cur, gp, 0.0)["PTS"].tolist() == [10.0, 10.0]
    assert seen_line(proj, cur, gp, 0.5)["PTS"].tolist() == [15.0, 10.0]    # no games: still the projection
    assert seen_line(proj, cur, gp, 1.0)["PTS"].tolist() == [20.0, 10.0]


def test_name_value_moves_from_adp_to_headline_rank(healthy):
    early = build_trade_model(healthy.assign(S_G=np.nan, T_GP=np.nan), EIGHT, TEAMS).perception
    assert early.phase == 0
    late = build_trade_model(healthy.assign(S_G=30.0), EIGHT, TEAMS)
    assert late.perception.phase == 1
    assert late.perception.name_rank == pytest.approx(late.perception.head_rank)
    drafted = early.name_rank[:5]
    assert np.isfinite(drafted).all()


def test_loss_aversion_scales_what_they_give_up(model):
    pv = model.perception
    b = TEAMS[1]
    give, get = roster_of(model, TEAMS[0])[:1], roster_of(model, b)[:1]
    look = pv.accept(b, give, get)
    assert look.lam == 1.3 and look.score == pytest.approx(look.inn - 1.3 * look.out)
    pv.lambdas[b] = 2.0
    assert pv.accept(b, give, get).score == pytest.approx(look.inn - 2.0 * look.out)
    assert verdict(Look(20, 5, 1.3))[0] == "Likely yes" and verdict(Look(10, 10, 1.3))[0] == "Likely no"


# ---------------------------------------------------------------- finder

def test_finder_pruning_filters_and_variety(model):
    me = TEAMS[0]
    lg, pv = model.lg, model.perception
    j = lg.teams.index(me)
    deals = find_trades(model, me, veto_cap=4)
    assert deals
    for d in deals:
        offer = sorted(lg.rosters[me], key=lambda x: -pv.pv(d.partner, x))[:FINDER_POOL]
        want = sorted(lg.rosters[d.partner], key=lambda x: -lg.worth[x, j])[:FINDER_POOL]
        assert set(d.give) <= set(offer) and set(d.get) <= set(want)
        assert not (len(d.give) == 2 and len(d.get) == 2)                   # 2-for-2 only behind the toggle
        assert d.result.me.cats > MIN_GAIN and d.result.them.cats >= -4 and d.result.fits
        assert d.look.score >= 0
    keys = pd.Series([tuple(sorted(d.get)) for d in deals]).value_counts()
    assert keys.max() <= VARIETY
    assert [d.score for d in deals] == sorted((d.score for d in deals), reverse=True)


def test_finder_modes_partner_filter_and_two_for_two(model):
    me = TEAMS[0]
    only = find_trades(model, me, partners=[TEAMS[2]])
    assert {d.partner for d in only} <= {TEAMS[2]}
    blk = find_trades(model, me, mode="cat", cat="BLK")
    k = model.lg.cats.index("BLK")
    assert [d.score for d in blk] == pytest.approx([d.result.me.by_cat[k] + 0.05 * d.result.me.cats for d in blk])
    punt = find_trades(model, me, mode="punt", punt=["AST"])
    keep = [c != "AST" for c in model.lg.cats]
    assert [d.score for d in punt] == pytest.approx([d.result.me.by_cat[keep].sum() for d in punt])
    wide = find_trades(model, me, two_for_two=True, limit=200, veto_cap=99, min_look=-1e9)
    assert any(len(d.give) == 2 and len(d.get) == 2 for d in wide)


def test_finder_built_around_one_target(model):
    me, b = TEAMS[0], TEAMS[1]
    target = roster_of(model, b)[0]
    deals = find_trades(model, me, must_get=target, veto_cap=99, min_look=-1e9)
    assert deals and all(target in d.get and d.partner == b for d in deals)


def test_tighter_veto_cap_never_adds_deals(model):
    me = TEAMS[0]
    loose = find_trades(model, me, veto_cap=12, limit=500)
    tight = find_trades(model, me, veto_cap=0.5, limit=500)
    assert len(tight) <= len(loose) and all(d.result.them.cats >= -0.5 for d in tight)
