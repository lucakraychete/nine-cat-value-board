"""Trades: what a deal really does for both teams over the rest of the
season, how the other manager is likely to see it, and the finder that
searches for deals that are good on the first and look good on the second.

Every real number is on the rest-of-season line, trade_line() (the
Overview's interim ros_line() until the Part 5 model replaces it), whatever
stats source the board shows. A trade only changes the games left, so the
league is rebuilt for the season that remains (weekly_games()):

- each player's weekly games are GAMES_PER_WEEK x his NBA team's remaining
  games / the average team's remaining games, with availability (from the
  projection) on top as in every team-week;
- changes count over WEEKS x the average team's remaining games / season
  games weeks.

Games x weeks is then 3.5 x 20 x his team's games left / season games, the
same scaling as the Overview's ROS column, and a player whose team has no
games left adds nothing.

Ported from the original app: applyTrade (apply_trade), evalTrade
(evaluate), rosterFits (roster_fits), sellMap (surpluses), buildPerception
(build_perception) and findGapTrades (find_trades). Pure: frames in,
dataclasses out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from ninecat.board import BOARD_MIN_MINUTES, player_key
from ninecat.config import ROS_PRIOR_GAMES
from ninecat.league import FREE_AGENT, POSITIONS, League, build_league
from ninecat.lines import LINE_STATS, stat_line
from ninecat.matchup import GAMES_PER_WEEK, WEEKS, Change, outlook
from ninecat.overview import games_played, ros_line, season_rate_line
from ninecat.valuation import value_players

MAX_SIDE = 3            # calculator trades go up to 3-for-3
FINDER_POOL = 8         # finder candidates per side per partner (the original's 8)
FINDER_FILLERS = 6      # extra pieces tried alongside a must-get target
VARIETY = 2             # at most two finder deals built around the same incoming players
MIN_GAIN = 0.3          # the finder keeps deals worth more than this to you


# ---------------------------------------------------------------- the rest-of-season league

def trade_line(players: pd.DataFrame, preseason: pd.DataFrame | None = None,
               k: float = ROS_PRIOR_GAMES) -> pd.DataFrame:
    """The per-game line every trade is valued on. Part 5 swaps the model in here."""
    return ros_line(players, preseason, k)


def trade_board(players: pd.DataFrame, cats: Sequence[str], punt: Sequence[str] = (),
                pool_size: int = 130, season_games: int = 82,
                preseason: pd.DataFrame | None = None) -> pd.DataFrame:
    """Everyone valued on trade_line(): every rostered player with a line (so
    rosters stay whole) and free agents with BOARD_MIN_MINUTES or more.

    Adds the inputs perception needs: the ESPN projection (pv_proj_*), this
    season so far (pv_cur_*), games played (pv_gp) and ESPN's last-15-day
    minutes (pv_l15_min).
    """
    p = players.drop_duplicates(subset="player_id" if "player_id" in players else "name")
    line = trade_line(p, preseason)
    rostered = line["fantasy_team"].fillna(FREE_AGENT) != FREE_AGENT
    keep = line[LINE_STATS].notna().all(axis=1) & (rostered | (line["MIN"].fillna(0) >= BOARD_MIN_MINUTES))
    proj, cur = stat_line(p, "proj"), season_rate_line(p, season_games)
    extra = pd.DataFrame(index=p.index)
    for s in ("MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK"):
        extra[f"pv_proj_{s}"] = pd.to_numeric(proj[s], errors="coerce")
        extra[f"pv_cur_{s}"] = pd.to_numeric(cur[s], errors="coerce")
    extra["pv_gp"] = games_played(p)
    extra["pv_l15_min"] = pd.to_numeric(p["R15_MIN"], errors="coerce") if "R15_MIN" in p else np.nan
    rows = line[keep].join(extra)
    return value_players(rows, cats, punt=punt, pool_size=pool_size, season_games=season_games, games_discount=True)


def weekly_games(pro_team: Sequence, remaining: Mapping[str, float] | None,
                 season_games: int = 82) -> tuple[np.ndarray, float]:
    """Each player's games in a typical week of the season left, and how many weeks that is.

    Without a schedule, the full season (GAMES_PER_WEEK over WEEKS). A player
    whose NBA team isn't in the schedule gets the average team's games.
    """
    n = len(pro_team)
    if not remaining:
        return np.full(n, GAMES_PER_WEEK), float(WEEKS)
    avg = float(np.mean(list(remaining.values())))
    if avg <= 0:
        return np.zeros(n), 0.0
    left = np.array([float(remaining.get(t, avg)) for t in pro_team])
    return GAMES_PER_WEEK * left / avg, WEEKS * min(avg / season_games, 1.0)


@dataclass
class TradeModel:
    """The rest-of-season league, and how each manager sees every player."""
    board: pd.DataFrame
    lg: League
    perception: Perception


def build_trade_model(players: pd.DataFrame, cats: Sequence[str], teams: Sequence[str],
                      punt: Sequence[str] = (), pool_size: int = 130, season_games: int = 82,
                      preseason: pd.DataFrame | None = None, remaining: Mapping[str, float] | None = None,
                      lambdas: Mapping[str, float] | None = None) -> TradeModel:
    board = trade_board(players, cats, punt, pool_size, season_games, preseason)
    games, weeks = weekly_games(board["pro_team"].tolist(), remaining, season_games)
    lg = build_league(board, cats, teams, season_games=season_games, games=games, season_weeks=weeks)
    return TradeModel(board, lg, build_perception(board, lg, lambdas))


# ---------------------------------------------------------------- one trade

@dataclass(frozen=True)
class Side:
    """One team's roster after a trade, and the moves that settled its size."""
    team: str
    roster: list[int]
    incoming: list[int]
    outgoing: list[int]
    drops: list[int]
    adds: list[int]


def apply_trade(lg: League, a: str, give: Sequence[int], b: str, get: Sequence[int]) -> tuple[Side, Side]:
    """Both rosters after a sends `give` to b for `get`, sizes kept.

    The side with more players drops its lowest-value players sharing a
    position with what came in (never the arrivals); the side with fewer
    picks up the best free agents at the positions it sent away. The two
    sides never pick up the same free agent.
    """
    used: list[int] = []

    def settle(team: str, incoming: list[int], outgoing: list[int]) -> Side:
        orig = list(lg.rosters[team])
        roster = [x for x in orig if x not in outgoing] + incoming
        drops, adds = [], []
        while len(roster) > len(orig):
            k = len(roster) - len(orig) - 1
            d = lg.drop_for(roster, incoming[k] if k < len(incoming) else incoming[0], keep=incoming)
            if d is None:
                break
            roster.remove(d)
            drops.append(d)
        i = 0
        while len(roster) < len(orig) and outgoing:
            f = lg.best_fa_at(outgoing[i % len(outgoing)], exclude=used)
            if f is None:
                break
            used.append(f)
            roster.append(f)
            adds.append(f)
            i += 1
        return Side(team, roster, incoming, outgoing, drops, adds)

    give, get = [int(x) for x in give], [int(x) for x in get]
    return settle(a, get, give), settle(b, give, get)


def roster_fits(lg: League, roster: Sequence[int]) -> bool:
    """Someone not listed OUT can play every position."""
    return all(any(p in lg.positions[x] and lg.injury[x] != "OUT" for x in roster) for p in POSITIONS)


@dataclass(frozen=True)
class TradeResult:
    a: Side
    b: Side
    me: Change       # a's change over the season left
    them: Change     # b's
    fits: bool       # both rosters legal
    weeks: dict      # every team-week after the trade

    @property
    def gap(self) -> float:
        return self.me.cats - self.them.cats


def evaluate(lg: League, a: str, give: Sequence[int], b: str, get: Sequence[int]) -> TradeResult:
    """Category wins and weekly matchup win chance for both teams, against the league."""
    sa, sb = apply_trade(lg, a, give, b, get)
    weeks = dict(lg.weeks)
    weeks[a], weeks[b] = lg.week(sa.roster), lg.week(sb.roster)
    return TradeResult(sa, sb, lg.change(lg.base[a], outlook(a, weeks, lg.stats)),
                       lg.change(lg.base[b], outlook(b, weeks, lg.stats)),
                       roster_fits(lg, sa.roster) and roster_fits(lg, sb.roster), weeks)


# ---------------------------------------------------------------- who to trade, and to whom

@dataclass(frozen=True)
class Surplus:
    player: int
    mine: float              # worth to you: vs the best free agent at his position
    best: str                # the team he'd help most
    best_worth: float
    second: str | None
    second_worth: float | None
    by_cat: np.ndarray       # his worth to `best`, by category

    @property
    def surplus(self) -> float:
        return self.best_worth - self.mine


def surpluses(lg: League, me: str) -> list[Surplus]:
    """Each of my players, the team he'd add the most for, and how much more
    that is than he's worth to me; biggest surplus first."""
    j = lg.teams.index(me)
    out = []
    for r in lg.rosters[me]:
        others = sorted(((lg.worth[r, k], t) for k, t in enumerate(lg.teams) if t != me), reverse=True)
        if not others:
            continue
        second = others[1] if len(others) > 1 else (None, None)
        out.append(Surplus(int(r), float(lg.worth[r, j]), others[0][1], float(others[0][0]), second[1],
                           None if second[0] is None else float(second[0]),
                           lg.worth_by_cat[r, lg.teams.index(others[0][1])]))
    return sorted(out, key=lambda s: -s.surplus)


# ---------------------------------------------------------------- how other managers see players

# Most managers don't run category simulations. They judge by the name (draft
# position before the season, this season's headline numbers once games
# start), the headline stats, whether he beats who they start at his
# position, recent form, and they value what they give up more than what
# they get (loss aversion).
PV_NAME, PV_HEADLINE, PV_FIT = 0.45, 0.35, 0.20
PACKAGE_WEIGHTS = (1.0, 0.7, 0.5, 0.4)   # best player fully, extras less
DEFAULT_LAMBDA = 1.3
PHASE_GAMES = 15                          # median games played at which this season fully takes over
SEEN_STATS = ("PTS", "REB", "AST", "3PM", "STL", "BLK", "MIN")


def pv_scale(rank) -> np.ndarray:
    """Rank -> 0..100 points."""
    return 100 * np.exp(-(np.maximum(1, np.asarray(rank, float)) - 1) / 45)


def season_phase(gp: Sequence[float]) -> float:
    """0 = preseason inputs only, 1 = this season drives everything: the
    median games played (among players with any) over PHASE_GAMES."""
    g = np.sort(np.asarray(gp, float)[np.asarray(gp, float) > 0])
    return 0.0 if not len(g) else min(1.0, g[len(g) // 2] / PHASE_GAMES)


def seen_line(proj: pd.DataFrame, cur: pd.DataFrame, gp: pd.Series, phase: float) -> pd.DataFrame:
    """The per-game line a manager looks at: the projection, shifting to this season as games are played."""
    out = pd.DataFrame(index=proj.index)
    played = gp.fillna(0) > 0
    for s in SEEN_STATS:
        v = proj[s].fillna(0)
        out[s] = v.where(~(played & cur[s].notna()), (1 - phase) * v + phase * cur[s])
    return out


def headline_score(line: pd.DataFrame) -> pd.Series:
    """The box-score numbers people read, weighted roughly the way they read them."""
    return line["PTS"] + 1.2 * line["REB"] + 1.5 * line["AST"] + line["3PM"] + 1.5 * (line["STL"] + line["BLK"])


@dataclass(frozen=True)
class Look:
    """How an offer reads to the manager receiving `incoming` for `outgoing`."""
    inn: float
    out: float
    lam: float

    @property
    def score(self) -> float:
        return self.inn - self.lam * self.out

    @property
    def ratio(self) -> float:
        return self.inn / max(1e-6, self.lam * self.out)


def verdict(look: Look) -> tuple[str, str]:
    """A label and its css class."""
    return ("Likely yes", "pos") if look.score >= 8 else ("Borderline", "") if look.score >= 0 else ("Likely no", "neg")


@dataclass
class Perception:
    phase: float
    seen: pd.DataFrame
    head: np.ndarray
    head_rank: np.ndarray
    name_rank: np.ndarray
    recency: np.ndarray
    base: np.ndarray
    positions: list[frozenset[str]]
    rosters: dict[str, np.ndarray]
    lambdas: Mapping[str, float]

    def lam(self, team: str) -> float:
        return float(self.lambdas.get(team, DEFAULT_LAMBDA))

    def starter(self, team: str, r: int, exclude: Sequence[int] = ()) -> int | None:
        """The best player (by headline score) they already have at his position."""
        cands = [x for x in self.rosters.get(team, []) if x != r and x not in exclude
                 and self.positions[x] & self.positions[r]]
        return max(cands, key=lambda x: self.head[x]) if cands else None

    def fit(self, team: str, r: int, exclude: Sequence[int] = ()) -> float:
        s = self.starter(team, r, exclude)
        return 100.0 if s is None else float(np.clip(50 + 2.5 * (self.head[r] - self.head[s]), 0, 100))

    def pv(self, team: str, r: int, exclude: Sequence[int] = ()) -> float:
        """How this manager rates him, 0-100."""
        return float(self.base[r] / (PV_NAME + PV_HEADLINE) * (1 - PV_FIT) + PV_FIT * self.fit(team, r, exclude))

    def package(self, team: str, players: Sequence[int], exclude: Sequence[int] = ()) -> float:
        vals = sorted((self.pv(team, x, exclude) for x in players), reverse=True)
        return float(sum(v * PACKAGE_WEIGHTS[min(i, len(PACKAGE_WEIGHTS) - 1)] for i, v in enumerate(vals)))

    def accept(self, team: str, incoming: Sequence[int], outgoing: Sequence[int]) -> Look:
        """What they get is judged against their roster without what they send."""
        return Look(self.package(team, incoming, outgoing), self.package(team, outgoing), self.lam(team))


def build_perception(board: pd.DataFrame, lg: League, lambdas: Mapping[str, float] | None = None) -> Perception:
    """buildPerception, on a trade_board() (rows in the league's order).

    The original also nudged the preseason view with preseason games; the
    cache has no preseason stats, so that step is left out.
    """
    gp = board["pv_gp"].astype(float)
    phase = season_phase(gp)
    proj = pd.DataFrame({s: board[f"pv_proj_{s}"] for s in SEEN_STATS})
    proj = proj.where(proj.notna(), board[list(SEEN_STATS)].to_numpy())   # no projection: the trade line
    cur = pd.DataFrame({s: board[f"pv_cur_{s}"] for s in SEEN_STATS})
    seen = seen_line(proj, cur, gp, phase)
    head = headline_score(seen).to_numpy(float)
    head_rank = pd.Series(-head).rank(method="first").to_numpy(float)
    adp = pd.to_numeric(board["ADP"], errors="coerce") if "ADP" in board else pd.Series(np.nan, index=board.index)
    adp = adp.where(adp > 0, 20 + 1.2 * board["rank"]).to_numpy(float)
    name_rank = (1 - phase) * adp + phase * head_rank
    l15, cur_min = board["pv_l15_min"].to_numpy(float), cur["MIN"].to_numpy(float)
    trend = np.where((phase > 0) & (l15 > 0) & (cur_min > 0), l15 / np.where(cur_min > 0, cur_min, 1), 1.0)
    recency = np.clip(0.5 + 0.5 * trend, 0.85, 1.15)
    base = (PV_NAME * pv_scale(name_rank) + PV_HEADLINE * pv_scale(head_rank)) * recency
    return Perception(phase, seen, head, head_rank, name_rank, recency, base, lg.positions, lg.rosters,
                      dict(lambdas or {}))


# ---------------------------------------------------------------- the finder

FINDER_MODES = {"gap": "Biggest gap", "cat": "Max one category", "punt": "Punt build"}


@dataclass(frozen=True)
class Deal:
    partner: str
    give: tuple[int, ...]
    get: tuple[int, ...]
    result: TradeResult
    look: Look
    score: float


def _pairs(xs: Sequence[int]) -> list[tuple[int, ...]]:
    return [tuple(p) for p in combinations(xs, 2)]


def find_trades(model: TradeModel, me: str, mode: str = "gap", cat: str | None = None,
                punt: Sequence[str] = (), veto_cap: float = 4, min_look: float = 0.0,
                partners: Sequence[str] | None = None, two_for_two: bool = False, limit: int = 30,
                must_get: int | None = None) -> list[Deal]:
    """findGapTrades: deals that read as fair or better to the other manager
    (after loss aversion), gain you more than MIN_GAIN, cost them no more
    than veto_cap and leave both rosters legal; best first by `mode`:

    - gap: your gain minus theirs;
    - cat: your gain in `cat` (plus 5% of your total, to break ties);
    - punt: your gain over the categories you don't punt.

    Candidates are pruned first: per partner, the FINDER_POOL players of
    yours they'd rate highest and the FINDER_POOL of theirs worth most to
    you, and an offer is only simulated once it reads well to them.
    1-for-1, 2-for-1 and 1-for-2 by default, 2-for-2 with two_for_two. With
    must_get, every deal brings him in (alone or with one of the
    FINDER_FILLERS players his owner rates lowest).
    """
    lg, pv = model.lg, model.perception
    if me not in lg.rosters:
        return []
    j = lg.teams.index(me)
    k_cat = lg.cats.index(cat) if cat in lg.cats else 0
    scored = np.array([c not in punt for c in lg.cats])

    def my_score(res: TradeResult) -> float:
        if mode == "cat":
            return float(res.me.by_cat[k_cat] + 0.05 * res.me.cats)
        if mode == "punt":
            return float(res.me.by_cat[scored].sum())
        return res.gap

    if must_get is not None:
        partners = [lg.owner[must_get]]
    out: list[Deal] = []
    for b in lg.teams:
        if b == me or (partners is not None and b not in partners):
            continue
        mine, theirs = list(lg.rosters[me]), list(lg.rosters[b])
        offer = sorted(mine, key=lambda x: -pv.pv(b, x))[:FINDER_POOL]
        gives = [(x,) for x in offer] + _pairs(offer)
        if must_get is not None:
            fillers = sorted((x for x in theirs if x != must_get), key=lambda x: pv.pv(b, x))[:FINDER_FILLERS]
            gets = [(must_get,)] + [(must_get, x) for x in fillers]
        else:
            want = sorted(theirs, key=lambda x: -lg.worth[x, j])[:FINDER_POOL]
            gets = [(x,) for x in want] + _pairs(want)
        for give in gives:
            for get in gets:
                if len(give) == len(get) == 2 and not two_for_two:
                    continue
                look = pv.accept(b, give, get)
                if look.score < min_look:
                    continue
                res = evaluate(lg, me, give, b, get)
                if res.me.cats <= MIN_GAIN or res.them.cats < -veto_cap or not res.fits:
                    continue
                out.append(Deal(b, give, get, res, look, my_score(res)))
    out.sort(key=lambda d: -d.score)
    seen: dict[tuple, int] = {}
    kept = []
    for d in out:
        key = tuple(sorted(d.get))
        if seen.get(key, 0) >= VARIETY:
            continue
        seen[key] = seen.get(key, 0) + 1
        kept.append(d)
        if len(kept) >= limit:
            break
    return kept


def board_ids(model: TradeModel, positions: Sequence[int | None]) -> list:
    """Player ids for league positions (None stays None)."""
    keys = player_key(model.board).to_numpy(object)
    return [keys[p] if p is not None and p >= 0 else None for p in positions]
