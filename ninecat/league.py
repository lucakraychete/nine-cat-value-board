"""League-level models: team category totals, and what each player is worth
to every team in real head-to-head category wins.

Everything is built on the matchup model (matchup.py) and a valued board
(build_board), and is pure: board in, arrays and frames out.

Roster moves follow two rules throughout:

- Adding a player means dropping the lowest-value player who shares a
  position with him (else the lowest-value player overall).
- Losing a player means picking up the best free agent who can fill his
  position (else the best free agent overall).

League impact for a player is, for every other team, the change in its
expected category wins over the season (20 weeks against every other
team's current roster) if it added him under the first rule; his own
team's number compares keeping him with replacing him under the second.
Impact counts real wins over every scored category, so punts don't
change it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

from ninecat.config import NEGATIVE_CATS, PERCENT_CATS
from ninecat.matchup import GAMES_PER_WEEK, WEEKS, Change, Outlook, StatMatrix, Week, change, outlook

FREE_AGENT = "FA"
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
ACTIVE_PLAYERS = 10     # team category totals use each team's top 10 by value
LOCK, SWING = 0.7, 0.3  # category bands: win 70%+ of weeks, 30-70%, under 30%


def position_sets(board: pd.DataFrame) -> list[frozenset[str]]:
    elig = board["eligible"] if "eligible" in board else pd.Series("", index=board.index)
    pos = board["position"] if "position" in board else pd.Series("", index=board.index)
    return [frozenset((e if isinstance(e, str) and e else (p if isinstance(p, str) else "")).split("/")) - {""}
            for e, p in zip(elig, pos)]


# ---------------------------------------------------------------- team totals

def team_category_totals(board: pd.DataFrame, cats: Sequence[str], teams: Sequence[str],
                         active: int = ACTIVE_PLAYERS) -> pd.DataFrame:
    """Per-game category totals of each team's top `active` players by value,
    with a league rank per category (1 = best; fewest turnovers is best).

    Percentages are makes over attempts across those players. Returns one
    row per team: n, each category, and r_<cat> ranks.
    """
    rows = []
    for team in teams:
        mine = board[board["fantasy_team"] == team].sort_values("value", ascending=False).head(active)
        row = {"team": team, "n": len(mine)}
        for c in cats:
            if c in PERCENT_CATS:
                made, att = PERCENT_CATS[c]
                a = mine[att].sum()
                row[c] = mine[made].sum() / a if a > 0 else np.nan
            else:
                row[c] = mine[c].sum()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("team")
    for c in cats:
        out[f"r_{c}"] = out[c].rank(ascending=c in NEGATIVE_CATS, method="first").astype(int)
    return out


def average_rank(totals: pd.DataFrame, cats: Sequence[str]) -> pd.Series:
    return totals[[f"r_{c}" for c in cats]].mean(axis=1) if cats else pd.Series(np.nan, index=totals.index)


# ---------------------------------------------------------------- the league model

@dataclass
class League:
    """The matchup model for one valued board.

    Player arrays are indexed by board position (0..n-1). `worth[i, j]` is
    the season category wins player i is worth to team j (for his own team:
    vs the best free agent at his position); `worth_by_cat` breaks it down.
    `games` is each player's games in a typical week (one number for
    everyone, or an array by board position) and `season_weeks` the weeks
    every change is counted over; trades.py sets both to the season left.
    """
    cats: list[str]
    teams: list[str]
    stats: StatMatrix
    owner: np.ndarray
    names: np.ndarray
    injury: np.ndarray
    positions: list[frozenset[str]]
    rosters: dict[str, np.ndarray]
    weeks: dict[str, Week]
    base: dict[str, Outlook]
    cat_rank: pd.DataFrame
    games: np.ndarray | float = GAMES_PER_WEEK
    season_weeks: float = WEEKS
    worth: np.ndarray = field(default=None)
    worth_by_cat: np.ndarray = field(default=None)
    impact: np.ndarray = field(default=None)
    impact_by_cat: np.ndarray = field(default=None)
    impact_rank: np.ndarray = field(default=None)
    vs_repl: np.ndarray = field(default=None)
    vs_repl_by_cat: np.ndarray = field(default=None)
    repl: np.ndarray = field(default=None)
    pos_impact_rank: dict[str, np.ndarray] = field(default=None)

    # ---- roster rules
    def shares_position(self, a: int, b: int) -> bool:
        return bool(self.positions[a] & self.positions[b])

    def drop_for(self, roster: Sequence[int], incoming: int, keep: Sequence[int] = ()) -> int | None:
        """Who leaves when `incoming` arrives: the lowest-value player sharing a position."""
        cands = [x for x in roster if x != incoming and x not in keep]
        same = [x for x in cands if self.shares_position(x, incoming)]
        pool = same or cands
        return min(pool, key=lambda x: self.stats.value[x]) if pool else None

    def best_fa_at(self, r: int, exclude: Sequence[int] = ()) -> int | None:
        """The best free agent who can fill r's position (else the best overall)."""
        fas = [x for x in self._fa_by_value if x != r and x not in exclude]
        same = [x for x in fas if self.shares_position(x, r)]
        pool = same or fas
        return pool[0] if pool else None

    @cached_property
    def _fa_by_value(self) -> list[int]:
        fa = np.flatnonzero(self.owner == FREE_AGENT)
        return list(fa[np.argsort(-self.stats.value[fa], kind="stable")])

    # ---- outlooks
    def outlook_with(self, team: str, roster: Sequence[int], weeks: dict[str, Week] | None = None,
                     games: np.ndarray | float | None = None) -> Outlook:
        weeks = dict(weeks or self.weeks)
        weeks[team] = self.stats.week(roster, self.games if games is None else games)
        return outlook(team, weeks, self.stats)

    def week(self, roster: Sequence[int]) -> Week:
        return self.stats.week(roster, self.games)

    def change(self, before: Outlook, after: Outlook) -> Change:
        return change(before, after, self.season_weeks)

    @property
    def weeks_text(self) -> str:
        """The season length for display: "20", or "18.7" for the season left."""
        w = self.season_weeks
        return f"{w:.0f}" if abs(w - round(w)) < 0.05 else f"{w:.1f}"

    def swap_weeks(self, team: str, incoming: int, outgoing: int | None) -> dict[str, Week]:
        """Team-weeks after `team` gains `incoming` and loses `outgoing`; in a
        trade, incoming's team receives outgoing."""
        weeks = dict(self.weeks)
        mine = [x for x in self.rosters[team] if x != outgoing] + [incoming]
        weeks[team] = self.week(mine)
        other = self.owner[incoming]
        if other in self.rosters and other != team:
            theirs = [x for x in self.rosters[other] if x != incoming] + ([outgoing] if outgoing is not None else [])
            weeks[other] = self.week(theirs)
        return weeks

    def ranks(self, weeks: dict[str, Week]) -> pd.DataFrame:
        return category_ranks(weeks, self.cats)

    def standings(self) -> pd.DataFrame:
        """Each team's average weekly categories won and matchup win chance, best first."""
        rows = [{"team": t, "win": o.avg_win, "cats": o.avg_cats} for t, o in self.base.items()]
        return pd.DataFrame(rows).sort_values("win", ascending=False).reset_index(drop=True)

    def band(self, team: str, cat: str) -> str:
        p = self.base[team].by_cat[self.cats.index(cat)]
        return "lock" if p >= LOCK else "swing" if p >= SWING else "weak"


def category_ranks(weeks: dict[str, Week], cats: Sequence[str]) -> pd.DataFrame:
    """League rank of each team's expected week in each category (1 = best)."""
    m = pd.DataFrame({t: w.m for t, w in weeks.items()}, index=list(cats)).T
    out = pd.DataFrame(index=m.index)
    for c in cats:
        out[c] = m[c].rank(ascending=c in NEGATIVE_CATS, method="first").astype(int)
    return out


def build_league(board: pd.DataFrame, cats: Sequence[str], teams: Sequence[str],
                 season_games: int = 82, impact: bool = True,
                 games: np.ndarray | float = GAMES_PER_WEEK, season_weeks: float = WEEKS) -> League:
    """Rosters, team-weeks and base outlooks; with impact=True also every
    player's worth to every team, league impact and value over replacement."""
    stats = StatMatrix(board, cats, season_games)
    owner = board["fantasy_team"].to_numpy(object)
    order = np.argsort(-stats.value, kind="stable")
    rosters = {t: order[owner[order] == t] for t in teams}
    weeks = {t: stats.week(rosters[t], games) for t in teams}
    league = League(
        cats=list(cats), teams=list(teams), stats=stats, owner=owner,
        names=board["name"].to_numpy(object),
        injury=board.get("injury_status", pd.Series("ACTIVE", index=board.index)).fillna("ACTIVE").to_numpy(object),
        positions=position_sets(board), rosters=rosters, weeks=weeks,
        base={t: outlook(t, weeks, stats) for t in teams},
        cat_rank=category_ranks(weeks, cats), games=games, season_weeks=season_weeks,
    )
    if impact:
        _league_impact(league)
    return league


def _league_impact(lg: League) -> None:
    n, T, k = len(lg.stats.value), len(lg.teams), len(lg.cats)
    worth, worth_by_cat = np.zeros((n, T)), np.zeros((n, T, k))
    imp_by_cat = np.zeros((n, k))
    best_fa = [lg.best_fa_at(r) for r in range(n)]
    for r in range(n):
        own = lg.owner[r]
        others = [t for t in lg.teams if t != own]
        for j, t in enumerate(lg.teams):
            roster = list(lg.rosters[t])
            if t == own:
                f = best_fa[r]
                new = [x for x in roster if x != r] + ([f] if f is not None else [])
            else:
                d = lg.drop_for(roster, r)
                new = [x for x in roster if x != d] + [r]
            ch = lg.change(lg.base[t], lg.outlook_with(t, new))
            if t == own:
                worth[r, j], worth_by_cat[r, j] = -ch.cats, -ch.by_cat
            else:
                worth[r, j], worth_by_cat[r, j] = ch.cats, ch.by_cat
                imp_by_cat[r] += ch.by_cat / len(others)
    lg.worth, lg.worth_by_cat = worth, worth_by_cat
    lg.impact_by_cat, lg.impact = imp_by_cat, imp_by_cat.sum(1)
    lg.impact_rank = pd.Series(-lg.impact).rank(method="first").to_numpy(int)
    # Impact over the best free agent at his position: what he adds that the wire can't.
    lg.repl = np.array([-1 if f is None else f for f in best_fa])
    repl_by_cat = np.where(lg.repl[:, None] >= 0, imp_by_cat[np.maximum(lg.repl, 0)], 0.0)
    lg.vs_repl_by_cat = imp_by_cat - repl_by_cat
    lg.vs_repl = lg.vs_repl_by_cat.sum(1)
    lg.pos_impact_rank = {}
    for p in POSITIONS:
        rank = np.full(n, np.nan)
        members = [i for i in range(n) if p in lg.positions[i]]
        for place, i in enumerate(sorted(members, key=lambda i: -lg.impact[i]), start=1):
            rank[i] = place
        lg.pos_impact_rank[p] = rank


# ---------------------------------------------------------------- waivers

@dataclass(frozen=True)
class DropOption:
    drop: int
    change: Change


def drop_options(lg: League, team: str, add: int) -> list[DropOption]:
    """Every possible drop when `add` joins `team`, best first (the drop that
    leaves the team winning the most categories over the season)."""
    base = lg.base[team]
    roster = list(lg.rosters[team])
    opts = [DropOption(d, lg.change(base, lg.outlook_with(team, [x for x in roster if x != d] + [add])))
            for d in roster if d != add]
    return sorted(opts, key=lambda o: -o.change.cats)


@dataclass(frozen=True)
class Pickup:
    player: int
    drop: int
    change: Change
    alt: int | None = None
    alt_cats: float | None = None
    il: int | None = None
    il_cats: float | None = None


@dataclass(frozen=True)
class DropCandidate:
    player: int
    best: int | None
    cats: float
    by_cat: np.ndarray | None


def waivers(lg: League, team: str) -> tuple[list[Pickup], list[DropCandidate], int | None]:
    """Rest of season: every free agent against every possible drop.

    Returns pickups ranked by their best drop (with the runner-up), each of
    the team's players with the best swap available for him, and the
    team's player listed OUT, if any. That player could go to an injured
    list spot instead, so "add without dropping" is reported alongside
    each pickup, never mixed into the ranking.
    """
    base = lg.base[team]
    roster = list(lg.rosters[team])
    out_player = next((x for x in roster if lg.injury[x] == "OUT"), None)
    best_swap = {d: [None, -np.inf, None] for d in roster}
    picks = []
    for f in np.flatnonzero(lg.owner == FREE_AGENT):
        opts = []
        for d in roster:
            ch = lg.change(base, lg.outlook_with(team, [x for x in roster if x != d] + [f]))
            opts.append((d, ch))
            if ch.cats > best_swap[d][1]:
                best_swap[d] = [f, ch.cats, ch.by_cat]
        if not opts:
            continue
        opts.sort(key=lambda o: -o[1].cats)
        il_cats = None
        if out_player is not None:
            il_cats = lg.change(base, lg.outlook_with(team, roster + [f])).cats
        alt = opts[1] if len(opts) > 1 else None
        picks.append(Pickup(int(f), opts[0][0], opts[0][1], alt[0] if alt else None, alt[1].cats if alt else None,
                            out_player if il_cats is not None else None, il_cats))
    picks.sort(key=lambda p: -p.change.cats)
    drops = sorted((DropCandidate(d, b, c, bc) for d, (b, c, bc) in best_swap.items()), key=lambda x: -x.cats)
    return picks, drops, out_player


# ---------------------------------------------------------------- team fit

BAND_LABEL = {"lock": "already wins", "swing": "swing", "weak": "weak"}


@dataclass(frozen=True)
class Fit:
    text: str
    good: bool
    gains: str
    losses: str


def fit_note(lg: League, by_cat: np.ndarray, team: str, threshold: float = 0.15) -> Fit:
    """How a category change lines up with where the team stands.

    A swing category (won 30-70% of weeks) is where extra production flips
    matchups; adding to a category the team already wins, or still loses,
    moves less.
    """
    cats = lg.cats
    d = 2 if threshold < 0.05 else 1
    gains = sorted([c for i, c in enumerate(cats) if by_cat[i] > threshold], key=lambda c: -by_cat[cats.index(c)])
    losses = sorted([c for i, c in enumerate(cats) if by_cat[i] < -threshold], key=lambda c: by_cat[cats.index(c)])
    swing = [c for c in gains if lg.band(team, c) == "swing"]
    net = float(by_cat.sum())
    if net < -2 * threshold:
        text = "Net loss: the costs outweigh " + (f"swing gains in {', '.join(swing)}" if swing else "the gains")
    elif not gains:
        text = "No meaningful category gain"
    elif swing:
        text = f"Fills swing: {', '.join(swing)}"
    elif all(lg.band(team, c) == "lock" for c in gains):
        text = "Piles onto categories already won"
    else:
        text = "Helps categories mostly lost"

    def fmt(c):
        return f"{by_cat[cats.index(c)]:+.{d}f}"

    gain_txt = ", ".join(f"{c} {fmt(c)} ({BAND_LABEL[lg.band(team, c)]}, #{lg.cat_rank.loc[team, c]})"
                         for c in gains[:3]) or "–"
    loss_txt = ", ".join(f"{c} {fmt(c)}" for c in losses[:2])
    return Fit(text, net >= 0 and bool(swing), gain_txt, loss_txt)


# ---------------------------------------------------------------- uncertainty

@dataclass(frozen=True)
class Uncertainty:
    low: float
    high: float
    label: str
    why: str


MODEL_RANGE_LABELS = ((0.15, "narrow"), (0.35, "moderate"))   # half-width over |impact|, else "wide"


def uncertainty(impact: float, last_games: float | None, injury: str | None,
                value_range: tuple[float, float] | None = None, slope: float | None = None,
                games: float | None = None) -> Uncertainty:
    """How much to trust his impact.

    With the trained model (value_range: its 80% range for his per-game value,
    as offsets; slope: impact per unit of per-game value across the board), the
    range is measured, not set by hand: impact + slope x each offset. Without
    it, rookies and players who missed last season ride on projections alone,
    so they get wider hand-set ranges.
    """
    if value_range is not None and slope is not None and all(np.isfinite(value_range)) and np.isfinite(slope):
        low, high = sorted((impact + slope * value_range[0], impact + slope * value_range[1]))
        half = (high - low) / 2 / max(abs(impact), 1.0)
        label = next((name for cut, name in MODEL_RANGE_LABELS if half < cut), "wide")
        at = f" at {int(games)} games played" if games is not None and np.isfinite(games) else ""
        return Uncertainty(low, high, label, f"The model's 80% range: how far players like him{at} and at his "
                                             "minutes landed from their projection in the backtests.")
    g = 0 if last_games is None or pd.isna(last_games) else int(last_games)
    if g == 0:
        share, label, why = 0.4, "wide", "No NBA stats last season: rookie, or missed the year. Projection only."
    elif g < 30:
        share, label, why = 0.25, "moderate", f"Only {g} games last season, so his rates are noisy."
    elif injury == "OUT":
        share, label, why = 0.25, "moderate", "Currently out, so the games he plays are uncertain."
    else:
        share, label, why = 0.12, "narrow", f"{g} games last season and a stable role."
    spread = max(0.6, abs(impact) * share)
    return Uncertainty(impact - spread, impact + spread, label, why)
