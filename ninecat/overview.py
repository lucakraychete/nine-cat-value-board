"""The Players overview: what each player was expected to add, has added so
far, and should add over the rest of the season, all in one unit.

The unit is category wins added (CWA): league impact from league.py, the
change in a team's expected category wins over the season (WEEKS weeks of
GAMES_PER_WEEK games against every other roster), averaged over the teams
that could add him. Games enter impact only through availability,
min(GP / season games, 1).

- Pre: impact with everyone on the preseason ESPN projection (the
  earliest snapshot on or before opening night), availability from its GP.
- So far: impact with everyone on this season's per-game line at full
  availability (a full-season rate), times his games played / season games.
- ROS (interim): impact on ros_line() with availability from the
  projection, times his NBA team's remaining games / season games.

Each total sums the per-category numbers over the scored (non-punted)
categories; punted categories are still reported per category. Every
number is pure: frames in, frames out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ninecat.board import BOARD_MIN_MINUTES, player_key
from ninecat.config import NEGATIVE_CATS, PERCENT_CATS, ROS_PRIOR_GAMES
from ninecat.league import build_league
from ninecat.lines import ESPN_TOTAL_PREFIX, LINE_STATS, SEASON_PREFIX, stat_line
from ninecat.valuation import value_players

LINE_COLS = [*LINE_STATS, "GP"]


@dataclass(frozen=True)
class Preseason:
    """The snapshot used as the preseason line, and whether it predates opening night."""
    day: date | None
    before_opening: bool


def preseason_snapshot(days: Iterable[date], opening: date | None) -> Preseason:
    """The earliest snapshot on or before opening night, else the earliest one."""
    days = sorted(days)
    if not days:
        return Preseason(None, False)
    early = [d for d in days if opening is not None and d <= opening]
    return Preseason(early[0], True) if early else Preseason(days[0], False)


# ---------------------------------------------------------------- lines

def games_played(players: pd.DataFrame) -> pd.Series:
    """Games this season: nba.com's count, else ESPN's, else 0."""
    nba = pd.to_numeric(players.get(SEASON_PREFIX + "G"), errors="coerce")
    espn = pd.to_numeric(players.get(ESPN_TOTAL_PREFIX + "GP"), errors="coerce")
    nba = nba if nba is not None else pd.Series(np.nan, index=players.index)
    espn = espn if espn is not None else pd.Series(np.nan, index=players.index)
    return nba.where(nba > 0, espn).fillna(0.0).astype(float)


def _with_line(players: pd.DataFrame, line: pd.DataFrame) -> pd.DataFrame:
    out = players.copy()
    out[LINE_COLS] = line[LINE_COLS].to_numpy(float)
    return out


def season_rate_line(players: pd.DataFrame, season_games: int) -> pd.DataFrame:
    """This season's per-game line (nba.com's S_, else ESPN's T_) at full availability."""
    use_nba = pd.to_numeric(players.get(SEASON_PREFIX + "G"), errors="coerce").fillna(0) > 0
    line = pd.DataFrame(index=players.index)
    for s in LINE_STATS:
        nba = pd.to_numeric(players.get(SEASON_PREFIX + s), errors="coerce")
        espn = pd.to_numeric(players.get(ESPN_TOTAL_PREFIX + s), errors="coerce")
        nba = nba if nba is not None else pd.Series(np.nan, index=players.index)
        espn = espn if espn is not None else pd.Series(np.nan, index=players.index)
        line[s] = nba.where(use_nba, espn)
    line["GP"] = float(season_games)
    line.loc[games_played(players) <= 0, LINE_STATS] = np.nan
    return _with_line(players, line)


def snapshot_line(players: pd.DataFrame, snapshot: pd.DataFrame | None) -> pd.DataFrame:
    """Players with their snapshot projection (by player_id); NaN where he isn't in it."""
    line = pd.DataFrame(np.nan, index=players.index, columns=LINE_COLS)
    if snapshot is not None and not snapshot.empty:
        snap = snapshot.drop_duplicates("player_id").set_index("player_id")
        found = snap.reindex(players["player_id"])
        for c in LINE_COLS:
            if c in found:
                line[c] = pd.to_numeric(found[c], errors="coerce").to_numpy(float)
    return _with_line(players, line)


def ros_line(players: pd.DataFrame, preseason: pd.DataFrame | None, k: float = ROS_PRIOR_GAMES) -> pd.DataFrame:
    """Interim rest-of-season per-game line (PLAN.md Part 5 replaces this).

    Each stat, makes and attempts included, is w x this season + (1 - w) x
    the preseason projection, with w = GP / (GP + k) and GP his games this
    season. Without a preseason projection the prior is the current ESPN
    projection (else last season, as the "proj" source does); with no prior
    at all, this season alone. GP (availability) is the projection's.
    """
    gp = games_played(players)
    w = gp / (gp + k)
    pre = snapshot_line(players, preseason)
    current = stat_line(players, "proj")
    prior = pre[LINE_STATS].where(pre[LINE_STATS].notna().all(axis=1), current[LINE_STATS], axis=0)
    now = season_rate_line(players, 82)[LINE_STATS]
    line = pd.DataFrame(index=players.index)
    for s in LINE_STATS:
        mixed = w * now[s] + (1 - w) * prior[s]
        line[s] = mixed.where(prior[s].notna() & now[s].notna(), prior[s].where(prior[s].notna(), now[s]))
    line["GP"] = pre["GP"].where(pre["GP"].notna(), current["GP"])
    return _with_line(players, line)


# ---------------------------------------------------------------- impact

def impact_by_category(line: pd.DataFrame, cats: Sequence[str], teams: Sequence[str], punt: Sequence[str],
                       pool_size: int, season_games: int, games_discount: bool,
                       min_minutes: float = BOARD_MIN_MINUTES) -> pd.DataFrame:
    """League impact per category for everyone with a line, indexed by player key.

    The league is valued on this one line for everybody, so every player is
    measured against teams built from the same kind of numbers.
    """
    has = line[LINE_STATS].notna().all(axis=1) & (line["MIN"].fillna(0) >= min_minutes)
    rows = line[has]
    if len(rows) < 2:
        return pd.DataFrame(columns=list(cats), dtype=float)
    valued = value_players(rows, cats, punt=punt, pool_size=pool_size, season_games=season_games,
                           games_discount=games_discount)
    lg = build_league(valued, cats, teams, season_games=season_games)
    return pd.DataFrame(lg.impact_by_cat, columns=list(cats), index=player_key(valued).to_numpy())


def overview(players: pd.DataFrame, board: pd.DataFrame, cats: Sequence[str], teams: Sequence[str],
             punt: Sequence[str] = (), pool_size: int = 130, season_games: int = 82,
             games_discount: bool = True, preseason: pd.DataFrame | None = None,
             remaining: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Pre, so far and ROS CWA (totals over scored categories, and ranks) and
    the per-category so far, one row per board row in board order.

    remaining maps NBA team -> games left; without it ROS is a full season.
    """
    cats, punt = list(cats), list(punt)
    scored = [c for c in cats if c not in punt]
    keys = player_key(board).to_numpy()
    p = players.drop_duplicates(subset="player_id" if "player_id" in players else "name")
    args = (cats, teams, punt, pool_size, season_games)

    gp = games_played(p).set_axis(player_key(p).to_numpy())
    gp = gp[~gp.index.duplicated()].reindex(keys).fillna(0.0)
    rate = impact_by_category(season_rate_line(p, season_games), *args, games_discount=False, min_minutes=0)
    so_far = rate.reindex(keys).fillna(0.0).mul((gp / season_games).to_numpy(), axis=0)

    pre = impact_by_category(snapshot_line(p, preseason), *args, games_discount=games_discount).reindex(keys)

    ros_rate = impact_by_category(ros_line(p, preseason), *args, games_discount=games_discount).reindex(keys)
    team = board["pro_team"].to_numpy(object)
    left = np.array([remaining.get(t, np.nan) if remaining else season_games for t in team], dtype=float)
    ros = ros_rate.mul(np.clip(left / season_games, 0, 1), axis=0)

    out = pd.DataFrame(index=board.index)
    out["gp"] = gp.to_numpy()
    out["so_far"] = so_far[scored].sum(axis=1).to_numpy()
    out["pre"] = pre[scored].sum(axis=1, min_count=1).to_numpy() if scored else np.nan
    out["ros"] = ros[scored].sum(axis=1, min_count=1).to_numpy() if scored else np.nan
    for c in cats:
        out[f"sf_{c}"] = so_far[c].to_numpy()
    for col in ("so_far", "pre", "ros"):
        out[f"{col}_rank"] = out[col].rank(ascending=False, method="first").astype("Int64")
    out["rank_delta"] = out["pre_rank"] - out["so_far_rank"]
    return out


# ---------------------------------------------------------------- game logs

def recent_minutes(logs: pd.DataFrame, asof, days: int) -> pd.Series:
    """Average minutes per game by name_key over the `days` days ending `asof`."""
    if logs is None or logs.empty:
        return pd.Series(dtype=float)
    d = pd.to_datetime(logs["date"])
    end = pd.Timestamp(asof).normalize()
    recent = logs[(d > end - pd.Timedelta(days=days)) & (d <= end)]
    return recent.groupby("name_key")["MIN"].mean()


def game_values(logs: pd.DataFrame, board: pd.DataFrame, cats: Sequence[str], punt: Sequence[str] = (),
                last: int = 10) -> pd.Series:
    """Each player's last `last` games as per-game value (sum of category z-scores).

    Every game is scored against the board's pool on the per-game scale, the
    same way value_players scores a season line, so a typical game by a
    replacement-level player sits near the pool's average. By name_key.
    """
    if logs is None or logs.empty:
        return pd.Series(dtype=object)
    pool = board[board["in_pool"]] if "in_pool" in board else board
    g = logs.sort_values("date", kind="stable")
    total = pd.Series(0.0, index=g.index)
    for c in cats:
        if c in punt:
            continue
        if c in PERCENT_CATS:
            made, att = PERCENT_CATS[c]
            lp = pool[made].sum() / pool[att].sum() if pool[att].sum() > 0 else 0.0
            ref, x = pool[made] - lp * pool[att], g[made] - lp * g[att]
        else:
            ref, x = pool[c].astype(float), g[c].astype(float)
        sd = ref.std(ddof=0)
        z = (x - ref.mean()) / sd if np.isfinite(sd) and sd > 0 else 0.0 * x
        total += -z if c in NEGATIVE_CATS else z
    g = g.assign(value=total.round(2))
    return g.groupby("name_key")["value"].apply(lambda v: v.tail(last).tolist())


# ---------------------------------------------------------------- schedule windows

def week_windows(league_schedule: pd.DataFrame, current_period: int | None, today: date
                 ) -> tuple[tuple[date, date], tuple[date, date]]:
    """This week's and next week's days: the current and next ESPN matchup
    periods when known, else Monday-Sunday of this week and the next."""
    def period(p):
        if league_schedule is None or league_schedule.empty or p is None:
            return None
        rows = league_schedule[league_schedule["period"] == p]
        if rows.empty:
            return None
        return pd.Timestamp(rows["start"].iloc[0]).date(), pd.Timestamp(rows["end"].iloc[0]).date()

    monday = today - timedelta(days=today.weekday())
    this = period(current_period) or (monday, monday + timedelta(days=6))
    after = period(current_period + 1 if current_period else None)
    nxt = after or (this[1] + timedelta(days=1), this[1] + timedelta(days=7))
    return this, nxt


def playoff_windows(league_schedule: pd.DataFrame) -> list[tuple[date, date]]:
    """Start and end of each fantasy-playoff matchup period."""
    if league_schedule is None or league_schedule.empty:
        return []
    rows = league_schedule[league_schedule["playoff"].astype(bool)].drop_duplicates("period")
    return [(pd.Timestamp(s).date(), pd.Timestamp(e).date()) for s, e in zip(rows["start"], rows["end"])]
