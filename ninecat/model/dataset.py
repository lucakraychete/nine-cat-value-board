"""Checkpoints for the rest-of-season model, built from nba.com game logs.

For each season and each player, cut his season after his k-th game (k in
CHECKPOINTS). Everything known at the cut is a feature; his per-game line
over the rest of that season is the target. A checkpoint needs at least
MIN_REMAINING games after the cut.

Features never use a game on or after the cut:
- cur_*: his first k games (season to date); l5_* and l10_*: the last 5
  and 10 of those (fewer when k is smaller);
- p1_* and p2_*: last season and the season before, from their logs;
- age (nba.com's, for the season), team_games (the league's average team
  games played before the cut day) and games_share = k / team_games.

Windows are stored as totals (G, MIN, each stat, GS) so rates come out
exactly: per-game minutes, per-minute rates, and FG% / FT% as makes over
attempts (window_rates()). Targets use the y_ prefix the same way.

nba.com's game logs don't say who started, so GS counts games where he was
among his team's top five in minutes (started()).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ninecat.config import PERCENT_CATS

CHECKPOINTS = (3, 5, 10, 20, 40)
MIN_REMAINING = 5
STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
# Modeled per minute; FGM and FTM come back through FG% and FT% (makes over attempts).
RATE_STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGA", "FTA"]
PCT_STATS = {c: tuple(PERCENT_CATS[c]) for c in ("FG%", "FT%")}
TOTALS = ["MIN", *STATS, "GS"]
WINDOWS = ("cur", "l5", "l10", "p1", "p2")

TRAIN = tuple(f"{y}-{str(y + 1)[2:]}" for y in range(2017, 2024))
VALIDATE = ("2024-25",)
TEST = ("2025-26",)


def season_before(season: str, n: int = 1) -> str:
    y = int(season[:4]) - n
    return f"{y}-{str(y + 1)[2:]}"


def started(logs: pd.DataFrame, starters: int = 5) -> pd.Series:
    """1 when he was among his team's top `starters` in minutes that game (ties by order)."""
    rank = logs.groupby(["game_id", "team"])["MIN"].rank(ascending=False, method="first")
    return (rank <= starters).astype(float)


def team_games_before(logs: pd.DataFrame) -> pd.Series:
    """The league's average team games played before each day, by date (as YYYY-MM-DD)."""
    per_day = logs.drop_duplicates(["game_id", "team"]).groupby("date").size().sort_index()
    teams = logs["team"].nunique()
    return (per_day.cumsum() - per_day) / max(teams, 1)


def season_length(logs: pd.DataFrame | None, default: float = 82.0) -> float:
    """Games per team in a season's logs (82, except the shortened 2019-20 and 2020-21)."""
    if logs is None or logs.empty:
        return default
    return float(logs.drop_duplicates(["game_id", "team"]).groupby("team").size().mean())


def season_totals(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per player: G and the TOTALS over the season, by nba_id."""
    g = logs.assign(GS=started(logs))
    out = g.groupby("nba_id")[TOTALS].sum()
    out.insert(0, "G", g.groupby("nba_id").size())
    return out


def _prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.add_prefix(f"{prefix}_")


def checkpoints(logs: pd.DataFrame, prev: pd.DataFrame | None = None, prev2: pd.DataFrame | None = None,
                ages: pd.Series | None = None, ks: Sequence[int] = CHECKPOINTS,
                min_remaining: int = MIN_REMAINING, prev_lengths: tuple[float, float] = (82.0, 82.0)
                ) -> pd.DataFrame:
    """Every checkpoint in one season's logs.

    prev and prev2 are season_totals() of the two seasons before (by
    nba_id), prev_lengths their games per team; ages maps nba_id -> age.
    One row per player and k.
    """
    g = logs.sort_values(["nba_id", "date", "game_id"], kind="stable").reset_index(drop=True)
    g["GS"] = started(g)
    by = g.groupby("nba_id", sort=False)
    g["i"] = by.cumcount()
    n_games = by["MIN"].transform("size")
    cum = by[TOTALS].cumsum()
    total = by[TOTALS].transform("sum")
    next_date = by["date"].shift(-1)
    before = team_games_before(g)

    rows = []
    for k in ks:
        at = (g["i"] == k - 1) & (n_games >= k + min_remaining)
        if not at.any():
            continue
        idx = g.index[at]
        row = pd.DataFrame({"nba_id": g.loc[idx, "nba_id"].to_numpy(), "name_key": g.loc[idx, "name_key"].to_numpy(),
                            "season": g.loc[idx, "season"].to_numpy(), "k": k,
                            "cut_date": next_date.loc[idx].to_numpy()})
        row["cur_G"] = float(k)
        row[[f"cur_{c}" for c in TOTALS]] = cum.loc[idx].to_numpy()
        for w in (5, 10):
            lag = cum.groupby(g["nba_id"]).shift(w).loc[idx].fillna(0.0)
            row[f"l{w}_G"] = float(min(w, k))
            row[[f"l{w}_{c}" for c in TOTALS]] = cum.loc[idx].to_numpy() - lag.to_numpy()
        row["y_G"] = (n_games.loc[idx] - k).to_numpy(float)
        row[[f"y_{c}" for c in TOTALS]] = total.loc[idx].to_numpy() - cum.loc[idx].to_numpy()
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    _attach_context(out, prev, prev2, ages, prev_lengths)
    out["team_games"] = out["cut_date"].map(before).astype(float)
    out["games_share"] = out["cur_G"] / out["team_games"].where(out["team_games"] > 0)
    out["season_team_games"] = season_length(g)
    out["team_left"] = (out["season_team_games"] - out["team_games"]).clip(lower=0)
    return out


def _attach_context(out: pd.DataFrame, prev, prev2, ages, prev_lengths) -> None:
    """Prior seasons' totals and lengths, and age, by nba_id."""
    for name, frame in (("p1", prev), ("p2", prev2)):
        cols = [f"{name}_G", *[f"{name}_{c}" for c in TOTALS]]
        if frame is None or frame.empty:
            out[cols] = np.nan
        else:
            out[cols] = _prefixed(frame[["G", *TOTALS]], name).reindex(out["nba_id"]).to_numpy()
    out["p1_team_games"], out["p2_team_games"] = prev_lengths
    out["age"] = out["nba_id"].map(ages) if ages is not None else np.nan


def current_rows(logs: pd.DataFrame, prev: pd.DataFrame | None, prev2: pd.DataFrame | None,
                 ages: pd.Series | None = None, prev_lengths: tuple[float, float] = (82.0, 82.0),
                 season_games: float = 82.0, players: Sequence[int] = ()) -> pd.DataFrame:
    """One row per player as of now, shaped like a checkpoint without a target.

    Everyone with a game this season, plus `players` (nba ids) who have
    none yet (cur_G = 0). k is his games so far.
    """
    g = logs.sort_values(["nba_id", "date", "game_id"], kind="stable").reset_index(drop=True)
    if not g.empty:
        g["GS"] = started(g)
    ids = list(dict.fromkeys([*g["nba_id"].tolist(), *players]))
    out = pd.DataFrame({"nba_id": ids})
    by = g.groupby("nba_id")
    out["name_key"] = out["nba_id"].map(by["name_key"].last()) if not g.empty else None
    cur = by[TOTALS].sum().reindex(ids).fillna(0.0)
    out["cur_G"] = by.size().reindex(ids).fillna(0).to_numpy(float)
    out[[f"cur_{c}" for c in TOTALS]] = cur.to_numpy()
    for w in (5, 10):
        last = g.groupby("nba_id").tail(w).groupby("nba_id")
        out[f"l{w}_G"] = last.size().reindex(ids).fillna(0).to_numpy(float)
        out[[f"l{w}_{c}" for c in TOTALS]] = last[TOTALS].sum().reindex(ids).fillna(0.0).to_numpy()
    out["k"] = out["cur_G"]
    _attach_context(out, prev, prev2, ages, prev_lengths)
    played = g.drop_duplicates(["game_id", "team"]).groupby("team").size() if not g.empty else pd.Series(dtype=float)
    out["team_games"] = float(played.mean()) if len(played) else 0.0
    out["games_share"] = out["cur_G"] / out["team_games"].where(out["team_games"] > 0)
    out["season_team_games"] = season_games
    out["team_left"] = (season_games - out["team_games"]).clip(lower=0)
    return out


def build_dataset(logs: Mapping[str, pd.DataFrame], totals: Mapping[str, pd.DataFrame] | None = None,
                  seasons: Sequence[str] | None = None, ks: Sequence[int] = CHECKPOINTS) -> pd.DataFrame:
    """Checkpoints for every season in `seasons` (default: every season in logs).

    logs and totals map season label -> that season's history frames
    (totals only supplies age). Seasons without logs for the year before
    get NaN priors, as do players who didn't play that year.
    """
    seasons = list(seasons or sorted(logs))
    summary = {s: season_totals(f) for s, f in logs.items() if f is not None and not f.empty}
    parts = []
    for s in seasons:
        if s not in logs or logs[s] is None or logs[s].empty:
            continue
        t = (totals or {}).get(s)
        ages = t.drop_duplicates("nba_id").set_index("nba_id")["age"] if t is not None and "age" in t else None
        lengths = (season_length(logs.get(season_before(s))), season_length(logs.get(season_before(s, 2))))
        parts.append(checkpoints(logs[s], summary.get(season_before(s)), summary.get(season_before(s, 2)), ages, ks,
                                 prev_lengths=lengths))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def split_of(season: str) -> str:
    return "train" if season in TRAIN else "validate" if season in VALIDATE else "test" if season in TEST else "other"


# ---------------------------------------------------------------- rates and lines

def window_rates(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-game minutes (MPG), per-minute RATE_STATS and FG%/FT% for one window.

    No minutes gives NaN rates; no attempts gives a NaN percentage.
    """
    out = pd.DataFrame(index=df.index)
    g = df[f"{prefix}_G"].astype(float)
    m = df[f"{prefix}_MIN"].astype(float)
    out["MPG"] = m / g.where(g > 0)
    for s in RATE_STATS:
        out[s] = df[f"{prefix}_{s}"] / m.where(m > 0)
    for c, (made, att) in PCT_STATS.items():
        a = df[f"{prefix}_{att}"].astype(float)
        out[c] = df[f"{prefix}_{made}"] / a.where(a > 0)
    return out


def per_game(rates: pd.DataFrame) -> pd.DataFrame:
    """Back to a per-game line: each rate x minutes, makes = percentage x attempts."""
    out = pd.DataFrame(index=rates.index)
    out["MIN"] = rates["MPG"]
    for s in RATE_STATS:
        out[s] = rates["MPG"] * rates[s]
    for c, (made, att) in PCT_STATS.items():
        out[made] = rates[c] * out[att]
        out[c] = rates[c]
    return out


def target_line(df: pd.DataFrame) -> pd.DataFrame:
    """What actually happened over the rest of the season, per game."""
    return per_game(window_rates(df, "y"))
