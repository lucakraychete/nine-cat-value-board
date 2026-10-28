"""Testing against the real world from now on (PLAN.md 5.6). Pure: frames in, frames out.

- score_snapshots(): every saved projection snapshot (the model's daily
  model_proj_*.csv and ESPN's espn_proj_*.csv) against what each player has
  actually done in this season's games after that day. Days where both
  exist are scored on the players both cover, so the two compare fairly.
- The matchup log: each refresh records, for the current matchup period,
  the matchup model's win chance for every team (once, the first time it
  sees the period; `pre_week` says whether that was before the period
  started) and the live category score; when the period is over the
  result is final. brier() and calibration() grade those win chances.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

from ninecat.league import build_league
from ninecat.lines import MODEL_PREFIX, name_key
from ninecat.matchup import matchup
from ninecat.model.evaluate import EVAL_CATS
from ninecat.schedule import games_between

MIN_GAMES_AFTER = 3
LOG_COLUMNS = ["period", "team_id", "opponent_id", "p_win", "logged_at", "pre_week", "wins", "losses", "ties",
               "final", "result"]
PROB_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


# ---------------------------------------------------------------- projections vs results

def _projection(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    """A snapshot as name_key -> per-game line with FG% and FT%."""
    if kind == "model":
        cols = {c[len(MODEL_PREFIX):]: frame[c] for c in frame.columns
                if c.startswith(MODEL_PREFIX) and c[len(MODEL_PREFIX):] in
                ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA")}
        out = pd.DataFrame(cols)
        out["name_key"] = frame["name_key"]
    else:
        out = frame[["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]].copy()
        out["name_key"] = frame["name"].map(name_key)
    out = out.drop_duplicates("name_key").set_index("name_key").astype(float)
    out["FG%"] = out["FGM"] / out["FGA"].where(out["FGA"] > 0)
    out["FT%"] = out["FTM"] / out["FTA"].where(out["FTA"] > 0)
    return out


def actual_since(logs: pd.DataFrame, day: date, min_games: int = MIN_GAMES_AFTER) -> pd.DataFrame:
    """Per-game line over games after `day`, for players with min_games or more; plus G and total minutes."""
    after = logs[pd.to_datetime(logs["date"]).dt.date > day]
    g = after.groupby("name_key")
    tot = g[["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]].sum()
    n = g.size()
    tot = tot[n >= min_games]
    out = tot.div(n[n >= min_games], axis=0)
    out["FG%"] = tot["FGM"] / tot["FGA"].where(tot["FGA"] > 0)
    out["FT%"] = tot["FTM"] / tot["FTA"].where(tot["FTA"] > 0)
    out["G"], out["minutes"] = n[n >= min_games], tot["MIN"]
    return out


def score_snapshots(snapshots: Iterable[tuple[str, date, pd.DataFrame]], logs: pd.DataFrame,
                    min_games: int = MIN_GAMES_AFTER) -> pd.DataFrame:
    """Minutes-weighted MAE per category for each snapshot. snapshots: (kind, day, frame), kind 'model' or 'espn'."""
    by_day: dict[date, dict[str, pd.DataFrame]] = {}
    for kind, day, frame in snapshots:
        by_day.setdefault(day, {})[kind] = _projection(kind, frame)
    rows = []
    for day, projs in sorted(by_day.items()):
        actual = actual_since(logs, day, min_games) if logs is not None and not logs.empty else pd.DataFrame()
        if actual.empty:
            continue
        common = actual.index
        for p in projs.values():
            common = common.intersection(p.index)
        for kind, p in projs.items():
            keys = common if len(projs) > 1 else actual.index.intersection(p.index)
            if not len(keys):
                continue
            a, q, w = actual.loc[keys], p.loc[keys], actual.loc[keys, "minutes"]
            row = {"source": kind, "day": day.isoformat(), "players": len(keys),
                   "games_after": float(a["G"].mean())}
            for c in EVAL_CATS:
                ok = a[c].notna() & q[c].notna()
                row[c] = float(np.average((q[c] - a[c]).abs()[ok], weights=w[ok])) if ok.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the matchup log

def matchup_probabilities(board: pd.DataFrame, cats: list[str], team_names: dict[int, str], live: pd.DataFrame,
                          start: date, end: date, schedule: pd.DataFrame) -> dict[int, float]:
    """The matchup model's chance for each team to win this period's matchup: each
    player's games are his NBA team's games in the period (none if listed OUT)."""
    lg = build_league(board, cats, list(team_names.values()), impact=False)
    per_team = games_between(schedule, start, end) if schedule is not None and not schedule.empty else pd.Series()
    games = np.array(board["pro_team"].map(per_team).fillna(0), dtype=float)   # writable copy
    games[lg.injury == "OUT"] = 0
    weeks = {t: lg.stats.week(lg.rosters[t], games) for t in lg.teams}
    out = {}
    for _, r in live.iterrows():
        a, b = team_names.get(int(r["team_id"])), team_names.get(int(r["opponent_id"]))
        if a in weeks and b in weeks:
            out[int(r["team_id"])] = float(matchup(weeks[a], weeks[b], lg.stats)[2])
    return out


def update_matchup_log(log: pd.DataFrame | None, live: pd.DataFrame, probs: dict[int, float],
                       current_period: int | None, period_start: date | None, now: str) -> pd.DataFrame:
    """The log after one refresh.

    live is live_matchup.csv (the current period's W-L-T per team), probs the
    model's win chance per team_id for the current period, now an ISO time.
    """
    log = (log.copy() if log is not None and not log.empty else pd.DataFrame(columns=LOG_COLUMNS))
    if current_period is not None:
        # periods before the current one are over: their last score is final
        done = (log["period"].astype(int) < int(current_period)) & ~log["final"].astype(bool)
        log.loc[done, "final"] = True
        w, lo = log.loc[done, "wins"].astype(float), log.loc[done, "losses"].astype(float)
        log.loc[done, "result"] = np.where(w > lo, 1.0, np.where(w < lo, 0.0, 0.5))
    if live is None or live.empty or current_period is None:
        return log[LOG_COLUMNS]
    live = live[live["period"] == current_period]
    pre = period_start is not None and date.fromisoformat(now[:10]) < period_start
    for _, r in live.iterrows():
        at = (log["period"].astype(int) == int(current_period)) & (log["team_id"].astype(int) == int(r["team_id"]))
        if not at.any():
            new = {"period": int(current_period), "team_id": int(r["team_id"]), "opponent_id": r["opponent_id"],
                   "p_win": probs.get(int(r["team_id"]), np.nan), "logged_at": now, "pre_week": pre,
                   "wins": r["wins"], "losses": r["losses"], "ties": r["ties"], "final": False, "result": np.nan}
            log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
        else:
            log.loc[at, ["wins", "losses", "ties"]] = [r["wins"], r["losses"], r["ties"]]
    return log[LOG_COLUMNS].reset_index(drop=True)


def _graded(log: pd.DataFrame, pre_week_only: bool) -> pd.DataFrame:
    if log is None or log.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)
    g = log[log["final"].astype(bool) & log["p_win"].notna()]
    return g[g["pre_week"].astype(bool)] if pre_week_only else g


def brier(log: pd.DataFrame, pre_week_only: bool = False) -> tuple[float, int]:
    """Mean squared gap between the win chance and the result (1, 0, or 0.5 for a tie); lower is better.
    A coin flip scores 0.25."""
    g = _graded(log, pre_week_only)
    if g.empty:
        return np.nan, 0
    return float(((g["p_win"].astype(float) - g["result"].astype(float)) ** 2).mean()), len(g)


def calibration(log: pd.DataFrame, pre_week_only: bool = False) -> pd.DataFrame:
    """Per win-chance bin: average chance, actual win rate, matchups."""
    g = _graded(log, pre_week_only)
    if g.empty:
        return pd.DataFrame(columns=["bin", "p_win", "won", "n"])
    b = pd.cut(g["p_win"].astype(float), PROB_BINS, include_lowest=True)
    out = g.groupby(b, observed=True).agg(p_win=("p_win", "mean"), won=("result", "mean"), n=("result", "size"))
    out.index = [f"{i.left:.0%}–{i.right:.0%}" for i in out.index]
    return out.rename_axis("bin").reset_index()
