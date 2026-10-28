"""Pull the league from ESPN and stats from nba.com, and cache them locally.

ESPN and nba.com are slow and rate-limited, so the app never calls them
while rendering: `python -m ninecat.ingest` (or the app's background
refresh) writes data/, and everything else reads that cache.

Written each refresh:
- players.csv: rosters and top free agents with ESPN's projection, its
  season-to-date and last 7/15/30-day averages (T_, R7_, R15_, R30_),
  injury status and ADP. meta.json: league, teams and source status.
- transactions.csv: league adds, drops and trades, appended and deduped,
  so the history outlasts ESPN's recent-activity window.
- live_matchup.csv: this matchup period's category totals for every team.
- league_schedule.csv: every matchup period's dates and opponents.
- snapshots/espn_proj_YYYY-MM-DD.csv: that day's ESPN projections.
- season.csv and game_logs.csv: this season from nba.com game logs.
About once a week: schedule.csv (the NBA schedule), and last season's
nba.com numbers (last_season.csv, pace.json). Once, with --history: past
seasons' logs and totals under history/, for the projection model.

ESPN rosters and projections are required. Every other source fails
soft: its old file is kept and the failure recorded in meta["sources"],
which also keeps when each source last succeeded.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ninecat.config import (
    DATA_DIR,
    HISTORY_FIRST_SEASON_START,
    RAW_STATS,
    STATIC_MAX_AGE_DAYS,
    Settings,
    load_settings,
)
from ninecat.lines import ESPN_TOTAL_PREFIX, attach_by_name
from ninecat.nbastats import LOG_COLUMNS, NOT_YET, attempt, season_label
from ninecat.schedule import SCHEDULE_COLUMNS, apply_known_periods, league_schedule_frame, period_calendar

PLAYERS_FILE = "players.csv"
META_FILE = "meta.json"
LAST_SEASON_FILE = "last_season.csv"
PACE_FILE = "pace.json"
GAME_LOGS_FILE = "game_logs.csv"
SEASON_FILE = "season.csv"
SCHEDULE_FILE = "schedule.csv"
LEAGUE_SCHEDULE_FILE = "league_schedule.csv"
LIVE_MATCHUP_FILE = "live_matchup.csv"
TRANSACTIONS_FILE = "transactions.csv"
SNAPSHOT_DIR = "snapshots"
HISTORY_DIR = "history"
MODELS_DIR = "models"
MODEL_FILE = "ros_model.pkl"        # written by `python -m ninecat.model train`
REPORT_FILE = "report.json"
MODEL_PROJ_FILE = "model_proj.csv"  # the trained model's rest-of-season lines (M_ columns), each refresh
MATCHUP_LOG_FILE = "matchup_log.csv"
FREE_AGENT = "FA"
# ESPN per-game stats kept per player: the valuation stats plus minutes.
ESPN_STATS = [*RAW_STATS, "MIN"]
UNDRAFTED_ADP = 139.5
ET = "America/New_York"

# Stat splits to try for the main line, in order of preference.
STAT_SOURCES = ("projected", "total")
# ESPN's season-to-date and recent averages, kept beside the projection.
SPLIT_PREFIXES = {"total": ESPN_TOTAL_PREFIX, "last_7": "R7_", "last_15": "R15_", "last_30": "R30_"}

# Every source the status line reports, in display order.
SOURCE_LABELS = {
    "espn": "ESPN rosters and projections",
    "espn_adp": "ESPN draft position and % rostered",
    "espn_transactions": "ESPN league transactions",
    "espn_matchup": "ESPN live matchups",
    "espn_league_schedule": "ESPN matchup schedule",
    "nba_season": "nba.com this season's game logs",
    "nba_ages": "nba.com player ages",
    "nba_schedule": "nba.com NBA schedule",
    "nba_teams": "nba.com last season's team stats",
    "nba_players": "nba.com last season's player stats",
    "nba_prev_season": "nba.com minutes two seasons ago",
    "nba_last_logs": "nba.com last season's game logs",
    "bbr_bpm": "Basketball Reference BPM",
    "nba_history": "nba.com past seasons (model)",
    "model": "Rest-of-season model predictions",
    "matchup_log": "Matchup model log (win chance vs results)",
}

# ESPN activity message types -> move. 188 (lineup moves) is left out.
TRANSACTION_ACTIONS = {178: "FA add", 180: "Waiver add", 179: "Drop", 181: "Drop", 239: "Drop", 244: "Trade"}
TRANSACTION_COLUMNS = ["msg_id", "date", "team_id", "action", "player_id", "player", "from_team_id", "msg_type"]
ACTIVITY_PAGE = 100
ACTIVITY_MAX_PAGES = 50


# ---------------------------------------------------------------- files

def _replace(tmp: Path, path: Path) -> None:
    """os.replace, retried briefly: on Windows a reader holding the file blocks it."""
    for wait in (0.2, 0.5, 1.0, None):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if wait is None:
                raise
            time.sleep(wait)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write atomically, so the app never reads a half-written file mid-refresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip" if path.name.endswith(".gz") else None)
    _replace(tmp, path)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    _replace(tmp, path)


def load_meta(data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / META_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def snapshot_path(data_dir: Path, day: date) -> Path:
    return data_dir / SNAPSHOT_DIR / f"espn_proj_{day:%Y-%m-%d}.csv"


def history_paths(data_dir: Path, season: str) -> tuple[Path, Path]:
    d = data_dir / HISTORY_DIR
    return d / f"logs_{season}.csv.gz", d / f"totals_{season}.csv.gz"


# ---------------------------------------------------------------- source status

def merge_sources(old: Mapping, new: Mapping[str, str], now: str, old_fetched_at: str | None = None) -> dict:
    """Status per source: the latest result and when it last succeeded.

    A failed source keeps its last success time, so the app can say it's
    running on a saved copy from then. Sources not tried this run keep
    their old entry. Old meta files stored plain status strings.
    """
    out = {}
    for key, entry in old.items():
        if key not in SOURCE_LABELS:
            continue
        if isinstance(entry, str):
            entry = {"status": entry, "updated": old_fetched_at if entry == "ok" else None}
        out[key] = dict(entry)
    for key, status in new.items():
        prev = out.get(key, {})
        out[key] = {"status": status, "updated": now if status == "ok" else prev.get("updated")}
    return out


def is_stale(sources: Mapping, key: str, now: datetime, max_age_days: float = STATIC_MAX_AGE_DAYS) -> bool:
    """True when `key` never succeeded or last succeeded more than max_age_days ago."""
    entry = sources.get(key)
    updated = entry.get("updated") if isinstance(entry, dict) else None
    if not updated:
        return True
    return now - datetime.fromisoformat(updated) > timedelta(days=max_age_days)


def source_rows(meta: Mapping) -> list[dict]:
    """The status line's per-source view: label, state, last success.

    state is "ok", "waiting" (expected to be empty, e.g. before opening
    night), "saved" (failed, running on the copy from `updated`) or
    "missing" (failed and never fetched).
    """
    rows = []
    sources = meta.get("sources", {})
    for key, label in SOURCE_LABELS.items():
        entry = sources.get(key)
        if entry is None:
            continue
        if isinstance(entry, str):
            entry = {"status": entry, "updated": meta.get("fetched_at") if entry == "ok" else None}
        status = str(entry.get("status", ""))
        if status == "ok":
            state = "ok"
        elif status.startswith(NOT_YET):
            state = "waiting"
        else:
            state = "saved" if entry.get("updated") else "missing"
        rows.append({"key": key, "label": label, "state": state, "status": status, "updated": entry.get("updated")})
    return rows


# ---------------------------------------------------------------- ESPN players

def _per_game_stats(player, year: int) -> tuple[dict, str | None]:
    """Return (per-game stats incl. GP, source name) for one espn_api Player.

    Prefers ESPN's season projection; falls back to season-to-date averages
    when a player has no projection (e.g. a mid-season call-up).
    """
    for source in STAT_SOURCES:
        split = player.stats.get(f"{year}_{source}") or {}
        avg, total = split.get("avg"), split.get("total") or {}
        if not avg:
            continue
        row = {stat: float(avg.get(stat, 0.0) or 0.0) for stat in ESPN_STATS}
        row["GP"] = float(total.get("GP", 0.0) or 0.0)
        return row, source
    return {stat: 0.0 for stat in [*ESPN_STATS, "GP"]}, None


def split_stats(player, year: int) -> dict:
    """ESPN's season-to-date and last 7/15/30-day averages, prefixed; blank when missing."""
    row = {}
    for split, prefix in SPLIT_PREFIXES.items():
        data = player.stats.get(f"{year}_{split}") or {}
        avg, total = data.get("avg") or {}, data.get("total") or {}
        for stat in ESPN_STATS:
            row[prefix + stat] = float(avg[stat]) if avg.get(stat) is not None else None
        row[prefix + "GP"] = float(total["GP"]) if total.get("GP") is not None else None
        if split == "total":
            row[prefix + "GS"] = float(total["GS"]) if total.get("GS") is not None else None
    return row


def player_row(player, year: int, fantasy_team: str, team_id: int | None,
               adp: dict[int, float] | None = None, rostered: dict[int, float] | None = None) -> dict:
    stats, source = _per_game_stats(player, year)
    back = getattr(player, "expected_return_date", None)
    return {
        "player_id": player.playerId,
        "name": player.name,
        "pro_team": player.proTeam,
        "position": player.position,
        "eligible": "/".join(
            s for s in player.eligibleSlots if s in {"PG", "SG", "SF", "PF", "C"}
        ),
        "injury_status": player.injuryStatus or "ACTIVE",
        "injured": bool(getattr(player, "injured", False)),
        "return_date": back.isoformat() if back else None,
        "fantasy_team": fantasy_team,
        "team_id": team_id,
        "stat_source": source,
        "ADP": (adp or {}).get(player.playerId),
        "pct_rostered": (rostered or {}).get(player.playerId),
        **stats,
        **split_stats(player, year),
    }


def fetch_league(settings: Settings):
    """Connect to ESPN. Imported lazily so tests never need espn_api or network."""
    from espn_api.basketball import League

    return League(
        league_id=settings.league_id,
        year=settings.year,
        espn_s2=settings.espn_s2,
        swid=settings.swid,
    )


def adp_from_players(players: list[dict]) -> dict[int, float]:
    """ESPN average draft position by player id, drafted players only.

    ESPN scores a player nobody drafts at about pick 140, one past a
    standard 10-team draft, so ADP at or past UNDRAFTED_ADP means undrafted.
    """
    out = {}
    for entry in players:
        player = entry.get("player", entry)
        adp = (player.get("ownership") or {}).get("averageDraftPosition")
        if adp and 0 < adp < UNDRAFTED_ADP:
            out[int(player["id"])] = float(adp)
    return out


def rostered_from_players(players: list[dict]) -> dict[int, float]:
    """ESPN's percent rostered across all its leagues, by player id.

    espn-api's Player doesn't keep it, so it comes from the raw feed's
    ownership.percentOwned, the same block ADP comes from.
    """
    out = {}
    for entry in players:
        player = entry.get("player", entry)
        pct = (player.get("ownership") or {}).get("percentOwned")
        if pct is not None:
            out[int(player["id"])] = round(float(pct), 1)
    return out


def fetch_player_feed(league, size: int = 1000) -> list[dict]:
    """ESPN's top players by draft rank with their ownership block (ADP, % rostered),
    from the same player feed free_agents uses."""
    filters = {"players": {
        "limit": size,
        "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
    }}
    data = league.espn_request.league_get(
        params={"view": "kona_player_info", "scoringPeriodId": league.current_week},
        headers={"x-fantasy-filter": json.dumps(filters)},
    )
    return data.get("players", [])


def build_players_frame(league, year: int, free_agent_count: int = 300,
                        adp: dict[int, float] | None = None,
                        rostered: dict[int, float] | None = None) -> pd.DataFrame:
    rows = []
    for team in league.teams:
        for player in team.roster:
            rows.append(player_row(player, year, team.team_name, team.team_id, adp, rostered))
    for player in league.free_agents(size=free_agent_count):
        rows.append(player_row(player, year, FREE_AGENT, None, adp, rostered))

    df = pd.DataFrame(rows).drop_duplicates(subset="player_id", keep="first")
    df["team_id"] = df["team_id"].astype("Int64")
    return df.reset_index(drop=True)


def projection_snapshot(players: pd.DataFrame) -> pd.DataFrame:
    """ESPN's projection for everyone who has one, to score against results later."""
    cols = ["player_id", "name", "pro_team", "fantasy_team", "injury_status", *ESPN_STATS, "GP"]
    return players.loc[players["stat_source"] == "projected", cols].reset_index(drop=True)


def find_my_team_id(league, swid: str | None) -> int | None:
    """Match the SWID cookie against team owners to find the user's team."""
    if not swid:
        return None
    wanted = swid.strip().strip("{}").lower()
    for team in league.teams:
        for owner in team.owners or []:
            owner_id = owner.get("id", "") if isinstance(owner, dict) else str(owner)
            if owner_id.strip("{}").lower() == wanted:
                return team.team_id
    return None


# ---------------------------------------------------------------- ESPN transactions

def _activity_page(league, offset: int, size: int = ACTIVITY_PAGE) -> list[dict]:
    """One page of raw league activity, newest first (what league.recent_activity reads)."""
    filters = {"topics": {
        "filterType": {"value": ["ACTIVITY_TRANSACTIONS"]},
        "limit": size,
        "limitPerMessageSet": {"value": 25},
        "offset": offset,
        "sortMessageDate": {"sortPriority": 1, "sortAsc": False},
        "sortFor": {"sortPriority": 2, "sortAsc": False},
        "filterIncludeMessageTypeIds": {"value": list(TRANSACTION_ACTIONS)},
    }}
    data = league.espn_request.league_get(
        extend="/communication/", params={"view": "kona_league_communication"},
        headers={"x-fantasy-filter": json.dumps(filters)})
    return data.get("topics", [])


def fetch_activity(league, known_ids: Iterable[str] = ()) -> list[dict]:
    """Page back through league activity until a page holds nothing new.

    The first run (nothing known) reads as far back as ESPN goes.
    """
    known, topics = set(map(str, known_ids)), []
    for page in range(ACTIVITY_MAX_PAGES):
        batch = _activity_page(league, page * ACTIVITY_PAGE)
        topics += batch
        ids = {str(m.get("id")) for t in batch for m in t.get("messages", [])}
        if len(batch) < ACTIVITY_PAGE or (ids and ids <= known):
            break
    return topics


def transactions_frame(topics: Iterable[dict], player_names: Mapping[int, str]) -> pd.DataFrame:
    """One row per add, drop or trade leg.

    team_id is the team making the move (the receiving team in a trade,
    with the sender in from_team_id). Drops of type 239 name the team in
    "for"; the rest in "to". date is UTC.
    """
    rows = []
    for topic in topics:
        when = pd.Timestamp(topic["date"], unit="ms", tz="UTC").isoformat()
        for m in topic.get("messages", []):
            kind = m.get("messageTypeId")
            if kind not in TRANSACTION_ACTIONS:
                continue
            team = m.get("for") if kind == 239 else m.get("to")
            pid = m.get("targetId")
            rows.append({
                "msg_id": str(m["id"]), "date": when, "team_id": team,
                "action": TRANSACTION_ACTIONS[kind], "player_id": pid,
                "player": player_names.get(pid, ""),
                "from_team_id": m.get("from") if kind == 244 else None, "msg_type": kind,
            })
    out = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    for c in ("team_id", "player_id", "from_team_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def merge_transactions(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Old history plus new moves, deduped on ESPN's message id, newest first."""
    frames = [f for f in (old, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["msg_id"] = out["msg_id"].astype(str)
    out = out.drop_duplicates("msg_id", keep="last")
    out = out.sort_values(["date", "msg_id"], ascending=False, kind="stable")
    for c in ("team_id", "player_id", "from_team_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)[TRANSACTION_COLUMNS]


def ingest_transactions(league, data_dir: Path) -> int:
    """Append new league moves to transactions.csv; returns how many were new."""
    path = data_dir / TRANSACTIONS_FILE
    old = pd.read_csv(path, dtype={"msg_id": str}) if path.exists() else None
    known = old["msg_id"] if old is not None else ()
    names = {k: v for k, v in league.player_map.items() if isinstance(k, int)}
    new = transactions_frame(fetch_activity(league, known), names)
    merged = merge_transactions(old, new)
    write_csv(merged, path)
    return len(merged) - (0 if old is None else len(old))


# ---------------------------------------------------------------- ESPN matchups

def _team_id(team) -> int | None:
    tid = getattr(team, "team_id", team)
    return int(tid) if isinstance(tid, int) and tid > 0 else None


def live_matchup_frame(sides: Iterable[dict], period: int) -> pd.DataFrame:
    """One row per team in this period's matchups: opponent, W-L-T in
    categories, and each stat's running total (FG% and FT% as fractions)."""
    rows = []
    for side in sides:
        row = {"period": period, "team_id": side["team_id"], "opponent_id": side.get("opponent_id"),
               "wins": side.get("wins"), "losses": side.get("losses"), "ties": side.get("ties")}
        for stat, cell in (side.get("stats") or {}).items():
            row[stat] = cell.get("value", cell.get("score")) if isinstance(cell, dict) else cell
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["opponent_id"] = out["opponent_id"].astype("Int64")
    return out


def fetch_live_matchup(league) -> pd.DataFrame:
    sides = []
    for box in league.box_scores():
        home, away = _team_id(box.home_team), _team_id(box.away_team)
        for me, opp, prefix in ((home, away, "home"), (away, home, "away")):
            if me is None:
                continue
            sides.append({"team_id": me, "opponent_id": opp, "stats": getattr(box, f"{prefix}_stats", None),
                          "wins": getattr(box, f"{prefix}_wins", None),
                          "losses": getattr(box, f"{prefix}_losses", None),
                          "ties": getattr(box, f"{prefix}_ties", None)})
    return live_matchup_frame(sides, int(league.currentMatchupPeriod))


def _et_day(ms: float) -> date:
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(ET).date()


def opening_day(pro_schedule: Mapping) -> date | None:
    """The day of scoring period 1, from ESPN's pro schedule."""
    days = [_et_day(g["date"]) for games in pro_schedule.values()
            for sp, gl in games.items() if str(sp) == "1" for g in gl]
    return min(days) if days else None


def espn_schedule_frame(pro_schedule: Mapping, team_abbr: Mapping[int, str]) -> pd.DataFrame:
    """ESPN's pro schedule as a schedule frame; the fallback when nba.com fails."""
    rows = {}
    for games in pro_schedule.values():
        for gl in games.values():
            for g in gl:
                home, away = team_abbr.get(g["homeProTeamId"]), team_abbr.get(g["awayProTeamId"])
                if home and away and home != "FA" and away != "FA":
                    rows[g["id"]] = {"date": _et_day(g["date"]).isoformat(), "game_id": str(g["id"]),
                                     "home": home, "away": away}
    out = pd.DataFrame(list(rows.values()), columns=SCHEDULE_COLUMNS)
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def matchup_opponents(schedule_items: Iterable[dict]) -> dict[tuple[int, int], int]:
    """(period, team_id) -> opponent team_id, from ESPN's raw mMatchup schedule."""
    out = {}
    for m in schedule_items:
        period = m.get("matchupPeriodId")
        home, away = (m.get("home") or {}).get("teamId"), (m.get("away") or {}).get("teamId")
        if period is None or home is None or away is None:
            continue
        out[(int(period), int(home))] = int(away)
        out[(int(period), int(away))] = int(home)
    return out


def fetch_league_schedule(league, schedule: pd.DataFrame | None) -> tuple[pd.DataFrame, str, date]:
    """Matchup periods with dates and opponents, a note on how dates were found, and opening day."""
    opening = opening_day(league.pro_schedule)
    if opening is None and schedule is not None and len(schedule):
        opening = pd.Timestamp(schedule["date"].min()).date()
    if opening is None:
        raise ValueError("no opening day in ESPN's or nba.com's schedule")
    final = opening + timedelta(days=int(league.finalScoringPeriod) - 1)
    calendar, note = period_calendar(opening, final, league.settings.matchup_periods, schedule)
    calendar, differs = apply_known_periods(calendar, league.matchup_ids, opening, int(league.currentMatchupPeriod))
    if differs:
        note += "; ESPN's own dates used where they differ (" + "; ".join(differs) + ")"
    raw = league.espn_request.league_get(params={"view": "mMatchup"}).get("schedule", [])
    frame = league_schedule_frame(calendar, matchup_opponents(raw), [t.team_id for t in league.teams],
                                  int(league.settings.reg_season_count))
    return frame, note, opening


# ---------------------------------------------------------------- nba.com

def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def combine_logs(old: pd.DataFrame | None, new: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """game_logs.csv: old seasons kept, any season in `new` replaced.

    Files from before full box scores were kept (no MIN column) are dropped.
    """
    frames = []
    if old is not None and not old.empty and "MIN" in old:
        frames.append(old[~old["season"].isin(list(new))])
    frames += [f for f in new.values() if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.concat(frames, ignore_index=True)[LOG_COLUMNS]


def ingest_nba(year: int, data_dir: Path, sources: Mapping, now: datetime) -> dict[str, str]:
    """Refresh the nba.com caches; keep the old file for any source that fails.

    Last season's numbers are refetched when a week old (or missing), and
    last season's logs only when they're in neither history/ nor game_logs.csv.
    """
    from ninecat.nbastats import fetch_all

    last = season_label(year, 1)
    static = is_stale(sources, "nba_players", now) or not (data_dir / LAST_SEASON_FILE).exists()
    old_logs = load_game_logs(data_dir)
    hist_logs = history_paths(data_dir, last)[0]
    have_last = hist_logs.exists() or ("MIN" in old_logs and (old_logs["season"] == last).any())
    frames, status = fetch_all(year, static=static, last_logs=not have_last, asof=now.astimezone().date())

    if "last_season" in frames:
        write_csv(frames["last_season"], data_dir / LAST_SEASON_FILE)
    if "pace" in frames:
        write_json(frames["pace"], data_dir / PACE_FILE)
    if "last_logs" in frames:
        write_csv(frames["last_logs"], hist_logs)
    if "season" in frames:
        write_csv(frames["season"], data_dir / SEASON_FILE)

    new = {}
    if "season_logs" in frames:
        new[season_label(year)] = frames["season_logs"]
    if "last_logs" in frames:
        new[last] = frames["last_logs"]
    elif "MIN" not in old_logs or not (old_logs["season"] == last).any():
        if hist_logs.exists():
            new[last] = pd.read_csv(hist_logs)
    if new:
        write_csv(combine_logs(old_logs, new), data_dir / GAME_LOGS_FILE)
    return status


def ingest_schedule(year: int, data_dir: Path, sources: Mapping, now: datetime, league=None) -> dict[str, str]:
    """The NBA schedule, about once a week; ESPN's copy if nba.com fails and there's none saved."""
    from ninecat.nbastats import fetch_schedule

    path = data_dir / SCHEDULE_FILE
    if path.exists() and not is_stale(sources, "nba_schedule", now):
        return {}
    status: dict[str, str] = {}
    frame = attempt(status, "nba_schedule", lambda: fetch_schedule(season_label(year)))
    if frame is not None and not frame.empty:
        write_csv(frame, path)
    elif not path.exists() and league is not None:
        from espn_api.basketball.constant import PRO_TEAM_MAP

        write_csv(espn_schedule_frame(league.pro_schedule, PRO_TEAM_MAP), path)
        status["nba_schedule"] += "; using ESPN's schedule instead"
    return status


def ingest_history(year: int, data_dir: Path = DATA_DIR, first: int = HISTORY_FIRST_SEASON_START,
                   log=print) -> dict[str, str]:
    """Past regular seasons for the model, each fetched once: logs and totals with age."""
    from ninecat.nbastats import fetch_history_season

    status = {}
    last_start = year - 2
    for start in range(first, last_start + 1):
        season = f"{start}-{str(start + 1)[2:]}"
        logs_path, totals_path = history_paths(data_dir, season)
        if logs_path.exists() and totals_path.exists():
            status[season] = "cached"
            continue
        result = attempt(status, season, lambda s=season: fetch_history_season(s))
        if result is not None:
            write_csv(result[0], logs_path)
            write_csv(result[1], totals_path)
            status[season] = f"ok ({len(result[0])} games)"
        log(f"  {season}: {status[season]}")
    return status


# ---------------------------------------------------------------- refresh

def ingest(
    settings: Settings | None = None,
    data_dir: Path = DATA_DIR,
    free_agent_count: int = 300,
    nba: bool = True,
    now: datetime | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    data_dir.mkdir(parents=True, exist_ok=True)
    old_meta = load_meta(data_dir)
    old_sources = merge_sources(old_meta.get("sources", {}), {}, stamp, old_meta.get("fetched_at"))

    league = fetch_league(settings)
    status: dict[str, str] = {}
    feed = attempt(status, "espn_adp", lambda: fetch_player_feed(league)) or []
    df = build_players_frame(league, settings.year, free_agent_count,
                             adp_from_players(feed), rostered_from_players(feed))
    status["espn"] = "ok"
    write_csv(df, data_dir / PLAYERS_FILE)
    write_csv(projection_snapshot(df), snapshot_path(data_dir, now.astimezone().date()))

    attempt(status, "espn_transactions", lambda: ingest_transactions(league, data_dir))
    live = attempt(status, "espn_matchup", lambda: fetch_live_matchup(league))
    if live is not None:
        write_csv(live, data_dir / LIVE_MATCHUP_FILE)

    if nba:
        status.update(ingest_schedule(settings.year, data_dir, old_sources, now, league))
    schedule = _read_csv(data_dir / SCHEDULE_FILE)
    calendar = attempt(status, "espn_league_schedule", lambda: fetch_league_schedule(league, schedule))
    if calendar is not None:
        write_csv(calendar[0], data_dir / LEAGUE_SCHEDULE_FILE)
    if nba:
        status.update(ingest_nba(settings.year, data_dir, old_sources, now))
    status["model"] = ingest_model(data_dir, settings.year, now)
    period = int(league.currentMatchupPeriod)
    attempt(status, "matchup_log", lambda: ingest_matchup_log(data_dir, settings, period, now))

    meta = {
        "league_id": settings.league_id,
        "year": settings.year,
        "fetched_at": stamp,
        "my_team_id": find_my_team_id(league, settings.swid),
        "teams": {str(t.team_id): t.team_name for t in league.teams},
        "team_abbrevs": {str(t.team_id): t.team_abbrev for t in league.teams},
        "current_period": int(league.currentMatchupPeriod),
        "scoring_period": int(league.scoringPeriodId),
        "regular_periods": int(league.settings.reg_season_count),
        "season_start": calendar[2].isoformat() if calendar else old_meta.get("season_start"),
        "calendar_note": calendar[1] if calendar else old_meta.get("calendar_note"),
        "sources": merge_sources(old_sources, status, stamp),
    }
    write_json(meta, data_dir / META_FILE)
    return df


# ---------------------------------------------------------------- the rest-of-season model

def load_ros_model(data_dir: Path = DATA_DIR):
    """The model saved by `python -m ninecat.model train`, or None (missing, or saved by an incompatible version)."""
    path = data_dir / MODELS_DIR / MODEL_FILE
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:  # an old or broken pickle just means "no model"
        return None


def load_model_report(data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / MODELS_DIR / REPORT_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def model_snapshot_path(data_dir: Path, day: date) -> Path:
    return data_dir / SNAPSHOT_DIR / f"model_proj_{day.isoformat()}.csv"


def snapshot_files(data_dir: Path = DATA_DIR) -> list[tuple[str, date, pd.DataFrame]]:
    """Every saved projection snapshot as (kind, day, frame), kind "espn" or "model"."""
    out = []
    for kind in ("espn", "model"):
        for path in sorted((data_dir / SNAPSHOT_DIR).glob(f"{kind}_proj_*.csv")):
            try:
                day = date.fromisoformat(path.stem.removeprefix(f"{kind}_proj_"))
            except ValueError:
                continue
            out.append((kind, day, pd.read_csv(path)))
    return out


def _history_logs(data_dir: Path, season: str) -> pd.DataFrame | None:
    return _read_csv(history_paths(data_dir, season)[0])


def model_projections(data_dir: Path, year: int, today: date, model) -> pd.DataFrame:
    """Everyone's rest-of-season line from the saved model, with projected games left."""
    from ninecat.lines import name_key
    from ninecat.model.predict import live_rows, with_games
    from ninecat.schedule import games_between, remaining_games

    players = pd.read_csv(data_dir / PLAYERS_FILE)
    logs = load_game_logs(data_dir)
    current, last, prev = season_label(year), season_label(year, 1), season_label(year, 2)
    cur = logs[logs["season"] == current]
    last_logs = logs[logs["season"] == last]
    if last_logs.empty:
        last_logs = _history_logs(data_dir, last)
    season = _read_csv(data_dir / SEASON_FILE)
    ages = {}
    if season is not None and "age" in season:
        ages = season.dropna(subset=["age"]).set_index("name_key")["age"].to_dict()
    keys = players["name"].map(name_key)
    pred = model.predict(live_rows(cur, last_logs, _history_logs(data_dir, prev), ages, keys))

    espn = players.assign(name_key=keys).drop_duplicates("name_key").set_index("name_key")
    espn = espn.reindex(pred["name_key"]).reset_index(drop=True)
    schedule = _read_csv(data_dir / SCHEDULE_FILE)
    if schedule is None or schedule.empty:
        return pred
    remaining = remaining_games(schedule, today).to_dict()
    sunday = today + timedelta(days=6 - today.weekday())
    week = espn["pro_team"].map(games_between(schedule, today, sunday)).fillna(0)
    back = pd.to_datetime(espn["return_date"], errors="coerce") if "return_date" in espn \
        else pd.Series(pd.NaT, index=espn.index)
    before = [float(games_between(schedule, today, d.date() - timedelta(days=1)).get(t, 0))
              if pd.notna(d) and d.date() > today else np.nan for d, t in zip(back, espn["pro_team"])]
    return with_games(pred, espn["pro_team"], remaining, espn["injury_status"].eq("OUT"),
                      pd.Series(before, dtype=float), week)


def ingest_model(data_dir: Path, year: int, now: datetime) -> str:
    """Predict everyone's rest of season with the saved model: model_proj.csv and today's snapshot.

    Never trains. Returns the source status ("ok", "not available yet ..." or the failure).
    """
    model = load_ros_model(data_dir)
    if model is None:
        return f"{NOT_YET} (run python -m ninecat.model train)"
    try:
        today = now.astimezone().date()
        pred = model_projections(data_dir, year, today, model)
        write_csv(pred, data_dir / MODEL_PROJ_FILE)
        write_csv(pred, model_snapshot_path(data_dir, today))
        return "ok"
    except Exception as exc:  # fail soft: keep the old predictions
        return f"failed: {type(exc).__name__}: {exc}"[:300]


def ingest_matchup_log(data_dir: Path, settings: Settings, period: int, now: datetime) -> None:
    """Log this period's matchup-model win chances (first sighting) and the live score."""
    from ninecat.board import build_board
    from ninecat.config import categories
    from ninecat.model.live import matchup_probabilities, update_matchup_log

    live = _read_csv(data_dir / LIVE_MATCHUP_FILE)
    calendar = _read_csv(data_dir / LEAGUE_SCHEDULE_FILE)
    old = _read_csv(data_dir / MATCHUP_LOG_FILE)
    rows = calendar[calendar["period"] == period] if calendar is not None else pd.DataFrame()
    start = date.fromisoformat(str(rows["start"].iloc[0])[:10]) if not rows.empty else None
    end = date.fromisoformat(str(rows["end"].iloc[0])[:10]) if not rows.empty else None
    probs = {}
    seen = old is not None and not old.empty and bool((old["period"] == period).any())
    if live is not None and not live.empty and start is not None and not seen:
        players = load_cached(data_dir)[0]
        cats = categories(settings.exclude_to)
        names = players.dropna(subset=["team_id"]).groupby("team_id")["fantasy_team"].first()
        probs = matchup_probabilities(build_board(players, cats), cats, {int(k): v for k, v in names.items()},
                                      live, start, end, _read_csv(data_dir / SCHEDULE_FILE))
    log = update_matchup_log(old, live if live is not None else pd.DataFrame(), probs, period, start,
                             now.isoformat(timespec="seconds"))
    write_csv(log, data_dir / MATCHUP_LOG_FILE)


# ---------------------------------------------------------------- reading the cache

def load_cached(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict] | None:
    """Read the cache written by ingest(), or None if it doesn't exist yet.

    Players come back with last season's (L_*, adv_*) and this season's
    (S_*, age) nba.com columns and the model's rest-of-season line (M_*,
    when a model has been trained) attached by name, and meta carries team
    pace under "pace" (empty if nba.com was never fetched).
    """
    players_path = data_dir / PLAYERS_FILE
    if not players_path.exists():
        return None
    df = pd.read_csv(players_path, dtype={"team_id": "Int64"})
    meta = load_meta(data_dir)
    for name in (LAST_SEASON_FILE, SEASON_FILE, MODEL_PROJ_FILE):
        extra = _read_csv(data_dir / name)
        if extra is not None:
            df = attach_by_name(df, extra)
    pace_path = data_dir / PACE_FILE
    meta["pace"] = json.loads(pace_path.read_text()) if pace_path.exists() else {}
    return df, meta


def load_game_logs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Box score by game (this season and last) for the player page; empty if never fetched."""
    logs = _read_csv(data_dir / GAME_LOGS_FILE)
    if logs is None or "MIN" not in logs:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return logs


def load_table(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Any other cached table (schedule, league_schedule, live_matchup, transactions); empty if missing."""
    path = data_dir / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def snapshot_days(data_dir: Path = DATA_DIR) -> list[date]:
    """Days with a saved ESPN projection snapshot, oldest first."""
    days = []
    for path in (data_dir / SNAPSHOT_DIR).glob("espn_proj_*.csv"):
        try:
            days.append(date.fromisoformat(path.stem.removeprefix("espn_proj_")))
        except ValueError:
            continue
    return sorted(days)


def load_snapshot(day: date | None, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = snapshot_path(data_dir, day) if day else None
    return pd.read_csv(path) if path is not None and path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache ESPN league data and nba.com stats to data/.")
    parser.add_argument(
        "--free-agents", type=int, default=300, help="free agents to pull (default 300)"
    )
    parser.add_argument(
        "--skip-nba", action="store_true", help="only refresh ESPN, keep the nba.com cache"
    )
    parser.add_argument(
        "--history", action="store_true",
        help="fetch past seasons' game logs and totals for the model (each season once), then exit",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.history:
        print(f"Fetching past seasons into {DATA_DIR / HISTORY_DIR} (seasons already there are skipped)")
        status = ingest_history(settings.year)
        failed = [s for s, v in status.items() if v.startswith("failed")]
        meta = load_meta()
        if meta:
            note = "ok" if not failed else f"failed: {len(failed)} seasons ({', '.join(failed)})"
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["sources"] = merge_sources(meta.get("sources", {}), {"nba_history": note}, now,
                                            meta.get("fetched_at"))
            write_json(meta, DATA_DIR / META_FILE)
        print(f"{len(status) - len(failed)} of {len(status)} seasons cached")
        return

    df = ingest(settings, free_agent_count=args.free_agents, nba=not args.skip_nba)
    rostered = (df["fantasy_team"] != FREE_AGENT).sum()
    print(
        f"Cached {len(df)} players ({rostered} rostered, "
        f"{len(df) - rostered} free agents) to {DATA_DIR / PLAYERS_FILE}"
    )
    meta = load_meta()
    for row in source_rows(meta):
        print(f"  {row['label']}: {row['status']}")
    if meta.get("calendar_note"):
        print(f"  Matchup calendar: {meta['calendar_note']}")


if __name__ == "__main__":
    main()
