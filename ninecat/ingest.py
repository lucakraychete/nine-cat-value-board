"""Pull rosters, free agents and projections from ESPN and cache them locally.

The ESPN API is slow and rate-limited, so the app never calls it directly:
run `python -m ninecat.ingest` (or press Refresh in the app) to write
data/players.csv and data/meta.json, and everything else reads that cache.

The same run pulls last season's stats, pace and game logs from nba.com
(see nbastats.py) into last_season.csv, pace.json and game_logs.csv. Those
are optional: if nba.com fails, the previous files are kept and the
status is recorded in meta.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ninecat.config import DATA_DIR, RAW_STATS, Settings, load_settings
from ninecat.lines import attach_last_season

PLAYERS_FILE = "players.csv"
META_FILE = "meta.json"
LAST_SEASON_FILE = "last_season.csv"
PACE_FILE = "pace.json"
GAME_LOGS_FILE = "game_logs.csv"
FREE_AGENT = "FA"
# ESPN per-game stats kept per player: the valuation stats plus minutes.
ESPN_STATS = [*RAW_STATS, "MIN"]
UNDRAFTED_ADP = 139.5

# Stat splits to try, in order of preference.
STAT_SOURCES = ("projected", "total")


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


def player_row(player, year: int, fantasy_team: str, team_id: int | None,
               adp: dict[int, float] | None = None) -> dict:
    stats, source = _per_game_stats(player, year)
    return {
        "player_id": player.playerId,
        "name": player.name,
        "pro_team": player.proTeam,
        "position": player.position,
        "eligible": "/".join(
            s for s in player.eligibleSlots if s in {"PG", "SG", "SF", "PF", "C"}
        ),
        "injury_status": player.injuryStatus or "ACTIVE",
        "fantasy_team": fantasy_team,
        "team_id": team_id,
        "stat_source": source,
        "ADP": (adp or {}).get(player.playerId),
        **stats,
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


def fetch_adp(league, size: int = 1000) -> dict[int, float]:
    """ESPN's ADP across all its leagues, from the same player feed free_agents uses."""
    filters = {"players": {
        "limit": size,
        "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
    }}
    data = league.espn_request.league_get(
        params={"view": "kona_player_info", "scoringPeriodId": league.current_week},
        headers={"x-fantasy-filter": json.dumps(filters)},
    )
    return adp_from_players(data.get("players", []))


def build_players_frame(league, year: int, free_agent_count: int = 300,
                        adp: dict[int, float] | None = None) -> pd.DataFrame:
    rows = []
    for team in league.teams:
        for player in team.roster:
            rows.append(player_row(player, year, team.team_name, team.team_id, adp))
    for player in league.free_agents(size=free_agent_count):
        rows.append(player_row(player, year, FREE_AGENT, None, adp))

    df = pd.DataFrame(rows).drop_duplicates(subset="player_id", keep="first")
    df["team_id"] = df["team_id"].astype("Int64")
    return df.reset_index(drop=True)


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


def ingest_nba(year: int, data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Refresh the nba.com caches; keep the old file for any source that fails."""
    from ninecat.nbastats import fetch_all

    frames, status = fetch_all(year)
    if "last_season" in frames:
        frames["last_season"].to_csv(data_dir / LAST_SEASON_FILE, index=False)
    if "pace" in frames:
        (data_dir / PACE_FILE).write_text(json.dumps(frames["pace"], indent=2))
    if "game_logs" in frames:
        frames["game_logs"].to_csv(data_dir / GAME_LOGS_FILE, index=False)
    return status


def ingest(
    settings: Settings | None = None,
    data_dir: Path = DATA_DIR,
    free_agent_count: int = 300,
    nba: bool = True,
) -> pd.DataFrame:
    settings = settings or load_settings()
    league = fetch_league(settings)
    sources = {}
    try:
        adp = fetch_adp(league)
        sources["espn_adp"] = "ok"
    except Exception as exc:  # ADP is optional; the board shows a dash
        adp = {}
        sources["espn_adp"] = f"failed: {type(exc).__name__}: {exc}"[:300]
    df = build_players_frame(league, settings.year, free_agent_count, adp)
    sources["espn"] = "ok"

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / PLAYERS_FILE, index=False)
    if nba:
        sources.update(ingest_nba(settings.year, data_dir))
    meta = {
        "league_id": settings.league_id,
        "year": settings.year,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "my_team_id": find_my_team_id(league, settings.swid),
        "teams": {str(t.team_id): t.team_name for t in league.teams},
        "team_abbrevs": {str(t.team_id): t.team_abbrev for t in league.teams},
        "sources": sources,
    }
    (data_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return df


def load_cached(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict] | None:
    """Read the cache written by ingest(), or None if it doesn't exist yet.

    Players come back with last season's columns attached, and meta
    carries team pace under "pace" (empty if nba.com was never fetched).
    """
    players_path, meta_path = data_dir / PLAYERS_FILE, data_dir / META_FILE
    if not players_path.exists():
        return None
    df = pd.read_csv(players_path, dtype={"team_id": "Int64"})
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    last_path, pace_path = data_dir / LAST_SEASON_FILE, data_dir / PACE_FILE
    if last_path.exists():
        df = attach_last_season(df, pd.read_csv(last_path))
    meta["pace"] = json.loads(pace_path.read_text()) if pace_path.exists() else {}
    return df, meta


def load_game_logs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Minutes by game for the player page; empty if never fetched."""
    path = data_dir / GAME_LOGS_FILE
    if not path.exists():
        return pd.DataFrame(columns=["name_key", "date", "min", "season"])
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache ESPN league data to data/.")
    parser.add_argument(
        "--free-agents", type=int, default=300, help="free agents to pull (default 300)"
    )
    parser.add_argument(
        "--skip-nba", action="store_true", help="only refresh ESPN, keep the nba.com cache"
    )
    args = parser.parse_args()

    df = ingest(free_agent_count=args.free_agents, nba=not args.skip_nba)
    rostered = (df["fantasy_team"] != FREE_AGENT).sum()
    print(
        f"Cached {len(df)} players ({rostered} rostered, "
        f"{len(df) - rostered} free agents) to {DATA_DIR / PLAYERS_FILE}"
    )
    meta = json.loads((DATA_DIR / META_FILE).read_text())
    for name, state in meta.get("sources", {}).items():
        print(f"  {name}: {state}")


if __name__ == "__main__":
    main()
