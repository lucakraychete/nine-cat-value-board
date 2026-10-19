"""Pull rosters, free agents and projections from ESPN and cache them locally.

The ESPN API is slow and rate-limited, so the app never calls it directly:
run `python -m ninecat.ingest` (or press Refresh in the app) to write
data/players.csv and data/meta.json, and everything else reads that cache.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ninecat.config import DATA_DIR, RAW_STATS, Settings, load_settings

PLAYERS_FILE = "players.csv"
META_FILE = "meta.json"
FREE_AGENT = "FA"

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
        row = {stat: float(avg.get(stat, 0.0) or 0.0) for stat in RAW_STATS}
        row["GP"] = float(total.get("GP", 0.0) or 0.0)
        return row, source
    return {stat: 0.0 for stat in [*RAW_STATS, "GP"]}, None


def player_row(player, year: int, fantasy_team: str, team_id: int | None) -> dict:
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


def build_players_frame(league, year: int, free_agent_count: int = 300) -> pd.DataFrame:
    rows = []
    for team in league.teams:
        for player in team.roster:
            rows.append(player_row(player, year, team.team_name, team.team_id))
    for player in league.free_agents(size=free_agent_count):
        rows.append(player_row(player, year, FREE_AGENT, None))

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


def ingest(
    settings: Settings | None = None,
    data_dir: Path = DATA_DIR,
    free_agent_count: int = 300,
) -> pd.DataFrame:
    settings = settings or load_settings()
    league = fetch_league(settings)
    df = build_players_frame(league, settings.year, free_agent_count)

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / PLAYERS_FILE, index=False)
    meta = {
        "league_id": settings.league_id,
        "year": settings.year,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "my_team_id": find_my_team_id(league, settings.swid),
        "teams": {str(t.team_id): t.team_name for t in league.teams},
    }
    (data_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return df


def load_cached(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict] | None:
    """Read the cache written by ingest(), or None if it doesn't exist yet."""
    players_path, meta_path = data_dir / PLAYERS_FILE, data_dir / META_FILE
    if not players_path.exists():
        return None
    df = pd.read_csv(players_path, dtype={"team_id": "Int64"})
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache ESPN league data to data/.")
    parser.add_argument(
        "--free-agents", type=int, default=300, help="free agents to pull (default 300)"
    )
    args = parser.parse_args()

    df = ingest(free_agent_count=args.free_agents)
    rostered = (df["fantasy_team"] != FREE_AGENT).sum()
    print(
        f"Cached {len(df)} players ({rostered} rostered, "
        f"{len(df) - rostered} free agents) to {DATA_DIR / PLAYERS_FILE}"
    )


if __name__ == "__main__":
    main()
