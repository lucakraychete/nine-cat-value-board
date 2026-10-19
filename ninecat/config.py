"""League settings and category definitions.

Everything that varies between leagues lives here, read once from the
environment (.env), so the rest of the code never touches os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Counting categories: more is better, except TO where less is better.
COUNTING_CATS = ["3PM", "REB", "AST", "STL", "BLK", "PTS"]
# Percentage categories, valued by volume-weighted impact (see valuation.py).
# Each maps to its (makes, attempts) columns.
PERCENT_CATS = {"FG%": ("FGM", "FGA"), "FT%": ("FTM", "FTA")}
NEGATIVE_CATS = {"TO"}

# Raw per-game stats a player row needs for valuation.
RAW_STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]

# 10 teams x 13 roster spots: the players who actually get rostered.
DEFAULT_POOL_SIZE = 130
SEASON_GAMES = 82


def categories(exclude_to: bool) -> list[str]:
    """Scoring categories in display order. 8 cats when turnovers are excluded."""
    cats = ["FG%", "FT%", *COUNTING_CATS]
    if not exclude_to:
        cats.append("TO")
    return cats


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    league_id: int
    year: int
    espn_s2: str | None
    swid: str | None
    exclude_to: bool = True

    @property
    def categories(self) -> list[str]:
        return categories(self.exclude_to)


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings from .env (or the real environment, which wins)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    league_id = os.getenv("LEAGUE_ID", "").strip()
    year = os.getenv("YEAR", "").strip()
    if not league_id.isdigit() or not year.isdigit():
        raise ValueError(
            "LEAGUE_ID and YEAR must be set to numbers in .env (see .env.example)."
        )

    return Settings(
        league_id=int(league_id),
        year=int(year),
        espn_s2=os.getenv("ESPN_S2") or None,
        swid=os.getenv("SWID") or None,
        exclude_to=_env_bool(os.getenv("EXCLUDE_TO"), default=True),
    )
