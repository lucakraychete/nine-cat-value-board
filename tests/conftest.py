"""Shared synthetic league: ESPN-style projections plus last-season columns."""

import numpy as np
import pandas as pd
import pytest

TEAMS = ["Alpha", "Bravo", "Charlie"]
POSITIONS = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C"]
NBA = ["DEN", "BOS", "PHL", "LAL", "MIA"]


def synthetic_players(n: int = 90, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(8, 37, n)
    scale = minutes / 30
    fga = rng.uniform(4, 18, n) * scale
    fta = rng.uniform(0.5, 7, n) * scale
    df = pd.DataFrame({
        "player_id": np.arange(1000, 1000 + n),
        "name": [f"Player {i}" for i in range(n)],
        "pro_team": rng.choice(NBA, n),
        "position": "PG",
        "eligible": rng.choice(POSITIONS, n),
        "injury_status": rng.choice(["ACTIVE", "ACTIVE", "ACTIVE", "OUT", "DAY_TO_DAY"], n),
        "fantasy_team": [TEAMS[i % 3] if i < 39 else "FA" for i in range(n)],
        "team_id": [i % 3 + 1 if i < 39 else None for i in range(n)],
        "stat_source": "projected",
        "ADP": np.where(np.arange(n) < 45, np.arange(n) * 1.5 + 1 + rng.uniform(0, 3, n), np.nan),
        "MIN": minutes,
        "PTS": rng.uniform(6, 30, n) * scale,
        "REB": rng.uniform(1, 12, n) * scale,
        "AST": rng.uniform(0.5, 9, n) * scale,
        "STL": rng.uniform(0.2, 2, n) * scale,
        "BLK": rng.uniform(0.1, 2.5, n) * scale,
        "3PM": rng.uniform(0, 4, n) * scale,
        "TO": rng.uniform(0.5, 4, n) * scale,
        "FGA": fga,
        "FGM": fga * rng.uniform(0.38, 0.62, n),
        "FTA": fta,
        "FTM": fta * rng.uniform(0.55, 0.92, n),
        "GP": rng.integers(40, 83, n).astype(float),
    })
    df["team_id"] = df["team_id"].astype("Int64")
    # Last season: similar but noisier, and missing for the last 10 (rookies).
    last = df[["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGA", "FGM", "FTA", "FTM"]]
    last = last * rng.uniform(0.8, 1.2, (n, 1))
    for c in last.columns:
        df["L_" + c] = last[c]
    df["L_G"] = rng.integers(5, 82, n).astype(float)
    df["prev_MIN"] = df["L_MIN"] + rng.uniform(-4, 4, n)
    for k in ["usg", "ts", "ast", "trb", "stl", "blk", "tov", "per", "bpm"]:
        df["adv_" + k] = rng.normal(20 if k != "ts" else 0.57, 5 if k != "ts" else 0.04, n)
    rookies = df.index >= n - 10
    df.loc[rookies, [c for c in df.columns if c.startswith(("L_", "adv_", "prev_"))]] = np.nan
    # This season so far (nba.com S_ line): the first 60 players, a few games in.
    now = df[["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGA", "FGM", "FTA", "FTM"]]
    now = now * rng.uniform(0.7, 1.3, (n, 1))
    for c in now.columns:
        df["S_" + c] = now[c].where(df.index < 60)
    df["S_G"] = np.where(df.index < 60, rng.integers(1, 8, n), np.nan).astype(float)
    return df


@pytest.fixture
def players() -> pd.DataFrame:
    return synthetic_players()
