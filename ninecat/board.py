"""The full board: value under a stats source, plus the layers around it.

build_board() runs value_players() on the chosen per-game line, then adds:

- per-36 and per-100-possession value and rank: the same valuation with
  every stat scaled to 36 minutes or 100 team possessions (no games
  discount), for players with 15+ minutes. Talent independent of role.
- market rank from ADP (undrafted players share the slot after the last
  drafted one) and gap = market rank - value rank, so a positive gap is
  a player the market undervalues.
- advanced-stat z-scores against the top of the board, for shading.
- fantasy tier (from value rank) and NBA role tier (minutes and usage).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.config import DEFAULT_POOL_SIZE, SEASON_GAMES
from ninecat.lines import LAST_PREFIX, per36, per100, stat_line
from ninecat.valuation import value_players

# A line needs this many minutes to be valued at all, and more to be
# scaled up to 36 minutes or 100 possessions without amplifying noise.
BOARD_MIN_MINUTES = 10
RATE_MIN_MINUTES = 15

FANTASY_TIERS = [(10, "Superstar"), (30, "Star"), (70, "Starter"), (110, "Rotation"),
                 (150, "Bench"), (math.inf, "Waiver")]
ROLE_TIERS = [(33, "Franchise"), (29, "Starter"), (23, "Rotation"), (16, "Bench"), (0, "Fringe")]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    help: str
    lower_is_better: bool = False


ADVANCED_METRICS = [
    Metric("usg", "USG%", "Usage rate: share of team plays he finishes while on court"),
    Metric("ts", "TS%", "True shooting: scoring efficiency including threes and free throws"),
    Metric("ast", "AST%", "Share of teammate baskets he assisted while on court"),
    Metric("trb", "TRB%", "Share of available rebounds he grabbed while on court"),
    Metric("stl", "STL%", "Opponent possessions ending in his steal"),
    Metric("blk", "BLK%", "Opponent two-point attempts he blocked"),
    Metric("tov", "TOV%", "Turnovers per 100 plays he used (lower is better)", True),
    Metric("per", "PER", "Player efficiency rating, per-minute production (15 = average)"),
    Metric("bpm", "BPM", "Box plus/minus: points per 100 possessions above an average player"),
]


def fantasy_tier(rank: float) -> str:
    return next(name for limit, name in FANTASY_TIERS if rank <= limit)


def role_tier(minutes: float, usg: float | None = None) -> str:
    """NBA role from minutes; the top tier also needs high usage when it's known."""
    if minutes >= 33 and (usg is None or pd.isna(usg) or usg >= 27):
        return "Franchise"
    for limit, name in ROLE_TIERS[1:]:
        if minutes >= limit:
            return name
    return "Fringe"


def player_key(df: pd.DataFrame) -> pd.Series:
    """Stable per-player key for joining board layers."""
    return df["player_id"] if "player_id" in df else df["name"]


def market_ranks(rank: pd.Series, adp: pd.Series) -> pd.DataFrame:
    """Market rank by ADP and gap = market rank - value rank."""
    drafted = adp.where(adp > 0).dropna().sort_values(kind="stable")
    mkt = pd.Series(len(drafted) + 1, index=rank.index, dtype=int)
    mkt[drafted.index] = np.arange(1, len(drafted) + 1)
    return pd.DataFrame({"mkt_rank": mkt, "gap": mkt - rank})


def advanced_zscores(board: pd.DataFrame, pool_size: int = DEFAULT_POOL_SIZE) -> pd.DataFrame:
    """z of each advanced metric against the top `pool_size` players who have it.

    TOV% is negated so higher is always better, like the category z-scores.
    """
    out = pd.DataFrame(index=board.index)
    top = board["rank"] <= pool_size
    for m in ADVANCED_METRICS:
        col = f"adv_{m.key}"
        if col not in board:
            out[f"az_{m.key}"] = np.nan
            continue
        vals = pd.to_numeric(board[col], errors="coerce")
        ref = vals[top & vals.notna()]
        sd = ref.std(ddof=0) if len(ref) else np.nan
        sd = sd if np.isfinite(sd) and sd > 0 else 1.0
        z = (vals - ref.mean()) / sd
        out[f"az_{m.key}"] = -z if m.lower_is_better else z
    return out


def _rate_layer(line: pd.DataFrame, scaled: pd.DataFrame, cats, punt, pool_size, prefix) -> pd.DataFrame:
    ok = line["MIN"] >= RATE_MIN_MINUTES
    if not ok.any():
        return pd.DataFrame(columns=[f"{prefix}_value", f"{prefix}_rank"])
    v = value_players(scaled[ok], cats, punt=punt, pool_size=pool_size, games_discount=False)
    return pd.DataFrame({f"{prefix}_value": v["value"].values, f"{prefix}_rank": v["rank"].values},
                        index=player_key(v).values)


def build_board(
    players: pd.DataFrame,
    cats: Iterable[str],
    punt: Iterable[str] = (),
    source: str = "blend",
    games_discount: bool = True,
    pool_size: int = DEFAULT_POOL_SIZE,
    season_games: int = SEASON_GAMES,
    pace: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Value every player with a usable line and add the board layers.

    Players with no line under `source`, or under 10 minutes a game, are
    left off (when minutes are known at all). Returns value_players()'s
    output, sorted by rank, plus: p36_value/p36_rank, p100_value/p100_rank,
    mkt_rank, gap, proj_MIN, d_min, az_<metric>, fantasy_tier, role_tier.
    """
    cats, punt = list(cats), list(punt)
    line = stat_line(players, source)
    keep = line["has_line"]
    if (line["MIN"].fillna(0) > 0).any():
        keep &= line["MIN"] >= BOARD_MIN_MINUTES
    line = line[keep]
    if line.empty:
        raise ValueError(f"No players have a usable {source!r} line.")

    board = value_players(line, cats, punt=punt, pool_size=pool_size,
                          season_games=season_games, games_discount=games_discount)
    key = player_key(board)

    for prefix, scaled in (("p36", per36(line)), ("p100", per100(line, pace))):
        layer = _rate_layer(line, scaled, cats, punt, pool_size, prefix)
        for col in layer.columns:
            board[col] = key.map(layer[col]).values
        board[f"{prefix}_rank"] = board[f"{prefix}_rank"].astype("Int64")

    adp = board["ADP"] if "ADP" in board else pd.Series(np.nan, index=board.index)
    board = board.join(market_ranks(board["rank"], pd.to_numeric(adp, errors="coerce")))

    # Projected minutes, and minus last season's: a bigger or smaller role ahead.
    board["proj_MIN"] = board["d_min"] = np.nan
    if "MIN" in players:
        p = players.set_index(player_key(players))
        p = p[~p.index.duplicated()]
        proj_min = p["MIN"].where(p["stat_source"].notna()) if "stat_source" in p else p["MIN"]
        board["proj_MIN"] = key.map(proj_min).values
        if LAST_PREFIX + "MIN" in p:
            board["d_min"] = key.map(proj_min - p[LAST_PREFIX + "MIN"]).values

    board = board.join(advanced_zscores(board, pool_size))
    board["fantasy_tier"] = board["rank"].map(fantasy_tier)
    usg = board["adv_usg"] if "adv_usg" in board else pd.Series(np.nan, index=board.index)
    board["role_tier"] = [role_tier(m, u) for m, u in zip(board["MIN"].fillna(0), usg)]
    return board
