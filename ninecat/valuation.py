"""Per-game z-score valuation for head-to-head category leagues.

How a player's value is built:

1. Counting cats (PTS, REB, ...): z = (x - pool mean) / pool std. TO is
   negated because fewer turnovers is better.
2. Percentage cats (FG%, FT%): a 40% shooter on 2 attempts barely moves your
   team's FG%, a 40% shooter on 20 attempts sinks it. So we score *impact*:
   makes - league_pct * attempts, i.e. attempts * (pct - league_pct), which is
   the number of makes above what a league-average shooter would produce on
   the same volume. That impact is then z-scored like a counting stat.
3. The z-scores are summed over every category that isn't punted. That sum
   is the player's per-game value.
4. Means and stds come from the *pool* of players who will actually be
   rostered (default top 130), not from all ~500 NBA players. The pool
   depends on the values and the values depend on the pool, so we iterate:
   value everyone, take the top N, recompute stats from that pool, and
   repeat until the pool stops changing.
5. Games-missed discount, measured from replacement level. When a player
   sits, his roster spot doesn't go blank: you start the best player left
   over, and in a league that rosters N players that's roughly the Nth-best
   player. So replacement level is the per-game value of the pool's last
   player, and only what a player adds *above* that level depends on him
   being on the court. Value = replacement + (per-game value - replacement)
   * GP / 82, with GP clipped to [0, 82] and an unknown GP taken as 70.
   A star who plays 41 games keeps replacement value plus half his edge
   over it, not half his total. A player below replacement level moves up
   toward it when he misses games, because those are games a replacement
   plays instead of him.

The pool is chosen on per-game value using all categories, before the
discount. Punting is a personal strategy and shouldn't change which players
the league as a whole rosters; replacement level does use the punted
per-game values, since it's measured on the same scale as the values it
anchors.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ninecat.config import (
    DEFAULT_POOL_SIZE,
    NEGATIVE_CATS,
    PERCENT_CATS,
    SEASON_GAMES,
)

# Games assumed for a player with no GP projection.
DEFAULT_GP = 70


def z_col(cat: str) -> str:
    return f"z_{cat}"


def _zscore(values: pd.Series, pool: pd.Series) -> pd.Series:
    """Standardize values using the mean/std of the pool rows only."""
    ref = values[pool]
    std = ref.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - ref.mean()) / std


def percentage_impact(df: pd.DataFrame, cat: str, pool: pd.Series) -> pd.Series:
    """Makes above a league-average shooter on the same attempts."""
    makes_col, attempts_col = PERCENT_CATS[cat]
    makes, attempts = df[makes_col], df[attempts_col]
    pool_attempts = attempts[pool].sum()
    league_pct = makes[pool].sum() / pool_attempts if pool_attempts > 0 else 0.0
    return makes - league_pct * attempts


def category_zscores(
    df: pd.DataFrame, cats: Iterable[str], pool: pd.Series
) -> pd.DataFrame:
    """One z-score column per category, standardized against the pool."""
    out = {}
    for cat in cats:
        if cat in PERCENT_CATS:
            raw = percentage_impact(df, cat, pool)
        else:
            raw = df[cat].astype(float)
        z = _zscore(raw, pool)
        out[z_col(cat)] = -z if cat in NEGATIVE_CATS else z
    return pd.DataFrame(out, index=df.index)


def availability(games_played: pd.Series, season_games: int = SEASON_GAMES) -> pd.Series:
    """Fraction of the season a player is projected to play, in [0, 1].

    A missing or zero GP means no projection, not a lost season, so it
    counts as DEFAULT_GP games.
    """
    gp = games_played.astype(float)
    gp = gp.where(gp > 0, DEFAULT_GP)
    return (gp / season_games).clip(0.0, 1.0)


def replacement_level(per_game: pd.Series, pool_size: int) -> float:
    """Per-game value of the last player in a pool of pool_size."""
    ranked = per_game.sort_values(ascending=False, kind="stable")
    return float(ranked.iloc[min(pool_size, len(ranked)) - 1])


def apply_games_discount(per_game: pd.Series, avail: pd.Series, replacement: float) -> pd.Series:
    """Scale each player's value above replacement level by his availability."""
    return replacement + (per_game - replacement) * avail


def _top_n(value: pd.Series, n: int) -> pd.Series:
    keep = value.sort_values(ascending=False, kind="stable").index[:n]
    return pd.Series(value.index.isin(keep), index=value.index)


def value_players(
    df: pd.DataFrame,
    cats: Iterable[str],
    punt: Iterable[str] = (),
    pool_size: int = DEFAULT_POOL_SIZE,
    season_games: int = SEASON_GAMES,
    games_discount: bool = True,
    max_iter: int = 25,
) -> pd.DataFrame:
    """Value every player in df.

    df needs per-game columns for each counting cat plus FGM/FGA/FTM/FTA and
    GP (projected games). Returns a copy with added columns: FG%, FT%,
    z_<cat> per category, z_total (non-punted per-game value),
    availability, value (z_total after the games discount), rank, in_pool.
    """
    cats = list(cats)
    punt = set(punt)
    unknown = punt - set(cats)
    if unknown:
        raise ValueError(f"Cannot punt categories not in the league: {sorted(unknown)}")

    out = df.copy().reset_index(drop=True)
    for cat, (makes, attempts) in PERCENT_CATS.items():
        out[cat] = (out[makes] / out[attempts].where(out[attempts] > 0)).round(3)

    def score(pool: pd.Series, scored_cats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        z = category_zscores(out, cats, pool)
        return z, z[[z_col(c) for c in scored_cats]].sum(axis=1)

    # Iterate to a fixed point: the pool used for the stats is the top-N by
    # the per-game values those stats produce. Start from the whole list.
    pool = pd.Series(True, index=out.index)
    for _ in range(max_iter):
        _, league_value = score(pool, cats)
        new_pool = _top_n(league_value, min(pool_size, len(out)))
        if new_pool.equals(pool):
            break
        pool = new_pool

    scored = [c for c in cats if c not in punt]
    z, per_game = score(pool, scored)
    avail = availability(out["GP"], season_games)
    if games_discount:
        value = apply_games_discount(per_game, avail, replacement_level(per_game, pool_size))
    else:
        value = per_game

    out = pd.concat([out, z], axis=1)
    out["z_total"] = per_game
    out["availability"] = avail.round(3)
    out["value"] = value
    out["in_pool"] = pool
    out["rank"] = value.rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def team_profile(
    valued: pd.DataFrame, cats: Iterable[str], team_col: str = "fantasy_team"
) -> pd.DataFrame:
    """Sum of per-game z-scores by fantasy team and category.

    A quick read on where each roster is strong or weak; higher is better
    in every column (TO is already negated).
    """
    cols = [z_col(c) for c in cats]
    profile = valued.groupby(team_col)[cols].sum()
    profile.columns = list(cats)
    profile["total"] = profile.sum(axis=1)
    return profile.sort_values("total", ascending=False)
