"""Per-game z-score valuation for head-to-head category leagues.

How a player's value is built:

1. Counting cats (PTS, REB, ...): z = (x - pool mean) / pool std. TO is
   negated because fewer turnovers is better.
2. Percentage cats (FG%, FT%): a 40% shooter on 2 attempts barely moves your
   team's FG%, a 40% shooter on 20 attempts sinks it. So we score *impact*:
   makes - league_pct * attempts, i.e. attempts * (pct - league_pct), which is
   the number of makes above what a league-average shooter would produce on
   the same volume. That impact is then z-scored like a counting stat.
3. The z-scores are summed over every category that isn't punted.
4. Games-missed discount: positive totals are scaled by projected GP /
   season games, since a great player who misses half the season delivers
   about half the value. Negative totals are not scaled; missing games
   doesn't make a bad player good.
5. Means and stds come from the *pool* of players who will actually be
   rostered (default top 130), not from all ~500 NBA players. The pool
   depends on the values and the values depend on the pool, so we iterate:
   value everyone, take the top N, recompute stats from that pool, and
   repeat until the pool stops changing.

The pool is always chosen using all categories. Punting is a personal
strategy and shouldn't change which players the league as a whole rosters.
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
    """Fraction of the season a player is projected to play, in [0, 1]."""
    return (games_played.astype(float) / season_games).clip(0.0, 1.0)


def apply_games_discount(total: pd.Series, avail: pd.Series) -> pd.Series:
    """Scale positive totals by availability; leave negative totals alone."""
    return total.where(total <= 0, total * avail)


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
    z_<cat> per category, z_total (non-punted sum), availability, value,
    rank, in_pool.
    """
    cats = list(cats)
    punt = set(punt)
    unknown = punt - set(cats)
    if unknown:
        raise ValueError(f"Cannot punt categories not in the league: {sorted(unknown)}")

    out = df.copy().reset_index(drop=True)
    for cat, (makes, attempts) in PERCENT_CATS.items():
        out[cat] = (out[makes] / out[attempts].where(out[attempts] > 0)).round(3)

    # No GP data at all (e.g. preseason with empty projections) means we
    # can't discount anyone, rather than discounting everyone to zero.
    if games_discount and out["GP"].max() > 0:
        avail = availability(out["GP"], season_games)
    else:
        avail = pd.Series(1.0, index=out.index)

    def score(pool: pd.Series, scored_cats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        z = category_zscores(out, cats, pool)
        total = z[[z_col(c) for c in scored_cats]].sum(axis=1)
        return z, apply_games_discount(total, avail)

    # Iterate to a fixed point: the pool used for the stats is the top-N by
    # the values those stats produce. Start from the whole player list.
    pool = pd.Series(True, index=out.index)
    for _ in range(max_iter):
        _, league_value = score(pool, cats)
        new_pool = _top_n(league_value, min(pool_size, len(out)))
        if new_pool.equals(pool):
            break
        pool = new_pool

    scored = [c for c in cats if c not in punt]
    z, value = score(pool, scored)

    out = pd.concat([out, z], axis=1)
    out["z_total"] = z[[z_col(c) for c in scored]].sum(axis=1)
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
