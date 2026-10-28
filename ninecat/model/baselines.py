"""Baselines for the rest-of-season model: what the app could do without one.

Each returns a per-game line (dataset.per_game() columns) for every
checkpoint row:

- season_to_date: his line so far;
- last_season: last season's line (season to date for anyone without one);
- blend: the app's "Blend" source, 60% projection + 40% last season;
- interim: the app's current ROS line (overview.ros_line()), this season
  weighted GP / (GP + 20) against the preseason projection.

ESPN's preseason projections don't exist for past seasons, so the
backtests stand in preseason_prior() for them: last season and the season
before, rates weighted 2:1 by minutes, and last season's minutes per game.
Players with neither (mostly rookies) get the training pool's average
rates at their minutes so far (RookiePrior). See docs/MODEL.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.config import ROS_PRIOR_GAMES
from ninecat.model.dataset import PCT_STATS, RATE_STATS, TOTALS, per_game, window_rates

PRIOR_WEIGHTS = {"p1": 2.0, "p2": 1.0}          # last season counts twice the season before
MPG_BUCKETS = (0, 10, 15, 20, 25, 30, 35, 60)   # minutes-per-game bands for the rookie prior
BLEND_PROJECTION = 0.6                          # the app's Blend: 60% projection, 40% last season
COUNT_COLS = ["MIN", *RATE_STATS, *[m for m, _ in PCT_STATS.values()]]


def mix(a: pd.DataFrame, b: pd.DataFrame, w) -> pd.DataFrame:
    """w x a + (1 - w) x b for every counting column, percentages from the mixed
    makes and attempts. Where one side is missing, the other alone."""
    w = pd.Series(w, index=a.index) if np.ndim(w) == 0 else pd.Series(np.asarray(w, float), index=a.index)
    out = pd.DataFrame(index=a.index)
    for c in COUNT_COLS:
        both = w * a[c] + (1 - w) * b[c]
        out[c] = both.where(a[c].notna() & b[c].notna(), a[c].where(a[c].notna(), b[c]))
    for c, (made, att) in PCT_STATS.items():
        out[c] = out[made] / out[att].where(out[att] > 0)
    return out[per_game_columns()]


def per_game_columns() -> list[str]:
    return ["MIN", *RATE_STATS, *[m for m, _ in PCT_STATS.values()], *PCT_STATS]


def _bucket(mpg: pd.Series) -> pd.Series:
    return pd.cut(mpg, MPG_BUCKETS, right=False, labels=False)


@dataclass(frozen=True)
class RookiePrior:
    """Average per-minute rates over the rest of the season, by minutes-so-far band (training pool)."""
    rates: pd.DataFrame   # index: bucket, columns: RATE_STATS and PCT_STATS

    @classmethod
    def fit(cls, train: pd.DataFrame) -> RookiePrior:
        b = _bucket(window_rates(train, "cur")["MPG"])
        sums = train[[f"y_{c}" for c in TOTALS]].groupby(b).sum()
        sums.columns = TOTALS
        rates = pd.DataFrame(index=sums.index)
        for s in RATE_STATS:
            rates[s] = sums[s] / sums["MIN"]
        for c, (made, att) in PCT_STATS.items():
            rates[c] = sums[made] / sums[att]
        return cls(rates)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rates for each row from its minutes-so-far band; minutes are his minutes so far."""
        cur = window_rates(df, "cur")
        out = self.rates.reindex(_bucket(cur["MPG"])).set_axis(df.index)
        out.insert(0, "MPG", cur["MPG"])
        return out


def preseason_prior(df: pd.DataFrame, rookie: RookiePrior | None = None) -> pd.DataFrame:
    """The stand-in for a preseason projection, as a per-game line."""
    return per_game(prior_rates(df, rookie))


def prior_rates(df: pd.DataFrame, rookie: RookiePrior | None = None) -> pd.DataFrame:
    """The stand-in projection as minutes per game, per-minute rates and percentages."""
    weights = {p: w * (df[f"{p}_MIN"].fillna(0) > 0) for p, w in PRIOR_WEIGHTS.items()}
    pooled = pd.DataFrame(index=df.index)
    for c in ["G", *TOTALS]:
        pooled[f"pr_{c}"] = sum(weights[p] * df[f"{p}_{c}"].fillna(0) for p in PRIOR_WEIGHTS)
    rates = window_rates(pooled, "pr")
    mpg = window_rates(df, "p1")["MPG"]
    rates["MPG"] = mpg.where(mpg.notna(), window_rates(df, "p2")["MPG"])
    if rookie is not None:
        none = rates[RATE_STATS].isna().any(axis=1) | rates["MPG"].isna()
        rates.loc[none] = rookie.predict(df.loc[none])[rates.columns].to_numpy()
    return rates


def season_to_date(df: pd.DataFrame) -> pd.DataFrame:
    return per_game(window_rates(df, "cur"))


def last_season(df: pd.DataFrame) -> pd.DataFrame:
    last = per_game(window_rates(df, "p1"))
    return last.where(last["MIN"].notna(), season_to_date(df), axis=0)


def blend(df: pd.DataFrame, rookie: RookiePrior | None = None) -> pd.DataFrame:
    """The app's Blend: projection 60%, last season 40% (the projection alone without last season)."""
    last = per_game(window_rates(df, "p1"))
    return mix(preseason_prior(df, rookie), last, BLEND_PROJECTION)


def interim(df: pd.DataFrame, rookie: RookiePrior | None = None, k: float = ROS_PRIOR_GAMES) -> pd.DataFrame:
    """overview.ros_line(): w = GP / (GP + k) on this season, the rest on the preseason projection."""
    g = df["cur_G"].astype(float)
    return mix(season_to_date(df), preseason_prior(df, rookie), g / (g + k))


def all_baselines(df: pd.DataFrame, rookie: RookiePrior) -> dict[str, pd.DataFrame]:
    return {"Season to date": season_to_date(df), "Last season": last_season(df),
            "Blend 60/40": blend(df, rookie), "Interim (app today)": interim(df, rookie)}
