"""Games played over the rest of the season (PLAN.md 5.3).

Availability is the share of his team's games he plays. So far it's his
games over his team's games before the cut; his prior is the same share
over his last two seasons (each season's own length, so the shortened
2019-20 and 2020-21 seasons count fairly), and the training pool's average
for players with neither. The two are mixed like the rates:

    availability = (team games x so far + k x prior) / (team games + k)

with k (in team games) fit on the training seasons. Projected games are
his NBA team's remaining games times availability, less what an injury
costs: a player listed OUT misses every game before his expected return
date when ESPN has one, else the rest of the current week.

One bias to know: checkpoints need 5+ games after the cut, so players who
got hurt for the season right after it aren't in the data, and the fitted
availability runs a little high. See docs/MODEL.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.model.shrinkage import fit_k, shrink


def prior_availability(df: pd.DataFrame) -> pd.Series:
    """Games over team games in his last two seasons, pooled (NaN with neither)."""
    g = sum(df[f"{p}_G"].fillna(0) for p in ("p1", "p2"))
    t = sum(df[f"{p}_team_games"].where(df[f"{p}_G"].fillna(0) > 0, 0) for p in ("p1", "p2"))
    return (g / t.where(t > 0)).clip(upper=1)


def so_far_availability(df: pd.DataFrame) -> pd.Series:
    return (df["cur_G"] / df["team_games"].where(df["team_games"] > 0)).clip(upper=1)


def actual_availability(df: pd.DataFrame) -> pd.Series:
    return (df["y_G"] / df["team_left"].where(df["team_left"] > 0)).clip(upper=1)


@dataclass(frozen=True)
class AvailabilityModel:
    k: float              # team games of evidence the prior is worth
    newcomer: float       # prior for players with no previous season

    @classmethod
    def fit(cls, train: pd.DataFrame) -> AvailabilityModel:
        prior = prior_availability(train)
        actual, w = actual_availability(train), train["team_left"].astype(float)
        newcomer = float(np.average(actual[prior.isna() & actual.notna()],
                                    weights=w[prior.isna() & actual.notna()])) if (prior.isna() & actual.notna()).any() else 0.8
        prior = prior.fillna(newcomer)
        return cls(fit_k(so_far_availability(train), train["team_games"].astype(float), prior, actual, w), newcomer)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        prior = prior_availability(df).fillna(self.newcomer)
        return shrink(so_far_availability(df), df["team_games"].astype(float), prior, self.k).clip(0, 1)


def projected_games(availability: pd.Series, remaining: pd.Series, out: pd.Series,
                    before_return: pd.Series, this_week: pd.Series) -> pd.Series:
    """Rest-of-season games: team games left x availability, minus the games an injury costs.

    out marks players listed OUT; before_return is his team's games before
    ESPN's expected return date (NaN when there's no date), this_week his
    team's games left this week.
    """
    lost = before_return.where(before_return.notna(), this_week).where(out, 0).fillna(0)
    return ((remaining - lost).clip(lower=0) * availability).fillna(0)
