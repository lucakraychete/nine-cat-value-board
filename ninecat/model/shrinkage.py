"""Model 2: empirical-Bayes shrinkage, the main rest-of-season model.

Per-minute rates. Each stat's rest-of-season rate is his rate so far,
pulled toward his prior in proportion to how little he's played:

    rate = (n x so_far + k x prior) / (n + k)

with n his minutes so far and k learned per stat on the training seasons.
k is "how many minutes of evidence the prior is worth": a stat with a big k
(steals, blocks) is mostly noise over a few games, so the prior holds on
long; a stat with a small k (points, field-goal attempts) shows who he is
quickly. FG% and FT% shrink the same way with n and k in shot attempts.
The prior is baselines.prior_rates() (last two seasons 2:1, or the rookie
pool at his minutes), optionally scaled by an age curve.

Minutes per game move for different reasons (a role change, not luck), so
they get their own small model: a linear regression per checkpoint,
weighted by games left, on his minutes so far, over his last 5, last
season's, whether he has a prior season, the change in his start rate
(last 5 vs season so far) and age.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.model.baselines import RookiePrior, prior_rates
from ninecat.model.dataset import PCT_STATS, RATE_STATS, per_game, window_rates

K_GRID = np.geomspace(1, 1e5, 241)
AGE_BANDS = (0, 23, 26, 29, 32, 99)          # under 23, 23-25, 26-28, 29-31, 32+
MINUTES_FEATURES = ("const", "so_far", "last5", "prior", "has_prior", "start_change", "age")
MAX_MPG = 48.0


def shrink(so_far: pd.Series, n: pd.Series, prior: pd.Series, k: float) -> pd.Series:
    """(n x so_far + k x prior) / (n + k); the prior alone with no evidence, the evidence alone with no prior."""
    n = n.fillna(0).clip(lower=0)
    mixed = (n * so_far.fillna(0) + k * prior) / (n + k)
    return mixed.where(prior.notna(), so_far.where(n > 0))


def fit_k(so_far: pd.Series, n: pd.Series, prior: pd.Series, actual: pd.Series, weight: pd.Series,
          grid: np.ndarray = K_GRID) -> float:
    """The k minimizing weighted squared error of shrink() against what happened (grid, then a finer grid)."""
    ok = prior.notna() & actual.notna() & weight.notna() & (weight > 0)
    s, nn, p = so_far[ok].fillna(0).to_numpy(), n[ok].fillna(0).to_numpy(), prior[ok].to_numpy()
    a, w = actual[ok].to_numpy(), weight[ok].to_numpy()

    def best(ks: np.ndarray) -> float:
        pred = (nn[:, None] * s[:, None] + ks[None, :] * p[:, None]) / (nn[:, None] + ks[None, :])
        sse = (w[:, None] * (pred - a[:, None]) ** 2).sum(0)
        return float(ks[int(np.argmin(sse))])

    coarse = best(grid)
    return best(np.geomspace(coarse / 1.1, coarse * 1.1, 81))


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, right=False, labels=False)


@dataclass(frozen=True)
class AgeCurve:
    """Multiplier on prior per-minute rates by age band: how players that age tend to
    change from what their last two seasons say (young players improve, old ones fade)."""
    factors: pd.DataFrame   # index: age band, columns: RATE_STATS

    @classmethod
    def fit(cls, train: pd.DataFrame, rookie: RookiePrior) -> AgeCurve:
        has = train["p1_MIN"].fillna(0) > 0
        t = train[has]
        prior = prior_rates(t, rookie)
        band = _age_band(t["age"])
        f = pd.DataFrame({s: t[f"y_{s}"].groupby(band).sum() / (prior[s] * t["y_MIN"]).groupby(band).sum()
                          for s in RATE_STATS})
        return cls(f)

    def apply(self, df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
        has = df["p1_MIN"].fillna(0) > 0
        f = self.factors.reindex(_age_band(df["age"])).set_axis(df.index).fillna(1.0)
        out = rates.copy()
        out.loc[has, RATE_STATS] = rates.loc[has, RATE_STATS] * f.loc[has, RATE_STATS]
        return out


@dataclass(frozen=True)
class MinutesModel:
    """Minutes per game over the rest of the season: one games-weighted linear fit per checkpoint."""
    coef: dict[int, np.ndarray]

    @staticmethod
    def features(df: pd.DataFrame, prior_mpg: pd.Series) -> np.ndarray:
        cur = window_rates(df, "cur")["MPG"].fillna(prior_mpg)     # no games yet: his prior
        l5 = window_rates(df, "l5")["MPG"]
        has = prior_mpg.notna() & ((df["p1_MIN"].fillna(0) > 0) | (df["p2_MIN"].fillna(0) > 0))
        start = df["l5_GS"] / df["l5_G"] - df["cur_GS"] / df["cur_G"]
        cols = {"const": 1.0, "so_far": cur, "last5": l5.fillna(cur), "prior": prior_mpg.where(has, cur),
                "has_prior": has.astype(float), "start_change": start.fillna(0),
                "age": (df["age"] - 27).fillna(0)}
        return np.column_stack([np.broadcast_to(np.asarray(cols[f], float), len(df)) for f in MINUTES_FEATURES])

    @classmethod
    def fit(cls, train: pd.DataFrame, prior_mpg: pd.Series) -> MinutesModel:
        coef = {}
        for k, idx in train.groupby("k").groups.items():
            x = cls.features(train.loc[idx], prior_mpg.loc[idx])
            y = window_rates(train.loc[idx], "y")["MPG"].to_numpy(float)
            w = np.sqrt(train.loc[idx, "y_G"].to_numpy(float))
            ok = np.isfinite(x).all(1) & np.isfinite(y)
            coef[int(k)] = np.linalg.lstsq(x[ok] * w[ok, None], y[ok] * w[ok], rcond=None)[0]
        return cls(coef)

    def checkpoint_for(self, games: pd.Series) -> pd.Series:
        """The largest fitted checkpoint at or below his games played (the smallest one below that)."""
        ks = np.array(sorted(self.coef))
        pos = np.searchsorted(ks, games.fillna(0).to_numpy(float), side="right") - 1
        return pd.Series(ks[np.clip(pos, 0, len(ks) - 1)], index=games.index)

    def predict(self, df: pd.DataFrame, prior_mpg: pd.Series) -> pd.Series:
        x = self.features(df, prior_mpg)
        k = self.checkpoint_for(df["k"] if "k" in df else df["cur_G"])
        beta = np.stack([self.coef[int(v)] for v in k])
        return pd.Series(np.clip((x * beta).sum(1), 0, MAX_MPG), index=df.index)


@dataclass(frozen=True)
class ShrinkageModel:
    k: dict[str, float]          # rate stats: minutes; FG% / FT%: attempts
    minutes: MinutesModel
    rookie: RookiePrior
    age: AgeCurve | None

    def prior(self, df: pd.DataFrame) -> pd.DataFrame:
        rates = prior_rates(df, self.rookie)
        return self.age.apply(df, rates) if self.age is not None else rates

    def rates(self, df: pd.DataFrame) -> pd.DataFrame:
        prior, cur = self.prior(df), window_rates(df, "cur")
        out = pd.DataFrame(index=df.index)
        out["MPG"] = self.minutes.predict(df, prior["MPG"])
        for s in RATE_STATS:
            out[s] = shrink(cur[s], df["cur_MIN"], prior[s], self.k[s])
        for c, (_, att) in PCT_STATS.items():
            out[c] = shrink(cur[c], df[f"cur_{att}"], prior[c], self.k[c])
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return per_game(self.rates(df))


def fit_shrinkage(train: pd.DataFrame, age_adjust: bool = True) -> ShrinkageModel:
    """Learn the rookie pool, the age curve (optional), k per stat and the minutes model on `train`."""
    rookie = RookiePrior.fit(train)
    age = AgeCurve.fit(train, rookie) if age_adjust else None
    model = ShrinkageModel({}, MinutesModel({}), rookie, age)
    prior, cur = model.prior(train), window_rates(train, "cur")
    actual = window_rates(train, "y")
    k = {s: fit_k(cur[s], train["cur_MIN"], prior[s], actual[s], train["y_MIN"]) for s in RATE_STATS}
    for c, (_, att) in PCT_STATS.items():
        k[c] = fit_k(cur[c], train[f"cur_{att}"], prior[c], actual[c], train[f"y_{att}"])
    return ShrinkageModel(k, MinutesModel.fit(train, prior["MPG"]), rookie, age)


def equivalent_games(model: ShrinkageModel, train: pd.DataFrame, mpg: float = 30.0) -> dict[str, float]:
    """k as games at `mpg` minutes a game; for FG%/FT%, at the pool's attempts per minute."""
    out = {s: model.k[s] / mpg for s in RATE_STATS}
    for c, (_, att) in PCT_STATS.items():
        per_min = train[f"y_{att}"].sum() / train["y_MIN"].sum()
        out[c] = model.k[c] / (per_min * mpg)
    return out
