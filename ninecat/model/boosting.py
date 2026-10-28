"""Model 3: a gradient-boosting challenger on top of the shrinkage model.

For each target (minutes per game, each per-minute rate, FG%, FT%) a
HistGradientBoostingRegressor learns the shrinkage model's residual, what
it got wrong on the training seasons, from everything known at the cut:
the windows' rates (season so far, last 5 and 10, last two seasons), age,
games share, start rates, the checkpoint, and model 2's own prediction.
The prediction is model 2 plus the learned correction.

It only replaces model 2 if it beats it on the validation season by more
than noise: compare_models() bootstraps over players (each player's
checkpoints stay together, since they share a season) for the change in
error, and the challenger is kept only if the 95% interval for the average
relative change is entirely below zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ninecat.model.dataset import PCT_STATS, RATE_STATS, per_game, target_line, window_rates
from ninecat.model.evaluate import EVAL_CATS
from ninecat.model.shrinkage import ShrinkageModel

TARGETS = ["MPG", *RATE_STATS, *PCT_STATS]
GB_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
             "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
BOOTSTRAPS = 1000


def features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Everything known at the cut, plus model 2's prediction (`base`, rates)."""
    out = pd.DataFrame(index=df.index)
    for w in ("cur", "l5", "l10", "p1", "p2"):
        r = window_rates(df, w)
        for c in r.columns:
            out[f"{w}_{c}"] = r[c]
        out[f"{w}_start_rate"] = df[f"{w}_GS"] / df[f"{w}_G"].where(df[f"{w}_G"] > 0)
    for c in base.columns:
        out[f"m2_{c}"] = base[c]
    out["k"] = df["k"].astype(float)
    out["age"] = df["age"]
    out["games_share"] = df["games_share"]
    return out


def _weights(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "MPG":
        return df["y_G"].astype(float)
    if target in PCT_STATS:
        return df[f"y_{PCT_STATS[target][1]}"].astype(float)
    return df["y_MIN"].astype(float)


@dataclass
class BoostedModel:
    base: ShrinkageModel
    models: dict[str, HistGradientBoostingRegressor]

    def rates(self, df: pd.DataFrame) -> pd.DataFrame:
        base = self.base.rates(df)
        x = features(df, base)
        out = base.copy()
        for t, m in self.models.items():
            out[t] = base[t] + m.predict(x)
        out["MPG"] = out["MPG"].clip(0, 48)
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return per_game(self.rates(df))


def fit_boosted(train: pd.DataFrame, base: ShrinkageModel, params: dict | None = None,
                targets: Sequence[str] = TARGETS) -> BoostedModel:
    """One residual regressor per target, weighted like the shrinkage fit (targets left out keep model 2)."""
    pred = base.rates(train)
    actual = window_rates(train, "y")
    x = features(train, pred)
    models = {}
    for t in targets:
        resid, w = actual[t] - pred[t], _weights(train, t)
        ok = resid.notna() & w.notna() & (w > 0)
        m = HistGradientBoostingRegressor(**(params or GB_PARAMS))
        models[t] = m.fit(x[ok], resid[ok], sample_weight=w[ok])
    return BoostedModel(base, models)


# ---------------------------------------------------------------- is it better than noise?

@dataclass(frozen=True)
class Comparison:
    """Challenger vs incumbent on one season: per-category relative change in weighted MAE
    (negative = challenger better) with bootstrap 95% intervals, and the average over categories."""
    by_cat: pd.DataFrame        # index: category; columns: change, low, high
    overall: tuple[float, float, float]

    @property
    def challenger_wins(self) -> bool:
        return self.overall[2] < 0


def compare_models(df: pd.DataFrame, incumbent: pd.DataFrame, challenger: pd.DataFrame,
                   cats: Sequence[str] = EVAL_CATS, n_boot: int = BOOTSTRAPS, seed: int = 0) -> Comparison:
    """Bootstrap over players for the relative change in minutes-weighted MAE."""
    actual = target_line(df)
    w = df["y_MIN"].astype(float)
    players, idx = np.unique(df["nba_id"].to_numpy(), return_inverse=True)
    sums = np.zeros((3, len(players), len(cats)))     # weighted |err| incumbent, challenger, weight
    for j, c in enumerate(cats):
        ok = (actual[c].notna() & incumbent[c].notna() & challenger[c].notna() & (w > 0)).to_numpy()
        for s, arr in enumerate((np.abs(incumbent[c] - actual[c]) * w, np.abs(challenger[c] - actual[c]) * w, w)):
            np.add.at(sums[s, :, j], idx[ok], arr.to_numpy(float)[ok])

    def rel(counts: np.ndarray) -> np.ndarray:            # counts: (draws, players)
        a, b, ww = (counts @ sums[s] for s in range(3))
        return (b / ww) / (a / ww) - 1

    point = rel(np.ones((1, len(players))))[0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=n_boot).astype(float)
    boot = rel(draws)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    mean_boot = boot.mean(1)
    by_cat = pd.DataFrame({"change": point, "low": lo, "high": hi}, index=list(cats))
    return Comparison(by_cat, (float(point.mean()), *map(float, np.percentile(mean_boot, [2.5, 97.5]))))
