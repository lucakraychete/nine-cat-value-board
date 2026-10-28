"""Backtest errors: how far each prediction lands from the rest of the season.

Errors are per game and weighted by the minutes he went on to play, so a
miss on a 34-minute starter counts more than one on a 6-minute reserve.
FG% and FT% are compared as percentages, over rows where he took the shot
at all after the cut.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ninecat.config import categories
from ninecat.model.dataset import PCT_STATS, RATE_STATS, target_line
from ninecat.valuation import category_zscores, value_players

EVAL_CATS = ["MIN", *RATE_STATS[:7], *PCT_STATS]   # MIN, PTS..TO, FG%, FT%


def weighted_errors(pred: pd.Series, actual: pd.Series, weight: pd.Series) -> tuple[float, float, int]:
    """(MAE, RMSE, rows) weighted by `weight`, over rows where both are known."""
    ok = pred.notna() & actual.notna() & weight.notna() & (weight > 0)
    if not ok.any():
        return np.nan, np.nan, 0
    e, w = (pred[ok] - actual[ok]).to_numpy(float), weight[ok].to_numpy(float)
    return float(np.average(np.abs(e), weights=w)), float(np.sqrt(np.average(e ** 2, weights=w))), int(ok.sum())


def backtest_table(df: pd.DataFrame, preds: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model, checkpoint and category: mae, rmse, n."""
    actual = target_line(df)
    weight = df["y_MIN"].astype(float)
    rows = []
    for name, pred in preds.items():
        for k, idx in df.groupby("k").groups.items():
            for c in EVAL_CATS:
                mae, rmse, n = weighted_errors(pred.loc[idx, c], actual.loc[idx, c], weight.loc[idx])
                rows.append({"model": name, "k": int(k), "cat": c, "mae": mae, "rmse": rmse, "n": n})
    return pd.DataFrame(rows)


def mae_grid(table: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Models x categories MAE, at one checkpoint or averaged over all."""
    t = table if k is None else table[table["k"] == k]
    return t.pivot_table(index="model", columns="cat", values="mae", aggfunc="mean", sort=False)[EVAL_CATS]


# ---------------------------------------------------------------- fantasy value

VALUE_CATS = categories(exclude_to=True)     # the 8-cat value the rank check uses
TOP = 200


def _valued(line: pd.DataFrame, pool_size: int = 130) -> pd.DataFrame:
    """value_players on a per-game line, no games discount, back in the line's row order."""
    v = value_players(line.assign(GP=82.0, _row=line.index), VALUE_CATS, pool_size=pool_size, games_discount=False)
    return v.set_index("_row").loc[line.index]


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation (Pearson on ranks; ties share their average rank)."""
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


def value_rank_check(df: pd.DataFrame, preds: Mapping[str, pd.DataFrame], top: int = TOP) -> pd.DataFrame:
    """Spearman between predicted and actual rest-of-season 8-cat value for the top
    `top` players by actual value, per model and checkpoint (each season and k valued on its own)."""
    actual = target_line(df)
    rows = []
    for (season, k), idx in df.groupby(["season", "k"]).groups.items():
        act = _valued(actual.loc[idx].dropna(subset=["PTS", "MIN"]))
        best = act["value"].nlargest(top).index
        for name, pred in preds.items():
            p = pred.loc[idx].dropna(subset=["PTS", "MIN"])
            pv = _valued(p)["value"]
            rows.append({"model": name, "season": season, "k": int(k),
                         "spearman": spearman(pv.reindex(best), act.loc[best, "value"]), "n": len(best)})
    return pd.DataFrame(rows)


def value_residuals(df: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    """Actual minus predicted per-game 8-cat value, both on the actual season's z scale."""
    actual = target_line(df)
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(["season", "k"]).groups.items():
        act = actual.loc[idx].dropna(subset=["PTS", "MIN"])
        prd = pred.loc[act.index].dropna(subset=["PTS", "MIN"])
        act = act.loc[prd.index]
        pool = _valued(act)["in_pool"]
        both = pd.concat([act, prd], keys=["a", "p"])
        both_pool = pd.concat([pool, pd.Series(False, index=prd.index)], keys=["a", "p"])
        z = category_zscores(both, VALUE_CATS, both_pool).sum(axis=1)
        out.loc[act.index] = (z.loc["a"] - z.loc["p"]).to_numpy()
    return out


# ---------------------------------------------------------------- prediction intervals

INTERVAL = (0.1, 0.9)                     # an 80% interval
MPG_BANDS = (0, 15, 25, 32, 60)
MIN_GROUP = 40
INTERVAL_CATS = [*EVAL_CATS, "VALUE"]


def mpg_band(minutes: pd.Series) -> pd.Series:
    return pd.cut(minutes, MPG_BANDS, right=False, labels=False)


def residual_frame(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Actual minus predicted per game for every interval category."""
    actual = target_line(df)
    res = pd.DataFrame({c: actual[c] - pred[c] for c in EVAL_CATS}, index=df.index)
    res["VALUE"] = value_residuals(df, pred)
    return res


def fit_intervals(df: pd.DataFrame, pred: pd.DataFrame, q: tuple[float, float] = INTERVAL) -> pd.DataFrame:
    """Residual quantiles by checkpoint and predicted-minutes band (the checkpoint alone
    where a band has fewer than MIN_GROUP players). Columns: k, band, cat, lo, hi, n."""
    res = residual_frame(df, pred)
    keys = pd.DataFrame({"k": df["k"].astype(int), "band": mpg_band(pred["MIN"])}, index=df.index)
    rows = []
    for k in sorted(keys["k"].unique()):
        at_k = keys["k"] == k
        for band in [*range(len(MPG_BANDS) - 1), -1]:          # -1: every band at this checkpoint
            sel = at_k if band == -1 else at_k & (keys["band"] == band)
            for c in INTERVAL_CATS:
                r = res.loc[sel, c].dropna()
                if len(r) >= MIN_GROUP or band == -1:
                    rows.append({"k": int(k), "band": band, "cat": c, "lo": float(r.quantile(q[0])),
                                 "hi": float(r.quantile(q[1])), "n": len(r)})
    return pd.DataFrame(rows)


def interval_bounds(table: pd.DataFrame, k: pd.Series, minutes: pd.Series, cat: str) -> tuple[pd.Series, pd.Series]:
    """(lo, hi) residual offsets for each row: its checkpoint's band if fitted, else the whole checkpoint.

    k is mapped to the largest fitted checkpoint at or below it (the smallest below that).
    """
    t = table[table["cat"] == cat]
    ks = np.array(sorted(t["k"].unique()))
    pos = np.clip(np.searchsorted(ks, k.fillna(0).to_numpy(float), side="right") - 1, 0, len(ks) - 1)
    kk = pd.Series(ks[pos], index=k.index)
    band = mpg_band(minutes).fillna(-1).astype(int)
    by = t.set_index(["k", "band"])
    lo = pd.Series(np.nan, index=k.index)
    hi = pd.Series(np.nan, index=k.index)
    for i in k.index:
        key = (int(kk[i]), int(band[i])) if (int(kk[i]), int(band[i])) in by.index else (int(kk[i]), -1)
        lo[i], hi[i] = by.loc[key, "lo"], by.loc[key, "hi"]
    return lo, hi


def coverage(df: pd.DataFrame, pred: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Share of rows whose actual lands inside its interval, per category and checkpoint (and overall)."""
    res = residual_frame(df, pred)
    rows = []
    for c in INTERVAL_CATS:
        lo, hi = interval_bounds(table, df["k"], pred["MIN"], c)
        inside = (res[c] >= lo) & (res[c] <= hi)
        ok = res[c].notna()
        for k, idx in df.groupby("k").groups.items():
            sel = ok.loc[idx]
            rows.append({"cat": c, "k": int(k), "coverage": float(inside.loc[idx][sel].mean()), "n": int(sel.sum())})
        rows.append({"cat": c, "k": 0, "coverage": float(inside[ok].mean()), "n": int(ok.sum())})
    return pd.DataFrame(rows)
