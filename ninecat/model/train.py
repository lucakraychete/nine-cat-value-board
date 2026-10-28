"""Fit, evaluate and choose: everything `python -m ninecat.model train` does except file I/O.

1. Fit shrinkage and the boosting challenger on the training seasons, keep
   boosting only if it beats shrinkage on validation by more than noise.
2. Score every model and baseline on validation and, once, on the test
   season: per-category weighted MAE/RMSE by checkpoint, and the top-200
   rank check on rest-of-season 8-cat value.
3. Fit prediction intervals on the validation residuals and report their
   coverage on the test season.
4. Refit the chosen model and availability on every season for live use.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ninecat.model.baselines import RookiePrior, all_baselines
from ninecat.model.boosting import GB_PARAMS, TARGETS, compare_models, fit_boosted
from ninecat.model.dataset import RATE_STATS, TEST, TRAIN, VALIDATE
from ninecat.model.evaluate import EVAL_CATS, backtest_table, coverage, fit_intervals, mae_grid, value_rank_check
from ninecat.model.games import AvailabilityModel, prior_availability, so_far_availability
from ninecat.model.predict import RosModel
from ninecat.model.shrinkage import MINUTES_FEATURES, equivalent_games, fit_shrinkage

MODEL_NAMES = {"shrinkage": "Shrinkage", "boosted": "Boosted"}


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
            for r in frame.to_dict("records")]


def availability_table(df: pd.DataFrame, model: AvailabilityModel) -> pd.DataFrame:
    """MAE in games left: the model against last seasons' share and this season's share alone."""
    actual = df["y_G"].astype(float)
    left = df["team_left"].astype(float)
    preds = {"Model": model.predict(df), "Last two seasons": prior_availability(df).fillna(model.newcomer),
             "This season so far": so_far_availability(df)}
    rows = []
    for name, a in preds.items():
        for k, idx in df.groupby("k").groups.items():
            err = (a.loc[idx] * left.loc[idx] - actual.loc[idx]).abs()
            rows.append({"model": name, "k": int(k), "mae_games": float(err.mean())})
    return pd.DataFrame(rows)


def relative_change(table: pd.DataFrame, model: str, against: str) -> float:
    """Average over categories of model MAE / other MAE - 1 (negative = model better)."""
    g = mae_grid(table)
    return float((g.loc[model] / g.loc[against] - 1).mean())


def train_and_report(df: pd.DataFrame, boost_params: dict | None = None,
                     boost_targets: Sequence[str] = TARGETS, n_boot: int = 1000) -> tuple[RosModel, dict]:
    train = df[df["season"].isin(TRAIN)]
    val = df[df["season"].isin(VALIDATE)].reset_index(drop=True)
    test = df[df["season"].isin(TEST)].reset_index(drop=True)

    shrink = fit_shrinkage(train)
    boosted = fit_boosted(train, shrink, boost_params or GB_PARAMS, boost_targets)
    cmp = compare_models(val, shrink.predict(val), boosted.predict(val), n_boot=n_boot)
    kind = "boosted" if cmp.challenger_wins else "shrinkage"
    rookie = RookiePrior.fit(train)

    def preds(part: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {**all_baselines(part, rookie), "Shrinkage": shrink.predict(part), "Boosted": boosted.predict(part)}

    val_preds, test_preds = preds(val), preds(test)
    val_table, test_table = backtest_table(val, val_preds), backtest_table(test, test_preds)
    rank_val, rank_test = value_rank_check(val, val_preds), value_rank_check(test, test_preds)

    intervals = fit_intervals(val, val_preds[MODEL_NAMES[kind]])
    cov = coverage(test, test_preds[MODEL_NAMES[kind]], intervals)

    avail = AvailabilityModel.fit(train)
    avail_val, avail_test = availability_table(val, avail), availability_table(test, avail)

    name = MODEL_NAMES[kind]
    spear = rank_test.groupby("model")["spearman"].mean()
    verdict = {
        f"vs {other}": {"mae_change": relative_change(test_table, name, other),
                        "spearman": float(spear[name]), "spearman_other": float(spear[other])}
        for other in ("Blend 60/40", "Interim (app today)")
    }
    beats_blend = verdict["vs Blend 60/40"]["mae_change"] < 0 and spear[name] > spear["Blend 60/40"]

    # live model: the chosen kind refit on every season
    all_rows = df[df["season"].isin((*TRAIN, *VALIDATE, *TEST))]
    live_shrink = fit_shrinkage(all_rows)
    live = fit_boosted(all_rows, live_shrink, boost_params or GB_PARAMS, boost_targets) if kind == "boosted" \
        else live_shrink
    seasons = tuple(sorted(all_rows["season"].unique()))
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model = RosModel(live, AvailabilityModel.fit(all_rows), intervals, kind, seasons, created)

    report = {
        "created": created,
        "seasons": {"train": list(TRAIN), "validate": list(VALIDATE), "test": list(TEST), "live_fit": list(seasons)},
        "checkpoints": {"train": len(train), "validate": len(val), "test": len(test)},
        "chosen": kind,
        "boosting_vs_shrinkage": {"by_cat": {c: dict(r) for c, r in cmp.by_cat.iterrows()},
                                  "overall": dict(zip(("change", "low", "high"), cmp.overall))},
        "k": {s: {"k": shrink.k[s], "games_at_30": g} for s, g in equivalent_games(shrink, train).items()},
        "k_units": {s: ("attempts" if s in ("FG%", "FT%") else "minutes") for s in shrink.k},
        "availability": {"k_team_games": avail.k, "newcomer": avail.newcomer,
                         "validate": _records(avail_val), "test": _records(avail_test)},
        "age_factors": {str(b): {s: float(v) for s, v in r.items()} for b, r in shrink.age.factors.iterrows()},
        "minutes_coef": {str(k): dict(zip(MINUTES_FEATURES, map(float, c))) for k, c in shrink.minutes.coef.items()},
        "backtest": {"validate": _records(val_table), "test": _records(test_table)},
        "value_rank": {"validate": _records(rank_val), "test": _records(rank_test)},
        "intervals": _records(intervals),
        "coverage_test": _records(cov),
        "test_verdict": verdict,
        "model_beats_blend": bool(beats_blend),
        "eval_cats": EVAL_CATS,
        "rate_stats": RATE_STATS,
    }
    return model, report
