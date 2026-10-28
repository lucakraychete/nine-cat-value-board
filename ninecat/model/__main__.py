"""Training CLI: the one part of ninecat.model that reads files.

    python -m ninecat.model baselines    # baseline backtest on the validation season
    python -m ninecat.model shrinkage    # fit model 2 on train, compare on validation
    python -m ninecat.model boosting     # model 3 vs model 2 on validation, with a bootstrap
    python -m ninecat.model train        # fit, score once on the test season, save data/models/
    python -m ninecat.model score        # model and ESPN snapshots vs what has happened since

Training is never part of boot; the app's refresh only predicts.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import pandas as pd

from ninecat.ingest import DATA_DIR, HISTORY_DIR, MODEL_FILE, MODELS_DIR, REPORT_FILE, write_json
from ninecat.model.baselines import RookiePrior, all_baselines
from ninecat.model.dataset import VALIDATE, build_dataset, split_of
from ninecat.model.evaluate import backtest_table, mae_grid
from ninecat.model.shrinkage import MINUTES_FEATURES, equivalent_games, fit_shrinkage


def load_history(data_dir: Path = DATA_DIR) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Every season's logs and totals under data/history, by season label."""
    d = data_dir / HISTORY_DIR
    logs = {p.name[5:12]: pd.read_csv(p) for p in sorted(d.glob("logs_*.csv.gz"))}
    totals = {p.name[7:14]: pd.read_csv(p) for p in sorted(d.glob("totals_*.csv.gz"))}
    return logs, totals


def dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    logs, totals = load_history(data_dir)
    seasons = [s for s in logs if split_of(s) != "other"]
    df = build_dataset(logs, totals, seasons)
    df["split"] = df["season"].map(split_of)
    return df


def run_baselines() -> None:
    df = dataset()
    train, val = df[df["split"] == "train"], df[df["split"] == "validate"].reset_index(drop=True)
    print(f"checkpoints: train {len(train)}, validate {len(val)} ({', '.join(VALIDATE)})")
    rookie = RookiePrior.fit(train)
    table = backtest_table(val, all_baselines(val, rookie))
    pd.set_option("display.width", 200)
    print("\nValidation MAE per game, weighted by remaining minutes, averaged over checkpoints:")
    print(mae_grid(table).round(3).to_string())
    for k in sorted(table["k"].unique()):
        print(f"\nk = {k}:")
        print(mae_grid(table, k).round(3).to_string())


def run_shrinkage() -> None:
    df = dataset()
    train, val = df[df["split"] == "train"], df[df["split"] == "validate"].reset_index(drop=True)
    plain, aged = fit_shrinkage(train, age_adjust=False), fit_shrinkage(train, age_adjust=True)
    rookie = plain.rookie
    preds = {**all_baselines(val, rookie), "Shrinkage": plain.predict(val), "Shrinkage + age": aged.predict(val)}
    table = backtest_table(val, preds)
    pd.set_option("display.width", 200)
    print("Validation MAE per game, weighted by remaining minutes, averaged over checkpoints:")
    print(mae_grid(table).round(3).to_string())
    for k in sorted(table["k"].unique()):
        print(f"\nk = {k}:")
        print(mae_grid(table, k).round(3).to_string())
    for name, m in (("no age", plain), ("age", aged)):
        eq = equivalent_games(m, train)
        print(f"\nLearned k ({name}): stat, k (minutes; attempts for FG%/FT%), games at 30 min")
        for s, v in m.k.items():
            print(f"  {s:4s} {v:9.0f}  {eq[s]:6.1f}")
    print("\nAge factors on the prior (rows: <23, 23-25, 26-28, 29-31, 32+):")
    print(aged.age.factors.round(3).to_string())
    print("\nMinutes model coefficients by checkpoint:", MINUTES_FEATURES)
    for k, c in aged.minutes.coef.items():
        print(f"  k={k:2d}", " ".join(f"{v:+.3f}" for v in c))


def run_boosting() -> None:
    import time

    from ninecat.model.boosting import compare_models, fit_boosted

    df = dataset()
    train, val = df[df["split"] == "train"], df[df["split"] == "validate"].reset_index(drop=True)
    base = fit_shrinkage(train)
    t = time.perf_counter()
    boosted = fit_boosted(train, base)
    print(f"fit {time.perf_counter() - t:.1f}s")
    t = time.perf_counter()
    m3 = boosted.predict(val)
    print(f"predict {len(val)} rows {time.perf_counter() - t:.2f}s")
    m2 = base.predict(val)
    table = backtest_table(val, {"Shrinkage + age": m2, "Boosted": m3})
    pd.set_option("display.width", 200)
    print("\nValidation MAE, averaged over checkpoints:")
    print(mae_grid(table).round(4).to_string())
    for k in sorted(table["k"].unique()):
        print(f"\nk = {k}:")
        print(mae_grid(table, k).round(4).to_string())
    cmp = compare_models(val, m2, m3)
    print("\nRelative change in weighted MAE, boosted vs shrinkage (negative = boosted better), "
          "95% bootstrap over players:")
    print((cmp.by_cat * 100).round(2).to_string())
    print(f"average over categories: {cmp.overall[0] * 100:+.2f}% [{cmp.overall[1] * 100:+.2f}%, "
          f"{cmp.overall[2] * 100:+.2f}%] -> {'keep the challenger' if cmp.challenger_wins else 'keep model 2'}")


def save_model(model, data_dir: Path = DATA_DIR) -> Path:
    """Pickle atomically (the app may load while this writes)."""
    path = data_dir / MODELS_DIR / MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(model, f)
    os.replace(tmp, path)
    return path


def run_train() -> None:
    from ninecat.model.train import train_and_report

    t = time.perf_counter()
    df = dataset()
    print(f"checkpoints: {df['split'].value_counts().to_dict()}")
    model, report = train_and_report(df)
    path = save_model(model)
    write_json(report, DATA_DIR / MODELS_DIR / REPORT_FILE)
    print(f"saved {path} and {REPORT_FILE} in {time.perf_counter() - t:.0f}s; chosen: {report['chosen']}")
    test = pd.DataFrame(report["backtest"]["test"])
    pd.set_option("display.width", 200)
    print("\nTest season (2025-26), weighted MAE averaged over checkpoints:")
    print(mae_grid(test).round(3).to_string())
    for k in sorted(test["k"].unique()):
        print(f"\nk = {k}:")
        print(mae_grid(test, k).round(3).to_string())
    rank = pd.DataFrame(report["value_rank"]["test"])
    print("\nSpearman, predicted vs actual rest-of-season 8-cat value, top 200 (test):")
    print(rank.pivot_table(index="model", columns="k", values="spearman", sort=False).round(3)
          .assign(mean=lambda f: f.mean(axis=1).round(3)).to_string())
    print("\nAvailability, MAE in games left (test):")
    print(pd.DataFrame(report["availability"]["test"]).pivot_table(index="model", columns="k", values="mae_games")
          .round(2).to_string())
    cov = pd.DataFrame(report["coverage_test"])
    print("\n80% interval coverage on test (all checkpoints):")
    print(cov[cov["k"] == 0].set_index("cat")["coverage"].round(3).to_string())
    print("\nVerdict:", report["test_verdict"], "\nmodel beats blend:", report["model_beats_blend"])


def run_score() -> None:
    from ninecat.ingest import load_game_logs, snapshot_files
    from ninecat.model.live import score_snapshots

    logs = load_game_logs()
    current = logs[logs["season"] == logs["season"].max()] if not logs.empty else logs
    table = score_snapshots(snapshot_files(), current)
    if table.empty:
        print("Nothing to score yet: snapshots need players with games after their date.")
        return
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ninecat.model", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["baselines", "shrinkage", "boosting", "train", "score"])
    args = parser.parse_args()
    {"baselines": run_baselines, "shrinkage": run_shrinkage, "boosting": run_boosting,
     "train": run_train, "score": run_score}[args.command]()


if __name__ == "__main__":
    main()
