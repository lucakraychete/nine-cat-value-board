"""The trained rest-of-season model as one object, and its live predictions.

RosModel bundles the per-game model (boosted or shrinkage, whichever won
validation), the availability model and the prediction intervals. Its
predictions come out as M_ columns keyed by name_key, the shape
load_cached() attaches to players and lines.stat_line("model") and
overview.ros_line() read:

- M_MIN, M_PTS ... M_FGM, M_FGA, M_FTM, M_FTA: per game, rest of season;
- M_avail (share of team games he'll play) and M_GP = M_avail x season
  games, so the games discount reads his projected rest-of-season share;
- M_G: projected games left (filled in by with_games());
- M_lo_<cat> / M_hi_<cat>: the 80% interval per game, from residual
  quantiles at his checkpoint and minutes. For VALUE (per-game 8-cat value)
  they're offsets from his predicted value, which needs a league to value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.lines import MODEL_PREFIX as M
from ninecat.model.dataset import PCT_STATS, RATE_STATS, current_rows, season_length, season_totals
from ninecat.model.evaluate import INTERVAL_CATS, interval_bounds
from ninecat.model.games import AvailabilityModel, projected_games

LINE_COLS = ["MIN", *RATE_STATS, *[m for m, _ in PCT_STATS.values()]]


@dataclass
class RosModel:
    rates_model: object              # ShrinkageModel or BoostedModel: .predict(rows) -> per-game line
    availability: AvailabilityModel
    intervals: pd.DataFrame          # evaluate.fit_intervals() output
    kind: str                        # "boosted" or "shrinkage"
    trained_on: tuple[str, ...]
    created: str

    def predict(self, rows: pd.DataFrame, season_games: float = 82.0) -> pd.DataFrame:
        """M_ columns for rows shaped like dataset.current_rows(); rows with no usable line are dropped."""
        line = self.rates_model.predict(rows)
        out = pd.DataFrame({"name_key": rows["name_key"].to_numpy(), "nba_id": rows["nba_id"].to_numpy()},
                           index=rows.index)
        for c in LINE_COLS:
            out[M + c] = line[c]
        out[M + "avail"] = self.availability.predict(rows)
        out[M + "GP"] = out[M + "avail"] * season_games
        out[M + "k"] = rows["k"].astype(float)
        for c in INTERVAL_CATS:
            lo, hi = interval_bounds(self.intervals, rows["k"].astype(float), line["MIN"], c)
            base = line[c] if c in line else pd.Series(0.0, index=rows.index)
            out[M + "lo_" + c], out[M + "hi_" + c] = base + lo, base + hi
        keep = out[[M + c for c in LINE_COLS]].notna().all(axis=1) & out["name_key"].notna()
        return out[keep].drop_duplicates("name_key").reset_index(drop=True)


def live_rows(current: pd.DataFrame, prev: pd.DataFrame | None, prev2: pd.DataFrame | None,
              ages_by_name: Mapping[str, float], name_keys: Iterable[str], season_games: float = 82.0) -> pd.DataFrame:
    """dataset.current_rows() for this season's logs, plus everyone in name_keys
    who hasn't played yet but appears in one of the earlier seasons' logs."""
    frames = [f for f in (prev2, prev, current) if f is not None and not f.empty]
    ids = pd.concat([f[["name_key", "nba_id"]] for f in frames]).drop_duplicates("name_key", keep="last") \
        if frames else pd.DataFrame(columns=["name_key", "nba_id"])
    id_by_name = ids.set_index("name_key")["nba_id"]
    name_by_id = ids.drop_duplicates("nba_id", keep="last").set_index("nba_id")["name_key"]
    waiting = [int(id_by_name[k]) for k in dict.fromkeys(name_keys) if k in id_by_name.index]
    ages = pd.Series(ages_by_name, dtype=float)
    age_by_id = name_by_id.map(ages)
    totals = [season_totals(f) if f is not None and not f.empty else None for f in (prev, prev2)]
    rows = current_rows(current if current is not None else pd.DataFrame(columns=ids.columns), totals[0], totals[1],
                        age_by_id, (season_length(prev), season_length(prev2)), season_games, waiting)
    rows["name_key"] = rows["name_key"].fillna(rows["nba_id"].map(name_by_id))
    return rows


def with_games(pred: pd.DataFrame, team: pd.Series, remaining: dict[str, float], out: pd.Series,
               before_return: pd.Series, this_week: pd.Series, season_games: float = 82.0) -> pd.DataFrame:
    """Add M_G, projected games left, and make M_GP his share of his team's games
    left with the injury counted (x season games). Every series is aligned to
    pred's rows; team is his NBA team."""
    res = pred.copy()
    left = team.map(remaining).astype(float).set_axis(res.index)
    left = left.fillna(float(np.mean(list(remaining.values()))) if remaining else np.nan)
    games = projected_games(res[M + "avail"], left, out.fillna(False).astype(bool).set_axis(res.index),
                            before_return.set_axis(res.index), this_week.set_axis(res.index))
    res[M + "G"] = games
    res[M + "GP"] = (games / left.where(left > 0) * season_games).fillna(res[M + "GP"])
    return res
