"""Numbers behind a player's page: where he ranks, overall and at his position.

Everything takes the board from build_board() (sorted by rank) and the
board index of the player. Percentiles follow one rule throughout: the
share of the comparison group strictly below him, 0-100.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ninecat.board import ADVANCED_METRICS, RATE_MIN_MINUTES
from ninecat.lines import LAST_PREFIX, last_line, per36, per100, projection_line
from ninecat.valuation import z_col

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
PROFILE_TOP = 150      # category percentiles are against the top 150
EFFICIENCY_TOP = 200   # efficiency ranks are against the top 200
ROLLING_GAMES = 60


def positions(row: pd.Series) -> list[str]:
    elig = row.get("eligible")
    if isinstance(elig, str) and elig:
        return elig.split("/")
    pos = row.get("position")
    return [pos] if isinstance(pos, str) and pos else []


def primary_position(row: pd.Series) -> str:
    pos = positions(row)
    return pos[0] if pos else ""


def position_group(board: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Everyone eligible at `pos`, in board (value) order."""
    elig = board["eligible"].fillna("").str.split("/")
    return board[elig.apply(lambda s: pos in s)]


def percentile(x: float, values: Iterable[float]) -> float:
    vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(vals) == 0 or pd.isna(x):
        return 50.0
    return 100.0 * (vals < x).sum() / len(vals)


def category_profile(board: pd.DataFrame, idx, cats: list[str]) -> pd.DataFrame:
    """Per category: his percentile among the top 150, the average top-150
    player at his position's percentile, and his rank within the position."""
    row = board.loc[idx]
    pos = primary_position(row)
    group = position_group(board, pos)
    top = board[board["rank"] <= PROFILE_TOP]
    group_top = group[group["rank"] <= PROFILE_TOP]
    out = []
    for c in cats:
        ref = top[z_col(c)]
        pct = percentile(row[z_col(c)], ref)
        avg = (np.mean([percentile(z, ref) for z in group_top[z_col(c)]])
               if len(group_top) else 50.0)
        order = group[z_col(c)].sort_values(ascending=False, kind="stable")
        out.append({"cat": c, "pct": pct, "pos_avg_pct": avg,
                    "pos_rank": order.index.get_loc(idx) + 1 if idx in order.index else None})
    return pd.DataFrame(out)


@dataclass(frozen=True)
class RankItem:
    label: str
    sub: str
    value: str
    pct_all: float | None
    pct_pos: float | None
    rank: str


def _rank_in(group: pd.DataFrame, idx, key: pd.Series, lower_is_better: bool) -> int | None:
    vals = key.reindex(group.index).dropna()
    if idx not in vals.index:
        return None
    order = vals.sort_values(ascending=lower_is_better, kind="stable")
    return order.index.get_loc(idx) + 1


def _pct_in(group: pd.DataFrame, idx, key: pd.Series, lower_is_better: bool) -> float | None:
    x = key.get(idx)
    if x is None or pd.isna(x):
        return None
    p = percentile(x, key.reindex(group.index))
    return 100 - p if lower_is_better else p


def _items(board, idx, specs) -> list[RankItem]:
    row = board.loc[idx]
    pos = primary_position(row)
    pool, group = board[board["rank"] <= EFFICIENCY_TOP], position_group(board, pos)
    items = []
    for label, sub, key, fmt, inv in specs:
        x = key.get(idx)
        rk, prk = _rank_in(pool, idx, key, inv), _rank_in(group, idx, key, inv)
        items.append(RankItem(
            label=label, sub=sub,
            value="–" if x is None or pd.isna(x) else fmt(x),
            pct_all=_pct_in(pool, idx, key, inv), pct_pos=_pct_in(group, idx, key, inv),
            rank=f"#{rk} · {pos} #{prk}" if rk else "",
        ))
    return items


def _col(board: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(board[name], errors="coerce") if name in board else pd.Series(np.nan, index=board.index)


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def pct3(x: float) -> str:
    """.563 style, as box scores print shooting percentages."""
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def efficiency_items(board: pd.DataFrame, idx) -> list[RankItem]:
    """Value and efficiency, percentile overall (top 200) and at his position."""
    one = lambda x: f"{x:.1f}"
    specs: list[tuple[str, str, pd.Series, Callable, bool]] = [
        ("Fantasy value / game", "per-game z-score total", board["value"], signed, False),
        ("Fantasy value / 36 min", "same, every player at 36 min", _col(board, "p36_value"), signed, False),
        ("Fantasy value / 100 poss", "adjusts for team pace", _col(board, "p100_value"), signed, False),
        ("True shooting", "scoring efficiency", _col(board, "adv_ts"), pct3, False),
        ("Usage rate", "share of team plays", _col(board, "adv_usg"), lambda x: f"{x:.1f}%", False),
        ("PER", "per-minute production", _col(board, "adv_per"), one, False),
        ("Box plus/minus", "pts/100 vs average", _col(board, "adv_bpm"), one, False),
        ("Turnover rate", "lower is better", _col(board, "adv_tov"), lambda x: f"{x:.1f}%", True),
    ]
    return _items(board, idx, specs)


RATE_SUBS = {"ast": "teammate baskets assisted", "trb": "available rebounds grabbed",
             "stl": "opponent plays ending in a steal", "blk": "opponent twos blocked"}


def rate_items(board: pd.DataFrame, idx) -> list[RankItem]:
    """AST%, TRB%, STL%, BLK%: his share of the plays available to him."""
    specs = [(m.label, RATE_SUBS[m.key], _col(board, f"adv_{m.key}"), lambda x: f"{x:.1f}%", False)
             for m in ADVANCED_METRICS if m.key in RATE_SUBS]
    return _items(board, idx, specs)


def comparables(board: pd.DataFrame, idx, n: int = 6) -> pd.DataFrame:
    """The n players at his position closest in value, plus him, by value."""
    row = board.loc[idx]
    group = position_group(board, primary_position(row)).drop(index=idx, errors="ignore")
    near = (group["value"] - row["value"]).abs().sort_values(kind="stable").index[:n]
    return board.loc[[idx, *near]].sort_values("value", ascending=False)


def minutes_history(row: pd.Series, seasons: dict[str, str]) -> list[tuple[str, float, bool]]:
    """(season label, minutes per game, is projection) for the minutes chart."""
    pts = []
    if pd.notna(row.get("prev_MIN")):
        pts.append((seasons["prev"], float(row["prev_MIN"]), False))
    if pd.notna(row.get(LAST_PREFIX + "MIN")) and row.get(LAST_PREFIX + "G", 0) > 0:
        pts.append((seasons["last"], float(row[LAST_PREFIX + "MIN"]), False))
    if pd.notna(row.get("proj_MIN")) and row["proj_MIN"] > 0:
        pts.append((f"{seasons['current']} proj", float(row["proj_MIN"]), True))
    return pts


def minutes_trend(d_min: float | None) -> str:
    if d_min is None or pd.isna(d_min):
        return "No prior-season minutes to compare."
    if d_min >= 2:
        return f"Projected for {d_min:.1f} more minutes a game than last season, a bigger role."
    if d_min <= -2:
        return f"Projected for {-d_min:.1f} fewer minutes a game than last season, a smaller role."
    return "Minutes projected to stay about the same as last season."


def rolling_minutes(logs: pd.DataFrame, games: int = ROLLING_GAMES) -> pd.DataFrame:
    """His last `games` games with 5- and 10-game rolling average minutes."""
    g = logs.sort_values("date").tail(games).reset_index(drop=True)
    g["r5"] = g["min"].rolling(5, min_periods=1).mean()
    g["r10"] = g["min"].rolling(10, min_periods=1).mean()
    return g


def stat_lines(raw: pd.DataFrame, board_row: pd.Series, pace: dict[str, float] | None,
               labels: dict[str, str]) -> pd.DataFrame:
    """Projection, last season, the board's line, per 36 and per 100, per game.

    `raw` is the player's single-row slice of the unvalued players frame.
    """
    cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "FG%", "FT%", "TO", "FGA", "FTA"]

    def pcts(line: pd.DataFrame) -> pd.DataFrame:
        line = line.copy()
        line["FG%"] = line["FGM"] / line["FGA"].where(line["FGA"] > 0)
        line["FT%"] = line["FTM"] / line["FTA"].where(line["FTA"] > 0)
        return line

    board_line = board_row.to_frame().T
    for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]:
        board_line[c] = pd.to_numeric(board_line[c], errors="coerce")
    board_line["pro_team"] = board_row.get("pro_team")

    rows = []
    has_proj = "stat_source" in raw and raw["stat_source"].notna().iloc[0]
    if has_proj:
        rows.append((labels["proj"], projection_line(raw)))
    if raw.get(LAST_PREFIX + "G", pd.Series([0])).fillna(0).iloc[0] > 0:
        rows.append((labels["last"], last_line(raw)))
    rows.append((labels["board"], board_line))
    if board_line["MIN"].iloc[0] >= RATE_MIN_MINUTES:
        rows.append(("Per 36 minutes", per36(board_line)))
        rows.append((labels["per100"], per100(board_line, pace)))
    out = pd.concat([pcts(line)[cols].assign(Line=label) for label, line in rows], ignore_index=True)
    return out[["Line", *cols]]
