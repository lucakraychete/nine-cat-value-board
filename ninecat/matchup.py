"""Head-to-head matchup model: one team's week, and the odds of winning it.

A team's week is every rostered player's games, scaled by availability and
by how often his lineup slot gets used:

- Counting stats: mean = sum over players of games x per-game average;
  variance = games x (mean + (0.3 x mean)^2), i.e. Poisson-like game-to-game
  noise plus a 30% swing in how well he plays.
- Percentages: makes and attempts summed over the week. Variance is the
  binomial p(1 - p) / attempts, inflated 1.6x for shooting streakiness.

Each category is won with probability Phi(edge / sd), with sd from both
teams' variances. A matchup is won by taking a majority of categories (a
tied split counts as half), so the number of categories won is treated as
normal with the per-category win chances' mean, widened by assumed
correlations between categories that move together (bigs drive REB, BLK
and FG%; shooters drive PTS, 3PM and FT%), plus a 1/12 continuity term.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ninecat.config import NEGATIVE_CATS, PERCENT_CATS

GAMES_PER_WEEK = 3.5
WEEKS = 20
GAME_SWING = 0.3       # sd of a player's game-to-game form, as a share of his average
SHOT_STREAK = 1.6      # inflation of binomial shooting variance
# Daily lineups use nearly every rostered game: the top 10 by value count
# fully, the next three at 75%, anyone beyond at 40%.
SLOT_WEIGHTS = (10, 1.0), (3, 0.75)
BENCH_WEIGHT = 0.4

# Assumed correlations between weekly category results, from how stats
# cluster by player type. Pairs not listed are independent.
CORR = {("REB", "BLK"): 0.35, ("REB", "FG%"): 0.3, ("BLK", "FG%"): 0.3, ("PTS", "3PM"): 0.3,
        ("PTS", "FT%"): 0.15, ("3PM", "FT%"): 0.2, ("PTS", "AST"): 0.2, ("AST", "STL"): 0.15,
        ("PTS", "REB"): 0.1}


def ncdf(x):
    """Standard normal CDF, Abramowitz-Stegun 26.2.17 with the original's
    7-digit constants (|error| < 1e-6)."""
    x = np.asarray(x, dtype=float)
    t = 1 / (1 + 0.2316419 * np.abs(x))
    d = 0.3989423 * np.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return np.where(x > 0, 1 - p, p)


def corr_matrix(cats: Sequence[str]) -> np.ndarray:
    """Symmetric category correlation matrix with a zero diagonal."""
    k = len(cats)
    r = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                r[i, j] = CORR.get((cats[i], cats[j]), CORR.get((cats[j], cats[i]), 0.0))
    return r


def slot_weights(n: int) -> np.ndarray:
    w = np.full(n, BENCH_WEIGHT)
    start = 0
    for count, weight in SLOT_WEIGHTS:
        w[start:start + count] = weight
        start += count
    return w


@dataclass(frozen=True)
class Week:
    """Expected weekly result and its variance, one entry per category."""
    m: np.ndarray
    v: np.ndarray


class StatMatrix:
    """Per-player arrays the week model needs, built once from a valued board.

    Rows follow the board's order. Availability is GP / season games capped
    at 1, with a missing GP taken as 70, as in the valuation.
    """

    def __init__(self, board, cats: Sequence[str], season_games: int = 82, default_gp: float = 70):
        self.cats = list(cats)
        self.value = board["value"].to_numpy(float)
        gp = board["GP"].to_numpy(float)
        gp = np.where(np.isfinite(gp) & (gp > 0), gp, default_gp)
        self.avail = np.minimum(1.0, gp / season_games)
        self.sign = np.array([-1.0 if c in NEGATIVE_CATS else 1.0 for c in self.cats])
        self.count_ix = [i for i, c in enumerate(self.cats) if c not in PERCENT_CATS]
        self.pct_ix = [i for i, c in enumerate(self.cats) if c in PERCENT_CATS]
        count = board[[self.cats[i] for i in self.count_ix]].to_numpy(float)
        self.count = np.nan_to_num(count)
        self.count_var = self.count + (GAME_SWING * self.count) ** 2
        self.made = np.nan_to_num(board[[PERCENT_CATS[self.cats[i]][0] for i in self.pct_ix]].to_numpy(float))
        self.att = np.nan_to_num(board[[PERCENT_CATS[self.cats[i]][1] for i in self.pct_ix]].to_numpy(float))
        self.corr = corr_matrix(self.cats)

    def week(self, idx: Sequence[int], games: np.ndarray | float = GAMES_PER_WEEK) -> Week:
        """A team's week from the players at board positions `idx`.

        `games` is each player's games in the period (an array over all
        players, or one number for everyone); availability applies on top.
        """
        idx = np.asarray(idx, dtype=int)
        k = len(self.cats)
        m, v = np.zeros(k), np.zeros(k)
        if len(idx) == 0:
            m[self.pct_ix] = 0.0
            v[self.pct_ix] = 1.0
            return Week(m, v)
        idx = idx[np.argsort(-self.value[idx], kind="stable")]
        base = games[idx] if isinstance(games, np.ndarray) else np.full(len(idx), float(games))
        g = base * self.avail[idx] * slot_weights(len(idx))
        m[self.count_ix] = g @ self.count[idx]
        v[self.count_ix] = g @ self.count_var[idx]
        att, made = g @ self.att[idx], g @ self.made[idx]
        p = np.divide(made, att, out=np.zeros_like(att), where=att > 0)
        m[self.pct_ix] = p
        v[self.pct_ix] = np.where(att > 0, SHOT_STREAK * p * (1 - p) / np.where(att > 0, att, 1), 1.0)
        return Week(m, v)


def category_odds(a: Week, bm: np.ndarray, bv: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Chance team a wins each category against each opponent row of (bm, bv)."""
    sd = np.sqrt(a.v + bv)
    sd = np.where(sd > 0, sd, 1.0)
    return ncdf(sign * (a.m - bm) / sd)


def win_probability(p: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Chance of winning a majority of categories, from per-category odds (last axis)."""
    k = p.shape[-1]
    mu = p.sum(-1)
    s = np.sqrt(p * (1 - p))
    var = (p * (1 - p)).sum(-1) + np.einsum("...i,ij,...j->...", s, corr, s)
    return 1 - ncdf((k / 2 - mu) / np.sqrt(np.maximum(var, 1e-6) + 1 / 12))


@dataclass(frozen=True)
class Outlook:
    """A team's expected weekly results against each opponent."""
    opponents: list[str]
    p: np.ndarray        # (opponents, cats): chance to win each category
    cats_won: np.ndarray  # (opponents,): expected categories won
    win: np.ndarray       # (opponents,): chance to win the matchup

    @property
    def by_cat(self) -> np.ndarray:
        return self.p.mean(0)

    @property
    def avg_cats(self) -> float:
        return float(self.cats_won.mean())

    @property
    def avg_win(self) -> float:
        return float(self.win.mean())


def matchup(a: Week, b: Week, stats: StatMatrix) -> tuple[np.ndarray, float, float]:
    """(category odds, expected categories won, matchup win chance) for a vs b."""
    p = category_odds(a, b.m, b.v, stats.sign)
    return p, float(p.sum()), float(win_probability(p, stats.corr))


def outlook(team: str, weeks: dict[str, Week], stats: StatMatrix) -> Outlook:
    """Team's odds against every other team, from precomputed team-weeks."""
    others = [t for t in weeks if t != team]
    if not others:
        k = len(stats.cats)
        return Outlook([], np.zeros((0, k)), np.zeros(0), np.zeros(0))
    bm = np.stack([weeks[t].m for t in others])
    bv = np.stack([weeks[t].v for t in others])
    p = category_odds(weeks[team], bm, bv, stats.sign)
    return Outlook(others, p, p.sum(1), win_probability(p, stats.corr))


@dataclass(frozen=True)
class Change:
    """Difference between two outlooks: category wins over `weeks` weeks, and
    the change in average weekly matchup win chance."""
    by_cat: np.ndarray
    cats: float
    win: float


def change(before: Outlook, after: Outlook, weeks: float = WEEKS) -> Change:
    return Change((after.by_cat - before.by_cat) * weeks, (after.avg_cats - before.avg_cats) * weeks,
                  after.avg_win - before.avg_win)
