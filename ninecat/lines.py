"""Per-game stat lines: ESPN projection, last season, a blend, or this season.

A "line" is the per-game row the valuation runs on: MIN, the counting
stats, FGM/FGA/FTM/FTA and GP. Projection columns are the ESPN ones
(PTS, REB, ...); last season's are the same names with an L_ prefix and
this season's from nba.com an S_ prefix, both joined by player name (see
attach_by_name). ESPN's own season-to-date averages carry a T_ prefix.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# Stats every line carries, per game. GP is games played (or projected).
LINE_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
LAST_PREFIX = "L_"
SEASON_PREFIX = "S_"       # this season to date, from nba.com game logs
ESPN_TOTAL_PREFIX = "T_"   # this season to date, as ESPN reports it

SOURCES = {"blend": "Blend", "proj": "Projection", "last": "Last season", "season": "This season"}
# Fewer games than this and "This season" falls back to the blend.
SEASON_MIN_GAMES = 3
# Blend = 60% projection, 40% last season, when he played enough to trust it.
BLEND_WEIGHT = 0.6
BLEND_MIN_GAMES = 15
# Counting stats (and attempts) that scale with minutes or possessions.
RATE_SCALED = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]
LG_PACE = 100.0

# nba.com / BBR names that differ from ESPN's beyond accents and suffixes.
NAME_ALIASES = {
    "herbert jones": "herb jones",
    "cameron johnson": "cam johnson",
    "nicolas claxton": "nic claxton",
    "carlton carrington": "bub carrington",
    "ronald holland": "ron holland",
    "alexandre sarr": "alex sarr",
}

_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def name_key(name: str) -> str:
    """Join key for player names across ESPN, nba.com and Basketball Reference."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace(".", "").replace("'", "").replace("’", "").strip()
    s = re.sub(r"\s+", " ", s)
    s = _SUFFIX.sub("", s)
    return NAME_ALIASES.get(s, s)


def attach_by_name(players: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join nba.com columns (keyed by name_key) onto ESPN players."""
    out = players.copy()
    if extra is None or extra.empty:
        return out
    keys = out["name"].map(name_key)
    extra = extra.drop_duplicates("name_key").set_index("name_key")
    extra = extra.drop(columns=[c for c in extra.columns if c in out.columns])
    joined = extra.reindex(keys)
    joined.index = out.index
    return pd.concat([out, joined], axis=1)


def _has_projection(df: pd.DataFrame) -> pd.Series:
    minutes = df["MIN"].fillna(0) if "MIN" in df else 0
    produces = (df["PTS"].fillna(0) > 0) | (minutes > 0)
    if "stat_source" in df:
        return df["stat_source"].notna() & produces
    return produces


def _has_last(df: pd.DataFrame) -> pd.Series:
    col = LAST_PREFIX + "G"
    if col not in df:
        return pd.Series(False, index=df.index)
    return df[col].fillna(0) > 0


def projection_line(df: pd.DataFrame) -> pd.DataFrame:
    """ESPN's projection as a line.

    ESPN projections normally include makes and attempts. When a player's
    are missing, attempts are estimated from his last-season shots per
    point and makes from his last-season percentage. Without a last season
    there's nothing to estimate from, so he stays neutral (0 attempts).
    """
    line = pd.DataFrame(index=df.index)
    for s in LINE_STATS:
        line[s] = df[s].astype(float) if s in df else np.nan
    line["GP"] = df["GP"].astype(float) if "GP" in df else np.nan

    last_pts = df.get(LAST_PREFIX + "PTS")
    if last_pts is None:
        return line
    for made, att in (("FGM", "FGA"), ("FTM", "FTA")):
        l_att, l_made = df[LAST_PREFIX + att], df[LAST_PREFIX + made]
        missing = (line[att].fillna(0) <= 0) & (line["PTS"] > 0) & (l_att > 0)
        if missing.any():
            est = line["PTS"] * l_att / last_pts.where(last_pts > 0)
            line.loc[missing, att] = est[missing]
            line.loc[missing, made] = (est * l_made / l_att)[missing]
    return line


def last_line(df: pd.DataFrame) -> pd.DataFrame:
    """Last season's per-game line; GP is the games he actually played."""
    line = pd.DataFrame(index=df.index)
    for s in LINE_STATS:
        col = LAST_PREFIX + s
        line[s] = df[col].astype(float) if col in df else np.nan
    line["GP"] = df[LAST_PREFIX + "G"].astype(float) if LAST_PREFIX + "G" in df else np.nan
    return line


def season_line(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """This season's per-game line, and who has one (3+ games).

    nba.com's numbers come first; ESPN's season-to-date averages fill in
    players the name join missed. GP is the projection's games for the
    whole season, so the games discount means the same as for the others.
    """
    line = pd.DataFrame(index=df.index)
    nba_g = df.get(SEASON_PREFIX + "G", pd.Series(np.nan, index=df.index)).fillna(0)
    espn_g = df.get(ESPN_TOTAL_PREFIX + "GP", pd.Series(np.nan, index=df.index)).fillna(0)
    use_nba = nba_g >= SEASON_MIN_GAMES
    for s in LINE_STATS:
        nba = df.get(SEASON_PREFIX + s, pd.Series(np.nan, index=df.index)).astype(float)
        espn = df.get(ESPN_TOTAL_PREFIX + s, pd.Series(np.nan, index=df.index)).astype(float)
        line[s] = nba.where(use_nba, espn)
    line["GP"] = df["GP"].astype(float) if "GP" in df else np.nan
    return line, use_nba | (espn_g >= SEASON_MIN_GAMES)


def stat_line(df: pd.DataFrame, source: str = "blend") -> pd.DataFrame:
    """The per-game line each player is valued on, as a copy of df.

    - "proj": ESPN projection, falling back to last season.
    - "last": last season only.
    - "blend": 60/40 projection/last season, with makes and attempts mixed
      separately so the percentages stay volume-weighted. Players with
      fewer than 15 games last season, or none, use the projection alone;
      players with no projection use last season. GP is the projection's.
    - "season": this season so far (see season_line); players with fewer
      than 3 games use the blend.

    Rows with no usable line get NaN stats and has_line False.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown stats source {source!r}; expected one of {list(SOURCES)}")
    proj, last = projection_line(df), last_line(df)
    has_p, has_l = _has_projection(df), _has_last(df)

    if source == "last":
        line, has = last, has_l
    elif source == "proj":
        line = proj.where(has_p, last, axis=0)
        has = has_p | has_l
    else:
        w = BLEND_WEIGHT
        mixed = w * proj + (1 - w) * last
        mixed["GP"] = proj["GP"]
        use_mix = has_p & has_l & (last["GP"] >= BLEND_MIN_GAMES)
        line = proj.where(has_p, last, axis=0)
        line = mixed.where(use_mix, line, axis=0)
        has = has_p | has_l
        if source == "season":
            season, has_s = season_line(df)
            season["GP"] = season["GP"].where(has_p, line["GP"])
            line = season.where(has_s, line, axis=0)
            has = has | has_s

    out = df.copy()
    cols = [*LINE_STATS, "GP"]
    out[cols] = line[cols].where(has, axis=0)
    out["has_line"] = has
    return out


def per36(line: pd.DataFrame) -> pd.DataFrame:
    """Every counting stat and attempt scaled to 36 minutes."""
    out = line.copy()
    k = 36 / line["MIN"].where(line["MIN"] > 0)
    for s in RATE_SCALED:
        out[s] = line[s] * k
    return out


def per100(line: pd.DataFrame, pace: dict[str, float] | None, team_col: str = "pro_team") -> pd.DataFrame:
    """Every counting stat scaled to 100 of his team's possessions on court."""
    out = line.copy()
    team_pace = line[team_col].map(pace or {}).fillna(LG_PACE).astype(float)
    poss = team_pace * line["MIN"] / 48
    k = 100 / poss.where(poss > 0)
    for s in RATE_SCALED:
        out[s] = line[s] * k
    return out
