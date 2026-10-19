"""Advanced rate stats from box-score season totals.

Basketball Reference publishes USG%, TS%, AST%, TRB%, STL%, BLK%, TOV% and
PER, but nba.com (via nba_api) only serves the raw totals. These are BBR's
glossary formulas applied to those totals, so the numbers mean the same
thing as on BBR without scraping it.

Inputs are season totals. `players` has one row per player with MIN, FGM,
FGA, FG3M, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS and a
`team` key. `teams` is indexed by that key with the team's own totals
(MIN is game minutes, i.e. BBR's Tm MP / 5), the opponents' totals as
OPP_* columns, and PACE. A traded player is measured against the team he
finished with, the one nba.com reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
              "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def _div(a, b):
    """Elementwise a / b with NaN instead of inf where b is zero."""
    b = pd.Series(b, dtype=float) if not isinstance(b, pd.Series) else b.astype(float)
    return a / b.where(b != 0)


def possessions(t: pd.DataFrame) -> pd.Series:
    """BBR team possession estimate, averaged over both sides of the ball."""
    def side(fga, fta, orb, opp_drb, fgm, tov):
        return fga + 0.4 * fta - 1.07 * (orb / (orb + opp_drb)) * (fga - fgm) + tov

    own = side(t["FGA"], t["FTA"], t["OREB"], t["OPP_DREB"], t["FGM"], t["TOV"])
    opp = side(t["OPP_FGA"], t["OPP_FTA"], t["OPP_OREB"], t["DREB"], t["OPP_FGM"], t["OPP_TOV"])
    return 0.5 * (own + opp)


def rate_stats(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """usg, ts, ast, trb, stl, blk, tov (percent, ts as a fraction) and per.

    Returns a frame aligned with `players`. Players with no minutes get NaN.
    """
    p = players
    t = teams.reindex(p["team"]).set_axis(p.index)
    mp = p["MIN"].where(p["MIN"] > 0)
    team_mp = t["MIN"]  # BBR's Tm MP / 5
    opp_poss = possessions(t)

    out = pd.DataFrame(index=p.index)
    plays = p["FGA"] + 0.44 * p["FTA"] + p["TOV"]
    out["usg"] = 100 * _div(plays * team_mp, mp * (t["FGA"] + 0.44 * t["FTA"] + t["TOV"]))
    out["ts"] = _div(p["PTS"], 2 * (p["FGA"] + 0.44 * p["FTA"]))
    out["ast"] = 100 * _div(p["AST"], (mp / team_mp) * t["FGM"] - p["FGM"])
    out["trb"] = 100 * _div(p["REB"] * team_mp, mp * (t["REB"] + t["OPP_REB"]))
    out["stl"] = 100 * _div(p["STL"] * team_mp, mp * opp_poss)
    out["blk"] = 100 * _div(p["BLK"] * team_mp, mp * (t["OPP_FGA"] - t["OPP_FG3A"]))
    out["tov"] = 100 * _div(p["TOV"], plays)
    out["per"] = per(p, teams)
    return out.where(mp.notna())


def per(players: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    """Hollinger's PER: pace-adjusted per-minute production, league average 15."""
    p = players
    t = teams.reindex(p["team"]).set_axis(p.index)
    lg = teams[TEAM_STATS].sum()
    lg_pace = teams["PACE"].mean()

    factor = (2 / 3) - (0.5 * (lg["AST"] / lg["FGM"])) / (2 * (lg["FGM"] / lg["FTM"]))
    vop = lg["PTS"] / (lg["FGA"] - lg["OREB"] + lg["TOV"] + 0.44 * lg["FTA"])
    drbp = (lg["REB"] - lg["OREB"]) / lg["REB"]
    tm_ast_fg = t["AST"] / t["FGM"]

    mp = p["MIN"].where(p["MIN"] > 0)
    uper = (1 / mp) * (
        p["FG3M"]
        + (2 / 3) * p["AST"]
        + (2 - factor * tm_ast_fg) * p["FGM"]
        + p["FTM"] * 0.5 * (1 + (1 - tm_ast_fg) + (2 / 3) * tm_ast_fg)
        - vop * p["TOV"]
        - vop * drbp * (p["FGA"] - p["FGM"])
        - vop * 0.44 * (0.44 + 0.56 * drbp) * (p["FTA"] - p["FTM"])
        + vop * (1 - drbp) * (p["REB"] - p["OREB"])
        + vop * drbp * p["OREB"]
        + vop * p["STL"]
        + vop * drbp * p["BLK"]
        - p["PF"] * ((lg["FTM"] / lg["PF"]) - 0.44 * (lg["FTA"] / lg["PF"]) * vop)
    )
    aper = (lg_pace / t["PACE"]) * uper
    valid = aper.notna() & mp.notna()
    lg_aper = np.average(aper[valid], weights=mp[valid]) if valid.any() else np.nan
    return aper * (15 / lg_aper)
