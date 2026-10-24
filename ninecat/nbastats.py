"""NBA stats from nba.com (nba_api), plus BPM from Basketball Reference.

From nba.com: this season's box-score game logs (and the per-game line,
recent minutes and age built from them), last season's per-game lines,
the season before's minutes, team pace, the league schedule, and past
seasons' logs and totals for the projection model. Advanced rates are
computed from nba.com totals (see rates.py). BPM has no nba.com
equivalent, so it's read from Basketball Reference's advanced table.

nba.com hangs and rate-limits, so every call has a timeout, one retry
after a backoff, and a short pause after it (with_retry). nba_api is
imported lazily inside the fetch functions, so tests never need it or
the network. Everything else here is pure and tested.
"""

from __future__ import annotations

import time
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from typing import TypeVar

import pandas as pd

from ninecat.lines import LAST_PREFIX, LINE_STATS, SEASON_PREFIX, name_key
from ninecat.rates import TEAM_STATS, rate_stats
from ninecat.schedule import SCHEDULE_COLUMNS

T = TypeVar("T")

# nba.com abbreviations that ESPN spells differently.
NBA_TO_ESPN = {"PHI": "PHL", "PHX": "PHO"}
BBR_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
USER_AGENT = "nine-cat-value-board (+https://github.com/lucakraychete/nine-cat-value-board)"
TIMEOUT = 60
RETRIES = 1
BACKOFF = 5.0  # seconds before the retry
PAUSE = 0.6    # seconds after every call, to stay under nba.com's rate limit
# nba.com game ids start 001 preseason, 002 regular season, 004 playoffs.
REGULAR_SEASON = "002"
# Status prefix for sources that are expected to be empty, such as this
# season's game logs before opening night. Not shown as a failure.
NOT_YET = "not available yet"

# Advanced metrics on the board, in display order. TS is a fraction, the rest
# are percentages or ratings. Lower TOV% is better.
ADVANCED = ["usg", "ts", "ast", "trb", "stl", "blk", "tov", "per", "bpm"]

# nba.com totals -> the line names used everywhere else.
_PER_GAME = {"MIN": "MIN", "PTS": "PTS", "REB": "REB", "AST": "AST", "STL": "STL",
             "BLK": "BLK", "FG3M": "3PM", "TOV": "TO", "FGM": "FGM", "FGA": "FGA",
             "FTM": "FTM", "FTA": "FTA"}
# Game logs: one row per player per game, box score in line names.
LOG_COLUMNS = ["name_key", "nba_id", "game_id", "date", "team", "opp", "home", *LINE_STATS, "season"]
# This season to date, per player (S_ line plus recent minutes and age).
SEASON_COLUMNS = ["name_key", SEASON_PREFIX + "G", *(SEASON_PREFIX + s for s in LINE_STATS),
                  SEASON_PREFIX + "team", SEASON_PREFIX + "G_7D", SEASON_PREFIX + "MIN_7D",
                  SEASON_PREFIX + "G_15D", SEASON_PREFIX + "MIN_15D", SEASON_PREFIX + "MIN_L5", "age"]
# Season totals kept for the model history.
TOTALS_COLUMNS = ["name_key", "nba_id", "season", "team", "age", "GP", *LINE_STATS]


def season_label(espn_year: int, back: int = 0) -> str:
    """nba.com season label. ESPN's year 2027 is the 2026-27 season (back=0)."""
    start = espn_year - 1 - back
    return f"{start}-{str(start + 1)[2:]}"


def espn_team(abbr: str) -> str:
    return NBA_TO_ESPN.get(abbr, abbr)


def with_retry(fn: Callable[[], T], retries: int = RETRIES, backoff: float = BACKOFF,
               pause: float = PAUSE, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call fn, retrying after a growing backoff; pause after the attempt that ends it."""
    for attempt in range(retries + 1):
        try:
            result = fn()
        except Exception:
            if attempt == retries:
                sleep(pause)
                raise
            sleep(backoff * (attempt + 1))
        else:
            sleep(pause)
            return result
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- network

def fetch_player_totals(season: str, timeout: int = TIMEOUT) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerstats

    return with_retry(lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, per_mode_detailed="Totals", timeout=timeout
    ).get_data_frames()[0])


def fetch_team_totals(season: str) -> pd.DataFrame:
    """Team totals, opponent totals and pace, indexed by ESPN team abbreviation."""
    from nba_api.stats.endpoints import leaguedashteamstats
    from nba_api.stats.static import teams as static_teams

    def get(**kw):
        return with_retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season, timeout=TIMEOUT, **kw
        ).get_data_frames()[0]).set_index("TEAM_ID")

    base = get(per_mode_detailed="Totals")
    opp = get(per_mode_detailed="Totals", measure_type_detailed_defense="Opponent")
    adv = get(per_mode_detailed="PerGame", measure_type_detailed_defense="Advanced")
    abbr = {t["id"]: espn_team(t["abbreviation"]) for t in static_teams.get_teams()}
    return team_frame(base, opp, adv, abbr)


def fetch_game_logs(season: str, timeout: int = TIMEOUT) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs

    logs = with_retry(lambda: playergamelogs.PlayerGameLogs(
        season_nullable=season, season_type_nullable="Regular Season", timeout=timeout
    ).get_data_frames()[0])
    return game_log_frame(logs, season)


def fetch_schedule(season: str, timeout: int = TIMEOUT) -> pd.DataFrame:
    from nba_api.stats.endpoints import scheduleleaguev2

    raw = with_retry(lambda: scheduleleaguev2.ScheduleLeagueV2(
        season=season, timeout=timeout).get_data_frames()[0])
    return schedule_frame(raw)


def fetch_bbr_advanced(season_end_year: int) -> str:
    req = urllib.request.Request(
        BBR_URL.format(year=season_end_year), headers={"User-Agent": USER_AGENT}
    )

    def get() -> str:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")

    return with_retry(get)


def fetch_history_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One past regular season for the model: game logs, and totals with age."""
    return fetch_game_logs(season), totals_frame(fetch_player_totals(season), season)


def attempt(status: dict[str, str], key: str, fn: Callable[[], T]) -> T | None:
    """Run one source; record "ok" or why it failed under `key`, never raise."""
    try:
        result = fn()
    except Exception as exc:  # network, rate limits, schema changes
        status[key] = f"failed: {type(exc).__name__}: {exc}"[:300]
        return None
    status[key] = "ok"
    return result


def fetch_all(espn_year: int, static: bool = True, last_logs: bool = True,
              asof=None) -> tuple[dict[str, pd.DataFrame | dict], dict[str, str]]:
    """Everything the board needs from nba.com and BBR, plus a status per source.

    `static` refetches last season's numbers (they only change when nba.com
    corrects a stat), and `last_logs` last season's game logs. This season's
    logs are always fetched. Each source fails independently: a failure is
    reported in the status and that piece is left out, so the caller keeps
    its old cache.
    """
    last, prev, current = season_label(espn_year, 1), season_label(espn_year, 2), season_label(espn_year)
    out: dict[str, pd.DataFrame | dict] = {}
    status: dict[str, str] = {}

    if static:
        teams = attempt(status, "nba_teams", lambda: fetch_team_totals(last))
        totals = attempt(status, "nba_players", lambda: fetch_player_totals(last))
        prev_totals = attempt(status, "nba_prev_season", lambda: fetch_player_totals(prev))
        html = attempt(status, "bbr_bpm", lambda: fetch_bbr_advanced(espn_year - 1))
        bpm = parse_bbr_advanced(html) if html else None
        if teams is not None and totals is not None:
            out["last_season"] = last_season_frame(totals, teams, prev_totals, bpm)
            out["pace"] = {k: round(float(v), 2) for k, v in teams["PACE"].items()}
    if last_logs:
        logs = attempt(status, "nba_last_logs", lambda: fetch_game_logs(last))
        if logs is not None and not logs.empty:
            out["last_logs"] = logs

    logs = attempt(status, "nba_season", lambda: fetch_game_logs(current))
    ages = attempt(status, "nba_ages", lambda: player_ages(fetch_player_totals(current)))
    if logs is not None and logs.empty:
        status["nba_season"] = f"{NOT_YET} (no games yet)"
    elif logs is not None:
        out["season_logs"] = logs
        out["season"] = season_to_date(logs, ages, asof)
    return out, status


# ---------------------------------------------------------------- pure

def team_frame(base: pd.DataFrame, opp: pd.DataFrame, adv: pd.DataFrame,
               abbr: dict[int, str]) -> pd.DataFrame:
    """Join own totals, OPP_* totals and PACE; index by ESPN abbreviation."""
    t = base[TEAM_STATS].copy()
    opp_cols = [c for c in opp.columns if c.startswith("OPP_")]
    t = t.join(opp[opp_cols]).join(adv[["PACE"]])
    t.index = [abbr.get(i, str(i)) for i in t.index]
    return t


def last_season_frame(totals: pd.DataFrame, teams: pd.DataFrame,
                      prev_totals: pd.DataFrame | None = None,
                      bpm: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player: per-game L_* line, advanced rates, prior-season minutes.

    Keyed by name_key for joining onto ESPN players.
    """
    p = totals.copy()
    p["team"] = p["TEAM_ABBREVIATION"].map(espn_team)
    gp = p["GP"].where(p["GP"] > 0)

    out = pd.DataFrame({"name_key": p["PLAYER_NAME"].map(name_key), "nba_name": p["PLAYER_NAME"]})
    out[LAST_PREFIX + "G"] = p["GP"].astype(float)
    for src, dst in _PER_GAME.items():
        out[LAST_PREFIX + dst] = p[src] / gp

    rates = rate_stats(p, teams)
    for k in rates.columns:
        out["adv_" + k] = rates[k]
    out["adv_bpm"] = pd.NA
    if bpm is not None and not bpm.empty:
        out["adv_bpm"] = out["name_key"].map(bpm.drop_duplicates("name_key").set_index("name_key")["bpm"])
    out["adv_bpm"] = pd.to_numeric(out["adv_bpm"], errors="coerce")

    if prev_totals is not None and not prev_totals.empty:
        prev = prev_totals.assign(name_key=prev_totals["PLAYER_NAME"].map(name_key))
        prev_min = (prev["MIN"] / prev["GP"].where(prev["GP"] > 0)).set_axis(prev["name_key"])
        out["prev_MIN"] = out["name_key"].map(prev_min[~prev_min.index.duplicated()])
    # Same name twice (rare): keep the one who played more.
    out = out.sort_values(LAST_PREFIX + "MIN", ascending=False, na_position="last")
    return out.drop_duplicates("name_key").reset_index(drop=True)


def _regular_season(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop preseason and playoff rows, which nba.com sometimes returns anyway."""
    ids = df[col].astype(str).str.zfill(10)
    return df[ids.str.startswith(REGULAR_SEASON)]


def game_log_frame(logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """Full box score per player per game, regular season only.

    MATCHUP reads "LAL vs. GSW" at home and "LAL @ GSW" away.
    """
    if logs.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)
    logs = _regular_season(logs, "GAME_ID")
    matchup = logs["MATCHUP"].astype(str)
    out = pd.DataFrame({
        "name_key": logs["PLAYER_NAME"].map(name_key),
        "nba_id": logs["PLAYER_ID"],
        "game_id": logs["GAME_ID"].astype(str).str.zfill(10),
        "date": pd.to_datetime(logs["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "team": logs["TEAM_ABBREVIATION"].map(espn_team),
        "opp": matchup.str.split(r" vs\. | @ ", regex=True).str[-1].str.strip().map(espn_team),
        "home": matchup.str.contains(" vs. ", regex=False),
    })
    for src, dst in _PER_GAME.items():
        out[dst] = pd.to_numeric(logs[src], errors="coerce").astype(float)
    out["MIN"] = out["MIN"].round(2)
    out["season"] = season
    return out[LOG_COLUMNS].reset_index(drop=True)


def player_ages(totals: pd.DataFrame) -> pd.Series:
    """Age by name_key from a LeagueDashPlayerStats frame."""
    ages = pd.Series(totals["AGE"].astype(float).values, index=totals["PLAYER_NAME"].map(name_key))
    return ages[~ages.index.duplicated()]


def season_to_date(logs: pd.DataFrame, ages: pd.Series | None = None, asof=None) -> pd.DataFrame:
    """This season per player from his game logs.

    The S_ per-game line and games played, his latest team, average minutes
    and games over the last 7 and 15 days counting `asof` (default: the
    latest game day), average minutes over his last 5 games, and age.
    """
    if logs.empty:
        return pd.DataFrame(columns=SEASON_COLUMNS)
    g = logs.sort_values("date", kind="stable")
    dates = pd.to_datetime(g["date"])
    asof = pd.Timestamp(asof).normalize() if asof is not None else dates.max()
    by = g.groupby("name_key", sort=False)
    out = by[LINE_STATS].mean().add_prefix(SEASON_PREFIX)
    out[SEASON_PREFIX + "G"] = by.size()
    out[SEASON_PREFIX + "team"] = by["team"].last()
    for days in (7, 15):
        recent = g[dates > asof - pd.Timedelta(days=days)].groupby("name_key")["MIN"]
        out[f"{SEASON_PREFIX}G_{days}D"] = recent.size().reindex(out.index, fill_value=0)
        out[f"{SEASON_PREFIX}MIN_{days}D"] = recent.mean()
    out[SEASON_PREFIX + "MIN_L5"] = by["MIN"].apply(lambda m: m.tail(5).mean())
    out["age"] = out.index.map(ages) if ages is not None else float("nan")
    return out.rename_axis("name_key").reset_index()[SEASON_COLUMNS]


def totals_frame(totals: pd.DataFrame, season: str) -> pd.DataFrame:
    """A LeagueDashPlayerStats Totals frame in line names, for the model history."""
    out = pd.DataFrame({
        "name_key": totals["PLAYER_NAME"].map(name_key),
        "nba_id": totals["PLAYER_ID"],
        "season": season,
        "team": totals["TEAM_ABBREVIATION"].map(espn_team),
        "age": pd.to_numeric(totals["AGE"], errors="coerce"),
        "GP": totals["GP"],
    })
    for src, dst in _PER_GAME.items():
        out[dst] = pd.to_numeric(totals[src], errors="coerce")
    return out[TOTALS_COLUMNS]


def schedule_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """ScheduleLeagueV2 -> regular-season games with both teams known.

    NBA Cup games still to be decided have no teams yet and are left out.
    gameDateEst is the US Eastern day, the one ESPN's scoring periods use.
    """
    raw = _regular_season(raw, "gameId")
    home, away = raw["homeTeam_teamTricode"].fillna(""), raw["awayTeam_teamTricode"].fillna("")
    raw = raw[(home != "") & (away != "")]
    day = raw["gameDateEst"] if "gameDateEst" in raw else raw["gameDate"]
    out = pd.DataFrame({
        "date": pd.to_datetime(day, format="mixed").dt.strftime("%Y-%m-%d"),
        "game_id": raw["gameId"].astype(str).str.zfill(10),
        "home": raw["homeTeam_teamTricode"].map(espn_team),
        "away": raw["awayTeam_teamTricode"].map(espn_team),
    })
    out = out.drop_duplicates("game_id").sort_values(["date", "game_id"])
    return out.reset_index(drop=True)[SCHEDULE_COLUMNS]


class _AdvancedTable(HTMLParser):
    """Collect data-stat cells from BBR's <table id="advanced">."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict[str, str]] = []
        self._in_table = self._in_thead = self._skip_row = False
        self._row: dict[str, str] | None = None
        self._stat: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table" and a.get("id") == "advanced":
            self._in_table = True
        elif self._in_table and tag == "thead":
            self._in_thead = True
        elif self._in_table and tag == "tr":
            self._row = {}
            # header rows: the real <thead>, and the ones BBR repeats every 20 rows
            self._skip_row = self._in_thead or "thead" in (a.get("class") or "")
        elif self._row is not None and tag in ("td", "th") and a.get("data-stat"):
            self._stat = a["data-stat"]
            self._row[self._stat] = ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._stat = None
        elif tag == "thead":
            self._in_thead = False
        elif tag == "tr" and self._row is not None:
            if not self._skip_row and self._row.get("name_display"):
                self.rows.append(self._row)
            self._row = None
        elif tag == "table" and self._in_table:
            self._in_table = False

    def handle_data(self, data):
        if self._row is not None and self._stat:
            self._row[self._stat] += data


def parse_bbr_advanced(html: str) -> pd.DataFrame:
    """name_key, team and bpm from BBR's advanced page.

    Traded players have a combined row (2TM, 3TM) listed before the
    per-team rows; that first row is the season total and the one kept.
    """
    parser = _AdvancedTable()
    parser.feed(html)
    rows = [r for r in parser.rows if r.get("name_display", "").strip() != "League Average"]
    if not rows:
        return pd.DataFrame(columns=["name_key", "team", "bpm"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "name_key": df["name_display"].str.strip().map(name_key),
        "team": df.get("team_name_abbr", pd.Series("", index=df.index)).str.strip(),
        "bpm": pd.to_numeric(df.get("bpm"), errors="coerce"),
    })
    return out.drop_duplicates("name_key").reset_index(drop=True)
