"""NBA schedule and ESPN matchup-period calendar: games per team between dates.

A schedule frame has one row per regular-season game: date (ISO, US
Eastern), game_id, home, away (ESPN team abbreviations).

ESPN lists a matchup period's days (scoring periods) only once it has
started. For later periods its settings say which schedule weeks each
period spans, so the calendar is rebuilt from weeks: Monday to Sunday,
starting with opening night's week. ESPN folds the All-Star break into
one longer week, so when there is one calendar week more than ESPN has
weeks, the two calendar weeks the All-Star break straddles (the longest
February gap in the NBA schedule) count as one. Periods ESPN has mapped are checked against
the rebuilt calendar, and ESPN's own dates win (see apply_known_periods).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta

import pandas as pd

SCHEDULE_COLUMNS = ["date", "game_id", "home", "away"]
CALENDAR_COLUMNS = ["period", "start", "end", "source"]


def _day(d) -> date:
    return pd.Timestamp(d).date()


def scoring_period_date(opening: date, scoring_period: int) -> date:
    """ESPN scoring periods are days, numbered from 1 on opening night."""
    return opening + timedelta(days=int(scoring_period) - 1)


def team_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one for each side: date, team, opp, home."""
    home = pd.DataFrame({"date": schedule["date"], "team": schedule["home"],
                         "opp": schedule["away"], "home": True})
    away = pd.DataFrame({"date": schedule["date"], "team": schedule["away"],
                         "opp": schedule["home"], "home": False})
    return pd.concat([home, away], ignore_index=True)


def _teams(schedule: pd.DataFrame) -> list[str]:
    return sorted(set(schedule["home"]) | set(schedule["away"]))


def games_between(schedule: pd.DataFrame, start, end, teams: Iterable[str] | None = None) -> pd.Series:
    """Games per NBA team from start to end, both days included. Every team appears, 0 if idle."""
    g = team_games(schedule)
    d = pd.to_datetime(g["date"]).dt.date
    counts = g[(d >= _day(start)) & (d <= _day(end))].groupby("team").size()
    return counts.reindex(list(teams) if teams is not None else _teams(schedule), fill_value=0).astype(int)


def remaining_games(schedule: pd.DataFrame, today, teams: Iterable[str] | None = None) -> pd.Series:
    """Games per team from today on, today's included (they haven't been played yet)."""
    last = pd.to_datetime(schedule["date"]).max() if len(schedule) else pd.Timestamp(today)
    return games_between(schedule, today, max(_day(last), _day(today)), teams)


def games_per_period(schedule: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Games per team (columns) in each matchup period (rows)."""
    teams = _teams(schedule)
    rows = {int(p.period): games_between(schedule, p.start, p.end, teams) for p in calendar.itertuples()}
    return pd.DataFrame(rows).T.rename_axis("period")


def calendar_weeks(opening: date, final: date) -> pd.DataFrame:
    """Monday-to-Sunday weeks covering opening to final, clipped to those days."""
    monday = opening - timedelta(days=opening.weekday())
    rows = []
    while monday <= final:
        rows.append({"start": max(monday, opening), "end": min(monday + timedelta(days=6), final)})
        monday += timedelta(days=7)
    return pd.DataFrame(rows)


def all_star_break(schedule: pd.DataFrame) -> tuple[date, date] | None:
    """First and last day of the longest run of game-less days starting in February.

    Only February counts: early in the season the NBA Cup knockout games
    have no teams yet, which leaves a longer hole in December.
    """
    days = sorted(set(pd.to_datetime(schedule["date"]).dt.date))
    best, gap = 1, None
    for a, b in zip(days, days[1:]):
        first = a + timedelta(days=1)
        if first.month == 2 and (b - a).days > best:
            best, gap = (b - a).days, (first, b - timedelta(days=1))
    return gap


def _merge_weeks(weeks: pd.DataFrame, i: int) -> pd.DataFrame:
    """Calendar weeks i and i+1 as one."""
    merged = {"start": weeks.loc[i, "start"], "end": weeks.loc[i + 1, "end"]}
    rest = weeks.drop(index=[i, i + 1])
    return pd.concat([rest, pd.DataFrame([merged], index=[i])]).sort_index().reset_index(drop=True)


def espn_weeks(opening: date, final: date, n_weeks: int,
               schedule: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    """ESPN's schedule weeks as calendar spans, and a note on how they were matched.

    With one calendar week too many, the two weeks the All-Star break
    straddles are merged. Any other mismatch is reported and the last
    week stretches to the final day, so every day still belongs to a week.
    """
    weeks = calendar_weeks(opening, final)
    extra = len(weeks) - n_weeks
    note = "Monday-Sunday weeks"
    if extra == 1 and schedule is not None and len(schedule):
        gap = all_star_break(schedule)
        if gap:
            starts = [i for i, w in weeks.iterrows() if w["start"] <= gap[0] <= w["end"]]
            ends = [i for i, w in weeks.iterrows() if w["start"] <= gap[1] <= w["end"]]
            if starts and ends and ends[0] == starts[0] + 1:
                weeks = _merge_weeks(weeks, starts[0])
                note = (f"Monday-Sunday weeks, with the All-Star break ({gap[0]:%b %d}-{gap[1]:%b %d}) "
                        "folded into one two-week period")
    if len(weeks) != n_weeks:
        note = (f"unverified: {len(calendar_weeks(opening, final))} calendar weeks for ESPN's {n_weeks}; "
                "the last period runs to the final day")
        weeks = weeks.iloc[:n_weeks].copy()
        weeks.iloc[-1, weeks.columns.get_loc("end")] = final
    weeks.index = range(1, len(weeks) + 1)
    return weeks, note


def period_calendar(opening: date, final: date, weeks_by_period: Mapping, schedule: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, str]:
    """Start and end day of every matchup period, rebuilt from ESPN's weeks.

    weeks_by_period is ESPN's settings.matchup_periods: {"1": [1], ..., "20": [20, 21]}.
    """
    periods = {int(p): [int(w) for w in ws] for p, ws in weeks_by_period.items()}
    n_weeks = max(w for ws in periods.values() for w in ws)
    weeks, note = espn_weeks(opening, final, n_weeks, schedule)
    rows = [{"period": p, "start": weeks.loc[min(ws), "start"], "end": weeks.loc[max(ws), "end"],
             "source": "derived"} for p, ws in sorted(periods.items())]
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS), note


def apply_known_periods(calendar: pd.DataFrame, known: Mapping, opening: date,
                        current_period: int) -> tuple[pd.DataFrame, list[str]]:
    """Overwrite derived dates with ESPN's for periods that have started.

    known is ESPN's matchup_ids: {period: [scoring period, ...]}, which for
    the current period only runs through today, so only its start is final.
    Returns the calendar and a list of periods where ESPN disagreed.
    """
    cal = calendar.copy()
    differs = []
    for period, sps in known.items():
        period, sps = int(period), [int(s) for s in sps]
        if not sps or period not in set(cal["period"]):
            continue
        i = cal.index[cal["period"] == period][0]
        start = scoring_period_date(opening, min(sps))
        end = scoring_period_date(opening, max(sps)) if period < current_period else cal.at[i, "end"]
        if (start, end) != (cal.at[i, "start"], cal.at[i, "end"]):
            differs.append(f"period {period}: ESPN {start:%b %d}-{end:%b %d}, "
                           f"rebuilt {cal.at[i, 'start']:%b %d}-{cal.at[i, 'end']:%b %d}")
        cal.loc[i, ["start", "end", "source"]] = [start, end, "espn" if period < current_period else "espn start"]
    return cal, differs


def league_schedule_frame(calendar: pd.DataFrame, opponents: Mapping[tuple[int, int], int | None],
                          team_ids: Iterable[int], regular_periods: int) -> pd.DataFrame:
    """One row per fantasy team per matchup period: dates, playoff flag and opponent."""
    rows = []
    for p in calendar.itertuples():
        for t in team_ids:
            rows.append({"period": int(p.period), "start": str(p.start), "end": str(p.end),
                         "playoff": int(p.period) > regular_periods, "team_id": int(t),
                         "opponent_id": opponents.get((int(p.period), int(t)))})
    out = pd.DataFrame(rows)
    out["opponent_id"] = out["opponent_id"].astype("Int64")
    return out
