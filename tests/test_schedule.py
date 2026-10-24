from datetime import date, timedelta

import pandas as pd
import pytest

from ninecat.schedule import (
    apply_known_periods,
    calendar_weeks,
    espn_weeks,
    games_between,
    games_per_period,
    league_schedule_frame,
    all_star_break,
    period_calendar,
    remaining_games,
    scoring_period_date,
)


def sched(games):
    """Hand-built schedule from (date, home, away)."""
    return pd.DataFrame([{"date": d, "game_id": f"g{i}", "home": h, "away": a}
                         for i, (d, h, a) in enumerate(games)])


SCHED = sched([
    ("2026-10-20", "NYK", "BOS"),
    ("2026-10-21", "LAL", "NYK"),
    ("2026-10-25", "BOS", "LAL"),
    ("2026-10-26", "NYK", "LAL"),   # Monday: next week
    ("2026-10-28", "BOS", "NYK"),
])


def test_games_between_counts_both_sides_and_every_team():
    g = games_between(SCHED, "2026-10-20", "2026-10-25")
    assert g.to_dict() == {"BOS": 2, "LAL": 2, "NYK": 2}
    assert games_between(SCHED, date(2026, 10, 26), date(2026, 10, 26), ["NYK", "LAL", "MIA"]).to_dict() == {
        "NYK": 1, "LAL": 1, "MIA": 0}


def test_remaining_games_from_today_including_today():
    assert remaining_games(SCHED, "2026-10-26").to_dict() == {"BOS": 1, "LAL": 1, "NYK": 2}
    assert remaining_games(SCHED, "2027-01-01").sum() == 0


def test_games_per_period_from_a_calendar():
    cal = pd.DataFrame({"period": [1, 2], "start": [date(2026, 10, 20), date(2026, 10, 26)],
                        "end": [date(2026, 10, 25), date(2026, 11, 1)]})
    g = games_per_period(SCHED, cal)
    assert g.loc[1].to_dict() == {"BOS": 2, "LAL": 2, "NYK": 2}
    assert g.loc[2].to_dict() == {"BOS": 1, "LAL": 1, "NYK": 2}


def test_scoring_period_one_is_opening_night():
    assert scoring_period_date(date(2026, 10, 20), 1) == date(2026, 10, 20)
    assert scoring_period_date(date(2026, 10, 20), 167) == date(2027, 4, 4)


def test_calendar_weeks_run_monday_to_sunday_clipped():
    w = calendar_weeks(date(2026, 10, 20), date(2026, 11, 4))   # Tuesday to Wednesday
    assert w.to_dict("records") == [
        {"start": date(2026, 10, 20), "end": date(2026, 10, 25)},
        {"start": date(2026, 10, 26), "end": date(2026, 11, 1)},
        {"start": date(2026, 11, 2), "end": date(2026, 11, 4)},
    ]


def season_with_gaps(opening, final, *gaps):
    """A game every day except the gaps."""
    days = [opening + timedelta(days=i) for i in range((final - opening).days + 1)]
    return sched([(d.isoformat(), "NYK", "BOS") for d in days if not any(a <= d <= b for a, b in gaps)])


def test_espn_weeks_fold_the_all_star_break_into_one_period():
    opening, final = date(2026, 11, 30), date(2027, 3, 7)    # 14 calendar weeks, Monday to Sunday
    cup = (date(2026, 12, 4), date(2026, 12, 11))            # longer, but undecided Cup games
    break_ = (date(2027, 2, 19), date(2027, 2, 24))          # straddles the weeks of Feb 15 and Feb 22
    s = season_with_gaps(opening, final, cup, break_)
    assert all_star_break(s) == break_
    weeks, note = espn_weeks(opening, final, 13, s)
    assert len(weeks) == 13 and "All-Star" in note
    assert (weeks.loc[12, "start"], weeks.loc[12, "end"]) == (date(2027, 2, 15), date(2027, 2, 28))
    assert (weeks.loc[13, "start"], weeks.loc[13, "end"]) == (date(2027, 3, 1), final)
    assert all_star_break(season_with_gaps(opening, final, cup)) is None


def test_espn_weeks_flag_a_mismatch_they_cannot_explain():
    opening, final = date(2026, 10, 20), date(2026, 11, 29)
    weeks, note = espn_weeks(opening, final, 4)
    assert note.startswith("unverified") and len(weeks) == 4 and weeks.loc[4, "end"] == final
    weeks, note = espn_weeks(opening, final, 6)
    assert note == "Monday-Sunday weeks" and len(weeks) == 6


def test_period_calendar_spans_playoff_rounds():
    opening, final = date(2026, 10, 20), date(2026, 11, 29)
    cal, _ = period_calendar(opening, final, {"1": [1], "2": [2], "3": [3], "4": [4], "5": [5, 6]})
    assert cal["period"].tolist() == [1, 2, 3, 4, 5]
    assert (cal.iloc[-1]["start"], cal.iloc[-1]["end"]) == (date(2026, 11, 16), final)


def test_known_periods_win_and_report_differences():
    opening = date(2026, 10, 20)
    cal, _ = period_calendar(opening, date(2026, 11, 29), {str(p): [p] for p in range(1, 7)})
    # ESPN says period 1 ran 13 days (Oct 20-Nov 1); period 2 has started Nov 2.
    known = {1: [str(s) for s in range(1, 14)], 2: ["14", "15"]}
    fixed, differs = apply_known_periods(cal, known, opening, current_period=2)
    p1, p2 = fixed.iloc[0], fixed.iloc[1]
    assert (p1["start"], p1["end"], p1["source"]) == (opening, date(2026, 11, 1), "espn")
    assert (p2["start"], p2["source"]) == (date(2026, 11, 2), "espn start")
    assert len(differs) == 2 and differs[0].startswith("period 1")
    same, none = apply_known_periods(cal, {1: [str(s) for s in range(1, 7)]}, opening, current_period=2)
    assert none == [] and same.iloc[0]["source"] == "espn"


def test_league_schedule_frame_has_every_team_every_period():
    cal = pd.DataFrame({"period": [1, 2], "start": [date(2026, 10, 20), date(2026, 10, 26)],
                        "end": [date(2026, 10, 25), date(2026, 11, 1)], "source": "derived"})
    out = league_schedule_frame(cal, {(1, 1): 2, (1, 2): 1}, [1, 2], regular_periods=1)
    assert len(out) == 4 and out["playoff"].tolist() == [False, False, True, True]
    assert out.loc[0, "opponent_id"] == 2 and pd.isna(out.loc[2, "opponent_id"])
    assert out.loc[0, "start"] == "2026-10-20"


@pytest.mark.parametrize("today,expected", [("2026-10-19", 10), ("2026-10-28", 2), ("2026-10-29", 0)])
def test_remaining_total(today, expected):
    assert remaining_games(SCHED, today).sum() == expected
