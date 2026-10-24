import numpy as np
import pandas as pd
import pytest

from ninecat import nbastats
from ninecat.nbastats import (
    LOG_COLUMNS,
    NOT_YET,
    SEASON_COLUMNS,
    TOTALS_COLUMNS,
    espn_team,
    game_log_frame,
    last_season_frame,
    parse_bbr_advanced,
    player_ages,
    schedule_frame,
    season_label,
    season_to_date,
    team_frame,
    totals_frame,
    with_retry,
)
from tests.test_rates import team

BBR_HTML = """<html><body><table id="other"><tr><td data-stat="name_display">Nope</td></tr></table>
<table id="advanced"><thead><tr><th data-stat="name_display">Player</th></tr></thead><tbody>
<tr><td data-stat="name_display"><a href="#">Nikola Jokić</a></td><td data-stat="team_name_abbr">DEN</td><td data-stat="bpm">14.2</td></tr>
<tr class="thead"><td data-stat="name_display">Player</td></tr>
<tr><td data-stat="name_display">Mover Guy</td><td data-stat="team_name_abbr">2TM</td><td data-stat="bpm">1.5</td></tr>
<tr><td data-stat="name_display">Mover Guy</td><td data-stat="team_name_abbr">BOS</td><td data-stat="bpm">3.0</td></tr>
<tr><td data-stat="name_display">League Average</td><td data-stat="bpm"></td></tr>
</tbody></table></body></html>"""


def totals(**kw):
    base = dict(PLAYER_NAME="Nikola Jokić", TEAM_ABBREVIATION="PHI", GP=70, MIN=2450.0, FGM=700,
                FGA=1200, FG3M=120, FTM=350, FTA=430, OREB=200, DREB=700, REB=900, AST=700, STL=100,
                BLK=60, TOV=230, PF=180, PTS=1870)
    base.update(kw)
    return base


@pytest.fixture
def teams():
    return pd.DataFrame([team(), team(PACE=98.0)], index=["PHL", "BOS"])


def test_season_label_from_espn_year():
    assert season_label(2027) == "2026-27"
    assert season_label(2027, 1) == "2025-26"
    assert season_label(2000, 1) == "1998-99"


def test_espn_team_abbreviations():
    assert espn_team("PHI") == "PHL" and espn_team("PHX") == "PHO" and espn_team("DEN") == "DEN"


def test_parse_bbr_keeps_season_total_row_and_drops_headers():
    out = parse_bbr_advanced(BBR_HTML).set_index("name_key")
    assert list(out.index) == ["nikola jokic", "mover guy"]
    assert out.loc["mover guy", "bpm"] == 1.5 and out.loc["mover guy", "team"] == "2TM"
    assert parse_bbr_advanced("<html></html>").empty


def test_team_frame_joins_opponents_pace_and_renames():
    base = pd.DataFrame([team()], index=[1610612755])
    opp = base[[c for c in base.columns if c.startswith("OPP_")]]
    adv = pd.DataFrame({"PACE": [101.5]}, index=[1610612755])
    out = team_frame(base, opp, adv, {1610612755: "PHL"})
    assert list(out.index) == ["PHL"] and out.loc["PHL", "PACE"] == 101.5 and "OPP_REB" in out


def test_last_season_frame_per_game_rates_bpm_and_prev_minutes(teams):
    tot = pd.DataFrame([totals(), totals(PLAYER_NAME="Bench Guy", TEAM_ABBREVIATION="BOS", GP=0, MIN=0.0)])
    prev = pd.DataFrame([totals(GP=50, MIN=1500.0)])
    out = last_season_frame(tot, teams, prev, parse_bbr_advanced(BBR_HTML)).set_index("name_key")
    j = out.loc["nikola jokic"]
    assert j["L_G"] == 70 and j["L_PTS"] == pytest.approx(1870 / 70) and j["L_3PM"] == pytest.approx(120 / 70)
    assert j["L_TO"] == pytest.approx(230 / 70)
    assert j["adv_ts"] == pytest.approx(1870 / (2 * (1200 + 0.44 * 430)))
    assert j["adv_bpm"] == 14.2 and j["prev_MIN"] == pytest.approx(30.0)
    assert np.isnan(out.loc["bench guy", "adv_usg"]) and np.isnan(out.loc["bench guy", "L_PTS"])


def test_duplicate_names_keep_the_bigger_role(teams):
    tot = pd.DataFrame([totals(MIN=500.0), totals(MIN=2000.0)])
    out = last_season_frame(tot, teams)
    assert len(out) == 1 and out["L_MIN"].iloc[0] == pytest.approx(2000 / 70)


def log_rows(**kw):
    """Raw PlayerGameLogs rows (the columns game_log_frame reads)."""
    base = dict(PLAYER_ID=203999, PLAYER_NAME="Nikola Jokić", TEAM_ABBREVIATION="DEN", GAME_ID="0022600011",
                GAME_DATE="2026-10-21T00:00:00", MATCHUP="DEN vs. PHX", MIN=34.567, PTS=30, REB=12, AST=9,
                STL=2, BLK=1, FG3M=2, TOV=3, FGM=11, FGA=19, FTM=6, FTA=7)
    base.update(kw)
    return base


def test_game_log_frame_keeps_the_full_box_score_regular_season_only():
    logs = pd.DataFrame([
        log_rows(),
        log_rows(GAME_ID="0022600030", GAME_DATE="2026-10-23T00:00:00", MATCHUP="DEN @ PHI", PTS=22),
        log_rows(GAME_ID="0012600009", GAME_DATE="2026-10-03T00:00:00"),   # preseason
        log_rows(GAME_ID="0042500101", GAME_DATE="2026-04-20T00:00:00"),   # playoffs
        log_rows(GAME_ID=22600040, PLAYER_NAME="A B Jr.", MATCHUP="LAL vs. GSW", TEAM_ABBREVIATION="LAL"),
    ])
    out = game_log_frame(logs, "2026-27")
    assert list(out.columns) == LOG_COLUMNS
    assert out["game_id"].tolist() == ["0022600011", "0022600030", "0022600040"]
    first = out.iloc[0]
    assert (first["date"], first["team"], first["opp"], first["home"]) == ("2026-10-21", "DEN", "PHO", True)
    assert (first["PTS"], first["REB"], first["AST"], first["3PM"], first["TO"]) == (30, 12, 9, 2, 3)
    assert (first["FGM"], first["FGA"], first["FTM"], first["FTA"]) == (11, 19, 6, 7)
    assert first["MIN"] == pytest.approx(34.57) and first["season"] == "2026-27"
    away = out.iloc[1]
    assert (away["opp"], away["home"]) == ("PHL", False)
    assert out.iloc[2]["name_key"] == "a b"
    assert list(game_log_frame(pd.DataFrame(), "x").columns) == LOG_COLUMNS


def frame_logs(rows):
    return game_log_frame(pd.DataFrame([log_rows(**r) for r in rows]), "2026-27")


def test_season_to_date_line_recent_minutes_and_age():
    rows = [dict(GAME_ID=f"00226000{i:02d}", GAME_DATE=f"2026-11-{d:02d}", MIN=m, PTS=p)
            for i, (d, m, p) in enumerate([(1, 20, 10), (5, 24, 14), (10, 30, 20), (12, 32, 22),
                                           (13, 34, 24), (14, 36, 26)])]
    rows.append(dict(PLAYER_NAME="Other Guy", GAME_ID="0022600099", GAME_DATE="2026-11-02", MIN=10))
    ages = pd.Series({"nikola jokic": 31.0})
    out = season_to_date(frame_logs(rows), ages, asof="2026-11-14").set_index("name_key")
    assert list(out.reset_index().columns) == SEASON_COLUMNS
    j = out.loc["nikola jokic"]
    assert j["S_G"] == 6 and j["S_PTS"] == pytest.approx(116 / 6) and j["S_MIN"] == pytest.approx(176 / 6)
    assert j["S_FGA"] == 19 and j["S_team"] == "DEN" and j["age"] == 31.0
    # last 7 days counting Nov 14: Nov 8-14, four games
    assert j["S_G_7D"] == 4 and j["S_MIN_7D"] == pytest.approx((30 + 32 + 34 + 36) / 4)
    assert j["S_G_15D"] == 6 and j["S_MIN_15D"] == pytest.approx(176 / 6)  # Oct 31-Nov 14
    assert j["S_MIN_L5"] == pytest.approx((24 + 30 + 32 + 34 + 36) / 5)
    o = out.loc["other guy"]
    assert o["S_G_7D"] == 0 and np.isnan(o["S_MIN_7D"]) and np.isnan(o["age"])
    assert season_to_date(pd.DataFrame(columns=LOG_COLUMNS)).empty


def test_totals_frame_and_ages():
    raw = pd.DataFrame([totals(PLAYER_ID=7, AGE=29.0)])
    out = totals_frame(raw, "2019-20")
    assert list(out.columns) == TOTALS_COLUMNS
    row = out.iloc[0]
    assert (row["season"], row["team"], row["age"], row["GP"], row["3PM"], row["TO"]) == ("2019-20", "PHL", 29.0, 70, 120, 230)
    assert player_ages(raw).to_dict() == {"nikola jokic": 29.0}


def test_schedule_frame_regular_season_with_known_teams():
    raw = pd.DataFrame({
        "gameId": ["0012600009", "0022600001", "0022600002", "0062600001", "0022600090"],
        "gameDateEst": ["2026-10-03T00:00:00Z", "2026-10-20T00:00:00Z", "2026-10-20T00:00:00Z",
                        "2026-12-16T00:00:00Z", "2026-12-12T00:00:00Z"],
        "homeTeam_teamTricode": ["TOR", "NYK", "LAL", "LAL", ""],
        "awayTeam_teamTricode": ["MIA", "PHI", "PHX", "NYK", ""],
    })
    out = schedule_frame(raw)
    assert out.to_dict("records") == [
        {"date": "2026-10-20", "game_id": "0022600001", "home": "NYK", "away": "PHL"},
        {"date": "2026-10-20", "game_id": "0022600002", "home": "LAL", "away": "PHO"},
    ]


def test_with_retry_retries_once_with_backoff_then_pauses():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return "ok"

    assert with_retry(flaky, backoff=5, pause=0.5, sleep=sleeps.append) == "ok"
    assert len(calls) == 2 and sleeps == [5, 0.5]
    sleeps.clear()
    with pytest.raises(TimeoutError):
        with_retry(lambda: (_ for _ in ()).throw(TimeoutError("down")), backoff=5, pause=0.5, sleep=sleeps.append)
    assert sleeps == [5, 0.5]


def test_fetch_all_reports_each_source_and_survives_failures(monkeypatch, teams):
    def boom(*a, **k):
        raise TimeoutError("slow")

    monkeypatch.setattr(nbastats, "fetch_team_totals", lambda s: teams)
    monkeypatch.setattr(nbastats, "fetch_player_totals",
                        lambda s, timeout=60: boom() if s == "2026-27" else pd.DataFrame([totals()]))
    monkeypatch.setattr(nbastats, "fetch_bbr_advanced", boom)
    monkeypatch.setattr(nbastats, "fetch_game_logs", lambda s, t=60: frame_logs([{}]) if s == "2026-27"
                        else frame_logs([dict(GAME_DATE="2026-01-01")]))
    frames, status = nbastats.fetch_all(2027)
    assert status["nba_players"] == "ok" and status["bbr_bpm"].startswith("failed: TimeoutError")
    assert status["nba_season"] == "ok" and status["nba_ages"].startswith("failed: TimeoutError")
    assert frames["last_season"]["adv_bpm"].isna().all()
    assert frames["pace"] == {"PHL": 100.0, "BOS": 98.0}
    assert len(frames["last_logs"]) == 1 and len(frames["season_logs"]) == 1
    assert frames["season"]["S_G"].tolist() == [1] and frames["season"]["age"].isna().all()


def test_fetch_all_skips_static_sources_and_waits_for_opening_night(monkeypatch):
    def must_not(*a, **k):
        raise AssertionError("static source refetched")

    monkeypatch.setattr(nbastats, "fetch_team_totals", must_not)
    monkeypatch.setattr(nbastats, "fetch_bbr_advanced", must_not)
    monkeypatch.setattr(nbastats, "fetch_player_totals", lambda s, timeout=60: pd.DataFrame([totals(AGE=30)]))
    monkeypatch.setattr(nbastats, "fetch_game_logs",
                        lambda s, t=60: must_not() if s != "2026-27" else pd.DataFrame(columns=LOG_COLUMNS))
    frames, status = nbastats.fetch_all(2027, static=False, last_logs=False)
    assert set(status) == {"nba_season", "nba_ages"} and status["nba_season"].startswith(NOT_YET)
    assert "season" not in frames and "season_logs" not in frames
