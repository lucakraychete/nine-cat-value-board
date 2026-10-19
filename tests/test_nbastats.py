import numpy as np
import pandas as pd
import pytest

from ninecat import nbastats
from ninecat.nbastats import (
    NOT_YET,
    espn_team,
    game_log_frame,
    last_season_frame,
    parse_bbr_advanced,
    season_label,
    team_frame,
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


def test_game_log_frame():
    logs = pd.DataFrame({"PLAYER_NAME": ["A B Jr."], "GAME_DATE": ["2026-01-05T00:00:00"], "MIN": [31.26]})
    out = game_log_frame(logs, "2025-26")
    assert out.to_dict("records") == [{"name_key": "a b", "date": "2026-01-05", "min": 31.3, "season": "2025-26"}]
    assert list(game_log_frame(pd.DataFrame(), "x").columns) == ["name_key", "date", "min", "season"]


def test_fetch_all_reports_each_source_and_survives_failures(monkeypatch, teams):
    def boom(*a, **k):
        raise TimeoutError("slow")

    monkeypatch.setattr(nbastats, "fetch_team_totals", lambda s: teams)
    monkeypatch.setattr(nbastats, "fetch_player_totals", lambda s: pd.DataFrame([totals()]))
    monkeypatch.setattr(nbastats, "fetch_bbr_advanced", boom)
    monkeypatch.setattr(nbastats, "fetch_game_logs", lambda s, t=60: boom() if s == "2026-27" else
                        game_log_frame(pd.DataFrame({"PLAYER_NAME": ["X"], "GAME_DATE": ["2026-01-01"], "MIN": [20]}), s))
    frames, status = nbastats.fetch_all(2027)
    assert status["nba_players"] == "ok" and status["bbr_bpm"].startswith("failed: TimeoutError")
    assert status["game_logs_2026-27"].startswith(NOT_YET)
    assert frames["last_season"]["adv_bpm"].isna().all()
    assert frames["pace"] == {"PHL": 100.0, "BOS": 98.0}
    assert len(frames["game_logs"]) == 1
