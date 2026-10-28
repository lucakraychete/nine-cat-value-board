"""Smoke test: the app renders every view on synthetic data, no network."""

import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ninecat import ingest
from ninecat.urlstate import TABS
from tests.conftest import TEAMS, synthetic_players

APP = str(Path(__file__).resolve().parent.parent / "app.py")
META = {"year": 2027, "my_team_id": 1, "teams": {str(i + 1): t for i, t in enumerate(TEAMS)},
        "team_abbrevs": {str(i + 1): t[:3].upper() for i, t in enumerate(TEAMS)},
        "sources": {"espn": {"status": "ok", "updated": "2026-10-23T12:00:00+00:00"},
                    "bbr_bpm": {"status": "failed: TimeoutError", "updated": "2026-10-20T12:00:00+00:00"}},
        "fetched_at": "2026-10-23T12:00:00+00:00", "pace": {"DEN": 101.0}}


def synthetic_logs() -> pd.DataFrame:
    """Last season for player 0, and this season for players 0 and 1."""
    rows = [{"name_key": "player 0", "date": f"2026-01-{d:02d}", "season": "2025-26"} for d in range(1, 20)]
    rows += [{"name_key": f"player {i}", "date": f"2026-10-{d}", "season": "2026-27"}
             for i in (0, 1) for d in (21, 23, 25)]
    logs = pd.DataFrame(rows)
    for c, v in {"MIN": 30.0, "PTS": 20.0, "REB": 6.0, "AST": 4.0, "STL": 1.0, "BLK": 1.0, "3PM": 2.0,
                 "TO": 2.0, "FGM": 7.0, "FGA": 15.0, "FTM": 4.0, "FTA": 5.0}.items():
        logs[c] = v
    return logs


SCHEDULE = pd.DataFrame({"date": ["2026-10-20", "2026-10-27", "2026-11-03", "2027-03-10"],
                         "game_id": ["g1", "g2", "g3", "g4"], "home": ["DEN", "BOS", "PHL", "LAL"],
                         "away": ["MIA", "DEN", "LAL", "BOS"]})
LEAGUE_SCHEDULE = pd.DataFrame({"period": [1, 2, 20], "start": ["2026-10-20", "2026-10-26", "2027-03-08"],
                                "end": ["2026-10-25", "2026-11-01", "2027-03-21"],
                                "playoff": [False, False, True], "team_id": 1, "opponent_id": 2})


def serve_synthetic(monkeypatch, players, auto_refresh: bool, report: dict | None = None):
    tables = {ingest.SCHEDULE_FILE: SCHEDULE, ingest.LEAGUE_SCHEDULE_FILE: LEAGUE_SCHEDULE}
    preseason = players[["player_id", "name", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA",
                         "FTM", "FTA", "MIN", "GP"]].iloc[:70]
    monkeypatch.setenv("LEAGUE_ID", "1")
    monkeypatch.setenv("YEAR", "2027")
    monkeypatch.setenv("AUTO_REFRESH", str(auto_refresh).lower())
    monkeypatch.setattr(ingest, "load_cached", lambda *a, **k: (players.copy(), dict(META)))
    monkeypatch.setattr(ingest, "load_game_logs", lambda *a, **k: synthetic_logs())
    monkeypatch.setattr(ingest, "load_meta", lambda *a, **k: dict(META))
    monkeypatch.setattr(ingest, "load_table", lambda name, *a, **k: tables.get(name, pd.DataFrame()).copy())
    monkeypatch.setattr(ingest, "snapshot_days", lambda *a, **k: [date(2026, 10, 18), date(2026, 10, 23)])
    monkeypatch.setattr(ingest, "load_snapshot", lambda day, *a, **k: preseason.copy())
    monkeypatch.setattr(ingest, "load_model_report", lambda *a, **k: dict(report or {}))
    monkeypatch.setattr(ingest, "snapshot_files", lambda *a, **k: [])
    st.cache_resource.clear()   # a fresh "server start"
    st.cache_data.clear()       # and nothing cached from your real data/


@pytest.fixture
def app(monkeypatch):
    players = synthetic_players()
    fetched = []
    serve_synthetic(monkeypatch, players, auto_refresh=False)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: fetched.append(1))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    yield at, players
    assert not fetched, "the app must not fetch on load with AUTO_REFRESH=false"


def status_text(at) -> str:
    return "".join(h.value for h in at.get("html"))


def test_boot_refresh_runs_in_background_once_then_offers_reload(monkeypatch):
    players = synthetic_players()
    serve_synthetic(monkeypatch, players, auto_refresh=True)
    gate, runs = threading.Event(), []
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: (runs.append(1), gate.wait(10)))

    first = AppTest.from_file(APP, default_timeout=60)
    first.run()                                   # renders while the refresh is still going
    no_errors(first)
    assert "Updating data…" in status_text(first) and first.tabs
    second = AppTest.from_file(APP, default_timeout=60)
    second.run()                                  # another browser tab: no second refresh
    assert runs == [1]

    newer = dict(META, fetched_at="2026-10-23T13:00:00+00:00")
    monkeypatch.setattr(ingest, "load_meta", lambda *a, **k: newer)
    gate.set()
    from ui.refresh import refresh_job
    refresh_job().join(5)
    first.run()
    no_errors(first)
    assert "Updating data…" not in status_text(first)
    assert any(b.label == "New data ready, reload" for b in first.button)
    assert runs == [1]
    st.cache_resource.clear()


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


def test_board_renders_and_controls_work(app):
    at, players = app
    no_errors(at)
    assert [t.label for t in at.tabs] == ["Players", "My team", "Waiver wire", "Trades", "Team categories", "Model"]
    assert "1 source on a saved copy" in "".join(h.value for h in at.get("html"))
    for source in ("proj", "last", "season", "blend"):
        at.segmented_control(key="source").set_value(source).run()
        no_errors(at)
    for view, column in (("Categories", "Impact"), ("Advanced", "P36 Rk"), ("Schedule & role", "Last 10"),
                         ("Overview", "So far")):
        at.segmented_control(key="cols").set_value(view).run()
        no_errors(at)
        assert column in at.dataframe[0].value.columns, view
    at.segmented_control(key="cols").set_value("Advanced").run()
    no_errors(at)
    at.pills(key="punt").set_value(["AST", "FT%"]).run()
    no_errors(at)
    at.toggle(key="to").set_value(False).run()        # 9-cat: the punts carry over
    no_errors(at)
    assert at.pills(key="punt").value == ["AST", "FT%"] and "TO" in at.pills(key="punt").options


def test_overview_is_the_default_and_sorted_by_so_far(app):
    at, players = app
    assert at.segmented_control(key="cols").value == "Overview"
    table = at.dataframe[0].value
    assert list(table.columns[:14]) == ["Player", "Pos", "NBA", "Owner", "GP", "Min", "Min L14", "So far",
                                        "So far Rk", "Pre", "Pre Rk", "Rk Δ", "ROS", "ROS Rk"]
    assert table["So far"].is_monotonic_decreasing
    assert (table.loc[table["GP"] == 0, "So far"] == 0).all()
    assert table["Min L14"].notna().sum() == 2                    # only players 0 and 1 have recent games
    at.pills(key="punt").set_value(["BLK"]).run()
    no_errors(at)
    assert "BLK (punt)" in at.dataframe[0].value.columns


def test_schedule_view_counts_games_and_sparklines(app):
    at, players = app
    at.segmented_control(key="cols").set_value("Schedule & role").run()
    no_errors(at)
    table = at.dataframe[0].value.set_index("Player")
    for _, p in players.iterrows():
        if p["name"] in table.index:     # the one playoff game (Mar 10) is LAL vs BOS
            assert table.loc[p["name"], "Playoffs"] == (1 if p["pro_team"] in ("LAL", "BOS") else 0)
    assert len(table.loc[players.loc[0, "name"], "Last 10"]) == 3


def test_old_column_urls_still_resolve(monkeypatch):
    players = synthetic_players()
    serve_synthetic(monkeypatch, players, auto_refresh=False)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["cols"] = "Categories"
    at.run()
    no_errors(at)
    assert at.segmented_control(key="cols").value == "Categories" and "Impact" in at.dataframe[0].value.columns


def test_every_tab_renders_its_summaries(app):
    at, players = app
    text = all_text(at)
    assert [t.label for t in at.tabs] == TABS
    for marker in ("Category outlook", "Strengths and weaknesses", "Roster",        # My team
                   "#1 pickup", "Best drop", "Every pickup", "Who to drop",          # Waiver wire
                   "Your build", "Projected standings"):                            # Team categories
        assert marker in text, marker
    names = [b.label for b in at.button if b.key and b.key.startswith("pname_pick_")]
    assert len(names) == 5 and all(n in set(players["name"]) for n in names)


def test_url_restores_every_setting(monkeypatch):
    players = synthetic_players()
    serve_synthetic(monkeypatch, players, auto_refresh=False)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"team": "Bravo", "source": "This season", "punt": ["AST", "BLK"], "disc": "false",
                            "tab": "Waiver wire", "cols": "Advanced", "pos": "C", "pool": "100"})
    at.run()
    no_errors(at)
    assert at.selectbox(key="team").value == "Bravo"
    assert at.segmented_control(key="source").value == "season"
    assert at.pills(key="punt").value == ["AST", "BLK"]
    assert at.toggle(key="disc").value is False
    assert at.segmented_control(key="cols").value == "Advanced"
    assert at.selectbox(key="pos").value == "C" and at.number_input(key="pool").value == 100
    assert "Waiver wire for Bravo" in all_text(at)


def test_player_url_opens_his_page_and_back_returns_to_the_tab(monkeypatch):
    players = synthetic_players()
    serve_synthetic(monkeypatch, players, auto_refresh=False)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    pid = str(players["player_id"].iloc[3])
    at.query_params.update({"tab": "My team", "source": "ESPN proj", "player": pid})
    at.run()
    no_errors(at)
    assert not at.tabs and players["name"].iloc[3] in all_text(at)
    back = next(b for b in at.button if b.label.startswith("← Back"))
    assert back.label == "← Back to My team"
    back.click().run()
    no_errors(at)
    assert "player" not in at.query_params and at.query_params["tab"] == "My team"
    assert at.query_params["source"] == "ESPN proj" and [t.label for t in at.tabs] == TABS


def all_text(at) -> str:
    parts = [h.value for h in at.get("html")] + [m.value for m in at.markdown]
    return " ".join(parts)


def test_without_a_team_league_views_fall_back(app):
    at, players = app
    at.selectbox(key="team").set_value(None).run()
    no_errors(at)


# 0 is on "my team" (Alpha), 1 on another team, 85 a free agent: three impact framings.
@pytest.mark.parametrize("which", [0, 1, 85])
def test_player_page_renders(app, which):
    at, players = app
    at.query_params["player"] = str(players["player_id"].iloc[which])
    at.run()
    no_errors(at)
    assert len(at.get("plotly_chart")) >= 5


# ---------------------------------------------------------------- trades

def test_trades_tab_renders_calculator_finder_and_surplus(app):
    at, players = app
    no_errors(at)
    text = all_text(at)
    for marker in ("Valued on: rest of season (interim)", "Trade calculator", "Trade finder",
                   "Who to trade, and to whom"):
        assert marker in text, marker
    finder = next(d.value for d in at.dataframe if "Open" in d.value.columns and "Gap" in d.value.columns)
    assert len(finder) > 0 and (finder["You"] > 0).all()


def test_a_bookmarked_trade_opens_in_the_calculator(monkeypatch):
    players = synthetic_players()
    serve_synthetic(monkeypatch, players, auto_refresh=False)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    mine, theirs = players.loc[0, "name"], players.loc[1, "name"]      # Alpha (my team) and Bravo
    at.query_params.update({"tab": "Trades", "tp": "Bravo", "give": [mine], "get": [theirs], "veto": "6",
                            "fm": "cat", "fcat": "REB", "f22": "true"})
    at.run()
    no_errors(at)
    assert at.selectbox(key="tp").value == "Bravo"
    assert at.multiselect(key="give").value == [mine] and at.multiselect(key="get").value == [theirs]
    assert at.slider(key="veto").value == 6 and at.toggle(key="f22").value is True
    text = all_text(at)
    assert "category wins for you over the" in text and "Roster check" in text and "after the trade" in text


def test_shop_him_fills_the_calculator(app):
    at, players = app
    at.session_state["pending_trade"] = {"partner": "Charlie", "give": [players.loc[0, "name"]], "get": []}
    at.run()
    no_errors(at)
    assert at.selectbox(key="tp").value == "Charlie"
    assert at.multiselect(key="give").value == [players.loc[0, "name"]] and at.multiselect(key="get").value == []
    assert at.query_params["tab"] == "Trades"


def test_trade_numbers_ignore_the_stats_source(app):
    at, players = app

    def finder():
        return next(d.value for d in at.dataframe if "Open" in d.value.columns and "Gap" in d.value.columns)

    before = finder()
    at.segmented_control(key="source").set_value("last").run()
    no_errors(at)
    pd.testing.assert_frame_equal(before, finder())


def test_player_page_offers_trades_for_another_teams_player(app):
    at, players = app
    at.query_params["player"] = str(players["player_id"].iloc[1])    # on Bravo
    at.run()
    no_errors(at)
    assert f"Trades for {players.loc[1, 'name']}" in all_text(at)


# ---------------------------------------------------------------- the rest-of-season model

from tests.conftest import with_model

MODELS = ["Season to date", "Last season", "Blend 60/40", "Interim (app today)", "Shrinkage", "Boosted"]
EVAL = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FG%", "FT%"]


def fake_report(beats_blend: bool = True) -> dict:
    rows = [{"model": m, "k": k, "cat": c, "mae": 1.0 + i / 10, "rmse": 1.5, "n": 400}
            for i, m in enumerate(MODELS) for k in (3, 5, 10, 20, 40) for c in EVAL]
    return {
        "created": "2026-10-28T07:00:00+00:00", "chosen": "boosted", "model_beats_blend": beats_blend,
        "seasons": {"train": ["2017-18", "2023-24"], "validate": ["2024-25"], "test": ["2025-26"]},
        "checkpoints": {"train": 15028, "validate": 2261, "test": 2261},
        "backtest": {"test": rows, "validate": rows},
        "value_rank": {"test": [{"model": m, "k": k, "spearman": 0.6 + i / 50} for i, m in enumerate(MODELS)
                                for k in (3, 5, 10, 20, 40)]},
        "test_verdict": {"vs Blend 60/40": {"mae_change": -0.25}, "vs Interim (app today)": {"mae_change": -0.13}},
        "eval_cats": EVAL, "k": {"PTS": {"k": 257.0, "games_at_30": 8.6}, "FT%": {"k": 78.0, "games_at_30": 27.8}},
        "k_units": {"PTS": "minutes", "FT%": "attempts"},
        "availability": {"k_team_games": 30.0, "newcomer": 0.8},
        "boosting_vs_shrinkage": {"by_cat": {"PTS": {"change": -0.04, "low": -0.07, "high": -0.01}},
                                  "overall": {"change": -0.046, "low": -0.061, "high": -0.031}},
        "coverage_test": [{"cat": c, "k": 0, "coverage": 0.76, "n": 2000} for c in [*EVAL, "VALUE"]],
    }


@pytest.fixture
def model_app(monkeypatch):
    players = with_model(synthetic_players())
    serve_synthetic(monkeypatch, players, auto_refresh=False, report=fake_report())
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at, players


def test_with_a_model_it_is_the_default_source_and_ros_shows_its_range(model_app):
    at, players = model_app
    no_errors(at)
    assert at.segmented_control(key="source").value == "model"
    table = at.dataframe[0].value
    assert "ROS 80%" in table.columns and table["ROS 80%"].str.contains(" to ").any()
    text = all_text(at)
    assert "rest-of-season projection" in text and "Valued on: rest of season (model)" in text
    for marker in ("Rest-of-season model", "Backtest", "How fast each stat settles",
                   "Did gradient boosting earn its place?", "Are the ranges honest?", "This season, as it happens"):
        assert marker in text, marker
    at.segmented_control(key="mk").set_value("10").run()
    no_errors(at)


def test_without_a_win_over_the_blend_the_blend_stays_default(monkeypatch):
    serve_synthetic(monkeypatch, with_model(synthetic_players()), auto_refresh=False, report=fake_report(False))
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    no_errors(at)
    assert at.segmented_control(key="source").value == "blend" and "Model (ROS)" in at.segmented_control(key="source").options


def test_without_a_model_the_source_is_hidden_and_the_tab_explains(app):
    at, players = app
    assert "Model (ROS)" not in at.segmented_control(key="source").options
    assert "No model trained yet" in all_text(at)


def test_player_page_shows_the_models_rest_of_season_line(model_app):
    at, players = model_app
    at.query_params["player"] = str(players["player_id"].iloc[1])
    at.run()
    no_errors(at)
    text = all_text(at)
    assert "Rest of season (model)" in text and "projected games left" in text
    assert any("80% range" in d.value.columns for d in at.dataframe)
