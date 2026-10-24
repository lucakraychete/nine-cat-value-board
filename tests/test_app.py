"""Smoke test: the app renders every view on synthetic data, no network."""

import threading
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


def serve_synthetic(monkeypatch, players, auto_refresh: bool):
    logs = pd.DataFrame({"name_key": "player 0", "date": [f"2026-01-{d:02d}" for d in range(1, 20)],
                         "MIN": 30.0, "season": "2025-26"})
    monkeypatch.setenv("LEAGUE_ID", "1")
    monkeypatch.setenv("YEAR", "2027")
    monkeypatch.setenv("AUTO_REFRESH", str(auto_refresh).lower())
    monkeypatch.setattr(ingest, "load_cached", lambda *a, **k: (players.copy(), dict(META)))
    monkeypatch.setattr(ingest, "load_game_logs", lambda *a, **k: logs)
    monkeypatch.setattr(ingest, "load_meta", lambda *a, **k: dict(META))
    st.cache_resource.clear()   # a fresh "server start"


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
    assert [t.label for t in at.tabs] == ["Players", "My team", "Waiver wire", "Team categories"]
    assert "Impact" in at.dataframe[0].value.columns
    assert "1 source on a saved copy" in "".join(h.value for h in at.get("html"))
    for source in ("proj", "last", "season", "blend"):
        at.segmented_control(key="source").set_value(source).run()
        no_errors(at)
    at.segmented_control(key="cols").set_value("Advanced").run()
    no_errors(at)
    at.pills(key="punt").set_value(["AST", "FT%"]).run()
    no_errors(at)
    at.toggle(key="to").set_value(False).run()        # 9-cat: the punts carry over
    no_errors(at)
    assert at.pills(key="punt").value == ["AST", "FT%"] and "TO" in at.pills(key="punt").options


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
