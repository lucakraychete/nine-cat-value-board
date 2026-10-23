"""Smoke test: the app renders every view on synthetic data, no network."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ninecat import ingest
from tests.conftest import TEAMS, synthetic_players

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    players = synthetic_players()
    meta = {"year": 2027, "my_team_id": 1, "teams": {str(i + 1): t for i, t in enumerate(TEAMS)},
            "team_abbrevs": {str(i + 1): t[:3].upper() for i, t in enumerate(TEAMS)},
            "sources": {"espn": "ok"}, "pace": {"DEN": 101.0}}
    logs = pd.DataFrame({"name_key": "player 0", "date": [f"2026-01-{d:02d}" for d in range(1, 20)],
                         "min": 30.0, "season": "2025-26"})
    monkeypatch.setattr(ingest, "load_cached", lambda *a, **k: (players.copy(), dict(meta)))
    monkeypatch.setattr(ingest, "load_game_logs", lambda *a, **k: logs)
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: pytest.fail("the app must not fetch on load"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at, players


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


def test_board_renders_and_controls_work(app):
    at, players = app
    no_errors(at)
    assert [t.label for t in at.tabs] == ["Players", "My team", "Waiver wire", "Team categories"]
    assert "Impact" in at.dataframe[0].value.columns
    for source in ("proj", "last", "blend"):
        at.segmented_control(key="source").set_value(source).run()
        no_errors(at)
    at.segmented_control(key="cols").set_value("Advanced").run()
    no_errors(at)
    at.pills(key="punt_8").set_value(["AST", "FT%"]).run()
    no_errors(at)


def test_without_a_team_league_views_fall_back(app):
    at, players = app
    at.selectbox[0].set_value(None).run()
    no_errors(at)


# 0 is on "my team" (Alpha), 1 on another team, 85 a free agent: three impact framings.
@pytest.mark.parametrize("which", [0, 1, 85])
def test_player_page_renders(app, which):
    at, players = app
    at.query_params["player"] = str(players["player_id"].iloc[which])
    at.run()
    no_errors(at)
    assert len(at.get("plotly_chart")) >= 5
