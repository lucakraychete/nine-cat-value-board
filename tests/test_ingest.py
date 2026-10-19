import json

import pandas as pd

from ninecat import ingest
from ninecat.ingest import UNDRAFTED_ADP, adp_from_players, load_cached, load_game_logs


def test_adp_keeps_drafted_players_only():
    feed = [
        {"player": {"id": 1, "ownership": {"averageDraftPosition": 1.4}}},
        {"player": {"id": 2, "ownership": {"averageDraftPosition": UNDRAFTED_ADP + 0.4}}},
        {"player": {"id": 3, "ownership": {"averageDraftPosition": 0}}},
        {"player": {"id": 4}},
        {"id": 5, "ownership": {"averageDraftPosition": 88.0}},
    ]
    assert adp_from_players(feed) == {1: 1.4, 5: 88.0}


def test_load_cached_attaches_last_season_and_pace(tmp_path):
    pd.DataFrame({"player_id": [1], "name": ["Nikola Jokić"], "team_id": [2], "PTS": [28.0]}).to_csv(
        tmp_path / ingest.PLAYERS_FILE, index=False)
    pd.DataFrame({"name_key": ["nikola jokic"], "L_PTS": [29.6]}).to_csv(tmp_path / ingest.LAST_SEASON_FILE, index=False)
    (tmp_path / ingest.PACE_FILE).write_text(json.dumps({"DEN": 99.5}))
    df, meta = load_cached(tmp_path)
    assert df["L_PTS"].iloc[0] == 29.6 and meta["pace"] == {"DEN": 99.5}
    assert load_game_logs(tmp_path).empty


def test_load_cached_without_cache(tmp_path):
    assert load_cached(tmp_path) is None
