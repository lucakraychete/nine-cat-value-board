import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd

from ninecat import ingest
from ninecat.ingest import UNDRAFTED_ADP, adp_from_players, load_cached, load_game_logs
from ninecat.nbastats import LOG_COLUMNS


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


def test_load_cached_attaches_this_season(tmp_path):
    pd.DataFrame({"player_id": [1], "name": ["Nikola Jokić"], "team_id": [2], "PTS": [28.0]}).to_csv(
        tmp_path / ingest.PLAYERS_FILE, index=False)
    pd.DataFrame({"name_key": ["nikola jokic"], "S_G": [4], "S_PTS": [31.5], "age": [31.0]}).to_csv(
        tmp_path / ingest.SEASON_FILE, index=False)
    df, _ = load_cached(tmp_path)
    assert df.loc[0, "S_PTS"] == 31.5 and df.loc[0, "age"] == 31.0
    assert ingest.load_table(ingest.SCHEDULE_FILE, tmp_path).empty


def test_load_cached_without_cache(tmp_path):
    assert load_cached(tmp_path) is None


def test_old_minutes_only_logs_are_ignored(tmp_path):
    pd.DataFrame({"name_key": ["a"], "date": ["2026-01-01"], "min": [30.0], "season": ["2025-26"]}).to_csv(
        tmp_path / ingest.GAME_LOGS_FILE, index=False)
    assert load_game_logs(tmp_path).empty


# ---------------------------------------------------------------- ESPN players

def espn_player(**stats):
    return SimpleNamespace(playerId=7, name="Some Guy", proTeam="DEN", position="PG", eligibleSlots=["PG", "G", "UT"],
                           injuryStatus="OUT", injured=True, expected_return_date=date(2026, 11, 2), stats=stats)


def split(avg, gp, gs=None):
    total = {"GP": gp, **({"GS": gs} if gs is not None else {})}
    return {"avg": avg, "total": total}


def test_player_row_keeps_projection_and_stores_splits_and_injury():
    p = espn_player(**{"2027_projected": split({"PTS": 20.0, "MIN": 30.0}, 70),
                       "2027_total": split({"PTS": 25.0, "MIN": 33.0, "TO": 2.0}, 4, 4),
                       "2027_last_7": split({"PTS": 27.0}, 3)})
    row = ingest.player_row(p, 2027, "FA", None)
    assert (row["PTS"], row["GP"], row["stat_source"]) == (20.0, 70.0, "projected")
    assert (row["T_PTS"], row["T_MIN"], row["T_TO"], row["T_GP"], row["T_GS"]) == (25.0, 33.0, 2.0, 4.0, 4.0)
    assert row["R7_PTS"] == 27.0 and row["R7_GP"] == 3.0 and row["R7_REB"] is None
    assert row["R15_PTS"] is None and row["R30_GP"] is None
    assert (row["injury_status"], row["injured"], row["return_date"]) == ("OUT", True, "2026-11-02")


# ---------------------------------------------------------------- transactions

def topic(ms, *messages):
    keys = ("id", "messageTypeId", "targetId", "to", "from", "for")
    return {"date": ms, "messages": [dict(zip(keys, m)) for m in messages]}


TOPICS = [
    topic(1792700000000, ("m1", 178, 11, 1, None, None), ("m2", 179, 12, 1, None, None)),
    topic(1792600000000, ("m3", 188, 11, 12, 1, 1)),                 # lineup move: skipped
    topic(1792500000000, ("m4", 239, 13, -1, 12, 6)),                # drop: team in "for"
    topic(1792400000000, ("m5", 244, 14, 2, 1, None), ("m6", 244, 15, 1, 2, None)),
    topic(1792300000000, ("m7", 180, 16, 8, 0, 1), ("m8", 181, 17, 8, None, None)),
]
NAMES = {11: "GG Jackson", 12: "Peyton Watson", 14: "Jakob Poeltl"}


def test_transactions_frame_normalizes_moves():
    tx = ingest.transactions_frame(TOPICS, NAMES).set_index("msg_id")
    assert list(tx.reset_index().columns) == ingest.TRANSACTION_COLUMNS
    assert "m3" not in tx.index and len(tx) == 7
    assert tx.loc["m1", ["team_id", "action", "player"]].tolist() == [1, "FA add", "GG Jackson"]
    assert tx.loc["m2", "action"] == "Drop" and tx.loc["m4", "team_id"] == 6
    assert tx.loc["m5", ["team_id", "from_team_id", "action"]].tolist() == [2, 1, "Trade"]
    assert tx.loc["m7", "action"] == "Waiver add" and pd.isna(tx.loc["m7", "from_team_id"])
    assert tx.loc["m8", "action"] == "Drop" and tx.loc["m8", "player"] == ""
    assert tx.loc["m1", "date"].startswith("2026-10-")


def test_merge_transactions_dedupes_overlapping_refreshes():
    first = ingest.transactions_frame(TOPICS[2:], NAMES)
    second = ingest.transactions_frame(TOPICS[:4], NAMES)   # overlaps m4-m6
    merged = ingest.merge_transactions(first, second)
    assert sorted(merged["msg_id"]) == ["m1", "m2", "m4", "m5", "m6", "m7", "m8"]
    assert merged["date"].is_monotonic_decreasing
    assert ingest.merge_transactions(merged, second).equals(merged)
    assert ingest.merge_transactions(None, ingest.transactions_frame([], {})).empty


def test_transactions_append_through_csv(tmp_path):
    path = tmp_path / ingest.TRANSACTIONS_FILE
    ingest.write_csv(ingest.transactions_frame(TOPICS[3:], NAMES), path)
    old = pd.read_csv(path, dtype={"msg_id": str})
    merged = ingest.merge_transactions(old, ingest.transactions_frame(TOPICS, NAMES))
    assert len(merged) == 7 and merged["team_id"].dtype == "Int64"


class FakeLeague:
    """Serves raw activity in pages of ACTIVITY_PAGE topics, newest first."""

    def __init__(self, topics):
        self.topics, self.offsets = topics, []
        self.espn_request = SimpleNamespace(league_get=self._get)

    def _get(self, extend="", params=None, headers=None):
        off = json.loads(headers["x-fantasy-filter"])["topics"]["offset"]
        self.offsets.append(off)
        return {"topics": self.topics[off:off + ingest.ACTIVITY_PAGE]}


def many_topics(n):
    return [topic(1792700000000 - i, (f"m{i}", 178, i, 1, None, None)) for i in range(n)]


def test_fetch_activity_backfills_then_stops_at_known_moves():
    lg = FakeLeague(many_topics(250))
    assert len(ingest.fetch_activity(lg)) == 250 and lg.offsets == [0, 100, 200]
    lg = FakeLeague(many_topics(250))
    ingest.fetch_activity(lg, known_ids=[f"m{i}" for i in range(250)])
    assert lg.offsets == [0]


# ---------------------------------------------------------------- source status

def test_merge_sources_keeps_last_success_and_upgrades_old_meta():
    old = {"espn": "ok", "bbr_bpm": "failed: x", "game_logs_2025-26": "ok"}
    up = ingest.merge_sources(old, {}, "2026-10-23T10:00:00+00:00", "2026-10-19T05:00:00+00:00")
    assert up == {"espn": {"status": "ok", "updated": "2026-10-19T05:00:00+00:00"},
                  "bbr_bpm": {"status": "failed: x", "updated": None}}
    now = "2026-10-23T10:00:00+00:00"
    m = ingest.merge_sources(up, {"espn": "failed: auth", "bbr_bpm": "ok", "nba_season": "ok"}, now)
    assert m["espn"] == {"status": "failed: auth", "updated": "2026-10-19T05:00:00+00:00"}
    assert m["bbr_bpm"]["updated"] == now and m["nba_season"]["updated"] == now


def test_is_stale_after_a_week_or_never():
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    src = {"a": {"status": "ok", "updated": "2026-10-20T00:00:00+00:00"},
           "b": {"status": "ok", "updated": "2026-10-10T00:00:00+00:00"},
           "c": {"status": "failed", "updated": None}}
    assert not ingest.is_stale(src, "a", now)
    assert ingest.is_stale(src, "b", now) and ingest.is_stale(src, "c", now) and ingest.is_stale(src, "d", now)


def test_source_rows_flag_saved_copies():
    meta = {"sources": {"espn": {"status": "ok", "updated": "t1"},
                        "bbr_bpm": {"status": "failed: slow", "updated": "t0"},
                        "nba_ages": {"status": "failed: 500", "updated": None},
                        "nba_season": {"status": "not available yet (no games yet)", "updated": None}}}
    states = {r["key"]: r["state"] for r in ingest.source_rows(meta)}
    assert states == {"espn": "ok", "nba_season": "waiting", "nba_ages": "missing", "bbr_bpm": "saved"}


# ---------------------------------------------------------------- snapshots, matchups, schedule

def test_snapshot_path_is_one_file_per_day(tmp_path):
    assert ingest.snapshot_path(tmp_path, date(2026, 10, 23)) == tmp_path / "snapshots" / "espn_proj_2026-10-23.csv"


def test_projection_snapshot_keeps_projected_players(players):
    players = players.copy()
    players.loc[0, "stat_source"] = "total"
    snap = ingest.projection_snapshot(players)
    assert len(snap) == len(players) - 1 and {"player_id", "PTS", "GP", "MIN"} <= set(snap.columns)


def test_live_matchup_frame_one_row_per_side():
    sides = [{"team_id": 1, "opponent_id": 2, "wins": 5, "losses": 3, "ties": 0,
              "stats": {"PTS": {"value": 289.0, "result": "WIN"}, "FG%": {"value": 0.5636}}},
             {"team_id": 2, "opponent_id": 1, "stats": {"PTS": {"score": 264.0}}}]
    out = ingest.live_matchup_frame(sides, 3)
    assert out[["period", "team_id", "opponent_id", "PTS"]].values.tolist() == [[3, 1, 2, 289.0], [3, 2, 1, 264.0]]
    assert out.loc[0, "FG%"] == 0.5636 and out.loc[0, "wins"] == 5


def test_matchup_opponents_both_ways():
    raw = [{"matchupPeriodId": 1, "home": {"teamId": 1}, "away": {"teamId": 2}},
           {"matchupPeriodId": 20, "home": {"teamId": 3}}]               # bye or not set yet
    assert ingest.matchup_opponents(raw) == {(1, 1): 2, (1, 2): 1}


PRO = {11: {"1": [{"id": 9, "date": 1792539000000, "homeProTeamId": 3, "awayProTeamId": 11}],
            "2": [{"id": 10, "date": 1792724400000, "homeProTeamId": 11, "awayProTeamId": 0}]},
       3: {"1": [{"id": 9, "date": 1792539000000, "homeProTeamId": 3, "awayProTeamId": 11}]}}


def test_espn_pro_schedule_as_fallback_schedule():
    out = ingest.espn_schedule_frame(PRO, {3: "DAL", 11: "IND", 0: "FA"})
    assert out.to_dict("records") == [{"date": "2026-10-20", "game_id": "9", "home": "DAL", "away": "IND"}]
    assert ingest.opening_day(PRO) == date(2026, 10, 20)


def test_combine_logs_replaces_refreshed_seasons_only():
    def logs(season, n):
        return pd.DataFrame({c: [0] * n for c in LOG_COLUMNS}).assign(season=season)

    out = ingest.combine_logs(pd.concat([logs("2025-26", 3), logs("2026-27", 1)]), {"2026-27": logs("2026-27", 2)})
    assert out["season"].value_counts().to_dict() == {"2025-26": 3, "2026-27": 2}
    legacy = pd.DataFrame({"name_key": ["a"], "date": ["2026-01-01"], "min": [30.0], "season": ["2025-26"]})
    assert len(ingest.combine_logs(legacy, {"2026-27": logs("2026-27", 2)})) == 2


def test_write_csv_is_atomic_and_gzips(tmp_path):
    path = tmp_path / "history" / "logs_2019-20.csv.gz"
    ingest.write_csv(pd.DataFrame({"a": [1, 2]}), path)
    assert pd.read_csv(path)["a"].tolist() == [1, 2]
    assert not list(path.parent.glob("*.tmp")) and path.read_bytes()[:2] == bytes([0x1F, 0x8B])


def test_ingest_history_skips_cached_seasons(tmp_path, monkeypatch):
    from ninecat import nbastats

    fetched = []

    def fake(season):
        fetched.append(season)
        if season == "2019-20":
            raise TimeoutError("slow")
        return pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})

    monkeypatch.setattr(nbastats, "fetch_history_season", fake)
    for p in ingest.history_paths(tmp_path, "2017-18"):
        ingest.write_csv(pd.DataFrame({"z": [0]}), p)
    status = ingest.ingest_history(2027, tmp_path, log=lambda *_: None)
    assert list(status) == [f"{y}-{str(y + 1)[2:]}" for y in range(2017, 2026)]
    assert status["2017-18"] == "cached" and "2017-18" not in fetched
    assert status["2019-20"].startswith("failed") and status["2020-21"].startswith("ok")
    assert ingest.history_paths(tmp_path, "2025-26")[0].exists()
