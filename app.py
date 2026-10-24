"""Streamlit UI: player board, player pages, my team, waiver wire.

Run with: streamlit run app.py
Renders from the cache written by `python -m ninecat.ingest`. When the
server starts (AUTO_REFRESH=true), or on "Refresh now", ingest runs in a
background thread and the page offers a reload once new data is ready;
rendering never waits on ESPN or nba.com.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from ninecat.board import ADVANCED_METRICS, build_board, player_key
from ninecat.config import DEFAULT_POOL_SIZE, SEASON_GAMES, categories, load_settings
from ninecat.ingest import FREE_AGENT, load_cached, load_game_logs, load_meta, source_rows
from ninecat.lines import SOURCES
from ninecat.nbastats import season_label
from ninecat.theme import app_css, tokens
from ninecat.league import build_league
from ninecat.urlstate import PARAMS as P
from ninecat.urlstate import PLAYER, TABS, player_href
from ui import league_views, player_page
from ui.components import (
    advanced_table,
    category_table,
    html,
    show_table,
    theme_mode,
)
from ui.refresh import boot_refresh, refresh_job

st.set_page_config(page_title="Nine-Cat Value Board", layout="wide", initial_sidebar_state="collapsed")

MODE = theme_mode()
TOKENS = tokens(MODE)
html(f"<style>{app_css(MODE)}\n[class*='st-key-card_'] {{ background: {TOKENS['surface']}; "
     f"border-radius: 12px; }}</style>")


@st.cache_data(show_spinner=False)
def cached_data():
    return load_cached()


@st.cache_data(show_spinner=False)
def cached_logs():
    return load_game_logs()


@st.cache_data(show_spinner="Valuing players...")
def cached_board(players, cats, punt, source, games_discount, pool_size, season_games, pace):
    return build_board(players, cats, punt=punt, source=source, games_discount=games_discount,
                       pool_size=pool_size, season_games=season_games, pace=dict(pace))


@st.cache_data(show_spinner="Simulating every player on every team...")
def cached_league(_board, key, cats, teams, season_games):
    """The league model for one board; `key` stands in for hashing the board."""
    return build_league(_board, cats, teams, season_games=season_games)


def when(iso: str | None) -> str:
    if not iso:
        return "never"
    t = datetime.fromisoformat(iso).astimezone()
    return f"{t:%a %b} {t.day}, {t:%I:%M %p}".replace(" 0", " ")


def refresh_now() -> None:
    JOB.start()
    st.rerun()


def reload_data() -> None:
    st.cache_data.clear()
    st.rerun()


def status_line(meta: dict, polling: bool) -> None:
    """Data age, the background refresh, and sources running on saved copies.

    While a refresh runs this is a fragment polling every 2 seconds; when it
    ends, one full rerun stops the polling and shows the reload button.
    """
    if polling and not JOB.running:
        st.rerun()
    bits = []
    if meta.get("fetched_at"):
        bits.append(f"Data updated {when(meta['fetched_at'])}")
    if JOB.running:
        bits.append('<span class="updating">Updating data…</span>')
    elif JOB.error:
        bits.append(f'<span class="neg" title="{escape(JOB.error)}">Refresh failed (hover for details)</span>')
    old = [r for r in source_rows(meta) if r["state"] in ("saved", "missing")]
    if old:
        detail = escape("; ".join(f"{r['label']}: {r['status']}" for r in old))
        bits.append(f'<span class="neg" title="{detail}">{len(old)} source{"s" if len(old) > 1 else ""} '
                    'on a saved copy or unavailable (hover for details)</span>')
    html(f'<div class="status">{" · ".join(bits)}</div>')
    if JOB.succeeded and JOB.runs and load_meta().get("fetched_at") != meta.get("fetched_at"):
        if st.button("New data ready, reload", type="primary"):
            reload_data()


def sources_popover(meta: dict) -> None:
    with st.popover("Sources", width="stretch"):
        rows = source_rows(meta)
        if not rows:
            st.caption("No refresh recorded yet.")
        marks = {"ok": "✓", "waiting": "…", "saved": "!", "missing": "✕"}
        for r in rows:
            note = {"ok": "", "waiting": " (nothing to fetch yet)",
                    "saved": f" (latest refresh failed; using the copy from {when(r['updated'])})",
                    "missing": " (unavailable)"}[r["state"]]
            st.markdown(f"{marks[r['state']]} **{r['label']}**: updated {when(r['updated'])}{note}",
                        help=r["status"] if r["state"] != "ok" else None)
        if meta.get("calendar_note"):
            st.caption(f"Matchup calendar: {meta['calendar_note']}.")


def open_player(pid) -> None:
    """Open a player's page in this tab; a new history entry, so browser Back returns."""
    if pid is None:
        return
    st.query_params[PLAYER] = str(pid)
    st.session_state.epoch += 1
    st.rerun()


def close_player() -> None:
    st.query_params.pop(PLAYER, None)
    st.session_state.epoch += 1
    st.rerun()


def links(view: pd.DataFrame) -> dict[str, list]:
    return {"Player": list(player_key(view))}


# Bumped whenever the page changes, so tables come back with nothing selected
# (a stale selection would reopen the player after Back).
st.session_state.setdefault("epoch", 0)

# ---------------------------------------------------------------- model settings
try:
    settings = load_settings()
except ValueError:  # no league configured yet: nothing to refresh from
    settings = None
exclude_to_default = settings.exclude_to if settings else True

JOB = refresh_job()
boot_refresh(JOB, enabled=bool(settings and settings.auto_refresh))

URL = {"bind": "query-params"}   # every setting lives in the URL (see ninecat/urlstate.py)

with st.sidebar:
    st.subheader("Model")
    exclude_to = st.toggle("Exclude turnovers (8-cat)", value=exclude_to_default,
                           help="Set the default with EXCLUDE_TO in .env", key=P["exclude_to"], **URL)
    pool_size = st.number_input(
        "Replacement pool size", min_value=20, max_value=400, value=DEFAULT_POOL_SIZE, step=10,
        help="Players who get rostered league-wide (teams x roster spots). "
             "Z-scores are computed against this pool.", key=P["pool_size"], **URL)
    season_games = st.number_input("Season games", min_value=1, max_value=82, value=SEASON_GAMES,
                                   key=P["season_games"], **URL)

cats = categories(exclude_to)

# ---------------------------------------------------------------- data
cache = cached_data()
if cache is None:
    st.title("Nine-Cat Value Board")
    if JOB.running:
        @st.fragment(run_every=2)
        def wait_for_first_refresh():
            if not JOB.running:
                reload_data()
            st.info("Pulling your league from ESPN and stats from nba.com for the first time. "
                    "This takes a minute; the page loads when it's done.")
        wait_for_first_refresh()
    else:
        if JOB.error:
            st.error(f"Refresh failed: {JOB.error}")
        st.info("No cached league data yet. Fill in `.env` (see README), then run "
                "`python -m ninecat.ingest` or press **Refresh now**.")
        if st.button("Refresh now"):
            refresh_now()
    st.stop()

players, meta = cache
teams = {int(k): v for k, v in meta.get("teams", {}).items()}
if not teams:
    named = players.dropna(subset=["team_id"]).groupby("team_id")["fantasy_team"].first()
    teams = {int(t): n for t, n in named.items()}
abbrevs = {teams[int(k)]: v for k, v in meta.get("team_abbrevs", {}).items() if int(k) in teams}
year = int(meta.get("year") or datetime.now().year)
seasons = {"current": season_label(year), "last": season_label(year, 1), "prev": season_label(year, 2)}

# ---------------------------------------------------------------- header
scoring = "scored without turnovers" if exclude_to else "9-cat with turnovers"
html(f"""<div class="eyebrow">ESPN · H2H · {len(teams)} teams · {scoring}</div>
<h1>Nine-Cat Value Board</h1>
<p class="sub">Every rostered player and the top free agents in your league, valued on per-game z-scores
and checked against market draft position. Click any player's name for his full page.</p>""")

status, pop, button = st.columns([4.4, 0.8, 1], vertical_alignment="center")
with status:
    polling = JOB.running
    st.fragment(run_every=2 if polling else None)(status_line)(meta, polling)
with pop:
    sources_popover(meta)
with button:
    if st.button("Refresh now", width="stretch", disabled=JOB.running,
                 help="Pull ESPN and nba.com again in the background"):
        refresh_now()

# ---------------------------------------------------------------- controls
team_names = sorted(teams.values())
my_default = teams.get(meta.get("my_team_id"))
# The URL stores these labels, so they stay the same from season to season.
source_labels = {"blend": "Blend", "proj": "ESPN proj", "last": "Last season", "season": "This season"}
source_help = (f"Blend: 60% ESPN {seasons['current']} projection, 40% {seasons['last']} actual. "
               f"ESPN proj: the projection alone. Last season: {seasons['last']} per game. "
               f"This season: {seasons['current']} per game so far (3+ games, else the blend).")
with st.container(border=True, key="card_controls"):
    c1, c2, c3 = st.columns([0.9, 1.75, 0.85])
    my_team = c1.selectbox("My team", team_names, placeholder="Pick your team…",
                           index=team_names.index(my_default) if my_default in team_names else None,
                           key=P["team"], **URL)
    source = c2.segmented_control("Stats source", list(SOURCES), format_func=source_labels.get,
                                  default="blend", required=True, key=P["source"], help=source_help, **URL)
    with c3:
        st.markdown("")
        games_discount = st.toggle("Discount for games missed", value=True, key=P["discount"], **URL)
    punt = st.pills("Punt categories", cats, selection_mode="multi", key=P["punt"], **URL) or []
    punt = [c for c in punt if c in cats]

board_args = (tuple(cats), tuple(punt), source, games_discount, int(pool_size), int(season_games),
              tuple(sorted(meta.get("pace", {}).items())))
board = cached_board(players, *board_args)
league_key = (meta.get("fetched_at"), board_args)
league = cached_league(board, league_key, tuple(cats), tuple(team_names), int(season_games))
board = board.assign(impact=league.impact, impact_rank=league.impact_rank)
ctx = {"cats": cats, "punt": punt, "source": source, "my_team": my_team, "abbrevs": abbrevs,
       "seasons": seasons, "pace": meta.get("pace", {}), "mode": MODE, "tokens": TOKENS,
       "n_teams": len(teams), "epoch": st.session_state.epoch, "league": league, "league_key": league_key}

# ---------------------------------------------------------------- player page
pid = st.query_params.get(PLAYER)
if pid:
    bar_l, bar_r = st.columns([5, 1], vertical_alignment="center")
    here = player_href(pid, {k: st.query_params.get_all(k) for k in st.query_params})
    bar_l.html(f'<span class="eyebrow">Player page · <a href="{escape(here)}" target="_blank" rel="noopener">'
               'open in a new tab</a></span>')
    if bar_r.button(f"← Back to {st.query_params.get(P['tab'], TABS[0])}", width="stretch"):
        close_player()
    ctx["logs"] = cached_logs()
    player_page.render(board, players, pid, ctx, open_player)
    st.stop()

# ---------------------------------------------------------------- tabs
tab_players, tab_team, tab_fa, tab_cats = st.tabs(TABS, key=P["tab"], **URL)

with tab_players:
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 0.9, 1.1], vertical_alignment="bottom")
    search = f1.text_input("Search", placeholder="Search players", label_visibility="collapsed",
                           key=P["search"], **URL)
    owner = f2.selectbox("Owner", ["All players", "Free agents", "Rostered", *team_names],
                         label_visibility="collapsed", key=P["owner"], **URL)
    position = f3.selectbox("Position", ["All positions", "PG", "SG", "SF", "PF", "C"],
                            label_visibility="collapsed", key=P["position"], **URL)
    cols = f4.segmented_control("Columns", ["Categories", "Advanced"], default="Categories", required=True,
                                label_visibility="collapsed", key=P["columns"], **URL)

    view = board
    if search:
        view = view[view["name"].str.contains(search, case=False, na=False, regex=False)]
    if owner == "Free agents":
        view = view[view["fantasy_team"] == FREE_AGENT]
    elif owner == "Rostered":
        view = view[view["fantasy_team"] != FREE_AGENT]
    elif owner in team_names:
        view = view[view["fantasy_team"] == owner]
    if position != "All positions":
        view = view[view["eligible"].fillna("").str.split("/").apply(lambda s: position in s)]

    if view.empty:
        st.info("No players match those filters.")
    elif cols == "Advanced":
        styler, config = advanced_table(view, my_team, abbrevs, MODE)
        open_player(show_table(styler, config, key=f"tbl_adv_{ctx['epoch']}", links=links(view), height=640))
        html('<p class="note">Shading compares each metric with the top of the board: green is better, red is '
             'worse (lower TOV% counts as better). Usage and minutes drive counting stats, so a per-36 rank 25+ '
             'spots better than the value rank (green) means production waiting on a bigger role. Hover a column '
             'header for its definition; click a name for the player page.</p>')
    else:
        styler, config = category_table(view, cats, punt, my_team, abbrevs, MODE)
        open_player(show_table(styler, config, key=f"tbl_cats_{ctx['epoch']}", links=links(view), height=640))
        to_note = "Turnovers are excluded, so every number uses the other eight categories. " if exclude_to else ""
        html(f'<p class="note">{to_note}Value is the sum of category z-scores against the top {pool_size} players. '
             'Category cells show the per-game number, shaded by its z-score. FG% and FT% are weighted by shot '
             'attempts. Gap is market rank (by ESPN ADP) minus value rank: positive means the market undervalues '
             'him. Click a column header to sort; click a name for the player page.</p>')

with tab_team:
    league_views.render_my_team(board, league, ctx, open_player)

with tab_fa:
    league_views.render_waivers(board, league, ctx, open_player)

with tab_cats:
    league_views.render_team_categories(board, league, ctx)

# ---------------------------------------------------------------- footer
adv_names = ", ".join(m.label for m in ADVANCED_METRICS)
html(f"""<div class="note" style="max-width:90ch;margin-top:28px;display:grid;gap:4px;font-size:12px">
<div>Sources: ESPN {seasons['current']} projections, rosters and average draft position from your league ·
nba.com {seasons['current']} game logs and {seasons['last']} and {seasons['prev']} stats via nba_api ·
<a href="https://www.basketball-reference.com/" target="_blank" rel="noopener">Basketball Reference</a> for BPM.</div>
<div>Advanced metrics ({adv_names}) are {seasons['last']} regular season, computed from nba.com box-score totals
with Basketball Reference's formulas; players with no minutes last season show a dash. Per-36 and per-100 values
re-run the z-scores with every stat scaled to 36 minutes or 100 team possessions, for players with 15+ minutes.</div>
<div>Blend = 60% ESPN projection, 40% last season (when he played 15+ games). Rookies and players who missed
{seasons['last']} use projections only. This season = {seasons['current']} per-game averages from nba.com game
logs (ESPN's season-to-date averages where the names don't match); players with fewer than 3 games use the
blend. Games played for the games discount stay ESPN's projection.</div></div>""")
