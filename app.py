"""Streamlit UI: player board, player pages, my team, waiver wire.

Run with: streamlit run app.py
Reads the cache written by `python -m ninecat.ingest`; never calls ESPN or
nba.com unless you press "Refresh now".
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from ninecat.board import ADVANCED_METRICS, build_board, player_key
from ninecat.config import DEFAULT_POOL_SIZE, SEASON_GAMES, categories, load_settings
from ninecat.ingest import FREE_AGENT, ingest, load_cached, load_game_logs
from ninecat.lines import SOURCES
from ninecat.nbastats import NOT_YET, season_label
from ninecat.theme import app_css, tokens
from ninecat.league import build_league
from ui import league_views, player_page
from ui.components import (
    advanced_table,
    category_table,
    html,
    show_table,
    theme_mode,
)

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


def refresh() -> None:
    try:
        with st.spinner("Pulling ESPN, nba.com and Basketball Reference..."):
            ingest()
    except Exception as exc:  # surface ESPN/auth errors in the UI
        st.error(f"Refresh failed: {exc}")
        return
    st.cache_data.clear()
    st.rerun()


def open_player(pid) -> None:
    st.query_params["player"] = str(pid)
    st.session_state.epoch += 1
    st.rerun()


def close_player() -> None:
    st.query_params.pop("player", None)
    st.session_state.epoch += 1
    st.rerun()


def pick_player(view: pd.DataFrame, pos) -> None:
    if pos is not None:
        open_player(player_key(view).iloc[pos])


# Bumped whenever the page changes, so tables come back with no row selected.
st.session_state.setdefault("epoch", 0)

# ---------------------------------------------------------------- model settings
try:
    exclude_to_default = load_settings().exclude_to
except ValueError:
    exclude_to_default = True

with st.sidebar:
    st.subheader("Model")
    exclude_to = st.toggle("Exclude turnovers (8-cat)", value=exclude_to_default,
                           help="Set the default with EXCLUDE_TO in .env")
    pool_size = st.number_input(
        "Replacement pool size", min_value=20, max_value=400, value=DEFAULT_POOL_SIZE, step=10,
        help="Players who get rostered league-wide (teams x roster spots). "
             "Z-scores are computed against this pool.")
    season_games = st.number_input("Season games", min_value=1, max_value=82, value=SEASON_GAMES)

cats = categories(exclude_to)

# ---------------------------------------------------------------- data
cache = cached_data()
if cache is None:
    st.title("Nine-Cat Value Board")
    st.info("No cached league data yet. Fill in `.env` (see README), then run "
            "`python -m ninecat.ingest` or press **Refresh now**.")
    if st.button("Refresh now"):
        refresh()
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
and checked against market draft position. Select any player for his full page.</p>""")

status, button = st.columns([5, 1], vertical_alignment="center")
with status:
    bits = []
    if meta.get("fetched_at"):
        when = datetime.fromisoformat(meta["fetched_at"]).astimezone()
        bits.append(f"Data updated {when:%a %b} {when.day}, {when:%I:%M %p}".replace(" 0", " "))
    failed = {k: v for k, v in meta.get("sources", {}).items() if not str(v).startswith(("ok", NOT_YET))}
    if failed:
        detail = escape("; ".join(f"{k}: {v}" for k, v in failed.items()))
        bits.append(f'<span class="neg" title="{detail}">{len(failed)} source'
                    f'{"s" if len(failed) > 1 else ""} unavailable (hover for details)</span>')
    html(f'<div class="status">{" · ".join(bits)}</div>')
with button:
    if st.button("Refresh now", width="stretch"):
        refresh()

# ---------------------------------------------------------------- controls
team_names = sorted(teams.values())
my_default = teams.get(meta.get("my_team_id"))
source_labels = {"blend": "Blend", "proj": f"ESPN {seasons['current'][2:]} proj",
                 "last": f"{seasons['last']} actual"}
with st.container(border=True, key="card_controls"):
    c1, c2, c3 = st.columns([1, 1.25, 1.15])
    my_team = c1.selectbox("My team", team_names, placeholder="Pick your team…",
                           index=team_names.index(my_default) if my_default in team_names else None)
    source = c2.segmented_control("Stats source", list(SOURCES), format_func=source_labels.get,
                                  default="blend", required=True, key="source")
    with c3:
        st.markdown("")
        games_discount = st.toggle("Discount for games missed", value=True)
    punt = st.pills("Punt categories", cats, selection_mode="multi", key=f"punt_{len(cats)}") or []

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
pid = st.query_params.get("player")
if pid:
    bar_l, bar_r = st.columns([5, 1], vertical_alignment="center")
    bar_l.html('<span class="eyebrow">Player page</span>')
    if bar_r.button("← Back to board", width="stretch"):
        close_player()
    ctx["logs"] = cached_logs()
    player_page.render(board, players, pid, ctx, open_player)
    st.stop()

# ---------------------------------------------------------------- tabs
tab_players, tab_team, tab_fa, tab_cats = st.tabs(["Players", "My team", "Waiver wire", "Team categories"])

with tab_players:
    f1, f2, f3, f4 = st.columns([1.6, 1.2, 0.9, 1.1], vertical_alignment="bottom")
    search = f1.text_input("Search", placeholder="Search players", label_visibility="collapsed")
    owner = f2.selectbox("Owner", ["All players", "Free agents", "Rostered", *team_names],
                         label_visibility="collapsed")
    position = f3.selectbox("Position", ["All positions", "PG", "SG", "SF", "PF", "C"],
                            label_visibility="collapsed")
    cols = f4.segmented_control("Columns", ["Categories", "Advanced"], default="Categories", required=True,
                                label_visibility="collapsed", key="cols")

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
        pick_player(view, show_table(styler, config, key=f"tbl_adv_{ctx['epoch']}", height=640))
        html('<p class="note">Shading compares each metric with the top of the board: green is better, red is '
             'worse (lower TOV% counts as better). Usage and minutes drive counting stats, so a per-36 rank 25+ '
             'spots better than the value rank (green) means production waiting on a bigger role. Hover a column '
             'header for its definition; select a row for the player page.</p>')
    else:
        styler, config = category_table(view, cats, punt, my_team, abbrevs, MODE)
        pick_player(view, show_table(styler, config, key=f"tbl_cats_{ctx['epoch']}", height=640))
        to_note = "Turnovers are excluded, so every number uses the other eight categories. " if exclude_to else ""
        html(f'<p class="note">{to_note}Value is the sum of category z-scores against the top {pool_size} players. '
             'Category cells show the per-game number, shaded by its z-score. FG% and FT% are weighted by shot '
             'attempts. Gap is market rank (by ESPN ADP) minus value rank: positive means the market undervalues '
             'him. Click a column header to sort; select a row for the player page.</p>')

with tab_team:
    league_views.render_my_team(board, league, ctx, pick_player)

with tab_fa:
    league_views.render_waivers(board, league, ctx, pick_player)

with tab_cats:
    league_views.render_team_categories(board, league, ctx)

# ---------------------------------------------------------------- footer
adv_names = ", ".join(m.label for m in ADVANCED_METRICS)
html(f"""<div class="note" style="max-width:90ch;margin-top:28px;display:grid;gap:4px;font-size:12px">
<div>Sources: ESPN {seasons['current']} projections, rosters and average draft position from your league ·
nba.com {seasons['last']} and {seasons['prev']} stats via nba_api ·
<a href="https://www.basketball-reference.com/" target="_blank" rel="noopener">Basketball Reference</a> for BPM.</div>
<div>Advanced metrics ({adv_names}) are {seasons['last']} regular season, computed from nba.com box-score totals
with Basketball Reference's formulas; players with no minutes last season show a dash. Per-36 and per-100 values
re-run the z-scores with every stat scaled to 36 minutes or 100 team possessions, for players with 15+ minutes.</div>
<div>Blend = 60% ESPN projection, 40% last season (when he played 15+ games). Rookies and players who missed
{seasons['last']} use projections only.</div></div>""")
