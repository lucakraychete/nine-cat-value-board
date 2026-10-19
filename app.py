"""Streamlit UI: player board, my team, waiver wire.

Run with: streamlit run app.py
Reads the cache written by `python -m ninecat.ingest`; never calls ESPN
unless you press "Refresh from ESPN".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ninecat.config import DEFAULT_POOL_SIZE, SEASON_GAMES, categories, load_settings
from ninecat.ingest import FREE_AGENT, ingest, load_cached
from ninecat.valuation import team_profile, value_players, z_col

st.set_page_config(page_title="Nine-Cat Value Board", layout="wide")

INFO_COLS = ["rank", "name", "pro_team", "eligible", "fantasy_team", "injury_status", "GP"]


@st.cache_data(show_spinner=False)
def cached_data():
    return load_cached()


@st.cache_data(show_spinner="Valuing players...")
def valued_players(
    df: pd.DataFrame,
    cats: tuple[str, ...],
    punt: tuple[str, ...],
    pool_size: int,
    season_games: int,
    games_discount: bool,
) -> pd.DataFrame:
    return value_players(
        df,
        cats,
        punt=punt,
        pool_size=pool_size,
        season_games=season_games,
        games_discount=games_discount,
    )


def color_z(value) -> str:
    """Green for positive z-scores, red for negative, stronger as |z| grows."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return ""
    alpha = min(abs(value) / 2.5, 1.0) * 0.55
    rgb = "34,139,34" if value > 0 else "200,40,40"
    return f"background-color: rgba({rgb},{alpha:.2f})"


def show_board(df: pd.DataFrame, cats: list[str], height: int = 600) -> None:
    """Sortable table: click any column header to sort."""
    z_cols = [z_col(c) for c in cats]
    cols = [c for c in INFO_COLS if c in df.columns] + ["value", "z_total", *z_cols]
    table = df[cols].rename(columns={z_col(c): c for c in cats})
    styled = table.style.map(color_z, subset=[*cats, "value", "z_total"]).format(
        {**{c: "{:+.2f}" for c in [*cats, "value", "z_total"]}, "GP": "{:.0f}"}
    )
    st.dataframe(styled, hide_index=True, width="stretch", height=height)


def refresh() -> None:
    try:
        with st.spinner("Pulling rosters, free agents and projections from ESPN..."):
            ingest()
    except Exception as exc:  # surface ESPN/auth errors in the UI
        st.sidebar.error(f"Refresh failed: {exc}")
        return
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------- sidebar
try:
    exclude_to_default = load_settings().exclude_to
except ValueError:
    exclude_to_default = True

st.sidebar.title("Nine-Cat Value Board")
if st.sidebar.button("Refresh from ESPN", width="stretch"):
    refresh()

cache = cached_data()
if cache is None:
    st.title("Nine-Cat Value Board")
    st.info(
        "No cached league data yet. Fill in `.env` (see README), then run "
        "`python -m ninecat.ingest` or press **Refresh from ESPN** in the sidebar."
    )
    st.stop()

players, meta = cache
if meta.get("fetched_at"):
    st.sidebar.caption(f"Data fetched {meta['fetched_at'][:16].replace('T', ' ')} UTC")

exclude_to = st.sidebar.toggle(
    "Exclude turnovers (8-cat)", value=exclude_to_default, help="Set default via EXCLUDE_TO in .env"
)
cats = categories(exclude_to)

st.sidebar.subheader("Punt categories")
punt = [c for c in cats if st.sidebar.checkbox(f"Punt {c}", key=f"punt_{c}")]

st.sidebar.subheader("Model")
pool_size = st.sidebar.number_input(
    "Replacement pool size",
    min_value=20,
    max_value=400,
    value=DEFAULT_POOL_SIZE,
    step=10,
    help="Players who get rostered league-wide (teams x roster spots). "
    "Z-scores are computed against this pool.",
)
games_discount = st.sidebar.toggle("Discount for games missed", value=True)
season_games = st.sidebar.number_input(
    "Season games", min_value=1, max_value=82, value=SEASON_GAMES, disabled=not games_discount
)

valued = valued_players(
    players, tuple(cats), tuple(punt), int(pool_size), int(season_games), games_discount
)
scored_cats = [c for c in cats if c not in punt]

teams = {int(k): v for k, v in meta.get("teams", {}).items()}
if not teams:
    teams = {
        int(t): n
        for t, n in valued.dropna(subset=["team_id"])
        .groupby("team_id")["fantasy_team"]
        .first()
        .items()
    }

# ---------------------------------------------------------------- main
title = "Nine-Cat Value Board" if not exclude_to else "Eight-Cat Value Board (no TO)"
st.title(title)
if punt:
    st.caption(f"Punting {', '.join(punt)}: excluded from value and hidden from tables.")

tab_board, tab_team, tab_waiver = st.tabs(["Player board", "My team", "Waiver wire"])

with tab_board:
    c1, c2, c3 = st.columns([2, 1, 1])
    search = c1.text_input("Search player", placeholder="Name...")
    status = c2.selectbox("Show", ["All", "Rostered", "Free agents"])
    position = c3.selectbox("Position", ["Any", "PG", "SG", "SF", "PF", "C"])

    board = valued
    if search:
        board = board[board["name"].str.contains(search, case=False, na=False)]
    if status == "Rostered":
        board = board[board["fantasy_team"] != FREE_AGENT]
    elif status == "Free agents":
        board = board[board["fantasy_team"] == FREE_AGENT]
    if position != "Any":
        board = board[board["eligible"].fillna("").str.split("/").apply(lambda s: position in s)]
    show_board(board, scored_cats)

with tab_team:
    if not teams:
        st.warning("No fantasy teams in the cached data.")
    else:
        team_ids = sorted(teams, key=lambda t: teams[t])
        my_id = meta.get("my_team_id")
        default = team_ids.index(my_id) if my_id in team_ids else 0
        team_id = st.selectbox(
            "Team", team_ids, index=default, format_func=lambda t: teams[t]
        )
        roster = valued[valued["team_id"] == team_id]

        rostered = valued[valued["fantasy_team"] != FREE_AGENT]
        profile = team_profile(rostered, scored_cats)
        team_name = teams[team_id]
        if team_name in profile.index:
            st.subheader("Category profile")
            st.caption(
                "Sum of per-game z-scores across the roster. Rank is out of "
                f"{len(profile)} teams (1 = best)."
            )
            ranks = profile.rank(ascending=False, method="min").astype(int)
            summary = pd.DataFrame(
                {"z sum": profile.loc[team_name], "league rank": ranks.loc[team_name]}
            ).T
            st.dataframe(
                summary.style.format("{:+.2f}", subset=pd.IndexSlice["z sum", :]).format(
                    "{:.0f}", subset=pd.IndexSlice["league rank", :]
                ),
                width="stretch",
            )

        st.subheader("Roster")
        show_board(roster, scored_cats, height=500)

with tab_waiver:
    fas = valued[valued["fantasy_team"] == FREE_AGENT]

    my_id = meta.get("my_team_id")
    profile = team_profile(valued[valued["fantasy_team"] != FREE_AGENT], scored_cats)
    if my_id in teams and teams[my_id] in profile.index:
        weakest = profile.loc[teams[my_id], scored_cats].nsmallest(3)
        st.caption(f"Your weakest categories: {', '.join(weakest.index)}")

    c1, c2 = st.columns([1, 1])
    sort_by = c1.selectbox("Sort by", ["value", *scored_cats])
    healthy_only = c2.checkbox("Hide injured (OUT)", value=True)
    if healthy_only:
        fas = fas[fas["injury_status"].fillna("ACTIVE") != "OUT"]

    key = "value" if sort_by == "value" else z_col(sort_by)
    show_board(fas.sort_values(key, ascending=False), scored_cats)
