"""Small HTML pieces and the dense, z-shaded player tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from ninecat.board import ADVANCED_METRICS
from ninecat.player import pct3
from ninecat.theme import heat, rgba, tokens
from ninecat.urlstate import link_target
from ninecat.valuation import z_col

FREE_AGENT = "FA"
DASH = "–"
ROW_HEIGHT = 30
LINK_HELP = "Click a name to open his page"


def theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if theme is not None and getattr(theme, "type", None) == "dark" else "light"


def html(body: str) -> None:
    st.html(body)


def card(key: str):
    """A bordered surface panel, like the original's .ppcard."""
    return st.container(border=True, key=f"card_{key}")


def card_title(title: str, note: str = "") -> None:
    note_html = f'<p class="note">{note}</p>' if note else ""
    html(f'<div class="card-title">{escape(title)}</div>{note_html}')


def h4(text: str) -> None:
    html(f'<div class="card-h4">{escape(text)}</div>')


def legend(items: list[tuple[str, str]]) -> None:
    html('<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{escape(label)}</span>' for c, label in items) + "</div>")


INJURY_TAGS = {"OUT": ("out", "OUT"), "DAY_TO_DAY": ("dtd", "DTD")}


def tags(row: pd.Series) -> str:
    t = ""
    inj = INJURY_TAGS.get(row.get("injury_status"))
    if inj:
        t += f'<span class="tag {inj[0]}">{inj[1]}</span>'
    if not (row.get("L_G", 0) or 0) > 0:
        t += '<span class="tag new" title="No NBA stats last season: rookie or missed the season">PROJ ONLY</span>'
    return t


def name_with_tag(row: pd.Series) -> str:
    """His name with the injury tag in plain text, for table cells."""
    inj = INJURY_TAGS.get(row.get("injury_status"))
    return f"{row['name']} · {inj[1]}" if inj else str(row["name"])


def fmt_num(x, digits: int = 1) -> str:
    return DASH if x is None or pd.isna(x) else f"{x:.{digits}f}"


def fmt_signed(x, digits: int = 2) -> str:
    return DASH if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def fmt_pct(x) -> str:
    return DASH if x is None or pd.isna(x) else pct3(x)


def fmt_rank(x) -> str:
    return DASH if x is None or pd.isna(x) else f"{int(x)}"


def owner_labels(board: pd.DataFrame, abbrevs: dict[str, str]) -> pd.Series:
    return board["fantasy_team"].map(lambda t: t if t == FREE_AGENT else abbrevs.get(t, t))


# ---------------------------------------------------------------- tables

IMPACT_COLUMN = st.column_config.NumberColumn(
    width=62, help="Extra categories won over the season if the average team in your league added him "
                   "(dropping its lowest-value player at his position). Counts real head-to-head wins, "
                   "so punts don't change it.")


def _add_impact(frame: pd.DataFrame, board: pd.DataFrame) -> None:
    if "impact" in board:
        frame.insert(frame.columns.get_loc("Value") + 1, "Impact", board["impact"].values)


def _impact_css(css: pd.DataFrame, frame: pd.DataFrame, t: dict) -> None:
    if "Impact" in frame:
        css["Impact"] = np.where(frame["Impact"] >= 0, f"color: {t['good']}; font-weight: 600",
                                 f"color: {t['bad']}; font-weight: 600")


def delta_frame(cats: list[str], rows: list[np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(np.vstack(rows) if rows else np.zeros((0, len(cats))), columns=cats)


def delta_css(deltas: pd.DataFrame, mode: str, scale: float = 1.5) -> pd.DataFrame:
    """Shade per-category win changes like z-scores (a season gain of 1.7 wins saturates)."""
    return deltas.apply(lambda col: [heat(v * scale, mode) for v in col])


def show_plain(styler, config=None, height: int | str = "auto", key: str | None = None):
    """A dense table without row selection."""
    height = fit_height(styler, height)
    st.dataframe(styler, column_config=config, hide_index=True, height=height, row_height=ROW_HEIGHT,
                 placeholder=DASH, key=key)

def _style(frame: pd.DataFrame, css: pd.DataFrame, formats: dict, mine: pd.Series, mode: str):
    """Styler with per-cell CSS and the user's own players tinted."""
    t = tokens(mode)
    tint = f"background-color: {rgba(t['accent'], 0.10)}"
    css = css.copy()
    for col in css.columns:
        css[col] = np.where(mine.values & (css[col] == ""), tint, css[col])
    return frame.style.apply(lambda _: css, axis=None).format(formats, na_rep=DASH)


def category_table(board: pd.DataFrame, cats: list[str], punt: list[str], my_team: str | None,
                   abbrevs: dict[str, str], mode: str, extra: dict[str, pd.Series] | None = None):
    """Rank, player, owner, value, each category's raw number shaded by its z-score,
    minutes, games, ADP and market gap. Returns (styler, column_config)."""
    labels = {c: f"{c} (punt)" if c in punt else c for c in cats}
    frame = pd.DataFrame({
        "Rk": board["rank"].values,
        "Player": board["name"].values,
        "Pos": board["eligible"].fillna(board.get("position", "")).values,
        "NBA": board["pro_team"].values,
        "Tier": board["fantasy_tier"].values,
        "Owner": owner_labels(board, abbrevs).values,
        "Value": board["value"].values,
    })
    _add_impact(frame, board)
    for c in cats:
        frame[labels[c]] = board[c].values
    frame["Min"] = board["MIN"].values
    frame["GP"] = board["GP"].values
    frame["ADP"] = pd.to_numeric(board.get("ADP"), errors="coerce").values
    frame["Gap"] = board["gap"].values
    for name, series in (extra or {}).items():
        frame[name] = series.values

    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for c in cats:
        css[labels[c]] = [heat(z, mode) for z in board[z_col(c)].values]
    t = tokens(mode)
    css["Gap"] = np.where(frame["Gap"] >= 10, f"color: {t['good']}; font-weight: 600",
                          np.where(frame["Gap"] <= -10, f"color: {t['bad']}; font-weight: 600", ""))
    css["Value"] = "font-weight: 600"
    css["Player"] = link_style(t)
    _impact_css(css, frame, t)

    formats = {"Value": "{:+.2f}", "Impact": "{:+.1f}", "Min": "{:.1f}", "GP": "{:.0f}", "ADP": "{:.1f}",
               "Gap": "{:+d}"}
    for c in cats:
        formats[labels[c]] = fmt_pct if c in ("FG%", "FT%") else "{:.1f}"
    for name, series in (extra or {}).items():
        if pd.api.types.is_numeric_dtype(series):
            formats[name] = "{:+.1f}"
            css[name] = np.where(series.values > 0.05, f"color: {t['good']}; font-weight: 600",
                                 np.where(series.values < -0.05, f"color: {t['bad']}; font-weight: 600", ""))
    mine = board["fantasy_team"].eq(my_team) if my_team else pd.Series(False, index=board.index)
    styler = _style(frame, css, formats, mine, mode)

    config = {
        "Rk": st.column_config.NumberColumn(width=44, pinned=True),
        "Player": st.column_config.TextColumn(width="medium", pinned=True),
        "Pos": st.column_config.TextColumn(width=62),
        "NBA": st.column_config.TextColumn(width=48),
        "Tier": st.column_config.TextColumn(width=74),
        "Owner": st.column_config.TextColumn(width=62),
        "Value": st.column_config.NumberColumn(width=62, help="Sum of category z-scores among the top of the pool"),
        "Impact": IMPACT_COLUMN,
        "ADP": st.column_config.NumberColumn(width=54, help="ESPN average draft position (blank = undrafted)"),
        "Gap": st.column_config.NumberColumn(width=54, help="Market rank by ADP minus value rank: positive = undervalued"),
        "Min": st.column_config.NumberColumn(width=50),
        "GP": st.column_config.NumberColumn(width=44, help="Games played (projected, or last season's)"),
    }
    for c in cats:
        config[labels[c]] = st.column_config.NumberColumn(
            width=58, help=f"{c} per game, shaded by z-score" + (" (punted: not in Value)" if c in punt else ""))
    return styler, config


def advanced_table(board: pd.DataFrame, my_team: str | None, abbrevs: dict[str, str], mode: str):
    """Value, per-36 and per-100 ranks, minutes trend and advanced rates shaded vs the pool."""
    frame = pd.DataFrame({
        "Rk": board["rank"].values,
        "Player": board["name"].values,
        "Pos": board["eligible"].fillna("").values,
        "NBA": board["pro_team"].values,
        "Owner": owner_labels(board, abbrevs).values,
        "Value": board["value"].values,
    })
    _add_impact(frame, board)
    frame = frame.assign(**{
        "P36 Rk": board["p36_rank"].astype("float").values,
        "P100 Rk": board["p100_rank"].astype("float").values,
        "Min": board["MIN"].values,
        "ΔMin": board["d_min"].values,
    })
    for m in ADVANCED_METRICS:
        frame[m.label] = pd.to_numeric(board.get(f"adv_{m.key}"), errors="coerce").values

    t = tokens(mode)
    good, bad = f"color: {t['good']}; font-weight: 600", f"color: {t['bad']}; font-weight: 600"
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css["Value"] = "font-weight: 600"
    css["Player"] = link_style(t)
    _impact_css(css, frame, t)
    css["P36 Rk"] = np.where(frame["Rk"] - frame["P36 Rk"] >= 25, good, "")
    css["P100 Rk"] = np.where(frame["Rk"] - frame["P100 Rk"] >= 25, good, "")
    css["ΔMin"] = np.where(frame["ΔMin"] >= 2, good, np.where(frame["ΔMin"] <= -2, bad, ""))
    for m in ADVANCED_METRICS:
        css[m.label] = [heat(z, mode) for z in board[f"az_{m.key}"].values]

    formats = {"Value": "{:+.2f}", "Impact": "{:+.1f}", "P36 Rk": "{:.0f}", "P100 Rk": "{:.0f}", "Min": "{:.1f}", "ΔMin": "{:+.1f}",
               "TS%": fmt_pct}
    for m in ADVANCED_METRICS:
        formats.setdefault(m.label, "{:.1f}")
    mine = board["fantasy_team"].eq(my_team) if my_team else pd.Series(False, index=board.index)
    styler = _style(frame, css, formats, mine, mode)

    config = {
        "Rk": st.column_config.NumberColumn(width=44, pinned=True),
        "Player": st.column_config.TextColumn(width="medium", pinned=True),
        "Pos": st.column_config.TextColumn(width=62),
        "NBA": st.column_config.TextColumn(width=48),
        "Owner": st.column_config.TextColumn(width=62),
        "Value": st.column_config.NumberColumn(width=62),
        "Impact": IMPACT_COLUMN,
        "P36 Rk": st.column_config.NumberColumn(width=62, help="Value rank if every player got 36 minutes: talent independent of role"),
        "P100 Rk": st.column_config.NumberColumn(width=66, help="Value rank per 100 team possessions: also adjusts for pace"),
        "Min": st.column_config.NumberColumn(width=50, help="Minutes per game on the selected line"),
        "ΔMin": st.column_config.NumberColumn(width=56, help="ESPN projected minutes minus last season's"),
    }
    for m in ADVANCED_METRICS:
        config[m.label] = st.column_config.NumberColumn(width=58, help=m.help)
    return styler, config


def _base_frame(board: pd.DataFrame, abbrevs: dict[str, str], tagged: bool = False) -> pd.DataFrame:
    return pd.DataFrame({
        "Player": [name_with_tag(r) for _, r in board.iterrows()] if tagged else board["name"].values,
        "Pos": board["eligible"].fillna(board.get("position", "")).values,
        "NBA": board["pro_team"].values,
        "Owner": owner_labels(board, abbrevs).values,
    })


def _base_config() -> dict:
    return {"Player": st.column_config.TextColumn(width="medium", pinned=True),
            "Pos": st.column_config.TextColumn(width=62),
            "NBA": st.column_config.TextColumn(width=48),
            "Owner": st.column_config.TextColumn(width=62)}


def season_minutes(board: pd.DataFrame) -> pd.Series:
    """Minutes per game this season: nba.com's, else ESPN's."""
    nba = pd.to_numeric(board.get("S_MIN"), errors="coerce")
    espn = pd.to_numeric(board.get("T_MIN"), errors="coerce")
    if nba is None:
        return espn if espn is not None else pd.Series(np.nan, index=board.index)
    return nba.where(nba.notna(), espn) if espn is not None else nba


def overview_table(board: pd.DataFrame, ov: pd.DataFrame, cats: list[str], punt: list[str], my_team: str | None,
                   abbrevs: dict[str, str], mode: str, min_l14: pd.Series, pre_day: str, ros_label: str = "interim"):
    """Pre, so far and ROS category wins added, with ranks, and the so-far per category.

    `ov` is overview() for these board rows; min_l14 is aligned to them too.
    ros_label is "model" or "interim"; with the model, an 80% range column
    follows ROS Rk. Returns (styler, column_config).
    """
    labels = {c: f"{c} (punt)" if c in punt else c for c in cats}
    frame = _base_frame(board, abbrevs, tagged=True)
    frame["GP"] = ov["gp"].values
    frame["Min"] = season_minutes(board).values
    frame["Min L14"] = min_l14.values
    for col, src in (("So far", "so_far"), ("So far Rk", "so_far_rank"), ("Pre", "pre"), ("Pre Rk", "pre_rank"),
                     ("Rk Δ", "rank_delta"), ("ROS", "ros"), ("ROS Rk", "ros_rank")):
        frame[col] = pd.to_numeric(ov[src], errors="coerce").astype(float).values
    ranged = ros_label == "model" and "ros_lo" in ov and ov["ros_lo"].notna().any()
    if ranged:
        frame["ROS 80%"] = [f"{lo:+.1f} to {hi:+.1f}" if pd.notna(lo) and pd.notna(hi) else DASH
                            for lo, hi in zip(ov["ros_lo"], ov["ros_hi"])]
    for c in cats:
        frame[labels[c]] = ov[f"sf_{c}"].values

    t = tokens(mode)
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css["Player"] = link_style(t)
    css["So far"] = "font-weight: 700"
    # Shade per-category so far against the board's bigger contributions, so
    # the colors mean something on day 5 as well as day 150.
    scored = [c for c in cats if c not in punt]
    vals = np.abs(ov[[f"sf_{c}" for c in scored]].to_numpy(float)) if scored else np.zeros(1)
    top = float(np.nanquantile(vals, 0.95)) if np.isfinite(vals).any() else 0.0
    scale = 2.5 / top if top > 0 else 1.0
    for c in cats:
        if c in punt:
            css[labels[c]] = f"color: {t['muted']}"
        else:
            css[labels[c]] = [heat(v * scale, mode) for v in ov[f"sf_{c}"].values]

    formats = {"GP": "{:.0f}", "Min": "{:.1f}", "Min L14": "{:.1f}", "So far": "{:+.2f}", "So far Rk": "{:.0f}",
               "Pre": "{:+.1f}", "Pre Rk": "{:.0f}", "Rk Δ": "{:+.0f}", "ROS": "{:+.1f}", "ROS Rk": "{:.0f}",
               **{labels[c]: "{:+.1f}" for c in cats}}
    mine = board["fantasy_team"].eq(my_team) if my_team else pd.Series(False, index=board.index)
    styler = _style(frame, css, formats, mine, mode)

    as_of = f" (ESPN's projection as of {pre_day})" if pre_day else ""
    config = {
        **_base_config(),
        "GP": st.column_config.NumberColumn(width=44, help="Games he has played this season"),
        "Min": st.column_config.NumberColumn(width=50, help="Minutes per game this season"),
        "Min L14": st.column_config.NumberColumn(
            width=62, help="Minutes per game over the last 14 days, from nba.com game logs (blank: no games)"),
        "So far": st.column_config.NumberColumn(
            width=64, help="Category wins added so far = his full-season league impact at this season's per-game "
                           "rates (no games discount) × games played ÷ season games"),
        "So far Rk": st.column_config.NumberColumn(width=70, help="Rank by So far among every player on the board"),
        "Pre": st.column_config.NumberColumn(
            width=58, help=f"Category wins added over a full season on the preseason ESPN projection{as_of}"),
        "Pre Rk": st.column_config.NumberColumn(width=58, help="Rank by Pre among every player on the board"),
        "Rk Δ": st.column_config.NumberColumn(width=52, help="Pre Rk minus So far Rk"),
        "ROS": st.column_config.NumberColumn(
            label="ROS (model)", width=86,
            help="ROS (model): category wins added over his team's remaining games on the trained model's "
                 "rest-of-season line and his projected share of those games (injuries included)")
        if ros_label == "model" else st.column_config.NumberColumn(
            label="ROS (interim)", width=86,
            help="ROS (interim): category wins added over his team's remaining games on a blend of this season "
                 "and the preseason projection, weighted GP ÷ (GP + 20); the trained model replaces it"),
        "ROS Rk": st.column_config.NumberColumn(width=62, help="Rank by ROS among every player on the board"),
    }
    if ranged:
        config["ROS 80%"] = st.column_config.TextColumn(
            width=104, help="80% range for ROS: the model's backtest error range for his per-game value at his "
                            "games played and minutes, converted to category wins added")
    for c in cats:
        config[labels[c]] = st.column_config.NumberColumn(
            width=62, help=f"{c} category wins added so far" + (" (punted: not in the totals)" if c in punt else ""))
    return styler, config


def schedule_table(board: pd.DataFrame, info: pd.DataFrame, my_team: str | None, abbrevs: dict[str, str],
                   mode: str, labels: dict[str, str]):
    """Games ahead, recent role and the last 10 games' value for these board rows.

    `info` is aligned to board and holds: week, next, playoffs, started,
    min_l7, min_l14, rostered, spark. labels carry the week date ranges.
    """
    frame = _base_frame(board, abbrevs)
    frame["Status"] = board.get("injury_status", pd.Series("", index=board.index)).fillna("ACTIVE").map(
        {"ACTIVE": "Active", "OUT": "Out", "DAY_TO_DAY": "Day to day", "SUSPENSION": "Suspended"}).fillna(
        board.get("injury_status", pd.Series("", index=board.index)).astype(str).str.title()).values
    frame["This wk"] = info["week"].values
    frame["Next wk"] = info["next"].values
    frame["Playoffs"] = info["playoffs"].values
    frame["Started"] = info["started"].values
    frame["Min"] = season_minutes(board).values
    frame["Min L7"] = info["min_l7"].values
    frame["Min L14"] = info["min_l14"].values
    frame["% Rost"] = info["rostered"].values
    frame["Last 10"] = info["spark"].values

    t = tokens(mode)
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css["Player"] = link_style(t)
    css["Status"] = np.where(frame["Status"] == "Out", f"color: {t['bad']}; font-weight: 600", "")
    formats = {"This wk": "{:.0f}", "Next wk": "{:.0f}", "Playoffs": "{:.0f}", "Min": "{:.1f}", "Min L7": "{:.1f}",
               "Min L14": "{:.1f}", "% Rost": "{:.0f}%"}
    mine = board["fantasy_team"].eq(my_team) if my_team else pd.Series(False, index=board.index)
    styler = _style(frame, css, formats, mine, mode)
    config = {
        **_base_config(),
        "Status": st.column_config.TextColumn(width=82, help="ESPN injury status"),
        "This wk": st.column_config.NumberColumn(width=60, help=f"His NBA team's games {labels['week']}"),
        "Next wk": st.column_config.NumberColumn(width=62, help=f"His NBA team's games {labels['next']}"),
        "Playoffs": st.column_config.NumberColumn(
            width=64, help=f"His NBA team's games in your fantasy playoffs ({labels['playoffs']})"),
        "Started": st.column_config.TextColumn(
            width=64, help="Games started of games played this season, from ESPN's season totals "
                           "(nba.com's game logs don't record starts)"),
        "Min": st.column_config.NumberColumn(width=50, help="Minutes per game this season"),
        "Min L7": st.column_config.NumberColumn(width=56, help="Minutes per game over the last 7 days (nba.com)"),
        "Min L14": st.column_config.NumberColumn(width=62, help="Minutes per game over the last 14 days (nba.com)"),
        "% Rost": st.column_config.NumberColumn(width=60, help="Share of ESPN leagues that roster him"),
        "Last 10": st.column_config.LineChartColumn(
            width="medium", help="Value of each of his last 10 games: the sum of category z-scores against "
                                 "the board's pool, oldest to newest"),
    }
    return styler, config


def fit_height(styler, height: int | str, max_height: int = 680) -> int | str:
    """Streamlit's "auto" stops at about 10 rows; size short tables to show every row."""
    if height != "auto":
        return height
    return min(max_height, ROW_HEIGHT * (len(styler.data) + 1) + 3)


def show_table(styler, config, key: str, links: Mapping[str, Sequence] | None = None,
               height: int | str = 560):
    """Dense sortable table whose name cells are links.

    links maps each name column to the player ids in row order; clicking
    one of those cells returns its player id (None otherwise). Sorting by
    a column header doesn't change which id a row maps to.
    """
    height = fit_height(styler, height)
    config = dict(config)
    for col in links or {}:
        own = dict(config.get(col) or {})
        config[col] = {**own, "help": own.get("help") or LINK_HELP}
    event = st.dataframe(styler, column_config=config, hide_index=True, height=height, row_height=ROW_HEIGHT,
                         on_select="rerun", selection_mode="single-cell", key=key, placeholder=DASH)
    cells = event.selection.cells if event is not None else []
    return link_target(cells, links or {})


def link_style(t: dict) -> str:
    """Cell CSS for a clickable player name."""
    return f"color: {t['accent']}; font-weight: 600"


def name_button(name: str, pid, key: str, on_open) -> None:
    """A player's name as a link-styled button that opens his page."""
    if st.button(name, key=key, type="tertiary", help="Open his page"):
        on_open(pid)
