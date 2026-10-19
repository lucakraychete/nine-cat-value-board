"""Small HTML pieces and the dense, z-shaded player tables."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from ninecat.board import ADVANCED_METRICS
from ninecat.player import pct3
from ninecat.theme import heat, rgba, tokens
from ninecat.valuation import z_col

FREE_AGENT = "FA"
DASH = "–"


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


def tags(row: pd.Series) -> str:
    t = ""
    inj = row.get("injury_status")
    if inj == "OUT":
        t += '<span class="tag out">OUT</span>'
    elif inj == "DAY_TO_DAY":
        t += '<span class="tag dtd">DTD</span>'
    if not (row.get("L_G", 0) or 0) > 0:
        t += '<span class="tag new" title="No NBA stats last season: rookie or missed the season">PROJ ONLY</span>'
    return t


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
    css["Player"] = "font-weight: 500"

    formats = {"Value": "{:+.2f}", "Min": "{:.1f}", "GP": "{:.0f}", "ADP": "{:.1f}", "Gap": "{:+d}"}
    for c in cats:
        formats[labels[c]] = fmt_pct if c in ("FG%", "FT%") else "{:.1f}"
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
    css["Player"] = "font-weight: 500"
    css["P36 Rk"] = np.where(frame["Rk"] - frame["P36 Rk"] >= 25, good, "")
    css["P100 Rk"] = np.where(frame["Rk"] - frame["P100 Rk"] >= 25, good, "")
    css["ΔMin"] = np.where(frame["ΔMin"] >= 2, good, np.where(frame["ΔMin"] <= -2, bad, ""))
    for m in ADVANCED_METRICS:
        css[m.label] = [heat(z, mode) for z in board[f"az_{m.key}"].values]

    formats = {"Value": "{:+.2f}", "P36 Rk": "{:.0f}", "P100 Rk": "{:.0f}", "Min": "{:.1f}", "ΔMin": "{:+.1f}",
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
        "P36 Rk": st.column_config.NumberColumn(width=62, help="Value rank if every player got 36 minutes: talent independent of role"),
        "P100 Rk": st.column_config.NumberColumn(width=66, help="Value rank per 100 team possessions: also adjusts for pace"),
        "Min": st.column_config.NumberColumn(width=50, help="Minutes per game on the selected line"),
        "ΔMin": st.column_config.NumberColumn(width=56, help="ESPN projected minutes minus last season's"),
    }
    for m in ADVANCED_METRICS:
        config[m.label] = st.column_config.NumberColumn(width=58, help=m.help)
    return styler, config


def show_table(styler, config, key: str, height: int | str = 560):
    """Dense sortable table; returns the selected row position, if any."""
    event = st.dataframe(styler, column_config=config, hide_index=True, height=height, row_height=30,
                         on_select="rerun", selection_mode="single-row", key=key, placeholder=DASH)
    rows = event.selection.rows if event is not None else []
    return rows[0] if rows else None
