"""Trades: the calculator, the finder and "Who to trade, and to whom" on the
Trades tab, a compact surplus table on My team, and "Trades for [player]"
on player pages.

Every number comes from the rest-of-season trade model (ninecat/trades.py),
whatever stats source the board shows. "Shop him" and "Open in calculator"
can't set the calculator's widgets once they've rendered, so they leave a
request in session state and rerun; apply_pending() fills the widgets in
before anything renders.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from ninecat.league import FREE_AGENT
from ninecat.trades import (
    FINDER_MODES,
    FINDER_POOL,
    MAX_SIDE,
    MIN_GAIN,
    TradeModel,
    board_ids,
    evaluate,
    find_trades,
    surpluses,
    verdict,
)
from ninecat.urlstate import PARAMS as P
from ninecat.urlstate import PLAYER
from ui.components import DASH, card, card_title, delta_css, delta_frame, h4, html, link_style, name_button, show_table
from ui.league_views import fit_columns, pos_neg

URL = {"bind": "query-params"}
PENDING = "pending_trade"
TRADES_TAB = "Trades"
VALUED_ON = {"model": "Valued on: rest of season (model)", "interim": "Valued on: rest of season (interim)"}
DEFAULT_VETO = 4


# ---------------------------------------------------------------- opening the calculator

def request_trade(partner: str, give: list[str], get: list[str]) -> None:
    """Open the calculator on the Trades tab with these players (by name) ticked."""
    st.session_state[PENDING] = {"partner": partner, "give": give, "get": get}
    st.session_state.epoch += 1
    st.rerun()


def apply_pending() -> None:
    """Run before any widget renders: fill the calculator from a request."""
    req = st.session_state.pop(PENDING, None)
    if not req:
        return
    st.session_state[P["tab"]] = TRADES_TAB
    st.session_state[P["trade_partner"]] = req["partner"]
    st.session_state[P["trade_give"]] = req["give"]
    st.session_state[P["trade_get"]] = req["get"]
    st.query_params.pop(PLAYER, None)


def _clear_get() -> None:
    st.session_state[P["trade_get"]] = []


def _cancel() -> None:
    st.session_state[P["trade_give"]] = []
    st.session_state[P["trade_get"]] = []


def _on_pick(pick, open_player, deals=None) -> None:
    """A table click: a player id opens his page; ("trade", i) opens deal i in the calculator."""
    if isinstance(pick, tuple) and deals is not None:
        d = deals[pick[1]]
        names = deals.names
        request_trade(d.partner, [names[x] for x in d.give], [names[x] for x in d.get])
    elif pick is not None:
        open_player(pick)


class _Deals(list):
    """Finder deals plus the names to fill the calculator with."""
    def __init__(self, deals, names):
        super().__init__(deals)
        self.names = names


@st.cache_data(show_spinner="Searching trades...")
def cached_find(_model: TradeModel, key: tuple, me: str, mode: str, cat: str | None, punt: tuple,
                veto: float, partner: str | None, two_for_two: bool, must_get: int | None, limit: int):
    """find_trades for one model; `key` identifies the data and model settings."""
    return find_trades(_model, me, mode=mode, cat=cat, punt=punt, veto_cap=veto,
                       partners=[partner] if partner else None, two_for_two=two_for_two,
                       must_get=must_get, limit=limit)


# ---------------------------------------------------------------- shared pieces

def _weeks_note(model: TradeModel) -> str:
    return f"over the {model.lg.season_weeks:.1f} weeks left"


def _deal_frame(model: TradeModel, deals, ctx: dict, with_partner: bool) -> tuple:
    """Finder deals as a table, with links for names and for "Open"."""
    lg, cats, t = model.lg, ctx["cats"], ctx["tokens"]

    def slot(xs, i):
        return lg.names[xs[i]] if i < len(xs) else ""

    deltas = delta_frame(cats, [d.result.me.by_cat for d in deals])
    frame = pd.DataFrame({
        **({"Partner": [d.partner for d in deals]} if with_partner else {}),
        "Give": [slot(d.give, 0) for d in deals], "Give 2": [slot(d.give, 1) for d in deals],
        "Get": [slot(d.get, 0) for d in deals], "Get 2": [slot(d.get, 1) for d in deals],
        "You": [d.result.me.cats for d in deals], "Them": [d.result.them.cats for d in deals],
        "Gap": [d.result.gap for d in deals], "Win": [d.result.me.win * 100 for d in deals],
        "They see": [verdict(d.look)[0] for d in deals],
    })
    frame = pd.concat([frame, deltas], axis=1)
    frame["Open"] = "Open in calculator"
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css[cats] = delta_css(deltas, ctx["mode"])
    for c in ("Give", "Give 2", "Get", "Get 2", "Open"):
        css[c] = link_style(t)
    css["You"] = f"color: {t['good']}; font-weight: 700"
    css["Them"] = [f"color: {t['good'] if v > 0.005 else t['bad'] if v < -0.005 else t['ink']}" for v in frame["Them"]]
    css["They see"] = [f"color: {t['good']}; font-weight: 600" if verdict(d.look)[1] == "pos" else "" for d in deals]
    fmt = {"You": "{:+.1f}", "Them": "{:+.1f}", "Gap": "{:+.1f}", "Win": "{:+.1f} pts", **{c: "{:+.1f}" for c in cats}}
    styler = frame.style.apply(lambda _: css, axis=None).format(fmt)
    weeks = _weeks_note(model)
    config = {
        **({"Partner": st.column_config.TextColumn(width="medium", pinned=True)} if with_partner else {}),
        "You": st.column_config.NumberColumn(width=56, help=f"Category wins you gain {weeks}"),
        "Them": st.column_config.NumberColumn(width=60, help=f"Category wins they gain (or lose) {weeks}"),
        "Gap": st.column_config.NumberColumn(width=56, help="Your gain minus theirs"),
        "Win": st.column_config.NumberColumn(width=78, help="Change in your weekly matchup win chance"),
        "They see": st.column_config.TextColumn(
            width=90, help="How the offer likely reads to them: name, headline stats, upgrade at the position, "
                           "and valuing what they give up more than what they get"),
        "Open": st.column_config.TextColumn(width="medium", help="Open this deal in the calculator"),
        **{c: st.column_config.NumberColumn(width=52, help=f"Your change in {c} wins {weeks}") for c in cats},
    }
    links = {}
    for col, side, i in (("Give", "give", 0), ("Give 2", "give", 1), ("Get", "get", 0), ("Get 2", "get", 1)):
        links[col] = board_ids(model, [getattr(d, side)[i] if i < len(getattr(d, side)) else None for d in deals])
    links["Open"] = [("trade", i) for i in range(len(deals))]
    return styler, config, links


# ---------------------------------------------------------------- the Trades tab

def render_trades(model: TradeModel, ctx: dict, open_player) -> None:
    me, lg = ctx["my_team"], model.lg
    if not me or me not in lg.rosters:
        st.info("Pick your team at the top to use the trade calculator and finder.")
        return
    phase = model.perception.phase
    phase_txt = ("Preseason: managers are judging on draft position and projections." if phase == 0 else
                 f"Early season ({phase:.0%} of the way to stats-driven): name value is fading as games pile up."
                 if phase < 1 else "In season: this season's stats drive how managers see players.")
    label = ctx.get("ros_label", "interim")
    line_txt = ("the trained model's projection" if label == "model" else
                "this season blended with the preseason projection until a model is trained")
    html(f'<div class="callout"><span class="eyebrow">{VALUED_ON[label]}</span><b>Trades for {escape(me)}</b>'
         f'<span class="note">Every deal is simulated for both teams {_weeks_note(model)}, each player on his '
         f'rest-of-season line ({line_txt}) '
         'and his NBA team\'s remaining games, whatever stats source the board shows. Roster sizes are kept: the '
         'side getting more players drops its lowest-value player at the position, the side getting fewer picks up '
         f'the best free agent at the position it lost. {phase_txt}</span></div>')
    _calculator(model, ctx, open_player)
    _finder(model, ctx, open_player)
    with card("surplus"):
        card_title("Who to trade, and to whom", "Each of your players' worth to you (vs the best free agent at his "
                   "position) against what he'd really add for every other team, rest of season. A big surplus "
                   "means he's a better fit somewhere else. \"Shop him\" opens the calculator with that team.")
        surplus_table(model, ctx, open_player, n=14, key="tr_surplus")


def _calculator(model: TradeModel, ctx: dict, open_player) -> None:
    me, lg, pv = ctx["my_team"], model.lg, model.perception
    partners = [t for t in lg.teams if t != me]
    with card("calc"):
        card_title("Trade calculator", f"Pick a partner and up to {MAX_SIDE} players on each side. Your players are "
                   "sorted by how much this manager would want them, theirs by how much they'd help you.")
        if st.session_state.get(P["trade_partner"]) not in partners:
            st.session_state.pop(P["trade_partner"], None)
        b = st.selectbox("Trade with", partners, key=P["trade_partner"], on_change=_clear_get, **URL)
        j = lg.teams.index(me)
        mine = sorted(lg.rosters[me], key=lambda x: -pv.pv(b, x))
        theirs = sorted(lg.rosters[b], key=lambda x: -lg.worth[x, j])
        by_name = {lg.names[x]: int(x) for x in [*mine, *theirs]}
        for key, side in ((P["trade_give"], mine), (P["trade_get"], theirs)):
            valid = {lg.names[x] for x in side}
            if key in st.session_state:
                st.session_state[key] = [n for n in st.session_state[key] if n in valid][:MAX_SIDE]
        left, right = st.columns(2)
        with left:
            give_names = st.multiselect("You give", [lg.names[x] for x in mine], max_selections=MAX_SIDE,
                                        placeholder="Your players", key=P["trade_give"], **URL)
            _pick_table(model, mine, me, b, "give", give_names, ctx, open_player)
        with right:
            get_names = st.multiselect("You get", [lg.names[x] for x in theirs], max_selections=MAX_SIDE,
                                       placeholder="Their players", key=P["trade_get"], **URL)
            _pick_table(model, theirs, me, b, "get", get_names, ctx, open_player)
        if not give_names or not get_names:
            html('<p class="note">Pick at least one player on each side to see the result.</p>')
            return
        st.button("Clear the trade", on_click=_cancel, key="tc_cancel")
        give, get = [by_name[n] for n in give_names], [by_name[n] for n in get_names]
        _result(model, ctx, open_player, b, give, get)


def _pick_table(model: TradeModel, rows, me: str, b: str, side: str, selected: list[str], ctx: dict,
                open_player) -> None:
    lg, t = model.lg, ctx["tokens"]
    j, k = lg.teams.index(me), lg.teams.index(b)
    give = side == "give"
    frame = pd.DataFrame({
        "Player": [lg.names[x] for x in rows],
        "Pos": [model.board["eligible"].iloc[x] for x in rows],
        "Worth to you" if give else "For you": [lg.worth[x, j] for x in rows],
        "For them" if give else "Worth to them": [lg.worth[x, k] for x in rows],
        "ROS value": [lg.stats.value[x] for x in rows],
    })
    picked = frame["Player"].isin(selected).to_numpy()
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css["Player"] = link_style(t)
    for c in frame.columns[2:4]:
        css[c] = [f"color: {t['good']}" if v > 0.005 else f"color: {t['bad']}" if v < -0.005 else "" for v in frame[c]]
    tint = f"background-color: {t['accent']}22"
    css = css.apply(lambda col: [f"{v}; {tint}" if p else v for v, p in zip(col, picked)])
    weeks = _weeks_note(model)
    help_you = (f"Category wins {weeks} you'd lose replacing him with the best free agent at his position" if give
                else f"Category wins {weeks} you'd gain with him, dropping your lowest-value player at his position")
    help_them = (f"Category wins {weeks} he'd really add for them" if give
                 else f"Category wins {weeks} they'd lose replacing him with the best free agent at his position")
    config = {"Player": st.column_config.TextColumn(width="medium"),
              frame.columns[2]: st.column_config.NumberColumn(width=78, help=help_you),
              frame.columns[3]: st.column_config.NumberColumn(width=78, help=help_them),
              "ROS value": st.column_config.NumberColumn(width=70, help="Rest-of-season value (sum of z-scores)")}
    styler = frame.style.apply(lambda _: css, axis=None).format(
        {c: "{:+.1f}" for c in frame.columns[2:4]} | {"ROS value": "{:+.2f}"})
    _on_pick(show_table(styler, config, key=f"tc_{side}_{ctx['epoch']}", height=300,
                        links={"Player": board_ids(model, rows)}), open_player)


def _result(model: TradeModel, ctx: dict, open_player, b: str, give: list[int], get: list[int]) -> None:
    me, lg, cats = ctx["my_team"], model.lg, ctx["cats"]
    res = evaluate(lg, me, give, b, get)
    veto = st.session_state.get(P["veto"], DEFAULT_VETO)
    weeks = _weeks_note(model)
    html(f"""<div class="hero">
        <div><b class="{pos_neg(res.me.cats)}">{res.me.cats:+.1f}</b><span>category wins for you {weeks}</span></div>
        <div><b class="{pos_neg(res.them.cats)}">{res.them.cats:+.1f}</b><span>category wins for {escape(b)}</span></div>
        <div><b class="{pos_neg(res.gap)}">{res.gap:+.1f}</b><span>gap (your gain minus theirs)</span></div>
        <div><b class="{pos_neg(res.me.win)}">{res.me.win * 100:+.1f} pts</b><span>your weekly matchup win chance
        against the league</span></div>
        <div><b class="{pos_neg(res.them.win)}">{res.them.win * 100:+.1f} pts</b><span>theirs</span></div></div>""")
    checks = []
    checks.append('<span class="pos">Roster check: both teams still have a healthy player at every position.</span>'
                  if res.fits else
                  '<span class="neg">Roster check: one side would have no healthy player at a position.</span>')
    if res.them.cats < -veto:
        checks.append(f'<span class="neg">Veto risk: it costs them {res.them.cats:+.1f} real category wins, past '
                      f'your {veto}-win cap (set in the finder).</span>')
    html(f'<p class="note">{" ".join(checks)}</p>')

    ids = dict(zip(range(len(lg.names)), board_ids(model, range(len(lg.names)))))
    for side, label in ((res.a, "You"), (res.b, b)):
        if not side.drops and not side.adds:
            continue
        cols = st.columns([1.1] + [1] * (len(side.drops) + len(side.adds)) + [3], vertical_alignment="center")
        cols[0].html(f'<span class="note"><b>{escape(label)}</b> {"drop" if label == "You" else "drops"} / '
                     'picks up:</span>')
        for n, x in enumerate(side.drops + side.adds, start=1):
            with cols[n]:
                name_button(lg.names[x] + (" (drop)" if x in side.drops else " (FA add)"), ids[x],
                            f"plink_tc_{label[:6]}_{x}_{ctx['epoch']}", open_player)

    left, right = st.columns([1, 1.4])
    with left:
        h4(f"Category wins {weeks}")
        rows = "".join(
            f'<tr><td class="l"><b>{c}</b></td><td class="{pos_neg(res.me.by_cat[i])}">{res.me.by_cat[i]:+.2f}</td>'
            f'<td class="{pos_neg(res.them.by_cat[i])}">{res.them.by_cat[i]:+.2f}</td></tr>'
            for i, c in enumerate(cats))
        rows += (f'<tr><td class="l"><b>Total</b></td><td class="{pos_neg(res.me.cats)}"><b>{res.me.cats:+.2f}</b></td>'
                 f'<td class="{pos_neg(res.them.cats)}"><b>{res.them.cats:+.2f}</b></td></tr>')
        html(f'<div class="ltable"><table><thead><tr><th class="l">Category</th><th>You</th><th>{escape(b)}</th>'
             f'</tr></thead><tbody>{rows}</tbody></table></div>')
    with right:
        rk0, rk1 = lg.cat_rank, lg.ranks(res.weeks)
        for side, label in ((res.a, "Your roster"), (res.b, f"{b}'s roster")):
            h4(f"{label} after the trade")
            _roster_after(model, side, ctx, open_player)
            leaving = [f"{escape(lg.names[x])} (traded)" for x in side.outgoing] + \
                      [f"{escape(lg.names[x])} (dropped)" for x in side.drops]
            html(f'<p class="note"><b>Leaving:</b> {", ".join(leaving) or "nobody"}. League rank by category, '
                 'before → after (1 = best):</p><div class="catranks">' + "".join(
                     f'<div><span>{c}</span><b class="{"pos" if rk1.loc[side.team, c] < rk0.loc[side.team, c] else "neg" if rk1.loc[side.team, c] > rk0.loc[side.team, c] else ""}">'
                     f'#{rk0.loc[side.team, c]} → #{rk1.loc[side.team, c]}</b></div>' for c in cats) + "</div>")


def _roster_after(model: TradeModel, side, ctx: dict, open_player) -> None:
    lg, t = model.lg, ctx["tokens"]
    rows = sorted(side.roster, key=lambda x: -lg.stats.value[x])
    move = {x: "New" for x in side.incoming} | {x: "FA add" for x in side.adds}
    frame = pd.DataFrame({
        "Player": [lg.names[x] for x in rows],
        "Pos": [model.board["eligible"].iloc[x] for x in rows],
        "NBA": [model.board["pro_team"].iloc[x] for x in rows],
        "Move": [move.get(x, "") for x in rows],
        "ROS value": [lg.stats.value[x] for x in rows],
    })
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css["Player"] = link_style(t)
    css["Move"] = [f"color: {t['good']}; font-weight: 600" if m else "" for m in frame["Move"]]
    styler = frame.style.apply(lambda _: css, axis=None).format({"ROS value": "{:+.2f}"})
    _on_pick(show_table(styler, {"Player": st.column_config.TextColumn(width="medium"),
                                 "Move": st.column_config.TextColumn(width=64, help="New: arrives in the trade. "
                                                                                    "FA add: picked up to fill the spot")},
                        key=f"tc_after_{side.team[:12]}_{ctx['epoch']}", height="auto",
                        links={"Player": board_ids(model, rows)}), open_player)


def _finder(model: TradeModel, ctx: dict, open_player) -> None:
    me, lg, cats = ctx["my_team"], model.lg, ctx["cats"]
    with card("finder"):
        card_title("Trade finder", f"Searches 1-for-1, 2-for-1 and 1-for-2 deals (2-for-2 with the toggle), using "
                   f"the {FINDER_POOL} players of yours each manager would most want and the {FINDER_POOL} of theirs "
                   f"that help you most. It keeps deals worth more than {MIN_GAIN} category wins to you that read as "
                   "fair or better to them (after loss aversion), stay under your veto cap and leave both rosters "
                   "legal; at most two deals per incoming player.")
        c1, c2, c3, c4, c5 = st.columns([1.6, 0.8, 1.3, 1.2, 0.8], vertical_alignment="bottom")
        mode = c1.segmented_control("Goal", list(FINDER_MODES), format_func=FINDER_MODES.get, default="gap",
                                    required=True, key=P["finder_mode"], **URL)
        cat = (c2.selectbox("Category", cats, index=cats.index("BLK") if "BLK" in cats else 0,
                            key=P["finder_cat"], **URL) if mode == "cat" else None)
        veto = c3.slider("Veto cap: they can lose at most", 1, 12, DEFAULT_VETO, key=P["veto"],
                         help="Real category wins the other team may lose before the deal looks vetoable", **URL)
        partner = c4.selectbox("Partner", ["All teams", *[t for t in lg.teams if t != me]], key=P["finder_partner"],
                               **URL)
        two = c5.toggle("2-for-2", value=False, key=P["two_for_two"], **URL)
        if mode == "punt":
            html(f'<p class="note">Uses your punt picks at the top ({", ".join(ctx["punt"]) or "none set"}): losses '
                 'there are free, gains everywhere else count.</p>')
        deals = cached_find(model, ctx["trade_key"], me, mode, cat, tuple(ctx["punt"]), float(veto),
                            None if partner == "All teams" else partner, bool(two), None, 30)
        if not deals:
            html('<p class="note">No deal fits: try a higher veto cap, a different goal, or All teams.</p>')
            return
        styler, config, links = _deal_frame(model, deals, ctx, with_partner=True)
        _on_pick(show_table(styler, config, key=f"tf_{ctx['epoch']}", height="auto", links=links), open_player,
                 _Deals(deals, lg.names))


def surplus_table(model: TradeModel, ctx: dict, open_player, n: int, key: str) -> None:
    me, lg, cats, t = ctx["my_team"], model.lg, ctx["cats"], ctx["tokens"]
    rows = surpluses(lg, me)[:n]
    if not rows:
        html('<p class="note">No players to shop.</p>')
        return
    deltas = delta_frame(cats, [s.by_cat for s in rows])
    fits = [fit_columns(lg, s.by_cat, s.best) for s in rows]
    frame = pd.DataFrame({
        "Your player": [lg.names[s.player] for s in rows],
        "Worth to you": [s.mine for s in rows],
        "Worth most to": [s.best for s in rows],
        "Worth to them": [s.best_worth for s in rows],
        "Surplus": [s.surplus for s in rows],
        "Then": [f"{s.second} ({s.second_worth:+.1f})" if s.second else DASH for s in rows],
        "Fit with them": [f[0] for f in fits],
    })
    frame = pd.concat([frame, deltas], axis=1)
    frame["Shop"] = "Shop him"
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css[cats] = delta_css(deltas, ctx["mode"])
    css["Your player"] = css["Shop"] = link_style(t)
    css["Surplus"] = [f"color: {t['good']}; font-weight: 700" if v > 0 else f"color: {t['bad']}" for v in frame["Surplus"]]
    weeks = _weeks_note(model)
    styler = frame.style.apply(lambda _: css, axis=None).format(
        {"Worth to you": "{:+.1f}", "Worth to them": "{:+.1f}", "Surplus": "{:+.1f}", **{c: "{:+.1f}" for c in cats}})
    config = {
        "Your player": st.column_config.TextColumn(width="medium", pinned=True),
        "Worth to you": st.column_config.NumberColumn(
            width=82, help=f"Category wins {weeks} you'd lose replacing him with the best free agent at his position"),
        "Worth most to": st.column_config.TextColumn(width="medium", help="The team he'd add the most for"),
        "Worth to them": st.column_config.NumberColumn(width=86, help=f"Category wins {weeks} he'd add for them"),
        "Surplus": st.column_config.NumberColumn(width=70, help="Worth to them minus worth to you"),
        "Then": st.column_config.TextColumn(width="medium", help="The next team he'd help most"),
        "Fit with them": st.column_config.TextColumn(width="medium", help="How his gains line up with where they stand"),
        "Shop": st.column_config.TextColumn(width=80, help="Open the calculator with him on your side, with that team"),
        **{c: st.column_config.NumberColumn(width=52, help=f"His {c} wins for them {weeks}") for c in cats},
    }
    links = {"Your player": board_ids(model, [s.player for s in rows]),
             "Shop": [("shop", i) for i in range(len(rows))]}
    pick = show_table(styler, config, key=f"{key}_{ctx['epoch']}", height="auto", links=links)
    if isinstance(pick, tuple):
        s = rows[pick[1]]
        request_trade(s.best, [lg.names[s.player]], [])
    elif pick is not None:
        open_player(pick)


# ---------------------------------------------------------------- my team and player pages

def render_my_team_surplus(model: TradeModel, ctx: dict, open_player) -> None:
    me = ctx["my_team"]
    if not me or me not in model.lg.rosters:
        return
    with card("mt_surplus"):
        card_title("Who to trade, and to whom", f"Your top five by surplus: how much more they'd add for another team "
                   f"than they're worth to you, rest of season. The full list and the calculator are on the "
                   f"{TRADES_TAB} tab.")
        surplus_table(model, ctx, open_player, n=5, key="mt_surplus")


def render_trades_for(model: TradeModel, pid, ctx: dict, open_player) -> None:
    """Top three finder deals that bring in another team's player."""
    me, lg = ctx["my_team"], model.lg
    keys = np.array([str(k) for k in board_ids(model, range(len(lg.names)))])
    hit = np.flatnonzero(keys == str(pid))
    if not me or me not in lg.rosters or not len(hit):
        return
    i = int(hit[0])
    owner = lg.owner[i]
    if owner in (me, FREE_AGENT) or owner not in lg.rosters:
        return
    veto = float(st.session_state.get(P["veto"], DEFAULT_VETO))
    deals = cached_find(model, ctx["trade_key"], me, "gap", None, tuple(ctx["punt"]), veto, None, False, i, 3)
    with card("trades_for"):
        card_title(f"Trades for {lg.names[i]}", f"The best deals from your roster that bring him in, ranked by "
                   f"the gap between your real gain and {owner}'s {_weeks_note(model)}, kept only if they read as "
                   f"fair or better to that manager and stay under your {veto:.0f}-win veto cap. "
                   f"{VALUED_ON[ctx.get('ros_label', 'interim')]}.")
        if not deals:
            html('<p class="note">Nothing from your roster both helps you and reads as fair to them under the veto '
                 f'cap. Raise the cap on the {TRADES_TAB} tab or try the calculator.</p>')
            if st.button("Open the calculator with him", key=f"tfor_open_{i}"):
                request_trade(owner, [], [lg.names[i]])
            return
        styler, config, links = _deal_frame(model, deals, ctx, with_partner=False)
        _on_pick(show_table(styler, config, key=f"tfor_{ctx['epoch']}", height="auto", links=links), open_player,
                 _Deals(deals, lg.names))
