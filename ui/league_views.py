"""Views built on the league model: team categories, my team, waiver wire."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from ninecat import charts
from ninecat.board import FANTASY_TIERS
from ninecat.league import FREE_AGENT, League, average_rank, fit_note, team_category_totals, waivers
from ninecat.matchup import WEEKS
from ninecat.player import pct3
from ninecat.theme import heat
from ui.components import (
    DASH,
    card,
    card_title,
    category_table,
    delta_css,
    delta_frame,
    html,
    show_plain,
    show_table,
)

PLOTLY_CONFIG = {"displayModeBar": False}


def _chart(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def signed(x: float, d: int = 1) -> str:
    return f"{x:+.{d}f}"


def pos_neg(x: float, threshold: float = 0.005) -> str:
    return "pos" if x > threshold else "neg" if x < -threshold else ""


@st.cache_data(show_spinner="Trying every free agent against every drop...")
def cached_waivers(_lg: League, key: tuple, team: str):
    """Waivers for one team; `key` identifies the league model for caching."""
    return waivers(_lg, team)


def fit_columns(lg: League, by_cat: np.ndarray, team: str, threshold: float = 0.15) -> tuple[str, str]:
    f = fit_note(lg, by_cat, team, threshold)
    detail = f.gains + (f" · costs {f.losses}" if f.losses else "")
    return ("✓ " if f.good else "") + f.text, detail


# ---------------------------------------------------------------- team categories

def render_team_categories(board: pd.DataFrame, lg: League, ctx: dict) -> None:
    cats, punt, my_team = ctx["cats"], ctx["punt"], ctx["my_team"]
    scored = [c for c in cats if c not in punt]
    totals = team_category_totals(board, cats, lg.teams)
    totals["avg"] = average_rank(totals, scored)
    totals = totals.sort_values("avg")
    n = len(totals)
    mid = (n + 1) / 2

    if my_team in totals.index:
        me = totals.loc[my_team]
        rank = {c: int(me[f"r_{c}"]) for c in cats}
        ordered = sorted(cats, key=lambda c: -rank[c])
        weak, strong = ordered[:2], ordered[-3:][::-1]
        advice = (f"{weak[0]} looks like a natural punt: you're already near the bottom, so trading away from it "
                  "costs little. Toggle it in Punt categories to re-rank everyone for that build."
                  if rank[weak[0]] >= max(2, n - 2) else
                  "No category is far enough behind to be an obvious punt. A balanced build is reasonable here.")
        html(f'<div class="callout"><span class="eyebrow">Your build</span>'
             f'<b>Strongest: {", ".join(f"{c} (#{rank[c]})" for c in strong)}. '
             f'Weakest: {", ".join(f"{c} (#{rank[c]})" for c in weak)}.</b>'
             f'<span class="note">{advice}</span></div>')

    html(f'<p class="note">Rank 1–{n} in each category from each team\'s top 10 players by value, summed per game. '
         'Green is the best in the league, red the worst. Sorted by average rank over non-punted categories.</p>')

    def cell(row, c):
        r = int(row[f"r_{c}"])
        z = (mid - r) / (mid - 1) * 2.5 if mid > 1 else 0
        v = DASH if pd.isna(row[c]) else pct3(row[c]) if c in ("FG%", "FT%") else f"{row[c]:.1f}"
        return f'<td style="{heat(z, ctx["mode"])}" title="{v} per game">{r}<small>{v}</small></td>'

    head = "".join(f'<th style="{"text-decoration:line-through" if c in punt else ""}">{c}</th>' for c in cats)
    body = "".join(
        f'<tr class="{"mine" if team == my_team else ""}"><td class="l"><b>{escape(team)}</b>'
        f'<small>{escape(ctx["abbrevs"].get(team, ""))}</small></td><td><b>{row["avg"]:.1f}</b></td>'
        + "".join(cell(row, c) for c in cats) + "</tr>"
        for team, row in totals.iterrows())
    html(f'<div class="ltable"><table><thead><tr><th class="l">Team</th><th>Avg rk</th>{head}</tr></thead>'
         f'<tbody>{body}</tbody></table></div>')

    with card("matchup_ranks"):
        card_title("Projected standings", f"From the matchup model: each team's chance to win a weekly matchup "
                   f"against every other roster as it stands, and expected categories won per week (of {len(cats)}).")
        st_rows = lg.standings()
        rows = "".join(
            f'<tr class="{"mine" if r.team == my_team else ""}"><td>{i}</td><td class="l">{escape(r.team)}</td>'
            f'<td><b>{r.win:.0%}</b></td><td>{r.win * WEEKS:.1f}–{(1 - r.win) * WEEKS:.1f}</td><td>{r.cats:.2f}</td></tr>'
            for i, r in enumerate(st_rows.itertuples(), start=1))
        html('<div class="ltable"><table><thead><tr><th>#</th><th class="l">Team</th><th>Win chance</th>'
             f'<th>Record over {WEEKS} weeks</th><th>Cats / week</th></tr></thead><tbody>{rows}</tbody></table></div>')


# ---------------------------------------------------------------- my team

def render_my_team(board: pd.DataFrame, lg: League, ctx: dict, pick_player) -> None:
    me, cats, t = ctx["my_team"], ctx["cats"], ctx["tokens"]
    if not me or me not in lg.base:
        st.info("Pick your team at the top to see your roster, category outlook and best moves.")
        return
    o = lg.base[me]
    standings = lg.standings()
    standing = int(standings.index[standings["team"] == me][0]) + 1
    ranks = lg.cat_rank.loc[me]
    by_cat = dict(zip(cats, o.by_cat))
    ordered = sorted(cats, key=lambda c: -by_cat[c])

    roster = board[board["fantasy_team"] == me]
    counts = roster["fantasy_tier"].value_counts()
    chips = "".join(f'<span class="tier t-{n.lower()}">{counts[n]} {n}</span>' for _, n in FANTASY_TIERS if n in counts)
    head, tiles = st.columns([1, 1.15], vertical_alignment="bottom")
    head.html(f'<div class="pphead"><div class="eyebrow">My team</div><h2 style="font-size:32px">{escape(me)}</h2>'
              f'<div class="chips">{chips}</div></div>')
    tiles.html(f"""<div class="tiles">
        <div><b>#{standing}</b><span>projected standing of {len(lg.teams)}</span></div>
        <div><b>{o.avg_win * WEEKS:.1f}–{(1 - o.avg_win) * WEEKS:.1f}</b><span>expected matchup record</span></div>
        <div><b>{o.avg_cats:.2f}</b><span>categories won per week (of {len(cats)})</span></div>
        <div><b>{o.avg_cats * WEEKS:.0f}</b><span>category wins over {WEEKS} weeks</span></div></div>""")

    left, right = st.columns(2)
    with left, card("outlook"):
        card_title("Category outlook", f"Share of weeks you're expected to win each category against the league, "
                   f"your expected wins out of {WEEKS}, and your league rank. Strongest: {', '.join(ordered[:3])}. "
                   f"Weakest: {', '.join(ordered[-2:])}.")
        _chart(charts.category_strength(cats, o.by_cat, [ranks[c] for c in cats], WEEKS, ctx["punt"], t), "mt_strength")
    with right, card("matchups"):
        half = len(cats) / 2
        card_title("Matchups", f"Chance to win the weekly matchup (a majority of the {len(cats)} scored categories"
                   + (f", with a {half:.0f}–{half:.0f} split counted as half" if len(cats) % 2 == 0 else "")
                   + ") against each team as rosters stand today.")
        _chart(charts.win_bars(o.opponents, o.win, o.cats_won, t), "mt_wins")

    band = {c: lg.band(me, c) for c in cats}
    groups = {b: [c for c in ordered if band[c] == b] for b in ("lock", "swing", "weak")}
    swing = sorted(groups["swing"], key=lambda c: abs(by_cat[c] - 0.5))
    chip = lambda c: f'<span class="catchip"><b>{c}</b> {by_cat[c]:.0%} · #{ranks[c]}</span>'
    swing_help = (f"Your swing categories are where one move changes the most weeks: {', '.join(swing[:3])}. "
                  "Target players strong there." if swing else
                  "No category sits near 50%, so you're locked in or out almost everywhere.")
    weak = groups["weak"]
    punt_help = (f"{', '.join(weak)} {'are' if len(weak) > 1 else 'is'} already a lost cause most weeks. Consider "
                 f"punting {'them' if len(weak) > 1 else 'it'}: trade away players whose value comes from "
                 f"{'those categories' if len(weak) > 1 else 'that category'} for swing-category help." if weak else "")
    with card("sw"):
        card_title("Strengths and weaknesses")
        none = '<span class="note">None</span>'
        html(f'<div class="sw"><div><div class="card-h4">Locks (win 70%+ of weeks)</div><div class="chips">'
             f'{"".join(map(chip, groups["lock"])) or none}</div></div>'
             f'<div><div class="card-h4">Swing (30–70%)</div><div class="chips">{"".join(map(chip, swing)) or none}</div></div>'
             f'<div><div class="card-h4">Weak (under 30%)</div><div class="chips">{"".join(map(chip, weak)) or none}</div></div></div>'
             f'<p class="note">{swing_help} {punt_help}</p>')

    j = lg.teams.index(me)
    pos = np.flatnonzero(board["fantasy_team"].to_numpy(object) == me)
    worth = pd.Series(lg.worth[pos, j], index=board.index[pos]).sort_values(ascending=False)
    roster = board.loc[worth.index]
    with card("roster"):
        card_title("Roster", "Worth to you is how many category wins over the season you'd lose by replacing him "
                   "with the best free agent at his position. The bottom of this list is where your drops come from. "
                   "Select a player for his page.")
        repl = pd.Series([lg.names[lg.repl[i]] if lg.repl[i] >= 0 else "replacement" for i in pos],
                         index=board.index[pos])
        styler, config = category_table(roster, cats, ctx["punt"], None, ctx["abbrevs"], ctx["mode"],
                                        extra={"Worth to you": worth, "vs FA": repl.loc[worth.index]})
        config["Worth to you"] = st.column_config.NumberColumn(
            width=80, help="Category wins over the season you'd lose replacing him with the best FA")
        config["vs FA"] = st.column_config.TextColumn(width="medium", help="The free agent he's measured against")
        pick_player(roster, show_table(styler, config, key=f"mt_roster_{ctx['epoch']}", height="auto"))

    with card("adds"):
        card_title("Best free-agent adds for you", f"Each free agent is tried against every possible drop from your "
                   f"roster, simulated over {WEEKS} weeks; the best drop is shown. Category columns show the change "
                   "in expected wins.")
        picks, _, _ = cached_waivers(lg, ctx["league_key"], me)
        _pickup_table(lg, picks[:10], ctx, key=f"mt_adds_{ctx['epoch']}", pick_player=pick_player, board=board)


def _pickup_table(lg: League, picks, ctx: dict, key: str, pick_player, board: pd.DataFrame) -> None:
    cats, me = ctx["cats"], ctx["my_team"]
    if not picks:
        html('<p class="note">No free agents to try.</p>')
        return
    deltas = delta_frame(cats, [p.change.by_cat for p in picks])
    fits = [fit_columns(lg, p.change.by_cat, me) for p in picks]
    frame = pd.DataFrame({
        "Player": [lg.names[p.player] for p in picks],
        "Pos": [board["eligible"].iloc[p.player] for p in picks],
        "Season +": [p.change.cats for p in picks],
        "Matchup win": [p.change.win * 100 for p in picks],
        "Best drop": [lg.names[p.drop] for p in picks],
        "Next best": [f"{lg.names[p.alt]} ({p.alt_cats:+.1f})" if p.alt is not None else DASH for p in picks],
        "Team fit": [f[0] for f in fits],
    })
    if any(p.il is not None for p in picks):
        frame["Or IL, no drop"] = [f"IL {lg.names[p.il]}: {p.il_cats:+.1f}" if p.il is not None else DASH for p in picks]
    frame = pd.concat([frame, deltas], axis=1)
    frame["Fit detail"] = [f[1] for f in fits]
    frame["Value"] = [lg.stats.value[p.player] for p in picks]

    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css[cats] = delta_css(deltas, ctx["mode"])
    good, bad = f"color: {ctx['tokens']['good']}; font-weight: 600", f"color: {ctx['tokens']['bad']}; font-weight: 600"
    css["Season +"] = np.where(frame["Season +"] > 0.005, good, np.where(frame["Season +"] < -0.005, bad, ""))
    css["Team fit"] = ["font-weight: 600; color: " + ctx["tokens"]["good"] if f[0].startswith("✓") else "" for f in fits]
    fmt = {"Season +": "{:+.1f}", "Matchup win": "{:+.1f} pts", "Value": "{:+.2f}", **{c: "{:+.1f}" for c in cats}}
    styler = frame.style.apply(lambda _: css, axis=None).format(fmt)
    config = {"Player": st.column_config.TextColumn(width="medium", pinned=True),
              "Season +": st.column_config.NumberColumn(width=72, help=f"Extra category wins over {WEEKS} weeks"),
              "Matchup win": st.column_config.NumberColumn(width=90, help="Change in weekly matchup win chance"),
              "Team fit": st.column_config.TextColumn(width="medium"),
              "Fit detail": st.column_config.TextColumn(width="large"),
              **{c: st.column_config.NumberColumn(width=52) for c in cats}}
    view = board.iloc[[p.player for p in picks]]
    pick_player(view, show_table(styler, config, key=key, height="auto"))


# ---------------------------------------------------------------- waiver wire

def render_waivers(board: pd.DataFrame, lg: League, ctx: dict, pick_player) -> None:
    me, cats = ctx["my_team"], ctx["cats"]
    if not me or me not in lg.base:
        fa = board[board["fantasy_team"] == FREE_AGENT].sort_values("impact", ascending=False).head(30)
        html('<div class="callout"><span class="eyebrow">Waiver wire</span><b>Best free agents for the average team</b>'
             '<span class="note">Pick your team at the top to rank pickups for your roster and see who to drop.</span></div>')
        def top_gains(i: int) -> str:
            ranked = sorted(zip(cats, lg.impact_by_cat[i]), key=lambda x: -x[1])
            return ", ".join(f"{c} {v:+.2f}" for c, v in ranked[:3] if v > 0.005) or DASH

        gains = [top_gains(i) for i in fa.index]
        styler, config = category_table(fa, cats, ctx["punt"], None, ctx["abbrevs"], ctx["mode"],
                                        extra={"Biggest gains": pd.Series(gains, index=fa.index)})
        config["Biggest gains"] = st.column_config.TextColumn(width="large")
        pick_player(fa, show_table(styler, config, key=f"fa_avg_{ctx['epoch']}", height=640))
        return

    picks, drops, out_player = cached_waivers(lg, ctx["league_key"], me)
    il_note = (f"{escape(lg.names[out_player])} is listed OUT: if you have an open injured-list spot, the \"Or IL\" "
               "column shows the result of moving him there and adding without dropping anyone. "
               if out_player is not None else "")
    html(f'<div class="callout"><span class="eyebrow">Waiver wire for {escape(me)}</span>'
         f'<b>Best pickups for the rest of the season</b><span class="note">Every free agent is tried against every '
         f'possible drop from your roster, and the drop that leaves you winning the most categories is shown. '
         f'{il_note}Values are extra category wins over {WEEKS} weeks.</span></div>')
    healthy = st.toggle("Hide injured (OUT)", value=True, key="fa_healthy")
    shown = [p for p in picks if not (healthy and lg.injury[p.player] == "OUT")][:30]
    _pickup_table(lg, shown, ctx, key=f"fa_picks_{ctx['epoch']}", pick_player=pick_player, board=board)

    with card("drops"):
        card_title("Who to drop to make room", "Each player on your roster, with the best pickup you could swap him "
                   "for over the season and what that swap is worth. Positive means dropping him for that player "
                   "makes you better; the most positive rows are your drops. Negative means keep him.")
        j = lg.teams.index(me)
        k = len(cats)
        deltas = delta_frame(cats, [d.by_cat if d.by_cat is not None else np.zeros(k) for d in drops])
        frame = pd.DataFrame({
            "Your player": [lg.names[d.player] for d in drops],
            "Best swap": [d.cats for d in drops],
            "Pick up": [lg.names[d.best] if d.best is not None else DASH for d in drops],
            "Worth to you": [lg.worth[d.player, j] for d in drops],
        })
        frame = pd.concat([frame, deltas], axis=1)
        frame["Verdict"] = ["Drop candidate" if d.cats > 0.5 else "Marginal" if d.cats > 0 else "Keep" for d in drops]
        css = pd.DataFrame("", index=frame.index, columns=frame.columns)
        css[cats] = delta_css(deltas, ctx["mode"])
        good = f"color: {ctx['tokens']['good']}; font-weight: 600"
        css["Verdict"] = np.where(frame["Verdict"] == "Drop candidate", good, "")
        css["Best swap"] = np.where(frame["Best swap"] > 0, good, "")
        fmt = {"Best swap": "{:+.1f}", "Worth to you": "{:+.1f}", **{c: "{:+.1f}" for c in cats}}
        show_plain(frame.style.apply(lambda _: css, axis=None).format(fmt),
                   {"Your player": st.column_config.TextColumn(width="medium", pinned=True),
                    "Pick up": st.column_config.TextColumn(width="medium"),
                    **{c: st.column_config.NumberColumn(width=52) for c in cats}})
    html('<p class="note">Week-by-week streams against your actual opponent need the NBA weekly schedule and your '
         'league schedule, which the board doesn\'t pull yet.</p>')
