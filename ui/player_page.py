"""A player's own page: tiers, ranks, category profile, minutes and efficiency."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from ninecat import charts
from ninecat.board import player_key
from ninecat.league import League, drop_options, fit_note, uncertainty
from ninecat.matchup import WEEKS, change, outlook
from ninecat.lines import SOURCES, name_key
from ninecat.player import (
    category_profile,
    comparables,
    efficiency_items,
    minutes_history,
    minutes_trend,
    position_group,
    primary_position,
    rate_items,
    rolling_minutes,
    season_so_far,
    stat_lines,
)
from ui.components import (
    FREE_AGENT,
    card,
    card_title,
    category_table,
    delta_css,
    delta_frame,
    fmt_num,
    fmt_pct,
    h4,
    html,
    legend,
    link_style,
    show_plain,
    show_table,
    tags,
)

PLOTLY_CONFIG = {"displayModeBar": False}


def _chart(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def pos_neg(x: float, threshold: float = 0.005) -> str:
    return "pos" if x > threshold else "neg" if x < -threshold else ""


def _uncertainty_tile(lg: League, idx: int, r: pd.Series) -> str:
    u = uncertainty(lg.impact[idx], r.get("L_G"), r.get("injury_status"))
    return (f'<div title="{escape(u.why)}"><b>{u.low:+.1f} to {u.high:+.1f}</b>'
            f'<span>impact range · {u.label}</span></div>')


def _season_strip(r: pd.Series, season: str) -> str:
    """This season's per-game line in one row of tiles, plus recent minutes."""
    now = season_so_far(r)
    if now is None:
        return f'<p class="note">No {season} games yet.</p>'
    games = int(now["G"])
    pct = lambda m, a: f"{now[m] / now[a]:.3f}".lstrip("0") if now[a] > 0 else "–"
    cells = [(f"{now['PTS']:.1f}", "PTS"), (f"{now['REB']:.1f}", "REB"), (f"{now['AST']:.1f}", "AST"),
             (f"{now['STL']:.1f}", "STL"), (f"{now['BLK']:.1f}", "BLK"), (f"{now['3PM']:.1f}", "3PM"),
             (pct("FGM", "FGA"), "FG%"), (pct("FTM", "FTA"), "FT%"), (f"{now['TO']:.1f}", "TO"),
             (f"{now['MIN']:.1f}", "MIN")]
    recent = []
    for days in (7, 15):
        g, m = r.get(f"S_G_{days}D"), r.get(f"S_MIN_{days}D")
        if pd.notna(g) and g > 0 and pd.notna(m):
            recent.append(f"last {days} days {m:.1f} min ({int(g)} g)")
    if pd.notna(r.get("S_MIN_L5")):
        recent.append(f"last 5 games {r['S_MIN_L5']:.1f} min")
    sub = f"{season} so far · {games} game{'s' if games != 1 else ''}"
    if games < 3:
        sub += " · the This season source waits for 3"
    tiles = "".join(f"<div><b>{escape(v)}</b><span>{k}</span></div>" for v, k in cells)
    tail = f'<p class="note">Minutes: {", ".join(recent)}.</p>' if recent else ""
    return f'<div class="eyebrow" style="margin-top:10px">{sub}</div><div class="tiles">{tiles}</div>{tail}'


def _impact_section(board: pd.DataFrame, lg: League, i: int, ctx: dict, open_player) -> None:
    t, cats, me = ctx["tokens"], ctx["cats"], ctx["my_team"]
    imp = lg.impact[i]
    league_tile = (f'<div><b class="{pos_neg(imp)}">{imp:+.1f}</b><span>for the average team in your league · '
                   f'#{lg.impact_rank[i]} of {len(board)}</span></div>')
    if not me or me not in lg.base:
        with card("impact"):
            card_title("Impact", "League impact: how many more categories the average team in your league would win "
                       "over the season by adding him, dropping its lowest-value player at his position. Pick your "
                       "team at the top for your own numbers.")
            html(f'<div class="hero">{league_tile}</div>')
            h4("Where it comes from (average team)")
            _chart(charts.category_delta(cats, lg.impact_by_cat[i], t), "pp_imp_avg")
        return

    own = lg.owner[i] == me
    fa = lg.owner[i] == FREE_AGENT
    with card("impact"):
        card_title(f"Impact on {me}")
        if own:
            incoming, outgoing = lg.best_fa_at(i), i
            if incoming is None:
                html('<p class="note">No free agents to compare him with.</p>')
                return
            framing = (f"What he's worth to you: your team with him compared with replacing him by the best free "
                       f"agent at his position ({escape(lg.names[incoming])}, {escape(board['eligible'].iloc[incoming])})")
        else:
            mine = list(lg.rosters[me])
            dopts = drop_options(lg, me, i) if fa else None
            options = [o.drop for o in dopts] if fa else sorted(mine, key=lambda x: lg.stats.value[x])
            default = dopts[0].drop if fa else lg.drop_for(mine, i)
            season = {o.drop: o.change.cats for o in dopts} if fa else {}
            verb = "Add him, drop" if fa else f"Trade for him (from {r_owner(lg, i)}), give up"
            outgoing = st.selectbox(verb, options, index=options.index(default) if default in options else 0,
                                    format_func=lambda x: lg.names[x] + (f" ({season[x]:+.1f} season)" if x in season else ""),
                                    key=f"pp_out_{i}_{me}")
            incoming = i
            best = (" (the best drop)" if fa and outgoing == default else
                    " (your lowest-value player at his position)" if not fa and outgoing == default else "")
            framing = ((f"Adding {escape(lg.names[i])} and dropping {escape(lg.names[outgoing])}" if fa else
                        f"Trading {escape(lg.names[outgoing])} for {escape(lg.names[i])}") + best)
        html(f'<p class="note">{framing}. Each category is simulated against every other team\'s current roster for '
             f'{WEEKS} head-to-head weeks, so "expected wins" is out of {WEEKS} in each category.</p>')

        weeks = lg.swap_weeks(me, incoming, outgoing)
        base, swapped = lg.base[me], outlook(me, weeks, lg.stats)
        rk0, rk1 = lg.cat_rank.loc[me], lg.ranks(weeks).loc[me]
        before, after = (swapped, base) if own else (base, swapped)
        r_before, r_after = (rk1, rk0) if own else (rk0, rk1)
        ch = change(before, after)
        html(f"""<div class="hero">
            <div><b class="{pos_neg(ch.cats)}">{ch.cats:+.1f}</b><span>category wins over the season for you</span></div>
            <div><b class="{pos_neg(ch.win)}">{ch.win * 100:+.1f} pts</b><span>weekly matchup win chance</span></div>
            {league_tile}</div>""")

        left, right = st.columns(2)
        with left:
            h4("Expected category wins over the season")
            rows = "".join(
                f'<tr><td class="l"><b>{c}</b></td><td>{before.by_cat[k] * WEEKS:.1f}</td>'
                f'<td>{after.by_cat[k] * WEEKS:.1f}</td><td class="{pos_neg(ch.by_cat[k])}">{ch.by_cat[k]:+.2f}</td>'
                f'<td>{r_before[c]} → <b>{r_after[c]}</b></td>'
                f'<td class="{pos_neg(lg.impact_by_cat[i, k])}">{lg.impact_by_cat[i, k]:+.2f}</td></tr>'
                for k, c in enumerate(cats))
            tot_b, tot_a = before.by_cat.sum() * WEEKS, after.by_cat.sum() * WEEKS
            rows += (f'<tr><td class="l"><b>Total</b></td><td><b>{tot_b:.1f}</b></td><td><b>{tot_a:.1f}</b></td>'
                     f'<td class="{pos_neg(tot_a - tot_b)}"><b>{tot_a - tot_b:+.2f}</b></td><td></td>'
                     f'<td class="{pos_neg(imp)}"><b>{imp:+.2f}</b></td></tr>')
            html(f'<div class="ltable"><table><thead><tr><th class="l">Cat</th><th>{"Without" if own else "Now"}</th>'
                 f'<th>{"With him" if own else "After"}</th><th>Change</th>'
                 '<th title="Your rank in the league in this category (1 = best)">League rank</th>'
                 f'<th title="Change for the average team in your league">Avg team</th></tr></thead><tbody>{rows}'
                 '</tbody></table></div>')
        with right:
            h4("Change in category wins")
            legend([(t["good"], "For you (bar)"), (t["s2"], "Average team (dot)")])
            _chart(charts.category_delta(cats, ch.by_cat, t, league=lg.impact_by_cat[i]), "pp_imp_delta")
        h4("Matchup win chance by opponent")
        labels = ("Without him", "With him") if own else ("Now", "After the move")
        legend([(t["s2"], labels[0]), (t["s1"], labels[1])])
        _chart(charts.win_change(before.opponents, before.win, after.win, t, labels), "pp_imp_opp")

        if fa:
            h4("Who to drop to make room")
            out_p = next((x for x in lg.rosters[me] if lg.injury[x] == "OUT"), None)
            html('<p class="note">Every player on your roster, ranked by how many category wins you\'d have over the '
                 'season after the swap. The top name is the best drop.'
                 + (f" {escape(lg.names[out_p])} is listed OUT: if you have an open injured-list spot, move him there "
                    "and add without dropping anyone." if out_p is not None else "") + "</p>")
            top = dopts[:5]
            deltas = delta_frame(cats, [o.change.by_cat for o in top])
            fits = [fit_note(lg, o.change.by_cat, me) for o in top]
            frame = pd.concat([pd.DataFrame({
                "Drop": [lg.names[o.drop] + (" (best drop)" if k == 0 else "") for k, o in enumerate(top)],
                "Season +": [o.change.cats for o in top]}), deltas], axis=1)
            frame["Fit"] = [f.text for f in fits]
            frame["Fit detail"] = [f.gains + (f" · costs {f.losses}" if f.losses else "") for f in fits]
            css = pd.DataFrame("", index=frame.index, columns=frame.columns)
            css[cats] = delta_css(deltas, ctx["mode"])
            css["Drop"] = link_style(t)
            keys = player_key(board).to_numpy(object)
            open_player(show_table(frame.style.apply(lambda _: css, axis=None).format(
                {"Season +": "{:+.1f}", **{c: "{:+.1f}" for c in cats}}),
                {"Drop": st.column_config.TextColumn(width="medium"),
                 **{c: st.column_config.NumberColumn(width=52) for c in cats}},
                key=f"pp_drops_{ctx['epoch']}", height="auto", links={"Drop": [keys[o.drop] for o in top]}))


def r_owner(lg: League, i: int) -> str:
    return escape(str(lg.owner[i]))


def _fit_table(lg: League, i: int, ctx: dict) -> None:
    cats = ctx["cats"]
    order = sorted(range(len(lg.teams)), key=lambda j: -lg.worth[i, j])
    deltas = delta_frame(cats, [lg.worth_by_cat[i, j] for j in order])
    fits = [fit_note(lg, lg.worth_by_cat[i, j], lg.teams[j]) for j in order]
    own = lg.owner[i]
    frame = pd.concat([pd.DataFrame({
        "Team": [lg.teams[j] for j in order],
        "Note": ["owner: vs best FA at his position" if lg.teams[j] == own else
                 "your team" if lg.teams[j] == ctx["my_team"] else "" for j in order],
        "Total": [lg.worth[i, j] for j in order]}), deltas], axis=1)
    frame["Fit"] = [("✓ " if f.good else "") + f.text for f in fits]
    frame["Fit detail"] = [f.gains + (f" · costs {f.losses}" if f.losses else "") for f in fits]
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    css[cats] = delta_css(deltas, ctx["mode"])
    tk = ctx["tokens"]
    css["Total"] = np.where(frame["Total"] > 0.005, f"color: {tk['good']}; font-weight: 600",
                            np.where(frame["Total"] < -0.005, f"color: {tk['bad']}; font-weight: 600", ""))
    css["Fit"] = [f"color: {tk['good']}; font-weight: 600" if f.good else "" for f in fits]
    show_plain(frame.style.apply(lambda _: css, axis=None).format({"Total": "{:+.1f}", **{c: "{:+.1f}" for c in cats}}),
               {"Team": st.column_config.TextColumn(width="medium", pinned=True),
                "Note": st.column_config.TextColumn(width="medium"),
                "Fit": st.column_config.TextColumn(width="medium"),
                "Fit detail": st.column_config.TextColumn(width="large"),
                **{c: st.column_config.NumberColumn(width=52) for c in cats}})


def render(board: pd.DataFrame, players: pd.DataFrame, pid, ctx: dict, open_player) -> None:
    """ctx: cats, punt, source, my_team, abbrevs, seasons, pace, logs, mode, tokens, league."""
    t, cats, lg = ctx["tokens"], ctx["cats"], ctx["league"]
    keys = player_key(board).astype(str)
    match = board.index[keys == str(pid)]
    raw = players[player_key(players).astype(str) == str(pid)]
    if match.empty:
        name = raw["name"].iloc[0] if not raw.empty else "This player"
        st.info(f"{name} doesn't have a usable line (or plays under 10 minutes) under the "
                f"{SOURCES[ctx['source']]} stats source. Switch the source to Blend or Projection.")
        return
    idx = match[0]
    r = board.loc[idx]
    pos = primary_position(r)
    group = position_group(board, pos)
    pos_rank = group.index.get_loc(idx) + 1

    # ---- header and tiles
    owner = "Free agent" if r["fantasy_team"] == FREE_AGENT else escape(str(r["fantasy_team"]))
    tier, role = r["fantasy_tier"], r["role_tier"]
    inj = r.get("injury_status")
    inj_chip = (f'<span class="tier">{"Out" if inj == "OUT" else "Day to day"}</span>'
                if isinstance(inj, str) and inj not in ("ACTIVE", "") else "")
    adp = r.get("ADP")
    gap = int(r["gap"])
    gap_txt = f"(+{gap} value)" if gap > 0 else f"({gap})" if gap < 0 else ""
    head, tiles = st.columns([1, 1.15], vertical_alignment="bottom")
    with head:
        html(f"""<div class="pphead"><div class="eyebrow">{escape(str(r['eligible']))} · {escape(str(r['pro_team']))} · {owner}</div>
            <h2>{escape(r['name'])}{tags(r)}</h2>
            <div class="chips"><span class="tier t-{tier.lower()}">{tier}</span><span class="tier">NBA role: {role}</span>{inj_chip}</div></div>""")
    with tiles:
        html(f"""<div class="tiles">
            <div><b>#{r['rank']}</b><span>Value rank</span></div>
            <div><b>{pos} #{pos_rank}</b><span>of {len(group)} {pos}s</span></div>
            <div><b>{'#' + str(int(r['p36_rank'])) if pd.notna(r['p36_rank']) else '–'}</b><span>Per-36 rank</span></div>
            <div><b>{'#' + str(int(r['p100_rank'])) if pd.notna(r['p100_rank']) else '–'}</b><span>Per-100-poss rank</span></div>
            <div><b>{fmt_num(adp)}</b><span>ADP {gap_txt}</span></div>{_uncertainty_tile(lg, idx, r)}</div>""")
    html(_season_strip(r, ctx["seasons"]["current"]))

    # ---- tier ladder
    with card("tier"):
        card_title("Tier", f"Fantasy tier from his value rank under your current settings "
                           f"({ctx['n_teams']} teams). NBA role from projected minutes and usage.")
        _chart(charts.tier_ladder(int(r["rank"]), t), "pp_ladder")

    # ---- league model: impact, worth to each team, replacement, fit
    _impact_section(board, lg, idx, ctx, open_player)
    left, right = st.columns(2)
    with left, card("worth"):
        card_title("Worth to each team", "Category wins over the season each team would gain by getting him, "
                   "dropping its lowest-value player at his position. For his current owner (blue), it's what he's "
                   "worth compared with the best free agent at his position.")
        owner_name = r["fantasy_team"] if r["fantasy_team"] in lg.teams else None
        _chart(charts.worth_bars(lg.teams, lg.worth[idx], owner_name, ctx["my_team"], t), "pp_worth")
    with right, card("replacement"):
        repl = lg.repl[idx]
        repl_txt = f" ({escape(lg.names[repl])})" if repl >= 0 else ""
        pir = lg.pos_impact_rank.get(pos, np.full(len(board), np.nan))[idx]
        card_title("Above position replacement", f"League impact minus the impact of the best free agent at his "
                   f"position{repl_txt}: what he adds that you can't get off the wire. {pos} impact rank: "
                   f"#{'–' if np.isnan(pir) else int(pir)} of {len(group)}.")
        vr = lg.vs_repl[idx]
        html(f'<div class="hero"><div><b class="{pos_neg(vr)}">{vr:+.1f}</b>'
             '<span>category wins over the season vs replacement</span></div></div>')
        _chart(charts.category_delta(cats, lg.vs_repl_by_cat[idx], t), "pp_repl")
    with card("fit"):
        card_title("Category change and team fit", "Change in expected category wins over the season for each team "
                   "that got him. Fit compares the gains with where that team stands: a swing category (won 30–70% "
                   "of weeks) is where extra production flips matchups; adding to a category a team already wins, "
                   "or still loses, moves less.")
        _fit_table(lg, idx, ctx)

    # ---- category profile and position
    prof = category_profile(board, idx, cats)
    left, right = st.columns(2)
    with left, card("profile"):
        best = prof.sort_values("pct", ascending=False)
        card_title("Category profile",
                   "Percentile in each category among the top 150 players. Best: "
                   + ", ".join(f"{c} ({p:.0f})" for c, p in zip(best["cat"].iloc[:3], best["pct"].iloc[:3]))
                   + ". Weakest: " + ", ".join(f"{c} ({p:.0f})" for c, p in
                                               zip(best["cat"].iloc[-2:], best["pct"].iloc[-2:])) + ".")
        legend([(t["s1"], r["name"]), (t["s2"], f"Average {pos} in the top 150")])
        _chart(charts.radar(list(prof["cat"]), list(prof["pct"]), list(prof["pos_avg_pct"]), t,
                            (r["name"], f"Average {pos}")), "pp_radar")
    with right, card("position"):
        card_title(f"Among {pos}s", f"#{pos_rank} of {len(group)} {pos}-eligible players by value. "
                                    "Category ranks within the position:")
        html('<div class="catranks">' + "".join(
            f'<div><span>{c}</span><b>#{k}</b></div>' for c, k in zip(prof["cat"], prof["pos_rank"])) + "</div>")
        hi = list(group.index).index(idx)
        _chart(charts.value_strip(list(group["value"]), list(group["name"]), list(group["rank"]), hi, t), "pp_strip")

    # ---- comparables
    with card("comps"):
        card_title(f"Comparable {pos}s", f"The six {pos}-eligible players closest in value, for trade and "
                                         "waiver comparisons. Click a name to open his page.")
        comps = comparables(board, idx)
        styler, config = category_table(comps, cats, ctx["punt"], ctx["my_team"], ctx["abbrevs"], ctx["mode"])
        pick = show_table(styler, config, key=f"pp_comps_{pid}_{ctx['epoch']}", height="auto",
                          links={"Player": list(player_key(comps))})
        if pick is not None and str(pick) != str(pid):
            open_player(pick)

    # ---- minutes and efficiency
    left, right = st.columns(2)
    with left, card("minutes"):
        hist = minutes_history(r, ctx["seasons"])
        extra = ""
        if pd.notna(r.get("prev_MIN")) and pd.notna(r.get("L_MIN")):
            dirn = "up" if r["L_MIN"] >= r["prev_MIN"] else "down"
            extra = (f" From {ctx['seasons']['prev']} to {ctx['seasons']['last']} he went {dirn} "
                     f"{abs(r['L_MIN'] - r['prev_MIN']):.1f} minutes.")
        card_title("Minutes trend", minutes_trend(r.get("d_min")) + extra +
                   " Shaded bands are NBA role tiers; the hollow dot is the projection.")
        if hist:
            _chart(charts.minutes_chart(hist, t), "pp_minutes")
        else:
            html('<p class="note">No minutes data.</p>')
        logs = ctx["logs"]
        mine = logs[logs["name_key"] == name_key(r["name"])] if not logs.empty else logs
        if not mine.empty:
            g = rolling_minutes(mine)
            seasons = ", ".join(sorted(g["season"].dropna().unique()))
            h4(f"Rolling minutes, last {len(g)} games ({seasons})")
            legend([(t["s1"], "5-game average"), (t["s2"], "10-game average")])
            _chart(charts.rolling_minutes_chart(g, t), "pp_rolling")
            html(f'<p class="note">Last 5 games {g["r5"].iloc[-1]:.1f} min, last 10 {g["r10"].iloc[-1]:.1f}, '
                 f'against {g["MIN"].mean():.1f} over the window.</p>')
    with right, card("efficiency"):
        card_title("Efficiency ranks", f"Percentile overall (top 200) and among {pos}s. "
                                       "The middle tick is the 50th percentile.")
        legend([(t["s1"], "vs all players"), (t["s2"], f"vs {pos}s")])
        _chart(charts.percentile_bars(efficiency_items(board, idx), t, pos), "pp_eff")

    with card("rates"):
        has_adv = pd.notna(r.get("adv_usg"))
        card_title("Rate stats", f"{ctx['seasons']['last']} rates computed from nba.com box scores with "
                                 "Basketball Reference's formulas: the share of available rebounds, teammate "
                                 "baskets, opponent possessions and two-point shots he accounted for while on court."
                   if has_adv else f"No {ctx['seasons']['last']} advanced stats (rookie, or no minutes last season).")
        if has_adv:
            legend([(t["s1"], "vs all players"), (t["s2"], f"vs {pos}s")])
            _chart(charts.percentile_bars(rate_items(board, idx), t, pos), "pp_rates")

    with card("lines"):
        card_title("Stat lines", "Per game, per 36 minutes and per 100 possessions. G is the games behind each "
                                 "line (projected games for the projection and the board line).")
        pace = ctx["pace"].get(r["pro_team"])
        lines = stat_lines(raw.head(1), r, ctx["pace"], {
            "season": f"{ctx['seasons']['current']} so far",
            "recent": "Last 15 days (ESPN)",
            "proj": f"{ctx['seasons']['current']} ESPN projection",
            "last": f"{ctx['seasons']['last']} actual",
            "board": f"Board line: {SOURCES[ctx['source']]}",
            "per100": f"Per 100 possessions ({r['pro_team']} pace {pace:.1f})" if pace else "Per 100 possessions",
        })
        fmt = {c: "{:.1f}" for c in lines.columns if c not in ("Line", "G", "FG%", "FT%")}
        fmt.update({"G": "{:.0f}", "FG%": fmt_pct, "FT%": fmt_pct})
        st.dataframe(lines.style.format(fmt, na_rep="–"), hide_index=True, row_height=30, placeholder="–",
                     column_config={"Line": st.column_config.TextColumn(width="large")})
