"""A player's own page: tiers, ranks, category profile, minutes and efficiency."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ninecat import charts
from ninecat.board import player_key
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
    stat_lines,
)
from ui.components import (
    FREE_AGENT,
    card,
    card_title,
    category_table,
    fmt_num,
    fmt_pct,
    h4,
    html,
    legend,
    show_table,
    tags,
)

PLOTLY_CONFIG = {"displayModeBar": False}


def _chart(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def render(board: pd.DataFrame, players: pd.DataFrame, pid, ctx: dict, open_player) -> None:
    """ctx: cats, punt, source, my_team, abbrevs, team_names, seasons, pace, logs, mode, tokens."""
    t, cats = ctx["tokens"], ctx["cats"]
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
            <div><b>{fmt_num(adp)}</b><span>ADP {gap_txt}</span></div></div>""")

    # ---- tier ladder
    with card("tier"):
        card_title("Tier", f"Fantasy tier from his value rank under your current settings "
                           f"({ctx['n_teams']} teams). NBA role from projected minutes and usage.")
        _chart(charts.tier_ladder(int(r["rank"]), t), "pp_ladder")

    # ---- category profile and position
    prof = category_profile(board, idx, cats)
    left, right = st.columns(2)
    with left, card("profile"):
        best = prof.sort_values("pct", ascending=False)
        card_title("Category profile",
                   f"Percentile in each category among the top 150 players. Best: "
                   + ", ".join(f"{c} ({p:.0f})" for c, p in zip(best["cat"][:3], best["pct"][:3]))
                   + ". Weakest: " + ", ".join(f"{c} ({p:.0f})" for c, p in zip(best["cat"][-2:], best["pct"][-2:])) + ".")
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
                                         "waiver comparisons. Select one to open his page.")
        comps = comparables(board, idx)
        styler, config = category_table(comps, cats, ctx["punt"], ctx["my_team"], ctx["abbrevs"], ctx["mode"])
        pick = show_table(styler, config, key=f"pp_comps_{pid}_{ctx['epoch']}", height="auto")
        if pick is not None and player_key(comps).iloc[pick] != r.get("player_id", r["name"]):
            open_player(player_key(comps).iloc[pick])

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
                 f'against {g["min"].mean():.1f} over the window.</p>')
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
        card_title("Stat lines", "Per game, per 36 minutes and per 100 possessions.")
        pace = ctx["pace"].get(r["pro_team"])
        lines = stat_lines(raw.head(1), r, ctx["pace"], {
            "proj": f"{ctx['seasons']['current']} ESPN projection",
            "last": f"{ctx['seasons']['last']} actual",
            "board": f"Board line: {SOURCES[ctx['source']]}",
            "per100": f"Per 100 possessions ({r['pro_team']} pace {pace:.1f})" if pace else "Per 100 possessions",
        })
        fmt = {c: "{:.1f}" for c in lines.columns if c not in ("Line", "FG%", "FT%")}
        fmt.update({"FG%": fmt_pct, "FT%": fmt_pct})
        st.dataframe(lines.style.format(fmt, na_rep="–"), hide_index=True, row_height=30, placeholder="–",
                     column_config={"Line": st.column_config.TextColumn(width="large")})
