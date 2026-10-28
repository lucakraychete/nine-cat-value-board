"""The Model tab: how the rest-of-season model was built and how well it does.

Everything comes from data/models/report.json (written by `python -m
ninecat.model train`) plus the live scoring of saved snapshots and the
matchup log; nothing here trains or predicts.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ninecat import charts
from ninecat.model.evaluate import INTERVAL
from ninecat.model.live import brier, calibration
from ninecat.urlstate import PARAMS as P
from ui.components import DASH, card, card_title, html

URL = {"bind": "query-params"}
PLOTLY_CONFIG = {"displayModeBar": False}
HEADLINE = ["Blend 60/40", "Interim (app today)", "Shrinkage", "Boosted"]


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _best_css(frame: pd.DataFrame, t: dict) -> pd.DataFrame:
    """Bold the lowest error in each column."""
    css = pd.DataFrame("", index=frame.index, columns=frame.columns)
    for c in frame.columns:
        col = pd.to_numeric(frame[c], errors="coerce")
        if col.notna().any():
            css.loc[col.idxmin(), c] = f"font-weight: 700; color: {t['good']}"
    return css


def render_model(report: dict, scores: pd.DataFrame, matchup_log: pd.DataFrame, ctx: dict, loaded: bool) -> None:
    t = ctx["tokens"]
    if not report:
        html('<div class="callout"><span class="eyebrow">Rest-of-season model</span><b>No model trained yet</b>'
             '<span class="note">Run <code>python -m ninecat.ingest --history</code> once, then '
             '<code>python -m ninecat.model train</code> (a few minutes). Until then the Overview\'s ROS column and '
             'trades use the interim line. How it works: docs/MODEL.md.</span></div>')
        _live(scores, matchup_log, t)
        return

    test = pd.DataFrame(report["backtest"]["test"])
    val = pd.DataFrame(report["backtest"]["validate"])
    rank = pd.DataFrame(report["value_rank"]["test"])
    chosen = {"boosted": "Boosted", "shrinkage": "Shrinkage"}[report["chosen"]]
    verdict = report["test_verdict"]
    spear = rank.groupby("model")["spearman"].mean()
    status = ("In use: the Overview's ROS column, trades and the Model (ROS) stats source read its projections."
              if loaded else "Trained, but no predictions yet: press Refresh now (the refresh predicts).")
    default = ("It beat the 60/40 blend on the test season, so Model (ROS) is the default stats source."
               if report.get("model_beats_blend") else
               "It did not beat the 60/40 blend on the test season, so Blend stays the default stats source.")
    html(f'<div class="callout"><span class="eyebrow">Rest-of-season model · trained {escape(report["created"][:10])}'
         f'</span><b>{chosen} model: per-game production and games played for the rest of the season</b>'
         f'<span class="note">{status} {default} Backtested on past seasons with the season held out: trained on '
         f'{report["seasons"]["train"][0]} to {report["seasons"]["train"][-1]}, chosen on '
         f'{report["seasons"]["validate"][0]}, scored once on {report["seasons"]["test"][0]}. The live model is '
         'then refit on every season. Full write-up: docs/MODEL.md.</span></div>')
    b, i = verdict["vs Blend 60/40"], verdict["vs Interim (app today)"]
    html(f"""<div class="tiles">
        <div><b class="pos">{_pct(b['mae_change'])}</b><span>error vs the 60/40 blend (test season, averaged over categories)</span></div>
        <div><b class="pos">{_pct(i['mae_change'])}</b><span>error vs the interim line the app used before</span></div>
        <div><b>{spear[chosen]:.2f}</b><span>rank correlation, top-200 8-cat value (blend {spear['Blend 60/40']:.2f}, interim {spear['Interim (app today)']:.2f})</span></div>
        <div><b>{report['checkpoints']['train']:,}</b><span>training checkpoints</span></div></div>""")

    with card("model_backtest"):
        card_title("Backtest", "Average miss per game over the rest of the season, weighted by the minutes each "
                   "player went on to play (a miss on a starter counts more than on a reserve). Lower is better; "
                   "the best in each column is green. A checkpoint is the point in the season (games played) the "
                   "prediction was made from.")
        c1, c2 = st.columns([1, 2])
        season = c1.segmented_control("Season", ["Test", "Validation"], default="Test", required=True,
                                      key=P["model_season"], **URL)
        ks = sorted(test["k"].unique())
        k = c2.segmented_control("Checkpoint (games played)", ["All", *map(str, ks)], default="All", required=True,
                                 key=P["model_k"], **URL)
        table = test if season == "Test" else val
        if k != "All":
            table = table[table["k"] == int(k)]
        grid = table.pivot_table(index="model", columns="cat", values="mae", aggfunc="mean", sort=False)
        grid = grid[report["eval_cats"]]
        styler = grid.style.apply(lambda f: _best_css(f, t), axis=None).format(
            {c: "{:.3f}" for c in grid.columns if "%" not in c} | {c: "{:.4f}" for c in grid.columns if "%" in c})
        st.dataframe(styler, row_height=30)
        pivot = rank.pivot_table(index="model", columns="k", values="spearman", sort=False)
        pivot.columns = [f"k={c}" for c in pivot.columns]
        pivot["mean"] = pivot.mean(axis=1)
        html('<div class="card-h4">Does it rank players right? (test season)</div><p class="note">Spearman rank '
             'correlation between predicted and actual rest-of-season 8-cat value (turnovers excluded) for the '
             '200 most valuable players at each checkpoint. 1 is a perfect ranking.</p>')
        st.dataframe(pivot.style.apply(lambda f: _best_css(-f, t), axis=None).format("{:.3f}"), row_height=30)

    left, right = st.columns(2)
    with left, card("model_k"):
        card_title("How fast each stat settles", "Shrinkage: rate = (minutes so far × his rate + k × prior) ÷ "
                   "(minutes so far + k). k is how much evidence the prior is worth; after k minutes (shot attempts "
                   "for FG% and FT%) his own numbers count as much as the prior. Big k = mostly noise early.")
        k_rows = pd.DataFrame([{"Stat": s, "k": v["k"], "Units": report["k_units"][s], "Games at 30 min": v["games_at_30"]}
                               for s, v in report["k"].items()]).sort_values("Games at 30 min")
        st.dataframe(k_rows.style.format({"k": "{:.0f}", "Games at 30 min": "{:.1f}"}), hide_index=True,
                     row_height=30)
        av = report["availability"]
        html(f'<p class="note">Games played: availability (share of his team\'s games he plays) shrinks the same '
             f'way toward his last two seasons, with k = {av["k_team_games"]:.0f} team games; newcomers start at '
             f'{av["newcomer"]:.0%}.</p>')
    with right, card("model_boost"):
        bv = report["boosting_vs_shrinkage"]
        card_title("Did gradient boosting earn its place?",
                   "The boosting model learns what shrinkage gets wrong. It was kept only if it beat shrinkage on "
                   "the validation season by more than noise: the 95% bootstrap interval (resampling players) for "
                   "the average change had to be entirely below zero.")
        ov = bv["overall"]
        html(f'<div class="hero"><div><b class="{"pos" if ov["high"] < 0 else ""}">{_pct(ov["change"])}</b>'
             f'<span>average change in error, 95% interval {_pct(ov["low"])} to {_pct(ov["high"])}: '
             f'{"kept" if report["chosen"] == "boosted" else "not kept"}</span></div></div>')
        by = pd.DataFrame(bv["by_cat"]).T
        st.dataframe(by.style.format("{:+.1%}"), row_height=30)

    with card("model_calibration"):
        cov = pd.DataFrame(report["coverage_test"])
        overall = cov[cov["k"] == 0]
        nominal = INTERVAL[1] - INTERVAL[0]
        card_title("Are the ranges honest?", f"Each projection carries an {nominal:.0%} range from the model's "
                   "validation-season errors at that checkpoint and minutes level. On the test season, this is how "
                   f"often the real outcome landed inside it; well calibrated is {nominal:.0%} (the dashed line). "
                   "VALUE is per-game 8-cat value. Slightly under means the ranges are a little narrow.")
        st.plotly_chart(charts.coverage_bars(list(overall["cat"]), list(overall["coverage"]), nominal, t),
                        theme=None, config=PLOTLY_CONFIG, key="model_coverage")

    _live(scores, matchup_log, t)


def _live(scores: pd.DataFrame, matchup_log: pd.DataFrame, t: dict) -> None:
    with card("model_live"):
        card_title("This season, as it happens", "The model's and ESPN's daily projections, scored against what "
                   "each player has done in his games since (3+ games). Days with both are scored on the same "
                   "players. Run python -m ninecat.model score for the same table in the terminal.")
        if scores is None or scores.empty:
            html('<p class="note">Nothing to score yet: snapshots need a few games played after their date.</p>')
        else:
            show = scores.rename(columns={"source": "Projection", "day": "Made on", "players": "Players",
                                          "games_after": "Games since"})
            show["Projection"] = show["Projection"].map({"model": "Model", "espn": "ESPN"})
            st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns
                                            if c not in ("Projection", "Made on", "Players")}),
                         hide_index=True, row_height=30)
        score, n = brier(matchup_log)
        cal = calibration(matchup_log)
        html('<div class="card-h4">Matchup model: weekly win chances vs results</div>'
             f'<p class="note">Each refresh logs the matchup model\'s win chance for every team\'s current matchup '
             f'(the first time it sees the week) and the final result. Brier score: '
             f'{"–" if n == 0 else f"{score:.3f} over {n} matchups"} (0 is perfect, a coin flip scores 0.250). '
             'This tests the matchup model\'s assumed 30% game-to-game swing, 1.6× shooting variance and category '
             'correlations.</p>')
        if not cal.empty:
            st.plotly_chart(charts.calibration_plot(list(cal["p_win"]), list(cal["won"]), list(cal["n"]), t),
                            theme=None, config=PLOTLY_CONFIG, key="model_matchup_cal")
            st.dataframe(cal.rename(columns={"bin": "Win chance", "p_win": "Average chance", "won": "Won", "n": "Matchups"})
                         .style.format({"Average chance": "{:.0%}", "Won": "{:.0%}"}), hide_index=True, row_height=30)
        else:
            html(f'<p class="note">{DASH} No finished matchups logged yet.</p>')
