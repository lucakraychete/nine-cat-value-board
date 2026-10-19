"""Plotly figures for the player page.

Each builder takes plain numbers plus a token dict from theme.tokens(),
and returns a Figure; nothing here touches Streamlit or the network.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from ninecat.board import FANTASY_TIERS
from ninecat.theme import BODY_FONT, rgba

ROLE_BANDS = [(33, 40, "Franchise"), (29, 33, "Starter"), (23, 29, "Rotation"),
              (16, 23, "Bench"), (10, 16, "Fringe")]


def _layout(fig: go.Figure, t: dict, height: int, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=margin or dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=t["ink"], size=12),
        showlegend=False, hoverlabel=dict(font_family=BODY_FONT),
    )
    return fig


def _hide_axes(fig: go.Figure) -> None:
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)


def ladder_position(rank: int) -> tuple[int, float]:
    """(tier index, x in 0..len(tiers)) for the tier ladder's marker.

    Tiers get equal-width segments; within his tier the marker sits in
    proportion to his rank across that tier's range (the last tier's
    open end is capped at 200).
    """
    lo = 0
    for i, (hi, _) in enumerate(FANTASY_TIERS):
        if lo < rank <= hi:
            top = min(hi, 200)
            return i, i + min(1.0, (rank - lo) / (top - lo))
        lo = hi
    return len(FANTASY_TIERS) - 1, float(len(FANTASY_TIERS))


def tier_ladder(rank: int, t: dict) -> go.Figure:
    n = len(FANTASY_TIERS)
    current, mx = ladder_position(rank)
    fig = go.Figure()
    for i, (_, name) in enumerate(FANTASY_TIERS):
        fig.add_shape(type="rect", x0=i + 0.02, x1=i + 0.98, y0=0, y1=1, line_width=0,
                      fillcolor=t["line"])
        fig.add_annotation(x=i + 0.5, y=-0.9, text=name, showarrow=False,
                           font=dict(size=12, color=t["ink"] if i == current else t["muted"]),
                           font_weight=600 if i == current else 400)
    fig.add_shape(type="rect", x0=mx - 0.03, x1=mx + 0.03, y0=-0.4, y1=1.4, line=dict(color=t["surface"], width=2),
                  fillcolor=t["s1"])
    fig.add_annotation(x=min(n - 0.15, max(0.15, mx)), y=2.2, text=f"<b>#{rank}</b>", showarrow=False,
                       font=dict(size=12, color=t["ink"]))
    _layout(fig, t, 92)
    fig.update_xaxes(range=[-0.05, n + 0.05])
    fig.update_yaxes(range=[-1.6, 2.8])
    _hide_axes(fig)
    return fig


def minutes_chart(points: Sequence[tuple[str, float, bool]], t: dict) -> go.Figure:
    """Minutes per game by season over the NBA role-tier bands; projection hollow and dashed."""
    fig = go.Figure()
    labels = [p[0] for p in points]
    for k, (a, b, name) in enumerate(ROLE_BANDS):
        fig.add_hrect(y0=a, y1=b, line_width=0, fillcolor=rgba(t["line"], 0.55 if k % 2 == 0 else 0.25),
                      layer="below")
        fig.add_annotation(xref="paper", x=1.0, xanchor="left", y=(a + b) / 2, text=name, showarrow=False,
                           font=dict(size=10, color=t["muted"]))
    for i in range(1, len(points)):
        fig.add_trace(go.Scatter(
            x=labels[i - 1:i + 1], y=[points[i - 1][1], points[i][1]], mode="lines",
            line=dict(color=t["s1"], width=2, dash="dash" if points[i][2] else "solid"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=labels, y=[p[1] for p in points], mode="markers+text",
        text=[f"<b>{p[1]:.1f}</b>" for p in points], textposition="top center",
        marker=dict(size=12, color=[t["surface"] if p[2] else t["s1"] for p in points],
                    line=dict(color=t["s1"], width=2)),
        hovertemplate="%{x}: %{y:.1f} min<extra></extra>"))
    _layout(fig, t, 240, margin=dict(l=30, r=70, t=8, b=30))
    fig.update_yaxes(range=[10, 41], tickvals=[10, 20, 30, 40], gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=t["muted"], size=10), fixedrange=True, zeroline=False)
    fig.update_xaxes(type="category", tickfont=dict(color=t["ink"], size=11), fixedrange=True, showgrid=False)
    return fig


def rolling_minutes_chart(games, t: dict) -> go.Figure:
    """Minutes by game (dots) with 5- and 10-game rolling averages and the window average."""
    avg = float(games["min"].mean())
    x = list(range(len(games)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=games["min"], mode="markers", marker=dict(size=5, color=rgba(t["muted"], 0.5)),
                             customdata=games["date"], hovertemplate="%{customdata}: %{y:.0f} min<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=games["r10"], mode="lines", line=dict(color=t["s2"], width=2),
                             hovertemplate="10-game avg %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=games["r5"], mode="lines", line=dict(color=t["s1"], width=2),
                             hovertemplate="5-game avg %{y:.1f}<extra></extra>"))
    fig.add_hline(y=avg, line=dict(color=t["muted"], dash="dot", width=1),
                  annotation_text=f"avg {avg:.1f}", annotation_position="top right",
                  annotation_font=dict(size=10, color=t["muted"]))
    _layout(fig, t, 220, margin=dict(l=30, r=12, t=8, b=30))
    hi = max(40.0, float(games["min"].max()))
    fig.update_yaxes(range=[0, hi + 2], tickvals=[0, 10, 20, 30, 40], gridcolor=t["line"],
                     tickfont=dict(color=t["muted"], size=10), fixedrange=True, zeroline=False)
    fig.update_xaxes(tickvals=[0, len(games) - 1], ticktext=[games["date"].iloc[0], games["date"].iloc[-1]],
                     tickfont=dict(color=t["muted"], size=10), showgrid=False, zeroline=False, fixedrange=True)
    return fig


def radar(labels: Sequence[str], player: Sequence[float], average: Sequence[float], t: dict,
          names: tuple[str, str] = ("Player", "Position average")) -> go.Figure:
    """Category percentiles: the player (solid) over his position's average (dashed)."""
    theta = [*labels, labels[0]]
    fig = go.Figure()
    for vals, color, dash, name in ((average, t["s2"], "dash", names[1]), (player, t["s1"], "solid", names[0])):
        r = [*vals, vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta, name=name, fill="toself", fillcolor=rgba(color, 0.14),
            line=dict(color=color, width=2, dash=dash), mode="lines+markers" if dash == "solid" else "lines",
            marker=dict(size=7, color=color, line=dict(color=t["surface"], width=2)),
            hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + name + "</extra>"))
    _layout(fig, t, 340, margin=dict(l=40, r=40, t=24, b=24))
    fig.update_polars(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], showticklabels=False,
                        gridcolor=t["line"], linecolor=t["line"]),
        angularaxis=dict(gridcolor=t["line"], linecolor=t["line"], tickfont=dict(size=11, color=t["ink"]),
                         rotation=90, direction="clockwise"))
    return fig


def beeswarm_levels(values: Sequence[float], width: float = 560, gap: float = 9) -> list[int]:
    """Stack dots that would overlap: alternate levels 0, 1, -1, 2, -2, ...

    Placed in descending value order, on a `width`-pixel axis, so the
    layout matches the original strip chart.
    """
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    xs = [10 + (v - lo) / span * (width - 20) for v in values]
    order = sorted(range(len(values)), key=lambda i: -values[i])
    placed: list[tuple[float, int]] = []
    levels = [0] * len(values)
    for i in order:
        lvl = 0
        while any(abs(px - xs[i]) < gap and pl == lvl for px, pl in placed):
            lvl = -lvl if lvl > 0 else -lvl + 1
        placed.append((xs[i], lvl))
        levels[i] = lvl
    return levels


def value_strip(values: Sequence[float], names: Sequence[str], ranks: Sequence[int], highlight: int,
                t: dict) -> go.Figure:
    """Everyone at his position by value, one dot each; him highlighted and labeled."""
    levels = beeswarm_levels(values)
    others = [i for i in range(len(values)) if i != highlight]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[values[i] for i in others], y=[levels[i] for i in others], mode="markers",
        marker=dict(size=8, color=rgba(t["muted"], 0.45)),
        customdata=[[names[i], ranks[i]] for i in others],
        hovertemplate="%{customdata[0]}: %{x:.2f} (#%{customdata[1]} overall)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[values[highlight]], y=[levels[highlight]], mode="markers+text", text=[f"<b>{names[highlight]}</b>"],
        textposition="top center", textfont=dict(color=t["ink"], size=12),
        marker=dict(size=14, color=t["s1"], line=dict(color=t["surface"], width=2)),
        hovertemplate=f"{names[highlight]}: %{{x:.2f}} (#{ranks[highlight]} overall)<extra></extra>"))
    lo, hi = min(values), max(values)
    top = max(abs(lv) for lv in levels) + 2
    if lo < 0 < hi:
        fig.add_shape(type="line", x0=0, x1=0, y0=-top + 0.6, y1=top, line=dict(color=t["line"], dash="dot"))
        fig.add_annotation(x=0, y=-top, text="pool avg", showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=t["muted"]))
    for x, anchor, text in ((0, "left", "lower value"), (1, "right", "higher value")):
        fig.add_annotation(xref="paper", x=x, xanchor=anchor, y=-top, yanchor="bottom", text=text,
                           showarrow=False, font=dict(size=10, color=t["muted"]))
    _layout(fig, t, 160)
    fig.update_yaxes(range=[-top, top + 1.5], visible=False, fixedrange=True)
    fig.update_xaxes(visible=False, fixedrange=True)
    return fig


def percentile_bars(items, t: dict, pos: str) -> go.Figure:
    """One row per metric: a 0-100 track, his percentile overall (orange) and at his position (blue)."""
    n = len(items)
    fig = go.Figure()
    ys = list(range(n))[::-1]
    for y, it in zip(ys, items):
        fig.add_shape(type="line", x0=0, x1=100, y0=y, y1=y, line=dict(color=t["line"], width=6))
        fig.add_shape(type="line", x0=50, x1=50, y0=y - 0.3, y1=y + 0.3, line=dict(color=rgba(t["muted"], 0.5), width=1))
        # Labels sit in the margins, so they're placed in paper coordinates.
        for x, anchor, top, bottom in ((-0.03, "right", it.label, it.sub), (1.03, "left", it.value, it.rank)):
            fig.add_annotation(xref="paper", x=x, y=y + 0.2, xanchor=anchor, text=f"<b>{top}</b>",
                               showarrow=False, font=dict(size=12, color=t["ink"]))
            fig.add_annotation(xref="paper", x=x, y=y - 0.22, xanchor=anchor, text=bottom,
                               showarrow=False, font=dict(size=10.5, color=t["muted"]))
    for attr, color, size, label in (("pct_pos", t["s2"], 10, f"among {pos}s"), ("pct_all", t["s1"], 12, "overall")):
        pts = [(getattr(it, attr), y, it.label) for y, it in zip(ys, items) if getattr(it, attr) is not None]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                marker=dict(size=size, color=color, line=dict(color=t["surface"], width=2)),
                customdata=[p[2] for p in pts],
                hovertemplate="%{customdata}: %{x:.0f}th percentile " + label + "<extra></extra>"))
    _layout(fig, t, 44 * n + 16, margin=dict(l=170, r=110, t=8, b=8))
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-0.7, n - 0.3], visible=False, fixedrange=True)
    return fig
