"""Design tokens and CSS for the app, in light and dark.

The palette, type and z-score shading come from the original single-page
board: Barlow Condensed headings, IBM Plex Sans body, an orange accent.
.streamlit/config.toml sets Streamlit's own theme to the same values;
this module covers what the theme can't: cell shading, chart colors and
the custom HTML pieces (tiles, chips, tags).
"""

from __future__ import annotations

import math

DISPLAY_FONT = '"Barlow Condensed", "Arial Narrow", system-ui, sans-serif'
BODY_FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif'

TOKENS = {
    "light": {
        "bg": "#f4f6f8", "surface": "#ffffff", "ink": "#17212c", "muted": "#5b6876",
        "line": "#dde2e8", "accent": "#c2410c", "accent_soft": "#fbe7dc",
        "good": "#17824f", "bad": "#bf3a2b", "s1": "#c2410c", "s2": "#2f6fb0",
    },
    "dark": {
        "bg": "#0e1319", "surface": "#151c24", "ink": "#e5ebf1", "muted": "#8e9aa7",
        "line": "#263241", "accent": "#f0874a", "accent_soft": "#3a2418",
        "good": "#3cbf82", "bad": "#e86b5c", "s1": "#d96e32", "s2": "#5b94cf",
    },
}

# Shading saturates at |z| = 2.5, at 42% of the full color.
HEAT_CAP = 2.5
HEAT_MAX_ALPHA = 0.42


def tokens(mode: str | None) -> dict[str, str]:
    return TOKENS["dark" if mode == "dark" else "light"]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def heat_alpha(z: float) -> float:
    """Opacity of a z-score's shading: rounded to whole percent like the original."""
    return round(min(abs(z), HEAT_CAP) / HEAT_CAP * HEAT_MAX_ALPHA * 100) / 100


def heat(z, mode: str | None = "light") -> str:
    """CSS background for a z-score cell: green above average, red below."""
    if z is None or not isinstance(z, (int, float)) or math.isnan(z):
        return ""
    t = tokens(mode)
    return f"background-color: {rgba(t['good'] if z >= 0 else t['bad'], heat_alpha(z))}"


def app_css(mode: str | None) -> str:
    t = tokens(mode)
    return f"""
:root {{
  --bg: {t['bg']}; --surface: {t['surface']}; --ink: {t['ink']}; --muted: {t['muted']};
  --line: {t['line']}; --accent: {t['accent']}; --accent-soft: {t['accent_soft']};
  --good: {t['good']}; --bad: {t['bad']}; --s1: {t['s1']}; --s2: {t['s2']};
  --display: {DISPLAY_FONT}; --body: {BODY_FONT};
}}
.block-container {{ max-width: 1240px; padding-top: 3.6rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ font-family: var(--display) !important; letter-spacing: .01em; }}
h1 {{ font-size: 34px !important; font-weight: 700 !important; line-height: 1 !important; padding: 0 !important; }}
.eyebrow {{ font-size: 11px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--accent); }}
.sub, .note {{ color: var(--muted); font-size: 13px; max-width: 75ch; margin: 4px 0 0; }}
.sub {{ font-size: 14px; max-width: 70ch; }}
.num {{ font-variant-numeric: tabular-nums; }}
.pos {{ color: var(--good); font-weight: 600; }} .neg {{ color: var(--bad); font-weight: 600; }}

/* widget labels read like the original's small caps control labels */
[data-testid="stWidgetLabel"] p {{ font-size: 11px !important; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); }}
/* tabs in the display face */
[data-testid="stTab"] p {{ font-family: var(--display) !important; font-size: 19px !important; font-weight: 600;
  letter-spacing: .02em; }}
[data-testid="stTab"]:not([aria-selected="true"]) p {{ color: var(--muted); }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: var(--accent) !important; height: 3px !important; }}

.tag {{ display: inline-block; font-size: 10.5px; font-weight: 600; letter-spacing: .04em; padding: 1px 6px;
  border-radius: 4px; margin-left: 8px; vertical-align: 6px; border: 1px solid currentColor; font-family: var(--body); }}
.tag.out {{ color: var(--bad); }} .tag.dtd {{ color: var(--muted); }} .tag.new {{ color: var(--accent); }}

.pphead h2 {{ font-size: clamp(30px, 7vw, 40px); font-weight: 700; line-height: 1.05; margin: 2px 0 0; padding: 0; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.tier {{ font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--surface); color: var(--ink); }}
.tier.t-superstar, .tier.t-star {{ background: var(--accent); border-color: var(--accent); color: var(--surface); }}
.tier.t-starter {{ border-color: var(--accent); color: var(--accent); }}

.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr)); gap: 8px; }}
.tiles div {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  display: grid; gap: 2px; min-width: 0; overflow-wrap: anywhere; }}
.tiles b {{ font-family: var(--display); font-size: clamp(20px, 5.5vw, 26px); font-weight: 700; line-height: 1;
  font-variant-numeric: tabular-nums; color: var(--ink); }}
.tiles span {{ font-size: 12px; color: var(--muted); }}

.catranks {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(64px, 1fr)); gap: 6px; margin: 4px 0 8px; }}
.catranks div {{ border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; display: grid; }}
.catranks span {{ font-size: 11px; color: var(--muted); font-weight: 600; letter-spacing: .04em; }}
.catranks b {{ font-variant-numeric: tabular-nums; font-size: 15px; color: var(--ink); }}

.legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
.callout {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 12px; display: grid; gap: 4px; }}
.callout b {{ font-family: var(--display); font-size: 18px; font-weight: 600; }}
.status {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 13px; color: var(--muted); }}
.hero {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 6px 0; }}
.hero div {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  display: grid; gap: 2px; min-width: 0; }}
.hero b {{ font-family: var(--display); font-size: clamp(24px, 6.5vw, 32px); font-weight: 700; line-height: 1;
  font-variant-numeric: tabular-nums; color: var(--ink); }}
.hero b.pos {{ color: var(--good); }} .hero b.neg {{ color: var(--bad); }}
.hero span {{ font-size: 12px; color: var(--muted); }}
.catchip {{ font-size: 12px; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--bg); font-variant-numeric: tabular-nums; color: var(--ink); }}
.sw {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); }}
.ltable {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }}
.ltable table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; color: var(--ink); }}
.ltable th, .ltable td {{ padding: 6px 8px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }}
.ltable th {{ font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.ltable th.l, .ltable td.l {{ text-align: left; }}
.ltable td small {{ display: block; color: var(--muted); font-size: 11px; }}
.ltable tr.mine td {{ box-shadow: inset 0 0 0 999px {rgba(t['accent'], 0.08)}; }}
.card-title {{ font-family: var(--display); font-weight: 700; font-size: 22px; margin: 0; letter-spacing: .01em; color: var(--ink); }}
.card-h4 {{ font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 6px 0 4px; }}
"""
