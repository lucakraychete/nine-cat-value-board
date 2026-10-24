"""View state in the URL: query-parameter names, player links, table clicks.

Every setting widget is bound to a query parameter (Streamlit's
bind="query-params"), so the URL always holds the whole view: team,
stats source, punts, games discount, tab and the rest. The browser keeps
the URL in sync with the widgets and restores them on load. Opening a
player adds `player=<id>` (a new history entry), so browser Back returns
to the board with every setting and the same tab.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import urlencode

# Setting -> query parameter (also the widget key). Defaults stay out of the URL.
PARAMS = {
    "team": "team",
    "source": "source",
    "punt": "punt",
    "discount": "disc",
    "tab": "tab",
    "columns": "cols",
    "search": "q",
    "owner": "owner",
    "position": "pos",
    "exclude_to": "to",
    "pool_size": "pool",
    "season_games": "games",
}
PLAYER = "player"
TABS = ["Players", "My team", "Waiver wire", "Team categories"]


def player_href(pid, params: Mapping[str, str | Sequence[str]]) -> str:
    """Relative link to a player's page that keeps every current setting."""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if key == PLAYER:
            continue
        values = [value] if isinstance(value, str) else list(value)
        pairs += [(key, str(v)) for v in values]
    pairs.append((PLAYER, str(pid)))
    return "?" + urlencode(pairs)


def link_target(cells: Iterable[tuple[int, str]], links: Mapping[str, Sequence]):
    """The player id behind a clicked table cell, or None.

    cells is st.dataframe's selection (row position in the data passed,
    whatever the sort, and column name); links maps each name column to
    the player ids in row order. Clicks outside name columns are ignored.
    """
    for row, col in cells:
        ids = links.get(col)
        if ids is not None and 0 <= row < len(ids):
            return ids[row]
    return None
