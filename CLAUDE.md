# CLAUDE.md

## Project

nine-cat-value-board: a fantasy basketball value board for ESPN head-to-head
category leagues (9-cat, or 8-cat when `EXCLUDE_TO=true`). It pulls league data
with the `espn-api` package, values players with per-game z-scores, and serves a
Streamlit UI. Public source; PolyForm Noncommercial 1.0.0 (free for personal,
noncommercial use).

Phase 1 is the current scope: ingest, valuation, app, tests, CI.

## Structure

```
ninecat/config.py      Settings dataclass loaded from .env; category lists
                       (COUNTING_CATS, PERCENT_CATS, NEGATIVE_CATS); pool/season defaults
ninecat/ingest.py      ESPN -> data/players.csv + data/meta.json; load_cached()
ninecat/valuation.py   value_players(), category_zscores(), percentage_impact(),
                       games discount, iterative pool, team_profile()
app.py                 Streamlit UI (board, my team, waiver wire); reads the cache only
tests/test_valuation.py  pytest suite for valuation.py (synthetic data, no network)
.github/workflows/tests.yml  runs pytest on 3.11-3.13
setup.sh / setup.ps1   venv + deps + .env from .env.example (bash / Windows PowerShell)
```

## Commands

```bash
./setup.sh                       # Windows PowerShell: .\setup.ps1
source .venv/bin/activate        # Windows Git Bash: .venv/Scripts/activate
python -m ninecat.ingest         # needs a filled-in .env and network
streamlit run app.py
pytest
```

## Data flow

ESPN -> `ingest.py` -> `data/` cache -> `value_players()` -> `app.py`.
The app never calls ESPN except via the Refresh button, which runs `ingest()`.
One row per player in `players.csv`: identity columns, `fantasy_team` ("FA" for
free agents), `team_id`, `stat_source` ("projected" or "total"), per-game
PTS/REB/AST/STL/BLK/3PM/TO/FGM/FGA/FTM/FTA, and projected `GP`.

## Valuation rules (keep these invariant)

- Z-scores use the mean and population std (ddof=0) of the **pool** rows only.
- FG%/FT% are scored as impact `makes - league_pct * attempts`, with
  `league_pct` attempt-weighted over the pool. Never z-score raw percentages.
- TO z-scores are negated so higher is always better.
- The pool is the top `pool_size` by value using **all** categories, iterated to
  a fixed point. Punts never change the pool; they only drop categories from
  `z_total` and `value`.
- The games discount scales positive totals by `GP / season_games` (clipped to
  [0, 1]) and leaves negative totals alone. If no player has GP data, there is no discount.
- A zero std yields z = 0, never NaN.
- `value_players()` does not mutate its input.

## Conventions

- Keep `valuation.py` pure: pandas/numpy in, DataFrame out, no I/O, no Streamlit.
- Only `config.py` reads environment variables. Only `ingest.py` talks to ESPN,
  and imports `espn_api` lazily inside `fetch_league()` so tests don't need it.
- Categories are data (`config.categories()`), not hard-coded branches. Add a
  category by extending the lists in `config.py`.
- Every change to `valuation.py` comes with a test in `tests/test_valuation.py`.
  Tests build synthetic DataFrames; never hit the network in tests.
- Python 3.11+, type hints, `from __future__ import annotations`.
- Match existing style: short docstrings explaining *why*, few inline comments.

## Secrets

- `.env` holds `ESPN_S2` and `SWID`, which are ESPN login cookies. It is gitignored
  and must never be committed, printed, logged, or written into code, tests or docs.
- `data/` is gitignored (league-specific cached data); only `data/.gitkeep` is tracked.
- The owner runs `git commit` and `git push` themselves; don't commit or push.
