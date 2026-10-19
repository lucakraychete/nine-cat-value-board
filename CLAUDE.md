# CLAUDE.md

## Project

nine-cat-value-board: a fantasy basketball value board for ESPN head-to-head
category leagues (9-cat, or 8-cat when `EXCLUDE_TO=true`). It pulls league data
with the `espn-api` package, values players with per-game z-scores, and serves a
Streamlit UI. Public source; PolyForm Noncommercial 1.0.0 (free for personal,
noncommercial use).

Current work: porting the original single-page JS board (`reference/original_app.html`,
gitignored, local only) into this app, reusing its formulas. Phase A (look, stats
sources, advanced stats, ADP, player pages) is in; Phase B (team category totals,
H2H matchup model, league impact) is next.

## Structure

```
ninecat/config.py      Settings dataclass loaded from .env; category lists
                       (COUNTING_CATS, PERCENT_CATS, NEGATIVE_CATS); pool/season defaults
ninecat/ingest.py      ESPN (+ ADP) -> data/players.csv + meta.json; calls nbastats for
                       last_season.csv, pace.json, game_logs.csv; load_cached()
ninecat/nbastats.py    nba.com via nba_api (last season, prior-season minutes, pace,
                       game logs) and BPM from Basketball Reference; pure frame builders
ninecat/rates.py       USG/TS/AST/TRB/STL/BLK/TOV% and PER from box-score totals (BBR formulas)
ninecat/lines.py       name_key join, stat_line() (proj / last / 60-40 blend), per36, per100
ninecat/valuation.py   value_players(), category_zscores(), percentage_impact(),
                       games discount, iterative pool, team_profile()
ninecat/board.py       build_board(): value on a line + per-36/per-100 values, ADP market
                       rank and gap, advanced z-scores, fantasy and role tiers
ninecat/player.py      player-page numbers: percentiles, position ranks, comparables
ninecat/charts.py      plotly figures for the player page (pure, no Streamlit)
ninecat/theme.py       design tokens (light/dark), z-score heat colors, app CSS
app.py                 Streamlit entry: header, controls, tabs, routing; reads the cache only
ui/                    Streamlit rendering: tables (components.py), player_page.py
.streamlit/config.toml theme: fonts, light and dark palettes
tests/                 one test file per ninecat module, plus test_app.py (AppTest smoke)
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

ESPN + nba.com -> `ingest.py` -> `data/` cache -> `load_cached()` (joins last season
by `name_key`) -> `build_board()` (`stat_line()` then `value_players()`) -> `app.py`.
The app never calls ESPN or nba.com except via the Refresh button, which runs `ingest()`.
One row per player in `players.csv`: identity columns, `fantasy_team` ("FA" for
free agents), `team_id`, `stat_source` ("projected" or "total"), `ADP` (blank =
undrafted), per-game MIN/PTS/REB/AST/STL/BLK/3PM/TO/FGM/FGA/FTM/FTA, and projected `GP`.
`last_season.csv` adds `L_*` per-game columns, `L_G`, `prev_MIN` and `adv_*` rates.
nba.com sources are optional: a failure keeps the old file and is recorded in
`meta.json["sources"]`.

## Valuation rules (keep these invariant)

- Z-scores use the mean and population std (ddof=0) of the **pool** rows only.
- FG%/FT% are scored as impact `makes - league_pct * attempts`, with
  `league_pct` attempt-weighted over the pool. Never z-score raw percentages.
- TO z-scores are negated so higher is always better.
- The pool is the top `pool_size` by **per-game** value (before the games
  discount) using **all** categories, iterated to a fixed point. Punts never
  change the pool; they only drop categories from `z_total` and `value`.
- The games discount works from replacement level: `replacement` is the
  `pool_size`-th best per-game value (`z_total`, after punts), and
  `value = replacement + (z_total - replacement) * GP / season_games`, with GP
  clipped to [0, season_games] and a missing or zero GP taken as 70 games.
  A player exactly at replacement level is unaffected by missed games.
- A zero std yields z = 0, never NaN.
- `value_players()` does not mutate its input.

## Conventions

- Keep `valuation.py` pure: pandas/numpy in, DataFrame out, no I/O, no Streamlit.
  The same goes for every `ninecat` module except `ingest.py` and the `fetch_*`
  functions in `nbastats.py`. Streamlit code lives in `app.py` and `ui/` only.
- Only `config.py` reads environment variables. Only `ingest.py` talks to ESPN,
  and imports `espn_api` lazily inside `fetch_league()` so tests don't need it.
  Only `nbastats.py` talks to nba.com and Basketball Reference, importing
  `nba_api` lazily inside its `fetch_*` functions.
- Every new `ninecat` module has its own `tests/test_<module>.py`.
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
