# CLAUDE.md

## Project

nine-cat-value-board: a fantasy basketball value board for ESPN head-to-head
category leagues (9-cat, or 8-cat when `EXCLUDE_TO=true`). It pulls league data
with the `espn-api` package, values players with per-game z-scores, and serves a
Streamlit UI. Public source; PolyForm Noncommercial 1.0.0 (free for personal,
noncommercial use).

Current work: `PLAN.md` (six parts, stop after each). Earlier phases ported the
original single-page JS board (`reference/original_app.html`, gitignored, local only),
reusing its formulas: look, stats sources, advanced stats, ADP, player pages, team
category totals, H2H matchup model, league impact, rest-of-season waivers. Part 1
(live data layer, background refresh on boot, "This season" source) and Part 2
(clickable names, every setting in the URL, waiver cards, build card) are in. Next:
Part 3 (this week, schedule and playoffs).

## Structure

```
ninecat/config.py      Settings dataclass loaded from .env; category lists
                       (COUNTING_CATS, PERCENT_CATS, NEGATIVE_CATS); pool/season defaults
ninecat/ingest.py      the refresh: ESPN players (projection + T_/R7_/R15_/R30_ splits,
                       injuries, ADP), transactions (append + dedupe), live matchups,
                       matchup calendar, projection snapshots; calls nbastats; per-source
                       status in meta; atomic writes; --history; load_cached(), load_table()
ninecat/nbastats.py    nba.com via nba_api (this season's box-score logs -> season.csv,
                       last season, prior-season minutes, pace, schedule, history) and BPM
                       from Basketball Reference; with_retry(); pure frame builders
ninecat/schedule.py    NBA schedule helpers (games between dates, per period, remaining)
                       and the ESPN matchup-period calendar rebuilt from weeks
ninecat/rates.py       USG/TS/AST/TRB/STL/BLK/TOV% and PER from box-score totals (BBR formulas)
ninecat/lines.py       name_key join, stat_line() (proj / last / 60-40 blend / this season),
                       per36, per100
ninecat/valuation.py   value_players(), category_zscores(), percentage_impact(),
                       games discount, iterative pool, team_profile()
ninecat/board.py       build_board(): value on a line + per-36/per-100 values, ADP market
                       rank and gap, advanced z-scores, fantasy and role tiers
ninecat/matchup.py     H2H model: team-week means/variances, category odds, correlated
                       matchup win probability, outlooks against every team
ninecat/league.py      team category totals, build_league(): rosters, worth to every
                       team, league impact, value over replacement; waivers, fit notes
ninecat/player.py      player-page numbers: percentiles, position ranks, comparables
ninecat/charts.py      plotly figures for the player page (pure, no Streamlit)
ninecat/urlstate.py    query-parameter names for every setting, player_href(), link_target()
ninecat/theme.py       design tokens (light/dark), z-score heat colors, app CSS
app.py                 Streamlit entry: header, status line, controls, tabs, routing;
                       renders from the cache only
ui/                    Streamlit rendering: tables (components.py), player_page.py,
                       league_views.py (my team, waiver wire, team categories),
                       refresh.py (background refresh job, once-per-server boot guard)
.streamlit/config.toml theme: fonts, light and dark palettes
tests/                 one test file per ninecat module, test_refresh.py, and test_app.py
                       (AppTest smoke, including the boot refresh)
.github/workflows/tests.yml  runs pytest on 3.11-3.13
setup.sh / setup.ps1   venv + deps + .env from .env.example (bash / Windows PowerShell)
```

## Commands

```bash
./setup.sh                       # Windows PowerShell: .\setup.ps1
source .venv/bin/activate        # Windows Git Bash: .venv/Scripts/activate
python -m ninecat.ingest         # needs a filled-in .env and network
python -m ninecat.ingest --history   # past seasons for the model, once (~20 s)
streamlit run app.py             # refreshes in the background on start (AUTO_REFRESH)
pytest
```

## Data flow

ESPN + nba.com -> `ingest.py` -> `data/` cache -> `load_cached()` (joins last season
and this season by `name_key`) -> `build_board()` (`stat_line()` then `value_players()`)
-> `app.py`. The app never calls ESPN or nba.com while rendering: `ui/refresh.py` runs
`ingest()` in a background thread once per server start (if `AUTO_REFRESH`) and on
"Refresh now", and the status line offers a reload when meta's `fetched_at` changes.

`data/` files:
- `players.csv`, one row per player: identity, `fantasy_team` ("FA" for free agents),
  `team_id`, `stat_source` ("projected" or "total"), `ADP` (blank = undrafted),
  `injury_status`, `injured`, `return_date`, per-game MIN/PTS/REB/AST/STL/BLK/3PM/TO/
  FGM/FGA/FTM/FTA and projected `GP` (the projection), then ESPN's season-to-date
  `T_*` (+ `T_GP`, `T_GS`) and last 7/15/30-day `R7_*`, `R15_*`, `R30_*` averages.
- `last_season.csv`: `L_*` per-game, `L_G`, `prev_MIN`, `adv_*` rates (weekly).
- `season.csv`: this season from nba.com logs: `S_*` per-game, `S_G`, `S_team`,
  `S_G_7D`/`S_MIN_7D`, `S_G_15D`/`S_MIN_15D`, `S_MIN_L5`, `age`.
- `game_logs.csv`: box score per player-game (`nba.LOG_COLUMNS`), this season and last.
- `schedule.csv`: NBA regular season (date ET, game_id, home, away; weekly; ESPN's
  pro schedule as fallback). NBA Cup knockout games appear once their teams are known.
- `league_schedule.csv`: per matchup period and team: start, end, playoff, opponent_id.
- `live_matchup.csv`: current period, per team: opponent, category W-L-T, stat totals.
- `transactions.csv`: msg_id, date (UTC), team_id, action (FA add / Waiver add /
  Drop / Trade), player_id, player, from_team_id (trades), msg_type. Append-only, deduped.
- `snapshots/espn_proj_YYYY-MM-DD.csv`: the day's ESPN projections (last refresh wins).
- `history/logs_<season>.csv.gz`, `totals_<season>.csv.gz`: 2017-18 to last season.
- `meta.json`: league, teams, `current_period`, `regular_periods`, `season_start`,
  `calendar_note`, and `sources` = {key: {"status", "updated"}} where `updated` is the
  last success (`ingest.SOURCE_LABELS` lists the keys).

## Data rules (keep these invariant)

- ESPN rosters and projections are the only required source. Every other source fails
  soft: keep the old file, record the failure, keep the last success time.
- Write cache files with `ingest.write_csv` / `write_json` (atomic), since the app may
  read while the background refresh writes.
- nba.com calls go through `nbastats.with_retry` (timeout, one retry with backoff, a
  pause after each call). Logs and schedule keep regular-season game ids (`002…`) only.
- Last season's nba.com numbers and the NBA schedule refetch only when a week old;
  last season's logs and history seasons are fetched once.
- "This season" (`S_`, falling back to ESPN's `T_`) needs 3+ games, else the blend;
  its GP stays the projection's so the games discount means the same for every source.
- Matchup period days: ESPN's own scoring periods for periods that have started;
  later ones rebuilt as Monday-Sunday weeks with the two weeks around the All-Star
  break (longest February gap) merged. Scoring period 1 is opening night.

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

## Matchup model rules (from the original; keep these invariant)

- A team-week counts each player's games (3.5 a week) x availability (GP/82,
  missing GP = 70) x lineup slot weight (top 10 by value 1.0, next 3 0.75, rest 0.4).
- Counting variance = games x (mean + (0.3 mean)^2); percentage variance =
  1.6 p(1-p) / attempts. Category odds = Phi(edge / sd), with TO reversed.
- Matchup win = majority of categories, normal approximation with the CORR table
  widening the variance plus 1/12; the 7-digit Abramowitz-Stegun `ncdf` is kept.
- Season numbers are over 20 weeks. Adding a player drops the team's lowest-value
  player sharing a position; losing one picks up the best free agent at it.
  League impact averages the other teams; punts don't enter the matchup model.

## Conventions

- Keep `valuation.py` pure: pandas/numpy in, DataFrame out, no I/O, no Streamlit.
  The same goes for every `ninecat` module except `ingest.py` and the `fetch_*`
  functions in `nbastats.py`. Streamlit code lives in `app.py` and `ui/` only.
- Only `config.py` reads environment variables. Only `ingest.py` talks to ESPN,
  and imports `espn_api` lazily inside functions (`fetch_league()`, the schedule
  fallback) so tests don't need it.
  Only `nbastats.py` talks to nba.com and Basketball Reference, importing
  `nba_api` lazily inside its `fetch_*` functions.
- Every new `ninecat` module has its own `tests/test_<module>.py`.
- Categories are data (`config.categories()`), not hard-coded branches. Add a
  category by extending the lists in `config.py`.
- Every change to `valuation.py` comes with a test in `tests/test_valuation.py`.
  Tests build synthetic DataFrames; never hit the network in tests.
- Python 3.11+, type hints, `from __future__ import annotations`.
- Match existing style: short docstrings explaining *why*, few inline comments.

## Navigation and URL state (keep these invariant)

- Every setting widget is bound to the URL: `key=urlstate.PARAMS[...]`,
  `bind="query-params"` (app.py's `**URL`), including `st.tabs`. Add new settings to
  `PARAMS`. Widgets with `format_func` store the *label* in the URL, so keep labels
  stable across seasons (no years in them).
- A player name opens his page in the same tab via `open_player(pid)`, which sets
  `?player=` (a browser history entry, so Back works) and bumps `session_state.epoch`.
  Player tables use `show_table(..., links={column: [player ids in row order]})`
  (single-cell selection; clicks outside name columns do nothing) with
  `link_style()` on name cells, and every table key includes the epoch so a stale
  selection can't reopen a player after Back. Names in cards use `name_button()`
  with a key starting `pname_` (headline) or `plink_` (inline).
- Lists of players are tables; summaries are tiles, cards and charts.

## Secrets

- `.env` holds `ESPN_S2` and `SWID`, which are ESPN login cookies. It is gitignored
  and must never be committed, printed, logged, or written into code, tests or docs.
- `data/` is gitignored (league-specific cached data); only `data/.gitkeep` is tracked.
- The owner runs `git commit` and `git push` themselves; don't commit or push.
