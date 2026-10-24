# Next phase: live season data, clickable UI, weekly tools, transactions, ML projections, trades

Save this whole prompt as `PLAN.md` in the repo root and work from it. Read `CLAUDE.md` first and follow it. Keep its invariants: pure `ninecat` modules, no network in tests, a test file for every new module, secrets never printed or committed, and **never run `git commit` or `git push`**. Use `reference/original_app.html` as the reference for formulas and layout wherever this plan says "as in the original".

The NBA 2026-27 regular season has started. The app should now run on real stats, refresh itself every time it starts, and project the rest of the season with a model that is validated against past seasons.

Do the six parts in order. **Stop after each part** and give me a short summary: what changed, how to test it, the test count, and anything that failed soft. Wait for me before starting the next part.

**Out of scope for now:** the Breakouts & risks tab (no tab, no placeholder). It comes back after the ML work.

---

## Part 1: Data layer and refresh on boot

Everything later depends on this part. Every new source must **fail soft**: if a source fails, keep its last good file, record the failure in `meta.json["sources"]`, and keep the app working.

### 1.1 Current-season stats (nba.com via `nba_api`, in `nbastats.py`)
- **Full box-score game logs** for the current season, one row per player per game: date, NBA team, opponent, home/away, MIN, PTS, REB, AST, STL, BLK, FG3M, FGM, FGA, FTM, FTA, TOV, and starter if available. Right now `game_log_frame()` keeps only minutes. Keep every column above, and keep the minutes chart working.
- **Season-to-date per-game totals** per player (LeagueDashPlayerStats or derived from the logs), plus age.
- Derived per player: games played, minutes over the last 7 and 15 days, and minutes in the last 5 games.

### 1.2 ESPN extras (`ingest.py`; verify each espn-api attribute against the installed version before relying on it)
- Keep the projection, and **also store** ESPN's season-to-date averages and its last-7, last-15 and last-30 splits (`{year}_total`, `{year}_last_7`, and so on) as separate prefixed columns. Don't replace the projection.
- **Injury status**, plus injury detail and expected return date if espn-api exposes them.
- **League schedule:** each matchup period's dates (from the scoring-period mapping), and each team's opponent per period.
- **Live matchup:** current category totals for both sides of every matchup this period (box scores or scoreboard for H2H category leagues).
- **Transactions:** `league.recent_activity()` (page through it with offset), normalized to date, team, action (FA add, waiver add, drop, trade, and so on) and player. **Append to `data/transactions.csv`, deduplicated**, so the history builds up beyond ESPN's recent window. The first run backfills as far back as ESPN allows.
- **Projection snapshots:** each refresh, append a dated copy of ESPN's projections to `data/snapshots/espn_proj_YYYY-MM-DD.csv` (one per day). Part 5 uses these to test the model against ESPN later.

### 1.3 NBA schedule
- The full 2026-27 regular-season schedule (prefer nba.com's league schedule endpoint) with date, home and away teams, cached to `data/schedule.csv`. Refresh it about once a week, not on every boot.
- Pure helpers: games per NBA team between two dates, games per team per ESPN matchup period, and **remaining** games per team from today.

### 1.4 Historical data for the model (fetched once, not every boot)
- Player game logs for the 2017-18 through 2025-26 regular seasons, one league game-log call per season, saved to `data/history/` as gzipped CSV. Add a `python -m ninecat.ingest --history` command. A normal refresh never refetches seasons it already has.
- Per-season player totals with age for the same seasons.

### 1.5 Refresh on boot
- On server start, the first session kicks off a **background** refresh (thread, guarded with `st.cache_resource` so it runs once per server start, not once per browser tab). The app shows the cached data immediately with an "Updating data…" status line, then shows a "New data ready, reload" button or reruns when the refresh finishes. Never block the first render on nba.com.
- The setting `AUTO_REFRESH` (default `true`) lives in `.env.example` and `config.py`. Keep the "Refresh now" button.
- nba.com hangs and rate-limits, so give every call a timeout, one retry with backoff, and a short pause between calls.
- The status line shows when each source was last updated, and flags sources running on an old saved copy.

### 1.6 New stats sources on the board
Add a **"This season"** source (season-to-date actual per game) to the stats-source control. Players with fewer than 3 games fall back to the blend. Leave the existing sources as they are. Part 5 adds the model source.

### Tests for Part 1
Pure builders on synthetic frames: game-log frame keeps every stat column; transaction dedupe and append; games-per-period from a hand-built schedule; remaining games; snapshot naming; the "This season" source and its fallback; the boot-refresh guard runs once.

---

## Part 2: Navigation and layout fixes

### 2.1 Clicking a player opens his page, everywhere
Right now a player page only opens when you select the row's checkbox at the far left of a `st.dataframe`. That's not discoverable, and clicking the name does nothing. Fix it:
- **Clicking a player's name** in any table or list opens his player page in the **same tab**. This covers the board (both column sets), My team roster and adds, Waiver wire, comparables on the player page, and every new table in Parts 3, 4 and 6.
- Check what the installed Streamlit version actually supports, and pick the cleanest option that meets this: a link column pointing to `?player=<id>` that opens in the same tab, cell selection, or HTML tables (rendered with `st.html` or markdown) with name links where a table doesn't need sorting. Tell me which option you chose and why.
- Keep all settings in the URL query params (team, source, punts, games discount, tab), so opening a player page and pressing Back keeps everything, even if navigation reloads the page.
- "← Back to board" returns to the tab I came from.

### 2.2 Less "everything is a table"
Follow the original's layout. Tables are for lists of players; summaries should be tiles, cards and charts.
- **My team:** header tiles (projected standing, expected record, categories per week, category wins over the season), a category outlook bar chart, matchup win-chance bars against each team, and Lock / Swing / Weak chips. The roster and adds stay as tables.
- **Waiver wire:** the top 5 pickups as cards (player, gain, best drop, top category gains as chips, fit note), then the full table below.
- **Team categories:** keep the heat table and show "Your build" as a callout card.
- Player names in cards are clickable as in 2.1.

### Tests for Part 2
The query-param round trip (settings to URL to settings), a player link builder, and AppTest smoke tests checking that each tab renders and that a `?player=` URL renders the player page.

---

## Part 3: This week, schedule and playoffs (as in the original)
- **This week's matchup card** on My team (`liveMatchupHtml` in the original): current ESPN totals plus projected remaining games for each roster, per-category final projection and win chance, Safe / Swing / Out of reach status, overall win chance, projected category score, and games left on each side. Players listed OUT get 0 games.
- **Upcoming matchups** (`extrasHtml`): the next weeks against my real opponents, with games per side, win chance and swing categories.
- **Waiver wire, "Rest of season / This week / next weeks" switch** (`waivers` and `dropOptions` in weekly mode against my actual opponent): every free agent against every drop, plus the injured-list option, showing the best drop and the runner-up.
- **Fantasy playoff games:** weeks of Mar 8–14, Mar 15–21 and Mar 22–28, 2027, set in config with those defaults. Show them as a column on the board and the waiver wire, and as a player-page tile with the league average.
- **Recommended adds** on My team (`recommendAdds` and `recommendHtml`): waiver adds and trade targets that keep my locks and upgrade my swing and weak categories. Trade targets get a "Build trade" button that does nothing until Part 6.
- **Player page:** add "This season" and "Last 15 days" rows to the stat-lines table. Add a rolling-average chart for any stat (pick MIN, PTS, REB and so on) from the full game logs.

### Tests for Part 3
Team-week with schedule-based games; an OUT player contributes 0; the live-matchup blend of banked totals and the remaining projection (counting stats and percentages); playoff game counts; weekly `vsOne` against a single opponent.

---

## Part 4: League activity (transactions inside the app)
Add a new **League activity** tab built from `data/transactions.csv` (as in the original's `renderActivity`):
- Most added and most dropped over the last 7 days (with a 7 / 14 / 30 day switch).
- **Manager profiles:** adds, drops and trades per team, the date of the last move, and "chasing": the categories where the players each team added have the highest average z-score.
- **Recent transactions feed** with filters by team and by move type.
- On each player page, his transaction history in this league.
- Every name is clickable. Free agents who were just dropped get a "Recently dropped" tag on the waiver wire.

This tab tracks moves only. **Don't add any way to make moves** (adds, drops, trade offers) through ESPN.

### Tests for Part 4
Counting adds and drops within a window, manager profile counts and "chasing", and dedupe across overlapping refreshes.

---

## Part 5: Rest-of-season projections with machine learning
Goal: predict each player's **per-game production and games played for the rest of 2026-27**, validated on past seasons. Interviewers will ask about this part, so keep every choice explainable and write it down in `docs/MODEL.md`.

### 5.1 Setup (new `ninecat/model/` package, pure except a training CLI)
- **Unit:** player-game. Predict **minutes per game** and **per-minute rates** (PTS, REB, AST, STL, BLK, 3PM, FGA, FTA, and FG% and FT% as makes over attempts) separately, then multiply them back into per-game lines.
- **Checkpoints:** for each historical season, cut at the point each player has played k = 3, 5, 10, 20 and 40 games. Features use only games **before** the cut. The target is his per-game average over the **rest of that season** (minimum 5 remaining games).
- **Features:** season-to-date rates and minutes, the last 5 and 10 games, the previous season's and the season before's rates and minutes, age, games played so far versus team games, and the change in starter rate. Rookies get a prior of the pool average at their minutes level.
- **No leakage:** split by season (train 2017-18 to 2023-24, validate 2024-25, test 2025-26). Never split randomly across rows.

### 5.2 Models, simplest first
1. **Baselines:** last season, season to date, and the current fixed 60/40 blend.
2. **Empirical-Bayes shrinkage** (main model): `rate = (n · current + k · prior) / (n + k)`, where n is minutes played so far, with a learned **k per stat**, fit on the training seasons. Write down what k means: stats with a big k (steals, blocks, FT%) settle slowly; stats with a small k (minutes, points) settle fast.
3. **Gradient boosting challenger:** scikit-learn `HistGradientBoostingRegressor` predicting the residual of model 2. No extra heavy dependencies; add `scikit-learn` to requirements.
- Keep model 3 only if it beats model 2 on the **validation** season. Report everything on the **test** season once.

### 5.3 Games played rest of season
Remaining NBA schedule games times projected availability. Availability is games played versus team games so far, shrunk toward the player's previous two seasons. Players listed OUT lose the games before their expected return date when ESPN provides one; otherwise they lose the current week. Keep it simple and documented.

### 5.4 Evaluation and report
- Per category: MAE and RMSE per game, weighted by remaining minutes, for each model at each checkpoint.
- The fantasy-relevant check: Spearman correlation between the predicted and the actual rest-of-season 8-cat **value rank** (using `value_players`) for the top 200.
- **Prediction intervals** from residual quantiles by checkpoint and minutes bucket. These replace the hand-set percentages in `league.uncertainty()`.
- Save to `data/models/` (fitted parameters plus `report.json`). Add a `python -m ninecat.model train` CLI. Training is never part of boot; boot only predicts, which must take under 2 seconds.
- A **Model** tab in the app with the backtest table (baselines versus models, by checkpoint), the learned k per stat, and a calibration chart. Plain-language notes throughout.

### 5.5 Plug it in
- Add the stats source **"Model (rest of season)"** and make it the default **only if** it beats the 60/40 blend on the test season. Otherwise keep it as an option and tell me.
- For this source, the games discount must use projected **remaining** games against remaining team games, not GP/82, and season numbers use the remaining matchup weeks rather than a fixed 20.
- Player page: projected rest-of-season line with its interval, next to ESPN's projection and this season's line.

### 5.6 Testing against the real world from now on
- Score the model's predictions and the daily ESPN projection snapshots against what actually happens, in a script I can run mid-season and at season end.
- Each week, log the matchup model's pre-week win probabilities and the final results, and report the Brier score and a calibration table on the Model tab. This tests the matchup model's assumed constants (the 30% swing, 1.6× shooting variance and CORR table).

### Tests for Part 5
Checkpoint building with no leakage (no feature uses a game on or after the cut); the shrinkage formula and k fit on synthetic data with a known answer; per-minute to per-game recombination; percentage stats from makes over attempts; the remaining-games logic; and a report shape test. All synthetic, with no network.

---

## Part 6: Trades (as in the original)
### 6a: Real impact
- The **trade calculator** (`applyTrade` and `evalTrade`): pick a partner, players to give and to get; roster sizes are settled with drops and free-agent pickups; show the category change for both sides, projected rosters after the trade, and a roster legality check (`rosterFits`).
- **"Who to trade, and to whom"** surplus table on My team and the Trades tab (`sellMap` and `sellTable`), with "Shop him" opening the calculator.
- The **trade finder** (`findGapTrades`) with gap, one-category and punt modes, a veto cap setting, a partner filter, and variety (at most 2 deals per incoming player).
- On the player page, **"Trades for [player]"** for players on other teams. Wire up "Build trade" from Part 3.

### 6b: How other managers see it
- Perceived value (`buildPerception`): name value shifting from ADP to this season's headline rank as games are played, headline stats, positional fit, recency from last-15-day minutes, and loss aversion. Show it beside the real impact in the calculator and finder, with a verdict, pitch text and a likely counteroffer.
- **Logged offers:** accept, decline or cancel; Undo right after logging; Remove on past entries. Loss aversion per manager is replayed from the log, so removing an entry reverses it exactly. Store the log in `data/trade_log.csv`, not browser storage.

### Tests for Part 6
Trade settlement keeps roster sizes; roster legality; finder pruning and variety; lambda replay after a remove; and the season-phase blend.

---

## Throughout
- Update `CLAUDE.md` (structure, data flow, new invariants) and `README.md` (new commands, `AUTO_REFRESH`, `--history`, model training) at the end of each part.
- Keep the speed reasonable: cache the league model and model predictions per data version. Tell me if any view takes more than 2 seconds.
- If something in this plan doesn't match what the APIs actually return, stop and tell me rather than guessing.
- Don't commit or push.
