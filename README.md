# nine-cat-value-board

[![tests](https://github.com/lucakraychete/nine-cat-value-board/actions/workflows/tests.yml/badge.svg)](https://github.com/lucakraychete/nine-cat-value-board/actions/workflows/tests.yml)

A player value board for **ESPN head-to-head category fantasy basketball** leagues.
It pulls your league's rosters, free agents and projections, values every
player across the scoring categories, and shows a sortable board, your team's
category strengths and weaknesses, and the best waiver pickups.

Works for standard **9-cat** leagues (FG%, FT%, 3PM, REB, AST, STL, BLK, PTS, TO)
and **8-cat** leagues without turnovers (`EXCLUDE_TO=true`, the default).

## What it does

- **Per-game z-scores.** Each category is standardized so a block and a point
  are on the same scale. Turnovers count against a player.
- **Volume-weighted FG% and FT%.** A 100% shooter on 2 attempts doesn't help
  your team's percentage much; a 45% shooter on 20 attempts hurts it a lot.
  Percentages are scored as *impact*: makes above what a league-average
  shooter would produce on the same attempts.
- **Punting.** Toggle categories you're giving up; they drop out of player
  values so the board reflects your build.
- **Games-missed discount.** What a player adds above replacement level (the
  last player in the rostered pool, who'd start when he sits) is scaled by
  projected games out of 82, so injury-prone stars aren't overrated.
- **Iterative top-130 pool.** Averages are computed against the players who
  actually get rostered (10 teams x 13 spots by default), not the whole NBA.
  Because the pool depends on the values and vice versa, it iterates until
  the pool stops changing.
- **Stats source.** Value players on ESPN's projection, last season's actual
  stats, or a blend (60% projection, 40% last season for anyone who played
  15+ games).
- **Advanced stats.** USG%, TS%, AST%, TRB%, STL%, BLK%, TOV%, PER and BPM,
  plus per-36-minute and per-100-possession value ranks: who produces when
  he plays, regardless of role.
- **Market check.** ESPN average draft position, market rank and the gap
  between market rank and value rank.
- **Player pages.** Select any player for his fantasy tier and NBA role,
  category profile against his position, comparable players, minutes trend
  and efficiency ranks.

- **Head-to-head matchup model.** Each team's week is simulated category by
  category, with weekly variance and correlated categories (rebounds, blocks
  and FG% tend to move together, as do points, threes and FT%), to get a
  chance of winning each category and each weekly matchup.
- **League impact.** Every player's worth to every team in extra category
  wins over a 20-week season, his impact for the average team, and his
  impact over the best free agent at his position.

The app has four tabs: **Players** (everyone, filterable, with a Categories
or Advanced column set and league impact), **My team** (projected standing,
category outlook, matchup odds against every team, what each player is
worth to you, best pickups), **Waiver wire** (every free agent tried
against every possible drop) and **Team categories** (each team's
per-game category totals and league ranks, plus projected standings).
It follows your system's light or dark mode; switch in the app's settings menu.

Data comes from ESPN (your league, projections, ADP), nba.com via
[nba_api](https://github.com/swar/nba_api) (last season's stats, team pace,
game logs) and one request to Basketball Reference for BPM. The advanced
rates are computed from nba.com box scores with Basketball Reference's
published formulas.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/lucakraychete/nine-cat-value-board.git
cd nine-cat-value-board
```

Then run the setup script for your shell. Both create `.venv`, install
dependencies, and copy `.env.example` to `.env`.

**macOS/Linux** (or Git Bash/WSL on Windows):

```bash
./setup.sh
```

**Windows PowerShell:**

```powershell
.\setup.ps1
# if script execution is blocked:
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Edit `.env` with your league details:

| Variable     | What it is |
|--------------|------------|
| `LEAGUE_ID`  | The number after `leagueId=` in your ESPN league URL |
| `YEAR`       | Season as ESPN labels it: the 2025-26 season is `2026` |
| `ESPN_S2`    | Login cookie, needed for private leagues (see below) |
| `SWID`       | Login cookie, needed for private leagues; also identifies your team |
| `EXCLUDE_TO` | `true` for 8-cat (no turnovers), `false` for 9-cat |

Then pull data and start the app:

```bash
source .venv/bin/activate       # Windows Git Bash: source .venv/Scripts/activate
                                # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m ninecat.ingest        # caches rosters, free agents, projections to data/
streamlit run app.py
```

Re-run the ingest (or press **Refresh from ESPN** in the app) whenever you
want fresh rosters and projections.

## Finding ESPN_S2 and SWID

ESPN has no public API keys. Private leagues are read with two cookies from
your logged-in browser session:

1. Log in to [fantasy.espn.com](https://fantasy.espn.com) and open your league.
2. Open developer tools (F12, or right-click → Inspect).
3. Go to cookies:
   - **Chrome/Edge:** Application tab → Storage → Cookies → `https://fantasy.espn.com`
   - **Firefox:** Storage tab → Cookies → `https://fantasy.espn.com`
   - **Safari:** enable the Develop menu first, then Storage → Cookies
4. Copy the value of `espn_s2` (a long string) into `ESPN_S2`.
5. Copy the value of `SWID` into `SWID`, **including the curly braces**,
   e.g. `{1A2B3C4D-...}`.

Public leagues work without either cookie, but you'll need `SWID` for the app
to open on your team automatically.

> [!WARNING]
> **Never commit `.env`.** `ESPN_S2` and `SWID` are login credentials for your
> ESPN account: anyone with them can act as you. `.env` is in `.gitignore`;
> keep it there. If you ever push them by accident, log out of ESPN on all
> devices to invalidate the cookies, and remove the file from git history.

## Development

```bash
pytest                          # valuation tests, also run by GitHub Actions on every push/PR
```

Project layout:

```
ninecat/
  config.py      settings from .env, category definitions
  ingest.py      ESPN (+ nba.com via nbastats.py) -> data/ cache
  nbastats.py    nba.com and Basketball Reference fetches
  rates.py       advanced rate stats from box-score totals
  lines.py       stats sources (projection, last season, blend), per-36, per-100
  valuation.py   z-scores, percentage impact, punts, games discount, iterative pool
  board.py       the board: value plus per-36/per-100, market gap, tiers
  matchup.py     head-to-head matchup model
  league.py      team totals, league impact, waivers
  player.py      player-page numbers
  charts.py      plotly charts
  theme.py       colors, fonts, CSS
app.py           Streamlit entry point
ui/              Streamlit rendering (tables, player page)
tests/           pytest suite, synthetic data, no network
```

## Disclaimer

Uses the unofficial ESPN Fantasy API via [espn-api](https://github.com/cwendt94/espn-api).
Not affiliated with ESPN.

## License

Free for personal, noncommercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted.
