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
- **Games-missed discount.** Value is scaled by projected games played, so
  injury-prone stars aren't overrated.
- **Iterative top-130 pool.** Averages are computed against the players who
  actually get rostered (10 teams x 13 spots by default), not the whole NBA.
  Because the pool depends on the values and vice versa, it iterates until
  the pool stops changing.

The app has three tabs: **Player board** (everyone, filterable),
**My team** (your roster and category profile with league ranks), and
**Waiver wire** (free agents, sortable by any category).

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
  ingest.py      ESPN -> data/players.csv + data/meta.json
  valuation.py   z-scores, percentage impact, punts, games discount, iterative pool
app.py           Streamlit UI
tests/           pytest suite for valuation.py
```

## Disclaimer

Uses the unofficial ESPN Fantasy API via [espn-api](https://github.com/cwendt94/espn-api).
Not affiliated with ESPN.

## License

Free for personal, noncommercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use is not permitted.
