# Rest-of-season model

What the projection model predicts, what it learns from, and every choice
behind it, in plain language. Code lives in `ninecat/model/`; the plan is
PLAN.md Part 5.

**In one paragraph.** Partway through a season, the model predicts each
player's per-game line and games played for the rest of it. Minutes per
game get a small regression; each per-minute rate is the player's rate so
far shrunk toward his last two seasons by an amount learned per stat; a
gradient-boosting layer corrects what shrinkage systematically misses. On
the held-out 2025-26 season it misses by 25% less than the app's old 60/40
blend and 13% less than the interim line, and ranks the top 200 players
better (rank correlation 0.73 vs 0.65 and 0.71). Every projection carries
an 80% range from its backtest errors.

```
python -m ninecat.ingest --history   # once: past seasons' game logs (2015-16 on)
python -m ninecat.model train        # ~2.5 minutes: fit, score once on the test season, save data/models/
python -m ninecat.model score        # any time: model and ESPN snapshots vs what has happened since
```

Training is never part of the app's boot. The app's refresh (on start and
on "Refresh now") only predicts with the saved model, about 0.8 seconds
for the whole league, in the background.

## The question

Partway through a season, what will a player average **per game** over the
**rest of the season**? Fantasy decisions (trades, pickups, drops) are about
the games left, not the games already banked.

## Data

nba.com game logs, one row per player per game, for the 2015-16 through
2025-26 regular seasons (`data/history/`, fetched once with
`python -m ninecat.ingest --history`). Each row has minutes, points,
rebounds, assists, steals, blocks, threes, turnovers, and makes and
attempts for field goals and free throws, plus date, team and opponent.
Season totals add each player's age.

| Split | Seasons | Checkpoints |
|---|---|---|
| Train | 2017-18 to 2023-24 | 15,028 |
| Validate | 2024-25 | 2,261 |
| Test | 2025-26 | 2,261 |
| Priors only | 2015-16, 2016-17 | – |

Seasons are split whole, never row by row: a model must never see the
season it's graded on. The validation season picks between models; the
test season is scored **once**, at the end. 2015-16 and 2016-17 are only
there so the first training seasons have "last season" and "the season
before" to look back on. The shortened 2019-20 (bubble) and 2020-21
(72-game) seasons stay in training: it's still NBA basketball, just fewer
games after the cut.

## Checkpoints

For every season and player, cut his season after his k-th game, for
k = 3, 5, 10, 20 and 40, provided he played at least 5 more games after
the cut. Each cut is one training example:

- **Features** use only his games before the cut: season to date, his last
  5 and last 10 games, last season, the season before, age, and how many of
  his team's games he's played (games so far over the league's average team
  games before the cut day, which also works for players traded mid-season).
- **Target**: his per-game line over every game after the cut.

A test checks that changing any game on or after the cut never changes a
feature.

**Starts.** nba.com's game log doesn't record who started. Fetching real
starts would take about 11,000 box-score requests, so a player counts as a
starter in a game when he's among his team's five players with the most
minutes that game. It's a common proxy; it can't be checked against the
truth offline (ESPN's games-started count only exists for this season).

## What gets predicted

Minutes per game, and per-minute rates for points, rebounds, assists,
steals, blocks, threes, turnovers, field-goal attempts and free-throw
attempts, with FG% and FT% as makes over attempts. The per-game line is
minutes times each rate (makes are percentage times attempts). Splitting
minutes from rates matters because they move for different reasons:
minutes jump with a role change, while a player's per-minute production is
much steadier.

## Baselines

What the app could do without a model, each scored the same way:

- **Season to date**: his line so far.
- **Last season**: last season's line (season to date for anyone without one).
- **Blend 60/40**: the app's "Blend" stats source, 60% preseason
  projection + 40% last season. It never looks at this season.
- **Interim (app today)**: the app's current rest-of-season line,
  `overview.ros_line()`: this season weighted GP / (GP + 20), the rest on
  the preseason projection.

**The preseason projection is a stand-in.** ESPN's preseason projections
for past seasons don't exist, so the backtests use a simple projection in
their place: last season's and the season before's per-minute rates,
weighted 2:1 by minutes, at last season's minutes per game. Players with
neither season (mostly rookies) get the training pool's average per-minute
rates for players at their minutes per game so far (in bands: under 10,
10-15, 15-20, 20-25, 25-30, 30-35, 35+). This matters for reading the
numbers: in the live app the prior is ESPN's real projection, which likely
beats this stand-in, so the backtest understates how good the Blend and
Interim lines are in practice and the model's real-world margin may be
smaller than the backtest says. Age adjustment of the stand-in is tried
in Step 2 and kept only if it helps on validation.

## Scoring

Per category, the mean absolute error per game between prediction and what
happened over the rest of the season, weighted by the minutes he went on to
play (a miss on a 34-minute starter matters more than one on a 6-minute
reserve). RMSE is reported alongside. FG% and FT% are scored as
percentages, over players who took the shot after the cut.

### Baselines on the validation season (2024-25)

Weighted MAE per game, averaged over the five checkpoints
(`python -m ninecat.model baselines`):

| | MIN | PTS | REB | AST | STL | BLK | 3PM | TO | FG% | FT% |
|---|---|---|---|---|---|---|---|---|---|---|
| Season to date | 3.48 | 2.62 | 1.01 | 0.73 | 0.31 | 0.23 | 0.45 | 0.44 | .062 | .109 |
| Last season | 4.00 | 2.71 | 0.92 | 0.72 | 0.25 | 0.17 | 0.41 | 0.36 | .038 | .070 |
| Blend 60/40 | 4.07 | 2.72 | 0.95 | 0.74 | 0.25 | 0.17 | 0.42 | 0.37 | .035 | .060 |
| Interim (app today) | **3.38** | **2.27** | **0.81** | **0.62** | **0.22** | **0.15** | **0.35** | **0.32** | **.032** | .065 |

The interim line already beats the other baselines almost everywhere:
mixing this season with a prior is better than either alone. Season to
date is the worst early (k = 3) and becomes competitive by k = 20-40; the
prior-only lines get worse as the season goes on, since they never learn
from it. That pattern is exactly what shrinkage formalizes.

## Model 2: empirical-Bayes shrinkage (the main model)

`ninecat/model/shrinkage.py`; run `python -m ninecat.model shrinkage`.

### Per-minute rates

Each stat's rest-of-season rate is his rate so far, pulled toward his prior
in proportion to how little he's played:

    rate = (n × so far + k × prior) / (n + k)

where n is his minutes so far. **k is how many minutes of evidence the
prior is worth.** After k minutes his own numbers count as much as the
prior; before that, the prior dominates. k is learned once per stat on the
training seasons by picking the value that minimizes the error (squared,
weighted by the minutes he went on to play) of the rate over the rest of
the season. FG% and FT% shrink the same way, with n and k counted in shot
attempts. The prior is the stand-in projection above, scaled by an age
curve (below).

This is the statistically principled version of what the app's interim
line does by hand: the interim line uses one fixed weight (20 games) for
every stat, while shrinkage learns that each stat settles at its own speed.

### Learned k (training seasons 2017-18 to 2023-24)

| Stat | k | Equivalent games at 30 min |
|---|---|---|
| FGA | 139 minutes | 4.6 |
| AST | 169 minutes | 5.6 |
| REB | 205 minutes | 6.8 |
| PTS | 257 minutes | 8.6 |
| 3PM | 260 minutes | 8.7 |
| BLK | 332 minutes | 11.1 |
| FTA | 361 minutes | 12.0 |
| TO | 461 minutes | 15.4 |
| STL | 722 minutes | 24.1 |
| FG% | 212 attempts | 19.3 |
| FT% | 78 attempts | 27.8 |

How to read it: **shot volume, assists, rebounds and points settle fast**
(after 5 to 9 games of starter minutes his own rate counts as much as the
prior); these reflect role and skill, which show up immediately. **Steals,
turnovers, FG% and FT% settle slowly** (15 to 28 games): a hot shooting
week or a few lucky steals say little, so the prior holds on far longer.
FT% has the smallest k in attempts (78) but the biggest in games, because
players take only a few free throws a game.

### Age

Players change from what their last two seasons say in a predictable
direction: under-23s score and assist about 10–15% more per minute than
their prior, players 32 and up about 5% less scoring and 10% fewer free
throws. The model multiplies the prior's per-minute rates by a factor per
age band (under 23, 23–25, 26–28, 29–31, 32+) and stat, learned on the
training seasons as actual over predicted. Rookies' pool prior isn't
age-adjusted. On validation it helped points, assists and threes and cost
almost nothing elsewhere (steals 0.206 vs 0.203), so it's kept.

### Minutes per game

Minutes move for different reasons than rates: a role change, a trade, a
teammate's injury. They get their own model: a linear regression per
checkpoint, weighted by the games he has left, on his minutes so far, over
his last 5 games, last season's minutes, whether he has a prior season, the
change in his start rate (last 5 games vs the season so far) and age. The
learned weights move as the season goes: after 3 games the prior gets 0.31
and this season 0.48 (the rest is a constant pulling toward the league
middle); by 40 games this season gets 0.76 (0.43 season, 0.33 last 5) and
the prior 0.12. Each extra year of age is worth about 0.16 fewer minutes.

### Validation (2024-25)

Weighted MAE per game, averaged over the five checkpoints:

| | MIN | PTS | REB | AST | STL | BLK | 3PM | TO | FG% | FT% |
|---|---|---|---|---|---|---|---|---|---|---|
| Interim (app today) | 3.38 | 2.27 | 0.81 | 0.62 | 0.22 | 0.15 | 0.35 | 0.32 | .032 | .065 |
| Blend 60/40 | 4.07 | 2.72 | 0.95 | 0.74 | 0.25 | 0.17 | 0.42 | 0.37 | .035 | .060 |
| Shrinkage | 2.68 | 1.99 | 0.73 | 0.58 | 0.20 | 0.15 | 0.33 | 0.29 | .032 | .055 |
| **Shrinkage + age** | **2.68** | **1.96** | **0.73** | **0.56** | 0.21 | **0.15** | **0.32** | **0.29** | **.032** | **.055** |

Shrinkage beats the interim line in every category at every checkpoint,
with two exceptions: blocks at 20 games (0.145 vs 0.143) and FG%, where
the two are within a thousandth everywhere. The biggest gain
is minutes (3.38 → 2.68), from the dedicated minutes model; points improve
by 0.31 a game.

## Model 3: gradient-boosting challenger

`ninecat/model/boosting.py`; run `python -m ninecat.model boosting`.

Shrinkage treats every player the same way: one k per stat, one prior
formula. It can't notice, say, that a guard whose last 10 games show a big
jump in shot attempts tends to keep them, or that a high-minute big's FG%
regresses less than a reserve's. The challenger looks for patterns like
that in **what shrinkage gets wrong**.

For each target (minutes per game, each per-minute rate, FG%, FT%), a
scikit-learn `HistGradientBoostingRegressor` learns the shrinkage model's
residual on the training seasons, from everything known at the cut: every
window's rates (season so far, last 5 and 10, last two seasons), start
rates, age, games share, the checkpoint, and the shrinkage model's own
prediction. Its prediction is shrinkage plus the learned correction, so
where it has nothing to add it stays close to shrinkage.

Settings are deliberately modest and fixed (200 trees, learning rate 0.05,
at most 15 leaves, at least 40 players per leaf, L2 regularization 1.0),
each fit weighted like the shrinkage fit (minutes left for rates, attempts
for percentages, games left for minutes). Training takes about a minute,
offline; predicting a whole league takes a fraction of a second.

### Did it earn its place?

The rule set in advance: keep the challenger only if it beats shrinkage on
the validation season by more than noise. "Noise" is measured with a
bootstrap over players: resample the 2024-25 players with replacement
1,000 times (each player's checkpoints stay together, since they share his
season), recompute both models' errors, and take the 95% interval of the
relative change. The challenger had to have the whole interval for the
average across categories below zero.

Relative change in minutes-weighted MAE, boosted vs shrinkage, validation
2024-25 (negative = boosted better):

| | MIN | PTS | REB | AST | STL | BLK | 3PM | TO | FG% | FT% |
|---|---|---|---|---|---|---|---|---|---|---|
| Change | −1.0% | −3.8% | −3.6% | −3.9% | −1.6% | −8.1% | −2.1% | −5.4% | −9.8% | −7.1% |
| 95% low | −3.9% | −7.3% | −6.9% | −7.3% | −4.6% | −12.2% | −5.5% | −8.9% | −13.3% | −11.4% |
| 95% high | +1.9% | −0.5% | −0.2% | −0.3% | +1.4% | −3.6% | +1.5% | −1.7% | −6.5% | −2.7% |

**Average over categories: −4.6%, 95% interval [−6.1%, −3.1%]. It's kept.**
The gains are clearest on FG%, blocks, FT% and turnovers, the slow-settling
stats where shrinkage's one-size k leaves the most on the table; minutes,
steals and threes improve too but within noise. One caveat: at the 40-game
checkpoint boosting is slightly worse than shrinkage on minutes,
rebounds, steals and threes (it gains most early in the season, when
shrinkage has least to go on).

Weighted MAE, averaged over checkpoints:

| | MIN | PTS | REB | AST | STL | BLK | 3PM | TO | FG% | FT% |
|---|---|---|---|---|---|---|---|---|---|---|
| Shrinkage + age | 2.676 | 1.961 | 0.734 | 0.562 | 0.206 | 0.149 | 0.318 | 0.292 | .0323 | .0553 |
| **Boosted** | **2.661** | **1.894** | **0.714** | **0.542** | **0.204** | **0.138** | **0.314** | **0.277** | **.0294** | **.0516** |

## Games played

`ninecat/model/games.py`. Rest-of-season games = his NBA team's remaining
games × his availability, minus what an injury costs.

**Availability** is the share of his team's games he plays. It's shrunk
exactly like the rates: his share so far (games over his team's games
before the cut), pulled toward his share over his last two seasons, with
k learned on the training seasons as **21 team games**. So after about a
quarter of the season his own attendance counts as much as his history.
Each past season is measured against its own length, so the 72-game
2020-21 season counts fairly. Players with no previous season start at the
training pool's 65%.

**Injuries.** A player listed OUT loses every one of his team's games
before ESPN's expected return date when ESPN gives one; otherwise he loses
the rest of the current week (Monday to Sunday). Day-to-day players lose
nothing. Kept deliberately simple: ESPN's return dates are often vague,
and anything cleverer would be guessing.

Test season, mean absolute error in games left:

| Checkpoint (games played) | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Model | **14.1** | **13.3** | **11.9** | **9.4** | **5.4** |
| Last two seasons only | 14.7 | 14.3 | 13.4 | 11.1 | 7.2 |
| This season so far only | 16.7 | 15.0 | 13.6 | 10.1 | 5.6 |

Games are hard to predict (injuries are mostly unforecastable), and the
error is large early. One known bias: a checkpoint needs 5+ games after
the cut, so players who went down for the season right after it aren't in
the data, and fitted availability runs a little high.

## Evaluation

Run once, by `python -m ninecat.model train`, on the **test season
(2025-26)**, which no model or setting had seen. The full tables are in
`data/models/report.json` and on the app's Model tab.

### Weighted MAE per game, test season, averaged over checkpoints

| | MIN | PTS | REB | AST | STL | BLK | 3PM | TO | FG% | FT% |
|---|---|---|---|---|---|---|---|---|---|---|
| Season to date | 3.73 | 2.75 | 1.01 | 0.78 | 0.32 | 0.22 | 0.46 | 0.45 | .062 | .106 |
| Last season | 4.56 | 3.17 | 0.99 | 0.81 | 0.25 | 0.19 | 0.42 | 0.43 | .041 | .072 |
| Blend 60/40 | 4.55 | 3.11 | 1.00 | 0.83 | 0.24 | 0.18 | 0.43 | 0.43 | .037 | .064 |
| Interim (app before) | 3.78 | 2.55 | 0.85 | 0.68 | 0.22 | 0.15 | 0.37 | 0.36 | .033 | .065 |
| Shrinkage | **2.87** | 2.19 | 0.73 | **0.60** | 0.20 | 0.15 | **0.34** | 0.33 | .034 | .059 |
| **Boosted (chosen)** | 2.89 | **2.15** | **0.73** | 0.60 | **0.19** | **0.14** | 0.34 | **0.30** | **.030** | **.054** |

The chosen model's error, averaged over categories, is **25% lower than the
60/40 blend's** and **13% lower than the interim line's**. On the test season
the boosting layer's gain over shrinkage is smaller than on validation:
clear on turnovers, FG%, FT%, blocks and steals, a wash on minutes,
assists and threes. It was chosen on validation, and the test season is
reported as it came out.

### Does it rank players right?

The fantasy-relevant check: at each checkpoint, value every player's
predicted and actual rest-of-season line on 8-cat value (turnovers left
out, `value_players`, no games discount), take the 200 most valuable by
actual value, and compute the Spearman rank correlation between predicted
and actual value. 1 would be a perfect ranking.

| Test season | k=3 | k=5 | k=10 | k=20 | k=40 | Mean |
|---|---|---|---|---|---|---|
| Season to date | .566 | .625 | .680 | .714 | .664 | .650 |
| Last season | .637 | .650 | .651 | .617 | .578 | .627 |
| Blend 60/40 | .669 | .681 | .678 | .628 | .577 | .647 |
| Interim (app before) | .693 | .723 | .738 | .714 | .661 | .706 |
| Shrinkage | .729 | .754 | .750 | **.739** | .677 | .730 |
| **Boosted (chosen)** | **.738** | **.764** | **.753** | .731 | **.680** | **.733** |

### Did it beat the 60/40 blend?

Yes, on both counts set in advance: lower average error **and** higher rank
correlation on the test season. So "Model (rest of season)" is the app's
default stats source (`report.json` records this as `model_beats_blend`;
the app reads it).

### Prediction intervals

Every projection carries an **80% range**. The ranges are the 10th and
90th percentiles of the model's errors on the validation season, by
checkpoint and by predicted minutes (under 15, 15–25, 25–32, 32+; a band
with fewer than 40 players falls back to the whole checkpoint), per
category and for per-game 8-cat value. They're used three ways:

- on the player page, per category;
- in the Overview, as a ROS range: across the board, ROS category wins
  added move almost linearly with per-game value × the share of the season
  left, so the value range is converted with that fitted slope;
- for the "impact range" tile on the player page, which used to be a
  hand-set ±12/25/40% (`league.uncertainty()`); with a model it's the
  measured range converted the same way.

Honest check, on the test season: how often did the real outcome land
inside its range? Target 80%. Result 74–78% by category (per-game value
74%). The ranges are **slightly too narrow**, because they were measured on
validation errors of a model trained without the validation season, and
the test season was a bit harder to predict. Widening them by ~10% would
be the fix if it persists in live scoring.

## In the app

- **The model's line everywhere ROS matters.** `overview.ros_line()` keeps
  its name and signature but now returns the model's line (with its
  projected availability) wherever the model has one; the old formula is
  `interim_ros_line()`, used for players the model has no line for and
  when no model is trained. So the Overview's ROS column ("ROS (model)",
  with an "ROS 80%" range column) and every trade number switched over
  with no other change.
- **"Model (ROS)" stats source**, the default because it beat the blend.
  Its games discount uses his projected share of his team's remaining
  games (injuries included) instead of GP/82, and season totals (My team,
  waivers, player impact) count over the weeks left rather than a fixed 20.
- **Player page**: a "Rest of season (model)" card with each category's
  projection and 80% range next to ESPN's projection and this season, and
  a model row in the stat lines.
- **Model tab**: the backtest tables (either season, any checkpoint), the
  learned k per stat, the boosting decision with its bootstrap interval,
  interval coverage, and the live scoring below.

How predictions get there: the refresh loads `data/models/ros_model.pkl`,
builds every player's current row from this season's game logs plus his
last two seasons (`dataset.current_rows()`), predicts, and writes
`data/model_proj.csv` (`M_*` columns); `load_cached()` attaches them by
name. Players with no games yet are predicted from their prior seasons;
true rookies with no games yet fall back to the interim line.

## Testing against the real world

From now on, without retraining:

- **Daily snapshots.** Each refresh saves the model's predictions to
  `data/snapshots/model_proj_YYYY-MM-DD.csv`, next to ESPN's
  `espn_proj_YYYY-MM-DD.csv`. `python -m ninecat.model score` (and the
  Model tab) scores every snapshot against each player's actual per-game
  line in his games since that day (3+ games). On days with both, model
  and ESPN are scored on the same players.
- **The matchup model.** Each refresh logs, for the current matchup
  period, the matchup model's win chance for every team the first time it
  sees the period (`pre_week` records whether that was before the period
  began) and the live category score; once the period is over the result
  is final (`data/matchup_log.csv`). The Model tab reports the Brier score
  (0 is perfect, always guessing 50% scores 0.25) and a calibration chart:
  of matchups given about 70%, did about 70% win? This tests the matchup
  model's assumed constants: the 30% game-to-game swing, the 1.6× shooting
  variance and the category correlations.

## Limits

- The backtest's preseason prior is a stand-in (last two seasons), not
  ESPN's real projection; see Baselines.
- Starts are a proxy (top five in minutes).
- No information about trades, coaching changes or teammates' injuries
  beyond what shows up in his minutes and last 5 games.
- Availability can't foresee injuries; its error is large early in the
  season.
- The ranges are slightly narrow (about 76% coverage for a nominal 80%).
