# f1wdc — 2027 F1 World Drivers' Championship predictor

Predicts the 2027 Formula 1 Drivers' Championship from everything known today, and keeps
updating as the rest of 2026 and then 2027 race by race come in.

```
uv sync
uv run f1wdc fetch            # download 1994→now from the Jolpica API (cached; ~430 requests the first time)
uv run f1wdc predict          # 20,000 simulated 2027 seasons → terminal table + out/report_2027.html
uv run f1wdc backtest         # replay 1997–2026 using only prior data, score against real champions
```

## How it works

1. **Ratings** (`model/plackett_luce.py`, `model/ratings.py`). Every race, sprint and qualifying
   result is fitted with a Plackett-Luce ranking model:
   `strength = driver skill (per season, random-walk linked) + car level + in-season car trend`
   (car terms per team per season). Drivers moving between teams separate driver from car.
   Each season is fitted on a rolling 3-season window that only sees data up to that season,
   so the ratings are leak-free for backtesting. Results far below a driver's own norm for the
   season (quali failures, damage) are dropped, because Plackett-Luce otherwise treats one
   freak P20 as losing to the whole field.
2. **Projection** (`model/projection.py`). Next season's car pace = `ρ × end-of-season pace`,
   with ρ and its noise learned separately for stable seasons, the first year of new rules and the
   **second year of new rules (2027)**. Driver skill shifts with an age curve, and rookies start from
   the average debut rating. Teams are tracked by lineage across rebrands (`config/teams.yaml`).
3. **Reliability** (`model/reliability.py`). Per-race DNF probability from recent team and driver
   records (beta-binomial shrinkage).
4. **Simulation** (`sim/season.py`). Per simulated season, driver and car strengths are drawn once
   (a quick car is quick all year). Each race then draws DNFs, samples the finishing order from the
   same Plackett-Luce model (Gumbel-max trick) and awards points (sprints included, countback
   tie-breaks).
5. **In-season updating** (`pipeline.py`). Once the target season has started, its car ratings are
   refitted with priors centred on the pre-season projection (Laplace approximation for
   uncertainty), points already scored are kept, and only the remaining rounds are simulated.

## Configuration

| File | Purpose |
|---|---|
| `config/season_2027.yaml` | 2027 grid (team → drivers), calendar fallback, points. **Edit this when line-ups change.** |
| `config/model.yaml` | Model settings chosen by `f1wdc backtest --tune` (grid search on 1997–2019) |
| `config/regulations.yaml` | First seasons of major rule changes (drives the carry-over type) |
| `config/teams.yaml` | Constructor → team lineage (e.g. Toro Rosso → AlphaTauri → RB) |

Useful flags: `predict --carryover 0.6` (scenario: more catch-up than history suggests),
`predict --update` (refresh data first), `backtest --tune` (re-tune settings),
`--grid other.yaml` (another season or line-up).

## Backtest (settings in `config/model.yaml`)

Pre-season WDC log-loss (lower is better), each season predicted from earlier data only:

| Seasons | Model | Last year's points | Last year's top-2 teams | Champion = model's favourite |
|---|---|---|---|---|
| 1997–2019 (tuning) | **1.27** | 1.63 | 2.00 | 65% |
| 2020–2026 (held out) | 1.52 | 1.54 | **1.40** | 71% |
| 2nd year of new rules (1999, 2010, 2015, 2018, 2023) | **0.56** | 0.85 | 1.38 | 5 of 5 |

The model is weakest in the first year of new rules (2009 Brawn, 2026 Antonelli) and loses to
the crude "top-2 teams" baseline on 2020–2026. Its strongest category, the second year of a
rule set, is the one 2027 belongs to.

## Data

[Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (the Ergast successor). The client is
throttled to the published limits (4 req/s burst, 500 req/hour), so separate runs share the hourly
budget. Finished seasons are cached forever, and the current season is refreshed after 6 hours.
