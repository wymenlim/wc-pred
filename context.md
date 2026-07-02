# World Cup 2026 Match Predictor — Project Context

Design agreed on 2026-07-02 via a structured interview. This document is the full record of every decision and the reasoning behind it.

## Purpose

Predict the remaining knockout matches of the **2026 FIFA World Cup, which is currently in progress** (June 11 – July 19, 2026, hosted by USA/Mexico/Canada). The group stage is complete; knockout rounds are underway (e.g. Spain–Austria and Portugal–Croatia on July 2, Argentina–Cape Verde on July 3).

The workflow is: rerun the notebook → it downloads the latest results → refits the model → predicts all upcoming fixtures. Predictions stay current as each round completes.

Alternatives considered and rejected: learning/portfolio project (historical data would suffice), betting decision support (would demand bookmaker benchmarking) *(partially reversed later the same day — see Revision below)*, general-purpose any-match tool (unfocused).

## Outputs (per match)

1. **Expected goals per team** — the model's Poisson rates, e.g. France 1.9, Paraguay 0.7.
2. **Most likely full-time (90-minute) scoreline** with its probability, e.g. 2-0 (14%). Draws are valid predictions — in knockout football a draw after 90 minutes is often the single most likely score.
3. **P(advances) per team** (knockout bonus output) — computed as P(win in 90) + P(draw in 90) × P(wins extra time / penalties). Extra time is modeled as a short, low-scoring extension of the same Poisson rates (~⅓ of match length, typically with a further tempo reduction); penalties are treated as ≈ 50/50.

Rejected output framings: total-goals-only (over/under style), full scoreline probability table as the primary output (available internally anyway — the score matrix is computed en route), suppressing draws in knockouts (dishonest about the true distribution).

## Model

**Dixon-Coles-style Poisson model:**

- Each national team gets an **attack rating** and a **defense rating**, fit jointly on historical international results.
- Expected goals for team A vs team B = f(attack_A, defense_B, home advantage).
- **Home advantage term applied only when the match is not at a neutral venue.** The dataset's `neutral` column handles this — matches involving hosts USA/Mexico/Canada at home are flagged non-neutral.
- **Dixon-Coles tau correction** for the known low-score dependence (0-0, 1-0, 0-1, 1-1 cells deviate from independent Poissons).
- **Exponential time decay** on match weights, half-life ≈ 2 years, so recent form (including the 2026 group stage just played) dominates.
- **Friendlies down-weighted** relative to competitive matches.
- Training window: matches from ~2000 onward.
- Fit by maximum likelihood via `scipy.optimize`.

Alternatives rejected: Elo-to-goals mapping (too coarse, no attack/defense split so "expected goals" loses meaning), XGBoost/ML (heavy feature engineering for marginal gain, harder to interpret), ensemble of several models (doubles the work).

## Data

**Source:** the martj42 open dataset of all international results since 1872:

```
https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

Verified current on 2026-07-02:
- Contains completed 2026 World Cup results through June 30 (e.g. France 3-0 Sweden, Brazil 2-1 Japan).
- Contains **upcoming fixtures as rows with NA scores** — so rows with scores are training data and NA rows are the prediction queue. No fixture list needs to be maintained by hand.
- Columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral.

No API keys, no scraping, no cost. Rejected alternatives: football-data.org (rate-limited, thin history), API-Football (paid, unneeded richness), scraping (fragile).

## Validation

**Full self-calibration analysis** (the user chose the most rigorous self-contained option):

- **Backtests:** fit only on data available before each of the 2018 and 2022 World Cups, predict those tournaments, and score the predictions. Also score against the 2026 group stage (72 matches — the most relevant possible test set).
- **Metrics:** log-loss and Ranked Probability Score (RPS) on win/draw/loss outcomes vs naive baselines (e.g. uniform, historical frequencies); error metrics on expected goals; scoreline hit rates.
- **Calibration curves:** when the model says an outcome has 60% probability, does it occur ~60% of the time?

**Explicitly out of scope:** bookmaker-odds benchmarking. Historical closing odds for international matches are patchy to source for free; decided not worth the effort for this purpose. (A "compare against current bookie lines by eyeball" option was also considered and rejected as statistically meaningless on few matches.)

*Superseded 2026-07-02: a free structured odds source was found and the benchmark was added — see Revision below. The premise ("patchy to source for free") turned out to be wrong for World Cups specifically.*

## Scope boundaries

- **Single-match predictions only.** No Monte Carlo simulation of the remaining bracket / P(champion). Deliberately deferred — it is an easy later addition (replay the match model over the bracket thousands of times) but needs the bracket structure encoded and maintained.
- No player-level data, no shot-based real xG, no live in-match updating.

## Interface & structure

- **Jupyter notebook** as the primary interface: `predictor.ipynb`, telling the story top-to-bottom — data loading → model fitting → calibration/backtests → today's predictions. "Run all cells" produces fresh predictions.
- **`model.py` helper module** holding the Poisson fitting and prediction code, so it is testable, reusable, and not buried in notebook cells.
- **Stack:** Python, pandas, scipy (numpy, matplotlib implied). Environment managed with `uv`; project under `git`.

Rejected: CLI tool (was the recommendation, user preferred notebook), Streamlit app, multiple notebooks split by stage (state-juggling overhead at this size).

## Revision 2026-07-02: bookmaker-odds benchmarking added (`odds.py`)

Later on 2026-07-02 (knockouts underway) the user asked how good the notebook is **for betting** — a purpose the original design had rejected. Assessment: strong self-calibration, but the only benchmark that matters for betting (the market) was untested. Rather than paying for a historical-odds API (~$30/month, The Odds API) or hand-entering odds, a free structured source was found:

**Source:** football-data.co.uk's World Cup workbook — `https://www.football-data.co.uk/WorldCup2026.xlsx`. One sheet per tournament (2014–2026) with per-match 1X2 decimal odds: bet365, Betfair Exchange, market max, market average. Covers every match of the 2018, 2022, and 2026 tournaments, so all 168 backtested group-stage matches get real market prices. (Their league CSVs don't cover internationals; this one-off workbook does.)

**What was added:**

- **`odds.py` (Part 6):** downloads/caches the workbook (same 1-day-cache pattern as `data.py`), normalizes the four team-name mismatches vs martj42 (`USA` → `United States`, `Bosnia & Herzegovina`, `Curacao`, `D.R. Congo`), joins to the backtest's per-match records on (tournament, teams) — not dates, which drift a day due to UK-time kickoffs — handling home/away flips. `compare_to_market(records)` scores the model against de-vigged market-average probabilities (log-loss, RPS) and runs a flat-stake betting sim: 1 unit on every outcome where the model beats the market by ≥ 5 points, settled at average and best-available prices.
- **Notebook:** a "Market comparison" cell in section 3; the odds-check caveats updated to cite the result.
- `openpyxl` added to dependencies.

**Findings (the honest, negative result):**

- The de-vigged market average beats the model on **both log-loss and RPS in all three tournaments** — pooled 0.912 vs 0.965 log-loss; widest gap in 2026 (0.820 vs 0.899).
- The betting sim shows +36 units pooled (+18.9% ROI) but this is **longshot variance, not edge**: flagged bets average odds ~7, z ≈ 0.86, and two shock results (Saudi Arabia over Argentina 2022, South Korea over Germany 2018) exceed the entire profit by themselves. The 2026 group stage alone was flat (−0.2 units at average prices).

**Conclusion recorded:** the model beats naive baselines comfortably but is dominated by the market unblended. It should be used as a sanity check on the user's own opinions, not a standalone source of +EV bets; `check_odds` edges are more likely model blind spots than market mistakes. A defensible betting layer would blend the market price in as a prior and bet only residual disagreement (not built; candidate next step alongside a bet log with closing-line-value tracking).

## Revision 2026-07-02 (later): model improvements 1–4 implemented

Same day, the user asked to implement the four improvements proposed after the market comparison. All four landed (details and verification numbers in `parts.md`, Parts 8–10):

1. **Global intercept** in `model.py` — the L2 penalty had been implicitly anchoring the baseline rate at exp(0) = 1.0 goals/team; an unpenalized intercept lets shrinkage compress the rating *spread* without dragging the *level*. Structurally right, but only trimmed the goals bias −0.45 → −0.43: most of that bias is real scoring-environment drift, not shrinkage. Two latent numerical bugs were found and fixed en route (exp overflow NaNs in the line search; ~1e10 gradient spikes at the tau floor — the "clip" a Part-3 comment referred to had never actually existed).
2. **Widened backtests + tuning** — `backtest.editions()` auto-detects all 29 WC/Euro/Copa/AFCON editions 2006–2026 (911 group matches; knockouts excluded by the both-teams-under-3-appearances rule). `tune.py` grid-searched (half_life, friendly_weight, l2) on the 28 pre-2026 editions, WC 2026 held out. Winners: **730 / 1.0 / 0.25** — i.e. keep the 2-year half-life, *stop down-weighting friendlies*, and cut L2 to a quarter. Old defaults ranked 51/100. This was the big win: WC 2026 holdout log-loss 0.899 → 0.845.
3. **Recalibration** (`calibrate.py`) — vector scaling fit on the 839 pre-2026 records (T = 1.067; the tuned model needed little correction). Holdout: 0.8446 → 0.8412.
4. **Market blend** (`odds.holdout_ladder`) — log-space blend, weight fit on WC 2018+2022 only: **w_model = 0.23**. Final WC 2026 holdout ladder: raw 0.8446 → calibrated 0.8412 → market 0.8195 → **blend 0.8194**.

**Updated conclusion:** the tuned+calibrated model closed most of the gap to the market (0.079 → 0.022) and even beats it on WC 2018, but the blend only *ties* the market on the 2026 holdout — the model earns a real 0.23 weight yet adds ≈nothing over the market alone there. `check_odds` now shows raw → calibrated → blend probabilities and flags a bet only when the calibrated edge ≥ 5 points *and* the blend EV is positive at the offered odds; on realistic prices this correctly answers NO BET where the raw model used to see phantom value. Betting stance unchanged: use it to find prices worth investigating, not as a money printer.
