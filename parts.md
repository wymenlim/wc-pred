# Work Breakdown & Difficulty

Each part below is a self-contained chunk of work, ordered by dependency. Difficulty is rated for how much reasoning/judgment the part demands — the main signal for choosing which AI model to assign. Rule of thumb: ★☆☆ parts are safe for a fast/cheap model; ★★☆ parts want a mid-tier model; ★★★ parts are where subtle bugs hide silently and a frontier model pays for itself.

## Part 1 — Project scaffolding

Set up `uv` environment, dependencies (pandas, numpy, scipy, matplotlib, jupyter), `git init`, empty `predictor.ipynb` and `model.py` skeletons.

- **Difficulty: ★☆☆ trivial.** Pure boilerplate, no judgment.
- **Model:** any fast/cheap model.
- **Status: ✅ DONE (2026-07-02, Sonnet subagent).** uv project (Python 3.14, pandas 3.0.3 / numpy 2.5.0 / scipy 1.18.0 / matplotlib 3.11.0 / jupyter 1.1.1), git initialized (no commits yet), `.gitignore`, stub-only `model.py` (fit / FittedModel.expected_goals raising NotImplementedError), skeleton `predictor.ipynb` with the four section headers and a working data cell.

## Part 2 — Data loading & preparation

Download the martj42 CSV, parse dates, split scored rows (training) from NA-score rows (prediction queue), filter to matches from ~2000 on, compute per-match weights (exponential time decay, half-life ≈ 2 years; friendlies down-weighted), handle team-name consistency.

- **Difficulty: ★☆☆–★★☆ easy.** Mostly routine pandas. The only judgment calls are the weighting scheme details and edge cases like renamed/dissolved national teams — but errors here are visible and easy to fix.
- **Model:** fast/cheap model is fine; mid-tier if you want the weighting choices sanity-checked.
- **Status: ✅ DONE (2026-07-02, Sonnet subagent; independently verified).** `data.py` implements the interface contract exactly: `download_results` (urllib, 1-day cache in `data/`), `prepare(df, start, as_of)` with the leakage guard, `validate_teams` (min 10 matches), `load` wrapper. Verified: 25,422 training rows (2000-01-04 → 2026-06-30), weights ∈ (0, 0.998], 9 upcoming knockout fixtures (Spain–Austria Jul 2 … Brazil–Norway Jul 5), `validate_teams` passes with no name fixes needed. Leakage check: `as_of="2018-06-13"` yields 17,577 rows, max date 2018-06-12, weights correctly relative to the cutoff. Notebook executes clean via nbconvert. Known quirk: NA-score rows dated *before* today (e.g. yesterday's unbackfilled results) fall in neither training nor upcoming until upstream adds scores. **Addressed 2026-07-02:** `patches.csv` + `_apply_patches()` in `data.py` fill unbackfilled results locally (upstream wins once it catches up; knockout patch scores include ET per upstream convention). Currently patched: the three July 1 R32 results (England 2-1 DR Congo, Belgium 3-2 aet Senegal, USA 2-0 Bosnia).

## Part 3 — Core model: Dixon-Coles Poisson fitting (`model.py`)

Implement the likelihood: per-team attack/defense parameters, home-advantage term gated on the `neutral` flag, Dixon-Coles tau low-score correction, weighted maximum-likelihood fit via `scipy.optimize`, identifiability constraint (e.g. mean attack = 0), and a fitted-model object exposing expected goals for any pairing.

- **Difficulty: ★★★ hard.** This is the heart of the project and the classic place for silent bugs: sign errors in the log-likelihood, the tau correction applied to the wrong cells, degenerate optima, parameters for teams with few matches blowing up. The code will *run* and produce plausible-looking numbers even when wrong — only careful reasoning or the Part 5 calibration catches it.
- **Model:** strongest model available. Do not economize here.
- **Status: ✅ DONE (2026-07-02, frontier model).** `model.py` implements the full contract: vectorized weighted log-likelihood with analytic gradients (verified against finite differences, max rel. error 5e-6), L-BFGS-B, rho bounded to ±0.15 with a tau floor, L2 ridge (l2=1.0) that both fixes the shift degeneracy and shrinks near-zero-weight teams. Fit: 0.1s, converged, 321 teams, home_adv=0.256, rho=−0.047 (both in the literature-typical range). Sanity: top 15 = Argentina, Spain, Brazil, England, France, Portugal…; San Marino ranks 306/321. NOT yet validated by backtests — Part 5 still pending.
- **Revised later on 2026-07-02 (Part 9 fallout):** (a) added an **unpenalized global intercept** — the L2 penalty was implicitly anchoring the baseline rate at exp(0)=1.0 goals; (b) fixed two latent numerical bugs exposed by the tuning grid: exp() under/overflow during line search producing log(0) NaNs (Poisson term now computed as x·z − exp(z) with the linear predictor clipped to [−30, 5]), and ~1e10 gradient spikes when tau hit its floor (floored entries' tau-gradients now zeroed); (c) l2 default retuned 1.0 → 0.25. Gradient re-verified to 3e-6 after each change.

## Part 4 — Prediction outputs

From fitted rates, build the scoreline probability matrix (with tau correction), extract: per-team expected goals, most likely FT scoreline + probability, win/draw/loss probabilities, and P(advances) for knockouts (P(win) + P(draw) × P(wins ET/pens), ET as a low-scoring ~⅓-length extension, pens 50/50). Clean display formatting in the notebook.

- **Difficulty: ★★☆ moderate.** Straightforward once Part 3 is correct, but the ET/pens advance-probability composition has room for probability-logic slips (conditioning errors, forgetting the draw-in-ET branch).
- **Model:** mid-tier model; strong model if bundled with Part 3.
- **Status: ✅ DONE (2026-07-02, frontier model, bundled with Part 3).** `predict.py`: tau-corrected score matrix (normalized, verified sums to 1), `Prediction` dataclass with `summary()`, `predict()` and `predict_fixtures()`. Advance probs: ET as 30-min Poisson extension at λ/3 (no tau reapplied — it's a 90-min low-score effect), pens 50/50; verified P(adv_a)+P(adv_b)=1 and W/D/L sums to 1. Notebook sections 2 and 4 wired up and executing clean. All 9 knockout fixtures predicted (e.g. Argentina 89.9% to advance past Cape Verde; Paraguay–France 0-1, France 73.1%). Predictions are UNVALIDATED until Part 5 runs.

## Part 5 — Backtesting & calibration analysis

Temporal backtest harness (fit only on pre-tournament data for 2018 WC, 2022 WC, 2026 group stage), score with log-loss and RPS vs naive baselines, expected-goals error metrics, scoreline hit rates, calibration curves with binned reliability plots.

- **Difficulty: ★★★ hard.** The #1 risk is **data leakage** — accidentally letting post-cutoff matches into the fit, which makes the model look great and means nothing. RPS and calibration-binning implementations are also easy to get subtly wrong. This part is what makes the whole project trustworthy, so errors here are the most expensive kind: invisible.
- **Model:** strongest model available.
- **Status: ✅ DONE (2026-07-02, frontier model).** `backtest.py`: leakage-guarded harness (asserts training max date ≤ cutoff < first test match), log-loss + RPS (both unit-tested against hand-computed values), uniform + empirical baselines, scoreline hit rate vs modal-score baseline, xG MAE, calibration table/plot. **Design decision: group-stage matches only** (48+48+72=168) because this dataset records knockout scores including extra time, which would mislabel 90-min draws. **Verdict: model beats both baselines on every tournament and metric** — pooled log-loss 0.965 (uniform 1.099, empirical 1.075), pooled RPS 0.196 (baselines 0.239/0.237), scoreline hit rate 10.7% (modal baseline 8.9%). Calibration tracks the diagonal; caveats: mild underconfidence on 0.5–0.6 favorites (empirical 0.81, n=36 so noisy), and the model under-predicts total goals by ~0.45/match (recent WCs higher-scoring than history). Notebook section 3 wired and executing.

## Part 6 — Notebook assembly & narrative

Wire Parts 2–5 into `predictor.ipynb` as a clean top-to-bottom story: data → fit → validation → today's predictions. Markdown explanations, plots (scoreline heatmap, calibration curves), a final cell that predicts all NA-score fixtures.

- **Difficulty: ★☆☆–★★☆ easy.** Mostly glue and presentation; the quality bar is readability, not correctness.
- **Model:** fast/cheap model, or whichever model you enjoy the prose of.

## Part 7 (deferred) — Bracket Monte Carlo

Simulate the remaining knockout bracket to get P(champion) etc. Explicitly out of scope for now.

- **Difficulty: ★★☆ moderate** (bracket encoding + reusing the match model). Listed only so it isn't forgotten.

## Part 8 — Bookmaker-odds benchmarking (`odds.py`)

Added 2026-07-02 when the project pivoted toward betting use (originally out of scope — see context.md Revision).

- **Status: ✅ DONE (2026-07-02, frontier model).** Downloads football-data.co.uk's free `WorldCup2026.xlsx` (per-match 1X2 odds, sheets for 2014–2026; bet365 / Betfair / max / average), normalizes 4 team names, joins to backtest records on (tournament, teams) — not dates, which drift a day on UK kickoff times — with home/away-flip retry. Scores the model against de-vigged market-average probabilities and runs a flat-stake ≥5pt-edge betting sim at avg and max prices. **Finding (pre-tuning): market beat the raw model on log-loss and RPS in all three WCs (pooled 0.912 vs 0.965); sim "profit" was longshot variance (z≈0.9).**

## Part 9 — Widened backtests & hyperparameter tuning (`backtest.editions`, `tune.py`)

- **Status: ✅ DONE (2026-07-02, frontier model).** `backtest.editions()` auto-detects every completed WC/Euro/Copa/AFCON edition (2006+) by clustering same-name matches on >60-day gaps; group stage identified by the both-teams-under-3-prior-appearances rule (can never leak a knockout in; conservatively drops 4th group games in groups-of-5 formats). 29 editions, 911 group matches; sanity-checked against AFCON 2010 (Togo withdrawal → 21 matches) and the hand-coded WC windows. `tune.py` grid-searched half_life × friendly_weight × l2 (100 combos) on pooled leak-free log-loss over the 28 pre-2026 editions, WC 2026 held out. **Winners: half_life 730 (unchanged), friendly_weight 1.0 (down-weighting friendlies hurt), l2 0.25 (1.0 over-compressed ratings). Old defaults ranked 51/100. WC 2026 holdout: log-loss 0.899 → 0.845.** Full grid in `tune_results.csv`.

## Part 10 — Recalibration & market blend (`calibrate.py`, `odds.holdout_ladder`)

- **Status: ✅ DONE (2026-07-02, frontier model).** Vector scaling (temperature + per-outcome offsets, b_win≡0) fit by ML on the 839 pre-2026 edition records: T=1.067, small offsets (the tuned model needs little correction); WC 2026 holdout 0.8446 → 0.8412. Log-space blend with the de-vigged market, weight fit on WC 2018+2022 only: w_model = 0.23. **Holdout ladder (WC 2026): raw 0.8446 → calibrated 0.8412 → market 0.8195 → blend 0.8194 — the blend ties the market; the model carries real weight (0.23) but adds ≈nothing on 2026.** Notebook `check_odds` now flags a bet only when the calibrated edge ≥ min_edge AND the blend EV is positive — on realistic prices this correctly says NO BET where the raw model used to scream value.

## Dependencies & parallelism

Which parts must wait for others, and which can run independently (e.g. farmed out to different models at the same time).

**Hard dependencies (must be sequential):**

- **Part 2 → Part 3** — the fitter needs the prepared, weighted training data.
- **Part 3 → Part 4** — the scoreline matrix and advance probabilities are computed from the fitted model object.
- **Parts 3 + 4 → Part 5 (final backtest run)** — you can't score predictions that don't exist yet.
- **Everything → Part 6** — it's the glue, so it goes last.

**Independently workable:**

- **Part 1** — anytime, trivially first.
- **Part 2 vs Part 3** — parallel *if* the interface contract is agreed upfront (see below). Part 3 can be developed against a small hand-made dummy DataFrame.
- **Part 5's metric functions** — log-loss, RPS, and calibration binning are pure math with no dependency on the model at all; only the final "run the backtest" step needs Parts 2–4 working.
- **Part 7 (deferred)** — only needs Part 4's interface.

**Interface contract for parallel work** (agree on this first, ~10 lines):

- Part 2 produces a DataFrame with columns: `date, home_team, away_team, home_score, away_score, neutral, weight` (training rows) and the same minus scores (prediction queue).
- Part 3 exposes `fit(df) -> FittedModel` and `FittedModel.expected_goals(team_a, team_b, neutral) -> (lambda_a, lambda_b)`.
- Part 4 exposes `predict(model, team_a, team_b, neutral, knockout) -> Prediction` (xG pair, score matrix, most likely score, W/D/L probs, advance probs).

**Dependency graph:**

```
Part 1 ──────────────┐
Part 2 ──────┐       │
             ├─ Part 3 ─ Part 4 ─┐
(contract) ──┘                   ├─ Part 5 (backtest run) ─ Part 6
Part 5 metrics (independent) ────┘
```

**Two sensible orderings:**

1. **Simple path (solo / one model):** 1 → 2 → 3 → 4 → 5 → 6. Natural dependencies, no contract-mismatch risk between separately built pieces.
2. **Parallel path (multiple models at once):** write the interface contract first, then Parts 2, 3, and 5-metrics proceed concurrently and converge at the backtest run.

**Non-negotiable regardless of ordering:** don't trust Part 4's predictions until Part 5's backtests have run. That's a verification dependency rather than a build dependency, but it's the one that matters most.

## Suggested grouping

If you want to minimize model-switching: give **Parts 3 + 4 + 5 to a frontier model** in one session (they share the mathematical context), and **Parts 1 + 2 + 6 to a cheaper model**. Verification order matters: run Part 5's backtests before believing anything Part 4 prints.
