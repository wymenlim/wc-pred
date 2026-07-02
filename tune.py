"""Hyperparameter tuning for the World Cup predictor (Part 7).

Grid search over the three hand-picked knobs nobody had validated:

    half_life_days   exponential time-decay half-life (was 730)
    friendly_weight  down-weighting of friendlies     (was 0.5)
    l2               ridge penalty on attack/defense  (was 1.0)

Each combination is scored by pooled leak-free log-loss over the group
stages of every major-tournament edition detected by backtest.editions
(WC / Euro / Copa América / AFCON, 2006 onward) — EXCLUDING WC 2026,
which is reserved as the untouched holdout for judging the final
configuration. Every edition is fit strictly on pre-tournament data.

Run:  uv run python tune.py
Writes tune_results.csv (all combinations, sorted by log-loss) and prints
the winner plus its WC 2026 holdout score. The winning values are then
baked in as the defaults in data.prepare and model.fit.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from backtest import editions, log_loss, rps, run_edition
from data import download_results

HALF_LIVES = [365.0, 550.0, 730.0, 1095.0, 1460.0]
FRIENDLY_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
L2S = [0.25, 0.5, 1.0, 2.0, 4.0]

HOLDOUT = "WC 2026"


def score_editions(df, eds, half_life_days, friendly_weight, l2):
    """Pooled log-loss / RPS over the given editions at one setting."""
    recs = pd.concat(
        [
            run_edition(
                df, label, cutoff, test,
                half_life_days=half_life_days,
                friendly_weight=friendly_weight,
                l2=l2,
            )
            for label, cutoff, test in eds
        ],
        ignore_index=True,
    )
    probs = recs[["p_win", "p_draw", "p_loss"]].to_numpy()
    outcomes = recs["outcome"].to_numpy()
    bias = float(
        np.mean(
            (recs["lambda_home"] + recs["lambda_away"])
            - (recs["home_goals"] + recs["away_goals"])
        )
    )
    return log_loss(probs, outcomes), rps(probs, outcomes), bias, len(recs)


def main():
    df = download_results()
    eds = editions(df)
    tune_eds = [e for e in eds if e[0] != HOLDOUT]
    holdout = [e for e in eds if e[0] == HOLDOUT]
    print(
        f"tuning on {len(tune_eds)} editions "
        f"({sum(len(t) for *_, t in tune_eds)} matches), "
        f"holdout: {HOLDOUT}"
    )

    rows = []
    grid = list(itertools.product(HALF_LIVES, FRIENDLY_WEIGHTS, L2S))
    t0 = time.time()
    for i, (hl, fw, l2) in enumerate(grid):
        ll, r, bias, n = score_editions(df, tune_eds, hl, fw, l2)
        rows.append(
            {
                "half_life_days": hl,
                "friendly_weight": fw,
                "l2": l2,
                "log_loss": ll,
                "rps": r,
                "total_goals_bias": bias,
                "n": n,
            }
        )
        print(
            f"[{i + 1:3d}/{len(grid)}] hl={hl:6.0f} fw={fw:.2f} l2={l2:.2f}"
            f" -> ll={ll:.4f} rps={r:.4f} bias={bias:+.3f}"
            f"  ({time.time() - t0:.0f}s)"
        )

    results = pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)
    results.to_csv("tune_results.csv", index=False)

    best = results.iloc[0]
    print("\nbest:", best.to_dict())
    ll, r, bias, n = score_editions(
        df, holdout,
        best["half_life_days"], best["friendly_weight"], best["l2"],
    )
    print(f"{HOLDOUT} holdout at best params: ll={ll:.4f} rps={r:.4f} "
          f"bias={bias:+.3f} (n={n})")


if __name__ == "__main__":
    main()
