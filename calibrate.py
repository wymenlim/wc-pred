"""Probability recalibration for the World Cup predictor (Part 8).

The backtest calibration table showed a systematic pattern: the raw model
overstates longshots and understates favorites (its rating spread is too
compressed). This module corrects the win/draw/loss probabilities with
vector scaling — the 3-outcome generalization of temperature scaling:

    q_k  ∝  p_k^T · exp(b_k)        k in {win, draw, loss}, b_win = 0

T > 1 stretches the distribution apart (fixes underconfidence), and the
per-outcome offsets b absorb any draw-rate bias. The three parameters are
fit by maximum likelihood on leak-free backtest records from tournaments
BEFORE the one being predicted (see tune.py / parts.md for provenance of
the baked-in TUNED constants).

Typical use:
    from calibrate import apply_calibration, TUNED
    q = apply_calibration(p, TUNED)           # p: (n, 3) [win, draw, loss]

Recalibration applies to the 1X2 probabilities only — the scoreline
matrix and P(advances) outputs remain the raw model's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-10


@dataclass(frozen=True)
class CalibrationParams:
    temperature: float
    b_draw: float
    b_loss: float


def apply_calibration(probs: np.ndarray, params: CalibrationParams) -> np.ndarray:
    """Recalibrate an (n, 3) array of [p_win, p_draw, p_loss] rows."""
    probs = np.atleast_2d(probs)
    b = np.array([0.0, params.b_draw, params.b_loss])
    z = params.temperature * np.log(np.clip(probs, _EPS, 1.0)) + b
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_calibration(records) -> CalibrationParams:
    """Fit vector scaling by maximum likelihood on backtest records.

    ``records`` needs p_win/p_draw/p_loss and outcome columns (as produced
    by backtest.run_edition). Fit on tournaments strictly before the one
    you intend to predict.
    """
    probs = records[["p_win", "p_draw", "p_loss"]].to_numpy()
    outcomes = records["outcome"].to_numpy()
    idx = np.arange(len(outcomes))

    def nll(x):
        q = apply_calibration(probs, CalibrationParams(*x))
        return -np.sum(np.log(np.clip(q[idx, outcomes], _EPS, 1.0)))

    res = minimize(nll, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
    assert res.success, "calibration fit did not converge"
    return CalibrationParams(*(float(v) for v in res.x))


# Fit 2026-07-02 on the group stages of all 28 major-tournament editions
# 2006-2025 (839 matches, backtest.backtest_editions at the tuned defaults,
# WC 2026 excluded). Held-out WC 2026 group stage: log-loss 0.8446 -> 0.8412.
# Refit after any change to the model or its tuned defaults:
#   fit_calibration(recs[recs.tournament != "WC 2026"])
TUNED = CalibrationParams(
    temperature=1.0668, b_draw=0.0157, b_loss=0.0321
)
