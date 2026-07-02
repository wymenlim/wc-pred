"""Archived closing-odds comparison for the World Cup predictor (Part 6).

football-data.co.uk publishes a World Cup workbook with per-match 1X2
decimal odds (bet365, Betfair Exchange, market max, market average) for
the 2014-2026 tournaments. This module downloads it, normalizes team
names to the martj42 convention, joins it to the backtest's per-match
records, and scores the de-vigged market-average odds on exactly the
matches the model predicted.

This answers the question the naive baselines in backtest.py cannot:
when the model disagrees with the bookmakers, who tends to be right?
The market is the benchmark that matters for betting — beating uniform
1/3s is table stakes, beating the de-vigged closing average is an edge.

Typical use:
    from odds import compare_to_market
    summary, merged = compare_to_market(records)   # records from backtest_all
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import log_loss, rps

ODDS_URL = "https://www.football-data.co.uk/WorldCup2026.xlsx"

_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day

# workbook sheet per backtest tournament name (see backtest.GROUP_STAGES)
_SHEETS = {
    "WC 2018": "WorldCup2018",
    "WC 2022": "WorldCup2022",
    "WC 2026": "WorldCup2026",
}

# football-data.co.uk name -> martj42 name
_TEAM_FIXES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "D.R. Congo": "DR Congo",
    "USA": "United States",
}


def download_odds(
    cache_path: str = "data/WorldCup2026.xlsx", refresh: bool = False
) -> pd.DataFrame:
    """Download (or reuse a cached copy of) the World Cup odds workbook.

    Returns a tidy DataFrame with one row per match:
        tournament, home_team, away_team,
        avg_h, avg_d, avg_a  (market-average 1X2 odds),
        max_h, max_d, max_a  (best available price).

    Team names are normalized to the martj42 convention. Dates are not
    returned: the workbook uses UK kickoff dates, which drift a day from
    martj42's local dates, so joins should use (tournament, teams).
    """
    path = Path(cache_path)
    stale = (
        refresh
        or not path.exists()
        or (time.time() - path.stat().st_mtime) > _CACHE_MAX_AGE_SECONDS
    )
    if stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(ODDS_URL) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)

    frames = []
    for tournament, sheet in _SHEETS.items():
        df = pd.read_excel(path, sheet_name=sheet)
        frames.append(
            pd.DataFrame(
                {
                    "tournament": tournament,
                    "home_team": df["Home"].replace(_TEAM_FIXES),
                    "away_team": df["Away"].replace(_TEAM_FIXES),
                    "avg_h": df["H-Avg"],
                    "avg_d": df["D-Avg"],
                    "avg_a": df["A-Avg"],
                    "max_h": df["H-Max"],
                    "max_d": df["D-Max"],
                    "max_a": df["A-Max"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def devig(odds: np.ndarray) -> np.ndarray:
    """Proportionally de-vig an (n, 3) array of decimal odds into probs."""
    implied = 1.0 / odds
    return implied / implied.sum(axis=1, keepdims=True)


_EPS = 1e-10


def blend(
    model_probs: np.ndarray, market_probs: np.ndarray, w: float
) -> np.ndarray:
    """Geometric (log-space) blend: q ∝ model^w * market^(1-w), per row.

    w = 0 is the market alone, w = 1 the model alone. Log-space rather
    than linear so a strong disagreement dampens multiplicatively — the
    standard form for combining probability forecasts.
    """
    z = w * np.log(np.clip(model_probs, _EPS, 1.0)) + (1.0 - w) * np.log(
        np.clip(market_probs, _EPS, 1.0)
    )
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_blend_weight(
    model_probs: np.ndarray, market_probs: np.ndarray, outcomes: np.ndarray
) -> float:
    """Max-likelihood blend weight on held-out-from-the-target matches.

    Fit on tournaments strictly before the one you intend to bet. A weight
    near 0 means the model adds nothing over the market.
    """
    from scipy.optimize import minimize_scalar

    idx = np.arange(len(outcomes))

    def nll(w):
        q = blend(model_probs, market_probs, w)
        return -np.sum(np.log(np.clip(q[idx, outcomes], _EPS, 1.0)))

    res = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def _join(records: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join backtest records to odds on (tournament, home, away).

    The workbook sometimes lists the fixture with home/away swapped
    relative to martj42, so unmatched rows are retried with the teams
    (and the corresponding odds columns) flipped. Raises if any backtest
    match has no odds.
    """
    keys = ["tournament", "home_team", "away_team"]
    merged = records.merge(odds, on=keys, how="left")

    missing = merged["avg_h"].isna()
    if missing.any():
        flipped = odds.rename(
            columns={
                "home_team": "away_team",
                "away_team": "home_team",
                "avg_h": "avg_a",
                "avg_a": "avg_h",
                "max_h": "max_a",
                "max_a": "max_h",
            }
        )
        retry = records.loc[missing.to_numpy()].merge(
            flipped, on=keys, how="left"
        )
        merged = pd.concat(
            [merged.loc[~missing.to_numpy()], retry], ignore_index=True
        )

    still = merged["avg_h"].isna()
    assert not still.any(), (
        "Backtest matches with no odds in the workbook: "
        + ", ".join(
            f"{r.home_team} vs {r.away_team} ({r.tournament})"
            for r in merged.loc[still].itertuples()
        )
    )
    return merged


def _bet_pnl(merged: pd.DataFrame, min_edge: float, price: str) -> dict:
    """Flat 1-unit bets on every outcome where model prob exceeds the
    de-vigged market prob by >= min_edge, settled at ``price`` odds
    ('avg' or 'max')."""
    model = merged[["p_win", "p_draw", "p_loss"]].to_numpy()
    market = devig(merged[["avg_h", "avg_d", "avg_a"]].to_numpy())
    odds = merged[[f"{price}_h", f"{price}_d", f"{price}_a"]].to_numpy()
    won = np.eye(3)[merged["outcome"].to_numpy()].astype(bool)

    bets = (model - market) >= min_edge
    pnl = np.where(won, odds - 1.0, -1.0)[bets].sum()
    n = int(bets.sum())
    return {"n_bets": n, "pnl": float(pnl), "roi": float(pnl / n) if n else np.nan}


def holdout_ladder(
    merged: pd.DataFrame,
    calib=None,
    holdout: str = "WC 2026",
) -> tuple[pd.DataFrame, float]:
    """Log-loss ladder on the holdout tournament: is each layer earning?

    Rows: raw model -> recalibrated model -> market -> blend. The blend
    weight is fit (with calibrated model probs) only on the non-holdout
    tournaments, so the holdout column is honest out-of-sample for every
    row. Returns ``(ladder table, fitted blend weight)``.
    """
    from calibrate import TUNED, apply_calibration

    from backtest import log_loss, rps  # local: avoid cycle at import time

    calib = TUNED if calib is None else calib
    model_cols = ["p_win", "p_draw", "p_loss"]
    market_cols = ["mkt_win", "mkt_draw", "mkt_loss"]

    fit_part = merged.loc[merged["tournament"] != holdout]
    hold = merged.loc[merged["tournament"] == holdout]

    w = fit_blend_weight(
        apply_calibration(fit_part[model_cols].to_numpy(), calib),
        fit_part[market_cols].to_numpy(),
        fit_part["outcome"].to_numpy(),
    )

    raw = hold[model_cols].to_numpy()
    cal = apply_calibration(raw, calib)
    market = hold[market_cols].to_numpy()
    blended = blend(cal, market, w)
    outcomes = hold["outcome"].to_numpy()

    ladder = pd.DataFrame(
        [
            {"probs": "model (raw)", "log_loss": log_loss(raw, outcomes),
             "rps": rps(raw, outcomes)},
            {"probs": "model (calibrated)",
             "log_loss": log_loss(cal, outcomes), "rps": rps(cal, outcomes)},
            {"probs": "market (de-vigged avg)",
             "log_loss": log_loss(market, outcomes),
             "rps": rps(market, outcomes)},
            {"probs": f"blend (w_model={w:.2f})",
             "log_loss": log_loss(blended, outcomes),
             "rps": rps(blended, outcomes)},
        ]
    ).round(4)
    return ladder, w


def compare_to_market(
    records: pd.DataFrame,
    odds: pd.DataFrame | None = None,
    min_edge: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score model vs de-vigged market-average odds on the backtest matches.

    Returns ``(summary, merged)``. summary has one row per tournament plus
    a pooled row: log-loss and RPS for the model and the market on the
    same matches, and a flat-stake betting simulation (1 unit on every
    outcome where the model beats the market by >= min_edge) settled at
    market-average and at best-available (max) prices.

    merged is the per-match join, with market probabilities alongside the
    model's, for inspecting where the disagreements were and how they
    resolved.
    """
    if odds is None:
        odds = download_odds()
    merged = _join(records, odds)

    market = devig(merged[["avg_h", "avg_d", "avg_a"]].to_numpy())
    merged["mkt_win"] = market[:, 0]
    merged["mkt_draw"] = market[:, 1]
    merged["mkt_loss"] = market[:, 2]

    rows = []
    groups = [(name, g) for name, g in merged.groupby("tournament")]
    groups.append(("pooled", merged))
    for name, g in groups:
        model_p = g[["p_win", "p_draw", "p_loss"]].to_numpy()
        market_p = g[["mkt_win", "mkt_draw", "mkt_loss"]].to_numpy()
        outcomes = g["outcome"].to_numpy()
        sim_avg = _bet_pnl(g, min_edge, "avg")
        sim_max = _bet_pnl(g, min_edge, "max")
        rows.append(
            {
                "tournament": name,
                "n": len(g),
                "ll_model": log_loss(model_p, outcomes),
                "ll_market": log_loss(market_p, outcomes),
                "rps_model": rps(model_p, outcomes),
                "rps_market": rps(market_p, outcomes),
                "n_bets": sim_avg["n_bets"],
                "pnl_avg": round(sim_avg["pnl"], 2),
                "roi_avg": round(sim_avg["roi"], 3),
                "pnl_max": round(sim_max["pnl"], 2),
                "roi_max": round(sim_max["roi"], 3),
            }
        )
    summary = pd.DataFrame(rows).round(4)
    return summary, merged
