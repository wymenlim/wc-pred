"""Prediction outputs for the World Cup 2026 match predictor (Part 4).

Turns a fitted Dixon-Coles model into the project's outputs per match:

    1. per-team expected goals (the model's Poisson rates),
    2. the most likely full-time (90-minute) scoreline with its probability,
    3. for knockouts, P(advances) per team:
       P(win in 90) + P(draw in 90) * P(wins extra time / penalties),
       with extra time modeled as a 30-minute Poisson extension at rate
       lambda/3 per team (independent Poissons — the tau correction is a
       90-minute low-score effect and is not reapplied) and penalties 50/50.

Typical use:
    from predict import predict, predict_fixtures
    p = predict(model, "Spain", "Austria", neutral=True, knockout=True)
    print(p.summary())
    predict_fixtures(model, upcoming)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_GOALS = 10
ET_FACTOR = 30.0 / 90.0  # extra time = 30 minutes at the same scoring rate


def _modal_cell(m: np.ndarray) -> tuple[tuple[int, int], float]:
    """The single most likely (score_a, score_b) cell of a score matrix."""
    i, j = np.unravel_index(np.argmax(m), m.shape)
    return (int(i), int(j)), float(m[i, j])


def modal_by_outcome(
    m: np.ndarray,
) -> dict[str, tuple[tuple[int, int], float]]:
    """Most likely scoreline *within* each 1X2 outcome.

    Returns {"win": ((i, j), prob), "draw": ..., "loss": ...} where "win" is
    team_a winning. Probabilities are joint (directly comparable; with every
    other cell they sum to 1). This keeps the reported scorelines consistent
    with the win/draw/loss call — the global modal score can otherwise be a
    draw even when one side is a clear favorite.
    """
    diag = np.zeros_like(m)
    np.fill_diagonal(diag, np.diag(m))
    return {
        "win": _modal_cell(np.tril(m, -1)),
        "draw": _modal_cell(diag),
        "loss": _modal_cell(np.triu(m, 1)),
    }


def score_matrix(
    lam_a: float, lam_b: float, rho: float = 0.0, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Joint P(score_a = i, score_b = j) for i, j in 0..max_goals.

    Independent Poissons with the Dixon-Coles tau correction applied to the
    four low-score cells, then renormalized (tau plus truncation leave the
    mass a hair off 1).
    """
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam_a), poisson.pmf(goals, lam_b))
    m[0, 0] *= 1.0 - lam_a * lam_b * rho
    m[0, 1] *= 1.0 + lam_a * rho
    m[1, 0] *= 1.0 + lam_b * rho
    m[1, 1] *= 1.0 - rho
    np.clip(m, 0.0, None, out=m)
    return m / m.sum()


@dataclass
class Prediction:
    team_a: str
    team_b: str
    neutral: bool
    lambda_a: float
    lambda_b: float
    matrix: np.ndarray
    score: tuple[int, int]
    score_prob: float
    # most likely scoreline within each outcome: {"win"/"draw"/"loss": ((i,j), p)}
    modal_scores: dict[str, tuple[tuple[int, int], float]]
    p_win_a: float
    p_draw: float
    p_win_b: float
    p_advance_a: float | None  # None for non-knockout matches
    p_advance_b: float | None

    def summary(self) -> str:
        win = self.modal_scores["win"]
        drw = self.modal_scores["draw"]
        los = self.modal_scores["loss"]
        lines = [
            f"{self.team_a} vs {self.team_b}"
            + ("" if self.neutral else f" ({self.team_a} at home)"),
            f"  Expected goals:  {self.team_a} {self.lambda_a:.2f} — "
            f"{self.lambda_b:.2f} {self.team_b}",
            f"  Win/Draw/Win: {self.p_win_a:.1%} / {self.p_draw:.1%} / "
            f"{self.p_win_b:.1%}",
            "  Likeliest score by outcome:",
            f"    {self.team_a} win  {win[0][0]}-{win[0][1]} ({win[1]:.1%})",
            f"    draw       {drw[0][0]}-{drw[0][1]} ({drw[1]:.1%})",
            f"    {self.team_b} win  {los[0][0]}-{los[0][1]} ({los[1]:.1%})",
        ]
        if self.p_advance_a is not None:
            lines.append(
                f"  Advances: {self.team_a} {self.p_advance_a:.1%}, "
                f"{self.team_b} {self.p_advance_b:.1%}"
            )
        return "\n".join(lines)


def _advance_given_draw(lam_a: float, lam_b: float) -> float:
    """P(team_a advances | 90-minute draw): extra time, then penalties."""
    et = score_matrix(lam_a * ET_FACTOR, lam_b * ET_FACTOR, rho=0.0)
    p_a_et = np.tril(et, -1).sum()  # team_a outscores team_b in ET
    p_draw_et = np.trace(et)
    return p_a_et + 0.5 * p_draw_et


def predict(
    model,
    team_a: str,
    team_b: str,
    neutral: bool = True,
    knockout: bool = True,
    max_goals: int = MAX_GOALS,
) -> Prediction:
    """Predict one match. team_a is the 'home' side of the fixture."""
    lam_a, lam_b = model.expected_goals(team_a, team_b, neutral)
    m = score_matrix(lam_a, lam_b, rho=model.rho, max_goals=max_goals)

    (i, j), _ = _modal_cell(m)
    modal = modal_by_outcome(m)
    p_win_a = np.tril(m, -1).sum()
    p_draw = np.trace(m)
    p_win_b = np.triu(m, 1).sum()

    p_adv_a = p_adv_b = None
    if knockout:
        adv_a_given_draw = _advance_given_draw(lam_a, lam_b)
        p_adv_a = p_win_a + p_draw * adv_a_given_draw
        p_adv_b = p_win_b + p_draw * (1.0 - adv_a_given_draw)

    return Prediction(
        team_a=team_a,
        team_b=team_b,
        neutral=neutral,
        lambda_a=lam_a,
        lambda_b=lam_b,
        matrix=m,
        score=(int(i), int(j)),
        score_prob=float(m[i, j]),
        modal_scores=modal,
        p_win_a=float(p_win_a),
        p_draw=float(p_draw),
        p_win_b=float(p_win_b),
        p_advance_a=None if p_adv_a is None else float(p_adv_a),
        p_advance_b=None if p_adv_b is None else float(p_adv_b),
    )


def predict_fixtures(
    model, upcoming: pd.DataFrame, knockout: bool = True
) -> pd.DataFrame:
    """Predict every fixture in an ``upcoming`` DataFrame (see data.prepare)."""
    rows = []
    for f in upcoming.itertuples(index=False):
        p = predict(
            model, f.home_team, f.away_team,
            neutral=bool(f.neutral), knockout=knockout,
        )
        row = {
            "date": f.date,
            "match": f"{f.home_team} vs {f.away_team}",
            "xg_a": round(p.lambda_a, 2),
            "xg_b": round(p.lambda_b, 2),
            "score": f"{p.score[0]}-{p.score[1]}",
            "score_prob": round(p.score_prob, 3),
            "likeliest_by_outcome": " | ".join(
                f"{k[0].upper()} {ij[0]}-{ij[1]} ({prob:.0%})"
                for k, (ij, prob) in p.modal_scores.items()
            ),
            "p_win_a": round(p.p_win_a, 3),
            "p_draw": round(p.p_draw, 3),
            "p_win_b": round(p.p_win_b, 3),
        }
        if knockout:
            row["p_adv_a"] = round(p.p_advance_a, 3)
            row["p_adv_b"] = round(p.p_advance_b, 3)
        rows.append(row)
    return pd.DataFrame(rows)
