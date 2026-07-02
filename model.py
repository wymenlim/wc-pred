"""Dixon-Coles Poisson model for international football match prediction.

Interface contract:
    fit(df) -> FittedModel
        Fit attack/defense ratings + home advantage + Dixon-Coles tau on a
        weighted training DataFrame (columns: date, home_team, away_team,
        home_score, away_score, neutral, weight).

    FittedModel.expected_goals(team_a, team_b, neutral) -> (lambda_a, lambda_b)
        Poisson goal rates for a 90-minute match between team_a and team_b.
        Home advantage for team_a applies only when neutral is False.

Model:
    lambda_home = exp(intercept + attack_home - defense_away
                      + home_adv * [not neutral])
    lambda_away = exp(intercept + attack_away - defense_home)

with the Dixon-Coles tau correction on the 0-0 / 1-0 / 0-1 / 1-1 cells:

    tau(0,0) = 1 - lam*mu*rho     tau(1,0) = 1 + mu*rho
    tau(0,1) = 1 + lam*rho        tau(1,1) = 1 - rho

Fitting is weighted maximum likelihood (scipy L-BFGS-B with analytic
gradients). A small L2 penalty on attack/defense serves two purposes: it
pins down the two shift degeneracies (a -> a+k, d -> d+k; and moving a
constant between the intercept and all attacks — both leave every rate
unchanged, so the unpenalized likelihood has flat directions) and it
shrinks teams whose matches have decayed to near-zero total weight toward
an average team instead of letting their parameters run off. The global
intercept is deliberately NOT penalized: it absorbs the baseline scoring
rate, so shrinkage compresses the spread of team ratings without dragging
every goal rate toward exp(0) = 1.0 (which biased totals low before the
intercept existed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RHO_BOUNDS = (-0.15, 0.15)
# tau can dip negative for extreme lam*mu; the floor keeps log(tau) finite.
_TAU_FLOOR = 1e-10
# Clip on the linear predictor log(rate): exp(5) ~ 148 goals is far beyond
# any real match, exp(-30) is effectively zero. Keeps line-search excursions
# from under/overflowing exp() into log(0)/inf NaNs that break L-BFGS-B; the
# optimum never sits in the clipped region, so the (technically inconsistent)
# gradient there only ever points the search back toward sanity.
_Z_BOUNDS = (-30.0, 5.0)


@dataclass
class FittedModel:
    """A fitted Dixon-Coles Poisson model. Constructed by :func:`fit`."""

    teams: list[str]
    attack: np.ndarray
    defense: np.ndarray
    intercept: float
    home_adv: float
    rho: float
    log_likelihood: float
    n_matches: int
    converged: bool
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._index = {t: i for i, t in enumerate(self.teams)}

    def expected_goals(
        self, team_a: str, team_b: str, neutral: bool
    ) -> tuple[float, float]:
        """Return (lambda_a, lambda_b): expected goals for each team.

        team_a is the home side; the home-advantage term is applied to
        team_a only when ``neutral`` is False.
        """
        ia, ib = self._team_index(team_a), self._team_index(team_b)
        h = 0.0 if neutral else self.home_adv
        lam_a = np.exp(self.intercept + self.attack[ia] - self.defense[ib] + h)
        lam_b = np.exp(self.intercept + self.attack[ib] - self.defense[ia])
        return float(lam_a), float(lam_b)

    def ratings(self) -> pd.DataFrame:
        """Team ratings sorted by overall strength (attack + defense)."""
        return (
            pd.DataFrame(
                {
                    "team": self.teams,
                    "attack": self.attack,
                    "defense": self.defense,
                    "strength": self.attack + self.defense,
                }
            )
            .sort_values("strength", ascending=False)
            .reset_index(drop=True)
        )

    def _team_index(self, team: str) -> int:
        try:
            return self._index[team]
        except KeyError:
            raise KeyError(
                f"Unknown team {team!r} — not present in training data."
            ) from None


def _nll_grad(params, hi, ai, x, y, w, home, n_teams, l2):
    """Weighted negative log-likelihood and its gradient.

    params = [attack (T), defense (T), intercept, home_adv, rho].
    """
    a = params[:n_teams]
    d = params[n_teams : 2 * n_teams]
    c = params[-3]
    h = params[-2]
    rho = params[-1]

    z_lam = np.clip(c + a[hi] - d[ai] + h * home, *_Z_BOUNDS)
    z_mu = np.clip(c + a[ai] - d[hi], *_Z_BOUNDS)
    lam = np.exp(z_lam)
    mu = np.exp(z_mu)

    # Dixon-Coles tau on the four low-score cells.
    tau = np.ones_like(lam)
    tau_lam = np.zeros_like(lam)
    tau_mu = np.zeros_like(lam)
    tau_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau_lam[m00] = -mu[m00] * rho
    tau_mu[m00] = -lam[m00] * rho
    tau_rho[m00] = -lam[m00] * mu[m00]

    tau[m01] = 1.0 + lam[m01] * rho
    tau_lam[m01] = rho
    tau_rho[m01] = lam[m01]

    tau[m10] = 1.0 + mu[m10] * rho
    tau_mu[m10] = rho
    tau_rho[m10] = mu[m10]

    tau[m11] = 1.0 - rho
    tau_rho[m11] = -1.0

    # Where tau is floored, zero its gradients as well: the floored region
    # is flat by construction, and dividing tau_* by the floor instead
    # produces ~1e10 gradient spikes that abort the L-BFGS-B line search.
    floored = tau < _TAU_FLOOR
    tau_lam[floored] = 0.0
    tau_mu[floored] = 0.0
    tau_rho[floored] = 0.0
    tau = np.maximum(tau, _TAU_FLOOR)

    # x*z instead of x*log(lam): identical value, but immune to log(0)
    # when exp() underflows during a line-search excursion.
    ll = np.sum(w * (x * z_lam - lam + y * z_mu - mu + np.log(tau)))
    nll = -ll + l2 * (np.sum(a**2) + np.sum(d**2))

    # d(ll)/d(lam) * lam etc., accumulated back onto team parameters.
    g_lam = w * (x - lam + lam * tau_lam / tau)  # = dll/d(log lam)
    g_mu = w * (y - mu + mu * tau_mu / tau)

    grad_a = np.bincount(hi, g_lam, n_teams) + np.bincount(ai, g_mu, n_teams)
    grad_d = -np.bincount(ai, g_lam, n_teams) - np.bincount(hi, g_mu, n_teams)
    grad_c = np.sum(g_lam) + np.sum(g_mu)
    grad_h = np.sum(g_lam * home)
    grad_rho = np.sum(w * tau_rho / tau)

    grad = np.concatenate(
        [
            -grad_a + 2.0 * l2 * a,
            -grad_d + 2.0 * l2 * d,
            [-grad_c, -grad_h, -grad_rho],
        ]
    )
    return nll, grad


def fit(df: pd.DataFrame, l2: float = 0.25) -> FittedModel:
    """Fit the model on a prepared training DataFrame (see data.prepare).

    Weighted maximum likelihood via scipy L-BFGS-B with analytic gradients.
    ``l2`` is the ridge penalty on attack/defense ratings (see module
    docstring for why it must be > 0). The default is the tune.py grid
    winner; the original 1.0 over-compressed the rating spread.
    """
    if l2 <= 0:
        raise ValueError("l2 must be > 0 (identifiability requires it).")

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    hi = df["home_team"].map(index).to_numpy()
    ai = df["away_team"].map(index).to_numpy()
    x = df["home_score"].to_numpy(dtype=float)
    y = df["away_score"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    home = (~df["neutral"].to_numpy(dtype=bool)).astype(float)

    x0 = np.zeros(2 * n_teams + 3)
    x0[-3] = 0.3  # intercept ~ log(mean goals per team per match)
    x0[-2] = 0.25  # home advantage
    x0[-1] = -0.05  # rho

    bounds = [(None, None)] * (2 * n_teams + 2) + [RHO_BOUNDS]

    res = minimize(
        _nll_grad,
        x0,
        args=(hi, ai, x, y, w, home, n_teams, l2),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000},
    )

    return FittedModel(
        teams=teams,
        attack=res.x[:n_teams].copy(),
        defense=res.x[n_teams : 2 * n_teams].copy(),
        intercept=float(res.x[-3]),
        home_adv=float(res.x[-2]),
        rho=float(res.x[-1]),
        log_likelihood=float(-res.fun),
        n_matches=len(df),
        converged=bool(res.success),
    )
