"""Backtesting & calibration analysis for the World Cup predictor (Part 5).

For each World Cup, fit the model only on data available before the
tournament (data.prepare's ``as_of`` leakage guard), predict every
group-stage match, and score the predictions:

    - log-loss and Ranked Probability Score (RPS) on win/draw/loss, vs two
      naive baselines (uniform 1/3s, and the empirical W/D/L frequency of
      competitive neutral-venue matches known at the cutoff),
    - exact-scoreline hit rate vs always predicting the historically most
      common score,
    - expected-goals MAE and total-goals bias,
    - pooled calibration table / reliability plot.

Group stage only, deliberately: this dataset records knockout scores
*including extra time*, so a 90-minute draw settled in ET shows up as a
win — scoring our 90-minute model against that would corrupt the metrics.
Group-stage matches cannot go to extra time.

Besides the headline World Cup backtests (GROUP_STAGES), :func:`editions`
auto-detects every completed World Cup / Euro / Copa América / AFCON
edition in the dataset and :func:`backtest_editions` scores them all —
the widened evaluation set used for hyperparameter tuning (see tune.py)
and for fitting the probability recalibration (see calibrate.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import prepare
from model import fit
from predict import predict

# name: (fit cutoff / as_of, group-stage start, group-stage end)
GROUP_STAGES = {
    "WC 2018": ("2018-06-13", "2018-06-14", "2018-06-28"),
    "WC 2022": ("2022-11-19", "2022-11-20", "2022-12-02"),
    "WC 2026": ("2026-06-10", "2026-06-11", "2026-06-27"),
}

# exact martj42 tournament name -> label prefix
TOURNAMENTS = {
    "FIFA World Cup": "WC",
    "UEFA Euro": "Euro",
    "Copa América": "Copa",
    "African Cup of Nations": "AFCON",
}

_OUTCOMES = ["win", "draw", "loss"]  # from the listed home side's view
_EPS = 1e-10


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log probability assigned to the actual outcome.

    probs: (n, 3) rows [p_win, p_draw, p_loss]; outcomes: (n,) in {0, 1, 2}.
    """
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Ranked Probability Score over the ordered outcomes win<draw<loss.

    Per match: (1/2) * sum_k (cum_pred_k - cum_obs_k)^2 over k = 1, 2.
    0 is perfect; lower is better.
    """
    cum_pred = np.cumsum(probs, axis=1)[:, :2]
    onehot = np.eye(3)[outcomes]
    cum_obs = np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_pred - cum_obs) ** 2, axis=1) / 2.0))


def _outcome_index(home_goals, away_goals):
    return np.where(
        home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2)
    )


def _reference_matches(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Competitive neutral-venue matches known at the cutoff (baseline pool)."""
    scored = df["home_score"].notna() & df["away_score"].notna()
    return df.loc[
        scored
        & df["neutral"]
        & (df["tournament"] != "Friendly")
        & (df["date"] >= pd.Timestamp("2000-01-01"))
        & (df["date"] <= cutoff)
    ]


def run_edition(
    df: pd.DataFrame,
    label: str,
    cutoff: pd.Timestamp,
    test: pd.DataFrame,
    half_life_days: float | None = None,
    friendly_weight: float | None = None,
    l2: float | None = None,
) -> pd.DataFrame:
    """Fit at ``cutoff`` and predict the given ``test`` matches.

    The weighting/regularization parameters default to None = whatever
    data.prepare / model.fit default to (the tuned values). Returns one
    record per match; raises if the leakage guard is violated.
    """
    prep_kw = {
        k: v
        for k, v in {
            "half_life_days": half_life_days,
            "friendly_weight": friendly_weight,
        }.items()
        if v is not None
    }
    training, _ = prepare(df, as_of=cutoff, **prep_kw)
    assert (
        training["date"].max() <= cutoff < test["date"].min()
    ), "leakage guard violated"
    model = fit(training, **({"l2": l2} if l2 is not None else {}))
    assert model.converged, f"{label}: fit did not converge"

    records = []
    for f in test.itertuples(index=False):
        p = predict(
            model, f.home_team, f.away_team,
            neutral=bool(f.neutral), knockout=False,
        )
        records.append(
            {
                "tournament": label,
                "date": f.date,
                "home_team": f.home_team,
                "away_team": f.away_team,
                "lambda_home": p.lambda_a,
                "lambda_away": p.lambda_b,
                "p_win": p.p_win_a,
                "p_draw": p.p_draw,
                "p_loss": p.p_win_b,
                "pred_score": f"{p.score[0]}-{p.score[1]}",
                "actual_score": f"{f.home_score}-{f.away_score}",
                "home_goals": int(f.home_score),
                "away_goals": int(f.away_score),
            }
        )
    rec = pd.DataFrame(records)
    rec["outcome"] = _outcome_index(rec["home_goals"], rec["away_goals"])
    rec["score_hit"] = rec["pred_score"] == rec["actual_score"]
    return rec


def run_backtest(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fit at the pre-tournament cutoff and predict one GROUP_STAGES entry."""
    cutoff, start, end = (pd.Timestamp(d) for d in GROUP_STAGES[name])
    scored = df["home_score"].notna() & df["away_score"].notna()
    test = df.loc[
        scored
        & (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= start)
        & (df["date"] <= end)
    ]
    return run_edition(df, name, cutoff, test)


def _group_stage(edition: pd.DataFrame) -> pd.DataFrame:
    """Group-stage matches of one tournament edition.

    A match is group-stage iff both teams have fewer than 3 prior
    appearances in the edition. In every WC/Euro/Copa/AFCON format since
    2006 a team plays at least 3 group matches before any knockout, so
    this can never let a knockout match through; it only (conservatively)
    drops 4th group matches in the occasional groups-of-5 format, e.g.
    Copa América 2021.
    """
    counts: dict[str, int] = {}
    keep = []
    for f in edition.itertuples(index=False):
        keep.append(
            counts.get(f.home_team, 0) < 3 and counts.get(f.away_team, 0) < 3
        )
        counts[f.home_team] = counts.get(f.home_team, 0) + 1
        counts[f.away_team] = counts.get(f.away_team, 0) + 1
    return edition.loc[np.array(keep)]


def editions(
    df: pd.DataFrame, first_year: int = 2006
) -> list[tuple[str, pd.Timestamp, pd.DataFrame]]:
    """Auto-detect completed major-tournament editions in the dataset.

    Returns [(label, fit cutoff, group-stage matches)] sorted by date,
    e.g. ("Euro 2021", 2021-06-10, 36 matches). Editions of the same
    tournament are split on gaps > 60 days between consecutive matches
    (editions are years apart; a single edition never pauses that long).
    The cutoff is the day before the edition's first match. Labels use
    the year the edition was *played* (so Euro 2020 -> "Euro 2021").
    """
    scored = df.loc[df["home_score"].notna() & df["away_score"].notna()]
    out = []
    for name, short in TOURNAMENTS.items():
        t = scored.loc[scored["tournament"] == name].sort_values("date")
        if t.empty:
            continue
        gap = t["date"].diff().dt.days.gt(60).cumsum()
        for _, ed in t.groupby(gap):
            first = ed["date"].iloc[0]
            if first.year < first_year:
                continue
            label = f"{short} {first.year}"
            out.append((label, first - pd.Timedelta(days=1), _group_stage(ed)))
    return sorted(out, key=lambda e: e[1])


def backtest_editions(
    df: pd.DataFrame,
    first_year: int = 2006,
    half_life_days: float | None = None,
    friendly_weight: float | None = None,
    l2: float | None = None,
) -> pd.DataFrame:
    """Leak-free per-match records for every detected edition's group stage."""
    recs = [
        run_edition(
            df, label, cutoff, test,
            half_life_days=half_life_days,
            friendly_weight=friendly_weight,
            l2=l2,
        )
        for label, cutoff, test in editions(df, first_year=first_year)
    ]
    return pd.concat(recs, ignore_index=True)


def _metrics_row(name, rec, df, cutoff):
    probs = rec[["p_win", "p_draw", "p_loss"]].to_numpy()
    outcomes = rec["outcome"].to_numpy()
    n = len(rec)

    ref = _reference_matches(df, pd.Timestamp(cutoff))
    ref_outcomes = _outcome_index(
        ref["home_score"].to_numpy(int), ref["away_score"].to_numpy(int)
    )
    emp = np.bincount(ref_outcomes, minlength=3) / len(ref_outcomes)
    emp_probs = np.tile(emp, (n, 1))
    uni_probs = np.full((n, 3), 1.0 / 3.0)

    ref_scores = (
        ref["home_score"].astype(str) + "-" + ref["away_score"].astype(str)
    )
    modal_score = ref_scores.mode().iloc[0]

    goals_mae = float(
        np.mean(
            np.abs(rec["lambda_home"] - rec["home_goals"])
            + np.abs(rec["lambda_away"] - rec["away_goals"])
        )
        / 2.0
    )
    total_bias = float(
        np.mean(
            (rec["lambda_home"] + rec["lambda_away"])
            - (rec["home_goals"] + rec["away_goals"])
        )
    )

    return {
        "tournament": name,
        "n": n,
        "log_loss": log_loss(probs, outcomes),
        "ll_uniform": log_loss(uni_probs, outcomes),
        "ll_empirical": log_loss(emp_probs, outcomes),
        "rps": rps(probs, outcomes),
        "rps_uniform": rps(uni_probs, outcomes),
        "rps_empirical": rps(emp_probs, outcomes),
        "score_hit": float(rec["score_hit"].mean()),
        "score_hit_modal": float((rec["actual_score"] == modal_score).mean()),
        "xg_mae": goals_mae,
        "total_goals_bias": total_bias,
    }


def backtest_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every backtest. Returns (summary metrics, per-match records)."""
    all_rec = []
    rows = []
    for name, (cutoff, _, _) in GROUP_STAGES.items():
        rec = run_backtest(df, name)
        all_rec.append(rec)
        rows.append(_metrics_row(name, rec, df, cutoff))

    records = pd.concat(all_rec, ignore_index=True)
    # pooled row: baselines from the earliest cutoff (strictest, leak-free)
    rows.append(
        _metrics_row("pooled", records, df, GROUP_STAGES["WC 2018"][0])
    )
    return pd.DataFrame(rows).round(4), records


def calibration_table(records: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Pool all per-outcome probabilities and bin them by predicted prob.

    Each match contributes three (predicted prob, happened) pairs. A
    calibrated model has bin_mean_predicted ~= bin_empirical_rate.
    """
    probs = records[["p_win", "p_draw", "p_loss"]].to_numpy().ravel()
    onehot = np.eye(3)[records["outcome"].to_numpy()].ravel()
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin": f"[{b / n_bins:.1f}, {(b + 1) / n_bins:.1f})",
                "n": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()),
                "empirical_rate": float(onehot[mask].mean()),
            }
        )
    return pd.DataFrame(out)


def plot_calibration(records: pd.DataFrame, n_bins: int = 10, ax=None):
    """Reliability plot of the pooled outcome probabilities."""
    import matplotlib.pyplot as plt

    tab = calibration_table(records, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(tab["mean_predicted"], tab["empirical_rate"], "o-", label="model")
    for _, r in tab.iterrows():
        ax.annotate(
            str(r["n"]), (r["mean_predicted"], r["empirical_rate"]),
            textcoords="offset points", xytext=(6, -4), fontsize=8,
        )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_title("Calibration (pooled W/D/L probs, n per bin)")
    ax.legend()
    return ax
