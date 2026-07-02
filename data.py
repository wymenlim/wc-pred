"""Data loading & preparation for the World Cup 2026 match predictor.

Downloads the martj42 dataset of international results (1872-present),
splits it into a weighted training set (completed matches) and a prediction
queue (upcoming fixtures with NA scores), and validates team-name coverage.

Typical use:
    from data import load
    training, upcoming = load()

The ``as_of`` parameter on :func:`prepare` / :func:`load` exists so that
backtests can build leakage-free training sets, e.g.
``prepare(df, as_of="2018-06-13")`` — no match after that date will appear
in training, and time-decay weights are computed relative to that date.
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

PATCHES_PATH = "patches.csv"

_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day


def download_results(
    cache_path: str = "data/results.csv", refresh: bool = False
) -> pd.DataFrame:
    """Download (or reuse a cached copy of) the full results dataset.

    Re-downloads if ``refresh`` is True, the cache file is missing, or the
    cache is older than 1 day. Returns a DataFrame with ``date`` parsed as
    datetime, ``neutral`` as bool, and scores as nullable integers (upcoming
    fixtures have missing scores).
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
        with urllib.request.urlopen(RESULTS_URL) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)

    df = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"home_score": "Int64", "away_score": "Int64"},
    )
    df["neutral"] = df["neutral"].astype(bool)
    return _apply_patches(df)


def _apply_patches(
    df: pd.DataFrame, patches_path: str = PATCHES_PATH
) -> pd.DataFrame:
    """Fill in results the upstream CSV hasn't backfilled yet.

    ``patches.csv`` (same columns as the upstream file; knockout scores
    include extra time, matching the upstream convention) is applied after
    download. Upstream wins: a patch only fills rows whose scores are still
    missing, so stale patches become no-ops once upstream catches up.
    Patch rows with no matching upstream fixture are appended.
    """
    path = Path(patches_path)
    if not path.exists():
        return df

    patches = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"home_score": "Int64", "away_score": "Int64"},
    )
    patches["neutral"] = patches["neutral"].astype(bool)

    unmatched = []
    for i, p in patches.iterrows():
        match = (
            (df["date"] == p["date"])
            & (df["home_team"] == p["home_team"])
            & (df["away_team"] == p["away_team"])
        )
        if match.any():
            fill = match & df["home_score"].isna()
            df.loc[fill, ["home_score", "away_score"]] = (
                p["home_score"], p["away_score"],
            )
        else:
            unmatched.append(i)

    if unmatched:
        df = pd.concat(
            [df, patches.loc[unmatched]], ignore_index=True
        ).sort_values("date", ignore_index=True)
    return df


def prepare(
    df: pd.DataFrame,
    start: str = "2000-01-01",
    as_of=None,
    half_life_days: float = 730.0,
    friendly_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the raw results into ``(training, upcoming)``.

    The half_life_days / friendly_weight defaults are the tune.py grid
    winners (leak-free log-loss over 28 major-tournament editions
    2006-2025; see tune_results.csv). Down-weighting friendlies, the
    original hand-picked 0.5, turned out to hurt.

    training: completed matches with ``start <= date <= as_of``, with a
        ``weight`` column = time_decay * importance, where
        time_decay = 0.5 ** (days_between(as_of, date) / half_life_days)
        and importance = friendly_weight for friendlies, 1.0 otherwise.
        Columns: date, home_team, away_team, home_score, away_score,
        neutral, weight.

    upcoming: fixtures with missing scores and ``date >= as_of``.
        Columns: date, home_team, away_team, neutral, tournament, city,
        country.

    ``as_of`` defaults to today. No match after ``as_of`` can appear in
    training (leakage guard for backtesting).
    """
    as_of = pd.Timestamp.today().normalize() if as_of is None else pd.Timestamp(as_of)
    start = pd.Timestamp(start)

    scored = df["home_score"].notna() & df["away_score"].notna()

    training = df.loc[
        scored & (df["date"] >= start) & (df["date"] <= as_of)
    ].copy()
    days_ago = (as_of - training["date"]).dt.days
    time_decay = 0.5 ** (days_ago / half_life_days)
    friendly = training["tournament"] == "Friendly"
    importance = np.where(friendly, friendly_weight, 1.0)
    training["weight"] = time_decay * importance
    training = training[
        ["date", "home_team", "away_team", "home_score", "away_score",
         "neutral", "weight"]
    ].reset_index(drop=True)

    upcoming = df.loc[~scored & (df["date"] >= as_of)].copy()
    upcoming = upcoming[
        ["date", "home_team", "away_team", "neutral", "tournament", "city",
         "country"]
    ].reset_index(drop=True)

    return training, upcoming


def validate_teams(
    training: pd.DataFrame, upcoming: pd.DataFrame, min_matches: int = 10
) -> None:
    """Assert every team in ``upcoming`` has >= min_matches training matches.

    Catches team-name mismatches between eras/datasets. Raises
    AssertionError listing every offending team and its match count.
    """
    counts = (
        pd.concat([training["home_team"], training["away_team"]])
        .value_counts()
    )
    upcoming_teams = pd.unique(
        pd.concat([upcoming["home_team"], upcoming["away_team"]])
    )
    offenders = {
        team: int(counts.get(team, 0))
        for team in upcoming_teams
        if counts.get(team, 0) < min_matches
    }
    assert not offenders, (
        f"Teams in upcoming fixtures with fewer than {min_matches} training "
        f"matches (possible team-name mismatch): "
        + ", ".join(f"{t!r} ({n})" for t, n in sorted(offenders.items()))
    )


def load(
    start: str = "2000-01-01",
    as_of=None,
    refresh: bool = False,
    half_life_days: float = 730.0,
    friendly_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: download -> prepare -> validate -> return."""
    df = download_results(refresh=refresh)
    training, upcoming = prepare(
        df,
        start=start,
        as_of=as_of,
        half_life_days=half_life_days,
        friendly_weight=friendly_weight,
    )
    validate_teams(training, upcoming)
    return training, upcoming
