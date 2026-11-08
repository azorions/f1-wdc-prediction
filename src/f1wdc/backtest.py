"""Leak-free backtest: predict each past season from data before it, score vs reality."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1wdc.config import points_system, transition_type
from f1wdc.model.ratings import RatingConfig
from f1wdc.pipeline import Data, Hyper, compute_ratings, predict

TRAIN_YEARS = list(range(1997, 2020))
HOLDOUT_YEARS = list(range(2020, 2027))
SMOOTH = 0.5   # additive smoothing for simulated win counts


def season_points(entries: pd.DataFrame, season: int) -> pd.Series:
    x = entries[(entries["season"] == season) & entries["session"].isin(["race", "sprint"])]
    return x.groupby("driverId")["points"].sum().sort_values(ascending=False)


def smoothed(p: np.ndarray, n: int) -> np.ndarray:
    counts = p * n
    return (counts + SMOOTH) / (n + SMOOTH * len(p))


def score(p: np.ndarray, champ_idx: int) -> tuple[float, float, int]:
    """(log-loss, Brier, rank of champion in the probability order, 1-based)."""
    y = np.zeros_like(p)
    y[champ_idx] = 1
    rank = int((p > p[champ_idx]).sum()) + 1
    return float(-np.log(p[champ_idx])), float(((p - y) ** 2).sum()), rank


def baseline_points_share(entries: pd.DataFrame, season: int, drivers: list[str], temp: float) -> np.ndarray:
    prev = season_points(entries, season - 1)
    share = np.array([prev.get(d, 0.0) for d in drivers]) / max(prev.max(), 1e-9)
    z = np.exp(temp * (share - share.max()))
    return z / z.sum()


def baseline_top_teams(entries: pd.DataFrame, season: int, grid: dict[str, list[str]]) -> np.ndarray:
    prev = entries[(entries["season"] == season - 1) & entries["session"].isin(["race", "sprint"])]
    top = prev.groupby("lineage")["points"].sum().sort_values(ascending=False).index[:2]
    drivers = [d for t in grid for d in grid[t]]
    fav = np.array([t in top for t in grid for _ in grid[t]], float)
    n_fav = max(fav.sum(), 1)
    p = np.where(fav > 0, 0.95 / n_fav, 0.05 / max(len(drivers) - n_fav, 1))
    return p / p.sum()


@dataclass
class YearResult:
    row: dict
    drivers: pd.DataFrame   # per-driver predictions vs actual for calibration


def evaluate_year(data: Data, ratings, year: int, hyper: Hyper, n_sims: int, seed: int,
                  carryover: float | None = None, as_of_round: int = 0) -> YearResult:
    grid = data.grid_at(year, as_of_round or None)
    n_rounds, sprints = data.calendar(year) or (data.n_rounds[year], [])
    race_pts, sprint_pts = points_system(year)
    pred = predict(data, ratings, year, grid, n_rounds, sprints, race_pts, sprint_pts, hyper,
                   n_sims=n_sims, seed=seed, carryover=carryover, as_of_round=as_of_round)
    tab = pred.sim.driver_table()
    drivers = pred.sim.drivers
    p = smoothed(pred.sim.p_wdc(), n_sims)
    actual = season_points(data.entries, year)
    champion = actual.index[0]
    act = np.array([actual.get(d, 0.0) for d in drivers])
    ci = drivers.index(champion)
    ll, brier, rank = score(p, ci)
    exp_pts = tab.set_index("driverId").loc[drivers, "exp_points"].to_numpy()
    rho_s = float(spearmanr(exp_pts, act).statistic)
    order = np.argsort(-p)
    act_rank = pd.Series(act, index=drivers).rank(ascending=False, method="first")
    per_driver = pd.DataFrame({
        "season": year, "driverId": drivers, "p_wdc": p,
        "p_top3": tab.set_index("driverId").loc[drivers, "p_top3"].to_numpy(),
        "exp_points": exp_pts, "actual_points": act,
        "actual_top3": (act_rank.loc[drivers] <= 3).to_numpy(),
        "champion": [d == champion for d in drivers],
    })
    row = {
        "season": year, "type": transition_type(year), "champion": champion,
        "p_champion": p[ci], "rank_champion": rank, "logloss": ll, "brier": brier, "spearman": rho_s,
        "top_pick": drivers[order[0]], "p_top_pick": p[order[0]],
        "provisional": data.completed_rounds(year) < n_rounds,
    }
    return YearResult(row, per_driver)


def add_baselines(data: Data, df: pd.DataFrame, temp: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        y = int(r["season"])
        grid = data.grid_at(y)
        drivers = [d for t in grid for d in grid[t]]
        ci = drivers.index(r["champion"])
        pa = np.clip(baseline_points_share(data.entries, y, drivers, temp), 1e-4, None)
        pb = baseline_top_teams(data.entries, y, grid)
        rows.append({"season": y, "logloss_share": score(pa / pa.sum(), ci)[0], "logloss_top2": score(pb, ci)[0]})
    return df.merge(pd.DataFrame(rows), on="season")


def tune_share_temperature(data: Data, years: list[int]) -> float:
    best = (np.inf, 1.0)
    for temp in np.arange(1, 31, 1.0):
        lls = []
        for y in years:
            grid = data.grid_at(y)
            drivers = [d for t in grid for d in grid[t]]
            champ = season_points(data.entries, y).index[0]
            p = np.clip(baseline_points_share(data.entries, y, drivers, temp), 1e-4, None)
            lls.append(score(p / p.sum(), drivers.index(champ))[0])
        best = min(best, (float(np.mean(lls)), float(temp)))
    return best[1]


def run_backtest(data: Data, hyper: Hyper, years: list[int], n_sims: int = 5000, seed: int = 7,
                 progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = compute_ratings(data, hyper)
    rows, drivers = [], []
    for y in years:
        if progress:
            progress(y)
        res = evaluate_year(data, ratings, y, hyper, n_sims, seed)
        rows.append(res.row)
        drivers.append(res.drivers)
    return pd.DataFrame(rows), pd.concat(drivers, ignore_index=True)


DEFAULT_GRID = {
    "sigma_driver": [0.5, 0.8, 1.1, 1.5],
    "sigma_car": [0.5, 0.7, 1.0],
    "sigma_rw": [0.15, 0.3],
    "outlier_min": [3.0, 4.0],
}
DEFAULT_INFLATIONS = [0.4, 0.5, 0.6, 0.8, 1.0]


def tune(data: Data, n_sims: int = 2000, years: list[int] | None = None, progress=None,
         grid: dict | None = None, inflations: list[float] | None = None) -> tuple[Hyper, pd.DataFrame]:
    """Grid search over rating priors and uncertainty inflation, minimising mean WDC log-loss
    on the training years (holdout years are never looked at)."""
    years = years or TRAIN_YEARS
    grid = grid or DEFAULT_GRID
    inflations = inflations or DEFAULT_INFLATIONS
    rows = []
    combos = list(itertools.product(*grid.values()))
    for i, values in enumerate(combos):
        rc = replace(RatingConfig(), **dict(zip(grid, values)))
        ratings = compute_ratings(data, Hyper(rc))
        for infl in inflations:
            hyper = Hyper(rc, infl)
            lls, sps = [], []
            for y in years:
                r = evaluate_year(data, ratings, y, hyper, n_sims, seed=11).row
                lls.append(r["logloss"])
                sps.append(r["spearman"])
            rows.append({**dict(zip(grid, values)), "inflation": infl,
                         "logloss": float(np.mean(lls)), "spearman": float(np.mean(sps))})
        if progress:
            progress(f"{i + 1}/{len(combos)} rating configs; best so far "
                     f"{min(r['logloss'] for r in rows):.3f}")
    res = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    best = res.iloc[0]
    rc = replace(RatingConfig(), **{k: float(best[k]) for k in grid})
    return Hyper(rc, float(best["inflation"])), res


def tune_inseason(data: Data, hyper: Hyper, fractions=(0.33, 0.5, 0.75), mults=(0.0, 0.5, 1.0, 2.0, 4.0),
                  years: list[int] | None = None, n_sims: int = 2000, progress=None) -> tuple[float, pd.DataFrame]:
    """Pick the in-season development multiplier by WDC log-loss of predictions made a third,
    half and three quarters of the way through each training season."""
    years = years or TRAIN_YEARS
    ratings = compute_ratings(data, hyper)
    rows = []
    for mult in mults:
        h = replace(hyper, dev_mult=mult)
        lls = []
        for y in years:
            n = (data.calendar(y) or (data.n_rounds[y], []))[0]
            for f in fractions:
                try:
                    r = evaluate_year(data, ratings, y, h, n_sims, seed=13, as_of_round=max(1, round(f * n)))
                except ValueError:  # champion absent from that round's entry list
                    continue
                lls.append(r.row["logloss"])
        rows.append({"dev_mult": mult, "logloss": float(np.mean(lls))})
        if progress:
            progress(f"dev_mult {mult}: {rows[-1]['logloss']:.3f}")
    res = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    return float(res.iloc[0]["dev_mult"]), res


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics for train / holdout / regulation year-2 seasons."""
    groups = {
        "train 1997-2019": df["season"].isin(TRAIN_YEARS),
        "holdout 2020-2026": df["season"].isin(HOLDOUT_YEARS),
        "reg year 2 seasons": df["type"] == "reg_year_2",
        "all": df["season"] > 0,
    }
    out = []
    for name, mask in groups.items():
        sub = df[mask]
        if not len(sub):
            continue
        out.append({
            "group": name, "seasons": len(sub),
            "logloss_model": sub["logloss"].mean(), "logloss_points_share": sub["logloss_share"].mean(),
            "logloss_top2_teams": sub["logloss_top2"].mean(), "brier": sub["brier"].mean(),
            "spearman": sub["spearman"].mean(), "champion_top_pick": (sub["rank_champion"] == 1).mean(),
            "champion_in_top3": (sub["rank_champion"] <= 3).mean(),
        })
    return pd.DataFrame(out)
