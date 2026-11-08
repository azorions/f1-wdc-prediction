"""Rolling Plackett-Luce fits producing per-season driver and car ratings.

For season N we fit the window N-2..N (older seasons down-weighted). Each driver gets a rating
per season, linked across seasons by a random-walk prior (so a fast-improving young driver is
not held back by an old rookie year); team transfers separate driver from car. Each team-season
gets its own car level and in-season slope. The fit for N only sees data up to N, so the
``ratings_by_season`` tables can be used directly for leak-free backtests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from f1wdc.model.plackett_luce import QUALI, RACE, SPRINT, PlackettLuce, make_design

SESSION_TYPE = {"race": RACE, "quali": QUALI, "sprint": SPRINT}


@dataclass(frozen=True)
class RatingConfig:
    sigma_driver: float = 0.5
    sigma_car: float = 1.0
    sigma_slope: float = 0.5
    sigma_rw: float = 0.25
    window: int = 3
    season_decay: float = 0.5
    # Drop freak results (mechanical failure in qualifying, damage, back-of-grid starts):
    # positions worse than the driver's own season/session median by more than
    # max(outlier_min, outlier_k * robust sd). Plackett-Luce punishes such results as losses
    # to the whole field, which badly distorts driver ratings. 0 disables the filter.
    outlier_min: float = 4.0
    outlier_k: float = 2.0

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def rounds_per_season(schedule: pd.DataFrame, entries: pd.DataFrame) -> dict[int, int]:
    n = schedule.groupby("season")["round"].max().to_dict()
    for season, r in entries.groupby("season")["round"].max().items():
        n[season] = max(int(n.get(season, 0)), int(r))
    return {int(k): int(v) for k, v in n.items()}


def round_fraction(rnd: np.ndarray | int, n_rounds: int) -> np.ndarray:
    return (np.asarray(rnd, float) - 1) / max(n_rounds - 1, 1) - 0.5


@dataclass
class WindowFit:
    model: PlackettLuce
    x: np.ndarray
    driver_keys: list[tuple[str, int]]       # (driverId, season)
    team_keys: list[tuple[int, str]]         # (season, lineage)
    race_starts: pd.DataFrame                # season, driverId, starts
    success: bool

    @property
    def kappa(self) -> np.ndarray:
        return self.model.kappa(self.x)

    def driver_index(self) -> dict[tuple[str, int], int]:
        return {k: i for i, k in enumerate(self.driver_keys)}

    def team_index(self) -> dict[tuple[int, str], int]:
        return {k: i for i, k in enumerate(self.team_keys)}

    def season_tables(self, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Centered driver and car ratings for ``season`` from this window."""
        m = self.model
        s = self.x[m.sl_s]
        cl, cb = self.x[m.sl_cl], self.x[m.sl_cb]
        starts = self.race_starts[self.race_starts["season"] == season]
        di = self.driver_index()
        starts = starts[[(d, season) in di for d in starts["driverId"]]]
        drv = starts.assign(s=[s[di[(d, season)]] for d in starts["driverId"]])
        if len(drv):
            drv["s"] -= np.average(drv["s"], weights=drv["starts"])
        rows = [
            {"season": season, "lineage": lin, "c_level": cl[i], "c_slope": cb[i], "c_end": cl[i] + 0.5 * cb[i]}
            for i, (se, lin) in enumerate(self.team_keys) if se == season
        ]
        teams = pd.DataFrame(rows)
        if len(teams):
            mean = teams["c_level"].mean()
            teams["c_level"] -= mean
            teams["c_end"] -= mean
        return drv.reset_index(drop=True), teams


def drop_outliers(df: pd.DataFrame, cfg: RatingConfig) -> pd.DataFrame:
    """Remove results far below a driver's own typical position in that season and session."""
    if cfg.outlier_min <= 0:
        return df.copy()
    g = df.groupby(["season", "driverId", "session"])["position"]
    med = g.transform("median")
    mad = (df["position"] - med).abs().groupby([df["season"], df["driverId"], df["session"]]).transform("median")
    limit = np.maximum(cfg.outlier_min, cfg.outlier_k * 1.4826 * mad)
    return df[(df["position"] - med) <= limit].copy()


def fit_window(
    entries: pd.DataFrame,
    n_rounds: dict[int, int],
    end_season: int,
    cfg: RatingConfig,
    as_of_round: int | None = None,
    car_prior: dict[tuple[int, str], tuple[float, float]] | None = None,
    x0: np.ndarray | None = None,
) -> WindowFit:
    """Fit seasons ``end_season - window + 1 .. end_season``.

    ``as_of_round`` truncates ``end_season`` to rounds <= as_of_round.
    ``car_prior`` maps (season, lineage) -> (mean, sd) for the car level, replacing the
    default N(0, sigma_car) prior (used for in-season updating of the target season).
    """
    first = end_season - cfg.window + 1
    df = entries[(entries["season"] >= first) & (entries["season"] <= end_season)]
    if as_of_round is not None:
        df = df[(df["season"] < end_season) | (df["round"] <= as_of_round)]
    df = drop_outliers(df[df["classified"]], cfg)

    driver_keys = sorted({(d, int(s)) for d, s in zip(df["driverId"], df["season"])})
    team_keys = sorted({(int(s), l) for s, l in zip(df["season"], df["lineage"])})
    di = {k: i for i, k in enumerate(driver_keys)}
    ti = {k: i for i, k in enumerate(team_keys)}

    df = df.sort_values(["season", "round", "session", "position"]).reset_index(drop=True)
    df["d_idx"] = [di[(d, int(s))] for d, s in zip(df["driverId"], df["season"])]
    df["t_idx"] = [ti[(int(s), l)] for s, l in zip(df["season"], df["lineage"])]
    df["u"] = [round_fraction(r, n_rounds[int(s)]) for s, r in zip(df["season"], df["round"])]

    orders, types, weights = [], [], []
    for (season, _rnd, session), grp in df.groupby(["season", "round", "session"], sort=False):
        if len(grp) < 2:
            continue
        orders.append(grp.index.to_numpy())
        types.append(SESSION_TYPE[session])
        weights.append(cfg.season_decay ** (end_season - season))

    # Random-walk links between a driver's seasons: penalty (s_b - s_a)^2 / (2 * gap * sigma_rw^2)
    pi, pj, pw = [], [], []
    prev: dict[str, tuple[int, int]] = {}
    for (d, season), idx in sorted(di.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if d in prev:
            p_season, p_idx = prev[d]
            pi.append(p_idx)
            pj.append(idx)
            pw.append(1.0 / ((season - p_season) * cfg.sigma_rw**2))
        prev[d] = (season, idx)

    design = make_design(orders, types, df["d_idx"], df["t_idx"], df["u"], len(driver_keys), len(team_keys), weights)
    model = PlackettLuce(design, cfg.sigma_driver, cfg.sigma_car, cfg.sigma_slope,
                         pairs=(np.array(pi, np.int64), np.array(pj, np.int64), np.array(pw)))
    if car_prior:
        for key, (mean, sd) in car_prior.items():
            if key in ti:
                j = model.sl_cl.start + ti[key]
                model.prior_mean[j] = mean
                model.prior_prec[j] = 1 / sd**2
    fit = model.fit(x0=x0)

    race = entries[(entries["season"] >= first) & (entries["season"] <= end_season) & (entries["session"] == "race")]
    if as_of_round is not None:
        race = race[(race["season"] < end_season) | (race["round"] <= as_of_round)]
    race = race[race["driverId"].isin({d for d, _ in driver_keys})]
    starts = race.groupby(["season", "driverId"]).size().rename("starts").reset_index()
    return WindowFit(model, fit.x, driver_keys, team_keys, starts, fit.success)


def ratings_by_season(
    entries: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: RatingConfig,
    seasons: list[int] | None = None,
    cache_dir: Path | None = None,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driver table (season, driverId, starts, s) and car table
    (season, lineage, c_level, c_slope, c_end, kappa_q, kappa_sprint) for each season,
    each taken from the window that ends in that season."""
    seasons = seasons or sorted(entries["season"].unique())
    fingerprint = f"{len(entries)}-{entries['season'].max()}-{entries[entries['season'] == entries['season'].max()]['round'].max()}"
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(f"{cfg.key()}|{fingerprint}|{seasons[0]}-{seasons[-1]}".encode()).hexdigest()[:12]
        cache = (cache_dir / f"ratings_drv_{tag}.csv", cache_dir / f"ratings_car_{tag}.csv")
        if cache[0].exists() and cache[1].exists():
            return pd.read_csv(cache[0]), pd.read_csv(cache[1])

    n_rounds = rounds_per_season(pd.DataFrame(columns=["season", "round"]) if schedule is None else schedule, entries)
    drv_parts, car_parts = [], []
    for season in seasons:
        if progress:
            progress(season)
        wf = fit_window(entries, n_rounds, int(season), cfg)
        drv, car = wf.season_tables(int(season))
        k = wf.kappa
        car["kappa_q"], car["kappa_sprint"] = k[QUALI], k[SPRINT]
        drv_parts.append(drv)
        car_parts.append(car)
    drivers = pd.concat(drv_parts, ignore_index=True)
    cars = pd.concat(car_parts, ignore_index=True)
    if cache:
        drivers.to_csv(cache[0], index=False)
        cars.to_csv(cache[1], index=False)
    return drivers, cars
