"""Project driver and car ratings from season N to N+1.

Car:    c_{N+1} = rho_k * c_end_N + eps,  eps ~ N(0, tau_k^2),  k = transition type
Driver: s_{N+1} = s_N + age_effect(age) + eta,  eta ~ N(0, omega^2)

All parameters are estimated only from transitions that finished before the target season.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1wdc.config import transition_type

TYPES = ("stable", "reg_year_1", "reg_year_2")
AGE_BINS = [0, 23, 26, 29, 32, 35, 99]       # (lo, hi] edges for age at season start
AGE_LABELS = ["<=23", "24-26", "27-29", "30-32", "33-35", "36+"]


@dataclass
class ProjectionParams:
    rho: dict[str, float]
    tau: dict[str, float]
    n_pairs: dict[str, int]
    rho_raw: dict[str, float]
    new_team_mean: float
    new_team_sd: float
    age_effect: dict[str, float]
    omega: float
    rookie_mean: float
    rookie_sd: float
    slope_sd: float
    car_pairs: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def age_label(age: float) -> str:
    for lo, hi, lab in zip(AGE_BINS[:-1], AGE_BINS[1:], AGE_LABELS):
        if lo < age <= hi:
            return lab
    return AGE_LABELS[-1]


def age_at_season(dob: pd.Series | str, season: int) -> np.ndarray:
    dob = pd.to_datetime(pd.Series(dob), errors="coerce")
    ref = pd.Timestamp(f"{season}-03-01")
    return ((ref - dob).dt.days / 365.25).to_numpy()


def car_transition_pairs(cars: pd.DataFrame, upto: int) -> pd.DataFrame:
    """(lineage, N) pairs with c_end_N and c_level_{N+1}, for N+1 <= upto."""
    a = cars[["season", "lineage", "c_end"]]
    b = cars[["season", "lineage", "c_level"]].assign(season=lambda d: d["season"] - 1)
    pairs = a.merge(b, on=["season", "lineage"]).rename(columns={"c_level": "c_next"})
    pairs = pairs[pairs["season"] + 1 <= upto].copy()
    pairs["type"] = [transition_type(int(n) + 1) for n in pairs["season"]]
    return pairs


def estimate(
    drivers: pd.DataFrame,
    cars: pd.DataFrame,
    driver_info: pd.DataFrame,
    target: int,
    shrink_pairs: float = 20.0,
    min_starts: int = 5,
) -> ProjectionParams:
    upto = target - 1
    pairs = car_transition_pairs(cars, upto)

    def fit(df):
        sxx = float((df["c_end"] ** 2).sum())
        rho = float((df["c_end"] * df["c_next"]).sum() / sxx) if sxx > 0 else 0.7
        var = float(((df["c_next"] - rho * df["c_end"]) ** 2).mean()) if len(df) else 0.5
        return rho, var

    rho_pool, var_pool = fit(pairs) if len(pairs) else (0.7, 0.5)
    rho, tau, n_pairs, rho_raw = {}, {}, {}, {}
    for k in TYPES:
        sub = pairs[pairs["type"] == k]
        n = len(sub)
        r, v = fit(sub) if n >= 3 else (rho_pool, var_pool)
        rho_raw[k] = r
        n_pairs[k] = n
        rho[k] = (n * r + shrink_pairs * rho_pool) / (n + shrink_pairs)
        tau[k] = float(np.sqrt((n * v + shrink_pairs * var_pool) / (n + shrink_pairs)))

    # New teams: lineage present in N+1 but not in N
    hist = cars[cars["season"] <= upto]
    first_season = hist.groupby("lineage")["season"].min()
    start = hist["season"].min()
    newcomers = hist.merge(first_season.rename("first"), left_on="lineage", right_index=True)
    newcomers = newcomers[(newcomers["season"] == newcomers["first"]) & (newcomers["season"] > start)]
    nt_mean = float(newcomers["c_level"].mean()) if len(newcomers) >= 3 else -1.0
    nt_sd = float(newcomers["c_level"].std()) if len(newcomers) >= 3 else 0.7

    slope_sd = float(hist["c_slope"].std()) if len(hist) > 5 else 0.3

    # Driver age curve from consecutive-season rating changes (relative to the field)
    drv = drivers[(drivers["season"] <= upto) & (drivers["starts"] >= min_starts)]
    nxt = drv.assign(season=drv["season"] - 1)[["season", "driverId", "s"]].rename(columns={"s": "s_next"})
    dp = drv.merge(nxt, on=["season", "driverId"])
    dp = dp[dp["season"] + 1 <= upto]
    dob = driver_info.set_index("driverId")["dateOfBirth"]
    if len(dp):
        dp["delta"] = dp["s_next"] - dp["s"]
        dp["delta"] -= dp.groupby("season")["delta"].transform("mean")
        dp["age"] = [age_at_season(dob.get(d, None), int(n) + 1)[0] for d, n in zip(dp["driverId"], dp["season"])]
        dp["bin"] = [age_label(a) if np.isfinite(a) else "27-29" for a in dp["age"]]
        stats = dp.groupby("bin")["delta"].agg(["sum", "count"])
        age_effect = {lab: float(stats.loc[lab, "sum"] / (stats.loc[lab, "count"] + 10)) if lab in stats.index else 0.0
                      for lab in AGE_LABELS}
        resid = dp["delta"] - dp["bin"].map(age_effect)
        omega = float(resid.std())
    else:
        age_effect, omega = {lab: 0.0 for lab in AGE_LABELS}, 0.3

    # Rookies: centered rating in debut season (excluding the first season of data)
    all_drv = drivers[drivers["season"] <= upto]
    debut = all_drv.groupby("driverId")["season"].min()
    rk = all_drv.merge(debut.rename("debut"), left_on="driverId", right_index=True)
    rk = rk[(rk["season"] == rk["debut"]) & (rk["season"] > start) & (rk["starts"] >= min_starts)]
    rookie_mean = float(rk["s"].mean()) if len(rk) >= 3 else -0.2
    rookie_sd = float(rk["s"].std()) if len(rk) >= 3 else 0.5

    return ProjectionParams(rho, tau, n_pairs, rho_raw, nt_mean, nt_sd, age_effect, omega,
                            rookie_mean, rookie_sd, slope_sd, pairs)


@dataclass
class Projection:
    """Distribution of season parameters: [s per driver | c_level per team | c_slope per team]."""
    drivers: list[str]
    teams: list[str]
    driver_team_idx: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    source: dict[str, str]              # driverId -> "rated" | "returning" | "rookie" | "limited"
    # In-season only: car development still to come. Each team draws a drift eta ~ N(0, dev_sd^2)
    # per simulated season, added as (u - u_now) * eta to rounds after ``u_now``.
    u_now: float = -0.5
    dev_sd: float = 0.0

    @property
    def n_drivers(self) -> int:
        return len(self.drivers)

    def car_mean(self) -> np.ndarray:
        n = self.n_drivers
        return self.mean[n:n + len(self.teams)]

    def car_sd(self) -> np.ndarray:
        n = self.n_drivers
        return np.sqrt(np.diag(self.cov)[n:n + len(self.teams)])

    def driver_mean(self) -> np.ndarray:
        return self.mean[: self.n_drivers]

    def driver_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov)[: self.n_drivers])


def project(
    params: ProjectionParams,
    drivers: pd.DataFrame,
    cars: pd.DataFrame,
    driver_info: pd.DataFrame,
    grid: dict[str, list[str]],
    target: int,
    inflation: float = 1.0,
    carryover: float | None = None,
    lookback: int = 3,
    min_starts: int = 5,
) -> Projection:
    kind = transition_type(target)
    rho = params.rho[kind] if carryover is None else carryover
    tau = params.tau[kind]
    teams = list(grid)
    drv_list = [d for t in teams for d in grid[t]]
    team_idx = np.array([teams.index(t) for t in teams for _ in grid[t]])
    dob = driver_info.set_index("driverId")["dateOfBirth"]

    prev = cars[cars["season"] == target - 1].set_index("lineage")
    car_mean, car_var = [], []
    for t in teams:
        if t in prev.index:
            car_mean.append(rho * prev.loc[t, "c_end"])
            car_var.append(tau**2)
        else:
            car_mean.append(params.new_team_mean)
            car_var.append(params.new_team_sd**2 + tau**2)
    car_mean = np.array(car_mean) - np.mean(car_mean)

    s_mean, s_var, source = [], [], {}
    hist = drivers[(drivers["season"] < target) & (drivers["season"] >= target - lookback)]
    for d in drv_list:
        rows = hist[hist["driverId"] == d].sort_values("season")
        age = age_at_season(dob.get(d, None), target)[0]
        effect = params.age_effect[age_label(age)] if np.isfinite(age) else 0.0
        if len(rows):
            last = rows.iloc[-1]
            gap = target - 1 - int(last["season"])
            mean, var = float(last["s"]) + effect, params.omega**2 * (1 + gap)
            starts = float(rows["starts"].sum())
            if starts < min_starts:  # a handful of stand-in races: lean on the rookie prior
                w = starts / min_starts
                mean = w * mean + (1 - w) * params.rookie_mean
                var = w * var + (1 - w) * params.rookie_sd**2
            s_mean.append(mean)
            s_var.append(var)
            source[d] = ("rated" if gap == 0 else "returning") if starts >= min_starts else "limited"
        else:
            s_mean.append(params.rookie_mean)
            s_var.append(params.rookie_sd**2)
            source[d] = "rookie"
    s_mean = np.array(s_mean) - np.mean(s_mean)

    n_t = len(teams)
    mean = np.concatenate([s_mean, car_mean, np.zeros(n_t)])
    var = np.concatenate([np.array(s_var) * inflation**2, np.array(car_var) * inflation**2,
                          np.full(n_t, params.slope_sd**2)])
    return Projection(drv_list, teams, team_idx, mean, np.diag(var), source)
