"""End-to-end prediction of one season, shared by ``predict`` and the backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from f1wdc.config import transition_type
from f1wdc.data.build import load_tables
from f1wdc.model import projection as proj_mod
from f1wdc.model.ratings import RatingConfig, fit_window, ratings_by_season, round_fraction, rounds_per_season
from f1wdc.model.reliability import dnf_probabilities
from f1wdc.sim.season import Event, SimResult, simulate

DEFAULT_KAPPA_SPRINT = 1.6


@dataclass(frozen=True)
class Hyper:
    rating: RatingConfig = field(default_factory=RatingConfig)
    inflation: float = 1.0
    dev_mult: float = 1.0      # in-season: scale of car development still to come (x historical slope sd)
    inflation_new_rules: float | None = None   # first season of new rules; None = same as inflation

    def inflation_for(self, kind: str) -> float:
        if kind == "reg_year_1" and self.inflation_new_rules is not None:
            return self.inflation_new_rules
        return self.inflation


@dataclass
class Data:
    entries: pd.DataFrame
    drivers: pd.DataFrame
    schedule: pd.DataFrame
    n_rounds: dict[int, int]

    @classmethod
    def load(cls, data_dir: Path | str = "data") -> "Data":
        e, d, s = load_tables(data_dir)
        return cls(e, d, s, rounds_per_season(s, e))

    def completed_rounds(self, season: int) -> int:
        r = self.entries[(self.entries["season"] == season) & (self.entries["session"] == "race")]["round"]
        return int(r.max()) if len(r) else 0

    def calendar(self, season: int) -> tuple[int, list[int]] | None:
        sch = self.schedule[self.schedule["season"] == season]
        if not len(sch):
            return None
        return int(sch["round"].max()), sorted(int(r) for r in sch[sch["has_sprint"]]["round"])

    def grid_at(self, season: int, rnd: int | None = None) -> dict[str, list[str]]:
        """Race entry list (lineage -> drivers) at round ``rnd`` (default: first round)."""
        r = self.entries[(self.entries["season"] == season) & (self.entries["session"] == "race")]
        rnd = int(r["round"].min()) if rnd is None else rnd
        r = r[r["round"] == rnd]
        grid: dict[str, list[str]] = {}
        for lin, d in zip(r["lineage"], r["driverId"]):
            grid.setdefault(lin, []).append(d)
        return grid


@dataclass
class Prediction:
    target: int
    as_of_round: int
    kind: str
    params: proj_mod.ProjectionParams
    preseason: proj_mod.Projection
    projection: proj_mod.Projection
    p_dnf: np.ndarray
    kappa_sprint: float
    events: list[Event]
    sim: SimResult
    carryover: float


def season_events(n_rounds: int, sprint_rounds: list[int], after_round: int = 0) -> list[Event]:
    events = []
    for r in range(after_round + 1, n_rounds + 1):
        u = float(round_fraction(r, n_rounds))
        if r in sprint_rounds:
            events.append(Event(r, u, "sprint"))
        events.append(Event(r, u, "race"))
    return events


def sprints_run_after(entries: pd.DataFrame, season: int, as_of: int) -> set[int]:
    """Rounds after ``as_of`` whose sprint is already in the data (half-finished weekend)."""
    x = entries[(entries["season"] == season) & (entries["session"] == "sprint") & (entries["round"] > as_of)]
    return set(int(r) for r in x["round"].unique())


def points_so_far(entries: pd.DataFrame, season: int, as_of: int, drivers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = entries[(entries["season"] == season) & entries["session"].isin(["race", "sprint"])]
    x = x[(x["round"] <= as_of) | ((x["session"] == "sprint") & x["round"].isin(sprints_run_after(entries, season, as_of)))]
    pts = x.groupby("driverId")["points"].sum()
    race = x[(x["session"] == "race") & x["classified"]]
    cb = np.zeros((len(drivers), 3))
    for k in range(3):
        c = race[race["position"] == k + 1]["driverId"].value_counts()
        cb[:, k] = [c.get(d, 0) for d in drivers]
    return np.array([pts.get(d, 0.0) for d in drivers]), cb


def inseason_projection(data: Data, target: int, as_of: int, cfg: RatingConfig,
                        pre: proj_mod.Projection, inflation: float, dev_sd: float) -> proj_mod.Projection:
    """Update the pre-season projection with races already run in ``target``.

    Refits the window ending in ``target`` (up to ``as_of``) with the car levels' prior
    centred on the pre-season projection, then takes a Laplace approximation (inverse
    Hessian) for the covariance of the grid's driver, car and slope parameters.
    """
    car_mean, car_sd = pre.car_mean(), pre.car_sd() / max(inflation, 1e-9)
    car_prior = {(target, t): (car_mean[i], car_sd[i]) for i, t in enumerate(pre.teams)}
    wf = fit_window(data.entries, data.n_rounds, target, cfg, as_of_round=as_of, car_prior=car_prior)
    m, di, ti = wf.model, wf.driver_index(), wf.team_index()
    nd, nt = pre.n_drivers, len(pre.teams)
    idx, pos = [], []
    for k, d in enumerate(pre.drivers):
        if (d, target) in di:
            idx.append(m.sl_s.start + di[(d, target)])
            pos.append(k)
    for k, t in enumerate(pre.teams):
        if (target, t) in ti:
            idx += [m.sl_cl.start + ti[(target, t)]]
            pos += [nd + k]
    for k, t in enumerate(pre.teams):
        if (target, t) in ti:
            idx += [m.sl_cb.start + ti[(target, t)]]
            pos += [nd + nt + k]
    mean, cov = pre.mean.copy(), pre.cov.copy()
    if idx:
        H = m.hessian(wf.x, np.array(idx))
        cov_sub = np.linalg.inv(H + 1e-9 * np.eye(len(idx)))
        mean[pos] = wf.x[idx]
        cov[pos, :] = 0.0
        cov[:, pos] = 0.0
        cov[np.ix_(pos, pos)] = cov_sub * inflation**2
    mean[:nd] -= mean[:nd].mean()
    mean[nd:nd + nt] -= mean[nd:nd + nt].mean()
    u_now = float(round_fraction(as_of, data.n_rounds.get(target, as_of)))
    return proj_mod.Projection(pre.drivers, pre.teams, pre.driver_team_idx, mean, cov, pre.source,
                               u_now=u_now, dev_sd=dev_sd * inflation)


def predict(
    data: Data,
    ratings: tuple[pd.DataFrame, pd.DataFrame],
    target: int,
    grid: dict[str, list[str]],
    n_rounds: int,
    sprint_rounds: list[int],
    race_points: list[int],
    sprint_points: list[int],
    hyper: Hyper,
    n_sims: int = 20_000,
    seed: int | None = 2027,
    carryover: float | None = None,
    as_of_round: int | None = None,
    dob_overrides: dict[str, str] | None = None,
) -> Prediction:
    drv_tab, car_tab = ratings
    info = data.drivers.copy()
    for d, dob in (dob_overrides or {}).items():
        if d in set(info["driverId"]):
            info.loc[info["driverId"] == d, "dateOfBirth"] = dob
        else:
            info = pd.concat([info, pd.DataFrame([{"driverId": d, "dateOfBirth": dob}])], ignore_index=True)

    params = proj_mod.estimate(drv_tab, car_tab, info, target)
    kind = transition_type(target)
    pre = proj_mod.project(params, drv_tab, car_tab, info, grid, target, hyper.inflation_for(kind), carryover)
    as_of = data.completed_rounds(target) if as_of_round is None else as_of_round
    proj = (inseason_projection(data, target, as_of, hyper.rating, pre, hyper.inflation_for(kind),
                                params.slope_sd * hyper.dev_mult) if as_of > 0 else pre)

    driver_team = {d: t for t in grid for d in grid[t]}
    p_map = dnf_probabilities(data.entries, target, driver_team, as_of_round=as_of)
    p_dnf = np.array([p_map[d] for d in proj.drivers])

    prev = car_tab[car_tab["season"] == target - 1]
    kappa_sprint = float(prev["kappa_sprint"].iloc[0]) if len(prev) and target - 1 >= 2021 else DEFAULT_KAPPA_SPRINT
    done_sprints = sprints_run_after(data.entries, target, as_of)
    events = [e for e in season_events(n_rounds, sprint_rounds, after_round=as_of)
              if not (e.kind == "sprint" and e.round in done_sprints)]
    pts, cb = points_so_far(data.entries, target, as_of, proj.drivers)
    sim = simulate(proj, p_dnf, events, race_points, sprint_points, kappa_sprint=kappa_sprint,
                   n_sims=n_sims, seed=seed, points_so_far=pts, countback_so_far=cb)
    rho = params.rho[kind] if carryover is None else carryover
    return Prediction(target, as_of, kind, params, pre, proj, p_dnf, kappa_sprint, events, sim, rho)


def compute_ratings(data: Data, hyper: Hyper, cache_dir: Path | None = Path("out/cache"), progress=None):
    return ratings_by_season(data.entries, data.schedule, hyper.rating, cache_dir=cache_dir, progress=progress)
