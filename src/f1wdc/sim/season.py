"""Vectorized Monte Carlo season simulator.

Per simulated season: draw driver skills and car levels/slopes once (season-level uncertainty
that is shared by every race), then for each event draw DNFs and sample the finishing order of
the remaining cars from the Plackett-Luce model (Gumbel-max), and award points. Championship
ties are broken by countback (wins, then P2s, then P3s).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1wdc.model.projection import Projection


@dataclass
class Event:
    round: int
    u: float          # position in season, [-0.5, 0.5]
    kind: str         # "race" | "sprint"


@dataclass
class SimResult:
    drivers: list[str]
    teams: list[str]
    driver_team_idx: np.ndarray
    points: np.ndarray            # (n_sims, n_drivers) final points
    final_pos: np.ndarray         # (n_sims, n_drivers) 0 = champion
    team_points: np.ndarray       # (n_sims, n_teams)
    wcc_pos: np.ndarray           # (n_sims, n_teams)

    @property
    def n_sims(self) -> int:
        return self.points.shape[0]

    def p_wdc(self) -> np.ndarray:
        return (self.final_pos == 0).mean(axis=0)

    def position_matrix(self) -> np.ndarray:
        """(n_drivers, n_drivers): P(driver finishes championship in position j)."""
        n = len(self.drivers)
        return np.stack([(self.final_pos == j).mean(axis=0) for j in range(n)], axis=1)

    def driver_table(self) -> pd.DataFrame:
        q = np.percentile(self.points, [10, 50, 90], axis=0)
        df = pd.DataFrame({
            "driverId": self.drivers,
            "team": [self.teams[i] for i in self.driver_team_idx],
            "p_wdc": self.p_wdc(),
            "p_top3": (self.final_pos < 3).mean(axis=0),
            "exp_points": self.points.mean(axis=0),
            "p10": q[0], "p50": q[1], "p90": q[2],
            "exp_position": self.final_pos.mean(axis=0) + 1,
        })
        return df.sort_values(["p_wdc", "exp_points"], ascending=False).reset_index(drop=True)

    def team_table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "team": self.teams,
            "p_wcc": (self.wcc_pos == 0).mean(axis=0),
            "exp_points": self.team_points.mean(axis=0),
        })
        return df.sort_values(["p_wcc", "exp_points"], ascending=False).reset_index(drop=True)


def _sample_params(proj: Projection, n: int, rng: np.random.Generator) -> np.ndarray:
    cov = proj.cov
    if np.count_nonzero(cov - np.diag(np.diag(cov))) == 0:
        return proj.mean + rng.standard_normal((n, len(proj.mean))) * np.sqrt(np.diag(cov))
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(cov)))
    return proj.mean + rng.standard_normal((n, len(proj.mean))) @ L.T


def simulate(
    proj: Projection,
    p_dnf: np.ndarray,
    events: list[Event],
    race_points: list[int],
    sprint_points: list[int],
    kappa_sprint: float = 1.0,
    n_sims: int = 20_000,
    seed: int | None = None,
    chunk: int = 5_000,
    points_so_far: np.ndarray | None = None,
    countback_so_far: np.ndarray | None = None,
) -> SimResult:
    """Simulate ``events`` ``n_sims`` times.

    ``points_so_far`` (n_drivers,) and ``countback_so_far`` (n_drivers, 3: wins, P2, P3)
    hold results already in the books when simulating the rest of a season.
    """
    rng = np.random.default_rng(seed)
    nd, nt = proj.n_drivers, len(proj.teams)
    tix = proj.driver_team_idx
    p_dnf = np.asarray(p_dnf, float)
    p_dnf_sprint = 1 - (1 - p_dnf) ** (1 / 3)   # sprints are ~1/3 race distance
    race_tab = np.zeros(nd)
    race_tab[: min(nd, len(race_points))] = race_points[:nd]
    sprint_tab = np.zeros(nd)
    sprint_tab[: min(nd, len(sprint_points))] = sprint_points[:nd]
    base_pts = np.zeros(nd) if points_so_far is None else np.asarray(points_so_far, float)
    base_cb = np.zeros((nd, 3)) if countback_so_far is None else np.asarray(countback_so_far, float)

    all_pts, all_pos, all_tpts, all_tpos = [], [], [], []
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        x = _sample_params(proj, m, rng)
        s = x[:, :nd]
        cl = x[:, nd:nd + nt][:, tix]
        cb = x[:, nd + nt:nd + 2 * nt][:, tix]
        drift = (rng.standard_normal((m, nt)) * proj.dev_sd)[:, tix] if proj.dev_sd > 0 else None
        pts = np.tile(base_pts, (m, 1))
        cnt = np.tile(base_cb, (m, 1, 1))
        rows = np.arange(m)[:, None]
        for ev in events:
            sprint = ev.kind == "sprint"
            theta = s + cl + ev.u * cb
            if drift is not None and ev.u > proj.u_now:
                theta = theta + (ev.u - proj.u_now) * drift
            if sprint:
                theta = kappa_sprint * theta
            dnf = rng.random((m, nd)) < (p_dnf_sprint if sprint else p_dnf)
            score = theta + rng.gumbel(size=(m, nd))
            score[dnf] = -np.inf
            order = np.argsort(-score, axis=1)
            ranks = np.empty_like(order)
            ranks[rows, order] = np.arange(nd)
            gained = (sprint_tab if sprint else race_tab)[ranks]
            gained[dnf] = 0.0
            pts += gained
            if not sprint:
                for k in range(3):
                    cnt[:, :, k] += (ranks == k) & ~dnf
        # Countback composite: points dominate, then wins, P2s, P3s, then a random tiebreak.
        key = pts * 1e6 + cnt[:, :, 0] * 1e4 + cnt[:, :, 1] * 1e2 + cnt[:, :, 2] + rng.random((m, nd)) * 1e-3
        order = np.argsort(-key, axis=1)
        pos = np.empty_like(order)
        pos[rows, order] = np.arange(nd)
        tpts = np.zeros((m, nt))
        np.add.at(tpts, (slice(None), tix), pts)
        tkey = tpts + rng.random((m, nt)) * 1e-3
        torder = np.argsort(-tkey, axis=1)
        tpos = np.empty_like(torder)
        tpos[np.arange(m)[:, None], torder] = np.arange(nt)
        all_pts.append(pts.astype(np.float32))
        all_pos.append(pos.astype(np.int16))
        all_tpts.append(tpts.astype(np.float32))
        all_tpos.append(tpos.astype(np.int16))
        done += m

    return SimResult(proj.drivers, proj.teams, tix, np.concatenate(all_pts), np.concatenate(all_pos),
                     np.concatenate(all_tpts), np.concatenate(all_tpos))
