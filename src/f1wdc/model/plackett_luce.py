"""Plackett-Luce (rank-ordered logit) model with driver + team-season car effects.

Strength of entry i (one driver in one session):

    theta_i = s[driver_i] + c_level[team_i] + u_i * c_slope[team_i]

where ``u_i`` in [-0.5, 0.5] is the round's position within its season. An event (a race,
sprint or qualifying session) is an ordered list of entries, winner first, and has

    P(order) = prod_k exp(eta_(k)) / sum_{j >= k} exp(eta_(j)),   eta = kappa[type] * theta

kappa[race] = 1 is the reference scale; qualifying and sprint sharpness factors are learned
(as log-kappa). Gaussian priors on every parameter make it a penalised (MAP) fit; the prior
means/precisions can be set per parameter, which in-season updating uses.

Parameter vector layout: [ s (D) | c_level (T) | c_slope (T) | log_kappa (n_types - 1) ].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

RACE, QUALI, SPRINT = 0, 1, 2
N_TYPES = 3


@dataclass
class Design:
    entry_driver: np.ndarray   # (N,) int, driver index per entry
    entry_team: np.ndarray     # (N,) int, team-season index per entry
    entry_u: np.ndarray        # (N,) float, round position in season, [-0.5, 0.5]
    events: np.ndarray         # (E, L) int, entry indices in finishing order, -1 padding
    event_type: np.ndarray     # (E,) int in {RACE, QUALI, SPRINT}
    event_weight: np.ndarray   # (E,) float
    n_drivers: int
    n_teams: int

    @property
    def n_params(self) -> int:
        return self.n_drivers + 2 * self.n_teams + (N_TYPES - 1)


def make_design(
    event_orders: list[np.ndarray],
    event_types: list[int],
    entry_driver: np.ndarray,
    entry_team: np.ndarray,
    entry_u: np.ndarray,
    n_drivers: int,
    n_teams: int,
    event_weights: list[float] | None = None,
) -> Design:
    """Build a padded :class:`Design`. ``event_orders[e]`` lists entry indices, winner first."""
    width = max(len(o) for o in event_orders)
    events = np.full((len(event_orders), width), -1, dtype=np.int64)
    for e, order in enumerate(event_orders):
        events[e, : len(order)] = order
    weights = np.ones(len(event_orders)) if event_weights is None else np.asarray(event_weights, float)
    return Design(
        entry_driver=np.asarray(entry_driver, np.int64),
        entry_team=np.asarray(entry_team, np.int64),
        entry_u=np.asarray(entry_u, float),
        events=events,
        event_type=np.asarray(event_types, np.int64),
        event_weight=weights,
        n_drivers=n_drivers,
        n_teams=n_teams,
    )


@dataclass
class FitResult:
    x: np.ndarray
    nll: float
    success: bool
    message: str


class PlackettLuce:
    def __init__(
        self,
        design: Design,
        sigma_driver: float = 0.5,
        sigma_car: float = 1.0,
        sigma_slope: float = 0.5,
        sigma_log_kappa: float = 1.0,
        prior_mean: np.ndarray | None = None,
        prior_prec: np.ndarray | None = None,
        pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ):
        """``pairs`` = (i, j, w): extra penalty 0.5 * w * (x_i - x_j)^2 per pair, used as a
        random-walk prior linking a driver's ratings in consecutive seasons."""
        self.d = design
        if pairs is None:
            pairs = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
        self.pair_i, self.pair_j, self.pair_w = (np.asarray(pairs[0], np.int64),
                                                 np.asarray(pairs[1], np.int64), np.asarray(pairs[2], float))
        D, T = design.n_drivers, design.n_teams
        self.sl_s = slice(0, D)
        self.sl_cl = slice(D, D + T)
        self.sl_cb = slice(D + T, D + 2 * T)
        self.sl_k = slice(D + 2 * T, D + 2 * T + N_TYPES - 1)
        n = design.n_params
        if prior_prec is None:
            prior_prec = np.empty(n)
            prior_prec[self.sl_s] = 1 / sigma_driver**2
            prior_prec[self.sl_cl] = 1 / sigma_car**2
            prior_prec[self.sl_cb] = 1 / sigma_slope**2
            prior_prec[self.sl_k] = 1 / sigma_log_kappa**2
        self.prior_prec = np.asarray(prior_prec, float)
        self.prior_mean = np.zeros(n) if prior_mean is None else np.asarray(prior_mean, float)
        self._mask = design.events >= 0
        self._idx = np.where(self._mask, design.events, 0)

    # ----------------------------------------------------------- unpacking
    def kappa(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([[1.0], np.exp(x[self.sl_k])])

    def theta(self, x: np.ndarray) -> np.ndarray:
        d = self.d
        s, cl, cb = x[self.sl_s], x[self.sl_cl], x[self.sl_cb]
        return s[d.entry_driver] + cl[d.entry_team] + d.entry_u * cb[d.entry_team]

    # ----------------------------------------------------------- objective
    def objective(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Negative log posterior (up to a constant) and its gradient."""
        d, mask, idx = self.d, self._mask, self._idx
        th = self.theta(x)
        k_e = self.kappa(x)[d.event_type][:, None]
        th_m = th[idx]
        eta = np.where(mask, k_e * th_m, -np.inf)
        m = eta.max(axis=1, keepdims=True)
        z = np.exp(eta - m)                                   # padding -> 0
        S = np.cumsum(z[:, ::-1], axis=1)[:, ::-1]            # S_k = sum_{j>=k} z_j
        S_safe = np.where(mask, S, 1.0)
        w = d.event_weight[:, None]
        ll = float((w * np.where(mask, eta - m - np.log(S_safe), 0.0)).sum())

        cum_inv = np.cumsum(np.where(mask, 1.0 / S_safe, 0.0), axis=1)
        g_eta = np.where(mask, 1.0 - z * cum_inv, 0.0) * w   # d ll / d eta

        g_th = np.zeros(th.shape[0])
        g_th[idx[mask]] = (g_eta * k_e)[mask]
        D, T = d.n_drivers, d.n_teams
        g = np.empty_like(x)
        g[self.sl_s] = np.bincount(d.entry_driver, weights=g_th, minlength=D)
        g[self.sl_cl] = np.bincount(d.entry_team, weights=g_th, minlength=T)
        g[self.sl_cb] = np.bincount(d.entry_team, weights=g_th * d.entry_u, minlength=T)
        per_event = np.where(mask, g_eta * k_e * th_m, 0.0).sum(axis=1)
        g[self.sl_k] = np.bincount(d.event_type, weights=per_event, minlength=N_TYPES)[1:]

        diff = x - self.prior_mean
        f = -ll + 0.5 * float((self.prior_prec * diff**2).sum())
        grad = -g + self.prior_prec * diff
        if len(self.pair_w):
            gap = x[self.pair_i] - x[self.pair_j]
            f += 0.5 * float((self.pair_w * gap**2).sum())
            np.add.at(grad, self.pair_i, self.pair_w * gap)
            np.add.at(grad, self.pair_j, -self.pair_w * gap)
        return f, grad

    # ----------------------------------------------------------- fitting
    def fit(self, x0: np.ndarray | None = None, maxiter: int = 5000) -> FitResult:
        x0 = self.prior_mean.copy() if x0 is None else x0
        # Bound log-kappa: with a single sprint in a window the sharpness is barely identified and
        # an unbounded line search can overflow exp().
        bounds = [(None, None)] * len(x0)
        for j in range(self.sl_k.start, self.sl_k.stop):
            bounds[j] = (-3.0, 3.0)
        res = minimize(self.objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "gtol": 1e-6, "maxcor": 30})
        return FitResult(x=res.x, nll=float(res.fun), success=bool(res.success), message=str(res.message))

    def hessian(self, x: np.ndarray, idx: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """Numerical Hessian of the objective restricted to parameters ``idx``
        (central differences of the analytic gradient)."""
        idx = np.asarray(idx)
        H = np.empty((len(idx), len(idx)))
        for col, j in enumerate(idx):
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            H[:, col] = (self.objective(xp)[1][idx] - self.objective(xm)[1][idx]) / (2 * h)
        return 0.5 * (H + H.T)


def sample_orders(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample Plackett-Luce orders via the Gumbel-max trick.

    ``theta`` has shape (..., n); returns indices sorted best-first along the last axis.
    """
    g = rng.gumbel(size=theta.shape)
    return np.argsort(-(theta + g), axis=-1)
