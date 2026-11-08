import numpy as np
import pytest

from f1wdc.model.plackett_luce import QUALI, RACE, SPRINT, PlackettLuce, make_design, sample_orders


def _random_design(rng, n_events=12, n_drivers=8, n_teams=4):
    orders, types, ed, et, eu = [], [], [], [], []
    for e in range(n_events):
        n = rng.integers(4, n_drivers + 1)
        drivers = rng.permutation(n_drivers)[:n]
        start = len(ed)
        for d in drivers:
            ed.append(d)
            et.append(d % n_teams)
            eu.append(rng.uniform(-0.5, 0.5))
        orders.append(np.arange(start, start + n))
        types.append([RACE, QUALI, SPRINT][e % 3])
    return make_design(orders, types, ed, et, eu, n_drivers, n_teams,
                       event_weights=rng.uniform(0.5, 1.5, n_events))


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    design = _random_design(rng)
    pairs = (np.array([0, 2, 5]), np.array([1, 3, 6]), np.array([4.0, 2.0, 9.0]))
    model = PlackettLuce(design, prior_mean=None, pairs=pairs)
    x = rng.normal(0, 0.7, design.n_params)
    _, g = model.objective(x)
    h = 1e-6
    num = np.empty_like(x)
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        num[j] = (model.objective(xp)[0] - model.objective(xm)[0]) / (2 * h)
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-6)


def test_two_driver_likelihood_is_logistic():
    # one race, A beats B: P = e^a / (e^a + e^b)
    design = make_design([np.array([0, 1])], [RACE], [0, 1], [0, 0], [0.0, 0.0], 2, 1)
    model = PlackettLuce(design, prior_prec=np.zeros(design.n_params))
    x = np.zeros(design.n_params)
    x[0], x[1] = 0.7, -0.2
    f, _ = model.objective(x)
    expected = -np.log(np.exp(0.7) / (np.exp(0.7) + np.exp(-0.2)))
    assert f == pytest.approx(expected)


def test_gumbel_sampling_matches_softmax_win_probabilities():
    rng = np.random.default_rng(1)
    theta = np.array([1.2, 0.5, 0.0, -0.8])
    orders = sample_orders(np.broadcast_to(theta, (200_000, 4)), rng)
    freq = np.bincount(orders[:, 0], minlength=4) / orders.shape[0]
    p = np.exp(theta) / np.exp(theta).sum()
    np.testing.assert_allclose(freq, p, atol=0.005)


def test_recovers_driver_and_car_effects_with_transfers():
    rng = np.random.default_rng(2)
    n_drivers, n_teams, n_seasons, races = 24, 8, 3, 22
    s_true = rng.normal(0, 0.4, n_drivers)
    car_true = rng.normal(0, 1.0, (n_seasons, n_teams))
    seat = np.repeat(np.arange(n_teams), 3)           # 3 drivers per team
    orders, types, ed, et, eu = [], [], [], [], []
    for season in range(n_seasons):
        if season:  # shuffle a third of the drivers to new teams each season
            movers = rng.choice(n_drivers, n_drivers // 3, replace=False)
            seat[movers] = rng.permutation(seat[movers])
        for _ in range(races):
            for typ, kappa in ((RACE, 1.0), (QUALI, 2.0)):
                field = np.arange(n_drivers)
                theta = s_true[field] + car_true[season, seat[field]]
                order = field[sample_orders(kappa * theta, rng)]
                start = len(ed)
                for d in order:
                    ed.append(d)
                    et.append(season * n_teams + seat[d])
                    eu.append(0.0)
                orders.append(np.arange(start, start + n_drivers))
                types.append(typ)
    design = make_design(orders, types, ed, et, eu, n_drivers, n_seasons * n_teams)
    model = PlackettLuce(design, sigma_driver=0.5, sigma_car=1.0, sigma_slope=0.1)
    fit = model.fit()
    assert fit.success
    s_hat = fit.x[model.sl_s]
    c_hat = fit.x[model.sl_cl].reshape(n_seasons, n_teams)
    assert np.corrcoef(s_hat, s_true)[0, 1] > 0.85
    for season in range(n_seasons):
        assert np.corrcoef(c_hat[season], car_true[season])[0, 1] > 0.9
    assert model.kappa(fit.x)[QUALI] == pytest.approx(2.0, rel=0.2)
