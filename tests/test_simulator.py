import numpy as np
import pytest

from f1wdc.model.projection import Projection
from f1wdc.sim.season import Event, simulate

RACE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT = [8, 7, 6, 5, 4, 3, 2, 1]


def _proj(s, car, teams_of, sd=0.0):
    n, t = len(s), len(car)
    mean = np.concatenate([s, car, np.zeros(t)])
    cov = np.diag(np.full(len(mean), sd**2))
    return Projection([f"d{i}" for i in range(n)], [f"t{j}" for j in range(t)], np.array(teams_of),
                      mean, cov, {})


def test_dominant_driver_wins_everything_and_points_add_up():
    proj = _proj([10.0, 0, 0, 0], [0.0, 0.0], [0, 0, 1, 1])
    events = [Event(r, 0.0, "race") for r in range(1, 6)]
    res = simulate(proj, np.zeros(4), events, RACE, SPRINT, n_sims=500, seed=0)
    assert res.p_wdc()[0] == 1.0
    np.testing.assert_allclose(res.points[:, 0], 5 * 25)
    # every race hands out 25 + 18 + 15 + 12 points across four finishers
    np.testing.assert_allclose(res.points.sum(axis=1), 5 * (25 + 18 + 15 + 12))
    np.testing.assert_allclose(res.team_points.sum(axis=1), res.points.sum(axis=1))


def test_sprint_points_and_dnfs():
    proj = _proj([40.0, 20.0, 0.0], [0.0], [0, 0, 0])
    events = [Event(1, 0.0, "sprint")]
    res = simulate(proj, np.array([0.0, 0.0, 0.0]), events, RACE, SPRINT, n_sims=200, seed=0)
    np.testing.assert_allclose(res.points.mean(axis=0), [8, 7, 6])
    # A certain DNF scores nothing.
    res = simulate(proj, np.array([0.5, 0.0, 0.0]), [Event(1, 0.0, "race")] * 1, RACE, SPRINT, n_sims=4000, seed=1)
    assert (res.points[:, 0] == 0).mean() == pytest.approx(0.5, abs=0.03)


def test_countback_breaks_points_ties():
    # d0 and d1 tie on points so far; d0 has more wins -> champion when nothing is left to race.
    proj = _proj([0.0, 0.0], [0.0], [0, 0])
    res = simulate(proj, np.zeros(2), [], RACE, SPRINT, n_sims=100, seed=0,
                   points_so_far=np.array([100.0, 100.0]),
                   countback_so_far=np.array([[3, 0, 0], [2, 5, 0]]))
    assert res.p_wdc()[0] == 1.0


def test_in_season_points_carry_over():
    proj = _proj([0.0, 0.0, 0.0], [0.0], [0, 0, 0])
    res = simulate(proj, np.zeros(3), [Event(23, 0.5, "race")], RACE, SPRINT, n_sims=2000, seed=0,
                   points_so_far=np.array([200.0, 190.0, 0.0]))
    # leader with a 10-point cushion and one race left wins unless rival wins and leader is 3rd
    p = res.p_wdc()
    assert p[2] == 0.0 and p[0] > p[1]
