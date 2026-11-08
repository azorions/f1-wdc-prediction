import numpy as np
import pandas as pd
import pytest

from f1wdc.model import projection as P


def synthetic_tables(rho_by_type: dict[str, float], n_teams=10, seasons=range(1994, 2027), noise=0.3, seed=0):
    from f1wdc.config import transition_type

    rng = np.random.default_rng(seed)
    car_rows, drv_rows = [], []
    level = rng.normal(0, 1, n_teams)
    for s in seasons:
        if s > seasons[0]:
            rho = rho_by_type[transition_type(s)]
            level = rho * prev_end + rng.normal(0, noise, n_teams)
        level = level - level.mean()
        slope = rng.normal(0, 0.1, n_teams)
        end = level + 0.5 * slope
        prev_end = end
        for t in range(n_teams):
            car_rows.append({"season": s, "lineage": f"t{t}", "c_level": level[t], "c_slope": slope[t], "c_end": end[t]})
            for k in range(2):
                drv_rows.append({"season": s, "driverId": f"d{t}_{k}", "starts": 20, "s": 0.0})
    return pd.DataFrame(drv_rows), pd.DataFrame(car_rows)


def test_carry_over_regression_recovers_rho_per_type():
    truth = {"stable": 0.8, "reg_year_1": 0.3, "reg_year_2": 0.8}
    drivers, cars = synthetic_tables(truth, n_teams=40, noise=0.2)
    info = pd.DataFrame({"driverId": drivers["driverId"].unique(), "dateOfBirth": "1995-01-01"})
    params = P.estimate(drivers, cars, info, target=2027, shrink_pairs=0)
    assert params.rho["stable"] == pytest.approx(0.8, abs=0.05)
    assert params.rho["reg_year_1"] == pytest.approx(0.3, abs=0.1)
    assert params.tau["stable"] == pytest.approx(0.2, abs=0.05)


def test_projection_uses_transition_type_and_override():
    drivers, cars = synthetic_tables({"stable": 0.8, "reg_year_1": 0.3, "reg_year_2": 0.8})
    info = pd.DataFrame({"driverId": drivers["driverId"].unique(), "dateOfBirth": "1995-01-01"})
    params = P.estimate(drivers, cars, info, target=2027)
    grid = {f"t{t}": [f"d{t}_0", f"d{t}_1"] for t in range(10)}
    proj = P.project(params, drivers, cars, info, grid, 2027, carryover=0.5)
    prev = cars[cars["season"] == 2026].set_index("lineage")["c_end"]
    expected = 0.5 * prev.loc[list(grid)].to_numpy()
    np.testing.assert_allclose(proj.car_mean(), expected - expected.mean(), atol=1e-9)


def test_rookie_gets_rookie_prior():
    drivers, cars = synthetic_tables({"stable": 0.8, "reg_year_1": 0.3, "reg_year_2": 0.8})
    info = pd.DataFrame({"driverId": list(drivers["driverId"].unique()) + ["newbie"], "dateOfBirth": "2005-01-01"})
    params = P.estimate(drivers, cars, info, target=2027)
    grid = {"t0": ["d0_0", "newbie"]}
    proj = P.project(params, drivers, cars, info, grid, 2027)
    assert proj.source["newbie"] == "rookie"
    assert proj.driver_sd()[1] == pytest.approx(params.rookie_sd)
