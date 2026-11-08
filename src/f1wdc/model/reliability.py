"""Race DNF probabilities via beta-binomial shrinkage.

2026 status strings do not distinguish mechanical failures from crashes, so a DNF is a single
event type. Probability per driver = team-lineage rate (recent seasons, shrunk toward the grid
mean), lightly adjusted by the driver's own record, then scaled by the historical change in
the grid-wide DNF rate for this kind of season transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1wdc.config import transition_type


def race_starts(entries: pd.DataFrame) -> pd.DataFrame:
    return entries[entries["session"] == "race"][["season", "round", "driverId", "lineage", "dnf"]]


def grid_rate_by_season(entries: pd.DataFrame) -> pd.Series:
    return race_starts(entries).groupby("season")["dnf"].mean()


def season_shift(entries: pd.DataFrame, target: int) -> float:
    """Mean ratio grid_rate(N+1) / grid_rate(N) over past transitions of the same type."""
    rates = grid_rate_by_season(entries)
    kind = transition_type(target)
    ratios = [rates[n + 1] / rates[n] for n in rates.index
              if n + 1 in rates.index and n + 1 < target and transition_type(n + 1) == kind and rates[n] > 0]
    return float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0


def dnf_probabilities(
    entries: pd.DataFrame,
    target: int,
    driver_team: dict[str, str],
    as_of_round: int | None = None,
    alpha_team: float = 30.0,
    alpha_driver: float = 80.0,
) -> dict[str, float]:
    """P(DNF) per race for each driver in ``driver_team`` (driverId -> lineage) in ``target``.

    Pre-season rates come from seasons target-1 (weight 1) and target-2 (weight 0.5).
    Races of ``target`` already run (round <= as_of_round) update them as extra evidence.
    """
    rs = race_starts(entries)
    past = rs[rs["season"].isin([target - 1, target - 2])].copy()
    past["w"] = np.where(past["season"] == target - 1, 1.0, 0.5)
    grid_mean = float(rs[rs["season"] == target - 1]["dnf"].mean()) if (rs["season"] == target - 1).any() else 0.15

    team = past.groupby("lineage").apply(lambda g: pd.Series({"d": (g["w"] * g["dnf"]).sum(), "n": g["w"].sum()}),
                                         include_groups=False)
    drv = past.groupby("driverId").apply(lambda g: pd.Series({"d": (g["w"] * g["dnf"]).sum(), "n": g["w"].sum()}),
                                         include_groups=False)
    shift = season_shift(entries, target)

    cur = rs[(rs["season"] == target) & (rs["round"] <= (as_of_round or 0))]
    cur_team = cur.groupby("lineage")["dnf"].agg(["sum", "count"])

    out = {}
    for d, lin in driver_team.items():
        t_d, t_n = (team.loc[lin, "d"], team.loc[lin, "n"]) if lin in team.index else (0.0, 0.0)
        p_team = (t_d + alpha_team * grid_mean) / (t_n + alpha_team)
        d_d, d_n = (drv.loc[d, "d"], drv.loc[d, "n"]) if d in drv.index else (0.0, 0.0)
        p = (d_d + alpha_driver * p_team) / (d_n + alpha_driver)
        p *= shift
        if lin in cur_team.index:  # in-season evidence for this team
            c_sum, c_n = cur_team.loc[lin, "sum"], cur_team.loc[lin, "count"]
            p = (c_sum + alpha_team * p) / (c_n + alpha_team)
        out[d] = float(np.clip(p, 0.01, 0.5))
    return out
