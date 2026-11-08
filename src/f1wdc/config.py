"""Loading of YAML configuration and historical points systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

RACE_POINTS_2010 = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS_2022 = [8, 7, 6, 5, 4, 3, 2, 1]


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@lru_cache
def lineage_map() -> dict[str, str]:
    raw = _load("teams.yaml")
    mapping = {cid: lineage for lineage, cids in raw["lineages"].items() for cid in cids}
    mapping.update({cid: cid for cid in raw.get("standalone", [])})
    return mapping


@lru_cache
def reg_year_1() -> frozenset[int]:
    return frozenset(_load("regulations.yaml")["reg_year_1"])


def transition_type(next_season: int) -> str:
    """Type of the transition from ``next_season - 1`` to ``next_season``."""
    ry1 = reg_year_1()
    if next_season in ry1:
        return "reg_year_1"
    if next_season - 1 in ry1:
        return "reg_year_2"
    return "stable"


def points_system(season: int) -> tuple[list[int], list[int]]:
    """(race points, sprint points) by finishing position. Fastest-lap bonus ignored."""
    if season >= 2010:
        race = RACE_POINTS_2010
    elif season >= 2003:
        race = [10, 8, 6, 5, 4, 3, 2, 1]
    elif season >= 1991:
        race = [10, 6, 4, 3, 2, 1]
    else:
        race = [9, 6, 4, 3, 2, 1]
    if season >= 2022:
        sprint = SPRINT_POINTS_2022
    elif season == 2021:
        sprint = [3, 2, 1]
    else:
        sprint = []
    return race, sprint


@dataclass
class SeasonConfig:
    season: int
    grid: dict[str, list[str]]                 # lineage -> driverIds
    team_names: dict[str, str]                 # lineage -> display name
    n_rounds: int
    sprint_rounds: list[int]
    race_points: list[int] = field(default_factory=lambda: list(RACE_POINTS_2010))
    sprint_points: list[int] = field(default_factory=lambda: list(SPRINT_POINTS_2022))
    dob_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def drivers(self) -> list[str]:
        return [d for ds in self.grid.values() for d in ds]

    @property
    def driver_team(self) -> dict[str, str]:
        return {d: t for t, ds in self.grid.items() for d in ds}


def load_season_config(path: Path | str) -> SeasonConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    grid, names = {}, {}
    for team in data["teams"]:
        grid[team["lineage"]] = list(team["drivers"])
        names[team["lineage"]] = team["name"]
    cal = data.get("calendar", {})
    pts = data.get("points", {})
    return SeasonConfig(
        season=int(data["season"]),
        grid=grid,
        team_names=names,
        n_rounds=int(cal.get("rounds", 24)),
        sprint_rounds=list(cal.get("sprint_rounds", [])),
        race_points=list(pts.get("race", RACE_POINTS_2010)),
        sprint_points=list(pts.get("sprint", SPRINT_POINTS_2022)),
        dob_overrides=dict(data.get("dob_overrides") or {}),
    )
