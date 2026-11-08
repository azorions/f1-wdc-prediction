"""Flatten cached Jolpica pages into tidy CSV tables under ``data/processed``.

Tables
------
entries.csv   one row per (season, round, session, driver): race / sprint / quali
drivers.csv   driverId, name, dateOfBirth, nationality
schedule.csv  season, round, raceName, date, has_sprint
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from f1wdc.config import lineage_map

log = logging.getLogger(__name__)

_LAPPED = re.compile(r"^\+\d+ Laps?$")
FINISH_STATUSES = {"Finished", "Lapped"}
DNS_STATUSES = {"Did not start", "Withdrew", "Did not qualify", "Did not prequalify", "Not classified "}
SESSION_KEYS = {"results": ("race", "Results"), "sprint": ("sprint", "SprintResults"), "qualifying": ("quali", "QualifyingResults")}


def _load_pages(raw_dir: Path, season: int, endpoint: str) -> list[dict]:
    path = raw_dir / str(season) / f"{endpoint}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["pages"]


def classify(position_text: str, status: str) -> tuple[bool, bool, bool]:
    """Return (classified, dnf, dns) for a race/sprint result row.

    classified: has a numeric finishing position (counts for ranking and points)
    dnf:        started but retired / not classified
    dns:        never started (withdrawn, DNQ, DNS) -> dropped from the data
    Disqualified / excluded rows are neither classified nor DNF (left out of ranking).
    """
    if position_text.isdigit():
        return True, False, False
    if position_text in {"W", "F"} or status in DNS_STATUSES or status.startswith("Did not"):
        return False, False, True
    if position_text in {"D", "E"}:
        return False, False, False
    return False, True, False  # "R", "N" and anything else = did not finish


def season_rows(raw_dir: Path, season: int, lineage: dict[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    entries: list[dict] = []
    for endpoint, (session, key) in SESSION_KEYS.items():
        for page in _load_pages(raw_dir, season, endpoint):
            for race in page["RaceTable"]["Races"]:
                rnd = int(race["round"])
                for r in race.get(key, []):
                    cid = r["Constructor"]["constructorId"]
                    row = {
                        "season": season,
                        "round": rnd,
                        "session": session,
                        "driverId": r["Driver"]["driverId"],
                        "constructorId": cid,
                        "lineage": lineage.get(cid, cid),
                        "position": int(r["position"]),
                    }
                    if session == "quali":
                        row.update(positionText=str(r["position"]), status="", grid=None, points=0.0,
                                   classified=True, dnf=False, dns=False)
                    else:
                        status = r.get("status", "")
                        ptext = r.get("positionText", "")
                        classified, dnf, dns = classify(ptext, status)
                        row.update(positionText=ptext, status=status, grid=int(r.get("grid") or 0),
                                   points=float(r.get("points") or 0.0), classified=classified, dnf=dnf, dns=dns)
                    entries.append(row)

    drivers = []
    for page in _load_pages(raw_dir, season, "drivers"):
        for d in page["DriverTable"]["Drivers"]:
            drivers.append({
                "driverId": d["driverId"],
                "code": d.get("code", ""),
                "givenName": d.get("givenName", ""),
                "familyName": d.get("familyName", ""),
                "dateOfBirth": d.get("dateOfBirth", ""),
                "nationality": d.get("nationality", ""),
            })

    schedule = []
    for page in _load_pages(raw_dir, season, "races"):
        for race in page["RaceTable"]["Races"]:
            schedule.append({
                "season": season,
                "round": int(race["round"]),
                "raceName": race["raceName"],
                "date": race["date"],
                "has_sprint": "Sprint" in race,
            })
    return entries, drivers, schedule


def build_all(data_dir: Path) -> str:
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage = lineage_map()
    seasons = sorted(int(p.name) for p in raw_dir.iterdir() if p.is_dir() and p.name.isdigit())
    all_e, all_d, all_s = [], [], []
    for s in seasons:
        e, d, sch = season_rows(raw_dir, s, lineage)
        all_e += e
        all_d += d
        all_s += sch
    entries = pd.DataFrame(all_e)
    drivers = pd.DataFrame(all_d).drop_duplicates("driverId", keep="last")
    schedule = pd.DataFrame(all_s).drop_duplicates(["season", "round"])

    unmapped = sorted(set(entries["constructorId"]) - set(lineage))
    if unmapped:
        log.warning("constructorIds without lineage mapping (treated as own lineage): %s", ", ".join(unmapped))

    entries = entries[~entries["dns"]].reset_index(drop=True)
    entries.to_csv(out_dir / "entries.csv", index=False)
    drivers.to_csv(out_dir / "drivers.csv", index=False)
    schedule.to_csv(out_dir / "schedule.csv", index=False)
    n_events = entries.groupby(["season", "round", "session"]).ngroups
    return (f"Built tables: {len(entries)} entries across {n_events} sessions, "
            f"{len(drivers)} drivers, seasons {seasons[0]}-{seasons[-1]}")


def load_tables(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = Path(data_dir) / "processed"
    entries = pd.read_csv(p / "entries.csv")
    drivers = pd.read_csv(p / "drivers.csv")
    schedule = pd.read_csv(p / "schedule.csv")
    return entries, drivers, schedule
