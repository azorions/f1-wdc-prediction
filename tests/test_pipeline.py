import numpy as np
import pandas as pd

from f1wdc.pipeline import points_so_far, season_events, sprints_run_after


def _entries():
    rows = []
    # round 1: sprint + race done; round 2: only the sprint done so far
    for rnd, session, order in [(1, "sprint", ["a", "b"]), (1, "race", ["b", "a"]), (2, "sprint", ["a", "b"])]:
        pts = {"race": [25, 18], "sprint": [8, 7]}[session]
        for pos, d in enumerate(order):
            rows.append({"season": 2027, "round": rnd, "session": session, "driverId": d, "position": pos + 1,
                         "points": pts[pos], "classified": True})
    return pd.DataFrame(rows)


def test_half_finished_weekend_counts_sprint_and_skips_it_in_simulation():
    e = _entries()
    assert sprints_run_after(e, 2027, as_of=1) == {2}
    pts, cb = points_so_far(e, 2027, 1, ["a", "b"])
    np.testing.assert_allclose(pts, [8 + 18 + 8, 7 + 25 + 7])
    np.testing.assert_allclose(cb[:, 0], [0, 1])  # countback uses grand prix wins only


def test_season_events_order_and_sprints():
    ev = season_events(4, [2, 4], after_round=1)
    assert [(e.round, e.kind) for e in ev] == [(2, "sprint"), (2, "race"), (3, "race"), (4, "sprint"), (4, "race")]
    assert ev[-1].u == 0.5
