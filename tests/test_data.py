import json

import pytest

from f1wdc.config import lineage_map, points_system, transition_type
from f1wdc.data.build import classify
from f1wdc.data.jolpica import JolpicaClient, RateLimiter


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def page(total, offset):
    return {"MRData": {"total": str(total), "offset": str(offset), "RaceTable": {"Races": []}}}


def make_client(tmp_path, responses):
    sleeps = []
    limiter = RateLimiter(per_second=1000, hourly_cap=10_000, sleep=sleeps.append)
    session = FakeSession(responses)
    return JolpicaClient(cache_dir=tmp_path, session=session, limiter=limiter, sleep=sleeps.append), session, sleeps


def test_paginates_until_total(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, page(250, 0)), FakeResponse(200, page(250, 100)),
                                                FakeResponse(200, page(250, 200))])
    pages = client.fetch_pages("2024/results.json")
    assert len(pages) == 3
    assert [c[1]["offset"] for c in session.calls] == [0, 100, 200]
    assert all(c[1]["limit"] == 100 for c in session.calls)


def test_finished_season_cache_never_refetches(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, page(5, 0))])
    client.season_endpoint(2020, "results", max_age_hours=None)
    client.season_endpoint(2020, "results", max_age_hours=None)
    assert len(session.calls) == 1
    assert json.loads((tmp_path / "2020" / "results.json").read_text())["pages"][0]["total"] == "5"


def test_429_backs_off_and_honours_retry_after(tmp_path):
    client, session, sleeps = make_client(tmp_path, [FakeResponse(429, headers={"Retry-After": "7"}),
                                                     FakeResponse(200, page(1, 0))])
    client.fetch_pages("2024/races.json")
    assert 7.0 in sleeps
    assert len(session.calls) == 2


def test_rate_limiter_enforces_hourly_cap():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    lim = RateLimiter(per_second=100, hourly_cap=3, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        lim.wait()
    assert sum(slept) >= 3600 - 1


@pytest.mark.parametrize("ptext,status,expected", [
    ("1", "Finished", (True, False, False)),
    ("12", "Lapped", (True, False, False)),
    ("15", "Engine", (True, False, False)),      # classified despite retiring late
    ("R", "Retired", (False, True, False)),
    ("N", "Not classified", (False, True, False)),
    ("W", "Withdrew", (False, False, True)),
    ("F", "Did not qualify", (False, False, True)),
    ("R", "Did not start", (False, False, True)),
    ("D", "Disqualified", (False, False, False)),
])
def test_classify(ptext, status, expected):
    assert classify(ptext, status) == expected


def test_transition_types_and_lineage():
    assert transition_type(2027) == "reg_year_2"
    assert transition_type(2026) == "reg_year_1"
    assert transition_type(2025) == "stable"
    lm = lineage_map()
    assert lm["toro_rosso"] == lm["rb"] == "faenza"
    assert lm["sauber"] == lm["audi"] == "hinwil"
    assert lm["ferrari"] == "ferrari"


def test_points_systems():
    assert points_system(2027)[0][:3] == [25, 18, 15]
    assert points_system(2005)[0] == [10, 8, 6, 5, 4, 3, 2, 1]
    assert points_system(2021)[1] == [3, 2, 1]
    assert points_system(1998)[1] == []
