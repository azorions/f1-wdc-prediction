"""Throttled, paginated, cached client for the Jolpica (Ergast-compatible) F1 API.

Limits (https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md):
4 requests/second burst, 500 requests/hour sustained, max ``limit=100`` per page.

Each (season, endpoint) is cached as one JSON file holding every raw page. Seasons
that are over are cached forever; the current season is refetched once its cache
is older than ``max_age_hours``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1/"
USER_AGENT = "f1-wdc-prediction (research project)"
PAGE_LIMIT = 100


class RateLimiter:
    """Minimum spacing between requests plus a rolling one-hour cap.

    With ``state_path`` the request timestamps are persisted, so separate runs share the
    hourly budget (the API counts every request, including throttled ones).
    """

    def __init__(
        self,
        per_second: float = 2.0,
        hourly_cap: int = 450,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        state_path: Path | None = None,
    ):
        self.min_interval = 1.0 / per_second
        self.hourly_cap = hourly_cap
        self.clock = clock
        self.sleep = sleep
        self.state_path = state_path
        self._last = -1e18
        self._window: deque[float] = deque()
        if state_path is not None and state_path.exists():
            try:
                now = clock()
                stamps = json.loads(state_path.read_text(encoding="utf-8"))
                self._window.extend(t for t in stamps if now - t < 3600)
            except (ValueError, OSError):
                pass

    def _save(self) -> None:
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(list(self._window)), encoding="utf-8")

    def wait(self) -> None:
        now = self.clock()
        while self._window and now - self._window[0] >= 3600:
            self._window.popleft()
        if len(self._window) >= self.hourly_cap:
            pause = 3600 - (now - self._window[0]) + 1
            log.warning("Hourly request cap reached; pausing %.0f s", pause)
            self.sleep(pause)
            now = self.clock()
            while self._window and now - self._window[0] >= 3600:
                self._window.popleft()
        gap = now - self._last
        if gap < self.min_interval:
            self.sleep(self.min_interval - gap)
            now = self.clock()
        self._last = now
        self._window.append(now)
        self._save()


class JolpicaClient:
    def __init__(
        self,
        cache_dir: Path | str = "data/raw",
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.limiter = limiter or RateLimiter(sleep=sleep, state_path=self.cache_dir / ".request_log.json")
        self.sleep = sleep
        self.max_retries = max_retries
        self.requests_made = 0

    # ------------------------------------------------------------------ HTTP
    def get_json(self, path: str, params: dict | None = None) -> dict:
        url = BASE_URL + path
        backoff = 5.0
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            self.requests_made += 1
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:  # network blip
                if attempt == self.max_retries:
                    raise
                log.warning("Request error %s (%s); retrying in %.0f s", url, exc, backoff)
                self.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                pause = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                log.warning("HTTP %s for %s; retrying in %.0f s", resp.status_code, url, pause)
                self.sleep(pause)
                backoff = min(backoff * 2, 300)
                continue
            resp.raise_for_status()
            return resp.json()["MRData"]
        raise RuntimeError(f"unreachable: {url}")

    def fetch_pages(self, path: str) -> list[dict]:
        """Fetch every page of ``path`` (e.g. ``"2024/results.json"``)."""
        pages: list[dict] = []
        offset = 0
        while True:
            page = self.get_json(path, {"limit": PAGE_LIMIT, "offset": offset})
            pages.append(page)
            total = int(page.get("total", 0))
            offset += PAGE_LIMIT
            if offset >= total:
                return pages

    # ----------------------------------------------------------------- cache
    def cache_path(self, season: int, endpoint: str) -> Path:
        return self.cache_dir / str(season) / f"{endpoint}.json"

    def season_endpoint(
        self, season: int, endpoint: str, max_age_hours: float | None, refresh: bool = False
    ) -> list[dict]:
        """Return cached pages for ``season/endpoint``, fetching when missing or stale.

        ``max_age_hours=None`` means the cache never expires (finished seasons).
        """
        path = self.cache_path(season, endpoint)
        if path.exists() and not refresh:
            blob = json.loads(path.read_text(encoding="utf-8"))
            if max_age_hours is None:
                return blob["pages"]
            fetched = datetime.fromisoformat(blob["fetched_at"])
            age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age_h < max_age_hours:
                return blob["pages"]
        pages = self.fetch_pages(f"{season}/{endpoint}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"fetched_at": datetime.now(timezone.utc).isoformat(), "pages": pages}
        path.write_text(json.dumps(blob), encoding="utf-8")
        return pages


def endpoints_for(season: int) -> list[str]:
    eps = ["races", "results", "drivers"]
    if season >= 1994:
        eps.append("qualifying")
    if season >= 2021:
        eps.append("sprint")
    return eps


def fetch_all(
    client: JolpicaClient,
    start: int = 1994,
    end: int | None = None,
    refresh: bool = False,
    max_age_hours: float = 6.0,
    today: date | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Fetch every season from ``start`` to ``end`` (default: current year), plus the
    next season's calendar. Seasons before the current calendar year never expire."""
    today = today or date.today()
    end = end or today.year
    for season in range(start, end + 1):
        age = None if season < today.year else max_age_hours
        for ep in endpoints_for(season):
            if progress:
                progress(f"{season} {ep}")
            client.season_endpoint(season, ep, age, refresh=refresh and age is not None)
    # Next season's calendar (may be empty if not yet published).
    if progress:
        progress(f"{end + 1} races")
    client.season_endpoint(end + 1, "races", max_age_hours, refresh=refresh)
