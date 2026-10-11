"""Thread-safe snapshot cache with bounded stale responses and refresh backoff."""

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class CacheUnavailable(RuntimeError):
    pass


class InjuryCache:
    def __init__(
        self,
        loader: Callable[[], dict],
        ttl: int = 120,
        max_age: int = 900,
        retry_seconds: int = 30,
        clock: Callable[[], float] = time.monotonic,
        refresh_wait_seconds: float = 15,
    ):
        if ttl <= 0 or max_age < ttl or retry_seconds <= 0 or refresh_wait_seconds <= 0:
            raise ValueError("Cache durations must be positive; max_age must be at least ttl")
        self.loader, self.ttl, self.max_age = loader, ttl, max_age
        self.retry_seconds, self.clock = retry_seconds, clock
        self.refresh_wait_seconds = refresh_wait_seconds
        self._state = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._payload = None
        self._loaded_at = 0.0
        self._retry_at = 0.0
        self._stop = threading.Event()
        self._worker = None

    def _read(self):
        with self._state:
            age = max(0.0, self.clock() - self._loaded_at)
            return self._payload, age, self._retry_at

    def refresh(self, blocking: bool = False):
        acquired = (
            self._refresh_lock.acquire(timeout=self.refresh_wait_seconds)
            if blocking
            else self._refresh_lock.acquire(blocking=False)
        )
        if not acquired:
            return
        try:
            payload, age, retry_at = self._read()
            if (payload is not None and age < self.ttl) or self.clock() < retry_at:
                return
            try:
                replacement = self.loader()
            except Exception as exc:
                # Never log source bodies, request URLs, or authentication values.
                logger.warning("Injury source refresh failed (%s)", type(exc).__name__)
                with self._state:
                    self._retry_at = self.clock() + self.retry_seconds
                return
            with self._state:
                self._payload = replacement
                self._loaded_at = self.clock()
                self._retry_at = 0.0
        finally:
            self._refresh_lock.release()

    def get(self) -> tuple[dict, dict]:
        payload, age, _ = self._read()
        if payload is None or age > self.max_age:
            self.refresh(blocking=True)
            payload, age, _ = self._read()
        if payload is None or age > self.max_age:
            raise CacheUnavailable("No sufficiently recent injury snapshot is available")
        # The lifespan worker refreshes expired entries. A bounded stale snapshot
        # stays immediately available while that refresh is in progress.
        return payload, {
            "ageSeconds": int(age),
            "ttlSeconds": self.ttl,
            "maxAgeSeconds": self.max_age,
            "stale": age >= self.ttl,
        }

    def _run(self):
        while not self._stop.is_set():
            self.refresh()
            payload, age, retry_at = self._read()
            delay = (
                max(0.1, retry_at - self.clock())
                if retry_at
                else max(0.1, self.ttl - age)
                if payload is not None
                else self.retry_seconds
            )
            self._stop.wait(delay)

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="injury-cache-refresh", daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=15)
