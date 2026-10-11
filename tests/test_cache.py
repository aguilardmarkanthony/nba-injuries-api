import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from injury_cache import CacheUnavailable, InjuryCache


def test_concurrent_cold_requests_share_one_source_fetch(snapshot):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def loader():
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return snapshot

    cache = InjuryCache(loader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(cache.get) for _ in range(8)]
        assert entered.wait(3)
        release.set()
        assert all(job.result()[0] == snapshot for job in jobs)
    assert len(calls) == 1


def test_failed_refresh_keeps_bounded_stale_snapshot_and_uses_backoff(snapshot):
    now, calls = [100.0], []

    def loader():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("upstream unavailable")
        return snapshot

    cache = InjuryCache(loader, ttl=10, max_age=20, retry_seconds=5, clock=lambda: now[0])
    original, fresh = cache.get()
    assert not fresh["stale"]
    now[0] += 11
    cache.refresh()
    stale, info = cache.get()
    assert stale == original and stale["updatedAt"] == original["updatedAt"]
    assert info["stale"] and info["ageSeconds"] == 11
    cache.refresh()
    assert len(calls) == 2
    now[0] += 10
    with pytest.raises(CacheUnavailable):
        cache.get()
    assert len(calls) == 3
    with pytest.raises(CacheUnavailable):
        cache.get()
    assert len(calls) == 3


def test_stale_read_does_not_wait_for_background_refresh(snapshot):
    now = [100.0]
    cache = InjuryCache(lambda: snapshot, ttl=10, max_age=20, clock=lambda: now[0])
    cache.get()
    now[0] += 11
    cache._refresh_lock.acquire()
    try:
        assert cache.get()[1]["stale"]
    finally:
        cache._refresh_lock.release()


def test_background_worker_populates_and_stops(snapshot):
    fetched = threading.Event()

    def loader():
        fetched.set()
        return snapshot

    cache = InjuryCache(loader)
    cache.start()
    assert fetched.wait(3)
    cache.stop()
    assert not cache._worker.is_alive()
    assert cache.get()[0] == snapshot


@pytest.mark.parametrize("ttl,max_age", [(0, 10), (-1, 10), (10, 9)])
def test_invalid_cache_configuration_fails_clearly(ttl, max_age):
    with pytest.raises(ValueError):
        InjuryCache(lambda: {}, ttl=ttl, max_age=max_age)


def test_cold_waiter_has_a_bounded_wait(snapshot):
    cache = InjuryCache(lambda: snapshot, refresh_wait_seconds=0.01)
    cache._refresh_lock.acquire()
    try:
        with pytest.raises(CacheUnavailable):
            cache.get()
    finally:
        cache._refresh_lock.release()
