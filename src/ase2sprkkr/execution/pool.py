"""Reusable execution backend pooling independent of run persistence."""

from __future__ import annotations

from threading import RLock

from .base import _close_distinct


class _ExecutorPool:
    """Process-local cache of reusable execution backends."""

    def __init__(self):
        self._executors = {}
        self._lock = RLock()

    def get(self, key, factory):
        with self._lock:
            executor = self._executors.get(key)
            if executor is None:
                executor = factory()
                self._executors[key] = executor
            return executor

    def for_endpoint(self, endpoint):
        """Return the shared backend reconstructed from *endpoint*."""
        from .endpoint import ExecutorEndpoint, create_executor_from_endpoint

        if isinstance(endpoint, dict):
            endpoint = ExecutorEndpoint(**endpoint)
        return self.get(endpoint, lambda: create_executor_from_endpoint(endpoint))

    def discard(self, key):
        """Forget one pooled backend and close it when supported."""
        with self._lock:
            executor = self._executors.pop(key, None)
        if executor is not None:
            close = getattr(executor, "close", None)
            if close is not None:
                close()
        return executor

    def close(self):
        """Close every pooled executor, propagating the first close failure."""
        with self._lock:
            executors = tuple(self._executors.values())
            self._executors.clear()
        _close_distinct(executors)


class Executors:
    """Own executor registration and endpoint-scoped lookup for a manager."""

    def __init__(self):
        self._pool = _ExecutorPool()

    def register(self, executor, endpoint=None):
        from .endpoint import endpoint_for_executor

        endpoint = endpoint or endpoint_for_executor(executor)
        registered = self._pool.get(endpoint, lambda: executor)
        if registered is not executor:
            raise ValueError(f"Another executor is already registered for {endpoint!r}")
        return endpoint

    def for_endpoint(self, endpoint):
        return self._pool.for_endpoint(endpoint)

    def for_record(self, record):
        return self.for_endpoint(record.execution.endpoint)

    def for_profile(self, profile):
        return self.for_endpoint(profile.endpoint)

    def discard(self, endpoint):
        return self._pool.discard(endpoint)

    def close(self):
        self._pool.close()
