"""Reusable execution backend pooling independent of run persistence."""

from __future__ import annotations

from threading import RLock


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


class Executors:
    """Own executor registration and endpoint-scoped lookup for a manager."""

    def __init__(self):
        self._pool = _ExecutorPool()

    def register(self, executor, endpoint=None):
        from .endpoint import ExecutorEndpoint, endpoint_for_executor

        if endpoint is None:
            endpoint = endpoint_for_executor(executor)
        elif isinstance(endpoint, dict):
            endpoint = ExecutorEndpoint(**endpoint)
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

