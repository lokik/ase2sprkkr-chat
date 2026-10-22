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
        from .endpoint import create_executor_from_endpoint

        return self.get(endpoint, lambda: create_executor_from_endpoint(endpoint))


class Executors:
    """Own executor registration and endpoint-scoped lookup for a manager."""

    def __init__(self):
        self._pool = _ExecutorPool()

    @staticmethod
    def _endpoint(value):
        from .endpoint import ExecutorEndpoint

        return ExecutorEndpoint(**value) if isinstance(value, dict) else value

    def register(self, executor, endpoint=None):
        from .endpoint import endpoint_for_executor

        endpoint = self._endpoint(
            endpoint_for_executor(executor) if endpoint is None else endpoint
        )
        registered = self._pool.get(endpoint, lambda: executor)
        if registered is not executor:
            raise ValueError(f"Another executor is already registered for {endpoint!r}")
        return endpoint

    def for_endpoint(self, endpoint):
        return self._pool.for_endpoint(self._endpoint(endpoint))

    def for_record(self, record):
        return self.for_endpoint(record.execution.endpoint)

    def for_profile(self, profile):
        return self.for_endpoint(profile.endpoint)

