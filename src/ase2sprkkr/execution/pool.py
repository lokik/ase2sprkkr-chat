"""Endpoint-scoped executor identity and reuse."""

from __future__ import annotations

from threading import RLock


class Executors:
    """Own executor registration and endpoint-scoped lookup for a manager."""

    def __init__(self):
        self._instances = {}
        self._lock = RLock()

    def _get(self, endpoint, factory):
        with self._lock:
            executor = self._instances.get(endpoint)
            if executor is None:
                executor = factory()
                self._instances[endpoint] = executor
            return executor

    @staticmethod
    def _endpoint(value):
        from .endpoint import ExecutorEndpoint

        return ExecutorEndpoint(**value) if isinstance(value, dict) else value

    def register(self, executor, endpoint=None):
        from .endpoint import endpoint_for_executor

        endpoint = self._endpoint(
            endpoint_for_executor(executor) if endpoint is None else endpoint
        )
        registered = self._get(endpoint, lambda: executor)
        if registered is not executor:
            raise ValueError(f"Another executor is already registered for {endpoint!r}")
        return endpoint

    def for_endpoint(self, endpoint):
        from .endpoint import create_executor_from_endpoint

        endpoint = self._endpoint(endpoint)
        return self._get(endpoint, lambda: create_executor_from_endpoint(endpoint))

    def for_record(self, record):
        return self.for_endpoint(record.execution.endpoint)

    def for_profile(self, profile):
        return self.for_endpoint(profile.endpoint)

