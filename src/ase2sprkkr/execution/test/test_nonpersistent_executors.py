"""Non-project runs share Executors directly without a RunStore."""

from ase2sprkkr.execution.persistence import default_executors
from ase2sprkkr.execution.pool import Executors
from ase2sprkkr.execution.endpoint import ExecutorEndpoint


def test_default_executors_has_no_manager_or_store():
    registry = default_executors()
    assert isinstance(registry, Executors)
    assert registry is default_executors()
    endpoint = ExecutorEndpoint()
    assert registry.for_endpoint(endpoint) is registry.for_endpoint(endpoint)
