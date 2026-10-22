import pytest

from ase2sprkkr.execution import Executors
from ase2sprkkr.sprkkr.profile import SPRKKRProfile


def test_executors_reuse_backend_by_endpoint():
    pool = Executors()
    first_profile = SPRKKRProfile(executor="local", mpi=8)
    second_profile = SPRKKRProfile(executor="local", mpi=64)

    first = pool.for_endpoint(first_profile.endpoint)
    second = pool.for_endpoint(second_profile.endpoint)

    assert first is second


def test_ssh_profiles_with_different_staging_share_endpoint_executor():
    pool = Executors()
    first_profile = SPRKKRProfile(
        executor="ssh",
        host="cluster",
        local_root="/home/a",
        remote_root="/scratch/a",
        input_sync="missing",
    )
    second_profile = SPRKKRProfile(
        executor="ssh",
        host="cluster",
        local_root="/home/b",
        remote_root="/scratch/b",
        input_sync="always",
    )

    assert first_profile.endpoint == second_profile.endpoint
    assert first_profile.launch_parameters != second_profile.launch_parameters
    assert pool.for_endpoint(first_profile.endpoint) is pool.for_endpoint(second_profile.endpoint)


def test_separate_executors_own_separate_pools():
    profile = SPRKKRProfile(executor="local", mpi=8)

    first = Executors()
    second = Executors()
    assert first.for_endpoint(profile.endpoint) is first.for_endpoint(profile.endpoint)
    assert first.for_endpoint(profile.endpoint) is not second.for_endpoint(profile.endpoint)


def test_explicit_registration_becomes_endpoint_authority():
    from ase2sprkkr.execution.local import LocalExecutor

    executors = Executors()
    backend = LocalExecutor()
    endpoint = executors.register(backend)

    assert executors.for_endpoint(endpoint) is backend
    assert executors.for_endpoint(SPRKKRProfile(executor="local").endpoint) is backend


def test_duplicate_registration_for_same_endpoint_is_rejected():
    from ase2sprkkr.execution.local import LocalExecutor

    executors = Executors()
    original = LocalExecutor()
    endpoint = executors.register(original)
    assert executors.register(original, endpoint=endpoint) == endpoint

    with pytest.raises(ValueError, match="already registered"):
        executors.register(LocalExecutor(), endpoint=endpoint)
    assert executors.for_endpoint(endpoint) is original


def test_persisted_endpoint_identity_uses_registered_executor():
    from types import SimpleNamespace
    from ase2sprkkr.execution.local import LocalExecutor

    executors = Executors()
    backend = LocalExecutor()
    endpoint = executors.register(backend)
    record = SimpleNamespace(execution=SimpleNamespace(endpoint=endpoint))

    assert executors.for_endpoint(record.execution.endpoint) is backend
    assert executors.for_endpoint(vars(endpoint)) is backend


def test_parallel_lookup_reuses_one_executor():
    from concurrent.futures import ThreadPoolExecutor
    from ase2sprkkr.execution.endpoint import ExecutorEndpoint

    executors = Executors()
    endpoint = ExecutorEndpoint(executor="local")
    with ThreadPoolExecutor(max_workers=12) as workers:
        instances = list(workers.map(executors.for_endpoint, [endpoint] * 48))

    assert all(instance is instances[0] for instance in instances)
