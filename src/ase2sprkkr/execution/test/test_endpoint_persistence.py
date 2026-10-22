from threading import Event

from ase2sprkkr.execution import (
    
    ExecutorEndpoint,
    LocalExecutor,
    RemoteEndpointExecutor,
    RunState,
    RunStore,
    SshTransport,
)

from ase2sprkkr.execution.manager import ExecutionManager
from ase2sprkkr.execution.pool import Executors
from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.sprkkr.profile import SPRKKRProfile


def test_profile_uses_execution_endpoint_value():
    profile = SPRKKRProfile(
        executor="ssh",
        host="cluster",
        ssh_options=("-p", "2222"),
        ssh_zstd=True,
        remote_command=("ase2sprkkr", "remote"),
        psij_backend="slurm",
        psij_url="local://slurm",
    )

    assert profile.endpoint == ExecutorEndpoint(
        executor="ssh",
        host="cluster",
        ssh_options=("-p", "2222"),
        ssh_zstd=True,
        remote_command=("ase2sprkkr", "remote"),
        psij_backend="slurm",
        psij_url="local://slurm",
    )


def test_run_store_roundtrips_endpoint(tmp_path):
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store, executors=Executors())
    endpoint = ExecutorEndpoint(
        executor="ssh", host="cluster", ssh_options=("-v",), ssh_zstd=True,
    )
    record = manager.new_record(
        task="DOS",
        directory=tmp_path,
        dataset="case",
        endpoint=endpoint,
    )
    stored = store.save(record)

    loaded = store.get(stored.id)

    assert loaded.execution.endpoint == endpoint
    assert isinstance(loaded.execution.endpoint, ExecutorEndpoint)


def test_manager_reconstructs_local_executor_after_restart(tmp_path):
    store = RunStore(tmp_path / "runs")
    endpoint = ExecutorEndpoint()
    first = ExecutionManager(store, executors=Executors())
    record = first.new_record(
        task="DOS",
        directory=tmp_path,
        endpoint=endpoint,
    )
    store.save(record)

    restarted = ExecutionManager(store, executors=Executors())
    executor = restarted.executors.for_record(store.get(record.id))

    assert isinstance(executor, LocalExecutor)


def test_attach_reconstructs_persisted_endpoint_without_task_reader(tmp_path):
    class AttachedRunner:
        def __init__(self, run, executor):
            self.callback = None
            self.files = run.files
            self.executor = executor
            self.backend_id = run.execution.backend_id
            self.state = RunState.RUNNING
            self.result = None
            self.exception = None
            self.failure_phase = None
            self.failure_message = None
            self._done = Event()

        def start(self):
            pass

        def wait(self):
            self._done.wait()

        def cancel(self):
            self.state = RunState.CANCELLED
            self._done.set()

        def add_output_listener(self, callback):
            pass

        def add_output_error_listener(self, callback):
            pass

    class ReconstructedExecutor:
        def __init__(self):
            self.calls = []

        def attach(
            self,
            run,
            callback=None,
            output_listener=None,
            start=True,
        ):
            self.calls.append((run, callback, output_listener, start))
            runner = AttachedRunner(run, self)
            if start:
                runner.start()
            return runner

    store = RunStore(tmp_path / "runs")
    endpoint = ExecutorEndpoint(executor="ssh", host="cluster")
    first = ExecutionManager(store, executors=Executors())
    record = first.new_record(
        task="DOS",
        directory=tmp_path,
        endpoint=endpoint,
    )
    store.save(record)
    record = first.transition(record, state=RunState.RUNNING, execution=record.execution.submitted("42"))

    executor = ReconstructedExecutor()
    executors = Executors()
    executors.register(executor, endpoint)
    restarted = ExecutionManager(store, executors=executors)

    attached = restarted.attach(
        record.id,
        process_result=lambda outcome: outcome,
    )

    assert attached.executor is executor
    assert executor.calls == [(record, None, None, False)]

    attached.cancel()


def test_pool_reconstructs_and_reuses_ssh_endpoint():
    endpoint = ExecutorEndpoint(
        executor="ssh",
        host="cluster",
        ssh_options=("-p", "2222"),
        ssh_zstd=True,
        remote_command=("ase2sprkkr", "remote"),
        psij_backend="slurm",
        psij_url="local://slurm",
    )
    executors = Executors()

    first = executors.for_endpoint(endpoint)
    second = executors.for_endpoint(endpoint)

    assert first is second
    assert isinstance(first, RemoteEndpointExecutor)
    assert isinstance(first.transport, SshTransport)
    assert first.transport.host == "cluster"
    assert first.transport.options == ("-p", "2222")
    assert first.transport.zstd is True
    assert first.backend == "psij"
    assert first.backend_options == {"backend": "slurm", "url": "local://slurm"}


def test_default_remote_psij_endpoint_survives_restart():
    live = RemoteEndpointExecutor(
        SshTransport("cluster"),
        backend="psij",
    )

    endpoint = endpoint_for_executor(live)
    restored = Executors().for_endpoint(endpoint)

    assert endpoint.psij_backend == "slurm"
    assert isinstance(restored, RemoteEndpointExecutor)
    assert restored.backend == "psij"
    assert restored.backend_options == {"backend": "slurm", "url": None}


def test_psij_profile_polling_interval_survives_endpoint_roundtrip():
    from ase2sprkkr.execution.endpoint import create_executor_from_endpoint
    from ase2sprkkr.execution.psij import PsijExecutor
    from types import SimpleNamespace
    from unittest.mock import patch

    profile = SPRKKRProfile(
        executor="psij", psij_backend="slurm", psij_poll_interval=5,
    )
    assert profile.endpoint.psij_poll_interval == 5
    with patch("ase2sprkkr.execution.psij._load_psij") as load:
        load.return_value = SimpleNamespace(
            JobExecutor=SimpleNamespace(get_instance=lambda *a, **kw: object())
        )
        executor = create_executor_from_endpoint(profile.endpoint)
    assert isinstance(executor, PsijExecutor)
    assert executor.poll_interval == 5
