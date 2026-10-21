"""Remote execution with client-owned lifecycle persistence."""

from __future__ import annotations

import os
import posixpath
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from .base import (
    ExecutionSpec,
    Executor,
    FailurePhase,
    LaunchSpec,
    RunState,
)
from .remote import FileSyncPolicy, RemoteRunner, _run_sync
from .remote_file_handler import RemoteFileHandler
from .target_connection import SSHConnection
from .supervisor_monitor import backend_status_from_facts
from .run_model import FileManifest, WorkingDirectory
from .endpoint import endpoint_for_executor


_REMOTE_BACKENDS = {"local", "psij"}


def _relative_local_path(path, local_root):
    """Resolve a local path and enforce the run's mapped root boundary."""
    path = Path(path).expanduser().resolve()
    root = Path(local_root).expanduser().resolve()
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Local path {path} is outside mapped root {root}; "
            "specify local_root/remote_root explicitly"
        ) from exc


class RemoteEndpointExecutor(Executor):
    """Reusable remote endpoint with client-owned lifecycle state."""

    name = "remote"

    def __init__(
        self,
        transport,
        *,
        backend="local",
        backend_options=None,
        command=("ase2sprkkr", "remote"),
    ):
        if backend not in _REMOTE_BACKENDS:
            raise ValueError(f"Unsupported remote executor: {backend!r}")
        self.transport = transport
        self.file_handler = RemoteFileHandler(transport)
        self.backend = backend
        self.backend_options = dict(backend_options or {})
        self.command = tuple(command)
        self.computer = getattr(transport, "host", None)
        self.connection = SSHConnection(
            transport, command=self.command, backend=backend,
            backend_options=self.backend_options,
        )

    async def resolve_mapping(self, launch):
        """Resolve a run's workspace once for transfer and persistence."""
        if not isinstance(launch, LaunchSpec):
            launch = LaunchSpec(**dict(launch))
        local_root = Path(launch.local_root or Path.home()).expanduser().resolve()
        root = launch.remote_root
        if root is None or root == "~":
            remote_root = await self.connection.remote_home()
        elif root.startswith("~/"):
            remote_root = await self.connection.resolve_directory(root)
        elif posixpath.isabs(root):
            remote_root = posixpath.normpath(root)
        else:
            remote_root = posixpath.join(await self.connection.remote_home(), root)
        return FileManifest(
            local_root=os.fspath(local_root),
            remote_root=posixpath.normpath(remote_root),
            input_sync=launch.input_sync,
            output_sync=launch.output_sync,
        )

    def _validate_run(self, run):
        if run.execution.endpoint != endpoint_for_executor(self):
            raise ValueError("Run belongs to a different remote endpoint")
        if not run.execution.backend_id or not run.execution.directory.remote:
            raise ValueError("Remote run lacks backend identity or working directory")

    def reservation_context(self, spec, files):
        """Resolve the remote targets into the one authoritative file manifest."""
        context = _run_sync(self.resolve_mapping(spec.launch))
        prepared = files.with_remote(
            local_root=context.local_root, remote_root=context.remote_root,
            input_sync=context.input_sync, output_sync=context.output_sync,
        )
        return prepared, prepared.map_path(spec.directory)

    def submission_directory(self, spec):
        launch = spec.launch
        relative = _relative_local_path(
            spec.directory, launch.local_root or Path.home(),
        )

        root = launch.remote_root
        if root is None or root == "~":
            remote_root = "~"
        elif root.startswith("~/"):
            remote_root = posixpath.normpath(root)
        elif posixpath.isabs(root):
            remote_root = posixpath.normpath(root)
        else:
            remote_root = posixpath.normpath(posixpath.join("~", root))
        if not relative.parts:
            return remote_root
        return posixpath.join(remote_root, *PurePosixPath(*relative.parts).parts)

    def acknowledge_submission(self, run):
        """Release temporary submission identity after client persistence."""
        return _run_sync(self.connection.acknowledge_submission(run.id))

    def acknowledge(self, run):
        """Release a delivered receipt only after durable RunRecord completion."""
        self._validate_run(run)
        return _run_sync(self.connection.acknowledge(run.execution.backend_id))

    def status(self, *runs):
        for run in runs:
            self._validate_run(run)
        if not runs:
            return ()

        async def query():
            result = []
            for run in runs:
                execution = run.execution
                state = await self.connection.status(execution.backend_id, run.files)
                result.append(backend_status_from_facts(
                    state, execution.backend_id, needs_delivery=True,
                ))
            return tuple(result)

        return _run_sync(query())

    async def _recover_submission(self, run_id, submission, *, submitted_at=None):
        directory = await self.connection.resolve_directory(submission.execution.directory.target)
        state = await self.connection.recover(
            run_id, directory, submitted_at=submitted_at,
        )
        if state is None:
            return None
        if state.get("unsupported"):
            raise NotImplementedError(
                state.get("error") or "Remote recovery unsupported"
            )
        remote_id = state.get("remote_id")
        if not remote_id:
            raise RuntimeError("Recovered remote submission has no remote_id")

        execution = submission.execution.submitted(remote_id)
        target_directory = state.get("directory") or directory
        execution = replace(
            execution, directory=WorkingDirectory(
                execution.directory.local, target_directory,
            ),
        )
        files = submission.files.with_remote_stream_paths(
            stdout=state.get("output_file"), stderr=state.get("error_file"),
        )
        return backend_status_from_facts(
            state, str(remote_id), needs_delivery=True,
            execution=execution, files=files,
        )

    def recover_submission(self, run_id, submission, *, submitted_at=None):
        return _run_sync(
            self._recover_submission(
                run_id, submission, submitted_at=submitted_at
            )
        )

    def create_run(self, spec, callback=None):
        return RemoteRunner(self, spec, callback=callback)

    def create_project_run(self, spec, *, run_id, callback=None):
        return RemoteRunner(self, spec, callback=callback, run_id=run_id)

    async def _fetch_result_files(self, run, files, *, destinations=None):
        self._validate_run(run)
        return await self.file_handler.restore(
            run, files, destinations=destinations,
        )

    def fetch_result_files(self, run, files, *, destinations=None):
        return _run_sync(self._fetch_result_files(
            run, files, destinations=destinations,
        ))

    async def _attach(self, run, callback=None, output_listener=None, start=True):
        self._validate_run(run)
        execution = run.execution
        if run.files.remote_root is None:
            raise ValueError("Remote run has no persisted file mapping")
        status = await self.connection.status(execution.backend_id, run.files)
        local_directory = Path(execution.directory.local)
        local_directory.mkdir(parents=True, exist_ok=True)
        input_local = run.files.input_file
        output_local = run.files.output_file
        error_local = None if run.files.stderr is None else run.files.stderr.local_path

        with ExitStack() as streams:
            output_file = (None if output_local is None else
                           streams.enter_context(open(output_local, "ab")))
            if error_local is None:
                error_file = None
            elif error_local == output_local:
                error_file = output_file
            else:
                error_file = streams.enter_context(open(error_local, "ab"))
            spec = ExecutionSpec(
                argv=(),
                input_file=None if input_local is None else os.fspath(input_local),
                output_file=output_file,
                error_file=error_file,
                directory=os.fspath(local_directory),
                launch=LaunchSpec(
                    local_root=run.files.local_root,
                    remote_root=run.files.remote_root,
                    input_sync=run.files.input_sync,
                    output_sync=run.files.output_sync,
                ),
            )
            runner = RemoteRunner(
                self, spec, callback=callback, output_listener=output_listener,
                backend_id=execution.backend_id,
                remote_directory=execution.directory.remote,
                files=run.files,
                state=backend_status_from_facts(
                    status, execution.backend_id, needs_delivery=True,
                ).state,
            )
            runner._return_code = status.get("return_code")
            runner.result.return_code = runner._return_code
            phase = status.get("failure_phase")
            runner._failure_phase = None if phase is None else FailurePhase(phase)
            runner._failure_message = status.get("error")
            if start:
                runner.start()
            streams.pop_all()
            return runner

    def attach(self, run, callback=None, output_listener=None, start=True):
        return _run_sync(self._attach(
            run, callback=callback, output_listener=output_listener, start=start,
        ))
