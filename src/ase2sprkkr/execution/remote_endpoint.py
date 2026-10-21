"""Remote execution with client-owned lifecycle persistence."""

from __future__ import annotations

import os
import posixpath
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from .base import (
    ExecutionSpec,
    file_path,
    start_monitor,
    Executor,
    FailurePhase,
    LaunchSpec,
    RunState,
    SubmissionNotStarted,
)
from .remote import FileSyncPolicy, RemoteRunner, _run_sync
from .remote_file_handler import RemoteFileHandler
from ..common.file_versions import file_digest
from .target_connection import SSHConnection
from .supervisor_monitor import backend_status_from_facts
from .run_model import WorkingDirectory
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


class _RemoteRun(RemoteRunner):
    """One remote run whose persistent lifecycle is owned by the client."""

    def __init__(self, *args, run_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_id = run_id
        self._prepared = None

    async def _complete_terminal(self, facts, backend_state):
        """Finalize delivery once, including integrity of streamed channels."""
        await super()._complete_terminal(facts, backend_state)
        if self._state != RunState.FINISHED or self.files is None:
            return

        # A broken follow connection can leave either channel incomplete even
        # when the backend finished successfully. Verify both physical files.
        checked = set()
        for kind, entry in (("stdout", self.files.stdout),
                            ("stderr", self.files.stderr)):
            if entry is None or entry.local_path is None or entry.remote_path is None:
                continue
            local = entry.local_path
            if local in checked:
                continue  # Merged stdout/stderr is one physical file.
            checked.add(local)
            try:
                signature = await self.executor.endpoint.transport.file_signature(entry.remote_path)
                if signature is None:
                    raise FileNotFoundError(
                        f"Remote {kind} output is missing: {entry.remote_path}"
                    )
                # FOLLOW writes through a separate descriptor. Flush the
                # runner's buffered handles before verifying the local copy.
                for stream in (self.spec.output_file, self.spec.error_file):
                    if stream is not None and not stream.closed:
                        stream.flush()
                stat = os.stat(local)
                if stat.st_size != signature[0]:
                    raise RuntimeError(
                        f"Streamed local {kind} size does not match the "
                        f"authoritative remote output: {local}"
                    )
                remote_digest = await self.executor.endpoint.transport.file_digest(entry.remote_path)
                if remote_digest is None or file_digest(local) != remote_digest:
                    raise RuntimeError(
                        f"Streamed local {kind} contents differ from remote output: {local}"
                    )
                if await self.executor.endpoint.transport.file_signature(entry.remote_path) != signature:
                    raise RuntimeError(f"Remote {kind} output changed during verification")

                os.utime(local, ns=(stat.st_atime_ns, signature[1]))
            except Exception as exc:
                self._state = RunState.FAILED
                self._failure_phase = FailurePhase.DOWNLOAD
                self._failure_message = str(exc)
                self._files = self._files.with_file_availability(local, "failed", str(exc))
                raise

    async def _prepare_async(self):
        try:
            remote_directory = await self.executor.map_path(self.spec.directory)
            await self.executor.endpoint.transport.run(("mkdir", "-p", "--", remote_directory))
            manifest = await self.file_handler.prepare(
                self.executor, self.spec,
            )
            # The runner owns backend identity, command construction and the
            # submission protocol. FileHandler knows only file locations.
            mapped = {entry.local_path: entry.remote_path
                      for entry in (*manifest.inputs, *(cache.file for cache in manifest.caches))
                      if entry.local_path is not None}
            def argv(args):
                return [mapped.get(file_path(os.fspath(arg), self.spec.directory),
                                   os.fspath(arg)) for arg in args]
            request = {
                "id": self._run_id,
                "executor": self.executor.endpoint.backend,
                "executor_options": self.executor.endpoint.backend_options,
                "directory": remote_directory,
                **self.spec.command_request(argv),
                "input_file": None if self.spec.input_file is None else
                    mapped.get(file_path(self.spec.input_file, self.spec.directory)),
                "stdin": None if self.spec.stdin is None else
                    mapped.get(file_path(self.spec.stdin, self.spec.directory)),
                "output_file": (None if manifest.stdout is None
                                else manifest.stdout.remote_path),
                "error_file": (None if manifest.stderr is None
                               else manifest.stderr.remote_path),
                "files": asdict(manifest),
            }
            return request, manifest
        except Exception as exc:
            # No calculation has been launched yet. Persisted preflight and
            # direct submission must classify this known failure identically.
            self._failure_phase = FailurePhase.SUBMISSION
            self._failure_message = str(exc)
            raise SubmissionNotStarted(str(exc)) from exc

    def prepare_submission(self):
        """Complete all staging before the client commits SUBMITTING metadata."""
        with self._lock:
            if self._state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._state.value}")
            if self._prepared is not None:
                return
            request, manifest = _run_sync(self._prepare_async())
            self._prepared = (request, manifest)
            self._remote_directory = request["directory"]
            self._files = manifest

    async def _submit_async(self):
        if self._prepared is None:
            request, manifest = await self._prepare_async()
        else:
            request, manifest = self._prepared

        request["detached"] = self._run_id is not None
        status = await self.executor.endpoint.connection.submit(request)
        remote_id = status.get("remote_id")
        if not remote_id:
            raise RuntimeError("Remote submit returned no remote_id")
        if status.get("files") is not None:
            from .run_model import FileManifest
            manifest = FileManifest(**status["files"])
        return remote_id, manifest, status, request

    def submit(self):
        with self._lock:
            if self._state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._state.value}")
            remote_id, manifest, status, request = _run_sync(self._submit_async())
            self._state = RunState(status["state"])
            self._return_code = status.get("return_code")
            self.result.return_code = self._return_code
            phase = status.get("failure_phase")
            self._failure_phase = None if phase is None else FailurePhase(phase)
            self._failure_message = status.get("error")
            self._backend_id = str(remote_id)
            self._remote_directory = request["directory"]
            self._files = manifest
        start_monitor(self, missing_backend="Run has no backend reference")


class _RemoteRunContext:
    """Per-run filesystem and synchronization policy."""

    def __init__(self, endpoint, launch):
        self.endpoint = endpoint
        self.local_root = Path(launch.local_root or Path.home()).expanduser().resolve()
        self.remote_root = (
            None if launch.remote_root is None else os.fspath(launch.remote_root)
        )
        self.input_files = FileSyncPolicy(launch.input_sync)
        self.output_files = FileSyncPolicy(launch.output_sync)

    async def _resolved_remote_root(self):
        root = self.remote_root
        if root is not None and posixpath.isabs(root):
            return posixpath.normpath(root)
        home = await self.endpoint.connection.remote_home()
        if root is None or root == "~":
            return home
        if root.startswith("~/"):
            return posixpath.join(home, root[2:])
        return posixpath.join(home, root)

    async def map_path(self, path):
        relative = _relative_local_path(path, self.local_root)
        remote_root = await self._resolved_remote_root()
        if not relative.parts:
            return remote_root
        return posixpath.join(remote_root, *relative.parts)

    async def unmap_path(self, path):
        remote_root = await self._resolved_remote_root()
        path = posixpath.normpath(os.fspath(path))
        if not posixpath.isabs(path):
            raise ValueError(f"Remote path must be absolute: {path}")
        try:
            common = posixpath.commonpath((remote_root, path))
        except ValueError as exc:
            raise ValueError(
                f"Remote path {path} is outside mapped root {remote_root}"
            ) from exc
        if common != remote_root:
            raise ValueError(f"Remote path {path} is outside mapped root {remote_root}")
        relative = posixpath.relpath(path, remote_root)
        if relative == ".":
            return self.local_root
        local = self.local_root.joinpath(*PurePosixPath(relative).parts)
        _relative_local_path(local, self.local_root)
        return local

    def create_run(self, spec, callback=None, *, run_id=None):
        return _RemoteRun(self, spec, callback=callback, run_id=run_id)

    async def attach(self, run, callback=None, output_listener=None, start=True):
        execution = run.execution
        if execution.backend_id is None or execution.directory.remote is None:
            raise ValueError("Remote run lacks backend ID or target directory")
        status = await self.endpoint.connection.status(execution.backend_id, run.files)
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
                    local_root=os.fspath(self.local_root),
                    remote_root=self.remote_root,
                    input_sync=self.input_files.value,
                    output_sync=self.output_files.value,
                ),
            )
            phase = status.get("failure_phase")
            runner = _RemoteRun(
                self,
                spec,
                callback=callback,
                output_listener=output_listener,
                start=False,
                backend_id=execution.backend_id,
                remote_directory=execution.directory.remote,
                files=run.files,
                state=backend_status_from_facts(
                    status, execution.backend_id, needs_delivery=True,
                ).state,
            )
            runner._return_code = status.get("return_code")
            runner.result.return_code = runner._return_code
            runner._failure_phase = None if phase is None else FailurePhase(phase)
            runner._failure_message = status.get("error")
            if start:
                runner.start()
            streams.pop_all()  # Runner owns the streams after successful attach.
            return runner


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

    @staticmethod
    def _context(endpoint, launch):
        if not isinstance(launch, LaunchSpec):
            launch = LaunchSpec(**dict(launch))
        return _RemoteRunContext(endpoint, launch)

    def _validate_run(self, run):
        if run.execution.endpoint != endpoint_for_executor(self):
            raise ValueError("Run belongs to a different remote endpoint")
        if not run.execution.backend_id or not run.execution.directory.remote:
            raise ValueError("Remote run lacks backend identity or working directory")

    @staticmethod
    def _launch_from_manifest(files):
        mapping = files.mapping
        if mapping is None:
            raise ValueError("Remote run has no persisted file mapping")
        return LaunchSpec(
            local_root=mapping.local_root, remote_root=mapping.remote_root,
            input_sync=mapping.input_sync, output_sync=mapping.output_sync,
        )

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
        return self._context(self, spec.launch).create_run(spec, callback=callback)

    def create_project_run(self, spec, *, run_id, callback=None):
        return self._context(self, spec.launch).create_run(
            spec, callback=callback, run_id=run_id
        )

    async def _fetch_result_files(self, run, files, *, destinations=None):
        self._validate_run(run)
        context = self._context(self, self._launch_from_manifest(run.files))
        return await self.file_handler.restore(
            run, files, context, destinations=destinations,
        )

    def fetch_result_files(self, run, files, *, destinations=None):
        return _run_sync(self._fetch_result_files(
            run, files, destinations=destinations,
        ))

    async def _attach(self, run, callback=None, output_listener=None, start=True):
        self._validate_run(run)
        launch = self._launch_from_manifest(run.files)
        return await self._context(self, launch).attach(
            run, callback=callback, output_listener=output_listener, start=start,
        )

    def attach(self, run, callback=None, output_listener=None, start=True):
        return _run_sync(self._attach(
            run, callback=callback, output_listener=output_listener, start=start,
        ))
