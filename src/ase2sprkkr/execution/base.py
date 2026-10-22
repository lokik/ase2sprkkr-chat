"""Backend-independent calculation execution contracts."""

from __future__ import annotations

import asyncio
import codecs
import os
import warnings
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, Tuple

from ..common.decorators import class_property
from .run_model import Execution, FileManifest, FileSyncPolicy

if TYPE_CHECKING:
    from .job_files import JobFiles


def file_path(value, directory=None, *, flush=False, required=False):
    """Resolve a named file or path, returning None for unnamed descriptors."""
    if value is None:
        return None
    if flush and hasattr(value, "flush"):
        value.flush()
    if not isinstance(value, (str, bytes, os.PathLike)):
        value = getattr(value, "name", value)
    if isinstance(value, int):
        if required:
            raise ValueError("A named file or path is required")
        return None
    try:
        path = os.fspath(value)
    except TypeError as exc:
        if required:
            raise ValueError("A named file or path is required") from exc
        return None
    if directory is not None and not os.path.isabs(path):
        path = os.path.join(directory, path)
    return os.path.abspath(path)


def _close_distinct(resources, *, suppress_errors=False):
    """Close each distinct resource once; keep closing after an error."""
    seen = set()
    first_error = None
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if close is None or getattr(resource, "closed", False):
            continue
        try:
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error


def close_output_files(*files):
    """Close distinct output handles, including merged stdout/stderr.

    Paths and absent streams need no closing. An error must not leave another
    open output handle behind; propagate the first error after cleanup.
    """
    _close_distinct(files)


def dispatch_cancellation(action):
    """Run a blocking backend cancellation off the calling/GUI thread.

    The returned Future exposes errors, while the normal monitor remains
    authoritative for the terminal run state.
    """
    future = Future()

    def worker():
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(action())
            except BaseException as exc:
                future.set_exception(exc)

    Thread(target=worker, daemon=True).start()
    return future


def start_monitor(runner, *, missing_backend=None):
    """Start a runner's monitor once, under its lifecycle lock."""
    with runner._lock:
        if runner._thread is not None:
            return
        if missing_backend is not None and runner._backend_id is None:
            raise RuntimeError(missing_backend)
        runner._thread = Thread(target=runner._thread_main, daemon=True)
        runner._thread.start()


def request_cancellation(runner, action):
    """Dispatch cancellation once and reuse its Future on repeat requests."""
    with runner._lock:
        if runner._cancel_future is None:
            runner._cancel_future = dispatch_cancellation(action)
        return runner._cancel_future


def finish_monitor(runner, *, cleanup=None):
    """Persist callback state and release owned resources before unblocking waiters."""
    try:
        if runner.callback is not None:
            runner.callback(runner.result)
    except BaseException as exc:
        runner._callback_exception = exc
    finally:
        try:
            if cleanup is not None:
                cleanup()
        finally:
            runner._done.set()


def wait_for_monitor(runner):
    """Await a started monitor; propagate execution errors before callback errors."""
    if runner._thread is None:
        if runner._state == RunState.NEW:
            raise RuntimeError("Run has not been submitted")
        raise RuntimeError("Attached run has not been started")
    runner._done.wait()
    if runner._exception is not None:
        raise runner._exception
    if runner._callback_exception is not None:
        raise runner._callback_exception
    return runner.result


class MpiNotFoundWarning(UserWarning):
    """MPI autodetection was requested, but no MPI launcher was found."""


class OutputObserverWarning(RuntimeWarning):
    """An output observer failed while the calculation itself keeps running."""


class SubmissionNotStarted(RuntimeError):
    """Submission failed before the backend could possibly accept the run."""


class OutputObservers:
    """Deliver raw output without allowing observers to affect execution."""

    def __init__(self, listener=None):
        self._listeners = [] if listener is None else [listener]
        self._error_listeners = []
        self._lock = Lock()
        self._decoders = {}

    def add(self, callback):
        with self._lock:
            self._listeners.append(callback)

    def add_error_listener(self, callback):
        with self._lock:
            self._error_listeners.append(callback)

    def emit_bytes(self, data, kind):
        """Preserve UTF-8 characters spanning successive byte chunks."""
        if not data:
            return
        with self._lock:
            decoder = self._decoders.setdefault(
                kind, codecs.getincrementaldecoder("utf8")(errors="replace")
            )
            text = decoder.decode(data)
        if text:
            self.emit(text, kind)

    def finish_bytes(self):
        """Flush incomplete UTF-8 sequences once at the end of observation."""
        with self._lock:
            pending = [
                (kind, decoder.decode(b"", final=True))
                for kind, decoder in self._decoders.items()
            ]
            self._decoders.clear()
        for kind, text in pending:
            if text:
                self.emit(text, kind)

    def emit(self, text, kind):
        with self._lock:
            listeners = tuple(self._listeners)
        for callback in listeners:
            try:
                callback(text, kind)
            except Exception as exc:
                with self._lock:
                    try:
                        self._listeners.remove(callback)
                    except ValueError:
                        pass
                    handlers = tuple(self._error_listeners)
                if not handlers:
                    warnings.warn(
                        f"Output observer {callback!r} failed: {exc}",
                        OutputObserverWarning,
                        stacklevel=4,
                    )
                    continue
                for handler in handlers:
                    try:
                        handler(callback, exc)
                    except Exception as handler_exc:
                        warnings.warn(
                            f"Output observer error handler {handler!r} failed: "
                            f"{handler_exc}",
                            OutputObserverWarning,
                            stacklevel=4,
                        )


class RunState(str, Enum):
    """Single authoritative lifecycle state of one calculation run."""

    NEW = "new"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    POSTPROCESSING = "postprocessing"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self):
        """Whether this run is finished, failed or cancelled (not UNKNOWN)."""
        return self in (RunState.FINISHED, RunState.FAILED, RunState.CANCELLED)


class SubmissionRecovery(str, Enum):
    """Conclusive absence of backend submission (None is uncertain)."""
    ABSENT = "absent"


class FailurePhase(str, Enum):
    """Phase in which a run entered :attr:`RunState.FAILED`."""

    SUBMISSION = "submission"
    LAUNCH = "launch"
    EXECUTION = "execution"
    FILES = "files"
    DOWNLOAD = "download"
    POSTPROCESSING = "postprocessing"


@dataclass
class ExecutionOutcome:
    """Raw execution facts available before task-specific interpretation."""

    output_file: Any = None
    input_file: Any = None
    error_file: Any = None
    return_code: Optional[int] = None

    @classmethod
    def from_spec(cls, spec: "ExecutionSpec", *, flush=False, strict_paths=False):
        """Snapshot named local paths before the specification's streams close.

        The specification may contain open file objects needed by the executor.
        The outcome holds their normalized paths for readers and persistence,
        not another mutable copy of the original stream objects.
        """
        def normalized(value):
            return file_path(
                value, spec.directory, flush=flush,
                required=strict_paths and value is not None,
            )

        return cls(
            output_file=normalized(spec.output_file),
            input_file=normalized(spec.input_file),
            error_file=normalized(spec.error_file),
        )


class ExecutionResources:
    """Closable resources whose ownership is transferred to a calculation run."""

    def __init__(self, *resources):
        self._resources = tuple(resources)
        self._closed = False
        self._lock = Lock()

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def close(self, *, suppress_errors=False):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            resources = self._resources
            self._resources = ()

        _close_distinct(resources, suppress_errors=suppress_errors)

@dataclass(frozen=True)
class RunStatus:
    """Backend status facts; the client owns durable RunRecord state."""

    state: RunState
    backend_id: Optional[str] = None
    execution: Optional[Execution] = None
    files: Optional[FileManifest] = None
    return_code: Optional[int] = None
    message: Optional[str] = None
    failure_phase: Optional[FailurePhase] = None

    def __post_init__(self):
        if not isinstance(self.state, RunState):
            object.__setattr__(self, "state", RunState(self.state))
        if self.failure_phase is not None and not isinstance(
            self.failure_phase, FailurePhase
        ):
            object.__setattr__(
                self, "failure_phase", FailurePhase(self.failure_phase)
            )


@dataclass(frozen=True)
class MpiSpec:
    """Portable MPI launch semantics independent of the execution host."""

    required: bool = False
    autodetect: bool = False
    launcher: Tuple[str, ...] = ()
    process_count: Optional[int] = None
    warning: bool = True

    def __post_init__(self):
        object.__setattr__(
            self, "launcher", tuple(os.fspath(part) for part in self.launcher)
        )
        if self.process_count is not None and self.process_count < 1:
            raise ValueError("process_count must be positive")
        if self.launcher and self.autodetect:
            raise ValueError("explicit MPI launcher cannot be combined with autodetection")
        if self.launcher and not self.required:
            raise ValueError("explicit MPI launcher must be required")
        if self.required and not self.autodetect and not self.launcher:
            raise ValueError("required MPI needs autodetection or an explicit launcher")
        if not self.requested and self.process_count is not None:
            raise ValueError("process_count requires MPI")

    @classmethod
    def from_value(cls, mpi, *, process_count=None, warning=True):
        if isinstance(mpi, cls):
            if process_count is not None or warning is not True:
                raise ValueError("MPI options cannot override an existing MpiSpec")
            return mpi
        if mpi is None or mpi is False:
            if process_count is not None:
                raise ValueError("process_count requires MPI")
            return cls(warning=warning)
        if isinstance(mpi, bool):
            return cls(
                required=True,
                autodetect=True,
                process_count=process_count,
                warning=warning,
            )
        if isinstance(mpi, int):
            if process_count is not None:
                raise ValueError("MPI process count was specified twice")
            return cls(
                required=True,
                autodetect=True,
                process_count=mpi,
                warning=warning,
            )
        if mpi == "auto":
            return cls(
                autodetect=True,
                process_count=process_count,
                warning=warning,
            )
        launcher = (mpi,) if isinstance(mpi, str) else tuple(os.fspath(part) for part in mpi)
        return cls(
            required=True,
            launcher=launcher,
            process_count=process_count,
            warning=warning,
        )

    @property
    def requested(self) -> bool:
        return self.required or self.autodetect or bool(self.launcher)

    @property
    def allows_serial_fallback(self) -> bool:
        return self.autodetect and not self.required


@dataclass(frozen=True)
class LaunchSpec:
    """Portable launch and staging parameters for one concrete run."""

    mpi: MpiSpec = MpiSpec()
    local_root: Optional[str] = None
    remote_root: Optional[str] = None
    input_sync: str = FileSyncPolicy.CHANGED.value
    output_sync: str = FileSyncPolicy.CHANGED.value
    cleanup_outputs: str = "both"

    def __post_init__(self):
        mpi = self.mpi
        if isinstance(mpi, dict):
            mpi = MpiSpec(**mpi)
        elif not isinstance(mpi, MpiSpec):
            raise TypeError("LaunchSpec.mpi must be an MpiSpec")
        object.__setattr__(self, "mpi", mpi)
        if self.local_root is not None:
            object.__setattr__(self, "local_root", os.fspath(self.local_root))
        if self.remote_root is not None:
            object.__setattr__(self, "remote_root", os.fspath(self.remote_root))
        try:
            input_sync = FileSyncPolicy(self.input_sync)
        except ValueError as exc:
            raise ValueError("Invalid input_sync file synchronization policy") from exc
        try:
            output_sync = FileSyncPolicy(self.output_sync)
        except ValueError as exc:
            raise ValueError("Invalid output_sync file synchronization policy") from exc
        object.__setattr__(self, "input_sync", input_sync.value)
        object.__setattr__(self, "output_sync", output_sync.value)
        from .file_availability import normalize_output_cleanup
        object.__setattr__(self, "cleanup_outputs", normalize_output_cleanup(self.cleanup_outputs))

    def __bool__(self):
        return self.mpi.requested

    @classmethod
    def from_mpi(cls, mpi):
        if isinstance(mpi, cls):
            return mpi
        return cls(mpi=MpiSpec.from_value(mpi))


def _pathlike(value):
    # Keep a TaskFile-like declaration's optional policy until FileManifest
    # has converted it to a file entry. Ordinary PathLike values stay strings.
    if isinstance(value, os.PathLike):
        if hasattr(value, "optional"):
            return value
        return os.fspath(value)
    return value


@dataclass(frozen=True)
class ExecutionSpec:
    """Command, files and launch intent handed to an execution backend."""

    argv: Sequence[str]
    input_file: Any
    output_file: Any
    directory: str
    stdin: Any = None
    error_file: Any = None
    input_files: Sequence[Any] = ()
    output_files: Sequence[Any] = ()
    cache_files: Sequence[Any] = ()
    mpi_argv: Optional[Sequence[str]] = None
    launch: LaunchSpec = LaunchSpec()
    # Each command has an independent input stream.
    mpi_stdin: Any = None

    def __post_init__(self):
        object.__setattr__(self, "directory", os.fspath(self.directory))
        object.__setattr__(self, "input_file", _pathlike(self.input_file))
        object.__setattr__(self, "output_file", _pathlike(self.output_file))
        object.__setattr__(self, "stdin", _pathlike(self.stdin))
        object.__setattr__(self, "mpi_stdin", _pathlike(self.mpi_stdin))
        object.__setattr__(self, "error_file", _pathlike(self.error_file))
        object.__setattr__(
            self, "input_files", tuple(_pathlike(value) for value in self.input_files),
        )
        object.__setattr__(
            self, "output_files", tuple(_pathlike(value) for value in self.output_files),
        )
        object.__setattr__(
            self, "cache_files", tuple(_pathlike(value) for value in self.cache_files),
        )
        object.__setattr__(self, "argv", tuple(os.fspath(value) for value in self.argv))
        if self.mpi_argv is not None:
            object.__setattr__(
                self,
                "mpi_argv",
                tuple(os.fspath(value) for value in self.mpi_argv),
            )

    def command_request(self, transform=list):
        """Serialize common command and launch fields for execution services.

        The caller supplies the mapping for command arguments; for remote runs
        this resolves already-staged local file names to target paths.
        """
        return {
            "argv": transform(self.argv),
            "mpi_argv": None if self.mpi_argv is None else transform(self.mpi_argv),
            "launch": asdict(self.launch),
        }


class Runner(Protocol):
    """One concrete calculation run."""

    result: Any

    job_files: "JobFiles"

    @property
    def state(self) -> RunState: ...

    @property
    def backend_id(self) -> Optional[str]: ...

    @property
    def exception(self) -> Optional[BaseException]: ...

    def submit(self) -> None: ...
    def wait(self) -> Any: ...
    def run(self) -> Any: ...
    def cancel(self) -> None: ...
    def add_output_listener(self, callback) -> None: ...
    def add_output_error_listener(self, callback) -> None: ...


class Executor:
    """Execution backend. It never knows task readers or TaskResult types."""

    name = None

    @class_property
    def supports_result_file_retrieval(cls):
        return cls.fetch_result_files is not Executor.fetch_result_files

    def preflight_result_files(self, run, files):
        """Validate the entire historical fetch request without transferring.

        Executors with a remote file mapping override this hook. Restoration
        still validates each destination immediately before publication.
        """

    def abandoned_preparation_is_safe(self):
        """Return true only when abandoned preparation cannot keep writing files."""
        return False

    def fetch_result_files(self, run, files, *, destinations=None):
        """Retrieve *files*, optionally staging each canonical local path.

        destinations maps requested local paths to alternate local targets;
        the requested paths still select the remote sources. The caller owns
        verification and publication of staged downloads.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot retrieve persistent result files"
        )

    def acknowledge_submission(self, run):
        """No temporary submission receipt for this executor."""

    def acknowledge(self, run):
        """No terminal receipt for this executor."""

    def reservation_context(self, spec: ExecutionSpec, files):
        """Prepare canonical run paths before atomic submission reservation."""
        return files, None

    def submission_directory(self, spec: ExecutionSpec) -> str:
        return os.path.abspath(os.fspath(spec.directory))

    def resolve_spec(self, spec: ExecutionSpec) -> ExecutionSpec:
        """Choose a concrete command before reserving/staging its files."""
        from .local_process import materialize_local_spec
        return materialize_local_spec(spec)

    def status(self, *runs) -> Tuple[RunStatus, ...]:
        if not runs:
            return ()
        raise NotImplementedError(
            f"{type(self).__name__} cannot query persistent run status"
        )

    def recover_submission(
        self, run_id, submission, *, submitted_at=None
    ) -> Optional[RunStatus]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot recover an uncertain submission"
        )

    def create_project_run(
        self, spec: ExecutionSpec, *, run_id: str, callback=None
    ) -> Runner:
        return self.create_run(spec, callback=callback)

    def create_run(self, spec: ExecutionSpec, callback=None) -> Runner:
        raise NotImplementedError

    def attach(
        self,
        run,
        callback=None,
        output_listener=None,
        start=True,
    ) -> Runner:
        raise NotImplementedError(
            f"{type(self).__name__} cannot attach to an existing run"
        )
