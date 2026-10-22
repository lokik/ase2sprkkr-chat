"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

from threading import Lock

from .run_model import FileAvailabilityState, FileManifest
from .manifest_merge import merge_snapshot, merge_terminal_files
from .transport import FILE_DELIVERY_ERRORS


class RunFiles:
    """Mutable working file state owned by one runner.

    Persisted RunRecord snapshots and historical file recovery are separate.
    """

    def __init__(self, manifest: FileManifest):
        if not isinstance(manifest, FileManifest):
            raise TypeError("RunFiles expects a FileManifest")
        self._manifest = manifest
        self._lock = Lock()

    @property
    def manifest(self) -> FileManifest:
        with self._lock:
            return self._manifest

    def replace(self, manifest: FileManifest) -> FileManifest:
        """Install a reserved or recovered authoritative file state."""
        if not isinstance(manifest, FileManifest):
            raise TypeError("RunFiles expects a FileManifest")
        with self._lock:
            self._manifest = manifest
            return manifest

    def merge(self, observed: FileManifest) -> FileManifest:
        """Reconcile outside the lock and atomically publish the latest state."""
        while True:
            with self._lock:
                previous = self._manifest
            merged = merge_snapshot(previous, observed)
            with self._lock:
                if self._manifest is previous:
                    self._manifest = merged
                    return merged

    def mark_availability(self, path, state, message=None, *, remote_path=None):
        """Update a declared file in working memory, retaining older snapshots."""
        with self._lock:
            self._manifest = self._manifest.with_file_availability(
                path, FileAvailabilityState(state).value, message,
                remote_path=remote_path,
            )
            return self._manifest

    async def collect_remote(
        self, frozen, handler, outcome, spec, *,
        required, materialization_error=None,
    ):
        """Collect terminal files into this job's mutable working state.

        The frozen membership is authoritative. Transfer and stream
        verification observations are installed independently so a later
        failure cannot discard a successful earlier delivery.
        """
        while True:
            with self._lock:
                previous = self._manifest
            merged = merge_terminal_files(previous, frozen)
            with self._lock:
                if self._manifest is previous:
                    self._manifest = merged
                    manifest = merged
                    break

        try:
            observed, transfer_error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except FILE_DELIVERY_ERRORS as exc:
            # A preflight failure cannot revoke independently delivered
            # files. Still verify FOLLOW streams for successful jobs.
            transfer_error = exc
        else:
            if observed is not None:
                merged = self.merge(observed)
                # An unrelated failed attempt may race with a successfully
                # verified download. Only suppress the transfer error if
                # every reported failure was superseded by intact delivery.
                if (transfer_error is not None and observed.failed_files
                        and not merged.failed_files):
                    transfer_error = None
            else:
                manifest = None

        if required and manifest is not None:
            try:
                observed, stream_error = await handler.verify_stream_availability(
                    self.manifest, spec,
                )
            except FILE_DELIVERY_ERRORS as exc:
                stream_error = exc
            else:
                if observed is not None:
                    self.merge(observed)
            if transfer_error is None:
                transfer_error = stream_error

        # Wildcard materialization is an independent backend-side failure
        # that precedes transfer/stream diagnostics.
        return (
            RuntimeError(materialization_error)
            if materialization_error else transfer_error
        )

