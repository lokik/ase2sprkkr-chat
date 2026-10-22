"""User-facing handle for a stored calculation, independent of a live runner."""

from __future__ import annotations

from .file_history import file_is_current


class Job:
    """Operate on a persisted run through the existing execution manager."""

    def __init__(self, manager, run_id: str):
        self._manager = manager
        self.id = run_id

    @property
    def record(self):
        """Always read the latest durable run snapshot."""
        return self._manager.store.get(self.id)

    @property
    def files(self):
        return self.record.files

    @property
    def state(self):
        return self.record.state

    def refresh(self):
        return self.record

    def restore_files(self):
        """Restore the missing historical files without resubmitting."""
        return self._manager.retrieve_result_files(self.id)

    def input_is_current(self):
        """Check the input against its captured historical version."""
        manifest = self.files
        return (manifest.captured_at is not None and file_is_current(
            manifest, manifest.input_file, manifest.input_versions,
        ))

    def output_is_current(self):
        """Never certify an uncaptured file as this job's output."""
        manifest = self.files
        return (manifest.captured_at is not None and file_is_current(
            manifest, manifest.output_file, manifest.output_versions,
        ))

    def load_result(self):
        from ..outputs.project_results import load_project_result
        return load_project_result(self._manager, self.id)

    def attach(self, *, executor=None, callback=None, output_listener=None):
        from ..outputs.project_results import attach_project_run
        return attach_project_run(
            self._manager, self.id, executor=executor,
            callback=callback, output_listener=output_listener,
        )

    def retry_postprocessing(self):
        from ..outputs.project_results import retry_project_postprocessing
        return retry_project_postprocessing(self._manager, self.id)
