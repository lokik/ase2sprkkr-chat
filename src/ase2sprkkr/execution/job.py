"""User-facing persistent job handle.

A Job is an identity and a manager reference, not a second mutable RunRecord
or an active execution observer. All operations load current durable state.
"""
from __future__ import annotations

from .file_history import file_is_current


class Job:
    def __init__(self, manager, run_id: str):
        self.manager = manager
        self.id = str(run_id)

    @property
    def record(self):
        return self.manager.store.get(self.id)

    @property
    def files(self):
        return self.record.files

    @property
    def state(self):
        return self.record.state

    @property
    def project_id(self):
        return self.record.project_id

    def refresh(self):
        """Read the latest durable record, not a cached snapshot."""
        return self.record

    def restore_files(self):
        """Restore terminal historical files without resubmitting the job."""
        return self.manager.retrieve_result_files(self.id)

    def input_is_current(self):
        files = self.files
        return file_is_current(files, files.input_file, files.input_versions)

    def output_is_current(self):
        files = self.files
        return file_is_current(files, files.output_file, files.output_versions)

    def load_result(self):
        from ..outputs.project_results import load_project_result
        return load_project_result(self.manager, self.id)

    def attach(self, *, executor=None, callback=None, output_listener=None):
        from ..outputs.project_results import attach_project_run
        return attach_project_run(
            self.manager, self.id, executor=executor, callback=callback,
            output_listener=output_listener,
        )

    def retry_postprocessing(self):
        from ..outputs.project_results import retry_project_postprocessing
        return retry_project_postprocessing(self.manager, self.id)
