"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
from dataclasses import replace

from .file_history import (
    capture_result_files, file_is_current, missing_result_files, sync_result_files,
)


class JobFiles:
    """Operate on one run's manifest without duplicating its persistent state."""

    def __init__(self, record):
        self.record = record

    @property
    def manifest(self):
        return self.record.files

    def input_file_is_current(self):
        files = self.manifest
        return file_is_current(files, files.input_file, files.input_versions)

    def output_file_is_current(self):
        files = self.manifest
        return file_is_current(files, files.output_file, files.output_versions)

    def sync_inputs(self):
        sync_result_files(self.manifest.input_files)

    def capture(self, manifest=None):
        return capture_result_files(self.manifest if manifest is None else manifest)

    def missing_files(self):
        return missing_result_files(self.manifest)

    @staticmethod
    def mark_pending_failed(manifest, message):
        return manifest._map_files(
            lambda item: replace(
                item, availability_state="failed", availability_message=message,
            ) if item.availability_state in (None, "pending") else item
        )

    @staticmethod
    def merge_terminal_files(previous, received):
        """Preserve completed local deliveries in frozen terminal facts."""
        if previous is None:
            return received
        delivered = {
            (entry.local_path, entry.remote_path): entry
            for entry in previous.files
        }

        def preserve(entry):
            if entry is None or entry.availability_state in ("failed", "skipped"):
                return entry
            prior = delivered.get((entry.local_path, entry.remote_path))
            if prior is None or prior.availability_state not in ("available", "skipped"):
                return entry
            if (prior.availability_state == "available"
                    and (entry.local_path is None
                         or not os.path.isfile(entry.local_path))):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return received._map_files(preserve)


    def retry_failed_downloads(self):
        return self.manifest._map_files(
            lambda item: replace(
                item, availability_state="pending", availability_message=None,
            ) if item.availability_state == "failed" else item
        )
