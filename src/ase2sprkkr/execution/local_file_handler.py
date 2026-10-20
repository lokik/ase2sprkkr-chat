"""Local run file operations shared by process, supervisor and PSI/J backends.

This service is stateless: paths, manifests and output stream ownership all
belong to the caller. A single instance is safe to reuse across different runs.
"""
from __future__ import annotations

import os

from .base import file_path
from .file_availability import (
    clear_declared_outputs, discover_local_files, validate_output_collisions,
)


class LocalFileHandler:
    """Prepare one target filesystem and discover files after execution."""

    @staticmethod
    def prepare_target(spec, *, output_file=None, error_file=None,
                       preserve_open_streams=False):
        """Protect inputs and restart caches, then clear declared outputs.

        If streams are already open, do not unlink those descriptors' named
        destinations: doing so would redirect output into orphaned inodes.
        Otherwise primary streams belong to target-side prelaunch cleanup.
        """
        directory = spec.directory

        def paths(values):
            return tuple(
                path for value in values
                if (path := file_path(value, directory)) is not None
            )

        protected = paths((
            spec.input_file, spec.stdin, *spec.input_files, *spec.cache_files,
        ))
        streams = paths((
            spec.output_file if output_file is None else output_file,
            spec.error_file if error_file is None else error_file,
        ))
        declared = paths(spec.output_files)
        validate_output_collisions((*declared, *streams), protected=protected)
        if preserve_open_streams:
            clear_declared_outputs(declared, protected=(*protected, *streams))
        else:
            clear_declared_outputs((*declared, *streams), protected=protected)

    @staticmethod
    def collect(manifest, *, include_streams=True):
        return discover_local_files(manifest, include_streams=include_streams)

    @staticmethod
    def restore(manifest, paths):
        """Local data has no other source to fetch from.

        Keep missing files missing; never synthesize a new copy in place
        of an already historical result.
        """
        return manifest
