"""SPR-KKR output readers and project-result reconstruction helpers."""

from .file_provenance import (
    capture_result_file_manifest,
    capture_result_file_versions,
    result_file_version,
)
from .output_detection import looks_like_scf_output
from .output_only_result import OutputOnlyResult
from .project_results import (
    ProjectResultUnavailableError,
    UnrecoverableProjectResultError,
    attach_project_run,
    load_project_result,
    retry_project_postprocessing,
)
from .task_result import TaskResult

__all__ = [
    "OutputOnlyResult",
    "ProjectResultUnavailableError",
    "UnrecoverableProjectResultError",
    "TaskResult",
    "attach_project_run",
    "capture_result_file_manifest",
    "capture_result_file_versions",
    "load_project_result",
    "looks_like_scf_output",
    "retry_project_postprocessing",
    "result_file_version",
]
