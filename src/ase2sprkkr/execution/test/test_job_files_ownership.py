"""Stable ownership boundary for run file operations."""

from ase2sprkkr.execution import JobFiles
from ase2sprkkr.execution.job_files import JobFiles as ModuleJobFiles
from ase2sprkkr.execution.manager import RunRecord


def test_job_files_public_identity():
    assert JobFiles is ModuleJobFiles
    assert JobFiles.__module__ == "ase2sprkkr.execution.job_files"
    assert isinstance(RunRecord.job_files, property)
