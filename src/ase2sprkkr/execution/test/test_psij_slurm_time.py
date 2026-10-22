import re

from ase2sprkkr.execution.psij import _slurm_time


def test_slurm_accounting_time_drops_timezone_offset():
    value = _slurm_time("2026-10-13T08:00:00+02:00")

    assert re.fullmatch(r"2026-10-13T\d{2}:\d{2}:\d{2}", value)
    assert "+" not in value


def test_slurm_new_output_requires_newer_nonempty_file(tmp_path):
    from datetime import datetime, timezone
    from ase2sprkkr.execution.psij import _slurm_new_output_since
    out = tmp_path / "job.out"
    out.write_text("old")
    submission = datetime.now(timezone.utc).isoformat()
    assert not _slurm_new_output_since((out,), submission)
    out.write_text("fresh")
    import os
    from datetime import timedelta
    newer = datetime.now(timezone.utc) + timedelta(seconds=3)
    os.utime(out, (newer.timestamp(), newer.timestamp()))
    assert _slurm_new_output_since((out,), submission)
    out.write_text("")
    assert not _slurm_new_output_since((out,), submission)
    assert not _slurm_new_output_since((out,), "invalid")
