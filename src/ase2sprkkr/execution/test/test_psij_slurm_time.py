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


def test_unknown_slurm_needs_new_output_before_inference(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    from ase2sprkkr.execution import psij
    from ase2sprkkr.execution.base import RunState
    import os

    monkeypatch.setattr(psij, "_slurm_details", lambda job: {})
    monkeypatch.setattr(psij, "_scheduler_output", lambda argv: "")
    result = tmp_path / "job.out"
    result.write_text("old")
    submitted_at = datetime.now(timezone.utc).isoformat()

    status = psij._slurm_status_native(
        "42", submitted_at=submitted_at, output_paths=(str(result),))
    assert status.state == RunState.UNKNOWN

    future = datetime.now(timezone.utc) + timedelta(seconds=2)
    os.utime(result, (future.timestamp(), future.timestamp()))
    status = psij._slurm_status_native(
        "42", submitted_at=submitted_at, output_paths=(str(result),))
    assert status.state == RunState.FINISHED

    status = psij._slurm_status_native("42", output_paths=(str(result),))
    assert status.state == RunState.UNKNOWN


def test_unknown_slurm_does_not_infer_finished_while_job_is_queued(tmp_path, monkeypatch):
    from datetime import datetime, timezone, timedelta
    import os
    from ase2sprkkr.execution import psij
    from ase2sprkkr.execution.base import RunState

    output = tmp_path / "run.out"
    output.write_text("running")
    submitted = datetime.now(timezone.utc)
    fresh = submitted + timedelta(seconds=2)
    os.utime(output, (fresh.timestamp(), fresh.timestamp()))
    monkeypatch.setattr(psij, "_slurm_details", lambda _: {})

    def scheduler(argv):
        return "42" if argv[0] == "squeue" else ""

    monkeypatch.setattr(psij, "_scheduler_output", scheduler)
    status = psij._slurm_status_native(
        "42", submitted_at=submitted.isoformat(), output_paths=(str(output),)
    )
    assert status.state == RunState.UNKNOWN


def test_unknown_slurm_keeps_unknown_for_unrecognized_scheduler_state(
    tmp_path, monkeypatch,
):
    from datetime import datetime, timezone, timedelta
    import os
    from ase2sprkkr.execution import psij
    from ase2sprkkr.execution.base import RunState

    output = tmp_path / "run.out"
    output.write_text("fresh")
    submitted = datetime.now(timezone.utc)
    fresh = submitted + timedelta(seconds=2)
    os.utime(output, (fresh.timestamp(), fresh.timestamp()))
    monkeypatch.setattr(psij, "_slurm_details",
                        lambda _: {"JobState": "NOT_RECOGNIZED"})
    monkeypatch.setattr(psij, "_scheduler_output",
                        lambda _: (_ for _ in ()).throw(AssertionError("unexpected")))
    status = psij._slurm_status_native(
        "42", submitted_at=submitted.isoformat(), output_paths=(str(output),)
    )
    assert status.state == RunState.UNKNOWN
