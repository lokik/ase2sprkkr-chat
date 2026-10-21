import pytest

from ase2sprkkr.execution import RunState
from ase2sprkkr.execution import remote_mailbox, remote_service


def test_direct_remote_process_without_pending_identity_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    assert remote_service.recover_remote(
        "a" * 32, {"executor": "local", "directory": "/scratch/case"},
    ) is None


def test_psij_recovery_uses_backend_native_identity(monkeypatch):
    calls = []

    class Backend:
        backend = "slurm"
        url = None

        def recover_native(self, run_id, directory, *, submitted_at=None):
            calls.append((run_id, directory, submitted_at))
            return {
                "native_id": "scheduler-42",
                "state": RunState.RUNNING,
                "directory": "/scratch/case",
                "stdout": "/scratch/case/run.out",
                "stderr": "/scratch/case/run.err",
                "return_code": None,
            }

    monkeypatch.setattr(remote_service, "_psij_executor", lambda options: Backend())

    recovered = remote_service.recover_remote(
        "client-run",
        {
            "executor": "psij",
            "executor_options": {"backend": "slurm", "url": None},
            "directory": "/scratch/case",
        },
        submitted_at="2026-10-13T08:00:00+02:00",
    )

    assert calls == [
        ("client-run", "/scratch/case", "2026-10-13T08:00:00+02:00")
    ]
    assert recovered["state"] == RunState.RUNNING.value
    assert recovered["directory"] == "/scratch/case"
    assert recovered["output_file"] == "/scratch/case/run.out"

    handle = remote_service._decode_handle(recovered["remote_id"])
    assert handle == {
        "kind": "psij",
        "backend": "slurm",
        "url": None,
        "native_id": "scheduler-42",
        "receipt_id": remote_service._psij_receipt_id(
            "client-run", "slurm", None, "scheduler-42", "/scratch/case",
        ),
        "directory": "/scratch/case",
    }


def test_psij_recovery_returns_none_when_backend_has_no_match(monkeypatch):
    class Backend:
        backend = "slurm"
        url = None

        def recover_native(self, run_id, directory, *, submitted_at=None):
            return None

    monkeypatch.setattr(remote_service, "_psij_executor", lambda options: Backend())

    assert remote_service.recover_remote(
        "client-run",
        {
            "executor": "psij",
            "executor_options": {"backend": "slurm"},
            "directory": "/scratch/case",
        },
    ) is None


def test_ticket_and_outcome_share_configured_state_root(tmp_path, monkeypatch):
    key = "f" * 32
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    ticket = remote_mailbox._validated_path(key, "remote-submissions")
    receipt = remote_mailbox._path(key)
    assert ticket == tmp_path / "state" / "remote-submissions" / f"{key}.json"
    assert receipt == tmp_path / "state" / "remote-outcomes" / f"{key}.json"

    remote_mailbox.publish_submission(key, {"remote_id": "job-12"})
    assert remote_mailbox.read_submission(key) == {"remote_id": "job-12"}
    assert remote_service.acknowledge_submission(key) == {"acknowledged": True}
    assert remote_mailbox.read_submission(key) is None
    assert remote_service.acknowledge_submission(key) == {"acknowledged": False}


def test_ticket_uses_xdg_state_home_without_override(tmp_path, monkeypatch):
    key = "e" * 32
    monkeypatch.delenv("ASE2SPRKKR_REMOTE_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert remote_mailbox._validated_path(key, "remote-submissions") == (
        tmp_path / "xdg" / "ase2sprkkr" / "remote-submissions" / f"{key}.json"
    )


def test_ticket_validation_is_independent_from_receipt_layout():
    with pytest.raises(ValueError, match="Invalid remote receipt ID"):
        remote_mailbox._validated_path("../invalid", "remote-submissions")


def test_recovered_psij_receipt_freezes_manifest_and_acknowledges(tmp_path, monkeypatch):
    from dataclasses import asdict
    from ase2sprkkr.execution import FileEntry, FileManifest, FileMapping, RunStatus
    from ase2sprkkr.execution import psij

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "one.dat").write_text("first")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.dat"), str(remote / "*.dat"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: type(
        "Backend", (), {
            "backend": "slurm", "url": None,
            "recover_native": lambda self, *args, **kwargs: {
                "native_id": "42", "state": RunState.FINISHED,
                "return_code": 0, "directory": str(remote),
                "stdout": None, "stderr": None,
            },
        },
    )())
    monkeypatch.setattr(psij, "_slurm_status_native", lambda _: RunStatus(
        RunState.FINISHED, backend_id="42", return_code=0,
    ))
    recovered = remote_service.recover_remote("project-run", {
        "executor": "psij", "directory": str(remote),
    })
    job = recovered["remote_id"]
    first = remote_service.status_remote(job, asdict(manifest))
    assert first["state"] == "finished"
    assert len(first["files"]["outputs"]) == 1
    (remote / "late.dat").write_text("late")
    assert remote_service.status_remote(job, asdict(manifest))["files"] == first["files"]
    assert remote_service.acknowledge_remote(job) == {"acknowledged": True}
    assert remote_service.acknowledge_remote(job) == {"acknowledged": False}
