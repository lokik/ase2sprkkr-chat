from uuid import uuid4

import pytest

from ase2sprkkr.execution import FailurePhase, RunState
from ase2sprkkr.execution import remote_mailbox


@pytest.fixture(autouse=True)
def isolated_mailbox(tmp_path, monkeypatch):
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))


def test_terminal_outcome_survives_reads_until_ack():
    key = uuid4().hex
    assert remote_mailbox.read_outcome(key) is None
    assert remote_mailbox.publish_outcome(key, RunState.FAILED, 7, FailurePhase.EXECUTION)
    expected = {
        "state": "failed",
        "return_code": 7,
        "failure_phase": "execution",
        "error": None,
        "files": None,
        "materialization_error": None,
    }
    assert remote_mailbox.read_outcome(key) == expected
    assert remote_mailbox.read_outcome(key) == expected
    assert remote_mailbox.acknowledge_outcome(key)
    assert not remote_mailbox.acknowledge_outcome(key)
    assert remote_mailbox.read_outcome(key) is None


def test_first_writer_wins_even_when_cancel_races_with_completion():
    key = uuid4().hex
    assert remote_mailbox.publish_outcome(key, RunState.FINISHED, 0)
    assert not remote_mailbox.publish_outcome(key, RunState.CANCELLED, None)
    assert remote_mailbox.read_outcome(key)["state"] == "finished"


def test_rejects_invalid_id_and_nonterminal_state(tmp_path):
    with pytest.raises(ValueError, match="receipt ID"):
        remote_mailbox.read_outcome("../outside")
    with pytest.raises(ValueError, match="terminal"):
        remote_mailbox.publish_outcome(uuid4().hex, RunState.RUNNING, None)
    assert list(tmp_path.rglob("*.json")) == []


@pytest.mark.parametrize("override", (True, False), ids=("explicit", "xdg"))
def test_receipt_state_root_respects_override_and_xdg(tmp_path, monkeypatch, override):
    key = uuid4().hex
    if override:
        monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "explicit"))
        root = tmp_path / "explicit"
    else:
        monkeypatch.delenv("ASE2SPRKKR_REMOTE_STATE_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
        root = tmp_path / "xdg" / "ase2sprkkr"
    assert remote_mailbox._path(key) == root / "remote-outcomes" / f"{key}.json"


def test_submission_ticket_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    key = uuid4().hex
    assert remote_mailbox.read_submission(key) is None
    remote_mailbox.publish_submission(key, {"remote_id": "native"})
    assert remote_mailbox.read_submission(key) == {"remote_id": "native"}
    assert remote_mailbox.drop_submission(key)
    assert not remote_mailbox.drop_submission(key)


def test_publication_error_survives_cleanup_fsync_error(monkeypatch):
    key = uuid4().hex
    original_fsync = remote_mailbox.fsync_directory

    def failed_write(*args, **kwargs):
        raise OSError("primary publication failure")

    def failed_cleanup(*args, **kwargs):
        raise OSError("secondary directory fsync failure")

    monkeypatch.setattr(remote_mailbox, "durable_write_json", failed_write)
    monkeypatch.setattr(remote_mailbox, "fsync_directory", failed_cleanup)
    with pytest.raises(OSError, match="primary publication failure"):
        remote_mailbox.publish_outcome(key, RunState.FINISHED, 0)
    monkeypatch.setattr(remote_mailbox, "fsync_directory", original_fsync)
    assert remote_mailbox.read_outcome(key) is None


def test_successful_receipt_survives_temporary_cleanup_failure(monkeypatch):
    key = uuid4().hex
    original_fsync = remote_mailbox.fsync_directory
    calls = []

    def fail_second_fsync(directory):
        calls.append(directory)
        if len(calls) == 2:
            raise OSError("temporary cleanup fsync failed")
        return original_fsync(directory)

    monkeypatch.setattr(remote_mailbox, "fsync_directory", fail_second_fsync)
    assert remote_mailbox.publish_outcome(key, RunState.FINISHED, 0) is True
    assert len(calls) == 2
    assert remote_mailbox.read_outcome(key)["state"] == "finished"


def test_retry_after_first_publication_fsync_failure(monkeypatch):
    key = uuid4().hex
    original_fsync = remote_mailbox.fsync_directory
    calls = 0

    def fail_link_directory_sync(directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("receipt link was not confirmed durable")
        return original_fsync(directory)

    monkeypatch.setattr(remote_mailbox, "fsync_directory", fail_link_directory_sync)
    with pytest.raises(OSError, match="not confirmed durable"):
        remote_mailbox.publish_outcome(key, RunState.FINISHED, 0)

    # The link exists despite the first fsync failure. Retrying must fsync
    # that existing name before reporting first-writer-wins.
    assert remote_mailbox.read_outcome(key)["state"] == "finished"
    assert remote_mailbox.publish_outcome(key, RunState.CANCELLED, None) is False
    assert calls >= 4
    assert remote_mailbox.read_outcome(key)["state"] == "finished"


@pytest.mark.parametrize("field,value", [
    ("state", "running"),
    ("return_code", True),
    ("return_code", "0"),
    ("failure_phase", "invalid-phase"),
    ("files", []),
    ("error", {"unexpected": "mapping"}),
    ("materialization_error", 4),
])
def test_rejects_malformed_terminal_receipt_fields(tmp_path, field, value):
    import json
    key = uuid4().hex
    assert remote_mailbox.publish_outcome(key, RunState.FINISHED, 0)
    path = remote_mailbox._path(key)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid terminal receipt"):
        remote_mailbox.read_outcome(key)


def test_rejects_non_object_terminal_receipt():
    key = uuid4().hex
    assert remote_mailbox.publish_outcome(key, RunState.FINISHED, 0)
    remote_mailbox._path(key).write_text("[]")
    with pytest.raises(ValueError, match="Invalid terminal receipt"):
        remote_mailbox.read_outcome(key)
