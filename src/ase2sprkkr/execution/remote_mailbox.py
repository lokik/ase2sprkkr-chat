"""Durable handoff of temporary submission identities and terminal results."""

import json
import os
import re
from pathlib import Path
from uuid import uuid4

from ._durability import durable_mkdir, durable_write_json, fsync_directory
from .base import FailurePhase, RunState

_RECEIPT = re.compile(r"[0-9a-f]{32}\Z")


def _state_root():
    """Common state directory for short-lived submissions and outcomes."""
    root = os.environ.get("ASE2SPRKKR_REMOTE_STATE_DIR")
    if root:
        return Path(root).expanduser()
    return Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser() / "ase2sprkkr"


def _validated_path(receipt_id, folder):
    if not isinstance(receipt_id, str) or not _RECEIPT.fullmatch(receipt_id):
        raise ValueError("Invalid remote receipt ID")
    return _state_root() / folder / (receipt_id + ".json")


def _path(receipt_id):
    return _validated_path(receipt_id, "remote-outcomes")


def publish_submission(run_id, reply):
    """Persist short-lived backend identity until acknowledged."""
    path = _validated_path(run_id, "remote-submissions")
    durable_mkdir(path.parent, parents=True, exist_ok=True, mode=0o700)
    durable_write_json(path, reply, mode=0o600)


def read_submission(run_id):
    try:
        return json.loads(_validated_path(run_id, "remote-submissions").read_text())
    except FileNotFoundError:
        return None


def drop_submission(run_id):
    path = _validated_path(run_id, "remote-submissions")
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    fsync_directory(path.parent)
    return True


def publish_outcome(receipt_id, state, return_code, failure_phase=None, *,
                    error=None, files=None, materialization_error=None):
    """Publish once: cancellation must not overwrite natural completion."""
    state = RunState(state)
    if not state.is_terminal:
        raise ValueError("Backend receipt requires a terminal state")
    path = _path(receipt_id)
    durable_mkdir(path.parent, parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / ("." + receipt_id + "." + uuid4().hex + ".prepared")
    try:
        durable_write_json(
            temporary,
            {
                "state": state.value,
                "return_code": None if return_code is None else int(return_code),
                "failure_phase": None if failure_phase is None else FailurePhase(failure_phase).value,
                "error": error,
                "files": files,
                "materialization_error": materialization_error,
            },
            separators=(",", ":"),
            mode=0o600,
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        fsync_directory(path.parent)
        return True
    finally:
        # The receipt's link has its own fsync above. Removal of the
        # temporary staging name is best-effort: failure here must neither
        # mask a publication error nor turn a durable receipt into failure.
        try:
            temporary.unlink(missing_ok=True)
            fsync_directory(path.parent)
        except OSError:
            pass


def read_outcome(receipt_id):
    """Read non-destructively so interrupted clients can retry."""
    try:
        result = json.loads(_path(receipt_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    try:
        if not RunState(result["state"]).is_terminal:
            raise ValueError("Receipt is not terminal")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid terminal receipt") from exc
    return result


def acknowledge_outcome(receipt_id):
    """Delete only after the client has durably committed its own run state."""
    path = _path(receipt_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    fsync_directory(path.parent)
    return True
