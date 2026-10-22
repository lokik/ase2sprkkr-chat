import io
import json
from types import SimpleNamespace

import ase2sprkkr.execution.remote_service as remote_service
from ase2sprkkr.tools.commands import remote as remote_command


def test_status_command_forwards_opaque_remote_handle(monkeypatch):
    remote_id = "opaque-handle"
    calls = []
    monkeypatch.setattr(
        remote_service,
        "status_remote",
        lambda value, files=None: calls.append((value, files)) or {"state": "running"},
    )
    output = []
    monkeypatch.setattr(remote_command, "_print_json", output.append)
    monkeypatch.setattr(remote_command.sys, "stdin", io.StringIO(""))

    remote_command.run(
        SimpleNamespace(remote_action="status", remote_id=remote_id),
        None,
    )

    assert calls == [(remote_id, None)]
    assert output == [{"state": "running"}]


def test_submit_command_passes_complete_request_to_stateless_service(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote_service,
        "submit_remote",
        lambda request: calls.append(dict(request))
        or {"remote_id": "opaque", "state": "running"},
    )
    monkeypatch.setattr(
        remote_command.sys,
        "stdin",
        io.StringIO(json.dumps({"id": "run-1", "argv": ["true"]})),
    )
    output = []
    monkeypatch.setattr(remote_command, "_print_json", output.append)

    remote_command.run(
        SimpleNamespace(remote_action="submit", directory="/remote/work"),
        None,
    )

    assert calls == [
        {"id": "run-1", "argv": ["true"], "directory": "/remote/work"}
    ]
    assert output == [{"remote_id": "opaque", "state": "running"}]


def test_resolve_mpi_command_preserves_integer_process_count(monkeypatch):
    monkeypatch.setattr(
        "ase2sprkkr.execution.mpi.find_default_mpi_launcher",
        lambda: ("mpirun",),
    )
    monkeypatch.setattr(remote_command.sys, "stdin", io.StringIO('{"mpi": 4}'))
    output = []
    monkeypatch.setattr(remote_command, "_print_json", output.append)

    remote_command.run(SimpleNamespace(remote_action="resolve-mpi"), None)

    assert output == [{"mpi": ("mpirun", "-np", "4")}]


def test_status_command_forwards_manifest_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(
        remote_service, "status_remote",
        lambda remote_id, files=None: calls.append((remote_id, files))
        or {"state": "finished", "return_code": 0},
    )
    payload = {"files": {"outputs": [{"remote_path": "/remote/result.dat"}]}}
    monkeypatch.setattr(remote_command.sys, "stdin", io.StringIO(json.dumps(payload)))
    results = []
    monkeypatch.setattr(remote_command, "_print_json", results.append)

    remote_command.run(
        SimpleNamespace(remote_action="status", remote_id="job-with-manifest"), None,
    )

    assert calls == [("job-with-manifest", payload["files"])]
    assert results == [{"state": "finished", "return_code": 0}]
