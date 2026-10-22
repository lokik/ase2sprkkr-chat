"""Target-service command semantics independent of the execution location."""
import asyncio
from dataclasses import asdict
import json

from ase2sprkkr.execution import FileManifest
from ase2sprkkr.execution.target_connection import LocalConnection, SSHConnection


def test_local_connection_uses_target_service(monkeypatch):
    from ase2sprkkr.execution import remote_service

    calls = []
    monkeypatch.setattr(
        remote_service, "submit_remote",
        lambda request: calls.append(("submit", request)) or {"remote_id": "job"},
    )
    monkeypatch.setattr(
        remote_service, "status_remote",
        lambda job: calls.append(("status", job)) or {"state": "running"},
    )
    monkeypatch.setattr(
        remote_service, "cancel_remote",
        lambda job: calls.append(("cancel", job)) or {"state": "cancelled"},
    )
    connection = LocalConnection()
    assert asyncio.run(connection.submit({"executor": "local"}))["remote_id"] == "job"
    assert asyncio.run(connection.status("job", FileManifest()))["state"] == "running"
    assert asyncio.run(connection.cancel("job"))["state"] == "cancelled"
    assert calls == [
        ("submit", {"executor": "local"}),
        ("status", "job"),
        ("cancel", "job"),
    ]


def test_ssh_connection_preserves_status_manifest_and_acknowledgements():
    calls = []

    class Transport:
        async def run(self, command, input=None):
            import json
            calls.append((command, None if input is None else json.loads(input)))
            return type("Reply", (), {"stdout": b'{"state":"running"}'})()

    conn = SSHConnection(
        Transport(), command=("ase2sprkkr", "remote"),
        backend="local", backend_options={},
    )
    files = FileManifest()
    assert asyncio.run(conn.status("job", files)) == {"state": "running"}
    asyncio.run(conn.acknowledge_submission("run"))
    asyncio.run(conn.acknowledge("job"))
    assert calls == [
        (("ase2sprkkr", "remote", "status", "job"),
         json.loads(json.dumps({"files": asdict(files)}))),
        (("ase2sprkkr", "remote", "ack-submit", "run"), None),
        (("ase2sprkkr", "remote", "ack", "job"), None),
    ]


def test_ssh_follow_delivers_decoded_chunks_without_exposing_transport(monkeypatch):
    from ase2sprkkr.execution.target_connection import SSHConnection
    from ase2sprkkr.execution import target_connection

    seen = []
    class Framed:
        terminal = {"state": "finished", "return_code": 0}
        def __init__(self, process):
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_data(b"standard")
            self.stdout.feed_eof()
            self.stderr.feed_data(b"diagnostic")
            self.stderr.feed_eof()
        async def wait(self):
            return 0
        def terminate(self):
            raise AssertionError("successful follow must not be terminated")

    class Transport:
        async def start(self, command):
            assert "follow" in command and "--framed" in command
            return object()

    monkeypatch.setattr(target_connection, "FramedRunningCommand", Framed)
    connection = SSHConnection(
        Transport(), command=("ase2sprkkr", "remote"),
        backend="local", backend_options={},
    )
    receipt = asyncio.run(connection.follow(
        "opaque", FileManifest(),
        on_chunk=lambda kind, data: seen.append((kind, data)),
    ))
    assert receipt == {"state": "finished", "return_code": 0}
    assert sorted(seen) == [("err", b"diagnostic"), ("out", b"standard")]
