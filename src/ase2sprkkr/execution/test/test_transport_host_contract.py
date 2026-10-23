"""Endpoint hosts are part of the transport contract, not duck-typed fallbacks."""

from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.execution.remote_endpoint import RemoteEndpointExecutor
from ase2sprkkr.execution.transport import LocalTransport, SshTransport, Transport
from ase2sprkkr.execution.run_model import FileEntry


def test_transport_host_is_explicit_for_local_and_ssh():
    assert Transport.host is None
    local = LocalTransport()
    assert local.host is None
    ssh = SshTransport("remote-host")
    assert ssh.host == "remote-host"
    assert RemoteEndpointExecutor(local).computer is None
    assert RemoteEndpointExecutor(ssh).computer == "remote-host"


def test_endpoint_identity_uses_transport_host():
    local = RemoteEndpointExecutor(LocalTransport())
    assert endpoint_for_executor(local).host is None

    ssh = RemoteEndpointExecutor(SshTransport("cluster"))
    endpoint = endpoint_for_executor(ssh)
    assert endpoint.executor == "ssh"
    assert endpoint.host == "cluster"


def test_file_entry_normalizes_paths_without_dynamic_field_lookup(tmp_path):
    local = tmp_path / "local.dat"
    entry = FileEntry(local_path=local, remote_path=tmp_path / "remote.dat")
    assert entry.local_path == str(local)
    assert entry.remote_path == str(tmp_path / "remote.dat")
