from kkrunner.backends.config import BackendConfig
from kkrunner.transports import SshTransport

from ase2sprkkr.sprkkr.profile import SPRKKRProfile


def test_profile_persists_transport_separately_from_backend():
    profile = SPRKKRProfile(
        executor="ssh", host="cluster",
        ssh_options=("-p", "2222"), ssh_compression="gz",
        remote_python=("python3",),
        backend="psij", psij_backend="slurm", psij_url="local://slurm",
    )
    assert profile.transport == SshTransport(
        host="cluster", options=("-p", "2222"), compression="gz",
        python_command=("python3",),
    )
    assert profile.backend_config == BackendConfig(
        "psij", {"executor_name": "slurm", "url": "local://slurm"},
    )


def test_ssh_compression_defaults_to_auto():
    assert SPRKKRProfile(executor="ssh", host="cluster").transport.compression == "auto"
