"""Shared fixtures for execution integration tests."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.metadata import requires as distribution_requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from kkrunner.client import pool as execution_pool


@dataclass(frozen=True)
class DockerExecutionImage:
    """Built Docker image and paths shared by execution integration tests."""

    docker: str
    repository: Path
    image: str


def _package_mount_root() -> Path:
    """Return a host path importable inside both source and wheel test layouts."""
    import ase2sprkkr

    package_dir = Path(ase2sprkkr.__file__).resolve().parent
    package_parent = package_dir.parent
    if package_parent.name == "src":
        return package_parent.parent
    return package_parent


@pytest.fixture(autouse=True)
def isolated_default_client_pool():
    """Keep the process-wide executor pool from leaking between test cases."""
    previous = execution_pool._DEFAULT_CLIENT_POOL
    execution_pool._DEFAULT_CLIENT_POOL = execution_pool.ClientPool()
    try:
        yield
    finally:
        execution_pool._DEFAULT_CLIENT_POOL.close()
        execution_pool._DEFAULT_CLIENT_POOL = previous


def _declared_kkrunner_requirement():
    """Install precisely the KKRunner range declared by ASE2SPRKKR.

    The remote Docker container needs PSI/J, so enable the declared
    optional extra without pinning a separate KKRunner checkout.
    """
    requirements = (
        Requirement(value) for value in distribution_requires("ase2sprkkr") or ()
    )
    matching = [
        dependency for dependency in requirements
        if dependency.name.lower() == "kkrunner" and dependency.marker is None
    ]
    if len(matching) != 1:
        raise RuntimeError("Expected one unconditional KKRunner runtime requirement")
    return f"kkrunner[psij]{matching[0].specifier}"


@pytest.fixture(scope="session")
def docker_execution_image():
    """Build the common execution-test image once per pytest session."""
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is required for the execution integration tests")

    repository = _package_mount_root()
    source_context = Path(__file__).with_name("docker")
    image = "ase2sprkkr-execution-test"
    with tempfile.TemporaryDirectory(prefix="ase2sprkkr-docker-") as temporary:
        context = Path(temporary)
        shutil.copytree(source_context, context, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(
            [
                docker, "build", "--build-arg",
                f"KKRUNNER_REQUIREMENT={_declared_kkrunner_requirement()}",
                "-t", image, str(context),
            ],
            check=True,
        )
    return DockerExecutionImage(docker=docker, repository=repository, image=image)
