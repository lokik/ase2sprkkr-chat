"""Terminal collection is one operation, with durable file membership."""

import asyncio
from types import SimpleNamespace

from ase2sprkkr.execution import FileEntry, FileManifest, JobFiles


def test_collection_preserves_a_previously_delivered_output(tmp_path):
    output = tmp_path / "downloaded.out"
    output.write_text("delivered")
    previous = FileManifest(outputs=(
        FileEntry(str(output), "/remote/output", availability_state="available"),
    ))
    frozen = FileManifest(outputs=(
        FileEntry(str(output), "/remote/output", availability_state="pending"),
    ))

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            assert manifest.outputs[0].availability_state == "available"
            return manifest, None

        async def verify_stream_availability(self, manifest, spec):
            return manifest, None

    owner = JobFiles(previous)
    error = asyncio.run(owner.collect_remote(
        frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=True,
    ))
    actual = owner.manifest
    assert error is None
    assert actual is frozen or actual.outputs[0].availability_state == "available"
    assert actual.outputs[0].availability_state == "available"
    assert previous.outputs[0].availability_state == "available"
    assert frozen.outputs[0].availability_state == "pending"


def test_materialization_error_wins_without_losing_partial_delivery(tmp_path):
    stream = FileEntry(str(tmp_path / "stdout"), "/remote/stdout")
    product = FileEntry(str(tmp_path / "result"), "/remote/result")
    frozen = FileManifest(stdout=stream, outputs=(product,))
    transfer_error = OSError("transfer failed")
    calls = []

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            calls.append("collect")
            return manifest.with_file_availability(
                product.local_path, "failed", str(transfer_error),
                remote_path=product.remote_path,
            ), transfer_error

        async def verify_stream_availability(self, manifest, spec):
            calls.append("verify")
            return manifest.with_file_availability(
                stream.local_path, "available",
                remote_path=stream.remote_path,
            ), None

    owner = JobFiles(frozen)
    error = asyncio.run(owner.collect_remote(
        frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=True, materialization_error="wildcard scan failed",
    ))
    result = owner.manifest
    assert calls == ["collect", "verify"]
    assert isinstance(error, RuntimeError)
    assert str(error) == "wildcard scan failed"
    assert result.stdout.availability_state == "available"
    assert result.outputs[0].availability_state == "failed"
    assert result.outputs[0].availability_message == "transfer failed"


def test_failed_backend_skips_stream_verification_but_collects_diagnostics(tmp_path):
    diagnostic = FileEntry(str(tmp_path / "diagnostic"), "/remote/log")
    frozen = FileManifest(outputs=(diagnostic,))
    calls = []

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            assert required is False
            calls.append("collect")
            return manifest.with_file_availability(
                diagnostic.local_path, "available",
                remote_path=diagnostic.remote_path,
            ), None

        async def verify_stream_availability(self, manifest, spec):
            raise AssertionError("Failed backend must not verify FOLLOW streams")

    owner = JobFiles(frozen)
    error = asyncio.run(owner.collect_remote(
        frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=False,
    ))
    result = owner.manifest
    assert error is None
    assert calls == ["collect"]
    assert result.outputs[0].availability_state == "available"


def test_concurrent_failed_transfer_keeps_verified_local_copy(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion
    from dataclasses import asdict

    target = tmp_path / "result.dat"
    target.write_bytes(b"verified output")
    version = FileVersion.capture(target, require_digest=True)
    path = str(target)
    entry = FileEntry(
        path, "/remote/result.dat", availability_state="available",
        verified_local_version=version,
    )
    initial = FileManifest(outputs=(entry,))
    assert FileManifest(**asdict(initial)) == initial

    class FailedHandler:
        async def collect(self, mapping, manifest, outcome, *, required):
            return manifest.with_file_availability(
                path, "failed", "connection reset",
                remote_path="/remote/result.dat",
            ), OSError("connection reset")

        async def verify_stream_availability(self, manifest, spec):
            return manifest, None

    owner = JobFiles(initial)
    error = asyncio.run(owner.collect_remote(
        initial, FailedHandler(), SimpleNamespace(), SimpleNamespace(),
        required=True,
    ))
    assert error is None
    assert owner.manifest.outputs[0].availability_state == "available"
    assert owner.manifest.outputs[0].verified_local_version == version

    # The hash is authoritative: a later damaged copy must not be protected.
    target.write_bytes(b"corrupted")
    error = asyncio.run(owner.collect_remote(
        initial, FailedHandler(), SimpleNamespace(), SimpleNamespace(),
        required=True,
    ))
    assert isinstance(error, OSError)
    assert owner.manifest.outputs[0].availability_state == "failed"


def test_stale_failed_snapshot_preserves_only_unchanged_verified_file(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.manifest_merge import merge_snapshot

    path = tmp_path / "result.out"
    path.write_bytes(b"success")
    current = FileManifest(outputs=(FileEntry(
        str(path), "/remote/result.out", availability_state="available",
        verified_local_version=FileVersion.capture(path, require_digest=True),
    ),))
    stale = current.with_file_availability(
        str(path), "failed", "later failed transfer",
    )
    assert merge_snapshot(current, stale).outputs[0].availability_state == "available"
    path.unlink()
    assert merge_snapshot(current, stale).outputs[0].availability_state == "failed"


def test_frozen_failure_does_not_erase_verified_download(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.manifest_merge import merge_terminal_files

    path = tmp_path / "result.dat"
    path.write_bytes(b"valid")
    available = FileManifest(outputs=(FileEntry(
        str(path), "/remote/result.dat", availability_state="available",
        verified_local_version=FileVersion.capture(path, require_digest=True),
    ),))
    failed = available.with_file_availability(
        str(path), "failed", "remote scan failed",
    )
    assert merge_terminal_files(available, failed).outputs[0].availability_state == "available"
    path.unlink()
    assert merge_terminal_files(available, failed).outputs[0].availability_state == "failed"
