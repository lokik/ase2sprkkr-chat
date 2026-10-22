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

    actual, error = asyncio.run(JobFiles.collect_remote(
        previous, frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=True,
    ))
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

    result, error = asyncio.run(JobFiles.collect_remote(
        frozen, frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=True, materialization_error="wildcard scan failed",
    ))
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

    result, error = asyncio.run(JobFiles.collect_remote(
        None, frozen, Handler(), SimpleNamespace(), SimpleNamespace(),
        required=False,
    ))
    assert error is None
    assert calls == ["collect"]
    assert result.outputs[0].availability_state == "available"
