"""Availability resets belong to the immutable FileManifest."""

from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_retry_failed_downloads_changes_only_failed_delivery_state(tmp_path):
    path = lambda name: str(tmp_path / name)
    manifest = FileManifest(
        inputs=(FileEntry(path("input"), availability_state="failed",
                          availability_message="input diagnostic"),),
        stdout=FileEntry(path("stdout"), availability_state="available"),
        stderr=FileEntry(path("stderr"), availability_state="failed",
                         availability_message="lost stream"),
        outputs=(
            FileEntry(path("completed"), availability_state="available"),
            FileEntry(path("failed"), availability_state="failed",
                      availability_message="lost transfer"),
            FileEntry(path("waiting"), availability_state="pending"),
        ),
        caches=(FileEntry(path("restart"), availability_state="skipped"),),
    )
    reopened = manifest.retry_failed_downloads()
    assert reopened.inputs == manifest.inputs
    assert reopened.stdout == manifest.stdout
    assert reopened.outputs[0] == manifest.outputs[0]
    assert reopened.outputs[2] == manifest.outputs[2]
    assert reopened.caches == manifest.caches
    assert reopened.stderr.availability_state == "pending"
    assert reopened.stderr.availability_message is None
    assert reopened.outputs[1].availability_state == "pending"
    assert reopened.outputs[1].availability_message is None
    assert manifest.stderr.availability_state == "failed"
    assert reopened.retry_failed_downloads() == reopened
