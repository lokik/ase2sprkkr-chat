"""Ownership and single-pass validation for per-result file attribution."""

from types import SimpleNamespace

import pytest

from kkrunner.files import FileManifest, FileVersion
from kkrunner.runs import HistoricalFileMismatchError

from ase2sprkkr.outputs import FileAttribution, TaskResult


def test_registered_file_checked_once_for_each_public_operation(tmp_path, monkeypatch):
    path = tmp_path / "current.dat"
    path.write_text("original")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("current", path)
    observed = FileVersion.capture(path, digest=False)
    option.set_attribution(FileAttribution.LIKELY, observed_version=observed)

    actual_state = FileVersion.state
    checks = []

    def counted_state(self):
        if self is observed:
            checks.append(self)
        return actual_state(self)

    monkeypatch.setattr(FileVersion, "state", counted_state)

    assert result.file_is_current(path)
    assert len(checks) == 1
    checks.clear()

    result.require_current_file(path)
    assert len(checks) == 1
    checks.clear()

    assert option.require_current() == str(path)
    assert len(checks) == 1
    checks.clear()

    assert option.is_current()
    assert len(checks) == 1
    checks.clear()

    assert option.attribution_details() == (FileAttribution.LIKELY, None)
    assert len(checks) == 1


def test_competing_scf_and_tied_mtimes_never_use_potential(tmp_path):
    path = tmp_path / "Fe.pot_new"
    path.write_text("potential")
    result = TaskResult(None, None, tmp_path, task_name="SCF")
    option = result.files.add_file("converged", path)
    observed = FileVersion.capture(path, digest=False)

    option.set_attribution(
        FileAttribution.NOT_THIS_RESULT,
        observed_version=observed,
        evidence="newer_scf_stdout",
        attributed_to=tmp_path / "newer.out",
    )
    assert option.attribution is FileAttribution.NOT_THIS_RESULT
    assert option.attributed_to == tmp_path / "newer.out"
    assert not result.file_is_current(path)
    with pytest.raises(HistoricalFileMismatchError):
        option.require_current()

    option.set_attribution(
        FileAttribution.UNKNOWN,
        observed_version=observed,
        evidence="scf_mtime_tie_or_unreadable",
    )
    assert option.attribution is FileAttribution.UNKNOWN
    assert not option.is_current()
    with pytest.raises(HistoricalFileMismatchError):
        result.require_current_file(path)


def test_historical_file_without_loader_uses_manifest_proof(tmp_path):
    path = tmp_path / "saved.dat"
    path.write_text("old")
    version = FileVersion.capture(path, require_digest=True)
    manifest = FileManifest(
        output_versions=(version,),
        captured_at="2026-11-22T00:00:00+00:00",
    )
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    result.run_record = SimpleNamespace(files=manifest)
    option = result.files.add_file("saved", path)

    assert option.is_current()
    assert option.attribution is FileAttribution.VERIFIED
    assert option.attribution_evidence == "run_manifest_digest"
    path.write_text("overwritten")
    assert option.attribution is FileAttribution.NOT_THIS_RESULT
    assert not result.file_is_current(path)

    path.unlink()
    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "recorded_file_missing"
    with pytest.raises(HistoricalFileMismatchError):
        option.require_current()


def test_unrecorded_historical_file_cannot_adopt_new_bytes(tmp_path):
    path = tmp_path / "unknown.dat"
    path.write_text("fresh data")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    result.run_record = SimpleNamespace(
        files=FileManifest(captured_at="2026-11-22T00:00:00+00:00"),
    )
    option = result.files.add_file("unknown", path)

    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "not_in_manifest"
    assert not option.is_current()
    with pytest.raises(HistoricalFileMismatchError):
        option.require_current()


def test_unregistered_potential_uses_manifest_fallback(tmp_path):
    output = tmp_path / "saved.out"
    output.write_text("output")
    host = tmp_path / "host.pot"
    host.write_text("host")
    version = FileVersion.capture(host, require_digest=True)
    result = TaskResult(None, None, tmp_path, output_file=output, task_name="SCF")
    result.run_record = SimpleNamespace(files=FileManifest(
        output_versions=(version,), captured_at="2026-11-22T00:00:00+00:00",
    ))

    assert "host" not in result.files
    assert result.file_is_current(host)
    result.require_current_file(host)
    host.write_text("changed")
    assert not result.file_is_current(host)
    with pytest.raises(HistoricalFileMismatchError):
        result.require_current_file(host)


def test_observed_version_fences_against_later_rewrites(tmp_path):
    path = tmp_path / "potential.dat"
    path.write_text("original")
    result = TaskResult(None, None, tmp_path, task_name="SCF")
    option = result.files.add_file("potential", path)
    option.set_attribution(
        FileAttribution.LIKELY, evidence="output_or_input_reference",
        observed_version=FileVersion.capture(path, digest=False),
    )
    assert option.attribution is FileAttribution.LIKELY
    path.write_text("rewritten contents")
    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "observed_version_changed"
    assert not option.is_current()
    with pytest.raises(HistoricalFileMismatchError):
        result.require_current_file(path)


def test_currentness_checks_do_not_reenter_owner(tmp_path, monkeypatch):
    path = tmp_path / "result.dat"
    path.write_text("available")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("result", path)

    def owner_cannot_decide(_path):
        raise AssertionError("File option must not call TaskResult.file_is_current")

    monkeypatch.setattr(result, "file_is_current", owner_cannot_decide)
    assert option.is_current()
    assert option.require_current() == str(path)


def test_path_level_validation_considers_all_registered_roles(tmp_path):
    path = tmp_path / "shared.pot"
    path.write_text("shared")
    result = TaskResult(None, None, tmp_path, task_name="SCF")
    input_option = result.files.add_file("potential", path)
    output_option = result.files.add_file("converged", path)
    observed = FileVersion.capture(path, digest=False)
    input_option.set_attribution(FileAttribution.LIKELY, observed_version=observed)
    output_option.set_attribution(
        FileAttribution.NOT_THIS_RESULT, evidence="newer_scf_stdout",
        observed_version=observed,
    )

    # Role-specific checks are independent. The path-only check has no role
    # information, so it must honor all evidence and reject conflicting files.
    assert input_option.is_current()
    assert not output_option.is_current()
    assert not result.file_is_current(path)
    with pytest.raises(HistoricalFileMismatchError):
        result.require_current_file(path)


def test_missing_unversioned_file_has_unknown_current_attribution(tmp_path):
    path = tmp_path / "unversioned.dat"
    path.write_bytes(b"temporary")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("auxiliary", path)
    option.set_attribution(FileAttribution.LIKELY, evidence="manual_reference")
    assert option.attribution is FileAttribution.LIKELY
    path.unlink()
    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "missing"
    assert not option.is_current()


def test_metadata_only_history_is_likely_not_verified(tmp_path):
    path = tmp_path / "metadata.dat"
    path.write_bytes(b"original")
    version = FileVersion.capture(path, digest=False)
    result = TaskResult(None, None, tmp_path, task_name="SCF")
    result.run_record = SimpleNamespace(files=FileManifest(
        output_versions=(version,), captured_at="2026-11-22T00:00:00+00:00",
    ))
    option = result.files.add_file("metadata", path)
    assert option.attribution is FileAttribution.LIKELY
    assert option.is_current()
    path.write_bytes(b"rewritten")
    assert option.attribution is FileAttribution.UNKNOWN
    assert not option.is_current()


def test_specific_result_file_action_uses_its_own_role(tmp_path, monkeypatch):
    from ase2sprkkr import result_files

    path = tmp_path / "shared.agr"
    path.write_text("@type xy\\n0 1\\n&\\n")
    result = TaskResult(None, None, tmp_path, task_name="SCF")
    valid = result.files.add_file("input_plot", path)
    rejected = result.files.add_file("output_plot", path)
    valid.set_attribution(FileAttribution.LIKELY, observed_version=FileVersion.capture(path, digest=False))
    rejected.set_attribution(FileAttribution.NOT_THIS_RESULT, evidence="newer_scf_stdout")

    parsed = object()
    monkeypatch.setattr(result_files, "read_plottable_file", lambda *_args, **_kwargs: parsed)
    assert result.parse_result_file(valid) is parsed
    with pytest.raises(HistoricalFileMismatchError):
        result.parse_result_file(rejected)


def test_verified_label_without_content_proof_is_only_likely(tmp_path):
    path = tmp_path / "unverified.dat"
    path.write_bytes(b"bytes")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("unverified", path)
    option.set_attribution(FileAttribution.VERIFIED)
    assert option.attribution is FileAttribution.LIKELY
    assert option.is_current()
    option.set_attribution(
        FileAttribution.VERIFIED, observed_version=FileVersion.capture(path, digest=False),
    )
    assert option.attribution is FileAttribution.LIKELY


def test_modified_observed_file_cannot_remain_verified_without_digest(tmp_path):
    path = tmp_path / "observed.dat"
    path.write_bytes(b"original")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("observed", path)
    option.set_attribution(
        FileAttribution.VERIFIED, observed_version=FileVersion.capture(path, digest=False),
    )
    assert option.attribution is FileAttribution.LIKELY
    path.write_bytes(b"modified longer")
    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "observed_version_changed"
    assert not option.is_current()


def test_directory_snapshot_cannot_validate_a_file_option(tmp_path):
    folder = tmp_path / "output"
    folder.mkdir()
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    option = result.files.add_file("unexpected", folder)
    option.set_attribution(
        FileAttribution.LIKELY,
        recorded_version=FileVersion.capture(folder, digest=False, kind="directory"),
    )
    assert option.attribution is FileAttribution.UNKNOWN
    assert option.attribution_evidence == "not_regular_file"
    assert not option.is_current()
    assert option.actions() == ()


def test_parse_result_file_rejects_foreign_file_option(tmp_path):
    path = tmp_path / "a.agr"
    path.write_text("@type xy\\n0 1\\n&\\n")
    owner = TaskResult(None, None, tmp_path, task_name="SCF")
    other = TaskResult(None, None, tmp_path, task_name="DOS")
    foreign = other.files.add_file("plot", path)
    with pytest.raises(ValueError, match="belong to this TaskResult"):
        owner.parse_result_file(foreign)


def test_nexus_export_checks_file_state_once(tmp_path, monkeypatch):
    from ase2sprkkr import result_files

    path = tmp_path / "data.agr"
    path.write_text("@type xy\\n0 1\\n&\\n")
    result = TaskResult(None, None, tmp_path, task_name="ARPES")
    option = result.files.add_file("spectrum", path)
    version = FileVersion.capture(path, digest=False)
    option.set_attribution(FileAttribution.LIKELY, observed_version=version)

    from types import SimpleNamespace
    monkeypatch.setattr(
        result_files, "read_plottable_file",
        lambda *_args, **_kwargs: SimpleNamespace(to_nexus=lambda destination: destination),
    )
    original_state = FileVersion.state
    checks = []

    def counted_state(self):
        if self is version:
            checks.append(1)
        return original_state(self)

    monkeypatch.setattr(FileVersion, "state", counted_state)
    destination = tmp_path / "saved.nxs"
    assert option.export_nexus(destination) == destination
    assert len(checks) == 1
