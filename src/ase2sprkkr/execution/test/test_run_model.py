from dataclasses import asdict

import pytest

from ase2sprkkr.execution.base import LaunchSpec
from ase2sprkkr.execution.endpoint import ExecutorEndpoint
from ase2sprkkr.execution.run_model import (
    FileAvailabilityState, Execution, FileEntry, FileManifest,
    FileMapping, FileSyncPolicy, WorkingDirectory,
)


def test_execution_roundtrip_includes_backend_and_both_directories():
    original = Execution(
        ExecutorEndpoint(executor="ssh", host="cluster"),
        WorkingDirectory("/local/case", "/scratch/case"),
        backend_id="slurm-42",
    )
    assert Execution(**asdict(original)) == original
    assert original.directory.target == "/scratch/case"


def test_local_working_directory_does_not_need_remote_copy():
    run = Execution(ExecutorEndpoint(), WorkingDirectory("/tmp/work"))
    assert run.backend_id is None
    assert run.directory.target == "/tmp/work"


def test_manifest_roundtrip_includes_streams_and_cache_baseline():
    original = FileManifest(
        inputs=(FileEntry("/local/input", "/remote/input"),),
        outputs=(FileEntry("/local/result", "/remote/result"),),
        caches=(FileEntry("/local/cache", "/remote/cache", pre_run_signature=(100, 123)),),
        stdout=FileEntry("/local/out", "/remote/out"),
        stderr=FileEntry("/local/err", "/remote/err"),
    )
    restored = FileManifest(**asdict(original))
    assert restored == original
    assert restored.caches[0].pre_run_signature == (100, 123)


def test_empty_backend_identity_and_pathless_file_are_invalid():
    with pytest.raises(ValueError, match="backend_id"):
        Execution(ExecutorEndpoint(), WorkingDirectory("/tmp"), backend_id="")
    with pytest.raises(ValueError, match="location"):
        FileEntry()


def test_manifest_versions_and_availability_roundtrip():
    manifest = FileManifest(
        inputs=(FileEntry("/local/in", "/remote/in"),),
        primary_input=FileEntry("/local/in", "/remote/in"),
        outputs=(FileEntry("/local/data", "/remote/data"),),
        stdout=FileEntry("/local/stdout", "/remote/stdout"),
        stderr=FileEntry("/local/stderr", "/remote/stderr"),
    ).initialize_availability()
    restored = FileManifest(**asdict(manifest))
    assert restored == manifest
    assert restored.input_file == "/local/in"
    assert restored.output_file == "/local/stdout"
    assert restored.input_files == ("/local/in",)
    assert restored.output_files == ("/local/stdout", "/local/stderr", "/local/data")
    assert restored.manifests == ("/local/in", "/local/stdout", "/local/stderr", "/local/data")
    assert restored.all_versions == ()
    with pytest.raises(ValueError, match="availability"):
        FileEntry("/local/test", availability_state="inconsistent")


def test_each_file_tracks_its_own_transfer_outcome():
    manifest = FileManifest(
        outputs=(
            FileEntry("/local/one", "/remote/one", "pending"),
            FileEntry("/local/two", "/remote/two", "pending"),
        ),
        caches=(FileEntry("/local/cache", "/remote/cache", "pending"),),
    )
    manifest = manifest.with_file_availability("/local/one", "available")
    manifest = manifest.with_file_availability(
        "/local/two", "failed", "connection lost",
    )
    assert manifest.outputs[0].availability_state == "available"
    assert manifest.outputs[1].availability_state == "failed"
    assert manifest.failed_files == (manifest.outputs[1],)
    assert manifest.availability_state == "failed"
    manifest = manifest.with_file_availability("/local/cache", "skipped", kind="cache")
    assert [entry.availability_state for entry in manifest.files] == [
        "available", "failed", "skipped",
    ]
    assert manifest.availability_message == "connection lost"
    assert FileManifest(**asdict(manifest)) == manifest


def test_individual_wildcard_matches_are_recorded_separately():
    manifest = FileManifest(
        outputs=(FileEntry("/local/*.dat", "/remote/*.dat", "pending"),)
    )
    manifest = manifest.with_file_availability(
        "/local/one.dat", "available", remote_path="/remote/one.dat",
    )
    manifest = manifest.with_file_availability(
        "/local/two.dat", "failed", "I/O error", remote_path="/remote/two.dat",
    )
    assert len(manifest.outputs) == 3
    assert manifest.outputs[1].availability_state == "available"
    assert manifest.outputs[2].availability_state == "failed"
    assert FileManifest(**asdict(manifest)) == manifest

def test_merged_stdout_stderr_has_only_one_file_status():
    manifest = FileManifest(
        stdout=FileEntry("/local/combined", "/remote/combined", "pending"),
        stderr=FileEntry("/local/combined", "/remote/combined", "pending"),
    )
    manifest = manifest.with_file_availability("/local/combined", "available")
    assert len(manifest.files) == 1
    assert manifest.stdout.availability_state == "available"
    assert manifest.stderr.availability_state == "available"
    assert FileManifest(**asdict(manifest)) == manifest


def test_materialized_wildcard_members_have_independent_availability():
    """Concrete results keep source provenance but no wildcard parent state."""
    files = FileManifest(
        outputs=(
            FileEntry("/local/one.dat", "/remote/one.dat", "pending",
                      source="/local/*.dat"),
            FileEntry("/local/two.dat", "/remote/two.dat", "pending",
                      source="/local/*.dat"),
        ),
    )
    files = files.with_file_availability("/local/one.dat", "available")
    files = files.with_file_availability(
        "/local/two.dat", "failed", "transfer failed",
    )
    assert [entry.availability_state for entry in files.outputs] == [
        "available", "failed",
    ]
    assert files.availability_state == "failed"
    assert files.availability_message == "transfer failed"
    assert all(entry.source == "/local/*.dat" for entry in files.outputs)
    assert not files.has_wildcards()
    assert FileManifest(**asdict(files)) == files


def test_recovered_stream_paths_update_only_declared_stream_locations():
    manifest = FileManifest(
        inputs=(FileEntry("/local/input", "/remote/input"),),
        outputs=(FileEntry("/local/file", "/remote/file"),),
        caches=(FileEntry("/local/cache", "/remote/cache", pre_run_signature=(12, 34)),),
        stdout=FileEntry(
            "/local/stdout", "/previous/stdout",
            availability_state="pending",
        ),
        stderr=FileEntry("/local/stderr"),
    )

    updated = manifest.with_remote_stream_paths(
        stdout="/recovered/out", stderr="/recovered/err",
    )

    assert updated.stdout.remote_path == "/recovered/out"
    assert updated.stderr.remote_path == "/recovered/err"
    assert updated.stdout.local_path == manifest.stdout.local_path
    assert updated.stdout.availability_state == "pending"
    assert updated.inputs == manifest.inputs
    assert updated.outputs == manifest.outputs
    assert updated.caches == manifest.caches
    assert manifest.stdout.remote_path == "/previous/stdout"
    assert updated.with_remote_stream_paths(stdout="", stderr=None) is updated


def test_recovered_stream_paths_do_not_create_unrecorded_stream():
    manifest = FileManifest(stdout=FileEntry("/local/stdout"))

    updated = manifest.with_remote_stream_paths(
        stdout="/remote/out", stderr="/remote/unrecorded.err",
    )
    assert updated.stdout.remote_path == "/remote/out"
    assert updated.stderr is None


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "state, message, error",
    [
        ("invalid", None, "availability state"),
        ("available", "unexpected error", "availability error"),
    ],
)
def test_manifest_availability_validation_is_shared_with_file_entry(existing, state, message, error):
    manifest = FileManifest(
        outputs=(FileEntry("/local/result"),) if existing else (),
    )
    with pytest.raises(ValueError, match=error):
        manifest.with_file_availability("/local/result", state, message)


def test_submitted_uses_execution_backend_id_validation():
    execution = Execution(ExecutorEndpoint(), WorkingDirectory("/tmp"))
    assert execution.backend_id is None
    with pytest.raises(ValueError, match="backend_id cannot be empty"):
        execution.submitted("")
    bound = execution.submitted(42)
    assert execution.backend_id is None
    assert bound.backend_id == "42"
    assert bound.submitted(42) == bound
    with pytest.raises(ValueError, match="backend_id cannot be empty"):
        bound.submitted("")
    with pytest.raises(ValueError, match="Cannot replace"):
        bound.submitted("other")


def test_manifest_from_spec_preserves_distinct_roles(tmp_path):
    from types import SimpleNamespace

    spec = SimpleNamespace(
        directory=str(tmp_path), input_file="input.inp",
        input_files=("other.inp",), output_file="stdout.out",
        error_file="stderr.err", output_files=("result.dat",),
        cache_files=("restart.pot",),
    )
    files = FileManifest.from_spec(spec)
    assert files.input_files == (
        str(tmp_path / "input.inp"), str(tmp_path / "other.inp"),
    )
    assert files.output_files == (
        str(tmp_path / "stdout.out"), str(tmp_path / "stderr.err"),
        str(tmp_path / "result.dat"),
    )
    assert FileManifest(**asdict(files)) == files


@pytest.mark.parametrize("role,overlap", [
    ("output_files", ("stdout.out",)),
    ("output_files", ("result.dat", "./result.dat")),
    ("cache_files", ("cache.dat", "./cache.dat")),
    ("cache_files", ("input.inp",)),
    ("cache_files", ("result.dat",)),
])
def test_duplicate_file_declarations_are_errors(tmp_path, role, overlap):
    from types import SimpleNamespace

    spec = SimpleNamespace(
        directory=str(tmp_path), input_file="input.inp",
        input_files=("other.inp",), output_file="stdout.out",
        error_file="stderr.err", output_files=("result.dat",),
        cache_files=(),
    )
    setattr(spec, role, overlap)
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec)


@pytest.mark.parametrize("extras,stdin,expected", [
    (("input.inp",), None, ("input.inp",)),
    (("other.inp", "./other.inp"), None, ("input.inp", "other.inp")),
    (("input.inp", "other.inp"), "input.inp", ("input.inp", "other.inp")),
    (("other.inp",), "other.inp", ("input.inp", "other.inp")),
])
def test_duplicate_input_roles_share_one_staged_file(tmp_path, extras, stdin, expected):
    from types import SimpleNamespace
    spec = SimpleNamespace(
        directory=str(tmp_path), input_file="input.inp", input_files=extras,
        stdin=stdin, output_file=None, error_file=None,
        output_files=(), cache_files=(),
    )
    files = FileManifest.from_spec(spec)
    assert tuple(x.local_path for x in files.inputs) == tuple(
        str(tmp_path / name) for name in expected
    )
    assert files.primary_input.local_path == str(tmp_path / "input.inp")
    assert (None if files.stdin is None else files.stdin.local_path) == (
        None if stdin is None else str(tmp_path / stdin)
    )
    assert FileManifest(**asdict(files)) == files


def test_only_stdout_and_stderr_may_share_a_file():
    files = FileManifest(
        stdout=FileEntry("/local/shared", "/remote/shared"),
        stderr=FileEntry("/local/shared", "/remote/shared"),
    )
    assert len(files.files) == 1
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            outputs=(FileEntry("/local/shared", "/remote/shared"),),
            stdout=FileEntry("/local/shared", "/remote/shared"),
        )


def test_availability_enum_is_shared_between_model_and_manager():
    from ase2sprkkr.execution import FileAvailabilityState
    from ase2sprkkr.execution.manager import FileAvailabilityState as ManagerState
    from ase2sprkkr.execution.run_model import FileAvailabilityState as ModelState

    assert ModelState is ManagerState is FileAvailabilityState
    assert {state.value for state in ModelState} == {
        "pending", "available", "failed", "skipped",
    }


@pytest.mark.parametrize("state", list(FileAvailabilityState))
def test_file_entry_accepts_enum_but_serializes_status_as_string(state):
    entry = FileEntry("/local/file", availability_state=state)
    assert type(entry.availability_state) is str
    assert entry.availability_state == state.value
    assert asdict(entry)["availability_state"] == state.value
    assert FileEntry(**asdict(entry)) == entry


def test_invalid_file_availability_is_rejected():
    with pytest.raises(ValueError, match="availability"):
        FileEntry("/local/file", availability_state="new-state")
    with pytest.raises(ValueError, match="error belongs"):
        FileEntry(
            "/local/file",
            availability_state=FileAvailabilityState.PENDING,
            availability_message="error",
        )


def test_file_sync_policy_validation_uses_the_runtime_enum():
    from ase2sprkkr.execution.remote import FileSyncPolicy as RemoteFileSyncPolicy

    assert RemoteFileSyncPolicy is FileSyncPolicy
    for input_policy in FileSyncPolicy:
        for output_policy in FileSyncPolicy:
            mapping = FileMapping("/local", "/remote", input_policy, output_policy)
            assert type(mapping.input_sync) is str
            assert type(mapping.output_sync) is str
            assert mapping.input_sync == input_policy.value
            assert mapping.output_sync == output_policy.value
            assert FileMapping(**asdict(mapping)) == mapping

    with pytest.raises(ValueError, match="synchronization policy"):
        FileMapping("/local", "/remote", input_sync="unsupported")
    with pytest.raises(ValueError, match="synchronization policy"):
        FileMapping("/local", "/remote", output_sync="unsupported")


@pytest.mark.parametrize("name", ("input_sync", "output_sync"))
@pytest.mark.parametrize("policy", list(FileSyncPolicy))
def test_launch_sync_policies_are_validated_and_serialized(name, policy):
    launch = LaunchSpec(**{name: policy})
    assert type(getattr(launch, name)) is str
    assert getattr(launch, name) == policy.value
    assert LaunchSpec(**asdict(launch)) == launch


@pytest.mark.parametrize("name", ("input_sync", "output_sync"))
def test_launch_rejects_unknown_sync_policies(name):
    with pytest.raises(ValueError, match="synchronization policy"):
        LaunchSpec(**{name: "unsupported"})


def test_manifest_accepts_output_that_is_also_stdout_or_stderr():
    shared = FileEntry("/local/run.out", "/remote/run.out")
    manifest = FileManifest(stdout=shared, outputs=(shared,))
    assert manifest.stdout == manifest.outputs[0]
    assert len(manifest.files) == 1

    errors = FileEntry("/local/run.err", "/remote/run.err")
    manifest = FileManifest(stderr=errors, outputs=(errors,))
    assert manifest.stderr == manifest.outputs[0]


def test_manifest_rejects_stream_output_remote_source_disagreement():
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            stdout=FileEntry("/local/run.out", "/remote/run.out"),
            outputs=(FileEntry("/local/run.out", "/remote/another.out"),),
        )



def test_optional_taskfile_survives_manifest_and_roundtrip(tmp_path):
    from ase2sprkkr.execution.base import ExecutionSpec
    from ase2sprkkr.input_parameters.task_definition import TaskFile

    optional_output = TaskFile(tmp_path / "maybe-*.dat", optional=True)
    required_output = TaskFile(tmp_path / "required.dat")
    optional_cache = TaskFile(tmp_path / "restart.pot", optional=True)
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path), input_file=None,
        output_file=None, output_files=(optional_output, required_output),
        cache_files=(optional_cache,),
    )
    manifest = FileManifest.from_spec(spec)
    assert [(item.local_path, item.optional) for item in manifest.outputs] == [
        (str(optional_output), True), (str(required_output), False),
    ]
    assert manifest.caches[0].optional is True
    assert FileManifest(**asdict(manifest)) == manifest
    with pytest.raises(TypeError, match="optional"):
        FileEntry(str(tmp_path / "wrong"), optional="true")


def test_manifest_allows_one_remote_output_downloaded_to_distinct_local_paths():
    shared_source = "/remote/result.out"
    manifest = FileManifest(outputs=(
        FileEntry("/local/a.out", shared_source),
        FileEntry("/local/b.out", shared_source),
    ))
    assert len(manifest.outputs) == 2


def test_manifest_rejects_two_remote_outputs_claiming_one_local_destination():
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(outputs=(
            FileEntry("/local/result.out", "/remote/first.out"),
            FileEntry("/local/result.out", "/remote/second.out"),
        ))
