import asyncio

from ..base import SubmissionNotStarted
from ..remote import FileSyncPolicy
from ..transport import LocalTransport
from .test_remote import _processed_run, _text_run


def test_local_transport_put_files_expands_patterns_and_preserves_paths(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "a").mkdir(parents=True)
    (source / "b").mkdir()
    (source / "a" / "one.dat").write_text("one")
    (source / "b" / "two.dat").write_text("two")
    (source / "a" / "ignored.txt").write_text("ignored")

    asyncio.run(
        LocalTransport().put_files(
            (source / "a" / "*.dat", source / "b" / "*.dat"),
            source,
            destination,
        )
    )

    assert (destination / "a" / "one.dat").read_text() == "one"
    assert (destination / "b" / "two.dat").read_text() == "two"
    assert not (destination / "a" / "ignored.txt").exists()


def test_local_transport_get_files_expands_sources(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "one.out").write_text("one")
    (source / "nested" / "two.out").write_text("two")

    asyncio.run(
        LocalTransport().get_files(
            source / "nested" / "*.out",
            source,
            destination,
        )
    )

    assert (destination / "nested" / "one.out").read_text() == "one"
    assert (destination / "nested" / "two.out").read_text() == "two"


def test_remote_executor_expands_input_wildcards(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    workdir.mkdir(parents=True)
    (workdir / "file-a.pot").write_text("a")
    (workdir / "file-b.pot").write_text("b")
    (workdir / "ignored.txt").write_text("ignored")

    executor, spec, _, remote_root = _text_run(
        tmp_path,
        declared_input_files=(workdir / "file*.pot",),
        code=(
            "import pathlib; "
            "print(':'.join(pathlib.Path(name).read_text() "
            "for name in ('file-a.pot', 'file-b.pot')))"
        ),
    )
    result = _processed_run(executor, spec).run()

    assert result.output == b"a:b\n"
    remote_workdir = remote_root / "calculation"
    assert (remote_workdir / "file-a.pot").read_text() == "a"
    assert (remote_workdir / "file-b.pot").read_text() == "b"
    assert not (remote_workdir / "ignored.txt").exists()


def test_remote_executor_rejects_unmatched_input_wildcard(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    executor, spec, _, _ = _text_run(
        tmp_path,
        declared_input_files=(workdir / "missing*.pot",),
    )

    try:
        _processed_run(executor, spec).run()
    except SubmissionNotStarted as exc:
        assert "input pattern did not match" in str(exc)
    else:
        raise AssertionError("unmatched input wildcard was accepted")


def test_remote_executor_downloads_output_wildcards_and_discards_stale_matches(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    remote_workdir = tmp_path / "remote" / "calculation"
    remote_workdir.mkdir(parents=True)
    (remote_workdir / "file-stale.dat").write_text("stale")

    executor, spec, _, _ = _text_run(
        tmp_path,
        declared_output_files=(workdir / "file-*.dat",),
        code=(
            "import pathlib; "
            "pathlib.Path('file-a.dat').write_text('a'); "
            "pathlib.Path('file-b.dat').write_text('b'); print('done')"
        ),
    )
    result = _processed_run(executor, spec).run()

    assert result.return_code == 0
    assert (workdir / "file-a.dat").read_text() == "a"
    assert (workdir / "file-b.dat").read_text() == "b"
    assert not (workdir / "file-stale.dat").exists()
    assert not (remote_workdir / "file-stale.dat").exists()


def test_remote_executor_syncs_cache_wildcards_in_both_directions(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    workdir.mkdir(parents=True)
    (workdir / "cache-a.dat").write_text("old")

    executor, spec, _, remote_root = _text_run(
        tmp_path,
        declared_cache_files=(workdir / "cache-*.dat",),
        code=(
            "import pathlib; "
            "a=pathlib.Path('cache-a.dat'); assert a.read_text() == 'old'; "
            "a.write_text('new'); pathlib.Path('cache-b.dat').write_text('created'); "
            "print('done')"
        ),
    )
    result = _processed_run(executor, spec).run()

    assert result.return_code == 0
    assert (workdir / "cache-a.dat").read_text() == "new"
    assert (workdir / "cache-b.dat").read_text() == "created"
    assert (remote_root / "calculation" / "cache-a.dat").read_text() == "new"
    assert (remote_root / "calculation" / "cache-b.dat").read_text() == "created"


def test_exact_and_wildcard_outputs_share_availability_tracking(tmp_path):
    from .test_remote import _SpyTransport

    transport = _SpyTransport()
    workdir = tmp_path / "local" / "calculation"
    exact = workdir / "bands.dat"
    overlap = workdir / "part-a.dat"
    pattern = workdir / "part-*.dat"
    executor, spec, _, remote_root = _text_run(
        tmp_path, transport,
        declared_output_files=(exact, overlap, pattern),
        code=(
            "from pathlib import Path; "
            "Path('bands.dat').write_text('exact'); "
            "Path('part-a.dat').write_text('a'); "
            "Path('part-b.dat').write_text('b'); "
            "Path('part-c.dat').write_text('c')"
        ),
    )
    runner = _processed_run(executor, spec)
    runner.run()

    entries = {entry.local_path: entry for entry in runner.job_files.snapshot().outputs}
    assert exact.read_text() == "exact"
    assert overlap.read_text() == "a"
    for name, expected in (("part-b.dat", "b"), ("part-c.dat", "c")):
        path = workdir / name
        assert path.read_text() == expected
        assert entries[str(path)].availability_state == "available"
        assert entries[str(path)].source == str(pattern)

    assert entries[str(exact)].availability_state == "available"
    assert entries[str(overlap)].availability_state == "available"

    remote_workdir = remote_root / "calculation"
    for name in ("bands.dat", "part-a.dat", "part-b.dat", "part-c.dat"):
        assert sum(
            src == remote_workdir / name for src, _ in transport.get_files
        ) == 1


def test_exact_and_wildcard_caches_keep_individual_sync_states(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    workdir.mkdir(parents=True)
    exact = workdir / "restart.dat"
    unchanged = workdir / "generated-base.dat"
    pattern = workdir / "generated-*.dat"
    exact.write_text("before")
    unchanged.write_text("unchanged")

    executor, spec, _, _ = _text_run(
        tmp_path,
        declared_cache_files=(exact, pattern),
        output_files=FileSyncPolicy.MISSING,
        code=(
            "from pathlib import Path; "
            "Path('restart.dat').write_text('after'); "
            "Path('generated-new.dat').write_text('new')"
        ),
    )
    runner = _processed_run(executor, spec)
    runner.run()

    entries = {entry.local_path: entry for entry in runner.job_files.snapshot().caches}
    assert exact.read_text() == "before"
    assert unchanged.read_text() == "unchanged"
    assert (workdir / "generated-new.dat").read_text() == "new"
    assert entries[str(exact)].availability_state == "skipped"
    assert entries[str(unchanged)].availability_state == "skipped"
    new_entry = entries[str(workdir / "generated-new.dat")]
    assert new_entry.availability_state == "available"
    assert new_entry.source == str(pattern)


def test_remote_executor_tolerates_unmatched_optional_cache_wildcard(tmp_path):
    workdir = tmp_path / "local" / "calculation"
    cache_pattern = workdir / "not-present-*.pot"
    executor, spec, _, remote_root = _text_run(
        tmp_path,
        declared_cache_files=(cache_pattern,),
        code="print('done')",
    )
    runner = _processed_run(executor, spec)
    result = runner.run()

    assert result.return_code == 0
    assert result.output == b"done\n"
    assert not tuple(workdir.glob("not-present-*.pot"))
    assert not tuple((remote_root / "calculation").glob("not-present-*.pot"))


def test_local_transfer_batches_use_their_directional_file_primitives(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "result.dat").write_text("data")

    class TracedTransport(LocalTransport):
        def __init__(self):
            self.operations = []

        async def put_file(self, src, dst):
            self.operations.append("put")
            await super().put_file(src, dst)

        async def get_file(self, src, dst):
            self.operations.append("get")
            await super().get_file(src, dst)

    transport = TracedTransport()
    async def check():
        await transport.put_files("nested/*.dat", source, dest)
        assert transport.operations == ["put"]
        transport.operations.clear()
        await transport.get_files("nested/*.dat", source, dest)
        assert transport.operations == ["get", "put"]

    asyncio.run(check())
    assert (dest / "nested" / "result.dat").read_text() == "data"


def test_overlapping_exact_and_wildcard_caches_are_transferred_once(tmp_path):
    from .test_remote import _SpyTransport

    transport = _SpyTransport()
    workdir = tmp_path / "local" / "calculation"
    workdir.mkdir(parents=True)
    exact = workdir / "restart-a.dat"
    exact.write_text("old")
    pattern = workdir / "restart-*.dat"
    executor, spec, _, remote_root = _text_run(
        tmp_path, transport,
        declared_cache_files=(exact, pattern),
        code=(
            "from pathlib import Path; "
            "Path('restart-a.dat').write_text('new'); "
            "Path('restart-b.dat').write_text('created')"
        ),
    )
    runner = _processed_run(executor, spec)
    runner.run()
    remote_a = remote_root / "calculation" / "restart-a.dat"
    remote_b = remote_root / "calculation" / "restart-b.dat"
    assert sum(src == remote_a for src, _ in transport.get_files) == 1
    assert sum(src == remote_b for src, _ in transport.get_files) == 1
    entries = {item.local_path: item for item in runner.job_files.snapshot().caches}
    assert entries[str(exact)].availability_state == "available"
    assert exact.read_text() == "new"


def test_remote_completed_run_keeps_missing_optional_wildcard_skipped(tmp_path):
    from .test_taskfile_optional_contract import DeclaredPath

    workdir = tmp_path / "local" / "calculation"
    executor, spec, _, _ = _text_run(
        tmp_path,
        declared_output_files=(
            DeclaredPath(str(workdir / "optional-*.dat"), optional=True),
        ),
        code="print('done')",
    )
    runner = _processed_run(executor, spec)
    result = runner.run()
    assert result.return_code == 0
    missing = [
        e for e in runner.job_files.snapshot().outputs
        if e.local_path == str(workdir / "optional-*.dat")
    ]
    assert len(missing) == 1
    assert missing[0].availability_state == "skipped"


def test_remote_completed_run_fails_on_missing_mandatory_wildcard(tmp_path):
    import pytest

    workdir = tmp_path / "local" / "calculation"
    executor, spec, _, _ = _text_run(
        tmp_path,
        declared_output_files=(workdir / "required-*.dat",),
        code="print('done')",
    )
    runner = _processed_run(executor, spec)
    with pytest.raises((FileNotFoundError, RuntimeError),
                       match="required-\\*\\.dat|remote output pattern"):
        runner.run()
    missing = [
        e for e in runner.job_files.snapshot().outputs
        if e.local_path == str(workdir / "required-*.dat")
    ]
    assert len(missing) == 1
    assert missing[0].availability_state == "failed"


def test_overlapping_required_and_optional_output_patterns_download_once(tmp_path):
    """Overlapping globs do not duplicate physical downloads."""
    from .test_remote import _SpyTransport
    from .test_taskfile_optional_contract import DeclaredPath

    transport = _SpyTransport()
    workdir = tmp_path / "local" / "calculation"
    executor, spec, _, remote_root = _text_run(
        tmp_path, transport,
        declared_output_files=(
            DeclaredPath(str(workdir / "part-*.dat"), optional=True),
            DeclaredPath(str(workdir / "part-a*.dat")),
        ),
        output_files=FileSyncPolicy.ALWAYS,
        code=(
            "from pathlib import Path; "
            "Path('part-a.dat').write_text('a'); "
            "Path('part-b.dat').write_text('b')"
        ),
    )
    runner = _processed_run(executor, spec)
    assert runner.run().return_code == 0
    assert (workdir / "part-a.dat").read_text() == "a"
    assert (workdir / "part-b.dat").read_text() == "b"
    for name in ("part-a.dat", "part-b.dat"):
        assert sum(src == remote_root / "calculation" / name
                   for src, _ in transport.get_files) == 1
