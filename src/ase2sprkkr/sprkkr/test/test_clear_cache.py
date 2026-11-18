"""Cache invalidation affects the run manifest, never the task registry."""

from kkrunner.files import TaskFile

from ..calculation import _run_cache_files


class _Parameters:
    def output_files(self, directory):
        return (f"{directory}/result.dat",)

    def cache_files(self, directory):
        return (
            TaskFile(f"{directory}/tau.dir", kind="directory"),
            TaskFile(f"{directory}/table.dat", optional=False),
        )


def test_reuse_retains_cache_declarations():
    outputs, caches = _run_cache_files(_Parameters(), "/tmp/work")
    assert outputs == ("/tmp/work/result.dat",)
    assert len(caches) == 2
    assert caches[0].kind == "directory"


def test_clear_cache_promotes_caches_to_optional_outputs():
    parameters = _Parameters()
    outputs, caches = _run_cache_files(parameters, "/tmp/work", clear_cache=True)
    assert caches == ()
    assert len(outputs) == 3
    assert outputs[0] == "/tmp/work/result.dat"
    assert all(item.optional for item in outputs[1:])
    assert outputs[1].kind == "directory"
    assert outputs[2].kind == "file"
    assert len(parameters.cache_files("/tmp/work")) == 2
