"""Large modern outputs need only a header probe to select the parser."""

from ase2sprkkr.outputs.reader_registry import reader_registry


def test_reader_detection_uses_only_header_for_native_task(monkeypatch):
    import builtins
    import importlib

    module = importlib.import_module("ase2sprkkr.outputs.reader_registry")
    sizes = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            sizes.append(size)
            assert size > 0, "Modern task detection must not read the full OUT"
            return b"TASK = SCF\n" + b" " * 1024

    monkeypatch.setattr(module, "open", lambda *_a, **_kw: Stream(), raising=False)
    assert reader_registry.detect("big.out") == "SCF"
    assert sizes == [module._TASK_HEADER_BYTES]


def test_reader_detection_falls_back_to_complete_output_for_late_task(tmp_path):
    output = tmp_path / "legacy.out"
    output.write_bytes(b" " * (256 * 1024 + 1) + b"\nTASK = DOS\n")
    assert reader_registry.detect(output) == "DOS"
