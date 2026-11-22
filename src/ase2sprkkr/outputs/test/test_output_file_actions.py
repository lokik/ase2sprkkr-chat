"""Scriptable result-file operations must not depend on a Qt application."""
import subprocess
from types import SimpleNamespace

import pytest
from kkrunner.files import FileVersion
from kkrunner.runs import HistoricalFileMismatchError

from ase2sprkkr.outputs.result_options import FileAssociation, OutputFileOption
from ase2sprkkr.outputs.task_result import TaskResult


def _file(tmp_path):
    path = tmp_path / "result.dat"
    path.write_text("original", encoding="utf-8")
    result = TaskResult(None, None, str(tmp_path), task_name="DOS")
    option = result.files.add_file("extra", str(path))
    return option, path


def test_copy_to_has_no_dialog_and_preserves_file_metadata(tmp_path):
    option, path = _file(tmp_path)
    destination = tmp_path / "copy.dat"

    assert {"open", "open_directory", "copy_to"} <= set(option.actions())
    assert "save" not in option.actions()
    assert option.copy_to(destination) == destination
    assert destination.read_bytes() == path.read_bytes()
    assert destination.stat().st_mtime_ns == path.stat().st_mtime_ns


def test_copy_and_open_reject_overwritten_historical_file(tmp_path, monkeypatch):
    option, path = _file(tmp_path)
    option.set_association(
        FileAssociation.VERIFIED,
        recorded_version=FileVersion.capture(path, require_digest=True),
    )
    path.write_text("new contents", encoding="utf-8")
    assert option.actions() == ()
    opened = []
    monkeypatch.setattr(
        "ase2sprkkr.outputs.result_options._open_system_path", opened.append,
    )

    with pytest.raises(HistoricalFileMismatchError):
        option.copy_to(tmp_path / "copy.dat")
    with pytest.raises(HistoricalFileMismatchError):
        option.open()
    with pytest.raises(HistoricalFileMismatchError):
        option.export_nexus(tmp_path / "archive.nxs")
    assert opened == []
    assert not (tmp_path / "copy.dat").exists()


def test_open_and_open_directory_launch_nonblocking_system_opener(
    tmp_path, monkeypatch,
):
    option, path = _file(tmp_path)
    calls = []
    monkeypatch.setattr(
        "ase2sprkkr.outputs.result_options.platform.system", lambda: "Linux",
    )
    monkeypatch.setattr(
        "ase2sprkkr.outputs.result_options.subprocess.Popen",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )

    option.open()
    path.unlink()  # The containing directory is still useful to inspect.
    option.open_directory()

    assert [argv for argv, _ in calls] == [
        ["xdg-open", str(path)],
        ["xdg-open", str(tmp_path)],
    ]
    assert all(kwargs["start_new_session"] for _, kwargs in calls)
    assert all(kwargs["stdout"] == subprocess.DEVNULL for _, kwargs in calls)


def test_plot_forwards_options_to_native_plotter(tmp_path, monkeypatch):
    option, _ = _file(tmp_path)
    calls = []
    monkeypatch.setattr(
        OutputFileOption, "parsed_file",
        lambda self: SimpleNamespace(plot=lambda **kw: calls.append(kw) or "plot"),
    )

    assert option.plot(show=False, color="black") == "plot"
    assert calls == [{"show": False, "color": "black"}]


def test_nexus_export_delegates_destination_and_metadata(tmp_path, monkeypatch):
    option, _ = _file(tmp_path)
    calls = []

    def write(destination, **kwargs):
        calls.append((destination, kwargs))
        return destination

    monkeypatch.setattr(
        OutputFileOption, "parsed_file",
        lambda self: SimpleNamespace(to_nexus=write),
    )
    destination = tmp_path / "data.nxs"
    assert option.export_nexus(destination, metadata={"source": "test"}) == destination
    assert calls == [(destination, {"metadata": {"source": "test"}})]

    monkeypatch.setattr(
        OutputFileOption, "parsed_file", lambda self: SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="NeXus export is unavailable"):
        option.export_nexus(destination)
