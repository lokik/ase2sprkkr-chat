"""Scriptable result-file operations must not depend on a Qt application."""
import shutil
import subprocess
from pathlib import Path
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
    assert option.actions() == ("open_directory",)
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
    assert all("stderr" not in kwargs for _, kwargs in calls)


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


def test_actions_on_spc_with_unknown_task_keep_generic_file_operations(tmp_path):
    result = TaskResult(None, None, str(tmp_path))
    path = tmp_path / "unknown.spc"
    path.write_text("SPC data", encoding="utf-8")
    option = result.files.add_file("spectrum", str(path), "spc")

    assert "open" in option.actions()
    assert "copy_to" in option.actions()
    assert "export_nexus" not in option.actions()


def test_directory_action_survives_missing_file_but_not_missing_directory(tmp_path):
    option, path = _file(tmp_path)
    path.unlink()
    assert option.actions() == ("open_directory",)
    tmp_path.rmdir()
    assert option.actions() == ()


def test_export_nexus_prefers_owned_parameters_and_respects_overrides(
    tmp_path, monkeypatch,
):
    from ase2sprkkr.input_parameters.input_parameters import InputParameters

    params = InputParameters.create("ARPES")
    result = TaskResult(params, None, str(tmp_path), task_name="ARPES")
    source = tmp_path / "spectrum.spc"
    source.write_text("not parsed by mock", encoding="utf-8")
    option = result.files.add_file("SPC", str(source), "spc")
    calls = []
    monkeypatch.setattr(
        OutputFileOption, "parsed_file",
        lambda self: SimpleNamespace(
            to_nexus=lambda filename, **kw: calls.append((filename, kw))
        ),
    )
    destination = tmp_path / "target.nxs"
    option.export_nexus(destination)
    assert calls == [(destination, {
        "discover_input": False, "input_parameters": params,
    })]

    other = InputParameters.create("ARPES")
    option.export_nexus(
        destination, input_parameters=other, discover_input=True,
    )
    assert calls[-1] == (destination, {
        "discover_input": True, "input_parameters": other,
    })


def test_export_nexus_uses_verified_registered_input_without_guessing(
    tmp_path, monkeypatch,
):
    source = tmp_path / "spectrum.spc"
    source.write_text("SPC data", encoding="utf-8")
    original = tmp_path / "input.inp"
    original.write_text("INPUT DATA", encoding="utf-8")
    result = TaskResult(
        None, None, str(tmp_path), input_file=str(original),
        task_name="ARPES",
    )
    option = result.files.add_file("SPC", str(source), "spc")
    calls = []
    monkeypatch.setattr(
        OutputFileOption, "parsed_file",
        lambda self: SimpleNamespace(
            to_nexus=lambda filename, **kw: calls.append(kw)
        ),
    )
    option.export_nexus(tmp_path / "first.nxs")
    assert calls[-1] == {
        "discover_input": False, "input_file": str(original),
    }

    # A neighboring file must not be silently read after the registered
    # historical input was proven to be from a different version.
    result.files["input"].set_association(
        FileAssociation.VERIFIED,
        recorded_version=FileVersion.capture(original, require_digest=True),
    )
    original.write_text("REPLACED INPUT", encoding="utf-8")
    option.export_nexus(tmp_path / "second.nxs")
    assert calls[-1] == {"discover_input": False}


def test_export_nexus_real_spc_contains_owned_task_parameters(tmp_path):
    h5py = pytest.importorskip("h5py")
    from ase2sprkkr.input_parameters.input_parameters import InputParameters

    source = (
        Path(__file__).resolve().parents[2]
        / "result_files" / "examples" / "Au_kx_ky_ARPES_data.spc"
    )
    spc_path = tmp_path / source.name
    shutil.copy2(source, spc_path)
    # This sibling must not override authoritative in-memory parameters.
    (tmp_path / "Au_kx_ky_ARPES.inp").write_text(
        "INVALID FOREIGN INPUT", encoding="utf-8",
    )
    params = InputParameters.create("ARPES")
    result = TaskResult(params, None, str(tmp_path), task_name="ARPES")
    option = result.files.add_file("SPC", str(spc_path), "spc")
    destination = tmp_path / "export.nxs"

    assert option.export_nexus(destination) == destination
    with h5py.File(destination) as exported:
        text = exported["entry/calculation/parameters/sprkkr_input"].asstr()[()]
        assert text == params.to_string(validate=False)
        assert "original_input" in exported["entry/calculation"]
