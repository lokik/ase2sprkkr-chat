import os

from ase2sprkkr.input_parameters.input_parameters import InputParameters


DATASET = "manifest_test"


def _names(task, tmp_path):
    parameters = InputParameters.create(task)
    parameters.CONTROL.DATASET = DATASET
    parameters.CONTROL.ADSI = ""
    return parameters, {
        os.path.basename(os.fspath(path))
        for path in parameters.output_files(tmp_path)
    }


def test_jalf_and_jorb_native_names(tmp_path):
    _, jalf = _names("JALF", tmp_path)
    assert {f"{DATASET}_JALF.dat", "curdns1", "curdns2", "curdns3"} <= jalf
    assert f"{DATASET}0_JALF.dat" not in jalf

    _, jorb = _names("JORB", tmp_path)
    assert {
        f"{DATASET}_JORB.dat",
        f"{DATASET}_JORB_UP.dat",
        f"{DATASET}_JORB_DN.dat",
    } <= jorb
    assert f"{DATASET}0_JORB.dat" not in jorb


def test_dmi_outputs(tmp_path):
    _, names = _names("DMI", tmp_path)
    assert {
        f"{DATASET}_DMIVEC_Dij.dat",
        f"{DATASET}_DMIVEC_Dij_[xyz]_*.agr",
    } <= names
