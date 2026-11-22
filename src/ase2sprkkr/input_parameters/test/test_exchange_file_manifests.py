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


def test_jxc_outputs_without_adsi_follow_relativistic_mode(tmp_path):
    # An explicitly empty ADSI produces different paths from the default
    # JXC names. Keep this independent contract while reusing the setup.
    parameters, names = _names("JXC", tmp_path)
    assert {
        f"{DATASET}_XCPLTEN_Jij.dat",
        f"{DATASET}_XCPLTEN_Jij_*.agr",
        f"{DATASET}_XCPLTEN_Dij.dat",
        f"{DATASET}_XCPLTEN_Dij_[xyz]_*.agr",
    } <= names

    parameters.MODE.MODE = "SP-SREL"
    names = {
        os.path.basename(os.fspath(path))
        for path in parameters.output_files(tmp_path)
    }
    assert {f"{DATASET}_J_ij.dat", f"{DATASET}_Jij_*.agr"} <= names
    assert f"{DATASET}_XCPLTEN_Jij.dat" not in names


def test_dmi_outputs(tmp_path):
    _, names = _names("DMI", tmp_path)
    assert {
        f"{DATASET}_DMIVEC_Dij.dat",
        f"{DATASET}_DMIVEC_Dij_[xyz]_*.agr",
    } <= names
