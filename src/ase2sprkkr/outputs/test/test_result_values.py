"""Scientifically meaningful result tabs are selected by task metadata."""
from ase2sprkkr.input_parameters.input_parameters import InputParameters
from ase2sprkkr.outputs.task_result import TaskResult


def _result(parameters, directory):
    return TaskResult(
        parameters, None, str(directory), task_name=parameters.task_name
    )


def test_scf_potential_and_command_log_remain_managed_but_not_result_tabs(tmp_path):
    p = InputParameters.create("SCF")
    p.CONTROL.POTFIL = "Fe.pot"
    result = _result(p, tmp_path)
    result.files.add_file("potential-new", "Fe.pot_new")
    result.files.add_file("output", "Fe_SCF.out")

    assert "potential-new" in result.output_values
    assert "output" in result.output_values
    assert result.result_values == {}


def test_dos_data_is_presented_but_log_and_input_are_not(tmp_path):
    p = InputParameters.create("DOS")
    p.CONTROL.DATASET = "Fe"
    result = _result(p, tmp_path)
    result.files.add_file("DOS", "Fe_DOS.dos", "dos")
    result.files.add_file("output", "Fe_DOS.out")
    result.files.add_file("input", "Fe_DOS.inp")

    assert list(result.result_values) == ["DOS"]
    assert set(result.output_values) == {"DOS", "output", "input"}


def test_jxc_result_patterns_include_multiple_agr_outputs(tmp_path):
    p = InputParameters.create("JXC")
    p.CONTROL.DATASET = "Fe"
    p.MODE.MODE = "FREL"
    result = _result(p, tmp_path)
    result.files.add_file("jxc", "Fe_JXC_XCPLTEN_Jij.dat", "jxc")
    result.files.add_file("plot1", "Fe_JXC_XCPLTEN_Jij_x.agr", "agr")
    result.files.add_file("plot2", "Fe_JXC_XCPLTEN_Jij_y.agr", "agr")
    result.files.add_file("unrelated", "Fe_JXC_tmp.dat")

    assert set(result.result_values) == {"jxc", "plot1", "plot2"}
    assert "unrelated" in result.output_values


def test_result_values_deduplicates_parser_aliases_for_same_output(tmp_path):
    p = InputParameters.create("DOS")
    p.CONTROL.DATASET = "Fe"
    result = _result(p, tmp_path)
    result.files.add_file("DOS", "Fe_DOS.dos", "dos")
    result.files.add_file("alias", "Fe_DOS.dos", "dos")
    assert len(result.output_values) == 2
    assert len(result.result_values) == 1
    assert next(iter(result.result_values)) == "DOS"
