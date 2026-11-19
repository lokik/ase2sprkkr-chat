"""Native filename identifiers must be safe in SPR-KKR shell commands."""

import pytest
import pyparsing as pp

from ..input_parameters import InputParameters
from ...common.warnings import DataValidityError


@pytest.mark.parametrize("value", [
    "-3", "~Fe", "Fe$(id)", "Fe`id`", "Fe|id", "Fe&id", "Fe>out",
    "Fe<in", "Fe;id", "Fe*", "Fe?", "Fe\\name", "Fe name", "Fe\nname",
    "'Fe'", '"Fe"', "", "Fe#comment", "Fe=value", "Fe[1]", "Fe(1)", "Fe{1}",
])
def test_reject_unsafe_dataset_without_changing_existing_value(value):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.DATASET = "Fe"
    with pytest.raises(DataValidityError):
        parameters.CONTROL.DATASET = value
    assert parameters.CONTROL.DATASET() == "Fe"


@pytest.mark.parametrize("value", [
    "Fe-3", "_Fe", ".Fe", "dir/Fe-3", "../Fe", "/tmp/Fe", "dir/-3", "./-3",
    "Fe+Co@50%!:^~", "Železo", ".", "..", "dir/",
])
def test_dataset_keeps_safe_prefixes_and_paths(value):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.DATASET = value
    assert parameters.CONTROL.DATASET() == value


@pytest.mark.parametrize("value", [
    "$(id)", "`id`", "a|b", "a&b", "a>b", "a<b", "a;b",
    "a*b", "a?b", "a\\b", "a b", "a\nb", "'a'", '"a"', "a#b", "a=b", "a[b]", "a(b)", "a{b}",
])
def test_reject_unsafe_adsi(value):
    parameters = InputParameters.create("SCF")
    with pytest.raises(DataValidityError):
        parameters.CONTROL.ADSI = value
    assert parameters.CONTROL.ADSI() == "SCF"


@pytest.mark.parametrize("value", ["", "SCF", "-3", "relaxed_2", "v1.2", "labels/-run+@%!:^~", "běh", "~run"])
def test_adsi_keeps_safe_suffixes_and_empty_value(value):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.ADSI = value
    assert parameters.CONTROL.ADSI() == value


@pytest.mark.parametrize("name,value", [("DATASET", "-3"), ("DATASET", "Fe$(id)"), ("ADSI", "bad*name")])
def test_native_input_parser_rejects_unsafe_identifiers(name, value):
    parameters = InputParameters.create("SCF")
    text = "CONTROL DATASET=Fe POTFIL=Fe.pot\nTASK SCF\n"
    if name == "DATASET":
        text = text.replace("DATASET=Fe", f"DATASET={value}")
    else:
        text = text.replace("POTFIL=Fe.pot", f"POTFIL=Fe.pot ADSI={value}")
    with pytest.raises((DataValidityError, ValueError, pp.ParseBaseException)):
        parameters._definition.read_from_string(text)


@pytest.mark.parametrize("dataset,adsi", [
    ("sub/-Fe+@%!:^~", "SCF"), ("Fe", "labels/-run+@%!:^~"), ("Železo", "běh"),
])
def test_native_identifiers_round_trip(dataset, adsi, tmp_path):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.DATASET = dataset
    parameters.CONTROL.ADSI = adsi
    parameters.CONTROL.POTFIL = "Fe.pot"
    path = tmp_path / "run.inp"
    parameters.save_to_file(path)
    loaded = InputParameters.from_file(path)
    assert loaded.CONTROL.DATASET() == dataset
    assert loaded.CONTROL.ADSI() == adsi


def test_dataset_validation_checks_the_serialized_relative_path(tmp_path):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.DATASET = str(tmp_path / "-3")
    parameters.directory = str(tmp_path)
    with pytest.raises(DataValidityError, match="hyphen"):
        parameters.CONTROL.DATASET.validate()


@pytest.mark.parametrize("task", ["TDOBS", "COMPTON", "POSANI"])
def test_absolute_utf8_dataset_keeps_implicit_tau_directory_valid(task, tmp_path):
    parameters = InputParameters.create(task)
    parameters.directory = str(tmp_path)
    parameters.CONTROL.DATASET = str(tmp_path / "Železo/Fe")
    assert parameters.TAU.TAUDIR().endswith(f"Železo/Fe_{task}.tau.dir")
    parameters.TAU.TAUDIR.validate()
