from ase2sprkkr.input_parameters.input_parameters import InputParameters


def test_dos_output_is_namespaced_by_dataset(tmp_path):
    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "Fe"

    assert tuple(map(str, parameters.output_files(str(tmp_path)))) == (str(tmp_path / "Fe_DOS.dos"),)
    assert parameters.output_files(str(tmp_path))[0].type == "DOS"


def test_scf_declares_new_potential_as_output(tmp_path):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "Fe.pot"

    assert tuple(map(str, parameters.output_files(str(tmp_path)))) == (
        str(tmp_path / "Fe.pot_new"), str(tmp_path / "*_SCFSTART.dos"),
    )


def test_scf_fullpot_shape_function_is_optional_cache(tmp_path):
    parameters = InputParameters.create("SCF")
    parameters.SCF.FULLPOT = True
    parameters.CONTROL.SFNFIL = "Fe.sfn"

    assert parameters.cache_files(str(tmp_path)) == (str(tmp_path / "Fe.sfn"),)


def test_xas_reuses_dataset_tau_directory_as_optional_cache(tmp_path):
    parameters = InputParameters.create("XAS")
    parameters.CONTROL.DATASET = "Fe"

    caches = parameters.cache_files(str(tmp_path))
    assert tuple(map(str, caches)) == (str(tmp_path / "Fe_XAS.tau.dir"),)
    assert caches[0].kind == "directory"


def test_bsfek_reuses_dataset_tau_directory_as_optional_cache(tmp_path):
    parameters = InputParameters.create("BSFEK")
    parameters.CONTROL.DATASET = "Fe"

    caches = parameters.cache_files(str(tmp_path))
    assert tuple(map(str, caches)) == (str(tmp_path / "Fe_BLOCHSF.tau.dir"),)
    assert caches[0].kind == "directory"


def test_gilbert_fluctuation_file_is_required_for_m_t(tmp_path):
    parameters = InputParameters.create("GILBERT")
    parameters.TASK.SETFLUCT = "M_T"
    parameters.TASK.FLUCTFIL = "magnetization.dat"

    assert parameters.input_files(str(tmp_path)) == (
        str(tmp_path / "magnetization.dat"),
    )


def test_phagen_outputs_follow_explicit_potential_name(tmp_path):
    parameters = InputParameters.create("PHAGEN")
    parameters.CONTROL.POTFIL = "Fe.pot"
    parameters.CONTROL.DATASET = "unrelated"

    assert parameters.output_files(str(tmp_path)) == (
        str(tmp_path / "Fe.pot_ex_sprkkr"),
        str(tmp_path / "Fe.pot.only_coulomb_new"),
    )


def test_jxc_frel_declares_tensor_and_plot_outputs(tmp_path):
    parameters = InputParameters.create("JXC")
    parameters.CONTROL.DATASET = "Fe"
    parameters.MODE.MODE = "FREL"

    assert parameters.output_files(str(tmp_path)) == (
        str(tmp_path / "Fe_JXC_XCPLTEN_Jij.dat"),
        str(tmp_path / "Fe_JXC_XCPLTEN_Jij_*.agr"),
        str(tmp_path / "Fe_JXC_XCPLTEN_Dij.dat"),
        str(tmp_path / "Fe_JXC_XCPLTEN_Dij_[xyz]_*.agr"),
    )


def test_jxc_sp_srel_declares_scalar_and_plot_outputs(tmp_path):
    parameters = InputParameters.create("JXC")
    parameters.CONTROL.DATASET = "Fe"
    parameters.MODE.MODE = "SP-SREL"

    assert parameters.output_files(str(tmp_path)) == (
        str(tmp_path / "Fe_JXC_J_ij.dat"),
        str(tmp_path / "Fe_JXC_Jij_*.agr"),
    )


def test_spec_inputs_follow_native_flags(tmp_path):
    for task in ("ARPES", "AIPES", "BAND", "SPLEED", "CLPES"):
        parameters = InputParameters.create(task)
        parameters.CONTROL.DATASET = "Fe"
        assert parameters.input_files(str(tmp_path)) == ()
        parameters.TASK.INPVER = 0
        assert parameters.input_files(str(tmp_path)) == (
            str(tmp_path / "input.inp"),
        )
        parameters.TASK.STRVER = 0
        assert parameters.input_files(str(tmp_path)) == (
            str(tmp_path / "input.inp"), str(tmp_path / "struc.inp"),
        )


def test_spec_outputs_follow_polarization_and_task(tmp_path):
    for task in ("ARPES", "AIPES", "CLPES"):
        parameters = InputParameters.create(task)
        parameters.CONTROL.DATASET = "Fe"
        outputs = (
            str(tmp_path / f"Fe_{task}_SPEC.out"),
            str(tmp_path / f"Fe_{task}_data.spc"),
        )
        if task == "CLPES":
            outputs += (str(tmp_path / "Fe_CLPES_corewf"),)
        assert tuple(file for file in parameters.output_files(str(tmp_path)) if file.collect) == outputs

        # Selecting S polarization does not itself enable two output channels.
        parameters.SPEC_PH.POL_P = "S"
        assert tuple(file for file in parameters.output_files(str(tmp_path)) if file.collect) == outputs

        if task != "CLPES":
            for npol in (0, 3):
                parameters.SPEC_PH.NPOL = npol
                output_files = parameters.output_files(str(tmp_path))
                expected = (
                    str(tmp_path / f"Fe_{task}_SPEC.out"),
                    str(tmp_path / f"Fe_{task}_data_pol1.spc"),
                    str(tmp_path / f"Fe_{task}_data_pol2.spc"),
                )
                assert tuple(file for file in output_files if file.collect) == expected

    parameters = InputParameters.create("BAND")
    parameters.CONTROL.DATASET = "Fe"
    assert tuple(file for file in parameters.output_files(str(tmp_path)) if file.collect) == (
        str(tmp_path / "Fe_BAND_SPEC.out"), str(tmp_path / "Fe_BAND_data.spc"),
    )

    parameters = InputParameters.create("SPLEED")
    parameters.CONTROL.DATASET = "Fe"
    for diagram in (0, 1, 2):
        parameters.SPEC_EL.TYP = diagram
        declared = parameters.output_files(str(tmp_path))
        assert tuple(map(str, (f for f in declared if f.collect))) == (
            str(tmp_path / "Fe_SPLEED_SPEC.out"), str(tmp_path / "Fe_SPLEED_data.spc"),
        )
        assert all(not f.optional for f in declared)
        assert (str(declared[-1]), declared[-1].collect) == (str(tmp_path / "Fe_SPLEED.dos"), False)


def test_scf_invalid_grid_and_mixing_are_rejected():
    import pytest
    from ase2sprkkr.common.warnings import DataValidityError

    parameters = InputParameters.create("SCF")
    parameters.ENERGY.GRID = 3
    parameters.ENERGY.NE = 1
    with pytest.raises(DataValidityError, match="NE>=2"):
        parameters.validate("save")

    parameters = InputParameters.create("SCF")
    parameters.SCF.FULLPOT = True
    parameters.SCF.ALG = "TCHEBY"
    with pytest.raises(DataValidityError, match="TCHEBY"):
        parameters.validate("save")

def test_dos_native_energy_constraints():
    import pytest
    from ase2sprkkr.common.warnings import DataValidityError, DataValidityWarning

    parameters = InputParameters.create("DOS")
    parameters.CONTROL.POTFIL = "Fe.pot"
    parameters.ENERGY.NE = 1
    with pytest.raises(DataValidityError, match="two energy points"):
        parameters.validate("save")
    parameters.ENERGY.NE = 300
    parameters.ENERGY.ImE = 0.0
    with pytest.warns(DataValidityWarning, match="real axis"):
        parameters.validate("save")


def test_scf_startup_dos_is_optional(tmp_path):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "Fe.pot"
    files = parameters.output_files(str(tmp_path))
    assert next(f for f in files if str(f).endswith("_SCFSTART.dos")).optional
    assert not next(f for f in files if str(f).endswith("Fe.pot_new")).optional


def test_bsf_output_file_declarations(tmp_path):
    for task in ("BSFEK", "BSFKK"):
        parameters = InputParameters.create(task)
        parameters.CONTROL.DATASET = "Fe"
        assert tuple(map(str, parameters.output_files(str(tmp_path)))) == (
            str(tmp_path / "Fe_BLOCHSF.bsf"),
            str(tmp_path / "Fe_BLOCHSF_spol.bsf"),
        )

def test_clpes_core_wavefunction_is_a_declared_output(tmp_path):
    parameters = InputParameters.create("CLPES")
    parameters.CONTROL.DATASET = "Fe"
    assert str(tmp_path / "Fe_CLPES_corewf") in parameters.output_files(str(tmp_path))


def test_manifest_rejects_duplicate_output_declarations(tmp_path):
    import pytest
    from kkrunner.runs.base import ExecutionSpec
    from kkrunner.files.manifest import FileManifest
    from ase2sprkkr.input_parameters.task_definition import TaskFile

    result = tmp_path / "result.dat"
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=tmp_path / "input.inp",
        output_file=tmp_path / "stdout.out",
        output_files=(TaskFile(result, optional=True), TaskFile(result)),
    )
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec)


def test_embedded_jxc_and_dmi_use_cluster_tensor_filenames(tmp_path):
    # DMI selects a submode of the native JXC task: the output ADSI
    # remains JXC, and CLUXCPLTENSOR writes both J and D files.
    for task in ("JXC", "DMI"):
        parameters = InputParameters.create(task)
        parameters.CONTROL.DATASET = "Fe"
        parameters.CONTROL.CLUTYPE = "embedded"

        assert parameters.CONTROL.ADSI() == "JXC"
        assert tuple(map(str, parameters.output_files(str(tmp_path)))) == (
            str(tmp_path / "Fe_JXC_JJij.dat"),
            str(tmp_path / "Fe_JXC_Jij_*.agr"),
            str(tmp_path / "Fe_JXC_Dij_x.dat"),
            str(tmp_path / "Fe_JXC_Dij_y.dat"),
            str(tmp_path / "Fe_JXC_Dij_z.dat"),
            str(tmp_path / "Fe_JXC_Dij_[xyz]_*.agr"),
        )


def test_embedded_jxc_requires_frel_in_current_native_audit():
    import pytest
    from ase2sprkkr.common.warnings import DataValidityError

    parameters = InputParameters.create("JXC")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.clupot"
    parameters.CONTROL.CLUTYPE = "embedded"
    parameters.MODE.MODE = "SP-SREL"
    with pytest.raises(DataValidityError, match="Embedded JXC/DMI currently requires FREL"):
        parameters.validate("save")

def test_user_facing_result_files_are_also_collected_outputs(tmp_path):
    directory = str(tmp_path)
    for task in ("DOS", "BSFEK", "BSFKK", "XAS", "JXC", "DMI", "GILBERT", "ARPES", "AIPES", "BAND", "CLPES", "SPLEED"):
        parameters = InputParameters.create(task)
        parameters.CONTROL.DATASET = "Fe"
        results = parameters.result_files(directory)
        outputs = parameters.output_files(directory)
        assert results, task
        assert set(map(str, results)) <= set(map(str, outputs)), task
        if task in ("ARPES", "AIPES", "BAND", "CLPES", "SPLEED"):
            assert all(str(file).endswith(".spc") for file in results)


def test_scf_potential_is_output_but_not_a_scientific_result(tmp_path):
    p = InputParameters.create("SCF")
    p.CONTROL.POTFIL = "Fe.pot"
    outputs = p.output_files(str(tmp_path))
    assert str(tmp_path / "Fe.pot_new") in set(map(str, outputs))
    assert p.result_files(str(tmp_path)) == ()


def test_spec_result_declarations_obey_polarization(tmp_path):
    parameters = InputParameters.create("ARPES")
    parameters.CONTROL.DATASET = "Fe"
    assert tuple(map(str, parameters.result_files(str(tmp_path)))) == (
        str(tmp_path / "Fe_ARPES_data.spc"),
    )
    parameters.SPEC_PH.NPOL = 0
    assert tuple(map(str, parameters.result_files(str(tmp_path)))) == (
        str(tmp_path / "Fe_ARPES_data_pol1.spc"),
        str(tmp_path / "Fe_ARPES_data_pol2.spc"),
    )


def test_spec_output_manifest_retains_native_order_and_collision_only_dos(tmp_path):
    p = InputParameters.create("ARPES")
    p.CONTROL.DATASET = "Fe"
    outputs = p.output_files(str(tmp_path))
    assert tuple(str(f) for f in outputs) == (
        str(tmp_path / "Fe_ARPES_SPEC.out"),
        str(tmp_path / "Fe_ARPES_data.spc"),
        str(tmp_path / "Fe_ARPES.dos"),
    )
    assert tuple(f.collect for f in outputs) == (True, True, False)
    assert tuple(str(f) for f in p.result_files(str(tmp_path))) == (
        str(tmp_path / "Fe_ARPES_data.spc"),
    )


@pytest.mark.parametrize(
    ("task", "diagnostic"),
    [("COMPTON", "Fe_COMPTON_data.mcp"), ("POSANI", "Fe_POSANI_data.mpa")],
)
def test_optional_diagnostic_dump_is_not_a_user_facing_result(tmp_path, task, diagnostic):
    p = InputParameters.create(task)
    p.CONTROL.DATASET = "Fe"
    p.CONTROL.PRINT = 1
    outputs = p.output_files(str(tmp_path))
    results = p.result_files(str(tmp_path))
    expected = str(tmp_path / diagnostic)
    assert expected in tuple(map(str, outputs))
    assert expected not in tuple(map(str, results))


def test_phagen_export_is_a_result_not_a_temporary_cache(tmp_path):
    p = InputParameters.create("PHAGEN")
    p.CONTROL.POTFIL = "Fe.pot"
    expected = (
        str(tmp_path / "Fe.pot_ex_sprkkr"),
        str(tmp_path / "Fe.pot.only_coulomb_new"),
    )
    assert tuple(map(str, p.result_files(str(tmp_path)))) == expected
    assert tuple(map(str, p.output_files(str(tmp_path)))) == expected


def test_direct_task_definition_output_contract_includes_results_without_duplicates():
    from ase2sprkkr.input_parameters.task_definition import TaskDefinition, TaskFile

    task = TaskDefinition(
        "DOS",
        output_files=("data.dos", "log.out"),
        result_files=(TaskFile("data.dos", type="DOS"),),
    )
    assert tuple(map(str, task.output_files)) == ("data.dos", "log.out")
    assert tuple(map(str, task.result_files)) == ("data.dos",)

    # A scientific declaration not explicitly listed in output_files must
    # still remain visible in the full legacy TaskDefinition.output_files API.
    dos = InputParameters.create("DOS").task_definition
    assert tuple(map(str, dos.output_files)) == ("{DATASET_ADSI}.dos",)
    assert tuple(map(str, dos.result_files)) == ("{DATASET_ADSI}.dos",)
