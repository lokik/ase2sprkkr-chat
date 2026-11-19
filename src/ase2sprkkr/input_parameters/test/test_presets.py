"""Tests for sparse native input presets."""
import numpy as np
import pytest

from ..input_parameters import InputParameters
from ..presets import PresetStore


def test_sparse_save_and_merge(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.ENERGY.NE.set(341)
    source.save_preset("mesh", store=store)
    text = (tmp_path / "input_parameters" / "dos" / "mesh.inp").read_text()
    assert "NE=" in text
    assert "POTFIL=" not in text
    assert "DATASET=" not in text

    target = InputParameters.for_task("DOS")
    target.TAU.NKTAB.set(567)
    target.load_preset("mesh", clear_first=False, store=store)
    assert int(np.asarray(target.ENERGY.NE()).flat[0]) == 341
    assert target.TAU.NKTAB() == 567


def test_reset_and_explicit_default(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.TAU.NKTAB.set(250)
    source.save_preset("default_k", store=store)
    target = InputParameters.for_task("DOS")
    target.TAU.NKTAB.set(123)
    target.ENERGY.NE.set(444)
    target.load_preset("default_k", clear_first=True, store=store)
    assert target.TAU.NKTAB() == 250
    assert int(np.asarray(target.ENERGY.NE()).flat[0]) != 444


def test_ambiguous_name_and_cross_schema_rejected(tmp_path):
    store = PresetStore(tmp_path)
    dos = InputParameters.for_task("DOS")
    scf = InputParameters.for_task("SCF")
    dos.save_preset("same", store=store)
    scf.save_preset("same", store=store)
    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve("same")
    assert store.resolve("dos.same").name == "same.inp"
    with pytest.raises(ValueError, match="Invalid"):
        dos.save_preset("../escape", store=store)
    with pytest.raises(FileNotFoundError, match="incompatible"):
        dos.load_preset("scf.same", store=store)


def test_invalid_fragment_is_atomic(tmp_path):
    store = PresetStore(tmp_path)
    InputParameters.for_task("DOS").save_preset("bad", store=store)
    store.resolve("dos.bad").write_text("ENERGY\n  NE=NOT_A_NUMBER\n")
    target = InputParameters.for_task("DOS")
    before = target.as_dict(only_changed="explicit")
    with pytest.raises(Exception):
        target.load_preset("bad", store=store)
    assert target.as_dict(only_changed="explicit") == before


def test_absent_adsi_does_not_overwrite(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.CONTROL.PRINT.set(3)
    source.ENERGY.NE.set(314)
    source.save_preset("mesh", store=store)
    target = InputParameters.for_task("DOS")
    target.CONTROL.ADSI.set("CUSTOM")
    target.load_preset("mesh", store=store)
    assert target.CONTROL.ADSI() == "CUSTOM"


def test_bsf_shared_ipd(tmp_path):
    store = PresetStore(tmp_path)
    assert store.definition_name("BSFEK") == "bsf"
    assert store.definition_name("BSFKK") == "bsf"


def test_default_replaces_and_explicit_false_clears_flag(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.CONTROL.NOSYM.set(False)
    source.save_preset("symmetry", store=store)

    target = InputParameters.for_task("DOS")
    target.CONTROL.NOSYM.set(True)
    target.load_preset("symmetry", clear_first=False, store=store)
    assert target.CONTROL.NOSYM() is True

    target.load_preset("symmetry", store=store)
    assert target.CONTROL.NOSYM() is False


def test_structural_info_is_preserved_even_when_replacing(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.CONTROL.DATASET.set("FROM_PRESET")
    source.CONTROL.ADSI.set("FROM")
    source.CONTROL.POTFIL.set("from.pot")
    source.CONTROL.CLUTYPE.set("embedded")
    source.CONTROL.POTFIL_CLU.set("from.clu")
    source.CONTROL.SFNFIL.set("from.shp")
    source.ENERGY.NE.set(419)
    source.save_preset("with_paths", store=store)

    target = InputParameters.for_task("DOS")
    target.CONTROL.DATASET.set("CURRENT")
    target.CONTROL.ADSI.set("KEEP")
    target.CONTROL.POTFIL.set("keep.pot")
    target.CONTROL.CLUTYPE.set("embedded")
    target.CONTROL.POTFIL_CLU.set("keep.clu")
    target.CONTROL.SFNFIL.set("keep.shp")
    target.TAU.NKTAB.set(681)
    target.load_preset("with_paths", store=store)

    assert str(target.CONTROL.DATASET()) == "CURRENT"
    assert str(target.CONTROL.ADSI()) == "KEEP"
    assert str(target.CONTROL.POTFIL()) == "keep.pot"
    assert str(target.CONTROL.POTFIL_CLU()) == "keep.clu"
    assert target.CONTROL.CLUTYPE() == "embedded"
    assert str(target.CONTROL.SFNFIL()) == "keep.shp"
    assert int(np.asarray(target.ENERGY.NE()).flat[0]) == 419
    assert target.TAU.NKTAB() != 681


def test_structural_info_can_be_loaded_explicitly(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.CONTROL.DATASET.set("FROM_PRESET")
    source.CONTROL.POTFIL.set("from.pot")
    source.ENERGY.NE.set(410)
    source.save_preset("paths", store=store)

    target = InputParameters.for_task("DOS")
    target.CONTROL.DATASET.set("CURRENT")
    target.CONTROL.POTFIL.set("keep.pot")
    target.load_preset("paths", load_structural_info=True, store=store)
    assert str(target.CONTROL.DATASET()) == "FROM_PRESET"
    assert str(target.CONTROL.POTFIL()) == "from.pot"


def test_merge_keeps_structural_info_and_unrelated_values(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.CONTROL.POTFIL.set("other.pot")
    source.ENERGY.NE.set(499)
    source.save_preset("mesh", store=store)
    target = InputParameters.for_task("DOS")
    target.CONTROL.POTFIL.set("own.pot")
    target.TAU.NKTAB.set(388)
    target.load_preset("mesh", clear_first=False, store=store)
    assert str(target.CONTROL.POTFIL()) == "own.pot"
    assert target.TAU.NKTAB() == 388
    assert int(np.asarray(target.ENERGY.NE()).flat[0]) == 499


def test_malformed_preset_keeps_all_existing_values(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_task("DOS")
    source.save_preset("bad", store=store)
    store.resolve("dos.bad").write_text("ENERGY\n  NE=NOT_A_NUMBER\n")
    target = InputParameters.for_task("DOS")
    target.ENERGY.NE.set(317)
    target.CONTROL.POTFIL.set("original.pot")
    with pytest.raises(Exception):
        target.load_preset("bad", store=store)
    assert int(np.asarray(target.ENERGY.NE()).flat[0]) == 317
    assert str(target.CONTROL.POTFIL()) == "original.pot"



def test_native_scoped_storage_and_contextual_discovery(tmp_path):
    store = PresetStore(tmp_path)
    neutral = InputParameters.for_task("BSFEK")
    neutral.TAU.NKTAB.set(542)
    neutral.save_preset("common", store=store)
    assert (tmp_path / "input_parameters" / "bsf" / "common.inp").is_file()

    ek = InputParameters.for_task("BSFEK")
    ek.ENERGY.NE.set([300])
    ek.save_preset("mesh", store=store)
    assert (tmp_path / "task" / "bsfek" / "mesh.inp").is_file()

    kk = InputParameters.for_task("BSFKK")
    kk.ENERGY.NE.set([1])
    kk.save_preset("mesh", store=store)
    assert (tmp_path / "task" / "bsfkk" / "mesh.inp").is_file()

    assert store.list("input_parameters:bsf") == (
        "input_parameters:bsf.common",
        "task:bsfek.mesh",
        "task:bsfkk.mesh",
    )
    assert store.list("task:bsfkk") == (
        "input_parameters:bsf.common",
        "task:bsfkk.mesh",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve("mesh", definition="bsf")
    assert store.resolve("mesh", task="bsfkk").parent.name == "bsfkk"


def test_explicit_task_preset_can_switch_effective_task_on_replace(tmp_path):
    store = PresetStore(tmp_path)
    kk = InputParameters.for_task("BSFKK")
    kk.ENERGY.NE.set([1])
    kk.TASK.NK1.set(211)
    kk.save_preset("task:bsfkk.fermi", store=store)

    ip = InputParameters.for_definition("BSF")
    ip.ENERGY.NE.set([2])
    ip.CONTROL.POTFIL.set("keep.pot")
    assert ip.requested_task_name is None
    assert ip.task_name == "BSFEK"

    ip.load_preset("task:bsfkk.fermi", store=store)
    assert ip.task_name == "BSFKK"
    assert ip.requested_task_name == "BSFKK"
    assert int(np.asarray(ip.ENERGY.NE()).flat[0]) == 1
    assert ip.TASK.NK1() == 211
    assert str(ip.CONTROL.POTFIL()) == "keep.pot"


def test_cross_task_merge_is_rejected_without_side_effects(tmp_path):
    store = PresetStore(tmp_path)
    kk = InputParameters.for_task("BSFKK")
    kk.ENERGY.NE.set([1])
    kk.save_preset("task:bsfkk.fermi", store=store)
    ip = InputParameters.for_definition("BSF")
    ip.ENERGY.NE.set([3])
    with pytest.raises(ValueError, match="Merging"):
        ip.load_preset("task:bsfkk.fermi", clear_first=False, store=store)
    assert ip.task_name == "BSFEK"
    assert ip.requested_task_name is None
    assert int(np.asarray(ip.ENERGY.NE()).flat[0]) == 3


def test_explicit_task_requirement_prevents_conflicting_replacement(tmp_path):
    store = PresetStore(tmp_path)
    InputParameters.for_task("BSFKK").save_preset(
        "task:bsfkk.defaults", store=store,
    )
    ip = InputParameters.for_task("BSFEK")
    with pytest.raises(ValueError, match="explicitly requested"):
        ip.load_preset(
            "task:bsfkk.defaults", required_task="BSFEK", store=store,
        )
    assert ip.task_name == "BSFEK"
    assert ip.requested_task_name == "BSFEK"


def test_neutral_scope_preserves_current_effective_task_after_reset(tmp_path):
    store = PresetStore(tmp_path)
    neutral = InputParameters.for_definition("BSF")
    neutral.TAU.NKTAB.set(504)
    neutral.save_preset("input_parameters:bsf.shared", store=store)
    ip = InputParameters.for_definition("BSF")
    ip.ENERGY.NE.set([275])
    assert ip.requested_task_name is None
    ip.load_preset("input_parameters:bsf.shared", store=store)
    assert ip.task_name == "BSFEK"
    assert ip.requested_task_name == "BSFEK"
    assert ip.TAU.NKTAB() == 504
    assert int(np.asarray(ip.ENERGY.NE()).flat[0]) > 1


def test_neutral_scope_rejects_task_selecting_values(tmp_path):
    store = PresetStore(tmp_path)
    ek = InputParameters.for_task("BSFEK")
    ek.ENERGY.NE.set([280])
    with pytest.raises(ValueError, match="specific task"):
        ek.save_preset("input_parameters:bsf.invalid", store=store)
    assert not (tmp_path / "input_parameters" / "bsf" / "invalid.inp").exists()


def test_jxc_dmi_colliding_name_requires_explicit_scope(tmp_path):
    store = PresetStore(tmp_path)
    common = InputParameters.for_task("JXC")
    common.TAU.NKTAB.set(412)
    common.save_preset("input_parameters:jxc.shared", store=store)
    jxc = InputParameters.for_task("JXC")
    jxc.save_preset("task:jxc.shared", store=store)
    dmi = InputParameters.for_task("DMI")
    dmi.save_preset("task:dmi.dm", store=store)

    assert store.resolve("task:jxc.shared").parent.name == "jxc"
    assert store.resolve("input_parameters:jxc.shared").parent.name == "jxc"
    with pytest.raises(ValueError, match="ambiguous"):
        store.resolve("jxc.shared")
    with pytest.raises(ValueError, match="both a shared IPD"):
        store.list("jxc")
    assert store.list("input_parameters:jxc") == (
        "input_parameters:jxc.shared",
        "task:dmi.dm",
        "task:jxc.shared",
    )


def test_wrong_scope_cannot_mask_wrong_effective_task(tmp_path):
    store = PresetStore(tmp_path)
    ek = InputParameters.for_task("BSFEK")
    ek.ENERGY.NE.set([220])
    with pytest.raises(ValueError, match="Cannot save"):
        ek.save_preset("task:bsfkk.invalid", store=store)



def test_modified_ipd_preset_is_revalidated_on_load(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_definition("BSF")
    source.TAU.NKTAB.set(550)
    source.save_preset("input_parameters:bsf.neutral", store=store)
    store.resolve("input_parameters:bsf.neutral").write_text(
        "ENERGY\n  NE={300}\n"
    )
    target = InputParameters.for_task("BSFKK")
    assert target.task_name == "BSFKK"
    with pytest.raises(ValueError, match="not neutral"):
        target.load_preset("input_parameters:bsf.neutral", store=store)
    assert target.task_name == "BSFKK"



def test_task_inferred_from_ne_controls_automatic_scope(tmp_path):
    store = PresetStore(tmp_path)
    source = InputParameters.for_definition("BSF")
    assert source.requested_task_name is None
    source.ENERGY.NE.set([2])
    assert source.task_name == "BSFEK"
    source.save_preset("from_ne", store=store)
    assert (tmp_path / "task" / "bsfek" / "from_ne.inp").is_file()

    target = InputParameters.for_definition("BSF")
    assert target.requested_task_name is None
    target.load_preset("task:bsfek.from_ne", store=store)
    assert target.task_name == "BSFEK"
