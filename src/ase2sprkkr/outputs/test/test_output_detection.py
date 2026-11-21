"""The SCF prefilter must avoid parsing unrelated outputs."""

from pathlib import Path

import pytest

from ase2sprkkr.outputs import looks_like_scf_output
from ase2sprkkr.outputs.output_detection import (
    looks_like_scf_output as implementation,
)


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _banner(executable="KKRSCF", version="9.7"):
    return (
        b"**************************************************************\n"
        + f"  *       {executable}  VERSION  {version}     (C) H. Ebert    *\n".encode()
        + b"**************************************************************\n"
    )


def test_scf_header_probe_is_public():
    assert looks_like_scf_output is implementation


@pytest.mark.parametrize("version", ["9.7", "10.0", "2026.1", "8.5.aug08"])
def test_scf_header_probe_accepts_multiple_versions(tmp_path, version):
    output = tmp_path / "calculation.out"
    output.write_bytes(_banner(version=version) + b"  TASK = SCF\n")
    assert looks_like_scf_output(output)


@pytest.mark.parametrize("banner", [
    b"  *       KKR DOS VERSION 9.7      *\n",
    b"  *       KKRBSF VERSION 9.7       *\n",
    b"  *       ANOTHERKKRSCF VERSION    *\n",
    b"  *       KKRSCF_SUFFIX VERSION    *\n",
    b"",
])
def test_scf_header_probe_rejects_unrelated_outputs(tmp_path, banner):
    output = tmp_path / "not_scf.out"
    output.write_bytes(banner)
    assert not looks_like_scf_output(output)


def test_scf_header_probe_rejects_kkrgen_even_with_scf_mentions(tmp_path):
    output = tmp_path / "bsf.out"
    output.write_bytes(
        _banner("KKRGEN")
        + b" Previous executable: KKRSCF\n"
        + b"  *       KKRSCF VERSION 9.7       *\n"
        + b" TASK = BSF\n"
    )
    assert not looks_like_scf_output(output)


@pytest.mark.parametrize("task", ["BSF", "DOS", "JXC", "ARPES"])
def test_scf_header_probe_rejects_other_task_under_scf_banner(tmp_path, task):
    output = tmp_path / "not_scf.out"
    output.write_bytes(_banner() + f" TASK = {task}\n".encode())
    assert not looks_like_scf_output(output)


def test_scf_header_probe_finds_task_after_eight_kib(tmp_path):
    output = tmp_path / "scf.out"
    output.write_bytes(_banner() + b" preceding native output\n" * 700
                       + b" TASK       = SCF   \n")
    assert output.stat().st_size > 8192
    assert looks_like_scf_output(output)


def test_scf_header_probe_finds_non_scf_task_after_eight_kib(tmp_path):
    output = tmp_path / "not_scf.out"
    output.write_bytes(_banner() + b" preceding native output\n" * 700
                       + b" TASK       = BSF   \n")
    assert output.stat().st_size > 8192
    assert not looks_like_scf_output(output)


@pytest.mark.parametrize("suffix", [
    b"",
    b"  FILES:\n potential: Fe.pot\n",
    b"  TASK = NONE\n  FILES:\n",
])
def test_scf_header_probe_legacy_fallback_without_explicit_task(tmp_path, suffix):
    output = tmp_path / "legacy.out"
    output.write_bytes(_banner() + suffix)
    assert looks_like_scf_output(output)


def test_scf_header_probe_uses_first_task(tmp_path):
    output = tmp_path / "first.out"
    output.write_bytes(_banner() + b" TASK = BSF\n TASK = SCF\n")
    assert not looks_like_scf_output(output)


def test_scf_header_probe_does_not_read_entire_output(tmp_path):
    output = tmp_path / "large.out"
    output.write_bytes(b" " * 8192 + _banner() + b" TASK = SCF\n")
    assert not looks_like_scf_output(output)


def test_scf_header_probe_treats_unreadable_paths_as_non_matches(tmp_path):
    assert not looks_like_scf_output(tmp_path / "missing.out")
    assert not looks_like_scf_output(tmp_path)


@pytest.mark.parametrize(("filename", "expected"), [
    ("Al_SCF.out", True),
    ("Fe2Al_SCF.out", True),
    ("scf.out", True),
    ("Fe_jxc.out", False),
])
def test_scf_header_probe_on_real_sprkkr_examples(filename, expected):
    assert looks_like_scf_output(EXAMPLES / filename) is expected


@pytest.mark.parametrize(("reported", "expected", "matches"), [
    ("Fe.pot_new", "Fe.pot_new", True),
    ("Fe_other.pot_new", "Fe.pot_new", False),
    ("subdir/Fe.pot_new", "subdir/Fe.pot_new", True),
    ("subdir/Fe.pot_new", "Fe.pot_new", False),
])
def test_scf_header_probe_checks_produced_potential_not_input(
    tmp_path, reported, expected, matches,
):
    out = tmp_path / "Fe_SCF.out"
    out.write_bytes(
        _banner()
        + b"  TASK = SCF\n  FILES:\n"
        + b"  potential : ( 4) original_input.pot\n"
        + b"  parameters for SCF-cycle\n"
        + f"  pot-file :  {reported}\n".encode()
    )
    assert looks_like_scf_output(out, tmp_path / expected) is matches
    assert looks_like_scf_output(out)


def test_scf_header_probe_potential_unknown_preserves_fallback(tmp_path):
    out = tmp_path / "legacy.out"
    out.write_bytes(_banner() + b" TASK = SCF\n FILES:\n potential: Fe.pot\n")
    assert looks_like_scf_output(out, tmp_path / "unavailable.pot_new")
    out.write_bytes(_banner() + b" TASK = DOS\n pot-file : Fe.pot_new\n")
    assert not looks_like_scf_output(out, tmp_path / "Fe.pot_new")


@pytest.mark.parametrize(("example", "produced"), [
    ("Al_SCF.out", "Al.pot_new_new"),
    ("scf.out", "CoPt_mag001-bas.pot_new"),
])
def test_scf_header_probe_matches_native_scf_pot_file(example, produced):
    out = EXAMPLES / example
    assert looks_like_scf_output(out, out.parent / produced)
    assert not looks_like_scf_output(out, out.parent / "unrelated.pot_new")
