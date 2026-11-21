"""A quick SCF banner probe must not trigger a full output parse."""

import pytest

from ase2sprkkr.outputs import looks_like_scf_output
from ase2sprkkr.outputs.output_detection import (
    looks_like_scf_output as implementation,
)


def test_scf_header_probe_is_public():
    assert looks_like_scf_output is implementation


@pytest.mark.parametrize("version", ["9.7", "10.0", "2026.1"])
def test_scf_header_probe_accepts_multiple_versions(tmp_path, version):
    output = tmp_path / "calculation.out"
    output.write_bytes(
        b"**************************************************************\n"
        + f"  *       KKRSCF  VERSION  {version}     (C) H. Ebert    *\n".encode()
        + b"**************************************************************\n"
    )
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


def test_scf_header_probe_does_not_read_entire_output(tmp_path):
    output = tmp_path / "large.out"
    output.write_bytes(b" " * 8192 + b" KKRSCF VERSION 9.7\n")
    assert not looks_like_scf_output(output)


def test_scf_header_probe_treats_unreadable_paths_as_non_matches(tmp_path):
    assert not looks_like_scf_output(tmp_path / "missing.out")
    assert not looks_like_scf_output(tmp_path)
