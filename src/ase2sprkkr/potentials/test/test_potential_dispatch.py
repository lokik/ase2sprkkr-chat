"""Native grammar-alternative dispatch and strict potential subclasses."""

from io import BytesIO, StringIO
from pathlib import Path

import pytest
from pyparsing import ParseBaseException
from ase.build import bulk

from ..potential_base import Potential
from ..potentials import PeriodicPotential
from ..cluster_potentials import EmbeddedPotential, ClusterPotential
from ...sprkkr.sprkkr_atoms import SPRKKRAtoms


CLUPOT = Path(__file__).parent / "data" / "Fe3CoFe3.clu.pot"
PERIODIC = Path(__file__).parent.parent / "examples" / "FePt.new.pot"


def test_uploaded_native_clupot_uses_embedded_grammar():
    potential = Potential.from_file(CLUPOT)
    assert isinstance(potential, Potential)
    assert isinstance(potential, EmbeddedPotential)
    assert isinstance(potential, ClusterPotential)
    assert potential.host is None
    assert potential.atoms is None
    assert potential.GLOBAL_SYSTEM_PARAMETER.NQCLU() == 7
    assert potential.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 7
    assert potential.GLOBAL_SYSTEM_PARAMETER.NMCLU() == 0
    assert len(potential.SITES.DATA()) == 7
    assert [row["TXT"] for row in potential.TYPES.DATA()] == [
        "Fe", "Fe", "Fe", "Co", "Fe", "Fe", "Fe"
    ]
    assert potential.SCF_INFO.SCFSTATUS() == "START"
    assert potential.MAGNETISATION_DIRECTION.DATA()[0]["IQ"] == 3


def test_periodic_potfil_uses_periodic_grammar():
    potential = Potential.from_file(PERIODIC)
    assert isinstance(potential, Potential)
    assert isinstance(potential, PeriodicPotential)
    assert not isinstance(potential, EmbeddedPotential)
    assert potential.atoms is not None


def test_specific_potential_classes_reject_opposite_format():
    with pytest.raises(ParseBaseException):
        PeriodicPotential.from_file(CLUPOT)
    with pytest.raises(ParseBaseException):
        EmbeddedPotential.from_file(PERIODIC)

    assert isinstance(PeriodicPotential.from_file(PERIODIC), PeriodicPotential)
    assert isinstance(EmbeddedPotential.from_file(CLUPOT), EmbeddedPotential)


def test_common_reader_consumes_nonseekable_stream_once():
    contents = CLUPOT.read_text(encoding="utf-8")

    class Stream:
        def __init__(self, data):
            self.data = data
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("The stream was read a second time")
            return self.data

    source = Stream(contents)
    potential = Potential.from_file(source)
    assert isinstance(potential, EmbeddedPotential)
    assert source.calls == 1

    binary = Potential.from_file(BytesIO(contents.encode("utf-8")))
    assert isinstance(binary, EmbeddedPotential)


def test_malformed_embedded_file_not_accepted_as_periodic():
    content = CLUPOT.read_text(encoding="utf-8").replace(
        "NQCLU       \t7", "NQCLU       \tnot_an_integer", 1
    )
    assert content != CLUPOT.read_text(encoding="utf-8")
    with pytest.raises(ParseBaseException):
        Potential.from_string(content)
    with pytest.raises(ParseBaseException):
        PeriodicPotential.from_string(content)


def test_legacy_potential_constructor_creates_periodic_subclass():
    potential = Potential()
    assert isinstance(potential, PeriodicPotential)
    assert isinstance(potential, Potential)


def test_common_from_atoms_uses_concrete_subclasses():
    from ..potentials import PeriodicPotential

    host = Potential.from_atoms(bulk("Cu", "fcc", a=3.61))
    assert isinstance(host, PeriodicPotential)
    cluster_atoms = SPRKKRAtoms.from_embedding(host, [(0, (0, 0, 0), "Fe")])
    embedded = Potential.from_atoms(cluster_atoms)
    assert isinstance(embedded, EmbeddedPotential)
    assert embedded.host is host
