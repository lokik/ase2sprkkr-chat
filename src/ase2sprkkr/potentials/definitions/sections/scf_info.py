"""
Scf info section contains configuration options used in computation
"""

from ...potential_definitions import PotSectionDefinition, PotValueDefinition
from ...potential_sections import PotentialSection

from ....common.grammar_types import DefKeyword, Array, line_string


class ScfInfoSection(PotentialSection):
    """This section retrieves the atomic positions and
    it creates (during reading) the ASE Atoms object"""

    def _depends_on(self):
        return ["SITES"]

    def _update_atoms(self, atoms, read_io_data):
        atoms.info["fermi_energy"] = self.EF()
        atoms.info["sprkkr_rmsavv"] = self.RMSAVV()
        atoms.info["sprkkr_rmsavb"] = self.RMSAVB()
        atoms.info["sprkkr_vmtz"] = self.VMTZ()

    def _set_from_atoms(self, atoms, write_io_data):
        info = atoms.info
        if "fermi_energy" in info:
            self.EF = atoms.info["fermi_energy"]
        if "sprkkr_rmsavv" in info:
            self.RMSAVV = atoms.info["sprkkr_rmsavv"]
        if "sprkkr_rmsavb" in info:
            self.RMSAVB = atoms.info["sprkkr_rmsavb"]
        if "sprkkr_vmtz" in info:
            self.VMTZ = atoms.info["sprkkr_vmtz"]


def scf_info_members(*, cluster=False):
    """Shared POTFIL/CLUPOT SCF fields, retaining native order and types."""
    V = PotValueDefinition
    fields = [
        V("INFO", line_string, "NONE"),
        (V("SCFSTATUS", str, "START") if cluster else
         V("SCFSTATUS", DefKeyword(
             "START", "CONVERGED", "ITR-BULK", "ITR-R-BULK",
             "ITR-L-BULK", "ITR-I-ZONE", "ITR",
         ))),
        V("FULLPOT", False),
    ]
    if cluster:
        fields.append(V("SPHERCELL", False, is_optional=True))
    fields.extend([
        V("BREITINT", False),
        V("NONMAG" if cluster else ("NONMAG", "NOMAG"), False),
        V("ORBPOL", str, "NONE"),
        V("EXTFIELD", False),
        V("BLCOUPL", False),
        V("BEXT", 0.0),
        V("SEMICORE", False),
        V("LLOYD", False),
        V("NE", int if cluster else Array(int),
          **({} if cluster else {"is_optional": True})),
        V("IBZINT", int, **({} if cluster else {"is_optional": True})),
        V("NKTAB", int, **({} if cluster else {"is_optional": True})),
    ])
    if not cluster:
        fields.append(V("TETDEPPOT", False, is_optional=True))
    fields.extend([
        V("XC-POT", str, **({} if cluster else {"is_optional": True})),
        V("SCF-ALG", str, **({} if cluster else {"is_optional": True})),
        V("SCF-ITER", int if cluster else 0),
        V("SCF-MIX", float if cluster else 0.2),
        V("SCF-TOL", float if cluster else 1e-5),
        V("RMSAVV", float if cluster else 999999.0),
        V("RMSAVB", float if cluster else 999999.0),
        V("EF", float if cluster else 999999.0),
        V("VMTZ", float if cluster else 0.7),
    ])
    if not cluster:
        fields.append(V("EWORK", float, is_optional=True))
    return fields


class ScfInfoSectionDefinition(PotSectionDefinition):
    def __init__(self, name="SCF-INFO", **kwargs):
        super().__init__(name, scf_info_members(), has_hidden_members=True, **kwargs)

    result_class = ScfInfoSection


section = ScfInfoSectionDefinition
