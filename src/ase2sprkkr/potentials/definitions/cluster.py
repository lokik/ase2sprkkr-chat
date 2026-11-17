"""Grammar for native SPR-KKR embedded-cluster potential files.

The format follows CLUPOTRD/CLUPOTWR in the SPR-KKR sources.  In particular,
cluster site/type numbers and host site numbers are *different* namespaces.
A cluster potential is not a periodic potential with extra atomic sites.
"""

import re

from ...common.configuration_definitions import BaseDefinition, SeparatorDefinition
from ...common.grammar_types import Array, DefKeyword, NumpyArray, Sequence, Table, line_string
from ..potential_definitions import (
    PotSectionDefinition,
    PotValueDefinition as V,
    PotentialDefinition,
    Separator,
)
from ..potential_sections import RepeatedPotentialSection
from ..cluster_potentials import ClusterPotential
from .sections import ChargeSectionDefinition, PotentialSectionDefinition, TypesSectionDefinition


def _occupation_count_matches(row):
    """NOQCLU must equal the number of (ITCLU, concentration) pairs."""
    return row[1] == len(row[2]) or "NOQCLU differs from the number of occupations"


class ClusterPotentialDefinition(PotentialDefinition):
    """Definition of the separate POTFIL_CLU format."""

    result_class = ClusterPotential
    configuration_type_name = "CLUSTER POTENTIAL FILE"

    def _generic_info(self):
        return "Definition of the SPR-KKR embedded-cluster potential format"


class ClusterRepeatedSectionDefinition(PotSectionDefinition):
    """Repeated radial or moment block within a cluster potential."""

    repeated_class = RepeatedPotentialSection


def _repeated_block(name, members):
    """Define a repeated TYPE block, separated by 79 equal signs."""
    return ClusterRepeatedSectionDefinition(
        name,
        [V("TYPE", int), *members, SeparatorDefinition("=", length=79)],
        has_hidden_members=True,
        is_repeated=BaseDefinition.Repeated.LIST_SECTION,
        is_optional=True,
    )


def _definition():
    sections = []

    def section(name, values=None, *, cls=PotSectionDefinition, **kwargs):
        result = cls(name, *([values] if values is not None else []), **kwargs)
        sections.append(result)
        return result

    # The cluster header has neither PACKAGE nor FORMAT, unlike POTFIL.
    section(
        "HEADER",
        [
            Separator(),
            V("HEADER", line_string, "SPR-KKR cluster potential"),
            Separator(),
            V("TITLE", line_string, "Embedded cluster"),
            V("SYSTEM", line_string, "Cluster in a reference host"),
        ],
        name_in_grammar=False,
    )
    section(
        "GLOBAL SYSTEM PARAMETER",
        [V("NQCLU", int), V("NTCLU", int), V("NMCLU", int)],
        has_hidden_members=True,
    )

    # CLUPOTWR uses ITERATING (rather than the periodic potential's
    # CONVERGED/ITR-* convention), and NE is a scalar.
    section(
        "SCF-INFO",
        [
            V("INFO", line_string, "NONE"),
            V("SCFSTATUS", str, "START"),
            V("FULLPOT", False),
            V("SPHERCELL", False, is_optional=True),
            V("BREITINT", False),
            V("NONMAG", False),
            V("ORBPOL", str, "NONE"),
            V("EXTFIELD", False),
            V("BLCOUPL", False),
            V("BEXT", 0.0),
            V("SEMICORE", False),
            V("LLOYD", False),
            V("NE", int),
            V("IBZINT", int),
            V("NKTAB", int),
            V("XC-POT", str),
            V("SCF-ALG", str),
            V("SCF-ITER", int),
            V("SCF-MIX", float),
            V("SCF-TOL", float),
            V("RMSAVV", float),
            V("RMSAVB", float),
            V("EF", float),
            V("VMTZ", float),
        ],
        has_hidden_members=True,
    )

    # The cluster file contains only the dimensionality/type of the
    # lattice.  All lattice vectors belong to the separate host POTFIL.
    section("LATTICE", [V("SYSDIM", str), V("SYSTYPE", str)], has_hidden_members=True)

    # N5VEC(1:5) identifies a lattice translation, IQ is the host basis
    # index, and (QX,QY,QZ) are printed for information only.  The
    # translations, IQ and optional shifts are the authoritative geometry.
    section(
        "SITES",
        [
            V("DESCRIPTION", line_string, "positions of cluster sites", name_in_grammar=False),
            V(
                "DATA",
                Table(
                    {
                        "N5VEC1": int,
                        "N5VEC2": int,
                        "N5VEC3": int,
                        "N5VEC4": int,
                        "N5VEC5": int,
                        "IQ": int,
                        "QX": float,
                        "QY": float,
                        "QZ": float,
                    },
                    numbering="IQCLU",
                    free_header=True,
                ),
                name_in_grammar=False,
            ),
        ],
        has_hidden_members=True,
    )
    section(
        "SHIFTS",
        [
            V(
                "DESCRIPTION",
                line_string,
                "shift from original lattice site positions",
                name_in_grammar=False,
            ),
            V("SHIFT", False, is_optional=True),
            V(
                "DATA",
                Table({"DQX": float, "DQY": float, "DQZ": float}, numbering="IQCLU", free_header=True),
                name_in_grammar=False,
            ),
        ],
        has_hidden_members=True,
    )
    section(
        "FORCES",
        [
            V("DESCRIPTION", line_string, "forces acting on the cluster atoms", name_in_grammar=False),
            V(
                "DATA",
                Table({"FQX": float, "FQY": float, "FQZ": float}, numbering="IQCLU", free_header=True),
                name_in_grammar=False,
            ),
        ],
        has_hidden_members=True,
        is_optional=True,
    )
    section(
        "OCCUPATION",
        [
            V(
                "DATA",
                Table(
                    {
                        "IMQCLU": int,
                        "NOQCLU": int,
                        "ITCLU CONC": Array(Sequence(int, float), as_list=tuple),
                    },
                    numbering="IQCLU",
                    free_header=True,
                    row_condition=_occupation_count_matches,
                ),
                name_in_grammar=False,
            )
        ],
        has_hidden_members=True,
    )
    # CLUPOTWR prints GLOBAL indices (NQHOST + IQCLU), not 1..NQCLU.
    section(
        "MAGNETISATION DIRECTION",
        [
            V(
                "DATA",
                Table(
                    {"IQ": int, "QMTET": float, "QMPHI": float},
                    free_header=True,
                ),
                name_in_grammar=False,
            ),
        ],
        has_hidden_members=True,
    )

    # Only present when NMCLU != 0.  IMs are likewise GLOBAL mesh IDs.
    section(
        "MESH INFORMATION",
        [
            V("MESH-TYPE", DefKeyword("EXPONENTIAL")),
            V(
                "DATA",
                Table(
                    {"IM": int, "R(1)": float, "DX": float, "JRMT": int, "RMT": float, "JRWS": int, "RWS": float},
                    free_header=True,
                ),
                name_in_grammar=False,
            ),
            V(
                "FULLPOT",
                Table(
                    {"IM": int, "JRNS1": int, "JRCRI": int, "NPAN": int, "JRCUT": Array(int)},
                    free_header=True,
                ),
                name_in_grammar=False,
                is_optional=True,
            ),
        ],
        has_hidden_members=True,
        is_optional=True,
    )

    # TYPES are locally numbered ITCLU = 1..NTCLU, while TYPE entries in
    # radial sections carry the global identifier NTHOST + ITCLU.
    section("TYPES", cls=TypesSectionDefinition)
    section("POTENTIAL", cls=PotentialSectionDefinition)
    section("CHARGE", cls=ChargeSectionDefinition)
    sections.append(
        _repeated_block(
            "MOMENTS",
            [
                V(
                    "DATA",
                    NumpyArray(
                        lines=1, shape=(5,), written_shape=(1, 5),
                        item_format="% .14E", indented=1,
                    ),
                    name_in_grammar=False,
                ),
            ],
        )
    )

    # CLUPOTWR emits optional Brooks orbital-polarization and DMFT blocks.
    sections.append(
        _repeated_block(
            "VECTOR-POTENTIAL",
            [
                V("LOPT", int),
                V(
                    "DATA",
                    NumpyArray(
                        shape=(2, -1),
                        ends_with=re.compile(r"\n?(={79}|-{79})"),
                        item_format="% .14E",
                        indented=1,
                    ),
                    name_in_grammar=False,
                    is_optional=True,
                ),
            ],
        )
    )
    sections.append(_repeated_block("SELF-ENERGY", [V("LOPT", int)]))
    section("CORE STATES", [], is_optional=True)

    return ClusterPotentialDefinition(sections)


cluster_definition = _definition()
"""Native CLUPOTRD/CLUPOTWR embedded-cluster potential grammar."""
