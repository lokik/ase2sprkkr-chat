"""Read/write tests for the CLUPOTRD/CLUPOTWR file grammar."""

from io import StringIO

import pytest

from ..cluster_potentials import ClusterPotential
from ..definitions.cluster import cluster_definition


def _section(name, *lines):
    return "\n".join(("*" * 79, name, *lines))


def _block(name, typ, *data):
    return "\n".join((_section(name, f"TYPE      {typ}", *data), "=" * 79))


def _sample(*, shifted=False, mesh=False, broken_occupation=False):
    """Small native-writer-style file; IQ and TYPE include host offsets."""
    parts = [
        "*" * 79,
        "HEADER    'SPR-KKR cluster dataset created by CLUSCF'",
        "*" * 79,
        "TITLE     'Fe substitution in Cu'",
        "SYSTEM    'Cu with an isolated Fe impurity'",
        _section("GLOBAL SYSTEM PARAMETER", "NQCLU     1", "NTCLU     1", f"NMCLU     {int(mesh)}"),
        _section(
            "SCF-INFO",
            "INFO      'embedded cluster'",
            "SCFSTATUS ITERATING",
            "FULLPOT   F",
            "BREITINT  F",
            "NONMAG    F",
            "ORBPOL    NONE",
            "EXTFIELD  F",
            "BLCOUPL   F",
            "BEXT      0.0",
            "SEMICORE  F",
            "LLOYD     F",
            "NE        32",
            "IBZINT    2",
            "NKTAB     100",
            "XC-POT    VWN",
            "SCF-ALG   BROYDEN",
            "SCF-ITER  3",
            "SCF-MIX   0.2",
            "SCF-TOL   1e-5",
            "RMSAVV    1e-4",
            "RMSAVB    1e-4",
            "EF        0.4",
            "VMTZ      0.1",
        ),
        _section("LATTICE", "SYSDIM    3D", "SYSTYPE   EMBEDDED-CLUSTER"),
        _section(
            "SITES",
            "positions of cluster sites       (QX,QY,QZ) given only for info include SHIFTS",
            "     IQCLU            N5VEC_IQCLU          IQ       QX        QY        QZ",
            "         1         0   0   0   0   0         1       0.0       0.0       0.0",
        ),
        _section(
            "SHIFTS",
            "shift from original lattice site positions",
            f"SHIFT     {'T' if shifted else 'F'}",
            "     IQCLU        DQX             DQY             DQZ",
            f"         1   {'0.25' if shifted else '0.0'}  0.0  0.0",
        ),
        _section(
            "FORCES",
            "forces acting on the cluster atoms",
            "     IQCLU        FQX             FQY             FQZ",
            "         1   0.0  0.0  0.0",
        ),
        _section(
            "OCCUPATION",
            "     IQCLU    IMQCLU    NOQCLU   ITOQCLU      CONC",
            "         1         7         "
            + ("2" if broken_occupation else "1")
            + "         1   1.00000",
        ),
        _section(
            "MAGNETISATION DIRECTION",
            "        IQ      QMTET      QMPHI",
            "        21       0.0        0.0",
        ),
    ]
    if mesh:
        parts.append(
            _section(
                "MESH INFORMATION",
                "MESH-TYPE EXPONENTIAL",
                " IM     R(1)       DX       JRMT     RMT     JRWS    RWS",
                "  7   0.000001   0.02       2       2.5      3      3.0",
            )
        )
    parts.extend(
        [
            _section(
                "TYPES",
                "ITCLU TXT_TCLU Z_TCLU NCOR_TCLU NVAL_TCLU NSEMCORSHL_TCLU",
                "     1 Fe 26 18 8 0",
            ),
            _block(
                "POTENTIAL",
                9,
                " 1.0E+00  2.0E+00  3.0E+00",
                " 0.0E+00  0.0E+00  0.0E+00",
            ),
            _block(
                "CHARGE",
                9,
                " 1.0E+00  2.0E+00  3.0E+00",
                " 0.0E+00  0.0E+00  0.0E+00",
            ),
            _block("MOMENTS", 9, " 1.0E+00  0.0E+00  2.0E+00  0.0E+00  0.0E+00"),
            _section("CORE STATES"),
        ]
    )
    return "\n".join(parts) + "\n"


def test_cluster_grammar_preserves_newlines_as_significant():
    grammar = cluster_definition.grammar()

    def inspect(expression):
        assert "\n" not in expression.whiteChars
        for child in getattr(expression, "exprs", []):
            inspect(child)
        if hasattr(expression, "expr"):
            inspect(expression.expr)

    inspect(grammar)


@pytest.mark.parametrize("shifted,mesh", [(False, False), (True, False), (True, True)])
def test_cluster_potential_reads_native_structure(shifted, mesh):
    cluster = ClusterPotential.from_string(_sample(shifted=shifted, mesh=mesh))
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NQCLU() == 1
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 1
    assert cluster.SITES.DATA()[0]["IQ"] == 1
    assert cluster.SITES.DATA()[0]["N5VEC1"] == 0
    assert cluster.SHIFTS.SHIFT() is shifted
    assert cluster.SHIFTS.DATA()[0]["DQX"] == pytest.approx(0.25 if shifted else 0.0)
    assert cluster.OCCUPATION.DATA()[0]["NOQCLU"] == 1
    assert cluster.MAGNETISATION_DIRECTION.DATA()[0]["IQ"] == 21
    assert cluster.MAGNETISATION_DIRECTION.DATA()[0]["QMTET"] == 0.0
    assert len(cluster.POTENTIAL) == 1
    assert cluster.POTENTIAL[0].TYPE() == 9
    if mesh:
        assert cluster.MESH_INFORMATION.DATA()[0]["IM"] == 7
        assert cluster.MESH_INFORMATION.DATA()[0]["JRWS"] == 3


@pytest.mark.parametrize("mesh", [False, True])
def test_cluster_potential_round_trip(mesh):
    cluster = ClusterPotential.from_string(_sample(mesh=mesh))
    output = StringIO()
    cluster.save_to_file(output, validate=False)
    restored = ClusterPotential.from_string(output.getvalue())
    assert restored.SITES.DATA()[0]["IQ"] == 1
    assert restored.MAGNETISATION_DIRECTION.DATA()[0]["IQ"] == 21
    if mesh:
        assert restored.MESH_INFORMATION.DATA()[0]["IM"] == 7
    assert restored.POTENTIAL[0].TYPE() == 9
    assert restored.OCCUPATION.DATA()[0]["NOQCLU"] == 1


def test_invalid_occupation_count_is_rejected():
    with pytest.raises(Exception):
        ClusterPotential.from_string(_sample(broken_occupation=True))


def test_embedded_atoms_to_start_cluster_potential_and_back():
    """START CLUPOT has a second magnetisation header and no radial data."""
    from types import SimpleNamespace
    import numpy as np
    from ase import Atoms
    from ...sprkkr.sprkkr_atoms import SPRKKRAtoms

    # Minimal host Potential protocol: lattice/occupations plus ASE atoms.
    host = SimpleNamespace(
        atoms=Atoms("Cu", positions=[[0., 0., 0.]], cell=[2., 2., 2.], pbc=True),
        LATTICE=SimpleNamespace(ALAT=lambda: 4.0, SYSDIM=lambda: "3D",
                                SYSTYPE=lambda: "BULK"),
        OCCUPATION=SimpleNamespace(
            DATA=lambda: np.array([(1,)], dtype=[("IMQ", int)])
        ),
    )
    atoms = SPRKKRAtoms.from_embedding(host, [(0, (1, 0, 0), "Fe")])
    atoms.sites[0].embedding.displacement = (0.1, 0.0, 0.0)
    cluster = ClusterPotential.from_atoms(atoms)

    assert cluster.SCF_INFO.SCFSTATUS() == "START"
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NQCLU() == 1
    assert cluster.SITES.DATA()[0]["N5VEC1"] == 1
    assert cluster.SITES.DATA()[0]["IQ"] == 1
    assert cluster.OCCUPATION.DATA()[0]["NOQCLU"] == 1
    assert len(cluster.POTENTIAL) == 0

    text = StringIO()
    cluster.save_to_file(text, validate=False)
    native = text.getvalue()
    assert "Initial cluster magnetisation directions" in native
    restored = ClusterPotential.from_string(native)
    result = restored.to_atoms(host)

    assert result.info["sprkkr_embedding_host"] is host
    assert result.sites[0].embedding.cell_index == (1, 0, 0)
    assert result.sites[0].occupation.as_dict == {"Fe": 1.0}
    np.testing.assert_allclose(result.positions, atoms.positions)
    np.testing.assert_allclose(result.sites[0].embedding.displacement, [0.1, 0, 0])


def test_native_start_with_two_magnetisation_headers_is_readable():
    text = _sample().replace("SCFSTATUS ITERATING", "SCFSTATUS START")
    text = text.replace(
        "MAGNETISATION DIRECTION\\n        IQ      QMTET",
        "MAGNETISATION DIRECTION\\ninitial directions\\n        IQ      QMTET",
    )
    cluster = ClusterPotential.from_string(text)
    assert cluster.SCF_INFO.SCFSTATUS() == "START"
    assert cluster.MAGNETISATION_DIRECTION.DATA()[0]["QMTET"] == 0


def _radial_host():
    """Host with eight native types and six radial meshes."""
    from types import SimpleNamespace
    import numpy as np
    from ase import Atoms

    mesh_dtype = [
        ("R(1)", float), ("DX", float), ("JRMT", int),
        ("RMT", float), ("JRWS", int), ("RWS", float),
    ]
    meshes = np.array(
        [(1e-6, 0.02, 2, 2.5, 721, 3.0)] * 6, dtype=mesh_dtype
    )
    host_occ = np.array([(1,)] * 20, dtype=[("IMQ", int)])
    return SimpleNamespace(
        atoms=Atoms("Cu20", positions=np.zeros((20, 3)),
                    cell=[2, 2, 2], pbc=True),
        LATTICE=SimpleNamespace(ALAT=lambda: 4.0, SYSDIM=lambda: "3D",
                                SYSTYPE=lambda: "BULK"),
        TYPES=SimpleNamespace(DATA=lambda: [None] * 8),
        MESH_INFORMATION=SimpleNamespace(DATA=lambda: meshes),
        OCCUPATION=SimpleNamespace(DATA=lambda: host_occ),
    )


def test_iterating_radial_data_map_to_atomic_type_and_round_trip():
    import numpy as np

    host = _radial_host()
    source = ClusterPotential.from_string(_sample(mesh=True))
    atoms = source.to_atoms(host)
    atom_type = next(iter(atoms.sites[0].occupation.items()))[0]
    assert atom_type.mesh.jrws == 3
    np.testing.assert_allclose(atom_type.potential.raw_value,
                               [[1, 2, 3], [0, 0, 0]])
    np.testing.assert_allclose(atom_type.charge.raw_value,
                               [[1, 2, 3], [0, 0, 0]])
    assert atom_type.moments.as_tuple() == (1, 0, 2, 0, 0)

    written = ClusterPotential.from_atoms(atoms)
    assert written.SCF_INFO.SCFSTATUS() == "ITERATING"
    assert written.GLOBAL_SYSTEM_PARAMETER.NMCLU() == 1
    assert written.POTENTIAL[0].TYPE() == 9
    assert written.OCCUPATION.DATA()[0]["IMQCLU"] == 7
    restored = written.to_atoms(host)
    restored_type = next(iter(restored.sites[0].occupation.items()))[0]
    np.testing.assert_allclose(restored_type.potential.raw_value,
                               atom_type.potential.raw_value)
    np.testing.assert_allclose(restored_type.charge.raw_value,
                               atom_type.charge.raw_value)
    assert restored_type.moments.as_tuple() == atom_type.moments.as_tuple()


def test_invalid_global_radial_type_is_rejected():
    host = _radial_host()
    source = _sample(mesh=True).replace("TYPE      9", "TYPE      1")
    cluster = ClusterPotential.from_string(source)
    with pytest.raises(ValueError, match="does not refer to a cluster type"):
        cluster.to_atoms(host)


def test_shared_cluster_type_remains_shared_across_start_round_trip():
    from types import SimpleNamespace
    import numpy as np
    from ase import Atoms
    from ...sprkkr.sprkkr_atoms import SPRKKRAtoms

    host = SimpleNamespace(
        atoms=Atoms("Cu2", positions=[[0, 0, 0], [1, 0, 0]],
                    cell=[2, 2, 2], pbc=True),
        LATTICE=SimpleNamespace(ALAT=lambda: 4.0, SYSDIM=lambda: "3D",
                                SYSTYPE=lambda: "BULK"),
        OCCUPATION=SimpleNamespace(DATA=lambda: np.array(
            [(1,), (1,)], dtype=[("IMQ", int)]
        )),
    )
    atoms = SPRKKRAtoms.from_embedding(
        host, [(0, (0, 0, 0), "Fe"), (1, (0, 0, 0), "Fe")]
    )
    atoms.sites[1].site_type = atoms.sites[0].site_type
    cluster = ClusterPotential.from_atoms(atoms)
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 1

    restored = cluster.to_atoms(host)
    assert restored.sites[0].site_type is restored.sites[1].site_type
    written = ClusterPotential.from_atoms(restored)
    assert written.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 1
