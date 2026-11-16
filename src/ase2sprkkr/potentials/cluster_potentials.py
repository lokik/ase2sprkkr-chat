"""Native SPR-KKR embedded-cluster potential and conversion to SPRKKRAtoms.

The host remains a separate POTFIL. START files contain the embedding, atomic
occupations, and initial moment directions, but not converged radial data.
"""

from io import StringIO
import re

import numpy as np
from ase.units import Bohr

from ..sprkkr.configuration import ConfigurationFile
from ..sprkkr.embedding import EMBEDDING_ARRAY, EMBEDDING_HOST, embedding_data, host_atoms


def _section(name, *lines):
    return "\n".join(("*" * 79, name, *lines))


def _from_start_native(text):
    """CLUPOTRD reads two magnetisation header lines for START, one otherwise.

    The grammar's table parser needs just one, so discard the additional
    descriptive line on input. All numerical rows are kept verbatim.
    """
    if not re.search(r"(?m)^SCFSTATUS\s+START\b", text):
        return text
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() == "MAGNETISATION DIRECTION" and i + 2 < len(lines):
            # A native START file has two successive non-numerical header
            # lines before the first row. A one-header file also parses.
            def is_data(line):
                return bool(re.match(r"^\s*\d+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+", line))
            if not is_data(lines[i + 2]):
                del lines[i + 1]
            break
    return "".join(lines)


def _to_start_native(text):
    if not re.search(r"(?m)^SCFSTATUS\s+START\b", text):
        return text
    return text.replace("MAGNETISATION DIRECTION\n",
                        "MAGNETISATION DIRECTION\n"
                        "Initial cluster magnetisation directions\n", 1)


def _radial_block(name, type_number, values):
    """Serialize native spherical radial data for one global TYPE."""
    output = StringIO()
    np.savetxt(output, np.atleast_2d(values), fmt="% .14E")
    return _section(
        name, f"TYPE      {type_number}", output.getvalue().rstrip("\n"), "=" * 79
    )


def _host_potential(atoms):
    host = atoms.info.get(EMBEDDING_HOST)
    if host is None or not hasattr(host, "LATTICE") or not hasattr(host, "OCCUPATION"):
        raise ValueError("Creating a CLUPOT requires sprkkr_embedding_host to be a host Potential")
    return host


def _host_value(host, name, default):
    try:
        value = host.SCF_INFO[name]()
        if value is None:
            return default
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[-1] if len(value) else default
        return value
    except (AttributeError, KeyError, TypeError):
        return default


class ClusterPotential(ConfigurationFile):
    """A native POTFIL_CLU and its mapping to an external host potential."""

    def __init__(self, definition=None):
        if definition is None:
            from .definitions.cluster import cluster_definition
            definition = cluster_definition
        super().__init__(definition)

    @classmethod
    def from_file(cls, filename, *, allow_dangerous=False):
        """Read a native cluster potential, preserving all five translations."""
        from .definitions.cluster import cluster_definition
        if hasattr(filename, "read"):
            content = filename.read()
        else:
            with open(filename, encoding="utf-8") as file:
                content = file.read()
        return cluster_definition.read_from_file(
            StringIO(_from_start_native(content)), allow_dangerous=allow_dangerous
        )

    @classmethod
    def from_string(cls, content, *, allow_dangerous=False):
        return cls.from_file(StringIO(content), allow_dangerous=allow_dangerous)

    def save_to_file(self, file, *, validate="save"):
        """Write the native format, including START-specific header handling."""
        out = StringIO()
        super().save_to_file(out, atoms=False, validate=validate)
        text = _to_start_native(out.getvalue())
        if hasattr(file, "write"):
            file.write(text)
            file.flush()
        else:
            with open(file, "w", encoding="utf-8") as target:
                target.write(text)

    @classmethod
    def from_atoms(cls, atoms, *, title="Embedded cluster", scf_status=None):
        """Create START or iterating CLUPOT, including spherical radial data.

        In the absence of radial data the result is START. Complete radial
        data produce ITERATING, unless scf_status is explicitly specified.
        """
        from ..sprkkr.sprkkr_atoms import SPRKKRAtoms
        SPRKKRAtoms.promote_ase_atoms(atoms)
        host = _host_potential(atoms)
        reference = host_atoms(host)
        rows = embedding_data(atoms)
        identities = [tuple(int(v) for v in row) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("A host site and cell translation may only occur once in a cluster")
        n = len(atoms)
        if n == 0:
            raise ValueError("An embedded cluster must contain at least one site")
        alat = float(host.LATTICE.ALAT()) * Bohr
        if alat <= 0:
            raise ValueError("Invalid host ALAT")

        # A native ITCLU identifies an AtomicType, independently of SiteType.
        known_types = {}
        types = []
        occupations = []
        for site in atoms.sites:
            components = []
            for atom_type, fraction in site.occupation.items():
                key = id(atom_type)
                if key not in known_types:
                    known_types[key] = len(types) + 1
                    types.append(atom_type)
                components.append((known_types[key], float(fraction)))
            if not components:
                raise ValueError("Each embedded site needs at least one atomic type")
            occupations.append(components)

        complete_radial = all(t.has_converged_data() for t in types)
        any_radial = any(
            t.potential is not None or t.charge is not None or t.moments is not None
            for t in types
        )
        if scf_status is None:
            if any_radial and not complete_radial:
                raise ValueError(
                    "Partial radial data cannot be serialized implicitly; "
                    "use scf_status='START' to intentionally discard them"
                )
            scf_status = "ITERATING" if complete_radial else "START"
        scf_status = str(scf_status).upper()
        if scf_status not in ("START", "ITERATING", "CONVERGED"):
            raise ValueError(f"Unsupported cluster SCFSTATUS {scf_status!r}")
        with_radial = scf_status != "START"
        if with_radial and not complete_radial:
            raise ValueError("Writing an iterating/converged CLUPOT requires complete radial data")

        shifted = any(np.linalg.norm(site.embedding.displacement) > 1e-10 for site in atoms.sites)
        mesh_ids = []
        cluster_meshes = []
        host_type_count = 0
        if with_radial:
            if not hasattr(host, "TYPES") or not hasattr(host, "MESH_INFORMATION"):
                raise ValueError("Writing radial cluster data requires host TYPES and MESH INFORMATION")
            host_type_count = len(host.TYPES.DATA())
            host_mesh_data = host.MESH_INFORMATION.DATA()
            mesh_lookup = {
                tuple(row[name] for name in ("R(1)", "DX", "JRMT", "RMT", "JRWS", "RWS")): i
                for i, row in enumerate(host_mesh_data, 1)
            }
            for site in atoms.sites:
                mesh = site.site_type.mesh
                if mesh.fullpot is not None:
                    raise NotImplementedError("Writing full-potential cluster meshes is not implemented")
                signature = tuple(mesh.to_tuple())
                number = mesh_lookup.get(signature)
                if number is None:
                    number = len(host_mesh_data) + len(cluster_meshes) + 1
                    cluster_meshes.append((number, mesh))
                    mesh_lookup[signature] = number
                mesh_ids.append(number)
        # CLUSYMMETRY assigns displaced meshes when reading START.
        nm = len(cluster_meshes) if with_radial else (n if shifted else 0)
        lines = [
            "*" * 79,
            "HEADER    'SPR-KKR cluster dataset created by ASE2SPRKKR'",
            "*" * 79,
            f"TITLE     '{title}'",
            "SYSTEM    'Embedded cluster in a host potential'",
            _section("GLOBAL SYSTEM PARAMETER",
                     f"NQCLU     {n}", f"NTCLU     {len(types)}", f"NMCLU     {nm}"),
            _section("SCF-INFO",
                     "INFO      'Initial embedded cluster'",
                     f"SCFSTATUS {scf_status}",
                     f"FULLPOT   {'T' if _host_value(host, 'FULLPOT', False) else 'F'}",
                     f"BREITINT  {'T' if _host_value(host, 'BREITINT', False) else 'F'}",
                     f"NONMAG    {'T' if _host_value(host, 'NONMAG', False) else 'F'}",
                     f"ORBPOL    {_host_value(host, 'ORBPOL', 'NONE')}",
                     f"EXTFIELD  {'T' if _host_value(host, 'EXTFIELD', False) else 'F'}",
                     f"BLCOUPL   {'T' if _host_value(host, 'BLCOUPL', False) else 'F'}",
                     f"BEXT      {_host_value(host, 'BEXT', 0.0)}",
                     f"SEMICORE  {'T' if _host_value(host, 'SEMICORE', False) else 'F'}",
                     f"LLOYD     {'T' if _host_value(host, 'LLOYD', False) else 'F'}",
                     f"NE        {_host_value(host, 'NE', 32)}",
                     f"IBZINT    {_host_value(host, 'IBZINT', 2)}",
                     f"NKTAB     {_host_value(host, 'NKTAB', 100)}",
                     f"XC-POT    {_host_value(host, 'XC-POT', 'VWN')}",
                     f"SCF-ALG   {_host_value(host, 'SCF-ALG', 'BROYDEN')}",
                     "SCF-ITER  0",
                     f"SCF-MIX   {_host_value(host, 'SCF-MIX', 0.2)}",
                     f"SCF-TOL   {_host_value(host, 'SCF-TOL', 1e-5)}",
                     "RMSAVV    999999.0",
                     "RMSAVB    999999.0",
                     f"EF        {_host_value(host, 'EF', 0.0)}",
                     f"VMTZ      {_host_value(host, 'VMTZ', 0.7)}"),
            _section("LATTICE",
                     f"SYSDIM    {host.LATTICE.SYSDIM()}",
                     f"SYSTYPE   {host.LATTICE.SYSTYPE()}"),
        ]
        site_rows = []
        shift_rows = []
        mag_rows = []
        occ_rows = []
        host_occupations = host.OCCUPATION.DATA()
        for i, (site, row, components) in enumerate(zip(atoms.sites, rows, occupations), 1):
            original = int(row[0])
            if not 0 <= original < len(reference):
                raise ValueError(f"Host-site index {original} is invalid")
            if host.LATTICE.SYSDIM() == "2D":
                n_left = int(host.LATTICE.NQ_L())
                n_right = int(host.LATTICE.NQ_R())
                if original < n_left or original >= len(reference) - n_right:
                    raise ValueError("CLUPOT cannot embed a site in the semi-infinite bulk")
            native = " ".join(map(str, row[1:6]))
            x, y, z = atoms.positions[i - 1] / alat
            site_rows.append(f"{i} {native} {original + 1} {x:.14g} {y:.14g} {z:.14g}")
            dx, dy, dz = site.embedding.displacement / alat
            shift_rows.append(f"{i} {dx:.14g} {dy:.14g} {dz:.14g}")
            theta, phi = site.magnetisation
            mag_rows.append(f"{len(reference) + i} {float(theta):.14g} {float(phi):.14g}")
            mesh = mesh_ids[i - 1] if with_radial else int(host_occupations[original]["IMQ"])
            fields = " ".join(f"{it} {fraction:.14g}" for it, fraction in components)
            occ_rows.append(f"{i} {mesh} {len(components)} {fields}")

        lines += [
            _section("SITES",
                     "positions of cluster sites (QX,QY,QZ) include SHIFTS",
                     "IQCLU N5VEC1 N5VEC2 N5VEC3 N5VEC4 N5VEC5 IQ QX QY QZ",
                     *site_rows),
            _section("SHIFTS",
                     "shift from original lattice site positions",
                     f"SHIFT     {'T' if shifted else 'F'}",
                     "IQCLU DQX DQY DQZ", *shift_rows),
            _section("OCCUPATION", "IQCLU IMQCLU NOQCLU ITCLU CONC", *occ_rows),
            _section("MAGNETISATION DIRECTION",
                     "initial magnetisation directions",
                     "IQ QMTET QMPHI", *mag_rows),
            _section("TYPES",
                     "ITCLU TXT_TCLU Z_TCLU NCOR_TCLU NVAL_TCLU NSEMCORSHL_TCLU",
                     *(f"{i} {at.symbol} {at.atomic_number} {at.n_core} "
                       f"{at.n_valence} {at.n_semicore}"
                       for i, at in enumerate(types, 1))),
        ]
        if with_radial:
            if cluster_meshes:
                # MESH INFORMATION precedes TYPES in the native grammar.
                lines.insert(-1, _section(
                    "MESH INFORMATION", "MESH-TYPE EXPONENTIAL",
                    "IM R(1) DX JRMT RMT JRWS RWS",
                    *(f"{number} " + " ".join(str(value) for value in mesh.to_tuple())
                      for number, mesh in cluster_meshes),
                ))
            # Repeated radial sections must remain grouped by section name,
            # not interleaved by TYPE.
            for name in ("POTENTIAL", "CHARGE", "MOMENTS"):
                for i, atom_type in enumerate(types, 1):
                    global_type = host_type_count + i
                    if name == "MOMENTS":
                        values = atom_type.moments.as_tuple()
                    else:
                        value = getattr(atom_type, name.lower())
                        values = np.asarray(value.raw_value)
                        mesh = atom_type.mesh
                        if values.shape != (2, len(mesh)):
                            raise ValueError(
                                f"{name} for cluster type {i} has {values.shape}, "
                                f"expected (2, {len(mesh)})"
                            )
                    lines.append(_radial_block(name, global_type, values))
        return cls.from_string("\n".join(lines) + "\n")

    def to_atoms(self, host):
        """Convert cluster geometry, occupations and available radial data to atoms.

        Host and cluster TYPE/mesh numbers are independent. Cluster radial TYPE
        records use the native global host-plus-cluster numbering; occupation
        TYPE references are local ITCLU values.
        """
        from ..sprkkr.atomic_types import AtomicType
        from ..sprkkr.radial_meshes import ExponentialMesh, Mesh
        from ..sprkkr.sprkkr_atoms import SPRKKRAtoms

        reference = host_atoms(host)
        if len(reference) == 0:
            raise ValueError("Host has no sites")
        if not hasattr(host, "LATTICE"):
            raise ValueError("Reading CLUPOT into Atoms requires a host Potential")
        alat = float(host.LATTICE.ALAT()) * Bohr
        if alat <= 0:
            raise ValueError("Invalid host ALAT")

        types = [
            AtomicType(
                row["TXT"], atomic_number=int(row["ZT"]),
                n_core=int(row["NCORT"]), n_valence=int(row["NVALT"]),
                n_semicore=int(row["NSEMCORSHLT"]),
            )
            for row in self.TYPES.DATA()
        ]
        site_rows = self.SITES.DATA()
        occ_rows = self.OCCUPATION.DATA()
        shift_rows = self.SHIFTS.DATA()
        if not (len(site_rows) == len(occ_rows) == len(shift_rows)):
            raise ValueError("Inconsistent cluster site/occupation/shift counts")
        if len(site_rows) != int(self.GLOBAL_SYSTEM_PARAMETER.NQCLU()):
            raise ValueError("NQCLU does not match the number of cluster sites")
        if len(types) != int(self.GLOBAL_SYSTEM_PARAMETER.NTCLU()):
            raise ValueError("NTCLU does not match the number of cluster types")

        # Collect all meshes by their *native global* identifiers. A cluster
        # can reuse a host mesh, or define a new one in MESH INFORMATION.
        meshes = {}
        if hasattr(host, "MESH_INFORMATION"):
            for number, row in enumerate(host.MESH_INFORMATION.DATA(), 1):
                meshes[number] = ExponentialMesh(
                    *(row[name] for name in ("R(1)", "DX", "JRMT", "RMT", "JRWS", "RWS"))
                )
        if hasattr(self, "MESH_INFORMATION") and self.MESH_INFORMATION.DATA() is not None:
            for row in self.MESH_INFORMATION.DATA():
                index = int(row["IM"])
                if index in meshes:
                    raise ValueError(f"Cluster mesh {index} conflicts with a host mesh")
                meshes[index] = ExponentialMesh(
                    *(row[name] for name in ("R(1)", "DX", "JRMT", "RMT", "JRWS", "RWS"))
                )
            fullpot = self.MESH_INFORMATION.FULLPOT()
            if fullpot is not None:
                for row in fullpot:
                    index = int(row["IM"])
                    if index not in meshes:
                        raise ValueError(f"FULLPOT refers to unknown mesh {index}")
                    meshes[index].set_fullpot((
                        int(row["JRNS1"]), int(row["JRCRI"]),
                        int(row["NPAN"]), row["JRCUT"],
                    ))

        # These sections use global TYPE = NTHOST + ITCLU, unlike the local
        # ITCLU indices used by the OCCUPATION table.
        radial_blocks = []
        for section_name, property_name in (
            ("POTENTIAL", "potential"),
            ("CHARGE", "charge"),
            ("MOMENTS", "moments"),
        ):
            for _, block in getattr(self, section_name).items():
                radial_blocks.append((section_name, property_name, block))
        if radial_blocks:
            if not hasattr(host, "TYPES"):
                raise ValueError("Reading radial cluster data requires host TYPES")
            n_host_types = len(host.TYPES.DATA())
            present = set()
            for section_name, property_name, block in radial_blocks:
                global_index = int(block.TYPE())
                local_index = global_index - n_host_types
                if not 1 <= local_index <= len(types):
                    raise ValueError(
                        f"{section_name} TYPE {global_index} does not refer to a cluster type"
                    )
                key = (property_name, local_index)
                if key in present:
                    raise ValueError(f"Duplicate {section_name} TYPE {global_index}")
                present.add(key)
                if property_name in ("potential", "charge"):
                    if block.FULLPOT() not in (None, ""):
                        raise NotImplementedError(
                            "Non-spherical FULLPOT radial data cannot be represented in AtomicType"
                        )
                setattr(types[local_index - 1], property_name, block.DATA())

        # Build geometry without handing the canonical types to from_embedding:
        # that constructor initially allocates default meshes, and would
        # remesh (copy/interpolate) the radial data before we assign the
        # correct cluster mesh.
        entries = []
        for row, occ in zip(site_rows, occ_rows):
            original = int(row["IQ"]) - 1
            if not 0 <= original < len(reference):
                raise ValueError(f"Invalid host-site index {original + 1}")
            indices = tuple(int(row[f"N5VEC{i}"]) for i in range(1, 6))
            components = occ["ITCLU CONC"]
            if not components:
                raise ValueError("Cluster sites need at least one atomic type")
            first_type = int(components[0][0])
            if not 1 <= first_type <= len(types):
                raise ValueError(f"Unknown cluster type {first_type}")
            symbol = types[first_type - 1].symbol
            entries.append((original, indices, "X" if symbol == "Vc" else symbol))

        atoms = SPRKKRAtoms.from_embedding(host, entries)

        # Reuse both the mesh and AtomicType objects where native identifiers
        # are equal. SiteTypes are shared only for identical occupations on
        # the same mesh; independent types with the same element remain distinct.
        typed_for_mesh = {}
        shared_site_types = {}
        for index, (site, occ) in enumerate(zip(atoms.sites, occ_rows)):
            mesh_id = int(occ["IMQCLU"])
            mesh = meshes.get(mesh_id)
            if mesh is None:
                if radial_blocks:
                    raise ValueError(f"Unknown radial mesh IMQCLU={mesh_id}")
                mesh = meshes.setdefault(mesh_id, Mesh.default())
            site.site_type.mesh = mesh
            occupation = {}
            identity = []
            for type_index, concentration in occ["ITCLU CONC"]:
                type_index = int(type_index)
                if not 1 <= type_index <= len(types):
                    raise ValueError(f"Unknown cluster type {type_index}")
                key = (type_index, mesh_id)
                if key not in typed_for_mesh:
                    typed_for_mesh[key] = types[type_index - 1].for_mesh(mesh)
                atomic_type = typed_for_mesh[key]
                fraction = float(concentration)
                occupation[atomic_type] = fraction
                identity.append((type_index, fraction))
            site.occupation = occupation
            group = (mesh_id, tuple(identity))
            if group in shared_site_types:
                site.site_type = shared_site_types[group]
            else:
                shared_site_types[group] = site.site_type

            shift = shift_rows[index]
            site.embedding.displacement = (
                np.asarray([shift["DQX"], shift["DQY"], shift["DQZ"]]) * alat
            )

        # Native indices for these rows are NQHOST + IQCLU (1-based).
        directions = {}
        for row in self.MAGNETISATION_DIRECTION.DATA():
            number = int(row["IQ"])
            if number in directions:
                raise ValueError(f"Duplicate magnetisation direction for site {number}")
            directions[number] = row
        for i, site in enumerate(atoms.sites, 1):
            number = len(reference) + i
            if number not in directions:
                raise ValueError(f"Missing magnetisation direction for site {number}")
            row = directions[number]
            site.magnetisation = np.array((row["QMTET"], row["QMPHI"]))

        return atoms
