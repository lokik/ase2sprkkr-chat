"""Native SPR-KKR embedded-cluster potential and conversion to SPRKKRAtoms.

The host remains a separate POTFIL. START files contain the embedding, atomic
occupations, and initial moment directions, but not converged radial data.
"""

from io import StringIO
import re

import numpy as np
from ase.units import Bohr

from .potential_base import Potential
from .potentials import PeriodicPotential
from ..sprkkr.embedding import EMBEDDING_HOST, embedding_data, host_atoms


def _from_start_native(text):
    """Remove the extra native START direction heading, if present.

    This transform is safe to apply before the root-level grammar choice:
    it only recognizes CLUPOT's specific IQ/QMTET/QMPHI table header,
    never the periodic potential's different magnetisation fields.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() != "MAGNETISATION DIRECTION" or i + 2 >= len(lines):
            continue
        if re.match(r"^\s*IQ\s+QMTET\s+QMPHI\s*$", lines[i + 2], re.IGNORECASE):
            del lines[i + 1]
            break
    return "".join(lines)


def _to_start_native(text):
    if not re.search(r"(?m)^SCFSTATUS\s+START\b", text):
        return text
    return text.replace("MAGNETISATION DIRECTION\n",
                        "MAGNETISATION DIRECTION\n"
                        "Initial cluster magnetisation directions\n", 1)


def _host_potential(atoms):
    host = atoms.info.get(EMBEDDING_HOST)
    if not isinstance(host, PeriodicPotential):
        raise ValueError("Creating a CLUPOT requires sprkkr_embedding_host to be a host Potential")
    return host


def _host_value(host, name, default):
    value = host.SCF_INFO[name]()
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[-1] if len(value) else default
    return value


class EmbeddedPotential(Potential):
    """A native POTFIL_CLU and its mapping to an external host potential."""

    def __init__(self, definition=None):
        if definition is None:
            from .definitions.cluster import cluster_definition
            definition = cluster_definition
        super().__init__(definition)
        self._host = None
        self._atoms = None

    @property
    def host(self):
        """The host Potential, retained by reference (not copied)."""
        return self._host

    @host.setter
    def host(self, value):
        if value is not self._host:
            self._atoms = None
        self._host = value

    @property
    def atoms(self):
        """Embedded atoms associated with this cluster, if a host is known."""
        if self._atoms is None and self.host is not None:
            self._atoms = self.to_atoms()
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        """Keep the cluster's attached atoms compatible with its host."""
        if atoms is not None:
            host = atoms.info.get(EMBEDDING_HOST)
            if host is None:
                raise ValueError("ClusterPotential atoms require an embedding host")
            if self.host is not None and self.host is not host:
                raise ValueError("ClusterPotential atoms belong to another host")
            self.host = host
        self._atoms = atoms

    def host_geometry_warning(self, tolerance=1e-3):
        """Warn about an incompatible host using native informational Q coordinates.

        CLUPOT does not contain the host cell, so this is a heuristic, never
        an authoritative identity check. Coordinates are in units of host ALAT.
        """
        if self.host is None:
            return None
        from ..sprkkr.embedding_geometry import HostSite

        expected_dimension = str(self.LATTICE.SYSDIM())
        actual_dimension = str(self.host.LATTICE.SYSDIM())
        if expected_dimension != actual_dimension:
            return (
                f"CLUPOT expects a {expected_dimension} host, but the selected "
                f"potential is {actual_dimension}."
            )
        alat = float(self.host.LATTICE.ALAT()) * Bohr
        errors = []
        try:
            rows = self.SITES.DATA()
            shifts = self.SHIFTS.DATA()
            for row, shift in zip(rows, shifts):
                point = HostSite(
                    int(row["IQ"]) - 1,
                    tuple(int(row[f"N5VEC{i}"]) for i in range(1, 6)),
                ).position(self.host)
                point += np.array([shift["DQX"], shift["DQY"], shift["DQZ"]]) * alat
                recorded = np.array([row["QX"], row["QY"], row["QZ"]]) * alat
                errors.append(float(np.linalg.norm(point - recorded)))
        except (ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
            return f"Unable to verify the host geometry: {exc}"
        if errors and max(errors) > tolerance:
            return (
                "The host geometry may not match this CLUPOT "
                f"(maximum site-position deviation {max(errors):.4g} Å)."
            )
        return None

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
            scf_status = atoms.info.get("sprkkr_embedding_scf_status")
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
            host_types = host.TYPES.DATA()
            host_mesh_data = host.MESH_INFORMATION.DATA()
            if host_types is None or host_mesh_data is None:
                raise ValueError("Writing radial cluster data requires host TYPES and MESH INFORMATION")
            host_type_count = len(host_types)
            # Only match ordinary host meshes. FULLPOT host meshes may have
            # different radial panel information not present in DATA().
            host_mesh_fullpot = host.MESH_INFORMATION.FULLPOT()
            mesh_lookup = {} if host_mesh_fullpot is not None else {
                tuple(row[name] for name in ("R(1)", "DX", "JRMT", "RMT", "JRWS", "RWS")) + (None,): i
                for i, row in enumerate(host_mesh_data, 1)
            }
            for site in atoms.sites:
                mesh = site.site_type.mesh
                # Distinguish independent FULLPOT meshes even when their
                # spherical part is numerically identical.
                extra = mesh.fullpot_tuple()
                extra = None if extra is None else (
                    extra[0], extra[1], extra[2], tuple(extra[3])
                )
                signature = tuple(mesh.to_tuple()) + (extra,)
                number = mesh_lookup.get(signature)
                if number is None:
                    number = len(host_mesh_data) + len(cluster_meshes) + 1
                    cluster_meshes.append((number, mesh))
                    mesh_lookup[signature] = number
                mesh_ids.append(number)
        # CLUSYMMETRY assigns displaced meshes when reading START.
        nm = len(cluster_meshes) if with_radial else (n if shifted else 0)
        has_fullpot = with_radial and any(
            t.potential.fullpot_data or t.charge.fullpot_data for t in types
        )
        if has_fullpot:
            if any(not t.potential.fullpot_data or not t.charge.fullpot_data for t in types):
                raise ValueError("Every cluster type needs both FULLPOT potential and charge data")
            if any(site.mesh.fullpot is None for site in atoms.sites):
                raise ValueError("FULLPOT radial data require matching FULLPOT mesh information")
        # Fill the same section definitions used by the CLUPOT parser.
        # Do not render native text and parse it back into a second object:
        # that would duplicate the grammar and hide formatting differences.
        cluster = cls()
        cluster.host = host
        cluster.HEADER.HEADER = "SPR-KKR cluster dataset created by ASE2SPRKKR"
        cluster.HEADER.TITLE = title
        cluster.HEADER.SYSTEM = "Embedded cluster in a host potential"
        cluster.GLOBAL_SYSTEM_PARAMETER.NQCLU = n
        cluster.GLOBAL_SYSTEM_PARAMETER.NTCLU = len(types)
        cluster.GLOBAL_SYSTEM_PARAMETER.NMCLU = nm

        scf_values = {
            "INFO": "Initial embedded cluster",
            "SCFSTATUS": scf_status,
            "FULLPOT": has_fullpot or _host_value(host, "FULLPOT", False),
            "BREITINT": _host_value(host, "BREITINT", False),
            "NONMAG": _host_value(host, "NONMAG", False),
            "ORBPOL": _host_value(host, "ORBPOL", "NONE"),
            "EXTFIELD": _host_value(host, "EXTFIELD", False),
            "BLCOUPL": _host_value(host, "BLCOUPL", False),
            "BEXT": _host_value(host, "BEXT", 0.0),
            "SEMICORE": _host_value(host, "SEMICORE", False),
            "LLOYD": _host_value(host, "LLOYD", False),
            "NE": _host_value(host, "NE", 32),
            "IBZINT": _host_value(host, "IBZINT", 2),
            "NKTAB": _host_value(host, "NKTAB", 100),
            "XC-POT": _host_value(host, "XC-POT", "VWN"),
            "SCF-ALG": _host_value(host, "SCF-ALG", "BROYDEN"),
            "SCF-ITER": 0,
            "SCF-MIX": _host_value(host, "SCF-MIX", 0.2),
            "SCF-TOL": _host_value(host, "SCF-TOL", 1e-5),
            "RMSAVV": 999999.0,
            "RMSAVB": 999999.0,
            "EF": _host_value(host, "EF", 0.0),
            "VMTZ": _host_value(host, "VMTZ", 0.7),
        }
        cluster.SCF_INFO.set(scf_values)
        cluster.LATTICE.SYSDIM = host.LATTICE.SYSDIM()
        cluster.LATTICE.SYSTYPE = host.LATTICE.SYSTYPE()

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
            x, y, z = atoms.positions[i - 1] / alat
            site_rows.append((*map(int, row[1:6]), original + 1, x, y, z))
            shift_rows.append(tuple(site.embedding.displacement / alat))
            theta, phi = site.magnetisation
            mag_rows.append((len(reference) + i, float(theta), float(phi)))
            mesh = mesh_ids[i - 1] if with_radial else int(host_occupations[original]["IMQ"])
            occ_rows.append((mesh, len(components), tuple(components)))

        cluster.SITES.DATA = site_rows
        cluster.SHIFTS.SHIFT = shifted
        cluster.SHIFTS.DATA = shift_rows
        cluster.OCCUPATION.DATA = occ_rows
        cluster.MAGNETISATION_DIRECTION.DATA = mag_rows
        cluster.TYPES.DATA = [
            (at.symbol, at.atomic_number, at.n_core, at.n_valence, at.n_semicore)
            for at in types
        ]

        if with_radial:
            if cluster_meshes:
                cluster.MESH_INFORMATION.DATA = [
                    (number, *mesh.to_tuple())
                    for number, mesh in cluster_meshes
                ]
                fullpot_meshes = [
                    (number, mesh.fullpot_tuple())
                    for number, mesh in cluster_meshes if mesh.fullpot is not None
                ]
                if fullpot_meshes:
                    if len(fullpot_meshes) != len(cluster_meshes):
                        raise ValueError("Mixed FULLPOT/ordinary cluster mesh definitions")
                    cluster.MESH_INFORMATION.FULLPOT = [
                        (number, values[0], values[1], values[2], values[3])
                        for number, values in fullpot_meshes
                    ]
            # Native radial blocks have GLOBAL type numbers, while TYPES and
            # OCCUPATION use ITCLU = 1..NTCLU.  Preserve that distinction.
            for name in ("POTENTIAL", "CHARGE", "MOMENTS"):
                for i, atom_type in enumerate(types, 1):
                    global_type = host_type_count + i
                    fullpot_data = None
                    if name == "MOMENTS":
                        values = atom_type.moments.as_tuple()
                    else:
                        radial = getattr(atom_type, name.lower())
                        values = np.asarray(radial.raw_value)
                        fullpot_data = radial.fullpot_data
                        mesh = atom_type.mesh
                        if values.shape != (2, len(mesh)):
                            raise ValueError(
                                f"{name} for cluster type {i} has {values.shape}, "
                                f"expected (2, {len(mesh)})"
                            )
                    block = getattr(cluster, name).add()
                    block.TYPE = global_type
                    block.DATA = values
                    if fullpot_data:
                        if not fullpot_data.lstrip().startswith("NFP"):
                            raise ValueError("Non-spherical radial data must contain an NFP header")
                        block.FULLPOT = fullpot_data

        cluster._atoms = atoms
        return cluster

    def to_atoms(self, host=None):
        """Convert cluster geometry, occupations and available radial data to atoms.

        Host and cluster TYPE/mesh numbers are independent. Cluster radial TYPE
        records use the native global host-plus-cluster numbering; occupation
        TYPE references are local ITCLU values.
        """
        from ..sprkkr.atomic_types import AtomicType
        from ..sprkkr.radial_meshes import ExponentialMesh, Mesh
        from ..sprkkr.sprkkr_atoms import SPRKKRAtoms

        if host is None:
            host = self.host
        reference = host_atoms(host)
        self.host = host
        if len(reference) == 0:
            raise ValueError("Host has no sites")
        if not isinstance(host, PeriodicPotential):
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
        host_mesh_data = host.MESH_INFORMATION.DATA()
        if host_mesh_data is not None:
            for number, row in enumerate(host_mesh_data, 1):
                meshes[number] = ExponentialMesh(
                    *(row[name] for name in ("R(1)", "DX", "JRMT", "RMT", "JRWS", "RWS"))
                )
        if self.MESH_INFORMATION.DATA() is not None:
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
            host_types = host.TYPES.DATA()
            if host_types is None:
                raise ValueError("Reading radial cluster data requires host TYPES")
            n_host_types = len(host_types)
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
                atom_type = types[local_index - 1]
                setattr(atom_type, property_name, block.DATA())
                if property_name in ("potential", "charge"):
                    getattr(atom_type, property_name).fullpot_data = block.FULLPOT()

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

        # Reuse meshes and AtomicTypes according to native identities.
        # Sharing ITCLU does *not* imply that the sites are symmetry equivalent,
        # so keep individual SiteType instances.
        typed_for_mesh = {}
        radial_type_mesh = {}
        for index, (site, occ) in enumerate(zip(atoms.sites, occ_rows)):
            mesh_id = int(occ["IMQCLU"])
            mesh = meshes.get(mesh_id)
            if mesh is None:
                if radial_blocks:
                    raise ValueError(f"Unknown radial mesh IMQCLU={mesh_id}")
                mesh = meshes.setdefault(mesh_id, Mesh.default())
            site.site_type.mesh = mesh
            occupation = {}
            for type_index, concentration in occ["ITCLU CONC"]:
                type_index = int(type_index)
                if not 1 <= type_index <= len(types):
                    raise ValueError(f"Unknown cluster type {type_index}")
                key = (type_index, mesh_id)
                if radial_blocks:
                    existing_mesh = radial_type_mesh.setdefault(type_index, mesh_id)
                    if existing_mesh != mesh_id:
                        raise ValueError(
                            f"Cluster type {type_index} is used with several radial meshes"
                        )
                if key not in typed_for_mesh:
                    typed_for_mesh[key] = types[type_index - 1].for_mesh(mesh)
                atomic_type = typed_for_mesh[key]
                fraction = float(concentration)
                occupation[atomic_type] = fraction
            site.occupation = occupation

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

        atoms.info["sprkkr_embedding_scf_status"] = self.SCF_INFO.SCFSTATUS()
        atoms._potential = self
        self._atoms = atoms
        return atoms


# Compatibility alias for the pre-refactoring embedded potential API.
ClusterPotential = EmbeddedPotential
