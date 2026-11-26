"""Common SPR-KKR potential API and native-grammar dispatch.

The public Potential entry point parses POTFIL and CLUPOT with a single
pyparsing alternative expression. Each concrete subclass uses only its native
grammar. No NQ/NQCLU text probe, file rewind or parser retry is involved.
"""

from functools import lru_cache
from io import StringIO
from os import PathLike
from pathlib import Path
import re

import pyparsing as pp

from kkrunner.files.fingerprints import file_fingerprint

from ..common.decorators import class_property, cache
from ..common.file_utils import filename_from_file
from ..common.grammar import generate_grammar
from ..common.warnings import ValidationResult, catch_warnings_of
from ..sprkkr.configuration import ConfigurationFile
from ..sprkkr.embedding import EMBEDDING_HOST
from ..sprkkr.sprkkr_atoms import SPRKKRAtoms
from .scf_state import PotentialScfState


class PotentialFileChanged(RuntimeError):
    """A potential file changed while it was being parsed."""


class _NativeAlternative(pp.ParseElementEnhance):
    """Keep a native grammar's committed failures local to its alternative."""

    def parseImpl(self, instring, loc, do_actions=True):
        try:
            return self.expr._parse(instring, loc, do_actions)
        except pp.ParseFatalException as exc:
            raise pp.ParseException(
                exc.pstr, exc.loc, exc.msg, exc.parser_element
            ) from None


@lru_cache(maxsize=4)
def _file_grammar(allow_dangerous=False, kind=None):
    """Build the two native grammar alternatives once, without mutation.

    Grouping the alternatives names their parse results while keeping the
    original root grammar and its parse_return() representation intact.
    """
    from .definitions.potential import potential_definition
    from .definitions.cluster import cluster_definition

    definitions = {
        "periodic": potential_definition,
        "embedded": cluster_definition,
    }
    names = (kind,) if kind else ("periodic", "embedded")
    # Naming a Group copies its child grammar using the current whitespace
    # defaults. Keep newlines significant throughout wrapper construction.
    with generate_grammar():
        alternatives = []
        for name in names:
            grammar = definitions[name].grammar(allow_dangerous=allow_dangerous)
            if kind is None:
                grammar = _NativeAlternative(grammar)
            alternatives.append(pp.Group(grammar)(name))
        # Or probes without parse actions, but native grammars depend on those
        # actions for delimiters and table lengths. Their headers distinguish
        # the two formats, so ordered alternatives can parse them normally.
        return (alternatives[0] if len(alternatives) == 1
                else pp.MatchFirst(alternatives))


class Potential(PotentialScfState, ConfigurationFile):
    """Base of periodic and embedded potentials, with a unified reader.

    For backward compatibility, direct Potential() construction returns a
    PeriodicPotential. Explicitly use PeriodicPotential or EmbeddedPotential
    when the native format is known.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Potential:
            from .potentials import PeriodicPotential
            return super().__new__(PeriodicPotential)
        return super().__new__(cls)

    @class_property
    @cache
    def potential_definition(cls):
        """Periodic definition retained for historical Potential usage."""
        from .definitions.potential import potential_definition
        return potential_definition

    @classmethod
    def from_file(
        cls, filename, atoms=None, allow_dangerous=False, *,
        host=None, check_file_unchanged=False,
    ):
        """Read POTFIL or CLUPOT from a path, seekable fd or one-pass stream.

        A base-class call uses one alternative grammar; a concrete subclass accepts
        only its own native grammar. A host is optional for parsing CLUPOT,
        but required to reconstruct its ASE atoms.
        """
        from .potentials import PeriodicPotential
        from .cluster_potentials import EmbeddedPotential, _from_start_native
        from .definitions.potential import potential_definition
        from .definitions.cluster import cluster_definition

        if cls is Potential:
            kind = None
        elif issubclass(cls, EmbeddedPotential):
            kind = "embedded"
        elif issubclass(cls, PeriodicPotential):
            kind = "periodic"
        else:
            raise TypeError(f"Unsupported Potential subclass: {cls!r}")

        path = None
        fingerprint = None
        if check_file_unchanged:
            if not isinstance(filename, (str, bytes, PathLike)):
                raise TypeError(
                    "check_file_unchanged=True requires a filesystem path"
                )
            path = Path(filename)
            fingerprint = file_fingerprint(path)

        if hasattr(filename, "read"):
            contents = filename.read()
        else:
            with open(filename, encoding="utf-8") as fd:
                contents = fd.read()
        if isinstance(contents, bytes):
            contents = contents.decode("utf-8")

        # CLUPOT START has an extra descriptive header before its direction
        # table. Normalize that unambiguous table layout before grammar choice.
        contents = _from_start_native(contents)

        # Both full grammars are tried within a single pyparsing expression;
        # no retry and no separate, possibly fragile format discriminator.
        with catch_warnings_of(ValidationResult) as parse_issues:
            parsed = _file_grammar(allow_dangerous, kind).parse_string(
                contents, parse_all=True
            )

        definitions = {
            "periodic": potential_definition,
            "embedded": cluster_definition,
        }
        selected = next(name for name in definitions if name in parsed)
        definition = definitions[selected]
        values = definition.parse_return(parsed[selected])
        potential = definition.result_class(definition=definition)

        # Mirror RootConfigurationContainer.read_from_file(), without
        # re-parsing the stream (and retaining parse-phase diagnostics).
        with potential._mutation("parse") as (transaction, policy):
            policy.add_semantic(parse_issues)
            potential.stage_clear(transaction, check_required=False)
            potential.stage(transaction, values, unknown="add")

        potential._filename = filename_from_file(filename, None)
        if selected == "periodic":
            potential.make_complete()
            if atoms is not False:
                potential._atoms = potential.update_atoms(
                    None if atoms is None else atoms
                )
        else:
            if host is None and atoms is not None and atoms is not False:
                host = atoms.info.get(EMBEDDING_HOST)
            if isinstance(host, (str, bytes, PathLike)) or hasattr(host, "read"):
                host = PeriodicPotential.from_file(host)
            potential.host = host
            if atoms is not None and atoms is not False:
                potential.atoms = atoms

        if check_file_unchanged:
            try:
                current = file_fingerprint(path)
            except OSError as exc:
                raise PotentialFileChanged(
                    f"Potential file changed while it was being read: {path}"
                ) from exc
            if current != fingerprint:
                raise PotentialFileChanged(
                    f"Potential file changed while it was being read: {path}"
                )
        return potential

    @classmethod
    def from_string(cls, content, atoms=None, allow_dangerous=False, *, host=None):
        return cls.from_file(
            StringIO(content), atoms, allow_dangerous, host=host
        )

    @staticmethod
    def is_cluster_file(filename):
        """Legacy format probe (not used for dispatch in from_file)."""
        def detect(lines):
            in_globals = False
            for line in lines:
                name = line.strip().upper()
                if name == "GLOBAL SYSTEM PARAMETER":
                    in_globals = True
                elif in_globals and re.match(r"^NQCLU\s+\d+\b", name):
                    return True
                elif in_globals and re.match(r"^NQ\s+\d+\b", name):
                    return False
                elif in_globals and name.startswith("*"):
                    continue
                elif in_globals and name in ("SCF-INFO", "LATTICE"):
                    return False
            return False

        if hasattr(filename, "read"):
            position = filename.tell() if hasattr(filename, "tell") else None
            try:
                return detect(filename)
            finally:
                if position is not None:
                    filename.seek(position)
        try:
            with open(filename, encoding="utf-8") as source:
                return detect(source)
        except FileNotFoundError:
            return False

    @classmethod
    def from_atoms(cls, atoms):
        """Select a native potential for the periodicity of an ASE structure."""
        from .potentials import PeriodicPotential
        from .cluster_potentials import EmbeddedPotential

        SPRKKRAtoms.promote_ase_atoms(atoms)
        if not any(atoms.pbc):
            host = atoms.embedding_host
            if host is None:
                raise ValueError(
                    "A 0D SPR-KKR structure requires sprkkr_embedding_host "
                    "to create a cluster potential"
                )
            if atoms.has_potential():
                cluster = atoms.potential
                if not isinstance(cluster, EmbeddedPotential) or cluster.host is not host:
                    raise ValueError(
                        "A 0D structure's attached potential must be an "
                        "EmbeddedPotential referencing its current host"
                    )
                return cluster
            cluster = EmbeddedPotential.from_atoms(atoms)
            atoms._potential = cluster
            return cluster
        if atoms.has_potential():
            return atoms.potential
        return PeriodicPotential(atoms=atoms)
