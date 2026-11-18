"""Declarative metadata for SPR-KKR executables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..common.module_registry import RegistryDefinitionError, RegistryEntryNotFound


__all__ = ("ExecutableDefinition", "executables", "resolve_task_executable")


def _nonblank_string(value, field):
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _normalize_reader_name(value):
    return _nonblank_string(value, "default_output_reader").strip().upper()


@dataclass(frozen=True)
class ExecutableDefinition:
    """Metadata shared by tasks executed by the same SPR-KKR binary."""

    name: str
    default_output_reader: str = "DEFAULT"
    description: str = ""
    cluster_variant: str | None = None

    def __post_init__(self):
        name = _nonblank_string(self.name, "name")
        reader = _normalize_reader_name(self.default_output_reader)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "default_output_reader", reader)
        if self.cluster_variant is not None:
            _nonblank_string(self.cluster_variant, "cluster_variant")

    def resolve(self, input_parameters):
        """Select the native executable from the prepared CONTROL parameters.

        An absent CLUTYPE retains normal execution, including for older or
        user-defined input schemas. Unsupported cluster executables fail
        explicitly rather than running the periodic binary by mistake.
        """
        try:
            cluster_type = input_parameters.CONTROL.CLUTYPE()
        except (AttributeError, KeyError):
            cluster_type = None
        if cluster_type is None or cluster_type == "":
            return self.name
        if cluster_type != "embedded":
            raise ValueError(f"Unsupported SPR-KKR CLUTYPE {cluster_type!r}")
        if self.cluster_variant is None:
            raise ValueError(
                f"Executable {self.name!r} does not support embedded clusters"
            )
        return self.cluster_variant


class _ExecutableRegistry(Mapping):
    """Read-only mapping of known SPR-KKR executable metadata."""

    def __init__(self, definitions=()):
        items = {}
        for definition in definitions:
            if not isinstance(definition, ExecutableDefinition):
                raise TypeError("definition must be an ExecutableDefinition")
            if definition.name in items:
                raise RegistryDefinitionError(
                    f"Duplicate executable definition {definition.name!r}"
                )
            items[definition.name] = definition
        self._definitions = items

    def __getitem__(self, name):
        try:
            return self._definitions[name]
        except KeyError:
            raise RegistryEntryNotFound(name) from None

    def __iter__(self):
        return iter(self._definitions)

    def __len__(self):
        return len(self._definitions)

    def resolve(self, name, input_parameters):
        """Resolve a known binary or retain an ordinary custom executable.

        Users may define TaskDefinitions with executable names absent from the
        built-in metadata registry. Such binaries retain their native names in
        normal mode; cluster mode remains opt-in and fails without metadata.
        """
        definition = self._definitions.get(name)
        if definition is None:
            definition = ExecutableDefinition(name)
        return definition.resolve(input_parameters)


# CLUSCF/CLUGEN are Fortran program identifiers; native makefile targets install
# the corresponding binaries as embscf/embgen (with MPI variants suffixed MPI).
executables = _ExecutableRegistry(
    (
        ExecutableDefinition(
            "kkrscf",
            cluster_variant="embscf",
            description="Potential generation and self-consistent calculations.",
        ),
        ExecutableDefinition(
            "kkrgen",
            cluster_variant="embgen",
            description="Electronic-structure and general property calculations.",
        ),
        ExecutableDefinition(
            "kkrspec",
            default_output_reader="SPEC",
            description="Spectroscopy and photoemission calculations.",
        ),
        ExecutableDefinition(
            "kkrchi",
            description="Response-function calculations.",
        ),
    )
)


def resolve_task_executable(task, input_parameters):
    """Resolve the executable for an enabled *task*, not only its base binary.

    The existence of an embedded variant of KKRGEN is not evidence that every
    KKRGEN task can be run in an embedded cluster. Keep this opt-in decision in
    TaskDefinition, and the native program-name mapping in ExecutableDefinition.
    """
    try:
        cluster_type = input_parameters.CONTROL.CLUTYPE()
    except (AttributeError, KeyError):
        cluster_type = None
    if cluster_type == "embedded" and not task.supports_embedded_cluster:
        raise ValueError(
            f"Task {task.name!r} is not enabled for embedded-cluster calculations; "
            "a CLUGEN/CLUSCF executable alone does not establish task support"
        )
    return executables.resolve(task.executable, input_parameters)
