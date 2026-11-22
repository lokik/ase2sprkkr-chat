"""Definitions of SPR-KKR input-parameter file formats."""

import os
import pkgutil
import re
from io import StringIO

import pyparsing as pp
from ..common.warnings import DataValidityError
from ..common.native_preset_io import reading_preset
from ..common.container_definitions import dict_from_parsed
from ..sprkkr.configuration import (
    ConfigurationValueDefinition,
    ConfigurationSectionDefinition,
    ConfigurationFileDefinition,
    CustomConfigurationValue,
    CustomConfigurationSection,
)
from ..common.grammar_types import mixed, flag
from ..common.grammar import generate_grammar, delimitedList
from ..common.decorators import cached_class_property, cache, maybeclassmethod
from ..common.file_utils import filename_from_file
from .input_parameters import InputParameters, InputSection, InputValue

with generate_grammar():
    section_line_ends = pp.Regex("(\n[ \t]*)*[ \t]")


MAX_INPUT_FILE_SIZE = 1024 * 1024


def _read_input_text(file):
    """Return a UTF-8 snapshot no larger than MAX_INPUT_FILE_SIZE bytes."""
    if isinstance(file, (str, bytes, os.PathLike)):
        with open(file, "rb") as stream:
            content = stream.read(MAX_INPUT_FILE_SIZE + 1)
    elif hasattr(file, "read"):
        content = file.read(MAX_INPUT_FILE_SIZE + 1)
    else:
        raise TypeError("Input file must be a path or a readable stream")

    if isinstance(content, str):
        data = content.encode("utf-8")
        if len(data) > MAX_INPUT_FILE_SIZE:
            raise ValueError(
                f"SPR-KKR input exceeds {MAX_INPUT_FILE_SIZE} bytes"
            )
        return content
    if not isinstance(content, bytes):
        raise TypeError("Input stream must return str or bytes")
    if len(content) > MAX_INPUT_FILE_SIZE:
        raise ValueError(
            f"SPR-KKR input exceeds {MAX_INPUT_FILE_SIZE} bytes"
        )
    return content.decode("utf-8")


def _snapshot_stream(text, filename):
    """Reparse the same bytes without losing the origin of a named input."""
    stream = StringIO(text)
    if filename is not None:
        stream.name = filename
    return stream


def _normalized_task_selectors(declared, name):
    """Validate metadata on a class or a materialized IPD."""
    names = (name,) if declared is None else (declared,) if isinstance(declared, str) else tuple(declared)
    if not names:
        raise ValueError(f"IPD {name}: task_selectors may not be empty")
    normalized = []
    for token in names:
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"IPD {name}: TASK selectors must be nonempty strings")
        normalized.append(token.strip().upper())
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"IPD {name}: duplicate TASK selectors")
    return tuple(normalized)


# A native TASK value normally matches the basename of its built-in IPD
# module. DMI is a second native selector belonging to the shared JXC IPD.
# BSFEK and BSFKK are task identities, not native TASK selectors: both use BSF.
_BUILTIN_TASK_SELECTOR_ALIASES = {"DMI": "JXC"}


def _builtin_ipd_names():
    """Discover built-in IPD names without importing their modules."""
    package = InputParameters.input_parameters_registry.package
    return tuple(sorted(
        entry.name.upper()
        for entry in pkgutil.iter_modules(tuple(package.__path__)[:1])
        if not entry.name.startswith("_") and entry.name != "sections"
    ))


@cache
def _user_task_selectors():
    """Discover only user IPD modules, importing classes but not building IPDs.

    Legacy user factories without a definition class are materialized once
    through the existing InputParametersRegistry cache. An unrelated invalid
    user module is reported and skipped, just as with registry enumeration.
    """
    import warnings

    from ..common.module_registry import RegistryLoadWarning

    registry = InputParameters.input_parameters_registry
    package_paths = tuple(registry.package.__path__)
    builtin_names = set(_builtin_ipd_names())
    available = set(registry.names(include_aliases=False))
    selectors = {}
    for entry in pkgutil.iter_modules(package_paths[1:]):
        name = entry.name.upper()
        if (entry.name.startswith("_") or entry.name == "sections"
                or name in builtin_names or name not in available):
            continue
        try:
            cls = registry.definition_class(name)
            if cls is None:
                # Backward-compatible user factory. Do not offer presets as
                # alternative definitions, even if their selectors collide.
                definition = registry[name]
                if definition.is_derived_definition():
                    continue
                tokens = definition.accepted_task_selectors()
            else:
                tokens = _normalized_task_selectors(cls.task_selectors, name)
            for selector in tokens:
                selectors.setdefault(selector, []).append(name)
        except Exception as exc:
            warnings.warn(
                f"Skipping invalid user IPD {name}: "
                f"{type(exc).__name__}: {exc}",
                RegistryLoadWarning,
                stacklevel=2,
            )
    return {selector: tuple(names) for selector, names in selectors.items()}


class _TaskSelectorMap(dict):
    """Select user IPDs first, falling back to a built-in when none qualifies.

    Materialize only selected candidates. A user preset copied from another
    schema is not an override; two canonical user IPDs for one native TASK
    are an error, regardless of whether their parsers could both succeed.
    """

    def __init__(self, values, registry=None, *, user=None):
        super().__init__(values)
        self._registry = registry
        self._user = user or {}

    def __getitem__(self, selector):
        builtins = super().__getitem__(selector)
        if self._registry is None:
            return builtins

        users = []
        for name in self._user.get(selector, ()):
            definition = self._registry[name]
            if not definition.is_derived_definition():
                users.append(definition)
        if len(users) > 1:
            names = ", ".join(definition.name for definition in users)
            raise ValueError(
                f"Multiple user-defined IPDs claim TASK {selector}: {names}"
            )
        if users:
            # A user's schema overrides the built-in. Never retry the built-in
            # if this chosen definition subsequently fails to parse.
            return (users[0],)
        return tuple(self._registry[name] for name in builtins)

    def definitions(self, selector, *, user_defined=None):
        candidates = self[selector]
        if user_defined is None:
            return candidates
        if type(user_defined) is not bool:
            raise TypeError("user_defined must be True, False, or None")
        return tuple(
            definition for definition in candidates
            if definition.is_user_defined is user_defined
        )


@cache
def _task_selector_map():
    """Map native TASK selectors without importing any unrelated built-in IPD."""
    registry = InputParameters.input_parameters_registry
    builtin = {name: (name,) for name in _builtin_ipd_names()}
    for selector, schema in _BUILTIN_TASK_SELECTOR_ALIASES.items():
        builtin[selector] = (schema,)
    user = _user_task_selectors()
    # Include custom-only native selectors without constructing their IPDs.
    for selector in user:
        builtin.setdefault(selector, ())
    return _TaskSelectorMap(builtin, registry, user=user)


_TASK_SECTION_START = re.compile(r"^TASK(?=[ \t=]|$)", re.IGNORECASE)
_TASK_ASSIGNMENT = re.compile(
    r"(?<!\S)[A-Z_][A-Z0-9_-]*\s*=\s*"
    r"(?:\{(?:[^{}]|\{[^{}]*\})*\}"
    r"|\[(?:[^\[\]]|\[[^\[\]]*\])*\]"
    r"|\([^()]*\)|[^\s,;]+)",
    re.IGNORECASE,
)
_TASK_FLAG = re.compile(r"(?<!\S)[A-Z_][A-Z0-9_]*(?!\S)", re.IGNORECASE)


def _scan_task_selectors(text, known_selectors):
    """Preview native TASK switches without constructing a parsing grammar.

    SPR-KKR's INPUT_FIND_SECTION reads 80 columns, starts sections in column
    one, and appends indented continuation lines. Comments start at '#'.
    SECTION_FIND_KEYWORD searches for whitespace-delimited flags; here,
    assignment values are additionally excluded to avoid false task matches.
    The selected IPD still performs the full, authoritative parse.
    """
    parts = []
    in_task = False
    for raw_line in text.splitlines():
        line = raw_line.replace("\t", " ")[:80]
        if not in_task:
            match = _TASK_SECTION_START.match(line)
            if match is None:
                continue
            in_task = True
            part = line[match.end():]
            # INPUT_FIND_SECTION accepts TASK=JXC (the '=' becomes a space).
            if part.lstrip().startswith("="):
                part = part.replace("=", " ", 1)
        else:
            if line and not line[0].isspace() and line[0] != "#":
                break
            part = line
        parts.append(part.split("#", 1)[0])

    if not in_task:
        return ()

    section = " ".join(parts)
    section = _TASK_ASSIGNMENT.sub(" ", section)
    return tuple(
        match.group().upper()
        for match in _TASK_FLAG.finditer(section)
        if match.group().upper() in known_selectors
    )


def _read_selected_definitions(text, selectors, *, filename=None,
                               allow_dangerous=False, allow_incomplete=False,
                               report_invalid=None):
    """Try only IPDs nominated by detected native TASK keywords."""
    mapping = _task_selector_map()
    candidates = []
    for selector in selectors:
        selector = str(selector).upper()
        if selector not in mapping:
            raise ValueError(f"Unknown TASK selector: {selector}")
        for ipd in mapping[selector]:
            if all(ipd is not candidate for candidate in candidates):
                candidates.append(ipd)

    def parse(definition):
        return definition.read_from_file(
            _snapshot_stream(text, filename),
            allow_dangerous=allow_dangerous,
            filename=filename,
            **({"allow_incomplete": True} if allow_incomplete else {}),
            **({"report_invalid": report_invalid} if report_invalid is not None else {}),
        )

    if len(candidates) == 1:
        return parse(candidates[0])

    successful = []
    failures = []
    for definition in candidates:
        try:
            result = parse(definition)
        except (pp.ParseBaseException, DataValidityError) as exc:
            failures.append((definition, exc))
        else:
            successful.append((definition, result))
    if len(successful) == 1:
        return successful[0][1]
    if len(successful) > 1:
        names = ", ".join(definition.name for definition, _ in successful)
        raise ValueError(f"Ambiguous TASK selectors, multiple IPDs accept input: {names}")
    if failures:
        details = "\n".join(
            f"  {definition.name}: {type(error).__name__}: {error}"
            for definition, error in failures
        )
        raise ValueError(
            f"No selected IPD can parse the input:\n{details}"
        ) from failures[-1][1]
    raise ValueError("No input-parameter definition selected")


class InputValueDefinition(ConfigurationValueDefinition):
    result_class = InputValue

    def _resolve_is_optional(self, is_optional):
        return True if is_optional is None else is_optional

    @cached_class_property
    def delimiter():
        with generate_grammar():
            return pp.Suppress("=").set_name("=")

    prefix = "\t"
    type_from_type_map = {bool: flag}
    type_of_dangerous = mixed


class InputSectionDefinition(ConfigurationSectionDefinition):
    child_class = InputValueDefinition
    result_class = InputSection

    def _resolve_is_optional(self, is_optional):
        return True if is_optional is None else is_optional
    custom_class = staticmethod(
        CustomConfigurationValue.factory(InputValueDefinition, mixed)
    )

    @cached_class_property
    def delimiter():
        with generate_grammar():
            out = (pp.Optional(section_line_ends) + pp.WordStart()).suppress()
            return out.set_name("\n")

    do_not_skip_whitespaces_before_name = True


class InputParametersDefinition(ConfigurationFileDefinition):
    """Syntax and parameter semantics of one SPR-KKR input-file schema."""

    _task_definition = None
    """Canonical 1:1 task declared as an IPD *class* attribute.

    Discovery and task listing can inspect this without constructing the IPD.
    Schemas shared by multiple tasks leave it as ``None`` and override
    :meth:`task_definition_name` using their values. The association is
    internal and must not be changed after registration.
    """

    is_user_defined = True
    """Whether the definition comes from user code.

    Standalone definitions default to True; registration overrides this
    according to the defining module's origin.
    """

    _derived_from_name = None
    """ If one copies the definition, it should be listed under its new name, but
    retain the original name as well, to retain the information, to which "task"
    it originally belongs """


    child_class = InputSectionDefinition
    result_class = InputParameters
    custom_class = staticmethod(
        CustomConfigurationSection.factory(InputSectionDefinition)
    )
    configuration_type_name = "INPUT PARAMETERS"

    task_selectors = None
    """Native TASK selectors, defaulting to the IPD name.

    A string denotes one selector; an iterable denotes multiple selectors.
    Like other registration metadata, this must not change once the singleton
    registry has constructed the cached selector map and preparse grammar.
    """

    def accepted_task_selectors(self):
        return _normalized_task_selectors(self.task_selectors, self.name)

    @maybeclassmethod
    def from_file(self, cls, file, allow_dangerous=False, *,
                  allow_incomplete=False, report_invalid=None):
        """Read one input; on the class autodetect an IPD, on an instance use it.

        Read a bounded snapshot of the input once, independently of path/stream.
        A missing TASK selector selects SCF, not an ADSI-derived definition.
        """
        filename = filename_from_file(file, None)
        text = _read_input_text(file)
        if self is not None:
            return self.read_from_file(
                _snapshot_stream(text, filename),
                allow_dangerous=allow_dangerous,
                allow_incomplete=allow_incomplete,
                filename = filename,
                report_invalid=report_invalid,
            )

        selectors = _scan_task_selectors(text, _task_selector_map())
        if not selectors:
            if allow_incomplete:
                raise ValueError("Cannot autodetect incomplete input without TASK selector; supply an explicit definition or task")
            selectors = ("SCF",)
        return _read_selected_definitions(
            text, selectors, filename=filename,
            allow_dangerous=allow_dangerous,
            allow_incomplete=allow_incomplete,
            report_invalid=report_invalid,
        )

    def parse_return(self, val, return_value_only=True):
        values = super().parse_return(val, return_value_only)
        # Task defaults are useful for newly created input, but an omitted
        # ADSI in a native input means an empty suffix, not the task name.
        if (not reading_preset() and "CONTROL" in values and "CONTROL" in self
                and "ADSI" in self["CONTROL"]):
            values["CONTROL"].setdefault("ADSI", "")
        return values

    def read_from_file(self, file, allow_dangerous=False, *, filename=None,
                       allow_incomplete=False, report_invalid=None, **kwargs):
        out = super().read_from_file(file, allow_dangerous,
                               allow_incomplete=allow_incomplete,
                               report_invalid=report_invalid, **kwargs)
        if filename:
            out.directory = os.path.dirname(filename)
        return out

    def task_definition_name(self, parameters):
        """Return the ASE2SPRKKR task name represented by *parameters*.

        The standard one-schema/one-task case is answered directly by the
        schema declaration. Shared schemas override this method and derive the
        identity from concrete parameter values. Resolving the returned name to
        a :class:`TaskDefinition` remains the responsibility of ``TaskRegistry``.
        """
        task = self._task_definition
        return None if task is None else task.name

    def initialize_from_result(self, parameters, result):
        """Initialize instance-specific defaults derived from a previous result."""

    def input_files(self, parameters, directory):
        """Return schema-specific auxiliary input files."""
        return ()

    def output_files(self, parameters, directory):
        """Return schema-specific output files."""
        return ()

    def result_files(self, parameters, directory):
        """Return scientifically meaningful outputs for user-facing results.

        Also included by output_files() through the task registry; these
        declarations do not change execution/collection semantics.
        """
        return ()

    def cache_files(self, parameters, directory):
        """Return schema-specific optional cache files."""
        return ()

    @classmethod
    @cache
    def custom_value_grammar(cls):
        value = cls.child_class.custom_member_grammar()
        delim = cls.child_class.delimiter
        return delimitedList(value, delim).set_parse_action(
            lambda x: dict_from_parsed(x.asList())
        )

    @cached_class_property
    def delimiter():
        with generate_grammar():
            def ws(x):
                return x.set_whitespace_chars("")

            after_consumed_line_end = pp.Empty().add_condition(
                lambda s, loc, _t: loc > 0 and s[:loc].rstrip(" \t\r").endswith("\n")
            )
            out = (
                pp.Optional(section_line_ends)
                + pp.OneOrMore(ws(pp.LineEnd()))
                + pp.FollowedBy(ws(pp.Regex(r"[^\s]")))
                | after_consumed_line_end
            ).suppress()
            return out.set_name("\n")

    def _generic_info(self):
        return f"Input parameters format {self.name}"

    def copy(self, *args, new_schema=False, **kwargs):
        copied = super().copy(**kwargs)
        if not new_schema:
            copied._derived_from_name = self.derived_from_name
        copied.is_user_defined = self.is_user_defined
        if '_task_definition' in self.__dict__:
            copied._task_definition = self._task_definition
        return copied

    @property
    def derived_from_name(self):
        return self._derived_from_name or self.name

    def is_derived_definition(self):
        return self._derived_from_name and self._derived_from_name != self.name

    @property
    def display_name(self):
        name = self.name.upper()
        if self._derived_from_name:
            derived_from = self._derived_from_name.upper()
        else:
            derived_from = name

        return (
            derived_from if derived_from == name
            else f"{name}({derived_from})"
        )

    def info(self, generic=True):
        description = super().info(generic)
        return (
            f"{self.display_name} - {description}"
            if description else self.display_name
        )
