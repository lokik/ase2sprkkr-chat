"""Definitions of SPR-KKR input-parameter file formats."""

import os
import pkgutil
import re
from io import StringIO

import pyparsing as pp
from ..common.grammar_types import Keyword
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


class _TaskSelectorMap(dict):
    """Map TASK selectors to cached IPDs, without materializing non-candidates."""

    def __init__(self, values, registry=None):
        super().__init__(values)
        self._registry = registry

    def __getitem__(self, selector):
        values = super().__getitem__(selector)
        if self._registry is None:
            return values
        return tuple(
            self._registry[value] if isinstance(value, str) else value
            for value in values
        )

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
    """Discover TASK selectors by importing classes, not building their IPDs."""
    registry = InputParameters.input_parameters_registry
    selectors = {}
    for name in registry.names(include_aliases=False):
        try:
            definition_class = registry.definition_class(name)
            if definition_class is None:
                # Backwards compatibility for old user modules that export an
                # already materialized definition or a legacy factory.
                definition = next(registry.iter_valid_values(
                    names=(name,), condition=lambda item: not item.is_derived_definition(),
                ), None)
                if definition is None:
                    continue
                tokens = definition.accepted_task_selectors()
                candidate = definition
            else:
                tokens = _normalized_task_selectors(
                    definition_class.task_selectors, name,
                )
                candidate = name
            for token in tokens:
                members = selectors.setdefault(token, [])
                if candidate not in members:
                    members.append(candidate)
        except Exception:
            # Match existing registry enumeration policy: a broken unrelated
            # user definition must not block reading a valid native input.
            # The registry emits a warning once for the offending module.
            list(registry.iter_valid_values(names=(name,)))
    return _TaskSelectorMap(
        {key: tuple(value) for key, value in selectors.items()},
        registry=registry,
    )


@cache
def _task_preparse_definition():
    """Build the complete TASK-selector preview, including custom aliases."""
    return _build_task_preparse_definition(tuple(sorted(_task_selector_map())))


@cache
def _fast_task_preparse_definition():
    """Build the preview from names only: no definition module is imported."""
    registry = InputParameters.input_parameters_registry
    return _build_task_preparse_definition(
        tuple(sorted(registry.names(include_aliases=False)))
    )


def _build_task_preparse_definition(values):
    """Build a tiny grammar that finds known selectors in an unordered TASK."""
    from .definitions._identifiers import _FilenameString

    if not values:
        raise ValueError("No registered input-parameter TASK selectors")
    preview_value = mixed.copy()
    preview_value.types = (*mixed.types, _FilenameString())

    class _PreparseSection(InputSectionDefinition):
        custom_class = staticmethod(
            CustomConfigurationValue.factory(InputValueDefinition, preview_value)
        )

    class _PreparseIPD(_CachedPreparseIPD):
        child_class = _PreparseSection
        custom_class = staticmethod(CustomConfigurationSection.factory(_PreparseSection))

    ipd = _PreparseIPD(
        "_TASK_PREPARSE",
        [InputSectionDefinition(
            "TASK",
            [InputValueDefinition(
                "TASK",
                Keyword(*values),
                name_in_grammar=False,
                is_optional=True,
                is_required=False,
                is_repeated=True,
                repeated_delimiter=pp.Empty(),
            )],
            is_optional=True,
        )],
    )
    ipd.grammar()
    return ipd


def _has_external_input_definitions():
    """A user module might claim an existing TASK selector; disable fast lookup."""
    paths = tuple(InputParameters.input_parameters_registry.package.__path__)
    return any(
        info.name != "sections" and not info.name.startswith("_")
        for info in pkgutil.iter_modules(paths[1:])
    )


def _direct_definition_for_input(text, *, allow_incomplete=False):
    """Try a same-named IPD before importing any unrelated definition module.

    Full discovery remains authoritative for aliases, multiple selectors and
    installations with user modules that might claim built-in TASK values.
    """
    if _has_external_input_definitions():
        return None

    preview = _fast_task_preparse_definition().read_from_string(
        text, allow_dangerous=False
    )
    selected = preview.TASK.TASK()
    names = (tuple(dict.fromkeys(str(token).upper() for token in selected))
             if selected is not None else ())
    if len(names) == 1:
        name = names[0]
    elif not names and not allow_incomplete and not re.search(r"(?m)^[ \t]*TASK\b", text):
        name = "SCF"
    else:
        return None

    registry = InputParameters.input_parameters_registry
    if name not in registry.names(include_aliases=False):
        return None
    definition = registry[name]
    if definition.is_derived_definition() or name not in definition.accepted_task_selectors():
        return None
    return definition


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
    """Canonical 1:1 task declared together with this schema.

    This is declaration-time metadata for the simple case where one input
    schema represents exactly one task. Schemas shared by multiple tasks leave
    it as ``None`` and override :meth:`task_definition_name` using their values.

    The association is internal and must not be changed after registration.
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

        direct = _direct_definition_for_input(
            text, allow_incomplete=allow_incomplete,
        )
        if direct is not None:
            return direct.read_from_file(
                _snapshot_stream(text, filename),
                allow_dangerous=allow_dangerous,
                allow_incomplete=allow_incomplete,
                filename=filename,
                **({"report_invalid": report_invalid} if report_invalid is not None else {}),
            )

        preview = _task_preparse_definition().read_from_string(
            text, allow_dangerous=False
        )
        selected = preview.TASK.TASK()
        selectors = tuple(selected) if selected is not None else ()
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



class _CachedPreparseIPD(InputParametersDefinition):
    """Preparser only: no mutations after grammar compilation."""

    @cache
    def grammar(self, allow_dangerous=False, *, allow_incomplete=False):
        return super().grammar(allow_dangerous, allow_incomplete=allow_incomplete)
