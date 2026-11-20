"""Sparse YAML presets for SPR-KKR input parameters, scoped by IPD or task."""
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from unyt.array import unyt_quantity

from ..configuration import user_preferences_directory
from ..common.warnings import ValidationResult


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SCOPES = ("input_parameters", "task")
_SUFFIX = ".yaml"
_VERSION = 1

# File identity and embedding mode belong to the source structure. By default
# loading a preset never replaces them, even in clear_first mode.
STRUCTURAL_CONTROL_OPTIONS = frozenset({
    "DATASET", "ADSI", "POTFIL", "POTFIL_CLU", "SFNFIL", "CLUTYPE",
})


def _yaml_value(value):
    """Represent typed input data using YAML scalars, sequences and mappings."""
    if isinstance(value, unyt_quantity):
        return [value.value.item(), str(value.units)]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            key.item() if isinstance(key, np.generic) else key: _yaml_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return value


def _omit_null_assignments(value):
    """Drop null mapping entries without shifting positional values."""
    if isinstance(value, Mapping):
        filtered = {}
        for key, item in value.items():
            if item is None:
                continue
            remaining = _omit_null_assignments(item)
            # A mapping that only contained null assignments is itself absent.
            if isinstance(item, Mapping) and item and not remaining:
                continue
            filtered[key] = remaining
        return filtered
    if isinstance(value, list):
        return [_omit_null_assignments(item) for item in value]
    return value


class PresetStore:
    """Store YAML input overlays under input_parameters/ or task/."""

    def __init__(self, root=None):
        self.root = (
            Path(root) if root is not None
            else Path(user_preferences_directory()) / "presets"
        )

    @staticmethod
    def _part(value):
        if not isinstance(value, str) or _NAME.fullmatch(value) is None:
            raise ValueError(f"Invalid preset identifier {value!r}")
        return value.lower()

    @classmethod
    def _identifier(cls, value):
        """Return (scope, owner, name), with missing parts represented by None."""
        if not isinstance(value, str) or not value:
            raise ValueError("Preset name must be a nonempty string")
        scope = None
        if ":" in value:
            prefix, value = value.split(":", 1)
            if prefix not in _SCOPES:
                raise ValueError(
                    f"Unknown preset scope {prefix!r}; use input_parameters: or task:"
                )
            scope = prefix
        parts = value.split(".")
        if len(parts) == 1:
            if scope is not None:
                raise ValueError("Scoped presets must be named SCOPE:OWNER.NAME")
            return None, None, cls._part(parts[0])
        if len(parts) == 2:
            return scope, cls._part(parts[0]), cls._part(parts[1])
        raise ValueError(
            f"Invalid preset name {value!r}; use NAME, OWNER.NAME, "
            "or SCOPE:OWNER.NAME"
        )

    @classmethod
    def _valid_owner_scopes(cls, owner):
        """Look up the two independent registry namespaces."""
        from .input_parameters import InputParameters

        upper = cls._part(owner).upper()
        found = []
        if upper in InputParameters.input_parameters_registry:
            found.append("input_parameters")
        if upper in InputParameters.task_registry:
            found.append("task")
        return tuple(found)

    @classmethod
    def definition_name(cls, selector):
        """Return the exact registered IPD, not its derived_from_name."""
        from .input_parameters import InputParameters

        scope, owner = cls._context(selector)
        if scope == "input_parameters":
            return owner
        task = InputParameters.task_registry[owner.upper()]
        return InputParameters.input_parameters_registry.definition_for_task(task).name.lower()

    @classmethod
    def _tasks_for_definition(cls, definition):
        from .input_parameters import InputParameters

        definition = cls._part(definition)
        registry = InputParameters.input_parameters_registry
        return tuple(
            name.lower() for name in InputParameters.task_registry.names()
            if registry.definition_for_task(
                InputParameters.task_registry[name]
            ).name.lower() == definition
        )

    @classmethod
    def _context(cls, selector):
        """Resolve a context selector to (scope, owner).

        For single-task IPDs sharing a name with their task, the short form
        selects the input-parameter schema, so existing commands like
        'presets list dos' remain meaningful. For JXC/DMI, require a qualifier.
        """
        if not isinstance(selector, str) or not selector:
            raise ValueError("Context must name an input_parameters or task owner")
        if ":" in selector:
            scope, owner = selector.split(":", 1)
            if scope not in _SCOPES:
                raise ValueError(f"Unknown preset scope {scope!r}")
            owner = cls._part(owner)
            if scope not in cls._valid_owner_scopes(owner):
                raise ValueError(f"Unknown {scope} {owner!r}")
            return scope, owner
        owner = cls._part(selector)
        scopes = cls._valid_owner_scopes(owner)
        if not scopes:
            raise ValueError(f"Unknown input_parameters or task {owner!r}")
        if len(scopes) == 1:
            return scopes[0], owner
        if len(cls._tasks_for_definition(owner)) > 1:
            raise ValueError(
                f"{owner.upper()} names both a shared IPD and a task; "
                f"use input_parameters:{owner} or task:{owner}"
            )
        return "input_parameters", owner

    def _path(self, scope, owner, name):
        if scope not in _SCOPES:
            raise ValueError(f"Invalid preset scope {scope!r}")
        owner = self._part(owner)
        name = self._part(name)
        path = self.root / scope / owner / f"{name}{_SUFFIX}"
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Preset path escapes its storage root")
        return path

    @classmethod
    def _qualified(cls, scope, owner, name):
        return f"{scope}:{owner}.{name}"

    def _entries(self):
        if not self.root.is_dir():
            return ()
        found = []
        for scope in _SCOPES:
            root = self.root / scope
            if not root.is_dir() or root.is_symlink():
                continue
            for folder in root.iterdir():
                if (
                    not folder.is_dir() or folder.is_symlink()
                    or _NAME.fullmatch(folder.name) is None
                    or folder.name != folder.name.lower()
                ):
                    continue
                for path in folder.glob(f"*{_SUFFIX}"):
                    if path.is_file() and not path.is_symlink() and _NAME.fullmatch(path.stem):
                        found.append((scope, folder.name, path.stem))
        return tuple(found)

    def list(self, selector=None):
        """List fully qualified preset names for a task, IPD, or globally.

        An input_parameters context exposes all task subscopes of its IPD.
        A task context exposes only its own task and neutral IPD presets.
        """
        entries = self._entries()
        if selector is None:
            return tuple(sorted(self._qualified(*entry) for entry in entries))

        scope, owner = self._context(selector)
        if scope == "input_parameters":
            applicable = set(self._tasks_for_definition(owner))
            entries = (
                entry for entry in entries
                if (entry[0] == "input_parameters" and entry[1] == owner)
                or (entry[0] == "task" and entry[1] in applicable)
            )
        else:
            definition = self.definition_name(f"task:{owner}")
            entries = (
                entry for entry in entries
                if (entry[0] == "task" and entry[1] == owner)
                or (entry[0] == "input_parameters" and entry[1] == definition)
            )
        return tuple(sorted(self._qualified(*entry) for entry in entries))

    def _describe_path(self, path):
        path = Path(path)
        if path.suffix != _SUFFIX:
            raise ValueError(f"Not a YAML preset: {path}")
        return path.parent.parent.name, path.parent.name, path.stem

    def resolve(self, name, *, definition=None, task=None):
        """Resolve identifiers without silently picking an ambiguous scope."""
        scope, owner, bare = self._identifier(name)
        expected = self._part(definition) if definition is not None else None
        if expected is not None and "input_parameters" not in self._valid_owner_scopes(expected):
            raise ValueError(f"Unknown input-parameter definition {expected}")
        if task is not None:
            task = self._part(task)
            if "task" not in self._valid_owner_scopes(task):
                raise ValueError(f"Unknown task {task!r}")
            task_definition = self.definition_name(f"task:{task}")
            if expected is not None and task_definition != expected:
                raise ValueError(f"Task {task} is not compatible with {expected}")
            expected = task_definition

        if owner is not None:
            if scope is None:
                valid_scopes = self._valid_owner_scopes(owner)
                if not valid_scopes:
                    raise ValueError(f"Unknown preset owner {owner!r}")
            else:
                valid_scopes = (scope,)
                if scope not in self._valid_owner_scopes(owner):
                    raise ValueError(f"Unknown {scope} {owner!r}")
            candidates = [(kind, owner, bare) for kind in valid_scopes]
        elif expected is not None:
            candidates = [("input_parameters", expected, bare)]
            candidates.extend(
                ("task", item, bare)
                for item in (self._tasks_for_definition(expected) if task is None else (task,))
            )
        else:
            candidates = [
                (kind, entry_owner, bare)
                for kind, entry_owner, entry_name in self._entries()
                if entry_name == bare
            ]

        if expected is not None:
            candidates = [
                entry for entry in candidates
                if (entry[0] == "input_parameters" and entry[1] == expected)
                or (
                    entry[0] == "task"
                    and self.definition_name(f"task:{entry[1]}") == expected
                )
            ]
        paths = [
            self._path(*entry) for entry in set(candidates)
            if self._path(*entry).is_file() and not self._path(*entry).is_symlink()
        ]
        if not paths:
            raise FileNotFoundError(f"Preset {name!r} does not exist or is incompatible")
        if len(paths) > 1:
            choices = ", ".join(
                self._qualified(*self._describe_path(path))
                for path in sorted(paths)
            )
            raise ValueError(f"Preset {name!r} is ambiguous: {choices}")
        return paths[0]

    @staticmethod
    def _read_values(path):
        try:
            with open(path, encoding="utf-8") as stream:
                document = YAML(typ="safe", pure=True).load(stream)
        except YAMLError as exc:
            raise ValueError(f"Invalid preset YAML in {path}: {exc}") from exc
        if not isinstance(document, dict) or document.keys() != {"version", "parameters"}:
            raise ValueError(f"Preset {path} must contain version and parameters")
        version = document["version"]
        if type(version) is not int or version != _VERSION:
            raise ValueError(f"Unsupported preset version {version!r} in {path}")
        values = document["parameters"]
        if not isinstance(values, dict) or any(
            not isinstance(name, str) or "." in name
            or (section is not None and not isinstance(section, (dict, list)))
            for name, section in values.items()
        ):
            raise ValueError(f"Preset parameters in {path} must be grouped by section name")
        return values

    @staticmethod
    def _apply_values(
        parameters, values, *, clear_first,
        load_structural_info, expected_task,
    ):
        values = _omit_null_assignments(values)
        preserved = {}
        if not load_structural_info:
            for section, control in values.items():
                if section.upper() != "CONTROL":
                    continue
                if not isinstance(control, dict) or any(not isinstance(key, str) for key in control):
                    raise ValueError("Preset CONTROL must be a mapping of option names")
                values[section] = {
                    key: value for key, value in control.items()
                    if key.upper() not in STRUCTURAL_CONTROL_OPTIONS
                }
            if clear_first:
                explicit = parameters.as_dict(
                    only_changed="explicit", generated=False, copy=True,
                ) or {}
                preserved = {
                    key: value for key, value in explicit.get("CONTROL", {}).items()
                    if key in STRUCTURAL_CONTROL_OPTIONS
                }

        original_requested = parameters.requested_task_name
        # Requested task influences callable defaults in BSF and JXC/DMI.
        # Values remain authoritative and are validated after applying.
        if clear_first and expected_task is not None:
            parameters._requested_task_name = expected_task
        try:
            with parameters._mutation("set", report_invalid="raise") as (tx, _):
                if clear_first:
                    parameters.stage_clear(tx, check_required=False)
                parameters.stage(tx, values, unknown="add")
                if preserved:
                    parameters.stage(tx, {"CONTROL": preserved}, unknown="add")
                if expected_task is not None and parameters.task_name != expected_task:
                    raise ValueError(
                        f"Preset would select {parameters.task_name}, "
                        f"but {expected_task} is required"
                    )
        except Exception:
            parameters._requested_task_name = original_requested
            raise
        return parameters

    def _neutral_for_tasks(self, definition, values):
        """Check the overlay against fresh defaults for every task sharing an IPD."""
        from .input_parameters import InputParameters

        tasks = self._tasks_for_definition(definition)
        for task in tasks:
            candidate = InputParameters.for_task(task)
            try:
                self._apply_values(
                    candidate, values,
                    clear_first=True, load_structural_info=False,
                    expected_task=task.upper(),
                )
            except (ValueError, ValidationResult):
                return False
        return True

    def save(self, parameters, name, *, overwrite=False):
        """Save explicit settings, selecting task scope if they constrain the task."""
        definition = self._part(parameters.definition_name)
        scope, owner, bare = self._identifier(name)
        if owner is not None:
            if scope is None:
                scope, owner = self._context(owner)
            else:
                scope, owner = self._context(f"{scope}:{owner}")
        is_derived = parameters._definition.is_derived_definition()

        if scope == "task":
            if is_derived:
                raise ValueError(
                    "Task-scoped presets cannot discard a derived IPD; "
                    "save under input_parameters:<derived-name> instead"
                )
            target_task = owner.upper()
            if self.definition_name(f"task:{owner}") != definition:
                raise ValueError(f"Task {target_task} uses a different IPD")
            if parameters.task_name != target_task:
                raise ValueError(
                    f"Cannot save {parameters.task_name} values as {target_task}"
                )
        elif scope == "input_parameters":
            if owner != definition:
                raise ValueError(
                    f"Cannot save {definition.upper()} parameters under {owner.upper()}"
                )

        values = _yaml_value(parameters.as_dict(only_changed="explicit") or {})
        neutral = self._neutral_for_tasks(definition, values)
        if scope is None:
            scope = "input_parameters" if neutral or is_derived else "task"
            owner = definition if scope == "input_parameters" else self._part(parameters.task_name)
        elif scope == "input_parameters" and not neutral:
            raise ValueError(
                f"These settings select a specific task within {definition.upper()}; "
                "save under task:<task> instead"
            )
        path = self._path(scope, owner, bare)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Preset {self._qualified(scope, owner, bare)} exists")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{bare}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                yaml = YAML(typ="safe", pure=True)
                yaml.default_flow_style = False
                yaml.indent(mapping=2, sequence=4, offset=2)
                yaml.sort_base_mapping_type_on_output = False
                try:
                    yaml.dump({"version": _VERSION, "parameters": values}, stream)
                except YAMLError as exc:
                    raise ValueError(f"Cannot serialize preset {name!r}: {exc}") from exc
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, path)
            else:
                # A hard link publishes the complete file atomically and fails
                # if another writer has already claimed the destination.
                try:
                    os.link(temporary, path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Preset {self._qualified(scope, owner, bare)} exists"
                    ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def import_yaml(self, name, path, *, overwrite=False):
        """Validate a standalone YAML preset against its qualified destination."""
        from .input_parameters import InputParameters

        scope, owner, _ = self._identifier(name)
        if owner is None:
            raise ValueError("YAML import requires a qualified preset name such as dos.mesh")
        scope, owner = self._context(f"{scope}:{owner}" if scope else owner)
        parameters = (
            InputParameters.for_task(owner) if scope == "task"
            else InputParameters.for_definition(owner)
        )
        self._apply_values(
            parameters, self._read_values(path), clear_first=True,
            load_structural_info=True,
            expected_task=owner.upper() if scope == "task" else None,
        )
        return self.save(parameters, name, overwrite=overwrite)

    def load_into(
        self, parameters, name, *, clear_first=True,
        load_structural_info=False, required_task=None,
    ):
        """Atomically apply a preset while honoring its scope and caller constraints."""
        definition = self._part(parameters.definition_name)
        context_task = parameters.task_name if parameters.requested_task_name else None
        path = self.resolve(
            name, definition=definition, task=(
                context_task if "." not in name and ":" not in name else None
            ),
        )
        scope, owner, _ = self._describe_path(path)
        values = self._read_values(path)
        if scope == "input_parameters":
            # Edited or moved presets must still be neutral for the directory's IPD.
            if not self._neutral_for_tasks(owner, values):
                raise ValueError(
                    f"Preset {name!r} is not neutral across the tasks "
                    f"sharing input_parameters:{owner}"
                )

        current_task = parameters.task_name
        if scope == "task":
            if not clear_first and current_task not in (None, owner.upper()):
                raise ValueError(
                    f"Merging {owner.upper()} into {current_task} would change the task; "
                    "use clear_first=True to replace it"
                )
            expected_task = owner.upper()
        else:
            expected_task = current_task

        if required_task is not None:
            required_task = self._part(required_task).upper()
            if expected_task not in (None, required_task):
                raise ValueError(
                    f"Preset selects {expected_task}, but {required_task} was explicitly requested"
                )
            expected_task = required_task

        return self._apply_values(
            parameters, values,
            clear_first=clear_first,
            load_structural_info=load_structural_info,
            expected_task=expected_task,
        )
