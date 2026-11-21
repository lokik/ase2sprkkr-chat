"""Input-parameter schema definitions.

Each public module exposes ``definition``: an InputParametersDefinition
subclass with a ``create_definition()`` classmethod. The registry reads its
``task_selectors`` metadata without building the schema and caches the
result of ``create_definition()`` when first requested. Legacy user modules
with an ``input_parameters_definition`` factory remain supported.

Standard IPD classes declare their 1:1 task as a class attribute
``_task_definition``. ``TaskRegistry.names()`` reads this declaration without
constructing the IPD; schemas shared by multiple tasks leave it unset and
resolve task names from parameter values.
"""

import os
import platformdirs

__path__.append(
    os.path.join(
        platformdirs.user_config_dir("ase2sprkkr", "ase2sprkkr"),
        "input_parameters",
    )
)
