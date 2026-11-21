"""Input-parameter schema definitions.

Each public module exposes ``definition``: an InputParametersDefinition
subclass with a ``create_definition()`` classmethod. The registry reads its
``task_selectors`` metadata without building the schema and caches the
result of ``create_definition()`` when first requested. Legacy user modules
with an ``input_parameters_definition`` factory remain supported.

Standard schemas attach their 1:1 task as ``_task_definition``; schemas
shared by several tasks leave it unset and resolve task names from values.
"""

import os
import platformdirs

__path__.append(
    os.path.join(
        platformdirs.user_config_dir("ase2sprkkr", "ase2sprkkr"),
        "input_parameters",
    )
)
