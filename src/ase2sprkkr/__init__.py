"""ASE interface to SPR-KKR, with lazy public class exports.

Keep the user configuration initialized on import as before. Importing the
package must not construct calculators, potential grammars, result parsers, or
ASE 4 plugins until they are actually requested.
"""

import os
import warnings
from importlib import import_module
from importlib.util import find_spec

from .version import __version__

__all__ = (
    "SPRKKR",
    "InputParameters",
    "Potential",
    "PeriodicPotential",
    "EmbeddedPotential",
    "ClusterPotential",
    "SPRKKRAtoms",
    "promote_ase_atoms",
    "ResultFile",
    "TaskResult",
    "ResultLoader",
    "FileAttribution",
    "__version__",
    "config",
)

# Import only the requested object, and cache its identity in module globals.
# __getattr__ also preserves the deprecated OutputFile spelling.
_LAZY_EXPORTS = {
    "SPRKKR": (".sprkkr.calculator", "SPRKKR"),
    "InputParameters": (".input_parameters.input_parameters", "InputParameters"),
    "Potential": (".potentials.potential_base", "Potential"),
    "PeriodicPotential": (".potentials.potentials", "PeriodicPotential"),
    "EmbeddedPotential": (".potentials.cluster_potentials", "EmbeddedPotential"),
    "ClusterPotential": (".potentials.cluster_potentials", "ClusterPotential"),
    "SPRKKRAtoms": (".sprkkr.sprkkr_atoms", "SPRKKRAtoms"),
    "ResultFile": (".result_files", "ResultFile"),
    "TaskResult": (".outputs.task_result", "TaskResult"),
    "ResultLoader": (".outputs.result_loader", "ResultLoader"),
    "FileAttribution": (".outputs.result_options", "FileAttribution"),
}


def __getattr__(name):
    if name == "OutputFile":
        warnings.warn(
            "ase2sprkkr.OutputFile is deprecated (the class have been renamed); "
            "use ase2sprkkr.ResultFile instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return __getattr__("ResultFile")
    if name == "promote_ase_atoms":
        value = __getattr__("SPRKKRAtoms").promote_ase_atoms
    elif name in _LAZY_EXPORTS:
        module_name, attribute = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"OutputFile"})


def _init():
    from .configuration import load_user_preferences

    # ASE 4 discovers our plugins through the 'ase.plugins' entry point.
    # Importing ase._4.plugins here pulls in ASE I/O and viewers (~650 ms).
    # Older ASE versions have no such discovery and need our legacy registration.
    if find_spec("ase._4") is None:
        from .ase.register import register
        register()

    if not os.environ.get("ASE2SPRKKR_NO_USER_PROFILE", False):
        load_user_preferences()


_init()
del _init

# Retain the established global configuration identity and initialization order.
from .configuration import config as config
