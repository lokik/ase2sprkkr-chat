"""Native PLOTRHO selector."""
from ._plot_selectors import plot_parameters
from ._plot_selectors import _OUTPUTS
from ..task_definition import TaskDefinition

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTRHOInputParametersDefinition(_BaseInputParametersDefinition):
    _task_definition = TaskDefinition(
        "PLOTRHO", supports=(), experimental=True, executable="kkrgen",
        mpi=False, result_files=_OUTPUTS["PLOTRHO"],
    )

    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTRHO", 'Plot the stored charge and spin densities; FULLPOT includes nonspherical density components.', task_definition=cls._task_definition)

definition = PLOTRHOInputParametersDefinition
