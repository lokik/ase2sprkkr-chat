"""Native PLOTSFN selector."""
from ._plot_selectors import plot_parameters
from ._plot_selectors import _OUTPUTS
from ..task_definition import TaskDefinition

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTSFNInputParametersDefinition(_BaseInputParametersDefinition):
    _task_definition = TaskDefinition(
        "PLOTSFN", supports={"3d", "2d"}, experimental=True, executable="kkrgen",
        mpi=False, result_files=_OUTPUTS["PLOTSFN"],
    )

    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTSFN", 'Plot the radial spherical-harmonic expansion coefficients of shape functions from an existing FULLPOT potential, for each mesh. ASA has no such output.', task_definition=cls._task_definition)

definition = PLOTSFNInputParametersDefinition
