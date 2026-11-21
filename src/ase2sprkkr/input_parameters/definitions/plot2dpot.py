from ._plot2d_selectors import plot2d_parameters
from ._plot2d_selectors import _PLY_OUTPUTS
from ..task_definition import TaskDefinition

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOT2DPOTInputParametersDefinition(_BaseInputParametersDefinition):
    _task_definition = TaskDefinition(
        "PLOT2DPOT", supports={"3d", "2d"}, experimental=True, executable="kkrgen",
        mpi=False, output_files=_PLY_OUTPUTS,
    )

    @classmethod
    def create_definition(cls):
        return plot2d_parameters("PLOT2DPOT", task_definition=cls._task_definition)

definition = PLOT2DPOTInputParametersDefinition
