"""Native PLOTPOT selector."""
from ._plot_selectors import plot_parameters
from ._plot_selectors import _OUTPUTS
from ..task_definition import TaskDefinition

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTPOTInputParametersDefinition(_BaseInputParametersDefinition):
    _task_definition = TaskDefinition(
        "PLOTPOT", supports={"3d", "2d"}, experimental=True, executable="kkrgen",
        mpi=False, output_files=_OUTPUTS["PLOTPOT"],
    )

    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTPOT", 'Plot the stored radial potential and exchange field; FULLPOT includes nonspherical potential components.', task_definition=cls._task_definition)

definition = PLOTPOTInputParametersDefinition
