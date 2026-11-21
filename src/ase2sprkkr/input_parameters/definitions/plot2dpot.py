from ._plot2d_selectors import plot2d_parameters

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOT2DPOTInputParametersDefinition(_BaseInputParametersDefinition):
    @classmethod
    def create_definition(cls):
        return plot2d_parameters("PLOT2DPOT")

definition = PLOT2DPOTInputParametersDefinition
