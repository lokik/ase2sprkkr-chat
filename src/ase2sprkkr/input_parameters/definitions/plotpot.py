"""Native PLOTPOT selector."""
from ._plot_selectors import plot_parameters

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTPOTInputParametersDefinition(_BaseInputParametersDefinition):
    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTPOT", 'Plot the stored radial potential and exchange field; FULLPOT includes nonspherical potential components.')

definition = PLOTPOTInputParametersDefinition
