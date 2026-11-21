"""Native PLOTRHO selector."""
from ._plot_selectors import plot_parameters

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTRHOInputParametersDefinition(_BaseInputParametersDefinition):
    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTRHO", 'Plot the stored charge and spin densities; FULLPOT includes nonspherical density components.')

definition = PLOTRHOInputParametersDefinition
