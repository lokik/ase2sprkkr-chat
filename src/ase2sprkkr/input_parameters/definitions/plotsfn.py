"""Native PLOTSFN selector."""
from ._plot_selectors import plot_parameters

from ..input_parameters_definitions import InputParametersDefinition as _BaseInputParametersDefinition

class PLOTSFNInputParametersDefinition(_BaseInputParametersDefinition):
    @classmethod
    def create_definition(cls):
        return plot_parameters("PLOTSFN", 'Plot the radial spherical-harmonic expansion coefficients of shape functions from an existing FULLPOT potential, for each mesh. ASA has no such output.')

definition = PLOTSFNInputParametersDefinition
