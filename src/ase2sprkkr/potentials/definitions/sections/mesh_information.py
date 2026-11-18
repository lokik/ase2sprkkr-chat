from ...potential_definitions import PotSectionDefinition, PotValueDefinition
from ...potential_sections import UniqueListSection

from ....common.grammar_types import DefKeyword, Table, Array
from ....sprkkr.radial_meshes import ExponentialMesh

from ....sprkkr.io_data import ReadIoData, WriteIoData
from ....sprkkr.sprkkr_atoms import SPRKKRAtoms


class MeshInformationSection(UniqueListSection):
    _value_name = "meshes"
    _value_class = ExponentialMesh

    def _set_from_atoms(self, atoms: SPRKKRAtoms, write_io_data: WriteIoData):
        ul = getattr(write_io_data, self._value_name)
        self["DATA"].set([i.to_tuple() for i in ul.iter_unique()])
        fullpot = [i.fullpot_tuple() for i in ul.iter_unique()]

        found = None
        for i in fullpot:
            if i is None:
                if found is True:
                    break
                found = False
            else:
                if found is False:
                    break
                found = True
        else:
            if found:
                self["FULLPOT"].set(fullpot)
            return
        raise Exception("FULLPOT mesh information missing for some meshes")

    def _update_atoms(self, atoms: SPRKKRAtoms, read_io_data: ReadIoData):
        super()._update_atoms(atoms, read_io_data)
        if self.FULLPOT() is not None:
            for mesh, full in zip(read_io_data[self._value_name], self["FULLPOT"]()):
                mesh.set_fullpot(full)


def mesh_information_members(*, cluster=False):
    """Reuse radial mesh fields while respecting CLUPOT global mesh IDs."""
    V = PotValueDefinition
    radial_defaults = {
        "R(1)": 1e-6, "DX": 2e-2, "JRMT": 0,
        "RMT": 0.0, "JRWS": 721, "RWS": 0.0,
    }
    radial_columns = {name: type(default) for name, default in radial_defaults.items()}
    fullpot_columns = {
        "JRNS1": int, "JRCRI": int, "NPAN": int, "JRCUT": Array(int),
    }
    if cluster:
        data = Table({"IM": int, **radial_columns}, free_header=True)
        fullpot = Table({"IM": int, **fullpot_columns}, free_header=True)
    else:
        data = Table(radial_defaults, numbering="IM",
                     numbering_format=">5", format=">16")
        fullpot = Table(fullpot_columns, numbering="IM",
                        numbering_format=">5", format=">16")
    return [
        V("MESH-TYPE", DefKeyword("EXPONENTIAL")),
        V("DATA", data, **({
            "name_in_grammar": False, "is_required": False,
        } if cluster else {})),
        V("FULLPOT", fullpot, is_optional=True,
          **({"name_in_grammar": False} if cluster else {})),
    ]


class MeshInformationSectionDefinition(PotSectionDefinition):
    def __init__(self, name="MESH INFORMATION", **kwargs):
        super().__init__(name, mesh_information_members(),
                         has_hidden_members=True, **kwargs)

    result_class = MeshInformationSection


section = MeshInformationSectionDefinition
