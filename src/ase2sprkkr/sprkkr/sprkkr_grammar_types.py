"""In this module, the :class:`GrammarTypes<ase2sprkkr.common.grammar_types.GrammarType>`,
that are specific for SPR-KKR, are present."""

from ..common.grammar_types import ObjectNumber
from .sites import Site as _Site
from .atomic_types import AtomicType as _AtomicType


class Site(ObjectNumber):
    type = _Site

    def set_from_atoms(self, option, atoms, io_data):
        if atoms is not None:
            option.clear_result()
        value = option()
        if isinstance(value, _Site):
            if atoms is None:
                raise ValueError("Resolving a site reference requires atoms")
            option.result = io_data.site_number(value)


Site.I = Site()


class AtomicType(ObjectNumber):
    type = _AtomicType

    def set_from_atoms(self, option, atoms, io_data):
        if atoms is not None:
            option.clear_result()
        value = option()
        if isinstance(value, _AtomicType):
            if atoms is None:
                raise ValueError("Resolving an atomic type reference requires atoms")
            option.result = io_data.types.value_to_class_id[value]


AtomicType.I = AtomicType()
