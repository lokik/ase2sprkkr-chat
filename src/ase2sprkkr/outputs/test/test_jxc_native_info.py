"""Native JXC INFO branch coverage from SPR-KKR 9.7.

Output formats verified in xcpljij.f, xcpltensor.f, cluxcpljij.f,
cluxcpltensor.f, and init_mod_tauij_star.f.
"""

import asyncio
import warnings
from types import SimpleNamespace

import pytest

from ase2sprkkr.outputs.readers.jxc import (
    JxcOutputParser, JxcOutputFileInfo, JxcProgressParser,
)


@pytest.mark.parametrize("name,kind", [
    ("Fe_J_ij.dat", "jxc"),             # XCPLJIJ
    ("Fe_JJij.dat", "jxc"),              # CLUXCPLTENSOR
    ("Fe_XCPLTEN_Jij.dat", "jxc"),       # XCPLTENSOR
    ("Fe_XCPLTEN_Dij.dat", "dij"),       # XCPLTENSOR
    ("Fe_DMIVEC_Dij.dat", "dmi"),        # DMI vector
    ("Fe_Dij_x.dat", "dmi_x"),           # CLUXCPLTENSOR
    ("Fe_Dij_y.dat", "dmi_y"),
    ("Fe_Dij_z.dat", "dmi_z"),
])
def test_native_jxc_filename_branches(name, kind):
    events = []
    parser = JxcProgressParser(on_info=lambda **e: events.append(e))
    parser.feed("          results written to file:" + name[:9])
    parser.feed(name[9:] + "\n")
    assert events == [{
        "kind": "jxc_result_file",
        "data": JxcOutputFileInfo(kind, name),
    }]


def test_native_jxc_curie_and_pair_count():
    events = []
    parser = JxcProgressParser(on_info=lambda **e: events.append(e))
    parser.feed("     number of TAU(I,J)'s to be calculated:   16\n")
    parser.feed("     Curie temperature within mean field")
    parser.feed(" approximation  T_C =    123.4 K\n")
    assert [e["kind"] for e in events] == [
        "jxc_pair_count", "jxc_curie_temperature",
    ]
    assert events[-1]["data"].temperature_kelvin == 123.4


def test_native_jxc_plot_file_is_not_misparsed_as_exchange_tensor():
    events = []
    parser = JxcProgressParser(on_info=lambda **e: events.append(e))
    parser.feed("     exchange coupling parameters written to file Fe_Jij.agr\n")
    assert events == [{
        "kind": "jxc_plot_file", "data": "Fe_Jij.agr",
    }]


def test_jxc_no_curie_is_allowed_by_native_cluster_and_eigensolver_branches(monkeypatch):
    async def skip_commons(_stream, _result):
        pass

    parser = JxcOutputParser()
    monkeypatch.setattr(parser, "read_commons", skip_commons)
    result = SimpleNamespace()

    async def parse():
        stream = asyncio.StreamReader()
        stream.feed_data(b"regular output without a Curie temperature\n")
        stream.feed_eof()
        await parser.read_output(stream, result)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(parse())
    assert not caught
    assert result.mean_field_curie_temperature is None
