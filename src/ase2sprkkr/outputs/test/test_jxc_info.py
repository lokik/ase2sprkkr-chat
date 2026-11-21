"""JXC INFO events use the same parser for saved files and live stdout."""

from pathlib import Path

from ase2sprkkr.outputs.readers.jxc import (
    JxcCurieTemperatureInfo,
    JxcOutputFileInfo,
    JxcOutputReader,
    JxcPairCountInfo,
    JxcProgressParser,
)


EXAMPLE = Path(__file__).parent.parent / "examples" / "Fe_jxc.out"


def test_jxc_native_output_reports_real_milestones():
    events = []
    reader = JxcOutputReader(
        None, None, str(EXAMPLE.parent),
        on_info=lambda **event: events.append(event),
    )
    result = reader.read_from_file(EXAMPLE)
    assert result.tau_pair_count == 89
    assert JxcPairCountInfo(89) in [
        event["data"] for event in events if event["kind"] == "jxc_pair_count"
    ]
    assert JxcCurieTemperatureInfo(763.4) in [
        event["data"] for event in events if event["kind"] == "jxc_curie_temperature"
    ]
    produced = [
        event["data"] for event in events if event["kind"] == "jxc_result_file"
    ]
    assert JxcOutputFileInfo("jxc", "Fe_JXC_XCPLTEN_Jij.dat") in produced
    assert JxcOutputFileInfo("dij", "Fe_JXC_XCPLTEN_Dij.dat") in produced
    assert JxcOutputFileInfo("dmi", "Fe_JXC_DMIVEC_Dij.dat") in produced
    assert reader.parser.format_info("jxc_pair_count", JxcPairCountInfo(89)) == (
        "Calculating 89 TAU(I,J) pairs"
    )


def test_jxc_live_info_handles_chunk_boundaries():
    events = []
    parser = JxcProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(" number of TAU(I,J)'s to be calculated: ")
    assert not events
    parser.feed("89\nresults written to file:Fe_JXC_XCPLTEN_Jij.dat\n")
    assert events == [
        {"kind": "jxc_pair_count", "data": JxcPairCountInfo(89)},
        {"kind": "jxc_result_file", "data": JxcOutputFileInfo(
            "jxc", "Fe_JXC_XCPLTEN_Jij.dat",
        )},
    ]
