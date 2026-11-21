"""SPLEED structured INFO uses exactly the native two-column SPLOUT format."""

import asyncio

import numpy as np

from ase2sprkkr.outputs.reader_registry import reader_registry
from ase2sprkkr.outputs.readers.spec import SpecResult
from ase2sprkkr.outputs.readers.spleed import (
    SpleedOutputParser, SpleedOutputReader, SpleedPointInfo, SpleedDiagramInfo,
    SpleedProgressParser,
)


def diagram(rows):
    return "i(e)-diagram\n\n" + "".join(
        f"{energy:12.5E}  {intensity:12.5E}\n".replace("E", "D")
        for energy, intensity in rows
    )


def test_streamed_progress_chunk_safe_and_no_fabricated_percent():
    events = []
    parser = SpleedProgressParser(on_info=lambda **event: events.append(event))
    output = diagram([(11., 2.), (12., 3.)])
    parser.feed(output[:9])
    assert events == []
    parser.feed(output[9:30])
    parser.feed(output[30:])
    parser.feed("End of i(e) data\n")
    parser.feed("program stopped regularly via STOP_REGULAR\n")
    parser.finish()

    assert [event["kind"] for event in events] == [
        "spleed_started", "spleed_diagram", "spleed_point",
        "spleed_diagram_complete", "spleed_finished",
    ]
    assert events[2]["data"] == SpleedPointInfo(1, 11., 2., "i(e)-diagram")
    assert events[3]["data"] == SpleedDiagramInfo("i(e)-diagram", 2)
    assert parser.format_info(**events[3]) == "SPLEED i(e)-diagram: 2 points read"
    assert not any("%" in (parser.format_info(**e) or "") for e in events)


def test_spleed_full_parser_preserves_spectrum_and_emits_same_events(monkeypatch, tmp_path):
    async def skip_common_header(_stream, _result):
        pass

    parser = SpleedOutputParser()
    events = []
    parser.on_info = lambda **event: events.append(event)
    monkeypatch.setattr(parser, "read_commons", skip_common_header)
    result = SpecResult(None, None, tmp_path, task_name="SPLEED")

    async def parse():
        stream = asyncio.StreamReader()
        stream.feed_data(
            (diagram([(11., 2.), (12., 3.)])
             + "End of i(e) data\nprogram stopped regularly via STOP_REGULAR\n").encode()
        )
        stream.feed_eof()
        await parser.read_output(stream, result)

    asyncio.run(parse())
    np.testing.assert_allclose(result.spleed_intensity.RAW_DATA(), [[11., 2.], [12., 3.]])
    assert result.spleed_info_points == 2
    assert [e["kind"] for e in events] == [
        "spleed_started", "spleed_diagram", "spleed_point",
        "spleed_diagram_complete", "spleed_finished",
    ]


def test_spleed_ignores_seven_column_diagram_and_does_not_invent_finish():
    events = []
    parser = SpleedProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("i(e)-diagram\n\n")
    parser.feed("1.0 2.0 3.0 4.0 5.0 6.0 7.0\n")
    parser.finish()
    assert parser._points == 0
    assert not any(e["kind"] in {"spleed_point", "spleed_diagram_complete", "spleed_finished"}
                   for e in events)


def test_spleed_registered_to_dedicated_reader():
    reader = reader_registry.class_for_task_name("SPLEED")
    assert reader is SpleedOutputReader
    assert reader.parser_class is SpleedOutputParser


def test_spleed_print_info_matches_callback(capsys):
    events = []
    parser = SpleedProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed(diagram([(10., 1.)]))
    parser.feed("end of table\n")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [parser.format_info(**event) for event in events]


def test_rotation_diagram_does_not_claim_angles_are_energies():
    events = []
    parser = SpleedProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(diagram([(45., 0.5)]).replace("i(e)-diagram", "rotation diagram"))
    point = next(e for e in events if e["kind"] == "spleed_point")
    assert point["data"].diagram == "rotation diagram"
    description = parser.format_info(**point)
    assert "rotation diagram" in description
    # SPLOUT writes ENERGY in the first column even for TYP=1/2.
    assert "E = 45 eV" in description


def test_full_spleed_reader_observes_lines_consumed_by_common_parser(monkeypatch, tmp_path):
    """The shared SPEC capture hook must see lines before FILES parsing ends."""
    async def read_commons(stream, _result):
        assert await stream.readline() == b"i(e)-diagram\\n"

    events = []
    parser = SpleedOutputParser(on_info=lambda **event: events.append(event))
    monkeypatch.setattr(parser, "read_commons", read_commons)
    result = SpecResult(None, None, tmp_path, task_name="SPLEED")

    async def parse():
        stream = asyncio.StreamReader()
        stream.feed_data(diagram([(11., 2.)]).encode())
        stream.feed_eof()
        await parser.read_output(stream, result)

    asyncio.run(parse())
    assert result.spleed_info_points == 1
    assert events[1] == {"kind": "spleed_diagram", "data": "i(e)-diagram"}
    assert events[2]["kind"] == "spleed_point"
    assert result.spleed_intensity.RAW_DATA()[0, 0] == 11.


def native_record(values, exponent="E"):
    """SPLOUT FORMAT 99004 / 99005 in SPR-KKR 9.7."""
    return "  ".join(f"{value:12.5E}".replace("E", exponent) for value in values)


def test_native_splout_branches_2_6_7_columns():
    from ase2sprkkr.outputs.readers.spec import SpecOutputParser

    # spec_rslabsub.f: TYP=0 or NREL=0: WRITE(*,99005) two D12.5.
    row2 = native_record([11., 0.25], exponent="D")
    # TYP=1/2, NREL=1: WRITE(*,99004) omits THETA (6 E12.5).
    row6 = native_record([12., 0.8, 0.2, 0.3, 0.4, 0.5])
    # TYP=1/2, NREL=1: WRITE(NOUT1,99004) includes THETA (7 E12.5).
    row7 = native_record([13., 45., 0.9, 0.2, 0.3, 0.4, 0.5])

    assert [len(row) for row in (row2, row6, row7)] == [26, 82, 96]
    assert SpecOutputParser.parse_spleed_diagram_row(row2) == (11., 0.25)
    assert SpecOutputParser.parse_spleed_diagram_row(row6) == (12., 0.8)
    assert SpecOutputParser.parse_spleed_diagram_row(row7) == (13., 0.9)

    for row, expected in ((row2, [11., 0.25]),
                          (row6, [12., 0.8]),
                          (row7, [13., 0.9])):
        # NOUT1 headers do not appear on WRITE(*) stdout.
        spectrum = SpecOutputParser.parse_spleed_intensity([row])
        np.testing.assert_allclose(spectrum.RAW_DATA(), [expected])
        events = []
        parser = SpleedProgressParser(on_info=lambda **event: events.append(event))
        parser.feed(row[:12])
        assert not any(e["kind"] == "spleed_point" for e in events)
        parser.feed(row[12:] + "\nnext section\n")
        point = next(e["data"] for e in events if e["kind"] == "spleed_point")
        assert point.energy_ev == expected[0]
        assert point.intensity == expected[1]
        assert point.diagram == "spectral output"

    assert SpecOutputParser.parse_spleed_diagram_row(" 2.0 3.0") is None


def test_spleed_typ_3_and_4_produce_no_splout_points():
    # SPLOUT has no WRITE branch for TYP=3/4.
    events = []
    parser = SpleedProgressParser(on_info=lambda **e: events.append(e))
    parser.feed("other SPLEED task output\nprogram stopped regularly via STOP_REGULAR\n")
    assert [e["kind"] for e in events] == ["spleed_started", "spleed_finished"]
