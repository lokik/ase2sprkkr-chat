"""Native XRAYSPEC 9.7 INFO: FORMAT 99012/99013 and early-exit branches."""

import asyncio
from types import SimpleNamespace

from ase2sprkkr.outputs.reader_registry import reader_registry
from ase2sprkkr.outputs.readers.xas import (
    XasCoreInfo, XasEnergyInfo, XasOutputParser, XasOutputReader,
    XasProgressParser, xas_total_points_from_parameters,
)


def loop(index, passage=1):
    # XRAYSPEC FORMAT 99012: ' LOOP',I2,I5,' E=',2F6.3,'  IQ=',I2,'  IT=',I2
    return f" LOOP{passage:2d}{index:5d} E={0.735:6.3f}{0.001:6.3f}  IQ={1:2d}  IT={1:2d}\n"


def complete(n):
    # XRAYSPEC FORMAT 99013 (WRITE(6), not the separate .rat file)
    return (
        f" X-RAY ABSORPTION-RATE CALCULATED FOR {n:4d}"
        " ENERGY-POINTS   >>> FEIERABEND\n"
    )


def test_native_xas_progress_is_chunk_safe_and_counts_only_completed_energies():
    events = []
    parser = XasProgressParser(
        on_info=lambda **event: events.append(event), total_points=180
    )
    source = (
        "          RATE-FILE :  ( 7) Fe_Cu_L23.rat\n"
        "    CORE QUANTUM-NUMBERS  FOR  IT= 1:   N= 2  L= 1\n"
        + loop(1) + loop(2) + loop(2) + loop(180)
        + complete(180)
        + "    finished in subroutine <XRAYSPEC>\n"
        + "    program stopped regularly via STOP_REGULAR\n"
    )
    for index in range(0, len(source), 7):
        parser.feed(source[index:index + 7])
    parser.finish()
    assert [x["kind"] for x in events] == [
        "xas_started", "xas_rate_file", "xas_core",
        "xas_energy", "xas_energy", "xas_energy",
        "xas_complete", "xas_finished",
    ]
    assert events[1]["data"] == "Fe_Cu_L23.rat"
    assert events[2]["data"] == XasCoreInfo(1, 2, 1)
    assert [e["data"] for e in events if e["kind"] == "xas_energy"] == [
        XasEnergyInfo(1, 0, 180),
        XasEnergyInfo(2, 1, 180),
        XasEnergyInfo(180, 179, 180),
    ]
    assert parser.format_info(**events[-2]) == (
        "XAS spectrum calculated: 180 energy points"
    )
    assert parser.completed_points == 180


def test_format_widths_touch_without_whitespace_and_unknown_total():
    events = []
    parser = XasProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(loop(99999))
    assert events[-1]["data"] == XasEnergyInfo(99999, 99998, None)
    assert parser.format_info(**events[-1]) == (
        "Calculating XAS energy point 99999..."
    )


def test_wrong_pass_and_generic_stop_do_not_claim_xas_completion():
    events = []
    parser = XasProgressParser(
        on_info=lambda **event: events.append(event), total_points=20
    )
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(loop(1, passage=2))
    parser.feed("program stopped regularly via STOP_REGULAR\n")
    parser.finish()
    assert events == []
    assert parser.completed_points is None


def test_truncated_xas_does_not_claim_completion():
    events = []
    parser = XasProgressParser(
        on_info=lambda **event: events.append(event), total_points=20
    )
    parser.feed(loop(5) + "finished in subroutine <XRAYSPEC>\n")
    parser.finish()
    assert events[-1]["data"] == XasEnergyInfo(5, 4, 20)
    assert not any(e["kind"] in ("xas_complete", "xas_finished") for e in events)


def test_source_summary_is_authoritative_without_input_parameters():
    events = []
    parser = XasProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(complete(120))
    assert [e["kind"] for e in events] == ["xas_started", "xas_complete"]
    assert parser.completed_points == 120


def test_full_output_reader_observes_lines_consumed_by_files_parser(
    tmp_path, monkeypatch
):
    events = []
    parser = XasOutputParser(on_info=lambda **event: events.append(event))
    async def read_first_line(stream, _result):
        assert await stream.readline() == b"          RATE-FILE :  ( 7) Fe_L23.rat\n"
    monkeypatch.setattr(parser, "read_commons", read_first_line)
    result = SimpleNamespace(input_parameters=None)

    async def parse():
        stream = asyncio.StreamReader()
        stream.feed_data(
            ( "          RATE-FILE :  ( 7) Fe_L23.rat\n"
              + loop(1) + loop(2) + complete(2)
              + "finished in subroutine <XRAYSPEC>\n").encode()
        )
        stream.feed_eof()
        await parser.read_output(stream, result)

    asyncio.run(parse())
    assert result.xas_rate_filename == "Fe_L23.rat"
    assert result.xas_completed_points == 2
    assert [e["kind"] for e in events] == [
        "xas_started", "xas_rate_file", "xas_energy",
        "xas_energy", "xas_complete", "xas_finished",
    ]
    assert result.output_lines[0] == loop(1).rstrip()
    assert result.output_lines[-1] == "finished in subroutine <XRAYSPEC>"


def test_native_info_print_uses_same_formatter_as_callbacks(capsys):
    events = []
    parser = XasProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed("RATE-FILE :  ( 7) Fe_L23.rat\n" + loop(1))
    assert capsys.readouterr().out.splitlines() == [
        parser.format_info(**event) for event in events
    ]


def test_registered_xas_reader_and_input_energy_mesh():
    assert reader_registry.class_for_task_name("XAS") is XasOutputReader

    def parameters(ne):
        return SimpleNamespace(ENERGY=SimpleNamespace(NE=lambda: ne))

    assert xas_total_points_from_parameters(parameters([180])) == 180
    assert xas_total_points_from_parameters(parameters([100, 80])) is None
    assert xas_total_points_from_parameters(parameters([])) is None
    assert xas_total_points_from_parameters(None) is None
