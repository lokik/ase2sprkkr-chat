"""ARPES INFO callbacks reflect only native SPEC.out milestones."""

from ase2sprkkr.outputs.reader_registry import reader_registry
from ase2sprkkr.outputs.readers.arpes import (
    ArpesBarrierInfo,
    ArpesOutputParser,
    ArpesOutputReader,
    ArpesPhotonInfo,
    ArpesProgressParser,
    ArpesStokesInfo,
)

# Copied from the native SPEC.out geometry/polarization example used by
# test_spec_parser.py. No synthetic energy-point counter is introduced.
NATIVE = (
    "parameter for potential barrier:\n"
    "# ibar:  1\n"
    "# epsx: 0.5000000D-02\n"
    "** start new input **\n"
    "lattice constant      :     4.6298300\n"
    "vacuum-wavevector of the photonfield:   0.00000000   0.00000000   0.00611428\n"
    "stookes-vector    of the photonfield  0.10000E+01  100.000%    0.000%    0.000%\n"
)


def test_arpes_live_info_is_structured_and_chunk_safe():
    events = []
    parser = ArpesProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(NATIVE[:20])
    assert events == []
    parser.feed(NATIVE[20:70])
    parser.feed(NATIVE[70:123])
    parser.feed(NATIVE[123:])
    parser.feed("lattice constant: 4.6298300\n")  # repeated setup
    assert events == [
        {"kind": "arpes_started", "data": None},
        {"kind": "arpes_barrier", "data": ArpesBarrierInfo("ibar", 1.)},
        {"kind": "arpes_barrier", "data": ArpesBarrierInfo("epsx", 0.005)},
        {"kind": "arpes_lattice", "data": 4.62983},
        {"kind": "arpes_photon", "data": ArpesPhotonInfo((0., 0., 0.00611428))},
        {"kind": "arpes_stokes", "data": ArpesStokesInfo(1., (100., 0., 0.))},
    ]
    assert parser.format_info(**events[4]).startswith("Photon wavevector:")
    assert not any(event["kind"] == "arpes_progress" for event in events)

    parser.feed("program stopped regularly via STOP_REG")
    assert events[-1]["kind"] == "arpes_stokes"
    parser.feed("ULAR\nprogram stopped regularly via STOP_REGULAR\n")
    assert events[-1] == {"kind": "arpes_finished", "data": None}
    assert [event["kind"] for event in events].count("arpes_finished") == 1


def test_arpes_no_completion_from_eof_or_requested_mesh():
    events = []
    parser = ArpesProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(NATIVE + "100 of 300 energy points\n")
    parser.finish()
    assert "arpes_finished" not in [event["kind"] for event in events]
    assert "arpes_progress" not in [event["kind"] for event in events]


def test_arpes_saved_output_reader_uses_same_info_and_keeps_output(tmp_path):
    output = tmp_path / "Fe_SPEC.out"
    output.write_bytes(
        b"SPRKKR VERSION 9.7 build\nFILES:\n\n\n"
        + NATIVE.encode()
        + b"program stopped regularly via STOP_REGULAR\n"
    )
    events = []
    reader = ArpesOutputReader(
        None, None, str(tmp_path), task_name="ARPES",
        on_info=lambda **event: events.append(event),
    )
    result = reader.read_from_file(output)
    assert result.output_lines[0] == "parameter for potential barrier:"
    assert result.output_lines[-1] == "program stopped regularly via STOP_REGULAR"
    assert [event["kind"] for event in events] == [
        "arpes_started", "arpes_barrier", "arpes_barrier",
        "arpes_lattice", "arpes_photon", "arpes_stokes", "arpes_finished",
    ]
    assert reader_registry.class_for_task_name("ARPES") is ArpesOutputReader
    assert reader.parser.__class__ is ArpesOutputParser


def test_arpes_printed_info_is_the_callback_format(capsys):
    events = []
    parser = ArpesProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed(NATIVE)
    assert capsys.readouterr().out.splitlines() == [
        parser.format_info(**event) for event in events
    ]


def test_arpes_finish_flushes_incomplete_final_line():
    events = []
    parser = ArpesProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("lattice constant: 4.5")
    assert events == []
    parser.finish()
    assert events == [
        {"kind": "arpes_started", "data": None},
        {"kind": "arpes_lattice", "data": 4.5},
    ]
