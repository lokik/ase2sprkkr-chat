"""DOS INFO callbacks share one parser and never infer progress."""

from types import SimpleNamespace

from ase2sprkkr.outputs.readers.dos import (
    DosOutputParser,
    DosProgressInfo,
    DosProgressParser,
)


def test_dos_progress_is_structured_and_chunk_safe():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(" 13/300 do")
    assert events == []
    parser.feed("ne.\n14/300 completed\n")
    assert events == [
        {"kind": "dos_started", "data": None},
        {"kind": "dos_progress", "data": DosProgressInfo(13, 300)},
        {"kind": "dos_progress", "data": DosProgressInfo(14, 300)},
    ]
    assert parser.format_info(**events[-1]) == "14/300 done."


def test_dos_does_not_invent_progress_from_fractions_or_unfinished_points():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("ENERGY NE=300\n13/300\n1/0 done.\n301/300 done.\n")
    assert events == [{"kind": "dos_started", "data": None}]
    parser.feed("12/300 done.\n12/300 done.\n")
    assert events[-1] == {"kind": "dos_progress", "data": DosProgressInfo(12, 300)}
    assert len(events) == 2


def test_dos_progress_prints_the_same_info_as_callback(capsys):
    events = []
    parser = DosProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed("DOS energy points: 3 of 20 processed\n")
    assert events[-1] == {"kind": "dos_progress", "data": DosProgressInfo(3, 20)}
    assert capsys.readouterr().out == "Calculating density of states...\n3/20 done.\n"


def test_full_dos_output_parser_emits_same_events_and_retains_lines(tmp_path):
    class ParserWithoutHeader(DosOutputParser):
        async def read_commons(self, stdout, result):
            pass

    path = tmp_path / "dos.out"
    path.write_text("13/300 done.\n14/300 done.\nUnrelated native message\n")
    events = []
    parser = ParserWithoutHeader(on_info=lambda **event: events.append(event))
    result = SimpleNamespace()
    parsed, error, return_code = parser.read_from_file(path, read_args=[result])
    assert parsed is result
    assert error is None and return_code == 0
    assert result.output_lines == [
        "13/300 done.",
        "14/300 done.",
        "Unrelated native message",
    ]
    assert [event["data"].completed for event in events if event["kind"] == "dos_progress"] == [13, 14]
    assert events[0] == {"kind": "dos_started", "data": None}



def test_native_dos_energy_headers_are_per_energy_not_per_atomic_type():
    events = []
    parser = DosProgressParser(
        on_info=lambda **event: events.append(event), total_points=300
    )
    parser.feed(" 13 E= 0.4061 0.0100 IT= 1 Ti\n")
    parser.feed(" 13 E= 0.4061 0.0100 IT= 2 Co\n")
    assert [event["kind"] for event in events] == ["dos_started"]

    parser.feed(" 14 E= 0.4081 0.0100 IT= 1 Ti\n")
    assert events[-1] == {
        "kind": "dos_progress", "data": DosProgressInfo(13, 300)
    }
    parser.feed(" 14 E= 0.4081 0.0100 IT= 2 Co\n")
    assert sum(event["kind"] == "dos_progress" for event in events) == 1
    parser.feed(" 15 E= 0.4101 0.0100 IT= 1 Ti\n")
    assert events[-1] == {
        "kind": "dos_progress", "data": DosProgressInfo(14, 300)
    }


def test_native_dos_progress_without_known_total_has_no_fake_fraction():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("34 E= 0.4061 0.0100 IT= 1 Ti\n")
    parser.feed("35 E= 0.4161 0.0100 IT= 1 Ti\n")
    assert events[-1] == {
        "kind": "dos_progress", "data": DosProgressInfo(34, None)
    }
    assert parser.format_info(**events[-1]) == "34 energy points done."


def test_dos_total_requires_one_unambiguous_energy_segment():
    from ase2sprkkr.outputs.readers.dos import dos_total_points_from_parameters

    def params(ne):
        return SimpleNamespace(ENERGY=SimpleNamespace(NE=lambda: ne))

    assert dos_total_points_from_parameters(params([300])) == 300
    assert dos_total_points_from_parameters(params(300)) == 300
    assert dos_total_points_from_parameters(params([100, 200])) is None
    assert dos_total_points_from_parameters(params([])) is None
    assert dos_total_points_from_parameters(None) is None



def test_native_dos_mpi_energy_index_does_not_invent_global_completion():
    """GEN MPI_ELOOP/MPI_KLOOP may print only one process's energies."""
    from ase2sprkkr.outputs.readers.dos import DosEnergyInfo

    events = []
    parser = DosProgressParser(
        on_info=lambda **event: events.append(event), total_points=200
    )
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 151 E= 0.4101 0.0010 IT= 1 Fe\n")
    parser.feed(" 152 E= 0.4111 0.0010 IT= 1 Fe\n")
    assert not any(e["kind"] == "dos_progress" for e in events)
    assert [e["data"] for e in events if e["kind"] == "dos_energy_index"] == [
        DosEnergyInfo(151, 200), DosEnergyInfo(152, 200),
    ]
    assert parser.format_info("dos_energy_index", DosEnergyInfo(151, 200)) == (
        "DOS energy index 151 observed (MPI completion unknown)"
    )


def test_native_dos_both_printed_header_variants():
    """CALCDOS 99006 contains L=, while 99008 omits it (NWRLOG)."""
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=3)
    parser.feed(" 1 E= 0.1234 0.0010 L= 2 IT= 1 Fe\n")
    parser.feed(" 2 E= 0.1235 0.0010          IT= 1 Fe\n")
    assert [e["data"].completed for e in events if e["kind"] == "dos_progress"] == [1]
    parser.feed(" 3 E= 0.1236 0.0010 L= 0 IT= 1 Fe\n")
    assert [e["data"].completed for e in events if e["kind"] == "dos_progress"] == [1, 2]
