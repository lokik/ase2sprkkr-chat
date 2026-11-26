"""DOS INFO callbacks report measured serial and estimated MPI progress."""

from types import SimpleNamespace

from ase2sprkkr.outputs.task_result import OutputMatch, TaskResult
from ase2sprkkr.outputs.readers.dos import (
    DosOutputParser,
    DosOutputReader,
    DosProgressInfo,
    DosProgressParser,
)


def test_dos_files_entry_is_tentative_not_decisive():
    output = b"TASK = NONE\nFILES:\n potential : Fe2.pot\n DOS : x.dos\n\n"
    assert DosOutputReader.matches_output(output) is OutputMatch.MAYBE
    assert DosOutputReader.matches_output(b"TASK = NONE\nFILES:\n potential : Fe2.pot\n\n") is False


def test_dos_progress_is_structured_and_chunk_safe():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed(" 13/300 do")
    assert events == []
    parser.feed("ne.\n14/300 completed\n")
    assert [e for e in events if e["kind"] != "progress"] == [
        {"kind": "dos_started", "data": None},
        {"kind": "dos_progress", "data": DosProgressInfo(13, 300)},
        {"kind": "dos_progress", "data": DosProgressInfo(14, 300)},
    ]
    assert parser.format_info(**events[-1]) == "14/300 done."


def test_dos_does_not_invent_progress_from_fractions_or_unfinished_points():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("ENERGY NE=300\n13/300\n1/0 done.\n301/300 done.\n")
    assert [e for e in events if e["kind"] != "progress"] == [{"kind": "dos_started", "data": None}]
    parser.feed("12/300 done.\n12/300 done.\n")
    assert events[-1] == {"kind": "dos_progress", "data": DosProgressInfo(12, 300)}
    assert len([e for e in events if e["kind"] != "progress"]) == 2


def test_dos_progress_prints_the_same_info_as_callback(capsys):
    events = []
    parser = DosProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed("DOS energy points: 3 of 20 processed\n")
    assert events[-1] == {"kind": "dos_progress", "data": DosProgressInfo(3, 20)}
    assert capsys.readouterr().out == "Calculating density of states...\n3/20 done.\n"


def test_standalone_dos_output_loads_with_mpi_header(tmp_path):
    """Importing a saved DOS .out must not confuse the MPI regex with its count."""
    from ase2sprkkr.outputs.task_result import TaskResult

    output = tmp_path / "Fe_DOS.out"
    output.write_text(
        "TASK = DOS\n"
        "FILES:\n\n"
        "DOS: Fe_DOS.dos\n\n"
        "MPI calculation with NPROCS = 4\n"
        " 151 E= 0.4101 0.0010 IT= 1 Fe\n",
        encoding="utf-8",
    )

    result = TaskResult.from_file(output)

    assert result.task_name == "dos"
    assert result.output_lines == [
        "MPI calculation with NPROCS = 4",
        " 151 E= 0.4101 0.0010 IT= 1 Fe",
    ]


def test_full_dos_output_parser_emits_same_events_and_retains_lines(tmp_path):
    class ParserWithoutHeader(DosOutputParser):
        async def read_commons(self, stdout, result):
            pass

    path = tmp_path / "dos.out"
    path.write_text("13/300 done.\n14/300 done.\nUnrelated native message\n")
    events = []
    parser = ParserWithoutHeader(on_info=lambda **event: events.append(event))
    result = TaskResult(None, None, tmp_path, task_name="DOS")
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
    assert [event["kind"] for event in events if event["kind"] != "progress"] == ["dos_started"]

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



def test_native_dos_mpi_rank_zero_block_provides_approximate_progress():
    """GEN MPI_SET_IPROCE assigns high contiguous IE block to rank 0."""
    events = []
    parser = DosProgressParser(
        on_info=lambda **event: events.append(event), total_points=200
    )
    parser.feed("MPI calculation with NPROCS = 4\n")
    # Rank 0 gets IE=151..200. It is NOT 75% done on IE=151.
    parser.feed(" 151 E= 0.4101 0.0010 IT= 1 Fe\n")
    parser.feed(" 151 E= 0.4101 0.0010 IT= 2 Fe\n")
    parser.feed(" 152 E= 0.4111 0.0010 IT= 1 Fe\n")
    parser.feed(" 176 E= 0.4121 0.0010 IT= 1 Fe\n")
    progress = [e["data"] for e in events if e["kind"] == "dos_progress"]
    assert progress == [
        DosProgressInfo(0, 200, estimated=True),
        DosProgressInfo(4, 200, estimated=True),
        DosProgressInfo(100, 200, estimated=True),
    ]
    assert parser.format_info("dos_progress", progress[-1]) == (
        "~50% done (estimated from MPI rank 0)."
    )
    assert not any(e["kind"] == "dos_energy_index" for e in events)
    parser.feed(" 200 E= 0.4131 0.0010 IT= 1 Fe\n")
    assert [e["data"] for e in events if e["kind"] == "dos_progress"][-1] == (
        DosProgressInfo(196, 200, estimated=True)
    )
    parser.feed("program stopped regularly via STOP_REGULAR\n")
    assert [e["data"] for e in events if e["kind"] == "dos_progress"][-1] == (
        DosProgressInfo(200, 200)
    )


def test_native_dos_mpi_uneven_block_size_and_unknown_mapping():
    from ase2sprkkr.outputs.readers.dos import DosEnergyInfo

    # Rank zero receives 3 of 10 energies (indices 8, 9, 10).
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=10)
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 8 E= 0.4 0.01 IT= 1 Fe\n")
    parser.feed(" 9 E= 0.5 0.01 IT= 1 Fe\n")
    assert [e["data"] for e in events if e["kind"] == "dos_progress"] == [
        DosProgressInfo(0, 10, estimated=True),
        DosProgressInfo(3, 10, estimated=True),
    ]
    # Without known total, an observed index is not sufficient.
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e))
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 8 E= 0.4 0.01 IT= 1 Fe\n")
    assert events[-1] == {"kind": "dos_energy_index", "data": DosEnergyInfo(8, None)}
    assert parser.format_info(**events[-1]) == (
        "DOS energy index 8 observed (MPI completion unknown)"
    )


def test_mpi_parser_does_not_estimate_outside_rank_zero_block():
    from ase2sprkkr.outputs.readers.dos import DosEnergyInfo
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=200)
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 100 E= 0.4 0.01 IT= 1 Fe\n")
    assert events[-1] == {"kind": "dos_energy_index", "data": DosEnergyInfo(100, 200)}


def test_mpi_progress_uses_first_observed_index_and_checks_incomplete_log():
    """From IE=151 of NE=200, the reporting rank owns precisely 50 energies."""
    from ase2sprkkr.outputs.readers.dos import DosEnergyInfo

    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=200)
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 151 E= 0.41 0.01 IT= 1 Fe\n")
    assert parser._first_mpi_point == 151
    assert parser.total_points - parser._first_mpi_point + 1 == 50
    parser.feed(" 176 E= 0.42 0.01 IT= 1 Fe\n")
    assert events[-1]["data"] == DosProgressInfo(100, 200, estimated=True)

    # If the beginning of the reporter's block is missing, do not invent
    # a fresh 0% from IE=152 (nor pretend IE/NE means 76% finished).
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=200)
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 152 E= 0.41 0.01 IT= 1 Fe\n")
    assert parser._first_mpi_point is None
    assert events[-1] == {"kind": "dos_energy_index", "data": DosEnergyInfo(152, 200)}


def test_mpi_with_more_ranks_than_energies_uses_observed_first_point():
    """MPI_EKLOOP can group several ranks per energy; NPROCS is not NGROUPS."""
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=3)
    parser.feed("MPI calculation with NPROCS = 8\n")
    parser.feed(" 3 E= 0.43 0.01 IT= 1 Fe\n")
    assert parser._first_mpi_point == 3
    assert events[-1]["data"] == DosProgressInfo(0, 3, estimated=True)


def test_native_dos_both_printed_header_variants():
    """CALCDOS 99006 contains L=, while 99008 omits it (NWRLOG)."""
    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=3)
    parser.feed(" 1 E= 0.1234 0.0010 L= 2 IT= 1 Fe\n")
    parser.feed(" 2 E= 0.1235 0.0010          IT= 1 Fe\n")
    assert [e["data"].completed for e in events if e["kind"] == "dos_progress"] == [1]
    parser.feed(" 3 E= 0.1236 0.0010 L= 0 IT= 1 Fe\n")
    assert [e["data"].completed for e in events if e["kind"] == "dos_progress"] == [1, 2]

def test_task_independent_progress_reports_dos_mpi_estimate_and_completion():
    from ase2sprkkr.outputs.progress import ProgressInfo

    events = []
    parser = DosProgressParser(on_info=lambda **e: events.append(e), total_points=200)
    parser.feed("MPI calculation with NPROCS = 4\n")
    parser.feed(" 151 E= 0.41 0.01 IT= 1 Fe\n")
    parser.feed(" 176 E= 0.42 0.01 IT= 1 Fe\n")
    assert [e["data"] for e in events if e["kind"] == "progress"] == [
        ProgressInfo(0, 200, estimated=True), ProgressInfo(100, 200, estimated=True)
    ]
    parser.feed("program stopped regularly via STOP_REGULAR\n")
    assert [e["data"] for e in events if e["kind"] == "progress"][-1] == ProgressInfo(200, 200)
