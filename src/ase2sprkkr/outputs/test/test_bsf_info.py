"""INFO events for both native BSF modes; no fabricated MPI progress."""

from ase2sprkkr.outputs.readers.bsf import (
    BsfEnergyInfo,
    BsfFinishedInfo,
    BsfGridInfo,
    BsfRankProgressInfo,
    BsfProgressParser,
)


def _collect():
    events = []
    return BsfProgressParser(on_info=lambda **event: events.append(event)), events


def test_bsf_ek_native_energy_indices_are_milestones_not_completion_counts():
    parser, events = _collect()
    parser.feed("E(k)-relation mode for:\nwith  NE =  200  NK =  297\n")
    assert events[:2] == [
        {"kind": "bsf_mode", "data": "EK"},
        {"kind": "bsf_grid", "data": BsfGridInfo("EK", 200, 297)},
    ]

    # The sample native MPI stdout has ONLY IE=151..200, not IE=1..150.
    line = "    IE= 151  E=   -0.8079    0.0010    all matrices set up  -  enter k-loop\n"
    parser.feed(line[:23])
    assert len(events) == 2
    parser.feed(line[23:])
    assert events[-1] == {
        "kind": "bsf_energy",
        "data": BsfEnergyInfo(151, -0.8079, 0.001, 200),
    }
    assert parser.format_info(**events[-1]).startswith("Entering k-loop for energy index 151 (NE=200)")
    assert "completed" not in parser.format_info(**events[-1]).lower()
    assert not any(event["kind"] == "bsf_finished" for event in events)

    parser.feed(
        "    IE= 200  E=   -1.0000    0.0010    all matrices set up  -  enter k-loop\n"
        "         finished in subroutine <BLOCHSF>\n"
    )
    assert events[-1] == {
        "kind": "bsf_finished",
        "data": BsfFinishedInfo(0),
    }


def test_bsf_kk_has_grid_fixed_energy_and_warnings_but_no_kpoint_progress():
    parser, events = _collect()
    parser.feed(
        "constant E mode for:\n"
        "         fixed energy         0.125439  0.000000\n"
        "         k-space parameters:\n"
        "         NK1 =   60    K1 =  (   1.000   0.000   0.000 )\n"
        "         NK2 =   60    K2 =  (   0.000   1.000   0.000 )\n"
        "    IE=   1  E=    0.1254    0.0000    all matrices set up  -  enter k-loop\n"
    )
    assert any(event == {"kind": "bsf_grid", "data": BsfGridInfo("KK", nk1=60, nk2=60)}
               for event in events)
    assert any(event == {"kind": "bsf_fixed_energy", "data": 0.125439}
               for event in events)
    energies = [event["data"] for event in events if event["kind"] == "bsf_energy"]
    assert energies == [BsfEnergyInfo(1, 0.1254, 0.0, None)]

    # Native output may contain thousands of identical diagnostics.
    parser.feed("WARNING <BLOCHBSF>: Spectral function negative\n" * 1461)
    assert parser.negative_warnings == 1461
    counts = [event["data"] for event in events if event["kind"] == "bsf_negative_warnings"]
    assert counts == [1, 10, 100, 1000]
    assert len([event for event in events if event["kind"] == "bsf_energy"]) == 1

    parser.feed("         finished in subroutine <BLOCHSF>\n")
    assert events[-1] == {"kind": "bsf_finished", "data": BsfFinishedInfo(1461)}
    assert "1,461" in parser.format_info(**events[-1])
    assert not any(event["kind"] == "bsf_progress" for event in events)


def test_bsf_info_parser_handles_partial_last_line_and_d_exponents():
    parser, events = _collect()
    parser.feed("constant E mode for:\nIE= 1 E= 1.0D-01 2.0D-03 all matrices set up - enter k-loop")
    assert not any(event["kind"] == "bsf_energy" for event in events)
    parser.finish()
    assert events[-1]["data"] == BsfEnergyInfo(1, 0.1, 0.002, None)


def test_bsf_ek_local_mpi_rank_progress_from_partitioned_energy_indices():
    parser, events = _collect()
    parser.feed(
        "MPI calculation with NPROCS = 4\n"
        "E(k)-relation mode for:\n"
        "with  NE =  200  NK =  297\n"
    )
    parser.feed("IE= 151 E= -0.8079 0.0010 all matrices set up - enter k-loop\n")
    progress = [event["data"] for event in events if event["kind"] == "bsf_rank_progress"]
    assert progress == [BsfRankProgressInfo(1, 50, 151, 151)]
    assert progress[-1].completed == 0

    parser.feed("IE= 175 E= -0.9020 0.0010 all matrices set up - enter k-loop\n")
    progress = [event["data"] for event in events if event["kind"] == "bsf_rank_progress"]
    assert progress[-1] == BsfRankProgressInfo(25, 50, 151, 175)
    assert progress[-1].completed == 24

    parser.feed("IE= 200 E= -1.0000 0.0010 all matrices set up - enter k-loop\n")
    progress = [event["data"] for event in events if event["kind"] == "bsf_rank_progress"]
    assert progress[-1] == BsfRankProgressInfo(50, 50, 151, 200)
    assert progress[-1].completed == 49
    assert parser.format_info("bsf_rank_progress", progress[-1]).startswith(
        "Reporting MPI rank"
    )
    # Native completion is a distinct event; entering the last k-loop
    # does not justify claiming it has finished.
    assert not any(event["kind"] == "bsf_finished" for event in events)
    parser.feed("finished in subroutine <BLOCHSF>\n")
    assert events[-1]["kind"] == "bsf_finished"


def test_bsf_ek_single_process_uses_whole_energy_range():
    parser, events = _collect()
    parser.feed(
        "MPI calculation with NPROCS = 1\n"
        "E(k)-relation mode for:\n"
        "with  NE =  3  NK =  20\n"
        "IE= 1 E= -0.2 0.0010 all matrices set up - enter k-loop\n"
        "IE= 2 E= -0.1 0.0010 all matrices set up - enter k-loop\n"
    )
    local = [x["data"] for x in events if x["kind"] == "bsf_rank_progress"]
    assert local == [
        BsfRankProgressInfo(1, 3, 1, 1),
        BsfRankProgressInfo(2, 3, 1, 2),
    ]



def test_native_bsf_photon_shift_two_windows_reset_rank_progress():
    """BLOCHSF EPHOT>0 writes 99002 energy loop twice."""
    parser, events = _collect()
    parser.feed(
        "MPI calculation with NPROCS = 2\n"
        "E(k)-relation mode for:\n"
        "with  NE =  4  NK =  30\n"
        "energy loop   1\n"
        "IE= 3 E= 0.2000 0.0010 all matrices set up - enter k-loop\n"
        "energy loop   2\n"
        "IE= 3 E= 0.3000 0.0010 all matrices set up - enter k-loop\n"
    )
    assert [event["data"] for event in events
            if event["kind"] == "bsf_energy_window"] == [1, 2]
    rank = [event["data"] for event in events
            if event["kind"] == "bsf_rank_progress"]
    assert rank == [BsfRankProgressInfo(1, 2, 3, 3)] * 2
    energy = [event["data"] for event in events if event["kind"] == "bsf_energy"]
    assert len(energy) == 2
    assert energy[0].real_ry == 0.2
    assert energy[1].real_ry == 0.3
    assert not any(event["kind"] == "bsf_finished" for event in events)


def test_native_bsf_outer_row_marker_precedes_processing():
    """BLOCHSF writes 'k-points done' before its inner row is evaluated."""
    parser, events = _collect()
    parser.feed(
        "constant E mode for:\n"
        "NK1 = 200 K1 = (1.0 0.0 0.0)\n"
        "NK2 = 100 K2 = (0.0 1.0 0.0)\n"
        "outer k-loop:   100 k-points done\n"
        "inner k-loop:     1 k-points done\n"
    )
    assert ("bsf_outer_krow", 100) in [
        (event["kind"], event["data"]) for event in events
    ]
    assert "not yet complete" in parser.format_info("bsf_outer_krow", 100)
    assert not any(event["kind"] == "bsf_rank_progress" for event in events)


def test_native_bsf_e_plots_uses_ek_header_when_one_k_vector():
    """BLOCHSF MODE='E-plots' when NK1=1 and NK2<=2."""
    parser, events = _collect()
    parser.feed("E(k)-relation mode for:\nwith  NE =  200  NK =    1\n")
    assert ("bsf_grid", BsfGridInfo("EK", 200, 1)) in [
        (event["kind"], event["data"]) for event in events
    ]

def test_task_independent_progress_counts_local_bsf_rank_until_native_finish():
    from ase2sprkkr.outputs.progress import ProgressInfo

    parser, events = _collect()
    parser.feed(
        "MPI calculation with NPROCS = 4\n"
        "E(k)-relation mode for:\nwith NE = 200 NK = 297\n"
        "IE= 151 E= -0.8 0.001 all matrices set up - enter k-loop\n"
        "IE= 175 E= -0.9 0.001 all matrices set up - enter k-loop\n"
    )
    assert [e["data"] for e in events if e["kind"] == "progress"] == [
        ProgressInfo(0, 50, label="Reporting MPI rank: energy loops"),
        ProgressInfo(24, 50, label="Reporting MPI rank: energy loops"),
    ]
    parser.feed("finished in subroutine <BLOCHSF>\n")
    assert [e["data"] for e in events if e["kind"] == "progress"][-1] == ProgressInfo(
        50, 50, label="Reporting MPI rank: energy loops"
    )


def test_task_independent_bsf_progress_is_absent_for_kk_without_counter():
    parser, events = _collect()
    parser.feed(
        "constant E mode for:\nNK1 = 60 K1 = (1 0 0)\n"
        "NK2 = 60 K2 = (0 1 0)\n"
        "IE= 1 E= 0.12 0.001 all matrices set up - enter k-loop\n"
        "finished in subroutine <BLOCHSF>\n"
    )
    assert not any(e["kind"] == "progress" for e in events)
