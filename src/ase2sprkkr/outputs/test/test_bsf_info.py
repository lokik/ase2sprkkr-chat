"""INFO events for both native BSF modes; no fabricated MPI progress."""

from ase2sprkkr.outputs.readers.bsf import (
    BsfEnergyInfo,
    BsfFinishedInfo,
    BsfGridInfo,
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
    assert parser.format_info(**events[-1]).startswith("Entering k-loop for energy 151/200")
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
