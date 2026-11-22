"""Live SCF progress uses the full parser's iteration formatting."""

from pathlib import Path

from ase2sprkkr.outputs.readers.scf import (
    ScfOutputParser, ScfProgressParser, ScfIterationInfo, ScfToleranceInfo, ScfResult
)
from ase2sprkkr.outputs.task_result import TaskResult
from ase2sprkkr.common.formats import fortran_format


EXAMPLE = Path(__file__).parent.parent / "examples" / "scf.out"


def test_latest_scf_iteration_matches_complete_parser():
    result = TaskResult.from_file(EXAMPLE)
    latest = ScfProgressParser.latest_from_file(EXAMPLE)
    last = result.last_iteration
    assert latest["iteration"] == last.iteration()
    assert latest["error"] == last.error()
    assert latest["moment"]["spin"] == last.moment.spin()
    assert latest["converged"] == last.converged()


def test_scf_progress_ignores_incomplete_iteration():
    output = []
    parser = ScfProgressParser(on_info=lambda **event: output.append(event))
    header = "  8 ERR 0.123E-04 0.456E-06 EF 0.6 0.01 D 12.00 M 1.2 0.3\n"
    footer = "    ETOT    -42.5    SCF - cycle not converged - continue\n"
    parser.feed(header[:20])
    assert not output
    parser.feed(header[20:] + footer[:20])
    assert not output
    parser.feed(footer[20:])
    assert len(output) == 1
    assert output[0]["kind"] == "scf_iteration"
    assert isinstance(output[0]["data"], ScfIterationInfo)
    assert output[0]["data"].iteration == 8
    assert output[0]["data"].convergence_error == 0.0000123
    assert output[0]["data"].error == 0.0000123
    assert output[0]["data"].b_error == 0.000000456
    formatted = ScfOutputParser.format_info(**output[0])
    assert formatted.startswith("Iteration")
    assert "RMS(V)" in formatted and "RMS(B)" in formatted
    assert f"RMS(V) {fortran_format(0.0000123, ':>12e')}" in formatted
    assert f"RMS(B) {fortran_format(0.000000456, ':>12e')}" in formatted
    parser.feed("  9 ERR 0.111E-05 0.000E+00 EF 0.6 0.0 D 11 M 1.2 0.3\n")
    assert parser.last_iteration["iteration"] == 8


def test_scf_progress_tail_ignores_partial_cycle(tmp_path):
    content = ("  8 ERR 0.123E-04 0.456E-06 EF 0.6 0.01 D 12 M 1.2 0.3\n"
               " ETOT -42.5 SCF - cycle not converged - continue\n"
               "  9 ERR 0.222E-04 0.333E-06 EF 0.6 0.01 D 12 M 1.2 0.3\n")
    path = tmp_path / "unfinished.out"
    path.write_text(content)
    assert ScfProgressParser.latest_from_file(path)["iteration"] == 8



def test_tolerance_is_emitted_once_and_separately_from_iterations():
    events = []
    parser = ScfProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("  SCF-tol : 0.00001\n")
    parser.feed("  SCF-tol : 0.00001\n")
    parser.feed("  1 ERR 0.1E-03 0.2E-03 EF 0.7 0 D 0 M 1 2\n")
    parser.feed("  ETOT -42.5 SCF - cycle not converged - continue\n")
    assert [event["kind"] for event in events] == ["scf_tolerance", "scf_iteration"]
    assert events[0]["data"] == ScfToleranceInfo(1e-5)
    assert events[1]["data"].convergence_error == 2e-4


def test_full_parser_emits_typed_info_and_retains_native_tolerance():
    events = []
    result = ScfResult(None, None, EXAMPLE.parent)
    parser = ScfOutputParser(on_info=lambda **event: events.append(event))
    parser.read_from_file(EXAMPLE.parent / "Al_SCF.out", read_args=[result])
    tolerance = [e["data"].tolerance for e in events if e["kind"] == "scf_tolerance"]
    iterations = [e["data"] for e in events if e["kind"] == "scf_iteration"]
    assert tolerance == [1e-5]
    assert result.scf_tolerance == 1e-5
    assert len(iterations) == 1
    assert iterations[0].converged
    assert events[-1] == {"kind": "scf_finished", "data": True}



def test_print_info_formats_the_same_events_as_callback(capsys):
    events = []
    parser = ScfProgressParser(
        print_info=True, on_info=lambda **event: events.append(event),
    )
    parser.feed(" SCF-tol: 1.0D-5\n")
    parser.feed(" 1 ERR 0.1E-03 0.2E-03 EF 0.7 0 D 0 M 1 2\n")
    parser.feed(" ETOT -42.5 SCF - cycle not converged - continue\n")
    output = capsys.readouterr().out.splitlines()
    assert len(events) == len(output) == 2
    assert output[0] == "SCF tolerance: 1e-05"
    assert output[1] == ScfOutputParser.format_info(**events[1])
    assert "RMS(V)" in output[1]
    assert "RMS(B)" in output[1]



def test_native_scf_all_three_iteration_footers_and_qmvec_branch():
    """scf.f 2321-2372 writes 99011 for converged/continue/exhausted."""
    headers = [
        "  1 ERR 0.300E-04 0.200E-04 EF 0.6 0.01 D 12.0 M 1.2 0.3\n",
        "  2 ERR 0.100E-05 0.200E-05 EF 0.6 0.01 D 12.0 M 1.2 0.3\n",
        "  3 ERR 0.100E-07 0.200E-07 EF 0.6 0.01 D 12.0 M 1.2 0.3\n",
    ]
    footers = [
        " ETOT    -42.5    SCF - cycle not converged  -  continue\n",
        " ETOT    -42.6    SCF - cycle not converged - NSCFITER exhausted\n",
        " ETOT    -42.7    SCF - cycle converged !!!!!!!!!\n",
    ]
    events = []
    parser = ScfProgressParser(on_info=lambda **e: events.append(e))
    parser.feed(" SCF-tol : 0.0000100000\n")
    for header, footer in zip(headers, footers):
        parser.feed(header[:18])
        parser.feed(header[18:] + footer)
    parser.feed(" 1 # QMVEC ETOT MSPIN SPIRAL\n")
    cycles = [e["data"] for e in events if e["kind"] == "scf_iteration"]
    assert len(cycles) == 3
    assert [cycle.converged for cycle in cycles] == [False, False, True]
    assert cycles[0].iteration == 1
    assert cycles[-1].iteration == 3
    assert len([e for e in events if e["kind"] == "scf_tolerance"]) == 1


def test_native_scf_final_exhausted_is_not_converged():
    """Native STOP_REGULAR can follow NSCFITER exhaustion without convergence."""
    parser = ScfProgressParser()
    parser.feed(
        " 10 ERR 0.100E-02 0.200E-02 EF 0.7 0.01 D 12 M 1 0\n"
        " ETOT -42.5 SCF - cycle not converged - NSCFITER exhausted\n"
    )
    assert parser.last_iteration["iteration"] == 10
    assert parser.last_iteration["converged"] is False
