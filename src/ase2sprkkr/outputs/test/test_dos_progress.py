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
        {"kind": "dos_progress", "data": DosProgressInfo(13, 300)},
        {"kind": "dos_progress", "data": DosProgressInfo(14, 300)},
    ]
    assert parser.format_info(**events[-1]) == "14/300 done."


def test_dos_does_not_invent_progress_from_fractions_or_unfinished_points():
    events = []
    parser = DosProgressParser(on_info=lambda **event: events.append(event))
    parser.feed("ENERGY NE=300\n13/300\n1/0 done.\n301/300 done.\n")
    assert events == []
    parser.feed("12/300 done.\n12/300 done.\n")
    assert events == [{"kind": "dos_progress", "data": DosProgressInfo(12, 300)}]


def test_dos_progress_prints_the_same_info_as_callback(capsys):
    events = []
    parser = DosProgressParser(
        print_info=True, on_info=lambda **event: events.append(event)
    )
    parser.feed("DOS energy points: 3 of 20 processed\n")
    assert events == [{"kind": "dos_progress", "data": DosProgressInfo(3, 20)}]
    assert capsys.readouterr().out == "3/20 done.\n"


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
    assert [event["data"].completed for event in events] == [13, 14]
