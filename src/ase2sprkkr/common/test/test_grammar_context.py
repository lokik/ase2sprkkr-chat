from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pyparsing as pp
import pytest

from ..grammar import generate_grammar, GrammarPool


def parser_defaults():
    return pp.ParserElement.DEFAULT_WHITE_CHARS, pp.Keyword.DEFAULT_KEYWORD_CHARS


def test_nested_grammar_context_restores_defaults_after_exception():
    original = parser_defaults()
    with pytest.raises(ValueError, match="grammar failed"), generate_grammar():
        active = parser_defaults()
        assert active[0] == " \t\r"
        assert "-" in active[1]
        with generate_grammar():
            assert parser_defaults() == active
        assert parser_defaults() == active
        raise ValueError("grammar failed")
    assert parser_defaults() == original


def test_overlapping_grammar_contexts_restore_only_after_last_exit():
    original = parser_defaults()
    first_entered, second_entered, first_exited = Event(), Event(), Event()

    def first():
        with generate_grammar():
            first_entered.set()
            assert second_entered.wait(5)
        first_exited.set()

    def second():
        assert first_entered.wait(5)
        with generate_grammar():
            second_entered.set()
            assert first_exited.wait(5)
            # Construction still sees SPRKKR defaults after the first exit.
            assert "\n" not in pp.Word("x").whiteChars
            assert "-" in pp.Keyword("x").identChars

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(first), workers.submit(second)]
        for future in futures:
            future.result(timeout=15)
    assert parser_defaults() == original


def test_grammar_pool_reuses_parsers_after_success_and_failure():
    built = []

    def factory():
        grammar = pp.Word(pp.nums)
        built.append(grammar)
        return grammar

    pool = GrammarPool(factory)
    assert pool.parse_string("123", parse_all=True)[0] == "123"
    with pytest.raises(pp.ParseBaseException):
        pool.parse_string("bad", parse_all=True)
    assert pool.parse_string("456", parse_all=True)[0] == "456"
    with pool.acquire() as first:
        with pool.acquire() as second:
            assert first is not second
    assert len(built) == 2

    # Later short-lived threads borrow an existing parser.
    for _ in range(3):
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert workers.submit(pool.parse_string, "789").result(timeout=5)[0] == "789"
    assert len(built) == 2
