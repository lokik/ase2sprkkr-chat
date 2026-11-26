"""
Various pyparsing grammar elements and a few useful routines.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Optional

import pyparsing as pp

from .decorators import cache


_grammar_lock = Lock()
_grammar_users = 0
_grammar_defaults = None


@contextmanager
def generate_grammar():
    """Share SPRKKR parser defaults until the last grammar builder exits.

    Only entry and exit are synchronized; grammar construction can overlap.
    Other pyparsing users must not rely on independent global defaults while
    this context is active.
    """
    global _grammar_users, _grammar_defaults

    pe = pp.ParserElement
    with _grammar_lock:
        if _grammar_users == 0:
            _grammar_defaults = (pe.DEFAULT_WHITE_CHARS, pp.Keyword.DEFAULT_KEYWORD_CHARS)
            pe.set_default_whitespace_chars(" \t\r")
            if "-" not in pp.Keyword.DEFAULT_KEYWORD_CHARS:
                pp.Keyword.DEFAULT_KEYWORD_CHARS += "-"
        _grammar_users += 1
    try:
        yield
    finally:
        with _grammar_lock:
            _grammar_users -= 1
            if _grammar_users == 0:
                whitespace, keywords = _grammar_defaults
                pe.set_default_whitespace_chars(whitespace)
                pp.Keyword.DEFAULT_KEYWORD_CHARS = keywords
                _grammar_defaults = None


_grammar_build_state = ContextVar("grammar_build_state", default=None)


@contextmanager
def grammar_build():
    """Isolate temporary state used to assemble one compiled parser."""
    resets = []
    token = _grammar_build_state.set(({}, resets))
    try:
        with generate_grammar():
            yield resets
    finally:
        _grammar_build_state.reset(token)


def grammar_build_active():
    return _grammar_build_state.get() is not None


def grammar_build_state(owner):
    """Return one construction-local slot, retained only by its parse actions."""
    return _grammar_build_state.get()[0].setdefault(owner, {})


def register_grammar_reset(reset):
    """Restore one piece of mutable parser state after returning its lease."""
    _grammar_build_state.get()[1].append(reset)


class GrammarPool:
    """Lend exclusive compiled parsers; build another only during overlap."""

    def __init__(self, factory):
        self._factory = factory
        self._available = []
        self._lock = Lock()

    @contextmanager
    def acquire(self):
        with self._lock:
            entry = self._available.pop() if self._available else None
        if entry is None:
            with grammar_build() as resets:
                grammar = self._factory()
            entry = grammar, resets
        grammar, resets = entry
        try:
            yield grammar
        finally:
            for reset in resets:
                reset()
            with self._lock:
                self._available.append(entry)

    def parse_string(self, *args, **kwargs):
        with self.acquire() as grammar:
            return grammar.parse_string(*args, **kwargs)

    def parse_file(self, *args, **kwargs):
        with self.acquire() as grammar:
            return grammar.parse_file(*args, **kwargs)


def replace_whitechars(expr):
    expr = expr.copy()
    expr.set_whitespace_chars(" \t\r")
    return expr


with generate_grammar():
    optional_line_end = pp.Suppress(pp.LineEnd() | pp.WordStart()).set_name(" ")
    """ Grammar for an optinal newline """
    line_end = pp.Suppress(pp.LineEnd()).set_name("\n")
    """ Grammar for a required newline """
    end_of_file = (pp.Regex(r"[\s]*") + pp.StringEnd()).suppress().set_name("<EOF>")
    """ Grammar for an end of file (ending whitespaces are allowed) """

    optional_quote = pp.Optional("'").suppress()
    """ Grammar for an optional quote """


def as_delimiter(grammar, written=None):
    """Return a suppressed delimiter grammar named by its written form."""
    if isinstance(grammar, str):
        if written is None:
            written = grammar
        grammar = pp.Suppress(grammar) if grammar else pp.Empty()
    if written is not None:
        grammar = grammar.set_name(written)
    return grammar


def separator_pattern(char):
    return f"[{char}]" * 11 + "*"


@cache
def separator_grammar(char):
    """Pattern for separating sections in an input file"""
    separator = pp.Regex(separator_pattern(char)).set_name(f"{char * 10}[{char * 4}....]").suppress()
    return separator


def delimitedList(expr, delim):
    """Delimited list with already suppressed delimiter (or with a in-results-wanted one)"""
    return expr + pp.ZeroOrMore(delim + expr)


def add_condition_ex(self, condition, message):
    """Add check condition to the pyparsing ParseElement,
    that, if it failed, raise a parse exception with a given message."""

    def check_condition(s, loc, tocs):
        m = message
        if condition(tocs):
            return tocs
        if not isinstance(m, str):
            m = m(tocs)
        raise pp.ParseException(s, loc, m)

    self.add_parse_action(check_condition)
    return self


def add_parse_action_ex(self, pa, message=None):
    """
    Add parse action to a given pyparsing ParseElemenet,
    that, if it raise an exception, fail with a given message
    """

    def parse_action(s, loc, x):
        try:
            return pa(x)
        except Exception as e:
            if message:
                msg = f"{message}\n{e}"
            else:
                msg = str(e)
            raise pp.ParseException(s, loc, msg).with_traceback(e.__traceback__) from e

    self.add_parse_action(parse_action)


class White(pp.White):
    """Fix for whitechars in pp.White

    In Python 3.10, pyparsing.White do not respect default_white_chars.
    This class fixes this.
    """

    def __init__(self, white):
        super().__init__(white)
        self.set_whitespace_chars("")


class SkipToRegex(pp.Token):
    """Skip to given regex"""

    def __init__(self, pattern, include_pattern: Optional[bool] = None, parse_pattern: bool = False):
        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        self.pattern = pattern
        if include_pattern is None:
            include_pattern = parse_pattern
        self.include_pattern = include_pattern
        self.parse_pattern = parse_pattern
        super().__init__()

    @property
    def custom_name(self):
        self.customName = "skipTo{self.pattern}"

    def parseImpl(self, instr, loc, doActions=True):
        result = self.pattern.search(instr, loc)
        if result:
            start = result.start()
            out = instr[loc:start]
            if self.parse_pattern:
                out = (out, instr[start : result.end()])
            if self.include_pattern:
                loc = result.end()
            else:
                loc = result.start()
            return loc, pp.ParseResults(out)
        raise pp.ParseException(instr, loc, "Pattern {self.pattern} not found", self)

class Transparent(pp.ParseElementEnhance):
    """ Pyparsing ParseElementEnhance wrongly takes care about whitespaces, if there is
    nested Forward with whitespaces not already set. So we have to use out own lightweight
    replacement.
    """

    def preParse(self, instring, loc):
        return loc

    def parseImpl(self, instring, loc, do_actions=True):
        return self.expr._parse(instring, loc, do_actions)

    def __str__(self):
        return f"{self.__class__.__name__}({self.expr})"

class Suppress(Transparent):

    def __init__(self, expr):
        super().__init__(expr)
        self.add_parse_action(lambda x:[])
        self.set_name('Suppress')

class TokenConverter(Transparent):
    pass

class Forward(Transparent):

    def __init__(self, expr=None):
        super().__init__(expr)

    def __lshift__(self, other):
        self.expr = other


def dependent_forward():
    """A forward whose parse-time target must not survive its grammar lease."""
    unresolved = pp.NoMatch().set_name("<unresolved dependency>")
    forward = Forward(unresolved)
    register_grammar_reset(lambda: forward << unresolved)
    return forward


pp.ParserElement.add_condition_ex = add_condition_ex
pp.ParserElement.add_parse_action_ex = add_parse_action_ex
