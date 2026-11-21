"""Native CONTROL identifiers used verbatim in filenames and shell commands.

SPR-KKR 9.7's INPUT_FIND_SECTION/STRING_GET_STRING treat whitespace, ``#``,
``=`` and collection delimiters as input syntax. TAU_DIR_INIT and
TAUIJ_DIR_INIT pass derived paths to unquoted ``mkdir``/``rm`` commands.
Reject those syntactic characters and shell expansions, while preserving
literal punctuation and directory separators in both identifiers.
"""

import pyparsing as pp

from ...common.grammar_types import String


class _FilenameString(String):
    """String's filename syntax, including native UTF-8 filenames."""

    _grammar = pp.Regex(r"[^\s,;{}\x00-\x1f\x7f]+")

    @property
    def _validate_grammar(self):
        # BaseString's class cache may already hold generic String's ASCII
        # grammar. Resolve this subclass's grammar independently of that cache.
        return self._grammar.copy().leave_whitespace() + pp.StringEnd().set_whitespace_chars("")

    def _validate(self, value, why="set"):
        if not value.isprintable():
            return "Native filenames must be nonempty and cannot contain control characters."
        return super()._validate(value, why)


class _IdentifierString(_FilenameString):
    _grammar = pp.Regex(r'''[^\s,;{}\[\]()=#|&<>$`\\"'*?\x00-\x1f\x7f]+''')


class _DatasetString(_IdentifierString):
    """A native filename prefix, optionally including directories."""

    def _validate(self, value, why="set"):
        if value.startswith("-"):
            return "DATASET must not start with a hyphen (-); SPR-KKR interprets it as a command option."
        if value.startswith("~"):
            return "DATASET must not start with a tilde (~); SPR-KKR's shell commands expand it."
        return super()._validate(value, why)


class _AdsiString(_IdentifierString):
    """A suffix appended after DATASET and an underscore; empty omits ADSI."""

    def _validate(self, value, why="set"):
        if value == "":
            return True
        return super()._validate(value, why)
