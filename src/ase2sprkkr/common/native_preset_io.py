"""Context-local native input read/write policies for incomplete value presets.

Normal SPR-KKR input parsing and serialization are unaffected outside these contexts.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_write_explicit = ContextVar("ase2sprkkr_preset_write_explicit", default=False)
_read_incomplete = ContextVar("ase2sprkkr_preset_read_incomplete", default=False)


def writing_preset():
    return _write_explicit.get()


def reading_preset():
    return _read_incomplete.get()


@contextmanager
def preset_write_context():
    token = _write_explicit.set(True)
    try:
        yield
    finally:
        _write_explicit.reset(token)


@contextmanager
def preset_read_context():
    token = _read_incomplete.set(True)
    try:
        yield
    finally:
        _read_incomplete.reset(token)
