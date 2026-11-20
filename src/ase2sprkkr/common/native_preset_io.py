"""Context-local parsing policy for importing incomplete native input as a preset.

Normal SPR-KKR input parsing is unaffected outside this context.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_read_incomplete = ContextVar("ase2sprkkr_preset_read_incomplete", default=False)


def reading_preset():
    return _read_incomplete.get()


@contextmanager
def preset_read_context():
    token = _read_incomplete.set(True)
    try:
        yield
    finally:
        _read_incomplete.reset(token)
