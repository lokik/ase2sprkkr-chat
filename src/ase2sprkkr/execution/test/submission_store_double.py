"""Minimal submission-store boundary for lifecycle test managers.

These tests deliberately replace manager policy, but still exercise the
real RunPersistence adapter. The store double owns only the submission
hand-off protocol; it does not emulate durable file persistence.
"""

from contextlib import contextmanager

from ase2sprkkr.execution.base import RunState


class SubmissionStoreDouble:
    @contextmanager
    def submission_owner(self, record):
        yield True

    def begin_dispatch(self, record):
        assert record.state is RunState.SUBMITTING
        return record
