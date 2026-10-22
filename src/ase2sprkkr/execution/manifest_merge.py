"""Pure transformations of immutable file manifests."""
from __future__ import annotations
import os
from dataclasses import replace
from .run_model import FileManifest


def _merge_availability(previous, received, preserve):
    """Reconcile matching files without altering the received membership.

    FileManifest already rejects output/cache identity collisions.
    """
    # Remote-only caches are included even though FileManifest.files
    # deliberately omits them. Merged streams share one physical identity.
    recorded = {}
    priority = {"failed": 3, "available": 2, "skipped": 1, "pending": 0, None: 0}
    for entry in (previous.stdout, previous.stderr,
                  *previous.outputs, *previous.caches):
        if entry is None:
            continue
        identity = (entry.local_path, entry.remote_path)
        prior = recorded.get(identity)
        if (prior is None or
                priority[entry.availability_state] > priority[prior.availability_state]):
            recorded[identity] = entry

    def merged(entry):
        if entry is None:
            return None
        prior = recorded.get((entry.local_path, entry.remote_path))
        if prior is None or not preserve(prior, entry):
            return entry
        if (entry.availability_state == prior.availability_state
                and entry.availability_message == prior.availability_message):
            return entry
        return replace(
            entry, availability_state=prior.availability_state,
            availability_message=prior.availability_message,
            verified_local_version=prior.verified_local_version,
        )

    return received._map_files(merged)

def discard_capture(manifest):
    """Invalidate speculative captured versions, preserving file statuses."""
    return replace(manifest, captured_at=None, input_versions=(), output_versions=())

def merge_snapshot(current, requested):
    """Do not let stale observations erase independently persisted delivery."""
    if isinstance(requested, dict):
        requested = FileManifest(**requested)
    if current is None:
        return requested
    if current.captured_at is not None:
        if requested != current:
            raise ValueError("Cannot replace a captured file manifest")
        return current

    def preserve(prior, entry):
        if prior.availability_state not in ("available", "failed", "skipped"):
            return False
        if entry.availability_state == "pending":
            return True
        if (prior.availability_state == "available"
                and entry.availability_state == "failed"
                and prior.verified_local_version is not None
                and prior.verified_local_version.is_current()):
            return True
        # A concurrent failed transfer invalidates speculative capture.
        return (
            requested.captured_at is not None
            and entry.availability_state in ("available", "skipped")
            and prior.availability_state == "failed"
        )

    return _merge_availability(current, requested, preserve)

def merge_terminal_files(previous, received):
    """Retain only still-present local deliveries in terminal facts."""
    if previous is None:
        return received

    def preserve(prior, entry):
        if entry.availability_state in ("failed", "skipped"):
            return (
                prior.availability_state == "available"
                and prior.verified_local_version is not None
                and prior.verified_local_version.is_current()
            )
        if prior.availability_state == "skipped":
            return True
        return (
            prior.availability_state == "available"
            and entry.local_path is not None
            and os.path.isfile(entry.local_path)
        )

    return _merge_availability(previous, received, preserve)

