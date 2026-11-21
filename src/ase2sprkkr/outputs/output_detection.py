"""Fast, non-authoritative probes of SPR-KKR output file headers."""

from __future__ import annotations

import os
import re
from pathlib import Path


# Only the program banner counts, not an incidental mention of KKRSCF
# elsewhere in a KKRGEN (DOS/BSF/...) output. Prefer the *first* banner.
_EXECUTABLE_BANNER = re.compile(
    rb"(?m)^[ \t]*\*[ \t]*(KKR[A-Z0-9_]+)[ \t]+VERSION\b",
    re.IGNORECASE,
)
_TASK_LINE = re.compile(
    rb"^[ \t]*TASK[ \t]*=[ \t]*([A-Z][A-Z0-9_]*)[ \t]*(?:\r?\n)?$",
    re.IGNORECASE,
)
_FILES_LINE = re.compile(rb"^[ \t]*FILES:[ \t]*(?:\r?\n)?$", re.IGNORECASE)
# The SCF-cycle "pot-file:" entry names the produced potential; FILES/potential
# instead refers to the input POTFIL.
_SCF_POT_FILE = re.compile(rb"^[ \t]*pot-file[ \t]*:[ \t]*(.+?)[ \t]*(?:\r?\n)?$", re.IGNORECASE)
_HEADER_BYTES = 8192
_TASK_SEARCH_BYTES = 256 * 1024


def looks_like_scf_output(
    file: str | Path, potential_filename: str | Path | None = None,
) -> bool:
    """Quickly identify probable SCF outputs, optionally checking their potential.

    Require KKRSCF in the executable's initial VERSION banner. The first
    explicit TASK other than SCF rejects the file (TASK=NONE is ignored).

    When `potential_filename` is provided, compare it against the SCF-cycle
    `pot-file:` line, which names the *produced* potential. The FILES section's
    `potential:` entry instead names the input and must not be compared.

    A missing `pot-file:` is inconclusive: keep the file as a candidate so
    the full SCF parser can verify it. This remains a non-authoritative probe.
    Scan no more than 256 KiB, and never read whole SCF iterations.
    """
    path = Path(file)
    expected = (
        Path(potential_filename).resolve()
        if potential_filename is not None else None
    )
    try:
        with path.open("rb") as stream:
            banner = _EXECUTABLE_BANNER.search(stream.read(_HEADER_BYTES))
            if banner is None or banner.group(1).upper() != b"KKRSCF":
                return False

            stream.seek(0)
            remaining = _TASK_SEARCH_BYTES
            # Continue past FILES only if a potential was requested: native
            # SPR-KKR writes the produced pot-file line after FILES.
            while remaining > 0:
                line = stream.readline(min(remaining, 8192))
                if not line:
                    break
                remaining -= len(line)
                task = _TASK_LINE.match(line)
                if task is not None:
                    value = task.group(1).upper()
                    if value != b"NONE":
                        if value != b"SCF":
                            return False
                        if expected is None:
                            return True
                if expected is None:
                    if _FILES_LINE.match(line):
                        break
                    continue

                found = _SCF_POT_FILE.match(line)
                if found is not None:
                    name = os.fsdecode(found.group(1)).strip()
                    if not name:
                        continue
                    if len(name) >= 2 and name[0] == name[-1] and name[0] in ('"', "'"):
                        name = name[1:-1]
                    produced = Path(name)
                    if not produced.is_absolute():
                        produced = path.resolve().parent / produced
                    return produced.resolve() == expected
            return True
    except (OSError, ValueError):
        return False
