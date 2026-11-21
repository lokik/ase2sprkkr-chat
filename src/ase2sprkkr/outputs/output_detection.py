"""Fast, non-authoritative probes of SPR-KKR output file headers."""

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
_HEADER_BYTES = 8192
_TASK_SEARCH_BYTES = 256 * 1024


def looks_like_scf_output(file: str | Path) -> bool:
    """Quickly identify probable SCF outputs without invoking the parser.

    Require KKRSCF in the executable's initial VERSION banner. Check the first
    TASK marker too, which is sometimes more than 8 KiB into a native output.
    A different task rejects the file even if KKRSCF is the executable.
    For older outputs with no TASK marker, the banner is the fallback.

    Scan only as far as FILES, EOF, or 256 KiB. This is a preliminary check:
    callers must still parse and validate the output and its provenance.
    Missing or unreadable files return False.
    """
    try:
        with Path(file).open("rb") as stream:
            banner = _EXECUTABLE_BANNER.search(stream.read(_HEADER_BYTES))
            if banner is None or banner.group(1).upper() != b"KKRSCF":
                return False

            stream.seek(0)
            remaining = _TASK_SEARCH_BYTES
            while remaining > 0:
                line = stream.readline(min(remaining, 8192))
                if not line:
                    break
                remaining -= len(line)
                task = _TASK_LINE.match(line)
                if task is not None:
                    value = task.group(1).upper()
                    if value != b"NONE":
                        return value == b"SCF"
                if _FILES_LINE.match(line):
                    break
            return True
    except OSError:
        return False
