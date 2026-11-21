"""Fast, non-authoritative probes of SPR-KKR output file headers."""

import re
from pathlib import Path


_SCF_BANNER = re.compile(rb"\bKKRSCF\b")
_HEADER_BYTES = 8192


def looks_like_scf_output(file: str | Path) -> bool:
    """Check for the KKRSCF executable banner without parsing the output.

    Only the first 8 KiB are read, irrespective of the SPR-KKR version.
    This is a cheap preliminary check, not proof of an SCF result: callers
    needing provenance must still parse and validate the candidate.
    Unreadable/missing files return False.
    """
    try:
        with Path(file).open("rb") as stream:
            return _SCF_BANNER.search(stream.read(_HEADER_BYTES)) is not None
    except OSError:
        return False
