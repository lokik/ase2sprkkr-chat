"""Structured scientific result files produced by SPR-KKR calculations."""

from .result_files import ResultFile
from .grace import GraceDataset, GraceFile, GraceGraph, GraceParseError, read_agr

__all__ = ["ResultFile", "GraceFile", "GraceGraph", "GraceDataset", "GraceParseError", "read_agr", "read_plottable_file"]


def read_plottable_file(filename, **kwargs):
    """Read a plottable SPR-KKR result or Grace project.

    Grace files are recognized by extension (or explicit try_only="agr");
    structured result files retain their normal registry-based detection.
    """
    import os

    path = getattr(filename, "name", filename)
    try:
        is_grace = os.path.splitext(os.fspath(path))[1].lower() == ".agr"
    except TypeError:
        is_grace = False
    if is_grace or kwargs.get("try_only") == "agr":
        return read_agr(filename)
    return ResultFile.from_file(filename, **kwargs)
