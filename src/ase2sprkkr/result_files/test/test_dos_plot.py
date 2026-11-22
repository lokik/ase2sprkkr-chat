"""Plot regressions for native DOS files with variable column counts."""

from pathlib import Path

import pytest

from .. import ResultFile


@pytest.mark.slow
@pytest.mark.parametrize(
    ("filename", "shape"),
    (
        ("Mo2Se4_SCF_SCF_SCFSTART.dos", (65, 20)),
        ("TaAs_DOS.dos", (1200, 20)),
    ),
)
def test_multitype_dos_plot(filename, shape, tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / filename
    result = ResultFile.from_file(source, try_only="dos", unknown=False)
    assert result.RAW_DATA().shape == shape

    destination = tmp_path / "dos.png"
    result.plot(filename=str(destination), show=False, dpi=72)
    assert destination.is_file()
    assert destination.stat().st_size > 0
