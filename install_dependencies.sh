#!/bin/bash
set -e

PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if (( PYTHON_MINOR >= 14 )); then
    echo "Python >= 3.14 detected. Installing monty 2025.3.3 ignoring Python version metadata..."
    pip install --ignore-requires-python monty==2025.3.3 pymatgen==2025.10.7
    # The fork carries compatibility changes needed by current Python releases.
    python3 -m pip install git+https://github.com/lokik/spglib.git
fi

# Integration and prerelease CI use the matching KKRunner source tree.
# Production installs retain the declared PyPI dependency.
if [[ -n "${ASE2SPRKKR_KKRUNNER_SOURCE:-}" ]]; then
    python3 -m pip install "${ASE2SPRKKR_KKRUNNER_SOURCE}[psij]"
fi

python3 -m pip install .
