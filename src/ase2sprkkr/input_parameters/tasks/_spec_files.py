"""Shared declarative native KKRSPEC file specifications.

Each callback returns a filename or None when the file is not used.
"""


from kkrunner import TaskFile


def input_inp(parameters):
    return "input.inp" if parameters.TASK.INPVER() == 0 else None


def struc_inp(parameters):
    return "struc.inp" if parameters.TASK.STRVER() == 0 else None


def _single_polarization(parameters):
    # BAND/SPLEED never enter the photon-polarization output branch.
    if parameters.TASK.TASK() in ("BAND", "SPLEED"):
        return True

    # NPOL selects the actual number of native outputs; POL_P specifies
    # the polarization orientation, not the number of calculated channels.
    # NPOL=0 (unpolarized/dichroism) and NPOL=3 (dichroism) use two.
    return str(parameters.SPEC_PH.NPOL()) not in ("0", "3")


def data_spc(parameters):
    if parameters.TASK.TASK() == "SPLEED":
        # Native SPLEED always opens the SPC (sometimes header-only).
        return "{DATASET_ADSI}_data.spc"
    return "{DATASET_ADSI}_data.spc" if _single_polarization(parameters) else None


def data_pol1_spc(parameters):
    return None if _single_polarization(parameters) else "{DATASET_ADSI}_data_pol1.spc"


def data_pol2_spc(parameters):
    return None if _single_polarization(parameters) else "{DATASET_ADSI}_data_pol2.spc"


SPEC_INPUT_FILES = (input_inp, struc_inp)
SPEC_RESULT_FILES = (data_spc, data_pol1_spc, data_pol2_spc)

# Native logs and reserved-but-uncollectable generic DOS headers are not
# user-facing spectra. They still participate in the ordinary output manifest.
SPEC_AUX_OUTPUT_FILES = (
    "{DATASET_ADSI}_SPEC.out",
    TaskFile("{DATASET_ADSI}.dos", collect=False),
)

# Historical combined declaration retained for consumers that need all files.
# It is assembled from the same objects rather than independently maintained.
SPEC_OUTPUT_FILES = (
    SPEC_AUX_OUTPUT_FILES[0],
    *SPEC_RESULT_FILES,
    SPEC_AUX_OUTPUT_FILES[1],
)
