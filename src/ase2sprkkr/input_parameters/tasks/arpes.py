from ..task_definition import TaskDefinition
from ._spec_files import SPEC_INPUT_FILES, SPEC_OUTPUT_FILES, SPEC_RESULT_FILES


def task_definition():
    return TaskDefinition(
        "ARPES",
        supports={"3d", "2d"},
        name="ARPES",
        executable="kkrspec",
        mpi=False,
        input_files=SPEC_INPUT_FILES,
        output_files=SPEC_OUTPUT_FILES, result_files=SPEC_RESULT_FILES,
        output_reader="ARPES",
    )
