from ..task_definition import TaskDefinition, TaskFile


def task_definition():
    return TaskDefinition(
        "BSF",
        supports={"3d", "2d"},
        name="BSFEK",
        executable="kkrgen",
        mpi=True,
        output_reader="BSF",
        result_files=("{DATASET_ADSI}.bsf", TaskFile("{DATASET_ADSI}_spol.bsf", optional=True)),
    )
