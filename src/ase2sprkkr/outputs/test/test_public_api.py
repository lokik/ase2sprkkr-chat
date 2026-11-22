"""The public result API is project based; no parallel persistent API."""

def test_project_result_helpers_are_public():
    from ase2sprkkr.outputs import (
        OutputMatch, OutputOnlyResult, ProjectResultUnavailableError, TaskResult,
        UnrecoverableProjectResultError, attach_project_run,
        load_project_result,
    )
    from ase2sprkkr.outputs.project_results import (
        ProjectResultUnavailableError as implementation_error,
        UnrecoverableProjectResultError as implementation_unrecoverable,
        attach_project_run as implementation_attach,
        load_project_result as implementation_load,
    )
    from ase2sprkkr.outputs.task_result import (
        OutputMatch as implementation_match,
        TaskResult as implementation_result,
    )

    assert ProjectResultUnavailableError is implementation_error
    assert UnrecoverableProjectResultError is implementation_unrecoverable
    assert issubclass(UnrecoverableProjectResultError, ProjectResultUnavailableError)
    assert issubclass(OutputOnlyResult, TaskResult)
    assert TaskResult is implementation_result
    assert OutputMatch is implementation_match
    assert attach_project_run is implementation_attach
    assert load_project_result is implementation_load
