"""The public result API is project based; no parallel persistent API."""

def test_project_result_helpers_are_public():
    from ase2sprkkr.outputs import (
        OutputOnlyResult, ProjectResultUnavailableError, TaskResult,
        UnrecoverableProjectResultError, attach_project_run,
        load_project_result,
    )
    from ase2sprkkr.outputs.project_results import (
        ProjectResultUnavailableError as implementation_error,
        UnrecoverableProjectResultError as implementation_unrecoverable,
        attach_project_run as implementation_attach,
        load_project_result as implementation_load,
    )
    from ase2sprkkr.outputs.task_result import TaskResult as implementation_result

    assert ProjectResultUnavailableError is implementation_error
    assert UnrecoverableProjectResultError is implementation_unrecoverable
    assert issubclass(UnrecoverableProjectResultError, ProjectResultUnavailableError)
    assert issubclass(OutputOnlyResult, TaskResult)
    assert TaskResult is implementation_result
    assert attach_project_run is implementation_attach
    assert load_project_result is implementation_load


def test_file_attribution_is_public_without_old_association_alias():
    import ase2sprkkr
    import ase2sprkkr.outputs as outputs

    assert ase2sprkkr.FileAttribution is outputs.FileAttribution
    assert set(outputs.FileAttribution.__members__) == {
        "UNKNOWN", "NOT_THIS_RESULT", "LIKELY", "VERIFIED",
    }
    assert not hasattr(ase2sprkkr, "FileAssociation")
    assert not hasattr(outputs, "FileAssociation")
