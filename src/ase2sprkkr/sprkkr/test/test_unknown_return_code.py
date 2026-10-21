"""An unknown exit code must remain unknown after SPR-KKR postprocessing.

Use a fake raw executor rather than performing file serialization or invoking
an SPR-KKR executable to check this lightweight adapter contract.
"""

from types import SimpleNamespace

from ...execution.base import ExecutionOutcome, ExecutionSpec, RunState
from .. import calculation


def test_finished_execution_passes_unknown_return_code_to_reader(tmp_path):
    calls = []
    closed = []

    def read_from_file(output, error, *, return_code, input_file):
        calls.append((output, error, return_code, input_file))
        return SimpleNamespace(return_code=return_code)

    reader = SimpleNamespace(read_from_file=read_from_file)
    task = SimpleNamespace(
        executable="unused", mpi=False,
        output_reader=lambda **kwargs: reader,
    )
    input_parameters = SimpleNamespace(
        task_definition_required=lambda: task,
        input_files=lambda directory: (),
        output_files=lambda directory: (),
        cache_files=lambda directory: (),
    )
    input_path = str(tmp_path / "input.inp")
    output_path = str(tmp_path / "output.out")
    prepared = SimpleNamespace(
        input_parameters=input_parameters,
        input_file=input_path,
        output_file=output_path,
        potential_file=None,
        directory=str(tmp_path),
        close=lambda: closed.append(True),
    )

    def create_execution(command, input_file, output_file, directory, **kwargs):
        return ExecutionSpec(
            argv=(command,),
            input_file=input_file,
            output_file=output_file,
            error_file=kwargs["error_file"],
            directory=directory,
        )

    profile = SimpleNamespace(create_execution=create_execution)

    class FinishedRunner:
        def __init__(self, spec):
            self.spec = spec
            self.state = RunState.NEW
            self.exception = None
            self.result = ExecutionOutcome(
                input_file=input_path, output_file=output_path, return_code=None,
            )

        def submit(self):
            self.state = RunState.RUNNING

        def wait(self):
            self.state = RunState.FINISHED
            return self.result

    executor = SimpleNamespace(create_run=lambda spec: FinishedRunner(spec))
    runner = calculation._create_sprkkr_run(
        prepared,
        result_context=None,
        profile=profile,
        executor=executor,
        error_file=True,
    )

    result = runner.run()

    assert result.return_code is None
    assert calls == [(output_path, None, None, input_path)]
    assert closed == [True]
