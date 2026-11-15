#!/usr/bin/env python
"""
This is a script to run calculation from commandline. Run it to see the doc.
"""

import argparse

from ase2sprkkr_cli.errors import CLIUsageError




help = "Run SPRKKR calculation."
description = "Run a calculation with optional NAME=VALUE input parameters."
unknowns = "options"  # Allow positional overrides after CLI flags across Python versions.


def parser(parser):
    bool_arg = getattr(argparse, "BooleanOptionalAction", "store_true")
    parser.add_argument("pot", help="SPR-KKR potential file or input file. The type is determined by its content.")

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "-t",
        "--task",
        dest="task",
        type=str,
        help="Task to compute (see 'ase2sprkkr info --task').",
    )
    grp.add_argument(
        "--input-file",
        "-i",
        help="Use this SPR-KKR input file with the supplied potential.",
        type=str,
    )

    parser.add_argument("--profile", dest="execution_profile", help="Named execution profile to use for the calculation.")
    parser.add_argument("--project-id", help="Existing project UUID for a persistent calculation.")
    parser.add_argument("--print-output", "-O", help="Print the output of SPRKKR.", default="None", action=bool_arg)
    parser.add_argument("--output-file", "-o", help="Output file", type=str)
    parser.add_argument(
        "--empty-spheres",
        "-e",
        help="Control automatic addition of empty spheres.",
        action=bool_arg,
    )
    parser.add_argument(
        "options", nargs="*", help="Input parameters in the form <name>=<value>.", type=str
    )


def run(args, global_args):
    from ase2sprkkr import InputParameters
    from ase2sprkkr.common.module_registry import RegistryEntryNotFound
    from ase2sprkkr.common.tools import parse_named_option
    import pyparsing

    if args.task is not None:
        try:
            args.task = InputParameters.task_registry[args.task].name.lower()
        except RegistryEntryNotFound:
            raise CLIUsageError(f"Unknown task {args.task!r}") from None
    try:
        options = dict(parse_named_option(option) for option in args.options)
    except (ValueError, pyparsing.ParseBaseException) as exc:
        raise CLIUsageError(f"Invalid input parameter: {exc}") from None
    from ase2sprkkr import SPRKKR, Potential  # NOQA
    import sys

    calc = SPRKKR()
    ip = None
    try:
        potential = Potential.from_file(args.pot)
    except pyparsing.ParseException:
        try:
            calc.input_parameters = ip = InputParameters.from_file(args.pot)
            potential = ip.CONTROL.POTFIL()
        except pyparsing.ParseException:
            print(f"File {args.pot} is neither potential nor input_parameters file")
            sys.exit(-1)

    if args.input_file:
        calc.input_parameters = args.input_file
    elif args.task:
        calc.input_parameters = args.task
    kwargs = {}
    for i in "empty_spheres", "print_output", "output_file":
        o = getattr(args, i)
        if o is not None:
            kwargs[i] = o

    kwargs["profile"] = args.execution_profile
    if args.project_id is None:
        calc.calculate(potential=potential, input_parameters=ip, options=options, **kwargs)
    else:
        # Persistent project calculations use the same preparation as a normal
        # run, but expose the durable ID before the blocking wait.
        runner = calc.create_run(
            potential=potential,
            input_parameters=ip,
            options=options,
            project=args.project_id,
            **kwargs,
        )
        print(f"Persistent run ID: {runner.id}", flush=True)
        runner.run()
        from kkrunner import RunState
        from ase.calculators.calculator import CalculationFailed

        if runner.state != RunState.FINISHED:
            raise CalculationFailed(
                f"SPR-KKR calculation {runner.state.value} "
                f"(return code {runner.return_code}): {runner.failure_message}"
            )

