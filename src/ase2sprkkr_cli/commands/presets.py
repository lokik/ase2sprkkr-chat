"""CLI for YAML input-parameter presets."""

from pathlib import Path

help = "List, show, create, and delete input presets."
description = "Manage YAML input presets and import SPR-KKR .inp files."


def parser(parser):
    parser.add_argument("--list", action="store_true", dest="legacy_list",
                        help="Alias for 'list' (optionally followed by a task/IPD).")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing preset with create.")
    parser.add_argument("words", nargs="*", metavar="COMMAND [ARGS ...]")


def run(args, global_args):
    from ase2sprkkr import InputParameters
    from ase2sprkkr.input_parameters.presets import PresetStore
    from ase2sprkkr.common.native_preset_io import preset_read_context
    from ase2sprkkr.common.warnings import DataValidityError
    from ase2sprkkr_cli.errors import CLIUsageError
    import pyparsing

    store = PresetStore()
    words = list(args.words)
    action = "list" if args.legacy_list else (words.pop(0) if words else "list")
    try:
        if action == "list":
            if len(words) > 1:
                raise CLIUsageError("Usage: presets list [TASK_OR_IPD]")
            for name in store.list(words[0] if words else None):
                print(name)
        elif action == "show":
            if len(words) != 1:
                raise CLIUsageError("Usage: presets show NAME or IPD.NAME")
            print(store.resolve(words[0]).read_text(encoding="utf-8"), end="")
        elif action == "delete":
            if len(words) != 1:
                raise CLIUsageError("Usage: presets delete NAME or IPD.NAME")
            store.resolve(words[0]).unlink()
        elif action == "create":
            if len(words) != 2:
                raise CLIUsageError(
                    "Usage: presets create NAME INPUT.inp or presets create IPD.NAME PRESET.yaml"
                )
            name, source = words
            if Path(source).suffix.lower() in (".yaml", ".yml"):
                print(store.import_yaml(name, source, overwrite=args.force))
                return
            scope, owner, _bare = store._identifier(name)
            if owner is None:
                # Autodetection requires an actual native TASK selector.
                # An incomplete fragment must use a qualified destination.
                with preset_read_context():
                    ip = InputParameters.from_file(
                        source, allow_incomplete=True, report_invalid="raise",
                    )
            else:
                if scope is None:
                    scope, owner = store._context(owner)
                ip = (
                    InputParameters.for_task(owner)
                    if scope == "task"
                    else InputParameters.for_definition(owner)
                )
                with preset_read_context():
                    ip.read_from_file(
                        source, clear_first=True, allow_incomplete=True,
                        report_invalid="raise",
                    )
            print(store.save(ip, name, overwrite=args.force))
        else:
            raise CLIUsageError("Unknown command; use list, show, create, or delete")
    except CLIUsageError:
        raise
    except (OSError, KeyError, ValueError, pyparsing.ParseBaseException,
            DataValidityError) as exc:
        raise CLIUsageError(f"Preset {action!r} failed: {exc}") from None
