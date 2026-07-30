from __future__ import annotations

import argparse
import sys


INSTALL_HINT = (
    "Install project dependencies first:\n"
    "  pip install .\n"
    "or\n"
    "  pip install -r requirements.txt"
)
SUPPORT_HINT = (
    "If you hit an error or have questions about this TI workflow, "
    "please contact Hoshin Kim (hoshin.kim@pnnl.gov)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AMBER metal-decoupling TI setup generator in interactive mode or from a TOML config.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive TI setup wizard and then run it.",
    )
    mode.add_argument(
        "--config",
        help="Path to a TOML config file for non-interactive TI setup generation.",
    )
    parser.add_argument(
        "--write-config",
        help="When using --interactive, save the prompted answers to this TOML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate TI files without executing cpptraj, ParmEd, or Amber binaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config and args.write_config:
        parser.error("--write-config can only be used together with --interactive.")

    try:
        from amber_metallo.free_energy.config import from_ti_config
        from amber_metallo.free_energy.workflow import print_free_energy_workflow_summary, run_free_energy_workflow
        from amber_metallo.ti.config import load_config

        if args.interactive:
            from amber_metallo.ti.cli import build_ti_wizard_config

            config = from_ti_config(build_ti_wizard_config(args.write_config, dry_run=args.dry_run))
        else:
            config = from_ti_config(load_config(args.config))
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        parser.exit(
            1,
            f"Missing dependency: {missing}\n{INSTALL_HINT}\n",
        )

    try:
        result = run_free_energy_workflow(config=config, dry_run=args.dry_run)
    except Exception:
        print(SUPPORT_HINT, file=sys.stderr)
        raise
    print_free_energy_workflow_summary(result)
    return 0
