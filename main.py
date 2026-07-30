from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

INSTALL_HINT = (
    "Install project dependencies first:\n"
    "  pip install .\n"
    "or\n"
    "  pip install -r requirements.txt"
)
SUPPORT_HINT = (
    "If you hit an error or have questions about this workflow, "
    "please contact Hoshin Kim (hoshin.kim@pnnl.gov)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SIMPLE in interactive mode or from a TOML config.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive workflow wizard and then run it.",
    )
    mode.add_argument(
        "--config",
        help="Path to a TOML config file for non-interactive execution.",
    )
    parser.add_argument(
        "--write-config",
        help="When using --interactive, save the prompted answers to this TOML file.",
    )
    parser.add_argument(
        "--from-stage",
        default="prepare",
        choices=("prepare", "system", "md"),
        help="Stage to start from.",
    )
    parser.add_argument(
        "--to-stage",
        default="md",
        choices=("prepare", "system", "md"),
        help="Stage to stop at.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate files without executing Amber binaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config and args.write_config:
        parser.error("--write-config can only be used together with --interactive.")

    try:
        from amber_metallo.config import load_config
        from amber_metallo.workflow import run_workflow

        if args.interactive:
            from amber_metallo.cli import build_wizard_configs, execute_wizard_configs

            wizard_result = build_wizard_configs(args.write_config)
            return execute_wizard_configs(
                wizard_result.configs,
                from_stage=args.from_stage,
                to_stage=args.to_stage,
                dry_run=args.dry_run,
                failure_hint=SUPPORT_HINT,
            )
        else:
            config = load_config(args.config)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        parser.exit(
            1,
            f"Missing dependency: {missing}\n{INSTALL_HINT}\n",
        )

    try:
        result = run_workflow(
            config=config,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            dry_run=args.dry_run,
        )
    except Exception:
        print(SUPPORT_HINT, file=sys.stderr)
        raise
    try:
        from amber_metallo.reporting import print_workflow_summary

        print_workflow_summary(result)
    except ModuleNotFoundError:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
