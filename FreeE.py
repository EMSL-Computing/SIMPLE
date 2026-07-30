from __future__ import annotations

import argparse
import re
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
    "If you hit an error or have questions about this free-energy workflow, "
    "please contact Hoshin Kim (hoshin.kim@pnnl.gov)."
)
_MMPBSA_OUTPUT_DIR_RE = re.compile(r"^MM[-_]PBSA(?:(?:-|_)\d+)?$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SIMPLE free-energy launcher in interactive mode or from a TOML config.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive free-energy wizard and then run it.",
    )
    mode.add_argument(
        "--config",
        help="Path to a TOML config file for non-interactive free-energy setup generation.",
    )
    mode.add_argument(
        "--refresh-summaries",
        help="Existing MM-PBSA output directory. Rebuild summary.txt and summary_decomp.txt from saved output files only.",
    )
    mode.add_argument(
        "--refresh-summaries-batch",
        help="Batch root. Rebuild summary files for every matching MM-PBSA output directory under this root.",
    )
    parser.add_argument(
        "--write-config",
        help="When using --interactive, save the prompted answers to this TOML file.",
    )
    parser.add_argument(
        "--refresh-output-name",
        default="MM-PBSA",
        help="When using --refresh-summaries-batch, only refresh directories whose name exactly matches this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate free-energy files without validating runtime Amber executables.",
    )
    return parser


def _resolve_batch_refresh_arguments(search_root_arg: str, refresh_output_name: str) -> tuple[str, str]:
    candidate = Path(search_root_arg).expanduser()
    if candidate.exists():
        return str(candidate), refresh_output_name
    if refresh_output_name == "MM-PBSA" and _MMPBSA_OUTPUT_DIR_RE.match(search_root_arg.strip()):
        return ".", search_root_arg.strip()
    return search_root_arg, refresh_output_name


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config and args.write_config:
        parser.error("--write-config can only be used together with --interactive.")
    if args.refresh_output_name != "MM-PBSA" and not args.refresh_summaries_batch:
        parser.error("--refresh-output-name can only be used together with --refresh-summaries-batch.")

    try:
        from amber_metallo.free_energy.config import load_config
        from amber_metallo.free_energy.workflow import print_free_energy_workflow_summary, run_free_energy_workflow

        if args.interactive:
            from amber_metallo.free_energy.cli import build_free_energy_wizard_configs, execute_free_energy_configs

            wizard_result = build_free_energy_wizard_configs(args.write_config, dry_run=args.dry_run)
            return execute_free_energy_configs(
                wizard_result,
                dry_run=args.dry_run,
                failure_hint=SUPPORT_HINT,
            )
        elif args.refresh_summaries:
            from amber_metallo.free_energy.mmpbsa import refresh_mmpbsa_summaries

            refreshed = refresh_mmpbsa_summaries(args.refresh_summaries)
            print("Refreshed MM-PBSA summary files:")
            print(f"  summary.txt: {refreshed['summary_text']}")
            print(f"  summary.json: {refreshed['summary_json']}")
            print(f"  summary_decomp.txt: {refreshed['summary_decomp_text']}")
            print(f"  summary_decomp.json: {refreshed['summary_decomp_json']}")
            return 0
        elif args.refresh_summaries_batch:
            from amber_metallo.free_energy.mmpbsa import refresh_mmpbsa_summaries_batch

            search_root, output_dir_name = _resolve_batch_refresh_arguments(
                args.refresh_summaries_batch,
                args.refresh_output_name,
            )
            refreshed = refresh_mmpbsa_summaries_batch(
                search_root,
                output_dir_name=output_dir_name,
            )
            print(
                f"Refreshed MM-PBSA summaries under {refreshed['search_root']} "
                f"for directories named {refreshed['output_dir_name']}:"
            )
            print(f"  matched: {len(refreshed['matched_output_dirs'])}")
            print(f"  refreshed: {len(refreshed['refreshed'])}")
            print(f"  failed: {len(refreshed['failed'])}")
            print(f"  root summary.txt: {refreshed['root_summary_text']}")
            print(f"  root summary.json: {refreshed['root_summary_json']}")
            print(f"  root summary_decomp.txt: {refreshed['root_summary_decomp_text']}")
            print(f"  root summary_decomp.json: {refreshed['root_summary_decomp_json']}")
            for item in refreshed["refreshed"]:
                print(f"  OK: {item['output_dir']}")
            for item in refreshed["failed"]:
                print(f"  FAIL: {item['output_dir']} ({item['error']})")
            return 0 if not refreshed["failed"] else 1
        else:
            config = load_config(args.config)
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


if __name__ == "__main__":
    raise SystemExit(main())
