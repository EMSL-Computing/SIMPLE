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
    "If you hit an error or have questions about this analysis workflow, "
    "please contact Hoshin Kim (hoshin.kim@pnnl.gov)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the interactive ABFE/RBFE or trajectory analysis launcher.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Open the trajectory analysis wizard directly.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from amber_metallo.analysis_cli import run_analysis_wizard
        from amber_metallo.ti.abfe import print_analysis_summary
        from amber_metallo.trajectory_analysis import TrajectoryAnalysisRunResult
        from amber_metallo.trajectory_analysis_cli import (
            print_trajectory_analysis_summary,
            run_trajectory_analysis_wizard,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        parser.exit(
            1,
            f"Missing dependency: {missing}\n{INSTALL_HINT}\n",
        )

    try:
        from amber_metallo.subdirectory_search import (
            prompt_for_subdirectory_search,
            subdirectory_search_scope,
        )

        search_subdirectories = prompt_for_subdirectory_search(Path.cwd())
        with subdirectory_search_scope(search_subdirectories):
            if args.trajectory:
                result = run_trajectory_analysis_wizard()
            else:
                result = run_analysis_wizard()
    except Exception:
        print(SUPPORT_HINT, file=sys.stderr)
        raise
    if isinstance(result, TrajectoryAnalysisRunResult):
        print_trajectory_analysis_summary(result)
    else:
        print_analysis_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
