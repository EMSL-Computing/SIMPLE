#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the SIMPLE GUI.")
    parser.add_argument(
        "--web",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=0,
        help="Local GUI port. Default 0 chooses a free local port.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start without opening the default browser automatically.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from amber_metallo.gui_app.web_app import run_web_gui
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        print(
            f"Missing dependency: {missing}\n"
            "Install web GUI dependencies first:\n"
            "  pip install fastapi uvicorn python-multipart\n"
            "or recreate/update the conda environment and retry."
        )
        return 1
    try:
        return run_web_gui(
            REPO_ROOT,
            port=int(args.web_port or 0) or None,
            open_browser=not bool(args.no_browser),
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
