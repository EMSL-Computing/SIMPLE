#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent
BASE_ENVIRONMENT = REPO_ROOT / "environment.yml"
AMBERTOOLS_PACKAGES = (
    "ambertools>=26,<27",
    "biopython>=1.83,<1.86",
    "docutils>=0.17,<0.18",
)
NWCHEM_PACKAGES = (
    "nwchem>=7.3.1,<8",
    "openmpi>=5,<6",
    "mpi4py>=4.0",
)


class Colors:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, text: str) -> str:
        return self._wrap("1;32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("1;33", text)

    def red(self, text: str) -> str:
        return self._wrap("1;31", text)

    def cyan(self, text: str) -> str:
        return self._wrap("1;36", text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Conda installer for SIMPLE.")
    parser.add_argument("--env-name", default="simple", help="Conda environment name (default: simple)")
    parser.add_argument("--conda-executable", help="Path or command name for conda/mamba")
    parser.add_argument("--ambertools", choices=("conda", "external", "disabled"))
    parser.add_argument("--ambertools-home", default="")
    parser.add_argument("--amber", choices=("external", "module", "disabled"))
    parser.add_argument("--amber-home", default="")
    parser.add_argument("--amber-module", default="")
    parser.add_argument("--nwchem", choices=("conda", "external", "module", "disabled"))
    parser.add_argument("--nwchem-binary", default="")
    parser.add_argument("--mpi-launcher", default="")
    parser.add_argument("--nwchem-module", default="")
    parser.add_argument("--config", help="Override the generated tools.toml path")
    parser.add_argument("--yes", action="store_true", help="Accept safe defaults for omitted choices")
    parser.add_argument("--force-config", action="store_true", help="Replace an existing tools.toml")
    parser.add_argument("--skip-env", action="store_true", help="Configure tools without creating/updating Conda")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without changing anything")
    parser.add_argument("--no-color", action="store_true")
    return parser


def _confirm(prompt: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer Y or N.")


def _choice(prompt: str, choices: dict[str, str], *, default: str) -> str:
    while True:
        print(prompt)
        for key, label in choices.items():
            marker = " (default)" if key == default else ""
            print(f"  {key}. {label}{marker}")
        answer = input("Selection: ").strip() or default
        if answer in choices:
            return choices[answer].split(" ", maxsplit=1)[0].lower()
        print(f"Choose one of: {', '.join(choices)}")


def _required_path(prompt: str, current: str = "") -> str:
    while True:
        suffix = f" [{current}]" if current else ""
        value = input(f"{prompt}{suffix}: ").strip() or current
        if value:
            return value
        print("A path is required for this selection.")


def _select_modes(args: argparse.Namespace, colors: Colors) -> None:
    interactive = not args.yes
    print(colors.cyan("\nAmberTools 26"))
    print(
        "AmberTools 26 is recommended for SIMPLE system setup, 12-6-4 parameter preparation, "
        "analysis, and GUI workflows. It does not provide licensed pmemd executables."
    )
    if args.ambertools is None:
        if interactive:
            if _confirm("Install AmberTools 26 in the SIMPLE Conda environment?", default=True):
                args.ambertools = "conda"
            elif _confirm("Use an existing AmberTools installation?", default=False):
                args.ambertools = "external"
            else:
                args.ambertools = "disabled"
        else:
            args.ambertools = "conda"
    if args.ambertools == "external" and not args.ambertools_home:
        if args.yes:
            raise RuntimeError("--ambertools-home is required with --ambertools external and --yes.")
        args.ambertools_home = _required_path("AmberTools home directory")

    print(colors.red("\nLICENSED AMBER WARNING"))
    print(
        colors.red(
            "SIMPLE never downloads or installs licensed AMBER. Production MD and TI/free-energy "
            "simulation execution using pmemd, pmemd.MPI, or pmemd.cuda requires a separately "
            "licensed and installed AMBER distribution."
        )
    )
    if args.amber is None:
        if interactive:
            selected = _choice(
                "How should generic SIMPLE jobs access licensed AMBER?",
                {
                    "1": "external AMBERHOME path",
                    "2": "module environment",
                    "3": "disabled (system setup/analysis only)",
                },
                default="3",
            )
            args.amber = {"external": "external", "module": "module", "disabled": "disabled"}[selected]
        else:
            args.amber = "disabled"
    if args.amber == "external" and not args.amber_home:
        if args.yes:
            raise RuntimeError("--amber-home is required with --amber external and --yes.")
        args.amber_home = _required_path("Licensed AMBER home directory (AMBERHOME)")
    if args.amber == "module" and not args.amber_module:
        if args.yes:
            raise RuntimeError("--amber-module is required with --amber module and --yes.")
        args.amber_module = _required_path("Licensed AMBER module name", "amber")
    if args.amber == "disabled":
        print(
            colors.red(
                "Licensed AMBER remains disabled. Generic MD and TI/free-energy sbatch files will "
                "refuse to run until tools.toml is updated."
            )
        )

    print(colors.cyan("\nNWChem and MPI"))
    print(
        "Choose the Conda NWChem/MPI stack or an existing matched NWChem+MPI installation. "
        "Do not mix a Conda NWChem binary with an external MPI launcher."
    )
    if args.nwchem is None:
        if interactive:
            selected = _choice(
                "How should SIMPLE access NWChem?",
                {
                    "1": "conda NWChem and OpenMPI",
                    "2": "external NWChem and matching MPI",
                    "3": "disabled",
                },
                default="3",
            )
            args.nwchem = {"conda": "conda", "external": "external", "disabled": "disabled"}[selected]
        else:
            args.nwchem = "disabled"
    if args.nwchem == "external":
        if args.yes and (not args.nwchem_binary or not args.mpi_launcher):
            raise RuntimeError(
                "--nwchem-binary and --mpi-launcher are required with --nwchem external and --yes."
            )
        args.nwchem_binary = args.nwchem_binary or _required_path("Absolute NWChem executable path")
        args.mpi_launcher = args.mpi_launcher or _required_path("Matching mpirun/mpiexec path")
    if args.nwchem == "module" and not args.nwchem_module:
        if args.yes:
            raise RuntimeError("--nwchem-module is required with --nwchem module and --yes.")
        args.nwchem_module = _required_path("NWChem module name", "nwchem")

    print(colors.yellow("\nTahoma users"))
    print(
        "Conda AmberTools is recommended on Tahoma for SIMPLE preparation and analysis. "
        "Tahoma-specific sbatch files retain their existing site configuration, so a local "
        "licensed-AMBER path is not required to generate or use those Tahoma scripts."
    )


def _find_conda(requested: str | None) -> str:
    if requested:
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise RuntimeError(f"Conda executable was not found: {requested}")
    for name in ("conda", "mamba"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("Conda or Mamba was not found on PATH.")


def _run(command: Sequence[str], *, colors: Colors, dry_run: bool, capture: bool = False) -> str:
    print(colors.cyan("+ " + " ".join(command)))
    if dry_run:
        return ""
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def _environment_exists(conda: str, name: str, *, colors: Colors, dry_run: bool) -> bool:
    if dry_run:
        return False
    raw = _run((conda, "env", "list", "--json"), colors=colors, dry_run=False, capture=True)
    document = json.loads(raw)
    for prefix in document.get("envs", []):
        if Path(prefix).name == name:
            return True
    return False


def _install_environment(args: argparse.Namespace, conda: str, colors: Colors) -> None:
    if args.skip_env:
        return
    exists = _environment_exists(conda, args.env_name, colors=colors, dry_run=args.dry_run)
    action = "update" if exists else "create"
    command = [conda, "env", action, "--name", args.env_name, "--file", str(BASE_ENVIRONMENT)]
    _run(command, colors=colors, dry_run=args.dry_run)

    optional_packages: list[str] = []
    if args.ambertools == "conda":
        optional_packages.extend(AMBERTOOLS_PACKAGES)
    if args.nwchem == "conda":
        optional_packages.extend(NWCHEM_PACKAGES)
    if optional_packages:
        _run(
            [conda, "install", "--name", args.env_name, "--channel", "conda-forge", "--yes", *optional_packages],
            colors=colors,
            dry_run=args.dry_run,
        )


def _config_path(args: argparse.Namespace, conda: str, colors: Colors) -> Path | None:
    if args.config:
        return Path(args.config).expanduser()
    if args.dry_run:
        return None
    output = _run(
        (
            conda,
            "run",
            "--name",
            args.env_name,
            "python",
            "-c",
            "from amber_metallo.tool_config import default_tool_config_path; print(default_tool_config_path())",
        ),
        colors=colors,
        dry_run=False,
        capture=True,
    )
    return Path(output.splitlines()[-1].strip())


def _write_config(args: argparse.Namespace, conda: str, colors: Colors) -> Path | None:
    target = _config_path(args, conda, colors)
    if target and target.exists() and not args.force_config:
        if args.yes or not _confirm(f"Replace existing configuration {target}?", default=False):
            print(colors.yellow(f"Keeping existing configuration: {target}"))
            return target

    command = [
        conda,
        "run",
        "--name",
        args.env_name,
        "python",
        "-m",
        "amber_metallo.tool_config",
        "--ambertools-mode",
        args.ambertools,
        "--ambertools-home",
        args.ambertools_home,
        "--amber-mode",
        args.amber,
        "--amber-home",
        args.amber_home,
        "--amber-module",
        args.amber_module,
        "--nwchem-mode",
        args.nwchem,
        "--nwchem-binary",
        args.nwchem_binary,
        "--mpi-launcher",
        args.mpi_launcher,
        "--nwchem-module",
        args.nwchem_module,
    ]
    if target:
        command.extend(("--output", str(target)))
    output = _run(command, colors=colors, dry_run=args.dry_run, capture=not args.dry_run)
    if args.dry_run:
        return target
    return Path(output.splitlines()[-1].strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    colors = Colors(
        enabled=not args.no_color
        and not os.environ.get("NO_COLOR")
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )
    print(colors.green("SIMPLE interactive installer"))
    try:
        _select_modes(args, colors)
        conda = _find_conda(args.conda_executable)
        _install_environment(args, conda, colors)
        config_path = _write_config(args, conda, colors)
    except (RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(colors.red(f"Installation failed: {exc}"), file=sys.stderr)
        return 1

    if args.dry_run:
        print(colors.yellow("Dry run complete; no environment or configuration was changed."))
        return 0
    print(colors.green("\nSIMPLE installation/configuration completed."))
    if config_path:
        print(colors.cyan(f"Software configuration: {config_path}"))
    print("Edit this TOML later or run: simple configure")
    print("Changing a path updates discovery; changing to Conda mode does not install a package automatically.")
    if args.amber == "disabled":
        print(colors.red("Licensed AMBER is still required before running generic MD or TI/free-energy simulations."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
