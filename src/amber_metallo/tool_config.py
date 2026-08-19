from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import tomllib
from typing import Callable, Mapping, Sequence

try:
    from platformdirs import user_config_path
except ModuleNotFoundError:  # Allows the bootstrap/config path to work before dependencies are installed.
    def user_config_path(appname: str, *, appauthor: bool = False) -> Path:
        del appauthor
        if os.name == "nt":
            root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return root / appname


CONFIG_VERSION = 1
_AMBERTOOLS_MODES = {"conda", "external", "disabled"}
_AMBER_MODES = {"external", "module", "disabled"}
_NWCHEM_MODES = {"conda", "external", "module", "disabled"}
_ACTIVATION_MODES = {"amber_sh", "none"}
_MODULE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]*$")
_EXECUTABLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_RUNNER_PATH_RE = re.compile(r"^[A-Za-z0-9_./+:-]+$")


def default_tool_config_path() -> Path:
    override = os.environ.get("SIMPLE_TOOLS_CONFIG")
    if override:
        return Path(override).expanduser()
    return user_config_path("simple", appauthor=False) / "tools.toml"


@dataclass(slots=True)
class AmberToolsSettings:
    mode: str = "disabled"
    home: str = ""


@dataclass(slots=True)
class AmberSettings:
    mode: str = "disabled"
    home: str = ""
    activation: str = "amber_sh"
    setup_script: str = ""
    module_name: str = ""
    serial: str = "pmemd"
    mpi: str = "pmemd.MPI"
    gpu: str = "pmemd.cuda"
    gpu_mpi: str = "pmemd.cuda.MPI"


@dataclass(slots=True)
class NWChemSettings:
    mode: str = "disabled"
    binary: str = ""
    mpi_launcher: str = ""
    module_name: str = ""


@dataclass(slots=True)
class ToolConfig:
    version: int = CONFIG_VERSION
    ambertools: AmberToolsSettings = field(default_factory=AmberToolsSettings)
    amber: AmberSettings = field(default_factory=AmberSettings)
    nwchem: NWChemSettings = field(default_factory=NWChemSettings)
    executables: dict[str, str] = field(default_factory=dict)
    source_path: Path | None = None


def _table(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = document.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"[{name}] must be a TOML table.")
    return value


def _string(table: Mapping[str, object], key: str, default: str = "") -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string in tools.toml.")
    return value.strip()


def _mode(table: Mapping[str, object], key: str, allowed: set[str], default: str = "disabled") -> str:
    value = _string(table, key, default).lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {key} value {value!r}; choose one of: {choices}.")
    return value


def _executable_name(table: Mapping[str, object], key: str, default: str) -> str:
    value = _string(table, key, default)
    if not _EXECUTABLE_NAME_RE.fullmatch(value):
        raise ValueError(f"software.amber.executables.{key} must be a plain executable filename.")
    return value


def load_tool_config(path: str | Path | None = None, *, required: bool = False) -> ToolConfig:
    config_path = Path(path).expanduser() if path is not None else default_tool_config_path()
    if not config_path.is_file():
        if required:
            raise FileNotFoundError(f"SIMPLE tool configuration was not found: {config_path}")
        return ToolConfig(source_path=config_path)

    try:
        with config_path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {config_path}: {exc}") from exc

    version = document.get("version", CONFIG_VERSION)
    if not isinstance(version, int) or version != CONFIG_VERSION:
        raise ValueError(f"Unsupported tools.toml version {version!r}; expected {CONFIG_VERSION}.")

    software = _table(document, "software")
    ambertools_table = _table(software, "ambertools")
    amber_table = _table(software, "amber")
    amber_executables = _table(amber_table, "executables")
    nwchem_table = _table(software, "nwchem")
    executables_table = _table(software, "executables")

    activation = _mode(amber_table, "activation", _ACTIVATION_MODES, "amber_sh")
    module_name = _string(amber_table, "module_name")
    nwchem_module = _string(nwchem_table, "module_name")
    if module_name and not _MODULE_NAME_RE.fullmatch(module_name):
        raise ValueError("software.amber.module_name contains unsupported shell characters.")
    if nwchem_module and not _MODULE_NAME_RE.fullmatch(nwchem_module):
        raise ValueError("software.nwchem.module_name contains unsupported shell characters.")

    extra_executables: dict[str, str] = {}
    for key, value in executables_table.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("[software.executables] keys and values must be strings.")
        extra_executables[key] = value.strip()

    return ToolConfig(
        version=version,
        ambertools=AmberToolsSettings(
            mode=_mode(ambertools_table, "mode", _AMBERTOOLS_MODES),
            home=_string(ambertools_table, "home"),
        ),
        amber=AmberSettings(
            mode=_mode(amber_table, "mode", _AMBER_MODES),
            home=_string(amber_table, "home"),
            activation=activation,
            setup_script=_string(amber_table, "setup_script"),
            module_name=module_name,
            serial=_executable_name(amber_executables, "serial", "pmemd"),
            mpi=_executable_name(amber_executables, "mpi", "pmemd.MPI"),
            gpu=_executable_name(amber_executables, "gpu", "pmemd.cuda"),
            gpu_mpi=_executable_name(amber_executables, "gpu_mpi", "pmemd.cuda.MPI"),
        ),
        nwchem=NWChemSettings(
            mode=_mode(nwchem_table, "mode", _NWCHEM_MODES),
            binary=_string(nwchem_table, "binary"),
            mpi_launcher=_string(nwchem_table, "mpi_launcher"),
            module_name=nwchem_module,
        ),
        executables=extra_executables,
        source_path=config_path,
    )


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_tool_config(config: ToolConfig) -> str:
    lines = [
        "# SIMPLE per-user external software configuration.",
        "# Edit paths here, or run `simple configure` again.",
        f"version = {CONFIG_VERSION}",
        "",
        "[software.ambertools]",
        f"mode = {_toml_string(config.ambertools.mode)}  # conda | external | disabled",
        f"home = {_toml_string(config.ambertools.home)}",
        "",
        "[software.amber]",
        "# Licensed AMBER is never installed by SIMPLE.",
        f"mode = {_toml_string(config.amber.mode)}  # external | module | disabled",
        f"home = {_toml_string(config.amber.home)}",
        f"activation = {_toml_string(config.amber.activation)}  # amber_sh | none",
        f"setup_script = {_toml_string(config.amber.setup_script)}",
        f"module_name = {_toml_string(config.amber.module_name)}",
        "",
        "[software.amber.executables]",
        f"serial = {_toml_string(config.amber.serial)}",
        f"mpi = {_toml_string(config.amber.mpi)}",
        f"gpu = {_toml_string(config.amber.gpu)}",
        f"gpu_mpi = {_toml_string(config.amber.gpu_mpi)}",
        "",
        "[software.nwchem]",
        f"mode = {_toml_string(config.nwchem.mode)}  # conda | external | module | disabled",
        f"binary = {_toml_string(config.nwchem.binary)}",
        f"mpi_launcher = {_toml_string(config.nwchem.mpi_launcher)}",
        f"module_name = {_toml_string(config.nwchem.module_name)}",
        "",
        "[software.executables]",
    ]
    if config.executables:
        for key, value in sorted(config.executables.items()):
            lines.append(f"{key} = {_toml_string(value)}")
    else:
        lines.extend(
            [
                'packmol = ""',
                'openbabel = ""',
                'pdbfixer = ""',
                'apptainer = ""',
            ]
        )
    return "\n".join(lines) + "\n"


def save_tool_config(
    config: ToolConfig,
    path: str | Path | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    config_path = Path(path).expanduser() if path is not None else default_tool_config_path()
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing tool configuration: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_tool_config(config)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        temporary_path.replace(config_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    config.source_path = config_path
    return config_path


def infer_amber_home(binary: str | Path | None) -> str:
    if not binary:
        return ""
    path = Path(binary).expanduser()
    if path.parent.name.lower() == "bin":
        return str(path.parent.parent)
    return ""


def discover_ambertools_home() -> str:
    env_home = os.environ.get("AMBERHOME", "").strip()
    if env_home and (Path(env_home).expanduser() / "dat" / "leap").is_dir():
        return str(Path(env_home).expanduser())
    return infer_amber_home(shutil.which("tleap"))


def discover_binary(name: str) -> str:
    return shutil.which(name) or ""


def resolve_tool_binary(
    name: str,
    *aliases: str,
    config: ToolConfig | None = None,
    path_finder: Callable[[str], str | None] | None = None,
) -> str | None:
    loaded = config or load_tool_config()
    config_path = loaded.source_path
    has_explicit_config = bool(config_path and config_path.is_file())
    for key in (name, *aliases):
        configured = loaded.executables.get(key, "")
        if configured:
            return str(Path(configured).expanduser())

    ambertools_names = {
        "tleap",
        "antechamber",
        "parmchk2",
        "parmed",
        "cpptraj",
        "ambpdb",
        "sander",
        "sander.MPI",
        "MMPBSA.py",
        "MMPBSA.py.MPI",
        "packmol",
    }
    amber_names = {"pmemd", "pmemd.MPI", "pmemd.cuda", "pmemd.cuda.MPI"}
    if name in ambertools_names and has_explicit_config:
        if loaded.ambertools.mode == "disabled" and loaded.amber.mode != "external":
            return None
        home = loaded.ambertools.home or (loaded.amber.home if loaded.amber.mode == "external" else "")
        if home:
            return str(Path(home).expanduser() / "bin" / name)
        if loaded.ambertools.mode != "conda":
            return None
    if name in amber_names and has_explicit_config:
        return configured_amber_binary(loaded, {
            "pmemd": "serial",
            "pmemd.MPI": "mpi",
            "pmemd.cuda": "gpu",
            "pmemd.cuda.MPI": "gpu_mpi",
        }[name])
    if name == "nwchem" and has_explicit_config:
        if loaded.nwchem.mode == "disabled":
            return None
        return loaded.nwchem.binary or ("nwchem" if loaded.nwchem.mode == "module" else None)

    finder = path_finder or shutil.which
    for candidate in (name, *aliases):
        found = finder(candidate)
        if found:
            return found
    return None


def configured_amber_binary(config: ToolConfig, kind: str) -> str | None:
    settings = config.amber
    if settings.mode == "disabled":
        return None
    names = {
        "serial": settings.serial,
        "mpi": settings.mpi,
        "gpu": settings.gpu,
        "gpu_mpi": settings.gpu_mpi,
    }
    try:
        name = names[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown AMBER executable kind: {kind}") from exc
    if settings.mode == "external":
        if not settings.home:
            return None
        home = settings.home.rstrip("/\\")
        return f"{home}/bin/{name}"
    return name


def _safe_runner_path(value: str, label: str) -> str:
    if not _RUNNER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"{label} contains whitespace or shell metacharacters that cannot be represented safely in a runner."
        )
    return value


def amber_sbatch_setup(
    config: ToolConfig | None = None,
    *,
    required_kinds: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    loaded = config or load_tool_config()
    settings = loaded.amber
    if settings.mode == "disabled":
        path = loaded.source_path or default_tool_config_path()
        return (
            [
                "# Licensed AMBER was not configured when this generic script was generated.",
                'echo "ERROR: Licensed AMBER is required for this MD/TI/free-energy calculation." >&2',
                f"echo {shlex.quote(f'Configure it in {path} or run: simple configure')} >&2",
                "exit 78",
            ],
            {},
        )

    binaries: dict[str, str] = {}
    setup_lines: list[str] = []
    if settings.mode == "external":
        if not settings.home:
            raise ValueError("software.amber.home is required when AMBER mode is 'external'.")
        home = settings.home.rstrip("/\\")
        setup_lines.append(f"export AMBERHOME={shlex.quote(home)}")
        if settings.activation == "amber_sh":
            script = settings.setup_script or f"{home}/amber.sh"
            setup_lines.extend(
                [
                    f"AMBER_SETUP={shlex.quote(script)}",
                    'if [ ! -f "$AMBER_SETUP" ]; then',
                    '  echo "ERROR: Configured AMBER activation script was not found: $AMBER_SETUP" >&2',
                    "  exit 78",
                    "fi",
                    'source "$AMBER_SETUP"',
                ]
            )
    elif settings.mode == "module":
        if not settings.module_name:
            raise ValueError("software.amber.module_name is required when AMBER mode is 'module'.")
        if not _MODULE_NAME_RE.fullmatch(settings.module_name):
            raise ValueError("software.amber.module_name contains unsupported shell characters.")
        setup_lines.extend(
            [
                "source /etc/profile.d/modules.sh",
                f"module load {settings.module_name}",
            ]
        )

    for kind in dict.fromkeys(required_kinds):
        binary = configured_amber_binary(loaded, kind)
        if not binary:
            raise ValueError(f"Licensed AMBER executable {kind!r} is not configured.")
        binary = _safe_runner_path(binary, f"AMBER {kind} executable")
        binaries[kind] = binary
        setup_lines.extend(
            [
                f"if [ ! -x {shlex.quote(binary)} ] && ! command -v {shlex.quote(binary)} >/dev/null 2>&1; then",
                f"  echo {shlex.quote(f'ERROR: Configured AMBER executable is unavailable: {binary}')} >&2",
                "  exit 78",
                "fi",
            ]
        )
    return setup_lines, binaries


def ambertools_sbatch_setup(
    config: ToolConfig | None = None,
    *,
    required_binaries: Sequence[str],
) -> list[str]:
    loaded = config or load_tool_config()
    settings = loaded.ambertools
    config_path = loaded.source_path or default_tool_config_path()
    if settings.mode == "disabled" and loaded.amber.mode != "external":
        return [
            "# AmberTools was not configured when this generic script was generated.",
            'echo "ERROR: AmberTools is required for this setup/analysis calculation." >&2',
            f"echo {shlex.quote(f'Configure it in {config_path} or run: simple configure')} >&2",
            "exit 78",
        ]

    home = settings.home or (loaded.amber.home if loaded.amber.mode == "external" else "")
    lines: list[str] = []
    if home:
        home = home.rstrip("/\\")
        lines.extend(
            [
                f"export AMBERHOME={shlex.quote(home)}",
                'export PATH="$AMBERHOME/bin:$PATH"',
            ]
        )
    for name in dict.fromkeys(required_binaries):
        if not _EXECUTABLE_NAME_RE.fullmatch(name):
            raise ValueError(f"Unsafe AmberTools executable name: {name!r}")
        lines.extend(
            [
                f"if ! command -v {name} >/dev/null 2>&1; then",
                f"  echo {shlex.quote(f'ERROR: Required AmberTools executable is unavailable: {name}')} >&2",
                "  exit 78",
                "fi",
            ]
        )
    return lines


def nwchem_sbatch_setup(config: ToolConfig | None = None) -> tuple[list[str], str, str]:
    loaded = config or load_tool_config()
    settings = loaded.nwchem
    config_path = loaded.source_path or default_tool_config_path()
    if settings.mode == "disabled":
        return (
            [
                "# NWChem was not configured when this generic script was generated.",
                'echo "ERROR: NWChem is required for this RESP/QM calculation." >&2',
                f"echo {shlex.quote(f'Configure it in {config_path} or run: simple configure')} >&2",
                "exit 78",
            ],
            "nwchem",
            "srun",
        )

    lines: list[str] = []
    if settings.mode == "module":
        if not settings.module_name or not _MODULE_NAME_RE.fullmatch(settings.module_name):
            raise ValueError("A safe software.nwchem.module_name is required for module mode.")
        lines.extend(["source /etc/profile.d/modules.sh", f"module load {settings.module_name}"])

    binary = settings.binary or "nwchem"
    binary = _safe_runner_path(binary, "NWChem executable")
    launcher = settings.mpi_launcher or "srun"
    launcher = _safe_runner_path(launcher, "NWChem MPI launcher")
    lines.extend(
        [
            f"if [ ! -x {shlex.quote(binary)} ] && ! command -v {shlex.quote(binary)} >/dev/null 2>&1; then",
            f"  echo {shlex.quote(f'ERROR: Configured NWChem executable is unavailable: {binary}')} >&2",
            "  exit 78",
            "fi",
            f"if [ ! -x {shlex.quote(launcher)} ] && ! command -v {shlex.quote(launcher)} >/dev/null 2>&1; then",
            f"  echo {shlex.quote(f'ERROR: Configured NWChem MPI launcher is unavailable: {launcher}')} >&2",
            "  exit 78",
            "fi",
        ]
    )
    return lines, binary, launcher


def tool_config_summary(config: ToolConfig | None = None) -> dict[str, str]:
    loaded = config or load_tool_config()
    return {
        "tools_config": str(loaded.source_path or default_tool_config_path()),
        "tools_config_exists": str(bool((loaded.source_path or default_tool_config_path()).is_file())),
        "ambertools_mode": loaded.ambertools.mode,
        "ambertools_home": loaded.ambertools.home or "Not configured",
        "amber_mode": loaded.amber.mode,
        "amber_home": loaded.amber.home or "Not configured",
        "nwchem_mode": loaded.nwchem.mode,
        "nwchem_binary": loaded.nwchem.binary or "Not configured",
        "nwchem_mpi_launcher": loaded.nwchem.mpi_launcher or "Not configured",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write SIMPLE's per-user tools.toml file.")
    parser.add_argument("--output")
    parser.add_argument("--ambertools-mode", choices=sorted(_AMBERTOOLS_MODES), required=True)
    parser.add_argument("--ambertools-home", default="")
    parser.add_argument("--amber-mode", choices=sorted(_AMBER_MODES), required=True)
    parser.add_argument("--amber-home", default="")
    parser.add_argument("--amber-module", default="")
    parser.add_argument("--nwchem-mode", choices=sorted(_NWCHEM_MODES), required=True)
    parser.add_argument("--nwchem-binary", default="")
    parser.add_argument("--mpi-launcher", default="")
    parser.add_argument("--nwchem-module", default="")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ambertools_home = args.ambertools_home
    if args.ambertools_mode == "conda" and not ambertools_home:
        ambertools_home = discover_ambertools_home()
    nwchem_binary = args.nwchem_binary
    mpi_launcher = args.mpi_launcher
    if args.nwchem_mode == "conda":
        nwchem_binary = nwchem_binary or discover_binary("nwchem")
        mpi_launcher = mpi_launcher or discover_binary("mpirun") or discover_binary("mpiexec")
    config = ToolConfig(
        ambertools=AmberToolsSettings(mode=args.ambertools_mode, home=ambertools_home),
        amber=AmberSettings(
            mode=args.amber_mode,
            home=args.amber_home,
            module_name=args.amber_module,
        ),
        nwchem=NWChemSettings(
            mode=args.nwchem_mode,
            binary=nwchem_binary,
            mpi_launcher=mpi_launcher,
            module_name=args.nwchem_module,
        ),
    )
    output = save_tool_config(config, args.output, overwrite=not args.no_overwrite)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
