from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import shlex
import subprocess

from amber_metallo.amber.leap import (
    TLEAP_ADDITIVE_ION_UNIT_NAMES,
    DEFAULT_TLEAP_METAL_CHARGES,
    LeapBuildResult,
    TLEAP_ION_LIBRARY_LABELS,
    calculate_salt_ions,
    calculate_salt_ions_from_volume,
    combine_ion_counts,
    extract_total_charges,
    predict_neutralizing_ions,
    salt_concentration_m,
    salt_formula_units,
)
from amber_metallo.c4_assets import (
    opc_duvail_c4_file,
    opc_duvail_ion_frcmod,
    opc_duvail_polarizability_file,
)
from amber_metallo.config import (
    DESC4ParameterSet,
    DESComponent,
    DESConfig,
    DESMixingMode,
    DESReplicateOrder,
    DESSizeMode,
    NeutralizationIon,
    SaltConfig,
    SaltKind,
    SaltMode,
    SystemConfig,
)
from amber_metallo.environment import AmberEnvironment
from amber_metallo.execution import ensure_execution_host, run_command
from amber_metallo.reporting import write_json


@dataclass(frozen=True, slots=True)
class DESResidueDefinition:
    residue_name: str
    pdb: str | None
    frcmod: str | None = None
    lib: str | None = None
    atom_name: str | None = None
    element: str | None = None
    c4_capable: bool = False
    formal_charge: int | None = None


@dataclass(frozen=True, slots=True)
class DESComponentDefinition:
    key: DESComponent | str
    label: str
    description: str
    directory: str
    residues: tuple[DESResidueDefinition, ...]


@dataclass(frozen=True, slots=True)
class DESPlan:
    components: list[str]
    ratios: list[int]
    component_counts: dict[str, int]
    residue_counts: dict[str, int]
    ratio_units: int
    box_length_angstrom: float
    box_lengths_angstrom: tuple[float, float, float]
    box_volume_angstrom3: float
    total_residues: int
    total_atoms: int
    mixing_mode: str
    replicate_order: str
    size_mode: str
    spacing_angstrom: float
    packmol_tolerance_angstrom: float
    packmol_fill_fraction: float
    target_density_g_ml: float
    estimated_initial_density_g_ml: float
    c4_mask: str | None
    c4_residue_names: list[str]
    central_metal_element: str | None = None
    central_metal_charge: int | None = None
    central_metal_residue_name: str | None = None
    metal_sites: list[dict[str, object]] | None = None
    charge_before_ions: float | None = None
    neutralizing_ions: dict[str, int] | None = None
    extra_ions: dict[str, int] | None = None
    added_ions: dict[str, int] | None = None
    final_charge: float | None = None
    actual_salt_concentration_m: float | None = None
    neutralization_ion: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DES_COMPONENTS: dict[DESComponent | str, DESComponentDefinition] = {
    DESComponent.N8888_BR: DESComponentDefinition(
        key=DESComponent.N8888_BR,
        label="[N8888][Br]",
        description="Tetraoctylammonium bromide ion pair; expands to N88 + BR residues.",
        directory="N8888",
        residues=(
            DESResidueDefinition("N88", "N8888_h.pdb", "N88_h.frcmod", "N88_h_ptmpsi.lib"),
            DESResidueDefinition("BR", None, atom_name="BR", element="Br", c4_capable=True, formal_charge=-1),
        ),
    ),
    DESComponent.HEXANOIC_ACID: DESComponentDefinition(
        key=DESComponent.HEXANOIC_ACID,
        label="Hexanoic Acid",
        description="Neutral hexanoic acid form used by the REF_DATA recommended DES set.",
        directory="HAH",
        residues=(DESResidueDefinition("HAH", "HAH_h.pdb", "HAH_h.frcmod", "HAH_h_ptmpsi.lib"),),
    ),
    DESComponent.CHOLINE_CHLORIDE: DESComponentDefinition(
        key=DESComponent.CHOLINE_CHLORIDE,
        label="Choline Chloride",
        description="Choline chloride ion pair; expands to CH1 + CL residues.",
        directory="Choline",
        residues=(
            DESResidueDefinition("CH1", "CH1_h.pdb", "CH1_h.frcmod", "CH1_h_ptmpsi.lib"),
            DESResidueDefinition("CL", None, atom_name="CL", element="Cl", c4_capable=True, formal_charge=-1),
        ),
    ),
    DESComponent.ETHYLENE_GLYCOL: DESComponentDefinition(
        key=DESComponent.ETHYLENE_GLYCOL,
        label="Ethylene Glycol",
        description="Ethylene glycol HBD component.",
        directory="Ethylene-glycol",
        residues=(DESResidueDefinition("EG1", "EG1_h.pdb", "EG1_h.frcmod", "EG1_h_ptmpsi.lib"),),
    ),
    DESComponent.ACETONE: DESComponentDefinition(
        key=DESComponent.ACETONE,
        label="Acetone (ACN)",
        description="Neutral acetone solvent/diluent component.",
        directory="Acetone",
        residues=(DESResidueDefinition("ACN", None, "ACN_h.frcmod", "ACN_h_ptmpsi.lib"),),
    ),
    DESComponent.ETHANOL: DESComponentDefinition(
        key=DESComponent.ETHANOL,
        label="Ethanol (EtOH/ETO)",
        description="Neutral ethanol solvent/diluent component.",
        directory="Ethanol",
        residues=(DESResidueDefinition("ETO", None, "EtOH_h.frcmod", "EtOH_h_ptmpsi.lib"),),
    ),
    DESComponent.METHANOL: DESComponentDefinition(
        key=DESComponent.METHANOL,
        label="Methanol (MeOH/MEO)",
        description="Neutral methanol solvent/diluent component.",
        directory="Methanol",
        residues=(DESResidueDefinition("MEO", None, "MeOH_h.frcmod", "MeOH_h_ptmpsi.lib"),),
    ),
}

DES_RECOMMENDED_SETS: tuple[tuple[tuple[DESComponent, DESComponent], tuple[int, int]], ...] = (
    ((DESComponent.N8888_BR, DESComponent.HEXANOIC_ACID), (1, 2)),
    ((DESComponent.CHOLINE_CHLORIDE, DESComponent.ETHYLENE_GLYCOL), (1, 2)),
)

CUSTOM_DES_REGISTRY_FILENAME = "custom_des_components.json"
_CUSTOM_DES_KEY_RE = re.compile(r"[^a-z0-9_]+")


def _component_key_value(component: DESComponent | str) -> str:
    return component.value if isinstance(component, DESComponent) else str(component).strip()


def _component_registry_key(component: DESComponent | str) -> DESComponent | str:
    key = _component_key_value(component)
    try:
        return DESComponent(key)
    except ValueError:
        return key


def _component_count_key(component: DESComponent | str) -> str:
    return _component_key_value(component)


def _safe_custom_component_key(label: str, *, default: str = "custom_des") -> str:
    cleaned = _CUSTOM_DES_KEY_RE.sub("_", str(label or "").strip().lower()).strip("_")
    return cleaned or default


def custom_des_registry_path(ref_data_dir: str | Path = "REF_DATA") -> Path:
    return resolve_ref_data_dir(ref_data_dir) / CUSTOM_DES_REGISTRY_FILENAME


def _load_custom_des_registry(ref_data_dir: Path) -> dict[str, object]:
    path = ref_data_dir / CUSTOM_DES_REGISTRY_FILENAME
    if not path.exists():
        return {"components": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"components": {}}
    if not isinstance(payload, dict):
        return {"components": {}}
    components = payload.get("components")
    if not isinstance(components, dict):
        payload["components"] = {}
    return payload


def _custom_component_from_record(key: str, record: dict[str, object]) -> DESComponentDefinition | None:
    residue_name = str(record.get("residue_name") or "").strip().upper()
    lib = str(record.get("lib") or "").strip()
    frcmod = str(record.get("frcmod") or "").strip()
    directory = str(record.get("directory") or "").strip()
    if not residue_name or not lib or not frcmod or not directory:
        return None
    label = str(record.get("label") or key).strip() or key
    description = str(record.get("description") or "User-registered DES component.").strip()
    return DESComponentDefinition(
        key=key,
        label=label,
        description=description,
        directory=directory,
        residues=(DESResidueDefinition(residue_name, None, frcmod, lib),),
    )


def load_custom_des_components(ref_data_dir: str | Path = "REF_DATA") -> dict[str, DESComponentDefinition]:
    resolved = resolve_ref_data_dir(ref_data_dir)
    payload = _load_custom_des_registry(resolved)
    components: dict[str, DESComponentDefinition] = {}
    for key, record in (payload.get("components") or {}).items():
        if not isinstance(record, dict):
            continue
        definition = _custom_component_from_record(str(key), record)
        if definition is not None:
            components[str(key)] = definition
    return components


def available_des_components(ref_data_dir: str | Path = "REF_DATA") -> dict[DESComponent | str, DESComponentDefinition]:
    components: dict[DESComponent | str, DESComponentDefinition] = dict(DES_COMPONENTS)
    components.update(load_custom_des_components(ref_data_dir))
    return components


def _component_definition(ref_data_dir: Path, component: DESComponent | str) -> DESComponentDefinition:
    key = _component_registry_key(component)
    components = DES_COMPONENTS if isinstance(key, DESComponent) else available_des_components(ref_data_dir)
    try:
        return components[key]
    except KeyError as exc:
        raise KeyError(f"DES component is not registered: {_component_key_value(component)}") from exc


@dataclass(frozen=True, slots=True)
class DESLibraryCandidate:
    lib_path: Path
    frcmod_path: Path
    residue_name: str
    status: str
    matched_component: str | None = None
    matched_label: str | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_residue_name(path: str | Path) -> str:
    candidate = Path(path).expanduser().resolve()
    fallback: str | None = None
    expect_name = False
    for raw_line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("!entry."):
            parts = line.split(".")
            if len(parts) >= 2 and not fallback:
                fallback = parts[1].strip().upper()
            expect_name = ".unit.name" in line
            continue
        if expect_name and line.startswith('"'):
            try:
                return shlex.split(line)[0].strip().upper()
            except (IndexError, ValueError):
                break
    if fallback:
        return fallback
    raise ValueError(f"Could not infer residue name from Amber library file: {candidate}")


def _component_asset_hashes(ref_data_dir: Path, component: DESComponentDefinition) -> list[tuple[str, str, str]]:
    hashes: list[tuple[str, str, str]] = []
    for residue in component.residues:
        if not residue.lib or not residue.frcmod:
            continue
        lib_path = _component_dir(ref_data_dir, component) / residue.lib
        frcmod_path = _component_dir(ref_data_dir, component) / residue.frcmod
        if lib_path.exists() and frcmod_path.exists():
            hashes.append((residue.residue_name, _file_sha256(lib_path), _file_sha256(frcmod_path)))
    return hashes


def classify_des_library_bundle(
    *,
    lib_path: str | Path,
    frcmod_path: str | Path,
    ref_data_dir: str | Path = "REF_DATA",
) -> DESLibraryCandidate:
    resolved_ref = resolve_ref_data_dir(ref_data_dir)
    lib = Path(lib_path).expanduser().resolve()
    frcmod = Path(frcmod_path).expanduser().resolve()
    residue_name = library_residue_name(lib)
    lib_hash = _file_sha256(lib)
    frcmod_hash = _file_sha256(frcmod)
    for key, component in available_des_components(resolved_ref).items():
        for known_residue, known_lib_hash, known_frcmod_hash in _component_asset_hashes(resolved_ref, component):
            if known_residue.upper() != residue_name:
                continue
            if known_lib_hash == lib_hash and known_frcmod_hash == frcmod_hash:
                return DESLibraryCandidate(
                    lib_path=lib,
                    frcmod_path=frcmod,
                    residue_name=residue_name,
                    status="already_registered",
                    matched_component=_component_key_value(key),
                    matched_label=component.label,
                )
            return DESLibraryCandidate(
                lib_path=lib,
                frcmod_path=frcmod,
                residue_name=residue_name,
                status="different_values",
                matched_component=_component_key_value(key),
                matched_label=component.label,
            )
    return DESLibraryCandidate(lib_path=lib, frcmod_path=frcmod, residue_name=residue_name, status="new")


def _candidate_component_key(base: str, existing_keys: set[str]) -> str:
    key = _safe_custom_component_key(base)
    if key not in existing_keys:
        return key
    index = 2
    while f"{key}_{index}" in existing_keys:
        index += 1
    return f"{key}_{index}"


def register_custom_des_component(
    *,
    lib_path: str | Path,
    frcmod_path: str | Path,
    component_key: str | None = None,
    label: str | None = None,
    ref_data_dir: str | Path = "REF_DATA",
) -> DESComponentDefinition:
    resolved_ref = resolve_ref_data_dir(ref_data_dir)
    lib = Path(lib_path).expanduser().resolve()
    frcmod = Path(frcmod_path).expanduser().resolve()
    if not lib.exists() or not frcmod.exists():
        raise FileNotFoundError("Both Amber library and frcmod files are required for DES registration.")
    residue_name = library_residue_name(lib)
    payload = _load_custom_des_registry(resolved_ref)
    records = payload.setdefault("components", {})
    if not isinstance(records, dict):
        records = {}
        payload["components"] = records
    existing_keys = {_component_key_value(key) for key in available_des_components(resolved_ref)}
    base_key = component_key or f"custom_{residue_name.lower()}"
    key = _candidate_component_key(base_key, existing_keys)
    directory = Path("Custom_DES") / key
    target_dir = resolved_ref / directory
    target_dir.mkdir(parents=True, exist_ok=True)
    target_lib = target_dir / lib.name
    target_frcmod = target_dir / frcmod.name
    shutil.copy2(lib, target_lib)
    shutil.copy2(frcmod, target_frcmod)
    records[key] = {
        "key": key,
        "label": (label or residue_name).strip() or residue_name,
        "description": "User-registered DES component.",
        "directory": directory.as_posix(),
        "residue_name": residue_name,
        "lib": target_lib.name,
        "frcmod": target_frcmod.name,
        "lib_sha256": _file_sha256(target_lib),
        "frcmod_sha256": _file_sha256(target_frcmod),
    }
    registry_path = resolved_ref / CUSTOM_DES_REGISTRY_FILENAME
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    definition = _custom_component_from_record(key, records[key])
    if definition is None:
        raise RuntimeError(f"Failed to reload registered DES component: {key}")
    return definition


def overwrite_des_library_component(
    *,
    lib_path: str | Path,
    frcmod_path: str | Path,
    component_key: str,
    ref_data_dir: str | Path = "REF_DATA",
) -> DESComponentDefinition:
    """Replace one registered residue's Amber assets while preserving its component definition."""
    resolved_ref = resolve_ref_data_dir(ref_data_dir)
    component_key = str(component_key).strip()
    if component_key not in load_custom_des_components(resolved_ref):
        raise ValueError(
            "Only user-registered DES components can be overwritten; built-in components are protected."
        )
    lib = Path(lib_path).expanduser().resolve()
    frcmod = Path(frcmod_path).expanduser().resolve()
    if not lib.exists() or not frcmod.exists():
        raise FileNotFoundError("Both Amber library and frcmod files are required for DES replacement.")

    residue_name = library_residue_name(lib)
    components = available_des_components(resolved_ref)
    matched_key = next(
        (key for key in components if _component_key_value(key) == component_key),
        None,
    )
    if matched_key is None:
        raise KeyError(f"DES component is not registered: {component_key}")
    component = components[matched_key]
    residue = next(
        (
            item
            for item in component.residues
            if item.residue_name.upper() == residue_name and item.lib and item.frcmod
        ),
        None,
    )
    if residue is None:
        raise ValueError(
            f"Component '{component.label}' does not have replaceable .lib/.off and .frcmod assets for {residue_name}."
        )

    target_dir = _component_dir(resolved_ref, component)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lib, target_dir / str(residue.lib))
    shutil.copy2(frcmod, target_dir / str(residue.frcmod))

    payload = _load_custom_des_registry(resolved_ref)
    records = payload.get("components")
    record = records.get(str(component_key)) if isinstance(records, dict) else None
    if isinstance(record, dict):
        record["lib_sha256"] = _file_sha256(target_dir / str(residue.lib))
        record["frcmod_sha256"] = _file_sha256(target_dir / str(residue.frcmod))
        registry_path = resolved_ref / CUSTOM_DES_REGISTRY_FILENAME
        registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return component


def unregister_custom_des_component(
    component_key: str,
    *,
    ref_data_dir: str | Path = "REF_DATA",
) -> DESComponentDefinition:
    """Remove a user-registered DES component and its managed asset directory."""
    resolved_ref = resolve_ref_data_dir(ref_data_dir)
    key = str(component_key).strip()
    payload = _load_custom_des_registry(resolved_ref)
    records = payload.get("components")
    if not isinstance(records, dict) or not isinstance(records.get(key), dict):
        raise ValueError("Only user-registered DES components can be removed; built-in components are protected.")
    record = records[key]
    definition = _custom_component_from_record(key, record)
    if definition is None:
        raise ValueError(f"The custom DES component registry entry is incomplete: {key}")

    directory = (resolved_ref / definition.directory).resolve()
    custom_root = (resolved_ref / "Custom_DES").resolve()
    try:
        directory.relative_to(custom_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove a component directory outside {custom_root}") from exc

    del records[key]
    registry_path = resolved_ref / CUSTOM_DES_REGISTRY_FILENAME
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    return definition


def _library_pair_score(lib_path: Path, frcmod_path: Path) -> int:
    lib_stem = lib_path.stem.lower().replace("_ptmpsi", "")
    frc_stem = frcmod_path.stem.lower().replace("_ptmpsi", "")
    lib_tokens = [token for token in re.split(r"[_\-.]+", lib_stem) if token]
    frc_tokens = [token for token in re.split(r"[_\-.]+", frc_stem) if token]
    if lib_stem == frc_stem:
        return 100
    if lib_tokens and frc_tokens and lib_tokens[0] == frc_tokens[0]:
        return 80
    if lib_stem.startswith(frc_stem) or frc_stem.startswith(lib_stem):
        return 60
    return 0


def discover_des_library_candidates(
    search_dir: str | Path = ".",
    *,
    ref_data_dir: str | Path = "REF_DATA",
    recursive: bool = True,
) -> list[DESLibraryCandidate]:
    root = Path(search_dir).expanduser().resolve()
    excluded = {
        ".git",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "REF_DATA",
        "src",
        "tests",
        "docs",
        "gui_outputs",
        "tmp_plan_fetch",
    }
    library_files: list[Path] = []
    frcmod_files: list[Path] = []
    paths = root.rglob("*") if recursive else root.iterdir()
    for path in paths:
        if any(part in excluded for part in path.relative_to(root).parts[:-1]):
            continue
        if path.suffix.lower() in {".lib", ".off"}:
            library_files.append(path)
        elif path.suffix.lower() == ".frcmod":
            frcmod_files.append(path)
    candidates: list[DESLibraryCandidate] = []
    seen_pairs: set[tuple[Path, Path]] = set()
    for lib in sorted(library_files):
        same_folder_frcmods = [path for path in frcmod_files if path.parent == lib.parent]
        if not same_folder_frcmods:
            continue
        scored = sorted(
            ((_library_pair_score(lib, frcmod), frcmod) for frcmod in same_folder_frcmods),
            key=lambda item: (item[0], item[1].name.lower()),
            reverse=True,
        )
        for score, frcmod in scored:
            if score <= 0 and len(same_folder_frcmods) > 1:
                continue
            pair = (lib.resolve(), frcmod.resolve())
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidates.append(
                classify_des_library_bundle(
                    lib_path=pair[0],
                    frcmod_path=pair[1],
                    ref_data_dir=ref_data_dir,
                )
            )
    return candidates


@dataclass(slots=True)
class _AtomRecord:
    name: str
    residue_name: str
    x: float
    y: float
    z: float
    element: str


_ATOMIC_NUMBER_TO_ELEMENT = {
    1: "H",
    3: "Li",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    20: "Ca",
    26: "Fe",
    29: "Cu",
    35: "Br",
    34: "Se",
    39: "Y",
    53: "I",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
}

# Standard atomic weights (Da) for every element accepted by the DES builder.
# Pm has no standard atomic weight, so the mass number of its longest-lived
# naturally relevant isotope is used, as is conventional in periodic tables.
_ATOMIC_MASS_DA = {
    "H": 1.008,
    "Li": 6.94,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Br": 79.904,
    "Se": 78.971,
    "I": 126.90447,
    "Y": 88.90584,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.500,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,
}

_DALTON_PER_ANGSTROM3_TO_G_ML = 1.66053906660


@dataclass(frozen=True, slots=True)
class _DESMetalSpec:
    element: str
    charge: int
    residue_name: str
    coordinate: tuple[float, float, float] | None = None


_DES_ADDED_ION_PROPERTIES: dict[str, tuple[str, int]] = {
    "Na+": ("Na", 1),
    "K+": ("K", 1),
    "Cl-": ("Cl", -1),
    "Br-": ("Br", -1),
    "Ca2+": ("Ca", 2),
}


@dataclass(frozen=True, slots=True)
class _ReplicateGridLayout:
    sequence: list[tuple[DESComponentDefinition, DESResidueDefinition]]
    side_lengths: list[float]
    cell_side: float
    dimensions: tuple[int, int, int]
    occupied_lengths: tuple[float, float, float]
    cube_length: float


def default_ref_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "REF_DATA"


def resolve_ref_data_dir(ref_data_dir: str | Path) -> Path:
    candidate = Path(ref_data_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists():
        return candidate.resolve()
    fallback = default_ref_data_dir()
    if str(ref_data_dir) == "REF_DATA" and fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"DES REF_DATA directory was not found: {ref_data_dir}")


def recommended_ratio_for_components(components: list[DESComponent | str]) -> list[int]:
    selected = tuple(_component_key_value(component) for component in components)
    for pair, ratio in DES_RECOMMENDED_SETS:
        if selected == tuple(component.value for component in pair):
            return list(ratio)
    return [1 for _ in components]


def _component_dir(ref_data_dir: Path, component: DESComponentDefinition) -> Path:
    return ref_data_dir / component.directory


def _residue_file(ref_data_dir: Path, component: DESComponentDefinition, residue: DESResidueDefinition, kind: str) -> Path:
    name = getattr(residue, kind)
    if not name:
        raise ValueError(f"Residue {residue.residue_name} does not define a {kind} file")
    path = _component_dir(ref_data_dir, component) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing REF_DATA file for {component.label}: {path}")
    return path


def _infer_element(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return atom_name[:1].upper() or "X"
    if len(letters) >= 2 and letters[:2].title() in {"Cl", "Br"}:
        return letters[:2].title()
    return letters[:1].upper()


def _read_pdb_atoms(path: Path) -> list[_AtomRecord]:
    atoms: list[_AtomRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = line[12:16].strip() or "X"
        residue_name = line[17:20].strip() or "MOL"
        element = line[76:78].strip() if len(line) >= 78 else ""
        atoms.append(
            _AtomRecord(
                name=atom_name,
                residue_name=residue_name,
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
                element=element or _infer_element(atom_name),
            )
        )
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms


def _lib_atom_names(path: Path) -> list[str]:
    names: list[str] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if ".unit.atoms table" in line:
            in_atoms = True
            continue
        if in_atoms and line.startswith("!entry"):
            break
        stripped = line.strip()
        if in_atoms and stripped.startswith('"'):
            names.append(stripped.split('"', 2)[1])
    return names


def _lib_atom_charges(path: Path) -> list[float]:
    charges: list[float] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ".unit.atoms table" in line:
            in_atoms = True
            continue
        if in_atoms and line.startswith("!entry"):
            break
        stripped = line.strip()
        if not in_atoms or not stripped.startswith('"'):
            continue
        tokens = shlex.split(stripped)
        if len(tokens) < 8:
            continue
        try:
            charges.append(float(tokens[-1]))
        except ValueError:
            continue
    return charges


def _normalize_library_charge_sum(path: Path) -> tuple[float, float] | None:
    """Correct small text-rounding drift so each library unit has an integral charge."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    charge_rows: list[tuple[int, re.Match[str], float]] = []
    in_atoms = False
    charge_pattern = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*$")
    for index, line in enumerate(lines):
        if ".unit.atoms table" in line:
            in_atoms = True
            continue
        if in_atoms and line.startswith("!entry"):
            break
        if not in_atoms or not line.strip().startswith('"'):
            continue
        match = charge_pattern.search(line)
        if match is None:
            continue
        charge_rows.append((index, match, float(match.group(1))))
    if not charge_rows:
        return None
    original = sum(item[2] for item in charge_rows)
    target = float(round(original))
    if abs(original - target) > 0.05 or abs(original - target) < 1.0e-10:
        return None
    correction = (target - original) / len(charge_rows)
    corrected = [round(item[2] + correction, 8) for item in charge_rows]
    corrected[-1] = round(corrected[-1] + target - sum(corrected), 8)
    for (line_index, match, _old), charge in zip(charge_rows, corrected, strict=True):
        lines[line_index] = lines[line_index][: match.start(1)] + f"{charge:.8f}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return original, sum(corrected)


def _element_from_atomic_number(value: str) -> str:
    try:
        number = int(value)
    except ValueError:
        return "X"
    return _ATOMIC_NUMBER_TO_ELEMENT.get(number, "X")


def _lib_template_atoms(path: Path, residue_name: str) -> list[_AtomRecord]:
    atoms: list[tuple[str, str]] = []
    positions: list[tuple[float, float, float]] = []
    mode: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!entry"):
            if ".unit.atoms table" in line:
                mode = "atoms"
            elif ".unit.positions table" in line:
                mode = "positions"
            else:
                mode = None
            continue
        if mode == "atoms" and line.startswith('"'):
            tokens = shlex.split(line)
            if len(tokens) >= 7:
                atoms.append((tokens[0], _element_from_atomic_number(tokens[6])))
        elif mode == "positions":
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                positions.append((float(tokens[0]), float(tokens[1]), float(tokens[2])))
            except ValueError:
                continue

    if not atoms:
        raise ValueError(f"No Amber library atom table was found in {path}")
    if positions and len(positions) != len(atoms):
        raise ValueError(
            f"Amber library positions in {path} do not match atom count "
            f"({len(positions)} positions for {len(atoms)} atoms)."
        )
    if not positions:
        positions = [(index * 1.5, 0.0, 0.0) for index in range(len(atoms))]
    return [
        _AtomRecord(
            name=name,
            residue_name=residue_name,
            x=x,
            y=y,
            z=z,
            element=element if element != "X" else _infer_element(name),
        )
        for (name, element), (x, y, z) in zip(atoms, positions, strict=True)
    ]


def _with_library_atom_names(atoms: list[_AtomRecord], lib_path: Path) -> list[_AtomRecord]:
    names = _lib_atom_names(lib_path)
    if not names or len(names) != len(atoms):
        return atoms
    return [
        _AtomRecord(
            name=name,
            residue_name=atom.residue_name,
            x=atom.x,
            y=atom.y,
            z=atom.z,
            element=atom.element,
        )
        for atom, name in zip(atoms, names, strict=True)
    ]


def _synthetic_ion_atoms(residue: DESResidueDefinition) -> list[_AtomRecord]:
    return [
        _AtomRecord(
            name=residue.atom_name or residue.residue_name,
            residue_name=residue.residue_name,
            x=0.0,
            y=0.0,
            z=0.0,
            element=residue.element or _infer_element(residue.atom_name or residue.residue_name),
        )
    ]


def _added_ion_residue(ion_name: str) -> DESResidueDefinition:
    try:
        element, formal_charge = _DES_ADDED_ION_PROPERTIES[ion_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported DES salt ion: {ion_name}") from exc
    residue_name = TLEAP_ADDITIVE_ION_UNIT_NAMES[ion_name]
    return DESResidueDefinition(
        residue_name,
        None,
        atom_name=residue_name,
        element=element,
        c4_capable=True,
        formal_charge=formal_charge,
    )


def _added_ion_atoms(added_ions: dict[str, int] | None) -> list[tuple[DESResidueDefinition, _AtomRecord]]:
    atoms: list[tuple[DESResidueDefinition, _AtomRecord]] = []
    for ion_name, count in sorted((added_ions or {}).items()):
        residue = _added_ion_residue(ion_name)
        atom = _synthetic_ion_atoms(residue)[0]
        atoms.extend((residue, atom) for _ in range(max(0, int(count))))
    return atoms


def _template_atoms(ref_data_dir: Path, component: DESComponentDefinition, residue: DESResidueDefinition) -> list[_AtomRecord]:
    if residue.pdb is None:
        if residue.lib is not None:
            return _lib_template_atoms(_residue_file(ref_data_dir, component, residue, "lib"), residue.residue_name)
        return _synthetic_ion_atoms(residue)
    atoms = _read_pdb_atoms(_residue_file(ref_data_dir, component, residue, "pdb"))
    if residue.lib is None:
        return atoms
    return _with_library_atom_names(atoms, _residue_file(ref_data_dir, component, residue, "lib"))


def _bounding_span(atoms: list[_AtomRecord]) -> tuple[float, float, float]:
    xs = [atom.x for atom in atoms]
    ys = [atom.y for atom in atoms]
    zs = [atom.z for atom in atoms]
    return max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)


def _cell_side(atoms: list[_AtomRecord], spacing_angstrom: float) -> float:
    return max(_bounding_span(atoms)) + spacing_angstrom


def _component_unit_volume(
    ref_data_dir: Path,
    component: DESComponentDefinition,
    spacing_angstrom: float,
) -> float:
    volume = 0.0
    for residue in component.residues:
        atoms = _template_atoms(ref_data_dir, component, residue)
        volume += _cell_side(atoms, spacing_angstrom) ** 3
    return max(volume, spacing_angstrom**3)


def _component_atom_count(ref_data_dir: Path, component: DESComponentDefinition) -> int:
    return sum(len(_template_atoms(ref_data_dir, component, residue)) for residue in component.residues)


def _atom_mass_da(element: str) -> float:
    symbol = str(element or "").strip().title()
    try:
        return _ATOMIC_MASS_DA[symbol]
    except KeyError as exc:
        raise ValueError(
            f"No atomic mass is registered for element {element!r}; cannot determine a physical DES box density."
        ) from exc


def _residue_mass_da(
    ref_data_dir: Path,
    component: DESComponentDefinition,
    residue: DESResidueDefinition,
) -> float:
    return sum(_atom_mass_da(atom.element) for atom in _template_atoms(ref_data_dir, component, residue))


def _component_mass_da(ref_data_dir: Path, component: DESComponentDefinition) -> float:
    return sum(_residue_mass_da(ref_data_dir, component, residue) for residue in component.residues)


def _des_mass_da(
    config: DESConfig,
    ref_data_dir: Path,
    *,
    ratio_units: int,
    added_ion_count_by_name: dict[str, int] | None = None,
) -> float:
    total = 0.0
    for component_key, ratio in zip(config.components, config.ratios, strict=True):
        component = _component_definition(ref_data_dir, component_key)
        total += _component_mass_da(ref_data_dir, component) * ratio * ratio_units
    total += sum(_atom_mass_da(spec.element) for spec in _expanded_des_metal_specs(config))
    for ion_name, count in (added_ion_count_by_name or {}).items():
        residue = _added_ion_residue(ion_name)
        total += _atom_mass_da(str(residue.element)) * max(0, int(count))
    return total


def _density_g_ml(total_mass_da: float, volume_angstrom3: float) -> float:
    if volume_angstrom3 <= 0.0:
        return 0.0
    return total_mass_da * _DALTON_PER_ANGSTROM3_TO_G_ML / volume_angstrom3


def _minimum_packmol_box_length(config: DESConfig, ref_data_dir: Path) -> float:
    largest_span = 0.0
    largest_rotation_diameter = 0.0
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        for residue in component.residues:
            atoms = _template_atoms(ref_data_dir, component, residue)
            largest_span = max(largest_span, max(_bounding_span(atoms)))
            center = (
                sum(atom.x for atom in atoms) / len(atoms),
                sum(atom.y for atom in atoms) / len(atoms),
                sum(atom.z for atom in atoms) / len(atoms),
            )
            largest_rotation_diameter = max(
                largest_rotation_diameter,
                2.0
                * max(
                    math.dist((atom.x, atom.y, atom.z), center)
                    for atom in atoms
                ),
            )
    if config.mixing_mode == DESMixingMode.PACKMOL:
        # Packmol keeps atoms half a tolerance away from each periodic face.
        # Retain the older, more conservative span floor as well.
        return max(
            largest_span + 2.0 * config.packmol_tolerance_angstrom,
            largest_rotation_diameter + config.packmol_tolerance_angstrom,
        )
    # A freely rotated rigid molecule can project farther onto an axis than
    # its unrotated x/y/z span. The bounding-sphere diameter is orientation
    # independent and prevents small random-mix boxes rejecting the molecule.
    return max(largest_span, largest_rotation_diameter) + config.spacing_angstrom


def _minimum_automatic_metal_box_length(config: DESConfig) -> float:
    auto_count = sum(1 for spec in _expanded_des_metal_specs(config) if spec.coordinate is None)
    if auto_count <= 1:
        return 0.0
    dimensions = _balanced_grid_dimensions(auto_count)
    # Keep the centered lattice within the middle half of the box as well as
    # preserving its periodic clearance.  With n sites on the longest axis,
    # (n + 1) spacings place the endpoints at approximately L/4 and 3L/4.
    return (max(dimensions) + 1) * config.metal_spacing_angstrom + 1.0e-3


def _max_residue_cell_side(config: DESConfig, ref_data_dir: Path) -> float:
    max_cell_side = config.spacing_angstrom
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        for residue in component.residues:
            atoms = _template_atoms(ref_data_dir, component, residue)
            max_cell_side = max(max_cell_side, _cell_side(atoms, config.spacing_angstrom))
    return max_cell_side


def _residues_per_ratio_unit(config: DESConfig, ref_data_dir: Path) -> int:
    return sum(
        ratio * len(_component_definition(ref_data_dir, component_key).residues)
        for component_key, ratio in zip(config.components, config.ratios, strict=True)
    )


def _packmol_box_length_for_ratio_units(
    config: DESConfig,
    ref_data_dir: Path,
    ratio_units: int,
    *,
    added_ions: dict[str, int] | None = None,
) -> float:
    # The former implementation assigned a full largest-molecule cube to
    # every residue. For N8888/Br/2 HAH this produced a 139.6 A box for 100
    # formula units (~0.043 g/mL), i.e. a gas-like rather than DES starting
    # state. Size Packmol boxes from mass density and retain only a molecular
    # span floor for very small systems.
    total_mass_da = _des_mass_da(
        config,
        ref_data_dir,
        ratio_units=ratio_units,
        added_ion_count_by_name=added_ions,
    )
    effective_density = config.target_density_g_ml * (
        config.packmol_fill_fraction if config.mixing_mode == DESMixingMode.PACKMOL else 1.0
    )
    density_length = (total_mass_da * _DALTON_PER_ANGSTROM3_TO_G_ML / effective_density) ** (1.0 / 3.0)
    return max(
        density_length,
        _minimum_packmol_box_length(config, ref_data_dir),
        _minimum_automatic_metal_box_length(config),
    )


def _component_units(config: DESConfig, ref_data_dir: Path) -> list[DESComponentDefinition]:
    units: list[DESComponentDefinition] = []
    for component_key, ratio in zip(config.components, config.ratios, strict=True):
        component = _component_definition(ref_data_dir, component_key)
        for _ in range(ratio):
            units.append(component)
    return units


def _expanded_residue_sequence(
    config: DESConfig,
    ratio_units: int,
    ref_data_dir: Path | None = None,
) -> list[tuple[DESComponentDefinition, DESResidueDefinition]]:
    if ref_data_dir is None:
        ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    component_units: list[DESComponentDefinition]
    if config.replicate_order == DESReplicateOrder.GROUPED:
        component_units = []
        for component_key, ratio in zip(config.components, config.ratios, strict=True):
            component = _component_definition(ref_data_dir, component_key)
            component_units.extend([component for _ in range(ratio * ratio_units)])
    else:
        component_units = []
        per_unit = list(_component_units(config, ref_data_dir))
        for _ in range(ratio_units):
            component_units.extend(per_unit)
        if config.replicate_order == DESReplicateOrder.RANDOM:
            random.Random(20240517).shuffle(component_units)

    sequence: list[tuple[DESComponentDefinition, DESResidueDefinition]] = []
    for component in component_units:
        for residue in component.residues:
            sequence.append((component, residue))
    return sequence


def _pack_side_lengths(side_lengths: list[float], target_edge: float) -> tuple[float, float, float]:
    x = y = z = 0.0
    row_depth = 0.0
    layer_depth = 0.0
    max_x = max_y = max_z = 0.0
    for side in side_lengths:
        if x > 0.0 and x + side > target_edge:
            x = 0.0
            y += row_depth
            row_depth = 0.0
        if y > 0.0 and y + side > target_edge:
            y = 0.0
            z += layer_depth
            layer_depth = 0.0
        max_x = max(max_x, x + side)
        max_y = max(max_y, y + side)
        max_z = max(max_z, z + side)
        x += side
        row_depth = max(row_depth, side)
        layer_depth = max(layer_depth, side)
    return (max_x, max_y, max_z)


def _replicate_target_edge(side_lengths: list[float], requested_box_length: float | None) -> float:
    if requested_box_length is not None:
        return requested_box_length
    if not side_lengths:
        return 1.0
    total_cell_volume = sum(side**3 for side in side_lengths)
    lower = max(max(side_lengths), total_cell_volume ** (1 / 3))
    upper = max(lower, sum(side_lengths))
    best_target = lower
    best_score: tuple[float, float] | None = None
    steps = 80
    for index in range(steps):
        target = lower if steps == 1 else lower + (upper - lower) * index / (steps - 1)
        lengths = _pack_side_lengths(side_lengths, target)
        shortest = max(min(lengths), 1e-6)
        aspect_ratio = max(lengths) / shortest
        volume_ratio = (lengths[0] * lengths[1] * lengths[2]) / max(total_cell_volume, 1e-6)
        score = (aspect_ratio + 0.25 * volume_ratio, volume_ratio)
        if best_score is None or score < best_score:
            best_score = score
            best_target = target
    return best_target


def _balanced_grid_dimensions(count: int) -> tuple[int, int, int]:
    if count <= 1:
        return (1, 1, 1)
    max_axis = max(2, math.ceil(count ** (1 / 3)) + 4)
    while max_axis**3 < count:
        max_axis += 1

    best_dims = (1, 1, count)
    best_score: tuple[float, float, int] | None = None
    for a in range(1, max_axis + 1):
        for b in range(a, max_axis + 1):
            for c in range(b, max_axis + 1):
                capacity = a * b * c
                if capacity < count:
                    continue
                aspect = c / a
                empty_fraction = (capacity - count) / capacity
                score = (aspect + 3.0 * empty_fraction, aspect, capacity)
                if best_score is None or score < best_score:
                    best_score = score
                    best_dims = (a, b, c)
    return best_dims


def _replicate_grid_layout(
    config: DESConfig,
    ref_data_dir: Path,
    ratio_units: int,
    *,
    added_ion_count: int = 0,
) -> _ReplicateGridLayout:
    sequence = _expanded_residue_sequence(config, ratio_units, ref_data_dir)
    auto_metal_count = sum(1 for spec in _expanded_des_metal_specs(config) if spec.coordinate is None)
    reserved_cell_count = auto_metal_count + max(0, int(added_ion_count))
    if not sequence:
        cube_length = config.box_length_angstrom or config.spacing_angstrom
        return _ReplicateGridLayout(
            sequence=[],
            side_lengths=[],
            cell_side=config.spacing_angstrom,
            dimensions=(1, 1, 1),
            occupied_lengths=(cube_length, cube_length, cube_length),
            cube_length=cube_length,
        )

    side_lengths = [
        _cell_side(_template_atoms(ref_data_dir, component, residue), config.spacing_angstrom)
        for component, residue in sequence
    ]
    cell_side = max(
        max(side_lengths),
        config.metal_spacing_angstrom if auto_metal_count else config.spacing_angstrom,
    )
    dimensions = _balanced_grid_dimensions(len(sequence) + reserved_cell_count)
    occupied_lengths = (
        dimensions[0] * cell_side,
        dimensions[1] * cell_side,
        dimensions[2] * cell_side,
    )
    cube_length = max(max(occupied_lengths), config.box_length_angstrom or 0.0)
    return _ReplicateGridLayout(
        sequence=sequence,
        side_lengths=side_lengths,
        cell_side=cell_side,
        dimensions=dimensions,
        occupied_lengths=occupied_lengths,
        cube_length=cube_length,
    )


def _replicate_box_lengths(
    config: DESConfig,
    ref_data_dir: Path,
    ratio_units: int,
    *,
    added_ion_count: int = 0,
) -> tuple[float, float, float]:
    layout = _replicate_grid_layout(
        config,
        ref_data_dir,
        ratio_units,
        added_ion_count=added_ion_count,
    )
    return (layout.cube_length, layout.cube_length, layout.cube_length)


def _ratio_units_for_box(config: DESConfig, ref_data_dir: Path, *, added_ion_count: int = 0) -> int:
    assert config.box_length_angstrom is not None
    if config.mixing_mode in {DESMixingMode.RANDOM_MIX, DESMixingMode.PACKMOL}:
        unit_mass_da = 0.0
        for component_key, ratio in zip(config.components, config.ratios, strict=True):
            component = _component_definition(ref_data_dir, component_key)
            unit_mass_da += ratio * _component_mass_da(ref_data_dir, component)
        if unit_mass_da <= 0.0:
            return 1
        target_mass_da = (
            config.target_density_g_ml
            * config.box_length_angstrom**3
            * (config.packmol_fill_fraction if config.mixing_mode == DESMixingMode.PACKMOL else 1.0)
            / _DALTON_PER_ANGSTROM3_TO_G_ML
        )
        fixed_mass_da = sum(_atom_mass_da(spec.element) for spec in _expanded_des_metal_specs(config))
        # The exact ion identities are not available at this inversion stage;
        # chloride is a conservative mass proxy and the final iterative ion
        # plan recomputes the reported density from the actual species.
        fixed_mass_da += max(0, int(added_ion_count)) * _atom_mass_da("Cl")
        capacity_units = max(0.0, target_mass_da - fixed_mass_da) / unit_mass_da
        return max(1, int(math.floor(capacity_units + 1.0e-9)))

    if config.mixing_mode == DESMixingMode.REPLICATE:
        max_cell_side = _max_residue_cell_side(config, ref_data_dir)
        grid_axis = max(1, math.floor((config.box_length_angstrom + 1.0e-6) / max_cell_side))
        residues_per_unit = _residues_per_ratio_unit(config, ref_data_dir)
        if residues_per_unit <= 0:
            return 1
        reserved_residues = len(_expanded_des_metal_specs(config)) + max(0, int(added_ion_count))
        residue_capacity = max(0.0, grid_axis**3 - reserved_residues)
        return max(1, int(residue_capacity // residues_per_unit))

    unit_volume = 0.0
    for component_key, ratio in zip(config.components, config.ratios, strict=True):
        component = _component_definition(ref_data_dir, component_key)
        unit_volume += ratio * _component_unit_volume(ref_data_dir, component, config.spacing_angstrom)
    if unit_volume <= 0:
        return 1
    return max(1, int((config.box_length_angstrom**3) // unit_volume))


def _des_metal_residue_name(element: str, charge: int) -> str:
    normalized_element = element.title()
    return TLEAP_ION_LIBRARY_LABELS.get((normalized_element, int(charge)), normalized_element.upper()[:3])


def _expanded_des_metal_specs(config: DESConfig) -> list[_DESMetalSpec]:
    specs: list[_DESMetalSpec] = []
    for site in config.metal_sites:
        residue_name = _des_metal_residue_name(site.element, site.charge)
        for index in range(site.count):
            coordinate = None
            if site.coordinates is not None and index < len(site.coordinates):
                raw = site.coordinates[index]
                coordinate = (float(raw[0]), float(raw[1]), float(raw[2]))
            specs.append(
                _DESMetalSpec(
                    element=site.element.title(),
                    charge=int(site.charge),
                    residue_name=residue_name,
                    coordinate=coordinate,
                )
            )
    return specs


def _residue_nominal_charge(
    ref_data_dir: Path,
    component: DESComponentDefinition,
    residue: DESResidueDefinition,
) -> float:
    if residue.formal_charge is not None:
        return float(residue.formal_charge)
    if residue.lib is None:
        return 0.0
    charges = _lib_atom_charges(_residue_file(ref_data_dir, component, residue, "lib"))
    if not charges:
        raise ValueError(f"Could not read Amber library charges for {component.label}/{residue.residue_name}.")
    total = sum(charges)
    rounded = round(total)
    return float(rounded) if abs(total - rounded) <= 0.05 else total


def _component_counts_for_config(
    config: DESConfig,
    ref_data_dir: Path,
    *,
    added_ion_count: int = 0,
) -> tuple[int, dict[str, int]]:
    ratio_units = (
        config.ratio_units
        if config.size_mode == DESSizeMode.RATIO_UNITS
        else _ratio_units_for_box(config, ref_data_dir, added_ion_count=added_ion_count)
    )
    assert ratio_units is not None
    component_counts = {
        _component_count_key(component): ratio * ratio_units
        for component, ratio in zip(config.components, config.ratios, strict=True)
    }
    return ratio_units, component_counts


def estimate_des_net_charge(
    config: DESConfig,
    *,
    component_counts: dict[str, int] | None = None,
) -> float:
    """Estimate the integral DES charge from library units and configured metal oxidation states."""
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    if component_counts is None:
        _ratio_units, component_counts = _component_counts_for_config(config, ref_data_dir)
    total = 0.0
    for component_key, count in component_counts.items():
        component = _component_definition(ref_data_dir, component_key)
        component_charge = sum(
            _residue_nominal_charge(ref_data_dir, component, residue)
            for residue in component.residues
        )
        total += component_charge * count
    total += sum(spec.charge for spec in _expanded_des_metal_specs(config))
    rounded = round(total)
    return float(rounded) if abs(total - rounded) <= 0.05 else total


def resolve_des_neutralization_ion(
    config: DESConfig,
    salt_config: SaltConfig,
    net_charge: float,
) -> NeutralizationIon:
    """Resolve Auto to the native monatomic counter-ion of the selected DES."""
    requested = salt_config.neutralization_ion
    if requested != NeutralizationIon.AUTO:
        return requested
    if net_charge > 0.05:
        ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
        for component_key in config.components:
            component = _component_definition(ref_data_dir, component_key)
            for residue in component.residues:
                if residue.formal_charge is None or residue.formal_charge >= 0:
                    continue
                element = str(residue.element or "").title()
                if element == "Br":
                    return NeutralizationIon.BROMIDE
                if element == "Cl":
                    return NeutralizationIon.CHLORIDE
    # Negative DES boxes need a small monatomic cation; use the selected salt
    # pair because the native DES cations are polyatomic molecular species.
    return NeutralizationIon.SALT_DEFAULT


def _ion_plan_for_des(
    config: DESConfig,
    salt_config: SaltConfig,
    *,
    volume_angstrom3: float,
    component_counts: dict[str, int],
) -> dict[str, object]:
    charge_before = estimate_des_net_charge(config, component_counts=component_counts)
    neutralizing: dict[str, int] = {}
    extra: dict[str, int] = {}
    neutralization_ion = resolve_des_neutralization_ion(config, salt_config, charge_before)
    if salt_config.mode != SaltMode.NONE and salt_config.kind != SaltKind.NONE:
        neutralizing, _remaining = predict_neutralizing_ions(
            charge_before,
            salt_config.kind,
            neutralization_ion,
        )
        if salt_config.mode == SaltMode.COUNT:
            extra = calculate_salt_ions(0, salt_config)
        elif salt_config.mode == SaltMode.CONCENTRATION:
            extra = calculate_salt_ions_from_volume(volume_angstrom3, salt_config)
    added = combine_ion_counts(neutralizing, extra)
    final_charge = charge_before
    for ion_name, count in added.items():
        ion_charge = _added_ion_residue(ion_name).formal_charge
        assert ion_charge is not None
        final_charge += ion_charge * count
    if abs(final_charge) <= 0.05:
        final_charge = 0.0
    formula_units = salt_formula_units(extra, salt_config.kind)
    return {
        "charge_before_ions": charge_before,
        "neutralizing_ions": neutralizing,
        "extra_ions": extra,
        "added_ions": added,
        "final_charge": final_charge,
        "actual_salt_concentration_m": salt_concentration_m(formula_units, volume_angstrom3),
        "neutralization_ion": neutralization_ion.value,
    }


def _auto_metal_coordinates(
    *,
    count: int,
    box_lengths: tuple[float, float, float],
    spacing_angstrom: float,
    existing_coordinates: list[tuple[float, float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []
    dimensions = _balanced_grid_dimensions(count)
    axes: list[list[float]] = []
    for length, points in zip(box_lengths, dimensions, strict=True):
        extent = (points - 1) * spacing_angstrom
        if extent >= length:
            raise ValueError(
                f"The DES box length {length:.2f} A is too small for a centered {dimensions} metal lattice "
                f"at {spacing_angstrom:.2f} A spacing."
            )
        start = 0.5 * (length - extent)
        axes.append([start + index * spacing_angstrom for index in range(points)])
    candidates = [
        (x, y, z)
        for x in axes[0]
        for y in axes[1]
        for z in axes[2]
    ]
    center = tuple(length * 0.5 for length in box_lengths)
    selected = _center_symmetric_subset(candidates, count=count, center=center)
    all_coordinates = [*(existing_coordinates or []), *selected]
    for index, first in enumerate(all_coordinates):
        for second in all_coordinates[index + 1 :]:
            distance = _periodic_distance(first, second, box_lengths)
            if distance + 1.0e-6 < spacing_angstrom:
                raise ValueError(
                    f"Centered automatic DES metal sites are only {distance:.2f} A apart under periodic "
                    f"boundaries; the configured minimum is {spacing_angstrom:.2f} A."
                )
    return selected


def _center_symmetric_subset(
    candidates: list[tuple[float, float, float]],
    *,
    count: int,
    center: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Choose a deterministic, approximately inversion-symmetric subset around the box center."""
    if count >= len(candidates):
        return list(candidates)
    remaining = set(candidates)
    selected: list[tuple[float, float, float]] = []
    while remaining and len(selected) < count:
        point = min(
            remaining,
            key=lambda item: (
                math.dist(item, center),
                abs(item[2] - center[2]),
                abs(item[1] - center[1]),
                abs(item[0] - center[0]),
                item,
            ),
        )
        mirror = tuple(2.0 * center[axis] - point[axis] for axis in range(3))
        remaining.remove(point)
        selected.append(point)
        if len(selected) < count and mirror in remaining:
            remaining.remove(mirror)
            selected.append(mirror)
    return selected


def _periodic_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    box_lengths: tuple[float, float, float],
) -> float:
    deltas = []
    for a, b, length in zip(first, second, box_lengths, strict=True):
        delta = abs(a - b)
        deltas.append(min(delta, max(0.0, length - delta)))
    return math.sqrt(sum(delta * delta for delta in deltas))


def _select_spaced_coordinates(
    candidates: list[tuple[float, float, float]],
    *,
    count: int,
    box_lengths: tuple[float, float, float],
    minimum_spacing: float,
    existing_coordinates: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    selected: list[tuple[float, float, float]] = []
    remaining = list(dict.fromkeys(candidates))
    center = tuple(length * 0.5 for length in box_lengths)
    while remaining and len(selected) < count:
        occupied = [*existing_coordinates, *selected]
        if occupied:
            choice = max(
                remaining,
                key=lambda point: (
                    min(_periodic_distance(point, other, box_lengths) for other in occupied),
                    -_periodic_distance(point, center, box_lengths),
                    point,
                ),
            )
        else:
            choice = min(remaining, key=lambda point: (_periodic_distance(point, center, box_lengths), point))
        selected.append(choice)
        remaining.remove(choice)
    if len(selected) != count:
        raise ValueError(
            f"Could not generate {count} distinct automatic metal coordinates inside the DES box."
        )
    all_coordinates = [*existing_coordinates, *selected]
    for index, first in enumerate(all_coordinates):
        for second in all_coordinates[index + 1 :]:
            distance = _periodic_distance(first, second, box_lengths)
            if distance + 1.0e-6 < minimum_spacing:
                raise ValueError(
                    f"The DES box is too small for the requested metal count at a minimum metal-metal "
                    f"spacing of {minimum_spacing:.2f} A (closest generated distance: {distance:.2f} A). "
                    "Increase the DES box/ratio units, reduce the metal count, or lower metal_spacing_angstrom."
                )
    return selected


def _replicate_reserved_cell_coordinates(
    config: DESConfig,
    *,
    ratio_units: int,
    box_lengths: tuple[float, float, float],
    added_ion_count: int,
) -> list[tuple[float, float, float]]:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    layout = _replicate_grid_layout(
        config,
        ref_data_dir,
        ratio_units,
        added_ion_count=added_ion_count,
    )
    offset = (
        max((box_lengths[0] - layout.occupied_lengths[0]) / 2.0, 0.0),
        max((box_lengths[1] - layout.occupied_lengths[1]) / 2.0, 0.0),
        max((box_lengths[2] - layout.occupied_lengths[2]) / 2.0, 0.0),
    )
    return [
        _replicate_cell_coordinate(index, layout=layout, offset=offset)
        for index in _replicate_reserved_cell_indices(
            config,
            layout=layout,
            added_ion_count=added_ion_count,
        )
    ]


def _replicate_cell_coordinate(
    index: int,
    *,
    layout: _ReplicateGridLayout,
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    nx, ny, _nz = layout.dimensions
    ix = index % nx
    iy = (index // nx) % ny
    iz = index // (nx * ny)
    return (
        offset[0] + (ix + 0.5) * layout.cell_side,
        offset[1] + (iy + 0.5) * layout.cell_side,
        offset[2] + (iz + 0.5) * layout.cell_side,
    )


def _replicate_reserved_cell_indices(
    config: DESConfig,
    *,
    layout: _ReplicateGridLayout,
    added_ion_count: int,
) -> list[int]:
    auto_metal_count = sum(1 for spec in _expanded_des_metal_specs(config) if spec.coordinate is None)
    ion_count = max(0, int(added_ion_count))
    reserved_count = auto_metal_count + ion_count
    if reserved_count <= 0:
        return []
    nx, ny, nz = layout.dimensions
    capacity = nx * ny * nz
    if len(layout.sequence) + reserved_count > capacity:
        raise RuntimeError("The replicate DES grid does not contain enough reserved metal/ion cells.")

    metal_indices: list[int] = []
    if auto_metal_count:
        center = ((nx - 1) * 0.5, (ny - 1) * 0.5, (nz - 1) * 0.5)
        candidate_points = [
            (float(index % nx), float((index // nx) % ny), float(index // (nx * ny)))
            for index in range(capacity)
        ]
        selected_points = _center_symmetric_subset(candidate_points, count=auto_metal_count, center=center)
        point_to_index = {point: index for index, point in enumerate(candidate_points)}
        metal_indices = [point_to_index[point] for point in selected_points]

    remaining = [index for index in range(capacity) if index not in set(metal_indices)]
    center = ((nx - 1) * 0.5, (ny - 1) * 0.5, (nz - 1) * 0.5)
    ion_indices = sorted(
        remaining,
        key=lambda index: (
            -math.dist(
                (float(index % nx), float((index // nx) % ny), float(index // (nx * ny))),
                center,
            ),
            index,
        ),
    )[:ion_count]
    return [*metal_indices, *ion_indices]


def _des_metal_plan_entries(
    config: DESConfig,
    *,
    box_lengths: tuple[float, float, float],
    ratio_units: int,
    added_ion_count: int = 0,
) -> list[dict[str, object]]:
    specs = _expanded_des_metal_specs(config)
    explicit_coordinates = [spec.coordinate for spec in specs if spec.coordinate is not None]
    for coordinate in explicit_coordinates:
        assert coordinate is not None
        if any(value < 0.0 or value >= length for value, length in zip(coordinate, box_lengths, strict=True)):
            raise ValueError(
                f"Explicit DES metal coordinate {coordinate} lies outside the periodic box {box_lengths}."
            )
    auto_count = sum(1 for spec in specs if spec.coordinate is None)
    if config.mixing_mode == DESMixingMode.REPLICATE:
        candidates = _replicate_reserved_cell_coordinates(
            config,
            ratio_units=ratio_units,
            box_lengths=box_lengths,
            added_ion_count=added_ion_count,
        )
        auto_coordinates = candidates[:auto_count]
    else:
        auto_coordinates = _auto_metal_coordinates(
            count=auto_count,
            box_lengths=box_lengths,
            spacing_angstrom=config.metal_spacing_angstrom,
            existing_coordinates=[item for item in explicit_coordinates if item is not None],
        )
    auto_index = 0
    entries: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        if spec.coordinate is None:
            coordinate = auto_coordinates[auto_index]
            auto_index += 1
            placement = "auto"
        else:
            coordinate = spec.coordinate
            placement = "explicit"
        entries.append(
            {
                "index": index,
                "element": spec.element,
                "charge": spec.charge,
                "residue_name": spec.residue_name,
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "z": float(coordinate[2]),
                "placement": placement,
            }
        )
    coordinates = [(float(item["x"]), float(item["y"]), float(item["z"])) for item in entries]
    for index, first in enumerate(coordinates):
        for second in coordinates[index + 1 :]:
            distance = _periodic_distance(first, second, box_lengths)
            if distance + 1.0e-6 < config.metal_spacing_angstrom:
                raise ValueError(
                    f"DES metal sites are only {distance:.2f} A apart under periodic boundaries; "
                    f"the configured minimum is {config.metal_spacing_angstrom:.2f} A."
                )
    return entries


def _des_c4_residue_names(
    config: DESConfig,
    metal_specs: list[_DESMetalSpec],
    added_ions: dict[str, int],
) -> list[str]:
    if not config.apply_1264:
        return []
    names = {spec.residue_name for spec in metal_specs}
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    # Native monatomic DES ions (BR in [N8888][Br], CL in choline chloride)
    # have published/default C4 entries and must be selected in addition to
    # the highly charged metal for the symmetric ion-ion C4 contribution.
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        names.update(residue.residue_name for residue in component.residues if residue.c4_capable)

    if config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL:
        # The bundled Duvail C4 table currently provides these additive
        # monatomic ions. K+ and Ca2+ retain their compatible 12-6 parameters.
        supported_added_ions = {"Na+", "Cl-", "Br-"}
    else:
        # ParmEd's SPC/E Li/Merz table includes the common monovalent ions and
        # Ca2+ exposed by SaltConfig.
        supported_added_ions = set(_DES_ADDED_ION_PROPERTIES)
    for ion_name, count in added_ions.items():
        if count > 0 and ion_name in supported_added_ions:
            names.add(_added_ion_residue(ion_name).residue_name)
    return sorted(names)


def _estimate_des_plan(config: DESConfig, ion_plan: dict[str, object] | None = None) -> DESPlan:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    added_ions = dict((ion_plan or {}).get("added_ions") or {})
    added_ion_count = sum(max(0, int(count)) for count in added_ions.values())
    ratio_units, component_counts = _component_counts_for_config(
        config,
        ref_data_dir,
        added_ion_count=added_ion_count,
    )
    residue_counts: dict[str, int] = {}
    total_atoms = 0
    total_residues = 0
    for component_key, component_count in component_counts.items():
        component = _component_definition(ref_data_dir, component_key)
        total_atoms += _component_atom_count(ref_data_dir, component) * component_count
        for residue in component.residues:
            residue_counts[residue.residue_name] = residue_counts.get(residue.residue_name, 0) + component_count
            total_residues += component_count
    metal_specs = _expanded_des_metal_specs(config)
    for spec in metal_specs:
        residue_counts[spec.residue_name] = residue_counts.get(spec.residue_name, 0) + 1
        total_atoms += 1
        total_residues += 1
    for ion_name, count in added_ions.items():
        residue = _added_ion_residue(str(ion_name))
        ion_count = max(0, int(count))
        residue_counts[residue.residue_name] = residue_counts.get(residue.residue_name, 0) + ion_count
        total_atoms += ion_count
        total_residues += ion_count
    if config.mixing_mode in {DESMixingMode.RANDOM_MIX, DESMixingMode.PACKMOL} and config.box_length_angstrom is not None:
        box_lengths = (config.box_length_angstrom, config.box_length_angstrom, config.box_length_angstrom)
    elif config.mixing_mode in {DESMixingMode.RANDOM_MIX, DESMixingMode.PACKMOL}:
        box_length = _packmol_box_length_for_ratio_units(
            config,
            ref_data_dir,
            ratio_units,
            added_ions=added_ions,
        )
        box_lengths = (box_length, box_length, box_length)
    else:
        occupied_lengths = _replicate_box_lengths(
            config,
            ref_data_dir,
            ratio_units,
            added_ion_count=added_ion_count,
        )
        cube_length = max(occupied_lengths)
        if config.box_length_angstrom is not None:
            cube_length = max(cube_length, config.box_length_angstrom)
        box_lengths = (cube_length, cube_length, cube_length)
    box_length = max(box_lengths)
    total_mass_da = _des_mass_da(
        config,
        ref_data_dir,
        ratio_units=ratio_units,
        added_ion_count_by_name=added_ions,
    )
    estimated_initial_density = _density_g_ml(
        total_mass_da,
        box_lengths[0] * box_lengths[1] * box_lengths[2],
    )
    c4_residue_names = _des_c4_residue_names(config, metal_specs, added_ions)
    c4_mask = ":" + ",".join(c4_residue_names) if c4_residue_names else None
    box_volume = box_lengths[0] * box_lengths[1] * box_lengths[2]
    metal_sites = _des_metal_plan_entries(
        config,
        box_lengths=box_lengths,
        ratio_units=ratio_units,
        added_ion_count=added_ion_count,
    )
    first_metal = metal_sites[0] if metal_sites else None
    return DESPlan(
        components=[_component_key_value(component) for component in config.components],
        ratios=list(config.ratios),
        component_counts=component_counts,
        residue_counts=dict(sorted(residue_counts.items())),
        ratio_units=ratio_units,
        box_length_angstrom=box_length,
        box_lengths_angstrom=box_lengths,
        box_volume_angstrom3=box_volume,
        total_residues=total_residues,
        total_atoms=total_atoms,
        mixing_mode=config.mixing_mode.value,
        replicate_order=config.replicate_order.value,
        size_mode=config.size_mode.value,
        spacing_angstrom=config.spacing_angstrom,
        packmol_tolerance_angstrom=config.packmol_tolerance_angstrom,
        packmol_fill_fraction=config.packmol_fill_fraction,
        target_density_g_ml=config.target_density_g_ml,
        estimated_initial_density_g_ml=estimated_initial_density,
        c4_mask=c4_mask,
        c4_residue_names=c4_residue_names,
        central_metal_element=None if first_metal is None else str(first_metal["element"]),
        central_metal_charge=None if first_metal is None else int(first_metal["charge"]),
        central_metal_residue_name=None if first_metal is None else str(first_metal["residue_name"]),
        metal_sites=metal_sites,
        charge_before_ions=float(
            (ion_plan or {}).get("charge_before_ions", estimate_des_net_charge(config, component_counts=component_counts))
        ),
        neutralizing_ions=dict((ion_plan or {}).get("neutralizing_ions") or {}),
        extra_ions=dict((ion_plan or {}).get("extra_ions") or {}),
        added_ions=added_ions,
        final_charge=float(
            (ion_plan or {}).get("final_charge", estimate_des_net_charge(config, component_counts=component_counts))
        ),
        actual_salt_concentration_m=(
            None
            if (ion_plan or {}).get("actual_salt_concentration_m") is None
            else float((ion_plan or {})["actual_salt_concentration_m"])
        ),
        neutralization_ion=(
            None
            if (ion_plan or {}).get("neutralization_ion") is None
            else str((ion_plan or {})["neutralization_ion"])
        ),
    )


def estimate_des_plan(config: DESConfig, salt_config: SaltConfig | None = None) -> DESPlan:
    base_plan = _estimate_des_plan(config)
    if salt_config is None or salt_config.mode == SaltMode.NONE:
        return base_plan
    plan = base_plan
    ion_plan: dict[str, object] | None = None
    for _iteration in range(4):
        ion_plan = _ion_plan_for_des(
            config,
            salt_config,
            volume_angstrom3=plan.box_volume_angstrom3,
            component_counts=plan.component_counts,
        )
        updated = _estimate_des_plan(config, ion_plan)
        if updated.added_ions == plan.added_ions and updated.component_counts == plan.component_counts:
            plan = updated
            break
        plan = updated
    return plan


def _atom_name_field(name: str) -> str:
    return name[:4].rjust(4)


def _pdb_serial_number(serial: int) -> int:
    return ((serial - 1) % 99999) + 1


def _pdb_residue_number(residue_number: int) -> int:
    return ((residue_number - 1) % 9999) + 1


def _pdb_chain_id(residue_number: int) -> str:
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return chain_ids[((residue_number - 1) // 9999) % len(chain_ids)]


def _format_pdb_atom(
    *,
    serial: int,
    atom: _AtomRecord,
    residue_name: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
) -> str:
    element = (atom.element or _infer_element(atom.name))[:2].rjust(2)
    pdb_serial = _pdb_serial_number(serial)
    pdb_residue_number = _pdb_residue_number(residue_number)
    chain_id = _pdb_chain_id(residue_number)
    return (
        f"HETATM{pdb_serial:5d} {_atom_name_field(atom.name)} {residue_name:>3s} {chain_id}{pdb_residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element}\n"
    )


def _central_metal_residue_name(config: DESConfig) -> str | None:
    specs = _expanded_des_metal_specs(config)
    if specs:
        return specs[0].residue_name
    if not config.central_metal_enabled or not config.central_metal_element:
        return None
    element = config.central_metal_element.title()
    charge = int(config.central_metal_charge or DEFAULT_TLEAP_METAL_CHARGES.get(element, 0) or 0)
    return _des_metal_residue_name(element, charge)


def _central_metal_atom(config: DESConfig) -> _AtomRecord | None:
    specs = _expanded_des_metal_specs(config)
    if not specs:
        return None
    spec = specs[0]
    return _AtomRecord(
        name=spec.residue_name,
        residue_name=spec.residue_name,
        x=0.0,
        y=0.0,
        z=0.0,
        element=spec.element,
    )


def _des_metal_atoms_for_plan(plan: DESPlan) -> list[tuple[_AtomRecord, float, float, float]]:
    atoms: list[tuple[_AtomRecord, float, float, float]] = []
    for entry in plan.metal_sites or []:
        residue_name = str(entry["residue_name"])
        element = str(entry["element"]).title()
        atoms.append(
            (
                _AtomRecord(
                    name=residue_name,
                    residue_name=residue_name,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    element=element,
                ),
                float(entry["x"]),
                float(entry["y"]),
                float(entry["z"]),
            )
        )
    return atoms


def _centered_coordinates(atoms: list[_AtomRecord], center: tuple[float, float, float]) -> list[tuple[_AtomRecord, float, float, float]]:
    cx = sum(atom.x for atom in atoms) / len(atoms)
    cy = sum(atom.y for atom in atoms) / len(atoms)
    cz = sum(atom.z for atom in atoms) / len(atoms)
    return [
        (atom, atom.x - cx + center[0], atom.y - cy + center[1], atom.z - cz + center[2])
        for atom in atoms
    ]


def _random_rotation_matrix(rng: random.Random) -> tuple[tuple[float, float, float], ...]:
    """Return a deterministic uniform random 3-D rotation matrix."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)),
        (2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)),
        (2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )


def _rotated_relative_coordinates(
    atoms: list[_AtomRecord],
    rotation: tuple[tuple[float, float, float], ...],
) -> list[tuple[float, float, float]]:
    cx = sum(atom.x for atom in atoms) / len(atoms)
    cy = sum(atom.y for atom in atoms) / len(atoms)
    cz = sum(atom.z for atom in atoms) / len(atoms)
    rotated: list[tuple[float, float, float]] = []
    for atom in atoms:
        relative = (atom.x - cx, atom.y - cy, atom.z - cz)
        rotated.append(
            tuple(
                sum(rotation[row][column] * relative[column] for column in range(3))
                for row in range(3)
            )
        )
    return rotated


def _random_mix_blocks(
    config: DESConfig,
    plan: DESPlan,
) -> list[tuple[DESResidueDefinition, list[_AtomRecord]]]:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    blocks = [
        (residue, _template_atoms(ref_data_dir, component, residue))
        for component, residue in _expanded_residue_sequence(config, plan.ratio_units, ref_data_dir)
    ]
    blocks.extend((residue, [atom]) for residue, atom in _added_ion_atoms(plan.added_ions))
    rng = random.Random(20240517)
    rng.shuffle(blocks)
    # Large rigid molecules are easiest to place first; their spatial positions
    # and orientations remain random, while small molecules/ions fill the gaps.
    blocks.sort(key=lambda item: -len(item[1]))
    return blocks


def _random_mix_pdb_text(config: DESConfig, plan: DESPlan) -> str:
    """Fast in-process rigid-body packing with periodic atom-contact checks."""
    box_lengths = plan.box_lengths_angstrom
    minimum_distance = config.spacing_angstrom
    metal_contact_limit = max(1.5, min(2.0, config.packmol_tolerance_angstrom))
    search_cutoff = max(minimum_distance, metal_contact_limit)
    cell_counts = tuple(max(1, int(length // search_cutoff)) for length in box_lengths)
    cells: dict[
        tuple[int, int, int],
        list[tuple[tuple[float, float, float], float]],
    ] = {}

    def wrapped(coordinate: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(value % length for value, length in zip(coordinate, box_lengths, strict=True))

    def cell_for(coordinate: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(
            int(value / length * count) % count
            for value, length, count in zip(wrapped(coordinate), box_lengths, cell_counts, strict=True)
        )

    def collides(coordinates: list[tuple[float, float, float]]) -> bool:
        for coordinate in coordinates:
            point = wrapped(coordinate)
            base = cell_for(point)
            neighbor_cells = {
                (
                    (base[0] + dx) % cell_counts[0],
                    (base[1] + dy) % cell_counts[1],
                    (base[2] + dz) % cell_counts[2],
                )
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            }
            for neighbor_cell in neighbor_cells:
                for other, other_exclusion_distance in cells.get(neighbor_cell, []):
                    distance2 = 0.0
                    for value, other_value, length in zip(point, other, box_lengths, strict=True):
                        delta = abs(value - other_value)
                        delta = min(delta, length - delta)
                        distance2 += delta * delta
                    required_distance = max(minimum_distance, other_exclusion_distance)
                    if distance2 + 1.0e-12 < required_distance * required_distance:
                        return True
        return False

    def register(
        coordinates: list[tuple[float, float, float]],
        exclusion_distance: float,
    ) -> None:
        for coordinate in coordinates:
            point = wrapped(coordinate)
            cells.setdefault(cell_for(point), []).append((point, exclusion_distance))

    # Metals remain at their requested/centered coordinates and act as fixed
    # excluded-volume sites during the random packing.
    register(
        [(x, y, z) for _atom, x, y, z in _des_metal_atoms_for_plan(plan)],
        metal_contact_limit,
    )

    rng = random.Random(20240517)
    packed: list[tuple[DESResidueDefinition, list[_AtomRecord], list[tuple[float, float, float]]]] = []
    boundary_margin = 0.5 * minimum_distance
    blocks = _random_mix_blocks(config, plan)
    for block_index, (residue, atoms) in enumerate(blocks, start=1):
        accepted: list[tuple[float, float, float]] | None = None
        max_attempts = 2400
        for _attempt in range(max_attempts):
            rotation = _random_rotation_matrix(rng)
            relative = _rotated_relative_coordinates(atoms, rotation)
            minima = tuple(min(point[axis] for point in relative) for axis in range(3))
            maxima = tuple(max(point[axis] for point in relative) for axis in range(3))
            center_ranges = [
                (boundary_margin - minima[axis], box_lengths[axis] - boundary_margin - maxima[axis])
                for axis in range(3)
            ]
            if any(high <= low for low, high in center_ranges):
                raise ValueError(
                    f"The random-mix DES box is too small to contain residue {residue.residue_name}. "
                    "Lower the initial density or use a larger explicit box."
                )
            center = tuple(rng.uniform(low, high) for low, high in center_ranges)
            coordinates = [
                tuple(center[axis] + point[axis] for axis in range(3))
                for point in relative
            ]
            if not collides(coordinates):
                accepted = coordinates
                break
        if accepted is None:
            raise RuntimeError(
                f"Fast random mixing could not place residue {residue.residue_name} ({block_index}/"
                f"{len(blocks)}) at {plan.estimated_initial_density_g_ml:.3f} g/mL "
                f"and {minimum_distance:.2f} A spacing. Lower the initial density/spacing or choose Packmol."
            )
        register(accepted, minimum_distance)
        packed.append((residue, atoms, accepted))

    serial = 1
    residue_number = 1
    lines = ["COMPND    SIMPLE fast random-mix DES\n", "AUTHOR    GENERATED BY SIMPLE\n"]
    for residue, atoms, coordinates in packed:
        for atom, coordinate in zip(atoms, coordinates, strict=True):
            lines.append(
                _format_pdb_atom(
                    serial=serial,
                    atom=atom,
                    residue_name=residue.residue_name,
                    residue_number=residue_number,
                    x=coordinate[0],
                    y=coordinate[1],
                    z=coordinate[2],
                )
            )
            serial += 1
        lines.append("TER\n")
        residue_number += 1
    for metal_atom, x, y, z in _des_metal_atoms_for_plan(plan):
        lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=metal_atom,
                residue_name=metal_atom.residue_name,
                residue_number=residue_number,
                x=x,
                y=y,
                z=z,
            )
        )
        serial += 1
        residue_number += 1
        lines.append("TER\n")
    lines.append("END\n")
    return "".join(lines)


def _write_random_mixture_pdb(config: DESConfig, plan: DESPlan, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_random_mix_pdb_text(config, plan), encoding="utf-8")
    return output_path


def _grid_added_ion_atoms_for_plan(
    config: DESConfig,
    plan: DESPlan,
) -> list[tuple[DESResidueDefinition, _AtomRecord, float, float, float]]:
    added_ion_count = sum((plan.added_ions or {}).values())
    if added_ion_count <= 0:
        return []
    reserved_coordinates = _replicate_reserved_cell_coordinates(
        config,
        ratio_units=plan.ratio_units,
        box_lengths=plan.box_lengths_angstrom,
        added_ion_count=added_ion_count,
    )
    auto_metal_coordinates = {
        (round(float(item["x"]), 6), round(float(item["y"]), 6), round(float(item["z"]), 6))
        for item in (plan.metal_sites or [])
        if item.get("placement") == "auto"
    }
    ion_coordinates = [
        coordinate
        for coordinate in reserved_coordinates
        if tuple(round(value, 6) for value in coordinate) not in auto_metal_coordinates
    ]
    ion_atoms = _added_ion_atoms(plan.added_ions)
    if len(ion_coordinates) < len(ion_atoms):
        raise RuntimeError("The replicate DES layout did not reserve enough non-overlapping counter-ion cells.")
    return [
        (residue, ion_atom, x, y, z)
        for (residue, ion_atom), (x, y, z) in zip(ion_atoms, ion_coordinates, strict=False)
    ]


def _write_grid_mixture_pdb(config: DESConfig, plan: DESPlan, output_path: Path) -> Path:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    added_ion_count = sum((plan.added_ions or {}).values())
    layout = _replicate_grid_layout(
        config,
        ref_data_dir,
        plan.ratio_units,
        added_ion_count=added_ion_count,
    )
    offset = (
        max((plan.box_lengths_angstrom[0] - layout.occupied_lengths[0]) / 2.0, 0.0),
        max((plan.box_lengths_angstrom[1] - layout.occupied_lengths[1]) / 2.0, 0.0),
        max((plan.box_lengths_angstrom[2] - layout.occupied_lengths[2]) / 2.0, 0.0),
    )
    serial = 1
    residue_number = 1
    lines: list[str] = ["COMPND    SIMPLE DES mixture\n", "AUTHOR    GENERATED BY SIMPLE\n"]
    capacity = layout.dimensions[0] * layout.dimensions[1] * layout.dimensions[2]
    reserved_indices = set(
        _replicate_reserved_cell_indices(
            config,
            layout=layout,
            added_ion_count=added_ion_count,
        )
    )
    molecule_indices = [index for index in range(capacity) if index not in reserved_indices]
    for cell_index, ((component, residue), _side) in zip(
        molecule_indices,
        zip(layout.sequence, layout.side_lengths, strict=True),
        strict=False,
    ):
        center = _replicate_cell_coordinate(cell_index, layout=layout, offset=offset)
        atoms = _template_atoms(ref_data_dir, component, residue)
        for atom, x, y, z in _centered_coordinates(atoms, center):
            lines.append(
                _format_pdb_atom(
                    serial=serial,
                    atom=atom,
                    residue_name=residue.residue_name,
                    residue_number=residue_number,
                    x=x,
                    y=y,
                    z=z,
                )
            )
            serial += 1
        lines.append("TER\n")
        residue_number += 1
    for metal_atom, x, y, z in _des_metal_atoms_for_plan(plan):
        lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=metal_atom,
                residue_name=metal_atom.residue_name,
                residue_number=residue_number,
                x=x,
                y=y,
                z=z,
            )
        )
        serial += 1
        lines.append("TER\n")
        residue_number += 1
    for residue, ion_atom, x, y, z in _grid_added_ion_atoms_for_plan(config, plan):
        lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=ion_atom,
                residue_name=residue.residue_name,
                residue_number=residue_number,
                x=x,
                y=y,
                z=z,
            )
        )
        serial += 1
        lines.append("TER\n")
        residue_number += 1
    lines.append("END\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _write_single_residue_pdb(
    residue: DESResidueDefinition,
    output_path: Path,
    *,
    atoms: list[_AtomRecord] | None = None,
) -> Path:
    template_atoms = atoms or _synthetic_ion_atoms(residue)
    lines = ["COMPND    SIMPLE DES residue template\n", "AUTHOR    GENERATED BY SIMPLE\n"]
    for serial, atom in enumerate(template_atoms, start=1):
        lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=atom,
                residue_name=residue.residue_name,
                residue_number=1,
                x=0.0,
                y=0.0,
                z=0.0,
            )
        )
    lines.extend(["TER\n", "END\n"])
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _append_central_metal_to_existing_pdb(config: DESConfig, plan: DESPlan, output_path: Path) -> None:
    metal_atoms = _des_metal_atoms_for_plan(plan)
    if not metal_atoms:
        return
    lines = output_path.read_text(encoding="utf-8").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    serial = len(atom_lines) + 1
    residue_numbers: list[int] = []
    for line in atom_lines:
        try:
            residue_numbers.append(int(line[22:26]))
        except ValueError:
            continue
    residue_number = (max(residue_numbers) + 1) if residue_numbers else 1
    metal_lines: list[str] = []
    for metal_atom, x, y, z in metal_atoms:
        metal_lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=metal_atom,
                residue_name=metal_atom.residue_name,
                residue_number=residue_number,
                x=x,
                y=y,
                z=z,
            ).rstrip("\n")
        )
        metal_lines.append("TER")
        serial += 1
        residue_number += 1
    if lines and lines[-1].strip() == "END":
        lines = lines[:-1] + metal_lines + ["END"]
    else:
        lines.extend(metal_lines + ["END"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_des_metal_contacts(
    config: DESConfig,
    plan: DESPlan,
    mixture_pdb: Path,
) -> dict[str, float | None]:
    metal_sites = list(plan.metal_sites or [])
    if not metal_sites:
        return {"minimum_metal_metal_distance_angstrom": None, "minimum_metal_other_distance_angstrom": None}
    atom_records: list[tuple[str, tuple[float, float, float]]] = []
    for line in mixture_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_records.append(
            (
                line[17:20].strip(),
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
        )
    metal_coordinates = [
        (float(item["x"]), float(item["y"]), float(item["z"]))
        for item in metal_sites
    ]
    minimum_metal_metal: float | None = None
    for index, first in enumerate(metal_coordinates):
        for second in metal_coordinates[index + 1 :]:
            distance = _periodic_distance(first, second, plan.box_lengths_angstrom)
            minimum_metal_metal = distance if minimum_metal_metal is None else min(minimum_metal_metal, distance)
    minimum_metal_other: float | None = None
    contact_limit = max(1.5, min(2.0, config.packmol_tolerance_angstrom))
    for site, coordinate in zip(metal_sites, metal_coordinates, strict=True):
        residue_name = str(site["residue_name"])
        matching = [
            (index, _periodic_distance(coordinate, atom_coordinate, plan.box_lengths_angstrom))
            for index, (atom_residue, atom_coordinate) in enumerate(atom_records)
            if atom_residue == residue_name
        ]
        if not matching:
            raise RuntimeError(f"Placed DES metal residue {residue_name} was not found in {mixture_pdb}.")
        own_index, own_distance = min(matching, key=lambda item: item[1])
        if own_distance > 0.05:
            raise RuntimeError(
                f"Placed DES metal {residue_name} moved unexpectedly by {own_distance:.3f} A during system assembly."
            )
        for index, (_atom_residue, atom_coordinate) in enumerate(atom_records):
            if index == own_index:
                continue
            distance = _periodic_distance(coordinate, atom_coordinate, plan.box_lengths_angstrom)
            minimum_metal_other = distance if minimum_metal_other is None else min(minimum_metal_other, distance)
            if distance + 1.0e-6 < contact_limit:
                raise ValueError(
                    f"Unsafe initial DES geometry: metal {residue_name} has another atom only {distance:.3f} A away "
                    f"(minimum allowed {contact_limit:.3f} A). Regenerate with automatic placement, increase the "
                    "box/spacing, or correct the explicit XYZ coordinates before running MD."
                )
    return {
        "minimum_metal_metal_distance_angstrom": minimum_metal_metal,
        "minimum_metal_other_distance_angstrom": minimum_metal_other,
    }


def _validate_inter_residue_contacts(
    mixture_pdb: Path,
    box_lengths: tuple[float, float, float],
    *,
    minimum_allowed_angstrom: float = 1.2,
    search_cutoff_angstrom: float = 3.0,
) -> dict[str, float | None]:
    """Detect short periodic contacts without an all-pairs distance matrix."""
    atoms: list[tuple[tuple[str, str], tuple[float, float, float], str]] = []
    for line in mixture_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms.append(
            (
                (line[21:22], line[22:26]),
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                f"{line[17:20].strip()}:{line[12:16].strip()}",
            )
        )
    cell_counts = tuple(max(1, int(length // search_cutoff_angstrom)) for length in box_lengths)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for atom_index, (_residue, coordinate, _label) in enumerate(atoms):
        cell = tuple(
            int((value % length) / length * count) % count
            for value, length, count in zip(coordinate, box_lengths, cell_counts, strict=True)
        )
        cells.setdefault(cell, []).append(atom_index)

    minimum: float | None = None
    minimum_pair: tuple[str, str] | None = None
    for first_index, (first_residue, first_coordinate, first_label) in enumerate(atoms):
        first_cell = tuple(
            int((value % length) / length * count) % count
            for value, length, count in zip(first_coordinate, box_lengths, cell_counts, strict=True)
        )
        neighbor_cells = {
            (
                (first_cell[0] + dx) % cell_counts[0],
                (first_cell[1] + dy) % cell_counts[1],
                (first_cell[2] + dz) % cell_counts[2],
            )
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        }
        for cell in neighbor_cells:
            for second_index in cells.get(cell, []):
                if second_index <= first_index:
                    continue
                second_residue, second_coordinate, second_label = atoms[second_index]
                if first_residue == second_residue:
                    continue
                distance = _periodic_distance(first_coordinate, second_coordinate, box_lengths)
                if distance > search_cutoff_angstrom:
                    continue
                if minimum is None or distance < minimum:
                    minimum = distance
                    minimum_pair = (first_label, second_label)
    if minimum is not None and minimum + 1.0e-6 < minimum_allowed_angstrom:
        pair_text = " / ".join(minimum_pair or ("unknown", "unknown"))
        raise ValueError(
            f"Unsafe initial DES geometry: different residues contain atoms only {minimum:.3f} A apart "
            f"under periodic boundaries ({pair_text}); minimum allowed is {minimum_allowed_angstrom:.3f} A."
        )
    return {
        "minimum_inter_residue_distance_angstrom": minimum,
        "inter_residue_search_cutoff_angstrom": search_cutoff_angstrom,
    }


def _tleap_parameter_errors(output_text: str) -> list[str]:
    patterns = (
        "could not find bond parameter",
        "could not find angle parameter",
        "could not find dihedral parameter",
        "does not have a type",
        "parameter file was not saved",
        "fatal error",
    )
    return [
        line.strip()
        for line in output_text.splitlines()
        if any(pattern in line.lower() for pattern in patterns)
    ]


def _pdb_atom_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )


def _copy_residue_assets(config: DESConfig, output_dir: Path) -> dict[str, Path]:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    inputs_dir = output_dir / "des_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        for residue in component.residues:
            for kind in ("pdb", "frcmod", "lib"):
                name = getattr(residue, kind)
                if not name:
                    continue
                source = _residue_file(ref_data_dir, component, residue, kind)
                target = inputs_dir / source.name
                if str(source.resolve()) != str(target.resolve()):
                    shutil.copy2(source, target)
                copied[f"{residue.residue_name}_{kind}"] = target
    return copied


def _write_packmol_input(config: DESConfig, plan: DESPlan, copied_assets: dict[str, Path], output_dir: Path) -> Path:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    packmol_input = output_dir / "packmol.inp"
    # Packmol 20.14.x (still common on HPC systems) does not recognize the
    # newer top-level `pbc` keyword. Keep every flexible atom half a tolerance
    # away from each face instead. Therefore atoms on opposite faces also
    # remain at least one full Packmol tolerance apart under minimum-image PBC.
    boundary_margin = 0.5 * config.packmol_tolerance_angstrom
    lower_bounds = (boundary_margin, boundary_margin, boundary_margin)
    upper_bounds = tuple(length - boundary_margin for length in plan.box_lengths_angstrom)
    if any(upper <= lower for lower, upper in zip(lower_bounds, upper_bounds, strict=True)):
        raise ValueError(
            "The DES box is too small for the requested Packmol tolerance and periodic boundary margin."
        )
    inside_box = (
        f"{lower_bounds[0]:.3f} {lower_bounds[1]:.3f} {lower_bounds[2]:.3f} "
        f"{upper_bounds[0]:.3f} {upper_bounds[1]:.3f} {upper_bounds[2]:.3f}"
    )
    lines = [
        f"tolerance {config.packmol_tolerance_angstrom:.3f}\n",
        "seed 20240517\n",
        "filetype pdb\n",
        "output des_mixture.pdb\n\n",
    ]
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        count = plan.component_counts[_component_count_key(component_key)]
        for residue in component.residues:
            if residue.pdb is None:
                ion_pdb = output_dir / "des_inputs" / f"{residue.residue_name}.pdb"
                _write_single_residue_pdb(
                    residue,
                    ion_pdb,
                    atoms=_template_atoms(ref_data_dir, component, residue),
                )
                structure_path = ion_pdb
            else:
                structure_path = copied_assets[f"{residue.residue_name}_pdb"]
            lines.extend(
                [
                    f"structure {_tleap_path(structure_path, output_dir)}\n",
                    f"  number {count}\n",
                    f"  inside box {inside_box}\n",
                    "end structure\n\n",
                ]
            )
    metal_template_paths: dict[str, Path] = {}
    for metal_atom, x, y, z in _des_metal_atoms_for_plan(plan):
        structure_path = metal_template_paths.get(metal_atom.residue_name)
        if structure_path is None:
            structure_path = output_dir / "des_inputs" / f"{metal_atom.residue_name}_metal.pdb"
            residue = DESResidueDefinition(
                metal_atom.residue_name,
                None,
                atom_name=metal_atom.name,
                element=metal_atom.element,
            )
            _write_single_residue_pdb(residue, structure_path, atoms=[metal_atom])
            metal_template_paths[metal_atom.residue_name] = structure_path
        lines.extend(
            [
                f"structure {_tleap_path(structure_path, output_dir)}\n",
                "  number 1\n",
                "  center\n",
                f"  fixed {x:.3f} {y:.3f} {z:.3f} 0.0 0.0 0.0\n",
                "end structure\n\n",
            ]
        )
    for ion_name, count in sorted((plan.added_ions or {}).items()):
        if count <= 0:
            continue
        residue = _added_ion_residue(ion_name)
        structure_path = output_dir / "des_inputs" / f"{residue.residue_name}_added_ion.pdb"
        _write_single_residue_pdb(residue, structure_path)
        lines.extend(
            [
                f"structure {_tleap_path(structure_path, output_dir)}\n",
                f"  number {count}\n",
                f"  inside box {inside_box}\n",
                "end structure\n\n",
            ]
        )
    packmol_input.write_text("".join(lines), encoding="utf-8")
    return packmol_input


def _packmol_residue_blocks(
    config: DESConfig,
    plan: DESPlan,
    ref_data_dir: Path,
) -> list[tuple[DESResidueDefinition, list[_AtomRecord]]]:
    blocks: list[tuple[DESResidueDefinition, list[_AtomRecord]]] = []
    for component_key in config.components:
        component = _component_definition(ref_data_dir, component_key)
        count = plan.component_counts[_component_count_key(component_key)]
        for residue in component.residues:
            template_atoms = _template_atoms(ref_data_dir, component, residue)
            blocks.extend((residue, template_atoms) for _ in range(count))
    for metal_atom, _x, _y, _z in _des_metal_atoms_for_plan(plan):
        residue = DESResidueDefinition(
            metal_atom.residue_name,
            None,
            atom_name=metal_atom.name,
            element=metal_atom.element,
        )
        blocks.append((residue, [metal_atom]))
    for residue, ion_atom in _added_ion_atoms(plan.added_ions):
        blocks.append((residue, [ion_atom]))
    return blocks


def _sanitize_packmol_mixture_pdb(config: DESConfig, plan: DESPlan, output_path: Path) -> Path:
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    packmol_atoms = _read_pdb_atoms(output_path)
    blocks = _packmol_residue_blocks(config, plan, ref_data_dir)
    expected_atoms = sum(len(template_atoms) for _residue, template_atoms in blocks)
    if len(packmol_atoms) != expected_atoms:
        raise RuntimeError(
            "Packmol wrote an unexpected atom count. "
            f"Expected {expected_atoms} atoms from the DES plan, found {len(packmol_atoms)} in {output_path}."
        )

    serial = 1
    residue_number = 1
    cursor = 0
    lines: list[str] = [
        "COMPND    SIMPLE DES mixture from Packmol\n",
        "AUTHOR    GENERATED BY SIMPLE\n",
    ]
    for residue, template_atoms in blocks:
        placed_atoms = packmol_atoms[cursor : cursor + len(template_atoms)]
        cursor += len(template_atoms)
        for template_atom, placed_atom in zip(template_atoms, placed_atoms, strict=True):
            atom = _AtomRecord(
                name=template_atom.name,
                residue_name=residue.residue_name,
                x=placed_atom.x,
                y=placed_atom.y,
                z=placed_atom.z,
                element=template_atom.element,
            )
            lines.append(
                _format_pdb_atom(
                    serial=serial,
                    atom=atom,
                    residue_name=residue.residue_name,
                    residue_number=residue_number,
                    x=placed_atom.x,
                    y=placed_atom.y,
                    z=placed_atom.z,
                )
            )
            serial += 1
        lines.append("TER\n")
        residue_number += 1
    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _resolve_1264_polfile(config: DESConfig, output_dir: Path) -> Path | None:
    if config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL:
        source = opc_duvail_polarizability_file()
        if source.exists():
            target = output_dir / source.name
            shutil.copy2(source, target)
            return target
        return None

    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    candidates = [
        ref_data_dir / "N8888_Br_Hexanoic_Acid" / "lj_1264_pol_augmented.dat",
        ref_data_dir / "Choline_Cl_Ethylene_glycol" / "lj_1264_pol_augmented.dat",
    ]
    for candidate in candidates:
        if candidate.exists():
            target = output_dir / "lj_1264_pol_augmented.dat"
            shutil.copy2(candidate, target)
            return target
    return None


def _resolve_1264_c4file(config: DESConfig, output_dir: Path) -> Path | None:
    if config.c4_parameter_set != DESC4ParameterSet.OPC_DUVAIL:
        return None
    source = opc_duvail_c4_file()
    if not source.exists():
        return None
    target = output_dir / source.name
    shutil.copy2(source, target)
    return target


def _ion_1264_parameter_line(config: DESConfig, amber_env: AmberEnvironment, output_dir: Path) -> str:
    if config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL:
        source = opc_duvail_ion_frcmod()
        if source.exists():
            target = output_dir / source.name
            shutil.copy2(source, target)
            return f"loadamberparams {_tleap_path(target, output_dir)}"
        return "loadamberparams frcmod.ionslm_1264_opc"

    candidates = amber_env.matching_monovalent_1264_files("spce")
    if candidates:
        return f"loadamberparams {candidates[0].as_posix()}"
    return "loadamberparams frcmod.ions1lm_1264_spce"


def _tleap_path(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _render_tleap(
    config: DESConfig,
    plan: DESPlan,
    copied_assets: dict[str, Path],
    output_dir: Path,
    amber_env: AmberEnvironment,
) -> str:
    water_source = (
        "source leaprc.water.opc"
        if config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL
        else "source leaprc.water.spce"
    )
    lines = [
        "source leaprc.gaff2",
        water_source,
    ]
    if plan.c4_mask:
        lines.append(_ion_1264_parameter_line(config, amber_env, output_dir))
    loaded_frcmods: set[Path] = set()
    loaded_libs: set[Path] = set()
    for component_key in config.components:
        component = _component_definition(resolve_ref_data_dir(config.ref_data_dir), component_key)
        for residue in component.residues:
            frcmod = copied_assets.get(f"{residue.residue_name}_frcmod")
            lib = copied_assets.get(f"{residue.residue_name}_lib")
            if frcmod is not None and frcmod not in loaded_frcmods:
                lines.append(f"loadamberparams {_tleap_path(frcmod, output_dir)}")
                loaded_frcmods.add(frcmod)
            if lib is not None and lib not in loaded_libs:
                lines.append(f"loadoff {_tleap_path(lib, output_dir)}")
                loaded_libs.add(lib)
    lines.extend(
        [
            "DES = loadpdb des_mixture.pdb",
            "check DES",
            "charge DES",
            "set DES box "
            f"{{{plan.box_lengths_angstrom[0]:.3f} {plan.box_lengths_angstrom[1]:.3f} {plan.box_lengths_angstrom[2]:.3f}}}",
            "savePDB DES system.pdb",
            "saveAmberParm DES system.prmtop system.inpcrd",
            "quit",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_parmed_1264(
    config: DESConfig,
    plan: DESPlan,
    output_dir: Path,
    polfile: Path | None,
    c4file: Path | None,
) -> str:
    polfile_clause = f" polfile {_tleap_path(polfile, output_dir)}" if polfile is not None else ""
    c4file_clause = f" c4file {_tleap_path(c4file, output_dir)}" if c4file is not None else ""
    water_model = "OPC" if config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL else "SPCE"
    if c4file is not None:
        command = f"add12_6_4 {plan.c4_mask}{polfile_clause}{c4file_clause}"
    else:
        command = f"add12_6_4 {plan.c4_mask} watermodel {water_model}{polfile_clause}"
    return (
        "setOverwrite True\n"
        f"{command}\n"
        "outparm system.prmtop system.inpcrd\n"
        "quit\n"
    )


def _run_packmol(packmol_path: str, input_path: Path, output_dir: Path) -> None:
    with input_path.open("r", encoding="utf-8") as handle:
        result = subprocess.run(
            [packmol_path],
            cwd=str(output_dir),
            stdin=handle,
            check=False,
            capture_output=True,
            text=True,
        )
    log_path = output_dir / "packmol.log"
    log_path.write_text(
        "\n".join(
            [
                f"$ {packmol_path} < {input_path}",
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Packmol failed ({result.returncode}). See log: {log_path}")


def _read_inpcrd_box_lengths(path: Path) -> tuple[float, float, float] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        values = [float(token) for token in lines[-1].split()]
    except ValueError:
        return None
    if len(values) < 3:
        return None
    return (values[0], values[1], values[2])


def build_des_system(
    *,
    des_config: DESConfig,
    amber_env: AmberEnvironment,
    output_dir: Path,
    dry_run: bool,
    system_config: SystemConfig | None = None,
) -> LeapBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    warnings: list[str] = []
    salt_config = system_config.salt if system_config is not None else SaltConfig()
    plan = estimate_des_plan(des_config, salt_config)
    if plan.estimated_initial_density_g_ml < 0.25:
        warnings.append(
            "The planned DES initial density is only "
            f"{plan.estimated_initial_density_g_ml:.3f} g/mL. This is a gas-like starting box; "
            "use Packmol with the default density-based sizing for condensed-phase DES MD."
        )
    copied_assets = _copy_residue_assets(des_config, output_dir)
    charge_normalizations: list[dict[str, object]] = []
    for key, path in copied_assets.items():
        if not key.endswith("_lib"):
            continue
        normalized = _normalize_library_charge_sum(path)
        if normalized is None:
            continue
        original, corrected = normalized
        charge_normalizations.append(
            {"file": str(path), "original_charge": original, "normalized_charge": corrected}
        )
        warnings.append(
            f"Normalized small Amber library charge-rounding drift in {path.name}: "
            f"{original:+.8f} -> {corrected:+.8f}."
        )

    mixture_pdb = output_dir / "des_mixture.pdb"
    if des_config.mixing_mode == DESMixingMode.PACKMOL:
        packmol_path = shutil.which("packmol")
        if packmol_path is None:
            raise RuntimeError(
                "Packmol mixing mode was selected, but `packmol` was not found on PATH. "
                "Install it with the provided conda environment (`conda env create -f environment.yml`) "
                "or choose the fast random-mix mode."
            )
        packmol_input = _write_packmol_input(des_config, plan, copied_assets, output_dir)
        commands.append([packmol_path, "<", str(packmol_input)])
        if dry_run:
            warnings.append("Dry-run: Packmol input was written, but Packmol was not executed.")
            _write_random_mixture_pdb(des_config, plan, mixture_pdb)
        else:
            ensure_execution_host(dry_run=False)
            _run_packmol(packmol_path, packmol_input, output_dir)
            if not mixture_pdb.exists():
                raise RuntimeError(f"Packmol completed but did not write the expected output: {mixture_pdb}")
            _sanitize_packmol_mixture_pdb(des_config, plan, mixture_pdb)
    else:
        _write_random_mixture_pdb(des_config, plan, mixture_pdb)
    geometry_checks = {
        **_validate_des_metal_contacts(des_config, plan, mixture_pdb),
        **_validate_inter_residue_contacts(mixture_pdb, plan.box_lengths_angstrom),
    }

    tleap_script = _render_tleap(des_config, plan, copied_assets, output_dir, amber_env)
    tleap_path = output_dir / "tleap.in"
    tleap_path.write_text(tleap_script, encoding="utf-8")
    tleap_command = ["tleap", "-f", str(tleap_path)]
    commands.append(tleap_command)
    actual_tleap_charge: float | None = None
    actual_tleap_atom_count: int | None = None
    if not dry_run:
        ensure_execution_host(dry_run=False)
        tleap_result = run_command(tleap_command, cwd=output_dir, log_path=output_dir / "tleap.log")
        tleap_output = f"{tleap_result.stdout}\n{tleap_result.stderr}"
        parameter_errors = _tleap_parameter_errors(tleap_output)
        if parameter_errors:
            raise RuntimeError(
                "tLEaP reported missing or invalid DES force-field parameters:\n- "
                + "\n- ".join(parameter_errors[:20])
            )
        actual_tleap_atom_count = _pdb_atom_count(output_dir / "system.pdb")
        if actual_tleap_atom_count != plan.total_atoms:
            raise RuntimeError(
                "tLEaP did not preserve the complete DES mixture: "
                f"planned {plan.total_atoms} atoms but system.pdb contains {actual_tleap_atom_count}. "
                "This usually indicates duplicate/mismatched library atom names or an invalid residue template."
            )
        charge_values = extract_total_charges(tleap_output)
        if charge_values:
            actual_tleap_charge = charge_values[-1]
            if salt_config.mode != SaltMode.NONE and abs(actual_tleap_charge) > 0.05:
                raise RuntimeError(
                    "DES counter-ion insertion did not produce a neutral Amber topology: "
                    f"tLEaP reported charge {actual_tleap_charge:+.6f}. Inspect tleap.log and des_mixture.pdb."
                )
        else:
            warnings.append(
                "tLEaP completed, but its final DES charge could not be parsed from tleap.log; "
                "verify the `charge DES` output before MD."
            )
    actual_box_lengths = _read_inpcrd_box_lengths(output_dir / "system.inpcrd")

    c4_script_path: str | None = None
    c4_applied = False
    if plan.c4_mask:
        polfile = _resolve_1264_polfile(des_config, output_dir)
        c4file = _resolve_1264_c4file(des_config, output_dir)
        if polfile is None:
            warnings.append("12-6-4 was requested, but no REF_DATA lj_1264_pol_augmented.dat file was found.")
        else:
            c4_script = _render_parmed_1264(des_config, plan, output_dir, polfile, c4file)
            c4_path = output_dir / "parmed_1264.in"
            c4_path.write_text(c4_script, encoding="utf-8")
            c4_script_path = str(c4_path)
            parmed_status = amber_env.binaries.get("parmed")
            parmed_path = parmed_status.path if parmed_status is not None else None
            parmed_command = [
                str(parmed_path or "parmed"),
                "-i",
                str(c4_path),
                "-p",
                str(output_dir / "system.prmtop"),
                "-c",
                str(output_dir / "system.inpcrd"),
            ]
            commands.append(parmed_command)
            if dry_run:
                warnings.append("Dry-run: generated a DES ParmEd add12_6_4 helper script but did not execute it.")
            else:
                if parmed_path is None:
                    raise RuntimeError(
                        "12-6-4 C4 post-processing requires ParmEd (`parmed`) on PATH. "
                        f"A helper script was written to {c4_path}."
                    )
                run_command(parmed_command, cwd=output_dir, log_path=output_dir / "parmed_1264.log")
                c4_applied = True
                actual_box_lengths = _read_inpcrd_box_lengths(output_dir / "system.inpcrd") or actual_box_lengths

    box_lengths = actual_box_lengths or plan.box_lengths_angstrom
    volume_angstrom3 = box_lengths[0] * box_lengths[1] * box_lengths[2]

    write_json(
        output_dir / "des_manifest.json",
        {
            "plan": plan.to_dict(),
            "copied_assets": {key: str(path) for key, path in copied_assets.items()},
            "mixture_pdb": str(mixture_pdb),
            "charge_normalizations": charge_normalizations,
            "initial_geometry_checks": geometry_checks,
            "actual_tleap_charge": actual_tleap_charge,
            "actual_tleap_atom_count": actual_tleap_atom_count,
        },
    )
    return LeapBuildResult(
        script_path=str(tleap_path),
        output_files={
            "pdb": str(output_dir / "system.pdb"),
            "prmtop": str(output_dir / "system.prmtop"),
            "inpcrd": str(output_dir / "system.inpcrd"),
        },
        warnings=warnings,
        commands=commands,
        water_count=None,
        extra_ions=dict(plan.extra_ions or {}),
        neutralizing_ions=dict(plan.neutralizing_ions or {}),
        added_ions=dict(plan.added_ions or {}),
        volume_angstrom3=volume_angstrom3,
        charge_before_ions=plan.charge_before_ions,
        final_charge=actual_tleap_charge if actual_tleap_charge is not None else plan.final_charge,
        salt_formula_units=salt_formula_units(dict(plan.extra_ions or {}), salt_config.kind),
        actual_salt_concentration_m=plan.actual_salt_concentration_m,
        c4_script_path=c4_script_path,
        c4_mask=plan.c4_mask,
        c4_applied=c4_applied,
        box_lengths_angstrom=box_lengths,
        system_metadata={
            "workflow_type": "des",
            "des": plan.to_dict(),
            "actual_tleap_charge": actual_tleap_charge,
            "actual_tleap_atom_count": actual_tleap_atom_count,
            "box_lengths_source": "inpcrd" if actual_box_lengths is not None else "planned",
        },
    )
