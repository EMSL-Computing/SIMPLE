from __future__ import annotations

import json
import ipaddress
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from amber_metallo.amber.leap import (
    DEFAULT_TLEAP_METAL_CHARGES,
    allowed_metal_charges,
    c4_parameter_set_supports_metal_charge,
)
from amber_metallo.config import (
    BoxShape,
    ChargeMethod,
    DESComponent,
    DESConfig,
    DESC4ParameterSet,
    DESMixingMode,
    DESReplicateOrder,
    DESSizeMode,
    InputConfig,
    InputSource,
    LigandsConfig,
    LigandMode,
    MDConfig,
    MetalAnchorMode,
    MetalChargeAssignment,
    MetalInsertion,
    MetalReplacement,
    NeutralizationIon,
    PrepareConfig,
    ProteinSiteRespConfig,
    ProteinSiteRespMode,
    ProteinSiteRespScope,
    ProtonationChange,
    ProtonationConfig,
    ProtocolKind,
    RespApplyMode,
    SaltConfig,
    SaltKind,
    SaltMode,
    SlurmConfig,
    SlurmProfile,
    SystemConfig,
    WorkflowConfig,
    dump_config,
    save_config,
)
from amber_metallo.des import (
    DES_COMPONENTS,
    DES_RECOMMENDED_SETS,
    available_des_components,
    classify_des_library_bundle,
    discover_des_library_candidates,
    load_custom_des_components,
    overwrite_des_library_component,
    recommended_ratio_for_components,
    register_custom_des_component,
    resolve_ref_data_dir,
    unregister_custom_des_component,
)
from amber_metallo.environment import detect_amber_environment, environment_summary
from amber_metallo.inspection import fetch_pdb_structure, inspect_structure, load_structure, looks_like_pdb_id
from amber_metallo.ligand_param import prepare_canonical_small_molecule_mol2
from amber_metallo.prep import prepare_structure
from amber_metallo.protonation import predict_protonation_prediction
from amber_metallo.protein_site_resp import site_resp_result_path
from amber_metallo.qm.nwchem import (
    AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM,
    AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY,
    AUTO_GROUP_GRAPH_METHODS,
    AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY,
    AUTO_GROUP_MODE_HYDROGEN_ONLY,
    MoleculeAtom,
    MoleculeBond,
    MoleculeData,
    QM_BASIS_OPTIONS,
    QM_FUNCTIONAL_OPTIONS,
    QM_GEOMETRY_MODE_OPTIONS,
    QM_GRID_OPTIONS,
    build_default_session_state,
    find_resp_source_candidates,
    load_molecule,
    load_resp_job_candidate,
    molecule_fingerprint,
    normalize_qm_settings,
    render_preview_mol2,
    select_job_dir,
    suggest_group_constraints,
    write_resp_job_assets,
)
from amber_metallo.qm.resp_fit import load_resp_charge_result
from amber_metallo.slurm import render_slurm_script
from amber_metallo.workflow import run_workflow
from amber_metallo.metal_insert import donor_candidates_for_residue_selectors

from .preview import (
    des_heavy_atom_preview,
    insert_metal_atom_into_molecule,
    load_small_molecule_for_preview,
    molecule_to_pdb_text,
    molecule_payload,
    protein_preview_payload,
    quick_minimize_with_openbabel,
    resolve_coordination_donors,
)

try:
    from fastapi import Body, FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover - launcher reports this path
    Body = FastAPI = File = Form = Request = UploadFile = None  # type: ignore[assignment]
    FileResponse = JSONResponse = PlainTextResponse = StaticFiles = None  # type: ignore[assignment]
    TrustedHostMiddleware = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]


WORKFLOW_OPTIONS = [
    ("Metallophore (S)", "metallophore"),
    ("MetalloProtein (P)", "metalloprotein"),
    ("Deep Eutectic (D)", "deep_eutectic"),
    ("Add Component Library (A)", "add_library"),
]
SUPPORTED_METAL_CHOICES = [
    "Co",
    "Cu",
    "Ni",
    "Mn",
    "Fe",
    "Sc",
    "Y",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]
DEFAULT_WATER_MODELS = ["opc", "spce", "tip3p", "opc3", "tip4pew", "tip5p"]
DEFAULT_PROTEIN_FFS = ["ff19SB", "ff14SB", "ff99SB", "ff99SBildn"]
DEFAULT_LIGAND_FFS = ["gaff2", "gaff"]
SMALL_MOLECULE_SUFFIXES = {".pdb", ".mol2", ".sdf", ".sd", ".smi", ".smiles", ".txt"}
GENERAL_UPLOAD_SUFFIXES = SMALL_MOLECULE_SUFFIXES | {".cif", ".mmcif", ".pqr"}
MAX_UPLOAD_FILES = 512
MAX_UPLOAD_FILE_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 500 * 1024 * 1024
SITE_RESP_BROWSER_SUFFIXES = {
    ".json",
    ".grid",
    ".xyz",
    ".nw",
    ".sbatch",
    ".py",
    ".txt",
    ".log",
    ".out",
    ".pdb",
    ".prmtop",
    ".inpcrd",
    ".toml",
}
AUTO_GROUP_MODE_OPTIONS = [
    (AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY, "H + Symmetry"),
    (AUTO_GROUP_MODE_HYDROGEN_ONLY, "H Only"),
]
AUTO_GROUP_GRAPH_METHOD_OPTIONS = [
    (AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY, "Connectivity (fast)"),
    (AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM, "Exact graph symmetry"),
]
COORDINATION_NUMBER_HINTS = {
    "Co": "4-6",
    "Cu": "4-6",
    "Ni": "4-6",
    "Mn": "4-7",
    "Fe": "4-6",
    "Sc": "6-8",
    "Y": "8-9",
    "La": "8-12",
    "Ce": "8-10",
    "Pr": "8-10",
    "Nd": "8-10",
    "Pm": "8-10",
    "Sm": "8-10",
    "Eu": "8-10",
    "Gd": "8-9",
    "Tb": "8-9",
    "Dy": "8-9",
    "Ho": "8-9",
    "Er": "8-9",
    "Tm": "7-8",
    "Yb": "7-8",
    "Lu": "6-8",
}
METAL_COORDINATION_DEFAULTS = {
    ("Co", 2): {"default_cn": 6, "allowed_cn": [4, 5, 6]},
    ("Cu", 1): {"default_cn": 2, "allowed_cn": [2, 3, 4]},
    ("Cu", 2): {"default_cn": 4, "allowed_cn": [4, 5, 6]},
    ("Ni", 2): {"default_cn": 6, "allowed_cn": [4, 5, 6]},
    ("Mn", 2): {"default_cn": 6, "allowed_cn": [4, 5, 6, 7]},
    ("Fe", 2): {"default_cn": 6, "allowed_cn": [4, 5, 6]},
    ("Fe", 3): {"default_cn": 6, "allowed_cn": [4, 5, 6]},
    ("Sc", 3): {"default_cn": 8, "allowed_cn": [6, 7, 8]},
    ("Y", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("La", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10, 11, 12]},
    ("Ce", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10]},
    ("Pr", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10]},
    ("Nd", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10]},
    ("Pm", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10]},
    ("Sm", 3): {"default_cn": 9, "allowed_cn": [8, 9, 10]},
    ("Eu", 3): {"default_cn": 8, "allowed_cn": [8, 9, 10]},
    ("Gd", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("Tb", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("Dy", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("Ho", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("Er", 3): {"default_cn": 8, "allowed_cn": [8, 9]},
    ("Tm", 3): {"default_cn": 8, "allowed_cn": [7, 8]},
    ("Yb", 3): {"default_cn": 8, "allowed_cn": [7, 8]},
    ("Lu", 3): {"default_cn": 8, "allowed_cn": [6, 7, 8]},
}


def _coordination_metadata_for_element(element: str) -> dict[str, dict[str, Any]]:
    symbol = str(element or "").strip().title()
    metadata: dict[str, dict[str, Any]] = {}
    for charge in allowed_metal_charges(symbol):
        record = METAL_COORDINATION_DEFAULTS.get((symbol, int(charge)))
        if record is None:
            continue
        metadata[str(int(charge))] = {
            "default_cn": int(record["default_cn"]),
            "allowed_cn": [int(item) for item in record["allowed_cn"]],
        }
    return metadata


def _default_coordination_number(element: str, charge: int | None) -> int | None:
    symbol = str(element or "").strip().title()
    if charge is None:
        charge = DEFAULT_TLEAP_METAL_CHARGES.get(symbol)
    if charge is None:
        return None
    record = METAL_COORDINATION_DEFAULTS.get((symbol, int(charge)))
    return None if record is None else int(record["default_cn"])


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(slots=True)
class WebGuiState:
    repo_root: Path
    launch_cwd: Path
    session_root: Path
    api_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    last_config: WorkflowConfig | None = None
    last_toml: str = ""
    last_config_path: Path | None = None
    last_result: dict[str, Any] | None = None
    uploads: list[Path] = field(default_factory=list)


def _require_web_dependencies() -> None:
    if FastAPI is None or uvicorn is None:
        raise RuntimeError(
            "The browser GUI requires FastAPI and uvicorn.\n"
            "Install them with: pip install fastapi uvicorn python-multipart"
        )


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _safe_name(value: str, default: str = "simple_gui") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned or default


def _path_from_payload(value: object, *, base: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("A file or folder path is required.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _output_dir_from_payload(payload: dict[str, Any], state: WebGuiState) -> Path:
    root_text = str(payload.get("output_root") or "gui_outputs").strip() or "gui_outputs"
    root = Path(root_text).expanduser()
    if not root.is_absolute():
        root = state.launch_cwd / root
    resolved_root = root.resolve()
    if not _path_is_under(resolved_root, state.launch_cwd):
        raise ValueError(f"GUI output_root must remain inside the launch directory: {state.launch_cwd}")
    return (resolved_root / _safe_name(str(payload.get("job_name") or "simple_gui"))).resolve()


async def _write_upload_limited(
    upload: Any,
    target: Path,
    *,
    total_bytes: int = 0,
) -> int:
    """Stream one upload to disk while enforcing per-file and request limits."""

    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with target.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_FILE_BYTES:
                    raise ValueError(f"Upload exceeds the {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MiB per-file limit.")
                if total_bytes + written > MAX_UPLOAD_TOTAL_BYTES:
                    raise ValueError(f"Uploads exceed the {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MiB request limit.")
                stream.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return written


def _api_error(exc: Exception, state: WebGuiState, *, public_message: str | None = None) -> Any:
    """Return a useful API error without exposing local absolute paths."""

    print(f"SIMPLE Web GUI request failed: {exc}", file=sys.stderr)
    message = str(public_message if public_message is not None else exc) or exc.__class__.__name__
    replacements = {
        str(state.session_root): "<session directory>",
        str(state.launch_cwd): "<launch directory>",
        str(state.repo_root): "<application directory>",
        str(Path.home().resolve()): "<home directory>",
    }
    for private_path, label in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        message = message.replace(private_path, label)
        message = message.replace(private_path.replace("\\", "/"), label)
    message = re.sub(r"(?i)\b[A-Z]:[\\/][^\r\n,;]+", "<local path>", message)
    return JSONResponse({"ok": False, "error": message}, status_code=400)  # type: ignore[operator]


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _stage_gui_input_file(path: Path, *, state: WebGuiState, output_dir: Path, category: str) -> Path:
    source = path.expanduser().resolve()
    if not source.exists() or not _path_is_under(source, state.session_root):
        return source
    target_dir = output_dir / "00_inputs" / _safe_name(category, "inputs")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / _safe_name(source.name, "input.dat")).resolve()
    if source != target:
        shutil.copy2(source, target)
    return target


def _write_smiles_source(payload: dict[str, Any], state: WebGuiState, *, residue_name: str) -> str | None:
    smiles = str(payload.get("smiles_text") or "").strip()
    if not smiles:
        return None
    if any(line.startswith(("@<TRIPOS>", "ATOM", "HETATM", "HEADER")) for line in smiles.splitlines()):
        raise ValueError(
            "The SMILES field looks like a structure file, not a SMILES string. "
            "Upload PDB/MOL2/SDF files with Choose File, or enter one valid SMILES record."
        )
    target = state.session_root / "inputs" / f"{_safe_name(residue_name, 'LIG')}.smi"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(smiles + "\n", encoding="utf-8")
    return str(target.resolve())


def _small_molecule_source(payload: dict[str, Any], state: WebGuiState, *, residue_name: str) -> str:
    smiles_source = _write_smiles_source(payload, state, residue_name=residue_name)
    if smiles_source is not None:
        return smiles_source
    source = _path_from_payload(payload.get("input_path"), base=state.launch_cwd)
    if not source.exists():
        raise ValueError(f"Small-molecule input was not found: {source}")
    if source.suffix.lower() not in SMALL_MOLECULE_SUFFIXES:
        raise ValueError(
            f"Unsupported molecule format '{source.suffix or '(none)'}'. "
            "Use PDB, MOL2, SDF, or SMILES input."
        )
    return str(source)


def _edited_molecule_from_payload(payload: dict[str, Any]) -> MoleculeData | None:
    atoms_payload = payload.get("edited_atoms") or []
    if not atoms_payload:
        return None
    atoms: list[MoleculeAtom] = []
    for item in atoms_payload:
        atom = dict(item or {})
        index = int(atom.get("index") or 0)
        if index <= 0:
            continue
        element = str(atom.get("element") or atom.get("name") or "C").strip().title() or "C"
        atoms.append(
            MoleculeAtom(
                index=index,
                name=str(atom.get("name") or element).strip() or element,
                element=element,
                x=float(atom.get("x") or 0.0),
                y=float(atom.get("y") or 0.0),
                z=float(atom.get("z") or 0.0),
                charge=None if atom.get("partial_charge") is None else float(atom.get("partial_charge")),
            )
        )
    if not atoms:
        return None
    bonds = [
        MoleculeBond(
            first=int(item.get("first") or 0),
            second=int(item.get("second") or 0),
            order=int(item.get("order") or 1),
        )
        for item in (payload.get("edited_bonds") or [])
        if int(item.get("first") or 0) > 0 and int(item.get("second") or 0) > 0
    ]
    return MoleculeData(source_file="gui-edited-preview", source_format="gui", atoms=atoms, bonds=bonds)


def _materialize_edited_molecule_source(
    payload: dict[str, Any],
    state: WebGuiState,
    *,
    residue_name: str,
) -> str | None:
    molecule = _edited_molecule_from_payload(payload)
    if molecule is None:
        return None
    target = state.session_root / "edited_metallophore_sources" / f"{_safe_name(residue_name, 'LIG')}_gui_edited.mol2"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_preview_mol2(molecule, residue_name=residue_name), encoding="utf-8")
    return str(target.resolve())


def _friendly_small_molecule_error(exc: Exception, payload: dict[str, Any]) -> str:
    message = str(exc)
    if payload.get("smiles_text"):
        return (
            "SMILES input could not be parsed or converted to 3D. "
            "Please check the SMILES string and Open Babel availability. "
            f"Details: {message}"
        )
    return (
        "Molecule input could not be read. Please check that the file is PDB, MOL2, SDF, "
        f"or SMILES and that the file is not corrupted. Details: {message}"
    )


def _molecule_with_resp_charges(molecule: Any, charge_path: Path | None) -> Any:
    if charge_path is None or not charge_path.exists():
        return molecule
    try:
        if charge_path.suffix.lower() == ".json":
            charges = load_resp_charge_result(charge_path)
        else:
            charges = [
                float(line.strip())
                for line in charge_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    except Exception:
        return molecule
    if len(charges) != len(molecule.atoms):
        return molecule
    from dataclasses import replace

    return molecule.__class__(
        source_file=molecule.source_file,
        source_format=molecule.source_format,
        atoms=[replace(atom, charge=float(charge)) for atom, charge in zip(molecule.atoms, charges, strict=True)],
        bonds=list(molecule.bonds),
    )


def _group_mode_from_payload(payload: dict[str, Any]) -> str:
    constraints = payload.get("group_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    value = str(
        payload.get("auto_group_mode")
        or constraints.get("auto_group_mode")
        or AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY
    )
    valid = {key for key, _label in AUTO_GROUP_MODE_OPTIONS}
    return value if value in valid else AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY


def _group_graph_method_from_payload(payload: dict[str, Any]) -> str:
    constraints = payload.get("group_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    value = str(
        payload.get("auto_group_graph_method")
        or constraints.get("auto_group_graph_method")
        or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
    )
    valid = AUTO_GROUP_GRAPH_METHODS
    return value if value in valid else AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY


def _suggest_group_constraints_for_payload(molecule: Any, payload: dict[str, Any]) -> dict[str, object]:
    return suggest_group_constraints(
        molecule,
        auto_group_mode=_group_mode_from_payload(payload),
        auto_group_graph_method=_group_graph_method_from_payload(payload),
    )


def _salt_config(payload: dict[str, Any] | None) -> SaltConfig:
    data = dict(payload or {})
    kind = SaltKind(str(data.get("kind") or SaltKind.NONE.value))
    mode = SaltMode(str(data.get("mode") or SaltMode.NONE.value))
    value = data.get("value")
    neutralization_ion = NeutralizationIon(
        str(data.get("neutralization_ion") or NeutralizationIon.AUTO.value)
    )
    return SaltConfig(
        kind=kind,
        mode=mode,
        value=value,
        neutralization_ion=neutralization_ion,
    )


def _metal_charges(items: list[dict[str, Any]] | None) -> list[MetalChargeAssignment]:
    assignments = []
    for item in items or []:
        site = int(item.get("site") or 0)
        charge = int(item.get("charge") or 0)
        if site > 0 and charge > 0:
            assignments.append(MetalChargeAssignment(site=site, charge=charge))
    return assignments


def _metal_insertions(items: list[dict[str, Any]] | None) -> list[MetalInsertion]:
    insertions: list[MetalInsertion] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        element = str(item.get("element") or "").strip()
        if not element:
            continue
        anchors = [str(anchor).strip() for anchor in item.get("anchors") or [] if str(anchor).strip()]
        coordinates = item.get("coordinates")
        mode = str(item.get("anchor_mode") or MetalAnchorMode.DONOR_ATOMS.value)
        try:
            insertions.append(
                MetalInsertion(
                    element=element,
                    charge=_optional_int(item.get("charge")),
                    anchor_mode=MetalAnchorMode(mode),
                    anchors=anchors,
                    target_coordination_number=_optional_int(item.get("target_coordination_number")),
                    coordinates=coordinates,
                    label=str(item.get("label") or "").strip() or None,
                )
            )
        except Exception as exc:
            raise ValueError(f"Invalid metal insertion request for {element}: {exc}") from exc
    return insertions


def _system_config(data: dict[str, Any] | None, *, include_protein: bool) -> SystemConfig:
    payload = dict(data or {})
    protein_ff = str(payload.get("protein_ff") or "ff19SB")
    ligand_ff = str(payload.get("ligand_ff") or "gaff2")
    water_model = str(payload.get("water_model") or "opc").lower()
    raw_parameter_set = payload.get("c4_parameter_set")
    parameter_set = DESC4ParameterSet(
        str(raw_parameter_set)
        if raw_parameter_set
        else (
            DESC4ParameterSet.OPC_DUVAIL.value
            if water_model == "opc"
            else DESC4ParameterSet.SPCE_LIMERZ.value
        )
    )
    box_shape = BoxShape(str(payload.get("box_shape") or BoxShape.OCT.value))
    return SystemConfig(
        protein_ff=protein_ff if include_protein else "ff19SB",
        ligand_ff=ligand_ff,
        apply_1264=bool(payload.get("apply_1264", True)),
        c4_parameter_set=parameter_set,
        metal_charges=_metal_charges(payload.get("metal_charges") or []),
        water_model=water_model,
        box_shape=box_shape,
        buffer_angstrom=float(payload.get("buffer_angstrom") or 10.0),
        salt=_salt_config(payload.get("salt") or {}),
    )


def _md_config(data: dict[str, Any] | None, *, des: bool = False) -> MDConfig:
    payload = dict(data or {})
    protocol = ProtocolKind.DES_SOLVENT if des else ProtocolKind(str(payload.get("protocol") or ProtocolKind.FIFTEEN_STEP.value))
    focused_mask = str(payload.get("focused_restraint_mask") or "").strip() or None
    focused_weight_raw = payload.get("focused_restraint_weight")
    focused_weight = None if focused_weight_raw in {None, ""} else float(focused_weight_raw)
    return MDConfig(
        protocol=protocol,
        temperature_k=float(payload.get("temperature_k") or 300.0),
        pressure_bar=float(payload.get("pressure_bar") or 1.0),
        production_time_ns=float(payload.get("production_time_ns") or (100.0 if des else 10.0)),
        des_mixing_enabled=bool(payload.get("des_mixing_enabled", True)) if des else False,
        des_mixing_temperature_k=float(payload.get("des_mixing_temperature_k") or 500.0),
        des_mixing_time_ns=float(payload.get("des_mixing_time_ns") or 50.0),
        focused_restraint_mask=focused_mask,
        focused_restraint_weight=focused_weight,
        stage_overrides=dict(payload.get("stage_overrides") or {}),
    )


def _slurm_config(data: dict[str, Any] | None, *, job_name: str) -> SlurmConfig:
    payload = dict(data or {})
    profile = SlurmProfile(str(payload.get("profile") or SlurmProfile.GPU.value))
    return SlurmConfig(
        profile=profile,
        partition=str(payload.get("partition") or "").strip() or None,
        account=str(payload.get("account") or "").strip() or None,
        ntasks=int(payload.get("ntasks") or 8),
        gpus=int(payload.get("gpus") if payload.get("gpus") is not None else (1 if profile == SlurmProfile.GPU else 0)),
        walltime=str(payload.get("walltime") or "24:00:00"),
        binary_override=str(payload.get("binary_override") or "").strip() or None,
        job_name=_safe_name(job_name),
    )


def _ligands_config(data: dict[str, Any] | None, *, residue_name: str) -> LigandsConfig:
    payload = dict(data or {})
    mode = LigandMode(str(payload.get("mode") or LigandMode.GAFF2.value))
    charge_method = ChargeMethod(str(payload.get("charge_method") or ChargeMethod.RESP_ANTECHAMBER.value))
    return LigandsConfig(
        mode=mode,
        charge_method=charge_method,
        manual_files=[str(item) for item in payload.get("manual_files") or []],
        residue_name=residue_name,
        net_charge=int(payload.get("net_charge") or 0),
        multiplicity=int(payload.get("multiplicity") or 1),
        resp_job_dir=str(payload.get("resp_job_dir") or "").strip() or None,
        resp_group_file=str(payload.get("resp_group_file") or "").strip() or None,
        resp_session_file=str(payload.get("resp_session_file") or "").strip() or None,
        resp_apply_mode=RespApplyMode(str(payload.get("resp_apply_mode") or RespApplyMode.NEW_DIRECTORY.value)),
    )


def _prepare_config(data: dict[str, Any] | None) -> PrepareConfig:
    payload = dict(data or {})
    replacements = [
        MetalReplacement(site=int(item.get("site") or 0), target=str(item.get("target") or ""))
        for item in payload.get("metal_replacements") or []
        if int(item.get("site") or 0) > 0 and str(item.get("target") or "").strip()
    ]
    deletions = [int(item) for item in payload.get("metal_deletions") or [] if int(item) > 0]
    return PrepareConfig(
        remove_waters=bool(payload.get("remove_waters", True)),
        remove_other_hetero=bool(payload.get("remove_other_hetero", True)),
        remove_metals=False,
        repair_missing_loops=bool(payload.get("repair_missing_loops", False)),
        kept_ligands=[str(item).strip() for item in payload.get("kept_ligands") or [] if str(item).strip()],
        metal_replacements=replacements,
        metal_deletions=deletions,
        metal_insertions=_metal_insertions(payload.get("metal_insertions") or []),
    )


def _protein_raw_source(protein: dict[str, Any], state: WebGuiState) -> tuple[InputSource, str, Path]:
    raw_value = str(protein.get("input_value") or "").strip()
    mode = str(protein.get("input_mode") or "").strip().lower()
    if not raw_value:
        raw_value = str(protein.get("pdb_id") or "").strip() if mode == "pdb_id" else str(protein.get("path") or "").strip()
    if not raw_value:
        raise ValueError("Enter a protein path/PDB ID or upload a PDB file.")

    candidate_path = Path(raw_value).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = state.launch_cwd / candidate_path
    use_pdb_id = mode == "pdb_id" or (looks_like_pdb_id(raw_value) and not candidate_path.exists())
    if use_pdb_id:
        pdb_id = raw_value.strip().upper()
        if not looks_like_pdb_id(pdb_id):
            raise ValueError("Enter a valid 4-character PDB ID.")
        raw_path = fetch_pdb_structure(pdb_id, state.session_root / "pdb_downloads")
        return InputSource.PDB_ID, pdb_id, raw_path
    raw_path = _path_from_payload(raw_value, base=state.launch_cwd)
    if not raw_path.exists():
        raise ValueError(f"Protein input was not found: {raw_path}")
    return InputSource.PDB_FILE, str(raw_path), raw_path


def _missing_loop_dialog_payload(summary: Any) -> dict[str, Any] | None:
    missing = getattr(summary, "missing_loops", None)
    if missing is None or getattr(missing, "detection_status", "") != "available" or not getattr(missing, "internal_blocks", []):
        return None
    return {
        "ok": False,
        "action_required": "missing_loops",
        "message": f"Detected {len(missing.internal_blocks)} internal missing loop block(s).",
        "missing_loops": missing.to_dict(),
    }


def _missing_loop_locators(summary: Any) -> set[tuple[str, str]]:
    missing = getattr(summary, "missing_loops", None)
    if missing is None:
        return set()
    try:
        return set(missing.boundary_residue_locators())
    except Exception:
        return set()


def _metal_insertion_guide_links(inserted_sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for site in inserted_sites or []:
        metal_xyz = site.get("coordinates") or []
        if len(metal_xyz) != 3:
            continue
        metal_key = str(site.get("key") or "")
        metal_element = str(site.get("element") or "")
        donors = [
            *(site.get("donor_atoms") or []),
            *(site.get("auto_filled_donor_atoms") or []),
        ]
        for donor in donors:
            donor_xyz = [donor.get("x"), donor.get("y"), donor.get("z")]
            if any(value is None for value in donor_xyz):
                continue
            links.append(
                {
                    "residue_key": donor.get("residue_key"),
                    "residue_label": (
                        f"{donor.get('chain') or '_'}:{donor.get('resid') or donor.get('seqid')} "
                        f"{donor.get('residue_name') or ''}"
                    ),
                    "donor_atom_name": donor.get("atom_name"),
                    "donor": [float(value) for value in donor_xyz],
                    "metal_key": metal_key,
                    "metal_element": metal_element,
                    "metal_atom_name": site.get("residue_name") or metal_element,
                    "metal": [float(value) for value in metal_xyz],
                    "distance_angstrom": (
                        (
                            (float(donor_xyz[0]) - float(metal_xyz[0])) ** 2
                            + (float(donor_xyz[1]) - float(metal_xyz[1])) ** 2
                            + (float(donor_xyz[2]) - float(metal_xyz[2])) ** 2
                        )
                        ** 0.5
                    ),
                    "inserted": True,
                }
            )
    return links


def _protonation_config(data: dict[str, Any] | None) -> ProtonationConfig:
    payload = dict(data or {})
    changes = [
        ProtonationChange.model_validate(item)
        for item in payload.get("selected_changes") or []
        if isinstance(item, dict)
    ]
    enabled = bool(payload.get("enabled")) and bool(changes)
    return ProtonationConfig(
        enabled=enabled,
        ph=float(payload.get("ph") or 7.0) if enabled else None,
        selected_changes=changes if enabled else [],
    )


def _protein_site_resp_config(data: dict[str, Any] | None, state: WebGuiState) -> ProteinSiteRespConfig:
    payload = dict(data or {})
    mode = ProteinSiteRespMode(str(payload.get("mode") or ProteinSiteRespMode.STANDARD_FF.value))
    if mode == ProteinSiteRespMode.STANDARD_FF:
        return ProteinSiteRespConfig()

    job_text = str(payload.get("job_dir") or "").strip()
    job_path: Path | None = None
    imported_manifest: dict[str, Any] = {}
    if job_text:
        job_path = Path(job_text).expanduser()
        if not job_path.is_absolute():
            job_path = state.launch_cwd / job_path
        job_path = job_path.resolve()
        manifest_path = job_path / "manifests" / "site_resp_manifest.json"
        if manifest_path.exists():
            try:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read the selected site-RESP manifest: {manifest_path}") from exc
            if isinstance(loaded_manifest, dict):
                imported_manifest = loaded_manifest

    if job_path is None or not imported_manifest:
        raise ValueError(
            "The GUI imports completed protein-site RESP results only. Create and review the site-RESP job "
            "through main.py, then select its completed case folder with Scan / Browse RESP Results."
        )
    if site_resp_result_path(job_path) is None:
        raise ValueError(
            "The selected main.py site-RESP job is not complete yet. Finish the NWChem/RESP calculation, then "
            "scan the case folder again."
        )

    multiplicity_confirmed = bool(payload.get("multiplicity_confirmed")) or bool(imported_manifest)
    if not multiplicity_confirmed:
        raise ValueError(
            "Confirm the spin multiplicity through main.py first. The selected protein-site RESP result has no "
            "readable main.py manifest with a confirmed multiplicity for GUI import."
        )
    multiplicity = int(
        imported_manifest.get("multiplicity")
        or payload.get("default_multiplicity")
        or 0
    )
    if multiplicity < 1:
        raise ValueError("Protein-site RESP spin multiplicity must be at least 1.")

    search_roots: list[str] = []
    search_text = str(payload.get("search_root") or "").strip()
    if search_text:
        search_path = Path(search_text).expanduser()
        if not search_path.is_absolute():
            search_path = state.launch_cwd / search_path
        search_roots.append(str(search_path.resolve()))
    job_dirs: list[str] = []
    if job_path is not None:
        job_dirs.append(str(job_path.resolve()))
    raw_clusters: list[dict[str, Any]] = []
    cluster = dict(imported_manifest.get("cluster") or {})
    if cluster.get("metal_sites"):
        raw_clusters = [
            {
                "metal_sites": cluster.get("metal_sites") or [],
                "donor_residues": cluster.get("donor_residue_keys") or [],
                "fixed_environment": cluster.get("fixed_environment_keys") or [],
                "multiplicity": imported_manifest.get("multiplicity") or cluster.get("multiplicity"),
                "job_dir": str(job_path) if job_path is not None else None,
            }
        ]
    return ProteinSiteRespConfig(
        mode=mode,
        scope=ProteinSiteRespScope(
            str(
                imported_manifest.get("scope")
                or payload.get("scope")
                or ProteinSiteRespScope.SIDECHAIN.value
            )
        ),
        apply_mode=RespApplyMode(
            str(payload.get("apply_mode") or RespApplyMode.DETECT.value)
        ),
        default_multiplicity=multiplicity,
        search_roots=search_roots,
        job_dirs=job_dirs,
        review_clusters=True,
        clusters=raw_clusters,
    )


def _resolve_resp_resume_source(candidate: Any, state: WebGuiState) -> str:
    payload = getattr(candidate, "payload", {}) or {}
    for key in ("resume_source_file", "source_file", "canonical_source_file"):
        raw = str(payload.get(key) or "").strip()
        if raw and Path(raw).expanduser().exists():
            return str(Path(raw).expanduser().resolve())
    popup_state = Path(candidate.job_dir) / "manifests" / "popup_state.json"
    if popup_state.exists():
        session_state = json.loads(popup_state.read_text(encoding="utf-8"))
        preview = str(session_state.get("mol2_preview") or "").strip()
        if preview:
            target = state.session_root / "resp_resume" / f"{Path(candidate.job_dir).name}_resume.mol2"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(preview + "\n", encoding="utf-8")
            return str(target.resolve())
    raise ValueError("The selected RESP job does not contain a reusable source file or saved MOL2 preview.")


def _validate_gui_c4_selection(payload: dict[str, Any], *, workflow_type: str) -> None:
    """Reject a stale or hand-edited GUI payload that bypasses disabled Duvail choices."""
    if workflow_type == "deep_eutectic":
        # DES has a separate parameter-set selector and validation path.  This
        # guard mirrors the Protein/Small Molecule metal table controlled by
        # system.c4_parameter_set in the browser GUI.
        return
    species: list[tuple[str, int]] = []
    system = dict(payload.get("system") or {})
    if not bool(system.get("apply_1264", True)):
        return
    parameter_set = DESC4ParameterSet(
        str(system.get("c4_parameter_set") or DESC4ParameterSet.OPC_DUVAIL.value)
    )
    species.extend(
        (
            str(item.get("element") or "").strip().title(),
            int(item.get("charge") or 0),
        )
        for item in system.get("metal_charges") or []
        if isinstance(item, dict) and str(item.get("element") or "").strip()
    )
    if parameter_set != DESC4ParameterSet.OPC_DUVAIL:
        return
    unsupported = sorted(
        {
            (element, charge)
            for element, charge in species
            if element
            and charge > 0
            and not c4_parameter_set_supports_metal_charge(parameter_set, element, charge)
        }
    )
    if not unsupported:
        return
    unsupported_text = ", ".join(f"{element}{charge}+" for element, charge in unsupported)
    raise ValueError(
        f"OPC + Duvail does not support the selected ion(s): {unsupported_text}. "
        "Choose SPC/E + Li/Merz; SPC/E will be used as the compatible solvation default."
    )


def build_workflow_config(
    payload: dict[str, Any],
    state: WebGuiState,
    *,
    stage_gui_inputs: bool = False,
) -> WorkflowConfig:
    workflow_type = str(payload.get("workflow_type") or "metalloprotein")
    job_name = _safe_name(str(payload.get("job_name") or "simple_gui"))
    output_dir = _output_dir_from_payload(payload, state)
    _validate_gui_c4_selection(payload, workflow_type=workflow_type)
    system = _system_config(payload.get("system") or {}, include_protein=workflow_type == "metalloprotein")
    slurm = _slurm_config(payload.get("slurm") or {}, job_name=job_name)

    if workflow_type == "deep_eutectic":
        data = dict(payload.get("des") or {})
        components = []
        for item in data.get("components") or []:
            try:
                components.append(DESComponent(str(item)))
            except ValueError:
                components.append(str(item))
        ratios = [int(item) for item in data.get("ratios") or recommended_ratio_for_components(components)]
        size_mode = DESSizeMode(str(data.get("size_mode") or DESSizeMode.RATIO_UNITS.value))
        des_config = DESConfig(
            components=components,
            ratios=ratios,
            mixing_mode=DESMixingMode(
                str(data.get("mixing_mode") or DESMixingMode.RANDOM_MIX.value)
            ),
            replicate_order=DESReplicateOrder(str(data.get("replicate_order") or DESReplicateOrder.UNIFORM.value)),
            size_mode=size_mode,
            ratio_units=int(data.get("ratio_units") or 100) if size_mode == DESSizeMode.RATIO_UNITS else None,
            box_length_angstrom=(
                float(data.get("box_length_angstrom") or 80.0)
                if size_mode == DESSizeMode.BOX_LENGTH
                else None
            ),
            spacing_angstrom=float(data.get("spacing_angstrom") or 1.3),
            packmol_tolerance_angstrom=float(data.get("packmol_tolerance_angstrom") or 2.0),
            packmol_fill_fraction=float(data.get("packmol_fill_fraction") or 1.0),
            target_density_g_ml=float(data.get("target_density_g_ml") or 0.40),
            apply_1264=bool(data.get("apply_1264", True)),
            c4_parameter_set=DESC4ParameterSet(str(data.get("c4_parameter_set") or DESC4ParameterSet.OPC_DUVAIL.value)),
            central_metal_enabled=bool(data.get("central_metal_enabled", False)),
            central_metal_element=str(data.get("central_metal_element") or "Fe"),
            central_metal_charge=int(data.get("central_metal_charge") or 2),
            metal_sites=list(data.get("metal_sites") or []),
            metal_spacing_angstrom=float(data.get("metal_spacing_angstrom") or 8.0),
        )
        return WorkflowConfig(
            input=InputConfig(source=InputSource.DES),
            des=des_config,
            prepare=PrepareConfig(remove_waters=False, remove_other_hetero=False, remove_metals=False),
            md=_md_config(payload.get("md") or {}, des=True),
            system=system.model_copy(
                update={
                    "water_model": "opc" if des_config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL else "spce",
                    "c4_parameter_set": des_config.c4_parameter_set,
                    "box_shape": BoxShape.CUBIC,
                }
            ),
            slurm=slurm,
            output_dir=str(output_dir),
        )

    if workflow_type == "metallophore":
        data = dict(payload.get("metallophore") or {})
        residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
        mode = str(data.get("mode") or "resp_input")
        ligand_payload = dict(data.get("ligands") or {})
        if mode == "existing_resp":
            job_dir = _path_from_payload(data.get("resp_job_dir"), base=state.launch_cwd)
            candidate = load_resp_job_candidate(job_dir)
            if candidate is None:
                raise ValueError(f"RESP job manifest was not found under {job_dir}")
            source_file = _resolve_resp_resume_source(candidate, state)
            manifest = candidate.payload
            residue_name = str(manifest.get("residue_name") or residue_name).strip().upper()[:3] or residue_name
            ligand_payload.update(
                {
                    "resp_job_dir": str(candidate.job_dir),
                    "resp_group_file": str(candidate.job_dir / "group_constraints.json")
                    if (candidate.job_dir / "group_constraints.json").exists()
                    else None,
                    "resp_session_file": str(candidate.job_dir / "manifests" / "popup_state.json")
                    if (candidate.job_dir / "manifests" / "popup_state.json").exists()
                    else None,
                    "resp_apply_mode": RespApplyMode.APPLY_EXISTING.value,
                    "charge_method": manifest.get("charge_method") or ChargeMethod.RESP_ANTECHAMBER.value,
                    "net_charge": int(manifest.get("net_charge") or 0),
                    "multiplicity": int(manifest.get("multiplicity") or 1),
                }
            )
        else:
            source_file = (
                _materialize_edited_molecule_source(data, state, residue_name=residue_name)
                or _small_molecule_source(data, state, residue_name=residue_name)
            )
            ligand_payload.setdefault("resp_apply_mode", RespApplyMode.NEW_DIRECTORY.value)
            ligand_payload.setdefault("charge_method", ChargeMethod.RESP_ANTECHAMBER.value)
        if stage_gui_inputs:
            source_file = str(
                _stage_gui_input_file(
                    Path(source_file),
                    state=state,
                    output_dir=output_dir,
                    category="metallophore",
                )
            )
        return WorkflowConfig(
            input=InputConfig(source=InputSource.SMALL_MOLECULE, small_molecule_files=[source_file]),
            ligands=_ligands_config(ligand_payload, residue_name=residue_name),
            system=system,
            md=_md_config(payload.get("md") or {}),
            slurm=slurm,
            output_dir=str(output_dir),
        )

    protein = dict(payload.get("protein") or {})
    input_mode = str(protein.get("input_mode") or "path")
    if input_mode == "pdb_id":
        input_config = InputConfig(source=InputSource.PDB_ID, pdb_id=str(protein.get("pdb_id") or "").strip().upper())
    else:
        protein_path = _path_from_payload(protein.get("path"), base=state.launch_cwd)
        if not protein_path.exists():
            raise ValueError(f"Protein PDB file was not found: {protein_path}")
        if stage_gui_inputs:
            protein_path = _stage_gui_input_file(
                protein_path,
                state=state,
                output_dir=output_dir,
                category="protein",
            )
        input_config = InputConfig(source=InputSource.PDB_FILE, path=str(protein_path))
    return WorkflowConfig(
        input=input_config,
        prepare=_prepare_config(protein.get("prepare") or {}),
        protonation=_protonation_config(protein.get("protonation") or {}),
        protein_site_resp=_protein_site_resp_config(protein.get("site_resp") or {}, state),
        ligands=_ligands_config(protein.get("ligands") or {"mode": LigandMode.MANUAL.value}, residue_name="CUSTOM"),
        system=system,
        md=_md_config(payload.get("md") or {}),
        slurm=slurm,
        output_dir=str(output_dir),
    )


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    payload = getattr(candidate, "payload", {}) or {}
    return {
        "job_dir": str(candidate.job_dir),
        "manifest_path": str(candidate.manifest_path),
        "completed": bool(candidate.completed),
        "ready_to_continue": bool(candidate.ready_to_continue),
        "source_file": payload.get("source_file"),
        "residue_name": payload.get("residue_name"),
        "net_charge": payload.get("net_charge"),
        "multiplicity": payload.get("multiplicity"),
        "charge_method": payload.get("charge_method"),
        "created_at": payload.get("created_at"),
    }


def _des_library_ref_data_dir(state: WebGuiState) -> Path:
    return resolve_ref_data_dir("REF_DATA")


def _des_library_candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "lib_path": str(candidate.lib_path),
        "frcmod_path": str(candidate.frcmod_path),
        "residue_name": candidate.residue_name,
        "status": candidate.status,
        "matched_component": candidate.matched_component,
        "matched_label": candidate.matched_label,
        "files": [
            {"name": candidate.lib_path.name, "path": str(candidate.lib_path), "kind": "library"},
            {"name": candidate.frcmod_path.name, "path": str(candidate.frcmod_path), "kind": "frcmod"},
        ],
    }


def _des_library_component_payloads(state: WebGuiState) -> list[dict[str, Any]]:
    ref_data_dir = _des_library_ref_data_dir(state)
    custom_keys = set(load_custom_des_components(ref_data_dir))
    payloads: list[dict[str, Any]] = []
    for key, definition in available_des_components(ref_data_dir).items():
        key_value = key.value if isinstance(key, DESComponent) else str(key)
        files: list[dict[str, str]] = []
        for residue in definition.residues:
            for kind in ("lib", "frcmod"):
                name = getattr(residue, kind)
                if not name:
                    continue
                path = ref_data_dir / definition.directory / name
                if path.exists():
                    files.append(
                        {
                            "name": path.name,
                            "path": str(path.resolve()),
                            "kind": kind,
                            "editable": key_value in custom_keys,
                        }
                    )
        payloads.append(
            {
                "key": key_value,
                "label": definition.label,
                "description": definition.description,
                "residues": [item.residue_name for item in definition.residues],
                "files": files,
                "custom": key_value in custom_keys,
                "removable": key_value in custom_keys,
                "editable": key_value in custom_keys,
            }
        )
    return payloads


def _des_library_file_path(value: object, state: WebGuiState) -> Path:
    path = _path_from_payload(value, base=state.launch_cwd)
    if path.suffix.lower() not in {".lib", ".off", ".frcmod"}:
        raise ValueError("Only Amber .lib/.off and .frcmod text files can be opened here.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Amber library file was not found: {path}")
    return path


def _built_in_des_library_asset_paths(state: WebGuiState) -> set[Path]:
    ref_data_dir = _des_library_ref_data_dir(state)
    paths: set[Path] = set()
    for definition in DES_COMPONENTS.values():
        for residue in definition.residues:
            for kind in ("lib", "frcmod"):
                name = getattr(residue, kind)
                if name:
                    paths.add((ref_data_dir / definition.directory / name).resolve())
    return paths


def _des_library_file_is_editable(state: WebGuiState, path: Path) -> bool:
    return path.resolve() not in _built_in_des_library_asset_paths(state)


def _des_library_upload_target(
    upload_root: Path,
    relative_path: object,
    fallback_name: object,
) -> Path:
    """Return a safe session-local destination for a browser-selected library file."""
    raw = str(relative_path or fallback_name or "upload.dat").strip().replace("\\", "/")
    browser_path = PurePosixPath(raw)
    if (
        not raw
        or browser_path.is_absolute()
        or re.match(r"^[A-Za-z]:/", raw)
        or any(part == ".." for part in browser_path.parts)
    ):
        raise ValueError(f"Unsafe browser upload path: {raw or '<empty>'}")
    safe_parts = [_safe_name(part, "upload") for part in browser_path.parts if part not in {"", "."}]
    if not safe_parts:
        raise ValueError("The selected library file has no usable name.")
    target = (upload_root.resolve() / Path(*safe_parts)).resolve()
    try:
        target.relative_to(upload_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe browser upload path: {raw}") from exc
    if target.suffix.lower() not in {".lib", ".off", ".frcmod"}:
        raise ValueError("Only Amber .lib/.off and .frcmod files can be selected here.")
    return target


def _site_resp_upload_target(
    upload_root: Path,
    relative_path: object,
    fallback_name: object,
) -> Path:
    """Return a safe session-local destination for a browser-selected RESP case file."""
    raw = str(relative_path or fallback_name or "upload.dat").strip().replace("\\", "/")
    browser_path = PurePosixPath(raw)
    if (
        not raw
        or browser_path.is_absolute()
        or re.match(r"^[A-Za-z]:/", raw)
        or any(part == ".." for part in browser_path.parts)
    ):
        raise ValueError(f"Unsafe browser upload path: {raw or '<empty>'}")
    safe_parts = [_safe_name(part, "upload") for part in browser_path.parts if part not in {"", "."}]
    if not safe_parts:
        raise ValueError("The selected protein-site RESP file has no usable name.")
    target = (upload_root.resolve() / Path(*safe_parts)).resolve()
    try:
        target.relative_to(upload_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe browser upload path: {raw}") from exc
    if target.suffix.lower() not in SITE_RESP_BROWSER_SUFFIXES:
        raise ValueError(
            "This file type is not needed for protein-site RESP result import: "
            f"{target.suffix or '<no suffix>'}"
        )
    return target


def scan_protein_site_resp_directory(state: WebGuiState, raw_path: object) -> dict[str, Any]:
    raw = str(raw_path or ".").strip() or "."
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = state.launch_cwd / root
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Protein-site RESP search folder was not found: {root}")
    candidates: list[dict[str, Any]] = []
    for manifest_path in root.rglob("site_resp_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job_dir = manifest_path.parent.parent
        result_path = site_resp_result_path(job_dir)
        candidates.append(
            {
                "job_dir": str(job_dir.resolve()),
                "description": manifest.get("description"),
                "source_label": manifest.get("source_label"),
                "metal_sites": (manifest.get("cluster") or {}).get("metal_sites"),
                "cluster": manifest.get("cluster") or {},
                "scope": manifest.get("scope"),
                "multiplicity": manifest.get("multiplicity"),
                "created_at": manifest.get("created_at"),
                "completed": result_path is not None,
            }
        )
    candidates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"ok": True, "search_root": str(root), "candidates": candidates}


def scan_des_library_directory(state: WebGuiState, raw_path: object) -> dict[str, Any]:
    path_text = str(raw_path or "").strip()
    if not path_text:
        raise ValueError("Enter a directory containing an Amber .lib/.off and .frcmod pair.")
    search_dir = _path_from_payload(path_text, base=state.launch_cwd)
    if not search_dir.exists() or not search_dir.is_dir():
        raise ValueError(f"Library directory was not found: {search_dir}")
    candidates = discover_des_library_candidates(
        search_dir,
        ref_data_dir=_des_library_ref_data_dir(state),
    )
    return {
        "ok": True,
        "path": str(search_dir),
        "candidates": [_des_library_candidate_payload(item) for item in candidates],
    }


def read_des_library_file(state: WebGuiState, raw_path: object) -> dict[str, Any]:
    path = _des_library_file_path(raw_path, state)
    editable = _des_library_file_is_editable(state, path)
    return {
        "ok": True,
        "path": str(path),
        "name": path.name,
        "content": path.read_text(encoding="utf-8"),
        "editable": editable,
        "protected": not editable,
    }


def save_des_library_file(state: WebGuiState, raw_path: object, content: object) -> dict[str, Any]:
    path = _des_library_file_path(raw_path, state)
    if not _des_library_file_is_editable(state, path):
        raise ValueError("Built-in DES component files are protected and cannot be modified.")
    path.write_text(str(content or ""), encoding="utf-8")
    return {"ok": True, "path": str(path), "message": f"Saved {path.name}."}


def register_des_library_candidate(state: WebGuiState, payload: dict[str, Any]) -> dict[str, Any]:
    ref_data_dir = _des_library_ref_data_dir(state)
    lib_path = _des_library_file_path(payload.get("lib_path"), state)
    frcmod_path = _des_library_file_path(payload.get("frcmod_path"), state)
    candidate = classify_des_library_bundle(
        lib_path=lib_path,
        frcmod_path=frcmod_path,
        ref_data_dir=ref_data_dir,
    )
    action = str(payload.get("action") or "register_new")
    if action == "overwrite":
        if candidate.status != "different_values" or not candidate.matched_component:
            raise ValueError("Overwrite is only available when the same residue has different parameter values.")
        definition = overwrite_des_library_component(
            lib_path=lib_path,
            frcmod_path=frcmod_path,
            component_key=candidate.matched_component,
            ref_data_dir=ref_data_dir,
        )
        message = f"Overwrote {definition.label} with the selected Amber parameter files."
    elif action == "register_new":
        definition = register_custom_des_component(
            lib_path=lib_path,
            frcmod_path=frcmod_path,
            component_key=str(payload.get("component_key") or "").strip() or None,
            label=str(payload.get("label") or "").strip() or None,
            ref_data_dir=ref_data_dir,
        )
        message = f"Registered {definition.label} as a DES library component."
    else:
        raise ValueError(f"Unsupported DES library action: {action}")
    return {
        "ok": True,
        "message": message,
        "components": _des_library_component_payloads(state),
    }


def remove_des_library_component(state: WebGuiState, component_key: object) -> dict[str, Any]:
    definition = unregister_custom_des_component(
        str(component_key or ""),
        ref_data_dir=_des_library_ref_data_dir(state),
    )
    return {
        "ok": True,
        "message": f"Removed custom DES component {definition.label}.",
        "components": _des_library_component_payloads(state),
    }


def _bootstrap_payload(state: WebGuiState) -> dict[str, Any]:
    amber_env = detect_amber_environment()
    protein_ffs = amber_env.available_protein_force_fields() or DEFAULT_PROTEIN_FFS
    ligand_ffs = amber_env.available_small_molecule_force_fields() or DEFAULT_LIGAND_FFS
    water_models = amber_env.available_water_models() or DEFAULT_WATER_MODELS
    des_component_map = available_des_components(_des_library_ref_data_dir(state))
    des_components = [
        {
            "key": key.value if isinstance(key, DESComponent) else str(key),
            "label": definition.label,
            "description": definition.description,
        }
        for key, definition in des_component_map.items()
    ]
    recommended = [
        {
            "key": chr(ord("A") + index),
            "components": [item.value for item in components],
            "ratios": list(ratio),
            "label": " : ".join(DES_COMPONENTS[item].label for item in components),
        }
        for index, (components, ratio) in enumerate(DES_RECOMMENDED_SETS)
    ]
    return {
        "api_token": state.api_token,
        "workflow_options": WORKFLOW_OPTIONS,
        "repo_root": str(state.repo_root),
        "launch_cwd": str(state.launch_cwd),
        "protein_force_fields": protein_ffs,
        "ligand_force_fields": ligand_ffs,
        "water_models": water_models,
        "c4_parameter_sets": [
            {
                "key": DESC4ParameterSet.OPC_DUVAIL.value,
                "label": "OPC + Duvail (default)",
                "water_model": "opc",
                "description": "Bundled Duvail ion, polarizability, and C4 data for OPC.",
            },
            {
                "key": DESC4ParameterSet.SPCE_LIMERZ.value,
                "label": "SPC/E + Li/Merz",
                "water_model": "spce",
                "description": "Amber Li/Merz 12-6-4 parameters for SPC/E.",
            },
        ],
        "box_shapes": [BoxShape.OCT.value, BoxShape.CUBIC.value],
        "salt_kinds": [item.value for item in SaltKind],
        "salt_modes": [SaltMode.NONE.value, SaltMode.NEUTRALIZE.value, SaltMode.COUNT.value, SaltMode.CONCENTRATION.value],
        "md_protocols": [ProtocolKind.FIFTEEN_STEP.value, ProtocolKind.FOUR_STEP.value],
        "slurm_profiles": [item.value for item in SlurmProfile],
        "charge_methods": [ChargeMethod.RESP_ANTECHAMBER.value, ChargeMethod.ANTECHAMBER.value],
        "supported_metals": [
            {
                "element": element,
                "charges": list(allowed_metal_charges(element)),
                "duvail_charges": [
                    charge
                    for charge in allowed_metal_charges(element)
                    if c4_parameter_set_supports_metal_charge(
                        DESC4ParameterSet.OPC_DUVAIL,
                        element,
                        charge,
                    )
                ],
                "default_charge": DEFAULT_TLEAP_METAL_CHARGES.get(element),
                "coordination": COORDINATION_NUMBER_HINTS.get(element, ""),
                "coordination_by_charge": _coordination_metadata_for_element(element),
            }
            for element in SUPPORTED_METAL_CHOICES
        ],
        "metal_coordination_defaults": {
            f"{element}{charge}": {
                "element": element,
                "charge": charge,
                "default_cn": record["default_cn"],
                "allowed_cn": list(record["allowed_cn"]),
            }
            for (element, charge), record in METAL_COORDINATION_DEFAULTS.items()
        },
        "resp_group_modes": AUTO_GROUP_MODE_OPTIONS,
        "resp_group_graph_methods": AUTO_GROUP_GRAPH_METHOD_OPTIONS,
        "des_components": des_components,
        "des_recommended_sets": recommended,
        "qm": {
            "geometry_modes": list(QM_GEOMETRY_MODE_OPTIONS),
            "functionals": list(QM_FUNCTIONAL_OPTIONS),
            "basis": list(QM_BASIS_OPTIONS),
            "grids": list(QM_GRID_OPTIONS),
        },
        "environment": environment_summary(),
    }


def create_app(repo_root: Path, *, launch_cwd: Path | None = None) -> Any:
    _require_web_dependencies()
    root = repo_root.resolve()
    state = WebGuiState(
        repo_root=root,
        launch_cwd=(launch_cwd or Path.cwd()).resolve(),
        session_root=Path(tempfile.mkdtemp(prefix="simple_gui_")).resolve(),
    )
    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            yield
        finally:
            shutil.rmtree(state.session_root, ignore_errors=True)

    app = FastAPI(  # type: ignore[operator]
        title="SIMPLE Web GUI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(  # type: ignore[union-attr]
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]"],
    )

    @app.middleware("http")
    async def protect_local_api(request: Request, call_next: Any):  # type: ignore[misc]
        if request.url.path.startswith("/api/") and request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            origin = str(request.headers.get("origin") or "").rstrip("/").lower()
            expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}".rstrip("/").lower()
            if origin and origin != expected_origin:
                return JSONResponse({"ok": False, "error": "Cross-origin API requests are not allowed."}, status_code=403)
            supplied_token = str(request.headers.get("x-simple-token") or "")
            if not supplied_token or not secrets.compare_digest(supplied_token, state.api_token):
                return JSONResponse({"ok": False, "error": "Missing or invalid local GUI session token."}, status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    static_dir = Path(__file__).resolve().parent / "web_static"
    vendor_dir = Path(__file__).resolve().parents[1] / "qm" / "editor" / "assets"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")  # type: ignore[operator]
    app.mount("/vendor", StaticFiles(directory=str(vendor_dir)), name="vendor")  # type: ignore[operator]

    @app.get("/")
    def index():  # type: ignore[misc]
        return FileResponse(static_dir / "index.html")  # type: ignore[operator]

    @app.get("/api/bootstrap")
    def bootstrap():  # type: ignore[misc]
        return _bootstrap_payload(state)

    @app.get("/api/des-library/components")
    def des_library_components():  # type: ignore[misc]
        try:
            return {"ok": True, "components": _des_library_component_payloads(state)}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/scan")
    def des_library_scan(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return scan_des_library_directory(state, payload.get("path"))
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/upload")
    async def des_library_upload(
        files: list[UploadFile] = File(...),
        relative_paths: list[str] = Form(...),
    ):  # type: ignore[misc]
        try:
            if len(files) != len(relative_paths):
                raise ValueError("The browser did not provide a path for every selected file.")
            if len(files) > MAX_UPLOAD_FILES:
                raise ValueError(f"Select no more than {MAX_UPLOAD_FILES} files at once.")
            upload_root = state.session_root / "des_library_uploads" / str(time.time_ns())
            staged: list[Path] = []
            total_bytes = 0
            for file, relative_path in zip(files, relative_paths, strict=True):
                target = _des_library_upload_target(upload_root, relative_path, file.filename)
                if target in staged:
                    raise ValueError(f"Two selected files resolve to the same upload path: {target.name}")
                total_bytes += await _write_upload_limited(file, target, total_bytes=total_bytes)
                staged.append(target)
            if not staged:
                raise ValueError("Select at least one Amber .lib/.off or .frcmod file.")
            state.uploads.extend(staged)
            result = scan_des_library_directory(state, upload_root)
            result["uploaded_files"] = [str(path) for path in staged]
            return result
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/file")
    def des_library_file(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return read_des_library_file(state, payload.get("path"))
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/file/save")
    def des_library_file_save(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return save_des_library_file(state, payload.get("path"), payload.get("content"))
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/register")
    def des_library_register(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return register_des_library_candidate(state, payload)
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des-library/remove")
    def des_library_remove(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return remove_des_library_component(state, payload.get("component_key"))
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):  # type: ignore[misc]
        try:
            target = state.session_root / "uploads" / Path(file.filename or "upload.dat").name
            if target.suffix.lower() not in GENERAL_UPLOAD_SUFFIXES:
                allowed = ", ".join(sorted(GENERAL_UPLOAD_SUFFIXES))
                raise ValueError(f"Unsupported molecular file type. Allowed: {allowed}")
            await _write_upload_limited(file, target)
            state.uploads.append(target)
            return {"ok": True, "path": str(target.resolve())}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/metallophore/load")
    def metallophore_load(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or payload)
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            charge_path: Path | None = None
            if str(data.get("mode") or "resp_input") == "existing_resp":
                job_dir = _path_from_payload(data.get("resp_job_dir"), base=state.launch_cwd)
                candidate = load_resp_job_candidate(job_dir)
                if candidate is None:
                    raise ValueError(f"RESP job manifest was not found under {job_dir}")
                residue_name = str(candidate.payload.get("residue_name") or residue_name).strip().upper()[:3] or residue_name
                source = _resolve_resp_resume_source(candidate, state)
                charge_path = candidate.charge_result_path
            else:
                source = _small_molecule_source(data, state, residue_name=residue_name)
            canonical, molecule = load_small_molecule_for_preview(
                source,
                residue_name=residue_name,
                output_dir=state.session_root / "metallophore_preview",
            )
            molecule = _molecule_with_resp_charges(molecule, charge_path)
            group_constraints = _suggest_group_constraints_for_payload(molecule, data)
            return {
                "ok": True,
                "source_path": str(canonical),
                **molecule_payload(molecule, residue_name=residue_name, group_constraints=group_constraints),
            }
        except Exception as exc:
            data = dict(payload.get("metallophore") or payload)
            return _api_error(exc, state, public_message=_friendly_small_molecule_error(exc, data))

    @app.post("/api/metallophore/groups")
    def metallophore_groups(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or payload)
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            if str(data.get("mode") or "resp_input") == "existing_resp":
                job_dir = _path_from_payload(data.get("resp_job_dir"), base=state.launch_cwd)
                candidate = load_resp_job_candidate(job_dir)
                if candidate is None:
                    raise ValueError(f"RESP job manifest was not found under {job_dir}")
                source = _resolve_resp_resume_source(candidate, state)
            else:
                source = _small_molecule_source(data, state, residue_name=residue_name)
            _canonical, molecule = load_small_molecule_for_preview(
                source,
                residue_name=residue_name,
                output_dir=state.session_root / "metallophore_groups",
            )
            group_constraints = _suggest_group_constraints_for_payload(molecule, data)
            return {"ok": True, "group_constraints": group_constraints}
        except Exception as exc:
            data = dict(payload.get("metallophore") or payload)
            return _api_error(exc, state, public_message=_friendly_small_molecule_error(exc, data))

    @app.post("/api/metallophore/quick-minimize")
    def metallophore_quick_minimize(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or payload)
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            molecule = _edited_molecule_from_payload(data)
            if molecule is None:
                source = _small_molecule_source(data, state, residue_name=residue_name)
                _canonical, molecule = load_small_molecule_for_preview(
                    source,
                    residue_name=residue_name,
                    output_dir=state.session_root / "quick_minimize" / "input",
                )
            coordination = data.get("metal_coordination") if isinstance(data.get("metal_coordination"), dict) else {}
            metal_atom_index = int(coordination.get("metal_atom_index") or data.get("metal_atom_index") or 0)
            atom_by_index = {int(atom.index): atom for atom in molecule.atoms}
            metal_atom = atom_by_index.get(metal_atom_index)
            metal_element = str(coordination.get("element") or getattr(metal_atom, "element", "") or "").strip().title()
            formal_charge = _optional_int(coordination.get("formal_charge"))
            if metal_element and formal_charge is not None and formal_charge not in allowed_metal_charges(metal_element):
                raise ValueError(f"{metal_element}+{formal_charge} is not supported by the current 12-6-4 charge table.")
            manual_coordination = str(coordination.get("coordination_mode") or "").strip().lower() == "manual_selection"
            target_cn = None if manual_coordination else _optional_int(coordination.get("target_cn"))
            if target_cn is None and not manual_coordination:
                target_cn = _default_coordination_number(metal_element, formal_charge)
            required_donors = [
                int(item)
                for item in (
                    coordination.get("required_donor_atom_indices")
                    or data.get("donor_atom_indices")
                    or []
                )
            ]
            effective_donors, auto_filled_donors, coordination_warnings = resolve_coordination_donors(
                molecule,
                metal_atom_index=metal_atom_index,
                required_donor_atom_indices=required_donors,
                target_coordination_number=target_cn,
            )
            output_path, minimized, warnings = quick_minimize_with_openbabel(
                molecule,
                residue_name=residue_name,
                metal_atom_index=metal_atom_index,
                donor_atom_indices=effective_donors,
                output_dir=state.session_root / "quick_minimize",
            )
            warnings = [*coordination_warnings, *warnings]
            group_constraints = _suggest_group_constraints_for_payload(minimized, data)
            return {
                "ok": True,
                "output_path": str(output_path),
                "warnings": warnings,
                "metal_coordination": {
                    "metal_atom_index": metal_atom_index,
                    "element": metal_element,
                    "formal_charge": formal_charge,
                    "coordination_mode": "manual_selection" if manual_coordination else "target_cn",
                    "target_cn": target_cn,
                    "required_donor_atom_indices": required_donors,
                    "effective_donor_atom_indices": effective_donors,
                    "auto_filled_donor_atom_indices": auto_filled_donors,
                },
                **molecule_payload(minimized, residue_name=residue_name, group_constraints=group_constraints),
            }
        except Exception as exc:
            data = dict(payload.get("metallophore") or payload)
            return _api_error(exc, state, public_message=_friendly_small_molecule_error(exc, data))

    @app.post("/api/metallophore/add-metal")
    def metallophore_add_metal(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or payload)
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            molecule = _edited_molecule_from_payload(data)
            if molecule is None:
                source = _small_molecule_source(data, state, residue_name=residue_name)
                _canonical, molecule = load_small_molecule_for_preview(
                    source,
                    residue_name=residue_name,
                    output_dir=state.session_root / "add_metal" / "input",
                )
            insertion = data.get("metal_insertion") if isinstance(data.get("metal_insertion"), dict) else {}
            element = str(insertion.get("element") or data.get("insert_element") or "Fe").strip().title()
            formal_charge = _optional_int(insertion.get("charge") or data.get("insert_charge"))
            if formal_charge is not None and formal_charge not in allowed_metal_charges(element):
                raise ValueError(f"{element}+{formal_charge} is not supported by the current 12-6-4 charge table.")
            required_donors = [
                int(item)
                for item in (
                    insertion.get("donor_atom_indices")
                    or data.get("donor_atom_indices")
                    or []
                )
            ]
            inserted_molecule, metal_atom_index = insert_metal_atom_into_molecule(
                molecule,
                element=element,
                donor_atom_indices=required_donors,
            )
            manual_coordination = str(insertion.get("coordination_mode") or "").strip().lower() == "manual_selection"
            target_cn = None if manual_coordination else _optional_int(insertion.get("target_cn") or data.get("target_cn"))
            if target_cn is None and not manual_coordination:
                target_cn = _default_coordination_number(element, formal_charge)
            effective_donors, auto_filled_donors, coordination_warnings = resolve_coordination_donors(
                inserted_molecule,
                metal_atom_index=metal_atom_index,
                required_donor_atom_indices=required_donors,
                target_coordination_number=target_cn,
            )
            output_path, minimized, warnings = quick_minimize_with_openbabel(
                inserted_molecule,
                residue_name=residue_name,
                metal_atom_index=metal_atom_index,
                donor_atom_indices=effective_donors,
                output_dir=state.session_root / "add_metal",
            )
            warnings = [*coordination_warnings, *warnings]
            group_constraints = _suggest_group_constraints_for_payload(minimized, data)
            return {
                "ok": True,
                "output_path": str(output_path),
                "warnings": warnings,
                "metal_coordination": {
                    "metal_atom_index": metal_atom_index,
                    "element": element,
                    "formal_charge": formal_charge,
                    "coordination_mode": "manual_selection" if manual_coordination else "target_cn",
                    "target_cn": target_cn,
                    "required_donor_atom_indices": required_donors,
                    "effective_donor_atom_indices": effective_donors,
                    "auto_filled_donor_atom_indices": auto_filled_donors,
                },
                **molecule_payload(minimized, residue_name=residue_name, group_constraints=group_constraints),
            }
        except Exception as exc:
            data = dict(payload.get("metallophore") or payload)
            return _api_error(exc, state, public_message=_friendly_small_molecule_error(exc, data))

    @app.post("/api/metallophore/export-pdb")
    def metallophore_export_pdb(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or payload)
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            molecule = _edited_molecule_from_payload(data)
            if molecule is None:
                if str(data.get("mode") or "resp_input") == "existing_resp":
                    job_dir = _path_from_payload(data.get("resp_job_dir"), base=state.launch_cwd)
                    candidate = load_resp_job_candidate(job_dir)
                    if candidate is None:
                        raise ValueError(f"RESP job manifest was not found under {job_dir}")
                    source = _resolve_resp_resume_source(candidate, state)
                else:
                    source = _small_molecule_source(data, state, residue_name=residue_name)
                _canonical, molecule = load_small_molecule_for_preview(
                    source,
                    residue_name=residue_name,
                    output_dir=state.session_root / "metallophore_pdb_export" / "input",
                )
            target_dir = _output_dir_from_payload(payload, state) / "01_prepare" / "pdb_only"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{residue_name}_gui_preview.pdb"
            pdb_text = molecule_to_pdb_text(molecule, residue_name=residue_name)
            target.write_text(pdb_text, encoding="utf-8")
            return {"ok": True, "path": str(target.resolve()), "pdb": pdb_text}
        except Exception as exc:
            data = dict(payload.get("metallophore") or payload)
            return _api_error(exc, state, public_message=_friendly_small_molecule_error(exc, data))

    @app.post("/api/resp/candidates")
    def resp_candidates(payload: dict[str, Any] = Body(default={})):  # type: ignore[misc]
        search_root = _path_from_payload(payload.get("search_root") or state.launch_cwd, base=state.launch_cwd)
        source_file = str(payload.get("source_file") or "").strip()
        explicit = str(payload.get("explicit_job_dir") or "").strip() or None
        if source_file:
            candidates = find_resp_source_candidates(
                search_root=search_root,
                source_file=_path_from_payload(source_file, base=state.launch_cwd),
                explicit_job_dir=explicit,
            )
        else:
            candidates = []
            seen = set()
            for manifest_path in search_root.rglob("resp_apply_manifest.json") if search_root.exists() else []:
                candidate = load_resp_job_candidate(manifest_path.parent.parent)
                if candidate is None or candidate.job_dir in seen:
                    continue
                seen.add(candidate.job_dir)
                candidates.append(candidate)
        return {"ok": True, "candidates": [_candidate_payload(item) for item in candidates]}

    @app.post("/api/resp/build-assets")
    def resp_build_assets(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            data = dict(payload.get("metallophore") or {})
            residue_name = str(data.get("residue_name") or "LIG").strip().upper()[:3] or "LIG"
            source_file = _small_molecule_source(data, state, residue_name=residue_name)
            net_charge = int(data.get("ligands", {}).get("net_charge") or 0)
            multiplicity = int(data.get("ligands", {}).get("multiplicity") or 1)
            charge_method = ChargeMethod(str(data.get("ligands", {}).get("charge_method") or ChargeMethod.RESP_ANTECHAMBER.value))
            seed_dir = _output_dir_from_payload(payload, state) / "01_prepare" / "_gui_resp_seed"
            seed_dir.mkdir(parents=True, exist_ok=True)
            canonical = prepare_canonical_small_molecule_mol2(
                source_file=source_file,
                residue_name=residue_name,
                output_dir=seed_dir,
                split_supported_metals=charge_method != ChargeMethod.FULL_RESP,
                canonical_filename=f"{residue_name}_gui_resp_input.mol2",
            )
            molecule = load_molecule(canonical)
            fingerprint = molecule_fingerprint(source_file, residue_name=residue_name, net_charge=net_charge, multiplicity=multiplicity)
            session_state = build_default_session_state(
                molecule,
                residue_name=residue_name,
                fingerprint=fingerprint,
                net_charge=net_charge,
                multiplicity=multiplicity,
            )
            if isinstance(data.get("group_constraints"), dict):
                session_state["group_constraints"] = data["group_constraints"]
            if isinstance(data.get("qm_settings"), dict):
                session_state["qm_settings"] = normalize_qm_settings(
                    data["qm_settings"],
                    net_charge=net_charge,
                    multiplicity=multiplicity,
                )
            if isinstance(data.get("metal_coordination"), dict):
                session_state["metal_coordination"] = data["metal_coordination"]
            session_state["charge_method"] = charge_method.value
            job_dir = select_job_dir(
                base_dir=_output_dir_from_payload(payload, state) / "01_prepare" / "resp_jobs",
                fingerprint=fingerprint,
                apply_mode=RespApplyMode.NEW_DIRECTORY,
            )
            assets = write_resp_job_assets(
                source_file=source_file,
                coordinate_source_file=canonical,
                resume_source_file=source_file,
                residue_name=residue_name,
                net_charge=net_charge,
                multiplicity=multiplicity,
                job_dir=job_dir,
                slurm_config=_slurm_config(payload.get("slurm") or {}, job_name=str(payload.get("job_name") or residue_name)),
                session_state=session_state,
            )
            return {"ok": True, "assets": assets}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/protein/metal-donor-candidates")
    def protein_metal_donor_candidates(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            protein = dict(payload.get("protein") or payload)
            residue_keys = [
                str(item).strip()
                for item in (
                    payload.get("residue_keys")
                    or protein.get("residue_keys")
                    or []
                )
                if str(item).strip()
            ]
            if not residue_keys:
                raise ValueError("Select one or more residues before requesting donor candidates.")
            _source_kind, _source_value, raw_path = _protein_raw_source(protein, state)
            prepare = _prepare_config(protein.get("prepare") or {}).model_copy(update={"metal_insertions": []})
            result = prepare_structure(
                source=InputSource.PDB_FILE,
                source_value=str(raw_path),
                prepare_config=prepare,
                protonation_config=None,
                kept_ligands=prepare.kept_ligands,
                output_dir=state.session_root / "protein_insert_candidates",
                apply_loop_repair=True,
            )
            structure = inspect_structure(result["cleaned_pdb"], detect_missing_loops=False)
            candidates = donor_candidates_for_residue_selectors(
                load_structure(result["cleaned_pdb"]),
                residue_keys,
            )
            return {
                "ok": True,
                "source_summary": structure.to_dict(),
                "donor_candidates": candidates,
            }
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/protein/load")
    def protein_load(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            protein = dict(payload.get("protein") or payload)
            _source_kind, _source_value, raw_path = _protein_raw_source(protein, state)
            source_summary = inspect_structure(raw_path)
            loop_action = str((payload.get("ui") or {}).get("missing_loop_action") or "").strip().lower()
            prepare = _prepare_config(protein.get("prepare") or {})
            if loop_action not in {"repair", "skip"}:
                dialog_payload = _missing_loop_dialog_payload(source_summary)
                if dialog_payload is not None:
                    return JSONResponse(dialog_payload, status_code=409)  # type: ignore[operator]
            if loop_action in {"repair", "skip"}:
                prepare = prepare.model_copy(update={"repair_missing_loops": loop_action == "repair"})
            result = prepare_structure(
                source=InputSource.PDB_FILE,
                source_value=str(raw_path),
                prepare_config=prepare,
                protonation_config=None,
                kept_ligands=prepare.kept_ligands,
                output_dir=state.session_root / "protein_preview",
                apply_loop_repair=True,
            )
            warnings = [str(item) for item in result.get("warnings") or []]
            if not prepare.remove_other_hetero:
                warnings.append("Non-standard molecules are retained; provide matching force-field parameters before production runs.")
            cleaned_path = Path(result["cleaned_pdb"])
            response = protein_preview_payload(
                cleaned_path,
                missing_loop_locators=_missing_loop_locators(source_summary),
                warnings=warnings,
            )
            inserted_sites = [dict(item) for item in result.get("inserted_metal_sites") or []]
            insertion_links = _metal_insertion_guide_links(inserted_sites)
            if insertion_links:
                response["metal_binding_links"] = [
                    *(response.get("metal_binding_links") or []),
                    *insertion_links,
                ]
            if inserted_sites:
                inserted_keys = [str(item.get("key") or "") for item in inserted_sites if item.get("key")]
                response.setdefault("highlight_sets", {})["inserted_metals"] = inserted_keys
            response["source_path"] = str(raw_path)
            response["path"] = str(cleaned_path)
            response["source_metals"] = [asdict(item) for item in source_summary.metals]
            response["inserted_metal_sites"] = inserted_sites
            response["prepare"] = {
                "raw_input": str(result.get("raw_input") or raw_path),
                "cleaned_pdb": str(cleaned_path),
                "repaired_input": None if result.get("repaired_input") is None else str(result.get("repaired_input")),
                "repair_applied": result.get("repaired_input") is not None,
            }
            return {"ok": True, **response}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/protein/propka")
    def protein_propka(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            protein = dict(payload.get("protein") or {})
            source = (
                fetch_pdb_structure(str(protein.get("pdb_id") or "").strip().upper(), state.session_root / "pdb_downloads")
                if str(protein.get("input_mode") or "path") == "pdb_id"
                else _path_from_payload(protein.get("path"), base=state.launch_cwd)
            )
            prepare = _prepare_config(protein.get("prepare") or {})
            ph = float(protein.get("protonation", {}).get("ph") or 7.0)
            prediction = predict_protonation_prediction(source, prepare, ph=ph)
            candidates = [
                {
                    "chain": item.chain,
                    "seqid": item.seqid,
                    "original_residue_name": item.original_residue_name,
                    "target_residue_name": item.target_residue_name,
                    "predicted_pka": item.predicted_pka,
                    "metal_near": item.metal_near,
                    "reason": item.reason,
                    "selectable": item.selectable,
                    "change": None if item.change is None else item.change.model_dump(mode="json"),
                }
                for item in prediction.metal_coordination_candidates + prediction.propka_candidates
            ]
            return {"ok": True, "warnings": prediction.warnings, "candidates": candidates}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/des/preview")
    def des_preview(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            cfg = build_workflow_config({**payload, "workflow_type": "deep_eutectic"}, state)
            return {"ok": True, **des_heavy_atom_preview(cfg.des, cfg.system.salt)}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/toml")
    def toml(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            config = build_workflow_config(payload, state, stage_gui_inputs=True)
            text = dump_config(config)
            state.last_config = config
            state.last_toml = text
            return {"ok": True, "toml": text}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/toml/save")
    def save_toml(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            config = build_workflow_config(payload, state, stage_gui_inputs=True)
            allowed_dir = _output_dir_from_payload(payload, state)
            requested = str(payload.get("save_path") or "").strip()
            target = Path(requested).expanduser() if requested else allowed_dir / f"{_safe_name(str(payload.get('job_name') or 'simple_gui'))}.toml"
            if not target.is_absolute():
                target = allowed_dir / target
            target = target.resolve()
            if target.suffix.lower() != ".toml":
                raise ValueError("Saved configurations must use the .toml extension.")
            if not _path_is_under(target, allowed_dir):
                raise ValueError(f"The configuration must be saved inside its GUI output directory: {allowed_dir}")
            save_config(config, target)
            state.last_config = config
            state.last_toml = dump_config(config)
            state.last_config_path = target.resolve()
            return {"ok": True, "path": str(target.resolve()), "toml": state.last_toml}
        except Exception as exc:
            return _api_error(exc, state)

    @app.get("/api/toml/download", response_class=PlainTextResponse)
    def download_toml():  # type: ignore[misc]
        return state.last_toml or ""

    @app.post("/api/finish")
    def finish(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            config = build_workflow_config(payload, state, stage_gui_inputs=True)
            config_path = _output_dir_from_payload(payload, state) / f"{_safe_name(str(payload.get('job_name') or 'simple_gui'))}.toml"
            save_config(config, config_path)
            result = run_workflow(config=config, dry_run=False)
            state.last_config = config
            state.last_toml = dump_config(config)
            state.last_config_path = config_path.resolve()
            state.last_result = result
            command = f"python main.py --config {shlex.quote(str(config_path.resolve()))}"
            print(f"SIMPLE GUI workflow complete. To rerun from the command line: {command}")
            return {
                "ok": True,
                "config_path": str(config_path.resolve()),
                "result": result,
                "command": command,
                "resp_pending": result.get("resp") if isinstance(result, dict) else None,
                "protein_site_resp": result.get("protein_site_resp") if isinstance(result, dict) else None,
            }
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/protein-site-resp/candidates")
    def protein_site_resp_candidates(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            return scan_protein_site_resp_directory(state, payload.get("search_root"))
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/protein-site-resp/upload")
    async def protein_site_resp_upload(
        files: list[UploadFile] = File(...),
        relative_paths: list[str] = Form(...),
    ):  # type: ignore[misc]
        try:
            if len(files) != len(relative_paths):
                raise ValueError("The browser did not provide a path for every selected RESP file.")
            if len(files) > MAX_UPLOAD_FILES:
                raise ValueError(f"Select no more than {MAX_UPLOAD_FILES} files at once.")
            upload_root = state.session_root / "protein_site_resp_uploads" / str(time.time_ns())
            staged: list[Path] = []
            total_bytes = 0
            for file, relative_path in zip(files, relative_paths, strict=True):
                target = _site_resp_upload_target(upload_root, relative_path, file.filename)
                if target in staged:
                    raise ValueError(f"Two selected files resolve to the same upload path: {target.name}")
                total_bytes += await _write_upload_limited(file, target, total_bytes=total_bytes)
                staged.append(target)
            if not staged:
                raise ValueError("Select a case folder containing protein-site RESP results.")
            state.uploads.extend(staged)
            result = scan_protein_site_resp_directory(state, upload_root)
            result["uploaded_files"] = [str(path) for path in staged]
            return result
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/slurm/preview")
    def slurm_preview(payload: dict[str, Any] = Body(...)):  # type: ignore[misc]
        try:
            config = build_workflow_config(payload, state)
            from amber_metallo.md_protocols import generate_md_inputs

            temp_dir = state.session_root / "slurm_preview"
            stages = generate_md_inputs(
                config.md,
                temp_dir,
                small_molecule_only=config.input.source == InputSource.SMALL_MOLECULE,
                des_solvent=config.input.source == InputSource.DES,
            )
            return {"ok": True, "script": render_slurm_script(stages=stages, slurm_config=config.slurm)}
        except Exception as exc:
            return _api_error(exc, state)

    @app.post("/api/quit")
    def quit_app():  # type: ignore[misc]
        server = getattr(app.state, "server", None)

        def _stop() -> None:
            if server is not None:
                server.should_exit = True
                if hasattr(server, "force_exit"):
                    server.force_exit = True
            else:
                os._exit(0)

        threading.Timer(0.15, _stop).start()
        return {"ok": True, "message": "Shutting down SIMPLE Web GUI."}

    return app


def _popen_detached(args: list[str]) -> bool:
    kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(args, **kwargs)
    time.sleep(0.25)
    rc = proc.poll()
    return rc is None or rc == 0


def _try_browser_command(command: str, url: str) -> bool:
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return False
    if not parts:
        return False
    exe = parts[0]
    if not Path(exe).is_absolute():
        resolved = shutil.which(exe)
        if not resolved:
            return False
        parts[0] = resolved
    if any("%s" in part for part in parts):
        args = [part.replace("%s", url) for part in parts]
    else:
        args = [*parts, url]
    try:
        return _popen_detached(args)
    except OSError:
        return False


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        text = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8", errors="ignore")
        return "microsoft" in text.lower() or "wsl" in text.lower()
    except Exception:
        return False


def _try_browser_args(args: list[str]) -> bool:
    exe = args[0]
    resolved = exe if Path(exe).is_absolute() else shutil.which(exe)
    if not resolved:
        return False
    try:
        return _popen_detached([resolved, *args[1:]])
    except OSError:
        return False


def _open_browser_url(url: str) -> bool:
    browser_env = os.environ.get("BROWSER", "").strip()
    if browser_env:
        for candidate in browser_env.split(os.pathsep):
            if candidate.strip() and _try_browser_command(candidate.strip(), url):
                return True
    if sys.platform.startswith("win"):
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False

    if sys.platform == "darwin":
        try:
            return _popen_detached(["open", url])
        except OSError:
            return False

    if sys.platform.startswith("linux") and _is_wsl():
        for args in (
            ["powershell.exe", "-NoProfile", "-Command", "Start-Process", url],
            ["cmd.exe", "/c", "start", "", url],
            ["wslview", url],
        ):
            if _try_browser_args(args):
                return True

    for exe in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "microsoft-edge-stable",
        "brave-browser",
        "firefox",
        "wslview",
        "sensible-browser",
        "xdg-open",
    ):
        path = shutil.which(exe)
        if not path:
            continue
        try:
            if _popen_detached([path, url]):
                return True
        except OSError:
            continue
    if _try_browser_args(["cmd.exe", "/c", "start", "", url]):
        return True
    return False


def _print_browser_fallback(url: str, host: str, port: int) -> None:
    print(f"Could not open a browser automatically. Open this URL manually: {url}")
    if sys.platform.startswith("linux"):
        print(
            "If this is a headless/HPC session, automatic popup is not possible. "
            f"Use SSH tunneling, for example: ssh -L {port}:{host}:{port} <user>@<host>"
        )


def run_web_gui(
    repo_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> int:
    _require_web_dependencies()
    normalized_host = str(host).strip().lower()
    try:
        loopback_host = normalized_host == "localhost" or ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        loopback_host = False
    if not loopback_host:
        raise ValueError("The SIMPLE Web GUI may only bind to localhost or a loopback IP address.")
    selected_port = int(port or _find_free_port(host))
    url = f"http://{host}:{selected_port}/"
    if open_browser:
        def open_later() -> None:
            if not _open_browser_url(url):
                _print_browser_fallback(url, host, selected_port)

        threading.Timer(0.8, open_later).start()
    print(f"SIMPLE local web GUI: {url}")
    print("This server is bound to 127.0.0.1 for local use only. Press Ctrl+C to stop.")
    app = create_app(repo_root, launch_cwd=Path.cwd())
    config = uvicorn.Config(app, host=host, port=selected_port, log_level="info")  # type: ignore[union-attr]
    server = uvicorn.Server(config)  # type: ignore[union-attr]
    app.state.server = server
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    finally:
        time.sleep(0.1)
    return 0
