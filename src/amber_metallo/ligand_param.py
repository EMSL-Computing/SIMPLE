from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from amber_metallo.config import (
    ChargeMethod,
    LigandMode,
    RespApplyMode,
    SlurmConfig,
    charge_method_uses_resp,
    normalize_charge_method,
)
from amber_metallo.execution import ensure_execution_host, run_command
from amber_metallo.qm.editor import launch_resp_editor
from amber_metallo.qm.mol2_patch import apply_charges_to_mol2
from amber_metallo.qm.nwchem import (
    build_default_session_state,
    load_molecule,
    load_resp_charges,
    load_session_state,
    molecule_fingerprint,
    MoleculeAtom,
    MoleculeBond,
    MoleculeData,
    render_preview_mol2,
    select_job_dir,
    write_resp_job_assets,
)
from amber_metallo.inspection import SUPPORTED_METALS
from amber_metallo.reporting import print_notice, write_json


@dataclass(slots=True)
class ManualLigandBundle:
    complete: bool
    bundle_type: str | None
    files: dict[str, str]
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LigandArtifacts:
    mode: str
    residue_name: str
    source_file: str | None
    coordinate_source: str | None
    files: dict[str, str]
    commands: list[list[str]]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class CanonicalSourceBundle:
    source_path: Path
    canonical_mol2: Path
    resp_coordinate_mol2: Path
    ligand_reference_mol2: Path
    metal_pdb: str | None
    materialized_pdb: Path | None = None
    full_resp_charge_projection: bool = False


def manual_ligand_requirements_text() -> str:
    return (
        "Accepted manual bundles: mol2 + frcmod, prepi + frcmod, or off/lib + frcmod. "
        "Files must encode atom types, partial charges, bonded connectivity, and residue naming "
        "compatible with the cleaned PDB loaded into tleap."
    )


def validate_manual_ligand_bundle(paths: list[str]) -> ManualLigandBundle:
    normalized = {Path(path).suffix.lower().lstrip("."): str(Path(path).resolve()) for path in paths}
    has_frcmod = "frcmod" in normalized
    if has_frcmod and "mol2" in normalized:
        return ManualLigandBundle(True, "mol2+frcmod", {"mol2": normalized["mol2"], "frcmod": normalized["frcmod"]}, "Complete manual ligand bundle detected.")
    if has_frcmod and "prepi" in normalized:
        return ManualLigandBundle(True, "prepi+frcmod", {"prepi": normalized["prepi"], "frcmod": normalized["frcmod"]}, "Complete manual ligand bundle detected.")
    if has_frcmod and "off" in normalized:
        return ManualLigandBundle(True, "off+frcmod", {"off": normalized["off"], "frcmod": normalized["frcmod"]}, "Complete manual ligand bundle detected.")
    if has_frcmod and "lib" in normalized:
        return ManualLigandBundle(True, "lib+frcmod", {"lib": normalized["lib"], "frcmod": normalized["frcmod"]}, "Complete manual ligand bundle detected.")
    return ManualLigandBundle(False, None, normalized, manual_ligand_requirements_text())


def _copy_manual_files(paths: list[str], output_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        target = output_dir / source.name
        shutil.copy2(source, target)
        copied[target.suffix.lower().lstrip(".")] = str(target)
    return copied


def _input_file_type(source_file: Path) -> str:
    extension = source_file.suffix.lower().lstrip(".")
    if extension not in {"mol2", "sdf", "sd", "pdb", "smi", "smiles", "txt"}:
        raise ValueError(f"Unsupported ligand input format: {source_file.suffix}")
    if extension == "sd":
        return "sdf"
    return extension


def _is_smiles_input(source_file: Path) -> bool:
    return source_file.suffix.lower().lstrip(".") in {"smi", "smiles", "txt"}


def _validate_single_record_smiles_input(source_file: Path) -> None:
    records = [
        line.strip()
        for line in source_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(records) != 1:
        raise ValueError(
            "SMILES input must contain exactly one non-comment record for this workflow. "
            f"Found {len(records)} records in {source_file}."
        )


def _materialized_smiles_paths(source_file: Path, output_dir: Path, residue_name: str) -> tuple[Path, Path]:
    stem = source_file.stem or residue_name
    safe_stem = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in stem)
    return (
        (output_dir / f"{safe_stem}_materialized_from_smiles.mol2").resolve(),
        (output_dir / f"{safe_stem}_materialized_from_smiles.pdb").resolve(),
    )


def _materialize_smiles_input(source_file: Path, *, output_dir: Path, residue_name: str) -> tuple[Path, Path]:
    _validate_single_record_smiles_input(source_file)
    obabel_binary = _openbabel_binary()
    if obabel_binary is None:
        raise RuntimeError(
            "SMILES input requires Open Babel (`obabel` or `babel`) so SIMPLE can generate a 3D MOL2 "
            "and PDB before Amber parameterization. Install Open Babel on this host or provide a MOL2, SDF, "
            "or PDB structure file instead."
        )

    mol2_path, pdb_path = _materialized_smiles_paths(source_file, output_dir, residue_name)
    for output_path, output_format in ((mol2_path, "mol2"), (pdb_path, "pdb")):
        command = [
            obabel_binary,
            "-ismi",
            str(source_file),
            f"-o{output_format}",
            "-O",
            str(output_path),
            "--gen3d",
        ]
        try:
            run_command(command, cwd=output_dir, log_path=output_dir / f"{output_path.stem}.log")
        except RuntimeError as exc:
            raise RuntimeError(
                "Open Babel could not materialize the SMILES input into a 3D structure.\n"
                f"Input: {source_file}\n"
                "Check that the file contains one valid SMILES record, or provide MOL2/SDF/PDB input instead.\n\n"
                f"{exc}"
            ) from exc
        if not output_path.exists():
            raise RuntimeError(
                "Open Babel finished without producing the expected materialized structure.\n"
                f"Missing file: {output_path}"
            )
    return mol2_path, pdb_path


def _is_supported_metal_atom(atom: MoleculeAtom) -> bool:
    return str(atom.element or "").strip().title() in SUPPORTED_METALS


def _split_supported_metals(molecule: MoleculeData) -> tuple[MoleculeData, list[MoleculeAtom]]:
    metal_indices = {int(atom.index) for atom in molecule.atoms if _is_supported_metal_atom(atom)}
    if not metal_indices:
        return molecule, []

    index_map: dict[int, int] = {}
    ligand_atoms: list[MoleculeAtom] = []
    metal_atoms: list[MoleculeAtom] = []
    for atom in molecule.atoms:
        if int(atom.index) in metal_indices:
            metal_atoms.append(atom)
            continue
        new_index = len(ligand_atoms) + 1
        index_map[int(atom.index)] = new_index
        ligand_atoms.append(
            MoleculeAtom(
                index=new_index,
                name=atom.name,
                element=atom.element,
                x=atom.x,
                y=atom.y,
                z=atom.z,
            )
        )

    ligand_bonds = [
        MoleculeBond(
            first=index_map[int(bond.first)],
            second=index_map[int(bond.second)],
            order=int(bond.order or 1),
        )
        for bond in molecule.bonds
        if int(bond.first) in index_map and int(bond.second) in index_map
    ]
    ligand_molecule = MoleculeData(
        source_file=molecule.source_file,
        source_format=molecule.source_format,
        atoms=ligand_atoms,
        bonds=ligand_bonds,
    )
    return ligand_molecule, metal_atoms


def _render_supported_metal_pdb(metal_atoms: list[MoleculeAtom]) -> str:
    lines: list[str] = []
    for serial, atom in enumerate(metal_atoms, start=1):
        element = str(atom.element or atom.name).strip().title()
        residue_name = element.upper()[:3] or "M"
        atom_name = (str(atom.name).strip() or element.upper())[:4]
        lines.append(
            f"HETATM{serial:>5d} {_format_pdb_atom_name(atom_name, element)} "
            f"{residue_name:>3s} M{serial:>4d}    "
            f"{float(atom.x):>8.3f}{float(atom.y):>8.3f}{float(atom.z):>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}          {element[:2].upper():>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def prepare_canonical_small_molecule_mol2(
    *,
    source_file: str | Path,
    residue_name: str,
    output_dir: Path,
    metal_pdb_path: str | Path | None = None,
    split_supported_metals: bool = True,
    canonical_filename: str | None = None,
) -> Path:
    source_path = Path(source_file).expanduser().resolve()
    _input_file_type(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = (output_dir / (canonical_filename or f"{residue_name}_canonical_input.mol2")).resolve()

    structure_source = source_path
    if _is_smiles_input(source_path):
        structure_source, _materialized_pdb = _materialize_smiles_input(
            source_path,
            output_dir=output_dir,
            residue_name=residue_name,
        )

    molecule = load_molecule(structure_source)
    ligand_molecule, metal_atoms = _split_supported_metals(molecule)
    if metal_atoms and split_supported_metals:
        if not ligand_molecule.atoms:
            raise ValueError(
                "Automatic GAFF/GAFF2 ligand parameterization cannot run when the small-molecule input contains "
                "only supported metal atoms and no ligand atoms."
            )
        mol2_text = render_preview_mol2(ligand_molecule, residue_name=residue_name)
        canonical_path.write_text(mol2_text + ("\n" if not mol2_text.endswith("\n") else ""), encoding="utf-8")
        if metal_pdb_path is not None:
            metal_target = Path(metal_pdb_path).expanduser().resolve()
            metal_target.parent.mkdir(parents=True, exist_ok=True)
            metal_target.write_text(_render_supported_metal_pdb(metal_atoms), encoding="utf-8")
        return canonical_path

    if metal_atoms and metal_pdb_path is not None:
        metal_target = Path(metal_pdb_path).expanduser().resolve()
        metal_target.parent.mkdir(parents=True, exist_ok=True)
        metal_target.write_text(_render_supported_metal_pdb(metal_atoms), encoding="utf-8")

    if structure_source.suffix.lower() == ".mol2":
        if structure_source != canonical_path:
            shutil.copy2(structure_source, canonical_path)
        return canonical_path

    mol2_text = render_preview_mol2(molecule, residue_name=residue_name)
    canonical_path.write_text(mol2_text + ("\n" if not mol2_text.endswith("\n") else ""), encoding="utf-8")
    return canonical_path


def _looks_like_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _parse_pdb_serial(line: str, tokens: list[str], fallback: int) -> int:
    serial_text = line[6:11].strip() if len(line) >= 11 else ""
    if serial_text.isdigit():
        return int(serial_text)
    if len(tokens) >= 2 and tokens[1].isdigit():
        return int(tokens[1])
    return fallback


def _format_pdb_atom_name(atom_name: str, element: str) -> str:
    cleaned = atom_name.strip()[:4] or "X"
    element_token = "".join(character for character in element.strip() if character.isalpha())[:2]
    if len(element_token) == 2:
        return cleaned.ljust(4)
    return cleaned.rjust(4)


def _render_strict_antechamber_pdb(source_path: Path, *, output_path: Path) -> Path:
    molecule = load_molecule(source_path)
    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    rendered_lines: list[str] = []
    atom_counter = 0
    for raw_line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith(("ATOM", "HETATM")):
            continue
        atom_counter += 1
        tokens = raw_line.split()
        serial = _parse_pdb_serial(raw_line, tokens, atom_counter)
        atom = atom_lookup.get(serial) or atom_lookup.get(atom_counter)
        if atom is None:
            continue
        atom_name = atom.name.strip()[:4] or f"A{atom_counter}"
        residue_name = (raw_line[17:20].strip() if len(raw_line) >= 20 else "") or (
            tokens[3] if len(tokens) >= 4 and not _looks_like_float_token(tokens[3]) else "LIG"
        )
        residue_name = residue_name[:3] or "LIG"
        chain_id = (raw_line[21].strip() if len(raw_line) >= 22 else "") or (
            tokens[4][:1] if len(tokens) >= 5 and not _looks_like_float_token(tokens[4]) else "A"
        )
        residue_number_text = raw_line[22:26].strip() if len(raw_line) >= 26 else ""
        if not residue_number_text and len(tokens) >= 6 and tokens[5].lstrip("+-").isdigit():
            residue_number_text = tokens[5]
        try:
            residue_number = int(residue_number_text)
        except ValueError:
            residue_number = atom_counter
        insertion_code = raw_line[26].strip()[:1] if len(raw_line) >= 27 else ""
        alt_loc = raw_line[16].strip()[:1] if len(raw_line) >= 17 else ""
        occupancy_text = raw_line[54:60].strip() if len(raw_line) >= 60 else ""
        temp_factor_text = raw_line[60:66].strip() if len(raw_line) >= 66 else ""
        try:
            occupancy = float(occupancy_text)
        except ValueError:
            occupancy = 1.00
        try:
            temp_factor = float(temp_factor_text)
        except ValueError:
            temp_factor = 0.00
        record = "HETATM" if raw_line.startswith("HETATM") else "ATOM  "
        rendered_lines.append(
            f"{record:<6}{serial:>5d} {_format_pdb_atom_name(atom_name, atom.element)}{alt_loc:1s}"
            f"{residue_name:>3s} {chain_id[:1] or 'A'}{residue_number:>4d}{insertion_code:1s}   "
            f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}{occupancy:>6.2f}{temp_factor:>6.2f}"
            f"          {atom.element[:2].upper():>2s}"
        )
    rendered_lines.append("END")
    output_path.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")
    return output_path


def _prepare_parameterization_input(source_file: str | Path, *, output_dir: Path) -> Path:
    source_path = Path(source_file).expanduser().resolve()
    file_type = _input_file_type(source_path)
    if file_type != "pdb":
        return source_path
    sanitized = (output_dir / f"{source_path.stem}_for_antechamber.pdb").resolve()
    return _render_strict_antechamber_pdb(source_path, output_path=sanitized)


def _openbabel_binary() -> str | None:
    return shutil.which("obabel") or shutil.which("babel")


def _prepare_typing_input(
    source_file: str | Path,
    *,
    canonical_mol2_path: Path,
    output_dir: Path,
    separated_metal_pdb: str | None,
) -> tuple[Path, list[list[str]], list[str]]:
    source_path = Path(source_file).expanduser().resolve()
    if separated_metal_pdb is not None:
        return canonical_mol2_path, [], []

    file_type = _input_file_type(source_path)
    if file_type in {"smi", "smiles", "txt"}:
        return canonical_mol2_path, [], [
            "Open Babel materialized the SMILES input into 3D MOL2/PDB files before Amber typing; "
            "Antechamber will use the materialized MOL2."
        ]
    if file_type == "mol2":
        return canonical_mol2_path, [], []
    if file_type != "pdb":
        return source_path, [], []

    sanitized_pdb = _prepare_parameterization_input(source_path, output_dir=output_dir)
    obabel_binary = _openbabel_binary()
    if obabel_binary is None:
        return sanitized_pdb, [], [
            "Open Babel was not detected on this execution host, so small-molecule PDB typing will fall back to "
            "the strict PDB input. Providing MOL2 or SDF is still recommended when bond orders matter."
        ]

    perceived_mol2 = (output_dir / f"{source_path.stem}_perceived_for_antechamber.mol2").resolve()
    command = [
        obabel_binary,
        "-ipdb",
        str(sanitized_pdb),
        "-omol2",
        "-O",
        str(perceived_mol2),
    ]
    return perceived_mol2, [command], [
        "Open Babel will convert the small-molecule PDB to a perceived MOL2 before Antechamber typing so bond "
        "orders and aromaticity can be inferred from the coordinates and CONECT records."
    ]


def _resp_jobs_base_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.parent.name == "ligand_params" and resolved.parent.parent.name == "01_prepare":
        return resolved.parent.parent.parent
    if resolved.name == "01_prepare":
        return resolved.parent
    return resolved.parent


def _resp_pending_artifact(
    *,
    residue_name: str,
    source_path: Path,
    job_assets: dict[str, str],
) -> LigandArtifacts:
    notes = [
        "RESP assets were generated for this ligand, but Amber parameterization is paused until the NWChem job is run.",
        "Run the generated sbatch script, wait for output/resp_charges.json, then rerun the workflow and choose/apply the existing RESP result.",
    ]
    files = {
        "resp_job_dir": job_assets["job_dir"],
        "resume_source_file": job_assets["resume_source_file"],
        "canonical_mol2": job_assets["canonical_source_file"],
        "group_constraints": job_assets["group_constraints"],
        "popup_state": job_assets["popup_state"],
        "nwchem_input": job_assets["nwchem_input"],
        "slurm": job_assets["slurm"],
        "tahoma": job_assets["tahoma"],
        "expected_charge_result": job_assets["expected_charge_result"],
    }
    if job_assets.get("metal_pdb"):
        files["metal_pdb"] = job_assets["metal_pdb"]
    if job_assets.get("ligand_reference_mol2"):
        files["ligand_reference_mol2"] = job_assets["ligand_reference_mol2"]
    if job_assets.get("materialized_pdb"):
        files["materialized_pdb"] = job_assets["materialized_pdb"]
    return LigandArtifacts(
        mode="resp_setup_pending",
        residue_name=residue_name,
        source_file=str(source_path),
        coordinate_source=job_assets.get("canonical_source_file"),
        files=files,
        commands=[],
        notes=notes,
    )


def _describe_possible_open_valence_fragment(molecule: MoleculeData) -> list[str]:
    adjacency: dict[int, list[int]] = {atom.index: [] for atom in molecule.atoms}
    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    for bond in molecule.bonds:
        adjacency.setdefault(int(bond.first), []).append(int(bond.second))
        adjacency.setdefault(int(bond.second), []).append(int(bond.first))

    hints: list[str] = []
    for atom in molecule.atoms:
        if atom.element.upper() != "C":
            continue
        neighbors = [atom_lookup[index] for index in adjacency.get(atom.index, []) if index in atom_lookup]
        hydrogen_neighbors = [neighbor for neighbor in neighbors if neighbor.element.upper() == "H"]
        heavy_neighbors = [neighbor for neighbor in neighbors if neighbor.element.upper() != "H"]
        if len(heavy_neighbors) == 1 and len(hydrogen_neighbors) <= 1:
            neighbor_text = ", ".join(neighbor.name for neighbor in neighbors) or "no neighbors"
            hints.append(
                f"{atom.name} is connected only to {neighbor_text}, which looks like an open-valence fragment "
                "rather than a complete GAFF-typable ligand scaffold."
            )
        elif len(heavy_neighbors) == 0 and len(hydrogen_neighbors) < 3:
            neighbor_text = ", ".join(neighbor.name for neighbor in neighbors) or "no neighbors"
            hints.append(
                f"{atom.name} has only {neighbor_text}; this usually means the input is a truncated fragment."
            )
        if len(hints) >= 3:
            break
    return hints


def _augment_antechamber_failure_message(
    *,
    base_message: str,
    exc: RuntimeError,
    canonical_mol2_path: Path,
    source_path: Path,
    separated_metal_pdb: str | None,
) -> str:
    details = [base_message, "", f"{exc}"]
    if "Weird atomic valence" not in str(exc):
        return "\n".join(details)

    try:
        canonical_molecule = load_molecule(canonical_mol2_path)
    except Exception:
        canonical_molecule = None

    fragment_hints = _describe_possible_open_valence_fragment(canonical_molecule) if canonical_molecule else []
    explanation = (
        "The small-molecule scaffold being sent to Antechamber appears to contain an open valence. "
        "This usually happens when the input PDB is only a coordinating fragment around a metal site, "
        "not the chemically complete ligand."
    )
    if separated_metal_pdb is not None:
        explanation += (
            " SIMPLE separated the supported metal ion before GAFF/GAFF2 typing, "
            "so the remaining ligand-only fragment still has to be chemically complete on its own."
        )
    suggestion = (
        "Please provide the full ligand structure, include the missing heavy-atom connectivity in the input, "
        "or switch to a manual Amber-ready ligand bundle (mol2/prepi/off + frcmod) for this case."
    )
    details.extend(["", explanation])
    if fragment_hints:
        details.append("Detected fragment hints:")
        details.extend(f"  - {hint}" for hint in fragment_hints)
    details.extend(["", suggestion])
    return "\n".join(details)


def _bcc_commands(
    *,
    antechamber_input: Path,
    file_type: str,
    mol2_path: Path,
    frcmod_path: Path,
    atom_type: str,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
) -> list[list[str]]:
    return [
        [
            "antechamber",
            "-i",
            str(antechamber_input),
            "-fi",
            file_type,
            "-o",
            str(mol2_path),
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-nc",
            str(net_charge),
            "-m",
            str(multiplicity),
            "-s",
            "2",
            "-at",
            atom_type,
            "-rn",
            residue_name,
        ],
        [
            "parmchk2",
            "-i",
            str(mol2_path),
            "-f",
            "mol2",
            "-o",
            str(frcmod_path),
            "-s",
            atom_type,
        ],
    ]


def _resp_commands(
    *,
    antechamber_input: Path,
    file_type: str,
    typed_mol2_path: Path,
    final_mol2_path: Path,
    frcmod_path: Path,
    atom_type: str,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
) -> list[list[str]]:
    return [
        [
            "antechamber",
            "-i",
            str(antechamber_input),
            "-fi",
            file_type,
            "-o",
            str(typed_mol2_path),
            "-fo",
            "mol2",
            "-c",
            "gas",
            "-nc",
            str(net_charge),
            "-m",
            str(multiplicity),
            "-s",
            "2",
            "-at",
            atom_type,
            "-rn",
            residue_name,
        ],
        [
            "parmchk2",
            "-i",
            str(final_mol2_path),
            "-f",
            "mol2",
            "-o",
            str(frcmod_path),
            "-s",
            atom_type,
        ],
    ]


def _resp_job_metal_pdb(resp_job_dir: str | Path | None) -> str | None:
    if resp_job_dir is None:
        return None
    manifest_path = Path(resp_job_dir).expanduser().resolve() / "manifests" / "resp_apply_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw_path = payload.get("metal_pdb_file") or (payload.get("files") or {}).get("metal_pdb")
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser().resolve()
    return str(candidate) if candidate.exists() else None


def _smiles_materialized_pdb_if_present(source_path: Path, output_dir: Path, residue_name: str) -> Path | None:
    if not _is_smiles_input(source_path):
        return None
    _mol2_path, pdb_path = _materialized_smiles_paths(source_path, output_dir, residue_name)
    return pdb_path if pdb_path.exists() else None


def _write_ligand_reference_mol2_from_full_resp(
    *,
    full_resp_mol2_path: Path,
    residue_name: str,
    output_dir: Path,
) -> tuple[Path, bool]:
    full_molecule = load_molecule(full_resp_mol2_path)
    ligand_molecule, metal_atoms = _split_supported_metals(full_molecule)
    if not metal_atoms:
        return full_resp_mol2_path, False
    if not ligand_molecule.atoms:
        raise ValueError(
            "Full RESP cannot continue because the metal-inclusive small-molecule input contains no ligand atoms "
            "after supported metal atoms are separated for the Amber handoff."
        )
    ligand_reference_path = (output_dir / f"{residue_name}_canonical_input.mol2").resolve()
    mol2_text = render_preview_mol2(ligand_molecule, residue_name=residue_name)
    ligand_reference_path.write_text(mol2_text + ("\n" if not mol2_text.endswith("\n") else ""), encoding="utf-8")
    return ligand_reference_path, True


def _prepare_canonical_source_bundle(
    *,
    source_path: Path,
    charge_method: ChargeMethod,
    residue_name: str,
    output_dir: Path,
    metal_pdb_path: Path,
) -> CanonicalSourceBundle:
    selected_charge_method = normalize_charge_method(charge_method)
    if selected_charge_method == ChargeMethod.FULL_RESP:
        resp_coordinate_mol2 = prepare_canonical_small_molecule_mol2(
            source_file=source_path,
            residue_name=residue_name,
            output_dir=output_dir,
            metal_pdb_path=metal_pdb_path,
            split_supported_metals=False,
            canonical_filename=f"{residue_name}_full_resp_input.mol2",
        )
        ligand_reference_mol2, needs_projection = _write_ligand_reference_mol2_from_full_resp(
            full_resp_mol2_path=resp_coordinate_mol2,
            residue_name=residue_name,
            output_dir=output_dir,
        )
        return CanonicalSourceBundle(
            source_path=source_path,
            canonical_mol2=ligand_reference_mol2,
            resp_coordinate_mol2=resp_coordinate_mol2,
            ligand_reference_mol2=ligand_reference_mol2,
            metal_pdb=str(metal_pdb_path) if metal_pdb_path.exists() else None,
            materialized_pdb=_smiles_materialized_pdb_if_present(source_path, output_dir, residue_name),
            full_resp_charge_projection=needs_projection,
        )

    canonical_mol2 = prepare_canonical_small_molecule_mol2(
        source_file=source_path,
        residue_name=residue_name,
        output_dir=output_dir,
        metal_pdb_path=metal_pdb_path,
    )
    return CanonicalSourceBundle(
        source_path=source_path,
        canonical_mol2=canonical_mol2,
        resp_coordinate_mol2=canonical_mol2,
        ligand_reference_mol2=canonical_mol2,
        metal_pdb=str(metal_pdb_path) if metal_pdb_path.exists() else None,
        materialized_pdb=_smiles_materialized_pdb_if_present(source_path, output_dir, residue_name),
        full_resp_charge_projection=False,
    )


def project_ligand_charges_from_full_resp(
    *,
    full_resp_mol2_path: str | Path,
    charges: list[float],
) -> list[float]:
    molecule = load_molecule(full_resp_mol2_path)
    if len(molecule.atoms) != len(charges):
        raise ValueError(
            "Full RESP charge projection failed because the RESP charge count does not match the metal-inclusive "
            f"QM structure. Expected {len(molecule.atoms)}, received {len(charges)}."
        )
    projected = [
        float(charges[position])
        for position, atom in enumerate(molecule.atoms)
        if not _is_supported_metal_atom(atom)
    ]
    if not projected:
        raise ValueError("Full RESP charge projection found no ligand atoms after supported metals were excluded.")
    return projected


def parameterize_ligand(
    *,
    source_file: str | Path | None,
    mode: LigandMode,
    charge_method: ChargeMethod,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
    manual_files: list[str],
    output_dir: Path,
    slurm_config: SlurmConfig | None = None,
    resp_job_dir: str | Path | None = None,
    resp_group_file: str | Path | None = None,
    resp_session_file: str | Path | None = None,
    resp_apply_mode: RespApplyMode = RespApplyMode.DETECT,
    allow_popup: bool = False,
    dry_run: bool,
) -> LigandArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    selected_charge_method = normalize_charge_method(charge_method)

    if mode == LigandMode.MANUAL:
        bundle = validate_manual_ligand_bundle(manual_files)
        files = _copy_manual_files(manual_files, output_dir) if manual_files else {}
        notes.append(bundle.message)
        result = LigandArtifacts(
            mode=mode.value,
            residue_name=residue_name,
            source_file=str(source_file) if source_file else None,
            coordinate_source=str(source_file) if source_file else files.get("mol2"),
            files=files,
            commands=[],
            notes=notes,
        )
        write_json(output_dir / "ligand_manifest.json", result.to_dict())
        return result

    if source_file is None:
        raise ValueError("Automatic ligand parameterization requires a source file.")

    source_path = Path(source_file).expanduser().resolve()
    separated_metal_pdb_path = (output_dir / f"{residue_name}_separated_metals.pdb").resolve()
    canonical_bundle = _prepare_canonical_source_bundle(
        source_path=source_path,
        charge_method=selected_charge_method,
        residue_name=residue_name,
        output_dir=output_dir,
        metal_pdb_path=separated_metal_pdb_path,
    )
    canonical_mol2_path = canonical_bundle.canonical_mol2
    resp_coordinate_mol2_path = canonical_bundle.resp_coordinate_mol2
    separated_metal_pdb = canonical_bundle.metal_pdb
    mol2_path = output_dir / f"{residue_name}.mol2"
    typed_mol2_path = output_dir / f"{residue_name}_typed.mol2"
    frcmod_path = output_dir / f"{residue_name}.frcmod"
    atom_type = "gaff2" if mode == LigandMode.GAFF2 else "gaff"
    notes.append(
        "RESP setup and the downstream Amber handoff are MOL2-centric. "
        "When the accepted input is not already MOL2, SIMPLE still keeps the first Antechamber typing step on the original chemistry-aware source to preserve bond-order information."
    )
    notes.append(
        "Automatic ligand parameterization uses Antechamber to generate a MOL2 template "
        "and parmchk2 to generate an FRCMOD file for tleap."
    )
    notes.append(
        f"Requested ligand electronic settings: net charge = {net_charge}, multiplicity = {multiplicity}."
    )
    if separated_metal_pdb is not None:
        notes.append(
            "Supported metal atom(s) were separated before GAFF/GAFF2 parameterization so Antechamber/SQM runs "
            "on the ligand-only scaffold. The separated metal PDB is retained for tleap and 12-6-4 post-processing; "
            "the ligand net charge should exclude the separated metal ion charge."
        )
    if canonical_bundle.materialized_pdb is not None:
        notes.append(
            "The SMILES input was materialized with Open Babel into 3D MOL2/PDB files before the normal "
            "structure-based parameterization path was used."
        )
    if selected_charge_method == ChargeMethod.FULL_RESP:
        notes.append(
            "Full RESP uses the metal-inclusive structure for QM/RESP and projects only ligand atom charges back "
            "onto the ligand-only MOL2 used by Amber; atom indices shown in the RESP editor remain the original "
            "metal-inclusive indices."
        )

    if charge_method_uses_resp(selected_charge_method):
        if resp_apply_mode != RespApplyMode.APPLY_EXISTING:
            resume_source_path = (
                source_path
                if _is_smiles_input(source_path)
                else _prepare_parameterization_input(source_path, output_dir=output_dir)
            )
            molecule = load_molecule(resp_coordinate_mol2_path)
            fingerprint = molecule_fingerprint(
                source_path,
                residue_name=residue_name,
                net_charge=net_charge,
                multiplicity=multiplicity,
            )
            default_session = build_default_session_state(
                molecule,
                residue_name=residue_name,
                fingerprint=fingerprint,
                net_charge=net_charge,
                multiplicity=multiplicity,
            )
            default_session["canonical_source_file"] = str(resp_coordinate_mol2_path)
            if selected_charge_method == ChargeMethod.FULL_RESP:
                default_session["charge_method"] = ChargeMethod.FULL_RESP.value
                default_session["ligand_reference_mol2"] = str(canonical_bundle.ligand_reference_mol2)
            session_state = load_session_state(resp_session_file) if resp_session_file else default_session
            session_state["charge_method"] = selected_charge_method.value
            if allow_popup and resp_group_file is None and resp_session_file is None:
                session_state = launch_resp_editor(session_state=default_session, output_dir=output_dir)
                if session_state.get("editor_mode") == "cancelled":
                    raise RuntimeError("RESP popup was cancelled before any NWChem assets were generated.")
                if session_state.get("editor_mode") == "auto_defaults":
                    warning = str(session_state.get("editor_warning") or "No popup backend could be launched.")
                    raise RuntimeError(
                        "RESP popup could not be launched, so the workflow stopped before asset generation.\n"
                        f"{warning}\n"
                        "Please fix the popup backend and rerun the RESP setup."
                    )
                if session_state.get("editor_warning"):
                    print_notice(
                        "RESP Popup Unavailable",
                        f"{session_state['editor_warning']}\n"
                        "Continuing with auto-suggested charge-equivalence groups and the default RESP preset instead.",
                        border_style="yellow",
                    )
            session_state["canonical_source_file"] = str(resp_coordinate_mol2_path)
            selected_job_dir = select_job_dir(
                base_dir=_resp_jobs_base_dir(output_dir),
                fingerprint=fingerprint,
                apply_mode=resp_apply_mode,
                existing_job_dir=resp_job_dir,
            )
            job_assets = write_resp_job_assets(
                source_file=source_path,
                coordinate_source_file=resp_coordinate_mol2_path,
                resume_source_file=resume_source_path,
                metal_pdb_file=separated_metal_pdb,
                residue_name=residue_name,
                net_charge=net_charge,
                multiplicity=multiplicity,
                job_dir=selected_job_dir,
                slurm_config=slurm_config or SlurmConfig(),
                session_state=session_state,
                group_file=resp_group_file,
            )
            if canonical_bundle.full_resp_charge_projection:
                job_assets["ligand_reference_mol2"] = str(canonical_bundle.ligand_reference_mol2)
            if canonical_bundle.materialized_pdb is not None:
                job_assets["materialized_pdb"] = str(canonical_bundle.materialized_pdb)
            result = _resp_pending_artifact(
                residue_name=residue_name,
                source_path=source_path,
                job_assets=job_assets,
            )
            write_json(output_dir / "ligand_manifest.json", result.to_dict())
            return result

        if resp_job_dir is None:
            raise ValueError("RESP apply mode requires an existing resp_job_dir.")
        antechamber_input, prep_commands, prep_notes = _prepare_typing_input(
            source_path,
            canonical_mol2_path=canonical_mol2_path,
            output_dir=output_dir,
            separated_metal_pdb=separated_metal_pdb,
        )
        notes.extend(prep_notes)
        file_type = _input_file_type(antechamber_input)
        charges = load_resp_charges(resp_job_dir)
        if canonical_bundle.full_resp_charge_projection:
            charges = project_ligand_charges_from_full_resp(
                full_resp_mol2_path=resp_coordinate_mol2_path,
                charges=charges,
            )
        if separated_metal_pdb is None:
            separated_metal_pdb = _resp_job_metal_pdb(resp_job_dir)
        commands = prep_commands + _resp_commands(
            antechamber_input=antechamber_input,
            file_type=file_type,
            typed_mol2_path=typed_mol2_path,
            final_mol2_path=mol2_path,
            frcmod_path=frcmod_path,
            atom_type=atom_type,
            residue_name=residue_name,
            net_charge=net_charge,
            multiplicity=multiplicity,
        )
        notes.append(f"RESP charges will be read from {Path(resp_job_dir).expanduser().resolve()}.")
        if not dry_run:
            ensure_execution_host(dry_run=False)
            for index, command in enumerate(prep_commands, start=1):
                run_command(command, cwd=output_dir, log_path=output_dir / f"ligand_step_{index}.log")
                if not Path(command[-1]).exists():
                    raise RuntimeError(
                        "Small-molecule PDB preprocessing finished without producing the expected perceived MOL2 file.\n"
                        f"Missing file: {command[-1]}"
                    )
            try:
                run_command(
                    commands[len(prep_commands)],
                    cwd=output_dir,
                    log_path=output_dir / f"ligand_step_{len(prep_commands) + 1}.log",
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    _augment_antechamber_failure_message(
                        base_message=(
                            "Amber GAFF atom typing could not be completed for this RESP ligand, so the workflow cannot "
                            "continue into bonded-parameter generation.\n"
                            "Missing or inconsistent GAFF/GAFF2 typing information was detected while preparing the typed "
                            "MOL2 template."
                        ),
                        exc=exc,
                        canonical_mol2_path=canonical_mol2_path,
                        source_path=source_path,
                        separated_metal_pdb=separated_metal_pdb,
                    )
                ) from exc
            if not typed_mol2_path.exists():
                raise RuntimeError(
                    "Amber GAFF atom typing finished without producing the expected typed MOL2 file.\n"
                    f"Missing file: {typed_mol2_path}"
                )
            apply_charges_to_mol2(
                typed_mol2_path,
                charges,
                output_mol2=mol2_path,
                reference_structure=canonical_mol2_path,
            )
            try:
                run_command(
                    commands[len(prep_commands) + 1],
                    cwd=output_dir,
                    log_path=output_dir / f"ligand_step_{len(prep_commands) + 2}.log",
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Amber GAFF bonded-parameter generation could not be completed for this RESP ligand.\n"
                    "The workflow could not obtain the bonded terms needed for the FRCMOD file from the available "
                    "GAFF/GAFF2 typing information.\n\n"
                    f"{exc}"
                ) from exc
            if not frcmod_path.exists():
                raise RuntimeError(
                    "Amber GAFF bonded-parameter generation finished without producing the expected FRCMOD file.\n"
                    f"Missing file: {frcmod_path}"
                )
        files = {
            "canonical_mol2": str(canonical_mol2_path),
            "mol2": str(mol2_path),
            "frcmod": str(frcmod_path),
            "resp_job_dir": str(Path(resp_job_dir).expanduser().resolve()),
        }
        if resp_coordinate_mol2_path != canonical_mol2_path:
            files["full_resp_mol2"] = str(resp_coordinate_mol2_path)
            files["ligand_reference_mol2"] = str(canonical_bundle.ligand_reference_mol2)
        if separated_metal_pdb is not None:
            files["metal_pdb"] = separated_metal_pdb
        if canonical_bundle.materialized_pdb is not None:
            files["materialized_pdb"] = str(canonical_bundle.materialized_pdb)
        result = LigandArtifacts(
            mode=f"{mode.value}+{selected_charge_method.value}",
            residue_name=residue_name,
            source_file=str(source_path),
            coordinate_source=str(mol2_path),
            files=files,
            commands=commands,
            notes=notes,
        )
        write_json(output_dir / "ligand_manifest.json", result.to_dict())
        return result

    antechamber_input, prep_commands, prep_notes = _prepare_typing_input(
        source_path,
        canonical_mol2_path=canonical_mol2_path,
        output_dir=output_dir,
        separated_metal_pdb=separated_metal_pdb,
    )
    notes.extend(prep_notes)
    file_type = _input_file_type(antechamber_input)

    commands = prep_commands + _bcc_commands(
        antechamber_input=antechamber_input,
        file_type=file_type,
        mol2_path=mol2_path,
        frcmod_path=frcmod_path,
        atom_type=atom_type,
        residue_name=residue_name,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )

    if not dry_run:
        ensure_execution_host(dry_run=False)
        for index, command in enumerate(commands, start=1):
            try:
                run_command(command, cwd=output_dir, log_path=output_dir / f"ligand_step_{index}.log")
            except RuntimeError as exc:
                if prep_commands and index <= len(prep_commands):
                    raise RuntimeError(
                        "Small-molecule PDB preprocessing could not be completed before Antechamber typing.\n\n"
                        f"{exc}"
                    ) from exc
                if index == len(prep_commands) + 1:
                    raise RuntimeError(
                        _augment_antechamber_failure_message(
                            base_message=(
                                "Amber GAFF atom typing could not be completed for this ligand.\n"
                                "Missing or inconsistent GAFF/GAFF2 typing information was detected while preparing the "
                                "ligand template."
                            ),
                            exc=exc,
                            canonical_mol2_path=canonical_mol2_path,
                            source_path=source_path,
                            separated_metal_pdb=separated_metal_pdb,
                        )
                    ) from exc
                raise RuntimeError(
                    "Amber GAFF bonded-parameter generation could not be completed for this ligand.\n"
                    "The workflow could not obtain the bonded terms needed for the FRCMOD file from the available "
                    "GAFF/GAFF2 typing information.\n\n"
                    f"{exc}"
                ) from exc
            if prep_commands and index <= len(prep_commands) and not Path(command[-1]).exists():
                raise RuntimeError(
                    "Small-molecule PDB preprocessing finished without producing the expected perceived MOL2 file.\n"
                    f"Missing file: {command[-1]}"
                )
        if not mol2_path.exists():
            raise RuntimeError(
                "Amber GAFF parameter generation finished without producing the expected MOL2 file.\n"
                f"Missing file: {mol2_path}"
            )
        if not frcmod_path.exists():
            raise RuntimeError(
                "Amber GAFF bonded-parameter generation finished without producing the expected FRCMOD file.\n"
                f"Missing file: {frcmod_path}"
            )

    files = {
        "canonical_mol2": str(canonical_mol2_path),
        "mol2": str(mol2_path),
        "frcmod": str(frcmod_path),
    }
    if separated_metal_pdb is not None:
        files["metal_pdb"] = separated_metal_pdb
    if canonical_bundle.materialized_pdb is not None:
        files["materialized_pdb"] = str(canonical_bundle.materialized_pdb)
    result = LigandArtifacts(
        mode=f"{mode.value}+{selected_charge_method.value}",
        residue_name=residue_name,
        source_file=str(source_path),
        coordinate_source=str(mol2_path),
        files=files,
        commands=commands,
        notes=notes,
    )
    write_json(output_dir / "ligand_manifest.json", result.to_dict())
    return result
