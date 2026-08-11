from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import typer

from amber_metallo.amber.leap import (
    build_protein_site_resp_reference_with_tleap,
    build_system_with_tleap,
)
from amber_metallo.config import (
    InputSource,
    LigandMode,
    MetalChargeAssignment,
    ProteinSiteRespClusterConfig,
    ProteinSiteRespConfig,
    ProteinSiteRespMode,
    RespApplyMode,
    ResidueMaskNumbering,
    SystemConfig,
    WorkflowConfig,
    charge_method_uses_resp,
    normalize_charge_method,
    save_config,
)
from amber_metallo.des import build_des_system
from amber_metallo.environment import detect_amber_environment
from amber_metallo.inspection import load_structure
from amber_metallo.ligand_param import (
    LigandArtifacts,
    parameterize_ligand,
    validate_manual_ligand_bundle,
)
from amber_metallo.md_protocols import generate_md_inputs
from amber_metallo.prep import prepare_structure
from amber_metallo.protein_site_resp import (
    apply_site_resp_results,
    build_site_resp_jobs,
    discover_site_clusters,
    load_topology_atoms,
    review_site_resp_result,
    suggested_low_spin_multiplicity,
    suggested_spin_multiplicity,
    validate_retained_direct_environment,
)
from amber_metallo.qm.nwchem import find_resp_job_candidates, load_resp_job_candidate, molecule_fingerprint
from amber_metallo.reporting import activity_status, print_notice, write_json
from amber_metallo.slurm import write_slurm_script

COMMON_GLYCAN_RESIDUES = {"NAG", "NDG", "MAN", "BMA", "GAL", "FUC", "NANA", "SIA"}
_SIMPLE_NUMERIC_RESIDUE_MASK = re.compile(
    r"^\s*\(?\s*:(?P<residues>\d+(?:\s*,\s*\d+)*)\s*\)?(?P<suffix>\s*(?:&.*)?)\s*$"
)


def _stage_directories(root: Path) -> dict[str, Path]:
    return {
        "prepare": root / "01_prepare",
        "system": root / "02_system",
        "md": root / "03_md",
    }


def _resolve_ligand_electronic_settings(config: WorkflowConfig, residue_name: str) -> tuple[int, int]:
    residue_key = residue_name.strip().upper()
    for assignment in config.ligands.parameter_assignments:
        if assignment.residue_name == residue_key:
            return int(assignment.net_charge), int(assignment.multiplicity)
    return int(config.ligands.net_charge), int(config.ligands.multiplicity)


def _selected_ligand_forcefield_label(config: WorkflowConfig) -> str:
    return "GAFF" if config.ligands.mode == LigandMode.GAFF else "GAFF2"


def _ordered_residue_numbers(path: Path) -> list[int]:
    structure = load_structure(path)
    while len(structure) > 1:
        del structure[1]
    residue_numbers: list[int] = []
    for chain in structure[0]:
        for residue in chain:
            residue_numbers.append(int(residue.seqid.num))
    return residue_numbers


def _translate_pdb_number_mask_to_topology(mask: str | None, prepared_pdb: Path | None, final_pdb: Path | None) -> str | None:
    if mask is None or prepared_pdb is None or final_pdb is None:
        return mask
    if not prepared_pdb.exists() or not final_pdb.exists():
        return mask

    match = _SIMPLE_NUMERIC_RESIDUE_MASK.fullmatch(mask)
    if not match:
        return mask

    requested_numbers = {
        int(token.strip()) for token in match.group("residues").split(",") if token.strip()
    }
    if not requested_numbers:
        return mask

    prepared_numbers = _ordered_residue_numbers(prepared_pdb)
    final_numbers = _ordered_residue_numbers(final_pdb)
    if len(final_numbers) < len(prepared_numbers):
        return mask

    translated_indices = [
        str(index)
        for index, residue_number in enumerate(prepared_numbers, start=1)
        if residue_number in requested_numbers
    ]
    if not translated_indices:
        return mask

    suffix = match.group("suffix") or ""
    return f"(:{','.join(translated_indices)}){suffix}"


def _translate_focused_restraint_mask_to_topology(config: WorkflowConfig, prepared_pdb: Path | None, final_pdb: Path | None) -> None:
    if config.md.focused_restraint_mask_numbering == ResidueMaskNumbering.PREPARED:
        return
    translated = _translate_pdb_number_mask_to_topology(config.md.focused_restraint_mask, prepared_pdb, final_pdb)
    if translated and translated != config.md.focused_restraint_mask:
        config.md.focused_restraint_mask = translated
        config.md.focused_restraint_mask_numbering = ResidueMaskNumbering.PREPARED


def _system_config_with_inserted_metal_charges(
    system_config: SystemConfig,
    prepared: dict[str, Any] | None,
) -> SystemConfig:
    assignments = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    for item in (prepared or {}).get("inserted_metal_sites") or []:
        site = item.get("site")
        charge = item.get("charge")
        if site is None or charge is None:
            continue
        assignments.setdefault(int(site), int(charge))
    if len(assignments) == len(system_config.metal_charges):
        return system_config
    return system_config.model_copy(
        update={
            "metal_charges": [
                MetalChargeAssignment(site=site, charge=charge)
                for site, charge in sorted(assignments.items())
            ]
        }
    )


def _load_prepare_manifest(prepare_dir: Path) -> dict[str, Any] | None:
    manifest_path = prepare_dir / "prepare_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _source_identity_label(config: WorkflowConfig) -> str:
    if config.input.source == InputSource.PDB_ID:
        return str(config.input.pdb_id or "PDB").strip().upper()
    if config.input.path:
        return Path(config.input.path).expanduser().stem
    return "protein"


def _confirm_protein_site_resp_application(jobs: list[dict[str, object]]) -> bool:
    reviews = [review_site_resp_result(str(job["job_dir"])) for job in jobs]
    lines = [
        "A completed, fingerprint-matching protein-site RESP result was detected.",
        "Review the site-specific charge changes before applying this experimental hybrid model:",
    ]
    for review in reviews:
        lines.append(f"\n{review.get('description')}")
        fitted_rmse = float(review.get("esp_rmse") or 0.0)
        baseline_rmse_raw = review.get("baseline_esp_rmse")
        baseline_rmse = None if baseline_rmse_raw is None else float(baseline_rmse_raw)
        improvement = (
            None
            if baseline_rmse is None or baseline_rmse <= 1.0e-15
            else 100.0 * (baseline_rmse - fitted_rmse) / baseline_rmse
        )
        lines.append(
            f"ESP RMSE: {fitted_rmse:.6g} | baseline ff19SB RMSE: "
            f"{baseline_rmse:.6g}" if baseline_rmse is not None else "ESP RMSE baseline: unavailable"
        )
        if improvement is not None:
            lines.append(f"ESP RMSE improvement over baseline: {improvement:+.2f}%")
        lines.append(
            "Normalized ESP residual: "
            f"{float(review.get('esp_relative_rmse')):.4f} | "
            if review.get("esp_relative_rmse") is not None
            else "Normalized ESP residual: unavailable | "
        )
        lines[-1] += (
            "maximum constraint residual: "
            f"{float(review.get('maximum_constraint_residual') or 0.0):.3e}"
        )
        for change in list(review.get("changes") or []):
            lines.append(
                f"- {change.get('residue_key')}@{change.get('atom_name')}: "
                f"{float(change.get('original_charge') or 0.0):+.6f} -> "
                f"{float(change.get('charge') or 0.0):+.6f} "
                f"({float(change.get('delta') or 0.0):+.6f})"
            )
        for residue_sum in list(review.get("residue_sums") or []):
            lines.append(
                f"Residue sum {residue_sum.get('label')}: "
                f"{float(residue_sum.get('baseline') or 0.0):+.6f} -> "
                f"{float(residue_sum.get('fitted') or 0.0):+.6f}"
            )
        if review.get("symmetry_constraints"):
            lines.append(
                f"Verified symmetry constraints: {len(list(review.get('symmetry_constraints') or []))}"
            )
        for warning in list(review.get("warnings") or []):
            lines.append(f"WARNING: {warning}")
    print_notice("Protein-Site RESP Review", "\n".join(lines), border_style="yellow")
    return typer.confirm("Apply these protein-site RESP charges to the final topology?", default=False)


def _review_protein_site_resp_clusters(
    *,
    resp_config: ProteinSiteRespConfig,
    system_config: SystemConfig,
    system_pdb: Path,
    system_prmtop: Path,
) -> ProteinSiteRespConfig:
    """Require terminal users to confirm cluster membership and spin after TLeap mapping."""
    probe_config = resp_config.model_copy(deep=True)
    if probe_config.default_multiplicity is None:
        probe_config.default_multiplicity = 1
    atoms = load_topology_atoms(system_prmtop, system_pdb)
    clusters = discover_site_clusters(
        atoms=atoms,
        system_config=system_config,
        resp_config=probe_config,
    )
    atom_by_index = {atom.topology_index: atom for atom in atoms}
    charge_by_site = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    reviewed: list[ProteinSiteRespClusterConfig] = []
    for number, cluster in enumerate(clusters, start=1):
        species = [
            (
                atom_by_index[topology_index].element,
                charge_by_site[site],
            )
            for site, topology_index in zip(cluster.metal_sites, cluster.metal_atom_indices, strict=True)
        ]
        suggested = (
            suggested_spin_multiplicity(species[0][0], species[0][1])
            if len(species) == 1
            else None
        )
        low_spin = (
            suggested_low_spin_multiplicity(species[0][0], species[0][1])
            if len(species) == 1
            else None
        )
        species_text = ", ".join(f"{element}{charge:+d}" for element, charge in species)
        print_notice(
            f"Protein-Site RESP Cluster {number}",
            "Connected metal site(s): " + ", ".join(str(site) for site in cluster.metal_sites)
            + f" ({species_text})\nDirect protein donors: "
            + (", ".join(cluster.donor_residue_keys) or "None")
            + "\nFixed QM environment: "
            + (", ".join(cluster.fixed_environment_keys) or "None")
            + "\nDefault/high-spin multiplicity: "
            + (str(suggested) if suggested is not None else "no single curated value")
            + "\nLow-spin alternative: "
            + (
                str(low_spin)
                if low_spin is not None
                else "not conventionally assigned for this ion; inspect the electronic state"
            ),
            border_style="yellow",
        )
        donor_text = typer.prompt(
            "Direct donor residues (comma separated; review/edit)",
            default=", ".join(cluster.donor_residue_keys),
        ).strip()
        fixed_text = typer.prompt(
            "Fixed QM-environment residues (comma separated; review/edit; blank allowed)",
            default=", ".join(cluster.fixed_environment_keys),
            show_default=bool(cluster.fixed_environment_keys),
        ).strip()
        multiplicity = typer.prompt(
            "Confirmed spin multiplicity for this connected cluster",
            default=int(suggested or resp_config.default_multiplicity or 1),
            type=int,
        )
        if multiplicity < 1:
            raise ValueError("Protein-site RESP spin multiplicity must be at least 1.")
        if not typer.confirm(
            f"Use multiplicity {multiplicity} for cluster {number} ({species_text})?",
            default=True,
        ):
            raise ValueError("Protein-site RESP multiplicity was not confirmed.")
        reviewed.append(
            ProteinSiteRespClusterConfig(
                metal_sites=cluster.metal_sites,
                donor_residues=[item.strip() for item in donor_text.split(",") if item.strip()],
                fixed_environment=[item.strip() for item in fixed_text.split(",") if item.strip()],
                multiplicity=multiplicity,
            )
        )
    return resp_config.model_copy(update={"clusters": reviewed}, deep=True)


def _prompt_resp_apply_mode(*, has_completed_result: bool) -> RespApplyMode:
    if not _is_interactive_terminal():
        return RespApplyMode.APPLY_EXISTING if has_completed_result else RespApplyMode.NEW_DIRECTORY

    print_notice(
        "RESP Result Detected" if has_completed_result else "RESP Job Directory Detected",
        (
            "A matching RESP setup/result was found for this ligand.\n"
            "Choose whether to apply the existing RESP result, generate a fresh job in a new directory, "
            "or replace the existing job assets and regenerate them."
            if has_completed_result
            else "A matching RESP job directory already exists, but a completed RESP charge file was not found.\n"
            "Choose whether to create a fresh directory or replace the existing assets."
        ),
        border_style="cyan",
    )
    if has_completed_result:
        default = "a"
        prompt = "RESP handling ([a]pply existing / [n]ew directory / [r]eplace existing)"
        valid = {
            "a": RespApplyMode.APPLY_EXISTING,
            "n": RespApplyMode.NEW_DIRECTORY,
            "r": RespApplyMode.REBUILD,
        }
    else:
        default = "n"
        prompt = "RESP handling ([n]ew directory / [r]eplace existing)"
        valid = {
            "n": RespApplyMode.NEW_DIRECTORY,
            "r": RespApplyMode.REBUILD,
        }
    while True:
        choice = typer.prompt(prompt, default=default).strip().lower()
        if choice in valid:
            return valid[choice]


def _resolve_resp_context(
    *,
    config: WorkflowConfig,
    prepare_dir: Path,
    source_file: str | Path,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
    ) -> tuple[RespApplyMode, str | None, str | None]:
    fingerprint = molecule_fingerprint(
        source_file,
        residue_name=residue_name,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    search_root = prepare_dir.parent.parent.parent if prepare_dir.parent.parent.parent.exists() else prepare_dir.parent.parent
    candidates = find_resp_job_candidates(
        search_root=search_root,
        fingerprint=fingerprint,
        explicit_job_dir=config.ligands.resp_job_dir,
    )
    explicit_candidate = load_resp_job_candidate(config.ligands.resp_job_dir)
    if explicit_candidate is not None and all(item.job_dir != explicit_candidate.job_dir for item in candidates):
        candidates.append(explicit_candidate)
    candidates.sort(
        key=lambda item: (
            1 if item.ready_to_continue else 0,
            1 if item.completed else 0,
            str(item.payload.get("created_at") or ""),
        ),
        reverse=True,
    )
    preferred = next((candidate for candidate in candidates if candidate.ready_to_continue), candidates[0] if candidates else None)
    selected_candidate = explicit_candidate if explicit_candidate is not None else preferred
    apply_mode = config.ligands.resp_apply_mode
    if apply_mode == RespApplyMode.DETECT:
        prompt_target = selected_candidate if selected_candidate is not None else preferred
        if prompt_target is None:
            apply_mode = RespApplyMode.NEW_DIRECTORY
        else:
            apply_mode = _prompt_resp_apply_mode(has_completed_result=prompt_target.completed)

    selected_job_dir = str(selected_candidate.job_dir) if selected_candidate is not None else config.ligands.resp_job_dir
    readiness_target = selected_candidate if selected_candidate is not None else preferred
    if apply_mode == RespApplyMode.APPLY_EXISTING and (readiness_target is None or not readiness_target.ready_to_continue):
        print_notice(
            "RESP Reuse Unavailable",
            "The requested existing RESP job is not ready to continue. "
            "SIMPLE will preserve that directory and create a fresh RESP setup directory instead.",
            border_style="yellow",
        )
        apply_mode = RespApplyMode.NEW_DIRECTORY
    selected_group_file = config.ligands.resp_group_file
    if selected_group_file is None and selected_candidate is not None:
        group_path = selected_candidate.job_dir / "group_constraints.json"
        if group_path.exists():
            selected_group_file = str(group_path)
    return apply_mode, selected_job_dir, selected_group_file


def _prepare_ligands(
    *,
    config: WorkflowConfig,
    prepare_dir: Path,
    prepared: dict[str, Any] | None,
    dry_run: bool,
) -> list[Any]:
    artifacts = []
    ligand_sources: list[tuple[str | None, str]] = []

    if config.input.source == InputSource.SMALL_MOLECULE:
        ligand_sources.append((config.input.small_molecule_files[0], config.ligands.residue_name))
    elif prepared:
        for item in prepared.get("ligand_inputs", []):
            ligand_sources.append((item["path"], item["residue_name"]))

    if ligand_sources and config.input.source != InputSource.SMALL_MOLECULE and config.ligands.mode != LigandMode.MANUAL:
        raise ValueError(
            "Automatic Antechamber parameterization for retained non-standard residues in protein workflows is "
            "currently disabled. Please provide manual Amber-ready custom-residue files instead."
        )

    if ligand_sources and config.ligands.mode == LigandMode.MANUAL:
        bundle = validate_manual_ligand_bundle(config.ligands.manual_files)
        if not bundle.complete:
            raise ValueError(
                "Manual Amber files are required for the selected non-standard molecules. "
                f"{bundle.message}"
            )
        if config.input.source != InputSource.SMALL_MOLECULE:
            manual_dir = prepare_dir / "ligand_params" / "manual_bundle"
            return [
                parameterize_ligand(
                    source_file=None,
                    mode=config.ligands.mode,
                    charge_method=config.ligands.charge_method,
                    residue_name="CUSTOM",
                    net_charge=int(config.ligands.net_charge),
                    multiplicity=int(config.ligands.multiplicity),
                    manual_files=config.ligands.manual_files,
                    output_dir=manual_dir,
                    slurm_config=config.slurm,
                    resp_job_dir=config.ligands.resp_job_dir,
                    resp_group_file=config.ligands.resp_group_file,
                    resp_session_file=config.ligands.resp_session_file,
                    resp_apply_mode=config.ligands.resp_apply_mode,
                    allow_popup=False,
                    dry_run=dry_run,
                )
            ]
    elif ligand_sources and not dry_run:
        ligand_ff_label = _selected_ligand_forcefield_label(config)
        selected_charge_method = normalize_charge_method(config.ligands.charge_method)
        if charge_method_uses_resp(selected_charge_method) and config.ligands.resp_apply_mode == RespApplyMode.APPLY_EXISTING:
            print_notice(
                f"RESP + {ligand_ff_label} Parameter Update",
                f"The workflow will now apply the available RESP charges and prepare the Amber {ligand_ff_label} bonded "
                "parameter files needed for system setup. Depending on system size, this may take from a few "
                f"minutes to several hours. If required {ligand_ff_label} typing or bonded-parameter information is missing, "
                "the workflow will stop with a specific error message.",
                border_style="yellow",
            )
        elif charge_method_uses_resp(selected_charge_method):
            print_notice(
                "RESP Asset Generation",
                "The workflow is preparing the RESP/NWChem job assets for this ligand. Run the generated RESP "
                f"job first, then rerun the workflow to continue into the Amber {ligand_ff_label} bonded-parameter step.",
                border_style="yellow",
            )
        else:
            print_notice(
                "Automatic Ligand Parameterization",
                "The workflow will now prepare the automatic ligand parameter files. Depending on system size, "
                "this may take from a few minutes to several hours. Unless you intend to cancel the job, please "
                "do not stop the run while this calculation is in progress.",
                border_style="yellow",
            )
        glycan_like = sorted({residue_name.upper() for _, residue_name in ligand_sources if residue_name.upper() in COMMON_GLYCAN_RESIDUES})
        if glycan_like:
            print_notice(
                "Carbohydrate/Glycan Warning",
                "The following residue names look like carbohydrates/glycans: "
                f"{', '.join(glycan_like)}. In Amber, these are often better handled with GLYCAM "
                "or manual carbohydrate parameters than with automatic GAFF/Antechamber parameterization.",
                border_style="magenta",
            )

    for index, (source_file, residue_name) in enumerate(ligand_sources, start=1):
        ligand_dir = prepare_dir / "ligand_params" / f"{index:02d}_{residue_name}"
        net_charge, multiplicity = _resolve_ligand_electronic_settings(config, residue_name)
        resp_apply_mode = config.ligands.resp_apply_mode
        resp_job_dir = config.ligands.resp_job_dir
        resp_group_file = config.ligands.resp_group_file
        resp_session_file = config.ligands.resp_session_file
        if (
            charge_method_uses_resp(config.ligands.charge_method)
            and config.ligands.mode != LigandMode.MANUAL
            and source_file is not None
        ):
            resp_apply_mode, resp_job_dir, resp_group_file = _resolve_resp_context(
                config=config,
                prepare_dir=prepare_dir,
                source_file=source_file,
                residue_name=residue_name,
                net_charge=net_charge,
                multiplicity=multiplicity,
            )
        if config.ligands.mode == LigandMode.MANUAL or dry_run:
            artifacts.append(
                parameterize_ligand(
                    source_file=source_file,
                    mode=config.ligands.mode,
                    charge_method=config.ligands.charge_method,
                    residue_name=residue_name,
                    net_charge=net_charge,
                    multiplicity=multiplicity,
                    manual_files=config.ligands.manual_files,
                    output_dir=ligand_dir,
                    slurm_config=config.slurm,
                    resp_job_dir=resp_job_dir,
                    resp_group_file=resp_group_file,
                    resp_session_file=resp_session_file,
                    resp_apply_mode=resp_apply_mode,
                    allow_popup=False,
                    dry_run=dry_run,
                )
            )
            continue

        with activity_status(
            "[blink bold yellow]Calculating...[/] Antechamber is parameterizing the ligand. "
            "This may take from a few minutes to several hours depending on system size. "
            "Please do not stop the run unless you intend to cancel it.",
            plain_message=(
                "Calculating... Antechamber is parameterizing the ligand. "
                "This may take from a few minutes to several hours depending on system size. "
                "Please do not stop the run unless you intend to cancel it."
            ),
        ):
            artifacts.append(
                parameterize_ligand(
                    source_file=source_file,
                    mode=config.ligands.mode,
                    charge_method=config.ligands.charge_method,
                    residue_name=residue_name,
                    net_charge=net_charge,
                    multiplicity=multiplicity,
                    manual_files=config.ligands.manual_files,
                    output_dir=ligand_dir,
                    slurm_config=config.slurm,
                    resp_job_dir=resp_job_dir,
                    resp_group_file=resp_group_file,
                    resp_session_file=resp_session_file,
                    resp_apply_mode=resp_apply_mode,
                    allow_popup=charge_method_uses_resp(config.ligands.charge_method),
                    dry_run=dry_run,
                )
            )

    return artifacts


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_existing_ligand_artifacts(prepare_dir: Path) -> list[LigandArtifacts]:
    """Restore the ligand templates used by a paused site-RESP workflow."""

    ligand_root = prepare_dir / "ligand_params"
    if not ligand_root.exists():
        return []
    artifacts: list[LigandArtifacts] = []
    for manifest_path in sorted(ligand_root.rglob("ligand_manifest.json")):
        payload = _load_json_dict(manifest_path)
        if payload is None:
            continue
        raw_files = payload.get("files") or {}
        if not isinstance(raw_files, dict):
            continue
        files: dict[str, str] = {}
        for label, raw_path in raw_files.items():
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.exists():
                local_candidate = manifest_path.parent / candidate.name
                if local_candidate.exists():
                    candidate = local_candidate
            files[str(label)] = str(candidate.resolve())
        raw_commands = payload.get("commands") or []
        commands = [
            [str(token) for token in command]
            for command in raw_commands
            if isinstance(command, list)
        ]
        artifacts.append(
            LigandArtifacts(
                mode=str(payload.get("mode") or "existing"),
                residue_name=str(payload.get("residue_name") or "CUSTOM"),
                source_file=(
                    None
                    if payload.get("source_file") is None
                    else str(payload.get("source_file"))
                ),
                coordinate_source=(
                    None
                    if payload.get("coordinate_source") is None
                    else str(payload.get("coordinate_source"))
                ),
                files=files,
                commands=commands,
                notes=[str(note) for note in payload.get("notes") or []],
            )
        )
    return artifacts


def _resume_existing_protein_site_resp_workflow(
    *,
    config: WorkflowConfig,
    root: Path,
    stages: dict[str, Path],
    selected: list[str],
    dry_run: bool,
    amber_env: Any,
) -> dict[str, Any]:
    job_dirs = [Path(item).expanduser().resolve() for item in config.protein_site_resp.job_dirs]
    if not job_dirs:
        raise ValueError("Protein-site RESP continuation requires at least one selected RESP job directory.")
    system_dir = stages["system"]
    system_prmtop = stages["system"] / "system.prmtop"
    system_inpcrd = stages["system"] / "system.inpcrd"
    reference_paths = (
        system_dir / "site_resp_reference_manifest.json",
        system_dir / "system.unsolvated.pdb",
        system_dir / "system.unsolvated.prmtop",
        system_dir / "system.unsolvated.inpcrd",
    )
    final_system_available = system_prmtop.exists() and system_inpcrd.exists()
    if not final_system_available and not all(path.exists() for path in reference_paths):
        raise FileNotFoundError(
            "The selected RESP workflow has neither a final 02_system/system.prmtop + system.inpcrd "
            "nor the complete site-RESP reference set (site_resp_reference_manifest.json and "
            "system.unsolvated.pdb/.prmtop/.inpcrd); SIMPLE cannot safely continue it."
        )

    jobs = [{"job_dir": str(job_dir)} for job_dir in job_dirs]
    prepared = _load_prepare_manifest(stages["prepare"])
    system_payload = (
        _load_json_dict(system_dir / "system_manifest.json")
        or _load_json_dict(system_dir / "site_resp_reference_manifest.json")
        or {}
    )
    protein_site_resp_result: dict[str, object]
    md_stages = []
    slurm_path: Path | None = None

    if "system" not in selected and "md" not in selected:
        protein_site_resp_result = {
            "status": "review_required",
            "jobs": [{**job, "review": review_site_resp_result(str(job["job_dir"]))} for job in jobs],
            "message": "The selected stage range stops before topology application.",
        }
    else:
        should_apply = True
        if _is_interactive_terminal():
            should_apply = _confirm_protein_site_resp_application(jobs)
        if not should_apply:
            protein_site_resp_result = {
                "status": "review_required",
                "jobs": [{**job, "review": review_site_resp_result(str(job["job_dir"]))} for job in jobs],
                "message": "The matching charges were reviewed but not approved; the topology was not changed.",
            }
        else:
            if not final_system_available:
                if prepared is None:
                    raise FileNotFoundError(
                        "Protein-site RESP continuation requires 01_prepare/prepare_manifest.json "
                        "to build the deferred final solvated system."
                    )
                prepared_pdb_value = prepared.get("cleaned_pdb")
                prepared_pdb = (
                    Path(str(prepared_pdb_value)).expanduser().resolve()
                    if prepared_pdb_value
                    else stages["prepare"] / "cleaned_input.pdb"
                )
                if not prepared_pdb.exists():
                    raise FileNotFoundError(
                        "Protein-site RESP continuation could not find the prepared protein PDB needed "
                        f"to build the final solvated system: {prepared_pdb}"
                    )
                source_files = [
                    Path(str(item["path"])).expanduser().resolve()
                    for item in prepared.get("ligand_inputs") or []
                    if isinstance(item, dict) and item.get("path")
                ]
                ligand_artifacts = _load_existing_ligand_artifacts(stages["prepare"])
                if source_files and not ligand_artifacts:
                    raise FileNotFoundError(
                        "The paused protein-site RESP workflow contains retained non-standard residues, "
                        "but their 01_prepare/ligand_params/ligand_manifest.json files are unavailable."
                    )
                final_system_config = _system_config_with_inserted_metal_charges(config.system, prepared)
                final_system_result = build_system_with_tleap(
                    system_config=final_system_config,
                    amber_env=amber_env,
                    prepared_pdb=prepared_pdb,
                    ligand_artifacts=ligand_artifacts,
                    source_files=source_files,
                    output_dir=system_dir,
                    dry_run=dry_run,
                )
                system_payload = final_system_result.to_dict()
                final_system_available = system_prmtop.exists() and system_inpcrd.exists()
                if not final_system_available:
                    raise FileNotFoundError(
                        "The deferred final system build did not create 02_system/system.prmtop and "
                        "system.inpcrd, so RESP charges cannot yet be applied."
                    )
            parmed_status = amber_env.binaries.get("parmed")
            protein_site_resp_result = apply_site_resp_results(
                job_dirs=job_dirs,
                prmtop_path=system_prmtop,
                inpcrd_path=system_inpcrd,
                parmed_binary=parmed_status.path if parmed_status is not None else None,
                output_dir=stages["system"],
                dry_run=dry_run,
            )

            if "md" in selected:
                prepared_pdb_for_mask = None
                if prepared and prepared.get("cleaned_pdb"):
                    prepared_pdb_for_mask = Path(str(prepared["cleaned_pdb"]))
                final_system_pdb = stages["system"] / "system.pdb"
                _translate_focused_restraint_mask_to_topology(
                    config,
                    prepared_pdb_for_mask,
                    final_system_pdb,
                )
                md_input_dir = stages["md"] / "inputs"
                md_stages = generate_md_inputs(config.md, md_input_dir)
                slurm_path = write_slurm_script(
                    stages=md_stages,
                    slurm_config=config.slurm,
                    output_dir=stages["md"],
                )

    result = {
        "output_dir": str(root),
        "prepare": str(stages["prepare"] / "prepare_manifest.json") if prepared else None,
        "system": system_payload,
        "md_inputs": [str(stages["md"] / "inputs" / stage.filename) for stage in md_stages],
        "slurm": str(slurm_path) if slurm_path else None,
        "dry_run": dry_run,
        "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
        "protein_site_resp": protein_site_resp_result,
    }
    write_json(root / "workflow_manifest.json", result)
    return result


def run_workflow(
    *,
    config: WorkflowConfig,
    from_stage: str = "prepare",
    to_stage: str = "md",
    dry_run: bool = False,
) -> dict[str, Any]:
    root = config.output_path()
    root.mkdir(parents=True, exist_ok=True)
    stages = _stage_directories(root)
    amber_env = detect_amber_environment()
    prepared: dict[str, Any] | None = None
    ligand_artifacts: list[Any] = []
    system_result: Any = None
    protein_site_resp_result: dict[str, object] | None = None
    md_stages = []
    slurm_path: Path | None = None

    ordered = ["prepare", "system", "md"]
    start_index = ordered.index(from_stage)
    end_index = ordered.index(to_stage)
    selected = ordered[start_index : end_index + 1]

    snapshot_name = (
        "workflow_resume_config.toml"
        if config.protein_site_resp.resume_existing_system
        else "workflow_config.toml"
    )
    save_config(config, root / snapshot_name)

    if config.protein_site_resp.resume_existing_system:
        return _resume_existing_protein_site_resp_workflow(
            config=config,
            root=root,
            stages=stages,
            selected=selected,
            dry_run=dry_run,
            amber_env=amber_env,
        )

    if "prepare" in selected:
        if config.input.source == InputSource.DES:
            stages["prepare"].mkdir(parents=True, exist_ok=True)
            prepared = {
                "raw_input": [],
                "cleaned_pdb": None,
                "summary": None,
                "ligand_inputs": [],
                "des": config.des.model_dump(mode="json"),
            }
            write_json(stages["prepare"] / "prepare_manifest.json", prepared)
        elif config.input.source == InputSource.SMALL_MOLECULE:
            stages["prepare"].mkdir(parents=True, exist_ok=True)
            prepared = {
                "raw_input": [str(Path(path).expanduser().resolve()) for path in config.input.small_molecule_files],
                "cleaned_pdb": None,
                "summary": None,
                "ligand_inputs": [
                    {
                        "path": str(Path(config.input.small_molecule_files[0]).expanduser().resolve()),
                        "residue_name": config.ligands.residue_name,
                    }
                ],
            }
            write_json(stages["prepare"] / "prepare_manifest.json", prepared)
        else:
            source_value = config.input.path or config.input.pdb_id or ""
            with activity_status(
                "[blink bold cyan]Processing...[/] Preparing and cleaning the input protein structure. "
                "This can take several seconds while the structure is downloaded, inspected, and rewritten.",
                plain_message=(
                    "Processing... Preparing and cleaning the input protein structure. "
                    "This can take several seconds while the structure is downloaded, inspected, and rewritten."
                ),
            ):
                prepared = prepare_structure(
                    source=config.input.source,
                    source_value=source_value,
                    prepare_config=config.prepare,
                    protonation_config=config.protonation,
                    kept_ligands=config.prepare.kept_ligands,
                    output_dir=stages["prepare"],
                )
            prepare_warnings = prepared.get("warnings") if prepared else None
            if prepare_warnings:
                print_notice(
                    "Prepare Warning",
                    "\n".join(f"- {warning}" for warning in prepare_warnings),
                    border_style="yellow",
                )
        ligand_artifacts = _prepare_ligands(
            config=config,
            prepare_dir=stages["prepare"],
            prepared=prepared,
            dry_run=dry_run,
        )
        pending_resp = next((artifact for artifact in ligand_artifacts if getattr(artifact, "mode", "") == "resp_setup_pending"), None)
        if pending_resp is not None:
            print_notice(
                "RESP Setup Complete",
                "NWChem input, RESP fitting helper, and sbatch assets were generated.\n"
                "Run the generated job, wait for output/resp_charges.json, then rerun this workflow to apply the RESP charges.",
                border_style="green",
            )
            result = {
                "output_dir": str(root),
                "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
                "system": None,
                "md_inputs": [],
                "slurm": None,
                "dry_run": dry_run,
                "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
                "resp": pending_resp.to_dict(),
            }
            write_json(root / "workflow_manifest.json", result)
            return result

    if "system" in selected:
        if prepared is None:
            prepared = _load_prepare_manifest(stages["prepare"])
            if prepared is None:
                prepared_pdb = stages["prepare"] / "cleaned_input.pdb"
                prepared = {"cleaned_pdb": str(prepared_pdb) if prepared_pdb.exists() else None, "ligand_inputs": []}
        site_resp_enabled = (
            config.input.source not in {InputSource.SMALL_MOLECULE, InputSource.DES}
            and config.protein_site_resp.mode == ProteinSiteRespMode.RESP
        )
        source_files: list[Path] = []
        prepared_pdb: Path | None = None
        site_system_config: SystemConfig | None = None
        if config.input.source == InputSource.DES:
            system_result = build_des_system(
                des_config=config.des,
                amber_env=amber_env,
                output_dir=stages["system"],
                dry_run=dry_run,
                system_config=config.system,
            )
        else:
            source_files = [Path(item["path"]) for item in prepared.get("ligand_inputs", [])] if prepared else []
            prepared_pdb = Path(prepared["cleaned_pdb"]) if prepared and prepared.get("cleaned_pdb") else None
            site_system_config = _system_config_with_inserted_metal_charges(config.system, prepared)
            if site_resp_enabled:
                if prepared_pdb is None:
                    raise ValueError("Protein-site RESP requires a prepared protein PDB reference.")
                system_result = build_protein_site_resp_reference_with_tleap(
                    system_config=site_system_config,
                    amber_env=amber_env,
                    prepared_pdb=prepared_pdb,
                    ligand_artifacts=ligand_artifacts,
                    source_files=source_files,
                    output_dir=stages["system"],
                    dry_run=dry_run,
                )
            else:
                system_result = build_system_with_tleap(
                    system_config=site_system_config,
                    amber_env=amber_env,
                    prepared_pdb=prepared_pdb,
                    ligand_artifacts=ligand_artifacts,
                    source_files=source_files,
                    output_dir=stages["system"],
                    dry_run=dry_run,
                )

        if site_resp_enabled:
            assert site_system_config is not None
            assert prepared_pdb is not None
            system_pdb = stages["system"] / "system.pdb"
            system_prmtop = stages["system"] / "system.prmtop"
            system_inpcrd = stages["system"] / "system.inpcrd"
            reference_pdb = stages["system"] / "system.unsolvated.pdb"
            reference_prmtop = stages["system"] / "system.unsolvated.prmtop"
            if dry_run and not (reference_pdb.exists() and reference_prmtop.exists()):
                protein_site_resp_result = {
                    "status": "reference_pending",
                    "message": (
                        "Protein-site RESP needs an executed TLeap reference topology to obtain hydrogens, "
                        "baseline ff19SB charges, and an exact atom mapping. Rerun without --dry-run on the Amber host."
                    ),
                }
                result = {
                    "output_dir": str(root),
                    "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
                    "system": system_result.to_dict() if system_result else None,
                    "md_inputs": [],
                    "slurm": None,
                    "dry_run": dry_run,
                    "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
                    "protein_site_resp": protein_site_resp_result,
                }
                write_json(root / "workflow_manifest.json", result)
                return result
            site_resp_config = config.protein_site_resp
            raw_source = (prepared or {}).get("raw_input")
            if raw_source and Path(raw_source).exists():
                validate_retained_direct_environment(
                    source_pdb=raw_source,
                    reference_pdb=reference_pdb,
                )
            if site_resp_config.review_clusters and not site_resp_config.clusters:
                reference_atoms = load_topology_atoms(reference_prmtop, reference_pdb)
                proposed_clusters = discover_site_clusters(
                    atoms=reference_atoms,
                    system_config=site_system_config,
                    resp_config=site_resp_config,
                )
                atom_by_index = {atom.topology_index: atom for atom in reference_atoms}
                charge_by_site = {int(item.site): int(item.charge) for item in site_system_config.metal_charges}
                cluster_rows: list[dict[str, object]] = []
                for cluster in proposed_clusters:
                    species = [
                        {
                            "site": site,
                            "element": atom_by_index[topology_index].element,
                            "formal_charge": charge_by_site[site],
                        }
                        for site, topology_index in zip(
                            cluster.metal_sites,
                            cluster.metal_atom_indices,
                            strict=True,
                        )
                    ]
                    suggestion = (
                        suggested_spin_multiplicity(
                            str(species[0]["element"]),
                            int(species[0]["formal_charge"]),
                        )
                        if len(species) == 1
                        else None
                    )
                    cluster_rows.append(
                        {
                            **cluster.to_dict(),
                            "species": species,
                            "suggested_multiplicity": suggestion,
                        }
                    )
                protein_site_resp_result = {
                    "status": "cluster_review_required",
                    "clusters": cluster_rows,
                    "message": (
                        "Review/edit the detected donor residues, fixed QM environment, and spin multiplicity "
                        "for every connected metal cluster before generating NWChem jobs."
                    ),
                }
                result = {
                    "output_dir": str(root),
                    "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
                    "system": system_result.to_dict() if system_result else None,
                    "md_inputs": [],
                    "slurm": None,
                    "dry_run": dry_run,
                    "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
                    "protein_site_resp": protein_site_resp_result,
                }
                write_json(root / "workflow_manifest.json", result)
                return result
            if _is_interactive_terminal():
                site_resp_config = _review_protein_site_resp_clusters(
                    resp_config=site_resp_config,
                    system_config=site_system_config,
                    system_pdb=reference_pdb,
                    system_prmtop=reference_prmtop,
                )
            site_jobs = build_site_resp_jobs(
                system_pdb=reference_pdb,
                system_prmtop=reference_prmtop,
                system_config=site_system_config,
                resp_config=site_resp_config,
                slurm_config=config.slurm,
                base_dir=stages["prepare"] / "protein_site_resp_jobs",
                source_label=_source_identity_label(config),
                source_pdb=(prepared or {}).get("raw_input"),
            )
            pending_jobs = [job for job in site_jobs if job.get("status") == "setup_pending"]
            if pending_jobs:
                protein_site_resp_result = {
                    "status": "setup_pending",
                    "jobs": site_jobs,
                    "message": (
                        "Run each generated Tahoma CPU RESP job, then rerun SIMPLE. "
                        "The matching results will be reviewed before topology application."
                    ),
                }
                print_notice(
                    "Protein-Site RESP Setup Complete",
                    protein_site_resp_result["message"],
                    border_style="green",
                )
                result = {
                    "output_dir": str(root),
                    "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
                    "system": system_result.to_dict() if system_result else None,
                    "md_inputs": [],
                    "slurm": None,
                    "dry_run": dry_run,
                    "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
                    "protein_site_resp": protein_site_resp_result,
                }
                write_json(root / "workflow_manifest.json", result)
                return result

            should_apply = site_resp_config.apply_mode == RespApplyMode.APPLY_EXISTING
            if site_resp_config.apply_mode == RespApplyMode.DETECT and _is_interactive_terminal():
                should_apply = _confirm_protein_site_resp_application(site_jobs)
            if not should_apply:
                protein_site_resp_result = {
                    "status": "review_required",
                    "jobs": [
                        {**job, "review": review_site_resp_result(str(job["job_dir"]))}
                        for job in site_jobs
                    ],
                    "message": "Review and approve the matching site-specific charges before application.",
                }
                result = {
                    "output_dir": str(root),
                    "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
                    "system": system_result.to_dict() if system_result else None,
                    "md_inputs": [],
                    "slurm": None,
                    "dry_run": dry_run,
                    "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
                    "protein_site_resp": protein_site_resp_result,
                }
                write_json(root / "workflow_manifest.json", result)
                return result
            system_result = build_system_with_tleap(
                system_config=site_system_config,
                amber_env=amber_env,
                prepared_pdb=prepared_pdb,
                ligand_artifacts=ligand_artifacts,
                source_files=source_files,
                output_dir=stages["system"],
                dry_run=dry_run,
            )
            parmed_status = amber_env.binaries.get("parmed")
            protein_site_resp_result = apply_site_resp_results(
                job_dirs=[str(job["job_dir"]) for job in site_jobs],
                prmtop_path=system_prmtop,
                inpcrd_path=system_inpcrd,
                parmed_binary=parmed_status.path if parmed_status is not None else None,
                output_dir=stages["system"],
                dry_run=dry_run,
            )

    if "md" in selected:
        prepared_pdb_for_mask = None
        if prepared and prepared.get("cleaned_pdb"):
            prepared_pdb_for_mask = Path(prepared["cleaned_pdb"])
        else:
            candidate_prepared = stages["prepare"] / "cleaned_input.pdb"
            if candidate_prepared.exists():
                prepared_pdb_for_mask = candidate_prepared

        final_system_pdb = stages["system"] / "system.pdb"
        _translate_focused_restraint_mask_to_topology(config, prepared_pdb_for_mask, final_system_pdb)

        md_input_dir = stages["md"] / "inputs"
        md_stages = generate_md_inputs(
            config.md,
            md_input_dir,
            small_molecule_only=config.input.source == InputSource.SMALL_MOLECULE,
            des_solvent=config.input.source == InputSource.DES,
        )
        slurm_path = write_slurm_script(
            stages=md_stages,
            slurm_config=config.slurm,
            output_dir=stages["md"],
        )

    result = {
        "output_dir": str(root),
        "prepare": str(prepared.get("manifest")) if prepared and prepared.get("manifest") else None,
        "system": system_result.to_dict() if system_result else None,
        "md_inputs": [str((stages["md"] / "inputs" / stage.filename)) for stage in md_stages],
        "slurm": str(slurm_path) if slurm_path else None,
        "dry_run": dry_run,
        "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
        "protein_site_resp": protein_site_resp_result,
    }
    write_json(root / "workflow_manifest.json", result)
    return result
