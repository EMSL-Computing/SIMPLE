from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import typer

from amber_metallo.amber.leap import build_system_with_tleap
from amber_metallo.config import (
    InputSource,
    LigandMode,
    MetalChargeAssignment,
    RespApplyMode,
    ResidueMaskNumbering,
    SystemConfig,
    WorkflowConfig,
    charge_method_uses_resp,
    normalize_charge_method,
)
from amber_metallo.des import build_des_system
from amber_metallo.environment import detect_amber_environment
from amber_metallo.inspection import load_structure
from amber_metallo.ligand_param import parameterize_ligand, validate_manual_ligand_bundle
from amber_metallo.md_protocols import generate_md_inputs
from amber_metallo.prep import prepare_structure
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
    md_stages = []
    slurm_path: Path | None = None

    ordered = ["prepare", "system", "md"]
    start_index = ordered.index(from_stage)
    end_index = ordered.index(to_stage)
    selected = ordered[start_index : end_index + 1]

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
            system_result = build_system_with_tleap(
                system_config=_system_config_with_inserted_metal_charges(config.system, prepared),
                amber_env=amber_env,
                prepared_pdb=prepared_pdb,
                ligand_artifacts=ligand_artifacts,
                source_files=source_files,
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
    }
    write_json(root / "workflow_manifest.json", result)
    return result
