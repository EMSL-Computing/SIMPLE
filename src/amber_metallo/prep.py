from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from amber_metallo.config import InputSource, PrepareConfig, ProtonationConfig
import gemmi

from amber_metallo.inspection import (
    SUPPORTED_METALS,
    classify_residue,
    fetch_pdb_structure,
    inspect_structure,
    load_structure,
    residue_key,
)
from amber_metallo.metal_insert import append_metal_residue, resolve_metal_insertion
from amber_metallo.missing_loops import repair_internal_missing_loops, write_missing_loop_report
from amber_metallo.protonation import apply_protonation_changes_to_structure
from amber_metallo.reporting import write_json


ION_LABELS = {
    "Co": "CO",
    "Cu": "CU",
    "Ni": "NI",
    "Mn": "MN",
    "Fe": "FE",
    "Y": "Y",
    "La": "LA",
    "Nd": "Nd",
    "Eu": "EU3",
    "Lu": "LU",
}
STANDARD_AMINO_ACID_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
STANDARD_NUCLEIC_RESIDUES = {
    "A",
    "C",
    "G",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
    "RA",
    "RC",
    "RG",
    "RU",
}
HISTIDINE_ALIASES = {
    "HSD": "HID",
    "HSE": "HIE",
    "HSP": "HIP",
    "HSC": "HIE",
}
TERMINAL_RESIDUE_ALIASES = {
    **{f"N{name}": name for name in STANDARD_AMINO_ACID_RESIDUES if len(name) == 3},
    **{f"C{name}": name for name in STANDARD_AMINO_ACID_RESIDUES if len(name) == 3},
}
AMBER_RESIDUE_ALIASES = {
    **HISTIDINE_ALIASES,
    **TERMINAL_RESIDUE_ALIASES,
}


def _remove_extra_models(structure: Any) -> None:
    while len(structure) > 1:
        del structure[1]


def _normalize_ion_label(element: str) -> str:
    label = ION_LABELS[element.title()]
    return label[:4]


def _normalize_standard_residues_for_amber(structure: gemmi.Structure) -> list[str]:
    changes: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                original_name = residue.name.strip().upper()
                normalized_name = AMBER_RESIDUE_ALIASES.get(original_name, original_name)
                if normalized_name != original_name:
                    changes.append(f"{chain.name or '_'}:{residue.seqid} {original_name}->{normalized_name}")
                    residue.name = normalized_name
                if normalized_name in STANDARD_AMINO_ACID_RESIDUES or normalized_name in STANDARD_NUCLEIC_RESIDUES:
                    residue.het_flag = "A"
    return changes


def extract_ligand_inputs(cleaned_pdb: Path, kept_ligands: list[str], output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = load_structure(cleaned_pdb)
    _remove_extra_models(structure)
    model = structure[0]
    requested = {token.strip() for token in kept_ligands}
    extracted: list[dict[str, str]] = []
    seen_residues: set[str] = set()

    for chain in model:
        for residue in chain:
            key = residue_key(chain.name, residue)
            residue_name = residue.name.strip()
            if residue_name not in requested and key not in requested:
                continue
            if residue_name in seen_residues:
                continue

            ligand_structure = structure.clone()
            ligand_model = ligand_structure[0]
            for chain_index in reversed(range(len(ligand_model))):
                ligand_chain = ligand_model[chain_index]
                if ligand_chain.name != chain.name:
                    del ligand_model[chain_index]
                    continue
                for residue_index in reversed(range(len(ligand_chain))):
                    current = ligand_chain[residue_index]
                    if residue_key(ligand_chain.name, current) != key:
                        del ligand_chain[residue_index]
            ligand_structure.remove_empty_chains()

            target = output_dir / f"{residue_name}.pdb"
            ligand_structure.write_pdb(str(target))
            extracted.append({"residue_name": residue_name, "path": str(target), "key": key})
            seen_residues.add(residue_name)

    return extracted


def _apply_prepare_operations(
    structure: gemmi.Structure,
    *,
    prepare_config: PrepareConfig,
    kept_ligands: list[str],
) -> list[Any]:
    _remove_extra_models(structure)
    model = structure[0]

    replacement_map = {item.site: item.target.title() for item in prepare_config.metal_replacements}
    deletion_sites = set(prepare_config.metal_deletions)
    supported = {metal.title() for metal in SUPPORTED_METALS}
    invalid_targets = set(replacement_map.values()) - supported
    if invalid_targets:
        raise ValueError(f"Unsupported metal replacement targets: {sorted(invalid_targets)}")

    metal_index = 0
    kept_tokens = {token.strip() for token in kept_ligands}
    for chain in model:
        delete_indices: list[int] = []
        for residue_index, residue in enumerate(chain):
            key = residue_key(chain.name, residue)
            classification = classify_residue(residue)

            if classification == "water" and prepare_config.remove_waters:
                delete_indices.append(residue_index)
                continue

            if classification == "hetero":
                keep = residue.name.strip() in kept_tokens or key in kept_tokens
                if prepare_config.remove_other_hetero and not keep:
                    delete_indices.append(residue_index)
                    continue

            if classification == "metal":
                metal_index += 1
                if prepare_config.remove_metals or metal_index in deletion_sites:
                    delete_indices.append(residue_index)
                    continue
                if metal_index in replacement_map:
                    target = replacement_map[metal_index]
                    atom = residue[0]
                    atom.element = gemmi.Element(target.upper())
                    label = _normalize_ion_label(target)
                    atom.name = label
                    residue.name = label

        for residue_index in reversed(delete_indices):
            del chain[residue_index]

    structure.remove_empty_chains()

    inserted_metals = []
    for insertion in prepare_config.metal_insertions:
        resolved = resolve_metal_insertion(structure, insertion)
        label = _normalize_ion_label(resolved.element)
        inserted_metals.append(
            append_metal_residue(
                structure,
                resolved,
                residue_name=label,
                atom_name=label,
            )
        )

    structure.remove_empty_chains()
    return inserted_metals


def prepare_structure(
    *,
    source: InputSource,
    source_value: str,
    prepare_config: PrepareConfig,
    protonation_config: ProtonationConfig | None,
    kept_ligands: list[str],
    output_dir: Path,
    apply_loop_repair: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if source == InputSource.PDB_ID:
        raw_path = fetch_pdb_structure(source_value, output_dir / "downloads")
    else:
        raw_path = Path(source_value).expanduser().resolve()

    source_summary = inspect_structure(raw_path, source_label=source.value)
    kept_tokens = {token.strip() for token in kept_ligands}
    missing_loop_summary = source_summary.missing_loops
    missing_loop_report: Path | None = None
    missing_loop_warnings: list[str] = []
    repaired_source_path: Path | None = None

    if missing_loop_summary is not None and missing_loop_summary.has_missing_blocks:
        missing_loop_report = write_missing_loop_report(output_dir / "missing_loops.tsv", missing_loop_summary)

    if apply_loop_repair and prepare_config.repair_missing_loops:
        if missing_loop_summary is None or missing_loop_summary.detection_status != "available":
            message = (
                None
                if missing_loop_summary is None
                else missing_loop_summary.detection_message
            )
            raise RuntimeError(
                message
                or "Missing-loop repair was requested, but missing-loop detection is unavailable for this structure."
            )
        if missing_loop_summary.has_internal_blocks:
            repaired_source_path = repair_internal_missing_loops(
                raw_path,
                output_dir / "repaired_input_loops.pdb",
            )
            missing_loop_warnings.append(
                "PDBFixer rebuilt one or more internal missing loop blocks. This is a rough repair; inspect the "
                "rebuilt region carefully before simulation."
            )

    structure = load_structure(repaired_source_path or raw_path)
    residue_normalization_warnings = _normalize_standard_residues_for_amber(structure)
    inserted_metals = _apply_prepare_operations(
        structure,
        prepare_config=prepare_config,
        kept_ligands=list(kept_tokens),
    )
    cleaned_path = output_dir / "cleaned_input.pdb"
    applied_protonation_changes = []
    if protonation_config and protonation_config.enabled and protonation_config.selected_changes:
        applied_protonation_changes = apply_protonation_changes_to_structure(
            structure,
            protonation_config.selected_changes,
        )
    structure.write_pdb(str(cleaned_path))
    cleaned_summary = inspect_structure(cleaned_path, source_label=source.value, detect_missing_loops=False)
    metal_by_key = {site.key: site.site for site in cleaned_summary.metals}
    inserted_metal_payloads = []
    insertion_warnings: list[str] = []
    for item in inserted_metals:
        key = f"{item.chain}:{item.residue_name}:{item.seqid}"
        item.site = metal_by_key.get(key)
        payload = item.to_dict()
        payload["key"] = key
        inserted_metal_payloads.append(payload)
        insertion_warnings.extend(str(warning) for warning in item.warnings)
    residue_normalization_messages = [
        "Normalized Amber residue aliases before tleap: " + ", ".join(residue_normalization_warnings[:12])
        + (" ..." if len(residue_normalization_warnings) > 12 else "")
    ] if residue_normalization_warnings else []
    all_warnings = [*missing_loop_warnings, *residue_normalization_messages, *insertion_warnings]
    extracted_ligands = extract_ligand_inputs(cleaned_path, list(kept_tokens), output_dir / "ligands")
    manifest_path = write_json(
        output_dir / "prepare_manifest.json",
        {
            "raw_input": str(raw_path),
            "repaired_input": None if repaired_source_path is None else str(repaired_source_path),
            "cleaned_pdb": str(cleaned_path),
            "source_summary": source_summary.to_dict(),
            "summary": cleaned_summary.to_dict(),
            "missing_loops": {
                **(
                    {}
                    if missing_loop_summary is None
                    else missing_loop_summary.to_dict()
                ),
                "report": None if missing_loop_report is None else str(missing_loop_report),
                "repair_requested": bool(prepare_config.repair_missing_loops),
                "repair_applied": repaired_source_path is not None,
                "repaired_pdb": None if repaired_source_path is None else str(repaired_source_path),
                "warnings": all_warnings,
            },
            "kept_ligands": sorted(kept_tokens),
            "metal_sites": [asdict(item) for item in source_summary.metals],
            "requested_replacements": {item.site: item.target.title() for item in prepare_config.metal_replacements},
            "requested_deletions": sorted(set(prepare_config.metal_deletions)),
            "requested_insertions": [
                item.model_dump(mode="json") for item in prepare_config.metal_insertions
            ],
            "inserted_metal_sites": inserted_metal_payloads,
            "protonation": {
                "enabled": bool(protonation_config and protonation_config.enabled),
                "ph": None if protonation_config is None else protonation_config.ph,
                "engine": None if protonation_config is None else protonation_config.engine.value,
                "selected_changes": [
                    change.model_dump(mode="json")
                    for change in (protonation_config.selected_changes if protonation_config else [])
                ],
                "applied_changes": [change.model_dump(mode="json") for change in applied_protonation_changes],
            },
            "ligand_inputs": extracted_ligands,
        },
    )
    return {
        "raw_input": raw_path,
        "repaired_input": repaired_source_path,
        "cleaned_pdb": cleaned_path,
        "summary": cleaned_summary,
        "source_summary": source_summary,
        "ligand_inputs": extracted_ligands,
        "missing_loops": missing_loop_summary,
        "missing_loop_report": missing_loop_report,
        "warnings": all_warnings,
        "inserted_metal_sites": inserted_metal_payloads,
        "manifest": manifest_path,
    }
