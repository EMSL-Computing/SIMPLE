"""Validated, lambda-independent metal/donor restraints for single-topology GTI.

This module deliberately does not integrate TI data or estimate restraint release
free energies. Multiple donor restraints do not have the legacy radial correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from amber_metallo.inspection import load_structure
from amber_metallo.reporting import write_json
from amber_metallo.ti.analysis import (
    MetalSiteCandidate,
    SUPPORTED_DONOR_ELEMENTS,
    _fixed_width_tokens_from_section,
    _iter_indexed_atoms,
    _parse_prmtop_sections,
    _tokens_from_section,
    build_cluster_atom_mask,
    detect_bound_metal_sites,
    run_cluster_representative_selection,
    run_last_snapshot_extraction,
)
from amber_metallo.ti.config import (
    CoordinationRestraintConfig,
    PreparedRestraintSnapshot,
    RestraintReference,
    SnapshotMode,
    TIWorkflowConfig,
)
from amber_metallo.ti.snapshots import run_time_snapshot_extraction
from amber_metallo.ti.atom_mapping import map_to_topology, map_between_structures, remap_candidates, _residues
from amber_metallo.ti.full_structure import normalize_full_reference


CORRECTION_NOTICE = (
    "Coordination restraints stay active through lambda=1. Their attachment/release "
    "free-energy corrections have NOT been computed. Do not subtract the mean restraint "
    "energy or use the legacy single-distance standard-state correction. Corrected binding "
    "free energies remain unavailable (NaN); the restrained TI window integrals are unchanged. "
    "Different donors, reference distances or force constants do not automatically cancel "
    "between metal ions. This is a restrained calculation, not a corrected binding affinity."
)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def snapshot_input_signature(config: TIWorkflowConfig, *, dry_run: bool) -> str:
    inputs = {}
    for key in ("prmtop_path", "trajectory_path", "reference_structure_path", "production_mdin_path"):
        raw = getattr(config.complex_input, key)
        if raw is not None:
            path = Path(raw).expanduser().resolve()
            stat = path.stat()
            inputs[key] = [str(path), stat.st_size, stat.st_mtime_ns]
    payload = {
        "inputs": inputs,
        "snapshot": config.snapshot.model_dump(mode="json"),
        "sites": config.metal.selected_sites or [config.metal.selected_site],
        "dry_run": dry_run,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_restraint_snapshot(
    config: TIWorkflowConfig, candidates: list[MetalSiteCandidate], *, output_dir: Path, dry_run: bool,
) -> PreparedRestraintSnapshot:
    """Save the exact frame previewed by the wizard, to reuse during generation."""
    inputs = config.complex_input
    reference_path = normalize_full_reference(
        reference_pdb=inputs.reference_structure_path, prmtop_path=inputs.prmtop_path,
        output_dir=output_dir / "full_reference")
    candidates = remap_candidates(candidates, map_to_topology(inputs.reference_structure_path, inputs.prmtop_path))
    common = dict(
        prmtop_path=str(Path(inputs.prmtop_path).expanduser().resolve()),
        trajectory_path=str(Path(inputs.trajectory_path).expanduser().resolve()),
        reference_structure_path=str(reference_path),
        output_dir=output_dir.resolve(), dry_run=dry_run,
    )
    if config.snapshot.mode == SnapshotMode.TIME:
        result = run_time_snapshot_extraction(
            **common, production_mdin_path=inputs.production_mdin_path, time_ns=config.snapshot.time_ns,
        )
        pdb, restart = result["snapshot_pdb"], result["snapshot_rst7"]
    elif config.snapshot.mode == SnapshotMode.CLUSTER:
        indices = set()
        mapping = map_to_topology(reference_path, inputs.prmtop_path)
        for candidate in candidates:
            mask = build_cluster_atom_mask(
                reference_path, candidate,
                radius_angstrom=config.snapshot.cluster_radius_angstrom,
            )
            indices.update(mapping[int(value)] for value in mask.lstrip("@").split(",")
                           if value and int(value) in mapping)
        remap_candidates(candidates, mapping)  # Fail rather than cluster around a guessed metal.
        result = run_cluster_representative_selection(
            **common, atom_mask="@" + ",".join(map(str, sorted(indices))),
            epsilon_angstrom=config.snapshot.cluster_epsilon_angstrom, sieve=config.snapshot.cluster_sieve,
        )
        pdb, restart = result["representative_snapshot_pdb"], result["representative_snapshot_rst7"]
    else:
        result = run_last_snapshot_extraction(**common)
        pdb, restart = result["last_snapshot_pdb"], result["last_snapshot_rst7"]
    validate_restraint_atom_order(pdb, inputs.prmtop_path)
    result["full_reference_pdb"] = str(reference_path)
    return PreparedRestraintSnapshot(
        pdb_path=str(Path(pdb).resolve()), restart_path=str(Path(restart).resolve()),
        input_signature=snapshot_input_signature(config, dry_run=dry_run),
        pdb_sha256=file_sha256(pdb), restart_sha256=file_sha256(restart), dry_run=dry_run,
        metadata=result,
    )


def checked_prepared_snapshot(config: TIWorkflowConfig, *, dry_run: bool) -> PreparedRestraintSnapshot | None:
    restraint = config.ti.coordination_restraint
    prepared = restraint.prepared_snapshot if restraint is not None and restraint.enabled else None
    if prepared is None:
        return None
    # A dry-run preview is the reference PDB, not a measured trajectory frame.
    # Never reuse it in a real calculation; real extraction will run normally.
    if prepared.dry_run and not dry_run:
        return None
    if prepared.input_signature != snapshot_input_signature(config, dry_run=dry_run):
        raise ValueError("Restraint preview inputs/snapshot selection changed. Rerun the restraint wizard.")
    for path, expected in ((prepared.pdb_path, prepared.pdb_sha256), (prepared.restart_path, prepared.restart_sha256)):
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Restraint preview is missing or changed: {path}. Rerun the restraint wizard.")
    return prepared


def validate_restraint_atom_order(
    pdb_path: str | Path, prmtop_path: str | Path, *, atom_indices: list[int] | None = None,
) -> None:
    """Never guess indices from PDB serials, residue numbers or a stripped PDB."""
    atoms = _iter_indexed_atoms(pdb_path)
    sections = _parse_prmtop_sections(prmtop_path)
    names = _fixed_width_tokens_from_section(sections, "ATOM_NAME")
    labels = _fixed_width_tokens_from_section(sections, "RESIDUE_LABEL")
    pointers = [int(item) for item in _tokens_from_section(sections, "RESIDUE_POINTER")]
    if not names or not labels or not pointers or len(labels) != len(pointers):
        raise ValueError("Cannot validate restraint indices: topology lacks atom/residue identity records.")
    if atom_indices is None and len(atoms) != len(names):
        raise ValueError(
            f"Restraint PDB/topology atom count mismatch ({len(atoms)} vs {len(names)}). "
            "Use a full PDB in topology atom order, including hydrogens, solvent and extra points (EP)."
        )
    wanted = set(atom_indices or range(1, len(atoms) + 1))
    if any(index > min(len(atoms), len(names)) or index < 1 for index in wanted):
        raise ValueError("Restraint atom index is outside the PDB/topology.")
    residue_index = 0
    pdb_residue_index = -1
    previous_residue = None
    for atom in atoms:
        # Residue sequence labels may be renumbered by cpptraj; order may not.
        if atom.residue_key != previous_residue:
            pdb_residue_index += 1
            previous_residue = atom.residue_key
        while residue_index + 1 < len(pointers) and atom.atom_index >= pointers[residue_index + 1]:
            residue_index += 1
        if atom.atom_index not in wanted:
            continue
        if (
            names[atom.atom_index - 1].upper() != atom.atom_name.upper()
            or labels[residue_index][:3].upper() != atom.residue_name[:3].upper()
            or residue_index != pdb_residue_index
        ):
            raise ValueError(
                f"Restraint atom-order mismatch at topology atom {atom.atom_index} ({atom.atom_name}). "
                "Regenerate the full reference PDB from this topology; no automatic remapping was attempted."
            )


@dataclass
class CoordinationGeometry:
    source_path: str
    sites: dict[int, dict]


def candidates_in_frame(reference_pdb, frame_pdb, candidates):
    """Resolve metal identities for display without requiring a topology."""
    from dataclasses import replace
    try:
        return remap_candidates(candidates, map_between_structures(reference_pdb, frame_pdb))
    except ValueError:
        detected = detect_bound_metal_sites(frame_pdb, include_unbound_metals=True)
        matched = []
        for candidate in candidates:
            same_element = [c for c in detected if c.element.upper() == candidate.element.upper()]
            exact = [c for c in same_element if c.key == candidate.key and c.atom_name == candidate.atom_name]
            hits = exact or same_element
            if len(hits) != 1:
                raise ValueError(f"Cannot uniquely identify site {candidate.site} ({candidate.element}) in this frame.")
            matched.append(replace(hits[0], site=candidate.site))
        if len({c.atom_index for c in matched}) != len(matched):
            raise ValueError("Multiple selected metals map to the same frame atom.")
        return matched


def restraint_neighborhood(pdb_path, candidate, mapping, *, selected, cutoff, expansion):
    """Expand candidate visibility, never the flat-bottom width or selection.

Protein residues containing selected atoms supply expansion centers; for
ligands the selected atoms themselves do. Explicit residue selection in the
editor can select all visible non-water heavy atoms from a residue.
"""
    structure = load_structure(pdb_path)
    atoms = _iter_indexed_atoms(structure)
    metal = atoms[candidate.atom_index - 1]
    selected = set(selected)
    anchor_residues = {a.residue_key for a in atoms if mapping.get(a.atom_index) in selected
                       and a.classification == "standard"}
    centers = [a for a in atoms if a.element not in {"H", "D", "X"} and
               (mapping.get(a.atom_index) in selected or a.residue_key in anchor_residues)]
    rows = []
    residue_ids = {block[0].residue_key: i for i, block in enumerate(_residues(atoms), start=1)}
    for atom in atoms:
        if atom.element in {"H", "D", "X"} or atom.classification in {"water", "metal"}:
            continue
        index = mapping.get(atom.atom_index)
        distance = metal.position.dist(atom.position)
        near_anchor = min((a.position.dist(atom.position) for a in centers), default=math.inf)
        if index not in selected and distance > cutoff + expansion and near_anchor > expansion:
            continue
        if not math.isfinite(distance):
            raise ValueError("Non-finite coordinates in restraint neighborhood.")
        if index in selected and structure.cell.is_crystal():
            nearest = structure.cell.find_nearest_pbc_image(metal.position, atom.position, 0).dist()
            if abs(distance - nearest) > .01:
                raise ValueError("Selected restraint crosses a periodic boundary. Image the coordinating molecules together.")
        rows.append({"atom_index": index, "topology_index": index, "pdb_atom_index": atom.atom_index,
                     "residue_id": residue_ids[atom.residue_key], "residue": atom.residue_key,
                     "residue_name": atom.residue_name, "atom_name": atom.atom_name,
                     "label": f"{atom.residue_key}:{atom.atom_name}", "element": atom.element,
                     "distance_angstrom": distance, "selected": index in selected})
    return sorted(rows, key=lambda row: (row["residue_id"], row["pdb_atom_index"]))


def inspect_coordination(
    pdb_path: str | Path, candidates: list[MetalSiteCandidate], *, cutoff: float,
    report_only: bool = False,
) -> CoordinationGeometry:
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Coordination cutoff must be finite and positive")
    structure = load_structure(pdb_path)
    atoms = _iter_indexed_atoms(structure)
    sites = {}
    for candidate in candidates:
        if candidate.atom_index > len(atoms):
            raise ValueError("Selected metal is missing from the restraint structure")
        metal = atoms[candidate.atom_index - 1]
        if metal.element != candidate.element.upper() or metal.atom_name.upper() != candidate.atom_name.upper():
            raise ValueError("Metal identity differs between reference PDB and selected frame")
        donors, waters = [], []
        for atom in atoms:
            if atom.atom_index == metal.atom_index or atom.classification == "metal":
                continue
            if atom.element not in SUPPORTED_DONOR_ELEMENTS:
                continue
            distance = metal.position.dist(atom.position)
            if not math.isfinite(distance):
                raise ValueError("Non-finite coordinates in restraint structure")
            if structure.cell.is_crystal():
                nearest = structure.cell.find_nearest_pbc_image(metal.position, atom.position, 0).dist()
                if atom.classification == "water":
                    distance = nearest  # Report hydration CN without restraining water identities.
                elif report_only:
                    distance = nearest
                elif nearest <= cutoff and abs(distance - nearest) > 0.01:
                    raise ValueError(
                        f"Coordination site crosses a periodic boundary near atom {atom.atom_index}. "
                        "Image the metal and coordinating molecules together before setting restraints."
                    )
            if distance > cutoff:
                continue
            entry = {"atom_index": atom.atom_index, "label": f"{atom.residue_key}:{atom.atom_name}",
                     "residue": atom.residue_key, "residue_name": atom.residue_name,
                     "atom_name": atom.atom_name,
                     "element": atom.element, "distance_angstrom": distance}
            (waters if atom.classification == "water" else donors).append(entry)
        sites[candidate.site] = {"metal_atom_index": candidate.atom_index, "element": candidate.element,
                                 "donors": donors, "waters": waters,
                                 "donor_count": len(donors), "water_count": len(waters),
                                 "oxygen_count": sum(d["element"] == "O" for d in donors),
                                 "water_oxygen_count": sum(d["element"] == "O" for d in waters)}
    return CoordinationGeometry(str(Path(pdb_path).resolve()), sites)


def build_coordination_restraints(
    *, reference_pdb: str | Path, snapshot_pdb: str | Path, source_prmtop: str | Path,
    ti_prmtop: str | Path, candidates: list[MetalSiteCandidate], config: CoordinationRestraintConfig,
    output_dir: Path, dry_run: bool,
) -> dict:
    if not candidates:
        raise ValueError("No selected metal sites exist for coordination restraints.")
    snapshot_pdb = normalize_full_reference(
        reference_pdb=snapshot_pdb, prmtop_path=source_prmtop, output_dir=output_dir / "full_snapshot")
    validate_restraint_atom_order(snapshot_pdb, source_prmtop)
    mapping = map_to_topology(reference_pdb, source_prmtop)
    candidates = remap_candidates(candidates, mapping)
    reference_pdb = normalize_full_reference(
        reference_pdb=reference_pdb, prmtop_path=source_prmtop, output_dir=output_dir / "full_reference",
        coordinate_candidates=[snapshot_pdb])
    mapping = map_to_topology(reference_pdb, source_prmtop)
    snapshot_candidates = candidates
    reference = inspect_coordination(reference_pdb, candidates, cutoff=config.donor_cutoff_angstrom)
    snapshot = inspect_coordination(snapshot_pdb, snapshot_candidates, cutoff=config.donor_cutoff_angstrom)
    basis = reference if config.reference == RestraintReference.REFERENCE_PDB else snapshot
    unknown_sites = set(config.donor_indices_by_site) - {item.site for item in candidates}
    if unknown_sites:
        raise ValueError(f"Restraint donor selections refer to unselected metal sites: {sorted(unknown_sites)}")
    start_atoms = _iter_indexed_atoms(snapshot_pdb)
    sites, chunks = [], []
    for candidate, top_candidate in zip(candidates, snapshot_candidates, strict=True):
        detected = basis.sites[candidate.site]
        basis_mapping = mapping if config.reference == RestraintReference.REFERENCE_PDB else {
            a.atom_index: a.atom_index for a in start_atoms}
        initial = [basis_mapping[d["atom_index"]] for d in detected["donors"] if d["atom_index"] in basis_mapping]
        chosen = config.donor_indices_by_site.get(candidate.site, initial)
        # Explicit selections may extend beyond the initial CN cutoff.
        available = {item["topology_index"]: item for item in restraint_neighborhood(
            basis.source_path, candidate if config.reference == RestraintReference.REFERENCE_PDB else top_candidate,
            basis_mapping, selected=chosen, cutoff=config.donor_cutoff_angstrom, expansion=0.0,
        )}
        if not chosen or any(index not in available for index in chosen):
            raise ValueError(
                f"Site {candidate.site}: selected donors are missing, ambiguous, water, hydrogen, extra points or metal atoms. "
                "Review the reference/frame and topology mapping."
            )
        validate_restraint_atom_order(
            snapshot_pdb, ti_prmtop, atom_indices=[top_candidate.atom_index, *chosen],
        )
        start_metal = start_atoms[top_candidate.atom_index - 1]
        pairs = []
        for index in chosen:
            donor = available[index]
            distance = donor["distance_angstrom"]
            if distance <= config.half_width_angstrom:
                raise ValueError("Flat-bottom half-width must be smaller than every target metal-donor distance")
            r2, r3 = distance - config.half_width_angstrom, distance + config.half_width_angstrom
            r1, r4 = max(0.0, r2 - config.half_width_angstrom), r3 + config.half_width_angstrom
            start_distance = start_metal.position.dist(start_atoms[index - 1].position)
            pair = {**donor, "target_distance_angstrom": distance,
                    "r1_angstrom": r1, "r2_angstrom": r2, "r3_angstrom": r3, "r4_angstrom": r4,
                    "force_constant": config.force_constant, "start_distance_angstrom": start_distance,
                    "start_outside_flat_bottom": not r2 <= start_distance <= r3}
            pairs.append(pair)
            chunks.append(
                "&rst\n"
                f"  iat = {top_candidate.atom_index}, {index},\n"
                f"  r1 = {r1:.6f}, r2 = {r2:.6f}, r3 = {r3:.6f}, r4 = {r4:.6f},\n"
                f"  rk2 = {config.force_constant:.6f}, rk3 = {config.force_constant:.6f},\n"
                "  ifvari = 0,\n/\n"
            )
        sites.append({"site": candidate.site, "metal_atom_index": top_candidate.atom_index, "pairs": pairs,
                      "correction_kcal_mol": None})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "coordination.disang"
    target.write_text("".join(chunks), encoding="utf-8")
    payload = {
        "scheme_version": "flat_bottom_donor_pairs_v1", "restraint_file": str(target),
        "reference_source": config.reference.value, "reference_path": basis.source_path,
        "reference_sha256": file_sha256(basis.source_path), "sites": sites,
        "donor_cutoff_angstrom": config.donor_cutoff_angstrom, "water_donors_restrained": False,
        "comparison": {"reference_pdb": reference.sites, "snapshot": snapshot.sites},
        "lambda_dependent": False, "active_at_dummy_endpoint": True,
        "correction_status": "not_computed", "correction_kcal_mol": None,
        "warning": CORRECTION_NOTICE, "dry_run": dry_run,
    }
    write_json(output_dir / "coordination.json", payload)
    (output_dir / "README.txt").write_text(CORRECTION_NOTICE + "\n", encoding="utf-8")
    return payload
