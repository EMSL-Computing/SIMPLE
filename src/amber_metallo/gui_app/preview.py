from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from amber_metallo.config import DESConfig, DESMixingMode, SaltConfig
from amber_metallo.des import (
    DES_COMPONENTS,
    _centered_coordinates,
    _des_metal_atoms_for_plan,
    _format_pdb_atom,
    _grid_added_ion_atoms_for_plan,
    _replicate_cell_coordinate,
    _replicate_grid_layout,
    _replicate_reserved_cell_indices,
    _random_mix_pdb_text,
    _template_atoms,
    estimate_des_plan,
    resolve_ref_data_dir,
)
from amber_metallo.inspection import SUPPORTED_METALS, classify_residue, inspect_structure, load_structure, residue_key
from amber_metallo.ligand_param import prepare_canonical_small_molecule_mol2
from amber_metallo.qm.nwchem import MoleculeAtom, MoleculeBond, MoleculeData, load_molecule, render_preview_mol2, suggest_group_constraints


VMD_CPK_COLORS = {
    "H": "#ffffff",
    "C": "#8c8c8c",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "P": "#ff8000",
    "S": "#ffff30",
    "CL": "#1ff01f",
    "BR": "#a62929",
    "I": "#940094",
    "FE": "#e06633",
    "CO": "#f090a0",
    "CU": "#c88033",
    "NI": "#50d050",
    "MN": "#9c7ac7",
    "Y": "#94ffff",
    "LA": "#70d4ff",
    "ND": "#c7ffc7",
    "EU": "#61ffc7",
    "LU": "#00ab24",
    "NA": "#ab5cf2",
    "K": "#8f40d4",
    "CA": "#3dff00",
}

DONOR_ELEMENTS = {"N", "O", "S", "P"}
SUPPORTED_METAL_TITLES = {item.title() for item in SUPPORTED_METALS}
MAX_QUICK_MIN_DONORS = 12
DEFAULT_INSERTED_METAL_DONOR_DISTANCE_ANGSTROM = 2.15
MIN_INSERTED_METAL_HEAVY_DISTANCE_ANGSTROM = 2.35
MIN_INSERTED_METAL_HYDROGEN_DISTANCE_ANGSTROM = 1.45
MIN_INSERTED_METAL_METAL_DISTANCE_ANGSTROM = 3.00
INSERTED_METAL_SOFT_SHELL_ANGSTROM = 0.65
PROTEIN_METAL_BINDING_DONORS = {
    "ASP": ("OD1", "OD2"),
    "ASH": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "GLH": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "HID": ("ND1", "NE2"),
    "HIE": ("ND1", "NE2"),
    "HIP": ("ND1", "NE2"),
    "CYS": ("SG",),
    "CYM": ("SG",),
    "CYX": ("SG",),
    "MET": ("SD",),
}
PROTEIN_METAL_BINDING_CUTOFF_ANGSTROM = 3.0
DISULFIDE_CUTOFF_ANGSTROM = 2.35


def _element_key(element: str | None) -> str:
    token = str(element or "C").strip()
    if not token:
        return "C"
    return token.upper()


def color_for_element(element: str | None) -> str:
    key = _element_key(element)
    return VMD_CPK_COLORS.get(key, VMD_CPK_COLORS.get(key[:1], "#9ca3af"))


def _format_atom_name(name: str, element: str) -> str:
    atom_name = (name.strip() or element.strip() or "X")[:4]
    element_token = "".join(ch for ch in element.strip() if ch.isalpha())[:2]
    return atom_name.ljust(4) if len(element_token) == 2 else atom_name.rjust(4)


def _preview_bond_allowed(bond: MoleculeBond, atom_lookup: dict[int, MoleculeAtom]) -> bool:
    first = atom_lookup.get(int(bond.first))
    second = atom_lookup.get(int(bond.second))
    if first is None or second is None:
        return False
    first_element = str(first.element or "").strip().title()
    second_element = str(second.element or "").strip().title()
    first_is_metal = first_element in SUPPORTED_METAL_TITLES
    second_is_metal = second_element in SUPPORTED_METAL_TITLES
    if first_is_metal == second_is_metal:
        return True
    partner = second_element if first_is_metal else first_element
    return partner.upper() in DONOR_ELEMENTS


def preview_bonds(molecule: MoleculeData) -> list[MoleculeBond]:
    atom_lookup = {int(atom.index): atom for atom in molecule.atoms}
    return [bond for bond in molecule.bonds if _preview_bond_allowed(bond, atom_lookup)]


def molecule_to_pdb_text(molecule: MoleculeData, *, residue_name: str = "LIG") -> str:
    serial_for_index = {int(atom.index): serial for serial, atom in enumerate(molecule.atoms, start=1)}
    lines = ["HEADER    SIMPLE GUI molecule preview\n"]
    residue = (residue_name.strip().upper() or "LIG")[:3]
    for serial, atom in enumerate(molecule.atoms, start=1):
        element = str(atom.element or "C").strip().title()
        lines.append(
            f"HETATM{serial:5d} {_format_atom_name(atom.name, element)} {residue:>3s} A{1:4d}    "
            f"{float(atom.x):8.3f}{float(atom.y):8.3f}{float(atom.z):8.3f}  1.00  0.00          {element[:2].upper():>2s}\n"
        )
    conect: dict[int, set[int]] = {}
    for bond in preview_bonds(molecule):
        first = serial_for_index.get(int(bond.first))
        second = serial_for_index.get(int(bond.second))
        if first is None or second is None:
            continue
        conect.setdefault(first, set()).add(second)
        conect.setdefault(second, set()).add(first)
    for first in sorted(conect):
        bonded = sorted(conect[first])
        for start in range(0, len(bonded), 4):
            chunk = bonded[start : start + 4]
            lines.append(f"CONECT{first:5d}" + "".join(f"{item:5d}" for item in chunk) + "\n")
    lines.append("END\n")
    return "".join(lines)


def molecule_payload(
    molecule: MoleculeData,
    *,
    residue_name: str = "LIG",
    group_constraints: dict[str, object] | None = None,
) -> dict[str, Any]:
    metals = []
    donors = []
    atoms = []
    for atom in molecule.atoms:
        element = str(atom.element or "").strip().title()
        is_metal = element in SUPPORTED_METAL_TITLES
        item = {
            "index": int(atom.index),
            "name": atom.name,
            "element": element,
            "x": float(atom.x),
            "y": float(atom.y),
            "z": float(atom.z),
            "partial_charge": None if atom.charge is None else float(atom.charge),
            "color": color_for_element(element),
            "is_metal": is_metal,
            "is_donor_candidate": element.upper() in DONOR_ELEMENTS and not is_metal,
        }
        atoms.append(item)
        if item["is_metal"]:
            metals.append(item)
        if item["is_donor_candidate"]:
            donors.append(item)
    return {
        "atoms": atoms,
        "bonds": [
            {"first": int(bond.first), "second": int(bond.second), "order": int(bond.order or 1)}
            for bond in preview_bonds(molecule)
        ],
        "metals": metals,
        "donor_candidates": donors,
        "pdb": molecule_to_pdb_text(molecule, residue_name=residue_name),
        "mol2": render_preview_mol2(molecule, residue_name=residue_name),
        "group_constraints": group_constraints or suggest_group_constraints(molecule),
    }


def load_small_molecule_for_preview(
    source_file: str | Path,
    *,
    residue_name: str,
    output_dir: Path,
) -> tuple[Path, MoleculeData]:
    source = Path(source_file).expanduser().resolve()
    if source.suffix.lower().lstrip(".") in {"smi", "smiles", "txt"}:
        preview_source = prepare_canonical_small_molecule_mol2(
            source_file=source,
            residue_name=residue_name,
            output_dir=output_dir,
            split_supported_metals=False,
            canonical_filename=f"{residue_name}_gui_preview.mol2",
        )
        return preview_source, load_molecule(preview_source)
    return source, load_molecule(source)


def _molecule_centroid(molecule: MoleculeData) -> tuple[float, float, float]:
    if not molecule.atoms:
        return (0.0, 0.0, 0.0)
    n = float(len(molecule.atoms))
    return (
        sum(float(atom.x) for atom in molecule.atoms) / n,
        sum(float(atom.y) for atom in molecule.atoms) / n,
        sum(float(atom.z) for atom in molecule.atoms) / n,
    )


def _distance_between_atoms(first: Any, second: Any) -> float:
    return math.sqrt(
        (float(first.x) - float(second.x)) ** 2
        + (float(first.y) - float(second.y)) ** 2
        + (float(first.z) - float(second.z)) ** 2
    )


def _coordination_atom_label(atom: Any) -> str:
    name = str(getattr(atom, "name", "") or "").strip()
    element = str(getattr(atom, "element", "") or "").strip().title() or "X"
    index = int(getattr(atom, "index"))
    if name:
        return f"{name}{index if not any(ch.isdigit() for ch in name) else ''}"
    return f"{element}{index}"


def resolve_coordination_donors(
    molecule: MoleculeData,
    *,
    metal_atom_index: int,
    required_donor_atom_indices: list[int],
    target_coordination_number: int | None = None,
    max_donors: int = MAX_QUICK_MIN_DONORS,
) -> tuple[list[int], list[int], list[str]]:
    """Return required donors plus nearest eligible donor candidates up to target CN."""
    atom_by_index = {int(atom.index): atom for atom in molecule.atoms}
    metal_index = int(metal_atom_index)
    metal = atom_by_index.get(metal_index)
    if metal is None:
        raise ValueError(f"Metal atom index {metal_atom_index} was not found.")

    target = None if target_coordination_number is None else int(target_coordination_number)
    if target is not None and target < 1:
        raise ValueError("Target coordination number must be 1 or greater.")

    required: list[int] = []
    seen: set[int] = set()
    for raw_index in required_donor_atom_indices or []:
        donor_index = int(raw_index)
        if donor_index in seen:
            continue
        donor = atom_by_index.get(donor_index)
        if donor is None:
            raise ValueError(f"Required donor atom index {donor_index} was not found.")
        if donor_index == metal_index:
            raise ValueError("The selected metal atom cannot also be used as a donor.")
        donor_element = str(donor.element or "").strip().title()
        if donor_element in SUPPORTED_METAL_TITLES:
            raise ValueError(f"Required donor atom {donor_index} is another supported metal site.")
        required.append(donor_index)
        seen.add(donor_index)

    if target is None:
        target = len(required)

    warnings: list[str] = []
    effective = list(required)
    auto_filled: list[int] = []
    if len(required) > target:
        raise ValueError(
            f"Selected donor count ({len(required)}) exceeds target CN {target}. "
            "Choose Manual selection to minimize with all selected donors, or deselect donors until the count matches Target CN."
        )
    elif len(required) < target:
        candidates = []
        for atom in molecule.atoms:
            atom_index = int(atom.index)
            if atom_index == metal_index or atom_index in seen:
                continue
            element = str(atom.element or "").strip().title()
            if element in SUPPORTED_METAL_TITLES or element.upper() not in DONOR_ELEMENTS:
                continue
            candidates.append((_distance_between_atoms(metal, atom), atom_index, atom))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _distance, atom_index, atom in candidates:
            if len(effective) >= target:
                break
            effective.append(atom_index)
            auto_filled.append(atom_index)
            seen.add(atom_index)
        if auto_filled:
            labels = ", ".join(_coordination_atom_label(atom_by_index[index]) for index in auto_filled)
            warnings.append(f"Auto-filled donor anchors to target CN {target}: {labels}.")
        if len(effective) < target:
            warnings.append(
                f"Only {len(effective)} eligible donor anchor(s) were available for target CN {target}."
            )

    if len(effective) > max_donors:
        raise ValueError(
            f"Effective donor count ({len(effective)}) exceeds the quick minimization limit of {max_donors}. "
            "Lower the target CN or build RESP/QM assets for manual inspection."
        )
    return effective, auto_filled, warnings


def _normalized_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _point_distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))


def _point_from_atom(atom: MoleculeAtom) -> tuple[float, float, float]:
    return (float(atom.x), float(atom.y), float(atom.z))


def _add_points(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (first[0] + second[0], first[1] + second[1], first[2] + second[2])


def _scale_point(point: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (point[0] * scale, point[1] * scale, point[2] * scale)


def _vector_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def _dot_points(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross_points(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _append_molecule_direction(
    directions: list[tuple[float, float, float]],
    vector: tuple[float, float, float],
) -> None:
    if math.sqrt(sum(component * component for component in vector)) < 1.0e-6:
        return
    unit = _normalized_vector(vector)
    if any(_dot_points(unit, existing) > 0.98 for existing in directions):
        return
    directions.append(unit)


def _molecule_repulsion_direction(
    molecule: MoleculeData,
    point: tuple[float, float, float],
    *,
    excluding_indices: set[int],
) -> tuple[float, float, float]:
    vector = (0.0, 0.0, 0.0)
    for atom in molecule.atoms:
        if int(atom.index) in excluding_indices:
            continue
        atom_position = _point_from_atom(atom)
        distance = max(_point_distance(point, atom_position), 0.25)
        if distance > 6.0:
            continue
        direction = _normalized_vector(_vector_between(atom_position, point))
        vector = _add_points(vector, _scale_point(direction, 1.0 / (distance * distance)))
    return vector


def _inserted_metal_candidate_directions(
    molecule: MoleculeData,
    donors: list[MoleculeAtom],
    center: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    donor_positions = [_point_from_atom(atom) for atom in donors]
    excluding = {int(atom.index) for atom in donors}
    molecule_center = _molecule_centroid(molecule)
    directions: list[tuple[float, float, float]] = []
    _append_molecule_direction(directions, _molecule_repulsion_direction(molecule, center, excluding_indices=excluding))
    _append_molecule_direction(directions, _vector_between(molecule_center, center))

    relative = [_vector_between(center, point) for point in donor_positions]
    for first_index, first in enumerate(relative):
        for second in relative[first_index + 1 :]:
            normal = _cross_points(first, second)
            _append_molecule_direction(directions, normal)
            _append_molecule_direction(directions, _scale_point(normal, -1.0))

    if len(donor_positions) == 2:
        donor_axis = _vector_between(donor_positions[0], donor_positions[1])
        outward = directions[0] if directions else _normalized_vector(_vector_between(molecule_center, center))
        first_perp = _cross_points(donor_axis, outward)
        second_perp = _cross_points(donor_axis, first_perp)
        for vector in (first_perp, second_perp):
            _append_molecule_direction(directions, vector)
            _append_molecule_direction(directions, _scale_point(vector, -1.0))

    for vector in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        _append_molecule_direction(directions, vector)
    return directions or [(1.0, 0.0, 0.0)]


def _inserted_metal_clash_threshold(atom: MoleculeAtom) -> float:
    element = str(atom.element or atom.name or "C").strip().title()
    if element in SUPPORTED_METAL_TITLES:
        return MIN_INSERTED_METAL_METAL_DISTANCE_ANGSTROM
    if element.upper() == "H":
        return MIN_INSERTED_METAL_HYDROGEN_DISTANCE_ANGSTROM
    return MIN_INSERTED_METAL_HEAVY_DISTANCE_ANGSTROM


def _inserted_metal_clash_metrics(
    molecule: MoleculeData,
    position: tuple[float, float, float],
    *,
    excluding_indices: set[int],
) -> tuple[int, float]:
    hard_clashes = 0
    penalty = 0.0
    for atom in molecule.atoms:
        if int(atom.index) in excluding_indices:
            continue
        distance = _point_distance(position, _point_from_atom(atom))
        threshold = _inserted_metal_clash_threshold(atom)
        if distance < threshold:
            hard_clashes += 1
            penalty += (threshold - distance + 1.0) ** 2
        elif distance < threshold + INSERTED_METAL_SOFT_SHELL_ANGSTROM:
            penalty += 0.05 * (threshold + INSERTED_METAL_SOFT_SHELL_ANGSTROM - distance) ** 2
    return hard_clashes, penalty


def _inserted_metal_donor_penalty(position: tuple[float, float, float], donors: list[MoleculeAtom]) -> float:
    if not donors:
        return 0.0
    squared = [
        (_point_distance(position, _point_from_atom(atom)) - DEFAULT_INSERTED_METAL_DONOR_DISTANCE_ANGSTROM) ** 2
        for atom in donors
    ]
    return sum(squared) / len(squared)


def _clash_aware_inserted_metal_position(
    molecule: MoleculeData,
    donors: list[MoleculeAtom],
    initial_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    donor_positions = [_point_from_atom(atom) for atom in donors]
    center = donor_positions[0] if len(donors) == 1 else (
        sum(point[0] for point in donor_positions) / len(donor_positions),
        sum(point[1] for point in donor_positions) / len(donor_positions),
        sum(point[2] for point in donor_positions) / len(donor_positions),
    )
    directions = _inserted_metal_candidate_directions(molecule, donors, center)
    candidates = [initial_position]
    if len(donors) == 1:
        radii = [DEFAULT_INSERTED_METAL_DONOR_DISTANCE_ANGSTROM, 1.90, 2.40, 2.65, 3.00, 3.35]
        for radius in radii:
            for direction in directions:
                candidates.append(_add_points(center, _scale_point(direction, radius)))
    else:
        rms_from_center = math.sqrt(sum(_point_distance(point, center) ** 2 for point in donor_positions) / len(donor_positions))
        ideal_offset = math.sqrt(max(DEFAULT_INSERTED_METAL_DONOR_DISTANCE_ANGSTROM**2 - rms_from_center**2, 0.0))
        offsets = {
            0.0,
            ideal_offset,
            max(0.0, ideal_offset - 0.35),
            ideal_offset + 0.35,
            max(0.0, ideal_offset - 0.70),
            ideal_offset + 0.70,
            0.35,
            0.70,
            1.05,
            1.40,
            1.75,
            2.15,
            2.50,
            2.85,
            3.20,
            3.60,
            4.00,
        }
        for offset in sorted(value for value in offsets if value <= 4.00):
            if offset < 1.0e-6:
                candidates.append(center)
                continue
            for direction in directions:
                candidates.append(_add_points(center, _scale_point(direction, offset)))

    excluding = {int(atom.index) for atom in donors}
    best_position = initial_position
    hard, clash_penalty = _inserted_metal_clash_metrics(molecule, initial_position, excluding_indices=excluding)
    donor_penalty = _inserted_metal_donor_penalty(initial_position, donors)
    best_score = hard * 1_000_000.0 + clash_penalty * 1000.0 + donor_penalty * 10.0
    for candidate in candidates:
        hard, clash_penalty = _inserted_metal_clash_metrics(molecule, candidate, excluding_indices=excluding)
        donor_penalty = _inserted_metal_donor_penalty(candidate, donors)
        displacement = _point_distance(candidate, initial_position)
        score = hard * 1_000_000.0 + clash_penalty * 1000.0 + donor_penalty * 10.0 + displacement * 0.1
        if score < best_score:
            best_score = score
            best_position = candidate
    return best_position


def move_metal_toward_donors(
    molecule: MoleculeData,
    *,
    metal_atom_index: int,
    donor_atom_indices: list[int],
    single_donor_distance_angstrom: float = 2.15,
) -> MoleculeData:
    atom_by_index = {int(atom.index): atom for atom in molecule.atoms}
    metal = atom_by_index.get(int(metal_atom_index))
    donors = [atom_by_index[int(index)] for index in donor_atom_indices if int(index) in atom_by_index]
    if metal is None:
        raise ValueError(f"Metal atom index {metal_atom_index} was not found.")
    if not donors:
        raise ValueError("Choose at least one donor atom before quick minimization.")
    if len(donors) > MAX_QUICK_MIN_DONORS:
        raise ValueError(
            f"Choose {MAX_QUICK_MIN_DONORS} or fewer donor atoms for quick minimization. "
            "For larger coordination environments, prepare a RESP input and inspect the geometry manually."
        )

    donor_center = (
        sum(float(atom.x) for atom in donors) / len(donors),
        sum(float(atom.y) for atom in donors) / len(donors),
        sum(float(atom.z) for atom in donors) / len(donors),
    )
    if len(donors) == 1:
        centroid = _molecule_centroid(molecule)
        direction = _normalized_vector(
            (
                donor_center[0] - centroid[0],
                donor_center[1] - centroid[1],
                donor_center[2] - centroid[2],
            )
        )
        metal_position = (
            donor_center[0] + direction[0] * single_donor_distance_angstrom,
            donor_center[1] + direction[1] * single_donor_distance_angstrom,
            donor_center[2] + direction[2] * single_donor_distance_angstrom,
        )
    else:
        metal_position = donor_center

    moved_atoms = [
        replace(
            atom,
            x=metal_position[0],
            y=metal_position[1],
            z=metal_position[2],
        )
        if int(atom.index) == int(metal_atom_index)
        else atom
        for atom in molecule.atoms
    ]
    return MoleculeData(
        source_file=molecule.source_file,
        source_format=molecule.source_format,
        atoms=moved_atoms,
        bonds=list(molecule.bonds),
    )


def _molecule_with_coordination_bonds(
    molecule: MoleculeData,
    *,
    metal_atom_index: int,
    donor_atom_indices: list[int],
) -> MoleculeData:
    atom_indices = {int(atom.index) for atom in molecule.atoms}
    metal_index = int(metal_atom_index)
    if metal_index not in atom_indices:
        raise ValueError(f"Metal atom index {metal_atom_index} was not found.")
    desired_donors = {int(index) for index in donor_atom_indices}
    existing_pairs = {
        tuple(sorted((int(bond.first), int(bond.second))))
        for bond in molecule.bonds
        if not (
            metal_index in {int(bond.first), int(bond.second)}
            and (int(bond.first) if int(bond.second) == metal_index else int(bond.second)) not in desired_donors
        )
    }
    bonds = [
        bond
        for bond in molecule.bonds
        if not (
            metal_index in {int(bond.first), int(bond.second)}
            and (int(bond.first) if int(bond.second) == metal_index else int(bond.second)) not in desired_donors
        )
    ]
    for donor_index in donor_atom_indices:
        donor = int(donor_index)
        if donor not in atom_indices or donor == metal_index:
            continue
        pair = tuple(sorted((metal_index, donor)))
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        bonds.append(MoleculeBond(first=metal_index, second=donor, order=1))
    return MoleculeData(
        source_file=molecule.source_file,
        source_format=molecule.source_format,
        atoms=list(molecule.atoms),
        bonds=bonds,
    )


def quick_minimize_with_openbabel(
    molecule: MoleculeData,
    *,
    residue_name: str,
    metal_atom_index: int,
    donor_atom_indices: list[int],
    output_dir: Path,
) -> tuple[Path, MoleculeData, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    moved = move_metal_toward_donors(
        molecule,
        metal_atom_index=metal_atom_index,
        donor_atom_indices=donor_atom_indices,
    )
    coordinated = _molecule_with_coordination_bonds(
        moved,
        metal_atom_index=metal_atom_index,
        donor_atom_indices=donor_atom_indices,
    )
    seed_path = output_dir / f"{residue_name}_quick_min_seed.mol2"
    minimized_path = output_dir / f"{residue_name}_quick_minimized.mol2"
    seed_path.write_text(render_preview_mol2(coordinated, residue_name=residue_name), encoding="utf-8")

    obabel = shutil.which("obabel") or shutil.which("babel")
    if obabel is None:
        warnings.append(
            "Open Babel was not found; preview used donor-centered metal placement with temporary metal-donor bonds."
        )
        return seed_path, coordinated, warnings

    command = [
        obabel,
        "-imol2",
        str(seed_path),
        "-omol2",
        "-O",
        str(minimized_path),
        "--minimize",
        "--ff",
        "UFF",
        "--steps",
        "250",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path = output_dir / "quick_minimize_obabel.log"
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
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
    if result.returncode != 0 or not minimized_path.exists():
        warnings.append(f"Open Babel UFF cleanup failed; using donor-centered placement with coordination bonds. See {log_path}.")
        return seed_path, coordinated, warnings
    return minimized_path, load_molecule(minimized_path), warnings


def insert_metal_atom_into_molecule(
    molecule: MoleculeData,
    *,
    element: str,
    donor_atom_indices: list[int],
) -> tuple[MoleculeData, int]:
    symbol = str(element or "").strip().title()
    if symbol not in SUPPORTED_METAL_TITLES:
        raise ValueError(f"Unsupported metal insertion element: {element}")
    atom_by_index = {int(atom.index): atom for atom in molecule.atoms}
    donors = [atom_by_index[int(index)] for index in donor_atom_indices if int(index) in atom_by_index]
    if not donors:
        raise ValueError("Choose at least one donor atom before adding a metal.")
    donor_center = (
        sum(float(atom.x) for atom in donors) / len(donors),
        sum(float(atom.y) for atom in donors) / len(donors),
        sum(float(atom.z) for atom in donors) / len(donors),
    )
    if len(donors) == 1:
        centroid = _molecule_centroid(molecule)
        direction = _normalized_vector(
            (
                donor_center[0] - centroid[0],
                donor_center[1] - centroid[1],
                donor_center[2] - centroid[2],
            )
        )
        position = (
            donor_center[0] + direction[0] * 2.15,
            donor_center[1] + direction[1] * 2.15,
            donor_center[2] + direction[2] * 2.15,
        )
    else:
        position = donor_center
    position = _clash_aware_inserted_metal_position(molecule, donors, position)
    next_index = max((int(atom.index) for atom in molecule.atoms), default=0) + 1
    metal_atom = MoleculeAtom(
        index=next_index,
        name=symbol.upper()[:4],
        element=symbol,
        x=position[0],
        y=position[1],
        z=position[2],
        charge=None,
    )
    return (
        MoleculeData(
            source_file=molecule.source_file,
            source_format=molecule.source_format,
            atoms=[*molecule.atoms, metal_atom],
            bonds=list(molecule.bonds),
        ),
        next_index,
    )


def _seqid_number(residue: Any) -> str:
    number = getattr(residue.seqid, "num", None)
    if number is not None:
        return str(number)
    text = str(residue.seqid).strip()
    return text.split()[0] if text else ""


def _seqid_icode(residue: Any) -> str:
    icode = getattr(residue.seqid, "icode", "")
    return str(icode or "").strip()


def _residue_selection(chain_name: str, residue: Any) -> str:
    seqid = _seqid_number(residue)
    chain = str(chain_name or "").strip()
    if chain and seqid:
        return f":{chain} and {seqid}"
    return seqid


def _atom_xyz(atom: Any) -> list[float]:
    return [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]


def _protein_residue_and_binding_payload(source_path: Path) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    structure = load_structure(source_path)
    metals: list[dict[str, Any]] = []
    residues: list[dict[str, Any]] = []

    for chain in structure[0]:
        chain_name = str(chain.name or "").strip()
        for residue in chain:
            classification = classify_residue(residue)
            key = residue_key(chain_name, residue)
            if classification != "water":
                residues.append(
                    {
                        "key": key,
                        "chain": chain_name,
                        "seqid": str(residue.seqid).strip(),
                        "resid": _seqid_number(residue),
                        "icode": _seqid_icode(residue),
                        "resname": residue.name.strip(),
                        "classification": classification,
                        "atom_count": len(residue),
                        "selection": _residue_selection(chain_name, residue),
                        "notes": [classification.title()] if classification in {"metal", "hetero"} else [],
                    }
                )
            if classification != "metal":
                continue
            for atom in residue:
                element = atom.element.name.title()
                if element not in SUPPORTED_METAL_TITLES:
                    element = residue.name.strip().title()
                if element not in SUPPORTED_METAL_TITLES:
                    continue
                metals.append(
                    {
                        "key": key,
                        "chain": chain_name,
                        "seqid": str(residue.seqid).strip(),
                        "element": element,
                        "atom_name": atom.name.strip(),
                        "xyz": _atom_xyz(atom),
                        "position": atom.pos,
                    }
                )

    binding_keys: set[str] = set()
    links: list[dict[str, Any]] = []
    if metals:
        for chain in structure[0]:
            chain_name = str(chain.name or "").strip()
            for residue in chain:
                if classify_residue(residue) != "standard":
                    continue
                donors = PROTEIN_METAL_BINDING_DONORS.get(residue.name.strip().upper())
                if not donors:
                    continue
                best: tuple[float, Any, dict[str, Any]] | None = None
                for atom in residue:
                    if atom.name.strip().upper() not in donors:
                        continue
                    for metal in metals:
                        distance = float(atom.pos.dist(metal["position"]))
                        if distance > PROTEIN_METAL_BINDING_CUTOFF_ANGSTROM:
                            continue
                        if best is None or distance < best[0]:
                            best = (distance, atom, metal)
                if best is None:
                    continue
                distance, donor_atom, metal = best
                key = residue_key(chain_name, residue)
                binding_keys.add(key)
                links.append(
                    {
                        "residue_key": key,
                        "residue_label": f"{chain_name or '_'}:{_seqid_number(residue)} {residue.name.strip()}",
                        "donor_atom_name": donor_atom.name.strip(),
                        "donor": _atom_xyz(donor_atom),
                        "metal_key": metal["key"],
                        "metal_element": metal["element"],
                        "metal_atom_name": metal["atom_name"],
                        "metal": metal["xyz"],
                        "distance_angstrom": distance,
                    }
                )

    for row in residues:
        if row["key"] in binding_keys:
            row["notes"] = ["Metal binding", *[note for note in row["notes"] if note != "Metal binding"]]
    return residues, sorted(binding_keys), links


def _keys_for_residue_locators(source_path: Path, locators: set[tuple[str, str]]) -> list[str]:
    if not locators:
        return []
    structure = load_structure(source_path)
    keys: list[str] = []
    for chain in structure[0]:
        chain_name = str(chain.name or "").strip()
        for residue in chain:
            locator = (chain_name, _seqid_number(residue))
            if locator in locators:
                keys.append(residue_key(chain_name, residue))
    return sorted(set(keys))


def _disulfide_candidates(source_path: Path) -> list[dict[str, Any]]:
    structure = load_structure(source_path)
    sulfur_atoms: list[tuple[str, str, str, Any]] = []
    for chain in structure[0]:
        chain_name = str(chain.name or "").strip()
        for residue in chain:
            if residue.name.strip().upper() not in {"CYS", "CYX"}:
                continue
            sg = next((atom for atom in residue if atom.name.strip().upper() == "SG"), None)
            if sg is None:
                continue
            key = residue_key(chain_name, residue)
            label = f"{chain_name or '_'}:{_seqid_number(residue)} {residue.name.strip()}"
            sulfur_atoms.append((key, label, chain_name, sg))

    candidates: list[dict[str, Any]] = []
    for idx, first in enumerate(sulfur_atoms):
        for second in sulfur_atoms[idx + 1 :]:
            distance = float(first[3].pos.dist(second[3].pos))
            if distance > DISULFIDE_CUTOFF_ANGSTROM:
                continue
            candidates.append(
                {
                    "token": f"{first[0]}--{second[0]}",
                    "key_a": first[0],
                    "key_b": second[0],
                    "a": first[1],
                    "b": second[1],
                    "distance_angstrom": distance,
                }
            )
    return candidates


def protein_preview_payload(
    source_path: str | Path,
    *,
    missing_loop_locators: set[tuple[str, str]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    summary = inspect_structure(path)
    protein_residues, metal_binding_keys, metal_binding_links = _protein_residue_and_binding_payload(path)
    missing_keys = _keys_for_residue_locators(path, missing_loop_locators or set())
    disulfide_candidates = _disulfide_candidates(path)
    disulfide_keys = sorted(
        {
            key
            for item in disulfide_candidates
            for key in (str(item.get("key_a") or ""), str(item.get("key_b") or ""))
            if key
        }
    )
    return {
        "pdb": path.read_text(encoding="utf-8", errors="ignore"),
        "summary": summary.to_dict(),
        "metals": [asdict(item) for item in summary.metals],
        "hetero_residues": [asdict(item) for item in summary.hetero_residues],
        "protein_residues": protein_residues,
        "metal_binding_links": metal_binding_links,
        "disulfide_candidates": disulfide_candidates,
        "warnings": warnings or [],
        "highlight_sets": {
            "metals": [item.key for item in summary.metals],
            "hetero": [item.key for item in summary.hetero_residues],
            "fixed": [],
            "propka": [],
            "metal_binding": metal_binding_keys,
            "missing": missing_keys,
            "disulfide": disulfide_keys,
        },
    }


def box_lines_angstrom(x: float, y: float, z: float) -> list[dict[str, list[float]]]:
    corners = [
        (0.0, 0.0, 0.0),
        (x, 0.0, 0.0),
        (x, y, 0.0),
        (0.0, y, 0.0),
        (0.0, 0.0, z),
        (x, 0.0, z),
        (x, y, z),
        (0.0, y, z),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [{"start": list(corners[first]), "end": list(corners[second])} for first, second in edges]


def des_heavy_atom_preview(config: DESConfig, salt_config: SaltConfig | None = None) -> dict[str, Any]:
    plan = estimate_des_plan(config, salt_config)
    if config.mixing_mode == DESMixingMode.RANDOM_MIX:
        packed_text = _random_mix_pdb_text(config, plan)
        heavy_lines: list[str] = []
        for line in packed_text.splitlines(keepends=True):
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip().upper() == "H":
                continue
            heavy_lines.append(line)
        return {
            "pdb": "".join(heavy_lines),
            "plan": plan.to_dict(),
            "box_lines": box_lines_angstrom(*plan.box_lengths_angstrom),
            "preview_note": "Fast random-mix coordinates; the system build uses the same deterministic seed.",
            "components": [
                {
                    "key": component.value if hasattr(component, "value") else str(component),
                    "label": DES_COMPONENTS[component].label,
                    "ratio": ratio,
                }
                for component, ratio in zip(config.components, config.ratios, strict=True)
            ],
        }
    ref_data_dir = resolve_ref_data_dir(config.ref_data_dir)
    layout = _replicate_grid_layout(
        config,
        ref_data_dir,
        plan.ratio_units,
        added_ion_count=sum((plan.added_ions or {}).values()),
    )
    offset = (
        max((plan.box_lengths_angstrom[0] - layout.occupied_lengths[0]) / 2.0, 0.0),
        max((plan.box_lengths_angstrom[1] - layout.occupied_lengths[1]) / 2.0, 0.0),
        max((plan.box_lengths_angstrom[2] - layout.occupied_lengths[2]) / 2.0, 0.0),
    )
    serial = 1
    residue_number = 1
    lines = ["HEADER    SIMPLE GUI DES heavy atom preview\n"]
    added_ion_count = sum((plan.added_ions or {}).values())
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
            if atom.element.upper() == "H":
                continue
            if config.mixing_mode == DESMixingMode.PACKMOL:
                x %= plan.box_lengths_angstrom[0]
                y %= plan.box_lengths_angstrom[1]
                z %= plan.box_lengths_angstrom[2]
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
        residue_number += 1
    for central, x, y, z in _des_metal_atoms_for_plan(plan):
        lines.append(
            _format_pdb_atom(
                serial=serial,
                atom=central,
                residue_name=central.residue_name,
                residue_number=residue_number,
                x=x,
                y=y,
                z=z,
            )
        )
        serial += 1
        residue_number += 1
    for residue, ion_atom, x, y, z in _grid_added_ion_atoms_for_plan(config, plan):
        if config.mixing_mode == DESMixingMode.PACKMOL:
            x %= plan.box_lengths_angstrom[0]
            y %= plan.box_lengths_angstrom[1]
            z %= plan.box_lengths_angstrom[2]
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
        residue_number += 1
    lines.append("END\n")
    return {
        "pdb": "".join(lines),
        "plan": plan.to_dict(),
        "box_lines": box_lines_angstrom(*plan.box_lengths_angstrom),
        "preview_note": (
            "The Packmol view is a wrapped schematic. Final random, clash-optimized coordinates are generated "
            "when the system is built."
            if config.mixing_mode == DESMixingMode.PACKMOL
            else ""
        ),
        "components": [
            {
                "key": component.value,
                "label": DES_COMPONENTS[component].label,
                "ratio": ratio,
            }
            for component, ratio in zip(config.components, config.ratios, strict=True)
        ],
    }
