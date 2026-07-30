from __future__ import annotations

import math
from pathlib import Path

from amber_metallo.qm.nwchem import load_molecule


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )


def _reorder_charges_from_reference(
    *,
    mol2_path: Path,
    charges: list[float],
    reference_structure: str | Path,
    coordinate_tolerance: float,
) -> list[float]:
    target_molecule = load_molecule(mol2_path)
    reference_molecule = load_molecule(reference_structure)

    if len(reference_molecule.atoms) != len(charges):
        raise ValueError(
            "RESP charge count does not match the number of atoms in the reference structure. "
            f"Expected {len(reference_molecule.atoms)}, received {len(charges)}."
        )
    if len(target_molecule.atoms) != len(charges):
        raise ValueError(
            "RESP charge count does not match the number of atoms in the target MOL2 file. "
            f"Expected {len(target_molecule.atoms)}, received {len(charges)}."
        )

    reordered = [0.0] * len(charges)
    unused_reference_positions = set(range(len(reference_molecule.atoms)))
    for target_position, target_atom in enumerate(target_molecule.atoms):
        target_coords = (target_atom.x, target_atom.y, target_atom.z)
        candidates: list[tuple[float, int]] = []
        for reference_position in unused_reference_positions:
            reference_atom = reference_molecule.atoms[reference_position]
            if reference_atom.element.upper() != target_atom.element.upper():
                continue
            reference_coords = (reference_atom.x, reference_atom.y, reference_atom.z)
            distance = _distance(target_coords, reference_coords)
            if distance <= coordinate_tolerance:
                candidates.append((distance, reference_position))
        if not candidates:
            raise ValueError(
                "RESP charge mapping failed because the typed MOL2 atom order no longer matches the RESP/source atom order, "
                f"and no coordinate match within {coordinate_tolerance:.4f} A was found for target atom "
                f"'{target_atom.name}' ({target_atom.element})."
            )
        candidates.sort(key=lambda item: item[0])
        _, best_reference_position = candidates[0]
        reordered[target_position] = float(charges[best_reference_position])
        unused_reference_positions.remove(best_reference_position)
    return reordered


def apply_charges_to_mol2(
    input_mol2: str | Path,
    charges: list[float],
    *,
    output_mol2: str | Path | None = None,
    reference_structure: str | Path | None = None,
    coordinate_tolerance: float = 0.05,
) -> Path:
    source = Path(input_mol2)
    target = Path(output_mol2) if output_mol2 is not None else source
    lines = source.read_text(encoding="utf-8").splitlines()

    in_atom_section = False
    atom_line_indexes: list[int] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "@<TRIPOS>ATOM":
            in_atom_section = True
            continue
        if stripped.startswith("@<TRIPOS>") and stripped != "@<TRIPOS>ATOM":
            in_atom_section = False
        if in_atom_section and stripped:
            atom_line_indexes.append(index)

    if len(atom_line_indexes) != len(charges):
        raise ValueError(
            "RESP charge count does not match the number of atoms in the MOL2 file. "
            f"Expected {len(atom_line_indexes)}, received {len(charges)}."
        )

    resolved_charges = (
        _reorder_charges_from_reference(
            mol2_path=source,
            charges=charges,
            reference_structure=reference_structure,
            coordinate_tolerance=coordinate_tolerance,
        )
        if reference_structure is not None
        else [float(charge) for charge in charges]
    )

    for atom_index, line_index in enumerate(atom_line_indexes):
        tokens = lines[line_index].split()
        if len(tokens) < 9:
            raise ValueError(f"Unexpected MOL2 atom line format: {lines[line_index]}")
        tokens[8] = f"{float(resolved_charges[atom_index]):.6f}"
        lines[line_index] = (
            f"{int(tokens[0]):>7d} "
            f"{tokens[1]:<8s} "
            f"{float(tokens[2]):>10.4f} "
            f"{float(tokens[3]):>10.4f} "
            f"{float(tokens[4]):>10.4f} "
            f"{tokens[5]:<10s} "
            f"{int(tokens[6]):>4d} "
            f"{tokens[7]:<8s} "
            f"{float(tokens[8]):>10.6f}"
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
