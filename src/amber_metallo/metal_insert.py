from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import gemmi

from amber_metallo.config import MetalAnchorMode, MetalInsertion
from amber_metallo.inspection import SUPPORTED_METALS, classify_residue, residue_key


DONOR_ELEMENTS = {"N", "O", "S", "P"}
DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM = 2.15
MAX_AUTO_DONOR_SEARCH_DISTANCE_ANGSTROM = 6.0
MIN_NON_DONOR_METAL_DISTANCE_ANGSTROM = 2.35
MIN_NON_DONOR_HYDROGEN_DISTANCE_ANGSTROM = 1.45
MIN_EXISTING_METAL_DISTANCE_ANGSTROM = 3.00
CLASH_SOFT_SHELL_ANGSTROM = 0.65
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


@dataclass(slots=True)
class AtomReference:
    atom_serial: int
    chain: str
    seqid: str
    resid: str
    residue_name: str
    atom_name: str
    element: str
    classification: str
    x: float
    y: float
    z: float
    residue_key: str
    selector: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _ResolvedAtom:
    reference: AtomReference
    chain: gemmi.Chain
    residue: gemmi.Residue
    atom: gemmi.Atom


@dataclass(slots=True)
class ResolvedMetalInsertion:
    element: str
    charge: int | None
    anchor_mode: str
    anchors: list[str]
    target_coordination_number: int | None
    label: str | None
    coordinates: tuple[float, float, float]
    donor_atoms: list[AtomReference]
    auto_filled_donor_atoms: list[AtomReference]
    warnings: list[str]
    chain: str
    residue_name: str
    seqid: str | None = None
    site: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coordinates"] = [float(value) for value in self.coordinates]
        return payload


def _seqid_number(residue: gemmi.Residue) -> str:
    number = getattr(residue.seqid, "num", None)
    if number is not None:
        return str(number)
    text = str(residue.seqid).strip()
    return text.split()[0] if text else ""


def _seqid_text(residue: gemmi.Residue) -> str:
    return str(residue.seqid).strip()


def _atom_xyz(atom: gemmi.Atom) -> tuple[float, float, float]:
    return (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))


def _centroid(points: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    values = list(points)
    if not values:
        raise ValueError("Cannot compute a centroid without coordinates.")
    count = float(len(values))
    return (
        sum(point[0] for point in values) / count,
        sum(point[1] for point in values) / count,
        sum(point[2] for point in values) / count,
    )


def _normalized(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm < 1.0e-8:
        return (1.0, 0.0, 0.0)
    return tuple(component / norm for component in vector)  # type: ignore[return-value]


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _add_vectors(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (first[0] + second[0], first[1] + second[1], first[2] + second[2])


def _scale_vector(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _vector_from_to(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def _dot(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _expanded_anchor_tokens(anchors: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for anchor in anchors:
        for token in str(anchor or "").split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _reference_for_atom(
    *,
    chain_name: str,
    residue: gemmi.Residue,
    atom: gemmi.Atom,
) -> AtomReference:
    chain = str(chain_name or "").strip()
    seqid = _seqid_text(residue)
    resid = _seqid_number(residue)
    residue_name = residue.name.strip()
    atom_name = atom.name.strip()
    element = atom.element.name.title()
    selector = f"{chain}:{resid}@{atom_name}" if chain else f"{resid}@{atom_name}"
    return AtomReference(
        atom_serial=int(atom.serial),
        chain=chain,
        seqid=seqid,
        resid=resid,
        residue_name=residue_name,
        atom_name=atom_name,
        element=element,
        classification=classify_residue(residue),
        x=float(atom.pos.x),
        y=float(atom.pos.y),
        z=float(atom.pos.z),
        residue_key=residue_key(chain, residue),
        selector=selector,
    )


def _iter_atoms(structure: gemmi.Structure) -> Iterable[_ResolvedAtom]:
    model = structure[0]
    for chain in model:
        chain_name = str(chain.name or "").strip()
        for residue in chain:
            for atom in residue:
                yield _ResolvedAtom(
                    reference=_reference_for_atom(chain_name=chain_name, residue=residue, atom=atom),
                    chain=chain,
                    residue=residue,
                    atom=atom,
                )


def structure_atom_reference_rows(structure: gemmi.Structure) -> list[dict[str, Any]]:
    return [item.reference.to_dict() for item in _iter_atoms(structure)]


def _residue_matches_selector(chain_name: str, residue: gemmi.Residue, selector: str) -> bool:
    token = selector.strip()
    if not token:
        return False
    key = residue_key(chain_name, residue)
    if token == key:
        return True

    parts = [part.strip() for part in token.split(":")]
    if len(parts) == 3:
        chain_token, residue_token, seq_token = parts
        if chain_token and chain_token != chain_name:
            return False
        if residue_token and residue_token.upper() != residue.name.strip().upper():
            return False
        return _seqid_matches(residue, seq_token)
    if len(parts) == 2:
        chain_token, seq_token = parts
        if chain_token and chain_token != chain_name:
            return False
        return _seqid_matches(residue, seq_token)
    if len(parts) == 1:
        return _seqid_matches(residue, parts[0])
    return False


def _seqid_matches(residue: gemmi.Residue, token: str) -> bool:
    cleaned = token.strip()
    if not cleaned:
        return False
    return cleaned == _seqid_text(residue) or cleaned == _seqid_number(residue)


def _find_residues(structure: gemmi.Structure, selectors: Iterable[str]) -> list[tuple[gemmi.Chain, gemmi.Residue]]:
    selected: list[tuple[gemmi.Chain, gemmi.Residue]] = []
    seen: set[tuple[str, str, str]] = set()
    selector_tokens = _expanded_anchor_tokens(selectors)
    for chain in structure[0]:
        chain_name = str(chain.name or "").strip()
        for residue in chain:
            if not any(_residue_matches_selector(chain_name, residue, selector) for selector in selector_tokens):
                continue
            key = (chain_name, residue.name.strip(), _seqid_text(residue))
            if key in seen:
                continue
            seen.add(key)
            selected.append((chain, residue))
    return selected


def _is_donor_atom(residue: gemmi.Residue, atom: gemmi.Atom) -> bool:
    element = atom.element.name.title()
    if element.upper() not in DONOR_ELEMENTS:
        return False
    if element in {metal.title() for metal in SUPPORTED_METALS}:
        return False
    if classify_residue(residue) == "metal":
        return False
    return True


def _donor_atoms_for_residue(chain: gemmi.Chain, residue: gemmi.Residue) -> list[_ResolvedAtom]:
    chain_name = str(chain.name or "").strip()
    residue_name = residue.name.strip().upper()
    preferred = set(PROTEIN_METAL_BINDING_DONORS.get(residue_name, ()))
    donors: list[_ResolvedAtom] = []
    for atom in residue:
        atom_name = atom.name.strip().upper()
        if preferred and atom_name not in preferred:
            continue
        if not _is_donor_atom(residue, atom):
            continue
        donors.append(
            _ResolvedAtom(
                reference=_reference_for_atom(chain_name=chain_name, residue=residue, atom=atom),
                chain=chain,
                residue=residue,
                atom=atom,
            )
        )
    if donors or preferred:
        return donors
    return [
        _ResolvedAtom(
            reference=_reference_for_atom(chain_name=chain_name, residue=residue, atom=atom),
            chain=chain,
            residue=residue,
            atom=atom,
        )
        for atom in residue
        if _is_donor_atom(residue, atom)
    ]


def donor_candidates_for_residue_selectors(
    structure: gemmi.Structure,
    selectors: Iterable[str],
) -> list[dict[str, Any]]:
    donors: list[AtomReference] = []
    seen: set[int] = set()
    for chain, residue in _find_residues(structure, selectors):
        for donor in _donor_atoms_for_residue(chain, residue):
            serial = int(donor.reference.atom_serial)
            if serial in seen:
                continue
            seen.add(serial)
            donors.append(donor.reference)
    return [donor.to_dict() for donor in donors]


def _find_atom_by_selector(structure: gemmi.Structure, selector: str) -> _ResolvedAtom | None:
    token = selector.strip()
    if not token:
        return None
    if token.isdigit():
        return _find_atom_by_serial(structure, int(token))
    if "@" not in token:
        return None
    residue_selector, atom_selector = token.split("@", 1)
    atom_names = {name.strip().upper() for name in atom_selector.split("|") if name.strip()}
    if not atom_names:
        return None
    for chain, residue in _find_residues(structure, [residue_selector]):
        chain_name = str(chain.name or "").strip()
        for atom in residue:
            if atom.name.strip().upper() not in atom_names:
                continue
            return _ResolvedAtom(
                reference=_reference_for_atom(chain_name=chain_name, residue=residue, atom=atom),
                chain=chain,
                residue=residue,
                atom=atom,
            )
    return None


def _find_atom_by_serial(structure: gemmi.Structure, serial: int) -> _ResolvedAtom | None:
    for item in _iter_atoms(structure):
        if int(item.atom.serial) == int(serial):
            return item
    return None


def _resolve_anchor_atoms(structure: gemmi.Structure, insertion: MetalInsertion) -> list[_ResolvedAtom]:
    mode = insertion.anchor_mode
    tokens = _expanded_anchor_tokens(insertion.anchors)
    resolved: list[_ResolvedAtom] = []
    seen: set[int] = set()

    if mode == MetalAnchorMode.RESIDUE_DONORS:
        candidates = []
        for chain, residue in _find_residues(structure, tokens):
            candidates.extend(_donor_atoms_for_residue(chain, residue))
        for item in candidates:
            serial = int(item.reference.atom_serial)
            if serial not in seen:
                seen.add(serial)
                resolved.append(item)
        return resolved

    if mode == MetalAnchorMode.ATOM_SERIALS:
        for token in tokens:
            if not token.isdigit():
                raise ValueError(f"Atom serial anchor must be an integer: {token}")
            item = _find_atom_by_serial(structure, int(token))
            if item is None:
                raise ValueError(f"Atom serial {token} was not found.")
            serial = int(item.reference.atom_serial)
            if serial not in seen:
                seen.add(serial)
                resolved.append(item)
        return resolved

    if mode == MetalAnchorMode.DONOR_ATOMS:
        for token in tokens:
            item = _find_atom_by_selector(structure, token)
            if item is None:
                raise ValueError(f"Atom anchor '{token}' was not found. Use A:45@ND1 or an atom serial.")
            if not _is_donor_atom(item.residue, item.atom):
                raise ValueError(f"Atom anchor '{token}' is not an O/N/S/P donor atom.")
            serial = int(item.reference.atom_serial)
            if serial not in seen:
                seen.add(serial)
                resolved.append(item)
        return resolved

    return []


def _atom_lookup(residue: gemmi.Residue) -> dict[str, gemmi.Atom]:
    return {atom.name.strip().upper(): atom for atom in residue}


def _heavy_atom_positions(
    residue: gemmi.Residue,
    *,
    exclude_atom: gemmi.Atom | None = None,
) -> list[tuple[float, float, float]]:
    positions = []
    for atom in residue:
        if exclude_atom is not None and atom.name == exclude_atom.name:
            continue
        if atom.element.name.upper() == "H":
            continue
        positions.append(_atom_xyz(atom))
    return positions


def _reference_point_for_single_donor(item: _ResolvedAtom) -> tuple[float, float, float]:
    residue_name = item.residue.name.strip().upper()
    donor_name = item.atom.name.strip().upper()
    atoms = _atom_lookup(item.residue)

    if residue_name in {"HIS", "HID", "HIE", "HIP"}:
        ring_atoms = [
            atoms[name]
            for name in ("CG", "ND1", "CE1", "NE2", "CD2")
            if name in atoms and name != donor_name
        ]
        if ring_atoms:
            return _centroid(_atom_xyz(atom) for atom in ring_atoms)

    if residue_name in {"ASP", "ASH"} and donor_name in {"OD1", "OD2"} and "CG" in atoms:
        return _atom_xyz(atoms["CG"])

    if residue_name in {"GLU", "GLH"} and donor_name in {"OE1", "OE2"} and "CD" in atoms:
        return _atom_xyz(atoms["CD"])

    if residue_name in {"CYS", "CYM", "CYX"} and donor_name == "SG" and "CB" in atoms:
        return _atom_xyz(atoms["CB"])

    if residue_name == "MET" and donor_name == "SD":
        methionine_refs = [atoms[name] for name in ("CG", "CE") if name in atoms]
        if methionine_refs:
            return _centroid(_atom_xyz(atom) for atom in methionine_refs)

    positions = _heavy_atom_positions(item.residue, exclude_atom=item.atom)
    if positions:
        return _centroid(positions)
    return _atom_xyz(item.atom)


def _metal_position_from_donors(donors: list[_ResolvedAtom]) -> tuple[float, float, float]:
    if not donors:
        raise ValueError("At least one donor atom is required for donor-based metal insertion.")
    if len(donors) >= 2:
        return _centroid(_atom_xyz(item.atom) for item in donors)
    donor = donors[0]
    donor_position = _atom_xyz(donor.atom)
    reference_position = _reference_point_for_single_donor(donor)
    direction = _normalized(_vector_from_to(reference_position, donor_position))
    return (
        donor_position[0] + direction[0] * DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM,
        donor_position[1] + direction[1] * DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM,
        donor_position[2] + direction[2] * DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM,
    )


def _structure_centroid(structure: gemmi.Structure) -> tuple[float, float, float]:
    positions = [_atom_xyz(item.atom) for item in _iter_atoms(structure) if item.atom.element.name.upper() != "H"]
    if not positions:
        positions = [_atom_xyz(item.atom) for item in _iter_atoms(structure)]
    return _centroid(positions) if positions else (0.0, 0.0, 0.0)


def _append_direction(
    directions: list[tuple[float, float, float]],
    vector: tuple[float, float, float],
) -> None:
    if _norm(vector) < 1.0e-6:
        return
    unit = _normalized(vector)
    if any(_dot(unit, existing) > 0.98 for existing in directions):
        return
    directions.append(unit)


def _repulsion_direction(
    structure: gemmi.Structure,
    point: tuple[float, float, float],
    *,
    excluding_serials: set[int],
) -> tuple[float, float, float]:
    vector = (0.0, 0.0, 0.0)
    for item in _iter_atoms(structure):
        if int(item.reference.atom_serial) in excluding_serials:
            continue
        atom_position = _atom_xyz(item.atom)
        distance = max(_distance(point, atom_position), 0.25)
        if distance > 6.0:
            continue
        direction = _normalized(_vector_from_to(atom_position, point))
        vector = _add_vectors(vector, _scale_vector(direction, 1.0 / (distance * distance)))
    return vector


def _metal_position_directions(
    structure: gemmi.Structure,
    donors: list[_ResolvedAtom],
    center: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    donor_positions = [_atom_xyz(item.atom) for item in donors]
    excluding = {int(item.reference.atom_serial) for item in donors}
    structure_center = _structure_centroid(structure)
    directions: list[tuple[float, float, float]] = []

    _append_direction(directions, _repulsion_direction(structure, center, excluding_serials=excluding))
    _append_direction(directions, _vector_from_to(structure_center, center))

    relative = [_vector_from_to(center, point) for point in donor_positions]
    for first_index, first in enumerate(relative):
        for second in relative[first_index + 1 :]:
            normal = _cross(first, second)
            _append_direction(directions, normal)
            _append_direction(directions, _scale_vector(normal, -1.0))

    if len(donor_positions) == 2:
        donor_axis = _vector_from_to(donor_positions[0], donor_positions[1])
        outward = directions[0] if directions else _normalized(_vector_from_to(structure_center, center))
        first_perp = _cross(donor_axis, outward)
        second_perp = _cross(donor_axis, first_perp)
        for vector in (first_perp, second_perp):
            _append_direction(directions, vector)
            _append_direction(directions, _scale_vector(vector, -1.0))

    for vector in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        _append_direction(directions, vector)
    return directions or [(1.0, 0.0, 0.0)]


def _clash_threshold(item: _ResolvedAtom) -> float:
    element = item.reference.element.title()
    if element in {metal.title() for metal in SUPPORTED_METALS}:
        return MIN_EXISTING_METAL_DISTANCE_ANGSTROM
    if element.upper() == "H":
        return MIN_NON_DONOR_HYDROGEN_DISTANCE_ANGSTROM
    return MIN_NON_DONOR_METAL_DISTANCE_ANGSTROM


def _clash_metrics(
    structure: gemmi.Structure,
    position: tuple[float, float, float],
    *,
    excluding_serials: set[int],
) -> tuple[int, float | None, float]:
    hard_clashes = 0
    closest: float | None = None
    penalty = 0.0
    for item in _iter_atoms(structure):
        if int(item.reference.atom_serial) in excluding_serials:
            continue
        distance = _distance(position, _atom_xyz(item.atom))
        closest = distance if closest is None else min(closest, distance)
        threshold = _clash_threshold(item)
        if distance < threshold:
            hard_clashes += 1
            penalty += (threshold - distance + 1.0) ** 2
        elif distance < threshold + CLASH_SOFT_SHELL_ANGSTROM:
            penalty += 0.05 * (threshold + CLASH_SOFT_SHELL_ANGSTROM - distance) ** 2
    return hard_clashes, closest, penalty


def _donor_distance_penalty(position: tuple[float, float, float], donors: list[_ResolvedAtom]) -> float:
    if not donors:
        return 0.0
    squared = [
        (_distance(position, _atom_xyz(item.atom)) - DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM) ** 2
        for item in donors
    ]
    return sum(squared) / len(squared)


def _candidate_positions_for_donors(
    structure: gemmi.Structure,
    donors: list[_ResolvedAtom],
    initial_position: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    donor_positions = [_atom_xyz(item.atom) for item in donors]
    center = donor_positions[0] if len(donors) == 1 else _centroid(donor_positions)
    directions = _metal_position_directions(structure, donors, center)
    candidates = [initial_position]

    if len(donors) == 1:
        radii = [DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM, 1.90, 2.40, 2.65, 3.00, 3.35]
        for radius in radii:
            for direction in directions:
                candidates.append(_add_vectors(center, _scale_vector(direction, radius)))
    else:
        rms_from_center = math.sqrt(sum(_distance(point, center) ** 2 for point in donor_positions) / len(donor_positions))
        ideal_offset = math.sqrt(max(DEFAULT_METAL_DONOR_DISTANCE_ANGSTROM**2 - rms_from_center**2, 0.0))
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
                candidates.append(_add_vectors(center, _scale_vector(direction, offset)))

    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        key = tuple(int(round(value * 1000.0)) for value in candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _clash_aware_metal_position(
    structure: gemmi.Structure,
    donors: list[_ResolvedAtom],
    initial_position: tuple[float, float, float],
    *,
    warnings: list[str],
) -> tuple[float, float, float]:
    excluding = {int(item.reference.atom_serial) for item in donors}
    initial_hard, initial_closest, initial_clash_penalty = _clash_metrics(
        structure,
        initial_position,
        excluding_serials=excluding,
    )
    best_position = initial_position
    best_metrics = (
        initial_hard,
        initial_closest if initial_closest is not None else float("inf"),
        initial_clash_penalty,
        _donor_distance_penalty(initial_position, donors),
        0.0,
    )
    best_score = (
        best_metrics[0] * 1_000_000.0
        + best_metrics[2] * 1000.0
        + best_metrics[3] * 10.0
        + best_metrics[4] * 0.1
    )

    for candidate in _candidate_positions_for_donors(structure, donors, initial_position):
        hard, closest, clash_penalty = _clash_metrics(structure, candidate, excluding_serials=excluding)
        donor_penalty = _donor_distance_penalty(candidate, donors)
        displacement = _distance(candidate, initial_position)
        score = hard * 1_000_000.0 + clash_penalty * 1000.0 + donor_penalty * 10.0 + displacement * 0.1
        if score < best_score:
            best_score = score
            best_position = candidate
            best_metrics = (hard, closest if closest is not None else float("inf"), clash_penalty, donor_penalty, displacement)

    if _distance(best_position, initial_position) > 0.10:
        warnings.append(
            "Adjusted inserted metal placement by "
            f"{_distance(best_position, initial_position):.2f} A to reduce overlap/clash risk."
        )
    if best_metrics[0] > 0:
        warnings.append(
            "Inserted metal placement still has "
            f"{best_metrics[0]} close contact(s); closest non-donor atom is {best_metrics[1]:.2f} A away."
        )
    return best_position


def _warn_if_xyz_position_clashes(
    structure: gemmi.Structure,
    coordinates: tuple[float, float, float],
    *,
    warnings: list[str],
) -> None:
    hard, closest, _penalty = _clash_metrics(structure, coordinates, excluding_serials=set())
    if hard > 0:
        closest_text = f"{closest:.2f} A" if closest is not None else "unknown distance"
        warnings.append(
            f"Exact XYZ insertion has {hard} close contact(s); closest existing atom is {closest_text} away."
        )


def _all_donor_atoms(structure: gemmi.Structure, *, excluding_serials: set[int]) -> list[_ResolvedAtom]:
    donors = []
    for item in _iter_atoms(structure):
        if int(item.reference.atom_serial) in excluding_serials:
            continue
        if _is_donor_atom(item.residue, item.atom):
            donors.append(item)
    return donors


def _auto_fill_donors(
    structure: gemmi.Structure,
    donors: list[_ResolvedAtom],
    *,
    target_coordination_number: int | None,
    warnings: list[str],
) -> list[_ResolvedAtom]:
    if target_coordination_number is None or len(donors) >= target_coordination_number:
        if target_coordination_number is not None and len(donors) > target_coordination_number:
            warnings.append(
                f"Selected donor count ({len(donors)}) exceeds target CN {target_coordination_number}; "
                "using all selected donors."
            )
        return []

    selected_serials = {int(item.reference.atom_serial) for item in donors}
    probe_position = _metal_position_from_donors(donors)
    candidates = []
    for item in _all_donor_atoms(structure, excluding_serials=selected_serials):
        distance = _distance(probe_position, _atom_xyz(item.atom))
        if distance > MAX_AUTO_DONOR_SEARCH_DISTANCE_ANGSTROM:
            continue
        candidates.append((distance, int(item.reference.atom_serial), item))
    candidates.sort(key=lambda row: (row[0], row[1]))

    auto_filled: list[_ResolvedAtom] = []
    for _distance_value, _serial, item in candidates:
        if len(donors) + len(auto_filled) >= target_coordination_number:
            break
        auto_filled.append(item)
    if auto_filled:
        labels = ", ".join(item.reference.selector for item in auto_filled)
        warnings.append(f"Auto-filled donor anchors to target CN {target_coordination_number}: {labels}.")
    if len(donors) + len(auto_filled) < target_coordination_number:
        warnings.append(
            f"Only {len(donors) + len(auto_filled)} donor anchor(s) were available for target CN "
            f"{target_coordination_number}."
        )
    return auto_filled


def _normalize_supported_element(element: str) -> str:
    symbol = str(element or "").strip().title()
    if symbol not in {metal.title() for metal in SUPPORTED_METALS}:
        raise ValueError(f"Unsupported metal insertion element: {element}")
    return symbol


def resolve_metal_insertion(
    structure: gemmi.Structure,
    insertion: MetalInsertion,
) -> ResolvedMetalInsertion:
    element = _normalize_supported_element(insertion.element)
    residue_name = element.upper()[:3]
    warnings: list[str] = []

    if insertion.anchor_mode == MetalAnchorMode.XYZ:
        if insertion.coordinates is None:
            raise ValueError("XYZ metal insertion requires coordinates.")
        coordinates = tuple(float(value) for value in insertion.coordinates)
        donors: list[_ResolvedAtom] = []
        auto_filled: list[_ResolvedAtom] = []
        chain = "Z"
        _warn_if_xyz_position_clashes(structure, coordinates, warnings=warnings)
    else:
        donors = _resolve_anchor_atoms(structure, insertion)
        if not donors:
            raise ValueError("No donor atoms were resolved for metal insertion.")
        auto_filled = _auto_fill_donors(
            structure,
            donors,
            target_coordination_number=insertion.target_coordination_number,
            warnings=warnings,
        )
        effective_donors = [*donors, *auto_filled]
        initial_coordinates = _metal_position_from_donors(effective_donors)
        coordinates = _clash_aware_metal_position(
            structure,
            effective_donors,
            initial_coordinates,
            warnings=warnings,
        )
        chain = donors[0].reference.chain or "Z"

    return ResolvedMetalInsertion(
        element=element,
        charge=insertion.charge,
        anchor_mode=insertion.anchor_mode.value,
        anchors=_expanded_anchor_tokens(insertion.anchors),
        target_coordination_number=insertion.target_coordination_number,
        label=insertion.label,
        coordinates=coordinates,  # type: ignore[arg-type]
        donor_atoms=[item.reference for item in donors],
        auto_filled_donor_atoms=[item.reference for item in auto_filled],
        warnings=warnings,
        chain=chain,
        residue_name=residue_name,
    )


def resolve_metal_insertions(
    structure: gemmi.Structure,
    insertions: Iterable[MetalInsertion],
) -> list[ResolvedMetalInsertion]:
    return [resolve_metal_insertion(structure, insertion) for insertion in insertions]


def _find_or_create_chain(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    name = chain_name or "Z"
    for chain in model:
        if str(chain.name or "").strip() == name:
            return chain
    chain = gemmi.Chain(name)
    model.add_chain(chain)
    return model[-1]


def _next_residue_number(chain: gemmi.Chain) -> int:
    numbers = [
        int(getattr(residue.seqid, "num", 0) or 0)
        for residue in chain
        if int(getattr(residue.seqid, "num", 0) or 0) > 0
    ]
    return (max(numbers) + 1) if numbers else 1


def _next_atom_serial(structure: gemmi.Structure) -> int:
    serials = [int(item.atom.serial) for item in _iter_atoms(structure) if int(item.atom.serial) > 0]
    return (max(serials) + 1) if serials else 1


def append_metal_residue(
    structure: gemmi.Structure,
    resolved: ResolvedMetalInsertion,
    *,
    residue_name: str,
    atom_name: str,
) -> ResolvedMetalInsertion:
    model = structure[0]
    chain = _find_or_create_chain(model, resolved.chain or "Z")
    seqid_number = _next_residue_number(chain)

    residue = gemmi.Residue()
    residue.name = residue_name[:3]
    residue.seqid = gemmi.SeqId(seqid_number, " ")
    residue.het_flag = "H"

    atom = gemmi.Atom()
    atom.name = atom_name[:4]
    atom.element = gemmi.Element(resolved.element.upper())
    atom.pos = gemmi.Position(*resolved.coordinates)
    atom.occ = 1.0
    atom.b_iso = 20.0
    atom.serial = _next_atom_serial(structure)
    residue.add_atom(atom)
    chain.add_residue(residue)

    resolved.seqid = str(residue.seqid).strip()
    resolved.chain = str(chain.name or "").strip()
    resolved.residue_name = residue.name.strip()
    return resolved
