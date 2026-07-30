from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from amber_metallo.reporting import write_json
from amber_metallo.ti.analysis import MetalSiteCandidate, _iter_indexed_atoms


STANDARD_STATE_VOLUME_ANGSTROM3 = 1660.539067
GAS_CONSTANT_KCAL_PER_MOL_K = 0.0019872041


@dataclass(slots=True)
class SiteRestraintSetup:
    restraint_file: str
    metal_atom_index: int
    anchor_atom_indices: list[int]
    anchor_labels: list[str]
    target_distance_angstrom: float
    r1_angstrom: float
    r2_angstrom: float
    r3_angstrom: float
    r4_angstrom: float
    force_constant: float
    correction_kcal_mol: float
    scheme_version: str = "flat_bottom_group_v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _group_centroid(positions: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = float(len(positions))
    return (
        sum(position[0] for position in positions) / count,
        sum(position[1] for position in positions) / count,
        sum(position[2] for position in positions) / count,
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _amber_group_restraint_text(
    *,
    metal_atom_index: int,
    anchor_atom_indices: list[int],
    r1: float,
    r2: float,
    r3: float,
    r4: float,
    force_constant: float,
) -> str:
    anchor_text = ", ".join(str(index) for index in anchor_atom_indices)
    return (
        "&rst\n"
        "  iat = -1, -1,\n"
        f"  igr1 = {metal_atom_index},\n"
        f"  igr2 = {anchor_text},\n"
        f"  r1 = {r1:.3f}, r2 = {r2:.3f}, r3 = {r3:.3f}, r4 = {r4:.3f},\n"
        f"  rk2 = {force_constant:.3f}, rk3 = {force_constant:.3f},\n"
        "/\n"
    )


def _flat_bottom_potential(
    distance_angstrom: float,
    *,
    r2: float,
    r3: float,
    force_constant: float,
) -> float:
    if r2 <= distance_angstrom <= r3:
        return 0.0
    if distance_angstrom < r2:
        return force_constant * (distance_angstrom - r2) ** 2
    return force_constant * (distance_angstrom - r3) ** 2


def compute_standard_state_correction(
    *,
    temperature_k: float,
    r2_angstrom: float,
    r3_angstrom: float,
    force_constant: float,
    integration_limit_angstrom: float | None = None,
    step_angstrom: float = 0.01,
) -> float:
    beta = 1.0 / (GAS_CONSTANT_KCAL_PER_MOL_K * temperature_k)
    max_radius = integration_limit_angstrom or max(r3_angstrom + 15.0, 25.0)
    previous_r = 0.0
    previous_value = 0.0
    partition_volume = 0.0
    steps = max(1, int(math.ceil(max_radius / step_angstrom)))
    for index in range(1, steps + 1):
        radius = min(max_radius, index * step_angstrom)
        potential = _flat_bottom_potential(
            radius,
            r2=r2_angstrom,
            r3=r3_angstrom,
            force_constant=force_constant,
        )
        value = 4.0 * math.pi * radius * radius * math.exp(-beta * potential)
        partition_volume += 0.5 * (value + previous_value) * (radius - previous_r)
        previous_r = radius
        previous_value = value
    return -GAS_CONSTANT_KCAL_PER_MOL_K * temperature_k * math.log(partition_volume / STANDARD_STATE_VOLUME_ANGSTROM3)


def build_bound_site_restraint(
    *,
    reference_structure_path: str | Path,
    candidate: MetalSiteCandidate,
    anchor_count: int,
    force_constant: float,
    half_width_angstrom: float,
    temperature_k: float,
    output_path: str | Path,
) -> SiteRestraintSetup:
    anchors = candidate.donors[:anchor_count]
    if not anchors:
        raise ValueError("At least one coordinating donor atom is required to build the bound-site restraint.")

    indexed_atoms = {atom.atom_index: atom for atom in _iter_indexed_atoms(reference_structure_path)}
    metal = indexed_atoms.get(candidate.atom_index)
    if metal is None:
        raise ValueError(f"Metal atom index {candidate.atom_index} is not present in the reference structure.")

    anchor_atoms = []
    anchor_labels: list[str] = []
    for donor in anchors:
        atom = indexed_atoms.get(donor.atom_index)
        if atom is None:
            raise ValueError(f"Anchor atom index {donor.atom_index} is not present in the reference structure.")
        anchor_atoms.append((float(atom.position.x), float(atom.position.y), float(atom.position.z)))
        anchor_labels.append(f"{atom.residue_key}:{atom.atom_name}")

    metal_position = (float(metal.position.x), float(metal.position.y), float(metal.position.z))
    centroid = _group_centroid(anchor_atoms)
    target_distance = _distance(metal_position, centroid)
    r2 = max(0.0, target_distance - half_width_angstrom)
    r3 = target_distance + half_width_angstrom
    r1 = max(0.0, r2 - half_width_angstrom)
    r4 = r3 + half_width_angstrom
    correction = compute_standard_state_correction(
        temperature_k=temperature_k,
        r2_angstrom=r2,
        r3_angstrom=r3,
        force_constant=force_constant,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _amber_group_restraint_text(
            metal_atom_index=candidate.atom_index,
            anchor_atom_indices=[item.atom_index for item in anchors],
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            force_constant=force_constant,
        ),
        encoding="utf-8",
    )
    setup = SiteRestraintSetup(
        restraint_file=str(target),
        metal_atom_index=candidate.atom_index,
        anchor_atom_indices=[item.atom_index for item in anchors],
        anchor_labels=anchor_labels,
        target_distance_angstrom=target_distance,
        r1_angstrom=r1,
        r2_angstrom=r2,
        r3_angstrom=r3,
        r4_angstrom=r4,
        force_constant=force_constant,
        correction_kcal_mol=correction,
    )
    write_json(target.with_suffix(".json"), setup.to_dict())
    return setup


def write_qoff_duplicate_bound_site_restraint(
    *,
    setup: SiteRestraintSetup,
    duplicate_metal_atom_index: int,
    output_path: str | Path,
) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    original = _amber_group_restraint_text(
        metal_atom_index=setup.metal_atom_index,
        anchor_atom_indices=setup.anchor_atom_indices,
        r1=setup.r1_angstrom,
        r2=setup.r2_angstrom,
        r3=setup.r3_angstrom,
        r4=setup.r4_angstrom,
        force_constant=setup.force_constant,
    )
    duplicate = _amber_group_restraint_text(
        metal_atom_index=duplicate_metal_atom_index,
        anchor_atom_indices=setup.anchor_atom_indices,
        r1=setup.r1_angstrom,
        r2=setup.r2_angstrom,
        r3=setup.r3_angstrom,
        r4=setup.r4_angstrom,
        force_constant=setup.force_constant,
    )
    target.write_text(original + duplicate, encoding="utf-8")
    return str(target)


def write_combined_bound_site_restraints(
    *,
    setups: list[SiteRestraintSetup],
    output_path: str | Path,
) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for setup in setups:
        chunks.append(
            _amber_group_restraint_text(
                metal_atom_index=setup.metal_atom_index,
                anchor_atom_indices=setup.anchor_atom_indices,
                r1=setup.r1_angstrom,
                r2=setup.r2_angstrom,
                r3=setup.r3_angstrom,
                r4=setup.r4_angstrom,
                force_constant=setup.force_constant,
            )
        )
    target.write_text("".join(chunks), encoding="utf-8")
    return str(target)


def write_qoff_duplicate_bound_site_restraints(
    *,
    setups: list[SiteRestraintSetup],
    duplicate_metal_atom_indices: dict[int, int],
    output_path: str | Path,
) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for setup in setups:
        chunks.append(
            _amber_group_restraint_text(
                metal_atom_index=setup.metal_atom_index,
                anchor_atom_indices=setup.anchor_atom_indices,
                r1=setup.r1_angstrom,
                r2=setup.r2_angstrom,
                r3=setup.r3_angstrom,
                r4=setup.r4_angstrom,
                force_constant=setup.force_constant,
            )
        )
        duplicate_index = duplicate_metal_atom_indices.get(setup.metal_atom_index)
        if duplicate_index is not None:
            chunks.append(
                _amber_group_restraint_text(
                    metal_atom_index=duplicate_index,
                    anchor_atom_indices=setup.anchor_atom_indices,
                    r1=setup.r1_angstrom,
                    r2=setup.r2_angstrom,
                    r3=setup.r3_angstrom,
                    r4=setup.r4_angstrom,
                    force_constant=setup.force_constant,
                )
            )
    target.write_text("".join(chunks), encoding="utf-8")
    return str(target)
