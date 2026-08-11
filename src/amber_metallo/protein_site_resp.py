from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from amber_metallo.config import (
    ProteinSiteRespClusterConfig,
    ProteinSiteRespConfig,
    ProteinSiteRespScope,
    RespApplyMode,
    SlurmConfig,
    SystemConfig,
)
from amber_metallo.protonation import DIRECT_COORDINATION_DONOR_ATOMS
from amber_metallo.qm.nwchem import (
    MoleculeAtom,
    MoleculeData,
    metal_safe_qm_settings,
    render_nwchem_input,
)
from amber_metallo.qm.resp_fit import (
    CONSTRAINED_RESP_SOLVER_VERSION,
    fit_constrained_resp_payload,
    render_constrained_charge_table,
    render_constrained_runtime_resp_fit_script,
)
from amber_metallo.qm.slurm import render_resp_slurm_script, render_tahoma_resp_script
from amber_metallo.reporting import write_json
from amber_metallo.subdirectory_search import search_subdirectories_enabled
from amber_metallo.execution import run_command


AMBER_CHARGE_SCALE = 18.2223
DIRECT_COORDINATION_CUTOFF_ANGSTROM = 3.0
LINK_CAP_MIN_NONBONDED_DISTANCE_ANGSTROM = 1.50
_LINK_CAP_DIRECTION_SAMPLES = 512
SUPPORTED_TARGET_RESIDUES = {
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "CYS",
    "CYM",
    "ASP",
    "ASH",
    "GLU",
    "GLH",
    "MET",
}
WATER_RESIDUES = {"HOH", "WAT", "OPC", "SPC", "SPCE", "TIP3", "TIP3P"}
BACKBONE_ATOMS = {
    "N",
    "H",
    "H1",
    "H2",
    "H3",
    "HN",
    "CA",
    "HA",
    "HA2",
    "HA3",
    "C",
    "O",
    "OXT",
}
ENVIRONMENT_DONOR_ELEMENTS = {"N", "O", "S"}
METAL_ELEMENTS = {
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
}
STANDARD_PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH",
    "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "NALA", "CALA", "NGLY", "CGLY",
}


@dataclass(slots=True)
class TopologyAtom:
    topology_index: int
    name: str
    element: str
    residue_index: int
    residue_name: str
    chain: str
    seqid: str
    x: float
    y: float
    z: float
    charge: float

    @property
    def residue_key(self) -> str:
        return f"{self.chain}:{self.residue_name}:{self.seqid}"


@dataclass(slots=True)
class ProteinSiteRespResumeCandidate:
    """A completed site-RESP job that can patch an existing SIMPLE system."""

    job_dir: Path
    manifest_path: Path
    workflow_root: Path
    payload: dict[str, object]
    result_kind: str

    @property
    def source_label(self) -> str:
        return str(self.payload.get("source_label") or self.workflow_root.name)

    @property
    def description(self) -> str:
        return str(self.payload.get("description") or self.job_dir.name)


@dataclass(slots=True)
class SiteBinding:
    metal_site: int
    metal_atom_index: int
    residue_index: int
    residue_key: str
    donor_atom_indices: list[int]
    donor_atom_names: list[str]
    distances_angstrom: list[float]
    target_residue: bool


@dataclass(slots=True)
class SiteCluster:
    metal_sites: list[int]
    metal_atom_indices: list[int]
    donor_residue_indices: list[int]
    donor_residue_keys: list[str]
    fixed_environment_indices: list[int]
    fixed_environment_keys: list[str]
    bindings: list[SiteBinding]
    multiplicity: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metal_sites": self.metal_sites,
            "metal_atom_indices": self.metal_atom_indices,
            "donor_residue_indices": self.donor_residue_indices,
            "donor_residue_keys": self.donor_residue_keys,
            "fixed_environment_indices": self.fixed_environment_indices,
            "fixed_environment_keys": self.fixed_environment_keys,
            "bindings": [asdict(item) for item in self.bindings],
            "multiplicity": self.multiplicity,
        }


def suggested_spin_multiplicity(element: str, charge: int) -> int | None:
    """Return the existing default/high-spin heuristic multiplicity when available."""
    return {
        ("Sc", 3): 1,
        ("Mn", 2): 6,
        ("Fe", 2): 5,
        ("Fe", 3): 6,
        ("Co", 2): 4,
        ("Co", 3): 5,
        ("Ni", 2): 3,
        ("Cu", 1): 1,
        ("Cu", 2): 2,
        ("Y", 3): 1,
        ("La", 3): 1,
        ("Ce", 3): 2,
        ("Pr", 3): 3,
        ("Nd", 3): 4,
        ("Pm", 3): 5,
        ("Sm", 3): 6,
        ("Eu", 3): 7,
        ("Gd", 3): 8,
        ("Tb", 3): 7,
        ("Dy", 3): 6,
        ("Ho", 3): 5,
        ("Er", 3): 4,
        ("Tm", 3): 3,
        ("Yb", 3): 2,
        ("Lu", 3): 1,
    }.get((element.title(), int(charge)))


def suggested_low_spin_multiplicity(element: str, charge: int) -> int | None:
    """Return a common low-spin alternative for ligand-field-sensitive ions.

    Lanthanide 4f configurations generally are not described by the same
    high-spin/low-spin ligand-field choice, so no alternative is suggested for
    those open-shell ions.  The user must still confirm the electronic state.
    """

    return {
        ("Sc", 3): 1,
        ("Mn", 2): 2,
        ("Fe", 2): 1,
        ("Fe", 3): 2,
        ("Co", 2): 2,
        ("Co", 3): 1,
        ("Ni", 2): 1,
        ("Cu", 1): 1,
        ("Cu", 2): 2,
        ("Y", 3): 1,
        ("La", 3): 1,
        ("Lu", 3): 1,
    }.get((element.title(), int(charge)))


def _parse_prmtop_format(format_line: str) -> tuple[int, str, int]:
    body = format_line.strip().removeprefix("%FORMAT(").removesuffix(")")
    if "a" in body.lower():
        count, width = body.lower().split("a", maxsplit=1)
        return int(count), "a", int(width)
    token = body.lower().split("e", maxsplit=1)[-1] if "e" in body.lower() else body.lower().split("i", maxsplit=1)[-1]
    count_text = body.lower().split("e", maxsplit=1)[0] if "e" in body.lower() else body.lower().split("i", maxsplit=1)[0]
    width = token.split(".", maxsplit=1)[0]
    return int(count_text), "e" if "e" in body.lower() else "i", int(width)


def _parse_prmtop_sections(path: Path) -> dict[str, tuple[str, list[str]]]:
    sections: dict[str, tuple[str, list[str]]] = {}
    current: str | None = None
    format_line = ""
    data: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("%FLAG"):
            if current is not None:
                sections[current] = (format_line, data)
            current = line.split(maxsplit=1)[1].strip()
            format_line = ""
            data = []
        elif current is not None and line.startswith("%FORMAT"):
            format_line = line
        elif current is not None and line.startswith("%COMMENT"):
            # ParmEd's add12_6_4 output may place descriptive metadata inside
            # LENNARD_JONES_CCOEF (and other sections).  Amber treats these as
            # comments, not fixed-width section values.
            continue
        elif current is not None:
            data.append(line)
    if current is not None:
        sections[current] = (format_line, data)
    return sections


def _section_values(sections: dict[str, tuple[str, list[str]]], name: str) -> list[str]:
    if name not in sections:
        return []
    format_line, lines = sections[name]
    _, kind, width = _parse_prmtop_format(format_line)
    values: list[str] = []
    for line in lines:
        for start in range(0, len(line), width):
            value = line[start : start + width].strip()
            if value:
                values.append(value if kind == "a" else value.replace("D", "E"))
    return values


def _guess_element(atom_name: str, residue_name: str, pdb_element: str = "") -> str:
    # TLeap/Amber ion residue labels commonly carry the oxidation state
    # (FE2, FE2+, CO2, and so on), while some PDB writers leave the element
    # column blank or truncate Fe to F.  Prefer an unambiguous supported-metal
    # match from either field before falling back to ordinary atom-name rules.
    pdb_token = pdb_element.strip()
    if pdb_token:
        pdb_symbol = pdb_token[0].upper() + pdb_token[1:].lower()
        if pdb_symbol in METAL_ELEMENTS:
            return pdb_symbol
    residue_letters = "".join(character for character in residue_name if character.isalpha())
    residue_symbol = residue_letters[0].upper() + residue_letters[1:].lower() if residue_letters else ""
    if residue_symbol in METAL_ELEMENTS:
        return residue_symbol
    if pdb_token:
        return pdb_token[0].upper() + pdb_token[1:].lower()
    cleaned = "".join(character for character in atom_name if character.isalpha())
    if not cleaned:
        return "C"
    return cleaned[0].upper()


def _pdb_atom_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith(("ATOM  ", "HETATM")):
            continue
        padded = raw_line.ljust(80)
        rows.append(
            {
                "name": padded[12:16].strip(),
                "residue_name": padded[17:20].strip(),
                "chain": padded[21:22].strip() or "_",
                "seqid": (padded[22:27].strip() or str(len(rows) + 1)),
                "x": float(padded[30:38]),
                "y": float(padded[38:46]),
                "z": float(padded[46:54]),
                "element": padded[76:78].strip(),
            }
        )
    return rows


def validate_retained_direct_environment(
    *,
    source_pdb: str | Path,
    reference_pdb: str | Path,
    cutoff_angstrom: float = DIRECT_COORDINATION_CUTOFF_ANGSTROM,
) -> list[str]:
    """Block RESP when source water/heteroligand donors were removed before TLeap."""
    source_rows = _pdb_atom_rows(Path(source_pdb))
    reference_rows = _pdb_atom_rows(Path(reference_pdb))

    def point(row: dict[str, object]) -> tuple[float, float, float]:
        return float(row["x"]), float(row["y"]), float(row["z"])

    def row_distance(first: dict[str, object], second: dict[str, object]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point(first), point(second), strict=True)))

    source_metals = [
        row for row in source_rows
        if _guess_element(str(row["name"]), str(row["residue_name"]), str(row["element"])).title() in METAL_ELEMENTS
    ]
    reference_metals = [
        row for row in reference_rows
        if _guess_element(str(row["name"]), str(row["residue_name"]), str(row["element"])).title() in METAL_ELEMENTS
    ]
    retained_source_metals = [
        metal for metal in source_metals
        if any(row_distance(metal, retained) <= 0.75 for retained in reference_metals)
    ]
    missing: set[str] = set()
    for row in source_rows:
        residue_name = str(row["residue_name"]).strip().upper()
        if residue_name in STANDARD_PROTEIN_RESIDUES or residue_name.title() in METAL_ELEMENTS:
            continue
        element = _guess_element(str(row["name"]), residue_name, str(row["element"])).upper()
        if element not in ENVIRONMENT_DONOR_ELEMENTS:
            continue
        if not any(row_distance(row, metal) <= cutoff_angstrom for metal in retained_source_metals):
            continue
        retained = any(
            _guess_element(
                str(candidate["name"]),
                str(candidate["residue_name"]),
                str(candidate["element"]),
            ).upper()
            == element
            and row_distance(row, candidate) <= 0.25
            for candidate in reference_rows
        )
        if not retained:
            missing.add(f"{row['chain']}:{residue_name}:{row['seqid']}@{row['name']}")
    if missing:
        raise ValueError(
            "Protein-site RESP requires every directly coordinating water or non-standard ligand/cofactor to remain "
            "in the parameterized, unsolvated reference topology. The following source donor atoms were removed or "
            "could not be mapped: " + ", ".join(sorted(missing))
        )
    return sorted(missing)


def load_topology_atoms(prmtop_path: str | Path, pdb_path: str | Path) -> list[TopologyAtom]:
    prmtop = Path(prmtop_path)
    pdb = Path(pdb_path)
    sections = _parse_prmtop_sections(prmtop)
    atom_names = _section_values(sections, "ATOM_NAME")
    charges = [float(value) / AMBER_CHARGE_SCALE for value in _section_values(sections, "CHARGE")]
    residue_labels = _section_values(sections, "RESIDUE_LABEL")
    residue_pointers = [int(value) for value in _section_values(sections, "RESIDUE_POINTER")]
    pdb_rows = _pdb_atom_rows(pdb)
    if not atom_names or len(charges) != len(atom_names):
        raise ValueError(f"Could not read atom names and charges from Amber topology: {prmtop}")
    if len(pdb_rows) != len(atom_names):
        raise ValueError(
            f"System PDB contains {len(pdb_rows)} atoms but topology contains {len(atom_names)}; "
            "protein-site RESP requires an exact coordinate/topology atom mapping."
        )
    residue_ends = [pointer - 1 for pointer in residue_pointers[1:]] + [len(atom_names)]
    atoms: list[TopologyAtom] = []
    residue_index = 1
    for atom_offset, (atom_name, charge) in enumerate(zip(atom_names, charges, strict=True)):
        atom_number = atom_offset + 1
        while residue_index <= len(residue_ends) and atom_number > residue_ends[residue_index - 1]:
            residue_index += 1
        row = pdb_rows[atom_offset]
        pdb_name = str(row["name"]).replace("*", "'").upper()
        topology_name = atom_name.replace("*", "'").upper()
        if pdb_name != topology_name:
            raise ValueError(
                f"System PDB/topology atom order mismatch at atom {atom_number}: "
                f"PDB={row['name']}, topology={atom_name}."
            )
        residue_name = residue_labels[residue_index - 1].strip().upper()
        atoms.append(
            TopologyAtom(
                topology_index=atom_number,
                name=atom_name.strip(),
                element=_guess_element(atom_name, residue_name, str(row["element"])),
                residue_index=residue_index,
                residue_name=residue_name,
                chain=str(row["chain"]),
                seqid=str(row["seqid"]),
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                charge=float(charge),
            )
        )
    return atoms


def _distance(first: TopologyAtom, second: TopologyAtom) -> float:
    return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2)


def _family(residue_name: str) -> str:
    name = residue_name.upper()
    if name in {"HIS", "HID", "HIE", "HIP"}:
        return "HIS"
    if name in {"CYS", "CYM"}:
        return "CYS"
    if name in {"ASP", "ASH"}:
        return "ASP"
    if name in {"GLU", "GLH"}:
        return "GLU"
    return name


def _residue_map(atoms: Iterable[TopologyAtom]) -> dict[int, list[TopologyAtom]]:
    result: dict[int, list[TopologyAtom]] = {}
    for atom in atoms:
        result.setdefault(atom.residue_index, []).append(atom)
    return result


def _cluster_config_for_sites(
    config: ProteinSiteRespConfig,
    metal_sites: list[int],
) -> ProteinSiteRespClusterConfig | None:
    site_set = set(metal_sites)
    exact = [item for item in config.clusters if set(item.metal_sites) == site_set]
    if len(exact) > 1:
        raise ValueError(f"Multiple protein_site_resp cluster entries target metal sites {metal_sites}.")
    return exact[0] if exact else None


def discover_site_clusters(
    *,
    atoms: list[TopologyAtom],
    system_config: SystemConfig,
    resp_config: ProteinSiteRespConfig,
    cutoff_angstrom: float = DIRECT_COORDINATION_CUTOFF_ANGSTROM,
) -> list[SiteCluster]:
    residues = _residue_map(atoms)
    metal_atoms = [atom for atom in atoms if atom.element.title() in METAL_ELEMENTS and len(residues[atom.residue_index]) == 1]
    if not metal_atoms:
        monatomic_labels = sorted(
            {
                residues[index][0].residue_name
                for index in residues
                if len(residues[index]) == 1
            }
        )
        label_text = ", ".join(monatomic_labels) if monatomic_labels else "none"
        raise ValueError(
            "No supported metal atom could be identified in the unsolvated reference topology. "
            f"Monatomic residue labels found: {label_text}."
        )
    charge_by_site = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    bindings: list[SiteBinding] = []
    metal_by_site: dict[int, TopologyAtom] = {}
    for site, metal in enumerate(metal_atoms, start=1):
        metal_by_site[site] = metal
        for residue_index, residue_atoms in residues.items():
            if residue_index == metal.residue_index:
                continue
            residue_name = residue_atoms[0].residue_name.upper()
            target = residue_name in SUPPORTED_TARGET_RESIDUES
            if target:
                donor_names = set(DIRECT_COORDINATION_DONOR_ATOMS.get(_family(residue_name), ()))
                candidates = [atom for atom in residue_atoms if atom.name.upper() in donor_names]
            else:
                candidates = [atom for atom in residue_atoms if atom.element.upper() in ENVIRONMENT_DONOR_ELEMENTS]
            direct = sorted(
                [(atom, _distance(metal, atom)) for atom in candidates if _distance(metal, atom) <= cutoff_angstrom],
                key=lambda item: item[1],
            )
            if not direct:
                continue
            bindings.append(
                SiteBinding(
                    metal_site=site,
                    metal_atom_index=metal.topology_index,
                    residue_index=residue_index,
                    residue_key=residue_atoms[0].residue_key,
                    donor_atom_indices=[item[0].topology_index for item in direct],
                    donor_atom_names=[item[0].name for item in direct],
                    distances_angstrom=[item[1] for item in direct],
                    target_residue=target,
                )
            )

    adjacency: dict[int, set[int]] = {site: set() for site in metal_by_site}
    sites_by_residue: dict[int, set[int]] = {}
    for binding in bindings:
        sites_by_residue.setdefault(binding.residue_index, set()).add(binding.metal_site)
    for site_group in sites_by_residue.values():
        for site in site_group:
            adjacency[site].update(site_group - {site})

    components: list[list[int]] = []
    visited: set[int] = set()
    for site in sorted(adjacency):
        if site in visited:
            continue
        stack = [site]
        component: list[int] = []
        visited.add(site)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    clusters: list[SiteCluster] = []
    for sites in components:
        component_bindings = [binding for binding in bindings if binding.metal_site in sites]
        configured = _cluster_config_for_sites(resp_config, sites)
        target_bindings = [binding for binding in component_bindings if binding.target_residue]
        fixed_bindings = [binding for binding in component_bindings if not binding.target_residue]
        if configured and configured.donor_residues:
            requested = set(configured.donor_residues)
            target_bindings = [binding for binding in target_bindings if binding.residue_key in requested]
            missing = requested - {binding.residue_key for binding in target_bindings}
            if missing:
                raise ValueError(
                    "Configured protein-site RESP donor residues were not detected as direct donors: "
                    + ", ".join(sorted(missing))
                )
        fixed_indices = {binding.residue_index for binding in fixed_bindings}
        if configured:
            requested_fixed = set(configured.fixed_environment)
            matched_fixed = {
                residue_index
                for residue_index, residue_atoms in residues.items()
                if residue_atoms[0].residue_key in requested_fixed
            }
            missing_fixed = requested_fixed - {
                residues[residue_index][0].residue_key for residue_index in matched_fixed
            }
            if missing_fixed:
                raise ValueError(
                    "Configured protein-site RESP fixed-environment residues were not found in the reference topology: "
                    + ", ".join(sorted(missing_fixed))
                )
            fixed_indices.update(matched_fixed)
        if not target_bindings:
            continue
        multiplicity = (
            configured.multiplicity
            if configured is not None and configured.multiplicity is not None
            else resp_config.default_multiplicity
        )
        if multiplicity is None:
            species = [
                f"{metal_by_site[site].element}{charge_by_site.get(site, '?')}+"
                for site in sites
            ]
            raise ValueError(
                "Protein-site RESP requires a user-confirmed spin multiplicity for each connected metal cluster. "
                f"Missing multiplicity for sites {sites} ({', '.join(species)})."
            )
        donor_indices = sorted({binding.residue_index for binding in target_bindings})
        clusters.append(
            SiteCluster(
                metal_sites=sites,
                metal_atom_indices=[metal_by_site[site].topology_index for site in sites],
                donor_residue_indices=donor_indices,
                donor_residue_keys=[residues[index][0].residue_key for index in donor_indices],
                fixed_environment_indices=sorted(fixed_indices),
                fixed_environment_keys=[residues[index][0].residue_key for index in sorted(fixed_indices)],
                bindings=component_bindings,
                multiplicity=int(multiplicity),
            )
        )
    if not clusters:
        nearest: list[str] = []
        for site, metal in metal_by_site.items():
            candidates: list[tuple[float, TopologyAtom]] = []
            for residue_atoms in residues.values():
                residue_name = residue_atoms[0].residue_name.upper()
                if residue_name not in SUPPORTED_TARGET_RESIDUES:
                    continue
                donor_names = set(DIRECT_COORDINATION_DONOR_ATOMS.get(_family(residue_name), ()))
                candidates.extend(
                    (_distance(metal, atom), atom)
                    for atom in residue_atoms
                    if atom.name.upper() in donor_names
                )
            if candidates:
                distance, donor = min(candidates, key=lambda item: item[0])
                nearest.append(
                    f"site {site} {metal.element}: {donor.residue_key}@{donor.name} = {distance:.2f} A"
                )
            else:
                nearest.append(f"site {site} {metal.element}: no supported donor residue atoms present")
        raise ValueError(
            "Supported metal atoms were identified, but no directly coordinating HIS/HID/HIE/HIP, "
            "CYS/CYM, ASP/ASH, GLU/GLH, or MET side chain was found within "
            f"{cutoff_angstrom:.2f} A. Nearest supported donor by site: " + "; ".join(nearest)
        )
    return clusters


def _unit_vector_toward(first: TopologyAtom, second: TopologyAtom) -> tuple[float, float, float]:
    """Return the direction from ``first`` toward ``second``."""

    dx, dy, dz = second.x - first.x, second.y - first.y, second.z - first.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1.0e-12:
        return (1.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def _link_cap_direction_candidates(
    preferred: tuple[float, float, float],
) -> Iterable[tuple[float, float, float]]:
    """Yield the cut-bond direction first, followed by deterministic alternatives."""

    yield preferred
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(_LINK_CAP_DIRECTION_SAMPLES):
        z = 1.0 - 2.0 * (index + 0.5) / _LINK_CAP_DIRECTION_SAMPLES
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        azimuth = golden_angle * index
        yield (radial * math.cos(azimuth), radial * math.sin(azimuth), z)


def _place_link_cap(
    *,
    name: str,
    anchor: TopologyAtom,
    removed_neighbor: TopologyAtom,
    bond_length: float,
    obstacles: list[tuple[str, str, float, float, float]],
) -> tuple[float, float, float]:
    """Place a hydrogen cap toward the removed atom without a nonbonded clash."""

    preferred = _unit_vector_toward(anchor, removed_neighbor)
    anchor_position = (anchor.x, anchor.y, anchor.z)
    nonbonded: list[tuple[str, str, float, float, float]] = []
    skipped_anchor = False
    for obstacle in obstacles:
        distance_to_anchor = math.dist(anchor_position, obstacle[2:5])
        if not skipped_anchor and distance_to_anchor <= 1.0e-6:
            skipped_anchor = True
            continue
        nonbonded.append(obstacle)

    valid: list[tuple[float, float, tuple[float, float, float]]] = []
    best_clearance = 0.0
    for direction in _link_cap_direction_candidates(preferred):
        candidate = tuple(
            coordinate + bond_length * component
            for coordinate, component in zip(anchor_position, direction, strict=True)
        )
        clearance = min(
            (math.dist(candidate, obstacle[2:5]) for obstacle in nonbonded),
            default=math.inf,
        )
        best_clearance = max(best_clearance, clearance)
        if clearance + 1.0e-9 < LINK_CAP_MIN_NONBONDED_DISTANCE_ANGSTROM:
            continue
        alignment = sum(
            component * preferred_component
            for component, preferred_component in zip(direction, preferred, strict=True)
        )
        # Preserve the severed-bond direction whenever possible.  Clearance
        # breaks ties and keeps a displaced cap away from the acceptance edge.
        valid.append((alignment, clearance, candidate))
    if not valid:
        raise ValueError(
            f"Could not place protein-site RESP link cap {name} without a nonbonded contact shorter than "
            f"{LINK_CAP_MIN_NONBONDED_DISTANCE_ANGSTROM:.2f} A (best available clearance "
            f"{best_clearance:.2f} A). Check the prepared structure for atomic overlaps."
        )
    return max(valid, key=lambda item: (item[0], item[1]))[2]


def _link_caps(
    *,
    selected_residue_indices: set[int],
    retained_residue_indices: set[int],
    residues: dict[int, list[TopologyAtom]],
    retained_atoms: Iterable[TopologyAtom | MoleculeAtom],
) -> list[dict[str, object]]:
    caps: list[dict[str, object]] = []
    obstacles = [
        (atom.name, atom.element, float(atom.x), float(atom.y), float(atom.z))
        for atom in retained_atoms
    ]
    ordered_indices = sorted(residues)
    position = {residue_index: index for index, residue_index in enumerate(ordered_indices)}
    for residue_index in sorted(selected_residue_indices):
        residue_atoms = residues[residue_index]
        chain = residue_atoms[0].chain
        atom_by_name = {atom.name.upper(): atom for atom in residue_atoms}
        list_position = position[residue_index]
        previous_index = ordered_indices[list_position - 1] if list_position > 0 else None
        next_index = ordered_indices[list_position + 1] if list_position + 1 < len(ordered_indices) else None
        if previous_index not in retained_residue_indices and previous_index is not None:
            previous_atoms = residues[previous_index]
            if previous_atoms[0].chain == chain and "N" in atom_by_name:
                previous_c = next((atom for atom in previous_atoms if atom.name.upper() == "C"), None)
                if previous_c is not None:
                    anchor = atom_by_name["N"]
                    name = f"HN{residue_index}"
                    x, y, z = _place_link_cap(
                        name=name,
                        anchor=anchor,
                        removed_neighbor=previous_c,
                        bond_length=1.01,
                        obstacles=obstacles,
                    )
                    cap = {
                        "name": name,
                        "element": "H",
                        "x": x,
                        "y": y,
                        "z": z,
                        "residue_key": residue_atoms[0].residue_key,
                        "role": "link_cap",
                    }
                    caps.append(cap)
                    obstacles.append((name, "H", x, y, z))
        if next_index not in retained_residue_indices and next_index is not None:
            next_atoms = residues[next_index]
            if next_atoms[0].chain == chain and "C" in atom_by_name:
                next_n = next((atom for atom in next_atoms if atom.name.upper() == "N"), None)
                if next_n is not None:
                    anchor = atom_by_name["C"]
                    name = f"HC{residue_index}"
                    x, y, z = _place_link_cap(
                        name=name,
                        anchor=anchor,
                        removed_neighbor=next_n,
                        bond_length=1.09,
                        obstacles=obstacles,
                    )
                    cap = {
                        "name": name,
                        "element": "H",
                        "x": x,
                        "y": y,
                        "z": z,
                        "residue_key": residue_atoms[0].residue_key,
                        "role": "link_cap",
                    }
                    caps.append(cap)
                    obstacles.append((name, "H", x, y, z))
    return caps


def _safe_equality_pairs(
    *,
    metadata: list[dict[str, object]],
    cluster: SiteCluster,
) -> list[tuple[int, int]]:
    direct_topology_indices = {
        index
        for binding in cluster.bindings
        if binding.target_residue
        for index in binding.donor_atom_indices
    }
    by_residue: dict[str, list[int]] = {}
    for atom_index, item in enumerate(metadata):
        residue_key = str(item.get("residue_key") or "")
        if item.get("role") != "target" or not residue_key:
            continue
        by_residue.setdefault(residue_key, []).append(atom_index)
    pairs: list[tuple[int, int]] = []
    for atom_indices in by_residue.values():
        rows = [metadata[index] for index in atom_indices]
        by_hydrogen_prefix: dict[str, list[int]] = {}
        for index, row in zip(atom_indices, rows, strict=True):
            name = str(row.get("atom_name") or "").upper()
            if not name.startswith("H") or not name[-1:].isdigit():
                continue
            prefix = name.rstrip("1234567890")
            by_hydrogen_prefix.setdefault(prefix, []).append(index)
        for group in by_hydrogen_prefix.values():
            charges = {round(float(metadata[index].get("original_charge") or 0.0), 6) for index in group}
            if len(group) > 1 and len(charges) == 1:
                pairs.extend((group[0], index) for index in group[1:])

        residue_name = str(rows[0].get("residue_name") or "").upper()
        symmetric_names = ("OD1", "OD2") if _family(residue_name) == "ASP" else (("OE1", "OE2") if _family(residue_name) == "GLU" else ())
        if symmetric_names:
            symmetric_indices = {
                str(metadata[index].get("atom_name") or "").upper(): index
                for index in atom_indices
                if str(metadata[index].get("atom_name") or "").upper() in symmetric_names
            }
            if all(name in symmetric_indices for name in symmetric_names):
                first, second = (symmetric_indices[name] for name in symmetric_names)
                first_top = int(metadata[first].get("topology_index") or 0)
                second_top = int(metadata[second].get("topology_index") or 0)
                first_direct = first_top in direct_topology_indices
                second_direct = second_top in direct_topology_indices
                distance_by_topology = {
                    atom_index: distance
                    for binding in cluster.bindings
                    for atom_index, distance in zip(
                        binding.donor_atom_indices,
                        binding.distances_angstrom,
                        strict=True,
                    )
                }
                distance_match = abs(distance_by_topology.get(first_top, 999.0) - distance_by_topology.get(second_top, 999.0)) <= 0.10
                if first_direct == second_direct and (not first_direct or distance_match):
                    pairs.append((first, second))
    return sorted(set(tuple(sorted(pair)) for pair in pairs))


def _cluster_payload(
    *,
    atoms: list[TopologyAtom],
    cluster: SiteCluster,
    scope: ProteinSiteRespScope,
    system_config: SystemConfig,
    source_label: str,
) -> dict[str, object]:
    residues = _residue_map(atoms)
    selected_residues = set(cluster.donor_residue_indices)
    included_residues = selected_residues | set(cluster.fixed_environment_indices)
    included_atoms = [
        atom
        for atom in atoms
        if atom.residue_index in included_residues or atom.topology_index in set(cluster.metal_atom_indices)
    ]
    included_atoms.sort(key=lambda atom: atom.topology_index)
    metal_charge_by_site = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    metal_charge_by_topology = {
        atom_index: float(metal_charge_by_site[site])
        for site, atom_index in zip(cluster.metal_sites, cluster.metal_atom_indices, strict=True)
        if site in metal_charge_by_site
    }
    if len(metal_charge_by_topology) != len(cluster.metal_atom_indices):
        raise ValueError("Every protein-site RESP metal requires an explicit oxidation-state assignment.")

    molecule_atoms: list[MoleculeAtom] = []
    metadata: list[dict[str, object]] = []
    fixed_charges: dict[int, float] = {}
    sum_constraints: list[dict[str, object]] = []
    target_cluster_indices_by_residue: dict[int, list[int]] = {}
    for atom in included_atoms:
        cluster_index = len(molecule_atoms)
        is_metal = atom.topology_index in metal_charge_by_topology
        is_target = atom.residue_index in selected_residues
        apply_charge = is_target and (
            scope == ProteinSiteRespScope.WHOLE_RESIDUE or atom.name.upper() not in BACKBONE_ATOMS
        )
        role = "metal" if is_metal else ("target" if is_target else "fixed_environment")
        original_charge = metal_charge_by_topology.get(atom.topology_index, atom.charge)
        molecule_atoms.append(
            MoleculeAtom(
                index=cluster_index + 1,
                name=atom.name,
                element=atom.element,
                x=atom.x,
                y=atom.y,
                z=atom.z,
                charge=original_charge,
            )
        )
        metadata.append(
            {
                "topology_index": atom.topology_index,
                "atom_name": atom.name,
                "element": atom.element,
                "residue_index": atom.residue_index,
                "residue_name": atom.residue_name,
                "residue_key": atom.residue_key,
                "original_charge": original_charge,
                "apply": apply_charge,
                "role": role,
            }
        )
        if is_target:
            target_cluster_indices_by_residue.setdefault(atom.residue_index, []).append(cluster_index)
        if not apply_charge:
            fixed_charges[cluster_index] = original_charge

    for cap in _link_caps(
        selected_residue_indices=selected_residues,
        retained_residue_indices=included_residues,
        residues=residues,
        retained_atoms=molecule_atoms,
    ):
        cluster_index = len(molecule_atoms)
        molecule_atoms.append(
            MoleculeAtom(
                index=cluster_index + 1,
                name=str(cap["name"]),
                element="H",
                x=float(cap["x"]),
                y=float(cap["y"]),
                z=float(cap["z"]),
                charge=0.0,
            )
        )
        metadata.append(
            {
                "topology_index": None,
                "atom_name": cap["name"],
                "element": "H",
                "residue_index": None,
                "residue_name": "CAP",
                "residue_key": cap["residue_key"],
                "original_charge": 0.0,
                "apply": False,
                "role": "link_cap",
            }
        )
        fixed_charges[cluster_index] = 0.0

    for residue_index, cluster_indices in sorted(target_cluster_indices_by_residue.items()):
        residue_charge = sum(atom.charge for atom in residues[residue_index])
        sum_constraints.append(
            {
                "label": f"residue_total:{residues[residue_index][0].residue_key}",
                "atom_indices": cluster_indices,
                "charge": residue_charge,
            }
        )

    raw_total = sum(float(item.get("original_charge") or 0.0) for item in metadata)
    net_charge = int(round(raw_total))
    if abs(raw_total - net_charge) > 0.05:
        raise ValueError(
            f"Protein-site QM cluster has non-integral reference charge {raw_total:.6f}; "
            "check retained environment residue parameters and protonation states."
        )
    equality_pairs = _safe_equality_pairs(metadata=metadata, cluster=cluster)
    atom_by_topology_index = {atom.topology_index: atom for atom in atoms}
    formal_metal_states = [
        {
            "site": site,
            "element": atom_by_topology_index[topology_index].element,
            "formal_charge": int(metal_charge_by_site[site]),
            "topology_index": topology_index,
        }
        for site, topology_index in zip(cluster.metal_sites, cluster.metal_atom_indices, strict=True)
    ]
    identity = {
        "schema_version": 1,
        "source_label": source_label,
        "scope": scope.value,
        "protein_ff": system_config.protein_ff,
        "metal_sites": cluster.metal_sites,
        "formal_metal_states": formal_metal_states,
        "multiplicity": cluster.multiplicity,
        "donor_residues": cluster.donor_residue_keys,
        "fixed_environment": cluster.fixed_environment_keys,
        "atoms": [
            {
                "topology_index": item.get("topology_index"),
                "atom_name": item.get("atom_name"),
                "residue_key": item.get("residue_key"),
                "original_charge": round(float(item.get("original_charge") or 0.0), 8),
                "coordinates": [
                    round(molecule_atoms[index].x, 5),
                    round(molecule_atoms[index].y, 5),
                    round(molecule_atoms[index].z, 5),
                ],
            }
            for index, item in enumerate(metadata)
        ],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "molecule": MoleculeData(source_file=source_label, source_format="pdb", atoms=molecule_atoms, bonds=[]),
        "metadata": metadata,
        "fixed_charges": fixed_charges,
        "sum_constraints": sum_constraints,
        "equality_pairs": equality_pairs,
        "net_charge": net_charge,
        "identity": identity,
        "fingerprint": fingerprint,
    }


def _next_job_dir(base_dir: Path, label: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = base_dir / f"SITE_RESP_JOBS_{index:03d}_{label}"
        if not candidate.exists():
            return candidate
        index += 1


def _site_job_candidates(search_roots: Iterable[Path], fingerprint: str) -> list[Path]:
    matches: list[tuple[str, Path]] = []
    checked: set[Path] = set()
    for root in search_roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            continue
        manifest_paths = (
            resolved.rglob("site_resp_manifest.json")
            if search_subdirectories_enabled()
            else (
                path
                for path in (
                    resolved / "site_resp_manifest.json",
                    resolved / "manifests" / "site_resp_manifest.json",
                )
                if path.is_file()
            )
        )
        for manifest_path in manifest_paths:
            job_dir = manifest_path.parent.parent
            if job_dir in checked:
                continue
            checked.add(job_dir)
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("fingerprint") == fingerprint:
                matches.append((str(payload.get("created_at") or ""), job_dir))
    return [item[1] for item in sorted(matches, reverse=True)]


def _job_manifest_matches(job_dir: Path, fingerprint: str) -> bool:
    manifest_path = job_dir / "manifests" / "site_resp_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("fingerprint") == fingerprint


def site_resp_result_path(job_dir: str | Path) -> Path | None:
    root = Path(job_dir)
    candidate = root / "output" / "site_resp_charges.json"
    if candidate.exists():
        return candidate
    return None


def _site_resp_grid_available(job_dir: str | Path) -> bool:
    root = Path(job_dir)
    output_dir = root / "output"
    has_grid = any(output_dir.glob("*.grid"))
    has_xyz = (
        (output_dir / "resp_job.xyz").exists()
        or (root / "inputs" / "resp_job.xyz").exists()
        or any(output_dir.glob("*.xyz"))
    )
    return has_grid and has_xyz


def _site_resp_workflow_system_available(workflow_root: Path) -> bool:
    """Return whether a workflow has the files needed for a safe RESP resume.

    Older workflows may already have their final solvated topology, while the
    current deferred-solvation workflow intentionally stops with only an
    unsolvated site-RESP reference topology.
    """

    system_dir = workflow_root / "02_system"
    final_system = (system_dir / "system.prmtop").exists() and (
        system_dir / "system.inpcrd"
    ).exists()
    reference_system = all(
        path.exists()
        for path in (
            system_dir / "site_resp_reference_manifest.json",
            system_dir / "system.unsolvated.pdb",
            system_dir / "system.unsolvated.prmtop",
            system_dir / "system.unsolvated.inpcrd",
        )
    )
    return final_system or reference_system


def _site_resp_final_system_available(workflow_root: Path) -> bool:
    system_dir = workflow_root / "02_system"
    return (system_dir / "system.prmtop").exists() and (
        system_dir / "system.inpcrd"
    ).exists()


def _site_resp_workflow_root(job_dir: Path, payload: dict[str, object]) -> Path | None:
    prepared_system = str(payload.get("prepared_system_pdb") or "").strip()
    if prepared_system:
        prepared_path = Path(prepared_system).expanduser()
        if prepared_path.parent.name == "02_system":
            candidate = prepared_path.parent.parent.resolve()
            if _site_resp_workflow_system_available(candidate):
                return candidate

    for candidate in (job_dir, *job_dir.parents):
        if _site_resp_workflow_system_available(candidate):
            return candidate.resolve()
    return None


def _already_applied_site_resp_jobs(workflow_root: Path) -> set[Path]:
    application_path = workflow_root / "02_system" / "site_resp_application.json"
    if not application_path.exists():
        return set()
    try:
        payload = json.loads(application_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if payload.get("status") != "applied":
        return set()
    applied: set[Path] = set()
    for item in payload.get("job_dirs") or []:
        try:
            applied.add(Path(str(item)).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    return applied


def find_protein_site_resp_resume_candidates(
    search_root: str | Path,
    *,
    recursive: bool = True,
) -> list[ProteinSiteRespResumeCandidate]:
    """Find completed site-RESP jobs that can resume their original workflow.

    A result is offered only when its manifest, ESP fit (or refittable grid), and
    original final or unsolvated reference topology/coordinates are all
    available.  This keeps the Protein start screen from offering orphaned or
    already-applied jobs.
    """

    root = Path(search_root).expanduser().resolve()
    if not root.exists():
        return []
    candidates: list[ProteinSiteRespResumeCandidate] = []
    seen: set[Path] = set()
    try:
        manifests = (
            root.rglob("site_resp_manifest.json")
            if recursive
            else (
                path
                for path in (
                    root / "site_resp_manifest.json",
                    root / "manifests" / "site_resp_manifest.json",
                )
                if path.is_file()
            )
        )
        for manifest_path in manifests:
            job_dir = manifest_path.parent.parent.resolve()
            if job_dir in seen:
                continue
            seen.add(job_dir)
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("fingerprint"):
                continue
            workflow_root = _site_resp_workflow_root(job_dir, payload)
            if workflow_root is None:
                continue
            # A final solvated topology means this workflow has already moved
            # beyond the deferred site-RESP stage.  Do not offer it again on
            # the Protein start screen, even if an older run lacks an explicit
            # application marker.
            if _site_resp_final_system_available(workflow_root):
                continue
            if job_dir in _already_applied_site_resp_jobs(workflow_root):
                continue
            result_path = site_resp_result_path(job_dir)
            result_payload: dict[str, object] = {}
            if result_path is not None:
                try:
                    loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_result, dict):
                        result_payload = loaded_result
                except (OSError, json.JSONDecodeError):
                    result_payload = {}
            current_result = (
                result_payload.get("solver_version") == CONSTRAINED_RESP_SOLVER_VERSION
                and bool(result_payload.get("numerically_stable"))
            )
            if current_result:
                result_kind = "fitted charges"
            elif _site_resp_grid_available(job_dir):
                result_kind = "ESP grid (will refit)"
            else:
                continue
            candidates.append(
                ProteinSiteRespResumeCandidate(
                    job_dir=job_dir,
                    manifest_path=manifest_path.resolve(),
                    workflow_root=workflow_root,
                    payload=payload,
                    result_kind=result_kind,
                )
            )
    except OSError:
        return candidates
    candidates.sort(
        key=lambda item: (
            str(item.payload.get("created_at") or ""),
            str(item.workflow_root),
            str(item.job_dir),
        ),
        reverse=True,
    )
    unique: list[ProteinSiteRespResumeCandidate] = []
    seen_fingerprints: set[tuple[Path, str]] = set()
    for candidate in candidates:
        key = (candidate.workflow_root, str(candidate.payload.get("fingerprint") or ""))
        if key in seen_fingerprints:
            continue
        seen_fingerprints.add(key)
        unique.append(candidate)
    return unique


def _materialize_detected_site_resp_result(job_dir: Path) -> Path | None:
    existing = site_resp_result_path(job_dir)
    if existing is not None:
        return existing
    if not _site_resp_grid_available(job_dir):
        return None
    return materialize_site_resp_result(job_dir)


def _render_xyz(molecule: MoleculeData, comment: str) -> str:
    lines = [str(len(molecule.atoms)), comment]
    lines.extend(
        f"{atom.element:<2s} {atom.x: .8f} {atom.y: .8f} {atom.z: .8f}"
        for atom in molecule.atoms
    )
    return "\n".join(lines) + "\n"


def build_site_resp_jobs(
    *,
    system_pdb: str | Path,
    system_prmtop: str | Path,
    system_config: SystemConfig,
    resp_config: ProteinSiteRespConfig,
    slurm_config: SlurmConfig,
    base_dir: str | Path,
    source_label: str,
    source_pdb: str | Path | None = None,
) -> list[dict[str, object]]:
    system_pdb_path = Path(system_pdb).expanduser().resolve()
    system_prmtop_path = Path(system_prmtop).expanduser().resolve()
    source_pdb_path = Path(source_pdb).expanduser().resolve() if source_pdb is not None else None
    if source_pdb_path is not None and source_pdb_path.exists():
        validate_retained_direct_environment(source_pdb=source_pdb_path, reference_pdb=system_pdb_path)
    atoms = load_topology_atoms(system_prmtop_path, system_pdb_path)
    clusters = discover_site_clusters(atoms=atoms, system_config=system_config, resp_config=resp_config)
    if not clusters:
        raise ValueError("No supported directly coordinating protein side chains were found for protein-site RESP.")
    base_path = Path(base_dir).expanduser().resolve()
    # Search the new job folder, user-configured roots, and the launch/output
    # parent automatically. The latter is where incremented interactive runs
    # (3FS9_FE, 3FS9_FE_1, ...) normally sit beside the original RESP job.
    automatic_roots = [base_path]
    if len(base_path.parents) >= 3:
        automatic_roots.append(base_path.parents[2])
    search_roots = list(
        dict.fromkeys(
            path.expanduser().resolve()
            for path in [*automatic_roots, *(Path(item) for item in resp_config.search_roots)]
        )
    )
    results: list[dict[str, object]] = []
    for cluster in clusters:
        payload = _cluster_payload(
            atoms=atoms,
            cluster=cluster,
            scope=resp_config.scope,
            system_config=system_config,
            source_label=source_label,
        )
        fingerprint = str(payload["fingerprint"])
        configured = _cluster_config_for_sites(resp_config, cluster.metal_sites)
        configured_job_dirs = [
            *(Path(item).expanduser().resolve() for item in resp_config.job_dirs),
            *(
                [Path(configured.job_dir).expanduser().resolve()]
                if configured and configured.job_dir
                else []
            ),
        ]
        explicit_matches = [path for path in configured_job_dirs if _job_manifest_matches(path, fingerprint)]
        candidates = _site_job_candidates(search_roots, fingerprint)
        candidates = [*explicit_matches, *[candidate for candidate in candidates if candidate not in explicit_matches]]
        completed: Path | None = None
        completed_result: Path | None = None
        for candidate in candidates:
            detected_result = _materialize_detected_site_resp_result(candidate)
            if detected_result is not None:
                completed = candidate
                completed_result = detected_result
                break
        if resp_config.apply_mode in {RespApplyMode.DETECT, RespApplyMode.APPLY_EXISTING} and completed is not None:
            results.append(
                {
                    "status": "ready_to_apply",
                    "job_dir": str(completed),
                    "fingerprint": fingerprint,
                    "cluster": cluster.to_dict(),
                    "result": str(completed_result),
                }
            )
            continue
        if resp_config.apply_mode == RespApplyMode.APPLY_EXISTING:
            raise ValueError(
                f"No completed, fingerprint-matching protein-site RESP result was found for metal sites {cluster.metal_sites}. "
                "The prepared structure, metal identity/oxidation state, protonation, donor mapping, force field, "
                "scope, and multiplicity must all match."
            )
        metal_labels = []
        charge_by_site = {int(item.site): int(item.charge) for item in system_config.metal_charges}
        atom_by_index = {atom.topology_index: atom for atom in atoms}
        for site, atom_index in zip(cluster.metal_sites, cluster.metal_atom_indices, strict=True):
            metal_labels.append(f"{atom_by_index[atom_index].element}{charge_by_site.get(site, 0)}")
        label = "_".join(metal_labels)[:40]
        # Never overwrite an explicitly supplied completed/result folder when
        # creating a new calculation; only matching existing jobs are reused.
        job_dir = _next_job_dir(base_path, label)
        inputs_dir = job_dir / "inputs"
        output_dir = job_dir / "output"
        slurm_dir = job_dir / "slurm"
        manifests_dir = job_dir / "manifests"
        for directory in (job_dir, inputs_dir, output_dir, slurm_dir, manifests_dir):
            directory.mkdir(parents=True, exist_ok=True)
        molecule = payload["molecule"]
        assert isinstance(molecule, MoleculeData)
        description = (
            f"SIMPLE protein-site RESP | source={source_label} | metals={','.join(metal_labels)} | "
            f"sites={','.join(str(site) for site in cluster.metal_sites)} | scope={resp_config.scope.value}"
        )
        qm_settings = metal_safe_qm_settings(
            None,
            net_charge=int(payload["net_charge"]),
            multiplicity=cluster.multiplicity,
        )
        session_state = {
            "residue_name": "SITE",
            "qm_settings": qm_settings,
            "fingerprint": fingerprint,
        }
        nwchem_text = render_nwchem_input(
            molecule,
            session_state=session_state,
            convergence_profile="metal_robust",
        )
        retry_state = {**session_state, "residue_name": "SITE_RETRY"}
        nwchem_retry_text = render_nwchem_input(
            molecule,
            session_state=retry_state,
            convergence_profile="metal_retry",
        )
        header = "\n".join(
            [
                f"# {description}",
                "# Hybrid model: formal integer metal charge + 12-6-4 + site-specific residue RESP redistribution.",
                "# Metal and per-residue total charges are exact fitting constraints.",
            ]
        )
        (inputs_dir / "resp_job.nw").write_text(header + "\n" + nwchem_text, encoding="utf-8")
        (inputs_dir / "resp_job_retry.nw").write_text(
            header
            + "\n# Automatic fallback: small-basis PBE orbital preconditioning, basis projection, then the requested r2SCAN ESP level.\n"
            + nwchem_retry_text,
            encoding="utf-8",
        )
        (inputs_dir / "resp_job.xyz").write_text(_render_xyz(molecule, description), encoding="utf-8")
        (inputs_dir / "resp_fit.py").write_text(
            render_constrained_runtime_resp_fit_script(
                atom_names=[atom.name for atom in molecule.atoms],
                total_charge=float(payload["net_charge"]),
                equality_pairs=list(payload["equality_pairs"]),
                fixed_charges=dict(payload["fixed_charges"]),
                sum_constraints=list(payload["sum_constraints"]),
                atom_metadata=list(payload["metadata"]),
                xyz_filename="resp_job.xyz",
                grid_filename="resp_job.grid",
                fingerprint=fingerprint,
            ),
            encoding="utf-8",
        )
        constraint_payload = {
            "solver_version": CONSTRAINED_RESP_SOLVER_VERSION,
            "fixed_charges": payload["fixed_charges"],
            "sum_constraints": payload["sum_constraints"],
            "equality_pairs": payload["equality_pairs"],
            "atom_metadata": payload["metadata"],
        }
        write_json(job_dir / "site_resp_constraints.json", constraint_payload)
        # The shared NWChem Slurm renderers copy this conventional filename.
        # Keep the site-specific name above as the public manifest target and
        # provide this exact alias so both generic and Tahoma scripts are runnable.
        write_json(job_dir / "group_constraints.json", constraint_payload)
        generic = render_resp_slurm_script(
            job_root=job_dir,
            slurm_config=slurm_config,
            job_name=f"site_resp_{label}",
            retry_input="resp_job_retry.nw",
        )
        tahoma = render_tahoma_resp_script(
            job_root=job_dir,
            job_name=f"site_resp_{label}",
            retry_input="resp_job_retry.nw",
        )
        # Slurm requires the interpreter directive to be the first line.  The
        # descriptive text is already retained in the manifest and README.
        (slurm_dir / "run_resp.sbatch").write_text(generic, encoding="utf-8")
        (slurm_dir / "tahoma_resp.sbatch").write_text(tahoma, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "setup_pending",
            "created_at": datetime.now(UTC).isoformat(),
            "source_label": source_label,
            "source_pdb": None if source_pdb_path is None else str(source_pdb_path),
            "source_pdb_sha256": (
                None
                if source_pdb_path is None or not source_pdb_path.exists()
                else hashlib.sha256(source_pdb_path.read_bytes()).hexdigest()
            ),
            "prepared_system_pdb": str(system_pdb_path),
            "prepared_system_pdb_sha256": hashlib.sha256(system_pdb_path.read_bytes()).hexdigest(),
            "reference_topology_sha256": hashlib.sha256(system_prmtop_path.read_bytes()).hexdigest(),
            "description": description,
            "fingerprint": fingerprint,
            "identity": payload["identity"],
            "cluster": cluster.to_dict(),
            "scope": resp_config.scope.value,
            "protein_ff": system_config.protein_ff,
            "formal_metal_states": payload["identity"]["formal_metal_states"],
            "donor_residues": cluster.donor_residue_keys,
            "protonation_variants": {
                key: key.split(":", maxsplit=2)[1]
                for key in cluster.donor_residue_keys
            },
            "net_charge": payload["net_charge"],
            "multiplicity": cluster.multiplicity,
            "resp_fitter": {
                "version": CONSTRAINED_RESP_SOLVER_VERSION,
                "method": "baseline-centered constrained RESP in the exact-constraint null space",
            },
            "scf_strategy": {
                "primary": "r2SCAN/def2-TZVP with damping, level shifting, Rabuck fractional occupation, and DIIS",
                "retry": "PBE/def2-SVP quadratic-convergence preconditioning projected into r2SCAN/def2-TZVP",
                "final_esp_level": "r2SCAN/def2-TZVP",
            },
            "expected_result": str((output_dir / "site_resp_charges.json").resolve()),
            "constraints_file": str((job_dir / "site_resp_constraints.json").resolve()),
        }
        write_json(manifests_dir / "site_resp_manifest.json", manifest)
        (job_dir / "README.txt").write_text(
            description
            + "\n\nThis is a site-specific hybrid charge model. The metal formal charge and every target residue's "
            "net charge remain fixed. Run slurm/tahoma_resp.sbatch on Tahoma. The script retries a failed primary "
            "SCF using small-basis PBE orbital preconditioning and basis projection, while the final ESP remains "
            "r2SCAN/def2-TZVP. Then rerun SIMPLE "
            "and review the charge-delta table before application.\n",
            encoding="utf-8",
        )
        results.append(
            {
                "status": "setup_pending",
                "job_dir": str(job_dir.resolve()),
                "fingerprint": fingerprint,
                "cluster": cluster.to_dict(),
                "manifest": str((manifests_dir / "site_resp_manifest.json").resolve()),
                "nwchem_input": str((inputs_dir / "resp_job.nw").resolve()),
                "nwchem_retry_input": str((inputs_dir / "resp_job_retry.nw").resolve()),
                "generic_sbatch": str((slurm_dir / "run_resp.sbatch").resolve()),
                "tahoma_sbatch": str((slurm_dir / "tahoma_resp.sbatch").resolve()),
                "expected_result": str((output_dir / "site_resp_charges.json").resolve()),
            }
        )
    return results


def _load_xyz_bohr(path: Path, atom_count: int) -> list[list[float]]:
    rows = path.read_text(encoding="utf-8").splitlines()[2:]
    if len(rows) < atom_count:
        raise ValueError(f"RESP XYZ contains {len(rows)} atoms; expected {atom_count}.")
    return [[float(value) / 0.529177 for value in row.split()[1:4]] for row in rows[:atom_count]]


def _load_grid(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    point_count = int(lines[0].split()[0])
    rows = [[float(value) for value in line.split()[:4]] for line in lines[1 : point_count + 1]]
    if len(rows) != point_count:
        raise ValueError(f"RESP grid contains {len(rows)} rows; expected {point_count}.")
    return rows


def materialize_site_resp_result(job_dir: str | Path, *, force_refit: bool = False) -> Path | None:
    root = Path(job_dir).expanduser().resolve()
    existing = site_resp_result_path(root)
    if existing is not None and not force_refit:
        return existing
    manifest_path = root / "manifests" / "site_resp_manifest.json"
    constraints_path = root / "site_resp_constraints.json"
    if not manifest_path.exists() or not constraints_path.exists():
        return None
    grid_matches = [root / "output" / "resp_job.grid", *sorted((root / "output").glob("*.grid"))]
    grid_path = next((path for path in grid_matches if path.exists()), None)
    xyz_matches = [root / "output" / "resp_job.xyz", root / "inputs" / "resp_job.xyz"]
    xyz_path = next((path for path in xyz_matches if path.exists()), None)
    if grid_path is None or xyz_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    metadata = list(constraints.get("atom_metadata") or [])
    atom_names = [str(item.get("atom_name") or f"A{index + 1}") for index, item in enumerate(metadata)]
    result = fit_constrained_resp_payload(
        atom_names=atom_names,
        coordinates_bohr=_load_xyz_bohr(xyz_path, len(atom_names)),
        grid_rows=_load_grid(grid_path),
        total_charge=float(manifest.get("net_charge") or 0.0),
        equality_pairs=[tuple(pair) for pair in constraints.get("equality_pairs") or []],
        fixed_charges={int(index): float(value) for index, value in dict(constraints.get("fixed_charges") or {}).items()},
        sum_constraints=list(constraints.get("sum_constraints") or []),
        atom_metadata=metadata,
    )
    result["fingerprint"] = manifest.get("fingerprint")
    target = root / "output" / "site_resp_charges.json"
    text_target = root / "output" / "site_resp_charges.txt"
    if target.exists() and force_refit:
        backup = root / "output" / "site_resp_charges.pre_v2.json"
        if not backup.exists():
            shutil.copy2(target, backup)
    if text_target.exists() and force_refit:
        text_backup = root / "output" / "site_resp_charges.pre_v2.txt"
        if not text_backup.exists():
            shutil.copy2(text_target, text_backup)
    write_json(target, result)
    text_target.write_text(render_constrained_charge_table(result), encoding="utf-8")
    return target


def review_site_resp_result(job_dir: str | Path) -> dict[str, object]:
    root = Path(job_dir).expanduser().resolve()
    result_path = materialize_site_resp_result(root)
    if result_path is None:
        raise ValueError(f"Completed protein-site RESP grid/charge output was not found under {root / 'output'}.")
    manifest = json.loads((root / "manifests" / "site_resp_manifest.json").read_text(encoding="utf-8"))
    constraints = json.loads((root / "site_resp_constraints.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("solver_version") != CONSTRAINED_RESP_SOLVER_VERSION:
        result_path = materialize_site_resp_result(root, force_refit=True)
        if result_path is None:
            raise ValueError(
                "This protein-site RESP result was produced by an obsolete fitter and its NWChem grid is not "
                "available for safe refitting. Regenerate the RESP job instead of applying these charges."
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("solver_version") != CONSTRAINED_RESP_SOLVER_VERSION:
        raise ValueError("Protein-site RESP result does not use the current numerically stable constrained fitter.")
    if not bool(result.get("numerically_stable")):
        raise ValueError(
            "Protein-site RESP fitting did not reach a numerically stable constrained solution; no charge patch "
            "will be applied. Inspect the ESP grid and constraint diagnostics."
        )
    if result.get("fingerprint") != manifest.get("fingerprint"):
        raise ValueError(
            "Protein-site RESP result fingerprint does not match its generated structure/site manifest."
        )
    expected_rows = list(constraints.get("atom_metadata") or [])
    fitted_rows = list(result.get("charges") or [])
    if len(fitted_rows) != len(expected_rows):
        raise ValueError(
            "Protein-site RESP atom mapping is incomplete: "
            f"result contains {len(fitted_rows)} atoms, expected {len(expected_rows)}."
        )
    seen_result_indices: set[int] = set()
    seen_topology_indices: set[int] = set()
    for position, (expected, fitted) in enumerate(zip(expected_rows, fitted_rows, strict=True), start=1):
        result_index = int(fitted.get("index") or 0)
        if result_index != position or result_index in seen_result_indices:
            raise ValueError("Protein-site RESP output contains a missing, duplicated, or reordered atom index.")
        seen_result_indices.add(result_index)
        for key in ("topology_index", "atom_name", "residue_key", "role", "apply"):
            if fitted.get(key) != expected.get(key):
                raise ValueError(
                    "Protein-site RESP atom mapping does not match the generated constraint manifest "
                    f"at cluster atom {position} ({key})."
                )
        if not math.isclose(
            float(fitted.get("original_charge") or 0.0),
            float(expected.get("original_charge") or 0.0),
            abs_tol=1.0e-8,
        ):
            raise ValueError(f"Protein-site RESP baseline charge mismatch at cluster atom {position}.")
        topology_index = fitted.get("topology_index")
        if topology_index is not None:
            topology_index = int(topology_index)
            if topology_index in seen_topology_indices:
                raise ValueError("Protein-site RESP output contains a duplicated topology atom mapping.")
            seen_topology_indices.add(topology_index)
    maximum_residual = float(result.get("maximum_constraint_residual") or 0.0)
    esp_rmse = float(result.get("esp_rmse") or 0.0)
    relative_rmse_raw = result.get("esp_relative_rmse")
    if not math.isfinite(maximum_residual) or not math.isfinite(esp_rmse):
        raise ValueError("Protein-site RESP output contains a non-finite fit metric.")
    if relative_rmse_raw is not None and not math.isfinite(float(relative_rmse_raw)):
        raise ValueError("Protein-site RESP output contains a non-finite relative ESP residual.")
    if maximum_residual > 1.0e-6:
        raise ValueError(f"Protein-site RESP constraint residual is too large: {maximum_residual:.3e}.")
    changes: list[dict[str, object]] = []
    warnings: list[str] = []
    if relative_rmse_raw is not None and float(relative_rmse_raw) > 0.15:
        warnings.append(
            f"High normalized ESP residual ({float(relative_rmse_raw):.4f}): the fitted point-charge model leaves "
            "a large fraction of the QM ESP magnitude unexplained. Compare the fitted and baseline RMSE values; "
            "do not approve this hybrid charge patch unless the fit is scientifically acceptable."
        )
    charge_by_cluster_index: dict[int, float] = {}
    for cluster_index, row in enumerate(fitted_rows):
        charge = float(row["charge"])
        charge_by_cluster_index[cluster_index] = charge
        original = float(row.get("original_charge") or 0.0)
        if not math.isfinite(charge):
            raise ValueError("Protein-site RESP output contains a non-finite charge.")
        if bool(row.get("fixed")) and not math.isclose(charge, original, abs_tol=1.0e-6):
            raise ValueError(
                f"A fixed protein-site RESP atom changed charge: {row.get('residue_key')}@{row.get('atom_name')}."
            )
        if bool(row.get("apply")):
            delta = charge - original
            if abs(delta) > 0.5:
                warnings.append(
                    f"Large charge change {delta:+.4f} e for {row.get('residue_key')}@{row.get('atom_name')}."
                )
            changes.append(
                {
                    "topology_index": int(row["topology_index"]),
                    "residue_key": row.get("residue_key"),
                    "atom_name": row.get("atom_name"),
                    "original_charge": original,
                    "charge": charge,
                    "delta": delta,
                }
            )
    residue_sums: list[dict[str, object]] = []
    for constraint in constraints.get("sum_constraints") or []:
        indices = [int(value) for value in constraint.get("atom_indices") or []]
        actual = sum(charge_by_cluster_index[index] for index in indices)
        expected = float(constraint.get("charge") or 0.0)
        if not math.isclose(actual, expected, abs_tol=1.0e-6):
            raise ValueError(
                f"Protein-site RESP residue-total constraint failed for {constraint.get('label')}: "
                f"{actual:.8f} versus {expected:.8f}."
            )
        baseline = sum(float(expected_rows[index].get("original_charge") or 0.0) for index in indices)
        residue_sums.append(
            {
                "label": constraint.get("label"),
                "baseline": baseline,
                "fitted": actual,
                "target": expected,
                "residual": actual - expected,
            }
        )
    symmetry_constraints: list[dict[str, object]] = []
    for first, second in constraints.get("equality_pairs") or []:
        if not math.isclose(
            charge_by_cluster_index[int(first)],
            charge_by_cluster_index[int(second)],
            abs_tol=1.0e-6,
        ):
            raise ValueError(f"Protein-site RESP symmetry constraint failed for atoms {first} and {second}.")
        symmetry_constraints.append(
            {
                "first": int(first) + 1,
                "second": int(second) + 1,
                "first_atom": f"{expected_rows[int(first)].get('residue_key')}@{expected_rows[int(first)].get('atom_name')}",
                "second_atom": f"{expected_rows[int(second)].get('residue_key')}@{expected_rows[int(second)].get('atom_name')}",
                "charge": charge_by_cluster_index[int(first)],
            }
        )
    expected_total = float(manifest.get("net_charge") or 0.0)
    fitted_total = sum(charge_by_cluster_index.values())
    if not math.isclose(fitted_total, expected_total, abs_tol=1.0e-6):
        raise ValueError(
            f"Protein-site RESP cluster total charge changed from {expected_total:.8f} to {fitted_total:.8f}."
        )
    return {
        "job_dir": str(root),
        "fingerprint": manifest.get("fingerprint"),
        "description": manifest.get("description"),
        "cluster": manifest.get("cluster"),
        "esp_rmse": result.get("esp_rmse"),
        "esp_relative_rmse": result.get("esp_relative_rmse"),
        "baseline_esp_rmse": result.get("baseline_esp_rmse"),
        "maximum_absolute_delta": result.get("maximum_absolute_delta"),
        "solver_version": result.get("solver_version"),
        "esp_design_condition_number": result.get("esp_design_condition_number"),
        "maximum_constraint_residual": maximum_residual,
        "changes": changes,
        "baseline_atoms": [
            {
                "topology_index": int(row["topology_index"]),
                "residue_key": row.get("residue_key"),
                "atom_name": row.get("atom_name"),
                "original_charge": float(row.get("original_charge") or 0.0),
            }
            for row in fitted_rows
            if row.get("topology_index") is not None
        ],
        "residue_sums": residue_sums,
        "symmetry_constraints": symmetry_constraints,
        "warnings": warnings,
        "result_path": str(result_path),
    }


def _topology_charge_values(path: Path) -> list[float]:
    sections = _parse_prmtop_sections(path)
    return [float(value) / AMBER_CHARGE_SCALE for value in _section_values(sections, "CHARGE")]


def apply_site_resp_results(
    *,
    job_dirs: list[str | Path],
    prmtop_path: str | Path,
    inpcrd_path: str | Path,
    parmed_binary: str | Path | None,
    output_dir: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    prmtop = Path(prmtop_path).resolve()
    inpcrd = Path(inpcrd_path).resolve()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    reviews = [review_site_resp_result(job_dir) for job_dir in job_dirs]
    changes_by_atom: dict[int, dict[str, object]] = {}
    baseline_by_atom: dict[int, dict[str, object]] = {}
    for review in reviews:
        for baseline in review["baseline_atoms"]:
            index = int(baseline["topology_index"])
            previous_baseline = baseline_by_atom.get(index)
            if previous_baseline is not None and not math.isclose(
                float(previous_baseline["original_charge"]),
                float(baseline["original_charge"]),
                abs_tol=1.0e-8,
            ):
                raise ValueError(f"Protein-site RESP jobs contain conflicting baseline mappings for atom {index}.")
            baseline_by_atom[index] = dict(baseline)
        for change in review["changes"]:
            index = int(change["topology_index"])
            previous = changes_by_atom.get(index)
            if previous is not None and not math.isclose(float(previous["charge"]), float(change["charge"]), abs_tol=1.0e-8):
                raise ValueError(f"Protein-site RESP jobs contain conflicting charges for topology atom {index}.")
            changes_by_atom[index] = dict(change)
    if not changes_by_atom:
        raise ValueError("Protein-site RESP results did not contain any target topology charge changes.")
    original_charges = _topology_charge_values(prmtop)
    for index, baseline in baseline_by_atom.items():
        if index < 1 or index > len(original_charges):
            raise ValueError(f"Protein-site RESP references topology atom {index}, but topology has {len(original_charges)} atoms.")
        if not math.isclose(original_charges[index - 1], float(baseline["original_charge"]), abs_tol=2.0e-5):
            raise ValueError(
                f"Topology charge mismatch for atom {index}; the RESP result belongs to a different prepared system."
            )
    backup = target_dir / "system.standard_ff.prmtop"
    # run_workflow rebuilds the standard-FF + C4 topology before every apply.
    # Refresh the preserved baseline so an older patched-run backup is never reused.
    if prmtop != backup:
        shutil.copy2(prmtop, backup)
    candidate = target_dir / "system.site_resp.prmtop"
    script_path = target_dir / "parmed_site_resp.in"
    script_lines = [
        "# SIMPLE protein-site RESP charge patch",
        "# Metal formal charges and residue total charges were fixed during RESP fitting.",
        "setOverwrite True",
    ]
    for index, change in sorted(changes_by_atom.items()):
        script_lines.append(f"change CHARGE @{index} {float(change['charge']):.10f}")
    script_lines.extend([f"outparm {candidate.as_posix()} {inpcrd.as_posix()}", "quit"])
    script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    application = {
        "status": "planned" if dry_run else "applied",
        "created_at": datetime.now(UTC).isoformat(),
        "job_dirs": [str(Path(path).resolve()) for path in job_dirs],
        "source_topology": str(backup),
        "patched_topology": str(prmtop),
        "parmed_script": str(script_path),
        "changes": [changes_by_atom[index] for index in sorted(changes_by_atom)],
        "reviews": reviews,
    }
    if dry_run:
        write_json(target_dir / "site_resp_application.json", application)
        return application
    if parmed_binary is None:
        raise RuntimeError("Applying protein-site RESP charges requires ParmEd (`parmed`) on the execution host.")
    command = [str(parmed_binary), "-i", str(script_path), "-p", str(backup), "-c", str(inpcrd)]
    run_command(command, cwd=target_dir, log_path=target_dir / "parmed_site_resp.log")
    if not candidate.exists():
        raise RuntimeError("ParmEd did not create the protein-site RESP candidate topology.")
    updated_charges = _topology_charge_values(candidate)
    if not math.isclose(sum(original_charges), sum(updated_charges), abs_tol=1.0e-6):
        raise RuntimeError(
            "Protein-site RESP topology validation failed because the total system charge changed "
            f"from {sum(original_charges):.8f} to {sum(updated_charges):.8f}."
        )
    for index, change in changes_by_atom.items():
        if not math.isclose(updated_charges[index - 1], float(change["charge"]), abs_tol=2.0e-6):
            raise RuntimeError(f"ParmEd did not apply the expected protein-site RESP charge to atom {index}.")
    original_c4 = [float(value) for value in _section_values(_parse_prmtop_sections(backup), "LENNARD_JONES_CCOEF")]
    updated_c4 = [float(value) for value in _section_values(_parse_prmtop_sections(candidate), "LENNARD_JONES_CCOEF")]
    if len(original_c4) != len(updated_c4) or any(
        not math.isclose(before, after, rel_tol=1.0e-10, abs_tol=1.0e-12)
        for before, after in zip(original_c4, updated_c4, strict=True)
    ):
        raise RuntimeError("Protein-site RESP topology patch did not preserve the complete 12-6-4 C4 data.")
    os.replace(candidate, prmtop)
    application["preserved_lennard_jones_ccoef"] = bool(updated_c4)
    application["total_charge_before"] = sum(original_charges)
    application["total_charge_after"] = sum(updated_charges)
    write_json(target_dir / "site_resp_application.json", application)
    return application
