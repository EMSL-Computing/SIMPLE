from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import gemmi

from amber_metallo.amber.leap import DEFAULT_TLEAP_METAL_CHARGES
from amber_metallo.execution import run_command
from amber_metallo.inspection import SUPPORTED_METALS, classify_residue, load_structure, residue_key
from amber_metallo.reporting import write_json


SUPPORTED_DONOR_ELEMENTS = {"N", "O", "S"}
_CNTRL_PAIR_PATTERN = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^,]+)")


@dataclass(slots=True)
class IndexedAtom:
    atom_index: int
    chain: str
    seqid: str
    residue_name: str
    residue_key: str
    atom_name: str
    element: str
    classification: str
    position: gemmi.Position


@dataclass(slots=True)
class DonorAtom:
    atom_index: int
    residue_key: str
    residue_name: str
    atom_name: str
    element: str
    distance_angstrom: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MetalSiteCandidate:
    site: int
    atom_index: int
    key: str
    chain: str
    seqid: str
    residue_name: str
    atom_name: str
    element: str
    donor_count: int
    nearest_donor_distance_angstrom: float | None
    c4_supported: bool | None
    donors: list[DonorAtom] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "atom_index": self.atom_index,
            "key": self.key,
            "chain": self.chain,
            "seqid": self.seqid,
            "residue_name": self.residue_name,
            "atom_name": self.atom_name,
            "element": self.element,
            "donor_count": self.donor_count,
            "nearest_donor_distance_angstrom": self.nearest_donor_distance_angstrom,
            "c4_supported": self.c4_supported,
            "donors": [item.to_dict() for item in self.donors],
        }


@dataclass(slots=True)
class SiteStabilityAssessment:
    site: int
    stable: bool
    displacement_angstrom: float
    retained_donor_count: int
    reference_donor_count: int
    note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class InheritedMDSettings:
    dt_ps: float = 0.002
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    cut_angstrom: float = 8.0
    ntb: int = 2
    ntp: int = 1
    ntc: int = 2
    ntf: int = 2
    ntt: int = 3
    gamma_ln: float = 2.0
    ntpr: int = 1000
    ntwx: int = 1000
    ntwr: int = 1000
    ioutfm: int = 1
    iwrap: int = 0
    barostat: int | None = None
    taup: float | None = None
    tempi_k: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_formal_charge(element: str) -> int:
    return DEFAULT_TLEAP_METAL_CHARGES.get(element.title(), 2)


def _remove_extra_models(structure: gemmi.Structure) -> None:
    while len(structure) > 1:
        del structure[1]


def _iter_indexed_atoms(source: str | Path | gemmi.Structure) -> list[IndexedAtom]:
    structure = source if isinstance(source, gemmi.Structure) else load_structure(source)
    _remove_extra_models(structure)
    indexed: list[IndexedAtom] = []
    atom_index = 1
    for chain in structure[0]:
        for residue in chain:
            residue_label = residue_key(chain.name, residue)
            classification = classify_residue(residue)
            for atom in residue:
                indexed.append(
                    IndexedAtom(
                        atom_index=atom_index,
                        chain=chain.name.strip(),
                        seqid=str(residue.seqid).strip(),
                        residue_name=residue.name.strip(),
                        residue_key=residue_label,
                        atom_name=atom.name.strip(),
                        element=atom.element.name.upper(),
                        classification=classification,
                        position=atom.pos,
                    )
                )
                atom_index += 1
    return indexed


def _parse_prmtop_sections(path: str | Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if line.startswith("%FLAG"):
            current = line.split(maxsplit=1)[1].strip()
            sections[current] = []
            continue
        if line.startswith("%FORMAT") or not line or current is None:
            continue
        sections[current].append(line)
    return sections


def _tokens_from_section(sections: dict[str, list[str]], name: str) -> list[str]:
    values = []
    for line in sections.get(name, []):
        values.extend(line.split())
    return values


def atom_type_supports_c4(prmtop_path: str | Path, atom_index: int) -> bool | None:
    try:
        sections = _parse_prmtop_sections(prmtop_path)
        atom_type_indices = [int(token) for token in _tokens_from_section(sections, "ATOM_TYPE_INDEX")]
        nonbonded_index = [int(token) for token in _tokens_from_section(sections, "NONBONDED_PARM_INDEX")]
        ccoefs = [float(token.replace("D", "E")) for token in _tokens_from_section(sections, "LENNARD_JONES_CCOEF")]
    except Exception:
        return None

    if not atom_type_indices or not nonbonded_index or not ccoefs:
        return None
    if atom_index < 1 or atom_index > len(atom_type_indices):
        return None

    type_index = atom_type_indices[atom_index - 1]
    ntypes = max(atom_type_indices)
    if len(nonbonded_index) < ntypes * ntypes:
        return None

    for partner_type in range(1, ntypes + 1):
        pair_index = nonbonded_index[(type_index - 1) * ntypes + (partner_type - 1)]
        if pair_index <= 0 or pair_index > len(ccoefs):
            continue
        if abs(ccoefs[pair_index - 1]) > 1.0e-8:
            return True
    return False


def _c4_residue_names_from_system_manifest(prmtop_path: str | Path) -> set[str] | None:
    manifest_path = Path(prmtop_path).expanduser().resolve().with_name("system_manifest.json")
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not payload.get("c4_applied"):
        return None
    mask = str(payload.get("c4_mask") or "").strip()
    if not mask.startswith(":"):
        return None
    residue_names = {token.strip().upper() for token in mask[1:].split(",") if token.strip()}
    return residue_names or None


def detect_bound_metal_sites(
    reference_structure_path: str | Path,
    prmtop_path: str | Path | None = None,
    *,
    donor_cutoff_angstrom: float = 3.0,
    include_unbound_metals: bool = False,
) -> list[MetalSiteCandidate]:
    atoms = _iter_indexed_atoms(reference_structure_path)
    c4_residue_names = _c4_residue_names_from_system_manifest(prmtop_path) if prmtop_path else None
    candidates: list[MetalSiteCandidate] = []
    site_index = 0
    for atom in atoms:
        if atom.classification != "metal" or atom.element.title() not in SUPPORTED_METALS:
            continue
        site_index += 1
        donors: list[DonorAtom] = []
        for other in atoms:
            if other.atom_index == atom.atom_index:
                continue
            if other.classification in {"water", "metal"}:
                continue
            if other.element not in SUPPORTED_DONOR_ELEMENTS:
                continue
            distance = atom.position.dist(other.position)
            if distance > donor_cutoff_angstrom:
                continue
            donors.append(
                DonorAtom(
                    atom_index=other.atom_index,
                    residue_key=other.residue_key,
                    residue_name=other.residue_name,
                    atom_name=other.atom_name,
                    element=other.element,
                    distance_angstrom=distance,
                )
            )
        donors.sort(key=lambda item: (item.distance_angstrom, item.atom_index))
        c4_supported = atom_type_supports_c4(prmtop_path, atom.atom_index) if prmtop_path else None
        if c4_supported is None and c4_residue_names is not None and atom.residue_name.strip().upper() in c4_residue_names:
            c4_supported = True
        is_bound_like = len(donors) >= 2 or (len(donors) >= 1 and c4_supported is True)
        if not is_bound_like and not include_unbound_metals:
            continue
        candidates.append(
            MetalSiteCandidate(
                site=site_index,
                atom_index=atom.atom_index,
                key=atom.residue_key,
                chain=atom.chain,
                seqid=atom.seqid,
                residue_name=atom.residue_name,
                atom_name=atom.atom_name,
                element=atom.element.title(),
                donor_count=len(donors),
                nearest_donor_distance_angstrom=donors[0].distance_angstrom if donors else None,
                c4_supported=c4_supported,
                donors=donors,
            )
        )
    return candidates


def select_site(candidates: list[MetalSiteCandidate], site_number: int) -> MetalSiteCandidate:
    for candidate in candidates:
        if candidate.site == site_number:
            return candidate
    raise ValueError(f"Metal site {site_number} is not available.")


def build_cluster_atom_mask(
    reference_structure_path: str | Path,
    candidate: MetalSiteCandidate,
    *,
    radius_angstrom: float = 6.0,
) -> str:
    atoms = _iter_indexed_atoms(reference_structure_path)
    by_index = {atom.atom_index: atom for atom in atoms}
    metal = by_index.get(candidate.atom_index)
    if metal is None:
        return f"@{candidate.atom_index}"

    selected = [
        atom.atom_index
        for atom in atoms
        if atom.element != "H"
        and atom.classification != "water"
        and atom.position.dist(metal.position) <= radius_angstrom
    ]
    return "@" + ",".join(str(index) for index in sorted(selected))


def assess_site_stability(
    reference_structure_path: str | Path,
    last_snapshot_path: str | Path,
    candidate: MetalSiteCandidate,
    *,
    diffusion_cutoff_angstrom: float,
    retained_donor_cutoff_angstrom: float,
) -> SiteStabilityAssessment:
    ref_atoms = {atom.atom_index: atom for atom in _iter_indexed_atoms(reference_structure_path)}
    last_atoms = {atom.atom_index: atom for atom in _iter_indexed_atoms(last_snapshot_path)}
    ref_metal = ref_atoms.get(candidate.atom_index)
    last_metal = last_atoms.get(candidate.atom_index)
    if ref_metal is None or last_metal is None:
        return SiteStabilityAssessment(
            site=candidate.site,
            stable=False,
            displacement_angstrom=float("inf"),
            retained_donor_count=0,
            reference_donor_count=len(candidate.donors),
            note="Selected metal atom index was not found in the last snapshot.",
        )

    direct_displacement = ref_metal.position.dist(last_metal.position)
    distance_shift_terms: list[float] = []
    retained = 0
    for donor in candidate.donors:
        ref_donor = ref_atoms.get(donor.atom_index)
        last_donor = last_atoms.get(donor.atom_index)
        if ref_donor is None or last_donor is None:
            continue
        ref_distance = ref_donor.position.dist(ref_metal.position)
        last_distance = last_donor.position.dist(last_metal.position)
        distance_shift_terms.append((last_distance - ref_distance) ** 2)
        if last_donor.position.dist(last_metal.position) <= retained_donor_cutoff_angstrom:
            retained += 1

    displacement = (
        math.sqrt(sum(distance_shift_terms) / len(distance_shift_terms))
        if distance_shift_terms
        else direct_displacement
    )
    stable = displacement <= diffusion_cutoff_angstrom and retained >= min(2, max(1, len(candidate.donors)))
    note = "Metal site looks stable in the last snapshot."
    if displacement > diffusion_cutoff_angstrom:
        if distance_shift_terms:
            note = (
                f"Metal coordination geometry shifted by {displacement:.2f} A RMS relative to the reference donor "
                f"distances (cutoff {diffusion_cutoff_angstrom:.2f} A)."
            )
        else:
            note = (
                f"Metal moved {displacement:.2f} A from the reference site "
                f"(cutoff {diffusion_cutoff_angstrom:.2f} A)."
            )
    elif retained < min(2, max(1, len(candidate.donors))):
        note = (
            f"Only {retained} donor atoms remain within {retained_donor_cutoff_angstrom:.2f} A "
            "of the metal in the last snapshot."
        )
    elif distance_shift_terms and direct_displacement > diffusion_cutoff_angstrom:
        note = (
            "Metal site looks stable after discounting rigid-body translation/rotation; "
            f"local donor-distance shift is {displacement:.2f} A."
        )
    return SiteStabilityAssessment(
        site=candidate.site,
        stable=stable,
        displacement_angstrom=displacement,
        retained_donor_count=retained,
        reference_donor_count=len(candidate.donors),
        note=note,
    )


def render_last_snapshot_cpptraj_script(
    *,
    prmtop_path: str | Path,
    trajectory_path: str | Path,
    output_pdb: str | Path,
    output_rst7: str | Path,
) -> str:
    return (
        f"parm {Path(prmtop_path).as_posix()}\n"
        f"trajin {Path(trajectory_path).as_posix()} lastframe\n"
        "autoimage\n"
        f"trajout {Path(output_pdb).as_posix()} pdb nobox\n"
        f"trajout {Path(output_rst7).as_posix()} restart\n"
        "run\n"
    )


def render_cluster_cpptraj_script(
    *,
    prmtop_path: str | Path,
    trajectory_path: str | Path,
    atom_mask: str,
    output_dir: str | Path,
    epsilon_angstrom: float,
    sieve: int,
) -> str:
    cluster_dir = Path(output_dir)
    rep_prefix = cluster_dir / "representative"
    return (
        f"parm {Path(prmtop_path).as_posix()}\n"
        f"trajin {Path(trajectory_path).as_posix()}\n"
        "autoimage\n"
        f"rms first {atom_mask}\n"
        f"cluster C0 hieragglo epsilon {epsilon_angstrom:.3f} rms {atom_mask} "
        f"sieve {sieve} summary {cluster_dir.joinpath('cluster_summary.dat').as_posix()} "
        f"info {cluster_dir.joinpath('cluster_info.dat').as_posix()} "
        f"repout {rep_prefix.as_posix()} repfmt pdb\n"
        "run\n"
    )


def _cpptraj_binary() -> str | None:
    found = shutil.which("cpptraj")
    return found


def write_placeholder_restart(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("TI dry-run restart placeholder\n0\n", encoding="utf-8")
    return target


def copy_structure(source: str | Path, target: str | Path) -> Path:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return destination


_ATOMIC_SYMBOLS_BY_NUMBER = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    11: "Na",
    12: "Mg",
    15: "P",
    16: "S",
    17: "Cl",
    19: "K",
    20: "Ca",
    21: "Sc",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    39: "Y",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
}


def _fixed_width_tokens_from_section(sections: dict[str, list[str]], name: str) -> list[str]:
    section = sections.get(name)
    if not section:
        return []
    width = None
    # _parse_prmtop_sections keeps only data lines, so infer the common Amber widths.
    if name in {"ATOM_NAME", "AMBER_ATOM_TYPE", "RESIDUE_LABEL"}:
        width = 4
    if width is None:
        return _tokens_from_section(sections, name)
    values: list[str] = []
    for line in section:
        for start in range(0, len(line), width):
            token = line[start : start + width].strip()
            if token:
                values.append(token)
    return values


def _infer_element_from_prmtop_atom(atom_name: str, residue_name: str) -> str:
    candidates = []
    for raw in (atom_name, residue_name):
        cleaned = "".join(character for character in raw.strip() if character.isalpha())
        if not cleaned:
            continue
        candidates.append(cleaned[:2].title())
        candidates.append(cleaned[:1].title())
    for candidate in candidates:
        if candidate in SUPPORTED_METALS or candidate in _ATOMIC_SYMBOLS_BY_NUMBER.values():
            return candidate
    return candidates[-1] if candidates else "X"


def _parse_formatted_amber_restart_coordinates(path: str | Path) -> list[tuple[float, float, float]]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Amber restart is too short: {path}")
    header_tokens = lines[1].split()
    if not header_tokens:
        raise ValueError(f"Amber restart header is missing atom count: {path}")
    natom = int(header_tokens[0])
    values: list[float] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        try:
            values.extend(float(token.replace("D", "E").replace("d", "e")) for token in line.split())
            continue
        except ValueError:
            pass
        for start in range(0, len(line), 12):
            token = line[start : start + 12].strip()
            if token:
                values.append(float(token.replace("D", "E").replace("d", "e")))
    coord_values = values[: 3 * natom]
    if len(coord_values) < 3 * natom:
        raise ValueError(f"Amber restart has fewer coordinate values than expected for {natom} atoms: {path}")
    return [
        (coord_values[index], coord_values[index + 1], coord_values[index + 2])
        for index in range(0, len(coord_values), 3)
    ]


def _pdb_atom_name_field(atom_name: str, element: str) -> str:
    name = atom_name.strip()[:4] or element
    if len(name) < 4 and len(element.strip()) == 1:
        return f" {name:<3}"
    return f"{name:>4}"[:4]


def _write_reference_pdb_from_prmtop_restart(
    *,
    prmtop_path: str | Path,
    restart_path: str | Path,
    output_path: str | Path,
) -> Path:
    sections = _parse_prmtop_sections(prmtop_path)
    atom_names = _fixed_width_tokens_from_section(sections, "ATOM_NAME")
    residue_labels = _fixed_width_tokens_from_section(sections, "RESIDUE_LABEL")
    residue_pointers = [int(token) for token in _tokens_from_section(sections, "RESIDUE_POINTER")]
    atomic_numbers = [int(token) for token in _tokens_from_section(sections, "ATOMIC_NUMBER")]
    coords = _parse_formatted_amber_restart_coordinates(restart_path)
    natom = len(coords)
    if not atom_names:
        atom_names = [f"X{index}" for index in range(1, natom + 1)]
    if len(atom_names) < natom:
        atom_names.extend(f"X{index}" for index in range(len(atom_names) + 1, natom + 1))
    if not residue_labels or not residue_pointers:
        residue_labels = ["MOL"]
        residue_pointers = [1]
    residue_pointers = sorted(pointer for pointer in residue_pointers if pointer >= 1)
    residue_ranges: list[tuple[int, int, str]] = []
    for index, pointer in enumerate(residue_pointers):
        next_pointer = residue_pointers[index + 1] if index + 1 < len(residue_pointers) else natom + 1
        label = residue_labels[index] if index < len(residue_labels) else "MOL"
        residue_ranges.append((pointer, next_pointer, label[:3]))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["HEADER    GENERATED BY SIMPLE FROM AMBER PRMTOP/RST7\n"]
    residue_index = 0
    current_start, current_end, current_label = residue_ranges[0]
    for atom_index, (x, y, z) in enumerate(coords, start=1):
        while atom_index >= current_end and residue_index + 1 < len(residue_ranges):
            residue_index += 1
            current_start, current_end, current_label = residue_ranges[residue_index]
        atom_name = atom_names[atom_index - 1]
        if atom_index <= len(atomic_numbers) and atomic_numbers[atom_index - 1] in _ATOMIC_SYMBOLS_BY_NUMBER:
            element = _ATOMIC_SYMBOLS_BY_NUMBER[atomic_numbers[atom_index - 1]]
        else:
            element = _infer_element_from_prmtop_atom(atom_name, current_label)
        record = "HETATM" if element in SUPPORTED_METALS else "ATOM  "
        serial = atom_index % 100000
        resseq = (residue_index + 1) % 10000
        lines.append(
            f"{record}{serial:5d} {_pdb_atom_name_field(atom_name, element)} {current_label:>3} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
        )
    lines.append("END\n")
    target.write_text("".join(lines), encoding="utf-8")
    return target


def generate_reference_pdb_from_amber_restart(
    *,
    prmtop_path: str | Path,
    restart_path: str | Path,
    output_path: str | Path,
) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    ambpdb = shutil.which("ambpdb")
    if ambpdb is not None:
        log_path = target.with_suffix(".ambpdb.log")
        command_variants = [
            [ambpdb, "-p", str(Path(prmtop_path).expanduser().resolve()), "-c", str(Path(restart_path).expanduser().resolve())],
            [ambpdb, "-p", str(Path(prmtop_path).expanduser().resolve())],
        ]
        last_error: Exception | None = None
        for command in command_variants:
            try:
                stdin_payload = None
                if len(command) == 3:
                    stdin_payload = Path(restart_path).read_text(encoding="utf-8", errors="ignore")
                result = subprocess.run(
                    command,
                    cwd=str(target.parent),
                    input=stdin_payload,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                log_path.write_text(
                    "\n".join(
                        [
                            f"$ {' '.join(command)}",
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
                if result.returncode == 0 and ("ATOM" in result.stdout or "HETATM" in result.stdout):
                    target.write_text(result.stdout, encoding="utf-8")
                    return target
                last_error = RuntimeError(f"ambpdb did not produce a PDB; see {log_path}")
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            # Fall through to the lightweight parser; it is sufficient for metal-site detection.
            pass
    return _write_reference_pdb_from_prmtop_restart(
        prmtop_path=prmtop_path,
        restart_path=restart_path,
        output_path=target,
    )


def run_last_snapshot_extraction(
    *,
    prmtop_path: str | Path,
    trajectory_path: str | Path,
    reference_structure_path: str | Path,
    output_dir: str | Path,
    dry_run: bool,
) -> dict[str, str]:
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = target_dir / "last_snapshot.pdb"
    output_rst7 = target_dir / "last_snapshot.rst7"
    script_path = target_dir / "extract_last_snapshot.cpptraj.in"
    script_text = render_last_snapshot_cpptraj_script(
        prmtop_path=prmtop_path,
        trajectory_path=trajectory_path,
        output_pdb=output_pdb,
        output_rst7=output_rst7,
    )
    script_path.write_text(script_text, encoding="utf-8")
    if dry_run:
        copy_structure(reference_structure_path, output_pdb)
        write_placeholder_restart(output_rst7)
    else:
        binary = _cpptraj_binary()
        if binary is None:
            raise RuntimeError("cpptraj was not found on PATH, so the last snapshot could not be extracted.")
        run_command([binary, "-i", str(script_path)], cwd=target_dir, log_path=target_dir / "extract_last_snapshot.log")
    manifest = {
        "script": str(script_path),
        "last_snapshot_pdb": str(output_pdb),
        "last_snapshot_rst7": str(output_rst7),
    }
    write_json(target_dir / "last_snapshot_manifest.json", manifest)
    return manifest


def _pick_representative_pdb(directory: str | Path) -> Path | None:
    target_dir = Path(directory)
    for pattern in ("representative*.pdb", "rep*.pdb"):
        matches = sorted(target_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def run_cluster_representative_selection(
    *,
    prmtop_path: str | Path,
    trajectory_path: str | Path,
    reference_structure_path: str | Path,
    atom_mask: str,
    output_dir: str | Path,
    epsilon_angstrom: float,
    sieve: int,
    dry_run: bool,
) -> dict[str, str]:
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    script_path = target_dir / "cluster_representative.cpptraj.in"
    script_text = render_cluster_cpptraj_script(
        prmtop_path=prmtop_path,
        trajectory_path=trajectory_path,
        atom_mask=atom_mask,
        output_dir=target_dir,
        epsilon_angstrom=epsilon_angstrom,
        sieve=sieve,
    )
    script_path.write_text(script_text, encoding="utf-8")
    output_pdb = target_dir / "representative_snapshot.pdb"
    output_rst7 = target_dir / "representative_snapshot.rst7"
    if dry_run:
        copy_structure(reference_structure_path, output_pdb)
        write_placeholder_restart(output_rst7)
    else:
        binary = _cpptraj_binary()
        if binary is None:
            raise RuntimeError("cpptraj was not found on PATH, so cluster analysis could not be executed.")
        run_command([binary, "-i", str(script_path)], cwd=target_dir, log_path=target_dir / "cluster_representative.log")
        representative = _pick_representative_pdb(target_dir)
        if representative is None:
            raise RuntimeError(
                "cpptraj clustering completed, but no representative PDB file was produced."
            )
        copy_structure(representative, output_pdb)
        convert_script = target_dir / "representative_to_restart.cpptraj.in"
        convert_script.write_text(
            (
                f"parm {Path(prmtop_path).as_posix()}\n"
                f"trajin {output_pdb.as_posix()}\n"
                f"trajout {output_rst7.as_posix()} restart\n"
                "run\n"
            ),
            encoding="utf-8",
        )
        run_command([binary, "-i", str(convert_script)], cwd=target_dir, log_path=target_dir / "representative_to_restart.log")
    manifest = {
        "script": str(script_path),
        "representative_snapshot_pdb": str(output_pdb),
        "representative_snapshot_rst7": str(output_rst7),
        "atom_mask": atom_mask,
    }
    write_json(target_dir / "cluster_manifest.json", manifest)
    return manifest


def parse_cntrl_settings(path: str | Path | None) -> InheritedMDSettings:
    if path is None:
        return InheritedMDSettings()

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    in_cntrl = False
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.split("!")[0].split("#")[0].strip()
        if not line:
            continue
        if line.lower().startswith("&cntrl"):
            in_cntrl = True
            continue
        if in_cntrl and line.startswith("/"):
            break
        if not in_cntrl:
            continue
        for match in _CNTRL_PAIR_PATTERN.finditer(line):
            key = match.group("key").strip().lower()
            value = match.group("value").strip().strip("'").strip('"')
            values[key] = value

    def _get_float(name: str, default: float | None) -> float | None:
        raw = values.get(name)
        if raw is None:
            return default
        return float(raw.replace("d", "e").replace("D", "E"))

    def _get_int(name: str, default: int | None) -> int | None:
        raw = values.get(name)
        if raw is None:
            return default
        return int(float(raw))

    settings = InheritedMDSettings(
        dt_ps=float(_get_float("dt", 0.002) or 0.002),
        temperature_k=float(_get_float("temp0", 300.0) or 300.0),
        pressure_bar=float(_get_float("pres0", 1.0) or 1.0),
        cut_angstrom=float(_get_float("cut", 8.0) or 8.0),
        ntb=int(_get_int("ntb", 2) or 2),
        ntp=int(_get_int("ntp", 1) or 1),
        ntc=int(_get_int("ntc", 2) or 2),
        ntf=int(_get_int("ntf", 2) or 2),
        ntt=int(_get_int("ntt", 3) or 3),
        gamma_ln=float(_get_float("gamma_ln", 2.0) or 2.0),
        ntpr=int(_get_int("ntpr", 1000) or 1000),
        ntwx=int(_get_int("ntwx", 1000) or 1000),
        ntwr=int(_get_int("ntwr", 1000) or 1000),
        ioutfm=int(_get_int("ioutfm", 1) or 1),
        iwrap=int(_get_int("iwrap", 0) or 0),
        barostat=_get_int("barostat", None),
        taup=_get_float("taup", None),
        tempi_k=_get_float("tempi", None),
    )
    return settings


def _coerce_path_string(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def record_site_analysis(
    *,
    output_path: str | Path,
    candidates: list[MetalSiteCandidate],
    assessments: list[SiteStabilityAssessment],
) -> Path:
    payload: dict[str, Any] = {
        "candidates": [item.to_dict() for item in candidates],
        "assessments": [item.to_dict() for item in assessments],
    }
    return write_json(output_path, payload)
