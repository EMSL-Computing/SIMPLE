from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from amber_metallo.config import RespApplyMode, SlurmConfig
from amber_metallo.qm.resp_fit import (
    equality_groups_from_pairs,
    equality_pairs_from_group_payload,
    load_resp_charge_result,
    render_runtime_resp_fit_script,
)
from amber_metallo.qm.slurm import render_resp_slurm_script, render_tahoma_resp_script
from amber_metallo.reporting import write_json


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_RESP_TEMPLATE_PATH = _TEMPLATE_DIR / "resp_job.nw.tmpl"

_COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}
_ELEMENT_SYMBOLS = {
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
}

_RESP_PRESETS = {
    "ptmpsi_default": {
        "preset": "ptmpsi_default",
        "label": "PTMPSI default",
        "geometry_mode": "use_loaded_geometry",
        "functional": "r2scan",
        "basis": "def2-tzvp",
        "memory_mb": 2000,
        "grid": "fine",
        "maxiter": 200,
        "resp_hf_block": True,
    },
    "fast_xtb_assisted": {
        "preset": "fast_xtb_assisted",
        "label": "Fast xTB-assisted",
        "geometry_mode": "xtb_preopt",
        "functional": "r2scan",
        "basis": "def2-svp",
        "memory_mb": 1500,
        "grid": "medium",
        "maxiter": 120,
        "resp_hf_block": False,
    },
    "single_point_loaded_geometry": {
        "preset": "single_point_loaded_geometry",
        "label": "Single-point on loaded geometry",
        "geometry_mode": "use_loaded_geometry",
        "functional": "r2scan",
        "basis": "def2-svp",
        "memory_mb": 1500,
        "grid": "medium",
        "maxiter": 120,
        "resp_hf_block": False,
    },
}
QM_GEOMETRY_MODE_OPTIONS = (
    ("use_loaded_geometry", "Use loaded geometry"),
    ("xtb_preopt", "xTB pre-opt + DFT optimize"),
    ("dft_geo_opt", "DFT optimize"),
)
QM_FUNCTIONAL_OPTIONS = (
    ("r2scan", "R2SCAN"),
    ("pbe0", "PBE0"),
    ("b3lyp", "B3LYP"),
    ("m06-2x", "M06-2X"),
    ("hf", "HF"),
)
QM_BASIS_OPTIONS = (
    ("def2-svp", "def2-SVP"),
    ("def2-tzvp", "def2-TZVP"),
    ("6-31g*", "6-31G*"),
    ("6-311g*", "6-311G*"),
)
QM_GRID_OPTIONS = (
    ("medium", "Medium"),
    ("fine", "Fine"),
    ("xfine", "XFine"),
)
QM_DEFAULT_GEOMETRY_MODE = "use_loaded_geometry"
QM_DEFAULT_DFT_FUNCTIONAL = "r2scan"
QM_DEFAULT_DFT_BASIS = "def2-tzvp"
QM_DEFAULT_RESP_FUNCTIONAL = "hf"
QM_DEFAULT_RESP_BASIS = "6-31g*"
QM_DEFAULT_MEMORY_MB = 2000
QM_DEFAULT_GRID = "fine"
QM_DEFAULT_MAXITER = 200
QM_DEFAULT_XTB_ACC = 0.1
_GEOMETRY_MODE_KEYS = {key for key, _label in QM_GEOMETRY_MODE_OPTIONS}
_FUNCTIONAL_KEYS = {key for key, _label in QM_FUNCTIONAL_OPTIONS}
_BASIS_KEYS = {key for key, _label in QM_BASIS_OPTIONS}
_GRID_KEYS = {key for key, _label in QM_GRID_OPTIONS}
_RESP_JOB_DIR_PREFIX = "RESP_JOBS_"
AUTO_GROUP_MODE_HYDROGEN_ONLY = "hydrogen_only"
AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY = "hydrogen_and_symmetry"
AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY = "connectivity"
AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM = "graph_automorphism"
AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL = "extended_huckel"
AUTO_GROUP_GRAPH_METHOD_HYBRID_HUCKEL = "hybrid_huckel"
AUTO_GROUP_GRAPH_METHODS = {
    AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY,
    AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM,
    AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL,
    AUTO_GROUP_GRAPH_METHOD_HYBRID_HUCKEL,
}
_SUPPORTED_RESP_METALS = {
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
_HUCKEL_HEAVY_HEAVY_THRESHOLD = 0.30
_HUCKEL_HYDROGEN_THRESHOLD = 0.20
_GRAPH_BOND_CACHE: dict[tuple[str, str], tuple[list["MoleculeBond"], str | None]] = {}


@dataclass(slots=True)
class MoleculeAtom:
    index: int
    name: str
    element: str
    x: float
    y: float
    z: float
    charge: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MoleculeBond:
    first: int
    second: int
    order: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MoleculeData:
    source_file: str
    source_format: str
    atoms: list[MoleculeAtom]
    bonds: list[MoleculeBond]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "source_format": self.source_format,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "bonds": [bond.to_dict() for bond in self.bonds],
        }


@dataclass(slots=True)
class RespJobCandidate:
    job_dir: Path
    manifest_path: Path
    payload: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return self.job_dir / "output"

    @property
    def charge_result_path(self) -> Path | None:
        for candidate in (
            self.output_dir / "resp_charges.json",
            self.output_dir / "resp_charges.txt",
        ):
            if candidate.exists():
                return candidate
        return None

    @property
    def has_output_artifacts(self) -> bool:
        if not self.output_dir.exists():
            return False
        patterns = (
            "*.esp",
            "*.grid",
            "*.qrs",
            "resp_job.log",
            "resp_charges.json",
            "resp_charges.txt",
        )
        return any(any(self.output_dir.glob(pattern)) for pattern in patterns)

    @property
    def can_materialize_charge_result(self) -> bool:
        if self.completed:
            return True
        grid_path = _resolve_resp_grid_path(self.output_dir)
        xyz_path = _resolve_resp_xyz_path(output_dir=self.output_dir, grid_path=grid_path, job_dir=self.job_dir)
        return grid_path is not None and xyz_path is not None

    @property
    def ready_to_continue(self) -> bool:
        return self.completed or self.can_materialize_charge_result

    @property
    def completed(self) -> bool:
        return self.charge_result_path is not None


def _normalize_element(token: str) -> str:
    cleaned = "".join(character for character in token.strip() if character.isalpha())
    if not cleaned:
        return "C"
    if len(cleaned) == 1:
        return cleaned.upper()
    return cleaned[0].upper() + cleaned[1:].lower()


def _guess_element_from_name(name: str) -> str:
    cleaned = "".join(character for character in name.strip() if character.isalpha())
    if not cleaned:
        return "C"
    if len(cleaned) >= 2:
        two_letter = cleaned[:2].title()
        if two_letter in _ELEMENT_SYMBOLS:
            return two_letter
    one_letter = cleaned[0].upper()
    if one_letter in _ELEMENT_SYMBOLS:
        return one_letter
    return one_letter


def _infer_mol2_element(atom_name: str, atom_type: str) -> str:
    name_guess = _guess_element_from_name(atom_name)
    if name_guess in _ELEMENT_SYMBOLS:
        return name_guess

    type_root = "".join(character for character in atom_type.split(".", maxsplit=1)[0] if character.isalpha())
    if not type_root:
        return name_guess
    if len(type_root) == 1:
        return type_root.upper()

    if type_root[0].isupper() and type_root[1].islower():
        candidate = type_root[:2].title()
        if candidate in _ELEMENT_SYMBOLS:
            return candidate

    unambiguous_two_letter = {
        "cl", "br", "si", "mg", "zn", "fe", "cu", "ni", "mn", "co", "cr", "li", "al",
    }
    lowered = type_root.lower()
    if lowered in unambiguous_two_letter:
        return lowered.title()

    return type_root[0].upper()


def _infer_bonds(atoms: list[MoleculeAtom]) -> list[MoleculeBond]:
    bonds: list[MoleculeBond] = []
    for index, first_atom in enumerate(atoms):
        for second_atom in atoms[index + 1 :]:
            radius_a = _COVALENT_RADII.get(first_atom.element.upper(), 0.77)
            radius_b = _COVALENT_RADII.get(second_atom.element.upper(), 0.77)
            cutoff = 1.25 * (radius_a + radius_b)
            distance = math.sqrt(
                (first_atom.x - second_atom.x) ** 2
                + (first_atom.y - second_atom.y) ** 2
                + (first_atom.z - second_atom.z) ** 2
            )
            if distance <= cutoff:
                bonds.append(MoleculeBond(first=first_atom.index, second=second_atom.index, order=1))
    return bonds


def _parse_sdf(path: Path) -> MoleculeData:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError(f"SDF file is too short: {path}")
    counts = lines[3]
    atom_count = int(counts[0:3].strip())
    bond_count = int(counts[3:6].strip())
    atoms: list[MoleculeAtom] = []
    bonds: list[MoleculeBond] = []
    atom_lines = lines[4 : 4 + atom_count]
    bond_lines = lines[4 + atom_count : 4 + atom_count + bond_count]
    for offset, line in enumerate(atom_lines, start=1):
        atoms.append(
            MoleculeAtom(
                index=offset,
                name=f"{line[31:34].strip() or 'X'}{offset}",
                element=_normalize_element(line[31:34]),
                x=float(line[0:10].strip()),
                y=float(line[10:20].strip()),
                z=float(line[20:30].strip()),
            )
        )
    for line in bond_lines:
        bonds.append(
            MoleculeBond(
                first=int(line[0:3].strip()),
                second=int(line[3:6].strip()),
                order=max(1, int(line[6:9].strip() or "1")),
            )
        )
    return MoleculeData(source_file=str(path.resolve()), source_format="sdf", atoms=atoms, bonds=bonds)


def _parse_mol2(path: Path) -> MoleculeData:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms: list[MoleculeAtom] = []
    bonds: list[MoleculeBond] = []
    section: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            section = stripped
            continue
        if not stripped:
            continue
        if section == "@<TRIPOS>ATOM":
            tokens = stripped.split()
            atoms.append(
                MoleculeAtom(
                    index=int(tokens[0]),
                    name=tokens[1],
                    element=_infer_mol2_element(tokens[1], tokens[5]),
                    x=float(tokens[2]),
                    y=float(tokens[3]),
                    z=float(tokens[4]),
                    charge=float(tokens[8]) if len(tokens) > 8 and _looks_like_float(tokens[8]) else None,
                )
            )
        elif section == "@<TRIPOS>BOND":
            tokens = stripped.split()
            order_token = tokens[3].lower()
            order = 1
            if order_token.startswith("2"):
                order = 2
            elif order_token.startswith("3"):
                order = 3
            bonds.append(MoleculeBond(first=int(tokens[1]), second=int(tokens[2]), order=order))
    if not atoms:
        raise ValueError(f"MOL2 file did not contain any atoms: {path}")
    return MoleculeData(source_file=str(path.resolve()), source_format="mol2", atoms=atoms, bonds=bonds)


def _looks_like_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _contains_alpha(token: str) -> bool:
    return any(character.isalpha() for character in str(token))


def _parse_pdb_atom_line(line: str) -> MoleculeAtom:
    tokens = line.split()
    atom_index = int(line[6:11].strip())
    atom_name = line[12:16].strip() or f"A{atom_index}"
    if len(tokens) >= 3 and not _looks_like_float(tokens[2]):
        atom_name = tokens[2].strip() or atom_name
    try:
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError:
        coord_start: int | None = None
        for index in range(2, max(2, len(tokens) - 4)):
            if (
                _looks_like_float(tokens[index])
                and _looks_like_float(tokens[index + 1])
                and _looks_like_float(tokens[index + 2])
                and _looks_like_float(tokens[index + 3])
                and _looks_like_float(tokens[index + 4])
                and (index + 5 >= len(tokens) or not _looks_like_float(tokens[index + 5]))
            ):
                coord_start = index
                break
        if coord_start is None:
            for index in range(2, max(2, len(tokens) - 2)):
                if (
                    _looks_like_float(tokens[index])
                    and _looks_like_float(tokens[index + 1])
                    and _looks_like_float(tokens[index + 2])
                ):
                    coord_start = index
                    break
        if coord_start is None:
            raise ValueError(f"Could not locate x/y/z coordinates in PDB atom line: {line.rstrip()}")
        x = float(tokens[coord_start])
        y = float(tokens[coord_start + 1])
        z = float(tokens[coord_start + 2])

    element_field = line[76:78].strip() if len(line) >= 78 else ""
    if not _contains_alpha(element_field) and tokens:
        trailing_token = tokens[-1]
        if not _looks_like_float(trailing_token) and _contains_alpha(trailing_token):
            element_field = trailing_token
    if not _contains_alpha(element_field):
        element_field = ""
    element = _normalize_element(element_field) if element_field else _guess_element_from_name(atom_name)
    return MoleculeAtom(
        index=atom_index,
        name=atom_name,
        element=element,
        x=x,
        y=y,
        z=z,
    )


def _parse_pdb(path: Path) -> MoleculeData:
    atoms: list[MoleculeAtom] = []
    bonds: list[MoleculeBond] = []
    seen_pairs: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(_parse_pdb_atom_line(line))
        elif line.startswith("CONECT"):
            tokens = line.split()
            if len(tokens) < 3:
                continue
            first = int(tokens[1])
            for token in tokens[2:]:
                second = int(token)
                pair = tuple(sorted((first, second)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                bonds.append(MoleculeBond(first=pair[0], second=pair[1], order=1))
    if not atoms:
        raise ValueError(f"PDB file did not contain any atoms: {path}")
    if not bonds:
        bonds = _infer_bonds(atoms)
    return MoleculeData(source_file=str(path.resolve()), source_format="pdb", atoms=atoms, bonds=bonds)


def load_molecule(path: str | Path) -> MoleculeData:
    source = Path(path).expanduser().resolve()
    extension = source.suffix.lower()
    if extension in {".sdf", ".sd"}:
        return _parse_sdf(source)
    if extension == ".mol2":
        return _parse_mol2(source)
    if extension == ".pdb":
        return _parse_pdb(source)
    raise ValueError(f"Unsupported RESP source format: {source.suffix}")


def molecule_fingerprint(
    source_file: str | Path,
    *,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
) -> str:
    source = Path(source_file).expanduser().resolve()
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(residue_name.strip().upper().encode("utf-8"))
    digest.update(str(int(net_charge)).encode("ascii"))
    digest.update(str(int(multiplicity)).encode("ascii"))
    return digest.hexdigest()[:16]


def _molecule_graph_digest(molecule: MoleculeData) -> str:
    digest = hashlib.sha256()
    for atom in molecule.atoms:
        digest.update(f"A|{atom.index}|{atom.element}|{atom.x:.6f}|{atom.y:.6f}|{atom.z:.6f}\n".encode("utf-8"))
    for bond in sorted(molecule.bonds, key=lambda item: (min(item.first, item.second), max(item.first, item.second))):
        digest.update(f"B|{bond.first}|{bond.second}|{int(bond.order or 1)}\n".encode("utf-8"))
    return digest.hexdigest()[:24]


def _normalize_auto_group_graph_method(method: str | None) -> str:
    selected = str(method or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY).strip().lower()
    if selected in {"automorphism", "graph-automorphism", "graph_automorphism", "exact", "exact-graph-symmetry", "exact_graph_symmetry"}:
        return AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM
    if selected in {"huckel", "extended-huckel", "extended_huckel"}:
        return AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL
    if selected in {"hybrid", "hybrid-huckel", "hybrid_huckel"}:
        return AUTO_GROUP_GRAPH_METHOD_HYBRID_HUCKEL
    if selected in AUTO_GROUP_GRAPH_METHODS:
        return selected
    return AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY


def _rdkit_bond_type(order: int):
    from rdkit import Chem

    normalized = max(1, int(order or 1))
    if normalized >= 3:
        return Chem.BondType.TRIPLE
    if normalized == 2:
        return Chem.BondType.DOUBLE
    return Chem.BondType.SINGLE


def _huckel_order_class(value: float) -> int:
    if value >= 2.4:
        return 3
    if value >= 1.4:
        return 2
    return 1


def _is_hydrogen_element(element: str | None) -> bool:
    return str(element or "").strip().upper() == "H"


def _extended_huckel_bonds(molecule: MoleculeData) -> list[MoleculeBond]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdEHTTools
        from rdkit.Geometry import Point3D
    except Exception as exc:  # pragma: no cover - depends on optional runtime dependency
        raise RuntimeError("RDKit with Extended Hueckel support is not available.") from exc

    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    rdkit_index_for_atom: dict[int, int] = {}
    atom_index_for_rdkit: dict[int, int] = {}
    editable = Chem.RWMol()
    for atom in molecule.atoms:
        rd_atom = Chem.Atom(str(atom.element or "C").title())
        rd_atom.SetNoImplicit(True)
        rdkit_index = int(editable.AddAtom(rd_atom))
        rdkit_index_for_atom[int(atom.index)] = rdkit_index
        atom_index_for_rdkit[rdkit_index] = int(atom.index)
    for bond in molecule.bonds:
        first = rdkit_index_for_atom.get(int(bond.first))
        second = rdkit_index_for_atom.get(int(bond.second))
        if first is None or second is None or first == second:
            continue
        if editable.GetBondBetweenAtoms(first, second) is None:
            editable.AddBond(first, second, _rdkit_bond_type(int(bond.order or 1)))
    rdkit_molecule = editable.GetMol()
    conformer = Chem.Conformer(len(molecule.atoms))
    for atom in molecule.atoms:
        conformer.SetAtomPosition(
            rdkit_index_for_atom[int(atom.index)],
            Point3D(float(atom.x), float(atom.y), float(atom.z)),
        )
    rdkit_molecule.AddConformer(conformer, assignId=True)
    try:
        ok, result = rdEHTTools.RunMol(rdkit_molecule)
    except Exception as exc:  # pragma: no cover - depends on RDKit build details
        raise RuntimeError("Extended Hueckel bond-order calculation failed.") from exc
    if not ok:
        raise RuntimeError("Extended Hueckel bond-order calculation did not converge.")

    values = list(result.GetReducedOverlapPopulationMatrix())
    atom_count = len(molecule.atoms)

    def matrix_value(first: int, second: int) -> float:
        if len(values) == atom_count * atom_count:
            return abs(float(values[(first * atom_count) + second]))
        if len(values) == (atom_count * (atom_count + 1)) // 2:
            high = max(first, second)
            low = min(first, second)
            return abs(float(values[(high * (high + 1) // 2) + low]))
        raise RuntimeError("Extended Hueckel returned an unexpected bond-order matrix shape.")

    bonds: list[MoleculeBond] = []
    for first_rdkit in range(atom_count):
        for second_rdkit in range(first_rdkit + 1, atom_count):
            first_atom = atom_lookup[atom_index_for_rdkit[first_rdkit]]
            second_atom = atom_lookup[atom_index_for_rdkit[second_rdkit]]
            threshold = (
                _HUCKEL_HYDROGEN_THRESHOLD
                if _is_hydrogen_element(first_atom.element) or _is_hydrogen_element(second_atom.element)
                else _HUCKEL_HEAVY_HEAVY_THRESHOLD
            )
            value = matrix_value(first_rdkit, second_rdkit)
            if value >= threshold:
                bonds.append(
                    MoleculeBond(
                        first=int(first_atom.index),
                        second=int(second_atom.index),
                        order=_huckel_order_class(value),
                    )
                )
    if not bonds:
        raise RuntimeError("Extended Hueckel did not identify any graph edges.")
    return bonds


def _graph_bonds_for_auto_group_method(
    molecule: MoleculeData,
    method: str | None,
) -> tuple[list[MoleculeBond], str, str | None]:
    selected_method = _normalize_auto_group_graph_method(method)
    if selected_method in {AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY, AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM}:
        return list(molecule.bonds), selected_method, None

    cache_key = (_molecule_graph_digest(molecule), selected_method)
    cached = _GRAPH_BOND_CACHE.get(cache_key)
    if cached is not None:
        bonds, warning = cached
        return list(bonds), selected_method, warning

    try:
        huckel_bonds = _extended_huckel_bonds(molecule)
    except RuntimeError as exc:
        warning = f"{exc} Falling back to connectivity-based auto groups."
        bonds = list(molecule.bonds)
        _GRAPH_BOND_CACHE[cache_key] = (bonds, warning)
        return list(bonds), selected_method, warning

    if selected_method == AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL:
        bonds = huckel_bonds
    else:
        element_lookup = {int(atom.index): atom.element for atom in molecule.atoms}

        def has_hydrogen_endpoint(bond: MoleculeBond) -> bool:
            return (
                _is_hydrogen_element(element_lookup.get(int(bond.first)))
                or _is_hydrogen_element(element_lookup.get(int(bond.second)))
            )

        current_hydrogen_edges = [
            bond
            for bond in molecule.bonds
            if has_hydrogen_endpoint(bond)
        ]
        heavy_pairs = {
            tuple(sorted((int(bond.first), int(bond.second)))): bond
            for bond in huckel_bonds
            if not has_hydrogen_endpoint(bond)
        }
        for bond in current_hydrogen_edges:
            heavy_pairs.setdefault(tuple(sorted((int(bond.first), int(bond.second)))), bond)
        bonds = list(heavy_pairs.values())
    _GRAPH_BOND_CACHE[cache_key] = (bonds, None)
    return list(bonds), selected_method, None


def _adjacency_map(
    molecule: MoleculeData,
    *,
    bonds: list[MoleculeBond] | tuple[MoleculeBond, ...] | None = None,
) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = {atom.index: [] for atom in molecule.atoms}
    for bond in list(bonds) if bonds is not None else molecule.bonds:
        order = max(1, int(bond.order or 1))
        adjacency[bond.first].append((bond.second, order))
        adjacency[bond.second].append((bond.first, order))
    return adjacency


def _is_supported_resp_metal_element(element: str | None) -> bool:
    return str(element or "").strip().title() in _SUPPORTED_RESP_METALS


def supported_metal_atom_indices(molecule: MoleculeData) -> set[int]:
    return {
        int(atom.index)
        for atom in molecule.atoms
        if _is_supported_resp_metal_element(atom.element)
    }


def molecule_contains_supported_metal(molecule: MoleculeData) -> bool:
    return bool(supported_metal_atom_indices(molecule))


def default_auto_group_exclusion_indices(molecule: MoleculeData) -> set[int]:
    adjacency = _adjacency_map(molecule)
    metal_indices = supported_metal_atom_indices(molecule)
    excluded = set(metal_indices)
    for metal_index in metal_indices:
        excluded.update(neighbor for neighbor, _order in adjacency.get(metal_index, []))
    return excluded


def _hydrogen_constraint_groups(
    molecule: MoleculeData,
    *,
    excluded_atom_indices: set[int] | None = None,
    graph_bonds: list[MoleculeBond] | None = None,
) -> list[dict[str, object]]:
    adjacency = _adjacency_map(molecule, bonds=graph_bonds)
    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    excluded = set(excluded_atom_indices or set())
    groups: list[dict[str, object]] = []
    group_id = 1
    for atom in molecule.atoms:
        if atom.index in excluded or atom.element.upper() == "H":
            continue
        hydrogens = [
            neighbor
            for neighbor, _order in adjacency.get(atom.index, [])
            if neighbor not in excluded and atom_lookup[neighbor].element.upper() == "H"
        ]
        if len(hydrogens) < 2:
            continue
        groups.append(
            {
                "group_id": group_id,
                "label": f"H attached to {atom.name}",
                "atom_indices": sorted(hydrogens),
                "auto": True,
            }
        )
        group_id += 1
    return groups


class _AtomUnionFind:
    def __init__(self, items: list[int]):
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[int(item)]
        if parent != int(item):
            parent = self.find(parent)
            self.parent[int(item)] = parent
        return parent

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _atom_charge_key(charge: float | None) -> float | None:
    return None if charge is None else round(float(charge), 6)


def _automorphism_constraint_groups(
    molecule: MoleculeData,
    *,
    excluded_atom_indices: set[int] | None = None,
    graph_bonds: list[MoleculeBond] | None = None,
    max_automorphisms: int = 100000,
) -> list[dict[str, object]]:
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise RuntimeError("NetworkX is not available for exact graph symmetry.") from exc

    atom_lookup = {int(atom.index): atom for atom in molecule.atoms}
    graph = nx.Graph()
    for atom in molecule.atoms:
        graph.add_node(
            int(atom.index),
            element=str(atom.element or "").upper(),
            charge=_atom_charge_key(atom.charge),
        )
    for bond in list(graph_bonds) if graph_bonds is not None else molecule.bonds:
        first = int(bond.first)
        second = int(bond.second)
        if first not in atom_lookup or second not in atom_lookup or first == second:
            continue
        order = max(1, int(bond.order or 1))
        if graph.has_edge(first, second):
            graph[first][second]["order"] = max(int(graph[first][second].get("order", 1)), order)
        else:
            graph.add_edge(first, second, order=order)

    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["element", "charge"],
        ["", None],
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("order", 1)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=node_match,
        edge_match=edge_match,
    )

    union = _AtomUnionFind([int(atom.index) for atom in molecule.atoms])
    for count, mapping in enumerate(matcher.isomorphisms_iter(), start=1):
        for source, target in mapping.items():
            union.union(int(source), int(target))
        if count >= max_automorphisms:
            break

    grouped: dict[int, list[int]] = {}
    for atom in molecule.atoms:
        grouped.setdefault(union.find(int(atom.index)), []).append(int(atom.index))

    excluded = set(excluded_atom_indices or set())
    groups: list[dict[str, object]] = []
    group_id = 1
    for atom_indices in sorted(grouped.values(), key=lambda value: (len(value) < 2, value[0])):
        eligible_atom_indices = sorted(atom_index for atom_index in atom_indices if atom_index not in excluded)
        if len(eligible_atom_indices) < 2:
            continue
        first_atom = atom_lookup[eligible_atom_indices[0]]
        element = first_atom.element.upper()
        groups.append(
            {
                "group_id": group_id,
                "label": f"Symmetry-equivalent {element} atoms",
                "atom_indices": eligible_atom_indices,
                "auto": True,
            }
        )
        group_id += 1
    return groups


def _symmetry_constraint_groups(
    molecule: MoleculeData,
    *,
    excluded_atom_indices: set[int] | None = None,
    graph_bonds: list[MoleculeBond] | None = None,
) -> list[dict[str, object]]:
    adjacency = _adjacency_map(molecule, bonds=graph_bonds)
    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    excluded = set(excluded_atom_indices or set())
    labels = {
        atom.index: f"{atom.element.upper()}|d{len(adjacency.get(atom.index, []))}"
        for atom in molecule.atoms
    }
    for _ in range(max(1, len(molecule.atoms))):
        signatures = {}
        for atom in molecule.atoms:
            neighbor_signature = tuple(
                sorted(
                    f"{order}:{labels[neighbor]}"
                    for neighbor, order in adjacency.get(atom.index, [])
                )
            )
            signatures[atom.index] = (labels[atom.index], neighbor_signature)
        palette = {
            signature: f"S{index}"
            for index, signature in enumerate(sorted(set(signatures.values())), start=1)
        }
        next_labels = {atom_index: palette[signature] for atom_index, signature in signatures.items()}
        if next_labels == labels:
            break
        labels = next_labels

    grouped: dict[str, list[int]] = {}
    for atom in molecule.atoms:
        grouped.setdefault(labels[atom.index], []).append(atom.index)

    groups: list[dict[str, object]] = []
    group_id = 1
    for atom_indices in sorted(grouped.values(), key=lambda value: (len(value) < 2, value[0])):
        eligible_atom_indices = sorted(atom_index for atom_index in atom_indices if atom_index not in excluded)
        if len(eligible_atom_indices) < 2:
            continue
        first_atom = atom_lookup[eligible_atom_indices[0]]
        element = first_atom.element.upper()
        label = f"Symmetry-equivalent {element} atoms"
        groups.append(
            {
                "group_id": group_id,
                "label": label,
                "atom_indices": eligible_atom_indices,
                "auto": True,
            }
        )
        group_id += 1
    return groups


def suggest_group_constraints(
    molecule: MoleculeData,
    *,
    auto_group_mode: str = AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY,
    auto_group_graph_method: str = AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY,
    excluded_atom_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict[str, object]:
    adjacency: dict[int, list[int]] = {atom.index: [] for atom in molecule.atoms}
    atom_lookup = {atom.index: atom for atom in molecule.atoms}
    for bond in molecule.bonds:
        adjacency[bond.first].append(bond.second)
        adjacency[bond.second].append(bond.first)

    selected_mode = str(auto_group_mode or AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY)
    graph_bonds, selected_graph_method, graph_warning = _graph_bonds_for_auto_group_method(
        molecule,
        auto_group_graph_method,
    )
    auto_exclusions = (
        set(int(index) for index in excluded_atom_indices)
        if excluded_atom_indices is not None
        else default_auto_group_exclusion_indices(molecule)
    )
    if selected_mode == AUTO_GROUP_MODE_HYDROGEN_ONLY:
        groups = _hydrogen_constraint_groups(
            molecule,
            excluded_atom_indices=auto_exclusions,
            graph_bonds=graph_bonds,
        )
    elif selected_graph_method == AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM:
        try:
            groups = _automorphism_constraint_groups(
                molecule,
                excluded_atom_indices=auto_exclusions,
                graph_bonds=graph_bonds,
            )
        except RuntimeError as exc:
            graph_warning = f"{exc} Falling back to connectivity-based symmetry groups."
            groups = _symmetry_constraint_groups(
                molecule,
                excluded_atom_indices=auto_exclusions,
                graph_bonds=graph_bonds,
            )
    else:
        groups = _symmetry_constraint_groups(
            molecule,
            excluded_atom_indices=auto_exclusions,
            graph_bonds=graph_bonds,
        )
    atom_group_ids: dict[int, int] = {}
    for group in groups:
        group_id = int(group["group_id"])
        for atom_index in group["atom_indices"]:
            atom_group_ids[int(atom_index)] = group_id

    return {
        "auto_group_mode": selected_mode,
        "auto_group_graph_method": selected_graph_method,
        "auto_group_graph_warning": graph_warning or "",
        "auto_group_excluded_atom_indices": sorted(auto_exclusions),
        "auto_group_exclusion_reason": (
            "Supported metal atoms and atoms directly bonded to those metals are omitted from auto-generated "
            "RESP equality groups. They remain selectable for manual grouping with their original atom indices."
            if auto_exclusions
            else ""
        ),
        "atom_count": len(molecule.atoms),
        "atoms": [
            {
                "index": atom.index,
                "name": atom.name,
                "element": atom.element,
                "group_id": atom_group_ids.get(atom.index),
            }
            for atom in molecule.atoms
        ],
        "groups": groups,
    }


def _render_xyz(molecule: MoleculeData) -> str:
    lines = [str(len(molecule.atoms)), "SIMPLE RESP job"]
    for atom in molecule.atoms:
        lines.append(
            f"{atom.element:<2s} {atom.x: .8f} {atom.y: .8f} {atom.z: .8f}"
        )
    return "\n".join(lines) + "\n"


def render_preview_mol2(molecule: MoleculeData, *, residue_name: str) -> str:
    lines = [
        "@<TRIPOS>MOLECULE",
        residue_name,
        f"{len(molecule.atoms)} {len(molecule.bonds)} 1 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in molecule.atoms:
        lines.append(
            f"{atom.index:>7d} "
            f"{atom.name:<8s} "
            f"{atom.x:>10.4f} "
            f"{atom.y:>10.4f} "
            f"{atom.z:>10.4f} "
            f"{atom.element:<10s} "
            f"{1:>4d} "
            f"{residue_name:<8s} "
            f"{float(atom.charge or 0.0):>10.6f}"
        )
    lines.append("@<TRIPOS>BOND")
    for bond_index, bond in enumerate(molecule.bonds, start=1):
        lines.append(f"{bond_index:>6d} {bond.first:>4d} {bond.second:>4d} {bond.order}")
    lines.extend(
        [
            "@<TRIPOS>SUBSTRUCTURE",
            f"{1:>6d} {residue_name:<8s} {1:>4d}",
            "",
        ]
    )
    return "\n".join(lines)


def _coerce_int(value: object, *, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None:
        return max(parsed, minimum)
    return parsed


def _coerce_float(value: object, *, default: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if minimum is not None:
        return max(parsed, minimum)
    return parsed


def _normalize_functional(value: object, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token == "hfexch 1.0":
        return "hf"
    if token in _FUNCTIONAL_KEYS:
        return token
    return default


def _normalize_basis(value: object, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token in _BASIS_KEYS:
        return token
    return default


def _normalize_grid(value: object, *, default: str) -> str:
    token = str(value or default).strip().lower()
    if token in _GRID_KEYS:
        return token
    return default


def _normalize_geometry_mode(value: object, *, default: str = QM_DEFAULT_GEOMETRY_MODE) -> str:
    token = str(value or default).strip().lower()
    if token in _GEOMETRY_MODE_KEYS:
        return token
    return default


def _option_label(value: str, options: tuple[tuple[str, str], ...]) -> str:
    for option_key, option_label in options:
        if option_key == value:
            return option_label
    return value


def geometry_mode_uses_dft_optimization(mode: str) -> bool:
    return _normalize_geometry_mode(mode) in {"xtb_preopt", "dft_geo_opt"}


def geometry_mode_uses_xtb_preopt(mode: str) -> bool:
    return _normalize_geometry_mode(mode) == "xtb_preopt"


def _base_qm_settings(*, net_charge: int, multiplicity: int) -> dict[str, Any]:
    return {
        "label": "",
        "net_charge": int(net_charge),
        "multiplicity": int(multiplicity),
        "resources": {
            "memory_mb": QM_DEFAULT_MEMORY_MB,
            "grid": QM_DEFAULT_GRID,
            "maxiter": QM_DEFAULT_MAXITER,
        },
        "geometry": {
            "mode": QM_DEFAULT_GEOMETRY_MODE,
            "xtb_preopt": {
                "acc": QM_DEFAULT_XTB_ACC,
            },
            "dft_optimization": {
                "functional": QM_DEFAULT_DFT_FUNCTIONAL,
                "basis": QM_DEFAULT_DFT_BASIS,
            },
        },
        "resp": {
            "same_as_dft_optimization": False,
            "functional": QM_DEFAULT_RESP_FUNCTIONAL,
            "basis": QM_DEFAULT_RESP_BASIS,
        },
    }


def _qm_settings_label(qm_settings: dict[str, Any]) -> str:
    geometry = dict(qm_settings.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    resp = dict(qm_settings.get("resp") or {})
    geometry_mode = _normalize_geometry_mode(geometry.get("mode"), default=QM_DEFAULT_GEOMETRY_MODE)
    geometry_label = _option_label(geometry_mode, QM_GEOMETRY_MODE_OPTIONS)
    dft_label = (
        f"{_option_label(str(dft_optimization.get('functional') or QM_DEFAULT_DFT_FUNCTIONAL), QM_FUNCTIONAL_OPTIONS)}"
        f" / {_option_label(str(dft_optimization.get('basis') or QM_DEFAULT_DFT_BASIS), QM_BASIS_OPTIONS)}"
    )
    if geometry_mode == "use_loaded_geometry":
        geometry_summary = geometry_label
    else:
        geometry_summary = f"{geometry_label} ({dft_label})"

    if bool(resp.get("same_as_dft_optimization")):
        resp_summary = "RESP matches DFT optimization"
    else:
        resp_summary = (
            "RESP "
            f"{_option_label(str(resp.get('functional') or QM_DEFAULT_RESP_FUNCTIONAL), QM_FUNCTIONAL_OPTIONS)}"
            f" / {_option_label(str(resp.get('basis') or QM_DEFAULT_RESP_BASIS), QM_BASIS_OPTIONS)}"
        )
    return f"{geometry_summary}; {resp_summary}"


def normalize_qm_settings(
    qm_settings: dict[str, Any] | None,
    *,
    net_charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    raw = dict(qm_settings or {})
    nested_geometry = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else {}
    nested_resources = raw.get("resources") if isinstance(raw.get("resources"), dict) else {}
    nested_resp = raw.get("resp") if isinstance(raw.get("resp"), dict) else {}
    if not nested_geometry and not nested_resources and not nested_resp:
        preset_key = str(raw.get("preset") or "ptmpsi_default")
        raw = {**dict(_RESP_PRESETS.get(preset_key, _RESP_PRESETS["ptmpsi_default"])), **raw}

    normalized = _base_qm_settings(
        net_charge=_coerce_int(raw.get("net_charge"), default=net_charge),
        multiplicity=_coerce_int(raw.get("multiplicity"), default=multiplicity, minimum=1),
    )

    geometry = dict(raw.get("geometry") or {}) if isinstance(raw.get("geometry"), dict) else {}
    dft_optimization = (
        dict(geometry.get("dft_optimization") or {})
        if isinstance(geometry.get("dft_optimization"), dict)
        else {}
    )
    xtb_preopt = dict(geometry.get("xtb_preopt") or {}) if isinstance(geometry.get("xtb_preopt"), dict) else {}
    resources = dict(raw.get("resources") or {}) if isinstance(raw.get("resources"), dict) else {}
    resp = dict(raw.get("resp") or {}) if isinstance(raw.get("resp"), dict) else {}

    normalized["geometry"]["mode"] = _normalize_geometry_mode(
        geometry.get("mode") or raw.get("geometry_mode"),
        default=QM_DEFAULT_GEOMETRY_MODE,
    )
    normalized["geometry"]["xtb_preopt"]["acc"] = _coerce_float(
        xtb_preopt.get("acc") or raw.get("xtb_acc"),
        default=QM_DEFAULT_XTB_ACC,
        minimum=0.01,
    )
    normalized["geometry"]["dft_optimization"]["functional"] = _normalize_functional(
        dft_optimization.get("functional") or raw.get("functional"),
        default=QM_DEFAULT_DFT_FUNCTIONAL,
    )
    normalized["geometry"]["dft_optimization"]["basis"] = _normalize_basis(
        dft_optimization.get("basis") or raw.get("basis"),
        default=QM_DEFAULT_DFT_BASIS,
    )

    normalized["resources"]["memory_mb"] = _coerce_int(
        resources.get("memory_mb") or raw.get("memory_mb"),
        default=QM_DEFAULT_MEMORY_MB,
        minimum=250,
    )
    normalized["resources"]["grid"] = _normalize_grid(
        resources.get("grid") or raw.get("grid"),
        default=QM_DEFAULT_GRID,
    )
    normalized["resources"]["maxiter"] = _coerce_int(
        resources.get("maxiter") or raw.get("maxiter"),
        default=QM_DEFAULT_MAXITER,
        minimum=10,
    )

    same_as_dft_optimization = resp.get("same_as_dft_optimization")
    if same_as_dft_optimization is None and "resp_hf_block" in raw:
        same_as_dft_optimization = not bool(raw.get("resp_hf_block"))
    normalized["resp"]["same_as_dft_optimization"] = bool(same_as_dft_optimization)
    if normalized["resp"]["same_as_dft_optimization"]:
        normalized["resp"]["functional"] = normalized["geometry"]["dft_optimization"]["functional"]
        normalized["resp"]["basis"] = normalized["geometry"]["dft_optimization"]["basis"]
    else:
        legacy_resp_functional = resp.get("functional")
        legacy_resp_basis = resp.get("basis")
        if legacy_resp_functional is None and bool(raw.get("resp_hf_block")):
            legacy_resp_functional = QM_DEFAULT_RESP_FUNCTIONAL
        if legacy_resp_basis is None and bool(raw.get("resp_hf_block")):
            legacy_resp_basis = QM_DEFAULT_RESP_BASIS
        normalized["resp"]["functional"] = _normalize_functional(
            legacy_resp_functional,
            default=QM_DEFAULT_RESP_FUNCTIONAL,
        )
        normalized["resp"]["basis"] = _normalize_basis(
            legacy_resp_basis,
            default=QM_DEFAULT_RESP_BASIS,
        )

    normalized["label"] = _qm_settings_label(normalized)
    return normalized


def default_qm_settings(*, net_charge: int, multiplicity: int) -> dict[str, object]:
    return normalize_qm_settings(
        {
            "net_charge": int(net_charge),
            "multiplicity": int(multiplicity),
        },
        net_charge=net_charge,
        multiplicity=multiplicity,
    )


def metal_safe_qm_settings(
    qm_settings: dict[str, Any] | None,
    *,
    net_charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    qm = normalize_qm_settings(qm_settings, net_charge=net_charge, multiplicity=multiplicity)
    qm["geometry"]["dft_optimization"]["functional"] = (
        QM_DEFAULT_DFT_FUNCTIONAL
        if qm["geometry"]["dft_optimization"]["functional"] == "hf"
        else qm["geometry"]["dft_optimization"]["functional"]
    )
    if not str(qm["geometry"]["dft_optimization"]["basis"]).lower().startswith("def2-"):
        qm["geometry"]["dft_optimization"]["basis"] = QM_DEFAULT_DFT_BASIS
    qm["resp"]["same_as_dft_optimization"] = True
    qm["resp"]["functional"] = qm["geometry"]["dft_optimization"]["functional"]
    qm["resp"]["basis"] = qm["geometry"]["dft_optimization"]["basis"]
    qm["label"] = _qm_settings_label(qm)
    return qm


def _metal_unsafe_qm_reasons(qm_settings: dict[str, Any]) -> list[str]:
    qm = normalize_qm_settings(
        qm_settings,
        net_charge=int(qm_settings.get("net_charge") or 0),
        multiplicity=int(qm_settings.get("multiplicity") or 1),
    )
    geometry = dict(qm.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    resp = dict(qm.get("resp") or {})
    reasons: list[str] = []
    dft_functional = str(dft_optimization.get("functional") or "").lower()
    dft_basis = str(dft_optimization.get("basis") or "").lower()
    resp_functional = str(resp.get("functional") or "").lower()
    resp_basis = str(resp.get("basis") or "").lower()
    if dft_functional == "hf" or resp_functional == "hf":
        reasons.append("HF is not allowed for supported-metal RESP jobs.")
    if dft_basis and not dft_basis.startswith("def2-"):
        reasons.append(f"Geometry basis '{dft_basis}' is not a def2 basis.")
    if resp_basis and not resp_basis.startswith("def2-"):
        reasons.append(f"RESP basis '{resp_basis}' is not a def2 basis.")
    return reasons


def validate_qm_settings_for_molecule(molecule: MoleculeData, qm_settings: dict[str, Any]) -> None:
    if not molecule_contains_supported_metal(molecule):
        return
    reasons = _metal_unsafe_qm_reasons(qm_settings)
    if not reasons:
        return
    raise ValueError(
        "Full RESP for supported metal-containing structures requires a metal-safe QM preset. "
        "Use a def2 basis with a non-HF DFT functional for both geometry and RESP reference steps. "
        "Blocked settings: " + " ".join(reasons)
    )


def normalize_session_qm_settings(
    session_state: dict[str, Any],
    *,
    net_charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    session_state["qm_settings"] = normalize_qm_settings(
        session_state.get("qm_settings") if isinstance(session_state, dict) else None,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    return session_state


def build_default_session_state(
    molecule: MoleculeData,
    *,
    residue_name: str,
    fingerprint: str,
    net_charge: int,
    multiplicity: int,
    group_payload: dict[str, object] | None = None,
) -> dict[str, Any]:
    qm_settings = default_qm_settings(net_charge=net_charge, multiplicity=multiplicity)
    if molecule_contains_supported_metal(molecule):
        qm_settings = metal_safe_qm_settings(
            qm_settings,
            net_charge=net_charge,
            multiplicity=multiplicity,
        )
    state = {
        "fingerprint": fingerprint,
        "source_file": molecule.source_file,
        "source_format": molecule.source_format,
        "residue_name": residue_name,
        "molecule": molecule.to_dict(),
        "group_constraints": group_payload or suggest_group_constraints(molecule),
        "qm_settings": qm_settings,
        "mol2_preview": render_preview_mol2(molecule, residue_name=residue_name),
    }
    return normalize_session_qm_settings(state, net_charge=net_charge, multiplicity=multiplicity)


def _render_geometry_block(molecule: MoleculeData) -> str:
    return "\n".join(
        f"  {atom.element:<2s} {atom.x: .8f} {atom.y: .8f} {atom.z: .8f}"
        for atom in molecule.atoms
    )


def _aux_basis_for(basis: str) -> str:
    lowered = basis.lower()
    if lowered.startswith("def2-"):
        return 'basis "cd basis" spherical bse\n * library def2-universal-jfit\nend'
    return ""


def _xc_keyword(functional: str) -> str:
    # PBE is used internally only to precondition difficult open-shell metal
    # orbitals before the requested r2SCAN single point; it is not exposed as
    # a selectable final RESP level.
    if str(functional).strip().lower() == "pbe":
        return "pbe"
    normalized = _normalize_functional(functional, default=QM_DEFAULT_DFT_FUNCTIONAL)
    if normalized == "hf":
        return "hfexch 1.0"
    return normalized


def _render_dft_theory_block(
    *,
    functional: str,
    basis: str,
    multiplicity: int,
    grid: str,
    maxiter: int,
    reset_existing: bool = False,
    vectors_input_atomic: bool = False,
    vectors_input_file: str | None = None,
    vectors_output_file: str | None = None,
    convergence_profile: str = "default",
) -> str:
    if convergence_profile == "metal_robust":
        maxiter = max(int(maxiter), 500)
        convergence = " convergence energy 1d-7 damp 40 lshift 0.5 rabuck 30 diis 6"
    elif convergence_profile == "metal_retry":
        maxiter = max(int(maxiter), 1000)
        convergence = " convergence energy 1d-7 damp 70 lshift 1.0 rabuck 80 diis 8"
    else:
        convergence = " convergence energy 1d-7"
    lines: list[str] = []
    if reset_existing:
        lines.extend(
            [
                'unset "dft:cd*"',
                'unset "basis:cd*"',
            ]
        )
    lines.extend(
        [
            'basis "ao basis" spherical',
            f" * library {basis}",
            "end",
        ]
    )
    aux_basis = _aux_basis_for(basis)
    if aux_basis:
        lines.append(aux_basis)
    lines.extend(
        [
            "dft",
            f" mult {int(multiplicity)}",
            f" xc {_xc_keyword(functional)}",
            f" grid {grid} nodisk",
            f" iterations {int(maxiter)}",
            convergence,
        ]
    )
    if vectors_input_file:
        vector_line = f' vectors input "{vectors_input_file}"'
    elif vectors_input_atomic:
        vector_line = " vectors input atomic"
    else:
        vector_line = ""
    if vectors_output_file:
        vector_line += f' output "{vectors_output_file}"'
    if vector_line:
        lines.append(vector_line)
    lines.extend(
        [
            ' noprint "final vectors analysis"',
            "end",
        ]
    )
    return "\n".join(lines)


def _render_optimization_step_block(qm_settings: dict[str, Any]) -> str:
    geometry = dict(qm_settings.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    resources = dict(qm_settings.get("resources") or {})
    geometry_mode = _normalize_geometry_mode(geometry.get("mode"), default=QM_DEFAULT_GEOMETRY_MODE)
    if not geometry_mode_uses_dft_optimization(geometry_mode):
        return ""

    blocks: list[str] = []
    if geometry_mode_uses_xtb_preopt(geometry_mode):
        xtb_preopt = dict(geometry.get("xtb_preopt") or {})
        blocks.append(
            "\n".join(
                [
                    "xtb",
                    f" acc {_coerce_float(xtb_preopt.get('acc'), default=QM_DEFAULT_XTB_ACC, minimum=0.01):g}",
                    "end",
                    "driver",
                    " maxiter 100",
                    "end",
                    "task xtb optimize ignore",
                ]
            )
        )
    blocks.append(
        _render_dft_theory_block(
            functional=str(dft_optimization.get("functional") or QM_DEFAULT_DFT_FUNCTIONAL),
            basis=str(dft_optimization.get("basis") or QM_DEFAULT_DFT_BASIS),
            multiplicity=int(qm_settings.get("multiplicity") or 1),
            grid=str(resources.get("grid") or QM_DEFAULT_GRID),
            maxiter=int(resources.get("maxiter") or QM_DEFAULT_MAXITER),
        )
    )
    blocks.append("\n".join(["driver", " maxiter 100", "end", "task dft optimize ignore"]))
    return "\n\n".join(blocks)


def _resolve_resp_theory(qm_settings: dict[str, Any]) -> tuple[str, str]:
    geometry = dict(qm_settings.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    resp = dict(qm_settings.get("resp") or {})
    if bool(resp.get("same_as_dft_optimization")):
        return (
            str(dft_optimization.get("functional") or QM_DEFAULT_DFT_FUNCTIONAL),
            str(dft_optimization.get("basis") or QM_DEFAULT_DFT_BASIS),
        )
    return (
        str(resp.get("functional") or QM_DEFAULT_RESP_FUNCTIONAL),
        str(resp.get("basis") or QM_DEFAULT_RESP_BASIS),
    )


def _render_resp_reference_block(qm_settings: dict[str, Any], *, convergence_profile: str = "default") -> str:
    resources = dict(qm_settings.get("resources") or {})
    functional, basis = _resolve_resp_theory(qm_settings)
    if convergence_profile == "metal_retry":
        preconditioner = _render_dft_theory_block(
            functional="pbe",
            basis=basis,
            multiplicity=int(qm_settings.get("multiplicity") or 1),
            grid=str(resources.get("grid") or QM_DEFAULT_GRID),
            maxiter=int(resources.get("maxiter") or QM_DEFAULT_MAXITER),
            reset_existing=True,
            vectors_input_atomic=True,
            vectors_output_file="site_resp_precondition.movecs",
            convergence_profile="metal_retry",
        )
        final = _render_dft_theory_block(
            functional=functional,
            basis=basis,
            multiplicity=int(qm_settings.get("multiplicity") or 1),
            grid=str(resources.get("grid") or QM_DEFAULT_GRID),
            maxiter=int(resources.get("maxiter") or QM_DEFAULT_MAXITER),
            reset_existing=True,
            vectors_input_file="site_resp_precondition.movecs",
            convergence_profile="metal_retry",
        )
        return "\n\n".join([preconditioner, "task dft", final, "task dft"])
    return "\n".join(
        [
            _render_dft_theory_block(
                functional=functional,
                basis=basis,
                multiplicity=int(qm_settings.get("multiplicity") or 1),
                grid=str(resources.get("grid") or QM_DEFAULT_GRID),
                maxiter=int(resources.get("maxiter") or QM_DEFAULT_MAXITER),
                reset_existing=True,
                vectors_input_atomic=True,
                convergence_profile=convergence_profile,
            ),
            "task dft",
        ]
    )


def render_nwchem_input(
    molecule: MoleculeData,
    *,
    session_state: dict[str, Any],
    convergence_profile: str = "default",
) -> str:
    raw_qm = dict(session_state.get("qm_settings") or {})
    qm = normalize_qm_settings(
        raw_qm,
        net_charge=int(raw_qm.get("net_charge") or 0),
        multiplicity=int(raw_qm.get("multiplicity") or 1),
    )
    validate_qm_settings_for_molecule(molecule, qm)
    template = _RESP_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        title=f"RESP_{session_state['residue_name']}",
        memory_mb=int((qm.get("resources") or {}).get("memory_mb") or QM_DEFAULT_MEMORY_MB),
        geometry_block=_render_geometry_block(molecule),
        net_charge=int(qm["net_charge"]),
        multiplicity=int(qm["multiplicity"]),
        optimization_step_block=_render_optimization_step_block(qm),
        resp_reference_block=_render_resp_reference_block(qm, convergence_profile=convergence_profile),
    )


def load_group_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_resp_job_manifest(job_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(job_dir) / "manifests" / "resp_apply_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_resp_job_candidate(job_dir: str | Path | None) -> RespJobCandidate | None:
    if job_dir is None:
        return None
    resolved = Path(job_dir).expanduser().resolve()
    manifest_path = resolved / "manifests" / "resp_apply_manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RespJobCandidate(job_dir=resolved, manifest_path=manifest_path, payload=payload)


def _group_payload_from_session_state(session_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(session_state, dict):
        return None
    payload = session_state.get("group_constraints")
    return payload if isinstance(payload, dict) else None


def _resolve_group_payload(
    *,
    molecule: MoleculeData,
    group_file: str | Path | None,
    session_state: dict[str, Any] | None,
) -> dict[str, Any]:
    if group_file:
        candidate = Path(group_file).expanduser()
        if candidate.exists():
            return load_group_payload(candidate)
    session_payload = _group_payload_from_session_state(session_state)
    if session_payload is not None:
        return session_payload
    return suggest_group_constraints(molecule)


def _molecule_from_session_state(session_state: dict[str, Any] | None) -> MoleculeData | None:
    if not isinstance(session_state, dict):
        return None
    payload = session_state.get("molecule")
    if not isinstance(payload, dict):
        return None
    atoms_payload = payload.get("atoms") or []
    bonds_payload = payload.get("bonds") or []
    atoms = [
        MoleculeAtom(
            index=int(item["index"]),
            name=str(item["name"]),
            element=str(item["element"]),
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
        )
        for item in atoms_payload
    ]
    bonds = [
        MoleculeBond(
            first=int(item["first"]),
            second=int(item["second"]),
            order=int(item.get("order") or 1),
        )
        for item in bonds_payload
    ]
    if not atoms:
        return None
    return MoleculeData(
        source_file=str(payload.get("source_file") or ""),
        source_format=str(payload.get("source_format") or "unknown"),
        atoms=atoms,
        bonds=bonds,
    )


def _read_xyz_atom_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_lines = lines[2:]
    names: list[str] = []
    for index, line in enumerate(atom_lines, start=1):
        tokens = line.split()
        label = tokens[0] if tokens else f"A{index}"
        names.append(str(label))
    return names


def _load_xyz_coordinates(path: Path, natoms: int) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_lines = lines[2 : 2 + natoms]
    names: list[str] = []
    coordinates: list[list[float]] = []
    for index, line in enumerate(atom_lines, start=1):
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"RESP XYZ line {index} in {path} is incomplete.")
        names.append(tokens[0])
        coordinates.append([float(tokens[1]) / 0.529177, float(tokens[2]) / 0.529177, float(tokens[3]) / 0.529177])
    if len(coordinates) != natoms:
        raise ValueError(f"RESP XYZ file {path} did not contain {natoms} atoms.")
    return names, coordinates


def _load_grid_rows(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"RESP grid file is empty: {path}")
    point_count = int(lines[0].split()[0])
    rows: list[list[float]] = []
    for line in lines[1 : 1 + point_count]:
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"RESP grid line in {path} is incomplete.")
        rows.append([float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])])
    if len(rows) != point_count:
        raise ValueError(f"RESP grid file {path} did not contain {point_count} points.")
    return rows


def _resolve_resp_grid_path(output_dir: Path) -> Path | None:
    preferred = output_dir / "resp_job.grid"
    if preferred.exists():
        return preferred
    matches = sorted(output_dir.glob("*.grid"))
    return matches[0] if matches else None


def _resolve_resp_xyz_path(*, output_dir: Path, grid_path: Path | None, job_dir: Path) -> Path | None:
    if grid_path is not None:
        stem_match = output_dir / f"{grid_path.stem}.xyz"
        if stem_match.exists():
            return stem_match
    preferred = output_dir / "resp_job.xyz"
    if preferred.exists():
        return preferred
    xyz_matches = sorted(output_dir.glob("*.xyz"))
    if xyz_matches:
        return xyz_matches[0]
    input_xyz = job_dir / "inputs" / "resp_job.xyz"
    if input_xyz.exists():
        return input_xyz
    return None


def _fit_resp_charge_payload(
    *,
    atom_names: list[str],
    total_charge: int,
    equality_pairs: list[tuple[int, int]],
    xyz_path: Path,
    grid_path: Path,
) -> dict[str, Any]:
    import numpy as np

    natoms = len(atom_names)
    _, coords_list = _load_xyz_coordinates(xyz_path, natoms)
    grid_list = _load_grid_rows(grid_path)
    coords = np.array(coords_list, dtype=float)
    grid = np.array(grid_list, dtype=float)
    groups = equality_groups_from_pairs(natoms, equality_pairs)

    def norm2(vector) -> float:
        return float(math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2))

    group_count = len(groups)
    matrix_size = group_count + 1
    A = np.zeros((matrix_size, matrix_size))
    B = np.zeros(matrix_size)
    group_distances = np.zeros((group_count, len(grid)))
    group_sizes = np.zeros(group_count)
    restrained_counts = np.zeros(group_count)
    for group_index, group in enumerate(groups):
        group_sizes[group_index] = float(len(group))
        restrained_counts[group_index] = float(
            sum(0 if atom_names[atom_index].upper().startswith("H") else 1 for atom_index in group)
        )
        for atom_index in group:
            for point_index in range(len(grid)):
                group_distances[group_index, point_index] += 1.0 / norm2(coords[atom_index] - grid[point_index, :3])
        B[group_index] = np.dot(grid[:, 3], group_distances[group_index])
    for group_index in range(group_count):
        for other_index in range(group_index, group_count):
            A[group_index, other_index] = np.dot(group_distances[group_index], group_distances[other_index])
            A[other_index, group_index] = A[group_index, other_index]
    A[:group_count, group_count] = group_sizes
    A[group_count, :group_count] = group_sizes
    B[group_count] = total_charge
    qold, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    for _ in range(50):
        current = A.copy()
        for group_index in range(group_count):
            if restrained_counts[group_index] <= 0.0:
                continue
            current[group_index, group_index] += (
                restrained_counts[group_index] * 0.005 / math.sqrt(qold[group_index] ** 2 + 0.01)
            )
        q, _, _, _ = np.linalg.lstsq(current, B, rcond=None)
        delta = float(np.amax(np.abs(q - qold)))
        qold = q.copy()
        if delta < 0.000001:
            break
    charges = [0.0] * natoms
    for group_index, group in enumerate(groups):
        for atom_index in group:
            charges[atom_index] = float(qold[group_index])
    return {
        "charges": [
            {"index": index + 1, "name": atom_names[index], "charge": charge}
            for index, charge in enumerate(charges)
        ],
        "total_charge": float(sum(charges)),
        "constraint_count": len(equality_pairs),
    }


def _write_resp_charge_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    json_path = output_dir / "resp_charges.json"
    txt_path = output_dir / "resp_charges.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(
        "\n".join(f"{float(item['charge']):.10f}" for item in payload.get("charges") or []) + "\n",
        encoding="utf-8",
    )


def _resp_charge_values_look_physical(charges: list[float], *, max_abs_charge: float = 5.0) -> bool:
    return bool(charges) and all(math.isfinite(charge) and abs(charge) <= max_abs_charge for charge in charges)


def _materialize_resp_charge_result(job_dir: str | Path, *, force_refit: bool = False) -> Path | None:
    candidate = load_resp_job_candidate(job_dir)
    if candidate is None:
        return None
    if candidate.charge_result_path is not None and not force_refit:
        return candidate.charge_result_path

    output_dir = candidate.output_dir
    job_path = Path(job_dir)
    grid_path = _resolve_resp_grid_path(output_dir)
    xyz_path = _resolve_resp_xyz_path(output_dir=output_dir, grid_path=grid_path, job_dir=job_path)
    if grid_path is None or xyz_path is None:
        return None

    session_state_path = Path(job_dir) / "manifests" / "popup_state.json"
    session_state = load_session_state(session_state_path) if session_state_path.exists() else None
    molecule = _molecule_from_session_state(session_state)
    if molecule is None:
        source_file = Path(str(candidate.payload.get("source_file") or "")).expanduser()
        if source_file.exists():
            molecule = load_molecule(source_file)
    if molecule is None:
        canonical_source = Path(str(candidate.payload.get("canonical_source_file") or "")).expanduser()
        if canonical_source.exists():
            molecule = load_molecule(canonical_source)
    if molecule is None:
        return None
    group_payload = _resolve_group_payload(
        molecule=molecule,
        group_file=Path(job_dir) / "group_constraints.json",
        session_state=session_state,
    )
    atom_names = _read_xyz_atom_names(xyz_path)
    payload = _fit_resp_charge_payload(
        atom_names=atom_names,
        total_charge=int(candidate.payload.get("net_charge") or 0),
        equality_pairs=equality_pairs_from_group_payload(group_payload),
        xyz_path=xyz_path,
        grid_path=grid_path,
    )
    _write_resp_charge_payload(output_dir, payload)
    return output_dir / "resp_charges.json"


def load_resp_charges(job_dir: str | Path) -> list[float]:
    output_dir = Path(job_dir) / "output"
    json_path = output_dir / "resp_charges.json"
    if json_path.exists():
        charges = load_resp_charge_result(json_path)
        if _resp_charge_values_look_physical(charges):
            return charges
        generated = _materialize_resp_charge_result(job_dir, force_refit=True)
        if generated is not None and generated.exists():
            refreshed = load_resp_charge_result(generated)
            if _resp_charge_values_look_physical(refreshed):
                return refreshed
        raise ValueError(
            "RESP charge data was found, but the stored charge magnitudes look non-physical. "
            f"Existing file: {json_path}"
        )

    text_path = output_dir / "resp_charges.txt"
    if text_path.exists():
        charges: list[float] = []
        for line in text_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            charges.append(float(stripped))
        if _resp_charge_values_look_physical(charges):
            return charges
        generated = _materialize_resp_charge_result(job_dir, force_refit=True)
        if generated is not None and generated.exists():
            refreshed = load_resp_charge_result(generated)
            if _resp_charge_values_look_physical(refreshed):
                return refreshed
        raise ValueError(
            "RESP charge data was found, but the stored charge magnitudes look non-physical. "
            f"Existing file: {text_path}"
        )

    generated = _materialize_resp_charge_result(job_dir)
    if generated is not None and generated.exists():
        return load_resp_charge_result(generated)

    raise FileNotFoundError(
        "RESP charges were not found in the selected job directory. "
        f"Expected {json_path} or {text_path}. "
        "A local RESP refit also could not be generated from the existing NWChem output."
    )


def resp_job_completed(job_dir: str | Path) -> bool:
    output_dir = Path(job_dir) / "output"
    return (output_dir / "resp_charges.json").exists() or (output_dir / "resp_charges.txt").exists()


def _next_resp_job_dir(base_dir: Path) -> Path:
    max_index = 0
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip().upper()
        if not name.startswith(_RESP_JOB_DIR_PREFIX):
            continue
        suffix = child.name[len(_RESP_JOB_DIR_PREFIX):]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return base_dir / f"{_RESP_JOB_DIR_PREFIX}{max_index + 1:03d}"


def write_resp_job_assets(
    *,
    source_file: str | Path,
    coordinate_source_file: str | Path | None = None,
    resume_source_file: str | Path | None = None,
    metal_pdb_file: str | Path | None = None,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
    job_dir: str | Path,
    slurm_config: SlurmConfig,
    session_state: dict[str, Any] | None = None,
    group_file: str | Path | None = None,
) -> dict[str, Any]:
    target_dir = Path(job_dir)
    inputs_dir = target_dir / "inputs"
    slurm_dir = target_dir / "slurm"
    output_dir = target_dir / "output"
    manifest_dir = target_dir / "manifests"
    for path in (target_dir, inputs_dir, slurm_dir, output_dir, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_file).expanduser().resolve()
    coordinate_source_path = Path(coordinate_source_file).expanduser().resolve() if coordinate_source_file else source_path
    canonical_source_path = (inputs_dir / "resp_source.mol2").resolve()
    resume_source_path = Path(resume_source_file).expanduser().resolve() if resume_source_file else source_path
    resume_copy_path = (inputs_dir / f"resp_resume_input{resume_source_path.suffix.lower()}").resolve()
    if resume_source_path != resume_copy_path:
        shutil.copy2(resume_source_path, resume_copy_path)
    if coordinate_source_path.suffix.lower() == ".mol2":
        if coordinate_source_path != canonical_source_path:
            shutil.copy2(coordinate_source_path, canonical_source_path)
    else:
        preview_molecule = load_molecule(coordinate_source_path)
        preview_text = render_preview_mol2(preview_molecule, residue_name=residue_name)
        canonical_source_path.write_text(
            preview_text + ("\n" if not preview_text.endswith("\n") else ""),
            encoding="utf-8",
        )

    metal_copy_path: Path | None = None
    if metal_pdb_file is not None:
        metal_source = Path(metal_pdb_file).expanduser().resolve()
        if metal_source.exists():
            metal_copy_path = (inputs_dir / "resp_separated_metals.pdb").resolve()
            if metal_source != metal_copy_path:
                shutil.copy2(metal_source, metal_copy_path)

    molecule = load_molecule(canonical_source_path)
    fingerprint = molecule_fingerprint(
        source_path,
        residue_name=residue_name,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    group_payload = _resolve_group_payload(
        molecule=molecule,
        group_file=group_file,
        session_state=session_state,
    )
    state = session_state or build_default_session_state(
        molecule,
        residue_name=residue_name,
        fingerprint=fingerprint,
        net_charge=net_charge,
        multiplicity=multiplicity,
        group_payload=group_payload,
    )
    state["canonical_source_file"] = str(canonical_source_path)
    state["source_file"] = str(canonical_source_path)
    state["source_format"] = "mol2"
    state["molecule"] = molecule.to_dict()
    state["group_constraints"] = group_payload
    state["fingerprint"] = fingerprint
    state["residue_name"] = residue_name
    state["qm_settings"] = normalize_qm_settings(
        state.get("qm_settings") if isinstance(state, dict) else None,
        net_charge=int(net_charge),
        multiplicity=int(multiplicity),
    )
    state["mol2_preview"] = canonical_source_path.read_text(encoding="utf-8")

    equality_pairs = equality_pairs_from_group_payload(group_payload)
    (inputs_dir / "resp_job.xyz").write_text(_render_xyz(molecule), encoding="utf-8")
    (inputs_dir / "resp_job.nw").write_text(
        render_nwchem_input(molecule, session_state=state),
        encoding="utf-8",
    )
    (inputs_dir / "resp_fit.py").write_text(
        render_runtime_resp_fit_script(
            atom_names=[atom.name for atom in molecule.atoms],
            total_charge=int(net_charge),
            equality_pairs=equality_pairs,
        ),
        encoding="utf-8",
    )
    write_json(target_dir / "group_constraints.json", group_payload)
    write_json(manifest_dir / "popup_state.json", state)

    generic_slurm = slurm_dir / "run_resp.sbatch"
    generic_slurm.write_text(
        render_resp_slurm_script(job_root=target_dir, slurm_config=slurm_config, job_name=f"{residue_name}_resp"),
        encoding="utf-8",
    )
    tahoma_slurm = slurm_dir / "tahoma_resp.sbatch"
    tahoma_slurm.write_text(
        render_tahoma_resp_script(job_root=target_dir, job_name=f"{residue_name}_resp"),
        encoding="utf-8",
    )
    submit_script = slurm_dir / "submit_tahoma.sh"
    submit_script.write_text(
        "#!/bin/bash\nset -euo pipefail\ncd -- \"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"\nsbatch tahoma_resp.sbatch\n",
        encoding="utf-8",
    )

    manifest = {
        "fingerprint": fingerprint,
        "status": "setup_pending",
        "created_at": datetime.now(UTC).isoformat(),
        "source_file": str(source_path),
        "resume_source_file": str(resume_copy_path),
        "canonical_source_file": str(canonical_source_path),
        "metal_pdb_file": str(metal_copy_path) if metal_copy_path is not None else None,
        "charge_method": state.get("charge_method"),
        "residue_name": residue_name,
        "net_charge": int(net_charge),
        "multiplicity": int(multiplicity),
        "job_dir": str(target_dir.resolve()),
        "files": {
            "resume_source_file": str(resume_copy_path),
            "canonical_source_file": str(canonical_source_path),
            **({"metal_pdb": str(metal_copy_path)} if metal_copy_path is not None else {}),
            "group_constraints": str((target_dir / "group_constraints.json").resolve()),
            "popup_state": str((manifest_dir / "popup_state.json").resolve()),
            "nwchem_input": str((inputs_dir / "resp_job.nw").resolve()),
            "xyz_input": str((inputs_dir / "resp_job.xyz").resolve()),
            "resp_fit": str((inputs_dir / "resp_fit.py").resolve()),
            "slurm": str(generic_slurm.resolve()),
            "tahoma": str(tahoma_slurm.resolve()),
            "submit_tahoma": str(submit_script.resolve()),
            "expected_charge_result": str((output_dir / "resp_charges.json").resolve()),
        },
    }
    manifest_path = manifest_dir / "resp_apply_manifest.json"
    write_json(manifest_path, manifest)
    result = {
        "job_dir": str(target_dir.resolve()),
        "fingerprint": fingerprint,
        "manifest_path": str(manifest_path.resolve()),
        "resume_source_file": str(resume_copy_path),
        "canonical_source_file": str(canonical_source_path),
        "group_constraints": str((target_dir / "group_constraints.json").resolve()),
        "popup_state": str((manifest_dir / "popup_state.json").resolve()),
        "nwchem_input": str((inputs_dir / "resp_job.nw").resolve()),
        "slurm": str(generic_slurm.resolve()),
        "tahoma": str(tahoma_slurm.resolve()),
        "expected_charge_result": str((output_dir / "resp_charges.json").resolve()),
    }
    if metal_copy_path is not None:
        result["metal_pdb"] = str(metal_copy_path)
    return result


def load_session_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_resp_job_candidates(
    *,
    search_root: str | Path,
    fingerprint: str,
    explicit_job_dir: str | Path | None = None,
) -> list[RespJobCandidate]:
    candidates: list[RespJobCandidate] = []
    checked: set[Path] = set()

    def _maybe_add(job_dir: Path) -> None:
        resolved = job_dir.resolve()
        manifest_path = resolved / "manifests" / "resp_apply_manifest.json"
        if resolved in checked or not manifest_path.exists():
            return
        checked.add(resolved)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("fingerprint")) != fingerprint:
            return
        candidates.append(RespJobCandidate(job_dir=resolved, manifest_path=manifest_path, payload=payload))

    if explicit_job_dir:
        _maybe_add(Path(explicit_job_dir).expanduser())

    root = Path(search_root).expanduser().resolve()
    if root.exists():
        for manifest_path in root.rglob("resp_apply_manifest.json"):
            _maybe_add(manifest_path.parent.parent)

    candidates.sort(
        key=lambda item: (
            1 if item.completed else 0,
            str(item.payload.get("created_at") or ""),
        ),
        reverse=True,
    )
    return candidates


def find_resp_source_candidates(
    *,
    search_root: str | Path,
    source_file: str | Path,
    explicit_job_dir: str | Path | None = None,
    completed_only: bool = False,
) -> list[RespJobCandidate]:
    candidates: list[RespJobCandidate] = []
    checked: set[Path] = set()
    target_path = Path(source_file).expanduser().resolve()
    target_source = str(target_path)
    target_name = target_path.name.casefold()

    def _maybe_add(job_dir: Path) -> None:
        resolved = job_dir.resolve()
        manifest_path = resolved / "manifests" / "resp_apply_manifest.json"
        if resolved in checked or not manifest_path.exists():
            return
        checked.add(resolved)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_source = str(payload.get("source_file") or "")
        payload_name = Path(payload_source).name.casefold() if payload_source else ""
        if payload_source != target_source and payload_name != target_name:
            return
        candidate = RespJobCandidate(job_dir=resolved, manifest_path=manifest_path, payload=payload)
        if completed_only and not candidate.completed:
            return
        candidates.append(candidate)

    if explicit_job_dir:
        _maybe_add(Path(explicit_job_dir).expanduser())

    root = Path(search_root).expanduser().resolve()
    if root.exists():
        for manifest_path in root.rglob("resp_apply_manifest.json"):
            _maybe_add(manifest_path.parent.parent)

    candidates.sort(
        key=lambda item: (
            1 if item.ready_to_continue else 0,
            1 if item.completed else 0,
            str(item.payload.get("created_at") or ""),
        ),
        reverse=True,
    )
    return candidates


def select_job_dir(
    *,
    base_dir: str | Path,
    fingerprint: str,
    apply_mode: RespApplyMode,
    existing_job_dir: str | Path | None = None,
) -> Path:
    base_path = Path(base_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    if apply_mode == RespApplyMode.NEW_DIRECTORY:
        return _next_resp_job_dir(base_path)
    if apply_mode == RespApplyMode.REBUILD and existing_job_dir is not None:
        candidate = Path(existing_job_dir).expanduser().resolve()
        if candidate.exists():
            shutil.rmtree(candidate)
        return candidate
    if existing_job_dir:
        return Path(existing_job_dir).expanduser().resolve()
    return _next_resp_job_dir(base_path)
