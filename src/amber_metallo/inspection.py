from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import gemmi
import requests

from amber_metallo.missing_loops import MissingLoopSummary, analyze_missing_loops


PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "LYN",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}
NUCLEIC_RESIDUES = {
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
WATER_NAMES = {"HOH", "WAT", "H2O", "SOL", "SPC", "SPCE", "TIP3", "TIP3P"}
RARE_EARTH_METALS = {
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
SUPPORTED_METALS = {"Co", "Cu", "Ni", "Mn", "Fe", *RARE_EARTH_METALS}
PDB_BASE_URL = "https://files.rcsb.org/download"


@dataclass(slots=True)
class ResidueRecord:
    key: str
    chain: str
    seqid: str
    residue_name: str
    atom_count: int
    classification: str


@dataclass(slots=True)
class MetalSite:
    site: int
    key: str
    chain: str
    seqid: str
    residue_name: str
    atom_name: str
    atom_serial: int
    element: str


@dataclass(slots=True)
class StructureSummary:
    source: str
    source_path: str
    residue_counts: dict[str, int]
    metals: list[MetalSite] = field(default_factory=list)
    hetero_residues: list[ResidueRecord] = field(default_factory=list)
    ligand_candidates: list[ResidueRecord] = field(default_factory=list)
    missing_loops: MissingLoopSummary | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_path": self.source_path,
            "residue_counts": self.residue_counts,
            "metals": [asdict(site) for site in self.metals],
            "hetero_residues": [asdict(item) for item in self.hetero_residues],
            "ligand_candidates": [asdict(item) for item in self.ligand_candidates],
            "missing_loops": None if self.missing_loops is None else self.missing_loops.to_dict(),
        }


def looks_like_pdb_id(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) == 4 and stripped.isalnum()


def fetch_pdb_structure(pdb_id: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = pdb_id.lower()
    pdb_path = dest_dir / f"{token}.pdb"
    cif_path = dest_dir / f"{token}.cif"

    pdb_url = f"{PDB_BASE_URL}/{pdb_id.upper()}.pdb"
    response = requests.get(pdb_url, timeout=30)
    if response.ok and "HEADER" in response.text:
        pdb_path.write_text(response.text, encoding="utf-8")
        return pdb_path

    cif_url = f"{PDB_BASE_URL}/{pdb_id.upper()}.cif"
    response = requests.get(cif_url, timeout=30)
    response.raise_for_status()
    cif_path.write_text(response.text, encoding="utf-8")

    structure = gemmi.read_structure(str(cif_path))
    structure.remove_alternative_conformations()
    structure.write_minimal_pdb(str(pdb_path))
    return pdb_path


def load_structure(source: str | Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    return structure


def residue_key(chain_name: str, residue: gemmi.Residue) -> str:
    return f"{chain_name}:{residue.name.strip()}:{residue.seqid}"


def supported_metal_element(residue: gemmi.Residue) -> str | None:
    """Return the supported metal element represented by a monatomic residue.

    Some Amber ``ambpdb`` outputs label a neodymium atom/residue as ``Nd`` but
    write ``N`` in the PDB element column.  Gemmi correctly reports that
    column as nitrogen, so fall back to the monatomic residue name when the
    explicit element is not one of the metals supported by SIMPLE.
    """
    if len(residue) != 1:
        return None
    atom_element = residue[0].element.name.title()
    if atom_element in SUPPORTED_METALS:
        return atom_element
    residue_element = residue.name.strip().title()
    if residue_element in SUPPORTED_METALS:
        return residue_element
    return None


def classify_residue(residue: gemmi.Residue) -> str:
    name = residue.name.strip().upper()
    if residue.is_water() or name in WATER_NAMES:
        return "water"
    if name in PROTEIN_RESIDUES or name in NUCLEIC_RESIDUES:
        return "standard"
    if supported_metal_element(residue) is not None:
        return "metal"
    return "hetero"


def _iter_residues(structure: gemmi.Structure) -> Iterable[tuple[str, gemmi.Residue]]:
    model = structure[0]
    for chain in model:
        for residue in chain:
            yield chain.name, residue


def inspect_structure(
    source_path: str | Path,
    source_label: str = "pdb_file",
    *,
    detect_missing_loops: bool = True,
) -> StructureSummary:
    structure = load_structure(source_path)
    counts = {"standard": 0, "water": 0, "metal": 0, "hetero": 0}
    metals: list[MetalSite] = []
    hetero_residues: list[ResidueRecord] = []
    ligand_candidates: list[ResidueRecord] = []

    site_index = 1
    for chain_name, residue in _iter_residues(structure):
        classification = classify_residue(residue)
        counts[classification] += 1
        record = ResidueRecord(
            key=residue_key(chain_name, residue),
            chain=chain_name,
            seqid=str(residue.seqid),
            residue_name=residue.name.strip(),
            atom_count=len(residue),
            classification=classification,
        )
        if classification == "metal":
            atom = residue[0]
            element = supported_metal_element(residue) or atom.element.name.title()
            metals.append(
                MetalSite(
                    site=site_index,
                    key=record.key,
                    chain=chain_name,
                    seqid=record.seqid,
                    residue_name=record.residue_name,
                    atom_name=atom.name.strip(),
                    atom_serial=atom.serial,
                    element=element,
                )
            )
            site_index += 1
        elif classification == "hetero":
            hetero_residues.append(record)
            ligand_candidates.append(record)

    return StructureSummary(
        source=source_label,
        source_path=str(source_path),
        residue_counts=counts,
        metals=metals,
        hetero_residues=hetero_residues,
        ligand_candidates=ligand_candidates,
        missing_loops=analyze_missing_loops(source_path) if detect_missing_loops else None,
    )
