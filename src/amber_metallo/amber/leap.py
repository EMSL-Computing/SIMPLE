from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from amber_metallo.c4_assets import (
    opc_duvail_c4_file,
    opc_duvail_ion_frcmod,
    opc_duvail_polarizability_file,
)
from amber_metallo.config import (
    BoxShape,
    DESC4ParameterSet,
    MetalModel,
    NeutralizationIon,
    SaltConfig,
    SaltKind,
    SaltMode,
    SystemConfig,
)
from amber_metallo.environment import AmberEnvironment
from amber_metallo.execution import ensure_execution_host, run_command
from amber_metallo.inspection import inspect_structure
from amber_metallo.ligand_param import LigandArtifacts
from amber_metallo.qm.nwchem import load_molecule
from amber_metallo.reporting import write_json


PROTEIN_FORCEFIELD_MAP = {
    "ff19SB": "leaprc.protein.ff19SB",
    "ff14SB": "leaprc.protein.ff14SB",
    "ff99SB": "oldff/leaprc.ff99SB",
    "ff99SBildn": "oldff/leaprc.ff99SBildn",
}
WATER_FORCEFIELD_MAP = {
    "spce": ("leaprc.water.spce", "SPCBOX"),
    "spceb": ("leaprc.water.spceb", "SPCBOX"),
    "tip3p": ("leaprc.water.tip3p", "TIP3PBOX"),
    "opc": ("leaprc.water.opc", "OPCBOX"),
    "opc3": ("leaprc.water.opc3", "OPC3BOX"),
    "opc3pol": ("leaprc.water.opc3pol", "POL3BOX"),
    "tip4pew": ("leaprc.water.tip4pew", "TIP4PEWBOX"),
    "tip4pd": ("leaprc.water.tip4pd", "TIP4PBOX"),
    "tip5p": ("leaprc.water.tip5p", "TIP5PBOX"),
    "fb3": ("leaprc.water.fb3", "FB3BOX"),
    "fb4": ("leaprc.water.fb4", "FB4BOX"),
}
PARMED_WATER_MODEL_MAP = {
    "spce": "SPCE",
    "tip3p": "TIP3P",
    "opc": "OPC",
    "opc3": "OPC3",
    "opc3pol": "OPC3POL",
    "tip4pew": "TIP4PEW",
    "tip4pd": "TIP4PD",
    "tip5p": "TIP5P",
}
SUPPORTED_1264_METAL_CHARGES = {
    "Co": (2,),
    "Cu": (1, 2),
    "Ni": (2,),
    "Mn": (2,),
    "Fe": (2, 3),
    "Sc": (3,),
    "Y": (3,),
    "La": (3,),
    "Ce": (3,),
    "Pr": (3,),
    "Nd": (3,),
    "Pm": (3,),
    "Sm": (3,),
    "Eu": (3,),
    "Gd": (3,),
    "Tb": (3,),
    "Dy": (3,),
    "Ho": (3,),
    "Er": (3,),
    "Tm": (3,),
    "Yb": (3,),
    "Lu": (3,),
}
TLEAP_ION_LIBRARY_LABELS = {
    ("Co", 2): "CO",
    ("Cu", 1): "CU1",
    ("Cu", 2): "CU",
    ("Ni", 2): "NI",
    ("Mn", 2): "MN",
    ("Fe", 2): "FE2",
    ("Fe", 3): "FE",
    ("Sc", 3): "SC",
    ("Y", 3): "Y",
    ("La", 3): "LA",
    ("Ce", 3): "CE",
    ("Pr", 3): "PR",
    ("Nd", 3): "Nd",
    ("Pm", 3): "PM",
    ("Sm", 3): "SM",
    ("Eu", 3): "EU3",
    ("Gd", 3): "GD",
    ("Tb", 3): "TB",
    ("Dy", 3): "DY",
    ("Ho", 3): "HO",
    ("Er", 3): "ER",
    ("Tm", 3): "TM",
    ("Yb", 3): "YB",
    ("Lu", 3): "LU",
}
TLEAP_ADDITIVE_ION_UNIT_NAMES = {
    "Na+": "Na+",
    "K+": "K+",
    "Cl-": "Cl-",
    "Br-": "BR",
    "Ca2+": "CA",
}
DEFAULT_TLEAP_METAL_CHARGES = {
    "Co": 2,
    "Cu": 2,
    "Ni": 2,
    "Mn": 2,
    "Fe": 2,
    "Sc": 3,
    "Y": 3,
    "La": 3,
    "Ce": 3,
    "Pr": 3,
    "Nd": 3,
    "Pm": 3,
    "Sm": 3,
    "Eu": 3,
    "Gd": 3,
    "Tb": 3,
    "Dy": 3,
    "Ho": 3,
    "Er": 3,
    "Tm": 3,
    "Yb": 3,
    "Lu": 3,
}
NEUTRALIZATION_CHARGE_TOLERANCE = 0.05
_PRMTOP_FORMAT_PATTERN = re.compile(r"^%FORMAT\((?P<count>\d+)(?P<kind>[A-Za-z])(?P<width>\d+)(?:\.(?P<precision>\d+))?\)$")
_COMMON_ELEMENT_MASSES_DA = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.060,
    "Cl": 35.450,
    "K": 39.098,
    "Ca": 40.078,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.380,
    "Br": 79.904,
    "Y": 88.906,
    "I": 126.904,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.242,
    "Sm": 150.360,
    "Eu": 151.964,
    "Gd": 157.250,
    "Tb": 158.925,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.934,
    "Yb": 173.045,
    "Lu": 174.967,
}


@dataclass(slots=True)
class LeapBuildResult:
    script_path: str
    output_files: dict[str, str]
    warnings: list[str]
    commands: list[list[str]]
    water_count: int | None
    extra_ions: dict[str, int]
    neutralizing_ions: dict[str, int] = field(default_factory=dict)
    added_ions: dict[str, int] = field(default_factory=dict)
    volume_angstrom3: float | None = None
    charge_before_ions: float | None = None
    final_charge: float | None = None
    salt_formula_units: int | None = None
    actual_salt_concentration_m: float | None = None
    c4_script_path: str | None = None
    c4_mask: str | None = None
    c4_applied: bool = False
    box_lengths_angstrom: tuple[float, float, float] | None = None
    system_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _protein_forcefield_source(forcefield: str) -> str:
    return PROTEIN_FORCEFIELD_MAP.get(forcefield, f"leaprc.protein.{forcefield}")


def _salt_ion_names(kind: SaltKind) -> tuple[str | None, str | None]:
    if kind == SaltKind.NACL:
        return "Na+", "Cl-"
    if kind == SaltKind.KCL:
        return "K+", "Cl-"
    if kind == SaltKind.CACL2:
        return "Ca2+", "Cl-"
    return None, None


def _ion_formal_charges(kind: SaltKind) -> tuple[int | None, int | None]:
    if kind == SaltKind.NACL:
        return 1, -1
    if kind == SaltKind.KCL:
        return 1, -1
    if kind == SaltKind.CACL2:
        return 2, -1
    return None, None


def _normalized_total_charge(net_charge: float | None) -> float | None:
    if net_charge is None:
        return None
    rounded = round(float(net_charge))
    if abs(float(net_charge) - rounded) <= NEUTRALIZATION_CHARGE_TOLERANCE:
        return float(rounded)
    return float(net_charge)


def _predict_neutralizing_ions(
    net_charge: float | None,
    salt_kind: SaltKind,
    neutralization_ion: NeutralizationIon | str | None = None,
) -> tuple[dict[str, int], float | None]:
    cation_name, anion_name = _salt_ion_names(salt_kind)
    cation_charge, anion_charge = _ion_formal_charges(salt_kind)
    if net_charge is None or cation_name is None or anion_name is None or cation_charge is None or anion_charge is None:
        return {}, net_charge

    current_charge = _normalized_total_charge(net_charge)
    if current_charge is None:
        return {}, net_charge
    requested = (
        neutralization_ion.value
        if isinstance(neutralization_ion, NeutralizationIon)
        else str(neutralization_ion or NeutralizationIon.SALT_DEFAULT.value)
    )
    if requested not in {NeutralizationIon.AUTO.value, NeutralizationIon.SALT_DEFAULT.value}:
        ion_charges = {"Na+": 1, "K+": 1, "Cl-": -1, "Br-": -1}
        try:
            ion_charge = ion_charges[requested]
        except KeyError as exc:
            raise ValueError(f"Unsupported neutralization ion: {requested}") from exc
        if current_charge == 0:
            return {}, 0.0
        if current_charge * ion_charge >= 0:
            raise ValueError(
                f"Neutralization ion {requested} has the wrong charge sign for a system charge of "
                f"{current_charge:+.3f}."
            )
        needed = int(math.ceil(abs(current_charge) / abs(ion_charge)))
        return {requested: needed}, current_charge + needed * ion_charge
    neutralizing = {cation_name: 0, anion_name: 0}
    if current_charge < 0:
        needed_cations = int(math.ceil(abs(current_charge) / cation_charge))
        neutralizing[cation_name] = needed_cations
        current_charge += needed_cations * cation_charge
    if current_charge > 0:
        needed_anions = int(math.ceil(current_charge / abs(anion_charge)))
        neutralizing[anion_name] += needed_anions
        current_charge -= needed_anions * abs(anion_charge)

    neutralizing = {name: count for name, count in neutralizing.items() if count > 0}
    return neutralizing, current_charge


def predict_neutralizing_ions(
    net_charge: float | None,
    salt_kind: SaltKind,
    neutralization_ion: NeutralizationIon | str | None = None,
) -> tuple[dict[str, int], float | None]:
    """Return explicit counter-ion counts for a nominally integral system charge."""
    return _predict_neutralizing_ions(net_charge, salt_kind, neutralization_ion)


def _combine_ion_counts(*ion_maps: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for ion_map in ion_maps:
        for ion_name, count in ion_map.items():
            if count > 0:
                combined[ion_name] = combined.get(ion_name, 0) + count
    return combined


def combine_ion_counts(*ion_maps: dict[str, int]) -> dict[str, int]:
    return _combine_ion_counts(*ion_maps)


def _salt_formula_units(extra_ions: dict[str, int], salt_kind: SaltKind) -> int:
    if not extra_ions:
        return 0
    if salt_kind in {SaltKind.NACL, SaltKind.KCL}:
        cation, anion = _salt_ion_names(salt_kind)
        return min(extra_ions.get(cation or "", 0), extra_ions.get(anion or "", 0))
    if salt_kind == SaltKind.CACL2:
        return min(extra_ions.get("Ca2+", 0), extra_ions.get("Cl-", 0) // 2)
    return 0


def salt_formula_units(extra_ions: dict[str, int], salt_kind: SaltKind) -> int:
    return _salt_formula_units(extra_ions, salt_kind)


def _salt_concentration_m(formula_units: int, volume_angstrom3: float | None) -> float | None:
    if volume_angstrom3 is None or formula_units <= 0:
        return 0.0 if formula_units == 0 else None
    avogadro = 6.02214076e23
    volume_liters = volume_angstrom3 * 1e-27
    if volume_liters <= 0:
        return None
    return (formula_units / avogadro) / volume_liters


def salt_concentration_m(formula_units: int, volume_angstrom3: float | None) -> float | None:
    return _salt_concentration_m(formula_units, volume_angstrom3)


def _salt_formula_units_from_volume(volume_angstrom3: float | None, concentration_m: float) -> int:
    if volume_angstrom3 is None or concentration_m <= 0:
        return 0
    avogadro = 6.02214076e23
    volume_liters = volume_angstrom3 * 1e-27
    if volume_liters <= 0:
        return 0
    return max(0, round(float(concentration_m) * volume_liters * avogadro))


def calculate_salt_ions(water_count: int, salt_config: SaltConfig) -> dict[str, int]:
    if salt_config.kind == SaltKind.NONE or salt_config.mode in {SaltMode.NONE, SaltMode.NEUTRALIZE}:
        return {}
    if salt_config.mode == SaltMode.COUNT:
        units = int(salt_config.value)
    else:
        units = round(water_count * float(salt_config.value) / 55.5)
    if salt_config.kind == SaltKind.NACL:
        return {"Na+": units, "Cl-": units}
    if salt_config.kind == SaltKind.KCL:
        return {"K+": units, "Cl-": units}
    return {"Ca2+": units, "Cl-": units * 2}


def calculate_salt_ions_from_volume(volume_angstrom3: float | None, salt_config: SaltConfig) -> dict[str, int]:
    if salt_config.kind == SaltKind.NONE or salt_config.mode in {SaltMode.NONE, SaltMode.NEUTRALIZE}:
        return {}
    if salt_config.mode == SaltMode.COUNT:
        return calculate_salt_ions(0, salt_config)
    units = _salt_formula_units_from_volume(volume_angstrom3, float(salt_config.value))
    if salt_config.kind == SaltKind.NACL:
        return {"Na+": units, "Cl-": units}
    if salt_config.kind == SaltKind.KCL:
        return {"K+": units, "Cl-": units}
    return {"Ca2+": units, "Cl-": units * 2}


def count_waters_in_pdb(path: Path) -> int:
    oxygen_count = 0
    residues: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        residue_name = line[17:20].strip().upper()
        if residue_name not in {"WAT", "HOH", "SPC", "SPCE", "TIP3P", "OPC", "OP3", "FB3", "FB4", "T4E"}:
            continue
        atom_name = line[12:16].strip().upper()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if element == "O" or atom_name in {"O", "OW", "OH2"}:
            oxygen_count += 1
        chain = line[21].strip()
        resid = line[22:26].strip()
        residues.add((chain, resid, residue_name))
    if oxygen_count > 0:
        return oxygen_count
    return len(residues)


def extract_total_charges(output_text: str) -> list[float]:
    pattern = re.compile(r"Total unperturbed charge:\s*([-+]?\d+(?:\.\d+)?)")
    return [float(match.group(1)) for match in pattern.finditer(output_text)]


def read_pdb_cell_volume(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("CRYST1"):
            continue
        try:
            a = float(line[6:15].strip())
            b = float(line[15:24].strip())
            c = float(line[24:33].strip())
            alpha = math.radians(float(line[33:40].strip()))
            beta = math.radians(float(line[40:47].strip()))
            gamma = math.radians(float(line[47:54].strip()))
        except ValueError:
            return None
        volume = a * b * c * math.sqrt(
            max(
                0.0,
                1
                - math.cos(alpha) ** 2
                - math.cos(beta) ** 2
                - math.cos(gamma) ** 2
                + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma),
            )
        )
        return volume
    return None


def _ligand_template_lines(
    artifacts: list[LigandArtifacts],
    *,
    skip_mol2_files: set[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    skipped = {Path(path).as_posix() for path in (skip_mol2_files or set())}
    for index, artifact in enumerate(artifacts, start=1):
        if "mol2" in artifact.files:
            mol2_path = Path(artifact.files["mol2"]).as_posix()
            if mol2_path not in skipped:
                lines.append(f"ligand_{index} = loadMol2 {mol2_path}")
        if "prepi" in artifact.files:
            lines.append(f"loadAmberPrep {Path(artifact.files['prepi']).as_posix()}")
        if "off" in artifact.files:
            lines.append(f"loadOff {Path(artifact.files['off']).as_posix()}")
        if "lib" in artifact.files:
            lines.append(f"loadOff {Path(artifact.files['lib']).as_posix()}")
        if "frcmod" in artifact.files:
            lines.append(f"loadAmberParams {Path(artifact.files['frcmod']).as_posix()}")
    return lines


def _tleap_ion_library_label(element: str, charge: int | None) -> str | None:
    if charge is None:
        return None
    return TLEAP_ION_LIBRARY_LABELS.get((element.title(), int(charge)))


def _tleap_additive_ion_unit_name(ion_name: str) -> str:
    return TLEAP_ADDITIVE_ION_UNIT_NAMES.get(ion_name, ion_name)


def _metal_element_from_pdb_line(line: str) -> str | None:
    candidates = []
    if len(line) >= 78:
        candidates.append(line[76:78].strip())
    candidates.append(line[12:16].strip())
    candidates.append(line[17:20].strip())
    for candidate in candidates:
        cleaned = "".join(character for character in candidate if character.isalpha())
        if not cleaned:
            continue
        if len(cleaned) >= 2 and cleaned[:2].title() in SUPPORTED_1264_METAL_CHARGES:
            return cleaned[:2].title()
        if cleaned[:1].title() in SUPPORTED_1264_METAL_CHARGES:
            return cleaned[:1].title()
    return None


def _tleap_metal_label(element: str, charge: int | None) -> str | None:
    if charge is None:
        charge = DEFAULT_TLEAP_METAL_CHARGES.get(element.title())
    return _tleap_ion_library_label(element, charge)


def _tleap_metal_label_map(source_path: Path, system_config: SystemConfig) -> dict[str, str]:
    summary = inspect_structure(source_path, detect_missing_loops=False)
    explicit_charges = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    labels: dict[str, str] = {}
    for metal_site in summary.metals:
        element = metal_site.element.title()
        charge = explicit_charges.get(int(metal_site.site), DEFAULT_TLEAP_METAL_CHARGES.get(element))
        label = _tleap_metal_label(element, charge)
        if label is not None:
            labels[metal_site.key] = label
    return labels


def _write_tleap_pdb_without_conect(
    source_path: Path,
    output_dir: Path,
    label: str,
    *,
    system_config: SystemConfig | None = None,
) -> Path:
    target = output_dir / f"{label}_for_tleap.pdb"
    metal_label_map = {} if system_config is None else _tleap_metal_label_map(source_path, system_config)
    filtered_lines = [
        line
        for line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not line.startswith(("CONECT", "MASTER"))
    ]
    if metal_label_map:
        rewritten_lines: list[str] = []
        for line in filtered_lines:
            if not line.startswith(("ATOM", "HETATM")):
                rewritten_lines.append(line)
                continue
            residue_name = line[17:20].strip()
            chain = line[21].strip()
            seqid = line[22:27].strip()
            residue_key = f"{chain}:{residue_name}:{seqid}"
            mapped_label = metal_label_map.get(residue_key)
            if mapped_label is None:
                rewritten_lines.append(line)
                continue
            atom_field = f"{mapped_label:>4}"[:4]
            residue_field = f"{mapped_label:>3}"[:3]
            rewritten_lines.append(f"{line[:12]}{atom_field}{line[16:17]}{residue_field}{line[20:]}")
        filtered_lines = rewritten_lines
    target.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    return target


def _write_small_molecule_metal_pdb_for_tleap(
    source_path: Path,
    output_dir: Path,
    system_config: SystemConfig,
    label: str,
) -> Path:
    target = output_dir / f"{label}_for_tleap.pdb"
    explicit_charges = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    rewritten_lines: list[str] = []
    site_index = 0
    for line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("CONECT", "MASTER")):
            continue
        if not line.startswith(("ATOM", "HETATM")):
            rewritten_lines.append(line)
            continue
        site_index += 1
        element = _metal_element_from_pdb_line(line)
        mapped_label = None
        if element is not None:
            mapped_label = _tleap_metal_label(
                element,
                explicit_charges.get(site_index, DEFAULT_TLEAP_METAL_CHARGES.get(element)),
            )
        if mapped_label is None:
            rewritten_lines.append(line)
            continue
        atom_field = f"{mapped_label:>4}"[:4]
        residue_field = f"{mapped_label:>3}"[:3]
        rewritten_lines.append(f"{line[:12]}{atom_field}{line[16:17]}{residue_field}{line[20:]}")
    target.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
    return target


def _small_molecule_metal_pdb_files(ligand_artifacts: list[LigandArtifacts]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for artifact in ligand_artifacts:
        raw_path = artifact.files.get("metal_pdb")
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        files.append(candidate)
    return files


def _coordinate_unit_lines(
    system_config: SystemConfig,
    prepared_pdb: Path | None,
    ligand_artifacts: list[LigandArtifacts],
    source_files: list[Path],
    output_dir: Path,
) -> list[str]:
    if prepared_pdb is not None:
        tleap_pdb = _write_tleap_pdb_without_conect(
            prepared_pdb,
            output_dir,
            "prepared_input",
            system_config=system_config,
        )
        return [f"system = loadPDB {tleap_pdb.as_posix()}"]
    if not ligand_artifacts:
        raise ValueError("Small-molecule-only workflows require ligand artifacts.")
    primary = ligand_artifacts[0]
    if "mol2" in primary.files:
        ligand_unit = "ligand_1"
        lines = [f"{ligand_unit} = loadMol2 {Path(primary.files['mol2']).as_posix()}"]
        metal_units: list[str] = []
        for index, metal_pdb in enumerate(_small_molecule_metal_pdb_files(ligand_artifacts), start=1):
            metal_unit = f"metal_{index}"
            tleap_metal_pdb = _write_small_molecule_metal_pdb_for_tleap(
                metal_pdb,
                output_dir,
                system_config,
                f"small_molecule_metal_{index}",
            )
            lines.append(f"{metal_unit} = loadPDB {tleap_metal_pdb.as_posix()}")
            metal_units.append(metal_unit)
        if metal_units:
            lines.append("system = combine { " + " ".join([ligand_unit, *metal_units]) + " }")
        else:
            lines.append(f"system = {ligand_unit}")
        return lines
    if source_files and source_files[0].suffix.lower() == ".pdb":
        tleap_pdb = _write_tleap_pdb_without_conect(source_files[0], output_dir, "source_input")
        return [f"system = loadPDB {tleap_pdb.as_posix()}"]
    raise ValueError("Small-molecule-only tleap build needs a MOL2 artifact or a source PDB file.")


def _has_supported_metals(prepared_pdb: Path | None) -> bool:
    if prepared_pdb is None or not prepared_pdb.exists():
        return False
    return bool(inspect_structure(prepared_pdb, detect_missing_loops=False).metals)


def _configured_metal_charge_values(system_config: SystemConfig) -> list[int]:
    return [int(item.charge) for item in system_config.metal_charges]


def allowed_metal_charges(element: str) -> tuple[int, ...]:
    return SUPPORTED_1264_METAL_CHARGES.get(element.title(), ())


def _validate_supported_metal_charges(system_config: SystemConfig, prepared_pdb: Path | None) -> None:
    if prepared_pdb is None or not prepared_pdb.exists() or not system_config.metal_charges:
        return

    summary = inspect_structure(prepared_pdb, detect_missing_loops=False)
    metal_by_site = {int(site.site): site.element.title() for site in summary.metals}
    for assignment in system_config.metal_charges:
        element = metal_by_site.get(int(assignment.site))
        if element is None:
            continue
        allowed = allowed_metal_charges(element)
        if allowed and int(assignment.charge) not in allowed:
            allowed_text = ", ".join(f"+{value}" for value in allowed)
            raise ValueError(
                f"Metal site {assignment.site} ({element}) does not support the requested oxidation state "
                f"+{assignment.charge} in the current 12-6-4 workflow. Allowed values: {allowed_text}."
            )


def _validate_small_molecule_supported_metal_charges(
    system_config: SystemConfig,
    ligand_artifacts: list[LigandArtifacts],
    source_files: list[Path],
) -> None:
    if not system_config.metal_charges:
        return
    probe_paths: list[Path] = []
    for artifact in ligand_artifacts:
        probe_paths.extend(_loadable_ligand_molecule_paths(artifact))
    for source_path in source_files:
        candidate = source_path.expanduser().resolve()
        if candidate.exists():
            probe_paths.append(candidate)

    seen: set[Path] = set()
    for probe_path in probe_paths:
        if probe_path in seen:
            continue
        seen.add(probe_path)
        try:
            molecule = load_molecule(probe_path)
        except Exception:
            continue
        metal_by_site: dict[int, str] = {}
        site_index = 0
        for atom in molecule.atoms:
            element = str(atom.element or "").strip().title()
            if element not in SUPPORTED_1264_METAL_CHARGES:
                continue
            site_index += 1
            metal_by_site[site_index] = element
        if not metal_by_site:
            continue
        for assignment in system_config.metal_charges:
            element = metal_by_site.get(int(assignment.site))
            if element is None:
                continue
            allowed = allowed_metal_charges(element)
            if allowed and int(assignment.charge) not in allowed:
                allowed_text = ", ".join(f"+{value}" for value in allowed)
                raise ValueError(
                    f"Small-molecule metal site {assignment.site} ({element}) does not support the requested "
                    f"oxidation state +{assignment.charge} in the current 12-6-4 workflow. "
                    f"Allowed values: {allowed_text}."
                )
        return


def ion_parameter_requirements(
    salt_config: SaltConfig,
    *,
    metal_charges: list[int] | None = None,
    prepared_pdb: Path | None = None,
) -> tuple[bool, bool]:
    charges = [int(charge) for charge in (metal_charges or [])]
    include_monovalent = False
    include_multivalent = False
    if charges:
        include_monovalent = any(charge == 1 for charge in charges)
        include_multivalent = any(charge >= 2 for charge in charges)
    elif prepared_pdb is not None and _has_supported_metals(prepared_pdb):
        include_multivalent = True
    return include_monovalent, include_multivalent


def _ion_parameter_files(
    *,
    system_config: SystemConfig,
    amber_env: AmberEnvironment,
    water_model: str,
    include_monovalent: bool,
    include_multivalent: bool,
) -> list[Path]:
    if system_config.custom_ion_frcmods:
        return [Path(path).expanduser().resolve() for path in system_config.custom_ion_frcmods]

    if system_config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL:
        bundled = opc_duvail_ion_frcmod()
        if bundled.exists() and (include_monovalent or include_multivalent):
            return [bundled]

    selected: list[Path] = []
    if include_monovalent:
        for monovalent in amber_env.matching_monovalent_1264_files(
            water_model,
            include_bundled_opc=False,
        ):
            if monovalent not in selected:
                selected.append(monovalent)
    if include_multivalent:
        for multivalent in amber_env.matching_multivalent_1264_files(
            water_model,
            include_bundled_opc=False,
        ):
            if multivalent not in selected:
                selected.append(multivalent)
        if (
            water_model.lower() == "spce"
            and amber_env.ions_frcmod is not None
            and amber_env.ions_frcmod not in selected
        ):
            selected.append(amber_env.ions_frcmod)
    return selected


def _is_monovalent_1264_file(path: Path) -> bool:
    name = path.name.lower()
    return "1264" in name and ("ions1lm_1264" in name or "ionslm_1264" in name)


def _is_multivalent_1264_file(path: Path) -> bool:
    name = path.name.lower()
    return "1264" in name and ("ions234lm_1264" in name or "ionslm_1264" in name)


def _parmed_water_model_name(water_model: str) -> str:
    return PARMED_WATER_MODEL_MAP.get(water_model.lower(), water_model.upper())


def _water_source_and_box(system_config: SystemConfig) -> tuple[str, str]:
    water_key = system_config.water_model.lower()
    mapping = WATER_FORCEFIELD_MAP.get(water_key)
    if mapping is None:
        raise ValueError(
            f"Water model '{system_config.water_model}' is recognized for leaprc loading but does not yet have "
            "a confirmed solvent box mapping in SIMPLE. Please choose a supported water model or add the "
            "correct box-unit mapping before continuing."
        )
    return mapping


def _c4_residue_names(
    *,
    system_config: SystemConfig,
    prepared_pdb: Path | None,
    ion_names: list[str],
    include_monovalent_metals: bool,
    include_multivalent_metals: bool,
) -> list[str]:
    residue_names: list[str] = []
    if prepared_pdb is not None and prepared_pdb.exists():
        summary = inspect_structure(prepared_pdb, detect_missing_loops=False)
        assigned_charges = {int(item.site): int(item.charge) for item in system_config.metal_charges}
        for metal_site in summary.metals:
            element = metal_site.element.title()
            charge = assigned_charges.get(int(metal_site.site))
            if charge is None and assigned_charges:
                continue
            if charge is None:
                include_site = include_multivalent_metals
            elif charge == 1:
                include_site = include_monovalent_metals
            else:
                include_site = include_multivalent_metals
            if not include_site:
                continue
            residue_name = _tleap_metal_label(element, charge)
            if residue_name and residue_name not in residue_names:
                residue_names.append(residue_name)
    for ion_name in ion_names:
        if ion_name and ion_name not in residue_names:
            residue_names.append(ion_name)
    return residue_names


def _c4_mask_from_residue_names(residue_names: list[str]) -> str | None:
    if not residue_names:
        return None
    return ":" + ",".join(residue_names)


def _combine_c4_masks(mask_fragments: list[str]) -> str | None:
    cleaned = [fragment for fragment in mask_fragments if fragment]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return "|".join(f"({fragment})" for fragment in cleaned)


def _c4_charge_included(
    *,
    charge: int | None,
    assigned_charges_present: bool,
    include_monovalent_metals: bool,
    include_multivalent_metals: bool,
) -> bool:
    if charge is None and assigned_charges_present:
        return False
    if charge is None:
        return include_multivalent_metals
    if charge == 1:
        return include_monovalent_metals
    return include_multivalent_metals


def _loadable_ligand_molecule_paths(artifact: LigandArtifacts) -> list[Path]:
    candidates: list[Path] = []
    for key in ("mol2", "canonical_mol2"):
        raw_path = artifact.files.get(key)
        if raw_path:
            candidates.append(Path(raw_path).expanduser())
    if artifact.coordinate_source:
        candidates.append(Path(artifact.coordinate_source).expanduser())
    if artifact.source_file:
        candidates.append(Path(artifact.source_file).expanduser())

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            unique.append(resolved)
    return unique


def _small_molecule_metal_atom_names(
    source_path: Path,
    system_config: SystemConfig,
    *,
    include_monovalent_metals: bool,
    include_multivalent_metals: bool,
) -> list[str]:
    try:
        molecule = load_molecule(source_path)
    except Exception:
        return []
    assigned_charges = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    atom_names: list[str] = []
    site_index = 0
    for atom in molecule.atoms:
        element = str(atom.element or "").strip().title()
        if element not in SUPPORTED_1264_METAL_CHARGES:
            continue
        site_index += 1
        charge = assigned_charges.get(site_index, DEFAULT_TLEAP_METAL_CHARGES.get(element))
        if not _c4_charge_included(
            charge=charge,
            assigned_charges_present=bool(assigned_charges),
            include_monovalent_metals=include_monovalent_metals,
            include_multivalent_metals=include_multivalent_metals,
        ):
            continue
        atom_name = str(atom.name).strip()
        if atom_name and atom_name not in atom_names:
            atom_names.append(atom_name)
    return atom_names


def _small_molecule_metal_residue_names(
    source_path: Path,
    system_config: SystemConfig,
    *,
    include_monovalent_metals: bool,
    include_multivalent_metals: bool,
) -> list[str]:
    explicit_charges = {int(item.site): int(item.charge) for item in system_config.metal_charges}
    residue_names: list[str] = []
    site_index = 0
    for line in source_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = _metal_element_from_pdb_line(line)
        if element is None:
            continue
        site_index += 1
        charge = explicit_charges.get(site_index, DEFAULT_TLEAP_METAL_CHARGES.get(element))
        if not _c4_charge_included(
            charge=charge,
            assigned_charges_present=bool(explicit_charges),
            include_monovalent_metals=include_monovalent_metals,
            include_multivalent_metals=include_multivalent_metals,
        ):
            continue
        label = _tleap_metal_label(element, charge)
        if label and label not in residue_names:
            residue_names.append(label)
    return residue_names


def _small_molecule_c4_masks(
    *,
    system_config: SystemConfig,
    ligand_artifacts: list[LigandArtifacts],
    source_files: list[Path],
    include_monovalent_metals: bool,
    include_multivalent_metals: bool,
) -> list[str]:
    masks: list[str] = []
    metal_residue_names: list[str] = []
    for metal_pdb in _small_molecule_metal_pdb_files(ligand_artifacts):
        for residue_name in _small_molecule_metal_residue_names(
            metal_pdb,
            system_config,
            include_monovalent_metals=include_monovalent_metals,
            include_multivalent_metals=include_multivalent_metals,
        ):
            if residue_name not in metal_residue_names:
                metal_residue_names.append(residue_name)
    if metal_residue_names:
        masks.append(":" + ",".join(metal_residue_names))
    if masks:
        return masks

    parsed_paths: set[Path] = set()
    for artifact in ligand_artifacts:
        for candidate in _loadable_ligand_molecule_paths(artifact):
            parsed_paths.add(candidate)
            atom_names = _small_molecule_metal_atom_names(
                candidate,
                system_config,
                include_monovalent_metals=include_monovalent_metals,
                include_multivalent_metals=include_multivalent_metals,
            )
            if atom_names:
                residue_name = artifact.residue_name.strip() or "LIG"
                masks.append(f":{residue_name}@{','.join(atom_names)}")
                break

    if masks:
        return masks

    residue_name = ligand_artifacts[0].residue_name.strip() if ligand_artifacts else "LIG"
    for source_path in source_files:
        candidate = source_path.expanduser().resolve()
        if candidate in parsed_paths or not candidate.exists():
            continue
        atom_names = _small_molecule_metal_atom_names(
            candidate,
            system_config,
            include_monovalent_metals=include_monovalent_metals,
            include_multivalent_metals=include_multivalent_metals,
        )
        if atom_names:
            masks.append(f":{residue_name}@{','.join(atom_names)}")
            break
    return masks


def _load_polarizability_params(path: Path) -> dict[str, float]:
    params: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = raw_line.split()
        if len(tokens) < 2:
            continue
        try:
            params[tokens[0]] = float(tokens[1])
        except ValueError:
            continue
    return params


def _infer_element_from_atom_type(atom_type: str) -> str | None:
    stripped = "".join(character for character in atom_type.strip() if character.isalpha())
    if not stripped:
        return None
    title = stripped.title()
    for width in (2, 1):
        candidate = title[:width]
        if candidate in SUPPORTED_1264_METAL_CHARGES:
            return candidate
        if candidate in {"Cl", "Br", "Si", "Na", "Mg", "Ca", "Li", "Al", "Zn"}:
            return candidate
    return title[:1].upper()


def _element_fallback_polarizabilities(params: dict[str, float]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for atom_type, value in params.items():
        element = _infer_element_from_atom_type(atom_type)
        if element is None:
            continue
        grouped.setdefault(element, []).append(value)
    fallback: dict[str, float] = {}
    for element, values in grouped.items():
        exact_candidates = [element, element.upper(), element.title()]
        exact_value = next((params[candidate] for candidate in exact_candidates if candidate in params), None)
        fallback[element] = float(exact_value) if exact_value is not None else float(statistics.median(values))
    return fallback


def _iter_mol2_atom_types(path: Path) -> list[tuple[str, str]]:
    atom_types: list[tuple[str, str]] = []
    in_atom_section = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("@<TRIPOS>"):
            in_atom_section = stripped == "@<TRIPOS>ATOM"
            continue
        if not in_atom_section or not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 6:
            continue
        atom_type = tokens[5]
        element = _infer_element_from_atom_type(atom_type) or _infer_element_from_atom_type(tokens[1]) or "C"
        atom_types.append((atom_type, element))
    return atom_types


def _parse_prmtop_format(format_line: str) -> tuple[int, str, int]:
    match = _PRMTOP_FORMAT_PATTERN.match(format_line.strip())
    if match is None:
        raise ValueError(f"Unsupported prmtop format line: {format_line}")
    return int(match.group("count")), match.group("kind").upper(), int(match.group("width"))


def _parse_prmtop_sections(path: Path) -> dict[str, tuple[str, list[str]]]:
    sections: dict[str, tuple[str, list[str]]] = {}
    current_name: str | None = None
    current_format = ""
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("%VERSION"):
            continue
        if line.startswith("%FLAG"):
            if current_name is not None:
                sections[current_name] = (current_format, current_lines)
            current_name = line.split(maxsplit=1)[1].strip()
            current_format = ""
            current_lines = []
            continue
        if line.startswith("%COMMENT"):
            continue
        if line.startswith("%FORMAT"):
            current_format = line
            continue
        if current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        sections[current_name] = (current_format, current_lines)
    return sections


def _prmtop_section_values(sections: dict[str, tuple[str, list[str]]], name: str) -> list[str]:
    section = sections.get(name)
    if section is None:
        return []
    format_line, data_lines = section
    _count, kind, width = _parse_prmtop_format(format_line)
    values: list[str] = []
    if kind == "A":
        for line in data_lines:
            padded = line.rstrip("\n")
            for start in range(0, len(padded), width):
                token = padded[start : start + width].strip()
                if token:
                    values.append(token)
        return values
    for line in data_lines:
        values.extend(token for token in line.split() if token)
    return values


def _prmtop_section_ints(sections: dict[str, tuple[str, list[str]]], name: str) -> list[int]:
    values: list[int] = []
    for token in _prmtop_section_values(sections, name):
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _prmtop_section_floats(sections: dict[str, tuple[str, list[str]]], name: str) -> list[float]:
    values: list[float] = []
    for token in _prmtop_section_values(sections, name):
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return values


def _infer_element_from_mass(mass: float) -> str | None:
    if mass < 0.5:
        return None
    best_element, best_delta = min(
        ((element, abs(reference - mass)) for element, reference in _COMMON_ELEMENT_MASSES_DA.items()),
        key=lambda item: item[1],
    )
    tolerance = 0.45 if mass < 70.0 else 1.25
    return best_element if best_delta <= tolerance else None


def _fallback_element_from_atom_labels(atom_name: str, atom_type: str) -> str | None:
    for raw in (atom_name, atom_type):
        cleaned = "".join(character for character in raw.strip() if character.isalpha())
        if not cleaned:
            continue
        if len(cleaned) >= 2 and cleaned[:2].title() in SUPPORTED_1264_METAL_CHARGES:
            return cleaned[:2].title()
        if cleaned[0].upper() in {"C", "H", "N", "O", "P", "S", "F", "I"}:
            return cleaned[0].upper()
        if len(cleaned) >= 2 and cleaned[:2].title() in {"Cl", "Br", "Si", "Na", "Mg", "Ca", "Al", "Zn"}:
            return cleaned[:2].title()
    return None


def _iter_prmtop_atom_type_elements(path: Path) -> list[tuple[str, str]]:
    sections = _parse_prmtop_sections(path)
    atom_types = _prmtop_section_values(sections, "AMBER_ATOM_TYPE")
    atom_names = _prmtop_section_values(sections, "ATOM_NAME")
    raw_masses = _prmtop_section_values(sections, "MASS")
    resolved: list[tuple[str, str]] = []
    for index, atom_type in enumerate(atom_types):
        atom_type = atom_type.strip()
        if not atom_type:
            continue
        mass: float | None = None
        if index < len(raw_masses):
            try:
                mass = float(raw_masses[index].replace("D", "E").replace("d", "e"))
            except ValueError:
                mass = None
        element = _infer_element_from_mass(mass) if mass is not None else None
        if element is None:
            atom_name = atom_names[index] if index < len(atom_names) else ""
            element = _fallback_element_from_atom_labels(atom_name, atom_type)
        if element is not None:
            resolved.append((atom_type, element))
    return resolved


def _lj_signature_key(a_coefficient: float, b_coefficient: float) -> tuple[str, str]:
    return f"{a_coefficient:.12E}", f"{b_coefficient:.12E}"


def _prmtop_atom_type_lj_signatures(path: Path) -> dict[str, set[tuple[str, str]]]:
    sections = _parse_prmtop_sections(path)
    atom_types = _prmtop_section_values(sections, "AMBER_ATOM_TYPE")
    atom_type_indices = _prmtop_section_ints(sections, "ATOM_TYPE_INDEX")
    nonbonded_index = _prmtop_section_ints(sections, "NONBONDED_PARM_INDEX")
    acoefs = _prmtop_section_floats(sections, "LENNARD_JONES_ACOEF")
    bcoefs = _prmtop_section_floats(sections, "LENNARD_JONES_BCOEF")
    pointers = _prmtop_section_ints(sections, "POINTERS")
    if not atom_types or not atom_type_indices or not nonbonded_index or not acoefs or not bcoefs:
        return {}
    ntypes = pointers[1] if len(pointers) > 1 and pointers[1] > 0 else max(atom_type_indices)
    if len(nonbonded_index) < ntypes * ntypes:
        return {}

    signatures: dict[str, set[tuple[str, str]]] = {}
    for atom_type, type_index in zip(atom_types, atom_type_indices):
        atom_type = atom_type.strip()
        if not atom_type or type_index <= 0 or type_index > ntypes:
            continue
        lookup = nonbonded_index[(type_index - 1) * ntypes + (type_index - 1)]
        if lookup <= 0 or lookup > len(acoefs) or lookup > len(bcoefs):
            continue
        signatures.setdefault(atom_type, set()).add(_lj_signature_key(acoefs[lookup - 1], bcoefs[lookup - 1]))
    return signatures


def _vdw_equivalent_polarizability_fallbacks(
    *,
    params: dict[str, float],
    prmtop_path: Path,
) -> dict[str, tuple[str, float]]:
    signatures_by_type = _prmtop_atom_type_lj_signatures(prmtop_path)
    if not signatures_by_type:
        return {}

    values_by_signature: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for atom_type, signatures in signatures_by_type.items():
        if atom_type not in params or len(signatures) != 1:
            continue
        signature = next(iter(signatures))
        values_by_signature.setdefault(signature, []).append((atom_type, params[atom_type]))

    fallback_by_type: dict[str, tuple[str, float]] = {}
    for atom_type, signatures in signatures_by_type.items():
        if atom_type in params or len(signatures) != 1:
            continue
        signature = next(iter(signatures))
        candidates = values_by_signature.get(signature, [])
        if not candidates:
            continue
        first_type, first_value = candidates[0]
        if any(not math.isclose(value, first_value, rel_tol=1.0e-8, abs_tol=1.0e-8) for _type, value in candidates):
            continue
        fallback_by_type[atom_type] = (first_type, first_value)
    return fallback_by_type


def _write_augmented_polarizability_file(
    *,
    output_dir: Path,
    params: dict[str, float],
) -> Path:
    target = output_dir / "lj_1264_pol_augmented.dat"
    lines = [f"{atom_type} {value:.6f}" for atom_type, value in sorted(params.items())]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _augment_polarizability_file_for_ligands(
    *,
    base_file: Path,
    ligand_artifacts: list[LigandArtifacts],
    output_dir: Path,
) -> tuple[Path, list[str]]:
    base_params = _load_polarizability_params(base_file)
    fallback_by_element = _element_fallback_polarizabilities(base_params)
    augmented = dict(base_params)
    added_messages: list[str] = []

    for artifact in ligand_artifacts:
        raw_mol2 = artifact.files.get("mol2")
        if not raw_mol2:
            continue
        mol2_path = Path(raw_mol2).expanduser().resolve()
        if not mol2_path.exists():
            continue
        for atom_type, element in _iter_mol2_atom_types(mol2_path):
            if atom_type in augmented:
                continue
            fallback = fallback_by_element.get(element)
            if fallback is None:
                continue
            augmented[atom_type] = fallback
            added_messages.append(f"{atom_type} -> {element} ({fallback:.4f})")

    if not added_messages:
        return base_file, []

    target = _write_augmented_polarizability_file(output_dir=output_dir, params=augmented)
    return target, added_messages


def _augment_polarizability_file_for_prmtop_atom_types(
    *,
    base_file: Path,
    prmtop_path: Path,
    output_dir: Path,
) -> tuple[Path, list[str]]:
    if not prmtop_path.exists():
        return base_file, []
    base_params = _load_polarizability_params(base_file)
    fallback_by_element = _element_fallback_polarizabilities(base_params)
    fallback_by_vdw = _vdw_equivalent_polarizability_fallbacks(params=base_params, prmtop_path=prmtop_path)
    augmented = dict(base_params)
    elements_by_type: dict[str, set[str]] = {}
    for atom_type, element in _iter_prmtop_atom_type_elements(prmtop_path):
        if atom_type in augmented:
            continue
        elements_by_type.setdefault(atom_type, set()).add(element)

    added_messages: list[str] = []
    for atom_type in sorted(elements_by_type):
        vdw_fallback = fallback_by_vdw.get(atom_type)
        if vdw_fallback is not None:
            matched_type, fallback = vdw_fallback
            augmented[atom_type] = fallback
            added_messages.append(f"{atom_type} -> {matched_type} ({fallback:.4f}; matched VDW parameters)")
            continue
        elements = elements_by_type[atom_type]
        if len(elements) != 1:
            continue
        element = next(iter(elements))
        fallback = fallback_by_element.get(element)
        if fallback is None:
            continue
        augmented[atom_type] = fallback
        added_messages.append(f"{atom_type} -> {element} ({fallback:.4f})")

    if not added_messages:
        return base_file, []

    target = _write_augmented_polarizability_file(output_dir=output_dir, params=augmented)
    return target, added_messages


def _render_c4_parmed_script(
    *,
    c4_mask: str,
    water_model: str,
    output_dir: Path,
    polarizability_file: Path | None = None,
    c4_file: Path | None = None,
) -> str:
    prmtop = output_dir / "system.prmtop"
    inpcrd = output_dir / "system.inpcrd"
    polfile_clause = f" polfile {polarizability_file.as_posix()}" if polarizability_file is not None else ""
    c4file_clause = f" c4file {c4_file.as_posix()}" if c4_file is not None else ""
    if c4_file is not None:
        command = f"add12_6_4 {c4_mask}{polfile_clause}{c4file_clause}"
    else:
        command = f"add12_6_4 {c4_mask} watermodel {_parmed_water_model_name(water_model)}{polfile_clause}"
    return (
        "setOverwrite True\n"
        f"{command}\n"
        f"outparm {prmtop.as_posix()} {inpcrd.as_posix()}\n"
        "quit\n"
    )


def _parmed_polarizability_file(
    amber_env: AmberEnvironment,
    parameter_set: DESC4ParameterSet,
) -> Path | None:
    if parameter_set == DESC4ParameterSet.OPC_DUVAIL:
        candidate = opc_duvail_polarizability_file()
        if candidate.exists():
            return candidate
    if amber_env.amberhome is None:
        return None
    candidate = amber_env.amberhome / "dat" / "leap" / "parm" / "lj_1264_pol.dat"
    if candidate.exists():
        return candidate
    return None


def _parmed_c4_file(parameter_set: DESC4ParameterSet) -> Path | None:
    if parameter_set != DESC4ParameterSet.OPC_DUVAIL:
        return None
    candidate = opc_duvail_c4_file()
    return candidate if candidate.exists() else None


def _parmed_command(amber_env: AmberEnvironment, script_path: Path, output_dir: Path) -> list[str] | None:
    binary = amber_env.binaries.get("parmed")
    if binary is None or binary.path is None:
        return None
    return [
        str(binary.path),
        "-i",
        str(script_path),
        "-p",
        str(output_dir / "system.prmtop"),
        "-c",
        str(output_dir / "system.inpcrd"),
    ]


def render_tleap_script(
    *,
    system_config: SystemConfig,
    amber_env: AmberEnvironment,
    prepared_pdb: Path | None,
    ligand_artifacts: list[LigandArtifacts],
    source_files: list[Path],
    output_dir: Path,
    extra_ions: dict[str, int] | None,
    neutralizing_ions: dict[str, int] | None,
    save_prefix: str,
    ion_parameter_files: list[Path] | None = None,
    placeholder_salt_comment: bool = False,
) -> str:
    if system_config.metal_model != MetalModel.MODEL_1264:
        raise NotImplementedError(
            f"Metal model '{system_config.metal_model.value}' is not implemented in v1. Use '1264'."
        )

    water_source, water_box = _water_source_and_box(system_config)
    lines: list[str] = []
    if prepared_pdb is not None:
        lines.append(f"source {_protein_forcefield_source(system_config.protein_ff)}")
    lines.extend(
        [
            f"source leaprc.{system_config.ligand_ff}",
            f"source {water_source}",
        ]
    )
    selected_ion_files = ion_parameter_files
    if selected_ion_files is None and system_config.water_model.lower() == "spce" and amber_env.ions_frcmod:
        selected_ion_files = [amber_env.ions_frcmod]
    for ion_file in selected_ion_files or []:
        lines.append(f"loadAmberParams {ion_file.as_posix()}")
    skip_mol2_files: set[str] = set()
    if prepared_pdb is None and ligand_artifacts:
        primary = ligand_artifacts[0]
        if "mol2" in primary.files:
            skip_mol2_files.add(primary.files["mol2"])
    lines.extend(_ligand_template_lines(ligand_artifacts, skip_mol2_files=skip_mol2_files))
    lines.extend(_coordinate_unit_lines(system_config, prepared_pdb, ligand_artifacts, source_files, output_dir))
    lines.append("check system")
    lines.append("charge system")
    if system_config.box_shape == BoxShape.OCT:
        lines.append(f"solvateOct system {water_box} {system_config.buffer_angstrom:.2f}")
    else:
        lines.append(f"solvateBox system {water_box} {system_config.buffer_angstrom:.2f}")
    for ion_map in (neutralizing_ions or {}, extra_ions or {}):
        for ion_name, count in ion_map.items():
            if count > 0:
                lines.append(f"addionsrand system {_tleap_additive_ion_unit_name(ion_name)} {count}")
    if placeholder_salt_comment:
        lines.append("# Neutralizing and/or bulk-salt ion counts are computed at execution time.")
    lines.append("charge system")
    lines.extend(
        [
            f"savePDB system {(output_dir / f'{save_prefix}.pdb').as_posix()}",
            f"saveAmberParm system {(output_dir / f'{save_prefix}.prmtop').as_posix()} {(output_dir / f'{save_prefix}.inpcrd').as_posix()}",
            "quit",
        ]
    )
    return "\n".join(lines) + "\n"


def build_system_with_tleap(
    *,
    system_config: SystemConfig,
    amber_env: AmberEnvironment,
    prepared_pdb: Path | None,
    ligand_artifacts: list[LigandArtifacts],
    source_files: list[Path],
    output_dir: Path,
    dry_run: bool,
) -> LeapBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    commands: list[list[str]] = []
    if system_config.metal_model != MetalModel.MODEL_1264:
        raise NotImplementedError(
            f"Metal model '{system_config.metal_model.value}' is listed but not implemented in v1."
        )
    _validate_supported_metal_charges(system_config, prepared_pdb)
    if prepared_pdb is None:
        _validate_small_molecule_supported_metal_charges(system_config, ligand_artifacts, source_files)
    extra_ions: dict[str, int] = {}
    water_count: int | None = None
    neutralizing_ions: dict[str, int] = {}
    added_ions: dict[str, int] = {}
    volume_angstrom3: float | None = None
    pre_salt_volume_angstrom3: float | None = None
    charge_before_ions: float | None = None
    final_charge: float | None = None
    salt_formula_units: int | None = None
    actual_salt_concentration_m: float | None = None
    c4_script_path: str | None = None
    c4_mask: str | None = None
    c4_applied: bool = False
    include_monovalent, include_multivalent = ion_parameter_requirements(
        system_config.salt,
        metal_charges=_configured_metal_charge_values(system_config),
        prepared_pdb=prepared_pdb,
    )
    if (
        (include_monovalent or include_multivalent)
        and system_config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL
        and system_config.water_model.lower() != "opc"
    ):
        raise ValueError(
            "The OPC + Duvail 12-6-4 parameter set requires water_model='opc'. "
            "Choose the matching 12-6-4 set/water pair."
        )
    ion_parameter_files = _ion_parameter_files(
        system_config=system_config,
        amber_env=amber_env,
        water_model=system_config.water_model,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
    )
    uses_1264_files = any("1264" in path.name.lower() for path in ion_parameter_files)
    uses_monovalent_1264 = any(_is_monovalent_1264_file(path) for path in ion_parameter_files)
    uses_multivalent_1264 = any(_is_multivalent_1264_file(path) for path in ion_parameter_files)
    if (include_monovalent or include_multivalent) and not ion_parameter_files:
        missing_sets: list[str] = []
        if include_monovalent:
            missing_sets.append("monovalent/anion")
        if include_multivalent:
            missing_sets.append("multivalent")
        if dry_run and amber_env.amberhome is None:
            warnings.append(
                f"Could not verify 12-6-4 ion parameter availability for water model '{system_config.water_model}' "
                f"in dry-run mode because AMBERHOME was not detected. Required set(s): {', '.join(missing_sets)}."
            )
        else:
            raise ValueError(
                f"Water model '{system_config.water_model}' cannot be used for the selected 12-6-4 workflow because "
                f"no compatible 12-6-4 ion parameter file was found for the required set(s): {', '.join(missing_sets)}."
            )

    if not dry_run and system_config.salt.mode != SaltMode.NONE:
        ensure_execution_host(dry_run=False)
        pre_script = render_tleap_script(
            system_config=system_config,
            amber_env=amber_env,
            prepared_pdb=prepared_pdb,
            ligand_artifacts=ligand_artifacts,
            source_files=source_files,
            output_dir=output_dir,
            extra_ions=None,
            neutralizing_ions=None,
            save_prefix="system_pre_salt",
            ion_parameter_files=ion_parameter_files,
        )
        pre_script_path = output_dir / "tleap_pre_salt.in"
        pre_script_path.write_text(pre_script, encoding="utf-8")
        pre_command = ["tleap", "-f", str(pre_script_path)]
        commands.append(pre_command)
        pre_result = run_command(pre_command, cwd=output_dir, log_path=output_dir / "tleap_pre_salt.log")
        pre_charge_values = extract_total_charges(f"{pre_result.stdout}\n{pre_result.stderr}")
        if pre_charge_values:
            charge_before_ions = pre_charge_values[-1]
            neutralizing_ions, _ = _predict_neutralizing_ions(
                charge_before_ions,
                system_config.salt.kind,
                system_config.salt.neutralization_ion,
            )
        pre_salt_pdb = output_dir / "system_pre_salt.pdb"
        if pre_salt_pdb.exists():
            water_count = count_waters_in_pdb(pre_salt_pdb)
            pre_salt_volume_angstrom3 = read_pdb_cell_volume(pre_salt_pdb)
        if system_config.salt.mode == SaltMode.COUNT:
            extra_ions = calculate_salt_ions(0, system_config.salt)
        elif system_config.salt.mode == SaltMode.CONCENTRATION:
            if pre_salt_volume_angstrom3 is not None:
                extra_ions = calculate_salt_ions_from_volume(pre_salt_volume_angstrom3, system_config.salt)
            else:
                warnings.append(
                    "Could not read the pre-salt box volume, so concentration-mode salt was approximated from "
                    "the water count instead."
                )
                extra_ions = calculate_salt_ions(water_count or 0, system_config.salt)
    else:
        if system_config.salt.mode == SaltMode.COUNT:
            extra_ions = calculate_salt_ions(0, system_config.salt)

    script_text = render_tleap_script(
        system_config=system_config,
        amber_env=amber_env,
        prepared_pdb=prepared_pdb,
        ligand_artifacts=ligand_artifacts,
        source_files=source_files,
        output_dir=output_dir,
        extra_ions=extra_ions,
        neutralizing_ions=neutralizing_ions,
        save_prefix="system",
        ion_parameter_files=ion_parameter_files,
        placeholder_salt_comment=system_config.salt.mode != SaltMode.NONE and dry_run,
    )

    script_path = output_dir / "tleap.in"
    script_path.write_text(script_text, encoding="utf-8")
    if not dry_run:
        ensure_execution_host(dry_run=False)
        command = ["tleap", "-f", str(script_path)]
        commands.append(command)
        run_result = run_command(command, cwd=output_dir, log_path=output_dir / "tleap.log")
        charge_values = extract_total_charges(f"{run_result.stdout}\n{run_result.stderr}")
        if charge_values:
            charge_before_ions = charge_values[0]
            final_charge = charge_values[-1]
        final_pdb = output_dir / "system.pdb"
        if final_pdb.exists():
            water_count = count_waters_in_pdb(final_pdb)
            volume_angstrom3 = read_pdb_cell_volume(final_pdb)

    if dry_run:
        added_ions = extra_ions.copy()
    else:
        added_ions = _combine_ion_counts(neutralizing_ions, extra_ions)
        salt_formula_units = _salt_formula_units(extra_ions, system_config.salt.kind)
        actual_salt_concentration_m = _salt_concentration_m(salt_formula_units, volume_angstrom3)

    if uses_1264_files:
        c4_ion_names: list[str] = []
        c4_residue_names = _c4_residue_names(
            system_config=system_config,
            prepared_pdb=prepared_pdb,
            ion_names=c4_ion_names,
            include_monovalent_metals=uses_monovalent_1264,
            include_multivalent_metals=uses_multivalent_1264,
        )
        c4_mask = _combine_c4_masks(
            [
                *(
                    [_c4_mask_from_residue_names(c4_residue_names)]
                    if c4_residue_names
                    else []
                ),
                *(
                    _small_molecule_c4_masks(
                        system_config=system_config,
                        ligand_artifacts=ligand_artifacts,
                        source_files=source_files,
                        include_monovalent_metals=uses_monovalent_1264,
                        include_multivalent_metals=uses_multivalent_1264,
                    )
                    if prepared_pdb is None
                    else []
                ),
            ]
        )
        if c4_mask:
            polarizability_file = _parmed_polarizability_file(amber_env, system_config.c4_parameter_set)
            c4_file = _parmed_c4_file(system_config.c4_parameter_set)
            if not dry_run and polarizability_file is None:
                raise RuntimeError(
                    "12-6-4 C4 post-processing requires the Amber polarizability file `lj_1264_pol.dat`, "
                    "but it could not be found under AMBERHOME/dat/leap/parm.\n"
                    f"Detected AMBERHOME: {amber_env.amberhome or 'not set'}"
                )
            augmented_entries: list[str] = []
            if polarizability_file is not None and ligand_artifacts:
                polarizability_file, augmented_entries = _augment_polarizability_file_for_ligands(
                    base_file=polarizability_file,
                    ligand_artifacts=ligand_artifacts,
                    output_dir=output_dir,
                )
            topology_augmented_entries: list[str] = []
            if polarizability_file is not None and not dry_run:
                polarizability_file, topology_augmented_entries = _augment_polarizability_file_for_prmtop_atom_types(
                    base_file=polarizability_file,
                    prmtop_path=output_dir / "system.prmtop",
                    output_dir=output_dir,
                )
            c4_script = _render_c4_parmed_script(
                c4_mask=c4_mask,
                water_model=system_config.water_model,
                output_dir=output_dir,
                polarizability_file=polarizability_file,
                c4_file=c4_file,
            )
            c4_script_file = output_dir / "parmed_1264.in"
            c4_script_file.write_text(c4_script, encoding="utf-8")
            c4_script_path = str(c4_script_file)
            parmed_command = _parmed_command(amber_env, c4_script_file, output_dir)
            if dry_run:
                if parmed_command is not None:
                    commands.append(parmed_command)
                if polarizability_file is None:
                    warnings.append(
                        "Dry-run: Amber `lj_1264_pol.dat` was not detected, so ParmEd add12_6_4 may fail at execution time."
                    )
                if augmented_entries:
                    warnings.append(
                        "Dry-run: generated an augmented 12-6-4 polarizability file for ligand atom types: "
                        + ", ".join(augmented_entries)
                    )
                warnings.append(
                    "Dry-run: generated a ParmEd add12_6_4 helper script but did not execute it."
                )
            else:
                if parmed_command is None:
                    raise RuntimeError(
                        "12-6-4 C4 post-processing requires ParmEd (`parmed`) on the execution host. "
                        f"A helper script was written to {c4_script_file}."
                    )
                commands.append(parmed_command)
                try:
                    run_command(parmed_command, cwd=output_dir, log_path=output_dir / "parmed_1264.log")
                except RuntimeError as exc:
                    if "Could not find parameters for ATOM_TYPE" in str(exc):
                        raise RuntimeError(
                            "ParmEd add12_6_4 could not assign C4 terms because the polarizability file still "
                            "lacks one or more AMBER atom types from the final topology.\n"
                            "Even when the selected mask is only the metal ion, ParmEd needs polarizability "
                            "values for every AMBER_ATOM_TYPE present in the system.\n\n"
                            f"{exc}"
                        ) from exc
                    raise
                if augmented_entries:
                    warnings.append(
                        "Generated an augmented 12-6-4 polarizability file for ligand atom types: "
                        + ", ".join(augmented_entries)
                    )
                if topology_augmented_entries:
                    warnings.append(
                        "Generated an augmented 12-6-4 polarizability file for topology atom types: "
                        + ", ".join(topology_augmented_entries)
                    )
                c4_applied = True

    result = LeapBuildResult(
        script_path=str(script_path),
        output_files={
            "pdb": str(output_dir / "system.pdb"),
            "prmtop": str(output_dir / "system.prmtop"),
            "inpcrd": str(output_dir / "system.inpcrd"),
        },
        warnings=warnings,
        commands=commands,
        water_count=water_count,
        extra_ions=extra_ions,
        neutralizing_ions=neutralizing_ions,
        added_ions=added_ions,
        volume_angstrom3=volume_angstrom3,
        charge_before_ions=charge_before_ions,
        final_charge=final_charge,
        salt_formula_units=salt_formula_units,
        actual_salt_concentration_m=actual_salt_concentration_m,
        c4_script_path=c4_script_path,
        c4_mask=c4_mask,
        c4_applied=c4_applied,
        system_metadata={"c4_parameter_set": system_config.c4_parameter_set.value},
    )
    write_json(output_dir / "system_manifest.json", result.to_dict())
    return result
