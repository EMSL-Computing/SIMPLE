from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil

from amber_metallo.amber.leap import TLEAP_ION_LIBRARY_LABELS
from amber_metallo.execution import run_command
from amber_metallo.reporting import write_json
from amber_metallo.ti.config import TIImplementationMode
from amber_metallo.tool_config import resolve_tool_binary


_PRMTOP_FLAG_PATTERN = re.compile(r"^%FLAG\s+(?P<name>\S+)\s*$")
_PRMTOP_FORMAT_PATTERN = re.compile(r"^%FORMAT\((?P<count>\d+)(?P<kind>[A-Za-z])(?P<width>\d+)(?:\.(?P<precision>\d+))?\)$")
_TI_COMPATIBLE_126_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:1lm|234lm|lm)_126_[^.]+$", re.IGNORECASE)
_MONOVALENT_126_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:1lm|lm)_126_[^.]+$", re.IGNORECASE)
_MULTIVALENT_126_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:234lm|lm)_126_[^.]+$", re.IGNORECASE)
_TI_COMPATIBLE_1264_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:1lm|234lm|lm)_1264_[^.]+$", re.IGNORECASE)
_MONOVALENT_1264_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:1lm|lm)_1264_[^.]+$", re.IGNORECASE)
_MULTIVALENT_1264_FRCMOD_PATTERN = re.compile(r"^frcmod\.ions(?:234lm|lm)_1264_[^.]+$", re.IGNORECASE)
_SIMPLE_ATOM_MASK_PATTERN = re.compile(r"^@(?P<indices>\d+(?:,\d+)*)$")
_FRCMOD_SECTION_HEADERS = {
    "MASS",
    "BOND",
    "ANGLE",
    "DIHE",
    "IMPROPER",
    "NONBON",
    "LJEDIT",
}
_INFERRED_OFFICIAL_LABELS = {
    "CO": "CO",
    "CU1": "CU1",
    "CU": "CU",
    "NI": "NI",
    "MN": "MN",
    "FE2": "FE2",
    "FE": "FE",
    "Y": "Y",
    "LA": "LA",
    "ND": "Nd",
    "EU3": "EU3",
    "LU": "LU",
}


@dataclass(slots=True)
class PrmtopSection:
    format_line: str
    data_lines: list[str]


@dataclass(slots=True)
class LJParameters:
    label: str
    rmin_half: float
    epsilon: float


@dataclass(slots=True)
class PrmtopAtom:
    atom_index: int
    atom_name: str
    residue_label: str
    amber_atom_type: str
    atom_type_index: int


@dataclass(slots=True)
class ChargedPrmtopAtom:
    atom_index: int
    atom_name: str
    residue_index: int
    residue_label: str
    residue_atom_count: int
    charge: float


@dataclass(slots=True)
class PrmtopChargeState:
    net_charge: float
    atoms: list[ChargedPrmtopAtom]

    def monovalent_atoms(self, *, sign: int) -> list[ChargedPrmtopAtom]:
        return [
            atom
            for atom in self.atoms
            if atom.residue_atom_count == 1
            and atom.charge * sign > 0.0
            and abs(abs(atom.charge) - 1.0) <= 0.15
        ]


def is_ti_compatible_custom_126_frcmod(path: str | Path) -> bool:
    return bool(_TI_COMPATIBLE_126_FRCMOD_PATTERN.match(Path(path).name))


def is_ti_compatible_custom_1264_frcmod(path: str | Path) -> bool:
    return bool(_TI_COMPATIBLE_1264_FRCMOD_PATTERN.match(Path(path).name))


def filter_ti_compatible_custom_126_frcmods(paths: list[str] | tuple[str, ...] | None) -> list[Path]:
    if not paths:
        return []
    resolved: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.exists() and is_ti_compatible_custom_126_frcmod(candidate):
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def filter_ti_compatible_custom_1264_frcmods(paths: list[str] | tuple[str, ...] | None) -> list[Path]:
    if not paths:
        return []
    resolved: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.exists() and is_ti_compatible_custom_1264_frcmod(candidate):
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def resolve_official_126_ion_frcmods(
    *,
    amber_env,
    water_model: str,
    custom_ion_frcmods: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    custom_matches = filter_ti_compatible_custom_126_frcmods(list(custom_ion_frcmods or []))
    resolved: list[Path] = []
    for candidate in [*custom_matches, *amber_env.matching_126_files(water_model)]:
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def resolve_1264_ion_frcmods(
    *,
    amber_env,
    water_model: str,
    custom_ion_frcmods: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    custom_matches = filter_ti_compatible_custom_1264_frcmods(list(custom_ion_frcmods or []))
    resolved: list[Path] = []
    for candidate in [*custom_matches, *amber_env.matching_1264_files(water_model)]:
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def missing_required_126_charge_families(
    paths: list[str] | tuple[str, ...] | list[Path] | tuple[Path, ...],
    *,
    formal_charge: int,
) -> list[str]:
    has_monovalent = False
    has_multivalent = False
    for raw_path in paths:
        name = Path(raw_path).name
        if _MONOVALENT_126_FRCMOD_PATTERN.match(name):
            has_monovalent = True
        if _MULTIVALENT_126_FRCMOD_PATTERN.match(name):
            has_multivalent = True
    missing: list[str] = []
    if formal_charge == 1 and not has_monovalent:
        missing.append("monovalent")
    if formal_charge >= 2 and not has_multivalent:
        missing.append("multivalent")
    return missing


def missing_required_1264_charge_families(
    paths: list[str] | tuple[str, ...] | list[Path] | tuple[Path, ...],
    *,
    formal_charge: int,
) -> list[str]:
    has_monovalent = False
    has_multivalent = False
    for raw_path in paths:
        name = Path(raw_path).name
        if _MONOVALENT_1264_FRCMOD_PATTERN.match(name):
            has_monovalent = True
        if _MULTIVALENT_1264_FRCMOD_PATTERN.match(name):
            has_multivalent = True
    missing: list[str] = []
    if formal_charge == 1 and not has_monovalent:
        missing.append("monovalent")
    if formal_charge >= 2 and not has_multivalent:
        missing.append("multivalent")
    return missing


def strip_c4_from_prmtop_text(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    output_lines: list[str] = []
    skipping = False
    removed = False
    for line in lines:
        flag_match = _PRMTOP_FLAG_PATTERN.match(line.strip())
        if flag_match:
            section_name = flag_match.group("name")
            if section_name == "LENNARD_JONES_CCOEF":
                skipping = True
                removed = True
                continue
            if skipping:
                skipping = False
        if skipping:
            continue
        output_lines.append(line)
    return "".join(output_lines), removed


def _parse_prmtop_document(text: str) -> tuple[str, dict[str, PrmtopSection]]:
    version_line = ""
    sections: dict[str, PrmtopSection] = {}
    current_name: str | None = None
    current_format = ""
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("%VERSION"):
            version_line = line
            continue
        if line.startswith("%FLAG"):
            if current_name is not None:
                sections[current_name] = PrmtopSection(format_line=current_format, data_lines=current_lines)
            current_name = line.split(maxsplit=1)[1].strip()
            current_format = ""
            current_lines = []
            continue
        if line.startswith("%COMMENT"):
            # Amber prmtops may include optional comment metadata between
            # %FLAG/%FORMAT and data lines. These are not section payload.
            continue
        if line.startswith("%FORMAT"):
            current_format = line
            continue
        if current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        sections[current_name] = PrmtopSection(format_line=current_format, data_lines=current_lines)
    return version_line, sections


def _parse_format_spec(format_line: str) -> tuple[int, str, int, int | None]:
    match = _PRMTOP_FORMAT_PATTERN.match(format_line.strip())
    if match is None:
        raise ValueError(f"Unsupported prmtop format line: {format_line}")
    return (
        int(match.group("count")),
        match.group("kind").upper(),
        int(match.group("width")),
        None if match.group("precision") is None else int(match.group("precision")),
    )


def _section_values(section: PrmtopSection) -> list[str]:
    count, kind, width, _ = _parse_format_spec(section.format_line)
    values: list[str] = []
    if kind == "A":
        for line in section.data_lines:
            if line.startswith("%COMMENT"):
                continue
            padded = line.rstrip("\n")
            for start in range(0, len(padded), width):
                token = padded[start : start + width]
                if token:
                    values.append(token.strip())
        return values
    for line in section.data_lines:
        if line.startswith("%COMMENT"):
            continue
        values.extend(token for token in line.split() if token)
    return values


def _render_section_values(format_line: str, values: list[object]) -> list[str]:
    count, kind, width, precision = _parse_format_spec(format_line)
    lines: list[str] = []
    for start in range(0, len(values), count):
        chunk = values[start : start + count]
        if kind == "A":
            line = "".join(f"{str(item)[:width]:<{width}}" for item in chunk)
        elif kind == "I":
            line = "".join(f"{int(item):{width}d}" for item in chunk)
        elif kind == "E":
            resolved_precision = 8 if precision is None else precision
            line = "".join(f"{float(item):{width}.{resolved_precision}E}" for item in chunk)
        else:
            raise ValueError(f"Unsupported prmtop data kind: {kind}")
        lines.append(line)
    return lines


def _tokens_to_ints(section: PrmtopSection) -> list[int]:
    return [int(token) for token in _section_values(section)]


def _tokens_to_floats(section: PrmtopSection) -> list[float]:
    return [float(token.replace("D", "E").replace("d", "e")) for token in _section_values(section)]


def _atom_residue_labels(residue_labels: list[str], residue_pointers: list[int], *, natom: int) -> list[str]:
    atom_labels = [""] * natom
    for index, residue_label in enumerate(residue_labels, start=1):
        start = residue_pointers[index - 1]
        end = residue_pointers[index] - 1 if index < len(residue_pointers) else natom
        for atom_index in range(start, end + 1):
            atom_labels[atom_index - 1] = residue_label
    return atom_labels


def _prmtop_atoms(sections: dict[str, PrmtopSection]) -> list[PrmtopAtom]:
    atom_names = _section_values(sections["ATOM_NAME"])
    amber_atom_types = _section_values(sections["AMBER_ATOM_TYPE"])
    atom_type_indices = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
    residue_labels = _section_values(sections["RESIDUE_LABEL"])
    residue_pointers = _tokens_to_ints(sections["RESIDUE_POINTER"])
    natom = len(atom_names)
    atom_residue_labels = _atom_residue_labels(residue_labels, residue_pointers, natom=natom)
    return [
        PrmtopAtom(
            atom_index=index,
            atom_name=atom_names[index - 1].strip(),
            residue_label=atom_residue_labels[index - 1].strip(),
            amber_atom_type=amber_atom_types[index - 1].strip(),
            atom_type_index=atom_type_indices[index - 1],
        )
        for index in range(1, natom + 1)
    ]


def inspect_prmtop_charge_state(path: str | Path) -> PrmtopChargeState:
    _, sections = _parse_prmtop_document(Path(path).read_text(encoding="utf-8", errors="ignore"))
    required = {"ATOM_NAME", "CHARGE", "RESIDUE_LABEL", "RESIDUE_POINTER"}
    missing = sorted(required - set(sections))
    if missing:
        raise ValueError("The prmtop is missing charge-inspection sections: " + ", ".join(missing))
    atom_names = _section_values(sections["ATOM_NAME"])
    charges = [value / 18.2223 for value in _tokens_to_floats(sections["CHARGE"])]
    residue_labels = _section_values(sections["RESIDUE_LABEL"])
    residue_pointers = _tokens_to_ints(sections["RESIDUE_POINTER"])
    atoms: list[ChargedPrmtopAtom] = []
    natom = len(atom_names)
    for residue_index, (label, start) in enumerate(zip(residue_labels, residue_pointers), start=1):
        end = residue_pointers[residue_index] - 1 if residue_index < len(residue_pointers) else natom
        count = end - start + 1
        for atom_index in range(start, end + 1):
            atoms.append(
                ChargedPrmtopAtom(
                    atom_index=atom_index,
                    atom_name=atom_names[atom_index - 1].strip(),
                    residue_index=residue_index,
                    residue_label=label.strip(),
                    residue_atom_count=count,
                    charge=charges[atom_index - 1],
                )
            )
    return PrmtopChargeState(net_charge=math.fsum(charges), atoms=atoms)


def restore_solute_charges_and_c4(
    *,
    source_prmtop: str | Path,
    rebuilt_prmtop: str | Path,
    output_prmtop: str | Path,
) -> dict[str, object]:
    source_version, source_sections = _parse_prmtop_document(
        Path(source_prmtop).read_text(encoding="utf-8", errors="ignore")
    )
    rebuilt_version, rebuilt_sections = _parse_prmtop_document(
        Path(rebuilt_prmtop).read_text(encoding="utf-8", errors="ignore")
    )
    source_atoms = _prmtop_atoms(source_sections)
    rebuilt_atoms = _prmtop_atoms(rebuilt_sections)
    solvent_labels = {"WAT", "HOH", "OPC", "SPC", "TIP3", "TIP4", "TIP5"}
    prefix_count = next(
        (atom.atom_index - 1 for atom in source_atoms if atom.residue_label.upper() in solvent_labels),
        len(source_atoms),
    )
    if len(rebuilt_atoms) < prefix_count:
        raise ValueError("The counterion rebuild removed atoms from the non-solvent prefix.")
    for source_atom, rebuilt_atom in zip(source_atoms[:prefix_count], rebuilt_atoms[:prefix_count], strict=True):
        if (source_atom.atom_name, source_atom.residue_label) != (rebuilt_atom.atom_name, rebuilt_atom.residue_label):
            raise ValueError(
                "The counterion rebuild changed solute atom ordering before atom "
                f"{source_atom.atom_index}: {source_atom.residue_label}/{source_atom.atom_name} -> "
                f"{rebuilt_atom.residue_label}/{rebuilt_atom.atom_name}."
            )
    source_charges = _tokens_to_floats(source_sections["CHARGE"])
    rebuilt_charges = _tokens_to_floats(rebuilt_sections["CHARGE"])
    rebuilt_charges[:prefix_count] = source_charges[:prefix_count]
    rebuilt_sections["CHARGE"].data_lines = _render_section_values(
        rebuilt_sections["CHARGE"].format_line, rebuilt_charges
    )

    c4_transferred = False
    if "LENNARD_JONES_CCOEF" in source_sections:
        source_types = {atom.amber_atom_type: atom.atom_type_index for atom in source_atoms}
        rebuilt_types = {atom.amber_atom_type: atom.atom_type_index for atom in rebuilt_atoms}
        source_ntypes = max(source_types.values(), default=0)
        rebuilt_ntypes = max(rebuilt_types.values(), default=0)
        source_nb = _tokens_to_ints(source_sections["NONBONDED_PARM_INDEX"])
        rebuilt_nb = _tokens_to_ints(rebuilt_sections["NONBONDED_PARM_INDEX"])
        source_c4 = _tokens_to_floats(source_sections["LENNARD_JONES_CCOEF"])
        rebuilt_size = max(rebuilt_nb, default=0)
        if "LENNARD_JONES_CCOEF" in rebuilt_sections:
            rebuilt_c4 = _tokens_to_floats(rebuilt_sections["LENNARD_JONES_CCOEF"])
            if len(rebuilt_c4) < rebuilt_size:
                rebuilt_c4.extend([0.0] * (rebuilt_size - len(rebuilt_c4)))
            else:
                rebuilt_c4 = rebuilt_c4[:rebuilt_size]
        else:
            rebuilt_c4 = [0.0] * rebuilt_size
        for left_label, left_new in rebuilt_types.items():
            left_old = source_types.get(left_label)
            if left_old is None:
                continue
            for right_label, right_new in rebuilt_types.items():
                right_old = source_types.get(right_label)
                if right_old is None:
                    continue
                old_lookup = source_nb[(left_old - 1) * source_ntypes + (right_old - 1)]
                new_lookup = rebuilt_nb[(left_new - 1) * rebuilt_ntypes + (right_new - 1)]
                if old_lookup > 0 and new_lookup > 0 and old_lookup <= len(source_c4):
                    rebuilt_c4[new_lookup - 1] = source_c4[old_lookup - 1]
        rebuilt_sections["LENNARD_JONES_CCOEF"] = PrmtopSection(
            format_line=source_sections["LENNARD_JONES_CCOEF"].format_line,
            data_lines=_render_section_values(source_sections["LENNARD_JONES_CCOEF"].format_line, rebuilt_c4),
        )
        c4_transferred = True

    target = Path(output_prmtop)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_prmtop(rebuilt_version or source_version, rebuilt_sections), encoding="utf-8")
    return {
        "output_prmtop": str(target),
        "restored_charge_atom_count": prefix_count,
        "c4_transferred": c4_transferred,
        "net_charge": inspect_prmtop_charge_state(target).net_charge,
    }


def _pair_coefficients(nonbonded_index: list[int], coefficients: list[float], *, ntypes: int) -> dict[tuple[int, int], float]:
    pairs: dict[tuple[int, int], float] = {}
    for left in range(1, ntypes + 1):
        for right in range(1, ntypes + 1):
            lookup = nonbonded_index[(left - 1) * ntypes + (right - 1)]
            if lookup <= 0:
                continue
            if lookup > len(coefficients):
                raise ValueError("The NONBONDED_PARM_INDEX section references a missing LJ coefficient entry.")
            key = (left, right) if left <= right else (right, left)
            pairs.setdefault(key, coefficients[lookup - 1])
    return pairs


def _c4_dependent_types(
    nonbonded_index: list[int],
    ccoefs: list[float],
    *,
    ntypes: int,
) -> set[int]:
    dependent: set[int] = set()
    for left in range(1, ntypes + 1):
        for right in range(1, ntypes + 1):
            lookup = nonbonded_index[(left - 1) * ntypes + (right - 1)]
            if lookup <= 0 or lookup > len(ccoefs):
                continue
            if abs(ccoefs[lookup - 1]) > 1.0e-8:
                dependent.add(left)
                dependent.add(right)
    return dependent


def _derive_self_lj_parameters(
    pair_a: dict[tuple[int, int], float],
    pair_b: dict[tuple[int, int], float],
    *,
    ntypes: int,
) -> dict[int, LJParameters]:
    resolved: dict[int, LJParameters] = {}
    for type_index in range(1, ntypes + 1):
        acoef = pair_a.get((type_index, type_index), 0.0)
        bcoef = pair_b.get((type_index, type_index), 0.0)
        if abs(acoef) <= 1.0e-12 or abs(bcoef) <= 1.0e-12:
            resolved[type_index] = LJParameters(label=f"TYPE{type_index}", rmin_half=0.0, epsilon=0.0)
            continue
        pair_rmin = (2.0 * acoef / bcoef) ** (1.0 / 6.0)
        epsilon = (bcoef * bcoef) / (4.0 * acoef)
        resolved[type_index] = LJParameters(label=f"TYPE{type_index}", rmin_half=pair_rmin / 2.0, epsilon=epsilon)
    return resolved


def _combine_lj_pair(left: LJParameters, right: LJParameters) -> tuple[float, float]:
    pair_rmin = left.rmin_half + right.rmin_half
    pair_epsilon = math.sqrt(max(left.epsilon, 0.0) * max(right.epsilon, 0.0))
    if pair_rmin <= 0.0 or pair_epsilon <= 0.0:
        return 0.0, 0.0
    return pair_epsilon * (pair_rmin**12), 2.0 * pair_epsilon * (pair_rmin**6)


def _parse_official_126_nonbond(paths: list[Path]) -> dict[str, LJParameters]:
    resolved: dict[str, LJParameters] = {}
    for path in paths:
        current_section: str | None = None
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper in _FRCMOD_SECTION_HEADERS:
                current_section = upper
                continue
            if current_section != "NONBON":
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rmin_half = float(parts[1])
                epsilon = float(parts[2])
            except ValueError:
                continue
            resolved[parts[0].strip().upper()] = LJParameters(
                label=parts[0].strip(),
                rmin_half=rmin_half,
                epsilon=epsilon,
            )
    return resolved


def _official_label_for_charge(*, element: str, charge: int) -> str | None:
    return TLEAP_ION_LIBRARY_LABELS.get((element.title(), int(charge)))


def _normalize_official_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).upper()


def _charge_aware_label_candidates(*, element: str, charge: int) -> list[str]:
    candidates: list[str] = []
    canonical = _official_label_for_charge(element=element, charge=charge)
    if canonical:
        candidates.append(canonical)
        element_token = re.sub(r"[^A-Za-z]", "", element).upper()
        if canonical.upper() == element_token and charge > 1:
            explicit = f"{element_token}{int(charge)}"
            if explicit.upper() not in {item.upper() for item in candidates}:
                candidates.append(explicit)
        decorated = f"{element_token}{int(charge)}+"
        if decorated.upper() not in {item.upper() for item in candidates}:
            candidates.append(decorated)
        decorated_alt = f"{element_token}+{int(charge)}"
        if decorated_alt.upper() not in {item.upper() for item in candidates}:
            candidates.append(decorated_alt)
    return candidates


def _match_official_parameter_label(
    requested_labels: list[str],
    official_parameters: dict[str, LJParameters],
) -> str | None:
    for requested in requested_labels:
        if requested.upper() in official_parameters:
            return official_parameters[requested.upper()].label
    normalized_lookup: dict[str, LJParameters] = {}
    for key, parameter in official_parameters.items():
        normalized_lookup.setdefault(_normalize_official_label(key), parameter)
    for requested in requested_labels:
        normalized = _normalize_official_label(requested)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized].label
    return None


def _resolve_official_type_label(
    atom: PrmtopAtom,
    official_parameters: dict[str, LJParameters],
    *,
    alchemical_atom_index: int | None,
    alchemical_element: str | None,
    alchemical_charge: int | None,
    alchemical_atoms: dict[int, tuple[str, int]] | None = None,
) -> str | None:
    if alchemical_atoms and atom.atom_index in alchemical_atoms:
        element, charge = alchemical_atoms[atom.atom_index]
        return _match_official_parameter_label(
            _charge_aware_label_candidates(element=element, charge=charge),
            official_parameters,
        )
    if atom.atom_index == alchemical_atom_index and alchemical_element and alchemical_charge is not None:
        return _match_official_parameter_label(
            _charge_aware_label_candidates(element=alchemical_element, charge=alchemical_charge),
            official_parameters,
        )

    for candidate in (atom.amber_atom_type, atom.residue_label, atom.atom_name):
        normalized = candidate.strip().upper()
        if not normalized:
            continue
        if normalized in official_parameters:
            return official_parameters[normalized].label
        inferred = _INFERRED_OFFICIAL_LABELS.get(normalized)
        if inferred and inferred.upper() in official_parameters:
            return official_parameters[inferred.upper()].label
    return None


def _render_prmtop(version_line: str, sections: dict[str, PrmtopSection]) -> str:
    lines = [version_line]
    for name, section in sections.items():
        lines.append(f"%FLAG {name}")
        lines.append(section.format_line)
        lines.extend(section.data_lines)
    return "\n".join(lines) + "\n"


def _append_section_value(
    sections: dict[str, PrmtopSection],
    section_name: str,
    value: object,
) -> None:
    if section_name not in sections:
        return
    values = _section_values(sections[section_name])
    values.append(value)
    sections[section_name].data_lines = _render_section_values(sections[section_name].format_line, values)


def _increment_pointer(pointer_values: list[int], index: int, amount: int = 1) -> None:
    if len(pointer_values) > index:
        pointer_values[index] += amount


def _set_type_count(pointer_values: list[int], total_types: int) -> None:
    if len(pointer_values) > 1:
        pointer_values[1] = total_types
    if len(pointer_values) > 18 and pointer_values[18] < total_types:
        pointer_values[18] = total_types


def _qoff_c4off_type_label(type_number: int) -> str:
    return f"Q{type_number % 1000:03d}"[-4:]


def _zero_c4_for_atom_indices(
    *,
    source_text: str,
    atom_indices: list[int],
) -> tuple[str, dict[str, object]]:
    unique_atom_indices = sorted(set(atom_indices))
    if not unique_atom_indices:
        return source_text, {
            "zeroed_lennard_jones_ccoef_for_atom_indices": [],
            "c4_zeroed_atom_type_map": {},
            "created_c4off_atom_type_indices": [],
        }

    version_line, sections = _parse_prmtop_document(source_text)
    if "LENNARD_JONES_CCOEF" not in sections:
        return source_text, {
            "zeroed_lennard_jones_ccoef_for_atom_indices": [],
            "c4_zeroed_atom_type_map": {},
            "created_c4off_atom_type_indices": [],
        }

    required_sections = (
        "POINTERS",
        "ATOM_TYPE_INDEX",
        "AMBER_ATOM_TYPE",
        "NONBONDED_PARM_INDEX",
        "LENNARD_JONES_ACOEF",
        "LENNARD_JONES_BCOEF",
        "LENNARD_JONES_CCOEF",
    )
    missing_sections = [name for name in required_sections if name not in sections]
    if missing_sections:
        raise ValueError(
            "The input prmtop is missing required sections for zero-C4 atom type generation: "
            + ", ".join(missing_sections)
        )

    pointer_values = _tokens_to_ints(sections["POINTERS"])
    atom_type_indices = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
    amber_atom_types = _section_values(sections["AMBER_ATOM_TYPE"])
    atom_count = len(atom_type_indices)
    invalid_indices = [index for index in unique_atom_indices if index < 1 or index > atom_count]
    if invalid_indices:
        raise ValueError(
            "Atom index/indices outside the prmtop atom type range while zeroing C4 terms: "
            + ", ".join(str(index) for index in invalid_indices)
        )

    ntypes = max(atom_type_indices) if atom_type_indices else int(pointer_values[1])
    nonbonded_index = _tokens_to_ints(sections["NONBONDED_PARM_INDEX"])
    acoefs = _tokens_to_floats(sections["LENNARD_JONES_ACOEF"])
    bcoefs = _tokens_to_floats(sections["LENNARD_JONES_BCOEF"])
    ccoefs = _tokens_to_floats(sections["LENNARD_JONES_CCOEF"])
    pair_a = _pair_coefficients(nonbonded_index, acoefs, ntypes=ntypes)
    pair_b = _pair_coefficients(nonbonded_index, bcoefs, ntypes=ntypes)
    pair_c = _pair_coefficients(nonbonded_index, ccoefs, ntypes=ntypes)

    selected_old_types = sorted({atom_type_indices[index - 1] for index in unique_atom_indices})
    old_to_new_type = {old_type: ntypes + offset for offset, old_type in enumerate(selected_old_types, start=1)}
    total_types = ntypes + len(selected_old_types)
    new_to_old_type = {new_type: old_type for old_type, new_type in old_to_new_type.items()}

    for atom_index in unique_atom_indices:
        old_type = atom_type_indices[atom_index - 1]
        new_type = old_to_new_type[old_type]
        atom_type_indices[atom_index - 1] = new_type
        amber_atom_types[atom_index - 1] = _qoff_c4off_type_label(new_type)

    triangular_a: list[float] = []
    triangular_b: list[float] = []
    triangular_c: list[float] = []
    pair_indices: dict[tuple[int, int], int] = {}
    running_index = 1
    for left in range(1, total_types + 1):
        for right in range(left, total_types + 1):
            lookup_left = new_to_old_type.get(left, left)
            lookup_right = new_to_old_type.get(right, right)
            pair_key = (lookup_left, lookup_right) if lookup_left <= lookup_right else (lookup_right, lookup_left)
            pair_indices[(left, right)] = running_index
            triangular_a.append(pair_a.get(pair_key, 0.0))
            triangular_b.append(pair_b.get(pair_key, 0.0))
            triangular_c.append(0.0 if left in new_to_old_type or right in new_to_old_type else pair_c.get(pair_key, 0.0))
            running_index += 1

    rebuilt_nonbonded_index: list[int] = []
    for left in range(1, total_types + 1):
        for right in range(1, total_types + 1):
            pair_key = (left, right) if left <= right else (right, left)
            rebuilt_nonbonded_index.append(pair_indices[pair_key])

    _set_type_count(pointer_values, total_types)
    sections["POINTERS"].data_lines = _render_section_values(sections["POINTERS"].format_line, pointer_values)
    sections["ATOM_TYPE_INDEX"].data_lines = _render_section_values(
        sections["ATOM_TYPE_INDEX"].format_line,
        atom_type_indices,
    )
    sections["AMBER_ATOM_TYPE"].data_lines = _render_section_values(
        sections["AMBER_ATOM_TYPE"].format_line,
        amber_atom_types,
    )
    sections["NONBONDED_PARM_INDEX"].data_lines = _render_section_values(
        sections["NONBONDED_PARM_INDEX"].format_line,
        rebuilt_nonbonded_index,
    )
    sections["LENNARD_JONES_ACOEF"].data_lines = _render_section_values(
        sections["LENNARD_JONES_ACOEF"].format_line,
        triangular_a,
    )
    sections["LENNARD_JONES_BCOEF"].data_lines = _render_section_values(
        sections["LENNARD_JONES_BCOEF"].format_line,
        triangular_b,
    )
    sections["LENNARD_JONES_CCOEF"].data_lines = _render_section_values(
        sections["LENNARD_JONES_CCOEF"].format_line,
        triangular_c,
    )
    if "SOLTY" in sections:
        solty_values = _tokens_to_floats(sections["SOLTY"])
        if len(solty_values) < total_types:
            solty_values.extend([0.0] * (total_types - len(solty_values)))
            sections["SOLTY"].data_lines = _render_section_values(sections["SOLTY"].format_line, solty_values)

    return _render_prmtop(version_line, sections), {
        "zeroed_lennard_jones_ccoef_for_atom_indices": unique_atom_indices,
        "c4_zeroed_atom_type_map": {str(old_type): new_type for old_type, new_type in old_to_new_type.items()},
        "created_c4off_atom_type_indices": list(old_to_new_type.values()),
    }


def _append_qoff_duplicate_atom_to_prmtop(
    *,
    source_text: str,
    alchemical_atom_index: int,
) -> tuple[str, dict[str, object]]:
    version_line, sections = _parse_prmtop_document(source_text)
    required_sections = (
        "POINTERS",
        "ATOM_NAME",
        "CHARGE",
        "MASS",
        "ATOM_TYPE_INDEX",
        "AMBER_ATOM_TYPE",
        "RESIDUE_LABEL",
        "RESIDUE_POINTER",
        "NUMBER_EXCLUDED_ATOMS",
        "EXCLUDED_ATOMS_LIST",
    )
    missing_sections = [name for name in required_sections if name not in sections]
    if missing_sections:
        raise ValueError(
            "The input prmtop is missing required sections for split Q-off duplicate topology generation: "
            + ", ".join(missing_sections)
        )

    pointer_values = _tokens_to_ints(sections["POINTERS"])
    natom = pointer_values[0] if pointer_values else len(_section_values(sections["ATOM_NAME"]))
    if alchemical_atom_index < 1 or alchemical_atom_index > natom:
        raise ValueError(
            f"Selected alchemical atom index @{alchemical_atom_index} is outside the prmtop atom range 1-{natom}."
        )
    duplicate_atom_index = natom + 1

    atoms = _prmtop_atoms(sections)
    source_atom = atoms[alchemical_atom_index - 1]
    source_offset = alchemical_atom_index - 1

    atom_names = _section_values(sections["ATOM_NAME"])
    charges = _tokens_to_floats(sections["CHARGE"])
    masses = _tokens_to_floats(sections["MASS"])
    atom_type_indices = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
    amber_atom_types = _section_values(sections["AMBER_ATOM_TYPE"])
    number_excluded_atoms = _tokens_to_ints(sections["NUMBER_EXCLUDED_ATOMS"])
    excluded_atoms = _tokens_to_ints(sections["EXCLUDED_ATOMS_LIST"])
    residue_labels = _section_values(sections["RESIDUE_LABEL"])
    residue_pointers = _tokens_to_ints(sections["RESIDUE_POINTER"])

    atom_names.append(atom_names[source_offset])
    charges.append(charges[source_offset])
    masses.append(masses[source_offset])
    atom_type_indices.append(atom_type_indices[source_offset])
    amber_atom_types.append(amber_atom_types[source_offset])

    # Amber prmtops use a single 0 entry for atoms with no explicit exclusions.
    number_excluded_atoms.append(1)
    excluded_atoms.append(0)

    duplicate_residue_label = source_atom.residue_label[:4] or "QOF"
    residue_labels.append(duplicate_residue_label)
    residue_pointers.append(duplicate_atom_index)

    sections["ATOM_NAME"].data_lines = _render_section_values(sections["ATOM_NAME"].format_line, atom_names)
    sections["CHARGE"].data_lines = _render_section_values(sections["CHARGE"].format_line, charges)
    sections["MASS"].data_lines = _render_section_values(sections["MASS"].format_line, masses)
    sections["ATOM_TYPE_INDEX"].data_lines = _render_section_values(
        sections["ATOM_TYPE_INDEX"].format_line,
        atom_type_indices,
    )
    sections["AMBER_ATOM_TYPE"].data_lines = _render_section_values(
        sections["AMBER_ATOM_TYPE"].format_line,
        amber_atom_types,
    )
    sections["NUMBER_EXCLUDED_ATOMS"].data_lines = _render_section_values(
        sections["NUMBER_EXCLUDED_ATOMS"].format_line,
        number_excluded_atoms,
    )
    sections["EXCLUDED_ATOMS_LIST"].data_lines = _render_section_values(
        sections["EXCLUDED_ATOMS_LIST"].format_line,
        excluded_atoms,
    )
    sections["RESIDUE_LABEL"].data_lines = _render_section_values(
        sections["RESIDUE_LABEL"].format_line,
        residue_labels,
    )
    sections["RESIDUE_POINTER"].data_lines = _render_section_values(
        sections["RESIDUE_POINTER"].format_line,
        residue_pointers,
    )

    for section_name in ("TREE_CHAIN_CLASSIFICATION", "JOIN_ARRAY", "IROTAT", "RADII", "SCREEN", "ATOMIC_NUMBER"):
        if section_name not in sections:
            continue
        source_values = _section_values(sections[section_name])
        if len(source_values) >= alchemical_atom_index:
            _append_section_value(sections, section_name, source_values[source_offset])

    if "ATOMS_PER_MOLECULE" in sections:
        atoms_per_molecule = _tokens_to_ints(sections["ATOMS_PER_MOLECULE"])
        atoms_per_molecule.append(1)
        sections["ATOMS_PER_MOLECULE"].data_lines = _render_section_values(
            sections["ATOMS_PER_MOLECULE"].format_line,
            atoms_per_molecule,
        )
    if "SOLVENT_POINTERS" in sections:
        solvent_pointers = _tokens_to_ints(sections["SOLVENT_POINTERS"])
        if len(solvent_pointers) >= 2:
            solvent_pointers[1] += 1
            sections["SOLVENT_POINTERS"].data_lines = _render_section_values(
                sections["SOLVENT_POINTERS"].format_line,
                solvent_pointers,
            )

    _increment_pointer(pointer_values, 0, 1)   # NATOM
    _increment_pointer(pointer_values, 10, 1)  # NNB
    _increment_pointer(pointer_values, 11, 1)  # NRES
    if len(pointer_values) > 28:
        pointer_values[28] = max(pointer_values[28], 1)
    sections["POINTERS"].data_lines = _render_section_values(sections["POINTERS"].format_line, pointer_values)

    rendered = _render_prmtop(version_line, sections)
    rendered, c4_metadata = _zero_c4_for_atom_indices(
        source_text=rendered,
        atom_indices=[duplicate_atom_index],
    )

    return rendered, {
        "qoff_original_atom_index": alchemical_atom_index,
        "qoff_duplicate_atom_index": duplicate_atom_index,
        "qoff_timask1": f"@{alchemical_atom_index}",
        "qoff_timask2": f"@{duplicate_atom_index}",
        "qoff_crgmask": f"@{duplicate_atom_index}",
        "qoff_duplicate_residue_label": duplicate_residue_label,
        "qoff_duplicate_c4_zeroed": bool(c4_metadata["zeroed_lennard_jones_ccoef_for_atom_indices"]),
        "qoff_duplicate_c4off_atom_type_indices": c4_metadata["created_c4off_atom_type_indices"],
    }


def _qoff_disjoint_placeholder_metadata(alchemical_atom_index: int) -> dict[str, object]:
    duplicate_atom_index = alchemical_atom_index + 1
    return {
        "qoff_original_atom_index": alchemical_atom_index,
        "qoff_duplicate_atom_index": duplicate_atom_index,
        "qoff_original_atom_indices": [alchemical_atom_index],
        "qoff_duplicate_atom_indices": [duplicate_atom_index],
        "qoff_atom_pairs": [{"original_atom_index": alchemical_atom_index, "duplicate_atom_index": duplicate_atom_index}],
        "qoff_timask1": f"@{alchemical_atom_index}",
        "qoff_timask2": f"@{duplicate_atom_index}",
        "qoff_crgmask": f"@{duplicate_atom_index}",
        "qoff_duplicate_residue_label": "QOF",
    }


def _qoff_disjoint_placeholder_metadata_for_atoms(alchemical_atom_indices: list[int]) -> dict[str, object]:
    original_indices = sorted({int(index) for index in alchemical_atom_indices})
    if not original_indices:
        raise ValueError("At least one alchemical atom index is required for split Q-off topology generation.")
    duplicate_indices = [max(original_indices) + offset for offset in range(1, len(original_indices) + 1)]
    metadata = {
        "qoff_original_atom_index": original_indices[0],
        "qoff_duplicate_atom_index": duplicate_indices[0],
        "qoff_original_atom_indices": original_indices,
        "qoff_duplicate_atom_indices": duplicate_indices,
        "qoff_atom_pairs": [
            {"original_atom_index": original, "duplicate_atom_index": duplicate}
            for original, duplicate in zip(original_indices, duplicate_indices, strict=True)
        ],
        "qoff_timask1": "@" + ",".join(str(index) for index in original_indices),
        "qoff_timask2": "@" + ",".join(str(index) for index in duplicate_indices),
        "qoff_crgmask": "@" + ",".join(str(index) for index in duplicate_indices),
        "qoff_duplicate_residue_label": "QOF",
    }
    return metadata


def _append_qoff_duplicate_atoms_to_prmtop(
    *,
    source_text: str,
    alchemical_atom_indices: list[int],
) -> tuple[str, dict[str, object]]:
    original_indices = sorted({int(index) for index in alchemical_atom_indices})
    if not original_indices:
        raise ValueError("At least one alchemical atom index is required for split Q-off topology generation.")
    rendered = source_text
    pair_metadata: list[dict[str, int]] = []
    duplicate_labels: list[str] = []
    duplicate_c4_zeroed = True
    c4off_type_indices: list[int] = []
    for original_index in original_indices:
        rendered, metadata = _append_qoff_duplicate_atom_to_prmtop(
            source_text=rendered,
            alchemical_atom_index=original_index,
        )
        duplicate_index = int(metadata["qoff_duplicate_atom_index"])
        pair_metadata.append(
            {
                "original_atom_index": original_index,
                "duplicate_atom_index": duplicate_index,
            }
        )
        duplicate_labels.append(str(metadata.get("qoff_duplicate_residue_label") or "QOF"))
        duplicate_c4_zeroed = duplicate_c4_zeroed and bool(metadata.get("qoff_duplicate_c4_zeroed", False))
        c4off_type_indices.extend(int(index) for index in metadata.get("qoff_duplicate_c4off_atom_type_indices") or [])
    duplicate_indices = [item["duplicate_atom_index"] for item in pair_metadata]
    return rendered, {
        "qoff_original_atom_index": original_indices[0],
        "qoff_duplicate_atom_index": duplicate_indices[0],
        "qoff_original_atom_indices": original_indices,
        "qoff_duplicate_atom_indices": duplicate_indices,
        "qoff_atom_pairs": pair_metadata,
        "qoff_timask1": "@" + ",".join(str(index) for index in original_indices),
        "qoff_timask2": "@" + ",".join(str(index) for index in duplicate_indices),
        "qoff_crgmask": "@" + ",".join(str(index) for index in duplicate_indices),
        "qoff_duplicate_residue_label": duplicate_labels[0] if len(set(duplicate_labels)) == 1 else ",".join(duplicate_labels),
        "qoff_duplicate_c4_zeroed": duplicate_c4_zeroed,
        "qoff_duplicate_c4off_atom_type_indices": sorted(set(c4off_type_indices)),
    }


def prepare_qoff_disjoint_topology(
    *,
    input_prmtop: str | Path,
    output_dir: str | Path,
    label: str,
    alchemical_atom_index: int | None = None,
    alchemical_atom_indices: list[int] | None = None,
    dry_run: bool,
) -> dict[str, object]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_prmtop = target_dir / f"{label}_qoff_disjoint.prmtop"
    source = Path(input_prmtop)
    resolved_atom_indices = sorted(
        {
            int(index)
            for index in (alchemical_atom_indices or ([] if alchemical_atom_index is None else [alchemical_atom_index]))
        }
    )
    if not resolved_atom_indices:
        raise ValueError("At least one alchemical atom index is required for split Q-off topology generation.")
    if not source.exists():
        if not dry_run:
            raise FileNotFoundError(f"Input topology not found: {source}")
        output_prmtop.write_text("TI dry-run split Q-off duplicate topology placeholder\n", encoding="utf-8")
        metadata = _qoff_disjoint_placeholder_metadata_for_atoms(resolved_atom_indices)
    else:
        source_text = source.read_text(encoding="utf-8")
        _version_line, sections = _parse_prmtop_document(source_text)
        if dry_run and not sections:
            output_prmtop.write_text("TI dry-run split Q-off duplicate topology placeholder\n", encoding="utf-8")
            metadata = _qoff_disjoint_placeholder_metadata_for_atoms(resolved_atom_indices)
        else:
            rewritten, metadata = _append_qoff_duplicate_atoms_to_prmtop(
                source_text=source_text,
                alchemical_atom_indices=resolved_atom_indices,
            )
            output_prmtop.write_text(rewritten, encoding="utf-8")

    manifest = {
        "input_prmtop": str(source),
        "qoff_prmtop": str(output_prmtop),
        "purpose": "split_qoff_disjoint_two_state",
        **metadata,
    }
    write_json(target_dir / f"{label}_qoff_disjoint_manifest.json", manifest)
    return manifest


def _rewrite_prmtop_to_official_126(
    *,
    source_text: str,
    official_126_frcmods: list[Path],
    alchemical_atom_index: int | None,
    alchemical_element: str | None,
    alchemical_charge: int | None,
    alchemical_atoms: dict[int, tuple[str, int]] | None = None,
) -> tuple[str, dict[str, object]]:
    version_line, sections = _parse_prmtop_document(source_text)
    if "LENNARD_JONES_CCOEF" not in sections:
        return source_text, {
            "removed_lennard_jones_ccoef": False,
            "replaced_atom_count": 0,
            "replaced_atom_labels": [],
        }

    required_sections = (
        "POINTERS",
        "ATOM_NAME",
        "AMBER_ATOM_TYPE",
        "ATOM_TYPE_INDEX",
        "RESIDUE_LABEL",
        "RESIDUE_POINTER",
        "NONBONDED_PARM_INDEX",
        "LENNARD_JONES_ACOEF",
        "LENNARD_JONES_BCOEF",
        "LENNARD_JONES_CCOEF",
    )
    missing_sections = [name for name in required_sections if name not in sections]
    if missing_sections:
        raise ValueError(
            "The input prmtop is missing required sections for TI official 12-6 rebuild: "
            + ", ".join(missing_sections)
        )

    official_parameters = _parse_official_126_nonbond(official_126_frcmods)
    if not official_parameters:
        raise ValueError("No official Amber 12-6 NONBON parameters could be parsed from the selected ion frcmods.")

    pointer_values = _tokens_to_ints(sections["POINTERS"])
    atom_type_indices = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
    amber_atom_types = _section_values(sections["AMBER_ATOM_TYPE"])
    nonbonded_index = _tokens_to_ints(sections["NONBONDED_PARM_INDEX"])
    acoefs = _tokens_to_floats(sections["LENNARD_JONES_ACOEF"])
    bcoefs = _tokens_to_floats(sections["LENNARD_JONES_BCOEF"])
    ccoefs = _tokens_to_floats(sections["LENNARD_JONES_CCOEF"])
    atoms = _prmtop_atoms(sections)
    ntypes = max(atom_type_indices) if atom_type_indices else int(pointer_values[1])
    pair_a = _pair_coefficients(nonbonded_index, acoefs, ntypes=ntypes)
    pair_b = _pair_coefficients(nonbonded_index, bcoefs, ntypes=ntypes)
    self_parameters = _derive_self_lj_parameters(pair_a, pair_b, ntypes=ntypes)
    c4_dependent_types = _c4_dependent_types(nonbonded_index, ccoefs, ntypes=ntypes)

    replacements_by_group: dict[tuple[str, float, float], int] = {}
    replacement_parameters_by_type: dict[int, LJParameters] = {}
    replaced_atom_labels: list[str] = []
    next_type_index = ntypes + 1
    used_c4_dependent_atoms = [atom for atom in atoms if atom.atom_type_index in c4_dependent_types]
    unresolved_atoms: list[str] = []
    alchemical_candidates = (
        _charge_aware_label_candidates(element=alchemical_element, charge=alchemical_charge)
        if alchemical_element and alchemical_charge is not None
        else []
    )
    alchemical_atom_indices = sorted(alchemical_atoms or ([] if alchemical_atom_index is None else [alchemical_atom_index]))
    for atom in used_c4_dependent_atoms:
        target_label = _resolve_official_type_label(
            atom,
            official_parameters,
            alchemical_atom_index=alchemical_atom_index,
            alchemical_element=alchemical_element,
            alchemical_charge=alchemical_charge,
            alchemical_atoms=alchemical_atoms,
        )
        if target_label is None:
            if atom.atom_index in alchemical_atom_indices:
                unresolved_atoms.append(f"{atom.atom_index}:{atom.residue_label}:{atom.atom_name}")
            continue
        target_parameters = official_parameters[target_label.upper()]
        group_key = (target_parameters.label.upper(), target_parameters.rmin_half, target_parameters.epsilon)
        if group_key not in replacements_by_group:
            replacements_by_group[group_key] = next_type_index
            replacement_parameters_by_type[next_type_index] = target_parameters
            next_type_index += 1
        replacement_type_index = replacements_by_group[group_key]
        atom_type_indices[atom.atom_index - 1] = replacement_type_index
        amber_atom_types[atom.atom_index - 1] = target_parameters.label
        replaced_atom_labels.append(f"{atom.atom_index}:{atom.residue_label}:{target_parameters.label}")
    if unresolved_atoms:
        message = (
            "Could not map all 12-6-4-dependent ions/metals to official Amber 12-6 labels: "
            + ", ".join(unresolved_atoms)
        )
        if alchemical_atom_indices:
            matching_labels = sorted(
                parameter.label
                for parameter in official_parameters.values()
                if any(
                    _normalize_official_label(parameter.label).startswith(_normalize_official_label(element))
                    for element, _charge in (alchemical_atoms or {}).values()
                )
            )
            message += (
                f". Tried alchemical label candidates {alchemical_candidates or ['<none>']}"
                f"; matching element labels present in resolved official frcmods: {matching_labels or ['<none>']}."
            )
        raise ValueError(
            message
        )

    total_types = next_type_index - 1
    pair_coefficients_a: dict[tuple[int, int], float] = dict(pair_a)
    pair_coefficients_b: dict[tuple[int, int], float] = dict(pair_b)
    for new_type_index, new_parameters in replacement_parameters_by_type.items():
        for existing_type_index in range(1, total_types + 1):
            if existing_type_index in replacement_parameters_by_type:
                existing_parameters = replacement_parameters_by_type[existing_type_index]
            else:
                existing_parameters = self_parameters.get(existing_type_index)
            if existing_parameters is None:
                raise ValueError(
                    f"Could not derive self LJ parameters for atom type {existing_type_index} while rebuilding TI topology."
                )
            pair_key = (
                (existing_type_index, new_type_index)
                if existing_type_index <= new_type_index
                else (new_type_index, existing_type_index)
            )
            acoef, bcoef = _combine_lj_pair(existing_parameters, new_parameters)
            pair_coefficients_a[pair_key] = acoef
            pair_coefficients_b[pair_key] = bcoef

    triangular_a: list[float] = []
    triangular_b: list[float] = []
    pair_indices: dict[tuple[int, int], int] = {}
    running_index = 1
    for left in range(1, total_types + 1):
        for right in range(left, total_types + 1):
            pair_key = (left, right)
            pair_indices[pair_key] = running_index
            triangular_a.append(pair_coefficients_a.get(pair_key, 0.0))
            triangular_b.append(pair_coefficients_b.get(pair_key, 0.0))
            running_index += 1

    rebuilt_nonbonded_index: list[int] = []
    for left in range(1, total_types + 1):
        for right in range(1, total_types + 1):
            pair_key = (left, right) if left <= right else (right, left)
            rebuilt_nonbonded_index.append(pair_indices[pair_key])

    pointer_values[1] = total_types
    if len(pointer_values) > 18 and pointer_values[18] < total_types:
        pointer_values[18] = total_types
    sections["POINTERS"].data_lines = _render_section_values(sections["POINTERS"].format_line, pointer_values)
    sections["ATOM_TYPE_INDEX"].data_lines = _render_section_values(sections["ATOM_TYPE_INDEX"].format_line, atom_type_indices)
    sections["AMBER_ATOM_TYPE"].data_lines = _render_section_values(sections["AMBER_ATOM_TYPE"].format_line, amber_atom_types)
    sections["NONBONDED_PARM_INDEX"].data_lines = _render_section_values(
        sections["NONBONDED_PARM_INDEX"].format_line,
        rebuilt_nonbonded_index,
    )
    sections["LENNARD_JONES_ACOEF"].data_lines = _render_section_values(
        sections["LENNARD_JONES_ACOEF"].format_line,
        triangular_a,
    )
    sections["LENNARD_JONES_BCOEF"].data_lines = _render_section_values(
        sections["LENNARD_JONES_BCOEF"].format_line,
        triangular_b,
    )
    if "SOLTY" in sections:
        solty_values = _tokens_to_floats(sections["SOLTY"])
        if len(solty_values) < total_types:
            solty_values.extend([0.0] * (total_types - len(solty_values)))
            sections["SOLTY"].data_lines = _render_section_values(sections["SOLTY"].format_line, solty_values)
    del sections["LENNARD_JONES_CCOEF"]
    rewritten = _render_prmtop(version_line, sections)
    return rewritten, {
        "removed_lennard_jones_ccoef": True,
        "replaced_atom_count": len(replaced_atom_labels),
        "replaced_atom_labels": replaced_atom_labels,
    }


def _validate_prmtop_c4_for_gti(
    *,
    source_text: str,
    alchemical_atom_index: int | None,
    alchemical_atom_indices: list[int] | None = None,
) -> dict[str, object]:
    _version_line, sections = _parse_prmtop_document(source_text)
    if "LENNARD_JONES_CCOEF" not in sections:
        raise ValueError(
            "Amber TI 12-6-4 GTI mode requires an input prmtop with a LENNARD_JONES_CCOEF section."
        )

    ccoefs = _tokens_to_floats(sections["LENNARD_JONES_CCOEF"])
    nonzero_c4_count = sum(1 for value in ccoefs if abs(value) > 1.0e-8)
    if nonzero_c4_count == 0:
        raise ValueError(
            "Amber TI 12-6-4 GTI mode found LENNARD_JONES_CCOEF, but all C4 coefficients are zero."
        )

    atom_has_c4: bool | None = None
    atoms_have_c4: dict[str, bool] = {}
    c4_dependent_type_count: int | None = None
    resolved_atom_indices = sorted(
        {
            int(index)
            for index in (alchemical_atom_indices or ([] if alchemical_atom_index is None else [alchemical_atom_index]))
        }
    )
    if resolved_atom_indices:
        required_sections = (
            "ATOM_TYPE_INDEX",
            "NONBONDED_PARM_INDEX",
            "LENNARD_JONES_CCOEF",
        )
        missing_sections = [name for name in required_sections if name not in sections]
        if missing_sections:
            raise ValueError(
                "Amber TI 12-6-4 GTI mode could not verify the selected alchemical atom because "
                "the prmtop is missing required sections: "
                + ", ".join(missing_sections)
            )
        atom_type_indices = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
        invalid = [index for index in resolved_atom_indices if index < 1 or index > len(atom_type_indices)]
        if invalid:
            raise ValueError(
                "Selected alchemical atom index/indices "
                + ", ".join(f"@{index}" for index in invalid)
                + f" are outside the prmtop atom range (1-{len(atom_type_indices)})."
            )
        ntypes = max(atom_type_indices) if atom_type_indices else 0
        nonbonded_index = _tokens_to_ints(sections["NONBONDED_PARM_INDEX"])
        c4_dependent_types = _c4_dependent_types(nonbonded_index, ccoefs, ntypes=ntypes)
        c4_dependent_type_count = len(c4_dependent_types)
        atoms_have_c4 = {
            str(index): atom_type_indices[index - 1] in c4_dependent_types
            for index in resolved_atom_indices
        }
        atom_has_c4 = all(atoms_have_c4.values())
        missing_c4 = [index for index, has_c4 in atoms_have_c4.items() if not has_c4]
        if missing_c4:
            raise ValueError(
                "Amber TI 12-6-4 GTI mode requires every selected alchemical atom to use a C4-dependent "
                "atom type, but no nonzero C4 pair coefficient was found for: "
                + ", ".join(f"@{index}" for index in missing_c4)
            )

    return {
        "preserved_lennard_jones_ccoef": True,
        "nonzero_c4_coefficient_count": nonzero_c4_count,
        "c4_dependent_type_count": c4_dependent_type_count,
        "alchemical_atom_has_c4": atom_has_c4,
        "alchemical_atoms_have_c4": atoms_have_c4,
        "removed_lennard_jones_ccoef": False,
        "replaced_atom_count": 0,
        "replaced_atom_labels": [],
    }


def prepare_ti_input_topology(
    *,
    input_prmtop: str | Path,
    output_dir: str | Path,
    label: str,
    implementation_mode: TIImplementationMode,
    dry_run: bool,
    water_model: str,
    official_126_frcmods: list[str] | tuple[str, ...],
    alchemical_atom_index: int | None = None,
    alchemical_element: str | None = None,
    alchemical_charge: int | None = None,
    alchemical_atom_indices: list[int] | None = None,
    alchemical_elements_by_index: dict[int, str] | None = None,
    alchemical_charges_by_index: dict[int, int] | None = None,
) -> dict[str, object]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_prmtop = target_dir / f"{label}_ti_input.prmtop"
    source = Path(input_prmtop)

    if implementation_mode == TIImplementationMode.GROMACS_TABULATED_12_6_4:
        raise NotImplementedError(
            f"TI implementation mode '{implementation_mode.value}' is not implemented in the topology writer."
        )

    resolved_official_126_frcmods = [str(Path(path).expanduser().resolve()) for path in official_126_frcmods]
    resolved_atom_indices = sorted(
        {
            int(index)
            for index in (alchemical_atom_indices or ([] if alchemical_atom_index is None else [alchemical_atom_index]))
        }
    )
    alchemical_atoms: dict[int, tuple[str, int]] = {}
    for index in resolved_atom_indices:
        element = None if alchemical_elements_by_index is None else alchemical_elements_by_index.get(index)
        charge = None if alchemical_charges_by_index is None else alchemical_charges_by_index.get(index)
        if element is None and index == alchemical_atom_index:
            element = alchemical_element
        if charge is None and index == alchemical_atom_index:
            charge = alchemical_charge
        if element is not None and charge is not None:
            alchemical_atoms[index] = (str(element), int(charge))
    if not source.exists():
        if dry_run:
            output_prmtop.write_text("TI dry-run topology placeholder\n", encoding="utf-8")
            rewrite_metadata = {
                "preserved_lennard_jones_ccoef": False,
                "removed_lennard_jones_ccoef": False,
                "replaced_atom_count": 0,
                "replaced_atom_labels": [],
                "nonzero_c4_coefficient_count": None,
                "c4_dependent_type_count": None,
                "alchemical_atom_has_c4": None,
                "alchemical_atoms_have_c4": {},
            }
        else:
            raise FileNotFoundError(f"Input topology not found: {source}")
    elif implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI:
        source_text = source.read_text(encoding="utf-8")
        rewrite_metadata = _validate_prmtop_c4_for_gti(
            source_text=source_text,
            alchemical_atom_index=alchemical_atom_index,
            alchemical_atom_indices=resolved_atom_indices,
        )
        output_prmtop.write_text(source_text, encoding="utf-8")
    else:
        rewritten_text, rewrite_metadata = _rewrite_prmtop_to_official_126(
            source_text=source.read_text(encoding="utf-8"),
            official_126_frcmods=[Path(path) for path in resolved_official_126_frcmods],
            alchemical_atom_index=alchemical_atom_index,
            alchemical_element=alchemical_element,
            alchemical_charge=alchemical_charge,
            alchemical_atoms=alchemical_atoms,
        )
        output_prmtop.write_text(rewritten_text, encoding="utf-8")

    manifest = {
        "input_prmtop": str(source),
        "ti_prmtop": str(output_prmtop),
        "implementation_mode": implementation_mode.value,
        "water_model": water_model,
        "official_12_6_frcmods": resolved_official_126_frcmods,
        "alchemical_atom_index": alchemical_atom_index,
        "alchemical_atom_indices": resolved_atom_indices,
        "alchemical_element": alchemical_element,
        "alchemical_charge": alchemical_charge,
        "alchemical_elements_by_index": {str(key): value[0] for key, value in alchemical_atoms.items()},
        "alchemical_charges_by_index": {str(key): value[1] for key, value in alchemical_atoms.items()},
        **rewrite_metadata,
    }
    write_json(target_dir / f"{label}_ti_topology_manifest.json", manifest)
    return manifest


def render_decharge_parmed_script(*, atom_mask: str, output_prmtop: str | Path) -> str:
    return (
        f"change charge {atom_mask} 0.0 quiet\n"
        f"outparm {Path(output_prmtop).as_posix()}\n"
        "quit\n"
    )


def _atom_indices_from_simple_mask(atom_mask: str) -> list[int]:
    match = _SIMPLE_ATOM_MASK_PATTERN.match(atom_mask.strip())
    if match is None:
        raise ValueError(
            "C4-preserving decharge currently supports simple Amber atom masks like '@1' or '@1,2'. "
            f"Received: {atom_mask}"
        )
    indices = [int(token) for token in match.group("indices").split(",")]
    if any(index < 1 for index in indices):
        raise ValueError(f"Atom mask contains an invalid atom index: {atom_mask}")
    return sorted(set(indices))


def _zero_prmtop_charges_for_atom_mask(*, source_text: str, atom_mask: str) -> tuple[str, dict[str, object]]:
    version_line, sections = _parse_prmtop_document(source_text)
    if "CHARGE" not in sections:
        raise ValueError("The prmtop is missing a CHARGE section, so atom charges cannot be zeroed directly.")

    atom_indices = _atom_indices_from_simple_mask(atom_mask)
    charges = _tokens_to_floats(sections["CHARGE"])
    max_index = len(charges)
    invalid = [index for index in atom_indices if index > max_index]
    if invalid:
        raise ValueError(
            f"Atom mask {atom_mask} references atom index/indices outside the CHARGE section range 1-{max_index}: "
            + ", ".join(str(index) for index in invalid)
        )
    for atom_index in atom_indices:
        charges[atom_index - 1] = 0.0
    sections["CHARGE"].data_lines = _render_section_values(sections["CHARGE"].format_line, charges)
    return _render_prmtop(version_line, sections), {
        "decharge_method": "direct_prmtop_charge_section",
        "decharged_atom_indices": atom_indices,
        "preserved_lennard_jones_ccoef": "LENNARD_JONES_CCOEF" in sections,
    }


def prepare_decharged_topology(
    *,
    input_prmtop: str | Path,
    atom_mask: str,
    output_dir: str | Path,
    label: str,
    dry_run: bool,
    preserve_c4: bool = False,
) -> dict[str, object]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_prmtop = target_dir / f"{label}_decharged.prmtop"
    script_path = target_dir / f"{label}_decharge.parmed.in"
    script_path.write_text(
        render_decharge_parmed_script(atom_mask=atom_mask, output_prmtop=output_prmtop),
        encoding="utf-8",
    )
    source = Path(input_prmtop)
    decharge_metadata: dict[str, object] = {
        "decharge_method": "parmed",
        "decharged_atom_indices": [],
        "preserved_lennard_jones_ccoef": False,
    }
    if preserve_c4:
        if source.exists():
            source_text = source.read_text(encoding="utf-8")
            try:
                rewritten, decharge_metadata = _zero_prmtop_charges_for_atom_mask(
                    source_text=source_text,
                    atom_mask=atom_mask,
                )
                rewritten, c4_metadata = _zero_c4_for_atom_indices(
                    source_text=rewritten,
                    atom_indices=list(decharge_metadata["decharged_atom_indices"]),
                )
                decharge_metadata = {**decharge_metadata, **c4_metadata}
                output_prmtop.write_text(rewritten, encoding="utf-8")
            except ValueError:
                if not dry_run:
                    raise
                shutil.copy2(source, output_prmtop)
                decharge_metadata = {
                    "decharge_method": "dry_run_copy_without_charge_edit",
                    "decharged_atom_indices": [],
                    "preserved_lennard_jones_ccoef": "LENNARD_JONES_CCOEF" in source_text,
                }
        elif dry_run:
            output_prmtop.write_text("TI dry-run decharged topology placeholder\n", encoding="utf-8")
            decharge_metadata["decharge_method"] = "dry_run_placeholder"
        else:
            raise FileNotFoundError(f"Input topology not found: {source}")
    elif dry_run:
        if source.exists():
            shutil.copy2(source, output_prmtop)
        else:
            output_prmtop.write_text("TI dry-run decharged topology placeholder\n", encoding="utf-8")
        decharge_metadata["decharge_method"] = "dry_run_copy"
    else:
        binary = resolve_tool_binary("parmed", path_finder=shutil.which)
        if binary is None:
            raise RuntimeError("ParmEd (`parmed`) was not found on PATH, so the decharged topology could not be built.")
        run_command(
            [binary, "-p", str(input_prmtop), "-i", str(script_path)],
            cwd=target_dir,
            log_path=target_dir / f"{label}_decharge.parmed.log",
        )
    manifest = {
        "script": str(script_path),
        "decharged_prmtop": str(output_prmtop),
        "atom_mask": atom_mask,
        **decharge_metadata,
    }
    write_json(target_dir / f"{label}_decharge_manifest.json", manifest)
    return manifest
