from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import runpy
from typing import Any

from amber_metallo.config import SlurmConfig, SlurmProfile
from amber_metallo.environment import detect_amber_environment
from amber_metallo.inspection import load_structure, residue_key
from amber_metallo.reporting import console, print_notice, write_json
from amber_metallo.ti.analysis import default_formal_charge, detect_bound_metal_sites, parse_cntrl_settings, select_site
from amber_metallo.tool_config import ToolConfig, ambertools_sbatch_setup

from amber_metallo.free_energy.config import (
    FreeEnergyWorkflowConfig,
    MMPBSAConfig,
    MMPBSAEntropyMethod,
    MMPBSALigandSelectionMode,
    MMPBSAReceptorSelectionMode,
)
from amber_metallo.free_energy.trajectory import count_trajectory_frames


_SOLVENT_ION_STRIP_MASK = ":WAT,HOH,Na+,Cl-,K+,Rb+,Cs+,Li+,F-,Br-,I-,Mg2+,Ca2+,Sr2+,Ba2+"
_MMPBSA_EXISTING_ASSET_MARKERS = (
    "inputs/MMPBSA.in",
    "prep",
    "slurm",
    "output",
    "manifest.json",
)
_MMPBSA_SECTION_HEADINGS = {
    "GENERALIZED BORN": ("gb", "MM-GBSA"),
    "POISSON BOLTZMANN": ("pb", "MM-PBSA"),
}
_BATCH_LOGS_DIR_NAME = "LOGS_MMPBSA"
_MMPBSA_PH_DIR_RE = re.compile(r"^PH", re.IGNORECASE)
_DELTA_TOTAL_RE = re.compile(
    r"DELTA TOTAL\s+([-+0-9Ee.]+)\s+([-+0-9Ee.]+)\s+([-+0-9Ee.]+)"
)
_DELTA_G_BINDING_RE = re.compile(
    r"DELTA G binding\s*=\s*([-+0-9Ee.]+)\s*\+/-\s*([-+0-9Ee.]+)\s*([-+0-9Ee.]+)"
)
_DECOMP_VALUE_RE = re.compile(r"([-+0-9Ee.]+)\s*\+/-\s*([-+0-9Ee.]+)")
_DECOMP_RESIDUE_RE = re.compile(r"^(?P<name>[A-Za-z0-9+\-]+)\s+(?P<index>\d+)$")
_PRMTOP_FORMAT_PATTERN = re.compile(r"^%FORMAT\((?P<count>\d+)(?P<kind>[A-Za-z])(?P<width>\d+)(?:\.(?P<precision>\d+))?\)$")
_SOLVENT_ION_NAMES = {
    token.strip().upper()
    for token in _SOLVENT_ION_STRIP_MASK.removeprefix(":").split(",")
    if token.strip()
}
_RESIDUE_ONE_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "ASH": "D",
    "CYS": "C",
    "CYM": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLH": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "LYN": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
_TAHOMA_CONDA_AMBERTOOLS_LINES = [
    "SIMPLE_CONDA_ENV=${SIMPLE_CONDA_ENV:-${CONDA_DEFAULT_ENV:-metal}}",
    "_SIMPLE_CONDA_READY=0",
    "if [ -n \"${CONDA_PREFIX:-}\" ] && [ \"${CONDA_DEFAULT_ENV:-}\" = \"$SIMPLE_CONDA_ENV\" ]; then",
    "  _SIMPLE_CONDA_READY=1",
    "fi",
    "",
    "if [ \"$_SIMPLE_CONDA_READY\" -eq 0 ]; then",
    "  _SIMPLE_CONDA_SH=\"\"",
    "  for conda_sh in \"$HOME/miniforge3/etc/profile.d/conda.sh\" \"$HOME/miniconda3/etc/profile.d/conda.sh\" \"$HOME/anaconda3/etc/profile.d/conda.sh\"; do",
    "    if [ -r \"$conda_sh\" ]; then",
    "      _SIMPLE_CONDA_SH=\"$conda_sh\"",
    "      break",
    "    fi",
    "  done",
    "  if [ -n \"$_SIMPLE_CONDA_SH\" ]; then",
    "    source \"$_SIMPLE_CONDA_SH\"",
    "  elif command -v conda >/dev/null 2>&1; then",
    "    eval \"$(conda shell.bash hook)\" || exit 1",
    "  fi",
    "",
    "  if command -v conda >/dev/null 2>&1; then",
    "    conda activate \"$SIMPLE_CONDA_ENV\" || exit 1",
    "  elif command -v micromamba >/dev/null 2>&1; then",
    "    eval \"$(micromamba shell hook -s bash)\" || exit 1",
    "    micromamba activate \"$SIMPLE_CONDA_ENV\" || exit 1",
    "  else",
    "    echo \"Could not find conda or micromamba to activate SIMPLE_CONDA_ENV=$SIMPLE_CONDA_ENV\" >&2",
    "    exit 1",
    "  fi",
    "fi",
    "export PYTHONNOUSERSITE=1",
    "hash -r",
]


@dataclass(frozen=True, slots=True)
class MMPBSAPrmtopResidue:
    original_topology_index: int
    dry_topology_index: int | None
    residue_name: str
    first_atom_index: int
    last_atom_index: int

    @property
    def atom_count(self) -> int:
        return self.last_atom_index - self.first_atom_index + 1

    @property
    def is_solvent_or_ion(self) -> bool:
        return self.residue_name.strip().upper() in _SOLVENT_ION_NAMES


@dataclass(frozen=True, slots=True)
class MMPBSAResidueSummary:
    residue_name: str
    residue_count: int
    atom_count: int
    dry_residue_indices: tuple[int, ...]
    original_residue_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "residue_name": self.residue_name,
            "residue_count": self.residue_count,
            "atom_count": self.atom_count,
            "dry_residue_indices": list(self.dry_residue_indices),
            "original_residue_indices": list(self.original_residue_indices),
        }


def _residue_summary_label(residue_name: str, seqid: str) -> str:
    prefix = _RESIDUE_ONE_LETTER.get(residue_name.strip().upper(), residue_name.strip()[:1].upper() or "X")
    return f"{prefix}{str(seqid).strip().replace(' ', '')}"


def _parse_prmtop_sections(path: str | Path) -> dict[str, tuple[str, list[str]]]:
    sections: dict[str, tuple[str, list[str]]] = {}
    current_name: str | None = None
    current_format = ""
    current_lines: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip("\n")
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


def _parse_prmtop_format(format_line: str) -> tuple[int, str, int]:
    match = _PRMTOP_FORMAT_PATTERN.match(format_line.strip())
    if match is None:
        raise ValueError(f"Unsupported prmtop format line: {format_line}")
    return int(match.group("count")), match.group("kind").upper(), int(match.group("width"))


def _prmtop_section_values(sections: dict[str, tuple[str, list[str]]], name: str) -> list[str]:
    if name not in sections:
        raise ValueError(f"Missing required prmtop section: {name}")
    format_line, data_lines = sections[name]
    _, kind, width = _parse_prmtop_format(format_line)
    if kind == "A":
        values: list[str] = []
        for line in data_lines:
            padded = line.rstrip("\n")
            for start in range(0, len(padded), width):
                token = padded[start : start + width]
                if token:
                    values.append(token.strip())
        return values
    values = []
    for line in data_lines:
        values.extend(token for token in line.split() if token)
    return values


def _prmtop_residues(path: str | Path) -> list[MMPBSAPrmtopResidue]:
    sections = _parse_prmtop_sections(path)
    residue_labels = [item.strip().upper() for item in _prmtop_section_values(sections, "RESIDUE_LABEL")]
    residue_pointers = [int(item) for item in _prmtop_section_values(sections, "RESIDUE_POINTER")]
    atom_names = _prmtop_section_values(sections, "ATOM_NAME")
    if not residue_labels or not residue_pointers or not atom_names:
        raise ValueError(f"{path} does not contain enough topology metadata for residue selection.")
    residues: list[MMPBSAPrmtopResidue] = []
    dry_index = 0
    natom = len(atom_names)
    for index, residue_name in enumerate(residue_labels, start=1):
        first_atom = residue_pointers[index - 1]
        last_atom = residue_pointers[index] - 1 if index < len(residue_pointers) else natom
        normalized = residue_name.strip().upper()
        is_solvent_or_ion = normalized in _SOLVENT_ION_NAMES
        if not is_solvent_or_ion:
            dry_index += 1
        residues.append(
            MMPBSAPrmtopResidue(
                original_topology_index=index,
                dry_topology_index=None if is_solvent_or_ion else dry_index,
                residue_name=normalized,
                first_atom_index=first_atom,
                last_atom_index=last_atom,
            )
        )
    return residues


def summarize_mmpbsa_prmtop_residues(path: str | Path) -> list[MMPBSAResidueSummary]:
    grouped: dict[str, list[MMPBSAPrmtopResidue]] = {}
    for residue in _prmtop_residues(path):
        if residue.is_solvent_or_ion:
            continue
        grouped.setdefault(residue.residue_name, []).append(residue)
    summaries = [
        MMPBSAResidueSummary(
            residue_name=name,
            residue_count=len(residues),
            atom_count=sum(item.atom_count for item in residues),
            dry_residue_indices=tuple(int(item.dry_topology_index) for item in residues if item.dry_topology_index is not None),
            original_residue_indices=tuple(item.original_topology_index for item in residues),
        )
        for name, residues in grouped.items()
    ]
    return sorted(summaries, key=lambda item: (item.residue_name, item.dry_residue_indices))


def _selected_residue_name_metadata(prmtop_path: str | Path, residue_names: list[str]) -> list[dict[str, Any]]:
    requested = {name.strip().upper() for name in residue_names if name.strip()}
    if not requested:
        raise ValueError("At least one ligand residue name is required for residue-name MM-PBSA mode.")
    residues = _prmtop_residues(prmtop_path)
    selected = [
        residue
        for residue in residues
        if not residue.is_solvent_or_ion and residue.residue_name.strip().upper() in requested
    ]
    found = {item.residue_name.strip().upper() for item in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "Ligand residue names were not found among dry solute residues in the topology: "
            + ", ".join(missing)
        )
    metadata = [
        {
            "topology_index": int(residue.dry_topology_index),
            "original_topology_index": residue.original_topology_index,
            "summary_label": f"{residue.residue_name}{residue.dry_topology_index}",
            "residue_name": residue.residue_name,
            "chain": "",
            "seqid": str(residue.dry_topology_index),
            "residue_key": f"prmtop:{residue.residue_name}:{residue.original_topology_index}",
            "atom_count": residue.atom_count,
        }
        for residue in selected
        if residue.dry_topology_index is not None
    ]
    metadata.sort(key=lambda item: int(item["topology_index"]))
    return metadata


@dataclass(frozen=True, slots=True)
class MMPBSALigandPlan:
    selection_mode: str
    ligand_mask: str
    receptor_mask: str | None
    receptor_parmed_strip_mask: str
    receptor_policy: str
    binding_residues: list[dict[str, Any]]
    selected_ligand_residues: list[dict[str, Any]]
    selected_receptor_residues: list[dict[str, Any]]
    decomp_print_res: str
    selected_metal_label: str | None = None
    selected_site_payload: dict[str, Any] | None = None
    selected_formal_charge: int | None = None


def _build_metal_site_ligand_plan(
    *,
    config: FreeEnergyWorkflowConfig,
    selected: Any,
    formal_charge: int,
) -> MMPBSALigandPlan:
    ligand_mask = f"@{selected.atom_index}"
    binding_residues = _selected_binding_residues(config.complex_input.reference_structure_path, selected)
    selected_metal_residue = _selected_metal_residue(config.complex_input.reference_structure_path, selected)
    decomp_print_indices = [int(item["topology_index"]) for item in binding_residues]
    selected_ligand_residues: list[dict[str, Any]] = []
    if selected_metal_residue is not None and selected_metal_residue.get("topology_index") is not None:
        decomp_print_indices.append(int(selected_metal_residue["topology_index"]))
        selected_ligand_residues.append(
            {
                **selected_metal_residue,
                "summary_label": _residue_summary_label(
                    str(selected_metal_residue.get("residue_name") or selected.residue_name),
                    str(selected_metal_residue.get("seqid") or selected.seqid),
                ),
            }
        )
    return MMPBSALigandPlan(
        selection_mode=MMPBSALigandSelectionMode.METAL_SITE.value,
        ligand_mask=ligand_mask,
        receptor_mask=None,
        receptor_parmed_strip_mask=ligand_mask,
        receptor_policy="dry solute minus selected metal atom",
        binding_residues=binding_residues,
        selected_ligand_residues=selected_ligand_residues,
        selected_receptor_residues=[],
        decomp_print_res=_compress_residue_indices(decomp_print_indices),
        selected_metal_label=f"site {selected.site} ({selected.element} at {selected.key})",
        selected_site_payload=selected.to_dict(),
        selected_formal_charge=formal_charge,
    )


def _build_residue_name_ligand_plan(config: FreeEnergyWorkflowConfig) -> MMPBSALigandPlan:
    selected_ligand_residues = _selected_residue_name_metadata(
        config.complex_input.prmtop_path,
        config.mmpbsa.ligand_residue_names,
    )
    dry_indices = [int(item["topology_index"]) for item in selected_ligand_residues]
    ligand_mask = f":{_compress_residue_indices(dry_indices)}"
    selected_receptor_residues: list[dict[str, Any]] = []
    receptor_mask: str | None = None
    receptor_parmed_strip_mask = ligand_mask
    receptor_policy = "dry solute minus ligand residue mask"
    if config.mmpbsa.receptor_selection_mode == MMPBSAReceptorSelectionMode.RESIDUE_NAME:
        selected_receptor_residues = _selected_residue_name_metadata(
            config.complex_input.prmtop_path,
            config.mmpbsa.receptor_residue_names,
        )
        receptor_indices = [int(item["topology_index"]) for item in selected_receptor_residues]
        receptor_mask = f":{_compress_residue_indices(receptor_indices)}"
        receptor_parmed_strip_mask = f"!({receptor_mask})"
        receptor_policy = "manual receptor residue mask"
    return MMPBSALigandPlan(
        selection_mode=MMPBSALigandSelectionMode.RESIDUE_NAME.value,
        ligand_mask=ligand_mask,
        receptor_mask=receptor_mask,
        receptor_parmed_strip_mask=receptor_parmed_strip_mask,
        receptor_policy=receptor_policy,
        binding_residues=selected_ligand_residues,
        selected_ligand_residues=selected_ligand_residues,
        selected_receptor_residues=selected_receptor_residues,
        decomp_print_res=_compress_residue_indices(dry_indices),
    )


def _residue_index_metadata_by_key(reference_structure_path: str | Path) -> dict[str, dict[str, Any]]:
    structure = load_structure(reference_structure_path)
    residue_index_by_key: dict[str, dict[str, Any]] = {}
    topology_index = 1
    for chain in structure[0]:
        for residue in chain:
            key = residue_key(chain.name, residue)
            residue_index_by_key[key] = {
                "topology_index": topology_index,
                "chain": chain.name.strip(),
                "seqid": str(residue.seqid).strip(),
                "residue_name": residue.name.strip(),
                "residue_key": key,
            }
            topology_index += 1
    return residue_index_by_key


def _selected_binding_residues(reference_structure_path: str | Path, selected) -> list[dict[str, Any]]:
    residue_index_by_key = _residue_index_metadata_by_key(reference_structure_path)

    selected_residues: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for donor in selected.donors:
        if donor.residue_key in seen_keys:
            continue
        seen_keys.add(donor.residue_key)
        metadata = residue_index_by_key.get(donor.residue_key)
        if metadata is None:
            continue
        selected_residues.append(
            {
                **metadata,
                "summary_label": _residue_summary_label(metadata["residue_name"], metadata["seqid"]),
            }
        )
    selected_residues.sort(key=lambda item: int(item["topology_index"]))
    return selected_residues


def _selected_metal_residue(reference_structure_path: str | Path, selected) -> dict[str, Any] | None:
    residue_index_by_key = _residue_index_metadata_by_key(reference_structure_path)
    return residue_index_by_key.get(str(selected.key))


def _compress_residue_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(indices))
    ranges: list[str] = []
    start = ordered[0]
    end = ordered[0]
    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def _decomp_section_heading(line: str) -> tuple[str, str] | None:
    heading = line.strip().rstrip(":").upper()
    for key, payload in _MMPBSA_SECTION_HEADINGS.items():
        if key in heading:
            return payload
    return None


def _parse_decomp_residue_field(field: str) -> tuple[str | None, int | None]:
    match = _DECOMP_RESIDUE_RE.match(field.strip())
    if match is None:
        return None, None
    return match.group("name").strip(), int(match.group("index"))


def _parse_decomp_total_values(raw_line: str) -> tuple[str | None, int | None, float | None, float | None] | None:
    if "|" in raw_line:
        parts = [item.strip() for item in raw_line.split("|")]
        if len(parts) < 3:
            return None
        residue_name, topology_index = _parse_decomp_residue_field(parts[0])
        if topology_index is None:
            return None
        total_match = _DECOMP_VALUE_RE.search(parts[-1])
        if total_match is None:
            return None
        return (
            residue_name,
            topology_index,
            _float_or_none(total_match.group(1)),
            _float_or_none(total_match.group(2)),
        )

    if "," in raw_line:
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) < 5:
            return None
        residue_name, topology_index = _parse_decomp_residue_field(parts[0])
        if topology_index is None:
            return None
        return (
            residue_name,
            topology_index,
            _float_or_none(parts[-3]),
            _float_or_none(parts[-2]),
        )

    return None


def _input_source_label(output_dir: Path) -> str:
    if (output_dir.parent / "workflow_manifest.json").exists():
        return "workflow"
    return "raw_files"


def mmpbsa_output_dir_has_assets(output_dir: str | Path) -> bool:
    root = Path(output_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return False
    for marker in _MMPBSA_EXISTING_ASSET_MARKERS:
        if (root / marker).exists():
            return True
    return False


def next_mmpbsa_output_directory(output_dir: str | Path) -> Path:
    root = Path(output_dir).expanduser()
    parent = root.parent
    stem = root.name
    index = 1
    while True:
        candidate = parent / f"{stem}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def ensure_mmpbsa_output_dir_is_available(output_dir: str | Path) -> None:
    root = Path(output_dir).expanduser()
    if not mmpbsa_output_dir_has_assets(root):
        return
    raise FileExistsError(
        f"MM-PBSA assets already exist in {root}. Remove that directory or choose a new output_dir "
        f"such as {next_mmpbsa_output_directory(root)}."
    )


def _resolve_selected_site(config: FreeEnergyWorkflowConfig, candidates) -> Any:
    if not candidates:
        raise ValueError("No bound metal candidates were detected in the reference structure.")
    if config.metal.selected_site is not None:
        return select_site(candidates, config.metal.selected_site)
    if len(candidates) == 1:
        return candidates[0]
    available = ", ".join(str(candidate.site) for candidate in candidates)
    raise ValueError(
        "Multiple bound metal candidates were detected, but metal.selected_site was not provided. "
        f"Available sites: {available}."
    )


def _mmpbsa_binary_path() -> str | None:
    amber_env = detect_amber_environment()
    for name in ("MMPBSA.py.MPI", "MMPBSA.py"):
        status = amber_env.binaries.get(name)
        if status is not None and status.path is not None:
            return str(status.path)
    return None


def _entropy_warning_lines(mmpbsa: MMPBSAConfig) -> list[str]:
    return mmpbsa.warning_messages()


def _validate_frame_window(config: FreeEnergyWorkflowConfig) -> None:
    frame_count = count_trajectory_frames(config.complex_input.trajectory_path)
    if frame_count is None:
        return
    if config.mmpbsa.start_frame is not None and config.mmpbsa.start_frame > frame_count:
        raise ValueError(
            f"Configured MM-PBSA start_frame ({config.mmpbsa.start_frame}) exceeds the actual trajectory frame count "
            f"({frame_count}) in {config.complex_input.trajectory_path}. Regenerate the MM-PBSA assets or lower the "
            "start_frame/end_frame values."
        )
    if config.mmpbsa.end_frame is not None and config.mmpbsa.end_frame > frame_count:
        raise ValueError(
            f"Configured MM-PBSA end_frame ({config.mmpbsa.end_frame}) exceeds the actual trajectory frame count "
            f"({frame_count}) in {config.complex_input.trajectory_path}. Regenerate the MM-PBSA assets or lower the "
            "start_frame/end_frame values."
        )


def _solver_labels(config: MMPBSAConfig) -> list[str]:
    labels: list[str] = []
    if config.run_gb:
        labels.append("MM-GBSA")
    if config.run_pb:
        labels.append("MM-PBSA")
    return labels


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_delta_summary_line(line: str) -> dict[str, Any] | None:
    total_match = _DELTA_TOTAL_RE.search(line)
    if total_match is not None:
        return {
            "term_label": "DELTA TOTAL",
            "delta_g": _float_or_none(total_match.group(1)),
            "std_dev": _float_or_none(total_match.group(2)),
            "std_err": _float_or_none(total_match.group(3)),
        }
    binding_match = _DELTA_G_BINDING_RE.search(line)
    if binding_match is not None:
        return {
            "term_label": "DELTA G binding",
            "delta_g": _float_or_none(binding_match.group(1)),
            "std_dev": _float_or_none(binding_match.group(2)),
            "std_err": _float_or_none(binding_match.group(3)),
        }
    return None


def parse_mmpbsa_final_results_text(text: str) -> dict[str, Any]:
    sections: dict[str, dict[str, Any]] = {}
    current_key: str | None = None
    current_label: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = line.rstrip(":")
        heading_info = _MMPBSA_SECTION_HEADINGS.get(heading)
        if heading_info is not None:
            current_key, current_label = heading_info
            continue
        if current_key is None:
            continue
        delta_summary = _parse_delta_summary_line(line)
        if delta_summary is None:
            continue
        sections[current_key] = {
            "status": "success",
            "solver": current_key,
            "label": current_label,
            **delta_summary,
        }
    return sections


def parse_mmpbsa_final_results_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return parse_mmpbsa_final_results_text(target.read_text(encoding="utf-8"))


def build_mmpbsa_summary_payload(*, final_results_path: str | Path, requested_solvers: list[str]) -> dict[str, Any]:
    final_results = Path(final_results_path)
    parsed = parse_mmpbsa_final_results_file(final_results)
    summary = {
        "source_file": str(final_results),
        "requested_solvers": requested_solvers,
        "results": {},
    }
    for code in requested_solvers:
        label = _MMPBSA_SECTION_HEADINGS["GENERALIZED BORN"][1] if code == "gb" else _MMPBSA_SECTION_HEADINGS["POISSON BOLTZMANN"][1]
        result = dict(parsed.get(code, {}))
        if not result:
            result = {
                "status": "missing",
                "solver": code,
                "label": label,
                "term_label": None,
                "delta_g": None,
                "std_dev": None,
                "std_err": None,
            }
        result["source_file"] = str(final_results)
        summary["results"][label] = result
    return summary


def render_mmpbsa_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "MM-PBSA summary generated by SIMPLE",
        f"Source file: {summary.get('source_file', 'N/A')}",
    ]
    for label in ("MM-GBSA", "MM-PBSA"):
        result = (summary.get("results") or {}).get(label)
        if result is None:
            continue
        if result.get("status") != "success":
            lines.append(f"{label}: {result.get('status', 'missing')}")
            continue
        lines.append(
            f"{label}: {result.get('term_label', 'DELTA TOTAL')} = {result['delta_g']:.4f} kcal/mol | "
            f"Std. Dev. = {result['std_dev']:.4f} | Std. Err. = {result['std_err']:.4f}"
        )
    return "\n".join(lines) + "\n"


def parse_mmpbsa_decomp_text(
    text: str,
    *,
    selected_residues: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    selected_by_index = {
        int(item["topology_index"]): item
        for item in (selected_residues or [])
        if item.get("topology_index") is not None
    }
    sections: dict[str, dict[str, Any]] = {}
    current_solver: str | None = None
    current_label: str | None = None
    in_delta_section = False
    in_total_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = _decomp_section_heading(line)
        if heading is not None:
            current_solver, current_label = heading
            in_delta_section = False
            in_total_section = False
            continue
        upper = line.upper()
        if upper.startswith("DELTAS"):
            in_delta_section = True
            in_total_section = False
            continue
        if not in_delta_section:
            continue
        if upper == "TOTAL ENERGY DECOMPOSITION:":
            in_total_section = True
            continue
        if upper.endswith("ENERGY DECOMPOSITION:") and upper != "TOTAL ENERGY DECOMPOSITION:":
            in_total_section = False
            continue
        if current_solver is None or current_label is None or not in_total_section:
            continue
        parsed_row = _parse_decomp_total_values(raw_line)
        if parsed_row is None:
            continue
        residue_name, topology_index, delta_g, std_dev = parsed_row
        if topology_index is None:
            continue
        if selected_by_index and topology_index not in selected_by_index:
            continue
        if delta_g is None or std_dev is None:
            continue
        metadata = selected_by_index.get(topology_index, {})
        residue_label = str(metadata.get("summary_label") or _residue_summary_label(residue_name or "X", str(topology_index)))
        entry = {
            "topology_index": topology_index,
            "summary_label": residue_label,
            "residue_name": str(metadata.get("residue_name") or residue_name or ""),
            "chain": str(metadata.get("chain") or ""),
            "seqid": str(metadata.get("seqid") or ""),
            "residue_key": str(metadata.get("residue_key") or ""),
            "delta_g": delta_g,
            "std_dev": std_dev,
        }
        section = sections.setdefault(
            current_solver,
            {
                "status": "success",
                "solver": current_solver,
                "label": current_label,
                "residues": [],
            },
        )
        section["residues"].append(entry)

    for payload in sections.values():
        payload["residues"].sort(key=lambda item: int(item["topology_index"]))
    return sections


def build_mmpbsa_decomp_summary_payload(
    *,
    final_decomp_path: str | Path,
    requested_solvers: list[str],
    selected_residues: list[dict[str, Any]],
) -> dict[str, Any]:
    final_decomp = Path(final_decomp_path)
    parsed = parse_mmpbsa_decomp_text(
        final_decomp.read_text(encoding="utf-8"),
        selected_residues=selected_residues,
    )
    summary = {
        "source_file": str(final_decomp),
        "requested_solvers": requested_solvers,
        "selected_residues": selected_residues,
        "results": {},
    }
    for code in requested_solvers:
        label = _solver_label(code)
        result = dict(parsed.get(code, {}))
        if not result:
            result = {
                "status": "missing",
                "solver": code,
                "label": label,
                "residues": [],
            }
        result["source_file"] = str(final_decomp)
        summary["results"][label] = result
    return summary


def render_mmpbsa_decomp_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "MM-PBSA residue decomposition summary generated by SIMPLE",
        f"Source file: {summary.get('source_file', 'N/A')}",
    ]
    selected_residues = summary.get("selected_residues") or []
    if selected_residues:
        lines.append(
            "Selected residues: "
            + ", ".join(str(item.get("summary_label") or item.get("topology_index") or "?") for item in selected_residues)
        )
    for label in ("MM-GBSA", "MM-PBSA"):
        result = (summary.get("results") or {}).get(label)
        if result is None:
            continue
        lines.append(f"{label}:")
        if result.get("status") != "success":
            lines.append(f"  {result.get('status', 'missing')}")
            continue
        residues = result.get("residues") or []
        if not residues:
            lines.append("  no selected residue contributions were parsed")
            continue
        for residue in residues:
            delta_g = residue.get("delta_g")
            std_dev = residue.get("std_dev")
            if delta_g is None or std_dev is None:
                continue
            lines.append(f"  {residue.get('summary_label', '?')}: {delta_g:.4f} {std_dev:.4f}")
    return "\n".join(lines) + "\n"


def _solver_label(solver_code: str) -> str:
    return "MM-GBSA" if solver_code == "gb" else "MM-PBSA"


def _solver_short_label(solver_code: str) -> str:
    return "GB" if solver_code == "gb" else "PB"


def _load_requested_solvers(output_dir: Path) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    requested = payload.get("requested_solvers") or []
    return [str(item).lower() for item in requested if str(item).lower() in {"gb", "pb"}]


def load_mmpbsa_summary_from_output_dir(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    summary_path = root / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "results" in payload:
            return payload

    final_results_path = root / "output" / "FINAL_RESULTS_MMPBSA.dat"
    if not final_results_path.exists():
        raise FileNotFoundError(f"missing {final_results_path.name}")

    requested_solvers = _load_requested_solvers(root)
    if not requested_solvers:
        parsed = parse_mmpbsa_final_results_file(final_results_path)
        requested_solvers = [code for code in ("gb", "pb") if code in parsed] or ["gb", "pb"]
    return build_mmpbsa_summary_payload(
        final_results_path=final_results_path,
        requested_solvers=requested_solvers,
    )


def _load_requested_decomp_solvers(output_dir: Path) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    decomposition = payload.get("decomposition") or {}
    requested = decomposition.get("requested_solvers") or []
    return [str(item).lower() for item in requested if str(item).lower() in {"gb", "pb"}]


def load_mmpbsa_decomp_summary_from_output_dir(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    summary_path = root / "summary_decomp.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "results" in payload:
            return payload
    raise FileNotFoundError(f"missing {summary_path.name}")


def _mmpbsa_batch_ph_label(raw_spec: dict[str, Any]) -> str:
    raw_label = raw_spec.get("ph_label")
    return "PH?" if raw_label is None else str(raw_label)


def _mmpbsa_batch_display_name(case_id: str, ph_label: str) -> str:
    return case_id if not ph_label else f"{case_id} {ph_label}"


def _mmpbsa_batch_case_spec_from_output_dir(path: Path) -> dict[str, Any]:
    workflow_root = path.parent
    if _MMPBSA_PH_DIR_RE.match(workflow_root.name):
        case_root = workflow_root.parent
        case_id = case_root.name
        ph_label = workflow_root.name
    else:
        case_root = workflow_root
        case_id = workflow_root.name
        ph_label = ""
    return {
        "case_id": case_id,
        "ph_label": ph_label,
        "display_name": _mmpbsa_batch_display_name(case_id, ph_label),
        "case_root": str(case_root),
        "output_dir": str(path),
        "requested_solvers": _load_requested_solvers(path),
        "decomp_requested_solvers": _load_requested_decomp_solvers(path),
    }


def collect_mmpbsa_batch_decomp_results(case_specs: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for raw_spec in case_specs:
        case_id = str(raw_spec.get("case_id") or "Unknown")
        ph_label = _mmpbsa_batch_ph_label(raw_spec)
        display_name = str(raw_spec.get("display_name") or _mmpbsa_batch_display_name(case_id, ph_label))
        output_dir = Path(str(raw_spec.get("output_dir") or ".")).expanduser()
        requested_solvers = [str(item).lower() for item in raw_spec.get("decomp_requested_solvers") or []]
        entry: dict[str, Any] = {
            "case_id": case_id,
            "ph": ph_label,
            "display_name": display_name,
            "output_dir": str(output_dir),
            "status": "success",
            "error_message": None,
            "gb": {
                "requested": "gb" in requested_solvers if requested_solvers else False,
                "status": "not_requested",
                "residues": [],
                "source_file": "",
                "error_message": None,
            },
            "pb": {
                "requested": "pb" in requested_solvers if requested_solvers else False,
                "status": "not_requested",
                "residues": [],
                "source_file": "",
                "error_message": None,
            },
        }
        try:
            summary = load_mmpbsa_decomp_summary_from_output_dir(output_dir)
        except FileNotFoundError as exc:
            entry["status"] = "error"
            entry["error_message"] = str(exc)
            cases.append(entry)
            continue

        requested = [str(item).lower() for item in summary.get("requested_solvers") or requested_solvers]
        results = summary.get("results") or {}
        for solver_code in ("gb", "pb"):
            solver = entry[solver_code]
            solver["requested"] = solver_code in requested if requested else solver["requested"]
            if not solver["requested"]:
                solver["status"] = "not_requested"
                continue
            result_payload = results.get(_solver_label(solver_code)) or {}
            solver["status"] = str(result_payload.get("status") or "missing")
            solver["residues"] = list(result_payload.get("residues") or [])
            solver["source_file"] = str(result_payload.get("source_file") or "")
            solver["error_message"] = result_payload.get("error_message")
        cases.append(entry)
    return {
        "summary_type": "mmpbsa_batch_decomp",
        "cases": cases,
    }


def render_mmpbsa_batch_decomp_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for case in summary.get("cases") or []:
        display_name = str(case.get("display_name") or "Unknown")
        if case.get("status") == "error":
            lines.append(f"{display_name}: ERROR {case.get('error_message') or 'unknown error'}")
            continue
        emitted = False
        for solver_code in ("gb", "pb"):
            solver = case.get(solver_code) or {}
            if not solver.get("requested"):
                continue
            label = _solver_short_label(solver_code)
            status = str(solver.get("status") or "missing")
            if status != "success":
                detail = solver.get("error_message") or status
                lines.append(f"{display_name}: {label} {detail}")
                emitted = True
                continue
            residues = solver.get("residues") or []
            if not residues:
                lines.append(f"{display_name}: {label} no selected residue contributions were parsed")
                emitted = True
                continue
            residue_parts: list[str] = []
            for residue in residues:
                delta_g = residue.get("delta_g")
                std_dev = residue.get("std_dev")
                if delta_g is None or std_dev is None:
                    continue
                residue_parts.append(f"{residue.get('summary_label', '?')} {float(delta_g):.4f} {float(std_dev):.4f}")
            if residue_parts:
                lines.append(f"{display_name}: {label} " + " | ".join(residue_parts))
                emitted = True
        if not emitted:
            lines.append(f"{display_name}: no decomposition results requested")
    return "\n".join(lines) + ("\n" if lines else "")


def refresh_mmpbsa_summaries(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path.name}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {manifest_path}") from exc

    assets = manifest.get("assets") or {}
    prep_scripts = assets.get("prep_scripts") or {}
    planned_outputs = assets.get("planned_outputs") or {}
    decomposition = manifest.get("decomposition") or {}

    final_results_path = Path(str(planned_outputs.get("final_results") or root / "output" / "FINAL_RESULTS_MMPBSA.dat"))
    summary_text_path = Path(str(planned_outputs.get("summary_text") or root / "summary.txt"))
    summary_json_path = Path(str(planned_outputs.get("summary_json") or root / "summary.json"))
    decomp_summary_text_path = Path(str(planned_outputs.get("summary_decomp_text") or root / "summary_decomp.txt"))
    decomp_summary_json_path = Path(str(planned_outputs.get("summary_decomp_json") or root / "summary_decomp.json"))
    summary_helper_path = Path(str(prep_scripts.get("summary_helper_python") or root / "prep" / "07_write_summary.py"))

    requested_solvers = [str(item).lower() for item in manifest.get("requested_solvers") or [] if str(item).lower() in {"gb", "pb"}]
    if not requested_solvers and final_results_path.exists():
        parsed = parse_mmpbsa_final_results_file(final_results_path)
        requested_solvers = [code for code in ("gb", "pb") if code in parsed] or ["gb", "pb"]

    decomp_requested_solvers = [
        str(item).lower()
        for item in decomposition.get("requested_solvers") or []
        if str(item).lower() in {"gb", "pb"}
    ]
    selected_residues = [
        item
        for item in decomposition.get("selected_binding_residues") or []
        if isinstance(item, dict)
    ]
    raw_decomp_outputs = planned_outputs.get("decomp_solver_outputs") or {}
    decomp_solver_outputs = {
        str(code).lower(): {
            "final_results": str(payload.get("final_results") or ""),
            "final_decomp": str(payload.get("final_decomp") or ""),
            "status_file": str(payload.get("status_file") or ""),
        }
        for code, payload in raw_decomp_outputs.items()
        if str(code).lower() in {"gb", "pb"} and isinstance(payload, dict)
    }

    summary_helper_path.parent.mkdir(parents=True, exist_ok=True)
    summary_helper_path.write_text(
        _render_summary_helper_script(
            final_results_path=final_results_path,
            summary_text_path=summary_text_path,
            summary_json_path=summary_json_path,
            decomp_summary_text_path=decomp_summary_text_path,
            decomp_summary_json_path=decomp_summary_json_path,
            requested_solvers=requested_solvers,
            decomp_requested_solvers=decomp_requested_solvers,
            selected_residues=selected_residues,
            decomp_solver_outputs=decomp_solver_outputs,
        ),
        encoding="utf-8",
    )
    runpy.run_path(str(summary_helper_path), run_name="__main__")
    return {
        "output_dir": str(root),
        "manifest": str(manifest_path),
        "summary_helper": str(summary_helper_path),
        "summary_text": str(summary_text_path),
        "summary_json": str(summary_json_path),
        "summary_decomp_text": str(decomp_summary_text_path),
        "summary_decomp_json": str(decomp_summary_json_path),
    }


def refresh_mmpbsa_summaries_batch(
    search_root: str | Path,
    *,
    output_dir_name: str,
) -> dict[str, Any]:
    root = Path(search_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"missing directory: {root}")

    matches = sorted(
        [path for path in root.rglob(output_dir_name) if path.is_dir()],
        key=lambda item: str(item).lower(),
    )
    refreshed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for path in matches:
        try:
            refreshed.append(refresh_mmpbsa_summaries(path))
        except Exception as exc:
            failures.append(
                {
                    "output_dir": str(path),
                    "error": str(exc),
                }
            )

    case_specs = [_mmpbsa_batch_case_spec_from_output_dir(path) for path in matches]

    batch_summary = collect_mmpbsa_batch_results(case_specs)
    batch_decomp_summary = collect_mmpbsa_batch_decomp_results(case_specs)
    aggregate_paths = _next_batch_summary_paths(root)
    aggregate_paths["summary_text"].write_text(
        render_mmpbsa_batch_summary_text(batch_summary),
        encoding="utf-8",
    )
    aggregate_paths["summary_json"].write_text(
        json.dumps(batch_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    aggregate_paths["summary_decomp_text"].write_text(
        render_mmpbsa_batch_decomp_summary_text(batch_decomp_summary),
        encoding="utf-8",
    )
    aggregate_paths["summary_decomp_json"].write_text(
        json.dumps(batch_decomp_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "search_root": str(root),
        "output_dir_name": output_dir_name,
        "matched_output_dirs": [str(path) for path in matches],
        "refreshed": refreshed,
        "failed": failures,
        "root_summary_text": str(aggregate_paths["summary_text"]),
        "root_summary_json": str(aggregate_paths["summary_json"]),
        "root_summary_decomp_text": str(aggregate_paths["summary_decomp_text"]),
        "root_summary_decomp_json": str(aggregate_paths["summary_decomp_json"]),
    }


def collect_mmpbsa_batch_results(case_specs: list[dict[str, Any]]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for raw_spec in case_specs:
        case_id = str(raw_spec.get("case_id") or "Unknown")
        ph_label = _mmpbsa_batch_ph_label(raw_spec)
        display_name = str(raw_spec.get("display_name") or _mmpbsa_batch_display_name(case_id, ph_label))
        output_dir = Path(str(raw_spec.get("output_dir") or ".")).expanduser()
        requested_solvers = [str(item).lower() for item in raw_spec.get("requested_solvers") or []]
        entry: dict[str, Any] = {
            "case_id": case_id,
            "ph": ph_label,
            "display_name": display_name,
            "output_dir": str(output_dir),
            "status": "success",
            "source_file": None,
            "error_message": None,
            "gb": {
                "requested": "gb" in requested_solvers if requested_solvers else True,
                "status": "missing",
                "delta_g": None,
                "std_dev": None,
                "std_err": None,
            },
            "pb": {
                "requested": "pb" in requested_solvers if requested_solvers else True,
                "status": "missing",
                "delta_g": None,
                "std_dev": None,
                "std_err": None,
            },
        }
        try:
            summary = load_mmpbsa_summary_from_output_dir(output_dir)
        except FileNotFoundError as exc:
            entry["status"] = "error"
            entry["error_message"] = str(exc)
            cases.append(entry)
            continue

        results = summary.get("results") or {}
        requested = [str(item).lower() for item in summary.get("requested_solvers") or requested_solvers]
        entry["source_file"] = str(summary.get("source_file") or "")
        for solver_code in ("gb", "pb"):
            solver = entry[solver_code]
            solver["requested"] = solver_code in requested if requested else solver["requested"]
            result_payload = results.get(_solver_label(solver_code)) or {}
            solver["status"] = str(result_payload.get("status") or "missing")
            solver["delta_g"] = result_payload.get("delta_g")
            solver["std_dev"] = result_payload.get("std_dev")
            solver["std_err"] = result_payload.get("std_err")
        cases.append(entry)
    return {
        "summary_type": "mmpbsa_batch",
        "cases": cases,
    }


def render_mmpbsa_batch_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for case in summary.get("cases") or []:
        display_name = str(case.get("display_name") or "Unknown")
        if case.get("status") == "error":
            lines.append(f"{display_name}: ERROR {case.get('error_message') or 'unknown error'}")
            continue
        parts: list[str] = []
        for solver_code in ("gb", "pb"):
            solver = case.get(solver_code) or {}
            label = _solver_short_label(solver_code)
            if solver.get("status") == "success" and solver.get("delta_g") is not None:
                parts.append(
                    f"{label} {float(solver['delta_g']):.2f} {float(solver['std_dev']):.2f} {float(solver['std_err']):.2f}"
                )
            else:
                parts.append(f"{label} N/A")
        lines.append(f"{display_name}: {' '.join(parts)}")
    return "\n".join(lines) + ("\n" if lines else "")


def _batch_job_name(directory: Path) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", directory.name or "mmpbsa_batch")
    return sanitized[:48] or "mmpbsa_batch"


def _batch_run_suffix(index: int | None) -> str:
    return "" if index is None else f"_{index}"


def _batch_logs_dir_name(index: int | None) -> str:
    return f"{_BATCH_LOGS_DIR_NAME}{_batch_run_suffix(index)}"


def _batch_named_file(directory: Path, base_name: str, index: int | None) -> Path:
    suffix = _batch_run_suffix(index)
    base = Path(base_name)
    return directory / f"{base.stem}{suffix}{base.suffix}"


def _next_batch_summary_paths(root: Path) -> dict[str, Path]:
    index: int | None = None
    while True:
        paths = {
            "summary_text": _batch_named_file(root, "summary.txt", index),
            "summary_json": _batch_named_file(root, "summary.json", index),
            "summary_decomp_text": _batch_named_file(root, "summary_decomp.txt", index),
            "summary_decomp_json": _batch_named_file(root, "summary_decomp.json", index),
        }
        if not any(path.exists() for path in paths.values()):
            return paths
        index = 1 if index is None else index + 1


def _batch_asset_paths(root: Path, *, index: int | None) -> dict[str, Path]:
    logs_dir = root / _batch_logs_dir_name(index)
    return {
        "logs_dir": logs_dir,
        "summary_text": _batch_named_file(root, "summary.txt", index),
        "summary_json": _batch_named_file(root, "summary.json", index),
        "manifest": logs_dir / "mmpbsa_batch_manifest.json",
        "collector_helper": logs_dir / "collect_mmpbsa_batch.py",
        "collector": logs_dir / "collect_mmpbsa_batch.sbatch",
        "tahoma_collector": logs_dir / "tahoma_collect_mmpbsa_batch.sbatch",
        "submitter": logs_dir / "run_mmpbsa_batch.sbatch",
        "tahoma_submitter": logs_dir / "tahoma_mmpbsa_batch.sbatch",
    }


def _next_batch_asset_paths(root: Path) -> dict[str, Path]:
    index: int | None = None
    while True:
        paths = _batch_asset_paths(root, index=index)
        if not any(path.exists() for path in paths.values()):
            return paths
        index = 1 if index is None else index + 1


def _render_batch_collector_helper_script(*, manifest_path: Path) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import json",
            "import re",
            "from pathlib import Path",
            "",
            f"MANIFEST_PATH = Path(r'''{manifest_path}''')",
            "SECTION_HEADINGS = {",
            "    'GENERALIZED BORN': ('gb', 'MM-GBSA'),",
            "    'POISSON BOLTZMANN': ('pb', 'MM-PBSA'),",
            "}",
            "DELTA_TOTAL_RE = re.compile(r'DELTA TOTAL\\s+([-+0-9Ee.]+)\\s+([-+0-9Ee.]+)\\s+([-+0-9Ee.]+)')",
            "DELTA_RE = re.compile(r'DELTA G binding\\s*=\\s*([-+0-9Ee.]+)\\s*\\+/-\\s*([-+0-9Ee.]+)\\s*([-+0-9Ee.]+)')",
            "",
            "def parse_results(text: str):",
            "    parsed = {}",
            "    current_solver = None",
            "    current_label = None",
            "    for raw_line in text.splitlines():",
            "        line = raw_line.strip()",
            "        heading = line.rstrip(':')",
            "        if heading in SECTION_HEADINGS:",
            "            current_solver, current_label = SECTION_HEADINGS[heading]",
            "            continue",
            "        if current_solver is None:",
            "            continue",
            "        total_match = DELTA_TOTAL_RE.search(line)",
            "        if total_match is not None:",
            "            parsed[current_solver] = {",
            "                'status': 'success',",
            "                'solver': current_solver,",
            "                'label': current_label,",
            "                'delta_g': float(total_match.group(1)),",
            "                'std_dev': float(total_match.group(2)),",
            "                'std_err': float(total_match.group(3)),",
            "            }",
            "            continue",
            "        binding_match = DELTA_RE.search(line)",
            "        if binding_match is not None:",
            "            parsed[current_solver] = {",
            "                'status': 'success',",
            "                'solver': current_solver,",
            "                'label': current_label,",
            "                'delta_g': float(binding_match.group(1)),",
            "                'std_dev': float(binding_match.group(2)),",
            "                'std_err': float(binding_match.group(3)),",
            "            }",
            "    return parsed",
            "",
            "def load_requested_solvers(output_dir: Path):",
            "    manifest_path = output_dir / 'manifest.json'",
            "    if not manifest_path.exists():",
            "        return []",
            "    try:",
            "        payload = json.loads(manifest_path.read_text(encoding='utf-8'))",
            "    except json.JSONDecodeError:",
            "        return []",
            "    requested = payload.get('requested_solvers') or []",
            "    return [str(item).lower() for item in requested if str(item).lower() in {'gb', 'pb'}]",
            "",
            "def summary_from_output_dir(output_dir: Path):",
            "    summary_json = output_dir / 'summary.json'",
            "    if summary_json.exists():",
            "        payload = json.loads(summary_json.read_text(encoding='utf-8'))",
            "        if isinstance(payload, dict) and 'results' in payload:",
            "            return payload",
            "    final_results = output_dir / 'output' / 'FINAL_RESULTS_MMPBSA.dat'",
            "    if not final_results.exists():",
            "        raise FileNotFoundError(f'missing {final_results.name}')",
            "    parsed = parse_results(final_results.read_text(encoding='utf-8'))",
            "    requested = load_requested_solvers(output_dir) or [code for code in ('gb', 'pb') if code in parsed] or ['gb', 'pb']",
            "    results = {}",
            "    for solver_code in requested:",
            "        label = 'MM-GBSA' if solver_code == 'gb' else 'MM-PBSA'",
            "        entry = dict(parsed.get(solver_code) or {})",
            "        if not entry:",
            "            entry = {",
            "                'status': 'missing',",
            "                'solver': solver_code,",
            "                'label': label,",
            "                'delta_g': None,",
            "                'std_dev': None,",
            "                'std_err': None,",
            "            }",
            "        entry['source_file'] = str(final_results)",
            "        results[label] = entry",
            "    return {",
            "        'source_file': str(final_results),",
            "        'requested_solvers': requested,",
            "        'results': results,",
            "    }",
            "",
            "manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))",
            "summary = {'summary_type': 'mmpbsa_batch', 'cases': []}",
            "for case_spec in manifest.get('cases') or []:",
            "    case_id = str(case_spec.get('case_id') or 'Unknown')",
            "    raw_ph_label = case_spec.get('ph_label')",
            "    ph_label = 'PH?' if raw_ph_label is None else str(raw_ph_label)",
            "    display_name = str(case_spec.get('display_name') or (case_id if not ph_label else f'{case_id} {ph_label}'))",
            "    output_dir = Path(str(case_spec.get('output_dir') or '.'))",
            "    requested_solvers = [str(item).lower() for item in case_spec.get('requested_solvers') or []]",
            "    entry = {",
            "        'case_id': case_id,",
            "        'ph': ph_label,",
            "        'display_name': display_name,",
            "        'output_dir': str(output_dir),",
            "        'status': 'success',",
            "        'source_file': None,",
            "        'error_message': None,",
            "        'gb': {'requested': 'gb' in requested_solvers if requested_solvers else True, 'status': 'missing', 'delta_g': None, 'std_dev': None, 'std_err': None},",
            "        'pb': {'requested': 'pb' in requested_solvers if requested_solvers else True, 'status': 'missing', 'delta_g': None, 'std_dev': None, 'std_err': None},",
            "    }",
            "    try:",
            "        payload = summary_from_output_dir(output_dir)",
            "    except FileNotFoundError as exc:",
            "        entry['status'] = 'error'",
            "        entry['error_message'] = str(exc)",
            "        summary['cases'].append(entry)",
            "        continue",
            "    results = payload.get('results') or {}",
            "    requested = [str(item).lower() for item in payload.get('requested_solvers') or requested_solvers]",
            "    entry['source_file'] = str(payload.get('source_file') or '')",
            "    for solver_code in ('gb', 'pb'):",
            "        label = 'MM-GBSA' if solver_code == 'gb' else 'MM-PBSA'",
            "        solver = entry[solver_code]",
            "        solver['requested'] = solver_code in requested if requested else solver['requested']",
            "        result_payload = results.get(label) or {}",
            "        solver['status'] = str(result_payload.get('status') or 'missing')",
            "        solver['delta_g'] = result_payload.get('delta_g')",
            "        solver['std_dev'] = result_payload.get('std_dev')",
            "        solver['std_err'] = result_payload.get('std_err')",
            "    summary['cases'].append(entry)",
            "",
            "lines = []",
            "for case in summary.get('cases') or []:",
            "    display_name = str(case.get('display_name') or 'Unknown')",
            "    if case.get('status') == 'error':",
            "        lines.append(f\"{display_name}: ERROR {case.get('error_message') or 'unknown error'}\")",
            "        continue",
            "    parts = []",
            "    for solver_code, short_label in (('gb', 'GB'), ('pb', 'PB')):",
            "        solver = case.get(solver_code) or {}",
            "        if solver.get('status') == 'success' and solver.get('delta_g') is not None:",
            "            parts.append(",
            "                f\"{short_label} {float(solver['delta_g']):.2f} {float(solver['std_dev']):.2f} {float(solver['std_err']):.2f}\"",
            "            )",
            "        else:",
            "            parts.append(f'{short_label} N/A')",
            "    lines.append(f\"{display_name}: {' '.join(parts)}\")",
            "",
            "summary_text_path = Path(str((manifest.get('outputs') or {}).get('summary_text') or MANIFEST_PATH.with_name('summary.txt')))",
            "summary_json_path = Path(str((manifest.get('outputs') or {}).get('summary_json') or MANIFEST_PATH.with_name('summary.json')))",
            "summary_text_path.write_text('\\n'.join(lines) + ('\\n' if lines else ''), encoding='utf-8')",
            "summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')",
            "",
        ]
    ) + "\n"


def _render_batch_collector_slurm_script(
    *,
    helper_path: Path,
    job_name: str,
    root_dir: Path,
    logs_dir_name: str,
) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --account=[Account]",
            "#SBATCH --time=00:15:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --chdir={root_dir.resolve()}",
            f"#SBATCH --job-name=[{job_name}_collect]",
            f"#SBATCH --error={logs_dir_name}/[{job_name}_collect]-%j.err",
            f"#SBATCH --output={logs_dir_name}/[{job_name}_collect]-%j.out",
            "",
            "# Fill in the SBATCH placeholders above before submission.",
            "set -euo pipefail",
            "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"",
            "PYTHON_BIN=${PYTHON_BIN:-python3}",
            f"\"$PYTHON_BIN\" \"{helper_path.resolve()}\"",
            "",
        ]
    ) + "\n"


def _render_tahoma_batch_collector_slurm_script(
    *,
    helper_path: Path,
    job_name: str,
    root_dir: Path,
    logs_dir_name: str,
) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "",
            "#SBATCH --account emsl62113",
            "#SBATCH --time 00:15:00",
            "#SBATCH --nodes 1",
            "#SBATCH --ntasks-per-node 1",
            f"#SBATCH --chdir {root_dir.resolve()}",
            f"#SBATCH --job-name {job_name}_collect",
            "#SBATCH --exclude=t154",
            f"#SBATCH --error {logs_dir_name}/{job_name}_collect-%j.err",
            f"#SBATCH --output {logs_dir_name}/{job_name}_collect-%j.out",
            "",
            *_TAHOMA_CONDA_AMBERTOOLS_LINES,
            "",
            "set -euo pipefail",
            "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"",
            "PYTHON_BIN=${PYTHON_BIN:-python3}",
            f"\"$PYTHON_BIN\" \"{helper_path.resolve()}\"",
            "",
        ]
    ) + "\n"


def _render_batch_submitter_script(
    *,
    child_scripts: list[Path],
    collector_script: Path,
    job_name: str,
    tahoma: bool,
    root_dir: Path,
    logs_dir_name: str,
) -> str:
    header = [
        "#!/bin/bash",
        "#SBATCH --account=[Account]",
        "#SBATCH --time=00:15:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --chdir={root_dir.resolve()}",
        f"#SBATCH --job-name=[{job_name}_submit]",
        f"#SBATCH --error={logs_dir_name}/[{job_name}_submit]-%j.err",
        f"#SBATCH --output={logs_dir_name}/[{job_name}_submit]-%j.out",
        "",
        "# Fill in the SBATCH placeholders above before submission.",
        "set -euo pipefail",
    ]
    if tahoma:
        header = [
            "#!/bin/bash",
            "",
            "#SBATCH --account emsl62113",
            "#SBATCH --time 00:15:00",
            "#SBATCH --nodes 1",
            "#SBATCH --ntasks-per-node 1",
            f"#SBATCH --chdir {root_dir.resolve()}",
            f"#SBATCH --job-name {job_name}_submit",
            "#SBATCH --exclude=t154",
            f"#SBATCH --error {logs_dir_name}/{job_name}_submit-%j.err",
            f"#SBATCH --output {logs_dir_name}/{job_name}_submit-%j.out",
            "",
            *_TAHOMA_CONDA_AMBERTOOLS_LINES,
            "",
            "set -euo pipefail",
        ]
    body = [
        "SCRIPT_DIR=\"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"",
        "JOB_IDS=()",
        "CHILD_SCRIPTS=(",
    ]
    for path in child_scripts:
        body.append(f"  \"{path.resolve()}\"")
    body.extend(
        [
            ")",
            "",
            "for child in \"${CHILD_SCRIPTS[@]}\"; do",
            "  if [ ! -f \"$child\" ]; then",
            "    echo \"Skipping missing child sbatch: $child\" >&2",
            "    continue",
            "  fi",
            "  submit_output=$(sbatch \"$child\")",
            "  echo \"$submit_output\"",
            "  job_id=$(echo \"$submit_output\" | awk '{print $4}')",
            "  if [ -n \"$job_id\" ]; then",
            "    JOB_IDS+=(\"$job_id\")",
            "  fi",
            "done",
            "",
            f"COLLECTOR_SCRIPT=\"{collector_script.resolve()}\"",
            "if [ ${#JOB_IDS[@]} -gt 0 ]; then",
            "  dependency=$(IFS=:; echo \"${JOB_IDS[*]}\")",
            "  sbatch --dependency=afterany:\"$dependency\" \"$COLLECTOR_SCRIPT\"",
            "else",
            "  sbatch \"$COLLECTOR_SCRIPT\"",
            "fi",
            "",
        ]
    )
    return "\n".join(header + body) + "\n"


def write_mmpbsa_batch_submission_assets(
    *,
    batch_dir: str | Path,
    case_specs: list[dict[str, Any]],
) -> dict[str, str]:
    root = Path(batch_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resolved_paths = _next_batch_asset_paths(root)
    logs_dir = resolved_paths["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_text_path = resolved_paths["summary_text"]
    summary_json_path = resolved_paths["summary_json"]
    manifest_path = resolved_paths["manifest"]
    helper_path = resolved_paths["collector_helper"]
    collector_path = resolved_paths["collector"]
    tahoma_collector_path = resolved_paths["tahoma_collector"]
    submitter_path = resolved_paths["submitter"]
    tahoma_submitter_path = resolved_paths["tahoma_submitter"]
    job_name = _batch_job_name(Path(submitter_path.stem))
    logs_dir_name = logs_dir.name

    manifest_payload = {
        "summary_type": "mmpbsa_batch_manifest",
        "batch_dir": str(root),
        "logs_dir": str(logs_dir),
        "cases": [
            {
                "case_id": str(spec.get("case_id") or ""),
                "ph_label": str(spec.get("ph_label") or ""),
                "display_name": str(spec.get("display_name") or ""),
                "output_dir": str(spec.get("output_dir") or ""),
                "cluster_sbatch": str(spec.get("cluster_sbatch") or ""),
                "tahoma_sbatch": str(spec.get("tahoma_sbatch") or ""),
                "requested_solvers": [str(item).lower() for item in spec.get("requested_solvers") or []],
            }
            for spec in case_specs
        ],
        "outputs": {
            "summary_text": str(summary_text_path),
            "summary_json": str(summary_json_path),
        },
        "assets": {},
    }
    write_json(manifest_path, manifest_payload)
    helper_path.write_text(
        _render_batch_collector_helper_script(manifest_path=manifest_path),
        encoding="utf-8",
    )
    collector_path.write_text(
        _render_batch_collector_slurm_script(
            helper_path=helper_path,
            job_name=job_name,
            root_dir=root,
            logs_dir_name=logs_dir_name,
        ),
        encoding="utf-8",
    )
    tahoma_collector_path.write_text(
        _render_tahoma_batch_collector_slurm_script(
            helper_path=helper_path,
            job_name=job_name,
            root_dir=root,
            logs_dir_name=logs_dir_name,
        ),
        encoding="utf-8",
    )
    submitter_path.write_text(
        _render_batch_submitter_script(
            child_scripts=[Path(str(spec["cluster_sbatch"])) for spec in manifest_payload["cases"] if spec["cluster_sbatch"]],
            collector_script=collector_path,
            job_name=job_name,
            tahoma=False,
            root_dir=root,
            logs_dir_name=logs_dir_name,
        ),
        encoding="utf-8",
    )
    tahoma_submitter_path.write_text(
        _render_batch_submitter_script(
            child_scripts=[Path(str(spec["tahoma_sbatch"])) for spec in manifest_payload["cases"] if spec["tahoma_sbatch"]],
            collector_script=tahoma_collector_path,
            job_name=job_name,
            tahoma=True,
            root_dir=root,
            logs_dir_name=logs_dir_name,
        ),
        encoding="utf-8",
    )
    assets = {
        "logs_dir": str(logs_dir),
        "manifest": str(manifest_path),
        "collector_helper": str(helper_path),
        "collector": str(collector_path),
        "tahoma_collector": str(tahoma_collector_path),
        "submitter": str(submitter_path),
        "tahoma_submitter": str(tahoma_submitter_path),
        "summary_text": str(summary_text_path),
        "summary_json": str(summary_json_path),
    }
    manifest_payload["assets"] = assets
    write_json(manifest_path, manifest_payload)
    return assets


def _render_mmpbsa_input(config: MMPBSAConfig, *, decomp_print_res: str | None = None) -> str:
    lines = [
        "MM-PBSA input generated by SIMPLE",
        "&general",
        "  interval=1,",
        "  keep_files=2,",
        "  verbose=1,",
    ]
    if config.include_entropy and config.entropy_method == MMPBSAEntropyMethod.QHA:
        lines.append("  entropy=1,")
    lines.append("/")
    if config.run_gb:
        lines.extend(
            [
                "&gb",
                "  igb=2, saltcon=0.100,",
                "/",
            ]
        )
    if config.run_pb:
        lines.extend(
            [
                "&pb",
                "  istrng=0.100,",
                "  radiopt=0,",
                "/",
            ]
        )
    if config.include_entropy and config.entropy_method == MMPBSAEntropyMethod.NMODE:
        lines.extend(
            [
                "&nmode",
                "  nmstartframe=1,",
                "  nminterval=1,",
            ]
        )
        lines.extend(
            [
                "  nmode_igb=1,",
                "  nmode_istrng=0.100,",
                "/",
            ]
        )
    if config.include_decomposition and decomp_print_res:
        lines.extend(
            [
                "&decomp",
                f"  idecomp={config.decomposition_idecomp}, print_res=\"{decomp_print_res}\",",
                f"  dec_verbose={config.decomposition_verbose},",
                "/",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_complex_dry_parmed_script(*, input_prmtop: str | Path, output_prmtop: str | Path) -> str:
    return (
        f"parm {Path(input_prmtop).as_posix()}\n"
        f"strip {_SOLVENT_ION_STRIP_MASK} nobox\n"
        f"outparm {Path(output_prmtop).as_posix()}\n"
        "go\n"
    )


def _render_parmed_python_helper(
    *,
    input_prmtop: str | Path,
    strip_mask: str,
    output_prmtop: str | Path,
) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "",
            "def _import_parmed():",
            "    try:",
            "        import parmed as pmd",
            "        return pmd",
            "    except Exception as first_error:",
            "        amberhome = os.environ.get('AMBERHOME')",
            "        if amberhome:",
            "            amberhome_path = Path(amberhome)",
            "            candidates = sorted(amberhome_path.glob('lib/python*/site-packages'))",
            "            candidates.extend(sorted(amberhome_path.glob('lib/python*/dist-packages')))",
            "            for candidate in candidates:",
            "                sys.path.insert(0, str(candidate))",
            "                try:",
            "                    import parmed as pmd",
            "                    return pmd",
            "                except Exception:",
            "                    continue",
            "        raise first_error",
            "",
            "",
            "pmd = _import_parmed()",
            f"parm = pmd.load_file(r'''{Path(input_prmtop)}''')",
            f"parm.strip(r'''{strip_mask}''')",
            "parm.box = None",
            f"parm.write_parm(r'''{Path(output_prmtop)}''')",
            "",
        ]
    ) + "\n"


def _render_receptor_parmed_script(*, complex_dry_prmtop: str | Path, ligand_mask: str, output_prmtop: str | Path) -> str:
    return (
        f"parm {Path(complex_dry_prmtop).as_posix()}\n"
        f"strip {ligand_mask} nobox\n"
        f"outparm {Path(output_prmtop).as_posix()}\n"
        "go\n"
    )


def _render_ligand_parmed_script(*, complex_dry_prmtop: str | Path, ligand_mask: str, output_prmtop: str | Path) -> str:
    return (
        f"parm {Path(complex_dry_prmtop).as_posix()}\n"
        f"strip !({ligand_mask}) nobox\n"
        f"outparm {Path(output_prmtop).as_posix()}\n"
        "go\n"
    )


def _render_strip_cpptraj_script(
    *,
    input_prmtop: str | Path,
    trajectory_path: str | Path,
    output_trajectory: str | Path,
    strip_solvent: bool,
    start_frame: int | None,
    end_frame: int | None,
    frame_stride: int,
) -> str:
    frame_tokens = []
    if start_frame is not None:
        frame_tokens.append(str(start_frame))
    if end_frame is not None:
        frame_tokens.append(str(end_frame))
    if start_frame is not None or end_frame is not None or frame_stride != 1:
        while len(frame_tokens) < 2:
            frame_tokens.append("last")
        frame_tokens.append(str(frame_stride))
        frame_clause = " " + " ".join(frame_tokens)
    else:
        frame_clause = ""
    lines = [
        f"parm {Path(input_prmtop).as_posix()}",
        f"trajin {Path(trajectory_path).as_posix()}{frame_clause}",
        "autoimage",
    ]
    if strip_solvent:
        lines.append(f"strip {_SOLVENT_ION_STRIP_MASK}")
    lines.extend(
        [
            f"trajout {Path(output_trajectory).as_posix()} netcdf" + (" nobox" if strip_solvent else ""),
            "go",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_qha_cpptraj_script(*, complex_dry_prmtop: str | Path, complex_dry_traj: str | Path, output_path: str | Path) -> str:
    return (
        f"parm {Path(complex_dry_prmtop).as_posix()}\n"
        f"trajin {Path(complex_dry_traj).as_posix()}\n"
        "rms first mass\n"
        "matrix mwcovar name qha_cov out qha_covariance.dat\n"
        f"analyze matrix qha_cov thermo out {Path(output_path).as_posix()}\n"
        "go\n"
    )


def _render_summary_helper_script(
    *,
    final_results_path: str | Path,
    summary_text_path: str | Path,
    summary_json_path: str | Path,
    decomp_summary_text_path: str | Path,
    decomp_summary_json_path: str | Path,
    requested_solvers: list[str],
    decomp_requested_solvers: list[str],
    selected_residues: list[dict[str, Any]],
    decomp_solver_outputs: dict[str, dict[str, str]],
) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import json",
            "import re",
            "from pathlib import Path",
            "",
            f"FINAL_RESULTS = Path(r'''{Path(final_results_path)}''')",
            f"SUMMARY_TEXT = Path(r'''{Path(summary_text_path)}''')",
            f"SUMMARY_JSON = Path(r'''{Path(summary_json_path)}''')",
            f"DECOMP_SUMMARY_TEXT = Path(r'''{Path(decomp_summary_text_path)}''')",
            f"DECOMP_SUMMARY_JSON = Path(r'''{Path(decomp_summary_json_path)}''')",
            f"REQUESTED_SOLVERS = {requested_solvers!r}",
            f"DECOMP_REQUESTED_SOLVERS = {decomp_requested_solvers!r}",
            f"SELECTED_RESIDUES = {selected_residues!r}",
            f"DECOMP_SOLVER_OUTPUTS = {decomp_solver_outputs!r}",
            "SECTION_HEADINGS = {",
            "    'GENERALIZED BORN': ('gb', 'MM-GBSA'),",
            "    'POISSON BOLTZMANN': ('pb', 'MM-PBSA'),",
            "}",
            "DELTA_TOTAL_RE = re.compile(r'DELTA TOTAL\\s+([-+0-9Ee.]+)\\s+([-+0-9Ee.]+)\\s+([-+0-9Ee.]+)')",
            "DELTA_RE = re.compile(r'DELTA G binding\\s*=\\s*([-+0-9Ee.]+)\\s*\\+/-\\s*([-+0-9Ee.]+)\\s*([-+0-9Ee.]+)')",
            "DECOMP_VALUE_RE = re.compile(r'([-+0-9Ee.]+)\\s*\\+/-\\s*([-+0-9Ee.]+)')",
            "DECOMP_RESIDUE_RE = re.compile(r'^(?P<name>[A-Za-z0-9+\\-]+)\\s+(?P<index>\\d+)$')",
            "",
            "def parse_delta_line(line: str):",
            "    total_match = DELTA_TOTAL_RE.search(line)",
            "    if total_match is not None:",
            "        return {",
            "            'term_label': 'DELTA TOTAL',",
            "            'delta_g': float(total_match.group(1)),",
            "            'std_dev': float(total_match.group(2)),",
            "            'std_err': float(total_match.group(3)),",
            "        }",
            "    binding_match = DELTA_RE.search(line)",
            "    if binding_match is not None:",
            "        return {",
            "            'term_label': 'DELTA G binding',",
            "            'delta_g': float(binding_match.group(1)),",
            "            'std_dev': float(binding_match.group(2)),",
            "            'std_err': float(binding_match.group(3)),",
            "        }",
            "    return None",
            "",
            "def parse_results(text: str) -> dict[str, dict[str, object]]:",
            "    results: dict[str, dict[str, object]] = {}",
            "    current_solver = None",
            "    current_label = None",
            "    for raw_line in text.splitlines():",
            "        line = raw_line.strip()",
            "        heading = line.rstrip(':')",
            "        if heading in SECTION_HEADINGS:",
            "            current_solver, current_label = SECTION_HEADINGS[heading]",
            "            continue",
            "        if current_solver is None:",
            "            continue",
            "        delta = parse_delta_line(line)",
            "        if delta is None:",
            "            continue",
            "        results[current_solver] = {",
            "            'status': 'success',",
            "            'solver': current_solver,",
            "            'label': current_label,",
            "            **delta,",
            "            'source_file': str(FINAL_RESULTS),",
            "        }",
            "    return results",
            "",
            "def parse_decomp_residue(field: str):",
            "    match = DECOMP_RESIDUE_RE.match(field.strip())",
            "    if match is None:",
            "        return None, None",
            "    return match.group('name').strip(), int(match.group('index'))",
            "",
            "def parse_decomp_total_values(raw_line: str):",
            "    if '|' in raw_line:",
            "        parts = [item.strip() for item in raw_line.split('|')]",
            "        if len(parts) < 3:",
            "            return None",
            "        residue_name, topology_index = parse_decomp_residue(parts[0])",
            "        if topology_index is None:",
            "            return None",
            "        total_match = DECOMP_VALUE_RE.search(parts[-1])",
            "        if total_match is None:",
            "            return None",
            "        return residue_name, topology_index, float(total_match.group(1)), float(total_match.group(2))",
            "    if ',' in raw_line:",
            "        parts = [item.strip() for item in raw_line.split(',')]",
            "        if len(parts) < 5:",
            "            return None",
            "        residue_name, topology_index = parse_decomp_residue(parts[0])",
            "        if topology_index is None:",
            "            return None",
            "        try:",
            "            return residue_name, topology_index, float(parts[-3]), float(parts[-2])",
            "        except ValueError:",
            "            return None",
            "    return None",
            "",
            "def parse_decomp(text: str):",
            "    selected_by_index = {int(item['topology_index']): item for item in SELECTED_RESIDUES if item.get('topology_index') is not None}",
            "    results = {}",
            "    current_solver = None",
            "    current_label = None",
            "    in_delta_section = False",
            "    in_total_section = False",
            "    for raw_line in text.splitlines():",
            "        line = raw_line.strip()",
            "        if not line:",
            "            continue",
            "        heading = line.rstrip(':').upper()",
            "        for section_name, payload in SECTION_HEADINGS.items():",
            "            if section_name in heading:",
            "                current_solver, current_label = payload",
            "                in_delta_section = False",
            "                in_total_section = False",
            "                break",
            "        else:",
            "            upper = line.upper()",
            "            if upper.startswith('DELTAS'):",
            "                in_delta_section = True",
            "                in_total_section = False",
            "                continue",
            "            if not in_delta_section:",
            "                continue",
            "            if upper == 'TOTAL ENERGY DECOMPOSITION:':",
            "                in_total_section = True",
            "                continue",
            "            if upper.endswith('ENERGY DECOMPOSITION:') and upper != 'TOTAL ENERGY DECOMPOSITION:':",
            "                in_total_section = False",
            "                continue",
            "            if current_solver is None or current_label is None or not in_total_section:",
            "                continue",
            "            parsed_row = parse_decomp_total_values(raw_line)",
            "            if parsed_row is None:",
            "                continue",
            "            residue_name, topology_index, delta_g, std_dev = parsed_row",
            "            if topology_index is None or topology_index not in selected_by_index:",
            "                continue",
            "            metadata = selected_by_index[topology_index]",
            "            section = results.setdefault(",
            "                current_solver,",
            "                {'status': 'success', 'solver': current_solver, 'label': current_label, 'residues': []},",
            "            )",
            "            section['residues'].append({",
            "                'topology_index': topology_index,",
            "                'summary_label': str(metadata.get('summary_label') or f\"{residue_name}{topology_index}\"),",
            "                'residue_name': str(metadata.get('residue_name') or residue_name or ''),",
            "                'chain': str(metadata.get('chain') or ''),",
            "                'seqid': str(metadata.get('seqid') or ''),",
            "                'residue_key': str(metadata.get('residue_key') or ''),",
            "                'delta_g': delta_g,",
            "                'std_dev': std_dev,",
            "            })",
            "    for payload in results.values():",
            "        payload['residues'].sort(key=lambda item: int(item['topology_index']))",
            "    return results",
            "",
            "parsed = parse_results(FINAL_RESULTS.read_text(encoding='utf-8'))",
            "summary = {",
            "    'source_file': str(FINAL_RESULTS),",
            "    'requested_solvers': REQUESTED_SOLVERS,",
            "    'results': {},",
            "}",
            "for solver_code, label in (('gb', 'MM-GBSA'), ('pb', 'MM-PBSA')):",
            "    if solver_code not in REQUESTED_SOLVERS:",
            "        continue",
            "    entry = parsed.get(solver_code)",
            "    if entry is None:",
            "        entry = {",
            "            'status': 'missing',",
            "            'solver': solver_code,",
            "            'label': label,",
            "            'term_label': None,",
            "            'delta_g': None,",
            "            'std_dev': None,",
            "            'std_err': None,",
            "            'source_file': str(FINAL_RESULTS),",
            "        }",
            "    summary['results'][label] = entry",
            "",
            "lines = [",
            "    'MM-PBSA summary generated by SIMPLE',",
            "    f'Source file: {FINAL_RESULTS}',",
            "]",
            "for label in ('MM-GBSA', 'MM-PBSA'):",
            "    entry = summary['results'].get(label)",
            "    if entry is None:",
            "        continue",
            "    if entry['status'] != 'success':",
            "        lines.append(f\"{label}: {entry['status']}\")",
            "        continue",
            "    lines.append(",
            "        f\"{label}: {entry.get('term_label', 'DELTA TOTAL')} = {entry['delta_g']:.4f} kcal/mol | \"",
            "        f\"Std. Dev. = {entry['std_dev']:.4f} | Std. Err. = {entry['std_err']:.4f}\"",
            "    )",
            "",
            "SUMMARY_TEXT.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')",
            "SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')",
            "",
            "decomp_summary = {",
            "    'source_files': DECOMP_SOLVER_OUTPUTS,",
            "    'requested_solvers': DECOMP_REQUESTED_SOLVERS,",
            "    'selected_residues': SELECTED_RESIDUES,",
            "    'results': {},",
            "}",
            "decomp_lines = [",
            "    'MM-PBSA residue decomposition summary generated by SIMPLE',",
            "    'Source files:',",
            "]",
            "for solver_code in DECOMP_REQUESTED_SOLVERS:",
            "    spec = DECOMP_SOLVER_OUTPUTS.get(solver_code) or {}",
            "    label = 'MM-GBSA' if solver_code == 'gb' else 'MM-PBSA'",
            "    decomp_lines.append(f\"  {label}: {spec.get('final_decomp', 'N/A')}\")",
            "if SELECTED_RESIDUES:",
            "    selected_labels = ', '.join(str(item.get('summary_label') or item.get('topology_index') or '?') for item in SELECTED_RESIDUES)",
            "    decomp_lines.append(f'Selected residues: {selected_labels}')",
            "else:",
            "    decomp_lines.append('Selected residues: none')",
            "for solver_code, label in (('gb', 'MM-GBSA'), ('pb', 'MM-PBSA')):",
            "    if solver_code not in DECOMP_REQUESTED_SOLVERS:",
            "        continue",
            "    spec = DECOMP_SOLVER_OUTPUTS.get(solver_code) or {}",
            "    status_path = Path(str(spec.get('status_file') or '')) if spec.get('status_file') else None",
            "    final_decomp_path = Path(str(spec.get('final_decomp') or '')) if spec.get('final_decomp') else None",
            "    parsed_decomp = parse_decomp(final_decomp_path.read_text(encoding='utf-8')) if final_decomp_path is not None and final_decomp_path.exists() and SELECTED_RESIDUES else {}",
            "    entry = parsed_decomp.get(solver_code)",
            "    if status_path is not None and status_path.exists():",
            "        error_message = status_path.read_text(encoding='utf-8', errors='ignore').strip()",
            "        entry = {",
            "            'status': 'error',",
            "            'solver': solver_code,",
            "            'label': label,",
            "            'residues': [],",
            "            'source_file': '' if final_decomp_path is None else str(final_decomp_path),",
            "            'error_message': error_message,",
            "        }",
            "    elif entry is None:",
            "        status = 'missing' if final_decomp_path is not None else 'not_available'",
            "        entry = {",
            "            'status': status,",
            "            'solver': solver_code,",
            "            'label': label,",
            "            'residues': [],",
            "            'source_file': '' if final_decomp_path is None else str(final_decomp_path),",
            "        }",
            "    else:",
            "        entry['source_file'] = '' if final_decomp_path is None else str(final_decomp_path)",
            "    decomp_summary['results'][label] = entry",
            "    decomp_lines.append(f'{label}:')",
            "    if entry['status'] != 'success':",
            "        detail = entry.get('error_message') or entry['status']",
            "        decomp_lines.append(f\"  {detail}\")",
            "        continue",
            "    if not entry.get('residues'):",
            "        decomp_lines.append('  no selected residue contributions were parsed')",
            "        continue",
            "    for residue in entry['residues']:",
            "        decomp_lines.append(",
            "            f\"  {residue.get('summary_label', '?')}: {residue['delta_g']:.4f} {residue['std_dev']:.4f}\"",
            "        )",
            "",
            "DECOMP_SUMMARY_TEXT.write_text('\\n'.join(decomp_lines) + '\\n', encoding='utf-8')",
            "DECOMP_SUMMARY_JSON.write_text(json.dumps(decomp_summary, indent=2), encoding='utf-8')",
            "",
        ]
    ) + "\n"


def _render_mmpbsa_slurm_script(
    *,
    slurm_config: SlurmConfig,
    work_root: Path,
    prep_dir: Path,
    input_dir: Path,
    output_dir: Path,
    complex_prmtop: Path,
    trajectory_path: Path,
    mmpbsa_trajectory: Path,
    dry_complex_prmtop: Path,
    receptor_prmtop: Path,
    ligand_prmtop: Path,
    summary_helper_path: Path,
    decomp_solver_assets: dict[str, dict[str, Path]],
    tool_config: ToolConfig | None = None,
) -> str:
    ambertools_setup = ambertools_sbatch_setup(
        tool_config,
        required_binaries=("cpptraj", "MMPBSA.py"),
    )
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=[Account]",
        "#SBATCH --time=HH:MM:SS",
        "#SBATCH --nodes=[Number]",
        "#SBATCH --ntasks-per-node=[Number]",
        "#SBATCH --job-name=[mmpbsa]",
        "#SBATCH --error=[mmpbsa]-%j.err",
        "#SBATCH --output=[mmpbsa]-%j.out",
        "",
        "# Fill in the SBATCH placeholders above before submission.",
        "# This job prepares dry topologies and helper trajectories, then launches Amber MMPBSA.py.",
        "",
        "set -euo pipefail",
        "",
        *ambertools_setup,
        "",
        f"WORK_ROOT=\"{work_root.resolve()}\"",
        f"PREP_DIR=\"{prep_dir.resolve()}\"",
        f"INPUT_DIR=\"{input_dir.resolve()}\"",
        f"OUTPUT_DIR=\"{output_dir.resolve()}\"",
        "LOG_DIR=\"$OUTPUT_DIR/logs\"",
        f"COMPLEX_PRMTOP=\"{complex_prmtop.resolve()}\"",
        f"TRAJECTORY=\"{trajectory_path.resolve()}\"",
        f"MMPBSA_TRAJECTORY=\"{mmpbsa_trajectory.resolve()}\"",
        f"DRY_COMPLEX_PRMTOP=\"{dry_complex_prmtop.resolve()}\"",
        f"RECEPTOR_PRMTOP=\"{receptor_prmtop.resolve()}\"",
        f"LIGAND_PRMTOP=\"{ligand_prmtop.resolve()}\"",
        f"SUMMARY_HELPER=\"{summary_helper_path.resolve()}\"",
        "mkdir -p \"$OUTPUT_DIR\" \"$LOG_DIR\"",
        "cd \"$WORK_ROOT\"",
        "",
        "PARMED_BIN=${PARMED_BIN:-parmed}",
        "PARMED_MODE=${PARMED_MODE:-python}",
        "PARMED_PYTHON_BIN=${PARMED_PYTHON_BIN:-}",
        "PARMED_CONDA_ENV=${PARMED_CONDA_ENV:-${SIMPLE_CONDA_ENV:-simple}}",
        "PYTHON_BIN=${PYTHON_BIN:-python3}",
        "CPPTRAJ_BIN=${CPPTRAJ_BIN:-cpptraj}",
        "if command -v MMPBSA.py.MPI >/dev/null 2>&1 && [ \"${SLURM_NTASKS:-1}\" -gt 1 ]; then",
        "  MMPBSA_RUNNER=\"srun --export=ALL -n ${SLURM_NTASKS} MMPBSA.py.MPI\"",
        "else",
        "  MMPBSA_RUNNER=${MMPBSA_RUNNER:-MMPBSA.py}",
        "fi",
        "",
        "run_parmed_step () {",
        "  local script_name=\"$1\"",
        "  local helper_name=\"$2\"",
        "  local log_name=\"$3\"",
        "  if [ \"$PARMED_MODE\" = \"cli\" ] && command -v \"$PARMED_BIN\" >/dev/null 2>&1; then",
        "    \"$PARMED_BIN\" -i \"$PREP_DIR/$script_name\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  elif [ -n \"$PARMED_PYTHON_BIN\" ]; then",
        "    \"$PARMED_PYTHON_BIN\" \"$PREP_DIR/$helper_name\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  elif command -v conda >/dev/null 2>&1; then",
        "    conda run -n \"$PARMED_CONDA_ENV\" python \"$PREP_DIR/$helper_name\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  elif command -v micromamba >/dev/null 2>&1; then",
        "    micromamba run -n \"$PARMED_CONDA_ENV\" python \"$PREP_DIR/$helper_name\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  else",
        "    \"$PYTHON_BIN\" \"$PREP_DIR/$helper_name\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  fi",
        "}",
        "",
        "run_optional_decomp () {",
        "  local input_name=\"$1\"",
        "  local result_name=\"$2\"",
        "  local decomp_name=\"$3\"",
        "  local status_name=\"$4\"",
        "  local log_name=\"$5\"",
        "  rm -f \"$OUTPUT_DIR/$status_name\"",
        "  set +e",
        "  $MMPBSA_RUNNER -O -i \"$INPUT_DIR/$input_name\" -o \"$result_name\" -do \"$decomp_name\" "
        "-sp \"$COMPLEX_PRMTOP\" -cp \"$DRY_COMPLEX_PRMTOP\" -rp \"$RECEPTOR_PRMTOP\" -lp \"$LIGAND_PRMTOP\" "
        "-y \"$MMPBSA_TRAJECTORY\" > \"$LOG_DIR/$log_name\" 2>&1",
        "  local rc=$?",
        "  set -e",
        "  if [ $rc -ne 0 ]; then",
        "    echo \"Optional residue decomposition failed for $input_name (exit $rc); see $LOG_DIR/$log_name\" > \"$OUTPUT_DIR/$status_name\"",
        "    echo \"Optional residue decomposition failed for $input_name; continuing.\" >&2",
        "  fi",
        "}",
        "",
        "echo 'Preparing dry complex topology...'",
        "run_parmed_step \"01_complex_dry.parmed.in\" \"01_complex_dry.py\" \"01_complex_dry.parmed.log\"",
        "echo 'Preparing dry receptor topology...'",
        "run_parmed_step \"02_receptor.parmed.in\" \"02_receptor.py\" \"02_receptor.parmed.log\"",
        "echo 'Preparing dry ligand topology...'",
        "run_parmed_step \"03_ligand.parmed.in\" \"03_ligand.py\" \"03_ligand.parmed.log\"",
        "echo 'Writing autoimaged MM-PBSA trajectory...'",
        "$CPPTRAJ_BIN -i \"$PREP_DIR/04_complex_imaged.cpptraj.in\" > \"$LOG_DIR/04_complex_imaged.cpptraj.log\" 2>&1",
        "echo 'Writing helper dry trajectory...'",
        "$CPPTRAJ_BIN -i \"$PREP_DIR/05_complex_dry.cpptraj.in\" > \"$LOG_DIR/05_complex_dry.cpptraj.log\" 2>&1",
    ]
    if (prep_dir / "06_qha_prepare.cpptraj.in").exists():
        lines.extend(
            [
                "echo 'Preparing helper QHA trajectory summary...'",
                "$CPPTRAJ_BIN -i \"$PREP_DIR/06_qha_prepare.cpptraj.in\" > \"$LOG_DIR/06_qha_prepare.cpptraj.log\" 2>&1",
            ]
        )
    lines.extend(
        [
            "cd \"$OUTPUT_DIR\"",
            "echo 'Running MM-PBSA...' ",
            "$MMPBSA_RUNNER -O -i \"$INPUT_DIR/MMPBSA.in\" -o \"FINAL_RESULTS_MMPBSA.dat\" "
            + "-sp \"$COMPLEX_PRMTOP\" -cp \"$DRY_COMPLEX_PRMTOP\" -rp \"$RECEPTOR_PRMTOP\" -lp \"$LIGAND_PRMTOP\" "
            "-y \"$MMPBSA_TRAJECTORY\" > \"$LOG_DIR/mmpbsa_progress.log\" 2>&1",
        ]
    )
    for solver_code in ("gb", "pb"):
        assets = decomp_solver_assets.get(solver_code)
        if not assets:
            continue
        solver_label = "MM-GBSA" if solver_code == "gb" else "MM-PBSA"
        lines.extend(
            [
                f"echo 'Running optional {solver_label} residue decomposition...'",
                f"run_optional_decomp \"{assets['input'].name}\" \"{assets['results'].name}\" \"{assets['decomp'].name}\" \"{assets['status'].name}\" \"mmpbsa_decomp_{solver_code}.log\"",
            ]
        )
    lines.extend(
        [
            "echo 'Writing MM-PBSA summary...' ",
            "$PYTHON_BIN \"$SUMMARY_HELPER\" > \"$LOG_DIR/07_write_summary.log\" 2>&1",
            "",
        ]
    )
    if slurm_config.profile == SlurmProfile.GPU:
        lines.insert(4, "#SBATCH --gres=gpu:[Number]")
    return "\n".join(lines) + "\n"


def _render_tahoma_mmpbsa_script(
    *,
    work_root: Path,
    prep_dir: Path,
    input_dir: Path,
    output_dir: Path,
    complex_prmtop: Path,
    trajectory_path: Path,
    mmpbsa_trajectory: Path,
    dry_complex_prmtop: Path,
    receptor_prmtop: Path,
    ligand_prmtop: Path,
    summary_helper_path: Path,
    job_name: str,
    decomp_solver_assets: dict[str, dict[str, Path]],
) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "",
            "#SBATCH --account emsl62113                   # charged account",
            "#SBATCH --time  48:00:00                      # 30 minute time limit",
            "#SBATCH --nodes 4                             # 2 nodes",
            "#SBATCH --ntasks-per-node 32                  # 16 processes on each per node",
            "#SBATCH --exclude=t154",
            f"#SBATCH --job-name {job_name}                       # job name in queue (``squeue``)",
            f"#SBATCH --error {job_name}-%j.err            # stderr file with job_name-job_id.err",
            f"#SBATCH --output {job_name}-%j.out           # stdout file",
            "",
            "",
            *_TAHOMA_CONDA_AMBERTOOLS_LINES,
            "",
            "set -euo pipefail",
            "",
            f"WORK_ROOT=\"{work_root.resolve()}\"",
            f"PREP_DIR=\"{prep_dir.resolve()}\"",
            f"INPUT_DIR=\"{input_dir.resolve()}\"",
            f"OUTPUT_DIR=\"{output_dir.resolve()}\"",
            "LOG_DIR=\"$OUTPUT_DIR/logs\"",
            f"COMPLEX_PRMTOP=\"{complex_prmtop.resolve()}\"",
            f"TRAJECTORY=\"{trajectory_path.resolve()}\"",
            f"MMPBSA_TRAJECTORY=\"{mmpbsa_trajectory.resolve()}\"",
            f"DRY_COMPLEX_PRMTOP=\"{dry_complex_prmtop.resolve()}\"",
            f"RECEPTOR_PRMTOP=\"{receptor_prmtop.resolve()}\"",
            f"LIGAND_PRMTOP=\"{ligand_prmtop.resolve()}\"",
            f"SUMMARY_HELPER=\"{summary_helper_path.resolve()}\"",
            "mkdir -p \"$OUTPUT_DIR\" \"$LOG_DIR\"",
            "cd \"$WORK_ROOT\"",
            "",
            "PYTHON_BIN=${PYTHON_BIN:-python}",
            "CPPTRAJ_BIN=${CPPTRAJ_BIN:-cpptraj}",
            "for amber_tool in \"$PYTHON_BIN\" \"$CPPTRAJ_BIN\"; do",
            "  if ! command -v \"$amber_tool\" >/dev/null 2>&1; then",
            "    echo \"Required AmberTools command is not available in conda env $SIMPLE_CONDA_ENV: $amber_tool\" >&2",
            "    exit 1",
            "  fi",
            "done",
            "SIMPLE_FORCE_OPENMPI_TCP=${SIMPLE_FORCE_OPENMPI_TCP:-1}",
            "MPI_MCA_ARGS=${MPI_MCA_ARGS:-}",
            "if [ \"$SIMPLE_FORCE_OPENMPI_TCP\" = \"1\" ]; then",
            "  export OMPI_MCA_pml=ob1",
            "  export OMPI_MCA_btl=self,tcp",
            "  MPI_MCA_ARGS=${MPI_MCA_ARGS:-\"--mca pml ob1 --mca btl self,tcp\"}",
            "fi",
            "\"$PYTHON_BIN\" - <<'PY' > \"$LOG_DIR/00_conda_ambertools_check.log\" 2>&1",
            "import sys",
            "import parmed",
            "print(f'Python: {sys.executable}')",
            "print('parmed import OK')",
            "PY",
            "MMPBSA_MPI_AVAILABLE=0",
            "MMPBSA_MPI_PACKAGE_AVAILABLE=0",
            "if \"$PYTHON_BIN\" - <<'PY' >> \"$LOG_DIR/00_conda_ambertools_check.log\" 2>&1; then",
            "import importlib.util",
            "raise SystemExit(0 if importlib.util.find_spec('mpi4py') else 1)",
            "PY",
            "  MMPBSA_MPI_PACKAGE_AVAILABLE=1",
            "else",
            "  echo \"mpi4py Python package is not installed in conda env $SIMPLE_CONDA_ENV; MPI MM-PBSA will not be used.\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "fi",
            "if [ \"$MMPBSA_MPI_PACKAGE_AVAILABLE\" -eq 1 ]; then",
            "  echo \"MPI transport settings: OMPI_MCA_pml=${OMPI_MCA_pml:-unset} OMPI_MCA_btl=${OMPI_MCA_btl:-unset} MPI_MCA_ARGS=${MPI_MCA_ARGS:-none}\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "  if \"$PYTHON_BIN\" - <<'PY' >> \"$LOG_DIR/00_conda_ambertools_check.log\" 2>&1; then",
            "from mpi4py import MPI",
            "print('mpi4py MPI runtime import OK')",
            "PY",
            "    MMPBSA_MPI_AVAILABLE=1",
            "  else",
            "    echo \"mpi4py is installed, but the MPI runtime could not initialize in conda env $SIMPLE_CONDA_ENV; serial MM-PBSA will be used.\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "    echo \"For OpenMPI UCX/OpenFabrics errors, try exporting OMPI_MCA_pml=ob1 and OMPI_MCA_btl=self,tcp, or rebuild mpi4py against the MPI stack recommended on this cluster.\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "  fi",
            "fi",
            "MPI_LAUNCHER=${MPI_LAUNCHER:-}",
            "if [ -z \"$MPI_LAUNCHER\" ]; then",
            "  if command -v mpirun >/dev/null 2>&1; then",
            "    MPI_LAUNCHER=mpirun",
            "  elif command -v mpiexec >/dev/null 2>&1; then",
            "    MPI_LAUNCHER=mpiexec",
            "  fi",
            "fi",
            "if [ -z \"${MMPBSA_RUNNER:-}\" ]; then",
            "  if command -v MMPBSA.py.MPI >/dev/null 2>&1 && [ \"${SLURM_NTASKS:-1}\" -gt 1 ] && [ \"$MMPBSA_MPI_AVAILABLE\" -eq 1 ]; then",
            "    if [ -n \"$MPI_LAUNCHER\" ]; then",
            "      MMPBSA_RUNNER=\"$MPI_LAUNCHER $MPI_MCA_ARGS -np ${SLURM_NTASKS} MMPBSA.py.MPI\"",
            "    else",
            "      echo \"No mpirun/mpiexec was found in conda env $SIMPLE_CONDA_ENV; using serial MMPBSA.py.\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "    fi",
            "  fi",
            "  if [ -z \"${MMPBSA_RUNNER:-}\" ] && command -v MMPBSA.py >/dev/null 2>&1; then",
            "    if [ \"${SLURM_NTASKS:-1}\" -gt 1 ]; then",
            "      echo \"Using serial MMPBSA.py because MPI launch is unavailable; install mpi4py and mpirun/mpiexec for MMPBSA.py.MPI.\" >> \"$LOG_DIR/00_conda_ambertools_check.log\"",
            "    fi",
            "    MMPBSA_RUNNER=\"MMPBSA.py\"",
            "  fi",
            "  if [ -z \"${MMPBSA_RUNNER:-}\" ]; then",
            "    echo \"Neither usable MMPBSA.py.MPI nor MMPBSA.py is available in conda env $SIMPLE_CONDA_ENV. For MPI, install mpi4py with: conda install -c conda-forge mpi4py\" >&2",
            "    exit 1",
            "  fi",
            "fi",
            "",
            "run_parmed_step () {",
            "  local script_name=\"$1\"",
            "  local helper_name=\"$2\"",
            "  local log_name=\"$3\"",
            "  \"$PYTHON_BIN\" \"$PREP_DIR/$helper_name\" > \"$LOG_DIR/$log_name\" 2>&1",
            "}",
            "",
            "run_cpptraj_step () {",
            "  local input_name=\"$1\"",
            "  local log_name=\"$2\"",
            "  \"$CPPTRAJ_BIN\" -i \"$PREP_DIR/$input_name\" > \"$LOG_DIR/$log_name\" 2>&1",
            "}",
            "",
            "run_optional_decomp () {",
            "  local input_name=\"$1\"",
            "  local result_name=\"$2\"",
            "  local decomp_name=\"$3\"",
            "  local status_name=\"$4\"",
            "  local log_name=\"$5\"",
            "  rm -f \"$OUTPUT_DIR/$status_name\"",
            "  set +e",
            "  $MMPBSA_RUNNER -O -i \"$INPUT_DIR/$input_name\" -o \"$result_name\" -do \"$decomp_name\" "
            "-sp \"$COMPLEX_PRMTOP\" -cp \"$DRY_COMPLEX_PRMTOP\" -rp \"$RECEPTOR_PRMTOP\" -lp \"$LIGAND_PRMTOP\" "
            "-y \"$MMPBSA_TRAJECTORY\" > \"$LOG_DIR/$log_name\" 2>&1",
            "  local rc=$?",
            "  set -e",
            "  if [ $rc -ne 0 ]; then",
            "    echo \"Optional residue decomposition failed for $input_name (exit $rc); see $LOG_DIR/$log_name\" > \"$OUTPUT_DIR/$status_name\"",
            "    echo \"Optional residue decomposition failed for $input_name; continuing.\" >&2",
            "  fi",
            "}",
            "",
            "echo 'Preparing dry complex topology...'",
            "run_parmed_step \"01_complex_dry.parmed.in\" \"01_complex_dry.py\" \"01_complex_dry.parmed.log\"",
            "echo 'Preparing dry receptor topology...'",
            "run_parmed_step \"02_receptor.parmed.in\" \"02_receptor.py\" \"02_receptor.parmed.log\"",
            "echo 'Preparing dry ligand topology...'",
            "run_parmed_step \"03_ligand.parmed.in\" \"03_ligand.py\" \"03_ligand.parmed.log\"",
            "echo 'Writing autoimaged MM-PBSA trajectory...'",
            "run_cpptraj_step \"04_complex_imaged.cpptraj.in\" \"04_complex_imaged.cpptraj.log\"",
            "echo 'Writing helper dry trajectory...'",
            "run_cpptraj_step \"05_complex_dry.cpptraj.in\" \"05_complex_dry.cpptraj.log\"",
            "if [ -f \"$PREP_DIR/06_qha_prepare.cpptraj.in\" ]; then",
            "  echo 'Preparing helper QHA trajectory summary...'",
            "  run_cpptraj_step \"06_qha_prepare.cpptraj.in\" \"06_qha_prepare.cpptraj.log\"",
            "fi",
            "cd \"$OUTPUT_DIR\"",
            "echo 'Running MM-PBSA...'",
            "$MMPBSA_RUNNER -O -i \"$INPUT_DIR/MMPBSA.in\" -o \"FINAL_RESULTS_MMPBSA.dat\" "
            "-sp \"$COMPLEX_PRMTOP\" -cp \"$DRY_COMPLEX_PRMTOP\" -rp \"$RECEPTOR_PRMTOP\" -lp \"$LIGAND_PRMTOP\" "
            "-y \"$MMPBSA_TRAJECTORY\" > \"$LOG_DIR/mmpbsa_progress.log\" 2>&1",
        ]
        + [
            item
            for solver_code, solver_label in (("gb", "MM-GBSA"), ("pb", "MM-PBSA"))
            for assets in ([decomp_solver_assets.get(solver_code)] if decomp_solver_assets.get(solver_code) else [])
            for item in (
                f"echo 'Running optional {solver_label} residue decomposition...'",
                f"run_optional_decomp \"{assets['input'].name}\" \"{assets['results'].name}\" \"{assets['decomp'].name}\" \"{assets['status'].name}\" \"mmpbsa_decomp_{solver_code}.log\"",
            )
        ]
        + [
            "echo 'Writing MM-PBSA summary...'",
            "$PYTHON_BIN \"$SUMMARY_HELPER\" > \"$LOG_DIR/07_write_summary.log\" 2>&1",
            "",
        ]
    ) + "\n"


def _write_mmpbsa_assets(
    *,
    config: FreeEnergyWorkflowConfig,
    output_dir: Path,
    selected: Any | None,
    formal_charge: int | None,
) -> dict[str, Any]:
    input_dir = output_dir / "inputs"
    prep_dir = output_dir / "prep"
    slurm_dir = output_dir
    runtime_output_dir = output_dir / "output"
    runtime_logs_dir = runtime_output_dir / "logs"
    for path in (input_dir, prep_dir, slurm_dir, runtime_output_dir, runtime_logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    if config.mmpbsa.ligand_selection_mode == MMPBSALigandSelectionMode.RESIDUE_NAME:
        ligand_plan = _build_residue_name_ligand_plan(config)
    else:
        if selected is None or formal_charge is None:
            raise ValueError("A selected metal site is required for metal-site MM-PBSA mode.")
        ligand_plan = _build_metal_site_ligand_plan(
            config=config,
            selected=selected,
            formal_charge=formal_charge,
        )
    ligand_mask = ligand_plan.ligand_mask
    binding_residues = ligand_plan.binding_residues
    decomp_print_res = ligand_plan.decomp_print_res
    decomp_requested_solvers = config.mmpbsa.decomposition_requested_solvers()
    decomp_enabled = bool(binding_residues) and bool(decomp_requested_solvers)
    main_input_config = config.mmpbsa.model_copy(update={"include_decomposition": False})
    mmpbsa_input = input_dir / "MMPBSA.in"
    mmpbsa_input.write_text(
        _render_mmpbsa_input(
            main_input_config,
        ),
        encoding="utf-8",
    )

    dry_complex_prmtop = prep_dir / "complex_dry.prmtop"
    receptor_prmtop = prep_dir / "receptor.prmtop"
    ligand_prmtop = prep_dir / "ligand.prmtop"
    imaged_complex_trajectory = prep_dir / "complex_imaged.nc"
    dry_complex_trajectory = prep_dir / "complex_dry.nc"
    qha_output = runtime_output_dir / "qha_entropy_helper.dat"
    final_results_path = runtime_output_dir / "FINAL_RESULTS_MMPBSA.dat"
    summary_text_path = output_dir / "summary.txt"
    summary_json_path = output_dir / "summary.json"
    decomp_summary_text_path = output_dir / "summary_decomp.txt"
    decomp_summary_json_path = output_dir / "summary_decomp.json"
    summary_helper_path = prep_dir / "07_write_summary.py"
    decomp_solver_assets: dict[str, dict[str, Path]] = {}
    if decomp_enabled:
        for solver_code in decomp_requested_solvers:
            solver_input = input_dir / f"MMPBSA_decomp_{solver_code}.in"
            solver_results = runtime_output_dir / f"FINAL_RESULTS_DECOMP_{solver_code.upper()}.dat"
            solver_decomp = runtime_output_dir / f"FINAL_DECOMP_{solver_code.upper()}.dat"
            solver_status = runtime_output_dir / f"FINAL_DECOMP_{solver_code.upper()}.status"
            solver_config = config.mmpbsa.model_copy(
                update={
                    "run_gb": solver_code == "gb",
                    "run_pb": solver_code == "pb",
                    "include_entropy": False,
                    "include_decomposition": True,
                }
            )
            solver_input.write_text(
                _render_mmpbsa_input(solver_config, decomp_print_res=decomp_print_res),
                encoding="utf-8",
            )
            decomp_solver_assets[solver_code] = {
                "input": solver_input,
                "results": solver_results,
                "decomp": solver_decomp,
                "status": solver_status,
            }

    (prep_dir / "01_complex_dry.parmed.in").write_text(
        _render_complex_dry_parmed_script(
            input_prmtop=config.complex_input.prmtop_path,
            output_prmtop=dry_complex_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "01_complex_dry.py").write_text(
        _render_parmed_python_helper(
            input_prmtop=config.complex_input.prmtop_path,
            strip_mask=_SOLVENT_ION_STRIP_MASK,
            output_prmtop=dry_complex_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "02_receptor.parmed.in").write_text(
        _render_receptor_parmed_script(
            complex_dry_prmtop=dry_complex_prmtop,
            ligand_mask=ligand_plan.receptor_parmed_strip_mask,
            output_prmtop=receptor_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "02_receptor.py").write_text(
        _render_parmed_python_helper(
            input_prmtop=dry_complex_prmtop,
            strip_mask=ligand_plan.receptor_parmed_strip_mask,
            output_prmtop=receptor_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "03_ligand.parmed.in").write_text(
        _render_ligand_parmed_script(
            complex_dry_prmtop=dry_complex_prmtop,
            ligand_mask=ligand_mask,
            output_prmtop=ligand_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "03_ligand.py").write_text(
        _render_parmed_python_helper(
            input_prmtop=dry_complex_prmtop,
            strip_mask=f"!({ligand_mask})",
            output_prmtop=ligand_prmtop,
        ),
        encoding="utf-8",
    )
    (prep_dir / "04_complex_imaged.cpptraj.in").write_text(
        _render_strip_cpptraj_script(
            input_prmtop=config.complex_input.prmtop_path,
            trajectory_path=config.complex_input.trajectory_path,
            output_trajectory=imaged_complex_trajectory,
            strip_solvent=False,
            start_frame=config.mmpbsa.start_frame,
            end_frame=config.mmpbsa.end_frame,
            frame_stride=config.mmpbsa.frame_stride,
        ),
        encoding="utf-8",
    )
    (prep_dir / "05_complex_dry.cpptraj.in").write_text(
        _render_strip_cpptraj_script(
            input_prmtop=config.complex_input.prmtop_path,
            trajectory_path=imaged_complex_trajectory,
            output_trajectory=dry_complex_trajectory,
            strip_solvent=True,
            start_frame=None,
            end_frame=None,
            frame_stride=1,
        ),
        encoding="utf-8",
    )
    qha_script_path: Path | None = None
    if config.mmpbsa.include_entropy and config.mmpbsa.entropy_method == MMPBSAEntropyMethod.QHA:
        qha_script_path = prep_dir / "06_qha_prepare.cpptraj.in"
        qha_script_path.write_text(
            _render_qha_cpptraj_script(
                complex_dry_prmtop=dry_complex_prmtop,
                complex_dry_traj=dry_complex_trajectory,
                output_path=qha_output,
            ),
            encoding="utf-8",
        )
    summary_helper_path.write_text(
        _render_summary_helper_script(
            final_results_path=final_results_path,
            summary_text_path=summary_text_path,
            summary_json_path=summary_json_path,
            decomp_summary_text_path=decomp_summary_text_path,
            decomp_summary_json_path=decomp_summary_json_path,
            requested_solvers=config.mmpbsa.requested_solvers(),
            decomp_requested_solvers=decomp_requested_solvers,
            selected_residues=binding_residues,
            decomp_solver_outputs={
                code: {
                    "input": str(payload["input"]),
                    "final_results": str(payload["results"]),
                    "final_decomp": str(payload["decomp"]),
                    "status_file": str(payload["status"]),
                }
                for code, payload in decomp_solver_assets.items()
            },
        ),
        encoding="utf-8",
    )

    slurm_path = slurm_dir / f"run_mmpbsa_{config.slurm.profile.value}.sbatch"
    slurm_path.write_text(
        _render_mmpbsa_slurm_script(
            slurm_config=config.slurm,
            work_root=output_dir,
            prep_dir=prep_dir,
            input_dir=input_dir,
            output_dir=runtime_output_dir,
            complex_prmtop=Path(config.complex_input.prmtop_path),
            trajectory_path=Path(config.complex_input.trajectory_path),
            mmpbsa_trajectory=imaged_complex_trajectory,
            dry_complex_prmtop=dry_complex_prmtop,
            receptor_prmtop=receptor_prmtop,
            ligand_prmtop=ligand_prmtop,
            summary_helper_path=summary_helper_path,
            decomp_solver_assets=decomp_solver_assets,
        ),
        encoding="utf-8",
    )
    tahoma_path = slurm_dir / "tahoma_mmpbsa.sbatch"
    tahoma_path.write_text(
        _render_tahoma_mmpbsa_script(
            work_root=output_dir,
            prep_dir=prep_dir,
            input_dir=input_dir,
            output_dir=runtime_output_dir,
            complex_prmtop=Path(config.complex_input.prmtop_path),
            trajectory_path=Path(config.complex_input.trajectory_path),
            mmpbsa_trajectory=imaged_complex_trajectory,
            dry_complex_prmtop=dry_complex_prmtop,
            receptor_prmtop=receptor_prmtop,
            ligand_prmtop=ligand_prmtop,
            summary_helper_path=summary_helper_path,
            job_name=config.slurm.job_name or "TEST",
            decomp_solver_assets=decomp_solver_assets,
        ),
        encoding="utf-8",
    )
    (slurm_dir / "submit_mmpbsa_tahoma.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\ncd -- \"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"\nsbatch tahoma_mmpbsa.sbatch\n",
        encoding="utf-8",
    )

    inherited_settings = parse_cntrl_settings(config.complex_input.production_mdin_path)
    warnings = _entropy_warning_lines(config.mmpbsa)
    if warnings:
        for warning in warnings:
            print_notice("MM-PBSA Warning", warning, border_style="yellow")
    if config.mmpbsa.include_decomposition and not decomp_enabled:
        if not binding_residues:
            if config.mmpbsa.ligand_selection_mode == MMPBSALigandSelectionMode.RESIDUE_NAME:
                detail = "No ligand residues were selected for residue decomposition."
            else:
                detail = "No coordinating amino-acid residues were identified for the selected metal site."
        elif not decomp_requested_solvers:
            detail = "Residue decomposition is enabled, but no compatible decomposition solver was selected."
        else:
            detail = "Residue decomposition assets were not enabled."
        print_notice(
            "MM-PBSA Decomposition",
            f"{detail} Decomposition assets were not generated.",
            border_style="yellow",
        )

    manifest = {
        "free_energy_method": "mmpbsa",
        "output_dir": str(output_dir),
        "input_source": _input_source_label(output_dir),
        "complex_input": config.complex_input.model_dump(mode="json"),
        "inherited_md_settings": inherited_settings.to_dict(),
        "selected_metal": ligand_plan.selected_metal_label,
        "selected_site": ligand_plan.selected_site_payload,
        "selected_formal_charge": ligand_plan.selected_formal_charge,
        "ligand_selection": {
            "mode": ligand_plan.selection_mode,
            "ligand_residue_names": list(config.mmpbsa.ligand_residue_names),
            "selected_ligand_residues": ligand_plan.selected_ligand_residues,
            "receptor_selection_mode": config.mmpbsa.receptor_selection_mode.value,
            "receptor_residue_names": list(config.mmpbsa.receptor_residue_names),
            "selected_receptor_residues": ligand_plan.selected_receptor_residues,
            "receptor_policy": ligand_plan.receptor_policy,
            "ligand_mask": ligand_mask,
            "receptor_mask": ligand_plan.receptor_mask,
        },
        "ligand_mask": ligand_mask,
        "solute_strip_mask": _SOLVENT_ION_STRIP_MASK,
        "entropy": {
            "enabled": config.mmpbsa.include_entropy,
            "method": config.mmpbsa.entropy_method.value,
            "warnings": warnings,
        },
        "decomposition": {
            "enabled": decomp_enabled,
            "requested_solvers": decomp_requested_solvers,
            "idecomp": config.mmpbsa.decomposition_idecomp if decomp_enabled else None,
            "verbose": config.mmpbsa.decomposition_verbose if decomp_enabled else None,
            "print_res": decomp_print_res if decomp_enabled else None,
            "selected_binding_residues": binding_residues,
        },
        "requested_solvers": config.mmpbsa.requested_solvers(),
        "combined_input": config.mmpbsa.run_gb and config.mmpbsa.run_pb,
        "trajectory_window": {
            "start_frame": config.mmpbsa.start_frame,
            "end_frame": config.mmpbsa.end_frame,
            "frame_stride": config.mmpbsa.frame_stride,
        },
        "assets": {
            "mmpbsa_input": str(mmpbsa_input),
            "prep_scripts": {
                "complex_dry_parmed": str(prep_dir / "01_complex_dry.parmed.in"),
                "complex_dry_python": str(prep_dir / "01_complex_dry.py"),
                "receptor_parmed": str(prep_dir / "02_receptor.parmed.in"),
                "receptor_python": str(prep_dir / "02_receptor.py"),
                "ligand_parmed": str(prep_dir / "03_ligand.parmed.in"),
                "ligand_python": str(prep_dir / "03_ligand.py"),
                "complex_imaged_cpptraj": str(prep_dir / "04_complex_imaged.cpptraj.in"),
                "complex_dry_cpptraj": str(prep_dir / "05_complex_dry.cpptraj.in"),
                "qha_prepare_cpptraj": None if qha_script_path is None else str(qha_script_path),
                "summary_helper_python": str(summary_helper_path),
                "decomp_inputs": {
                    code: str(payload["input"])
                    for code, payload in decomp_solver_assets.items()
                },
            },
            "planned_outputs": {
                "complex_imaged_trajectory": str(imaged_complex_trajectory),
                "complex_dry_prmtop": str(dry_complex_prmtop),
                "receptor_prmtop": str(receptor_prmtop),
                "ligand_prmtop": str(ligand_prmtop),
                "complex_dry_trajectory": str(dry_complex_trajectory),
                "final_results": str(final_results_path),
                "decomp_solver_outputs": {
                    code: {
                        "final_results": str(payload["results"]),
                        "final_decomp": str(payload["decomp"]),
                        "status_file": str(payload["status"]),
                    }
                    for code, payload in decomp_solver_assets.items()
                },
                "summary_text": str(summary_text_path),
                "summary_json": str(summary_json_path),
                "summary_decomp_text": str(decomp_summary_text_path),
                "summary_decomp_json": str(decomp_summary_json_path),
            },
            "slurm": str(slurm_path),
            "tahoma": str(tahoma_path),
            "submit_tahoma": str(slurm_dir / "submit_mmpbsa_tahoma.sh"),
            "runtime_output_dir": str(runtime_output_dir),
            "runtime_logs_dir": str(runtime_logs_dir),
        },
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def run_mmpbsa_workflow(*, config: FreeEnergyWorkflowConfig, dry_run: bool = False) -> dict[str, Any]:
    output_dir = config.output_path()
    ensure_mmpbsa_output_dir_is_available(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_frame_window(config)
    if not dry_run and _mmpbsa_binary_path() is None:
        raise RuntimeError(
            "Amber MMPBSA.py was not found on PATH or under AMBERHOME/bin. "
            "Use --dry-run to generate assets without validating the execution host."
        )
    selected = None
    formal_charge = None
    if config.mmpbsa.ligand_selection_mode == MMPBSALigandSelectionMode.METAL_SITE:
        candidates = detect_bound_metal_sites(
            config.complex_input.reference_structure_path,
            config.complex_input.prmtop_path,
        )
        selected = _resolve_selected_site(config, candidates)
        formal_charge = config.metal.formal_charge or default_formal_charge(selected.element)
    return _write_mmpbsa_assets(
        config=config,
        output_dir=output_dir,
        selected=selected,
        formal_charge=formal_charge,
    )


def print_mmpbsa_summary(result: dict[str, Any]) -> None:
    console.print(f"MM-PBSA assets generated in {result.get('output_dir', 'N/A')}")
    ligand_selection = result.get("ligand_selection") or {}
    if ligand_selection.get("mode") == MMPBSALigandSelectionMode.RESIDUE_NAME.value:
        names = ", ".join(ligand_selection.get("ligand_residue_names") or [])
        console.print(f"Selected ligand residues: {names or 'N/A'}")
        console.print(f"Receptor policy: {ligand_selection.get('receptor_policy', 'N/A')}")
        receptor_names = ", ".join(ligand_selection.get("receptor_residue_names") or [])
        if receptor_names:
            console.print(f"Selected receptor residues: {receptor_names}")
    else:
        console.print(f"Selected metal: {result.get('selected_metal', 'N/A')}")
    console.print(f"Ligand mask: {result.get('ligand_mask', 'N/A')}")
    requested_solvers = result.get("requested_solvers") or []
    console.print(f"Requested solvers: {', '.join(solver.upper() for solver in requested_solvers) or 'N/A'}")
    entropy = result.get("entropy") or {}
    console.print(
        f"Entropy: {'off' if not entropy.get('enabled') else entropy.get('method', 'N/A')}"
    )
    assets = result.get("assets") or {}
    console.print(f"MMPBSA input: {assets.get('mmpbsa_input', 'N/A')}")
    console.print(f"Slurm script: {assets.get('slurm', 'N/A')}")
    planned_outputs = assets.get("planned_outputs") or {}
    console.print(f"Summary JSON: {planned_outputs.get('summary_json', 'N/A')}")
    console.print(f"Decomp Summary JSON: {planned_outputs.get('summary_decomp_json', 'N/A')}")
