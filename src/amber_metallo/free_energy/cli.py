from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from amber_metallo.cli import WizardChoice, _display_choice_table, _prompt_choice, _prompt_execution_profile, _print_step_header
from amber_metallo.config import SlurmProfile
from amber_metallo.environment import detect_amber_environment
from amber_metallo.free_energy.config import (
    FreeEnergyConfig,
    FreeEnergyMethod,
    FreeEnergyWorkflowConfig,
    MMPBSAConfig,
    MMPBSAEntropyMethod,
    MMPBSALigandSelectionMode,
    MMPBSAReceptorSelectionMode,
    save_config,
)
from amber_metallo.free_energy.mmpbsa import (
    mmpbsa_output_dir_has_assets,
    next_mmpbsa_output_directory,
    summarize_mmpbsa_prmtop_residues,
    write_mmpbsa_batch_submission_assets,
)
from amber_metallo.free_energy.trajectory import count_trajectory_frames
from amber_metallo.reporting import console, print_notice, write_json
from amber_metallo.ti.analysis import (
    assess_site_stability,
    default_formal_charge,
    detect_bound_metal_sites,
    parse_cntrl_settings,
    run_last_snapshot_extraction,
    select_site,
)
from amber_metallo.ti.cli import (
    _default_output_directory,
    _display_site_table,
    _infer_charge_from_selected_site,
    _input_selection_includes_unbound_metal_sites,
    _input_selection_uses_in_place_ti,
    _inspect_main_workflow_directory,
    _production_status_text,
    _prompt_existing_path,
    _prompt_complex_input_selection,
    _prompt_snapshot_mode,
    _prompt_ti_decoupling_mode,
    _prompt_ti_implementation_mode,
    _resolve_water_reference_settings,
    _resolve_water_reference_source_choice,
)
from amber_metallo.ti.config import (
    ComplexInputConfig,
    MetalSelectionConfig,
    SnapshotConfig,
    TIDecouplingMode,
    TIMetalSelectionMode,
    TIImplementationMode,
    TIProtocolConfig,
    WaterReferenceConfig,
)


_PH_DIR_RE = re.compile(r"^PH", re.IGNORECASE)
_MMPBSA_OUTPUT_DIR_RE = re.compile(r"^MM[-_]PBSA(?:(?:-|_)(?P<index>\d+))?$", re.IGNORECASE)
_BATCH_DIFFUSION_CUTOFF_ANGSTROM = 2.5
_BATCH_RETAINED_DONOR_CUTOFF_ANGSTROM = 3.5
_GENERAL_TOPOLOGY_EXTS = {".prmtop", ".parm7", ".top"}
_GENERAL_TRAJECTORY_EXTS = {".nc", ".mdcrd", ".crd", ".dcd", ".xtc", ".trr"}
_GENERAL_EXTENSIONLESS_TRAJECTORY_NAMES = {"mdcrd"}
_GENERAL_TRAJECTORY_TYPE_ORDER = (".nc", ".mdcrd", "mdcrd", ".crd", ".dcd", ".xtc", ".trr")
_GENERAL_TRAJECTORY_REQUIREMENT_TEXT = "*.nc, *.mdcrd, mdcrd, *.crd, *.dcd, *.xtc, or *.trr"
_GENERAL_REFERENCE_EXTS = {".pdb"}
_GENERAL_MDIN_EXTS = {".in"}
_GENERAL_SCAN_SKIP_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
}
_GENERAL_MMPBSA_REQUIRED_FILES_MESSAGE = (
    "This folder must contain at least: an AMBER topology file (*.prmtop, *.parm7, or *.top) "
    f"and a trajectory file ({_GENERAL_TRAJECTORY_REQUIREMENT_TEXT}). Optional but helpful: "
    "a production mdin file (*.in) and a reference PDB (*.pdb)."
)


def _path_type_key(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix:
        return suffix
    name = path.name.lower()
    if name in _GENERAL_EXTENSIONLESS_TRAJECTORY_NAMES:
        return name
    return ""


def _is_general_trajectory_file(path: Path) -> bool:
    type_key = _path_type_key(path)
    return type_key in _GENERAL_TRAJECTORY_EXTS or type_key in _GENERAL_EXTENSIONLESS_TRAJECTORY_NAMES


def _looks_like_main_workflow_root(root: Path) -> bool:
    return (
        (root / "workflow_manifest.json").exists()
        or (root / "02_system").exists()
        or (root / "03_md").exists()
    )


def _format_batch_case_label(case_id: str, ph_label: str, *, separator: str) -> str:
    return case_id if not ph_label else f"{case_id}{separator}{ph_label}"


@dataclass(slots=True)
class MMPBSABatchDiscovery:
    case_id: str
    ph_label: str
    workflow_root: Path
    workflow_discovery: Any | None
    selectable: bool
    workflow_note: str
    existing_status: str
    existing_output_dir: Path | None = None
    matching_output_dirs: list[Path] = field(default_factory=list)
    metal_candidates: list[Any] = field(default_factory=list)
    site_assessments: dict[int, Any] = field(default_factory=dict)
    md_status: str | None = None
    metal_status: str | None = None
    metal_note: str | None = None

    @property
    def display_key(self) -> str:
        return _format_batch_case_label(self.case_id, self.ph_label, separator="/")

    @property
    def summary_label(self) -> str:
        return _format_batch_case_label(self.case_id, self.ph_label, separator=" ")

    def existing_output_text(self, *, batch_root: Path | None = None) -> str:
        if self.existing_output_dir is None:
            return "-"
        path_text = _path_relative_to_batch_root(self.existing_output_dir, batch_root=batch_root)
        if len(self.matching_output_dirs) <= 1:
            return path_text
        return f"{path_text} (+{len(self.matching_output_dirs) - 1} more)"

    @property
    def md_status_text(self) -> str:
        return self.md_status or "-"

    @property
    def metal_status_text(self) -> str:
        return self.metal_status or "-"


def _quick_workflow_readiness(root: Path) -> tuple[bool, str]:
    if not root.is_dir():
        return False, "This workflow folder does not exist."
    if not _looks_like_main_workflow_root(root):
        return False, "This folder does not look like a main.py workflow directory."
    prmtop_path = root / "02_system" / "system.prmtop"
    if not prmtop_path.exists():
        return False, "Missing 02_system/system.prmtop."
    reference_candidates = [
        root / "02_system" / "system.pdb",
        root / "01_prepare" / "cleaned_input.pdb",
    ]
    if not any(candidate.exists() for candidate in reference_candidates):
        return False, "Missing 02_system/system.pdb (or 01_prepare/cleaned_input.pdb)."
    outputs_dir = root / "03_md" / "outputs"
    if not outputs_dir.exists():
        return False, "Missing 03_md/outputs."
    if not any(_is_general_trajectory_file(path) for path in outputs_dir.iterdir() if path.is_file()):
        return False, f"No usable trajectory ({_GENERAL_TRAJECTORY_REQUIREMENT_TEXT}) was found under 03_md/outputs."
    return True, "Ready"


def _batch_discovery_case_root(discovery: MMPBSABatchDiscovery) -> Path:
    return discovery.workflow_root.parent if discovery.ph_label else discovery.workflow_root


def _ensure_batch_workflow_discovery(discovery: MMPBSABatchDiscovery) -> Any:
    if discovery.workflow_discovery is not None:
        return discovery.workflow_discovery
    workflow_discovery = _inspect_main_workflow_directory(discovery.workflow_root)
    if workflow_discovery is None:
        raise ValueError(f"{discovery.display_key} is not a valid main.py workflow directory.")
    discovery.workflow_discovery = workflow_discovery
    discovery.selectable = workflow_discovery.selectable
    discovery.workflow_note = workflow_discovery.readiness_note
    return workflow_discovery


@dataclass(slots=True)
class MMPBSABatchCasePlan:
    discovery: MMPBSABatchDiscovery
    output_dir: Path
    cleanup_output_dir: Path | None = None


@dataclass(slots=True)
class MMPBSABatchPlan:
    batch_root: Path
    selection_mode: str
    conflict_policy: str | None
    cases: list[MMPBSABatchCasePlan] = field(default_factory=list)


@dataclass(slots=True)
class TIBatchCasePlan:
    site: int
    element: str
    atom_index: int
    output_dir: Path


@dataclass(slots=True)
class TIBatchPlan:
    batch_root: Path
    selection_mode: str
    cases: list[TIBatchCasePlan] = field(default_factory=list)


@dataclass(slots=True)
class FreeEnergyWizardBuildResult:
    configs: list[FreeEnergyWorkflowConfig]
    config_suffixes: list[tuple[str, ...]] = field(default_factory=list)
    saved_config_paths: list[Path] = field(default_factory=list)
    mmpbsa_batch_plan: MMPBSABatchPlan | None = None
    ti_batch_plan: TIBatchPlan | None = None

    @property
    def is_batch(self) -> bool:
        return len(self.configs) > 1


@dataclass(slots=True)
class GeneralMMPBSAInputDiscovery:
    root: Path
    prmtop_path: Path
    trajectory_path: Path
    reference_structure_path: Path | None
    production_mdin_path: Path | None
    existing_status: str
    existing_output_dir: Path | None = None
    matching_output_dirs: list[Path] = field(default_factory=list)
    trajectory_candidates: list[Path] = field(default_factory=list)
    production_mdin_candidates: list[Path] = field(default_factory=list)
    output_log_candidates: list[Path] = field(default_factory=list)

    @property
    def output_root(self) -> Path:
        return self.root

    @property
    def display_name(self) -> str:
        return self.root.name or str(self.root)

    def existing_output_text(self, *, search_root: Path | None = None) -> str:
        if self.existing_output_dir is None:
            return "-"
        return _path_relative_to_batch_root(self.existing_output_dir, batch_root=search_root)

    def complex_input(self) -> ComplexInputConfig:
        reference_path = self.reference_structure_path or self.prmtop_path
        return ComplexInputConfig(
            prmtop_path=str(self.prmtop_path),
            trajectory_path=str(self.trajectory_path),
            reference_structure_path=str(reference_path),
            production_mdin_path=None if self.production_mdin_path is None else str(self.production_mdin_path),
        )


def _ti_site_output_label(candidate) -> str:
    return f"site_{candidate.site:03d}_{candidate.element.lower()}_{candidate.atom_index}"


def _expand_ti_one_by_one_config(config: FreeEnergyWorkflowConfig) -> FreeEnergyWizardBuildResult:
    if config.metal.selection_mode != TIMetalSelectionMode.ONE_BY_ONE or len(config.metal.selected_sites) <= 1:
        return FreeEnergyWizardBuildResult(configs=[config], config_suffixes=[()])
    candidates = detect_bound_metal_sites(
        config.complex_input.reference_structure_path,
        config.complex_input.prmtop_path,
        donor_cutoff_angstrom=config.snapshot.donor_cutoff_angstrom,
        include_unbound_metals=bool(config.water_reference.bound_in_place or not config.water_reference.enabled),
    )
    candidate_by_site = {candidate.site: candidate for candidate in candidates}
    batch_root = Path(config.output_dir).expanduser().resolve()
    child_configs: list[FreeEnergyWorkflowConfig] = []
    suffixes: list[tuple[str, ...]] = []
    case_plans: list[TIBatchCasePlan] = []
    first_site = config.metal.selected_sites[0]
    first_candidate = candidate_by_site[first_site]
    first_charge = config.metal.formal_charges_by_site.get(first_site) or config.metal.formal_charge or default_formal_charge(first_candidate.element)
    for site in config.metal.selected_sites:
        candidate = candidate_by_site[site]
        formal_charge = config.metal.formal_charges_by_site.get(site) or config.metal.formal_charge or default_formal_charge(candidate.element)
        label = _ti_site_output_label(candidate)
        output_dir = batch_root / label
        water_reference = config.water_reference.model_copy(deep=True)
        if water_reference.reuse_from_library and (
            candidate.element != first_candidate.element or int(formal_charge) != int(first_charge)
        ):
            water_reference.reuse_from_library = False
            water_reference.library_key = None
        child_configs.append(
            config.model_copy(
                deep=True,
                update={
                    "metal": MetalSelectionConfig(
                        selection_mode=TIMetalSelectionMode.SINGLE,
                        selected_site=site,
                        selected_sites=[site],
                        formal_charge=formal_charge,
                        formal_charges_by_site={site: formal_charge},
                    ),
                    "water_reference": water_reference,
                    "slurm": config.slurm.model_copy(update={"job_name": label}),
                    "output_dir": str(output_dir),
                },
            )
        )
        suffixes.append((label,))
        case_plans.append(TIBatchCasePlan(site=site, element=candidate.element, atom_index=candidate.atom_index, output_dir=output_dir))
    return FreeEnergyWizardBuildResult(
        configs=child_configs,
        config_suffixes=suffixes,
        ti_batch_plan=TIBatchPlan(
            batch_root=batch_root,
            selection_mode=TIMetalSelectionMode.ONE_BY_ONE.value,
            cases=case_plans,
        ),
    )


def _free_energy_method_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            FreeEnergyMethod.TI.value,
            "TI",
            "Use the existing TI workflow with charge-off and VDW-off windows.",
        ),
        WizardChoice(
            FreeEnergyMethod.MMPBSA.value,
            "MM-PBSA",
            "Generate Amber MM-GBSA/MM-PBSA assets for a fast comparison path rooted at the selected bound trajectory.",
        ),
    ]


def _prompt_free_energy_method() -> FreeEnergyMethod:
    choices = _free_energy_method_choices()
    _display_choice_table("Free-Energy Method", choices)
    return FreeEnergyMethod(_prompt_choice("Choose the free-energy method", choices, default_key=FreeEnergyMethod.TI.value))


def _default_mmpbsa_output_directory(reference_structure_path: str, *, workflow_root: Path | None = None) -> Path:
    if workflow_root is not None:
        return workflow_root / "MM-PBSA"
    stem = Path(reference_structure_path).expanduser().stem
    return Path.cwd() / f"{stem}_MM-PBSA"


def _prompt_entropy_method() -> tuple[bool, MMPBSAEntropyMethod]:
    include_entropy = typer.confirm("Include an entropy correction?", default=False)
    if not include_entropy:
        return False, MMPBSAEntropyMethod.QHA
    choices = [
        WizardChoice(
            MMPBSAEntropyMethod.QHA.value,
            "QHA",
            "Recommended. Lower cost than nmode and suitable for a trajectory-based comparison.",
        ),
        WizardChoice(
            MMPBSAEntropyMethod.NMODE.value,
            "nmode",
            "Advanced option only. Very expensive and can fail with segmentation faults on larger systems.",
        ),
    ]
    _display_choice_table("MM-PBSA Entropy Method", choices)
    selection = MMPBSAEntropyMethod(
        _prompt_choice(
            "Choose the entropy method",
            choices,
            default_key=MMPBSAEntropyMethod.QHA.value,
        )
    )
    if selection == MMPBSAEntropyMethod.NMODE:
        print_notice(
            "nmode Warning",
            "nmode entropy is very expensive and can fail with segmentation faults on larger systems. "
            "Use it only when you explicitly want that cost profile.",
            border_style="bold yellow",
        )
    return True, selection


def _estimate_mmpbsa_saved_frames(
    *,
    trajectory_path: str | None,
    production_mdin_path: str | None,
    workflow_root: Path | None,
) -> tuple[int | None, str]:
    actual_frames = count_trajectory_frames(trajectory_path)
    if actual_frames is not None:
        return actual_frames, f"The selected trajectory file contains {actual_frames} saved frames."

    settings = parse_cntrl_settings(production_mdin_path)
    ntwx = max(1, int(settings.ntwx or 1))
    discovery = _inspect_main_workflow_directory(workflow_root) if workflow_root is not None else None
    selected_stage = discovery.selected_stage if discovery is not None else None
    current_steps = selected_stage.last_nstep if selected_stage is not None else None
    target_steps = selected_stage.target_nstlim if selected_stage is not None else None

    if current_steps is not None and current_steps > 0:
        frame_count = max(1, current_steps // ntwx)
        target_frames = max(1, target_steps // ntwx) if target_steps is not None and target_steps > 0 else None
        if target_frames is not None and target_frames != frame_count:
            return frame_count, (
                f"The current trajectory contains about {frame_count} saved frames so far "
                f"(target: about {target_frames} frames)."
            )
        return frame_count, f"The current trajectory contains about {frame_count} saved frames."

    if target_steps is not None and target_steps > 0:
        frame_count = max(1, target_steps // ntwx)
        return frame_count, f"The selected mdin target corresponds to about {frame_count} saved frames."

    return None, "The wizard could not estimate the saved-frame count from the selected trajectory."


def _prompt_frame_stride(window_frames: int) -> int:
    stride_10_frames = max(1, math.ceil(window_frames / 10))
    choices = [
        WizardChoice("1", "Stride 1", f"Use all about {window_frames} saved frames in this window."),
        WizardChoice("10", "Stride 10", f"Use about {stride_10_frames} saved frames in this window."),
        WizardChoice("custom", "Custom stride", "Enter another integer stride manually."),
    ]
    _display_choice_table("MM-PBSA Frame Stride", choices)
    selection = _prompt_choice("Choose the frame stride", choices, default_key="1")
    if selection != "custom":
        return int(selection)
    return typer.prompt("Custom frame stride", default=1, type=int)


def _prompt_mmpbsa_frame_window(
    *,
    trajectory_path: str | None,
    production_mdin_path: str | None,
    workflow_root: Path | None,
) -> tuple[int | None, int | None, int]:
    estimated_frames, estimate_note = _estimate_mmpbsa_saved_frames(
        trajectory_path=trajectory_path,
        production_mdin_path=production_mdin_path,
        workflow_root=workflow_root,
    )
    console.print(f"[dim]{estimate_note}[/dim]")
    if estimated_frames is None:
        stride = typer.prompt("Frame stride", default=1, type=int)
        return None, None, stride

    last_10_count = max(1, math.ceil(estimated_frames * 0.10))
    last_10_start = max(1, estimated_frames - last_10_count + 1)
    choices = [
        WizardChoice(
            "full",
            "Whole trajectory",
            f"Analyze frames 1-{estimated_frames} from the currently available trajectory.",
        ),
        WizardChoice(
            "last10",
            "Last 10%",
            f"Analyze frames {last_10_start}-{estimated_frames} ({last_10_count} frames).",
        ),
    ]
    _display_choice_table("MM-PBSA Frame Window", choices)
    selection = _prompt_choice("Choose the frame window", choices, default_key="last10")
    if selection == "full":
        start_frame = 1
        end_frame = estimated_frames
        window_frames = estimated_frames
    else:
        start_frame = last_10_start
        end_frame = estimated_frames
        window_frames = last_10_count

    stride = _prompt_frame_stride(window_frames)
    approx_analyzed = max(1, math.ceil(window_frames / stride))
    console.print(
        "[dim]"
        f"MM-PBSA will analyze frames {start_frame}-{end_frame} with stride {stride} "
        f"(about {approx_analyzed} saved frames)."
        "[/dim]"
    )
    return start_frame, end_frame, stride


def _resolve_mmpbsa_output_directory(output_dir: Path) -> Path:
    if not mmpbsa_output_dir_has_assets(output_dir):
        return output_dir
    print_notice(
        "Existing MM-PBSA Directory",
        "A completed or partial MM-PBSA setup already exists in this directory. "
        "ParmEd will stop when the prep files already exist.",
        border_style="yellow",
    )
    choices = [
        WizardChoice(
            "replace",
            "Delete and reuse",
            "Remove the existing MM-PBSA directory and regenerate the prep/output files in the same location.",
        ),
        WizardChoice(
            "newdir",
            "Append numeric suffix",
            f"Keep the existing directory and continue in a new directory with a numeric suffix, such as {next_mmpbsa_output_directory(output_dir).name}.",
        ),
    ]
    _display_choice_table("MM-PBSA Output Directory Conflict", choices)
    selection = _prompt_choice(
        "Choose how to handle the existing MM-PBSA directory",
        choices,
        default_key="newdir",
    )
    if selection == "replace":
        shutil.rmtree(output_dir)
        console.print(f"[dim]Removed the existing MM-PBSA directory: {output_dir}[/dim]")
        return output_dir
    resolved = next_mmpbsa_output_directory(output_dir)
    console.print(f"[dim]Continuing with a new MM-PBSA directory: {resolved}[/dim]")
    return resolved


def _mmpbsa_output_sort_key(path: Path) -> tuple[int, int, str]:
    match = _MMPBSA_OUTPUT_DIR_RE.match(path.name)
    index_text = None if match is None else match.group("index")
    index = 0 if index_text is None else int(index_text)
    canonical = 0 if path.name == "MM-PBSA" else 1
    return (index, canonical, path.name.lower())


def _discover_existing_mmpbsa_outputs(workflow_root: Path) -> tuple[str, Path | None, list[Path]]:
    matches = sorted(
        [
            child
            for child in workflow_root.iterdir()
            if child.is_dir() and _MMPBSA_OUTPUT_DIR_RE.match(child.name)
        ],
        key=_mmpbsa_output_sort_key,
    )
    complete = [path for path in matches if (path / "manifest.json").exists()]
    partial = [path for path in matches if path not in complete and mmpbsa_output_dir_has_assets(path)]
    if complete:
        return "complete", complete[-1], matches
    if partial:
        return "partial", partial[-1], matches
    return "missing", None, []


def _iter_batch_case_roots(search_root: Path) -> list[Path]:
    roots: list[Path] = []
    direct_ph_children = [child for child in search_root.iterdir() if child.is_dir() and _PH_DIR_RE.match(child.name)]
    if direct_ph_children or _looks_like_main_workflow_root(search_root):
        roots.append(search_root)
    roots.extend(
        child
        for child in sorted(search_root.iterdir(), key=lambda item: item.name.lower())
        if child.is_dir()
        and (
            _looks_like_main_workflow_root(child)
            or any(grandchild.is_dir() and _PH_DIR_RE.match(grandchild.name) for grandchild in child.iterdir())
        )
    )
    seen: set[Path] = set()
    ordered_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_roots.append(root)
    return ordered_roots


def _discover_mmpbsa_batch_cases(search_root: Path) -> list[MMPBSABatchDiscovery]:
    discoveries: list[MMPBSABatchDiscovery] = []
    for case_root in _iter_batch_case_roots(search_root):
        if _looks_like_main_workflow_root(case_root):
            selectable, workflow_note = _quick_workflow_readiness(case_root)
            existing_status, existing_output_dir, matching_output_dirs = _discover_existing_mmpbsa_outputs(case_root)
            discoveries.append(
                MMPBSABatchDiscovery(
                    case_id=case_root.name,
                    ph_label="",
                    workflow_root=case_root.resolve(),
                    workflow_discovery=None,
                    selectable=selectable,
                    workflow_note=workflow_note,
                    existing_status=existing_status,
                    existing_output_dir=existing_output_dir,
                    matching_output_dirs=matching_output_dirs,
                )
            )
        for ph_dir in sorted(
            [child for child in case_root.iterdir() if child.is_dir() and _PH_DIR_RE.match(child.name)],
            key=lambda item: item.name.lower(),
        ):
            selectable, workflow_note = _quick_workflow_readiness(ph_dir)
            existing_status, existing_output_dir, matching_output_dirs = _discover_existing_mmpbsa_outputs(ph_dir)
            discoveries.append(
                MMPBSABatchDiscovery(
                    case_id=case_root.name,
                    ph_label=ph_dir.name,
                    workflow_root=ph_dir.resolve(),
                    workflow_discovery=None,
                    selectable=selectable,
                    workflow_note=workflow_note,
                    existing_status=existing_status,
                    existing_output_dir=existing_output_dir,
                    matching_output_dirs=matching_output_dirs,
                )
            )
    return discoveries


def _is_general_scan_skipped_dir(path: Path) -> bool:
    name = path.name
    if name in _GENERAL_SCAN_SKIP_NAMES:
        return True
    if _MMPBSA_OUTPUT_DIR_RE.match(name):
        return True
    return name.upper().startswith("LOGS_MMPBSA")


def _iter_general_scan_files(search_root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_general_scan_skipped_dir(current / name)
        ]
        for filename in filenames:
            files.append(current / filename)
    return files


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _preferred_file(paths: list[Path], *, keywords: tuple[str, ...], exts: tuple[str, ...] = ()) -> Path | None:
    if not paths:
        return None

    def score(path: Path) -> tuple[int, int, int, int, str]:
        text = "/".join(part.lower() for part in path.parts)
        keyword_score = next((index for index, keyword in enumerate(keywords) if keyword in text), len(keywords))
        type_key = _path_type_key(path)
        ext_score = next((index for index, ext in enumerate(exts) if type_key == ext), len(exts))
        multi_penalty = 1 if "multi" in path.stem.lower() else 0
        return (keyword_score, multi_penalty, ext_score, len(path.parts), path.name.lower())

    return sorted(paths, key=score)[0]


def _general_bundle_root_from_topology(topology: Path, *, search_root: Path) -> Path:
    resolved_search_root = search_root.resolve()
    resolved_topology = topology.resolve()
    for parent in [resolved_topology.parent, *resolved_topology.parents]:
        if parent.name.lower() == "02_system":
            return parent.parent
        if parent == resolved_search_root:
            break
    return resolved_topology.parent


def _candidate_sort_key(path: Path) -> tuple[int, int, str]:
    text = "/".join(part.lower() for part in path.parts)
    stage_order = (
        "prod",
        "production",
        "equil",
        "eq",
        "npt",
        "nvt",
        "heat",
        "min",
    )
    stage_score = next((index for index, token in enumerate(stage_order) if token in text), len(stage_order))
    ext_order = (*_GENERAL_TRAJECTORY_TYPE_ORDER, ".out")
    type_key = _path_type_key(path)
    ext_score = next((index for index, ext in enumerate(ext_order) if type_key == ext), len(ext_order))
    return (stage_score, ext_score, path.name.lower())


def _matching_mdin_for_trajectory(trajectory: Path, mdins: list[Path]) -> Path | None:
    stem = trajectory.stem.lower()
    same_stem = [path for path in mdins if path.stem.lower() == stem]
    if same_stem:
        return sorted(same_stem, key=lambda item: item.name.lower())[0]
    return _preferred_file(mdins, keywords=("prod", "production", "md", "eq"))


def _matching_output_log_for_trajectory(trajectory: Path, output_logs: list[Path]) -> Path | None:
    stem = trajectory.stem.lower()
    same_stem = [path for path in output_logs if path.stem.lower() == stem]
    if same_stem:
        return sorted(same_stem, key=lambda item: item.name.lower())[0]
    return None


def _discover_general_mmpbsa_inputs(search_root: str | Path) -> list[GeneralMMPBSAInputDiscovery]:
    root = Path(search_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []
    files = _iter_general_scan_files(root)
    topologies = sorted(
        [path for path in files if path.suffix.lower() in _GENERAL_TOPOLOGY_EXTS],
        key=lambda item: (len(item.parts), item.name.lower()),
    )
    trajectories = [path for path in files if _is_general_trajectory_file(path)]
    references = [path for path in files if path.suffix.lower() in _GENERAL_REFERENCE_EXTS]
    mdins = [path for path in files if path.suffix.lower() in _GENERAL_MDIN_EXTS]
    output_logs = [path for path in files if path.suffix.lower() == ".out"]
    discoveries: list[GeneralMMPBSAInputDiscovery] = []
    seen: set[tuple[Path, Path]] = set()
    for topology in topologies:
        candidate_root = _general_bundle_root_from_topology(topology, search_root=root)
        nearby_trajectories = sorted(
            [path for path in trajectories if _is_relative_to(path, candidate_root)],
            key=_candidate_sort_key,
        )
        if not nearby_trajectories:
            continue
        trajectory = _preferred_file(
            nearby_trajectories,
            keywords=("prod", "production", "md", "eq"),
            exts=_GENERAL_TRAJECTORY_TYPE_ORDER,
        )
        if trajectory is None:
            continue
        nearby_references = [path for path in references if _is_relative_to(path, candidate_root)]
        nearby_mdins = [path for path in mdins if _is_relative_to(path, candidate_root)]
        nearby_output_logs = sorted(
            [path for path in output_logs if _is_relative_to(path, candidate_root)],
            key=_candidate_sort_key,
        )
        reference = _preferred_file(nearby_references, keywords=("system", "complex", "reference", "prod"))
        mdin = _matching_mdin_for_trajectory(trajectory, nearby_mdins)
        key = (candidate_root.resolve(), topology.resolve())
        if key in seen:
            continue
        seen.add(key)
        existing_status, existing_output_dir, matching_output_dirs = _discover_existing_mmpbsa_outputs(candidate_root)
        discoveries.append(
            GeneralMMPBSAInputDiscovery(
                root=candidate_root,
                prmtop_path=topology,
                trajectory_path=trajectory,
                reference_structure_path=reference,
                production_mdin_path=mdin,
                existing_status=existing_status,
                existing_output_dir=existing_output_dir,
                matching_output_dirs=matching_output_dirs,
                trajectory_candidates=nearby_trajectories,
                production_mdin_candidates=nearby_mdins,
                output_log_candidates=nearby_output_logs,
            )
        )
    return discoveries[:50]


def _display_general_mmpbsa_inputs(search_root: Path, discoveries: list[GeneralMMPBSAInputDiscovery]) -> None:
    table = Table(title="Detected AMBER Simulation Data", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Folder", style="bold white", overflow="fold")
    table.add_column("Topology", style="cyan", overflow="fold")
    table.add_column("Trajectory", style="cyan", overflow="fold")
    table.add_column("PDB", style="white", overflow="fold")
    table.add_column("mdin", style="white", overflow="fold")
    table.add_column("MM-PBSA", style="cyan")
    table.add_row("0", "Manual folder/raw files", "-", "-", "-", "-", "Manual")
    for index, discovery in enumerate(discoveries, start=1):
        table.add_row(
            str(index),
            _path_relative_to_batch_root(discovery.root, batch_root=search_root),
            _path_relative_to_batch_root(discovery.prmtop_path, batch_root=search_root),
            _path_relative_to_batch_root(discovery.trajectory_path, batch_root=search_root),
            "-" if discovery.reference_structure_path is None else _path_relative_to_batch_root(discovery.reference_structure_path, batch_root=search_root),
            "-" if discovery.production_mdin_path is None else _path_relative_to_batch_root(discovery.production_mdin_path, batch_root=search_root),
            discovery.existing_status.capitalize(),
        )
    console.print(table)
    console.print(f"[dim]Scanned {search_root.resolve()} recursively for general AMBER topology/trajectory inputs.[/dim]")


def _prompt_general_mmpbsa_input_selection(
    search_root: Path,
    discoveries: list[GeneralMMPBSAInputDiscovery],
) -> GeneralMMPBSAInputDiscovery | None:
    _display_general_mmpbsa_inputs(search_root, discoveries)
    while True:
        raw = typer.prompt("Choose a simulation data number (0 = enter paths manually)", default="1").strip()
        if raw == "0":
            return None
        try:
            index = int(raw)
        except ValueError:
            console.print("[bold red]Please enter 0 or one of the listed simulation data numbers.[/bold red]")
            continue
        if 1 <= index <= len(discoveries):
            return discoveries[index - 1]
        console.print("[bold red]Please choose a number from the table.[/bold red]")


def _with_selected_general_trajectory(
    discovery: GeneralMMPBSAInputDiscovery,
    trajectory_path: Path,
) -> GeneralMMPBSAInputDiscovery:
    mdin = _matching_mdin_for_trajectory(trajectory_path, discovery.production_mdin_candidates)
    return GeneralMMPBSAInputDiscovery(
        root=discovery.root,
        prmtop_path=discovery.prmtop_path,
        trajectory_path=trajectory_path,
        reference_structure_path=discovery.reference_structure_path,
        production_mdin_path=mdin,
        existing_status=discovery.existing_status,
        existing_output_dir=discovery.existing_output_dir,
        matching_output_dirs=discovery.matching_output_dirs,
        trajectory_candidates=discovery.trajectory_candidates,
        production_mdin_candidates=discovery.production_mdin_candidates,
        output_log_candidates=discovery.output_log_candidates,
    )


def _prompt_general_trajectory_selection(
    discovery: GeneralMMPBSAInputDiscovery,
    *,
    search_root: Path,
) -> GeneralMMPBSAInputDiscovery:
    candidates = discovery.trajectory_candidates or [discovery.trajectory_path]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    if len(unique_candidates) <= 1:
        return discovery

    table = Table(title="Available MM-PBSA Trajectory Inputs", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Trajectory", style="bold white", overflow="fold")
    table.add_column("Matched mdin", style="cyan", overflow="fold")
    table.add_column("Matched output", style="white", overflow="fold")
    default_index = 1
    for index, trajectory in enumerate(unique_candidates, start=1):
        if trajectory.resolve() == discovery.trajectory_path.resolve():
            default_index = index
        mdin = _matching_mdin_for_trajectory(trajectory, discovery.production_mdin_candidates)
        output_log = _matching_output_log_for_trajectory(trajectory, discovery.output_log_candidates)
        table.add_row(
            str(index),
            _path_relative_to_batch_root(trajectory, batch_root=search_root),
            "-" if mdin is None else _path_relative_to_batch_root(mdin, batch_root=search_root),
            "-" if output_log is None else _path_relative_to_batch_root(output_log, batch_root=search_root),
        )
    console.print(table)
    while True:
        raw = typer.prompt("Choose the trajectory to use for MM-PBSA", default=str(default_index)).strip()
        try:
            index = int(raw)
        except ValueError:
            console.print("[bold red]Please enter one of the listed trajectory numbers.[/bold red]")
            continue
        if 1 <= index <= len(unique_candidates):
            selected = unique_candidates[index - 1]
            console.print(f"[bold cyan]Selected trajectory:[/bold cyan] {selected}")
            return _with_selected_general_trajectory(discovery, selected)
        console.print("[bold red]Please choose a number from the trajectory table.[/bold red]")


def _prompt_existing_directory(message: str, *, optional: bool = False) -> Path | None:
    while True:
        raw = typer.prompt(message, default="" if optional else None).strip()
        if optional and not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
        console.print(f"[bold red]Directory not found:[/bold red] {raw}")


def _prompt_general_mmpbsa_manual_input() -> GeneralMMPBSAInputDiscovery:
    print_notice(
        "Manual MM-PBSA Input",
        _GENERAL_MMPBSA_REQUIRED_FILES_MESSAGE,
        border_style="yellow",
    )
    folder = _prompt_existing_directory("Path to simulation data folder (blank = enter raw file paths)", optional=True)
    if folder is not None:
        discoveries = _discover_general_mmpbsa_inputs(folder)
        if discoveries:
            selected = _prompt_general_mmpbsa_input_selection(folder, discoveries)
            if selected is not None:
                return selected
        console.print("[bold yellow]No usable topology/trajectory pair was found in that folder.[/bold yellow]")
        console.print(f"[dim]{_GENERAL_MMPBSA_REQUIRED_FILES_MESSAGE}[/dim]")

    console.print("[dim]Raw file input mode: enter the AMBER files directly.[/dim]")
    prmtop_path = Path(str(_prompt_existing_path("Path to AMBER topology (*.prmtop, *.parm7, or *.top)"))).expanduser().resolve()
    trajectory_path = Path(str(_prompt_existing_path(f"Path to trajectory ({_GENERAL_TRAJECTORY_REQUIREMENT_TEXT})"))).expanduser().resolve()
    reference_structure_path_raw = _prompt_existing_path("Path to reference PDB (*.pdb, blank = optional)", optional=True)
    production_mdin_path_raw = _prompt_existing_path("Path to production mdin (*.in, blank = optional)", optional=True)
    output_root = prmtop_path.parent
    existing_status, existing_output_dir, matching_output_dirs = _discover_existing_mmpbsa_outputs(output_root)
    return GeneralMMPBSAInputDiscovery(
        root=output_root,
        prmtop_path=prmtop_path,
        trajectory_path=trajectory_path,
        reference_structure_path=None if reference_structure_path_raw is None else Path(reference_structure_path_raw).expanduser().resolve(),
        production_mdin_path=None if production_mdin_path_raw is None else Path(production_mdin_path_raw).expanduser().resolve(),
        existing_status=existing_status,
        existing_output_dir=existing_output_dir,
        matching_output_dirs=matching_output_dirs,
    )


def _display_prmtop_residue_summaries(prmtop_path: Path) -> list[Any]:
    summaries = summarize_mmpbsa_prmtop_residues(prmtop_path)
    table = Table(title="Topology Residues Available as Ligand", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Residue", style="bold white")
    table.add_column("Residue count", style="cyan", justify="right")
    table.add_column("Atom count", style="cyan", justify="right")
    table.add_column("Dry residue indices", style="white", overflow="fold")
    for index, summary in enumerate(summaries, start=1):
        table.add_row(
            str(index),
            summary.residue_name,
            str(summary.residue_count),
            str(summary.atom_count),
            _compress_indices_for_display(list(summary.dry_residue_indices)),
        )
    console.print(table)
    return summaries


def _compress_indices_for_display(indices: list[int]) -> str:
    if not indices:
        return "-"
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


def _parse_ligand_residue_selection(raw: str, summaries: list[Any]) -> list[str]:
    if not raw.strip():
        raise ValueError("Please select at least one ligand residue.")
    names_by_index = {index: summary.residue_name for index, summary in enumerate(summaries, start=1)}
    names_by_token = {summary.residue_name.upper(): summary.residue_name for summary in summaries}
    selected: list[str] = []
    seen: set[str] = set()
    for token in [item.strip() for item in raw.split(",") if item.strip()]:
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match is not None:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start > end:
                raise ValueError("Manual ranges must increase from left to right.")
            indices = range(start, end + 1)
        elif token.isdigit():
            indices = [int(token)]
        else:
            name = names_by_token.get(token.upper())
            if name is None:
                raise ValueError(f"Residue name is not available: {token}")
            if name not in seen:
                seen.add(name)
                selected.append(name)
            continue
        for index in indices:
            name = names_by_index.get(index)
            if name is None:
                raise ValueError("Please choose residue numbers from the displayed table.")
            if name not in seen:
                seen.add(name)
                selected.append(name)
    return selected


def _prompt_ligand_residue_names(prmtop_path: Path) -> list[str]:
    while True:
        summaries = _display_prmtop_residue_summaries(prmtop_path)
        if not summaries:
            raise typer.BadParameter("No dry solute residues were found in the selected topology.")
        raw = typer.prompt("Ligand residue numbers or names (comma separated)").strip()
        try:
            selected_names = _parse_ligand_residue_selection(raw, summaries)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        console.print(f"[bold cyan]Selected ligand residue name(s):[/bold cyan] {', '.join(selected_names)}")
        if typer.confirm("Use selected ligand residue(s) as ligand and all remaining dry solute as receptor?", default=True):
            return selected_names


def _display_receptor_residue_summaries(prmtop_path: Path, *, excluded_ligand_names: list[str]) -> list[Any]:
    excluded = {name.strip().upper() for name in excluded_ligand_names}
    summaries = [
        summary
        for summary in summarize_mmpbsa_prmtop_residues(prmtop_path)
        if summary.residue_name.upper() not in excluded
    ]
    table = Table(title="Topology Residues Available as Manual Receptor", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Residue", style="bold white")
    table.add_column("Residue count", style="cyan", justify="right")
    table.add_column("Atom count", style="cyan", justify="right")
    table.add_column("Dry residue indices", style="white", overflow="fold")
    for index, summary in enumerate(summaries, start=1):
        table.add_row(
            str(index),
            summary.residue_name,
            str(summary.residue_count),
            str(summary.atom_count),
            _compress_indices_for_display(list(summary.dry_residue_indices)),
        )
    console.print(table)
    return summaries


def _prompt_receptor_residue_names(prmtop_path: Path, *, ligand_residue_names: list[str]) -> list[str]:
    while True:
        summaries = _display_receptor_residue_summaries(prmtop_path, excluded_ligand_names=ligand_residue_names)
        if not summaries:
            raise typer.BadParameter("No receptor residues remain after excluding the selected ligand residue names.")
        raw = typer.prompt("Manual receptor residue numbers or names (comma separated)").strip()
        try:
            selected_names = _parse_ligand_residue_selection(raw, summaries)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        console.print(f"[bold cyan]Selected receptor residue name(s):[/bold cyan] {', '.join(selected_names)}")
        return selected_names


def _prompt_ligand_and_receptor_residue_names(prmtop_path: Path) -> tuple[list[str], list[str]]:
    while True:
        summaries = _display_prmtop_residue_summaries(prmtop_path)
        if not summaries:
            raise typer.BadParameter("No dry solute residues were found in the selected topology.")
        raw = typer.prompt("Ligand residue numbers or names (comma separated)").strip()
        try:
            ligand_names = _parse_ligand_residue_selection(raw, summaries)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        console.print(f"[bold cyan]Selected ligand residue name(s):[/bold cyan] {', '.join(ligand_names)}")
        if typer.confirm(
            "Use selected ligand residue(s) as ligand and all remaining dry solute as receptor? (N = Manual Selection)",
            default=True,
        ):
            return ligand_names, []
        receptor_names = _prompt_receptor_residue_names(prmtop_path, ligand_residue_names=ligand_names)
        return ligand_names, receptor_names


def _normalized_batch_md_status(status: str) -> str:
    if status in {"Not completed", "Not found"}:
        return "Incomplete"
    return status


def _path_relative_to_batch_root(path: Path | None, *, batch_root: Path | None) -> str:
    if path is None:
        return "-"
    resolved = path.resolve()
    if batch_root is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(batch_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _minimum_retained_donor_count(reference_donor_count: int) -> int:
    return min(2, max(1, reference_donor_count))


def _site_assessment_reason(assessment: Any) -> str:
    reasons: list[str] = []
    if assessment.displacement_angstrom > _BATCH_DIFFUSION_CUTOFF_ANGSTROM:
        reasons.append("shift")
    if assessment.retained_donor_count < _minimum_retained_donor_count(assessment.reference_donor_count):
        reasons.append("donor-loss")
    return "+".join(reasons) if reasons else "ok"


def _site_assessment_summary(assessment: Any) -> str:
    status = "[green]Stable[/green]" if assessment.stable else "[red]Unstable[/red]"
    reason = _site_assessment_reason(assessment)
    summary = (
        f"S{assessment.site} {status} "
        f"({assessment.displacement_angstrom:.2f} A; donors {assessment.retained_donor_count}/{assessment.reference_donor_count})"
    )
    if not assessment.stable:
        summary += f" [{reason}]"
    return summary


def _populate_batch_case_runtime_statuses(
    discoveries: list[MMPBSABatchDiscovery],
    *,
    dry_run: bool,
) -> None:
    for discovery in discoveries:
        if not discovery.selectable:
            discovery.md_status = "Incomplete"
            discovery.metal_status = "N/A"
            discovery.metal_note = discovery.workflow_note
            continue

        try:
            workflow = _ensure_batch_workflow_discovery(discovery)
            discovery.md_status = _normalized_batch_md_status(_production_status_text(workflow))
            if (
                workflow.prmtop_path is None
                or workflow.reference_structure_path is None
                or workflow.selected_stage is None
                or workflow.selected_stage.trajectory_path is None
            ):
                discovery.metal_status = "N/A"
                discovery.metal_note = discovery.workflow_note
                continue

            snapshot_manifest = run_last_snapshot_extraction(
                prmtop_path=str(workflow.prmtop_path),
                trajectory_path=str(workflow.selected_stage.trajectory_path),
                reference_structure_path=str(workflow.reference_structure_path),
                output_dir=discovery.workflow_root / ".simple_freee_batch_probe",
                dry_run=dry_run,
            )
            candidates = detect_bound_metal_sites(workflow.reference_structure_path, workflow.prmtop_path)
            discovery.metal_candidates = candidates
            if not candidates:
                discovery.metal_status = "No bound metal"
                discovery.metal_note = "No bound metal candidates were detected in the reference structure."
                continue

            assessments = [
                assess_site_stability(
                    workflow.reference_structure_path,
                    snapshot_manifest["last_snapshot_pdb"],
                    candidate,
                    diffusion_cutoff_angstrom=_BATCH_DIFFUSION_CUTOFF_ANGSTROM,
                    retained_donor_cutoff_angstrom=_BATCH_RETAINED_DONOR_CUTOFF_ANGSTROM,
                )
                for candidate in candidates
            ]
            discovery.site_assessments = {assessment.site: assessment for assessment in assessments}
            discovery.metal_status = "; ".join(_site_assessment_summary(item) for item in assessments)
            discovery.metal_note = " | ".join(item.note for item in assessments)
        except Exception as exc:
            discovery.md_status = discovery.md_status or "Unknown"
            discovery.metal_status = "Unknown"
            discovery.metal_note = str(exc)


def _display_mmpbsa_batch_cases(search_root: Path, discoveries: list[MMPBSABatchDiscovery]) -> None:
    table = Table(title="Detected MM-PBSA Batch Cases", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Case / PH", style="bold white")
    table.add_column("MD", style="white", no_wrap=True)
    table.add_column("Metal", style="white", overflow="fold")
    table.add_column("Workflow", style="white")
    table.add_column("MM-PBSA", style="cyan")
    table.add_column("Existing output", style="white", overflow="fold")
    for index, discovery in enumerate(discoveries, start=1):
        workflow_text = "Ready" if discovery.selectable else discovery.workflow_note
        status_text = discovery.existing_status.capitalize()
        table.add_row(
            str(index),
            discovery.display_key,
            discovery.md_status_text,
            discovery.metal_status_text,
            workflow_text,
            status_text,
            discovery.existing_output_text(batch_root=search_root),
        )
    console.print(table)
    console.print(f"[dim]Scanned {search_root.resolve()} for PDBID_METAL and PDBID_METAL/PHx workflow folders.[/dim]")
    console.print("[dim]Existing-output paths are shown relative to the batch root.[/dim]")


def _prompt_batch_search_root() -> Path:
    default_root = Path.cwd().resolve()
    console.print(f"[bold cyan]Batch root:[/bold cyan] {default_root}")
    if typer.confirm("Use the current directory as the MM-PBSA batch root?", default=True):
        return default_root
    while True:
        raw = typer.prompt("Path to the MM-PBSA batch root").strip()
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
        console.print(f"[bold red]Path not found:[/bold red] {raw}")


def _prompt_mmpbsa_batch_selection_mode() -> str:
    choices = [
        WizardChoice("all", "All detected cases", "Select every usable workflow discovered under the batch root."),
        WizardChoice(
            "missing_only",
            "Only cases without MM-PBSA output",
            "Select only usable workflow folders that do not already have MM-PBSA output.",
        ),
        WizardChoice("manual", "Manual case selection", "Select the workflow entries to include by number."),
    ]
    _display_choice_table("MM-PBSA Batch Selection", choices)
    return _prompt_choice("Choose which cases to include", choices, default_key="all")


def _parse_manual_selection(raw: str, discoveries: list[MMPBSABatchDiscovery]) -> list[int]:
    selected_indices: list[int] = []
    seen: set[int] = set()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Please enter at least one case number.")
    for token in tokens:
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError("Ranges must use numeric values such as 1-4.") from exc
            if start > end:
                raise ValueError("Manual ranges must increase from left to right.")
            values = range(start, end + 1)
        else:
            try:
                values = [int(token)]
            except ValueError as exc:
                raise ValueError("Manual selections must use numbers such as 1,3,5-7.") from exc
        for index in values:
            if index < 1 or index > len(discoveries):
                raise ValueError("Please choose numbers from the displayed case table.")
            discovery = discoveries[index - 1]
            if not discovery.selectable:
                raise ValueError(f"{discovery.display_key} is not ready: {discovery.workflow_note}")
            if index not in seen:
                seen.add(index)
                selected_indices.append(index)
    return selected_indices


def _prompt_manual_case_selection(discoveries: list[MMPBSABatchDiscovery]) -> list[MMPBSABatchDiscovery]:
    while True:
        raw = typer.prompt("Case numbers (comma separated, ranges allowed)").strip()
        try:
            indices = _parse_manual_selection(raw, discoveries)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        return [discoveries[index - 1] for index in indices]


def _prompt_mmpbsa_output_conflict_policy() -> str:
    choices = [
        WizardChoice(
            "overwrite",
            "Overwrite existing output",
            "Delete the selected existing MM-PBSA directory before regenerating it.",
        ),
        WizardChoice(
            "append",
            "Append numeric suffix",
            "Keep existing output and write the new run to MM-PBSA_1, MM-PBSA_2, and so on.",
        ),
    ]
    _display_choice_table("MM-PBSA Output Conflict Policy", choices)
    return _prompt_choice("Choose how to handle existing MM-PBSA output", choices, default_key="append")


def _next_batch_mmpbsa_output_directory(workflow_root: Path) -> Path:
    base = workflow_root / "MM-PBSA"
    if not any(child.is_dir() and _MMPBSA_OUTPUT_DIR_RE.match(child.name) for child in workflow_root.iterdir()):
        return base
    index = 1
    while True:
        candidate = workflow_root / f"MM-PBSA_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _plan_batch_outputs(
    selected: list[MMPBSABatchDiscovery],
    *,
    conflict_policy: str | None,
) -> list[MMPBSABatchCasePlan]:
    plans: list[MMPBSABatchCasePlan] = []
    for discovery in selected:
        if conflict_policy == "overwrite" and discovery.existing_output_dir is not None:
            plans.append(
                MMPBSABatchCasePlan(
                    discovery=discovery,
                    output_dir=discovery.existing_output_dir,
                    cleanup_output_dir=discovery.existing_output_dir,
                )
            )
            continue
        if conflict_policy == "append" and discovery.existing_status != "missing":
            plans.append(
                MMPBSABatchCasePlan(
                    discovery=discovery,
                    output_dir=_next_batch_mmpbsa_output_directory(discovery.workflow_root),
                )
            )
            continue
        plans.append(
            MMPBSABatchCasePlan(
                discovery=discovery,
                output_dir=discovery.workflow_root / "MM-PBSA",
            )
        )
    return plans


def _prompt_mmpbsa_solver_selection() -> tuple[bool, bool]:
    choices = [
        WizardChoice("both", "GB + PB", "Run both MM-GBSA and MM-PBSA."),
        WizardChoice("gb", "GB only", "Run only MM-GBSA."),
        WizardChoice("pb", "PB only", "Run only MM-PBSA."),
    ]
    _display_choice_table("MM-PBSA Solvers", choices)
    selection = _prompt_choice("Choose the MM-PBSA solvers", choices, default_key="both")
    if selection == "gb":
        return True, False
    if selection == "pb":
        return False, True
    return True, True


def _prompt_batch_frame_policy() -> tuple[str, int]:
    window_choices = [
        WizardChoice("full", "Whole trajectory", "Use each case's full available trajectory."),
        WizardChoice("last10", "Last 10%", "Use only the last 10% of each case's saved frames."),
    ]
    _display_choice_table("MM-PBSA Batch Frame Window", window_choices)
    window_mode = _prompt_choice("Choose the frame window policy", window_choices, default_key="last10")
    stride_choices = [
        WizardChoice("1", "Stride 1", "Use every saved frame inside the selected window."),
        WizardChoice("10", "Stride 10", "Use every tenth saved frame inside the selected window."),
        WizardChoice("custom", "Custom stride", "Enter another integer stride manually."),
    ]
    _display_choice_table("MM-PBSA Batch Frame Stride", stride_choices)
    stride_selection = _prompt_choice("Choose the frame stride", stride_choices, default_key="1")
    if stride_selection == "custom":
        return window_mode, typer.prompt("Custom frame stride", default=1, type=int)
    return window_mode, int(stride_selection)


def _resolve_frame_window_from_policy(
    *,
    estimated_frames: int | None,
    window_mode: str,
    frame_stride: int,
) -> tuple[int | None, int | None, int]:
    if estimated_frames is None:
        return None, None, frame_stride
    if window_mode == "full":
        return 1, estimated_frames, frame_stride
    last_10_count = max(1, math.ceil(estimated_frames * 0.10))
    last_10_start = max(1, estimated_frames - last_10_count + 1)
    return last_10_start, estimated_frames, frame_stride


def _display_bound_site_choices(case_label: str, candidates: list[Any]) -> None:
    table = Table(title=f"Detected bound metal candidates for {case_label}", box=box.SIMPLE_HEAVY)
    table.add_column("Site", style="bold cyan", justify="right")
    table.add_column("Metal", style="bold white")
    table.add_column("Reference donors", style="white", justify="right")
    for candidate in candidates:
        table.add_row(
            str(candidate.site),
            f"{candidate.element} at {candidate.key}",
            str(candidate.donor_count),
        )
    console.print(table)


def _select_batch_case_site(discovery: MMPBSABatchDiscovery) -> tuple[Any, int]:
    workflow = _ensure_batch_workflow_discovery(discovery)
    if workflow is None or workflow.prmtop_path is None or workflow.reference_structure_path is None:
        raise ValueError(f"{discovery.display_key} is not ready for MM-PBSA.")
    candidates = discovery.metal_candidates or detect_bound_metal_sites(
        str(workflow.reference_structure_path),
        str(workflow.prmtop_path),
    )
    if not candidates:
        raise ValueError(f"No bound metal candidates were detected for {discovery.display_key}.")
    if len(candidates) == 1:
        selected = candidates[0]
        console.print(
            f"[dim]{discovery.summary_label}: one bound metal candidate was detected, so site {selected.site} is selected automatically.[/dim]"
        )
        formal_charge = _infer_charge_from_selected_site(selected) or default_formal_charge(selected.element)
        return selected, formal_charge

    _display_bound_site_choices(discovery.summary_label, candidates)
    choices = [
        WizardChoice(str(candidate.site), f"Site {candidate.site}", f"{candidate.element} at {candidate.key}")
        for candidate in candidates
    ]
    _display_choice_table(f"Bound metal site selection for {discovery.summary_label}", choices)
    selected = select_site(candidates, int(_prompt_choice("Choose the metal site to analyze", choices)))
    formal_charge = _infer_charge_from_selected_site(selected) or default_formal_charge(selected.element)
    return selected, formal_charge


def _batch_case_suffix(plan: MMPBSABatchCasePlan) -> tuple[str, ...]:
    return (plan.discovery.case_id, plan.discovery.ph_label)


def _suffixed_write_config_path(path: str | Path, suffix_tokens: tuple[str, ...]) -> Path:
    target = Path(path)
    suffix = "_".join(token.replace("/", "_") for token in suffix_tokens if token)
    if not suffix:
        return target
    return target.with_name(f"{target.stem}_{suffix}{target.suffix}")


def _save_free_energy_wizard_configs(result: FreeEnergyWizardBuildResult, write_config: str | None) -> list[Path]:
    if not write_config:
        return []
    if len(result.configs) == 1:
        return [save_config(result.configs[0], write_config)]
    saved_paths: list[Path] = []
    for config, suffix_tokens in zip(result.configs, result.config_suffixes, strict=True):
        saved_paths.append(save_config(config, _suffixed_write_config_path(write_config, suffix_tokens)))
    return saved_paths


def _free_energy_output_label(output_dir: str | Path) -> str:
    path = Path(output_dir)
    if path.parent.name.upper().startswith("PH") and path.parent.parent.name:
        return f"{path.parent.parent.name}/{path.parent.name}"
    if path.name.upper().startswith("PH") and path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name


def _build_single_free_energy_wizard_config(
    *,
    dry_run: bool,
    forced_method: FreeEnergyMethod | None,
) -> FreeEnergyWorkflowConfig:
    amber_env = detect_amber_environment()
    _print_step_header(
        1,
        "Load the Complex Trajectory Inputs",
        "Select a main.py workflow folder from the current directory, enter a workflow path manually, or fall back to the raw topology/trajectory files.",
    )
    input_selection = _prompt_complex_input_selection()
    prmtop_path = input_selection.complex_input.prmtop_path
    trajectory_path = input_selection.complex_input.trajectory_path
    reference_structure_path = input_selection.complex_input.reference_structure_path
    production_mdin_path = input_selection.complex_input.production_mdin_path

    wizard_tmp = Path(".simple_ti_wizard")
    console.print("[dim]The wizard will now inspect the last snapshot before choosing the free-energy path.[/dim]")
    last_snapshot = run_last_snapshot_extraction(
        prmtop_path=prmtop_path,
        trajectory_path=trajectory_path,
        reference_structure_path=reference_structure_path,
        output_dir=wizard_tmp / "snapshot_probe",
        dry_run=dry_run,
    )
    candidates = detect_bound_metal_sites(
        reference_structure_path,
        prmtop_path,
        include_unbound_metals=_input_selection_includes_unbound_metal_sites(input_selection),
    )
    if not candidates:
        raise typer.BadParameter("No bound metal candidates were detected in the reference structure.")
    assessments = [
        assess_site_stability(
            reference_structure_path,
            last_snapshot["last_snapshot_pdb"],
            candidate,
            diffusion_cutoff_angstrom=2.5,
            retained_donor_cutoff_angstrom=3.5,
        )
        for candidate in candidates
    ]
    _display_site_table(candidates, assessments)

    selected_sites = []
    ti_selection_mode = TIMetalSelectionMode.SINGLE
    if len(candidates) == 1:
        selected = candidates[0]
        selected_sites = [selected]
        console.print(f"[dim]One bound metal candidate was detected, so site {selected.site} is selected automatically.[/dim]")
    else:
        choices = [
            WizardChoice("0", "Each metal separately", "Default. Generate one TI setup per detected metal site."),
            WizardChoice("all", "All at once", "Decouple all detected metal sites together in one TI setup; reports one total dG."),
        ] + [
            WizardChoice(str(candidate.site), f"Site {candidate.site}", f"{candidate.element} at {candidate.key}")
            for candidate in candidates
        ]
        _display_choice_table("Bound metal site selection", choices)
        raw_selection = _prompt_choice("Choose the metal site to analyze", choices, default_key="0")
        if raw_selection == "0":
            ti_selection_mode = TIMetalSelectionMode.ONE_BY_ONE
            selected_sites = list(candidates)
            selected = selected_sites[0]
        elif raw_selection.lower() == "all":
            ti_selection_mode = TIMetalSelectionMode.ALL_AT_ONCE
            selected_sites = list(candidates)
            selected = selected_sites[0]
        else:
            selected = select_site(candidates, int(raw_selection))
            selected_sites = [selected]

    selected_assessments = [next(item for item in assessments if item.site == item_site.site) for item_site in selected_sites]
    selected_assessment = selected_assessments[0]
    allow_unstable = False
    unstable_assessments = [item for item in selected_assessments if not item.stable]
    if unstable_assessments:
        print_notice("Strong Warning", "\n".join(item.note for item in unstable_assessments), border_style="bold red")
        allow_unstable = typer.confirm("Proceed with the selected site anyway?", default=False)
        if not allow_unstable:
            raise typer.Abort()

    if forced_method is None:
        _print_step_header(
            2,
            "Choose the Free-Energy Method",
            "FreeE can either continue with the existing TI workflow or generate a lightweight MM-PBSA comparison path.",
        )
        method = _prompt_free_energy_method()
    else:
        method = forced_method

    if method == FreeEnergyMethod.TI:
        in_place_ti = _input_selection_uses_in_place_ti(input_selection)
        _print_step_header(
            3 if forced_method is None else 2,
            "Choose the TI Mode, Snapshot, and Reference Settings",
            (
                "DES/raw AMBER inputs use in-place TI from the existing MD topology and restart. "
                "A metal-in-water reference leg is optional for RBFE and will be offered separately."
                if in_place_ti
                else "First choose how TI should be implemented. Then pick the snapshot source and confirm the metal-in-water reference settings used for the DeltaG reference leg."
            ),
        )
        if in_place_ti:
            ti_implementation_mode = TIImplementationMode.AMBER_12_6_4_GTI
            ti_decoupling_mode = _prompt_ti_decoupling_mode(ti_implementation_mode)
            print_notice(
                "In-Place TI",
                "DES/raw AMBER input was detected, so FreeE will keep the existing prmtop nonbonded model, "
                "generate only the bound-system TI leg, and use the selected GTI decoupling path. "
                "The metal-in-water reference and water-model prompt are used only if you request the optional RBFE reference leg.",
                border_style="cyan",
            )
        else:
            ti_implementation_mode = _prompt_ti_implementation_mode()
            ti_decoupling_mode = _prompt_ti_decoupling_mode(ti_implementation_mode)
        snapshot_mode = _prompt_snapshot_mode(selected_stable=all(item.stable for item in selected_assessments))
        if in_place_ti:
            formal_charge = _infer_charge_from_selected_site(selected) or default_formal_charge(selected.element)
            water_reference_enabled = typer.confirm(
                "Generate a metal-in-water reference leg for RBFE analysis?",
                default=False,
            )
            if water_reference_enabled:
                formal_charge, water_model, custom_ion_frcmods = _resolve_water_reference_settings(
                    amber_env=amber_env,
                    selected=selected,
                    workflow_root=input_selection.workflow_root,
                    implementation_mode=ti_implementation_mode,
                )
                reuse_existing, reuse_from_library, library_key = _resolve_water_reference_source_choice(
                    complex_input=input_selection.complex_input,
                    selected=selected,
                    formal_charge=formal_charge,
                    water_model=water_model,
                    custom_ion_frcmods=custom_ion_frcmods,
                    implementation_mode=ti_implementation_mode,
                    decoupling_mode=ti_decoupling_mode,
                )
            else:
                water_model = "opc"
                custom_ion_frcmods = []
                reuse_existing = False
                reuse_from_library = False
                library_key = None
        else:
            water_reference_enabled = True
            formal_charge, water_model, custom_ion_frcmods = _resolve_water_reference_settings(
                amber_env=amber_env,
                selected=selected,
                workflow_root=input_selection.workflow_root,
                implementation_mode=ti_implementation_mode,
            )
            reuse_existing, reuse_from_library, library_key = _resolve_water_reference_source_choice(
                complex_input=input_selection.complex_input,
                selected=selected,
                formal_charge=formal_charge,
                water_model=water_model,
                custom_ion_frcmods=custom_ion_frcmods,
                implementation_mode=ti_implementation_mode,
                decoupling_mode=ti_decoupling_mode,
            )

        _print_step_header(
            4 if forced_method is None else 3,
            "Choose the Execution Script and Output Location",
            "Select whether to generate CPU or GPU master sbatch files. A separate Tahoma script will also be written automatically.",
        )
        profile = _prompt_execution_profile()
        if ti_implementation_mode.value == "amber_12_6_4_gti" and profile != SlurmProfile.GPU:
            console.print("[bold yellow]GTI/CUDA mode requires pmemd.cuda, so the execution profile was set to GPU.[/bold yellow]")
            profile = SlurmProfile.GPU
        output_dir_path = _default_output_directory(
            reference_structure_path,
            workflow_root=input_selection.workflow_root,
        )
        console.print(f"[bold cyan]Output directory:[/bold cyan] {output_dir_path}")
        return FreeEnergyWorkflowConfig(
            complex_input=input_selection.complex_input.model_dump(mode="json"),
            snapshot=SnapshotConfig(
                mode=snapshot_mode,
                allow_unstable_last_snapshot=allow_unstable,
            ),
            metal=MetalSelectionConfig(
                selection_mode=ti_selection_mode,
                selected_site=selected.site,
                selected_sites=[item.site for item in selected_sites],
                formal_charge=formal_charge,
                formal_charges_by_site={
                    item.site: _infer_charge_from_selected_site(item) or default_formal_charge(item.element)
                    for item in selected_sites
                },
            ),
            free_energy=FreeEnergyConfig(method=method),
            ti=TIProtocolConfig(
                implementation_mode=ti_implementation_mode,
                decoupling_mode=ti_decoupling_mode,
            ),
            water_reference=WaterReferenceConfig(
                enabled=water_reference_enabled,
                bound_in_place=in_place_ti,
                water_model=water_model,
                cache_dir=str((Path.cwd() / "water_ref").resolve()),
                reuse_existing=reuse_existing,
                reuse_from_library=reuse_from_library,
                library_key=library_key,
                custom_ion_frcmods=custom_ion_frcmods,
            ),
            slurm={
                "profile": profile,
                "partition": None,
                "account": None,
                "ntasks": 8,
                "gpus": 1,
                "walltime": "24:00:00",
                "binary_override": None,
                "job_name": output_dir_path.name,
            },
            output_dir=str(output_dir_path),
        )

    _print_step_header(
        3 if forced_method is None else 2,
        "Choose the MM-PBSA Settings",
        "MM-PBSA v1 uses Amber MM-GBSA/MM-PBSA as a simple comparison path rooted at the selected bound workflow. "
        "The selected metal atom is treated as the ligand.",
    )
    start_frame, end_frame, frame_stride = _prompt_mmpbsa_frame_window(
        trajectory_path=trajectory_path,
        production_mdin_path=production_mdin_path,
        workflow_root=input_selection.workflow_root,
    )
    include_entropy, entropy_method = _prompt_entropy_method()
    formal_charge = _infer_charge_from_selected_site(selected) or default_formal_charge(selected.element)
    if input_selection.workflow_root is not None:
        output_dir_path = _default_mmpbsa_output_directory(
            reference_structure_path,
            workflow_root=input_selection.workflow_root,
        )
    else:
        output_dir_path = _default_mmpbsa_output_directory(reference_structure_path)
    output_dir_path = _resolve_mmpbsa_output_directory(output_dir_path)
    console.print(f"[bold cyan]Output directory:[/bold cyan] {output_dir_path}")
    console.print("[dim]MM-PBSA uses a CPU-oriented sbatch template in this first version.[/dim]")
    return FreeEnergyWorkflowConfig(
        complex_input=input_selection.complex_input.model_dump(mode="json"),
        snapshot=SnapshotConfig(
            allow_unstable_last_snapshot=allow_unstable,
        ),
        metal=MetalSelectionConfig(
            selected_site=selected.site,
            formal_charge=formal_charge,
        ),
        free_energy=FreeEnergyConfig(method=method),
        mmpbsa=MMPBSAConfig(
            include_entropy=include_entropy,
            entropy_method=entropy_method,
            frame_stride=frame_stride,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
        slurm={
            "profile": SlurmProfile.CPU,
            "partition": None,
            "account": None,
            "ntasks": 8,
            "gpus": 0,
            "walltime": "24:00:00",
            "binary_override": None,
            "job_name": output_dir_path.name,
        },
        output_dir=str(output_dir_path),
    )


def build_free_energy_wizard_config(write_config: str | None, *, dry_run: bool) -> FreeEnergyWorkflowConfig:
    config = _build_single_free_energy_wizard_config(dry_run=dry_run, forced_method=None)
    if write_config:
        saved = save_config(config, write_config)
        console.print(f"Saved config to {saved}")
    return config


def _build_general_mmpbsa_wizard_result(
    *,
    search_root: Path,
    write_config: str | None,
) -> FreeEnergyWizardBuildResult:
    _print_step_header(
        3,
        "Choose General MM-PBSA Inputs",
        "No PDBID_METAL or PDBID_METAL/PHx batch cases were detected. FreeE will scan for a general AMBER topology/trajectory bundle instead.",
    )
    discoveries = _discover_general_mmpbsa_inputs(search_root)
    if discoveries:
        input_discovery = _prompt_general_mmpbsa_input_selection(search_root, discoveries)
        if input_discovery is None:
            input_discovery = _prompt_general_mmpbsa_manual_input()
    else:
        console.print("[bold yellow]No general AMBER simulation data folders were detected under the selected root.[/bold yellow]")
        console.print(f"[dim]{_GENERAL_MMPBSA_REQUIRED_FILES_MESSAGE}[/dim]")
        input_discovery = _prompt_general_mmpbsa_manual_input()

    input_discovery = _prompt_general_trajectory_selection(input_discovery, search_root=input_discovery.root)
    ligand_residue_names, receptor_residue_names = _prompt_ligand_and_receptor_residue_names(input_discovery.prmtop_path)

    _print_step_header(
        4,
        "Choose the General MM-PBSA Settings",
        "Pick the solvers, frame window, stride, and entropy settings for the selected topology/trajectory bundle.",
    )
    run_gb, run_pb = _prompt_mmpbsa_solver_selection()
    start_frame, end_frame, frame_stride = _prompt_mmpbsa_frame_window(
        trajectory_path=str(input_discovery.trajectory_path),
        production_mdin_path=None if input_discovery.production_mdin_path is None else str(input_discovery.production_mdin_path),
        workflow_root=None,
    )
    include_entropy, entropy_method = _prompt_entropy_method()
    output_dir_path = _resolve_mmpbsa_output_directory(input_discovery.output_root / "MM-PBSA")
    console.print(f"[bold cyan]Output directory:[/bold cyan] {output_dir_path}")
    console.print("[dim]MM-PBSA uses a CPU-oriented sbatch template in this first version.[/dim]")

    config = FreeEnergyWorkflowConfig(
        complex_input=input_discovery.complex_input().model_dump(mode="json"),
        snapshot=SnapshotConfig(),
        free_energy=FreeEnergyConfig(method=FreeEnergyMethod.MMPBSA),
        mmpbsa=MMPBSAConfig(
            run_gb=run_gb,
            run_pb=run_pb,
            include_entropy=include_entropy,
            entropy_method=entropy_method,
            frame_stride=frame_stride,
            start_frame=start_frame,
            end_frame=end_frame,
            ligand_selection_mode=MMPBSALigandSelectionMode.RESIDUE_NAME,
            ligand_residue_names=ligand_residue_names,
            receptor_selection_mode=(
                MMPBSAReceptorSelectionMode.AUTO
                if not receptor_residue_names
                else MMPBSAReceptorSelectionMode.RESIDUE_NAME
            ),
            receptor_residue_names=receptor_residue_names,
        ),
        slurm={
            "profile": SlurmProfile.CPU,
            "partition": None,
            "account": None,
            "ntasks": 8,
            "gpus": 0,
            "walltime": "24:00:00",
            "binary_override": None,
            "job_name": output_dir_path.name,
        },
        output_dir=str(output_dir_path),
    )
    result = FreeEnergyWizardBuildResult(configs=[config], config_suffixes=[()])
    result.saved_config_paths.extend(_save_free_energy_wizard_configs(result, write_config))
    if result.saved_config_paths:
        console.print(f"Saved config to {result.saved_config_paths[0]}")
    return result


def build_free_energy_wizard_configs(write_config: str | None, *, dry_run: bool) -> FreeEnergyWizardBuildResult:
    _print_step_header(
        1,
        "Choose the Free-Energy Method",
        "FreeE can either continue with the existing TI workflow or generate a batch MM-PBSA path from detected PDBID_METAL and PDBID_METAL/PHx folders.",
    )
    method = _prompt_free_energy_method()
    if method == FreeEnergyMethod.TI:
        config = _build_single_free_energy_wizard_config(dry_run=dry_run, forced_method=FreeEnergyMethod.TI)
        result = _expand_ti_one_by_one_config(config)
        result.saved_config_paths.extend(_save_free_energy_wizard_configs(result, write_config))
        if result.saved_config_paths:
            console.print("[bold cyan]Saved TI config(s):[/bold cyan]")
            for saved_path in result.saved_config_paths:
                console.print(f"  - {saved_path}")
        return result

    _print_step_header(
        2,
        "Choose the MM-PBSA Batch Root",
        "FreeE will scan a batch root for PDBID_METAL and PDBID_METAL/PHx workflow folders and show which entries already have MM-PBSA output.",
    )
    batch_root = _prompt_batch_search_root()
    discoveries = _discover_mmpbsa_batch_cases(batch_root)
    if not discoveries:
        print_notice(
            "General MM-PBSA Fallback",
            "No PDBID_METAL or PDBID_METAL/PHx workflow folders were detected under the selected batch root. "
            "FreeE will continue with a general AMBER topology/trajectory MM-PBSA setup.",
            border_style="yellow",
        )
        return _build_general_mmpbsa_wizard_result(search_root=batch_root, write_config=write_config)
    if not any(item.selectable for item in discoveries) and _discover_general_mmpbsa_inputs(batch_root):
        print_notice(
            "General MM-PBSA Fallback",
            "The detected folders are not usable main.py workflow directories, but raw AMBER topology/trajectory "
            "files were found under the selected batch root. FreeE will continue with the general raw AMBER setup.",
            border_style="yellow",
        )
        return _build_general_mmpbsa_wizard_result(search_root=batch_root, write_config=write_config)
    console.print("[dim]Inspecting production status and last-snapshot metal stability for detected cases...[/dim]")
    _populate_batch_case_runtime_statuses(discoveries, dry_run=dry_run)
    _display_mmpbsa_batch_cases(batch_root, discoveries)

    selection_mode = _prompt_mmpbsa_batch_selection_mode()
    if selection_mode == "all":
        selected = [item for item in discoveries if item.selectable]
    elif selection_mode == "missing_only":
        selected = [item for item in discoveries if item.selectable and item.existing_status == "missing"]
    else:
        selected = _prompt_manual_case_selection(discoveries)
    if not selected:
        raise typer.BadParameter("No usable MM-PBSA batch cases were selected.")

    conflict_policy = None
    if any(item.existing_status != "missing" for item in selected):
        conflict_policy = _prompt_mmpbsa_output_conflict_policy()
    case_plans = _plan_batch_outputs(selected, conflict_policy=conflict_policy)

    _print_step_header(
        3,
        "Choose the MM-PBSA Batch Settings",
        "Pick the solvers, frame-window policy, and entropy settings that will be applied across the selected batch.",
    )
    run_gb, run_pb = _prompt_mmpbsa_solver_selection()
    window_mode, frame_stride = _prompt_batch_frame_policy()
    include_entropy, entropy_method = _prompt_entropy_method()
    console.print("[dim]MM-PBSA batch generation uses the CPU-oriented sbatch template for each case.[/dim]")

    planned_output_table = Table(title="Planned MM-PBSA Batch Outputs", box=box.SIMPLE_HEAVY)
    planned_output_table.add_column("Case / PH", style="bold white")
    planned_output_table.add_column("Output directory", style="cyan", overflow="fold")

    configs: list[FreeEnergyWorkflowConfig] = []
    suffixes: list[tuple[str, ...]] = []
    for case_plan in case_plans:
        discovery = case_plan.discovery
        workflow = _ensure_batch_workflow_discovery(discovery)
        if workflow is None or workflow.prmtop_path is None or workflow.reference_structure_path is None or workflow.selected_stage is None or workflow.selected_stage.trajectory_path is None:
            raise ValueError(f"{discovery.display_key} is not ready for MM-PBSA.")
        selected_site, formal_charge = _select_batch_case_site(discovery)
        estimated_frames, estimate_note = _estimate_mmpbsa_saved_frames(
            trajectory_path=str(workflow.selected_stage.trajectory_path),
            production_mdin_path=str(workflow.selected_stage.input_path),
            workflow_root=workflow.root,
        )
        start_frame, end_frame, resolved_stride = _resolve_frame_window_from_policy(
            estimated_frames=estimated_frames,
            window_mode=window_mode,
            frame_stride=frame_stride,
        )
        console.print(f"[dim]{discovery.summary_label}: {estimate_note}[/dim]")
        if start_frame is not None and end_frame is not None:
            console.print(
                f"[dim]{discovery.summary_label}: using frames {start_frame}-{end_frame} with stride {resolved_stride}.[/dim]"
            )
        else:
            console.print(
                f"[dim]{discovery.summary_label}: frame count could not be estimated, so MM-PBSA will use the full available trajectory with stride {resolved_stride}.[/dim]"
            )
        planned_output_table.add_row(
            discovery.display_key,
            _path_relative_to_batch_root(case_plan.output_dir, batch_root=batch_root),
        )
        configs.append(
            FreeEnergyWorkflowConfig(
                complex_input={
                    "prmtop_path": str(workflow.prmtop_path),
                    "trajectory_path": str(workflow.selected_stage.trajectory_path),
                    "reference_structure_path": str(workflow.reference_structure_path),
                    "production_mdin_path": str(workflow.selected_stage.input_path),
                },
                snapshot=SnapshotConfig(),
                metal=MetalSelectionConfig(
                    selected_site=selected_site.site,
                    formal_charge=formal_charge,
                ),
                free_energy=FreeEnergyConfig(method=FreeEnergyMethod.MMPBSA),
                mmpbsa=MMPBSAConfig(
                    run_gb=run_gb,
                    run_pb=run_pb,
                    include_entropy=include_entropy,
                    entropy_method=entropy_method,
                    frame_stride=resolved_stride,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
                slurm={
                    "profile": SlurmProfile.CPU,
                    "partition": None,
                    "account": None,
                    "ntasks": 8,
                    "gpus": 0,
                    "walltime": "24:00:00",
                    "binary_override": None,
                    "job_name": case_plan.output_dir.name,
                },
                output_dir=str(case_plan.output_dir),
            )
        )
        suffixes.append(_batch_case_suffix(case_plan))

    console.print(planned_output_table)
    result = FreeEnergyWizardBuildResult(
        configs=configs,
        config_suffixes=suffixes,
        mmpbsa_batch_plan=MMPBSABatchPlan(
            batch_root=batch_root.resolve(),
            selection_mode=selection_mode,
            conflict_policy=conflict_policy,
            cases=case_plans,
        ),
    )
    result.saved_config_paths.extend(_save_free_energy_wizard_configs(result, write_config))
    if result.saved_config_paths:
        console.print("[bold cyan]Saved batch configs:[/bold cyan]")
        for saved_path in result.saved_config_paths:
            console.print(f"  - {saved_path}")
    return result


def _case_spec_from_execution(
    result: dict[str, Any],
    *,
    discovery: MMPBSABatchDiscovery | None = None,
) -> dict[str, Any]:
    output_dir = Path(str(result.get("output_dir") or ".")).resolve()
    if discovery is not None:
        case_id = discovery.case_id
        ph_label = discovery.ph_label
        display_name = discovery.summary_label
        case_root = _batch_discovery_case_root(discovery)
    else:
        workflow_root = output_dir.parent
        if _PH_DIR_RE.match(workflow_root.name):
            case_root = workflow_root.parent
            case_id = case_root.name
            ph_label = workflow_root.name
        else:
            case_root = workflow_root
            case_id = workflow_root.name
            ph_label = ""
        display_name = _format_batch_case_label(case_id, ph_label, separator=" ")
    assets = result.get("assets") or {}
    return {
        "case_id": case_id,
        "ph_label": ph_label,
        "display_name": display_name,
        "case_root": str(case_root),
        "output_dir": str(output_dir),
        "cluster_sbatch": str(assets.get("slurm") or ""),
        "tahoma_sbatch": str(assets.get("tahoma") or ""),
        "requested_solvers": [str(item).lower() for item in result.get("requested_solvers") or []],
    }


def _write_batch_root_manifest(
    *,
    batch_plan: MMPBSABatchPlan,
    top_level_assets: dict[str, str],
    generated_case_specs: list[dict[str, Any]],
    group_assets: dict[str, dict[str, str]],
    failed_cases: list[dict[str, str]],
) -> Path:
    payload = {
        "summary_type": "mmpbsa_batch_manifest",
        "batch_dir": str(batch_plan.batch_root),
        "selection_mode": batch_plan.selection_mode,
        "conflict_policy": batch_plan.conflict_policy,
        "cases": generated_case_specs,
        "outputs": {
            "summary_text": top_level_assets["summary_text"],
            "summary_json": top_level_assets["summary_json"],
        },
        "assets": top_level_assets,
        "selected_cases": [
            {
                "case_id": case_plan.discovery.case_id,
                "ph_label": case_plan.discovery.ph_label,
                "case_root": str(_batch_discovery_case_root(case_plan.discovery)),
                "workflow_root": str(case_plan.discovery.workflow_root),
                "existing_status": case_plan.discovery.existing_status,
                "existing_output_dir": None if case_plan.discovery.existing_output_dir is None else str(case_plan.discovery.existing_output_dir),
                "planned_output_dir": str(case_plan.output_dir),
            }
            for case_plan in batch_plan.cases
        ],
        "group_assets": group_assets,
        "failed_cases": failed_cases,
    }
    return write_json(Path(top_level_assets["manifest"]), payload)


def _finalize_mmpbsa_batch_assets(
    *,
    batch_plan: MMPBSABatchPlan,
    successful_results: list[dict[str, Any]],
    failed_cases: list[dict[str, str]],
) -> None:
    if not successful_results:
        return
    discovery_by_output_dir = {
        str(case_plan.output_dir.resolve()): case_plan.discovery
        for case_plan in batch_plan.cases
    }
    case_specs = [
        _case_spec_from_execution(
            result,
            discovery=discovery_by_output_dir.get(str(Path(str(result.get("output_dir") or ".")).resolve())),
        )
        for result in successful_results
    ]
    grouped_specs: dict[Path, list[dict[str, Any]]] = {}
    for spec in case_specs:
        output_dir = Path(str(spec["output_dir"])).resolve()
        group_root = Path(str(spec.get("case_root") or output_dir.parent.parent)).resolve()
        grouped_specs.setdefault(group_root, []).append(spec)

    group_assets: dict[str, dict[str, str]] = {}
    for group_root, group_case_specs in grouped_specs.items():
        if group_root.resolve() == batch_plan.batch_root.resolve():
            continue
        group_assets[str(group_root)] = write_mmpbsa_batch_submission_assets(
            batch_dir=group_root,
            case_specs=group_case_specs,
        )
    top_level_assets = write_mmpbsa_batch_submission_assets(
        batch_dir=batch_plan.batch_root,
        case_specs=case_specs,
    )
    _write_batch_root_manifest(
        batch_plan=batch_plan,
        top_level_assets=top_level_assets,
        generated_case_specs=case_specs,
        group_assets=group_assets,
        failed_cases=failed_cases,
    )


def _relative_script_path(path: str | Path | None, *, batch_root: Path) -> str | None:
    if not path:
        return None
    raw = Path(str(path)).expanduser()
    if not raw.exists():
        return raw.as_posix()
    return Path(os.path.relpath(raw.resolve(), start=batch_root.resolve())).as_posix()


def _tahoma_script_for_sbatch(path: str | Path | None) -> str | None:
    if not path:
        return None
    sbatch_path = Path(str(path))
    candidate = sbatch_path.parent / f"tahoma_{sbatch_path.name.removeprefix('run_').removesuffix('_gpu.sbatch').removesuffix('_cpu.sbatch')}.sbatch"
    if candidate.exists():
        return str(candidate)
    if sbatch_path.name.startswith("run_bound"):
        candidate = sbatch_path.parent / "tahoma_bound.sbatch"
    elif sbatch_path.name.startswith("run_water_ref"):
        candidate = sbatch_path.parent / "tahoma_water_ref.sbatch"
    return str(candidate) if candidate.exists() else None


def _write_ti_batch_submission_assets(
    *,
    batch_plan: TIBatchPlan,
    successful_results: list[dict[str, Any]],
    failed_cases: list[dict[str, str]],
) -> None:
    batch_root = batch_plan.batch_root
    batch_root.mkdir(parents=True, exist_ok=True)
    case_by_output = {str(case.output_dir.resolve()): case for case in batch_plan.cases}
    cases: list[dict[str, Any]] = []
    for result in successful_results:
        output_dir = Path(str(result.get("output_dir") or ".")).resolve()
        plan_case = case_by_output.get(str(output_dir))
        bound_slurm = _relative_script_path(result.get("bound_slurm"), batch_root=batch_root)
        water_slurm = _relative_script_path(result.get("water_slurm"), batch_root=batch_root)
        bound_tahoma = _relative_script_path(_tahoma_script_for_sbatch(result.get("bound_slurm")), batch_root=batch_root)
        water_tahoma = _relative_script_path(_tahoma_script_for_sbatch(result.get("water_slurm")), batch_root=batch_root)
        cases.append(
            {
                "site": None if plan_case is None else plan_case.site,
                "element": None if plan_case is None else plan_case.element,
                "atom_index": None if plan_case is None else plan_case.atom_index,
                "output_dir": _relative_script_path(output_dir, batch_root=batch_root),
                "selected_metal": result.get("selected_metal"),
                "bound_sbatch": bound_slurm,
                "water_sbatch": water_slurm,
                "bound_tahoma_sbatch": bound_tahoma,
                "water_tahoma_sbatch": water_tahoma,
            }
        )
    manifest_path = batch_root / "ti_batch_manifest.json"
    write_json(
        manifest_path,
        {
            "summary_type": "ti_batch_manifest",
            "selection_mode": batch_plan.selection_mode,
            "batch_root": str(batch_root),
            "cases": cases,
            "failed_cases": failed_cases,
            "submit_all_template": str(batch_root / "submit_all_template.sh"),
            "submit_all_tahoma": str(batch_root / "submit_all_tahoma.sh"),
        },
    )
    template_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd -- "$(cd -- "$(dirname -- "$0")" && pwd)"',
        "",
    ]
    tahoma_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'cd -- "$(cd -- "$(dirname -- "$0")" && pwd)"',
        "",
        'ACCOUNT="${ACCOUNT:-emsl62113}"',
        'TIME="${TIME:-48:00:00}"',
        'GPU_NODES="${GPU_NODES:-1}"',
        'GPU_GRES="${GPU_GRES:-gpu:2}"',
        'PARTITION="${PARTITION:-analysis}"',
        'JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-gTI}"',
        "",
    ]
    for index, case in enumerate(cases, start=1):
        for key in ("water_sbatch", "bound_sbatch"):
            script = case.get(key)
            if script:
                template_lines.append(f"sbatch {script}")
        for key in ("water_tahoma_sbatch", "bound_tahoma_sbatch"):
            script = case.get(key)
            if script:
                job_suffix = f"site{case.get('site') or index}_{'water' if 'water' in key else 'bound'}"
                tahoma_lines.append(
                    'sbatch --account="$ACCOUNT" --time="$TIME" --nodes="$GPU_NODES" '
                    '--gres="$GPU_GRES" -p "$PARTITION" --job-name="${JOB_NAME_PREFIX}_'
                    + job_suffix
                    + f'" {script}'
                )
    (batch_root / "submit_all_template.sh").write_text("\n".join(template_lines) + "\n", encoding="utf-8")
    (batch_root / "submit_all_tahoma.sh").write_text("\n".join(tahoma_lines) + "\n", encoding="utf-8")


def execute_free_energy_configs(
    result: FreeEnergyWizardBuildResult,
    *,
    dry_run: bool,
    failure_hint: str | None = None,
) -> int:
    from amber_metallo.free_energy.workflow import print_free_energy_workflow_summary, run_free_energy_workflow

    failures: list[tuple[FreeEnergyWorkflowConfig, Exception]] = []
    successful_results: list[dict[str, Any]] = []
    attempted_names: set[str] = set()
    succeeded_names: set[str] = set()
    total = len(result.configs)
    cleanup_map: dict[str, Path] = {}
    if result.mmpbsa_batch_plan is not None:
        cleanup_map = {
            str(case_plan.output_dir.resolve()): case_plan.cleanup_output_dir.resolve()
            for case_plan in result.mmpbsa_batch_plan.cases
            if case_plan.cleanup_output_dir is not None
        }

    for index, config in enumerate(result.configs, start=1):
        output_name = _free_energy_output_label(config.output_dir)
        attempted_names.add(output_name)
        if total > 1:
            console.print(
                Panel(
                    f"Starting output {index} of {total}: {output_name}",
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                )
            )
        cleanup_dir = cleanup_map.get(str(Path(config.output_dir).resolve()))
        if cleanup_dir is not None and cleanup_dir.exists():
            console.print(f"[dim]Removing existing MM-PBSA output before regeneration: {cleanup_dir}[/dim]")
            shutil.rmtree(cleanup_dir)
        try:
            execution_result = run_free_energy_workflow(config=config, dry_run=dry_run)
        except Exception as exc:
            failures.append((config, exc))
            console.print(f"[bold red]Free-energy workflow failed for {output_name}:[/bold red] {exc}")
            if failure_hint:
                console.print(f"[bold yellow]{failure_hint}[/bold yellow]")
            break
        successful_results.append(execution_result)
        succeeded_names.add(output_name)
        print_free_energy_workflow_summary(execution_result)

    if result.mmpbsa_batch_plan is not None and successful_results:
        failed_case_payload = [
            {
                "output_dir": str(config.output_dir),
                "error": str(exc),
            }
            for config, exc in failures
        ]
        try:
            _finalize_mmpbsa_batch_assets(
                batch_plan=result.mmpbsa_batch_plan,
                successful_results=successful_results,
                failed_cases=failed_case_payload,
            )
            console.print(
                f"[bold cyan]MM-PBSA aggregate assets generated under:[/bold cyan] {result.mmpbsa_batch_plan.batch_root}"
            )
        except Exception as exc:
            console.print(f"[bold red]Failed to generate MM-PBSA batch aggregate assets:[/bold red] {exc}")
            return 1

    if result.ti_batch_plan is not None and successful_results:
        failed_case_payload = [
            {
                "output_dir": str(config.output_dir),
                "error": str(exc),
            }
            for config, exc in failures
        ]
        try:
            _write_ti_batch_submission_assets(
                batch_plan=result.ti_batch_plan,
                successful_results=successful_results,
                failed_cases=failed_case_payload,
            )
            console.print(f"[bold cyan]TI batch submit assets generated under:[/bold cyan] {result.ti_batch_plan.batch_root}")
        except Exception as exc:
            console.print(f"[bold red]Failed to generate TI batch submit assets:[/bold red] {exc}")
            return 1

    if total > 1 or failures:
        table = Table(title="Free-energy batch summary", box=box.SIMPLE_HEAVY)
        table.add_column("Output", style="bold white")
        table.add_column("Status", style="cyan")
        failed_names = {_free_energy_output_label(config.output_dir): exc for config, exc in failures}
        for config in result.configs:
            name = _free_energy_output_label(config.output_dir)
            if name in failed_names:
                table.add_row(name, f"[bold red]Failed[/bold red] ({failed_names[name]})")
            elif name in succeeded_names:
                table.add_row(name, "[bold green]Succeeded[/bold green]")
            elif name in attempted_names:
                table.add_row(name, "[bold yellow]Stopped[/bold yellow]")
            else:
                table.add_row(name, "[dim]Not started[/dim]")
        console.print(table)

    return 0 if not failures else 1

