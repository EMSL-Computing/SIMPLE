from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import typer
from rich import box
from rich.table import Table

from amber_metallo.amber.leap import allowed_metal_charges
from amber_metallo.cli import (
    WizardChoice,
    _display_choice_table,
    _missing_required_1264_sets,
    _missing_required_126_sets,
    _prompt_choice,
    _prompt_execution_profile,
    _prompt_water_model_selection,
    _print_step_header,
)
from amber_metallo.config import SlurmProfile
from amber_metallo.environment import detect_amber_environment
from amber_metallo.reporting import console, print_notice
from amber_metallo.ti import abfe as ti_abfe
from amber_metallo.ti.analysis import (
    assess_site_stability,
    default_formal_charge,
    detect_bound_metal_sites,
    generate_reference_pdb_from_amber_restart,
    parse_cntrl_settings,
    run_last_snapshot_extraction,
    select_site,
)
from amber_metallo.ti.config import (
    ComplexInputConfig,
    MetalSelectionConfig,
    SnapshotConfig,
    SnapshotMode,
    TIDecouplingMode,
    TIImplementationMode,
    TIProtocolConfig,
    TIWorkflowConfig,
    WaterReferenceConfig,
    save_config,
)
from amber_metallo.ti.workflow import (
    water_reference_entry_dir,
    water_reference_entry_is_complete,
    water_reference_entry_matches,
    water_reference_root,
)
from amber_metallo.ti.topology import (
    filter_ti_compatible_custom_126_frcmods,
    filter_ti_compatible_custom_1264_frcmods,
    missing_required_1264_charge_families,
    missing_required_126_charge_families,
    resolve_1264_ion_frcmods,
    resolve_official_126_ion_frcmods,
)


_CNTRL_PAIR_PATTERN = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^,]+)")
_NSTEP_PATTERN = re.compile(r"\bNSTEP\s*=\s*(?P<nstep>\d+)", re.IGNORECASE)
_OUTPUT_TAIL_BYTES = 131072
_ACTIVE_STAGE_WINDOW_SECONDS = 600.0
_RAW_TI_TOPOLOGY_EXTS = {".prmtop", ".parm7", ".top"}
_RAW_TI_TRAJECTORY_EXTS = {".nc", ".mdcrd", ".crd", ".dcd", ".xtc", ".trr"}
_RAW_TI_EXTENSIONLESS_TRAJECTORY_NAMES = {"mdcrd"}
_RAW_TI_TRAJECTORY_TYPE_ORDER = (".nc", ".mdcrd", "mdcrd", ".crd", ".dcd", ".xtc", ".trr")
_RAW_TI_TRAJECTORY_REQUIREMENT_TEXT = "*.nc, *.mdcrd, mdcrd, *.crd, *.dcd, *.xtc, or *.trr"
_RAW_TI_REFERENCE_EXTS = {".pdb"}
_RAW_TI_MDIN_EXTS = {".in", ".mdin"}
_RAW_TI_RESTART_EXTS = {".rst7", ".rst", ".restrt", ".restart", ".ncrst", ".inpcrd"}
_RAW_TI_EXTENSIONLESS_RESTART_NAMES = {"rst", "restrt", "restart"}
_RAW_TI_SCAN_SKIP_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    ".simple_ti_wizard",
}
_RAW_TI_REQUIRED_FILES_MESSAGE = (
    "A raw TI input bundle needs at least an AMBER topology (*.prmtop, *.parm7, or *.top) "
    f"and a trajectory ({_RAW_TI_TRAJECTORY_REQUIREMENT_TEXT}). A reference PDB is required for metal-site "
    "detection; if it is not found automatically, the wizard will ask for it."
)


@dataclass(slots=True)
class WorkflowStageRun:
    index: int
    filename: str
    title: str
    writes_trajectory: bool
    input_path: Path
    output_path: Path
    restart_path: Path
    trajectory_path: Path | None
    target_nstlim: int | None
    dt_ps: float | None
    target_time_ns: float | None
    last_nstep: int | None
    progress_time_ns: float | None
    started: bool
    completed: bool
    latest_update_age_seconds: float | None = None

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


@dataclass(slots=True)
class WorkflowDiscovery:
    root: Path
    prmtop_path: Path | None
    reference_structure_path: Path | None
    production_stage: WorkflowStageRun | None
    selected_stage: WorkflowStageRun | None
    stages: list[WorkflowStageRun]
    selectable: bool
    readiness_note: str


@dataclass(slots=True)
class TIInputSelection:
    complex_input: ComplexInputConfig
    workflow_root: Path | None = None
    source_kind: str = "workflow"


@dataclass(slots=True)
class RawTIInputDiscovery:
    root: Path
    prmtop_path: Path
    trajectory_path: Path
    reference_structure_path: Path | None
    production_mdin_path: Path | None
    production_restart_path: Path | None
    trajectory_candidates: list[Path]
    production_mdin_candidates: list[Path]
    production_restart_candidates: list[Path]


@dataclass(slots=True)
class InferredWaterReferenceSettings:
    formal_charge: int | None = None
    water_model: str | None = None
    custom_ion_frcmods: list[str] | None = None


_WATER_SOURCE_PATTERN = re.compile(r"^\s*source\s+leaprc\.water\.(?P<water_model>[A-Za-z0-9_+-]+)\s*$", re.IGNORECASE)
_LOAD_AMBER_PARAMS_PATTERN = re.compile(r"^\s*loadAmberParams\s+(?P<path>\S+)\s*$", re.IGNORECASE)
_METAL_LABEL_CHARGE_MAP: dict[tuple[str, str], int] = {
    ("CO", "CO"): 2,
    ("CU", "CU1"): 1,
    ("CU", "CU"): 2,
    ("NI", "NI"): 2,
    ("MN", "MN"): 2,
    ("FE", "FE2"): 2,
    ("FE", "FE"): 3,
    ("Y", "Y"): 3,
    ("LA", "LA"): 3,
    ("ND", "ND"): 3,
    ("EU", "EU3"): 3,
    ("LU", "LU"): 3,
}


def _prompt_existing_path(message: str, *, optional: bool = False) -> str | None:
    while True:
        raw = (
            typer.prompt(message, default="").strip()
            if optional
            else typer.prompt(message).strip()
        )
        if optional and not raw:
            return None
        path = Path(raw).expanduser()
        if path.exists():
            return str(path.resolve())
        console.print(f"[bold red]Path not found:[/bold red] {raw}")


def _display_site_table(candidates, assessments) -> None:
    table = Table(title="Detected bound metal candidates", box=box.SIMPLE_HEAVY)
    table.add_column("Site", style="bold cyan", justify="right")
    table.add_column("Metal", style="bold white")
    table.add_column("Reference donors", style="white", justify="right")
    table.add_column("C4", style="cyan", justify="center")
    table.add_column("Last snapshot", style="white")
    for candidate in candidates:
        assessment = next(item for item in assessments if item.site == candidate.site)
        c4_text = "Yes" if candidate.c4_supported else ("No" if candidate.c4_supported is False else "Unknown")
        status = "Stable" if assessment.stable else "Unstable"
        table.add_row(
            str(candidate.site),
            f"{candidate.element} at {candidate.key}",
            str(candidate.donor_count),
            c4_text,
            f"{status} ({assessment.displacement_angstrom:.2f} A local shift)",
        )
    console.print(table)


def _default_output_directory(reference_structure_path: str, *, workflow_root: Path | None = None) -> Path:
    stem = workflow_root.name if workflow_root is not None else Path(reference_structure_path).expanduser().stem
    base = Path.cwd() / f"{stem}_ti"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = Path.cwd() / f"{stem}_ti_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _prompt_snapshot_mode(*, selected_stable: bool) -> SnapshotMode:
    if not selected_stable:
        console.print(
            "[bold yellow]Cluster analysis is disabled because the selected metal site is unstable in the last snapshot.[/bold yellow]"
        )
        return SnapshotMode.LAST

    choices = [
        WizardChoice(SnapshotMode.LAST.value, "Last snapshot", "Use the final frame from the production trajectory."),
        WizardChoice(
            SnapshotMode.CLUSTER.value,
            "Representative cluster snapshot",
            "Run cluster analysis on the binding-site neighborhood to choose a representative snapshot. This can take time.",
        ),
    ]
    _display_choice_table("Snapshot source", choices)
    return SnapshotMode(_prompt_choice("Choose the snapshot source", choices, default_key=SnapshotMode.LAST.value))


def _prompt_ti_implementation_mode() -> TIImplementationMode:
    choices = [
        WizardChoice(
            TIImplementationMode.AMBER_12_6_4_GTI.value,
            "Amber TI 12-6-4 (GTI/CUDA)",
            "Default. Keep the original 12-6-4 C4 terms and run the GTI path with GPU/pmemd.cuda.",
        ),
        WizardChoice(
            TIImplementationMode.AMBER_12_6_WORKAROUND.value,
            "Amber TI official 12-6 rebuild",
            "Legacy alternative. Rebuild TI-specific ion/metal non-bonded terms to the official Amber 12-6 set.",
        ),
        WizardChoice(
            TIImplementationMode.GROMACS_TABULATED_12_6_4.value,
            "GROMACS 12-6-4 (tabulated)",
            "Under investigation. Would require Amber-to-GROMACS conversion plus custom tabulated 12-6-4 non-bonded tables for TI.",
            enabled=False,
        ),
    ]
    _display_choice_table("TI implementation mode", choices)
    selection = TIImplementationMode(
        _prompt_choice(
            "Choose the TI implementation mode",
            choices,
            default_key=TIImplementationMode.AMBER_12_6_4_GTI.value,
        )
    )
    if selection == TIImplementationMode.AMBER_12_6_4_GTI:
        notice = (
            "SIMPLE will keep the original [bold]main.py[/bold] workflow untouched and generate "
            "[bold cyan]TI-specific prmtops[/bold cyan] that preserve the [bold red]Amber 12-6-4 C4[/bold red] "
            "terms. The generated TI scripts use [bold]pmemd.cuda[/bold] for the CUDA/GTI path; the Tahoma script "
            "keeps the current emsl62113 SBATCH header so you can edit the cluster resources manually."
        )
    else:
        notice = (
            "SIMPLE will keep the original [bold]main.py[/bold] workflow untouched and generate "
            "[bold cyan]TI-specific prmtops[/bold cyan] for the selected workflow. In this mode, "
            "SIMPLE rebuilds the metal/ion non-bonded terms to the official [bold red]Amber 12-6[/bold red] "
            "set that matches the selected water model before the TI windows are launched. Charge-off windows use a "
            "conservative timestep and denser endpoint schedule, and SIMPLE inserts a short "
            "[bold]qoff-endpoint relaxation[/bold] before the VDW-off leg."
        )
    print_notice("Selected TI Mode", notice, border_style="cyan")
    return selection


def _prompt_ti_decoupling_mode(implementation_mode: TIImplementationMode) -> TIDecouplingMode:
    if implementation_mode != TIImplementationMode.AMBER_12_6_4_GTI:
        return TIDecouplingMode.SPLIT_Q_VDW

    choices = [
        WizardChoice(
            TIDecouplingMode.SPLIT_Q_VDW.value,
            "Split Q then VDW (coming soon)",
            "Temporarily unavailable while the split-path error is being resolved.",
            enabled=False,
        ),
        WizardChoice(
            TIDecouplingMode.COMBINED_Q_VDW.value,
            "Combined softcore",
            "Default and currently supported. Decouple charge and VDW together in one CUDA/GTI softcore path.",
        ),
    ]
    _display_choice_table("TI decoupling path", choices)
    selection = TIDecouplingMode(
        _prompt_choice(
            "Choose the TI decoupling path",
            choices,
            default_key=TIDecouplingMode.COMBINED_Q_VDW.value,
        )
    )
    if selection == TIDecouplingMode.SPLIT_Q_VDW:
        notice = (
            "The CUDA/GTI run will keep the traditional two-leg protocol: Q-off windows first, then a short "
            "decharged-endpoint relaxation, then softcore VDW-off windows. The non-softcore Q-off input keeps "
            "timask1/timask2 atom counts matched as required by pmemd."
        )
    else:
        notice = (
            "The CUDA/GTI run will use one softcore decoupling path for charge and VDW together. This is useful for "
            "testing, but it changes the decomposition relative to the default Q-off then VDW-off protocol."
        )
    print_notice("Selected TI Decoupling", notice, border_style="cyan")
    return selection


def _prompt_ti_execution_profile(implementation_mode: TIImplementationMode) -> SlurmProfile:
    if implementation_mode != TIImplementationMode.AMBER_12_6_4_GTI:
        return _prompt_execution_profile()
    choices = [
        WizardChoice(
            SlurmProfile.GPU.value,
            "GPU / pmemd.cuda (required)",
            "GTI/CUDA requires pmemd.cuda; generated master and Tahoma scripts will target GPU execution.",
        ),
        WizardChoice(
            SlurmProfile.CPU.value,
            "CPU / pmemd.MPI (unavailable for GTI)",
            "GTI/CUDA cannot run with the CPU pmemd.MPI path.",
            enabled=False,
        ),
    ]
    print_notice(
        "GTI GPU Requirement",
        "Amber 12-6-4 GTI must run with [bold]GPU/pmemd.cuda[/bold]. GPU is selected by default; "
        "the CPU profile is unavailable for this mode.",
        border_style="yellow",
    )
    _display_choice_table("GTI execution target", choices)
    return SlurmProfile(
        _prompt_choice(
            "Choose the GTI execution target",
            choices,
            default_key=SlurmProfile.GPU.value,
        )
    )


def _prompt_formal_charge(element: str) -> int:
    return _prompt_formal_charge_with_default(element)


def _prompt_formal_charge_with_default(element: str, *, default_charge: int | None = None) -> int:
    allowed = allowed_metal_charges(element)
    if allowed:
        resolved_default = default_charge if default_charge in allowed else default_formal_charge(element)
        choices = [
            WizardChoice(str(charge), f"+{charge}", f"Use the supported {element} 12-6-4 charge state +{charge}.")
            for charge in allowed
        ]
        _display_choice_table(f"{element} oxidation-state options", choices)
        return int(_prompt_choice("Choose the metal oxidation state", choices, default_key=str(resolved_default)))
    return typer.prompt("Metal formal charge", default=default_charge or default_formal_charge(element), type=int)


def _infer_charge_from_selected_site(candidate) -> int | None:
    element_key = candidate.element.strip().upper()
    residue_key = candidate.residue_name.strip().upper()
    inferred = _METAL_LABEL_CHARGE_MAP.get((element_key, residue_key))
    if inferred is not None:
        return inferred
    allowed = allowed_metal_charges(candidate.element)
    if len(allowed) == 1:
        return int(allowed[0])
    return None


def _infer_workflow_water_settings(workflow_root: Path) -> tuple[str | None, list[str]]:
    tleap_path = workflow_root / "02_system" / "tleap.in"
    if not tleap_path.exists():
        return None, []

    water_model: str | None = None
    custom_ion_frcmods: list[str] = []
    for raw_line in tleap_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        water_match = _WATER_SOURCE_PATTERN.match(line)
        if water_match:
            water_model = water_match.group("water_model").strip().lower()
            continue
        frcmod_match = _LOAD_AMBER_PARAMS_PATTERN.match(line)
        if frcmod_match:
            frcmod_path = frcmod_match.group("path").strip()
            frcmod_name = Path(frcmod_path).name.lower()
            if frcmod_name.startswith("frcmod.ions") and frcmod_path not in custom_ion_frcmods:
                custom_ion_frcmods.append(frcmod_path)
    return water_model, custom_ion_frcmods


def _infer_water_reference_settings(
    workflow_root: Path | None,
    selected,
    *,
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
) -> InferredWaterReferenceSettings:
    if workflow_root is None:
        return InferredWaterReferenceSettings()
    formal_charge = _infer_charge_from_selected_site(selected)
    water_model, custom_ion_frcmods = _infer_workflow_water_settings(workflow_root)
    if implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI:
        ti_compatible_custom_frcmods = [str(path) for path in filter_ti_compatible_custom_1264_frcmods(custom_ion_frcmods)]
    else:
        ti_compatible_custom_frcmods = [str(path) for path in filter_ti_compatible_custom_126_frcmods(custom_ion_frcmods)]
    return InferredWaterReferenceSettings(
        formal_charge=formal_charge,
        water_model=water_model,
        custom_ion_frcmods=ti_compatible_custom_frcmods,
    )


def _validate_ti_water_model_support(
    *,
    amber_env,
    water_model: str,
    formal_charge: int,
    custom_ion_frcmods: list[str],
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
) -> None:
    if implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI:
        resolved_frcmods = resolve_1264_ion_frcmods(
            amber_env=amber_env,
            water_model=water_model,
            custom_ion_frcmods=custom_ion_frcmods,
        )
        missing = missing_required_1264_charge_families(resolved_frcmods, formal_charge=formal_charge)
        if not resolved_frcmods and not missing:
            missing = _missing_required_1264_sets(
                amber_env,
                water_model,
                include_monovalent=formal_charge == 1,
                include_multivalent=formal_charge >= 2,
            )
        if not missing:
            return
        raise ValueError(
            f"The inferred water model '{water_model}' does not have an Amber 12-6-4 ion family for the selected "
            f"TI metal charge state (+{formal_charge}). Choose a 12-6-4-capable water model or provide explicit "
            "custom 12-6-4 ion frcmods."
        )

    resolved_frcmods = resolve_official_126_ion_frcmods(
        amber_env=amber_env,
        water_model=water_model,
        custom_ion_frcmods=custom_ion_frcmods,
    )
    missing = missing_required_126_charge_families(resolved_frcmods, formal_charge=formal_charge)
    if not resolved_frcmods and not missing:
        missing = _missing_required_126_sets(
            amber_env,
            water_model,
            include_monovalent=formal_charge == 1,
            include_multivalent=formal_charge >= 2,
        )
    if not missing:
        return
    raise ValueError(
        f"The inferred water model '{water_model}' does not have an official Amber 12-6 ion family for the selected "
        f"TI metal charge state (+{formal_charge}). Choose one of the recommended metal/TI water models or provide "
        "explicit custom official 12-6 ion frcmods."
    )


def _maybe_print_legacy_126_warning(*, water_model: str, formal_charge: int) -> None:
    if formal_charge < 3 or water_model.lower() not in {"tip3p", "tip4pew"}:
        return
    print_notice(
        "Amber Version Note",
        "If you are using Amber 2016-2019 with the official multivalent 12-6 frcmods for this water model, "
        "please use the corrected official frcmod files manually before running TI. SIMPLE does not patch "
        "those legacy files automatically.",
        border_style="yellow",
    )


def _display_water_reference_notice(
    *,
    selected,
    inferred: InferredWaterReferenceSettings | None,
    workflow_root: Path | None,
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
) -> None:
    if workflow_root is not None and inferred and inferred.formal_charge is not None and inferred.water_model is not None:
        ion_files_text = ""
        if inferred.custom_ion_frcmods:
            ion_family = "12-6-4" if implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI else "official 12-6"
            ion_files_text = (
                f" TI-compatible custom {ion_family} ion parameter files were also detected in the main workflow "
                f"({len(inferred.custom_ion_frcmods)} file(s)) and will be reused."
            )
        print_notice(
            "DeltaG Reference Leg",
            "The next settings control the metal-in-water reference simulation used for the DeltaG reference state, "
            "not the protein complex leg you already selected.\n\n"
            f"Because the selected main.py workflow used [bold cyan]{selected.element}{inferred.formal_charge}+[/bold cyan] "
            f"with [bold green]{inferred.water_model.upper()}[/bold green], SIMPLE will default to the same oxidation "
            "state and water model for the reference leg. This is normally the correct choice for the TI cycle. "
            "The reusable solvent-reference setup will be written under [bold]water_ref[/bold], and a short water-reference "
            "pre-equilibration will run before the TI windows are started." + ion_files_text,
            border_style="cyan",
        )
        return

    print_notice(
        "DeltaG Reference Leg",
        "The next settings control the metal-in-water reference simulation used for the DeltaG reference state, "
        "not the protein complex leg you already selected.\n\n"
        "For this TI cycle, the reference leg should normally use the same oxidation state and water model as the "
        "complex-leg setup unless you intentionally want to change the thermodynamic state. SIMPLE will place "
        "that reusable solvent-reference setup under [bold]water_ref[/bold] and pre-equilibrate it before the TI windows.",
        border_style="cyan",
    )


def _resolve_water_reference_settings(
    *,
    amber_env,
    selected,
    workflow_root: Path | None,
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
) -> tuple[int, str, list[str]]:
    inferred = _infer_water_reference_settings(
        workflow_root,
        selected,
        implementation_mode=implementation_mode,
    )
    _display_water_reference_notice(
        selected=selected,
        inferred=inferred,
        workflow_root=workflow_root,
        implementation_mode=implementation_mode,
    )
    require_official_126 = implementation_mode == TIImplementationMode.AMBER_12_6_WORKAROUND

    if inferred.formal_charge is not None and inferred.water_model is not None:
        override = typer.confirm(
            "Override the inferred water-reference oxidation state or water model?",
            default=False,
        )
        if not override:
            custom_ion_frcmods = inferred.custom_ion_frcmods or []
            _validate_ti_water_model_support(
                amber_env=amber_env,
                water_model=inferred.water_model,
                formal_charge=inferred.formal_charge,
                custom_ion_frcmods=custom_ion_frcmods,
                implementation_mode=implementation_mode,
            )
            _maybe_print_legacy_126_warning(
                water_model=inferred.water_model,
                formal_charge=inferred.formal_charge,
            )
            return inferred.formal_charge, inferred.water_model, custom_ion_frcmods
        console.print(
            f"[dim]The selected main.py workflow used {selected.element}{inferred.formal_charge}+ with "
            f"{inferred.water_model.upper()} for the reference-compatible setup. Choose different settings only if "
            "you intentionally want a different water-reference state.[/dim]"
        )

    if inferred.formal_charge is not None:
        formal_charge = inferred.formal_charge
        console.print(f"[dim]Defaulting the reference-leg oxidation state to +{formal_charge} from the selected workflow.[/dim]")
        if typer.confirm("Choose a different oxidation state?", default=False):
            formal_charge = _prompt_formal_charge_with_default(selected.element, default_charge=inferred.formal_charge)
    else:
        formal_charge = _prompt_formal_charge(selected.element)

    inferred_water_model = inferred.water_model
    inferred_frcmods = inferred.custom_ion_frcmods or []
    if inferred_water_model is not None:
        console.print(
            f"[dim]Defaulting the water-reference model to {inferred_water_model.upper()} from the selected workflow.[/dim]"
        )
        if typer.confirm("Choose a different water model?", default=False):
            water_model, custom_ion_frcmods = _prompt_water_model_selection(
                amber_env,
                include_monovalent=formal_charge == 1,
                include_multivalent=formal_charge >= 2,
                require_official_126=require_official_126,
            )
        else:
            water_model, custom_ion_frcmods = inferred_water_model, inferred_frcmods
    else:
        water_model, custom_ion_frcmods = _prompt_water_model_selection(
            amber_env,
            include_monovalent=formal_charge == 1,
            include_multivalent=formal_charge >= 2,
            require_official_126=require_official_126,
        )
    _validate_ti_water_model_support(
        amber_env=amber_env,
        water_model=water_model,
        formal_charge=formal_charge,
        custom_ion_frcmods=custom_ion_frcmods,
        implementation_mode=implementation_mode,
    )
    _maybe_print_legacy_126_warning(
        water_model=water_model,
        formal_charge=formal_charge,
    )
    return formal_charge, water_model, custom_ion_frcmods


def _probe_water_reference_reuse(
    *,
    complex_input: ComplexInputConfig,
    selected,
    formal_charge: int,
    water_model: str,
    custom_ion_frcmods: list[str],
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
    decoupling_mode: TIDecouplingMode = TIDecouplingMode.SPLIT_Q_VDW,
) -> tuple[Path, Path, bool]:
    amber_env = detect_amber_environment()
    if implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI:
        ti_ion_frcmods = [
            str(path)
            for path in resolve_1264_ion_frcmods(
                amber_env=amber_env,
                water_model=water_model,
                custom_ion_frcmods=custom_ion_frcmods,
            )
        ]
    else:
        ti_ion_frcmods = [
            str(path)
            for path in resolve_official_126_ion_frcmods(
                amber_env=amber_env,
                water_model=water_model,
                custom_ion_frcmods=custom_ion_frcmods,
            )
        ]
    probe_config = TIWorkflowConfig(
        complex_input=complex_input,
        metal=MetalSelectionConfig(
            selected_site=selected.site,
            formal_charge=formal_charge,
        ),
        ti=TIProtocolConfig(
            implementation_mode=implementation_mode,
            decoupling_mode=decoupling_mode,
        ),
        water_reference=WaterReferenceConfig(
            water_model=water_model,
            custom_ion_frcmods=custom_ion_frcmods,
        ),
        output_dir=".",
    )
    inherited_settings = parse_cntrl_settings(complex_input.production_mdin_path)
    root = water_reference_root(probe_config)
    entry_dir = water_reference_entry_dir(
        probe_config,
        metal_element=selected.element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=ti_ion_frcmods,
    )
    reusable = water_reference_entry_is_complete(entry_dir) and water_reference_entry_matches(
        probe_config,
        entry_dir=entry_dir,
        metal_element=selected.element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=ti_ion_frcmods,
    )
    return root, entry_dir, reusable


def _resolve_water_reference_reuse_choice(
    *,
    complex_input: ComplexInputConfig,
    selected,
    formal_charge: int,
    water_model: str,
    custom_ion_frcmods: list[str],
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
    decoupling_mode: TIDecouplingMode = TIDecouplingMode.SPLIT_Q_VDW,
) -> bool:
    root_dir, entry_dir, reusable = _probe_water_reference_reuse(
        complex_input=complex_input,
        selected=selected,
        formal_charge=formal_charge,
        water_model=water_model,
        custom_ion_frcmods=custom_ion_frcmods,
        implementation_mode=implementation_mode,
        decoupling_mode=decoupling_mode,
    )
    if reusable:
        print_notice(
            "Reusable Water Reference Found",
            "A matching pre-equilibrated water-reference directory is already available for this "
            f"DeltaG reference state:\n\n[bold]{entry_dir}[/bold]\n\n"
            "If you reuse it, SIMPLE will keep using that equilibrated solvent-reference endpoint instead of "
            "rebuilding it from scratch.",
            border_style="green",
        )
        return typer.confirm("Reuse this existing water-reference directory?", default=True)

    if entry_dir.exists():
        console.print(
            f"[dim]A water-reference directory already exists at {entry_dir}, but it is either incomplete or does not "
            "exactly match the current TI settings. SIMPLE will refresh that setup there and generate the "
            "water-reference equilibration plus TI inputs again.[/dim]"
        )
    else:
        console.print(
            f"[dim]No matching reusable water-reference directory was found under {root_dir}. SIMPLE will create "
            f"{entry_dir.name} there, generate the water-reference pre-equilibration inputs, and then wire the TI "
            "windows to start from the equilibrated solvent endpoint.[/dim]"
        )
    return True


def _resolve_water_reference_source_choice(
    *,
    complex_input: ComplexInputConfig,
    selected,
    formal_charge: int,
    water_model: str,
    custom_ion_frcmods: list[str],
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND,
    decoupling_mode: TIDecouplingMode = TIDecouplingMode.SPLIT_Q_VDW,
) -> tuple[bool, bool, str | None]:
    library_entry = ti_abfe.lookup_water_library_entry(selected.element, formal_charge, water_model)
    if library_entry:
        aggregate = library_entry.get("aggregate") or {}
        total = aggregate.get("total") or {}
        ci95 = total.get("bootstrap_ci95") or {}
        print_notice(
            "Water-Reference Library Match",
            f"[bold cyan]{selected.element}{formal_charge}+[/bold cyan] in [bold green]{water_model.upper()}[/bold green] "
            "already exists in the shared analysis library.\n\n"
            f"Library mean: [bold]{float(total.get('delta_g_kcal_mol', 0.0)):.6f} +/- "
            f"{float(total.get('propagated_sem_kcal_mol', 0.0)):.6f} kcal/mol[/bold]\n"
            f"95% CI: [{float(ci95.get('low', 0.0)):.6f}, {float(ci95.get('high', 0.0)):.6f}]\n"
            f"Contributing cases: {int(aggregate.get('n_cases', 0))}\n\n"
            "If you reuse the library value, SIMPLE will skip generating a fresh water-reference leg and store "
            "the exact library snapshot in the TI manifest for reproducibility.",
            border_style="green",
        )
        if typer.confirm("Reuse this library value instead of preparing a fresh water-reference simulation?", default=True):
            return False, True, str(library_entry.get("key") or "")
    reuse_existing = _resolve_water_reference_reuse_choice(
        complex_input=complex_input,
        selected=selected,
        formal_charge=formal_charge,
        water_model=water_model,
        custom_ion_frcmods=custom_ion_frcmods,
        implementation_mode=implementation_mode,
        decoupling_mode=decoupling_mode,
    )
    return reuse_existing, False, None


def _parse_cntrl_values(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    in_cntrl = False
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
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
    return values


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    return float(raw.replace("d", "e").replace("D", "E"))


def _parse_stage_timing(path: Path) -> tuple[int | None, float | None, float | None]:
    if not path.exists():
        return None, None, None
    values = _parse_cntrl_values(path)
    nstlim_value = values.get("nstlim")
    if nstlim_value is None:
        return None, None, None
    nstlim = int(float(nstlim_value))
    dt_ps = _parse_float(values.get("dt")) or 0.002
    return nstlim, dt_ps, (nstlim * dt_ps) / 1000.0


def _read_text_tail(path: Path, *, max_bytes: int = _OUTPUT_TAIL_BYTES) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        handle.seek(max(0, file_size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="ignore")


def _parse_output_progress(path: Path, *, dt_ps: float | None) -> tuple[int | None, float | None]:
    if not path.exists():
        return None, None
    text = _read_text_tail(path)
    last_nstep: int | None = None
    for match in _NSTEP_PATTERN.finditer(text):
        last_nstep = int(match.group("nstep"))
    if last_nstep is None or dt_ps is None:
        return last_nstep, None
    return last_nstep, (last_nstep * dt_ps) / 1000.0


def _latest_stage_activity_age_seconds(*paths: Path | None) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path is not None and path.exists()]
    if not mtimes:
        return None
    return max(0.0, time.time() - max(mtimes))


def _load_stage_definitions(inputs_dir: Path) -> list[dict[str, Any]]:
    manifest_path = inputs_dir / "md_manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = data.get("stages")
        if isinstance(stages, list):
            return [item for item in stages if isinstance(item, dict) and item.get("filename")]

    definitions: list[dict[str, Any]] = []
    for input_path in sorted(inputs_dir.glob("*.in")):
        definitions.append(
            {
                "filename": input_path.name,
                "title": input_path.stem.replace("_", " ").title(),
                "writes_trajectory": "min" not in input_path.stem.lower(),
            }
        )
    return definitions


def _build_stage_run(index: int, spec: dict[str, Any], *, inputs_dir: Path, outputs_dir: Path) -> WorkflowStageRun | None:
    filename = str(spec.get("filename") or "").strip()
    if not filename:
        return None

    input_path = inputs_dir / filename
    if not input_path.exists():
        return None

    title = str(spec.get("title") or Path(filename).stem.replace("_", " ").title())
    writes_trajectory = bool(spec.get("writes_trajectory", "min" not in filename.lower()))
    stem = input_path.stem
    output_path = outputs_dir / f"{stem}.out"
    restart_path = outputs_dir / f"{stem}.rst7"
    trajectory_path = outputs_dir / f"{stem}.nc" if writes_trajectory else None
    target_nstlim, dt_ps, target_time_ns = _parse_stage_timing(input_path)
    last_nstep, progress_time_ns = _parse_output_progress(output_path, dt_ps=dt_ps)
    started = output_path.exists() or restart_path.exists() or (trajectory_path is not None and trajectory_path.exists())
    completed = bool(target_nstlim is not None and last_nstep is not None and last_nstep >= target_nstlim)
    latest_update_age_seconds = _latest_stage_activity_age_seconds(output_path, restart_path, trajectory_path)
    return WorkflowStageRun(
        index=index,
        filename=filename,
        title=title,
        writes_trajectory=writes_trajectory,
        input_path=input_path,
        output_path=output_path,
        restart_path=restart_path,
        trajectory_path=trajectory_path,
        target_nstlim=target_nstlim,
        dt_ps=dt_ps,
        target_time_ns=target_time_ns,
        last_nstep=last_nstep,
        progress_time_ns=progress_time_ns,
        started=started,
        completed=completed,
        latest_update_age_seconds=latest_update_age_seconds,
    )


def _looks_like_main_workflow_root(root: Path) -> bool:
    return (
        (root / "workflow_manifest.json").exists()
        or (root / "02_system").exists()
        or (root / "03_md").exists()
    )


def _raw_ti_path_type_key(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix:
        return suffix
    name = path.name.lower()
    if name in _RAW_TI_EXTENSIONLESS_TRAJECTORY_NAMES or name in _RAW_TI_EXTENSIONLESS_RESTART_NAMES:
        return name
    return ""


def _is_raw_ti_trajectory_file(path: Path) -> bool:
    type_key = _raw_ti_path_type_key(path)
    return type_key in _RAW_TI_TRAJECTORY_EXTS or type_key in _RAW_TI_EXTENSIONLESS_TRAJECTORY_NAMES


def _is_raw_ti_restart_file(path: Path) -> bool:
    type_key = _raw_ti_path_type_key(path)
    return type_key in _RAW_TI_RESTART_EXTS or type_key in _RAW_TI_EXTENSIONLESS_RESTART_NAMES


def _is_raw_ti_scan_skipped_dir(path: Path) -> bool:
    return path.name in _RAW_TI_SCAN_SKIP_NAMES


def _iter_raw_ti_scan_files(search_root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        current = Path(dirpath)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_raw_ti_scan_skipped_dir(current / name)
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


def _format_raw_ti_path(path: Path | None, *, search_root: Path | None) -> str:
    if path is None:
        return "-"
    if search_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(search_root.resolve()))
    except ValueError:
        return str(path)


def _preferred_raw_ti_file(paths: list[Path], *, keywords: tuple[str, ...], exts: tuple[str, ...] = ()) -> Path | None:
    if not paths:
        return None

    def score(path: Path) -> tuple[int, int, int, str]:
        text = "/".join(part.lower() for part in path.parts)
        keyword_score = next((index for index, keyword in enumerate(keywords) if keyword in text), len(keywords))
        type_key = _raw_ti_path_type_key(path)
        ext_score = next((index for index, ext in enumerate(exts) if type_key == ext), len(exts))
        return (keyword_score, ext_score, len(path.parts), path.name.lower())

    return sorted(paths, key=score)[0]


def _raw_ti_candidate_sort_key(path: Path) -> tuple[int, int, str]:
    text = "/".join(part.lower() for part in path.parts)
    stage_order = (
        "prod",
        "production",
        "md",
        "equil",
        "eq",
        "npt",
        "nvt",
        "heat",
        "min",
    )
    stage_score = next((index for index, token in enumerate(stage_order) if token in text), len(stage_order))
    ext_order = (*_RAW_TI_TRAJECTORY_TYPE_ORDER, ".rst7", ".rst", ".restrt", ".restart", ".ncrst", ".inpcrd")
    type_key = _raw_ti_path_type_key(path)
    ext_score = next((index for index, ext in enumerate(ext_order) if type_key == ext), len(ext_order))
    return (stage_score, ext_score, path.name.lower())


def _matching_raw_ti_mdin_for_trajectory(trajectory: Path, mdins: list[Path]) -> Path | None:
    stem = trajectory.stem.lower()
    same_stem = [path for path in mdins if path.stem.lower() == stem]
    if same_stem:
        return sorted(same_stem, key=lambda item: item.name.lower())[0]
    return _preferred_raw_ti_file(mdins, keywords=("prod", "production", "md", "eq"))


def _matching_raw_ti_restart_for_trajectory(trajectory: Path, restarts: list[Path]) -> Path | None:
    stem = trajectory.stem.lower()
    same_stem = [path for path in restarts if path.stem.lower() == stem]
    if same_stem:
        return sorted(same_stem, key=lambda item: item.name.lower())[0]
    return _preferred_raw_ti_file(
        restarts,
        keywords=("prod", "production", "md", "eq", "restart", "rst"),
        exts=(".rst7", ".rst", ".restrt", ".restart", "restart", "rst", "restrt", ".ncrst", ".inpcrd"),
    )


def _raw_ti_candidate_roots_for_topology(topology: Path, *, search_root: Path) -> list[Path]:
    roots: list[Path] = []
    parent = topology.parent
    roots.append(parent)
    if parent.name.lower() in {
        "02_system",
        "system",
        "systems",
        "topology",
        "topologies",
        "input",
        "inputs",
        "prep",
        "structure",
        "structures",
    }:
        roots.append(parent.parent)
    roots.append(search_root)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _discover_raw_ti_inputs_in_directory(search_root: Path) -> list[RawTIInputDiscovery]:
    root = search_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    files = _iter_raw_ti_scan_files(root)
    topologies = sorted(
        [path for path in files if path.suffix.lower() in _RAW_TI_TOPOLOGY_EXTS],
        key=lambda item: (len(item.parts), item.name.lower()),
    )
    trajectories = [path for path in files if _is_raw_ti_trajectory_file(path)]
    references = [path for path in files if path.suffix.lower() in _RAW_TI_REFERENCE_EXTS]
    mdins = [path for path in files if path.suffix.lower() in _RAW_TI_MDIN_EXTS]
    restarts = [path for path in files if _is_raw_ti_restart_file(path)]
    discoveries: list[RawTIInputDiscovery] = []
    seen: set[tuple[Path, Path]] = set()

    for topology in topologies:
        selected_root: Path | None = None
        nearby_trajectories: list[Path] = []
        for candidate_root in _raw_ti_candidate_roots_for_topology(topology, search_root=root):
            candidate_trajectories = sorted(
                [path for path in trajectories if _is_relative_to(path, candidate_root)],
                key=_raw_ti_candidate_sort_key,
            )
            if candidate_trajectories:
                selected_root = candidate_root
                nearby_trajectories = candidate_trajectories
                break
        if selected_root is None or not nearby_trajectories:
            continue

        trajectory = _preferred_raw_ti_file(
            nearby_trajectories,
            keywords=("prod", "production", "md", "eq"),
            exts=_RAW_TI_TRAJECTORY_TYPE_ORDER,
        )
        if trajectory is None:
            continue

        nearby_references = [path for path in references if _is_relative_to(path, selected_root)]
        nearby_mdins = [path for path in mdins if _is_relative_to(path, selected_root)]
        nearby_restarts = sorted(
            [path for path in restarts if _is_relative_to(path, selected_root)],
            key=_raw_ti_candidate_sort_key,
        )
        reference = _preferred_raw_ti_file(
            nearby_references,
            keywords=("system", "complex", "reference", "ref", "cleaned", "input", "prod"),
        )
        mdin = _matching_raw_ti_mdin_for_trajectory(trajectory, nearby_mdins)
        restart = _matching_raw_ti_restart_for_trajectory(trajectory, nearby_restarts)
        key = (selected_root.resolve(), topology.resolve())
        if key in seen:
            continue
        seen.add(key)
        discoveries.append(
            RawTIInputDiscovery(
                root=selected_root,
                prmtop_path=topology.resolve(),
                trajectory_path=trajectory.resolve(),
                reference_structure_path=None if reference is None else reference.resolve(),
                production_mdin_path=None if mdin is None else mdin.resolve(),
                production_restart_path=None if restart is None else restart.resolve(),
                trajectory_candidates=[path.resolve() for path in nearby_trajectories],
                production_mdin_candidates=[path.resolve() for path in nearby_mdins],
                production_restart_candidates=[path.resolve() for path in nearby_restarts],
            )
        )
    return discoveries[:50]


def _raw_ti_search_roots_for_path(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        return [resolved]
    roots: list[Path] = []
    current = resolved.parent
    for parent in [current, *current.parents]:
        roots.append(parent.resolve())
        if len(roots) >= 4:
            break
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def _raw_ti_discovery_mentions_path(discovery: RawTIInputDiscovery, path: Path) -> bool:
    resolved = path.expanduser().resolve()
    paths = [
        discovery.prmtop_path,
        discovery.trajectory_path,
        discovery.reference_structure_path,
        discovery.production_mdin_path,
        discovery.production_restart_path,
        *discovery.trajectory_candidates,
        *discovery.production_mdin_candidates,
        *discovery.production_restart_candidates,
    ]
    return any(item is not None and item.resolve() == resolved for item in paths)


def _with_selected_raw_ti_trajectory(discovery: RawTIInputDiscovery, trajectory_path: Path) -> RawTIInputDiscovery:
    trajectory = trajectory_path.expanduser().resolve()
    mdin = _matching_raw_ti_mdin_for_trajectory(trajectory, discovery.production_mdin_candidates)
    restart = _matching_raw_ti_restart_for_trajectory(trajectory, discovery.production_restart_candidates)
    return RawTIInputDiscovery(
        root=discovery.root,
        prmtop_path=discovery.prmtop_path,
        trajectory_path=trajectory,
        reference_structure_path=discovery.reference_structure_path,
        production_mdin_path=mdin,
        production_restart_path=restart,
        trajectory_candidates=discovery.trajectory_candidates,
        production_mdin_candidates=discovery.production_mdin_candidates,
        production_restart_candidates=discovery.production_restart_candidates,
    )


def _focus_raw_ti_discovery_on_path(discovery: RawTIInputDiscovery, path: Path) -> RawTIInputDiscovery:
    resolved = path.expanduser().resolve()
    if _is_raw_ti_trajectory_file(resolved) and any(candidate.resolve() == resolved for candidate in discovery.trajectory_candidates):
        return _with_selected_raw_ti_trajectory(discovery, resolved)
    if resolved.suffix.lower() in _RAW_TI_REFERENCE_EXTS:
        return RawTIInputDiscovery(
            root=discovery.root,
            prmtop_path=discovery.prmtop_path,
            trajectory_path=discovery.trajectory_path,
            reference_structure_path=resolved,
            production_mdin_path=discovery.production_mdin_path,
            production_restart_path=discovery.production_restart_path,
            trajectory_candidates=discovery.trajectory_candidates,
            production_mdin_candidates=discovery.production_mdin_candidates,
            production_restart_candidates=discovery.production_restart_candidates,
        )
    if resolved.suffix.lower() in _RAW_TI_MDIN_EXTS:
        return RawTIInputDiscovery(
            root=discovery.root,
            prmtop_path=discovery.prmtop_path,
            trajectory_path=discovery.trajectory_path,
            reference_structure_path=discovery.reference_structure_path,
            production_mdin_path=resolved,
            production_restart_path=discovery.production_restart_path,
            trajectory_candidates=discovery.trajectory_candidates,
            production_mdin_candidates=discovery.production_mdin_candidates,
            production_restart_candidates=discovery.production_restart_candidates,
        )
    if _is_raw_ti_restart_file(resolved):
        return RawTIInputDiscovery(
            root=discovery.root,
            prmtop_path=discovery.prmtop_path,
            trajectory_path=discovery.trajectory_path,
            reference_structure_path=discovery.reference_structure_path,
            production_mdin_path=discovery.production_mdin_path,
            production_restart_path=resolved,
            trajectory_candidates=discovery.trajectory_candidates,
            production_mdin_candidates=discovery.production_mdin_candidates,
            production_restart_candidates=discovery.production_restart_candidates,
        )
    return discovery


def _discover_raw_ti_inputs(search_path: str | Path) -> list[RawTIInputDiscovery]:
    path = Path(search_path).expanduser().resolve()
    if not path.exists():
        return []

    direct_file = path if path.is_file() else None
    discoveries: list[RawTIInputDiscovery] = []
    seen: set[tuple[Path, Path, Path]] = set()
    for root in _raw_ti_search_roots_for_path(path):
        root_matches = 0
        for discovery in _discover_raw_ti_inputs_in_directory(root):
            if direct_file is not None and not _raw_ti_discovery_mentions_path(discovery, direct_file):
                continue
            discovery = _focus_raw_ti_discovery_on_path(discovery, direct_file) if direct_file is not None else discovery
            key = (
                discovery.root.resolve(),
                discovery.prmtop_path.resolve(),
                discovery.trajectory_path.resolve(),
            )
            if key in seen:
                continue
            seen.add(key)
            discoveries.append(discovery)
            root_matches += 1
        if direct_file is not None and root_matches:
            break
    return discoveries[:50]


def _select_production_stage(stages: list[WorkflowStageRun]) -> WorkflowStageRun | None:
    for stage in reversed(stages):
        stem = stage.stem.lower()
        title = stage.title.lower()
        if "prod" in stem or title == "production":
            return stage
    return stages[-1] if stages else None


def _select_snapshot_stage(
    stages: list[WorkflowStageRun],
    production_stage: WorkflowStageRun | None,
) -> WorkflowStageRun | None:
    if (
        production_stage is not None
        and production_stage.trajectory_path is not None
        and production_stage.trajectory_path.exists()
    ):
        return production_stage

    completed_candidates = [
        stage
        for stage in stages
        if stage.trajectory_path is not None and stage.trajectory_path.exists() and stage.completed
    ]
    if completed_candidates:
        return completed_candidates[-1]

    available_candidates = [
        stage for stage in stages if stage.trajectory_path is not None and stage.trajectory_path.exists()
    ]
    return available_candidates[-1] if available_candidates else None


def _workflow_readiness_note(
    *,
    prmtop_path: Path | None,
    reference_structure_path: Path | None,
    stages: list[WorkflowStageRun],
    production_stage: WorkflowStageRun | None,
    selected_stage: WorkflowStageRun | None,
) -> str:
    if prmtop_path is None:
        return "Missing 02_system/system.prmtop."
    if reference_structure_path is None:
        return "Missing 02_system/system.pdb (or 01_prepare/cleaned_input.pdb)."
    if not stages:
        return "No MD input stages were found under 03_md/inputs."
    if selected_stage is None:
        return "No usable trajectory (*.nc) was found under 03_md/outputs."
    if production_stage is None:
        return f"Using {selected_stage.stem}.nc because a production stage was not found."
    if selected_stage.filename != production_stage.filename:
        if selected_stage.completed:
            return f"Production is unavailable; free-energy estimation will use completed stage {selected_stage.stem}.nc."
        return f"Production is unavailable; free-energy estimation will use the current {selected_stage.stem}.nc trajectory."
    if production_stage.completed:
        return "Production is complete and ready for free-energy estimation."
    if production_stage.started:
        return "Production is still running; free-energy estimation will use the frames written so far."
    return "Production is prepared but has not started yet."


def _inspect_main_workflow_directory(path: str | Path) -> WorkflowDiscovery | None:
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not _looks_like_main_workflow_root(root):
        return None

    prmtop_path = root / "02_system" / "system.prmtop"
    reference_candidates = [
        root / "02_system" / "system.pdb",
        root / "01_prepare" / "cleaned_input.pdb",
    ]
    reference_structure_path = next((candidate for candidate in reference_candidates if candidate.exists()), None)

    inputs_dir = root / "03_md" / "inputs"
    outputs_dir = root / "03_md" / "outputs"
    stages: list[WorkflowStageRun] = []
    if inputs_dir.exists():
        for index, spec in enumerate(_load_stage_definitions(inputs_dir), start=1):
            stage = _build_stage_run(index, spec, inputs_dir=inputs_dir, outputs_dir=outputs_dir)
            if stage is not None:
                stages.append(stage)

    production_stage = _select_production_stage(stages)
    selected_stage = _select_snapshot_stage(stages, production_stage)
    selectable = prmtop_path.exists() and reference_structure_path is not None and selected_stage is not None
    readiness_note = _workflow_readiness_note(
        prmtop_path=prmtop_path if prmtop_path.exists() else None,
        reference_structure_path=reference_structure_path,
        stages=stages,
        production_stage=production_stage,
        selected_stage=selected_stage,
    )

    return WorkflowDiscovery(
        root=root,
        prmtop_path=prmtop_path if prmtop_path.exists() else None,
        reference_structure_path=reference_structure_path,
        production_stage=production_stage,
        selected_stage=selected_stage,
        stages=stages,
        selectable=selectable,
        readiness_note=readiness_note,
    )


def _discover_main_workflow_directories(search_dir: Path) -> list[WorkflowDiscovery]:
    resolved_search_dir = search_dir.expanduser().resolve()
    candidate_roots = [resolved_search_dir]
    candidate_roots.extend(
        sorted((path for path in resolved_search_dir.iterdir() if path.is_dir()), key=lambda item: item.name.lower())
    )

    discoveries: list[WorkflowDiscovery] = []
    seen: set[Path] = set()
    for candidate in candidate_roots:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        discovery = _inspect_main_workflow_directory(resolved)
        if discovery is not None:
            discoveries.append(discovery)
    return discoveries


def _format_ns(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _stage_is_recently_active(stage: WorkflowStageRun) -> bool:
    return (
        stage.started
        and stage.latest_update_age_seconds is not None
        and stage.latest_update_age_seconds <= _ACTIVE_STAGE_WINDOW_SECONDS
    )


def _production_status_text(discovery: WorkflowDiscovery) -> str:
    stage = discovery.production_stage
    if stage is None:
        return "Not found"
    if stage.completed:
        return "Completed"
    if _stage_is_recently_active(stage):
        return "In progress"
    return "Not completed"


def _selected_trajectory_text(discovery: WorkflowDiscovery) -> str:
    stage = discovery.selected_stage
    if stage is None or stage.trajectory_path is None:
        return "None"
    suffix = " fallback" if discovery.production_stage and stage.filename != discovery.production_stage.filename else ""
    return f"{stage.stem}.nc{suffix}"


def _workflow_status_label(discovery: WorkflowDiscovery) -> str:
    if discovery.selectable:
        if discovery.production_stage is None:
            return "Ready (fallback)"
        if discovery.selected_stage is not None and discovery.selected_stage.filename != discovery.production_stage.filename:
            return "Ready (fallback)"
        if discovery.production_stage.completed:
            return "Ready"
        return "Ready (partial)"
    if discovery.prmtop_path is None:
        return "No prmtop"
    if discovery.reference_structure_path is None:
        return "No ref PDB"
    if not discovery.stages:
        return "No MD inputs"
    return "No trajectory"


def _display_discovered_workflows(search_dir: Path, discoveries: list[WorkflowDiscovery]) -> None:
    table = Table(title="Detected main.py Workflow Folders", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Folder", style="bold white")
    table.add_column("Prod", style="white", no_wrap=True)
    table.add_column("Traj", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_row(
        "0",
        "Manual path/raw files",
        "-",
        "-",
        "Manual",
    )
    for index, discovery in enumerate(discoveries, start=1):
        table.add_row(
            str(index),
            discovery.root.name,
            _production_status_text(discovery),
            _selected_trajectory_text(discovery),
            _workflow_status_label(discovery),
        )
    console.print(table)
    console.print(
        f"[dim]Scanned {search_dir.resolve()} and its immediate subdirectories for folders created by main.py.[/dim]"
    )


def _prompt_discovered_workflow(discoveries: list[WorkflowDiscovery]) -> WorkflowDiscovery | None:
    selectable_indices = [index for index, item in enumerate(discoveries, start=1) if item.selectable]
    default_choice = str(selectable_indices[0]) if selectable_indices else "0"
    while True:
        raw = typer.prompt(
            "Choose a workflow folder number (0 = enter a path manually)",
            default=default_choice,
        ).strip()
        if raw == "0":
            return None
        try:
            choice = int(raw)
        except ValueError:
            console.print("[bold red]Please enter 0 or one of the listed workflow numbers.[/bold red]")
            continue
        if choice < 1 or choice > len(discoveries):
            console.print("[bold red]Please choose a number from the table.[/bold red]")
            continue
        selected = discoveries[choice - 1]
        if not selected.selectable:
            console.print(f"[bold yellow]{selected.readiness_note}[/bold yellow]")
            continue
        return selected


def _parse_discovered_workflow_selection(
    raw: str,
    discoveries: list[WorkflowDiscovery],
) -> list[WorkflowDiscovery] | None:
    token = raw.strip().lower()
    if token == "0":
        return None
    if token in {"a", "all", "*"}:
        selected = [item for item in discoveries if item.selectable]
        if not selected:
            raise ValueError("No detected workflow folder is ready for TI.")
        return selected
    if not token:
        raise ValueError("Choose A, 0, one workflow number, or comma-separated workflow numbers.")

    selected: list[WorkflowDiscovery] = []
    seen: set[int] = set()
    for item in re.split(r"[\s,]+", token):
        if not item:
            continue
        if not item.isdigit():
            raise ValueError("Choose A, 0, one workflow number, or comma-separated workflow numbers.")
        index = int(item)
        if index == 0:
            raise ValueError("0 (manual input) cannot be combined with detected workflow numbers.")
        if index < 1 or index > len(discoveries):
            raise ValueError(f"Workflow number {index} is not present in the table.")
        discovery = discoveries[index - 1]
        if not discovery.selectable:
            raise ValueError(f"Workflow {index} is not ready: {discovery.readiness_note}")
        if index not in seen:
            seen.add(index)
            selected.append(discovery)
    if not selected:
        raise ValueError("Choose at least one ready workflow folder.")
    return selected


def _prompt_discovered_workflows(
    discoveries: list[WorkflowDiscovery],
) -> list[WorkflowDiscovery] | None:
    selectable_indices = [index for index, item in enumerate(discoveries, start=1) if item.selectable]
    default_choice = "A" if len(selectable_indices) > 1 else (str(selectable_indices[0]) if selectable_indices else "0")
    while True:
        raw = typer.prompt(
            "Choose A for all ready workflows, one number, comma-separated numbers, or 0 for manual input",
            default=default_choice,
        )
        try:
            return _parse_discovered_workflow_selection(raw, discoveries)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")


def _complex_input_from_discovery(discovery: WorkflowDiscovery) -> TIInputSelection:
    if (
        discovery.prmtop_path is None
        or discovery.reference_structure_path is None
        or discovery.selected_stage is None
        or discovery.selected_stage.trajectory_path is None
    ):
        raise ValueError("The selected workflow directory does not provide a usable TI input bundle.")
    return TIInputSelection(
        complex_input=ComplexInputConfig(
            prmtop_path=str(discovery.prmtop_path),
            trajectory_path=str(discovery.selected_stage.trajectory_path),
            reference_structure_path=str(discovery.reference_structure_path),
            production_mdin_path=str(discovery.selected_stage.input_path),
            production_restart_path=(
                str(discovery.selected_stage.restart_path)
                if discovery.selected_stage.restart_path.exists()
                else None
            ),
        ),
        workflow_root=discovery.root,
        source_kind=_workflow_source_kind(discovery.root),
    )


def _workflow_source_kind(root: Path | None) -> str:
    if root is None:
        return "raw"
    system_des_manifest = root / "02_system" / "des_manifest.json"
    if system_des_manifest.exists():
        return "des"
    system_manifest = root / "02_system" / "system_manifest.json"
    if system_manifest.exists():
        try:
            payload = json.loads(system_manifest.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("workflow_type") == "des":
            return "des"
    workflow_manifest = root / "workflow_manifest.json"
    if workflow_manifest.exists():
        try:
            payload = json.loads(workflow_manifest.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        system_payload = payload.get("system") if isinstance(payload, dict) else None
        if isinstance(system_payload, dict):
            metadata = system_payload.get("system_metadata")
            if isinstance(metadata, dict) and metadata.get("workflow_type") == "des":
                return "des"
    prepare_manifest = root / "01_prepare" / "prepare_manifest.json"
    if prepare_manifest.exists():
        try:
            payload = json.loads(prepare_manifest.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("des") is not None:
            return "des"
    return "workflow"


def _input_selection_uses_in_place_ti(selection: TIInputSelection) -> bool:
    return selection.source_kind in {"raw", "des"}


def _input_selection_includes_unbound_metal_sites(selection: TIInputSelection) -> bool:
    return _input_selection_uses_in_place_ti(selection)


def _display_selected_workflow_inputs(discovery: WorkflowDiscovery) -> None:
    if (
        discovery.prmtop_path is None
        or discovery.reference_structure_path is None
        or discovery.selected_stage is None
        or discovery.selected_stage.trajectory_path is None
    ):
        return

    table = Table(title="Selected Free-Energy Inputs from main.py Workflow", box=box.SIMPLE_HEAVY)
    table.add_column("Item", style="bold white")
    table.add_column("Path", style="cyan", overflow="fold")
    table.add_row("Workflow folder", str(discovery.root))
    table.add_row("Complex prmtop", str(discovery.prmtop_path))
    table.add_row("Reference structure", str(discovery.reference_structure_path))
    table.add_row("Selected trajectory", str(discovery.selected_stage.trajectory_path))
    table.add_row("Inherited mdin", str(discovery.selected_stage.input_path))
    table.add_row(
        "Production restart",
        str(discovery.selected_stage.restart_path) if discovery.selected_stage.restart_path.exists() else "-",
    )
    console.print(table)


def _display_raw_ti_inputs(search_root: Path, discoveries: list[RawTIInputDiscovery]) -> None:
    table = Table(title="Detected Raw AMBER TI Inputs", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Folder", style="bold white", overflow="fold")
    table.add_column("Topology", style="cyan", overflow="fold")
    table.add_column("Trajectory", style="cyan", overflow="fold")
    table.add_column("PDB", style="white", overflow="fold")
    table.add_column("mdin", style="white", overflow="fold")
    table.add_column("Restart", style="white", overflow="fold")
    table.add_row("0", "Manual raw files", "-", "-", "-", "-", "-")
    for index, discovery in enumerate(discoveries, start=1):
        table.add_row(
            str(index),
            _format_raw_ti_path(discovery.root, search_root=search_root),
            _format_raw_ti_path(discovery.prmtop_path, search_root=search_root),
            _format_raw_ti_path(discovery.trajectory_path, search_root=search_root),
            _format_raw_ti_path(discovery.reference_structure_path, search_root=search_root),
            _format_raw_ti_path(discovery.production_mdin_path, search_root=search_root),
            _format_raw_ti_path(discovery.production_restart_path, search_root=search_root),
        )
    console.print(table)
    console.print(f"[dim]Scanned {search_root.resolve()} recursively for raw AMBER topology/trajectory inputs.[/dim]")


def _prompt_raw_ti_input_selection(
    search_root: Path,
    discoveries: list[RawTIInputDiscovery],
) -> RawTIInputDiscovery | None:
    _display_raw_ti_inputs(search_root, discoveries)
    while True:
        raw = typer.prompt("Choose a raw AMBER input number (0 = enter paths manually)", default="1").strip()
        if raw == "0":
            return None
        try:
            index = int(raw)
        except ValueError:
            console.print("[bold red]Please enter 0 or one of the listed raw AMBER input numbers.[/bold red]")
            continue
        if 1 <= index <= len(discoveries):
            return discoveries[index - 1]
        console.print("[bold red]Please choose a number from the table.[/bold red]")


def _prompt_raw_ti_trajectory_selection(
    discovery: RawTIInputDiscovery,
    *,
    search_root: Path,
) -> RawTIInputDiscovery:
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

    table = Table(title="Available TI Trajectory Inputs", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Trajectory", style="bold white", overflow="fold")
    table.add_column("Matched mdin", style="cyan", overflow="fold")
    table.add_column("Matched restart", style="white", overflow="fold")
    default_index = 1
    for index, trajectory in enumerate(unique_candidates, start=1):
        if trajectory.resolve() == discovery.trajectory_path.resolve():
            default_index = index
        mdin = _matching_raw_ti_mdin_for_trajectory(trajectory, discovery.production_mdin_candidates)
        restart = _matching_raw_ti_restart_for_trajectory(trajectory, discovery.production_restart_candidates)
        table.add_row(
            str(index),
            _format_raw_ti_path(trajectory, search_root=search_root),
            _format_raw_ti_path(mdin, search_root=search_root),
            _format_raw_ti_path(restart, search_root=search_root),
        )
    console.print(table)
    while True:
        raw = typer.prompt("Choose the trajectory to use for TI snapshot extraction", default=str(default_index)).strip()
        try:
            index = int(raw)
        except ValueError:
            console.print("[bold red]Please enter one of the listed trajectory numbers.[/bold red]")
            continue
        if 1 <= index <= len(unique_candidates):
            selected = unique_candidates[index - 1]
            console.print(f"[bold cyan]Selected trajectory:[/bold cyan] {selected}")
            return _with_selected_raw_ti_trajectory(discovery, selected)
        console.print("[bold red]Please choose a number from the trajectory table.[/bold red]")


def _display_selected_raw_ti_inputs(discovery: RawTIInputDiscovery) -> None:
    table = Table(title="Selected Free-Energy Inputs from Raw AMBER Files", box=box.SIMPLE_HEAVY)
    table.add_column("Item", style="bold white")
    table.add_column("Path", style="cyan", overflow="fold")
    table.add_row("Raw data folder", str(discovery.root))
    table.add_row("Complex prmtop", str(discovery.prmtop_path))
    table.add_row("Reference structure", "-" if discovery.reference_structure_path is None else str(discovery.reference_structure_path))
    table.add_row("Selected trajectory", str(discovery.trajectory_path))
    table.add_row("Inherited mdin", "-" if discovery.production_mdin_path is None else str(discovery.production_mdin_path))
    table.add_row(
        "Production restart",
        "-" if discovery.production_restart_path is None else str(discovery.production_restart_path),
    )
    console.print(table)


def _complex_input_from_raw_discovery(discovery: RawTIInputDiscovery) -> TIInputSelection:
    reference_structure_path = discovery.reference_structure_path
    if reference_structure_path is None:
        if discovery.production_restart_path is not None:
            generated_path = discovery.root / "simple_reference.pdb"
            try:
                reference_structure_path = generate_reference_pdb_from_amber_restart(
                    prmtop_path=discovery.prmtop_path,
                    restart_path=discovery.production_restart_path,
                    output_path=generated_path,
                )
                console.print(
                    "[bold cyan]Generated reference PDB:[/bold cyan] "
                    f"{reference_structure_path} [dim](from prmtop + restart)[/dim]"
                )
            except Exception as exc:
                console.print(
                    "[bold yellow]Could not generate a reference PDB automatically from the detected restart:[/bold yellow] "
                    f"{exc}"
                )
        if reference_structure_path is None:
            print_notice(
                "Reference PDB Required",
                "A topology/trajectory pair was detected, but no reference PDB was found near it. "
                "TI needs a PDB-like reference structure to detect the metal site.",
                border_style="yellow",
            )
            reference_structure_path = Path(str(_prompt_existing_path("Path to reference structure (PDB)"))).expanduser().resolve()

    return TIInputSelection(
        complex_input=ComplexInputConfig(
            prmtop_path=str(discovery.prmtop_path),
            trajectory_path=str(discovery.trajectory_path),
            reference_structure_path=str(reference_structure_path),
            production_mdin_path=None if discovery.production_mdin_path is None else str(discovery.production_mdin_path),
            production_restart_path=(
                None if discovery.production_restart_path is None else str(discovery.production_restart_path)
            ),
        ),
        workflow_root=None,
        source_kind="raw",
    )


def _prompt_raw_ti_input_from_path(path: str | Path) -> TIInputSelection | None:
    search_path = Path(path).expanduser().resolve()
    discoveries = _discover_raw_ti_inputs(search_path)
    if not discoveries:
        return None

    search_root = search_path if search_path.is_dir() else discoveries[0].root
    if len(discoveries) == 1:
        selected = discoveries[0]
        console.print(f"[dim]Detected one raw AMBER topology/trajectory bundle under {search_root}.[/dim]")
    else:
        selected = _prompt_raw_ti_input_selection(search_root, discoveries)
        if selected is None:
            return None
    selected = _prompt_raw_ti_trajectory_selection(selected, search_root=search_root)
    _display_selected_raw_ti_inputs(selected)
    return _complex_input_from_raw_discovery(selected)


def _warn_for_selected_workflow(discovery: WorkflowDiscovery) -> None:
    selected_stage = discovery.selected_stage
    if selected_stage is None or selected_stage.trajectory_path is None:
        return

    production_stage = discovery.production_stage
    if production_stage is None:
        print_notice(
            "Production MD Warning",
            "A production MD input stage was not found, so free-energy estimation will use "
            f"{selected_stage.stem}.nc and inherit settings from {selected_stage.filename}.",
            border_style="yellow",
        )
        return

    if selected_stage.filename != production_stage.filename:
        source_label = "the latest completed trajectory stage" if selected_stage.completed else "the latest available trajectory stage"
        print_notice(
            "Production MD Warning",
            "A usable production trajectory was not found, so free-energy estimation will use "
            f"{selected_stage.stem}.nc from {source_label} and inherit settings from {selected_stage.filename}. "
            "The selected snapshot may be less equilibrated than a production snapshot.",
            border_style="yellow",
        )
        return

    if not production_stage.completed:
        print_notice(
            "Production MD In Progress",
            "Production has not yet reached its target duration "
            f"({_production_status_text(discovery)}). Free-energy estimation will extract the last available snapshot from "
            f"{selected_stage.trajectory_path.name}.",
            border_style="yellow",
        )


def _prompt_manual_workflow_or_raw_files() -> TIInputSelection:
    console.print(
        "[dim]Manual mode: enter a main.py workflow directory, a raw AMBER file/folder, or leave it blank to provide each file directly.[/dim]"
    )
    while True:
        input_root = _prompt_existing_path(
            "Path to a main.py workflow directory or raw AMBER file/folder (blank = raw files)",
            optional=True,
        )
        if input_root is None:
            raw_discoveries = _discover_raw_ti_inputs(Path.cwd())
            if raw_discoveries:
                selected_raw = _prompt_raw_ti_input_selection(Path.cwd(), raw_discoveries)
                if selected_raw is not None:
                    selected_raw = _prompt_raw_ti_trajectory_selection(selected_raw, search_root=Path.cwd())
                    _display_selected_raw_ti_inputs(selected_raw)
                    return _complex_input_from_raw_discovery(selected_raw)
            break
        discovery = _inspect_main_workflow_directory(input_root)
        if discovery is None:
            raw_selection = _prompt_raw_ti_input_from_path(input_root)
            if raw_selection is not None:
                return raw_selection
            console.print(
                "[bold red]That path is neither a main.py workflow directory nor a raw AMBER TI input bundle.[/bold red]"
            )
            console.print(f"[dim]{_RAW_TI_REQUIRED_FILES_MESSAGE}[/dim]")
            continue
        if not discovery.selectable:
            console.print(f"[bold yellow]{discovery.readiness_note}[/bold yellow]")
            continue
        _display_selected_workflow_inputs(discovery)
        _warn_for_selected_workflow(discovery)
        return _complex_input_from_discovery(discovery)

    console.print("[dim]Raw file input mode: enter the files TI should use directly.[/dim]")
    prmtop_path = _prompt_existing_path("Path to complex prmtop")
    trajectory_path = _prompt_existing_path("Path to trajectory for TI snapshot extraction")
    reference_structure_path = _prompt_existing_path("Path to reference structure (PDB)")
    production_mdin_path = _prompt_existing_path("Path to mdin to inherit settings (blank to use defaults)", optional=True)
    production_restart_path = _prompt_existing_path(
        "Path to production restart with velocities (blank = auto/fallback)",
        optional=True,
    )
    return TIInputSelection(
        complex_input=ComplexInputConfig(
            prmtop_path=prmtop_path,
            trajectory_path=trajectory_path,
            reference_structure_path=reference_structure_path,
            production_mdin_path=production_mdin_path,
            production_restart_path=production_restart_path,
        ),
        workflow_root=None,
        source_kind="raw",
    )


def _prompt_complex_input_selection() -> TIInputSelection:
    search_dir = Path.cwd()
    discoveries = _discover_main_workflow_directories(search_dir)
    if discoveries:
        _display_discovered_workflows(search_dir, discoveries)
        selected = _prompt_discovered_workflow(discoveries)
        if selected is not None:
            _display_selected_workflow_inputs(selected)
            _warn_for_selected_workflow(selected)
            return _complex_input_from_discovery(selected)
    else:
        raw_discoveries = _discover_raw_ti_inputs(search_dir)
        if raw_discoveries:
            console.print("[dim]No main.py workflow folders were detected, but raw AMBER inputs were found.[/dim]")
            selected_raw = _prompt_raw_ti_input_selection(search_dir, raw_discoveries)
            if selected_raw is not None:
                selected_raw = _prompt_raw_ti_trajectory_selection(selected_raw, search_root=search_dir)
                _display_selected_raw_ti_inputs(selected_raw)
                return _complex_input_from_raw_discovery(selected_raw)
        else:
            console.print(
                "[dim]No main.py workflow folders or raw AMBER topology/trajectory bundles were detected here, so the wizard will switch to manual input.[/dim]"
            )
    return _prompt_manual_workflow_or_raw_files()


def _prompt_complex_input_selections() -> list[TIInputSelection]:
    """FreeE TI input picker with multi-workflow selection support."""
    search_dir = Path.cwd()
    discoveries = _discover_main_workflow_directories(search_dir)
    if discoveries:
        _display_discovered_workflows(search_dir, discoveries)
        selected = _prompt_discovered_workflows(discoveries)
        if selected is not None:
            selections: list[TIInputSelection] = []
            for discovery in selected:
                _display_selected_workflow_inputs(discovery)
                _warn_for_selected_workflow(discovery)
                selections.append(_complex_input_from_discovery(discovery))
            return selections
    else:
        raw_discoveries = _discover_raw_ti_inputs(search_dir)
        if raw_discoveries:
            console.print("[dim]No main.py workflow folders were detected, but raw AMBER inputs were found.[/dim]")
            selected_raw = _prompt_raw_ti_input_selection(search_dir, raw_discoveries)
            if selected_raw is not None:
                selected_raw = _prompt_raw_ti_trajectory_selection(selected_raw, search_root=search_dir)
                _display_selected_raw_ti_inputs(selected_raw)
                return [_complex_input_from_raw_discovery(selected_raw)]
        else:
            console.print(
                "[dim]No main.py workflow folders or raw AMBER topology/trajectory bundles were detected here, "
                "so the wizard will switch to manual input.[/dim]"
            )
    return [_prompt_manual_workflow_or_raw_files()]


def build_ti_wizard_config(write_config: str | None, *, dry_run: bool) -> TIWorkflowConfig:
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
    production_restart_path = input_selection.complex_input.production_restart_path

    wizard_tmp = Path(".simple_ti_wizard")
    console.print("[dim]The wizard will now inspect the last snapshot before deciding whether cluster analysis is allowed.[/dim]")
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

    if len(candidates) == 1:
        selected = candidates[0]
        console.print(f"[dim]One bound metal candidate was detected, so site {selected.site} is selected automatically.[/dim]")
    else:
        choices = [
            WizardChoice(str(candidate.site), f"Site {candidate.site}", f"{candidate.element} at {candidate.key}")
            for candidate in candidates
        ]
        _display_choice_table("Bound metal site selection", choices)
        selected = select_site(candidates, int(_prompt_choice("Choose the metal site to decouple", choices)))

    selected_assessment = next(item for item in assessments if item.site == selected.site)
    allow_unstable = False
    if not selected_assessment.stable:
        print_notice("Strong Warning", selected_assessment.note, border_style="bold red")
        allow_unstable = typer.confirm(
            "Proceed with the last snapshot anyway?",
            default=False,
        )
        if not allow_unstable:
            raise typer.Abort()

    _print_step_header(
        2,
        "Choose the TI Mode, Snapshot, and Reference Settings",
        (
            "DES/raw AMBER inputs use in-place TI from the existing MD topology and restart. "
            "A metal-in-water reference leg is optional for RBFE and will be offered separately."
            if _input_selection_uses_in_place_ti(input_selection)
            else "First choose how TI should be implemented. Then pick the snapshot source and confirm the metal-in-water reference settings used for the DeltaG reference leg. The solvent reference will live under a reusable water_ref directory and will be pre-equilibrated before TI. Cluster analysis is only available when the selected metal site still looks stable in the last snapshot."
        ),
    )
    in_place_ti = _input_selection_uses_in_place_ti(input_selection)
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
    snapshot_mode = _prompt_snapshot_mode(selected_stable=selected_assessment.stable)
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
        3,
        "Choose the Execution Script and Output Location",
        (
            "GTI/CUDA requires GPU/pmemd.cuda. GPU master and Tahoma sbatch files will be generated."
            if ti_implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI
            else "Select whether to generate CPU or GPU master sbatch files. A separate Tahoma script will also be written automatically."
        ),
    )
    profile = _prompt_ti_execution_profile(ti_implementation_mode)
    output_dir_path = _default_output_directory(
        reference_structure_path,
        workflow_root=input_selection.workflow_root,
    )
    console.print(f"[bold cyan]Output directory:[/bold cyan] {output_dir_path}")

    config = TIWorkflowConfig(
        complex_input=ComplexInputConfig(
            prmtop_path=prmtop_path,
            trajectory_path=trajectory_path,
            reference_structure_path=reference_structure_path,
            production_mdin_path=production_mdin_path,
            production_restart_path=production_restart_path,
        ),
        snapshot=SnapshotConfig(
            mode=snapshot_mode,
            allow_unstable_last_snapshot=allow_unstable,
        ),
        metal=MetalSelectionConfig(
            selected_site=selected.site,
            formal_charge=formal_charge,
        ),
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
    if write_config:
        saved = save_config(config, write_config)
        console.print(f"Saved config to {saved}")
    return config


