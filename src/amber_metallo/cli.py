from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
import math
import re
from tempfile import TemporaryDirectory

import typer
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from amber_metallo.amber.leap import (
    allowed_metal_charges,
    c4_parameter_set_supports_metal_charge,
    calculate_salt_ions,
    ion_parameter_requirements,
)
from amber_metallo.config import (
    BoxShape,
    ChargeMethod,
    DESC4ParameterSet,
    DESComponent,
    DESConfig,
    DESMetalSiteConfig,
    DESMixingMode,
    DESReplicateOrder,
    DESSizeMode,
    InputConfig,
    InputSource,
    LigandsConfig,
    LigandParameterAssignment,
    LigandMode,
    MDConfig,
    MetalAnchorMode,
    MetalChargeAssignment,
    MetalInsertion,
    MetalModel,
    MetalReplacement,
    NeutralizationIon,
    PrepareConfig,
    ProteinSiteRespClusterConfig,
    ProteinSiteRespConfig,
    ProteinSiteRespMode,
    ProteinSiteRespScope,
    ProtonationChange,
    ProtonationConfig,
    ProtonationEngine,
    ProtocolKind,
    RespApplyMode,
    ResidueMaskNumbering,
    SaltConfig,
    SaltKind,
    SaltMode,
    SlurmConfig,
    SlurmProfile,
    SystemConfig,
    WorkflowConfig,
    charge_method_uses_resp,
    load_config,
    normalize_charge_method,
    save_config,
)
from amber_metallo.des import (
    DES_COMPONENTS,
    DES_RECOMMENDED_SETS,
    available_des_components,
    classify_des_library_bundle,
    discover_des_library_candidates,
    estimate_des_net_charge,
    estimate_des_plan,
    recommended_ratio_for_components,
    register_custom_des_component,
    resolve_des_neutralization_ion,
)
from amber_metallo.environment import detect_amber_environment, environment_summary, is_linux_execution_host
from amber_metallo.inspection import (
    MetalSite,
    ResidueRecord,
    SUPPORTED_METALS,
    classify_residue,
    fetch_pdb_structure,
    inspect_structure,
    load_structure,
    looks_like_pdb_id,
    residue_key,
    StructureSummary,
)
from amber_metallo.ligand_param import (
    manual_ligand_requirements_text,
    prepare_canonical_small_molecule_mol2,
    validate_manual_ligand_bundle,
)
from amber_metallo.metal_insert import (
    donor_candidates_for_residue_selectors,
    resolve_metal_insertion,
    structure_atom_reference_rows,
)
from amber_metallo.missing_loops import MissingLoopSummary
from amber_metallo.prep import prepare_structure
from amber_metallo.protonation import (
    direct_metal_coordination_observations,
    focused_restraint_residue_locators,
    ProtonationDisplayCandidate,
    predict_protonation_prediction,
    residue_locator,
)
from amber_metallo.protein_site_resp import (
    ProteinSiteRespResumeCandidate,
    find_protein_site_resp_resume_candidates,
    load_topology_atoms,
    suggested_low_spin_multiplicity,
    suggested_spin_multiplicity,
)
from amber_metallo.reporting import console, emit_key_value_table, print_workflow_summary
from amber_metallo.subdirectory_search import search_subdirectories_enabled
from amber_metallo.tool_config import (
    AmberSettings,
    AmberToolsSettings,
    NWChemSettings,
    ToolConfig,
    default_tool_config_path,
    discover_ambertools_home,
    discover_binary,
    load_tool_config,
    save_tool_config,
    tool_config_summary,
)
from amber_metallo.qm.editor import launch_resp_editor
from amber_metallo.qm.nwchem import (
    build_default_session_state,
    find_resp_job_candidates,
    find_resp_source_candidates,
    load_resp_job_candidate,
    load_molecule,
    molecule_fingerprint,
)
from amber_metallo.workflow import run_workflow


app = typer.Typer(help="SIMPLE molecular-simulation workflow automation.")

PROTEIN_FF_DESCRIPTIONS = {
    "ff19SB": "Modern Amber protein force field. Recommended default for most protein systems.",
    "ff14SB": "Widely used Amber protein force field with broad legacy compatibility.",
    "ff99SB": "Legacy Amber protein force field.",
    "ff99SBildn": "Legacy ff99SB variant with improved Ile/Leu/Asp/Asn side chains.",
}
SMALL_MOLECULE_MODE_DESCRIPTIONS = {
    "gaff2": (
        "Automatically parameterize non-standard molecules with Antechamber, parmchk2, and GAFF2 atom types. "
        "Choose AM1-BCC or RESP via NWChem in the next step."
    ),
    "gaff": (
        "Automatically parameterize non-standard molecules with Antechamber, parmchk2, and the older GAFF atom-type set. "
        "Choose AM1-BCC or RESP via NWChem in the next step."
    ),
    "manual": "Use your own Amber-ready files such as mol2/prepi/off/lib plus frcmod.",
}
SMALL_MOLECULE_INPUT_EXTENSIONS = (".pdb", ".mol2", ".txt", ".smi")
SMALL_MOLECULE_INPUT_PROMPT = "Small-molecule input file(s) [.pdb/.mol2/.txt/.smi], comma separated"
SMALL_MOLECULE_INPUT_HINT = (
    "Small-molecule inputs support [bold cyan]PDB[/bold cyan], [bold cyan]MOL2[/bold cyan], "
    "and [bold green]SMILES[/bold green] files. Accepted extensions: "
    "[bold white].pdb[/bold white], [bold white].mol2[/bold white], "
    "[bold white].txt[/bold white], [bold white].smi[/bold white]. "
    "Enter multiple files separated by commas."
)
CHARGE_METHOD_DESCRIPTIONS = {
    ChargeMethod.FULL_RESP.value: (
        "Use the metal-inclusive structure for RESP/QM and the popup editor, then project ligand atom charges back "
        "onto the ligand-only MOL2 used for Amber setup."
    ),
    ChargeMethod.RESP_ANTECHAMBER.value: (
        "Use the current split-metal RESP workflow: generate RESP charges with NWChem, apply them to an "
        "Antechamber-typed ligand MOL2, then run parmchk2 for bonded parameters."
    ),
    ChargeMethod.ANTECHAMBER.value: (
        "Use the Antechamber AM1-BCC workflow. SIMPLE will ask for ligand net charge and spin multiplicity next, "
        "then continue directly into Amber setup."
    ),
}
WATER_MODEL_DESCRIPTIONS = {
    "spce": "SPC/E water model. Uses the current Amber/Li-Merz-style ion files when 12-6-4 is requested.",
    "spceb": "SPC/E-b water model.",
    "tip3p": "TIP3P water model. Common and widely supported in Amber workflows.",
    "opc": "OPC 4-site water model with improved bulk-water properties. Default for 12-6-4 workflows; uses bundled Duvail parameters.",
    "opc3": "OPC3 3-site water model with improved efficiency and water behavior.",
    "opc3pol": "OPC3-pol polarizable 3-site water model.",
    "tip4pew": "TIP4P-Ew water model.",
    "tip4pd": "TIP4P-D water model with improved dispersion behavior.",
    "tip5p": "TIP5P water model.",
    "fb3": "TIP3P-FB water model.",
    "fb4": "TIP4P-FB water model.",
}
SUPPORTED_WATER_MODELS = {
    "spce",
    "spceb",
    "tip3p",
    "opc",
    "opc3",
    "opc3pol",
    "tip4pew",
    "tip4pd",
    "tip5p",
    "fb3",
    "fb4",
}
HIDDEN_WATER_MODELS = {"fb3mod", "tip4pd-a99sbdisp"}
DUAL_SUPPORTED_METAL_TI_WATER_MODELS = (
    "opc",
    "spce",
    "tip3p",
    "tip4pew",
    "opc3",
    "fb3",
    "fb4",
)
BOX_SHAPE_DESCRIPTIONS = {
    BoxShape.OCT.value: "Truncated octahedron. Usually needs fewer waters for compact globular systems.",
    BoxShape.CUBIC.value: "Cubic/rectangular box generated with tleap's solvateBox.",
}
PROTOCOL_DESCRIPTIONS = {
    ProtocolKind.FIFTEEN_STEP.value: (
        "Detailed relaxation workflow.\n"
        "Five restrained minimization stages, gradual heating, staged NVT/NPT equilibration,\n"
        "then unrestrained production. Best default for careful setup."
    ),
    ProtocolKind.FOUR_STEP.value: (
        "Compact workflow for faster turnaround.\n"
        "Restrained minimization, short heating/equilibration, then production.\n"
        "Useful when you want a quicker first-pass run."
    ),
}
DEFAULT_METAL_CHARGES = {
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
STANDARD_RESIDUE_CHARGES = {
    "ARG": 1,
    "ASP": -1,
    "ASH": 0,
    "GLU": -1,
    "GLH": 0,
    "LYS": 1,
    "LYN": 0,
    "HIP": 1,
    "HID": 0,
    "HIE": 0,
    "HIS": 0,
    "CYM": -1,
    "CYX": 0,
    "A": -1,
    "C": -1,
    "G": -1,
    "U": -1,
    "DA": -1,
    "DC": -1,
    "DG": -1,
    "DT": -1,
    "RA": -1,
    "RC": -1,
    "RG": -1,
    "RU": -1,
}
PROTONATION_STATE_GUIDE = (
    "HIS = neutral histidine with unspecified tautomer\n"
    "HID = histidine protonated at ND1 (NE2 is deprotonated)\n"
    "HIE = histidine protonated at NE2 (ND1 is deprotonated)\n"
    "HIP = doubly protonated histidinium (+1)\n"
    "ASH = protonated/neutral Asp\n"
    "GLH = protonated/neutral Glu\n"
    "LYN = neutral Lys\n"
    "CYM = deprotonated thiolate Cys"
)
ION_FORMAL_CHARGES = {
    "Na+": 1,
    "K+": 1,
    "Ca2+": 2,
    "Cl-": -1,
    "Br-": -1,
}
BACK_TOKENS = {"b", "back"}
ADD_DES_LIBRARY_MODE = "add_des_library"


class WizardBack(Exception):
    """Signal that the interactive wizard should return to the previous section."""


@dataclass(slots=True)
class _FocusedRestraintDisplayRow:
    residue_label: str
    reordered_number: int
    residue_name: str
    reason: str
    style: str


@dataclass(slots=True)
class _MetalReplacementVariant:
    replacements: list[MetalReplacement]
    suffix_tokens: tuple[str, ...]


@dataclass(slots=True)
class _MetalActionPlan:
    remove_metals: bool = False
    metal_deletions: list[int] = field(default_factory=list)
    metal_insertions: list[MetalInsertion] = field(default_factory=list)
    variants: list[_MetalReplacementVariant] = field(
        default_factory=lambda: [_MetalReplacementVariant(replacements=[], suffix_tokens=())]
    )

    @property
    def is_batch(self) -> bool:
        return len(self.variants) > 1


@dataclass(slots=True)
class WizardBuildResult:
    configs: list[WorkflowConfig]
    variant_suffixes: list[tuple[str, ...]] = field(default_factory=list)
    saved_config_paths: list[Path] = field(default_factory=list)

    @property
    def is_batch(self) -> bool:
        return len(self.configs) > 1


@dataclass(slots=True)
class _ProteinSiteRespResumeSelection:
    candidates: list[ProteinSiteRespResumeCandidate]


@dataclass(slots=True)
class _ProtonationVariant:
    protonation_config: ProtonationConfig
    ph: float | None = None
    ph_token: str | None = None
    signature: tuple[tuple[object, ...], ...] = field(default_factory=tuple)


METAL_REPLACEMENT_CHOICES = [
    ("Co", "Cobalt"),
    ("Cu", "Copper"),
    ("Ni", "Nickel"),
    ("Mn", "Manganese"),
    ("Fe", "Iron"),
    ("Sc", "Scandium"),
    ("Y", "Yttrium"),
    ("La", "Lanthanum"),
    ("Ce", "Cerium"),
    ("Pr", "Praseodymium"),
    ("Nd", "Neodymium"),
    ("Pm", "Promethium"),
    ("Sm", "Samarium"),
    ("Eu", "Europium"),
    ("Gd", "Gadolinium"),
    ("Tb", "Terbium"),
    ("Dy", "Dysprosium"),
    ("Ho", "Holmium"),
    ("Er", "Erbium"),
    ("Tm", "Thulium"),
    ("Yb", "Ytterbium"),
    ("Lu", "Lutetium"),
]
METAL_MODE_OPTIONS = [
    ("Leave unchanged", "Leave all detected metal sites unchanged."),
    ("Replace all sites", "Replace all detected metal sites with one supported metal."),
    ("Replace selected sites", "Replace only the selected metal sites."),
    ("Remove all sites", "Remove all detected metal sites."),
    ("Remove selected sites", "Remove only the selected metal sites."),
]
METAL_BATCH_STRATEGY_OPTIONS = [
    (
        "one_site_only",
        "One site only",
        "Choose one site to fan out across multiple metals. All other detected sites stay unchanged.",
    ),
    (
        "sites_together",
        "Sites together",
        "Replace all selected sites with the same chosen metal in each generated output.",
    ),
    (
        "full_combinations",
        "Full combinations",
        "Generate the full Cartesian product across the selected sites and chosen metals.",
    ),
]
METAL_INSERTION_DEFAULT_CN = {
    "Co": 6,
    "Cu": 4,
    "Ni": 6,
    "Mn": 6,
    "Fe": 6,
    "Sc": 8,
    "Y": 8,
    "La": 9,
    "Ce": 9,
    "Pr": 9,
    "Nd": 9,
    "Pm": 9,
    "Sm": 9,
    "Eu": 8,
    "Gd": 8,
    "Tb": 8,
    "Dy": 8,
    "Ho": 8,
    "Er": 8,
    "Tm": 8,
    "Yb": 8,
    "Lu": 8,
}


@dataclass(slots=True)
class WizardChoice:
    key: str
    label: str
    description: str
    enabled: bool = True


def _prompt_enum(message: str, enum_cls, default):
    value = typer.prompt(message, default=default.value if hasattr(default, "value") else str(default))
    return enum_cls(value)


def _prompt_csv(message: str) -> list[str]:
    raw = typer.prompt(_back_prompt_suffix(message), default="").strip()
    if _is_back_token(raw):
        raise WizardBack()
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_ph_value(value: float) -> float:
    normalized = round(float(value), 6)
    rounded = round(normalized)
    if math.isclose(normalized, rounded, abs_tol=1.0e-6):
        return float(rounded)
    return normalized


def _format_ph_display(ph: float) -> str:
    normalized = _normalize_ph_value(ph)
    if math.isclose(normalized, round(normalized), abs_tol=1.0e-6):
        return str(int(round(normalized)))
    return f"{normalized:.6f}".rstrip("0").rstrip(".")


def _ph_suffix_token(ph: float) -> str:
    return "PH" + _format_ph_display(ph).replace(".", "p")


def _parse_ph_values(raw: str) -> list[float]:
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        raise ValueError("Please enter at least one pH value.")

    normalized_values: list[float] = []
    seen: set[float] = set()
    for token in tokens:
        try:
            parsed = float(token)
        except ValueError as exc:
            raise ValueError("pH selections must be numeric values separated by commas.") from exc
        if parsed <= 0:
            raise ValueError("pH values must be positive numbers.")
        normalized = _normalize_ph_value(parsed)
        if normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return sorted(normalized_values)


def _prompt_ph_values() -> list[float]:
    while True:
        raw = typer.prompt(
            _back_prompt_suffix("Target pH value(s) for PROPKA analysis (comma separated)"),
            default="7",
        ).strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            return _parse_ph_values(raw)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")


def _workspace_detectable_resp_candidates() -> list[object]:
    candidates: list[object] = []
    seen: set[Path] = set()
    workspace = Path.cwd().resolve()
    manifest_paths = (
        workspace.rglob("resp_apply_manifest.json")
        if search_subdirectories_enabled()
        else (
            path
            for path in (
                workspace / "resp_apply_manifest.json",
                workspace / "manifests" / "resp_apply_manifest.json",
            )
            if path.is_file()
        )
    )
    for manifest_path in manifest_paths:
        candidate = load_resp_job_candidate(manifest_path.parent.parent)
        if candidate is None or not getattr(candidate, "ready_to_continue", False):
            continue
        if candidate.job_dir in seen:
            continue
        seen.add(candidate.job_dir)
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            1 if getattr(item, "ready_to_continue", False) else 0,
            1 if getattr(item, "completed", False) else 0,
            str(getattr(item, "payload", {}).get("created_at") or ""),
        ),
        reverse=True,
    )
    return candidates


def _workspace_detectable_protein_site_resp_candidates() -> list[ProteinSiteRespResumeCandidate]:
    return find_protein_site_resp_resume_candidates(
        Path.cwd(),
        recursive=search_subdirectories_enabled(),
    )


def _display_protein_site_resp_resume_candidates(
    candidates: list[ProteinSiteRespResumeCandidate],
) -> None:
    table = Table(title="Existing completed protein-site RESP jobs", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Protein / site", style="bold white", overflow="fold")
    table.add_column("Result", style="green", overflow="fold")
    table.add_column("Workflow output", style="cyan", overflow="fold")
    table.add_column("RESP job", style="dim", overflow="fold")
    for index, candidate in enumerate(candidates, start=1):
        payload = candidate.payload
        metal_states = list(payload.get("formal_metal_states") or [])
        metals = ", ".join(
            f"{item.get('element', '?')}{int(item.get('formal_charge') or 0):+d}"
            for item in metal_states
            if isinstance(item, dict)
        )
        donors = ", ".join(str(item) for item in payload.get("donor_residues") or [])
        site_text = candidate.source_label
        if metals:
            site_text += f" | {metals}"
        if donors:
            site_text += f" | {donors}"
        table.add_row(
            str(index),
            site_text,
            candidate.result_kind,
            str(candidate.workflow_root),
            str(candidate.job_dir),
        )
    console.print(table)


def _parse_protein_site_resp_resume_numbers(
    raw: str,
    candidates: list[ProteinSiteRespResumeCandidate],
) -> list[ProteinSiteRespResumeCandidate] | None:
    token = raw.strip().lower()
    if token in {"n", "no", "new", "fresh"}:
        return None
    if token in {"a", "all", "*"}:
        return list(candidates)
    selected_indices: list[int] = []
    for item in re.split(r"[\s,]+", token):
        if not item:
            continue
        if not item.isdigit() or not (1 <= int(item) <= len(candidates)):
            raise ValueError(
                f"Choose A for all, N for a new protein, or job number(s) from 1 to {len(candidates)}."
            )
        index = int(item)
        if index not in selected_indices:
            selected_indices.append(index)
    if not selected_indices:
        raise ValueError("Choose at least one RESP job, A for all, or N for a new protein.")
    return [candidates[index - 1] for index in selected_indices]


def _prompt_protein_site_resp_resume_selection(
    candidates: list[ProteinSiteRespResumeCandidate],
) -> _ProteinSiteRespResumeSelection | None:
    if not candidates:
        return None
    _display_protein_site_resp_resume_candidates(candidates)
    if not typer.confirm(
        "Completed protein-site RESP job(s) were found. Continue from existing RESP results?",
        default=True,
    ):
        console.print("[dim]Starting the normal new-protein workflow.[/dim]")
        return None
    if len(candidates) == 1:
        return _ProteinSiteRespResumeSelection(candidates=[candidates[0]])

    while True:
        raw = typer.prompt(
            "Select A for all jobs, one job number, comma-separated job numbers, or N for a new protein",
            default="A",
        )
        try:
            selected = _parse_protein_site_resp_resume_numbers(raw, candidates)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        if selected is None:
            console.print("[dim]Starting the normal new-protein workflow.[/dim]")
            return None
        return _ProteinSiteRespResumeSelection(candidates=selected)


def _materialize_resp_resume_input_file(candidate: object) -> str:
    payload = getattr(candidate, "payload", {}) or {}
    source_file = str(payload.get("source_file") or "").strip()
    if source_file:
        candidate_path = Path(source_file).expanduser()
        if candidate_path.exists():
            return str(candidate_path.resolve())

    resume_source_file = str(payload.get("resume_source_file") or "").strip()
    if resume_source_file:
        resume_path = Path(resume_source_file).expanduser()
        if resume_path.exists():
            return str(resume_path.resolve())

    canonical_source_file = str(payload.get("canonical_source_file") or "").strip()
    if canonical_source_file:
        canonical_path = Path(canonical_source_file).expanduser()
        if canonical_path.exists():
            return str(canonical_path.resolve())

    popup_state_path = Path(candidate.job_dir) / "manifests" / "popup_state.json"
    if popup_state_path.exists():
        session_state = json.loads(popup_state_path.read_text(encoding="utf-8"))
        preview = str(session_state.get("mol2_preview") or "").strip()
        if preview:
            resume_input = Path(candidate.job_dir) / "inputs" / "resp_resume_input.mol2"
            preview_text = preview if preview.endswith("\n") else preview + "\n"
            if not resume_input.exists() or resume_input.read_text(encoding="utf-8") != preview_text:
                resume_input.write_text(preview_text, encoding="utf-8")
            return str(resume_input.resolve())

    raise RuntimeError(
        "The selected RESP job does not have an accessible original input file or a saved preview MOL2, "
        "so SIMPLE cannot continue from it yet."
    )


def _input_config_from_resp_candidate(candidate: object) -> InputConfig:
    return InputConfig(
        source=InputSource.SMALL_MOLECULE,
        small_molecule_files=[_materialize_resp_resume_input_file(candidate)],
    )


def _prompt_small_molecule_start_mode(*, has_detected_resp_candidates: bool) -> str:
    if not has_detected_resp_candidates:
        return "input_files"
    choices = [
        WizardChoice(
            "input_files",
            "Input file(s)",
            "Provide local small-molecule file(s) such as TEST.pdb, ligand.sdf, ligand.mol2, or ligand.smi.",
        ),
        WizardChoice(
            "resp_continue",
            "Continue RESP",
            "Browse detected RESP job/result folders in this workspace and continue from one by number, even if the original input file is gone.",
        ),
    ]
    _display_choice_table("Small-molecule start mode", choices)
    return _prompt_choice(
        "Choose how to start the small-molecule workflow",
        choices,
        default_key="input_files",
    )


def _prompt_workspace_resp_resume_option() -> object | None:
    candidates = _workspace_detectable_resp_candidates()
    if not candidates:
        return None

    choices = [
        WizardChoice(
            "input_files",
            "Use input file(s) instead",
            "Go back and provide local small-molecule structure file(s) for a fresh run or source-matched RESP detection.",
        )
    ]
    candidate_lookup: dict[str, object] = {}
    for index, candidate in enumerate(candidates, start=1):
        payload = getattr(candidate, "payload", {}) or {}
        choice_key = f"resume_{index}"
        source_file = str(payload.get("source_file") or "").strip()
        source_name = Path(source_file).name if source_file else "source unavailable"
        has_source_file = bool(source_file and Path(source_file).expanduser().exists())
        source_text = (
            f"source file {source_name}"
            if has_source_file
            else f"source file {source_name} missing; stored RESP preview will be used"
        )
        residue_name = str(payload.get("residue_name") or "LIG")
        net_charge = int(payload.get("net_charge") or 0)
        multiplicity = int(payload.get("multiplicity") or 1)
        created_at = str(payload.get("created_at") or "unknown time")
        status_text = (
            "RESP charge files are ready to apply."
            if getattr(candidate, "completed", False)
            else "RESP outputs are present and the charges will be refit locally before Amber setup continues."
        )
        description = (
            f"{source_text}; {status_text} Residue {residue_name}, charge {net_charge}, "
            f"multiplicity {multiplicity}, created {created_at}, job dir {candidate.job_dir}."
        )
        choices.append(WizardChoice(choice_key, f"Continue from RESP #{index}", description))
        candidate_lookup[choice_key] = candidate

    _display_choice_table("Detected RESP job/result folders in this workspace", choices)
    selected = _prompt_choice(
        "Choose a RESP folder to continue from",
        choices,
        default_key=choices[1].key if len(choices) > 1 else "input_files",
    )
    if selected == "input_files":
        return None
    return candidate_lookup[selected]


def _parse_workflow_type_choice(raw: str) -> InputSource | str:
    token = raw.strip().lower()
    if token in {"a", "add", "add-component", "add_component", "library"}:
        return ADD_DES_LIBRARY_MODE
    if token in {"d", "des", "deep", "deep-eutectic", "deep_eutectic_solvent"}:
        return InputSource.DES
    if token in {"p", "protein"}:
        return InputSource.PDB_FILE
    if token in {"s", "small", "small-molecule", "small_molecule"}:
        return InputSource.SMALL_MOLECULE
    raise ValueError(
        "Please enter D for Deep Eutectic Solvent, S for Small-molecule, P for Protein, "
        "or A to add a DES component library."
    )


def _print_step_header(step_number: int, title: str, description: str) -> None:
    body = Text()
    body.append(f"{title}\n", style="bold white")
    body.append(description, style="cyan")
    console.print()
    console.print(
        Panel(
            body,
            title=f"[bold black on bright_cyan] Step {step_number} [/] ",
            border_style="bright_cyan",
            box=box.HEAVY,
            padding=(1, 2),
        )
    )


def _display_choice_table(title: str, choices: list[WizardChoice]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Choice", style="bold white")
    table.add_column("Description", style="white")
    if any(not choice.enabled for choice in choices):
        table.add_column("Status", style="yellow")
    for index, choice in enumerate(choices, start=1):
        row = [str(index), choice.label, choice.description]
        if any(not item.enabled for item in choices):
            row.append("Available" if choice.enabled else "Unavailable")
        table.add_row(*row)
    console.print(table)


def _is_back_token(raw: str) -> bool:
    return raw.strip().lower() in BACK_TOKENS


def _back_prompt_suffix(message: str) -> str:
    return f"{message} [B=back]"


def _prompt_yes_no(message: str, *, default: bool = True) -> bool:
    default_text = "Y" if default else "N"
    while True:
        raw = typer.prompt(_back_prompt_suffix(f"{message} [y/n]"), default=default_text).strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw == "":
            return default
        token = raw.lower()
        if token in {"y", "yes"}:
            return True
        if token in {"n", "no"}:
            return False
        console.print("[bold red]Please enter Y, N, or B.[/bold red]")


def _default_choice_index(choices: list[WizardChoice], default_key: str | None) -> int:
    if default_key is not None:
        for index, choice in enumerate(choices, start=1):
            if choice.key == default_key and choice.enabled:
                return index
    for index, choice in enumerate(choices, start=1):
        if choice.enabled:
            return index
    return 1


def _prompt_choice(message: str, choices: list[WizardChoice], *, default_key: str | None = None) -> str:
    default_index = _default_choice_index(choices, default_key)
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=str(default_index)).strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(choices):
                choice = choices[selected - 1]
                if not choice.enabled:
                    console.print("[bold yellow]That option is listed for reference but is not selectable yet.[/bold yellow]")
                    continue
                return choice.key
        console.print(f"[bold red]Please enter a number between 1 and {len(choices)}, or B to go back.[/bold red]")


def _validate_existing_paths(paths: list[str], *, label: str) -> bool:
    missing = [item for item in paths if not Path(item).expanduser().exists()]
    if not missing:
        return True
    console.print(f"[bold red]The following {label} do not exist:[/bold red]")
    for item in missing:
        console.print(f"  - {item}")
    return False


def _parse_site_selection(raw: str, available_sites: set[int]) -> list[int]:
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        raise ValueError("Please enter at least one site number.")

    selected: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError("Site selections must be integers separated by commas.")
        site = int(token)
        if site not in available_sites:
            available_text = ", ".join(str(item) for item in sorted(available_sites))
            raise ValueError(f"Site {site} is not valid. Available sites: {available_text}.")
        if site not in selected:
            selected.append(site)
    return selected


def _prompt_site_selection(action: str, available_sites: set[int]) -> list[int]:
    while True:
        raw = typer.prompt(_back_prompt_suffix(f"Enter site number(s) to {action} (comma separated)"), default="").strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            return _parse_site_selection(raw, available_sites)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")


def _parse_index_selection(raw: str, available_indices: set[int], *, label: str) -> list[int]:
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        return []

    selected: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"{label} selections must be integers separated by commas.")
        index = int(token)
        if index not in available_indices:
            available_text = ", ".join(str(item) for item in sorted(available_indices))
            raise ValueError(f"{label} {index} is not valid. Available {label.lower()}s: {available_text}.")
        if index not in selected:
            selected.append(index)
    return selected


def _prompt_replacement_metal() -> str:
    return _prompt_replacement_metals()[0]


def _prompt_multi_choice(
    message: str,
    choices: list[WizardChoice],
    *,
    default_keys: list[str] | tuple[str, ...] | None = None,
    label: str,
) -> list[str]:
    numbered_choices = {
        index: choice
        for index, choice in enumerate(choices, start=1)
        if choice.enabled
    }
    if not numbered_choices:
        raise ValueError(f"No selectable {label.lower()} options are available.")

    default_numbers = [
        number
        for number, choice in numbered_choices.items()
        if default_keys is not None and choice.key in default_keys
    ]
    default = ",".join(str(number) for number in default_numbers) or str(min(numbered_choices))
    available_numbers = set(numbered_choices)
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=default).strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            selected_numbers = _parse_index_selection(raw, available_numbers, label=label)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        if not selected_numbers:
            console.print(f"[bold red]Please choose at least one {label.lower()}.[/bold red]")
            continue
        return [numbered_choices[number].key for number in selected_numbers]


def _des_component_key(component: DESComponent | str) -> str:
    return component.value if isinstance(component, DESComponent) else str(component)


def _des_component_from_key(key: str) -> DESComponent | str:
    try:
        return DESComponent(key)
    except ValueError:
        return key


def _des_component_map() -> dict[DESComponent | str, object]:
    return available_des_components()


def _des_component_label(component: DESComponent | str) -> str:
    return _des_component_map()[component].label


def _des_component_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            _des_component_key(component),
            definition.label,
            definition.description,
        )
        for component, definition in _des_component_map().items()
    ]


def _display_des_recommended_sets() -> None:
    table = Table(title="Recommended DES sets", box=box.SIMPLE_HEAVY)
    table.add_column("Set", style="bold cyan")
    table.add_column("Components", style="bold white")
    table.add_column("Default ratio", style="green", justify="center", no_wrap=True)
    for index, (components, ratio) in enumerate(DES_RECOMMENDED_SETS):
        labels = [DES_COMPONENTS[component].label for component in components]
        table.add_row(f"R{index + 1}", " : ".join(labels), ":".join(str(value) for value in ratio))
    console.print(table)


def _prompt_des_component_selection() -> list[DESComponent | str]:
    choices = _des_component_choices()
    numbered_choices = {index: choice for index, choice in enumerate(choices, start=1)}
    set_lookup = {
        f"r{index + 1}": list(components)
        for index, (components, _ratio) in enumerate(DES_RECOMMENDED_SETS)
    }
    available_numbers = set(numbered_choices)
    while True:
        raw = typer.prompt(
            "Choose DES component number(s), recommended set R1/R2, or B to go back (examples: 1,2 | 1,2,3,4 | R1)",
            default="R1",
        ).strip()
        if _is_back_token(raw):
            raise WizardBack()
        token = raw.lower()
        if token in set_lookup:
            return set_lookup[token]
        try:
            selected_numbers = _parse_index_selection(raw, available_numbers, label="DES component")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        if not selected_numbers:
            console.print("[bold red]Please choose at least one DES component or recommended set.[/bold red]")
            continue
        return [_des_component_from_key(numbered_choices[number].key) for number in selected_numbers]


def _parse_des_ratio(raw: str, expected_count: int) -> list[int]:
    normalized = raw.replace(",", ":")
    tokens = [token.strip() for token in normalized.split(":") if token.strip()]
    if len(tokens) != expected_count:
        raise ValueError(f"Please enter exactly {expected_count} positive integer ratio value(s).")
    ratios: list[int] = []
    for token in tokens:
        if not token.isdigit() or int(token) < 1:
            raise ValueError("DES ratios must be positive integers separated by ':' or commas.")
        ratios.append(int(token))
    return ratios


def _prompt_des_ratio(components: list[DESComponent | str]) -> list[int]:
    if len(components) == 1:
        return [1]
    default_ratios = recommended_ratio_for_components(components)
    default_text = ":".join(str(value) for value in default_ratios)
    label_text = " : ".join(_des_component_label(component) for component in components)
    while True:
        raw = typer.prompt(_back_prompt_suffix(f"Molar ratio for {label_text}"), default=default_text).strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            return _parse_des_ratio(raw, len(components))
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")


def _display_des_plan(config: DESConfig) -> None:
    plan = estimate_des_plan(config)
    table = Table(title="DES build preview", box=box.SIMPLE_HEAVY)
    table.add_column("Component", style="bold white")
    table.add_column("Ratio", style="cyan", justify="right", no_wrap=True)
    table.add_column("Count", style="green", justify="right", no_wrap=True)
    for component, ratio in zip(config.components, config.ratios, strict=True):
        table.add_row(
            _des_component_label(component),
            str(ratio),
            str(plan.component_counts[_des_component_key(component)]),
        )
    console.print(table)

    residue_table = Table(title="Expanded residue counts", box=box.SIMPLE)
    residue_table.add_column("Residue", style="bold cyan")
    residue_table.add_column("Count", style="white", justify="right", no_wrap=True)
    for residue_name, count in plan.residue_counts.items():
        residue_table.add_row(residue_name, str(count))
    console.print(residue_table)

    if plan.metal_sites:
        metal_table = Table(title="DES metal placements", box=box.SIMPLE)
        metal_table.add_column("No.", style="bold cyan", justify="right")
        metal_table.add_column("Metal", style="bold white")
        metal_table.add_column("Residue", style="cyan")
        metal_table.add_column("XYZ (A)", style="white")
        metal_table.add_column("Placement", style="white")
        for entry in plan.metal_sites:
            metal_table.add_row(
                str(entry.get("index", "")),
                f"{entry.get('element', '?')}+{entry.get('charge', '?')}",
                str(entry.get("residue_name", "")),
                f"{float(entry.get('x', 0.0)):.3f}, {float(entry.get('y', 0.0)):.3f}, {float(entry.get('z', 0.0)):.3f}",
                str(entry.get("placement", "")),
            )
        console.print(metal_table)

    details = [
        f"Mixing mode: [bold]{plan.mixing_mode}[/bold]",
        f"Ratio units: [bold]{plan.ratio_units}[/bold]",
        "Estimated box lengths: "
        f"[bold]{plan.box_lengths_angstrom[0]:.2f} x {plan.box_lengths_angstrom[1]:.2f} x "
        f"{plan.box_lengths_angstrom[2]:.2f} A[/bold]",
        f"Estimated box volume: [bold]{plan.box_volume_angstrom3:.1f} A^3[/bold]",
        f"Estimated initial density: [bold]{plan.estimated_initial_density_g_ml:.3f} g/mL[/bold]",
        f"Total residues: [bold]{plan.total_residues}[/bold]",
        f"Estimated atoms: [bold]{plan.total_atoms}[/bold]",
        f"12-6-4 parameter set: [bold]{config.c4_parameter_set.value}[/bold]",
        f"12-6-4 C4 mask: [bold]{plan.c4_mask or 'not requested'}[/bold]",
    ]
    if config.mixing_mode == DESMixingMode.PACKMOL:
        details.insert(2, f"Packmol tolerance: [bold]{plan.packmol_tolerance_angstrom:.2f} A[/bold]")
        if config.size_mode == DESSizeMode.BOX_LENGTH:
            details.insert(3, f"Initial fill fraction: [bold]{plan.packmol_fill_fraction:.2f}[/bold]")
    console.print(
        Panel(
            "\n".join(details),
            title="[bold cyan]DES Output Preview[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    if config.mixing_mode == DESMixingMode.PACKMOL and (plan.total_atoms > 100000 or plan.total_residues > 2500):
        console.print(
            Panel(
                "This Packmol system is large enough to take many minutes or longer, especially with tight "
                "tolerances such as 1.5 A. Consider fewer ratio units, a lower fill fraction, or a looser "
                "2.5-3.0 A tolerance followed by NPT density relaxation.",
                title="[bold yellow]Packmol runtime warning[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )


def _parse_xyz_triplet(raw: str) -> list[float]:
    tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
    if len(tokens) != 3:
        raise ValueError("Enter exactly three coordinates: x y z")
    return [float(token) for token in tokens]


def _prompt_des_metal_sites() -> list[DESMetalSiteConfig]:
    if not _prompt_yes_no("Add one or more REE/metal ions to the DES box?", default=False):
        return []
    metal_sites: list[DESMetalSiteConfig] = []
    while True:
        element = _prompt_insertion_metal()
        charges = allowed_metal_charges(element)
        default_charge = charges[0] if charges else 3
        charge = _prompt_positive_int(f"{element} formal charge", default=default_charge)
        count = _prompt_positive_int(f"Number of {element}+{charge} ions to add", default=1)
        coordinates: list[list[float]] | None = None
        if _prompt_yes_no("Enter explicit XYZ coordinates for this metal group?", default=False):
            coordinates = []
            for index in range(count):
                while True:
                    raw = typer.prompt(
                        _back_prompt_suffix(f"XYZ for {element}+{charge} ion {index + 1} (x y z, Angstrom)")
                    ).strip()
                    if _is_back_token(raw):
                        raise WizardBack()
                    try:
                        coordinates.append(_parse_xyz_triplet(raw))
                        break
                    except ValueError as exc:
                        console.print(f"[bold red]{exc}[/bold red]")
        metal_sites.append(
            DESMetalSiteConfig(
                element=element,
                charge=charge,
                count=count,
                coordinates=coordinates,
            )
        )
        if not _prompt_yes_no("Add another DES metal group?", default=False):
            return metal_sites


@dataclass(slots=True)
class _DESPromptState:
    components: list[DESComponent | str] = field(default_factory=list)
    ratios: list[int] = field(default_factory=list)
    mixing_mode: DESMixingMode = DESMixingMode.RANDOM_MIX
    size_mode: DESSizeMode = DESSizeMode.RATIO_UNITS
    ratio_units: int | None = None
    box_length: float | None = None
    replicate_order: DESReplicateOrder = DESReplicateOrder.UNIFORM
    spacing: float = DESConfig().spacing_angstrom
    tolerance: float = 2.0
    packmol_fill_fraction: float = DESConfig().packmol_fill_fraction
    target_density_g_ml: float = DESConfig().target_density_g_ml
    metal_sites: list[DESMetalSiteConfig] = field(default_factory=list)
    c4_parameter_set: DESC4ParameterSet = DESC4ParameterSet.OPC_DUVAIL
    apply_1264: bool = False


def _des_config_from_prompt_state(state: _DESPromptState) -> DESConfig:
    return DESConfig(
        components=state.components,
        ratios=state.ratios,
        mixing_mode=state.mixing_mode,
        replicate_order=state.replicate_order,
        size_mode=state.size_mode,
        ratio_units=state.ratio_units,
        box_length_angstrom=state.box_length,
        spacing_angstrom=state.spacing,
        packmol_tolerance_angstrom=state.tolerance,
        packmol_fill_fraction=state.packmol_fill_fraction,
        target_density_g_ml=state.target_density_g_ml,
        apply_1264=state.apply_1264,
        c4_parameter_set=state.c4_parameter_set,
        metal_sites=state.metal_sites,
    )


def _prompt_des_config() -> DESConfig:
    title = Text()
    title.append("Deep Eutectic Solvent Builder\n", style="bold bright_cyan")
    title.append("Choose REF_DATA components, ratio, and initial packing.", style="bright_green")
    console.print(Panel(title, border_style="bright_magenta", box=box.DOUBLE, padding=(1, 2)))
    _display_choice_table("[bold bright_cyan]Deep Eutectic Solvent Components[/bold bright_cyan]", _des_component_choices())
    _display_des_recommended_sets()
    mixing_choices = [
        WizardChoice(
            DESMixingMode.RANDOM_MIX.value,
            "Random mix (fast)",
            "Fast built-in rigid-body packing with random rotations and periodic contact checks.",
        ),
        WizardChoice(
            DESMixingMode.PACKMOL.value,
            "Packmol (thorough)",
            "Use Packmol for a slower packing optimization. Requires `packmol` on PATH.",
        ),
    ]
    size_choices = [
        WizardChoice(
            DESSizeMode.RATIO_UNITS.value,
            "Molecule count",
            "Enter the number of ratio units; SIMPLE estimates a cubic box.",
        ),
        WizardChoice(
            DESSizeMode.BOX_LENGTH.value,
            "Cubic box length",
            "Enter X=Y=Z box length; SIMPLE estimates the largest ratio-consistent count.",
        ),
    ]
    c4_parameter_choices = [
        WizardChoice(
            DESC4ParameterSet.OPC_DUVAIL.value,
            "OPC + Duvail",
            "Default. Use bundled Duvail frcmod/c4file for DES ion residues and metal C4 post-processing; no water is added.",
        ),
        WizardChoice(
            DESC4ParameterSet.SPCE_LIMERZ.value,
            "SPCE + Li/Merz",
            "Legacy behavior. Use existing Amber/REF_DATA SPCE Li/Merz-style assets when available.",
        ),
    ]
    state = _DESPromptState()
    step = 0
    while True:
        back_step: int | None = None
        try:
            if step == 0:
                back_step = -1
                state.components = _prompt_des_component_selection()
                step = 1
                continue

            if step == 1:
                back_step = 0
                state.ratios = _prompt_des_ratio(state.components)
                step = 2
                continue

            if step == 2:
                _display_choice_table("DES mixing mode", mixing_choices)
                back_step = 1
                state.mixing_mode = DESMixingMode(
                    _prompt_choice(
                        "Choose the DES mixing mode",
                        mixing_choices,
                        default_key=state.mixing_mode.value,
                    )
                )
                step = 3
                continue

            if step == 3:
                _display_choice_table("DES system size mode", size_choices)
                back_step = 2
                state.size_mode = DESSizeMode(
                    _prompt_choice(
                        "Choose how to size the DES box",
                        size_choices,
                        default_key=state.size_mode.value,
                    )
                )
                state.ratio_units = None
                state.box_length = None
                back_step = 3
                if state.size_mode == DESSizeMode.RATIO_UNITS:
                    state.ratio_units = _prompt_positive_int("Number of DES ratio units", default=100)
                else:
                    state.box_length = _prompt_positive_float("Cubic PBC box length X=Y=Z (Angstrom)", default=80.0)
                step = 4
                continue

            if step == 4:
                state.target_density_g_ml = _prompt_positive_float(
                    "Safe initial DES density (g/mL)",
                    default=DESConfig().target_density_g_ml,
                )
                if state.mixing_mode == DESMixingMode.PACKMOL:
                    back_step = 3
                    state.tolerance = _prompt_positive_float("Packmol tolerance (Angstrom)", default=2.0)
                    state.spacing = state.tolerance
                    state.replicate_order = DESReplicateOrder.UNIFORM
                    state.packmol_fill_fraction = DESConfig().packmol_fill_fraction
                    if state.size_mode == DESSizeMode.BOX_LENGTH:
                        console.print(
                            "[dim]Packmol box-length mode estimates molecule count from mass and the requested density. "
                            "Use the default fill fraction 1.0 for that density; lower it if you want a less packed "
                            "starting box or a lighter Packmol job.[/dim]"
                        )
                        back_step = 4
                        state.packmol_fill_fraction = _prompt_fraction(
                            "Packmol initial fill fraction (0.10-1.00)",
                            default=state.packmol_fill_fraction,
                        )
                else:
                    state.packmol_fill_fraction = DESConfig().packmol_fill_fraction
                    back_step = 3
                    state.replicate_order = DESReplicateOrder.RANDOM
                    back_step = 4
                    state.spacing = _prompt_positive_float(
                        "Random-mix minimum inter-residue atom distance (Angstrom, >=1.2)",
                        default=DESConfig().spacing_angstrom,
                    )
                    state.tolerance = 2.0
                step = 5
                continue

            if step == 5:
                back_step = 4
                state.metal_sites = _prompt_des_metal_sites()
                step = 6
                continue

            if step == 6:
                _display_choice_table("DES 12-6-4 parameter set", c4_parameter_choices)
                back_step = 5
                state.c4_parameter_set = DESC4ParameterSet(
                    _prompt_choice(
                        "Choose the DES 12-6-4 parameter set",
                        c4_parameter_choices,
                        default_key=state.c4_parameter_set.value,
                    )
                )
                state.apply_1264 = True
                candidate_config = _des_config_from_prompt_state(state)
                candidate_plan = estimate_des_plan(candidate_config)
                if candidate_plan.c4_residue_names:
                    ions_text = ", ".join(candidate_plan.c4_residue_names)
                    back_step = 6
                    state.apply_1264 = _prompt_yes_no(
                        f"12-6-4-capable ion residue(s) detected ({ions_text}). Apply available 12-6-4 C4 parameters?",
                        default=True,
                    )
                else:
                    console.print("[dim]No ion residues with available 12-6-4 handling were detected for this DES selection.[/dim]")
                    state.apply_1264 = False
                step = 7
                continue

            config = _des_config_from_prompt_state(state)
            _display_des_plan(config)
            back_step = 6
            if _prompt_yes_no("Proceed with this DES build plan?", default=True):
                return config
            console.print("[dim]Let's adjust the DES component, ratio, or packing settings.[/dim]")
            step = 0
        except WizardBack:
            target_step = step - 1 if back_step is None else back_step
            if target_step < 0:
                raise
            step = target_step
            console.print(
                "[dim]Back: returning to the previous DES setup step. "
                "Use B again at the first DES step to return to workflow type selection.[/dim]"
            )


def _replacement_metal_choices() -> list[WizardChoice]:
    allowed = {metal.title() for metal in SUPPORTED_METALS}
    return [
        WizardChoice(symbol, symbol, f"Replace the selected site(s) with {name}.")
        for symbol, name in METAL_REPLACEMENT_CHOICES
        if symbol in allowed
    ]


def _prompt_replacement_metals() -> list[str]:
    choices = _replacement_metal_choices()
    _display_choice_table("Replacement metal options", choices)
    return _prompt_multi_choice(
        "Choose one or more replacement metals by number (comma separated)",
        choices,
        default_keys=["Co"],
        label="Metal",
    )


def _prompt_replacement_strategy(summary, selected_sites: list[int]) -> tuple[str, list[int]]:
    if len(selected_sites) <= 1:
        return "one_site_only", selected_sites

    choices = [
        WizardChoice(key, label, description)
        for key, label, description in METAL_BATCH_STRATEGY_OPTIONS
    ]
    _display_choice_table("How should the selected metal sites be expanded?", choices)
    strategy = _prompt_choice(
        "Choose how to expand the selected metal sites",
        choices,
        default_key="sites_together",
    )
    if strategy != "one_site_only":
        return strategy, sorted(selected_sites)

    site_lookup = {site.site: site for site in summary.metals}
    site_choices = [
        WizardChoice(
            str(site_number),
            f"Site {site_number}",
            f"{site_lookup[site_number].key} ({site_lookup[site_number].element.title()})",
        )
        for site_number in sorted(selected_sites)
    ]
    _display_choice_table("Choose the site to fan out across multiple metals", site_choices)
    chosen_site = int(
        _prompt_choice(
            "Choose the single site to fan out across multiple metals",
            site_choices,
            default_key=str(sorted(selected_sites)[0]),
        )
    )
    return strategy, [chosen_site]


def _copy_metal_replacements(replacements: list[MetalReplacement]) -> list[MetalReplacement]:
    return [MetalReplacement(site=item.site, target=item.target) for item in replacements]


def _expand_replacement_variants(
    selected_sites: list[int],
    selected_metals: list[str],
    *,
    strategy: str,
) -> list[_MetalReplacementVariant]:
    ordered_sites = sorted(selected_sites)
    if not ordered_sites:
        return [_MetalReplacementVariant(replacements=[], suffix_tokens=())]

    if len(ordered_sites) == 1 or strategy == "one_site_only":
        site = ordered_sites[0]
        return [
            _MetalReplacementVariant(
                replacements=[MetalReplacement(site=site, target=metal)],
                suffix_tokens=(metal,),
            )
            for metal in selected_metals
        ]

    if strategy == "sites_together":
        return [
            _MetalReplacementVariant(
                replacements=[MetalReplacement(site=site, target=metal) for site in ordered_sites],
                suffix_tokens=(metal,),
            )
            for metal in selected_metals
        ]

    if strategy == "full_combinations":
        return [
            _MetalReplacementVariant(
                replacements=[
                    MetalReplacement(site=site, target=metal)
                    for site, metal in zip(ordered_sites, metal_combo, strict=True)
                ],
                suffix_tokens=tuple(metal_combo),
            )
            for metal_combo in product(selected_metals, repeat=len(ordered_sites))
        ]

    raise ValueError(f"Unsupported metal replacement strategy: {strategy}")


def _variant_suffix_label(suffix_tokens: tuple[str, ...]) -> str:
    labels: list[str] = []
    for token in suffix_tokens:
        if not token:
            continue
        if token.upper().startswith("PH"):
            labels.append(token)
        else:
            labels.append(token.upper())
    return "_".join(labels)


def _variant_output_name(base_name: str, suffix_tokens: tuple[str, ...]) -> str:
    suffix_label = _variant_suffix_label(suffix_tokens)
    if not suffix_label:
        return base_name
    return f"{base_name}_{suffix_label}"


def _job_name_suffix_tokens(suffix_tokens: tuple[str, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for token in suffix_tokens:
        if not token:
            continue
        if token.upper().startswith("PH"):
            labels.append(token[2:] or token)
        else:
            labels.append(token.upper())
    return tuple(labels)


def _workflow_job_name(base_name: str, suffix_tokens: tuple[str, ...]) -> str:
    labels = _job_name_suffix_tokens(suffix_tokens)
    if not labels:
        return base_name
    return f"{base_name}_{'_'.join(labels)}"


def _preview_variant_names(input_config: InputConfig, variants: list[_MetalReplacementVariant]) -> list[str]:
    base_name = _base_output_name(input_config)
    return [_variant_output_name(base_name, variant.suffix_tokens) for variant in variants]


def _confirm_metal_batch_preview(input_config: InputConfig, variants: list[_MetalReplacementVariant]) -> bool:
    preview_names = _preview_variant_names(input_config, variants)
    preview_limit = 10
    preview_lines = [f"  - {name}" for name in preview_names[:preview_limit]]
    if len(preview_names) > preview_limit:
        preview_lines.append(f"  - ... {len(preview_names) - preview_limit} more")
    console.print(
        Panel(
            "SIMPLE will generate "
            f"{len(preview_names)} independent output folders that share the same non-metal settings.\n\n"
            + "\n".join(preview_lines)
            + "\n\nIf any folder already exists, an incremented suffix will be applied automatically.",
            title="[bold yellow]Batch Metal-Replacement Preview[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    return typer.confirm("Proceed with this batch metal-replacement plan?", default=True)


def _prompt_insertion_metal() -> str:
    choices = [
        WizardChoice(symbol, symbol, f"Insert a new {name} ion near selected donor atoms.")
        for symbol, name in METAL_REPLACEMENT_CHOICES
        if symbol in {metal.title() for metal in SUPPORTED_METALS}
    ]
    _display_choice_table("Insertion metal options", choices)
    return _prompt_choice("Choose the inserted metal", choices, default_key="Fe")


def _prompt_insertion_charge(element: str) -> int | None:
    charges = allowed_metal_charges(element)
    if not charges:
        return None
    choices = [WizardChoice(str(charge), f"+{charge}", _charge_choice_description(charge)) for charge in charges]
    _display_choice_table(f"Oxidation-state options for inserted {element}", choices)
    default_charge = str(DEFAULT_METAL_CHARGES.get(element, charges[0]))
    return int(_prompt_choice(f"Choose the oxidation state for inserted {element}", choices, default_key=default_charge))


def _prompt_insertion_target_cn(element: str) -> int | None:
    default_cn = METAL_INSERTION_DEFAULT_CN.get(element.title(), 6)
    while True:
        raw = typer.prompt(
            _back_prompt_suffix("Target coordination number, or 'm' for manual selected donors only"),
            default=str(default_cn),
        ).strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw.lower() in {"m", "manual"}:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        console.print("[bold red]Enter a positive integer coordination number or 'm'.[/bold red]")


def _parse_xyz_coordinates(raw: str) -> list[float]:
    tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    if len(tokens) != 3:
        raise ValueError("XYZ coordinates require exactly three comma-separated values.")
    return [float(token) for token in tokens]


def _atom_reference_matches_filter(row: dict[str, object], filter_text: str) -> bool:
    if not filter_text:
        return True
    haystack = " ".join(str(value) for value in row.values()).lower()
    return filter_text.lower() in haystack


def _browse_structure_atoms(structure, *, page_size: int = 90, columns: int = 3) -> None:
    rows = structure_atom_reference_rows(structure)
    page = 0
    filter_text = ""
    while True:
        visible = [row for row in rows if _atom_reference_matches_filter(row, filter_text)]
        if not visible:
            console.print("[bold yellow]No atoms match the current filter.[/bold yellow]")
            page = 0
        total_pages = max(1, math.ceil(len(visible) / page_size))
        page = min(max(page, 0), total_pages - 1)
        start = page * page_size
        chunk = visible[start : start + page_size]
        tables = []
        subpage_size = max(1, math.ceil(page_size / max(columns, 1)))
        for column_index in range(columns):
            subchunk = chunk[column_index * subpage_size : (column_index + 1) * subpage_size]
            if not subchunk:
                continue
            table = Table(
                title=f"{start + column_index * subpage_size + 1}-{start + column_index * subpage_size + len(subchunk)}",
                box=box.SIMPLE,
            )
            table.add_column("Ser", justify="right", style="bold cyan", no_wrap=True)
            table.add_column("Ch", no_wrap=True)
            table.add_column("Res", justify="right", no_wrap=True)
            table.add_column("Name", no_wrap=True)
            table.add_column("Atom", style="bold white", no_wrap=True)
            table.add_column("El", no_wrap=True)
            for row in subchunk:
                table.add_row(
                    str(row["atom_serial"]),
                    str(row["chain"] or "_"),
                    str(row["resid"] or row["seqid"]),
                    str(row["residue_name"]),
                    str(row["atom_name"]),
                    str(row["element"]),
                )
            tables.append(table)
        console.print(
            f"[bold cyan]Atom reference list[/bold cyan] "
            f"({len(visible)} shown; page {page + 1}/{total_pages}; filter: {filter_text or 'none'})"
        )
        console.print(Columns(tables, equal=True, expand=True))
        response = typer.prompt(
            "Atom list: Enter=next, p=previous, q=return, /text=filter",
            default="",
        ).strip()
        if response == "":
            if page + 1 >= total_pages:
                console.print("[dim]End of list. Returning to anchor input.[/dim]")
                return
            page += 1
            continue
        if response.lower() == "p":
            page = max(page - 1, 0)
            continue
        if response.lower() == "q":
            return
        if response.startswith("/"):
            filter_text = response[1:].strip()
            page = 0
            continue
        console.print("[bold red]Use Enter, p, q, or /text.[/bold red]")


def _prompt_anchor_text(structure, message: str) -> str:
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default="").strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw.lower() == "l":
            _browse_structure_atoms(structure)
            continue
        if raw:
            return raw
        console.print("[bold red]Please enter anchors, or type 'l' to browse atoms.[/bold red]")


def _display_donor_candidates(candidates: list[dict[str, object]]) -> None:
    if not candidates:
        console.print("[bold yellow]No donor candidates were found for the selected residues.[/bold yellow]")
        return
    table = Table(title="Resolved donor candidates", box=box.SIMPLE_HEAVY)
    table.add_column("Serial", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Selector", style="bold white")
    table.add_column("Element", no_wrap=True)
    table.add_column("Residue")
    for item in candidates:
        table.add_row(
            str(item["atom_serial"]),
            str(item["selector"]),
            str(item["element"]),
            f"{item.get('chain') or '_'}:{item.get('resid') or item.get('seqid')} {item.get('residue_name')}",
        )
    console.print(table)


def _prompt_serial_subset_from_candidates(candidates: list[dict[str, object]]) -> list[str]:
    available = {int(item["atom_serial"]) for item in candidates}
    if not available:
        return []
    if _prompt_yes_no("Use all listed donor atoms for the inserted metal?", default=True):
        return [str(item["atom_serial"]) for item in candidates]
    while True:
        raw = typer.prompt(_back_prompt_suffix("Donor atom serial number(s) to use (comma separated)"), default="").strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            selected = _parse_index_selection(raw, available, label="Atom serial")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        if selected:
            return [str(item) for item in selected]
        console.print("[bold red]Choose at least one donor atom serial.[/bold red]")


def _display_resolved_metal_insertion(resolved) -> None:
    table = Table(title="Inserted metal preview", box=box.SIMPLE_HEAVY)
    table.add_column("Metal", style="bold cyan")
    table.add_column("Charge", justify="center", no_wrap=True)
    table.add_column("Target CN", justify="center", no_wrap=True)
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Z", justify="right")
    table.add_row(
        resolved.element,
        "-" if resolved.charge is None else f"+{resolved.charge}",
        "-" if resolved.target_coordination_number is None else str(resolved.target_coordination_number),
        f"{resolved.coordinates[0]:.3f}",
        f"{resolved.coordinates[1]:.3f}",
        f"{resolved.coordinates[2]:.3f}",
    )
    console.print(table)
    _display_donor_candidates([item.to_dict() for item in resolved.donor_atoms])
    if resolved.auto_filled_donor_atoms:
        console.print("[bold yellow]Auto-filled donor atoms:[/bold yellow]")
        _display_donor_candidates([item.to_dict() for item in resolved.auto_filled_donor_atoms])
    for warning in resolved.warnings:
        console.print(f"[bold yellow]{warning}[/bold yellow]")


def _prompt_metal_insertions(summary) -> list[MetalInsertion]:
    if summary is None or not getattr(summary, "source_path", None):
        return []
    structure = load_structure(summary.source_path)
    insertions: list[MetalInsertion] = []
    while True:
        element = _prompt_insertion_metal()
        charge = _prompt_insertion_charge(element)
        target_cn = _prompt_insertion_target_cn(element)
        mode_choices = [
            WizardChoice(MetalAnchorMode.RESIDUE_DONORS.value, "Residue selector", "Examples: A:45,A:112"),
            WizardChoice(MetalAnchorMode.DONOR_ATOMS.value, "Atom selector", "Examples: A:45@ND1,A:112@SG"),
            WizardChoice(MetalAnchorMode.ATOM_SERIALS.value, "Atom serial", "Examples: 1234,1450"),
            WizardChoice(MetalAnchorMode.XYZ.value, "Exact XYZ", "Advanced coordinate input."),
        ]
        _display_choice_table("Metal insertion anchor mode", mode_choices)
        anchor_mode = MetalAnchorMode(
            _prompt_choice(
                "Choose how to specify the insertion anchors",
                mode_choices,
                default_key=MetalAnchorMode.RESIDUE_DONORS.value,
            )
        )

        if anchor_mode == MetalAnchorMode.XYZ:
            while True:
                raw_xyz = typer.prompt(_back_prompt_suffix("Metal XYZ coordinates as x,y,z"), default="").strip()
                if _is_back_token(raw_xyz):
                    raise WizardBack()
                try:
                    insertion = MetalInsertion(
                        element=element,
                        charge=charge,
                        anchor_mode=anchor_mode,
                        coordinates=_parse_xyz_coordinates(raw_xyz),
                        target_coordination_number=target_cn,
                    )
                    break
                except ValueError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")
        elif anchor_mode == MetalAnchorMode.RESIDUE_DONORS:
            raw = _prompt_anchor_text(
                structure,
                "Residue selector(s), or 'l' to list atoms (examples: A:45,A:112)",
            )
            candidates = donor_candidates_for_residue_selectors(structure, [raw])
            _display_donor_candidates(candidates)
            anchors = _prompt_serial_subset_from_candidates(candidates)
            insertion = MetalInsertion(
                element=element,
                charge=charge,
                anchor_mode=MetalAnchorMode.ATOM_SERIALS,
                anchors=anchors,
                target_coordination_number=target_cn,
            )
        else:
            example = "A:45@ND1,A:112@SG" if anchor_mode == MetalAnchorMode.DONOR_ATOMS else "1234,1450"
            raw = _prompt_anchor_text(
                structure,
                f"Anchor atom(s), or 'l' to list atoms (examples: {example})",
            )
            insertion = MetalInsertion(
                element=element,
                charge=charge,
                anchor_mode=anchor_mode,
                anchors=[raw],
                target_coordination_number=target_cn,
            )

        try:
            resolved = resolve_metal_insertion(structure, insertion)
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        _display_resolved_metal_insertion(resolved)
        if _prompt_yes_no("Add this metal insertion to the workflow?", default=True):
            insertions.append(insertion)
        if not _prompt_yes_no("Add another inserted metal?", default=False):
            return insertions


def _display_detected_metal_summary(summary) -> None:
    if summary is None:
        return
    metals = list(getattr(summary, "metals", []) or [])
    if not metals:
        console.print("[bold yellow]No supported metal sites were detected in the loaded structure.[/bold yellow]")
        return
    counts: dict[str, int] = {}
    for site in metals:
        element = str(getattr(site, "element", "") or "?").title()
        counts[element] = counts.get(element, 0) + 1
    summary_text = ", ".join(f"{element} x {count}" for element, count in sorted(counts.items()))
    console.print(f"[bold cyan]Detected supported metal sites:[/bold cyan] {summary_text} ({len(metals)} total).")


def _prompt_metal_actions(summary, input_config: InputConfig) -> _MetalActionPlan:
    if summary is None:
        return _MetalActionPlan()

    _display_detected_metal_summary(summary)
    insertions = []
    if _prompt_yes_no("Add a new metal near selected donor atoms?", default=False):
        insertions = _prompt_metal_insertions(summary)

    if not summary.metals:
        return _MetalActionPlan(metal_insertions=insertions)

    available_sites = {site.site for site in summary.metals}
    mode_choices = [
        WizardChoice(str(index), title, description)
        for index, (title, description) in enumerate(METAL_MODE_OPTIONS, start=1)
    ]
    _display_choice_table("Metal replacement/deletion mode", mode_choices)
    selection = int(_prompt_choice("Choose a metal handling mode", mode_choices, default_key="1"))

    if selection == 1:
        return _MetalActionPlan(metal_insertions=insertions)
    if selection == 4:
        return _MetalActionPlan(remove_metals=True, metal_insertions=insertions)

    if selection in {2, 3}:
        while True:
            selected_sites = (
                sorted(available_sites)
                if selection == 2
                else _prompt_site_selection("replace", available_sites)
            )
            strategy, variant_sites = _prompt_replacement_strategy(summary, selected_sites)
            selected_metals = _prompt_replacement_metals()
            variants = _expand_replacement_variants(
                variant_sites,
                selected_metals,
                strategy=strategy,
            )
            if len(variants) > 1 and not _confirm_metal_batch_preview(input_config, variants):
                console.print("[dim]Let's adjust the metal replacement plan.[/dim]")
                continue
            return _MetalActionPlan(
                remove_metals=False,
                metal_deletions=[],
                metal_insertions=insertions,
                variants=[_MetalReplacementVariant(_copy_metal_replacements(variant.replacements), variant.suffix_tokens) for variant in variants],
            )

    selected_sites = _prompt_site_selection("remove", available_sites)
    return _MetalActionPlan(remove_metals=False, metal_deletions=selected_sites, metal_insertions=insertions)


def _remaining_metal_sites(summary, prepare_config: PrepareConfig) -> list[tuple[object, str]]:
    if summary is None or not summary.metals:
        return []

    replacement_map = {item.site: item.target.title() for item in prepare_config.metal_replacements}
    deletion_sites = set(prepare_config.metal_deletions)
    remaining: list[tuple[object, str]] = []
    for metal_site in summary.metals:
        if prepare_config.remove_metals or metal_site.site in deletion_sites:
            continue
        final_element = replacement_map.get(metal_site.site, metal_site.element.title())
        remaining.append((metal_site, final_element))
    return remaining


def _display_metal_charge_summary(remaining_sites: list[tuple[object, str]]) -> None:
    if not remaining_sites:
        return

    table = Table(title="Remaining metal sites", box=box.SIMPLE_HEAVY)
    table.add_column("Site", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Residue", style="bold white")
    table.add_column("Final metal", style="cyan")
    table.add_column("Default", style="white", justify="center", no_wrap=True)
    for metal_site, final_element in remaining_sites:
        default_charge = DEFAULT_METAL_CHARGES.get(final_element, 2)
        table.add_row(
            str(metal_site.site),
            metal_site.key,
            final_element,
            f"+{default_charge}",
        )
    console.print(table)


def _charge_choice_description(charge: int) -> str:
    if charge == 1:
        return "Monovalent formal oxidation state; final ion-model compatibility is checked later."
    return "Multivalent formal oxidation state; final ion-model compatibility is checked later."


def _prompt_metal_charge_assignments(summary, prepare_config: PrepareConfig) -> list[MetalChargeAssignment]:
    remaining_sites = _remaining_metal_sites(summary, prepare_config)
    if not remaining_sites:
        return []

    console.print(
        "[dim]PDB files usually do not provide a reliably usable oxidation state, "
        "so please confirm it for each remaining metal site.[/dim]"
    )
    _display_metal_charge_summary(remaining_sites)
    assignments: list[MetalChargeAssignment] = []
    for metal_site, final_element in remaining_sites:
        choices = [
            WizardChoice(str(charge), f"+{charge}", _charge_choice_description(charge))
            for index, charge in enumerate(allowed_metal_charges(final_element), start=1)
        ]
        if not choices:
            raise ValueError(f"No supported 12-6-4 oxidation states are configured for metal {final_element}.")
        _display_choice_table(f"Oxidation-state options for site {metal_site.site} ({final_element})", choices)
        default_charge = str(DEFAULT_METAL_CHARGES.get(final_element, choices[0].key))
        selected = int(
            _prompt_choice(
                f"Choose the oxidation state for site {metal_site.site} ({final_element} at {metal_site.key})",
                choices,
                default_key=default_charge,
            )
        )
        assignments.append(MetalChargeAssignment(site=metal_site.site, charge=selected))
    return assignments


def _prepare_config_with_replacements(
    base_prepare_config: PrepareConfig,
    replacements: list[MetalReplacement],
) -> PrepareConfig:
    return base_prepare_config.model_copy(
        deep=True,
        update={"metal_replacements": _copy_metal_replacements(replacements)},
    )


def _unique_remaining_site_metals(
    summary,
    prepare_configs: list[PrepareConfig],
) -> list[tuple[object, str]]:
    ordered_pairs: list[tuple[object, str]] = []
    seen: set[tuple[int, str]] = set()
    for prepare_config in prepare_configs:
        for metal_site, final_element in _remaining_metal_sites(summary, prepare_config):
            key = (metal_site.site, final_element)
            if key in seen:
                continue
            seen.add(key)
            ordered_pairs.append((metal_site, final_element))
    return ordered_pairs


def _prompt_batch_metal_charge_assignments(
    summary,
    prepare_configs: list[PrepareConfig],
) -> dict[tuple[int, str], int]:
    unique_pairs = _unique_remaining_site_metals(summary, prepare_configs)
    if not unique_pairs:
        return {}

    console.print(
        "[dim]PDB files usually do not provide a reliably usable oxidation state, "
        "so please confirm it for each unique site/metal combination that can appear in this batch.[/dim]"
    )
    table = Table(title="Unique site/metal combinations across the batch", box=box.SIMPLE_HEAVY)
    table.add_column("Site", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Residue", style="bold white")
    table.add_column("Final metal", style="cyan")
    table.add_column("Default", style="white", justify="center", no_wrap=True)
    for metal_site, final_element in unique_pairs:
        default_charge = DEFAULT_METAL_CHARGES.get(final_element, 2)
        table.add_row(
            str(metal_site.site),
            metal_site.key,
            final_element,
            f"+{default_charge}",
        )
    console.print(table)

    assignments: dict[tuple[int, str], int] = {}
    for metal_site, final_element in unique_pairs:
        choices = [
            WizardChoice(str(charge), f"+{charge}", _charge_choice_description(charge))
            for charge in allowed_metal_charges(final_element)
        ]
        if not choices:
            raise ValueError(f"No supported 12-6-4 oxidation states are configured for metal {final_element}.")
        _display_choice_table(
            f"Oxidation-state options for site {metal_site.site} ({final_element})",
            choices,
        )
        default_charge = str(DEFAULT_METAL_CHARGES.get(final_element, int(choices[0].key)))
        selected = int(
            _prompt_choice(
                f"Choose the oxidation state for site {metal_site.site} ({final_element} at {metal_site.key})",
                choices,
                default_key=default_charge,
            )
        )
        assignments[(metal_site.site, final_element)] = selected
    return assignments


def _variant_metal_charge_assignments(
    summary,
    prepare_config: PrepareConfig,
    selected_charges: dict[tuple[int, str], int],
) -> list[MetalChargeAssignment]:
    assignments: list[MetalChargeAssignment] = []
    for metal_site, final_element in _remaining_metal_sites(summary, prepare_config):
        charge = selected_charges[(metal_site.site, final_element)]
        assignments.append(MetalChargeAssignment(site=metal_site.site, charge=charge))
    return assignments


def _classify_structure_input(structure_input: str) -> tuple[str, str]:
    candidate = Path(structure_input).expanduser()
    if candidate.exists() and candidate.is_file():
        return "file", structure_input
    if looks_like_pdb_id(structure_input):
        return "pdb_id", structure_input.upper()
    if candidate.exists():
        return "non_file_path", structure_input
    return "unknown", structure_input


def _display_des_library_candidates(candidates) -> None:
    table = Table(title="Detected DES library bundles", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Residue", style="bold white", no_wrap=True)
    table.add_column("Status", style="cyan")
    table.add_column("Match", style="white")
    table.add_column("Library", style="white")
    table.add_column("FRCMOD", style="white")
    for index, candidate in enumerate(candidates, start=1):
        status_text = {
            "already_registered": "already registered",
            "different_values": "same residue, different values",
            "new": "new",
        }.get(candidate.status, candidate.status)
        table.add_row(
            str(index),
            candidate.residue_name,
            status_text,
            candidate.matched_label or "",
            str(candidate.lib_path),
            str(candidate.frcmod_path),
        )
    console.print(table)


def _prompt_manual_des_library_candidate():
    console.print(
        "[dim]Enter a folder to scan recursively, or enter one .lib/.off file and one .frcmod file separated by a comma.[/dim]"
    )
    while True:
        raw = typer.prompt(_back_prompt_suffix("DES library folder or files (.lib/.off, .frcmod)"), default="").strip()
        if _is_back_token(raw):
            raise WizardBack()
        paths = [item.strip() for item in raw.split(",") if item.strip()]
        if len(paths) == 1:
            target = Path(paths[0]).expanduser().resolve()
            if target.is_dir():
                candidates = discover_des_library_candidates(target)
                if not candidates:
                    console.print("[bold red]No .lib/.off + .frcmod bundles were found under that folder.[/bold red]")
                    continue
                if len(candidates) == 1:
                    return candidates[0]
                _display_des_library_candidates(candidates)
                while True:
                    choice = typer.prompt(
                        _back_prompt_suffix("Choose a detected DES library bundle number"),
                        default="1",
                    ).strip()
                    if _is_back_token(choice):
                        raise WizardBack()
                    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                        return candidates[int(choice) - 1]
                    console.print("[bold red]Please choose one listed candidate number or B.[/bold red]")
            elif target.is_file() and target.suffix.lower() in {".lib", ".off", ".frcmod"}:
                candidates = [
                    candidate
                    for candidate in discover_des_library_candidates(target.parent)
                    if candidate.lib_path == target or candidate.frcmod_path == target
                ]
                if not candidates:
                    console.print("[bold red]No matching DES library pair was found next to that file.[/bold red]")
                    continue
                if len(candidates) == 1:
                    return candidates[0]
                _display_des_library_candidates(candidates)
                while True:
                    choice = typer.prompt(
                        _back_prompt_suffix("Choose a detected DES library bundle number"),
                        default="1",
                    ).strip()
                    if _is_back_token(choice):
                        raise WizardBack()
                    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                        return candidates[int(choice) - 1]
                    console.print("[bold red]Please choose one listed candidate number or B.[/bold red]")
            else:
                console.print("[bold red]Please enter an existing folder, or two existing files separated by a comma.[/bold red]")
                continue
        if len(paths) != 2:
            console.print("[bold red]Please provide a folder, or exactly two files: one .lib/.off and one .frcmod.[/bold red]")
            continue
        resolved = [Path(path).expanduser().resolve() for path in paths]
        if not all(path.exists() and path.is_file() for path in resolved):
            console.print("[bold red]Both paths must be existing files.[/bold red]")
            continue
        lib_files = [path for path in resolved if path.suffix.lower() in {".lib", ".off"}]
        frcmods = [path for path in resolved if path.suffix.lower() == ".frcmod"]
        if len(lib_files) != 1 or len(frcmods) != 1:
            console.print("[bold red]The pair must contain one .lib/.off file and one .frcmod file.[/bold red]")
            continue

        return classify_des_library_bundle(lib_path=lib_files[0], frcmod_path=frcmods[0])


def _register_des_library_candidate(candidate) -> None:
    if candidate.status == "already_registered":
        console.print(
            f"[bold cyan]{candidate.residue_name} is already registered[/bold cyan] "
            f"as {candidate.matched_label or candidate.matched_component}."
        )
        return
    if candidate.status == "different_values":
        console.print(
            f"[bold yellow]{candidate.residue_name} exists with different parameter values[/bold yellow] "
            f"({candidate.matched_label or candidate.matched_component})."
        )
        if not typer.confirm("Add this as a separate DES component variant?", default=True):
            return
    else:
        if not typer.confirm(f"Register {candidate.residue_name} as a DES component?", default=True):
            return

    default_key = f"custom_{candidate.residue_name.lower()}"
    component_key = typer.prompt("Custom component key", default=default_key).strip()
    label = typer.prompt("Display label", default=candidate.residue_name).strip()
    definition = register_custom_des_component(
        lib_path=candidate.lib_path,
        frcmod_path=candidate.frcmod_path,
        component_key=component_key,
        label=label,
    )
    console.print(
        f"[bold green]Registered DES component:[/bold green] {definition.label} "
        f"([cyan]{_des_component_key(definition.key)}[/cyan])"
    )


def _prompt_add_des_component_library() -> None:
    console.print(
        Panel(
            "This mode registers Amber-ready custom residues for the DES library only.\n"
            "Custom Ligand and MetalloProtein residue libraries are planned for a later update.",
            title="[bold cyan]Add Component In Library[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    candidates = discover_des_library_candidates(
        Path.cwd(),
        recursive=search_subdirectories_enabled(),
    )
    if candidates:
        _display_des_library_candidates(candidates)
        while True:
            raw = typer.prompt(
                "Choose a candidate number to inspect/register, M to enter paths manually, or B to go back",
                default="1",
            ).strip()
            if _is_back_token(raw):
                return
            if raw.lower() == "m":
                try:
                    _register_des_library_candidate(_prompt_manual_des_library_candidate())
                except WizardBack:
                    return
                return
            if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                _register_des_library_candidate(candidates[int(raw) - 1])
                return
            console.print("[bold red]Please choose one listed candidate number, M, or B.[/bold red]")
    else:
        console.print("[dim]No DES .lib/.off + .frcmod bundles were auto-detected in the launch directory.[/dim]")
        if typer.confirm("Enter DES library paths manually?", default=True):
            try:
                _register_des_library_candidate(_prompt_manual_des_library_candidate())
            except WizardBack:
                return


def _prompt_input_config() -> (
    tuple[InputConfig, object | None, object | None, DESConfig | None]
    | _ProteinSiteRespResumeSelection
):
    while True:
        try:
            while True:
                raw_workflow = typer.prompt(
                    "Workflow Type: [D] Deep Eutectic Solvent, [S] Small-molecule, [P] Protein, or [A] Add component in library? [d/s/P/A]",
                    default="P",
                ).strip()
                try:
                    workflow_type = _parse_workflow_type_choice(raw_workflow)
                    break
                except ValueError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")

            if workflow_type == ADD_DES_LIBRARY_MODE:
                _prompt_add_des_component_library()
                continue

            if workflow_type == InputSource.DES:
                des_config = _prompt_des_config()
                return InputConfig(source=InputSource.DES), None, None, des_config

            if workflow_type == InputSource.SMALL_MOLECULE:
                workspace_candidates = _workspace_detectable_resp_candidates()
                while True:
                    start_mode = _prompt_small_molecule_start_mode(
                        has_detected_resp_candidates=bool(workspace_candidates),
                    )
                    if start_mode == "resp_continue":
                        chosen = _prompt_workspace_resp_resume_option()
                        if chosen is None:
                            continue
                        try:
                            input_config = _input_config_from_resp_candidate(chosen)
                            return (
                                input_config,
                                _inspect_small_molecule_input(input_config, detected_resp_candidate=chosen),
                                chosen,
                                None,
                            )
                        except RuntimeError as exc:
                            console.print(f"[bold red]{exc}[/bold red]")
                        continue

                    console.print(SMALL_MOLECULE_INPUT_HINT)
                    files = _prompt_csv(SMALL_MOLECULE_INPUT_PROMPT)
                    if not files:
                        console.print("[bold red]Please provide at least one input file.[/bold red]")
                        continue
                    if not _validate_existing_paths(files, label="small-molecule input file(s)"):
                        continue
                    input_config = InputConfig(source=InputSource.SMALL_MOLECULE, small_molecule_files=files)
                    detected_resp_candidate = _prompt_detected_resp_resume_option(input_config=input_config)
                    return (
                        input_config,
                        _inspect_small_molecule_input(input_config, detected_resp_candidate=detected_resp_candidate),
                        detected_resp_candidate,
                        None,
                    )

            protein_resume = _prompt_protein_site_resp_resume_selection(
                _workspace_detectable_protein_site_resp_candidates()
            )
            if protein_resume is not None:
                return protein_resume

            console.print(
                "Enter a local PDB path or a 4-character PDB ID. "
                "If you enter a PDB ID, SIMPLE will download it automatically."
            )
            while True:
                structure_input = typer.prompt("Path to input PDB or PDB ID (e.g., 3FS9), or B to go back", default="").strip()
                if _is_back_token(structure_input):
                    raise WizardBack()
                if not structure_input:
                    console.print("[bold red]Please enter a PDB file path or a 4-character PDB ID.[/bold red]")
                    continue
                input_kind, resolved_value = _classify_structure_input(structure_input)
                if input_kind == "file":
                    try:
                        return (
                            InputConfig(source=InputSource.PDB_FILE, path=resolved_value),
                            inspect_structure(resolved_value),
                            None,
                            None,
                        )
                    except Exception as exc:
                        console.print(f"[bold red]The file could not be read as a structure: {exc}[/bold red]")
                        continue
                if input_kind == "pdb_id":
                    pdb_id = resolved_value
                    temp_dir = Path(".simple_wizard")
                    try:
                        fetched = fetch_pdb_structure(pdb_id, temp_dir)
                        return (
                            InputConfig(source=InputSource.PDB_ID, pdb_id=pdb_id),
                            inspect_structure(fetched, source_label="pdb_id"),
                            None,
                            None,
                        )
                    except Exception as exc:
                        console.print(f"[bold red]Failed to download or inspect PDB ID {pdb_id}: {exc}[/bold red]")
                        continue
                if input_kind == "non_file_path":
                    console.print(
                        "[bold red]That path exists, but it is not a structure file. "
                        "Please enter a local PDB file path or a 4-character PDB ID.[/bold red]"
                    )
                    continue
                console.print(
                    "[bold red]Input must be an existing PDB path or a 4-character PDB ID such as 3FS9. Please try again.[/bold red]"
                )
        except WizardBack:
            console.print("[dim]Back: returning to workflow type selection.[/dim]")
            continue


def _small_molecule_metal_summary(
    source_file: str | Path,
    *,
    residue_name: str = LigandsConfig().residue_name,
) -> StructureSummary:
    source_path = Path(source_file).expanduser().resolve()
    if source_path.suffix.lower().lstrip(".") in {"smi", "smiles", "txt"}:
        preview_dir = Path(".simple_wizard") / "smiles_preview"
        preview_source = prepare_canonical_small_molecule_mol2(
            source_file=source_path,
            residue_name=residue_name,
            output_dir=preview_dir,
            split_supported_metals=False,
            canonical_filename=f"{residue_name}_smiles_preview.mol2",
        )
        molecule = load_molecule(preview_source)
    else:
        molecule = load_molecule(source_path)
    residue_token = residue_name.strip().upper() or LigandsConfig().residue_name
    ligand_record = ResidueRecord(
        key=f"A:{residue_token}:1",
        chain="A",
        seqid="1",
        residue_name=residue_token,
        atom_count=len(molecule.atoms),
        classification="hetero",
    )
    metals: list[MetalSite] = []
    site_index = 1
    for atom in molecule.atoms:
        element = str(atom.element or "").strip().title()
        if element not in SUPPORTED_METALS:
            continue
        metals.append(
            MetalSite(
                site=site_index,
                key=f"A:{residue_token}:1@{atom.name}",
                chain="A",
                seqid="1",
                residue_name=residue_token,
                atom_name=str(atom.name),
                atom_serial=int(atom.index),
                element=element,
            )
        )
        site_index += 1

    return StructureSummary(
        source="small_molecule",
        source_path=str(source_path),
        residue_counts={
            "standard": 0,
            "water": 0,
            "metal": len(metals),
            "hetero": 1 if molecule.atoms else 0,
        },
        metals=metals,
        hetero_residues=[ligand_record] if molecule.atoms else [],
        ligand_candidates=[ligand_record] if molecule.atoms else [],
    )


def _inspect_small_molecule_input(
    input_config: InputConfig,
    *,
    detected_resp_candidate: object | None = None,
) -> StructureSummary | None:
    if input_config.source != InputSource.SMALL_MOLECULE or not input_config.small_molecule_files:
        return None
    residue_name = LigandsConfig().residue_name
    if detected_resp_candidate is not None:
        payload = getattr(detected_resp_candidate, "payload", {}) or {}
        residue_name = str(payload.get("residue_name") or residue_name)
    try:
        return _small_molecule_metal_summary(input_config.small_molecule_files[0], residue_name=residue_name)
    except Exception as exc:
        console.print(
            "[bold yellow]Small-molecule inspection warning:[/bold yellow] "
            f"Could not inspect the input for supported metals before parameterization ({exc}). "
            "The workflow will continue, but automatic 12-6-4 metal prompts may be unavailable."
        )
        return None


def _display_summary(summary) -> None:
    counts = Table(title="Structure summary", box=box.SIMPLE_HEAVY)
    counts.add_column("Component")
    counts.add_column("Count")
    labels = {
        "standard": "Standard amino acids / nucleic residues",
        "water": "Water",
        "metal": "Metal sites",
        "hetero": "Hetero / custom residues",
    }
    for key, value in summary.residue_counts.items():
        counts.add_row(labels.get(key, key.title()), str(value))
    console.print(counts)

    if summary.metals:
        metals = Table(title="Detected metal sites", box=box.SIMPLE_HEAVY)
        metals.add_column("Site")
        metals.add_column("Key")
        metals.add_column("Element")
        for site in summary.metals:
            metals.add_row(str(site.site), site.key, site.element)
        console.print(metals)

    if summary.hetero_residues:
        hetero = Table(title="Detected hetero/custom residues", box=box.SIMPLE_HEAVY)
        hetero.add_column("Key")
        hetero.add_column("Residue")
        hetero.add_column("Atoms")
        for item in summary.hetero_residues:
            hetero.add_row(item.key, item.residue_name, str(item.atom_count))
        console.print(hetero)

    _display_missing_loop_summary(summary.missing_loops)


def _display_missing_loop_summary(missing_loops: MissingLoopSummary | None) -> None:
    if missing_loops is None:
        return
    if missing_loops.detection_status != "available":
        if missing_loops.detection_message:
            console.print(
                Panel(
                    missing_loops.detection_message,
                    title="[bold yellow]Missing-loop Detection[/bold yellow]",
                    border_style="yellow",
                )
            )
        return

    if missing_loops.internal_blocks:
        internal = Table(title="Detected internal missing loop blocks", box=box.SIMPLE_HEAVY)
        internal.add_column("Range", style="bold white")
        internal.add_column("Length", style="bold cyan", justify="right")
        internal.add_column("Missing residues", style="white")
        internal.add_column("Boundary residues", style="yellow")
        for block in missing_loops.internal_blocks:
            boundary_labels = []
            try:
                boundary_labels = [
                    f"{block.chain_id.strip() or '(blank)'}:{int(block.start_resseq) - 1}",
                    f"{block.chain_id.strip() or '(blank)'}:{int(block.end_resseq) + 1}",
                ]
            except ValueError:
                boundary_labels = []
            internal.add_row(
                block.range_label,
                str(block.length),
                ", ".join(block.residue_names),
                ", ".join(boundary_labels) if boundary_labels else "-",
            )
        console.print(internal)
        console.print(
            "[bold yellow]Internal missing loops are repair-eligible only after PROPKA, and any PDBFixer rebuild "
            "should be checked manually before simulation.[/bold yellow]"
        )

    if missing_loops.terminal_blocks:
        terminal = Table(title="Detected terminal missing residue blocks", box=box.SIMPLE_HEAVY)
        terminal.add_column("Range", style="bold white")
        terminal.add_column("Length", style="bold cyan", justify="right")
        terminal.add_column("Missing residues", style="white")
        terminal.add_column("Handling", style="yellow")
        for block in missing_loops.terminal_blocks:
            terminal.add_row(
                block.range_label,
                str(block.length),
                ", ".join(block.residue_names),
                "Not repaired automatically",
            )
        console.print(terminal)


def _input_source_value(input_config: InputConfig) -> str:
    return input_config.path or input_config.pdb_id or ""


def _prompt_missing_loop_repair(summary) -> bool:
    if summary is None or summary.missing_loops is None:
        return False
    if summary.missing_loops.detection_status != "available":
        return False
    if not summary.missing_loops.internal_blocks:
        return False
    return typer.confirm(
        "Use PDBFixer after PROPKA to rebuild the detected internal missing loop residues? "
        "This is a rough repair and should be inspected before simulation.",
        default=False,
    )


def _generate_preview_prepared_pdb(
    input_config: InputConfig,
    prepare_config: PrepareConfig,
    protonation_config: ProtonationConfig | None,
    output_dir: Path,
) -> Path:
    prepared = prepare_structure(
        source=input_config.source,
        source_value=_input_source_value(input_config),
        prepare_config=prepare_config,
        protonation_config=protonation_config,
        kept_ligands=prepare_config.kept_ligands,
        output_dir=output_dir,
        apply_loop_repair=False,
    )
    cleaned_pdb = Path(prepared["cleaned_pdb"])
    return cleaned_pdb


def _estimate_charge_from_prepared_pdb(
    prepared_pdb: Path,
    *,
    metal_charges: list[MetalChargeAssignment] | None = None,
) -> tuple[int, str]:
    structure = load_structure(prepared_pdb)
    while len(structure) > 1:
        del structure[1]

    explicit_metal_charges = {item.site: int(item.charge) for item in metal_charges or []}
    charge = 0
    metal_index = 0
    for chain in structure[0]:
        for residue in chain:
            classification = classify_residue(residue)
            residue_name = residue.name.strip().upper()
            if classification in {"water", "hetero"}:
                continue
            if classification == "metal":
                metal_index += 1
                target = residue[0].element.name.title() if len(residue) == 1 else residue_name.title()
                charge += explicit_metal_charges.get(
                    metal_index,
                    DEFAULT_METAL_CHARGES.get(target or residue_name.title(), 0),
                )
                continue
            charge += STANDARD_RESIDUE_CHARGES.get(residue_name, 0)

    if explicit_metal_charges:
        note = (
            "Approximate net charge based on the cleaned/protonated structure and the oxidation states you selected "
            "for the remaining metal sites. Custom ligands and other non-standard residues are treated as neutral "
            "in this preview."
        )
    else:
        note = (
            "Approximate net charge based on the cleaned/protonated structure and default supported metal charges. "
            "Custom ligands and other non-standard residues are treated as neutral in this preview."
        )
    return charge, note


def _metal_charges_with_preview_insertions(
    base_charges: list[MetalChargeAssignment] | None,
    preview_dir: Path,
) -> list[MetalChargeAssignment]:
    assignment_map = {int(item.site): int(item.charge) for item in base_charges or []}
    manifest_path = preview_dir / "prepare_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        for item in manifest.get("inserted_metal_sites") or []:
            site = item.get("site")
            charge = item.get("charge")
            if site is not None and charge is not None:
                assignment_map.setdefault(int(site), int(charge))
    return [
        MetalChargeAssignment(site=site, charge=charge)
        for site, charge in sorted(assignment_map.items())
    ]


def _display_protonation_candidate_table(
    title: str,
    candidates: list[ProtonationDisplayCandidate],
    *,
    start_index: int,
) -> tuple[list[tuple[int, ProtonationChange]], int]:
    if not candidates:
        return [], start_index

    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Residue", style="bold white")
    table.add_column("Current", style="white")
    table.add_column("Suggested", style="cyan")
    table.add_column("Predicted pKa", style="white", justify="right")
    table.add_column("Metal-near", style="yellow")
    table.add_column("Why listed", style="white")

    numbered_changes: list[tuple[int, ProtonationChange]] = []
    next_index = start_index
    for candidate in candidates:
        chain_label = candidate.chain or "(blank)"
        metal_flag = "Yes" if candidate.metal_near else "No"
        residue_label = f"{chain_label}:{candidate.seqid}"
        if candidate.metal_near:
            residue_label = f"[bold yellow]{residue_label}[/bold yellow]"
            metal_flag = "[bold yellow]Yes[/bold yellow]"

        number_label = ""
        if candidate.selectable and candidate.change is not None:
            number_label = str(next_index)
            numbered_changes.append((next_index, candidate.change))
            next_index += 1

        predicted_pka = "-" if candidate.predicted_pka is None or math.isnan(candidate.predicted_pka) else f"{candidate.predicted_pka:.2f}"
        table.add_row(
            number_label,
            residue_label,
            candidate.original_residue_name,
            candidate.target_residue_name,
            predicted_pka,
            metal_flag,
            candidate.reason,
        )

    console.print(table)
    return numbered_changes, next_index


def _protonation_candidate_signature(
    candidate: ProtonationDisplayCandidate,
    *,
    candidate_kind: str,
) -> tuple[object, ...]:
    return (
        candidate_kind,
        candidate.chain.strip(),
        candidate.seqid.strip(),
        candidate.original_residue_name.strip().upper(),
        candidate.target_residue_name.strip().upper(),
        bool(candidate.selectable),
    )


def _protonation_prediction_signature(prediction) -> tuple[tuple[object, ...], ...]:
    signature = [
        _protonation_candidate_signature(candidate, candidate_kind="coordination")
        for candidate in prediction.metal_coordination_candidates
    ]
    signature.extend(
        _protonation_candidate_signature(candidate, candidate_kind="propka")
        for candidate in prediction.propka_candidates
    )
    return tuple(sorted(signature))


def _matching_previous_phs(
    signature: tuple[tuple[object, ...], ...],
    previous_variants: list[_ProtonationVariant],
) -> list[float]:
    matches: list[float] = []
    for variant in previous_variants:
        if variant.ph is None or variant.signature != signature:
            continue
        matches.append(float(variant.ph))
    return matches


def _confirm_duplicate_ph_protonation_state(*, ph: float, previous_phs: list[float]) -> bool:
    previous_labels = ", ".join(f"pH {_format_ph_display(value)}" for value in previous_phs)
    console.print(
        Panel(
            f"The PROPKA protonation-state proposal for pH {_format_ph_display(ph)} is identical to {previous_labels}.\n\n"
            "Splitting this state into a separate simulation may not add much value, because the proposed protonation "
            "states are the same across those pH values.",
            title="[bold yellow]Duplicate pH Protonation-State Warning[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    return typer.confirm("Keep this pH case anyway?", default=True)


def _display_protonation_candidates(prediction, *, ph: float) -> list[tuple[int, ProtonationChange]]:
    numbered_changes: list[tuple[int, ProtonationChange]] = []
    next_index = 1

    metal_numbered, next_index = _display_protonation_candidate_table(
        f"Direct metal-coordination review at pH {ph:.2f}",
        prediction.metal_coordination_candidates,
        start_index=next_index,
    )
    numbered_changes.extend(metal_numbered)

    propka_numbered, next_index = _display_protonation_candidate_table(
        f"Additional PROPKA protonation suggestions at pH {ph:.2f}",
        prediction.propka_candidates,
        start_index=next_index,
    )
    numbered_changes.extend(propka_numbered)

    if any(candidate.metal_near for candidate in prediction.metal_coordination_candidates + prediction.propka_candidates):
        console.print(
            "[bold yellow]Metal-near residues[/bold yellow] fall inside the same 4.0 A neighborhood used for the "
            "focused-restraint suggestion."
        )
    if any(not candidate.selectable for candidate in prediction.metal_coordination_candidates):
        console.print("[dim]Rows without a number are shown for review only and cannot be selected.[/dim]")
    console.print(f"[dim]{PROTONATION_STATE_GUIDE}[/dim]")
    console.print(
        "[dim]For directly metal-coordinating histidines, HID/HIE is chosen from the metal-facing nitrogen.\n"
        "ND1 coordination -> HIE\n"
        "NE2 coordination -> HID[/dim]"
    )
    return numbered_changes


def _display_protonation_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    body = "\n".join(f"- {warning}" for warning in warnings)
    console.print(
        Panel(
            body,
            title="[bold yellow]PROPKA Metal Assumptions[/bold yellow]",
            border_style="yellow",
        )
    )


def _display_missing_loop_protonation_notice(summary) -> None:
    if summary is None or summary.missing_loops is None:
        return

    missing_loops = summary.missing_loops
    if missing_loops.detection_status != "available":
        if missing_loops.detection_message:
            console.print(
                Panel(
                    missing_loops.detection_message,
                    title="[bold yellow]Missing-loop Detection[/bold yellow]",
                    border_style="yellow",
                )
            )
        return

    if not missing_loops.internal_blocks:
        return

    boundary_labels = ", ".join(item.label for item in missing_loops.boundary_residues) or "none"
    console.print(
        Panel(
            "Missing internal loop blocks were detected. PROPKA assignments are computed only for residues present "
            "in the experimental structure, and loop-boundary residues are excluded because chain-break-adjacent "
            f"geometry may be rough.\n\nExcluded boundary residues: {boundary_labels}\n\n"
            "If you later rebuild the loop with PDBFixer, inspect the rebuilt region carefully before simulation.",
            title="[bold yellow]Missing-loop Protonation Notice[/bold yellow]",
            border_style="yellow",
        )
    )

def _prompt_selected_numbered_protonation_changes(
    numbered_changes: list[tuple[int, ProtonationChange]],
) -> list[ProtonationChange]:
    if not numbered_changes:
        return []
    if typer.confirm("Apply all suggested protonation-state changes?", default=True):
        return [change for _, change in numbered_changes]

    available_indices = {index for index, _ in numbered_changes}
    change_by_index = {index: change for index, change in numbered_changes}
    while True:
        raw = typer.prompt(
            "Change number(s) to apply (comma separated, blank for none)",
            default="",
        ).strip()
        try:
            selected_indices = _parse_index_selection(raw, available_indices, label="Change")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        return [change_by_index[index] for index in selected_indices]


def _prompt_selected_protonation_changes(changes: list[ProtonationChange]) -> list[ProtonationChange]:
    if not changes:
        return []
    return _prompt_selected_numbered_protonation_changes(
        [(index, change) for index, change in enumerate(changes, start=1)]
    )


def _prompt_protonation_variants(
    *,
    input_config: InputConfig,
    prepare_config: PrepareConfig,
    summary,
) -> list[_ProtonationVariant]:
    if input_config.source == InputSource.SMALL_MOLECULE or summary is None:
        console.print("[dim]pH-guided protonation is skipped for a small-molecule-only workflow.[/dim]")
        return [_ProtonationVariant(protonation_config=ProtonationConfig())]

    use_protonation = typer.confirm(
        "Use pH-guided protonation-state assignment?",
        default=True,
    )
    if not use_protonation:
        console.print("[dim]Standard residue names from the cleaned input structure will be kept unchanged.[/dim]")
        return [_ProtonationVariant(protonation_config=ProtonationConfig())]

    ph_values = _prompt_ph_values()
    excluded_boundary_locators = (
        set()
        if summary is None or summary.missing_loops is None
        else summary.missing_loops.boundary_residue_locators()
    )
    _display_missing_loop_protonation_notice(summary)

    shown_warning_sets: set[tuple[str, ...]] = set()
    retained_variants: list[_ProtonationVariant] = []
    with TemporaryDirectory(prefix="simple_propka_") as temp_dir:
        preview_pdb = _generate_preview_prepared_pdb(
            input_config,
            prepare_config,
            protonation_config=None,
            output_dir=Path(temp_dir),
        )
        for index, ph in enumerate(ph_values, start=1):
            if len(ph_values) > 1:
                console.print(
                    Panel(
                        f"Reviewing PROPKA suggestions for pH {_format_ph_display(ph)} "
                        f"({index} of {len(ph_values)}).",
                        border_style="bright_cyan",
                        box=box.ROUNDED,
                    )
                )
            with console.status(
                "[blink bold cyan]Processing...[/] Running PROPKA on the cleaned preview structure.",
                spinner="dots",
            ):
                prediction = predict_protonation_prediction(
                    preview_pdb,
                    prepare_config,
                    ph=ph,
                    structure_is_prepared=True,
                    excluded_residue_locators=excluded_boundary_locators,
                )
            warning_key = tuple(prediction.warnings)
            if warning_key and warning_key not in shown_warning_sets:
                _display_protonation_warnings(prediction.warnings)
                shown_warning_sets.add(warning_key)

            signature = _protonation_prediction_signature(prediction)
            previous_phs = _matching_previous_phs(signature, retained_variants)
            if previous_phs and not _confirm_duplicate_ph_protonation_state(ph=ph, previous_phs=previous_phs):
                console.print(
                    f"[dim]Skipping pH {_format_ph_display(ph)} because it matches a previously retained protonation state.[/dim]"
                )
                continue

            if not prediction.metal_coordination_candidates and not prediction.propka_candidates:
                console.print(
                    "[dim]PROPKA did not suggest any supported sidechain protonation-state changes after the selected "
                    "cleanup steps.[/dim]"
                )
                protonation_config = ProtonationConfig(
                    enabled=True,
                    ph=ph,
                    engine=ProtonationEngine.PROPKA,
                    selected_changes=[],
                )
            else:
                numbered_changes = _display_protonation_candidates(prediction, ph=ph)
                selected_changes = _prompt_selected_numbered_protonation_changes(numbered_changes)
                if not selected_changes:
                    console.print("[dim]No protonation-state changes were selected for application.[/dim]")
                protonation_config = ProtonationConfig(
                    enabled=True,
                    ph=ph,
                    engine=ProtonationEngine.PROPKA,
                    selected_changes=selected_changes,
                )
            retained_variants.append(
                _ProtonationVariant(
                    protonation_config=protonation_config,
                    ph=ph,
                    ph_token=_ph_suffix_token(ph),
                    signature=signature,
                )
            )

    if not retained_variants:
        return [_ProtonationVariant(protonation_config=ProtonationConfig())]
    return retained_variants


def _prompt_protonation_config(
    *,
    input_config: InputConfig,
    prepare_config: PrepareConfig,
    summary,
) -> ProtonationConfig:
    return _prompt_protonation_variants(
        input_config=input_config,
        prepare_config=prepare_config,
        summary=summary,
    )[0].protonation_config


def _protein_force_field_choices(amber_env) -> list[WizardChoice]:
    available = amber_env.available_protein_force_fields() or ["ff19SB", "ff14SB", "ff99SB", "ff99SBildn"]
    return [
        WizardChoice(
            force_field,
            force_field,
            PROTEIN_FF_DESCRIPTIONS.get(force_field, f"Protein force field available in {force_field}."),
        )
        for force_field in available
    ]


def _nonstandard_molecule_choices(amber_env, *, detected_resp_resume: bool = False) -> list[WizardChoice]:
    available = amber_env.available_small_molecule_force_fields()
    if not available:
        available = ["gaff2", "gaff"]
    descriptions = dict(SMALL_MOLECULE_MODE_DESCRIPTIONS)
    if detected_resp_resume:
        descriptions["gaff2"] = (
            "Use Antechamber with GAFF2 atom types, then apply the detected RESP charges and run parmchk2 "
            "to generate the bonded-parameter frcmod."
        )
        descriptions["gaff"] = (
            "Use Antechamber with the older GAFF atom-type set, then apply the detected RESP charges and run "
            "parmchk2 to generate the bonded-parameter frcmod."
        )
    choices = [
        WizardChoice(
            force_field,
            force_field.upper(),
            descriptions[force_field],
            enabled=force_field in available,
        )
        for force_field in ("gaff2", "gaff")
    ]
    choices.append(
        WizardChoice(
            LigandMode.MANUAL.value,
            "Manual Amber files",
            SMALL_MOLECULE_MODE_DESCRIPTIONS[LigandMode.MANUAL.value],
        )
    )
    return choices


def _charge_method_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            ChargeMethod.FULL_RESP.value,
            "Full RESP",
            CHARGE_METHOD_DESCRIPTIONS[ChargeMethod.FULL_RESP.value],
        ),
        WizardChoice(
            ChargeMethod.RESP_ANTECHAMBER.value,
            "RESP + Antechamber",
            CHARGE_METHOD_DESCRIPTIONS[ChargeMethod.RESP_ANTECHAMBER.value],
        ),
        WizardChoice(
            ChargeMethod.ANTECHAMBER.value,
            "Antechamber AM1-BCC",
            CHARGE_METHOD_DESCRIPTIONS[ChargeMethod.ANTECHAMBER.value],
        ),
    ]


def _prompt_charge_method() -> ChargeMethod:
    choices = _charge_method_choices()
    _display_choice_table("Ligand charge workflow", choices)
    selected = _prompt_choice(
        "Choose how to derive ligand charges",
        choices,
        default_key=ChargeMethod.ANTECHAMBER.value,
    )
    return normalize_charge_method(selected)


def _prompt_resp_existing_action(*, has_completed_result: bool) -> RespApplyMode:
    if has_completed_result:
        console.print(
            "[bold cyan]A matching RESP result was found for this ligand.[/bold cyan] "
            "You can apply it directly, start a new RESP setup directory, or replace the previous assets."
        )
        choices = {
            "a": RespApplyMode.APPLY_EXISTING,
            "n": RespApplyMode.NEW_DIRECTORY,
            "r": RespApplyMode.REBUILD,
        }
        prompt = "RESP handling ([a]pply existing / [n]ew directory / [r]eplace existing)"
        default = "a"
    else:
        console.print(
            "[bold cyan]A matching RESP setup directory already exists, but no completed RESP charge file was found.[/bold cyan]"
        )
        console.print(
            "[bold yellow]Choosing replace existing deletes the previous RESP job directory before fresh assets are generated.[/bold yellow]"
        )
        choices = {
            "n": RespApplyMode.NEW_DIRECTORY,
            "r": RespApplyMode.REBUILD,
        }
        prompt = "RESP handling ([n]ew directory / [r]eplace existing)"
        default = "n"
    while True:
        response = typer.prompt(prompt, default=default).strip().lower()
        if response in choices:
            return choices[response]


def _lookup_assignment(
    assignments: list[LigandParameterAssignment],
    residue_name: str,
    *,
    default_charge: int = 0,
    default_multiplicity: int = 1,
) -> tuple[int, int]:
    residue_key = residue_name.strip().upper()
    for item in assignments:
        if item.residue_name == residue_key:
            return int(item.net_charge), int(item.multiplicity)
    return default_charge, default_multiplicity


def _build_resp_seed_files(
    *,
    input_config: InputConfig,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
    output_dir_path: Path,
    charge_method: ChargeMethod = ChargeMethod.RESP_ANTECHAMBER,
) -> tuple[str, str]:
    seed_dir = output_dir_path / "01_prepare" / "_resp_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    selected_charge_method = normalize_charge_method(charge_method)
    canonical_source = prepare_canonical_small_molecule_mol2(
        source_file=input_config.small_molecule_files[0],
        residue_name=residue_name,
        output_dir=seed_dir,
        split_supported_metals=selected_charge_method != ChargeMethod.FULL_RESP,
        canonical_filename=(
            f"{residue_name}_full_resp_input.mol2"
            if selected_charge_method == ChargeMethod.FULL_RESP
            else None
        ),
    )
    molecule = load_molecule(canonical_source)
    fingerprint = molecule_fingerprint(
        input_config.small_molecule_files[0],
        residue_name=residue_name,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    default_session = build_default_session_state(
        molecule,
        residue_name=residue_name,
        fingerprint=fingerprint,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    default_session["canonical_source_file"] = str(canonical_source)
    default_session["charge_method"] = selected_charge_method.value
    session_state = launch_resp_editor(session_state=default_session, output_dir=seed_dir)
    if session_state.get("editor_mode") == "cancelled":
        raise RuntimeError("RESP popup was cancelled before any NWChem assets were generated.")
    if session_state.get("editor_mode") == "auto_defaults":
        warning = str(session_state.get("editor_warning") or "No popup backend could be launched.")
        raise RuntimeError(
            "RESP popup could not be launched, so the workflow stopped before asset generation.\n"
            f"{warning}\n"
            "Please fix the popup backend and rerun the RESP setup."
        )
    if session_state.get("editor_warning"):
        console.print(
            "[bold yellow]RESP popup note:[/bold yellow] "
            f"{session_state['editor_warning']} "
            "Continuing with the default RESP preset and auto-suggested equality groups."
        )
    session_path = seed_dir / "resp_popup_state.json"
    group_path = seed_dir / "group_constraints.json"
    session_state["canonical_source_file"] = str(canonical_source)
    session_path.write_text(json.dumps(session_state, indent=2, sort_keys=True), encoding="utf-8")
    group_path.write_text(json.dumps(session_state["group_constraints"], indent=2, sort_keys=True), encoding="utf-8")
    return str(session_path), str(group_path)


def _existing_resp_candidates_for_small_molecule(
    *,
    input_config: InputConfig,
    residue_name: str,
    net_charge: int,
    multiplicity: int,
) -> list[object]:
    fingerprint = molecule_fingerprint(
        input_config.small_molecule_files[0],
        residue_name=residue_name,
        net_charge=net_charge,
        multiplicity=multiplicity,
    )
    return find_resp_job_candidates(search_root=Path.cwd(), fingerprint=fingerprint)


def _existing_detectable_resp_candidates_for_source_file(*, input_config: InputConfig) -> list[object]:
    candidates = find_resp_source_candidates(
        search_root=Path.cwd(),
        source_file=input_config.small_molecule_files[0],
    )
    return [candidate for candidate in candidates if getattr(candidate, "ready_to_continue", False)]


def _prompt_detected_resp_resume_option(*, input_config: InputConfig) -> object | None:
    candidates = _existing_detectable_resp_candidates_for_source_file(input_config=input_config)
    if not candidates:
        return None

    choices = [
        WizardChoice(
            "fresh",
            "Fresh setup",
            "Ignore the detected RESP results for now and start a new small-molecule setup from the input file. "
            "If the default output directory already exists, SIMPLE will continue in a new suffixed directory such as TEST_1.",
        )
    ]
    candidate_lookup: dict[str, object] = {}
    for index, candidate in enumerate(candidates, start=1):
        payload = getattr(candidate, "payload", {}) or {}
        choice_key = f"resume_{index}"
        residue_name = str(payload.get("residue_name") or "LIG")
        net_charge = int(payload.get("net_charge") or 0)
        multiplicity = int(payload.get("multiplicity") or 1)
        created_at = str(payload.get("created_at") or "unknown time")
        if getattr(candidate, "completed", False):
            status_text = "RESP charge files are ready to apply."
        else:
            status_text = "RESP output artifacts were detected and the charges will be refit locally before Amber setup continues."
        description = (
            f"Continue from the RESP result in {candidate.job_dir}. "
            f"{status_text} Residue {residue_name}, charge {net_charge}, multiplicity {multiplicity}, created {created_at}."
        )
        choices.append(WizardChoice(choice_key, f"Use detected RESP result #{index}", description))
        candidate_lookup[choice_key] = candidate

    _display_choice_table("Detected RESP job/result folders for this small molecule", choices)
    selected = _prompt_choice(
        "Choose how to start this small-molecule workflow",
        choices,
        default_key="fresh",
    )
    if selected == "fresh":
        return None
    chosen = candidate_lookup[selected]
    payload = getattr(chosen, "payload", {}) or {}
    console.print(
        "[bold cyan]Detected RESP resume selected:[/bold cyan] "
        f"{chosen.job_dir} "
        f"(residue {payload.get('residue_name')}, charge {payload.get('net_charge')}, multiplicity {payload.get('multiplicity')})."
    )
    return chosen


def _water_model_choices(
    amber_env,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
    require_official_126: bool = False,
    c4_parameter_set: DESC4ParameterSet | None = None,
) -> list[WizardChoice]:
    available = amber_env.available_water_models() or ["opc", "spce", "tip3p", "opc3", "tip5p"]
    metal_workflow = include_monovalent or include_multivalent
    dual_supported_order = {name: index for index, name in enumerate(DUAL_SUPPORTED_METAL_TI_WATER_MODELS)}
    ordered_available = sorted(
        available,
        key=lambda item: (
            0 if metal_workflow and item.lower() in dual_supported_order else 1,
            dual_supported_order.get(item.lower(), len(dual_supported_order)),
            item.lower(),
        ),
    )
    choices: list[WizardChoice] = []
    for water_model in ordered_available:
        normalized = water_model.lower()
        if normalized in HIDDEN_WATER_MODELS or normalized not in SUPPORTED_WATER_MODELS:
            continue
        missing_1264_sets = _missing_required_1264_sets(
            amber_env,
            water_model,
            include_monovalent=include_monovalent,
            include_multivalent=include_multivalent,
            c4_parameter_set=c4_parameter_set,
        )
        missing_126_sets = _missing_required_126_sets(
            amber_env,
            water_model,
            include_monovalent=include_monovalent,
            include_multivalent=include_multivalent,
        )
        description = WATER_MODEL_DESCRIPTIONS.get(
            normalized,
            f"Water model loaded from leaprc.water.{water_model}.",
        )
        if metal_workflow and missing_126_sets:
            description = (
                f"{description} Not recommended for metalloprotein / TI workflows because no official Amber 12-6 "
                "metal-ion path was detected for this model."
            )
        choices.append(
            WizardChoice(
                water_model,
                water_model.upper() if water_model.islower() else water_model,
                description,
                enabled=not (missing_126_sets if require_official_126 else missing_1264_sets),
            )
        )
    return choices


def _missing_required_126_sets(
    amber_env,
    water_model: str,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
) -> list[str]:
    missing: list[str] = []
    if include_monovalent and not amber_env.has_matching_monovalent_126(water_model):
        missing.append("1+/anion")
    if include_multivalent and not amber_env.has_matching_multivalent_126(water_model):
        missing.append("2+/3+/4+")
    return missing


def _missing_required_1264_sets(
    amber_env,
    water_model: str,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
    c4_parameter_set: DESC4ParameterSet | None = None,
) -> list[str]:
    if (
        c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL
        and water_model.lower() != "opc"
        and (include_monovalent or include_multivalent)
    ):
        return [
            label
            for needed, label in (
                (include_monovalent, "1+/anion"),
                (include_multivalent, "2+/3+/4+"),
            )
            if needed
        ]
    include_bundled_opc = c4_parameter_set != DESC4ParameterSet.SPCE_LIMERZ
    missing: list[str] = []
    if include_monovalent and not amber_env.matching_monovalent_1264_files(
        water_model,
        include_bundled_opc=include_bundled_opc,
    ):
        missing.append("1+/anion")
    if include_multivalent and not amber_env.matching_multivalent_1264_files(
        water_model,
        include_bundled_opc=include_bundled_opc,
    ):
        missing.append("2+/3+/4+")
    return missing


def _required_1264_status(
    amber_env,
    water_model: str,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
    c4_parameter_set: DESC4ParameterSet | None = None,
) -> str:
    if not include_monovalent and not include_multivalent:
        return "[dim]Not needed[/dim]"
    missing = _missing_required_1264_sets(
        amber_env,
        water_model,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
        c4_parameter_set=c4_parameter_set,
    )
    if missing:
        return "[bold red]Unavailable for selected 12-6-4 requirements[/bold red]"
    return "[bold cyan]Ready[/bold cyan]"


def _required_126_ti_status(
    amber_env,
    water_model: str,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
) -> str:
    if not include_monovalent and not include_multivalent:
        return "[dim]N/A[/dim]"
    missing = _missing_required_126_sets(
        amber_env,
        water_model,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
    )
    if missing:
        return "[bold yellow]Warning[/bold yellow]"
    if water_model.lower() in DUAL_SUPPORTED_METAL_TI_WATER_MODELS:
        return "[bold green]Recommended[/bold green]"
    return "[bold cyan]Ready[/bold cyan]"


def _enabled_water_choice_numbers(choices: list[WizardChoice]) -> dict[int, WizardChoice]:
    numbered: dict[int, WizardChoice] = {}
    next_number = 1
    for choice in choices:
        if not choice.enabled:
            continue
        numbered[next_number] = choice
        next_number += 1
    return numbered


def _display_water_model_table(
    amber_env,
    choices: list[WizardChoice],
    *,
    include_monovalent: bool,
    include_multivalent: bool,
    require_official_126: bool = False,
    c4_parameter_set: DESC4ParameterSet | None = None,
) -> None:
    numbered_choices = _enabled_water_choice_numbers(choices)
    reverse_numbers = {choice.key: number for number, choice in numbered_choices.items()}
    table = Table(title="Water model options", box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Choice", style="bold white")
    table.add_column("1+/anion", style="cyan", justify="center", no_wrap=True)
    table.add_column("2+/3+/4+", style="cyan", justify="center", no_wrap=True)
    table.add_column("12-6-4 status", style="white")
    table.add_column("Metal/TI 12-6", style="white")
    table.add_column("Description", style="white")
    for choice in choices:
        table.add_row(
            str(reverse_numbers.get(choice.key, "")),
            choice.label,
            "[bold cyan]Yes[/bold cyan]"
            if amber_env.matching_monovalent_1264_files(
                choice.key,
                include_bundled_opc=c4_parameter_set != DESC4ParameterSet.SPCE_LIMERZ,
            )
            else "[dim]No[/dim]",
            "[bold cyan]Yes[/bold cyan]"
            if amber_env.matching_multivalent_1264_files(
                choice.key,
                include_bundled_opc=c4_parameter_set != DESC4ParameterSet.SPCE_LIMERZ,
            )
            else "[dim]No[/dim]",
            _required_1264_status(
                amber_env,
                choice.key,
                include_monovalent=include_monovalent,
                include_multivalent=include_multivalent,
                c4_parameter_set=c4_parameter_set,
            ),
            _required_126_ti_status(
                amber_env,
                choice.key,
                include_monovalent=include_monovalent,
                include_multivalent=include_multivalent,
            ),
            choice.description,
        )
    if require_official_126:
        console.print(
            "[dim]TI setup requires an official Amber 12-6 ion family for the selected water model unless you "
            "explicitly provide compatible custom ion frcmods.[/dim]"
        )
    console.print(table)


def _prompt_water_model_selection(
    amber_env,
    *,
    include_monovalent: bool,
    include_multivalent: bool,
    require_official_126: bool = False,
    c4_parameter_set: DESC4ParameterSet | None = None,
    default_water_model: str = "opc",
) -> tuple[str, list[str]]:
    choices = _water_model_choices(
        amber_env,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
        require_official_126=require_official_126,
        c4_parameter_set=c4_parameter_set,
    )
    numbered_choices = _enabled_water_choice_numbers(choices)
    if not numbered_choices:
        if require_official_126:
            raise ValueError(
                "No supported water model with an official Amber 12-6 ion family is available for the selected "
                "metal/TI requirements in this Amber environment."
            )
        raise ValueError(
            "No supported water model is available for the selected 12-6-4 requirements in this Amber environment."
        )
    while True:
        _display_water_model_table(
            amber_env,
            choices,
            include_monovalent=include_monovalent,
            include_multivalent=include_multivalent,
            require_official_126=require_official_126,
            c4_parameter_set=c4_parameter_set,
        )
        default_number = next(
            (number for number, choice in numbered_choices.items() if choice.key == default_water_model),
            min(numbered_choices),
        )
        raw = typer.prompt("Choose the water model", default=str(default_number)).strip()
        if raw.isdigit() and int(raw) in numbered_choices:
            selected = numbered_choices[int(raw)].key
            if (
                not require_official_126
                and (include_monovalent or include_multivalent)
                and _missing_required_126_sets(
                    amber_env,
                    selected,
                    include_monovalent=include_monovalent,
                    include_multivalent=include_multivalent,
                )
            ):
                console.print(
                    "[bold yellow]Warning:[/bold yellow] This water model can be used for the main metalloprotein "
                    "workflow, but SIMPLE did not detect an official Amber 12-6 metal-ion path for TI. "
                    "If you plan to run TI later, prefer one of the recommended models shown at the top of the table."
                )
            return selected, []
        available_text = ", ".join(str(number) for number in sorted(numbered_choices))
        console.print(f"[bold red]Please enter one of the selectable numbers: {available_text}.[/bold red]")


def _resolve_water_model_requirements_after_salt(
    amber_env,
    *,
    water_model: str,
    custom_ion_frcmods: list[str],
    include_monovalent: bool,
    include_multivalent: bool,
    c4_parameter_set: DESC4ParameterSet | None = None,
) -> tuple[str, list[str]]:
    if custom_ion_frcmods:
        return water_model, custom_ion_frcmods

    missing_sets = _missing_required_1264_sets(
        amber_env,
        water_model,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
        c4_parameter_set=c4_parameter_set,
    )
    if not missing_sets:
        return water_model, custom_ion_frcmods

    console.print(
        "[bold yellow]Salt selection update:[/bold yellow] The current water model is unavailable for the "
        "selected 12-6-4 requirements after adding the selected solvent ions. Please choose a compatible model."
    )
    return _prompt_water_model_selection(
        amber_env,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
        c4_parameter_set=c4_parameter_set,
        default_water_model=(
            "spce" if c4_parameter_set == DESC4ParameterSet.SPCE_LIMERZ else "opc"
        ),
    )


def _box_shape_choices() -> list[WizardChoice]:
    return [
        WizardChoice(BoxShape.OCT.value, "Truncated octahedron", BOX_SHAPE_DESCRIPTIONS[BoxShape.OCT.value]),
        WizardChoice(BoxShape.CUBIC.value, "Cubic box", BOX_SHAPE_DESCRIPTIONS[BoxShape.CUBIC.value]),
    ]


def _metal_model_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            MetalModel.MODEL_1264.value,
            "12-6-4 ion model",
            "Implemented now. This is the currently supported option for metal-site workflows.",
        ),
        WizardChoice(
            MetalModel.MCPB.value,
            "MCPB.py bonded model",
            "Under development. Listed for roadmap visibility but not selectable yet.",
            enabled=False,
        ),
        WizardChoice(
            MetalModel.QM.value,
            "QM-derived model",
            "Under development. Listed for roadmap visibility but not selectable yet.",
            enabled=False,
        ),
    ]


def _metal_c4_parameter_set_choices(
    metal_species: list[tuple[str, int]] | None = None,
) -> list[WizardChoice]:
    unsupported_duvail = sorted(
        {
            (element.title(), int(charge))
            for element, charge in (metal_species or [])
            if not c4_parameter_set_supports_metal_charge(
                DESC4ParameterSet.OPC_DUVAIL,
                element,
                charge,
            )
        }
    )
    duvail_description = (
        "Use the bundled Duvail Ln3+ ion parameters, polarizability data, and C4 values with OPC water."
    )
    if unsupported_duvail:
        unsupported_text = ", ".join(f"{element}{charge}+" for element, charge in unsupported_duvail)
        duvail_description += f" Unavailable because the bundled files do not contain {unsupported_text}."
    return [
        WizardChoice(
            DESC4ParameterSet.OPC_DUVAIL.value,
            "OPC + Duvail",
            duvail_description,
            enabled=not unsupported_duvail,
        ),
        WizardChoice(
            DESC4ParameterSet.SPCE_LIMERZ.value,
            "SPC/E + Li/Merz",
            "Use compatible Amber Li/Merz 12-6-4 ion files for the selected water model (SPC/E is the default).",
        ),
    ]


def _salt_mode_choices(estimated_charge: int | None) -> list[WizardChoice]:
    no_salt_description = "Do not add counter-ions or salt."
    if estimated_charge not in {None, 0}:
        no_salt_description = "Do not add counter-ions or salt. Not recommended because the estimated charge is non-zero."
    return [
        WizardChoice(
            SaltMode.NEUTRALIZE.value,
            "Neutralize only",
            "Add only enough counter-ions to neutralize the estimated net charge.",
        ),
        WizardChoice(
            SaltMode.CONCENTRATION.value,
            "Add by molarity",
            "Neutralize first, then add additional salt at a target concentration in mol/L.",
        ),
        WizardChoice(
            SaltMode.COUNT.value,
            "Add by ion count",
            "Neutralize first, then add a fixed number of salt formula units.",
        ),
        WizardChoice(
            SaltMode.NONE.value,
            "Do not add salt",
            no_salt_description,
        ),
    ]


def _salt_kind_choices() -> list[WizardChoice]:
    return [
        WizardChoice(SaltKind.NACL.value, "NaCl", "Sodium chloride."),
        WizardChoice(SaltKind.CACL2.value, "CaCl2", "Calcium chloride."),
        WizardChoice(SaltKind.KCL.value, "KCl", "Potassium chloride."),
    ]


def _protocol_choices() -> list[WizardChoice]:
    return [
        WizardChoice(ProtocolKind.FIFTEEN_STEP.value, "15-step protocol", PROTOCOL_DESCRIPTIONS[ProtocolKind.FIFTEEN_STEP.value]),
        WizardChoice(ProtocolKind.FOUR_STEP.value, "4-step protocol", PROTOCOL_DESCRIPTIONS[ProtocolKind.FOUR_STEP.value]),
    ]


def _execution_profile_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            SlurmProfile.CPU.value,
            "CPU / pmemd.MPI",
            "Generate a Slurm script that uses `srun` with `pmemd.MPI`.",
        ),
        WizardChoice(
            SlurmProfile.GPU.value,
            "GPU / pmemd.cuda",
            "Generate a Slurm script that runs `pmemd.cuda` on a GPU node.",
        ),
    ]


def _salt_ion_names(kind: SaltKind) -> tuple[str | None, str | None]:
    if kind == SaltKind.NACL:
        return "Na+", "Cl-"
    if kind == SaltKind.KCL:
        return "K+", "Cl-"
    if kind == SaltKind.CACL2:
        return "Ca2+", "Cl-"
    return None, None


def _predict_neutralizing_ions(
    net_charge: int | None,
    salt_kind: SaltKind,
    neutralization_ion: NeutralizationIon = NeutralizationIon.SALT_DEFAULT,
) -> tuple[dict[str, int], int | None]:
    cation_name, anion_name = _salt_ion_names(salt_kind)
    if net_charge is None or cation_name is None or anion_name is None:
        return {}, net_charge

    current_charge = int(net_charge)
    if neutralization_ion not in {NeutralizationIon.AUTO, NeutralizationIon.SALT_DEFAULT}:
        ion_name = neutralization_ion.value
        ion_charge = ION_FORMAL_CHARGES[ion_name]
        if current_charge == 0:
            return {}, 0
        if current_charge * ion_charge >= 0:
            raise ValueError(
                f"Neutralization ion {ion_name} has the wrong charge sign for {current_charge:+d}."
            )
        count = int(math.ceil(abs(current_charge) / abs(ion_charge)))
        return {ion_name: count}, current_charge + count * ion_charge
    cation_charge = ION_FORMAL_CHARGES[cation_name]
    anion_charge = ION_FORMAL_CHARGES[anion_name]
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


def _combine_ion_counts(*ion_maps: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for ion_map in ion_maps:
        for ion_name, count in ion_map.items():
            if count > 0:
                combined[ion_name] = combined.get(ion_name, 0) + count
    return combined


def _display_ion_plan_summary(
    estimated_charge: int | None,
    salt_config: SaltConfig,
    des_config: DESConfig | None = None,
) -> None:
    rows: list[tuple[str, str]] = [("Estimated charge before ions", _format_charge(estimated_charge))]
    if salt_config.mode == SaltMode.NONE:
        rows.extend(
            [
                ("DES box volume", "Calculated during system build"),
                ("Added cations", "0"),
                ("Added anions", "0"),
                ("Expected final charge", _format_charge(estimated_charge)),
            ]
        )
    else:
        neutralization_ion = salt_config.neutralization_ion
        if des_config is not None and estimated_charge is not None:
            neutralization_ion = resolve_des_neutralization_ion(
                des_config,
                salt_config,
                float(estimated_charge),
            )
        neutralizing_ions, expected_final_charge = _predict_neutralizing_ions(
            estimated_charge,
            salt_config.kind,
            neutralization_ion,
        )
        extra_ions = calculate_salt_ions(0, salt_config) if salt_config.mode == SaltMode.COUNT else {}
        total_ions = _combine_ion_counts(neutralizing_ions, extra_ions)
        cation_name, anion_name = _salt_ion_names(salt_config.kind)
        rows.append(("Selected ion pair", salt_config.kind.value))
        rows.append(("Neutralizing counter-ion", neutralization_ion.value))
        rows.append(
            (
                "Neutralization plan",
                ", ".join(f"{ion} x {count}" for ion, count in neutralizing_ions.items()) or "No counter-ions predicted",
            )
        )
        if salt_config.mode == SaltMode.CONCENTRATION:
            rows.append(
                (
                    "Additional salt target",
                    f"{float(salt_config.value):.3f} M as neutral salt pairs (computed from the DES box volume)",
                )
            )
        elif salt_config.mode == SaltMode.COUNT:
            rows.append(("Additional salt units", str(int(salt_config.value))))
        else:
            rows.append(("Additional salt units", "0"))
        rows.append(("DES box volume", "Calculated during system build"))
        rows.append(("Added cations", str(total_ions.get(cation_name or "", 0))))
        rows.append(("Added anions", str(total_ions.get(anion_name or "", 0))))
        if salt_config.mode == SaltMode.CONCENTRATION:
            rows.append(("Expected final charge", "Target is neutral; exact value confirmed after system build"))
        else:
            rows.append(("Expected final charge", _format_charge(expected_final_charge)))

    table = Table(title="Ion addition plan", box=box.SIMPLE_HEAVY)
    table.add_column("Item", style="bold white")
    table.add_column("Value", style="cyan")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    console.print(
        "[dim]Exact DES box volume, explicit ion count, and final charge are written after the system build step.[/dim]"
    )


def _format_charge(value: int | None) -> str:
    if value is None:
        return "Unavailable"
    if value > 0:
        return f"+{value}"
    return str(value)


def _estimate_charge_after_cleanup(
    *,
    input_config: InputConfig,
    prepare_config: PrepareConfig,
    protonation_config: ProtonationConfig | None,
    summary,
    metal_charges: list[MetalChargeAssignment] | None = None,
) -> tuple[int | None, str]:
    if input_config.source == InputSource.SMALL_MOLECULE or summary is None:
        return None, "Charge preview is unavailable before small-molecule parameterization."

    if (protonation_config and protonation_config.enabled) or prepare_config.metal_insertions:
        with TemporaryDirectory(prefix="simple_charge_preview_") as temp_dir:
            preview_dir = Path(temp_dir)
            preview_pdb = _generate_preview_prepared_pdb(
                input_config,
                prepare_config,
                protonation_config,
                preview_dir,
            )
            return _estimate_charge_from_prepared_pdb(
                preview_pdb,
                metal_charges=_metal_charges_with_preview_insertions(metal_charges, preview_dir),
            )

    structure = load_structure(summary.source_path)
    while len(structure) > 1:
        del structure[1]

    replacement_map = {item.site: item.target.title() for item in prepare_config.metal_replacements}
    deletion_sites = set(prepare_config.metal_deletions)
    kept_tokens = {token.strip() for token in prepare_config.kept_ligands}
    explicit_metal_charges = {item.site: int(item.charge) for item in metal_charges or []}
    charge = 0
    metal_index = 0

    for chain in structure[0]:
        for residue in chain:
            classification = classify_residue(residue)
            key = residue_key(chain.name, residue)
            residue_name = residue.name.strip()

            if classification == "water" and prepare_config.remove_waters:
                continue
            if classification == "hetero":
                keep = residue_name in kept_tokens or key in kept_tokens
                if prepare_config.remove_other_hetero and not keep:
                    continue
                continue
            if classification == "metal":
                metal_index += 1
                if prepare_config.remove_metals or metal_index in deletion_sites:
                    continue
                target = replacement_map.get(metal_index)
                if target is None and len(residue) == 1:
                    target = residue[0].element.name.title()
                charge += explicit_metal_charges.get(
                    metal_index,
                    DEFAULT_METAL_CHARGES.get(target or residue_name.title(), 0),
                )
                continue
            charge += STANDARD_RESIDUE_CHARGES.get(residue_name.upper(), 0)

    if explicit_metal_charges:
        note = (
            "Approximate net charge based on standard residue names and the oxidation states you selected "
            "for the remaining metal sites. Custom ligands and other non-standard residues are treated "
            "as neutral in this preview."
        )
    else:
        note = (
            "Approximate net charge based on standard residue names and default supported metal charges. "
            "Custom ligands and other non-standard residues are treated as neutral in this preview."
        )
    return charge, note


def _display_charge_preview(charge: int | None, note: str) -> None:
    text = Text()
    text.append(f"Estimated net charge after the selected cleanup steps: {_format_charge(charge)}\n", style="bold white")
    text.append(note, style="cyan")
    console.print(
        Panel(
            text,
            title="[bold]Charge Preview[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def _focused_restraint_display_rows(
    input_config: InputConfig,
    prepare_config: PrepareConfig,
    protonation_config: ProtonationConfig | None,
    *,
    cutoff_angstrom: float = 4.0,
) -> tuple[str, list[_FocusedRestraintDisplayRow]]:
    if input_config.source == InputSource.SMALL_MOLECULE:
        return "(:1) & !@H=", []

    with TemporaryDirectory(prefix="simple_restraint_preview_") as temp_dir:
        preview_pdb = _generate_preview_prepared_pdb(
            input_config,
            prepare_config,
            protonation_config,
            Path(temp_dir),
        )
        structure = load_structure(preview_pdb)
        while len(structure) > 1:
            del structure[1]
        selected_locators = focused_restraint_residue_locators(
            structure,
            None,
            cutoff_angstrom=cutoff_angstrom,
        )
        if not selected_locators:
            return "(:1) & !@H=", []

        direct_by_locator = {
            observation.locator: observation
            for observation in direct_metal_coordination_observations(structure, None)
        }

        rows: list[_FocusedRestraintDisplayRow] = []
        reordered_numbers: list[int] = []
        reordered_number = 0
        for chain in structure[0]:
            for residue in chain:
                reordered_number += 1
                locator = residue_locator(chain.name, residue)
                if locator not in selected_locators:
                    continue
                reordered_numbers.append(reordered_number)
                residue_label = f"{chain.name.strip() or '(blank)'}:{str(residue.seqid).strip()}"
                residue_name = residue.name.strip().upper()
                classification = classify_residue(residue)
                if classification == "metal":
                    rows.append(
                        _FocusedRestraintDisplayRow(
                            residue_label=residue_label,
                            reordered_number=reordered_number,
                            residue_name=residue_name,
                            reason="Retained metal site included in the focused restraint.",
                            style="bold cyan",
                        )
                    )
                    continue
                direct = direct_by_locator.get(locator)
                if direct is not None:
                    rows.append(
                        _FocusedRestraintDisplayRow(
                            residue_label=residue_label,
                            reordered_number=reordered_number,
                            residue_name=residue_name,
                            reason=(
                                f"Direct metal coordination via {direct.donor_atom_name}-{direct.metal_element} "
                                f"({direct.distance_angstrom:.2f} A)."
                            ),
                            style="bold yellow",
                        )
                    )
                    continue
                rows.append(
                    _FocusedRestraintDisplayRow(
                        residue_label=residue_label,
                        reordered_number=reordered_number,
                        residue_name=residue_name,
                        reason=f"Within {cutoff_angstrom:.1f} A of a retained metal site.",
                        style="white",
                    )
                )

    residue_list = ",".join(str(number) for number in reordered_numbers)
    return f"(:{residue_list}) & !@H=", rows


def _sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or "simple_run"


def _base_output_name(input_config: InputConfig) -> str:
    if input_config.source == InputSource.DES:
        return "DES"
    if input_config.source == InputSource.PDB_ID and input_config.pdb_id:
        return _sanitize_name(input_config.pdb_id)
    if input_config.source == InputSource.SMALL_MOLECULE and input_config.small_molecule_files:
        return _sanitize_name(Path(input_config.small_molecule_files[0]).stem)
    if input_config.path:
        return _sanitize_name(Path(input_config.path).stem)
    return "simple_run"


def _default_output_directory(input_config: InputConfig) -> Path:
    base_name = _base_output_name(input_config)
    candidate = Path(base_name)
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = Path(f"{base_name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _workflow_root_from_resp_job_dir(resp_job_dir: str | Path | None) -> Path | None:
    if resp_job_dir is None:
        return None
    resolved = Path(resp_job_dir).expanduser().resolve()
    if resolved.name.startswith("RESP_JOBS_"):
        return resolved.parent
    if resolved.parent.name == "resp_jobs" and resolved.parent.parent.name == "01_prepare":
        return resolved.parent.parent.parent
    return None


def _suggest_output_directory(
    input_config: InputConfig,
    *,
    resp_job_dir: str | Path | None = None,
) -> Path:
    existing_root = _workflow_root_from_resp_job_dir(resp_job_dir)
    if existing_root is not None:
        return existing_root
    return _default_output_directory(input_config)


def _base_output_directory_candidate(
    input_config: InputConfig,
    *,
    resp_job_dir: str | Path | None = None,
) -> Path:
    existing_root = _workflow_root_from_resp_job_dir(resp_job_dir)
    if existing_root is not None:
        return existing_root
    return Path(_base_output_name(input_config))


def _next_available_directory(candidate: Path, *, reserved: set[Path] | None = None) -> Path:
    reserved_paths = reserved or set()
    if candidate not in reserved_paths and not candidate.exists():
        return candidate
    suffix = 1
    while True:
        trial = candidate.parent / f"{candidate.name}_{suffix}"
        if trial not in reserved_paths and not trial.exists():
            return trial
        suffix += 1


def _variant_output_directory(
    input_config: InputConfig,
    *,
    resp_job_dir: str | Path | None = None,
    suffix_tokens: tuple[str, ...] = (),
    reserved: set[Path] | None = None,
) -> Path:
    existing_root = _workflow_root_from_resp_job_dir(resp_job_dir)
    if existing_root is not None and not suffix_tokens:
        return existing_root
    base_candidate = _base_output_directory_candidate(input_config, resp_job_dir=resp_job_dir)
    named_candidate = (
        base_candidate
        if not suffix_tokens
        else base_candidate.parent / _variant_output_name(base_candidate.name, suffix_tokens)
    )
    return _next_available_directory(named_candidate, reserved=reserved)


def _batch_output_directory(
    input_config: InputConfig,
    *,
    resp_job_dir: str | Path | None = None,
    root_suffix_tokens: tuple[str, ...] = (),
    ph_token: str | None = None,
    reserved: set[Path] | None = None,
) -> Path:
    if ph_token is None:
        return _variant_output_directory(
            input_config,
            resp_job_dir=resp_job_dir,
            suffix_tokens=root_suffix_tokens,
            reserved=reserved,
        )

    base_candidate = _base_output_directory_candidate(input_config, resp_job_dir=resp_job_dir)
    root_candidate = (
        base_candidate
        if not root_suffix_tokens
        else base_candidate.parent / _variant_output_name(base_candidate.name, root_suffix_tokens)
    )
    return _next_available_directory(root_candidate / ph_token, reserved=reserved)


def _protein_config_build_specs(
    *,
    input_config: InputConfig,
    resp_job_dir: str | Path | None,
    metal_action_plan: _MetalActionPlan,
    prepare_configs: list[PrepareConfig],
    metal_charge_assignments: list[list[MetalChargeAssignment]],
    protonation_variants: list[_ProtonationVariant],
) -> list[
    tuple[
        PrepareConfig,
        list[MetalChargeAssignment],
        _ProtonationVariant,
        Path,
        tuple[str, ...],
    ]
]:
    """Allocate output folders before either RESP setup or final MD prompts."""

    ph_batch_active = len(protonation_variants) > 1
    reserved_output_paths: set[Path] = set()
    specs: list[
        tuple[
            PrepareConfig,
            list[MetalChargeAssignment],
            _ProtonationVariant,
            Path,
            tuple[str, ...],
        ]
    ] = []
    for variant, prepare_config, assignments in zip(
        metal_action_plan.variants,
        prepare_configs,
        metal_charge_assignments,
        strict=True,
    ):
        for protonation_variant in protonation_variants:
            output_dir_path = _batch_output_directory(
                input_config,
                resp_job_dir=resp_job_dir,
                root_suffix_tokens=variant.suffix_tokens,
                ph_token=protonation_variant.ph_token if ph_batch_active else None,
                reserved=reserved_output_paths,
            )
            reserved_output_paths.add(output_dir_path)
            combined_suffix_tokens = variant.suffix_tokens + (
                (protonation_variant.ph_token,)
                if ph_batch_active and protonation_variant.ph_token
                else ()
            )
            specs.append(
                (
                    prepare_config,
                    assignments,
                    protonation_variant,
                    output_dir_path,
                    combined_suffix_tokens,
                )
            )
    return specs


def _suffixed_write_config_path(path: str | Path, suffix_tokens: tuple[str, ...]) -> Path:
    target = Path(path)
    suffix_label = _variant_suffix_label(suffix_tokens)
    if not suffix_label:
        return target
    return target.with_name(f"{target.stem}_{suffix_label}{target.suffix}")


def _save_wizard_configs(result: WizardBuildResult, write_config: str | None) -> list[Path]:
    if not write_config:
        return []

    if len(result.configs) == 1:
        return [save_config(result.configs[0], write_config)]

    saved_paths: list[Path] = []
    for config, suffix_tokens in zip(result.configs, result.variant_suffixes, strict=True):
        saved_paths.append(save_config(config, _suffixed_write_config_path(write_config, suffix_tokens)))
    return saved_paths


def _prompt_manual_ligand_files(
    *,
    title: str = "[bold]Manual Amber ligand/custom-residue files[/bold]",
    prompt_text: str = "Manual Amber file(s), comma separated",
    description: str | None = None,
) -> list[str]:
    console.print(
        Panel(
            description or manual_ligand_requirements_text(),
            title=title,
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    while True:
        manual_files = _prompt_csv(prompt_text)
        if not manual_files:
            console.print("[bold red]Please provide the manual Amber files you want to load.[/bold red]")
            continue
        if not _validate_existing_paths(manual_files, label="manual Amber file(s)"):
            continue
        bundle = validate_manual_ligand_bundle(manual_files)
        if bundle.complete:
            console.print(bundle.message)
            return manual_files
        console.print(f"[bold red]{bundle.message}[/bold red]")


def _nonstandard_parameter_targets(input_config: InputConfig, summary, kept_ligands: list[str]) -> list[str]:
    if input_config.source == InputSource.SMALL_MOLECULE:
        return [LigandsConfig().residue_name]
    if summary is None:
        return []

    selected_tokens = {token.strip() for token in kept_ligands}
    residue_names: list[str] = []
    for item in summary.hetero_residues:
        if item.residue_name not in selected_tokens and item.key not in selected_tokens:
            continue
        residue_name = item.residue_name.strip().upper()
        if residue_name not in residue_names:
            residue_names.append(residue_name)
    return residue_names


def _prompt_ligand_parameter_assignments(targets: list[str]) -> list[LigandParameterAssignment]:
    if not targets:
        return []

    console.print(
        Panel(
            "AM1-BCC needs the chemically correct net charge (-nc) and spin multiplicity (-m = 2S+1).\n"
            "These values are not inferred reliably from the structure alone.\n"
            "Please inspect the ligand carefully and enter the values that match its protonation, "
            "oxidation state, and electronic state.",
            title="[bold]AM1-BCC charge and multiplicity[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    assignments: list[LigandParameterAssignment] = []
    for residue_name in targets:
        net_charge = typer.prompt(
            f"Net charge for {residue_name}",
            default=0,
            type=int,
        )
        multiplicity = _prompt_positive_int(
            f"Spin multiplicity for {residue_name}",
            default=1,
        )
        assignments.append(
            LigandParameterAssignment(
                residue_name=residue_name,
                net_charge=net_charge,
                multiplicity=multiplicity,
            )
        )
    return assignments


def _prompt_nonstandard_molecule_config(
    amber_env,
    *,
    input_config: InputConfig,
    summary,
    kept_ligands: list[str],
    detected_resp_candidate: object | None = None,
) -> tuple[LigandMode, list[str], str, list[LigandParameterAssignment]]:
    if input_config.source != InputSource.SMALL_MOLECULE and kept_ligands:
        available = amber_env.available_small_molecule_force_fields() or ["gaff2"]
        return (
            LigandMode.MANUAL,
            _prompt_manual_ligand_files(
                title="[bold]Manual Amber files for retained protein custom residues[/bold]",
                prompt_text="Path(s) to manual Amber custom-residue file(s), comma separated",
                description=(
                    "Automatic Antechamber parameterization is currently disabled for retained non-standard "
                    "residues in protein workflows.\n"
                    "Please provide Amber-ready manual files that cover the kept residues you want to preserve "
                    "in the cleaned protein structure.\n"
                    f"{manual_ligand_requirements_text()}"
                ),
            ),
            available[0],
            [],
        )

    choices = _nonstandard_molecule_choices(
        amber_env,
        detected_resp_resume=input_config.source == InputSource.SMALL_MOLECULE and detected_resp_candidate is not None,
    )
    _display_choice_table("Non-standard molecule handling", choices)
    selection = _prompt_choice(
        "Choose how to parameterize non-standard molecules",
        choices,
        default_key=LigandMode.GAFF2.value,
    )
    if selection == LigandMode.MANUAL.value:
        available = amber_env.available_small_molecule_force_fields() or ["gaff2"]
        return LigandMode.MANUAL, _prompt_manual_ligand_files(), available[0], []
    if input_config.source == InputSource.SMALL_MOLECULE and detected_resp_candidate is not None:
        payload = getattr(detected_resp_candidate, "payload", {}) or {}
        residue_name = str(payload.get("residue_name") or LigandsConfig().residue_name).strip().upper()
        assignment = LigandParameterAssignment(
            residue_name=residue_name,
            net_charge=int(payload.get("net_charge") or 0),
            multiplicity=int(payload.get("multiplicity") or 1),
        )
        console.print(
            "[dim]Using the detected RESP result metadata for this small molecule, so the ligand charge and multiplicity "
            "prompts are skipped here.[/dim]"
        )
        return LigandMode(selection), [], selection, [assignment]
    return LigandMode(selection), [], selection, []


def _prompt_positive_float(message: str, *, default: float) -> float:
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=str(default)).strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            value = float(raw)
        except ValueError:
            console.print("[bold red]Please enter a positive number, or B to go back.[/bold red]")
            continue
        if value > 0:
            return value
        console.print("[bold red]Please enter a positive number.[/bold red]")


def _prompt_fraction(message: str, *, default: float) -> float:
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=str(default)).strip()
        if _is_back_token(raw):
            raise WizardBack()
        try:
            value = float(raw)
        except ValueError:
            console.print("[bold red]Please enter a number, or B to go back.[/bold red]")
            continue
        if 0.0 < value <= 1.0:
            return value
        console.print("[bold red]Please enter a number greater than 0 and less than or equal to 1.[/bold red]")


def _prompt_nonnegative_int(message: str, *, default: int) -> int:
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=str(default)).strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw.isdigit() and int(raw) >= 0:
            return int(raw)
        console.print("[bold red]Please enter zero, a positive integer, or B to go back.[/bold red]")


def _prompt_positive_int(message: str, *, default: int) -> int:
    while True:
        raw = typer.prompt(_back_prompt_suffix(message), default=str(default)).strip()
        if _is_back_token(raw):
            raise WizardBack()
        if raw.isdigit() and int(raw) >= 1:
            return int(raw)
        console.print("[bold red]Please enter an integer greater than or equal to 1, or B to go back.[/bold red]")


def _prompt_salt_config(
    estimated_charge: int | None,
    *,
    des_config: DESConfig | None = None,
) -> SaltConfig:
    mode_choices = _salt_mode_choices(estimated_charge)
    _display_choice_table("Ion and salt handling", mode_choices)
    mode_key = _prompt_choice("Choose how to handle counter-ions and salt", mode_choices, default_key=SaltMode.NEUTRALIZE.value)

    if mode_key == SaltMode.NONE.value:
        if estimated_charge not in {None, 0}:
            console.print("[bold yellow]Proceeding without counter-ions is usually not recommended for a charged system.[/bold yellow]")
        return SaltConfig(kind=SaltKind.NONE, mode=SaltMode.NONE, value=0)

    salt_mode = SaltMode(mode_key)
    neutralization_choices = [
        WizardChoice(
            NeutralizationIon.AUTO.value,
            "Auto (DES-native)",
            "For a positive DES box, use Br- with [N8888][Br] or Cl- with choline chloride; otherwise use a Na+/Cl- fallback.",
        ),
        WizardChoice(
            NeutralizationIon.SALT_DEFAULT.value,
            "Selected salt pair",
            "Use the counter-ion supplied by NaCl, KCl, or CaCl2.",
        ),
        WizardChoice(NeutralizationIon.BROMIDE.value, "Br-", "Use bromide to neutralize a positive box."),
        WizardChoice(NeutralizationIon.CHLORIDE.value, "Cl-", "Use chloride to neutralize a positive box."),
        WizardChoice(NeutralizationIon.SODIUM.value, "Na+", "Use sodium to neutralize a negative box."),
        WizardChoice(NeutralizationIon.POTASSIUM.value, "K+", "Use potassium to neutralize a negative box."),
    ]
    _display_choice_table("Neutralizing counter-ion", neutralization_choices)
    default_neutralization = (
        NeutralizationIon.AUTO.value if des_config is not None else NeutralizationIon.SALT_DEFAULT.value
    )
    neutralization_ion = NeutralizationIon(
        _prompt_choice(
            "Choose the neutralizing counter-ion",
            neutralization_choices,
            default_key=default_neutralization,
        )
    )
    needs_salt_pair = (
        salt_mode in {SaltMode.COUNT, SaltMode.CONCENTRATION}
        or neutralization_ion == NeutralizationIon.SALT_DEFAULT
    )
    salt_kind = SaltKind.KCL if neutralization_ion == NeutralizationIon.POTASSIUM else SaltKind.NACL
    if needs_salt_pair:
        salt_choices = _salt_kind_choices()
        _display_choice_table("Available salt/counter-ion pairs", salt_choices)
        salt_kind = SaltKind(
            _prompt_choice(
                "Choose the salt/counter-ion pair",
                salt_choices,
                default_key=SaltKind.NACL.value,
            )
        )

    if salt_mode == SaltMode.NEUTRALIZE:
        return SaltConfig(
            kind=salt_kind,
            mode=salt_mode,
            value=0,
            neutralization_ion=neutralization_ion,
        )
    if salt_mode == SaltMode.CONCENTRATION:
        concentration = _prompt_positive_float("Target salt concentration (M)", default=0.150)
        return SaltConfig(
            kind=salt_kind,
            mode=salt_mode,
            value=concentration,
            neutralization_ion=neutralization_ion,
        )
    count = _prompt_nonnegative_int("Number of salt formula units to add", default=0)
    return SaltConfig(
        kind=salt_kind,
        mode=salt_mode,
        value=count,
        neutralization_ion=neutralization_ion,
    )


def _prompt_md_protocol() -> ProtocolKind:
    choices = _protocol_choices()
    _display_choice_table("MD protocol options", choices)
    return ProtocolKind(_prompt_choice("Choose the MD protocol", choices, default_key=ProtocolKind.FIFTEEN_STEP.value))


def _prompt_custom_restraints(
    input_config: InputConfig,
    summary,
    prepare_config: PrepareConfig,
    protonation_config: ProtonationConfig | None,
    remaining_metal_sites: int,
) -> tuple[str | None, float | None]:
    if remaining_metal_sites <= 0:
        console.print("[dim]No metal sites remain, so the standard protocol restraint mask will be used.[/dim]")
        return None, None

    apply_custom = typer.confirm(
        "Keep a focused restraint on the metal site(s) and nearby amino acids after the global equilibration restraints are released?",
        default=False,
    )
    if not apply_custom:
        console.print(
            "[dim]The standard protein heavy-atom restraint mask will be used during equilibration, "
            "and the late-stage NVT/NPT/production steps will remain unrestrained.[/dim]"
        )
        return None, None

    cutoff_angstrom = 4.0
    suggested_mask, restraint_rows = _focused_restraint_display_rows(
        input_config,
        prepare_config,
        protonation_config,
        cutoff_angstrom=cutoff_angstrom,
    )
    if restraint_rows:
        table = Table(
            title=f"Residues included in the {cutoff_angstrom:.1f} A focused-restraint neighborhood",
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("Residue", style="bold white")
        table.add_column("Reordered No.", style="bold cyan", justify="right")
        table.add_column("Name", style="white")
        table.add_column("Included because", style="white")
        for row in restraint_rows:
            table.add_row(
                f"[{row.style}]{row.residue_label}[/]" if row.style != "white" else row.residue_label,
                str(row.reordered_number),
                f"[{row.style}]{row.residue_name}[/]" if row.style != "white" else row.residue_name,
                f"[{row.style}]{row.reason}[/]" if row.style != "white" else row.reason,
            )
        console.print(table)
        console.print(
            "[dim]Directly metal-coordinating residues are highlighted separately from nearby residues. "
            "The reordered numbers are the prepared-system residue indices used in the mask below.[/dim]"
        )
    console.print(
        Panel(
            "Early minimization and equilibration will still use the standard protein heavy-atom restraint mask.\n"
            "This focused mask is only kept for the late-stage site restraint after the global restraints are released.\n\n"
            f"Included residues are listed above using a {cutoff_angstrom:.1f} A cutoff from the retained metal site(s).\n\n"
            f"Suggested focused mask:\n{suggested_mask}",
            title="[bold]Focused restraint suggestion[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    restraint_mask = typer.prompt(
        "Focused restraint mask (press Enter to accept the suggested mask)",
        default=suggested_mask,
    ).strip()
    restraint_weight = _prompt_positive_float(
        "Restraint force constant (kcal/mol·A^2)",
        default=2.0,
    )
    return restraint_mask or suggested_mask, restraint_weight


def _prompt_execution_profile() -> SlurmProfile:
    choices = _execution_profile_choices()
    _display_choice_table("Execution target", choices)
    profile = SlurmProfile(_prompt_choice("Choose the execution target", choices, default_key=SlurmProfile.CPU.value))
    console.print(
        "[dim]The generated sbatch script will use default Slurm resource lines. "
        "Edit the .sbatch file directly for partition, account, ntasks, GPUs, or walltime.[/dim]"
    )
    return profile


def _build_des_wizard_result(
    *,
    input_config: InputConfig,
    des_config: DESConfig,
    write_config: str | None,
) -> WizardBuildResult:
    _print_step_header(
        2,
        "Choose DES MD Settings",
        "Use a solvent-only equilibration protocol with NPT density relaxation before production.",
    )
    console.print(
        "[bold cyan]DES protocol:[/bold cyan] "
        "unrestrained minimization -> 1 K no-SHAKE settle -> NVT warm-up -> NVT heating -> soft NPT density relaxation -> "
        "NPT equilibration -> optional 500 K NPT mixing -> NPT production."
    )
    des_protocol_choices = [
        WizardChoice(
            "mixing",
            "Mixing + production",
            "Default. Insert a 500 K / 50 ns NPT mixing step immediately before the final NPT production run.",
        ),
        WizardChoice(
            "simple",
            "Simple protocol",
            "Use the DES protocol without the high-temperature mixing step.",
        ),
    ]
    _display_choice_table("DES MD protocol variant", des_protocol_choices)
    des_mixing_enabled = (
        _prompt_choice("Choose the DES MD protocol variant", des_protocol_choices, default_key="mixing") == "mixing"
    )
    temperature = typer.prompt("Target production temperature (K)", default=300.0, type=float)
    pressure = typer.prompt("Target pressure (bar)", default=1.0, type=float)
    production_time = _prompt_positive_float("Production time (ns)", default=100.0)
    estimated_charge = int(round(estimate_des_net_charge(des_config)))
    salt_config = _prompt_salt_config(estimated_charge, des_config=des_config)
    _display_ion_plan_summary(estimated_charge, salt_config, des_config)

    _print_step_header(
        3,
        "Choose the Execution Script and Output Location",
        "Select whether to generate a CPU or GPU Slurm script. "
        "Cluster-specific resource lines can be edited directly in the generated sbatch file.",
    )
    profile = _prompt_execution_profile()
    output_dir_path = _default_output_directory(input_config)
    console.print(
        f"[bold cyan]Output directory:[/bold cyan] {output_dir_path}\n"
        "[dim]If that name already exists, an incremented suffix is applied automatically.[/dim]"
    )
    console.print("[blink bold cyan]Processing...[/] DES setup is complete. Workflow file generation will begin now.")

    config = WorkflowConfig(
        input=input_config,
        des=des_config,
        prepare=PrepareConfig(remove_waters=False, remove_other_hetero=False, remove_metals=False),
        protonation=ProtonationConfig(),
        ligands=LigandsConfig(),
        system=SystemConfig(
            water_model="opc" if des_config.c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL else "spce",
            box_shape=BoxShape.CUBIC,
            c4_parameter_set=des_config.c4_parameter_set,
            salt=salt_config,
        ),
        md=MDConfig(
            protocol=ProtocolKind.DES_SOLVENT,
            temperature_k=temperature,
            pressure_bar=pressure,
            production_time_ns=production_time,
            des_mixing_enabled=des_mixing_enabled,
        ),
        slurm=SlurmConfig(
            profile=profile,
            partition=None,
            account=None,
            ntasks=8,
            gpus=1,
            walltime="24:00:00",
            binary_override=None,
            job_name=_base_output_name(input_config),
        ),
        output_dir=str(output_dir_path),
    )
    result = WizardBuildResult(configs=[config], variant_suffixes=[()])
    result.saved_config_paths.extend(_save_wizard_configs(result, write_config))
    if result.saved_config_paths:
        console.print(f"Saved config to {result.saved_config_paths[0]}")
    return result


def _aggregate_ion_parameter_requirements(
    salt_config: SaltConfig,
    variant_metal_charge_assignments: list[list[MetalChargeAssignment]],
    prepare_configs: list[PrepareConfig] | None = None,
) -> tuple[bool, bool]:
    include_monovalent = False
    include_multivalent = False
    for index, assignments in enumerate(variant_metal_charge_assignments):
        charge_values = [item.charge for item in assignments]
        if prepare_configs is not None and index < len(prepare_configs):
            for insertion in prepare_configs[index].metal_insertions:
                charge_values.append(
                    int(insertion.charge or DEFAULT_METAL_CHARGES.get(insertion.element.title(), 2))
                )
        monovalent, multivalent = ion_parameter_requirements(
            salt_config,
            metal_charges=charge_values,
        )
        include_monovalent = include_monovalent or monovalent
        include_multivalent = include_multivalent or multivalent
    return include_monovalent, include_multivalent


def _display_batch_charge_preview(
    input_config: InputConfig,
    prepare_configs: list[PrepareConfig],
    protonation_configs: list[ProtonationConfig],
    summary,
    variant_metal_charge_assignments: list[list[MetalChargeAssignment]],
) -> int | None:
    estimated_charges: list[int | None] = []
    note = (
        "Charge preview is unavailable before small-molecule parameterization."
        if input_config.source == InputSource.SMALL_MOLECULE
        else "Estimated net charge varies across the selected metal/pH variants."
    )
    for prepare_config, metal_charges in zip(prepare_configs, variant_metal_charge_assignments, strict=True):
        for protonation_config in protonation_configs:
            charge, charge_note = _estimate_charge_after_cleanup(
                input_config=input_config,
                prepare_config=prepare_config,
                protonation_config=protonation_config,
                summary=summary,
                metal_charges=metal_charges,
            )
            estimated_charges.append(charge)
            note = charge_note

    unique_charges = sorted({charge for charge in estimated_charges if charge is not None})
    if len(unique_charges) <= 1:
        charge_value = unique_charges[0] if unique_charges else None
        _display_charge_preview(charge_value, note)
        return charge_value

    console.print(
        Panel(
            "Estimated net charge after the selected cleanup steps varies across the selected metal/pH variants: "
            + ", ".join(_format_charge(charge) for charge in unique_charges)
            + "\n"
            + "Neutralize-only and molarity modes will still adapt per output during system build. "
            + "Fixed ion-count settings will be shared across the batch.",
            title="[bold]Charge Preview[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    return None


def _workflow_output_label(output_dir: str | Path) -> str:
    path = Path(output_dir)
    leaf = path.name
    if leaf.upper().startswith("PH") and path.parent.name:
        return f"{path.parent.name}/{leaf}"
    return leaf


def execute_wizard_configs(
    configs: list[WorkflowConfig],
    *,
    from_stage: str = "prepare",
    to_stage: str = "md",
    dry_run: bool,
    failure_hint: str | None = None,
) -> int:
    failures: list[tuple[WorkflowConfig, Exception]] = []
    total = len(configs)
    outcomes: list[tuple[str, object | None]] = [("not_started", None) for _ in configs]

    for index, config in enumerate(configs, start=1):
        output_name = _workflow_output_label(config.output_dir)
        if total > 1:
            console.print(
                Panel(
                    Text(f"Starting output {index} of {total}: {output_name}"),
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                )
            )
        try:
            result = run_workflow(
                config=config,
                from_stage=from_stage,
                to_stage=to_stage,
                dry_run=dry_run,
            )
        except Exception as exc:
            failures.append((config, exc))
            outcomes[index - 1] = ("failed", exc)
            failure_message = Text()
            failure_message.append(f"Workflow failed for {output_name}:", style="bold red")
            failure_message.append(f" {exc}")
            console.print(failure_message)
            if failure_hint:
                console.print(Text(failure_hint, style="bold yellow"))
            if index < total and typer.confirm("Continue with the remaining outputs?", default=True):
                continue
            break
        site_resp_status = str(((result.get("protein_site_resp") or {}).get("status") or ""))
        if site_resp_status in {"reference_pending", "cluster_review_required", "setup_pending"}:
            outcomes[index - 1] = ("resp_pending", site_resp_status)
        elif site_resp_status == "review_required":
            outcomes[index - 1] = ("resp_not_applied", site_resp_status)
        else:
            outcomes[index - 1] = ("succeeded", site_resp_status)
        print_workflow_summary(result)

    if total > 1 or failures:
        table = Table(title="Batch workflow summary", box=box.SIMPLE_HEAVY)
        table.add_column("Output", style="bold white")
        table.add_column("Status", style="cyan")
        for config, (outcome, detail) in zip(configs, outcomes, strict=True):
            name = _workflow_output_label(config.output_dir)
            if outcome == "failed":
                status = Text("Failed", style="bold red")
                status.append(f" ({detail})")
            elif outcome == "resp_pending":
                status = Text("RESP pending", style="bold yellow")
            elif outcome == "resp_not_applied":
                status = Text("RESP not applied", style="bold yellow")
            elif outcome == "succeeded":
                status = Text("Succeeded", style="bold green")
            else:
                status = Text("Not started", style="dim")
            table.add_row(Text(name), status)
        console.print(table)

    return 0 if not failures else 1


def _prompt_new_protein_site_resp_config(
    *,
    remaining_metal_site_count: int,
    batch_charge_map: dict[tuple[int, str], int],
) -> ProteinSiteRespConfig | None:
    if remaining_metal_site_count <= 0:
        return None
    _print_step_header(
        4,
        "Choose Metal-Site Protein Charges",
        "Keep standard force-field charges or stop after preparing an expert protein-site RESP/NWChem job.",
    )
    console.print(
        Panel(
            "[bold]Standard FF[/bold] remains the validated default.\n"
            "[bold yellow]Site-specific RESP[/bold yellow] redistributes partial charges on selected coordinating "
            "protein residues while fixing the metal at its integer oxidation state and preserving every target "
            "residue's total charge. Solvation, 12-6-4 selection, MD duration, and CPU/GPU settings are deliberately "
            "deferred until the NWChem result is available.",
            title="[bold]Protein Metal-Site Charge Model[/bold]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    if not _prompt_yes_no(
        "Prepare site-specific RESP charges for the directly coordinating residues?",
        default=False,
    ):
        return None

    scope_choices = [
        WizardChoice(
            ProteinSiteRespScope.SIDECHAIN.value,
            "Side chain (recommended)",
            "Fit CB-and-beyond atoms while fixing backbone, caps, metal, water, and other QM environment charges.",
        ),
        WizardChoice(
            ProteinSiteRespScope.WHOLE_RESIDUE.value,
            "Whole residue",
            "Fit all atoms of each coordinating residue while preserving each residue's total charge.",
        ),
    ]
    _display_choice_table("Protein-site RESP scope", scope_choices)
    scope = ProteinSiteRespScope(
        _prompt_choice(
            "Choose the protein-site RESP scope",
            scope_choices,
            default_key=ProteinSiteRespScope.SIDECHAIN.value,
        )
    )
    suggestion_rows: list[str] = []
    for (_site, element), charge in sorted(batch_charge_map.items()):
        suggestion = suggested_spin_multiplicity(element, charge)
        low_spin = suggested_low_spin_multiplicity(element, charge)
        suggestion_rows.append(
            f"{element}{charge:+d}: default/high-spin "
            f"{suggestion if suggestion is not None else 'no curated suggestion'}; low-spin "
            f"{low_spin if low_spin is not None else 'no conventional alternative'}"
        )
    console.print(
        "[bold cyan]Spin guidance:[/bold cyan] "
        + (", ".join(suggestion_rows) if suggestion_rows else "No curated suggestion is available.")
        + "\n[dim]SIMPLE will now build only the unsolvated reference topology. It will then show each connected "
        "metal cluster and ask you to review the donor/fixed-environment residues and confirm the multiplicity "
        "before writing the NWChem and Tahoma files.[/dim]"
    )
    return ProteinSiteRespConfig(
        mode=ProteinSiteRespMode.RESP,
        scope=scope,
        apply_mode=RespApplyMode.NEW_DIRECTORY,
    )


def _build_protein_site_resp_setup_result(
    *,
    input_config: InputConfig,
    protein_ff: str,
    system_ligand_ff: str,
    ligands_config: LigandsConfig,
    resp_config: ProteinSiteRespConfig,
    config_build_specs: list[
        tuple[
            PrepareConfig,
            list[MetalChargeAssignment],
            _ProtonationVariant,
            Path,
            tuple[str, ...],
        ]
    ],
    write_config: str | None,
) -> WizardBuildResult:
    configs: list[WorkflowConfig] = []
    suffixes: list[tuple[str, ...]] = []
    base_job_name = _base_output_name(input_config)
    for prepare_config, assignments, protonation_variant, output_dir_path, suffix_tokens in config_build_specs:
        job_name_tokens = suffix_tokens
        if (
            protonation_variant.protonation_config.enabled
            and protonation_variant.ph_token
            and protonation_variant.ph_token not in job_name_tokens
        ):
            job_name_tokens = suffix_tokens + (protonation_variant.ph_token,)
        configs.append(
            WorkflowConfig(
                input=input_config,
                prepare=prepare_config,
                protonation=protonation_variant.protonation_config.model_copy(deep=True),
                protein_site_resp=resp_config.model_copy(deep=True),
                ligands=ligands_config.model_copy(deep=True),
                # This is a reference-only configuration.  SPC/E ordinary 12-6
                # files provide a neutral setup basis; the user chooses the
                # final 12-6-4/water pair only after RESP completes.
                system=SystemConfig(
                    protein_ff=protein_ff,
                    ligand_ff=system_ligand_ff,
                    metal_model=MetalModel.MODEL_1264,
                    apply_1264=False,
                    c4_parameter_set=DESC4ParameterSet.SPCE_LIMERZ,
                    metal_charges=assignments,
                    water_model="spce",
                    salt=SaltConfig(),
                ),
                md=MDConfig(),
                slurm=SlurmConfig(
                    profile=SlurmProfile.CPU,
                    partition=None,
                    account=None,
                    ntasks=8,
                    gpus=0,
                    walltime="24:00:00",
                    binary_override=None,
                    job_name=_workflow_job_name(base_job_name, job_name_tokens),
                ),
                output_dir=str(output_dir_path),
            )
        )
        suffixes.append(suffix_tokens)

    if len(configs) == 1:
        console.print(f"[bold cyan]Protein-site RESP case:[/bold cyan] {configs[0].output_dir}")
    else:
        table = Table(title="Protein-site RESP setup cases", box=box.SIMPLE_HEAVY)
        table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
        table.add_column("Case directory", style="bold white")
        for index, config in enumerate(configs, start=1):
            table.add_row(str(index), config.output_dir)
        console.print(table)
    console.print(
        "[blink bold cyan]Processing...[/] SIMPLE will prepare the unsolvated reference, ask for the "
        "site residues and spin multiplicity, write the NWChem/Tahoma assets, mark the workflow pending, "
        "and stop. No solvation or MD/execution settings will be requested in this run."
    )
    result = WizardBuildResult(configs=configs, variant_suffixes=suffixes)
    result.saved_config_paths.extend(_save_wizard_configs(result, write_config))
    return result


def _resume_input_config(candidate: ProteinSiteRespResumeCandidate) -> InputConfig:
    payload = candidate.payload
    source_pdb = str(payload.get("source_pdb") or "").strip()
    if source_pdb and Path(source_pdb).expanduser().exists():
        return InputConfig(source=InputSource.PDB_FILE, path=str(Path(source_pdb).expanduser().resolve()))
    prepared_pdb = str(payload.get("prepared_system_pdb") or "").strip()
    if prepared_pdb and Path(prepared_pdb).expanduser().exists():
        return InputConfig(source=InputSource.PDB_FILE, path=str(Path(prepared_pdb).expanduser().resolve()))
    for fallback in (
        candidate.workflow_root / "02_system" / "system.pdb",
        candidate.workflow_root / "02_system" / "system.unsolvated.pdb",
    ):
        if fallback.exists():
            return InputConfig(source=InputSource.PDB_FILE, path=str(fallback.resolve()))
    raise ValueError(f"The original structure for RESP job {candidate.job_dir} is no longer available.")


def _resume_cluster_config(candidate: ProteinSiteRespResumeCandidate) -> ProteinSiteRespClusterConfig:
    cluster = candidate.payload.get("cluster") or {}
    if not isinstance(cluster, dict):
        cluster = {}
    return ProteinSiteRespClusterConfig(
        metal_sites=[int(item) for item in cluster.get("metal_sites") or []],
        donor_residues=[str(item) for item in cluster.get("donor_residue_keys") or []],
        fixed_environment=[str(item) for item in cluster.get("fixed_environment_keys") or []],
        multiplicity=int(cluster.get("multiplicity") or candidate.payload.get("multiplicity") or 1),
        job_dir=str(candidate.job_dir),
    )


def _candidate_formal_metal_states(
    candidates: list[ProteinSiteRespResumeCandidate],
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    seen: set[tuple[int, str, int]] = set()
    for candidate in candidates:
        for raw_state in candidate.payload.get("formal_metal_states") or []:
            if not isinstance(raw_state, dict):
                continue
            site = int(raw_state.get("site") or 0)
            element = str(raw_state.get("element") or "").strip().title()
            charge = int(raw_state.get("formal_charge") or 0)
            key = (site, element, charge)
            if site <= 0 or not element or key in seen:
                continue
            seen.add(key)
            states.append({"site": site, "element": element, "formal_charge": charge})
    return sorted(states, key=lambda item: (int(item["site"]), str(item["element"])))


def _case_metal_charge_assignments(
    config: WorkflowConfig,
    candidates: list[ProteinSiteRespResumeCandidate],
) -> list[MetalChargeAssignment]:
    """Preserve each paused workflow's own oxidation-state assignments."""

    charge_map = {int(item.site): int(item.charge) for item in config.system.metal_charges}
    for state in _candidate_formal_metal_states(candidates):
        charge_map.setdefault(int(state["site"]), int(state["formal_charge"]))
    return [
        MetalChargeAssignment(site=site, charge=charge)
        for site, charge in sorted(charge_map.items())
    ]


def _reference_system_net_charge(workflow_root: Path) -> int | None:
    system_dir = workflow_root / "02_system"
    try:
        atoms = load_topology_atoms(
            system_dir / "system.unsolvated.prmtop",
            system_dir / "system.unsolvated.pdb",
        )
    except (OSError, RuntimeError, ValueError):
        return None
    return int(round(sum(float(atom.charge) for atom in atoms)))


def _prompt_completed_protein_site_resp_settings(
    *,
    config: WorkflowConfig,
    candidates: list[ProteinSiteRespResumeCandidate],
    amber_env,
    case_configs: list[WorkflowConfig] | None = None,
) -> WorkflowConfig:
    """Collect one shared set of final settings after NWChem RESP completes."""

    workflow_root = candidates[0].workflow_root
    states = _candidate_formal_metal_states(candidates)
    metal_charges = _case_metal_charge_assignments(config, candidates)
    metal_species = [
        (str(state["element"]), int(state["formal_charge"]))
        for state in states
    ]

    _print_step_header(
        4,
        "Choose the Final Metal Treatment",
        "The RESP result is complete. Select the final 12-6-4 parameter set before solvation.",
    )
    metal_choices = _metal_model_choices()
    _display_choice_table("Metal treatment model", metal_choices)
    metal_model = MetalModel(
        _prompt_choice(
            "Choose the metal treatment model",
            metal_choices,
            default_key=MetalModel.MODEL_1264.value,
        )
    )
    c4_parameter_set = DESC4ParameterSet.SPCE_LIMERZ
    if metal_model == MetalModel.MODEL_1264:
        parameter_choices = _metal_c4_parameter_set_choices(metal_species)
        _display_choice_table("Metal 12-6-4 parameter set", parameter_choices)
        c4_parameter_set = DESC4ParameterSet(
            _prompt_choice(
                "Choose the metal 12-6-4 parameter set",
                parameter_choices,
                default_key=DESC4ParameterSet.OPC_DUVAIL.value,
            )
        )

    _print_step_header(
        5,
        "Choose Solvation and Ion Settings",
        "Select the final solvent model, box geometry, and ion conditions now that RESP is complete.",
    )
    prompt_configs = case_configs or [config]
    candidates_by_root: dict[Path, list[ProteinSiteRespResumeCandidate]] = {}
    for candidate in candidates:
        candidates_by_root.setdefault(candidate.workflow_root.resolve(), []).append(candidate)
    assignments_by_variant = [
        _case_metal_charge_assignments(
            case_config,
            candidates_by_root.get(case_config.output_path(), candidates),
        )
        for case_config in prompt_configs
    ]
    prepare_configs = [case_config.prepare for case_config in prompt_configs]
    pre_monovalent, pre_multivalent = _aggregate_ion_parameter_requirements(
        SaltConfig(),
        assignments_by_variant,
        prepare_configs,
    )
    water_model, custom_ion_frcmods = _prompt_water_model_selection(
        amber_env,
        include_monovalent=pre_monovalent,
        include_multivalent=pre_multivalent,
        c4_parameter_set=c4_parameter_set,
        default_water_model=(
            "opc" if c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL else "spce"
        ),
    )
    box_choices = _box_shape_choices()
    _display_choice_table("Solvation box options", box_choices)
    box_shape = BoxShape(
        _prompt_choice(
            "Choose the solvation box shape",
            box_choices,
            default_key=BoxShape.OCT.value,
        )
    )
    buffer_angstrom = _prompt_positive_float("Solvent buffer (Angstrom)", default=10.0)
    estimated_charge = _reference_system_net_charge(workflow_root)
    _display_charge_preview(
        estimated_charge,
        "Exact net charge from the unsolvated topology used to generate this protein-site RESP job."
        if estimated_charge is not None
        else "The unsolvated reference charge could not be read; final TLeap will report the exact charge.",
    )
    salt_config = _prompt_salt_config(estimated_charge)
    include_monovalent, include_multivalent = _aggregate_ion_parameter_requirements(
        salt_config,
        assignments_by_variant,
        prepare_configs,
    )
    water_model, custom_ion_frcmods = _resolve_water_model_requirements_after_salt(
        amber_env,
        water_model=water_model,
        custom_ion_frcmods=custom_ion_frcmods,
        include_monovalent=include_monovalent,
        include_multivalent=include_multivalent,
        c4_parameter_set=c4_parameter_set,
    )

    _print_step_header(
        6,
        "Choose the MD Protocol",
        "Pick the equilibration strategy and final production settings.",
    )
    protocol = _prompt_md_protocol()
    temperature = typer.prompt("Target temperature (K)", default=300.0, type=float)
    pressure = typer.prompt("Target pressure (bar)", default=1.0, type=float)
    production_time = _prompt_positive_float("Production time (ns)", default=100.0)
    restraint_input = config.input
    try:
        if restraint_input.source == InputSource.PDB_FILE and not Path(str(restraint_input.path)).expanduser().exists():
            restraint_input = _resume_input_config(candidates[0])
    except (OSError, ValueError):
        restraint_input = _resume_input_config(candidates[0])
    restraint_mask, restraint_weight = _prompt_custom_restraints(
        restraint_input,
        None,
        config.prepare,
        config.protonation,
        len({int(state["site"]) for state in states}),
    )

    _print_step_header(
        7,
        "Choose the Execution Script",
        "Select CPU/pmemd.MPI or GPU/pmemd.cuda for the completed workflow.",
    )
    profile = _prompt_execution_profile()

    config.system = SystemConfig(
        protein_ff=config.system.protein_ff,
        ligand_ff=config.system.ligand_ff,
        metal_model=metal_model,
        apply_1264=metal_model == MetalModel.MODEL_1264,
        c4_parameter_set=c4_parameter_set,
        metal_charges=metal_charges,
        water_model=water_model,
        box_shape=box_shape,
        buffer_angstrom=buffer_angstrom,
        salt=salt_config,
        custom_ion_frcmods=custom_ion_frcmods,
    )
    config.md = MDConfig(
        protocol=protocol,
        temperature_k=temperature,
        pressure_bar=pressure,
        production_time_ns=production_time,
        focused_restraint_mask=restraint_mask or None,
        focused_restraint_mask_numbering=(
            ResidueMaskNumbering.PREPARED if restraint_mask else ResidueMaskNumbering.PDB
        ),
        focused_restraint_weight=restraint_weight,
    )
    config.slurm = SlurmConfig(
        profile=profile,
        partition=None,
        account=None,
        ntasks=8,
        gpus=1 if profile == SlurmProfile.GPU else 0,
        walltime="24:00:00",
        binary_override=None,
        job_name=workflow_root.name,
    )
    return config


def _build_protein_site_resp_resume_result(
    selection: _ProteinSiteRespResumeSelection,
    write_config: str | None,
    *,
    amber_env,
) -> WizardBuildResult:
    grouped: dict[Path, list[ProteinSiteRespResumeCandidate]] = {}
    for candidate in selection.candidates:
        grouped.setdefault(candidate.workflow_root, []).append(candidate)

    if not grouped:
        raise ValueError("At least one completed protein-site RESP job must be selected.")

    continuation_cases: list[
        tuple[Path, list[ProteinSiteRespResumeCandidate], WorkflowConfig, ProteinSiteRespScope]
    ] = []
    suffixes: list[tuple[str, ...]] = []
    summary_table = Table(title="Protein-site RESP continuation plan", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Workflow output", style="bold cyan", overflow="fold")
    summary_table.add_column("Selected RESP job(s)", style="white", justify="right")
    summary_table.add_column("Continuation settings", style="green")

    for workflow_root, candidates in grouped.items():
        snapshot_path = workflow_root / "workflow_config.toml"
        if snapshot_path.exists():
            config = load_config(snapshot_path)
        else:
            config = WorkflowConfig(
                input=_resume_input_config(candidates[0]),
                output_dir=str(workflow_root),
            )

        scopes = {
            ProteinSiteRespScope(str(candidate.payload.get("scope") or ProteinSiteRespScope.SIDECHAIN.value))
            for candidate in candidates
        }
        if len(scopes) != 1:
            raise ValueError(
                f"Selected RESP jobs under {workflow_root} use different fitting scopes and cannot be applied together."
            )
        config.output_dir = str(workflow_root)
        continuation_cases.append((workflow_root, candidates, config, next(iter(scopes))))

    all_candidates = [
        candidate
        for _workflow_root, candidates, _config, _scope in continuation_cases
        for candidate in candidates
    ]
    shared_config = _prompt_completed_protein_site_resp_settings(
        config=continuation_cases[0][2].model_copy(deep=True),
        candidates=all_candidates,
        amber_env=amber_env,
        case_configs=[case[2] for case in continuation_cases],
    )

    configs: list[WorkflowConfig] = []
    for workflow_root, candidates, config, scope in continuation_cases:
        case_metal_charges = _case_metal_charge_assignments(config, candidates)
        config.system = shared_config.system.model_copy(
            deep=True,
            update={
                "protein_ff": config.system.protein_ff,
                "ligand_ff": config.system.ligand_ff,
                "metal_charges": case_metal_charges,
            },
        )
        config.md = shared_config.md.model_copy(deep=True)
        config.slurm = shared_config.slurm.model_copy(
            deep=True,
            update={"job_name": workflow_root.name},
        )
        config.output_dir = str(workflow_root)
        config.protein_site_resp = ProteinSiteRespConfig(
            mode=ProteinSiteRespMode.RESP,
            scope=scope,
            apply_mode=RespApplyMode.APPLY_EXISTING,
            search_roots=[str(workflow_root)],
            job_dirs=[str(candidate.job_dir) for candidate in candidates],
            resume_existing_system=True,
            clusters=[_resume_cluster_config(candidate) for candidate in candidates],
        )
        configs.append(config)
        suffixes.append((workflow_root.name,))
        summary_table.add_row(str(workflow_root), str(len(candidates)), "shared final settings")

    console.print(summary_table)
    console.print(
        "[blink bold cyan]Processing...[/] SIMPLE will refit any grid-only result, show the charge review, "
        "build the deferred final solvated system when needed, patch the reviewed charges, and then "
        "generate the MD/Slurm files."
    )
    result = WizardBuildResult(configs=configs, variant_suffixes=suffixes)
    result.saved_config_paths.extend(_save_wizard_configs(result, write_config))
    return result


def _build_wizard_configs_once(write_config: str | None) -> WizardBuildResult:
    amber_env = detect_amber_environment()
    _print_step_header(
        1,
        "Load the Structure and Review Cleanup Options",
        "Choose a PDB ID or local structure file, inspect the detected components, "
        "and decide how to handle waters, hetero residues, and metal sites.",
    )
    prompted_input = _prompt_input_config()
    if isinstance(prompted_input, _ProteinSiteRespResumeSelection):
        return _build_protein_site_resp_resume_result(
            prompted_input,
            write_config,
            amber_env=amber_env,
        )
    if len(prompted_input) == 3:
        input_config, inspection_summary, detected_resp_candidate = prompted_input
        des_config = None
    else:
        input_config, inspection_summary, detected_resp_candidate, des_config = prompted_input

    if input_config.source == InputSource.DES:
        if des_config is None:
            raise ValueError("DES workflow selected, but no DES configuration was provided.")
        return _build_des_wizard_result(
            input_config=input_config,
            des_config=des_config,
            write_config=write_config,
        )

    if inspection_summary:
        if input_config.source == InputSource.SMALL_MOLECULE and inspection_summary.metals:
            console.print(
                "[bold cyan]Supported metal atom(s) were detected in the small-molecule input.[/bold cyan] "
                "SIMPLE will treat them with the same 12-6-4 water-model checks used for protein metal sites."
            )
            console.print(
                "[bold yellow]Ligand-charge note:[/bold yellow] For automatic GAFF/GAFF2 setup, supported metal atoms "
                "are separated before Antechamber/SQM and re-added in tleap. Enter the ligand/scaffold net charge "
                "excluding the separated metal ion charge; the metal oxidation state is handled in the metal step."
            )
        _display_summary(inspection_summary)
    if inspection_summary and input_config.source != InputSource.SMALL_MOLECULE:
        remove_waters = typer.confirm("Remove crystallographic/solvent waters?", default=True)
        remove_other_hetero = False
        kept_ligands: list[str] = []
        if inspection_summary.hetero_residues:
            remove_other_hetero = typer.confirm(
                "Remove non-protein/non-metal hetero residues?",
                default=True,
            )
            if not remove_other_hetero:
                kept_ligands = _prompt_csv(
                    "Residue names or keys to keep as custom residues (comma separated, blank for none)"
                )
            if not remove_other_hetero and not kept_ligands:
                console.print(
                    "[bold yellow]No non-standard residues were explicitly selected. "
                    "Any remaining hetero residues will stay in the cleaned structure as-is and may still need "
                    "manual Amber parameters later.[/bold yellow]"
                )
        metal_action_plan = _prompt_metal_actions(inspection_summary, input_config)
        repair_missing_loops = _prompt_missing_loop_repair(inspection_summary)
    else:
        console.print(
            "[dim]Small-molecule-only input selected. Structure cleanup questions are skipped in this step.[/dim]"
        )
        remove_waters = False
        remove_other_hetero = False
        kept_ligands = []
        metal_action_plan = _MetalActionPlan()
        repair_missing_loops = False

    base_prepare_config = PrepareConfig(
        remove_waters=remove_waters,
        remove_other_hetero=remove_other_hetero,
        remove_metals=metal_action_plan.remove_metals,
        repair_missing_loops=repair_missing_loops,
        kept_ligands=kept_ligands,
        metal_replacements=[],
        metal_deletions=metal_action_plan.metal_deletions,
        metal_insertions=metal_action_plan.metal_insertions,
    )
    variant_prepare_configs = [
        _prepare_config_with_replacements(base_prepare_config, variant.replacements)
        for variant in metal_action_plan.variants
    ]

    _print_step_header(
        2,
        "Choose Force Fields and Metal Oxidation States",
        "Select the force fields and confirm metal oxidation states before reviewing protonation-state changes. "
        "The final 12-6-4/water model is chosen later.",
    )

    needs_nonstandard_molecules = input_config.source == InputSource.SMALL_MOLECULE or bool(kept_ligands)
    ligand_mode = LigandMode.GAFF2
    ligand_charge_method = ChargeMethod.ANTECHAMBER
    manual_files: list[str] = []
    ligand_parameter_assignments: list[LigandParameterAssignment] = []
    ligand_residue_name = LigandsConfig().residue_name
    resp_job_dir: str | None = None
    resp_group_file: str | None = None
    resp_session_file: str | None = None
    resp_apply_mode = RespApplyMode.DETECT
    system_ligand_ff = "gaff2"
    if needs_nonstandard_molecules:
        ligand_mode, manual_files, system_ligand_ff, ligand_parameter_assignments = _prompt_nonstandard_molecule_config(
            amber_env,
            input_config=input_config,
            summary=inspection_summary,
            kept_ligands=kept_ligands,
            detected_resp_candidate=detected_resp_candidate,
        )
        if ligand_mode != LigandMode.MANUAL:
            ligand_targets = _nonstandard_parameter_targets(input_config, inspection_summary, kept_ligands)
            if ligand_parameter_assignments:
                ligand_residue_name = ligand_parameter_assignments[0].residue_name
            elif ligand_targets:
                ligand_residue_name = ligand_targets[0]
            if input_config.source == InputSource.SMALL_MOLECULE and detected_resp_candidate is not None:
                payload = getattr(detected_resp_candidate, "payload", {}) or {}
                ligand_charge_method = normalize_charge_method(
                    payload.get("charge_method") or ChargeMethod.RESP_ANTECHAMBER.value
                )
                ligand_residue_name = str(payload.get("residue_name") or ligand_residue_name).strip().upper()
                ligand_parameter_assignments = [
                    LigandParameterAssignment(
                        residue_name=ligand_residue_name,
                        net_charge=int(payload.get("net_charge") or 0),
                        multiplicity=int(payload.get("multiplicity") or 1),
                    )
                ]
                resp_apply_mode = RespApplyMode.APPLY_EXISTING
                resp_job_dir = str(detected_resp_candidate.job_dir)
                group_path = detected_resp_candidate.job_dir / "group_constraints.json"
                resp_group_file = str(group_path) if group_path.exists() else None
                popup_state = detected_resp_candidate.job_dir / "manifests" / "popup_state.json"
                resp_session_file = str(popup_state) if popup_state.exists() else None
                console.print(
                    "[bold cyan]RESP continue mode:[/bold cyan] "
                    "The existing RESP result will be reused, then Amber bonded parameters and setup files will be generated."
                )
            else:
                ligand_charge_method = _prompt_charge_method()
                if normalize_charge_method(ligand_charge_method) == ChargeMethod.ANTECHAMBER:
                    ligand_parameter_assignments = _prompt_ligand_parameter_assignments(ligand_targets)
                    if ligand_parameter_assignments:
                        ligand_residue_name = ligand_parameter_assignments[0].residue_name
            if input_config.source == InputSource.SMALL_MOLECULE and charge_method_uses_resp(ligand_charge_method):
                residue_name = ligand_residue_name
                net_charge, multiplicity = _lookup_assignment(ligand_parameter_assignments, residue_name)
                if detected_resp_candidate is None:
                    resp_apply_mode = RespApplyMode.NEW_DIRECTORY

                if resp_apply_mode != RespApplyMode.APPLY_EXISTING:
                    output_dir_path = _suggest_output_directory(input_config, resp_job_dir=resp_job_dir)
                    console.print(
                        f"[bold cyan]RESP output directory:[/bold cyan] {output_dir_path}\n"
                        "[dim]The RESP popup opens now, before any water/MD questions. "
                        "After the NWChem input and sbatch assets are generated, the wizard will stop so you can run the RESP job.[/dim]"
                    )
                    resp_session_file, resp_group_file = _build_resp_seed_files(
                        input_config=input_config,
                        residue_name=residue_name,
                        net_charge=net_charge,
                        multiplicity=multiplicity,
                        output_dir_path=output_dir_path,
                        charge_method=ligand_charge_method,
                    )
                    session_payload = json.loads(Path(resp_session_file).read_text(encoding="utf-8"))
                    popup_qm = session_payload.get("qm_settings") or {}
                    ligand_residue_name = str(session_payload.get("residue_name") or residue_name).strip().upper()
                    net_charge = int(popup_qm.get("net_charge") or 0)
                    multiplicity = int(popup_qm.get("multiplicity") or 1)
                    ligand_parameter_assignments = [
                        LigandParameterAssignment(
                            residue_name=ligand_residue_name,
                            net_charge=net_charge,
                            multiplicity=multiplicity,
                        )
                    ]
                    profile = SlurmProfile.CPU
                    config = WorkflowConfig(
                        input=input_config,
                        prepare=base_prepare_config,
                        protonation=ProtonationConfig(),
                        ligands=LigandsConfig(
                            mode=ligand_mode,
                            charge_method=ligand_charge_method,
                            manual_files=manual_files,
                            residue_name=ligand_residue_name,
                            net_charge=net_charge,
                            multiplicity=multiplicity,
                            parameter_assignments=ligand_parameter_assignments,
                            resp_job_dir=resp_job_dir,
                            resp_group_file=resp_group_file,
                            resp_session_file=resp_session_file,
                            resp_apply_mode=resp_apply_mode,
                        ),
                        system=SystemConfig(ligand_ff=system_ligand_ff),
                        md=MDConfig(),
                        slurm=SlurmConfig(
                            profile=profile,
                            partition=None,
                            account=None,
                            ntasks=8,
                            gpus=0,
                            walltime="24:00:00",
                            binary_override=None,
                            job_name=output_dir_path.name,
                        ),
                        output_dir=str(output_dir_path),
                    )
                    result = WizardBuildResult(configs=[config], variant_suffixes=[()])
                    result.saved_config_paths.extend(_save_wizard_configs(result, write_config))
                    if result.saved_config_paths:
                        console.print(f"Saved config to {result.saved_config_paths[0]}")
                    return result
    else:
        console.print("[dim]No non-standard molecules were selected for parameterization, so GAFF/GAFF2 setup is skipped.[/dim]")

    protein_ff = "ff19SB"
    if input_config.source != InputSource.SMALL_MOLECULE:
        protein_choices = _protein_force_field_choices(amber_env)
        _display_choice_table("Protein force field options", protein_choices)
        protein_ff = _prompt_choice("Choose the protein force field", protein_choices, default_key="ff19SB")
    else:
        console.print("[dim]Protein force-field selection is skipped for a small-molecule-only workflow.[/dim]")

    preview_prepare_config = variant_prepare_configs[0]
    remaining_metal_sites = _remaining_metal_sites(inspection_summary, preview_prepare_config)
    remaining_metal_site_count = len(remaining_metal_sites) + len(preview_prepare_config.metal_insertions)

    variant_metal_charge_assignments = [[] for _ in variant_prepare_configs]
    batch_charge_map: dict[tuple[int, str], int] = {}
    if remaining_metal_site_count > 0:
        batch_charge_map = _prompt_batch_metal_charge_assignments(inspection_summary, variant_prepare_configs)
        variant_metal_charge_assignments = [
            _variant_metal_charge_assignments(inspection_summary, prepare_config, batch_charge_map)
            for prepare_config in variant_prepare_configs
        ]
    else:
        console.print("[dim]No metal sites remain after Step 1, so metal oxidation-state selection is skipped.[/dim]")

    _print_step_header(
        3,
        "Review pH-Guided Protonation",
        "Run PROPKA after the surviving metal sites and oxidation states are confirmed, "
        "then review and choose which sidechain protonation-state changes to apply.",
    )
    protonation_variants = _prompt_protonation_variants(
        input_config=input_config,
        prepare_config=preview_prepare_config,
        summary=inspection_summary,
    )
    protonation_configs = [variant.protonation_config for variant in protonation_variants]
    protonation_preview_config = protonation_configs[0]

    ligand_net_charge, ligand_multiplicity = _lookup_assignment(
        ligand_parameter_assignments,
        ligand_residue_name,
    )
    ligands_config = LigandsConfig(
        mode=ligand_mode,
        charge_method=ligand_charge_method,
        manual_files=manual_files,
        residue_name=ligand_residue_name,
        net_charge=ligand_net_charge,
        multiplicity=ligand_multiplicity,
        parameter_assignments=ligand_parameter_assignments,
        resp_job_dir=resp_job_dir,
        resp_group_file=resp_group_file,
        resp_session_file=resp_session_file,
        resp_apply_mode=resp_apply_mode,
    )
    config_build_specs = _protein_config_build_specs(
        input_config=input_config,
        resp_job_dir=resp_job_dir,
        metal_action_plan=metal_action_plan,
        prepare_configs=variant_prepare_configs,
        metal_charge_assignments=variant_metal_charge_assignments,
        protonation_variants=protonation_variants,
    )

    protein_site_resp_config = ProteinSiteRespConfig()
    if input_config.source != InputSource.SMALL_MOLECULE:
        requested_site_resp = _prompt_new_protein_site_resp_config(
            remaining_metal_site_count=remaining_metal_site_count,
            batch_charge_map=batch_charge_map,
        )
        if requested_site_resp is not None:
            return _build_protein_site_resp_setup_result(
                input_config=input_config,
                protein_ff=protein_ff,
                system_ligand_ff=system_ligand_ff,
                ligands_config=ligands_config,
                resp_config=requested_site_resp,
                config_build_specs=config_build_specs,
                write_config=write_config,
            )

    metal_model = MetalModel.MODEL_1264
    c4_parameter_set = DESC4ParameterSet.OPC_DUVAIL
    if remaining_metal_site_count > 0:
        metal_choices = _metal_model_choices()
        _display_choice_table("Metal treatment model", metal_choices)
        metal_model = MetalModel(
            _prompt_choice(
                "Choose the metal treatment model",
                metal_choices,
                default_key=MetalModel.MODEL_1264.value,
            )
        )
        if metal_model == MetalModel.MODEL_1264:
            selected_metal_species = [
                (element, charge)
                for (_site, element), charge in batch_charge_map.items()
            ]
            parameter_choices = _metal_c4_parameter_set_choices(selected_metal_species)
            _display_choice_table("Metal 12-6-4 parameter set", parameter_choices)
            c4_parameter_set = DESC4ParameterSet(
                _prompt_choice(
                    "Choose the metal 12-6-4 parameter set",
                    parameter_choices,
                    default_key=DESC4ParameterSet.OPC_DUVAIL.value,
                )
            )

    _print_step_header(
        5,
        "Choose Solvation and Ion Settings",
        "Select the solvent model, box geometry, and ion conditions for the prepared system.",
    )

    pre_salt_monovalent_1264, pre_salt_multivalent_1264 = _aggregate_ion_parameter_requirements(
        SaltConfig(),
        variant_metal_charge_assignments,
        variant_prepare_configs,
    )
    if pre_salt_monovalent_1264 or pre_salt_multivalent_1264:
        required_sets: list[str] = []
        if pre_salt_monovalent_1264:
            required_sets.append("1+/anion")
        if pre_salt_multivalent_1264:
            required_sets.append("2+/3+/4+")
        console.print(
            "[dim]The water-model table below checks whether the required 12-6-4 ion set(s) are available "
            f"for the selected metal-site charges: {', '.join(required_sets)}.[/dim]"
        )

    water_model, custom_ion_frcmods = _prompt_water_model_selection(
        amber_env,
        include_monovalent=pre_salt_monovalent_1264,
        include_multivalent=pre_salt_multivalent_1264,
        c4_parameter_set=c4_parameter_set,
        default_water_model=(
            "opc" if c4_parameter_set == DESC4ParameterSet.OPC_DUVAIL else "spce"
        ),
    )

    box_choices = _box_shape_choices()
    _display_choice_table("Solvation box options", box_choices)
    box_shape = BoxShape(_prompt_choice("Choose the solvation box shape", box_choices, default_key=BoxShape.OCT.value))
    buffer_angstrom = _prompt_positive_float("Solvent buffer (Angstrom)", default=10.0)

    estimated_charge = _display_batch_charge_preview(
        input_config,
        variant_prepare_configs,
        protonation_configs,
        inspection_summary,
        variant_metal_charge_assignments,
    )
    salt_config = _prompt_salt_config(estimated_charge)
    include_monovalent_1264, include_multivalent_1264 = _aggregate_ion_parameter_requirements(
        salt_config,
        variant_metal_charge_assignments,
        variant_prepare_configs,
    )
    water_model, custom_ion_frcmods = _resolve_water_model_requirements_after_salt(
        amber_env,
        water_model=water_model,
        custom_ion_frcmods=custom_ion_frcmods,
        include_monovalent=include_monovalent_1264,
        include_multivalent=include_multivalent_1264,
        c4_parameter_set=c4_parameter_set,
    )
    uses_1264_files = (include_monovalent_1264 or include_multivalent_1264) and (
        not custom_ion_frcmods or any("1264" in Path(path).name.lower() for path in custom_ion_frcmods)
    )
    if uses_1264_files:
        console.print(
            "[bold yellow]12-6-4 note:[/bold yellow] After tleap, SIMPLE will generate a ParmEd "
            "`add12_6_4` step for the selected metal-site ions. In dry-run mode that helper script is written but not executed."
        )

    _print_step_header(
        6,
        "Choose the MD Protocol",
        "Pick the equilibration strategy and production settings for the molecular dynamics run.",
    )

    protocol = _prompt_md_protocol()
    temperature = typer.prompt("Target temperature (K)", default=300.0, type=float)
    pressure = typer.prompt("Target pressure (bar)", default=1.0, type=float)
    production_time = _prompt_positive_float("Production time (ns)", default=100.0)
    restraint_mask, restraint_weight = _prompt_custom_restraints(
        input_config,
        inspection_summary,
        preview_prepare_config,
        protonation_preview_config,
        remaining_metal_site_count,
    )

    _print_step_header(
        7,
        "Choose the Execution Script and Output Location",
        "Select whether to generate a CPU or GPU Slurm script. "
        "Cluster-specific resource lines can be edited directly in the generated sbatch file.",
    )

    profile = _prompt_execution_profile()
    output_dir_paths = [item[3] for item in config_build_specs]

    if len(output_dir_paths) == 1:
        console.print(
            f"[bold cyan]Output directory:[/bold cyan] {output_dir_paths[0]}\n"
            "[dim]If that name already exists, an incremented suffix is applied automatically.[/dim]"
        )
    else:
        table = Table(title="Planned output directories", box=box.SIMPLE_HEAVY)
        table.add_column("No.", style="bold cyan", justify="right", no_wrap=True)
        table.add_column("Output directory", style="bold white")
        for index, output_dir_path in enumerate(output_dir_paths, start=1):
            table.add_row(str(index), str(output_dir_path))
        console.print(table)
        console.print(
            "[dim]If any of those names already exist, an incremented suffix is applied automatically before the run starts.[/dim]"
        )
    console.print(
        "[blink bold cyan]Processing...[/] Interactive setup is complete. "
        f"Structure preparation and workflow file generation will begin now for {len(output_dir_paths)} output(s)."
    )

    md_config = MDConfig(
        protocol=protocol,
        temperature_k=temperature,
        pressure_bar=pressure,
        production_time_ns=production_time,
        focused_restraint_mask=restraint_mask or None,
        focused_restraint_mask_numbering=(
            ResidueMaskNumbering.PREPARED if restraint_mask else ResidueMaskNumbering.PDB
        ),
        focused_restraint_weight=restraint_weight,
    )

    configs: list[WorkflowConfig] = []
    variant_suffixes: list[tuple[str, ...]] = []
    base_job_name = _base_output_name(input_config)
    for (
        prepare_config,
        metal_charge_assignments,
        protonation_variant,
        output_dir_path,
        combined_suffix_tokens,
    ) in config_build_specs:
        job_name_suffix_tokens = combined_suffix_tokens
        if (
            protonation_variant.protonation_config.enabled
            and protonation_variant.ph_token
            and protonation_variant.ph_token not in job_name_suffix_tokens
        ):
            job_name_suffix_tokens = combined_suffix_tokens + (protonation_variant.ph_token,)
        job_name = _workflow_job_name(
            base_job_name,
            job_name_suffix_tokens,
        )
        configs.append(
            WorkflowConfig(
                input=input_config,
                prepare=prepare_config,
                protonation=protonation_variant.protonation_config.model_copy(deep=True),
                protein_site_resp=protein_site_resp_config.model_copy(deep=True),
                ligands=ligands_config.model_copy(deep=True),
                system=SystemConfig(
                    protein_ff=protein_ff,
                    ligand_ff=system_ligand_ff,
                    metal_model=metal_model,
                    c4_parameter_set=c4_parameter_set,
                    metal_charges=metal_charge_assignments,
                    water_model=water_model,
                    box_shape=box_shape,
                    buffer_angstrom=buffer_angstrom,
                    salt=salt_config,
                    custom_ion_frcmods=custom_ion_frcmods,
                ),
                md=md_config.model_copy(deep=True),
                slurm=SlurmConfig(
                    profile=profile,
                    partition=None,
                    account=None,
                    ntasks=8,
                    gpus=1 if profile == SlurmProfile.GPU else 0,
                    walltime="24:00:00",
                    binary_override=None,
                    job_name=job_name,
                ),
                output_dir=str(output_dir_path),
            )
        )
        variant_suffixes.append(combined_suffix_tokens)

    result = WizardBuildResult(
        configs=configs,
        variant_suffixes=variant_suffixes,
    )
    result.saved_config_paths.extend(_save_wizard_configs(result, write_config))
    if result.saved_config_paths:
        if len(result.saved_config_paths) == 1:
            console.print(f"Saved config to {result.saved_config_paths[0]}")
        else:
            console.print("[bold cyan]Saved batch configs:[/bold cyan]")
            for saved_path in result.saved_config_paths:
                console.print(f"  - {saved_path}")
    return result


def build_wizard_configs(write_config: str | None) -> WizardBuildResult:
    while True:
        try:
            return _build_wizard_configs_once(write_config)
        except WizardBack:
            console.print("[dim]Back requested. Restarting the interactive wizard from the previous top-level choice.[/dim]")


def build_wizard_config(write_config: str | None) -> WorkflowConfig:
    return build_wizard_configs(write_config).configs[0]


def _prompt_tool_mode(label: str, choices: tuple[str, ...], default: str) -> str:
    options = "/".join(choices)
    while True:
        value = typer.prompt(f"{label} [{options}]", default=default).strip().lower()
        if value in choices:
            return value
        console.print(f"[red]Choose one of: {', '.join(choices)}.[/red]")


def _prompt_required_value(label: str, default: str = "") -> str:
    while True:
        value = typer.prompt(label, default=default).strip()
        if value:
            return value
        console.print(f"[red]{label} is required for this selection.[/red]")


@app.command()
def configure(
    config_path: str | None = typer.Option(None, "--config", help="Override the per-user tools.toml path"),
) -> None:
    """Configure AmberTools, licensed AMBER, NWChem, and other executables."""
    target = Path(config_path).expanduser() if config_path else default_tool_config_path()
    current = load_tool_config(target)
    console.print(
        Panel.fit(
            "AmberTools supports system setup, analysis, and the GUI. "
            "Licensed AMBER must be configured for production MD and TI/free-energy simulation execution.\n\n"
            "Tahoma users should use Conda AmberTools for local preparation. The generated Tahoma sbatch files "
            "keep their site-specific AMBER setup and do not use the local licensed-AMBER path below.",
            title="SIMPLE software configuration",
            border_style="cyan",
        )
    )

    ambertools_default = current.ambertools.mode if current.ambertools.mode != "disabled" else "conda"
    ambertools_mode = _prompt_tool_mode(
        "AmberTools source",
        ("conda", "external", "disabled"),
        ambertools_default,
    )
    ambertools_home = current.ambertools.home
    if ambertools_mode == "conda":
        ambertools_home = discover_ambertools_home()
    elif ambertools_mode == "external":
        ambertools_home = _prompt_required_value(
            "AmberTools home directory",
            default=ambertools_home or discover_ambertools_home(),
        )
    else:
        ambertools_home = ""

    amber_mode = _prompt_tool_mode(
        "Licensed AMBER source",
        ("external", "module", "disabled"),
        current.amber.mode,
    )
    amber_home = current.amber.home
    amber_module = current.amber.module_name
    setup_script = current.amber.setup_script
    if amber_mode == "external":
        amber_home = _prompt_required_value(
            "Licensed AMBER home directory (AMBERHOME)",
            default=amber_home or "",
        )
        setup_default = setup_script or (str(Path(amber_home).expanduser() / "amber.sh") if amber_home else "")
        setup_script = _prompt_required_value("AMBER activation script", default=setup_default)
        amber_module = ""
    elif amber_mode == "module":
        amber_module = _prompt_required_value("AMBER module name", default=amber_module or "amber")
        amber_home = ""
        setup_script = ""
    else:
        amber_home = ""
        amber_module = ""
        setup_script = ""
        console.print(
            "[bold red]Licensed AMBER is not configured. Generic production MD and TI/free-energy sbatch "
            "files will stop with a configuration error.[/bold red]"
        )

    nwchem_mode = _prompt_tool_mode(
        "NWChem source",
        ("conda", "external", "module", "disabled"),
        current.nwchem.mode,
    )
    nwchem_binary = current.nwchem.binary
    mpi_launcher = current.nwchem.mpi_launcher
    nwchem_module = current.nwchem.module_name
    if nwchem_mode == "conda":
        nwchem_binary = discover_binary("nwchem")
        mpi_launcher = discover_binary("mpirun") or discover_binary("mpiexec")
        nwchem_module = ""
    elif nwchem_mode == "external":
        nwchem_binary = _prompt_required_value("Absolute NWChem executable path", default=nwchem_binary or "")
        mpi_launcher = _prompt_required_value("Matching MPI launcher path", default=mpi_launcher or "")
        nwchem_module = ""
    elif nwchem_mode == "module":
        nwchem_module = _prompt_required_value("NWChem module name", default=nwchem_module or "nwchem")
        nwchem_binary = "nwchem"
        mpi_launcher = ""
    else:
        nwchem_binary = ""
        mpi_launcher = ""
        nwchem_module = ""

    configured = ToolConfig(
        ambertools=AmberToolsSettings(mode=ambertools_mode, home=ambertools_home),
        amber=AmberSettings(
            mode=amber_mode,
            home=amber_home,
            activation=current.amber.activation,
            setup_script=setup_script,
            module_name=amber_module,
            serial=current.amber.serial,
            mpi=current.amber.mpi,
            gpu=current.amber.gpu,
            gpu_mpi=current.amber.gpu_mpi,
        ),
        nwchem=NWChemSettings(
            mode=nwchem_mode,
            binary=nwchem_binary,
            mpi_launcher=mpi_launcher,
            module_name=nwchem_module,
        ),
        executables=current.executables,
    )
    saved = save_tool_config(configured, target)
    console.print(f"[bold green]Saved software configuration:[/bold green] {saved}")
    console.print(
        "[cyan]Paths may be edited later in this TOML file. Editing a mode to 'conda' does not install a "
        "package; rerun install_simple.py or install the selected package in the SIMPLE environment.[/cyan]"
    )


@app.command()
def doctor() -> None:
    """Report AMBERHOME, binaries, and available force-field assets."""
    summary = environment_summary()
    emit_key_value_table(
        "Environment",
        [
            (key, str(value))
            for key, value in summary.items()
            if key not in {"binaries", "leaprc_files", "ion_frcmods"}
        ],
    )
    binaries = Table(title="Binaries")
    binaries.add_column("Binary")
    binaries.add_column("Path")
    for name, path in summary["binaries"].items():
        binaries.add_row(name, str(path))
    console.print(binaries)

    leap_table = Table(title="Available leaprc files")
    leap_table.add_column("leaprc")
    for item in summary["leaprc_files"]:
        leap_table.add_row(item)
    console.print(leap_table)

    emit_key_value_table("Configured software", list(tool_config_summary().items()))


@app.command()
def inspect(input: str = typer.Option(..., "--input", help="PDB path or PDB ID")) -> None:
    """Inspect structure composition."""
    if Path(input).exists():
        summary = inspect_structure(input)
    elif looks_like_pdb_id(input):
        fetched = fetch_pdb_structure(input.upper(), Path(".simple_inspect"))
        summary = inspect_structure(fetched, source_label="pdb_id")
    else:
        raise typer.BadParameter("Input must be an existing file path or a 4-character PDB ID.")
    _display_summary(summary)


@app.command()
def wizard(
    write_config: str | None = typer.Option(None, "--write-config", help="Write TOML config"),
) -> None:
    """Interactive workflow builder."""
    result = build_wizard_configs(write_config)
    execute_now = typer.confirm("Execute workflow now?", default=False)
    if not execute_now:
        return
    dry_run = typer.confirm("Use dry-run mode?", default=not is_linux_execution_host())
    raise typer.Exit(code=execute_wizard_configs(result.configs, dry_run=dry_run))


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path to TOML config"),
    from_stage: str = typer.Option("prepare", "--from", help="Start stage"),
    to_stage: str = typer.Option("md", "--to", help="End stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not execute Amber binaries"),
) -> None:
    """Run the workflow from TOML."""
    loaded = load_config(config)
    result = run_workflow(config=loaded, from_stage=from_stage, to_stage=to_stage, dry_run=dry_run)
    print_workflow_summary(result)
