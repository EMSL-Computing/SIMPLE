from __future__ import annotations

import re
from pathlib import Path

import typer
from rich import box
from rich.table import Table

from amber_metallo.cli import WizardChoice, _display_choice_table, _prompt_choice, _print_step_header
from amber_metallo.reporting import console, print_notice
from amber_metallo.subdirectory_search import search_subdirectories_enabled
from amber_metallo.ti.abfe import (
    CASE_TYPE_BOUND,
    CASE_TYPE_WATER,
    SAMPLING_SELECTION_FORWARD_ONLY,
    SAMPLING_SELECTION_FORWARD_REVERSE,
    AnalysisCaseDiscovery,
    RBFEAnalysisResult,
    SingleCaseAnalysisResult,
    analyze_rbfe,
    analyze_single_case,
    case_has_bidirectional_sampling,
    decoupling_scheme_for_case,
    discover_analysis_cases,
    discover_water_library_cases,
    inspect_analysis_case,
    rbfe_pair_compatibility,
)


def _analysis_mode_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            "abfe",
            "ABFE calculation",
            "Analyze one or more completed TI cases and report standalone dG values.",
        ),
        WizardChoice(
            "rbfe",
            "RBFE calculation",
            "Select one or more completed bound cases and choose a water-reference case for each, then compute ddG.",
        ),
        WizardChoice(
            "extra",
            "Additional analyses",
            "Run RMSD, RMSF, radius of gyration, RDF, and distance analyses on one or more trajectories.",
        ),
    ]


def _analysis_method_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            "ti",
            "TI",
            "Use thermodynamic integration from the existing DV/DL output in the current Amber mdout files.",
        ),
        WizardChoice(
            "bar",
            "BAR",
            "Coming soon. Current TI runs were not generated with the extra overlap output needed for BAR postprocessing.",
            enabled=False,
        ),
        WizardChoice(
            "mbar",
            "MBAR",
            "Coming soon. Current TI runs were not generated with the extra overlap output needed for MBAR postprocessing.",
            enabled=False,
        ),
        WizardChoice(
            "all",
            "All",
            "Coming soon. This will compare every supported postprocessing method side by side once the needed outputs exist.",
            enabled=False,
        ),
    ]


def _sampling_selection_choices() -> list[WizardChoice]:
    return [
        WizardChoice(
            SAMPLING_SELECTION_FORWARD_ONLY,
            "Forward only",
            "Default. Use the conventional forward TI estimate and ignore reverse windows when they exist.",
        ),
        WizardChoice(
            SAMPLING_SELECTION_FORWARD_REVERSE,
            "Forward + Reverse",
            "Optional convergence diagnostic. Average both sweeps only when their hysteresis is acceptably small; a large difference signals inadequate equilibration or path dependence.",
        ),
    ]


def _prompt_sampling_selection(cases: list[AnalysisCaseDiscovery]) -> str:
    if not any(case_has_bidirectional_sampling(case) for case in cases):
        return SAMPLING_SELECTION_FORWARD_ONLY
    choices = _sampling_selection_choices()
    _display_choice_table("Bidirectional TI data to include", choices)
    return _prompt_choice(
        "Choose which TI directions to analyze",
        choices,
        default_key=SAMPLING_SELECTION_FORWARD_ONLY,
    )


def _case_sampling_label(case: AnalysisCaseDiscovery) -> str:
    library_selection = str(case.metadata.get("library_sampling_selection") or "")
    if library_selection == "legacy_unspecified":
        return "Legacy (Forward-compatible)"
    if library_selection == SAMPLING_SELECTION_FORWARD_ONLY:
        return "Forward"
    if library_selection == SAMPLING_SELECTION_FORWARD_REVERSE:
        return "Forward + Reverse"
    return "Forward + Reverse" if case_has_bidirectional_sampling(case) else "Forward"


def _display_cases(
    search_dir: Path,
    discoveries: list[AnalysisCaseDiscovery],
    *,
    title: str,
    allow_multi: bool = False,
) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("No.", style="bold cyan", justify="right")
    table.add_column("Case", style="bold white")
    table.add_column("Type", style="white")
    table.add_column("Scheme", style="magenta")
    table.add_column("Sampling", style="green")
    table.add_column("Charge mode", style="yellow")
    table.add_column("Description", style="cyan")
    table.add_column("Outputs", style="white")
    table.add_column("Status", style="white")
    if allow_multi:
        table.add_row(
            "0",
            "All ready cases",
            "-",
            "-",
            "-",
            "-",
            "Analyze every selectable case in this list.",
            "-",
            "Ready cases only",
        )
        table.add_row(
            "M",
            "Enter a path manually",
            "-",
            "-",
            "-",
            "-",
            "Use a case outside this list.",
            "-",
            "Always available",
        )
    else:
        table.add_row(
            "0",
            "Enter a path manually",
            "-",
            "-",
            "-",
            "-",
            "Use a case outside this list.",
            "-",
            "Always available",
        )
    for index, discovery in enumerate(discoveries, start=1):
        type_label = "Water ref" if discovery.case_type == CASE_TYPE_WATER else "Bound"
        if discovery.source_kind != "simulation":
            type_label = f"{type_label} ({discovery.source_kind})"
        table.add_row(
            str(index),
            f"{discovery.display_name}\n{discovery.root}",
            type_label,
            decoupling_scheme_for_case(discovery),
            _case_sampling_label(discovery),
            str(discovery.metadata.get("charge_compensation_mode") or "unknown"),
            discovery.description,
            discovery.completion_summary,
            discovery.readiness_note,
        )
    console.print(table)
    scope = "and its immediate subdirectories" if search_subdirectories_enabled() else "only"
    console.print(
        f"[dim]Scanned {search_dir.resolve()} {scope} for completed water_ref and bound TI cases.[/dim]"
    )


def _parse_case_indices(raw: str, *, max_index: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if start > end:
                raise ValueError("ranges must be ascending")
            candidates = range(start, end + 1)
        else:
            candidates = [int(token)]
        for candidate in candidates:
            if candidate < 1 or candidate > max_index:
                raise ValueError("index out of range")
            if candidate not in seen:
                seen.add(candidate)
                indices.append(candidate)
    if not indices:
        raise ValueError("no case numbers selected")
    return indices


def _prompt_discovered_cases(
    discoveries: list[AnalysisCaseDiscovery],
    *,
    prompt_text: str,
    allow_multi: bool = False,
    preferred: AnalysisCaseDiscovery | None = None,
) -> list[AnalysisCaseDiscovery] | None:
    selectable_indices = [index for index, item in enumerate(discoveries, start=1) if item.selectable]
    preferred_index = next(
        (
            index
            for index, item in enumerate(discoveries, start=1)
            if preferred is not None
            and item.root == preferred.root
            and item.case_type == preferred.case_type
            and item.library_key == preferred.library_key
            and item.selectable
        ),
        None,
    )
    default_choice = str(preferred_index or (selectable_indices[0] if selectable_indices else "0"))
    while True:
        raw = typer.prompt(prompt_text, default=default_choice).strip()
        if allow_multi and raw.lower() in {"m", "manual"}:
            return None
        if raw == "0":
            if allow_multi:
                selected_cases = [item for item in discoveries if item.selectable]
                if selected_cases:
                    return selected_cases
                console.print("[bold yellow]No ready cases are available to analyze from this list.[/bold yellow]")
                continue
            return None
        if allow_multi:
            try:
                selected_indices = _parse_case_indices(raw, max_index=len(discoveries))
            except ValueError:
                console.print("[bold red]Please enter 0, M, or listed case numbers such as 1,3-5.[/bold red]")
                continue
        else:
            try:
                selected_indices = [int(raw)]
            except ValueError:
                console.print("[bold red]Please enter 0 or one of the listed case numbers.[/bold red]")
                continue
            if selected_indices[0] < 1 or selected_indices[0] > len(discoveries):
                console.print("[bold red]Please choose a number from the table.[/bold red]")
                continue
        selected_cases = [discoveries[index - 1] for index in selected_indices]
        unavailable = [item for item in selected_cases if not item.selectable]
        if unavailable:
            for discovery in unavailable:
                console.print(f"[bold yellow]{discovery.display_name}: {discovery.readiness_note}[/bold yellow]")
            continue
        return selected_cases


def _prompt_manual_case_path(*, case_type: str | None = None) -> AnalysisCaseDiscovery:
    label = "completed TI case folder"
    if case_type == CASE_TYPE_BOUND:
        label = "completed bound TI case folder"
    elif case_type == CASE_TYPE_WATER:
        label = "completed water-reference TI case folder"

    while True:
        raw = typer.prompt(f"Path to a {label}").strip()
        candidate = Path(raw).expanduser()
        if not candidate.exists():
            console.print(f"[bold red]Path not found:[/bold red] {raw}")
            continue
        discovery = inspect_analysis_case(candidate, case_type=case_type)
        if discovery is None:
            console.print("[bold red]That path does not look like a valid TI analysis case.[/bold red]")
            continue
        if not discovery.selectable:
            console.print(f"[bold yellow]{discovery.readiness_note}[/bold yellow]")
            continue
        return discovery


def _prompt_case_selection(
    *,
    case_types: set[str],
    title: str,
    prompt_text: str,
    include_library_water: bool = False,
    preferred_for_bound: AnalysisCaseDiscovery | None = None,
    candidate_cases: list[AnalysisCaseDiscovery] | None = None,
) -> AnalysisCaseDiscovery:
    discoveries = (
        list(candidate_cases)
        if candidate_cases is not None
        else discover_analysis_cases(Path.cwd(), case_types=case_types)
    )
    if candidate_cases is None and include_library_water and CASE_TYPE_WATER in case_types:
        discoveries.extend(discover_water_library_cases())
    if discoveries:
        _display_cases(Path.cwd(), discoveries, title=title)
        preferred = (
            _preferred_water_case(preferred_for_bound, discoveries)
            if preferred_for_bound is not None and CASE_TYPE_WATER in case_types
            else None
        )
        if preferred is not None:
            console.print(
                f"[dim]Default water-reference match for {preferred_for_bound.display_name}: "
                f"{preferred.display_name} ({preferred.description}).[/dim]"
            )
        selected = _prompt_discovered_cases(discoveries, prompt_text=prompt_text, preferred=preferred)
        if selected is not None:
            return selected[0]
    else:
        console.print("[dim]No matching completed TI cases were detected here, so the launcher will switch to manual path mode.[/dim]")
    if case_types == {CASE_TYPE_BOUND}:
        return _prompt_manual_case_path(case_type=CASE_TYPE_BOUND)
    if case_types == {CASE_TYPE_WATER}:
        return _prompt_manual_case_path(case_type=CASE_TYPE_WATER)
    return _prompt_manual_case_path()


def _prompt_case_selections(
    *,
    case_types: set[str],
    title: str,
    prompt_text: str,
) -> list[AnalysisCaseDiscovery]:
    discoveries = discover_analysis_cases(Path.cwd(), case_types=case_types)
    if discoveries:
        _display_cases(Path.cwd(), discoveries, title=title, allow_multi=True)
        selected = _prompt_discovered_cases(discoveries, prompt_text=prompt_text, allow_multi=True)
        if selected is not None:
            return selected
    else:
        console.print("[dim]No matching completed TI cases were detected here, so the launcher will switch to manual path mode.[/dim]")
    if case_types == {CASE_TYPE_BOUND}:
        return [_prompt_manual_case_path(case_type=CASE_TYPE_BOUND)]
    if case_types == {CASE_TYPE_WATER}:
        return [_prompt_manual_case_path(case_type=CASE_TYPE_WATER)]
    return [_prompt_manual_case_path()]


def _case_metal_signatures(case: AnalysisCaseDiscovery) -> set[tuple[str, int | None]]:
    signatures: set[tuple[str, int | None]] = set()
    if case.case_type == CASE_TYPE_WATER:
        for item in case.metadata.get("metals") or []:
            if not isinstance(item, dict) or not item.get("element"):
                continue
            charge = item.get("formal_charge")
            signatures.add((str(item["element"]).title(), None if charge is None else int(charge)))
        metal = str(case.metadata.get("metal") or "").strip()
        if metal:
            charge = case.metadata.get("formal_charge")
            signatures.add((metal.title(), None if charge is None else int(charge)))
        return signatures

    charge_by_site = case.metadata.get("selected_formal_charges_by_site") or {}
    for item in case.metadata.get("selected_sites") or []:
        if not isinstance(item, dict) or not item.get("element"):
            continue
        site = item.get("site")
        charge = item.get("formal_charge")
        if charge is None and site is not None:
            charge = charge_by_site.get(str(site), charge_by_site.get(site))
        signatures.add((str(item["element"]).title(), None if charge is None else int(charge)))
    batch_element = str(case.metadata.get("batch_element") or "").strip()
    if batch_element:
        charge = case.metadata.get("selected_formal_charge")
        signatures.add((batch_element.title(), None if charge is None else int(charge)))
    if signatures:
        return signatures

    description = str(case.metadata.get("selected_metal") or case.description)
    supported = (
        "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
        "Tm", "Yb", "Lu", "Mn", "Fe", "Co", "Ni", "Cu",
    )
    for element in supported:
        if re.search(rf"(?<![A-Za-z]){element}(?![a-z])", description, flags=re.IGNORECASE):
            charge = case.metadata.get("selected_formal_charge")
            signatures.add((element, None if charge is None else int(charge)))
    return signatures


def _preferred_water_case(
    bound_case: AnalysisCaseDiscovery,
    water_cases: list[AnalysisCaseDiscovery],
) -> AnalysisCaseDiscovery | None:
    bound_signatures = _case_metal_signatures(bound_case)
    selectable = [item for item in water_cases if item.selectable]
    if not selectable:
        return None

    def score(water_case: AnalysisCaseDiscovery) -> tuple[int, int, int, int, int]:
        water_signatures = _case_metal_signatures(water_case)
        exact = bool(bound_signatures & water_signatures)
        same_element = any(
            bound_element == water_element
            for bound_element, _bound_charge in bound_signatures
            for water_element, _water_charge in water_signatures
        )
        bound_charge_mode = str(bound_case.metadata.get("charge_compensation_mode") or "unknown")
        water_charge_mode = str(water_case.metadata.get("charge_compensation_mode") or "unknown")
        charge_match = 2 if bound_charge_mode == water_charge_mode else 1 if "unknown" in {bound_charge_mode, water_charge_mode} else 0
        bound_scheme = decoupling_scheme_for_case(bound_case)
        water_scheme = decoupling_scheme_for_case(water_case)
        scheme_match = 2 if bound_scheme == water_scheme else 1 if "Unknown" in {bound_scheme, water_scheme} else 0
        metal_match = 2 if exact else 1 if same_element else 0
        sampling_metadata = str(water_case.metadata.get("library_sampling_selection") or "")
        sampling_specificity = 0 if sampling_metadata == "legacy_unspecified" else 1
        return (charge_match, scheme_match, metal_match, sampling_specificity, -selectable.index(water_case))

    preferred = max(selectable, key=score)
    return preferred if score(preferred)[2] > 0 else None


def run_analysis_wizard() -> (
    SingleCaseAnalysisResult
    | RBFEAnalysisResult
    | list[SingleCaseAnalysisResult]
    | list[RBFEAnalysisResult]
):
    _print_step_header(
        1,
        "Choose the Analysis Type",
        "ABFE analyzes one or more standalone cases. RBFE accepts one or more bound cases and pairs each with a separately confirmed water-reference case.",
    )
    mode_choices = _analysis_mode_choices()
    _display_choice_table("Analysis menu", mode_choices)
    mode = _prompt_choice("Choose the analysis type", mode_choices, default_key="abfe")

    if mode == "extra":
        from amber_metallo.trajectory_analysis_cli import run_trajectory_analysis_wizard

        return run_trajectory_analysis_wizard()

    _print_step_header(
        2,
        "Choose the Postprocessing Method",
        "Only TI is currently available because the existing runs provide DV/DL output but not the extra overlap information needed for BAR or MBAR.",
    )
    method_choices = _analysis_method_choices()
    _display_choice_table("Postprocessing methods", method_choices)
    method = _prompt_choice("Choose the postprocessing method", method_choices, default_key="ti")
    if method != "ti":
        raise typer.Abort()
    print_notice(
        "Selected Method",
        "SIMPLE will integrate the existing DV/DL values with the TI trapezoidal rule. BAR, MBAR, and All are listed for future support only.",
        border_style="cyan",
    )

    if mode == "abfe":
        _print_step_header(
            3,
            "Select TI Case(s)",
            "Choose completed cases from the mixed list. Water-reference and bound cases are both allowed, and each selected case will be analyzed on its own without pairing.",
        )
        case_selections = _prompt_case_selections(
            case_types={CASE_TYPE_WATER, CASE_TYPE_BOUND},
            title="Detected TI Cases for ABFE",
            prompt_text="Choose case number(s) (0 = analyze all ready, M = enter a path manually)",
        )
        sampling_selection = _prompt_sampling_selection(case_selections)
        results = [
            analyze_single_case(case_selection, sampling_selection=sampling_selection)
            for case_selection in case_selections
        ]
        return results[0] if len(results) == 1 else results

    if mode == "rbfe":
        _print_step_header(
            3,
            "Select Bound Case(s)",
            "Choose one or more completed bound cases. RBFE will use each TI total plus its stored restraint correction.",
        )
        bound_cases = _prompt_case_selections(
            case_types={CASE_TYPE_BOUND},
            title="Detected Bound TI Cases",
            prompt_text="Choose bound case number(s) (0 = analyze all ready, M = enter a path manually)",
        )
        sampling_selection = _prompt_sampling_selection(bound_cases)
        water_candidates = discover_analysis_cases(Path.cwd(), case_types={CASE_TYPE_WATER})
        water_candidates.extend(
            discover_water_library_cases(sampling_selection=sampling_selection)
        )
        pairings: list[tuple[AnalysisCaseDiscovery, AnalysisCaseDiscovery]] = []
        for index, bound_case in enumerate(bound_cases, start=1):
            _print_step_header(
                3 + index,
                f"Select Water Reference for {bound_case.display_name}",
                "Choose the completed water-reference case for this bound case. A matching REE identity is "
                "selected as the default when available.",
            )
            while True:
                water_case = _prompt_case_selection(
                    case_types={CASE_TYPE_WATER},
                    title=f"Water References for {bound_case.display_name}",
                    prompt_text="Choose a water-reference case number (0 = enter a path manually)",
                    preferred_for_bound=bound_case,
                    candidate_cases=water_candidates,
                )
                compatibility_errors, compatibility_warnings = rbfe_pair_compatibility(bound_case, water_case)
                if compatibility_errors:
                    console.print(
                        "[bold red]That bound/water pair is incompatible:[/bold red] "
                        + " ".join(compatibility_errors)
                    )
                    continue
                for warning in compatibility_warnings:
                    console.print(f"[bold yellow]Pairing warning:[/bold yellow] {warning}")
                pairings.append((bound_case, water_case))
                break
        print_notice(
            "RBFE Pairings Confirmed",
            "All bound/water selections are complete. SIMPLE will now run the calculations without pausing for more case choices.",
            border_style="cyan",
        )
        results = [
            analyze_rbfe(bound_case, water_case, sampling_selection=sampling_selection)
            for bound_case, water_case in pairings
        ]
        return results[0] if len(results) == 1 else results

    raise typer.Abort()
