from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.table import Table

from amber_metallo.cli import WizardChoice, _display_choice_table, _prompt_choice, _print_step_header
from amber_metallo.reporting import console, print_notice
from amber_metallo.ti.abfe import (
    CASE_TYPE_BOUND,
    CASE_TYPE_WATER,
    AnalysisCaseDiscovery,
    RBFEAnalysisResult,
    SingleCaseAnalysisResult,
    analyze_rbfe,
    analyze_single_case,
    decoupling_scheme_for_case,
    discover_analysis_cases,
    discover_water_library_cases,
    inspect_analysis_case,
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
            "Select one completed bound case and one completed water-reference case, then compute ddG.",
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
    table.add_column("Description", style="cyan")
    table.add_column("Outputs", style="white")
    table.add_column("Status", style="white")
    if allow_multi:
        table.add_row(
            "0",
            "All ready cases",
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
            discovery.description,
            discovery.completion_summary,
            discovery.readiness_note,
        )
    console.print(table)
    console.print(
        f"[dim]Scanned {search_dir.resolve()} and its immediate subdirectories for completed water_ref and bound TI cases.[/dim]"
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
) -> list[AnalysisCaseDiscovery] | None:
    selectable_indices = [index for index, item in enumerate(discoveries, start=1) if item.selectable]
    default_choice = str(selectable_indices[0]) if selectable_indices else "0"
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
) -> AnalysisCaseDiscovery:
    discoveries = discover_analysis_cases(Path.cwd(), case_types=case_types)
    if include_library_water and CASE_TYPE_WATER in case_types:
        discoveries.extend(discover_water_library_cases())
    if discoveries:
        _display_cases(Path.cwd(), discoveries, title=title)
        selected = _prompt_discovered_cases(discoveries, prompt_text=prompt_text)
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
    return [_prompt_manual_case_path()]


def run_analysis_wizard() -> SingleCaseAnalysisResult | RBFEAnalysisResult | list[SingleCaseAnalysisResult]:
    _print_step_header(
        1,
        "Choose the Analysis Type",
        "ABFE now means single-case dG analysis, while RBFE pairs one bound case with one water-reference case to compute ddG.",
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
        results = [analyze_single_case(case_selection) for case_selection in case_selections]
        return results[0] if len(results) == 1 else results

    if mode == "rbfe":
        _print_step_header(
            3,
            "Select the Bound Case",
            "Choose one completed bound case. RBFE will use its TI total plus the stored restraint correction.",
        )
        bound_case = _prompt_case_selection(
            case_types={CASE_TYPE_BOUND},
            title="Detected Bound TI Cases",
            prompt_text="Choose a bound case number (0 = enter a path manually)",
        )
        _print_step_header(
            4,
            "Select the Water-Reference Case",
            "Choose one completed water-reference case. RBFE will compute ddG = (dG_bound_ti + restraint_correction) - dG_water.",
        )
        water_case = _prompt_case_selection(
            case_types={CASE_TYPE_WATER},
            title="Detected Water-Reference TI Cases",
            prompt_text="Choose a water-reference case number (0 = enter a path manually)",
            include_library_water=True,
        )
        return analyze_rbfe(bound_case, water_case)

    raise typer.Abort()
