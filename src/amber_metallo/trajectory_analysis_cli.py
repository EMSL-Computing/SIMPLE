from __future__ import annotations

from pathlib import Path
from typing import Sequence

import typer
from rich import box
from rich.table import Table

from amber_metallo.cli import WizardChoice, _display_choice_table, _print_step_header
from amber_metallo.reporting import console, print_notice
from amber_metallo.trajectory_analysis import (
    MaskOption,
    SelectionValidationError,
    SimulationDiscovery,
    TrajectoryFrameSelection,
    TrajectoryAnalysisRequest,
    TrajectoryAnalysisRunResult,
    TrajectoryCase,
    candidate_masks_for_universe,
    default_output_root,
    dependency_install_hint,
    describe_frame_selection,
    discover_simulation_cases,
    generalize_mask_options,
    load_universe,
    run_trajectory_analyses,
    validate_selection_across_cases,
)


def _analysis_choices() -> list[WizardChoice]:
    return [
        WizardChoice("rmsd", "RMSD", "Frame-by-frame structural deviation with optional alignment mask."),
        WizardChoice("rmsf", "RMSF", "Per-atom fluctuation after alignment."),
        WizardChoice("rg", "Radius of gyration", "Frame-by-frame compactness for a selected atom group."),
        WizardChoice("rdf", "RDF", "Radial distribution function and cumulative coordination-style profile."),
        WizardChoice("distance", "Distance", "Frame-by-frame distance between two atom selections or group centers."),
    ]


def _parse_indices(raw: str, *, max_index: int) -> list[int]:
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
        raise ValueError("no selections were entered")
    return indices


def _display_discoveries(discoveries: Sequence[SimulationDiscovery]) -> None:
    table = Table(title="Detected Trajectory Cases", box=box.SIMPLE_HEAVY)
    table.add_column("No.", justify="right", style="bold cyan")
    table.add_column("Case", style="bold white")
    table.add_column("Topology", style="white")
    table.add_column("Trajectory", style="cyan")
    table.add_row("0", "Enter files manually", "-", "-")
    table.add_row("A", "All detected cases", "-", "-")
    for index, discovery in enumerate(discoveries, start=1):
        table.add_row(
            str(index),
            f"{discovery.display_name}\n{discovery.root}",
            str(discovery.topology_path),
            str(discovery.trajectory_path),
        )
    console.print(table)


def _prompt_existing_file(message: str) -> Path:
    while True:
        raw = typer.prompt(message).strip()
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            return path.resolve()
        console.print(f"[bold red]File not found:[/bold red] {raw}")


def _prompt_manual_case() -> TrajectoryCase:
    topology = _prompt_existing_file("Topology file")
    trajectory = _prompt_existing_file("Trajectory file")
    default_label = topology.parent.name or topology.stem
    label = typer.prompt("Case label", default=default_label).strip() or default_label
    return TrajectoryCase(label=label, root=topology.parent.resolve(), topology_path=topology, trajectory_path=trajectory)


def _prompt_case_selection() -> list[TrajectoryCase]:
    search_raw = typer.prompt("Directory to scan for topology/trajectory files", default=str(Path.cwd())).strip()
    search_root = Path(search_raw).expanduser()
    discoveries = discover_simulation_cases(search_root)
    if not discoveries:
        console.print("[dim]No trajectory cases were auto-detected, so the wizard will switch to manual file input.[/dim]")
        return [_prompt_manual_case()]
    _display_discoveries(discoveries)
    while True:
        raw = typer.prompt("Choose case number(s), A for all, or 0 for manual input", default="1").strip()
        if raw.lower() in {"0", "m", "manual"}:
            return [_prompt_manual_case()]
        if raw.lower() in {"a", "all"}:
            return [item.to_case() for item in discoveries]
        try:
            indices = _parse_indices(raw, max_index=len(discoveries))
        except ValueError:
            console.print("[bold red]Please enter 0, A, or listed case numbers such as 1,3-5.[/bold red]")
            continue
        return [discoveries[index - 1].to_case() for index in indices]


def _display_masks(mask_options: Sequence[MaskOption], cases: Sequence[TrajectoryCase]) -> None:
    table = Table(title="Available MDAnalysis Selection Masks", box=box.SIMPLE_HEAVY)
    table.add_column("No.", justify="right", style="bold cyan")
    table.add_column("Mask", style="bold white")
    table.add_column("Selection", style="cyan")
    table.add_column("Atom counts", style="white")
    for index, option in enumerate(mask_options, start=1):
        if option.key == "custom":
            counts = "manual"
        elif len(cases) == 1:
            counts = str(option.atom_count or 0)
        else:
            counts = ", ".join(f"{label}: {count}" for label, count in option.counts_by_case.items())
        table.add_row(str(index), option.label, option.selection or "custom", counts)
    console.print(table)


def _option_by_key(mask_options: Sequence[MaskOption], key: str) -> MaskOption | None:
    for option in mask_options:
        if option.key == key:
            return option
    return None


def _default_mask_index(mask_options: Sequence[MaskOption], preferred_keys: Sequence[str]) -> int:
    for key in preferred_keys:
        for index, option in enumerate(mask_options, start=1):
            if option.key == key:
                return index
    return 1


def _prompt_mask(
    message: str,
    mask_options: Sequence[MaskOption],
    cases: Sequence[TrajectoryCase],
    universes: Sequence[object],
    *,
    preferred_keys: Sequence[str],
) -> str:
    default_index = _default_mask_index(mask_options, preferred_keys)
    while True:
        raw = typer.prompt(message, default=str(default_index)).strip()
        if not raw.isdigit():
            console.print("[bold red]Please choose one of the mask numbers from the table.[/bold red]")
            continue
        selected = int(raw)
        if selected < 1 or selected > len(mask_options):
            console.print("[bold red]Please choose one of the mask numbers from the table.[/bold red]")
            continue
        option = mask_options[selected - 1]
        if option.key != "custom":
            return option.selection
        custom = typer.prompt("Enter an MDAnalysis selection string").strip()
        try:
            counts = validate_selection_across_cases(cases, universes, custom)
        except SelectionValidationError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            continue
        detail = ", ".join(f"{label}: {count}" for label, count in counts.items())
        console.print(f"[dim]Custom selection atom counts: {detail}[/dim]")
        return custom


def _prompt_analysis_types() -> list[str]:
    choices = _analysis_choices()
    _display_choice_table("Trajectory analysis types", choices)
    while True:
        raw = typer.prompt("Choose analysis number(s)", default="1").strip()
        try:
            indices = _parse_indices(raw, max_index=len(choices))
        except ValueError:
            console.print("[bold red]Enter analysis numbers such as 1,3-5.[/bold red]")
            continue
        return [choices[index - 1].key for index in indices]


def _prompt_positive_float(message: str, *, default: float) -> float:
    while True:
        raw = typer.prompt(message, default=str(default)).strip()
        try:
            value = float(raw)
        except ValueError:
            console.print("[bold red]Please enter a number.[/bold red]")
            continue
        if value > 0:
            return value
        console.print("[bold red]Please enter a positive number.[/bold red]")


def _prompt_positive_int(message: str, *, default: int) -> int:
    while True:
        raw = typer.prompt(message, default=str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            console.print("[bold red]Please enter an integer.[/bold red]")
            continue
        if value > 0:
            return value
        console.print("[bold red]Please enter a positive integer.[/bold red]")


def _prompt_range(message: str, *, default: tuple[float, float]) -> tuple[float, float]:
    default_text = f"{default[0]} {default[1]}"
    while True:
        raw = typer.prompt(message, default=default_text).strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            console.print("[bold red]Enter two numbers, for example: 0 12.[/bold red]")
            continue
        try:
            low, high = float(parts[0]), float(parts[1])
        except ValueError:
            console.print("[bold red]Enter two numeric range bounds.[/bold red]")
            continue
        if low >= 0 and high > low:
            return low, high
        console.print("[bold red]Lower bound must be >= 0 and upper bound must be larger.[/bold red]")


def _prompt_frame_selection() -> TrajectoryFrameSelection:
    stride = _prompt_positive_int("Trajectory frame stride (analyze every Nth frame)", default=1)
    choices = [
        WizardChoice("all", "All frames", "Analyze the full trajectory using the selected stride."),
        WizardChoice("last_ns", "Last N ns", "Analyze only the final time window using the selected stride."),
        WizardChoice("last_frames", "Last N frames", "Analyze only the final frame window using the selected stride."),
    ]
    _display_choice_table("Trajectory frame window", choices)
    while True:
        raw = typer.prompt("Choose frame window", default="1").strip().lower()
        if raw in {"1", "all", "a"}:
            return TrajectoryFrameSelection(stride=stride)
        if raw in {"2", "last_ns", "ns", "n"}:
            last_ns = _prompt_positive_float("Analyze only the last how many ns", default=10.0)
            return TrajectoryFrameSelection(stride=stride, last_ns=last_ns)
        if raw in {"3", "last_frames", "frames", "f"}:
            last_frames = _prompt_positive_int("Analyze only the last how many frames", default=1000)
            return TrajectoryFrameSelection(stride=stride, last_frames=last_frames)
        console.print("[bold red]Please choose 1, 2, or 3.[/bold red]")


def _build_requests(
    analysis_types: Sequence[str],
    mask_options: Sequence[MaskOption],
    cases: Sequence[TrajectoryCase],
    universes: Sequence[object],
) -> list[TrajectoryAnalysisRequest]:
    requests: list[TrajectoryAnalysisRequest] = []
    for analysis_type in analysis_types:
        if analysis_type == "rg":
            target = _prompt_mask(
                "Mask for radius of gyration",
                mask_options,
                cases,
                universes,
                preferred_keys=("protein", "des", "system"),
            )
            requests.append(TrajectoryAnalysisRequest("rg", target_selection=target))
        elif analysis_type == "distance":
            selection_a = _prompt_mask(
                "Distance mask A",
                mask_options,
                cases,
                universes,
                preferred_keys=("ree", "metals", "protein", "system"),
            )
            selection_b = _prompt_mask(
                "Distance mask B",
                mask_options,
                cases,
                universes,
                preferred_keys=("protein", "water", "des", "system"),
            )
            requests.append(TrajectoryAnalysisRequest("distance", selection_a=selection_a, selection_b=selection_b))
        elif analysis_type == "rdf":
            selection_a = _prompt_mask(
                "RDF mask A",
                mask_options,
                cases,
                universes,
                preferred_keys=("ree", "metals", "protein", "system"),
            )
            selection_b = _prompt_mask(
                "RDF mask B",
                mask_options,
                cases,
                universes,
                preferred_keys=("water", "des", "protein", "system"),
            )
            nbins = _prompt_positive_int("RDF bin count", default=75)
            rdf_range = _prompt_range("RDF range in Angstroms", default=(0.0, 12.0))
            requests.append(TrajectoryAnalysisRequest("rdf", selection_a=selection_a, selection_b=selection_b, nbins=nbins, rdf_range=rdf_range))
        elif analysis_type == "rmsd":
            target = _prompt_mask(
                "RMSD target mask",
                mask_options,
                cases,
                universes,
                preferred_keys=("protein", "des", "system"),
            )
            alignment_default = ("protein_backbone",) if _option_by_key(mask_options, "protein_backbone") is not None else ()
            alignment = _prompt_mask(
                "RMSD alignment mask",
                mask_options,
                cases,
                universes,
                preferred_keys=(*alignment_default, "protein", "system"),
            )
            requests.append(TrajectoryAnalysisRequest("rmsd", target_selection=target, alignment_selection=alignment))
        elif analysis_type == "rmsf":
            target = _prompt_mask(
                "RMSF target mask",
                mask_options,
                cases,
                universes,
                preferred_keys=("protein_ca", "protein_backbone", "protein", "des", "system"),
            )
            alignment_default = ("protein_backbone",) if _option_by_key(mask_options, "protein_backbone") is not None else ()
            alignment = _prompt_mask(
                "RMSF alignment mask",
                mask_options,
                cases,
                universes,
                preferred_keys=(*alignment_default, "protein", "system"),
            )
            requests.append(TrajectoryAnalysisRequest("rmsf", target_selection=target, alignment_selection=alignment))
    return requests


def print_trajectory_analysis_summary(result: TrajectoryAnalysisRunResult) -> None:
    table = Table(title="Trajectory Analysis Complete", box=box.SIMPLE_HEAVY)
    table.add_column("Case", style="bold white")
    table.add_column("Analysis", style="cyan")
    table.add_column("CSV", style="white")
    table.add_column("PNG", style="white")
    for output in result.outputs:
        table.add_row(output.case_label, output.analysis_type, str(output.csv_path), str(output.png_path))
    console.print(table)
    console.print(f"[bold green]Output directory:[/bold green] {result.output_dir}")
    if result.combined_summary_csv_path is not None:
        console.print(f"[bold green]Combined summary:[/bold green] {result.combined_summary_csv_path}")
    for overlay in result.overlay_paths:
        console.print(f"[bold green]Overlay plot:[/bold green] {overlay}")


def run_trajectory_analysis_wizard() -> TrajectoryAnalysisRunResult:
    _print_step_header(
        1,
        "Choose Trajectory Case(s)",
        "Select one or more completed simulations with a topology and trajectory file.",
    )
    cases = _prompt_case_selection()

    _print_step_header(
        2,
        "Load Trajectories and Build Masks",
        "The wizard will inspect each system and offer generalized masks such as Protein, REE, solvent, DES components, and System.",
    )
    universes = []
    for case in cases:
        try:
            universes.append(load_universe(case))
        except Exception as exc:
            if exc.__class__.__name__ == "TrajectoryAnalysisDependencyError":
                console.print(f"[bold red]{dependency_install_hint()}[/bold red]")
            raise
        console.print(f"[dim]Loaded {case.label}: {case.topology_path.name} + {case.trajectory_path.name}[/dim]")
    mask_options = generalize_mask_options(cases, universes)
    if len(cases) == 1:
        mask_options = candidate_masks_for_universe(universes[0])
        mask_options.append(MaskOption("custom", "Custom selection", "", "custom"))
    _display_masks(mask_options, cases)

    _print_step_header(
        3,
        "Choose Analyses and Masks",
        "Select one or more analyses, then choose the atom masks used by each calculation.",
    )
    analysis_types = _prompt_analysis_types()
    requests = _build_requests(analysis_types, mask_options, cases, universes)
    time_step_ps = _prompt_positive_float("Time between trajectory frames in ps", default=1.0)

    _print_step_header(
        4,
        "Choose Frame Sampling",
        "Limit expensive analyses by skipping frames or analyzing only the final part of each trajectory.",
    )
    frame_selection = _prompt_frame_selection()
    console.print(f"[dim]Frame sampling: {describe_frame_selection(frame_selection)}[/dim]")

    _print_step_header(
        5,
        "Choose Output Directory",
        "CSV data, PNG plots, and summary files will be written under this folder.",
    )
    default_root = default_output_root(Path.cwd())
    output_raw = typer.prompt("Output directory", default=str(default_root)).strip()
    output_root = Path(output_raw).expanduser()

    print_notice(
        "Running trajectory analyses",
        "This may take a while for large trajectories. CSV files are written first, followed by PNG plots and summaries.",
        border_style="cyan",
    )
    result = run_trajectory_analyses(
        cases,
        requests,
        output_root=output_root,
        time_step_ps=time_step_ps,
        frame_selection=frame_selection,
        universes=universes,
    )
    return result
