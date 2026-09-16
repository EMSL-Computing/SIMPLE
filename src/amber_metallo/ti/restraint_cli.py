"""Interactive setup only: never changes TI integration/analysis options."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile

import typer
from rich.table import Table

from amber_metallo.cli import WizardChoice, _display_choice_table, _prompt_choice
from amber_metallo.reporting import activity_status, console
from amber_metallo.ti.analysis import detect_bound_metal_sites
from amber_metallo.ti.config import (
    CoordinationRestraintConfig, RestraintReference, TIDecouplingMode, TIImplementationMode,
    DEFAULT_COORDINATION_CUTOFF_ANGSTROM,
)
from amber_metallo.ti.coordination import (
    CORRECTION_NOTICE, inspect_coordination, prepare_restraint_snapshot, validate_restraint_atom_order,
    candidates_in_frame, restraint_neighborhood,
)
from amber_metallo.ti.atom_mapping import map_to_topology, remap_candidates
from amber_metallo.ti.full_structure import IncompleteReferenceError, normalize_full_reference


def prepare_full_reference_input(complex_input, *, output_dir, coordinate_candidates=()):
    """Normalize before metal detection/CN, prompting only for genuinely missing coordinates."""
    candidates = list(coordinate_candidates)
    while True:
        try:
            full = normalize_full_reference(
                reference_pdb=complex_input.reference_structure_path, prmtop_path=complex_input.prmtop_path,
                output_dir=output_dir, coordinate_candidates=candidates)
            complex_input.reference_structure_path = str(full)
            console.print(f"[cyan]Topology-complete reference validated (H, solvent and EP included): {full}[/cyan]")
            return full
        except IncompleteReferenceError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            raw = typer.prompt("Matching full reference coordinates (PDB / rst7 / inpcrd / NetCDF; B cancels)").strip()
            if raw.lower() in {"b", "back", "cancel"}:
                raise typer.Abort()
            path = Path(raw).expanduser()
            if not path.is_file():
                console.print("[red]Coordinate file not found.[/red]")
                continue
            candidates.append(path)


def _positive_float(prompt: str, default: float, *, minimum: float = 0.001) -> float:
    while True:
        raw = typer.prompt(prompt, default=str(default))
        try:
            value = float(raw)
            if math.isfinite(value) and value >= minimum:
                return value
        except ValueError:
            pass
        console.print(f"[red]Enter a finite value >= {minimum:g}.[/red]")


def prompt_coordination_restraints(config, *, dry_run: bool) -> None:
    """Inspect the selected frame immediately after snapshot selection, then ask."""
    console.print("[bold cyan]Inspect selected TI frame and choose coordination restraints[/bold cyan]")
    prepare_full_reference_input(config.complex_input, output_dir=Path(".simple_ti_wizard") / "full_references")
    cutoff = DEFAULT_COORDINATION_CUTOFF_ANGSTROM
    candidates = detect_bound_metal_sites(
        config.complex_input.reference_structure_path, config.complex_input.prmtop_path,
        donor_cutoff_angstrom=config.snapshot.donor_cutoff_angstrom, include_unbound_metals=True,
    )
    selected = set(config.metal.selected_sites or [config.metal.selected_site])
    candidates = [item for item in candidates if item.site in selected]
    if not candidates:
        raise typer.BadParameter("No selected metal site is available for restraint setup")
    preview_root = Path(".simple_ti_wizard") / "restraint_previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(tempfile.mkdtemp(prefix="case_", dir=preview_root.resolve()))
    with activity_status("Preparing the actual TI frame for restraint comparison..."):
        prepared = prepare_restraint_snapshot(config, candidates, output_dir=preview_dir, dry_run=dry_run)
    if "selected_time_ns" in prepared.metadata:
        console.print(
            f"[cyan]Preview: frame {prepared.metadata['frame_index']} at "
            f"{prepared.metadata['selected_time_ns']:g} ns (requested {config.snapshot.time_ns:g} ns).[/cyan]"
        )
    comparison_kwargs = dict(
        reference_pdb=config.complex_input.reference_structure_path, frame_pdb=prepared.pdb_path,
        prmtop_path=config.complex_input.prmtop_path, candidates=candidates,
        frame_label=config.snapshot.mode.value.capitalize() + " frame", dry_run=dry_run,
    )
    geometries = display_reference_frame_comparison(**comparison_kwargs, cutoff=cutoff)
    adjusted = _positive_float("Coordination distance cutoff (Angstrom; geometric O count)", cutoff)
    if adjusted != cutoff:
        cutoff = adjusted
        geometries = display_reference_frame_comparison(**comparison_kwargs, cutoff=cutoff)
    console.print(
        "[dim]Native AMBER flat-bottom metal-donor distances (no PLUMED). "
        "Yes adds donor-pair restraints; No preserves existing TI and legacy restraint behavior.[/dim]"
    )
    if not typer.confirm("Apply metal-donor flat-bottom restraints throughout TI?", default=False):
        config.ti.coordination_restraint = None
        console.print("[dim]No new coordination restraint: continuing with the existing TI behavior.[/dim]")
        return
    if geometries is None:
        raise typer.BadParameter("Cannot identify the selected metal in the reference/frame. Fix that correspondence and rerun.")
    supported = (
        config.ti.implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI
        and config.ti.decoupling_mode == TIDecouplingMode.COMBINED_Q_VDW
    )
    if not supported:
        console.print(
            "[yellow]New donor-pair restraints require 12-6-4 GTI + combined Q/VDW. "
            "Split Q-off duplicate-metal weighting has not been validated.[/yellow]"
        )
        if not typer.confirm("Switch this setup to 12-6-4 GTI + combined Q/VDW to use these restraints?", default=False):
            config.ti.coordination_restraint = None
            console.print("[dim]TI mode unchanged; no new coordination restraint was added.[/dim]")
            return
        config.ti.implementation_mode = TIImplementationMode.AMBER_12_6_4_GTI
        config.ti.decoupling_mode = TIDecouplingMode.COMBINED_Q_VDW
    reference, snapshot = geometries
    choices = [
        WizardChoice("snapshot", "Selected TI frame", "Use the exact Last / Cluster / time frame previewed above."),
        WizardChoice("reference_pdb", "Reference PDB", "Use original input-PDB donor distances; TI starts from the selected frame."),
    ]
    _display_choice_table("Restraint distance reference", choices)
    source = RestraintReference(_prompt_choice("Choose the restraint reference", choices, default_key="snapshot"))
    geometry = reference if source == RestraintReference.REFERENCE_PDB else snapshot
    try:
        # Both structures must be full and topology ordered before selecting anchors.
        validate_restraint_atom_order(config.complex_input.reference_structure_path, config.complex_input.prmtop_path)
        validate_restraint_atom_order(prepared.pdb_path, config.complex_input.prmtop_path)
        mapping = map_to_topology(geometry.source_path, config.complex_input.prmtop_path)
        ref_mapping = map_to_topology(config.complex_input.reference_structure_path, config.complex_input.prmtop_path)
        remap_candidates(candidates, ref_mapping)
    except ValueError as exc:
        raise typer.BadParameter(f"CN was reported independently, but restraint indices must be unambiguous: {exc}") from exc
    basis_candidates = candidates if source == RestraintReference.REFERENCE_PDB else candidates_in_frame(
        config.complex_input.reference_structure_path, prepared.pdb_path, candidates)
    donors_by_site = {}
    targets = []
    for candidate in basis_candidates:
        initial = [mapping[item["atom_index"]] for item in geometry.sites[candidate.site]["donors"]
                   if item["atom_index"] in mapping]
        indices, distances = edit_restraint_atoms(geometry.source_path, candidate, mapping, initial, cutoff=cutoff)
        donors_by_site[candidate.site] = indices
        targets.extend(distances)
    while True:
        width = _positive_float("Flat-bottom half-width (+/- Angstrom)", 0.5)
        if width < min(targets):
            break
        console.print("[red]Half-width must be smaller than every selected metal-donor distance.[/red]")
    strength = _positive_float("AMBER rk2 = rk3 (kcal/mol/Angstrom^2; U = rk * deviation^2)", 5.0)
    console.print(f"[yellow]{CORRECTION_NOTICE}[/yellow]")
    console.print(
        "[dim]The exact preview frame is saved and reused. Keep .simple_ti_wizard/full_references and restraint_previews "
        "before generating the TI setup. Each selected metal has its own donor list; "
        "separate-site runs from this selection share the previewed frame.[/dim]"
    )
    config.ti.coordination_restraint = CoordinationRestraintConfig(
        enabled=True, reference=source, donor_cutoff_angstrom=cutoff, half_width_angstrom=width,
        force_constant=strength, donor_indices_by_site=donors_by_site, prepared_snapshot=prepared,
    )


def display_reference_frame_comparison(
    *, reference_pdb, frame_pdb, prmtop_path, candidates,
    frame_label: str = "Last frame", cutoff: float = DEFAULT_COORDINATION_CUTOFF_ANGSTROM,
    dry_run: bool,
):
    """Show actual contacts before selecting a frame; leave legacy No available on error."""
    console.print(
        f"[cyan]Geometric oxygen coordination; current cutoff: {cutoff:g} A.[/cyan]"
    )
    if dry_run:
        console.print("[bold yellow]DRY RUN: frame is a reference-PDB placeholder, NOT measured trajectory CN.[/bold yellow]")
    try:
        reference = inspect_coordination(reference_pdb, candidates, cutoff=cutoff, report_only=True)
        frame_candidates = candidates_in_frame(reference_pdb, frame_pdb, candidates)
        snapshot = inspect_coordination(frame_pdb, frame_candidates, cutoff=cutoff, report_only=True)
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"[yellow]Coordination comparison unavailable: {exc}[/yellow]")
        return None
    comparison = Table(title="Reference PDB vs " + frame_label + " coordination")
    for name in ("Site", "Source", "Non-water O-CN", "Water O-CN", "Total O-CN"):
        comparison.add_column(name)
    for candidate in candidates:
        for source, geometry in (("Reference PDB", reference), (frame_label, snapshot)):
            info = geometry.sites[candidate.site]
            comparison.add_row(f"{candidate.site}: {candidate.element}", source,
                               str(info["oxygen_count"]), str(info["water_oxygen_count"]),
                               str(info["oxygen_count"] + info["water_oxygen_count"]))
    console.print(comparison)
    try:
        validate_restraint_atom_order(reference_pdb, prmtop_path)
        validate_restraint_atom_order(frame_pdb, prmtop_path)
        index_label = "Topology atom (full PDB)"
        console.print("[dim]Both PDBs are topology-complete, including H, solvent and EP; atom identities/order validated.[/dim]")
    except (ValueError, OSError):
        # Read-only callers may still request a geometric comparison of incomplete PDBs.
        index_label = "PDB row (NOT topology)"
    console.print("[dim]O-CN counts nearby oxygen atoms only, independent of bond assignments. "
                  "Non-water N/O/S contacts initially seed restraints; water is display-only.[/dim]")
    for candidate in candidates:
        distances = Table(title=f"Site {candidate.site}: local oxygen identity and distance (A)")
        for name in ("Source", index_label, "Residue / atom", "Distance"):
            distances.add_column(name)
        for label, geometry in (("Reference PDB", reference), (frame_label, snapshot)):
            info = geometry.sites[candidate.site]
            for atom in info["donors"] + info["waters"]:
                if atom["element"] == "O":
                    distances.add_row(label, str(atom["atom_index"]), atom["label"], f"{atom['distance_angstrom']:.3f}")
        console.print(distances)
    return reference, snapshot


def edit_restraint_atoms(pdb_path, candidate, mapping, initial, *, cutoff):
    """Edit topology-indexed anchors; newly revealed neighbors are never added automatically."""
    selected = set(initial)
    expansion = 1.0
    console.print("[cyan]Initial selection: non-water N/O/S within the cutoff. "
                  "Candidates include other non-water heavy atoms for explicit selection.[/cyan]")
    console.print("[dim]Commands: +1 / +2 (expand by that many A, repeatedly); "
                  "add 123,145; remove 123; add r4; remove r4; done. "
                  "rN selects visible atoms in the listed residue, not a residue-COM restraint. "
                  "Expansion changes visibility only, NOT the flat-bottom width.[/dim]")
    while True:
        rows = restraint_neighborhood(pdb_path, candidate, mapping, selected=selected, cutoff=cutoff, expansion=expansion)
        table = Table(title=f"Site {candidate.site}: metal radius {cutoff + expansion:g} A; "
                            f"selected atom/protein-residue expansion {expansion:g} A")
        for name in ("Use", "Topology atom", "Residue ID", "Residue (chain/number/type)", "Atom", "Element", "Metal distance (A)"):
            table.add_column(name)
        for row in rows:
            table.add_row("*" if row["selected"] else "", str(row["topology_index"] or "unmapped"),
                          f"r{row['residue_id']}", row["residue"], row["atom_name"], row["element"],
                          f"{row['distance_angstrom']:.3f}")
        console.print(table)
        raw = typer.prompt(f"Site {candidate.site}: edit restraint atoms", default="done").strip().lower()
        if raw in {"done", "d", "all"}:  # 'all' retains the old initial-donor-list shortcut.
            if selected:
                distances = {row["topology_index"]: row["distance_angstrom"] for row in rows}
                return sorted(selected), [distances[i] for i in sorted(selected)]
            console.print("[red]Select at least one anchor (or Ctrl+C to cancel setup).[/red]")
            continue
        if raw.startswith("+"):
            try:
                increment = float(raw[1:])
                if not math.isfinite(increment) or increment <= 0:
                    raise ValueError
                expansion += increment
                continue
            except ValueError:
                console.print("[red]Enter a positive finite expansion, e.g. +1 or +2.[/red]")
                continue
        try:
            command, tokens = raw.split(maxsplit=1)
            if command not in {"add", "remove"}:
                raise ValueError
            wanted = set()
            available = {row["topology_index"] for row in rows if row["topology_index"] is not None}
            for token in tokens.replace(",", " ").split():
                if token.startswith("r"):
                    group = {row["topology_index"] for row in rows if row["residue_id"] == int(token[1:])}
                    if not group or None in group:
                        raise ValueError
                    wanted.update(group)
                else:
                    wanted.add(int(token))
            if not wanted or not wanted <= available:
                raise ValueError
            selected = selected | wanted if command == "add" else selected - wanted
        except ValueError:
            console.print("[red]Use listed, uniquely mapped topology atoms or rN residues with add/remove; "
                          "expand first to reveal other atoms.[/red]")
