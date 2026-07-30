from __future__ import annotations

from contextlib import contextmanager
import json
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ModuleNotFoundError:
    Console = None
    Panel = None
    Table = None


class _PlainConsole:
    def print(self, *args, **kwargs) -> None:
        print(*args)


console = Console() if Console is not None else _PlainConsole()
SUPPORT_CONTACT = "Hoshin Kim (hoshin.kim@pnnl.gov)"


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def emit_key_value_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    if Table is None:
        console.print(title)
        for key, value in rows:
            console.print(f"{key}: {value}")
        return
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value", overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def print_notice(title: str, body: str, *, border_style: str = "cyan") -> None:
    if Panel is None:
        console.print(f"{title}\n{body}")
        return
    console.print(Panel(body, title=f"[bold]{title}[/bold]", border_style=border_style))


@contextmanager
def activity_status(message: str, *, plain_message: str | None = None, spinner: str = "dots"):
    if Console is None or not hasattr(console, "status"):
        console.print(plain_message or message)
        yield
        return
    with console.status(message, spinner=spinner):
        yield


def print_support_contact() -> None:
    body = (
        "If you hit an error or have questions about this workflow, please contact "
        f"[bold]{SUPPORT_CONTACT}[/bold]."
    )
    if Panel is None:
        console.print(body.replace("[bold]", "").replace("[/bold]", ""))
        return
    console.print(Panel(body, title="[bold]Support[/bold]", border_style="magenta"))


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def _format_float(value: float | None, suffix: str = "", digits: int = 3) -> str:
    if value is None or not isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _format_ion_counts(ions: dict[str, int] | None) -> str:
    if not ions:
        return "None"
    return ", ".join(f"{name} x {count}" for name, count in ions.items() if count > 0) or "None"


def _format_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return "None"
    return ", ".join(f"{name} x {count}" for name, count in counts.items() if count) or "None"


def _format_box_lengths(lengths: object) -> str:
    if not isinstance(lengths, (list, tuple)) or len(lengths) < 3:
        return "N/A"
    try:
        x, y, z = (float(lengths[0]), float(lengths[1]), float(lengths[2]))
    except (TypeError, ValueError):
        return "N/A"
    return f"{x:.3f} x {y:.3f} x {z:.3f}"


def _formula_units_from_extra_ions(extra_ions: dict[str, int] | None) -> int | None:
    if not extra_ions:
        return 0
    if "Ca2+" in extra_ions:
        return min(extra_ions.get("Ca2+", 0), extra_ions.get("Cl-", 0) // 2)
    if "Na+" in extra_ions:
        return min(extra_ions.get("Na+", 0), extra_ions.get("Cl-", 0))
    if "K+" in extra_ions:
        return min(extra_ions.get("K+", 0), extra_ions.get("Cl-", 0))
    return None


def _actual_salt_concentration(system: dict[str, Any]) -> float | None:
    if system.get("actual_salt_concentration_m") is not None:
        return float(system["actual_salt_concentration_m"])
    volume_angstrom3 = system.get("volume_angstrom3")
    formula_units = system.get("salt_formula_units")
    if formula_units is None:
        formula_units = _formula_units_from_extra_ions(system.get("extra_ions"))
    if volume_angstrom3 is None or formula_units in {None, 0}:
        return 0.0 if formula_units == 0 else None
    avogadro = 6.02214076e23
    volume_liters = float(volume_angstrom3) * 1e-27
    if volume_liters <= 0:
        return None
    return (float(formula_units) / avogadro) / volume_liters


def _c4_status(system: dict[str, Any]) -> str:
    if system.get("c4_applied"):
        return "Applied"
    if system.get("c4_script_path"):
        return "Helper script generated"
    return "Not required"


def _prepare_component_summary(result: dict[str, Any]) -> dict[str, Any]:
    prepare_manifest = _read_json(result.get("prepare"))
    if not prepare_manifest:
        return {}

    cleaned_summary = prepare_manifest.get("summary") or {}
    cleaned_pdb = prepare_manifest.get("cleaned_pdb")
    if cleaned_pdb:
        try:
            from amber_metallo.inspection import inspect_structure

            cleaned_summary = inspect_structure(cleaned_pdb, detect_missing_loops=False).to_dict()
        except Exception:
            pass
    return {
        "standard_residues": cleaned_summary.get("residue_counts", {}).get("standard"),
        "retained_ligands": len(cleaned_summary.get("hetero_residues", [])),
        "metal_sites": len(cleaned_summary.get("metals", [])),
    }


def _plain_workflow_summary(result: dict[str, Any]) -> str:
    resp = result.get("resp") or {}
    if resp:
        lines = [
            "RESP Setup Complete",
            f"Output directory: {result.get('output_dir', 'N/A')}",
            f"RESP job dir: {(resp.get('files') or {}).get('resp_job_dir', 'N/A')}",
            f"NWChem input: {(resp.get('files') or {}).get('nwchem_input', 'N/A')}",
            f"Slurm script: {(resp.get('files') or {}).get('slurm', 'N/A')}",
            "Run the generated job, wait for output/resp_charges.json, then rerun the workflow to apply the RESP charges.",
        ]
        return "\n".join(lines)

    system = result.get("system") or {}
    metadata = system.get("system_metadata") or {}
    des_plan = metadata.get("des") if isinstance(metadata, dict) else None
    components = _prepare_component_summary(result)
    warnings = system.get("warnings") or []
    if isinstance(des_plan, dict):
        box_lengths = system.get("box_lengths_angstrom") or des_plan.get("box_lengths_angstrom")
        lines = [
            "DES Workflow Complete",
            f"Output directory: {result.get('output_dir', 'N/A')}",
            f"Component counts: {_format_counts(des_plan.get('component_counts'))}",
            f"Expanded residue counts: {_format_counts(des_plan.get('residue_counts'))}",
            f"Total selected molecules: {sum((des_plan.get('component_counts') or {}).values())}",
            f"Total topology residues: {des_plan.get('total_residues', 'N/A')}",
            f"Initial atom count: {des_plan.get('total_atoms', 'N/A')}",
            f"Box lengths (A): {_format_box_lengths(box_lengths)}",
            f"System volume (A^3): {_format_float(system.get('volume_angstrom3'), digits=1)}",
            f"12-6-4 C4 status: {_c4_status(system)}",
            f"Warnings: {'None' if not warnings else '; '.join(warnings)}",
            f"Slurm script: {result.get('slurm', 'N/A')}",
            f"Support: {SUPPORT_CONTACT}",
        ]
        return "\n".join(lines)
    lines = [
        "Workflow Complete",
        f"Output directory: {result.get('output_dir', 'N/A')}",
        f"Standard amino acids / nucleic residues: {components.get('standard_residues', 'N/A')}",
        f"Retained ligands/custom residues: {components.get('retained_ligands', 'N/A')}",
        f"Metal sites: {components.get('metal_sites', 'N/A')}",
        f"Water molecules: {system.get('water_count', 'N/A')}",
        f"Added ions: {_format_ion_counts(system.get('added_ions'))}",
        f"Charge before ions: {system.get('charge_before_ions', 'N/A')}",
        f"Final charge: {system.get('final_charge', 'N/A')}",
        f"System volume (A^3): {_format_float(system.get('volume_angstrom3'), digits=1)}",
        f"Actual salt concentration (M): {_format_float(_actual_salt_concentration(system), digits=4)}",
        f"12-6-4 C4 status: {_c4_status(system)}",
        f"Warnings: {'None' if not warnings else '; '.join(warnings)}",
        f"Slurm script: {result.get('slurm', 'N/A')}",
        f"Support: {SUPPORT_CONTACT}",
    ]
    return "\n".join(lines)


def print_workflow_summary(result: dict[str, Any]) -> None:
    if Table is None or Panel is None:
        console.print(_plain_workflow_summary(result))
        return

    resp = result.get("resp") or {}
    if resp:
        files = resp.get("files") or {}
        summary = Table(box=None, show_header=False)
        summary.add_column("Key", style="bold white")
        summary.add_column("Value", style="cyan", overflow="fold")
        summary.add_row("Output directory", str(result.get("output_dir", "N/A")))
        summary.add_row("RESP job dir", str(files.get("resp_job_dir", "N/A")))
        summary.add_row("NWChem input", str(files.get("nwchem_input", "N/A")))
        summary.add_row("Slurm script", str(files.get("slurm", "N/A")))
        summary.add_row("Next step", "Run the RESP job and rerun the workflow to apply the generated charges.")
        console.print(Panel(summary, title="[bold]RESP Setup Complete[/bold]", border_style="green"))
        return

    system = result.get("system") or {}
    metadata = system.get("system_metadata") or {}
    des_plan = metadata.get("des") if isinstance(metadata, dict) else None
    components = _prepare_component_summary(result)
    warnings = system.get("warnings") or []

    overview = Table(box=None, show_header=False)
    overview.add_column("Key", style="bold white")
    overview.add_column("Value", style="cyan", overflow="fold")
    overview.add_row("Output directory", str(result.get("output_dir", "N/A")))
    overview.add_row("Amber home", str(result.get("amberhome") or "Not detected"))
    overview.add_row("Warnings", "None" if not warnings else "; ".join(warnings))
    console.print(Panel(overview, title="[bold]Workflow Complete[/bold]", border_style="green"))

    if isinstance(des_plan, dict):
        system_table = Table(title="DES System Summary")
        system_table.add_column("Item", style="bold white")
        system_table.add_column("Value", style="cyan", overflow="fold")
        component_counts = des_plan.get("component_counts") or {}
        residue_counts = des_plan.get("residue_counts") or {}
        box_lengths = system.get("box_lengths_angstrom") or des_plan.get("box_lengths_angstrom")
        box_source = str((metadata or {}).get("box_lengths_source") or "planned")
        system_table.add_row("Component counts", _format_counts(component_counts))
        system_table.add_row("Expanded residue counts", _format_counts(residue_counts))
        system_table.add_row("Total selected molecules", str(sum(component_counts.values()) if isinstance(component_counts, dict) else "N/A"))
        system_table.add_row("Total topology residues", str(des_plan.get("total_residues", "N/A")))
        system_table.add_row("Initial atom count", str(des_plan.get("total_atoms", "N/A")))
        system_table.add_row("Box lengths (A)", f"{_format_box_lengths(box_lengths)} ({box_source})")
        system_table.add_row("System volume (A^3)", _format_float(system.get("volume_angstrom3"), digits=1))
        system_table.add_row("12-6-4 C4 status", _c4_status(system))
        console.print(system_table)
    else:
        system_table = Table(title="System Summary")
        system_table.add_column("Item", style="bold white")
        system_table.add_column("Value", style="cyan", overflow="fold")
        system_table.add_row("Standard amino acids / nucleic residues", str(components.get("standard_residues", "N/A")))
        system_table.add_row("Retained ligands/custom residues", str(components.get("retained_ligands", "N/A")))
        system_table.add_row("Metal sites", str(components.get("metal_sites", "N/A")))
        system_table.add_row("Water molecules", str(system.get("water_count", "N/A")))
        system_table.add_row("Added ions", _format_ion_counts(system.get("added_ions")))
        system_table.add_row("Charge before ions", _format_float(system.get("charge_before_ions"), digits=1))
        system_table.add_row("Final charge", _format_float(system.get("final_charge"), digits=1))
        system_table.add_row("System volume (A^3)", _format_float(system.get("volume_angstrom3"), digits=1))
        system_table.add_row("Actual salt concentration (M)", _format_float(_actual_salt_concentration(system), digits=4))
        system_table.add_row("12-6-4 C4 status", _c4_status(system))
        console.print(system_table)

    outputs = Table(title="Generated Files")
    outputs.add_column("Item", style="bold white")
    outputs.add_column("Path", style="cyan", overflow="fold")
    output_files = system.get("output_files") or {}
    outputs.add_row("Topology", str(output_files.get("prmtop", "N/A")))
    outputs.add_row("Coordinates", str(output_files.get("inpcrd", "N/A")))
    outputs.add_row("MD inputs", f"{len(result.get('md_inputs') or [])} files")
    outputs.add_row("Slurm script", str(result.get("slurm", "N/A")))
    console.print(outputs)
    print_support_contact()
