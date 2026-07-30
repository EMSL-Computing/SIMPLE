from __future__ import annotations

import re
import subprocess
from pathlib import Path

from amber_metallo.environment import is_linux_execution_host


def _tail_excerpt(text: str, *, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _log_hint(log_path: Path | None) -> str:
    return f"See log: {log_path}" if log_path is not None else "See the captured command output for details."


def _known_failure_message(command: list[str], stdout: str, stderr: str, log_path: Path | None) -> str | None:
    command_name = Path(command[0]).name.lower() if command else ""
    context_name = f"{command_name} {(log_path.name.lower() if log_path is not None else '')}"
    combined = f"{stdout}\n{stderr}"

    if "antechamber" in context_name:
        odd_electron = "The number of electrons is odd" in combined
        charge_hint = "Please check the total charge (-nc flag) and spin multiplicity (-m flag)." in combined
        sqm_failure = 'Cannot properly run "' in combined and "sqm" in combined
        if odd_electron and charge_hint and sqm_failure:
            return (
                "Antechamber could not run because the net charge and/or spin multiplicity do not match this "
                "structure. Please inspect the ligand and enter the chemically correct charge (-nc) and "
                f"multiplicity (-m), then run again.\n\n{_log_hint(log_path)}"
            )

    if "tleap" in context_name or "teleap" in context_name:
        if "does not have a type" in combined:
            residue_names = sorted(
                {
                    match.group(1)
                    for match in re.finditer(r"\.R<([A-Za-z0-9]+)\s+\d+>", combined)
                }
            )
            residue_text = ", ".join(residue_names) if residue_names else "one or more residues"
            glycan_hint = ""
            if any(name in {"NAG", "NDG", "MAN", "BMA", "GAL", "FUC", "NANA", "SIA"} for name in residue_names):
                glycan_hint = (
                    " If this residue is a carbohydrate/glycan (for example NAG), GLYCAM or manual "
                    "carbohydrate parameters are usually more appropriate than automatic GAFF/Antechamber."
                )
            standard_residues = {
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
                "GLU",
                "GLY",
                "HID",
                "HIE",
                "HIP",
                "HIS",
                "ILE",
                "LEU",
                "LYS",
                "MET",
                "PHE",
                "PRO",
                "SER",
                "THR",
                "TRP",
                "TYR",
                "VAL",
            }
            standard_hint = ""
            if sum(1 for name in residue_names if name in standard_residues) >= 3:
                standard_hint = (
                    " Because several standard amino-acid residues are affected, the input PDB likely still has "
                    "Amber-incompatible residue/atom naming, terminal aliases, HETATM protein records, or shifted "
                    "PDB columns. Try rerunning from the newly prepared cleaned_input.pdb, or inspect the tleap "
                    "input PDB and normalize it with pdb4amber if the problem persists."
                )
            return (
                f"tleap could not assign atom types to {residue_text}. This usually means the residue "
                "template/parameters were not loaded successfully, or they do not match the residue naming/connectivity "
                f"in the PDB.{glycan_hint}{standard_hint}\n\n{_log_hint(log_path)}"
            )

    return None


def ensure_execution_host(*, dry_run: bool) -> None:
    if dry_run:
        return
    if not is_linux_execution_host():
        raise RuntimeError(
            "Amber command execution is supported on Linux/WSL/HPC only. "
            "Use --dry-run on Windows."
        )


def run_command(command: list[str], *, cwd: Path, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            f"$ {' '.join(command)}",
            "",
            "STDOUT:",
            result.stdout,
            "",
            "STDERR:",
            result.stderr,
        ]
        log_path.write_text("\n".join(payload), encoding="utf-8")
    if result.returncode != 0:
        specialized = _known_failure_message(command, result.stdout, result.stderr, log_path)
        if specialized is not None:
            raise RuntimeError(specialized)
        details: list[str] = []
        if log_path is not None:
            details.append(f"See log: {log_path}")
        stderr_excerpt = _tail_excerpt(result.stderr)
        stdout_excerpt = _tail_excerpt(result.stdout)
        if stderr_excerpt:
            details.append(f"STDERR tail:\n{stderr_excerpt}")
        if stdout_excerpt:
            details.append(f"STDOUT tail:\n{stdout_excerpt}")
        if not details:
            details.append("No stdout/stderr output was captured.")
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n\n" + "\n\n".join(details)
        )
    return result
