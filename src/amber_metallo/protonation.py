from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import gemmi

from amber_metallo.config import PrepareConfig, ProtonationChange
from amber_metallo.inspection import classify_residue, load_structure


SUPPORTED_PROTONATION_RESIDUES = {
    "ASP",
    "ASH",
    "GLU",
    "GLH",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "LYS",
    "LYN",
    "CYS",
    "CYM",
}
_PROPKA_PKA_PATTERN = re.compile(r"[-+]?\d+\.\d+")
PROPKA_ION_LABELS = {
    "Co": "CO",
    "Cu": "CU",
    "Ni": "NI",
    "Mn": "MN",
    "Fe": "FE",
}
PROPKA_SURROGATE_ION_LABELS = {
    "Y": "FE",
    "La": "FE",
    "Nd": "FE",
    "Eu": "FE",
    "Lu": "FE",
}
DIRECT_METAL_COORDINATION_CUTOFF_ANGSTROM = 3.0
DIRECT_COORDINATION_DONOR_ATOMS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "CYS": ("SG",),
    "MET": ("SD",),
}
PROPKA_PYTHON_ENVVAR = "SIMPLE_PROPKA_PYTHON"


@dataclass(slots=True)
class _ResidueReference:
    chain: str
    seqid: str
    residue_name: str
    residue_number: int

    @property
    def locator(self) -> tuple[str, str]:
        return self.chain, self.seqid


@dataclass(slots=True)
class _PropkaEntry:
    chain: str
    residue_number: int
    residue_name: str
    predicted_pka: float


@dataclass(slots=True)
class _MetalAtomReference:
    chain: str
    seqid: str
    element: str
    position: gemmi.Position

    @property
    def locator(self) -> tuple[str, str]:
        return self.chain, self.seqid


@dataclass(slots=True)
class ProtonationPrediction:
    changes: list[ProtonationChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metal_coordination_candidates: list["ProtonationDisplayCandidate"] = field(default_factory=list)
    propka_candidates: list["ProtonationDisplayCandidate"] = field(default_factory=list)


@dataclass(slots=True)
class ProtonationDisplayCandidate:
    chain: str
    seqid: str
    original_residue_name: str
    target_residue_name: str
    predicted_pka: float | None
    metal_near: bool
    reason: str
    selectable: bool
    change: ProtonationChange | None = None

    @property
    def locator(self) -> tuple[str, str]:
        return self.chain, self.seqid


@dataclass(slots=True)
class _DirectCoordinationObservation:
    chain: str
    seqid: str
    residue_name: str
    donor_atom_name: str
    metal_element: str
    distance_angstrom: float
    recommended_target: str | None

    @property
    def locator(self) -> tuple[str, str]:
        return self.chain, self.seqid


def _normalize_seqid(seqid: str) -> str:
    return seqid.strip()


def residue_locator(chain_name: str, residue: gemmi.Residue) -> tuple[str, str]:
    return chain_name.strip(), _normalize_seqid(str(residue.seqid))


def _remove_extra_models(structure: gemmi.Structure) -> None:
    while len(structure) > 1:
        del structure[1]


def _protonation_family(residue_name: str) -> str | None:
    normalized = residue_name.strip().upper()
    if normalized in {"ASP", "ASH"}:
        return "ASP"
    if normalized in {"GLU", "GLH"}:
        return "GLU"
    if normalized in {"HIS", "HID", "HIE", "HIP"}:
        return "HIS"
    if normalized in {"LYS", "LYN"}:
        return "LYS"
    if normalized in {"CYS", "CYM"}:
        return "CYS"
    return None


def _direct_coordination_family(residue_name: str) -> str | None:
    protonation_family = _protonation_family(residue_name)
    if protonation_family is not None:
        return protonation_family
    if residue_name.strip().upper() == "MET":
        return "MET"
    return None


def _recommended_target_for_direct_coordination(residue_name: str, donor_atom_name: str) -> str | None:
    family = _direct_coordination_family(residue_name)
    donor = donor_atom_name.strip().upper()
    if family == "HIS":
        if donor == "ND1":
            return "HIE"
        if donor == "NE2":
            return "HID"
        return None
    if family == "CYS":
        return "CYM"
    if family == "ASP":
        return "ASP"
    if family == "GLU":
        return "GLU"
    return None


def _iter_supported_residues(structure: gemmi.Structure) -> list[_ResidueReference]:
    return _iter_supported_residues_excluding(structure, excluded_residue_locators=None)


def _iter_supported_residues_excluding(
    structure: gemmi.Structure,
    *,
    excluded_residue_locators: set[tuple[str, str]] | None,
) -> list[_ResidueReference]:
    references: list[_ResidueReference] = []
    excluded = excluded_residue_locators or set()
    for chain in structure[0]:
        for residue in chain:
            locator = residue_locator(chain.name, residue)
            if locator in excluded:
                continue
            if classify_residue(residue) != "standard":
                continue
            residue_name = residue.name.strip().upper()
            if residue_name not in SUPPORTED_PROTONATION_RESIDUES:
                continue
            references.append(
                _ResidueReference(
                    chain=chain.name.strip(),
                    seqid=_normalize_seqid(str(residue.seqid)),
                    residue_name=residue_name,
                    residue_number=int(residue.seqid.num),
                )
            )
    return references


def _selected_metal_atoms(
    structure: gemmi.Structure,
    prepare_config: PrepareConfig | None,
) -> list[_MetalAtomReference]:
    deletion_sites = set() if prepare_config is None else set(prepare_config.metal_deletions)
    metal_index = 0
    metal_atoms: list[_MetalAtomReference] = []

    for chain in structure[0]:
        for residue in chain:
            if classify_residue(residue) != "metal":
                continue
            metal_index += 1
            if prepare_config is not None and (prepare_config.remove_metals or metal_index in deletion_sites):
                continue
            for atom in residue:
                metal_atoms.append(
                    _MetalAtomReference(
                        chain=chain.name.strip(),
                        seqid=_normalize_seqid(str(residue.seqid)),
                        element=atom.element.name.title(),
                        position=atom.pos,
                    )
                )
    return metal_atoms


def _selected_metal_positions(
    structure: gemmi.Structure,
    prepare_config: PrepareConfig | None,
) -> tuple[set[tuple[str, str]], list[gemmi.Position]]:
    metal_atoms = _selected_metal_atoms(structure, prepare_config)
    return {atom.locator for atom in metal_atoms}, [atom.position for atom in metal_atoms]


def focused_restraint_residue_locators(
    source: str | Path | gemmi.Structure,
    prepare_config: PrepareConfig | None,
    *,
    cutoff_angstrom: float = 4.0,
) -> set[tuple[str, str]]:
    structure = source if isinstance(source, gemmi.Structure) else load_structure(source)
    _remove_extra_models(structure)
    selected_metals, metal_positions = _selected_metal_positions(structure, prepare_config)
    selected_locators = set(selected_metals)

    if not metal_positions:
        return selected_locators

    for chain in structure[0]:
        for residue in chain:
            if classify_residue(residue) != "standard":
                continue
            for atom in residue:
                if any(atom.pos.dist(metal_pos) <= cutoff_angstrom for metal_pos in metal_positions):
                    selected_locators.add(residue_locator(chain.name, residue))
                    break
    return selected_locators


def metal_neighbor_residue_locators(
    source: str | Path | gemmi.Structure,
    prepare_config: PrepareConfig | None,
    *,
    cutoff_angstrom: float = 4.0,
) -> set[tuple[str, str]]:
    structure = source if isinstance(source, gemmi.Structure) else load_structure(source)
    _remove_extra_models(structure)
    selected_metals, _ = _selected_metal_positions(structure, prepare_config)
    selected_locators = focused_restraint_residue_locators(
        structure,
        prepare_config,
        cutoff_angstrom=cutoff_angstrom,
    )
    return {locator for locator in selected_locators if locator not in selected_metals}


def direct_metal_coordination_observations(
    source: str | Path | gemmi.Structure,
    prepare_config: PrepareConfig | None,
    *,
    cutoff_angstrom: float = DIRECT_METAL_COORDINATION_CUTOFF_ANGSTROM,
    excluded_residue_locators: set[tuple[str, str]] | None = None,
) -> list[_DirectCoordinationObservation]:
    structure = source if isinstance(source, gemmi.Structure) else load_structure(source)
    _remove_extra_models(structure)
    metal_atoms = _selected_metal_atoms(structure, prepare_config)
    if not metal_atoms:
        return []

    observations: list[_DirectCoordinationObservation] = []
    excluded = excluded_residue_locators or set()
    for chain in structure[0]:
        for residue in chain:
            if residue_locator(chain.name, residue) in excluded:
                continue
            if classify_residue(residue) != "standard":
                continue
            family = _direct_coordination_family(residue.name)
            if family is None:
                continue
            donor_names = DIRECT_COORDINATION_DONOR_ATOMS.get(family, ())
            best: tuple[float, str, _MetalAtomReference] | None = None
            for atom in residue:
                atom_name = atom.name.strip().upper()
                if atom_name not in donor_names:
                    continue
                nearest = min(
                    ((atom.pos.dist(metal.position), metal) for metal in metal_atoms),
                    default=None,
                    key=lambda item: item[0],
                )
                if nearest is None:
                    continue
                distance, metal = nearest
                if best is None or distance < best[0]:
                    best = (distance, atom_name, metal)
            if best is None or best[0] > cutoff_angstrom:
                continue
            distance, donor_atom_name, metal = best
            observations.append(
                _DirectCoordinationObservation(
                    chain=chain.name.strip(),
                    seqid=_normalize_seqid(str(residue.seqid)),
                    residue_name=residue.name.strip().upper(),
                    donor_atom_name=donor_atom_name,
                    metal_element=metal.element.upper(),
                    distance_angstrom=distance,
                    recommended_target=_recommended_target_for_direct_coordination(residue.name, donor_atom_name),
                )
            )
    return observations


def _propka_command(structure_path: Path, ph: float, titrate_only: list[str]) -> list[str]:
    python_runner = os.environ.get(PROPKA_PYTHON_ENVVAR, "").strip() or sys.executable
    command = [
        python_runner,
        "-m",
        "propka",
        "--quiet",
        f"--pH={float(ph):.3f}",
    ]
    if titrate_only:
        command.append(f"--titrate_only={','.join(titrate_only)}")
    command.append(str(structure_path))
    return command


def _installed_propka_version() -> str | None:
    try:
        return importlib_metadata.version("propka")
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _looks_like_propka_runtime_compatibility_failure(output: str) -> bool:
    normalized = output.replace('"', "'")
    return (
        "propka/parameters.py" in normalized
        and "object has no attribute '__annotations__'" in normalized
    )


def _propka_failure_message(command: list[str], output: str) -> str:
    details = [
        "PROPKA failed while predicting protonation states.",
        "",
        f"Command: {' '.join(command)}",
        "",
    ]
    captured_output = output.strip() or "No stdout/stderr output was captured."
    details.append(captured_output)
    if _looks_like_propka_runtime_compatibility_failure(output):
        version = _installed_propka_version()
        version_note = f" Detected propka package version: {version}." if version else ""
        details.extend(
            [
                "",
                "This traceback matches a known PROPKA runtime compatibility problem, where an older PROPKA build "
                "is launched with a newer Python interpreter."
                + version_note,
                "Try `pip install -U propka` in the Python environment used for PROPKA. "
                f"If that still fails, run SIMPLE with Python 3.12/3.13 or set `{PROPKA_PYTHON_ENVVAR}` "
                "to a Python interpreter where PROPKA already works.",
            ]
        )
    return "\n".join(details)


def _run_propka(structure_path: Path, ph: float, titrate_only: list[str]) -> Path:
    command = _propka_command(structure_path, ph, titrate_only)
    try:
        result = subprocess.run(
            command,
            cwd=str(structure_path.parent),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        runner = command[0]
        if runner != sys.executable:
            raise RuntimeError(
                f"PROPKA runner '{runner}' from `{PROPKA_PYTHON_ENVVAR}` was not found. "
                "Point it to a valid Python interpreter or unset the variable."
            ) from exc
        raise RuntimeError("The active Python interpreter could not be used to launch PROPKA.") from exc
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        if "No module named propka" in combined:
            raise RuntimeError(
                "PROPKA is not installed in the active Python environment. "
                "Install project dependencies and try again."
            )
        raise RuntimeError(_propka_failure_message(command, combined))

    pka_path = structure_path.parent / f"{structure_path.stem}.pka"
    if pka_path.exists():
        return pka_path

    matches = sorted(structure_path.parent.glob("*.pka"))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError("PROPKA completed, but no .pka output file was found.")


def parse_propka_output(path: str | Path) -> list[_PropkaEntry]:
    entries: list[_PropkaEntry] = []
    in_summary = False
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if in_summary:
            if line.startswith("---"):
                break
            label = line[3:12]
            pka_match = _PROPKA_PKA_PATTERN.search(line[13:])
            if pka_match is None:
                continue
            residue_name = label[:3].strip().upper()
            if residue_name not in SUPPORTED_PROTONATION_RESIDUES:
                continue
            residue_number_text = label[3:7].strip()
            if not residue_number_text:
                continue
            entries.append(
                _PropkaEntry(
                    chain=label[7:].strip(),
                    residue_number=int(residue_number_text),
                    residue_name=residue_name,
                    predicted_pka=float(pka_match.group()),
                )
            )
            continue
        if "model-pKa" in line:
            in_summary = True
    return entries


def _propka_ion_label(element: str) -> str | None:
    normalized = element.strip().title()
    if normalized in PROPKA_ION_LABELS:
        return PROPKA_ION_LABELS[normalized]
    return PROPKA_SURROGATE_ION_LABELS.get(normalized)


def _format_surrogate_warning(chain: str, seqid: str, original_element: str, surrogate_label: str) -> str:
    return (
        f"Metal {chain}:{seqid} ({original_element}) is not in PROPKA's default ion list, so it is approximated as "
        f"{surrogate_label.title()}3+ ({surrogate_label}) for pKa prediction only."
    )


def _write_propka_input_structure(
    source_structure: gemmi.Structure,
    destination: Path,
    prepare_config: PrepareConfig | None,
) -> list[str]:
    structure = source_structure.clone()
    _remove_extra_models(structure)
    replacement_map = (
        {} if prepare_config is None else {item.site: item.target.title() for item in prepare_config.metal_replacements}
    )
    deletion_sites = set() if prepare_config is None else set(prepare_config.metal_deletions)
    warnings: list[str] = []
    seen_warnings: set[tuple[str, str, str]] = set()
    metal_index = 0

    for chain in structure[0]:
        delete_indices: list[int] = []
        for residue_index, residue in enumerate(chain):
            if classify_residue(residue) != "metal":
                continue
            metal_index += 1
            if prepare_config is not None and (prepare_config.remove_metals or metal_index in deletion_sites):
                delete_indices.append(residue_index)
                continue

            atom = residue[0]
            element = replacement_map.get(metal_index, atom.element.name.title())
            propka_label = _propka_ion_label(element)
            if propka_label is None:
                continue

            atom.element = gemmi.Element(propka_label.title())
            atom.name = propka_label[:4]
            residue.name = propka_label

            if element in PROPKA_SURROGATE_ION_LABELS:
                locator = (chain.name.strip(), _normalize_seqid(str(residue.seqid)), element)
                if locator not in seen_warnings:
                    warnings.append(
                        _format_surrogate_warning(
                            chain.name.strip() or "(blank)",
                            _normalize_seqid(str(residue.seqid)),
                            element,
                            propka_label,
                        )
                    )
                    seen_warnings.add(locator)
        for residue_index in reversed(delete_indices):
            del chain[residue_index]

    structure.remove_empty_chains()
    structure.write_pdb(str(destination))
    return warnings


def _target_residue_name(current_name: str, predicted_pka: float, ph: float) -> str:
    family = _protonation_family(current_name)
    if family is None:
        return current_name.strip().upper()

    protonated = float(ph) <= float(predicted_pka)
    current = current_name.strip().upper()
    if family == "ASP":
        return "ASH" if protonated else "ASP"
    if family == "GLU":
        return "GLH" if protonated else "GLU"
    if family == "LYS":
        return "LYS" if protonated else "LYN"
    if family == "CYS":
        return "CYS" if protonated else "CYM"
    if protonated:
        return "HIP"
    if current in {"HID", "HIE"}:
        return current
    return "HIS"


def _coordination_reason(observation: _DirectCoordinationObservation, *, selectable: bool) -> str:
    base = (
        f"Direct metal coordination via {observation.donor_atom_name}-{observation.metal_element} "
        f"({observation.distance_angstrom:.2f} A)."
    )
    if observation.recommended_target is None:
        return base + " Shown for review only; no automatic protonation-state rename is defined."
    if selectable:
        return base + " Coordinating donor is typically modeled in the deprotonated state."
    return base + " Current residue name already matches the recommended coordinated state."


def _propka_reason(change: ProtonationChange, *, ph: float) -> str:
    direction = "protonated" if float(ph) <= float(change.predicted_pka) else "deprotonated"
    return f"PROPKA predicts pKa {change.predicted_pka:.2f}; pH {ph:.2f} favors the {direction} form."


def _match_reference(
    entry: _PropkaEntry,
    candidates: list[_ResidueReference],
) -> _ResidueReference:
    if len(candidates) == 1:
        return candidates[0]

    same_family = [
        candidate
        for candidate in candidates
        if _protonation_family(candidate.residue_name) == _protonation_family(entry.residue_name)
    ]
    if len(same_family) == 1:
        return same_family[0]

    exact_name = [candidate for candidate in same_family if candidate.residue_name == entry.residue_name]
    if len(exact_name) == 1:
        return exact_name[0]

    raise RuntimeError(
        "PROPKA residue matching is ambiguous for "
        f"{entry.chain}:{entry.residue_number} ({entry.residue_name})."
    )


def predict_protonation_prediction(
    structure_path: str | Path,
    prepare_config: PrepareConfig,
    *,
    ph: float,
    structure_is_prepared: bool = False,
    excluded_residue_locators: set[tuple[str, str]] | None = None,
) -> ProtonationPrediction:
    target_path = Path(structure_path).expanduser().resolve()
    structure = load_structure(target_path)
    _remove_extra_models(structure)
    metal_selection_config = None if structure_is_prepared else prepare_config
    excluded = excluded_residue_locators or set()

    residue_references = _iter_supported_residues_excluding(
        structure,
        excluded_residue_locators=excluded,
    )
    if not residue_references:
        return ProtonationPrediction()

    titrate_only = [f"{reference.chain}:{reference.seqid}" for reference in residue_references]
    with TemporaryDirectory(prefix="simple_propka_input_") as temp_dir:
        propka_input_path = Path(temp_dir) / target_path.name
        warnings = _write_propka_input_structure(structure, propka_input_path, metal_selection_config)
        pka_path = _run_propka(propka_input_path, ph, titrate_only)
        entries = parse_propka_output(pka_path)
    if not entries:
        coordination_candidates: list[ProtonationDisplayCandidate] = []
        direct_observations = direct_metal_coordination_observations(
            structure,
            metal_selection_config,
            excluded_residue_locators=excluded,
        )
        for observation in direct_observations:
            target_residue_name = observation.recommended_target or observation.residue_name
            selectable = target_residue_name != observation.residue_name
            change = None
            if selectable:
                change = ProtonationChange(
                    chain=observation.chain,
                    seqid=observation.seqid,
                    original_residue_name=observation.residue_name,
                    target_residue_name=target_residue_name,
                    predicted_pka=float("nan"),
                    metal_near=True,
                )
            coordination_candidates.append(
                ProtonationDisplayCandidate(
                    chain=observation.chain,
                    seqid=observation.seqid,
                    original_residue_name=observation.residue_name,
                    target_residue_name=target_residue_name,
                    predicted_pka=None,
                    metal_near=True,
                    reason=_coordination_reason(observation, selectable=selectable),
                    selectable=selectable,
                    change=change,
                )
            )
        ordered_changes = [candidate.change for candidate in coordination_candidates if candidate.change is not None]
        return ProtonationPrediction(
            changes=ordered_changes,
            warnings=warnings,
            metal_coordination_candidates=coordination_candidates,
            propka_candidates=[],
        )

    metal_neighbors = metal_neighbor_residue_locators(structure, metal_selection_config)
    direct_observations = direct_metal_coordination_observations(
        structure,
        metal_selection_config,
        excluded_residue_locators=excluded,
    )
    by_number: dict[tuple[str, int], list[_ResidueReference]] = {}
    predicted_pka_by_locator: dict[tuple[str, str], float] = {}
    for reference in residue_references:
        key = (reference.chain, reference.residue_number)
        by_number.setdefault(key, []).append(reference)

    change_by_locator: dict[tuple[str, str], ProtonationChange] = {}
    for entry in entries:
        matches = by_number.get((entry.chain, entry.residue_number), [])
        if not matches:
            continue
        reference = _match_reference(entry, matches)
        predicted_pka_by_locator[reference.locator] = entry.predicted_pka
        target_residue_name = _target_residue_name(reference.residue_name, entry.predicted_pka, ph)
        if target_residue_name == reference.residue_name:
            continue
        change_by_locator[reference.locator] = ProtonationChange(
            chain=reference.chain,
            seqid=reference.seqid,
            original_residue_name=reference.residue_name,
            target_residue_name=target_residue_name,
            predicted_pka=entry.predicted_pka,
            metal_near=reference.locator in metal_neighbors,
        )

    coordination_candidates: list[ProtonationDisplayCandidate] = []
    direct_locators: set[tuple[str, str]] = set()
    for observation in direct_observations:
        direct_locators.add(observation.locator)
        target_residue_name = observation.recommended_target or observation.residue_name
        selectable = bool(observation.recommended_target) and target_residue_name != observation.residue_name
        change = None
        if selectable:
            change = ProtonationChange(
                chain=observation.chain,
                seqid=observation.seqid,
                original_residue_name=observation.residue_name,
                target_residue_name=target_residue_name,
                predicted_pka=predicted_pka_by_locator.get(observation.locator, float("nan")),
                metal_near=True,
            )
        coordination_candidates.append(
            ProtonationDisplayCandidate(
                chain=observation.chain,
                seqid=observation.seqid,
                original_residue_name=observation.residue_name,
                target_residue_name=target_residue_name,
                predicted_pka=predicted_pka_by_locator.get(observation.locator),
                metal_near=True,
                reason=_coordination_reason(observation, selectable=selectable),
                selectable=selectable,
                change=change,
            )
        )

    propka_candidates: list[ProtonationDisplayCandidate] = []
    ordered_changes: list[ProtonationChange] = []
    for candidate in coordination_candidates:
        if candidate.change is not None:
            ordered_changes.append(candidate.change)
    for reference in residue_references:
        change = change_by_locator.get(reference.locator)
        if change is None or reference.locator in direct_locators:
            continue
        propka_candidate = ProtonationDisplayCandidate(
            chain=change.chain,
            seqid=change.seqid,
            original_residue_name=change.original_residue_name,
            target_residue_name=change.target_residue_name,
            predicted_pka=change.predicted_pka,
            metal_near=change.metal_near,
            reason=_propka_reason(change, ph=ph),
            selectable=True,
            change=change,
        )
        propka_candidates.append(propka_candidate)
        ordered_changes.append(change)
    return ProtonationPrediction(
        changes=ordered_changes,
        warnings=warnings,
        metal_coordination_candidates=coordination_candidates,
        propka_candidates=propka_candidates,
    )


def predict_protonation_changes(
    structure_path: str | Path,
    prepare_config: PrepareConfig,
    *,
    ph: float,
    structure_is_prepared: bool = False,
    excluded_residue_locators: set[tuple[str, str]] | None = None,
) -> list[ProtonationChange]:
    return predict_protonation_prediction(
        structure_path,
        prepare_config,
        ph=ph,
        structure_is_prepared=structure_is_prepared,
        excluded_residue_locators=excluded_residue_locators,
    ).changes


def apply_protonation_changes_to_structure(
    structure: gemmi.Structure,
    changes: list[ProtonationChange],
) -> list[ProtonationChange]:
    _remove_extra_models(structure)
    residues_by_locator: dict[tuple[str, str], gemmi.Residue] = {}
    for chain in structure[0]:
        for residue in chain:
            residues_by_locator[residue_locator(chain.name, residue)] = residue

    applied: list[ProtonationChange] = []
    for change in changes:
        locator = (change.chain.strip(), _normalize_seqid(change.seqid))
        residue = residues_by_locator.get(locator)
        if residue is None:
            raise ValueError(
                "Protonation change could not be applied because the target residue "
                f"{change.chain}:{change.seqid} was not found."
            )
        current_name = residue.name.strip().upper()
        if current_name != change.original_residue_name:
            raise ValueError(
                "Protonation change could not be applied because the target residue "
                f"{change.chain}:{change.seqid} is {current_name}, expected {change.original_residue_name}."
            )
        residue.name = change.target_residue_name
        applied.append(change)
    return applied
