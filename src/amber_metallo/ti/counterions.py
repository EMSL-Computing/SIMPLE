from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import shutil

from amber_metallo.execution import run_command
from amber_metallo.reporting import print_notice, write_json
from amber_metallo.ti.config import TIProtocolConfig
from amber_metallo.ti.topology import inspect_prmtop_charge_state, restore_solute_charges_and_c4


_WATER_SOURCES = {
    "spce": "leaprc.water.spce",
    "spceb": "leaprc.water.spceb",
    "tip3p": "leaprc.water.tip3p",
    "opc": "leaprc.water.opc",
    "opc3": "leaprc.water.opc3",
    "opc3pol": "leaprc.water.opc3pol",
    "tip4pew": "leaprc.water.tip4pew",
    "tip4pd": "leaprc.water.tip4pd",
    "tip5p": "leaprc.water.tip5p",
    "fb3": "leaprc.water.fb3",
    "fb4": "leaprc.water.fb4",
}
_PARMED_WATER_MODELS = {
    "spce": "SPCE",
    "spceb": "SPCE",
    "tip3p": "TIP3P",
    "opc": "OPC",
    "opc3": "OPC3",
    "opc3pol": "OPC3POL",
    "tip4pew": "TIP4PEW",
    "tip4pd": "TIP4PD",
    "tip5p": "TIP5P",
}
_AMINO_ACIDS = {
    "ALA", "ARG", "ASH", "ASN", "ASP", "CYS", "CYM", "CYX", "GLH", "GLN", "GLU", "GLY",
    "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYN", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "NALA", "CALA", "NGLY", "CGLY",
}
_WATER_LABELS = {"WAT", "HOH", "OPC", "SPC", "TP3", "TP4", "TP5", "TIP3", "TIP4", "TIP5"}
_PERIODIC_BOX_VALUE_COUNT = 6


@dataclass(slots=True)
class CounterionPlan:
    enabled: bool
    status: str
    topology_path: str
    start_coord_path: str
    counterion_atom_indices: list[int]
    counterion_mask: str | None
    metal_charge: float
    counterion_charge: float
    alchemical_group_charge: float
    initial_system_charge: float
    endpoint_system_charge: float
    added_counterion_count: int = 0
    added_spectator_count: int = 0
    requires_preparation: bool = False
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _pdb_coordinates(path: Path) -> dict[int, tuple[float, float, float]]:
    coordinates: dict[int, tuple[float, float, float]] = {}
    index = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        index += 1
        try:
            coordinates[index] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
    return coordinates


def _valid_periodic_box(values: tuple[float, ...] | list[float]) -> bool:
    if len(values) != _PERIODIC_BOX_VALUE_COUNT:
        return False
    a, b, c, alpha, beta, gamma = values
    return (
        all(math.isfinite(value) for value in values)
        and all(1.0 <= value <= 10000.0 for value in (a, b, c))
        and all(30.0 <= value <= 150.0 for value in (alpha, beta, gamma))
    )


def _formatted_restart_box(path: Path) -> tuple[float, float, float, float, float, float] | None:
    """Read the six unit-cell values from a formatted Amber restart/inpcrd."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    if raw.startswith((b"CDF", b"\x89HDF")):
        return None
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    if len(lines) < 3:
        return None
    try:
        natom = int(lines[1].split()[0])
    except (IndexError, ValueError):
        return None
    values: list[float] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        try:
            parsed_line = [float(token.replace("D", "E").replace("d", "e")) for token in line.split()]
            values.extend(parsed_line)
            continue
        except ValueError:
            pass
        try:
            for start in range(0, len(line), 12):
                token = line[start : start + 12].strip()
                if token:
                    values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            return None
    coordinate_count = 3 * natom
    trailing_count = len(values) - coordinate_count
    if trailing_count not in {_PERIODIC_BOX_VALUE_COUNT, coordinate_count + _PERIODIC_BOX_VALUE_COUNT}:
        return None
    candidate = tuple(values[-_PERIODIC_BOX_VALUE_COUNT:])
    return candidate if _valid_periodic_box(candidate) else None


def _pdb_periodic_box(path: Path) -> tuple[float, float, float, float, float, float] | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("CRYST1"):
            continue
        try:
            candidate = (
                float(line[6:15]),
                float(line[15:24]),
                float(line[24:33]),
                float(line[33:40]),
                float(line[40:47]),
                float(line[47:54]),
            )
        except ValueError:
            return None
        return candidate if _valid_periodic_box(candidate) else None
    return None


def _box_cpptraj_args(box: tuple[float, float, float, float, float, float]) -> str:
    a, b, c, alpha, beta, gamma = box
    return (
        f"x {a:.7f} y {b:.7f} z {c:.7f} "
        f"alpha {alpha:.7f} beta {beta:.7f} gamma {gamma:.7f}"
    )


def _resolve_periodic_box(
    *,
    source_prmtop: Path,
    source_coord: Path,
    source_pdb: Path,
    cpptraj: str | None,
    output_dir: Path,
) -> tuple[float, float, float, float, float, float]:
    box = _formatted_restart_box(source_coord) or _pdb_periodic_box(source_pdb)
    if box is not None:
        return box

    if cpptraj is None:
        raise RuntimeError(
            "The selected TI restart is not a formatted Amber restart and cpptraj is unavailable, so its "
            "periodic-box parameters cannot be recovered."
        )

    # A production restart may be NetCDF. Let cpptraj convert one frame to a
    # formatted restart, then read the exact instantaneous cell from that file.
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_restart = output_dir / "source_box_probe.rst7"
    probe_input = output_dir / "source_box_probe.cpptraj.in"
    probe_input.write_text(
        (
            f"parm {source_prmtop.resolve().as_posix()}\n"
            f"trajin {source_coord.resolve().as_posix()} 1 1\n"
            f"trajout {probe_restart.resolve().as_posix()} restart\n"
            "run\n"
        ),
        encoding="utf-8",
    )
    try:
        run_command(
            [cpptraj, "-i", str(probe_input)],
            cwd=output_dir,
            log_path=output_dir / "source_box_probe.log",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not recover periodic-box parameters from the selected TI restart. "
            "The source restart/trajectory must contain unit-cell lengths and angles before periodic TI can run."
        ) from exc
    box = _formatted_restart_box(probe_restart)
    if box is None:
        raise RuntimeError(
            "The selected TI coordinates contain no periodic-box parameters. Refusing to create a periodic "
            "counterion topology with a guessed box; regenerate the snapshot from an unstripped periodic trajectory."
        )
    return box


def _minimum_distance(
    atom_index: int,
    *,
    coordinates: dict[int, tuple[float, float, float]],
    solute_indices: set[int],
) -> float:
    point = coordinates.get(atom_index)
    if point is None:
        return -1.0
    distances = [
        math.dist(point, coordinates[index])
        for index in solute_indices
        if index != atom_index and index in coordinates
    ]
    return min(distances) if distances else float("inf")


def _candidate_indices(
    *,
    prmtop_path: Path,
    pdb_path: Path,
    metal_atom_indices: list[int],
    required_count: int,
    minimum_distance: float,
) -> tuple[list[int], float, float]:
    state = inspect_prmtop_charge_state(prmtop_path)
    coordinates = _pdb_coordinates(pdb_path)
    solute_indices = set(metal_atom_indices)
    protein_present = any(atom.residue_label.upper() in _AMINO_ACIDS for atom in state.atoms)
    if protein_present:
        solute_indices.update(
            atom.atom_index
            for atom in state.atoms
            if atom.residue_atom_count > 1 and atom.residue_label.upper() not in _WATER_LABELS
        )
    candidates = state.monovalent_atoms(sign=-1)
    ranked = sorted(
        (
            (_minimum_distance(atom.atom_index, coordinates=coordinates, solute_indices=solute_indices), atom)
            for atom in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    selected = [atom.atom_index for distance, atom in ranked if distance >= minimum_distance][:required_count]
    selected_charge = math.fsum(
        atom.charge for atom in candidates if atom.atom_index in set(selected)
    )
    return selected, selected_charge, state.net_charge


def _binary(amber_env, name: str) -> str | None:
    status = amber_env.binaries.get(name)
    if status is not None and status.path is not None:
        return str(status.path)
    return shutil.which(name)


def _rebuild_with_salt_pairs(
    *,
    source_prmtop: Path,
    source_pdb: Path,
    output_dir: Path,
    count: int,
    water_model: str,
    ion_frcmods: list[str],
    amber_env,
    config: TIProtocolConfig,
    metal_atom_indices: list[int],
    periodic_box: tuple[float, float, float, float, float, float],
) -> tuple[Path, Path, list[int]]:
    tleap = _binary(amber_env, "tleap")
    cpptraj = _binary(amber_env, "cpptraj")
    if tleap is None or cpptraj is None:
        raise RuntimeError("Adding TI counterions requires both tleap and cpptraj on the preparation host.")
    water_source = _WATER_SOURCES.get(water_model.lower())
    if water_source is None:
        raise ValueError(f"Automatic TI counterion addition does not recognize water model '{water_model}'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_prmtop = output_dir / "counterion_rebuilt_raw.prmtop"
    raw_coord = output_dir / "counterion_rebuilt_raw.inpcrd"
    raw_pdb = output_dir / "counterion_rebuilt_raw.pdb"
    box_lengths = " ".join(f"{value:.7f}" for value in periodic_box[:3])
    lines = ["source leaprc.protein.ff19SB", "source leaprc.gaff2", f"source {water_source}"]
    lines.extend(f"loadAmberParams {Path(path).resolve().as_posix()}" for path in ion_frcmods)
    lines.extend(
        [
            f"system = loadPdb {source_pdb.resolve().as_posix()}",
            f"set system box {{{box_lengths}}}",
            f"addIonsRand system Cl- {count}",
            f"addIonsRand system Na+ {count}",
            "check system",
            "charge system",
            f"savePDB system {raw_pdb.resolve().as_posix()}",
            f"saveAmberParm system {raw_prmtop.resolve().as_posix()} {raw_coord.resolve().as_posix()}",
            "quit",
        ]
    )
    tleap_input = output_dir / "add_counterions.tleap.in"
    tleap_input.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        run_command([tleap, "-f", str(tleap_input)], cwd=output_dir, log_path=output_dir / "add_counterions.tleap.log")
    except Exception as exc:
        raise RuntimeError(
            "Automatic counterion insertion could not rebuild this snapshot with standard protein/water libraries. "
            "For a DES or custom-residue system, provide pre-equilibrated distant monovalent counterions in the "
            "input topology, or disable co-alchemical counterions."
        ) from exc
    rebuilt_for_restore = raw_prmtop
    source_has_c4 = "%FLAG LENNARD_JONES_CCOEF" in source_prmtop.read_text(
        encoding="utf-8", errors="ignore"
    )
    if source_has_c4:
        parmed = _binary(amber_env, "parmed")
        if parmed is None:
            raise RuntimeError(
                "The source topology uses 12-6-4 C4 terms. Adding new counterions therefore requires ParmEd "
                "add12_6_4 on the preparation host so the new Cl-/Na+ atom types receive matching C4 terms."
            )
        raw_state = inspect_prmtop_charge_state(raw_prmtop)
        ion_indices = [
            atom.atom_index
            for atom in raw_state.atoms
            if atom.residue_atom_count == 1 and abs(abs(atom.charge) - 1.0) <= 0.15
        ]
        c4_mask_indices = sorted({*metal_atom_indices, *ion_indices})
        c4_mask = "@" + ",".join(str(index) for index in c4_mask_indices)
        parmed_water_model = _PARMED_WATER_MODELS.get(water_model.lower())
        if parmed_water_model is None:
            raise ValueError(f"ParmEd 12-6-4 setup does not recognize water model '{water_model}'.")
        rebuilt_c4 = output_dir / "counterion_rebuilt_c4.prmtop"
        parmed_input = output_dir / "counterion_add12_6_4.parmed.in"
        parmed_input.write_text(
            f"add12_6_4 {c4_mask} watermodel {parmed_water_model}\n"
            f"outparm {rebuilt_c4.resolve().as_posix()}\n"
            "quit\n",
            encoding="utf-8",
        )
        run_command(
            [parmed, "-p", str(raw_prmtop), "-i", str(parmed_input)],
            cwd=output_dir,
            log_path=output_dir / "counterion_add12_6_4.parmed.log",
        )
        if not rebuilt_c4.exists():
            raise RuntimeError("ParmEd did not create the counterion 12-6-4 topology.")
        rebuilt_for_restore = rebuilt_c4

    prebox_prmtop = output_dir / "counterion_neutral_prebox.prmtop"
    restore_solute_charges_and_c4(
        source_prmtop=source_prmtop,
        rebuilt_prmtop=rebuilt_for_restore,
        output_prmtop=prebox_prmtop,
    )
    final_state = inspect_prmtop_charge_state(prebox_prmtop)
    anions = final_state.monovalent_atoms(sign=-1)
    selected = [atom.atom_index for atom in anions[-count:]]
    if len(selected) != count:
        raise RuntimeError("The counterion rebuild did not create the requested number of chloride ions.")
    mask = "@" + ",".join(str(index) for index in selected)
    final_prmtop = output_dir / "counterion_neutral.prmtop"
    randomized_coord = output_dir / "counterion_neutral.rst7"
    solute_mask = "!(:WAT,HOH,OPC,SPC,TP3,TP4,TP5,TIP3,TIP4,TIP5,Cl-,Na+,K+,Br-,F-,I-)"
    box_args = _box_cpptraj_args(periodic_box)
    cpptraj_input = output_dir / "place_counterions.cpptraj.in"
    cpptraj_input.write_text(
        "\n".join(
            [
                f"parm {prebox_prmtop.resolve().as_posix()}",
                f"parmbox {box_args}",
                f"trajin {raw_coord.resolve().as_posix()} 1 1",
                f"box {box_args}",
                f"randomizeions {mask} around {solute_mask} by {config.counterion_min_solute_distance_angstrom:.3f} "
                f"overlap {config.counterion_min_separation_angstrom:.3f}",
                f"trajout {randomized_coord.resolve().as_posix()} restart",
                "run",
                f"parmwrite out {final_prmtop.resolve().as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_command(
        [cpptraj, "-i", str(cpptraj_input)],
        cwd=output_dir,
        log_path=output_dir / "place_counterions.cpptraj.log",
    )
    if not randomized_coord.exists() or not final_prmtop.exists():
        raise RuntimeError("cpptraj did not create the boxed counterion topology/restart pair.")
    written_box = _formatted_restart_box(randomized_coord)
    if written_box is None or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-4)
        for actual, expected in zip(written_box, periodic_box, strict=True)
    ):
        raise RuntimeError(
            "Counterion placement did not preserve the source periodic-box lengths and angles; refusing to start TI."
        )
    return final_prmtop, randomized_coord, selected


def prepare_charge_compensating_counterions(
    *,
    source_prmtop: str | Path,
    source_pdb: str | Path,
    source_coord: str | Path,
    metal_atom_indices: list[int],
    formal_charges: list[int],
    output_dir: str | Path,
    water_model: str,
    ion_frcmods: list[str],
    amber_env,
    config: TIProtocolConfig,
    dry_run: bool,
) -> CounterionPlan:
    source_prmtop = Path(source_prmtop)
    source_pdb = Path(source_pdb)
    source_coord = Path(source_coord)
    output_dir = Path(output_dir)
    configured_metal_charge = float(math.fsum(formal_charges))
    try:
        charge_state = inspect_prmtop_charge_state(source_prmtop)
        metal_charge = math.fsum(charge_state.atoms[index - 1].charge for index in metal_atom_indices)
    except Exception:
        if not dry_run:
            raise
        metal_charge = configured_metal_charge
    required_count = int(round(metal_charge))
    if abs(metal_charge - required_count) > 0.05:
        raise ValueError(
            "Co-alchemical full counterion decoupling requires the selected metal atom charge change to be "
            f"integral, but the prmtop metal mask carries {metal_charge:+.5f} e. The configured oxidation-state "
            f"sum is {configured_metal_charge:+.0f} e. Use the legacy charge-changing path, or define an "
            "alchemical region whose actual prmtop charge is integral; SIMPLE will not silently alter RESP charges."
        )
    if required_count <= 0:
        raise ValueError("Co-alchemical counterions currently require a positive integral metal charge change.")
    try:
        selected, selected_charge, initial_charge = _candidate_indices(
            prmtop_path=source_prmtop,
            pdb_path=source_pdb,
            metal_atom_indices=metal_atom_indices,
            required_count=required_count,
            minimum_distance=config.counterion_min_solute_distance_angstrom,
        )
    except Exception:
        if not dry_run:
            raise
        selected, selected_charge, initial_charge = [], -float(required_count), 0.0
    if abs(initial_charge) > 0.05 and not dry_run:
        raise ValueError(
            f"Co-alchemical TI requires a neutral starting topology, but its net charge is {initial_charge:+.4f}. "
            "Neutralize and equilibrate the system first, or choose the legacy charge-changing option."
        )
    periodic_box: tuple[float, float, float, float, float, float] | None = None
    if not dry_run:
        periodic_box = _resolve_periodic_box(
            source_prmtop=source_prmtop,
            source_coord=source_coord,
            source_pdb=source_pdb,
            cpptraj=_binary(amber_env, "cpptraj"),
            output_dir=output_dir,
        )
    added = 0
    topology_path = source_prmtop
    start_coord_path = source_coord
    status = "reused_existing"
    requires_preparation = False
    message = None
    if len(selected) < required_count:
        added = required_count
        message = (
            f"No suitable set of {required_count} distant monovalent counterions was found. Adding {required_count} "
            "Cl- co-alchemical counterions plus the same number of Na+ spectator ions, then running required "
            "minimization/equilibration before TI."
        )
        print_notice("Adding TI Counterions", message, border_style="yellow")
        status = "added_and_relocated"
        requires_preparation = True
        if dry_run:
            # The added atom indices do not exist until tleap replaces waters.
            # Keep masks topology-valid in dry-run output and record the planned
            # charge/count in the manifest instead of inventing unusable indices.
            selected = []
            selected_charge = -float(required_count)
        else:
            if periodic_box is None:  # Defensive: non-dry runs resolve this above.
                raise RuntimeError("Periodic-box parameters were not resolved before counterion insertion.")
            topology_path, start_coord_path, selected = _rebuild_with_salt_pairs(
                source_prmtop=source_prmtop,
                source_pdb=source_pdb,
                output_dir=output_dir,
                count=required_count,
                water_model=water_model,
                ion_frcmods=ion_frcmods,
                amber_env=amber_env,
                config=config,
                metal_atom_indices=metal_atom_indices,
                periodic_box=periodic_box,
            )
            state = inspect_prmtop_charge_state(topology_path)
            selected_charge = math.fsum(state.atoms[index - 1].charge for index in selected)
            initial_charge = state.net_charge
    group_charge = metal_charge + selected_charge
    endpoint_charge = initial_charge - group_charge
    if not dry_run and (abs(initial_charge) > 0.05 or abs(group_charge) > 0.05 or abs(endpoint_charge) > 0.05):
        raise RuntimeError(
            "Counterion neutrality validation failed: "
            f"initial={initial_charge:+.5f}, alchemical_group={group_charge:+.5f}, endpoint={endpoint_charge:+.5f}."
        )
    mask = None if not selected else "@" + ",".join(str(index) for index in selected)
    plan = CounterionPlan(
        enabled=True,
        status=status,
        topology_path=str(topology_path),
        start_coord_path=str(start_coord_path),
        counterion_atom_indices=selected,
        counterion_mask=mask,
        metal_charge=metal_charge,
        counterion_charge=selected_charge,
        alchemical_group_charge=group_charge,
        initial_system_charge=initial_charge,
        endpoint_system_charge=endpoint_charge,
        added_counterion_count=added,
        added_spectator_count=added,
        requires_preparation=requires_preparation,
        message=message,
    )
    write_json(output_dir / "counterion_plan.json", plan.to_dict())
    return plan
