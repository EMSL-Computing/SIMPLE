from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TOPOLOGY_EXTS = {".prmtop", ".parm7", ".top", ".psf", ".pdb"}
TRAJECTORY_EXTS = {".nc", ".mdcrd", ".crd", ".dcd", ".xtc", ".trr", ".trj"}
EXTENSIONLESS_TRAJECTORY_NAMES = {"mdcrd", "traj", "trajectory"}
TRAJECTORY_TYPE_ORDER = (".nc", ".mdcrd", "mdcrd", ".crd", ".dcd", ".xtc", ".trr", ".trj")
SCAN_SKIP_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "analysis",
    "Analysis",
    "node_modules",
    "site-packages",
}

RARE_EARTH_RESNAMES = {
    "SC",
    "Y",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "LA3",
    "CE3",
    "PR3",
    "ND3",
    "PM3",
    "SM3",
    "EU3",
    "GD3",
    "TB3",
    "DY3",
    "HO3",
    "ER3",
    "TM3",
    "YB3",
    "LU3",
}
SUPPORTED_METAL_RESNAMES = RARE_EARTH_RESNAMES | {
    "LI",
    "NA",
    "K",
    "RB",
    "CS",
    "MG",
    "CA",
    "SR",
    "BA",
    "ZN",
    "CU",
    "FE",
    "MN",
    "CO",
    "NI",
    "CD",
    "HG",
    "PB",
}
WATER_RESNAMES = {"WAT", "HOH", "H2O", "SPC", "SPCE", "TIP3", "TIP3P", "TIP4P", "OPC", "SOL"}
DES_COMPONENT_RESNAMES = {
    "des_n8888_bromide": ("DES [N8888][Br]", {"N88", "BR"}),
    "des_hexanoic_acid": ("DES hexanoic acid", {"HAH"}),
    "des_choline_chloride": ("DES choline chloride", {"CH1", "CL"}),
    "des_ethylene_glycol": ("DES ethylene glycol", {"EG1"}),
}
DES_RESNAMES = set().union(*(tokens for _, tokens in DES_COMPONENT_RESNAMES.values()))


class TrajectoryAnalysisDependencyError(RuntimeError):
    """Raised when optional trajectory-analysis dependencies are unavailable."""


class SelectionValidationError(ValueError):
    """Raised when an atom selection is invalid or empty for one or more cases."""


@dataclass(frozen=True)
class TrajectoryCase:
    label: str
    root: Path
    topology_path: Path
    trajectory_path: Path
    reference_path: Path | None = None


@dataclass(frozen=True)
class SimulationDiscovery:
    root: Path
    display_name: str
    topology_path: Path
    trajectory_path: Path
    trajectory_candidates: tuple[Path, ...] = ()

    def to_case(self) -> TrajectoryCase:
        return TrajectoryCase(
            label=self.display_name,
            root=self.root,
            topology_path=self.topology_path,
            trajectory_path=self.trajectory_path,
        )


@dataclass(frozen=True)
class MaskOption:
    key: str
    label: str
    selection: str
    category: str
    atom_count: int | None = None
    counts_by_case: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseProfile:
    label: str
    family: str
    has_protein: bool
    has_ree: bool
    has_metals: bool
    has_water: bool
    has_des: bool
    des_components: frozenset[str] = frozenset()
    ree_labels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TrajectoryAnalysisRequest:
    analysis_type: str
    target_selection: str | None = None
    alignment_selection: str | None = None
    selection_a: str | None = None
    selection_b: str | None = None
    nbins: int = 75
    rdf_range: tuple[float, float] = (0.0, 12.0)
    label: str | None = None


@dataclass(frozen=True)
class TrajectoryFrameSelection:
    stride: int = 1
    last_ns: float | None = None
    last_frames: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stride", int(self.stride))
        if self.last_ns is not None:
            object.__setattr__(self, "last_ns", float(self.last_ns))
        if self.last_frames is not None:
            object.__setattr__(self, "last_frames", int(self.last_frames))
        if self.stride < 1:
            raise ValueError("Frame stride must be a positive integer")
        if self.last_ns is not None and self.last_ns <= 0:
            raise ValueError("Last-ns window must be positive")
        if self.last_frames is not None and self.last_frames < 1:
            raise ValueError("Last-frame window must be a positive integer")
        if self.last_ns is not None and self.last_frames is not None:
            raise ValueError("Choose either last_ns or last_frames, not both")


@dataclass(frozen=True)
class AnalysisOutput:
    case_label: str
    analysis_type: str
    csv_path: Path
    png_path: Path
    summary_txt_path: Path
    summary_json_path: Path
    stats: dict[str, float | int | str]
    x_column: str | None = None
    y_column: str | None = None


@dataclass(frozen=True)
class TrajectoryAnalysisRunResult:
    output_dir: Path
    outputs: tuple[AnalysisOutput, ...]
    overlay_paths: tuple[Path, ...] = ()
    combined_summary_csv_path: Path | None = None


@dataclass(frozen=True)
class _ComputedAnalysis:
    analysis_type: str
    slug: str
    rows: tuple[dict[str, object], ...]
    x_column: str | None
    y_column: str | None
    x_label: str
    y_label: str
    title: str
    stats: dict[str, float | int | str]


@dataclass(frozen=True)
class _ResolvedFrameSelection:
    start: int
    stop: int
    step: int
    total_frames: int
    analyzed_frames: int
    window_label: str


def default_output_root(base_dir: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (base_dir or Path.cwd()) / "Analysis" / f"trajectory_analysis_{timestamp}"


def dependency_install_hint() -> str:
    return (
        "Trajectory analysis requires MDAnalysis and matplotlib.\n"
        "Install or refresh dependencies with:\n"
        "  pip install .\n"
        "or\n"
        "  pip install -r requirements.txt\n"
        "For conda environments, update from environment.yml."
    )


def _import_mdanalysis() -> Any:
    try:
        import MDAnalysis as mda
    except ModuleNotFoundError as exc:
        raise TrajectoryAnalysisDependencyError(dependency_install_hint()) from exc
    return mda


def _import_mdanalysis_analysis() -> tuple[Any, Any, Any, Any]:
    try:
        from MDAnalysis.analysis import align, rms
        from MDAnalysis.analysis.distances import distance_array
        from MDAnalysis.analysis.rdf import InterRDF
    except ModuleNotFoundError as exc:
        raise TrajectoryAnalysisDependencyError(dependency_install_hint()) from exc
    return align, rms, distance_array, InterRDF


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise TrajectoryAnalysisDependencyError(dependency_install_hint()) from exc
    return plt


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "case"


def _path_type_key(path: Path) -> tuple[int, str]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    key = suffix if suffix else name
    try:
        rank = TRAJECTORY_TYPE_ORDER.index(key)
    except ValueError:
        rank = len(TRAJECTORY_TYPE_ORDER)
    return rank, path.name.lower()


def _is_trajectory_file(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix in TRAJECTORY_EXTS:
        return True
    return not suffix and path.name.lower() in EXTENSIONLESS_TRAJECTORY_NAMES


def _iter_scan_files(root: Path, *, max_depth: int = 4) -> Iterable[Path]:
    root = root.resolve()
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if child.name in SCAN_SKIP_NAMES:
                    continue
                stack.append((child, depth + 1))
            elif child.is_file():
                yield child


def _bundle_root_from_topology(topology: Path, search_root: Path) -> Path:
    parent = topology.parent
    if parent.name in {"02_system", "system", "topology"} and parent.parent != parent:
        return parent.parent
    try:
        topology.relative_to(search_root)
    except ValueError:
        return parent
    return parent


def _trajectory_score(topology: Path, candidate: Path, bundle_root: Path) -> tuple[int, int, tuple[int, str]]:
    try:
        distance = len(candidate.relative_to(bundle_root).parts)
    except ValueError:
        distance = 99
    name = candidate.name.lower()
    stem_bonus = 0
    if any(token in name for token in ("prod", "production", "traj", "md", "run")):
        stem_bonus -= 1
    if topology.stem.lower() in name:
        stem_bonus -= 1
    return distance, stem_bonus, _path_type_key(candidate)


def discover_simulation_cases(search_root: Path, *, max_depth: int = 4, limit: int = 50) -> list[SimulationDiscovery]:
    root = search_root.expanduser().resolve()
    if not root.exists():
        return []
    files = list(_iter_scan_files(root, max_depth=max_depth))
    topologies = [path for path in files if path.suffix.lower() in TOPOLOGY_EXTS]
    trajectories = [path for path in files if _is_trajectory_file(path)]
    discoveries: list[SimulationDiscovery] = []
    seen_roots: set[Path] = set()
    for topology in sorted(topologies, key=lambda item: (len(item.parts), item.name.lower())):
        bundle_root = _bundle_root_from_topology(topology, root).resolve()
        if bundle_root in seen_roots:
            continue
        nearby = [
            traj
            for traj in trajectories
            if traj != topology and (bundle_root in [traj.parent, *traj.parents] or traj.parent == topology.parent)
        ]
        if not nearby:
            continue
        nearby = sorted(nearby, key=lambda item: _trajectory_score(topology, item, bundle_root))
        discoveries.append(
            SimulationDiscovery(
                root=bundle_root,
                display_name=bundle_root.name or topology.stem,
                topology_path=topology,
                trajectory_path=nearby[0],
                trajectory_candidates=tuple(nearby),
            )
        )
        seen_roots.add(bundle_root)
        if len(discoveries) >= limit:
            break
    return discoveries


def load_universe(case: TrajectoryCase) -> Any:
    mda = _import_mdanalysis()
    topology = str(case.topology_path)
    trajectory = str(case.trajectory_path)
    try:
        universe = mda.Universe(topology, trajectory)
    except Exception as first_exc:
        try:
            universe = mda.Universe(topology, trajectory, format="TRJ")
        except Exception as second_exc:
            raise RuntimeError(
                f"Failed to load trajectory for {case.label}: {second_exc}. "
                f"Initial MDAnalysis load error: {first_exc}"
            ) from second_exc
    if len(universe.atoms) == 0:
        raise ValueError(f"{case.label}: topology contains no atoms")
    if len(universe.trajectory) == 0:
        raise ValueError(f"{case.label}: trajectory contains no frames")
    return universe


def _selection_count(universe: Any, selection: str, *, strict: bool = False) -> int:
    try:
        return len(universe.select_atoms(selection))
    except Exception as exc:
        if strict:
            raise SelectionValidationError(f"Invalid MDAnalysis selection '{selection}': {exc}") from exc
        return 0


def _unique_atom_values(universe: Any, attr_name: str) -> set[str]:
    try:
        values = getattr(universe.atoms, attr_name)
    except Exception:
        return set()
    return {str(value) for value in values if str(value)}


def _present_tokens(universe: Any, candidates: set[str], *, include_atom_names: bool = True) -> set[str]:
    resnames = _unique_atom_values(universe, "resnames")
    names = _unique_atom_values(universe, "names") if include_atom_names else set()
    upper_resnames = {item.upper(): item for item in resnames}
    upper_names = {item.upper(): item for item in names}
    present: set[str] = set()
    for candidate in candidates:
        if candidate in upper_resnames:
            present.add(upper_resnames[candidate])
        if include_atom_names and candidate in upper_names:
            present.add(upper_names[candidate])
    return present


def _token_selection(tokens: Iterable[str], *, include_atom_names: bool = True) -> str:
    token_list = sorted({token for token in tokens if token})
    if not token_list:
        return "resname __NONE__"
    res_sel = "resname " + " ".join(token_list)
    if include_atom_names:
        name_sel = "name " + " ".join(token_list)
        return f"({res_sel}) or ({name_sel})"
    return res_sel


def _des_component_selection(component_key: str, universe: Any | None = None) -> str:
    _, tokens = DES_COMPONENT_RESNAMES[component_key]
    if universe is not None:
        present = _present_tokens(universe, tokens, include_atom_names=False)
        if present:
            return _token_selection(present, include_atom_names=False)
    return _token_selection(tokens, include_atom_names=False)


def ree_selection(universe: Any | None = None) -> str:
    if universe is None:
        return _token_selection(RARE_EARTH_RESNAMES)
    present = _present_tokens(universe, RARE_EARTH_RESNAMES)
    return _token_selection(present or RARE_EARTH_RESNAMES)


def metal_selection(universe: Any | None = None) -> str:
    if universe is None:
        return _token_selection(SUPPORTED_METAL_RESNAMES)
    present = _present_tokens(universe, SUPPORTED_METAL_RESNAMES)
    return _token_selection(present or SUPPORTED_METAL_RESNAMES)


def water_selection(universe: Any | None = None) -> str:
    if universe is None:
        return _token_selection(WATER_RESNAMES, include_atom_names=False)
    present = _present_tokens(universe, WATER_RESNAMES, include_atom_names=False)
    return _token_selection(present or WATER_RESNAMES, include_atom_names=False)


def des_selection(universe: Any | None = None) -> str:
    if universe is None:
        return _token_selection(DES_RESNAMES, include_atom_names=False)
    present = _present_tokens(universe, DES_RESNAMES, include_atom_names=False)
    return _token_selection(present or DES_RESNAMES, include_atom_names=False)


def candidate_masks_for_universe(universe: Any) -> list[MaskOption]:
    candidates = [
        MaskOption("system", "System", "all", "general", _selection_count(universe, "all")),
        MaskOption("protein", "Protein", "protein", "protein", _selection_count(universe, "protein")),
        MaskOption(
            "protein_backbone",
            "Protein backbone",
            "protein and backbone",
            "protein",
            _selection_count(universe, "protein and backbone"),
        ),
        MaskOption(
            "protein_ca",
            "Protein CA",
            "protein and name CA",
            "protein",
            _selection_count(universe, "protein and name CA"),
        ),
    ]
    candidates.append(MaskOption("ree", "REE", ree_selection(universe), "metal", _selection_count(universe, ree_selection(universe))))
    candidates.append(
        MaskOption("metals", "Metals", metal_selection(universe), "metal", _selection_count(universe, metal_selection(universe)))
    )
    candidates.append(
        MaskOption("water", "Water/solvent", water_selection(universe), "solvent", _selection_count(universe, water_selection(universe)))
    )
    candidates.append(MaskOption("des", "DES components", des_selection(universe), "des", _selection_count(universe, des_selection(universe))))
    for key, (label, _) in DES_COMPONENT_RESNAMES.items():
        selection = _des_component_selection(key, universe)
        candidates.append(MaskOption(key, label, selection, "des", _selection_count(universe, selection)))
    return [item for item in candidates if item.key == "system" or (item.atom_count or 0) > 0]


def build_case_profile(label: str, universe: Any) -> CaseProfile:
    has_protein = _selection_count(universe, "protein") > 0
    has_ree = _selection_count(universe, ree_selection(universe)) > 0
    has_metals = _selection_count(universe, metal_selection(universe)) > 0
    has_water = _selection_count(universe, water_selection(universe)) > 0
    has_des = _selection_count(universe, des_selection(universe)) > 0
    des_components = {
        key for key in DES_COMPONENT_RESNAMES if _selection_count(universe, _des_component_selection(key, universe)) > 0
    }
    ree_labels = {token.upper() for token in _present_tokens(universe, RARE_EARTH_RESNAMES)}
    if has_protein:
        family = "protein"
    elif has_des:
        family = "des"
    elif has_water or has_metals or has_ree:
        family = "solvent"
    else:
        family = "unknown"
    return CaseProfile(
        label=label,
        family=family,
        has_protein=has_protein,
        has_ree=has_ree,
        has_metals=has_metals,
        has_water=has_water,
        has_des=has_des,
        des_components=frozenset(des_components),
        ree_labels=frozenset(ree_labels),
    )


def check_case_compatibility(profiles: Sequence[CaseProfile]) -> None:
    families = {profile.family for profile in profiles}
    if len(families) <= 1:
        return
    labels = ", ".join(f"{profile.label}={profile.family}" for profile in profiles)
    raise ValueError(f"Incompatible systems; analyze separately. Detected case families: {labels}")


def generalize_mask_options(cases: Sequence[TrajectoryCase], universes: Sequence[Any]) -> list[MaskOption]:
    if len(cases) != len(universes):
        raise ValueError("case and universe counts do not match")
    profiles = [build_case_profile(case.label, universe) for case, universe in zip(cases, universes)]
    check_case_compatibility(profiles)
    options_by_case = [candidate_masks_for_universe(universe) for universe in universes]
    by_key: dict[str, list[MaskOption]] = {}
    for options in options_by_case:
        for option in options:
            by_key.setdefault(option.key, []).append(option)

    ordered_keys = [
        "system",
        "protein",
        "protein_backbone",
        "protein_ca",
        "ree",
        "metals",
        "water",
        "des",
        *DES_COMPONENT_RESNAMES.keys(),
    ]
    generalized: list[MaskOption] = []
    for key in ordered_keys:
        per_case = by_key.get(key, [])
        if len(per_case) != len(cases):
            continue
        counts = {case.label: int(option.atom_count or 0) for case, option in zip(cases, per_case)}
        if any(count <= 0 for count in counts.values()):
            continue
        template = per_case[0]
        selection = template.selection
        label = template.label
        if key == "ree":
            selection = ree_selection()
            label = "REE"
        elif key == "metals":
            selection = metal_selection()
            label = "Metals"
        elif key == "water":
            selection = water_selection()
            label = "Water/solvent"
        elif key == "des":
            selection = des_selection()
            label = "DES components"
        elif key in DES_COMPONENT_RESNAMES:
            selection = _des_component_selection(key)
            label = DES_COMPONENT_RESNAMES[key][0]
        generalized.append(
            MaskOption(
                key=key,
                label=label,
                selection=selection,
                category=template.category,
                atom_count=min(counts.values()),
                counts_by_case=counts,
            )
        )
    generalized.append(MaskOption("custom", "Custom selection", "", "custom", None, {}))
    return generalized


def validate_selection_across_cases(
    cases: Sequence[TrajectoryCase],
    universes: Sequence[Any],
    selection: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case, universe in zip(cases, universes):
        counts[case.label] = _selection_count(universe, selection, strict=True)
    empty = {label: count for label, count in counts.items() if count == 0}
    if empty:
        detail = ", ".join(f"{label}: {count}" for label, count in counts.items())
        raise SelectionValidationError(f"Selection matched zero atoms in at least one case ({detail})")
    return counts


def describe_frame_selection(frame_selection: TrajectoryFrameSelection | None = None) -> str:
    selection = frame_selection or TrajectoryFrameSelection()
    if selection.last_ns is not None:
        window = f"last {selection.last_ns:g} ns"
    elif selection.last_frames is not None:
        window = f"last {selection.last_frames} frames"
    else:
        window = "all frames"
    return f"{window}, stride {selection.stride}"


def _frame_time(ts: Any, frame_index: int, time_step_ps: float) -> float:
    try:
        raw_time = float(ts.time)
    except Exception:
        raw_time = math.nan
    try:
        frame_number = int(ts.frame)
    except Exception:
        frame_number = frame_index
    if math.isfinite(raw_time) and (frame_number == 0 or abs(raw_time) > 1.0e-12):
        return raw_time
    return frame_number * time_step_ps


def _trajectory_dt_ps(universe: Any, time_step_ps: float) -> float:
    try:
        dt_ps = float(universe.trajectory.dt)
    except Exception:
        dt_ps = math.nan
    if math.isfinite(dt_ps) and dt_ps > 0:
        return dt_ps
    return time_step_ps


def _resolve_frame_selection(
    universe: Any,
    frame_selection: TrajectoryFrameSelection | None,
    *,
    time_step_ps: float,
) -> _ResolvedFrameSelection:
    if time_step_ps <= 0:
        raise ValueError("Time between trajectory frames must be positive")
    selection = frame_selection or TrajectoryFrameSelection()
    total_frames = len(universe.trajectory)
    if total_frames <= 0:
        raise ValueError("Trajectory contains no frames")

    start = 0
    if selection.last_frames is not None:
        start = max(0, total_frames - selection.last_frames)
    elif selection.last_ns is not None:
        window_ps = selection.last_ns * 1000.0
        frame_count = max(1, int(math.ceil(window_ps / _trajectory_dt_ps(universe, time_step_ps))))
        start = max(0, total_frames - frame_count)

    stop = total_frames
    analyzed_frames = len(range(start, stop, selection.stride))
    if analyzed_frames < 1:
        raise ValueError("Frame selection did not include any trajectory frames")
    if selection.last_ns is not None:
        window_label = f"last {selection.last_ns:g} ns"
    elif selection.last_frames is not None:
        window_label = f"last {selection.last_frames} frames"
    else:
        window_label = "all frames"
    return _ResolvedFrameSelection(
        start=start,
        stop=stop,
        step=selection.stride,
        total_frames=total_frames,
        analyzed_frames=analyzed_frames,
        window_label=window_label,
    )


def _selected_trajectory_frames(universe: Any, frame_info: _ResolvedFrameSelection) -> Any:
    return universe.trajectory[frame_info.start : frame_info.stop : frame_info.step]


def _frame_selection_stats(frame_info: _ResolvedFrameSelection) -> dict[str, float | int | str]:
    last_frame = frame_info.start + (frame_info.analyzed_frames - 1) * frame_info.step
    return {
        "trajectory_frame_window": frame_info.window_label,
        "trajectory_total_frames": frame_info.total_frames,
        "trajectory_analyzed_frames": frame_info.analyzed_frames,
        "trajectory_first_frame": frame_info.start,
        "trajectory_last_frame": last_frame,
        "trajectory_frame_stride": frame_info.step,
    }


def _stats(values: Sequence[float], *, prefix: str = "") -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    label = f"{prefix}_" if prefix else ""
    return {
        f"{label}mean": float(np.mean(array)),
        f"{label}std": float(np.std(array)),
        f"{label}min": float(np.min(array)),
        f"{label}max": float(np.max(array)),
    }


def _select_nonempty(universe: Any, selection: str, label: str) -> Any:
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise SelectionValidationError(f"{label} selection matched zero atoms: {selection}")
    return atoms


def calculate_radius_of_gyration(
    universe: Any,
    selection: str,
    *,
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    atoms = _select_nonempty(universe, selection, "Radius of gyration")
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    rows: list[dict[str, object]] = []
    values: list[float] = []
    for frame_index, ts in enumerate(_selected_trajectory_frames(universe, frame_info)):
        try:
            rg = float(atoms.radius_of_gyration(unwrap=True))
        except Exception:
            rg = float(atoms.radius_of_gyration())
        values.append(rg)
        rows.append({"frame": int(ts.frame), "time_ps": _frame_time(ts, frame_index, time_step_ps), "radius_of_gyration_A": rg})
    stats = {
        "n_atoms": len(atoms),
        "selection": selection,
        **_frame_selection_stats(frame_info),
        **_stats(values, prefix="rg_A"),
    }
    return _ComputedAnalysis(
        "rg",
        "radius_of_gyration",
        tuple(rows),
        "time_ps",
        "radius_of_gyration_A",
        "Time (ps)",
        "Radius of gyration (A)",
        "Radius of Gyration",
        stats,
    )


def calculate_distance(
    universe: Any,
    selection_a: str,
    selection_b: str,
    *,
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    _, _, distance_array, _ = _import_mdanalysis_analysis()
    atoms_a = _select_nonempty(universe, selection_a, "Distance A")
    atoms_b = _select_nonempty(universe, selection_b, "Distance B")
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    rows: list[dict[str, object]] = []
    values: list[float] = []
    mode = "atom-atom" if len(atoms_a) == 1 and len(atoms_b) == 1 else "center-of-geometry"
    for frame_index, ts in enumerate(_selected_trajectory_frames(universe, frame_info)):
        if mode == "atom-atom":
            pos_a = atoms_a.positions.reshape(1, 3)
            pos_b = atoms_b.positions.reshape(1, 3)
        else:
            pos_a = np.asarray(atoms_a.center_of_geometry(), dtype=float).reshape(1, 3)
            pos_b = np.asarray(atoms_b.center_of_geometry(), dtype=float).reshape(1, 3)
        dist = float(distance_array(pos_a, pos_b, box=getattr(ts, "dimensions", None))[0, 0])
        values.append(dist)
        rows.append({"frame": int(ts.frame), "time_ps": _frame_time(ts, frame_index, time_step_ps), "distance_A": dist})
    stats = {
        "selection_a": selection_a,
        "selection_b": selection_b,
        "n_atoms_a": len(atoms_a),
        "n_atoms_b": len(atoms_b),
        "mode": mode,
        **_frame_selection_stats(frame_info),
        **_stats(values, prefix="distance_A"),
    }
    return _ComputedAnalysis(
        "distance",
        "distance",
        tuple(rows),
        "time_ps",
        "distance_A",
        "Time (ps)",
        "Distance (A)",
        "Distance",
        stats,
    )


def calculate_rdf(
    universe: Any,
    selection_a: str,
    selection_b: str,
    *,
    nbins: int,
    rdf_range: tuple[float, float],
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    _, _, _, InterRDF = _import_mdanalysis_analysis()
    atoms_a = _select_nonempty(universe, selection_a, "RDF A")
    atoms_b = _select_nonempty(universe, selection_b, "RDF B")
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    rdf = InterRDF(atoms_a, atoms_b, nbins=nbins, range=rdf_range, norm="rdf")
    rdf.run(start=frame_info.start, stop=frame_info.stop, step=frame_info.step)
    bins = np.asarray(rdf.results.bins, dtype=float)
    rdf_values = np.asarray(rdf.results.rdf, dtype=float)
    bin_width = float(np.mean(np.diff(bins))) if len(bins) > 1 else (rdf_range[1] - rdf_range[0]) / max(nbins, 1)
    volume = _average_box_volume(universe, frame_info)
    density = len(atoms_b) / volume if volume > 0 else 0.0
    cumulative = np.cumsum(rdf_values * 4.0 * math.pi * bins**2 * bin_width * density)
    rows = tuple(
        {"r_A": float(radius), "g_r": float(value), "cumulative_coordination": float(coordination)}
        for radius, value, coordination in zip(bins, rdf_values, cumulative)
    )
    peak_index = int(np.argmax(rdf_values)) if len(rdf_values) else 0
    stats = {
        "selection_a": selection_a,
        "selection_b": selection_b,
        "n_atoms_a": len(atoms_a),
        "n_atoms_b": len(atoms_b),
        "nbins": nbins,
        "range_min_A": rdf_range[0],
        "range_max_A": rdf_range[1],
        **_frame_selection_stats(frame_info),
        "number_density_B_per_A3": density,
        "rdf_peak_g_r": float(rdf_values[peak_index]) if len(rdf_values) else 0.0,
        "rdf_peak_r_A": float(bins[peak_index]) if len(bins) else 0.0,
    }
    return _ComputedAnalysis("rdf", "rdf", rows, "r_A", "g_r", "r (A)", "g(r)", "Radial Distribution Function", stats)


def calculate_rmsd(
    universe: Any,
    target_selection: str,
    alignment_selection: str,
    *,
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    _, rms, _, _ = _import_mdanalysis_analysis()
    _select_nonempty(universe, target_selection, "RMSD target")
    _select_nonempty(universe, alignment_selection, "RMSD alignment")
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    try:
        universe.trajectory[frame_info.start]
    except Exception:
        pass
    groupselections = None if target_selection == alignment_selection else [target_selection]
    analysis = rms.RMSD(universe, universe, select=alignment_selection, groupselections=groupselections)
    analysis.run(start=frame_info.start, stop=frame_info.stop, step=frame_info.step)
    data = np.asarray(analysis.results.rmsd, dtype=float)
    rmsd_column = 3 if groupselections else 2
    rows = []
    values = []
    for row_index, item in enumerate(data):
        frame = int(item[0])
        time_ps = float(item[1]) if math.isfinite(float(item[1])) else frame * time_step_ps
        if frame > 0 and abs(time_ps) <= 1.0e-12:
            time_ps = frame * time_step_ps
        rmsd_value = float(item[rmsd_column])
        values.append(rmsd_value)
        rows.append({"frame": frame, "time_ps": time_ps, "rmsd_A": rmsd_value})
    stats = {
        "target_selection": target_selection,
        "alignment_selection": alignment_selection,
        **_frame_selection_stats(frame_info),
        **_stats(values, prefix="rmsd_A"),
    }
    return _ComputedAnalysis("rmsd", "rmsd", tuple(rows), "time_ps", "rmsd_A", "Time (ps)", "RMSD (A)", "RMSD", stats)


def calculate_rmsf(
    universe: Any,
    target_selection: str,
    alignment_selection: str,
    *,
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    align, rms, _, _ = _import_mdanalysis_analysis()
    target_atoms = _select_nonempty(universe, target_selection, "RMSF target")
    _select_nonempty(universe, alignment_selection, "RMSF alignment")
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    try:
        universe.trajectory[frame_info.start]
    except Exception:
        pass
    align.AlignTraj(universe, universe, select=alignment_selection, in_memory=True).run(
        start=frame_info.start,
        stop=frame_info.stop,
        step=frame_info.step,
    )
    rmsf = rms.RMSF(target_atoms).run(start=frame_info.start, stop=frame_info.stop, step=frame_info.step)
    values = np.asarray(rmsf.results.rmsf, dtype=float)
    rows: list[dict[str, object]] = []
    for atom, value in zip(target_atoms, values):
        rows.append(
            {
                "atom_index": int(getattr(atom, "index", 0)) + 1,
                "resid": int(getattr(atom, "resid", 0)),
                "resname": str(getattr(atom, "resname", "")),
                "name": str(getattr(atom, "name", "")),
                "rmsf_A": float(value),
            }
        )
    stats = {
        "target_selection": target_selection,
        "alignment_selection": alignment_selection,
        "n_atoms": len(target_atoms),
        **_frame_selection_stats(frame_info),
        **_stats(values, prefix="rmsf_A"),
    }
    return _ComputedAnalysis("rmsf", "rmsf", tuple(rows), "atom_index", "rmsf_A", "Atom index", "RMSF (A)", "RMSF", stats)


def _average_box_volume(universe: Any, frame_info: _ResolvedFrameSelection | None = None) -> float:
    volumes: list[float] = []
    try:
        current_frame = universe.trajectory.frame
    except Exception:
        current_frame = None
    trajectory = _selected_trajectory_frames(universe, frame_info) if frame_info is not None else universe.trajectory
    for ts in trajectory:
        dims = getattr(ts, "dimensions", None)
        if dims is not None and len(dims) >= 3:
            volume = float(np.prod(np.asarray(dims[:3], dtype=float)))
            if volume > 0:
                volumes.append(volume)
    if current_frame is not None:
        try:
            universe.trajectory[current_frame]
        except Exception:
            pass
    if volumes:
        return float(np.mean(volumes))
    positions = np.asarray(universe.atoms.positions, dtype=float)
    extent = np.max(positions, axis=0) - np.min(positions, axis=0)
    volume = float(np.prod(extent))
    return volume if volume > 0 else 1.0


def _compute_analysis(
    universe: Any,
    request: TrajectoryAnalysisRequest,
    *,
    time_step_ps: float,
    frame_selection: TrajectoryFrameSelection | None = None,
) -> _ComputedAnalysis:
    kind = request.analysis_type.lower()
    if kind == "rg":
        if request.target_selection is None:
            raise ValueError("Radius of gyration requires a target selection")
        return calculate_radius_of_gyration(
            universe,
            request.target_selection,
            time_step_ps=time_step_ps,
            frame_selection=frame_selection,
        )
    if kind == "distance":
        if request.selection_a is None or request.selection_b is None:
            raise ValueError("Distance analysis requires two selections")
        return calculate_distance(
            universe,
            request.selection_a,
            request.selection_b,
            time_step_ps=time_step_ps,
            frame_selection=frame_selection,
        )
    if kind == "rdf":
        if request.selection_a is None or request.selection_b is None:
            raise ValueError("RDF analysis requires two selections")
        return calculate_rdf(
            universe,
            request.selection_a,
            request.selection_b,
            nbins=request.nbins,
            rdf_range=request.rdf_range,
            time_step_ps=time_step_ps,
            frame_selection=frame_selection,
        )
    if kind == "rmsd":
        if request.target_selection is None:
            raise ValueError("RMSD requires a target selection")
        alignment = request.alignment_selection or request.target_selection
        return calculate_rmsd(
            universe,
            request.target_selection,
            alignment,
            time_step_ps=time_step_ps,
            frame_selection=frame_selection,
        )
    if kind == "rmsf":
        if request.target_selection is None:
            raise ValueError("RMSF requires a target selection")
        alignment = request.alignment_selection or request.target_selection
        return calculate_rmsf(
            universe,
            request.target_selection,
            alignment,
            time_step_ps=time_step_ps,
            frame_selection=frame_selection,
        )
    raise ValueError(f"Unsupported trajectory analysis type: {request.analysis_type}")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plot(path: Path, result: _ComputedAnalysis, *, case_label: str) -> None:
    if result.x_column is None or result.y_column is None:
        return
    plt = _import_pyplot()
    x_values = [float(row[result.x_column]) for row in result.rows]
    y_values = [float(row[result.y_column]) for row in result.rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(x_values, y_values, linewidth=1.6)
    ax.set_xlabel(result.x_label)
    ax.set_ylabel(result.y_label)
    ax.set_title(f"{case_label}: {result.title}")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_case_summary(case_dir: Path, case: TrajectoryCase, outputs: Sequence[AnalysisOutput]) -> tuple[Path, Path]:
    txt_path = case_dir / "summary.txt"
    json_path = case_dir / "summary.json"
    payload = {
        "case": case.label,
        "root": str(case.root),
        "topology": str(case.topology_path),
        "trajectory": str(case.trajectory_path),
        "analyses": [
            {
                "analysis_type": output.analysis_type,
                "csv": str(output.csv_path),
                "png": str(output.png_path),
                "stats": output.stats,
            }
            for output in outputs
        ],
    }
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Trajectory analysis summary: {case.label}\n")
        handle.write(f"Topology: {case.topology_path}\n")
        handle.write(f"Trajectory: {case.trajectory_path}\n\n")
        for output in outputs:
            handle.write(f"[{output.analysis_type}]\n")
            handle.write(f"CSV: {output.csv_path}\n")
            handle.write(f"PNG: {output.png_path}\n")
            for key, value in output.stats.items():
                handle.write(f"{key}: {value}\n")
            handle.write("\n")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return txt_path, json_path


def _write_combined_summary(path: Path, outputs: Sequence[AnalysisOutput]) -> None:
    rows: list[dict[str, object]] = []
    for output in outputs:
        row: dict[str, object] = {"case": output.case_label, "analysis_type": output.analysis_type}
        row.update(output.stats)
        rows.append(row)
    _write_csv(path, rows)


def _write_overlay_plots(output_root: Path, outputs: Sequence[AnalysisOutput]) -> tuple[Path, ...]:
    overlay_paths: list[Path] = []
    by_analysis: dict[str, list[AnalysisOutput]] = {}
    for output in outputs:
        if output.x_column and output.y_column and output.analysis_type != "rmsf":
            by_analysis.setdefault(output.analysis_type, []).append(output)
    for analysis_type, grouped in sorted(by_analysis.items()):
        if len(grouped) < 2:
            continue
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        x_label = ""
        y_label = ""
        for output in grouped:
            rows = _read_numeric_csv(output.csv_path)
            x_values = [row[output.x_column] for row in rows if output.x_column in row and output.y_column in row]
            y_values = [row[output.y_column] for row in rows if output.x_column in row and output.y_column in row]
            if not x_values or not y_values:
                continue
            x_label = output.x_column
            y_label = output.y_column
            ax.plot(x_values, y_values, label=output.case_label, linewidth=1.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Overlay: {analysis_type}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
        overlay_path = output_root / f"overlay_{analysis_type}.png"
        fig.savefig(overlay_path, dpi=180)
        plt.close(fig)
        overlay_paths.append(overlay_path)
    return tuple(overlay_paths)


def _read_numeric_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


def run_trajectory_analyses(
    cases: Sequence[TrajectoryCase],
    requests: Sequence[TrajectoryAnalysisRequest],
    *,
    output_root: Path,
    time_step_ps: float = 1.0,
    frame_selection: TrajectoryFrameSelection | None = None,
    universes: Sequence[Any] | None = None,
) -> TrajectoryAnalysisRunResult:
    if not cases:
        raise ValueError("At least one trajectory case is required")
    if not requests:
        raise ValueError("At least one trajectory analysis request is required")
    frame_selection = frame_selection or TrajectoryFrameSelection()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    loaded_universes = list(universes) if universes is not None else [load_universe(case) for case in cases]
    if len(loaded_universes) != len(cases):
        raise ValueError("case and universe counts do not match")
    profiles = [build_case_profile(case.label, universe) for case, universe in zip(cases, loaded_universes)]
    check_case_compatibility(profiles)

    all_outputs: list[AnalysisOutput] = []
    for case, universe in zip(cases, loaded_universes):
        case_dir = output_root / _slugify(case.label)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_outputs: list[AnalysisOutput] = []
        for request in requests:
            computed = _compute_analysis(
                universe,
                request,
                time_step_ps=time_step_ps,
                frame_selection=frame_selection,
            )
            csv_path = case_dir / f"{computed.slug}.csv"
            png_path = case_dir / f"{computed.slug}.png"
            _write_csv(csv_path, computed.rows)
            _write_plot(png_path, computed, case_label=case.label)
            placeholder_summary = case_dir / "summary.txt"
            placeholder_json = case_dir / "summary.json"
            output = AnalysisOutput(
                case_label=case.label,
                analysis_type=computed.analysis_type,
                csv_path=csv_path,
                png_path=png_path,
                summary_txt_path=placeholder_summary,
                summary_json_path=placeholder_json,
                stats=computed.stats,
                x_column=computed.x_column,
                y_column=computed.y_column,
            )
            case_outputs.append(output)
        summary_txt, summary_json = _write_case_summary(case_dir, case, case_outputs)
        finalized = [
            AnalysisOutput(
                case_label=output.case_label,
                analysis_type=output.analysis_type,
                csv_path=output.csv_path,
                png_path=output.png_path,
                summary_txt_path=summary_txt,
                summary_json_path=summary_json,
                stats=output.stats,
                x_column=output.x_column,
                y_column=output.y_column,
            )
            for output in case_outputs
        ]
        all_outputs.extend(finalized)

    combined_summary = output_root / "combined_summary.csv"
    _write_combined_summary(combined_summary, all_outputs)
    overlay_paths = _write_overlay_plots(output_root, all_outputs)
    return TrajectoryAnalysisRunResult(
        output_dir=output_root,
        outputs=tuple(all_outputs),
        overlay_paths=overlay_paths,
        combined_summary_csv_path=combined_summary,
    )
