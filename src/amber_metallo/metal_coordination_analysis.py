"""Geometric metal--oxygen coordination over selected trajectory frames."""
from __future__ import annotations

import math
import re

import numpy as np

from amber_metallo.trajectory_analysis import (
    SUPPORTED_METAL_RESNAMES, WATER_RESNAMES, _ComputedAnalysis,
    _frame_selection_stats, _frame_time, _resolve_frame_selection,
    _selected_trajectory_frames, _stats,
)


def _attribute(atom, name, default=""):
    try:
        return getattr(atom, name)
    except (AttributeError, LookupError):
        return default


def atom_identity(atom):
    chain = str(_attribute(atom, "chainID") or _attribute(atom, "segid"))
    residue = f"{atom.resid}{atom.resname}"
    return {"atom_index": int(atom.index) + 1, "residue_index": int(atom.resindex) + 1,
            "chain_or_segment": chain, "resid": int(atom.resid), "resname": str(atom.resname),
            "atom_name": str(atom.name),
            "atom_label": f"{chain + ':' if chain else ''}{residue}@{atom.name} [#{atom.index + 1}]"}


def metal_atom_options(universe):
    options = []
    for atom in universe.atoms:
        element = str(_attribute(atom, "element")).upper()
        # Never mistake a protein CA carbon for a calcium ion. When element
        # data are absent, only single-atom residues can be inferred as ions.
        if not element and len(atom.residue.atoms) == 1:
            element = re.sub(r"[0-9+\-]", "", str(atom.resname)).upper()
        if element in SUPPORTED_METAL_RESNAMES:
            options.append({**atom_identity(atom), "element": element})
    return options


def oxygen_atom_indices(universe):
    indices = []
    for atom in universe.atoms:
        element = str(_attribute(atom, "element")).upper()
        if element:
            is_oxygen = element == "O"
        else:
            mass = float(_attribute(atom, "mass", 0.0))
            is_oxygen = 15.5 < mass < 16.5 if mass > 0 else str(atom.name).upper().startswith("O")
        if is_oxygen:
            indices.append(int(atom.index))
    return np.asarray(indices, dtype=int)


def _box(ts):
    if ts.dimensions is None:
        return None
    box = np.asarray(ts.dimensions, dtype=float)
    if box.shape != (6,) or not np.all(np.isfinite(box)) or np.any(box[:3] <= 0) or np.any(box[3:] <= 0) or np.any(box[3:] >= 180):
        raise ValueError("Invalid periodic box in trajectory; cannot compute reliable metal--O distances.")
    return box


def calculate_metal_coordination(universe, metal_indices=None, *, cutoff=4.0, time_step_ps=1.0, frame_selection=None):
    """Yield per-metal CN and distance tables, including population SD (not SEM).

CN is recomputed from ALL oxygen atoms in each selected frame. Distance series
track the union of O identities ever inside the cutoff in those same frames,
including their out-of-shell distances (so the mean is not contact-censored).
    """
    from MDAnalysis.lib.distances import distance_array

    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Metal coordination cutoff must be finite and positive")
    options = {o["atom_index"]: o for o in metal_atom_options(universe)}
    selected = list(options) if metal_indices is None else list(metal_indices)
    if len(set(selected)) != len(selected) or any(i not in options for i in selected):
        raise ValueError("Choose unique metal topology atom indices from this case's metal list")
    if not selected:
        return
    oxygens = oxygen_atom_indices(universe)
    oxygen_group = universe.atoms[oxygens]
    metal_group = universe.atoms[np.asarray(selected, dtype=int) - 1]
    water = np.asarray([str(a.resname).upper() in WATER_RESNAMES for a in oxygen_group], dtype=bool)
    frame_info = _resolve_frame_selection(universe, frame_selection, time_step_ps=time_step_ps)
    contacts = [set() for _ in selected]
    cn_rows = [[] for _ in selected]
    periodic_frames = 0
    for ts in _selected_trajectory_frames(universe, frame_info):
        box = _box(ts)
        periodic_frames += box is not None
        distances = distance_array(metal_group.positions, oxygen_group.positions, box=box)
        if not np.all(np.isfinite(distances)):
            raise ValueError("Non-finite metal/oxygen coordinates in the selected trajectory frames")
        inside = distances <= cutoff
        for j, metal_index in enumerate(selected):
            contacts[j].update(np.flatnonzero(inside[j]).tolist())
            cn_rows[j].append({"frame_index": int(ts.frame), "time_ns": _frame_time(ts, ts.frame, time_step_ps) / 1000.,
                               "metal_atom_index": metal_index, "metal_label": options[metal_index]["atom_label"],
                               "oxygen_cn": int(inside[j].sum()), "nonwater_oxygen_cn": int(inside[j, ~water].sum()),
                               "water_oxygen_cn": int(inside[j, water].sum())})
    for j, metal_index in enumerate(selected):
        label = options[metal_index]["atom_label"]
        slug = f"metal_{metal_index}_oxygen"
        common = {**_frame_selection_stats(frame_info), "metal_atom_index": metal_index, "metal_label": label,
                  "cutoff_angstrom": cutoff, "periodic_frames": periodic_frames,
                  "distance_method": "minimum image where a box is present; Cartesian otherwise",
                  "std_definition": "population standard deviation (ddof=0), not SEM",
                  "candidate_rule": "O atoms inside cutoff in at least one analyzed frame"}
        stats = dict(common)
        cn_summary = []
        for key in ("oxygen_cn", "nonwater_oxygen_cn", "water_oxygen_cn"):
            values = [row[key] for row in cn_rows[j]]
            stats.update(_stats(values, prefix=key))
            cn_summary.append({"metal_atom_index": metal_index, "metal_label": label, "quantity": key,
                               "n_frames": len(values), "cutoff_angstrom": cutoff, **_stats(values)})
        yield _ComputedAnalysis("metal_oxygen_cn", slug + "_cn", tuple(cn_rows[j]), "time_ns", "oxygen_cn",
                                "Time (ns)", "O-CN", f"{label}: oxygen coordination", stats,
                                summary_rows=tuple(cn_summary))
        if not contacts[j]:
            continue
        offsets = sorted(contacts[j])
        group = oxygen_group[offsets]
        identities = [atom_identity(a) for a in group]
        matrix, times, frames = [], [], []
        for ts in _selected_trajectory_frames(universe, frame_info):
            values = distance_array(universe.atoms[[metal_index - 1]].positions, group.positions, box=_box(ts))[0]
            matrix.append(values)
            times.append(_frame_time(ts, ts.frame, time_step_ps) / 1000.)
            frames.append(int(ts.frame))
        matrix = np.asarray(matrix)
        rows, summaries = [], []
        for column, identity in enumerate(identities):
            values = matrix[:, column]
            contact = values <= cutoff
            summaries.append({"metal_atom_index": metal_index, "metal_label": label, **identity,
                              "cutoff_angstrom": cutoff, "n_frames": len(values), "contact_frames": int(contact.sum()),
                              "occupancy_fraction": float(contact.mean()), **_stats(values, prefix="distance_angstrom"),
                              **_stats(values[contact], prefix="in_contact_distance_angstrom")})
            for frame, time, distance in zip(frames, times, values, strict=True):
                rows.append({"frame_index": frame, "time_ns": time, "metal_atom_index": metal_index,
                             "metal_label": label, **identity, "distance_angstrom": float(distance),
                             "within_cutoff": int(distance <= cutoff)})
        yield _ComputedAnalysis("metal_oxygen_distance", slug + "_distances", tuple(rows), "time_ns", "distance_angstrom",
                                "Time (ns)", "Metal-O distance (Angstrom)", f"{label}: individual O distances",
                                {**common, "tracked_oxygen_atoms": len(offsets)}, series_column="atom_label",
                                summary_rows=tuple(summaries))
