"""Topology-complete coordinate views, including Amber water extra points.

Never borrow missing physical coordinates from an unrelated trajectory frame.
Only a matching complete coordinate source or a defined rigid-water EP frame
can complete a reference. Original input files are not overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import math
from pathlib import Path

import gemmi
import numpy as np

from amber_metallo.inspection import SUPPORTED_METALS, WATER_NAMES, load_structure
from amber_metallo.reporting import write_json
from amber_metallo.ti.analysis import (
    _fixed_width_tokens_from_section, _iter_indexed_atoms, _parse_prmtop_sections,
    _tokens_from_section,
)


class IncompleteReferenceError(ValueError):
    """A real coordinate source is needed; do not fabricate missing atoms."""


@dataclass
class TopologyIdentity:
    names: list[str]
    labels: list[str]
    starts: list[int]
    numbers: list[int]
    sections: dict

    @classmethod
    def read(cls, path):
        sections = _parse_prmtop_sections(path)
        names = _fixed_width_tokens_from_section(sections, "ATOM_NAME")
        labels = _fixed_width_tokens_from_section(sections, "RESIDUE_LABEL")
        starts = [int(v) - 1 for v in _tokens_from_section(sections, "RESIDUE_POINTER")]
        numbers = [int(v) for v in _tokens_from_section(sections, "ATOMIC_NUMBER")]
        pointers = _tokens_from_section(sections, "POINTERS")
        if (not names or not labels or len(labels) != len(starts) or starts[0] != 0
                or any(b <= a for a, b in zip(starts, starts[1:])) or starts[-1] >= len(names)
                or (pointers and int(pointers[0]) != len(names))
                or (numbers and len(numbers) != len(names))):
            raise ValueError("Topology has inconsistent NATOM/atom/residue identity records.")
        return cls(names, labels, starts, numbers, sections)

    def ranges(self):
        return zip(self.labels, self.starts, self.starts[1:] + [len(self.names)])

    @cached_property
    def extra_point_flags(self):
        types = _fixed_width_tokens_from_section(self.sections, "AMBER_ATOM_TYPE")
        masses = _tokens_from_section(self.sections, "MASS")
        # Atomic number 0 includes extra points, not an oxygen nucleus.
        return [bool((self.numbers and self.numbers[index] == 0)
                    or (index < len(masses) and float(masses[index]) == 0
                        and (self.names[index].upper().startswith(("EP", "LP"))
                             or (index < len(types) and types[index].upper().startswith(("EP", "LP"))))))
                for index in range(len(self.names))]

    def is_extra_point(self, index):
        return self.extra_point_flags[index]


def restart_coordinates(path, topology):
    """Read a full formatted or NetCDF Amber restart, preserving all sites."""
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
    box = None
    if magic.startswith(b"CDF") or magic == b"\x89HDF":
        from amber_metallo.ti.snapshots import open_netcdf
        with open_netcdf(path) as dataset:
            raw = dataset.variables["coordinates"][:]
            # scipy does not mask the standard NetCDF fill sentinel unless an
            # explicit _FillValue attribute exists. It is finite (~1e37), but
            # cannot be a physical molecular coordinate.
            if np.any(np.ma.getmaskarray(raw)) or np.any(np.abs(raw) >= 1e30):
                raise ValueError(f"Missing/masked coordinates in restart: {path}")
            coords = np.array(raw, dtype=float, copy=True)
            del raw
            if "cell_lengths" in dataset.variables:
                lengths = np.array(dataset.variables["cell_lengths"][:], dtype=float).ravel()
                angles = (np.array(dataset.variables["cell_angles"][:], dtype=float).ravel()
                          if "cell_angles" in dataset.variables else np.array([90., 90., 90.]))
                box = np.concatenate((lengths, angles))
        if coords.shape == (1, len(topology.names), 3):
            coords = coords[0]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or not lines[1].split():
            raise ValueError(f"Not a valid Amber restart: {path}")
        count = int(lines[1].split()[0])
        if count != len(topology.names):
            raise ValueError(f"Restart/topology NATOM mismatch ({count} vs {len(topology.names)}): {path}")
        values = []
        for line in lines[2:]:
            try:
                values.extend([float(v.replace("D", "E").replace("d", "e")) for v in line.split()])
            except ValueError:
                # Adjacent negative F12.7 values need fixed-width parsing.
                values.extend(float(line[i:i + 12].replace("D", "E").replace("d", "e"))
                              for i in range(0, len(line), 12) if line[i:i + 12].strip())
        if len(values) < 3 * count:
            raise ValueError(f"Restart is truncated: {path}")
        coords = np.asarray(values[:3 * count]).reshape(count, 3)
        pointers = _tokens_from_section(topology.sections, "POINTERS")
        periodic = len(pointers) > 27 and int(pointers[27]) != 0
        if periodic:
            remainder = len(values) - 3 * count
            if remainder in {6, 3 * count + 6}:
                box = np.array(values[-6:])
            elif remainder in {3, 3 * count + 3}:
                box = np.array([*values[-3:], 90., 90., 90.])
    if coords.shape != (len(topology.names), 3) or not np.all(np.isfinite(coords)):
        raise ValueError(f"Full coordinates must have exactly {len(topology.names)} finite XYZ rows: {path}")
    if box is not None and (box.shape != (6,) or not np.all(np.isfinite(box)) or np.any(box <= 0)):
        raise ValueError(f"Invalid periodic box in restart: {path}")
    if box is not None:
        cosines = np.cos(np.deg2rad(box[3:]))
        determinant = 1 - np.sum(cosines ** 2) + 2 * np.prod(cosines)
        if np.any(box[3:] >= 180) or determinant <= 0:
            raise ValueError(f"Degenerate periodic box in restart: {path}")
    return coords, box


def _write_full_pdb(topology, coords, output_path, box=None):
    if coords.shape != (len(topology.names), 3) or not np.all(np.isfinite(coords)):
        raise ValueError("Cannot write a full PDB from missing/non-finite coordinates.")
    structure = gemmi.Structure()
    structure.name = "SIMPLE topology-complete coordinates"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for number, (label, start, end) in enumerate(topology.ranges(), start=1):
        residue = gemmi.Residue()
        residue.name = label[:3]
        residue.seqid = gemmi.SeqId(number, " ")
        for index in range(start, end):
            atom = gemmi.Atom()
            atom.name = topology.names[index]
            atom.serial = index + 1
            if topology.numbers:
                atom.element = gemmi.Element(topology.numbers[index])
            elif topology.is_extra_point(index):
                atom.element = gemmi.Element("X")
            elif end - start == 1 and label.title() in SUPPORTED_METALS:
                atom.element = gemmi.Element(label.title())
            else:
                letters = "".join(c for c in atom.name if c.isalpha())
                atom.element = gemmi.Element(letters[:1] or "X")
            atom.pos = gemmi.Position(*coords[index])
            atom.occ = 1.
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    if box is not None:
        structure.cell = gemmi.UnitCell(*box)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = structure.make_minimal_pdb()
    # Stable preview input mtimes matter for saved-config integrity checks.
    if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
        path.write_text(rendered, encoding="utf-8")
    # Round-trip through the same parser used by CN/selection, not a raw ATOM
    # line count. This detects parser losses, duplicate atom names and overflow.
    atoms = _iter_indexed_atoms(path)
    if (len(atoms) != len(topology.names)
            or [a.atom_name.upper() for a in atoms] != [n.upper() for n in topology.names]
            or not np.allclose([[a.position.x, a.position.y, a.position.z] for a in atoms], coords, atol=.001, rtol=0)):
        raise ValueError("Full PDB round-trip lost/changed atoms. Coordinate data cannot be safely represented by this PDB.")
    return path


def full_pdb_from_restart(*, prmtop_path, restart_path, output_path):
    topology = TopologyIdentity.read(prmtop_path)
    coords, box = restart_coordinates(restart_path, topology)
    path = _write_full_pdb(topology, coords, output_path, box)
    write_json(path.with_suffix(".full_structure.json"), {
        "source_restart": str(Path(restart_path).resolve()), "topology": str(Path(prmtop_path).resolve()),
        "full_pdb": str(path), "atom_count": len(coords), "extra_points_included": True,
        "coordinate_source": "same-frame complete Amber restart", "validated_roundtrip": True,
    })
    return path


def _restore_water_extra_points(topology, coords, missing):
    """Only the defined symmetric O/H/H/M water frame (OPC/TIP4P family).

Weights are derived from this topology's equilibrium bonds/angle, not a
hard-coded water model distance and not another trajectory frame.
"""
    req = [float(v.replace("D", "E")) for v in _tokens_from_section(topology.sections, "BOND_EQUIL_VALUE")]
    bonds = {}
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(v) for v in _tokens_from_section(topology.sections, flag)]
        for offset in range(0, len(values), 3):
            a, b, kind = values[offset:offset + 3]
            bonds[frozenset((a // 3, b // 3))] = req[kind - 1]
    angles = {}
    eq = [float(v.replace("D", "E")) for v in _tokens_from_section(topology.sections, "ANGLE_EQUIL_VALUE")]
    for flag in ("ANGLES_INC_HYDROGEN", "ANGLES_WITHOUT_HYDROGEN"):
        values = [int(v) for v in _tokens_from_section(topology.sections, flag)]
        for offset in range(0, len(values), 4):
            a, b, c, kind = values[offset:offset + 4]
            angles[(frozenset((a // 3, c // 3)), b // 3)] = eq[kind - 1]
    restored = []
    for label, start, end in topology.ranges():
        candidates = [i for i in range(start, end) if i in missing]
        if not candidates:
            continue
        if label.upper() not in WATER_NAMES or end - start != 4 or len(candidates) != 1 or not topology.numbers:
            continue
        ep = candidates[0]
        oxygens = [i for i in range(start, end) if topology.numbers[i] == 8]
        hydrogens = [i for i in range(start, end) if topology.numbers[i] == 1]
        if not topology.is_extra_point(ep) or len(oxygens) != 1 or len(hydrogens) != 2:
            continue
        o, h1, h2 = oxygens[0], *hydrogens
        if not np.all(np.isfinite(coords[[o, h1, h2]])):
            continue
        try:
            r1, r2, rm = (bonds[frozenset((o, i))] for i in (h1, h2, ep))
            hh = bonds.get(frozenset((h1, h2)))
            if hh is None:
                theta = angles[(frozenset((h1, h2)), o)]
                hh = math.sqrt(r1*r1 + r2*r2 - 2*r1*r2*math.cos(theta))
            if abs(r1 - r2) > 1e-5 or not 0 < rm < r1:
                continue
            weight = rm / math.sqrt(r1*r2 - .25*hh*hh)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        # Reject a split/invalid water rather than placing its EP far away.
        if (any(abs(np.linalg.norm(coords[h] - coords[o]) - r1) > .05 for h in (h1, h2))
                or abs(np.linalg.norm(coords[h1] - coords[h2]) - hh) > .05):
            continue
        coords[ep] = (1 - weight)*coords[o] + .5*weight*(coords[h1] + coords[h2])
        restored.append(ep + 1)
    return restored


def normalize_full_reference(*, reference_pdb, prmtop_path, output_dir, coordinate_candidates=()):
    from amber_metallo.ti.atom_mapping import map_to_topology
    topology = TopologyIdentity.read(prmtop_path)
    reference_pdb = Path(reference_pdb).expanduser().resolve()
    atoms = _iter_indexed_atoms(reference_pdb)
    if not atoms or not np.all(np.isfinite([[a.position.x, a.position.y, a.position.z] for a in atoms])):
        raise ValueError("Reference is empty or contains non-finite coordinates; supply a valid coordinate file.")
    try:
        mapping = map_to_topology(reference_pdb, prmtop_path)
    except ValueError as exc:
        raise IncompleteReferenceError(str(exc)) from exc
    if len(mapping) != len(atoms):
        raise IncompleteReferenceError("Reference atom identities do not map uniquely to this topology. "
                                       "Supply the corresponding full topology-ordered reference coordinates.")
    coords = np.full((len(topology.names), 3), np.nan)
    for atom in atoms:
        coords[mapping[atom.atom_index] - 1] = [atom.position.x, atom.position.y, atom.position.z]
    present = np.all(np.isfinite(coords), axis=1)
    missing = set(np.flatnonzero(~present).tolist())
    structure = load_structure(reference_pdb)
    box = np.array(structure.cell.parameters) if structure.cell.is_crystal() else None
    provenance = {"source_reference": str(reference_pdb), "topology": str(Path(prmtop_path).resolve()),
                  "input_atom_count": len(atoms), "atom_count": len(coords), "restored_water_ep_indices": []}
    if missing:
        companions = [*coordinate_candidates]
        for base in (reference_pdb, Path(prmtop_path)):
            companions.extend(base.with_suffix(suffix) for suffix in (".rst7", ".inpcrd", ".ncrst", ".rst", ".crd"))
        for raw in dict.fromkeys(str(p) for p in companions if p):
            companion = Path(raw).expanduser().resolve()
            if not companion.is_file() or companion == reference_pdb:
                continue
            try:
                if companion.suffix.lower() in {".pdb", ".cif"}:
                    other_mapping = map_to_topology(companion, prmtop_path)
                    other_atoms = _iter_indexed_atoms(companion)
                    if len(other_mapping) != len(topology.names):
                        continue
                    complete = np.empty_like(coords)
                    for atom in other_atoms:
                        complete[other_mapping[atom.atom_index] - 1] = [atom.position.x, atom.position.y, atom.position.z]
                    other_cell = load_structure(companion).cell
                    other_box = np.array(other_cell.parameters) if other_cell.is_crystal() else None
                else:
                    complete, other_box = restart_coordinates(companion, topology)
                if not np.allclose(coords[present], complete[present], atol=.002, rtol=0):
                    continue  # Not this reference: never silently use a different time frame.
                coords[~present] = complete[~present]
                box = other_box if other_box is not None else box
                provenance["matching_complete_coordinate_source"] = str(companion)
                missing.clear()
                break
            except (ValueError, OSError, KeyError):
                continue
        if missing:
            provenance["restored_water_ep_indices"] = _restore_water_extra_points(topology, coords, missing)
        missing = np.flatnonzero(~np.all(np.isfinite(coords), axis=1))
        if len(missing):
            raise IncompleteReferenceError(
                f"Reference lacks {len(missing)} topology coordinates (first indices: {(missing[:10] + 1).tolist()}). "
                "No matching full restart or supported water-EP geometry could supply them. "
                "Provide this reference's full PDB/restart; physical atoms/solvent will not be fabricated or borrowed from another frame.")
    digest = hashlib.sha256(reference_pdb.read_bytes() + Path(prmtop_path).read_bytes() + coords.tobytes()).hexdigest()[:16]
    target = Path(output_dir) / f"reference_full_{digest}.pdb"
    path = _write_full_pdb(topology, coords, target, box)
    provenance.update(full_pdb=str(path), extra_points_included=True, validated_roundtrip=True,
                      original_coordinates_preserved=True)
    sidecar = path.with_suffix(".full_structure.json")
    if path != reference_pdb or not sidecar.exists():
        write_json(sidecar, provenance)
    return path
