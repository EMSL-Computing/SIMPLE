"""Conservative identity mapping for hydrogen/solvent-stripped reference PDBs.

PDB serials and coordinates are never used to guess topology indices. Residues
must retain their relative order. Repeated, indistinguishable residues are left
unmapped unless the ordered context uniquely identifies them.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import replace

from amber_metallo.inspection import load_structure
from amber_metallo.ti.analysis import (
    _iter_indexed_atoms, _parse_prmtop_sections, _fixed_width_tokens_from_section,
    _tokens_from_section, assess_site_stability, SiteStabilityAssessment,
)


def _residues(atoms):
    blocks = []
    for atom in atoms:
        if not blocks or blocks[-1][0].residue_key != atom.residue_key:
            blocks.append([])
        blocks[-1].append(atom)
    return blocks


def _label(name):
    # Amber protonation names have identical heavy-atom residue identities.
    return {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS",
            "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
            "HOH": "WAT"}.get(name[:3].upper(), name[:3].upper())


def _ordered_mapping(atoms, target_blocks):
    blocks = _residues(atoms)
    signatures = [(_label(name), Counter(atom_names)) for name, atom_names, _ in target_blocks]
    by_label = defaultdict(list)
    for i, (label, _) in enumerate(signatures):
        by_label[label].append(i)
    cache = {}
    possibilities = []
    for block in blocks:
        names = Counter(a.atom_name.upper() for a in block if a.element not in {"H", "D"})
        label = _label(block[0].residue_name)
        key = (label, tuple(sorted(names.items())))
        if key not in cache:
            cache[key] = [i for i in by_label[label] if names <= signatures[i][1]]
        possibilities.append(cache[key])
    # Earliest/latest monotone embeddings agree iff that residue's position is
    # unique. This supports missing H, omitted solvent and residue renumbering.
    earliest, cursor = [], -1
    for choices in possibilities:
        offset = bisect_right(choices, cursor)
        valid = choices[offset] if offset < len(choices) else None
        if valid is None:
            raise ValueError("Cannot map reference residues in order by residue/atom identity. "
                             "Use a reference from this topology (missing H/solvent is OK).")
        earliest.append(valid)
        cursor = valid
    latest, cursor = [], len(target_blocks)
    for choices in reversed(possibilities):
        offset = bisect_left(choices, cursor) - 1
        valid = choices[offset] if offset >= 0 else None
        if valid is None:
            raise ValueError("Reference residue correspondence is inconsistent.")
        latest.append(valid)
        cursor = valid
    mapping = {}
    for block, first, last in zip(blocks, earliest, reversed(latest), strict=True):
        if first != last:
            continue
        _, names, indices = target_blocks[first]
        for atom in block:
            hits = [index for name, index in zip(names, indices, strict=True)
                    if name.upper() == atom.atom_name.upper()]
            if len(hits) == 1:
                mapping[atom.atom_index] = hits[0]
    return mapping


def map_to_topology(pdb_path, prmtop_path):
    sections = _parse_prmtop_sections(prmtop_path)
    names = _fixed_width_tokens_from_section(sections, "ATOM_NAME")
    labels = _fixed_width_tokens_from_section(sections, "RESIDUE_LABEL")
    starts = [int(i) - 1 for i in _tokens_from_section(sections, "RESIDUE_POINTER")]
    if not names or not starts or len(labels) != len(starts):
        raise ValueError("Topology lacks atom/residue identity records for restraint mapping.")
    ends = starts[1:] + [len(names)]
    blocks = [(label, [n.upper() for n in names[start:end]], list(range(start + 1, end + 1)))
              for label, start, end in zip(labels, starts, ends, strict=True)]
    return _ordered_mapping(_iter_indexed_atoms(pdb_path), blocks)


def map_between_structures(reference_pdb, frame_pdb):
    blocks = [(b[0].residue_name, [a.atom_name.upper() for a in b], [a.atom_index for a in b])
              for b in _residues(_iter_indexed_atoms(frame_pdb))]
    return _ordered_mapping(_iter_indexed_atoms(reference_pdb), blocks)


def require_mapped(mapping, indices):
    missing = sorted(set(indices) - mapping.keys())
    if missing:
        raise ValueError(f"No unique residue/atom correspondence for PDB atom(s) {missing}. "
                         "Use an unambiguous topology-derived reference; atom numbers were not guessed.")


def remap_candidates(candidates, mapping):
    require_mapped(mapping, [c.atom_index for c in candidates])
    return [replace(c, atom_index=mapping[c.atom_index],
                    donors=[replace(d, atom_index=mapping[d.atom_index]) for d in c.donors
                            if d.atom_index in mapping]) for c in candidates]


def assess_mapped_site_stability(reference_pdb, frame_pdb, candidate, **kwargs):
    """Keep the existing metric, but do not compare unrelated PDB row numbers."""
    try:
        mapping = map_between_structures(reference_pdb, frame_pdb)
        require_mapped(mapping, [candidate.atom_index, *(d.atom_index for d in candidate.donors)])
        frame_atoms = _iter_indexed_atoms(frame_pdb)
        projected = load_structure(reference_pdb).clone()
        index = 0
        for chain in projected[0]:
            for residue in chain:
                for atom in residue:
                    index += 1
                    if index in mapping:
                        atom.pos = frame_atoms[mapping[index] - 1].position
        return assess_site_stability(reference_pdb, projected, candidate, **kwargs)
    except ValueError as exc:
        return SiteStabilityAssessment(candidate.site, False, float("inf"), 0, len(candidate.donors),
                                       f"Reference/frame donor correspondence unavailable: {exc}")
