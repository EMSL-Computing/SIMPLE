"""Emit JSON {relative path: text} for the bundled, explicitly hybrid ethaline FF.

Development-only, no runtime downloads. Run from the repository root. The pinned
Amber source is hash checked; --gaff2 accepts an offline copy of the same file.
Output is reviewed/applied as a patch, not written over library files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
from urllib.request import urlopen

COMMIT = "919e8b895a4566669325aee14ecf540a5fcb1b89"
SOURCE = f"https://raw.githubusercontent.com/Amber-MD/AmberClassic/{COMMIT}/dat/leap/parm/gaff2.dat"
SHA256 = "2244b58627a85693776d12a0155a292f72b65b4398a03b87958136ad7460c67f"
DIRECTORY = "REF_DATA/Perkins2014_Ethaline_GAFF2"
# Alias: original GAFF2 type, element, mass, Rmin/2 [A], epsilon [kcal/mol].
# ZE deliberately retains the different O3 epsilon printed in SI Table S2.
TYPES = {
    "ZC": ("c3", "C", 12.01, 1.9080, 0.1094),
    "ZN": ("n4", "N", 14.01, 1.8240, 0.1700),
    "ZX": ("hx", "H", 1.008, 1.1000, 0.0157),
    "ZH": ("h1", "H", 1.008, 1.3870, 0.0157),
    "ZO": ("oh", "O", 16.00, 1.7210, 0.2104),
    "ZE": ("oh", "O", 16.00, 1.7210, 0.1700),
    "ZQ": ("ho", "H", 1.008, 0.1120, 0.0010),
    "ZI": (None, "Cl", 35.45, 2.4700, 0.1000),
}
CHOLINE = [
    ("C2", "ZC", -.1208), ("N3", "ZN", .0452),
    ("C3", "ZC", -.1208), ("C4", "ZC", -.1208),
    ("C5", "ZC", -.0290), ("C6", "ZC", .1351), ("O2", "ZO", -.5570),
    *((f"H{i}", "ZX", .1074) for i in range(5, 14)),
    ("H14", "ZX", .1004), ("H15", "ZX", .1004),
    ("H16", "ZH", .0459), ("H17", "ZH", .0459), ("H18", "ZQ", .4091),
]
GLYCOL = [
    ("O3", "ZE", -.6340), ("C7", "ZC", .1615), ("C8", "ZC", .1615),
    ("O4", "ZO", -.6340), ("H19", "ZQ", .4069),
    *((f"H{i}", "ZH", .0328) for i in range(20, 24)), ("H24", "ZQ", .4069),
]


def off_table(text: str, name: str) -> list[list[str]]:
    rows, active = [], False
    for line in text.splitlines():
        if line.startswith("!entry"):
            active = f".unit.{name} " in line
        elif active and line.strip():
            rows.append(shlex.split(line))
    return rows


def make_library(template: str, residue: str, atoms: list[tuple]) -> str:
    """Reuse coordinates/connectivity only; charges/types are the new model."""
    old = re.search(r"!entry\.([^.]+)\.", template).group(1)
    assert len(off_table(template, "atoms")) == len(atoms)
    lines, mode, index = [], "", 0
    for line in template.splitlines():
        if line.startswith("!entry"):
            mode = "atoms" if ".unit.atoms table" in line else (
                "pert" if ".unit.atomspertinfo " in line else "")
            index = 0
        elif mode and line.strip():
            fields = shlex.split(line)
            name, atom_type, charge = atoms[index]
            if mode == "atoms":
                line = f' "{name}" "{atom_type}" ' + " ".join(fields[2:-1]) + f" {charge:.8f}"
            else:
                line = f' "{name}" "{atom_type}" 0 -1 0.0'
            index += 1
        lines.append(line.replace(f"!entry.{old}.", f"!entry.{residue}.").replace(f'"{old}"', f'"{residue}"'))
    return "\n".join(lines) + "\n\n"


def choline_charge_rounding_correction() -> list[tuple]:
    """SI four-decimal values sum to +0.9002; preserve equivalent-atom groups."""
    correction = (.9 - sum(row[2] for row in CHOLINE)) / len(CHOLINE)
    corrected = [(name, typ, round(charge + correction, 8)) for name, typ, charge in CHOLINE]
    # Sub-1e-7 final text-rounding residual on the unique nitrogen, not one
    # member of an equivalent methyl-H group.
    name, typ, charge = corrected[1]
    corrected[1] = (name, typ, round(charge + .9 - sum(row[2] for row in corrected), 8))
    return corrected


def chloride_library() -> str:
    # Complete, standalone OFF residue; never use Amber's charge -1 CL unit.
    sections = {
        "atoms table  str name  str type  int typex  int resx  int flags  int seq  int elmnt  dbl chg": [' "CL" "ZI" 0 1 131072 1 17 -0.90000000'],
        "atomspertinfo table  str pname  str ptype  int ptypex  int pelmnt  dbl pchg": [' "CL" "ZI" 0 -1 0.0'],
        "boundbox array dbl": [" -1.0", " 0.0", " 0.0", " 0.0", " 0.0"],
        "childsequence single int": [" 2"],
        "connect array int": [" 0", " 0"],
        "connectivity table  int atom1x  int atom2x  int flags": [],
        "hierarchy table  str abovetype  int abovex  str belowtype  int belowx": [' "U" 0 "R" 1', ' "R" 1 "A" 1'],
        "name single str": [' "PL4"'],
        "positions table  dbl x  dbl y  dbl z": [" 0.0 0.0 0.0"],
        "residueconnect table  int c1x  int c2x  int c3x  int c4x  int c5x  int c6x": [" 0 0 0 0 0 0"],
        "residues table  str name  int seq  int childseq  int startatomx  str restype  int imagingx": [' "PL4" 1 2 1 "?" 0'],
        "residuesPdbSequenceNumber array int": [" 0"],
        "solventcap array dbl": [" -1.0", " 0.0", " 0.0", " 0.0", " 0.0"],
        "velocities table  dbl x  dbl y  dbl z": [" 0.0 0.0 0.0"],
    }
    lines = ['!!index array str', ' "PL4"']
    for header, rows in sections.items():
        lines.extend([f"!entry.PL4.unit.{header}", *rows])
    return "\n".join(lines) + "\n\n"


def bonded_paths(library: str, length: int) -> set[tuple[str, ...]]:
    atoms = off_table(library, "atoms")
    adjacency = {i: set() for i in range(len(atoms))}
    for first, second, _ in off_table(library, "connectivity"):
        i, j = int(first) - 1, int(second) - 1
        adjacency[i].add(j)
        adjacency[j].add(i)
    paths = [(i,) for i in adjacency]
    for _ in range(length - 1):
        paths = [(*path, j) for path in paths for j in adjacency[path[-1]] if j not in path]
    typed = [tuple(atoms[i][1] for i in path) for path in paths]
    return {min(path, path[::-1]) for path in typed}


def parse_bonded(source: str, length: int) -> dict[tuple[str, ...], list[list[str]]]:
    """Read fixed-width Amber parm records, keeping complete torsion series."""
    width = 3 * length - 1
    pattern = re.compile(r"^" + r"[A-Za-z0-9+ ]{2}-" * (length - 1) + r"[A-Za-z0-9+ ]{2}\s+")
    result = {}
    continuing = False
    previous = None
    for line in source.splitlines():
        if not pattern.match(line):
            continue
        key = tuple(line[i:i + 2].strip() for i in range(0, width, 3))
        fields = line[width:].split()
        try:
            if length == 4:
                # Proper: IDIVF, PK, PHASE, PN. Improper has only three
                # numeric fields and starts with a floating-point force constant.
                int(fields[0])
                values = fields[:4]
            else:
                values = fields[:2]
            for value in values:
                float(value)
        except (ValueError, IndexError):
            continue
        if length == 4 and continuing:
            assert key == previous, "Unexpected torsion continuation"
            result[key].append(values)
        else:
            result[key] = [values]
        continuing = length == 4 and float(values[-1]) < 0
        previous = key
    assert not continuing
    return result


def lookup(table: dict, types: tuple[str, ...]) -> tuple[tuple, list]:
    for key in (types, types[::-1]):
        if key in table:
            return key, table[key]
    matches = []
    for key, values in table.items():
        if any(all(a == b or a == "X" for a, b in zip(key, trial)) for trial in (types, types[::-1])):
            matches.append((key.count("X"), key, values))
    if not matches:
        raise ValueError(f"No GAFF2 bonded parameter for {types}")
    best = min(item[0] for item in matches)
    matches = [item for item in matches if item[0] == best]
    assert len(matches) == 1, f"Ambiguous GAFF2 wildcard match: {types}"
    return matches[0][1:]


def generate(source: bytes, root: Path) -> dict[str, str]:
    assert hashlib.sha256(source).hexdigest() == SHA256, "GAFF2 source/version changed"
    source_text = source.decode()
    libraries = {
        "PC4.lib": make_library((root / "REF_DATA/Choline/CH1_h_ptmpsi.lib").read_text(), "PC4", choline_charge_rounding_correction()),
        "PE4.lib": make_library((root / "REF_DATA/Ethylene-glycol/EG1_h_ptmpsi.lib").read_text(), "PE4", GLYCOL),
        "PL4.lib": chloride_library(),
    }
    lines = ["Perkins 2014 SI charges/LJ + GAFF2 2.2.30 bonded; IM chloride LJ (HYBRID)", "MASS"]
    for alias, (_, _, mass, _, _) in TYPES.items():
        lines.append(f"{alias} {mass:.4f}")
    bonded_sources = {}
    for length, section in ((2, "BOND"), (3, "ANGLE"), (4, "DIHE")):
        table = parse_bonded(source_text, length)
        lines.extend(["", section])
        paths = set().union(*(bonded_paths(lib, length) for lib in libraries.values()))
        for path in sorted(paths):
            original = tuple(TYPES[t][0] for t in path)
            key, terms = lookup(table, original)
            bonded_sources["-".join(path)] = {"gaff2_types": original, "source_key": key, "terms": terms}
            for term in terms:
                scales = " SCEE=1.2 SCNB=2.0" if length == 4 else ""
                lines.append("-".join(path) + "  " + "  ".join(term) + scales)
    # These molecules have no three-connected planar centers (all C/N sp3).
    lines.extend(["", "IMPROPER", "", "NONBON"])
    for alias, (_, _, _, radius, epsilon) in TYPES.items():
        lines.append(f"{alias}  {radius:.4f}  {epsilon:.4f}")
    lines.extend(["", ""])
    files = {**libraries, "perkins2014_gaff2.frcmod": "\n".join(lines)}
    provenance = {
        "id": "perkins2014_gaff2_hybrid_v1",
        "label": "Perkins 2014 SI charges/LJ + GAFF2 bonded (hybrid)",
        "citation": "Perkins, S. L.; Painter, P.; Colina, C. M. J. Chem. Eng. Data 2014, 59, 3652-3662.",
        "title": "Experimental and Computational Studies of Choline Chloride-Based Deep Eutectic Solvents",
        "doi": "10.1021/je500520h",
        "si": "je500520h_si_001.pdf, pp. 6-7, Tables S1-S2; bonded terms described as GAFF",
        "gaff2_source": SOURCE, "gaff2_sha256": SHA256, "gaff2_header": source_text.splitlines()[0],
        "chloride_lj_source": "Amber parm99.dat IM: Smith & Dang, J. Chem. Phys. 1994, 100, 3757; Rmin/2=2.47 A, epsilon=0.1 kcal/mol",
        "chloride_lj_url": f"https://raw.githubusercontent.com/Amber-MD/AmberClassic/{COMMIT}/dat/leap/parm/parm99.dat",
        "chloride_charge": -0.9,
        "chloride_charge_rationale": "Chosen to balance +0.9000 e choline after correcting +0.0002 e SI text-rounding drift; chloride LJ not specified in supplied SI.",
        "choline_si_charges": {name: charge for name, _, charge in CHOLINE},
        "choline_rounding_correction": {"original_total": 0.9002, "target_total": 0.9, "method": "Uniform -0.0002/21 e per atom, rounded to 8 decimals; final residual on unique N3. Equivalent-atom charges remain equal."},
        "recommended_components": ["choline_chloride_perkins2014_gaff2", "ethylene_glycol_perkins2014_gaff2"],
        "recommended_ratio": [1, 2],
        "units": {"charge": "e", "radius": "Rmin/2, angstrom", "epsilon": "kcal/mol"},
        "mixing_rule": "Amber Lorentz-Berthelot: additive Rmin/2, geometric epsilon",
        "scee": 1.2, "scnb": 2.0,
        "warnings": [
            "Hybrid model, NOT an exact reproduction of Perkins et al. 2014; density/RDF/transport properties have not been validated.",
            "Standard GAFF bonded terms were replaced with pinned GAFF2 2.2.30 terms, not a claim of improved DES accuracy.",
            "EG O3 epsilon=0.1700 and O4 epsilon=0.2104 are retained literally from SI Table S2; possible SI inconsistency is unresolved.",
            "Chloride LJ is an explicitly substituted Amber IM value, not confirmed as the authors' chloride model.",
            "Choline SI charges sum to +0.9002 e; a documented symmetry-preserving text-rounding correction gives +0.9000 e to balance Cl -0.9000 e.",
            "No 12-6-4/C4 model is supplied for these scaled-charge types. Do not combine with C4 ions without new parameter validation.",
            "Coordinates are starting geometries reused from SIMPLE, not published equilibrated coordinates. Minimize/equilibrate before production.",
        ],
        "atom_type_mapping": {alias: {"gaff2": value[0], "element": value[1], "mass": value[2], "rmin_half": value[3], "epsilon": value[4]} for alias, value in TYPES.items()},
        "atoms": {name: [{"name": row[0], "type": row[1], "charge": float(row[-1])} for row in off_table(lib, "atoms")] for name, lib in libraries.items()},
        "bonded_sources": bonded_sources,
        "asset_hash_normalization": "UTF-8/LF text; strip trailing whitespace at EOF, then append one LF",
        "asset_sha256": {name: hashlib.sha256((value.rstrip() + "\n").encode()).hexdigest() for name, value in files.items()},
    }
    files["provenance.json"] = json.dumps(provenance, indent=2) + "\n"
    return {f"{DIRECTORY}/{name}": value for name, value in files.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaff2", type=Path)
    args = parser.parse_args()
    source = args.gaff2.read_bytes() if args.gaff2 else urlopen(SOURCE, timeout=30).read()
    print(json.dumps(generate(source, Path(__file__).resolve().parents[1])))
