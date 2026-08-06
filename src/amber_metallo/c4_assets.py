from __future__ import annotations

from pathlib import Path


def package_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def opc_duvail_dir() -> Path:
    return package_data_dir() / "opc_duvail"


def opc_duvail_ion_frcmod() -> Path:
    return opc_duvail_dir() / "frcmod.ionslm_1264_opc"


def opc_duvail_polarizability_file() -> Path:
    return opc_duvail_dir() / "lj_1264_pol_augmented.dat"


def opc_duvail_c4_file() -> Path:
    return opc_duvail_dir() / "c4_duvail.dat"


def _two_column_parameter_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = raw_line.split()
        if len(tokens) >= 2:
            keys.add(tokens[0])
    return keys


def _frcmod_nonbond_atom_types(path: Path) -> set[str]:
    if not path.exists():
        return set()
    atom_types: set[str] = set()
    in_nonbond = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() == "NONBON":
            in_nonbond = True
            continue
        if in_nonbond:
            atom_types.add(line.split()[0])
    return atom_types


def opc_duvail_supports_metal_charge(element: str, charge: int) -> bool:
    normalized_element = element.strip().title()
    normalized_charge = int(charge)
    c4_key = f"{normalized_element}{normalized_charge}"
    frcmod_atom_type = f"{normalized_element}{normalized_charge}+"
    return (
        c4_key in _two_column_parameter_keys(opc_duvail_c4_file())
        and frcmod_atom_type in _frcmod_nonbond_atom_types(opc_duvail_ion_frcmod())
    )


def opc_duvail_assets_available() -> bool:
    return all(
        path.exists()
        for path in (
            opc_duvail_ion_frcmod(),
            opc_duvail_polarizability_file(),
            opc_duvail_c4_file(),
        )
    )
