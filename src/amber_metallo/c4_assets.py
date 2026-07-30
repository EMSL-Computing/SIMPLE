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


def opc_duvail_assets_available() -> bool:
    return all(
        path.exists()
        for path in (
            opc_duvail_ion_frcmod(),
            opc_duvail_polarizability_file(),
            opc_duvail_c4_file(),
        )
    )
