from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from amber_metallo.c4_assets import opc_duvail_assets_available, opc_duvail_ion_frcmod

@dataclass(slots=True)
class BinaryStatus:
    name: str
    path: Path | None

    @property
    def found(self) -> bool:
        return self.path is not None


@dataclass(slots=True)
class AmberEnvironment:
    amberhome: Path | None
    binaries: dict[str, BinaryStatus] = field(default_factory=dict)
    leaprc_files: list[str] = field(default_factory=list)
    ions_frcmod: Path | None = None
    ion_frcmods: dict[str, Path] = field(default_factory=dict)

    def available_protein_force_fields(self) -> list[str]:
        ordered = ("ff19SB", "ff14SB", "ff99SB", "ff99SBildn")
        available = []
        for force_field in ordered:
            candidates = {
                f"leaprc.protein.{force_field}",
                f"oldff/leaprc.ff99SB",
                f"oldff/leaprc.ff99SBildn",
            }
            if any(candidate in self.leaprc_files for candidate in candidates):
                available.append(force_field)
        return available

    def available_small_molecule_force_fields(self) -> list[str]:
        ordered = ("gaff2", "gaff")
        available = []
        for force_field in ordered:
            if f"leaprc.{force_field}" in self.leaprc_files:
                available.append(force_field)
        return available

    def available_water_models(self) -> list[str]:
        discovered = []
        for item in self.leaprc_files:
            if not item.startswith("leaprc.water."):
                continue
            discovered.append(item.split("leaprc.water.", maxsplit=1)[1])

        preferred = ("opc", "spce", "tip3p", "opc3", "opc3pol", "tip4pew", "tip4pd", "tip5p")
        ordered = [name for name in preferred if name in discovered]
        ordered.extend(sorted(name for name in discovered if name not in ordered))
        return ordered

    def matching_1264_files(self, water_model: str, *, include_bundled_opc: bool = True) -> list[Path]:
        suffix = f"_{water_model.lower()}"
        matches = [path for name, path in self.ion_frcmods.items() if name.endswith(suffix) and "1264" in name]
        if include_bundled_opc and water_model.lower() == "opc" and opc_duvail_assets_available():
            matches.append(opc_duvail_ion_frcmod())
        return sorted(matches)

    def matching_126_files(self, water_model: str) -> list[Path]:
        suffix = f"_{water_model.lower()}"
        matches = [
            path
            for name, path in self.ion_frcmods.items()
            if name.endswith(suffix) and "_126_" in name and "1264" not in name
        ]
        return sorted(matches)

    def matching_monovalent_126_files(self, water_model: str) -> list[Path]:
        water_key = water_model.lower()
        names = (
            f"frcmod.ions1lm_126_{water_key}",
            f"frcmod.ionslm_126_{water_key}",
        )
        return [self.ion_frcmods[name] for name in names if name in self.ion_frcmods]

    def matching_multivalent_126_files(self, water_model: str) -> list[Path]:
        water_key = water_model.lower()
        names = (
            f"frcmod.ions234lm_126_{water_key}",
            f"frcmod.ionslm_126_{water_key}",
        )
        return [self.ion_frcmods[name] for name in names if name in self.ion_frcmods]

    def matching_monovalent_1264_files(
        self,
        water_model: str,
        *,
        include_bundled_opc: bool = True,
    ) -> list[Path]:
        water_key = water_model.lower()
        if include_bundled_opc and water_key == "opc" and opc_duvail_assets_available():
            return [opc_duvail_ion_frcmod()]
        names = (
            f"frcmod.ions1lm_1264_{water_key}",
            f"frcmod.ionslm_1264_{water_key}",
        )
        return [self.ion_frcmods[name] for name in names if name in self.ion_frcmods]

    def matching_multivalent_1264_files(
        self,
        water_model: str,
        *,
        include_bundled_opc: bool = True,
    ) -> list[Path]:
        water_key = water_model.lower()
        if include_bundled_opc and water_key == "opc" and opc_duvail_assets_available():
            return [opc_duvail_ion_frcmod()]
        names = (
            f"frcmod.ions234lm_1264_{water_key}",
            f"frcmod.ionslm_1264_{water_key}",
        )
        return [self.ion_frcmods[name] for name in names if name in self.ion_frcmods]

    def has_matching_1264(self, water_model: str) -> bool:
        return bool(self.matching_1264_files(water_model))

    def has_matching_126(self, water_model: str) -> bool:
        return bool(self.matching_126_files(water_model))

    def has_matching_monovalent_126(self, water_model: str) -> bool:
        return bool(self.matching_monovalent_126_files(water_model))

    def has_matching_multivalent_126(self, water_model: str) -> bool:
        return bool(self.matching_multivalent_126_files(water_model))

    def has_matching_monovalent_1264(self, water_model: str) -> bool:
        return bool(self.matching_monovalent_1264_files(water_model))

    def has_matching_multivalent_1264(self, water_model: str) -> bool:
        return bool(self.matching_multivalent_1264_files(water_model))


def is_linux_execution_host() -> bool:
    return os.name != "nt"


def infer_amberhome_from_binary(binary: Path | None) -> Path | None:
    if binary is None:
        return None
    parent = binary.resolve().parent
    if parent.name.lower() != "bin":
        return None
    amberhome = parent.parent
    if (amberhome / "dat" / "leap" / "cmd").exists():
        return amberhome
    return None


def detect_amber_environment() -> AmberEnvironment:
    binaries = {}
    binary_names = (
        "tleap",
        "antechamber",
        "parmchk2",
        "parmed",
        "cpptraj",
        "pmemd.MPI",
        "pmemd.cuda",
        "sander.MPI",
        "MMPBSA.py",
        "MMPBSA.py.MPI",
        "packmol",
    )
    for name in binary_names:
        raw_path = shutil.which(name)
        binaries[name] = BinaryStatus(name=name, path=Path(raw_path) if raw_path else None)

    amberhome = None
    env_value = os.environ.get("AMBERHOME")
    if env_value:
        candidate = Path(env_value).expanduser()
        if (candidate / "dat" / "leap" / "cmd").exists():
            amberhome = candidate

    if amberhome is None:
        amberhome = infer_amberhome_from_binary(binaries["tleap"].path)

    if amberhome is not None:
        bin_dir = amberhome / "bin"
        for name, status in binaries.items():
            if status.path is not None:
                continue
            candidate = bin_dir / name
            if candidate.exists():
                binaries[name] = BinaryStatus(name=name, path=candidate)

    leaprc_files: list[str] = []
    ions_frcmod = None
    ion_frcmods: dict[str, Path] = {}
    if amberhome:
        leap_cmd_dir = amberhome / "dat" / "leap" / "cmd"
        if leap_cmd_dir.exists():
            leaprc_files = sorted(
                str(path.relative_to(leap_cmd_dir)).replace("\\", "/")
                for path in leap_cmd_dir.rglob("leaprc*")
                if path.is_file()
            )
        parm_dir = amberhome / "dat" / "leap" / "parm"
        if parm_dir.exists():
            ion_frcmods = {
                path.name: path
                for path in parm_dir.glob("frcmod.ions*")
                if path.is_file()
            }
        ions_frcmod = ion_frcmods.get("frcmod.ions234lm_1264_spce")

    return AmberEnvironment(
        amberhome=amberhome,
        binaries=binaries,
        leaprc_files=leaprc_files,
        ions_frcmod=ions_frcmod,
        ion_frcmods=ion_frcmods,
    )


def environment_summary() -> dict[str, object]:
    amber_env = detect_amber_environment()
    return {
        "platform": platform.platform(),
        "linux_execution_supported": is_linux_execution_host(),
        "amberhome": str(amber_env.amberhome) if amber_env.amberhome else None,
        "binaries": {name: str(status.path) if status.path else None for name, status in amber_env.binaries.items()},
        "leaprc_files": amber_env.leaprc_files,
        "ions1264_frcmod": str(amber_env.ions_frcmod) if amber_env.ions_frcmod else None,
        "ion_frcmods": {name: str(path) for name, path in amber_env.ion_frcmods.items()},
    }
