from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import tomllib

from pydantic import BaseModel, Field, model_validator
from tomlkit import document, nl, table

from amber_metallo.config import SlurmConfig
from amber_metallo.ti.config import (
    ComplexInputConfig,
    MetalSelectionConfig,
    SnapshotConfig,
    TIProtocolConfig,
    TIWorkflowConfig,
    WaterReferenceConfig,
)


class FreeEnergyMethod(StrEnum):
    TI = "ti"
    MMPBSA = "mmpbsa"


class MMPBSASolvationModel(StrEnum):
    GB = "gb"


class MMPBSAEntropyMethod(StrEnum):
    QHA = "qha"
    NMODE = "nmode"


class MMPBSALigandSelectionMode(StrEnum):
    METAL_SITE = "metal_site"
    RESIDUE_NAME = "residue_name"


class MMPBSAReceptorSelectionMode(StrEnum):
    AUTO = "auto"
    RESIDUE_NAME = "residue_name"


class FreeEnergyConfig(BaseModel):
    method: FreeEnergyMethod = FreeEnergyMethod.TI


class MMPBSAConfig(BaseModel):
    solvation_model: MMPBSASolvationModel | None = None
    run_gb: bool = True
    run_pb: bool = True
    include_entropy: bool = False
    entropy_method: MMPBSAEntropyMethod = MMPBSAEntropyMethod.QHA
    include_decomposition: bool = True
    decomposition_run_gb: bool = False
    decomposition_run_pb: bool = True
    decomposition_idecomp: int = Field(default=1, ge=1, le=4)
    decomposition_verbose: int = Field(default=1, ge=0)
    frame_stride: int = Field(default=1, ge=1)
    start_frame: int | None = Field(default=None, ge=1)
    end_frame: int | None = Field(default=None, ge=1)
    ligand_selection_mode: MMPBSALigandSelectionMode = MMPBSALigandSelectionMode.METAL_SITE
    ligand_residue_names: list[str] = Field(default_factory=list)
    receptor_selection_mode: MMPBSAReceptorSelectionMode = MMPBSAReceptorSelectionMode.AUTO
    receptor_residue_names: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def apply_legacy_solvation_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "run_gb" in data or "run_pb" in data:
            return data
        if data.get("solvation_model") == MMPBSASolvationModel.GB.value:
            payload = dict(data)
            payload["run_gb"] = True
            payload["run_pb"] = False
            return payload
        return data

    @model_validator(mode="after")
    def validate_frames(self) -> "MMPBSAConfig":
        if not self.run_gb and not self.run_pb:
            raise ValueError("mmpbsa must enable at least one solver via run_gb or run_pb")
        if self.start_frame is not None and self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("mmpbsa.end_frame must be >= mmpbsa.start_frame")
        self.ligand_residue_names = [
            str(name).strip().upper()
            for name in self.ligand_residue_names
            if str(name).strip()
        ]
        self.receptor_residue_names = [
            str(name).strip().upper()
            for name in self.receptor_residue_names
            if str(name).strip()
        ]
        if self.ligand_selection_mode == MMPBSALigandSelectionMode.RESIDUE_NAME and not self.ligand_residue_names:
            raise ValueError("mmpbsa.ligand_residue_names is required when ligand_selection_mode='residue_name'")
        if self.receptor_selection_mode == MMPBSAReceptorSelectionMode.RESIDUE_NAME and not self.receptor_residue_names:
            raise ValueError("mmpbsa.receptor_residue_names is required when receptor_selection_mode='residue_name'")
        overlap = set(self.ligand_residue_names) & set(self.receptor_residue_names)
        if overlap:
            raise ValueError(
                "mmpbsa ligand_residue_names and receptor_residue_names must not overlap: "
                + ", ".join(sorted(overlap))
            )
        return self

    def warning_messages(self) -> list[str]:
        warnings: list[str] = []
        if self.include_entropy and self.entropy_method == MMPBSAEntropyMethod.NMODE:
            warnings.append(
                "nmode entropy is enabled. It is very expensive and can fail with segmentation faults on larger systems."
            )
        return warnings

    def requested_solvers(self) -> list[str]:
        solvers: list[str] = []
        if self.run_gb:
            solvers.append("gb")
        if self.run_pb:
            solvers.append("pb")
        return solvers

    def decomposition_requested_solvers(self) -> list[str]:
        solvers: list[str] = []
        if self.include_decomposition and self.decomposition_run_gb and self.run_gb:
            solvers.append("gb")
        if self.include_decomposition and self.decomposition_run_pb and self.run_pb:
            solvers.append("pb")
        return solvers


class FreeEnergyWorkflowConfig(BaseModel):
    complex_input: ComplexInputConfig
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    metal: MetalSelectionConfig = Field(default_factory=MetalSelectionConfig)
    free_energy: FreeEnergyConfig = Field(default_factory=FreeEnergyConfig)
    ti: TIProtocolConfig = Field(default_factory=TIProtocolConfig)
    mmpbsa: MMPBSAConfig = Field(default_factory=MMPBSAConfig)
    water_reference: WaterReferenceConfig = Field(default_factory=WaterReferenceConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    output_dir: str = "."

    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    def warning_messages(self) -> list[str]:
        if self.free_energy.method == FreeEnergyMethod.MMPBSA:
            return self.mmpbsa.warning_messages()
        return []


def _section_from_model(payload: dict[str, Any]) -> Any:
    section = table()
    for key, value in payload.items():
        if value is None:
            continue
        section[key] = value
    return section


def load_config(path: str | Path) -> FreeEnergyWorkflowConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return FreeEnergyWorkflowConfig.model_validate(data)


def dump_config(config: FreeEnergyWorkflowConfig) -> str:
    doc = document()
    doc.add("complex_input", _section_from_model(config.complex_input.model_dump(mode="json")))
    doc.add(nl())
    doc.add("snapshot", _section_from_model(config.snapshot.model_dump(mode="json")))
    doc.add(nl())
    doc.add("metal", _section_from_model(config.metal.model_dump(mode="json")))
    doc.add(nl())
    doc.add("free_energy", _section_from_model(config.free_energy.model_dump(mode="json")))
    doc.add(nl())
    doc.add("ti", _section_from_model(config.ti.model_dump(mode="json")))
    doc.add(nl())
    doc.add("mmpbsa", _section_from_model(config.mmpbsa.model_dump(mode="json")))
    doc.add(nl())
    doc.add("water_reference", _section_from_model(config.water_reference.model_dump(mode="json")))
    doc.add(nl())
    doc.add("slurm", _section_from_model(config.slurm.model_dump(mode="json")))
    doc.add(nl())
    doc["output_dir"] = config.output_dir
    return doc.as_string()


def save_config(config: FreeEnergyWorkflowConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_config(config), encoding="utf-8")
    return target


def to_ti_config(config: FreeEnergyWorkflowConfig) -> TIWorkflowConfig:
    return TIWorkflowConfig(
        complex_input=config.complex_input.model_dump(mode="json"),
        snapshot=config.snapshot.model_dump(mode="json"),
        metal=config.metal.model_dump(mode="json"),
        ti=config.ti.model_dump(mode="json"),
        water_reference=config.water_reference.model_dump(mode="json"),
        slurm=config.slurm.model_dump(mode="json"),
        output_dir=config.output_dir,
    )


def from_ti_config(config: TIWorkflowConfig) -> FreeEnergyWorkflowConfig:
    return FreeEnergyWorkflowConfig(
        complex_input=config.complex_input.model_dump(mode="json"),
        snapshot=config.snapshot.model_dump(mode="json"),
        metal=config.metal.model_dump(mode="json"),
        free_energy={"method": FreeEnergyMethod.TI},
        ti=config.ti.model_dump(mode="json"),
        water_reference=config.water_reference.model_dump(mode="json"),
        slurm=config.slurm.model_dump(mode="json"),
        output_dir=config.output_dir,
    )
