from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import tomllib

from pydantic import BaseModel, Field, model_validator
from tomlkit import document, nl, table

from amber_metallo.config import BoxShape, SlurmConfig


DEFAULT_WATER_REFERENCE_BUFFER_ANGSTROM = 18.0


def _default_charge_lambdas() -> list[float]:
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]


def _default_vdw_lambdas() -> list[float]:
    return [
        0.0,
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.975,
        1.0,
    ]


class SnapshotMode(StrEnum):
    LAST = "last"
    CLUSTER = "cluster"


class TIImplementationMode(StrEnum):
    AMBER_12_6_WORKAROUND = "amber_12_6_workaround"
    AMBER_12_6_4_GTI = "amber_12_6_4_gti"
    GROMACS_TABULATED_12_6_4 = "gromacs_tabulated_12_6_4"


class TIDecouplingMode(StrEnum):
    SPLIT_Q_VDW = "split_q_vdw"
    COMBINED_Q_VDW = "combined_q_vdw"


class TIMetalSelectionMode(StrEnum):
    SINGLE = "single"
    ONE_BY_ONE = "one_by_one"
    ALL_AT_ONCE = "all_at_once"


class ComplexInputConfig(BaseModel):
    prmtop_path: str
    trajectory_path: str
    reference_structure_path: str
    production_mdin_path: str | None = None
    production_restart_path: str | None = None


class SnapshotConfig(BaseModel):
    mode: SnapshotMode = SnapshotMode.LAST
    cluster_radius_angstrom: float = Field(default=6.0, gt=0.0)
    cluster_epsilon_angstrom: float = Field(default=2.0, gt=0.0)
    cluster_sieve: int = Field(default=10, ge=1)
    allow_unstable_last_snapshot: bool = False
    diffusion_cutoff_angstrom: float = Field(default=2.5, gt=0.0)
    donor_cutoff_angstrom: float = Field(default=3.0, gt=0.0)
    retained_donor_cutoff_angstrom: float = Field(default=3.5, gt=0.0)


class MetalSelectionConfig(BaseModel):
    selection_mode: TIMetalSelectionMode = TIMetalSelectionMode.SINGLE
    selected_site: int | None = Field(default=None, ge=1)
    selected_sites: list[int] = Field(default_factory=list)
    formal_charge: int | None = Field(default=None, ge=1, le=4)
    formal_charges_by_site: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_selection(self) -> "MetalSelectionConfig":
        self.selected_sites = sorted({int(site) for site in self.selected_sites})
        if any(site < 1 for site in self.selected_sites):
            raise ValueError("metal.selected_sites entries must be >= 1")
        self.formal_charges_by_site = {
            int(site): int(charge)
            for site, charge in self.formal_charges_by_site.items()
            if int(site) >= 1
        }
        invalid_charges = [
            charge
            for charge in self.formal_charges_by_site.values()
            if charge < 1 or charge > 4
        ]
        if invalid_charges:
            raise ValueError("metal.formal_charges_by_site values must be between 1 and 4")
        if self.selected_site is not None and not self.selected_sites:
            self.selected_sites = [int(self.selected_site)]
        if self.selected_sites and self.selected_site is None and self.selection_mode == TIMetalSelectionMode.SINGLE:
            self.selected_site = self.selected_sites[0]
        if self.selection_mode == TIMetalSelectionMode.ALL_AT_ONCE and len(self.selected_sites) < 1:
            raise ValueError("metal.selected_sites is required when selection_mode='all_at_once'")
        return self


class TIProtocolConfig(BaseModel):
    implementation_mode: TIImplementationMode = TIImplementationMode.AMBER_12_6_WORKAROUND
    decoupling_mode: TIDecouplingMode = TIDecouplingMode.SPLIT_Q_VDW
    production_time_ns: float = Field(default=1.0, gt=0.0)
    charge_lambdas: list[float] = Field(default_factory=_default_charge_lambdas)
    vdw_lambdas: list[float] = Field(default_factory=_default_vdw_lambdas)
    qoff_dt_ps: float = Field(default=0.001, gt=0.0)
    vdwoff_dt_ps: float = Field(default=0.001, gt=0.0)
    bound_start_min_cycles: int = Field(default=5000, ge=100)
    bound_start_eq_ns: float = Field(default=0.05, gt=0.0)
    bound_start_eq_dt_ps: float = Field(default=0.001, gt=0.0)
    qoff_endpoint_min_cycles: int = Field(default=5000, ge=100)
    qoff_endpoint_eq_ns: float = Field(default=0.05, gt=0.0)
    restraint_force_constant: float = Field(default=5.0, gt=0.0)
    restraint_half_width_angstrom: float = Field(default=0.5, gt=0.0)
    restraint_anchor_count: int = Field(default=3, ge=1, le=8)
    scalpha: float = Field(default=0.5, gt=0.0)
    scbeta: float = Field(default=12.0, gt=0.0)
    logdvdl: bool = True

    @model_validator(mode="after")
    def validate_lambdas(self) -> "TIProtocolConfig":
        self.charge_lambdas = _validate_lambda_schedule(self.charge_lambdas, label="charge_lambdas")
        self.vdw_lambdas = _validate_lambda_schedule(self.vdw_lambdas, label="vdw_lambdas")
        return self


class WaterReferenceConfig(BaseModel):
    enabled: bool = True
    bound_in_place: bool = False
    water_model: str = "opc"
    box_shape: BoxShape = BoxShape.OCT
    buffer_angstrom: float = Field(default=DEFAULT_WATER_REFERENCE_BUFFER_ANGSTROM, gt=0.0)
    cache_dir: str | None = None
    reuse_existing: bool = True
    reuse_from_library: bool = False
    library_key: str | None = None
    custom_ion_frcmods: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "WaterReferenceConfig":
        self.water_model = self.water_model.strip().lower()
        self.custom_ion_frcmods = [item.strip() for item in self.custom_ion_frcmods if item and item.strip()]
        if self.library_key is not None:
            self.library_key = self.library_key.strip() or None
        if not self.reuse_from_library:
            self.library_key = None
        return self


class TIWorkflowConfig(BaseModel):
    complex_input: ComplexInputConfig
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    metal: MetalSelectionConfig = Field(default_factory=MetalSelectionConfig)
    ti: TIProtocolConfig = Field(default_factory=TIProtocolConfig)
    water_reference: WaterReferenceConfig = Field(default_factory=WaterReferenceConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    output_dir: str = "."

    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


def _validate_lambda_schedule(values: list[float], *, label: str) -> list[float]:
    if not values:
        raise ValueError(f"ti.{label} cannot be empty")
    normalized = [round(float(value), 3) for value in values]
    if normalized != sorted(normalized):
        raise ValueError(f"ti.{label} must be sorted in ascending order")
    if normalized[0] != 0.0 or normalized[-1] != 1.0:
        raise ValueError(f"ti.{label} must start at 0.0 and end at 1.0")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"ti.{label} entries must be unique")
    return normalized


def load_config(path: str | Path) -> TIWorkflowConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return TIWorkflowConfig.model_validate(data)


def _section_from_model(payload: dict[str, Any]) -> Any:
    section = table()
    for key, value in payload.items():
        if value is None:
            continue
        section[key] = value
    return section


def dump_config(config: TIWorkflowConfig) -> str:
    doc = document()
    doc.add("complex_input", _section_from_model(config.complex_input.model_dump(mode="json")))
    doc.add(nl())
    doc.add("snapshot", _section_from_model(config.snapshot.model_dump(mode="json")))
    doc.add(nl())
    doc.add("metal", _section_from_model(config.metal.model_dump(mode="json")))
    doc.add(nl())
    doc.add("ti", _section_from_model(config.ti.model_dump(mode="json")))
    doc.add(nl())
    doc.add("water_reference", _section_from_model(config.water_reference.model_dump(mode="json")))
    doc.add(nl())
    doc.add("slurm", _section_from_model(config.slurm.model_dump(mode="json")))
    doc.add(nl())
    doc["output_dir"] = config.output_dir
    return doc.as_string()


def save_config(config: TIWorkflowConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_config(config), encoding="utf-8")
    return target
