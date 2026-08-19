from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re
from typing import Any

import tomllib

from pydantic import BaseModel, Field, field_validator, model_validator
from tomlkit import document, nl, table


class InputSource(StrEnum):
    PDB_FILE = "pdb_file"
    PDB_ID = "pdb_id"
    SMALL_MOLECULE = "small_molecule"
    DES = "deep_eutectic_solvent"


class DESComponent(StrEnum):
    N8888_BR = "n8888_br"
    HEXANOIC_ACID = "hexanoic_acid"
    CHOLINE_CHLORIDE = "choline_chloride"
    ETHYLENE_GLYCOL = "ethylene_glycol"
    ACETONE = "acetone"
    ETHANOL = "ethanol"
    METHANOL = "methanol"


class DESC4ParameterSet(StrEnum):
    OPC_DUVAIL = "opc_duvail"
    SPCE_LIMERZ = "spce_limerz"


class DESMixingMode(StrEnum):
    RANDOM_MIX = "random_mix"
    # Kept only so older TOML files continue to load; validation upgrades it
    # to RANDOM_MIX and it is no longer shown as a separate user option.
    REPLICATE = "replicate"
    PACKMOL = "packmol"


class DESReplicateOrder(StrEnum):
    UNIFORM = "uniform"
    RANDOM = "random"
    GROUPED = "grouped"


class DESSizeMode(StrEnum):
    RATIO_UNITS = "ratio_units"
    BOX_LENGTH = "box_length"


class LigandMode(StrEnum):
    GAFF2 = "gaff2"
    GAFF = "gaff"
    MANUAL = "manual"


class ChargeMethod(StrEnum):
    FULL_RESP = "full_resp"
    RESP_ANTECHAMBER = "resp_antechamber"
    ANTECHAMBER = "antechamber"
    BCC = "bcc"
    RESP = "resp"


_CHARGE_METHOD_LEGACY_MAP = {
    "bcc": ChargeMethod.ANTECHAMBER,
    "am1-bcc": ChargeMethod.ANTECHAMBER,
    "am1_bcc": ChargeMethod.ANTECHAMBER,
    "antechamber": ChargeMethod.ANTECHAMBER,
    "resp": ChargeMethod.RESP_ANTECHAMBER,
    "resp+antechamber": ChargeMethod.RESP_ANTECHAMBER,
    "resp-antechamber": ChargeMethod.RESP_ANTECHAMBER,
    "resp_antechamber": ChargeMethod.RESP_ANTECHAMBER,
    "full resp": ChargeMethod.FULL_RESP,
    "full-resp": ChargeMethod.FULL_RESP,
    "full_resp": ChargeMethod.FULL_RESP,
}


def normalize_charge_method(value: ChargeMethod | str) -> ChargeMethod:
    token = value.value if isinstance(value, ChargeMethod) else str(value)
    normalized = _CHARGE_METHOD_LEGACY_MAP.get(token.strip().lower())
    if normalized is not None:
        return normalized
    return ChargeMethod(token)


def charge_method_uses_resp(value: ChargeMethod | str) -> bool:
    return normalize_charge_method(value) in {
        ChargeMethod.FULL_RESP,
        ChargeMethod.RESP_ANTECHAMBER,
    }


class RespApplyMode(StrEnum):
    DETECT = "detect"
    APPLY_EXISTING = "apply_existing"
    REBUILD = "rebuild"
    NEW_DIRECTORY = "new_directory"


class ProteinSiteRespMode(StrEnum):
    STANDARD_FF = "standard_ff"
    RESP = "resp"


class ProteinSiteRespScope(StrEnum):
    SIDECHAIN = "sidechain"
    WHOLE_RESIDUE = "whole_residue"


class MetalModel(StrEnum):
    MODEL_1264 = "1264"
    MCPB = "mcpb"
    QM = "qm"


class MetalAnchorMode(StrEnum):
    DONOR_ATOMS = "donor_atoms"
    RESIDUE_DONORS = "residue_donors"
    ATOM_SERIALS = "atom_serials"
    XYZ = "xyz"


class BoxShape(StrEnum):
    OCT = "oct"
    CUBIC = "cubic"
    BOX = "box"


class SaltKind(StrEnum):
    NONE = "none"
    NACL = "NaCl"
    CACL2 = "CaCl2"
    KCL = "KCl"


class SaltMode(StrEnum):
    NONE = "none"
    NEUTRALIZE = "neutralize"
    COUNT = "count"
    CONCENTRATION = "concentration"


class NeutralizationIon(StrEnum):
    AUTO = "auto"
    SALT_DEFAULT = "salt_default"
    SODIUM = "Na+"
    POTASSIUM = "K+"
    CHLORIDE = "Cl-"
    BROMIDE = "Br-"


class ProtocolKind(StrEnum):
    FOUR_STEP = "4step"
    FIFTEEN_STEP = "15step"
    DES_SOLVENT = "des_solvent"


class ResidueMaskNumbering(StrEnum):
    PDB = "pdb"
    PREPARED = "prepared"


class SlurmProfile(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class ProtonationEngine(StrEnum):
    PROPKA = "propka"


class MetalReplacement(BaseModel):
    site: int = Field(..., ge=1)
    target: str

    @model_validator(mode="after")
    def normalize(self) -> "MetalReplacement":
        self.target = self.target.strip().title()
        return self


class MetalInsertion(BaseModel):
    element: str
    charge: int | None = Field(default=None, ge=1, le=4)
    anchor_mode: MetalAnchorMode = MetalAnchorMode.DONOR_ATOMS
    anchors: list[str] = Field(default_factory=list)
    target_coordination_number: int | None = Field(default=None, ge=1, le=12)
    coordinates: list[float] | None = None
    label: str | None = None

    @model_validator(mode="after")
    def normalize(self) -> "MetalInsertion":
        self.element = self.element.strip().title()
        self.anchors = [str(anchor).strip() for anchor in self.anchors if str(anchor).strip()]
        self.label = (self.label or "").strip() or None
        if self.coordinates is not None:
            if len(self.coordinates) != 3:
                raise ValueError("metal insertion coordinates must contain exactly three values")
            self.coordinates = [float(value) for value in self.coordinates]
        if self.anchor_mode == MetalAnchorMode.XYZ:
            if self.coordinates is None:
                raise ValueError("coordinates are required when anchor_mode='xyz'")
        elif not self.anchors:
            raise ValueError("anchors are required unless anchor_mode='xyz'")
        return self


class MetalChargeAssignment(BaseModel):
    site: int = Field(..., ge=1)
    charge: int = Field(..., ge=1, le=4)


class DESMetalSiteConfig(BaseModel):
    element: str
    charge: int = Field(..., ge=1, le=4)
    count: int = Field(default=1, ge=1)
    coordinates: list[list[float]] | None = None

    @model_validator(mode="after")
    def normalize(self) -> "DESMetalSiteConfig":
        self.element = self.element.strip().title()
        if not self.element:
            raise ValueError("des.metal_sites.element is required")
        if self.coordinates is not None:
            normalized: list[list[float]] = []
            for coordinate in self.coordinates:
                if len(coordinate) != 3:
                    raise ValueError("des.metal_sites.coordinates entries must contain exactly three values")
                normalized.append([float(value) for value in coordinate])
            if len(normalized) > self.count:
                raise ValueError("des.metal_sites.coordinates cannot contain more entries than des.metal_sites.count")
            self.coordinates = normalized
        return self


class InputConfig(BaseModel):
    source: InputSource = InputSource.PDB_FILE
    path: str | None = None
    pdb_id: str | None = None
    small_molecule_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "InputConfig":
        if self.source == InputSource.PDB_FILE and not self.path:
            raise ValueError("input.path is required when source='pdb_file'")
        if self.source == InputSource.PDB_ID and not self.pdb_id:
            raise ValueError("input.pdb_id is required when source='pdb_id'")
        if self.source == InputSource.SMALL_MOLECULE and not self.small_molecule_files:
            raise ValueError(
                "input.small_molecule_files is required when source='small_molecule'"
            )
        return self


class DESConfig(BaseModel):
    components: list[DESComponent | str] = Field(default_factory=list)
    ratios: list[int] = Field(default_factory=list)
    mixing_mode: DESMixingMode = DESMixingMode.RANDOM_MIX
    replicate_order: DESReplicateOrder = DESReplicateOrder.UNIFORM
    size_mode: DESSizeMode = DESSizeMode.RATIO_UNITS
    ratio_units: int | None = Field(default=None, ge=1)
    box_length_angstrom: float | None = Field(default=None, gt=0.0)
    spacing_angstrom: float = Field(default=1.3, ge=1.2)
    packmol_tolerance_angstrom: float = Field(default=2.0, gt=0.0)
    packmol_fill_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    target_density_g_ml: float = Field(
        default=0.40,
        gt=0.0,
        le=2.5,
        description="Safe pre-equilibration mass density used to size random-mix and Packmol DES boxes.",
    )
    ref_data_dir: str = "REF_DATA"
    apply_1264: bool = True
    c4_parameter_set: DESC4ParameterSet = DESC4ParameterSet.OPC_DUVAIL
    central_metal_enabled: bool = False
    central_metal_element: str | None = None
    central_metal_charge: int | None = Field(default=None, ge=1, le=4)
    metal_sites: list[DESMetalSiteConfig] = Field(default_factory=list)
    metal_spacing_angstrom: float = Field(default=8.0, gt=0.0)

    @model_validator(mode="after")
    def validate_des_settings(self) -> "DESConfig":
        if self.mixing_mode == DESMixingMode.REPLICATE:
            self.mixing_mode = DESMixingMode.RANDOM_MIX
        if self.mixing_mode == DESMixingMode.RANDOM_MIX:
            self.replicate_order = DESReplicateOrder.RANDOM
        unique_components: list[DESComponent | str] = []
        for component in self.components:
            key = component.value if isinstance(component, DESComponent) else str(component).strip()
            if key and key not in [item.value if isinstance(item, DESComponent) else str(item) for item in unique_components]:
                unique_components.append(component)
        self.components = unique_components

        if self.components and not self.ratios:
            self.ratios = [1 for _ in self.components]
        if len(self.ratios) != len(self.components):
            raise ValueError("des.ratios must contain one value per selected DES component")
        if any(int(value) < 1 for value in self.ratios):
            raise ValueError("des.ratios values must be positive integers")
        self.ratios = [int(value) for value in self.ratios]

        if not self.components:
            return self

        if self.size_mode == DESSizeMode.RATIO_UNITS:
            if self.ratio_units is None:
                raise ValueError("des.ratio_units is required when des.size_mode='ratio_units'")
            self.box_length_angstrom = None
        elif self.size_mode == DESSizeMode.BOX_LENGTH:
            if self.box_length_angstrom is None:
                raise ValueError("des.box_length_angstrom is required when des.size_mode='box_length'")
            self.ratio_units = None

        self.metal_sites = [DESMetalSiteConfig.model_validate(site) for site in self.metal_sites]

        if not self.central_metal_enabled:
            self.central_metal_element = None
            self.central_metal_charge = None
            return self

        self.central_metal_element = (self.central_metal_element or "").strip().title() or None
        if self.central_metal_element is None:
            raise ValueError("des.central_metal_element is required when des.central_metal_enabled is true")
        if self.central_metal_charge is None:
            raise ValueError("des.central_metal_charge is required when des.central_metal_enabled is true")
        self.metal_sites = [
            DESMetalSiteConfig(
                element=self.central_metal_element,
                charge=self.central_metal_charge,
                count=1,
            ),
            *self.metal_sites,
        ]
        return self


class PrepareConfig(BaseModel):
    remove_waters: bool = True
    remove_other_hetero: bool = False
    remove_metals: bool = False
    repair_missing_loops: bool = False
    kept_ligands: list[str] = Field(default_factory=list)
    metal_replacements: list[MetalReplacement] = Field(default_factory=list)
    metal_deletions: list[int] = Field(default_factory=list)
    metal_insertions: list[MetalInsertion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metal_actions(self) -> "PrepareConfig":
        self.metal_deletions = sorted({int(site) for site in self.metal_deletions})
        if any(site < 1 for site in self.metal_deletions):
            raise ValueError("prepare.metal_deletions entries must be >= 1")

        replacement_sites = [item.site for item in self.metal_replacements]
        if self.remove_metals and replacement_sites:
            raise ValueError("prepare.metal_replacements cannot be used when remove_metals is true")
        if set(replacement_sites) & set(self.metal_deletions):
            raise ValueError("prepare.metal_deletions cannot overlap with prepare.metal_replacements")
        if self.remove_metals:
            self.metal_deletions = []
        return self


class ProtonationChange(BaseModel):
    chain: str
    seqid: str
    original_residue_name: str
    target_residue_name: str
    predicted_pka: float
    metal_near: bool = False

    @model_validator(mode="after")
    def normalize(self) -> "ProtonationChange":
        self.chain = self.chain.strip()
        self.seqid = self.seqid.strip()
        self.original_residue_name = self.original_residue_name.strip().upper()
        self.target_residue_name = self.target_residue_name.strip().upper()
        return self


class ProtonationConfig(BaseModel):
    enabled: bool = False
    ph: float | None = None
    engine: ProtonationEngine = ProtonationEngine.PROPKA
    selected_changes: list[ProtonationChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enabled_state(self) -> "ProtonationConfig":
        unique_changes: dict[tuple[str, str, str], ProtonationChange] = {}
        for change in self.selected_changes:
            key = (change.chain, change.seqid, change.target_residue_name)
            unique_changes[key] = ProtonationChange.model_validate(change)
        self.selected_changes = list(unique_changes.values())

        if not self.enabled:
            self.ph = None
            self.selected_changes = []
            return self
        if self.ph is None:
            raise ValueError("protonation.ph is required when protonation.enabled is true")
        return self


class LigandParameterAssignment(BaseModel):
    residue_name: str
    net_charge: int = 0
    multiplicity: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def normalize(self) -> "LigandParameterAssignment":
        self.residue_name = self.residue_name.strip().upper()
        return self


class LigandsConfig(BaseModel):
    mode: LigandMode = LigandMode.GAFF2
    charge_method: ChargeMethod = ChargeMethod.ANTECHAMBER
    manual_files: list[str] = Field(default_factory=list)
    residue_name: str = "LIG"
    net_charge: int = 0
    multiplicity: int = Field(default=1, ge=1)
    parameter_assignments: list[LigandParameterAssignment] = Field(default_factory=list)
    resp_job_dir: str | None = None
    resp_group_file: str | None = None
    resp_session_file: str | None = None
    resp_apply_mode: RespApplyMode = RespApplyMode.DETECT

    @field_validator("charge_method", mode="before")
    @classmethod
    def normalize_charge_method_field(cls, value: ChargeMethod | str) -> ChargeMethod:
        return normalize_charge_method(value)

    @model_validator(mode="after")
    def validate_parameter_assignments(self) -> "LigandsConfig":
        self.charge_method = normalize_charge_method(self.charge_method)
        assignment_map = {
            item.residue_name: LigandParameterAssignment(
                residue_name=item.residue_name,
                net_charge=int(item.net_charge),
                multiplicity=int(item.multiplicity),
            )
            for item in self.parameter_assignments
        }
        self.parameter_assignments = [assignment_map[name] for name in sorted(assignment_map)]
        self.residue_name = self.residue_name.strip().upper()
        self.resp_job_dir = (self.resp_job_dir or "").strip() or None
        self.resp_group_file = (self.resp_group_file or "").strip() or None
        self.resp_session_file = (self.resp_session_file or "").strip() or None
        return self


class SaltConfig(BaseModel):
    kind: SaltKind = SaltKind.NONE
    mode: SaltMode = SaltMode.NONE
    value: float | int | None = None
    neutralization_ion: NeutralizationIon = NeutralizationIon.AUTO

    @model_validator(mode="after")
    def validate_value(self) -> "SaltConfig":
        if self.kind == SaltKind.NONE or self.mode == SaltMode.NONE:
            self.kind = SaltKind.NONE
            self.mode = SaltMode.NONE
            self.value = 0
            self.neutralization_ion = NeutralizationIon.AUTO
            return self
        if self.mode == SaltMode.NEUTRALIZE:
            self.value = 0
            return self
        if self.value is None:
            raise ValueError("system.salt.value is required when salt.kind is not 'none'")
        if self.mode == SaltMode.COUNT and int(self.value) < 0:
            raise ValueError("salt count must be non-negative")
        if self.mode == SaltMode.CONCENTRATION and float(self.value) < 0:
            raise ValueError("salt concentration must be non-negative")
        return self


class SystemConfig(BaseModel):
    protein_ff: str = "ff19SB"
    ligand_ff: str = "gaff2"
    metal_model: MetalModel = MetalModel.MODEL_1264
    apply_1264: bool = True
    c4_parameter_set: DESC4ParameterSet = DESC4ParameterSet.OPC_DUVAIL
    metal_charges: list[MetalChargeAssignment] = Field(default_factory=list)
    water_model: str = "opc"
    box_shape: BoxShape = BoxShape.OCT
    buffer_angstrom: float = Field(default=10.0, gt=0.0)
    salt: SaltConfig = Field(default_factory=SaltConfig)
    custom_ion_frcmods: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_c4_parameter_set(cls, value: Any) -> Any:
        if isinstance(value, dict) and "c4_parameter_set" not in value:
            payload = dict(value)
            water_model = str(payload.get("water_model") or "opc").strip().lower()
            payload["c4_parameter_set"] = (
                DESC4ParameterSet.OPC_DUVAIL.value
                if water_model == "opc"
                else DESC4ParameterSet.SPCE_LIMERZ.value
            )
            return payload
        return value

    @model_validator(mode="after")
    def validate_metal_charge_assignments(self) -> "SystemConfig":
        charge_map = {int(item.site): MetalChargeAssignment(site=int(item.site), charge=int(item.charge)) for item in self.metal_charges}
        self.metal_charges = [charge_map[site] for site in sorted(charge_map)]
        return self


class MDConfig(BaseModel):
    protocol: ProtocolKind = ProtocolKind.FOUR_STEP
    temperature_k: float = Field(default=300.0, gt=0.0)
    pressure_bar: float = Field(default=1.0, gt=0.0)
    production_time_ns: float = Field(default=10.0, gt=0.0)
    des_mixing_enabled: bool = True
    des_mixing_temperature_k: float = Field(default=500.0, gt=0.0)
    des_mixing_time_ns: float = Field(default=50.0, gt=0.0)
    restraint_mask_override: str | None = None
    restraint_weight_override: float | None = Field(default=None, gt=0.0)
    focused_restraint_mask: str | None = None
    focused_restraint_mask_numbering: ResidueMaskNumbering = ResidueMaskNumbering.PDB
    focused_restraint_weight: float | None = Field(default=None, gt=0.0)
    stage_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_focused_restraints(self) -> "MDConfig":
        has_mask = bool((self.focused_restraint_mask or "").strip())
        has_weight = self.focused_restraint_weight is not None
        if has_weight and not has_mask:
            raise ValueError(
                "md.focused_restraint_weight requires md.focused_restraint_mask. "
                "Provide both fields or omit both."
            )
        if has_mask and not has_weight:
            raise ValueError(
                "md.focused_restraint_mask requires md.focused_restraint_weight. "
                "Provide both fields or omit both."
            )
        if self.focused_restraint_mask is not None:
            self.focused_restraint_mask = self.focused_restraint_mask.strip() or None
        if self.focused_restraint_mask is None:
            self.focused_restraint_mask_numbering = ResidueMaskNumbering.PDB
        return self


_SLURM_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SLURM_WALLTIME_RE = re.compile(r"^(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$")
_SLURM_EXECUTABLE_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")


def validate_slurm_identifier(value: str, *, field_name: str = "Slurm value") -> str:
    """Return a value that is safe to interpolate into an ``#SBATCH`` line."""

    normalized = str(value).strip()
    if not normalized or not _SLURM_IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"{field_name} may contain only letters, digits, '.', '_' and '-'; "
            "whitespace, newlines, and shell metacharacters are not allowed."
        )
    return normalized


class SlurmConfig(BaseModel):
    profile: SlurmProfile = SlurmProfile.CPU
    partition: str | None = None
    account: str | None = None
    nodes: int | None = Field(default=None, ge=1)
    ntasks: int = Field(default=8, ge=1)
    gpus: int = Field(default=1, ge=0)
    walltime: str = "24:00:00"
    binary_override: str | None = None
    job_name: str = "simple"

    @field_validator("partition", "account", mode="before")
    @classmethod
    def validate_optional_identifiers(cls, value: object, info: Any) -> str | None:
        if value is None or not str(value).strip():
            return None
        return validate_slurm_identifier(str(value), field_name=f"slurm.{info.field_name}")

    @field_validator("job_name", mode="before")
    @classmethod
    def validate_job_name(cls, value: object) -> str:
        return validate_slurm_identifier(str(value or "simple"), field_name="slurm.job_name")

    @field_validator("walltime", mode="before")
    @classmethod
    def validate_walltime(cls, value: object) -> str:
        normalized = str(value).strip()
        if not _SLURM_WALLTIME_RE.fullmatch(normalized):
            raise ValueError("slurm.walltime must use HH:MM:SS or D-HH:MM:SS format.")
        return normalized

    @field_validator("binary_override", mode="before")
    @classmethod
    def validate_binary_override(cls, value: object) -> str | None:
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip()
        if not _SLURM_EXECUTABLE_RE.fullmatch(normalized):
            raise ValueError(
                "slurm.binary_override must be a single executable name or POSIX path without "
                "whitespace, newlines, variable expansion, or shell metacharacters."
            )
        return normalized


class ProteinSiteRespClusterConfig(BaseModel):
    metal_sites: list[int] = Field(default_factory=list)
    donor_residues: list[str] = Field(default_factory=list)
    fixed_environment: list[str] = Field(default_factory=list)
    multiplicity: int | None = Field(default=None, ge=1)
    job_dir: str | None = None

    @model_validator(mode="after")
    def normalize(self) -> "ProteinSiteRespClusterConfig":
        self.metal_sites = sorted({int(site) for site in self.metal_sites if int(site) > 0})
        self.donor_residues = sorted({item.strip() for item in self.donor_residues if item.strip()})
        self.fixed_environment = sorted({item.strip() for item in self.fixed_environment if item.strip()})
        self.job_dir = (self.job_dir or "").strip() or None
        return self


class ProteinSiteRespConfig(BaseModel):
    mode: ProteinSiteRespMode = ProteinSiteRespMode.STANDARD_FF
    scope: ProteinSiteRespScope = ProteinSiteRespScope.SIDECHAIN
    apply_mode: RespApplyMode = RespApplyMode.DETECT
    default_multiplicity: int | None = Field(default=None, ge=1)
    search_roots: list[str] = Field(default_factory=list)
    job_dirs: list[str] = Field(default_factory=list)
    review_clusters: bool = False
    # Internal continuation flag used when an existing RESP result is selected
    # from the interactive Protein start screen.  In this mode SIMPLE preserves
    # the prepared protein/reference topology, builds the deferred final solvated
    # system when needed, and then applies the reviewed RESP charges in place.
    resume_existing_system: bool = False
    clusters: list[ProteinSiteRespClusterConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "ProteinSiteRespConfig":
        if self.apply_mode == RespApplyMode.REBUILD:
            raise ValueError(
                "protein_site_resp.apply_mode must be detect, new_directory, or apply_existing"
            )
        self.search_roots = sorted({item.strip() for item in self.search_roots if item.strip()})
        self.job_dirs = sorted({item.strip() for item in self.job_dirs if item.strip()})
        self.clusters = [ProteinSiteRespClusterConfig.model_validate(item) for item in self.clusters]
        return self


class WorkflowConfig(BaseModel):
    input: InputConfig
    des: DESConfig = Field(default_factory=DESConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    protonation: ProtonationConfig = Field(default_factory=ProtonationConfig)
    protein_site_resp: ProteinSiteRespConfig = Field(default_factory=ProteinSiteRespConfig)
    ligands: LigandsConfig = Field(default_factory=LigandsConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    md: MDConfig = Field(default_factory=MDConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    output_dir: str = "."

    @model_validator(mode="after")
    def validate_protonation_scope(self) -> "WorkflowConfig":
        if self.input.source in {InputSource.SMALL_MOLECULE, InputSource.DES} and self.protonation.enabled:
            raise ValueError("protonation is only supported for protein workflows")
        if self.input.source in {InputSource.SMALL_MOLECULE, InputSource.DES} and self.protein_site_resp.mode == ProteinSiteRespMode.RESP:
            raise ValueError("protein_site_resp is only supported for protein workflows")
        if self.input.source == InputSource.DES and not self.des.components:
            raise ValueError("des.components is required when input.source='deep_eutectic_solvent'")
        return self

    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


def load_config(path: str | Path) -> WorkflowConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return WorkflowConfig.model_validate(data)


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value


def _section_from_model(payload: dict[str, Any]) -> Any:
    section = table()
    for key, value in _drop_none_values(payload).items():
        if value is None:
            continue
        section[key] = value
    return section


def dump_config(config: WorkflowConfig) -> str:
    doc = document()
    doc.add("input", _section_from_model(config.input.model_dump(mode="json")))
    doc.add(nl())
    doc.add("des", _section_from_model(config.des.model_dump(mode="json")))
    doc.add(nl())
    doc.add("prepare", _section_from_model(config.prepare.model_dump(mode="json")))
    doc.add(nl())
    doc.add("protonation", _section_from_model(config.protonation.model_dump(mode="json")))
    doc.add(nl())
    doc.add("protein_site_resp", _section_from_model(config.protein_site_resp.model_dump(mode="json")))
    doc.add(nl())
    doc.add("ligands", _section_from_model(config.ligands.model_dump(mode="json")))
    doc.add(nl())
    doc.add("system", _section_from_model(config.system.model_dump(mode="json")))
    doc.add(nl())
    doc.add("md", _section_from_model(config.md.model_dump(mode="json")))
    doc.add(nl())
    doc.add("slurm", _section_from_model(config.slurm.model_dump(mode="json")))
    doc.add(nl())
    doc["output_dir"] = config.output_dir
    return doc.as_string()


def save_config(config: WorkflowConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_config(config), encoding="utf-8")
    return target
