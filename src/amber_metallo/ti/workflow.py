from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from amber_metallo.amber.leap import build_system_with_tleap
from amber_metallo.config import (
    MetalChargeAssignment,
    MetalModel,
    SaltConfig,
    SaltKind,
    SaltMode,
    SlurmConfig,
    SlurmProfile,
    SystemConfig,
)
from amber_metallo.environment import detect_amber_environment
from amber_metallo.reporting import activity_status, console, print_notice, write_json
from amber_metallo.ti import abfe as ti_abfe
from amber_metallo.ti.analysis import (
    assess_site_stability,
    build_cluster_atom_mask,
    copy_structure,
    default_formal_charge,
    detect_bound_metal_sites,
    parse_cntrl_settings,
    record_site_analysis,
    run_cluster_representative_selection,
    run_last_snapshot_extraction,
    select_site,
)
from amber_metallo.ti.config import (
    TIChargeCompensationMode,
    TIDecouplingMode,
    SnapshotMode,
    TIImplementationMode,
    TISamplingMode,
    TIWorkflowConfig,
)
from amber_metallo.ti.counterions import CounterionPlan, prepare_charge_compensating_counterions
from amber_metallo.ti.protocols import (
    generate_bound_start_preparation_inputs,
    generate_qoff_endpoint_preparation_inputs,
    generate_ti_inputs,
    generate_water_reference_preparation_inputs,
)
from amber_metallo.ti.restraints import (
    build_bound_site_restraint,
    write_combined_bound_site_restraints,
    write_qoff_duplicate_bound_site_restraint,
    write_qoff_duplicate_bound_site_restraints,
)
from amber_metallo.ti.slurm import QoffCoordinateBridge, write_leg_slurm_scripts
from amber_metallo.ti.topology import (
    inspect_prmtop_charge_state,
    missing_required_1264_charge_families,
    missing_required_126_charge_families,
    prepare_decharged_topology,
    prepare_qoff_disjoint_topology,
    prepare_ti_input_topology,
    resolve_1264_ion_frcmods,
    resolve_official_126_ion_frcmods,
)


WATER_REFERENCE_SCHEME_VERSION = "water-ref-v8-counterion-periodic-box"


def _copy_if_present(source: str | Path | None, destination_dir: Path) -> str | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source_path.name
    shutil.copy2(source_path, target)
    return str(target)


def _uses_gti_1264(config: TIWorkflowConfig) -> bool:
    return config.ti.implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI


def _uses_combined_gti_decoupling(config: TIWorkflowConfig) -> bool:
    return _uses_gti_1264(config) and config.ti.decoupling_mode == TIDecouplingMode.COMBINED_Q_VDW


def _uses_in_place_bound_ti(config: TIWorkflowConfig) -> bool:
    return bool(config.water_reference.bound_in_place or not config.water_reference.enabled)


def _uses_in_place_bound_only_ti(config: TIWorkflowConfig) -> bool:
    return _uses_in_place_bound_ti(config) and not config.water_reference.enabled


def _uses_split_qoff_disjoint_topology(config: TIWorkflowConfig) -> bool:
    return not _uses_combined_gti_decoupling(config)


def _water_reference_salt(config: TIWorkflowConfig) -> SaltConfig:
    if config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS:
        return SaltConfig(kind=SaltKind.NACL, mode=SaltMode.NEUTRALIZE)
    return SaltConfig()


def _neutrality_validation(
    *,
    initial_prmtop: str | Path,
    endpoint_prmtop: str | Path,
    enabled: bool,
    dry_run: bool,
) -> dict[str, object]:
    if not enabled:
        return {"status": "not_requested"}
    if dry_run:
        return {"status": "deferred_dry_run"}
    initial_charge = inspect_prmtop_charge_state(initial_prmtop).net_charge
    endpoint_charge = inspect_prmtop_charge_state(endpoint_prmtop).net_charge
    if abs(initial_charge) > 0.05 or abs(endpoint_charge) > 0.05:
        raise RuntimeError(
            "Charge-neutral TI topology validation failed: "
            f"initial={initial_charge:+.6f} e, endpoint={endpoint_charge:+.6f} e."
        )
    return {
        "status": "passed",
        "initial_charge": initial_charge,
        "endpoint_charge": endpoint_charge,
        "tolerance": 0.05,
    }


def _qoff_disjoint_metadata_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(manifest.get("qoff_disjoint_metadata") or {})
    for key in (
        "qoff_original_atom_index",
        "qoff_duplicate_atom_index",
        "qoff_timask1",
        "qoff_timask2",
        "qoff_crgmask",
    ):
        if key in manifest and key not in metadata:
            metadata[key] = manifest[key]
    missing = [
        key
        for key in (
            "qoff_original_atom_index",
            "qoff_duplicate_atom_index",
            "qoff_timask1",
            "qoff_timask2",
            "qoff_crgmask",
        )
        if metadata.get(key) in (None, "")
    ]
    if missing:
        raise RuntimeError(
            "The water-reference manifest is missing split Q-off disjoint metadata: "
            + ", ".join(missing)
            + ". Regenerate the water-reference cache so the duplicate metal atom index is recorded."
        )
    return metadata


def _effective_slurm_config(config: TIWorkflowConfig) -> SlurmConfig:
    if not _uses_gti_1264(config) or config.slurm.profile == SlurmProfile.GPU:
        return config.slurm
    return config.slurm.model_copy(update={"profile": SlurmProfile.GPU})


def _resolve_ti_ion_frcmods(
    *,
    config: TIWorkflowConfig,
    amber_env,
    formal_charge: int,
) -> list[str]:
    if _uses_gti_1264(config):
        resolved = [
            str(path)
            for path in resolve_1264_ion_frcmods(
                amber_env=amber_env,
                water_model=config.water_reference.water_model,
                custom_ion_frcmods=config.water_reference.custom_ion_frcmods,
            )
        ]
        missing = missing_required_1264_charge_families(resolved, formal_charge=formal_charge)
        if missing:
            raise ValueError(
                f"The resolved Amber 12-6-4 ion frcmods for water model '{config.water_reference.water_model}' "
                f"do not cover the selected TI metal charge state (+{formal_charge}). Missing family: "
                + ", ".join(missing)
                + ". Provide matching custom 12-6-4 ion frcmods or choose a supported water model."
            )
        return resolved

    resolved = [
        str(path)
        for path in resolve_official_126_ion_frcmods(
            amber_env=amber_env,
            water_model=config.water_reference.water_model,
            custom_ion_frcmods=config.water_reference.custom_ion_frcmods,
        )
    ]
    if not resolved:
        raise ValueError(
            f"No official Amber 12-6 ion frcmods were found for water model '{config.water_reference.water_model}'. "
            "Provide explicit custom official 12-6 ion frcmods or choose a supported water model for TI."
        )
    missing = missing_required_126_charge_families(resolved, formal_charge=formal_charge)
    if missing:
        raise ValueError(
            f"The resolved official Amber 12-6 ion frcmods for water model '{config.water_reference.water_model}' "
            f"do not cover the selected TI metal charge state (+{formal_charge}). Missing family: "
            + ", ".join(missing)
            + ". Provide the matching official custom frcmod files or choose a supported water model."
        )
    return resolved


def _resolve_ti_ion_frcmods_for_charges(
    *,
    config: TIWorkflowConfig,
    amber_env,
    formal_charges: list[int],
) -> list[str]:
    unique_charges = sorted({int(charge) for charge in formal_charges})
    if not unique_charges:
        raise ValueError("At least one formal charge is required for TI ion frcmod resolution.")
    resolved = _resolve_ti_ion_frcmods(
        config=config,
        amber_env=amber_env,
        formal_charge=unique_charges[0],
    )
    for charge in unique_charges[1:]:
        _resolve_ti_ion_frcmods(
            config=config,
            amber_env=amber_env,
            formal_charge=charge,
        )
    return resolved


def _atom_mask_from_indices(atom_indices: list[int]) -> str:
    unique = sorted({int(index) for index in atom_indices})
    if not unique:
        raise ValueError("At least one atom index is required to build an Amber atom mask.")
    return "@" + ",".join(str(index) for index in unique)


def _resolve_selected_sites(config: TIWorkflowConfig, candidates) -> list[Any]:
    if not candidates:
        raise ValueError("No bound metal candidates were detected in the reference structure.")
    if config.metal.selected_sites:
        return [select_site(candidates, int(site)) for site in config.metal.selected_sites]
    if config.metal.selected_site is not None:
        return [select_site(candidates, config.metal.selected_site)]
    if len(candidates) == 1:
        return [candidates[0]]
    available = ", ".join(str(candidate.site) for candidate in candidates)
    raise ValueError(
        "Multiple bound metal candidates were detected, but metal.selected_site or metal.selected_sites was not provided. "
        f"Available sites: {available}."
    )


def _formal_charge_for_site(config: TIWorkflowConfig, selected) -> int:
    if selected.site in config.metal.formal_charges_by_site:
        return int(config.metal.formal_charges_by_site[selected.site])
    if config.metal.formal_charge is not None:
        return int(config.metal.formal_charge)
    return default_formal_charge(selected.element)


def _selected_sites_label(selected_sites: list[Any]) -> str:
    if len(selected_sites) == 1:
        selected = selected_sites[0]
        return f"site {selected.site} ({selected.element} at {selected.key})"
    return "; ".join(f"site {selected.site} ({selected.element} at {selected.key})" for selected in selected_sites)


def _qoff_pairs_from_metadata(metadata: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = []
    for item in metadata.get("qoff_atom_pairs") or []:
        pairs.append((int(item["original_atom_index"]), int(item["duplicate_atom_index"])))
    if pairs:
        return pairs
    return [(int(metadata["qoff_original_atom_index"]), int(metadata["qoff_duplicate_atom_index"]))]


def _qoff_duplicate_map_from_metadata(metadata: dict[str, Any]) -> dict[int, int]:
    return {original: duplicate for original, duplicate in _qoff_pairs_from_metadata(metadata)}


def water_reference_root(config: TIWorkflowConfig) -> Path:
    if config.water_reference.cache_dir:
        return Path(config.water_reference.cache_dir).expanduser().resolve()
    return (Path.cwd() / "water_ref").resolve()


def _water_reference_signature(
    config: TIWorkflowConfig,
    *,
    metal_element: str,
    formal_charge: int,
    inherited_settings,
    official_126_frcmods: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "metal": metal_element,
        "formal_charge": formal_charge,
        "water_model": config.water_reference.water_model,
        "box_shape": config.water_reference.box_shape.value,
        "buffer_angstrom": round(config.water_reference.buffer_angstrom, 3),
        "custom_ion_frcmods": config.water_reference.custom_ion_frcmods,
        "official_12_6_frcmods": list(official_126_frcmods or []),
        "inherited_md_settings": inherited_settings.to_dict(),
        "ti_protocol": config.ti.model_dump(mode="json"),
        "scheme_version": WATER_REFERENCE_SCHEME_VERSION,
    }


def _sanitize_water_reference_label(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "water_ref"


def water_reference_entry_dir(
    config: TIWorkflowConfig,
    *,
    metal_element: str,
    formal_charge: int,
    inherited_settings,
    official_126_frcmods: list[str] | None = None,
) -> Path:
    entry_label = _sanitize_water_reference_label(
        f"{metal_element}{formal_charge}_{config.water_reference.water_model}"
    )
    base_entry = water_reference_root(config) / entry_label
    manifest_path = base_entry / "water_reference_manifest.json"
    if manifest_path.exists() and not water_reference_entry_matches(
        config,
        entry_dir=base_entry,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    ):
        signature_hash = _water_reference_signature_hash(
            config,
            metal_element=metal_element,
            formal_charge=formal_charge,
            inherited_settings=inherited_settings,
            official_126_frcmods=official_126_frcmods,
        )
        return water_reference_root(config) / f"{entry_label}_{signature_hash}"
    return base_entry


def _water_reference_manifest_path(entry_dir: Path) -> Path:
    return entry_dir / "water_reference_manifest.json"


def _water_reference_signature_hash(
    config: TIWorkflowConfig,
    *,
    metal_element: str,
    formal_charge: int,
    inherited_settings,
    official_126_frcmods: list[str] | None = None,
) -> str:
    signature = _water_reference_signature(
        config,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    encoded = json.dumps(signature, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def water_reference_entry_matches(
    config: TIWorkflowConfig,
    *,
    entry_dir: str | Path,
    metal_element: str,
    formal_charge: int,
    inherited_settings,
    official_126_frcmods: list[str] | None = None,
) -> bool:
    manifest_path = _water_reference_manifest_path(Path(entry_dir))
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_hash = _water_reference_signature_hash(
        config,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    manifest_hash = manifest.get("signature_hash")
    if manifest_hash is not None:
        return manifest_hash == expected_hash
    expected_signature = _water_reference_signature(
        config,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    return all(manifest.get(key) == value for key, value in expected_signature.items())


def water_reference_entry_is_complete(entry_dir: str | Path) -> bool:
    manifest_path = _water_reference_manifest_path(Path(entry_dir))
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required_paths = [Path(item) for item in manifest.get("required_paths") or []]
    if not required_paths:
        return False
    return all(path.exists() for path in required_paths)


def _write_water_reference_manifest(
    *,
    entry_dir: Path,
    config: TIWorkflowConfig,
    metal_element: str,
    formal_charge: int,
    inherited_settings,
    ti_input_prmtop: Path,
    qoff_prmtop: Path,
    qoff_metadata: dict[str, Any] | None,
    water_decharged_prmtop: Path,
    official_126_frcmods: list[str],
    system_start_coord: Path | None = None,
    counterion_plan: CounterionPlan | None = None,
    alchemical_atom_indices: list[int] | None = None,
    counterion_mask: str | None = None,
    qoff_counterion_mask: str | None = None,
    neutrality_validation: dict[str, object] | None = None,
) -> dict[str, Any]:
    required_paths = [
        entry_dir / "system.prmtop",
        entry_dir / "system.inpcrd",
        entry_dir / "prep" / "prep_manifest.json",
        entry_dir / "prep" / "outputs" / "04_eq.rst7",
        ti_input_prmtop,
        qoff_prmtop,
        water_decharged_prmtop,
        entry_dir / "ti_manifest.json",
    ]
    system_manifest_path = entry_dir / "system_manifest.json"
    system_manifest = json.loads(system_manifest_path.read_text(encoding="utf-8")) if system_manifest_path.exists() else {}
    qoff_metadata = dict(qoff_metadata or {})
    manifest = {
        "water_reference_dir": str(entry_dir),
        "signature_hash": _water_reference_signature_hash(
            config,
            metal_element=metal_element,
            formal_charge=formal_charge,
            inherited_settings=inherited_settings,
            official_126_frcmods=official_126_frcmods,
        ),
        "system_manifest_path": str(system_manifest_path),
        "system_manifest": system_manifest,
        "prep_manifest_path": str(entry_dir / "prep" / "prep_manifest.json"),
        "prep_start_coord": str(entry_dir / "prep" / "outputs" / "04_eq.rst7"),
        "system_start_coord": str(system_start_coord or (entry_dir / "system.inpcrd")),
        "counterion_plan": None if counterion_plan is None else counterion_plan.to_dict(),
        "alchemical_atom_indices": list(alchemical_atom_indices or [1]),
        "alchemical_atom_mask": _atom_mask_from_indices(list(alchemical_atom_indices or [1])),
        "counterion_mask": counterion_mask,
        "qoff_counterion_mask": qoff_counterion_mask,
        "neutrality_validation": neutrality_validation or {"status": "not_requested"},
        "ti_input_prmtop": str(ti_input_prmtop),
        "qoff_prmtop": str(qoff_prmtop),
        "qoff_disjoint_metadata": qoff_metadata,
        "qoff_original_atom_index": qoff_metadata.get("qoff_original_atom_index"),
        "qoff_duplicate_atom_index": qoff_metadata.get("qoff_duplicate_atom_index"),
        "qoff_original_atom_indices": qoff_metadata.get("qoff_original_atom_indices"),
        "qoff_duplicate_atom_indices": qoff_metadata.get("qoff_duplicate_atom_indices"),
        "qoff_atom_pairs": qoff_metadata.get("qoff_atom_pairs"),
        "qoff_timask1": qoff_metadata.get("qoff_timask1"),
        "qoff_timask2": qoff_metadata.get("qoff_timask2"),
        "qoff_crgmask": qoff_metadata.get("qoff_crgmask"),
        "ti_manifest_path": str(entry_dir / "ti_manifest.json"),
        "decharge_manifest_path": str(entry_dir / "water_decharge_manifest.json"),
        "required_paths": [str(path) for path in required_paths],
        "ready_for_ti": all(path.exists() for path in required_paths),
        **_water_reference_signature(
            config,
            metal_element=metal_element,
            formal_charge=formal_charge,
            inherited_settings=inherited_settings,
            official_126_frcmods=official_126_frcmods,
        ),
    }
    write_json(_water_reference_manifest_path(entry_dir), manifest)
    return manifest


def _build_metal_only_pdb(*, metal_element: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        (
            "HEADER    WATER REFERENCE METAL\n"
            f"HETATM    1 {metal_element:>4} {metal_element:>3} A   1       0.000   0.000   0.000  1.00 20.00          {metal_element:>2}\n"
            "END\n"
        ),
        encoding="utf-8",
    )
    return output_path


def _build_multi_metal_only_pdb(
    *,
    selected_sites: list[Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not selected_sites:
        raise ValueError("At least one metal site is required for a multi-metal water reference.")
    spacing = 4.0
    center_offset = (len(selected_sites) - 1) * 0.5
    lines = ["HEADER    WATER REFERENCE MULTI METAL\n"]
    for index, selected in enumerate(selected_sites, start=1):
        x = (index - 1 - center_offset) * spacing
        element = selected.element.title()
        residue = element.upper()[:3]
        lines.append(
            f"HETATM{index:5d} {element:>4} {residue:>3} A{index:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00          {element:>2}\n"
        )
    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _relative_path(path: str | Path, *, from_dir: Path) -> str:
    return Path(os.path.relpath(Path(path), start=from_dir)).as_posix()


def _multi_water_reference_label(config: TIWorkflowConfig, selected_sites: list[Any], formal_charges_by_site: dict[int, int]) -> str:
    raw = "_".join(
        f"{selected.element}{formal_charges_by_site[selected.site]}_site{selected.site}"
        for selected in selected_sites
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return _sanitize_water_reference_label(f"multi_{raw}_{config.water_reference.water_model}_{digest}")


def _multi_water_reference_signature(
    config: TIWorkflowConfig,
    *,
    selected_sites: list[Any],
    formal_charges_by_site: dict[int, int],
    inherited_settings,
    official_126_frcmods: list[str],
) -> dict[str, Any]:
    return {
        "metal": "multi",
        "metals": [
            {
                "site": selected.site,
                "atom_index": selected.atom_index,
                "element": selected.element,
                "formal_charge": formal_charges_by_site[selected.site],
            }
            for selected in selected_sites
        ],
        "formal_charge": sum(formal_charges_by_site[selected.site] for selected in selected_sites),
        "water_model": config.water_reference.water_model,
        "box_shape": config.water_reference.box_shape.value,
        "buffer_angstrom": round(config.water_reference.buffer_angstrom, 3),
        "custom_ion_frcmods": config.water_reference.custom_ion_frcmods,
        "official_12_6_frcmods": list(official_126_frcmods or []),
        "inherited_md_settings": inherited_settings.to_dict(),
        "ti_protocol": config.ti.model_dump(mode="json"),
        "scheme_version": WATER_REFERENCE_SCHEME_VERSION + "-multi",
    }


def _multi_water_reference_signature_hash(
    config: TIWorkflowConfig,
    *,
    selected_sites: list[Any],
    formal_charges_by_site: dict[int, int],
    inherited_settings,
    official_126_frcmods: list[str],
) -> str:
    signature = _multi_water_reference_signature(
        config,
        selected_sites=selected_sites,
        formal_charges_by_site=formal_charges_by_site,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    return hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _prepare_multi_water_reference(
    *,
    config: TIWorkflowConfig,
    inherited_settings,
    selected_sites: list[Any],
    formal_charges_by_site: dict[int, int],
    amber_env,
    official_126_frcmods: list[str],
    dry_run: bool,
) -> tuple[Path, dict[str, Any], bool]:
    cache_root = water_reference_root(config)
    cache_dir = cache_root / _multi_water_reference_label(config, selected_sites, formal_charges_by_site)
    signature_hash = _multi_water_reference_signature_hash(
        config,
        selected_sites=selected_sites,
        formal_charges_by_site=formal_charges_by_site,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    manifest_path = _water_reference_manifest_path(cache_dir)
    reused = False
    if config.water_reference.reuse_existing and water_reference_entry_is_complete(cache_dir) and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        reused = existing.get("signature_hash") == signature_hash
    if not reused:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metal_pdb = _build_multi_metal_only_pdb(selected_sites=selected_sites, output_path=cache_dir / "metal_input.pdb")
        system_config = SystemConfig(
            protein_ff="ff19SB",
            ligand_ff="gaff2",
            metal_model=MetalModel.MODEL_1264,
            metal_charges=[
                MetalChargeAssignment(site=index, charge=formal_charges_by_site[selected.site])
                for index, selected in enumerate(selected_sites, start=1)
            ],
            water_model=config.water_reference.water_model,
            box_shape=config.water_reference.box_shape,
            buffer_angstrom=config.water_reference.buffer_angstrom,
            salt=_water_reference_salt(config),
            custom_ion_frcmods=official_126_frcmods,
        )
        with activity_status(
            "[blink bold cyan]Processing...[/] Building the multi-metal-in-water reference system.",
            plain_message="Processing... Building the multi-metal-in-water reference system.",
        ):
            build_system_with_tleap(
                system_config=system_config,
                amber_env=amber_env,
                prepared_pdb=metal_pdb,
                ligand_artifacts=[],
                source_files=[],
                output_dir=cache_dir,
                dry_run=dry_run,
            )
    metal_atom_indices = list(range(1, len(selected_sites) + 1))
    water_counterion_plan: CounterionPlan | None = None
    water_source_prmtop = cache_dir / "system.prmtop"
    water_start_coord = cache_dir / "system.inpcrd"
    if config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS:
        water_counterion_plan = prepare_charge_compensating_counterions(
            source_prmtop=water_source_prmtop,
            source_pdb=cache_dir / "system.pdb",
            source_coord=water_start_coord,
            metal_atom_indices=metal_atom_indices,
            formal_charges=[formal_charges_by_site[item.site] for item in selected_sites],
            output_dir=cache_dir / "counterion_prep",
            water_model=config.water_reference.water_model,
            ion_frcmods=official_126_frcmods,
            amber_env=amber_env,
            config=config.ti,
            dry_run=dry_run,
        )
        water_source_prmtop = Path(water_counterion_plan.topology_path)
        water_start_coord = Path(water_counterion_plan.start_coord_path)
    water_counterion_mask = None if water_counterion_plan is None else water_counterion_plan.counterion_mask
    generate_water_reference_preparation_inputs(
        inherited_settings=inherited_settings,
        output_dir=cache_dir,
        positional_restraint_mask=water_counterion_mask,
        restraint_force_constant=config.ti.counterion_restraint_force_constant,
    )

    water_atom_indices = [
        *metal_atom_indices,
        *([] if water_counterion_plan is None else water_counterion_plan.counterion_atom_indices),
    ]
    water_atom_mask = _atom_mask_from_indices(water_atom_indices)
    elements_by_index = {index: selected.element for index, selected in enumerate(selected_sites, start=1)}
    charges_by_index = {
        index: formal_charges_by_site[selected.site]
        for index, selected in enumerate(selected_sites, start=1)
    }
    ti_input_topology = prepare_ti_input_topology(
        input_prmtop=water_source_prmtop,
        output_dir=cache_dir,
        label="water_ref",
        implementation_mode=config.ti.implementation_mode,
        dry_run=dry_run,
        water_model=config.water_reference.water_model,
        official_126_frcmods=official_126_frcmods,
        alchemical_atom_index=water_atom_indices[0],
        alchemical_element=selected_sites[0].element,
        alchemical_charge=formal_charges_by_site[selected_sites[0].site],
        alchemical_atom_indices=water_atom_indices,
        alchemical_elements_by_index=elements_by_index,
        alchemical_charges_by_index=charges_by_index,
    )
    water_decharged = prepare_decharged_topology(
        input_prmtop=Path(str(ti_input_topology["ti_prmtop"])),
        atom_mask=water_atom_mask,
        output_dir=cache_dir,
        label="water",
        dry_run=dry_run,
        preserve_c4=_uses_gti_1264(config),
    )
    water_neutrality_validation = _neutrality_validation(
        initial_prmtop=ti_input_topology["ti_prmtop"],
        endpoint_prmtop=water_decharged["decharged_prmtop"],
        enabled=config.ti.charge_compensation_mode
        == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS,
        dry_run=dry_run,
    )
    water_qoff_topology: dict[str, Any] | None = None
    if _uses_split_qoff_disjoint_topology(config):
        water_qoff_topology = prepare_qoff_disjoint_topology(
            input_prmtop=Path(str(ti_input_topology["ti_prmtop"])),
            output_dir=cache_dir,
            label="water",
            alchemical_atom_index=water_atom_indices[0],
            alchemical_atom_indices=water_atom_indices,
            dry_run=dry_run,
        )
    water_qoff_counterion_mask = water_counterion_mask
    if (
        water_qoff_topology is not None
        and water_counterion_plan is not None
        and water_counterion_plan.counterion_atom_indices
    ):
        duplicate_map = _qoff_duplicate_map_from_metadata(water_qoff_topology)
        water_qoff_counterion_mask = _atom_mask_from_indices(
            sorted(
                {
                    *water_counterion_plan.counterion_atom_indices,
                    *[
                        duplicate_map[index]
                        for index in water_counterion_plan.counterion_atom_indices
                        if index in duplicate_map
                    ],
                }
            )
        )
    generate_ti_inputs(
        config=config.ti,
        inherited_settings=inherited_settings,
        atom_mask=water_atom_mask,
        restraint_file=None,
        output_dir=cache_dir,
        qoff_start_source="restart",
        qoff_timask1=None if water_qoff_topology is None else str(water_qoff_topology["qoff_timask1"]),
        qoff_timask2=None if water_qoff_topology is None else str(water_qoff_topology["qoff_timask2"]),
        qoff_charge_mask=None if water_qoff_topology is None else str(water_qoff_topology["qoff_crgmask"]),
        positional_restraint_mask=water_counterion_mask,
        qoff_positional_restraint_mask=water_qoff_counterion_mask,
    )
    required_paths = [
        cache_dir / "system.prmtop",
        cache_dir / "system.inpcrd",
        cache_dir / "prep" / "prep_manifest.json",
        cache_dir / "prep" / "outputs" / "04_eq.rst7",
        Path(str(ti_input_topology["ti_prmtop"])),
        Path(str(water_qoff_topology["qoff_prmtop"])) if water_qoff_topology else Path(str(ti_input_topology["ti_prmtop"])),
        Path(str(water_decharged["decharged_prmtop"])),
        cache_dir / "ti_manifest.json",
    ]
    signature = _multi_water_reference_signature(
        config,
        selected_sites=selected_sites,
        formal_charges_by_site=formal_charges_by_site,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    qoff_metadata = dict(water_qoff_topology or {})
    manifest = {
        "water_reference_dir": str(cache_dir),
        "signature_hash": signature_hash,
        "system_manifest_path": str(cache_dir / "system_manifest.json"),
        "prep_manifest_path": str(cache_dir / "prep" / "prep_manifest.json"),
        "prep_start_coord": str(cache_dir / "prep" / "outputs" / "04_eq.rst7"),
        "system_start_coord": str(water_start_coord),
        "counterion_plan": None if water_counterion_plan is None else water_counterion_plan.to_dict(),
        "alchemical_atom_indices": water_atom_indices,
        "alchemical_atom_mask": water_atom_mask,
        "counterion_mask": water_counterion_mask,
        "qoff_counterion_mask": water_qoff_counterion_mask,
        "neutrality_validation": water_neutrality_validation,
        "ti_input_prmtop": str(ti_input_topology["ti_prmtop"]),
        "qoff_prmtop": str(water_qoff_topology["qoff_prmtop"]) if water_qoff_topology else str(ti_input_topology["ti_prmtop"]),
        "qoff_disjoint_metadata": qoff_metadata,
        "qoff_original_atom_index": qoff_metadata.get("qoff_original_atom_index"),
        "qoff_duplicate_atom_index": qoff_metadata.get("qoff_duplicate_atom_index"),
        "qoff_original_atom_indices": qoff_metadata.get("qoff_original_atom_indices"),
        "qoff_duplicate_atom_indices": qoff_metadata.get("qoff_duplicate_atom_indices"),
        "qoff_atom_pairs": qoff_metadata.get("qoff_atom_pairs"),
        "qoff_timask1": qoff_metadata.get("qoff_timask1"),
        "qoff_timask2": qoff_metadata.get("qoff_timask2"),
        "qoff_crgmask": qoff_metadata.get("qoff_crgmask"),
        "ti_manifest_path": str(cache_dir / "ti_manifest.json"),
        "decharge_manifest_path": str(cache_dir / "water_decharge_manifest.json"),
        "required_paths": [str(path) for path in required_paths],
        "ready_for_ti": all(path.exists() for path in required_paths),
        **signature,
    }
    write_json(manifest_path, manifest)
    return cache_dir, manifest, reused


def _prepare_water_reference(
    *,
    config: TIWorkflowConfig,
    inherited_settings,
    metal_element: str,
    formal_charge: int,
    amber_env,
    official_126_frcmods: list[str],
    dry_run: bool,
) -> tuple[Path, dict[str, Any], bool]:
    cache_root = water_reference_root(config)
    cache_dir = water_reference_entry_dir(
        config,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    reused = False
    cache_root.mkdir(parents=True, exist_ok=True)
    existing_complete = water_reference_entry_is_complete(cache_dir)
    exact_match = water_reference_entry_matches(
        config,
        entry_dir=cache_dir,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        official_126_frcmods=official_126_frcmods,
    )
    if existing_complete and exact_match and config.water_reference.reuse_existing:
        reused = True
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metal_pdb = _build_metal_only_pdb(metal_element=metal_element, output_path=cache_dir / "metal_input.pdb")
        system_config = SystemConfig(
            protein_ff="ff19SB",
            ligand_ff="gaff2",
            metal_model=MetalModel.MODEL_1264,
            metal_charges=[MetalChargeAssignment(site=1, charge=formal_charge)],
            water_model=config.water_reference.water_model,
            box_shape=config.water_reference.box_shape,
            buffer_angstrom=config.water_reference.buffer_angstrom,
            salt=_water_reference_salt(config),
            custom_ion_frcmods=official_126_frcmods,
        )
        with activity_status(
            "[blink bold cyan]Processing...[/] Building the metal-in-water reference system.",
            plain_message="Processing... Building the metal-in-water reference system.",
        ):
            build_system_with_tleap(
                system_config=system_config,
                amber_env=amber_env,
                prepared_pdb=metal_pdb,
                ligand_artifacts=[],
                source_files=[],
                output_dir=cache_dir,
                dry_run=dry_run,
            )

    water_counterion_plan: CounterionPlan | None = None
    water_source_prmtop = cache_dir / "system.prmtop"
    water_start_coord = cache_dir / "system.inpcrd"
    if config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS:
        water_counterion_plan = prepare_charge_compensating_counterions(
            source_prmtop=water_source_prmtop,
            source_pdb=cache_dir / "system.pdb",
            source_coord=water_start_coord,
            metal_atom_indices=[1],
            formal_charges=[formal_charge],
            output_dir=cache_dir / "counterion_prep",
            water_model=config.water_reference.water_model,
            ion_frcmods=official_126_frcmods,
            amber_env=amber_env,
            config=config.ti,
            dry_run=dry_run,
        )
        water_source_prmtop = Path(water_counterion_plan.topology_path)
        water_start_coord = Path(water_counterion_plan.start_coord_path)
    water_counterion_mask = None if water_counterion_plan is None else water_counterion_plan.counterion_mask
    water_atom_indices = [
        1,
        *([] if water_counterion_plan is None else water_counterion_plan.counterion_atom_indices),
    ]
    water_atom_mask = _atom_mask_from_indices(water_atom_indices)
    generate_water_reference_preparation_inputs(
        inherited_settings=inherited_settings,
        output_dir=cache_dir,
        positional_restraint_mask=water_counterion_mask,
        restraint_force_constant=config.ti.counterion_restraint_force_constant,
    )
    ti_input_topology = prepare_ti_input_topology(
        input_prmtop=water_source_prmtop,
        output_dir=cache_dir,
        label="water_ref",
        implementation_mode=config.ti.implementation_mode,
        dry_run=dry_run,
        water_model=config.water_reference.water_model,
        official_126_frcmods=official_126_frcmods,
        alchemical_atom_index=1,
        alchemical_element=metal_element,
        alchemical_charge=formal_charge,
        alchemical_atom_indices=water_atom_indices,
        alchemical_elements_by_index={1: metal_element},
        alchemical_charges_by_index={1: formal_charge},
    )
    water_decharged = prepare_decharged_topology(
        input_prmtop=Path(str(ti_input_topology["ti_prmtop"])),
        atom_mask=water_atom_mask,
        output_dir=cache_dir,
        label="water",
        dry_run=dry_run,
        preserve_c4=_uses_gti_1264(config),
    )
    water_neutrality_validation = _neutrality_validation(
        initial_prmtop=ti_input_topology["ti_prmtop"],
        endpoint_prmtop=water_decharged["decharged_prmtop"],
        enabled=config.ti.charge_compensation_mode
        == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS,
        dry_run=dry_run,
    )
    water_qoff_topology: dict[str, Any] | None = None
    if _uses_split_qoff_disjoint_topology(config):
        water_qoff_topology = prepare_qoff_disjoint_topology(
            input_prmtop=Path(str(ti_input_topology["ti_prmtop"])),
            output_dir=cache_dir,
            label="water",
            alchemical_atom_index=1,
            alchemical_atom_indices=water_atom_indices,
            dry_run=dry_run,
        )
    water_qoff_counterion_mask = water_counterion_mask
    if (
        water_qoff_topology is not None
        and water_counterion_plan is not None
        and water_counterion_plan.counterion_atom_indices
    ):
        duplicate_map = _qoff_duplicate_map_from_metadata(water_qoff_topology)
        water_qoff_counterion_mask = _atom_mask_from_indices(
            sorted(
                {
                    *water_counterion_plan.counterion_atom_indices,
                    *[
                        duplicate_map[index]
                        for index in water_counterion_plan.counterion_atom_indices
                        if index in duplicate_map
                    ],
                }
            )
        )
    generate_ti_inputs(
        config=config.ti,
        inherited_settings=inherited_settings,
        atom_mask=water_atom_mask,
        restraint_file=None,
        output_dir=cache_dir,
        qoff_start_source="restart",
        qoff_timask1=None if water_qoff_topology is None else str(water_qoff_topology["qoff_timask1"]),
        qoff_timask2=None if water_qoff_topology is None else str(water_qoff_topology["qoff_timask2"]),
        qoff_charge_mask=None if water_qoff_topology is None else str(water_qoff_topology["qoff_crgmask"]),
        positional_restraint_mask=water_counterion_mask,
        qoff_positional_restraint_mask=water_qoff_counterion_mask,
    )
    manifest = _write_water_reference_manifest(
        entry_dir=cache_dir,
        config=config,
        metal_element=metal_element,
        formal_charge=formal_charge,
        inherited_settings=inherited_settings,
        ti_input_prmtop=Path(str(ti_input_topology["ti_prmtop"])),
        qoff_prmtop=Path(str(water_qoff_topology["qoff_prmtop"])) if water_qoff_topology else Path(str(ti_input_topology["ti_prmtop"])),
        qoff_metadata=water_qoff_topology,
        water_decharged_prmtop=Path(water_decharged["decharged_prmtop"]),
        official_126_frcmods=official_126_frcmods,
        system_start_coord=water_start_coord,
        counterion_plan=water_counterion_plan,
        alchemical_atom_indices=water_atom_indices,
        counterion_mask=water_counterion_mask,
        qoff_counterion_mask=water_qoff_counterion_mask,
        neutrality_validation=water_neutrality_validation,
    )
    return cache_dir, manifest, reused


def _resolve_selected_site(config: TIWorkflowConfig, candidates) -> Any:
    if not candidates:
        raise ValueError("No bound metal candidates were detected in the reference structure.")
    if config.metal.selected_site is not None:
        return select_site(candidates, config.metal.selected_site)
    if len(candidates) == 1:
        return candidates[0]
    available = ", ".join(str(candidate.site) for candidate in candidates)
    raise ValueError(
        "Multiple bound metal candidates were detected, but metal.selected_site was not provided. "
        f"Available sites: {available}."
    )


def _copy_complex_inputs(config: TIWorkflowConfig, root: Path) -> dict[str, str | None]:
    input_dir = root / "input"
    return {
        "prmtop": _copy_if_present(config.complex_input.prmtop_path, input_dir),
        "reference_structure": _copy_if_present(config.complex_input.reference_structure_path, input_dir),
        "production_mdin": _copy_if_present(config.complex_input.production_mdin_path, input_dir),
        "production_restart": _copy_if_present(config.complex_input.production_restart_path, input_dir),
        "trajectory": str(Path(config.complex_input.trajectory_path).expanduser().resolve()),
    }


_PRODUCTION_RESTART_EXTENSIONS = (".rst7", ".rst", ".restrt", ".restart", ".ncrst")


def _parse_restart_value_count(path: Path) -> tuple[int, int] | None:
    raw = path.read_bytes()
    if raw.startswith(b"CDF") or raw.startswith(b"\x89HDF"):
        return None
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    if len(lines) < 2:
        return None
    header_tokens = lines[1].split()
    if not header_tokens:
        return None
    try:
        natom = int(header_tokens[0])
    except ValueError:
        return None
    values = 0
    for line in lines[2:]:
        if not line.strip():
            continue
        try:
            values += len([float(token.replace("D", "E").replace("d", "e")) for token in line.split()])
            continue
        except ValueError:
            pass
        for start in range(0, len(line), 12):
            token = line[start : start + 12].strip()
            if token:
                float(token.replace("D", "E").replace("d", "e"))
                values += 1
    return natom, values


def _restart_has_velocity_records(path: Path) -> bool:
    parsed = _parse_restart_value_count(path)
    if parsed is None:
        # NetCDF/HDF restarts can contain velocities, but this lightweight check cannot inspect them.
        # Keep them eligible rather than discarding a valid production restart.
        return True
    natom, value_count = parsed
    if natom <= 0:
        return False
    coord_count = 3 * natom
    trailing_count = max(0, value_count - coord_count)
    if trailing_count >= coord_count:
        return True
    return False


def _eligible_production_restart(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None
    if _restart_has_velocity_records(candidate):
        return candidate.resolve()
    console.print(
        "[bold yellow]Detected restart without velocity records; FreeE will generate a short bound-start prep before TI:[/bold yellow] "
        f"{candidate}"
    )
    return None


def _resolve_coordinate_restart_path(config: TIWorkflowConfig) -> Path | None:
    configured = config.complex_input.production_restart_path
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return path.resolve()
    trajectory_path = Path(config.complex_input.trajectory_path).expanduser()
    candidate_dirs = [trajectory_path.parent]
    stems = [trajectory_path.stem]
    if config.complex_input.production_mdin_path:
        mdin_path = Path(config.complex_input.production_mdin_path).expanduser()
        stems.append(mdin_path.stem)
        if mdin_path.parent.name.lower() == "inputs":
            candidate_dirs.append(mdin_path.parent.parent / "outputs")
    seen: set[Path] = set()
    for directory in candidate_dirs:
        for stem in dict.fromkeys(stems):
            for suffix in _PRODUCTION_RESTART_EXTENSIONS:
                candidate = (directory / f"{stem}{suffix}").resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.exists():
                    return candidate
    for fallback_name in ("restrt", "md.rst", "md.rst7"):
        candidate = (trajectory_path.parent / fallback_name).resolve()
        if candidate.exists():
            return candidate
    return None


def _resolve_production_restart_path(config: TIWorkflowConfig) -> Path | None:
    configured = config.complex_input.production_restart_path
    if configured:
        path = Path(configured).expanduser()
        eligible = _eligible_production_restart(path)
        if eligible is not None:
            return eligible

    trajectory_path = Path(config.complex_input.trajectory_path).expanduser()
    candidate_dirs = [trajectory_path.parent]
    stems = [trajectory_path.stem]
    if config.complex_input.production_mdin_path:
        mdin_path = Path(config.complex_input.production_mdin_path).expanduser()
        stems.append(mdin_path.stem)
        if mdin_path.parent.name.lower() == "inputs":
            candidate_dirs.append(mdin_path.parent.parent / "outputs")
    seen: set[Path] = set()
    for directory in candidate_dirs:
        for stem in dict.fromkeys(stems):
            for suffix in _PRODUCTION_RESTART_EXTENSIONS:
                candidate = (directory / f"{stem}{suffix}").resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                eligible = _eligible_production_restart(candidate)
                if eligible is not None:
                    return eligible
    for fallback_name in ("restrt", "md.rst", "md.rst7"):
        candidate = (trajectory_path.parent / fallback_name).resolve()
        eligible = _eligible_production_restart(candidate)
        if eligible is not None:
            return eligible
    return None


def _write_bound_snapshot_placeholders(
    *,
    snapshot_source_pdb: Path,
    snapshot_source_rst7: Path,
    start_restart_source: Path | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    bound_snapshot_dir = output_dir / "snapshot"
    bound_snapshot_dir.mkdir(parents=True, exist_ok=True)
    selected_pdb = bound_snapshot_dir / "selected_snapshot.pdb"
    selected_rst7 = bound_snapshot_dir / "selected_snapshot.rst7"
    copy_structure(snapshot_source_pdb, selected_pdb)
    restart_source = start_restart_source if start_restart_source and start_restart_source.exists() else snapshot_source_rst7
    if restart_source.exists():
        shutil.copy2(restart_source, selected_rst7)
    else:
        selected_rst7.write_text("TI selected snapshot placeholder\n0\n", encoding="utf-8")
    return selected_pdb, selected_rst7


def print_ti_workflow_summary(result: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.table import Table

    overview = Table(box=None, show_header=False)
    overview.add_column("Key", style="bold white")
    overview.add_column("Value", style="cyan", overflow="fold")
    overview.add_row("Output directory", str(result.get("output_dir", "N/A")))
    overview.add_row("Selected metal", str(result.get("selected_metal", "N/A")))
    overview.add_row("TI mode", str(result.get("ti_implementation_mode", "N/A")))
    overview.add_row("TI decoupling", str(result.get("ti_decoupling_mode", "N/A")))
    overview.add_row("Snapshot source", str(result.get("snapshot_source", "N/A")))
    overview.add_row("Water term source", str(result.get("water_term_source", "simulation")))
    if result.get("water_term_source") == "disabled":
        overview.add_row("Water reference directory", "Disabled for in-place TI")
    else:
        overview.add_row("Water reference reused", "Yes" if result.get("water_reference_reused") else "No")
    if result.get("water_term_source") == "library":
        overview.add_row("Water library key", str(result.get("water_library_key", "N/A")))
    elif result.get("water_term_source") != "disabled":
        overview.add_row("Water reference directory", str(result.get("water_reference_dir", "N/A")))
    overview.add_row("Bound TI outputs", str(result.get("bound_runtime_output_dir", "N/A")))
    if result.get("water_term_source") != "disabled":
        overview.add_row("Water TI outputs", str(result.get("water_runtime_output_dir", "N/A")))
    overview.add_row("Restraint correction (kcal/mol)", f"{float(result.get('restraint_correction_kcal_mol', 0.0)):.3f}")
    console.print(Panel(overview, title="[bold]TI Setup Complete[/bold]", border_style="green"))

    outputs = Table(title="Generated TI Assets")
    outputs.add_column("Item", style="bold white")
    outputs.add_column("Path", style="cyan", overflow="fold")
    outputs.add_row("Bound leg sbatch", str(result.get("bound_slurm", "N/A")))
    if result.get("water_term_source") != "disabled":
        outputs.add_row("Water leg sbatch", str(result.get("water_slurm", "N/A")))
    outputs.add_row("Bound TI qoff prmtop", str(result.get("bound_ti_input_topology", "N/A")))
    if result.get("water_term_source") != "disabled":
        outputs.add_row("Water TI qoff prmtop", str(result.get("water_ti_input_topology", "N/A")))
    outputs.add_row("Bound start source", str(result.get("bound_start_source", "N/A")))
    outputs.add_row("Bound start restart", str(result.get("bound_start_restart", "N/A")))
    bound_start = result.get("bound_start_preparation") or {}
    if bound_start:
        outputs.add_row("Bound start prep manifest", str(bound_start.get("prep_manifest", "N/A")))
    outputs.add_row("Bound TI outputs", str(result.get("bound_runtime_output_dir", "N/A")))
    if result.get("water_term_source") != "disabled":
        outputs.add_row("Water TI outputs", str(result.get("water_runtime_output_dir", "N/A")))
    if result.get("water_term_source") == "library":
        outputs.add_row("Water library snapshot", str(result.get("water_library_key", "N/A")))
    elif result.get("water_term_source") != "disabled":
        outputs.add_row("Water reference manifest", str(result.get("water_reference_manifest_path", "N/A")))
        outputs.add_row("Water reference start restart", str(result.get("water_reference_start_coord", "N/A")))
    outputs.add_row("Manifest", str(result.get("manifest", "N/A")))
    console.print(outputs)


def run_ti_workflow(*, config: TIWorkflowConfig, dry_run: bool = False) -> dict[str, Any]:
    if config.ti.implementation_mode == TIImplementationMode.GROMACS_TABULATED_12_6_4:
        raise NotImplementedError(
            "GROMACS tabulated 12-6-4 TI mode is still under development and is not yet available."
        )

    root = config.output_path()
    root.mkdir(parents=True, exist_ok=True)
    copied_inputs = _copy_complex_inputs(config, root)
    amber_env = detect_amber_environment()

    with activity_status(
        "[blink bold cyan]Processing...[/] Detecting bound metal candidates from the reference structure.",
        plain_message="Processing... Detecting bound metal candidates from the reference structure.",
    ):
        candidates = detect_bound_metal_sites(
            config.complex_input.reference_structure_path,
            config.complex_input.prmtop_path,
            donor_cutoff_angstrom=config.snapshot.donor_cutoff_angstrom,
            include_unbound_metals=_uses_in_place_bound_ti(config),
        )
    selected_sites = _resolve_selected_sites(config, candidates)
    selected = selected_sites[0]
    multi_site = len(selected_sites) > 1
    formal_charges_by_site = {
        item.site: _formal_charge_for_site(config, item)
        for item in selected_sites
    }
    formal_charge = formal_charges_by_site[selected.site]
    selected_atom_indices = [item.atom_index for item in selected_sites]
    alchemical_atom_mask = _atom_mask_from_indices(selected_atom_indices)
    if _uses_in_place_bound_ti(config):
        if not _uses_gti_1264(config):
            raise ValueError(
                "water_reference.bound_in_place=true or water_reference.enabled=false keeps the existing prmtop "
                "in place and currently requires ti.implementation_mode='amber_12_6_4_gti'."
            )
        ti_ion_frcmods = (
            _resolve_ti_ion_frcmods_for_charges(
                config=config,
                amber_env=amber_env,
                formal_charges=list(formal_charges_by_site.values()),
            )
            if (
                config.water_reference.enabled
                or config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS
            )
            else []
        )
    else:
        ti_ion_frcmods = _resolve_ti_ion_frcmods_for_charges(
            config=config,
            amber_env=amber_env,
            formal_charges=list(formal_charges_by_site.values()),
        )
    slurm_config = _effective_slurm_config(config)

    snapshot_dir = root / "snapshot"
    with activity_status(
        "[blink bold cyan]Processing...[/] Extracting the last snapshot from the production trajectory.",
        plain_message="Processing... Extracting the last snapshot from the production trajectory.",
    ):
        last_snapshot = run_last_snapshot_extraction(
            prmtop_path=config.complex_input.prmtop_path,
            trajectory_path=config.complex_input.trajectory_path,
            reference_structure_path=config.complex_input.reference_structure_path,
            output_dir=snapshot_dir,
            dry_run=dry_run,
        )

    assessments = [
        assess_site_stability(
            config.complex_input.reference_structure_path,
            last_snapshot["last_snapshot_pdb"],
            candidate,
            diffusion_cutoff_angstrom=config.snapshot.diffusion_cutoff_angstrom,
            retained_donor_cutoff_angstrom=config.snapshot.retained_donor_cutoff_angstrom,
        )
        for candidate in candidates
    ]
    record_site_analysis(output_path=snapshot_dir / "site_analysis.json", candidates=candidates, assessments=assessments)
    selected_assessments = [next(item for item in assessments if item.site == item_site.site) for item_site in selected_sites]
    selected_assessment = selected_assessments[0]
    unstable_assessments = [item for item in selected_assessments if not item.stable]

    if unstable_assessments:
        print_notice(
            "Strong Warning",
            "\n".join(item.note for item in unstable_assessments),
            border_style="bold red",
        )
        if config.snapshot.mode == SnapshotMode.CLUSTER:
            raise RuntimeError(
                "Cluster analysis is disabled for unstable metal sites because the last snapshot already shows "
                "that the metal diffused away from the binding site."
            )
        if not config.snapshot.allow_unstable_last_snapshot:
            raise RuntimeError(
                "The selected metal site is unstable in the last snapshot. "
                "Set snapshot.allow_unstable_last_snapshot = true to proceed with the last snapshot anyway."
            )

    snapshot_source = "last"
    selected_snapshot_pdb = Path(last_snapshot["last_snapshot_pdb"])
    selected_snapshot_rst7 = Path(last_snapshot["last_snapshot_rst7"])
    cluster_manifest: dict[str, str] | None = None
    if config.snapshot.mode == SnapshotMode.CLUSTER and all(item.stable for item in selected_assessments):
        cluster_atom_indices: set[int] = set()
        for selected_site in selected_sites:
            mask = build_cluster_atom_mask(
                config.complex_input.reference_structure_path,
                selected_site,
                radius_angstrom=config.snapshot.cluster_radius_angstrom,
            )
            if mask.startswith("@"):
                cluster_atom_indices.update(int(token) for token in mask[1:].split(",") if token)
        atom_mask = _atom_mask_from_indices(sorted(cluster_atom_indices))
        with activity_status(
            "[blink bold cyan]Processing...[/] Running cluster analysis to select a representative snapshot.",
            plain_message="Processing... Running cluster analysis to select a representative snapshot.",
        ):
            cluster_manifest = run_cluster_representative_selection(
                prmtop_path=config.complex_input.prmtop_path,
                trajectory_path=config.complex_input.trajectory_path,
                reference_structure_path=config.complex_input.reference_structure_path,
                atom_mask=atom_mask,
                output_dir=snapshot_dir / "cluster",
                epsilon_angstrom=config.snapshot.cluster_epsilon_angstrom,
                sieve=config.snapshot.cluster_sieve,
                dry_run=dry_run,
            )
        selected_snapshot_pdb = Path(cluster_manifest["representative_snapshot_pdb"])
        selected_snapshot_rst7 = Path(cluster_manifest["representative_snapshot_rst7"])
        snapshot_source = "cluster"

    production_restart_path = _resolve_production_restart_path(config) if snapshot_source == "last" else None
    start_coordinate_path = (
        production_restart_path or _resolve_coordinate_restart_path(config)
        if snapshot_source == "last"
        else None
    )
    copied_selected_pdb, copied_selected_rst7 = _write_bound_snapshot_placeholders(
        snapshot_source_pdb=selected_snapshot_pdb,
        snapshot_source_rst7=selected_snapshot_rst7,
        start_restart_source=start_coordinate_path,
        output_dir=root,
    )
    bound_start_source = "production_restart" if production_restart_path is not None else "cpu_prep"

    inherited_settings = parse_cntrl_settings(config.complex_input.production_mdin_path)
    bound_dir = root / "bound"
    bound_dir.mkdir(parents=True, exist_ok=True)
    bound_counterion_plan: CounterionPlan | None = None
    bound_source_prmtop = Path(config.complex_input.prmtop_path)
    bound_start_coord = copied_selected_rst7
    all_alchemical_atom_indices = list(selected_atom_indices)
    bound_counterion_mask: str | None = None
    if config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS:
        bound_counterion_plan = prepare_charge_compensating_counterions(
            source_prmtop=bound_source_prmtop,
            source_pdb=copied_selected_pdb,
            source_coord=copied_selected_rst7,
            metal_atom_indices=selected_atom_indices,
            formal_charges=[formal_charges_by_site[item.site] for item in selected_sites],
            output_dir=bound_dir / "counterion_prep",
            water_model=config.water_reference.water_model,
            ion_frcmods=ti_ion_frcmods,
            amber_env=amber_env,
            config=config.ti,
            dry_run=dry_run,
        )
        bound_source_prmtop = Path(bound_counterion_plan.topology_path)
        bound_start_coord = Path(bound_counterion_plan.start_coord_path)
        bound_counterion_mask = bound_counterion_plan.counterion_mask
        all_alchemical_atom_indices.extend(bound_counterion_plan.counterion_atom_indices)
        all_alchemical_atom_indices = sorted(set(all_alchemical_atom_indices))
        alchemical_atom_mask = _atom_mask_from_indices(all_alchemical_atom_indices)
        if bound_counterion_plan.requires_preparation:
            bound_start_source = "counterion_rebuild_cpu_prep"
    bound_prmtop = Path(_copy_if_present(bound_source_prmtop, bound_dir) or bound_source_prmtop)
    bound_ti_topology = prepare_ti_input_topology(
        input_prmtop=bound_prmtop,
        output_dir=bound_dir,
        label="bound",
        implementation_mode=config.ti.implementation_mode,
        dry_run=dry_run,
        water_model=config.water_reference.water_model,
        official_126_frcmods=ti_ion_frcmods,
        alchemical_atom_index=selected.atom_index,
        alchemical_element=selected.element,
        alchemical_charge=formal_charge,
        alchemical_atom_indices=all_alchemical_atom_indices,
        alchemical_elements_by_index={item.atom_index: item.element for item in selected_sites},
        alchemical_charges_by_index={item.atom_index: formal_charges_by_site[item.site] for item in selected_sites},
    )
    if _uses_in_place_bound_ti(config):
        restraint_setup = None
        restraint_setups = []
        combined_restraint_file = None
        restraint_payload = {
            "scheme_version": "none_in_place_v1",
            "restraint_file": None,
            "sites": [],
            "correction_kcal_mol": 0.0,
        }
    elif multi_site:
        restraint_setups = [
            build_bound_site_restraint(
                reference_structure_path=config.complex_input.reference_structure_path,
                candidate=item,
                anchor_count=config.ti.restraint_anchor_count,
                force_constant=config.ti.restraint_force_constant,
                half_width_angstrom=config.ti.restraint_half_width_angstrom,
                temperature_k=inherited_settings.temperature_k,
                output_path=bound_dir / "restraints" / f"bound_site_{item.site}.disang",
            )
            for item in selected_sites
        ]
        combined_restraint_file = write_combined_bound_site_restraints(
            setups=restraint_setups,
            output_path=bound_dir / "restraints" / "bound_site.disang",
        )
        restraint_setup = restraint_setups[0]
        restraint_payload = {
            "scheme_version": "flat_bottom_group_multi_v1",
            "restraint_file": combined_restraint_file,
            "sites": [setup.to_dict() for setup in restraint_setups],
            "correction_kcal_mol": sum(setup.correction_kcal_mol for setup in restraint_setups),
        }
    else:
        restraint_setup = build_bound_site_restraint(
            reference_structure_path=config.complex_input.reference_structure_path,
            candidate=selected,
            anchor_count=config.ti.restraint_anchor_count,
            force_constant=config.ti.restraint_force_constant,
            half_width_angstrom=config.ti.restraint_half_width_angstrom,
            temperature_k=inherited_settings.temperature_k,
            output_path=bound_dir / "restraints" / "bound_site.disang",
        )
        restraint_setups = [restraint_setup]
        combined_restraint_file = restraint_setup.restraint_file
        restraint_payload = restraint_setup.to_dict()
    restraint_correction = sum(setup.correction_kcal_mol for setup in restraint_setups)
    bound_decharged = prepare_decharged_topology(
        input_prmtop=Path(str(bound_ti_topology["ti_prmtop"])),
        atom_mask=alchemical_atom_mask,
        output_dir=bound_dir,
        label="bound",
        dry_run=dry_run,
        preserve_c4=_uses_gti_1264(config),
    )
    bound_neutrality_validation = _neutrality_validation(
        initial_prmtop=bound_ti_topology["ti_prmtop"],
        endpoint_prmtop=bound_decharged["decharged_prmtop"],
        enabled=config.ti.charge_compensation_mode
        == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS,
        dry_run=dry_run,
    )
    bound_qoff_topology: dict[str, Any] | None = None
    bound_qoff_restraint_file: str | None = None
    if _uses_split_qoff_disjoint_topology(config):
        bound_qoff_topology = prepare_qoff_disjoint_topology(
            input_prmtop=Path(str(bound_ti_topology["ti_prmtop"])),
            output_dir=bound_dir,
            label="bound",
            alchemical_atom_index=selected.atom_index,
            alchemical_atom_indices=all_alchemical_atom_indices,
            dry_run=dry_run,
        )
        if multi_site and restraint_setups:
            bound_qoff_restraint_file = _relative_path(
                write_qoff_duplicate_bound_site_restraints(
                    setups=restraint_setups,
                    duplicate_metal_atom_indices=_qoff_duplicate_map_from_metadata(bound_qoff_topology),
                    output_path=bound_dir / "restraints" / "bound_site_qoff_disjoint.disang",
                ),
                from_dir=bound_dir,
            )
        elif restraint_setup is not None:
            bound_qoff_restraint_file = _relative_path(
                write_qoff_duplicate_bound_site_restraint(
                    setup=restraint_setup,
                    duplicate_metal_atom_index=int(bound_qoff_topology["qoff_duplicate_atom_index"]),
                    output_path=bound_dir / "restraints" / "bound_site_qoff_disjoint.disang",
                ),
                from_dir=bound_dir,
            )
    bound_qoff_counterion_mask = bound_counterion_mask
    if (
        bound_qoff_topology is not None
        and bound_counterion_plan is not None
        and bound_counterion_plan.counterion_atom_indices
    ):
        duplicate_map = _qoff_duplicate_map_from_metadata(bound_qoff_topology)
        qoff_counterion_indices = [
            *bound_counterion_plan.counterion_atom_indices,
            *[
                duplicate_map[index]
                for index in bound_counterion_plan.counterion_atom_indices
                if index in duplicate_map
            ],
        ]
        bound_qoff_counterion_mask = _atom_mask_from_indices(sorted(set(qoff_counterion_indices)))
    bound_restraint_path = (
        None
        if combined_restraint_file is None
        else Path(_relative_path(combined_restraint_file, from_dir=bound_dir)).as_posix()
    )
    bound_start_prep_stages = None
    if production_restart_path is None or (
        bound_counterion_plan is not None and bound_counterion_plan.requires_preparation
    ):
        bound_prep_config = config.ti
        if bound_counterion_plan is not None and bound_counterion_plan.requires_preparation:
            bound_prep_config = config.ti.model_copy(
                update={
                    "bound_start_min_cycles": config.ti.counterion_prep_min_cycles,
                    "bound_start_eq_ns": config.ti.counterion_prep_eq_ns,
                }
            )
        bound_start_prep_stages = generate_bound_start_preparation_inputs(
            config=bound_prep_config,
            inherited_settings=inherited_settings,
            restraint_file=bound_restraint_path,
            output_dir=bound_dir,
            positional_restraint_mask=bound_counterion_mask,
        )
    bound_windows = generate_ti_inputs(
        config=config.ti,
        inherited_settings=inherited_settings,
        atom_mask=alchemical_atom_mask,
        restraint_file=bound_restraint_path,
        output_dir=bound_dir,
        qoff_start_source="restart",
        qoff_timask1=None if bound_qoff_topology is None else str(bound_qoff_topology["qoff_timask1"]),
        qoff_timask2=None if bound_qoff_topology is None else str(bound_qoff_topology["qoff_timask2"]),
        qoff_charge_mask=None if bound_qoff_topology is None else str(bound_qoff_topology["qoff_crgmask"]),
        qoff_restraint_file=None if bound_qoff_restraint_file is None else Path(bound_qoff_restraint_file).as_posix(),
        positional_restraint_mask=bound_counterion_mask,
        qoff_positional_restraint_mask=bound_qoff_counterion_mask,
    )
    bound_endpoint_prep_stages = None
    if not _uses_combined_gti_decoupling(config):
        bound_endpoint_prep_stages = generate_qoff_endpoint_preparation_inputs(
            config=config.ti,
            inherited_settings=inherited_settings,
            restraint_file=bound_restraint_path,
            output_dir=bound_dir,
            positional_restraint_mask=bound_counterion_mask,
        )

    water_dir: Path | None = None
    water_manifest: dict[str, Any] | None = None
    water_reused = False
    water_decharged_topology: Path | None = None
    water_prep_stages = None
    water_endpoint_prep_stages = None
    water_windows = None
    water_qoff_metadata: dict[str, Any] | None = None
    water_runtime_output_dir: Path | None = None
    water_slurm = None
    water_term_source = "disabled" if _uses_in_place_bound_only_ti(config) else "simulation"
    water_library_key: str | None = None
    water_library_snapshot: dict[str, Any] | None = None

    if config.water_reference.enabled and config.water_reference.reuse_from_library and multi_site:
        raise ValueError(
            "Water-reference library reuse is only supported for single-metal TI cases. "
            "Disable water_reference.reuse_from_library for all-at-once multi-metal TI."
        )

    if not config.water_reference.enabled:
        water_reused = False
    elif config.water_reference.reuse_from_library:
        water_term_source = "library"
        water_library_key = config.water_reference.library_key
        water_library_snapshot = ti_abfe.get_water_library_entry_by_key(
            water_library_key or "",
            sampling_selection=(
                ti_abfe.SAMPLING_SELECTION_FORWARD_REVERSE
                if config.ti.sampling_mode == TISamplingMode.BIDIRECTIONAL
                else ti_abfe.SAMPLING_SELECTION_FORWARD_ONLY
            ),
        )
        if water_library_snapshot is None:
            raise ValueError(
                "The selected water-reference library entry could not be found. "
                "Run analyses.py first or choose a fresh water-reference simulation."
            )
        water_reused = True
    else:
        with activity_status(
            "[blink bold cyan]Processing...[/] Checking the reusable water-reference directory and preparing the solvent leg.",
            plain_message="Processing... Checking the reusable water-reference directory and preparing the solvent leg.",
        ):
            if multi_site:
                water_dir, water_manifest, water_reused = _prepare_multi_water_reference(
                    config=config,
                    inherited_settings=inherited_settings,
                    selected_sites=selected_sites,
                    formal_charges_by_site=formal_charges_by_site,
                    amber_env=amber_env,
                    official_126_frcmods=ti_ion_frcmods,
                    dry_run=dry_run,
                )
            else:
                water_dir, water_manifest, water_reused = _prepare_water_reference(
                    config=config,
                    inherited_settings=inherited_settings,
                    metal_element=selected.element,
                    formal_charge=formal_charge,
                    amber_env=amber_env,
                    official_126_frcmods=ti_ion_frcmods,
                    dry_run=dry_run,
                )
        water_decharged_topology = water_dir / "water_decharged.prmtop"
        water_counterion_mask = water_manifest.get("counterion_mask")
        water_qoff_counterion_mask = water_manifest.get("qoff_counterion_mask")
        water_alchemical_atom_mask = str(
            water_manifest.get("alchemical_atom_mask")
            or (_atom_mask_from_indices(list(range(1, len(selected_sites) + 1))) if multi_site else "@1")
        )
        water_prep_stages = generate_water_reference_preparation_inputs(
            inherited_settings=inherited_settings,
            output_dir=water_dir,
            positional_restraint_mask=None if water_counterion_mask is None else str(water_counterion_mask),
            restraint_force_constant=config.ti.counterion_restraint_force_constant,
        )
        water_qoff_metadata = (
            None if _uses_combined_gti_decoupling(config) else _qoff_disjoint_metadata_from_manifest(water_manifest)
        )
        water_windows = generate_ti_inputs(
            config=config.ti,
            inherited_settings=inherited_settings,
            atom_mask=water_alchemical_atom_mask,
            restraint_file=None,
            output_dir=water_dir,
            qoff_start_source="restart",
            qoff_timask1=None
            if water_qoff_metadata is None
            else str(water_qoff_metadata["qoff_timask1"]),
            qoff_timask2=None
            if water_qoff_metadata is None
            else str(water_qoff_metadata["qoff_timask2"]),
            qoff_charge_mask=None
            if water_qoff_metadata is None
            else str(water_qoff_metadata["qoff_crgmask"]),
            positional_restraint_mask=None if water_counterion_mask is None else str(water_counterion_mask),
            qoff_positional_restraint_mask=None
            if water_qoff_counterion_mask is None
            else str(water_qoff_counterion_mask),
        )
        if not _uses_combined_gti_decoupling(config):
            water_endpoint_prep_stages = generate_qoff_endpoint_preparation_inputs(
                config=config.ti,
                inherited_settings=inherited_settings,
                restraint_file=None,
                output_dir=water_dir,
                positional_restraint_mask=None if water_counterion_mask is None else str(water_counterion_mask),
            )

    slurm_dir = root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    bound_runtime_output_dir = root / "output"
    bound_qoff_bridge = (
        None
        if bound_qoff_topology is None
        else QoffCoordinateBridge(
            original_atom_index=int(bound_qoff_topology["qoff_original_atom_index"]),
            duplicate_atom_index=int(bound_qoff_topology["qoff_duplicate_atom_index"]),
            atom_pairs=_qoff_pairs_from_metadata(bound_qoff_topology),
        )
    )
    bound_slurm = write_leg_slurm_scripts(
        leg_name="bound",
        input_root=str(bound_dir.resolve()),
        runtime_output_root=str(bound_runtime_output_dir.resolve()),
        prep_stages=bound_start_prep_stages,
        prep_prmtop=None if bound_start_prep_stages is None else str(Path(str(bound_ti_topology["ti_prmtop"])).resolve()),
        prep_start_coord=None if bound_start_prep_stages is None else str(bound_start_coord.resolve()),
        endpoint_prep_stages=bound_endpoint_prep_stages,
        endpoint_prep_prmtop=str(Path(bound_decharged["decharged_prmtop"]).resolve()),
        windows=bound_windows,
        slurm_config=slurm_config,
        qoff_prmtop=str(
            Path(str(bound_qoff_topology["qoff_prmtop"] if bound_qoff_topology else bound_ti_topology["ti_prmtop"])).resolve()
        ),
        vdw_prmtop=str(Path(bound_decharged["decharged_prmtop"]).resolve()),
        start_coord=str(bound_start_coord.resolve()),
        qoff_coordinate_bridge=bound_qoff_bridge,
        ti_config=config.ti,
        output_dir=slurm_dir,
    )
    if water_dir is not None and water_manifest is not None and water_decharged_topology is not None and water_windows is not None:
        water_slurm_dir = water_dir / "slurm"
        water_slurm_dir.mkdir(parents=True, exist_ok=True)
        water_runtime_output_dir = water_dir / "output"
        water_qoff_bridge = (
            None
            if water_qoff_metadata is None
            else QoffCoordinateBridge(
                original_atom_index=int(water_qoff_metadata["qoff_original_atom_index"]),
                duplicate_atom_index=int(water_qoff_metadata["qoff_duplicate_atom_index"]),
                atom_pairs=_qoff_pairs_from_metadata(water_qoff_metadata),
            )
        )
        water_slurm = write_leg_slurm_scripts(
            leg_name="water_ref",
            input_root=str(water_dir.resolve()),
            runtime_output_root=str(water_runtime_output_dir.resolve()),
            prep_stages=water_prep_stages,
            prep_prmtop=str(Path(str(water_manifest["ti_input_prmtop"])).resolve()),
            prep_start_coord=str(
                Path(str(water_manifest.get("system_start_coord") or (water_dir / "system.inpcrd"))).resolve()
            ),
            endpoint_prep_stages=water_endpoint_prep_stages,
            endpoint_prep_prmtop=str(water_decharged_topology.resolve()),
            windows=water_windows,
            slurm_config=slurm_config,
            qoff_prmtop=str(Path(str(water_manifest.get("qoff_prmtop") or water_manifest["ti_input_prmtop"])).resolve()),
            vdw_prmtop=str(water_decharged_topology.resolve()),
            start_coord=str(Path(water_manifest["prep_start_coord"]).resolve()),
            qoff_coordinate_bridge=water_qoff_bridge,
            ti_config=config.ti,
            output_dir=water_slurm_dir,
        )

    result = {
        "output_dir": str(root),
        "selected_metal": _selected_sites_label(selected_sites),
        "selected_metals": _selected_sites_label(selected_sites),
        "ti_selection_mode": config.metal.selection_mode.value,
        "ti_implementation_mode": config.ti.implementation_mode.value,
        "ti_decoupling_mode": config.ti.decoupling_mode.value,
        "ti_sampling_mode": config.ti.sampling_mode.value,
        "ti_sampling_protocol": {
            "replicas": 1,
            "window_equilibration_ns": config.ti.window_equilibration_ns
            if config.ti.sampling_mode == TISamplingMode.BIDIRECTIONAL
            else 0.0,
            "directions": ["forward", "reverse"]
            if config.ti.sampling_mode == TISamplingMode.BIDIRECTIONAL
            else ["forward"],
        },
        "charge_compensation_mode": config.ti.charge_compensation_mode.value,
        "bound_counterion_plan": None
        if bound_counterion_plan is None
        else bound_counterion_plan.to_dict(),
        "bound_neutrality_validation": bound_neutrality_validation,
        "selected_site": selected.to_dict(),
        "selected_sites": [item.to_dict() for item in selected_sites],
        "selected_formal_charge": formal_charge,
        "selected_formal_charges_by_site": {str(site): charge for site, charge in formal_charges_by_site.items()},
        "snapshot_source": snapshot_source,
        "last_snapshot_assessment": selected_assessment.to_dict(),
        "last_snapshot_assessments": [item.to_dict() for item in selected_assessments],
        "inherited_md_settings": inherited_settings.to_dict(),
        "restraint": restraint_payload,
        "restraint_correction_kcal_mol": restraint_correction,
        "restraint_corrections_by_site": {
            str(site.site): setup.correction_kcal_mol
            for site, setup in zip(selected_sites, restraint_setups, strict=True)
        }
        if restraint_setups
        else {},
        "water_term_source": water_term_source,
        "water_library_key": water_library_key,
        "water_library_snapshot": water_library_snapshot,
        "water_reference_reused": water_reused,
        "water_reference_dir": None if water_dir is None else str(water_dir),
        "water_reference_manifest": water_manifest,
        "water_reference_manifest_path": None if water_dir is None else str(water_dir / "water_reference_manifest.json"),
        "water_reference_start_coord": None if water_manifest is None else str(water_manifest.get("prep_start_coord", "N/A")),
        "official_12_6_frcmods": ti_ion_frcmods if not _uses_gti_1264(config) else [],
        "ti_ion_frcmods": ti_ion_frcmods,
        "effective_slurm_profile": slurm_config.profile.value,
        "bound_ti_input_topology": str(bound_ti_topology["ti_prmtop"]),
        "bound_qoff_topology": None if bound_qoff_topology is None else str(bound_qoff_topology["qoff_prmtop"]),
        "water_ti_input_topology": None if water_manifest is None else str(water_manifest["ti_input_prmtop"]),
        "water_qoff_topology": None if water_manifest is None else str(water_manifest.get("qoff_prmtop") or water_manifest["ti_input_prmtop"]),
        "bound_runtime_output_dir": str(bound_runtime_output_dir),
        "water_runtime_output_dir": None if water_runtime_output_dir is None else str(water_runtime_output_dir),
        "bound_slurm": str(bound_slurm),
        "water_slurm": None if water_slurm is None else str(water_slurm),
        "bound_decharged_topology": bound_decharged["decharged_prmtop"],
        "water_decharged_topology": None if water_decharged_topology is None else str(water_decharged_topology),
        "bound_start_source": bound_start_source,
        "bound_start_restart": str(bound_start_coord),
        "bound_production_restart_source": None if production_restart_path is None else str(production_restart_path),
        "bound_start_preparation": None
        if bound_start_prep_stages is None
        else {
            "prep_prmtop": str(Path(str(bound_ti_topology["ti_prmtop"])).resolve()),
            "prep_start_coord": str(bound_start_coord.resolve()),
            "prep_manifest": str(bound_dir / "bound_start_prep" / "prep_manifest.json"),
            "stages": [stage.to_dict() for stage in bound_start_prep_stages],
        },
        "snapshot_paths": {
            "selected_pdb": str(copied_selected_pdb),
            "selected_rst7": str(copied_selected_rst7),
            "cluster_manifest": cluster_manifest,
        },
        "copied_inputs": copied_inputs,
    }
    manifest_path = write_json(root / "manifest.json", result)
    result["manifest"] = str(manifest_path)
    return result
