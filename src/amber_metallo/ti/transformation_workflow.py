from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from amber_metallo.reporting import write_json
from amber_metallo.ti.config import TIEndpointMode, TIImplementationMode, TIDecouplingMode
from amber_metallo.ti.coordination import build_coordination_restraints
from amber_metallo.ti.protocols import generate_bound_start_preparation_inputs, generate_ti_inputs, generate_water_reference_preparation_inputs
from amber_metallo.ti.slurm import QoffCoordinateBridge, write_leg_slurm_scripts
from amber_metallo.ti.transformation import mass_path_description, paired_restraints, prepare_metal_transformation


def validate_direct_config(config):
    if config.transformation.mode != TIEndpointMode.METAL:
        return
    if (config.ti.implementation_mode != TIImplementationMode.AMBER_12_6_4_GTI
            or config.ti.decoupling_mode != TIDecouplingMode.COMBINED_Q_VDW):
        raise ValueError("Direct metal transformation requires amber_12_6_4_gti + combined_q_vdw.")
    if config.water_reference.reuse_from_library:
        raise ValueError("Direct metal transformation needs a matching metal-to-metal water leg; dummy water-library results cannot be reused.")
    endpoints = [*config.transformation.endpoints_by_site.values()]
    if config.transformation.endpoint is not None:
        endpoints.append(config.transformation.endpoint)
    if config.water_reference.enabled and any(p.water_model != config.water_reference.water_model for p in endpoints):
        raise ValueError("The water reference model must match every metal endpoint's water model.")


def _write_leg(*, config, directory, output_dir, source_topology, source_pdb, source_coord,
               selected_sites, inherited_settings, amber_env, dry_run, restraint=None, water=False):
    directory.mkdir(parents=True, exist_ok=True)
    plan = prepare_metal_transformation(source_prmtop=source_topology, source_pdb=source_pdb,
        selected_sites=selected_sites, config=config, amber_env=amber_env, output_dir=directory, dry_run=dry_run,
        temperature_k=inherited_settings.temperature_k)
    source_mask = ("@" + ",".join(map(str, plan["counterion_indices"]))) if plan["counterion_indices"] else None
    paired_path = None
    if restraint:
        paired_path = paired_restraints(restraint, plan["atom_pairs"], directory / "restraints" / "metal_transform.disang")
    prep = (generate_water_reference_preparation_inputs(inherited_settings=inherited_settings, output_dir=directory,
                positional_restraint_mask=source_mask, restraint_force_constant=config.ti.counterion_restraint_force_constant)
            if water else generate_bound_start_preparation_inputs(config=config.ti, inherited_settings=inherited_settings,
                restraint_file=restraint, output_dir=directory, positional_restraint_mask=source_mask))
    protocol = config.ti.model_copy(update={"qoff_dt_ps": min(config.ti.qoff_dt_ps, 0.001)})
    windows = generate_ti_inputs(config=protocol, inherited_settings=inherited_settings,
        atom_mask=plan["timask1"], restraint_file=paired_path, output_dir=directory,
        qoff_start_source="restart", transformation=plan,
        positional_restraint_mask=plan["counterion_mask"])
    from amber_metallo.config import SlurmProfile
    slurm = write_leg_slurm_scripts(leg_name="water_ref" if water else "bound",
        input_root=str(directory.resolve()), runtime_output_root=str(output_dir.resolve()),
        prep_stages=prep, prep_prmtop=str(Path(source_topology).resolve()), prep_start_coord=str(Path(source_coord).resolve()),
        endpoint_prep_stages=None, endpoint_prep_prmtop=None,
        windows=windows, slurm_config=config.slurm.model_copy(update={"profile": SlurmProfile.GPU}),
        qoff_prmtop=str(Path(plan["ti_prmtop"]).resolve()), vdw_prmtop=str(Path(plan["ti_prmtop"]).resolve()),
        start_coord=str(Path(source_coord).resolve()),
        qoff_coordinate_bridge=QoffCoordinateBridge(atom_pairs=plan["atom_pairs"], use_gpu=True),
        ti_config=protocol, output_dir=directory / "slurm" if water else directory.parent / "slurm")
    return plan, slurm


def _water_leg(*, config, selected_sites, inherited_settings, amber_env, bound_plan, dry_run):
    from amber_metallo.amber.leap import build_system_with_tleap
    from amber_metallo.config import DESC4ParameterSet, MetalChargeAssignment, MetalModel, SaltConfig, SaltMode, SystemConfig
    from amber_metallo.ti.workflow import _build_multi_metal_only_pdb
    from amber_metallo.ti.metal_parameters import available_metal_endpoints
    # A direct reference has its own endpoint identity and cannot collide with
    # the historical single-metal -> dummy water cache.
    root = config.output_path() / "water_reference"
    root.mkdir(parents=True, exist_ok=True)
    charges = {site.site: round(next(s["source_charge"] for s in bound_plan["sites"] if s["site"] == site.site))
               for site in selected_sites}
    water_model = config.transformation.for_site(selected_sites[0].site).water_model
    catalog = available_metal_endpoints(amber_env=amber_env, water_model=water_model)
    source_parameters = []
    for site in selected_sites:
        source = next(s["source_parameters"] for s in bound_plan["sites"] if s["site"] == site.site)
        matches = [p for p in catalog if p.element == site.element and p.formal_charge == charges[site.site]
                   and abs(p.rmin_half - source["rmin_half"]) < 1e-4
                   and abs(p.epsilon - source["epsilon"]) < 1e-5
                   and abs(p.c4 - source["c4"]) < 1e-3]
        if len(matches) != 1:
            # Fe has identical parameters in the two OPC tables.
            if matches and all((p.rmin_half, p.epsilon, p.c4) == (matches[0].rmin_half, matches[0].epsilon, matches[0].c4) for p in matches):
                matches = matches[:1]
            else:
                raise ValueError(f"Cannot identify matching source {site.element} 12-6-4 parameters for the water leg.")
        source_parameters.append(matches[0])
    families = {p.parameter_set for p in source_parameters}
    if len(families) != 1:
        raise ValueError("A simultaneous water reference currently requires a single source parameter family.")
    family = source_parameters[0].parameter_set
    frcmods = list(dict.fromkeys(p.frcmod_path for p in source_parameters))
    frcmods.extend(str(p) for p in amber_env.matching_monovalent_1264_files(water_model, include_bundled_opc=family == "duvail")
                   if str(p) not in frcmods)
    pdb = _build_multi_metal_only_pdb(selected_sites=selected_sites, output_path=root / "metal_input.pdb")
    system = SystemConfig(protein_ff="ff19SB", ligand_ff="gaff2", metal_model=MetalModel.MODEL_1264,
        metal_charges=[MetalChargeAssignment(site=i, charge=charges[site.site]) for i, site in enumerate(selected_sites, start=1)],
        water_model=water_model, box_shape=config.water_reference.box_shape, buffer_angstrom=config.water_reference.buffer_angstrom,
        c4_parameter_set=DESC4ParameterSet.OPC_DUVAIL if family == "duvail" else DESC4ParameterSet.SPCE_LIMERZ,
        salt=SaltConfig(mode=SaltMode.NEUTRALIZE), custom_ion_frcmods=frcmods)
    build_system_with_tleap(system_config=system, amber_env=amber_env, prepared_pdb=pdb,
                           ligand_artifacts=[], source_files=[], output_dir=root, dry_run=dry_run)
    water_config = config.model_copy(deep=True)
    water_config.transformation = config.transformation.model_copy(deep=True, update={"endpoint": None,
        "endpoints_by_site": {i: config.transformation.for_site(site.site) for i, site in enumerate(selected_sites, start=1)}})
    water_config.metal.formal_charges_by_site = {i: charges[site.site] for i, site in enumerate(selected_sites, start=1)}
    sites = [replace(site, site=i, atom_index=i) for i, site in enumerate(selected_sites, start=1)]
    if dry_run and (not (root / "system.prmtop").exists() or "%FLAG" not in (root / "system.prmtop").read_text()):
        # tleap is deliberately not executed in a dry run. Record the deferred
        # water transformation, never fabricate a topology or executable leg.
        manifest = {"transformation": config.transformation.model_dump(mode="json"), "status": "deferred_dry_run",
                    "metal": ",".join(site.element for site in sites), "formal_charge": sum(charges.values()),
                    "water_model": water_model, "water_reference_dir": str(root)}
        write_json(root / "water_reference_manifest.json", manifest)
        return root, manifest, None
    # Keep the exact source masses from the bound topology, including isotope
    # choices and table rounding, so the two physical kinetic terms cancel.
    from amber_metallo.ti.topology import _parse_prmtop_document, _tokens_to_floats, _render_section_values, _render_prmtop
    water_topology = root / "system.prmtop"
    version, sections = _parse_prmtop_document(water_topology.read_text(encoding="utf-8"))
    water_masses = _tokens_to_floats(sections["MASS"])
    bound_masses = {site["site"]: site["source_mass_da"] for site in bound_plan["sites"]}
    for i, site in enumerate(selected_sites):
        water_masses[i] = bound_masses[site.site]
    sections["MASS"].data_lines = _render_section_values(sections["MASS"].format_line, water_masses)
    water_topology.write_text(_render_prmtop(version, sections), encoding="utf-8")
    plan, slurm = _write_leg(config=water_config, directory=root, output_dir=root / "output",
        source_topology=root / "system.prmtop", source_pdb=root / "system.pdb", source_coord=root / "system.inpcrd",
        selected_sites=sites, inherited_settings=inherited_settings, amber_env=amber_env, dry_run=dry_run, water=True)
    for source_site, water_site in zip(bound_plan["sites"], plan["sites"]):
        # Detect source tuning/polarizability changes that a standard water
        # build cannot reproduce, even if its LJ and metal-water C4 agree.
        if abs(source_site["source_parameters"]["self_c4"] - water_site["source_parameters"]["self_c4"]) > 1e-3:
            raise ValueError("Generated water reference does not reproduce the source metal C4 parameters. "
                             "Use matching source polarizabilities/tuning or disable the automatic water reference.")
    manifest = {"transformation": plan, "metal": ",".join(site.element for site in sites),
                "formal_charge": sum(charges.values()), "water_model": water_model,
                "ti_protocol": config.ti.model_dump(mode="json"),
                "neutrality_validation": {"status": "passed" if config.ti.charge_compensation_mode.value != "none" else "not_requested",
                                           "initial_charge": plan["initial_charge"], "endpoint_charge": plan["endpoint_charge"]},
                "ti_input_prmtop": plan["ti_prmtop"], "prep_start_coord": str(root / "system.inpcrd")}
    write_json(root / "water_reference_manifest.json", manifest)
    return root, manifest, slurm


def run_direct_metal_workflow(*, config, selected_sites, reference_selected_sites, selected_pdb, selected_rst7,
                              inherited_settings, amber_env, snapshot_source, assessments, copied_inputs, dry_run):
    validate_direct_config(config)
    root = config.output_path()
    bound = root / "bound"
    bound.mkdir(parents=True, exist_ok=True)
    restraint = None
    restraint_payload = {"scheme_version": "none_direct_metal", "correction_kcal_mol": 0.0}
    coordination = config.ti.coordination_restraint
    if coordination is not None and coordination.enabled:
        restraint_payload = build_coordination_restraints(
            reference_pdb=config.complex_input.reference_structure_path, snapshot_pdb=selected_pdb,
            source_prmtop=config.complex_input.prmtop_path, ti_prmtop=config.complex_input.prmtop_path,
            candidates=reference_selected_sites, config=coordination, output_dir=bound / "restraints", dry_run=dry_run)
        restraint = restraint_payload["restraint_file"]
        restraint_payload.update({"active_at_dummy_endpoint": False, "endpoint": "metal",
            "warning": "Restraints define both endpoint ensembles. No restraint release correction is computed."})
        write_json(bound / "restraints" / "coordination.json", restraint_payload)
        (bound / "restraints" / "README.txt").write_text(restraint_payload["warning"] + "\n", encoding="utf-8")
    plan, slurm = _write_leg(config=config, directory=bound, output_dir=root / "output",
        source_topology=config.complex_input.prmtop_path, source_pdb=selected_pdb, source_coord=selected_rst7,
        selected_sites=selected_sites, inherited_settings=inherited_settings, amber_env=amber_env, dry_run=dry_run, restraint=restraint)
    water_root, water_manifest, water_slurm = None, None, None
    if config.water_reference.enabled:
        water_root, water_manifest, water_slurm = _water_leg(config=config, selected_sites=selected_sites,
            inherited_settings=inherited_settings, amber_env=amber_env, bound_plan=plan, dry_run=dry_run)
    labels = "; ".join(f"site {s['site']}: {s['source_element']}{s['source_charge']:g}+ -> "
                       f"{s['endpoint']['element']}{s['endpoint']['formal_charge']}+" for s in plan["sites"])
    result = {
        "output_dir": str(root), "selected_metal": labels, "selected_metals": labels,
        "selected_site": selected_sites[0].to_dict(), "selected_sites": [site.to_dict() for site in selected_sites],
        "selected_formal_charge": plan["sites"][0]["source_charge"],
        "selected_formal_charges_by_site": {str(s["site"]): s["source_charge"] for s in plan["sites"]},
        "ti_selection_mode": config.metal.selection_mode.value,
        "ti_implementation_mode": config.ti.implementation_mode.value, "ti_decoupling_mode": "direct_metal",
        "ti_sampling_mode": config.ti.sampling_mode.value, "transformation": plan,
        "delta_g_definition": "Configurational G(end metal) - G(source metal); kinetic mass term reported separately",
        "mass_treatment": mass_path_description(plan),
        "charge_compensation_mode": config.ti.charge_compensation_mode.value,
        "bound_neutrality_validation": {"status": "passed" if config.ti.charge_compensation_mode.value != "none" else "not_requested",
                                       "initial_charge": plan["initial_charge"], "endpoint_charge": plan["endpoint_charge"]},
        "restraint": restraint_payload, "restraint_correction_kcal_mol": "NaN" if restraint else 0.0,
        "bound_ti_input_topology": plan["ti_prmtop"], "bound_qoff_topology": plan["ti_prmtop"],
        "bound_decharged_topology": None, "bound_start_restart": str(selected_rst7), "bound_start_source": "direct_metal_cpu_prep",
        "bound_runtime_output_dir": str(root / "output"), "bound_slurm": str(slurm),
        "water_reference_dir": str(water_root) if water_root else None, "water_reference_manifest": water_manifest,
        "water_reference_manifest_path": str(water_root / "water_reference_manifest.json") if water_root else None,
        "water_ti_input_topology": (water_manifest or {}).get("ti_input_prmtop"),
        "water_reference_start_coord": (water_manifest or {}).get("prep_start_coord"),
        "water_reference_reused": False, "water_term_source": "simulation" if water_root else "disabled",
        "water_runtime_output_dir": str(water_root / "output") if water_root else None,
        "water_slurm": str(water_slurm) if water_slurm else None,
        "snapshot_source": snapshot_source, "snapshot_paths": {"selected_pdb": str(selected_pdb), "selected_rst7": str(selected_rst7)},
        "last_snapshot_assessment": assessments[0].to_dict(), "last_snapshot_assessments": [a.to_dict() for a in assessments],
        "inherited_md_settings": inherited_settings.to_dict(), "copied_inputs": copied_inputs,
    }
    result["manifest"] = str(write_json(root / "manifest.json", result))
    return result
