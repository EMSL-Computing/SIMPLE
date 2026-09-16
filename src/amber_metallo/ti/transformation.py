"""Direct, matched-atom metal TI (Amber manual 25.1.8.1).

Both endpoints retain a repulsive core. Paired atoms are common TI atoms, so
their coordinates are synchronized and no dummy/softcore leg is necessary.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re

import gemmi

from amber_metallo.c4_assets import opc_duvail_polarizability_file
from amber_metallo.reporting import write_json
from amber_metallo.ti.config import TIChargeCompensationMode, TIMassMode
from amber_metallo.ti.metal_parameters import resolve_metal_endpoint
from amber_metallo.ti.topology import (
    LJParameters, _append_qoff_duplicate_atoms_to_prmtop, _combine_lj_pair,
    _derive_self_lj_parameters, _pair_coefficients, _parse_prmtop_document,
    _render_prmtop, _render_section_values, _section_values, _tokens_to_floats,
    _tokens_to_ints, inspect_prmtop_charge_state,
)


def _mask(indices):
    return "@" + ",".join(str(index) for index in indices)


def _polarizabilities(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and not line.lstrip().startswith("#"):
            values[tokens[0]] = float(tokens[1])
    return values


def prepare_metal_transformation(*, source_prmtop, source_pdb, selected_sites,
                                 config, amber_env, output_dir, dry_run, temperature_k=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_text = Path(source_prmtop).read_text(encoding="utf-8")
    version, source = _parse_prmtop_document(original_text)
    if "LENNARD_JONES_CCOEF" not in source:
        raise ValueError("Direct metal transformation requires an existing 12-6-4 topology with C4 coefficients.")
    state = inspect_prmtop_charge_state(source_prmtop)
    parameters = {site.atom_index: resolve_metal_endpoint(config.transformation.for_site(site.site), amber_env=amber_env)
                  for site in selected_sites}
    unknown = set(config.transformation.endpoints_by_site) - {site.site for site in selected_sites}
    if unknown:
        raise ValueError(f"Endpoint assignments refer to unselected sites: {sorted(unknown)}")
    atom_indices = sorted(parameters)
    source_masses = _tokens_to_floats(source["MASS"])
    source_types = _tokens_to_ints(source["ATOM_TYPE_INDEX"])
    source_labels = _section_values(source["AMBER_ATOM_TYPE"])
    if len({p.water_model for p in parameters.values()}) != 1:
        raise ValueError("All endpoints in a simultaneous transformation must use the same solvent parameter model.")
    for site in selected_sites:
        atom = state.atoms[site.atom_index - 1]
        mass = source_masses[site.atom_index - 1]
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Site {site.site}: source metal mass must be finite and positive.")
        if atom.residue_atom_count != 1:
            raise ValueError("Direct metal TI requires nonbonded, monatomic metal residues.")
        expected = config.metal.formal_charges_by_site.get(site.site, config.metal.formal_charge)
        if expected is not None and abs(atom.charge - expected) > 0.05:
            raise ValueError(f"Site {site.site}: configured source charge {expected} differs from topology charge {atom.charge:g}.")
        for name, width in (("BONDS_INC_HYDROGEN", 3), ("BONDS_WITHOUT_HYDROGEN", 3)):
            if name in source:
                bonds = _tokens_to_ints(source[name])
                if any(abs(bonds[start + offset]) // 3 + 1 == site.atom_index
                       for start in range(0, len(bonds), width) for offset in (0, 1)):
                    raise ValueError("Bonded metal models are not supported by the nonbonded metal transformation.")
    delta_charge = sum(p.formal_charge - state.atoms[index - 1].charge for index, p in parameters.items())
    counterions = []
    if config.ti.charge_compensation_mode == TIChargeCompensationMode.CO_ALCHEMICAL_COUNTERIONS:
        if abs(state.net_charge) > 0.05:
            raise ValueError(f"Charge-compensated metal transformation requires a neutral source topology (net {state.net_charge:+g}).")
        count = int(round(abs(delta_charge)))
        if abs(abs(delta_charge) - count) > 0.05:
            raise ValueError("Metal transformation charge difference must be integral for counterion compensation.")
        if count:
            counterions = _transformation_counterions(
                state=state, source_pdb=source_pdb, selected_indices=atom_indices,
                sign=1 if delta_charge > 0 else -1, count=count, config=config.ti)
    duplicate_text, pairs = _append_qoff_duplicate_atoms_to_prmtop(
        source_text=original_text, alchemical_atom_indices=sorted([*atom_indices, *counterions]))
    version, sections = _parse_prmtop_document(duplicate_text)
    atom_pairs = [(p["original_atom_index"], p["duplicate_atom_index"]) for p in pairs["qoff_atom_pairs"]]
    duplicate_by_original = dict(atom_pairs)
    types = _tokens_to_ints(sections["ATOM_TYPE_INDEX"])
    labels = _section_values(sections["AMBER_ATOM_TYPE"])
    q = _tokens_to_floats(sections["CHARGE"])
    names = _section_values(sections["ATOM_NAME"])
    masses = _tokens_to_floats(sections["MASS"])
    numbers = _tokens_to_ints(sections["ATOMIC_NUMBER"]) if "ATOMIC_NUMBER" in sections else None
    residues = _section_values(sections["RESIDUE_LABEL"])
    residue_pointers = _tokens_to_ints(sections["RESIDUE_POINTER"])
    endpoint_types, shared_types = {}, {}
    for original, duplicate in atom_pairs:
        if original not in parameters:
            # Charge-only co-alchemical ion: retains its existing LJ/C4 type.
            types[duplicate - 1] = source_types[original - 1]
            labels[duplicate - 1] = source_labels[original - 1]
            q[duplicate - 1] = 0.0
            continue
        p = parameters[original]
        key = (p.element, p.formal_charge, p.parameter_set, p.water_model)
        endpoint_type = shared_types.setdefault(key, types[duplicate - 1])
        types[duplicate - 1] = endpoint_type
        endpoint_types[endpoint_type] = p
        labels[duplicate - 1] = p.label
        q[duplicate - 1] = p.formal_charge * 18.2223
        names[duplicate - 1] = p.element.upper()
        # Both physical endpoint masses must be present for gti_syn_mass=0.
        # Amber interpolates the common-atom mass at each lambda window.
        masses[duplicate - 1] = gemmi.Element(p.element).weight
        if numbers is not None:
            numbers[duplicate - 1] = gemmi.Element(p.element).atomic_number
        residues[residue_pointers.index(duplicate)] = p.element.upper()[:3]
    ntypes = _tokens_to_ints(sections["POINTERS"])[1]
    index = _tokens_to_ints(sections["NONBONDED_PARM_INDEX"])
    coeffs = {kind: _tokens_to_floats(sections[f"LENNARD_JONES_{kind}COEF"]) for kind in "ABC"}
    lj = _derive_self_lj_parameters(_pair_coefficients(index, coeffs["A"], ntypes=ntypes),
                                   _pair_coefficients(index, coeffs["B"], ntypes=ntypes), ntypes=ntypes)
    source_lj = dict(lj)
    pol_path = config.transformation.polarizability_file or opc_duvail_polarizability_file()
    polar = _polarizabilities(pol_path)
    pol_by_type = {}
    used_types = set(types)
    for type_id in used_types:
        type_labels = {label for label, number in zip(labels, types) if number == type_id}
        missing = type_labels - polar.keys()
        if missing:
            raise ValueError(f"Missing polarizability for atom types {sorted(missing)}; set transformation.polarizability_file.")
        values = {polar[label] for label in type_labels}
        if len(values) != 1:
            raise ValueError(f"Conflicting polarizabilities for shared LJ type {type_id}: {type_labels}")
        pol_by_type[type_id] = values.pop()
    water_types = {number for number, label in zip(source_types, source_labels) if label == "OW"}
    original_ntypes = _tokens_to_ints(source["POINTERS"])[1]
    original_index = _tokens_to_ints(source["NONBONDED_PARM_INDEX"])
    original_c = _tokens_to_floats(source["LENNARD_JONES_CCOEF"])
    central_c4 = {}
    ion_types = {source_types[a.atom_index - 1] for a in state.atoms if a.residue_atom_count == 1 and abs(a.charge) > 0.1}
    for type_id in ion_types:
        if water_types:
            water_values = {original_c[original_index[(type_id - 1) * original_ntypes + water - 1] - 1] for water in water_types}
            if len(water_values) != 1:
                raise ValueError("Ambiguous water C4 parameters in source topology.")
            central_c4[type_id] = water_values.pop()
        else:
            self_c4 = original_c[original_index[(type_id - 1) * original_ntypes + type_id - 1] - 1]
            pol = pol_by_type[type_id]
            if pol == 0 and abs(self_c4) > 1e-8:
                raise ValueError("Cannot infer the existing ion C4 contribution from zero polarizability.")
            central_c4[type_id] = self_c4 * 1.444 / pol / config.transformation.tuning_factor if pol else 0.0
    for type_id, p in endpoint_types.items():
        lj[type_id] = LJParameters(p.label, p.rmin_half, p.epsilon)
        central_c4[type_id] = p.c4
    for left in range(1, ntypes + 1):
        for right in range(left, ntypes + 1):
            if left not in endpoint_types and right not in endpoint_types:
                continue
            offset = index[(left - 1) * ntypes + right - 1] - 1
            if left not in used_types or right not in used_types:
                a, b, c = 0.0, 0.0, 0.0
            else:
                a, b = _combine_lj_pair(lj[left], lj[right])
                c = central_c4.get(left, 0.0) * pol_by_type[right]
                if left != right:
                    c += central_c4.get(right, 0.0) * pol_by_type[left]
                c *= config.transformation.tuning_factor / 1.444
                if right in water_types:
                    c = central_c4[left]
                elif left in water_types:
                    c = central_c4[right]
            coeffs["A"][offset], coeffs["B"][offset], coeffs["C"][offset] = a, b, c
    for name, values in (("ATOM_TYPE_INDEX", types), ("AMBER_ATOM_TYPE", labels), ("CHARGE", q),
                         ("ATOM_NAME", names), ("MASS", masses), ("RESIDUE_LABEL", residues)):
        sections[name].data_lines = _render_section_values(sections[name].format_line, values)
    if numbers is not None:
        sections["ATOMIC_NUMBER"].data_lines = _render_section_values(sections["ATOMIC_NUMBER"].format_line, numbers)
    for kind, values in coeffs.items():
        name = f"LENNARD_JONES_{kind}COEF"
        sections[name].data_lines = _render_section_values(sections[name].format_line, values)
    target = output_dir / "metal_transform.prmtop"
    target.write_text(_render_prmtop(version, sections), encoding="utf-8")
    endpoint_charge = state.net_charge + delta_charge - sum(state.atoms[i - 1].charge for i in counterions)
    if temperature_k is None:
        from amber_metallo.ti.analysis import parse_cntrl_settings
        temperature_k = parse_cntrl_settings(config.complex_input.production_mdin_path).temperature_k
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Metal TI temperature must be finite and positive.")
    # Exact classical momentum integral for these unbonded, unconstrained ions.
    # Keep this separate from the configurational DV/DL integral. It cancels
    # between matching bound and water transformations at the same temperature.
    kinetic_mass_term = -1.5 * (8.31446261815324 / 4184) * temperature_k * math.fsum(
        math.log(masses[duplicate_by_original[index] - 1] / source_masses[index - 1]) for index in atom_indices)
    metadata = {
        "mode": "metal", "path": "direct_matched_atoms", "ti_prmtop": str(target),
        "timask1": _mask([a for a, _ in atom_pairs]), "timask2": _mask([b for _, b in atom_pairs]),
        "atom_pairs": atom_pairs, "counterion_indices": counterions,
        "counterion_mask": _mask([i for pair in atom_pairs if pair[0] in counterions for i in pair]) if counterions else None,
        "initial_charge": state.net_charge, "endpoint_charge": endpoint_charge,
        "delta_metal_charge": delta_charge,
        "mass_mode": config.transformation.mass_mode.value,
        "gti_syn_mass": 0 if config.transformation.mass_mode == TIMassMode.LINEAR else 1,
        "temperature_k": temperature_k,
        "classical_kinetic_mass_term_kcal_mol": kinetic_mass_term,
        "mass_term_included_in_reported_dg": False,
        "polarizability_file": str(pol_path), "tuning_factor": config.transformation.tuning_factor,
        "sites": [{"site": site.site, "source_element": site.element,
                   "source_charge": state.atoms[site.atom_index - 1].charge,
                   "source_mass_da": source_masses[site.atom_index - 1],
                   "endpoint_mass_da": masses[duplicate_by_original[site.atom_index] - 1],
                   "source_parameters": {
                       "rmin_half": source_lj[source_types[site.atom_index - 1]].rmin_half,
                       "epsilon": source_lj[source_types[site.atom_index - 1]].epsilon,
                       "c4": central_c4[source_types[site.atom_index - 1]],
                       "self_c4": original_c[original_index[
                           (source_types[site.atom_index - 1] - 1) * original_ntypes
                           + source_types[site.atom_index - 1] - 1] - 1],
                   },
                   "source_atom_index": site.atom_index, "endpoint_atom_index": duplicate_by_original[site.atom_index],
                   "endpoint": asdict(parameters[site.atom_index])} for site in selected_sites],
        "dry_run": dry_run,
    }
    write_json(output_dir / "metal_transform.json", metadata)
    return metadata


def _transformation_counterions(*, state, source_pdb, selected_indices, sign, count, config):
    from amber_metallo.ti.analysis import _iter_indexed_atoms
    from amber_metallo.inspection import load_structure
    structure = load_structure(source_pdb)
    atoms = _iter_indexed_atoms(structure)
    solute = [a for a in atoms if a.classification not in {"water", "ion"} or a.atom_index in selected_indices]
    ion_indices = {a.atom_index for a in state.monovalent_atoms(sign=sign)} - set(selected_indices)
    solute = [a for a in solute if a.atom_index not in ion_indices]
    def distance(a, b):
        return (structure.cell.find_nearest_pbc_image(a.position, b.position, 0).dist()
                if structure.cell.is_crystal() else a.position.dist(b.position))
    ranked = sorted((min((distance(a, b) for b in solute), default=math.inf), a.atom_index)
                    for a in atoms if a.atom_index in ion_indices)
    selected = []
    for minimum, index in reversed(ranked):
        if minimum < config.counterion_min_solute_distance_angstrom:
            continue
        if any(distance(atoms[index - 1], atoms[previous - 1]) < config.counterion_min_separation_angstrom for previous in selected):
            continue
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError(f"Direct transformation needs {count} distant monovalent ions with charge {sign:+d}; "
                     "add compatible salt to the source system or explicitly choose no charge compensation.")


def paired_restraints(source_file, atom_pairs, output_path):
    """Put identical restraints in V0 and V1; Amber linearly weights each state."""
    mapping = dict(atom_pairs)
    text = Path(source_file).read_text(encoding="utf-8")
    chunks = re.findall(r"&rst\b.*?/", text, flags=re.DOTALL | re.IGNORECASE)
    output = []
    for chunk in chunks:
        output.append(chunk)
        def replace_iat(match):
            values = [int(v) for v in re.findall(r"-?\d+", match[1])]
            if not any(v in mapping for v in values):
                return match[0]
            return "iat = " + ", ".join(str(mapping.get(v, v)) for v in values) + ","
        mapped = re.sub(r"iat\s*=\s*((?:-?\d+\s*,\s*)+)", replace_iat, chunk, flags=re.IGNORECASE)
        if mapped != chunk:
            output.append(mapped)
    Path(output_path).write_text("\n".join(output) + "\n", encoding="utf-8")
    return str(output_path)


def mass_path_description(plan):
    """Describe the recorded run, including manifests from before mass interpolation."""
    return {
        0: "lambda-dependent mass: m(lambda) = (1-lambda)*m_source + lambda*m_end (gti_syn_mass=0)",
        1: "constant source mass (gti_syn_mass=1)",
        2: "constant end mass (gti_syn_mass=2)",
    }.get(plan.get("gti_syn_mass"), "mass treatment not recorded")


def finalize_direct_inputs(output_dir, windows, *, transformation):
    mass_flag = transformation["gti_syn_mass"]
    if mass_flag not in (0, 1):
        raise ValueError("Direct metal TI supports linear or source mass synchronization.")
    for window in windows:
        for filename in (window.filename, window.equil_filename, window.restart_equil_filename):
            if filename is None:
                continue
            path = output_dir / filename
            text = path.read_text(encoding="utf-8").replace("  icfe = 1,", f"  icfe = 1,\n  gti_syn_mass = {mass_flag},")
            text = re.sub(r"(  nt[cf] = )\d+,", r"\g<1>1,", text)
            path.write_text(text, encoding="utf-8")
            if filename == window.filename:
                window.content = text
