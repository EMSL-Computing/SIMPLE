# Direct metal transformations in FreeE

Run `FreeE.py --interactive` and choose TI. After metal detection:

1. Select a single site, all sites together, each site separately, or a subset.
2. Choose the TI endpoint. **Dummy remains the default.** Choose **Metal -> another metal** for direct transformation.
3. For multiple workflows, choose one end metal for all cases or choose for each case. Within a case, selected sites can also have individual endpoints.
4. Select an available 12-6-4 end metal and, when both are available, its parameter family. **Fe3+ is the default candidate.**
5. Choose the starting frame, coordination restraints, sampling, charge compensation, and optional water reference.

## Parameters

The menu contains only metals with both LJ and C4 data for the selected MD water/parameter model. It reads the bundled OPC/Duvail files and installed Amber/ParmEd Li/Merz tables. The bundled Fe3+ values are Li/Merz values included alongside the Duvail lanthanide parameters. An ion with only a C4 entry (for example Ho3+ in the bundled files) is not offered.

The solvent model is retained. A SPC/E parameter family is not substituted into an OPC system. Raw inputs require the user to identify the water model used to parameterize the existing MD system, including DES inputs parameterized with that family.

Source interactions come from the input topology. Endpoint LJ parameters come from the selected frcmod, and endpoint C4 terms use the matching C4 coefficient and atom-type polarizabilities. Set `transformation.polarizability_file` and `transformation.tuning_factor` when the source used a custom polarizability model or tuning factor. Unknown atom types fail with an actionable error. Parameter paths and numerical values are recorded in `bound/metal_transform.json`.

Fe3+ is a selection default, not a prescribed coordination number. The new metal can reorganize its coordination shell during sampling. Enabled donor-distance restraints restrict that reorganization; review the selected contacts for the intended comparison.

## Direct TI and interpretation

The implementation follows Amber's matched-atom linear TI setup: lambda 0 is the original metal and lambda 1 is the selected end metal. The topology includes paired source/end atoms with synchronized coordinates, distinct `timask1`/`timask2`, and `ifsc=0`. Both endpoints retain a repulsive core. There is no intervening dummy state or subtraction of two decoupling calculations.

New setups default to `transformation.mass_mode = "linear"`, which writes `gti_syn_mass=0`. The source mass comes from the input topology and the end mass is the selected element's standard atomic weight. Amber interpolates the common-atom mass as `m(lambda) = (1-lambda)*m_source + lambda*m_end`: Eu -> Fe therefore reaches the Fe mass at lambda 1. Both masses and the mode are recorded in the manifests. Forward and reverse sweeps use the same mass path, with equilibration at each window. The scripts use GPU `pmemd.cuda` for TI; restart expansion happens after source preparation. See [Amber 2025 manual, section 26.1, especially pp. 520 and 527](https://ambermd.org/doc12/Amber25.pdf).

The reported TI integral remains the **configurational** free-energy change `G(end potential) - G(source potential)`. Mass affects dynamics and sampling times, but not the equilibrium coordinate distribution for this classical model. The physical endpoint momentum contribution for the unbonded, unconstrained metal ions is `-(3/2) R T sum(log(m_end/m_source))`. It is recorded and printed separately, and is not added to the reported configurational dG. For a standalone classical endpoint free-energy difference including momenta, add that term once to the configurational dG; it is not a restraint-release correction. In a bound-minus-water comparison with matching masses and temperature, it cancels. The water reference copies the exact bound source masses, including any isotope choice. See the [classical kinetic contribution in the GROMACS reference manual](https://manual.gromacs.org/current/reference-manual/algorithms/free-energy-calculations.html).

Earlier generated inputs used `gti_syn_mass=1` (constant source mass). Existing inputs and completed results retain that treatment; changing the code does not change an already generated or completed simulation. Their configurational free energies remain meaningful, subject to the usual sampling/convergence requirements. Analysis reads the recorded flag and does not relabel old results as mass-interpolated. To reproduce the earlier sampling protocol when regenerating inputs, explicitly set `transformation.mass_mode = "source"`. To sample the new mass path, generate new inputs and run them in a separate output directory.

For equal-charge transformations such as Eu3+ -> Fe3+, no co-alchemical ion is needed. For different charges, compensation discharges the required number of distant monovalent ions by the **charge difference**, retaining their LJ/C4 interactions. The source must be neutral and contain sufficient suitable ions. If it does not, add compatible salt to the source system or explicitly select no compensation. No-compensation runs record the net endpoint charge change.

Optional flat-bottom donor restraints are represented identically in the two TI states. Source preparation uses the original restraint file; TI uses the paired file. The raw result describes these restrained endpoint ensembles. A restraint-release correction is not calculated and is marked as unavailable.

The optional water leg applies the same source-to-end transformation, with matching source parameters. It is stored under this calculation's `water_reference/`, separate from dummy-reference caches. Analysis checks endpoint identities and refuses to pair a direct transformation with a dummy reference or another endpoint. Direct results are not added to the historical dummy water-reference library. The relative binding change is `dG_bound(source -> end) - dG_water(source -> end)` when the restraint treatment permits that interpretation.

## Configuration example

Add these sections to a FreeE config with the usual input, source-metal, and snapshot settings:

```toml
[transformation]
mode = "metal"
mass_mode = "linear"

[transformation.endpoint]
element = "Fe"
formal_charge = 3
parameter_set = "duvail"
water_model = "opc"

[ti]
implementation_mode = "amber_12_6_4_gti"
decoupling_mode = "combined_q_vdw"
charge_compensation_mode = "co_alchemical_counterions"

[water_reference]
enabled = false
water_model = "opc"
```

For different endpoints at selected sites, use `[transformation.endpoints_by_site."1"]`, `[transformation.endpoints_by_site."2"]`, etc., with the same four endpoint fields. Explicit site entries override a shared endpoint. Separate-site expansion preserves each site's endpoint and polarizability settings.

## Validation limits

Tests exercise parameter discovery, source-parameter preservation, endpoint coefficients, charge differences, paired restraints, site/batch selection, configuration round trips, analysis pairing, and GPU script/restart generation. Production Amber GPU trajectories have not been run as part of these software tests. Validate short lambda 0/1 runs and convergence on the execution host before production interpretation. A dry run can generate the bound transformation from an existing topology; water topology generation is recorded as deferred until tleap is actually executed.
