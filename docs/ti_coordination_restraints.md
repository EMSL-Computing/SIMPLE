# Optional metal–donor restraints in FreeE

Run `python FreeE.py --interactive` and choose TI. Before selecting a snapshot,
the wizard displays Reference PDB vs Last-frame coordination counts and donor
distances. After selecting Last / Cluster / a specified time, it shows that
frame's coordination again, lets you adjust the distance criterion, then asks
whether to apply metal–donor flat-bottom restraints (default No).
The initial supported path is **AMBER 12-6-4 GTI + combined Q/VDW**. Split Q-off
uses duplicate metal atoms and is deliberately not enabled for the new pairwise
restraint scheme until its restraint weighting is validated. The question is
still shown in those modes; opting in requires explicit confirmation to switch
to the supported combined-GTI protocol. No keeps the existing method unchanged.

When enabled:

1. Review the donor-detection cutoff. FreeE defaults to **2.5 Å** (independent
   of the Metalloprotein setup cutoff). This is a geometric screening
   threshold, not a universal definition of a metal coordination bond. Changing
   it updates the displayed CN and donor candidates before the Yes/No question.
2. Compare the Reference PDB and the actual Last / Cluster / specified-time frame.
   The displayed O-CN counts **oxygen only**, with water/non-water counts split.
   Both structures are converted to topology-complete PDBs before this comparison,
   including hydrogens, solvent and virtual extra points (EP). The full-PDB table
   shows validated topology atom indices; oxygen counts do not rely on bonds.
3. Choose which structure supplies target distances and select donor topology
   indices (non-water N/O/S within the cutoff initially selected). Water is
   display-only. Other non-water heavy atoms can be added explicitly.
4. Edit the candidate list: initially it extends to cutoff + 1 Å from the metal
   and 1 Å from selected atoms (the whole selected residue for proteins).
   Enter `+1` or `+2` repeatedly for cumulative expansion, `add 123,145` /
   `remove 123` for topology atoms, or `add r4` / `remove r4` for visible atoms
   in a listed residue. `done` accepts the selection. Expansion never selects
   new atoms automatically and does not change the restraint half-width.
   Each selected atom makes an individual metal–atom distance restraint, not
   a residue-COM restraint. Residue names/numbers and atom names are listed.
5. Set the flat-bottom half-width (default ±0.5 Å) and AMBER `rk2 = rk3`
   (default 5 kcal mol⁻¹ Å⁻²). These are starting settings, not a validated
   parameterization for every metal. Review them against equilibrated MD fluctuations.

Each chosen metal–donor pair gets a separate native AMBER `&rst` record, with
`nmropt=1` and a relative `DISANG` path in generated inputs. Inside `r2..r3` the
restraint is zero; adjacent walls use `rk * deviation²` (no additional factor 1/2).
Beyond `r1/r4`, AMBER's standard linear tails apply. No PLUMED or CN collective
variable is used. These restraints preserve selected contacts, not an exact total
CN, donor exchange equilibrium, or every aspect of the ligand shape.

The preview frame is reused exactly; the original PDB is only a distance reference
if that option was selected. Restrained minimization and equilibration are
generated before TI, and the same distance restraints remain active in every
window, including the fully decoupled endpoint and reverse sampling.

## Safety and reproducibility

- Reference, Last, specified-time and Cluster structures are always normalized
  to the topology's full atom order. Snapshot PDBs are generated from the **same
  frame's complete restart**, not from an EP-omitting intermediate PDB. This
  prevents the common OPC/TIP4P PDB-versus-topology atom-count mismatch. Atom
  counts, names/order and coordinate round-trips are checked before restraint
  selection. Virtual extra points are neither oxygen donors nor selectable anchors.
- For an incomplete reference, a matching full PDB or restart can supply missing
  coordinates only when all retained reference coordinates agree within 0.002 Å.
  Same-stem restart files are checked automatically; the wizard asks for a
  coordinate source when needed. If only the EP of a supported symmetric four-site
  water is absent, it can instead be restored from that water's O/H/H coordinates
  and the topology's equilibrium geometry. Missing physical H/solvent coordinates,
  unsupported virtual sites and ambiguous atom identities are **not guessed** or
  borrowed silently from another time frame. If no valid source exists, setup
  stops with an explanation. A PDB cannot recover coordinates it never contained.
- Original files are not overwritten. Full PDBs use topology residue ordinals
  and canonical atom names, so their residue numbering can differ from the input.
  Adjacent `*.full_structure.json` files record the coordinate source and any EP
  restoration. Selected atoms are also checked against the final TI topology;
  alchemical metal masks and restraints never use unvalidated PDB serials.
- Split periodic images of non-water coordinating atoms are rejected when box
  information is available. Image the metal and coordinating molecules together.
- Each input system gets its own frame choice and coordination comparison, even
  in a batch. Separate-site cases from one input use the same previewed frame,
  but each retains only its own selected donor list. Donor numbers are not copied
  between unrelated topologies. Different references do not imply that restraint
  free energies cancel between ions.
- Full references and previews are saved under `.simple_ti_wizard/full_references/`
  and `.simple_ti_wizard/restraint_previews/`. Keep both until generating the setup
  from a saved config. Changed inputs or preview files require a fresh preview.
  Generated setups contain their own full reference and selected snapshot.
- Dry-run previews are clearly labeled PDB placeholders, not measured trajectory
  coordination. They are never reused as real simulation frames.
- Inspect `bound/restraints/coordination.json` for both geometries, selected
  distances, starting violations, parameters and reference hash; the native input
  is `bound/restraints/coordination.disang`.

## Configuration files

Add the following to an existing combined-GTI TI or FreeE TOML config:

```toml
[ti.coordination_restraint]
enabled = true
reference = "snapshot"  # or "reference_pdb"
donor_cutoff_angstrom = 2.5
half_width_angstrom = 0.5
force_constant = 5.0

# Optional. Without this table, use all detected non-water donors for each site.
# Keys are selected metal-site numbers; values are donor topology atom indices.
# Explicit selections may include non-water heavy atoms beyond the initial cutoff.
# [ti.coordination_restraint.donor_indices_by_site]
# "1" = [123, 145, 201]
```

`enabled = false`, choosing No in the wizard, or omitting the entire option
preserves historical behavior (including the historical group restraint where
previously used). Charge-compensating counterion restraints are unaffected.

## Free-energy interpretation

This feature generates restraints only. TI schedules, endpoint inclusion,
integration and analysis code are unchanged. Attachment/release free energies
are **not calculated**. The legacy single-distance radial-volume correction is
not valid for this multi-distance scheme and is not applied.

`correction_status` is `not_computed`, and the workflow manifest uses the JSON
string `"NaN"` for the uncomputed correction. This is intentionally understood as
NaN by the existing numeric analysis, preventing a falsely finite corrected
binding free energy. The restrained TI integrals themselves remain available.
Do not substitute zero, subtract average restraint energy, or omit the last TI
window to manufacture a corrected result. `RESTRAINT_WARNING.txt` accompanies
the generated setup. A later, separate correction implementation is required
for fully corrected binding free energies.

The distinction between decoupling and releasing multiple distance restraints
is discussed by [Clark et al. (2023)](https://doi.org/10.1021/acs.jctc.3c00139).
