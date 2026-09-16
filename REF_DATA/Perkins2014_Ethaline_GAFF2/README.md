# Ethaline: Perkins 2014 SI charges/LJ + GAFF2 bonded (hybrid v1)

This is a selectable, **unvalidated hybrid**, not an exact reproduction of the
published force field. The original SIMPLE Choline Chloride / Ethylene Glycol
entries remain unchanged. Choose both entries labelled `Perkins 2014 / GAFF2
hybrid` at ChCl:EG = 1:2 (interactive recommended set **R3**, GUI third set).
ChCl expands to one PC4 choline and one PL4 chloride; each EG is PE4.

## Sources and deliberate differences

- Perkins, S. L.; Painter, P.; Colina, C. M. *Experimental and Computational
  Studies of Choline Chloride-Based Deep Eutectic Solvents.* J. Chem. Eng. Data
  **2014**, 59, 3652-3662. https://doi.org/10.1021/je500520h
- Supplied SI `je500520h_si_001.pdf`, pp. 6-7, Tables S1-S2 supplies choline/EG
  atomic charges and Lennard-Jones data. It describes the bonded parameters as
  GAFF; no DES-specific bonded refit is specified there. The full article could
  not be accessed during this implementation.
- Here, standard GAFF bonded terms are deliberately replaced by GAFF2 2.2.30
  (Oct 2025). Only the needed terms are included, with explicit atom-type
  aliases and all proper-torsion terms/divisors preserved. `provenance.json`
  records the pinned AmberClassic source URL, SHA-256 and every mapping.
  This does not imply GAFF2 gives more accurate DES predictions.
- Printed choline charges sum to **+0.9002 e**. A uniform correction of
  -0.0002/21 e per atom (eight-decimal rounding, residual on unique N3) makes the
  library sum **+0.9000 e** without breaking equivalent-atom charge symmetry.
  Both printed and applied charges are recorded in the provenance file.
  Chloride is assigned **-0.9000 e** for
  electroneutrality, in its own OFF residue, not the standard Amber -1 ion.
  Charges are already scaled; do not apply a second 0.9 scaling.
- Chloride LJ is absent from the supplied SI. This hybrid explicitly substitutes
  the Amber `parm99.dat` **IM** entry: Rmin/2 = **2.47 angstrom**, epsilon =
  **0.1000 kcal/mol**, attributed there to Smith & Dang, J. Chem. Phys. 1994,
  100, 3757. This has NOT been confirmed as the Perkins chloride model.
- EG O3/O4 epsilon values **0.1700 / 0.2104 kcal/mol** are preserved literally
  from Table S2 using different types (ZE/ZO). Both have Rmin/2 = 1.721 angstrom.
  The apparent asymmetry is unresolved; it was not silently corrected.
- LJ radii in the SI are already **Rmin/2**, not sigma. No radius conversion or
  energy-unit conversion is applied. Amber mixing is additive Rmin/2 and
  geometric epsilon, with 1-4 divisors SCEE = 1.2, SCNB = 2.0.
- There is no C4 term or validated 12-6-4 extension for this hybrid. GUI and
  interactive selection disable that option; the backend rejects combinations
  that would apply C4 to these types. Metal/salt mixtures using ordinary 12-6
  are possible but have not been parameter-validated against this study.

## Included assets and use

`PC4.lib`, `PL4.lib`, `PE4.lib`, and `perkins2014_gaff2.frcmod` are self-contained
Amber OFF/frcmod assets. ZC/ZN/ZX/ZH/ZO/ZE/ZQ/ZI types prevent overrides of
standard GAFF2, water, ions, or other existing solvent parameters. There are no
planar three-coordinate centers requiring improper torsions in these molecules.
Starting coordinates/connectivity are reused from SIMPLE's existing templates,
not the authors' equilibrated configurations.

For a saved SIMPLE configuration use:

```toml
[des]
components = ["choline_chloride_perkins2014_gaff2", "ethylene_glycol_perkins2014_gaff2"]
ratios = [1, 2]
ratio_units = 100
apply_1264 = false
```

Select DES workflow and packing/MD settings as usual. Minimize and equilibrate;
compare density and RDF at the paper's temperature/composition before making
any reproduction claim. No MD property agreement is asserted by this library.
The build copies provenance into `des_inputs` and records selected models and
warnings in `des_manifest.json`.

The development helper `scripts/build_perkins2014_gaff2.py` emits JSON containing
these assets from a hash-checked upstream GAFF2 source (or `--gaff2 PATH` for an
offline copy). There are no runtime downloads. Asset hashes use UTF-8/LF text,
with trailing whitespace at EOF stripped and exactly one final LF appended.
Only factual parameter values and attribution are included; no article/PDF or
complete upstream force-field file is redistributed.
