# Manual Ligand Input Contract

SIMPLE supports manual parameter input for custom residues and small molecules when you do not want to run `antechamber` and `parmchk2`.

## Accepted file bundles

- `mol2 + frcmod`
- `prepi + frcmod`
- `off + frcmod`
- `lib + frcmod`

## Required information

Your manual files must encode:

- Amber atom types for every atom
- Partial charges for every atom
- Bond connectivity
- Residue and atom names that are consistent with the cleaned PDB loaded into `tleap`

## Notes

- `mol2` is expected to include atom types and charges.
- `prepi` or `off/lib` defines the residue template used by `tleap`.
- `frcmod` must contain any missing bonded or Lennard-Jones parameters not already covered by the selected force field.
- If you provide only a partial bundle, SIMPLE will stop before `tleap` execution and print the missing requirements.
