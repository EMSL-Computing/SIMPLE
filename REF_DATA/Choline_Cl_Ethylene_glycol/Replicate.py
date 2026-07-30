#!/usr/bin/env python3
"""
PDB Molecule Replication Script

This script reads a PDB file and replicates the molecule in 3D space
according to user-specified dimensions in x, y, and z directions.
"""

import os
import sys
from typing import List, Tuple, Dict


class PDBAtom:
    """Class to represent a single atom from a PDB file"""
    
    def __init__(self, line: str):
        """Parse a PDB ATOM/HETATM line"""
        self.record_type = line[0:6].strip()
        self.atom_number = int(line[6:11].strip())
        self.atom_name = line[12:16].strip()
        self.alt_loc = line[16:17].strip()
        self.residue_name = line[17:20].strip()
        self.chain_id = line[21:22].strip()
        self.residue_number = int(line[22:26].strip()) if line[22:26].strip() else 0
        self.insertion_code = line[26:27].strip()
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())
        self.occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.00
        self.temp_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.00
        self.element = line[76:78].strip() if len(line) > 76 else ""
        self.charge = line[78:80].strip() if len(line) > 78 else ""
    
    def to_pdb_line(self, atom_number: int) -> str:
        """Convert atom back to PDB format line"""
        return (f"{self.record_type:<6}{atom_number:>5} "
                f"{self.atom_name:>4}{self.alt_loc:<1}"
                f"{self.residue_name:>3} {self.chain_id:<1}"
                f"{self.residue_number:>4}{self.insertion_code:<1}   "
                f"{self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}"
                f"{self.occupancy:>6.2f}{self.temp_factor:>6.2f}          "
                f"{self.element:>2}{self.charge:<2}")


class PDBReplicator:
    """Class to handle PDB file reading and molecule replication"""
    
    def __init__(self):
        self.atoms: List[PDBAtom] = []
        self.header_lines: List[str] = []
        self.footer_lines: List[str] = []
        self.box_dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def read_pdb_file(self, filename: str) -> bool:
        """Read and parse a PDB file"""
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            self.atoms = []
            self.header_lines = []
            self.footer_lines = []
            
            for line in lines:
                line = line.rstrip('\n')
                
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        atom = PDBAtom(line)
                        self.atoms.append(atom)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line}")
                        print(f"Error: {e}")
                        continue
                elif line.startswith(('COMPND', 'AUTHOR', 'HEADER', 'TITLE', 'REMARK')):
                    self.header_lines.append(line)
                elif line.startswith(('END', 'CONECT')):
                    self.footer_lines.append(line)
            
            if not self.atoms:
                print(f"Error: No atoms found in {filename}")
                return False
            
            print(f"Successfully read {len(self.atoms)} atoms from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            return False
    
    def calculate_box_dimensions(self) -> Tuple[float, float, float]:
        """Calculate the bounding box dimensions of the molecule"""
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        
        min_x = min(atom.x for atom in self.atoms)
        max_x = max(atom.x for atom in self.atoms)
        min_y = min(atom.y for atom in self.atoms)
        max_y = max(atom.y for atom in self.atoms)
        min_z = min(atom.z for atom in self.atoms)
        max_z = max(atom.z for atom in self.atoms)
        
        box_x = max_x - min_x
        box_y = max_y - min_y
        box_z = max_z - min_z
        
        self.box_dimensions = (box_x, box_y, box_z)
        
        print(f"Molecule dimensions:")
        print(f"  X: {min_x:.3f} to {max_x:.3f} (width: {box_x:.3f} Å)")
        print(f"  Y: {min_y:.3f} to {max_y:.3f} (width: {box_y:.3f} Å)")
        print(f"  Z: {min_z:.3f} to {max_z:.3f} (width: {box_z:.3f} Å)")
        
        return self.box_dimensions
    
    def replicate_molecule(self, nx: int, ny: int, nz: int, spacing_factor: float = 1.2) -> List[PDBAtom]:
        """
        Replicate the molecule in 3D space
        
        Args:
            nx, ny, nz: Number of replications in each dimension
            spacing_factor: Factor to multiply box dimensions for spacing between molecules
        
        Returns:
            List of all atoms in the replicated system
        """
        if not self.atoms:
            print("Error: No atoms to replicate")
            return []
        
        box_x, box_y, box_z = self.calculate_box_dimensions()
        
        # Calculate spacing between molecules
        spacing_x = box_x * spacing_factor
        spacing_y = box_y * spacing_factor
        spacing_z = box_z * spacing_factor
        
        replicated_atoms = []
        atom_counter = 1
        
        print(f"\nReplicating molecule {nx}x{ny}x{nz} times...")
        print(f"Spacing between molecules: {spacing_x:.3f} x {spacing_y:.3f} x {spacing_z:.3f} Å")
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate translation for this replica
                    translate_x = i * spacing_x
                    translate_y = j * spacing_y
                    translate_z = k * spacing_z
                    
                    # Create translated copy of each atom
                    for atom in self.atoms:
                        new_atom = PDBAtom(atom.to_pdb_line(atom.atom_number))
                        new_atom.x += translate_x
                        new_atom.y += translate_y
                        new_atom.z += translate_z
                        new_atom.atom_number = atom_counter
                        
                        replicated_atoms.append(new_atom)
                        atom_counter += 1
        
        total_atoms = len(replicated_atoms)
        total_molecules = nx * ny * nz
        
        print(f"Created {total_molecules} molecules with {total_atoms} total atoms")
        
        return replicated_atoms
    
    def write_replicated_pdb(self, atoms: List[PDBAtom], output_filename: str, nx: int, ny: int, nz: int):
        """Write the replicated system to a new PDB file with TER records between molecules"""
        try:
            with open(output_filename, 'w') as file:
                # Write header
                file.write("HEADER    REPLICATED MOLECULE SYSTEM\n")
                file.write("AUTHOR    GENERATED BY PDB REPLICATOR\n")
                
                # Write original header lines
                for line in self.header_lines:
                    file.write(line + '\n')
                
                # Write atoms with TER records between molecules
                atoms_per_molecule = len(self.atoms)
                total_molecules = nx * ny * nz
                
                for mol_idx in range(total_molecules):
                    start_idx = mol_idx * atoms_per_molecule
                    end_idx = start_idx + atoms_per_molecule
                    
                    # Write atoms for this molecule
                    for atom in atoms[start_idx:end_idx]:
                        file.write(atom.to_pdb_line(atom.atom_number) + '\n')
                    
                    # Add TER record after each molecule (except the last one)
                    if mol_idx < total_molecules - 1:
                        last_atom = atoms[end_idx - 1]
                        file.write(f"TER   {last_atom.atom_number + 1:>5}      {last_atom.residue_name:>3} {last_atom.chain_id:<1}{last_atom.residue_number:>4}\n")
                
                # Write footer
                for line in self.footer_lines:
                    file.write(line + '\n')
                
                file.write("END\n")
            
            print(f"Successfully wrote replicated system to '{output_filename}'")
            
        except Exception as e:
            print(f"Error writing output file '{output_filename}': {e}")


def get_user_input():
    """Get input from user for PDB file and replication parameters"""
    
    # Get PDB filename
    while True:
        pdb_file = input("\nEnter the PDB filename to read: ").strip()
        if not pdb_file:
            print("Please enter a filename.")
            continue
        
        # Add .pdb extension if not present
        if not pdb_file.lower().endswith('.pdb'):
            pdb_file += '.pdb'
        
        if os.path.exists(pdb_file):
            break
        else:
            print(f"File '{pdb_file}' not found. Please check the filename and try again.")
    
    # Get replication parameters
    while True:
        try:
            nx = int(input("Enter number of replications in X dimension: ").strip())
            if nx <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            ny = int(input("Enter number of replications in Y dimension: ").strip())
            if ny <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            nz = int(input("Enter number of replications in Z dimension: ").strip())
            if nz <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get output filename
    base_name = os.path.splitext(pdb_file)[0]
    default_output = f"{base_name}_replicated_{nx}x{ny}x{nz}.pdb"
    
    output_file = input(f"Enter output filename (default: {default_output}): ").strip()
    if not output_file:
        output_file = default_output
    
    if not output_file.lower().endswith('.pdb'):
        output_file += '.pdb'
    
    return pdb_file, nx, ny, nz, output_file


def main():
    """Main function to run the PDB replication program"""
    
    print("=" * 60)
    print("PDB Molecule Replication Tool")
    print("=" * 60)
    print("This program reads a PDB file and replicates the molecule")
    print("in 3D space according to your specifications.")
    print("=" * 60)
    
    try:
        # Get user input
        pdb_file, nx, ny, nz, output_file = get_user_input()
        
        # Create replicator instance
        replicator = PDBReplicator()
        
        # Read PDB file
        print(f"\nReading PDB file: {pdb_file}")
        if not replicator.read_pdb_file(pdb_file):
            return 1
        
        # Replicate molecule
        replicated_atoms = replicator.replicate_molecule(nx, ny, nz)
        
        if not replicated_atoms:
            print("Error: Failed to replicate molecule")
            return 1
        
        # Write output file
        print(f"\nWriting output to: {output_file}")
        replicator.write_replicated_pdb(replicated_atoms, output_file, nx, ny, nz)
        
        print("\n" + "=" * 60)
        print("Replication completed successfully!")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())