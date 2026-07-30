#!/usr/bin/env python3
import numpy as np
import copy
import math

def norm2(vec):
    return math.sqrt( vec[0]**2 + vec[1]**2 + vec[2]**2 )

# Parameters
natoms = 10
names  = ["O", "C", "C", "O", "H", "H", "H", "H", "H", "H"]
nconf  = 1
files  = ['conf0']
ncons  = 0
charge = 0.

# Constrain hydrogens bonded to the same atom to have same charge
hcons = [[0, 3], [1, 2], [4, 9], [5, 6], [5, 7], [5, 8]]
hncons = len(hcons)

# Constrain NME, ACE, and amide bond charges (for AMBER99)
cons = [

]

grids = []
geometries = np.zeros((nconf,natoms,3))
npoints = np.zeros(nconf, dtype=int)
A = np.zeros((natoms+ncons+hncons+1,natoms+ncons+hncons+1))
B = np.zeros(natoms+ncons+hncons+1)

for i,file in enumerate(files):
    filename = file + ".xyz"
    with open(filename,"r") as fh:
        fh.readline(); fh.readline()
        for atom in range(natoms):
            line = fh.readline().split()
            geometries[i,atom] = [float(x)/0.529177 for x in line[1:4]]
    filename = file + ".grid"
    with open(filename,"r") as fh:
        npoints[i] = int(fh.readline().split()[0])
        grids.append(np.zeros((npoints[i],4)))
        for point in range(npoints[i]):
            line = fh.readline().split()
            grids[i][point] = [float(x) for x in line]

for iconf in range(nconf):
    dists = np.zeros((natoms,npoints[iconf]))
    for iatom in range(natoms):
        for k in range(npoints[iconf]):
            dists[iatom,k] = 1.0/norm2(geometries[iconf,iatom]-grids[iconf][k,:3])
        B[iatom] += np.dot(grids[iconf][:,3],dists[iatom])
    for iatom in range(natoms):
        for jatom in range(iatom,natoms):
            A[iatom,jatom] += np.dot(dists[iatom],dists[jatom])

# Symmetrize matrix
for iatom in range(natoms):
    for jatom in range(iatom,natoms):
        A[jatom,iatom] = copy.copy(A[iatom,jatom])

# Total charge constraint
A[:natoms,natoms] = 1.0
A[natoms,:natoms] = 1.0
B[natoms] = charge

# NME, ACE, and amide bond constraints
for icons in range(ncons):
    A[cons[icons][0]-1,natoms+icons+1] = 1.0
    A[natoms+icons+1,cons[icons][0]-1] = 1.0
    B[natoms+icons+1] = cons[icons][1]

# Hydrogen bond constraints
for icons in range(hncons):
    A[hcons[icons][0],natoms+ncons+icons+1] = 1.0
    A[hcons[icons][1],natoms+ncons+icons+1] = -1.0
    A[natoms+ncons+icons+1,hcons[icons][0]] = 1.0
    A[natoms+ncons+icons+1,hcons[icons][1]] = -1.0

# Start from solution without restraints
qold, _, _, _ = np.linalg.lstsq(A,B,rcond=None)

# Hyperbolic restraints are non-linear. Do 50 iterations at most
for iter in range(50):
    Acur = copy.deepcopy(A)

    # Add restraint contribution to matrix
    for i in range(natoms):
        
        # Hydrogens are free of restraints
        if names[i][0] == "H": continue

        # Skip charges already constrained
        skip = False
        for j in range(ncons):
            if i == cons[j][0]-1: 
                skip = True
                break
        if skip: continue
        Acur[i,i] += 0.005 / np.sqrt(qold[i]**2 + 0.01)

    # Solve linear equation system
    q, _, _, _ = np.linalg.lstsq(Acur,B,rcond=None)

    # Check convergence
    delta = np.amax(np.abs(q-qold))
    print("iter {}, delta: {}".format(iter,delta))
    if delta < 0.000001: break

    # Copy current solution to qold
    qold = copy.deepcopy(q)

# Print charges to STDOUT
print("")
print(" RESP charges")
print(f"  1 O   : {q[0]: 10.5f}")
print(f"  2 C   : {q[1]: 10.5f}")
print(f"  3 C   : {q[2]: 10.5f}")
print(f"  4 O   : {q[3]: 10.5f}")
print(f"  5 H   : {q[4]: 10.5f}")
print(f"  6 H   : {q[5]: 10.5f}")
print(f"  7 H   : {q[6]: 10.5f}")
print(f"  8 H   : {q[7]: 10.5f}")
print(f"  9 H   : {q[8]: 10.5f}")
print(f" 10 H   : {q[9]: 10.5f}")

print("")
