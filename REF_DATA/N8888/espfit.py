#!/usr/bin/env python3
import numpy as np
import copy
import math

def norm2(vec):
    return math.sqrt( vec[0]**2 + vec[1]**2 + vec[2]**2 )

# Parameters
natoms = 101
names  = ["C", "C", "C", "C", "C", "C", "C", "C", "N", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C"\
, "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", \
"H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H\
", "H", "H", "H", "H", "H", "H", "H", "H"] 
nconf  = 1
files  = ['conf0']
ncons  = 0
charge = +1.

# Constrain hydrogens bonded to the same atom to have same charge
hcons = [[0, 16], [0, 24], [0, 32], [1, 15], [1, 23], [1, 31], [2, 14], [2, 22], [2, 30], [3, 13], [3, 21], [3, 29], [4, 12], [4, 20], [4, 28], [5, \
11], [5, 19], [5, 27], [6, 10], [6, 18], [6, 26], [7, 9], [7, 17], [7, 25], [33, 34], [33, 35], [33, 64], [33, 65], [33, 66], [33, 81], [33, 82], [33, 83], [\
33, 98], [33, 99], [33, 100], [36, 37], [36, 62], [36, 63], [36, 79], [36, 80], [36, 96], [36, 97], [38, 39], [38, 60], [38, 61], [38, 77], [38, 78], [38, 94\
], [38, 95], [40, 41], [40, 58], [40, 59], [40, 75], [40, 76], [40, 92], [40, 93], [42, 43], [42, 56], [42, 57], [42, 73], [42, 74], [42, 90], [42, 91], [44,\
 45], [44, 54], [44, 55], [44, 71], [44, 72], [44, 88], [44, 89], [46, 47], [46, 52], [46, 53], [46, 69], [46, 70], [46, 86], [46, 87], [48, 49], [48, 50], [\
48, 51], [48, 67], [48, 68], [48, 84], [48, 85]]
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
print(f"  1 C   : {q[0]: 10.5f}")
print(f"  2 C   : {q[1]: 10.5f}")
print(f"  3 C   : {q[2]: 10.5f}")
print(f"  4 C   : {q[3]: 10.5f}")
print(f"  5 C   : {q[4]: 10.5f}")
print(f"  6 C   : {q[5]: 10.5f}")
print(f"  7 C   : {q[6]: 10.5f}")
print(f"  8 C   : {q[7]: 10.5f}")
print(f"  9 N   : {q[8]: 10.5f}")
print(f" 10 C   : {q[9]: 10.5f}")
print(f" 11 C   : {q[10]: 10.5f}")
print(f" 12 C   : {q[11]: 10.5f}")
print(f" 13 C   : {q[12]: 10.5f}")
print(f" 14 C   : {q[13]: 10.5f}")
print(f" 15 C   : {q[14]: 10.5f}")
print(f" 16 C   : {q[15]: 10.5f}")
print(f" 17 C   : {q[16]: 10.5f}")
print(f" 18 C   : {q[17]: 10.5f}")
print(f" 19 C   : {q[18]: 10.5f}")
print(f" 20 C   : {q[19]: 10.5f}")
print(f" 21 C   : {q[20]: 10.5f}")
print(f" 22 C   : {q[21]: 10.5f}")
print(f" 23 C   : {q[22]: 10.5f}")
print(f" 24 C   : {q[23]: 10.5f}")
print(f" 25 C   : {q[24]: 10.5f}")
print(f" 26 C   : {q[25]: 10.5f}")
print(f" 27 C   : {q[26]: 10.5f}")
print(f" 28 C   : {q[27]: 10.5f}")
print(f" 29 C   : {q[28]: 10.5f}")
print(f" 30 C   : {q[29]: 10.5f}")
print(f" 31 C   : {q[30]: 10.5f}")
print(f" 32 C   : {q[31]: 10.5f}")
print(f" 33 C   : {q[32]: 10.5f}")
print(f" 34 H   : {q[33]: 10.5f}")
print(f" 35 H   : {q[34]: 10.5f}")
print(f" 36 H   : {q[35]: 10.5f}")
print(f" 37 H   : {q[36]: 10.5f}")
print(f" 38 H   : {q[37]: 10.5f}")
print(f" 39 H   : {q[38]: 10.5f}")
print(f" 40 H   : {q[39]: 10.5f}")
print(f" 41 H   : {q[40]: 10.5f}")
print(f" 42 H   : {q[41]: 10.5f}")
print(f" 43 H   : {q[42]: 10.5f}")
print(f" 44 H   : {q[43]: 10.5f}")
print(f" 45 H   : {q[44]: 10.5f}")
print(f" 46 H   : {q[45]: 10.5f}")
print(f" 47 H   : {q[46]: 10.5f}")
print(f" 48 H   : {q[47]: 10.5f}")
print(f" 49 H   : {q[48]: 10.5f}")
print(f" 50 H   : {q[49]: 10.5f}")
print(f" 51 H   : {q[50]: 10.5f}")
print(f" 52 H   : {q[51]: 10.5f}")
print(f" 53 H   : {q[52]: 10.5f}")
print(f" 54 H   : {q[53]: 10.5f}")
print(f" 55 H   : {q[54]: 10.5f}")
print(f" 56 H   : {q[55]: 10.5f}")
print(f" 57 H   : {q[56]: 10.5f}")
print(f" 58 H   : {q[57]: 10.5f}")
print(f" 59 H   : {q[58]: 10.5f}")
print(f" 60 H   : {q[59]: 10.5f}")
print(f" 61 H   : {q[60]: 10.5f}")
print(f" 62 H   : {q[61]: 10.5f}")
print(f" 63 H   : {q[62]: 10.5f}")
print(f" 64 H   : {q[63]: 10.5f}")
print(f" 65 H   : {q[64]: 10.5f}")
print(f" 66 H   : {q[65]: 10.5f}")
print(f" 67 H   : {q[66]: 10.5f}")
print(f" 68 H   : {q[67]: 10.5f}")
print(f" 69 H   : {q[68]: 10.5f}")
print(f" 70 H   : {q[69]: 10.5f}")
print(f" 71 H   : {q[70]: 10.5f}")
print(f" 72 H   : {q[71]: 10.5f}")
print(f" 73 H   : {q[72]: 10.5f}")
print(f" 74 H   : {q[73]: 10.5f}")
print(f" 75 H   : {q[74]: 10.5f}")
print(f" 76 H   : {q[75]: 10.5f}")
print(f" 77 H   : {q[76]: 10.5f}")
print(f" 78 H   : {q[77]: 10.5f}")
print(f" 79 H   : {q[78]: 10.5f}")
print(f" 80 H   : {q[79]: 10.5f}")
print(f" 81 H   : {q[80]: 10.5f}")
print(f" 82 H   : {q[81]: 10.5f}")
print(f" 83 H   : {q[82]: 10.5f}")
print(f" 84 H   : {q[83]: 10.5f}")
print(f" 85 H   : {q[84]: 10.5f}")
print(f" 86 H   : {q[85]: 10.5f}")
print(f" 87 H   : {q[86]: 10.5f}")
print(f" 88 H   : {q[87]: 10.5f}")
print(f" 89 H   : {q[88]: 10.5f}")
print(f" 90 H   : {q[89]: 10.5f}")
print(f" 91 H   : {q[90]: 10.5f}")
print(f" 92 H   : {q[91]: 10.5f}")
print(f" 93 H   : {q[92]: 10.5f}")
print(f" 94 H   : {q[93]: 10.5f}")
print(f" 95 H   : {q[94]: 10.5f}")
print(f" 96 H   : {q[95]: 10.5f}")
print(f" 97 H   : {q[96]: 10.5f}")
print(f" 98 H   : {q[97]: 10.5f}")
print(f" 99 H   : {q[98]: 10.5f}")
print(f" 100 H   : {q[99]: 10.5f}")
print(f" 101 H   : {q[100]: 10.5f}")

print("")
