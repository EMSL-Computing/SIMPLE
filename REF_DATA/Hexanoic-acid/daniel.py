from ptmpsi.protein import Protein
from ptmpsi.nwchem import get_qm_data
import os
#
nodes = 16
ntasks = 18
hours = 4
machine = 'tahoma'
account = 'emsl62129'
long_partition='normal'
submit_command = 'sbatch'
qm_path = os.getcwd()
test = Protein(filename="HA1_h.pdb", delhet=False)
kwargs = { 'charge': '-1.' , 'aobasis' : 'def2-tzvp' }
get_qm_data(test.chains[0].residues,ligand=True, path=qm_path, partition=long_partition, machine=machine, account=account, time=hours,  ntasks=ntasks, nthreads=2, nnodes=nodes, **kwargs)
