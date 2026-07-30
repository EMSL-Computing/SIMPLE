#!/bin/bash

#SBATCH --account emsl62129                   # charged account
#SBATCH --time  48:00:00                      # 30 minute time limit
#SBATCH --nodes 2                             # 2 nodes
#SBATCH --ntasks-per-node 32                  # 16 processes on each per node
#SBATCH --job-name R1                       # job name in queue (``squeue``)
#SBATCH --error my_job_name-%j.err            # stderr file with job_name-job_id.err
#SBATCH --output my_job_name-%j.out           # stdout file


source /etc/profile.d/modules.sh

module purge
module load amber22
module load gcc
module load openmpi

#sander -O -i min.in -p DES.prmtop -c DES.inpcrd -r DES.rst -o log.out
#cp DES.rst ../EQ/
#cd ../EQ

#sander -O -i md.in -p DES.prmtop -c DES.rst -r DES_1.rst -o log.out
#cp DES_1.rst ../PROD
#cd ../PROD

mpirun -np 64 pmemd.MPI -O -i md.in -p DES.prmtop -c DES.inpcrd -r DES_2.rst -o log.out


