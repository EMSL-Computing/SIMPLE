#!/bin/bash
#SBATCH --account=emsl62129
#SBATCH --time=04:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=18
#SBATCH --job-name=ptmpsi_nwchem
#SBATCH --error=ptmpsi_nwchem-%j.err
#SBATCH --output=ptmpsi_nwchem-%j.out
#SBATCH --partition=normal

cleanup()
{
rsync *.xyz $SLURM_SUBMIT_DIR/. || :
rsync *.log $SLURM_SUBMIT_DIR/. || :
rsync *.txt $SLURM_SUBMIT_DIR/. || :
rsync *.json $SLURM_SUBMIT_DIR/. || :
rsync *.grid $SLURM_SUBMIT_DIR/. || :
rsync *.qrs $SLURM_SUBMIT_DIR/. || :
rsync *.pdb $SLURM_SUBMIT_DIR/. || :
rsync *.rst $SLURM_SUBMIT_DIR/. || :
rsync *.top $SLURM_SUBMIT_DIR/. || :
rsync *.trj $SLURM_SUBMIT_DIR/. || :
rsync *.out $SLURM_SUBMIT_DIR/. || :
}
sync_file()
{
pid=$!
while :
do
    if [ `ps -ef | grep "$pid" | grep -v "grep" | wc -l` == 0 ]; then echo "exiting while";  break; fi
    sleep 30s
    rsync -a $1  $2/.
done
rsync -a $1  $2/.
}

trap cleanup SIGINT SIGTERM SIGKILL SIGSEGV SIGCONT
source /etc/profile.d/modules.sh
module purge
#module load python
#module load gcc/9.3.0
#module load openmpi

#export NWCHEM_BASIS_LIBRARY=/cluster/apps/nwchem/nwchem/src/basis/libraries/
#export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NWC_RANKS_PER_DEVICE=0
export ARMCI_OPENIB_DEVICE=mlx5_0
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export https_proxy="http://proxy.emsl.pnl.gov:3128"
export http_proxy="http://proxy.emsl.pnl.gov:3128"
export NWBIN=/big_scratch/nwchems_`id -u`.img
#export NWCHEM_IMAGE="ghcr.io/edoapra/nwchem-singularity/nwchem-720.ompi41x:latest"
#export NWCHEM_IMAGE="ghcr.io/edoapra/nwchem-singularity/nwchem-dev.mpi3.ompi5x:latest"
#export NWCHEM_IMAGE="ghcr.io/edoapra/nwchem-singularity/nwchem-mdev.ompi4.1.8.ucx1.15.0:latest"
export NWCHEM_IMAGE="ghcr.io/edoapra/nwchem-singularity/nwchem-dev.ompi41x:latest"

export APPTAINERENV_SCRATCH_DIR=/big_scratch
export APPTAINER_CACHEDIR=/${SYSTEM_NAME}/${SLURM_JOB_ACCOUNT}/cache
mkdir -p ${APPTAINER_CACHEDIR}
apptainer pull -F --name $NWBIN oras://$NWCHEM_IMAGE
srun -N $SLURM_NNODES -n $SLURM_NNODES apptainer pull -F --name $NWBIN oras://$NWCHEM_IMAGE
export APPTAINERENV_OMP_NUM_THREADS=${OMP_NUM_THREADS}
#export APPTAINERENV_NWCHEM_BASIS_LIBRARY=$NWCHEM_BASIS_LIBRARY

cd /big_scratch
cp ${SLURM_SUBMIT_DIR}/espfit.py . 
#cp ${SLURM_SUBMIT_DIR}/modseminario.py . 
cp ${SLURM_SUBMIT_DIR}/conf0.nw . 
echo "Running conf0 optimization"
srun --mpi=pmi2 -N $SLURM_NNODES -n $SLURM_NPROCS apptainer exec --bind /big_scratch,$NWCHEM_BASIS_LIBRARY $NWBIN nwchem conf0.nw > conf0.log & 
sync_file conf0.log ${SLURM_SUBMIT_DIR} 

#cp ${SLURM_SUBMIT_DIR}/conf0_hess.nw . 
#echo "Running conf0 hessian"
#srun --mpi=pmi2 -N $SLURM_NNODES -n $SLURM_NPROCS apptainer exec --bind /big_scratch,$NWCHEM_BASIS_LIBRARY $NWBIN nwchem conf0_hess.nw > conf0_hess.log & 
#sync_file conf0_hess.log ${SLURM_SUBMIT_DIR} 

# Create a Virtual Environment
if [ -d "venv" ]; then
  echo "Virtual environment already exists"
else
  python -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
python espfit.py

#python modseminario.py

deactivate


 cleanup
