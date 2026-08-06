from __future__ import annotations

from pathlib import Path

from amber_metallo.config import SlurmConfig


def _generic_header(*, job_name: str) -> list[str]:
    return [
        "#!/bin/bash",
        "#SBATCH --account=[Account]",
        "#SBATCH --time=HH:MM:SS",
        "#SBATCH --nodes=[Number]",
        "#SBATCH --ntasks-per-node=[Number]",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --error={job_name}-%j.err",
        f"#SBATCH --output={job_name}-%j.out",
        "",
        "# Fill in the SBATCH placeholders above before submission.",
    ]


def render_resp_slurm_script(
    *,
    job_root: Path,
    slurm_config: SlurmConfig,
    job_name: str,
    retry_input: str | None = None,
) -> str:
    input_dir = job_root / "inputs"
    output_dir = job_root / "output"
    runner = f"srun -n $SLURM_NTASKS {slurm_config.binary_override or 'nwchem'}"
    lines = _generic_header(job_name=job_name)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f'JOB_ROOT="{job_root.resolve()}"',
            f'INPUT_DIR="{input_dir.resolve()}"',
            f'OUTPUT_DIR="{output_dir.resolve()}"',
            f'RUNNER="{runner}"',
            "",
            'mkdir -p "$OUTPUT_DIR"',
            'cp "$INPUT_DIR/resp_job.nw" "$OUTPUT_DIR/"',
            'cp "$INPUT_DIR/resp_fit.py" "$OUTPUT_DIR/"',
            'cp "$INPUT_DIR/resp_job.xyz" "$OUTPUT_DIR/"',
            'cp "$JOB_ROOT/group_constraints.json" "$OUTPUT_DIR/"',
            'cd "$OUTPUT_DIR"',
            "",
            'echo "Running NWChem RESP job"',
            '$RUNNER resp_job.nw > resp_job.log',
            'echo "Running RESP fit helper"',
            'python resp_fit.py',
            "",
        ]
    )
    if retry_input:
        lines.insert(lines.index('cp "$INPUT_DIR/resp_fit.py" "$OUTPUT_DIR/"'), f'cp "$INPUT_DIR/{retry_input}" "$OUTPUT_DIR/"')
        run_index = lines.index('echo "Running NWChem RESP job"')
        lines[run_index : run_index + 2] = [
            'rm -f ./*.grid ./site_resp_precondition.movecs ./site_resp_charges.json ./site_resp_charges.txt',
            'echo "Running NWChem RESP job (SCF-stabilized primary input)"',
            'if ! $RUNNER resp_job.nw > resp_job.log; then',
            '  echo "Primary SCF did not converge; retrying with PBE orbital preconditioning"',
            '  rm -f ./*.grid ./site_resp_precondition.movecs',
            f'  $RUNNER {retry_input} > resp_job_retry.log',
            'fi',
        ]
    return "\n".join(lines) + "\n"


def render_tahoma_resp_script(*, job_root: Path, job_name: str, retry_input: str | None = None) -> str:
    input_dir = job_root / "inputs"
    output_dir = job_root / "output"
    run_dir = "/big_scratch/${USER}/simple_resp_${SLURM_JOB_ID}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --account=emsl62112",
        "#SBATCH --time=04:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=18",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --error={job_name}-%j.err",
        f"#SBATCH --output={job_name}-%j.out",
        "#SBATCH --partition=normal",
        "",
        "set -euo pipefail",
        "",
        f'JOB_ROOT="{job_root.resolve()}"',
        f'INPUT_DIR="{input_dir.resolve()}"',
        f'OUTPUT_DIR="{output_dir.resolve()}"',
        f'RUN_DIR="{run_dir}"',
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        'export OMP_NUM_THREADS=1',
        'export NWC_RANKS_PER_DEVICE=0',
        'export ARMCI_OPENIB_DEVICE=mlx5_0',
        'export OMPI_MCA_opal_warn_on_missing_libcuda=0',
        'export https_proxy="${https_proxy:-http://proxy.emsl.pnl.gov:3128}"',
        'export http_proxy="${http_proxy:-http://proxy.emsl.pnl.gov:3128}"',
        'export NWBIN="/big_scratch/nwchems_$(id -u).img"',
        'export NWCHEM_IMAGE="${NWCHEM_IMAGE:-ghcr.io/edoapra/nwchem-singularity/nwchem-dev.ompi41x:latest}"',
        'export APPTAINERENV_SCRATCH_DIR=/big_scratch',
        'export APPTAINER_CACHEDIR="/${SYSTEM_NAME}/${SLURM_JOB_ACCOUNT}/cache"',
        "",
        "cleanup()",
        "{",
        '  rsync -a "$RUN_DIR"/ "$OUTPUT_DIR"/. || :',
        '  rm -rf "$RUN_DIR" || :',
        "}",
        "trap cleanup EXIT SIGINT SIGTERM",
        "",
        "source /etc/profile.d/modules.sh",
        "module purge",
        "",
        'mkdir -p "$OUTPUT_DIR" "$RUN_DIR" "${APPTAINER_CACHEDIR}"',
        'apptainer pull -F --name "$NWBIN" "oras://$NWCHEM_IMAGE"',
        'srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" apptainer pull -F --name "$NWBIN" "oras://$NWCHEM_IMAGE"',
        "",
        'cp "$INPUT_DIR/resp_job.nw" "$RUN_DIR/"',
        'cp "$INPUT_DIR/resp_fit.py" "$RUN_DIR/"',
        'cp "$INPUT_DIR/resp_job.xyz" "$RUN_DIR/"',
        'cp "$JOB_ROOT/group_constraints.json" "$RUN_DIR/"',
        'cd "$RUN_DIR"',
        "",
        'echo "Running Tahoma RESP job"',
        'srun --mpi=pmi2 -N "$SLURM_NNODES" -n "$SLURM_NPROCS" apptainer exec --bind /big_scratch "$NWBIN" nwchem resp_job.nw > resp_job.log',
        'echo "Running RESP fit helper"',
        '"$PYTHON_BIN" resp_fit.py',
        "",
    ]
    if retry_input:
        lines.insert(lines.index('cp "$INPUT_DIR/resp_fit.py" "$RUN_DIR/"'), f'cp "$INPUT_DIR/{retry_input}" "$RUN_DIR/"')
        run_index = lines.index('echo "Running Tahoma RESP job"')
        lines[run_index : run_index + 2] = [
            'rm -f ./*.grid ./site_resp_precondition.movecs ./site_resp_charges.json ./site_resp_charges.txt',
            'echo "Running Tahoma RESP job (SCF-stabilized primary input)"',
            'if ! srun --mpi=pmi2 -N "$SLURM_NNODES" -n "$SLURM_NPROCS" apptainer exec --bind /big_scratch "$NWBIN" nwchem resp_job.nw > resp_job.log; then',
            '  echo "Primary SCF did not converge; retrying with PBE orbital preconditioning"',
            '  rm -f ./*.grid ./site_resp_precondition.movecs',
            f'  srun --mpi=pmi2 -N "$SLURM_NNODES" -n "$SLURM_NPROCS" apptainer exec --bind /big_scratch "$NWBIN" nwchem {retry_input} > resp_job_retry.log',
            'fi',
        ]
    return "\n".join(lines) + "\n"
