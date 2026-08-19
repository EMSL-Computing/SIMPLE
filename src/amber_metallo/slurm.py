from __future__ import annotations

from pathlib import Path

from amber_metallo.config import SlurmConfig, SlurmProfile
from amber_metallo.md_protocols import MDStage
from amber_metallo.tool_config import ToolConfig, amber_sbatch_setup


def _runner(config: SlurmConfig) -> str:
    if config.profile == SlurmProfile.GPU:
        if config.gpus > 1:
            binary = config.binary_override or "pmemd.cuda.MPI"
            return f"srun -n {config.gpus} {binary}"
        return config.binary_override or "pmemd.cuda"
    binary = config.binary_override or "pmemd.MPI"
    return f"srun -n {config.ntasks} {binary}"


def _amber_binary_kind(config: SlurmConfig) -> str:
    if config.profile == SlurmProfile.GPU:
        return "gpu_mpi" if config.gpus > 1 else "gpu"
    return "mpi"


def _placeholder_runner(config: SlurmConfig, amber_binaries: dict[str, str] | None = None) -> str:
    amber_binaries = amber_binaries or {}
    if config.profile == SlurmProfile.GPU:
        if config.gpus > 1:
            binary = config.binary_override or amber_binaries.get("gpu_mpi") or "pmemd.cuda.MPI"
            return f"srun -n ${{SLURM_GPUS_ON_NODE:-{config.gpus}}} {binary}"
        return config.binary_override or amber_binaries.get("gpu") or "pmemd.cuda"
    binary = config.binary_override or amber_binaries.get("mpi") or "pmemd.MPI"
    return f"srun -n ${{SLURM_NTASKS:?Submit this script with sbatch so Slurm sets SLURM_NTASKS.}} {binary}"


def _stage_requires_reference(stage: MDStage) -> bool:
    return "restraintmask" in stage.content


def _stage_execution_lines(*, stages: list[MDStage]) -> list[str]:
    lines = [
        "set -euo pipefail",
        "",
        "PRMTOP=\"../02_system/system.prmtop\"",
        "COORD=\"../02_system/system.inpcrd\"",
        "mkdir -p outputs",
        "",
    ]
    for stage in stages:
        stem = Path(stage.filename).stem
        lines.append(f"INPUT=\"inputs/{stage.filename}\"")
        lines.append(f"OUT=\"outputs/{stem}.out\"")
        lines.append(f"RST=\"outputs/{stem}.rst7\"")
        command = "$RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$PRMTOP\" -c \"$COORD\" -r \"$RST\""
        if _stage_requires_reference(stage):
            command += " -ref \"$COORD\""
        if stage.writes_trajectory:
            lines.append(f"TRAJ=\"outputs/{stem}.nc\"")
            lines.append(f"{command} -x \"$TRAJ\"")
        else:
            lines.append(command)
        lines.append('if [[ ! -s "$RST" ]]; then echo "ERROR: Amber did not write restart $RST" >&2; exit 20; fi')
        if stage.stage_type == "md":
            lines.extend(
                [
                    'if grep -Eiq "(^|[^A-Za-z])(NaN|Inf)([^A-Za-z]|$)|SHAKE cannot|vlimit exceeded|Coordinate resetting" "$OUT" || grep -Eq "^( Etot| EPtot| VDWAALS).*\\*{4,}" "$OUT"; then',
                    '  echo "ERROR: non-finite/overflow MD energy detected in $OUT" >&2',
                    '  tail -n 80 "$OUT" >&2',
                    "  exit 21",
                    "fi",
                ]
            )
        lines.append("COORD=\"$RST\"")
        lines.append("")
    return lines


def render_slurm_script(
    *,
    stages: list[MDStage],
    slurm_config: SlurmConfig,
    tool_config: ToolConfig | None = None,
) -> str:
    amber_setup, amber_binaries = amber_sbatch_setup(
        tool_config,
        required_kinds=[_amber_binary_kind(slurm_config)],
    )
    runner = _placeholder_runner(slurm_config, amber_binaries)
    header = [
        "#!/bin/bash",
        "#SBATCH --account=[Account]",
        "#SBATCH --time=HH:MM:SS",
    ]
    if slurm_config.profile == SlurmProfile.CPU:
        header.extend(
            [
                "#SBATCH --nodes=[Number]",
                "#SBATCH --ntasks-per-node=[Number]",
            ]
        )
    else:
        header.extend(
            [
                "#SBATCH --nodes=[Number]",
                "#SBATCH --gres=gpu:[Number]",
            ]
        )
    header.extend(
        [
            "#SBATCH --job-name=[JobName]",
            "#SBATCH --error=[JobName]-%j.err",
            "#SBATCH --output=[JobName]-%j.out",
        ]
    )
    lines = header + [
        "",
        "# Fill in the SBATCH placeholders above before submission.",
        "# The interactive wizard stores resource defaults in TOML, but this script is left editable on purpose.",
        "",
        *amber_setup,
        "",
        f'RUNNER="{runner}"',
        "",
    ]
    lines.extend(_stage_execution_lines(stages=stages))
    return "\n".join(lines) + "\n"


def render_tahoma_script(*, stages: list[MDStage], slurm_config: SlurmConfig) -> str:
    runner = _placeholder_runner(slurm_config)
    job_name = slurm_config.job_name.strip() or "simple"
    account = slurm_config.account or "emsl62113"
    walltime = slurm_config.walltime if slurm_config.walltime != "24:00:00" else "48:00:00"
    if slurm_config.profile == SlurmProfile.GPU:
        nodes = slurm_config.nodes or 1
        gpu_count = slurm_config.gpus if slurm_config.gpus not in (0, 1) else 2
        partition = slurm_config.partition or "analysis"
        header = [
            "#!/bin/bash",
            f"#SBATCH --account={account}",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --gres=gpu:{gpu_count}",
            f"#SBATCH -p {partition}",
            f"#SBATCH --job-name={job_name}",
            "#SBATCH --error=simple-%j.err",
            "#SBATCH --output=simple-%j.out",
            "",
            "source /etc/profile.d/modules.sh",
            "source /tahoma/emsl62112/meji656/pmemd26/amber.sh",
            "",
            "module load openmpi/4.1.4",
            "export UCX_LOG_LEVEL=TRACE",
            "export UCX_TLS=rc,cuda",
            "export CUDA_HOME=/cluster/apps/amber22/amber22_src/cuda118",
            "export PATH=${CUDA_HOME}/bin:${PATH}",
            "export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH",
            "",
            'RUNNER="/tahoma/emsl62112/meji656/pmemd26/bin/pmemd.cuda"',
            "",
            "ulimit -c unlimited",
            "",
        ]
        lines = header + _stage_execution_lines(stages=stages)
        return "\n".join(lines) + "\n"

    nodes = slurm_config.nodes or 4
    header = [
        "#!/bin/bash",
        "",
        f"#SBATCH --account {account}                   # charged account",
        f"#SBATCH --time  {walltime}                      # walltime",
        f"#SBATCH --nodes {nodes}                             # number of nodes",
        "#SBATCH --ntasks-per-node 32                  # MPI ranks per node",
        f"#SBATCH --job-name {job_name}                       # job name in queue (``squeue``)",
        f"#SBATCH --error {job_name}-%j.err            # stderr file with job_name-job_id.err",
        f"#SBATCH --output {job_name}-%j.out           # stdout file",
        "",
        "",
        "source /etc/profile.d/modules.sh",
        "",
        "module purge",
        "module load amber22",
        "module load gcc",
        "module load openmpi",
        "",
        f'RUNNER="{runner}"',
        "",
    ]
    lines = header + _stage_execution_lines(stages=stages)
    return "\n".join(lines) + "\n"


def write_slurm_script(*, stages: list[MDStage], slurm_config: SlurmConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = "15step" if len(stages) == 15 else "4step"
    path = output_dir / f"run_{protocol}_{slurm_config.profile.value}.sbatch"
    path.write_text(render_slurm_script(stages=stages, slurm_config=slurm_config), encoding="utf-8")
    (output_dir / "tahoma.sbatch").write_text(
        render_tahoma_script(stages=stages, slurm_config=slurm_config),
        encoding="utf-8",
    )
    (output_dir / "submit_tahoma.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nsbatch tahoma.sbatch\n",
        encoding="utf-8",
    )
    return path
