from __future__ import annotations

import math
import re
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from amber_metallo.execution import run_command
from amber_metallo.reporting import write_json
from amber_metallo.ti.analysis import (
    _cpptraj_binary,
    copy_structure,
    write_placeholder_restart,
)


@contextmanager
def open_netcdf(path: str | Path):
    """Read metadata without loading an entire coordinate trajectory into RAM."""
    target = Path(path)
    with target.open("rb") as handle:
        magic = handle.read(4)
    if magic in (b"CDF\x01", b"CDF\x02"):
        from scipy.io import netcdf_file

        with netcdf_file(str(target), mmap=True, maskandscale=True) as dataset:
            yield dataset
    else:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise ValueError(
                "This NetCDF format requires netCDF4. Install netCDF4 or convert the file "
                "to classic Amber NetCDF with cpptraj."
            ) from exc
        with Dataset(str(target)) as dataset:
            yield dataset


def netcdf_restart_has_velocities(path: Path) -> bool:
    import numpy as np

    with open_netcdf(path) as dataset:
        if (
            "coordinates" not in dataset.variables
            or "velocities" not in dataset.variables
        ):
            return False
        if (
            dataset.variables["coordinates"].shape
            != dataset.variables["velocities"].shape
        ):
            return False
        # Copy before closing the memory-mapped file. A declared but unwritten
        # velocity variable (masked/fill values) is not a usable restart either.
        velocities = np.ma.array(dataset.variables["velocities"][:], copy=True)
        return bool(
            velocities.size > 0
            and not np.ma.getmaskarray(velocities).any()
            and np.isfinite(velocities).all()
        )


@dataclass(frozen=True, slots=True)
class TrajectoryTimes:
    times_ns: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.times_ns or not all(math.isfinite(time) for time in self.times_ns):
            raise ValueError("The trajectory has no valid saved frame times.")
        if any(right <= left for left, right in zip(self.times_ns, self.times_ns[1:])):
            raise ValueError(
                "Trajectory times must increase strictly. The file may contain repeated or reset times; "
                "select a continuous trajectory segment before choosing a snapshot by time."
            )

    @property
    def start_ns(self) -> float:
        return self.times_ns[0]

    @property
    def end_ns(self) -> float:
        return self.times_ns[-1]

    def nearest_frame(self, time_ns: float) -> tuple[int, float]:
        tolerance = 1e-9 * max(1.0, abs(self.start_ns), abs(self.end_ns))
        if (
            not math.isfinite(time_ns)
            or not self.start_ns - tolerance <= time_ns <= self.end_ns + tolerance
        ):
            raise ValueError(
                f"Choose a time between {self.start_ns:g} and {self.end_ns:g} ns."
            )
        insertion = bisect_left(self.times_ns, time_ns)
        candidates = {max(0, insertion - 1), min(insertion, len(self.times_ns) - 1)}
        # Prefer the earlier saved frame when distances are equal.
        index = min(
            candidates, key=lambda item: (abs(self.times_ns[item] - time_ns), item)
        )
        return index + 1, self.times_ns[index]  # cpptraj frame numbers start at 1.


def _inferred_times(
    n_frames: int, production_mdin_path: str | Path | None
) -> TrajectoryTimes:
    if production_mdin_path is None:
        raise ValueError(
            "This trajectory has no timestamps. Select its production mdin with explicit dt and ntwx "
            "to use time relative to the start of that MD stage."
        )
    text = "\n".join(
        line.split("!")[0].split("#")[0]
        for line in Path(production_mdin_path).read_text(encoding="utf-8").splitlines()
    )
    cntrl = re.search(r"&cntrl\b(.*?)(?:/|&end)", text, re.IGNORECASE | re.DOTALL)
    values = dict(
        re.findall(
            r"\b(dt|ntwx)\s*=\s*([+\-\d.eEdD]+)", cntrl[1].lower() if cntrl else ""
        )
    )
    if "dt" not in values or "ntwx" not in values:
        raise ValueError(
            "Time selection needs explicit dt and ntwx in the production mdin when timestamps are absent."
        )
    dt = float(values["dt"].replace("d", "e"))
    ntwx = float(values["ntwx"].replace("d", "e"))
    if (
        not math.isfinite(dt)
        or not math.isfinite(ntwx)
        or dt <= 0
        or ntwx <= 0
        or not ntwx.is_integer()
    ):
        raise ValueError(
            "Production dt and ntwx must be positive, with an integer ntwx."
        )
    interval_ns = dt * ntwx / 1000.0
    return TrajectoryTimes(
        tuple((index + 1) * interval_ns for index in range(n_frames)),
        "Estimated from dt × ntwx; time is relative to this MD stage's start, with the first frame at dt × ntwx. "
        "Use only for the original, unstrided trajectory from that mdin.",
    )


def read_trajectory_times(
    *,
    trajectory_path: str | Path,
    prmtop_path: str | Path,
    production_mdin_path: str | Path | None = None,
) -> TrajectoryTimes:
    target = Path(trajectory_path).expanduser().resolve()
    with target.open("rb") as handle:
        magic = handle.read(4)
    if magic.startswith((b"CDF", b"\x89HDF")):
        import numpy as np

        with open_netcdf(target) as dataset:
            if "coordinates" not in dataset.variables:
                raise ValueError("The NetCDF file has no coordinates variable.")
            dimensions = dataset.variables["coordinates"].dimensions
            if not dimensions or dimensions[0] != "frame":
                raise ValueError(
                    "Select a trajectory with a frame dimension, rather than a single restart."
                )
            n_frames = dataset.variables["coordinates"].shape[0]
            if "time" in dataset.variables:
                units = getattr(dataset.variables["time"], "units", b"picosecond")
                units = (
                    units.decode("ascii") if isinstance(units, bytes) else str(units)
                )
                factors = {
                    "ps": 0.001,
                    "picosecond": 0.001,
                    "picoseconds": 0.001,
                    "ns": 1.0,
                    "nanosecond": 1.0,
                    "nanoseconds": 1.0,
                    "fs": 0.000001,
                    "femtosecond": 0.000001,
                    "femtoseconds": 0.000001,
                }
                if units.strip().lower() not in factors:
                    raise ValueError(f"Unsupported trajectory time units: {units!r}")
                times = np.ma.array(
                    dataset.variables["time"][:], dtype=float, copy=True
                )
                if times.shape != (n_frames,) or np.ma.getmaskarray(times).any():
                    raise ValueError(
                        "The trajectory does not contain one valid timestamp per saved frame."
                    )
                return TrajectoryTimes(
                    tuple(
                        float(value) * factors[units.strip().lower()] for value in times
                    ),
                    "Simulation time stored in the trajectory (not time since its first saved frame).",
                )
        return _inferred_times(n_frames, production_mdin_path)

    import MDAnalysis as mda

    ascii_amber = (
        target.suffix.lower() in {".mdcrd", ".crd"} or target.name.lower() == "mdcrd"
    )
    if not ascii_amber and target.suffix.lower() not in {".dcd", ".xtc", ".trr"}:
        raise ValueError(
            "Time selection supports Amber NetCDF/mdcrd, DCD, XTC, and TRR trajectories."
        )
    universe = mda.Universe(
        str(prmtop_path), str(target), **({"format": "TRJ"} if ascii_amber else {})
    )
    try:
        if ascii_amber:
            return _inferred_times(len(universe.trajectory), production_mdin_path)
        return TrajectoryTimes(
            tuple(float(frame.time) / 1000.0 for frame in universe.trajectory),
            "Simulation time read from the trajectory by MDAnalysis.",
        )
    finally:
        universe.trajectory.close()


def run_time_snapshot_extraction(
    *,
    prmtop_path: str | Path,
    trajectory_path: str | Path,
    reference_structure_path: str | Path,
    output_dir: str | Path,
    time_ns: float,
    production_mdin_path: str | Path | None = None,
    dry_run: bool,
) -> dict[str, object]:
    timeline = read_trajectory_times(
        trajectory_path=trajectory_path,
        prmtop_path=prmtop_path,
        production_mdin_path=production_mdin_path,
    )
    frame_index, actual_time = timeline.nearest_frame(time_ns)
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = target_dir / "time_snapshot.pdb"
    output_rst7 = target_dir / "time_snapshot.rst7"
    script_path = target_dir / "extract_time_snapshot.cpptraj.in"
    script_path.write_text(
        f'parm "{Path(prmtop_path).expanduser().resolve().as_posix()}"\n'
        f'trajin "{Path(trajectory_path).expanduser().resolve().as_posix()}" {frame_index} {frame_index}\n'
        "autoimage\n"
        f'trajout "{output_pdb.as_posix()}" pdb nobox\n'
        f'trajout "{output_rst7.as_posix()}" restart novelocity\n'
        "run\n",
        encoding="utf-8",
    )
    if dry_run:
        copy_structure(reference_structure_path, output_pdb)
        write_placeholder_restart(output_rst7)
    else:
        binary = _cpptraj_binary()
        if binary is None:
            raise RuntimeError(
                "cpptraj was not found, so the selected frame could not be extracted."
            )
        run_command(
            [binary, "-i", str(script_path)],
            cwd=target_dir,
            log_path=target_dir / "extract_time_snapshot.log",
        )
        if not output_pdb.is_file() or not output_rst7.is_file():
            raise RuntimeError(
                "cpptraj did not produce the requested time snapshot PDB and restart."
            )
    manifest = {
        "script": str(script_path),
        "snapshot_pdb": str(output_pdb),
        "snapshot_rst7": str(output_rst7),
        "requested_time_ns": time_ns,
        "selected_time_ns": actual_time,
        "frame_index": frame_index,
        "frame_count": len(timeline.times_ns),
        "trajectory_start_ns": timeline.start_ns,
        "trajectory_end_ns": timeline.end_ns,
        "time_source": timeline.source,
    }
    write_json(target_dir / "time_snapshot_manifest.json", manifest)
    return manifest
