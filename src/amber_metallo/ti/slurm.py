from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amber_metallo.config import SlurmConfig, SlurmProfile
from amber_metallo.ti.protocols import PreparationStage, TIWindow


@dataclass(slots=True)
class QoffCoordinateBridge:
    original_atom_index: int | None = None
    duplicate_atom_index: int | None = None
    atom_pairs: list[tuple[int, int]] | None = None

    def resolved_pairs(self) -> list[tuple[int, int]]:
        if self.atom_pairs:
            return [(int(original), int(duplicate)) for original, duplicate in self.atom_pairs]
        if self.original_atom_index is None or self.duplicate_atom_index is None:
            return []
        return [(int(self.original_atom_index), int(self.duplicate_atom_index))]


def _runner(config: SlurmConfig) -> str:
    if config.profile == SlurmProfile.GPU:
        if config.gpus > 1:
            binary = config.binary_override or "pmemd.cuda.MPI"
            return f"srun -n {config.gpus} {binary}"
        return config.binary_override or "pmemd.cuda"
    binary = config.binary_override or "pmemd.MPI"
    return f"srun -n {config.ntasks} {binary}"


def _placeholder_runner(config: SlurmConfig) -> str:
    if config.profile == SlurmProfile.GPU:
        if config.gpus > 1:
            binary = config.binary_override or "pmemd.cuda.MPI"
            return f"srun -n ${{SLURM_GPUS_ON_NODE:-{config.gpus}}} {binary}"
        return config.binary_override or "pmemd.cuda"
    binary = config.binary_override or "pmemd.MPI"
    return f"srun -n ${{SLURM_NTASKS:?Submit this script with sbatch so Slurm sets SLURM_NTASKS.}} {binary}"


def _placeholder_prep_runner(config: SlurmConfig) -> str:
    if config.profile == SlurmProfile.GPU:
        return "srun -n ${SLURM_NTASKS:-1} pmemd.MPI"
    return "$RUNNER"


def _placeholder_qoff_runner(config: SlurmConfig, *, disjoint_dual_topology: bool) -> str:
    if config.profile == SlurmProfile.GPU and disjoint_dual_topology:
        return "srun -n ${SLURM_NTASKS:-1} pmemd.MPI"
    return "$RUNNER"


def _endpoint_prep_output_stem(stage: PreparationStage) -> str:
    return f"endpoint_{Path(stage.filename).stem}"


def _prep_label(leg_name: str) -> str:
    if leg_name == "bound":
        return "bound starting-structure pre-equilibration"
    if leg_name == "water_ref":
        return "water-reference pre-equilibration"
    return "pre-equilibration"


def _workflow_description(*, prep_label: str, prep_stages: list[PreparationStage] | None, windows: list[TIWindow]) -> str:
    has_vdwoff = any(window.phase == "vdwoff" for window in windows)
    if has_vdwoff:
        ti_text = "the charge-off windows, then the qoff-endpoint pre-equilibration, then the VDW-off windows"
    else:
        ti_text = "the softcore decoupling windows"
    if prep_stages:
        return f"# This master script performs the {prep_label} stages first, then {ti_text}."
    return f"# This master script performs {ti_text}."


_RESTART_BRIDGE_PYTHON = r"""import struct
import sys
from pathlib import Path


def _parse_values(lines):
    values = []
    for line in lines:
        if not line.strip():
            continue
        tokens = line.split()
        try:
            values.extend(_parse_float_token(token) for token in tokens)
            continue
        except ValueError:
            pass
        try:
            values.extend(_parse_fixed_width_float_line(line, width=12))
        except ValueError as exc:
            raise SystemExit(
                "Could not parse a formatted Amber restart numeric line as whitespace-separated "
                f"or F12.7 fixed-width values: {line!r}"
            ) from exc
    return values


def _parse_float_token(token):
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_fixed_width_float_line(line, *, width):
    values = []
    for start in range(0, len(line), width):
        token = line[start : start + width].strip()
        if token:
            values.append(_parse_float_token(token))
    return values


_NC_DIMENSION = 10
_NC_VARIABLE = 11
_NC_ATTRIBUTE = 12
_NC_TYPE_SIZES = {
    1: 1,  # byte
    2: 1,  # char
    3: 2,  # short
    4: 4,  # int
    5: 4,  # float
    6: 8,  # double
}


def _align4(value):
    return value + ((4 - value % 4) % 4)


def _read_i32(raw, offset):
    return struct.unpack_from(">i", raw, offset)[0], offset + 4


def _read_i64(raw, offset):
    return struct.unpack_from(">q", raw, offset)[0], offset + 8


def _read_name(raw, offset):
    length, offset = _read_i32(raw, offset)
    name = raw[offset : offset + length].decode("ascii", errors="replace")
    return name, offset + _align4(length)


def _skip_nc_values(raw, offset, nc_type, count):
    if nc_type not in _NC_TYPE_SIZES:
        raise SystemExit(f"Unsupported NetCDF value type {nc_type} in restart header")
    return offset + _align4(_NC_TYPE_SIZES[nc_type] * count)


def _skip_attribute_list(raw, offset):
    tag, offset = _read_i32(raw, offset)
    if tag == 0:
        _empty, offset = _read_i32(raw, offset)
        return offset
    if tag != _NC_ATTRIBUTE:
        raise SystemExit("Unsupported NetCDF restart header: malformed attribute list")
    count, offset = _read_i32(raw, offset)
    for _ in range(count):
        _name, offset = _read_name(raw, offset)
        nc_type, offset = _read_i32(raw, offset)
        value_count, offset = _read_i32(raw, offset)
        offset = _skip_nc_values(raw, offset, nc_type, value_count)
    return offset


def _product(values):
    total = 1
    for value in values:
        total *= value
    return total


def _netcdf_variable_values(raw, variables, dimensions, numrecs, name, *, record_size):
    variable = variables.get(name)
    if variable is None:
        return []
    nc_type = variable["type"]
    if nc_type not in (5, 6):
        raise SystemExit(f"NetCDF restart variable '{name}' is not float/double")
    type_size = _NC_TYPE_SIZES[nc_type]
    dim_lengths = [dimensions[dim_id]["length"] for dim_id in variable["dimids"]]
    is_record = bool(dim_lengths and dim_lengths[0] == 0)
    if is_record:
        frame_count = max(numrecs, 1)
        values_per_frame = _product(dim_lengths[1:] or [1])
        begin = variable["begin"] + record_size * (frame_count - 1)
        value_count = values_per_frame
    else:
        value_count = _product([max(length, numrecs) if length == 0 else length for length in dim_lengths] or [1])
        begin = variable["begin"]
    payload = raw[begin : begin + value_count * type_size]
    if len(payload) < value_count * type_size:
        raise SystemExit(f"NetCDF restart variable '{name}' is shorter than expected")
    code = "f" if nc_type == 5 else "d"
    return list(struct.unpack(f">{value_count}{code}", payload))


def _read_netcdf_restart(path, raw):
    if len(raw) < 4 or not raw.startswith(b"CDF"):
        raise SystemExit(f"Not a classic NetCDF Amber restart: {path}")
    version = raw[3]
    if version not in (1, 2):
        raise SystemExit(
            "Disjoint Q-off restart bridge can read classic NetCDF restarts only. "
            "Convert NetCDF-4/HDF5 restarts to formatted rst7 with cpptraj first."
        )
    offset = 4
    numrecs, offset = _read_i32(raw, offset)

    tag, offset = _read_i32(raw, offset)
    dimensions = []
    if tag == 0:
        _empty, offset = _read_i32(raw, offset)
    elif tag == _NC_DIMENSION:
        count, offset = _read_i32(raw, offset)
        for dim_id in range(count):
            name, offset = _read_name(raw, offset)
            length, offset = _read_i32(raw, offset)
            dimensions.append({"id": dim_id, "name": name, "length": length})
    else:
        raise SystemExit("Unsupported NetCDF restart header: malformed dimension list")

    offset = _skip_attribute_list(raw, offset)

    tag, offset = _read_i32(raw, offset)
    variables = {}
    variable_list = []
    if tag == 0:
        _empty, offset = _read_i32(raw, offset)
    elif tag == _NC_VARIABLE:
        count, offset = _read_i32(raw, offset)
        for _ in range(count):
            name, offset = _read_name(raw, offset)
            dim_count, offset = _read_i32(raw, offset)
            dimids = []
            for _dim in range(dim_count):
                dim_id, offset = _read_i32(raw, offset)
                dimids.append(dim_id)
            offset = _skip_attribute_list(raw, offset)
            nc_type, offset = _read_i32(raw, offset)
            vsize, offset = _read_i32(raw, offset)
            begin, offset = _read_i32(raw, offset) if version == 1 else _read_i64(raw, offset)
            variable = {
                "name": name,
                "dimids": dimids,
                "type": nc_type,
                "vsize": vsize,
                "begin": begin,
            }
            variables[name] = variable
            variable_list.append(variable)
    else:
        raise SystemExit("Unsupported NetCDF restart header: malformed variable list")

    record_size = sum(_align4(variable["vsize"]) for variable in variable_list if variable["dimids"] and dimensions[variable["dimids"][0]]["length"] == 0)
    coords = _netcdf_variable_values(raw, variables, dimensions, numrecs, "coordinates", record_size=record_size)
    if not coords:
        raise SystemExit(f"NetCDF restart is missing coordinates: {path}")
    atom_dimension = next((dimension["length"] for dimension in dimensions if dimension["name"] == "atom"), None)
    natom = atom_dimension or (len(coords) // 3)
    coord_count = 3 * natom
    coords = coords[-coord_count:]
    velocities = _netcdf_variable_values(raw, variables, dimensions, numrecs, "velocities", record_size=record_size)
    velocities = velocities[-coord_count:] if len(velocities) >= coord_count else []
    cell_lengths = _netcdf_variable_values(raw, variables, dimensions, numrecs, "cell_lengths", record_size=record_size)
    cell_angles = _netcdf_variable_values(raw, variables, dimensions, numrecs, "cell_angles", record_size=record_size)
    time_values = _netcdf_variable_values(raw, variables, dimensions, numrecs, "time", record_size=record_size)
    header_rest = [f"{time_values[-1]:.7f}"] if time_values else []
    trailing = []
    if len(cell_lengths) >= 3 and len(cell_angles) >= 3:
        trailing = cell_lengths[-3:] + cell_angles[-3:]
    return f"Converted NetCDF restart from {Path(path).name}", header_rest, natom, coords, velocities, trailing, trailing


def _looks_like_periodic_box(values):
    if len(values) < 6:
        return False
    a, b, c, alpha, beta, gamma = values[-6:]
    return (
        1.0 <= a <= 1000.0
        and 1.0 <= b <= 1000.0
        and 1.0 <= c <= 1000.0
        and 30.0 <= alpha <= 150.0
        and 30.0 <= beta <= 150.0
        and 30.0 <= gamma <= 150.0
    )


def _read_restart(path):
    raw = Path(path).read_bytes()
    if raw.startswith(b"CDF"):
        return _read_netcdf_restart(path, raw)
    if raw.startswith(b"\x89HDF"):
        raise SystemExit(
            "Disjoint Q-off restart bridge cannot read NetCDF-4/HDF5 restarts directly. "
            "Convert the restart to formatted rst7 with cpptraj before this step."
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SystemExit(
            "Disjoint Q-off restart bridge could not read the restart as formatted text. "
            "Use a formatted Amber rst7 restart for split Q-off."
        ) from exc
    if len(lines) < 2:
        raise SystemExit(f"Amber restart is too short: {path}")
    title = lines[0]
    header_tokens = lines[1].split()
    if not header_tokens:
        raise SystemExit(f"Amber restart header is missing atom count: {path}")
    natom = int(header_tokens[0])
    values = _parse_values(lines[2:])
    coord_count = 3 * natom
    if len(values) < coord_count:
        raise SystemExit(f"Amber restart has fewer coordinate values than expected for {natom} atoms: {path}")
    coords = values[:coord_count]
    trailing = values[coord_count:]
    velocities = []
    if len(trailing) >= coord_count:
        velocities = trailing[:coord_count]
        trailing = trailing[coord_count:]
    source_box = values[-6:] if _looks_like_periodic_box(values[-6:]) else trailing
    return title, header_tokens[1:], natom, coords, velocities, trailing, source_box


def _atom_slice(index):
    start = 3 * (index - 1)
    return slice(start, start + 3)


def _without_atom(values, index):
    target = _atom_slice(index)
    return values[: target.start] + values[target.stop :]


def _without_atoms(values, indices):
    reduced = values
    for index in sorted(indices, reverse=True):
        reduced = _without_atom(reduced, index)
    return reduced


def _format_header(natom, rest):
    if not rest:
        return f"{natom:6d}"
    rendered = []
    for token in rest:
        try:
            rendered.append(f"{float(token):15.7f}")
        except ValueError:
            rendered.append(f" {token}")
    return f"{natom:6d}" + "".join(rendered)


def _format_values(values):
    lines = []
    for start in range(0, len(values), 6):
        chunk = values[start : start + 6]
        lines.append("".join(f"{value:12.7f}" for value in chunk))
    return lines


def _write_restart(path, title, header_rest, natom, coords, velocities, trailing):
    lines = [title, _format_header(natom, header_rest)]
    lines.extend(_format_values(coords))
    if velocities:
        lines.extend(_format_values(velocities))
    if trailing:
        lines.extend(_format_values(trailing))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_pairs(raw_args):
    if len(raw_args) == 1 and (":" in raw_args[0] or "," in raw_args[0]):
        pairs = []
        for item in raw_args[0].split(","):
            if not item.strip():
                continue
            original_raw, duplicate_raw = item.split(":", 1)
            pairs.append((int(original_raw), int(duplicate_raw)))
        return pairs
    if len(raw_args) == 2:
        return [(int(raw_args[0]), int(raw_args[1]))]
    raise SystemExit("Q-off restart bridge expects either original duplicate or original:duplicate pairs")


mode, input_path, output_path, *pair_args = sys.argv[1:]
pairs = _parse_pairs(pair_args)
title, header_rest, natom, coords, velocities, trailing, source_box = _read_restart(input_path)
if mode == "expand":
    for offset, (original, duplicate) in enumerate(pairs, start=1):
        if original < 1 or original > natom:
            raise SystemExit(f"Original atom @{original} is outside restart atom range 1-{natom}")
        expected_duplicate = natom + offset
        if duplicate != expected_duplicate:
            raise SystemExit(
                f"Q-off duplicate atom @{duplicate} does not match the expanded restart position @{expected_duplicate}"
            )
    new_coords = coords + [value for original, _duplicate in pairs for value in coords[_atom_slice(original)]]
    new_velocities = (
        velocities + [value for original, _duplicate in pairs for value in velocities[_atom_slice(original)]]
        if velocities
        else []
    )
    _write_restart(output_path, title, header_rest, natom + len(pairs), new_coords, new_velocities, trailing)
elif mode in ("collapse", "collapse_strip_velocities"):
    duplicate_indices = []
    for original, duplicate in pairs:
        if original < 1 or original > natom or duplicate < 1 or duplicate > natom:
            raise SystemExit(f"Q-off bridge atom indices @{original}/@{duplicate} are outside restart atom range 1-{natom}")
        coords[_atom_slice(original)] = coords[_atom_slice(duplicate)]
        if velocities:
            velocities[_atom_slice(original)] = velocities[_atom_slice(duplicate)]
        duplicate_indices.append(duplicate)
    collapsed_velocities = [] if mode == "collapse_strip_velocities" else (
        _without_atoms(velocities, duplicate_indices) if velocities else []
    )
    collapsed_trailing = source_box if mode == "collapse_strip_velocities" else trailing
    _write_restart(
        output_path,
        title,
        header_rest,
        natom - len(pairs),
        _without_atoms(coords, duplicate_indices),
        collapsed_velocities,
        collapsed_trailing,
    )
else:
    raise SystemExit(f"Unknown Q-off restart bridge mode: {mode}")
"""


def _restart_bridge_lines(
    *,
    mode: str,
    input_coord: str,
    output_coord: str,
    bridge: QoffCoordinateBridge,
) -> list[str]:
    pairs = bridge.resolved_pairs()
    if len(pairs) == 1:
        pair_args = f"{pairs[0][0]} {pairs[0][1]}"
    else:
        pair_args = '"' + ",".join(f"{original}:{duplicate}" for original, duplicate in pairs) + '"'
    return [
        (
            f"python - \"{mode}\" \"{input_coord}\" \"{output_coord}\" "
            f"{pair_args} <<'PY'"
        ),
        _RESTART_BRIDGE_PYTHON,
        "PY",
    ]


def _window_lines(
    *,
    prep_label: str,
    input_root: str,
    runtime_output_root: str,
    prep_stages: list[PreparationStage] | None,
    prep_prmtop: str | None,
    prep_start_coord: str | None,
    endpoint_prep_stages: list[PreparationStage] | None,
    endpoint_prep_prmtop: str | None,
    windows: list[TIWindow],
    qoff_prmtop: str,
    vdw_prmtop: str,
    start_coord: str,
    qoff_coordinate_bridge: QoffCoordinateBridge | None,
) -> list[str]:
    qoff_windows = [window for window in windows if window.phase == "qoff"]
    vdwoff_windows = [window for window in windows if window.phase == "vdwoff"]
    initial_qoff_windows = qoff_windows
    terminal_qoff_window: TIWindow | None = None
    if endpoint_prep_stages and qoff_windows:
        initial_qoff_windows = qoff_windows[:-1]
        terminal_qoff_window = qoff_windows[-1]
    lines = [
        "set -euo pipefail",
        "",
        f"INPUT_ROOT=\"{input_root}\"",
        f"LEG_OUTPUT_ROOT=\"{runtime_output_root}\"",
        "mkdir -p \"$LEG_OUTPUT_ROOT\"",
        "cd \"$INPUT_ROOT\"",
        "",
    ]
    if prep_stages:
        lines.extend(
            [
                f"PREP_PRMTOP=\"{prep_prmtop}\"",
                f"PREP_START_COORD=\"{prep_start_coord}\"",
                "PREP_OUTPUT_DIR=\"$LEG_OUTPUT_ROOT/prep\"",
                "mkdir -p \"$PREP_OUTPUT_DIR\"",
                "",
                f"echo 'Starting {prep_label}...'",
                "PREP_COORD=\"$PREP_START_COORD\"",
                "",
            ]
        )
        for stage in prep_stages:
            stem = Path(stage.filename).stem
            lines.extend(
                [
                    f"INPUT=\"$INPUT_ROOT/{stage.filename}\"",
                    f"OUT=\"$PREP_OUTPUT_DIR/{stem}.out\"",
                    f"RST=\"$PREP_OUTPUT_DIR/{stem}.rst7\"",
                ]
            )
            if stage.writes_trajectory:
                lines.extend(
                    [
                        f"TRAJ=\"$PREP_OUTPUT_DIR/{stem}.nc\"",
                        "$PREP_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$PREP_PRMTOP\" -c \"$PREP_COORD\" -r \"$RST\" -x \"$TRAJ\"",
                    ]
                )
            else:
                lines.append("$PREP_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$PREP_PRMTOP\" -c \"$PREP_COORD\" -r \"$RST\"")
            lines.extend(
                [
                    "PREP_COORD=\"$RST\"",
                    "",
                ]
            )
        lines.extend(
            [
                "if [ ! -f \"$PREP_COORD\" ]; then",
                f"  echo 'Expected {prep_label} restart was not created.' >&2",
                "  exit 1",
                "fi",
                "",
            ]
        )

    qoff_label = "softcore decoupling" if qoff_windows and not vdwoff_windows else "charge-off"
    lines.extend(
        [
            f"QOFF_PRMTOP=\"{qoff_prmtop}\"",
            f"VDWOFF_PRMTOP=\"{vdw_prmtop}\"",
            f"START_COORD=\"{start_coord}\"",
            "QOFF_OUTPUT_DIR=\"$LEG_OUTPUT_ROOT/qoff\"",
            "VDWOFF_OUTPUT_DIR=\"$LEG_OUTPUT_ROOT/vdwoff\"",
            "mkdir -p \"$QOFF_OUTPUT_DIR\"",
            "mkdir -p \"$VDWOFF_OUTPUT_DIR\"",
            "",
            f"echo 'Starting {qoff_label} TI windows...'",
        ]
    )
    if prep_stages:
        lines.append("START_COORD=\"$PREP_COORD\"")
        lines.append("")
    if qoff_coordinate_bridge is not None:
        lines.extend(
            [
                "QOFF_START_COORD=\"$QOFF_OUTPUT_DIR/qoff_start_disjoint.rst7\"",
                "echo 'Expanding start restart for disjoint Q-off topology...'",
                *_restart_bridge_lines(
                    mode="expand",
                    input_coord="$START_COORD",
                    output_coord="$QOFF_START_COORD",
                    bridge=qoff_coordinate_bridge,
                ),
                "START_COORD=\"$QOFF_START_COORD\"",
                "",
            ]
        )
    last_qoff_rst = ""
    for window in initial_qoff_windows:
        stem = Path(window.filename).stem
        last_qoff_rst = "$QOFF_OUTPUT_DIR/" + f"{stem}.rst7"
        lines.extend(
            [
                f"INPUT=\"$INPUT_ROOT/{window.filename}\"",
                f"OUT=\"$QOFF_OUTPUT_DIR/{stem}.out\"",
                f"RST=\"{last_qoff_rst}\"",
                f"TRAJ=\"$QOFF_OUTPUT_DIR/{stem}.nc\"",
                "$QOFF_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$QOFF_PRMTOP\" -c \"$START_COORD\" -r \"$RST\" -x \"$TRAJ\"",
                "START_COORD=\"$RST\"",
                "",
            ]
        )
    if endpoint_prep_stages:
        if qoff_coordinate_bridge is not None:
            pre_endpoint_source = last_qoff_rst or "$START_COORD"
            qoff_prep_source = "\"$QOFF_PRE_ENDPOINT\""
            lines.extend(
                [
                    "QOFF_PRE_ENDPOINT=\"$QOFF_OUTPUT_DIR/qoff_pre_endpoint_single_topology.rst7\"",
                    "echo 'Collapsing pre-endpoint Q-off restart for decharged endpoint preparation...'",
                    *_restart_bridge_lines(
                        mode="collapse_strip_velocities",
                        input_coord=pre_endpoint_source,
                        output_coord="$QOFF_PRE_ENDPOINT",
                        bridge=qoff_coordinate_bridge,
                    ),
                    "",
                ]
            )
        else:
            qoff_prep_source = f"\"{last_qoff_rst}\"" if last_qoff_rst else "\"$START_COORD\""
        lines.extend(
            [
                f"ENDPOINT_PREP_PRMTOP=\"{endpoint_prep_prmtop}\"",
                "ENDPOINT_PREP_OUTPUT_DIR=\"$QOFF_OUTPUT_DIR\"",
                "",
                "echo 'Starting qoff-endpoint pre-equilibration before terminal charge-off and VDW-off TI...'",
                f"QOFF_PREP_SOURCE={qoff_prep_source}",
                "if [ ! -f \"$QOFF_PREP_SOURCE\" ]; then",
                "  echo 'Expected charge-off restart for endpoint preparation was not created.' >&2",
                "  exit 1",
                "fi",
                "ENDPOINT_COORD=\"$QOFF_PREP_SOURCE\"",
                "",
            ]
        )
        for stage in endpoint_prep_stages:
            stem = _endpoint_prep_output_stem(stage)
            lines.extend(
                [
                    f"INPUT=\"$INPUT_ROOT/{stage.filename}\"",
                    f"OUT=\"$ENDPOINT_PREP_OUTPUT_DIR/{stem}.out\"",
                    f"RST=\"$ENDPOINT_PREP_OUTPUT_DIR/{stem}.rst7\"",
                ]
            )
            if stage.writes_trajectory:
                lines.extend(
                    [
                        f"TRAJ=\"$ENDPOINT_PREP_OUTPUT_DIR/{stem}.nc\"",
                        "$PREP_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$ENDPOINT_PREP_PRMTOP\" -c \"$ENDPOINT_COORD\" -r \"$RST\" -x \"$TRAJ\"",
                    ]
                )
            else:
                lines.append("$PREP_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$ENDPOINT_PREP_PRMTOP\" -c \"$ENDPOINT_COORD\" -r \"$RST\"")
            lines.extend(
                [
                    "ENDPOINT_COORD=\"$RST\"",
                    "",
                ]
            )
        if terminal_qoff_window is not None:
            stem = Path(terminal_qoff_window.filename).stem
            last_qoff_rst = "$QOFF_OUTPUT_DIR/" + f"{stem}.rst7"
            terminal_start_coord = "$ENDPOINT_COORD"
            if qoff_coordinate_bridge is not None:
                terminal_start_coord = "$QOFF_TERMINAL_START"
                lines.extend(
                    [
                        "QOFF_TERMINAL_START=\"$QOFF_OUTPUT_DIR/qoff_terminal_start_disjoint.rst7\"",
                        "echo 'Expanding decharged endpoint restart for terminal disjoint Q-off window...'",
                        *_restart_bridge_lines(
                            mode="expand",
                            input_coord="$ENDPOINT_COORD",
                            output_coord="$QOFF_TERMINAL_START",
                            bridge=qoff_coordinate_bridge,
                        ),
                        "",
                    ]
                )
            lines.extend(
                [
                    "echo 'Starting terminal charge-off TI window after endpoint pre-equilibration...'",
                    f"INPUT=\"$INPUT_ROOT/{terminal_qoff_window.filename}\"",
                    f"OUT=\"$QOFF_OUTPUT_DIR/{stem}.out\"",
                    f"RST=\"{last_qoff_rst}\"",
                    f"TRAJ=\"$QOFF_OUTPUT_DIR/{stem}.nc\"",
                    f"$QOFF_RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$QOFF_PRMTOP\" -c \"{terminal_start_coord}\" -r \"$RST\" -x \"$TRAJ\"",
                    "",
                ]
            )
        if qoff_coordinate_bridge is not None:
            lines.extend(
                [
                    "QOFF_ENDPOINT=\"$QOFF_OUTPUT_DIR/qoff_endpoint_single_topology.rst7\"",
                    "echo 'Collapsing terminal Q-off restart for VDW-off topology...'",
                    *_restart_bridge_lines(
                        mode="collapse",
                        input_coord=last_qoff_rst,
                        output_coord="$QOFF_ENDPOINT",
                        bridge=qoff_coordinate_bridge,
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                (f"QOFF_ENDPOINT=\"{last_qoff_rst}\"" if last_qoff_rst else "QOFF_ENDPOINT=\"$ENDPOINT_COORD\"")
                if qoff_coordinate_bridge is None
                else ":",
                "if [ ! -f \"$QOFF_ENDPOINT\" ]; then",
                "  echo 'Expected charge-off endpoint restart was not created.' >&2",
                "  exit 1",
                "fi",
                "",
            ]
        )
    else:
        if qoff_coordinate_bridge is not None:
            lines.extend(
                [
                    "QOFF_ENDPOINT=\"$QOFF_OUTPUT_DIR/qoff_endpoint_single_topology.rst7\"",
                    "echo 'Collapsing Q-off endpoint restart for VDW-off topology...'",
                    *_restart_bridge_lines(
                        mode="collapse",
                        input_coord=last_qoff_rst,
                        output_coord="$QOFF_ENDPOINT",
                        bridge=qoff_coordinate_bridge,
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                f"QOFF_ENDPOINT=\"{last_qoff_rst}\"" if qoff_coordinate_bridge is None else ":",
                "if [ ! -f \"$QOFF_ENDPOINT\" ]; then",
                "  echo 'Expected charge-off endpoint restart was not created.' >&2",
                "  exit 1",
                "fi",
                "",
            ]
        )
    if vdwoff_windows:
        lines.append("echo 'Starting VDW-off TI windows...'")
        for window in vdwoff_windows:
            stem = Path(window.filename).stem
            lines.extend(
                [
                    f"INPUT=\"$INPUT_ROOT/{window.filename}\"",
                    f"OUT=\"$VDWOFF_OUTPUT_DIR/{stem}.out\"",
                    f"RST=\"$VDWOFF_OUTPUT_DIR/{stem}.rst7\"",
                    f"TRAJ=\"$VDWOFF_OUTPUT_DIR/{stem}.nc\"",
                    "$RUNNER -O -i \"$INPUT\" -o \"$OUT\" -p \"$VDWOFF_PRMTOP\" -c \"$QOFF_ENDPOINT\" -r \"$RST\" -x \"$TRAJ\"",
                    "",
                ]
            )
    return lines


def render_leg_slurm_script(
    *,
    leg_name: str,
    input_root: str,
    runtime_output_root: str,
    prep_stages: list[PreparationStage] | None,
    prep_prmtop: str | None,
    prep_start_coord: str | None,
    endpoint_prep_stages: list[PreparationStage] | None,
    endpoint_prep_prmtop: str | None,
    windows: list[TIWindow],
    slurm_config: SlurmConfig,
    qoff_prmtop: str,
    vdw_prmtop: str,
    start_coord: str,
    qoff_coordinate_bridge: QoffCoordinateBridge | None = None,
) -> str:
    runner = _placeholder_runner(slurm_config)
    prep_runner = _placeholder_prep_runner(slurm_config)
    qoff_runner = _placeholder_qoff_runner(slurm_config, disjoint_dual_topology=qoff_coordinate_bridge is not None)
    prep_label = _prep_label(leg_name)
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
            f"#SBATCH --job-name=[{leg_name}]",
            f"#SBATCH --error=[{leg_name}]-%j.err",
            f"#SBATCH --output=[{leg_name}]-%j.out",
            "",
            "# Fill in the SBATCH placeholders above before submission.",
            _workflow_description(prep_label=prep_label, prep_stages=prep_stages, windows=windows),
            "",
            f'RUNNER="{runner}"',
            f'PREP_RUNNER="{prep_runner}"',
            f'QOFF_RUNNER="{qoff_runner}"',
            "",
        ]
    )
    return "\n".join(header + _window_lines(
        prep_label=prep_label,
        input_root=input_root,
        runtime_output_root=runtime_output_root,
        prep_stages=prep_stages,
        prep_prmtop=prep_prmtop,
        prep_start_coord=prep_start_coord,
        endpoint_prep_stages=endpoint_prep_stages,
        endpoint_prep_prmtop=endpoint_prep_prmtop,
        windows=windows,
        qoff_prmtop=qoff_prmtop,
        vdw_prmtop=vdw_prmtop,
        start_coord=start_coord,
        qoff_coordinate_bridge=qoff_coordinate_bridge,
    )) + "\n"


def render_tahoma_leg_script(
    *,
    leg_name: str,
    input_root: str,
    runtime_output_root: str,
    prep_stages: list[PreparationStage] | None,
    prep_prmtop: str | None,
    prep_start_coord: str | None,
    endpoint_prep_stages: list[PreparationStage] | None,
    endpoint_prep_prmtop: str | None,
    windows: list[TIWindow],
    slurm_config: SlurmConfig,
    qoff_prmtop: str,
    vdw_prmtop: str,
    start_coord: str,
    qoff_coordinate_bridge: QoffCoordinateBridge | None = None,
) -> str:
    runner = _placeholder_runner(slurm_config)
    prep_runner = _placeholder_prep_runner(slurm_config)
    qoff_runner = _placeholder_qoff_runner(slurm_config, disjoint_dual_topology=qoff_coordinate_bridge is not None)
    prep_label = _prep_label(leg_name)
    account = slurm_config.account or "emsl62113"
    walltime = slurm_config.walltime if slurm_config.walltime != "24:00:00" else "48:00:00"
    if slurm_config.profile == SlurmProfile.GPU:
        nodes = slurm_config.nodes or 1
        gpu_count = slurm_config.gpus if slurm_config.gpus not in (0, 1) else 2
        partition = slurm_config.partition or "analysis"
        job_name = slurm_config.job_name if slurm_config.job_name != "simple" else "gTI"
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
            "# Fill in the SBATCH placeholders above before submission.",
            _workflow_description(prep_label=prep_label, prep_stages=prep_stages, windows=windows),
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
            'PREP_RUNNER="${RUNNER}"',
            (
                'QOFF_RUNNER="srun -n ${SLURM_NTASKS:-1} /tahoma/emsl62112/meji656/pmemd26/bin/pmemd.MPI"'
                if qoff_coordinate_bridge is not None
                else 'QOFF_RUNNER="${RUNNER}"'
            ),
            "",
            "ulimit -c unlimited",
            "",
        ]
    else:
        nodes = slurm_config.nodes or 4
        job_name = slurm_config.job_name if slurm_config.job_name != "simple" else "TI"
        header = [
            "#!/bin/bash",
            "",
            f"#SBATCH --account {account}",
            f"#SBATCH --time  {walltime}",
            f"#SBATCH --nodes {nodes}",
            "#SBATCH --ntasks-per-node 32",
            f"#SBATCH --job-name {job_name}",
            "#SBATCH --error ti-%j.err",
            "#SBATCH --output ti-%j.out",
            "",
            "source /etc/profile.d/modules.sh",
            "",
            "module purge",
            "module load amber22",
            "module load gcc",
            "module load openmpi",
            "",
            f'RUNNER="{runner}"',
            f'PREP_RUNNER="{prep_runner}"',
            f'QOFF_RUNNER="{qoff_runner}"',
            "",
        ]
    return "\n".join(header + _window_lines(
        prep_label=prep_label,
        input_root=input_root,
        runtime_output_root=runtime_output_root,
        prep_stages=prep_stages,
        prep_prmtop=prep_prmtop,
        prep_start_coord=prep_start_coord,
        endpoint_prep_stages=endpoint_prep_stages,
        endpoint_prep_prmtop=endpoint_prep_prmtop,
        windows=windows,
        qoff_prmtop=qoff_prmtop,
        vdw_prmtop=vdw_prmtop,
        start_coord=start_coord,
        qoff_coordinate_bridge=qoff_coordinate_bridge,
    )) + "\n"


def write_leg_slurm_scripts(
    *,
    leg_name: str,
    input_root: str,
    runtime_output_root: str,
    prep_stages: list[PreparationStage] | None,
    prep_prmtop: str | None,
    prep_start_coord: str | None,
    endpoint_prep_stages: list[PreparationStage] | None,
    endpoint_prep_prmtop: str | None,
    windows: list[TIWindow],
    slurm_config: SlurmConfig,
    qoff_prmtop: str,
    vdw_prmtop: str,
    start_coord: str,
    qoff_coordinate_bridge: QoffCoordinateBridge | None = None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / f"run_{leg_name}_{slurm_config.profile.value}.sbatch"
    main_path.write_text(
        render_leg_slurm_script(
            leg_name=leg_name,
            input_root=input_root,
            runtime_output_root=runtime_output_root,
            prep_stages=prep_stages,
            prep_prmtop=prep_prmtop,
            prep_start_coord=prep_start_coord,
            endpoint_prep_stages=endpoint_prep_stages,
            endpoint_prep_prmtop=endpoint_prep_prmtop,
            windows=windows,
            slurm_config=slurm_config,
            qoff_prmtop=qoff_prmtop,
            vdw_prmtop=vdw_prmtop,
            start_coord=start_coord,
            qoff_coordinate_bridge=qoff_coordinate_bridge,
        ),
        encoding="utf-8",
    )
    tahoma_path = output_dir / f"tahoma_{leg_name}.sbatch"
    tahoma_path.write_text(
        render_tahoma_leg_script(
            leg_name=leg_name,
            input_root=input_root,
            runtime_output_root=runtime_output_root,
            prep_stages=prep_stages,
            prep_prmtop=prep_prmtop,
            prep_start_coord=prep_start_coord,
            endpoint_prep_stages=endpoint_prep_stages,
            endpoint_prep_prmtop=endpoint_prep_prmtop,
            windows=windows,
            slurm_config=slurm_config,
            qoff_prmtop=qoff_prmtop,
            vdw_prmtop=vdw_prmtop,
            start_coord=start_coord,
            qoff_coordinate_bridge=qoff_coordinate_bridge,
        ),
        encoding="utf-8",
    )
    (output_dir / f"submit_{leg_name}_tahoma.sh").write_text(
        f"#!/bin/bash\nset -euo pipefail\ncd -- \"$(cd -- \"$(dirname -- \"$0\")\" && pwd)\"\nsbatch {tahoma_path.name}\n",
        encoding="utf-8",
    )
    return main_path
