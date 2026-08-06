from __future__ import annotations

import json
import math
from pathlib import Path


CONSTRAINED_RESP_SOLVER_VERSION = "simple-protein-site-resp-v2"


def equality_pairs_from_group_payload(payload: dict[str, object]) -> list[tuple[int, int]]:
    groups = payload.get("groups") or []
    pairs: list[tuple[int, int]] = []
    for group in groups:
        atom_indices = [int(value) for value in (group.get("atom_indices") or [])]
        if len(atom_indices) < 2:
            continue
        anchor = atom_indices[0] - 1
        for atom_index in atom_indices[1:]:
            pairs.append((anchor, int(atom_index) - 1))
    return pairs


def equality_groups_from_pairs(atom_count: int, equality_pairs: list[tuple[int, int]]) -> list[list[int]]:
    adjacency = [set() for _ in range(atom_count)]
    for first_atom, second_atom in equality_pairs:
        if first_atom < 0 or second_atom < 0 or first_atom >= atom_count or second_atom >= atom_count:
            raise ValueError(
                "RESP equality constraints referenced an atom index outside the available atom range. "
                f"Received pair ({first_atom}, {second_atom}) for {atom_count} atoms."
            )
        adjacency[first_atom].add(second_atom)
        adjacency[second_atom].add(first_atom)

    groups: list[list[int]] = []
    visited: set[int] = set()
    for atom_index in range(atom_count):
        if atom_index in visited:
            continue
        stack = [atom_index]
        component: list[int] = []
        visited.add(atom_index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        groups.append(sorted(component))
    return groups


def load_resp_charge_result(path: str | Path) -> list[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [float(item["charge"]) for item in payload.get("charges") or []]


def fit_constrained_resp_payload(
    *,
    atom_names: list[str],
    coordinates_bohr: list[list[float]],
    grid_rows: list[list[float]],
    total_charge: float,
    equality_pairs: list[tuple[int, int]] | None = None,
    fixed_charges: dict[int, float] | None = None,
    sum_constraints: list[dict[str, object]] | None = None,
    atom_metadata: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Fit RESP charges with exact linear constraints.

    Atom indices in the constraint arguments are zero-based. Equality groups are
    collapsed before fixed-value and residue-sum constraints are applied.
    """
    import numpy as np

    atom_count = len(atom_names)
    if atom_count == 0 or len(coordinates_bohr) != atom_count:
        raise ValueError("RESP fitting requires one coordinate row per atom.")
    if not grid_rows:
        raise ValueError("RESP fitting requires at least one ESP grid point.")

    groups = equality_groups_from_pairs(atom_count, list(equality_pairs or []))
    atom_to_group: dict[int, int] = {}
    for group_index, group in enumerate(groups):
        for atom_index in group:
            atom_to_group[atom_index] = group_index

    coords = np.asarray(coordinates_bohr, dtype=float)
    grid = np.asarray(grid_rows, dtype=float)
    design = np.zeros((len(grid), len(groups)), dtype=float)
    restrained_counts = np.zeros(len(groups), dtype=float)
    for group_index, group in enumerate(groups):
        restrained_counts[group_index] = float(
            sum(0 if atom_names[atom_index].upper().startswith("H") else 1 for atom_index in group)
        )
        for atom_index in group:
            displacement = coords[atom_index][None, :] - grid[:, :3]
            distances = np.linalg.norm(displacement, axis=1)
            if np.any(distances <= 1.0e-12):
                raise ValueError("An ESP grid point overlaps an RESP atom coordinate.")
            design[:, group_index] += 1.0 / distances

    constraint_rows: list[list[float]] = []
    constraint_targets: list[float] = []
    constraint_labels: list[str] = []

    total_row = [float(len(group)) for group in groups]
    constraint_rows.append(total_row)
    constraint_targets.append(float(total_charge))
    constraint_labels.append("cluster_total")

    fixed_by_group: dict[int, float] = {}
    for atom_index, charge in dict(fixed_charges or {}).items():
        index = int(atom_index)
        if index < 0 or index >= atom_count:
            raise ValueError(f"Fixed RESP charge references atom index {index} outside 0..{atom_count - 1}.")
        group_index = atom_to_group[index]
        value = float(charge)
        previous = fixed_by_group.get(group_index)
        if previous is not None and not math.isclose(previous, value, abs_tol=1.0e-8):
            raise ValueError("An RESP equality group contains conflicting fixed charges.")
        fixed_by_group[group_index] = value
    for group_index, charge in sorted(fixed_by_group.items()):
        row = [0.0] * len(groups)
        row[group_index] = 1.0
        constraint_rows.append(row)
        constraint_targets.append(charge)
        constraint_labels.append(f"fixed_group_{group_index + 1}")

    for constraint_index, constraint in enumerate(sum_constraints or [], start=1):
        indices = [int(value) for value in constraint.get("atom_indices") or []]
        if not indices:
            raise ValueError("RESP sum constraints must contain at least one atom index.")
        row = [0.0] * len(groups)
        for atom_index in indices:
            if atom_index < 0 or atom_index >= atom_count:
                raise ValueError(
                    f"RESP sum constraint references atom index {atom_index} outside 0..{atom_count - 1}."
                )
            row[atom_to_group[atom_index]] += 1.0
        constraint_rows.append(row)
        constraint_targets.append(float(constraint.get("charge") or 0.0))
        constraint_labels.append(str(constraint.get("label") or f"sum_{constraint_index}"))

    constraint_matrix = np.asarray(constraint_rows, dtype=float)
    target_vector = np.asarray(constraint_targets, dtype=float)

    metadata = list(atom_metadata or [{} for _ in range(atom_count)])
    if len(metadata) != atom_count:
        raise ValueError("RESP atom metadata must contain one entry per atom.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(grid)):
        raise ValueError("RESP coordinates and grid values must all be finite.")

    # This hybrid model redistributes standard-FF charge; it does not replace
    # the entire site with an unconstrained molecular ESP charge model. Build
    # the nearest constraint-feasible standard-FF reference, then fit only in
    # the exact-constraint null space. This removes redundant KKT rows and
    # nearly-null charge-transfer modes that can create cancelling charges of
    # hundreds or thousands of electrons.
    baseline_groups = np.asarray(
        [
            sum(float(metadata[index].get("original_charge") or 0.0) for index in group) / len(group)
            for group in groups
        ],
        dtype=float,
    )
    feasibility_delta, _, _, _ = np.linalg.lstsq(
        constraint_matrix,
        target_vector - constraint_matrix @ baseline_groups,
        rcond=1.0e-12,
    )
    reference_groups = baseline_groups + feasibility_delta
    reference_constraint_residual = float(
        np.max(np.abs(constraint_matrix @ reference_groups - target_vector))
    )
    if reference_constraint_residual > 1.0e-8:
        raise ValueError(
            "Protein-site RESP constraints are mutually inconsistent; the standard-FF reference "
            f"cannot satisfy them (maximum residual {reference_constraint_residual:.3e})."
        )

    _u, constraint_singular_values, constraint_vt = np.linalg.svd(
        constraint_matrix,
        full_matrices=True,
    )
    largest_constraint_singular = (
        float(constraint_singular_values[0]) if len(constraint_singular_values) else 1.0
    )
    constraint_tolerance = (
        max(constraint_matrix.shape)
        * np.finfo(float).eps
        * max(largest_constraint_singular, 1.0)
        * 100.0
    )
    constraint_rank = int(np.sum(constraint_singular_values > constraint_tolerance))
    null_basis = constraint_vt[constraint_rank:, :].T

    # Use mean squared ESP error so a denser grid does not silently weaken the
    # restraint. Heavy atoms receive a baseline-centered RESP hyperbolic term.
    # A small harmonic term makes every unidentifiable direction coercive,
    # including hfree hydrogen directions, without relaxing exact constraints.
    point_count = float(len(grid))
    data_normal = (design.T @ design) / point_count
    data_rhs = (design.T @ grid[:, 3]) / point_count
    projected_design = design @ null_basis
    design_singular_values = np.linalg.svd(
        projected_design / math.sqrt(point_count),
        compute_uv=False,
    )
    if len(design_singular_values):
        design_tolerance = (
            max(projected_design.shape)
            * np.finfo(float).eps
            * float(design_singular_values[0])
        )
        nonzero_design_singulars = design_singular_values[design_singular_values > design_tolerance]
    else:
        nonzero_design_singulars = design_singular_values
    design_rank = int(len(nonzero_design_singulars))
    design_condition_number = (
        None
        if len(nonzero_design_singulars) == 0
        else float(nonzero_design_singulars[0] / nonzero_design_singulars[-1])
    )

    hyperbolic_scale = 0.005
    hyperbolic_tightness = 0.1
    harmonic_scale = 0.001
    iteration_count = 0
    converged = True
    if null_basis.shape[1] == 0:
        group_charges = reference_groups.copy()
    else:
        projected_normal = null_basis.T @ data_normal @ null_basis
        projected_rhs = null_basis.T @ (data_rhs - data_normal @ reference_groups)
        reduced = np.zeros(null_basis.shape[1], dtype=float)
        converged = False
        for iteration_count in range(1, 101):
            delta_groups = null_basis @ reduced
            restraint_weights = harmonic_scale * np.asarray(
                [float(len(group)) for group in groups],
                dtype=float,
            )
            for group_index, heavy_count in enumerate(restrained_counts):
                if heavy_count <= 0.0 or group_index in fixed_by_group:
                    continue
                restraint_weights[group_index] += (
                    heavy_count
                    * hyperbolic_scale
                    / math.sqrt(float(delta_groups[group_index]) ** 2 + hyperbolic_tightness**2)
                )
            restrained_normal = projected_normal + null_basis.T @ (
                restraint_weights[:, None] * null_basis
            )
            updated, _, _, _ = np.linalg.lstsq(restrained_normal, projected_rhs, rcond=1.0e-12)
            if float(np.max(np.abs(updated - reduced))) < 1.0e-8:
                reduced = updated
                converged = True
                break
            reduced = updated
        group_charges = reference_groups + null_basis @ reduced

    charges = [float(group_charges[atom_to_group[index]]) for index in range(atom_count)]
    predicted = design @ group_charges
    residual = predicted - grid[:, 3]
    esp_rmse = float(np.sqrt(np.mean(residual**2)))
    esp_reference_rms = float(np.sqrt(np.mean(grid[:, 3] ** 2)))
    constraint_values = constraint_matrix @ group_charges
    constraint_residuals = constraint_values - target_vector
    maximum_constraint_residual = float(np.max(np.abs(constraint_residuals)))

    charge_rows: list[dict[str, object]] = []
    for index, charge in enumerate(charges):
        row = dict(metadata[index])
        baseline = row.get("original_charge")
        row.update(
            {
                "index": index + 1,
                "name": atom_names[index],
                "charge": charge,
                "delta": None if baseline is None else charge - float(baseline),
                "fixed": index in dict(fixed_charges or {}),
            }
        )
        charge_rows.append(row)

    maximum_absolute_delta = max(abs(float(row.get("delta") or 0.0)) for row in charge_rows)
    numerically_stable = bool(
        converged
        and np.all(np.isfinite(group_charges))
        and maximum_constraint_residual <= 1.0e-7
    )
    baseline_residual = design @ reference_groups - grid[:, 3]
    return {
        "solver_version": CONSTRAINED_RESP_SOLVER_VERSION,
        "fit_method": "baseline-centered constrained RESP in the exact-constraint null space",
        "numerically_stable": numerically_stable,
        "converged": converged,
        "iteration_count": iteration_count,
        "charges": charge_rows,
        "total_charge": float(sum(charges)),
        "constraint_count": len(constraint_rows) + len(equality_pairs or []),
        "maximum_constraint_residual": maximum_constraint_residual,
        "esp_rmse": esp_rmse,
        "esp_relative_rmse": None if esp_reference_rms <= 1.0e-15 else esp_rmse / esp_reference_rms,
        "baseline_esp_rmse": float(np.sqrt(np.mean(baseline_residual**2))),
        "maximum_absolute_delta": float(maximum_absolute_delta),
        "constraint_rank": constraint_rank,
        "free_parameter_count": int(null_basis.shape[1]),
        "esp_design_rank": design_rank,
        "esp_design_condition_number": design_condition_number,
        "restraint": {
            "center": "standard_ff",
            "grid_objective": "mean_squared_error",
            "hyperbolic_scale": hyperbolic_scale,
            "hyperbolic_tightness": hyperbolic_tightness,
            "harmonic_stability_scale": harmonic_scale,
            "hydrogen_hyperbolic_restraint": False,
        },
        "constraints": [
            {
                "label": label,
                "target": target,
                "actual": float(actual),
                "residual": float(actual - target),
            }
            for label, target, actual in zip(
                constraint_labels,
                constraint_targets,
                constraint_values,
                strict=True,
            )
        ],
    }


def render_constrained_charge_table(result: dict[str, object]) -> str:
    """Render an auditable table; JSON remains the machine-readable result."""
    lines = [
        "# SIMPLE protein-site RESP charges",
        f"# solver: {result.get('solver_version')}",
        "# Only APPLY rows are patched into the topology; FIXED rows must remain unchanged.",
        "# idx  top_idx  residue                atom      original       fitted         delta  status",
    ]
    for row in result.get("charges") or []:
        topology_index = "-" if row.get("topology_index") is None else str(int(row["topology_index"]))
        original = float(row.get("original_charge") or 0.0)
        charge = float(row.get("charge") or 0.0)
        delta = charge - original
        if bool(row.get("fixed")):
            status = "FIXED"
        elif bool(row.get("apply")):
            status = "APPLY"
        else:
            status = "ENV"
        lines.append(
            f"{int(row.get('index') or 0):5d}  {topology_index:>7s}  "
            f"{str(row.get('residue_key') or '-'):22.22s}  {str(row.get('name') or '-'):>6.6s}  "
            f"{original: .8f}  {charge: .8f}  {delta:+.8f}  {status}"
        )
    lines.extend(
        [
            f"# ESP RMSE: {float(result.get('esp_rmse') or 0.0):.10g}",
            f"# Baseline ESP RMSE: {float(result.get('baseline_esp_rmse') or 0.0):.10g}",
            f"# Maximum |delta|: {float(result.get('maximum_absolute_delta') or 0.0):.10g} e",
            f"# Maximum constraint residual: {float(result.get('maximum_constraint_residual') or 0.0):.3e}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_constrained_runtime_resp_fit_script(
    *,
    atom_names: list[str],
    total_charge: float,
    equality_pairs: list[tuple[int, int]],
    fixed_charges: dict[int, float],
    sum_constraints: list[dict[str, object]],
    atom_metadata: list[dict[str, object]],
    xyz_filename: str = "site_resp.xyz",
    grid_filename: str = "site_resp.grid",
    fingerprint: str | None = None,
) -> str:
    """Render a standalone constrained fitter for cluster execution hosts."""
    module_source = Path(__file__).read_text(encoding="utf-8")
    payload = {
        "atom_names": atom_names,
        "total_charge": float(total_charge),
        "equality_pairs": [[int(first), int(second)] for first, second in equality_pairs],
        "fixed_charges": {str(int(index)): float(charge) for index, charge in fixed_charges.items()},
        "sum_constraints": sum_constraints,
        "atom_metadata": atom_metadata,
        "xyz_filename": xyz_filename,
        "grid_filename": grid_filename,
        "fingerprint": fingerprint,
    }
    runner = f'''\n\nif __name__ == "__main__":\n    import numpy as np\n    _PAYLOAD = {json.dumps(payload, sort_keys=True)}\n    _grid_path = Path(_PAYLOAD["grid_filename"])\n    if not _grid_path.exists():\n        _matches = sorted(Path(".").glob("*.grid"))\n        if not _matches:\n            raise FileNotFoundError(f"No NWChem ESP grid file was found; expected {{_grid_path}}.")\n        _grid_path = _matches[0]\n    _xyz_path = Path(_PAYLOAD["xyz_filename"])\n    if not _xyz_path.exists():\n        _stem_match = _grid_path.with_suffix(".xyz")\n        if _stem_match.exists():\n            _xyz_path = _stem_match\n    _xyz_lines = _xyz_path.read_text(encoding="utf-8").splitlines()[2:]\n    _coords = [[float(value) / 0.529177 for value in line.split()[1:4]] for line in _xyz_lines]\n    _grid_lines = _grid_path.read_text(encoding="utf-8").splitlines()\n    _point_count = int(_grid_lines[0].split()[0])\n    _grid = [[float(value) for value in line.split()[:4]] for line in _grid_lines[1:1 + _point_count]]\n    _result = fit_constrained_resp_payload(\n        atom_names=list(_PAYLOAD["atom_names"]),\n        coordinates_bohr=_coords,\n        grid_rows=_grid,\n        total_charge=float(_PAYLOAD["total_charge"]),\n        equality_pairs=[tuple(pair) for pair in _PAYLOAD["equality_pairs"]],\n        fixed_charges={{int(index): float(charge) for index, charge in _PAYLOAD["fixed_charges"].items()}},\n        sum_constraints=list(_PAYLOAD["sum_constraints"]),\n        atom_metadata=list(_PAYLOAD["atom_metadata"]),\n    )\n    Path("site_resp_charges.json").write_text(json.dumps(_result, indent=2, sort_keys=True), encoding="utf-8")\n    Path("site_resp_charges.txt").write_text("\\n".join(f"{{float(item['charge']):.10f}}" for item in _result["charges"]) + "\\n", encoding="utf-8")\n    print(json.dumps({{"esp_rmse": _result["esp_rmse"], "maximum_constraint_residual": _result["maximum_constraint_residual"]}}, indent=2))\n'''
    runner = runner.replace(
        "_PAYLOAD = " + json.dumps(payload, sort_keys=True),
        "_PAYLOAD = " + repr(payload),
    )
    runner = runner.replace(
        '    Path("site_resp_charges.json").write_text',
        '    _result["fingerprint"] = _PAYLOAD.get("fingerprint")\n'
        '    Path("site_resp_charges.json").write_text',
    )
    runner = "\n".join(
        (
            '    Path("site_resp_charges.txt").write_text('
            'render_constrained_charge_table(_result), encoding="utf-8")'
            if 'Path("site_resp_charges.txt").write_text' in line
            else line
        )
        for line in runner.splitlines()
    ) + "\n"
    return module_source + runner


def render_runtime_resp_fit_script(
    *,
    atom_names: list[str],
    total_charge: int,
    equality_pairs: list[tuple[int, int]],
    xyz_filename: str = "resp_job.xyz",
    grid_filename: str = "resp_job.grid",
) -> str:
    names_json = json.dumps(atom_names)
    pairs_json = json.dumps([[int(i), int(j)] for i, j in equality_pairs])
    lines = [
        "#!/usr/bin/env python3",
        "import copy",
        "import json",
        "import math",
        "import os",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "",
        f"NAMES = {names_json}",
        f"TOTAL_CHARGE = {int(total_charge)}",
        f"EQUALITY_PAIRS = {pairs_json}",
        f'XYZ_FILE = Path("{xyz_filename}")',
        f'GRID_FILE = Path("{grid_filename}")',
        "",
        "def resolve_grid_file(preferred):",
        "    if preferred.exists():",
        "        return preferred",
        "    matches = sorted(Path('.').glob('*.grid'))",
        "    if matches:",
        "        return matches[0]",
        "    raise FileNotFoundError(f'No RESP grid file was found. Expected {preferred} or any *.grid in {os.getcwd()}.')",
        "",
        "def resolve_xyz_file(preferred, grid_file):",
        "    stem_match = grid_file.with_suffix('.xyz')",
        "    if stem_match.exists():",
        "        return stem_match",
        "    if preferred.exists():",
        "        return preferred",
        "    matches = sorted(path for path in Path('.').glob('*.xyz') if path.name != preferred.name)",
        "    if matches:",
        "        return matches[0]",
        "    raise FileNotFoundError(f'No RESP XYZ file was found. Expected {preferred}, {stem_match.name}, or any *.xyz in {os.getcwd()}.')",
        "",
        "def norm2(vec):",
        "    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)",
        "",
        "def load_xyz(path, natoms):",
        "    with path.open('r', encoding='utf-8') as handle:",
        "        handle.readline()",
        "        handle.readline()",
        "        coords = np.zeros((natoms, 3))",
        "        for atom_index in range(natoms):",
        "            tokens = handle.readline().split()",
        "            coords[atom_index] = [float(value) / 0.529177 for value in tokens[1:4]]",
        "    return coords",
        "",
        "def load_grid(path):",
        "    with path.open('r', encoding='utf-8') as handle:",
        "        point_count = int(handle.readline().split()[0])",
        "        grid = np.zeros((point_count, 4))",
        "        for point_index in range(point_count):",
        "            grid[point_index] = [float(value) for value in handle.readline().split()]",
        "    return grid",
        "",
        "def build_groups(natoms, equality_pairs):",
        "    adjacency = [set() for _ in range(natoms)]",
        "    for first_atom, second_atom in equality_pairs:",
        "        if first_atom < 0 or second_atom < 0 or first_atom >= natoms or second_atom >= natoms:",
        "            raise ValueError(",
        "                'RESP equality constraints referenced an atom index outside the available atom range. '",
        "                f'Received pair ({first_atom}, {second_atom}) for {natoms} atoms.'",
        "            )",
        "        adjacency[first_atom].add(second_atom)",
        "        adjacency[second_atom].add(first_atom)",
        "    groups = []",
        "    visited = set()",
        "    for atom_index in range(natoms):",
        "        if atom_index in visited:",
        "            continue",
        "        stack = [atom_index]",
        "        component = []",
        "        visited.add(atom_index)",
        "        while stack:",
        "            current = stack.pop()",
        "            component.append(current)",
        "            for neighbor in sorted(adjacency[current]):",
        "                if neighbor in visited:",
        "                    continue",
        "                visited.add(neighbor)",
        "                stack.append(neighbor)",
        "        groups.append(sorted(component))",
        "    return groups",
        "",
        "def main():",
        "    grid_file = resolve_grid_file(GRID_FILE)",
        "    xyz_file = resolve_xyz_file(XYZ_FILE, grid_file)",
        "    natoms = len(NAMES)",
        "    coords = load_xyz(xyz_file, natoms)",
        "    grid = load_grid(grid_file)",
        "    groups = build_groups(natoms, EQUALITY_PAIRS)",
        "    group_count = len(groups)",
        "    matrix_size = group_count + 1",
        "    A = np.zeros((matrix_size, matrix_size))",
        "    B = np.zeros(matrix_size)",
        "    group_distances = np.zeros((group_count, len(grid)))",
        "    group_sizes = np.zeros(group_count)",
        "    restrained_counts = np.zeros(group_count)",
        "    for group_index, group in enumerate(groups):",
        "        group_sizes[group_index] = float(len(group))",
        "        restrained_counts[group_index] = float(sum(0 if NAMES[atom_index].upper().startswith('H') else 1 for atom_index in group))",
        "        for atom_index in group:",
        "            for point_index in range(len(grid)):",
        "                group_distances[group_index, point_index] += 1.0 / norm2(coords[atom_index] - grid[point_index, :3])",
        "        B[group_index] = np.dot(grid[:, 3], group_distances[group_index])",
        "    for group_index in range(group_count):",
        "        for other_index in range(group_index, group_count):",
        "            A[group_index, other_index] = np.dot(group_distances[group_index], group_distances[other_index])",
        "            A[other_index, group_index] = copy.copy(A[group_index, other_index])",
        "    A[:group_count, group_count] = group_sizes",
        "    A[group_count, :group_count] = group_sizes",
        "    B[group_count] = TOTAL_CHARGE",
        "    qold, _, _, _ = np.linalg.lstsq(A, B, rcond=None)",
        "    for _ in range(50):",
        "        current = copy.deepcopy(A)",
        "        for group_index in range(group_count):",
        "            if restrained_counts[group_index] <= 0.0:",
        "                continue",
        "            current[group_index, group_index] += (",
        "                restrained_counts[group_index] * 0.005 / math.sqrt(qold[group_index] ** 2 + 0.01)",
        "            )",
        "        q, _, _, _ = np.linalg.lstsq(current, B, rcond=None)",
        "        delta = np.amax(np.abs(q - qold))",
        "        qold = copy.deepcopy(q)",
        "        if delta < 0.000001:",
        "            break",
        "    charges = [0.0] * natoms",
        "    for group_index, group in enumerate(groups):",
        "        for atom_index in group:",
        "            charges[atom_index] = float(qold[group_index])",
        "    payload = {",
        "        'charges': [",
        "            {'index': index + 1, 'name': name, 'charge': float(charges[index])}",
        "            for index, name in enumerate(NAMES)",
        "        ],",
        "        'total_charge': float(np.sum(charges)),",
        "        'constraint_count': len(EQUALITY_PAIRS),",
        "    }",
        "    Path('resp_charges.json').write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')",
        "    Path('resp_charges.txt').write_text(",
        "        '\\n'.join(f\"{float(item['charge']):.10f}\" for item in payload['charges']) + '\\n',",
        "        encoding='utf-8',",
        "    )",
        "    print('RESP charges')",
        "    for item in payload['charges']:",
        "        print(f\"{item['index']:4d} {item['name']:<8s} {item['charge']: .8f}\")",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ]
    return "\n".join(lines)
