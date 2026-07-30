from __future__ import annotations

import json
from pathlib import Path


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
