from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from amber_metallo.qm.nwchem import (
    AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM,
    AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY,
    AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL,
    AUTO_GROUP_GRAPH_METHOD_HYBRID_HUCKEL,
    AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY,
    AUTO_GROUP_MODE_HYDROGEN_ONLY,
    MoleculeAtom,
    MoleculeBond,
    MoleculeData,
    QM_BASIS_OPTIONS,
    QM_DEFAULT_DFT_BASIS,
    QM_DEFAULT_DFT_FUNCTIONAL,
    QM_DEFAULT_GEOMETRY_MODE,
    QM_DEFAULT_GRID,
    QM_DEFAULT_MAXITER,
    QM_DEFAULT_MEMORY_MB,
    QM_DEFAULT_RESP_BASIS,
    QM_DEFAULT_RESP_FUNCTIONAL,
    QM_DEFAULT_XTB_ACC,
    QM_FUNCTIONAL_OPTIONS,
    QM_GEOMETRY_MODE_OPTIONS,
    QM_GRID_OPTIONS,
    geometry_mode_uses_dft_optimization,
    geometry_mode_uses_xtb_preopt,
    metal_safe_qm_settings,
    molecule_contains_supported_metal,
    normalize_qm_settings,
    suggest_group_constraints,
)
from amber_metallo.reporting import print_notice


_GEOMETRY_MODES = list(QM_GEOMETRY_MODE_OPTIONS)
_FUNCTIONAL_OPTIONS = list(QM_FUNCTIONAL_OPTIONS)
_BASIS_OPTIONS = list(QM_BASIS_OPTIONS)
_GRID_OPTIONS = list(QM_GRID_OPTIONS)
_GROUP_COLORS_HEX = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#ea580c",
    "#4338ca",
]
_ELEMENT_COLORS = {
    "H": "#ffffff",
    "C": "#8b95a1",
    "N": "#2563eb",
    "O": "#ef4444",
    "S": "#facc15",
    "P": "#f97316",
    "F": "#22c55e",
    "CL": "#16a34a",
    "BR": "#b45309",
    "I": "#7c3aed",
    "B": "#f59e0b",
}
_METAL_ELEMENTS = {"CO", "CU", "NI", "MN", "FE", "Y", "LA", "ND", "EU", "LU"}
_METAL_COLORS = {
    "FE": "#7f1d1d",
    "CO": "#6d28d9",
    "CU": "#92400e",
    "NI": "#14532d",
    "MN": "#581c87",
    "Y": "#0f766e",
    "LA": "#1d4ed8",
    "ND": "#4338ca",
    "EU": "#9f1239",
    "LU": "#0f172a",
}
_AUTO_GROUP_MODE_OPTIONS = [
    (AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY, "H + Symmetry"),
    (AUTO_GROUP_MODE_HYDROGEN_ONLY, "H Only"),
]
_AUTO_GROUP_GRAPH_METHOD_OPTIONS = [
    (AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY, "Connectivity (fast)"),
    (AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM, "Exact graph symmetry"),
]
_AUTO_GROUP_GRAPH_METHOD_DESCRIPTIONS = {
    AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY: "Connectivity uses the current bond graph and is the default fast path.",
    AUTO_GROUP_GRAPH_METHOD_AUTOMORPHISM: "Exact graph symmetry uses graph automorphisms to find symmetry-equivalent atoms.",
    AUTO_GROUP_GRAPH_METHOD_EXTENDED_HUCKEL: "Extended Huckel uses RDKit EHT bond orders and may be slow for metal complexes.",
    AUTO_GROUP_GRAPH_METHOD_HYBRID_HUCKEL: "Hybrid Huckel uses EHT for heavy-atom edges and current connectivity for H atoms.",
}
_VISIBLE_AUTO_GROUP_GRAPH_METHODS = {key for key, _label in _AUTO_GROUP_GRAPH_METHOD_OPTIONS}
_TK_INTERACTIVE_REDRAW_MS = 24
_RESP_EDITOR_REVIEW_TITLE = "Review Auto-Detected RESP Constraints"
_RESP_EDITOR_REVIEW_TEXT = (
    "Apply Auto can suggest H Only groups for hydrogens attached to the same heavy atom, "
    "or H + Symmetry groups using either fast connectivity or exact graph symmetry. "
    "Please review the suggested constraints and adjust them before saving."
)
_RESP_EDITOR_CONTROLS_HINT = (
    "Click selects one atom, Ctrl+click adds or removes atoms, and clicking empty space clears the selection. "
    "Right-drag rotates, the mouse wheel zooms, middle-drag pans, and '=' resets the view."
)
_RESP_EDITOR_SELECTION_IDLE_TEXT = "Click one atom, Ctrl+click to add atoms, or drag a box to highlight atoms in the molecule view."
_GEOMETRY_METHOD_LABEL = "Method"
_GEOMETRY_BASIS_SET_LABEL = "Basis Set"
_RESP_METHOD_LABEL = "Method"
_RESP_BASIS_SET_LABEL = "Basis Set"
_RESP_MATCH_GEOMETRY_LABEL = "Use the same method and basis set as geometry optimization"


def _resp_editor_selection_text(selected_count: int) -> str:
    if selected_count <= 0:
        return _RESP_EDITOR_SELECTION_IDLE_TEXT
    atom_label = "atom" if selected_count == 1 else "atoms"
    return f"{selected_count} {atom_label} highlighted in the molecule view."


def _option_label_for(options: list[tuple[str, str]] | tuple[tuple[str, str], ...], key: str, default: str) -> str:
    for option_key, option_label in options:
        if option_key == key:
            return option_label
    return default


def _option_key_for(options: list[tuple[str, str]] | tuple[tuple[str, str], ...], label: str, default: str) -> str:
    selected = str(label or "").strip()
    for option_key, option_label in options:
        if option_label == selected:
            return option_key
    return default


def _normalized_editor_qm_settings(qm_settings: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(qm_settings or {})
    return normalize_qm_settings(
        source,
        net_charge=int(source.get("net_charge") or 0),
        multiplicity=int(source.get("multiplicity") or 1),
    )


def _normalized_editor_state(session_state: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(json.dumps(session_state))
    state["qm_settings"] = _normalized_editor_qm_settings(state.get("qm_settings"))
    molecule = _molecule_from_editor_state(state)
    if molecule_contains_supported_metal(molecule):
        state["qm_settings"] = metal_safe_qm_settings(
            state.get("qm_settings"),
            net_charge=int(state["qm_settings"].get("net_charge") or 0),
            multiplicity=int(state["qm_settings"].get("multiplicity") or 1),
        )
    return state


def _geometry_mode_from_qm(qm_settings: dict[str, Any]) -> str:
    return str((qm_settings.get("geometry") or {}).get("mode") or QM_DEFAULT_GEOMETRY_MODE)


def _xtb_acc_from_qm(qm_settings: dict[str, Any]) -> float:
    geometry = dict(qm_settings.get("geometry") or {})
    xtb_preopt = dict(geometry.get("xtb_preopt") or {})
    return float(xtb_preopt.get("acc") or QM_DEFAULT_XTB_ACC)


def _dft_functional_from_qm(qm_settings: dict[str, Any]) -> str:
    geometry = dict(qm_settings.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    return str(dft_optimization.get("functional") or QM_DEFAULT_DFT_FUNCTIONAL)


def _dft_basis_from_qm(qm_settings: dict[str, Any]) -> str:
    geometry = dict(qm_settings.get("geometry") or {})
    dft_optimization = dict(geometry.get("dft_optimization") or {})
    return str(dft_optimization.get("basis") or QM_DEFAULT_DFT_BASIS)


def _resp_settings_from_qm(qm_settings: dict[str, Any]) -> tuple[bool, str, str]:
    resp = dict(qm_settings.get("resp") or {})
    same_as_dft = bool(resp.get("same_as_dft_optimization"))
    functional = str(resp.get("functional") or QM_DEFAULT_RESP_FUNCTIONAL)
    basis = str(resp.get("basis") or QM_DEFAULT_RESP_BASIS)
    return same_as_dft, functional, basis


def _resource_settings_from_qm(qm_settings: dict[str, Any]) -> tuple[int, str, int, int, int]:
    resources = dict(qm_settings.get("resources") or {})
    return (
        int(resources.get("memory_mb") or QM_DEFAULT_MEMORY_MB),
        str(resources.get("grid") or QM_DEFAULT_GRID),
        int(resources.get("maxiter") or QM_DEFAULT_MAXITER),
        int(qm_settings.get("net_charge") or 0),
        int(qm_settings.get("multiplicity") or 1),
    )


def _build_qm_settings(
    *,
    geometry_mode: str,
    xtb_acc: float,
    dft_functional: str,
    dft_basis: str,
    resp_same_as_dft_optimization: bool,
    resp_functional: str,
    resp_basis: str,
    memory_mb: int,
    grid: str,
    maxiter: int,
    net_charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    return normalize_qm_settings(
        {
            "net_charge": int(net_charge),
            "multiplicity": int(multiplicity),
            "resources": {
                "memory_mb": int(memory_mb),
                "grid": str(grid),
                "maxiter": int(maxiter),
            },
            "geometry": {
                "mode": str(geometry_mode),
                "xtb_preopt": {
                    "acc": float(xtb_acc),
                },
                "dft_optimization": {
                    "functional": str(dft_functional),
                    "basis": str(dft_basis),
                },
            },
            "resp": {
                "same_as_dft_optimization": bool(resp_same_as_dft_optimization),
                "functional": str(resp_functional),
                "basis": str(resp_basis),
            },
        },
        net_charge=int(net_charge),
        multiplicity=int(multiplicity),
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _loadable_shared_library(*names: str) -> bool:
    for name in names:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def resp_editor_launch_status() -> tuple[bool, str | None]:
    if _env_flag("SIMPLE_DISABLE_RESP_POPUP"):
        return False, "RESP popup is disabled by SIMPLE_DISABLE_RESP_POPUP."

    if not sys.platform.startswith("linux"):
        return True, None

    display = os.environ.get("DISPLAY", "").strip()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "").strip()
    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()

    if not display and not wayland_display:
        return (
            False,
            "No DISPLAY or WAYLAND_DISPLAY session was detected, so the RESP popup cannot be shown in this shell.",
        )

    if wayland_display and session_type == "wayland":
        return True, None

    if display and not _loadable_shared_library("libxcb-cursor.so.0", "libxcb-cursor.so"):
        return (
            False,
            "Qt needs libxcb-cursor to launch the xcb platform plugin, but it was not found on this system.",
        )

    return True, None


def _tk_popup_allowed() -> bool:
    if _env_flag("SIMPLE_DISABLE_RESP_TK_POPUP"):
        return False
    if not sys.platform.startswith("linux"):
        return True
    display = os.environ.get("DISPLAY", "").strip()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "").strip()
    return bool(display or wayland_display)


def _prefer_tk_popup() -> bool:
    if _env_flag("SIMPLE_PREFER_RESP_TK_POPUP"):
        return True
    return bool(os.environ.get("SSH_CONNECTION", "").strip() and os.environ.get("DISPLAY", "").strip())


def _group_color_hex(group_id: int | None) -> str:
    if not group_id:
        return "#94a3b8"
    return _GROUP_COLORS_HEX[(int(group_id) - 1) % len(_GROUP_COLORS_HEX)]


def _element_fill_color(element: str | None) -> str:
    token = str(element or "").strip().upper()
    if token in _METAL_COLORS:
        return _METAL_COLORS[token]
    return _ELEMENT_COLORS.get(token, "#cbd5e1")


def _is_metal_element(element: str | None) -> bool:
    return str(element or "").strip().upper() in _METAL_ELEMENTS


def _group_badge_text(group_id: int | None) -> str:
    return "" if not group_id else f"G{int(group_id)}"


def _auto_group_mode_label(mode: str | None) -> str:
    selected = str(mode or AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY)
    for option_key, option_label in _AUTO_GROUP_MODE_OPTIONS:
        if option_key == selected:
            return option_label
    return _AUTO_GROUP_MODE_OPTIONS[0][1]


def _auto_group_mode_key(label: str | None) -> str:
    selected = str(label or "").strip()
    for option_key, option_label in _AUTO_GROUP_MODE_OPTIONS:
        if option_label == selected:
            return option_key
    return AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY


def _auto_group_graph_method_label(method: str | None) -> str:
    selected = str(method or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY)
    if selected not in _VISIBLE_AUTO_GROUP_GRAPH_METHODS:
        selected = AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
    for option_key, option_label in _AUTO_GROUP_GRAPH_METHOD_OPTIONS:
        if option_key == selected:
            return option_label
    return _AUTO_GROUP_GRAPH_METHOD_OPTIONS[0][1]


def _auto_group_graph_method_key(label: str | None) -> str:
    selected = str(label or "").strip()
    for option_key, option_label in _AUTO_GROUP_GRAPH_METHOD_OPTIONS:
        if option_label == selected:
            return option_key
    return AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY


def _auto_group_graph_method_description(method: str | None) -> str:
    selected = str(method or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY)
    if selected not in _VISIBLE_AUTO_GROUP_GRAPH_METHODS:
        selected = AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
    return _AUTO_GROUP_GRAPH_METHOD_DESCRIPTIONS.get(
        selected,
        _AUTO_GROUP_GRAPH_METHOD_DESCRIPTIONS[AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY],
    )


def _lighten_hex(fill_hex: str, factor: float = 0.78) -> str:
    token = fill_hex.lstrip("#")
    if len(token) != 6:
        return "#e2e8f0"
    red = int(token[0:2], 16)
    green = int(token[2:4], 16)
    blue = int(token[4:6], 16)
    mix = max(0.0, min(1.0, factor))
    red = int(red + ((255 - red) * mix))
    green = int(green + ((255 - green) * mix))
    blue = int(blue + ((255 - blue) * mix))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _blend_hex(fill_hex: str, target_hex: str, factor: float) -> str:
    fill_token = fill_hex.lstrip("#")
    target_token = target_hex.lstrip("#")
    if len(fill_token) != 6 or len(target_token) != 6:
        return fill_hex
    mix = max(0.0, min(1.0, factor))
    fill_rgb = [int(fill_token[index : index + 2], 16) for index in (0, 2, 4)]
    target_rgb = [int(target_token[index : index + 2], 16) for index in (0, 2, 4)]
    blended = [
        int(fill_value + ((target_value - fill_value) * mix))
        for fill_value, target_value in zip(fill_rgb, target_rgb, strict=True)
    ]
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _text_color_for_fill(fill_hex: str) -> str:
    token = fill_hex.lstrip("#")
    if len(token) != 6:
        return "#111827"
    red = int(token[0:2], 16)
    green = int(token[2:4], 16)
    blue = int(token[4:6], 16)
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    return "#111827" if luminance >= 170 else "#ffffff"


def _identity_rotation_matrix() -> tuple[tuple[float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matrix_vector_multiply(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        (matrix[0][0] * vector[0]) + (matrix[0][1] * vector[1]) + (matrix[0][2] * vector[2]),
        (matrix[1][0] * vector[0]) + (matrix[1][1] * vector[1]) + (matrix[1][2] * vector[2]),
        (matrix[2][0] * vector[0]) + (matrix[2][1] * vector[1]) + (matrix[2][2] * vector[2]),
    )


def _matrix_multiply(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    rows: list[tuple[float, float, float]] = []
    for row_index in range(3):
        row_values: list[float] = []
        for column_index in range(3):
            row_values.append(
                sum(left[row_index][shared_index] * right[shared_index][column_index] for shared_index in range(3))
            )
        rows.append((row_values[0], row_values[1], row_values[2]))
    return tuple(rows)


def _axis_angle_rotation_matrix(
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[tuple[float, float, float], ...]:
    axis_length = math.sqrt((axis[0] ** 2) + (axis[1] ** 2) + (axis[2] ** 2))
    if axis_length <= 1e-8 or abs(angle) <= 1e-8:
        return _identity_rotation_matrix()
    axis_x = axis[0] / axis_length
    axis_y = axis[1] / axis_length
    axis_z = axis[2] / axis_length
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return (
        (
            cosine + (axis_x * axis_x * one_minus_cosine),
            (axis_x * axis_y * one_minus_cosine) - (axis_z * sine),
            (axis_x * axis_z * one_minus_cosine) + (axis_y * sine),
        ),
        (
            (axis_y * axis_x * one_minus_cosine) + (axis_z * sine),
            cosine + (axis_y * axis_y * one_minus_cosine),
            (axis_y * axis_z * one_minus_cosine) - (axis_x * sine),
        ),
        (
            (axis_z * axis_x * one_minus_cosine) - (axis_y * sine),
            (axis_z * axis_y * one_minus_cosine) + (axis_x * sine),
            cosine + (axis_z * axis_z * one_minus_cosine),
        ),
    )


def _friendly_popup_warning_message(warning: str | None) -> str | None:
    if not warning:
        return None
    return "Some enhanced viewer features were unavailable in this session, so the built-in RESP editor preview is being used."


def _qt_viewer_html(assets_dir: Path, initial_state: dict[str, Any]) -> str:
    index_path = assets_dir / "index.html"
    html_text = index_path.read_text(encoding="utf-8")
    initial_state_json = json.dumps(initial_state).replace("</", "<\\/")
    bootstrap = f'<script id="resp-initial-state" type="application/json">{initial_state_json}</script>'
    if "<script src=\"app.js\"></script>" in html_text:
        return html_text.replace("<script src=\"app.js\"></script>", f"{bootstrap}\n  <script src=\"app.js\"></script>", 1)
    return html_text + "\n" + bootstrap


def _molecule_from_editor_state(session_state: dict[str, Any]) -> MoleculeData:
    payload = (session_state.get("molecule") or {}) if isinstance(session_state, dict) else {}
    atoms = [
        MoleculeAtom(
            index=int(item["index"]),
            name=str(item["name"]),
            element=str(item["element"]),
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
        )
        for item in (payload.get("atoms") or [])
    ]
    bonds = [
        MoleculeBond(
            first=int(item["first"]),
            second=int(item["second"]),
            order=int(item.get("order") or 1),
        )
        for item in (payload.get("bonds") or [])
    ]
    return MoleculeData(
        source_file=str(session_state.get("source_file") or payload.get("source_file") or ""),
        source_format=str(session_state.get("source_format") or payload.get("source_format") or "unknown"),
        atoms=atoms,
        bonds=bonds,
    )


def _should_use_low_detail_canvas_render(*, is_rotating: bool, box_select_active: bool) -> bool:
    return bool(is_rotating or box_select_active)


def _launch_tk_editor(
    *,
    session_state: dict[str, Any],
    output_dir: str | Path,
    warning: str | None = None,
) -> dict[str, Any] | None:
    if not _tk_popup_allowed():
        return None

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return None

    state = _normalized_editor_state(session_state)
    initial_state = json.loads(json.dumps(state))
    loadable_molecule = _molecule_from_editor_state(initial_state)
    selected_atoms: set[int] = set()
    projected_positions: dict[int, tuple[float, float, float, float]] = {}
    left_drag_state: dict[str, Any] = {"start": None, "rect_id": None, "box_active": False}
    rotate_state: dict[str, float | None] = {"x": None, "y": None}
    pan_state: dict[str, float | None] = {"x": None, "y": None}
    render_state: dict[str, Any] = {"after_id": None, "last_draw_at": 0.0, "low_detail": False}
    view_orientation = {"matrix": _identity_rotation_matrix()}
    view_state = {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0}
    suppress_tree_callback = False
    result_payload: dict[str, Any] | None = None

    try:
        root = tk.Tk()
    except Exception:
        return None

    root.title("RESP Editor")
    root.geometry("1280x860")
    root.minsize(980, 700)
    root.resizable(True, True)

    friendly_warning = _friendly_popup_warning_message(warning)
    if warning:
        print_notice(
            "RESP Tk Popup",
            f"{warning}\n\nUsing the built-in RESP popup for this session.",
            border_style="cyan",
        )

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    paned = tk.PanedWindow(
        root,
        orient="horizontal",
        sashwidth=10,
        sashrelief="raised",
        showhandle=True,
        bg="#d7e1ec",
        bd=0,
    )
    paned.grid(row=0, column=0, sticky="nsew")

    controls_host = ttk.Frame(paned)
    viewer_host = ttk.Frame(paned)
    paned.add(controls_host, minsize=360)
    paned.add(viewer_host, minsize=560)

    def _set_initial_pane_sizes() -> None:
        try:
            paned.sash_place(0, 430, 1)
        except Exception:
            pass

    root.after_idle(_set_initial_pane_sizes)

    controls = ttk.Frame(controls_host, padding=16)
    controls.pack(fill="both", expand=True)
    controls.columnconfigure(0, weight=1)

    right_host = ttk.Frame(viewer_host, padding=(0, 16, 16, 16))
    right_host.pack(fill="both", expand=True)
    right_host.columnconfigure(0, weight=1)
    right_host.columnconfigure(1, weight=0, minsize=250)
    right_host.rowconfigure(0, weight=1)

    viewer_frame = ttk.Frame(right_host)
    viewer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    viewer_frame.columnconfigure(0, weight=1)
    viewer_frame.rowconfigure(2, weight=1)

    qm_host = ttk.Frame(right_host)
    qm_host.grid(row=0, column=1, sticky="nsew")
    qm_host.columnconfigure(0, weight=1)

    molecule_atoms = state["molecule"]["atoms"]
    if molecule_atoms:
        center_x = sum(float(atom["x"]) for atom in molecule_atoms) / len(molecule_atoms)
        center_y = sum(float(atom["y"]) for atom in molecule_atoms) / len(molecule_atoms)
        center_z = sum(float(atom["z"]) for atom in molecule_atoms) / len(molecule_atoms)
        centered_atom_positions = [
            (
                int(atom["index"]),
                float(atom["x"]) - center_x,
                float(atom["y"]) - center_y,
                float(atom["z"]) - center_z,
            )
            for atom in molecule_atoms
        ]
        molecule_radius = max(
            math.sqrt((x_coord ** 2) + (y_coord ** 2) + (z_coord ** 2))
            for _atom_index, x_coord, y_coord, z_coord in centered_atom_positions
        )
        molecule_radius = max(molecule_radius, 1.0)
    else:
        centered_atom_positions = []
        molecule_radius = 1.0

    ttk.Label(controls, text="RESP Editor", font=("TkDefaultFont", 15, "bold")).grid(row=0, column=0, sticky="w")
    if friendly_warning:
        tk.Label(
            controls,
            text=friendly_warning,
            wraplength=420,
            justify="left",
            bg="#fff7ed",
            fg="#9a3412",
            padx=12,
            pady=10,
            font=("TkDefaultFont", 10, "bold"),
            highlightbackground="#fdba74",
            highlightthickness=1,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 10))

    review_box = tk.Frame(controls, bg="#eff6ff", highlightbackground="#bfdbfe", highlightthickness=1)
    review_box.grid(row=2, column=0, sticky="ew", pady=(8, 12))
    tk.Label(
        review_box,
        text=_RESP_EDITOR_REVIEW_TITLE,
        font=("TkDefaultFont", 13, "bold"),
        bg="#eff6ff",
        fg="#1e3a8a",
        anchor="w",
        justify="left",
        padx=12,
        pady=8,
    ).pack(fill="x")
    review_text_label = tk.Label(
        review_box,
        text=_RESP_EDITOR_REVIEW_TEXT,
        wraplength=396,
        justify="left",
        bg="#eff6ff",
        fg="#0f172a",
        padx=12,
        font=("TkDefaultFont", 10),
    )
    review_text_label.pack(fill="x", pady=(0, 6))
    review_confirm_label = tk.Label(
        review_box,
        text="Please confirm the suggested groups before saving.",
        wraplength=396,
        justify="left",
        bg="#eff6ff",
        fg="#1d4ed8",
        padx=12,
        font=("TkDefaultFont", 10, "bold"),
    )
    review_confirm_label.pack(fill="x", pady=(0, 12))

    hint_var = tk.StringVar(value=_RESP_EDITOR_CONTROLS_HINT)
    ttk.Label(controls, textvariable=hint_var, wraplength=420, justify="left").grid(row=3, column=0, sticky="ew", pady=(0, 8))

    selection_var = tk.StringVar(value=_RESP_EDITOR_SELECTION_IDLE_TEXT)
    ttk.Label(
        controls,
        textvariable=selection_var,
        foreground="#1d4ed8",
        wraplength=396,
        justify="left",
    ).grid(row=4, column=0, sticky="ew", pady=(0, 8))

    atom_columns = ("index", "name", "element", "group")
    atom_tree = ttk.Treeview(controls, columns=atom_columns, show="headings", selectmode="extended", height=16)
    for column, heading, width in (
        ("index", "Index", 60),
        ("name", "Name", 90),
        ("element", "Element", 80),
        ("group", "Group", 80),
    ):
        atom_tree.heading(column, text=heading)
        atom_tree.column(column, width=width, anchor="center")
    atom_tree.grid(row=5, column=0, sticky="nsew")
    controls.rowconfigure(5, weight=1)

    tree_scroll = ttk.Scrollbar(controls, orient="vertical", command=atom_tree.yview)
    atom_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.grid(row=5, column=1, sticky="ns")

    button_frame = ttk.Frame(controls)
    button_frame.grid(row=6, column=0, sticky="ew", pady=(10, 10))
    ttk.Button(button_frame, text="New Group", command=lambda: assign_new_group()).pack(side="left", padx=(0, 6))
    ttk.Button(button_frame, text="Clear Group", command=lambda: clear_selected_groups()).pack(side="left", padx=6)
    ttk.Button(button_frame, text="Apply Auto", command=lambda: reset_auto()).pack(side="left", padx=6)
    auto_group_var = tk.StringVar()
    auto_group_graph_method_var = tk.StringVar()
    ttk.Label(button_frame, text="Auto groups").pack(side="right", padx=(10, 6))
    auto_group_combo = ttk.Combobox(
        button_frame,
        textvariable=auto_group_var,
        state="readonly",
        values=[label for _key, label in _AUTO_GROUP_MODE_OPTIONS],
        width=18,
    )
    auto_group_combo.pack(side="right")

    auto_method_frame = ttk.Frame(controls)
    auto_method_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))
    auto_method_frame.columnconfigure(1, weight=1)
    ttk.Label(auto_method_frame, text="Graph method").grid(row=0, column=0, sticky="w", padx=(0, 8))
    auto_group_graph_method_combo = ttk.Combobox(
        auto_method_frame,
        textvariable=auto_group_graph_method_var,
        state="readonly",
        values=[label for _key, label in _AUTO_GROUP_GRAPH_METHOD_OPTIONS],
        width=22,
    )
    auto_group_graph_method_combo.grid(row=0, column=1, sticky="ew")
    auto_group_graph_method_note_var = tk.StringVar()
    tk.Label(
        auto_method_frame,
        textvariable=auto_group_graph_method_note_var,
        fg="#475569",
        justify="left",
        wraplength=396,
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    legend_box = ttk.LabelFrame(controls, text="Groups", padding=12)
    legend_box.grid(row=8, column=0, sticky="ew")
    group_tree = ttk.Treeview(legend_box, columns=("group", "label", "atoms"), show="headings", selectmode="browse", height=8)
    for column, heading, width in (
        ("group", "Group", 70),
        ("label", "Label", 170),
        ("atoms", "Atoms", 160),
    ):
        group_tree.heading(column, text=heading)
        group_tree.column(column, width=width, anchor="center" if column == "group" else "w")
    group_tree.pack(fill="both", expand=True)

    footer = ttk.Frame(controls)
    footer.grid(row=9, column=0, sticky="e", pady=(12, 0))
    ttk.Button(footer, text="Cancel", command=lambda: on_cancel()).pack(side="left", padx=(0, 8))
    ttk.Button(footer, text="Save And Continue", command=lambda: on_save()).pack(side="left")
    ttk.Sizegrip(footer).pack(side="right", padx=(12, 0))

    ttk.Label(
        viewer_frame,
        text="Molecule View",
        font=("TkDefaultFont", 13, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 6))
    ttk.Label(
        viewer_frame,
        text="Right-drag rotates, the mouse wheel zooms, middle-drag pans, and '=' resets the view.",
        justify="left",
        wraplength=520,
    ).grid(row=1, column=0, sticky="w", pady=(0, 10))
    canvas = tk.Canvas(viewer_frame, background="#050814", highlightthickness=1, highlightbackground="#1e293b")
    canvas.grid(row=2, column=0, sticky="nsew")

    ttk.Label(
        qm_host,
        text="QM Input",
        font=("TkDefaultFont", 13, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 10))

    geometry_var = tk.StringVar()
    xtb_acc_var = tk.DoubleVar(value=QM_DEFAULT_XTB_ACC)
    dft_functional_var = tk.StringVar()
    dft_basis_var = tk.StringVar()
    resp_same_var = tk.BooleanVar(value=False)
    resp_functional_var = tk.StringVar()
    resp_basis_var = tk.StringVar()
    grid_var = tk.StringVar()
    memory_var = tk.IntVar(value=QM_DEFAULT_MEMORY_MB)
    maxiter_var = tk.IntVar(value=QM_DEFAULT_MAXITER)
    charge_var = tk.IntVar()
    multiplicity_var = tk.IntVar(value=1)

    geometry_box = ttk.LabelFrame(qm_host, text="Geometry Optimization", padding=12)
    geometry_box.grid(row=1, column=0, sticky="ew", pady=(0, 12))
    geometry_box.columnconfigure(1, weight=1)
    ttk.Label(geometry_box, text="Mode").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    geometry_combo = ttk.Combobox(
        geometry_box,
        textvariable=geometry_var,
        state="readonly",
        values=[label for _key, label in _GEOMETRY_MODES],
    )
    geometry_combo.grid(row=0, column=1, sticky="ew", pady=4)
    xtb_acc_label = ttk.Label(geometry_box, text="xTB acc")
    xtb_acc_spin = tk.Spinbox(geometry_box, textvariable=xtb_acc_var, from_=0.01, to=10.0, increment=0.01, format="%.2f")
    xtb_acc_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    xtb_acc_spin.grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Label(geometry_box, text=_GEOMETRY_METHOD_LABEL).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    dft_functional_combo = ttk.Combobox(
        geometry_box,
        textvariable=dft_functional_var,
        state="readonly",
        values=[label for _key, label in _FUNCTIONAL_OPTIONS],
    )
    dft_functional_combo.grid(row=2, column=1, sticky="ew", pady=4)
    ttk.Label(geometry_box, text=_GEOMETRY_BASIS_SET_LABEL).grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
    dft_basis_combo = ttk.Combobox(
        geometry_box,
        textvariable=dft_basis_var,
        state="readonly",
        values=[label for _key, label in _BASIS_OPTIONS],
    )
    dft_basis_combo.grid(row=3, column=1, sticky="ew", pady=4)

    resp_box = ttk.LabelFrame(qm_host, text="RESP Fitting", padding=12)
    resp_box.grid(row=2, column=0, sticky="ew", pady=(0, 12))
    resp_box.columnconfigure(1, weight=1)
    resp_same_check = ttk.Checkbutton(
        resp_box,
        text=_RESP_MATCH_GEOMETRY_LABEL,
        variable=resp_same_var,
    )
    resp_same_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
    ttk.Label(resp_box, text=_RESP_METHOD_LABEL).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    resp_functional_combo = ttk.Combobox(
        resp_box,
        textvariable=resp_functional_var,
        state="readonly",
        values=[label for _key, label in _FUNCTIONAL_OPTIONS],
    )
    resp_functional_combo.grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Label(resp_box, text=_RESP_BASIS_SET_LABEL).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    resp_basis_combo = ttk.Combobox(
        resp_box,
        textvariable=resp_basis_var,
        state="readonly",
        values=[label for _key, label in _BASIS_OPTIONS],
    )
    resp_basis_combo.grid(row=2, column=1, sticky="ew", pady=4)

    job_box = ttk.LabelFrame(qm_host, text="Job Settings", padding=12)
    job_box.grid(row=3, column=0, sticky="ew")
    for column in range(2):
        job_box.columnconfigure((column * 2) + 1, weight=1)
    ttk.Label(job_box, text="Grid").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    grid_combo = ttk.Combobox(
        job_box,
        textvariable=grid_var,
        state="readonly",
        values=[label for _key, label in _GRID_OPTIONS],
    )
    grid_combo.grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Label(job_box, text="Memory (MB)").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=4)
    memory_spin = tk.Spinbox(job_box, textvariable=memory_var, from_=250, to=20000, increment=250)
    memory_spin.grid(row=0, column=3, sticky="ew", pady=4)
    ttk.Label(job_box, text="Max Iter").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    maxiter_spin = tk.Spinbox(job_box, textvariable=maxiter_var, from_=10, to=1000, increment=10)
    maxiter_spin.grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Label(job_box, text="Net Charge").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=4)
    charge_spin = tk.Spinbox(job_box, textvariable=charge_var, from_=-10, to=10, increment=1)
    charge_spin.grid(row=1, column=3, sticky="ew", pady=4)
    ttk.Label(job_box, text="Multiplicity").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    multiplicity_spin = tk.Spinbox(job_box, textvariable=multiplicity_var, from_=1, to=10, increment=1)
    multiplicity_spin.grid(row=2, column=1, sticky="ew", pady=4)

    def _geometry_label_for(key: str) -> str:
        return _option_label_for(_GEOMETRY_MODES, key, _GEOMETRY_MODES[0][1])

    def _geometry_key_for(label: str) -> str:
        return _option_key_for(_GEOMETRY_MODES, label, QM_DEFAULT_GEOMETRY_MODE)

    def _functional_label_for(key: str) -> str:
        return _option_label_for(_FUNCTIONAL_OPTIONS, key, _FUNCTIONAL_OPTIONS[0][1])

    def _functional_key_for(label: str, default: str) -> str:
        return _option_key_for(_FUNCTIONAL_OPTIONS, label, default)

    def _basis_label_for(key: str) -> str:
        return _option_label_for(_BASIS_OPTIONS, key, _BASIS_OPTIONS[0][1])

    def _basis_key_for(label: str, default: str) -> str:
        return _option_key_for(_BASIS_OPTIONS, label, default)

    def _grid_label_for(key: str) -> str:
        return _option_label_for(_GRID_OPTIONS, key, _GRID_OPTIONS[0][1])

    def _grid_key_for(label: str) -> str:
        return _option_key_for(_GRID_OPTIONS, label, QM_DEFAULT_GRID)

    def _refresh_selection_label() -> None:
        selection_var.set(_resp_editor_selection_text(len(selected_atoms)))

    def _current_qm_settings() -> dict[str, Any]:
        return _build_qm_settings(
            geometry_mode=_geometry_key_for(geometry_var.get()),
            xtb_acc=float(xtb_acc_var.get()),
            dft_functional=_functional_key_for(dft_functional_var.get(), QM_DEFAULT_DFT_FUNCTIONAL),
            dft_basis=_basis_key_for(dft_basis_var.get(), QM_DEFAULT_DFT_BASIS),
            resp_same_as_dft_optimization=bool(resp_same_var.get()),
            resp_functional=_functional_key_for(resp_functional_var.get(), QM_DEFAULT_RESP_FUNCTIONAL),
            resp_basis=_basis_key_for(resp_basis_var.get(), QM_DEFAULT_RESP_BASIS),
            memory_mb=int(memory_var.get()),
            grid=_grid_key_for(grid_var.get()),
            maxiter=int(maxiter_var.get()),
            net_charge=int(charge_var.get()),
            multiplicity=int(multiplicity_var.get()),
        )

    def _sync_resp_level_from_dft(*_args) -> None:
        if not bool(resp_same_var.get()):
            return
        resp_functional_var.set(dft_functional_var.get())
        resp_basis_var.set(dft_basis_var.get())

    def _sync_qm_ui_state(*_args) -> None:
        geometry_mode = _geometry_key_for(geometry_var.get())
        if geometry_mode_uses_xtb_preopt(geometry_mode):
            xtb_acc_label.grid()
            xtb_acc_spin.grid()
        else:
            xtb_acc_label.grid_remove()
            xtb_acc_spin.grid_remove()

        dft_state = "readonly" if geometry_mode_uses_dft_optimization(geometry_mode) else "disabled"
        dft_functional_combo.configure(state=dft_state)
        dft_basis_combo.configure(state=dft_state)

        _sync_resp_level_from_dft()
        resp_state = "disabled" if bool(resp_same_var.get()) else "readonly"
        resp_functional_combo.configure(state=resp_state)
        resp_basis_combo.configure(state=resp_state)

    def _current_auto_group_mode() -> str:
        return _auto_group_mode_key(auto_group_var.get())

    def _current_auto_group_graph_method() -> str:
        return _auto_group_graph_method_key(auto_group_graph_method_var.get())

    def _next_group_id() -> int:
        values = [
            int(atom["group_id"])
            for atom in state["group_constraints"]["atoms"]
            if atom.get("group_id") is not None
        ]
        return max(values) + 1 if values else 1

    def _refresh_auto_group_mode() -> None:
        auto_group_var.set(
            _auto_group_mode_label((state.get("group_constraints") or {}).get("auto_group_mode"))
        )
        method = str(
            (state.get("group_constraints") or {}).get("auto_group_graph_method")
            or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
        )
        auto_group_graph_method_var.set(_auto_group_graph_method_label(method))
        warning = str((state.get("group_constraints") or {}).get("auto_group_graph_warning") or "")
        note = _auto_group_graph_method_description(method)
        auto_group_graph_method_note_var.set(f"{note} {warning}".strip())

    def _store_current_auto_group_mode() -> None:
        state["group_constraints"]["auto_group_mode"] = _current_auto_group_mode()
        state["group_constraints"]["auto_group_graph_method"] = _current_auto_group_graph_method()

    def _rebuild_groups() -> None:
        grouped: dict[int, list[int]] = {}
        label_lookup = {
            int(group["group_id"]): str(group.get("label") or f"Group {group['group_id']}")
            for group in state["group_constraints"].get("groups") or []
        }
        for atom in state["group_constraints"]["atoms"]:
            if atom.get("group_id") is None:
                continue
            grouped.setdefault(int(atom["group_id"]), []).append(int(atom["index"]))
        _store_current_auto_group_mode()
        state["group_constraints"]["groups"] = [
            {
                "group_id": group_id,
                "label": label_lookup.get(group_id, f"Group {group_id}"),
                "atom_indices": sorted(indices),
                "auto": False,
            }
            for group_id, indices in sorted(grouped.items())
            if len(indices) > 1
        ]

    def _refresh_legend() -> None:
        for item in group_tree.get_children():
            group_tree.delete(item)
        groups = state["group_constraints"].get("groups") or []
        if not groups:
            group_tree.insert("", "end", iid="empty", values=("", "No equality groups defined.", ""))
            return
        for group in groups:
            group_tree.insert(
                "",
                "end",
                iid=f"group_{group['group_id']}",
                values=(
                    f"G{group['group_id']}",
                    group.get("label") or "custom",
                    ", ".join(str(index) for index in group["atom_indices"]),
                ),
            )

    def _sync_tree_selection() -> None:
        nonlocal suppress_tree_callback
        suppress_tree_callback = True
        atom_tree.selection_remove(atom_tree.selection())
        if selected_atoms:
            atom_tree.selection_set([str(index) for index in sorted(selected_atoms)])
        suppress_tree_callback = False
        _refresh_selection_label()

    def _refresh_atom_tree() -> None:
        for item in atom_tree.get_children():
            atom_tree.delete(item)
        for atom in state["group_constraints"]["atoms"]:
            atom_tree.insert(
                "",
                "end",
                iid=str(atom["index"]),
                values=(
                    atom["index"],
                    atom["name"],
                    atom["element"],
                    "" if atom.get("group_id") is None else atom["group_id"],
                ),
            )
        _sync_tree_selection()

    def on_group_select(_event=None) -> None:
        nonlocal selected_atoms
        selection = group_tree.selection()
        if not selection:
            return
        item_id = str(selection[0])
        if not item_id.startswith("group_"):
            return
        group_id = int(item_id.split("_", maxsplit=1)[1])
        groups = state["group_constraints"].get("groups") or []
        for group in groups:
            if int(group["group_id"]) != group_id:
                continue
            selected_atoms = {int(atom_index) for atom_index in group["atom_indices"]}
            _sync_tree_selection()
            _schedule_canvas_redraw(immediate=True)
            return

    def _project_atoms(width: float, height: float) -> dict[int, tuple[float, float, float, float]]:
        if not centered_atom_positions:
            return {}

        rotated: list[tuple[int, float, float, float]] = []
        rotation_matrix = view_orientation["matrix"]
        for atom_index, centered_x, centered_y, centered_z in centered_atom_positions:
            rotated_x, rotated_y, rotated_z = _matrix_vector_multiply(
                rotation_matrix,
                (centered_x, centered_y, centered_z),
            )
            rotated.append((atom_index, rotated_x, rotated_y, rotated_z))

        pad = 88.0
        usable_w = max(width - (2 * pad), 140.0)
        usable_h = max(height - (2 * pad), 140.0)
        scale = (min(usable_w, usable_h) / max(2.55 * molecule_radius, 1.0)) * float(view_state["zoom"])
        camera_distance = max(8.0, molecule_radius * 5.0)

        projected: dict[int, tuple[float, float, float, float]] = {}
        for atom_index, rotated_x, rotated_y, rotated_z in rotated:
            perspective = camera_distance / max(camera_distance - rotated_z, camera_distance * 0.35)
            projected_x = (width / 2.0) + float(view_state["pan_x"]) + (rotated_x * scale * perspective)
            projected_y = (height / 2.0) + float(view_state["pan_y"]) - (rotated_y * scale * perspective)
            projected[atom_index] = (projected_x, projected_y, rotated_z, perspective)
        return projected

    def _find_atom_near(canvas_x: float, canvas_y: float, *, radius: float = 18.0) -> int | None:
        best_index: int | None = None
        best_distance = radius * radius
        for atom_index, (projected_x, projected_y, _depth, perspective) in projected_positions.items():
            distance = ((projected_x - canvas_x) ** 2) + ((projected_y - canvas_y) ** 2)
            if distance <= best_distance + (4.0 * perspective):
                best_distance = distance
                best_index = atom_index
        return best_index

    def _clear_selection_box() -> None:
        rect_id = left_drag_state.get("rect_id")
        if rect_id:
            canvas.delete(int(rect_id))
        left_drag_state["rect_id"] = None
        left_drag_state["box_active"] = False

    def _interactive_canvas_render() -> bool:
        return _should_use_low_detail_canvas_render(
            is_rotating=(
                (rotate_state.get("x") is not None and rotate_state.get("y") is not None)
                or (pan_state.get("x") is not None and pan_state.get("y") is not None)
            ),
            box_select_active=bool(left_drag_state.get("box_active")),
        )

    def _flush_canvas_redraw() -> None:
        render_state["after_id"] = None
        _redraw_canvas()

    def _schedule_canvas_redraw(*, immediate: bool = False) -> None:
        pending_after = render_state.get("after_id")
        if immediate:
            if pending_after is not None:
                root.after_cancel(pending_after)
                render_state["after_id"] = None
            _redraw_canvas()
            return
        if pending_after is not None:
            return
        delay = _TK_INTERACTIVE_REDRAW_MS if _interactive_canvas_render() else 0
        render_state["after_id"] = root.after(delay, _flush_canvas_redraw)

    def _redraw_canvas(*_args) -> None:
        nonlocal projected_positions
        canvas.delete("all")
        render_state["last_draw_at"] = time.perf_counter()
        render_state["low_detail"] = _interactive_canvas_render()
        low_detail = bool(render_state["low_detail"])
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 500)
        projected_positions = _project_atoms(width, height)
        atom_lookup = {
            int(atom["index"]): atom
            for atom in state["group_constraints"]["atoms"]
        }
        depth_values = [coords[2] for coords in projected_positions.values()]
        min_depth = min(depth_values, default=0.0)
        max_depth = max(depth_values, default=1.0)
        depth_span = max(max_depth - min_depth, 1.0)
        bonds_to_draw: list[tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]] = []
        for bond in state["molecule"].get("bonds") or []:
            first = projected_positions.get(int(bond["first"]))
            second = projected_positions.get(int(bond["second"]))
            if first and second:
                bonds_to_draw.append((((first[2] + second[2]) / 2.0), first, second))
        for average_depth, first, second in sorted(bonds_to_draw, key=lambda item: item[0]):
            depth_fraction = (average_depth - min_depth) / depth_span
            bond_color = _blend_hex("#334155", "#cbd5e1", 0.35 + (0.35 * depth_fraction))
            canvas.create_line(
                first[0],
                first[1],
                second[0],
                second[1],
                fill="#64748b" if low_detail else bond_color,
                width=max(1.0, (1.2 if low_detail else 2.0) * ((first[3] + second[3]) / 2.0)),
            )
        for atom_index, x, y, _depth, perspective in sorted(
            ((index, *coords) for index, coords in projected_positions.items()),
            key=lambda item: item[3],
        ):
            atom = atom_lookup[atom_index]
            group_id = atom.get("group_id")
            atom_element = str(atom.get("element") or atom.get("name") or atom_index)
            is_metal = _is_metal_element(atom_element)
            element_fill = _element_fill_color(atom_element)
            outline = _group_color_hex(group_id) if group_id is not None else "#cbd5e1"
            depth_light = max(0.0, min(1.0, (_depth - min_depth) / depth_span))
            shaded_fill = _blend_hex(element_fill, "#ffffff", 0.08 + (0.18 * depth_light))
            shaded_fill = _blend_hex(shaded_fill, "#020617", max(0.0, 0.10 - (0.06 * depth_light)))
            atom_radius = (11 if low_detail else 13) * perspective * (2.4 if is_metal else 1.0)
            selection_radius = atom_radius + (11 if not low_detail else 7)
            if atom_index in selected_atoms:
                highlight_color = "#fbbf24"
                highlight_fill = "#fde68a"
                halo_radius = selection_radius + (7 if not low_detail else 3)
                canvas.create_oval(
                    x - halo_radius,
                    y - halo_radius,
                    x + halo_radius,
                    y + halo_radius,
                    fill=highlight_fill,
                    outline=highlight_color,
                    width=2 if low_detail else 3,
                )
                canvas.create_oval(
                    x - selection_radius,
                    y - selection_radius,
                    x + selection_radius,
                    y + selection_radius,
                    outline=highlight_color,
                    width=3 if low_detail else 4,
                )
            canvas.create_oval(
                x - atom_radius,
                y - atom_radius,
                x + atom_radius,
                y + atom_radius,
                fill=shaded_fill,
                outline=outline,
                width=2 if low_detail else (3 if group_id is not None else 2),
            )
            if not low_detail:
                canvas.create_text(
                    x,
                    y,
                    text=atom_element if is_metal else str(atom_index),
                    fill=_text_color_for_fill(shaded_fill),
                    font=("TkDefaultFont", 12 if is_metal else 9, "bold"),
                )
                detail_text = f"{atom['name']} ({atom_index})" if is_metal else str(atom["name"])
                canvas.create_text(
                    x,
                    y + atom_radius + 10,
                    text=detail_text,
                    fill="#e2e8f0",
                    font=("TkDefaultFont", 8, "bold" if is_metal else "normal"),
                )
            badge_text = _group_badge_text(group_id)
            if badge_text and not low_detail:
                badge_fill = _group_color_hex(group_id)
                badge_width = max(24, (len(badge_text) * 7) + 8)
                badge_left = x + 10
                badge_top = y - 24
                canvas.create_rectangle(
                    badge_left,
                    badge_top,
                    badge_left + badge_width,
                    badge_top + 16,
                    fill=badge_fill,
                    outline="#020617",
                    width=1,
                )
                canvas.create_text(
                    badge_left + (badge_width / 2),
                    badge_top + 8,
                    text=badge_text,
                    fill=_text_color_for_fill(badge_fill),
                    font=("TkDefaultFont", 8, "bold"),
                )
        if left_drag_state.get("box_active") and left_drag_state.get("start") is not None:
            start_x, start_y = left_drag_state["start"]
            rect_id = canvas.create_rectangle(
                start_x,
                start_y,
                start_x,
                start_y,
                outline="#2563eb",
                dash=(4, 2),
                width=2,
            )
            left_drag_state["rect_id"] = rect_id

    def _refresh_all() -> None:
        state["qm_settings"] = _current_qm_settings()
        _store_current_auto_group_mode()
        _refresh_auto_group_mode()
        _refresh_atom_tree()
        _refresh_legend()
        _refresh_selection_label()
        _schedule_canvas_redraw(immediate=True)

    def _apply_qm_settings() -> None:
        qm = _normalized_editor_qm_settings(state.get("qm_settings"))
        state["qm_settings"] = qm
        resp_same_as_dft, resp_functional, resp_basis = _resp_settings_from_qm(qm)
        memory_mb, grid_key, maxiter, net_charge, multiplicity = _resource_settings_from_qm(qm)
        geometry_var.set(_geometry_label_for(_geometry_mode_from_qm(qm)))
        xtb_acc_var.set(_xtb_acc_from_qm(qm))
        dft_functional_var.set(_functional_label_for(_dft_functional_from_qm(qm)))
        dft_basis_var.set(_basis_label_for(_dft_basis_from_qm(qm)))
        resp_same_var.set(resp_same_as_dft)
        resp_functional_var.set(_functional_label_for(resp_functional))
        resp_basis_var.set(_basis_label_for(resp_basis))
        grid_var.set(_grid_label_for(grid_key))
        memory_var.set(memory_mb)
        maxiter_var.set(maxiter)
        charge_var.set(net_charge)
        multiplicity_var.set(multiplicity)
        _refresh_auto_group_mode()
        _sync_qm_ui_state()

    def on_auto_group_mode_change(_event=None) -> None:
        reset_auto()

    def assign_new_group() -> None:
        if not selected_atoms:
            return
        group_id = _next_group_id()
        for atom in state["group_constraints"]["atoms"]:
            if int(atom["index"]) in selected_atoms:
                atom["group_id"] = group_id
        _rebuild_groups()
        _refresh_all()

    def clear_selected_groups() -> None:
        if not selected_atoms:
            return
        for atom in state["group_constraints"]["atoms"]:
            if int(atom["index"]) in selected_atoms:
                atom["group_id"] = None
        _rebuild_groups()
        _refresh_all()

    def reset_auto() -> None:
        nonlocal state, selected_atoms
        state["group_constraints"] = suggest_group_constraints(
            loadable_molecule,
            auto_group_mode=_current_auto_group_mode(),
            auto_group_graph_method=_current_auto_group_graph_method(),
        )
        selected_atoms = set()
        view_orientation["matrix"] = _identity_rotation_matrix()
        view_state["zoom"] = 1.0
        view_state["pan_x"] = 0.0
        view_state["pan_y"] = 0.0
        _refresh_all()

    def on_tree_select(_event=None) -> None:
        nonlocal selected_atoms
        if suppress_tree_callback:
            return
        selected_atoms = {int(item_id) for item_id in atom_tree.selection()}
        _refresh_selection_label()
        _schedule_canvas_redraw(immediate=True)

    def on_left_press(event) -> None:
        left_drag_state["start"] = (event.x, event.y)
        left_drag_state["box_active"] = False
        canvas.focus_set()
        _clear_selection_box()

    def on_left_drag(event) -> None:
        start = left_drag_state.get("start")
        if start is None:
            return
        if not left_drag_state.get("box_active"):
            if max(abs(event.x - start[0]), abs(event.y - start[1])) < 6:
                return
            left_drag_state["box_active"] = True
            left_drag_state["rect_id"] = canvas.create_rectangle(
                start[0],
                start[1],
                event.x,
                event.y,
                outline="#2563eb",
                dash=(4, 2),
                width=2,
            )
            return
        rect_id = left_drag_state.get("rect_id")
        if rect_id:
            canvas.coords(int(rect_id), start[0], start[1], event.x, event.y)

    def on_left_release(event) -> None:
        nonlocal selected_atoms
        start = left_drag_state.get("start")
        if start is None:
            return
        append_selection = bool(event.state & 0x0004)
        if left_drag_state.get("box_active"):
            x0, y0 = start
            x_min, x_max = sorted((x0, event.x))
            y_min, y_max = sorted((y0, event.y))
            boxed = {
                atom_index
                for atom_index, (projected_x, projected_y, _depth, _perspective) in projected_positions.items()
                if x_min <= projected_x <= x_max and y_min <= projected_y <= y_max
            }
            selected_atoms = (selected_atoms | boxed) if append_selection else boxed
        else:
            atom_index = _find_atom_near(event.x, event.y)
            if atom_index is None:
                selected_atoms = set()
            elif append_selection:
                if atom_index in selected_atoms:
                    selected_atoms.remove(atom_index)
                else:
                    selected_atoms.add(atom_index)
            else:
                selected_atoms = {atom_index}
        left_drag_state["start"] = None
        _clear_selection_box()
        _sync_tree_selection()
        _schedule_canvas_redraw(immediate=True)

    def on_right_press(event) -> None:
        rotate_state["x"] = float(event.x)
        rotate_state["y"] = float(event.y)
        canvas.focus_set()
        _schedule_canvas_redraw()

    def on_right_drag(event) -> None:
        last_x = rotate_state.get("x")
        last_y = rotate_state.get("y")
        if last_x is None or last_y is None:
            return
        delta_x = float(event.x) - float(last_x)
        delta_y = float(event.y) - float(last_y)
        rotation_step = 0.009
        yaw_matrix = _axis_angle_rotation_matrix((0.0, 1.0, 0.0), delta_x * rotation_step)
        pitch_matrix = _axis_angle_rotation_matrix((1.0, 0.0, 0.0), delta_y * rotation_step)
        view_orientation["matrix"] = _matrix_multiply(
            pitch_matrix,
            _matrix_multiply(yaw_matrix, view_orientation["matrix"]),
        )
        rotate_state["x"] = float(event.x)
        rotate_state["y"] = float(event.y)
        _schedule_canvas_redraw()

    def on_right_release(_event=None) -> None:
        rotate_state["x"] = None
        rotate_state["y"] = None
        _schedule_canvas_redraw(immediate=True)

    def on_middle_press(event) -> None:
        pan_state["x"] = float(event.x)
        pan_state["y"] = float(event.y)
        canvas.focus_set()
        _schedule_canvas_redraw()

    def on_middle_drag(event) -> None:
        last_x = pan_state.get("x")
        last_y = pan_state.get("y")
        if last_x is None or last_y is None:
            return
        view_state["pan_x"] += float(event.x) - float(last_x)
        view_state["pan_y"] += float(event.y) - float(last_y)
        pan_state["x"] = float(event.x)
        pan_state["y"] = float(event.y)
        _schedule_canvas_redraw()

    def on_middle_release(_event=None) -> None:
        pan_state["x"] = None
        pan_state["y"] = None
        _schedule_canvas_redraw(immediate=True)

    def _zoom_canvas(event, factor: float) -> None:
        old_zoom = float(view_state["zoom"])
        new_zoom = max(0.25, min(5.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 500)
        cursor_dx = float(event.x) - (width / 2.0)
        cursor_dy = float(event.y) - (height / 2.0)
        zoom_factor = new_zoom / old_zoom
        view_state["pan_x"] = cursor_dx - ((cursor_dx - float(view_state["pan_x"])) * zoom_factor)
        view_state["pan_y"] = cursor_dy - ((cursor_dy - float(view_state["pan_y"])) * zoom_factor)
        view_state["zoom"] = new_zoom
        _schedule_canvas_redraw()

    def on_mouse_wheel(event) -> str:
        delta = float(getattr(event, "delta", 0) or 0)
        factor = 1.12 if delta > 0 else 1 / 1.12
        _zoom_canvas(event, factor)
        return "break"

    def on_linux_wheel_up(event) -> str:
        _zoom_canvas(event, 1.12)
        return "break"

    def on_linux_wheel_down(event) -> str:
        _zoom_canvas(event, 1 / 1.12)
        return "break"

    def reset_view(_event=None) -> str:
        view_orientation["matrix"] = _identity_rotation_matrix()
        view_state["zoom"] = 1.0
        view_state["pan_x"] = 0.0
        view_state["pan_y"] = 0.0
        _schedule_canvas_redraw(immediate=True)
        return "break"

    def on_cancel() -> None:
        nonlocal result_payload
        result_payload = dict(session_state)
        result_payload["edited_in_popup"] = False
        result_payload["editor_mode"] = "cancelled"
        root.quit()
        root.destroy()

    def on_save() -> None:
        nonlocal result_payload, state
        state["qm_settings"] = _current_qm_settings()
        _store_current_auto_group_mode()
        result_payload = dict(state)
        result_payload["edited_in_popup"] = True
        result_payload["editor_mode"] = "tk_popup"
        root.quit()
        root.destroy()

    atom_tree.bind("<<TreeviewSelect>>", on_tree_select)
    group_tree.bind("<<TreeviewSelect>>", on_group_select)
    canvas.bind("<ButtonPress-1>", on_left_press)
    canvas.bind("<B1-Motion>", on_left_drag)
    canvas.bind("<ButtonRelease-1>", on_left_release)
    canvas.bind("<ButtonPress-3>", on_right_press)
    canvas.bind("<B3-Motion>", on_right_drag)
    canvas.bind("<ButtonRelease-3>", on_right_release)
    canvas.bind("<ButtonPress-2>", on_middle_press)
    canvas.bind("<B2-Motion>", on_middle_drag)
    canvas.bind("<ButtonRelease-2>", on_middle_release)
    canvas.bind("<MouseWheel>", on_mouse_wheel)
    canvas.bind("<Button-4>", on_linux_wheel_up)
    canvas.bind("<Button-5>", on_linux_wheel_down)
    canvas.bind("<Enter>", lambda _event: canvas.focus_set())
    canvas.bind("<KeyPress-equal>", reset_view)
    canvas.bind("<KeyPress-KP_Equal>", reset_view)
    canvas.bind("<KeyPress-plus>", reset_view)
    root.bind("<KeyPress-equal>", reset_view)
    root.bind("<KeyPress-KP_Equal>", reset_view)
    root.bind("<KeyPress-plus>", reset_view)
    canvas.bind("<Configure>", lambda _event: _schedule_canvas_redraw())
    geometry_var.trace_add("write", _sync_qm_ui_state)
    resp_same_var.trace_add("write", _sync_qm_ui_state)
    dft_functional_var.trace_add("write", _sync_qm_ui_state)
    dft_basis_var.trace_add("write", _sync_qm_ui_state)
    auto_group_combo.bind("<<ComboboxSelected>>", on_auto_group_mode_change)
    auto_group_graph_method_combo.bind("<<ComboboxSelected>>", on_auto_group_mode_change)
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    _apply_qm_settings()
    _refresh_all()
    root.after(50, lambda: _schedule_canvas_redraw(immediate=True))
    root.mainloop()

    if result_payload is None:
        result_payload = dict(session_state)
        result_payload["edited_in_popup"] = False
        result_payload["editor_mode"] = "cancelled"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return result_payload


def _auto_default_payload(session_state: dict[str, Any], warning: str | None) -> dict[str, Any]:
    payload = dict(session_state)
    payload["edited_in_popup"] = False
    payload["editor_mode"] = "auto_defaults"
    payload["editor_warning"] = warning
    return payload


def launch_resp_editor(
    *,
    session_state: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if _prefer_tk_popup():
        tk_payload = _launch_tk_editor(session_state=session_state, output_dir=output_dir)
        if tk_payload is not None:
            return tk_payload

    can_launch, warning = resp_editor_launch_status()
    if not can_launch:
        tk_payload = _launch_tk_editor(session_state=session_state, output_dir=output_dir, warning=warning)
        if tk_payload is not None:
            return tk_payload
        return _auto_default_payload(session_state, warning)

    try:
        from PySide6.QtCore import QObject, QSize, Qt, QUrl, Signal, Slot
        from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizeGrip,
            QSpinBox,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
            QWidget,
        )
        from PySide6.QtWebChannel import QWebChannel
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
        except Exception:
            QWebEngineView = None
    except ModuleNotFoundError:
        warning = "PySide6 is not installed, so the Qt RESP popup could not be launched."
        tk_payload = _launch_tk_editor(session_state=session_state, output_dir=output_dir, warning=warning)
        if tk_payload is not None:
            return tk_payload
        return _auto_default_payload(session_state, warning)
    except Exception as exc:
        warning = f"RESP popup initialization failed: {exc}"
        tk_payload = _launch_tk_editor(session_state=session_state, output_dir=output_dir, warning=warning)
        if tk_payload is not None:
            return tk_payload
        return _auto_default_payload(session_state, warning)

    assets_dir = Path(__file__).resolve().parent / "assets"

    class _Bridge(QObject):
        stateChanged = Signal(str)

        def __init__(self, dialog) -> None:
            super().__init__()
            self._dialog = dialog

        @Slot(result=str)
        def getStateJson(self) -> str:
            return json.dumps(self._dialog.current_state())

        @Slot(int, bool)
        def atomPicked(self, atom_index: int, additive: bool = False) -> None:
            self._dialog.set_atom_selection(atom_index, additive=additive)

        @Slot()
        def clearSelection(self) -> None:
            self._dialog.clear_atom_selection()

        @Slot(int)
        def groupPicked(self, group_id: int) -> None:
            self._dialog.select_group(group_id)

    class _NativePreview(QWidget):
        atomPicked = Signal(int, bool)
        emptyPicked = Signal()

        def __init__(self, initial_state: dict[str, Any]) -> None:
            super().__init__()
            self._state = json.loads(json.dumps(initial_state))
            self._last_positions: list[dict[str, Any]] = []
            self._projected_positions: dict[int, tuple[float, float, float, float]] = {}
            self._centered_atom_positions: list[tuple[int, float, float, float]] = []
            self._molecule_radius = 1.0
            self._view_orientation = {"matrix": _identity_rotation_matrix()}
            self._view_state = {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0}
            self._rotate_last: tuple[float, float] | None = None
            self._pan_last: tuple[float, float] | None = None
            self._left_press: tuple[float, float] | None = None
            self._left_press_moved = False
            self.setMinimumSize(320, 320)
            self.setFocusPolicy(Qt.StrongFocus)
            self._rebuild_geometry_cache()

        def sizeHint(self) -> QSize:
            return QSize(720, 720)

        def set_state(self, next_state: dict[str, Any]) -> None:
            previous_molecule = json.dumps(self._state.get("molecule") or {}, sort_keys=True)
            self._state = json.loads(json.dumps(next_state))
            next_molecule = json.dumps(self._state.get("molecule") or {}, sort_keys=True)
            if previous_molecule != next_molecule:
                self._rebuild_geometry_cache()
            self.update()

        def reset_view(self) -> None:
            self._view_orientation["matrix"] = _identity_rotation_matrix()
            self._view_state["zoom"] = 1.0
            self._view_state["pan_x"] = 0.0
            self._view_state["pan_y"] = 0.0
            self.update()

        def _rebuild_geometry_cache(self) -> None:
            molecule = (self._state.get("molecule") or {}) if isinstance(self._state, dict) else {}
            atoms = list(molecule.get("atoms") or [])
            if not atoms:
                self._centered_atom_positions = []
                self._molecule_radius = 1.0
                return

            center_x = sum(float(atom.get("x") or 0.0) for atom in atoms) / len(atoms)
            center_y = sum(float(atom.get("y") or 0.0) for atom in atoms) / len(atoms)
            center_z = sum(float(atom.get("z") or 0.0) for atom in atoms) / len(atoms)
            self._centered_atom_positions = [
                (
                    int(atom["index"]),
                    float(atom.get("x") or 0.0) - center_x,
                    float(atom.get("y") or 0.0) - center_y,
                    float(atom.get("z") or 0.0) - center_z,
                )
                for atom in atoms
            ]
            self._molecule_radius = max(
                (
                    math.sqrt((centered_x ** 2) + (centered_y ** 2) + (centered_z ** 2))
                    for _atom_index, centered_x, centered_y, centered_z in self._centered_atom_positions
                ),
                default=1.0,
            )
            self._molecule_radius = max(self._molecule_radius, 1.0)

        def _project_atoms(self) -> dict[int, tuple[float, float, float, float]]:
            if not self._centered_atom_positions:
                return {}

            rotated: list[tuple[int, float, float, float]] = []
            rotation_matrix = self._view_orientation["matrix"]
            for atom_index, centered_x, centered_y, centered_z in self._centered_atom_positions:
                rotated_x, rotated_y, rotated_z = _matrix_vector_multiply(
                    rotation_matrix,
                    (centered_x, centered_y, centered_z),
                )
                rotated.append((atom_index, rotated_x, rotated_y, rotated_z))

            width = max(float(self.width()), 360.0)
            height = max(float(self.height()), 360.0)
            pad = 88.0
            usable_w = max(width - (2 * pad), 140.0)
            usable_h = max(height - (2 * pad), 140.0)
            scale = (min(usable_w, usable_h) / max(2.55 * self._molecule_radius, 1.0)) * float(self._view_state["zoom"])
            camera_distance = max(8.0, self._molecule_radius * 5.0)

            projected: dict[int, tuple[float, float, float, float]] = {}
            for atom_index, rotated_x, rotated_y, rotated_z in rotated:
                perspective = camera_distance / max(camera_distance - rotated_z, camera_distance * 0.35)
                projected_x = (width / 2.0) + float(self._view_state["pan_x"]) + (rotated_x * scale * perspective)
                projected_y = (height / 2.0) + float(self._view_state["pan_y"]) - (rotated_y * scale * perspective)
                projected[atom_index] = (projected_x, projected_y, rotated_z, perspective)
            return projected

        def _find_atom_near(self, click_x: float, click_y: float, *, radius: float = 18.0) -> int | None:
            best_index: int | None = None
            best_distance = radius * radius
            for atom_index, (projected_x, projected_y, _depth, perspective) in self._projected_positions.items():
                distance = ((projected_x - click_x) ** 2) + ((projected_y - click_y) ** 2)
                if distance <= best_distance + (4.0 * perspective):
                    best_distance = distance
                    best_index = atom_index
            return best_index

        def _zoom_view(self, cursor_x: float, cursor_y: float, factor: float) -> None:
            old_zoom = float(self._view_state["zoom"])
            new_zoom = max(0.25, min(5.0, old_zoom * factor))
            if abs(new_zoom - old_zoom) < 1e-6:
                return
            width = max(float(self.width()), 360.0)
            height = max(float(self.height()), 360.0)
            cursor_dx = float(cursor_x) - (width / 2.0)
            cursor_dy = float(cursor_y) - (height / 2.0)
            zoom_factor = new_zoom / old_zoom
            self._view_state["pan_x"] = cursor_dx - ((cursor_dx - float(self._view_state["pan_x"])) * zoom_factor)
            self._view_state["pan_y"] = cursor_dy - ((cursor_dy - float(self._view_state["pan_y"])) * zoom_factor)
            self._view_state["zoom"] = new_zoom
            self.update()

        def _positioned_atoms(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
            molecule = (self._state.get("molecule") or {}) if isinstance(self._state, dict) else {}
            atoms = list(molecule.get("atoms") or [])
            if not atoms:
                return [], {}

            group_lookup = {
                int(atom["index"]): atom.get("group_id")
                for atom in (((self._state.get("group_constraints") or {}).get("atoms") or []))
            }
            selected_atoms = {int(value) for value in (self._state.get("selected_atom_indices") or [])}

            self._projected_positions = self._project_atoms()

            positioned: list[dict[str, Any]] = []
            by_index: dict[int, dict[str, Any]] = {}
            for atom in atoms:
                atom_index = int(atom["index"])
                group_id = group_lookup.get(atom_index)
                element = str(atom.get("element") or atom.get("name") or atom_index)
                is_metal = _is_metal_element(element)
                fill_hex = _group_color_hex(int(group_id)) if group_id is not None else _element_fill_color(element)
                selected = atom_index in selected_atoms
                projected_x, projected_y, depth, perspective = self._projected_positions.get(
                    atom_index,
                    (float(self.width()) / 2.0, float(self.height()) / 2.0, 0.0, 1.0),
                )
                entry = {
                    "index": atom_index,
                    "element": element,
                    "name": str(atom.get("name") or atom.get("element") or atom_index),
                    "is_metal": is_metal,
                    "group_id": int(group_id) if group_id is not None else None,
                    "selected": selected,
                    "fill_hex": fill_hex,
                    "text_hex": _text_color_for_fill(fill_hex),
                    "radius": (13.0 if selected else 11.0) * perspective * (2.4 if is_metal else 1.0),
                    "x": projected_x,
                    "y": projected_y,
                    "z": depth,
                    "perspective": perspective,
                }
                positioned.append(entry)
                by_index[atom_index] = entry
            return positioned, by_index

        def mousePressEvent(self, event) -> None:
            self.setFocus()
            if event.button() == Qt.RightButton:
                self._rotate_last = (float(event.position().x()), float(event.position().y()))
                event.accept()
                return
            if event.button() == Qt.MiddleButton:
                self._pan_last = (float(event.position().x()), float(event.position().y()))
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                self._left_press = (float(event.position().x()), float(event.position().y()))
                self._left_press_moved = False
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:
            if self._rotate_last is not None:
                delta_x = float(event.position().x()) - float(self._rotate_last[0])
                delta_y = float(event.position().y()) - float(self._rotate_last[1])
                rotation_step = 0.009
                yaw_matrix = _axis_angle_rotation_matrix((0.0, 1.0, 0.0), delta_x * rotation_step)
                pitch_matrix = _axis_angle_rotation_matrix((1.0, 0.0, 0.0), delta_y * rotation_step)
                self._view_orientation["matrix"] = _matrix_multiply(
                    pitch_matrix,
                    _matrix_multiply(yaw_matrix, self._view_orientation["matrix"]),
                )
                self._rotate_last = (float(event.position().x()), float(event.position().y()))
                self.update()
                event.accept()
                return
            if self._pan_last is not None:
                self._view_state["pan_x"] += float(event.position().x()) - float(self._pan_last[0])
                self._view_state["pan_y"] += float(event.position().y()) - float(self._pan_last[1])
                self._pan_last = (float(event.position().x()), float(event.position().y()))
                self.update()
                event.accept()
                return
            if self._left_press is not None:
                if max(
                    abs(float(event.position().x()) - float(self._left_press[0])),
                    abs(float(event.position().y()) - float(self._left_press[1])),
                ) >= 6.0:
                    self._left_press_moved = True
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == Qt.RightButton:
                self._rotate_last = None
                event.accept()
                return
            if event.button() == Qt.MiddleButton:
                self._pan_last = None
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                try_pick = self._left_press is not None and not self._left_press_moved
                self._left_press = None
                self._left_press_moved = False
                if try_pick:
                    atom_index = self._find_atom_near(float(event.position().x()), float(event.position().y()))
                    if atom_index is not None:
                        additive = bool(event.modifiers() & Qt.ControlModifier)
                        self.atomPicked.emit(int(atom_index), additive)
                        event.accept()
                        return
                    self.emptyPicked.emit()
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def wheelEvent(self, event) -> None:
            delta = float(event.angleDelta().y())
            if abs(delta) < 1.0:
                super().wheelEvent(event)
                return
            self._zoom_view(
                float(event.position().x()),
                float(event.position().y()),
                1.12 if delta > 0 else (1 / 1.12),
            )
            event.accept()

        def keyPressEvent(self, event) -> None:
            if event.key() in {Qt.Key_Equal, Qt.Key_Plus}:
                self.reset_view()
                event.accept()
                return
            super().keyPressEvent(event)

        def paintEvent(self, _event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)

            gradient = QLinearGradient(0, 0, self.width(), self.height())
            gradient.setColorAt(0.0, QColor("#050814"))
            gradient.setColorAt(0.45, QColor("#0f172a"))
            gradient.setColorAt(1.0, QColor("#111827"))
            painter.fillRect(self.rect(), gradient)

            frame_pen = QPen(QColor("#334155"))
            frame_pen.setWidth(1)
            painter.setPen(frame_pen)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 22, 22)

            molecule = (self._state.get("molecule") or {}) if isinstance(self._state, dict) else {}
            atoms = list(molecule.get("atoms") or [])
            bonds = list(molecule.get("bonds") or [])
            if not atoms:
                painter.setPen(QColor("#cbd5e1"))
                painter.drawText(self.rect(), Qt.AlignCenter, "No atom coordinates were available for the molecule preview.")
                self._last_positions = []
                self._projected_positions = {}
                return

            positioned_atoms, by_index = self._positioned_atoms()
            self._last_positions = positioned_atoms
            depth_values = [coords[2] for coords in self._projected_positions.values()]
            min_depth = min(depth_values, default=0.0)
            max_depth = max(depth_values, default=1.0)
            depth_span = max(max_depth - min_depth, 1.0)

            bonds_to_draw: list[tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]] = []
            for bond in bonds:
                first = self._projected_positions.get(int(bond.get("first") or 0))
                second = self._projected_positions.get(int(bond.get("second") or 0))
                if first is not None and second is not None:
                    bonds_to_draw.append((((first[2] + second[2]) / 2.0), first, second))

            for average_depth, first, second in sorted(bonds_to_draw, key=lambda item: item[0]):
                depth_fraction = (average_depth - min_depth) / depth_span
                bond_pen = QPen(QColor(_blend_hex("#334155", "#cbd5e1", 0.35 + (0.35 * depth_fraction))))
                bond_pen.setWidthF(max(1.4, 2.0 * ((first[3] + second[3]) / 2.0)))
                bond_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(bond_pen)
                painter.drawLine(
                    int(round(float(first[0]))),
                    int(round(float(first[1]))),
                    int(round(float(second[0]))),
                    int(round(float(second[1]))),
                )

            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)

            for entry in sorted(positioned_atoms, key=lambda item: item["z"]):
                center_x = float(entry["x"])
                center_y = float(entry["y"])
                radius = float(entry["radius"])
                depth_light = max(0.0, min(1.0, (float(entry["z"]) - min_depth) / depth_span))
                shaded_fill_hex = _blend_hex(_element_fill_color(entry["element"]), "#ffffff", 0.08 + (0.18 * depth_light))
                shaded_fill_hex = _blend_hex(shaded_fill_hex, "#020617", max(0.0, 0.10 - (0.06 * depth_light)))
                fill_color = QColor(shaded_fill_hex)
                text_color = _text_color_for_fill(shaded_fill_hex)

                if bool(entry["selected"]):
                    halo_color = QColor("#fbbf24")
                    halo_color.setAlphaF(0.40)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(halo_color)
                    halo_radius = radius + 9.0
                    painter.drawEllipse(
                        int(round(center_x - halo_radius)),
                        int(round(center_y - halo_radius)),
                        int(round(halo_radius * 2.0)),
                        int(round(halo_radius * 2.0)),
                    )

                atom_pen = QPen(
                    QColor("#fbbf24") if bool(entry["selected"]) else (
                        QColor(str(entry["fill_hex"])) if entry["group_id"] is not None else QColor("#cbd5e1")
                    )
                )
                atom_pen.setWidthF(3.4 if bool(entry["selected"]) else (2.0 if entry["group_id"] is not None else 1.2))
                painter.setPen(atom_pen)
                painter.setBrush(fill_color)
                painter.drawEllipse(
                    int(round(center_x - radius)),
                    int(round(center_y - radius)),
                    int(round(radius * 2.0)),
                    int(round(radius * 2.0)),
                )

                painter.setPen(QColor(text_color))
                if bool(entry.get("is_metal")):
                    metal_font = painter.font()
                    metal_font.setPointSize(12)
                    metal_font.setBold(True)
                    painter.setFont(metal_font)
                painter.drawText(
                    int(round(center_x - radius)),
                    int(round(center_y - 9.0)),
                    int(round(radius * 2.0)),
                    18,
                    Qt.AlignCenter,
                    str(entry["element"]),
                )
                if bool(entry.get("is_metal")):
                    painter.setFont(font)
                    painter.setPen(QColor("#e2e8f0"))
                    painter.drawText(
                        int(round(center_x - max(30.0, radius))),
                        int(round(center_y + radius + 4.0)),
                        int(round(max(60.0, radius * 2.0))),
                        18,
                        Qt.AlignCenter,
                        f"{entry['name']} ({entry['index']})",
                    )

            painter.setPen(QColor("#cbd5e1"))
            footer_rect = self.rect().adjusted(18, self.height() - 42, -18, -12)
            painter.drawText(
                footer_rect,
                Qt.AlignCenter,
                f"{self._state.get('residue_name') or 'LIG'} projected in 2D for this Qt session. Atom colors reflect RESP equality groups.",
            )

    class _Dialog(QDialog):
        def __init__(self, initial_state: dict[str, Any]) -> None:
            super().__init__()
            self.setWindowTitle("RESP Editor")
            self.resize(1480, 940)
            self.setMinimumSize(1120, 720)
            self.setSizeGripEnabled(True)
            self._initial_state = _normalized_editor_state(initial_state)
            self._state = json.loads(json.dumps(self._initial_state))
            self._selected_atoms: set[int] = set()
            self._molecule = _molecule_from_editor_state(self._initial_state)
            self.bridge = None
            self.native_viewer: _NativePreview | None = None
            self._syncing_qm_controls = False
            self._last_emitted_state_json: str | None = None

            root = QHBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            splitter = QSplitter(Qt.Horizontal)
            root.addWidget(splitter)
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(10)
            splitter.setStyleSheet(
                "QSplitter::handle { background:#d7e1ec; } "
                "QSplitter::handle:hover { background:#93c5fd; }"
            )

            controls = QWidget()
            controls_layout = QVBoxLayout(controls)
            controls_layout.setContentsMargins(12, 12, 12, 12)
            controls_layout.setSpacing(8)
            splitter.addWidget(controls)

            title = QLabel("RESP charge-group editor")
            title.setStyleSheet("font-size: 18px; font-weight: 600;")
            controls_layout.addWidget(title)
            review_card = QLabel(
                "<div style='line-height:1.35;'>"
                "<span style='font-size:14px; font-weight:700; color:#1e3a8a;'>Auto groups loaded.</span> "
                "<span style='color:#0f172a;'>Review the H-only and symmetry suggestions before saving.</span>"
                "</div>"
            )
            review_card.setWordWrap(True)
            review_card.setTextFormat(Qt.RichText)
            review_card.setMaximumHeight(64)
            review_card.setStyleSheet(
                "background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:10px 12px;"
            )
            controls_layout.addWidget(review_card)

            control_hint = QLabel("Right-drag rotates, the mouse wheel zooms, middle-drag pans, and '=' resets the view.")
            control_hint.setWordWrap(True)
            control_hint.setStyleSheet("color:#475569; font-size:12px;")
            controls_layout.addWidget(control_hint)

            self.atom_table = QTableWidget()
            self.atom_table.setColumnCount(4)
            self.atom_table.setHorizontalHeaderLabels(["Index", "Name", "Element", "Group"])
            self.atom_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.atom_table.setSelectionMode(QTableWidget.MultiSelection)
            self.atom_table.setMinimumHeight(220)
            self.atom_table.verticalHeader().setVisible(False)
            self.atom_table.horizontalHeader().setStretchLastSection(True)
            self.atom_table.itemSelectionChanged.connect(self._sync_selection_from_table)
            controls_layout.addWidget(self.atom_table, stretch=2)

            button_row = QHBoxLayout()
            self.new_group_button = QPushButton("New Group")
            self.new_group_button.clicked.connect(self.assign_new_group)
            button_row.addWidget(self.new_group_button)
            self.clear_group_button = QPushButton("Clear Group")
            self.clear_group_button.clicked.connect(self.clear_selected_groups)
            button_row.addWidget(self.clear_group_button)
            self.reset_button = QPushButton("Apply Auto")
            self.reset_button.clicked.connect(self.reset_auto)
            button_row.addWidget(self.reset_button)
            self.auto_group_combo = QComboBox()
            for mode_key, mode_label in _AUTO_GROUP_MODE_OPTIONS:
                self.auto_group_combo.addItem(mode_label, mode_key)
            self.auto_group_combo.currentIndexChanged.connect(self._on_auto_group_mode_changed)
            button_row.addWidget(QLabel("Auto groups"))
            button_row.addWidget(self.auto_group_combo)
            controls_layout.addLayout(button_row)

            graph_row = QHBoxLayout()
            self.auto_group_graph_method_combo = QComboBox()
            for method_key, method_label in _AUTO_GROUP_GRAPH_METHOD_OPTIONS:
                self.auto_group_graph_method_combo.addItem(method_label, method_key)
            self.auto_group_graph_method_combo.currentIndexChanged.connect(self._on_auto_group_mode_changed)
            graph_row.addWidget(QLabel("Graph method"))
            graph_row.addWidget(self.auto_group_graph_method_combo)
            controls_layout.addLayout(graph_row)
            self.auto_group_graph_method_note = QLabel("")
            self.auto_group_graph_method_note.setWordWrap(True)
            self.auto_group_graph_method_note.setStyleSheet("color:#475569; font-size:12px;")
            controls_layout.addWidget(self.auto_group_graph_method_note)

            groups_box = QGroupBox("Groups")
            groups_layout = QVBoxLayout(groups_box)
            groups_layout.setContentsMargins(8, 8, 8, 8)
            groups_layout.setSpacing(6)
            self.group_table = QTableWidget()
            self.group_table.setColumnCount(3)
            self.group_table.setHorizontalHeaderLabels(["Group", "Label", "Atoms"])
            self.group_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.group_table.setSelectionMode(QTableWidget.SingleSelection)
            self.group_table.setMinimumHeight(220)
            self.group_table.verticalHeader().setVisible(False)
            self.group_table.horizontalHeader().setStretchLastSection(True)
            self.group_table.itemSelectionChanged.connect(self._select_group_from_table)
            groups_layout.addWidget(self.group_table)
            groups_box.setMinimumHeight(250)
            controls_layout.addWidget(groups_box, stretch=2)
            controls_layout.addStretch(1)

            save_row = QHBoxLayout()
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            save_row.addWidget(cancel_button)
            save_button = QPushButton("Save")
            save_button.clicked.connect(self.save_and_accept)
            save_row.addWidget(save_button)
            save_row.addStretch(1)
            self.size_grip = QSizeGrip(self)
            self.size_grip.setCursor(Qt.SizeFDiagCursor)
            self.size_grip.setToolTip("Drag to resize the window")
            save_row.addWidget(self.size_grip, alignment=Qt.AlignRight | Qt.AlignBottom)
            controls_layout.addLayout(save_row)

            right_splitter = QSplitter(Qt.Horizontal)
            right_splitter.setChildrenCollapsible(False)
            right_splitter.setHandleWidth(10)
            right_splitter.setStyleSheet(
                "QSplitter::handle { background:#d7e1ec; } "
                "QSplitter::handle:hover { background:#93c5fd; }"
            )
            splitter.addWidget(right_splitter)

            preview_host = QWidget()
            preview_layout = QVBoxLayout(preview_host)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            preview_layout.setSpacing(10)
            viewer_note = QLabel(
                "Right-drag rotates, the mouse wheel zooms, middle-drag pans, and '=' resets the view."
            )
            viewer_note.setWordWrap(True)
            viewer_note.setStyleSheet(
                "background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:8px 12px; color:#1e3a8a; font-size:12px;"
            )
            preview_layout.addWidget(viewer_note)

            use_native_linux_preview = sys.platform.startswith("linux")

            if use_native_linux_preview:
                self.native_viewer = _NativePreview(self._initial_state)
                self.native_viewer.atomPicked.connect(self.set_atom_selection)
                self.native_viewer.emptyPicked.connect(self.clear_atom_selection)
                preview_layout.addWidget(self.native_viewer, stretch=1)
            elif QWebEngineView is not None:
                viewer = QWebEngineView()
                channel = QWebChannel(viewer.page())
                self.bridge = _Bridge(self)
                channel.registerObject("bridge", self.bridge)
                viewer.page().setWebChannel(channel)
                viewer.setHtml(_qt_viewer_html(assets_dir, self.current_state()), QUrl.fromLocalFile(str(assets_dir.resolve()) + "/"))
                preview_layout.addWidget(viewer, stretch=1)
            else:
                viewer_fallback = QLabel(
                    "The interactive 3D preview is unavailable in this session.\n"
                    "Install PySide6 QtWebEngine for the local NGL viewer, and on Linux make sure DISPLAY "
                    "or WAYLAND_DISPLAY is available.\n"
                    "You can still review RESP groups and save your QM settings here."
                )
                viewer_fallback.setAlignment(Qt.AlignCenter)
                preview_layout.addWidget(viewer_fallback, stretch=1)
                self.bridge = None

            right_splitter.addWidget(preview_host)

            qm_panel = QWidget()
            qm_panel.setMinimumWidth(250)
            qm_panel.setMaximumWidth(320)
            qm_layout = QVBoxLayout(qm_panel)
            qm_layout.setContentsMargins(0, 0, 0, 0)
            qm_layout.setSpacing(10)
            qm_title = QLabel("QM Input")
            qm_title.setStyleSheet("font-size: 18px; font-weight: 600;")
            qm_layout.addWidget(qm_title)

            geometry_box = QGroupBox("Geometry Optimization")
            geometry_form = QFormLayout(geometry_box)
            self.geometry_combo = QComboBox()
            for key, label in _GEOMETRY_MODES:
                self.geometry_combo.addItem(label, key)
            geometry_form.addRow("Mode", self.geometry_combo)
            self.xtb_acc_label = QLabel("xTB acc")
            self.xtb_acc_spin = QDoubleSpinBox()
            self.xtb_acc_spin.setDecimals(2)
            self.xtb_acc_spin.setRange(0.01, 10.0)
            self.xtb_acc_spin.setSingleStep(0.01)
            geometry_form.addRow(self.xtb_acc_label, self.xtb_acc_spin)
            self.dft_functional_combo = QComboBox()
            for key, label in _FUNCTIONAL_OPTIONS:
                self.dft_functional_combo.addItem(label, key)
            geometry_form.addRow(_GEOMETRY_METHOD_LABEL, self.dft_functional_combo)
            self.dft_basis_combo = QComboBox()
            for key, label in _BASIS_OPTIONS:
                self.dft_basis_combo.addItem(label, key)
            geometry_form.addRow(_GEOMETRY_BASIS_SET_LABEL, self.dft_basis_combo)
            qm_layout.addWidget(geometry_box)

            resp_box = QGroupBox("RESP Fitting")
            resp_form = QFormLayout(resp_box)
            self.resp_same_checkbox = QCheckBox(_RESP_MATCH_GEOMETRY_LABEL)
            resp_form.addRow(self.resp_same_checkbox)
            self.resp_functional_combo = QComboBox()
            for key, label in _FUNCTIONAL_OPTIONS:
                self.resp_functional_combo.addItem(label, key)
            resp_form.addRow(_RESP_METHOD_LABEL, self.resp_functional_combo)
            self.resp_basis_combo = QComboBox()
            for key, label in _BASIS_OPTIONS:
                self.resp_basis_combo.addItem(label, key)
            resp_form.addRow(_RESP_BASIS_SET_LABEL, self.resp_basis_combo)
            qm_layout.addWidget(resp_box)

            job_box = QGroupBox("Job Settings")
            job_form = QFormLayout(job_box)
            self.grid_combo = QComboBox()
            for key, label in _GRID_OPTIONS:
                self.grid_combo.addItem(label, key)
            job_form.addRow("Grid", self.grid_combo)
            self.memory_spin = QSpinBox()
            self.memory_spin.setRange(250, 20000)
            self.memory_spin.setSingleStep(250)
            job_form.addRow("Memory (MB)", self.memory_spin)
            self.maxiter_spin = QSpinBox()
            self.maxiter_spin.setRange(10, 1000)
            job_form.addRow("Max Iter", self.maxiter_spin)
            self.charge_spin = QSpinBox()
            self.charge_spin.setRange(-10, 10)
            job_form.addRow("Net Charge", self.charge_spin)
            self.mult_spin = QSpinBox()
            self.mult_spin.setRange(1, 10)
            job_form.addRow("Multiplicity", self.mult_spin)
            qm_layout.addWidget(job_box)
            qm_layout.addStretch(1)
            right_splitter.addWidget(qm_panel)

            right_splitter.setStretchFactor(0, 1)
            right_splitter.setStretchFactor(1, 0)
            right_splitter.setSizes([920, 250])

            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            splitter.setSizes([520, 1080])

            self.geometry_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.xtb_acc_spin.valueChanged.connect(self._on_qm_controls_changed)
            self.dft_functional_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.dft_basis_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.resp_same_checkbox.toggled.connect(self._on_qm_controls_changed)
            self.resp_functional_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.resp_basis_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.grid_combo.currentIndexChanged.connect(self._on_qm_controls_changed)
            self.memory_spin.valueChanged.connect(self._on_qm_controls_changed)
            self.maxiter_spin.valueChanged.connect(self._on_qm_controls_changed)
            self.charge_spin.valueChanged.connect(self._on_qm_controls_changed)
            self.mult_spin.valueChanged.connect(self._on_qm_controls_changed)

            self._populate_controls()
            self._refresh_atom_table()

        def _populate_controls(self) -> None:
            qm = _normalized_editor_qm_settings(self._state.get("qm_settings"))
            self._state["qm_settings"] = qm
            resp_same_as_dft, resp_functional, resp_basis = _resp_settings_from_qm(qm)
            memory_mb, grid_key, maxiter, net_charge, multiplicity = _resource_settings_from_qm(qm)
            self._set_combo_value(self.geometry_combo, _geometry_mode_from_qm(qm))
            self.xtb_acc_spin.setValue(_xtb_acc_from_qm(qm))
            self._set_combo_value(self.dft_functional_combo, _dft_functional_from_qm(qm))
            self._set_combo_value(self.dft_basis_combo, _dft_basis_from_qm(qm))
            self.resp_same_checkbox.setChecked(resp_same_as_dft)
            self._set_combo_value(self.resp_functional_combo, resp_functional)
            self._set_combo_value(self.resp_basis_combo, resp_basis)
            self._set_combo_value(self.grid_combo, grid_key)
            self.memory_spin.setValue(memory_mb)
            self.maxiter_spin.setValue(maxiter)
            self.charge_spin.setValue(net_charge)
            self.mult_spin.setValue(multiplicity)
            self.auto_group_combo.blockSignals(True)
            self.auto_group_graph_method_combo.blockSignals(True)
            try:
                self._set_combo_value(
                    self.auto_group_combo,
                    str((self._state.get("group_constraints") or {}).get("auto_group_mode") or AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY),
                )
                self._set_combo_value(
                    self.auto_group_graph_method_combo,
                    str(
                        (self._state.get("group_constraints") or {}).get("auto_group_graph_method")
                        or AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
                    ),
                )
            finally:
                self.auto_group_graph_method_combo.blockSignals(False)
                self.auto_group_combo.blockSignals(False)
            self._refresh_auto_group_graph_method_note()
            self._sync_qm_ui_state()

        def _set_combo_value(self, combo, target: str) -> None:
            for index in range(combo.count()):
                if combo.itemData(index) == target:
                    combo.setCurrentIndex(index)
                    return

        def _current_auto_group_mode(self) -> str:
            current = self.auto_group_combo.currentData()
            if current is None:
                return AUTO_GROUP_MODE_HYDROGEN_AND_SYMMETRY
            return str(current)

        def _current_auto_group_graph_method(self) -> str:
            current = self.auto_group_graph_method_combo.currentData()
            if current is None:
                return AUTO_GROUP_GRAPH_METHOD_CONNECTIVITY
            return str(current)

        def _refresh_auto_group_graph_method_note(self) -> None:
            method = str(
                (self._state.get("group_constraints") or {}).get("auto_group_graph_method")
                or self._current_auto_group_graph_method()
            )
            warning = str((self._state.get("group_constraints") or {}).get("auto_group_graph_warning") or "")
            self.auto_group_graph_method_note.setText(
                f"{_auto_group_graph_method_description(method)} {warning}".strip()
            )

        def _sync_resp_level_from_dft(self) -> None:
            if not self.resp_same_checkbox.isChecked():
                return
            self._set_combo_value(self.resp_functional_combo, str(self.dft_functional_combo.currentData()))
            self._set_combo_value(self.resp_basis_combo, str(self.dft_basis_combo.currentData()))

        def _sync_qm_ui_state(self) -> None:
            if self._syncing_qm_controls:
                return
            self._syncing_qm_controls = True
            try:
                geometry_mode = str(self.geometry_combo.currentData() or QM_DEFAULT_GEOMETRY_MODE)
                show_xtb = geometry_mode_uses_xtb_preopt(geometry_mode)
                self.xtb_acc_label.setVisible(show_xtb)
                self.xtb_acc_spin.setVisible(show_xtb)

                dft_enabled = geometry_mode_uses_dft_optimization(geometry_mode)
                self.dft_functional_combo.setEnabled(dft_enabled)
                self.dft_basis_combo.setEnabled(dft_enabled)

                self._sync_resp_level_from_dft()
                resp_enabled = not self.resp_same_checkbox.isChecked()
                self.resp_functional_combo.setEnabled(resp_enabled)
                self.resp_basis_combo.setEnabled(resp_enabled)
            finally:
                self._syncing_qm_controls = False

        def _on_qm_controls_changed(self, *_args) -> None:
            if self._syncing_qm_controls:
                return
            self._sync_qm_ui_state()
            self._emit_state()

        def _selected_group_ids(self) -> set[int]:
            atom_lookup = {
                int(atom["index"]): atom
                for atom in ((self._state.get("group_constraints") or {}).get("atoms") or [])
            }
            return {
                int(atom_lookup[atom_index]["group_id"])
                for atom_index in self._selected_atoms
                if atom_index in atom_lookup and atom_lookup[atom_index].get("group_id") is not None
            }

        def _refresh_atom_table(self) -> None:
            atoms = (self._state.get("group_constraints") or {}).get("atoms") or []
            self.atom_table.blockSignals(True)
            self.atom_table.setRowCount(len(atoms))
            for row_index, atom in enumerate(atoms):
                values = [
                    str(atom["index"]),
                    str(atom["name"]),
                    str(atom["element"]),
                    "" if atom.get("group_id") is None else str(atom["group_id"]),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.UserRole, int(atom["index"]))
                    self.atom_table.setItem(row_index, column, item)
                if int(atom["index"]) in self._selected_atoms:
                    self.atom_table.selectRow(row_index)
            self.atom_table.blockSignals(False)
            self._refresh_auto_group_graph_method_note()
            self._refresh_group_table()
            self._emit_state()

        def _refresh_group_table(self) -> None:
            groups = (self._state.get("group_constraints") or {}).get("groups") or []
            selected_group_ids = self._selected_group_ids()
            self.group_table.blockSignals(True)
            self.group_table.setRowCount(max(1, len(groups)))
            if not groups:
                for column, value in enumerate(["", "No equality groups defined.", ""]):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.UserRole, None)
                    self.group_table.setItem(0, column, item)
                self.group_table.clearSelection()
                self.group_table.blockSignals(False)
                return

            for row_index, group in enumerate(groups):
                values = [
                    f"G{group['group_id']}",
                    str(group.get("label") or f"Group {group['group_id']}"),
                    ", ".join(str(atom_index) for atom_index in group["atom_indices"]),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.UserRole, int(group["group_id"]))
                    self.group_table.setItem(row_index, column, item)
                if int(group["group_id"]) in selected_group_ids:
                    self.group_table.selectRow(row_index)
            self.group_table.blockSignals(False)

        def _sync_selection_from_table(self) -> None:
            self._selected_atoms = set()
            for item in self.atom_table.selectedItems():
                atom_index = item.data(Qt.UserRole)
                if atom_index is not None:
                    self._selected_atoms.add(int(atom_index))
            self._refresh_group_table()
            self._emit_state()

        def set_atom_selection(self, atom_index: int, additive: bool = False) -> None:
            if additive:
                if atom_index in self._selected_atoms:
                    self._selected_atoms.remove(atom_index)
                else:
                    self._selected_atoms.add(atom_index)
            else:
                self._selected_atoms = {int(atom_index)}
            self._sync_atom_selection_widgets()

        def clear_atom_selection(self) -> None:
            if not self._selected_atoms:
                return
            self._selected_atoms = set()
            self._sync_atom_selection_widgets()

        def _sync_atom_selection_widgets(self) -> None:
            self.atom_table.blockSignals(True)
            self.atom_table.clearSelection()
            for row_index in range(self.atom_table.rowCount()):
                item = self.atom_table.item(row_index, 0)
                if item is not None and int(item.data(Qt.UserRole)) in self._selected_atoms:
                    self.atom_table.selectRow(row_index)
            self.atom_table.blockSignals(False)
            self._refresh_group_table()
            self._emit_state()

        def _next_group_id(self) -> int:
            atoms = (self._state.get("group_constraints") or {}).get("atoms") or []
            existing = [int(atom["group_id"]) for atom in atoms if atom.get("group_id") is not None]
            return (max(existing) + 1) if existing else 1

        def _rebuild_groups(self) -> None:
            atoms = (self._state.get("group_constraints") or {}).get("atoms") or []
            grouped: dict[int, list[int]] = {}
            label_lookup = {
                int(group["group_id"]): str(group.get("label") or f"Group {group['group_id']}")
                for group in ((self._state.get("group_constraints") or {}).get("groups") or [])
            }
            for atom in atoms:
                group_id = atom.get("group_id")
                if group_id is None:
                    continue
                grouped.setdefault(int(group_id), []).append(int(atom["index"]))
            self._state["group_constraints"]["auto_group_mode"] = self._current_auto_group_mode()
            self._state["group_constraints"]["auto_group_graph_method"] = self._current_auto_group_graph_method()
            self._state["group_constraints"]["groups"] = [
                {
                    "group_id": group_id,
                    "label": label_lookup.get(group_id, f"Group {group_id}"),
                    "atom_indices": sorted(indices),
                    "auto": False,
                }
                for group_id, indices in sorted(grouped.items())
                if len(indices) > 1
            ]

        def assign_new_group(self) -> None:
            if not self._selected_atoms:
                return
            group_id = self._next_group_id()
            for atom in self._state["group_constraints"]["atoms"]:
                if int(atom["index"]) in self._selected_atoms:
                    atom["group_id"] = group_id
            self._rebuild_groups()
            self._refresh_atom_table()

        def clear_selected_groups(self) -> None:
            if not self._selected_atoms:
                return
            for atom in self._state["group_constraints"]["atoms"]:
                if int(atom["index"]) in self._selected_atoms:
                    atom["group_id"] = None
            self._rebuild_groups()
            self._refresh_atom_table()

        def reset_auto(self) -> None:
            self._state["group_constraints"] = suggest_group_constraints(
                self._molecule,
                auto_group_mode=self._current_auto_group_mode(),
                auto_group_graph_method=self._current_auto_group_graph_method(),
            )
            self._selected_atoms = set()
            self._refresh_atom_table()

        def _on_auto_group_mode_changed(self) -> None:
            self.reset_auto()

        def _select_group_from_table(self) -> None:
            selected_group_id: int | None = None
            for item in self.group_table.selectedItems():
                candidate = item.data(Qt.UserRole)
                if candidate is not None:
                    selected_group_id = int(candidate)
                    break
            if selected_group_id is None:
                return
            self.select_group(selected_group_id)

        def select_group(self, group_id: int) -> None:
            for group in ((self._state.get("group_constraints") or {}).get("groups") or []):
                if int(group["group_id"]) != int(group_id):
                    continue
                self._selected_atoms = {int(atom_index) for atom_index in group["atom_indices"]}
                self.atom_table.blockSignals(True)
                self.atom_table.clearSelection()
                for row_index in range(self.atom_table.rowCount()):
                    item = self.atom_table.item(row_index, 0)
                    if item is not None and int(item.data(Qt.UserRole)) in self._selected_atoms:
                        self.atom_table.selectRow(row_index)
                self.atom_table.blockSignals(False)
                self._refresh_group_table()
                self._emit_state()
                return

        def current_state(self) -> dict[str, Any]:
            qm_settings = _build_qm_settings(
                geometry_mode=str(self.geometry_combo.currentData() or QM_DEFAULT_GEOMETRY_MODE),
                xtb_acc=float(self.xtb_acc_spin.value()),
                dft_functional=str(self.dft_functional_combo.currentData() or QM_DEFAULT_DFT_FUNCTIONAL),
                dft_basis=str(self.dft_basis_combo.currentData() or QM_DEFAULT_DFT_BASIS),
                resp_same_as_dft_optimization=bool(self.resp_same_checkbox.isChecked()),
                resp_functional=str(self.resp_functional_combo.currentData() or QM_DEFAULT_RESP_FUNCTIONAL),
                resp_basis=str(self.resp_basis_combo.currentData() or QM_DEFAULT_RESP_BASIS),
                memory_mb=int(self.memory_spin.value()),
                grid=str(self.grid_combo.currentData() or QM_DEFAULT_GRID),
                maxiter=int(self.maxiter_spin.value()),
                net_charge=int(self.charge_spin.value()),
                multiplicity=int(self.mult_spin.value()),
            )
            self._state["qm_settings"] = qm_settings
            self._state["group_constraints"]["auto_group_mode"] = self._current_auto_group_mode()
            self._state["group_constraints"]["auto_group_graph_method"] = self._current_auto_group_graph_method()
            self._state["selected_atom_indices"] = sorted(self._selected_atoms)
            return self._state

        def _emit_state(self) -> None:
            payload = self.current_state()
            if self.native_viewer is not None:
                self.native_viewer.set_state(payload)
            if self.bridge is not None:
                payload_json = json.dumps(payload)
                if payload_json != self._last_emitted_state_json:
                    self._last_emitted_state_json = payload_json
                    self.bridge.stateChanged.emit(payload_json)

        def save_and_accept(self) -> None:
            payload = self.current_state()
            payload["edited_in_popup"] = True
            payload["editor_mode"] = "qt_popup"
            self._state = payload
            self.accept()

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    dialog = _Dialog(initial_state=session_state)
    if dialog.exec() == QDialog.Accepted:
        payload = dialog.current_state()
        payload["edited_in_popup"] = True
        payload["editor_mode"] = "qt_popup"
    else:
        payload = dict(session_state)
        payload["edited_in_popup"] = False
        payload["editor_mode"] = "cancelled"

    if created_app:
        app.quit()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return payload
