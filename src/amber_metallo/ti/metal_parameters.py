"""Discover complete 12-6-4 endpoint parameters without changing the solvent.

LJ values come from the selected frcmod; C4 values come from the matching
bundled Duvail table or Amber/ParmEd's water-specific Li/Merz table.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re

import gemmi

from amber_metallo.c4_assets import opc_duvail_c4_file, opc_duvail_ion_frcmod
from amber_metallo.environment import AmberEnvironment
from amber_metallo.ti.config import MetalEndpointConfig
from amber_metallo.ti.topology import _parse_official_126_nonbond


@dataclass(frozen=True)
class MetalParameters:
    element: str
    formal_charge: int
    parameter_set: str
    water_model: str
    rmin_half: float
    epsilon: float
    c4: float
    frcmod_path: str
    c4_source: str

    @property
    def label(self) -> str:
        return f"{self.element}{self.formal_charge if self.formal_charge != 1 else ''}+"

    def endpoint(self) -> MetalEndpointConfig:
        return MetalEndpointConfig(**{key: getattr(self, key) for key in (
            "element", "formal_charge", "parameter_set", "water_model")})


def _li_merz_c4_tables(amber_env: AmberEnvironment) -> tuple[dict, str]:
    try:
        from parmed.tools import add1264
    except ImportError:
        pass
    else:
        return add1264.DEFAULT_C4_PARAMS, str(add1264.__file__)
    # Amber's Python may differ from the Python running FreeE. Read the data
    # literal only; do not execute code from an environment-discovered file.
    if amber_env.amberhome is not None:
        for pattern in ("lib/python*/site-packages/parmed/tools/add1264.py",
                        "lib/python*/site-packages/ParmEd*/parmed/tools/add1264.py"):
            for path in sorted(amber_env.amberhome.glob(pattern)):
                for node in ast.parse(path.read_text(encoding="utf-8")).body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "DEFAULT_C4_PARAMS"
                        for target in node.targets
                    ):
                        return ast.literal_eval(node.value), str(path)
    return {}, ""


def available_metal_endpoints(*, amber_env: AmberEnvironment, water_model: str) -> list[MetalParameters]:
    water_model = water_model.lower()
    families = []
    if water_model == "opc":
        c4_path = opc_duvail_c4_file()
        c4_values = {}
        for line in c4_path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if len(tokens) >= 2:
                c4_values[tokens[0]] = float(tokens[1])
        families.append(("duvail", [opc_duvail_ion_frcmod()], c4_values, str(c4_path)))
    tables, c4_source = _li_merz_c4_tables(amber_env)
    official_files = amber_env.matching_1264_files(water_model, include_bundled_opc=False)
    families.append(("li_merz", official_files, tables.get(water_model.upper(), {}), c4_source))
    result = {}
    for family, paths, c4_values, source in families:
        for path in paths:
            for lj in _parse_official_126_nonbond([path]).values():
                match = re.fullmatch(r"([A-Za-z]{1,2})([1-4]?)\+", lj.label)
                if match is None:
                    continue
                element = match[1].title()
                charge = int(match[2] or 1)
                c4 = c4_values.get(f"{element}{charge}")
                if not gemmi.Element(element).is_metal or c4 is None:
                    continue
                if not all(math.isfinite(value) for value in (lj.rmin_half, lj.epsilon, c4)):
                    continue
                if lj.rmin_half <= 0 or lj.epsilon <= 0:
                    continue
                key = (element, charge, family)
                result[key] = MetalParameters(element, charge, family, water_model,
                    lj.rmin_half, lj.epsilon, float(c4), str(path.resolve()), source)
    return sorted(result.values(), key=lambda p: (p.element != "Fe" or p.formal_charge != 3,
                                                 p.element, p.formal_charge, p.parameter_set))


def resolve_metal_endpoint(endpoint: MetalEndpointConfig, *, amber_env: AmberEnvironment) -> MetalParameters:
    for item in available_metal_endpoints(amber_env=amber_env, water_model=endpoint.water_model):
        if item.endpoint() == endpoint:
            return item
    raise ValueError(
        f"No complete {endpoint.parameter_set} 12-6-4 LJ/C4 parameters for "
        f"{endpoint.element}{endpoint.formal_charge}+ in {endpoint.water_model.upper()}. "
        "Install the matching Amber/ParmEd parameters or choose an available endpoint."
    )
