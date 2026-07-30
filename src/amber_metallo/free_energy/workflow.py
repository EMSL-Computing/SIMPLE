from __future__ import annotations

from typing import Any

from amber_metallo.free_energy.config import FreeEnergyMethod, FreeEnergyWorkflowConfig, to_ti_config
from amber_metallo.free_energy.mmpbsa import print_mmpbsa_summary, run_mmpbsa_workflow
from amber_metallo.ti.workflow import print_ti_workflow_summary, run_ti_workflow


def run_free_energy_workflow(*, config: FreeEnergyWorkflowConfig, dry_run: bool = False) -> dict[str, Any]:
    if config.free_energy.method == FreeEnergyMethod.TI:
        result = run_ti_workflow(config=to_ti_config(config), dry_run=dry_run)
        result["free_energy_method"] = FreeEnergyMethod.TI.value
        return result
    result = run_mmpbsa_workflow(config=config, dry_run=dry_run)
    result["free_energy_method"] = FreeEnergyMethod.MMPBSA.value
    return result


def print_free_energy_workflow_summary(result: dict[str, Any]) -> None:
    if result.get("free_energy_method") == FreeEnergyMethod.MMPBSA.value:
        print_mmpbsa_summary(result)
        return
    print_ti_workflow_summary(result)
