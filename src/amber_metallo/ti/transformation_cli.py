from __future__ import annotations

import typer

from amber_metallo.cli import WizardChoice, _display_choice_table, _prompt_choice
from amber_metallo.reporting import console
from amber_metallo.ti.config import TIEndpointMode, TITransformationConfig
from amber_metallo.ti.metal_parameters import available_metal_endpoints


def prompt_metal_transformation(*, candidates, input_selection, amber_env, batch_plan=None):
    """Prompt after site detection and before frame/restraint selection."""
    from amber_metallo.ti.cli import _infer_workflow_water_settings
    plan = batch_plan if batch_plan is not None else {}
    if "mode" not in plan:
        choices = [
            WizardChoice("dummy", "Metal / ion -> dummy", "Default. Remove the selected atom's interactions."),
            WizardChoice("metal", "Metal -> another metal", "Transform the selected metal using 12-6-4 endpoint parameters."),
        ]
        _display_choice_table("TI endpoint", choices)
        plan["mode"] = _prompt_choice("Choose the TI endpoint", choices, default_key="dummy")
    if plan["mode"] == "dummy":
        return TITransformationConfig()
    if plan.get("multiple_workflows") and "scope" not in plan:
        choices = [
            WizardChoice("shared", "Same end metal for all cases", "Choose one endpoint and reuse it across selected workflows."),
            WizardChoice("per_case", "Choose end metal for each case", "Each workflow can have a different endpoint."),
        ]
        _display_choice_table("End metal across cases", choices)
        plan["scope"] = _prompt_choice("Apply one end metal to all cases or choose per case", choices, default_key="shared")
    water_model = None
    if input_selection.workflow_root is not None:
        water_model, _ = _infer_workflow_water_settings(input_selection.workflow_root)
    if water_model is None:
        water_choices = [WizardChoice(water, water.upper(), "Use the parameter family matching the existing MD solvent model.")
                         for water in ("opc", "spce", "tip3p", "tip4pew", "opc3", "fb3", "fb4")
                         if available_metal_endpoints(amber_env=amber_env, water_model=water)]
        if not water_choices:
            raise typer.BadParameter("No complete 12-6-4 endpoint parameter families were found.")
        _display_choice_table("Existing MD parameter / water model", water_choices)
        water_model = _prompt_choice("Choose the water model used to parameterize the existing MD system",
                                     water_choices, default_key=water_choices[0].key)
    catalog = available_metal_endpoints(amber_env=amber_env, water_model=water_model)
    if not catalog:
        raise typer.BadParameter(f"No complete 12-6-4 endpoints were found for {water_model}.")
    console.print(f"[dim]Endpoint parameters: {water_model.upper()}. Fe3+ is the preferred default; CN is not fixed by this choice.[/dim]")
    console.print("[dim]Metal mass also changes from source to end metal with lambda.[/dim]")

    def choose_endpoint(label):
        species = list(dict.fromkeys(item.label for item in catalog))
        choices = [WizardChoice(ion, ion, ", ".join(
            "OPC/Duvail (Fe uses Li/Merz)" if p.parameter_set == "duvail" else "Li/Merz"
            for p in catalog if p.label == ion)) for ion in species]
        _display_choice_table(f"12-6-4 end metals: {label}", choices)
        chosen = _prompt_choice("Choose the end metal", choices,
                                default_key="Fe3+" if "Fe3+" in species else species[0])
        matches = [p for p in catalog if p.label == chosen]
        if len(matches) > 1:
            families = [WizardChoice(p.parameter_set, p.parameter_set, p.frcmod_path) for p in matches]
            _display_choice_table("Endpoint parameter family", families)
            family = _prompt_choice("Choose the end-metal parameter family", families, default_key=matches[0].parameter_set)
            return next(p.endpoint() for p in matches if p.parameter_set == family)
        return matches[0].endpoint()

    if plan.get("scope") == "shared" and "endpoint" in plan:
        endpoint = plan["endpoint"]
        if endpoint not in [p.endpoint() for p in catalog]:
            raise typer.BadParameter("The shared end metal is incompatible with this case's solvent/parameter family. Choose per-case endpoints.")
        console.print(f"[cyan]Shared endpoint: {endpoint.element}{endpoint.formal_charge}+ ({endpoint.parameter_set}).[/cyan]")
        return TITransformationConfig(mode=TIEndpointMode.METAL, endpoint=endpoint)
    scope = "shared"
    if len(candidates) > 1 and plan.get("scope") != "shared":
        choices = [WizardChoice("shared", "Same end metal at all selected sites", "Apply one endpoint to selected sites."),
                   WizardChoice("per_site", "Choose end metal for each site", "Assign each selected site its own endpoint.")]
        _display_choice_table("End metal across sites", choices)
        scope = _prompt_choice("Use one end metal or choose for each selected site", choices, default_key="shared")
    if scope == "per_site":
        return TITransformationConfig(mode=TIEndpointMode.METAL, endpoints_by_site={
            site.site: choose_endpoint(f"site {site.site}: {site.element} at {site.key}") for site in candidates})
    endpoint = choose_endpoint("all selected sites")
    if plan.get("scope") == "shared":
        plan["endpoint"] = endpoint
    return TITransformationConfig(mode=TIEndpointMode.METAL, endpoint=endpoint)


def prompt_transformation_charge_compensation():
    from amber_metallo.ti.config import TIChargeCompensationMode
    choices = [
        WizardChoice("co_alchemical_counterions", "Compensate the endpoint charge difference",
                     "Same-charge transformations need no co-alchemical ion; otherwise discharge distant ions by the charge difference."),
        WizardChoice("none", "Keep existing ion charges", "Allow a net charge change when the two metal oxidation states differ."),
    ]
    _display_choice_table("Direct metal TI charge compensation", choices)
    return TIChargeCompensationMode(_prompt_choice("Choose charge compensation", choices, default_key="co_alchemical_counterions"))
