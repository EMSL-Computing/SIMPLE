from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from amber_metallo.reporting import write_json
from amber_metallo.ti.analysis import InheritedMDSettings
from amber_metallo.ti.config import (
    TIDecouplingMode,
    TIImplementationMode,
    TIProductionEnsemble,
    TIProtocolConfig,
    TISamplingMode,
)


@dataclass(slots=True)
class TIWindow:
    filename: str
    title: str
    phase: str
    clambda: float
    start_source: str
    content: str
    equil_filename: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PreparationStage:
    filename: str
    title: str
    start_source: str
    writes_trajectory: bool
    content: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_WATER_REF_PREP_MIN_CYCLES = 5000
_WATER_REF_PREP_NVT_NS = 0.025
_WATER_REF_PREP_NPT_NS = 0.050
_WATER_REF_PREP_EQ_NS = 0.100
_SOFTCORE_MAX_DT_PS = 0.001
_QOFF_ENDPOINT_PREP_MAX_DT_PS = 0.001
_BOUND_START_PREP_MAX_DT_PS = 0.001


def _ns_to_nstlim(ns: float, dt_ps: float) -> int:
    return int(round((ns * 1000.0) / dt_ps))


def _merged_lambda_schedule(*schedules: list[float]) -> list[float]:
    return sorted({round(float(value), 3) for schedule in schedules for value in schedule})


def _lambda_label(value: float) -> str:
    rounded_two = round(float(value), 2)
    if abs(float(value) - rounded_two) <= 1.0e-9:
        return f"{rounded_two:.2f}"
    return f"{float(value):.3f}"


def _uses_single_topology_gti_decoupling(config: TIProtocolConfig) -> bool:
    return (
        config.implementation_mode == TIImplementationMode.AMBER_12_6_4_GTI
        and config.decoupling_mode == TIDecouplingMode.COMBINED_Q_VDW
    )


def _cntrl_lines(
    *,
    settings: InheritedMDSettings,
    production_ensemble: TIProductionEnsemble,
    nstlim: int,
    clambda: float,
    charge_mask: str | None,
    timask1: str | None,
    timask2: str | None,
    scmask1: str | None,
    scmask2: str | None,
    restraint_file: str | None,
    start_source: str,
    scalpha: float,
    scbeta: float,
    dt_ps_override: float | None = None,
    ntc_override: int | None = None,
    ntf_override: int | None = None,
    logdvdl: bool = True,
    random_seed: bool = False,
    positional_restraint_mask: str | None = None,
    positional_restraint_force_constant: float = 0.0,
) -> list[str]:
    ntx = 1 if start_source == "snapshot" else 5
    irest = 0 if start_source == "snapshot" else 1
    nmropt = 1 if restraint_file else 0
    softcore_enabled = bool(scmask1 or scmask2)
    resolved_dt_ps = dt_ps_override if dt_ps_override is not None else settings.dt_ps
    if softcore_enabled:
        resolved_dt_ps = min(resolved_dt_ps, _SOFTCORE_MAX_DT_PS)
    resolved_ntc = ntc_override if ntc_override is not None else (1 if softcore_enabled else settings.ntc)
    resolved_ntf = ntf_override if ntf_override is not None else (1 if softcore_enabled else settings.ntf)
    if production_ensemble == TIProductionEnsemble.NVT:
        resolved_ntb = 1
        resolved_ntp = 0
    else:
        resolved_ntb = 2
        resolved_ntp = 1
    lines = [
        "&cntrl",
        "  imin = 0,",
        f"  ntx = {ntx},",
        f"  irest = {irest},",
        f"  nstlim = {nstlim},",
        f"  dt = {resolved_dt_ps:.6f},",
        f"  tempi = {(settings.tempi_k or settings.temperature_k):.3f},",
        f"  temp0 = {settings.temperature_k:.3f},",
        f"  ntb = {resolved_ntb},",
        f"  ntp = {resolved_ntp},",
        *([f"  pres0 = {settings.pressure_bar:.3f},"] if resolved_ntp > 0 else []),
        f"  cut = {settings.cut_angstrom:.3f},",
        f"  ntc = {resolved_ntc},",
        f"  ntf = {resolved_ntf},",
        f"  ntt = {settings.ntt},",
        f"  gamma_ln = {settings.gamma_ln:.3f},",
        f"  ntpr = {settings.ntpr},",
        f"  ntwx = {settings.ntwx},",
        f"  ntwr = {settings.ntwr},",
        f"  ioutfm = {settings.ioutfm},",
        "  ntxo = 1,",
        f"  iwrap = {settings.iwrap},",
        "  icfe = 1,",
        f"  clambda = {clambda:.3f},",
        f"  logdvdl = {1 if logdvdl else 0},",
        f"  ifsc = {1 if softcore_enabled else 0},",
        f"  nmropt = {nmropt},",
    ]
    if random_seed:
        lines.append("  ig = -1,")
    if positional_restraint_mask:
        lines.extend(
            [
                "  ntr = 1,",
                f"  restraintmask = '{positional_restraint_mask}',",
                f"  restraint_wt = {positional_restraint_force_constant:.3f},",
            ]
        )
    if charge_mask:
        lines.append(f"  crgmask = '{charge_mask}',")
    if timask1 is not None or timask2 is not None:
        lines.append(f"  timask1 = '{timask1 or ''}',")
        lines.append(f"  timask2 = '{timask2 or ''}',")
    if scmask1 is not None or scmask2 is not None:
        lines.append(f"  scmask1 = '{scmask1 or ''}',")
        lines.append(f"  scmask2 = '{scmask2 or ''}',")
        lines.append(f"  scalpha = {scalpha:.3f},")
        lines.append(f"  scbeta = {scbeta:.3f},")
    if settings.barostat is not None and resolved_ntp > 0:
        lines.append(f"  barostat = {settings.barostat},")
    if settings.taup is not None and resolved_ntp > 0:
        lines.append(f"  taup = {settings.taup:.3f},")
    lines.append("/")
    return lines


def _water_ref_minimization_lines(
    *,
    settings: InheritedMDSettings,
    positional_restraint_mask: str | None = None,
    restraint_force_constant: float = 0.0,
) -> list[str]:
    lines = [
        "&cntrl",
        "  imin = 1,",
        "  ntx = 1,",
        "  irest = 0,",
        f"  maxcyc = {_WATER_REF_PREP_MIN_CYCLES},",
        f"  ncyc = {_WATER_REF_PREP_MIN_CYCLES // 2},",
        "  ntmin = 1,",
        "  ntb = 1,",
        f"  ntpr = {max(50, min(settings.ntpr, 500))},",
        "  cut = 9.0,",
        "/",
    ]
    if positional_restraint_mask:
        lines[-1:-1] = [
            "  ntr = 1,",
            f"  restraintmask = '{positional_restraint_mask}',",
            f"  restraint_wt = {restraint_force_constant:.3f},",
        ]
    return lines


def _water_ref_md_stage_lines(
    *,
    settings: InheritedMDSettings,
    nstlim: int,
    start_source: str,
    ntb: int,
    ntp: int,
    tempi_k: float | None = None,
    positional_restraint_mask: str | None = None,
    restraint_force_constant: float = 0.0,
) -> list[str]:
    ntx = 1 if start_source == "system" else 5
    irest = 0 if start_source == "system" else 1
    resolved_tempi = settings.tempi_k if tempi_k is None else tempi_k
    if resolved_tempi is None:
        resolved_tempi = settings.temperature_k
    lines = [
        "&cntrl",
        "  imin = 0,",
        f"  ntx = {ntx},",
        f"  irest = {irest},",
        f"  nstlim = {nstlim},",
        f"  dt = {settings.dt_ps:.6f},",
        f"  tempi = {resolved_tempi:.3f},",
        f"  temp0 = {settings.temperature_k:.3f},",
        f"  ntb = {ntb},",
        f"  ntp = {ntp},",
        f"  pres0 = {settings.pressure_bar:.3f},",
        f"  ntc = {settings.ntc},",
        f"  ntf = {settings.ntf},",
        f"  ntt = {settings.ntt},",
        f"  gamma_ln = {settings.gamma_ln:.3f},",
        f"  ntpr = {settings.ntpr},",
        f"  ntwx = {settings.ntwx},",
        f"  ntwr = {settings.ntwr},",
        f"  ioutfm = {settings.ioutfm},",
        "  ntxo = 1,",
        f"  iwrap = {settings.iwrap},",
    ]
    if settings.barostat is not None and ntp > 0:
        lines.append(f"  barostat = {settings.barostat},")
    if settings.taup is not None and ntp > 0:
        lines.append(f"  taup = {settings.taup:.3f},")
    if positional_restraint_mask:
        lines.extend(
            [
                "  ntr = 1,",
                f"  restraintmask = '{positional_restraint_mask}',",
                f"  restraint_wt = {restraint_force_constant:.3f},",
            ]
        )
    lines.append("/")
    return lines


def _render_water_ref_stage(
    *,
    title: str,
    lines: list[str],
) -> str:
    return "\n".join([title, *lines]) + "\n"


def generate_water_reference_preparation_inputs(
    *,
    inherited_settings: InheritedMDSettings,
    output_dir: Path,
    positional_restraint_mask: str | None = None,
    restraint_force_constant: float = 0.0,
) -> list[PreparationStage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = output_dir / "prep" / "inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)

    stages: list[PreparationStage] = []
    stage_specs = [
        (
            "01_min.in",
            "Water-reference minimization",
            "system",
            False,
            _render_water_ref_stage(
                title="Water-reference endpoint minimization",
                lines=_water_ref_minimization_lines(
                    settings=inherited_settings,
                    positional_restraint_mask=positional_restraint_mask,
                    restraint_force_constant=restraint_force_constant,
                ),
            ),
        ),
        (
            "02_heat.in",
            "Water-reference NVT heating",
            "previous_stage",
            True,
            _render_water_ref_stage(
                title="Water-reference endpoint NVT heating",
                lines=_water_ref_md_stage_lines(
                    settings=inherited_settings,
                    nstlim=_ns_to_nstlim(_WATER_REF_PREP_NVT_NS, inherited_settings.dt_ps),
                    start_source="system",
                    ntb=1,
                    ntp=0,
                    tempi_k=0.0,
                    positional_restraint_mask=positional_restraint_mask,
                    restraint_force_constant=restraint_force_constant,
                ),
            ),
        ),
        (
            "03_density.in",
            "Water-reference NPT density equilibration",
            "previous_stage",
            True,
            _render_water_ref_stage(
                title="Water-reference endpoint NPT density equilibration",
                lines=_water_ref_md_stage_lines(
                    settings=inherited_settings,
                    nstlim=_ns_to_nstlim(_WATER_REF_PREP_NPT_NS, inherited_settings.dt_ps),
                    start_source="previous_stage",
                    ntb=max(2, inherited_settings.ntb),
                    ntp=max(1, inherited_settings.ntp),
                    positional_restraint_mask=positional_restraint_mask,
                    restraint_force_constant=restraint_force_constant,
                ),
            ),
        ),
        (
            "04_eq.in",
            "Water-reference NPT equilibration",
            "previous_stage",
            True,
            _render_water_ref_stage(
                title="Water-reference endpoint NPT equilibration",
                lines=_water_ref_md_stage_lines(
                    settings=inherited_settings,
                    nstlim=_ns_to_nstlim(_WATER_REF_PREP_EQ_NS, inherited_settings.dt_ps),
                    start_source="previous_stage",
                    ntb=max(2, inherited_settings.ntb),
                    ntp=max(1, inherited_settings.ntp),
                    positional_restraint_mask=positional_restraint_mask,
                    restraint_force_constant=restraint_force_constant,
                ),
            ),
        ),
    ]

    for filename, title, start_source, writes_trajectory, content in stage_specs:
        (prep_dir / filename).write_text(content, encoding="utf-8")
        stages.append(
            PreparationStage(
                filename=(Path("prep") / "inputs" / filename).as_posix(),
                title=title,
                start_source=start_source,
                writes_trajectory=writes_trajectory,
                content=content,
            )
        )

    write_json(output_dir / "prep" / "prep_manifest.json", {"stages": [stage.to_dict() for stage in stages]})
    return stages


def generate_bound_start_preparation_inputs(
    *,
    config: TIProtocolConfig,
    inherited_settings: InheritedMDSettings,
    restraint_file: str | None,
    output_dir: Path,
    positional_restraint_mask: str | None = None,
) -> list[PreparationStage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = output_dir / "bound_start_prep" / "inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)

    eq_dt_ps = min(config.bound_start_eq_dt_ps, _BOUND_START_PREP_MAX_DT_PS)
    stages = [
        PreparationStage(
            filename=(Path("bound_start_prep") / "inputs" / "01_min.in").as_posix(),
            title="Bound starting-structure minimization",
            start_source="snapshot",
            writes_trajectory=False,
            content=_render_stage(
                title="Bound starting-structure minimization before TI",
                lines=_qoff_endpoint_minimization_lines(
                    settings=inherited_settings,
                    max_cycles=config.bound_start_min_cycles,
                    restraint_file=restraint_file,
                    positional_restraint_mask=positional_restraint_mask,
                    positional_restraint_force_constant=config.counterion_restraint_force_constant,
                ),
                restraint_file=restraint_file,
            ),
        ),
        PreparationStage(
            filename=(Path("bound_start_prep") / "inputs" / "02_eq.in").as_posix(),
            title="Bound starting-structure equilibration",
            start_source="previous_stage",
            writes_trajectory=True,
            content=_render_stage(
                title="Bound starting-structure CPU equilibration before TI",
                lines=_qoff_endpoint_eq_lines(
                    settings=inherited_settings,
                    nstlim=_ns_to_nstlim(config.bound_start_eq_ns, eq_dt_ps),
                    restraint_file=restraint_file,
                    dt_ps=eq_dt_ps,
                    positional_restraint_mask=positional_restraint_mask,
                    positional_restraint_force_constant=config.counterion_restraint_force_constant,
                ),
                restraint_file=restraint_file,
            ),
        ),
    ]
    for stage in stages:
        (prep_dir / Path(stage.filename).name).write_text(stage.content, encoding="utf-8")
    write_json(
        output_dir / "bound_start_prep" / "prep_manifest.json",
        {"stages": [stage.to_dict() for stage in stages]},
    )
    return stages


def _qoff_endpoint_minimization_lines(
    *,
    settings: InheritedMDSettings,
    max_cycles: int,
    restraint_file: str | None,
    positional_restraint_mask: str | None = None,
    positional_restraint_force_constant: float = 0.0,
) -> list[str]:
    lines = [
        "&cntrl",
        "  imin = 1,",
        "  ntx = 1,",
        "  irest = 0,",
        f"  maxcyc = {max_cycles},",
        f"  ncyc = {max_cycles // 2},",
        "  ntmin = 1,",
        "  ntb = 1,",
        f"  ntpr = {max(50, min(settings.ntpr, 500))},",
        f"  cut = {settings.cut_angstrom:.3f},",
        f"  nmropt = {1 if restraint_file else 0},",
        "/",
    ]
    if positional_restraint_mask:
        lines[-1:-1] = [
            "  ntr = 1,",
            f"  restraintmask = '{positional_restraint_mask}',",
            f"  restraint_wt = {positional_restraint_force_constant:.3f},",
        ]
    return lines


def _qoff_endpoint_eq_lines(
    *,
    settings: InheritedMDSettings,
    nstlim: int,
    restraint_file: str | None,
    dt_ps: float,
    positional_restraint_mask: str | None = None,
    positional_restraint_force_constant: float = 0.0,
) -> list[str]:
    lines = [
        "&cntrl",
        "  imin = 0,",
        "  ntx = 1,",
        "  irest = 0,",
        f"  nstlim = {nstlim},",
        f"  dt = {dt_ps:.6f},",
        f"  tempi = {(settings.tempi_k or settings.temperature_k):.3f},",
        f"  temp0 = {settings.temperature_k:.3f},",
        f"  ntb = {settings.ntb},",
        f"  ntp = {settings.ntp},",
        f"  pres0 = {settings.pressure_bar:.3f},",
        f"  cut = {settings.cut_angstrom:.3f},",
        f"  ntc = {settings.ntc},",
        f"  ntf = {settings.ntf},",
        f"  ntt = {settings.ntt},",
        f"  gamma_ln = {settings.gamma_ln:.3f},",
        f"  ntpr = {settings.ntpr},",
        f"  ntwx = {settings.ntwx},",
        f"  ntwr = {settings.ntwr},",
        f"  ioutfm = {settings.ioutfm},",
        "  ntxo = 1,",
        f"  iwrap = {settings.iwrap},",
        f"  nmropt = {1 if restraint_file else 0},",
    ]
    if positional_restraint_mask:
        lines.extend(
            [
                "  ntr = 1,",
                f"  restraintmask = '{positional_restraint_mask}',",
                f"  restraint_wt = {positional_restraint_force_constant:.3f},",
            ]
        )
    if settings.barostat is not None and settings.ntp > 0:
        lines.append(f"  barostat = {settings.barostat},")
    if settings.taup is not None and settings.ntp > 0:
        lines.append(f"  taup = {settings.taup:.3f},")
    lines.append("/")
    return lines


def _render_stage(*, title: str, lines: list[str], restraint_file: str | None = None) -> str:
    content_lines = [title, *lines]
    if restraint_file:
        content_lines.extend(
            [
                "&wt type='END' /",
                f"DISANG={Path(restraint_file).as_posix()}",
                "LISTOUT=POUT",
            ]
        )
    return "\n".join(content_lines) + "\n"


def generate_qoff_endpoint_preparation_inputs(
    *,
    config: TIProtocolConfig,
    inherited_settings: InheritedMDSettings,
    restraint_file: str | None,
    output_dir: Path,
    positional_restraint_mask: str | None = None,
) -> list[PreparationStage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prep_dir = output_dir / "qoff_endpoint_prep" / "inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)

    eq_dt_ps = min(config.qoff_dt_ps, _QOFF_ENDPOINT_PREP_MAX_DT_PS)
    stages = [
        PreparationStage(
            filename=(Path("qoff_endpoint_prep") / "inputs" / "01_min.in").as_posix(),
            title="Qoff-endpoint minimization",
            start_source="qoff_endpoint",
            writes_trajectory=False,
            content=_render_stage(
                title="Qoff-endpoint minimization before VDW-off TI",
                lines=_qoff_endpoint_minimization_lines(
                    settings=inherited_settings,
                    max_cycles=config.qoff_endpoint_min_cycles,
                    restraint_file=restraint_file,
                    positional_restraint_mask=positional_restraint_mask,
                    positional_restraint_force_constant=config.counterion_restraint_force_constant,
                ),
                restraint_file=restraint_file,
            ),
        ),
        PreparationStage(
            filename=(Path("qoff_endpoint_prep") / "inputs" / "02_eq.in").as_posix(),
            title="Qoff-endpoint equilibration",
            start_source="previous_stage",
            writes_trajectory=True,
            content=_render_stage(
                title="Qoff-endpoint equilibration before VDW-off TI",
                lines=_qoff_endpoint_eq_lines(
                    settings=inherited_settings,
                    nstlim=_ns_to_nstlim(config.qoff_endpoint_eq_ns, eq_dt_ps),
                    restraint_file=restraint_file,
                    dt_ps=eq_dt_ps,
                    positional_restraint_mask=positional_restraint_mask,
                    positional_restraint_force_constant=config.counterion_restraint_force_constant,
                ),
                restraint_file=restraint_file,
            ),
        ),
    ]
    for stage in stages:
        (prep_dir / Path(stage.filename).name).write_text(stage.content, encoding="utf-8")
    write_json(
        output_dir / "qoff_endpoint_prep" / "prep_manifest.json",
        {"stages": [stage.to_dict() for stage in stages]},
    )
    return stages


def _render_window(
    *,
    title: str,
    settings: InheritedMDSettings,
    production_ensemble: TIProductionEnsemble,
    production_time_ns: float,
    clambda: float,
    charge_mask: str | None,
    timask1: str | None,
    timask2: str | None,
    scmask1: str | None,
    scmask2: str | None,
    restraint_file: str | None,
    start_source: str,
    scalpha: float,
    scbeta: float,
    dt_ps_override: float | None = None,
    ntc_override: int | None = None,
    ntf_override: int | None = None,
    logdvdl: bool = True,
    random_seed: bool = False,
    positional_restraint_mask: str | None = None,
    positional_restraint_force_constant: float = 0.0,
) -> str:
    softcore_enabled = bool(scmask1 or scmask2)
    resolved_dt_ps = dt_ps_override if dt_ps_override is not None else settings.dt_ps
    if softcore_enabled:
        resolved_dt_ps = min(resolved_dt_ps, _SOFTCORE_MAX_DT_PS)
    lines = [
        title,
        *_cntrl_lines(
            settings=settings,
            production_ensemble=production_ensemble,
            nstlim=_ns_to_nstlim(production_time_ns, resolved_dt_ps),
            clambda=clambda,
            charge_mask=charge_mask,
            timask1=timask1,
            timask2=timask2,
            scmask1=scmask1,
            scmask2=scmask2,
            restraint_file=restraint_file,
            start_source=start_source,
            scalpha=scalpha,
            scbeta=scbeta,
            dt_ps_override=dt_ps_override,
            ntc_override=ntc_override,
            ntf_override=ntf_override,
            logdvdl=logdvdl,
            random_seed=random_seed,
            positional_restraint_mask=positional_restraint_mask,
            positional_restraint_force_constant=positional_restraint_force_constant,
        ),
    ]
    if restraint_file:
        lines.extend(
            [
                "&wt type='END' /",
                f"DISANG={Path(restraint_file).as_posix()}",
                "LISTOUT=POUT",
            ]
        )
    return "\n".join(lines) + "\n"


def generate_ti_inputs(
    *,
    config: TIProtocolConfig,
    inherited_settings: InheritedMDSettings,
    atom_mask: str,
    restraint_file: str | None,
    output_dir: Path,
    qoff_start_source: str = "snapshot",
    qoff_timask1: str | None = None,
    qoff_timask2: str | None = None,
    qoff_charge_mask: str | None = None,
    qoff_restraint_file: str | None = None,
    positional_restraint_mask: str | None = None,
    qoff_positional_restraint_mask: str | None = None,
) -> list[TIWindow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows: list[TIWindow] = []

    combined_layout = _uses_single_topology_gti_decoupling(config)
    qoff_dir = output_dir / "inputs" if combined_layout else output_dir / "qoff" / "inputs"
    vdwoff_dir = output_dir / "vdwoff" / "inputs"
    qoff_dir.mkdir(parents=True, exist_ok=True)
    if not combined_layout:
        vdwoff_dir.mkdir(parents=True, exist_ok=True)

    bidirectional_sampling = config.sampling_mode == TISamplingMode.BIDIRECTIONAL

    def write_equilibration_inputs(
        *,
        phase: str,
        filename: str,
        title: str,
        clambda: float,
        charge_mask: str | None,
        timask1: str | None,
        timask2: str | None,
        scmask1: str | None,
        scmask2: str | None,
        restraint_path: str | None,
        dt_ps_override: float | None,
        ntc_override: int | None = None,
        ntf_override: int | None = None,
        positional_mask: str | None = None,
    ) -> str | None:
        if not bidirectional_sampling:
            return None
        equil_dir = output_dir / "equil_inputs" if combined_layout else output_dir / phase / "equil_inputs"
        equil_dir.mkdir(parents=True, exist_ok=True)
        equil_name = filename.replace(".in", "_equil.in")
        common = dict(
            settings=inherited_settings,
            production_ensemble=config.production_ensemble,
            production_time_ns=config.window_equilibration_ns,
            clambda=clambda,
            charge_mask=charge_mask,
            timask1=timask1,
            timask2=timask2,
            scmask1=scmask1,
            scmask2=scmask2,
            restraint_file=restraint_path,
            scalpha=config.scalpha,
            scbeta=config.scbeta,
            dt_ps_override=dt_ps_override,
            ntc_override=ntc_override,
            ntf_override=ntf_override,
            logdvdl=False,
            random_seed=True,
            positional_restraint_mask=positional_mask,
            positional_restraint_force_constant=config.counterion_restraint_force_constant,
        )
        equil_content = _render_window(
            title=f"{title} - equilibration (excluded from analysis)",
            start_source="restart",
            **common,
        )
        (equil_dir / equil_name).write_text(equil_content, encoding="utf-8")
        return (
            (Path("equil_inputs") / equil_name).as_posix()
            if combined_layout
            else (Path(phase) / "equil_inputs" / equil_name).as_posix()
        )

    # PMEMD GTI can also use a one-step softcore path when explicitly requested.
    use_single_topology_gti_decoupling = combined_layout
    qoff_lambdas = (
        _merged_lambda_schedule(config.charge_lambdas, config.vdw_lambdas)
        if use_single_topology_gti_decoupling
        else config.charge_lambdas
    )

    for index, clambda in enumerate(qoff_lambdas, start=1):
        window_start_source = qoff_start_source if index == 1 else "restart"
        title_prefix = "12-6-4 GTI softcore decoupling" if use_single_topology_gti_decoupling else "Charge-off TI"
        lambda_label = _lambda_label(clambda)
        title = f"{title_prefix} window {index:02d} (lambda={lambda_label})"
        filename = f"{index:02d}_lambda_{lambda_label}.in"
        resolved_qoff_restraint_file = (
            restraint_file if use_single_topology_gti_decoupling else (qoff_restraint_file or restraint_file)
        )
        content = _render_window(
            title=title,
            settings=inherited_settings,
            production_ensemble=config.production_ensemble,
            production_time_ns=config.production_time_ns,
            clambda=clambda,
            charge_mask=None if use_single_topology_gti_decoupling else (qoff_charge_mask or atom_mask),
            timask1=atom_mask if use_single_topology_gti_decoupling else (qoff_timask1 or atom_mask),
            timask2="" if use_single_topology_gti_decoupling else (qoff_timask2 or atom_mask),
            scmask1=atom_mask if use_single_topology_gti_decoupling else None,
            scmask2="" if use_single_topology_gti_decoupling else None,
            restraint_file=resolved_qoff_restraint_file,
            start_source="restart" if bidirectional_sampling else window_start_source,
            scalpha=config.scalpha,
            scbeta=config.scbeta,
            dt_ps_override=config.qoff_dt_ps,
            logdvdl=config.logdvdl,
            positional_restraint_mask=qoff_positional_restraint_mask or positional_restraint_mask,
            positional_restraint_force_constant=config.counterion_restraint_force_constant,
        )
        equil_filename = write_equilibration_inputs(
            phase="qoff",
            filename=filename,
            title=title,
            clambda=clambda,
            charge_mask=None if use_single_topology_gti_decoupling else (qoff_charge_mask or atom_mask),
            timask1=atom_mask if use_single_topology_gti_decoupling else (qoff_timask1 or atom_mask),
            timask2="" if use_single_topology_gti_decoupling else (qoff_timask2 or atom_mask),
            scmask1=atom_mask if use_single_topology_gti_decoupling else None,
            scmask2="" if use_single_topology_gti_decoupling else None,
            restraint_path=resolved_qoff_restraint_file,
            dt_ps_override=config.qoff_dt_ps,
            positional_mask=qoff_positional_restraint_mask or positional_restraint_mask,
        )
        (qoff_dir / filename).write_text(content, encoding="utf-8")
        windows.append(
            TIWindow(
                filename=(Path("inputs") / filename).as_posix()
                if combined_layout
                else (Path("qoff") / "inputs" / filename).as_posix(),
                title=title,
                phase="qoff",
                clambda=clambda,
                start_source=window_start_source,
                content=content,
                equil_filename=equil_filename,
            )
        )

    if use_single_topology_gti_decoupling:
        write_json(output_dir / "ti_manifest.json", _ti_manifest_payload(config, windows))
        return windows

    for index, clambda in enumerate(config.vdw_lambdas, start=1):
        lambda_label = _lambda_label(clambda)
        title = f"VDW-off TI window {index:02d} (lambda={lambda_label})"
        filename = f"{index:02d}_lambda_{lambda_label}.in"
        content = _render_window(
            title=title,
            settings=inherited_settings,
            production_ensemble=config.production_ensemble,
            production_time_ns=config.production_time_ns,
            clambda=clambda,
            charge_mask=None,
            timask1=atom_mask,
            timask2="",
            scmask1=atom_mask,
            scmask2="",
            restraint_file=restraint_file,
            start_source="qoff_endpoint",
            scalpha=config.scalpha,
            scbeta=config.scbeta,
            dt_ps_override=config.vdwoff_dt_ps,
            ntc_override=1,
            ntf_override=1,
            logdvdl=config.logdvdl,
            positional_restraint_mask=positional_restraint_mask,
            positional_restraint_force_constant=config.counterion_restraint_force_constant,
        )
        equil_filename = write_equilibration_inputs(
            phase="vdwoff",
            filename=filename,
            title=title,
            clambda=clambda,
            charge_mask=None,
            timask1=atom_mask,
            timask2="",
            scmask1=atom_mask,
            scmask2="",
            restraint_path=restraint_file,
            dt_ps_override=config.vdwoff_dt_ps,
            ntc_override=1,
            ntf_override=1,
            positional_mask=positional_restraint_mask,
        )
        (vdwoff_dir / filename).write_text(content, encoding="utf-8")
        windows.append(
            TIWindow(
                filename=(Path("vdwoff") / "inputs" / filename).as_posix(),
                title=title,
                phase="vdwoff",
                clambda=clambda,
                start_source="qoff_endpoint",
                content=content,
                equil_filename=equil_filename,
            )
        )

    write_json(output_dir / "ti_manifest.json", _ti_manifest_payload(config, windows))
    return windows


def _ti_manifest_payload(config: TIProtocolConfig, windows: list[TIWindow]) -> dict[str, object]:
    bidirectional = config.sampling_mode == TISamplingMode.BIDIRECTIONAL
    combined_layout = _uses_single_topology_gti_decoupling(config)
    directions = ["forward", "reverse"] if bidirectional else ["forward"]
    runs = [
        {
            "run_id": direction,
            "replica": 1,
            "direction": direction,
            "output_root": direction if bidirectional else "",
        }
        for direction in directions
    ]
    return {
        "output_layout": "combined_flat" if combined_layout else "split_phase_directories",
        "production_ensemble": config.production_ensemble.value,
        "sampling_protocol": {
            "mode": config.sampling_mode.value,
            "replicas": 1,
            "directions": directions,
            "window_equilibration_ns": config.window_equilibration_ns if bidirectional else 0.0,
            "production_time_ns": config.production_time_ns,
            "equilibration_dvdl_excluded": bidirectional,
        },
        "runs": runs,
        "windows": [window.to_dict() for window in windows],
    }
