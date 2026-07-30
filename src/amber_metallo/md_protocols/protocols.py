from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from amber_metallo.config import MDConfig, ProtocolKind
from amber_metallo.reporting import write_json


DEFAULT_RESTRAINT_MASK = "!(:WAT,HOH,Na+,K+,Cl-,CA,Ca2+) & !@H="
_INTEGER_OVERRIDE_KEYS = {
    "nstlim", "maxcyc", "ncyc", "ntt", "barostat", "ntpr", "ntwx", "ntwr", "iwrap",
    "ntx", "irest", "ntc", "ntf", "ntxo", "ig",
}
_FLOAT_OVERRIDE_KEYS = {"dt", "temp0", "tempi", "pres0", "gamma_ln", "taup", "restraint_wt", "cut"}
_ALLOWED_OVERRIDE_KEYS = _INTEGER_OVERRIDE_KEYS | _FLOAT_OVERRIDE_KEYS


@dataclass(slots=True)
class MDStage:
    filename: str
    title: str
    stage_type: str
    content: str
    writes_trajectory: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ps_to_nstlim(ps: float, dt_ps: float = 0.002) -> int:
    return int(round(ps / dt_ps))


def _ns_to_nstlim(ns: float, dt_ps: float = 0.002) -> int:
    return int(round((ns * 1000.0) / dt_ps))


def _equilibration_restraint_mask(config: MDConfig) -> str:
    return config.restraint_mask_override or DEFAULT_RESTRAINT_MASK


def _equilibration_restraint_weight(config: MDConfig, default_weight: float | None) -> float | None:
    if default_weight is None:
        return None
    return config.restraint_weight_override or default_weight


def _focused_restraint_mask(config: MDConfig) -> str | None:
    return config.focused_restraint_mask


def _focused_restraint_weight(config: MDConfig) -> float | None:
    return config.focused_restraint_weight


def _stage_override_candidates(stage: MDStage) -> tuple[str, ...]:
    stem = Path(stage.filename).stem
    return (stage.filename, stem, stage.title)


def _stage_override_for(config: MDConfig, stage: MDStage) -> dict[str, object]:
    overrides = config.stage_overrides or {}
    for key in _stage_override_candidates(stage):
        if key in overrides and isinstance(overrides[key], dict):
            return overrides[key]
    return {}


def _format_override_value(key: str, value: object) -> str:
    if key in _INTEGER_OVERRIDE_KEYS:
        return str(int(value))
    if key == "dt":
        dt = float(value)
        return f"{dt:.4f}" if 0.0 < dt < 0.001 else f"{dt:.3f}"
    if key in {"temp0", "tempi", "gamma_ln"}:
        return f"{float(value):.1f}"
    if key in {"pres0", "taup", "restraint_wt", "cut"}:
        return f"{float(value):.2f}"
    return str(value)


def _replace_or_insert_cntrl(content: str, key: str, value: object) -> str:
    formatted = _format_override_value(key, value)
    pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)([^,\n]+)(.*)$", re.MULTILINE)
    if pattern.search(content):
        return pattern.sub(lambda match: f"{match.group(1)}{formatted}{match.group(3)}", content, count=1)
    return content.replace("/\n", f"  {key} = {formatted}, ! GUI stage override.\n/\n", 1)


def _apply_stage_overrides(stages: list[MDStage], config: MDConfig) -> list[MDStage]:
    for stage in stages:
        overrides = _stage_override_for(config, stage)
        if not overrides:
            continue
        content = stage.content
        for key, value in overrides.items():
            if key not in _ALLOWED_OVERRIDE_KEYS or value is None or value == "":
                continue
            content = _replace_or_insert_cntrl(content, key, value)
        stage.content = content
    return stages


def _render_minimization(
    title: str,
    restraint_wt: float | None,
    mask: str,
    *,
    maxcyc: int = 5000,
    ncyc: int = 2500,
    cut: float = 10.0,
    ntxo: int | None = None,
) -> str:
    ntr = 1 if restraint_wt and restraint_wt > 0 else 0
    restraint_block = (
        "  restraint_wt = "
        f"{restraint_wt:.2f}, ! Positional-restraint force constant in kcal/mol-A^2.\n"
        f"  restraintmask = '{mask}', ! Atoms included in the positional restraint.\n"
        if ntr
        else ""
    )
    restart_block = f"  ntxo = {ntxo}, ! 1 = formatted restart; 2 = NetCDF restart.\n" if ntxo is not None else ""
    return (
        f"{title}\n"
        "&cntrl\n"
        "  imin = 1, ! 1 = energy minimization; 0 = molecular dynamics.\n"
        f"  maxcyc = {maxcyc}, ! Total number of minimization cycles.\n"
        f"  ncyc = {ncyc}, ! Switch from steepest descent to conjugate gradient after this many cycles.\n"
        f"  cut = {cut:.1f}, ! Non-bonded cutoff in Angstrom.\n"
        f"{restart_block}"
        f"  ntr = {ntr}, ! 1 = apply positional restraints; 0 = no positional restraints.\n"
        f"{restraint_block}"
        "/\n"
    )


def _render_md_stage(
    title: str,
    *,
    ensemble: str,
    nstlim: int,
    dt_ps: float,
    temp0: float,
    tempi: float,
    pressure: float,
    restraint_wt: float | None,
    mask: str,
    ntx: int,
    irest: int,
    ntt: int = 3,
    gamma_ln: float = 2.0,
    barostat: int | None = None,
    taup: float | None = None,
    iwrap: int | None = None,
    ntpr: int = 1000,
    ntwx: int = 1000,
    ntwr: int = 1000,
    ntc: int = 2,
    ntf: int = 2,
    ntxo: int | None = None,
    cut: float | None = None,
    ig: int | None = None,
) -> str:
    ntr = 1 if restraint_wt and restraint_wt > 0 else 0
    ntb = 1 if ensemble == "nvt" else 2
    ntp = 0 if ensemble == "nvt" else 1
    restraint_block = (
        f"  restraint_wt = {restraint_wt:.2f},\n  restraintmask = '{mask}',\n" if ntr else ""
    )
    pressure_block = ""
    if ensemble == "npt":
        if barostat is not None:
            pressure_block += f"  barostat = {barostat}, ! Amber barostat selector for pressure relaxation.\n"
        if taup is not None:
            pressure_block += f"  taup = {taup:.2f}, ! Pressure relaxation time in ps.\n"
    wrap_block = f"  iwrap = {iwrap}, ! Wrap molecules back into the primary box.\n" if iwrap is not None else ""
    restart_block = f"  ntxo = {ntxo}, ! 1 = formatted restart; 2 = NetCDF restart.\n" if ntxo is not None else ""
    cutoff_block = f"  cut = {cut:.1f}, ! Non-bonded cutoff in Angstrom.\n" if cut is not None else ""
    seed_block = f"  ig = {ig}, ! -1 selects an independent wall-clock Langevin seed.\n" if ig is not None else ""
    dt_text = f"{dt_ps:.4f}" if 0.0 < dt_ps < 0.001 else f"{dt_ps:.3f}"
    return (
        f"{title}\n"
        "&cntrl\n"
        "  imin = 0, ! 0 = molecular dynamics; 1 = energy minimization.\n"
        f"  ntx = {ntx}, ! 1 = read coordinates only; 5 = read coordinates and velocities from restart.\n"
        f"  irest = {irest}, ! 0 = start a new run; 1 = continue from a previous restart.\n"
        f"  nstlim = {nstlim}, ! Number of MD integration steps.\n"
        f"  dt = {dt_text}, ! Time step in ps.\n"
        f"  tempi = {tempi:.1f}, ! Initial temperature for assigned velocities in K.\n"
        f"  temp0 = {temp0:.1f}, ! Target temperature in K.\n"
        f"  ntb = {ntb}, ! 1 = constant-volume periodic box; 2 = constant-pressure periodic box.\n"
        f"  ntp = {ntp}, ! 0 = no pressure coupling; 1 = isotropic pressure coupling.\n"
        f"  pres0 = {pressure:.2f}, ! Target pressure in bar.\n"
        f"  ntc = {ntc}, ! 2 constrains bonds involving hydrogen with SHAKE.\n"
        f"  ntf = {ntf}, ! 2 omits force evaluation for SHAKE-constrained bonds.\n"
        f"  ntt = {ntt}, ! Amber thermostat selector.\n"
        f"  gamma_ln = {gamma_ln:.1f}, ! Langevin collision frequency in ps^-1.\n"
        f"{seed_block}"
        f"{cutoff_block}"
        f"{pressure_block}"
        f"  ntpr = {ntpr}, ! Print energies and progress every ntpr steps.\n"
        f"  ntwx = {ntwx}, ! Write trajectory frames every ntwx steps.\n"
        f"  ntwr = {ntwr}, ! Write restart information every ntwr steps.\n"
        f"{restart_block}"
        f"{wrap_block}"
        f"  ntr = {ntr}, ! 1 = apply positional restraints; 0 = no positional restraints.\n"
        f"{restraint_block}"
        "/\n"
    )


def _four_step(config: MDConfig) -> list[MDStage]:
    mask = _equilibration_restraint_mask(config)
    focused_mask = _focused_restraint_mask(config) or mask
    focused_weight = _focused_restraint_weight(config)
    return [
        MDStage(
            "01_min.in",
            "Restrained minimization",
            "min",
            _render_minimization("Restrained minimization", _equilibration_restraint_weight(config, 10.0), mask),
            False,
        ),
        MDStage(
            "02_nvt.in",
            "NVT heating",
            "md",
            _render_md_stage(
                "NVT heating",
                ensemble="nvt",
                nstlim=_ps_to_nstlim(100.0),
                dt_ps=0.002,
                temp0=config.temperature_k,
                tempi=0.0,
                pressure=config.pressure_bar,
                restraint_wt=_equilibration_restraint_weight(config, 10.0),
                mask=mask,
                ntx=1,
                irest=0,
                gamma_ln=2.0,
            ),
            True,
        ),
        MDStage(
            "03_npt.in",
            "NPT equilibration",
            "md",
            _render_md_stage(
                "NPT equilibration",
                ensemble="npt",
                nstlim=_ps_to_nstlim(200.0),
                dt_ps=0.002,
                temp0=config.temperature_k,
                tempi=config.temperature_k,
                pressure=config.pressure_bar,
                restraint_wt=_equilibration_restraint_weight(config, 2.0),
                mask=mask,
                ntx=5,
                irest=1,
            ),
            True,
        ),
        MDStage(
            "04_prod.in",
            "Production",
            "md",
            _render_md_stage(
                "Production",
                ensemble="npt",
                nstlim=_ns_to_nstlim(config.production_time_ns),
                dt_ps=0.002,
                temp0=config.temperature_k,
                tempi=config.temperature_k,
                pressure=config.pressure_bar,
                restraint_wt=focused_weight,
                mask=focused_mask,
                ntx=5,
                irest=1,
            ),
            True,
        ),
    ]


def _fifteen_step(config: MDConfig, *, small_molecule_only: bool = False) -> list[MDStage]:
    mask = _equilibration_restraint_mask(config)
    focused_mask = _focused_restraint_mask(config) or mask
    focused_weight = _focused_restraint_weight(config)
    stages: list[MDStage] = []
    for index, weight in enumerate([25.0, 10.0, 5.0, 2.0, 1.0], start=1):
        stages.append(
            MDStage(
                f"{index:02d}_min.in",
                f"Minimization {index}",
                "min",
                _render_minimization(f"Minimization {index}", _equilibration_restraint_weight(config, weight), mask),
                False,
            )
        )

    if small_molecule_only:
        thermal_schedule = [
            ("06_nvt_100.in", "NVT 0->100 K", "nvt", 100.0, 0.0, 1.0, 50.0, 0.001, 2.0),
            ("07_nvt_200.in", "NVT 100->200 K", "nvt", 200.0, 100.0, 0.5, 50.0, 0.001, 2.0),
            ("08_nvt_target.in", "NVT 200->target", "nvt", config.temperature_k, 200.0, 0.1, 50.0, 0.001, 2.0),
            ("09_nvt_hold.in", "NVT hold", "nvt", config.temperature_k, config.temperature_k, 0.1, 50.0, 0.001, 2.0),
            ("10_npt_relax.in", "NPT relax", "npt", config.temperature_k, config.temperature_k, 0.1, 50.0, 0.001, 2.0),
            ("11_npt_soft.in", "NPT soft", "npt", config.temperature_k, config.temperature_k, 0.05, 50.0, 0.001, 2.0),
        ]
    else:
        thermal_schedule = [
            ("06_nvt_100.in", "NVT 0->100 K", "nvt", 100.0, 0.0, 1.0, 50.0, 0.002, 2.0),
            ("07_nvt_200.in", "NVT 100->200 K", "nvt", 200.0, 100.0, 0.5, 50.0, 0.002, 2.0),
            ("08_nvt_target.in", "NVT 200->target", "nvt", config.temperature_k, 200.0, 0.1, 50.0, 0.002, 2.0),
            ("09_nvt_hold.in", "NVT hold", "nvt", config.temperature_k, config.temperature_k, 0.1, 50.0, 0.002, 2.0),
            ("10_npt_relax.in", "NPT relax", "npt", config.temperature_k, config.temperature_k, 0.1, 50.0, 0.002, 2.0),
            ("11_npt_soft.in", "NPT soft", "npt", config.temperature_k, config.temperature_k, 0.05, 50.0, 0.002, 2.0),
        ]
    for index, (filename, title, ensemble, temp0, tempi, weight, duration_ps, dt_ps, gamma_ln) in enumerate(thermal_schedule, start=6):
        stages.append(
            MDStage(
                filename,
                title,
                "md",
                _render_md_stage(
                    title,
                    ensemble=ensemble,
                    nstlim=_ps_to_nstlim(duration_ps, dt_ps=dt_ps),
                    dt_ps=dt_ps,
                    temp0=temp0,
                    tempi=tempi,
                    pressure=config.pressure_bar,
                    restraint_wt=_equilibration_restraint_weight(config, weight),
                    mask=mask,
                    ntx=1 if index == 6 else 5,
                    irest=0 if index == 6 else 1,
                    gamma_ln=gamma_ln,
                ),
                True,
            )
        )

    stages.extend(
        [
            MDStage("12_unrestrained_min.in", "Unrestrained minimization", "min", _render_minimization("Unrestrained minimization", None, mask), False),
            MDStage(
                "13_unrestrained_nvt.in",
                "Unrestrained NVT",
                "md",
                _render_md_stage(
                    "Unrestrained NVT",
                    ensemble="nvt",
                    nstlim=_ps_to_nstlim(100.0, dt_ps=0.002 if small_molecule_only else 0.001),
                    dt_ps=0.002 if small_molecule_only else 0.001,
                    temp0=config.temperature_k,
                    tempi=0.0,
                    pressure=config.pressure_bar,
                    restraint_wt=focused_weight,
                    mask=focused_mask,
                    ntx=1,
                    irest=0,
                ),
                True,
            ),
            MDStage(
                "14_unrestrained_npt.in",
                "Unrestrained NPT",
                "md",
                _render_md_stage(
                    "Unrestrained NPT",
                    ensemble="npt",
                    nstlim=_ps_to_nstlim(100.0, dt_ps=0.002),
                    dt_ps=0.002,
                    temp0=config.temperature_k,
                    tempi=config.temperature_k,
                    pressure=config.pressure_bar,
                    restraint_wt=focused_weight,
                    mask=focused_mask,
                    ntx=5,
                    irest=1,
                ),
                True,
            ),
            MDStage(
                "15_prod.in",
                "Production",
                "md",
                _render_md_stage(
                    "Production",
                    ensemble="npt",
                    nstlim=_ns_to_nstlim(config.production_time_ns, dt_ps=0.002),
                    dt_ps=0.002,
                    temp0=config.temperature_k,
                    tempi=config.temperature_k,
                    pressure=config.pressure_bar,
                    restraint_wt=focused_weight,
                    mask=focused_mask,
                    ntx=5,
                    irest=1,
                ),
                True,
            ),
        ]
    )
    return stages


def _des_solvent(config: MDConfig) -> list[MDStage]:
    mask = _equilibration_restraint_mask(config)
    stages = [
        MDStage(
            "01_min.in",
            "Unrestrained DES minimization",
            "min",
            _render_minimization(
                "Unrestrained DES minimization",
                None,
                mask,
                maxcyc=10000,
                ncyc=5000,
                cut=8.0,
                ntxo=1,
            ),
            False,
        ),
        MDStage(
            "02_settle_noshake.in",
            "DES low-temperature no-SHAKE settle",
            "md",
            _render_md_stage(
                "DES low-temperature no-SHAKE settle",
                ensemble="nvt",
                nstlim=100,
                dt_ps=0.0001,
                temp0=1.0,
                tempi=1.0,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=1,
                irest=0,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntc=1,
                ntf=1,
                ntxo=1,
                iwrap=1,
                ntpr=10,
                ntwx=0,
                ntwr=100,
            ),
            False,
        ),
        MDStage(
            "03_warm_nvt.in",
            "DES NVT warm-up",
            "md",
            _render_md_stage(
                "DES NVT warm-up",
                ensemble="nvt",
                nstlim=_ps_to_nstlim(50.0, dt_ps=0.001),
                dt_ps=0.001,
                temp0=50.0,
                tempi=1.0,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=5,
                irest=1,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntxo=1,
                iwrap=1,
                ntpr=1000,
                ntwx=1000,
                ntwr=1000,
            ),
            True,
        ),
        MDStage(
            "04_heat_nvt.in",
            "DES NVT heating",
            "md",
            _render_md_stage(
                "DES NVT heating",
                ensemble="nvt",
                nstlim=_ps_to_nstlim(200.0, dt_ps=0.001),
                dt_ps=0.001,
                temp0=config.temperature_k,
                tempi=50.0,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=5,
                irest=1,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntxo=1,
                iwrap=1,
            ),
            True,
        ),
        MDStage(
            "05_density_soft_npt.in",
            "DES soft NPT density relaxation",
            "md",
            _render_md_stage(
                "DES soft NPT density relaxation",
                ensemble="npt",
                nstlim=_ps_to_nstlim(250.0, dt_ps=0.001),
                dt_ps=0.001,
                temp0=config.temperature_k,
                tempi=config.temperature_k,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=5,
                irest=1,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntxo=1,
                barostat=1,
                taup=0.5,
                iwrap=1,
            ),
            True,
        ),
        MDStage(
            "06_equil_npt.in",
            "DES NPT equilibration",
            "md",
            _render_md_stage(
                "DES NPT equilibration",
                ensemble="npt",
                nstlim=_ps_to_nstlim(1000.0, dt_ps=0.002),
                dt_ps=0.002,
                temp0=config.temperature_k,
                tempi=config.temperature_k,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=5,
                irest=1,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntxo=1,
                barostat=1,
                taup=1.0,
                iwrap=1,
            ),
            True,
        ),
    ]
    production_filename = "07_prod.in"
    if config.des_mixing_enabled:
        stages.append(
            MDStage(
                "07_mix_500k_npt.in",
                "DES 500 K NPT mixing",
                "md",
                _render_md_stage(
                    "DES 500 K NPT mixing",
                    ensemble="npt",
                    nstlim=_ns_to_nstlim(config.des_mixing_time_ns, dt_ps=0.002),
                    dt_ps=0.002,
                    temp0=config.des_mixing_temperature_k,
                    tempi=config.temperature_k,
                    pressure=config.pressure_bar,
                    restraint_wt=None,
                    mask=mask,
                    ntx=5,
                    irest=1,
                    gamma_ln=2.0,
                    ig=-1,
                    cut=8.0,
                    ntxo=1,
                    barostat=1,
                    taup=1.0,
                    iwrap=1,
                    ntpr=25000,
                    ntwx=25000,
                    ntwr=50000,
                ),
                True,
            )
        )
        production_filename = "08_prod.in"
    stages.append(
        MDStage(
            production_filename,
            "DES NPT production",
            "md",
            _render_md_stage(
                "DES NPT production",
                ensemble="npt",
                nstlim=_ns_to_nstlim(config.production_time_ns, dt_ps=0.002),
                dt_ps=0.002,
                temp0=config.temperature_k,
                tempi=config.temperature_k,
                pressure=config.pressure_bar,
                restraint_wt=None,
                mask=mask,
                ntx=5,
                irest=1,
                gamma_ln=2.0,
                ig=-1,
                cut=8.0,
                ntxo=1,
                barostat=1,
                taup=1.0,
                iwrap=1,
                ntpr=2500,
                ntwx=2500,
                ntwr=5000,
            ),
            True,
        )
    )
    return stages


def generate_md_inputs(
    config: MDConfig,
    output_dir: Path,
    *,
    small_molecule_only: bool = False,
    des_solvent: bool = False,
) -> list[MDStage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if des_solvent or config.protocol == ProtocolKind.DES_SOLVENT:
        stages = _des_solvent(config)
    elif config.protocol == ProtocolKind.FOUR_STEP:
        stages = _four_step(config)
    else:
        stages = _fifteen_step(config, small_molecule_only=small_molecule_only)
    stages = _apply_stage_overrides(stages, config)
    for stage in stages:
        (output_dir / stage.filename).write_text(stage.content, encoding="utf-8")
    write_json(output_dir / "md_manifest.json", {"stages": [stage.to_dict() for stage in stages]})
    return stages
