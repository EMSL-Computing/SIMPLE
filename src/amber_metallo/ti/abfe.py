from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from amber_metallo.reporting import console, write_json
from amber_metallo.subdirectory_search import search_subdirectories_enabled

try:
    from rich import box
    from rich.table import Table
except ModuleNotFoundError:
    box = None
    Table = None


CASE_TYPE_BOUND = "bound"
CASE_TYPE_WATER = "water"
SOURCE_KIND_SIMULATION = "simulation"
SOURCE_KIND_LIBRARY = "library"
METHOD_TI = "ti"
SAMPLING_SELECTION_FORWARD_REVERSE = "forward_reverse"
SAMPLING_SELECTION_FORWARD_ONLY = "forward_only"
SAMPLING_SELECTION_LEGACY_UNSPECIFIED = "legacy_unspecified"
DECOUPLING_SCHEME_COMBINED = "Combined"
DECOUPLING_SCHEME_SPLIT = "Split"
DECOUPLING_SCHEME_MIXED = "Mixed"
DECOUPLING_SCHEME_UNKNOWN = "Unknown"
_DEFAULT_BLOCK_COUNT = 5
_DEFAULT_BOOTSTRAP_ITERATIONS = 1000
_FINAL_AVERAGE_PATTERN = re.compile(
    r"\bDV/?DL\b\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)
_TIME_SERIES_PATTERN = re.compile(
    r"\bNSTEP\b.*?\bDV/?DL\b\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)
_DVDL_LINE_PATTERN = re.compile(
    r"\bDV/?DL\b\s*=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)
_STARRED_DVDL_PATTERN = re.compile(r"\bDV/?DL\b\s*=\s*\*+", re.IGNORECASE)


@dataclass(slots=True)
class ConfidenceInterval:
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}


@dataclass(slots=True)
class ParsedDVDL:
    value: float
    parser_mode: str
    warning: str | None
    sample_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "parser_mode": self.parser_mode,
            "warning": self.warning,
            "sample_values": self.sample_values,
        }


@dataclass(slots=True)
class TIWindowExpectation:
    phase: str
    clambda: float
    input_path: Path
    output_path: Path
    title: str
    replica: int = 1
    direction: str = "forward"
    run_id: str = "legacy"
    substitute_output_path: Path | None = None
    substitution_reason: str | None = None

    @property
    def analysis_output_path(self) -> Path:
        return self.substitute_output_path or self.output_path

    @property
    def endpoint_substituted(self) -> bool:
        return self.substitute_output_path is not None


def _decoupling_scheme_from_phase_counts(*, qoff_count: int, vdwoff_count: int) -> str:
    if qoff_count > 0 and vdwoff_count > 0:
        return DECOUPLING_SCHEME_SPLIT
    if qoff_count > 0 and vdwoff_count == 0:
        return DECOUPLING_SCHEME_COMBINED
    return DECOUPLING_SCHEME_UNKNOWN


def _decoupling_scheme_from_expectations(expectations: list[TIWindowExpectation]) -> str:
    return _decoupling_scheme_from_phase_counts(
        qoff_count=sum(1 for item in expectations if item.phase == "qoff"),
        vdwoff_count=sum(1 for item in expectations if item.phase == "vdwoff"),
    )


@dataclass(slots=True)
class AnalysisCaseDiscovery:
    root: Path
    case_type: str
    display_name: str
    description: str
    completion_summary: str
    readiness_note: str
    selectable: bool
    source_kind: str = SOURCE_KIND_SIMULATION
    library_key: str | None = None
    library_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        return payload


def decoupling_scheme_for_case(case: AnalysisCaseDiscovery) -> str:
    scheme = str(case.metadata.get("ti_decoupling_scheme") or "").strip()
    return scheme or DECOUPLING_SCHEME_UNKNOWN


def case_has_bidirectional_sampling(case: AnalysisCaseDiscovery) -> bool:
    protocol = case.metadata.get("ti_sampling_protocol") or {}
    mode = str(protocol.get("mode") or "").strip().lower()
    directions = {str(item).strip().lower() for item in (protocol.get("directions") or [])}
    return mode in {"bidirectional", "replicated_bidirectional"} or "reverse" in directions


def _normalize_sampling_selection(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {SAMPLING_SELECTION_FORWARD_REVERSE, SAMPLING_SELECTION_FORWARD_ONLY}:
        raise ValueError(
            "sampling_selection must be 'forward_reverse' or 'forward_only'."
        )
    return normalized


def _charge_compensation_mode(case: AnalysisCaseDiscovery) -> str:
    value = str(case.metadata.get("charge_compensation_mode") or "unknown").strip().lower()
    return value or "unknown"


def rbfe_pair_compatibility(
    bound_case: AnalysisCaseDiscovery,
    water_case: AnalysisCaseDiscovery,
) -> tuple[list[str], list[str]]:
    """Return blocking errors and non-blocking warnings for an RBFE leg pair."""
    errors: list[str] = []
    warnings: list[str] = []
    bound_charge_mode = _charge_compensation_mode(bound_case)
    water_charge_mode = _charge_compensation_mode(water_case)
    known_charge_modes = {bound_charge_mode, water_charge_mode} - {"unknown", ""}
    if len(known_charge_modes) > 1:
        errors.append(
            "Bound and water legs use different charge-compensation modes "
            f"({bound_charge_mode} vs {water_charge_mode}). Use a water reference generated with the same TI protocol."
        )

    bound_scheme = decoupling_scheme_for_case(bound_case)
    water_scheme = decoupling_scheme_for_case(water_case)
    if (
        bound_scheme not in {DECOUPLING_SCHEME_UNKNOWN, DECOUPLING_SCHEME_MIXED}
        and water_scheme not in {DECOUPLING_SCHEME_UNKNOWN, DECOUPLING_SCHEME_MIXED}
        and bound_scheme != water_scheme
    ):
        errors.append(
            "Bound and water legs use different TI decoupling schemes "
            f"({bound_scheme} vs {water_scheme}). Select a matching water reference."
        )

    if bound_charge_mode == "co_alchemical_counterions":
        for label, case in (("Bound", bound_case), ("Water", water_case)):
            validation = case.metadata.get("neutrality_validation") or {}
            status = str(validation.get("status") or "missing").strip().lower()
            if status not in {"passed", "missing"}:
                errors.append(f"{label} charge-neutrality validation status is '{status}', not 'passed'.")
            elif status == "missing":
                warnings.append(
                    f"{label} case has no recorded charge-neutrality validation; inspect its manifest before trusting ddG."
                )

    if case_has_bidirectional_sampling(bound_case) != case_has_bidirectional_sampling(water_case):
        warnings.append(
            "Only one RBFE leg has reverse-sweep data; the two legs therefore use different sampling depth."
        )

    bound_schedule = bound_case.metadata.get("ti_lambda_schedule") or {}
    water_schedule = water_case.metadata.get("ti_lambda_schedule") or {}
    if bound_schedule and water_schedule and bound_schedule != water_schedule:
        warnings.append("Bound and water legs use different lambda schedules; verify that this was intentional.")
    return errors, warnings


@dataclass(slots=True)
class AnalysisWindowResult:
    phase: str
    clambda: float
    mdout_path: Path
    delta_g_source_value: float
    sample_mean_dvdl: float
    sample_std_dvdl: float
    sem_dvdl: float
    sem_mode: str
    sample_count: int
    block_count: int
    parser_mode: str
    quality: Literal["ok", "warning"]
    warning: str | None = None
    replica: int = 1
    direction: str = "forward"
    run_id: str = "legacy"
    expected_mdout_path: Path | None = None
    endpoint_substituted: bool = False
    bootstrap_pool: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "clambda": self.clambda,
            "mdout_path": str(self.mdout_path),
            "delta_g_source_value": self.delta_g_source_value,
            "sample_mean_dvdl": self.sample_mean_dvdl,
            "sample_std_dvdl": self.sample_std_dvdl,
            "sem_dvdl": self.sem_dvdl,
            "sem_mode": self.sem_mode,
            "sample_count": self.sample_count,
            "block_count": self.block_count,
            "parser_mode": self.parser_mode,
            "quality": self.quality,
            "warning": self.warning,
            "replica": self.replica,
            "direction": self.direction,
            "run_id": self.run_id,
            "expected_mdout_path": None if self.expected_mdout_path is None else str(self.expected_mdout_path),
            "endpoint_substituted": self.endpoint_substituted,
        }


@dataclass(slots=True)
class PhaseAnalysis:
    phase: str
    delta_g_kcal_mol: float
    propagated_sem_kcal_mol: float
    bootstrap_ci95: ConfidenceInterval
    windows: list[AnalysisWindowResult]
    bootstrap_samples: list[float] = field(default_factory=list, repr=False)
    quality: Literal["ok", "warning"] = "ok"
    warnings: list[str] = field(default_factory=list)
    sampling_runs: list[dict[str, Any]] = field(default_factory=list)
    forward_reverse_difference_kcal_mol: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "delta_g_kcal_mol": self.delta_g_kcal_mol,
            "propagated_sem_kcal_mol": self.propagated_sem_kcal_mol,
            "bootstrap_ci95": self.bootstrap_ci95.to_dict(),
            "quality": self.quality,
            "warnings": self.warnings,
            "sampling_runs": self.sampling_runs,
            "forward_reverse_difference_kcal_mol": self.forward_reverse_difference_kcal_mol,
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(slots=True)
class SingleCaseAnalysisResult:
    case: AnalysisCaseDiscovery
    qoff: PhaseAnalysis
    vdwoff: PhaseAnalysis
    delta_g_kcal_mol: float
    propagated_sem_kcal_mol: float
    bootstrap_ci95: ConfidenceInterval
    output_dir: Path
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY
    restraint_correction_kcal_mol: float | None = None
    corrected_delta_g_kcal_mol: float | None = None
    corrected_propagated_sem_kcal_mol: float | None = None
    corrected_bootstrap_ci95: ConfidenceInterval | None = None
    quality: Literal["ok", "warning"] = "ok"
    warnings: list[str] = field(default_factory=list)
    bootstrap_samples: list[float] = field(default_factory=list, repr=False)
    corrected_bootstrap_samples: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "ti_decoupling_scheme": decoupling_scheme_for_result(self),
            "qoff": self.qoff.to_dict(),
            "vdwoff": self.vdwoff.to_dict(),
            "delta_g_kcal_mol": self.delta_g_kcal_mol,
            "propagated_sem_kcal_mol": self.propagated_sem_kcal_mol,
            "bootstrap_ci95": self.bootstrap_ci95.to_dict(),
            "sampling_selection": self.sampling_selection,
            "restraint_correction_kcal_mol": self.restraint_correction_kcal_mol,
            "corrected_delta_g_kcal_mol": self.corrected_delta_g_kcal_mol,
            "corrected_propagated_sem_kcal_mol": self.corrected_propagated_sem_kcal_mol,
            "corrected_bootstrap_ci95": (
                None if self.corrected_bootstrap_ci95 is None else self.corrected_bootstrap_ci95.to_dict()
            ),
            "quality": self.quality,
            "warnings": self.warnings,
            "output_dir": str(self.output_dir),
        }


@dataclass(slots=True)
class RBFEAnalysisResult:
    bound: SingleCaseAnalysisResult
    water: SingleCaseAnalysisResult
    ddg_kcal_mol: float
    propagated_sem_kcal_mol: float
    bootstrap_ci95: ConfidenceInterval
    output_dir: Path
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY
    quality: Literal["ok", "warning"] = "ok"
    warnings: list[str] = field(default_factory=list)
    bootstrap_samples: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bound": self.bound.to_dict(),
            "water": self.water.to_dict(),
            "ddg_kcal_mol": self.ddg_kcal_mol,
            "propagated_sem_kcal_mol": self.propagated_sem_kcal_mol,
            "bootstrap_ci95": self.bootstrap_ci95.to_dict(),
            "sampling_selection": self.sampling_selection,
            "quality": self.quality,
            "warnings": self.warnings,
            "output_dir": str(self.output_dir),
        }


def decoupling_scheme_for_result(result: SingleCaseAnalysisResult) -> str:
    scheme = decoupling_scheme_for_case(result.case)
    if scheme != DECOUPLING_SCHEME_UNKNOWN:
        return scheme
    return _decoupling_scheme_from_phase_counts(
        qoff_count=len(result.qoff.windows),
        vdwoff_count=len(result.vdwoff.windows),
    )


def _decoupling_scheme_detail(scheme: str) -> str:
    if scheme == DECOUPLING_SCHEME_COMBINED:
        return "Combined (single softcore path)"
    if scheme == DECOUPLING_SCHEME_SPLIT:
        return "Split (qoff + vdwoff)"
    return scheme


def _sampling_selection_detail(selection: str) -> str:
    if selection == SAMPLING_SELECTION_FORWARD_ONLY:
        return "Forward only (default)"
    if selection == SAMPLING_SELECTION_LEGACY_UNSPECIFIED:
        return "Legacy library value (direction metadata unavailable)"
    return "Forward + Reverse (optional convergence diagnostic)"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def analysis_library_root() -> Path:
    return repo_root() / "analysis_library"


def water_ref_library_path() -> Path:
    return analysis_library_root() / "water_ref_library.json"


def bound_library_path() -> Path:
    return analysis_library_root() / "bound_library.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_library_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json(path, payload)


def _coerce_float(raw: str) -> float:
    return float(raw.replace("D", "E").replace("d", "e"))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.fsum(values) / float(len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _sample_sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return _sample_std(values) / math.sqrt(float(len(values)))


def _contiguous_block_means(values: list[float], *, block_count: int = _DEFAULT_BLOCK_COUNT) -> list[float]:
    if not values:
        return []
    resolved_blocks = min(block_count, len(values))
    if resolved_blocks <= 1:
        return [_safe_mean(values)]
    base_size, remainder = divmod(len(values), resolved_blocks)
    means: list[float] = []
    offset = 0
    for index in range(resolved_blocks):
        size = base_size + (1 if index < remainder else 0)
        block = values[offset : offset + size]
        if block:
            means.append(_safe_mean(block))
        offset += size
    return means


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _confidence_interval(values: list[float]) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(0.0, 0.0)
    return ConfidenceInterval(_quantile(values, 0.025), _quantile(values, 0.975))


def _stable_seed(label: str) -> int:
    normalized = label.encode("utf-8")
    total = 0
    for index, value in enumerate(normalized, start=1):
        total += index * value
    return total % (2**31 - 1)


def water_library_key(metal: str, formal_charge: int, water_model: str) -> str:
    metal_token = re.sub(r"[^a-z0-9]+", "", metal.lower())
    water_token = re.sub(r"[^a-z0-9]+", "_", water_model.lower()).strip("_")
    return f"{metal_token}{formal_charge}_{water_token}"


def _phase_metric_payload(delta_g: float, sem: float, ci95: ConfidenceInterval) -> dict[str, Any]:
    return {
        "delta_g_kcal_mol": delta_g,
        "propagated_sem_kcal_mol": sem,
        "bootstrap_ci95": ci95.to_dict(),
    }


def _format_ci95(ci95: ConfidenceInterval | None) -> str:
    if ci95 is None:
        return "N/A"
    return f"[{ci95.low:.6f}, {ci95.high:.6f}]"


def _format_mean_sem(value: float | None, sem: float | None) -> str:
    if value is None:
        return "N/A"
    if sem is None:
        return f"{value:.6f}"
    return f"{value:.6f} +/- {sem:.6f}"


def _case_max_hysteresis(result: SingleCaseAnalysisResult) -> float | None:
    values = [
        phase.forward_reverse_difference_kcal_mol
        for phase in (result.qoff, result.vdwoff)
        if phase.forward_reverse_difference_kcal_mol is not None
    ]
    return max(values) if values else None


def _trapezoid_weights(lambdas: list[float]) -> list[float]:
    if not lambdas:
        return []
    if len(lambdas) == 1:
        return [1.0]
    weights: list[float] = []
    for index, current in enumerate(lambdas):
        if index == 0:
            weights.append((lambdas[1] - current) / 2.0)
        elif index == len(lambdas) - 1:
            weights.append((current - lambdas[index - 1]) / 2.0)
        else:
            weights.append((lambdas[index + 1] - lambdas[index - 1]) / 2.0)
    return weights


def integrate_trapezoid(points: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(lam), float(value)) for lam, value in points)
    if len(ordered) < 2:
        return ordered[0][1] if ordered else 0.0
    total = 0.0
    for (lam_a, value_a), (lam_b, value_b) in zip(ordered[:-1], ordered[1:]):
        total += (lam_b - lam_a) * (value_a + value_b) / 2.0
    return total


def parse_mdout_dvdl(path: str | Path) -> ParsedDVDL:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="ignore")
    if _STARRED_DVDL_PATTERN.search(text):
        raise ValueError(
            f"{target} contains overflowed DV/DL values ('**************'). "
            "This usually indicates a numerically unstable TI window, so the result should not be analyzed."
        )
    lines = text.splitlines()
    average_value: float | None = None
    average_index: int | None = None
    for index, line in enumerate(lines):
        if "A V E R A G E S" not in line.upper():
            continue
        average_index = index
        for trailing in lines[index + 1 :]:
            match = _FINAL_AVERAGE_PATTERN.search(trailing)
            if match:
                average_value = _coerce_float(match.group("value"))
                break
        if average_value is not None:
            break
    sample_values = [_coerce_float(match.group("value")) for match in _TIME_SERIES_PATTERN.finditer(text)]
    if not sample_values:
        sample_scan_lines = lines if average_index is None else lines[:average_index]
        sample_values = []
        for line in sample_scan_lines:
            match = _DVDL_LINE_PATTERN.search(line)
            if match:
                sample_values.append(_coerce_float(match.group("value")))
    if average_value is not None:
        return ParsedDVDL(
            value=average_value,
            parser_mode="final_average_block",
            warning=None,
            sample_values=sample_values,
        )
    if sample_values:
        return ParsedDVDL(
            value=_safe_mean(sample_values),
            parser_mode="time_series_mean",
            warning="Fell back to the DV/DL time-series mean because the final average block was not found.",
            sample_values=sample_values,
        )
    raise ValueError(f"Could not locate DV/DL data in {target}")


def _window_statistics(parse: ParsedDVDL) -> tuple[float, float, float, str, int, int, list[float]]:
    if parse.sample_values:
        samples = list(parse.sample_values)
    else:
        samples = [parse.value]
    sample_mean = _safe_mean(samples)
    sample_std = _sample_std(samples)
    if len(samples) >= (_DEFAULT_BLOCK_COUNT * 2):
        block_means = _contiguous_block_means(samples)
        sem = _sample_sem(block_means)
        sem_mode = "block_average"
        block_count = len(block_means)
        bootstrap_pool = block_means
    else:
        sem = _sample_sem(samples)
        sem_mode = "sample_sem"
        block_count = 1
        bootstrap_pool = samples
    return sample_mean, sample_std, sem, sem_mode, len(samples), block_count, bootstrap_pool


def _load_expected_windows(ti_manifest_path: Path, output_root: Path) -> list[TIWindowExpectation]:
    payload = _load_json(ti_manifest_path)
    windows = payload.get("windows") or []
    flat_combined_layout = payload.get("output_layout") == "combined_flat"
    expectations: list[TIWindowExpectation] = []
    runs = payload.get("runs") or []
    sampling = payload.get("sampling_protocol") or {}
    if str(sampling.get("mode") or "single_pass") == "single_pass":
        runs = []
    resolved_runs = runs or [{"replica": 1, "direction": "forward", "run_id": "legacy", "output_root": ""}]
    for run in resolved_runs:
        if not isinstance(run, dict):
            continue
        run_output_root = Path(str(run.get("output_root") or ""))
        replica = int(run.get("replica") or 1)
        direction = str(run.get("direction") or "forward")
        run_id = str(run.get("run_id") or f"replica_{replica:02d}/{direction}")
        for item in windows:
            if not isinstance(item, dict):
                continue
            input_path = Path(str(item.get("filename", "")))
            phase = str(item.get("phase") or "").strip()
            if not phase or not input_path.name:
                continue
            run_root = output_root / run_output_root
            output_path = (
                run_root / f"{input_path.stem}.out"
                if flat_combined_layout
                else run_root / phase / f"{input_path.stem}.out"
            )
            expectations.append(
                TIWindowExpectation(
                    phase=phase,
                    clambda=float(item.get("clambda", 0.0)),
                    input_path=input_path,
                    output_path=output_path,
                    title=str(item.get("title") or input_path.stem),
                    replica=replica,
                    direction=direction,
                    run_id=run_id,
                )
            )
    _configure_bidirectional_endpoint_substitution(
        expectations,
        sampling_mode=str(sampling.get("mode") or "single_pass"),
    )
    return expectations


def _lambda_schedule_payload(expectations: list[TIWindowExpectation]) -> dict[str, list[float]]:
    return {
        phase: sorted({item.clambda for item in expectations if item.phase == phase})
        for phase in ("qoff", "vdwoff")
    }


def _has_complete_final_average(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return parse_mdout_dvdl(path).parser_mode == "final_average_block"
    except (OSError, ValueError):
        return False


def _configure_bidirectional_endpoint_substitution(
    expectations: list[TIWindowExpectation],
    *,
    sampling_mode: str,
) -> None:
    if sampling_mode not in {"bidirectional", "replicated_bidirectional"}:
        return
    grouped: dict[tuple[str, int], list[TIWindowExpectation]] = {}
    for expectation in expectations:
        grouped.setdefault((expectation.phase, expectation.replica), []).append(expectation)
    for group in grouped.values():
        reverse_zero = [
            item
            for item in group
            if item.direction == "reverse" and math.isclose(item.clambda, 0.0, abs_tol=1.0e-12)
        ]
        if len(reverse_zero) != 1:
            continue
        target = reverse_zero[0]
        if _has_complete_final_average(target.output_path):
            continue
        # This fallback is intentionally narrow: every other production window
        # must exist, and only the terminal reverse lambda=0 result may fail.
        if any(not item.output_path.exists() for item in group if item is not target):
            continue
        forward_zero = [
            item
            for item in group
            if item.direction == "forward" and math.isclose(item.clambda, 0.0, abs_tol=1.0e-12)
        ]
        if len(forward_zero) != 1 or not _has_complete_final_average(forward_zero[0].output_path):
            continue
        target.substitute_output_path = forward_zero[0].output_path
        target.substitution_reason = (
            "Reverse lambda=0 production output is missing or incomplete; "
            "the completed forward lambda=0 DV/DL was substituted at the identical Hamiltonian endpoint."
        )


def _endpoint_substitution_payload(expectations: list[TIWindowExpectation]) -> list[dict[str, Any]]:
    return [
        {
            "phase": item.phase,
            "replica": item.replica,
            "direction": item.direction,
            "lambda": item.clambda,
            "expected_output_path": str(item.output_path),
            "substitute_output_path": str(item.analysis_output_path),
            "reason": item.substitution_reason,
        }
        for item in expectations
        if item.endpoint_substituted
    ]


def _summarize_completion(expectations: list[TIWindowExpectation]) -> tuple[str, bool]:
    phase_counts: dict[str, tuple[int, int]] = {}
    complete = True
    for phase in ("qoff", "vdwoff"):
        phase_windows = [item for item in expectations if item.phase == phase]
        expected = len(phase_windows)
        present = sum(1 for item in phase_windows if item.output_path.exists() or item.endpoint_substituted)
        phase_counts[phase] = (present, expected)
        if present != expected:
            complete = False
    summary = ", ".join(
        f"{phase} {phase_counts[phase][0]}/{phase_counts[phase][1]}" for phase in ("qoff", "vdwoff")
    )
    substitution_count = sum(1 for item in expectations if item.endpoint_substituted)
    if substitution_count:
        summary += f" ({substitution_count} reverse endpoint substituted)"
    return summary, complete


def _readiness_note(expectations: list[TIWindowExpectation], *, complete: bool) -> str:
    if not complete:
        return "Incomplete TI outputs"
    if any(item.endpoint_substituted for item in expectations):
        return "Ready with approximation: reverse lambda=0 uses the completed forward endpoint"
    return "Ready"


def _bootstrap_integral(
    windows: list[AnalysisWindowResult],
    *,
    seed_label: str,
    iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS,
) -> list[float]:
    if not windows:
        return []
    rng = random.Random(_stable_seed(seed_label))
    ordered = sorted(windows, key=lambda item: item.clambda)
    samples: list[float] = []
    for _ in range(iterations):
        dvdl_points: list[tuple[float, float]] = []
        for window in ordered:
            dvdl_points.append((window.clambda, _resampled_window_mean(window, rng)))
        samples.append(integrate_trapezoid(dvdl_points))
    return samples


def _resampled_window_mean(window: AnalysisWindowResult, rng: random.Random) -> float:
    pool = window.bootstrap_pool or [window.delta_g_source_value]
    picked = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
    # The Amber final-average DV/DL is the point estimate used for integration.
    # Recenter block resamples on that same value so the CI and point estimate
    # cannot silently describe different estimators.
    return _safe_mean(picked) + window.delta_g_source_value - _safe_mean(pool)


def _bootstrap_grouped_mean(
    grouped: dict[str, list[AnalysisWindowResult]],
    *,
    seed_label: str,
    iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS,
) -> list[float]:
    if not grouped:
        return []
    rng = random.Random(_stable_seed(seed_label))
    samples: list[float] = []
    for _ in range(iterations):
        shared_window_means: dict[str, float] = {}
        run_integrals: list[float] = []
        for run_windows in grouped.values():
            points: list[tuple[float, float]] = []
            for window in sorted(run_windows, key=lambda item: item.clambda):
                source_key = str(window.mdout_path.resolve())
                if source_key not in shared_window_means:
                    shared_window_means[source_key] = _resampled_window_mean(window, rng)
                points.append((window.clambda, shared_window_means[source_key]))
            run_integrals.append(integrate_trapezoid(points))
        # Resample complete sweeps as well as the within-window block means.
        # Without this second level, a forward/reverse disagreement contributes
        # to the reported SEM but is completely absent from the bootstrap CI.
        picked_runs = [run_integrals[rng.randrange(len(run_integrals))] for _ in range(len(run_integrals))]
        samples.append(_safe_mean(picked_runs))
    return samples


def _combine_bootstrap_samples(
    first: list[float],
    second: list[float],
    *,
    operation: Literal["add", "subtract"],
    constant: float = 0.0,
) -> list[float]:
    if not first and not second:
        return []
    if not first:
        first = [0.0]
    if not second:
        second = [0.0]
    size = min(len(first), len(second))
    if size <= 0:
        size = max(len(first), len(second))
    combined: list[float] = []
    for index in range(size):
        left = first[index % len(first)]
        right = second[index % len(second)]
        value = left + right if operation == "add" else left - right
        combined.append(value + constant)
    return combined


def _phase_analysis(phase: str, expectations: list[TIWindowExpectation], *, seed_label: str) -> PhaseAnalysis:
    phase_windows = sorted((item for item in expectations if item.phase == phase), key=lambda item: item.clambda)
    if not phase_windows:
        return PhaseAnalysis(
            phase=phase,
            delta_g_kcal_mol=0.0,
            propagated_sem_kcal_mol=0.0,
            bootstrap_ci95=ConfidenceInterval(0.0, 0.0),
            windows=[],
        )
    parsed_windows: list[AnalysisWindowResult] = []
    warnings: list[str] = []
    quality: Literal["ok", "warning"] = "ok"
    for expectation in phase_windows:
        analysis_output_path = expectation.analysis_output_path
        parsed = parse_mdout_dvdl(analysis_output_path)
        sample_mean, sample_std, sem, sem_mode, sample_count, block_count, bootstrap_pool = _window_statistics(parsed)
        warning_parts = [item for item in (expectation.substitution_reason, parsed.warning) if item]
        warning = " ".join(warning_parts) or None
        window_quality: Literal["ok", "warning"] = "ok"
        if expectation.endpoint_substituted or parsed.parser_mode != "final_average_block" or sem_mode != "block_average":
            window_quality = "warning"
        if warning:
            warnings.append(f"{expectation.output_path}: {warning}")
        if window_quality == "warning":
            quality = "warning"
        parsed_windows.append(
            AnalysisWindowResult(
                phase=phase,
                clambda=expectation.clambda,
                mdout_path=analysis_output_path,
                delta_g_source_value=parsed.value,
                sample_mean_dvdl=sample_mean,
                sample_std_dvdl=sample_std,
                sem_dvdl=sem,
                sem_mode=sem_mode,
                sample_count=sample_count,
                block_count=block_count,
                parser_mode=parsed.parser_mode,
                quality=window_quality,
                warning=warning,
                bootstrap_pool=bootstrap_pool,
                replica=expectation.replica,
                direction=expectation.direction,
                run_id=expectation.run_id,
                expected_mdout_path=expectation.output_path if expectation.endpoint_substituted else None,
                endpoint_substituted=expectation.endpoint_substituted,
            )
        )
    grouped: dict[str, list[AnalysisWindowResult]] = {}
    for window in parsed_windows:
        grouped.setdefault(window.run_id, []).append(window)
    if len(grouped) > 1:
        run_summaries: list[dict[str, Any]] = []
        for run_id, run_windows in grouped.items():
            ordered = sorted(run_windows, key=lambda item: item.clambda)
            value = integrate_trapezoid((item.clambda, item.delta_g_source_value) for item in ordered)
            weights = _trapezoid_weights([item.clambda for item in ordered])
            within_sem = math.sqrt(
                math.fsum((weight * item.sem_dvdl) ** 2 for weight, item in zip(weights, ordered))
            )
            run_summaries.append(
                {
                    "run_id": run_id,
                    "replica": ordered[0].replica,
                    "direction": ordered[0].direction,
                    "delta_g_kcal_mol": value,
                    "within_run_sem_kcal_mol": within_sem,
                    "approximate": any(item.endpoint_substituted for item in ordered),
                    "substituted_windows": sum(1 for item in ordered if item.endpoint_substituted),
                }
            )
        run_values = [float(item["delta_g_kcal_mol"]) for item in run_summaries]
        delta_g = _safe_mean(run_values)
        between_run_sem = _sample_sem(run_values)
        source_coefficients: dict[str, tuple[float, float]] = {}
        run_count = len(grouped)
        for run_windows in grouped.values():
            ordered = sorted(run_windows, key=lambda item: item.clambda)
            for weight, window in zip(_trapezoid_weights([item.clambda for item in ordered]), ordered):
                source_key = str(window.mdout_path.resolve())
                previous_coefficient, previous_sem = source_coefficients.get(source_key, (0.0, window.sem_dvdl))
                source_coefficients[source_key] = (
                    previous_coefficient + (weight / run_count),
                    max(previous_sem, window.sem_dvdl),
                )
        within_run_sem = math.sqrt(
            math.fsum((coefficient * sem) ** 2 for coefficient, sem in source_coefficients.values())
        )
        propagated_sem = math.sqrt(between_run_sem**2 + within_run_sem**2)
        bootstrap_samples = _bootstrap_grouped_mean(grouped, seed_label=f"{seed_label}:grouped")
        forward = [float(item["delta_g_kcal_mol"]) for item in run_summaries if item["direction"] == "forward"]
        reverse = [float(item["delta_g_kcal_mol"]) for item in run_summaries if item["direction"] == "reverse"]
        hysteresis = abs(_safe_mean(forward) - _safe_mean(reverse)) if forward and reverse else None
        if hysteresis is not None and hysteresis > max(1.0, 2.0 * within_run_sem):
            quality = "warning"
            warnings.append(
                f"Forward/reverse hysteresis is {hysteresis:.3f} kcal/mol, larger than the convergence threshold."
            )
        return PhaseAnalysis(
            phase=phase,
            delta_g_kcal_mol=delta_g,
            propagated_sem_kcal_mol=propagated_sem,
            bootstrap_ci95=_confidence_interval(bootstrap_samples or [delta_g]),
            windows=parsed_windows,
            bootstrap_samples=bootstrap_samples,
            quality=quality,
            warnings=warnings,
            sampling_runs=run_summaries,
            forward_reverse_difference_kcal_mol=hysteresis,
        )
    lambdas = [item.clambda for item in parsed_windows]
    weights = _trapezoid_weights(lambdas)
    delta_g = integrate_trapezoid((item.clambda, item.delta_g_source_value) for item in parsed_windows)
    propagated_sem = math.sqrt(
        math.fsum((weight * item.sem_dvdl) ** 2 for weight, item in zip(weights, parsed_windows))
    )
    bootstrap_samples = _bootstrap_integral(parsed_windows, seed_label=seed_label)
    return PhaseAnalysis(
        phase=phase,
        delta_g_kcal_mol=delta_g,
        propagated_sem_kcal_mol=propagated_sem,
        bootstrap_ci95=_confidence_interval(bootstrap_samples or [delta_g]),
        windows=parsed_windows,
        bootstrap_samples=bootstrap_samples,
        quality=quality,
        warnings=warnings,
    )


def inspect_water_case(path: str | Path) -> AnalysisCaseDiscovery | None:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "water_reference_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    ti_manifest_path = Path(str(manifest.get("ti_manifest_path") or root / "ti_manifest.json")).expanduser().resolve()
    if not ti_manifest_path.exists():
        return None
    expectations = _load_expected_windows(ti_manifest_path, root / "output")
    ti_manifest = _load_json(ti_manifest_path)
    sampling_protocol = ti_manifest.get("sampling_protocol") or {"mode": "single_pass"}
    completion_summary, complete = _summarize_completion(expectations)
    decoupling_scheme = _decoupling_scheme_from_expectations(expectations)
    metal = str(manifest.get("metal") or "Metal")
    formal_charge = int(manifest.get("formal_charge") or 0)
    water_model = str(manifest.get("water_model") or "tip3p").upper()
    metals = manifest.get("metals") or []
    description = (
        "multi-metal water reference: "
        + "; ".join(f"{item.get('element')}+{item.get('formal_charge')} atom {item.get('atom_index')}" for item in metals if isinstance(item, dict))
        if metals
        else f"{metal}{formal_charge}+ in {water_model} water"
    )
    return AnalysisCaseDiscovery(
        root=root,
        case_type=CASE_TYPE_WATER,
        display_name=root.name,
        description=description,
        completion_summary=completion_summary,
        readiness_note=_readiness_note(expectations, complete=complete),
        selectable=complete,
        metadata={
            "ti_manifest_path": str(ti_manifest_path),
            "output_root": str((root / "output").resolve()),
            "metal": metal,
            "formal_charge": formal_charge,
            "metals": metals,
            "water_model": str(manifest.get("water_model") or "tip3p").lower(),
            "ti_decoupling_scheme": decoupling_scheme,
            "ti_sampling_protocol": sampling_protocol,
            "endpoint_substitutions": _endpoint_substitution_payload(expectations),
            "charge_compensation_mode": str(
                (manifest.get("ti_protocol") or {}).get("charge_compensation_mode")
                or ("co_alchemical_counterions" if manifest.get("counterion_plan") else "none")
            ),
            "neutrality_validation": manifest.get("neutrality_validation") or {},
            "ti_lambda_schedule": _lambda_schedule_payload(expectations),
        },
    )


def inspect_bound_case(path: str | Path) -> AnalysisCaseDiscovery | None:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _load_json(manifest_path)
    if "bound_runtime_output_dir" not in manifest or "restraint_correction_kcal_mol" not in manifest:
        return None
    ti_manifest_path = (root / "bound" / "ti_manifest.json").resolve()
    if not ti_manifest_path.exists():
        return None
    output_root = Path(str(manifest.get("bound_runtime_output_dir"))).expanduser().resolve()
    expectations = _load_expected_windows(ti_manifest_path, output_root)
    ti_manifest = _load_json(ti_manifest_path)
    sampling_protocol = ti_manifest.get("sampling_protocol") or {"mode": "single_pass"}
    completion_summary, complete = _summarize_completion(expectations)
    decoupling_scheme = _decoupling_scheme_from_expectations(expectations)
    selected_sites = manifest.get("selected_sites") or ([] if manifest.get("selected_site") is None else [manifest.get("selected_site")])
    if len(selected_sites) > 1:
        site_labels = [
            f"site {item.get('site')} ({item.get('element')} atom {item.get('atom_index')})"
            for item in selected_sites
            if isinstance(item, dict)
        ]
        description = "multi-site total: " + "; ".join(site_labels)
    else:
        description = str(manifest.get("selected_metal") or root.name)
    snapshot_source = str(manifest.get("snapshot_source") or "unknown")
    return AnalysisCaseDiscovery(
        root=root,
        case_type=CASE_TYPE_BOUND,
        display_name=root.name,
        description=description,
        completion_summary=completion_summary,
        readiness_note=_readiness_note(expectations, complete=complete),
        selectable=complete,
        metadata={
            "snapshot_source": snapshot_source,
            "selected_metal": description,
            "selected_site": manifest.get("selected_site"),
            "selected_sites": selected_sites,
            "selected_formal_charge": manifest.get("selected_formal_charge"),
            "selected_formal_charges_by_site": manifest.get("selected_formal_charges_by_site") or {},
            "ti_selection_mode": manifest.get("ti_selection_mode") or "single",
            "restraint_correction_kcal_mol": float(manifest.get("restraint_correction_kcal_mol") or 0.0),
            "restraint_corrections_by_site": manifest.get("restraint_corrections_by_site") or {},
            "ti_manifest_path": str(ti_manifest_path),
            "output_root": str(output_root),
            "ti_decoupling_scheme": decoupling_scheme,
            "ti_sampling_protocol": sampling_protocol,
            "endpoint_substitutions": _endpoint_substitution_payload(expectations),
            "charge_compensation_mode": manifest.get("charge_compensation_mode") or "none",
            "neutrality_validation": manifest.get("bound_neutrality_validation") or {},
            "ti_lambda_schedule": _lambda_schedule_payload(expectations),
        },
    )


def inspect_analysis_case(path: str | Path, *, case_type: str | None = None) -> AnalysisCaseDiscovery | None:
    if case_type == CASE_TYPE_WATER:
        return inspect_water_case(path)
    if case_type == CASE_TYPE_BOUND:
        return inspect_bound_case(path)
    return inspect_water_case(path) or inspect_bound_case(path)


def _candidate_case_directories(search_dir: Path) -> list[Path]:
    resolved = search_dir.expanduser().resolve()
    candidates = [resolved]
    if search_subdirectories_enabled():
        candidates.extend(
            sorted((item for item in resolved.iterdir() if item.is_dir()), key=lambda item: item.name.lower())
        )
        if (resolved / "water_ref").is_dir():
            candidates.extend(
                sorted(
                    (item for item in (resolved / "water_ref").iterdir() if item.is_dir()),
                    key=lambda item: item.name.lower(),
                )
            )
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def discover_analysis_cases(search_dir: str | Path, *, case_types: set[str] | None = None) -> list[AnalysisCaseDiscovery]:
    resolved_case_types = case_types or {CASE_TYPE_WATER, CASE_TYPE_BOUND}
    discoveries: list[AnalysisCaseDiscovery] = []
    for candidate in _candidate_case_directories(Path(search_dir)):
        batch_manifest = candidate / "ti_batch_manifest.json"
        if batch_manifest.exists():
            try:
                payload = _load_json(batch_manifest)
            except Exception:
                payload = {}
            for case in payload.get("cases") or []:
                output_dir = case.get("output_dir")
                if not output_dir:
                    continue
                child = (candidate / str(output_dir)).resolve()
                discovery = inspect_analysis_case(child)
                if discovery is None or discovery.case_type not in resolved_case_types:
                    continue
                discovery.metadata["ti_batch_manifest"] = str(batch_manifest)
                discovery.metadata["batch_site"] = case.get("site")
                discovery.metadata["batch_element"] = case.get("element")
                discovery.metadata["batch_atom_index"] = case.get("atom_index")
                discoveries.append(discovery)
        discovery = inspect_analysis_case(candidate)
        if discovery is None or discovery.case_type not in resolved_case_types:
            continue
        discoveries.append(discovery)
    unique: list[AnalysisCaseDiscovery] = []
    seen: set[tuple[str, str]] = set()
    for discovery in discoveries:
        key = (str(discovery.root), discovery.case_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(discovery)
    unique.sort(key=lambda item: (0 if item.case_type == CASE_TYPE_WATER else 1, item.display_name.lower()))
    return unique


def _ensure_water_library_payload() -> dict[str, Any]:
    payload = _load_json(water_ref_library_path())
    if "entries" not in payload or not isinstance(payload.get("entries"), dict):
        payload = {"entries": {}}
    return payload


def _ensure_bound_library_payload() -> dict[str, Any]:
    payload = _load_json(bound_library_path())
    if "cases" not in payload or not isinstance(payload.get("cases"), dict):
        payload = {"cases": {}}
    return payload


def _library_key_parts(key: str) -> tuple[str, str | None]:
    base_key, separator, suffix = str(key).partition("::")
    if separator and suffix in {
        SAMPLING_SELECTION_FORWARD_ONLY,
        SAMPLING_SELECTION_FORWARD_REVERSE,
        SAMPLING_SELECTION_LEGACY_UNSPECIFIED,
    }:
        return base_key, suffix
    return str(key), None


def _library_sampling_groups(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = entry.get("sampling_groups")
    if isinstance(groups, dict) and groups:
        return {
            str(selection): group
            for selection, group in groups.items()
            if isinstance(group, dict) and isinstance(group.get("aggregate"), dict)
        }
    aggregate = entry.get("aggregate")
    if isinstance(aggregate, dict) and aggregate:
        return {
            SAMPLING_SELECTION_LEGACY_UNSPECIFIED: {
                "contributors": entry.get("contributors") or {},
                "aggregate": aggregate,
            }
        }
    return {}


def _library_group_snapshot(
    entry: dict[str, Any],
    *,
    base_key: str,
    sampling_selection: str,
    group: dict[str, Any],
) -> dict[str, Any]:
    snapshot = dict(entry)
    snapshot["base_key"] = base_key
    snapshot["key"] = f"{base_key}::{sampling_selection}"
    snapshot["sampling_selection"] = sampling_selection
    snapshot["contributors"] = group.get("contributors") or {}
    snapshot["aggregate"] = group.get("aggregate") or {}
    return snapshot


def _select_library_group(
    entry: dict[str, Any],
    *,
    base_key: str,
    sampling_selection: str | None,
) -> dict[str, Any] | None:
    groups = _library_sampling_groups(entry)
    if not groups:
        return None
    requested = sampling_selection
    if requested is None:
        requested = next(
            (
                selection
                for selection in (
                    SAMPLING_SELECTION_FORWARD_ONLY,
                    SAMPLING_SELECTION_LEGACY_UNSPECIFIED,
                    SAMPLING_SELECTION_FORWARD_REVERSE,
                )
                if selection in groups
            ),
            None,
        )
    if requested in groups:
        return _library_group_snapshot(
            entry,
            base_key=base_key,
            sampling_selection=requested,
            group=groups[requested],
        )
    if requested == SAMPLING_SELECTION_FORWARD_ONLY and SAMPLING_SELECTION_LEGACY_UNSPECIFIED in groups:
        return _library_group_snapshot(
            entry,
            base_key=base_key,
            sampling_selection=SAMPLING_SELECTION_LEGACY_UNSPECIFIED,
            group=groups[SAMPLING_SELECTION_LEGACY_UNSPECIFIED],
        )
    return None


def lookup_water_library_entry(
    metal: str,
    formal_charge: int,
    water_model: str,
    *,
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY,
) -> dict[str, Any] | None:
    return get_water_library_entry_by_key(
        water_library_key(metal, formal_charge, water_model),
        sampling_selection=sampling_selection,
    )


def get_water_library_entry_by_key(
    key: str,
    *,
    sampling_selection: str | None = None,
) -> dict[str, Any] | None:
    base_key, key_selection = _library_key_parts(key)
    entry = _ensure_water_library_payload().get("entries", {}).get(base_key)
    if not isinstance(entry, dict):
        return None
    return _select_library_group(
        entry,
        base_key=base_key,
        sampling_selection=key_selection or sampling_selection,
    )


def _decoupling_scheme_from_library_entry(entry: dict[str, Any]) -> str:
    aggregate = entry.get("aggregate") or {}
    scheme = str(aggregate.get("ti_decoupling_scheme") or "").strip()
    if scheme:
        return scheme
    contributors = entry.get("contributors") or {}
    schemes = {
        str(item.get("ti_decoupling_scheme")).strip()
        for item in contributors.values()
        if isinstance(item, dict) and item.get("ti_decoupling_scheme")
    }
    if len(schemes) == 1:
        return next(iter(schemes))
    if len(schemes) > 1:
        return DECOUPLING_SCHEME_MIXED
    return DECOUPLING_SCHEME_UNKNOWN


def discover_water_library_cases(
    *,
    sampling_selection: str | None = None,
) -> list[AnalysisCaseDiscovery]:
    if sampling_selection is not None:
        sampling_selection = _normalize_sampling_selection(sampling_selection)
    payload = _ensure_water_library_payload()
    discoveries: list[AnalysisCaseDiscovery] = []
    for key, entry in sorted(payload.get("entries", {}).items()):
        if not isinstance(entry, dict):
            continue
        for group_selection, group in _library_sampling_groups(entry).items():
            if sampling_selection is not None and group_selection != sampling_selection:
                if not (
                    sampling_selection == SAMPLING_SELECTION_FORWARD_ONLY
                    and group_selection == SAMPLING_SELECTION_LEGACY_UNSPECIFIED
                ):
                    continue
            snapshot = _library_group_snapshot(
                entry,
                base_key=key,
                sampling_selection=group_selection,
                group=group,
            )
            aggregate = snapshot.get("aggregate") or {}
            total = aggregate.get("total") or {}
            ci95 = total.get("bootstrap_ci95") or {}
            decoupling_scheme = _decoupling_scheme_from_library_entry(snapshot)
            selection_label = {
                SAMPLING_SELECTION_FORWARD_ONLY: "Forward",
                SAMPLING_SELECTION_FORWARD_REVERSE: "ForwardReverse",
                SAMPLING_SELECTION_LEGACY_UNSPECIFIED: "Legacy",
            }.get(group_selection, group_selection)
            is_legacy = group_selection == SAMPLING_SELECTION_LEGACY_UNSPECIFIED
            discoveries.append(
                AnalysisCaseDiscovery(
                    root=water_ref_library_path(),
                    case_type=CASE_TYPE_WATER,
                    display_name=(
                        f"{entry.get('metal', 'Metal')}{entry.get('formal_charge', '?')}+_"
                        f"{str(entry.get('water_model', 'tip3p')).upper()}_{selection_label}"
                    ),
                    description=(
                        f"{entry.get('metal', 'Metal')}{entry.get('formal_charge', '?')}+ in "
                        f"{str(entry.get('water_model', 'tip3p')).upper()} water [{selection_label}]"
                    ),
                    completion_summary=f"Library mean from {aggregate.get('n_cases', 0)} case(s)",
                    readiness_note=(
                        "Legacy library aggregate; direction metadata unavailable, accepted for Forward-only analysis"
                        if is_legacy
                        else f"Library aggregate ({selection_label})"
                    ),
                    selectable=True,
                    source_kind=SOURCE_KIND_LIBRARY,
                    library_key=str(snapshot["key"]),
                    library_snapshot=snapshot,
                    metadata={
                        "metal": entry.get("metal"),
                        "formal_charge": entry.get("formal_charge"),
                        "water_model": entry.get("water_model"),
                        "delta_g_kcal_mol": total.get("delta_g_kcal_mol"),
                        "propagated_sem_kcal_mol": total.get("propagated_sem_kcal_mol"),
                        "bootstrap_ci95": ci95,
                        "ti_decoupling_scheme": decoupling_scheme,
                        "charge_compensation_mode": aggregate.get("charge_compensation_mode") or "unknown",
                        "neutrality_validation": aggregate.get("neutrality_validation") or {},
                        "library_sampling_selection": group_selection,
                        "ti_sampling_protocol": {
                            "mode": (
                                "bidirectional"
                                if group_selection == SAMPLING_SELECTION_FORWARD_REVERSE
                                else group_selection
                            ),
                            "directions": (
                                ["forward", "reverse"]
                                if group_selection == SAMPLING_SELECTION_FORWARD_REVERSE
                                else ["forward"]
                            ),
                        },
                    },
                )
            )
    return discoveries


def _resolve_case(case_or_root: AnalysisCaseDiscovery | str | Path, *, case_type: str | None = None) -> AnalysisCaseDiscovery:
    if isinstance(case_or_root, AnalysisCaseDiscovery):
        return case_or_root
    discovery = inspect_analysis_case(case_or_root, case_type=case_type)
    if discovery is None:
        raise ValueError(f"That path does not look like a valid TI {case_type or 'analysis'} case: {case_or_root}")
    return discovery


def _result_from_library_case(
    case: AnalysisCaseDiscovery,
    *,
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY,
) -> SingleCaseAnalysisResult:
    if case.source_kind != SOURCE_KIND_LIBRARY or case.library_snapshot is None:
        raise ValueError("The provided case is not a library-backed water reference.")
    sampling_selection = _normalize_sampling_selection(sampling_selection)
    library_selection = str(
        case.metadata.get("library_sampling_selection")
        or case.library_snapshot.get("sampling_selection")
        or SAMPLING_SELECTION_LEGACY_UNSPECIFIED
    )
    legacy_forward_fallback = (
        sampling_selection == SAMPLING_SELECTION_FORWARD_ONLY
        and library_selection == SAMPLING_SELECTION_LEGACY_UNSPECIFIED
    )
    if library_selection != sampling_selection and not legacy_forward_fallback:
        raise ValueError(
            "The selected water-library value was generated for "
            f"'{library_selection}', not '{sampling_selection}'. Select a matching library entry."
        )
    aggregate = case.library_snapshot.get("aggregate") or {}
    qoff = aggregate.get("qoff") or {}
    vdwoff = aggregate.get("vdwoff") or {}
    total = aggregate.get("total") or {}
    output_dir = repo_root() / "analysis_library"
    qoff_analysis = PhaseAnalysis(
        phase="qoff",
        delta_g_kcal_mol=float(qoff.get("delta_g_kcal_mol", 0.0)),
        propagated_sem_kcal_mol=float(qoff.get("propagated_sem_kcal_mol", 0.0)),
        bootstrap_ci95=ConfidenceInterval(
            float((qoff.get("bootstrap_ci95") or {}).get("low", 0.0)),
            float((qoff.get("bootstrap_ci95") or {}).get("high", 0.0)),
        ),
        windows=[],
        bootstrap_samples=[float(qoff.get("delta_g_kcal_mol", 0.0))],
    )
    vdwoff_analysis = PhaseAnalysis(
        phase="vdwoff",
        delta_g_kcal_mol=float(vdwoff.get("delta_g_kcal_mol", 0.0)),
        propagated_sem_kcal_mol=float(vdwoff.get("propagated_sem_kcal_mol", 0.0)),
        bootstrap_ci95=ConfidenceInterval(
            float((vdwoff.get("bootstrap_ci95") or {}).get("low", 0.0)),
            float((vdwoff.get("bootstrap_ci95") or {}).get("high", 0.0)),
        ),
        windows=[],
        bootstrap_samples=[float(vdwoff.get("delta_g_kcal_mol", 0.0))],
    )
    warnings = (
        [
            "This is a legacy water-library value without direction metadata; it is being treated as "
            "Forward-only for backward compatibility."
        ]
        if legacy_forward_fallback
        else []
    )
    aggregate_quality = str(aggregate.get("quality") or "ok")
    return SingleCaseAnalysisResult(
        case=case,
        qoff=qoff_analysis,
        vdwoff=vdwoff_analysis,
        delta_g_kcal_mol=float(total.get("delta_g_kcal_mol", 0.0)),
        propagated_sem_kcal_mol=float(total.get("propagated_sem_kcal_mol", 0.0)),
        bootstrap_ci95=ConfidenceInterval(
            float((total.get("bootstrap_ci95") or {}).get("low", 0.0)),
            float((total.get("bootstrap_ci95") or {}).get("high", 0.0)),
        ),
        output_dir=output_dir,
        sampling_selection=sampling_selection,
        quality="warning" if warnings or aggregate_quality == "warning" else "ok",
        warnings=warnings,
        bootstrap_samples=[float(total.get("delta_g_kcal_mol", 0.0))],
    )


def _write_windows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "case_type",
        "source_kind",
        "run_id",
        "replica",
        "direction",
        "phase",
        "lambda",
        "mdout_path",
        "expected_mdout_path",
        "endpoint_substituted",
        "delta_g_source_value",
        "sample_mean_dvdl",
        "sample_std_dvdl",
        "sem_dvdl",
        "sem_mode",
        "sample_count",
        "block_count",
        "parser_mode",
        "quality",
        "warning",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def render_single_case_report(result: SingleCaseAnalysisResult) -> str:
    decoupling_scheme = decoupling_scheme_for_result(result)
    lines = [
        f"Case: {result.case.display_name}",
        f"Type: {result.case.case_type}",
        f"TI decoupling: {_decoupling_scheme_detail(decoupling_scheme)}",
        f"Sampling included: {_sampling_selection_detail(result.sampling_selection)}",
        f"Description: {result.case.description}",
        f"Quality: {result.quality}",
    ]
    selected_sites = result.case.metadata.get("selected_sites") or []
    if selected_sites:
        lines.append(
            "Selected metals: "
            + "; ".join(
                f"site {item.get('site')} {item.get('element')} atom {item.get('atom_index')}"
                for item in selected_sites
                if isinstance(item, dict)
            )
        )
        if len(selected_sites) > 1:
            lines.append("Multi-site note: this is a single all-at-once total dG, not a per-metal decomposition.")
    if result.case.case_type == CASE_TYPE_WATER:
        lines.append("This report summarizes the standalone water-reference dG.")
    else:
        lines.append("This report summarizes the standalone bound-case TI dG.")
    phase_label = "combined" if decoupling_scheme == DECOUPLING_SCHEME_COMBINED else "qoff"
    lines.extend(
        [
            "",
            f"{phase_label}: {_format_mean_sem(result.qoff.delta_g_kcal_mol, result.qoff.propagated_sem_kcal_mol)} kcal/mol",
            f"{phase_label} 95% CI: {_format_ci95(result.qoff.bootstrap_ci95)}",
        ]
    )
    if decoupling_scheme == DECOUPLING_SCHEME_COMBINED:
        lines.append("vdwoff: N/A (combined single softcore path)")
    else:
        lines.extend(
            [
                f"vdwoff: {_format_mean_sem(result.vdwoff.delta_g_kcal_mol, result.vdwoff.propagated_sem_kcal_mol)} kcal/mol",
                f"vdwoff 95% CI: {_format_ci95(result.vdwoff.bootstrap_ci95)}",
            ]
        )
    if result.qoff.sampling_runs:
        lines.extend(["", "Bidirectional sweeps:"])
        for sampling_run in result.qoff.sampling_runs:
            approximation = " [APPROXIMATE: forward lambda=0 substituted]" if sampling_run.get("approximate") else ""
            lines.append(
                f"- {sampling_run.get('direction')}: {float(sampling_run.get('delta_g_kcal_mol', 0.0)):.6f} "
                f"kcal/mol{approximation}"
            )
        if result.qoff.forward_reverse_difference_kcal_mol is not None:
            lines.append(
                "- forward/reverse hysteresis: "
                f"{result.qoff.forward_reverse_difference_kcal_mol:.6f} kcal/mol"
            )
    lines.extend(
        [
            f"Total dG: {_format_mean_sem(result.delta_g_kcal_mol, result.propagated_sem_kcal_mol)} kcal/mol",
            f"Total dG 95% CI: {_format_ci95(result.bootstrap_ci95)}",
        ]
    )
    if result.restraint_correction_kcal_mol is not None:
        lines.append(f"Restraint correction: {result.restraint_correction_kcal_mol:.6f} kcal/mol")
        lines.append(
            f"Corrected bound dG: {_format_mean_sem(result.corrected_delta_g_kcal_mol, result.corrected_propagated_sem_kcal_mol)} kcal/mol"
        )
        lines.append(f"Corrected bound dG 95% CI: {_format_ci95(result.corrected_bootstrap_ci95)}")
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def render_rbfe_report(result: RBFEAnalysisResult) -> str:
    lines = [
        f"Bound case: {result.bound.case.display_name}",
        f"Water case: {result.water.case.display_name}",
        f"Bound TI decoupling: {_decoupling_scheme_detail(decoupling_scheme_for_result(result.bound))}",
        f"Water TI decoupling: {_decoupling_scheme_detail(decoupling_scheme_for_result(result.water))}",
        f"Sampling included: {_sampling_selection_detail(result.sampling_selection)}",
        "Formula: ddG = (dG_bound_ti + restraint_correction) - dG_water",
        "",
        f"Bound qoff: {_format_mean_sem(result.bound.qoff.delta_g_kcal_mol, result.bound.qoff.propagated_sem_kcal_mol)} kcal/mol",
        f"Bound vdwoff: {_format_mean_sem(result.bound.vdwoff.delta_g_kcal_mol, result.bound.vdwoff.propagated_sem_kcal_mol)} kcal/mol",
        f"Bound corrected: {_format_mean_sem(result.bound.corrected_delta_g_kcal_mol, result.bound.corrected_propagated_sem_kcal_mol)} kcal/mol",
        f"Bound corrected 95% CI: {_format_ci95(result.bound.corrected_bootstrap_ci95)}",
        f"Water total: {_format_mean_sem(result.water.delta_g_kcal_mol, result.water.propagated_sem_kcal_mol)} kcal/mol",
        f"Water 95% CI: {_format_ci95(result.water.bootstrap_ci95)}",
        f"Final ddG: {_format_mean_sem(result.ddg_kcal_mol, result.propagated_sem_kcal_mol)} kcal/mol",
        f"Final ddG 95% CI: {_format_ci95(result.bootstrap_ci95)}",
    ]
    bound_hysteresis = _case_max_hysteresis(result.bound)
    water_hysteresis = _case_max_hysteresis(result.water)
    if bound_hysteresis is not None or water_hysteresis is not None:
        lines.extend(
            [
                "",
                "Direction diagnostics:",
                f"- Bound max |Forward-Reverse|: {_format_mean_sem(bound_hysteresis, None)} kcal/mol",
                f"- Water max |Forward-Reverse|: {_format_mean_sem(water_hysteresis, None)} kcal/mol",
            ]
        )
    if result.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def _result_library_sampling_selection(result: SingleCaseAnalysisResult) -> str:
    has_reverse = any(
        window.direction == "reverse"
        for phase in (result.qoff, result.vdwoff)
        for window in phase.windows
    )
    return (
        SAMPLING_SELECTION_FORWARD_REVERSE
        if result.sampling_selection == SAMPLING_SELECTION_FORWARD_REVERSE and has_reverse
        else SAMPLING_SELECTION_FORWARD_ONLY
    )


def _water_contributor_record(result: SingleCaseAnalysisResult) -> dict[str, Any]:
    library_sampling_selection = _result_library_sampling_selection(result)
    return {
        "case_root": str(result.case.root),
        "display_name": result.case.display_name,
        "quality": result.quality,
        "warnings": result.warnings,
        "endpoint_substitutions": result.case.metadata.get("endpoint_substitutions") or [],
        "ti_decoupling_scheme": decoupling_scheme_for_result(result),
        "sampling_selection": library_sampling_selection,
        "charge_compensation_mode": _charge_compensation_mode(result.case),
        "neutrality_validation": result.case.metadata.get("neutrality_validation") or {},
        "qoff": _phase_metric_payload(
            result.qoff.delta_g_kcal_mol,
            result.qoff.propagated_sem_kcal_mol,
            result.qoff.bootstrap_ci95,
        ),
        "vdwoff": _phase_metric_payload(
            result.vdwoff.delta_g_kcal_mol,
            result.vdwoff.propagated_sem_kcal_mol,
            result.vdwoff.bootstrap_ci95,
        ),
        "total": _phase_metric_payload(
            result.delta_g_kcal_mol,
            result.propagated_sem_kcal_mol,
            result.bootstrap_ci95,
        ),
    }


def _aggregate_metric(
    contributors: list[dict[str, Any]],
    metric_key: str,
    *,
    seed_label: str,
) -> dict[str, Any]:
    metrics = [item.get(metric_key) or {} for item in contributors]
    values = [float(item.get("delta_g_kcal_mol", 0.0)) for item in metrics]
    mean_value = _safe_mean(values)
    if len(values) > 1:
        sem = _sample_sem(values)
        rng = random.Random(_stable_seed(seed_label))
        bootstrap_samples: list[float] = []
        for _ in range(_DEFAULT_BOOTSTRAP_ITERATIONS):
            picked = [values[rng.randrange(len(values))] for _ in range(len(values))]
            bootstrap_samples.append(_safe_mean(picked))
        ci95 = _confidence_interval(bootstrap_samples)
    else:
        sem = float(metrics[0].get("propagated_sem_kcal_mol", 0.0)) if metrics else 0.0
        ci95_payload = (metrics[0].get("bootstrap_ci95") or {}) if metrics else {}
        ci95 = ConfidenceInterval(float(ci95_payload.get("low", mean_value)), float(ci95_payload.get("high", mean_value)))
    return _phase_metric_payload(mean_value, sem, ci95)


def _aggregate_water_contributors(
    contributors: dict[str, dict[str, Any]],
    *,
    key: str,
    sampling_selection: str,
) -> dict[str, Any]:
    contributor_values = list(contributors.values())
    decoupling_schemes = sorted(
        {
            str(item.get("ti_decoupling_scheme")).strip()
            for item in contributor_values
            if str(item.get("ti_decoupling_scheme") or "").strip()
        }
    )
    charge_compensation_modes = sorted(
        {
            str(item.get("charge_compensation_mode") or "unknown").strip().lower()
            for item in contributor_values
        }
    )
    neutrality_statuses = {
        str((item.get("neutrality_validation") or {}).get("status") or "missing").strip().lower()
        for item in contributor_values
    }
    return {
        "sampling_selection": sampling_selection,
        "n_cases": len(contributor_values),
        "quality": "warning" if any(item.get("quality") == "warning" for item in contributor_values) else "ok",
        "ti_decoupling_scheme": (
            decoupling_schemes[0]
            if len(decoupling_schemes) == 1
            else (DECOUPLING_SCHEME_MIXED if decoupling_schemes else DECOUPLING_SCHEME_UNKNOWN)
        ),
        "ti_decoupling_schemes": decoupling_schemes,
        "charge_compensation_mode": (
            charge_compensation_modes[0] if len(charge_compensation_modes) == 1 else "mixed"
        ),
        "neutrality_validation": {
            "status": "passed" if neutrality_statuses == {"passed"} else "mixed_or_missing"
        },
        "qoff": _aggregate_metric(
            contributor_values,
            "qoff",
            seed_label=f"{key}:{sampling_selection}:qoff",
        ),
        "vdwoff": _aggregate_metric(
            contributor_values,
            "vdwoff",
            seed_label=f"{key}:{sampling_selection}:vdwoff",
        ),
        "total": _aggregate_metric(
            contributor_values,
            "total",
            seed_label=f"{key}:{sampling_selection}:total",
        ),
    }


def _update_water_ref_library(result: SingleCaseAnalysisResult) -> None:
    if result.case.case_type != CASE_TYPE_WATER or result.case.source_kind != SOURCE_KIND_SIMULATION:
        return
    metal = str(result.case.metadata.get("metal") or "metal")
    formal_charge = int(result.case.metadata.get("formal_charge") or 0)
    water_model = str(result.case.metadata.get("water_model") or "tip3p")
    key = water_library_key(metal, formal_charge, water_model)
    library_sampling_selection = _result_library_sampling_selection(result)
    payload = _ensure_water_library_payload()
    entries = payload["entries"]
    entry = entries.setdefault(
        key,
        {
            "key": key,
            "metal": metal,
            "formal_charge": formal_charge,
            "water_model": water_model,
            "sampling_groups": {},
        },
    )
    sampling_groups = entry.setdefault("sampling_groups", {})
    if not sampling_groups and isinstance(entry.get("aggregate"), dict) and entry.get("aggregate"):
        sampling_groups[SAMPLING_SELECTION_LEGACY_UNSPECIFIED] = {
            "contributors": dict(entry.get("contributors") or {}),
            "aggregate": dict(entry["aggregate"]),
        }
    group = sampling_groups.setdefault(
        library_sampling_selection,
        {"contributors": {}, "aggregate": {}},
    )
    contributors = group.setdefault("contributors", {})
    contributors[str(result.case.root)] = _water_contributor_record(result)
    group["aggregate"] = _aggregate_water_contributors(
        contributors,
        key=key,
        sampling_selection=library_sampling_selection,
    )
    preferred_group = (
        sampling_groups.get(SAMPLING_SELECTION_FORWARD_ONLY)
        or sampling_groups.get(SAMPLING_SELECTION_LEGACY_UNSPECIFIED)
        or sampling_groups.get(SAMPLING_SELECTION_FORWARD_REVERSE)
        or group
    )
    # Keep the historical top-level fields as a Forward-first compatibility
    # alias while all new data lives in direction-specific sampling groups.
    entry["contributors"] = dict(preferred_group.get("contributors") or {})
    entry["aggregate"] = dict(preferred_group.get("aggregate") or {})
    entry["library_schema_version"] = 2
    payload["library_schema_version"] = 2
    _save_library_json(water_ref_library_path(), payload)


def _update_bound_library(result: SingleCaseAnalysisResult) -> None:
    if result.case.case_type != CASE_TYPE_BOUND or result.case.source_kind != SOURCE_KIND_SIMULATION:
        return
    payload = _ensure_bound_library_payload()
    library_sampling_selection = _result_library_sampling_selection(result)
    library_case_key = str(result.case.root)
    if library_sampling_selection != SAMPLING_SELECTION_FORWARD_ONLY:
        library_case_key = f"{library_case_key}::{library_sampling_selection}"
    payload["cases"][library_case_key] = {
        "case_root": str(result.case.root),
        "display_name": result.case.display_name,
        "description": result.case.description,
        "quality": result.quality,
        "warnings": result.warnings,
        "endpoint_substitutions": result.case.metadata.get("endpoint_substitutions") or [],
        "ti_decoupling_scheme": decoupling_scheme_for_result(result),
        "sampling_selection": library_sampling_selection,
        "charge_compensation_mode": _charge_compensation_mode(result.case),
        "neutrality_validation": result.case.metadata.get("neutrality_validation") or {},
        "snapshot_source": result.case.metadata.get("snapshot_source"),
        "qoff": _phase_metric_payload(result.qoff.delta_g_kcal_mol, result.qoff.propagated_sem_kcal_mol, result.qoff.bootstrap_ci95),
        "vdwoff": _phase_metric_payload(
            result.vdwoff.delta_g_kcal_mol, result.vdwoff.propagated_sem_kcal_mol, result.vdwoff.bootstrap_ci95
        ),
        "total": _phase_metric_payload(result.delta_g_kcal_mol, result.propagated_sem_kcal_mol, result.bootstrap_ci95),
        "restraint_correction_kcal_mol": result.restraint_correction_kcal_mol,
        "corrected_total": (
            None
            if result.corrected_delta_g_kcal_mol is None or result.corrected_bootstrap_ci95 is None
            else _phase_metric_payload(
                result.corrected_delta_g_kcal_mol,
                result.corrected_propagated_sem_kcal_mol or 0.0,
                result.corrected_bootstrap_ci95,
            )
        ),
    }
    _save_library_json(bound_library_path(), payload)


def _persist_single_case_result(result: SingleCaseAnalysisResult) -> SingleCaseAnalysisResult:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result.output_dir / "abfe_summary.json", result.to_dict())
    csv_rows: list[dict[str, Any]] = []
    for phase in (result.qoff, result.vdwoff):
        for window in phase.windows:
            csv_rows.append(
                {
                    "case_type": result.case.case_type,
                    "source_kind": result.case.source_kind,
                    "run_id": window.run_id,
                    "replica": window.replica,
                    "direction": window.direction,
                    "phase": window.phase,
                    "lambda": f"{window.clambda:.3f}",
                    "mdout_path": str(window.mdout_path),
                    "expected_mdout_path": ""
                    if window.expected_mdout_path is None
                    else str(window.expected_mdout_path),
                    "endpoint_substituted": window.endpoint_substituted,
                    "delta_g_source_value": window.delta_g_source_value,
                    "sample_mean_dvdl": window.sample_mean_dvdl,
                    "sample_std_dvdl": window.sample_std_dvdl,
                    "sem_dvdl": window.sem_dvdl,
                    "sem_mode": window.sem_mode,
                    "sample_count": window.sample_count,
                    "block_count": window.block_count,
                    "parser_mode": window.parser_mode,
                    "quality": window.quality,
                    "warning": window.warning or "",
                }
            )
    _write_windows_csv(result.output_dir / "abfe_windows.csv", csv_rows)
    (result.output_dir / "abfe_report.txt").write_text(render_single_case_report(result), encoding="utf-8")
    _update_water_ref_library(result)
    _update_bound_library(result)
    return result


def _persist_rbfe_result(result: RBFEAnalysisResult) -> RBFEAnalysisResult:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result.output_dir / "rbfe_summary.json", result.to_dict())
    csv_rows: list[dict[str, Any]] = []
    for leg_name, single in (("bound", result.bound), ("water", result.water)):
        for phase in (single.qoff, single.vdwoff):
            for window in phase.windows:
                csv_rows.append(
                    {
                        "case_type": leg_name,
                        "source_kind": single.case.source_kind,
                        "run_id": window.run_id,
                        "replica": window.replica,
                        "direction": window.direction,
                        "phase": window.phase,
                        "lambda": f"{window.clambda:.3f}",
                        "mdout_path": str(window.mdout_path),
                        "expected_mdout_path": ""
                        if window.expected_mdout_path is None
                        else str(window.expected_mdout_path),
                        "endpoint_substituted": window.endpoint_substituted,
                        "delta_g_source_value": window.delta_g_source_value,
                        "sample_mean_dvdl": window.sample_mean_dvdl,
                        "sample_std_dvdl": window.sample_std_dvdl,
                        "sem_dvdl": window.sem_dvdl,
                        "sem_mode": window.sem_mode,
                        "sample_count": window.sample_count,
                        "block_count": window.block_count,
                        "parser_mode": window.parser_mode,
                        "quality": window.quality,
                        "warning": window.warning or "",
                    }
                )
    _write_windows_csv(result.output_dir / "rbfe_windows.csv", csv_rows)
    (result.output_dir / "rbfe_report.txt").write_text(render_rbfe_report(result), encoding="utf-8")
    return result


def analyze_single_case(
    case_or_root: AnalysisCaseDiscovery | str | Path,
    *,
    case_type: str | None = None,
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY,
) -> SingleCaseAnalysisResult:
    sampling_selection = _normalize_sampling_selection(sampling_selection)
    case = _resolve_case(case_or_root, case_type=case_type)
    if case.source_kind == SOURCE_KIND_LIBRARY:
        return _result_from_library_case(case, sampling_selection=sampling_selection)
    if not case.selectable:
        raise ValueError(case.readiness_note)
    ti_manifest_path = Path(str(case.metadata["ti_manifest_path"]))
    output_root = Path(str(case.metadata["output_root"]))
    expectations = _load_expected_windows(ti_manifest_path, output_root)
    if sampling_selection == SAMPLING_SELECTION_FORWARD_ONLY:
        expectations = [item for item in expectations if item.direction == "forward"]
        if not expectations:
            raise ValueError("No forward TI windows were found in this case.")
    qoff = _phase_analysis("qoff", expectations, seed_label=f"{case.root}:{sampling_selection}:qoff")
    vdwoff = _phase_analysis("vdwoff", expectations, seed_label=f"{case.root}:{sampling_selection}:vdwoff")
    total_delta_g = qoff.delta_g_kcal_mol + vdwoff.delta_g_kcal_mol
    total_sem = math.sqrt(qoff.propagated_sem_kcal_mol**2 + vdwoff.propagated_sem_kcal_mol**2)
    total_bootstrap = _combine_bootstrap_samples(
        qoff.bootstrap_samples or [qoff.delta_g_kcal_mol],
        vdwoff.bootstrap_samples or [vdwoff.delta_g_kcal_mol],
        operation="add",
    )
    warnings = [*qoff.warnings, *vdwoff.warnings]
    quality: Literal["ok", "warning"] = "warning" if warnings or qoff.quality == "warning" or vdwoff.quality == "warning" else "ok"
    result = SingleCaseAnalysisResult(
        case=case,
        qoff=qoff,
        vdwoff=vdwoff,
        delta_g_kcal_mol=total_delta_g,
        propagated_sem_kcal_mol=total_sem,
        bootstrap_ci95=_confidence_interval(total_bootstrap or [total_delta_g]),
        output_dir=case.root / "analysis" / (
            "abfe" if sampling_selection == SAMPLING_SELECTION_FORWARD_ONLY else "abfe_forward_reverse"
        ),
        sampling_selection=sampling_selection,
        quality=quality,
        warnings=warnings,
        bootstrap_samples=total_bootstrap,
    )
    if case.case_type == CASE_TYPE_BOUND:
        correction = float(case.metadata.get("restraint_correction_kcal_mol") or 0.0)
        corrected_bootstrap = [sample + correction for sample in (total_bootstrap or [total_delta_g])]
        result.restraint_correction_kcal_mol = correction
        result.corrected_delta_g_kcal_mol = total_delta_g + correction
        result.corrected_propagated_sem_kcal_mol = total_sem
        result.corrected_bootstrap_ci95 = _confidence_interval(corrected_bootstrap)
        result.corrected_bootstrap_samples = corrected_bootstrap
    return _persist_single_case_result(result)


def analyze_rbfe(
    bound_case_or_root: AnalysisCaseDiscovery | str | Path,
    water_case_or_root: AnalysisCaseDiscovery | str | Path,
    *,
    sampling_selection: str = SAMPLING_SELECTION_FORWARD_ONLY,
) -> RBFEAnalysisResult:
    sampling_selection = _normalize_sampling_selection(sampling_selection)
    bound_case = _resolve_case(bound_case_or_root, case_type=CASE_TYPE_BOUND)
    if not bound_case.selectable:
        raise ValueError(bound_case.readiness_note)
    water_case = water_case_or_root if isinstance(water_case_or_root, AnalysisCaseDiscovery) else _resolve_case(water_case_or_root, case_type=CASE_TYPE_WATER)
    if not water_case.selectable:
        raise ValueError(water_case.readiness_note)
    compatibility_errors, compatibility_warnings = rbfe_pair_compatibility(bound_case, water_case)
    if compatibility_errors:
        raise ValueError("Incompatible RBFE pair: " + " ".join(compatibility_errors))
    bound = analyze_single_case(
        bound_case,
        case_type=CASE_TYPE_BOUND,
        sampling_selection=sampling_selection,
    )
    if isinstance(water_case, AnalysisCaseDiscovery) and water_case.source_kind == SOURCE_KIND_LIBRARY:
        water = _result_from_library_case(water_case, sampling_selection=sampling_selection)
    else:
        water = analyze_single_case(
            water_case,
            case_type=CASE_TYPE_WATER,
            sampling_selection=sampling_selection,
        )
    corrected_bound = bound.corrected_delta_g_kcal_mol if bound.corrected_delta_g_kcal_mol is not None else bound.delta_g_kcal_mol
    corrected_bound_sem = (
        bound.corrected_propagated_sem_kcal_mol
        if bound.corrected_propagated_sem_kcal_mol is not None
        else bound.propagated_sem_kcal_mol
    )
    ddg = corrected_bound - water.delta_g_kcal_mol
    ddg_sem = math.sqrt(corrected_bound_sem**2 + water.propagated_sem_kcal_mol**2)
    ddg_bootstrap = _combine_bootstrap_samples(
        bound.corrected_bootstrap_samples or bound.bootstrap_samples or [corrected_bound],
        water.bootstrap_samples or [water.delta_g_kcal_mol],
        operation="subtract",
    )
    warnings = [*compatibility_warnings, *bound.warnings, *water.warnings]
    quality: Literal["ok", "warning"] = "warning" if warnings or bound.quality == "warning" or water.quality == "warning" else "ok"
    result = RBFEAnalysisResult(
        bound=bound,
        water=water,
        ddg_kcal_mol=ddg,
        propagated_sem_kcal_mol=ddg_sem,
        bootstrap_ci95=_confidence_interval(ddg_bootstrap or [ddg]),
        output_dir=bound.case.root / "analysis" / (
            "rbfe" if sampling_selection == SAMPLING_SELECTION_FORWARD_ONLY else "rbfe_forward_reverse"
        ),
        sampling_selection=sampling_selection,
        quality=quality,
        warnings=warnings,
        bootstrap_samples=ddg_bootstrap,
    )
    return _persist_rbfe_result(result)


def print_analysis_summary(
    result: SingleCaseAnalysisResult
    | RBFEAnalysisResult
    | list[SingleCaseAnalysisResult]
    | list[RBFEAnalysisResult],
) -> None:
    if Table is None:
        if isinstance(result, list):
            console.print(json.dumps([item.to_dict() for item in result], indent=2))
            return
        console.print(json.dumps(result.to_dict(), indent=2))
        return
    if isinstance(result, list):
        if result and all(isinstance(item, RBFEAnalysisResult) for item in result):
            rbfe_results = [item for item in result if isinstance(item, RBFEAnalysisResult)]
            table = Table(title="Batch RBFE", box=box.SIMPLE_HEAVY)
            table.add_column("Bound case", style="bold white")
            table.add_column("Water reference", style="white")
            table.add_column("ddG", style="cyan", justify="right")
            table.add_column("SEM", style="green", justify="right")
            table.add_column("95% CI", style="magenta", justify="right")
            table.add_column("Bound |F-R|", style="yellow", justify="right")
            table.add_column("Water |F-R|", style="yellow", justify="right")
            for item in rbfe_results:
                table.add_row(
                    item.bound.case.display_name,
                    item.water.case.display_name,
                    f"{item.ddg_kcal_mol:.6f}",
                    f"{item.propagated_sem_kcal_mol:.6f}",
                    _format_ci95(item.bootstrap_ci95),
                    _format_mean_sem(_case_max_hysteresis(item.bound), None),
                    _format_mean_sem(_case_max_hysteresis(item.water), None),
                )
            console.print(table)
            console.print(
                f"[dim]Analyzed {len(rbfe_results)} RBFE pair(s). Individual outputs were saved under each "
                "bound case's analysis directory.[/dim]"
            )
            return
        table = Table(title="Batch Single-Case dG", box=box.SIMPLE_HEAVY)
        table.add_column("Case", style="bold white")
        table.add_column("Type", style="white")
        table.add_column("Scheme", style="magenta")
        table.add_column("qoff", style="cyan", justify="right")
        table.add_column("vdwoff", style="cyan", justify="right")
        table.add_column("Total dG", style="cyan", justify="right")
        table.add_column("SEM", style="green", justify="right")
        table.add_column("95% CI", style="magenta", justify="right")
        for item in result:
            scheme = decoupling_scheme_for_result(item)
            vdwoff_value = "N/A" if scheme == DECOUPLING_SCHEME_COMBINED else f"{item.vdwoff.delta_g_kcal_mol:.6f}"
            table.add_row(
                item.case.display_name,
                item.case.case_type,
                _decoupling_scheme_detail(scheme),
                f"{item.qoff.delta_g_kcal_mol:.6f}",
                vdwoff_value,
                f"{item.delta_g_kcal_mol:.6f}",
                f"{item.propagated_sem_kcal_mol:.6f}",
                _format_ci95(item.bootstrap_ci95),
            )
        console.print(table)
        console.print(
            f"[dim]Analyzed {len(result)} completed TI case(s). Individual outputs were saved under each "
            "case's analysis directory.[/dim]"
        )
        return
    if isinstance(result, SingleCaseAnalysisResult):
        title = "Single-Case dG"
        table = Table(title=title, box=box.SIMPLE_HEAVY)
        table.add_column("Component", style="bold white")
        table.add_column("DeltaG (kcal/mol)", style="cyan", justify="right")
        table.add_column("Propagated SEM", style="green", justify="right")
        table.add_column("95% CI", style="magenta", justify="right")
        decoupling_scheme = decoupling_scheme_for_result(result)
        rows: list[tuple[str, float | None, float | None, ConfidenceInterval | None]] = [
            ("qoff", result.qoff.delta_g_kcal_mol, result.qoff.propagated_sem_kcal_mol, result.qoff.bootstrap_ci95),
        ]
        if decoupling_scheme == DECOUPLING_SCHEME_COMBINED:
            rows.append(("vdwoff", None, None, None))
        else:
            rows.append(("vdwoff", result.vdwoff.delta_g_kcal_mol, result.vdwoff.propagated_sem_kcal_mol, result.vdwoff.bootstrap_ci95))
        rows.append(("Total dG", result.delta_g_kcal_mol, result.propagated_sem_kcal_mol, result.bootstrap_ci95))
        if result.restraint_correction_kcal_mol is not None:
            rows.append(("Restraint correction", result.restraint_correction_kcal_mol, 0.0, None))
            rows.append(
                (
                    "Corrected bound dG",
                    result.corrected_delta_g_kcal_mol or 0.0,
                    result.corrected_propagated_sem_kcal_mol or 0.0,
                    result.corrected_bootstrap_ci95,
                )
            )
        for name, value, sem, ci95 in rows:
            table.add_row(
                name,
                "N/A" if value is None else f"{value:.6f}",
                "N/A" if sem is None else f"{sem:.6f}",
                _format_ci95(ci95),
            )
        console.print(table)
        console.print(f"[dim]TI decoupling: {_decoupling_scheme_detail(decoupling_scheme)}[/dim]")
        if result.qoff.sampling_runs:
            sweep_table = Table(title="Bidirectional sweep diagnostics", box=box.SIMPLE)
            sweep_table.add_column("Direction", style="bold white")
            sweep_table.add_column("DeltaG (kcal/mol)", style="cyan", justify="right")
            sweep_table.add_column("Status", style="yellow")
            for sampling_run in result.qoff.sampling_runs:
                sweep_table.add_row(
                    str(sampling_run.get("direction") or "unknown"),
                    f"{float(sampling_run.get('delta_g_kcal_mol', 0.0)):.6f}",
                    "APPROXIMATE (forward lambda=0 substituted)"
                    if sampling_run.get("approximate")
                    else "complete",
                )
            if result.qoff.forward_reverse_difference_kcal_mol is not None:
                sweep_table.add_row(
                    "hysteresis |F-R|",
                    f"{result.qoff.forward_reverse_difference_kcal_mol:.6f}",
                    "diagnostic",
                )
            console.print(sweep_table)
        selected_sites = result.case.metadata.get("selected_sites") or []
        if selected_sites:
            metals = "; ".join(
                f"site {item.get('site')} {item.get('element')} atom {item.get('atom_index')}"
                for item in selected_sites
                if isinstance(item, dict)
            )
            console.print(f"[dim]Selected metal(s): {metals}[/dim]")
            if len(selected_sites) > 1:
                console.print("[dim]Multi-site all-at-once result: total dG only, not per-metal decomposition.[/dim]")
        console.print(f"[dim]Saved analysis outputs to {result.output_dir}[/dim]")
        for warning in result.warnings:
            console.print(f"[bold yellow]Warning:[/bold yellow] {warning}")
        return
    table = Table(title="RBFE", box=box.SIMPLE_HEAVY)
    table.add_column("Component", style="bold white")
    table.add_column("Value (kcal/mol)", style="cyan", justify="right")
    table.add_column("Propagated SEM", style="green", justify="right")
    table.add_column("95% CI", style="magenta", justify="right")
    table.add_row(
        "Bound corrected",
        f"{(result.bound.corrected_delta_g_kcal_mol or result.bound.delta_g_kcal_mol):.6f}",
        f"{(result.bound.corrected_propagated_sem_kcal_mol or result.bound.propagated_sem_kcal_mol):.6f}",
        _format_ci95(result.bound.corrected_bootstrap_ci95 or result.bound.bootstrap_ci95),
    )
    table.add_row(
        "Water",
        f"{result.water.delta_g_kcal_mol:.6f}",
        f"{result.water.propagated_sem_kcal_mol:.6f}",
        _format_ci95(result.water.bootstrap_ci95),
    )
    table.add_row("Final ddG", f"{result.ddg_kcal_mol:.6f}", f"{result.propagated_sem_kcal_mol:.6f}", _format_ci95(result.bootstrap_ci95))
    console.print(table)
    console.print(f"[dim]Bound TI decoupling: {_decoupling_scheme_detail(decoupling_scheme_for_result(result.bound))}[/dim]")
    console.print(f"[dim]Water TI decoupling: {_decoupling_scheme_detail(decoupling_scheme_for_result(result.water))}[/dim]")
    console.print("[dim]ddG = (dG_bound_ti + restraint_correction) - dG_water[/dim]")
    console.print(f"[dim]Saved analysis outputs to {result.output_dir}[/dim]")
