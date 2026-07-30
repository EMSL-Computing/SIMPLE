from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import re
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class MissingResidueBlock:
    chain_id: str
    start_resname: str
    start_resseq: str
    start_icode: str
    end_resname: str
    end_resseq: str
    end_icode: str
    residue_names: tuple[str, ...]
    source: str = "REMARK 465"

    @property
    def length(self) -> int:
        return len(self.residue_names)

    @property
    def start_label(self) -> str:
        return f"{self.start_resname} {_display_chain(self.chain_id)}:{self.start_resseq}{self.start_icode}"

    @property
    def end_label(self) -> str:
        return f"{self.end_resname} {_display_chain(self.chain_id)}:{self.end_resseq}{self.end_icode}"

    @property
    def range_label(self) -> str:
        if self.start_resseq == self.end_resseq and self.start_icode == self.end_icode:
            return self.start_label
        return f"{self.start_label} -> {self.end_label}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["length"] = self.length
        payload["range_label"] = self.range_label
        return payload


@dataclass(frozen=True, slots=True)
class BoundaryResidue:
    chain: str
    seqid: str

    @property
    def label(self) -> str:
        return format_residue_locator(self.chain, self.seqid)

    def to_dict(self) -> dict[str, str]:
        return {"chain": self.chain, "seqid": self.seqid, "label": self.label}


@dataclass(slots=True)
class MissingLoopSummary:
    detection_status: str
    detection_message: str | None = None
    remark_465_present: bool = False
    internal_blocks: list[MissingResidueBlock] = field(default_factory=list)
    terminal_blocks: list[MissingResidueBlock] = field(default_factory=list)
    boundary_residues: list[BoundaryResidue] = field(default_factory=list)

    @property
    def has_missing_blocks(self) -> bool:
        return bool(self.internal_blocks or self.terminal_blocks)

    @property
    def has_internal_blocks(self) -> bool:
        return bool(self.internal_blocks)

    def boundary_residue_locators(self) -> set[tuple[str, str]]:
        return {(item.chain, item.seqid) for item in self.boundary_residues}

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_status": self.detection_status,
            "detection_message": self.detection_message,
            "remark_465_present": self.remark_465_present,
            "internal_blocks": [item.to_dict() for item in self.internal_blocks],
            "terminal_blocks": [item.to_dict() for item in self.terminal_blocks],
            "boundary_residues": [item.to_dict() for item in self.boundary_residues],
        }


@dataclass(frozen=True, slots=True)
class _MissingResidueEntry:
    resname: str
    chain_id: str
    resseq: str
    icode: str


def _display_chain(chain_id: str) -> str:
    return chain_id.strip() or "(blank)"


def format_residue_locator(chain: str, seqid: str) -> str:
    return f"{_display_chain(chain)}:{seqid.strip()}"


def _boundary_sort_key(item: tuple[str, str]) -> tuple[str, int, int | str]:
    chain, seqid = item
    token = seqid.strip()
    if token.lstrip("-").isdigit():
        return chain, 0, int(token)
    return chain, 1, token


def _parse_resseq_token(token: str) -> tuple[str, str]:
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", token.strip())
    if match is None:
        raise ValueError(f"Invalid residue sequence token: {token}")
    return match.group(1), match.group(2)


def _parse_missing_residue_entries(lines: Sequence[str]) -> list[_MissingResidueEntry]:
    entries: list[_MissingResidueEntry] = []
    for raw in lines:
        if not raw.startswith("REMARK 465"):
            continue
        body = raw[10:].strip()
        if not body or "MISSING" in body or body.startswith("RES ") or body.startswith("M RES") or body.startswith("I="):
            continue
        tokens = body.split()
        if not tokens:
            continue
        if tokens[0].isdigit() and len(tokens) >= 4:
            tokens = tokens[1:]
        if len(tokens) < 2:
            continue
        resname = tokens[0].strip().upper()
        if len(tokens) == 2:
            chain_id = ""
            seq_token = tokens[1]
        else:
            chain_id = tokens[1].strip()
            seq_token = tokens[2]
        try:
            resseq, icode = _parse_resseq_token(seq_token)
        except ValueError:
            continue
        entries.append(
            _MissingResidueEntry(
                resname=resname,
                chain_id=chain_id,
                resseq=resseq,
                icode=icode,
            )
        )
    return entries


def _entries_are_contiguous(previous: _MissingResidueEntry, current: _MissingResidueEntry) -> bool:
    if previous.chain_id != current.chain_id:
        return False
    if previous.icode or current.icode:
        return False
    try:
        return int(current.resseq) == int(previous.resseq) + 1
    except ValueError:
        return False


def find_missing_residue_blocks(pdb_path: Path) -> list[MissingResidueBlock]:
    lines = Path(pdb_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    entries = _parse_missing_residue_entries(lines)
    if not entries:
        return []

    blocks: list[MissingResidueBlock] = []
    current: list[_MissingResidueEntry] = []
    for entry in entries:
        if not current or _entries_are_contiguous(current[-1], entry):
            current.append(entry)
            continue
        blocks.append(_block_from_entries(current))
        current = [entry]

    if current:
        blocks.append(_block_from_entries(current))
    return blocks


def _block_from_entries(entries: list[_MissingResidueEntry]) -> MissingResidueBlock:
    return MissingResidueBlock(
        chain_id=entries[0].chain_id,
        start_resname=entries[0].resname,
        start_resseq=entries[0].resseq,
        start_icode=entries[0].icode,
        end_resname=entries[-1].resname,
        end_resseq=entries[-1].resseq,
        end_icode=entries[-1].icode,
        residue_names=tuple(item.resname for item in entries),
    )


def _normalize_residue_key_parts(chain_id: str, resseq: str, icode: str) -> tuple[str, str, str]:
    chain = (chain_id or "").strip()
    return ("" if chain == "_" else chain, str(resseq).strip(), str(icode or "").strip())


def _observed_residue_keys(lines: Sequence[str]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for raw in lines:
        if not raw.startswith(("ATOM", "HETATM")) or len(raw) < 27:
            continue
        keys.add(_normalize_residue_key_parts(raw[21:22], raw[22:26], raw[26:27]))
    return keys


def partition_missing_residue_blocks(
    pdb_path: Path,
    blocks: Sequence[MissingResidueBlock] | None = None,
) -> tuple[list[MissingResidueBlock], list[MissingResidueBlock]]:
    lines = Path(pdb_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if blocks is None:
        blocks = find_missing_residue_blocks(pdb_path)
    observed = _observed_residue_keys(lines)
    internal: list[MissingResidueBlock] = []
    terminal: list[MissingResidueBlock] = []
    for block in blocks:
        if block.start_icode or block.end_icode:
            terminal.append(block)
            continue
        try:
            prev_resseq = str(int(block.start_resseq) - 1)
            next_resseq = str(int(block.end_resseq) + 1)
        except ValueError:
            terminal.append(block)
            continue
        prev_key = _normalize_residue_key_parts(block.chain_id, prev_resseq, "")
        next_key = _normalize_residue_key_parts(block.chain_id, next_resseq, "")
        if prev_key in observed and next_key in observed:
            internal.append(block)
        else:
            terminal.append(block)
    return internal, terminal


def _boundary_residues_for_blocks(
    pdb_path: Path,
    blocks: Sequence[MissingResidueBlock],
) -> list[BoundaryResidue]:
    lines = Path(pdb_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    observed = _observed_residue_keys(lines)
    boundary_keys: set[tuple[str, str]] = set()
    for block in blocks:
        if block.start_icode or block.end_icode:
            continue
        try:
            prev_resseq = str(int(block.start_resseq) - 1)
            next_resseq = str(int(block.end_resseq) + 1)
        except ValueError:
            continue
        prev_key = _normalize_residue_key_parts(block.chain_id, prev_resseq, "")
        next_key = _normalize_residue_key_parts(block.chain_id, next_resseq, "")
        if prev_key in observed:
            boundary_keys.add((prev_key[0], prev_key[1]))
        if next_key in observed:
            boundary_keys.add((next_key[0], next_key[1]))
    return [
        BoundaryResidue(chain=chain, seqid=seqid)
        for chain, seqid in sorted(boundary_keys, key=_boundary_sort_key)
    ]


def analyze_missing_loops(pdb_path: str | Path) -> MissingLoopSummary:
    path = Path(pdb_path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    remark_465_present = any(line.startswith("REMARK 465") for line in lines)
    if not remark_465_present:
        return MissingLoopSummary(
            detection_status="unavailable",
            detection_message=(
                "Missing-loop detection is unavailable because this structure does not contain PDB REMARK 465 "
                "missing-residue annotations. This commonly happens for files converted from mmCIF, so internal "
                "loop repair and boundary-aware PROPKA exclusions cannot be inferred automatically."
            ),
            remark_465_present=False,
        )

    blocks = find_missing_residue_blocks(path)
    internal_blocks, terminal_blocks = partition_missing_residue_blocks(path, blocks)
    return MissingLoopSummary(
        detection_status="available",
        detection_message=None,
        remark_465_present=True,
        internal_blocks=internal_blocks,
        terminal_blocks=terminal_blocks,
        boundary_residues=_boundary_residues_for_blocks(path, internal_blocks),
    )


def write_missing_loop_report(report_path: Path, summary: MissingLoopSummary) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "category\tchain\tstart_resseq\tstart_icode\tend_resseq\tend_icode\tlength\tboundary_residues\tresidue_names\trange\n"
    ]
    boundary_by_block: dict[tuple[str, str, str, str, str], str] = {}
    for block in summary.internal_blocks:
        boundary_labels: list[str] = []
        try:
            prev_seqid = str(int(block.start_resseq) - 1)
            next_seqid = str(int(block.end_resseq) + 1)
            boundary_labels = [
                format_residue_locator(block.chain_id, prev_seqid),
                format_residue_locator(block.chain_id, next_seqid),
            ]
        except ValueError:
            boundary_labels = []
        boundary_by_block[(block.chain_id, block.start_resseq, block.start_icode, block.end_resseq, block.end_icode)] = ",".join(boundary_labels)

    def append_block(category: str, block: MissingResidueBlock) -> None:
        key = (block.chain_id, block.start_resseq, block.start_icode, block.end_resseq, block.end_icode)
        lines.append(
            "\t".join(
                [
                    category,
                    block.chain_id or "",
                    block.start_resseq,
                    block.start_icode,
                    block.end_resseq,
                    block.end_icode,
                    str(block.length),
                    boundary_by_block.get(key, ""),
                    ",".join(block.residue_names),
                    block.range_label,
                ]
            )
            + "\n"
        )

    for block in summary.internal_blocks:
        append_block("internal", block)
    for block in summary.terminal_blocks:
        append_block("terminal", block)
    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


def _load_pdbfixer_api():
    try:
        pdbfixer_mod = importlib.import_module("pdbfixer")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing-loop repair requested, but the Python package 'pdbfixer' is not installed. "
            "Install PDBFixer and OpenMM first."
        ) from exc

    try:
        openmm_app = importlib.import_module("openmm.app")
    except ModuleNotFoundError:
        try:
            openmm_app = importlib.import_module("simtk.openmm.app")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDBFixer is installed, but OpenMM could not be imported. "
                "Install OpenMM so repaired coordinates can be written."
            ) from exc
    return pdbfixer_mod.PDBFixer, openmm_app.PDBFile


def repair_internal_missing_loops(source_pdb: Path, out_pdb: Path) -> Path:
    summary = analyze_missing_loops(source_pdb)
    if summary.detection_status != "available":
        message = summary.detection_message or "Missing-loop detection is unavailable for this structure."
        raise RuntimeError(message)
    if not summary.internal_blocks:
        return Path(source_pdb)

    PDBFixer, PDBFile = _load_pdbfixer_api()
    fixer = PDBFixer(filename=str(source_pdb))
    fixer.findMissingResidues()

    retained = {}
    chains = list(fixer.topology.chains())
    for key, residue_names in dict(getattr(fixer, "missingResidues", {}) or {}).items():
        try:
            chain_index, residue_index = key
            chain = chains[int(chain_index)]
            residue_count = len(list(chain.residues()))
            residue_index = int(residue_index)
        except Exception:
            continue
        if 0 < residue_index < residue_count:
            retained[key] = residue_names
    fixer.missingResidues = retained

    # PDBFixer inserts missing residues when addMissingAtoms() is called. We clear the
    # missing-atom maps first so existing residues are not also repaired here.
    fixer.findMissingAtoms()
    fixer.missingAtoms = {}
    fixer.missingTerminals = {}
    fixer.addMissingAtoms()

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w", encoding="utf-8") as handle:
        try:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
        except TypeError:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return out_pdb
