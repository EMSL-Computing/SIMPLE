from __future__ import annotations

from pathlib import Path
import struct


_NETCDF_CLASSIC_MAGIC = b"CDF\x01"
_NETCDF_64BIT_MAGIC = b"CDF\x02"


def count_trajectory_frames(path: str | Path | None) -> int | None:
    if path is None:
        return None
    target = Path(path).expanduser()
    if not target.exists() or not target.is_file():
        return None
    try:
        with target.open("rb") as handle:
            magic = handle.read(4)
            if magic not in {_NETCDF_CLASSIC_MAGIC, _NETCDF_64BIT_MAGIC}:
                return None
            raw_numrecs = handle.read(4)
            if len(raw_numrecs) != 4:
                return None
            return int(struct.unpack(">I", raw_numrecs)[0])
    except OSError:
        return None
