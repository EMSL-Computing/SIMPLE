from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

import typer


_SEARCH_SUBDIRECTORIES: ContextVar[bool] = ContextVar(
    "simple_search_subdirectories",
    default=True,
)


def search_subdirectories_enabled() -> bool:
    """Return the automatic-discovery scope for the current launcher run."""

    return _SEARCH_SUBDIRECTORIES.get()


@contextmanager
def subdirectory_search_scope(enabled: bool) -> Iterator[None]:
    """Temporarily apply one discovery choice across an interactive wizard."""

    token = _SEARCH_SUBDIRECTORIES.set(bool(enabled))
    try:
        yield
    finally:
        _SEARCH_SUBDIRECTORIES.reset(token)


def contains_subdirectories(search_root: str | Path) -> bool:
    """Check cheaply and stop as soon as the first subdirectory is found."""

    root = Path(search_root).expanduser().resolve()
    try:
        return any(path.is_dir() for path in root.iterdir())
    except OSError:
        return False


def prompt_for_subdirectory_search(search_root: str | Path = ".") -> bool:
    """Ask once before an interactive launcher searches below its start folder."""

    root = Path(search_root).expanduser().resolve()
    if not contains_subdirectories(root):
        return False
    return typer.confirm(
        f"Subdirectories were found under {root}. Search them for existing jobs/data?",
        default=False,
    )
