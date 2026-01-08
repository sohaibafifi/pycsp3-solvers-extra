from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConstraintCall:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
