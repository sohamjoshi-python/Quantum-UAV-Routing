from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, order=True)
class Request:
    origin: int | str
    destination: int | str
    trr: float
    tplr: float
    t_star: float
    id: str
    x: Optional[float] = None
    y: Optional[float] = None
    dest_x: Optional[float] = None
    dest_y: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "id", str(self.id))

    def __repr__(self) -> str:
        return f"R{self.id}"


@dataclass
class Vehicle:
    qv: int | str
    tv: float
    Pv: list[Request] = field(default_factory=list)
    id: str = "vehicle"

    def __post_init__(self):
        self.id = str(self.id)

    def __repr__(self) -> str:
        return f"V{self.id}"


def trip_name(trip_fset) -> str:
    req_ids = sorted(str(r.id if hasattr(r, "id") else r) for r in trip_fset)
    return "T[" + ",".join(req_ids) + "]"