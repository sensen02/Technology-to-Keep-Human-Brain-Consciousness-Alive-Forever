"""Shared voxel fields, module execution and simulation events."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Position = tuple[float, float, float]
Cell = tuple[int, int, int]
Field = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    shape: Cell
    spacing_m: float
    origin_m: Position = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(not isinstance(n, Integral) or n < 1 for n in self.shape):
            raise ValueError("shape must contain three positive integer sizes")
        if not isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("spacing_m must be finite and positive")
        if len(self.origin_m) != 3 or not all(map(isfinite, self.origin_m)):
            raise ValueError("origin_m must contain three finite coordinates")
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "origin_m", tuple(self.origin_m))

    @property
    def cell_volume_m3(self) -> float:
        return self.spacing_m ** 3

    def cell_at(self, position_m: Position) -> Cell:
        if len(position_m) != 3 or not all(map(isfinite, position_m)):
            raise ValueError("position_m must contain three finite coordinates")
        cell = []
        for coordinate, origin, n in zip(position_m, self.origin_m, self.shape):
            offset = (coordinate - origin) / self.spacing_m
            if not 0 <= offset < n:
                raise ValueError(f"position outside the world: {position_m}")
            cell.append(int(offset))
        return cell[0], cell[1], cell[2]

    def centers_m(self) -> tuple[Field, Field, Field]:
        """Return three sparse, broadcastable arrays of voxel centers in x, y, z order."""
        axes = [start + (np.arange(n) + 0.5) * self.spacing_m
                for start, n in zip(self.origin_m, self.shape)]
        x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
        return x, y, z


@dataclass(frozen=True)
class Event:
    time_s: float
    source: str
    kind: str
    data: dict[str, str | float | int]


class Module(Protocol):
    def attach(self, world: World, name: str) -> None: ...

    def step(self, world: World, name: str, dt_s: float) -> None: ...


class World:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._time_s = 0.0
        self._fields: dict[str, Field] = {}
        self.fields = MappingProxyType(self._fields)
        self._modules: dict[str, Module] = {}
        self._events: list[Event] = []
        self._stepping = False
        self._failed = False

    @property
    def time_s(self) -> float:
        return self._time_s

    def add_field(self, name: str, initial: float = 0.0) -> Field:
        if name in self.fields:
            raise ValueError(f"field already exists: {name}")
        if not isfinite(initial):
            raise ValueError("initial field value must be finite")
        field = np.full(self.grid.shape, initial, dtype=np.float64)
        self._fields[name] = field
        return field

    def sample(self, field: str, position_m: Position) -> float:
        return float(self.fields[field][self.grid.cell_at(position_m)])

    def add_module(self, name: str, module: Module) -> None:
        if self._stepping or self._failed:
            raise RuntimeError("modules can only be added between successful steps")
        if name in self._modules:
            raise ValueError(f"module already exists: {name}")
        module.attach(self, name)
        self._modules[name] = module

    def remove_module(self, name: str) -> Module:
        """Remove a module from future steps, retaining the shared fields it wrote."""
        if self._stepping or self._failed:
            raise RuntimeError("modules can only be removed between successful steps")
        return self._modules.pop(name)

    def emit(
        self, source: str, kind: str, *, time_s: float | None = None, **data: str | float | int
    ) -> None:
        self._events.append(Event(self.time_s if time_s is None else time_s, source, kind, data))

    def drain_events(self) -> list[Event]:
        events = sorted(self._events, key=lambda event: event.time_s)
        self._events.clear()
        return events

    def step(self, dt_s: float) -> None:
        end_s = self.time_s + dt_s
        if not isfinite(end_s) or end_s <= self.time_s:
            raise ValueError("dt_s must advance the finite world time")
        if self._stepping or self._failed:
            raise RuntimeError("the world is already stepping or a previous step failed")
        self._stepping = True
        source = "world"
        try:
            self._check_fields()
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for source, module in self._modules.items():
                    module.step(self, source, dt_s)
            source = "world"
            self._check_fields()
        except Exception as error:
            self._failed = True
            self.emit(source, "simulation_error", error=type(error).__name__, message=str(error))
            raise
        finally:
            self._stepping = False
        self._time_s = end_s

    def _check_fields(self) -> None:
        for name, values in self.fields.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite values in field: {name}")
