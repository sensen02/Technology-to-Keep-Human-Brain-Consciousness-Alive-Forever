"""engine.layers -- the layered, strictly feed-forward engine core.

THE RULE THIS ENFORCES
----------------------
``ORGANISM_SCALE_PLAN.md`` states the project's own coupling rule:

    "耦合规则必须是单向可追踪的：任一层读取上一层提供的数组，不反向读取
     「真值」，否则结果无法在真实实验里复现。"

("Coupling must be one-way and traceable: each layer reads the arrays provided
by the layer above it and must never reach back to the ground truth, otherwise
the result could not be reproduced in a real experiment.")

That was a convention.  Here it is a mechanism:

  * a Layer declares `requires` (ports it may read) and `provides` (ports it
    publishes), each with an explicit unit;
  * a Pipeline only ever hands a layer the ports it declared -- it physically
    cannot read anything else;
  * wiring is checked before the run: every requirement must be satisfied by a
    layer that comes strictly earlier, and the units must be compatible.  A
    requirement satisfied only by a later layer is a FeedforwardViolation.

TIME SCALES
-----------
The project spans calcium (seconds) to tissue growth (hours) -- 3 to 4 orders
of magnitude.  Each layer therefore declares its own `target_dt`, and the
pipeline sub-steps it inside the outer step.  Inputs are held constant across
the sub-steps, i.e. this is explicit first-order operator splitting; that is a
numerical statement, not an approximation that should be quoted as biology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .units import Quantity, UnitError, unit_of

__all__ = ["Port", "Layer", "Pipeline", "FeedforwardViolation", "WiringError"]


class FeedforwardViolation(RuntimeError):
    pass


class WiringError(RuntimeError):
    pass


@dataclass(frozen=True)
class Port:
    name: str
    unit: str
    description: str = ""


class Layer:
    """Base class.  Subclasses set name/requires/provides and implement step()."""

    name = "layer"
    requires: dict = {}      # port name -> unit string
    provides: dict = {}      # port name -> unit string
    target_dt = Quantity(1.0, "s")
    enabled = True

    def __init__(self, **params):
        self.cfg = dict(params)
        self.state = {}
        self.n_steps = 0

    # -- lifecycle hooks -------------------------------------------------
    def initialize(self, inputs):
        """Called once before the run with the initial inputs."""

    def step(self, dt, inputs):
        """Advance this layer by `dt` (a Quantity).  Return {port: value}.

        `inputs` contains ONLY the ports declared in `requires`.
        """
        raise NotImplementedError

    # -- introspection ---------------------------------------------------
    def required_ports(self):
        return {k: Port(k, v) for k, v in self.requires.items()}

    def provided_ports(self):
        return {k: Port(k, v) for k, v in self.provides.items()}

    def describe(self):
        return (f"{self.name}: requires={sorted(self.requires)} "
                f"provides={sorted(self.provides)} dt={self.target_dt}")


class Pipeline:
    """An ordered, unit-checked, strictly feed-forward stack of layers."""

    def __init__(self, name="pipeline", strict_units=True):
        self.name = name
        self.layers: list[Layer] = []
        self.strict_units = strict_units
        self._providers: dict[str, tuple[Layer, str]] = {}
        self.wired = False
        self.log = []

    # ------------------------------------------------------------------
    def add(self, layer):
        if self.wired:
            raise WiringError("cannot add a layer after wiring")
        if layer.name in [l.name for l in self.layers]:
            raise WiringError(f"duplicate layer name {layer.name!r}")
        self.layers.append(layer)
        return self

    def remove(self, name):
        self.layers = [l for l in self.layers if l.name != name]
        return self

    def by_name(self, name):
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(name)

    # ------------------------------------------------------------------
    def wire(self):
        """Validate the stack: feed-forward only, units compatible."""
        self._providers = {}
        for i, layer in enumerate(self.layers):
            if not layer.enabled:
                continue
            # 1. requirements must already be provided by EARLIER layers
            for port, unit in layer.requires.items():
                if port in layer.provides:
                    raise FeedforwardViolation(
                        f"{layer.name} both requires and provides {port!r}")
                if port not in self._providers:
                    later = [l.name for l in self.layers[i + 1:]
                             if port in l.provides]
                    if later:
                        raise FeedforwardViolation(
                            f"{layer.name} requires {port!r}, which is only "
                            f"provided by a LATER layer {later}. The project's "
                            f"coupling rule forbids reading forwards or reaching "
                            f"back to ground truth.")
                    raise WiringError(
                        f"{layer.name} requires {port!r}, which nothing provides.")
                prov_layer, prov_unit = self._providers[port]
                if self.strict_units and unit and prov_unit:
                    try:
                        Quantity(1.0, prov_unit).to(unit)
                    except UnitError as exc:
                        raise WiringError(
                            f"{layer.name} requires {port!r} in {unit!r} but "
                            f"{prov_layer.name} provides it in {prov_unit!r} "
                            f"({exc})")
            # 2. publish
            for port, unit in layer.provides.items():
                if port in self._providers:
                    raise WiringError(
                        f"{port!r} is provided twice: by "
                        f"{self._providers[port][0].name} and {layer.name}")
                self._providers[port] = (layer, unit)
        self.wired = True
        return self

    def ports(self):
        return dict(self._providers)

    def describe(self):
        lines = [f"pipeline {self.name!r} ({len(self.layers)} layers)"]
        for l in self.layers:
            flag = "" if l.enabled else "  [disabled]"
            lines.append("  " + l.describe() + flag)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def run(self, duration, dt_outer=None, state=None, on_step=None):
        """Run the stack.

        `duration` and `dt_outer` are Quantities with time units.  Returns
        (state, trace) where trace is a list of per-step dicts of scalar
        summaries for the monitoring layer.
        """
        if not self.wired:
            self.wire()
        dur_s = duration.si
        dt_s = (dt_outer or duration).si
        n_outer = max(1, int(math.ceil(dur_s / dt_s)))
        dt_s = dur_s / n_outer

        state = dict(state or {})
        active = [l for l in self.layers if l.enabled]
        for l in active:
            l.initialize({k: state.get(k) for k in l.requires})

        trace = []
        for step in range(n_outer):
            # a layer may only see the ports it declared
            for l in active:
                dt_layer = l.target_dt.si
                n_sub = max(1, int(math.ceil(dt_s / dt_layer)))
                sub_dt = Quantity(dt_s / n_sub, "s")
                inputs = {k: state.get(k) for k in l.requires}
                outs = None
                for _ in range(n_sub):
                    outs = l.step(sub_dt, inputs)
                    l.n_steps += 1
                if outs:
                    state.update(outs)
            if on_step is not None:
                on_step(step, state)
            trace.append(self._scalars(state))
        return state, trace

    @staticmethod
    def _scalars(state):
        out = {}
        for k, v in state.items():
            try:
                arr = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 0:
                out[k] = float(arr)
            else:
                out[k] = float(np.mean(arr))
        return out

    # ------------------------------------------------------------------
    def wiring_report(self):
        rows = [f"# 管线接线报告：{self.name}", "",
                "| 层 | 读取（端口 @ 单位） | 提供（端口 @ 单位） | 自身时间步 |",
                "|---|---|---|---|"]
        for l in self.layers:
            req = ", ".join(f"`{k}` @ {v}" for k, v in l.requires.items()) or "—"
            pro = ", ".join(f"`{k}` @ {v}" for k, v in l.provides.items()) or "—"
            rows.append(f"| {l.name} | {req} | {pro} | {l.target_dt} |")
        return "\n".join(rows)


# ----------------------------------------------------------------------
# Small standard layers used by the demos and tests
# ----------------------------------------------------------------------


class ConstantSource(Layer):
    """Publishes a constant field.  The only layer allowed to be a root."""

    name = "constant_source"

    def __init__(self, port, unit, value, target_dt=None, **kw):
        super().__init__(**kw)
        # `requires`/`provides` are class attributes on Layer, so each instance
        # must get its OWN dicts, otherwise the last instance wins globally.
        self.requires = {}
        self.provides = {port: unit}
        self._value = value
        if target_dt is not None:
            self.target_dt = (Quantity(target_dt, "s")
                              if isinstance(target_dt, (int, float)) else target_dt)

    def step(self, dt, inputs):
        return {k: self._value for k in self.provides}


class Integrator(Layer):
    """dX/dt = rate * input.  Used in tests to check wiring and dt handling."""

    name = "integrator"

    def __init__(self, name, in_port, in_unit, out_port, out_unit, rate=1.0, **kw):
        super().__init__(**kw)
        self.name = name
        self.requires = {in_port: in_unit}
        self.provides = {out_port: out_unit}
        self._in = in_port
        self._out = out_port
        self._rate = float(rate)

    def initialize(self, inputs):
        self.state.setdefault("x", 0.0)

    def step(self, dt, inputs):
        v = inputs.get(self._in)
        v = 0.0 if v is None else float(np.mean(np.asarray(v, dtype=float)))
        self.state["x"] += self._rate * v * dt.si
        return {self._out: self.state["x"]}


class RatioMonitor(Layer):
    """A monitoring layer: publishes a derived observable without feeding back."""

    name = "monitor"

    def __init__(self, name, in_port, in_unit, out_port, **kw):
        super().__init__(**kw)
        self.name = name
        self.requires = {in_port: in_unit}
        self.provides = {out_port: "1"}
        self._in, self._out = in_port, out_port

    def step(self, dt, inputs):
        v = inputs.get(self._in)
        arr = np.asarray(v, dtype=float) if v is not None else np.zeros(1)
        return {self._out: float(np.mean(arr))}
