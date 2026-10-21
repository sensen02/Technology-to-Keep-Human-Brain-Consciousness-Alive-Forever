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
from collections.abc import Mapping
from copy import deepcopy
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
        self._failure_reason = None

    @property
    def failed(self):
        return self._failure_reason is not None

    @property
    def failure_reason(self):
        return self._failure_reason

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
        # The provider table describes the old stack; validate again before run.
        self._providers = {}
        self.wired = False
        return self

    def by_name(self, name):
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(name)

    # ------------------------------------------------------------------
    def wire(self):
        """Validate the stack: feed-forward only, units compatible."""
        # Never expose a partially validated table, including after a failed rewire.
        self.wired = False
        self._providers = {}
        providers = {}
        names = set()
        for i, layer in enumerate(self.layers):
            if layer.name in names:
                raise WiringError(f"duplicate layer name {layer.name!r}")
            names.add(layer.name)
            if not layer.enabled:
                continue
            # 1. requirements must already be provided by EARLIER layers
            for port, unit in layer.requires.items():
                if port in layer.provides:
                    raise FeedforwardViolation(
                        f"{layer.name} both requires and provides {port!r}")
                if port not in providers:
                    later = [l.name for l in self.layers[i + 1:]
                             if l.enabled and port in l.provides]
                    if later:
                        raise FeedforwardViolation(
                            f"{layer.name} requires {port!r}, which is only "
                            f"provided by a LATER layer {later}. The project's "
                            f"coupling rule forbids reading forwards or reaching "
                            f"back to ground truth.")
                    raise WiringError(
                        f"{layer.name} requires {port!r}, which nothing provides.")
                prov_layer, prov_unit = providers[port]
                if self.strict_units:
                    try:
                        factor = Quantity(1.0, prov_unit).to(unit).value
                        scale, _ = unit_of(unit)
                        if (not math.isfinite(factor) or factor <= 0
                                or not math.isfinite(scale) or scale <= 0):
                            raise UnitError("unit scale must be finite and positive")
                    except (UnitError, OverflowError, ZeroDivisionError) as exc:
                        raise WiringError(
                            f"{layer.name} requires {port!r} in {unit!r} but "
                            f"{prov_layer.name} provides it in {prov_unit!r} "
                            f"({exc})")
            # 2. publish
            for port, unit in layer.provides.items():
                if self.strict_units:
                    try:
                        scale, _ = unit_of(unit)
                        if not math.isfinite(scale) or scale <= 0:
                            raise UnitError("unit scale must be finite and positive")
                    except (UnitError, OverflowError) as exc:
                        raise WiringError(
                            f"{layer.name} provides {port!r} with invalid unit "
                            f"{unit!r} ({exc})") from exc
                if port in providers:
                    raise WiringError(
                        f"{port!r} is provided twice: by "
                        f"{providers[port][0].name} and {layer.name}")
                providers[port] = (layer, unit)
        self._providers = providers
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
        summaries for the monitoring layer. Clocks must be finite time
        quantities; step sizes must be positive. Zero duration is a no-op
        (no initialization, layer steps, or callbacks).

        Contracts are revalidated and snapshotted for each run. Changes made
        during a run take effect on the next run. Each hook gets private input
        copies (also across substeps), so local mutation cannot write upstream.
        Strict mode rescales numeric inputs to each consumer's required unit;
        published state remains in provider units. Quantity outputs are normalized
        to raw numbers in their declared units. Factor-one inputs retain their
        raw type; nonnumeric payloads are supported only without rescaling.
        strict_units=False explicitly bypasses conversion and normalization.

        This is not transactional. Any runtime failure permanently fails this
        Pipeline; recovery requires a NEW Pipeline and NEW layer instances.
        There is no reset/rollback guarantee, nor global enforcement against
        reusing damaged layers in another Pipeline. Successful repeated runs
        retain the layer's lifecycle semantics. Preflight errors remain retryable.
        """
        if self.failed:
            raise RuntimeError(
                f"pipeline failed ({self.failure_reason}); recovery requires "
                "a NEW Pipeline and NEW layer instances")
        # Layer declarations and enabled/strict_units remain publicly mutable.
        self.wire()
        dur_s = self._time_seconds(duration, "duration", allow_zero=True)
        dt_s = (self._time_seconds(dt_outer, "dt_outer")
                if dt_outer is not None else dur_s)
        active = [l for l in self.layers if l.enabled]
        strict_units = self.strict_units
        contracts = {id(l): (dict(l.requires), dict(l.provides)) for l in active}
        input_units = {
            id(l): {k: (self._providers[k][1], unit)
                    for k, unit in contracts[id(l)][0].items()}
            for l in active
        }
        # Validate every clock before initialization or any numerical mutation.
        layer_dt = {id(l): self._time_seconds(l.target_dt, f"{l.name}.target_dt")
                    for l in active}
        state = dict(state or {})
        if dur_s == 0:
            return state, []
        n_outer = max(1, int(math.ceil(dur_s / dt_s)))
        dt_s = dur_s / n_outer

        phase = "initialization"
        try:
            for l in active:
                phase = f"{l.name}.initialize"
                l.initialize(self._read_inputs(state, input_units[id(l)], strict_units))

            trace = []
            for step in range(n_outer):
                # a layer may only see the ports it declared
                for l in active:
                    phase = f"{l.name}.step"
                    dt_layer = layer_dt[id(l)]
                    n_sub = max(1, int(math.ceil(dt_s / dt_layer)))
                    sub_dt = Quantity(dt_s / n_sub, "s")
                    required, provided = contracts[id(l)]
                    inputs = self._read_inputs(state, input_units[id(l)], strict_units)
                    outs = None
                    for _ in range(n_sub):
                        outs = l.step(sub_dt, deepcopy(inputs))
                        if outs is not None:
                            if not isinstance(outs, Mapping):
                                raise WiringError(
                                    f"{l.name} step must return a mapping or None")
                            outs = dict(outs)
                            undeclared = outs.keys() - provided.keys()
                            if undeclared:
                                raise WiringError(
                                    f"{l.name} returned undeclared ports "
                                    f"{sorted(undeclared, key=repr)!r}")
                            if strict_units:
                                # Validate ALL values before publishing any of them.
                                outs = {k: (v.to(provided[k]).value
                                            if isinstance(v, Quantity) else v)
                                        for k, v in outs.items()}
                        l.n_steps += 1
                    if outs is not None:
                        state.update(outs)
                if on_step is not None:
                    phase = "on_step callback"
                    on_step(step, state)
                phase = "trace summary"
                trace.append(self._scalars(state))
            return state, trace
        except BaseException as exc:
            self._failure_reason = f"{phase}: {type(exc).__name__}: {exc}"
            raise

    @staticmethod
    def _read_inputs(state, units, strict_units):
        inputs = {}
        for port, (provider_unit, required_unit) in units.items():
            value = state.get(port)
            if strict_units and value is not None:
                # Seed state uses the same provider-unit convention as publication.
                if isinstance(value, Quantity):
                    value = value.to(provider_unit).value
                factor = Quantity(1.0, provider_unit).to(required_unit).value
                if factor != 1.0:
                    try:
                        array = np.asarray(value)
                        if array.dtype.kind not in "iufc":
                            raise TypeError("expected numeric scalar, array or list")
                        scaled = array * factor
                        value = scaled.item() if scaled.ndim == 0 else scaled
                    except (TypeError, ValueError) as exc:
                        raise WiringError(
                            f"cannot rescale nonnumeric port {port!r} from "
                            f"{provider_unit!r} to {required_unit!r}") from exc
            inputs[port] = value
        return deepcopy(inputs)

    @staticmethod
    def _time_seconds(value, name, allow_zero=False):
        if not isinstance(value, Quantity):
            raise TypeError(f"{name} must be a Quantity with time units")
        seconds = value.to("s").value
        if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
            bound = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {bound}")
        return seconds

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
