"""Single-compartment Hodgkin–Huxley dynamics coupled to Cartesian ion fields."""

from dataclasses import dataclass
from math import ceil, exp, expm1, isfinite, log
from numbers import Integral

import numpy as np

from ..engine import AdaptiveWorld
from ..world import Position

GAS_CONSTANT_J_MOL_K = 8.31446261815324
FARADAY_C_MOL = 96_485.33212331002
REFERENCE_TEMPERATURE_K = 279.45


def nernst_voltage(inside_mol_m3: float, outside_mol_m3: float,
                   valence: int, temperature_K: float) -> float:
    """Return the ion equilibrium voltage, inside relative to outside."""
    if not isinstance(valence, Integral) or valence == 0:
        raise ValueError("ion valence must be a nonzero integer")
    if not all(isfinite(value) and value > 0 for value in
               (inside_mol_m3, outside_mol_m3, temperature_K)):
        raise ValueError("ion concentrations and temperature must be positive and finite")
    return (GAS_CONSTANT_J_MOL_K * temperature_K / (valence * FARADAY_C_MOL)
            * (log(outside_mol_m3) - log(inside_mol_m3)))


@dataclass(frozen=True)
class HHParameters:
    """SI membrane parameters. Defaults reproduce the classic squid-axon kinetics."""

    area_m2: float = 1e-9
    temperature_K: float = REFERENCE_TEMPERATURE_K
    sodium_inside_mol_m3: float = 40.0
    potassium_inside_mol_m3: float = 400.0
    capacitance_F_m2: float = 0.01
    sodium_conductance_S_m2: float = 1200.0
    potassium_conductance_S_m2: float = 360.0
    leak_conductance_S_m2: float = 3.0
    leak_reversal_V: float = -0.054387
    kinetic_q10: float = 3.0

    def __post_init__(self):
        positive = (self.area_m2, self.temperature_K, self.sodium_inside_mol_m3,
                    self.potassium_inside_mol_m3, self.capacitance_F_m2, self.kinetic_q10)
        conductances = (self.sodium_conductance_S_m2, self.potassium_conductance_S_m2,
                        self.leak_conductance_S_m2)
        if (not all(isfinite(value) and value > 0 for value in positive)
                or not all(isfinite(value) and value >= 0 for value in conductances)
                or not isfinite(self.leak_reversal_V)):
            raise ValueError("invalid membrane parameters")


@dataclass(frozen=True)
class MembraneSample:
    time_s: float
    voltage_V: float
    sodium_outside_mol_m3: float
    potassium_outside_mol_m3: float
    sodium_reversal_V: float
    potassium_reversal_V: float
    sodium_current_A: float
    potassium_current_A: float
    leak_current_A: float
    injected_current_A: float
    m: float
    h: float
    n: float


def _vtrap(x: float) -> float:
    if abs(x) < 1e-7:
        return 1 - x / 2
    return x / expm1(x)


def _gate_rates(voltage_V: float):
    # Published rate curves expressed in absolute voltage (mV), rates in ms⁻¹.
    v = voltage_V * 1000
    opening = (_vtrap(-(v + 40) / 10),
               0.07 * exp(-(v + 65) / 20),
               0.1 * _vtrap(-(v + 55) / 10))
    closing = (4 * exp(-(v + 65) / 18),
               1 / (1 + exp(-(v + 35) / 10)),
               0.125 * exp(-(v + 65) / 80))
    return opening, closing


class HodgkinHuxley:
    """A point membrane with fixed intracellular reservoirs and local extracellular input."""

    def __init__(
        self, position_m: Position, sodium_field: str, potassium_field: str, *,
        parameters: HHParameters = HHParameters(), initial_voltage_V: float = -0.065,
        max_step_s: float = 25e-6, spike_threshold_V: float = 0.0,
    ):
        if len(position_m) != 3 or not all(map(isfinite, position_m)):
            raise ValueError("position must have three finite coordinates")
        if sodium_field == potassium_field or not sodium_field or not potassium_field:
            raise ValueError("distinct sodium and potassium fields are required")
        if (not all(map(isfinite, (initial_voltage_V, max_step_s, spike_threshold_V)))
                or max_step_s <= 0):
            raise ValueError("voltages must be finite and maximum step must be finite and positive")
        self.position_m = tuple(position_m)
        self.sodium_field = sodium_field
        self.potassium_field = potassium_field
        self.parameters = parameters
        self.max_step_s = max_step_s
        self.spike_threshold_V = spike_threshold_V
        self._rate_scale = 1000 * parameters.kinetic_q10 ** (
            (parameters.temperature_K - REFERENCE_TEMPERATURE_K) / 10)
        opening, closing = _gate_rates(initial_voltage_V)
        self._state = np.array((initial_voltage_V, *(a / (a + b) for a, b in zip(opening, closing))))
        self._injected_A = 0.0
        self._world: AdaptiveWorld | None = None
        self._sample: MembraneSample | None = None

    @property
    def sample(self) -> MembraneSample:
        if self._sample is None:
            raise RuntimeError("neuron must be attached before reading a membrane sample")
        return self._sample

    def set_current(self, current_A: float) -> None:
        """Set the inward current used for the next world step."""
        if not isfinite(current_A):
            raise ValueError("injected current must be finite")
        self._injected_A = current_A

    def _environment(self, world: AdaptiveWorld):
        sodium = world.sample_concentration(self.sodium_field, self.position_m)
        potassium = world.sample_concentration(self.potassium_field, self.position_m)
        p = self.parameters
        ena = nernst_voltage(p.sodium_inside_mol_m3, sodium, 1, p.temperature_K)
        ek = nernst_voltage(p.potassium_inside_mol_m3, potassium, 1, p.temperature_K)
        return sodium, potassium, ena, ek

    def attach(self, world: AdaptiveWorld, name: str) -> None:
        if self._world is not None:
            raise RuntimeError("neuron is already attached")
        environment = self._environment(world)
        self._sample = self._observe(world.time_s, self._state, environment)
        self._world = world

    def _current_density(self, state, ena, ek):
        voltage, m, h, n = state
        p = self.parameters
        return (p.sodium_conductance_S_m2 * m ** 3 * h * (voltage - ena),
                p.potassium_conductance_S_m2 * n ** 4 * (voltage - ek),
                p.leak_conductance_S_m2 * (voltage - p.leak_reversal_V))

    def _observe(self, time_s, state, environment):
        currents = self._current_density(state, *environment[2:])
        return MembraneSample(time_s, float(state[0]), *environment,
                              *(float(current * self.parameters.area_m2) for current in currents),
                              self._injected_A, *map(float, state[1:]))

    def _derivative(self, state, ena, ek, input_A_m2):
        opening, closing = _gate_rates(state[0])
        gates = (self._rate_scale * (a * (1 - value) - b * value)
                 for a, b, value in zip(opening, closing, state[1:]))
        voltage_rate = (input_A_m2 - sum(self._current_density(state, ena, ek))) / self.parameters.capacitance_F_m2
        return np.array((voltage_rate, *gates))

    def step(self, world: AdaptiveWorld, name: str, dt_s: float) -> None:
        if world is not self._world:
            raise RuntimeError("neuron must be attached to this world")
        environment = self._environment(world)
        ena, ek = environment[2:]
        drive = self._injected_A / self.parameters.area_m2
        count = max(1, ceil(dt_s / self.max_step_s))
        duration = dt_s / count
        state = self._state
        spikes = []
        for index in range(count):
            a = self._derivative(state, ena, ek, drive)
            b = self._derivative(state + duration * a / 2, ena, ek, drive)
            c = self._derivative(state + duration * b / 2, ena, ek, drive)
            d = self._derivative(state + duration * c, ena, ek, drive)
            updated = state + (duration / 6) * (a + 2 * b + 2 * c + d)
            if not np.isfinite(updated).all() or np.any((updated[1:] < 0) | (updated[1:] > 1)):
                raise FloatingPointError("invalid Hodgkin–Huxley state")
            if state[0] < self.spike_threshold_V <= updated[0]:
                crossing = (self.spike_threshold_V - state[0]) / (updated[0] - state[0])
                spikes.append(float(world.time_s + (index + crossing) * duration))
            state = updated
        self._sample = self._observe(world.time_s + dt_s, state, environment)
        self._state = state
        for time_s in spikes:
            world.emit(name, "spike", time_s=time_s, voltage_V=self.spike_threshold_V)
