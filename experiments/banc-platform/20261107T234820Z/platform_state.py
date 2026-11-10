"""Platform skeleton: internal energy state, virtual food patch, feeding gate, behaviour selector.

Every component here is a DECLARED ENGINEERING component except the frozen body and the
frozen connectome module. None of it is presented as a measured fly circuit, metabolism,
food-detection or feeding biomechanics. No XML change, no food collision geometry.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class EnergyParams:
    initial: float = .6
    drain_per_s: float = .02
    intake_per_s: float = .25
    min_energy: float = 0.
    max_energy: float = 1.

    def validate(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('energy parameters must be finite')
        if self.min_energy >= self.max_energy:
            raise ValueError('energy bounds must be ordered')
        if self.drain_per_s < 0 or self.intake_per_s < 0:
            raise ValueError('rates must be nonnegative')
        if not (self.min_energy <= self.initial <= self.max_energy):
            raise ValueError('initial energy must be within bounds')


class EnergyState:
    """Declared energy integrator. Engineered, not a metabolic measurement."""

    def __init__(self, params=None):
        self.p = params or EnergyParams()
        self.p.validate()
        self.energy = float(self.p.initial)
        self.time_s = 0.

    def update(self, dt_s, feeding=False):
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError('dt_s must be finite and nonnegative')
        delta = -self.p.drain_per_s * dt_s + (self.p.intake_per_s * dt_s if feeding else 0.)
        self.energy = float(np.clip(self.energy + delta, self.p.min_energy, self.p.max_energy))
        self.time_s += float(dt_s)
        return self.energy

    @property
    def hunger(self):
        return float(1. - self.energy)


@dataclass(frozen=True)
class FoodParams:
    center: tuple = (0., 0., 0.)
    radius: float = .004
    total_mass: float = 1.
    intake_per_s: float = .25

    def validate(self):
        if not (np.isfinite(self.radius) and self.radius > 0):
            raise ValueError('radius must be positive and finite')
        if not (np.isfinite(self.total_mass) and self.total_mass >= 0):
            raise ValueError('mass must be finite and nonnegative')
        if len(self.center) != 3 or not np.isfinite(self.center).all():
            raise ValueError('center must be a finite 3-vector')


class FoodPatch:
    """Virtual food volume: no collision geometry exists in the frozen body."""

    def __init__(self, params=None):
        self.p = params or FoodParams()
        self.p.validate()
        self.mass = float(self.p.total_mass)

    def contains(self, point):
        d = np.asarray(point, float) - np.asarray(self.p.center, float)
        return bool(np.linalg.norm(d) <= self.p.radius)

    def consume(self, dt_s):
        if not np.isfinite(dt_s) or dt_s < 0:
            raise ValueError('dt_s must be finite and nonnegative')
        amount = min(self.p.intake_per_s * dt_s, self.mass)
        self.mass = float(self.mass - amount)
        return float(amount)

    @property
    def depleted(self):
        return self.mass <= 0.


class BehaviourSelector:
    """Declared threshold rule on the energy state. NOT a neural circuit."""

    def __init__(self, forage_threshold=.5, satiety=.95):
        self.forage_threshold = float(forage_threshold)
        self.satiety = float(satiety)
        if not (0 <= self.forage_threshold <= self.satiety <= 1):
            raise ValueError('require 0 <= forage_threshold <= satiety <= 1')

    def decide(self, energy, mouth_in_food, food_mass):
        if food_mass > 0 and mouth_in_food and energy < self.satiety:
            return 'feed'
        if energy < self.forage_threshold and food_mass > 0:
            return 'forage'
        return 'rest'

    def table(self):
        rows = []
        for energy in (0., .25, .5, .75, .99):
            for mouth in (False, True):
                for mass in (0., 1.):
                    rows.append(dict(energy=energy, mouth_in_food=mouth, food_mass=mass,
                                     decision=self.decide(energy, mouth, mass)))
        return rows


class PlatformStep:
    """One platform tick: energy, food, gate, selector, with explicit provenance."""

    def __init__(self, energy=None, food=None, selector=None):
        self.energy = energy or EnergyState()
        self.food = food or FoodPatch()
        self.selector = selector or BehaviourSelector()

    def step(self, dt_s, mouth_point):
        mouth_in_food = self.food.contains(mouth_point) and not self.food.depleted
        decision = self.selector.decide(self.energy.energy, mouth_in_food, self.food.mass)
        consumed = self.food.consume(dt_s) if decision == 'feed' else 0.
        self.energy.update(dt_s, feeding=(decision == 'feed' and consumed > 0))
        return dict(decision=decision, mouth_in_food=mouth_in_food, consumed=consumed,
                    energy=self.energy.energy, food_mass=self.food.mass,
                    provenance=dict(movement_command='connectome_module_or_declared_controller',
                                    behaviour_decision='engineering_selector_not_neural',
                                    feeding='declared_gate_virtual_food'))
