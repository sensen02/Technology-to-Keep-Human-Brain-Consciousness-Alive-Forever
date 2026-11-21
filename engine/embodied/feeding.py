"""Single-action feeding and body-side interoception for the Eon bench.

Amounts and rates are in engineering units. The body exposes receptor inputs;
the experiment must provide any neural route and its brain-side delivery.
"""
from dataclasses import dataclass
from math import expm1

from .interoception import SensorySample, _number


@dataclass(frozen=True)
class Food:
    volume: float
    nutrient: float

    def __post_init__(self):
        _number(self.volume, 'food volume', 0.)
        _number(self.nutrient, 'food nutrient', 0.)
        if self.volume == 0 and self.nutrient:
            raise ValueError('nutrient requires a nonempty food volume')

    def split(self, amount):
        """Return (remaining, transferred) with a well-mixed concentration."""
        amount = _number(amount, 'transferred volume', 0., self.volume)
        if amount == self.volume:
            return EMPTY, self
        nutrient = self.nutrient * (amount / self.volume)
        return Food(self.volume - amount, self.nutrient - nutrient), Food(amount, nutrient)

    def joined(self, other):
        return Food(self.volume + other.volume, self.nutrient + other.nutrient)


EMPTY = Food(0., 0.)


@dataclass(frozen=True)
class FeedingState:
    patch: Food
    crop: Food = EMPTY
    midgut: Food = EMPTY
    reserve: float = 0.2
    circulating: float = 0.2


@dataclass(frozen=True)
class FeedingStep:
    state: FeedingState
    ingested: Food
    excreted: Food
    absorbed: float
    metabolized: float
    can_ingest: bool
    intake_feedback: SensorySample
    crop_stretch: SensorySample
    circulating_nutrient: SensorySample


@dataclass(frozen=True)
class FeedingBody:
    intake_rate: float = 0.35
    crop_capacity: float = 1.
    midgut_capacity: float = 1.
    transit_rate: float = 0.095
    outflow_rate: float = 0.35
    absorption_rate: float = 0.5
    basal_use: float = 0.018
    circulating_time_constant: float = 2.

    def __post_init__(self):
        for name, value in vars(self).items():
            _number(value, name, 0.)
        if not all((self.intake_rate, self.crop_capacity, self.midgut_capacity,
                    self.circulating_time_constant)):
            raise ValueError('intake rate, capacities and time constant must be positive')

    def advance(self, state, dt_s, *, feed, contact, blocked=False, activity_use=0.):
        if any(type(value) is not bool for value in (feed, contact, blocked)):
            raise ValueError('feed, contact and blocked must be boolean')
        dt = _number(dt_s, 'timestep', 0.)
        if dt == 0:
            raise ValueError('timestep must be positive')
        _number(state.crop.volume, 'crop volume', 0., self.crop_capacity)
        _number(state.midgut.volume, 'midgut volume', 0., self.midgut_capacity)
        reserve = _number(state.reserve, 'reserve', 0.)
        circulating = _number(state.circulating, 'circulating nutrient', 0., 1.)
        extra_use = _number(activity_use, 'activity use rate', 0.)

        available = min(state.patch.volume, self.crop_capacity - state.crop.volume)
        can_ingest = contact and not blocked and available > 0.
        amount = min(dt * self.intake_rate, available) if feed and can_ingest else 0.
        patch, ingested = state.patch.split(amount)
        # Transit reads the old crop. Ingestion cannot credit energy in the same step.
        moved = min(dt * self.transit_rate, state.crop.volume,
                    self.midgut_capacity - state.midgut.volume)
        crop, to_midgut = state.crop.split(moved)
        # Absorb only from the old midgut, then let the unabsorbed fraction exit.
        absorbed = -state.midgut.nutrient * expm1(-self.absorption_rate * dt)
        gut = Food(state.midgut.volume, state.midgut.nutrient - absorbed)
        gut, excreted = gut.split(-gut.volume * expm1(-self.outflow_rate * dt))
        metabolized = min(reserve + absorbed, (self.basal_use + extra_use) * dt)
        reserve = reserve + absorbed - metabolized
        circulating -= (min(reserve, 1.) - circulating) * expm1(-dt / self.circulating_time_constant)
        state = FeedingState(patch, crop.joined(ingested), gut.joined(to_midgut), reserve, circulating)
        return FeedingStep(
            state, ingested, excreted, absorbed, metabolized, can_ingest,
            SensorySample('feeding/intake', min(1., amount / dt / self.intake_rate), 'head_local'),
            SensorySample('crop/stretch', min(1., state.crop.volume / self.crop_capacity), 'body_ascending'),
            SensorySample('nutrient/circulating', circulating, 'hemolymph'),
        )
