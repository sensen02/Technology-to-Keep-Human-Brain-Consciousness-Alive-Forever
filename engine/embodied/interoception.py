"""Artificial fly sensory proxies. No connector to a real neck or electrode."""
from dataclasses import dataclass
import math
from numbers import Real


LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
SCHEMA = "artificial_fly_interoception_v1"


def _number(value, name, lower=-math.inf, upper=math.inf):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not lower <= value <= upper):
        raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")
    return float(value)


@dataclass(frozen=True)
class SensorySample:
    name: str
    value: float | None
    route: str


@dataclass(frozen=True)
class SensoryFrame:
    time_s: float
    samples: tuple[SensorySample, ...]

    def local_inputs(self, side: str) -> dict[str, float | None]:
        """Only engineering local inputs. None marks missing observations."""
        if side not in ("body_local", "head_local"):
            raise ValueError("expected body_local or head_local")
        return {sample.name: sample.value for sample in self.samples if sample.route == side}


@dataclass(frozen=True)
class JointCalibration:
    lower_rad: float
    upper_rad: float
    flexion_sign: int

    def __post_init__(self):
        lower = _number(self.lower_rad, "lower_rad")
        upper = _number(self.upper_rad, "upper_rad")
        if lower >= upper or type(self.flexion_sign) is not int or self.flexion_sign not in (-1, 1):
            raise ValueError("joint range must increase and flexion_sign must be -1 or +1")


@dataclass(frozen=True)
class JointSample:
    angle_rad: float
    velocity_rad_s: float


class MujocoTibiaSensors:
    """Read six named scalar hinges from either live data or saved qpos/qvel."""

    def __init__(self, model, flexion_signs):
        import mujoco

        if set(flexion_signs) != set(LEGS):
            raise ValueError("supply a geometry-verified flexion sign for each leg")
        self._dimensions = (model.nq, model.nv)
        self._addresses = {}
        self.calibration = {}
        for leg in LEGS:
            joint_name = f"joint_{leg}Tibia_pitch"
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if (joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
                    or not model.jnt_limited[joint]):
                raise ValueError(f"missing limited hinge {joint_name}")
            self._addresses[leg] = (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))
            self.calibration[leg] = JointCalibration(*model.jnt_range[joint], flexion_signs[leg])

    def read(self, qpos, qvel):
        if (len(qpos), len(qvel)) != self._dimensions:
            raise ValueError("qpos/qvel do not match compiled body")
        return {leg: JointSample(qpos[qa], qvel[va]) for leg, (qa, va) in self._addresses.items()}


@dataclass(frozen=True)
class HeartObservation:
    """Measured/simulated normalized diameter and its signed rate; provider calibrated."""

    diameter_fraction: float
    diameter_rate_per_s: float | None = None


@dataclass(frozen=True)
class GutState:
    pharynx: float = 0.
    crop: float = 0.
    midgut: float = 0.
    hindgut: float = 0.


@dataclass(frozen=True)
class GutStep:
    state: GutState
    admitted: float
    swallowed: float
    crop_to_midgut: float
    midgut_to_hindgut: float
    exited_hindgut: float


@dataclass(frozen=True)
class DigestiveCompartments:
    """Volume ledger using a common arbitrary unit, with prescribed inputs and rates.

    Each transfer draws on the contents at tick start. This imposes one tick of
    transit per compartment. It models neither digestion nor neural pump control.
    """

    capacity: GutState = GutState(pharynx=.2, crop=1., midgut=1., hindgut=.6)
    swallow_rate: float = 2.
    crop_rate: float = .3
    midgut_rate: float = .2
    hindgut_rate: float = .1

    def __post_init__(self):
        for name in ("pharynx", "crop", "midgut", "hindgut"):
            if _number(getattr(self.capacity, name), f"{name} capacity", 0.) == 0:
                raise ValueError("capacities must be positive")
        for name in ("swallow_rate", "crop_rate", "midgut_rate", "hindgut_rate"):
            _number(getattr(self, name), name, 0.)

    def fractions(self, state: GutState) -> GutState:
        return GutState(*(
            _number(getattr(state, name), name, 0., getattr(self.capacity, name))
            / getattr(self.capacity, name)
            for name in ("pharynx", "crop", "midgut", "hindgut")
        ))

    def advance(self, state: GutState, dt_s: float, *, inlet_rate=0., swallow=0.) -> GutStep:
        self.fractions(state)
        dt = _number(dt_s, "dt_s", 0.)
        if dt == 0:
            raise ValueError("dt_s must be positive")
        inlet = _number(inlet_rate, "inlet_rate", 0.)
        pump = _number(swallow, "swallow", 0., 1.)
        cap = self.capacity
        a = min(dt * inlet, cap.pharynx - state.pharynx)
        b = min(dt * (pump * self.swallow_rate), state.pharynx, cap.crop - state.crop)
        c = min(dt * self.crop_rate, state.crop, cap.midgut - state.midgut)
        d = min(dt * self.midgut_rate, state.midgut, cap.hindgut - state.hindgut)
        e = min(dt * self.hindgut_rate, state.hindgut)
        after = GutState(
            min(cap.pharynx, (state.pharynx - b) + a),
            min(cap.crop, (state.crop - c) + b),
            min(cap.midgut, (state.midgut - d) + c),
            min(cap.hindgut, (state.hindgut - e) + d),
        )
        return GutStep(after, a, b, c, d, e)


class FlyInteroception:
    """Encode sensed states without applying action, neural currents or global inputs."""

    def __init__(self, calibration, *, joint_rate_rad_s=10., heart_rate_scale_per_s=4.,
                 digestion=DigestiveCompartments()):
        if set(calibration) != set(LEGS) or any(
                not isinstance(item, JointCalibration) for item in calibration.values()):
            raise ValueError("supply a JointCalibration for all six legs")
        self._calibration = dict(calibration)
        self._joint_rate = _number(joint_rate_rad_s, "joint_rate_rad_s", 0.)
        self._heart_rate = _number(heart_rate_scale_per_s, "heart_rate_scale_per_s", 0.)
        if not self._joint_rate or not self._heart_rate:
            raise ValueError("rate scales must be positive")
        self.digestion = digestion

    def observe(self, time_s, joints, *, heart=None, gut=None) -> SensoryFrame:
        time = _number(time_s, "time_s", 0.)
        if set(joints) - set(LEGS):
            raise ValueError("unknown leg in joint observations")
        samples = []
        for leg in LEGS:
            obs = joints.get(leg)
            flexion = extension = flexing = extending = None
            if obs is not None:
                cal = self._calibration[leg]
                q = _number(obs.angle_rad, f"{leg} angle")
                q = min(cal.upper_rad, max(cal.lower_rad, q))
                bend = (q - cal.lower_rad) / (cal.upper_rad - cal.lower_rad)
                flexion = bend if cal.flexion_sign == 1 else 1. - bend
                extension = 1. - flexion
                rate = _number(obs.velocity_rad_s, f"{leg} velocity") * cal.flexion_sign
                flexing = min(1., max(0., rate / self._joint_rate))
                extending = min(1., max(0., -rate / self._joint_rate))
            samples.extend(SensorySample(f"{leg}/{key}", value, "body_local") for key, value in (
                ("claw_flexion", flexion), ("claw_extension", extension),
                ("hook_flexing", flexing), ("hook_extending", extending),
            ))

        diameter = expanding = contracting = None
        if heart is not None:
            diameter = _number(heart.diameter_fraction, "heart diameter fraction", 0., 1.)
            if heart.diameter_rate_per_s is not None:
                rate = _number(heart.diameter_rate_per_s, "heart diameter rate")
                expanding = min(1., max(0., rate / self._heart_rate))
                contracting = min(1., max(0., -rate / self._heart_rate))
        samples.extend((
            SensorySample("heart/diameter", diameter, "diagnostic_only"),
            SensorySample("heart/expanding", expanding, "diagnostic_only"),
            SensorySample("heart/contracting", contracting, "diagnostic_only"),
        ))

        fractions = self.digestion.fractions(gut) if gut is not None else None
        for name, route in (("pharynx", "head_local"), ("crop", "route_unbound"),
                            ("midgut", "route_unbound"), ("hindgut", "diagnostic_only")):
            value = getattr(fractions, name) if fractions is not None else None
            samples.append(SensorySample(f"{name}/distension", value, route))
        return SensoryFrame(time, tuple(samples))
