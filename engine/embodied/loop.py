"""engine.embodied.loop -- ONE multirate scheduler that owns time, and the episode record.

THE THREE TIERS AND THEIR CLOCKS
--------------------------------
    body physics      dt = 1e-4 s   (MuJoCo, FlyGym/NeuroMechFly v2)
    neural tier       dt = 5e-4 s   (conductance point tier, engine.neural_cond)
    command decode    dt = 5e-3 s   (CPG intrinsic frequency + left/right asymmetry)

The command clock is an integer multiple of the neural clock (10 steps) and the
neural clock is an integer multiple of the body clock (5 steps); the config refuses
anything else, because a scheduler whose clocks do not align cannot state when a
signal exists.

CAUSAL ORDER PER OUTER STEP (enforced, recorded, and tested)
-----------------------------------------------------------
    k = 0, 1, 2, ...

    1. READ      the body state at the START of interval k (t = k * 5 ms)
    2. TRANSDUCE that state through the receptor bank into an external current
    3. ADVANCE   the neural tier by 10 substeps of 0.5 ms with that current held
                 constant; the interval's spikes are returned at interval END
    4. DECODE    the command from the spikes of interval k-1, the interval that
                 COMPLETED at t = k * 5 ms (at k = 0 no interval has completed and
                 the command is exactly neutral, source_interval = -1)
    5. HOLD      that command while the body takes its 50 substeps of 1e-4 s

Every tier therefore reads only data that already existed when its step began.  No
array in ``observed`` is a truth-only quantity: the observed channels are the
transduced receptor drives and the readout firing rates, and the split is tested by
replaying the declared transduction from ``truth`` and by checking the key names.

WHAT THE TWO ARMS ARE
---------------------
``cpg_baseline``    : the scheduler steps the body and nothing else.  The body runs
                      FlyGym's own tripod CPG through ``attach_cpg_baseline()``, so
                      this arm is bit-identical to a plain CPG walk (tested).
``neural_modulated``: the SAME CPG generates the gait; the neural tier's descending
                      firing modulates only (i) the intrinsic frequency and (ii) a
                      left/right asymmetry.  The connectome does NOT generate the
                      gait here and no result may be described that way.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
import os
import resource
import time

import numpy as np

from .adapters import (
    LEGS, HONESTY, UNITS, CommandDecoder, CommandedTripodCPG, DecodeConfig,
    MechanoReceptorBank, NeuralTier, OwnershipError, OwnershipLedger,
    ReceptorConfig, TierConfig, build_neural_tier, leg_rows_from_dof_order,
    model_fingerprint, sensor_matrix_row_map,
)
from .body_backend import BodyBackend, BodyConfig, summarise_walk

__all__ = ["ARMS", "FORCED_RATE_MODES", "LoopConfig", "MultirateScheduler",
           "Episode", "TRUTH_FIELDS", "OBSERVED_FIELDS", "COMMAND_FIELDS",
           "OBSERVED_PROVENANCE", "truth_only_names", "heading_yaw_rad"]

ARMS = ("cpg_baseline", "neural_modulated")
#: a command change is logged as an event only when it moves by more than this
COMMAND_EVENT_EPSILON = 0.01
FORCED_RATE_MODES = ("none", "reference", "high", "low", "turn_left", "turn_right")

#: truth = what the physical world is.  A later training step may not see any of it
#: through the ``observed`` namespace; the names are listed here so that a test can
#: check the two namespaces against each other mechanically.
TRUTH_FIELDS = ("time_s", "thorax_mm", "body_positions_mm", "body_rotations_wxyz",
                "joint_angles_rad", "joint_velocities_rad_s", "contact_present",
                "contact_found_raw", "contact_forces", "contact_positions_mm",
                "actuator_forces", "cpg_phase_rad", "cpg_magnitude",
                "cpg_intrinsic_freqs_hz")
OBSERVED_FIELDS = ("time_s", "force_strain_per_leg", "angle_strain_per_leg",
                   "receptor_drive_contact", "receptor_drive_proprio",
                   "current_nA_per_leg", "descending_rate_hz",
                   "descending_rate_left_hz", "descending_rate_right_hz")
COMMAND_FIELDS = ("time_s", "source_interval", "speed_scale", "turn",
                  "rate_total_hz", "rate_left_hz", "rate_right_hz",
                  "delta_total_hz", "delta_lr_hz", "forced")

#: declared information path for every observed channel: (truth sources, transform).
#: The test recomputes each channel from the listed truth fields and requires an exact
#: match, which is what makes "observed contains only transduced quantities" checkable
#: rather than asserted.
OBSERVED_PROVENANCE = {
    "force_strain_per_leg": (("contact_forces",),
                             "clip(|F_leg| / reference_force, 0, 1); max over the 3 "
                             "contact-frame components, ILLUSTRATIVE reference"),
    "angle_strain_per_leg": (("joint_angles_rad",),
                             "mean_j |q_j - q_j(spawn)| / reference_angle_rad, clipped "
                             "to [0, 1], over that leg's 7 actuated DOFs"),
    "receptor_drive_contact": (("contact_forces",),
                               "MechanoReceptor state * gain, first-order approach with "
                               "ILLUSTRATIVE tau_ms"),
    "receptor_drive_proprio": (("joint_angles_rad",),
                               "MechanoReceptor state * gain, first-order approach with "
                               "ILLUSTRATIVE tau_ms"),
    "current_nA_per_leg": (("contact_forces", "joint_angles_rad"),
                           "current_nA_per_drive * (w_c * drive_contact + w_p * "
                           "drive_proprio), ILLUSTRATIVE conversion"),
    "descending_rate_hz": ((), "firing rate of the tier's descending readout"),
    "descending_rate_left_hz": ((), "firing rate of the readout's left-biased cells"),
    "descending_rate_right_hz": ((), "firing rate of the readout's right-biased cells"),
    "time_s": ((),
               "the scheduler's own interval timestamp, not a body measurement"),
}


def truth_only_names():
    return tuple(TRUTH_FIELDS)


def heading_yaw_rad(quat_wxyz):
    """Yaw of the thorax's body x-axis from a unit quaternion (w, x, y, z)."""
    q = np.asarray(quat_wxyz, float)
    if q.shape != (4,):
        raise ValueError("a quaternion has four components")
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate quaternion")
    w, x, y, z = q / n
    vx = 1.0 - 2.0 * (y * y + z * z)
    vy = 2.0 * (x * y + w * z)
    return float(math.atan2(vy, vx))


def config_hash(cfg: "LoopConfig"):
    """sha256[:32] of the loop config, so an episode names the exact configuration."""
    payload = json.dumps(cfg.as_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _peak_ram_mb():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@dataclass(frozen=True)
class LoopConfig:
    """Everything that fixes one episode.  Validated, and refused if inconsistent."""

    arm: str = "cpg_baseline"
    seed: int = 0
    duration_s: float = 8.0
    #: THE CLOCKS.  dt_command must be an integer multiple of dt_neural, which must be
    #: an integer multiple of dt_body; duration must be a whole number of commands.
    dt_body_s: float = 1e-4
    dt_neural_s: float = 5e-4
    dt_command_s: float = 5e-3
    #: calibration = a fixed 500 ms run with the spawn sensation held constant, of
    #: which the FIRST 100 ms is discarded as the tier's startup transient and the
    #: remaining 400 ms sets the reference rate (the module's own phase/settle
    #: convention).
    calibration_ms: float = 500.0
    calibration_settle_ms: float = 100.0
    forced_rate_mode: str = "none"
    forced_rate_delta_hz: float = 15.0
    spawn_position_mm: tuple = (0.0, 0.0, 0.5)
    world_half_size_mm: float = 1000.0
    cpg_intrinsic_frequency_hz: float = 12.0
    add_tracking_camera: bool = True
    tracking_camera_name: str = "trackcam"
    gl_backend: str | None = None
    tier: TierConfig = field(default_factory=TierConfig)
    receptor: ReceptorConfig = field(default_factory=ReceptorConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

    # -- clock arithmetic ---------------------------------------------------
    def _ratio(self, big, small, name):
        r = big / small
        if not math.isfinite(r) or r < 1.0 or not math.isclose(r, round(r), rel_tol=0,
                                                               abs_tol=1e-9):
            raise ValueError("%s must be an integer multiple of the finer clock" % name)
        return int(round(r))

    def validate(self):
        if self.arm not in ARMS:
            raise ValueError("arm must be one of %r" % (ARMS,))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for nm in ("duration_s", "dt_body_s", "dt_neural_s", "dt_command_s",
                   "calibration_ms", "forced_rate_delta_hz",
                   "cpg_intrinsic_frequency_hz", "world_half_size_mm"):
            v = getattr(self, nm)
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(float(v)):
                raise ValueError("%s must be a finite number" % nm)
        if min(self.dt_body_s, self.dt_neural_s, self.dt_command_s) <= 0:
            raise ValueError("all three clocks must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")
        if self.calibration_settle_ms < 0:
            raise ValueError("calibration_settle_ms must be nonnegative")
        self.neural_substeps
        self.body_substeps
        self.n_intervals
        self.calibration_intervals
        self.calibration_settle_steps
        if self.forced_rate_mode not in FORCED_RATE_MODES:
            raise ValueError("forced_rate_mode must be one of %r" % (FORCED_RATE_MODES,))
        if self.arm == "cpg_baseline" and self.forced_rate_mode != "none":
            raise ValueError("a forced descending rate is meaningless in the baseline "
                             "arm, which has no neural tier")
        self.tier.validate()
        self.receptor.validate()
        self.decode.validate()
        if len(self.spawn_position_mm) != 3:
            raise ValueError("spawn_position_mm must be three numbers")
        return self

    @property
    def neural_substeps(self):
        return self._ratio(self.dt_command_s, self.dt_neural_s, "dt_command_s")

    @property
    def body_substeps(self):
        return self._ratio(self.dt_command_s, self.dt_body_s, "dt_command_s/body")

    @property
    def n_intervals(self):
        r = self.duration_s / self.dt_command_s
        if not math.isfinite(r) or not math.isclose(r, round(r), rel_tol=0, abs_tol=1e-9):
            raise ValueError("duration_s must be a whole number of command intervals")
        n = int(round(r))
        if n < 2:
            raise ValueError("the episode needs at least two command intervals")
        return n

    @property
    def calibration_settle_steps(self):
        r = self.calibration_settle_ms / self.dt_neural_s / 1000.0
        if not math.isfinite(r) or r < 0 or not math.isclose(r, round(r), rel_tol=0,
                                                             abs_tol=1e-9):
            raise ValueError("calibration_settle_ms must be a whole number of neural "
                             "steps")
        n = int(round(r))
        if n >= self.calibration_intervals * self.neural_substeps:
            raise ValueError("the settled calibration window is empty")
        return n

    @property
    def calibration_intervals(self):
        r = self.calibration_ms / (self.dt_command_s * 1000.0)
        if not math.isfinite(r) or not math.isclose(r, round(r), rel_tol=0, abs_tol=1e-9):
            raise ValueError("calibration_ms must be a whole number of command intervals")
        n = int(round(r))
        if n < 1:
            raise ValueError("calibration_ms must cover at least one interval")
        return n

    def as_dict(self):
        d = asdict(self)
        d["tier"] = asdict(self.tier)
        d["receptor"] = asdict(self.receptor)
        d["decode"] = asdict(self.decode)
        d["neural_substeps_per_command"] = self.neural_substeps
        d["body_substeps_per_command"] = self.body_substeps
        d["n_intervals"] = self.n_intervals
        d["calibration_intervals"] = self.calibration_intervals
        return d


class Episode:
    """One recorded episode with SEPARATED truth / observed / commands / events.

    The separation is the point: a later training step is allowed to read
    ``observed`` and ``commands`` and must never see ``truth``.  The namespaces are
    separate dicts AND separate npz key prefixes, and the test suite checks the split
    mechanically (names must be disjoint from truth-only names, and every observed
    channel must be exactly reproducible from ``truth`` through its declared
    transformation, so no hidden channel exists).
    """

    def __init__(self, arm, seed, cfg: LoopConfig):
        self.arm = arm
        self.seed = seed
        self.cfg = cfg
        self.truth = {}
        self.observed = {}
        self.commands = {}
        self.events = []
        self.meta = {}
        self.metrics = {}
        #: the heading series is kept OUT of ``metrics``: metrics are JSON numbers, and
        #: a raw array in there would be serialised as a megabyte string.
        self.yaw_series_rad = None

    # -- recording ----------------------------------------------------------
    def add_event(self, kind, time_s, **payload):
        ev = {"kind": str(kind), "time_s": float(time_s), "arm": self.arm,
              "seed": int(self.seed)}
        ev.update({k: (v if isinstance(v, (int, float, str, bool, type(None)))
                       else str(v)) for k, v in payload.items()})
        self.events.append(ev)
        return ev

    def finalise(self, wall_seconds, peak_ram_mb):
        self.meta.update({
            "arm": self.arm, "seed": int(self.seed),
            "config": self.cfg.as_dict(), "units": dict(UNITS),
            "honesty": list(HONESTY),
            "wall_seconds": float(wall_seconds),
            "peak_ram_mb": float(peak_ram_mb),
            "truth_fields": list(TRUTH_FIELDS),
            "observed_fields": list(OBSERVED_FIELDS),
            "command_fields": list(COMMAND_FIELDS),
            "observed_provenance": {k: {"truth_sources": list(v[0]), "transform": v[1]}
                                    for k, v in OBSERVED_PROVENANCE.items()},
            "truth_observed_split": (
                "truth = the physical world (body pose, joint angles, contact forces, "
                "CPG state); observed = ONLY what the neural tier is allowed to see "
                "(transduced receptor drives and its own readout rate); commands = the "
                "decoded CPG knobs with the interval they came from. A later training "
                "step must read observed/commands only."),
        })
        return self

    def arrays(self):
        out = {}
        for ns, d in (("truth", self.truth), ("observed", self.observed),
                      ("commands", self.commands)):
            for k, v in d.items():
                out["%s/%s" % (ns, k)] = np.asarray(v)
        out["events/json"] = np.array(json.dumps(self.events, sort_keys=True))
        out["meta/json"] = np.array(json.dumps(self.meta, sort_keys=True, default=str))
        return out

    def save_npz(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path

    def summary(self):
        return {"arm": self.arm, "seed": self.seed, "n_events": len(self.events),
                "n_intervals": len(self.commands.get("speed_scale", [])),
                "metrics": self.metrics}


class MultirateScheduler:
    """Owns the body, the neural tier and the clock.  Nothing else steps anything."""

    def __init__(self, cfg: LoopConfig, built_tier=None):
        self.cfg = cfg.validate()
        self.ledger = OwnershipLedger()
        self.built = None
        self.body = BodyBackend(BodyConfig(
            seed=self.cfg.seed, timestep_s=self.cfg.dt_body_s,
            spawn_position_mm=tuple(float(v) for v in self.cfg.spawn_position_mm),
            world_half_size_mm=self.cfg.world_half_size_mm,
            cpg_intrinsic_frequency_hz=self.cfg.cpg_intrinsic_frequency_hz,
            add_tracking_camera=self.cfg.add_tracking_camera,
            tracking_camera_name=self.cfg.tracking_camera_name,
        ), gl_backend=self.cfg.gl_backend)
        self.dof_order = self.body.dof_order
        # the angle map is resolved against the COMPILED model by joint name: the 42
        # actuated DOFs are NOT the first 42 joint angles (the passive tarsal joints sit
        # in between), which an earlier version of this file got wrong.
        self.body_maps = leg_rows_from_dof_order(self.dof_order, self.body.model)
        self.joint_rows = self.body_maps["dof_rows"]
        self.angle_rows = self.body_maps["angle_rows"]

        self.cpg = None
        self.tier = None
        self.receptors = None
        self.decoder = None
        self.sensor_map_audit = None
        self._prev_spikes = None
        self._current_command = None
        self._renderer = None
        #: (interval, speed_scale, turn) for every command actually WRITTEN into the CPG,
        #: in write order.  It makes "the command held during interval k is the command
        #: recorded for interval k" checkable instead of asserted.
        self.command_log = []

        if self.cfg.arm == "cpg_baseline":
            self.body.attach_cpg_baseline()
        else:
            self.cpg = CommandedTripodCPG(
                timestep_s=self.cfg.dt_body_s,
                base_frequency_hz=self.cfg.cpg_intrinsic_frequency_hz,
                seed=self.cfg.seed, dof_order=self.dof_order, ledger=self.ledger)
            self.body.attach_action_source(self.cpg)
            self.built = built_tier or build_neural_tier(self.cfg.tier)
            self.tier = NeuralTier(self.built, self.cfg.tier, self.ledger)
            self.decoder = CommandDecoder(
                self.cfg.decode,
                dt_ms_command=self.cfg.dt_command_s * 1000.0,
                dt_ms_neural=self.cfg.dt_neural_s * 1000.0,
                ledger=self.ledger, tier=self.tier)
            self.receptors = MechanoReceptorBank(
                self.built["sensory"], self.angle_rows, self.tier.n,
                self.cfg.receptor, dt_ms=self.cfg.dt_command_s * 1000.0,
                ledger=self.ledger, seed=self.cfg.seed)
            self.sensor_map_audit = sensor_matrix_row_map(self.built["sensory"],
                                                          self.angle_rows)
        self.calibration = None

    # ------------------------------------------------------------------ setup
    def attach_renderer(self, renderer):
        self._renderer = renderer
        return self

    def make_renderer(self, camera_res=(240, 320), output_fps=25):
        spec = self.body.camera_spec
        if spec is None:
            raise RuntimeError("this body has no tracking camera to render from")
        self._renderer = self.body.sim.set_renderer(
            spec, camera_res=camera_res, playback_speed=1.0, output_fps=output_fps,
            buffer_frames=True)
        return self._renderer

    def renderer_frames(self):
        if self._renderer is None:
            return []
        return [np.asarray(f).copy()
                for f in self._renderer.frames.get(self.body.camera_spec, [])]

    # ------------------------------------------------------------ calibration
    def calibrate(self, episode: Episode):
        """Pre-registered reference rates: the tier driven by the SPAWN sensation.

        Fixed procedure, identical for every seed and every episode: take the body
        state before any physics step, transduce it once per calibration interval with
        the receptor bank held at its own dynamics, advance the tier for
        ``calibration_ms``, and take the mean rates of the whole window.  The
        reference is then the zero point of the command, and the filter is
        initialised AT it, so "force the rate to its reference" gives an exactly
        neutral command.
        """
        obs = self.body.observe()                      # spawn state, no step taken
        self.receptors.set_reference_pose(obs.joint_angles_rad)
        n_cal = self.cfg.calibration_intervals
        spikes = np.zeros((n_cal * self.cfg.neural_substeps, self.tier.n), dtype=bool)
        for j in range(n_cal):
            cur, _ = self.receptors.transduce(obs)
            spikes[j * self.cfg.neural_substeps:(j + 1) * self.cfg.neural_substeps] = \
                self.tier.advance(self.cfg.neural_substeps, cur)
        settle_steps = int(round(self.cfg.calibration_settle_ms
                                 / self.cfg.dt_neural_s / 1000.0))
        cal = self.decoder.calibrate(self.tier, spikes, n_cal,
                                     settled_from_step=settle_steps)
        self.calibration = cal
        self.tier.reset()
        self.receptors.reset()
        self.receptors.set_reference_pose(obs.joint_angles_rad)
        episode.meta["calibration"] = cal
        episode.add_event("calibration_completed", 0.0,
                          reference_total_hz=cal["reference_rates_hz"]["total_hz"],
                          reference_left_hz=cal["reference_rates_hz"]["left_hz"],
                          reference_right_hz=cal["reference_rates_hz"]["right_hz"],
                          calibration_ms=self.cfg.calibration_ms)
        return cal

    def _forced_rate(self):
        mode = self.cfg.forced_rate_mode
        if mode == "none":
            return None
        r0 = self.decoder.rate0
        d = float(self.cfg.forced_rate_delta_hz)
        if mode == "reference":
            return dict(r0)
        if mode == "high":
            return {"total_hz": r0["total_hz"] + d, "left_hz": r0["left_hz"],
                    "right_hz": r0["right_hz"]}
        if mode == "low":
            return {"total_hz": max(0.0, r0["total_hz"] - d), "left_hz": r0["left_hz"],
                    "right_hz": r0["right_hz"]}
        if mode == "turn_left":
            return {"total_hz": r0["total_hz"], "left_hz": r0["left_hz"] + d,
                    "right_hz": max(0.0, r0["right_hz"] - d)}
        if mode == "turn_right":
            return {"total_hz": r0["total_hz"], "left_hz": max(0.0, r0["left_hz"] - d),
                    "right_hz": r0["right_hz"] + d}
        raise ValueError("unknown forced_rate_mode %r" % (mode,))

    # ------------------------------------------------------------------- run
    def run(self, want_frames=False):
        cfg = self.cfg
        t_start = time.perf_counter()
        ep = Episode(cfg.arm, cfg.seed, cfg)
        ep.meta["arm_description"] = (
            "FlyGym tripod CPG, unchanged: ENGINEERING BASELINE, not the connectome"
            if cfg.arm == "cpg_baseline" else
            "FlyGym tripod CPG still generates the gait; the neural tier modulates ONLY "
            "the intrinsic frequency and a left/right asymmetry. The connectome does "
            "NOT generate the gait.")
        if self.tier is None:
            ep.meta["observed_note"] = (
                "the cpg_baseline arm has NO neural tier: it observes nothing and "
                "decodes no command, and FlyGym's own CPG is never written to. The "
                "observed/ and commands/ namespaces are therefore empty by design, "
                "not by omission.")
        if self.tier is not None:
            ep.meta["tier_coverage"] = self.built["coverage"]
            ep.meta["tier_fingerprint_sha256_32"] = self.built["fingerprint"]
            ep.meta["sensor_matrix_row_map"] = self.sensor_map_audit
            ep.meta["leg_angle_rows"] = {k: [int(v) for v in arr]
                                         for k, arr in self.angle_rows.items()}
            ep.meta["leg_dof_rows"] = {k: [int(v) for v in arr]
                                       for k, arr in self.joint_rows.items()}
            ep.meta["joint_angle_index_note"] = self.body_maps["note"]
        ep.meta["fingerprint_sha256_32"] = model_fingerprint(
            self.built["fingerprint"] if self.built else "no-tier",
            cfg.tier, cfg.receptor, cfg.decode, cfg.as_dict())
        ep.meta["config_hash_sha256_32"] = config_hash(cfg)
        ep.meta["seeds"] = {
            "body_and_cpg_seed": int(cfg.seed),
            "receptor_seed": int(cfg.seed),
            "tier_background_seed": int(cfg.tier.background_seed),
            "gl_backend": cfg.gl_backend,
            "note": ("one seed drives the body's CPG initial phases, the receptor instances "
                     "and (through the config) the tier's fixed per-neuron background; the "
                     "two arms of a pair share it, so only the command source differs"),
        }
        ep.add_event("episode_start", 0.0, duration_s=cfg.duration_s,
                     n_intervals=cfg.n_intervals, config_hash=config_hash(cfg))

        self.ledger = OwnershipLedger()
        if self.cpg is not None:
            self.cpg.ledger = self.ledger
        if self.tier is not None:
            self.tier.ledger = self.ledger
        if self.receptors is not None:
            self.receptors.ledger = self.ledger
        if self.decoder is not None:
            self.decoder.ledger = self.ledger

        if self.tier is not None:
            self.tier.reset()
            self.receptors.reset()
            self.calibrate(ep)
        forced = self._forced_rate()
        if forced is not None:
            ep.add_event("forced_rate_enabled", 0.0, mode=cfg.forced_rate_mode,
                         **{k: float(v) for k, v in forced.items()})

        n_sub_n, n_sub_b = cfg.neural_substeps, cfg.body_substeps
        truth, observed, commands = ep.truth, ep.observed, ep.commands
        for name in TRUTH_FIELDS:
            truth[name] = []
        if self.tier is not None:
            for name in OBSERVED_FIELDS:
                observed[name] = []
        for name in COMMAND_FIELDS:
            commands[name] = []

        self._prev_spikes = None
        prev_speed, prev_turn = 1.0, 0.0
        first_loop_ms = None
        for k in range(cfg.n_intervals):
            t_interval = k * cfg.dt_command_s
            obs = self.body.observe()                      # 1. READ
            if first_loop_ms is None:
                first_loop_ms = 1000.0 * (time.perf_counter() - t_start)
            # ---- truth (recorded, and never given to the observed namespace)
            cpg_state = (self.cpg.state() if self.cpg is not None else self._base_cpg_state())
            truth["time_s"].append(float(obs.time_s))
            truth["thorax_mm"].append(np.asarray(obs.thorax_position_mm, float).copy())
            truth["body_positions_mm"].append(np.asarray(obs.body_positions_mm, float).copy())
            truth["body_rotations_wxyz"].append(
                np.asarray(obs.body_rotations_wxyz, float).copy())
            truth["joint_angles_rad"].append(np.asarray(obs.joint_angles_rad, float).copy())
            truth["joint_velocities_rad_s"].append(
                np.asarray(obs.joint_velocities_rad_s, float).copy())
            truth["contact_present"].append(np.asarray(obs.contact_present, bool).copy())
            truth["contact_found_raw"].append(
                np.asarray(obs.contact_found_raw, float).copy())
            truth["contact_forces"].append(np.asarray(obs.contact_forces, float).copy())
            truth["contact_positions_mm"].append(
                np.asarray(obs.contact_positions_mm, float).copy())
            truth["actuator_forces"].append(np.asarray(obs.actuator_forces, float).copy())
            truth["cpg_phase_rad"].append(cpg_state["phase_rad"])
            truth["cpg_magnitude"].append(cpg_state["magnitude"])
            truth["cpg_intrinsic_freqs_hz"].append(cpg_state["intrinsic_freqs_hz"])

            if self.tier is None:
                # ---- baseline arm: no neural tier, no decode, no command write
                for _ in range(n_sub_b):
                    self.body.step()
                    self.ledger.write("body_step", "scheduler")
                    if self._renderer is not None:
                        self.body.sim.render_as_needed()
                continue

            # ---- 2. TRANSDUCE
            current, drive = self.receptors.transduce(obs)
            # ---- 3. ADVANCE the neural tier over interval k (spikes at interval END)
            spikes = self.tier.advance(n_sub_n, current)
            # ---- 4. DECODE the command from the interval that COMPLETED (k-1)
            prev = self._prev_spikes
            if prev is None:
                cmd = self.decoder.neutral()
            else:
                cmd = self.decoder.decode(prev, source_interval=k - 1,
                                          forced_rate=forced)
            self._prev_spikes = spikes
            self._current_command = cmd
            if cmd.source_interval != k - 1:
                raise AssertionError("no-lookahead violated: command for interval %d "
                                     "came from interval %d" % (k, cmd.source_interval))

            observed["time_s"].append(float(t_interval))
            observed["force_strain_per_leg"].append(drive["force_strain"].copy())
            observed["angle_strain_per_leg"].append(drive["angle_strain"].copy())
            observed["receptor_drive_contact"].append(drive["drive_contact"].copy())
            observed["receptor_drive_proprio"].append(drive["drive_proprio"].copy())
            observed["current_nA_per_leg"].append(drive["current_nA_per_leg"].copy())
            observed["descending_rate_hz"].append(float(cmd.rate_total_hz))
            observed["descending_rate_left_hz"].append(float(cmd.rate_left_hz))
            observed["descending_rate_right_hz"].append(float(cmd.rate_right_hz))
            commands["time_s"].append(float(t_interval))
            commands["source_interval"].append(int(cmd.source_interval))
            commands["speed_scale"].append(float(cmd.speed_scale))
            commands["turn"].append(float(cmd.turn))
            commands["rate_total_hz"].append(float(cmd.rate_total_hz))
            commands["rate_left_hz"].append(float(cmd.rate_left_hz))
            commands["rate_right_hz"].append(float(cmd.rate_right_hz))
            commands["delta_total_hz"].append(float(cmd.delta_total_hz))
            commands["delta_lr_hz"].append(float(cmd.delta_left_hz - cmd.delta_right_hz))
            commands["forced"].append(bool(cmd.forced))
            # events are throttled by a pre-registered epsilon: a per-interval log of
            # every last-bit change would be 1600 entries of noise, not an event list.
            if k == 0:
                ep.add_event("first_command_neutral", float(t_interval),
                             speed_scale=float(cmd.speed_scale), turn=float(cmd.turn),
                             source_interval=int(cmd.source_interval))
            elif (abs(cmd.speed_scale - prev_speed) > COMMAND_EVENT_EPSILON
                  or abs(cmd.turn - prev_turn) > COMMAND_EVENT_EPSILON):
                ep.add_event("command_changed", float(t_interval),
                             speed_scale=float(cmd.speed_scale), turn=float(cmd.turn),
                             source_interval=int(cmd.source_interval))
            prev_speed, prev_turn = float(cmd.speed_scale), float(cmd.turn)
            # ---- 5. HOLD the command while the body substeps
            self.cpg.set_command(cmd.speed_scale, cmd.turn)
            self.command_log.append((k, float(cmd.speed_scale), float(cmd.turn)))
            for _ in range(n_sub_b):
                self.body.step()
                self.ledger.write("body_step", "scheduler")
                if self._renderer is not None:
                    self.body.sim.render_as_needed()

        ep.meta["first_interval_wall_ms"] = first_loop_ms
        # MEASURED: the body backend's Simulation.warmup() consumes 0.05 s of model time
        # before the loop starts, so the BODY clock is offset from the scheduler's
        # interval grid by that constant.  The offset is recorded (and asserted constant
        # in the tests) rather than papered over by pretending both clocks start at zero.
        if len(ep.truth["time_s"]):
            ep.meta["body_clock_offset_s"] = float(np.asarray(ep.truth["time_s"])[0])
            ep.meta["clock_note"] = ("truth/time_s is the BODY clock, which starts at "
                                     "body_clock_offset_s because the backend warms up; "
                                     "observed/time_s and commands/time_s are the "
                                     "scheduler's exact interval grid, and the two differ "
                                     "by that constant only")
        ep.meta["ownership_ledger"] = self.ledger.as_dict()
        ep.meta["n_neural_steps"] = (0 if self.tier is None
                                     else int(self.tier.steps))
        self._close_out(ep, t_start, want_frames)
        return ep

    # --------------------------------------------------------------- helpers
    def _base_cpg_state(self):
        net = getattr(getattr(self.body, "_controller", None), "cpg_network", None)
        if net is None:
            return {"phase_rad": np.zeros(6), "magnitude": np.zeros(6),
                    "intrinsic_freqs_hz": np.full(6, float(self.cfg.cpg_intrinsic_frequency_hz))}
        return {"phase_rad": np.asarray(net.curr_phases, float).copy(),
                "magnitude": np.asarray(net.curr_magnitudes, float).copy(),
                "intrinsic_freqs_hz": np.asarray(net.intrinsic_freqs, float).copy()}

    def _close_out(self, ep, t_start, want_frames):
        for ns in (ep.truth, ep.observed, ep.commands):
            for k, v in list(ns.items()):
                ns[k] = np.asarray(v)
        wall = time.perf_counter() - t_start
        ep.metrics = self.measure(ep)
        ep.finalise(wall, _peak_ram_mb())
        if want_frames:
            frames = self.renderer_frames()
            ep.meta["n_rendered_frames"] = len(frames)
        return ep

    def measure(self, ep: Episode):
        """Engineering metrics of one episode, in model units (mm, s, rad)."""
        t = np.asarray(ep.truth["time_s"], float)
        p = np.asarray(ep.truth["thorax_mm"], float)
        c = np.asarray(ep.truth["contact_present"], bool)
        raw = np.asarray(ep.truth["contact_found_raw"], float)
        q = np.asarray(ep.truth["joint_angles_rad"], float)
        ti = int(self.body.thorax_index)
        yaw = np.asarray([heading_yaw_rad(r[ti]) for r in ep.truth["body_rotations_wxyz"]])
        ep.yaw_series_rad = yaw
        ep.meta["thorax_index"] = ti
        body_cfg = BodyConfig(seed=self.cfg.seed, timestep_s=self.cfg.dt_body_s,
                              spawn_position_mm=tuple(self.cfg.spawn_position_mm),
                              world_half_size_mm=self.cfg.world_half_size_mm)
        m = summarise_walk(t, p, c, body_cfg, contact_found_raw=raw)
        d = np.diff(np.unwrap(yaw))
        m.update({
            "heading_start_rad": float(yaw[0]),
            "heading_end_rad": float(yaw[-1]),
            "heading_change_rad": float(yaw[-1] - yaw[0]),
            "heading_change_deg": float(math.degrees(yaw[-1] - yaw[0])),
            "heading_abs_turn_rad": float(np.abs(d).sum()),
            "turn_rate_rad_s": float((yaw[-1] - yaw[0]) / (t[-1] - t[0]))
            if t[-1] > t[0] else float("nan"),
            "displacement_mm": float(np.linalg.norm(p[-1, :2] - p[0, :2])),
            "n_true_joints_in_truth": int(q.shape[1]),
            "contact_duty_per_leg": (c.mean(axis=0).tolist() if c.size else []),
            "gate_note": ("the gates below are the milestone-1 ENGINEERING thresholds "
                          "for this model, declared in code before the run"),
        })
        s = np.asarray(ep.commands.get("speed_scale", []), float)
        if s.size:
            a = np.asarray(ep.commands["turn"], float)
            r = np.asarray(ep.commands["rate_total_hz"], float)
            m.update({
                "command_speed_scale_mean": float(s.mean()),
                "command_speed_scale_min": float(s.min()),
                "command_speed_scale_max": float(s.max()),
                "command_speed_scale_sd": float(s.std()),
                "command_turn_mean": float(a.mean()),
                "command_turn_min": float(a.min()),
                "command_turn_max": float(a.max()),
                "command_turn_sd": float(a.std()),
                "command_turn_abs_mean": float(np.abs(a).mean()),
                "descending_rate_total_mean_hz": float(r.mean()),
                "descending_rate_total_sd_hz": float(r.std()),
                "fraction_intervals_below_neutral_speed": float((s < 1.0).mean()),
                "forced_rate_mode": self.cfg.forced_rate_mode,
                "units": {"speed_scale": "dimensionless", "turn": "dimensionless",
                          "rate": "Hz", "heading": "rad", "displacement": "mm"},
            })
        if self.tier is not None:
            m["descending_readout_neurons"] = int(self.tier.readout_local.size)
            m["descending_readout_left_neurons"] = int(self.tier.left.size)
            m["descending_readout_right_neurons"] = int(self.tier.right.size)
        return m

    def check_ownership(self, n_intervals=None):
        """The ownership invariants, as numbers a test can assert on."""
        n = int(n_intervals if n_intervals is not None
                else self.cfg.n_intervals)
        led = self.ledger
        out = {
            "cpg_intrinsic_freqs_writers": led.writers("cpg_intrinsic_freqs"),
            "cpg_intrinsic_freqs_writes": led.total("cpg_intrinsic_freqs"),
            "body_step_writers": led.writers("body_step"),
            "body_steps": led.total("body_step"),
            "neural_state_writers": led.writers("neural_state"),
            "expected_cpg_writes": (0 if self.cfg.arm == "cpg_baseline" else n),
            "expected_body_steps": n * self.cfg.body_substeps,
            "neural_tier_has_no_body_reference": not any(
                hasattr(self.tier, name) for name in
                ("body", "backend", "cpg", "sim", "set_command", "write_joints",
                 "apply_locomotion_action")) if self.tier is not None else True,
        }
        out["ok"] = bool(
            out["cpg_intrinsic_freqs_writes"] == out["expected_cpg_writes"]
            and out["body_steps"] == out["expected_body_steps"]
            and (not out["cpg_intrinsic_freqs_writers"]
                 or out["cpg_intrinsic_freqs_writers"] == ("command_adapter",))
            and out["body_step_writers"] == ("scheduler",)
            and out["neural_tier_has_no_body_reference"])
        return out

    def close(self):
        self.body.close()
