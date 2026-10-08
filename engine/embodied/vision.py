"""engine.embodied.vision -- the visual pathway that the embodied loop did not have.

WHY THIS EXISTS
---------------
Every embodied result this project recorded before now was produced by a fly with
NO visual pathway at all:

* the locomotion fly is built by ``flygym_demo.complex_terrain.common.make_locomotion_fly``,
  which never touches ``add_vision``, so the compiled model shipped ``ncam == 0`` and
  ``Simulation.get_raw_vision()`` raised *"Fly '...' does not have any eye cameras
  defined. Make sure to call fly.add_vision()..."*;
* the scene had ``nlight == 0`` and no objects, so even after adding cameras the image
  was lit only by MuJoCo's camera headlight;
* the neural tier therefore contained only 11 visual-machinery neurons out of 79,538,
  and the photoreceptor class in ``engine.receptors`` was deliberately never used,
  because no light was ever computed;
* consequently the "notice a salient object" acceptance test was recorded as NOT RUN.

The defect this module addresses is narrower and MECHANICAL than "the fly cannot see":
a *featureless, static* scene yields a DEGENERATE input -- a constant image has zero
temporal contrast, so no motion signal exists and nothing visually guided can be tested
against it.  Whether a fly's visual system is "normal" cannot be assessed here (there is
no Drosophila visual-response reference dataset in this project).  Whether the input is
degenerate or informative IS measurable, and that is the claim this module makes
falsifiable: see ``non_degeneracy_statistics`` and the pre-registered threshold
``NONDEGENERACY_MIN_TEMPORAL_STD``.

THE PIPELINE (and what each stage is)
-------------------------------------
    eye cameras (FlyGym)          render 512x450 RGB per eye
      -> Retina.raw_image_to_hex_pxls   -> (2, 721, 2) yellow/pale ommatidia readouts
      -> eye_luminance (2, n)     yellow + pale sums, readout units 0..1 of full scale
      -> luminance (2, n)         divisive gain normalisation to the per-eye mean
      -> temporal_contrast (2, n) first-order high-pass, corner at temporal_tau_ms
      -> spatial_contrast (2, n)  deviation from the per-eye mean  <-- EXACTLY 0 for a
                                  spatially uniform image
      -> scalar energies -> visual_drive_nA -> per-channel current for the visual
                                  projection neurons of the neural tier

WHAT THIS IS NOT
----------------
* This is NOT a Drosophila motion-detection model.  There is no Reichardt correlator,
  no lamina monopolar cell, no T4/T5.  ``motion_energy`` is a temporal-variation energy
  proxy, nothing more; it is not a direction-selective response and must never be
  reported as one.
* The ommatidia readouts are APPROXIMATE OPTICS on a body FlyGym's ``Retina`` was not
  calibrated for (see ``HONESTY``); they are not measured Drosophila optics.
* No number here is a measured Drosophila constant.  Thresholds are DERIVED from the
  quantisation geometry of the readout (stated in full) or marked ASSUMED with a sweep
  range in ``ASSUMED_PARAMETERS``.

UNITS
-----
Feature arrays are in READOUT UNITS: 0..1 of the full 8-bit luminance scale, i.e. the
fraction of the maximum pixel value that an ommatidium reports.  They are dimensionless
and are NOT currents.  A current is produced only by ``visual_drive_nA``, which is
required to route the drive through ``engine.receptors.to_nA(drive_mV, nA_per_mV,
provenance)`` -- the project's one explicit unit-conversion helper -- and which REFUSES
to convert at all unless the caller stated the factor explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "VisionConfig", "VisualFeatures", "OmmatidiaFrontEnd",
    "visual_drive_nA", "non_degeneracy_statistics",
    "NONDEGENERACY_MIN_TEMPORAL_STD", "NONDEGENERACY_MIN_SPATIAL_CONTRAST",
    "MIN_SERIES_LENGTH", "THRESHOLD_DERIVATION", "ideal_observer_quantisation_floor",
    "ASSUMED_PARAMETERS", "DRIVE_UNIT_CONTRACT", "DRIVE_SCALE_PROVENANCE",
    "VisionUnavailable", "UnitConversionRefused", "HONESTY", "CAVEATS",
]


# ---------------------------------------------------------------------------
# Pre-registered non-degeneracy criterion
# ---------------------------------------------------------------------------
#
# DERIVATION OF NONDEGENERACY_MIN_TEMPORAL_STD  (fixed BEFORE the experiment ran;
# not tuned to any measured temporal std)
# ------------------------------------------------------------------------------
# The ommatidia readout is produced by ``Retina.raw_image_to_hex_pxls``, which for
# each ommatidium e averages the P_e raw 8-bit pixels that fall inside it and divides
# by 255 (read from flygym/vision/retina.py; P_e = num_pixels_per_ommatidia).  Read
# from the shipped asset flygym/assets/model/neuromechfly/compound_eye.npz:
#
#     n_ommatidia per eye = 721     P_min = 230     P_mean = 236.18
#
# Consequences, in closed form:
#   (a) A perfectly constant scene gives an EXACTLY constant readout: every pixel is
#       the same integer, so the mean is the same float.  Temporal std = 0.0, not
#       "small".  (Measured independently: a static fly in a static scene re-rendered
#       75 times gives median per-ommatidium temporal std == 0.000000; only pixels that
#       are re-sampled differently can move at all.)  So the true floor is 0.
#   (b) The ONLY way a constant image can move is integer rounding inside the fisheye
#       resampling: a one-grey-level change in one source pixel moves an ommatidium by
#         1 / (255 * P_e).
#       With the smallest ommatidium P_min = 230 that is 1/(255*230) = 1.705e-5.
#   (c) Nothing in this pipeline has a resampling kernel wider than a few pixels, so
#       allow an entire 4x4 source footprint to flip by one grey level at once:
#         16 / (255 * 230) = 2.73e-4.
#   (d) The declared threshold is 1.0e-3 = 3.7x the allowance in (c), and equals
#       0.26 of one grey level (of 255) in the readout.  It is therefore STRICTLY above
#       every quantisation artefact a constant image can produce, while remaining two
#       to three orders of magnitude below the luminance modulation any real scene
#       content produces (readout modulations of 1e-3..4e-1 are measured elsewhere for
#       non-static conditions).  A threshold of one grey level (3.9e-3) would reject
#       genuinely faint but real scene motion; one of one quantisation step (1.7e-5)
#       would accept integer-rounding noise as "signal".
#
# The value below is fixed.  It was NOT chosen after seeing the measured statistics,
# and must not be edited to move a verdict.
NONDEGENERACY_MIN_TEMPORAL_STD: float = 1.0e-3
"""Per-ommatidium temporal std of the raw readout that counts as non-degenerate.

Readout units (0..1 of full 8-bit scale).  Derivation: see the block comment above --
3.7x the worst-case quantisation allowance of a constant image, 0.26 grey levels.
"""

NONDEGENERACY_MIN_SPATIAL_CONTRAST: float = 3.9e-4
"""Declared floor for "spatial contrast clearly non-zero".

``spatial_contrast`` is a per-eye mean-removed quantity, so a spatially uniform image
gives EXACTLY 0.0 -- but the two saturation classes of this scene (the uniform white
skybox and the black outside-the-fisheye-disc region) do produce real, large, and
TEMPORALLY CONSTANT spatial structure, which is precisely the degenerate case the
report cares about.  The floor is derived the same way as the temporal one: one full
grey level of mean-removed structure on one ommatidium is 1/255 = 3.92e-3 of the
per-eye mean-removed scale... that is too coarse for a floor, so the declared floor is
set at one mean-removed grey level spread over 10 ommatidia:
    1 / (255 * 10) = 3.9e-4.
It is DERIVED, not measured, and it is used only to separate "clearly non-zero" from
"numerically zero"; the verdict's binding constraint is the temporal threshold."""

MIN_SERIES_LENGTH: int = 8
"""Fewer samples than this and the series is too short to settle anything."""

THRESHOLD_DERIVATION: str = (
    "Temporal threshold 1.0e-3 readout units: the readout is a mean of P_e raw 8-bit "
    "pixels / 255 (P_min=230, P_mean=236.2 for the shipped 721-ommatidia map), so a "
    "constant scene yields an EXACTLY constant readout (floor 0.0, measured); the only "
    "movement a constant image can produce is integer rounding in the fisheye "
    "resampling, 1/(255*P_e) = 1.7e-5 per pixel, or 2.7e-4 for a whole 4x4 footprint "
    "flipping by one grey level.  1.0e-3 is 3.7x that allowance and 0.26 of one grey "
    "level, and was fixed before the experiment ran.")


def ideal_observer_quantisation_floor(
    n_pixels_per_ommatidium: int = 230, n_pixels_perturbed: int = 16, n_levels: int = 255
) -> float:
    """Recompute the quantisation allowance used to derive the threshold.

    Defaults are the measured smallest ommatidium of the shipped map (P_min = 230) and
    a 4x4 source footprint.  Returns the per-ommatidium readout change caused by
    flipping that whole footprint by one grey level: ``n_pixels_perturbed / (n_levels *
    n_pixels_per_ommatidium)`` = 2.73e-4 by default.
    """
    return float(n_pixels_perturbed) / (float(n_levels) * float(n_pixels_per_ommatidium))


# ---------------------------------------------------------------------------
# Assumed (NOT measured) parameters -- every one carries its sweep range
# ---------------------------------------------------------------------------
ASSUMED_PARAMETERS: dict = {
    "temporal_tau_ms": {
        "value": 20.0,
        "assumed": True,
        "sweep_range_ms": (5.0, 50.0),
        "why": ("corner of the first-order high-pass that stands in for the lamina's "
                "temporal filtering.  A [5, 50] ms band is the range over which any "
                "result here should be re-checked.  This project has NO measured "
                "Drosophila lamina temporal filter, so 20 ms is a declared choice, "
                "not a literature value."),
    },
    "drive_scale_nA_per_unit": {
        "value": None,
        "assumed": True,
        "sweep_range_nA_per_unit": (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        "why": ("conversion from readout-unit visual drive to nanoamperes.  It has NO "
                "measured value: the project has no measured photoreceptor-to-"
                "projection-neuron transduction gain, which is exactly why "
                "engine.receptors.to_nA() refuses to default it.  It must be supplied "
                "explicitly, and any current reported under it is an ASSUMED scale."),
    },
    "contrast_gain": {
        "value": 1.0,
        "assumed": True,
        "sweep_range": (0.5, 1.0, 2.0),
        "why": "identity by default; no measured lamina gain exists to justify another value.",
    },
    "motion_gain": {
        "value": 1.0,
        "assumed": True,
        "sweep_range": (0.5, 1.0, 2.0),
        "why": "identity by default; motion_energy is a proxy, not a calibrated response.",
    },
}

DRIVE_UNIT_CONTRACT: str = (
    "features are in READOUT UNITS (0..1 of full 8-bit luminance scale, dimensionless); "
    "the pre-conversion visual drive is the R_m*I-equivalent DRIVE in millivolts that "
    "engine.receptors.DRIVE_UNIT defines and that engine.neural.LIFNetwork consumes; "
    "nanoamperes are produced ONLY by engine.receptors.to_nA(drive_mV, nA_per_mV, "
    "provenance) with a caller-supplied factor and provenance.  No conversion happens "
    "implicitly anywhere in this module."
)

DRIVE_SCALE_PROVENANCE: str = (
    "hand-set VisionConfig.drive_scale_nA_per_unit, supplied by the caller; this project "
    "has NO measured Drosophila photoreceptor-to-projection-neuron transduction gain, so "
    "this factor is ASSUMED, not measured (see ASSUMED_PARAMETERS)"
)


class VisionUnavailable(RuntimeError):
    """Raised when the compiled model carries no eye cameras (add_vision was not called)."""


class UnitConversionRefused(ValueError):
    """Raised when a visual drive would be turned into nanoamperes without an explicit factor."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionConfig:
    """Everything needed to reproduce one visual front end exactly.

    Attributes mirror the names the rest of the project imports.  ``blind`` and
    ``scramble`` are CONTROLS, not model variants:

    blind     force the visual drive to zero at the readout level.  Nothing is rendered
              and every feature array is exactly zero, so the control cannot be
              contaminated by an accidentally-working pathway.
    scramble  permute ommatidia assignments within each eye with a fixed ``seed``.
              The permutation preserves the per-eye total light exactly, so any change
              in a downstream statistic is caused by the spatial ARRANGEMENT, not by
              how much light arrived.
    """

    enabled: bool = True
    #: Eye-camera field of view in degrees.  NOTE: ``NeuroMechFly.add_vision()`` -- the
    #: fly this project actually builds -- takes NO fovy argument and returns None; its
    #: angle is fixed at 157 deg by the shipped asset.  This field is therefore a
    #: REQUEST, checked against the compiled model and reported as a warning when it
    #: could not be honoured.  145.0 is the default of the DIFFERENT class
    #: ``MusculoskeletalFly.add_vision``, kept here so the two paths are comparable.
    fovy: float = 145.0
    temporal_tau_ms: float = 20.0
    contrast_gain: float = 1.0
    motion_gain: float = 1.0
    n_channels: int = 2
    drive_scale_nA_per_unit: "float | None" = None
    unit_contract: str = DRIVE_UNIT_CONTRACT
    seed: int = 0
    blind: bool = False
    scramble: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "VisionConfig":
        if not (0.0 < float(self.fovy) < 180.0):
            raise ValueError("fovy must be a finite angle in (0, 180) degrees")
        if not np.isfinite(self.temporal_tau_ms) or self.temporal_tau_ms <= 0:
            raise ValueError("temporal_tau_ms must be positive and finite")
        if not np.isfinite(self.contrast_gain) or self.contrast_gain < 0:
            raise ValueError("contrast_gain must be finite and non-negative")
        if not np.isfinite(self.motion_gain) or self.motion_gain < 0:
            raise ValueError("motion_gain must be finite and non-negative")
        if int(self.n_channels) not in (1, 2, 4):
            raise ValueError(
                "n_channels must be 1 (pooled), 2 (left, right) or 4 "
                "(left motion, right motion, left spatial, right spatial); "
                f"got {self.n_channels!r}")
        if self.drive_scale_nA_per_unit is not None:
            f = float(self.drive_scale_nA_per_unit)
            if not np.isfinite(f) or f <= 0:
                raise ValueError("drive_scale_nA_per_unit must be None or positive and finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not str(self.unit_contract).strip():
            raise ValueError("unit_contract must be a non-empty string")
        return self

    def as_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled), "fovy_requested_deg": float(self.fovy),
            "temporal_tau_ms": float(self.temporal_tau_ms),
            "contrast_gain": float(self.contrast_gain),
            "motion_gain": float(self.motion_gain),
            "n_channels": int(self.n_channels),
            "drive_scale_nA_per_unit": (None if self.drive_scale_nA_per_unit is None
                                        else float(self.drive_scale_nA_per_unit)),
            "unit_contract": str(self.unit_contract),
            "seed": int(self.seed), "blind": bool(self.blind),
            "scramble": bool(self.scramble),
        }


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass
class VisualFeatures:
    """One visual sample.  All feature arrays are in READOUT UNITS (dimensionless).

    eye_luminance      (2, n)  RAW readout: yellow + pale channel sum per ommatidium.
                               This is the unmodified output of the Retina.
    luminance          (2, n)  gain-normalised: eye_luminance divided by the per-eye mean.
    temporal_contrast  (2, n)  first-order high-pass of ``luminance`` (tau =
                               temporal_tau_ms), scaled by ``contrast_gain``.
    spatial_contrast   (2, n)  ``luminance`` minus the SAME EYE's mean.  This term is
                               EXACTLY zero for a spatially uniform image; it is the
                               term that separates "the fly's eye receives light" from
                               "the fly's eye receives a pattern".
    """

    eye_luminance: np.ndarray
    luminance: np.ndarray
    temporal_contrast: np.ndarray
    spatial_contrast: np.ndarray
    mean_luminance: np.ndarray
    contrast_energy: float
    motion_energy: float
    time_s: float
    n_ommatidia: int
    blind: bool = False
    scrambled: bool = False
    warnings: tuple = ()

    def as_dict(self) -> dict:
        return {
            "time_s": float(self.time_s), "n_ommatidia": int(self.n_ommatidia),
            "blind": bool(self.blind), "scrambled": bool(self.scrambled),
            "contrast_energy": float(self.contrast_energy),
            "motion_energy": float(self.motion_energy),
            "mean_luminance_per_eye": [float(v) for v in np.ravel(self.mean_luminance)],
            "eye_luminance_std_per_eye": [float(v) for v in np.std(self.eye_luminance, axis=1)],
            "spatial_contrast_rms_per_eye": [
                float(np.sqrt(np.mean(np.square(self.spatial_contrast[i])))) for i in range(2)],
            "temporal_contrast_rms_per_eye": [
                float(np.sqrt(np.mean(np.square(self.temporal_contrast[i])))) for i in range(2)],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# The front end
# ---------------------------------------------------------------------------
class OmmatidiaFrontEnd:
    """Render one fly's compound eyes and turn them into features.

    Lifecycle::

        fe = OmmatidiaFrontEnd(VisionConfig(...))
        fe.bind(sim, fly_name)        # fails loudly if add_vision() never ran
        fe.reset()
        for t in times:
            feats = fe.read(sim, fly_name, t)
    """

    #: Both eyes of this model are produced from the SAME shipped ommatidia map, so the
    #: two per-eye counts are equal BY CONSTRUCTION (same asset), not measured twice.
    SAMPLES_BOTH_EYES_FROM_ONE_MAP = True

    def __init__(self, config: VisionConfig | None = None):
        self.config = (config or VisionConfig()).validate()
        self.bound = False
        self.fly_name: "str | None" = None
        self.n_cameras: int = 0
        self.n_ommatidia: int = 0
        self.per_eye_counts: tuple = ()
        self.both_eyes_present: bool = False
        self.n_yellow: int = 0
        self.n_pale: int = 0
        self.fovy_applied_deg: "tuple[float, ...]" = ()
        self.eye_camera_names: tuple = ()
        self.retina = None
        self.bind_report: dict = {}
        self.warnings: list = []
        self.reset()

    # ---------------------------------------------------------------- binding
    def bind(self, sim, fly_name: str) -> None:
        """Check the eye cameras exist and record the readout geometry.

        Raises ``VisionUnavailable`` -- naming the missing capability and the exact call
        that is missing -- if the fly was built without ``add_vision()`` before
        ``world.compile()``.
        """
        mapping = getattr(sim, "_intern_eye_camera_ids_by_fly", None)
        eye_ids = None
        if mapping is not None:
            eye_ids = mapping.get(fly_name) if hasattr(mapping, "get") else None
        if eye_ids is None:
            # Fall back to the PUBLIC check: FlyGym raises a clear ValueError here, and
            # the message names the missing call.  Re-raise it as our own error type so
            # callers can catch one exception class.
            try:
                sim.get_raw_vision(fly_name)
            except ValueError as exc:
                raise VisionUnavailable(
                    f"visual pathway unavailable for fly {fly_name!r}: no eye cameras in "
                    f"the compiled model. FlyGym said: {exc} The fix is to call "
                    "fly.add_vision(draw_sensor_markers=False) BEFORE world.add_fly(...)"
                    " and world.compile(); calling it afterwards fails because the fly's "
                    "parent bodies have already been re-parented into the world."
                ) from exc
            eye_ids = []
        eye_ids = list(eye_ids)
        if len(eye_ids) == 0:
            raise VisionUnavailable(
                f"visual pathway unavailable for fly {fly_name!r}: the compiled model has "
                "no eye cameras for this fly. Call fly.add_vision("
                "draw_sensor_markers=False) BEFORE world.add_fly(...)/world.compile().")

        # Retina: construct it here instead of letting get_raw_vision() do it lazily, so
        # bind() needs no render.  Same class, same shipped asset, assigned back to the
        # simulation so exactly one instance exists.
        retina = getattr(sim, "retina", None)
        if retina is None:
            from flygym.vision.retina import Retina  # noqa: PLC0415  (lazy: keeps this
            # module importable in environments without flygym/mujoco)
            retina = Retina()
            sim.retina = retina
        self.retina = retina

        self.n_cameras = len(eye_ids)
        self.n_ommatidia = int(retina.num_ommatidia_per_eye)
        # The shipped map defines ONE eye; the same map is applied to both cameras.
        self.per_eye_counts = tuple([self.n_ommatidia] * self.n_cameras)
        self.both_eyes_present = bool(self.n_cameras == 2)
        pale = np.asarray(retina.pale_type_mask).astype(int).ravel()
        self.n_pale = int((pale == 1).sum())
        self.n_yellow = int((pale == 0).sum())

        model = getattr(sim, "mj_model", None)
        names, fovys = [], []
        if model is not None:
            for cid in eye_ids:
                names.append(str(model.camera(int(cid)).name))
                fovys.append(float(model.cam_fovy[int(cid)]))
        self.eye_camera_names = tuple(names)
        self.fovy_applied_deg = tuple(fovys)

        self.warnings = []
        if not self.both_eyes_present:
            self.warnings.append(
                f"only {self.n_cameras} eye camera(s) present; expected 2 (left, right)")
        if fovys and any(abs(f - float(self.config.fovy)) > 1e-6 for f in fovys):
            self.warnings.append(
                f"requested fovy {self.config.fovy} deg could NOT be applied: the model's "
                f"eye camera fovy is {fovys} deg. NeuroMechFly.add_vision() takes no fovy "
                "argument (the 145 deg default belongs to MusculoskeletalFly); the angle "
                "is fixed by the shipped vision.yaml asset.")

        self.fly_name = fly_name
        self.bound = True
        self.bind_report = {
            "fly_name": fly_name,
            "model_ncam": (int(model.ncam) if model is not None else None),
            "model_nlight": (int(model.nlight) if model is not None else None),
            "model_ngeom": (int(model.ngeom) if model is not None else None),
            "n_eye_cameras": self.n_cameras,
            "eye_camera_names": list(self.eye_camera_names),
            "both_eyes_present": self.both_eyes_present,
            "n_ommatidia_per_eye": self.n_ommatidia,
            "per_eye_counts": list(self.per_eye_counts),
            "per_eye_counts_note": (
                "equal BY CONSTRUCTION: both cameras are resampled through the same "
                "shipped ommatidia map, so the two counts are not independent measurements"),
            "n_yellow_type": self.n_yellow,
            "n_pale_type": self.n_pale,
            "fovy_requested_deg": float(self.config.fovy),
            "fovy_applied_deg": list(self.fovy_applied_deg),
            "retina_raw_image_px": [int(retina.nrows), int(retina.ncols)],
            "retina_calibration_caveat": HONESTY["retina_calibration"],
            "bind_warnings": list(self.warnings),
            "lamina_of_the_pathway": (
                "each ommatidium carries only ONE of the two spectral channels "
                "(yellow type -> channel 0, pale type -> channel 1; the other is exactly "
                "0).  read() sums the two, which is a broadband intensity, NOT a "
                "colour-opponent signal."),
        }

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear the temporal filter state, the clock and the scramble permutation."""
        n = int(self.n_ommatidia)
        self._hp = np.zeros((2, n)) if n > 0 else np.zeros((2, 0))
        self._t_prev: "float | None" = None
        self._n_reads = 0
        rng = np.random.default_rng(int(self.config.seed))
        self._perm = np.asarray([rng.permutation(n) for _ in range(2)], dtype=np.int64)

    # ------------------------------------------------------------------ read
    def read(self, sim, fly_name: str, time_s: float) -> VisualFeatures:
        """Render (or, if blind, do not render) and return one ``VisualFeatures`` sample."""
        if not self.bound:
            raise RuntimeError("bind(sim, fly_name) must be called before read()")
        n = int(self.n_ommatidia)

        if not self.config.enabled:
            return self._zero_features(time_s, note="visual pathway disabled by config")
        if self.config.blind:
            # A blind fly has no photoreceptor input, so there is nothing to read and the
            # camera is not even queried.  This is the control: the drive is zero BY
            # CONSTRUCTION, not by a small gain.
            return self._zero_features(time_s, note="blind=True: drive forced to zero")

        raw = np.asarray(sim.get_ommatidia_readouts(fly_name), dtype=np.float64)
        if raw.ndim != 3 or raw.shape[1] != n or raw.shape[2] != 2:
            raise RuntimeError(
                f"unexpected ommatidia readout shape {raw.shape}; expected "
                f"(2, {n}, 2) -- FlyGym contract: (n_cameras, n_ommatidia, yellow|pale)")
        if raw.shape[0] != 2:
            raise RuntimeError(f"expected 2 eyes, got {raw.shape[0]}")

        # yellow + pale -> one broadband intensity per ommatidium (one channel is 0)
        eye_lum = raw.sum(axis=2)

        scrambled = bool(self.config.scramble)
        if scrambled:
            # Permute ommatidia ASSIGNMENTS per eye.  The per-eye total is unchanged
            # exactly (a permutation preserves the sum), so any downstream difference is
            # caused by spatial arrangement alone.
            eye_lum = np.stack([eye_lum[e][self._perm[e]] for e in range(2)], axis=0)

        mean_lum = eye_lum.mean(axis=1)
        lum = eye_lum / np.maximum(mean_lum[:, None], 1e-9)

        dt_s = 0.0 if self._t_prev is None else float(time_s) - float(self._t_prev)
        fresh = self._t_prev is None
        if fresh or dt_s <= 0.0:
            # First sample (or a non-advancing clock): adopt the state, emit no contrast.
            alpha = 1.0
            dt_s = 0.0
        else:
            # EXACT exponential discretisation of dh/dt = (x - h)/tau, i.e. the fraction
            # of the gap the low-pass state closes in one interval.  NB a naive
            # alpha = dt/tau would reach 1.0 whenever the sampling interval equals tau
            # (tau = 20 ms, dt = 20 ms here) and the high-pass would then output
            # identically zero -- a silent, total loss of the temporal signal.  This was
            # caught by a measured motion_energy of ~1e-14 and fixed; it changes no
            # verdict, because the verdict statistic is the temporal std of the RAW
            # readout, not of the high-pass output.
            alpha = 1.0 - float(np.exp(-dt_s / (float(self.config.temporal_tau_ms) / 1000.0)))
        self._hp = self._hp + alpha * (lum - self._hp)
        temporal = float(self.config.contrast_gain) * (lum - self._hp)
        if fresh:
            temporal = np.zeros_like(temporal)
        self._t_prev = float(time_s)
        self._n_reads += 1

        spatial = lum - lum.mean(axis=1, keepdims=True)
        contrast_energy = float(np.sqrt(np.mean(np.square(spatial))))
        motion_energy = float(self.config.motion_gain) * float(
            np.sqrt(np.mean(np.square(temporal))))

        warn = list(self.warnings)
        if fresh:
            warn.append("first sample: temporal contrast is 0 by definition (no history yet)")
        ptp = np.ptp(eye_lum, axis=1)
        for e, name in enumerate(("left", "right")):
            if ptp[e] == 0.0:
                warn.append(
                    f"{name} eye frame is CONSTANT over all {n} ommatidia (raw readout "
                    "span exactly 0): the render is blank or the scene is spatially "
                    "featureless.  Any statistic from this condition is invalid.")
        if not np.isfinite(eye_lum).all():
            warn.append("readout contains non-finite values")

        return VisualFeatures(
            eye_luminance=eye_lum, luminance=lum, temporal_contrast=temporal,
            spatial_contrast=spatial, mean_luminance=mean_lum,
            contrast_energy=contrast_energy, motion_energy=motion_energy,
            time_s=float(time_s), n_ommatidia=n, blind=False, scrambled=scrambled,
            warnings=tuple(warn),
        )

    def _zero_features(self, time_s: float, note: str) -> VisualFeatures:
        n = int(self.n_ommatidia)
        z = np.zeros((2, n))
        self._t_prev = float(time_s)
        self._n_reads += 1
        warn = list(self.warnings) + [note]
        return VisualFeatures(
            eye_luminance=z.copy(), luminance=z.copy(), temporal_contrast=z.copy(),
            spatial_contrast=z.copy(), mean_luminance=np.zeros(2),
            contrast_energy=0.0, motion_energy=0.0, time_s=float(time_s),
            n_ommatidia=n, blind=bool(self.config.blind), scrambled=False,
            warnings=tuple(warn),
        )


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
def _default_to_nA() -> Callable:
    """The project's one explicit mV -> nA helper, imported lazily."""
    try:
        from ..receptors import to_nA  # noqa: PLC0415
    except ImportError:  # pragma: no cover - only in a broken package layout
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from receptors import to_nA  # type: ignore  # noqa: PLC0415
    return to_nA


def visual_drive_nA(features: VisualFeatures, config: VisionConfig,
                    to_nA: "Callable | None" = None) -> dict:
    """Turn one visual sample into a per-channel current for the neural tier.

    ``to_nA`` MUST be ``engine.receptors.to_nA`` (the conversion helper that requires an
    explicit factor and a provenance string).  It is passed in so the dependency is
    visible at the call site; ``None`` imports it lazily.

    REFUSES -- raises ``UnitConversionRefused`` -- if
    ``config.drive_scale_nA_per_unit`` is not explicitly provided.  There is no default
    factor because this project has no measured Drosophila photoreceptor-to-projection
    transduction gain, so any implicit factor would be an invented number.

    Returns a dict with the drive in mV (the DRIVE convention), the drive in nA, the
    channel names, the factor and its provenance, and the caveats that apply.
    """
    if to_nA is None:
        to_nA = _default_to_nA()
    if config.drive_scale_nA_per_unit is None:
        raise UnitConversionRefused(
            "VisionConfig.drive_scale_nA_per_unit was not provided, so no visual drive "
            "can be expressed in nanoamperes. This project has no measured Drosophila "
            "photoreceptor-to-projection-neuron transduction gain; supply the factor "
            "explicitly (and record it as ASSUMED with a sweep range).")
    if not isinstance(features, VisualFeatures):
        raise TypeError("features must be a VisualFeatures instance")

    n_ch = int(config.n_channels)
    rms_temporal = [float(np.sqrt(np.mean(np.square(features.temporal_contrast[e]))))
                    for e in range(2)]
    rms_spatial = [float(np.sqrt(np.mean(np.square(features.spatial_contrast[e]))))
                   for e in range(2)]

    if n_ch == 1:
        names = ["pooled_visual_activation"]
        pooled = float(np.sqrt(np.mean(np.square(features.temporal_contrast))))
        drive_mV = np.array([pooled], dtype=float)
    elif n_ch == 2:
        names = ["left_eye_activation", "right_eye_activation"]
        drive_mV = np.array(rms_temporal, dtype=float)
    else:
        names = ["left_motion", "right_motion", "left_spatial", "right_spatial"]
        drive_mV = np.array(rms_temporal + rms_spatial, dtype=float)

    if features.blind:
        # Applied at the readout level (the features are already zero); asserted here so a
        # leak cannot pass silently.
        drive_mV = np.zeros_like(drive_mV)

    drive_nA = np.asarray(to_nA(drive_mV, float(config.drive_scale_nA_per_unit),
                                DRIVE_SCALE_PROVENANCE), dtype=float)
    if features.blind:
        drive_nA = np.zeros_like(drive_nA)

    caveats = [
        HONESTY["retina_calibration"],
        HONESTY["motion_energy_is_not_emd"],
        HONESTY["assumed_drive_scale"],
    ]
    if str(config.unit_contract) != DRIVE_UNIT_CONTRACT:
        caveats.append(
            "caller supplied a different unit_contract string; the drive below still went "
            "through engine.receptors.to_nA, but the contract it is reported under is the "
            f"caller's: {config.unit_contract!r}")
    return {
        "channel_names": names,
        "drive_mV": drive_mV,
        "drive_nA": drive_nA,
        "total_nA": float(np.abs(drive_nA).sum()),
        "n_channels": n_ch,
        "drive_scale_nA_per_unit": float(config.drive_scale_nA_per_unit),
        "drive_scale_provenance": DRIVE_SCALE_PROVENANCE,
        "unit_contract": str(config.unit_contract),
        "conversion_helper": "engine.receptors.to_nA(drive_mV, nA_per_mV, provenance)",
        "blind": bool(features.blind),
        "scrambled": bool(features.scrambled),
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Non-degeneracy statistics -- the falsifiable claim
# ---------------------------------------------------------------------------
def non_degeneracy_statistics(series: Sequence[VisualFeatures]) -> dict:
    """Decide DEGENERATE vs INFORMATIVE for one recorded series, with the raw numbers.

    ``verdict`` is exactly one of ``"INFORMATIVE"``, ``"DEGENERATE"`` or
    ``"EVIDENCE NOT FOUND"``:

    INFORMATIVE       median per-ommatidium temporal std is ABOVE
                      ``NONDEGENERACY_MIN_TEMPORAL_STD`` and the mean absolute spatial
                      contrast is above ``NONDEGENERACY_MIN_SPATIAL_CONTRAST``: the eye
                      receives a pattern that CHANGES over time.
    DEGENERATE        median temporal std at or below the quantisation floor: whatever
                      the eye receives, it does not change enough to carry a motion
                      signal.  A blind control lands here by construction.
    EVIDENCE NOT FOUND  the series is too short (< ``MIN_SERIES_LENGTH`` samples), or the
                      readouts are all constant so nothing was actually measured, or
                      temporal contrast exists with no spatial structure at all.  The
                      question is then not settled -- which is reported instead of a
                      verdict.

    The measured raw numbers are always returned, whatever the verdict.
    """
    series = list(series or [])
    out: dict[str, Any] = {
        "n_samples": len(series),
        "n_ommatidia": int(series[0].n_ommatidia) if series else 0,
        "min_temporal_std_threshold": float(NONDEGENERACY_MIN_TEMPORAL_STD),
        "min_spatial_contrast_floor": float(NONDEGENERACY_MIN_SPATIAL_CONTRAST),
        "threshold_derivation": THRESHOLD_DERIVATION,
        "min_series_length": int(MIN_SERIES_LENGTH),
    }
    if not series:
        out.update({"verdict": "EVIDENCE NOT FOUND",
                    "reason": "empty series: nothing was measured",
                    "all_readouts_constant": True})
        return out

    raw = np.stack([np.asarray(f.eye_luminance, float) for f in series], axis=0)  # (T,2,n)
    tstd = raw.std(axis=0)                                            # (2, n)
    spatial = np.stack([np.asarray(f.spatial_contrast, float) for f in series], axis=0)
    hp = np.stack([np.asarray(f.temporal_contrast, float) for f in series], axis=0)
    lum_mean = np.stack([np.asarray(f.mean_luminance, float) for f in series], axis=0)

    all_constant = bool(np.all(np.ptp(raw, axis=0) == 0.0))
    blind = bool(all(f.blind for f in series))
    scrambled = bool(any(f.scrambled for f in series))
    median_all = float(np.median(tstd))
    per_eye_median = [float(np.median(tstd[e])) for e in range(tstd.shape[0])]
    frac_above = float(np.mean(tstd > NONDEGENERACY_MIN_TEMPORAL_STD))
    frac_above_per_eye = [float(np.mean(tstd[e] > NONDEGENERACY_MIN_TEMPORAL_STD))
                          for e in range(tstd.shape[0])]
    spatial_mean = float(np.mean(np.abs(spatial)))
    spatial_mean_per_eye = [float(np.mean(np.abs(spatial[:, e]))) for e in range(2)]

    out.update({
        "blind": blind, "scrambled": scrambled,
        "all_readouts_constant": all_constant,
        # --- the raw numbers the verdict is built from -------------------------
        "temporal_std_median_per_ommatidium": median_all,
        "temporal_std_median_per_eye": per_eye_median,
        "temporal_std_mean_per_eye": [float(tstd[e].mean()) for e in range(tstd.shape[0])],
        "temporal_std_max_per_eye": [float(tstd[e].max()) for e in range(tstd.shape[0])],
        "temporal_std_p90_per_eye": [float(np.percentile(tstd[e], 90))
                                     for e in range(tstd.shape[0])],
        "fraction_ommatidia_above_threshold": frac_above,
        "fraction_ommatidia_above_threshold_per_eye": frac_above_per_eye,
        "temporal_std_over_threshold_ratio": float(median_all / NONDEGENERACY_MIN_TEMPORAL_STD),
        "spatial_contrast_mean_abs": spatial_mean,
        "spatial_contrast_mean_abs_per_eye": spatial_mean_per_eye,
        "spatial_contrast_rms": float(np.sqrt(np.mean(np.square(spatial)))),
        "motion_energy_mean": float(np.mean([f.motion_energy for f in series])),
        "motion_energy_max": float(np.max([f.motion_energy for f in series])),
        "highpass_contrast_rms": float(np.sqrt(np.mean(np.square(hp)))),
        "mean_luminance_per_eye": [float(v) for v in lum_mean.mean(axis=0)],
        "eye_luminance_std_per_eye": [float(raw[:, e].std()) for e in range(2)],
    })

    # --- verdict ------------------------------------------------------------
    if blind and all_constant:
        out["verdict"] = "DEGENERATE"
        out["reason"] = (
            "blind control: the visual drive is forced to zero at the readout level, so "
            "every feature array is exactly 0 and the series is degenerate BY "
            "CONSTRUCTION.  This is a declared control, not a failed measurement.")
    elif all_constant:
        out["verdict"] = "EVIDENCE NOT FOUND"
        out["reason"] = (
            "every ommatidium returned exactly the same value at every sample: the "
            "readouts are constant, so nothing about visual information can be settled "
            "(a blank render looks like this and must be reported as invalid, not as "
            "a degenerate-but-measured scene).")
    elif len(series) < MIN_SERIES_LENGTH:
        out["verdict"] = "EVIDENCE NOT FOUND"
        out["reason"] = (f"series has {len(series)} samples, fewer than the "
                         f"pre-registered minimum {MIN_SERIES_LENGTH}")
    elif median_all <= NONDEGENERACY_MIN_TEMPORAL_STD:
        out["verdict"] = "DEGENERATE"
        out["reason"] = (
            f"median per-ommatidium temporal std {median_all:.3e} is at or below the "
            f"quantisation floor {NONDEGENERACY_MIN_TEMPORAL_STD:.1e}: the image may vary "
            "spatially but it does not change over time, so it carries no motion signal.")
    elif spatial_mean > NONDEGENERACY_MIN_SPATIAL_CONTRAST:
        out["verdict"] = "INFORMATIVE"
        out["reason"] = (
            f"median per-ommatidium temporal std {median_all:.3e} is "
            f"{median_all / NONDEGENERACY_MIN_TEMPORAL_STD:.1f}x the quantisation floor "
            f"{NONDEGENERACY_MIN_TEMPORAL_STD:.1e}, and mean |spatial contrast| "
            f"{spatial_mean:.3e} is above the floor "
            f"{NONDEGENERACY_MIN_SPATIAL_CONTRAST:.1e}: the eye receives a pattern that "
            "changes over time.")
    else:
        out["verdict"] = "EVIDENCE NOT FOUND"
        out["reason"] = (
            f"temporal std {median_all:.3e} is above the floor but mean |spatial "
            f"contrast| {spatial_mean:.3e} is not above {NONDEGENERACY_MIN_SPATIAL_CONTRAST:.1e}: "
            "the readout changes without carrying spatial structure, which does not "
            "settle whether the scene is informative.")
    return out


# ---------------------------------------------------------------------------
# Honesty boundary
# ---------------------------------------------------------------------------
CAVEATS: tuple = (
    "APPROXIMATE OPTICS: FlyGym's fisheye Retina is calibrated for FlyGym's OWN eye "
    "placement, so ommatidia readouts on the NeuroMechFly body are approximate.",
    "NO REFERENCE DATASET: this project has no Drosophila visual-response dataset, so "
    "nothing here can say whether a fly's visual system is NORMAL.  Only degenerate vs "
    "informative is testable.",
    "NO MEASURED DROSOPHILA CONSTANTS: every gain and time constant is either derived "
    "from the readout quantisation geometry or marked ASSUMED with a sweep range.",
    "NOT A MOTION DETECTOR: no Reichardt correlator, no lamina/T4/T5 model, no "
    "direction selectivity.  motion_energy is a temporal-variation proxy.",
    "SPATIAL vs TEMPORAL: a static scene can have large SPATIAL contrast (sky against "
    "ground, the black outside of the fisheye disc) and exactly ZERO temporal contrast. "
    "The non-degeneracy criterion is about the temporal term.",
    "SELF-MOTION IS CONTENT: while the fly walks, its own legs and the ground under it "
    "move through its visual field, so 'no object in the scene' is not the same thing as "
    "'no temporal contrast'.  The experiment measures this rather than assuming it.",
)

HONESTY: dict = {
    "what_this_is": (
        "A minimal, falsifiable visual front end for the embodied fly: eye cameras -> "
        "ommatidia readouts -> luminance / temporal-contrast / spatial-contrast features "
        "-> a per-channel drive for the visual projection neurons. It exists so that "
        "'the input is degenerate or informative' can be MEASURED instead of assumed."),
    "retina_calibration": (
        "CAVEAT CARRIED FROM FLYGYM'S OWN DOCSTRING: the fisheye Retina is calibrated for "
        "FlyGym's own eye placement, so ommatidia readouts on the NeuroMechFly body are "
        "APPROXIMATE. These are approximate optics on a body the Retina was not "
        "calibrated for -- they are not measured Drosophila optics and must never be "
        "reported as such."),
    "no_reference_dataset": (
        "This project has NO Drosophila visual-response reference dataset: no measured "
        "photoreceptor responses, no ERG traces, no T4/T5 recordings. Therefore nothing "
        "here can assess whether the fly's visual system is normal, and no claim of that "
        "kind is made. The only claim available is the relative one: is the visual input "
        "degenerate or informative?"),
    "approximate_optics": (
        "Camera placement, fovy, fisheye distortion and the ommatidia lattice all come "
        "from FlyGym's shipped assets (fovy fixed at 157 deg for NeuroMechFly); they were "
        "not re-derived and were not validated against real Drosophila optics. The "
        "per-eye ommatidia count is equal between the eyes BY CONSTRUCTION (one shared "
        "map), not because both eyes were measured."),
    "assumed_drive_scale": (
        "every gain and every nA conversion factor is ASSUMED, and each carries a sweep "
        "range in ASSUMED_PARAMETERS. Absolute currents reported from this module are "
        "therefore NOT physiological values."),
    "motion_energy_is_not_emd": (
        "motion_energy is the RMS of a first-order high-pass of the gain-normalised "
        "luminance. It is NOT a Reichardt correlator and NOT a T4/T5 response: it has no "
        "direction selectivity and no preferred temporal frequency. It is a temporal-"
        "variation proxy and is reported as one."),
    "spatial_vs_temporal": (
        "A constant scene can still be spatially structured (sky vs ground, the black "
        "region outside the fisheye disc), so SPATIAL contrast alone is NOT evidence of an "
        "informative visual input. The failure mode this module is built to detect is a "
        "large spatial contrast with a temporal contrast at the quantisation floor."),
    "blind_control": (
        "blind=True is a real control: the camera is not queried and every feature array "
        "is exactly zero, so any residual drive would be a leak. A blind condition is "
        "DEGENERATE BY CONSTRUCTION and is reported as such rather than as a measurement."),
    "scramble_control": (
        "scramble=True permutes ommatidia assignments within each eye under a fixed seed. "
        "The per-eye total light is unchanged exactly, so a downstream difference proves "
        "the effect depends on spatial arrangement, not on total light."),
    "units": (
        "features are dimensionless READOUT UNITS (0..1 of full 8-bit luminance scale); "
        "the drive is the mV-equivalent DRIVE that engine.receptors.DRIVE_UNIT defines; "
        "nanoamperes appear only through engine.receptors.to_nA with an explicit factor."),
    "what_is_not_modelled": (
        "no phototransduction cascade, no quantum bumps, no adaptation, no spectral "
        "opponency (yellow and pale channels are simply summed), no optic-lobe circuitry, "
        "no motion detection beyond a temporal high-pass, and no self-occlusion model "
        "other than FlyGym's own hidden-segment geom group."),
}
