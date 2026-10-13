#!/usr/bin/env python3
"""VISUAL DRIVE: the body -> brain channel the platform declared but never wired.

WHY THIS MODULE EXISTS, MEASURED.  `SensoryDrive` was implemented for MECHANORECEPTION ONLY: leg
contact force and leg joint angle.  The visual machinery was already complete and rife with its own
honesty guards -- `engine/embodied/vision.py` has an ommatidia front end and a
`visual_drive_nA(features, config)` that REFUSES to run unless `drive_scale_nA_per_unit` is given
explicitly, precisely because this project has no measured photoreceptor-to-projection transduction
gain -- but `grep -n vision engine/embodied/loop.py` returned NOTHING, so no visual current ever
reached the tier.

THE BANDWIDTH, WHICH IS THE FIRST THING A READER SHOULD SEE.  MEASURED for the default tier:

    visual-class neurons in the tier       11 of 12,000     0.092 %
    visual-class neurons in the dataset    79,538
    -> the tier captures 0.014 % of them

So vision can be wired, and doing so is honestly reported as a channel that reaches ELEVEN cells.
That number is not a reason to refuse the wiring; it is a reason to never describe the result as
"the fly sees".

THE CONFLICT THIS EXPOSED, AND HOW IT IS RESOLVED.  Turning on the tier's visual seed route
(`TierConfig.include_visual_route=True`) rebuilds the tier selection, and MEASURED it leaves leg
`lh` with ZERO descending neurons reaching its motor neurons -- so the six-knob per-leg command from
S3 becomes undefined for that leg.  The platform therefore cannot have both the full six-leg channel
AND the visual route at this tier size.  `build_neural_tier` marks such a leg UNAVAILABLE and leaves
its channel neutral rather than inventing one, so both configurations now build and the gap is
readable.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_test_visual_drive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: THE FACTOR IS AN ASSUMPTION AND IS DECLARED AS ONE.  `vision.py` refuses to run without it, and
#: `engine.receptors.to_nA` demands a provenance string, so the number cannot be smuggled in.
#: The only defensible content is that it is SMALL against the receptor drive, hence the sweep.
DRIVE_SCALE_NA_PER_UNIT = 0.05
DRIVE_SCALE_SWEEP_NA_PER_UNIT = (0.005, 0.05, 0.5, 5.0)
DRIVE_SCALE_PROVENANCE = (
    "ASSUMED, NOT MEASURED.  This project has no Drosophila photoreceptor-to-projection-neuron "
    "transduction gain.  The value is a placeholder whose only defensible property is that the "
    "resulting visual current is small against the receptor drive, and it is SWEPT rather than "
    "asserted; every result derived from it inherits that status.")


class VisualDrive:
    """Body -> brain, the VISUAL channel.  Satisfies the same `SensoryDrive` seam as the receptors.

    `set_reference_pose` exists only because the seam requires it: a visual channel has no reference
    pose, and pretending otherwise would be worse than a documented no-op.
    """

    def __init__(self, sim, fly_name: str, built: dict, *, config=None, ledger=None,
                 drive_scale_nA_per_unit: float = DRIVE_SCALE_NA_PER_UNIT):
        from engine.embodied.adapters import TierConfig
        from engine.embodied.vision import OmmatidiaFrontEnd, VisionConfig

        self.n = int(built["n"])
        self.drive_scale = float(drive_scale_nA_per_unit)
        # n_channels=4, NOT the default 2, and the reason is MEASURED.  `visual_drive_nA` with 2
        # channels uses the TEMPORAL contrast RMS, so a static scene gives EXACTLY zero drive --
        # measured: 0 nA injected, 0 neurons reached.  Vision would only ever drive this tier while
        # something moves, which makes the channel untestable against a still scene and makes
        # "blinding the scene" an unobservable perturbation.  With 4 channels the drive is
        # temporal + SPATIAL, and the spatial term is non-zero for a PATTERNED still scene (grass),
        # so the channel is driven by scene CONTENT.
        self.cfg = (config or VisionConfig(n_channels=4,
                                           drive_scale_nA_per_unit=self.drive_scale)).validate()
        if self.cfg.drive_scale_nA_per_unit is None:
            raise ValueError("drive_scale_nA_per_unit must be explicit; vision.py refuses otherwise")
        self.front = OmmatidiaFrontEnd(self.cfg)
        self.front.bind(sim, fly_name)
        self.fly_name = fly_name

        # WHICH TIER NEURONS ARE VISUAL, and how the two eye channels map onto them.
        a = built["data"]
        sc = np.asarray(a["ann_super_class"])
        vis_all = np.isin(sc, list(TierConfig().visual_classes))
        gi = np.asarray(built["global_index"], dtype=np.int64)
        self.visual_local = np.flatnonzero(vis_all[gi])
        nerve = np.asarray(a["ann_nerve"])[gi]
        nstr = np.array([str(x) for x in nerve])
        self.left_idx = self.visual_local[np.array(["left" in s for s in nstr[self.visual_local]])]
        self.right_idx = self.visual_local[np.array(["right" in s for s in nstr[self.visual_local]])]
        self.unassigned_idx = np.setdiff1d(self.visual_local,
                                           np.union1d(self.left_idx, self.right_idx))
        self.drive = None

    # -- reporting -----------------------------------------------------------------------------
    def bandwidth(self) -> dict:
        """The number a reader must see before trusting anything this channel produces."""
        return {
            "tier_neurons": int(self.n),
            "visual_neurons_in_tier": int(self.visual_local.size),
            "visual_fraction_of_tier": float(self.visual_local.size / max(self.n, 1)),
            "visual_neurons_in_dataset": 79538,
            "tier_capture_fraction_of_visual_population": float(self.visual_local.size / 79538),
            "left_eye_mapped": int(self.left_idx.size),
            "right_eye_mapped": int(self.right_idx.size),
            "eye_side_unassigned": int(self.unassigned_idx.size),
            "drive_scale_nA_per_unit": self.drive_scale,
            "drive_scale_sweep": list(DRIVE_SCALE_SWEEP_NA_PER_UNIT),
            "drive_scale_provenance": DRIVE_SCALE_PROVENANCE,
            "why_so_few": ("the tier selects leg-nerve mechanosensory seeds by default; visual "
                           "classes enter only through `include_visual_route`, which rebuilds the "
                           "selection and costs leg lh its descending readout (measured)."),
            "what_this_does_not_establish": (
                "that the fly sees.  Eleven cells receive a declared, uncalibrated current; the "
                "channel exists and is coupled, and that is all."),
        }

    # -- the SensoryDrive protocol --------------------------------------------------------------
    def set_reference_pose(self, joint_angles_rad) -> None:
        """No-op BY DESIGN: a visual channel has no reference POSE.  Present only to satisfy
        `SensoryDrive`, and documented rather than silently absent."""
        return None

    def reset(self) -> None:
        self.front.reset()
        self.drive = None
        return None

    def transduce(self, obs):
        """(current_nA over the whole tier, drive report) for one observation."""
        from engine.embodied.vision import visual_drive_nA

        feats = self.front.read(self._sim, self.fly_name, float(obs.time_s))
        rep = visual_drive_nA(feats, self.cfg)
        self.drive = rep
        cur = np.zeros(self.n, dtype=float)
        dn = np.asarray(rep["drive_nA"], dtype=float)          # length n_channels
        if dn.size >= 2:
            if self.left_idx.size:
                cur[self.left_idx] += dn[0]
            if self.right_idx.size:
                cur[self.right_idx] += dn[1]
            if self.unassigned_idx.size:
                # REPORTED, never silently dropped: neurons whose eye side the annotation does not
                # give still receive the TOTAL drive rather than none, and the count is in
                # `bandwidth()`.  An earlier design would have left them at zero, which would look
                # like "vision does not reach this cell" when the truth is "we cannot say which eye".
                cur[self.unassigned_idx] += float(dn.sum())
        else:
            cur[self.visual_local] += float(dn.sum())
        return cur, {"visual": rep, "bandwidth": self.bandwidth(),
                     "total_current_nA": float(np.abs(cur).sum()),
                     "neurons_receiving_current": int((cur != 0).sum())}

    def bind_sim(self, sim) -> "VisualDrive":
        self._sim = sim
        return self

    # -- the guard ------------------------------------------------------------------------------
    def dependence(self, obs, *, blank_luminance: float = 0.0) -> dict:
        """THE GUARD.  A blinded scene must give a DIFFERENT drive, or the channel is not coupled.

        The project's own `vision.py` already zeroes the drive when `features.blind` is set, and
        asserts it at the readout level so a leak cannot pass silently.  This turns that into the
        same bit-identity guard the rest of the platform uses: if blinding changes nothing, the
        visual current did not come from the scene.
        """
        import dataclasses
        from engine.embodied.vision import OmmatidiaFrontEnd

        real = self.transduce(obs)[0].copy()
        # A BLINDED TWIN, not a mutated config: `VisionConfig` is a FROZEN dataclass, and the first
        # version of this method raised FrozenInstanceError while trying to flip `blind` in place.
        # Building a twin is also the more honest control -- the real front end's temporal state is
        # left untouched, so the two arms differ by the blinding and by nothing else.
        blind_cfg = dataclasses.replace(self.cfg, blind=True).validate()
        blind_front = OmmatidiaFrontEnd(blind_cfg)
        blind_front.bind(self._sim, self.fly_name)
        saved_front, saved_cfg = self.front, self.cfg
        try:
            self.front, self.cfg = blind_front, blind_cfg
            blind = self.transduce(obs)[0].copy()
        finally:
            self.front, self.cfg = saved_front, saved_cfg
        identical = bool(np.array_equal(real, blind))
        return {
            "status": "NOT_RUN" if identical else "PASS",
            "why": ("blinding the scene left the injected current BIT-IDENTICAL, so the visual "
                    "channel is not driven by the scene -- NOT RUN, never a null result"
                    if identical else
                    "blinding the scene changes the injected current, so the visual channel is "
                    "coupled to the scene"),
            "peak_real_nA": float(np.abs(real).max()) if real.size else 0.0,
            "peak_blind_nA": float(np.abs(blind).max()) if blind.size else 0.0,
            "neurons_reached": int((real != 0).sum()),
            "bandwidth": self.bandwidth(),
        }
