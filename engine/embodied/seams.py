#!/usr/bin/env python3
"""THE FOUR SEAMS OF THE CLOSED LOOP, AS EXPLICIT INTERFACES.

WHY THIS EXISTS, MEASURED.  The platform is two halves that were built separately and meet through
a handful of call sites.  Before this module the only record of those call sites was the call sites
themselves: answering "is the electrode in the loop?" meant `grep -n electrode
engine/embodied/loop.py`, and the answer was an empty line.  That is not a thing anyone should have
to rediscover.

WHAT A "SEAM" IS HERE.  A seam is a DIRECTION OF INFORMATION FLOW between two pillars.  Fixing the
directions makes the platform's shape checkable rather than inferred:

    SensoryDrive    body   -> brain      mechanoreceptors (and, later, vision) -> injected current
    MotorCommand    brain  -> body       the tier's firing -> how the body is actuated
    NeuralRecorder  brain  -> electrode  spikes -> a recorded voltage trace
    NeuralStimulus  electrode -> brain   stimulation current -> injected current

Each is a ``typing.Protocol``, so an implementation satisfies it STRUCTURALLY -- nothing has to
inherit from anything, and a module can be replaced without touching the others.  That is exactly
the property the interface documentation needs and the reason this file exists separately from the
implementations.

``survey()`` then reports, for every seam, WHICH implementations currently exist and whether they
satisfy the protocol.  A seam with no implementation is reported as ``DECLARED_NOT_IMPLEMENTED``
rather than omitted -- an absent seam is a fact about the platform and must be visible in the
documentation, not missing from it.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python -c "from engine.embodied.seams import survey; import json; print(json.dumps(survey(), indent=1))"
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["SensoryDrive", "MotorCommand", "NeuralRecorder", "NeuralStimulus",
           "SEAMS", "OBSERVATION_CHANNELS", "survey"]


@runtime_checkable
class SensoryDrive(Protocol):
    """BODY -> BRAIN.  Turns an observation of the body into current injected into the tier.

    Contract, and it is not decoration:
      * ``transduce`` returns ``(current_nA, drive)`` where ``current_nA`` is a length-n array in
        NANOAMPERES -- the tier's step takes nanoamperes, so returning a normalised drive here is a
        silent unit error of unknown size;
      * ``set_reference_pose`` MUST be called before the first ``transduce`` (the implementation
        raises rather than transducing against an undefined reference).
    """

    def transduce(self, obs: Any) -> tuple[np.ndarray, dict]: ...

    def set_reference_pose(self, joint_angles_rad: np.ndarray) -> None: ...

    def reset(self) -> None: ...


@runtime_checkable
class MotorCommand(Protocol):
    """BRAIN -> BODY.  Whatever turns neural state into an actuation command.

    DELIBERATELY LOOSE.  Today the only implementation decodes two CPG knobs, so the protocol says
    only ``step()``, which is the seam ``BodyBackend.attach_action_source`` already accepts.  When
    a motor-neuron-to-muscle path replaces it, it satisfies this protocol without changing the body.
    """

    def step(self) -> Any: ...


@runtime_checkable
class NeuralRecorder(Protocol):
    """BRAIN -> ELECTRODE.  Turns a spike matrix into a recorded trace.

    Contract:
      * ``record`` takes the tier's OWN spike matrix (``(n_substeps, n_neurons)`` bool).  An
        implementation that requires a pre-truncated matrix would push the neurons it cannot
        locate out of sight, so it must accept the full width and COUNT what it skips.
      * ``dependence`` is the guard: it MUST report NOT_RUN when the trace does not change under a
        zeroed spike matrix.  A recorder that cannot fail this check is not evidence of anything.
    """

    def record(self, spikes: np.ndarray, *, t0_ms: float = 0.0) -> np.ndarray: ...

    def dependence(self, spikes: np.ndarray) -> dict: ...


@runtime_checkable
class NeuralStimulus(Protocol):
    """ELECTRODE -> BRAIN.  Stimulation current injected into the tier.

    DECLARED, NOT IMPLEMENTED.  `engine/electrode_stim.py` models waveforms and geometry, but no
    current reaches the tier, so the record-stimulate loop (brain -> electrode -> stimulus ->
    brain) does not exist.  Declaring the interface is what makes that a fact you can read instead
    of a thing you have to discover.
    """

    def current_nA(self, t_ms: float, n_neurons: int) -> np.ndarray: ...


#: The declared seams, with the module each one is expected to be implemented in.  Module paths are
#: strings, not imports, so that surveying does not import half the project.
SEAMS: dict[str, dict] = {
    "SensoryDrive": {
        "direction": "body -> brain",
        "expected_in": ["engine/embodied/adapters.py"],
        "implementations": ["MechanoReceptorBank"],
        "downgraded_input": ("THIS SEAM'S INPUT IS A NON-PHYSICAL CHANNEL.  The mechanoreceptive "
                             "drive is computed from the contact-force channel, and "
                             "`engine/embodied/force_channel.py` declares that channel a "
                             "NON_PHYSICAL_LOAD: MEASURED, its summed normal force is 4.29-12.16x "
                             "body weight on average and 25.26-27.79x at peak across ten recorded "
                             "episodes, against about 1x for a body standing or walking at constant "
                             "height, and THE CAUSE IS NOT IDENTIFIED (the actuators are excluded as "
                             "the dominant cause by measurement -- gain x1000 and force range x130 "
                             "each moved the residual by only 18 % and 16 %).  So the drive's "
                             "MAGNITUDE is not physical either: only its STRUCTURE and its DEPENDENCE "
                             "on body state are.  'The body sensed X newtons' is not supportable "
                             "anywhere in this project; 'the drive rose when the leg loaded' is.  "
                             "Enforced by `tools_force_downgrade_audit.py` and pinned by "
                             "`tools_invariants.py`'s `force_unit_labels_are_honest`.  ROUTE CHOSEN: "
                             "option (b) of PLATFORM_SELFCONSISTENT_PLAN section 3/P2, explicit "
                             "downgrade, because option (a) -- softening the contacts to find the "
                             "residual's source -- would change every already-recorded episode and "
                             "require re-recording and re-registering all of them."),
        "limits": ("mechanoreception AND vision now, with three measured caveats.  (1) BANDWIDTH: "
                   "the default tier holds 11 visual-class neurons of 12,000 (0.092 %) against "
                   "79,538 in the dataset -- the tier captures 0.014 % of the visual population, and "
                   "the eye-side annotation is absent for all 11 (left 0, right 0, unassigned 11), "
                   "so the two eye channels CANNOT be separated.  (2) THE CONVERSION IS ASSUMED: "
                   "`drive_scale_nA_per_unit` is a placeholder with a sweep, because this project "
                   "has no measured photoreceptor-to-projection gain; `visual_drive_nA` still "
                   "refuses to run without it.  (3) A CONFIGURATION CONFLICT: enabling "
                   "`TierConfig.include_visual_route` rebuilds the selection and leaves leg `lh` "
                   "with ZERO descending neurons, so its per-leg command is UNAVAILABLE and held "
                   "neutral -- the six-leg channel and the visual route cannot both be full at this "
                   "tier size.  Also MEASURED and worth knowing: with the default n_channels=2 the "
                   "drive is TEMPORAL contrast, which is exactly zero for a still scene, so this "
                   "channel only responds to motion unless n_channels=4 (temporal + spatial) is "
                   "used.  FIXED EN ROUTE: `vision.py`'s `bind()` never re-initialised its state "
                   "after discovering the ommatidia count, so the first `read()` ALWAYS raised a "
                   "broadcast error -- the front end was unusable, which is very likely why vision "
                   "was never wired.  `bind()` now resets, and a shape guard names the fix."),
    },
    "MotorCommand": {
        "direction": "brain -> body",
        "expected_in": ["engine/embodied/loop.py"],
        "implementations": ["CommandedTripodCPG"],
        "actuator_defect_in_our_own_model_build": (
            "THE MUSCLE-DRIVEN STACK COULD NOT REACH SOME JOINTS IN BOTH DIRECTIONS.  THE CAUSE WAS "
            "THIS PROJECT'S OWN MODEL BUILD -- NOT AN UPSTREAM DEFECT, NOT A NEURAL ONE -- AND IT "
            "HAS NOW BEEN LOCATED, EXPLAINED AND REPAIRED.  READ THIS AS A RECORD OF A DEFECT THAT "
            "WAS, PLUS THE TWO THINGS THAT REMAIN.\n"
            "  WHAT IT WAS.  MEASURED: 6 of 42 leg joints were one-way actuated at the standing "
            "pose (LMTibia_pitch, RMTibia_pitch, LHTibia_pitch, RHTibia_pitch, LMTrochanter_pitch, "
            "RMTrochanter_pitch).  A random sweep of 400 configurations over all 7 of the middle "
            "leg's joints produced ZERO bidirectional cases for joint_LMTibia_pitch against 373 of "
            "400 for the fore leg's, so it was structural, not a configuration artefact.\n"
            "  WHERE IT WAS NOT.  The first conclusion was that FlyMimic ships the defect.  It does "
            "NOT: the shipped asset "
            "`flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml` has a "
            "properly antagonistic `joint_LFTibia_pitch` (flex arm -0.01524 -> torque +1.0377, "
            "extensor +0.03324 -> torque -8.5427).  That asset is a ONE-LEG model and this project "
            "builds the six-leg model from it, so the defect was IN THIS REPOSITORY.  Three earlier "
            "repair attempts died on a hypothesis: 'the coxa-anchored world rotation is the cause' "
            "was tested three ways (original transform 36/42 bidirectional, full local copy 34/42, "
            "per-segment mapping 26/42) and is FALSIFIED.\n"
            "  WHAT IT ACTUALLY WAS, NOW PROVEN.  `diagnostics/diag_moment_arm_algebra.py` proves, "
            "at machine precision over all 3780 (muscle, joint) pairs, that a muscle's moment arm "
            "about a joint is decided by ONE SEGMENT of its path -- the single point that joint "
            "moves, and its neighbour -- by the closed form  arm = |r| . |u_perp| . sin(theta_u - "
            "theta_r).  Nothing else in the path and nothing in the neural tier enters.  So a "
            "one-way joint means its muscles' spanning segments give same-signed arms.  MEASURED, "
            "as the angle between the two tibia muscles' approach directions in the joint plane: "
            "fore leg 69.94 deg (arms -0.01025 / +0.00917, antagonistic), right fore leg 55.51 deg, "
            "middle leg 1.79 deg (arms +0.12347 / +0.10027, same sign), right middle 3.59 deg, hind "
            "7.91 deg, right hind 8.16 deg.  The fore legs' two muscles approach ~60-70 deg apart; "
            "the middle and hind legs' approach within 8 deg and therefore push the same way.\n"
            "  THE REPAIR.  `tools_repair_moment_arms.py` takes the two endpoints of each spanning "
            "segment, reads their coordinates in the reference fore-leg joint's own frame (anchor, "
            "world axis, in-plane basis), and re-places the target sites at those same coordinates "
            "in the target joint's frame -- the invariant is then the reference's moment arms BY "
            "CONSTRUCTION.  Chirality is respected (LM, LH <- LF; RM, RH <- RF).  40 sites re-placed, "
            "0 refusals.  MEASURED after: 42 of 42 leg joints bidirectional at the keyframe, and on "
            "the repaired joints the arms equal the reference leg's to 3.0e-10 mm.  Pinned by "
            "`tools_test_bidirectionality.py`, which FAILS on the pre-repair model naming the same "
            "six joints (injection-verified).\n"
            "  STILL NOT ESTABLISHED, AND NOT TO BE SOFTENED.  (1) The arms are pose-dependent and "
            "are matched at the model's keyframe; over 100 poses perturbed by +-0.08 rad the count "
            "is 42 in the median but falls to 41 at the worst pose (joint_RHTrochanter_pitch "
            "momentarily one-way).  (2) Bidirectional moment arms are NECESSARY AND NOT SUFFICIENT. "
            "After the repair the fly STILL DOES NOT STAND: the static-equilibrium solve reports "
            "NO_EQUILIBRIUM_FOUND (residual reduced to 10.8 % of its no-activation value, best "
            "activation still sinking 1.286 mm with 20.5 deg of tilt on release), and the solved-"
            "activation test reports SOLVED_ACTIVATION_DOES_NOT_STAND.  The artificial torque "
            "profile's 8 largest requirements move from 3 of 8 reachable by ANY activation to 6 of "
            "8, with LMTrochanter_pitch still short by 5.92x and LHCoxa_pitch by 0.52x.  (3) A new "
            "and unexplained MEASUREMENT at the standing pose: many muscles output NEGATIVE force at "
            "ctrl=1 (LHC_sternal_posterior_rotator -143.67, LMF_trochanter_extensor -17.76, "
            "LHTibia_flex -10.24), so the torque a muscle can deliver is not simply arm x positive "
            "force.  That is open, and it means the bidirectionality census bounds the ACTIVE, "
            "controllable direction only.  NOT a neuron problem: both stacks run the same connectome "
            "tier, and every number above is simulator output -- no real-fly recording exists in "
            "this project."),
        "lengthrange_defect_in_our_own_model_build": (
            "THE MUSCLE-DRIVEN STACK HAD 14 ACTUATORS THAT COULD NOT PRODUCE FORCE AT ALL.  OURS, "
            "FOUND AND REPAIRED IN THE SAME ROUND AS THE ONE-WAY JOINTS, AND FIXED FOR A DIFFERENT "
            "REASON -- SO IT IS RECORDED SEPARATELY RATHER THAN FOLDED IN.\n"
            "  WHAT IT WAS.  MEASURED at the standing pose: 14 of 90 leg muscles produced EXACTLY "
            "ZERO force at ctrl=1, and 5 of the 6 muscles on joint_LMTrochanter_pitch were among "
            "them -- which is why that joint, even after its moment arms were repaired, could still "
            "only deliver [+0.000, +0.037] against a requirement of +0.221, short by 5.92x.  A "
            "muscle outside the band over which MuJoCo's force-length curve is defined returns "
            "zero, and these were outside it.\n"
            "  WHY.  `tools_mirror_muscles.py` scales the source muscle's band by the ratio of the "
            "two legs' tendon lengths, and it measured that ratio at qpos = 0 -- a fresh MjData's "
            "default, where every leg joint is zero, a collapsed pose no leg is ever in.  MEASURED "
            "consequence: the middle leg's trochanter extensor band sat at [0.7544, 0.7958] around "
            "a 0.3956-long tendon, i.e. 8.7 band-widths below its own curve.\n"
            "  THE INVARIANCE THAT MUST HOLD, and it needs no knowledge of what posture the "
            "upstream band was authored at: for HOMOLOGOUS muscles at HOMOLOGOUS postures, the "
            "fractional position in the band must be the same.  If it holds, every leg's muscle "
            "behaves like the fore leg's at the same joint angles, which is the entire point of "
            "building five legs by mirroring one.  MEASURED before: 75 of 75 homologous "
            "comparisons wrong, worst 9.18 band-widths.\n"
            "  THE REPAIR.  Measure the length ratio at the DECLARED NEUTRAL posture (all six legs "
            "set to the fore leg's own springref joint angles) instead of at qpos=0, and make that "
            "the default rather than a flag.  Then frac_target = (L_t - lo_f.r)/(w_f.r) with "
            "r = L_t/L_f, which equals (L_f - lo_f)/w_f = frac_fore identically.  The fore leg's "
            "band is always the base and never the target's current value, so the rule is "
            "IDEMPOTENT.  A SECOND PASS IS REQUIRED AND WAS MEASURED TO BE NECESSARY: the "
            "moment-arm repair moves 40 attachment sites, which changes tendon LENGTHS, and on its "
            "own it pushed the middle leg's tibia flexor from +0.238 to -11.047 band-widths.  So "
            "`tools_repair_muscle_lengthrange.py` runs on the FINAL geometry, as the standing "
            "builder's STAGE 2c.\n"
            "  MEASURED AFTER: homologous fraction worst difference 0.0000; muscles outside their "
            "band at the keyframe 28 -> 14 (magnitude -11.0 -> -1.02 band-widths); ZERO-force "
            "muscles 14 -> 1; muscles producing more than 1.0 at ctrl=1 76/90 -> 89/90.  Pinned by "
            "`tools_test_muscle_operating_range.py`, which FAILS before the repair.\n"
            "  WHAT IT CHANGED, HONESTLY.  The passive settle moved from 0.7163 mm (crouched but "
            "upright, up.up0 = 0.95) to 1.3917 mm (higher but tilted ~46 deg, up.up0 = 0.694).  The "
            "static-equilibrium solve moved from NO_EQUILIBRIUM_FOUND to "
            "EQUILIBRIUM_FOUND_BUT_UNSTABLE with the residual at 4.21 % of its no-activation value "
            "-- the first activation found that makes the pose an equilibrium.  THE FLY STILL DOES "
            "NOT STAND: on release the pose tilts to 47.37 deg and only 3 feet remain down.\n"
            "  STILL OPEN, WITH THE MEASUREMENT.  (1) The stance keyframe is NOT the declared "
            "neutral posture it claims to be: Trochanter_pitch is -1.34..-1.64 against a declared "
            "-2.8 and Tibia_pitch +0.71..+1.61 against +2.0, i.e. 1.3-1.5 rad out, so the middle "
            "leg's tibia flexor runs at 0.3956 mm at the keyframe against 1.017 mm at the neutral "
            "posture.  That is where the remaining 14 out-of-band muscles come from.  "
            "(2) At act=0, 42 of 90 muscles still carry PASSIVE force, up to 78.2 force units "
            "(3.2 body weights), and the net passive torque over left-leg joints (-12.34) and "
            "right-leg joints (-13.89) does not cancel; joint passive stiffness is 0.000, so "
            "contacts and these passive forces are all that hold the pose.  Every number is "
            "simulator output; no real-fly recording exists in this project."),
"limits": ("SIX knobs now: one intrinsic-frequency scale per leg, each a function of that "
                   "leg's own descending readout, with leg specificity measured at 0.00 % cross-leg "
                   "(the scalar speed/turn path is retained bit-identically for historical "
                   "episodes).  The GAIT itself is still a pre-recorded motion-capture chase, so the "
                   "connectome MODULATES the gait; it does not generate it.  112 of 195 readout "
                   "neurons reach no leg's motor neurons and carry no leg-specific command."),
    },
    "NeuralRecorder": {
        "direction": "brain -> electrode",
        "expected_in": ["engine/embodied/electrode_in_loop.py"],
        "implementations": ["InLoopElectrode"],
        "works_in": ("BOTH loops: `engine/embodied/loop.py` (whose motor path is two CPG knobs) and "
                     "`run_six_leg_neural_loop.py`, where each muscle's activation IS its own motor "
                     "neurons' firing rate.  The second is the one that matters causally -- an "
                     "electrode there records the spikes that move the body -- and it is verified: "
                     "silencing leg LF's motor neurons changes 212123 spike entries and 2244 trace "
                     "samples.  The spikes offered to the recorder are POST-lesion, so the record "
                     "cannot contain activity that never reached a muscle."),
        "limits": ("the spike -> current kernel is ILLUSTRATIVE (charge per spike, pulse duration), "
                   "and 4263 of 12000 tier neurons have no soma position, so 64 % of the network is "
                   "recordable.  Only the electrode-network COUPLING is established, never the "
                   "amplitude.  CAUSE, DIAGNOSED: those 4263 root ids are simply ABSENT from "
                   "somas_v1.parquet -- not invalid (0 rows have valid=False), not at the origin (0 "
                   "rows), and not a query or resolution artefact.  The other locally available BANC "
                   "file (banc_neurons.csv.gz) carries annotations only and NO coordinates, so this "
                   "coverage CANNOT be raised without another data source; it can only be declared, "
                   "which is why 7737 must never be described as the whole network."),
    },
    "NeuralStimulus": {
        "direction": "electrode -> brain",
        "expected_in": ["engine/embodied/electrode_in_loop.py"],
        "implementations": ["InLoopStimulus"],
        "works_in": ("THE MUSCLE-DRIVING LOOP, since R13: `run_six_leg_neural_loop.run(stimulus=...)`. "
                     "Before that this seam had only ever been exercised against a tier advanced IN "
                     "ISOLATION -- two identical networks, one stimulated, spikes compared -- which "
                     "proves the current reaches a membrane and proves nothing about whether it "
                     "reaches a MUSCLE.  Here it does: at 1000 nA on channel 0 with the DECLARED 50 um "
                     "capture radius, silencing nothing and recruiting 53 of 12000 tier neurons, the "
                     "joint excursion of the legs moves by up to 0.32256 rad against the unstimulated "
                     "run, the rendered frames change (mean |diff| 0.12017), AND the electrode trace "
                     "changes (max |diff| 0.019958 mV) -- so all four seams now sit in ONE chain.  "
                     "TWO GUARDS make that a measurement rather than an artefact: (1) attaching a "
                     "stimulus forces the interval to be advanced one substep at a time instead of in "
                     "one call, and those two paths are asserted BIT-IDENTICAL (0 differing entries "
                     "of 120000), so any change is the current and not the restructuring; (2) at "
                     "1e-6 nA the excursion changes by EXACTLY 0.0 rad while at 1000 nA it changes by "
                     "0.32256 rad, so the effect scales with amplitude.  The stimulus is built from "
                     "the loop's OWN context (`make_in_loop_stimulus`), and a stimulus addressing a "
                     "different tier width is REFUSED, because one built from a separate `build()` "
                     "would inject into different neurons, still change the spikes, and still look "
                     "correct."),
        "measured_range": ("THE DECLARED 50 um CAPTURE RADIUS NOW HAS A MEASURED OPERATING RANGE, "
                           "and the measurement does NOT confirm it -- see "
                           "`tools_capture_radius_dose_response.py` and "
                           "`outputs/capture_radius_dose_response.json`.  The full loop was run once "
                           "per radius at the same amplitude (1000 nA), repetition rate (20 Hz) and "
                           "duration (0.5 s), against one unstimulated baseline.  MEASURED: the "
                           "effect is EXACTLY 0.00000 rad at 5, 10 and 15 um, where the rule recruits "
                           "0 neurons, and first becomes non-zero at 18 um (2 neurons, 0.13310 rad); "
                           "the response is then NON-MONOTONE -- 30 um 0.50248 rad, 50 um (the "
                           "DECLARED value) 0.32688 rad, a 35 % DROP, and 80 um 0.66697 rad against "
                           "120 um 0.13698 rad, a 79 % DROP.  So no radius can be preferred on "
                           "behavioural grounds, and 50 um remains UNDEFENDED rather than confirmed.  "
                           "WHY THAT IS THE RIGHT ANSWER AND NOT A BUG: the recruitment weights are "
                           "normalised to sum to 1, so the charge delivered is IDENTICAL (5000 nA*ms) "
                           "at every recruiting radius -- what the curve varies is only the SPATIAL "
                           "DISTRIBUTION of the current, which is precisely the quantity this module "
                           "already declares as not established.  The non-monotone curve IS that "
                           "declaration, measured.  GUARDS: the baseline is asserted bit-identical on "
                           "a rerun (determinism is what makes any point of the curve mean "
                           "something); and 'no recruitment, no effect' is INJECTED, not asserted -- "
                           "transplanting the 200 um recruitment pattern onto a radius that recruits "
                           "nothing turns the effect from exactly 0.00000 into 0.13702 rad.  That "
                           "injection had to be fixed once: forcing an ARBITRARY single neuron gave "
                           "EXACTLY zero, because an arbitrary neuron need not be a motor neuron of "
                           "any leg, which is why the control transplants a recruitment PATTERN "
                           "rather than one index."),
        "limits": ("the per-phase charge is checked against the stim module's own limits -- a "
                   "MEASURED_CITED density ceiling (Shepherd et al.) and an ASSUMED current ceiling "
                   "with a sweep -- and those provenances are carried through.  THE ASSUMPTION IS "
                   "THE SIGN, and it was misdescribed at first: MEASURED, at the tier's dt (0.5 ms) "
                   "the pulse's recovery phase falls between substeps and is never sampled "
                   "(sampled signs = {+}), so both modes inject the same cathodic train and differ "
                   "ONLY in sign -- recruiting depolarises (12544 spikes/30 substeps), the literal "
                   "sign hyperpolarises (0).  Charge balance is therefore NOT the mechanism and no "
                   "self-cancellation occurs.  RECRUITMENT IS TINY: the DECLARED 50 um capture "
                   "radius reaches 53 neurons = 0.7 % of the 7737 recordable, 25 um reaches 11, and "
                   "10 um reaches ZERO (swept, not asserted).  So amplitude effects are meaningful; "
                   "the spatial recruitment pattern is not.  NO charge-safety claim."),
    },
}


#: WHAT CAN BE OBSERVED SIMULTANEOUSLY -- a separate question from the four seams, and one that must
#: be MEASURED rather than assumed.  Two channels demonstrated on two different runs are two
#: demonstrations, not an observation layer; so "together" below names the channels that were
#: actually recorded from ONE body, ONE tier, ONE clock, in ONE call to the loop.
OBSERVATION_CHANNELS: dict[str, dict] = {
    "camera_ring": {
        "in_stack": "B (FlyMimic muscles)",
        "together": True,
        "where": "tools_record_all_channels.py (and tools_render_neural_loop.py alone)",
        "evidence": ("outputs/all_channels_check.json, outputs/limb_stackb_check.json, "
                     "outputs/arena/stackb_all_channels/"),
        "measured": ("6 cameras, 20 frames, health gate PASS at 80.6 % fiducial hit rate; "
                     "reconstruction reaches 7 of 12 fiducials, world RMS 0.9663 mm.  THE FLY'S OWN "
                     "MARKERS ARE NOW ATTACHED -- 20 of them, one per segment, massless -- so "
                     "`limb_tracking.py` runs on these episodes: 119 of 400 frame-marker slots "
                     "accepted, marker position error median 0.0675 mm, and JOINT ANGLES come out "
                     "(leg RF's femur in 16 of 20 frames, median error 0.939 deg, p95 2.673 deg)"),
        "limits": ("the angles are PARTIAL, and that is the honest headline: only ONE bone clears 10 "
                   "frames, because a six-camera ring around a SELF-OCCLUDING body leaves most "
                   "markers visible from one or two cameras per frame while triangulation needs "
                   "three.  Frame 0's marker identity is seeded from the SIMULATOR as a stand-in for "
                   "a human annotation -- every marker is the same sphere, so the pixels alone cannot "
                   "say which is which -- and the truth the errors are scored against is the "
                   "simulator's own: there is no real-fly recording anywhere in this project.  All "
                   "six legs bind their FEMUR, because stack B has no trochanter on the mid and hind "
                   "legs, so a 'femur' bone here is MARKER-TO-MARKER, not joint-centre-to-joint-"
                   "centre.  DEFECTS FIXED HERE, all measured: (1) the recorder threw away each "
                   "blob's PIXEL AREA, which is the only thing that can separate the arena's 0.30 mm "
                   "fiducials from the fly's 0.12 mm markers, so the nearest-blob association let the "
                   "fly's markers steal the floor-level fiducial tier and the fiducial triangulation "
                   "fell from 7/12 to 3/12.  A global area cutoff was then tried and MEASURED not to "
                   "work -- the two size distributions nearly touch (fly markers to 37 px, fiducials "
                   "from 39 px) because perspective makes a near fly marker as large as a far "
                   "fiducial -- so the test is now DEPTH-RELATIVE, with an effective radius "
                   "CALIBRATED from the data: 0.230 mm measured against 0.300 mm nominal, because the "
                   "detector thresholds at grey 240 while the emissive markers peak near 251 and only "
                   "the saturated core is counted.  (2) The recorders converted RGB to grey with the "
                   "arithmetic MEAN while engine/embodied/natural_scene.py uses standard LUMINANCE; "
                   "the mean understates the warm emissive markers (243 against 250.9 at the "
                   "brightest pixel), and that is what destroyed the size separation.  Both now go "
                   "through `arena.marker_grey`.  (3) THE ONE THAT MATTERED MOST: the per-camera "
                   "offset fitted on the static fiducials was being applied to the fly's markers, and "
                   "it is MEASURED to make them worse -- median residual 1.57 px without it against "
                   "2.92 px with it, which took the triangulated fly markers from 19 of 20 to 0 of "
                   "20 and silently produced an EMPTY joint-angle table.  It stays off, and "
                   "`limb_tracking` records `projection_offsets.applied` in its output."),
    },
    "electrode": {
        "in_stack": "B (FlyMimic muscles)",
        "together": True,
        "where": "tools_record_all_channels.py, engine/embodied/electrode_in_loop.py",
        "evidence": "outputs/arena/stackb_all_channels/electrode.npz",
        "measured": ("4 channels, fed the POST-lesion spikes -- the ones that set the muscle "
                     "activations -- on the same interval counter as the frames; coupling guard PASS"),
        "limits": ("7737 of 12000 tier neurons have a soma position, so 64 % is recordable; the "
                   "spike -> current kernel is ILLUSTRATIVE, so amplitude is never established."),
    },
    "vision": {
        "in_stack": "A (NeuroMechFly, CPG replay) ONLY",
        "together": False,
        "where": "engine/embodied/vision.py, engine/embodied/visual_drive.py",
        "evidence": "measured in `vision_binding_probe()` and recorded in the episode's markers.json",
        "measured": ("binding the front end to stack B FAILS with "
                     "`AttributeError: 'mujoco._structs.MjData' object has no attribute "
                     "'get_raw_vision'` -- it needs a FlyGym `Simulation` and stack B is a raw "
                     "`MjData` built from FlyMimic's MJCF"),
        "limits": ("so this platform observes TWO channels on the muscle-driven body, never three.  "
                   "The visual channel also carries the bandwidth shortfall registered under "
                   "SensoryDrive (11 of 12000 tier neurons, 0.092 %)."),
    },
    "together": {
        "recorded_in_one_run": ["camera_ring", "electrode"],
        "absent_by_construction": ["vision"],
        "one_call": ("run_six_leg_neural_loop.run(..., frame_sink=..., spike_sink=...), so there is "
                     "one body, one tier and one interval counter; both sinks are called with the "
                     "SAME interval index `c`"),
        "clock": ("the frames are the intervals with `c % stride == 0`, so the frame times are "
                  "`interval_times[::stride]`; the artifact carries `truth/frame_interval_index` and "
                  "the check recomputes the relation rather than trusting a docstring"),
        "defect_fixed_here": ("the first version wrote `truth/time_s = interval_times[:n_frames]`, "
                              "which understated the frame clock by exactly `stride` -- MEASURED: "
                              "declared 0..0.095 s while the frames were rendered at 0..0.38 s "
                              "(dt ratio 0.25 = 1/4).  Any velocity across frames would have been 4x "
                              "wrong, and the 'shared clock' claim was false in the artifact itself.  "
                              "`record()` now derives the frame times from the recorded indices and "
                              "raises if `frame_dt != stride * DT_COMMAND_S`."),
        "injection": ("ONE lesion must move BOTH channels of the same run, and it is checked: "
                      "silencing leg LF's motor neurons changes the last frame (mean |diff| 0.0177) "
                      "and the electrode trace (max |diff| 0.005072 mV) in the SAME episode."),
        "status": "IMPLEMENTED",
    },
}


def _satisfies(obj: Any, proto: type) -> bool:
    """Structural check, method by method, so the answer does not depend on inheritance."""
    need = {"SensoryDrive": ("transduce", "set_reference_pose", "reset"),
            "MotorCommand": ("step",),
            "NeuralRecorder": ("record", "dependence"),
            "NeuralStimulus": ("current_nA",)}[proto.__name__]
    return all(callable(getattr(obj, m, None)) for m in need)


def survey() -> dict:
    """Which seams are implemented, and do the implementations actually satisfy the protocol?

    Imports are LOCAL and tolerant: this must be runnable in either python environment, and a
    module that cannot be imported (flygym absent in the CPU environment) is reported as
    UNVERIFIABLE rather than silently counted as absent.  "We could not check" is not "it is not
    there".
    """
    out: dict[str, dict] = {}
    for name, spec in SEAMS.items():
        proto = globals()[name]
        found, unverifiable = [], []
        for fq in _CANDIDATES.get(name, ()):
            mod_name, cls_name = fq.rsplit(".", 1)
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                inst_needs_args = True
            except Exception as exc:                                # noqa: BLE001
                unverifiable.append({"candidate": fq, "why": f"{type(exc).__name__}: {exc}"})
                continue
            # check on the CLASS: the methods are what the protocol constrains, and instantiating
            # a receptor bank or a body just to ask has side effects.
            ok = _satisfies(cls, proto)
            found.append({"candidate": fq, "satisfies_protocol": ok,
                          "instance_check_deferred": inst_needs_args})
        status = ("IMPLEMENTED" if any(f["satisfies_protocol"] for f in found)
                  else ("UNVERIFIABLE" if unverifiable and not found else "DECLARED_NOT_IMPLEMENTED"))
        out[name] = {**spec, "status": status, "candidates_checked": found,
                     "unverifiable": unverifiable}
    # The four seams answer "is the interface implemented".  This answers a DIFFERENT question --
    # "which of them can be observed at the same time, on the same body" -- and keeps the two from
    # being conflated, since all four seams can be IMPLEMENTED while only two channels can be
    # recorded together.
    out["observation_channels"] = {
        "status": "IMPLEMENTED",
        "recorded_in_one_run_on_one_body": list(
            OBSERVATION_CHANNELS["together"]["recorded_in_one_run"]),
        "absent_by_construction": list(OBSERVATION_CHANNELS["together"]["absent_by_construction"]),
        "n_simultaneous": len(OBSERVATION_CHANNELS["together"]["recorded_in_one_run"]),
        "detail": OBSERVATION_CHANNELS,
    }
    return out


#: Where each seam's implementation is expected.  Kept explicit so the survey knows what to look at
#: instead of scanning the whole tree on every call.
_CANDIDATES = {
    "SensoryDrive": ["engine.embodied.adapters.MechanoReceptorBank"],
    "MotorCommand": ["engine.embodied.adapters.CommandedTripodCPG"],
    "NeuralRecorder": ["engine.embodied.electrode_in_loop.InLoopElectrode"],
    "NeuralStimulus": ["engine.embodied.electrode_in_loop.InLoopStimulus"],
}
