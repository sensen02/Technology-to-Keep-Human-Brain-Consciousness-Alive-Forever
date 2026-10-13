"""THE CONTACT-FORCE CHANNEL IS A NON-PHYSICAL LOAD.  ONE DECLARATION, ENFORCED.

WHY THIS MODULE EXISTS.  `PLATFORM_SELFCONSISTENT_PLAN.zh-CN.md` names three admissible states for
the platform's contact-force channel and forbids a fourth:

    (a) find the cause and fix the residual to about 1x body weight;  or
    (b) EXPLICITLY DOWNGRADE the channel to a non-physical load, and forbid any physiological-
        magnitude conclusion from citing it;

and the state that is NOT allowed is the one this project was in: the numbers are large, the cause is
not identified, and they are still used as if they were a load.

MEASURED, and this is the basis for choosing (b) over (a): summing the six legs' normal forces during
the recorded episodes gives a mean of 4.38-12.16x the fly's body weight and a peak of 25.29-26.14x,
where a body standing or walking at constant height should show about 1x.  The ACTUATORS ARE
EXCLUDED as the dominant cause, by measurement rather than by argument: scaling the actuator gain by
1000 and the force range by 130 each moved the residual by only 18 % and 16 %.  What remains is
suspected to be position-servo internal stress against rigid contacts, and that is NOT established --
softening the contacts would change every already-recorded episode and would require re-recording
and re-registering all of them.

SO (b) IS CHOSEN, AND IT IS MADE MECHANICAL RATHER THAN DECLARED.  A downgrade written in a note is a
statement; a property of this platform is only established by something that can FAIL.  Two things
enforce it:

  * `contract()` produces the four result-contract fields for any output derived from this channel,
    with THIS module's measured numbers, so a stamped episode carries its own evidence rather than
    boilerplate;
  * `tools_force_downgrade_audit.py` scans every file that touches the channel and FAILS on any that
    does not carry the downgrade.

THE DOWNGRADE PROPAGATES, AND THAT IS THE PART EASIEST TO MISS.  `engine/embodied/adapters.py`'s
`MechanoReceptorBank` -- the `SensoryDrive` seam, body -> brain -- takes its input FROM THIS CHANNEL.
So the mechanoreceptive drive's MAGNITUDE is not physical either: only its STRUCTURE and its
DEPENDENCE on body state are.  A conclusion of the form "the body sensed X newtons" is not
supportable anywhere in this project; "the drive rose when the leg loaded" is.

UNITS NOTE, AND IT WAS VERIFIED RATHER THAN ASSUMED.  `truth/normal_force_N` really IS in newtons,
and one MODEL FORCE UNIT is 1 MICRO-NEWTON, because this model's length unit is millimetres and its
MASS unit is grams: ``1.02431e-3 g * 9810 mm/s^2 = 10.0485`` model force units of body weight.
So model units = the array times 1e6.  MEASURED on `tactile_bare_seed0`: the raw array sums to
4.350693e-05 per row, which is 43.5069 uN against a 10.0485 uN body weight, i.e. 4.33x -- the same
number `tools_invariants` reports.  Reading the array as model units gives 4.3e-06x, and converting
with 1e3 (millinewtons) instead of 1e6 is the exact 1000x error this project has already made once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[2]

__all__ = ["STATUS", "FORBIDDEN_USES", "ALLOWED_USES", "REQUIRED_REFERENCE", "measured_residual",
           "contract", "propagation"]

#: The single status token.  Consumers are audited for referencing this module.
STATUS = "NON_PHYSICAL_LOAD"

#: What may NOT be done with this channel.
FORBIDDEN_USES = (
    "quoting an absolute force from this channel as a biological or physical load",
    "calling it a tactile intensity, or ranking episodes by its magnitude",
    "any physiological-magnitude conclusion that cites it, directly or through the sensory drive",
    "reporting the sensory drive's current in nanoamperes as a receptor current",
)

#: R26 RETRACTION: the R25 contact-only attribution of stack-B tilt was invalid.
#: Setting muscle gainprm/biasprm column 0 to zero did NOT disable muscle force.
#: A separate model-build regression retained seven active right-foreleg equality locks.
#: Verified 0.5 s ablation (diagnostics/diag_lock_ablation_r26.py): locked tilt 46.00 deg,
#: unlocked tilt 9.62 deg with muscles enabled; with ACTUATION truly disabled and force
#: verified zero at every step, locked/unlocked tilts are 46.20/18.36 deg respectively.
#: Neither this experiment nor the stack-A force-channel downgrade establishes a contact
#: defect as the sole cause of stack-B standing failure. The unlocked model still collapses.
#: The NON_PHYSICAL_LOAD status below remains grounded in its separate force measurements.

#: What the channel remains good for.  Stated so the downgrade is not read as "unusable": the
#: channel is a usable SIGNAL and an unusable MAGNITUDE.
ALLOWED_USES = (
    "relative and structural comparisons: which leg loaded first, whether a leg loaded at all",
    "driving the mechanoreceptive seam as a SIGNAL, because its dependence on body state is real",
    "order-of-magnitude sanity checks, which is what tools_invariants does",
    "as a hypothesis-test variable where only the SIGN or RANK of the effect is used",
)

#: The literal every consumer must carry, so the audit is a text scan rather than a judgement call.
REQUIRED_REFERENCE = "force_channel"

#: The model's own weight in MODEL FORCE UNITS, for the stack A episodes that do not carry it.
#: MEASURED AND LEDGERED, not invented here: `engine/embodied/scene.py` writes
#: `led.record("model.weight_total", 10.0485, "uN", MEASURED_LOCAL, ...)`, and the arithmetic
#: checks out -- 1.02431e-3 g * 9810 mm/s^2 = 10.0485 model force units.  It is named as a
#: DECLARED FALLBACK so that a reader can see which episodes measured their own yardstick and which
#: borrowed this one.
STACK_A_MODEL_WEIGHT_MODEL_UNITS = 10.0485


def measured_residual(episode_dir: Path | str) -> dict[str, Any]:
    """Recompute THIS episode's residual from its own array, so no number here is copied.

    Returns None-valued fields rather than raising when the array is absent, because an episode
    without the channel is not an error -- it simply has nothing to downgrade.
    """
    import numpy as np

    d = Path(episode_dir)
    npz = d / "tactile.npz"
    if not npz.is_file():
        cands = sorted(d.glob("*.npz"))
        if not cands:
            return {"status": "NOT_RUN", "why": f"no npz under {d}"}
        npz = cands[0]
    z = np.load(npz, allow_pickle=True)
    key = next((k for k in z.files if "normal_force" in k), None)
    if key is None:
        return {"status": "NOT_RUN", "why": f"no normal-force array in {npz.name}"}
    # THE ARRAY IS IN NEWTONS; ONE MODEL FORCE UNIT IS 1 MICRO-NEWTON, so model units = N * 1e6.
    arr = np.asarray(z[key], dtype=float) * 1e6
    tot = arr.sum(axis=1)
    bw = None
    for name in ("body_weight_model_units", "body_weight", "weight_model_units"):
        if name in z.files:
            bw = float(np.asarray(z[name]).ravel()[0])
            break
    source = "the episode's own array"
    if bw is None:
        try:
            import run_tactile_record as rtr
            bw = float(getattr(rtr, "_LAST_BODY_WEIGHT_MODEL_UNITS", 0.0)) or None
            if bw:
                source = "run_tactile_record._LAST_BODY_WEIGHT_MODEL_UNITS"
        except Exception:                                              # noqa: BLE001
            bw = None
    if not bw:
        bw = STACK_A_MODEL_WEIGHT_MODEL_UNITS
        source = ("DECLARED FALLBACK: scene.py's ledgered model.weight_total (MEASURED_LOCAL), "
                  "1.02431e-3 g * 9810 mm/s^2")
    return {"status": "MEASURED", "episode": d.name, "array_key": key,
            "mean_over_weight": float(tot.mean() / bw),
            "peak_over_weight": float(tot.max() / bw),
            "n_samples": int(tot.size), "body_weight_model_units": float(bw),
            "body_weight_source": source,
            "expected_for_a_standing_or_walking_body": 1.0}


def propagation() -> dict:
    """WHERE THE DOWNGRADE GOES NEXT.  Written down because the easy mistake is to stop at the array."""
    return {
        "direct_consumers_must_carry_the_downgrade": True,
        "sensory_drive": (
            "engine/embodied/adapters.py's MechanoReceptorBank -- the SensoryDrive seam, body -> "
            "brain -- takes its INPUT from this channel.  Its MAGNITUDE is therefore not physical "
            "either; only its structure and its dependence on body state are"),
        "consequence": (
            "no conclusion of the form 'the body sensed X newtons' is supportable anywhere in this "
            "project; 'the drive rose when the leg loaded' is"),
        "also_downgraded": [
            "any per-leg current derived from the contact drive is a SIGNAL, not a receptor current "
            "in nanoamperes",
        ],
    }


def contract(residual: dict | None = None) -> dict:
    """The four contract fields for any output derived from the contact-force channel.

    `residual` is THIS output's own measurement (from `measured_residual`), so the statement carries
    the episode's numbers instead of a generic disclaimer.  Passing None is allowed and is reported,
    because an output that has not measured its own residual cannot claim the numbers.
    """
    r = residual or {}
    has = r.get("status") == "MEASURED"
    return {
        "units": ("MODEL FORCE UNITS, where ONE MODEL FORCE UNIT IS 1 MICRO-NEWTON (the model's "
                  "length unit is millimetres and its mass unit is grams).  "
                  "`truth/normal_force_N` is in NEWTONS and was verified to be, so model units = the "
                  "array x 1e6; the ratios below are dimensionless multiples of the model's own "
                  "body weight (10.0485 model units)."),
        "provenance": ("MuJoCo contact forces summed over the six legs in the simulator; the residual "
                       "below is recomputed per episode by "
                       "`engine/embodied/force_channel.measured_residual`, not copied."),
        "status": STATUS,
        "what_this_does_not_establish": (
            "THAT ANY FORCE HERE IS A LOAD.  The channel is a NON-PHYSICAL one: its summed normal "
            "force is "
            + (f"{r['mean_over_weight']:.2f}x body weight on average and {r['peak_over_weight']:.2f}x "
               f"at peak over {r['n_samples']} samples"
               if has else "NOT MEASURED FOR THIS OUTPUT")
            + ", against about 1x for a body standing or walking at constant height.  The cause is "
              "NOT identified.  The actuators are EXCLUDED as the dominant cause by measurement "
              "(scaling the actuator gain by 1000 and the force range by 130 each moved the residual "
              "by only 18 % and 16 %), which leaves position-servo internal stress against rigid "
              "contacts as a SUSPICION, not a finding.  Therefore: no absolute force may be quoted "
              "as a biological or physical load, no episode may be ranked by this channel's "
              "magnitude, and the mechanoreceptive drive derived from it has a non-physical "
              "MAGNITUDE too -- only its structure and its dependence on body state are usable.  "
              "Chosen route: OPTION (b) of `PLATFORM_SELFCONSISTENT_PLAN.zh-CN.md` section 3/P2, "
              "explicit downgrade, enforced by `tools_force_downgrade_audit.py`."),
        "forbidden_uses": list(FORBIDDEN_USES),
        "allowed_uses": list(ALLOWED_USES),
        "measured_residual": r if has else None,
        "propagation": propagation(),
    }


if __name__ == "__main__":                                             # pragma: no cover
    print(json.dumps({"status": STATUS, "forbidden": list(FORBIDDEN_USES),
                      "required_reference": REQUIRED_REFERENCE,
                      "propagation": propagation()}, indent=2, ensure_ascii=False))
