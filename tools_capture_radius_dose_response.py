#!/usr/bin/env python3
"""S5: IS THE DECLARED 50 um CAPTURE RADIUS DEFENSIBLE?  MEASURE THE DOSE-RESPONSE.

WHAT WAS MISSING.  `InLoopStimulus.CAPTURE_RADIUS_UM = 50.0` is DECLARED, and the only thing standing
behind it was a GEOMETRIC count: how many recorded somata lie within that radius of the contact.  A
count is not a response.  It says how many neurons the rule would touch; it says nothing about what
the loop does when you touch them.  So the parameter could not be defended, only asserted.

WHAT THIS MEASURES INSTEAD.  The full chain, once per radius, through the loop whose motor-neuron
firing IS the muscle activation:

    for each capture radius:  run(same amplitude, same repetition rate)  ->  joint excursion
                              against the same run with NO stimulus

That turns a declared number into a measured operating range: below some radius the effect is
EXACTLY zero (nothing is recruited), and above it the loop responds.  It also measures the SHAPE of
the response, and the shape is the finding -- see the guards below, one of which the data FAILS.

THE GUARDS, because a curve on its own is not evidence:

  G1  DETERMINISM.  The unstimulated baseline is run twice and must be bit-identical.  If the loop is
      not reproducible, no point of the curve means anything.
  G2  NO RECRUITMENT, NO EFFECT -- and this one is INJECTED rather than asserted.  Where the radius
      recruits zero neurons the change must be EXACTLY 0.0 rad.  To show that guard is not vacuous,
      the same zero-recruitment stimulus is then given a HAND-FORCED weight on one neuron: if the
      effect stays zero the test proves nothing, so the injection REQUIRES the effect to become
      non-zero once weights exist.  That distinguishes "no neuron was recruited" from "the pulse
      never reached the membrane".
  G3  MONOTONICITY IS TESTED AND IS EXPECTED TO FAIL.  A larger radius is meant to be a strictly
      stronger stimulus, so the response should not fall as the radius grows.  MEASURED: it does.
      The tool must REPORT that rather than smooth it, because it is the honest answer to "is 50 um
      defensible?" -- non-monotonicity means no radius can be preferred on behavioural grounds.

The charge delivered is CONSTANT across radii (the recruitment weights are normalised to sum to 1),
so what changes between points is only the SPATIAL DISTRIBUTION of the current -- which is exactly
the quantity this module declares as not established.  A non-monotone curve is that declaration,
measured.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_capture_radius_dose_response.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")


class _Got(Exception):
    """Raised to abort a throwaway run once the value we wanted has been captured."""
#: The radii swept.  Bracketed around the DECLARED 50 um, and reaching low enough to find the
#: recruitment threshold, which the geometric sweep alone put between 12 and 20 um.
RADII_UM = (5.0, 10.0, 15.0, 18.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0)
DURATION_S = 0.5
AMPLITUDE_NA = 1000.0
REPETITION_HZ = 20.0


def _stim(r, *, amplitude_nA=AMPLITUDE_NA, repetition_rate_Hz=REPETITION_HZ):
    def factory(ctx):
        import run_six_leg_neural_loop as loop
        st, _elec = loop.make_in_loop_stimulus(
            ctx, channel_index=0, amplitude_nA=amplitude_nA, capture_radius_um=r,
            repetition_rate_Hz=repetition_rate_Hz, mode="recruiting")
        return st
    return factory


def _vec(summary):
    return np.asarray([summary["excursion"][k] for k in LEGS], dtype=float)


def measure(duration_s=DURATION_S, radii=RADII_UM, verbose=True) -> dict:
    import run_six_leg_neural_loop as loop

    # ---- G1: the baseline must be reproducible, or no point of the curve means anything
    base = loop.run(lesion_leg=None, duration_s=duration_s, verbose=False)
    base_again = loop.run(lesion_leg=None, duration_s=duration_s, verbose=False)
    bs, bs2 = _vec(base), _vec(base_again)
    deterministic = bool(np.array_equal(bs, bs2))

    rows = []
    for r in radii:
        out = loop.run(lesion_leg=None, duration_s=duration_s, verbose=False, stimulus=_stim(r))
        d = _vec(out) - bs
        st = out["stimulus"]
        rows.append({
            "capture_radius_um": float(r),
            "n_recruited": int(st["n_recruited_neurons"]),
            "max_abs_delta_rad": float(np.abs(d).max()),
            "delta_per_leg_rad": {leg: float(v) for leg, v in zip(LEGS, d)},
            "worst_leg": LEGS[int(np.argmax(np.abs(d)))],
            "charge_nA_ms": float(st["total_charge_nA_ms"]),
            "n_intervals_with_current": int(st["n_intervals_with_current"]),
        })
        if verbose:
            print(f"   r={r:>6.1f} um  recruited {rows[-1]['n_recruited']:>4d}  "
                  f"max|d| {rows[-1]['max_abs_delta_rad']:.5f} rad  "
                  f"({rows[-1]['worst_leg']})  charge {rows[-1]['charge_nA_ms']:.1f} nA*ms")

    # ---- G2: where nothing is recruited, nothing may happen
    zero_rows = [x for x in rows if x["n_recruited"] == 0]
    zero_ok = all(x["max_abs_delta_rad"] == 0.0 for x in zero_rows) if zero_rows else None

    # ---- G2's INJECTION: force weights onto the SAME zero-recruitment stimulus.  If the effect
    # stays zero, then G2 was passing for the wrong reason (e.g. the pulse is never sampled) and the
    # whole curve is meaningless.  The injection must therefore make the effect NON-ZERO.
    injection = None
    if zero_rows:
        r0 = min(x["capture_radius_um"] for x in zero_rows)
        # WHICH WEIGHTS TO FORCE, AND WHY IT IS NOT AN ARBITRARY NEURON.  The first version of this
        # injection gave the weight to `local_index[0]`, and the effect stayed EXACTLY zero -- which
        # proves nothing about recruitment, because an arbitrary neuron need not be a motor neuron
        # of any leg and driving it cannot move a muscle.  MEASURED, and it is the difference
        # between a real control and a decorative one.  So the forced weights are the ones a LARGER
        # radius WOULD have recruited: the geometric radius stays at r0 (still recruiting nothing),
        # and only the weight vector is transplanted.  Now a non-zero effect can only come from the
        # recruitment pattern, which is exactly what G2 claims is missing.
        ref_radius = max([x["capture_radius_um"] for x in rows if x["n_recruited"] > 0], default=None)
        forced_target = None
        src_weights = None
        if ref_radius is not None:
            probe = {}

            def _grab(ctx):
                st_ref, _e = loop.make_in_loop_stimulus(
                    ctx, channel_index=0, amplitude_nA=AMPLITUDE_NA,
                    capture_radius_um=ref_radius, repetition_rate_Hz=REPETITION_HZ,
                    mode="recruiting")
                probe["w"] = np.asarray(st_ref.weights, dtype=float).copy()
                probe["n"] = int(st_ref.n_recruited)
                raise _Got(probe)          # abort the run: we only wanted the weight vector

            try:
                loop.run(lesion_leg=None, duration_s=0.02, verbose=False, stimulus=_grab)
            except _Got:
                pass
            src_weights = probe.get("w")
            forced_target = probe.get("n")

        def forced(ctx):
            st, _elec = loop.make_in_loop_stimulus(
                ctx, channel_index=0, amplitude_nA=AMPLITUDE_NA, capture_radius_um=r0,
                repetition_rate_Hz=REPETITION_HZ, mode="recruiting")
            if src_weights is not None:
                st.weights = np.asarray(src_weights, dtype=float)
                st.n_recruited = int(forced_target or 0)
            return st

        out = loop.run(lesion_leg=None, duration_s=duration_s, verbose=False, stimulus=forced)
        dmax = float(np.abs(_vec(out) - bs).max())
        injection = {
            "radius_um": float(r0),
            "geometric_recruitment_at_this_radius": int(
                next(x["n_recruited"] for x in zero_rows if x["capture_radius_um"] == r0)),
            "weights_transplanted_from_radius_um": ref_radius,
            "weights_transplanted_neurons": forced_target,
            "max_abs_delta_rad_after_transplant": dmax,
            "guard_is_not_vacuous": bool(dmax > 0.0),
            "why": ("the geometric radius still recruits NOTHING here; only the weight vector a "
                    "larger radius would have produced was transplanted.  A non-zero effect proves "
                    "the zero effect at this radius came from the absent recruitment and not from "
                    "the pulse failing to reach the membrane.  NOTE: forcing an ARBITRARY single "
                    "neuron (the first version of this injection) gave EXACTLY zero, because an "
                    "arbitrary neuron need not be a motor neuron of any leg -- which is itself the "
                    "reason this control has to transplant a recruitment PATTERN rather than one "
                    "index."),
        }

    # ---- G3: monotonicity, tested and reported even though it fails
    viol = []
    pos = [(i, x) for i, x in enumerate(rows) if x["max_abs_delta_rad"] > 0.0]
    for (_, a), (_, b) in zip(pos, pos[1:]):
        if b["max_abs_delta_rad"] < a["max_abs_delta_rad"]:
            viol.append({"from_radius_um": a["capture_radius_um"],
                         "to_radius_um": b["capture_radius_um"],
                         "from_rad": a["max_abs_delta_rad"], "to_rad": b["max_abs_delta_rad"],
                         "drop_frac": 1.0 - b["max_abs_delta_rad"] / a["max_abs_delta_rad"]})

    # ---- G4: the EFFECTIVE threshold, bracketed by the data
    last_zero = max([x["capture_radius_um"] for x in rows if x["max_abs_delta_rad"] == 0.0],
                    default=None)
    first_nonzero = min([x["capture_radius_um"] for x in rows if x["max_abs_delta_rad"] > 0.0],
                        default=None)
    charges = [x["charge_nA_ms"] for x in rows if x["n_recruited"] > 0]
    charge_constant = bool(charges) and bool(np.allclose(charges, charges[0], rtol=1e-9))

    return {
        "rows": rows, "deterministic": deterministic,
        "g1_determinism": {"baseline_bit_identical_on_rerun": deterministic},
        "g2_no_recruitment_no_effect": {"radius_with_zero_recruitment": [x["capture_radius_um"]
                                                                        for x in zero_rows],
                                        "all_exactly_zero": zero_ok},
        "g2_injection": injection,
        "g3_monotonic": {"violations": viol, "is_monotone": not viol,
                         "verdict": ("MONOTONE" if not viol else "MEASURED_NON_MONOTONE")},
        "g4_effective_threshold": {"largest_radius_with_exactly_zero_effect_um": last_zero,
                                   "smallest_radius_with_nonzero_effect_um": first_nonzero,
                                   "declared_radius_um": 50.0,
                                   "bracket_um": [last_zero, first_nonzero]},
        "charge_is_constant_across_radii": charge_constant,
        "charge_values_nA_ms": sorted({round(c, 6) for c in charges}),
        "charge_note": ("evaluated over the radii that recruit anything; a zero-recruitment radius "
                        "delivers exactly 0.  The weights are normalised to sum to 1, so a larger "
                        "radius SPREADS the same charge over more neurons rather than delivering "
                        "more of it -- which is why what this curve varies is the spatial pattern"),
        "charge_zero_at_radii": [x["capture_radius_um"] for x in rows if x["n_recruited"] == 0],
        "settings": {"duration_s": duration_s, "amplitude_nA": AMPLITUDE_NA,
                     "repetition_rate_Hz": REPETITION_HZ, "mode": "recruiting",
                     "radii_um": list(radii)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "capture_radius_dose_response.json"))
    args = ap.parse_args()

    print("capture-radius dose-response through the FULL loop (muscle-driving)")
    m = measure()

    g2 = m["g2_no_recruitment_no_effect"]
    inj = m["g2_injection"]
    g3 = m["g3_monotonic"]
    g4 = m["g4_effective_threshold"]
    print(f"\n   G1 deterministic baseline: {m['deterministic']}")
    print(f"   G2 zero recruitment at radii {g2['radius_with_zero_recruitment']} -> all exactly "
          f"zero: {g2['all_exactly_zero']}")
    if inj:
        print(f"      injection: geom. radius still {inj['radius_um']:.0f} um (recruits "
              f"{inj['geometric_recruitment_at_this_radius']}), weights transplanted from "
              f"{inj['weights_transplanted_from_radius_um']:.0f} um "
              f"({inj['weights_transplanted_neurons']} neurons) -> max|d| "
              f"{inj['max_abs_delta_rad_after_transplant']:.5f} rad "
              f"(guard not vacuous: {inj['guard_is_not_vacuous']})")
    print(f"   G3 monotone: {g3['is_monotone']} -- {len(g3['violations'])} violation(s)")
    for v in g3["violations"]:
        print(f"      {v['from_radius_um']:.0f} -> {v['to_radius_um']:.0f} um: "
              f"{v['from_rad']:.5f} -> {v['to_rad']:.5f} rad (drops {100*v['drop_frac']:.0f} %)")
    print(f"   G4 effect appears between {g4['largest_radius_with_exactly_zero_effect_um']} and "
          f"{g4['smallest_radius_with_nonzero_effect_um']} um; DECLARED radius "
          f"{g4['declared_radius_um']} um")
    print(f"   charge constant across radii: {m['charge_is_constant_across_radii']} "
          f"({m['charge_values_nA_ms']} nA*ms)")

    n_fail = 0
    if not m["deterministic"]:
        n_fail += 1
    if g2["all_exactly_zero"] is False:
        n_fail += 1
    if inj and not inj["guard_is_not_vacuous"]:
        n_fail += 1
    doc = {
        "overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
        "n_failed": n_fail,
        **m,
        "units": ("um for radii, rad for joint excursion, nA*ms for charge, counts for neurons"),
        "provenance": ("run_six_leg_neural_loop.run(stimulus=...) with the stimulus built from the "
                       "loop's own context; the same amplitude, repetition rate and duration at "
                       "every radius, against one unstimulated baseline"),
        "what_this_does_not_establish": (
            "that 50 um is the right radius, or that any of this describes a real fly.  It measures "
            "how THIS loop responds to a DECLARED geometric recruitment rule, and the response is "
            "NON-MONOTONE in that rule -- which means no radius can be preferred on behavioural "
            "grounds, and the declared 50 um remains unde-fended rather than confirmed.  The access "
            "map has soma voxel positions and nothing else: no morphology, no neurite, no soma "
            "radius.  Because the recruitment weights are normalised to sum to 1, the charge "
            "delivered is IDENTICAL at every radius, so what the curve varies is only the SPATIAL "
            "DISTRIBUTION of the current -- the quantity this module already declares as not "
            "established.  The non-monotone curve is that declaration, measured.  Nothing here is a "
            "charge-safety claim, and the fly still does not stand or walk under stimulation."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nOVERALL: {doc['overall']}\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
