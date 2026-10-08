"""Unit-contract tests for the receptor layer (item 1 of the approved plan).

WHY THIS EXISTS
---------------
The receptor models return an EQUIVALENT DRIVE in mV (the R_m*I convention that
``engine.neural.LIFNetwork`` consumes), NOT nanoamperes.  The conductance tiers
(``neural_cond``, ``neural_hist``) take nanoamperes.  Nothing in the repository
currently mixes the two -- I grepped every consumer: only ``run_virtual_env.py``
(LIF) and the receptor selftest itself -- but the approved embodied loop will
feed receptors into the conductance tier, so the contract must be machine-checked
BEFORE that happens rather than discovered afterwards.

WHAT THESE TESTS ESTABLISH, AND WHAT THEY DO NOT
------------------------------------------------
They establish that the drive unit is documented and distinguishable, that a
conversion cannot happen without a stated factor and provenance, and that the
existing LIF path is unchanged.  They do NOT establish any biological
transduction gain: none is measured for Drosophila, which is exactly why the
conversion refuses to have a default.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine.receptors import (DRIVE_UNIT, ConversionRefused, MechanoReceptor,  # noqa: E402
                              NO_MEASURED_TRANSDUCTION_GAIN, ReceptorBank, to_nA)

R = []


def check(name, ok, detail=""):
    R.append((name, bool(ok), str(detail)[:200]))
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""))
    return bool(ok)


def main():
    # 1. the unit is named and it is not nanoamperes -------------------------
    check("unit: DRIVE_UNIT names mV and explicitly is not nA",
          "mV" in DRIVE_UNIT and "nA" not in DRIVE_UNIT.replace("R_m*I", ""),
          DRIVE_UNIT)

    # 2. a conversion without a factor is REFUSED ---------------------------
    drive = np.array([1.0, 2.0, 3.0])
    refused_factor = 0
    for bad in (None, 0, -1.0, float("nan"), float("inf")):
        try:
            to_nA(drive, bad, provenance="anything long enough")
        except (ConversionRefused, TypeError, ValueError):
            refused_factor += 1
    check("conversion: every missing/zero/negative/non-finite factor is refused",
          refused_factor == 5, f"{refused_factor}/5 refused")

    # 3. a conversion without provenance is REFUSED -------------------------
    refused_prov = 0
    for bad in (None, "", "   ", "short"):
        try:
            to_nA(drive, 1.0, provenance=bad)
        except ConversionRefused:
            refused_prov += 1
    check("conversion: a missing or trivial provenance string is refused",
          refused_prov == 4, f"{refused_prov}/4 refused")

    # 4. the refusal message says WHY there is no default -------------------
    try:
        to_nA(drive, 0.0, provenance="a stated source, long enough")
        msg = ""
    except ConversionRefused as exc:
        msg = str(exc)
    check("conversion: the refusal states that no measured gain exists",
          "no measured Drosophila transduction gain" in msg,
          msg[:90])

    # 5. a valid conversion is linear and returns the requested unit --------
    out = to_nA(drive, 2.5, provenance="unit-test factor, illustrative only")
    check("conversion: a valid factor converts linearly",
          np.allclose(out, drive * 2.5), out.tolist())

    # 6. non-finite drives are refused, not propagated ----------------------
    try:
        to_nA(np.array([1.0, np.nan]), 1.0, provenance="a stated source, long enough")
        ok = False
    except ConversionRefused:
        ok = True
    check("conversion: a non-finite drive is refused rather than propagated", ok, "")

    # 7. the bank exposes both paths and they differ by exactly the factor --
    # NOTE: current()/current_nA() are STATEFUL -- each call advances the receptor
    # state, so comparing two calls on ONE bank compares two different steps.  An
    # earlier version of this test did that and measured a factor of 5.4 instead of
    # 3.0.  Use two identical banks and compare their FIRST step.
    n = 5
    def make_bank():
        b = ReceptorBank(n, dt_ms=1.0, seed=0)
        b.attach("mechanosensory", np.array([1, 2, 3]), MechanoReceptor,
                 gain_mV=14.0, k_gate=6.0, dG=4.0)
        return b
    stim = {"mechanosensory": 0.5}
    drive_mV = make_bank().current(stim)                       # first step
    conv = make_bank().current_nA(stim, 3.0, provenance="illustrative factor for the test")
    check("bank: on identical first steps, current_nA is exactly current x factor",
          np.allclose(conv, drive_mV * 3.0) and np.abs(drive_mV).max() > 0
          and not np.allclose(conv, drive_mV),
          f"max drive {np.abs(drive_mV).max():.4f} -> max nA {np.abs(conv).max():.4f}, "
          f"ratio {float(np.abs(conv).max()/np.abs(drive_mV).max()):.3f}")
    # and record the statefulness itself, since it silently changes repeated calls
    b3 = make_bank()
    a1 = b3.current(stim); a2 = b3.current(stim)
    check("bank: current() is stateful, so repeated calls with the same stimulus differ "
          "(documented rather than assumed)",
          not np.allclose(a1, a2), f"step1 max {np.abs(a1).max():.4f}, step2 max {np.abs(a2).max():.4f}")
    try:
        make_bank().current_nA(stim)
        no_factor_ok = False
    except TypeError:
        no_factor_ok = True                                  # required positional argument
    except ConversionRefused:
        no_factor_ok = True
    check("bank: current_nA refuses to run without a factor", no_factor_ok, "")

    # 8. the LIF convention is unchanged (no silent change to existing callers)
    from engine.neural import LIFNetwork, NetworkParams
    net = LIFNetwork(n, NetworkParams(), dt_ms=1.0, seed=0)
    net.set_connectivity(__import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix((n, n)))
    before = net.v.copy()
    net.step(i_ext=drive_mV)
    check("regression: the LIF tier still consumes current() directly, unchanged",
          not np.allclose(before, net.v),
          f"v moved from {before[1]:.3f} to {net.v[1]:.3f} mV")

    # 9. the no-default rule is documented where a reader will look ---------
    check("documentation: the no-default rationale is a named, importable string",
          "no measured" in NO_MEASURED_TRANSDUCTION_GAIN,
          NO_MEASURED_TRANSDUCTION_GAIN[:70])

    n_pass = sum(1 for _, ok, _ in R if ok)
    print(f"\n{n_pass}/{len(R)} checks passed")
    import json
    out_dir = os.path.join(HERE, "outputs", "embodied_body")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "receptor_unit_selftest.json"), "w") as fh:
        json.dump({"what_this_establishes": [
                       "the receptor output unit is named and is not nanoamperes",
                       "converting a drive to nA requires an explicit factor AND provenance",
                       "the existing LIF consumption path is unchanged"],
                   "what_it_does_not_establish": [
                       "any biological transduction gain: none is measured for Drosophila",
                       "that any particular conversion factor is correct"],
                   "n_checks": len(R), "n_passed": n_pass,
                   "passed": bool(n_pass == len(R)),
                   "results": [{"name": a, "ok": b, "detail": c} for a, b, c in R]},
                  fh, indent=2)
    return 0 if n_pass == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
