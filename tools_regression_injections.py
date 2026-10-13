#!/usr/bin/env python3
"""INJECTION REGRESSIONS.  Plant each real bug again and prove the check catches it.

WHY THIS FILE IS THE MOST IMPORTANT ONE IN THE REPAIR, and the argument is short: a check that has
never been seen to fail is indistinguishable from a check that cannot fail.  This project's three
worst failures were all SILENT -- a 1000x unit error, a pixel gate applied at the wrong resolution,
and a recording whose fiducial table was empty -- and all three were found by a human noticing a
number afterwards.  So each one is planted again here, on purpose, and the corresponding check is
REQUIRED to fail.  If a check passes with the bug present, the injection reports FAIL.

The three injections, and the check each one must trip:

  1. `unit_1000x`        put `MODEL_FORCE_TO_N` back to 1.0e-3
                         -> `derived_force_unit` must FAIL
  2. `gate_no_scaling`   make the reconstruction gate resolution-independent again
                         -> `gate_resolution_scale` must FAIL
  3. `empty_fiducials`   point the data-health gate at the episode whose recorded table has no
                         fiducials -> that episode must FAIL, and a good episode must still PASS

Each injection also has a CONTROL: with nothing planted, the same check must PASS.  Without the
control, a check hard-wired to FAIL would look like a success.

Nothing is written to the project by an injection: every patch is applied in-process and undone.
Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python tools_regression_injections.py
Exit code 0 means the harness ran.  The verdict is `overall` in the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _inj(name, description, check_name, expect, got_status, got_detail, control_status):
    """A check passed its injection test only if the injected fault changed the verdict."""
    caught = (got_status == expect)
    control_ok = (control_status == "PASS")
    status = "PASS" if (caught and control_ok) else "FAIL"
    return {"injection": name, "description": description, "check": check_name,
            "expected_verdict_with_fault": expect, "verdict_with_fault": got_status,
            "verdict_control_no_fault": control_status,
            "status": status, "detail": got_detail,
            "why": ("PASS = the check caught the planted fault AND passed without it. "
                    "FAIL = the check is blind to this fault, or is hard-wired to fail.")}


def inj_unit_1000x() -> dict:
    """Put the millinewton mistake back and require the unit derivation to catch it."""
    import run_tactile_record as rtr
    import tools_invariants as ti
    control = ti.check_derived_force_unit()["status"]
    saved = rtr.MODEL_FORCE_TO_N
    try:
        rtr.MODEL_FORCE_TO_N = 1.0e-3            # the historical wrong value
        r = ti.check_derived_force_unit()
    finally:
        rtr.MODEL_FORCE_TO_N = saved
    return _inj("unit_1000x",
                "run_tactile_record.MODEL_FORCE_TO_N forced back to 1.0e-3 (the real bug: a "
                "comment claimed the fly's mass is 1.02e-3 kg, making it a thousand times a fly)",
                "tools_invariants.check_derived_force_unit", "FAIL", r["status"],
                {"codes_constant": r["value"]["code_MODEL_FORCE_TO_N"],
                 "derived": r["value"]["derived_model_force_to_N"],
                 "implied_mass_unit": r["value"]["implied_mass_unit"]},
                control)


def inj_gate_no_scaling() -> dict:
    """Make the pixel gate resolution-independent again and require the check to catch it."""
    import run_arena_reconstruct as rar
    import tools_invariants as ti
    control = ti.check_gate_resolution_scale()["status"]
    saved = rar.gate_px_at_rows
    try:
        # the historical behaviour: use the reference-height number at every resolution
        rar.gate_px_at_rows = lambda max_px, rows: float(max_px)
        r = ti.check_gate_resolution_scale()
    finally:
        rar.gate_px_at_rows = saved
    return _inj("gate_no_scaling",
                "gate_px_at_rows replaced by the identity, i.e. a gate tuned at 200 rows applied "
                "unchanged at 800 rows (4x looser in millimetres; admitted grass specks)",
                "tools_invariants.check_gate_resolution_scale", "FAIL", r["status"],
                r["value"], control)


def inj_empty_fiducials() -> dict:
    """The recording whose fiducial table is empty must FAIL, and a good one must still PASS."""
    import tools_episode_health as teh
    bad = HERE / "outputs" / "arena" / "arena_bare_seed0"
    good = HERE / "outputs" / "arena" / "arena_bare_seed3"
    if not bad.is_dir():
        return {"injection": "empty_fiducials", "status": "NOT_RUN",
                "note": f"episode not present: {bad}"}
    rb = teh.health(bad)
    hr = next((c for c in rb["checks"] if c["check"] == "fiducial_hit_rate_3px"), None)
    # The CONTROL here is a different episode rather than the same one un-patched: the fault is
    # IN THE DATA, so it cannot be undone in-process.  A good episode standing in as the control
    # is what makes the verdict meaningful -- it shows the gate is not simply always red.
    rg = teh.health(good) if good.is_dir() else {"status": "NOT_RUN"}
    return _inj("empty_fiducials",
                "the episode recorded while the recorder's blob-area filter was hard-coded for a "
                "200-row frame, so every fiducial was rejected and the px/ table kept only grass",
                "tools_episode_health.health", "FAIL", rb["status"],
                {"episode": bad.name,
                 "hit_rate": (hr or {}).get("value"),
                 "control_episode": good.name, "control_status": rg["status"]},
                rg["status"])


INJECTIONS = (inj_unit_1000x, inj_gate_no_scaling, inj_empty_fiducials)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "regression_injections.json"))
    args = ap.parse_args()

    results = []
    for fn in INJECTIONS:
        try:
            results.append(fn())
        except Exception as exc:                                   # noqa: BLE001
            results.append({"injection": fn.__name__, "status": "NOT_RUN",
                            "note": f"harness error: {type(exc).__name__}: {exc}"})
    overall = ("FAIL" if any(r["status"] == "FAIL" for r in results)
               else ("NOT_RUN" if any(r["status"] == "NOT_RUN" for r in results) else "PASS"))

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"overall": overall, "results": results,
                                "units": ("not applicable: this harness compares VERDICTS, not "
                                          "quantities.  The quantities it triggers on carry their "
                                          "own units in tools_invariants.json"),
                                "provenance": ("in-process monkeypatches of run_tactile_record and "
                                               "run_arena_reconstruct, plus the recorded episode "
                                               "outputs/arena/arena_bare_seed{0,3}"),
                                "convention": ("an injection PASSES only if the planted fault "
                                               "changed the verdict AND the control was green.  "
                                               "Exit code 0 means the harness ran.")},
                               indent=2, sort_keys=True))
    print(f"{'injection':<20}{'status':>9}{'with fault':>12}{'control':>9}   check")
    for r in results:
        print(f"{r['injection']:<20}{r['status']:>9}{r.get('verdict_with_fault', '-'):>12}"
              f"{r.get('verdict_control_no_fault', '-'):>9}   {r.get('check', '')}")
    print(f"\nOVERALL: {overall}  (an injection passes only if the fault was CAUGHT and the "
          f"control was green)")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
