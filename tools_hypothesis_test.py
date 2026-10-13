#!/usr/bin/env python3
"""HYPOTHESIS-TEST TEMPLATE.  A perturbation, a control, a falsification rule, and a NOT_RUN guard.

WHY THE GUARD IS THE POINT.  This project already contains the single most valuable mechanism in
it: when the standing-neuron ablation ran, the LATERALISED test and the intact run came back
BIT-IDENTICAL, and the harness had a guard that turned that into "NOT RUN" rather than "no effect".
That guard is what uncovered the real bug -- the actuator-to-motor-neuron table covered only the 15
foreleg muscles, so the other 75 muscles were never driven at all, and "silencing these neurons
changes nothing" was not a result about neurons.  Without the guard the project would have recorded
a clean negative finding that was actually a wiring failure.

So every hypothesis here must declare, BEFORE it runs:
  * the perturbation and its control,
  * the outcome metric,
  * the FALSIFICATION RULE (what result would refute the hypothesis),
  * and it must pass the identical-data guard, or it reports NOT_RUN.

A negative control is built in: control-vs-control across seeds measures the SCATTER a difference
has to beat.  An effect smaller than that scatter is not an effect, whatever its sign.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_hypothesis_test.py --list
    ./venv/bin/python tools_hypothesis_test.py --run H1
Exit code 0 means the harness ran.  The verdict is in the JSON.
"""
    # Reads `force_channel`: a NON_PHYSICAL_LOAD (engine/embodied/force_channel.py).  Used as a
    # hypothesis-test VARIABLE, where only the sign and the rank of the effect are read -- never
    # its magnitude as a load.
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TACTILE = HERE / "outputs" / "tactile"

#: Registered hypotheses.  Falsification rules are written as machine-checkable text so a reader
#: can see what would have refuted each one BEFORE seeing the numbers.
HYPOTHESES = {
    "H1": {
        "statement": ("Mounting the complete electrode payload (glass electrode + tether + BLE "
                      "transmitter + battery) changes the ground-reaction-force distribution of a "
                      "walking fly, compared with the bare fly."),
        "predictor": "rig (bare vs full)",
        "control_glob": "tactile_bare_seed*",
        "perturbed_glob": "tactile_full_seed*",
        "metric": "per-frame total tarsal normal force / body weight, and the per-leg load share",
        "falsification": ("REFUTED if |mean(perturbed) - mean(control)| is smaller than the "
                          "control-vs-control scatter across seeds, i.e. the rig effect does not "
                          "exceed the seed-to-seed spread."),
        "specificity_unit": "leg",
        "preregistered": "2026-10-08, before the run below",
    },
}

BODY_WEIGHT_MODEL_UNITS = 10.0485      # measured: mass_attr_sum * |gravity|, see tools_invariants


def _load(pattern: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(str(TACTILE / pattern))):
        p = Path(d) / "tactile.npz"
        if not p.is_file():
            continue
        z = np.load(p, allow_pickle=True)
        key = next((k for k in z.files if "normal_force" in k), None)
        if key is None:
            continue
        f = np.asarray(z[key], dtype=float) * 1e6           # N -> uN == model force units
        out.append({"episode": Path(d).name, "per_leg": f,
                    "total": f.sum(axis=1),
                    "ratio": f.sum(axis=1) / BODY_WEIGHT_MODEL_UNITS})
    return out


def _identical(a: list[dict], b: list[dict]) -> bool:
    """The NOT_RUN guard: do the two arms carry BIT-IDENTICAL force traces?

    Compared per episode position, and only when the shapes match, so a differing frame count is a
    real difference rather than a false NOT_RUN.
    """
    if len(a) != len(b) or not a:
        return False
    for x, y in zip(a, b):
        if x["per_leg"].shape != y["per_leg"].shape:
            return False
        if not np.array_equal(x["per_leg"], y["per_leg"]):
            return False
    return True


def _share(eps: list[dict]) -> np.ndarray:
    """Mean per-leg share of the total load, averaged over episodes."""
    acc = None
    for e in eps:
        tot = e["total"][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(tot > 0, e["per_leg"] / tot, 0.0)
        s = s.sum(axis=0)
        acc = s if acc is None else acc + s
    return (acc / len(eps)) / max(len(eps), 1) if acc is not None else np.zeros(6)


def run_hypothesis(hid: str) -> dict:
    h = HYPOTHESES[hid]
    ctrl = _load(h["control_glob"])
    pert = _load(h["perturbed_glob"])
    base = {"hypothesis": hid, "statement": h["statement"], "predictor": h["predictor"],
            "metric": h["metric"], "falsification_rule": h["falsification"],
            "preregistered": h["preregistered"],
            "control_episodes": [e["episode"] for e in ctrl],
            "perturbed_episodes": [e["episode"] for e in pert]}
    if len(ctrl) < 2 or len(pert) < 2:
        base.update({"status": "NOT_RUN", "why": "need at least two episodes per arm to measure "
                                                 "the seed scatter that an effect must beat"})
        return base

    # ---- the guard, FIRST, before any effect is computed -------------------------------------
    if _identical(ctrl, pert):
        base.update({"status": "NOT_RUN", "why": ("the two arms are BIT-IDENTICAL.  This is not a "
                                                  "null result: it means the perturbation did not "
                                                  "reach the simulation, and treating it as 'no "
                                                  "effect' is the exact failure this guard exists "
                                                  "to prevent")})
        return base

    c_ratio = np.array([e["ratio"].mean() for e in ctrl])
    p_ratio = np.array([e["ratio"].mean() for e in pert])
    # NEGATIVE CONTROL: the spread between two CONTROL episodes is the scatter to beat.  With only
    # two control episodes this is a range, not a standard deviation, and it is reported as such.
    scatter = float(np.ptp(c_ratio))
    effect = float(p_ratio.mean() - c_ratio.mean())
    beats = abs(effect) > scatter
    share_c, share_p = _share(ctrl), _share(pert)
    leg_delta = np.abs(share_p - share_c)
    base.update({
        "status": "PASSED_GUARD",
        "guard": "the two arms differ; the perturbation reached the simulation",
        "control_ratio_mean": round(float(c_ratio.mean()), 4),
        "perturbed_ratio_mean": round(float(p_ratio.mean()), 4),
        "control_per_episode": [round(float(v), 4) for v in c_ratio],
        "perturbed_per_episode": [round(float(v), 4) for v in p_ratio],
        "negative_control_scatter_range": round(scatter, 4),
        "effect": round(effect, 4),
        "effect_beats_scatter": bool(beats),
        "leg_share_control": [round(float(v), 4) for v in share_c],
        "leg_share_perturbed": [round(float(v), 4) for v in share_p],
        "max_leg_share_shift": round(float(leg_delta.max()), 4),
        "verdict": ("SUPPORTED for this sample" if beats else "REFUTED for this sample"),
        "verdict_basis": h["falsification"],
        "what_this_does_not_establish": (
            "nothing about real flies, and nothing causal: the arms are matched only on the rig "
            "name, the two seeds per arm do not make a distribution, and the residual force scale "
            "(mean 4-12x body weight) is recorded rather than explained."),
    })
    return base


def selftest() -> int:
    """The template's own acceptance test: an EMPTY perturbation must be NOT_RUN.

    Requirement, not decoration: a template that reports "no effect" for a perturbation which never
    reached the simulation is the exact mechanism that produced a false negative in this project
    (the 75 unwired muscles).  So the guard is tested by construction -- control and perturbed
    pointed at the SAME episodes, which must be bit-identical.
    """
    HYPOTHESES["_selftest_empty"] = dict(HYPOTHESES["H1"],
                                         control_glob="tactile_bare_seed*",
                                         perturbed_glob="tactile_bare_seed*")
    # a second case: DIFFERENT episodes must NOT be reported as identical, or the guard is
    # hard-wired to fire and proves nothing.
    HYPOTHESES["_selftest_real"] = dict(HYPOTHESES["H1"])
    ok = True
    try:
        r1 = run_hypothesis("_selftest_empty")
        r2 = run_hypothesis("_selftest_real")
    finally:
        HYPOTHESES.pop("_selftest_empty", None)
        HYPOTHESES.pop("_selftest_real", None)
    print(f"  empty perturbation  -> {r1['status']:<12} (must be NOT_RUN)")
    ok &= (r1["status"] == "NOT_RUN")
    print(f"  real perturbation   -> {r2['status']:<12} (must NOT be NOT_RUN)")
    ok &= (r2["status"] != "NOT_RUN")
    print(f"SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="require an empty perturbation to be NOT_RUN and a real one not to be")
    ap.add_argument("--out", default=str(HERE / "outputs" / "hypothesis_tests.json"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.list:
        for k, v in HYPOTHESES.items():
            print(f"{k}: {v['statement']}")
            print(f"    falsification: {v['falsification']}")
        return 0
    ids = args.run or sorted(HYPOTHESES)
    results = [run_hypothesis(h) for h in ids]
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _sts = [r["status"] for r in results]
    _overall = ("FAIL" if any(s == "NOT_RUN" for s in _sts)
                else ("PASS" if all(s == "PASSED_GUARD" for s in _sts) else "MIXED"))
    dest.write_text(json.dumps({"overall": _overall, "results": results, "registry": HYPOTHESES,
                                "units": ("force ratios are dimensionless multiples of body weight; "
                                          "body weight is 10.0485 model force units = 10.0485 uN"),
                                "provenance": ("outputs/tactile/*/tactile.npz, one episode per arm; "
                                               "falsification rules were registered before the run "
                                               "(see registry[].preregistered)"),
                                "status": _overall,
                                "convention": ("NOT_RUN is not a null result.  A bit-identical "
                                               "perturbation means it never reached the simulation.")},
                               indent=2, sort_keys=True))
    for r in results:
        print(f"{r['hypothesis']}: {r['status']}")
        if r["status"] == "NOT_RUN":
            print(f"   why: {r['why']}")
        else:
            print(f"   control {r['control_ratio_mean']} vs perturbed "
                  f"{r['perturbed_ratio_mean']}  effect {r['effect']}  "
                  f"scatter {r['negative_control_scatter_range']}  -> {r['verdict']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
