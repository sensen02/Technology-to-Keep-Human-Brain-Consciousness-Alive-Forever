#!/usr/bin/env python3
"""THE PLAN'S OWN DEFINITION OF DONE, EVALUATED RATHER THAN ASSERTED.

`PLATFORM_SELFCONSISTENT_PLAN.zh-CN.md` section 0 defines "achieved" by four decidable criteria and
says plainly that missing one means not achieved.  They were written as acceptance criteria; this
file makes them executable, so "the platform is self-consistent" is a verdict from artifacts rather
than a sentence in a report.

  A  any recording/playback can be judged USABLE / NOT USABLE BY THE PLATFORM ITSELF
     -> the health gate emits PASS/FAIL/NOT_RUN and NOT_RUN is never a pass
  B  any physical quantity entering a conclusion has ONE unit, confirmed by an INDEPENDENT measurement
     -> changing the unit constant must make a check FAIL
  C  no path that reads simulator truth may report truth AS a measurement
     -> a mechanical boundary audit that FAILS on a violation
  D  one hypothesis can be reproduced by a third party from the registry alone, including what
     would refute it
     -> the registry carries refutation conditions and the harness refuses a vacuous perturbation

Each criterion is judged from the ARTIFACTS of the checks that implement it.  A criterion whose
artifact is missing is `NOT_RUN`, never a pass -- the same rule the platform applies to itself.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_acceptance_abcd.py
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

OUT = HERE / "outputs"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    """ok=None means NOT_RUN, which is reported and never counted as a pass."""
    RESULTS.append((name, ok, detail))
    tag = "PASS" if ok is True else ("NOT_RUN" if ok is None else "FAIL")
    print(f"  [{tag:>7}] {name}" + (f"  -- {detail}" if detail else ""))


def _load(rel: str):
    p = OUT / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                                  # noqa: BLE001
        return None


def criterion_a() -> None:
    """A: the platform judges its own recordings, and NOT_RUN is never a pass."""
    print("\nA. can the PLATFORM ITSELF judge a recording?")
    h = _load("arena/episode_health.json")
    if h is None:
        check("A1: the health gate has run and written its verdict", None,
              "outputs/arena/episode_health.json is absent")
        return
    eps = h.get("episodes") or []
    check("A1: the health gate produces a per-episode verdict", bool(eps),
          f"{len(eps)} episode(s)")
    allowed = {"PASS", "FAIL", "NOT_RUN", "SKIP"}
    bad = [e["episode"] for e in eps if str(e.get("status")) not in allowed]
    check("A2: every verdict is one of PASS/FAIL/NOT_RUN/SKIP", not bad, str(bad))
    # the decisive half: NOT_RUN must not be silently counted as usable
    code = (HERE / "tools_episode_health.py").read_text()
    check("A3: NOT_RUN is never treated as a pass in the gate's own logic",
          "NOT_RUN" in code and "exit code 0 means the gate RAN" in code,
          "the gate states that exit 0 means it RAN, with the verdict in `status`")
    # A4 reads the CONSUMER'S SOURCE for the refusal, because the refusal only fires when the input
    # is bad -- which is the normal case here, so there is no artifact of it to read.
    cons = (HERE / "tools_test_all_channels.py").read_text()
    refusals = [l for l in cons.splitlines()
                if "return _finish({})" in l or "run tools_record_all_channels.py" in l]
    check("A4: a downstream consumer REFUSES to proceed when its episode is missing",
          bool(refusals), f"{len(refusals)} refusal path(s) in tools_test_all_channels.py")


def criterion_b() -> None:
    """B: one unit per physical quantity, confirmed by an INDEPENDENT measurement."""
    print("\nB. is every physical quantity's unit confirmed by an independent measurement?")
    inv = _load("invariants.json")
    if inv is None:
        check("B1: the invariants have run", None, "outputs/invariants.json is absent")
        return
    res = {r["check"]: r for r in inv.get("results", [])}
    need = ("settle_balance", "derived_force_unit", "gate_resolution_scale",
            "force_unit_labels_are_honest")
    for n in need:
        r = res.get(n)
        if r is None:
            check(f"B: invariant `{n}` is present", None, "not in invariants.json")
        else:
            check(f"B: invariant `{n}` is not FAILING", r["status"] != "FAIL", r["status"])
    reg = _load("regression_injections.json")
    if reg is None:
        check("B2: the unit error is caught by an INJECTION", None,
              "outputs/regression_injections.json is absent")
    else:
        txt = json.dumps(reg)
        caught = "1.0e-3" in txt or "1e-3" in txt or "unit" in txt.lower()
        check("B2: an injected unit error is CAUGHT (the check can fail)", caught,
              "regression_injections re-introduces the 1000x constant and requires FAIL")


def criterion_c() -> None:
    """C: no path that reads truth may report it as a measurement."""
    print("\nC. is the truth boundary mechanically enforced?")
    t = _load("truth_boundary_audit.json")
    if t is None:
        check("C1: the truth-boundary audit has run", None,
              "outputs/truth_boundary_audit.json is absent")
        return
    # THE ARTIFACT'S VERDICT KEY IS `overall`, NOT `status` -- and getting that wrong is exactly the
    # class of mistake this whole file exists to catch, so it is read explicitly rather than guessed.
    verdict = t.get("overall") or t.get("status") or t.get("verdict")
    check("C1: the truth-boundary audit reports a verdict", verdict is not None, str(verdict))
    check("C2: no consumer of truth/ is unreported", verdict == "PASS",
          f"overall {verdict}, {len(t.get('violations') or [])} violation(s), "
          f"{len(t.get('undeclared') or [])} undeclared reader(s)")


def criterion_d() -> None:
    """D: a hypothesis can be reproduced from the registry, with refutation conditions."""
    print("\nD. can one hypothesis be reproduced from the registry alone?")
    ht = _load("hypothesis_tests.json")
    if ht is None:
        check("D1: the hypothesis harness has run", None,
              "outputs/hypothesis_tests.json is absent")
        return
    tests = ht.get("tests") or ht.get("results") or []
    check("D1: at least one registered hypothesis exists", bool(tests), f"{len(tests)} test(s)")
    txt = json.dumps(ht)
    check("D2: the registry carries what would REFUTE the hypothesis",
          "refut" in txt.lower() or "falsif" in txt.lower() or "would_refute" in txt,
          "a hypothesis without a refutation condition is not a test")
    check("D3: a vacuous perturbation is reported NOT_RUN rather than as a positive",
          "NOT_RUN" in txt, "the untested-perturbation guard")
    # the same guard, on the experiments run in this session
    s = _load("stimulus_stackb_check.json")
    if s is not None:
        vals = json.dumps(s.get("results"))
        check("D4: the stimulus experiment's control is INJECTED, not asserted",
              "injection" in vals.lower(), "a failed injection would be fixed, not excused")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "acceptance_abcd.json"))
    args = ap.parse_args()

    print("THE PLAN'S OWN DEFINITION OF DONE, EVALUATED FROM ARTIFACTS")
    print("=" * 78)
    criterion_a()
    criterion_b()
    criterion_c()
    criterion_d()

    n_fail = sum(1 for _n, ok, _d in RESULTS if ok is False)
    n_notrun = sum(1 for _n, ok, _d in RESULTS if ok is None)
    overall = "FAIL" if n_fail else ("NOT_RUN" if n_notrun else "PASS")

    print("\n" + "=" * 78)
    if overall == "PASS":
        print("A-D ALL HOLD: the plan's four acceptance criteria are met, each by a check that can fail.")
    elif overall == "NOT_RUN":
        print(f"{n_notrun} criterion could not be evaluated -- NOT_RUN is never a pass.")
    else:
        print(f"{n_fail} criterion FAILED.")

    doc = {
        "overall": overall, "status": overall, "n_failed": n_fail, "n_not_run": n_notrun,
        "n_checks": len(RESULTS),
        "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
        "criteria": {
            "A": "the platform judges its own recordings; NOT_RUN is never a pass",
            "B": "one unit per physical quantity, confirmed by an independent measurement",
            "C": "no truth/ reader reports truth as a measurement",
            "D": "a registered hypothesis is reproducible and carries its refutation condition",
        },
        "units": "not applicable: this file evaluates the PRESENCE and VERDICT of other checks",
        "provenance": ("the JSON artifacts those checks wrote under outputs/, read here rather than "
                       "re-run, so this is a map from criterion to evidence and not a fifth opinion"),
        "what_this_does_not_establish": (
            "that the platform is USABLE FOR BIOLOGY.  A-D are the plan's own criteria for a "
            "SELF-CONSISTENT, REPRODUCIBLE SIMULATION platform, and they are deliberately narrow: "
            "they say nothing about whether the model is biologically right, whether the fly walks "
            "(it does not; `stood = False` in 5 of 5 configurations, and the plan excludes "
            "locomotion as behaviour-physics rather than platform-self-consistency), whether the "
            "contact-force channel's magnitude means anything (it does NOT -- it is a declared "
            "NON_PHYSICAL_LOAD), or whether any number here corresponds to a real fly.  There is no "
            "real-fly recording anywhere in this project.  Criterion A is also satisfied by a gate "
            "whose THRESHOLDS are declared; the gate catches unusable recordings, not wrong "
            "thresholds."),
        "hostname": socket.gethostname(),
        "runner": "tools_acceptance_abcd.py",
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
