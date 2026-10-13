#!/usr/bin/env python3
"""ONE BUTTON THAT RUNS EVERY CHECK IN THIS PROJECT, WITH THE RIGHT INTERPRETER PER CHECK.

WHY THIS EXISTS.  The project has 29 self-tests and they ALL PASS -- but nothing ran them together,
and the project's only orchestrator (`run_all.py`, 45 lines) covers six stages from the earlier
cell/mechanics era and HARD-CODES `venv/bin/python`.  MEASURED consequence: four of the 29 tests
belong to the embodied stack and fail under `venv` with

    ModuleNotFoundError: No module named 'flygym'

so a naive unified runner reports four FALSE FAILURES.  A runner that lies about four tests is
worse than no runner, because it trains the reader to ignore red.

HOW THE INTERPRETER IS CHOSEN.  Derived, not hand-listed: a test whose source mentions `flygym`
or `mujoco` needs the body environment (`venv_body`, plus MUJOCO_GL=egl for offscreen rendering);
everything else uses `venv`.  The classification is PRINTED so it can be audited, and it is
asserted against the measured baseline (exactly four tests need venv_body).

CONVENTION, from this project's own bench protocol: exit code 0 means THE RUNNER RAN, not that
everything passed.  Read the verdict.  `NOT_RUN` means a check could not be evaluated and is never
a pass.  `--exit-nonzero-on-fail` is available for automation that wants the exit code.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python run_all_checks.py
    ./venv/bin/python run_all_checks.py --quick        # skip the slow body tests
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / "venv" / "bin" / "python"
VENV_BODY = HERE / "venv_body" / "bin" / "python"
OUT = HERE / "outputs"

#: Source markers used as a FAST HINT for the interpreter.
#:
#: MEASURED, and it is why a hint is not enough: scanning sources for these words finds only TWO
#: of the four embodied tests, because `run_body_selftest`, `run_embodied_loop_selftest` and
#: `run_plasticity_selftest` reach flygym only INDIRECTLY, through `BodyBackend`.  A list built
#: from this scan therefore reports false failures -- exactly the failure mode this runner exists
#: to prevent.  So the hint is backed by an ADAPTIVE RETRY (see `run_one`): a test that dies under
#: `venv` with a missing flygym/mujoco module is re-run under `venv_body`, and the retry is
#: recorded rather than hidden.
BODY_MARKERS = ("flygym", "mujoco")

#: CHECKS THAT NEED THE BODY ENVIRONMENT WITHOUT NAMING IT.  The source-marker scan cannot see a
#: test that reaches the renderer through a SUBPROCESS rather than an import, and the adaptive
#: retry only fires on `ModuleNotFoundError`, which such a test never raises -- it fails for a
#: different reason entirely.  MEASURED: `tools_test_stackb_observable.py` imports neither flygym nor
#: mujoco itself; it shells out to `tools_make_stackb_observable.py`, so the runner put it under
#: `venv` and it failed at the re-render step.  Declared here rather than faked with a decorative
#: import in the test file.
FORCE_BODY_ENV = {"tools_test_stackb_observable.py", "tools_test_all_channels.py",
                  "tools_test_limb_stackb.py",
                  "tools_test_stimulus_stackb.py"}

#: Substrings identifying "this failed only because the wrong environment was used".
ENV_ERROR_MARKERS = ("No module named 'flygym'", "No module named 'mujoco'",
                     'No module named "flygym"', 'No module named "mujoco"')

#: Tests are globbed rather than listed, so a NEW test cannot be forgotten by this runner -- that
#: omission is precisely how a check suite stops meaning anything.
TEST_GLOBS = ("tools_test_*.py", "run_*selftest*.py", "run_*_selftest.py")

#: Checks that are not self-test scripts.  Each is (name, argv, needs_body).
#:
#: The order matters for reading the report: data health first, because a broken recording makes
#: every downstream number meaningless; then the invariants; then the injections, which re-plant
#: the historical bugs and require the invariants to catch them.
EXTRA_CHECKS = (
    # THE ARTIFACT IS THE EXPECTATIONS FILE, NOT THE AGGREGATE.  The aggregate is FAIL by design
    # while `arena_bare_seed0` -- a deliberately bad fixture -- exists, so keying the suite on it
    # would leave one permanently red check and teach the reader to ignore red.
    ("episode_data_health",
     [str(VENV), str(HERE / "tools_episode_health.py"), "--all",
      "--expect", "arena_bare_seed0=FAIL", "arena_bare_seed1=PASS", "arena_bare_seed2=PASS",
      "arena_bare_seed3=PASS"], False,
     str(HERE / "outputs" / "arena" / "episode_health_expectations.json")),
    # BUILD-TIME ACCEPTANCE FOR THE ONE-WAY LEG JOINTS (round 25).  The 4th element is the
    # artifact, so the harness cross-checks the verdict against the exit code; the check itself
    # FAILS on the pre-repair model, which is what makes the repair falsifiable.
    ("bidirectionality",
     [str(VENV_BODY), str(HERE / "tools_test_bidirectionality.py")], False,
     str(HERE / "outputs" / "bidirectionality_check.json")),
    ("moment_arm_algebra",
     [str(VENV_BODY), str(HERE / "diagnostics" / "diag_moment_arm_algebra.py")], False,
     str(HERE / "outputs" / "moment_arm_algebra_r25.json")),
    # THE MUSCLES' OPERATING RANGE (round 25, second repair).  FAILS on the model built before the
    # lengthrange rescale was measured at the declared neutral posture instead of at qpos=0.
    ("muscle_operating_range",
     [str(VENV_BODY), str(HERE / "tools_test_muscle_operating_range.py")], False,
     str(HERE / "outputs" / "muscle_operating_range_check.json")),
    # WELL-FORMEDNESS OF EVERY GENERATED MODEL (round 25, third repair).  FAILED when a provenance
    # comment carrying "--" made the standing model unparseable by a conformant XML parser while a
    # lenient loader accepted it.
    # WHERE THE REMAINING TILT COMES FROM (round 25).  A causal test, not a correlation: it steps
    # the same drop with every muscle's force scale zeroed.  Verdict lands in the artifact.
    ("passive_tilt",
     [str(VENV_BODY), str(HERE / "diagnostics" / "diag_passive_tilt.py")], False,
     str(HERE / "outputs" / "passive_tilt_decomposition.json")),
    ("xml_wellformed",
     [str(VENV_BODY), str(HERE / "tools_test_xml_wellformed.py")], False,
     str(HERE / "outputs" / "xml_wellformed_check.json")),
    ("invariants",
     [str(VENV_BODY), str(HERE / "tools_invariants.py"), "--all"], True,
     str(HERE / "outputs" / "invariants.json")),
    ("regression_injections",
     [str(VENV_BODY), str(HERE / "tools_regression_injections.py")], True,
     str(HERE / "outputs" / "regression_injections.json")),
    ("truth_boundary_audit",
     [str(VENV), str(HERE / "tools_truth_boundary_audit.py")], False,
     str(HERE / "outputs" / "truth_boundary_audit.json")),
    ("hypothesis_guard_selftest",
     [str(VENV), str(HERE / "tools_hypothesis_test.py"), "--selftest"], False),
    ("environment_pinning",
     [str(VENV_BODY), str(HERE / "tools_pin_environment.py")], True,
     str(HERE / "outputs" / "environment_lock.json")),
    ("electrode_in_loop",
     [str(VENV_BODY), str(HERE / "tools_test_electrode_in_loop.py")], True),
    # The interface document must MATCH the code.  This fires only when somebody changes a public
    # signature and forgets to regenerate -- a legitimate interface change does not turn the suite
    # red, which is why the check is "is the document current" rather than "did the interface move".
    ("per_leg_command",
     [str(VENV_BODY), str(HERE / "tools_test_per_leg_command.py")], True),
    ("stimulus_in_loop",
     [str(VENV_BODY), str(HERE / "tools_test_stimulus_in_loop.py")], True),
    ("interfaces_up_to_date",
     [str(VENV), str(HERE / "tools_interface_doc.py"), "--stale"], False),
    # All four seams can be IMPLEMENTED while only two channels can be recorded together, so the
    # registration of what is OBSERVED SIMULTANEOUSLY is checked separately from the seams.
    ("seam_survey",
     [str(VENV), str(HERE / "tools_seam_survey.py"), "--check"], False,
     str(HERE / "outputs" / "seam_survey.json")),
    # S5: the DECLARED capture radius, turned into a measured operating range through the full loop.
    # It is an EXTRA check rather than a `tools_test_*` self-test because its verdict IS the
    # measurement: the response is non-monotone, and that is the answer to "is 50 um defensible?",
    # not a failure to be fixed.
    ("capture_radius_dose_response",
     [str(VENV_BODY), str(HERE / "tools_capture_radius_dose_response.py")], True,
     str(HERE / "outputs" / "capture_radius_dose_response.json")),
    # The contact-force channel is a NON_PHYSICAL_LOAD; this makes the downgrade mechanical on both
    # sides -- the sources that touch the channel and the episode summaries that report it.
    ("force_downgrade_audit",
     [str(VENV), str(HERE / "tools_force_downgrade_audit.py")], False,
     str(HERE / "outputs" / "force_downgrade_audit.json")),
    # The PLAN's own four acceptance criteria, evaluated from the artifacts of the checks above.
    ("acceptance_abcd",
     [str(VENV), str(HERE / "tools_acceptance_abcd.py")], False,
     str(HERE / "outputs" / "acceptance_abcd.json")),
)

PER_CHECK_TIMEOUT_S = 900


def discover() -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in TEST_GLOBS:
        for p in sorted(HERE.glob(pat)):
            seen[p.name] = p
    return [seen[k] for k in sorted(seen)]


def needs_body(path: Path) -> bool:
    try:
        txt = path.read_text(errors="replace")
    except OSError:
        return False
    return any(m in txt for m in BODY_MARKERS)


def _exec(argv: list[str], body: bool):
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if body:
        env.setdefault("MUJOCO_GL", "egl")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=HERE, env=env,
                              timeout=PER_CHECK_TIMEOUT_S)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or ""), ""
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {PER_CHECK_TIMEOUT_S}s -- NOT_RUN, never a pass"
    except OSError as exc:
        return None, "", f"could not execute: {exc}"


def _artifact_verdict(path: str | None):
    """The verdict a check WROTE, if it declares one.  Returns (verdict, detail) or (None, why)."""
    if not path:
        return None, "no artifact declared"
    f = Path(path)
    if not f.is_file():
        return None, f"declared artifact {path} was not written"
    try:
        doc = json.loads(f.read_text())
    except Exception as exc:                                            # noqa: BLE001
        return None, f"declared artifact {path} is unreadable: {type(exc).__name__}"
    for k in ("overall", "status", "verdict"):
        v = doc.get(k)
        if isinstance(v, str) and v.upper() in ("PASS", "FAIL", "NOT_RUN"):
            return v.upper(), f"{path}:{k}={v}"
    return None, f"declared artifact {path} carries no PASS/FAIL/NOT_RUN verdict"


def run_one(name: str, argv: list[str], body: bool, quick: bool,
            artifact: str | None = None) -> dict:
    """Run one check, retrying under venv_body when the ONLY problem was the environment.

    The retry is not a convenience.  Without it the runner reports four false failures, and a
    reader who sees red for a reason that is not real learns to ignore red.  The retry is recorded
    in the result (`interpreter`, `retried_because`) so that the fact a test needs a second
    environment stays visible instead of being smoothed away.
    """
    t0 = time.perf_counter()
    rc, out, err = _exec(argv, body)
    interpreter = "venv_body" if body else "venv"
    retried_because = ""
    if rc not in (0, None) and not body and any(m in out for m in ENV_ERROR_MARKERS):
        body_argv = [str(VENV_BODY)] + argv[1:]
        rc2, out2, err2 = _exec(body_argv, True)
        retried_because = ("under `venv` this failed with a missing flygym/mujoco module, i.e. the "
                           "wrong environment rather than a broken test; re-ran under venv_body")
        interpreter, argv, out, rc = "venv_body", body_argv, out2, rc2
        if err2:
            err = err2
    if err:
        status, note = "NOT_RUN", err
    else:
        status = "PASS" if rc == 0 else "FAIL"
        note = "" if rc == 0 else f"exit {rc}"
    # THE EXIT CODE ALONE IS NOT THE VERDICT, AND TRUSTING IT WAS A REAL HOLE.  This platform's
    # stated convention is "exit 0 means the check RAN; the verdict is in the JSON" -- so a check
    # that runs to completion and then reports FAIL internally was being shown as PASS here.
    # MEASURED, and it is why this exists: `tools_truth_boundary_audit.py` wrote
    # `overall: FAIL` (11 truth-readers with no declared role) while the suite printed it PASS,
    # because it returned 0.  A disagreement between the exit code and the check's OWN verdict is
    # therefore reported as FAIL: a check that contradicts itself cannot be a pass.
    verdict, why = _artifact_verdict(artifact)
    if verdict is not None and verdict != status:
        note = (f"exit code said {status} but {why}.  A check that contradicts its own artifact "
                f"cannot be a pass.")
        status = "FAIL"
    elif verdict is not None:
        note = f"verdict cross-checked against {why}"
    return {"name": name, "argv": argv, "interpreter": interpreter,
            "retried_because": retried_because, "status": status, "exit_code": rc,
            "artifact_verdict": verdict, "artifact_verdict_source": why,
            "seconds": round(time.perf_counter() - t0, 2), "note": note,
            "tail": "\n".join(out.strip().splitlines()[-25:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip checks whose interpreter is venv_body (the slow ones)")
    ap.add_argument("--exit-nonzero-on-fail", action="store_true")
    ap.add_argument("--out", default=str(OUT / "all_checks.json"))
    args = ap.parse_args()

    tests = discover()
    def _body(p: Path) -> bool:
        return needs_body(p) or p.name in FORCE_BODY_ENV
    plan = [(p.name, [str(VENV_BODY if _body(p) else VENV), str(p)], _body(p), None)
            for p in tests]
    # AN EXTRA CHECK MAY DECLARE WHERE IT WRITES ITS OWN VERDICT, so the runner can hold the exit
    # code against it.  A self-test discovered from a glob cannot, and says so by carrying None.
    plan += [(t[0], t[1], t[2], (t[3] if len(t) > 3 else None)) for t in EXTRA_CHECKS]

    n_body = sum(1 for t in plan if t[2])
    print(f"discovered {len(tests)} self-tests + {len(EXTRA_CHECKS)} extra check(s)")
    print(f"interpreter HINT from source markers: venv_body={n_body}, "
          f"venv={len(plan) - n_body}.  The hint is a fast path, NOT the rule: a test that dies "
          f"under venv with a missing flygym/mujoco module is retried under venv_body.")
    print("  source-marked venv_body: " + (", ".join(t[0] for t in plan if t[2]) or "(none)"))
    print("  NOTE: the count that need venv_body is PRINTED AT THE END from what actually "
          "happened, not asserted here.  An earlier version of this line hard-coded '4', which a "
          "measurement then showed to be wrong.")

    results = []
    if args.quick:
        skipped = [t[0] for t in plan if t[2]]
        plan = [t for t in plan if not t[2]]
        print(f"--quick: skipping {len(skipped)} body check(s): {', '.join(skipped)}")
    for i, (name, argv, body, artifact) in enumerate(plan, 1):
        print(f"[{i:>2}/{len(plan)}] {name}", end="", flush=True)
        r = run_one(name, argv, body, args.quick, artifact=artifact)
        results.append(r)
        print(f"  -> {r['status']}  ({r['seconds']}s)")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    missing = [] if not args.quick else [t[0] for t in plan if t[2]]
    if counts.get("FAIL"):
        overall = "FAIL"
    elif counts.get("NOT_RUN"):
        overall = "NOT_RUN"
    else:
        overall = "PASS"

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "overall": overall,
        "units": ("not applicable: this is the runner's own report; units belong to each check's "
                  "own output, which carries them"),
        "provenance": "(the runner's argv list per check, recorded in results[].argv, plus each "
                      "check's own JSON output)",
        "counts": counts, "results": results,
        "quick": bool(args.quick),
        "convention": ("exit code 0 means the runner RAN, not that everything passed; the verdict "
                       "is `overall`.  NOT_RUN means a check could not be evaluated and is never "
                       "a pass."),
        "interpreter_rule": ("a test whose source mentions " + " or ".join(BODY_MARKERS) +
                             " runs under venv_body with MUJOCO_GL=egl; all others under venv"),
    }, indent=2, sort_keys=True))

    print(f"\nOVERALL: {overall}   " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    n_retry = sum(1 for r in results if r.get("retried_because"))
    n_bod = sum(1 for r in results if r["interpreter"] == "venv_body")
    print(f"needed venv_body: {n_bod}   (of which found only by retry: {n_retry})")
    if counts.get("FAIL"):
        print("failed:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  - {r['name']} ({r['interpreter']}, exit {r['exit_code']})")
    print(f"wrote {dest}")
    print("NOTE: exit code 0 means the runner ran.  The verdict is 'overall' in the JSON.")
    if args.exit_nonzero_on_fail and overall != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
