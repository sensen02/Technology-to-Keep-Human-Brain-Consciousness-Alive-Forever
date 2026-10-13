#!/usr/bin/env python3
"""RESULT-CONTRACT VALIDATOR.  Does an output say what it is, and what it is NOT?

WHY THIS EXISTS.  The most expensive mistakes in this project were not wrong numbers; they were
numbers whose SCOPE was unstated.  Two examples, both real:

  * a force of "11,400x body weight" was quoted for weeks before anyone noticed it was a unit error,
    because the output did not carry the unit derivation it rested on;
  * marker-visibility percentages were cited as measurements of what the cameras can see, when the
    tool computes them by projecting the SIMULATOR's own marker positions -- a caveat that existed
    nowhere in its output.

Both would have cost one line to prevent.  So a contract-bearing output must carry:

    units                          what every number is in
    provenance                     where the numbers came from
    status                         PASS / FAIL / NOT_RUN  (NOT_RUN is never a pass)
    what_this_does_not_establish   the scope, stated by whoever produced the number

REQUIRED SET vs BACKLOG.  This validator is deliberately not applied retroactively to every JSON in
`outputs/`: most were written before the contract existed, and turning forty legacy files red would
create a permanent failure that trains the reader to ignore the check.  Instead:

  * CONTRACT_BEARING (below) MUST comply, and a violation is a FAIL;
  * everything else is counted and listed as BACKLOG, so the gap is measured rather than hidden.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_result_contract_audit.py
Exit code 0 means the audit ran.  The verdict is `overall` in the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

REQUIRED_FIELDS = ("units", "provenance", "status", "what_this_does_not_establish")

#: Outputs produced by the checks added in this repair.  These are the ones a reader will use to
#: decide whether to trust the platform, so they are the ones that must be self-describing.
CONTRACT_BEARING = {
    # subpaths allowed: the health gate writes beside the episodes it judges, which is where a
    # reader looking at an episode will find it
    "arena/episode_health.json": "recording data-health gate",
    "invariants.json": "physics and geometry invariants",
    "regression_injections.json": "the three planted-bug regressions",
    "truth_boundary_audit.json": "which code may read the simulator's state",
    "environment_lock.json": "environment and external-asset pinning",
    "hypothesis_tests.json": "registered hypothesis tests",
    "all_checks.json": "the unified runner's own report",
    "result_contract_audit.json": "this validator's own report",
}

#: Field names that carry the same MEANING under a different name, so a file is not failed for
#: naming its scope honestly in its own words.  Listed explicitly, because a fuzzy match here would
#: defeat the purpose of the check.
EQUIVALENTS = {
    "units": ("units", "units_note", "model_units_note", "thresholds"),
    "provenance": ("provenance", "inputs_read", "blob_source", "camera_pose_source", "references",
                   "thresholds", "registry", "roles", "scan_method", "interpreter_rule",
                   "environments", "requirements", "external_assets"),
    "status": ("status", "overall", "conclusion", "offline_reproducible"),
    "what_this_does_not_establish": ("what_this_does_not_establish", "what_they_do_not_establish",
                                     "caveat", "convention", "note"),
}


def check_file(name: str) -> dict:
    # `name` may be a subpath relative to outputs/
    p = OUT / name
    if not p.is_file():
        return {"file": name, "status": "NOT_RUN", "why": "not present"}
    try:
        doc = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return {"file": name, "status": "FAIL", "why": f"not valid JSON: {exc}"}
    if not isinstance(doc, dict):
        return {"file": name, "status": "FAIL", "why": "top level is not an object"}
    found, missing = {}, []
    for field in REQUIRED_FIELDS:
        hit = next((k for k in EQUIVALENTS[field] if k in doc), None)
        if hit is None:
            missing.append(field)
        else:
            found[field] = hit
    return {"file": name, "status": "PASS" if not missing else "FAIL",
            "required_satisfied_by": found, "missing": missing}


def audit() -> dict:
    results = [check_file(n) for n in sorted(CONTRACT_BEARING)]
    # the backlog: every other JSON in outputs/, counted so the gap is visible
    # MEASURED GAP THIS CLOSES: the glob used to be `OUT.glob("*.json")`, so every recorded episode's
    # `summary.json` -- the artifact a reader actually opens -- sat outside this validator entirely.
    # The ten tactile summaries were in that blind spot, and a downgraded channel was being reported
    # from them in absolute micro-newtons and megapascals.  The backlog now counts the whole tree, so
    # the gap is VISIBLE rather than absent.  Episode summaries specifically are made REQUIRED (not
    # backlog) by `tools_force_downgrade_audit.py`, which is where the force-channel contract lives.
    all_json = {str(p.relative_to(OUT)) for p in OUT.rglob("*.json")}
    backlog = sorted(all_json - set(CONTRACT_BEARING))
    violations = [r for r in results if r["status"] == "FAIL"]
    not_run = [r for r in results if r["status"] == "NOT_RUN"]
    overall = "FAIL" if violations else ("NOT_RUN" if not_run else "PASS")
    return {
        "overall": overall,
        "contract_bearing": results,
        "n_contract_bearing": len(results),
        "n_violations": len(violations),
        "backlog_not_yet_contract_bearing": backlog,
        "n_backlog": len(backlog),
        "required_fields": list(REQUIRED_FIELDS),
        "policy": ("CONTRACT_BEARING outputs must carry all four fields; a violation is a FAIL.  "
                   "Legacy outputs are counted as BACKLOG rather than failed, because forty "
                   "permanently-red files train the reader to ignore the check."),
        "units": "not applicable: this file checks the PRESENCE of scope declarations, not values",
        "provenance": "outputs/*.json read by this script; see contract_bearing[].required_satisfied_by",
        "what_this_does_not_establish": (
            "that a declared field is TRUE.  This validator checks that an output STATES its units, "
            "provenance, status and scope; it cannot check that any of them is correct.  That is "
            "what the invariants and the injection regressions are for."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "result_contract_audit.json"))
    args = ap.parse_args()
    res = audit()
    Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True))
    print(f"contract-bearing outputs checked : {res['n_contract_bearing']}")
    for r in res["contract_bearing"]:
        print(f"  {r['status']:<8}{r['file']}{'  missing ' + str(r['missing']) if r.get('missing') else ''}")
    print(f"violations                       : {res['n_violations']}")
    print(f"backlog (legacy json, not failed) : {res['n_backlog']}")
    print(f"\nOVERALL: {res['overall']}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
