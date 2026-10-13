#!/usr/bin/env python3
"""MAKE THE CONTACT-FORCE DOWNGRADE MECHANICAL, AND MAKE THE LEGACY OUTPUTS CARRY IT.

`PLATFORM_SELFCONSISTENT_PLAN.zh-CN.md` gives three admissible states for the contact-force channel
and forbids a fourth: fix the residual to about 1x (option a), or EXPLICITLY DOWNGRADE it and forbid
physiological-magnitude conclusions from citing it (option b).  The forbidden state is the one this
project was in -- large numbers, unidentified cause, still used as a load -- and it was forbidden in
prose, which is exactly the kind of rule that stops being true.

Two things are checked here, and they are deliberately different in strength:

  S  SOURCE ACKNOWLEDGEMENT.  Every source file that touches the channel must reference
     `engine/embodied/force_channel`.  This is a TEXT SCAN and it proves ACKNOWLEDGEMENT, not
     obedience -- stated plainly because a token check that is mistaken for a semantic one is worse
     than no check.
  O  OUTPUT CONTRACT.  Every recorded tactile episode's `summary.json` must carry the four result
     contract fields, with the DOWNGRADE in `what_this_does_not_establish`.  This one is about the
     artifact a reader actually opens, and it is the half that `tools_result_contract_audit.py` could
     not see: that validator globs only `outputs/*.json`, so all ten episode summaries sat outside it
     entirely.

  I  INJECTION.  A synthetic consumer that reads the channel without the acknowledgement must make
     the source scan FAIL.  Asserted, not assumed -- a scan that cannot fail is decoration.

ACCURACY NOTES.  `truth/normal_force_N` really is in newtons, and one MODEL FORCE UNIT is one
MICRO-NEWTON, so model units = the array x 1e6.  MEASURED residual across the ten episodes: mean
4.29-12.16x body weight, peak 25.26-27.79x, against about 1x for a body standing or walking at
constant height.  The CAUSE IS NOT IDENTIFIED; the actuators are excluded as the dominant cause by
measurement (gain x1000 and force range x130 each moved it only 18 % and 16 %), which leaves
position-servo internal stress as a suspicion.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_force_downgrade_audit.py            # audit only, read-only
    ./venv/bin/python tools_force_downgrade_audit.py --stamp    # annotate legacy outputs in place
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor" / "pylibs"))

#: Tokens that mean "this file touches the contact-force channel".
CHANNEL_TOKENS = ("normal_force", "contact_forces", "contact_force", "tangential_force")

#: Files allowed to mention the channel without carrying the reference, each with a REASON.  Kept
#: short on purpose: a long exempt list is how this kind of check dies.
EXEMPT = {
    "engine/embodied/force_channel.py":
        "this module IS the downgrade; it defines the tokens it would be asked to reference",
    "tools_force_downgrade_audit.py":
        "the auditor itself, which necessarily names the tokens it searches for",
}

CONTRACT_FIELDS = ("units", "provenance", "status", "what_this_does_not_establish")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def scan_sources(root: Path, extra: list[Path] | None = None) -> dict:
    """Every .py under `root` that touches the channel, and whether it acknowledges the downgrade."""
    consumers, unacknowledged = [], []
    # `startswith("venv")`, NOT `== "venv"`: MEASURED, the first version excluded only a part named
    # exactly "venv" and therefore scanned `venv_body/lib/.../flygym/*` as if it were project source,
    # reporting three third-party files as unacknowledged consumers of OUR channel.
    files = sorted(p for p in root.rglob("*.py")
                   if not any(part.startswith("venv") or part in ("vendor", "node_modules")
                              for part in p.parts))
    if extra:
        # DEDUPE BY RESOLVED PATH: the injection writes its file INSIDE the scanned root, so it is
        # already found by the glob; appending it again counted it twice and made the injection look
        # like it had added two consumers.
        seen = {f.resolve() for f in files}
        files = files + [e for e in extra if e.resolve() not in seen]
    for p in files:
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if not any(t in txt for t in CHANNEL_TOKENS):
            continue
        rel = str(p.relative_to(root)) if str(p).startswith(str(root)) else p.name
        # THE TOKEN IS `force_channel.py`, WITH THE EXTENSION, AND THAT IS NOT PEDANTRY.  MEASURED:
        # with the bare token `force_channel`, `tools_invariants.py` matched INCIDENTALLY, through its
        # unrelated function name `check_force_channel_order_of_magnitude` -- so the scan reported an
        # acknowledgement where none existed.  A substring match that a name collision can satisfy is
        # how a check becomes decoration; requiring the module reference makes the match deliberate.
        ack = "force_channel.py" in txt
        consumers.append({"file": rel, "acknowledges_downgrade": ack,
                          "exempt": rel in EXEMPT,
                          "exempt_reason": EXEMPT.get(rel)})
        if not ack and rel not in EXEMPT:
            unacknowledged.append(rel)
    return {"n_consumers": len(consumers), "consumers": consumers,
            "unacknowledged": unacknowledged}


def scan_outputs(root: Path) -> dict:
    """Every tactile episode summary must carry the four contract fields with the downgrade."""
    import importlib
    fc = importlib.import_module("engine.embodied.force_channel")

    rows = []
    for d in sorted((root / "outputs" / "tactile").glob("tactile_*/")):
        s = d / "summary.json"
        if not s.is_file():
            continue
        doc = json.loads(s.read_text())
        missing = [f for f in CONTRACT_FIELDS if f not in doc]
        stated = str(doc.get("what_this_does_not_establish", ""))
        has_downgrade = fc.STATUS in str(doc.get("status", "")) or fc.STATUS in stated
        rows.append({"episode": d.name, "missing_fields": missing,
                     "states_downgrade": has_downgrade,
                     "status": doc.get("status"),
                     "mean_over_weight": (doc.get("measured_force_residual") or {}).get(
                         "mean_over_weight"),
                     "compliant": not missing and has_downgrade})
    return {"episodes": rows, "n": len(rows),
            "n_noncompliant": sum(1 for r in rows if not r["compliant"])}


def stamp(root: Path) -> dict:
    """Annotate the legacy summaries IN PLACE, each with ITS OWN measured residual."""
    import importlib
    fc = importlib.import_module("engine.embodied.force_channel")

    stamped = []
    for d in sorted((root / "outputs" / "tactile").glob("tactile_*/")):
        s = d / "summary.json"
        if not s.is_file():
            continue
        doc = json.loads(s.read_text())
        res = fc.measured_residual(d)
        c = fc.contract(res)
        doc["units"] = c["units"]
        doc["provenance"] = c["provenance"]
        doc["status"] = c["status"]
        doc["what_this_does_not_establish"] = c["what_this_does_not_establish"]
        doc["forbidden_uses"] = c["forbidden_uses"]
        doc["allowed_uses"] = c["allowed_uses"]
        doc["measured_force_residual"] = c["measured_residual"]
        doc["force_channel_propagation"] = c["propagation"]
        doc["contract_stamped_by"] = "tools_force_downgrade_audit.py --stamp"
        s.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
        stamped.append({"episode": d.name, "residual_status": res.get("status"),
                        "mean_over_weight": res.get("mean_over_weight"),
                        "peak_over_weight": res.get("peak_over_weight"),
                        "body_weight_source": res.get("body_weight_source")})
    return {"stamped": stamped, "n": len(stamped)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", action="store_true",
                    help="annotate legacy tactile summaries in place with their own measured residual")
    ap.add_argument("--out", default=str(HERE / "outputs" / "force_downgrade_audit.json"))
    args = ap.parse_args()
    import importlib
    fc = importlib.import_module("engine.embodied.force_channel")

    print(f"contact-force channel status: {fc.STATUS}")
    print(f"required reference: {fc.REQUIRED_REFERENCE}\n")

    stamp_report = None
    if args.stamp:
        print("0. STAMP: annotate legacy outputs in place, each with its own measurement")
        stamp_report = stamp(HERE)
        for r in stamp_report["stamped"]:
            print(f"   {r['episode']:30s} {r['residual_status']:8s} "
                  f"mean {r['mean_over_weight']:.3f}x peak {r['peak_over_weight']:.3f}x"
                  if r["residual_status"] == "MEASURED" else
                  f"   {r['episode']:30s} {r['residual_status']}")

    print("\n1. O: does every recorded tactile episode carry the contract?")
    out = scan_outputs(HERE)
    print(f"   {out['n']} episodes, {out['n_noncompliant']} non-compliant")
    for r in out["episodes"]:
        print(f"   {r['episode']:30s} status={r['status']} missing={r['missing_fields']} "
              f"downgrade={r['states_downgrade']}")
    check("O: every tactile episode summary carries the contract and the downgrade",
          out["n"] > 0 and out["n_noncompliant"] == 0,
          f"{out['n'] - out['n_noncompliant']}/{out['n']} compliant")

    print("\n2. S: does every source that touches the channel acknowledge the downgrade?")
    src = scan_sources(HERE)
    print(f"   {src['n_consumers']} source(s) touch the channel; "
          f"{len(src['unacknowledged'])} do not acknowledge:")
    for f in src["unacknowledged"]:
        print(f"     - {f}")
    check("S: every channel consumer acknowledges the downgrade (or is exempt with a reason)",
          not src["unacknowledged"], f"{len(src['unacknowledged'])} unacknowledged")

    print("\n3. I: INJECTION -- a synthetic consumer must make the scan FAIL")
    with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
        inj = Path(td) / "_injected_force_consumer.py"
        inj.write_text("# a consumer added by the injection, deliberately without the acknowledgement\n"
                       "def total(forces):\n"
                       "    return sum(f['normal_force'] for f in forces)\n")
        s2 = scan_sources(HERE, extra=[inj])
        caught = len(s2["unacknowledged"]) == len(src["unacknowledged"]) + 1
        print(f"   injected consumer detected: {caught} "
              f"(unacknowledged went {len(src['unacknowledged'])} -> {len(s2['unacknowledged'])})")
        check("I: the scan CATCHES an unacknowledged consumer (it can fail)", caught,
              "a scan that cannot fail is decoration")

    n_fail = sum(1 for _n, ok, _d in RESULTS if not ok)
    doc = {
        "overall": "FAIL" if n_fail else "PASS", "status": "FAIL" if n_fail else "PASS",
        "n_checks": len(RESULTS), "n_failed": n_fail,
        "results": [{"check": c, "ok": o, "detail": d} for c, o, d in RESULTS],
        "force_channel_status": fc.STATUS,
        "forbidden_uses": list(fc.FORBIDDEN_USES),
        "allowed_uses": list(fc.ALLOWED_USES),
        "propagation": fc.propagation(),
        "source_scan": src, "output_scan": out, "stamp": stamp_report,
        "units": ("dimensionless multiples of the model's own body weight (10.0485 model units); "
                  "one MODEL FORCE UNIT is 1 MICRO-NEWTON"),
        "provenance": ("engine/embodied/force_channel.py for the declaration and the per-episode "
                       "measurement; the source scan reads .py files under the project and the "
                       "output scan reads outputs/tactile/*/summary.json"),
        "what_this_does_not_establish": (
            "THE CAUSE OF THE RESIDUAL, and it does not establish that anyone OBEYS the downgrade.  "
            "The source half is a TEXT SCAN: it proves a file ACKNOWLEDGES the downgrade, not that "
            "it avoids the forbidden uses -- a token check mistaken for a semantic one is worse than "
            "no check, which is why it is written here rather than left implied.  The output half is "
            "stronger because it reads the artifact.  What IS established: the residual is 4.29-12.16x "
            "body weight on average and 25.26-27.79x at peak across ten recorded episodes; the "
            "actuators are excluded as the dominant cause by measurement; and the mechanoreceptive "
            "drive derived from this channel therefore has a non-physical MAGNITUDE too, so no "
            "'sensed X newtons' claim is supportable anywhere in this project.  No charge-safety or "
            "biological-load claim is made, and nothing here is real-fly data."),
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=2, sort_keys=True, default=float))
    print(f"\nOVERALL: {doc['overall']}  ({len(RESULTS)} checks, {n_fail} failed)\nwrote {dest}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
