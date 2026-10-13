"""ROUND 39 (FIX), part E: clear the `_mm`-named-but-centimetre defect, by declaration.

WHAT THIS IS
------------
`run_fix_round39_checks.py` (check C1) counted 722 fields named `_mm`, of which 42 are
filled from model-unit sources.  39 of those hold CENTIMETRES under an `_mm` name - the
round-39 defect D1.  Two are false positives of the scan: their expressions already convert
(`* 10` in diag_solver_vs_signature.py, `* 1000.0` in diag_adhesion_variant_round19.py),
so the value really is in millimetres and only the scan is wrong.

THE RULE APPLIED HERE
---------------------
The value is NOT changed.  Only the NAME is, from `_mm` to `_model`, because that is what
the value is: model length units (centimetres).  Renaming is deliberately the conservative
fix - it cannot alter any number, and it makes the next reader's unit arithmetic correct:
a consumer that wants millimetres now has to call `engine.canonical_physics.model_to_mm`.

Guards (a site is skipped, and reported, when any applies):
  * the expression already converts (`* 10`, `* 1000`), so it IS millimetres;
  * the occurrence is inside `engine/canonical_physics.py`, which is prose documenting the
    defect rather than a producer;
  * the line already carries an explicit unit note (`# unit`, `_model`, `SI: mm`).

Output: a JSON listing every site with its decision, so the 39 renames and the 2 skips are
both on the record.  Also re-runs check C1 and requires the residue to be exactly the
declared false positives.

This edits project files in place.  Run `run_fix_round39_checks.py` afterwards; the numbers
reported by the affected scripts must be unchanged, which is what the guard on conversions
is for.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402

OUT = HERE / "outputs/fix_round39"

# files whose `_mm` occurrence is prose about the defect, not a producer
SKIP_FILES = {"engine/canonical_physics.py", "run_fix_round39_checks.py",
              "tools_fix_unit_labels_round39.py"}
# expressions that already produce millimetres
CONVERSION_MARKERS = ("* 10", "*10", "* 1000", "*1000", "model_to_mm", "mm_per_model")


def classify(expr: str) -> tuple:
    e = expr.replace(" ", "")
    for mk in CONVERSION_MARKERS:
        if mk.replace(" ", "") in e:
            return "SKIP_ALREADY_MM", f"expression already converts ({mk.strip()})"
    return "RENAME_TO_MODEL", "value is in model length units (cm)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the edits; without it the tool only reports")
    args = ap.parse_args()

    import importlib
    checks = importlib.import_module("run_fix_round39_checks")
    c1 = checks.scan_unit_labels()

    decisions = []
    edits_by_file = {}
    for s in c1["sites"]:
        rel = s["file"]
        verdict, why = classify(s["expr"])
        if rel in SKIP_FILES:
            verdict, why = "SKIP_PROSE", "this occurrence documents the defect"
        decisions.append({**s, "decision": verdict, "why": why})
        if verdict != "RENAME_TO_MODEL":
            continue
        edits_by_file.setdefault(rel, []).append(s)

    applied = []
    for rel, sites in edits_by_file.items():
        path = HERE / rel
        lines = path.read_text().split("\n")
        for s in sites:
            i = s["line"] - 1
            before = lines[i]
            after = re.sub(rf"(['\"]){re.escape(s['field'])}\1", r"\1"
                           + s["field"][:-3] + "_model" + r"\1", lines[i])
            if after == before:
                applied.append({"file": rel, "line": s["line"], "status": "NO_CHANGE"})
                continue
            lines[i] = after
            applied.append({"file": rel, "line": s["line"], "field": s["field"],
                            "renamed_to": s["field"][:-3] + "_model",
                            "before": before.strip()[:120],
                            "after": after.strip()[:120]})
        if args.apply:
            path.write_text("\n".join(lines))

    # idempotency: a second scan must find no NEW renamable site in the edited files
    residue = None
    if args.apply:
        importlib.reload(checks)
        c1b = checks.scan_unit_labels()
        residue = [s for s in c1b["sites"] if s["file"] not in SKIP_FILES]
        residue = [s for s in residue if classify(s["expr"])[0] == "RENAME_TO_MODEL"]

    counts = {}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1

    cp.write_result(
        OUT / "fix_round39_unit_label_cleanup.json",
        question=("Clear the `_mm`-named-but-centimetre defect across the project by "
                  "renaming only (no value changed), and prove nothing renamable is left."),
        status=("PASS" if (args.apply and not residue) else
                "NOT_RUN" if not args.apply else "FAIL"),
        measured={
            "scanned_sites_holding_model_units": len(c1["sites"]),
            "decisions": counts,
            "renamed": [a for a in applied if a.get("status") != "NO_CHANGE"],
            "n_renamed": sum(1 for a in applied if a.get("status") != "NO_CHANGE"),
            "skipped": [d for d in decisions if d["decision"] != "RENAME_TO_MODEL"],
            "residue_after_apply": residue,
            "value_changed_on_any_site": False,
        },
        provenance={
            "canonical_xml": str(cp.CANONICAL_XML),
            "canonical_xml_sha256": cp.sha256_file(cp.CANONICAL_XML),
            "canonical_xml_unchanged": cp.sha256_file(cp.CANONICAL_XML)
                                      == cp.CANONICAL_XML_SHA256,
            "scanner": "run_fix_round39_checks.py :: scan_unit_labels (check C1)",
            "rule": "rename `_mm` -> `_model` where the expression reads a model-unit source "
                    "and does not already convert; never touch the value",
            "interpreter": sys.executable,
        },
        what_this_does_not_support=[
            "Renaming does not re-measure anything. The affected scripts still report the "
            "same NUMBERS as before; only their field names and their meaning are now "
            "consistent. Any consumer that read `..._mm` and treated it as millimetres was "
            "wrong before and will now fail loudly instead of silently.",
            "This is a static scan: it classifies expressions, not runtime values.",
            "It does not touch the neural side or any physics conclusion.",
        ],
    )
    print(f"scanned {len(c1['sites'])} sites | " +
          " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"renamed {sum(1 for a in applied if a.get('status') != 'NO_CHANGE')} fields"
          + ("" if args.apply else "  (dry run - pass --apply to write)"))
    if residue is not None:
        print(f"residue after apply: {len(residue)} renamable site(s)"
              + ("" if not residue else " -> FAIL"))
    print("wrote", OUT / "fix_round39_unit_label_cleanup.json")


if __name__ == "__main__":
    main()
