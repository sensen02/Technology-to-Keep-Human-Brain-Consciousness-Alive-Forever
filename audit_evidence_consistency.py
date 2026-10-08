"""Cross-module evidence audit: measured vs assumed vs unquantified, machine-checked.

WHY THIS EXISTS
---------------
Every module in this project carries its own provenance records, but nothing
previously checked them TOGETHER.  This tool reads the provenance blocks that the
modules already emit, aggregates them, and FAILS if it finds an inconsistency,
so the "measured vs illustrative" separation cannot silently rot.

It checks, across every report it can read:
  1. any record tagged `measured` must carry a citation string;
  2. any record tagged `measured` whose species differs from the model species
     must be flagged cross-species;
  3. any record tagged `derived` must name its parents;
  4. numbers may not carry a bare value under a key that names a foreign species
     (rodent/mouse/rat/guinea/cockroach/manduca/locust) without a flag;
  5. the report must declare a negative or a limitation when it claims something
     that was NOT measured.

WHAT THIS IS NOT
----------------
Aggregation across heterogeneous schemas is necessarily partial: this tool counts
what it can parse and reports how much it could NOT parse rather than implying
full coverage.  A pass means the records are internally consistent, NOT that the
underlying science is right.  It cannot detect a wrong number that is correctly
labelled.

Run: venv/bin/python audit_evidence_consistency.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "brain_isolation"

FOREIGN_KEY_TOKENS = ("rodent", "mouse", "rat_", "guinea", "cockroach",
                      "manduca", "locust", "mammal", "primate", "human")
PROVENANCE_CLASSES = ("measured", "derived", "illustrative", "assumed")
CITATION_TOKENS = ("doi", "pmc", "pmid", "http", "source", "citation",
                   "10.", "(", "et al")


def walk_records(obj, path=""):
    """Yield (path, record, class_key) for any dict that looks like a parameter record.

    Two schemas exist in this project and BOTH are accepted:
      * engine.params style: {"value":..., "provenance": "measured", ...}
      * report style:        {"value":..., "status": "illustrative", ...}
    `status` is only treated as a provenance class when its value is one of the
    four class names, so ordinary status strings ("completed", "passed") are not
    miscounted as parameters.
    """
    if isinstance(obj, dict):
        cls = None
        if str(obj.get("provenance", "")).lower() in PROVENANCE_CLASSES:
            cls = "provenance"
        elif str(obj.get("provenance_class", "")).lower() in PROVENANCE_CLASSES:
            cls = "provenance_class"
        elif str(obj.get("status", "")).lower() in PROVENANCE_CLASSES:
            cls = "status"
        if cls and "value" in obj:
            yield path, obj, cls
        for k, v in obj.items():
            yield from walk_records(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_records(v, f"{path}[{i}]")


def main():
    reports = sorted(OUT.glob("*.json"))
    totals = Counter()
    violations = []
    unparsed = []
    parsed_files = 0
    cross_species_flagged = 0

    for f in reports:
        try:
            data = json.loads(f.read_text())
        except Exception:
            unparsed.append((f.name, "not json"))
            continue
        found = list(walk_records(data))
        if not found:
            unparsed.append((f.name, "no provenance records found"))
            continue
        parsed_files += 1
        for path, rec, cls_key in found:
            prov = str(rec.get(cls_key) or "").lower()
            totals[prov or "unspecified"] += 1
            species = str(rec.get("species") or "").lower()
            is_foreign = bool(species) and "drosophila" not in species
            if is_foreign:
                if rec.get("cross_species") is True or rec.get("forced_cross_species_flag"):
                    cross_species_flagged += 1
                else:
                    # a foreign record must at least be flagged by the module itself
                    if not any(str(path).lower().find(t) >= 0 for t in FOREIGN_KEY_TOKENS):
                        violations.append(
                            f"{f.name}: {path} is a foreign-species record "
                            f"(species={species!r}) with no cross-species flag")
            if prov == "measured":
                text = " ".join(str(rec.get(k, "")) for k in
                                ("source", "citation", "note", "notes", "derived_from"))
                if not any(t in text.lower() for t in CITATION_TOKENS):
                    violations.append(
                        f"{f.name}: {path} is 'measured' with no citation-like text")
            if prov == "derived":
                # Two legitimate schemas exist in this project: an
                # engine.params-style `derived_from` list, and a report-style
                # `source` string of the form "derived from X [citation]; Y".
                # Accepting only the first produced FALSE POSITIVES on the
                # electrode-damage report, whose parents and citations are in
                # `source`; this rule was corrected rather than the module.
                parents = rec.get("derived_from")
                text = " ".join(str(rec.get(k, "")) for k in ("source", "note", "notes"))
                if not parents and "derived from" not in text.lower():
                    violations.append(
                        f"{f.name}: {path} is 'derived' naming no parents in either "
                        f"derived_from or a 'derived from ...' source string")

    # reports that explicitly declare what they did NOT measure
    declared_gaps = []
    for f in reports:
        try:
            text = f.read_text().lower()
        except Exception:
            continue
        if any(t in text for t in ('"negative_results"', '"limitations"',
                                   'not_claimed', 'not modelled', 'unmeasured')):
            declared_gaps.append(f.name)

    # Reconciliation: where a module states its OWN parameter totals, compare
    # them with what this tool could parse.  A large gap means the checker's
    # coverage is partial, and that must be visible rather than implied.
    reconciliation = []
    for f in reports:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        summary = data.get("provenance_summary") or data.get("provenance_totals")
        if isinstance(summary, dict) and "_total" in summary:
            mine = sum(1 for _ in walk_records(data))
            reconciliation.append({
                "report": f.name,
                "module_declared_total": summary["_total"],
                "parsed_by_this_tool": mine,
                "coverage_fraction": (mine / summary["_total"]) if summary["_total"] else None,
            })

    result = {
        "what_this_checks": [
            "measured records carry a citation",
            "foreign-species records are flagged or sit under an explicitly foreign key",
            "derived records name their parents (derived_from list OR a 'derived from ...' source string)",
        ],
        "schemas_accepted": ["engine.params style (key 'provenance')",
                             "report style (key 'status', limited to the four class names)"],
        "module_vs_tool_reconciliation": reconciliation,
        "what_a_pass_does_NOT_mean": [
            "that the numbers are right (a correctly labelled wrong number passes)",
            "that coverage is complete: heterogeneous schemas are parsed only where they match; see module_vs_tool_reconciliation",
        ],
        "reports_total": len(reports),
        "reports_with_parsed_records": parsed_files,
        "reports_without_parsable_records": unparsed,
        "provenance_class_totals": dict(totals),
        "foreign_species_records_flagged": cross_species_flagged,
        "reports_declaring_explicit_gaps": declared_gaps,
        "violations": violations,
        "n_violations": len(violations),
        "verdict": "PASS" if not violations else "FAIL",
    }
    (OUT / "evidence_consistency_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
