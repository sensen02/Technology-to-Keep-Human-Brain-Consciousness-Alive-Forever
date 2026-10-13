#!/usr/bin/env python3
"""GUARD: every model this repository generates must be well-formed XML.

WHY.  Several tools here stamp a provenance comment into the model they generate.  XML forbids
"--" inside a comment, and a comment carrying an ordinary English phrase like
"arm -- sign included -- the reference's" makes the WHOLE DOCUMENT unparseable.  MEASURED:
`outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` failed under `xml.etree` with
"not well-formed (invalid token): line 15, column 36" while `_load_mjcf`'s lenient reader accepted
it -- so every tool in the pipeline that used a lenient loader worked, and everything that parses
XML properly (including `tools_repair_muscle_lengthrange.py`) broke.  A model that half the
toolchain cannot read is a trap, so it is checked rather than trusted.

The obvious repair, sanitising the whole string with `replace("--", "-")`, is wrong for the same
reason it is tempting: it also rewrites the `<!--` and `-->` delimiters.  That mistake was made and
caught by this guard within the same session; `tools_xml_comment.py` is the single place that gets
it right.

CHECKS, both of which can FAIL.
    A. non-vacuity: at least 3 generated XML files exist under outputs/;
    B. every one of them parses with `xml.etree.ElementTree` (a strictly conformant parser), and
       any failure is reported with the parser's own message and position.

SCOPE.  Well-formedness only.  It says nothing about whether a model is valid MJCF, physically
sensible, or able to stand, and it is not a substitute for compiling the model.  Every number is
simulator output; no real-fly recording exists in this project.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "outputs" / "xml_wellformed_check.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(HERE / "outputs"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-files", type=int, default=3)
    a = ap.parse_args(argv)

    root = Path(a.root)
    files = sorted(p for p in root.rglob("*.xml") if p.is_file())
    bad, ok = [], []
    for p in files:
        try:
            ET.parse(p)
            ok.append(str(p.relative_to(HERE)))
        except ET.ParseError as e:
            bad.append({"file": str(p.relative_to(HERE)), "error": str(e)})
        except Exception as e:                                     # noqa: BLE001
            bad.append({"file": str(p.relative_to(HERE)), "error": f"{type(e).__name__}: {e}"})

    check_A = len(files) >= a.min_files
    check_B = len(bad) == 0
    print(f"A) generated XML files found under {root.name}/: {len(files)} "
          f"(need >= {a.min_files}) -- {'PASS' if check_A else 'FAIL'}")
    print(f"B) parse with xml.etree.ElementTree: {len(ok)} well-formed, {len(bad)} malformed "
          f"-- {'PASS' if check_B else 'FAIL'}")
    for b in bad:
        print(f"     MALFORMED {b['file']}: {b['error']}")
    okall = check_A and check_B
    verdict = ("EVERY_GENERATED_MODEL_IS_WELL_FORMED_XML" if okall
               else "A_GENERATED_MODEL_IS_NOT_WELL_FORMED")
    print(f"\nVERDICT: {verdict}")

    out = {
        "overall": "PASS" if okall else "FAIL", "status": "PASS" if okall else "FAIL",
        "verdict": verdict, "root": str(root),
        "files_checked": len(files), "well_formed": ok, "malformed": bad,
        "checks": {"A_non_vacuous": bool(check_A), "B_all_well_formed": bool(check_B)},
        "why_this_exists": ("XML forbids '--' inside a comment, and a provenance comment carrying "
                            "an ordinary English phrase made the standing model unparseable by "
                            "xml.etree while a lenient loader accepted it"),
        "units": "counts only; no physical quantity is measured here",
        "provenance": "xml.etree.ElementTree (strictly conformant) over every *.xml under outputs/",
        "what_this_does_not_establish": (
            "validity as MJCF, compilability, or anything physical.  Well-formedness is a floor, "
            "not a claim about the model.  No real-fly recording exists in this project."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
