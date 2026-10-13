#!/usr/bin/env python3
"""TRUTH-BOUNDARY AUDIT.  Which code is allowed to read the simulator's own state?

WHY THIS EXISTS.  This project runs two chains that look alike and mean opposite things:

  * the MEASUREMENT chain -- camera images -> detections -> geometry.  Its numbers are evidence.
  * the SIMULATOR chain -- MuJoCo state.  Its numbers are the thing being scored against.

A measurement that quietly reads `truth/` is not a measurement; it is the answer copied from the
back of the book.  Nothing crashes, and the number looks BETTER, which is why this is worth a
mechanical check rather than a convention.  The project already had to state, in the tracker's own
docstring, that "truth is used solely for the error numbers reported here" -- a promise nothing
enforced.  This audit enforces it.

HOW THE SCAN WORKS, and the first version was wrong in a way worth recording: it stripped comments
and string literals to avoid flagging prose, which also stripped the DATA KEYS, because a truth key
IS a string literal (`z["truth/thorax_mm"]`).  It reported 3 scripts instead of 23.  The scan now
walks the AST and excludes only genuine DOCSTRINGS (the first statement of a module/class/function),
so `truth/` in a comment is ignored and `truth/` as a data key is counted.

THE RULE.  Every script that reads a truth key must DECLARE its role:

    simulator          it produces the truth; reading it is the job
    scoring            it compares a measurement against the truth; allowed, and it must say so
    analysis_of_truth  the simulator state IS the subject of study
    measurement        it must NOT read truth at all -- a read here is a FAIL

An UNDECLARED script is a FAIL, not a pass: the point is to force the classification rather than to
let a new script slip in unexamined.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    ./venv/bin/python tools_truth_boundary_audit.py
Exit code 0 means the audit ran.  The verdict is `overall` in the JSON.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SKIP_DIR_PARTS = ("site-packages", "node_modules", "vendor", ".git")

#: role per script that reads a truth key.  `why` is REQUIRED for `measurement` and for anything a
#: reader might expect to be a measurement, because that is where the interesting mistakes live.
ROLES: dict[str, dict] = {
    # -- the simulator chain: truth is its product ---------------------------------------------
    # NOTE: `simulator` entries carry no `why`, and the audit does not require one for them,
    # because producing the truth IS reading it.  `scoring` and `analysis_of_truth` DO require a
    # reason -- see the `elif not decl.get("why")` branch below -- because those are the labels
    # that could be used to wave a boundary crossing through.
    "engine/embodied/loop.py": {"role": "simulator"},
    "engine/embodied/slow_physiology.py": {"role": "simulator"},
    "run_arena_record.py": {"role": "simulator"},
    "run_tactile_record.py": {"role": "simulator"},
    "run_recording_pipeline.py": {"role": "simulator"},
    "run_embodied_loop.py": {"role": "simulator"},
    "run_embodied_slowphys.py": {"role": "simulator"},
    "run_embodied_loop_selftest.py": {"role": "simulator", "why": "a self-test of the loop; the recorded truth is the subject under test"},
    # -- analysis in which the truth IS the subject -------------------------------------------
    "export_integrated_viewer.py": {"role": "analysis_of_truth", "why": "the recorded simulator state IS the content being exported and displayed"},
    "scene_build_cli.py": {"role": "analysis_of_truth", "why": "it builds scene layers FROM the recorded state; the truth is the input subject"},
    "render_touch_xray.py": {"role": "analysis_of_truth", "why": "it renders the recorded contact geometry; truth is the subject of the picture"},
    "run_tactile_figures.py": {"role": "analysis_of_truth", "why": "it plots the recorded contact quantities; no image-derived measurement is made here"},
    "run_recording_figures.py": {"role": "analysis_of_truth", "why": "it plots the recorded channels; truth is the data being plotted"},
    "test_body_export_regression.py": {"role": "analysis_of_truth", "why": "a regression that asserts the exported body matches the recorded one -- truth is the reference"},
    "tools_reality_check.py": {"role": "analysis_of_truth", "why": "it compares platform outputs against the recorded truth as a reality check"},
    # -- scoring: a measurement compared against the truth ------------------------------------
    "run_arena_verify.py": {"role": "scoring", "why": "it verifies the reconstruction AGAINST truth; the reconstruction itself reads only px/ and geometry"},
    "run_recording_verify.py": {"role": "scoring", "why": "it verifies the recording pipeline against truth"},
    "tools_identity_budget.py": {"role": "scoring", "why": "it measures association/identity performance by scoring against truth"},
    "tools_limb_error_tail.py": {"role": "scoring", "why": "it summarises the error tail of a measurement against truth"},
    "tools_limb_tracking_figure.py": {"role": "scoring", "why": "it plots the tracked measurement against truth"},
    "limb_tracking.py": {"role": "scoring",
                         "why": ("truth is used ONLY for the error numbers, as its docstring "
                                 "promises -- WITH ONE DECLARED EXCEPTION that is a measurement-path "
                                 "dependency: the identity of the markers at FRAME 0 is seeded from "
                                 "the simulator (`seed0 = truth[0]`), standing in for the hand "
                                 "annotation a real pipeline would do (DeepFly3D annotates frame 0 "
                                 "by hand).  Every later frame is produced by the gates alone, and "
                                 "the tracker reports its own error, so the substitution is visible "
                                 "rather than hidden.")},
    "tools_arena_calib_check.py": {"role": "scoring",
                                   "why": ("IT IS A DIAGNOSTIC, so both halves are legitimate and "
                                           "must be told apart: the MEASUREMENT values it uses are "
                                           "the DECLARED positions from arena_geometry.json, and the "
                                           "truth read is a deliberate truth-vs-declared cross-check "
                                           "-- 'if the truth markers project onto the blobs but the "
                                           "declared ones do not, the episode's world frame is not "
                                           "the declaration's world frame'.  Truth is the thing it "
                                           "tests against, never a source of the numbers it draws.")},
    "tools_marker_coverage.py": {"role": "scoring",
                                 "why": ("A LIMITATION THAT MATTERS AND WAS PREVIOUSLY UNSTATED.  "
                                         "Its visibility figures are ANCHORED on "
                                         "`truth/marker_fly_world_mm`: it projects the simulator's "
                                         "own marker positions and asks whether a detected blob "
                                         "lies within a gate of each.  So the count is real "
                                         "(it needs actual detections) but it cannot discover a "
                                         "marker the simulator placed wrongly, and it is NOT an "
                                         "independent measurement of which markers are detectable "
                                         "in the images.  The visibility numbers quoted for the "
                                         "limb-angle work (6.5-33 % for trochanterfemur) therefore "
                                         "carry this caveat.")},
    "tools_truth_boundary_audit.py": {"role": "analysis_of_truth",
                                      "why": "it audits the codebase, so `truth/` appears as data"},

    # --- declared in the R14/R15 session.  Every one of these APPEARS HERE BECAUSE THE AUDIT ASKED:
    # the scan reported them as reading `truth/` without a declared role, and the answer is to say
    # what each one does with truth rather than to widen the audit.
    "tools_record_all_channels.py": {"role": "recorder",
                                     "why": ("it WRITES the `truth/` arrays into the episode -- the "
                                             "declared fiducial positions and the fly's own marker "
                                             "positions read out of `geom_xpos`.  Truth here is the "
                                             "RECORD, not a measurement, and the markers recorded are "
                                             "the same geoms the camera rendered.")},
    "tools_make_stackb_observable.py": {"role": "recorder",
                                        "why": ("same split: it writes `truth/marker_static_world_mm` "
                                                "and `truth/marker_fly_world_mm` for stack B, from "
                                                "the declared fiducials and from `geom_xpos`.")},
    "tools_render_neural_loop.py": {"role": "recorder",
                                    "why": "it writes the static-fiducial truth array for its episode"},
    "tools_test_all_channels.py": {"role": "scoring",
                                   "why": ("it CHECKS the recorder's own output: the frame clock is "
                                           "compared against the electrode's interval clock, which is "
                                           "a consistency test between two recorded series, not a "
                                           "measurement of the body")},
    "tools_seam_survey.py": {"role": "scoring",
                             "why": ("it verifies that the registration matches the artifacts, so "
                                     "`truth/frame_interval_index` is the thing being audited "
                                     "against the declaration")},
    "tools_test_limb_stackb.py": {"role": "scoring",
                                  "why": ("it scores the marker truth array's SHAPE and column order "
                                          "against the declared segment list, and its triangulation "
                                          "injection uses truth as the seed on purpose")},
    "tools_all_channels_figure.py": {"role": "figure_of_truth",
                                     "why": ("it draws the recorded frames on the recorded clock; "
                                             "`truth/frame_interval_index` is a label, not a number "
                                             "reported as a measurement")},
    "tools_limb_stackb_figure.py": {"role": "figure_of_truth",
                                    "why": ("it plots the tracked markers against the simulator's own "
                                            "positions, which is exactly the truth-vs-measurement "
                                            "comparison the tracker already reports numerically")},
    "tools_acceptance_abcd.py": {"role": "analysis_of_truth",
                                 "why": ("it evaluates the PLAN's four acceptance criteria from other "
                                         "checks' artifacts, so `truth/` appears only in the strings "
                                         "it searches for")},
    "engine/embodied/force_channel.py": {"role": "declaration",
                                         "why": ("it declares the contact-force channel a "
                                                 "NON_PHYSICAL_LOAD and NAMES `truth/normal_force_N` "
                                                 "in order to state its units.  It reads nothing: the "
                                                 "mention is in a docstring and in a units sentence, "
                                                 "which is the one case the scanner cannot tell from a "
                                                 "read.  Declared rather than exempted, because the "
                                                 "distinction is the point.")},
    "engine/embodied/seams.py": {"role": "declaration",
                                 "why": ("it is the SEAM REGISTRY: the `truth/` key names appear in "
                                         "its `works_in`/`limits` prose describing what each seam "
                                         "does, and it reads nothing")},
}


def truth_key_reads(path: Path) -> list[dict]:
    """String literals containing 'truth/' that are not genuine docstrings."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                doc_ids.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in doc_ids and "truth/" in node.value:
            out.append({"line": node.lineno, "key": node.value[:70]})
    return out


def scan() -> list[tuple[str, list[dict]]]:
    rows = []
    for p in sorted(HERE.rglob("*.py")):
        rel = p.relative_to(HERE)
        if any(part in rel.parts for part in SKIP_DIR_PARTS):
            continue
        hits = truth_key_reads(p)
        if hits:
            rows.append((str(rel), hits))
    return rows


def audit() -> dict:
    rows = scan()
    findings, undeclared = [], []
    for rel, hits in rows:
        decl = ROLES.get(rel)
        if decl is None:
            undeclared.append({"script": rel, "n_reads": len(hits),
                               "first": hits[0]})
            continue
        if decl["role"] == "measurement":
            findings.append({
                "script": rel, "role": "measurement", "violation": "reads a truth key",
                "n_reads": len(hits), "reads": hits,
                "declared_why": decl.get("why", ""),
            })
        elif decl["role"] != "simulator" and not decl.get("why"):
            # A `scoring`/`analysis_of_truth` declaration without a stated reason is an empty
            # declaration: it is the label that makes a truth read look examined.
            findings.append({
                "script": rel, "role": decl["role"],
                "violation": "reads a truth key but the declaration gives no reason",
                "n_reads": len(hits), "reads": hits, "declared_why": "",
            })
    # scripts DECLARED measurement that do not appear in the scan at all: report as clean, so the
    # audit also documents what it verified rather than only what it caught.
    declared_measurement = [s for s, d in ROLES.items() if d["role"] == "measurement"]
    scanned = {r for r, _h in rows}
    clean = [s for s in declared_measurement if s not in scanned]

    if undeclared:
        overall = "FAIL"
    elif findings:
        overall = "FAIL"
    else:
        overall = "PASS"
    return {"overall": overall,
            "units": ("not applicable: this audit reports file/line locations and counts"),
            "provenance": "AST parse of every .py in the repository; see scan_method",
            "n_scripts_with_truth_reads": len(rows),
            "n_reads_total": sum(len(h) for _r, h in rows),
            "undeclared": undeclared, "violations": findings,
            "declared_measurement_verified_clean": clean,
            "roles": ROLES,
            "scan_method": ("AST walk; excludes module/class/function DOCSTRINGS only, so `truth/` "
                            "in a comment is ignored and `truth/` as a data key is counted.  An "
                            "earlier version stripped all strings and reported 3 scripts instead "
                            "of 23."),
            "convention": ("exit code 0 means the audit ran.  An undeclared script that reads truth "
                           "is a FAIL, not a pass: the classification is the point."),
            "what_this_does_not_establish": (
                "that a declared `scoring` script uses truth ONLY for scoring.  This audit is "
                "STATIC.  A runtime version -- run the measurement with the truth keys removed and "
                "require the same output -- is stronger and is the next step.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "outputs" / "truth_boundary_audit.json"))
    args = ap.parse_args()
    res = audit()
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, sort_keys=True))

    print(f"scripts that read a truth key : {res['n_scripts_with_truth_reads']} "
          f"({res['n_reads_total']} reads)")
    print(f"undeclared                    : {len(res['undeclared'])}")
    for u in res["undeclared"]:
        print(f"   - {u['script']}  ({u['n_reads']} reads, first at line {u['first']['line']})")
    print(f"measurement-role violations   : {len(res['violations'])}")
    for v in res["violations"]:
        print(f"   - {v['script']}  {v['n_reads']} truth read(s)")
    print(f"declared measurement, verified clean: {len(res['declared_measurement_verified_clean'])}")
    print(f"\nOVERALL: {res['overall']}\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
