"""Controlled A/B: does correcting the unit LABEL move the PHYSICS?

WHY AN A/B AND NOT JUST A UNIT TEST
-----------------------------------
Round 39 declared the model length unit to be the centimetre and changed
``SEATING_PENETRATION_MM`` from -0.05 to -0.5 (so the product fed to `seat_root` stayed at
-0.05).  The correction in this round declares millimetres and puts the declaration back to
-0.05.  The claim that "only labels moved" is testable directly: run the SAME closure code
under both conventions and compare the integrated physics.

    legacy     : MM_PER_MODEL_LENGTH = 10.0, SEATING_PENETRATION_MM = -0.5   -> MODEL -0.05
    corrected  : MM_PER_MODEL_LENGTH =  1.0, SEATING_PENETRATION_MM = -0.05  -> MODEL -0.05

EXPECTED RESULT (pre-registered)
  * every trajectory / control / spike array is BIT-IDENTICAL between the two runs;
  * every reported millimetre label differs by exactly a factor of 10.
FALSIFIER: any trajectory difference.  Then the "label-only" claim is false and the correction
changed the simulation, which must be reported as a behavioural change, not a relabelling.

Usage: venv_body/bin/python run_unit_fix_ab.py --convention legacy|corrected --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convention", choices=("legacy", "corrected"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.05)
    args = parser.parse_args()

    import engine.canonical_physics as cp
    import engine.memory_guard as mg
    import run_neural_muscle_closure as clo

    if args.convention == "legacy":
        # reproduce round 39's convention exactly
        cp.MODEL_LENGTH_UNIT_NAME = "cm"
        cp.MM_PER_MODEL_LENGTH = 10.0
        cp.SEATING_PENETRATION_MM = -0.5
        cp.SEATING_PENETRATION_MODEL = float(cp.mm_to_model(cp.SEATING_PENETRATION_MM))
    # the corrected values are whatever the module now declares; nothing to patch
    product = float(cp.SEATING_PENETRATION_MODEL)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    clo.OUT = out
    sys.argv = ["run_neural_muscle_closure.py", "--duration", str(args.duration)]
    with mg.guarded_run(f"unit_ab_{args.convention}",
                        need_mb=mg.BUDGETS["neural_muscle_closure"]["need_mb"],
                        cap_mb=mg.BUDGETS["neural_muscle_closure"]["cap_mb"]):
        clo.main()

    doc = json.loads((out / "closure_results.json").read_text())
    reported = {arm: {"final_height_reported_mm": ep["records"][-1]["height_mm_declared"],
                      "escape_time_s": ep["escape_time_s"]}
                for arm, ep in doc["measured"]["episodes"].items()}
    (out / "ab_summary.json").write_text(json.dumps(
        {"convention": args.convention,
         "MM_PER_MODEL_LENGTH": cp.MM_PER_MODEL_LENGTH,
         "SEATING_PENETRATION_MM": cp.SEATING_PENETRATION_MM,
         "seat_root_target_model": product,
         "reported": reported}, indent=2))
    print(f"{args.convention}: factor={cp.MM_PER_MODEL_LENGTH} declared_mm="
          f"{cp.SEATING_PENETRATION_MM} seat_target={product}")


if __name__ == "__main__":
    main()
