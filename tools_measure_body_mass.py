#!/usr/bin/env python3
"""Measure the UNLOADED fly's body mass -- the yardstick for payload loading.

Writes ``outputs/electrode_payload/body_mass.json``.  Run with the body venv:

    cd /run/media/sensen/Data2/cell_wound_prototype
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      venv_body/bin/python tools_measure_body_mass.py

It also prints the largest body masses so the payload's magnitude relative to the
thorax is visible rather than only relative to the whole fly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from electrode_payload import (BODY_MASS_KG_FALLBACK,  # noqa: E402
                               assert_yardstick, electrode_glass_mass_kg,
                               measure_body_mass_kg, model_signature)


def main() -> int:
    from engine.embodied.body_backend import BodyBackend, BodyConfig

    # No payload -> this model IS the unloaded fly.
    be = BodyBackend(BodyConfig(), gl_backend=None)
    try:
        model = be.model
        mass = measure_body_mass_kg(model)
        sig = model_signature(model)
        masses = np.asarray(model.body_mass, dtype=float)
        order = np.argsort(masses)[::-1][:6]
        top = [{"body_id": int(i), "name": str(model.body(int(i)).name),
                "mass_kg": float(masses[i])} for i in order]
        check = assert_yardstick(model, BODY_MASS_KG_FALLBACK)
        glass = electrode_glass_mass_kg()
        out = {
            "unloaded_body_mass_kg": mass,
            "n_body": int(model.nbody),
            "largest_bodies": top,
            "yardstick_check": check,
            "bare_electrode_glass_kg": glass["total_kg"],
            "bare_electrode_fraction_of_body": glass["total_kg"] / mass,
            "bare_electrode_report": glass,
            "note": ("Total mass = sum of MuJoCo body_mass on the UNLOADED locomotion "
                     "model. This is the yardstick every payload load_fraction is a "
                     "fraction of."),
        }
        dest = ROOT / "outputs" / "electrode_payload" / "body_mass.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"unloaded body mass = {mass:.8e} kg  (n_body={model.nbody})")
        for t in top:
            print(f"  {t['mass_kg']:.6e}  {t['name']}")
        print(f"bare electrode glass+glue = {glass['total_kg']:.6e} kg "
              f"= {100 * glass['total_kg'] / mass:.4f} % of body mass")
        print(f"wrote {dest}")
        return 0
    finally:
        be.close()


if __name__ == "__main__":
    raise SystemExit(main())
