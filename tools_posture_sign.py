#!/usr/bin/env python3
"""MEASURE which sign of the knee offset actually pushes an inverted fly off its back.

I refuse to guess this.  A posture bias added to the knee pitch can either extend a leg
against the ground or fold it, and which one it does depends on the joint's axis convention in
this model -- the same class of assumption that has been wrong repeatedly in this project (a
geometry axis read as a row instead of a column, a radius in metres in a mm model).  So both
signs are run from an identical inverted state and the one that raises the fly is kept.

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_posture_sign.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BIASES_DEG = (-40.0, -20.0, 0.0, 20.0, 40.0)


def run(bias_deg: float, invert_at: float, duration: float):
    from engine.embodied.loop import LoopConfig, MultirateScheduler
    import mujoco
    cfg = LoopConfig(arm="neural_modulated", seed=0, duration_s=duration,
                     spawn_position_mm=(0.0, 0.0, 0.6), invert_at_s=invert_at,
                     invert_height_mm=1.6, gl_backend=None)
    sch = MultirateScheduler(cfg)
    # the bias is written through the SAME adapter the loop commands, so the causal path is
    # unchanged: the loop still writes speed/turn, this only adds the third dimension.
    orig_set = sch.cpg.set_command

    def patched(speed, turn):
        orig_set(speed, turn)
        sch.cpg.set_posture_bias(float(np.radians(bias_deg)))
        return sch.cpg
    sch.cpg.set_command = patched
    ep = sch.run(want_frames=False)
    t = np.asarray(ep.truth["time_s"], float)
    rots = np.asarray(ep.truth["body_rotations_wxyz"], float)
    up = 1.0 - 2.0 * (rots[:, 0, 1] ** 2 + rots[:, 0, 2] ** 2)
    after = t >= invert_at
    return {
        "bias_deg": bias_deg,
        "up_z_after_inversion_mean": float(up[after].mean()),
        "up_z_final": float(up[-1]),
        "up_z_max_after_inversion": float(up[after].max()),
        "upright_fraction_after": float((up[after] > 0.7).mean()),
    }


def main() -> int:
    out = []
    for b in BIASES_DEG:
        r = run(b, invert_at=1.0, duration=3.0)
        out.append(r)
        print(f"  bias {b:+6.1f} deg -> up_z after inversion: mean {r['up_z_after_inversion_mean']:+.3f}  "
              f"max {r['up_z_max_after_inversion']:+.3f}  final {r['up_z_final']:+.3f}  "
              f"upright {(r['upright_fraction_after'] * 100):.0f}%")
    best = max(out, key=lambda r: r["up_z_after_inversion_mean"])
    print(f"\nBEST SIGN: {best['bias_deg']:+.1f} deg (up_z after inversion "
          f"{best['up_z_after_inversion_mean']:+.3f})")
    if abs(best["bias_deg"]) < 1e-9:
        print("=> NO bias beats zero, i.e. the knee offset does not right the fly on its own.")
    dest = HERE / "outputs" / "posture_sign.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"sweep": out, "best": best}, indent=2, sort_keys=True))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
