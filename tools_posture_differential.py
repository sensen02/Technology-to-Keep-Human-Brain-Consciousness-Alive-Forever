#!/usr/bin/env python3
"""Which SPATIALLY STRUCTURED knee command rolls an inverted fly back onto its feet?

MEASURED FACT THIS FOLLOWS FROM: a UNIFORM knee offset is useless.  A sign sweep from -40 to
+40 degrees found no value beating zero, because a command identical on all six legs is
symmetric about the body's long axis and therefore cannot produce a torque about it.  Righting
needs a differential, so this sweeps the two differentials that can generate a roll and a pitch:

  LATERAL  (roll)  : left three legs +d, right three -d
  LONGITUDINAL (pitch): front three +d, hind three -d

Both signs of each are run from an identical walk-then-invert state.  The sign is measured, not
assumed: the same class of sign assumption has already been wrong twice in this project (a
geometry axis read as a row instead of a column; a "dorsal" index that was inverted).

Run:
    MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_posture_differential.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine.embodied.adapters import LEGS  # noqa: E402

#: RANGE WIDENED AFTER A MISS.  The first sweep stopped at +/-40 degrees and concluded "no
#: static command beats zero".  That was WRONG, and the error was the sweep's RANGE, not the
#: physics: a coxa-pitch offset of +80 degrees moves up_z after inversion from -0.84 to -0.29,
#: i.e. it visibly rolls the fly.  A negative result is only as strong as the range it covers,
#: so the range is now wide enough to include the effect it missed.
AMPS_DEG = (-120.0, -80.0, -40.0, 0.0, 40.0, 80.0, 120.0)


def bias_vector(kind: str, d_deg: float):
    d = float(np.radians(d_deg))
    if kind == "uniform":
        return np.full(6, d)
    if kind == "lateral":       # +d on the LEFT legs, -d on the right
        return np.array([d, d, d, -d, -d, -d])
    if kind == "longitudinal":  # +d on the FRONT legs, -d on the hind
        return np.array([d, -d, -d, d, -d, -d])
    raise ValueError(kind)


def run(kind: str, d_deg: float, invert_at=1.0, duration=3.0, target="knee"):
    from engine.embodied.loop import LoopConfig, MultirateScheduler
    cfg = LoopConfig(arm="neural_modulated", seed=0, duration_s=duration,
                     spawn_position_mm=(0.0, 0.0, 0.6), invert_at_s=invert_at,
                     invert_height_mm=1.6, gl_backend=None)
    sch = MultirateScheduler(cfg)
    orig = sch.cpg.set_command
    bias = bias_vector(kind, d_deg)

    sch.cpg.set_posture_target(target)

    def patched(speed, turn):
        orig(speed, turn)
        sch.cpg.set_posture_bias(bias)
        return sch.cpg
    sch.cpg.set_command = patched
    ep = sch.run(want_frames=False)
    t = np.asarray(ep.truth["time_s"], float)
    rots = np.asarray(ep.truth["body_rotations_wxyz"], float)
    up = 1.0 - 2.0 * (rots[:, 0, 1] ** 2 + rots[:, 0, 2] ** 2)
    after = t >= invert_at
    return {"kind": kind, "target": target, "amp_deg": d_deg,
            "up_z_after_mean": float(up[after].mean()),
            "up_z_final": float(up[-1]),
            "upright_fraction_after": float((up[after] > 0.7).mean()),
            "up_z_max_after": float(up[after].max())}


def main() -> int:
    out = []
    for target in ("coxa_pitch",):
        for kind in ("uniform", "lateral"):
            print(f"\n--- target={target}  {kind} differential ---")
            for d in AMPS_DEG:
                r = run(kind, d, target=target)
                out.append(r)
                print(f"   {d:+6.1f} deg -> up_z after inversion mean {r['up_z_after_mean']:+.3f} "
                      f"max {r['up_z_max_after']:+.3f} final {r['up_z_final']:+.3f} "
                      f"upright {r['upright_fraction_after']*100:5.0f}%")
    zero = [r for r in out if r["amp_deg"] == 0.0][0]["up_z_after_mean"]
    better = [r for r in out if r["amp_deg"] != 0.0 and r["up_z_after_mean"] > zero + 0.05]
    print("\n" + "=" * 84)
    if better:
        best = max(better, key=lambda r: r["up_z_after_mean"])
        print(f"A LEG-SWING COMMAND HELPS: target={best['target']} {best['kind']} "
              f"{best['amp_deg']:+.1f} deg gives "
              f"up_z {best['up_z_after_mean']:+.3f} vs {zero:+.3f} at zero bias, "
              f"upright {best['upright_fraction_after']*100:.0f}% of the time after inversion.")
    else:
        print(f"NOTHING BEATS ZERO (zero bias gives up_z {zero:+.3f}).  Neither a uniform nor a "
              "single lateral/longitudinal differential rolls the fly back up, so the missing "
              "ingredient is not just 'some asymmetry' -- it is a TIME-VARYING, closed-loop one.")
    dest = HERE / "outputs" / "posture_differential.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"zero_baseline": zero, "results": out}, indent=2, sort_keys=True))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
