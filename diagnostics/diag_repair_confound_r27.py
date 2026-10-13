"""R27: IS THE 'REPAIRS LIFTED THE PASSIVE BODY' RESULT A LOCK ARTEFACT?

WHY THIS EXISTS.  Round 25 (STANDING_DIAGNOSIS_ROUND10) reported, as its headline improvement:

    passive, no activation:  thorax 2.0454 -> 0.7163 mm   ("before the repairs")
                             thorax settled at 1.3917 mm  ("after the repairs")

and concluded the repaired skeleton was now holding the body up at 1.39 mm.  Round 26 then found that
the rebuilt canonical model carries SEVEN right-foreleg joint equality locks inherited from the
one-leg source asset, and that releasing only those locks takes the 0.5 s height from 1.402 mm to
0.726 mm.  1.3917 mm and 1.402 mm are the same number; 0.7163 mm and 0.726 mm are the same number.

So that before/after pair may not be measuring the repairs at all -- the locked foreleg is a rigid
strut -- and it is decidable by a 2 x 2 instead of by argument.

TWO CHANNELS, BECAUSE THE TWO REPAIRS ACT THROUGH DIFFERENT ONES.
  * the moment-arm repair (tools_repair_moment_arms.py) MOVED ATTACHMENT SITES.  Sites are massless,
    and a spatial tendon with no actuator exerts no force, so this repair can only act through
    ACTUATOR FORCE.
  * the operating-range repair (tools_repair_muscle_lengthrange.py) changed `lengthrange`, which sets
    the muscles' passive force-length curve -- so it can act even at act = 0, i.e. with
    actuation ENABLED and zero activation.
Measuring only the actuation-disabled condition would therefore be blind to the second repair by
construction, so both channels are measured.

THE ABLATION IS VERIFIED, NOT ASSUMED.  In the disabled conditions actuator_force and
qfrc_actuator are asserted exactly zero at EVERY step.  The previous round's "muscles off"
intervention left 77.59 units of actuator force and is the reason that attribution was withdrawn.

AND THE VARIANTS ARE FINGERPRINTED, because "all three identical" is exactly the kind of result this
project has learned to distrust: if the loader had silently returned one cached model, the table would
look the same.  So each variant reports its file hash, its size, and the measured difference in site
positions and tendon length ranges against the canonical model.

This does NOT modify the canonical model or any existing product.

Run:
    cd /run/media/sensen/Data2/cell_wound_prototype
    MUJOCO_GL=egl ./venv_body/bin/python diagnostics/diag_repair_confound_r27.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from flygym.compose.fly.musculoskeletal import _load_mjcf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "muscles_six_legs"
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

VARIANTS = (
    ("both repairs (canonical)", MODELS / "fruitfly_six_leg_standing.xml"),
    ("no arm repair", MODELS / "fruitfly_six_leg_standing_noarmrepair.xml"),
    ("qpos0 bands", MODELS / "fruitfly_six_leg_standing_qpos0bands.xml"),
)
SECONDS = 2.0
MARK_MS = 500.0


def compile_variant(xml: Path):
    m = _load_mjcf(str(xml)).compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = 0
    d.act[:] = 0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    sha = hashlib.sha256(xml.read_bytes()).hexdigest()[:12]
    return m, d, sha


def main() -> int:
    # ---- 0. FINGERPRINTS.  Distinct file, distinct site geometry, distinct length ranges.
    print("=== variant fingerprints (these SHARE their dynamics only if they share their structure) ===")
    ref = None
    fps = {}
    for label, xml in VARIANTS:
        assert xml.exists(), f"missing variant {xml}"
        m, d, sha = compile_variant(xml)
        site_x = np.asarray(d.site_xpos, dtype=float).copy()
        # MuJoCo 3.9 has no `tendon_lengthrange`; the band this project repairs is the ACTUATOR's
        # `lengthrange`, which is what tools_repair_muscle_lengthrange.py reads and writes.
        lr = np.asarray(m.actuator_lengthrange, dtype=float).copy()
        n_act = int(m.nu)
        if ref is None:
            ref = dict(site_x=site_x, lr=lr, nu=n_act)
            site_d, lr_d = 0.0, 0.0
        else:
            assert n_act == ref["nu"], "variant has a different actuator count"
            site_d = float(np.max(np.abs(site_x - ref["site_x"])))
            lr_d = float(np.max(np.abs(lr - ref["lr"])))
        fps[label] = dict(
            file=xml.name, sha256_12=sha, bytes=xml.stat().st_size,
            nq=int(m.nq), nu=int(m.nu), nsite=int(m.nsite), ngeom=int(m.ngeom),
            max_site_xpos_diff_vs_canonical_mm=site_d,
            max_actuator_lengthrange_diff_vs_canonical=lr_d,
            n_tendons=int(m.ntendon),
        )
        print(f"  {label:26s} {xml.name:44s} sha {sha}  nq {m.nq:3d} nu {m.nu:3d} "
              f"nsite {m.nsite:4d}  site diff {site_d:9.5f} mm  lengthrange diff {lr_d:11.6f}")
    assert len({f["sha256_12"] for f in fps.values()}) == len(VARIANTS), \
        "the variant files are not distinct on disk"
    assert fps["no arm repair"]["max_site_xpos_diff_vs_canonical_mm"] > 0, \
        "the arm repair left every site where it was -- the variant is not what it claims to be"
    assert fps["qpos0 bands"]["max_actuator_lengthrange_diff_vs_canonical"] > 0, \
        "the band repair changed no length range -- the variant is not what it claims to be"
    print("  -> the three files are distinct on disk AND distinct in structure. "
          "Any identical dynamics below are therefore a physical result, not a stale loader.\n")

    # ---- 1. THE MEASUREMENT.
    rows = []
    for label, xml in VARIANTS:
        for disable in (True, False):
            for released in (False, True):
                m, d, _ = compile_variant(xml)
                locks = [i for i in range(m.neq)
                         if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "")
                         .startswith("joint_RF")]
                assert len(locks) == 7, f"{label}: expected 7 RF locks, saw {len(locks)}"
                if disable:
                    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
                mujoco.mj_resetDataKeyframe(m, d, 0)
                if released:
                    d.eq_active[locks] = False
                d.ctrl[:] = 0
                d.act[:] = 0
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)

                th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
                z0 = float(d.xpos[th, 2])
                up0 = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2].copy()

                peak_force = peak_qfrc = peak_tilt = 0.0
                min_z = z0
                z_mark = None
                n_mark = round(MARK_MS * 1e-3 / m.opt.timestep)
                trace = []
                for step in range(round(SECONDS / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                    peak_force = max(peak_force,
                                     float(np.max(np.abs(d.actuator_force))) if m.nu else 0.0)
                    peak_qfrc = max(peak_qfrc,
                                    float(np.max(np.abs(d.qfrc_actuator))) if m.nu else 0.0)
                    up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
                    peak_tilt = max(peak_tilt,
                                    float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))))
                    z = float(d.xpos[th, 2])
                    min_z = min(min_z, z)
                    if step + 1 == n_mark:
                        z_mark = z
                    if (step + 1) % max(1, round(SECONDS / m.opt.timestep / 40)) == 0:
                        trace.append([round((step + 1) * m.opt.timestep, 4), round(z, 5)])
                mujoco.mj_forward(m, d)
                if disable:
                    assert peak_force == 0.0 and peak_qfrc == 0.0, \
                        f"{label}: ablation ineffective, peak force {peak_force}, qfrc {peak_qfrc}"

                up = np.asarray(d.xmat[th], dtype=float).reshape(3, 3)[:, 2]
                z = float(d.xpos[th, 2])
                feet = set()
                for c in range(int(d.ncon)):
                    cc = d.contact[c]
                    for g in (int(cc.geom1), int(cc.geom2)):
                        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
                            continue
                        other = int(cc.geom2) if g == int(cc.geom1) else int(cc.geom1)
                        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                        if nm[:2] in LEGS:
                            feet.add(nm[:2])
                row = dict(
                    variant=label,
                    activation_channel="disabled (no muscle force at all)" if disable
                    else "enabled, act = 0 (passive muscle force only)",
                    rf_locks_released=released,
                    start_thorax_z_mm=z0,
                    z_at_500ms_mm=z_mark,
                    final_thorax_z_mm=z,
                    sank_mm=z0 - z,
                    min_thorax_z_mm=min_z,
                    final_tilt_deg=float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))),
                    peak_tilt_deg=peak_tilt,
                    feet_in_contact=sorted(feet),
                    settled=bool(float(np.max(np.abs(d.qvel))) < 1e-3),
                    final_max_qvel=float(np.max(np.abs(d.qvel))),
                    peak_actuator_force=peak_force,
                    peak_qfrc_actuator=peak_qfrc,
                    z_trace=trace,
                )
                rows.append(row)
                print(f"  {label:26s} locks_rel={released!s:5s} {row['activation_channel'][:12]:12s} "
                      f"z@0.5s {z_mark:7.4f}  z@2s {z:7.4f}  tilt {row['final_tilt_deg']:7.2f} deg  "
                      f"feet {len(feet)}  peak|F| {peak_force:8.4f}", flush=True)

    # ---- 2. THE 2 x 2, PER CHANNEL.
    print("\n=== final thorax height (mm): the repair effect, per lock state ===")
    print(f"{'channel':34s} {'variant':26s} {'locks active':>13s} {'locks released':>15s}")
    idx = {(r["activation_channel"], r["variant"], r["rf_locks_released"]): r for r in rows}
    for ch in sorted({r["activation_channel"] for r in rows}):
        for label, _ in VARIANTS:
            a = idx[(ch, label, False)]["final_thorax_z_mm"]
            b = idx[(ch, label, True)]["final_thorax_z_mm"]
            print(f"{ch:34s} {label:26s} {a:13.4f} {b:15.4f}")

    print("\n=== the lock effect, and whether the repairs move it at all ===")
    for ch in sorted({r["activation_channel"] for r in rows}):
        vals = [idx[(ch, label, False)]["final_thorax_z_mm"] - idx[(ch, label, True)]["final_thorax_z_mm"]
                for label, _ in VARIANTS]
        print(f"{ch:34s} lock effect on final height (mm): " +
              ", ".join(f"{v:.4f}" for v in vals) + f"   spread {max(vals) - min(vals):.4f}")

    doc = dict(
        status="PASS",
        measurement="passive settle height and tilt over 2 s; actuator force asserted zero every step",
        units="length mm; angle degrees; model force units and uN mm for generalized torque",
        provenance=f"MuJoCo {mujoco.__version__}; {SECONDS} s; fresh model and data per condition",
        question="Was the R25 'passive settle 0.7163 -> 1.3917 mm' improvement produced by the two "
                 "repairs, or by the seven inherited right-foreleg locks that R26 found in the very "
                 "same rebuilt model?",
        variant_fingerprints=fps,
        rows=rows,
        what_this_does_not_establish=(
            "Standing, walking, or neural control, and not that the repairs are worthless: they remove "
            "one-way joints and put every muscle inside its force-length band, both of which are "
            "measured elsewhere. What is in question here is only the claim that they lifted the "
            "passive body. The keyframe is still not the pose the stance solver solved, the contact "
            "channel is still the declared NON_PHYSICAL_LOAD subsystem, and nothing here says a "
            "controller could not stand this body."),
    )
    out = ROOT / "outputs" / "repair_confound_r27.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
