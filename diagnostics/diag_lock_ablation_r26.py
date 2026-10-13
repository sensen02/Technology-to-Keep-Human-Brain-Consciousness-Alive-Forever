"""Read-only 2x2 ablation: right-foreleg locks versus verified actuator disabling.
No controller and no canonical XML changes. Measurements are simulator output.
"""
from pathlib import Path
import json
import os
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'

def main():
    rows = []
    for unlock in (False, True):
        for disabled in (False, True):
            m = _load_mjcf(str(XML)).compile()
            d = mujoco.MjData(m)
            locks = [i for i in range(m.neq)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or '').startswith('joint_RF')]
            assert len(locks) == 7, 'Expected the seven independently observed right-foreleg locks'
            if disabled:
                m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            if unlock:
                d.eq_active[locks] = False
            d.ctrl[:] = 0
            d.act[:] = 0
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            th = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Thorax')
            up = d.xmat[th].reshape(3, 3)[:, 2].copy()
            peak_force = peak_qfrc = 0.0
            for _ in range(round(.5 / m.opt.timestep)):
                mujoco.mj_step(m, d)
                peak_force = max(peak_force, float(np.max(np.abs(d.actuator_force))))
                peak_qfrc = max(peak_qfrc, float(np.max(np.abs(d.qfrc_actuator))))
            mujoco.mj_forward(m, d)
            if disabled:
                assert peak_force == 0 and peak_qfrc == 0, 'Actuator ablation was not effective'
            row = dict(unlocked=unlock, actuation_disabled=disabled,
                       final_height_mm=float(d.xpos[th, 2]),
                       final_tilt_deg=float(np.degrees(np.arccos(np.clip(
                           d.xmat[th].reshape(3, 3)[:, 2] @ up, -1, 1)))),
                       peak_actuator_force=peak_force, peak_qfrc_actuator=peak_qfrc,
                       active_rf_locks=int(np.count_nonzero(d.eq_active[locks])))
            rows.append(row)
            print(row, flush=True)
    doc = dict(status='PASS', rows=rows,
               units='height mm; angle degrees; model force uN and generalized joint torque uN mm',
               provenance=f'MuJoCo {mujoco.__version__}; {XML}; 0.5 s; fresh model per condition',
               what_this_does_not_establish='Standing, neural control, or a sole root cause. The unlocked model still collapses. Prior contact-only attribution is withdrawn; the previous muscle-off intervention did not zero actuator force.')
    out = ROOT / 'outputs/lock_ablation_r26.json'
    out.write_text(json.dumps(doc, indent=2), encoding='utf-8')
    print(out)

if __name__ == '__main__':
    main()
