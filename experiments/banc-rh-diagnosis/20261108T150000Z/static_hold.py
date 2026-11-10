"""Valid static hold: the frozen incremental actuator holds the key-frame targets, the body settles."""
from pathlib import Path
import json, sys, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
WEIGHT = 24.446354755878005
spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
m = act.load_full_model(); d = mj.MjData(m)
mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
actuator = act.IncrementalActuator(m, d); actuator.reset(d)
th = m.body('Thorax').id
planes = m.geom_type == mj.mjtGeom.mjGEOM_PLANE
gname = [m.geom(i).name or '' for i in range(m.ngeom)]
legend = next((j for j, L in enumerate(LEGS) if gname[L and 0] and False), -1)
def leg_of(geom):
    for j, L in enumerate(LEGS):
        if gname[geom].startswith(L) and 'Tars' in gname[geom]:
            return j
    return -1
rows = []
jax = np.zeros((3, m.nv))
for k in range(800):                      # 800 x 0.5 ms = 0.4 s of settling, sampled every 0.5 ms
    for _ in range(40):
        actuator.update(np.zeros(42), .001)   # rate 0 -> hold targets, 1 ms actuator step
        d.ctrl[:] = actuator.torque(d)
        mj.mj_step(m, d); mj.mj_forward(m, d)
    row = np.zeros(6); nonfoot = 0.
    for ci in range(d.ncon):
        c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
        if not (planes[ga] or planes[gb]):
            continue
        geom = gb if planes[ga] else ga
        f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
        normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
        leg = leg_of(geom)
        if leg < 0:
            nonfoot += normal
        else:
            row[leg] += normal
    rows.append((row / WEIGHT, nonfoot / WEIGHT, float(d.subtree_com[th, 2])))
rows = np.array([r[0] for r in rows]); nf = np.array([r[1] for r in rows]); cz = np.array([r[2] for r in rows])
tail = rows[len(rows) // 4:]
out = dict(native_steps=800 * 40, note='frozen incremental actuator holds the key-frame targets; rate 0 throughout',
           per_leg_load_share_mean=dict(zip(LEGS, np.round(tail.mean(0), 4).tolist())),
           per_leg_load_share_std=dict(zip(LEGS, np.round(tail.std(0), 4).tolist())),
           total_share_mean=round(float(tail.sum(1).mean()), 4),
           nonfoot_share_max=round(float(nf.max()), 4),
           com_z_tail_mean=round(float(cz[len(cz) // 4:].mean()), 4), com_z_min=round(float(cz.min()), 4),
           feet_in_contact_fraction=dict(zip(LEGS, np.round((tail > 1e-6).mean(0), 3).tolist())))
(HERE / 'STATIC_HOLD.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
