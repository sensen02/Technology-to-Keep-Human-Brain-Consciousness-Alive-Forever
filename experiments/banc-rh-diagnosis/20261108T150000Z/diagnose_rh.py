"""RH vs the other legs: geometry, ranges, contact geometry, and per-leg load share."""
from pathlib import Path
import json, sys
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
WEIGHT = 24.446354755878005
out = {}
import importlib.util
spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
m = act.load_full_model(); d = mj.MjData(m)
mj.mj_resetDataKeyframe(m, d, 0); mj.mj_forward(m, d)
th = m.body('Thorax').id
jids = m.actuator_trnid[:, 0]
# 1) joint ranges and key-frame angles per leg
rng = m.jnt_range[jids]; key = m.key_qpos[0][m.jnt_qposadr[jids]]
out['joints'] = {}
for l, leg in enumerate(LEGS):
    ix = np.arange(l * 7, l * 7 + 7)
    out['joints'][leg] = dict(names=[m.joint(int(jids[i])).name for i in ix],
                              range_lo=np.round(rng[ix, 0], 4).tolist(), range_hi=np.round(rng[ix, 1], 4).tolist(),
                              key_angle=np.round(key[ix], 4).tolist())
out['lh_vs_rh'] = dict(range_diff=np.round(np.abs(rng[7:14] - rng[35:42]), 6).tolist(),
                       key_diff=np.round(np.abs(key[7:14] - key[35:42]), 6).tolist())
# 2) lowest material point and contacting geom per leg in the key pose
planes = m.geom_type == mj.mjtGeom.mjGEOM_PLANE
gname = [m.geom(i).name or '' for i in range(m.ngeom)]
out['feet'] = {}
for leg in LEGS:
    rows = []
    for g in range(m.ngeom):
        if planes[g]:
            continue
        b = int(m.geom_bodyid[g])
        bn = m.body(b).name or ''
        if not bn.startswith(leg) or 'Tars' not in bn:
            continue
        # lowest point of this geom measured in world z, approximated by the geom's own support point
        p = np.asarray(d.geom_xpos[g]).copy()
        z_min = p[2] - float(m.geom_size[g]).__abs__() if m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE else p[2]
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mi = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float)
            R = np.zeros(9); mj.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            vw = (v @ R.T) + p
            z_min = float(vw[:, 2].min())
        rows.append(dict(geom=gname[g], body=bn, body_id=b, quat=np.round(m.geom_quat[g], 4).tolist(),
                         pos_z=round(float(p[2]), 5), lowest_z=round(z_min, 5), nverts=int(m.mesh_vertnum[int(m.geom_dataid[g])]) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH else None))
    rows.sort(key=lambda r: r['lowest_z'])
    out['feet'][leg] = dict(lowest=rows[0] if rows else None, all=rows)
# 3) static hold: hold the key frame for 1 s and record per-leg load share
NSTEPS = int(round(1.0 / m.opt.timestep))
d.ctrl[:] = 0.
loads = np.zeros(6); cnt = 0; share_hist = []
jacp = np.zeros((3, m.nv))
for k in range(NSTEPS):
    for j in range(m.nu):
        mj.mj_step(m, d)
    mj.mj_forward(m, d)
    row = np.zeros(6); nonfoot = 0.
    for ci in range(d.ncon):
        c = d.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
        if not (planes[ga] or planes[gb]):
            continue
        geom = gb if planes[ga] else ga
        f = np.zeros(6); mj.mj_contactForce(m, d, ci, f)
        normal = max(0., float((c.frame.reshape(3, 3).T @ f[:3])[2]) * (1 if planes[ga] else -1))
        leg = next((j for j, L in enumerate(LEGS) if gname[geom].startswith(L) and 'Tars' in gname[geom]), -1)
        if leg < 0:
            nonfoot += normal
        else:
            row[leg] += normal
    share_hist.append(row / WEIGHT)
share_hist = np.array(share_hist)
out['static_hold'] = dict(native_steps=NSTEPS, physics_steps_used=NSTEPS,
                          per_leg_load_share_mean=dict(zip(LEGS, np.round(share_hist.mean(0), 4).tolist())),
                          per_leg_load_share_std=dict(zip(LEGS, np.round(share_hist.std(0), 4).tolist())),
                          total_share_mean=round(float(share_hist.sum(1).mean()), 4),
                          com_z_end=round(float(d.subtree_com[th, 2]), 4))
# 4) slip/stance numbers from the best existing traces
prev = {}
for tag, path in (('schemeA_gripon', ROOT / 'experiments/banc-full42-ppo/20261108T110000Z/EVAL_gripON.json'),
                  ('schemeA_gripoff_steady', ROOT / 'experiments/banc-full42-ppo/20261108T110000Z/EVAL_gripOFF.json'),
                  ('height_w20_gripoff', ROOT / 'experiments/banc-full42-height/20261108T130000Z/EVAL_gripOFF_w20.json')):
    if path.exists():
        z = json.loads(path.read_text())
        prev[tag] = {k: dict(per_leg_slip_ratio=z[k]['per_leg_slip_ratio'], slip_ratio_max=z[k]['slip_ratio_max'])
                     for k in ('release', 'steady') if k in z}
out['previous_runs'] = prev
(HERE / 'RH_DIAGNOSIS.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
lowest = {leg: out['feet'][leg]['lowest'] for leg in LEGS}
print(json.dumps({'lowest_point_per_leg': {leg: (lowest[leg]['body'] if lowest[leg] else None, lowest[leg]['lowest_z'] if lowest[leg] else None) for leg in LEGS},
                  'load_share': out['static_hold']['per_leg_load_share_mean'],
                  'lh_vs_rh_key_angle_diff': out['lh_vs_rh']['key_diff']}))
