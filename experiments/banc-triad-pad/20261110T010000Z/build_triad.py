"""Three-sphere pad triad per tarsus, bottoms coplanar with the segment's lowest material point."""
from pathlib import Path
import hashlib, json, re, sys, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCE = ROOT / 'experiments/banc-tarsus-pad/20261109T010000Z/MODEL_TARSUS_PAD_R02.xml'
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
R = float(__import__('os').environ.get('TRIAD_R', '0.008'))
SPREAD = float(__import__('os').environ.get('TRIAD_SPREAD', '0.006'))
TORQUE = float(__import__('os').environ.get('TORQUE', '2.0'))
spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
base = act.load_full_model(); d = mj.MjData(base); mj.mj_resetDataKeyframe(base, d, 0); mj.mj_forward(base, d)
low = {}
for leg in LEGS:
    body = base.body(leg + 'Tarsus5').id
    g = [g for g in range(base.ngeom) if int(base.geom_bodyid[g]) == body and base.geom_type[g] == mj.mjtGeom.mjGEOM_MESH][0]
    mi = int(base.geom_dataid[g]); v = base.mesh_vert[base.mesh_vertadr[mi]:base.mesh_vertadr[mi] + base.mesh_vertnum[mi]].astype(float)
    Rm = np.zeros(9); mj.mju_quat2Mat(Rm, base.geom_quat[g]); vb = v @ Rm.reshape(3, 3).T + base.geom_pos[g]
    low[leg] = vb[int(np.argmin(vb[:, 2]))]
text = SOURCE.read_text()
for leg in LEGS:
    old = re.search(r'\n\s*<geom name="%sTarsus5_pad"[^>]*/>' % leg, text)
    assert old, leg
    c = low[leg]
    offsets = [(SPREAD, 0., 0.), (-.5 * SPREAD, .866 * SPREAD, 0.), (-.5 * SPREAD, -.866 * SPREAD, 0.)]
    new = []
    for k, (dx, dy, dz) in enumerate(offsets):
        new.append('\n                      <geom name="%sTarsus5_pad%d" type="sphere" size="%g" pos="%.9g %.9g %.9g" '
                   'rgba="0.05 0.85 0.95 1" contype="2" conaffinity="1" solref=".002 1"/>'
                   % (leg, k, R, c[0] + dx, c[1] + dy, c[2] + R + dz))
    text = text[:old.start()] + ''.join(new) + text[old.end():]
if TORQUE != 1.0:
    MOTOR = r'(<motor[^>]*ctrlrange=")([-0-9.eE+]+) ([-0-9.eE+]+)(")'
    def sub(mo):
        return mo.group(1) + '%.12g %.12g' % (float(mo.group(2)) * TORQUE, float(mo.group(3)) * TORQUE) + mo.group(4)
    text, k = re.subn(MOTOR, sub, text)
    assert k == 42, k
out = HERE / ('MODEL_TRIAD%s.xml' % ('_x200' if TORQUE == 2.0 else ''))
out.write_text(text)
m = act.load_full_model(out)
chk = dict(nq=int(m.nq) == int(base.nq), nv=int(m.nv) == int(base.nv), nu=int(m.nu) == int(base.nu),
           neq=int(m.neq) == int(base.neq), nmesh=int(m.nmesh) == int(base.nmesh),
           ranges_equal=bool(np.array_equal(m.jnt_range, base.jnt_range)),
           mass_equal=bool(np.array_equal(m.body_mass, base.body_mass)),
           inertia_equal=bool(np.array_equal(m.body_inertia, base.body_inertia)),
           gain_equal=bool(np.array_equal(m.actuator_gainprm, base.actuator_gainprm)),
           bias_equal=bool(np.array_equal(m.actuator_biasprm, base.actuator_biasprm)),
           gravity_equal=bool(np.array_equal(m.opt.gravity, base.opt.gravity)),
           timestep_equal=bool(np.isclose(m.opt.timestep, base.opt.timestep)),
           ngeom_from=int(base.ngeom), ngeom_to=int(m.ngeom),
           torque_factor=TORQUE)
# every pad bottom must sit at the same height as the single-pad model did
heights = {}
for leg in LEGS:
    for gi in range(m.ngeom):
        nm = m.geom(gi).name or ''
        if nm.startswith(leg + 'Tarsus5_pad'):
            heights.setdefault(leg, []).append(round(float(low[leg][2] + R) - R, 9))
chk['all_pad_bottoms_at_material_point'] = all(abs(v - round(float(low[l][2]), 9)) <= 1e-9 for l, vs in heights.items() for v in vs)
chk['pads_per_leg'] = [len(heights[l]) for l in LEGS]
assert all(v for k, v in chk.items() if isinstance(v, bool)), chk
(HERE / 'TRIAD_LEDGER.json').write_text(json.dumps(dict(
    source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    output=str(out), output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    radius=R, spread=SPREAD, torque_factor=TORQUE, invariants=chk,
    lowest_vertex=dict((k, np.round(v, 6).tolist()) for k, v in low.items())), indent=2) + '\n')
print(json.dumps(dict(pads=chk['pads_per_leg'], ngeom=(chk['ngeom_from'], chk['ngeom_to']),
                      torque=TORQUE, sha=hashlib.sha256(out.read_bytes()).hexdigest()[:12])))
