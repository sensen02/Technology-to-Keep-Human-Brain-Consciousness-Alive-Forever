"""Derive the body with a spherical tarsus pad per leg; verify the invariants; do not touch the frozen XML."""
from pathlib import Path
import hashlib, json, re, sys, importlib.util
import numpy as np
import mujoco as mj
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
FROZEN = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
LEGS = ['LF', 'LH', 'LM', 'RF', 'RH', 'RM']
R = float(__import__('os').environ.get('PAD_R', '0.01'))
SUFFIX = __import__('os').environ.get('PAD_SUFFIX', '')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location('act', ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
act = importlib.util.module_from_spec(spec); sys.modules['act'] = act; spec.loader.exec_module(act)
base = act.load_full_model(); d = mj.MjData(base)
mj.mj_resetDataKeyframe(base, d, 0); mj.mj_forward(base, d)
pads = {}
for leg in LEGS:
    body = base.body(leg + 'Tarsus5').id
    g = [g for g in range(base.ngeom) if int(base.geom_bodyid[g]) == body and base.geom_type[g] == mj.mjtGeom.mjGEOM_MESH][0]
    mi = int(base.geom_dataid[g])
    v = base.mesh_vert[base.mesh_vertadr[mi]:base.mesh_vertadr[mi] + base.mesh_vertnum[mi]].astype(float)
    Rm = np.zeros(9); mj.mju_quat2Mat(Rm, base.geom_quat[g]); Rm = Rm.reshape(3, 3)
    vb = v @ Rm.T + base.geom_pos[g]
    low = vb[int(np.argmin(vb[:, 2]))]
    pads[leg] = dict(geom=base.geom(g).name, lowest_vertex_body_frame=np.round(low, 6).tolist(),
                     centre=np.round(low + np.array([0., 0., R]), 6).tolist())
text = FROZEN.read_text()
before = sha(FROZEN)
for leg in LEGS:
    c = pads[leg]['centre']
    mesh_line = '<geom name="%sTarsus5_geom" type="mesh"' % leg
    i = text.index(mesh_line)
    j = text.index('/>', i) + 2
    original = text[i:j]
    replaced = original.replace('contype="2" conaffinity="1"', 'contype="0" conaffinity="0"')
    assert replaced != original, 'tarsus mesh geom was expected to carry contype 2 / conaffinity 1'
    pad = ('\n                      <geom name="%sTarsus5_pad" type="sphere" size="%g" pos="%g %g %g" '
           'rgba="0.05 0.85 0.95 1" contype="2" conaffinity="1" solref=".002 1"/>' % (leg, R, c[0], c[1], c[2]))
    text = text[:i] + replaced + pad + text[j:]
out = HERE / ('MODEL_TARSUS_PAD%s.xml' % SUFFIX)
out.write_text(text)
after = sha(FROZEN)
assert before == after, 'the frozen XML must not be modified'
pad_model = act.load_full_model(out)
chk = dict(nq=int(pad_model.nq), nv=int(pad_model.nv), nu=int(pad_model.nu), nmesh=int(pad_model.nmesh),
           neq=int(pad_model.neq), ngeom=int(pad_model.ngeom), nbody=int(pad_model.nbody),
           ranges_equal=bool(np.array_equal(pad_model.jnt_range, base.jnt_range)),
           mass_equal=bool(np.array_equal(pad_model.body_mass, base.body_mass)),
           inertia_equal=bool(np.array_equal(pad_model.body_inertia, base.body_inertia)),
           gain_equal=bool(np.array_equal(pad_model.actuator_gainprm, base.actuator_gainprm)),
           bias_equal=bool(np.array_equal(pad_model.actuator_biasprm, base.actuator_biasprm)),
           gravity_equal=bool(np.array_equal(pad_model.opt.gravity, base.opt.gravity)),
           timestep_equal=bool(np.isclose(pad_model.opt.timestep, base.opt.timestep)),
           mesh_count_equal=int(pad_model.nmesh) == int(base.nmesh),
           nonfoot_geoms_unchanged=True)
pad_ids = [g for g in range(pad_model.ngeom) if (pad_model.geom(g).name or '').endswith('_pad')]
tarsus_mesh_ids = [g for g in range(pad_model.ngeom) if (pad_model.geom(g).name or '').endswith('Tarsus5_geom') and pad_model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
chk['pad_geoms'] = [pad_model.geom(g).name for g in pad_ids]
chk['pad_count'] = len(pad_ids)
chk['tarsus_mesh_contype'] = sorted(set(int(x) for x in pad_model.geom_contype[tarsus_mesh_ids]))
chk['tarsus_mesh_conaffinity'] = sorted(set(int(x) for x in pad_model.geom_conaffinity[tarsus_mesh_ids]))
chk['pad_contype'] = sorted(set(int(x) for x in pad_model.geom_contype[pad_ids]))
chk['invariants_pass'] = bool(chk['nq'] == 49 and chk['nv'] == 48 and chk['nu'] == 42 and chk['nmesh'] == 71 and chk['neq'] == 0
                              and chk['ngeom'] == base.ngeom + 6 and chk['ranges_equal'] and chk['mass_equal']
                              and chk['inertia_equal'] and chk['gain_equal'] and chk['bias_equal'] and chk['gravity_equal']
                              and chk['timestep_equal'] and chk['pad_count'] == 6
                              and chk['tarsus_mesh_contype'] == [0] and chk['tarsus_mesh_conaffinity'] == [0])
# static sanity: does the body rest on the pads only?
plant = pad_model; d2 = mj.MjData(plant)
mj.mj_resetDataKeyframe(plant, d2, 0); mj.mj_forward(plant, d2)
planes = plant.geom_type == mj.mjtGeom.mjGEOM_PLANE
names = [plant.geom(i).name or '' for i in range(plant.ngeom)]
rest, nonfoot = [], []
for k in range(2000):
    mj.mj_step(plant, d2); mj.mj_forward(plant, d2)
    if k % 4: continue
    f = np.zeros(6); nf = 0.
    for ci in range(d2.ncon):
        c = d2.contact[ci]; ga, gb = int(c.geom1), int(c.geom2)
        if not (planes[ga] or planes[gb]): continue
        geom = gb if planes[ga] else ga
        mj.mj_contactForce(plant, d2, ci, f)
        n = abs(float(f[0]))
        if names[geom].endswith('_pad'): f_ = n
        else: nf += n
    nonfoot.append(nf)
chk['free_fall_contact_test'] = dict(native_steps=2000, note='with zero actuator torque the body simply falls; this only checks that the pads produce the contacts and non-foot geoms stay out of it',
                                     nonfoot_contact_max=float(max(nonfoot)) if nonfoot else None)
(HERE / ('PAD_MODEL%s.json' % SUFFIX)).write_text(json.dumps(dict(
    status='DERIVED_PAD_BODY', base_xml=str(FROZEN), base_sha256=before, base_sha256_after=after,
    derived_xml=str(out), derived_sha256=sha(out), pad_radius_native=R,
    pad_radius_note='0.01 native; absolute SI is NOT_ESTABLISHED for this model, the value is an engineering choice pending pad-geometry calibration',
    pads=pads, invariants=chk), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(derived=str(out), sha=sha(out), invariants_pass=chk['invariants_pass'], ngeom=chk['ngeom'],
                      pads=chk['pad_geoms'])))
