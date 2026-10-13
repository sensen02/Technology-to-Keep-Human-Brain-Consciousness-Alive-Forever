"""Read-only model audit: one compile, same-state forward only, no integration."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import hashlib
import json
from pathlib import Path
import signal
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
BASE = HERE/'outputs/model_branches/zero_floor'
OUT = HERE/'outputs/model_branches/timestep_constraint_audit.json'
DTS = (1e-4, 5e-5, 2.5e-5)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def effective_timeconst(tau, dt, refsafe=True):
    return max(tau, 2*dt) if refsafe and tau > 0 else tau

def static_xml(path):
    root = ET.parse(path).getroot()
    attrs = []
    for node in root.iter():
        for key, value in node.attrib.items():
            if 'solref' in key or 'solimp' in key:
                attrs.append(dict(tag=node.tag, name=node.get('name'), field=key, value=value))
    return dict(option=root.find('option').attrib, explicit_solver_attributes=attrs,
                includes=[n.attrib for n in root.iter('include')],
                defaults=[ET.tostring(n, encoding='unicode') for n in root.findall('default')],
                flags=[n.attrib for n in root.findall('option/flag')])

def main(report):
    import numpy as np
    import mujoco
    manifest_path = BASE/'model_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    canonical = Path(manifest['canonical_xml'])
    branch = BASE/'fruitfly_zero_floor.xml'
    paths = [canonical, branch, manifest_path, BASE/'zero_floor_traces.npz',
             HERE/'engine/canonical_physics.py', HERE/'run_zero_floor_branch.py', Path(__file__)]
    hashes = {str(p): sha(p) for p in paths}
    assert hashes[str(canonical)] == manifest['canonical_sha256']
    assert hashes[str(branch)] == manifest['branch_sha256']
    static = {str(p): static_xml(p) for p in (canonical, branch)}
    assert all(not s['includes'] and not s['explicit_solver_attributes'] for s in static.values())
    model = mujoco.MjModel.from_xml_path(str(branch))  # exactly one compile
    names = ('geom_solref', 'jnt_solref', 'dof_solref', 'tendon_solref_lim',
             'tendon_solref_fri', 'eq_solref', 'pair_solref', 'pair_solreffriction', 'flex_solref',
             'geom_solimp', 'jnt_solimp', 'dof_solimp', 'tendon_solimp_lim', 'tendon_solimp_fri',
             'eq_solimp', 'pair_solimp', 'geom_solmix', 'geom_priority', 'geom_condim',
             'geom_friction', 'geom_margin', 'geom_gap', 'jnt_limited', 'tendon_limited',
             'dof_frictionloss', 'tendon_frictionloss', 'actuator_dynprm', 'actuator_dyntype',
             'actuator_actlimited', 'actuator_ctrllimited', 'actuator_ctrlrange')
    fields = {}
    for name in names:
        a = getattr(model, name)
        fields[name] = dict(shape=list(a.shape), values=a.tolist(),
                           unique=np.unique(a, axis=0).tolist() if len(a) else [],
                           minimum_first=float(a[:, 0].min()) if a.ndim==2 and len(a) else None)
    opts = {}
    for name in ('timestep', 'integrator', 'solver', 'cone', 'jacobian', 'iterations', 'tolerance',
                 'ls_iterations', 'ls_tolerance', 'noslip_iterations', 'noslip_tolerance',
                 'impratio', 'disableflags', 'enableflags', 'o_solref', 'o_solimp', 'o_margin', 'o_friction'):
        value = getattr(model.opt, name)
        opts[name] = value.tolist() if isinstance(value, np.ndarray) else value
    refsafe = not bool(model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    with np.load(BASE/'zero_floor_traces.npz') as z:
        pose = z['seated_qpos'].copy()
    cases = []
    comparison_fields = ('efc_type','efc_id','efc_KBIP','efc_R','efc_D','efc_aref','efc_force','qacc')
    first = None
    for dt in DTS:
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        assert data.time == 0 and np.array_equal(data.qpos, pose) and not data.qvel.any() and not data.act.any()
        arrays = {k: getattr(data, k).copy() for k in comparison_fields}
        if first is None:
            first = arrays
        contacts = [dict(geom=c.geom.tolist(), solref=c.solref.tolist(), solreffriction=c.solreffriction.tolist(),
                         solimp=c.solimp.tolist(), dim=int(c.dim), efc_address=int(c.efc_address)) for c in data.contact]
        cases.append(dict(dt_s=dt, threshold_2dt_s=2*dt, time_s=float(data.time), ncon=data.ncon, nefc=data.nefc,
                          effective_positive_timeconst_s=effective_timeconst(.02, dt, refsafe), contacts=contacts,
                          constraints={k: v.tolist() for k,v in arrays.items()},
                          equal_to_100us={k:bool(np.array_equal(v, first[k])) for k,v in arrays.items()},
                          warning_counts=data.warning.number.tolist()))
    model.opt.timestep = opts['timestep']
    assert all(sha(p)==h for p,h in hashes.items())
    guard = report('completed')
    if not np.isfinite(guard['previous_soft_limit_mb']):
        guard['previous_soft_limit_mb'] = 'unlimited'
    result = dict(label='MANUAL READ_ONLY MODEL AUDIT; NO EPISODE; NO CONTROLLER; NO INTEGRATION',
                  mujoco_version=mujoco.__version__, compiled_models=1,
                  canonical_compilation='not repeated; static XML, loader source and frozen branch manifest inspected',
                  prior_manifest_changed_arrays=manifest['changed_arrays'],
                  counts={k:int(getattr(model,k)) for k in ('ngeom','njnt','nv','ntendon','neq','npair','nflex','nu','na')},
                  static=static, model_fields=fields, options=opts, refsafe_enabled=refsafe, cases=cases,
                  source_hashes=hashes, source_files_unchanged=True, memory_guard=guard)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), counts=result['counts'], refsafe=refsafe,
                         unique_solref={k:v['unique'] for k,v in fields.items() if 'solref' in k},
                         cases=[{k:c[k] for k in ('dt_s','ncon','nefc','equal_to_100us')} for c in cases],
                         memory_guard=guard), indent=2))

if __name__ == '__main__':
    from engine import memory_guard as mg
    def deadline(*_):
        raise TimeoutError('180-second audit deadline including <=60-second shared guard wait')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    with mg.guarded_run('timestep_constraint_audit', wait_s=60, **mg.BUDGETS['mechanics_audit']) as report:
        main(report)
    signal.alarm(0)
