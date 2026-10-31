"""Fail-closed independent endpoint reconstruction. Importing this file loads no physics.

Not historical state replay: unsaved warmstart is NOT recovered. Every endpoint
uses new data. No integration, control generation, plotting, retries or resume.
Routine project execution requires no per-call authorization or hard call quota.
Calls are still durably recorded; existing evidence is never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    """File fsync, atomic rename, directory fsync. Caller never retries failure."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with temp.open('xb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Refused(RuntimeError):
    pass


class Guard:
    def __init__(self, output, caps, identities, persist=atomic_json):
        self.output = Path(output)
        self.persist = persist
        # Exclusive directory is a permanent claim, including failures before ledger.
        # Never clear this claim, even when the first ledger write fails.
        if set(caps) != {'model_load', 'forward'} or any(type(v) is not int or v < 0 for v in caps.values()):
            raise Refused('invalid caps')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.output.mkdir(exist_ok=False)
        except FileExistsError as e:
            raise Refused('output/claim already exists; no retry/resume/reset') from e
        sync_dir(self.output.parent)
        self.ledger = dict(version=1, status='RUNNING', caps=dict(caps),
                           attempted=dict.fromkeys(caps, 0), completed=dict.fromkeys(caps, 0),
                           failed=dict.fromkeys(caps, 0), calls=[], identities=identities,
                           mode='independent endpoints, NOT historical replay',
                           historical_warmstart='UNSAVED_NOT_RESTORED',
                           prohibited=['mj_step', 'neural', 'retry', 'resume', 'plotting'])
        self.save()

    def save(self):
        self.persist(self.output / 'LEDGER.json', self.ledger)

    def call(self, kind, endpoint, fn):
        if self.ledger['status'] != 'RUNNING':
            raise Refused('run already stopped')
        # caps are historical/planned counts only, not execution permissions.
        # Actual calls remain recorded even when they exceed a planned count.
        event = dict(kind=kind, endpoint=endpoint, status='UNKNOWN',
                     note='reserved before call; interrupted outcome remains unknown')
        self.ledger['attempted'][kind] += 1
        self.ledger['calls'].append(event)
        self.save()  # If this fails, fn is NEVER called. Claim remains occupied.
        try:
            value = fn()
        except BaseException as e:
            event.update(status='FAILED', error=type(e).__name__ + ': ' + str(e))
            self.ledger['failed'][kind] += 1
            self.save()
            raise
        event.update(status='COMPLETED')
        self.ledger['completed'][kind] += 1
        self.save()  # Failure here leaves durable UNKNOWN, never triggers repeat.
        return value


def run(output, caps, identities, model_source, states, backend, persist=atomic_json):
    """Backend exposes ONLY load_model/new_data/restore/forward/export.

    All hashes/arrays must be validated before run; backend never generates states.
    attempted means charged reservation, not proof the backend was reached.
    """
    g = Guard(output, caps, identities, persist)
    try:
        model = g.call('model_load', None, lambda: backend.load_model(model_source))
        for i, state in enumerate(states):
            data = backend.new_data(model)  # Never reuse data between endpoints.
            backend.restore(model, data, state)
            g.call('forward', i, lambda: backend.forward(model, data))
            value = backend.export(model, data)
            record = dict(endpoint=i, input_state=state['identity'], identities=identities,
                          mode=g.ledger['mode'], warmstart=g.ledger['historical_warmstart'],
                          result=value)
            target = g.output / ('STATE_%06d.json' % i)
            persist(target, record)  # No call retries on serialization or I/O error.
            g.ledger.setdefault('shards', []).append(dict(path=str(target), sha256=sha(target)))
            g.save()
        g.ledger['status'] = 'COMPLETE'
        g.save()
    except BaseException as e:
        g.ledger.update(status='STOPPED', error=type(e).__name__ + ': ' + str(e))
        # Best-effort terminal record only, NEVER backend retry or budget refund.
        try:
            g.save()
        except BaseException:
            pass  # durable RUNNING/UNKNOWN or occupied directory rejects restart
        raise
    return g.ledger


class RealBackend:
    """Lazy physics import only inside a pre-reserved model-load call."""
    def load_model(self, source):
        import mujoco
        self.mj = mujoco
        if mujoco.__version__ != '3.9.0':
            raise Refused('requires archived 3.9.0')
        m = mujoco.MjModel.from_xml_path(str(source))
        # The archived fullbody model is used with its sole certified RK4 override.
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        if (m.nq, m.nv, m.nu) != (49, 48, 90) or m.nmocap or m.nplugin:
            raise Refused('unsupported model state requirements')
        if any(getattr(mujoco, 'get_mjcb_' + n)() is not None
               for n in ('passive', 'control', 'act_dyn', 'act_gain', 'act_bias')):
            raise Refused('callbacks forbidden')
        return m

    def new_data(self, model):
        return self.mj.MjData(model)

    def restore(self, model, data, state):
        for field in ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
            getattr(data, field)[:] = state[field]
        data.time = state['time']
        # MjData is fresh: no previous endpoint solver or constraint state.
        # Explicit zero is initialization, NOT a claim to restore history.
        data.qacc_warmstart[:] = 0

    def forward(self, model, data):
        self.mj.mj_forward(model, data)

    def export(self, model, data):
        import numpy as np
        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            f = np.zeros(6)
            self.mj.mj_contactForce(model, data, i, f)
            g1, g2 = int(c.geom1), int(c.geom2)
            if min(g1, g2) < 0:
                raise Refused('flex contact requires another reviewed adapter')
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            jp1, jr1 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            jp2, jr2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            self.mj.mj_jac(model, data, jp1, jr1, c.pos, b1)
            self.mj.mj_jac(model, data, jp2, jr2, c.pos, b2)
            frame = np.asarray(c.frame).reshape(3, 3)
            rel = (jp2 - jp1) @ data.qvel
            rel_contact = frame @ rel
            world_force, world_torque = frame.T @ f[:3], frame.T @ f[3:]
            objects = {}
            for side, geom, body in ((1, g1, b1), (2, g2, b2)):
                objects['side%d' % side] = dict(geom=geom, body=body,
                    geom_name=self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_GEOM, geom),
                    body_name=self.mj.mj_id2name(model, self.mj.mjtObj.mjOBJ_BODY, body))
            contacts.append(dict(objects=objects, body1=b1, body2=b2,
                                 relative_velocity_world_2_minus_1=rel.tolist(),
                                 relative_velocity_contact=rel_contact.tolist(),
                                 normal_separation_velocity=float(rel_contact[0]),
                                 tangent_velocity=rel_contact[1:3].tolist(),
                                 tangent_speed=float(np.linalg.norm(rel_contact[1:3])),
                                 relative_angular_velocity_contact=(frame @ ((jr2-jr1) @ data.qvel)).tolist(),
                                 world_force_on_geom2=world_force.tolist(),
                                 world_force_on_geom1=(-world_force).tolist(),
                                 world_torque_on_geom2=world_torque.tolist(),
                                 world_torque_on_geom1=(-world_torque).tolist(),
                                 geom1=g1, geom2=g2,
                                 dist=float(c.dist), frame=c.frame.tolist(), pos=c.pos.tolist(),
                                 friction=c.friction.tolist(), dim=int(c.dim),
                                 exclude=int(c.exclude), efc_address=int(c.efc_address),
                                 includemargin=float(c.includemargin), force_torque=f.tolist()))
        result = {k: getattr(data, k).tolist() for k in
                  ('qpos', 'qvel', 'act', 'ctrl', 'qacc', 'qfrc_passive', 'qfrc_actuator',
                   'qfrc_bias', 'qfrc_constraint', 'qfrc_applied', 'xfrc_applied',
                   'actuator_force', 'actuator_length', 'actuator_velocity')}
        # MuJoCo 3.9: M*a = passive + actuator + constraint + applied
        #                  + mapped body wrench - bias.
        # qfrc_applied does NOT contain xfrc_applied. Map body-COM world
        # wrenches exactly once; do not also add qfrc_smooth to this RHS.
        ma = np.zeros(model.nv)
        self.mj.mj_mulM(model, data, ma, data.qacc)
        external = np.zeros(model.nv)
        for body in range(1, model.nbody):
            wrench = data.xfrc_applied[body]
            self.mj.mj_applyFT(model, data, wrench[:3], wrench[3:],
                               data.xipos[body], body, external)
        if np.any(data.xfrc_applied[0]):
            raise Refused('world-body external wrench unsupported for certification')
        rhs_terms = dict(passive=data.qfrc_passive, actuator=data.qfrc_actuator,
                         constraint=data.qfrc_constraint, applied=data.qfrc_applied,
                         body_external=external, negative_bias=-data.qfrc_bias)
        rhs = sum(rhs_terms.values(), np.zeros(model.nv))
        residual = ma - rhs
        # Per-DOF denominator avoids summing translational forces and torques.
        scale = np.maximum(1e-12, np.maximum(np.abs(ma),
                          sum((np.abs(v) for v in rhs_terms.values()), np.zeros(model.nv))))
        if not all(np.isfinite(v).all() for v in [ma, rhs, residual, scale]):
            raise Refused('nonfinite dynamics evidence')
        result['dynamics'] = dict(mass_times_qacc=ma.tolist(), rhs=rhs.tolist(),
            mapped_xfrc_applied=external.tolist(), rhs_terms={k:v.tolist() for k,v in rhs_terms.items()},
            residual_per_dof=residual.tolist(), normalized_residual_per_dof=(residual/scale).tolist(),
            normalization_per_dof=scale.tolist(),
            normalization='max(1e-12 native generalized-force unit, abs(Ma_i), sum abs(rhs term_i)); no cross-DOF sum',
            equation='M*qacc=passive+actuator+constraint+qfrc_applied+mapped_xfrc_applied-qfrc_bias',
            certification='evidence only; no acceptance tolerance inferred')
        result.update(time=float(data.time), contacts=contacts, cone=int(model.opt.cone),
                      warnings=[int(w.number) for w in data.warning])
        if any(result['warnings']):
            raise Refused('warning: stop without retry')
        # Strict JSON serialization below rejects NaN/Inf.
        return result


# Closed allowlist from the reviewed preflight, NOT caller-supplied trust anchors.
REVIEWED_ZERO_LOAD_RUNS = {
    's100': ('s100-tick-consumption/20261023T222456Z/run', 'tick400ms',
             'run_native_rk4_tick_consumption.py',
             'a49965c2731efb00114c6942c445c18dfe7f8a21433fbe3cbc626851d76ddf3e',
             'f1ee58c5dac1595097b1da66d7e83cf798f20fb5518ca0a91eab52665363748f',
             '9f4b69fe5bef0b963bdddd269f10582354100ed7f2e195db63b7e96a8b737d0e',
             'feffe3f34359f575e0454f88f715fd4df83b262feb882ce71a7ad09960d1f961'),
    's102_minus': ('s102-tick-training-second/20261023T224501Z/run', 'round00_minus',
             'run_native_rk4_tick_training_second.py',
             '4275f93464bc7d641c60226766b4195e133c631ebd3deb335bf80811ab21136d',
             '851128b111919ed6f02d442ca05f2418ea22b39bf8642b2fe3e2d65c2c108138',
             '260fcd05745c28a947729a92842de13222ca0c5f8b633a416e1fc12118db6f55',
             '4c71728e81574e5f464dc60089e784a46b83a47c1e53e6e838df88dc7cba30bc')
}


def validate_zero_load_proof(cfg):
    proof_path = cfg['external_load_proof']
    if proof_path not in cfg['inputs']:
        raise Refused('unauthenticated proof')
    proof = json.loads(Path(proof_path).read_text())
    if proof.get('run') not in REVIEWED_ZERO_LOAD_RUNS:
        raise Refused('unreviewed zero-load run')
    run, arm, source, *digests = REVIEWED_ZERO_LOAD_RUNS[proof['run']]
    base = Path('/home/sensen/Desktop/engine修复验收') / run
    source = Path('/run/media/sensen/Data2/cell_wound_prototype') / source
    paths = [base/arm/'TRAJECTORY.npz', source, base/'INPUT_HASHES.json', base/'MODEL_END_CHECK.json']
    expected = {str(p): h for p, h in zip(paths, digests)}
    if proof.get('files') != expected or cfg['trajectory'] != str(paths[0]):
        raise Refused('proof/trajectory identity mismatch')
    if proof.get('not_saved_per_endpoint') != ['qfrc_applied', 'xfrc_applied'] or proof.get('basis') != 'successful-path initial and every-step pre/post zero assertions':
        raise Refused('missing provenance disclosure')
    for path, digest in expected.items():
        if cfg['inputs'].get(path) != digest or sha(path) != digest:
            raise Refused('reviewed evidence hash mismatch: ' + path)
    historical = json.loads(paths[2].read_text())
    end = json.loads(paths[3].read_text())
    if historical.get(str(source)) != digests[1] or not all(end.get(k) is True for k in ('passed', 'matches_after', 'callbacks_none', 'source_hashes_unchanged')):
        raise Refused('historical source or successful termination not established')
    if cfg['model'] not in historical or historical[cfg['model']] != sha(cfg['model']):
        raise Refused('unreviewed model')
    return dict(mode='source_invariant', description='源码不变量推导；非历史逐端点保存值',
                proof=proof_path, proof_sha256=sha(proof_path), run=proof['run'],
                historical_mesh_bytes='NOT_CERTIFIED; dependency hashes bind future reconstruction only')


def prepare(manifest):
    """Read and authenticate files without importing physics or original scripts.

    External loads MUST be explicit saved arrays. A zero-load assumption is not
    silently substituted. Equality/mocap/plugin models are out of this scope.
    """
    import numpy as np
    p = Path(manifest)
    cfg = json.loads(p.read_text())
    inputs = cfg['inputs']  # MUST include XML and every static dependency.
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise Refused('input hash mismatch: ' + path)
    mode = cfg.get('external_load_mode', 'saved_arrays')
    if mode not in ('saved_arrays', 'source_invariant'):
        raise Refused('unknown external load source; no fallback')
    for key in ('model', 'trajectory') + (('external_loads',) if mode == 'saved_arrays' else ('external_load_proof',)):
        if cfg[key] not in inputs:
            raise Refused('missing authenticated input ' + key)
    origin = validate_zero_load_proof(cfg) if mode == 'source_invariant' else dict(mode='saved_arrays')
    import xml.etree.ElementTree as ET
    xml = ET.parse(cfg['model']).getroot()
    # Reject includes here rather than guess dependency resolution. Mesh files
    # referenced by full path must be included in manifest hashes.
    if xml.findall('.//include') or xml.find('equality') is not None:
        raise Refused('includes/equality require a separately reviewed adapter')
    for asset in xml.findall('.//asset/*'):
        if 'file' in asset.attrib:
            q = Path(asset.attrib['file'])
            if not q.is_absolute() or str(q) not in inputs:
                raise Refused('unbound asset dependency')
    states = []
    from contextlib import ExitStack
    with ExitStack() as stack:
        z = stack.enter_context(np.load(cfg['trajectory'], allow_pickle=False))
        ext = stack.enter_context(np.load(cfg['external_loads'], allow_pickle=False)) if mode == 'saved_arrays' else None
        for i in cfg['indices']:
            if type(i) is not int or i < 0:
                raise Refused('invalid index')
            s = {k: z[k][i].copy() for k in ('qpos', 'qvel', 'act', 'ctrl')}
            s.update(time=float(z['time'][i]),
                     qfrc_applied=ext['qfrc_applied'][i].copy() if ext is not None else np.zeros(48),
                     xfrc_applied=ext['xfrc_applied'][i].copy() if ext is not None else np.zeros((73, 6)),
                     identity=dict(trajectory=cfg['trajectory'], index=i, external_load_origin=origin))
            for k, n in [('qpos',49),('qvel',48),('act',90),('ctrl',90),('qfrc_applied',48)]:
                if s[k].shape != (n,) or not np.isfinite(s[k]).all():
                    raise Refused('invalid state field ' + k)
            if s['xfrc_applied'].ndim != 2 or s['xfrc_applied'].shape[1] != 6 or not np.isfinite(s['xfrc_applied']).all() or not np.isfinite(s['time']):
                raise Refused('invalid external loads/time')
            states.append(s)
    identities = dict(inputs=inputs, manifest=str(p.resolve()), manifest_sha256=sha(p),
                      script=str(Path(__file__).resolve()), script_sha256=sha(__file__))
    return cfg, identities, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-real', action='store_true', help='Deprecated compatibility flag; real execution no longer requires this flag')
    args = parser.parse_args()
    # Occupied output refuses even before input preparation or backend creation.
    if Path(args.output).exists():
        raise Refused('existing output: no reset, retry or resume')
    cfg, identities, states = prepare(args.manifest)
    run(args.output, cfg['caps'], identities, cfg['model'], states, RealBackend())


if __name__ == '__main__':
    main()
